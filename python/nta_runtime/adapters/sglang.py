"""SGLang boundary adapter.

Only SGLang metadata extraction belongs here.  External-prefix lifetime,
demand semantics, work-unit state, and transport stay outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import (
    EngineBatch,
    RequestIdentityAdapter,
    _integer_vector,
    _request_id_vector,
)
from ..abi import MAX_REQUEST_PRIORITY
from ..execution_protocol import ExecutionProtocolConfig
from ..work_unit import Granularity


@dataclass(frozen=True)
class SglangExecutionConfig:
    """Validated SGLang projection of the engine-neutral protocol config."""

    protocol: ExecutionProtocolConfig
    prefetch: bool = True

    @property
    def grouping(self) -> str:
        """Map semantic granularity to the transport's physical grouping."""
        return (
            "request"
            if self.protocol.granularity in (Granularity.REQUEST, Granularity.LAYER)
            else "tile"
        )

    @classmethod
    def from_environment(
        cls, environ: dict[str, str] | None = None
    ) -> "SglangExecutionConfig":
        import os

        values = os.environ if environ is None else environ
        return cls(
            ExecutionProtocolConfig.from_environment(environ),
            values.get("NTA_EXECUTION_PREFETCH", "1") != "0",
        )


@dataclass(frozen=True, slots=True)
class SglangForwardMetadata:
    """NTA-owned metadata carried beside one SGLang ``ForwardBatch``.

    SGLang does not provide an extension field for request identity and
    tenant annotations on graph-derived forward views.  Keeping the values in
    one immutable sidecar makes the framework boundary explicit and prevents
    independently-sized dynamic attributes from being copied into a replay
    batch.  The sidecar is control-plane metadata; it never owns KV payloads
    or transport state.
    """

    request_slots: tuple[int, ...]
    priorities: tuple[int, ...]
    tenant_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not all(
            isinstance(values, tuple)
            for values in (self.request_slots, self.priorities, self.tenant_ids)
        ):
            raise TypeError("SGLang forward metadata vectors must be tuples")
        lengths = {
            len(self.request_slots),
            len(self.priorities),
            len(self.tenant_ids),
        }
        if len(lengths) != 1:
            raise ValueError("SGLang forward metadata vectors must be aligned")
        object.__setattr__(
            self,
            "request_slots",
            _integer_vector(
                self.request_slots, "SGLang request slots", maximum=(1 << 32) - 1
            ),
        )
        object.__setattr__(
            self,
            "priorities",
            _integer_vector(
                self.priorities,
                "SGLang priorities",
                maximum=MAX_REQUEST_PRIORITY,
            ),
        )
        object.__setattr__(
            self,
            "tenant_ids",
            _integer_vector(
                self.tenant_ids, "SGLang tenant IDs", maximum=(1 << 32) - 1
            ),
        )

    @classmethod
    def from_values(
        cls,
        request_slots: Any,
        *,
        priorities: Any = None,
        tenant_ids: Any = None,
        batch_size: int | None = None,
    ) -> "SglangForwardMetadata":
        def normalize(values: Any, name: str) -> tuple[int, ...]:
            if values is None:
                return ()
            try:
                maximum = MAX_REQUEST_PRIORITY if name == "priority" else (1 << 32) - 1
                return _integer_vector(
                    values, f"SGLang {name} metadata", maximum=maximum
                )
            except ValueError as error:
                raise RuntimeError(str(error)) from error

        slots = normalize(request_slots, "request-slot")
        if batch_size is not None and len(slots) != batch_size:
            raise RuntimeError("SGLang request slots do not match the forward batch")
        size = len(slots) if batch_size is None else batch_size
        raw_priorities = normalize(priorities, "priority")
        raw_tenants = normalize(tenant_ids, "tenant")
        if not raw_priorities:
            raw_priorities = (0,) * size
        if not raw_tenants:
            raw_tenants = (0,) * size
        if len(raw_priorities) != size or len(raw_tenants) != size:
            raise RuntimeError("SGLang forward metadata vectors do not match the batch")
        return cls(slots, raw_priorities, raw_tenants)

    def pad(self, padded_request_slots: Any) -> "SglangForwardMetadata":
        """Extend metadata using the slots from SGLang's padded view.

        Graph replay may add masked rows, but those rows still need the same
        slot vector that the framework will hand to attention.  Fabricating
        slots here would make request identity disagree with the actual KV
        page table, so the caller must supply the padded view explicitly.
        """
        if hasattr(padded_request_slots, "tolist"):
            padded_request_slots = padded_request_slots.tolist()
        try:
            padded_slots = _integer_vector(
                padded_request_slots,
                "padded SGLang request slots",
                maximum=(1 << 32) - 1,
            )
        except ValueError as error:
            raise ValueError(str(error)) from error
        raw_size = len(self.request_slots)
        padded_size = len(padded_slots)
        if padded_size < raw_size:
            raise ValueError("cannot shrink SGLang forward metadata with pad()")
        if padded_slots[:raw_size] != self.request_slots:
            raise ValueError("padded SGLang request slots changed live request order")
        padding = padded_size - raw_size
        return SglangForwardMetadata(
            padded_slots,
            self.priorities + (0,) * padding,
            self.tenant_ids + (0,) * padding,
        )


FORWARD_METADATA_ATTRIBUTE = "_nta_forward_metadata"


def forward_metadata(
    forward_batch: Any, *, allow_default_slots: bool = False
) -> SglangForwardMetadata:
    """Read the validated NTA sidecar from one SGLang forward view."""
    batch_size = int(getattr(forward_batch, "batch_size", 0) or 0)
    metadata = getattr(forward_batch, FORWARD_METADATA_ATTRIBUTE, None)
    if metadata is not None:
        if not isinstance(metadata, SglangForwardMetadata):
            raise RuntimeError("SGLang forward metadata has an invalid sidecar type")
        if len(metadata.request_slots) != batch_size:
            raise RuntimeError("SGLang forward metadata does not match the batch")
        return metadata
    slots = getattr(forward_batch, "req_pool_indices", None)
    if slots is None:
        if not allow_default_slots:
            raise RuntimeError("SGLang forward metadata omitted request-pool slots")
        slots = tuple(range(batch_size))
    return SglangForwardMetadata.from_values(slots, batch_size=batch_size)


class SglangAdapter(RequestIdentityAdapter):
    def __init__(self, runtime: Any, request_capacity: int) -> None:
        super().__init__(runtime, request_capacity, engine="sglang")

    def bind_forward(
        self,
        forward_batch: Any,
        *,
        allow_capture_ids: bool = False,
        stream: Any = None,
        epoch: int,
        granularity: Granularity = Granularity.PAGE_GROUP,
    ) -> EngineBatch:
        request_ids = getattr(forward_batch, "rids", None)
        batch_size = int(forward_batch.batch_size)
        if request_ids is None:
            if not allow_capture_ids:
                raise RuntimeError(
                    "SGLang CUDA replay omitted request IDs from its metadata view"
                )
            request_ids = [
                f"__nta_graph_capture_{index}" for index in range(batch_size)
            ]
        try:
            request_ids = _request_id_vector(request_ids, "SGLang request IDs")
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        if len(request_ids) != batch_size:
            raise RuntimeError("SGLang request IDs do not match the graph batch")
        try:
            metadata = forward_metadata(
                forward_batch, allow_default_slots=allow_capture_ids
            )
        except RuntimeError:
            raise
        request_slots = metadata.request_slots
        priorities = metadata.priorities
        tenant_ids = metadata.tenant_ids
        bindings = self.bind(
            request_ids,
            request_slots,
            priorities=priorities,
            tenant_ids=tenant_ids,
            stream=stream,
        )
        return EngineBatch(self.engine, epoch, bindings, granularity)
