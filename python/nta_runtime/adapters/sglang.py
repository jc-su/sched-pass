"""SGLang boundary adapter.

Only SGLang metadata extraction belongs here.  Claim lifetime, demand
semantics, work-unit state, and transport policy stay outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import EngineBatch, RequestIdentityAdapter
from ..execution_protocol import ExecutionProtocolConfig, ProtocolKind
from ..work_unit import Granularity


@dataclass(frozen=True)
class SglangExecutionConfig:
    """Validated SGLang projection of the engine-neutral protocol config."""

    protocol: ExecutionProtocolConfig

    @classmethod
    def from_environment(cls, environ: dict[str, str] | None = None) -> "SglangExecutionConfig":
        import os

        values = os.environ if environ is None else environ
        raw_protocol = values.get("NTA_SGLANG_EXECUTION_PROTOCOL", "late_bound").strip().lower()
        protocol_names = {
            "nta": ProtocolKind.LATE_BOUND,
            "late_bound": ProtocolKind.LATE_BOUND,
            "conventional": ProtocolKind.CONVENTIONAL,
            "partial": ProtocolKind.PARTIAL,
        }
        try:
            kind = protocol_names[raw_protocol]
        except KeyError as error:
            raise ValueError(
                "NTA_SGLANG_EXECUTION_PROTOCOL must be nta, late_bound, "
                "conventional, or partial"
            ) from error
        raw_granularity = values.get(
            "NTA_SGLANG_WORK_GRANULARITY", Granularity.PAGE_GROUP.value
        ).strip().lower()
        try:
            granularity = Granularity(raw_granularity)
        except ValueError as error:
            raise ValueError(
                "NTA_SGLANG_WORK_GRANULARITY must be request, layer, "
                "page_group, or cta_tile"
            ) from error
        try:
            max_inflight = int(values.get("NTA_SGLANG_MAX_INFLIGHT_UNITS", "4096"))
        except ValueError as error:
            raise ValueError("NTA_SGLANG_MAX_INFLIGHT_UNITS must be an integer") from error
        if kind is ProtocolKind.CONVENTIONAL:
            protocol = ExecutionProtocolConfig.conventional(
                granularity=granularity,
                max_inflight_units=max_inflight,
            )
        elif kind is ProtocolKind.PARTIAL:
            protocol = ExecutionProtocolConfig.partial(
                granularity=granularity,
                max_inflight_units=max_inflight,
            )
        else:
            protocol = ExecutionProtocolConfig.late_bound(
                granularity=granularity,
                max_inflight_units=max_inflight,
            )
        return cls(protocol)


class SglangAdapter(RequestIdentityAdapter):
    def __init__(self, runtime: Any, request_capacity: int) -> None:
        super().__init__(runtime, request_capacity, engine="sglang")

    def bind_forward(
        self,
        forward_batch: Any,
        *,
        allow_capture_ids: bool,
        stream: Any,
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
            request_ids = [f"__nta_graph_capture_{index}" for index in range(batch_size)]
        request_ids = tuple(str(request_id) for request_id in request_ids)
        if len(request_ids) != batch_size:
            raise RuntimeError("SGLang request IDs do not match the graph batch")
        priorities = tuple(
            int(priority)
            for priority in getattr(
                forward_batch, "_nta_request_priorities", (0,) * batch_size
            )
        )
        bindings = self.bind(
            request_ids,
            tuple(range(batch_size)),
            priorities=priorities,
            stream=stream,
        )
        return EngineBatch(self.engine, epoch, bindings, granularity)
