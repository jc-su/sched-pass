"""SGLang boundary adapter.

Only SGLang metadata extraction belongs here.  External-prefix lifetime,
demand semantics, work-unit state, and transport stay outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import EngineBatch, RequestIdentityAdapter
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
        request_ids = tuple(str(request_id) for request_id in request_ids)
        if len(request_ids) != batch_size:
            raise RuntimeError("SGLang request IDs do not match the graph batch")
        request_slots = getattr(forward_batch, "_nta_request_slots", None)
        if request_slots is None:
            request_slots = getattr(forward_batch, "req_pool_indices", None)
        if request_slots is not None and hasattr(request_slots, "tolist"):
            request_slots = request_slots.tolist()
        if request_slots is None:
            if not allow_capture_ids:
                raise RuntimeError("SGLang forward metadata omitted request-pool slots")
            request_slots = tuple(range(batch_size))
        request_slots = tuple(int(slot) for slot in request_slots)
        if len(request_slots) != batch_size:
            raise RuntimeError("SGLang request slots do not match the graph batch")
        priorities = tuple(
            int(priority)
            for priority in getattr(
                forward_batch, "_nta_request_priorities", (0,) * batch_size
            )
        )
        raw_tenant_ids = getattr(forward_batch, "_nta_request_tenant_ids", None)
        if raw_tenant_ids is None:
            raw_tenant_ids = (0,) * batch_size
        elif hasattr(raw_tenant_ids, "tolist"):
            raw_tenant_ids = raw_tenant_ids.tolist()
        tenant_ids = tuple(int(tenant_id) for tenant_id in raw_tenant_ids)
        if len(tenant_ids) != batch_size:
            raise RuntimeError("SGLang request tenants do not match the batch")
        bindings = self.bind(
            request_ids,
            request_slots,
            priorities=priorities,
            tenant_ids=tenant_ids,
            stream=stream,
        )
        return EngineBatch(self.engine, epoch, bindings, granularity)
