"""vLLM boundary adapter.

vLLM owns request IDs, block tables, cancellation, and scheduler slots.  The
adapter accepts those already-normalized values and deliberately does not
import vLLM, so contract tests run without a vLLM installation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import ExactDemandProjection, EngineBatch, RequestIdentityAdapter
from ..work_unit import Granularity


@dataclass(frozen=True)
class VllmSchedulerProjection:
    """The only vLLM-to-NTA projection owned by the integration layer."""

    request_ids: tuple[str, ...]
    request_slots: tuple[int, ...]
    priorities: tuple[int, ...] | None = None
    deadline_clocks: tuple[int, ...] | None = None
    tenant_ids: tuple[int, ...] | None = None
    block_tables: tuple[tuple[int, ...], ...] | None = None
    page_bytes: int | None = None

    @classmethod
    def from_scheduler_output(cls, output: Any) -> "VllmSchedulerProjection":
        request_ids = getattr(output, "request_ids", None)
        request_slots = getattr(output, "request_slots", None)
        if request_ids is None or request_slots is None:
            raise ValueError(
                "vLLM scheduler output must expose request_ids and request_slots"
            )
        priorities = getattr(output, "priorities", None)
        deadlines = getattr(output, "deadline_clocks", None)
        tenant_ids = getattr(output, "tenant_ids", None)
        block_tables = getattr(output, "block_tables", None)
        if block_tables is not None and hasattr(block_tables, "tolist"):
            block_tables = block_tables.tolist()
        normalized_tables = (
            None
            if block_tables is None
            else tuple(tuple(int(page) for page in row) for row in block_tables)
        )
        raw_page_bytes = getattr(output, "kv_page_bytes", None)
        if raw_page_bytes is None:
            raw_page_bytes = getattr(output, "page_bytes", None)
        return cls(
            tuple(str(request_id) for request_id in request_ids),
            tuple(int(request_slot) for request_slot in request_slots),
            None if priorities is None else tuple(int(value) for value in priorities),
            None if deadlines is None else tuple(int(value) for value in deadlines),
            None if tenant_ids is None else tuple(int(value) for value in tenant_ids),
            normalized_tables,
            None if raw_page_bytes is None else int(raw_page_bytes),
        )

    def exact_demand(self) -> ExactDemandProjection:
        """Return the exact block-table demand or reject an incomplete hook."""
        if self.block_tables is None or self.page_bytes is None:
            raise RuntimeError(
                "vLLM NTA integration requires exact block_tables and kv_page_bytes"
            )
        if len(self.block_tables) != len(self.request_ids):
            raise RuntimeError("vLLM block tables do not match request IDs")
        return ExactDemandProjection(self.block_tables, self.page_bytes)


class VllmAdapter(RequestIdentityAdapter):
    def __init__(self, runtime: Any, request_capacity: int) -> None:
        super().__init__(runtime, request_capacity, engine="vllm")

    def bind_batch(
        self,
        request_ids: tuple[str, ...],
        request_slots: tuple[int, ...],
        *,
        epoch: int,
        stream: Any = None,
        priorities: tuple[int, ...] | None = None,
        deadline_clocks: tuple[int, ...] | None = None,
        tenant_ids: tuple[int, ...] | None = None,
        exact_demand: ExactDemandProjection | None = None,
        granularity: Granularity = Granularity.PAGE_GROUP,
    ) -> EngineBatch:
        bindings = self.bind(
            request_ids,
            request_slots,
            priorities=priorities,
            deadline_clocks=deadline_clocks,
            tenant_ids=tenant_ids,
            stream=stream,
        )
        return EngineBatch(self.engine, epoch, bindings, granularity, exact_demand)

    def bind_forward(
        self,
        scheduler_output: Any,
        *,
        epoch: int,
        stream: Any = None,
        granularity: Granularity = Granularity.PAGE_GROUP,
    ) -> EngineBatch:
        """Bind one vLLM scheduler result without importing vLLM internals.

        The pinned vLLM integration supplies ``request_ids`` and
        ``request_slots`` on its scheduler projection.  Accepting a small
        structural protocol keeps the runtime independent of vLLM's rapidly
        changing Python class hierarchy while still making missing identity a
        hard error.
        """
        projection = VllmSchedulerProjection.from_scheduler_output(scheduler_output)
        exact_demand = projection.exact_demand()
        return self.bind_batch(
            projection.request_ids,
            projection.request_slots,
            epoch=epoch,
            stream=stream,
            priorities=projection.priorities,
            deadline_clocks=projection.deadline_clocks,
            tenant_ids=projection.tenant_ids,
            exact_demand=exact_demand,
            granularity=granularity,
        )
