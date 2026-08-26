"""vLLM boundary adapter.

vLLM owns request IDs, block tables, cancellation, and scheduler slots.  The
adapter accepts those already-normalized values and deliberately does not
import vLLM, so contract tests run without a vLLM installation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import (
    ExactDemandProjection,
    EngineBatch,
    RequestIdentityAdapter,
    _integer_vector,
    _nested_integer_vector,
    _request_id_vector,
)
from ..abi import MAX_REQUEST_PRIORITY
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

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_ids",
            _request_id_vector(self.request_ids, "vLLM request IDs"),
        )
        object.__setattr__(
            self,
            "request_slots",
            _integer_vector(
                self.request_slots, "vLLM request slots", maximum=(1 << 32) - 1
            ),
        )
        if len(self.request_ids) != len(self.request_slots):
            raise ValueError("vLLM request IDs and slots must be aligned")
        for name, value, maximum in (
            ("vLLM priorities", self.priorities, MAX_REQUEST_PRIORITY),
            ("vLLM deadline clocks", self.deadline_clocks, (1 << 64) - 1),
            ("vLLM tenant IDs", self.tenant_ids, (1 << 32) - 1),
        ):
            if value is not None:
                normalized = _integer_vector(value, name, maximum=maximum)
                if len(normalized) != len(self.request_ids):
                    raise ValueError(f"{name} must match the request batch")
                object.__setattr__(
                    self,
                    {
                        "vLLM priorities": "priorities",
                        "vLLM deadline clocks": "deadline_clocks",
                        "vLLM tenant IDs": "tenant_ids",
                    }[name],
                    normalized,
                )
        if self.block_tables is not None:
            normalized_tables = _nested_integer_vector(
                self.block_tables, "vLLM block tables"
            )
            if len(normalized_tables) != len(self.request_ids):
                raise ValueError("vLLM block tables must match the request batch")
            object.__setattr__(self, "block_tables", normalized_tables)
        if self.page_bytes is not None:
            page_bytes = _integer_vector(
                (self.page_bytes,), "vLLM page bytes", minimum=1
            )[0]
            object.__setattr__(self, "page_bytes", page_bytes)

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
        raw_page_bytes = getattr(output, "kv_page_bytes", None)
        if raw_page_bytes is None:
            raw_page_bytes = getattr(output, "page_bytes", None)
        return cls(
            request_ids,
            request_slots,
            priorities,
            deadlines,
            tenant_ids,
            block_tables,
            raw_page_bytes,
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
