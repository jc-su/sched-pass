"""Common request-identity adapter with no engine-specific imports."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from ..requests import RequestBinding, RequestIdentityRegistry
from ..work_unit import Granularity


@dataclass(frozen=True)
class ExactDemandProjection:
    """Engine-neutral exact page/unit demand supplied by an adapter."""

    request_unit_ids: tuple[tuple[int, ...], ...]
    unit_bytes: int

    def __post_init__(self) -> None:
        if self.unit_bytes <= 0:
            raise ValueError("exact demand unit_bytes must be positive")
        if not self.request_unit_ids or any(
            not units for units in self.request_unit_ids
        ):
            raise ValueError("exact demand must contain units for every request")
        if any(
            any(unit < 0 for unit in units) or len(set(units)) != len(units)
            for units in self.request_unit_ids
        ):
            raise ValueError("exact demand unit IDs must be unique and nonnegative")


@dataclass(frozen=True)
class EngineBatch:
    """The minimum engine-to-runtime handoff for one forward."""

    engine: str
    epoch: int
    bindings: tuple[RequestBinding, ...]
    granularity: Granularity
    exact_demand: ExactDemandProjection | None = None

    def __post_init__(self) -> None:
        if not self.engine:
            raise ValueError("engine name must be non-empty")
        if self.epoch < 0:
            raise ValueError("engine batch epoch cannot be negative")
        if not self.bindings:
            raise ValueError("engine batch must contain at least one request")
        if len({binding.request_slot for binding in self.bindings}) != len(
            self.bindings
        ):
            raise ValueError("engine batch cannot reuse a request slot")
        if len({binding.request_index for binding in self.bindings}) != len(
            self.bindings
        ):
            raise ValueError("engine batch cannot reuse a request index")
        if self.exact_demand is not None and len(
            self.exact_demand.request_unit_ids
        ) != len(self.bindings):
            raise ValueError("exact demand rows must match the engine batch")

    @property
    def request_ids(self) -> tuple[int, ...]:
        return tuple(binding.request_id for binding in self.bindings)

    @property
    def request_slots(self) -> tuple[int, ...]:
        return tuple(binding.request_slot for binding in self.bindings)

    @property
    def generations(self) -> tuple[int, ...]:
        return tuple(binding.generation for binding in self.bindings)

    @property
    def tenant_ids(self) -> tuple[int, ...]:
        """Logical tenant for each request in the engine batch."""
        return tuple(binding.tenant_id for binding in self.bindings)


@runtime_checkable
class EngineBoundary(Protocol):
    """The one lifecycle boundary shared by SGLang and vLLM adapters."""

    engine: str

    def bind_forward(
        self,
        forward_batch: Any,
        *,
        epoch: int,
        stream: Any = None,
        granularity: Granularity = Granularity.PAGE_GROUP,
        **kwargs: Any,
    ) -> EngineBatch: ...

    def cancel_matching(
        self, request_id_prefix: str = "", *, all: bool = False
    ) -> int: ...


class RequestIdentityAdapter:
    """Shared slot/generation implementation used by frontend adapters."""

    def __init__(self, runtime: Any, request_capacity: int, *, engine: str) -> None:
        self.engine = engine
        self._identity = RequestIdentityRegistry(runtime, request_capacity)

    def bind(
        self,
        request_ids: Sequence[str],
        request_slots: Sequence[int],
        *,
        priorities: Sequence[int] | None = None,
        deadline_clocks: Sequence[int] | None = None,
        tenant_ids: Sequence[int] | None = None,
        stream: Any = None,
    ) -> tuple[RequestBinding, ...]:
        return self._identity.bind(
            request_ids,
            request_slots,
            priorities=priorities,
            deadline_clocks=deadline_clocks,
            tenant_ids=tenant_ids,
            stream=stream,
        )

    def cancel_matching(self, request_id_prefix: str = "", *, all: bool = False) -> int:
        return self._identity.cancel_matching(request_id_prefix, all=all)

    def retire_request(self, request_id: str) -> bool:
        """Retire one engine-confirmed request at the lifecycle boundary."""
        return self._identity.cancel(request_id)

    @property
    def last_publish_count(self) -> int:
        return self._identity.last_publish_count

    @property
    def last_metadata_publish_count(self) -> int:
        return self._identity.last_metadata_publish_count
