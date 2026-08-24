"""Common request-identity adapter with no engine-specific imports."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

from ..requests import RequestBinding, RequestSlotTracker
from ..work_unit import Granularity


@dataclass(frozen=True)
class EngineBatch:
    """The minimum engine-to-runtime handoff for one forward."""

    engine: str
    epoch: int
    bindings: tuple[RequestBinding, ...]
    granularity: Granularity

    def __post_init__(self) -> None:
        if not self.engine:
            raise ValueError("engine name must be non-empty")
        if self.epoch < 0:
            raise ValueError("engine batch epoch cannot be negative")
        if not self.bindings:
            raise ValueError("engine batch must contain at least one request")
        if len({binding.request_slot for binding in self.bindings}) != len(self.bindings):
            raise ValueError("engine batch cannot reuse a request slot")


class RequestIdentityAdapter:
    """Shared slot/generation implementation used by frontend adapters."""

    def __init__(self, runtime: Any, request_capacity: int, *, engine: str) -> None:
        self.engine = engine
        self._tracker = RequestSlotTracker(runtime, request_capacity)

    def bind(
        self,
        request_ids: Sequence[str],
        request_slots: Sequence[int],
        *,
        priorities: Sequence[int] | None = None,
        deadline_clocks: Sequence[int] | None = None,
        stream: Any = None,
    ) -> tuple[RequestBinding, ...]:
        return self._tracker.bind(
            request_ids,
            request_slots,
            priorities=priorities,
            deadline_clocks=deadline_clocks,
            stream=stream,
        )

    def cancel_matching(self, request_id_prefix: str = "", *, all: bool = False) -> int:
        return self._tracker.cancel_matching(request_id_prefix, all=all)

    @property
    def last_publish_count(self) -> int:
        return self._tracker.last_publish_count

    @property
    def last_policy_publish_count(self) -> int:
        return self._tracker.last_policy_publish_count
