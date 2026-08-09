"""Engine-neutral request identity and slot-reuse tracking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from typing import Any

from .runtime import RequestSpec


@dataclass(frozen=True)
class RequestBinding:
    request_index: int
    request_slot: int
    generation: int
    request_id: int
    priority: int = 0
    deadline_clock: int = 0


def stable_request_id(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


class RequestSlotTracker:
    """Assign generations to engine-owned slots and publish changes once."""

    def __init__(self, runtime: Any, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("request slot capacity must be positive")
        self._runtime = runtime
        self._capacity = capacity
        self._slots: dict[int, tuple[str, int, int, int]] = {}
        self.last_publish_count = 0
        self.last_policy_publish_count = 0

    def bind(
        self,
        request_ids: Sequence[str],
        request_slots: Sequence[int],
        *,
        priorities: Sequence[int] | None = None,
        deadline_clocks: Sequence[int] | None = None,
        stream: Any = None,
    ) -> tuple[RequestBinding, ...]:
        if len(request_ids) != len(request_slots):
            raise ValueError("request IDs and slots must have equal length")
        if len(set(request_slots)) != len(request_slots):
            raise ValueError("a serving batch cannot reuse a request slot")
        if priorities is None:
            priorities = [0] * len(request_ids)
        if deadline_clocks is None:
            deadline_clocks = [0] * len(request_ids)
        if len(priorities) != len(request_ids) or len(deadline_clocks) != len(
            request_ids
        ):
            raise ValueError("request policy arrays must match request IDs")
        self.last_publish_count = 0
        self.last_policy_publish_count = 0
        bindings: list[RequestBinding] = []
        updates: list[RequestSpec] = []
        slot_updates: list[tuple[int, tuple[str, int, int, int]]] = []
        for request_index, (request_id, request_slot, priority, deadline_clock) in enumerate(
            zip(request_ids, request_slots, priorities, deadline_clocks)
        ):
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("serving request IDs must be non-empty strings")
            if request_slot < 0 or request_slot >= self._capacity:
                raise ValueError(
                    f"request slot {request_slot} exceeds capacity {self._capacity}"
                )
            if priority < 0 or priority > 7 or deadline_clock < 0:
                raise ValueError("request priority or deadline is invalid")
            previous = self._slots.get(request_slot)
            if previous is None:
                generation = 1
            elif previous[0] == request_id:
                generation = previous[1]
            else:
                generation = (previous[1] + 1) & 0xFFFFFFFF
                if generation == 0:
                    generation = 1
            current = (request_id, generation, priority, deadline_clock)
            if previous != current:
                updates.append(
                    RequestSpec(
                        request_slot,
                        stable_request_id(request_id),
                        generation,
                        priority=priority,
                        deadline_clock=deadline_clock,
                    )
                )
                slot_updates.append((request_slot, current))
                if previous is None or previous[:2] != current[:2]:
                    self.last_publish_count += 1
                else:
                    self.last_policy_publish_count += 1
            bindings.append(
                RequestBinding(
                    request_index,
                    request_slot,
                    generation,
                    stable_request_id(request_id),
                    priority,
                    deadline_clock,
                )
            )
        if updates:
            if stream is None:
                for update in updates:
                    self._runtime.set_request(
                        update.slot,
                        update.request_id,
                        update.generation,
                        priority=update.priority,
                        deadline_clock=update.deadline_clock,
                    )
            else:
                self._runtime.publish_requests_async(updates, stream)
            for request_slot, current in slot_updates:
                self._slots[request_slot] = current
        return tuple(bindings)

    def cancel(self, request_id: str) -> bool:
        for request_slot, (active_id, generation, _, _) in self._slots.items():
            if active_id == request_id:
                self._runtime.cancel_request(request_slot, generation)
                return True
        return False

    def cancel_matching(self, request_id_prefix: str = "", *, all: bool = False) -> int:
        """Cancel every current generation selected by an engine abort event."""
        if not all and not request_id_prefix:
            return 0
        matches = [
            (request_slot, generation)
            for request_slot, (request_id, generation, _, _) in self._slots.items()
            if all or request_id.startswith(request_id_prefix)
        ]
        for request_slot, generation in matches:
            self._runtime.cancel_request(request_slot, generation)
        return len(matches)
