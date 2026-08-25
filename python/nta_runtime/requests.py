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
    tenant_id: int = 0

    def __post_init__(self) -> None:
        if (
            min(self.request_index, self.request_slot, self.generation, self.request_id)
            < 0
        ):
            raise ValueError("request binding identity fields must be nonnegative")
        if self.generation == 0:
            raise ValueError("request generation must be positive")
        if not 0 <= self.priority <= 7 or self.deadline_clock < 0 or self.tenant_id < 0:
            raise ValueError("request priority, deadline, or tenant is invalid")


def stable_request_id(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


class RequestIdentityRegistry:
    """Own engine-slot identity and publish each generation exactly once."""

    def __init__(self, runtime: Any, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("request slot capacity must be positive")
        self._runtime = runtime
        self._capacity = capacity
        self._slots: dict[int, tuple[str, int, int, int, int]] = {}
        # The native ABI intentionally carries a compact uint64 identity.  Keep
        # the source spelling for every identity ever observed so a collision
        # fails closed instead of aliasing two tenants/requests after reuse.
        self._stable_ids: dict[int, str] = {}
        self._active_slots: set[int] = set()
        self.last_publish_count = 0
        self.last_metadata_publish_count = 0

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
        if len(request_ids) != len(request_slots):
            raise ValueError("request IDs and slots must have equal length")
        if len(set(request_slots)) != len(request_slots):
            raise ValueError("a serving batch cannot reuse a request slot")
        if priorities is None:
            priorities = [0] * len(request_ids)
        if deadline_clocks is None:
            deadline_clocks = [0] * len(request_ids)
        if tenant_ids is None:
            tenant_ids = [0] * len(request_ids)
        if (
            len(priorities) != len(request_ids)
            or len(deadline_clocks) != len(request_ids)
            or len(tenant_ids) != len(request_ids)
        ):
            raise ValueError("request metadata arrays must match request IDs")
        self.last_publish_count = 0
        self.last_metadata_publish_count = 0
        bindings: list[RequestBinding] = []
        updates: list[RequestSpec] = []
        slot_updates: list[tuple[int, tuple[str, int, int, int, int]]] = []
        batch_request_slots: dict[str, int] = {}
        for request_index, (
            request_id,
            request_slot,
            priority,
            deadline_clock,
            tenant_id,
        ) in enumerate(
            zip(
                request_ids,
                request_slots,
                priorities,
                deadline_clocks,
                tenant_ids,
                strict=True,
            )
        ):
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("serving request IDs must be non-empty strings")
            previous_slot = batch_request_slots.get(request_id)
            if previous_slot is not None and previous_slot != request_slot:
                raise ValueError(
                    "a serving batch cannot bind one request ID to multiple slots"
                )
            batch_request_slots[request_id] = request_slot
            stable_id = stable_request_id(request_id)
            previous_id = self._stable_ids.get(stable_id)
            if previous_id is not None and previous_id != request_id:
                raise ValueError(
                    "request ID hash collision for native uint64 identity: "
                    f"{previous_id!r} and {request_id!r}"
                )
            self._stable_ids.setdefault(stable_id, request_id)
            if request_slot < 0 or request_slot >= self._capacity:
                raise ValueError(
                    f"request slot {request_slot} exceeds capacity {self._capacity}"
                )
            if priority < 0 or priority > 7 or deadline_clock < 0 or tenant_id < 0:
                raise ValueError("request priority, deadline, or tenant is invalid")
            previous = self._slots.get(request_slot)
            if previous is None:
                generation = 1
            elif request_slot in self._active_slots and previous[0] == request_id:
                generation = previous[1]
            else:
                generation = (previous[1] + 1) & 0xFFFFFFFF
                if generation == 0:
                    generation = 1
            current = (request_id, generation, tenant_id, priority, deadline_clock)
            if previous != current:
                updates.append(
                    RequestSpec(
                        request_slot,
                        stable_id,
                        generation,
                        tenant_id=tenant_id,
                        priority=priority,
                        deadline_clock=deadline_clock,
                    )
                )
                slot_updates.append((request_slot, current))
                if previous is None or previous[:2] != current[:2]:
                    self.last_publish_count += 1
                else:
                    self.last_metadata_publish_count += 1
            bindings.append(
                RequestBinding(
                    request_index,
                    request_slot,
                    generation,
                    stable_id,
                    priority,
                    deadline_clock,
                    tenant_id,
                )
            )
        if updates:
            if stream is None:
                for update in updates:
                    self._runtime.set_request(
                        update.slot,
                        update.request_id,
                        update.generation,
                        tenant_id=update.tenant_id,
                        priority=update.priority,
                        deadline_clock=update.deadline_clock,
                    )
            else:
                self._runtime.publish_requests_async(updates, stream)
            for request_slot, current in slot_updates:
                self._slots[request_slot] = current
                self._active_slots.add(request_slot)
        return tuple(bindings)

    def cancel(self, request_id: str) -> bool:
        for request_slot, (active_id, generation, _, _, _) in self._slots.items():
            if request_slot in self._active_slots and active_id == request_id:
                self._runtime.cancel_request(request_slot, generation)
                self._active_slots.remove(request_slot)
                return True
        return False

    def cancel_matching(self, request_id_prefix: str = "", *, all: bool = False) -> int:
        """Cancel every current generation selected by an engine abort event."""
        if not all and not request_id_prefix:
            return 0
        matches = [
            (request_slot, generation)
            for request_slot, (request_id, generation, _, _, _) in self._slots.items()
            if request_slot in self._active_slots
            and (all or request_id.startswith(request_id_prefix))
        ]
        for request_slot, generation in matches:
            self._runtime.cancel_request(request_slot, generation)
            self._active_slots.remove(request_slot)
        return len(matches)
