"""Engine-neutral request identity and slot-reuse tracking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from typing import Any

from .abi import MAX_REQUEST_PRIORITY, bounded_integer, u32, u64
from .request_contract import RequestSpec


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
        # Normalize every value before it is retained in an immutable binding.
        # This keeps NumPy integer scalars usable while rejecting bools/floats
        # instead of allowing a later ctypes conversion to truncate them.
        object.__setattr__(
            self, "request_index", u32(self.request_index, "request index")
        )
        object.__setattr__(self, "request_slot", u32(self.request_slot, "request slot"))
        object.__setattr__(
            self,
            "generation",
            u32(self.generation, "request generation", positive=True),
        )
        object.__setattr__(self, "request_id", u64(self.request_id, "request id"))
        object.__setattr__(
            self,
            "priority",
            bounded_integer(
                self.priority,
                "request priority",
                minimum=0,
                maximum=MAX_REQUEST_PRIORITY,
            ),
        )
        object.__setattr__(
            self,
            "deadline_clock",
            u64(self.deadline_clock, "request deadline"),
        )
        object.__setattr__(self, "tenant_id", u32(self.tenant_id, "request tenant"))


def stable_request_id(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


class RequestIdentityRegistry:
    """Own engine-slot identity and publish each generation exactly once."""

    def __init__(self, runtime: Any, capacity: int) -> None:
        self._capacity = u32(capacity, "request slot capacity", positive=True)
        self._runtime = runtime
        self._slots: dict[int, tuple[str, int, int, int, int]] = {}
        # The native ABI intentionally carries a compact uint64 identity.  The
        # collision table is bounded by the number of active slots: generation
        # checks protect retired work, so retaining every historical spelling
        # would only create an unbounded control-plane leak in a long-running
        # server.
        self._active_ids: dict[int, tuple[str, int]] = {}
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
        try:
            normalized_slots = tuple(
                u32(value, "request slot") for value in request_slots
            )
        except ValueError as error:
            raise ValueError("request slots must be uint32 integers") from error
        if len(set(normalized_slots)) != len(normalized_slots):
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
        try:
            normalized_priorities = tuple(
                bounded_integer(
                    value,
                    "request priority",
                    minimum=0,
                    maximum=MAX_REQUEST_PRIORITY,
                )
                for value in priorities
            )
            normalized_deadlines = tuple(
                u64(value, "request deadline") for value in deadline_clocks
            )
            normalized_tenants = tuple(
                u32(value, "request tenant") for value in tenant_ids
            )
        except ValueError as error:
            raise ValueError("request metadata contains an invalid integer") from error
        self.last_publish_count = 0
        self.last_metadata_publish_count = 0
        bindings: list[RequestBinding] = []
        updates: list[RequestSpec] = []
        slot_updates: list[tuple[int, tuple[str, int, int, int, int]]] = []
        batch_request_slots: dict[str, int] = {}
        target_slots = set(normalized_slots)
        # The engine may temporarily omit an active request from a forward
        # batch.  Omission is not retirement: keep every other active slot in
        # the collision table and remove only identities whose slots are
        # actually being rebound by this batch.
        prospective_active_ids = dict(self._active_ids)
        for stable_id, identity in tuple(prospective_active_ids.items()):
            if identity[1] in target_slots:
                prospective_active_ids.pop(stable_id, None)
        batch_stable_ids: dict[int, str] = {}
        for request_index, (
            request_id,
            request_slot,
            priority,
            deadline_clock,
            tenant_id,
        ) in enumerate(
            zip(
                request_ids,
                normalized_slots,
                normalized_priorities,
                normalized_deadlines,
                normalized_tenants,
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
            previous_batch_id = batch_stable_ids.get(stable_id)
            if previous_batch_id is not None and previous_batch_id != request_id:
                raise ValueError(
                    "request ID hash collision for native uint64 identity: "
                    f"{previous_batch_id!r} and {request_id!r}"
                )
            previous_active = prospective_active_ids.get(stable_id)
            if previous_active is not None:
                previous_active_id, previous_active_slot = previous_active
                if previous_active_id != request_id:
                    raise ValueError(
                        "request ID hash collision for native uint64 identity: "
                        f"{previous_active_id!r} and {request_id!r}"
                    )
                if previous_active_slot != request_slot:
                    raise ValueError(
                        "a serving request ID cannot be active in multiple slots: "
                        f"request_id={request_id!r}, active_slot="
                        f"{previous_active_slot}, incoming_slot={request_slot}, "
                        f"batch_slots={normalized_slots!r}"
                    )
            batch_stable_ids[stable_id] = request_id
            prospective_active_ids[stable_id] = (request_id, request_slot)
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
        self._active_ids = prospective_active_ids
        return tuple(bindings)

    def cancel(self, request_id: str) -> bool:
        active = self._active_ids.get(stable_request_id(request_id))
        if active is None or active[0] != request_id:
            return False
        request_slot = active[1]
        current = self._slots.get(request_slot)
        if (
            request_slot not in self._active_slots
            or current is None
            or current[0] != request_id
        ):
            # Keep the bounded identity table self-healing if an engine-side
            # cancellation hook was called during exceptional teardown.
            self._active_ids.pop(stable_request_id(request_id), None)
            self._active_slots.discard(request_slot)
            return False
        generation = current[1]
        self._runtime.cancel_request(request_slot, generation)
        self._active_slots.remove(request_slot)
        self._active_ids.pop(stable_request_id(request_id), None)
        return True

    def cancel_matching(self, request_id_prefix: str = "", *, all: bool = False) -> int:
        """Cancel every current generation selected by an engine abort event."""
        if not all and not request_id_prefix:
            return 0
        matches = [
            (request_slot, request_id, generation)
            for request_slot, (request_id, generation, _, _, _) in self._slots.items()
            if request_slot in self._active_slots
            and (all or request_id.startswith(request_id_prefix))
        ]
        for request_slot, request_id, generation in matches:
            self._runtime.cancel_request(request_slot, generation)
            self._active_slots.remove(request_slot)
            self._active_ids.pop(stable_request_id(request_id), None)
        return len(matches)
