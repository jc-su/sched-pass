"""Fixed-capacity range leases for graph-stable device tables."""

from __future__ import annotations

from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class RangeLease:
    slot: int
    generation: int
    begin: int
    end: int
    owner: int


class FixedRangePool:
    """Lease equal-width ranges without allocation on the serving hot path."""

    def __init__(self, capacity: int, width: int, *, reserved_low: int = 0) -> None:
        if capacity <= 0 or width <= 0 or reserved_low < 0:
            raise ValueError("range-pool geometry must be positive")
        available = capacity - reserved_low
        self._slot_count = available // width
        if self._slot_count <= 0:
            raise ValueError("range pool has no usable slots")
        self._capacity = capacity
        self._width = width
        self._reserved_low = reserved_low
        self._generations = [0] * self._slot_count
        self._free = list(range(self._slot_count))
        self._by_owner: dict[int, RangeLease] = {}
        self._by_slot: dict[int, RangeLease] = {}
        self._high_watermark = 0
        self._lock = threading.Lock()

    @property
    def slot_count(self) -> int:
        return self._slot_count

    @property
    def in_use(self) -> int:
        with self._lock:
            return len(self._by_slot)

    @property
    def high_watermark(self) -> int:
        with self._lock:
            return self._high_watermark

    def acquire(self, owner: int) -> RangeLease:
        if owner <= 0:
            raise ValueError("range owner must be positive")
        with self._lock:
            if owner in self._by_owner:
                raise RuntimeError("range owner already has a live lease")
            if not self._free:
                raise RuntimeError("fixed range pool is exhausted")
            slot = self._free.pop()
            generation = (self._generations[slot] + 1) & 0xFFFFFFFF
            generation = generation or 1
            self._generations[slot] = generation
            end = self._capacity - slot * self._width
            begin = end - self._width
            if begin < self._reserved_low:
                raise RuntimeError("range-pool arithmetic crossed its reservation")
            lease = RangeLease(slot, generation, begin, end, owner)
            self._by_owner[owner] = lease
            self._by_slot[slot] = lease
            self._high_watermark = max(self._high_watermark, len(self._by_slot))
            return lease

    def release(self, lease: RangeLease) -> None:
        with self._lock:
            current = self._by_slot.get(lease.slot)
            if current != lease or self._by_owner.get(lease.owner) != lease:
                raise RuntimeError("stale or foreign range lease")
            self._by_slot.pop(lease.slot)
            self._by_owner.pop(lease.owner)
            self._free.append(lease.slot)

