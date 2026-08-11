"""Generation-safe ownership for fixed-capacity CUDA K/V staging."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any

from .fixed_range_pool import FixedRangePool, RangeLease


@dataclass(frozen=True)
class StagingLease:
    """One fixed row range; identity changes whenever its slot is reused."""

    range: RangeLease

    @property
    def owner(self) -> int:
        return self.range.owner

    @property
    def generation(self) -> int:
        return self.range.generation

    @property
    def row_begin(self) -> int:
        return self.range.begin

    @property
    def row_end(self) -> int:
        return self.range.end


class BoundedStagingPool:
    """Preallocate layered K/V rows and fence every range before reuse."""

    def __init__(self, key: Any, value: Any, rows_per_lease: int) -> None:
        import torch

        if (
            not isinstance(key, torch.Tensor)
            or not isinstance(value, torch.Tensor)
            or key.device.type != "cuda"
            or value.device != key.device
            or key.dtype != value.dtype
            or key.shape != value.shape
            or key.ndim < 3
            or not key.is_contiguous()
            or not value.is_contiguous()
        ):
            raise ValueError(
                "bounded staging requires matching contiguous layered CUDA K/V"
            )
        if rows_per_lease <= 0 or int(key.shape[1]) < rows_per_lease:
            raise ValueError("bounded staging row geometry is invalid")
        self.key = key
        self.value = value
        self.rows_per_lease = rows_per_lease
        self._ranges = FixedRangePool(int(key.shape[1]), rows_per_lease)
        self._active: dict[int, StagingLease] = {}
        self._retired: list[tuple[StagingLease, Any]] = []
        self._lock = threading.Lock()

    @classmethod
    def allocate(
        cls,
        layer_count: int,
        capacity_rows: int,
        rows_per_lease: int,
        row_shape: tuple[int, ...],
        *,
        dtype: Any,
        device: Any,
    ) -> "BoundedStagingPool":
        import torch

        if layer_count <= 0 or capacity_rows <= 0 or not row_shape:
            raise ValueError("bounded staging allocation geometry is invalid")
        key = torch.empty(
            (layer_count, capacity_rows, *row_shape), dtype=dtype, device=device
        )
        return cls(key, torch.empty_like(key), rows_per_lease)

    @property
    def layer_count(self) -> int:
        return int(self.key.shape[0])

    @property
    def slot_count(self) -> int:
        return self._ranges.slot_count

    @property
    def capacity_bytes(self) -> int:
        return (self.key.numel() + self.value.numel()) * self.key.element_size()

    @property
    def leased_bytes(self) -> int:
        row_bytes = (
            (self.key[0, 0].numel() + self.value[0, 0].numel())
            * self.key.element_size()
            * self.layer_count
        )
        with self._lock:
            return (
                len(self._active) + len(self._retired)
            ) * self.rows_per_lease * row_bytes

    @property
    def high_water_bytes(self) -> int:
        return (
            self._ranges.high_watermark
            * self.rows_per_lease
            * (self.key[0, 0].numel() + self.value[0, 0].numel())
            * self.key.element_size()
            * self.layer_count
        )

    def acquire(self, owner: int) -> StagingLease:
        lease = StagingLease(self._ranges.acquire(owner))
        with self._lock:
            self._active[lease.range.slot] = lease
        return lease

    def view(self, lease: StagingLease, layer: int) -> tuple[Any, Any]:
        if not 0 <= layer < self.layer_count:
            raise IndexError("staging layer is out of range")
        with self._lock:
            if self._active.get(lease.range.slot) != lease:
                raise RuntimeError("stale or retired staging lease")
        return (
            self.key[layer, lease.row_begin : lease.row_end],
            self.value[layer, lease.row_begin : lease.row_end],
        )

    def retire(self, lease: StagingLease, completion: Any) -> None:
        """Remove a lease from use; reclaim waits for ``completion``."""
        if not callable(getattr(completion, "query", None)) or not callable(
            getattr(completion, "synchronize", None)
        ):
            raise TypeError("staging retirement requires a CUDA-event-like fence")
        with self._lock:
            if self._active.get(lease.range.slot) != lease:
                raise RuntimeError("stale or foreign staging lease")
            self._active.pop(lease.range.slot)
            self._retired.append((lease, completion))

    def reclaim(self, *, wait: bool = False) -> int:
        """Return completed retired ranges to the allocator.

        Fence waits happen outside the lock: a blocking CUDA synchronize
        under the pool mutex would stall every acquire and view for the
        duration of a device drain.
        """
        with self._lock:
            retired = self._retired
            self._retired = []
        reclaimed: list[StagingLease] = []
        pending: list[tuple[StagingLease, Any]] = []
        for lease, completion in retired:
            if wait:
                completion.synchronize()
            elif not completion.query():
                pending.append((lease, completion))
                continue
            reclaimed.append(lease)
        if pending:
            with self._lock:
                self._retired.extend(pending)
        for lease in reclaimed:
            self._ranges.release(lease.range)
        return len(reclaimed)
