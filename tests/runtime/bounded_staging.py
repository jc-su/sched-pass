#!/usr/bin/env python3
"""Exercise bounded CUDA staging ownership and deferred slot reuse."""

from __future__ import annotations


class Fence:
    def __init__(self) -> None:
        self.complete = False

    def query(self) -> bool:
        return self.complete

    def synchronize(self) -> None:
        self.complete = True


def main() -> int:
    import torch

    from nta_runtime.bounded_staging import BoundedStagingPool

    if not torch.cuda.is_available():
        print("SKIP: CUDA is unavailable")
        return 0
    pool = BoundedStagingPool.allocate(
        2, 8, 4, (2, 8), dtype=torch.float16, device="cuda"
    )
    first = pool.acquire(101)
    second = pool.acquire(102)
    first_key, first_value = pool.view(first, 1)
    assert first_key.shape == (4, 2, 8)
    assert first_value.shape == first_key.shape
    assert first_key.is_contiguous() and first_value.is_contiguous()
    assert pool.capacity_bytes == 2 * 8 * 2 * 8 * 2 * 2
    assert pool.leased_bytes == pool.capacity_bytes
    assert pool.high_water_bytes == pool.capacity_bytes
    try:
        pool.acquire(103)
    except RuntimeError:
        pass
    else:
        raise AssertionError("bounded staging exceeded its fixed capacity")

    fence = Fence()
    pool.retire(first, fence)
    assert pool.leased_bytes == pool.capacity_bytes
    try:
        pool.view(first, 0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("retired staging remained addressable")
    assert pool.reclaim() == 0
    try:
        pool.acquire(103)
    except RuntimeError:
        pass
    else:
        raise AssertionError("staging reused a range before its fence")
    fence.complete = True
    assert pool.reclaim() == 1
    reused = pool.acquire(103)
    assert reused.range.slot == first.range.slot
    assert reused.generation != first.generation
    try:
        pool.retire(first, Fence())
    except RuntimeError:
        pass
    else:
        raise AssertionError("bounded staging accepted a stale generation")

    second_fence = Fence()
    reused_fence = Fence()
    pool.retire(second, second_fence)
    pool.retire(reused, reused_fence)
    assert pool.reclaim(wait=True) == 2
    assert pool.leased_bytes == 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
