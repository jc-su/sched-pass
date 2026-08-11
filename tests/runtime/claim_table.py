#!/usr/bin/env python3
"""Exercise the fixed-shape claim table's lifecycle and pointer stability."""

from __future__ import annotations


class Fence:
    def __init__(self) -> None:
        self.complete = False

    def query(self) -> bool:
        return self.complete


def main() -> int:
    import torch

    from nta_runtime.claim_table import ClaimTable

    if not torch.cuda.is_available():
        print("SKIP: CUDA is unavailable")
        return 0
    table = ClaimTable(4, 8, 16, device="cuda")

    pointers = {
        name: getattr(table, name).data_ptr()
        for name in (
            "claim_ids",
            "valid",
            "active_count",
            "selected_pages",
            "staging_rows",
            "selected_rows",
            "source_indices",
        )
    }

    first = table.acquire(11)
    second = table.acquire(12)
    assert int(table.active_count[0]) == 0, "activation is explicit"
    table.activate(first)
    table.activate(second)
    assert int(table.active_count[0]) == 2
    assert int(table.valid[first.index]) == 1

    views = table.views(first)
    views["selected_pages"][:3] = torch.tensor(
        [5, 9, 2], dtype=torch.int64, device="cuda"
    )
    assert int(table.selected_pages[first.index, 1]) == 9
    assert views["staging_rows"].shape == (8 * 16,)

    # Retirement invalidates immediately; reuse waits on the fence, and
    # the reused row carries a new generation.
    fence = Fence()
    table.retire(first, fence)
    assert int(table.valid[first.index]) == 0
    assert int(table.active_count[0]) == 1
    assert table.reclaim() == 0, "unfenced rows must not be reclaimed"
    try:
        table.views(first)
    except RuntimeError:
        pass
    else:
        raise AssertionError("stale slot view must fail")
    fence.complete = True
    assert table.reclaim() == 1
    third = table.acquire(13)
    reused = table.acquire(14)
    fourth = table.acquire(15)
    indexes = {slot.index for slot in (second, third, reused, fourth)}
    assert len(indexes) == 4, "all four rows are distinct while bound"
    generations = [
        slot.generation
        for slot in (third, reused, fourth)
        if slot.index == first.index
    ]
    assert generations and generations[0] != first.generation

    try:
        table.acquire(16)
    except RuntimeError:
        pass
    else:
        raise AssertionError("exhaustion must fail closed")

    for name, pointer in pointers.items():
        assert getattr(table, name).data_ptr() == pointer, (
            f"{name} pointer moved; capture would replay stale addresses"
        )
    assert table.high_watermark == 4
    print("claim table lifecycle holds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
