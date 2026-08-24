#!/usr/bin/env python3
"""Virtual-token namespace recycling invariants (RQ4 soak precondition)."""

from nta_runtime.virtual_namespace import (
    VIRTUAL_TOKEN_BASE,
    VIRTUAL_TOKEN_LIMIT,
    VirtualTokenNamespace,
)


def expect_error(action, needle):
    try:
        action()
    except RuntimeError as error:
        if needle not in str(error):
            raise AssertionError(
                f"error {error!r} does not mention {needle!r}"
            ) from error
        return
    raise AssertionError(f"expected a RuntimeError mentioning {needle!r}")


WIDTH = 1 << 17
namespace = VirtualTokenNamespace(WIDTH)
assert namespace.slot_count == (1 << 30) // WIDTH

# Live leases are disjoint and inside [BASE, LIMIT].
live = []
spans = []
for owner in range(1, 65):
    lease, begin = namespace.acquire(owner, 16384)
    end = begin + WIDTH
    assert VIRTUAL_TOKEN_BASE <= begin and end <= VIRTUAL_TOKEN_LIMIT + 1
    for other_begin, other_end in spans:
        assert not (begin < other_end and other_begin < end), "ranges overlap"
    spans.append((begin, end))
    live.append(lease)
assert namespace.in_use == 64

# Recycling: lifetime leases exceed slot_count many times over — the cursor
# design this replaces exhausts exactly here.
for lease in live:
    namespace.release(lease)
for owner in range(100, 100 + namespace.slot_count * 3):
    lease, _ = namespace.acquire(owner, 16384)
    namespace.release(lease)
assert namespace.in_use == 0

# Exhaustion is bounded by live leases and fail-closed.
held = [
    namespace.acquire(1_000_000 + index, 8)[0]
    for index in range(namespace.slot_count)
]
expect_error(lambda: namespace.acquire(9_999_999, 8), "exhausted")
for lease in held:
    namespace.release(lease)

# A lease wider than the range width is refused with the remedy named.
expect_error(
    lambda: namespace.acquire(7, WIDTH + 1), "NTA_EXECUTION_VIRTUAL_RANGE_TOKENS"
)

# Stale and double releases are refused.
lease, _ = namespace.acquire(8, 4)
namespace.release(lease)
expect_error(lambda: namespace.release(lease), "stale")

print("virtual-range recycling invariants hold")
