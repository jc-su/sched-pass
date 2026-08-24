"""Recyclable virtual-token ranges for external-prefix sidecar leases.

Virtual token ids live in [VIRTUAL_TOKEN_BASE, VIRTUAL_TOKEN_LIMIT]: above
any physical pool row, inside int32 for the engine's request tables. The
namespace is ~2^30 ids; a burn-forever cursor exhausts it after ~65K leases
of 16K tokens — hours into a soak run. Fixed-width range leases make
lifetime consumption bounded by *live* leases instead: a range returns to
the pool when its lease's resources release, which happens behind the GPU
completion fence, so no in-flight device work can still observe the ids.

Reuse is LIFO. Live-range disjointness plus the lease identity binding
(request id + pool row, refused on mismatch) make aliasing a hard error
rather than silent corruption, and VERIFY runs byte-check staged rows.
"""

from __future__ import annotations

from nta_runtime.fixed_range_pool import FixedRangePool, RangeLease

VIRTUAL_TOKEN_BASE = 1 << 30
VIRTUAL_TOKEN_LIMIT = (1 << 31) - 1


class VirtualTokenNamespace:
    """Lease fixed-width virtual-token ranges that recycle on release."""

    def __init__(self, range_tokens: int) -> None:
        if range_tokens <= 0:
            raise RuntimeError(
                "NTA_EXECUTION_VIRTUAL_RANGE_TOKENS must be positive"
            )
        self.range_tokens = range_tokens
        self._pool = FixedRangePool(
            VIRTUAL_TOKEN_LIMIT + 1 - VIRTUAL_TOKEN_BASE, range_tokens
        )

    @property
    def slot_count(self) -> int:
        return self._pool.slot_count

    @property
    def in_use(self) -> int:
        return self._pool.in_use

    @property
    def high_watermark(self) -> int:
        return self._pool.high_watermark

    def acquire(self, owner: int, token_count: int) -> tuple[RangeLease, int]:
        """Lease a range for ``token_count`` ids; returns (lease, begin)."""
        if token_count <= 0:
            raise RuntimeError("virtual range needs a positive token count")
        if token_count > self.range_tokens:
            raise RuntimeError(
                f"external prefix of {token_count} tokens exceeds the "
                f"virtual range width {self.range_tokens}; raise "
                "NTA_EXECUTION_VIRTUAL_RANGE_TOKENS"
            )
        try:
            lease = self._pool.acquire(owner)
        except RuntimeError as error:
            raise RuntimeError(
                "external virtual-token namespace is exhausted: "
                f"{self._pool.slot_count} ranges are live ({error})"
            ) from error
        begin = VIRTUAL_TOKEN_BASE + lease.begin
        if (
            begin < VIRTUAL_TOKEN_BASE
            or begin + token_count > VIRTUAL_TOKEN_LIMIT + 1
        ):
            self._pool.release(lease)
            raise RuntimeError("virtual range lease escaped the namespace")
        return lease, begin

    def release(self, lease: RangeLease) -> None:
        self._pool.release(lease)
