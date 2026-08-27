"""Dependency-free validation for values crossing the native ABI.

This module intentionally has no CUDA or shared-library imports. Semantic
contracts and engine adapters can use the same validation rules without
initializing the native runtime.
"""

from __future__ import annotations

import numbers


UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
MAX_REQUEST_PRIORITY = 7


def bounded_integer(value: int, name: str, *, minimum: int, maximum: int) -> int:
    """Validate and normalize an integer before it crosses the C ABI."""

    # Native-plan construction validates thousands of scalar fields per
    # serving batch.  Engine adapters overwhelmingly pass built-in ``int``;
    # avoid the comparatively expensive ``numbers.Integral`` ABC walk for that
    # exact type while retaining support for NumPy/PyTorch integer scalars.
    if type(value) is int:
        result = value
    else:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError(f"{name} must be an integer")
        result = int(value)
    if result < minimum or result > maximum:
        raise ValueError(f"{name} is outside [{minimum}, {maximum}]")
    return result


def u32(value: int, name: str, *, positive: bool = False) -> int:
    return bounded_integer(
        value, name, minimum=1 if positive else 0, maximum=UINT32_MAX
    )


def u64(value: int, name: str, *, positive: bool = False) -> int:
    return bounded_integer(
        value, name, minimum=1 if positive else 0, maximum=UINT64_MAX
    )
