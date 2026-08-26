"""Dependency-free request identity contract shared by engines and runtime.

The semantic request object is deliberately separate from ``runtime.py``.
Importing an engine adapter or constructing a work unit must not load a CUDA
shared library merely to represent slot/generation identity.
"""

from __future__ import annotations

import ctypes
import dataclasses

from .abi import u32 as _u32
from .abi import u64 as _u64
from .abi import MAX_REQUEST_PRIORITY as _MAX_REQUEST_PRIORITY
from .abi import bounded_integer as _bounded_integer


class _RequestSpec(ctypes.Structure):
    """Private ctypes representation of ``nta_request_spec``."""

    _fields_ = [
        ("request_id", ctypes.c_uint64),
        ("deadline_clock", ctypes.c_uint64),
        ("max_outstanding_bytes", ctypes.c_uint64),
        ("slot", ctypes.c_uint32),
        ("generation", ctypes.c_uint32),
        ("tenant_id", ctypes.c_uint32),
        ("priority", ctypes.c_uint32),
    ]


@dataclasses.dataclass(frozen=True)
class RequestSpec:
    """One generation-bound request slot published to the native runtime."""

    slot: int
    request_id: int
    generation: int
    tenant_id: int = 0
    priority: int = 0
    deadline_clock: int = 0
    max_outstanding_bytes: int = (1 << 64) - 1

    def __post_init__(self) -> None:
        _u32(self.slot, "request slot")
        _u64(self.request_id, "request id")
        _u32(self.generation, "request generation", positive=True)
        _u32(self.tenant_id, "request tenant")
        _bounded_integer(
            self.priority,
            "request priority",
            minimum=0,
            maximum=_MAX_REQUEST_PRIORITY,
        )
        _u64(self.deadline_clock, "request deadline")
        _u64(self.max_outstanding_bytes, "request max outstanding bytes")

    def native(self) -> _RequestSpec:
        return _RequestSpec(
            self.request_id,
            self.deadline_clock,
            self.max_outstanding_bytes,
            self.slot,
            self.generation,
            self.tenant_id,
            self.priority,
        )
