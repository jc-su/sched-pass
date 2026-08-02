"""Public NTA Python API with lazy native-runtime loading.

Serving engines discover plugins in frontend and worker processes. Keeping the
native bindings lazy lets discovery and argument parsing run without loading a
CUDA library or selecting a device.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "API_VERSION": ("runtime", "API_VERSION"),
    "AcquireRequirement": ("runtime", "AcquireRequirement"),
    "WorkTicketState": ("runtime", "WorkTicketState"),
    "DeviceWorkPlan": ("runtime", "DeviceWorkPlan"),
    "EpochStatus": ("runtime", "EpochStatus"),
    "JitPhaseProgram": ("runtime", "JitPhaseProgram"),
    "IndexedHostObject": ("runtime", "IndexedHostObject"),
    "NvmeCapabilities": ("runtime", "NvmeCapabilities"),
    "NvmeOptions": ("runtime", "NvmeOptions"),
    "NvmeQueueStats": ("runtime", "NvmeQueueStats"),
    "NvmeTransport": ("runtime", "NvmeTransport"),
    "Placement": ("runtime", "Placement"),
    "Replica": ("runtime", "Replica"),
    "RequestRange": ("runtime", "RequestRange"),
    "RequestProgress": ("runtime", "RequestProgress"),
    "Runtime": ("runtime", "Runtime"),
    "RuntimeConfig": ("runtime", "RuntimeConfig"),
    "RuntimeError": ("runtime", "RuntimeError"),
    "WorkItem": ("runtime", "WorkItem"),
    "device_abi_version": ("runtime", "device_abi_version"),
    "synchronize_stream": ("runtime", "synchronize_stream"),
    "BoundedEpoch": ("epoch", "BoundedEpoch"),
    "EpochResult": ("epoch", "EpochResult"),
    "FlashInferLayerEpoch": ("flashinfer", "FlashInferLayerEpoch"),
    "attention_jit_args": ("flashinfer", "attention_jit_args"),
    "enqueue_resident_attention": ("flashinfer", "enqueue_resident_attention"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value
