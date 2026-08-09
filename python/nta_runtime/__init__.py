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
    "OperatorCapability": ("runtime", "OperatorCapability"),
    "OperatorContract": ("runtime", "OperatorContract"),
    "OperatorCoordinateMap": ("runtime", "OperatorCoordinateMap"),
    "OperatorFamily": ("runtime", "OperatorFamily"),
    "OperatorForm": ("runtime", "OperatorForm"),
    "OperatorPartialState": ("runtime", "OperatorPartialState"),
    "OperatorPlan": ("runtime", "OperatorPlan"),
    "OperatorPlanFlag": ("runtime", "OperatorPlanFlag"),
    "OperatorReduction": ("runtime", "OperatorReduction"),
    "require_operator_pair": ("runtime", "require_operator_pair"),
    "NvmeOptions": ("runtime", "NvmeOptions"),
    "NvmeQueueStats": ("runtime", "NvmeQueueStats"),
    "NvmeTransport": ("runtime", "NvmeTransport"),
    "Placement": ("runtime", "Placement"),
    "Replica": ("runtime", "Replica"),
    "RequestRange": ("runtime", "RequestRange"),
    "RequestSpec": ("runtime", "RequestSpec"),
    "RequestProgress": ("runtime", "RequestProgress"),
    "RequestProgressSnapshot": ("runtime", "RequestProgressSnapshot"),
    "Runtime": ("runtime", "Runtime"),
    "RuntimeConfig": ("runtime", "RuntimeConfig"),
    "RuntimeError": ("runtime", "RuntimeError"),
    "WorkItem": ("runtime", "WorkItem"),
    "device_abi_version": ("runtime", "device_abi_version"),
    "copy_host_to_device_async": ("runtime", "copy_host_to_device_async"),
    "synchronize_stream": ("runtime", "synchronize_stream"),
    "CriticalWork": ("critical_work", "CriticalWork"),
    "CriticalWorkPlan": ("critical_work", "CriticalWorkPlan"),
    "RequestWork": ("critical_work", "RequestWork"),
    "ServiceModel": ("critical_work", "ServiceModel"),
    "estimate_critical_work": ("critical_work", "estimate_critical_work"),
    "plan_critical_work": ("critical_work", "plan_critical_work"),
    "BoundedEpoch": ("epoch", "BoundedEpoch"),
    "EpochResult": ("epoch", "EpochResult"),
    "FlashInferLayerEpoch": ("flashinfer", "FlashInferLayerEpoch"),
    "FlashInferHostWave": (
        "flashinfer_tier_streaming",
        "FlashInferHostWave",
    ),
    "FlashInferTierStreamingExecutor": (
        "flashinfer_tier_streaming",
        "FlashInferTierStreamingExecutor",
    ),
    "FlashInferTierStreamingGraph": (
        "flashinfer_tier_streaming",
        "FlashInferTierStreamingGraph",
    ),
    "FlashInferTierStreamingOperator": (
        "flashinfer_tier_streaming",
        "FlashInferTierStreamingOperator",
    ),
    "attention_jit_args": ("flashinfer", "attention_jit_args"),
    "request_bound_attention_jit_args": (
        "flashinfer",
        "request_bound_attention_jit_args",
    ),
    "enqueue_resident_attention": ("flashinfer", "enqueue_resident_attention"),
    "DeviceDemandCostModel": ("execution_policy", "DeviceDemandCostModel"),
    "DeviceDemandPlan": ("execution_policy", "DeviceDemandPlan"),
    "plan_device_demand": ("execution_policy", "plan_device_demand"),
    "SelectedPageAcquisition": ("selected_pages", "SelectedPageAcquisition"),
    "build_selected_page_work_plan": (
        "selected_pages",
        "build_selected_page_work_plan",
    ),
    "register_selected_host_pages": (
        "selected_pages",
        "register_selected_host_pages",
    ),
    "TierStreamingRequest": ("tier_streaming", "TierStreamingRequest"),
    "TierStreamingCostModel": ("tier_streaming", "TierStreamingCostModel"),
    "TierStreamingExecutionPlan": (
        "tier_streaming",
        "TierStreamingExecutionPlan",
    ),
    "TierStreamingSchedule": ("tier_streaming", "TierStreamingSchedule"),
    "TierStreamingSegment": ("tier_streaming", "TierStreamingSegment"),
    "TierStreamingWave": ("tier_streaming", "TierStreamingWave"),
    "build_tier_streaming_schedule": (
        "tier_streaming",
        "build_tier_streaming_schedule",
    ),
    "plan_tier_streaming_execution": (
        "tier_streaming",
        "plan_tier_streaming_execution",
    ),
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
