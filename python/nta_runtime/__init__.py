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
    "NvmeDmaTarget": ("runtime", "NvmeDmaTarget"),
    "NvmeHbmMappingBackend": ("runtime", "NvmeHbmMappingBackend"),
    "CxlDaxCapabilities": ("runtime", "CxlDaxCapabilities"),
    "OperatorCapability": ("runtime", "OperatorCapability"),
    "OperatorAccessProof": ("runtime", "OperatorAccessProof"),
    "OperatorContract": ("runtime", "OperatorContract"),
    "OperatorCoordinateMap": ("runtime", "OperatorCoordinateMap"),
    "OperatorFamily": ("runtime", "OperatorFamily"),
    "OperatorForm": ("runtime", "OperatorForm"),
    "OperatorPartialState": ("runtime", "OperatorPartialState"),
    "OperatorPlan": ("runtime", "OperatorPlan"),
    "OperatorPlanFlag": ("runtime", "OperatorPlanFlag"),
    "OperatorReduction": ("runtime", "OperatorReduction"),
    "OperatorDemandBinding": ("runtime", "OperatorDemandBinding"),
    "OperatorIdentityBinding": ("runtime", "OperatorIdentityBinding"),
    "OperatorInstrumentation": ("runtime", "OperatorInstrumentation"),
    "require_operator_pair": ("runtime", "require_operator_pair"),
    "NvmeOptions": ("runtime", "NvmeOptions"),
    "NvmeQueueStats": ("runtime", "NvmeQueueStats"),
    "NvmeTransport": ("runtime", "NvmeTransport"),
    "CxlDaxOptions": ("runtime", "CxlDaxOptions"),
    "CxlDaxTransport": ("runtime", "CxlDaxTransport"),
    "TierCapability": ("runtime", "TierCapability"),
    "TierDescriptor": ("runtime", "TierDescriptor"),
    "TierKind": ("runtime", "TierKind"),
    "ServingTier": ("tier", "ServingTier"),
    "ServingTierConfig": ("tier", "ServingTierConfig"),
    "ServingTierService": ("tier", "ServingTierService"),
    "TierPageCatalog": ("tier", "TierPageCatalog"),
    "PageExtent": ("tier", "PageExtent"),
    "ResourceCapability": ("resource_contract", "ResourceCapability"),
    "ResourceContract": ("resource_contract", "ResourceContract"),
    "ResourceKind": ("resource_contract", "ResourceKind"),
    "ResourceOwner": ("resource_contract", "ResourceOwner"),
    "resource_contract": ("resource_contract", "resource_contract"),
    "Placement": ("runtime", "Placement"),
    "Replica": ("runtime", "Replica"),
    "RequestRange": ("runtime", "RequestRange"),
    "RequestSpec": ("runtime", "RequestSpec"),
    "RequestProgress": ("runtime", "RequestProgress"),
    "RequestProgressSnapshot": ("runtime", "RequestProgressSnapshot"),
    "Runtime": ("runtime", "Runtime"),
    "RuntimeConfig": ("runtime", "RuntimeConfig"),
    "RuntimeResourceConfig": ("runtime_resources", "RuntimeResourceConfig"),
    "ServingRuntimeResources": ("runtime_resources", "ServingRuntimeResources"),
    "RuntimeError": ("runtime", "RuntimeError"),
    "WorkItem": ("runtime", "WorkItem"),
    "Availability": ("work_unit", "Availability"),
    "DemandDescriptor": ("work_unit", "DemandDescriptor"),
    "DemandSemantics": ("work_unit", "DemandSemantics"),
    "Granularity": ("work_unit", "Granularity"),
    "WorkBatch": ("work_unit", "WorkBatch"),
    "WorkUnit": ("work_unit", "WorkUnit"),
    "ExecutionProtocolConfig": (
        "execution_protocol",
        "ExecutionProtocolConfig",
    ),
    "GranularityCostModel": ("execution_protocol", "GranularityCostModel"),
    "GranularityEstimate": ("execution_protocol", "GranularityEstimate"),
    "ProtocolKind": ("execution_protocol", "ProtocolKind"),
    "WorkLedger": ("execution_protocol", "WorkLedger"),
    "ExecutionSession": ("execution_core", "ExecutionSession"),
    "ExecutionTile": ("execution_core", "ExecutionTile"),
    "EngineBatch": ("adapters", "EngineBatch"),
    "ExactDemandProjection": ("adapters", "ExactDemandProjection"),
    "EngineBoundary": ("adapters", "EngineBoundary"),
    "SglangAdapter": ("adapters", "SglangAdapter"),
    "SglangExecutionConfig": ("adapters", "SglangExecutionConfig"),
    "VllmAdapter": ("adapters", "VllmAdapter"),
    "VllmSchedulerProjection": ("adapters", "VllmSchedulerProjection"),
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
    "BoundedStagingPool": ("bounded_staging", "BoundedStagingPool"),
    "StagingLease": ("bounded_staging", "StagingLease"),
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
    "DeviceDemandCostModel": ("execution_planner", "DeviceDemandCostModel"),
    "DeviceDemandPlan": ("execution_planner", "DeviceDemandPlan"),
    "plan_device_demand": ("execution_planner", "plan_device_demand"),
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
