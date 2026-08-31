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
    "INVALID_INDEX": ("runtime", "INVALID_INDEX"),
    "MAX_EVENT_COMPLETION_CLASSES": (
        "runtime",
        "MAX_EVENT_COMPLETION_CLASSES",
    ),
    "AcquireRequirement": ("runtime", "AcquireRequirement"),
    "AcquireRequirementFlag": ("runtime", "AcquireRequirementFlag"),
    "WorkTicketState": ("runtime", "WorkTicketState"),
    "DeviceWorkPlan": ("runtime", "DeviceWorkPlan"),
    "EpochStatus": ("runtime", "EpochStatus"),
    "JitOperatorModule": ("runtime", "JitOperatorModule"),
    "JitPhaseProgram": ("runtime", "JitPhaseProgram"),
    "IndexedHostObject": ("runtime", "IndexedHostObject"),
    "IndexedHostIndexBinding": ("runtime", "IndexedHostIndexBinding"),
    "copy_strided_host_runs_async": (
        "runtime",
        "copy_strided_host_runs_async",
    ),
    "NvmeCapabilities": ("runtime", "NvmeCapabilities"),
    "NvmeDmaTarget": ("runtime", "NvmeDmaTarget"),
    "NvmeHbmMappingBackend": ("runtime", "NvmeHbmMappingBackend"),
    "NvmeHbmMappingPolicy": ("runtime", "NvmeHbmMappingPolicy"),
    "NvmeHbmRegion": ("runtime", "NvmeHbmRegion"),
    "RegisteredNvmeObjectInstall": ("runtime", "RegisteredNvmeObjectInstall"),
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
    "NvmeHbmBackendRequirement": ("tier", "NvmeHbmBackendRequirement"),
    "ServingTierConfig": ("tier", "ServingTierConfig"),
    "ServingTierService": ("tier", "ServingTierService"),
    "PHYSICAL_SERVING_TIERS": ("tier", "PHYSICAL_SERVING_TIERS"),
    "TierPageCatalog": ("tier", "TierPageCatalog"),
    "PageExtent": ("tier", "PageExtent"),
    "ResourceCapability": ("resource_contract", "ResourceCapability"),
    "ResourceAddressSpace": ("resource_contract", "ResourceAddressSpace"),
    "ResourceContract": ("resource_contract", "ResourceContract"),
    "ResourceDataPath": ("resource_contract", "ResourceDataPath"),
    "ResourceKind": ("resource_contract", "ResourceKind"),
    "ResourceOwner": ("resource_contract", "ResourceOwner"),
    "ResourcePath": ("resource_contract", "ResourcePath"),
    "require_numerical_binding": (
        "resource_contract",
        "require_numerical_binding",
    ),
    "resource_contract": ("resource_contract", "resource_contract"),
    "tenant_budget_specs": ("tenant", "tenant_budget_specs"),
    "tenant_isolation_required": ("tenant", "tenant_isolation_required"),
    "tenant_mapper_from_environment": ("tenant", "tenant_mapper_from_environment"),
    "Placement": ("runtime", "Placement"),
    "Replica": ("runtime", "Replica"),
    "RequestRange": ("runtime", "RequestRange"),
    "RequestSpec": ("request_contract", "RequestSpec"),
    "RequestProgress": ("runtime", "RequestProgress"),
    "RequestProgressSnapshot": ("runtime", "RequestProgressSnapshot"),
    "Runtime": ("runtime", "Runtime"),
    "RuntimeConfig": ("runtime", "RuntimeConfig"),
    "RuntimeResourceConfig": ("runtime_resources", "RuntimeResourceConfig"),
    "ServingRuntimeResources": ("runtime_resources", "ServingRuntimeResources"),
    "RuntimeError": ("runtime", "RuntimeError"),
    "WorkItem": ("runtime", "WorkItem"),
    "WorkItemFlag": ("runtime", "WorkItemFlag"),
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
    "ProtocolKind": ("execution_protocol", "ProtocolKind"),
    "WorkLedger": ("execution_protocol", "WorkLedger"),
    "ExecutionPlan": ("execution_core", "ExecutionPlan"),
    "ExecutionSession": ("execution_core", "ExecutionSession"),
    "ExecutionTile": ("execution_core", "ExecutionTile"),
    "EngineBatch": ("adapters", "EngineBatch"),
    "ConsumerContract": ("adapters", "ConsumerContract"),
    "ConsumerKind": ("adapters", "ConsumerKind"),
    "ExactDemandProjection": ("adapters", "ExactDemandProjection"),
    "EngineBoundary": ("adapters", "EngineBoundary"),
    "SglangAdapter": ("adapters", "SglangAdapter"),
    "SglangExecutionConfig": ("adapters", "SglangExecutionConfig"),
    "VllmAdapter": ("adapters", "VllmAdapter"),
    "VllmSchedulerProjection": ("adapters", "VllmSchedulerProjection"),
    "SUPPORTED_VLLM_V1_VERSION": ("adapters", "SUPPORTED_VLLM_V1_VERSION"),
    "VllmV1Hook": ("adapters", "VllmV1Hook"),
    "VllmV1SchedulerProjection": ("adapters", "VllmV1SchedulerProjection"),
    "device_abi_version": ("runtime", "device_abi_version"),
    "copy_host_to_device_async": ("runtime", "copy_host_to_device_async"),
    "synchronize_stream": ("runtime", "synchronize_stream"),
    "FrontierState": ("progress_frontier", "FrontierState"),
    "RequestFrontier": ("progress_frontier", "RequestFrontier"),
    "RequestFrontierEntry": ("progress_frontier", "RequestFrontierEntry"),
    "build_request_frontier": ("progress_frontier", "build_request_frontier"),
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
    "mapped_request_bound_attention_jit_args": (
        "flashinfer",
        "mapped_request_bound_attention_jit_args",
    ),
    "enqueue_resident_attention": ("flashinfer", "enqueue_resident_attention"),
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
