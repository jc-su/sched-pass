"""Owning Python bindings for the NTA engine/runtime boundary."""

from __future__ import annotations

import builtins
import ctypes
import ctypes.util
import dataclasses
import enum
import hashlib
import os
import pathlib
import struct
from collections.abc import Iterable
from typing import Any

from .abi import bounded_integer as _bounded_integer
from .abi import u32 as _u32
from .abi import u64 as _u64
from .execution_topology import ExactWorkTopology, WorkDependencySpan
from .indexed_transfer import (
    AcquisitionTopology,
    IndexedTensorLane,
)
from .request_contract import RequestSpec, _RequestSpec
from .requests import RequestBinding
from .resource_contract import ResourceCapability, ResourceOwner


API_VERSION = 52
_INT32_MAX = (1 << 31) - 1


def _device_ordinal(value: int, name: str = "device_ordinal") -> int:
    return _bounded_integer(value, name, minimum=-1, maximum=_INT32_MAX)


class RuntimeError(builtins.RuntimeError):
    """An error returned by the native NTA runtime."""


class Placement(enum.IntEnum):
    HBM = 0
    HOST_MAPPED = 1
    HOST_STAGED = 2
    CXL_MAPPED = 3


class TierKind(enum.IntEnum):
    HBM = 0
    HOST_MAPPED = 1
    HOST_STAGED = 2
    NVME = 3
    CXL = 4
    RDMA = 5


class NvmeDmaTarget(enum.IntEnum):
    HBM_PEER = 0
    HOST_MAPPED = 1


class NvmeHbmMappingBackend(enum.IntEnum):
    UNAVAILABLE = 0
    NVIDIA_PEER_PAGES = 1
    CUDA_DMA_BUF_IOAS = 2

    @property
    def artifact_name(self) -> str:
        """Stable cross-language label used by qualification artifacts."""

        return {
            NvmeHbmMappingBackend.UNAVAILABLE: "unavailable",
            NvmeHbmMappingBackend.NVIDIA_PEER_PAGES: "nvidia-peer-pages",
            NvmeHbmMappingBackend.CUDA_DMA_BUF_IOAS: "cuda-dmabuf-ioas",
        }[self]


class NvmeHbmMappingPolicy(enum.IntEnum):
    """Fail-closed setup policy for one direct-HBM mapping lease."""

    AUTO = 0
    NVIDIA_PEER_PAGES = 1
    CUDA_DMA_BUF_IOAS = 2


# Keep the public ABI-facing name while using the same dependency-free typed
# contract as catalog validation and framework adapters.
TierCapability = ResourceCapability


class WorkTicketState(enum.IntEnum):
    NEW = 0
    PENDING = 1
    READY = 2
    DONE = 3
    CANCELLED = 4
    FAILED = 5
    INITIALIZING = 6


class OperatorFamily(enum.IntEnum):
    GENERIC = 0
    FLASHINFER_DECODE = 1
    FLASHINFER_PAGED_PREFILL = 2


class OperatorForm(enum.IntEnum):
    UNSPECIFIED = 0
    DIRECT = 1
    INCREMENTAL = 2


class OperatorCapability(enum.IntFlag):
    REQUEST_BINDING = 1 << 0
    OBJECT_DEPENDENCIES = 1 << 1
    FINITE_DEFERRAL = 1 << 2
    PARTIAL_PUBLICATION = 1 << 3
    COMPLETE_CONTRIBUTOR_MERGE = 1 << 4
    RUNNABLE_COMPACTION = 1 << 5
    GRAPH_REPLAY = 1 << 6
    TYPED_FLASHINFER_FRONTEND = 1 << 7
    PREACQUIRED_PARTIAL_ENTRY = 1 << 8
    STREAM_ORDERED_RETIREMENT = 1 << 9


class OperatorInstrumentation(enum.IntFlag):
    TYPED_ACCESS_LOWERING = 1 << 0
    EXACT_DEMAND = 1 << 1
    GENERATION_SAFE_IDENTITY = 1 << 2
    TIER_OWNERSHIP = 1 << 3


class OperatorIdentityBinding(enum.IntEnum):
    NONE = 0
    REQUEST_SLOT_GENERATION = 1


class OperatorDemandBinding(enum.IntEnum):
    NONE = 0
    EXACT_WORK_UNIT = 1


class OperatorAccessProof(enum.IntEnum):
    NONE = 0
    LOADED_INDEX_STRIDE = 1
    CP_ASYNC_GLOBAL = 2
    TYPED_FRONTEND = 3


class OperatorCoordinateMap(enum.IntEnum):
    UNSPECIFIED = 0
    FLASHINFER_REQUEST_CONTIGUOUS = 1


class OperatorPartialState(enum.IntEnum):
    NONE = 0
    ONLINE_SOFTMAX_VALUE_LSE = 1


class OperatorReduction(enum.IntEnum):
    NONE = 0
    ORDERED_MERGE_STATE = 1


class OperatorPlanFlag(enum.IntFlag):
    FIXED_CAPACITY = 1 << 0
    GRAPH_STABLE = 1 << 1
    EXTERNAL_WAVE_SOURCES = 1 << 2
    GENERATION_BOUND = 1 << 3
    EXACT_COMPLETE_MERGE = 1 << 4


class _RuntimeConfig(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("api_version", ctypes.c_uint32),
        ("request_capacity", ctypes.c_uint32),
        ("object_capacity", ctypes.c_uint32),
        ("intent_capacity", ctypes.c_uint32),
        ("work_ticket_capacity", ctypes.c_uint32),
        ("max_replicas_per_object", ctypes.c_uint32),
        ("max_dependencies_per_work_ticket", ctypes.c_uint32),
        ("device_ordinal", ctypes.c_int32),
        ("enable_cta_nvme_try_issue", ctypes.c_uint32),
        ("tenant_capacity", ctypes.c_uint32),
        ("staging_byte_capacity", ctypes.c_uint64),
    ]


class _OperatorContract(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("schema_version", ctypes.c_uint16),
        ("struct_bytes", ctypes.c_uint16),
        ("runtime_abi_version", ctypes.c_uint32),
        ("family", ctypes.c_uint32),
        ("form", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint64),
        ("source_fingerprint_low", ctypes.c_uint64),
        ("source_fingerprint_high", ctypes.c_uint64),
        ("instrumentation_flags", ctypes.c_uint64),
        ("identity_binding", ctypes.c_uint32),
        ("demand_binding", ctypes.c_uint32),
        ("access_proof", ctypes.c_uint32),
        ("granularity_bytes", ctypes.c_uint32),
        ("tier_mask", ctypes.c_uint64),
    ]


class _OperatorPlan(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("schema_version", ctypes.c_uint16),
        ("struct_bytes", ctypes.c_uint16),
        ("runtime_abi_version", ctypes.c_uint32),
        ("family", ctypes.c_uint32),
        ("supported_forms", ctypes.c_uint32),
        ("coordinate_map", ctypes.c_uint32),
        ("partial_state", ctypes.c_uint32),
        ("reduction", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("source_fingerprint_low", ctypes.c_uint64),
        ("source_fingerprint_high", ctypes.c_uint64),
        ("plan_fingerprint_low", ctypes.c_uint64),
        ("plan_fingerprint_high", ctypes.c_uint64),
    ]


@dataclasses.dataclass(frozen=True)
class OperatorContract:
    schema_version: int
    runtime_abi_version: int
    family: OperatorFamily
    form: OperatorForm
    capabilities: OperatorCapability
    source_fingerprint: str
    instrumentation_flags: OperatorInstrumentation = OperatorInstrumentation(0)
    identity_binding: OperatorIdentityBinding = OperatorIdentityBinding.NONE
    demand_binding: OperatorDemandBinding = OperatorDemandBinding.NONE
    access_proof: OperatorAccessProof = OperatorAccessProof.NONE
    granularity_bytes: int = 0
    tier_mask: int = 0

    def __post_init__(self) -> None:
        known_capabilities = (1 << 10) - 1
        known_instrumentation = (1 << 4) - 1
        if int(self.capabilities) & ~known_capabilities:
            raise ValueError("JIT operator contract names unknown capabilities")
        if int(self.instrumentation_flags) & ~known_instrumentation:
            raise ValueError(
                "JIT operator contract names unknown instrumentation flags"
            )
        known_tiers = (1 << 6) - 1
        if self.tier_mask < 0 or self.tier_mask & ~known_tiers:
            raise ValueError("JIT operator contract names an unknown tier")
        _u32(self.granularity_bytes, "JIT operator granularity")

    def require(
        self,
        *,
        family: OperatorFamily,
        form: OperatorForm,
        capabilities: OperatorCapability = OperatorCapability(0),
        instrumentation: OperatorInstrumentation = OperatorInstrumentation(0),
        identity_binding: OperatorIdentityBinding | None = None,
        demand_binding: OperatorDemandBinding | None = None,
        access_proof: OperatorAccessProof | None = None,
        tier_mask: int | None = None,
    ) -> None:
        if self.family != family or self.form != form:
            raise RuntimeError(
                f"JIT operator contract is {self.family.name}/{self.form.name}, "
                f"expected {family.name}/{form.name}"
            )
        missing = capabilities & ~self.capabilities
        if missing:
            raise RuntimeError(f"JIT operator contract lacks capabilities {missing!s}")
        missing_instrumentation = instrumentation & ~self.instrumentation_flags
        if missing_instrumentation:
            raise RuntimeError(
                "JIT operator contract lacks instrumentation guarantees "
                f"{missing_instrumentation!s}"
            )
        if identity_binding is not None and self.identity_binding != identity_binding:
            raise RuntimeError(
                "JIT operator contract has incompatible identity binding"
            )
        if demand_binding is not None and self.demand_binding != demand_binding:
            raise RuntimeError("JIT operator contract has incompatible demand binding")
        if access_proof is not None and self.access_proof != access_proof:
            raise RuntimeError("JIT operator contract has incompatible access proof")
        if tier_mask is not None and self.tier_mask & tier_mask != tier_mask:
            raise RuntimeError("JIT operator contract does not own the required tiers")


@dataclasses.dataclass(frozen=True)
class OperatorPlan:
    schema_version: int
    runtime_abi_version: int
    family: OperatorFamily
    supported_forms: int
    coordinate_map: OperatorCoordinateMap
    partial_state: OperatorPartialState
    reduction: OperatorReduction
    flags: OperatorPlanFlag
    source_fingerprint: str
    plan_fingerprint: str

    def __post_init__(self) -> None:
        valid_forms = (1 << int(OperatorForm.DIRECT)) | (
            1 << int(OperatorForm.INCREMENTAL)
        )
        valid_flags = (1 << 5) - 1
        if self.supported_forms & ~valid_forms:
            raise ValueError("JIT operator plan names unknown forms")
        if int(self.flags) & ~valid_flags:
            raise ValueError("JIT operator plan names unknown flags")

    def supports(self, form: OperatorForm) -> bool:
        return form != OperatorForm.UNSPECIFIED and bool(
            self.supported_forms & (1 << int(form))
        )

    def require(
        self,
        *,
        family: OperatorFamily,
        forms: tuple[OperatorForm, ...],
        coordinate_map: OperatorCoordinateMap,
        partial_state: OperatorPartialState,
        reduction: OperatorReduction,
        flags: OperatorPlanFlag = OperatorPlanFlag(0),
    ) -> None:
        if self.family != family or any(not self.supports(form) for form in forms):
            raise RuntimeError("JIT operator plan lacks the required family or forms")
        if (
            self.coordinate_map != coordinate_map
            or self.partial_state != partial_state
            or self.reduction != reduction
        ):
            raise RuntimeError("JIT operator plan has incompatible numerical semantics")
        missing = flags & ~self.flags
        if missing:
            raise RuntimeError(f"JIT operator plan lacks flags {missing!s}")


def require_operator_pair(
    direct: "JitOperatorModule", incremental: "JitOperatorModule"
) -> OperatorPlan:
    """Return the common typed plan or reject an independently generated pair."""

    direct_contract = direct.operator_contract
    incremental_contract = incremental.operator_contract
    if (
        direct_contract.form != OperatorForm.DIRECT
        or incremental_contract.form != OperatorForm.INCREMENTAL
        or direct_contract.family != incremental_contract.family
        or direct_contract.source_fingerprint != incremental_contract.source_fingerprint
        or direct.operator_plan != incremental.operator_plan
    ):
        raise RuntimeError(
            "JIT direct and incremental modules have incompatible operator plans"
        )
    return direct.operator_plan


class _Replica(ctypes.Structure):
    _fields_ = [
        ("source_device_address", ctypes.c_uint64),
        ("tensor_map_address", ctypes.c_uint64),
        ("estimated_latency_ns", ctypes.c_uint64),
        ("estimated_bandwidth_bytes_per_second", ctypes.c_uint64),
        ("placement", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
    ]


class AcquireRequirement(ctypes.Structure):
    _fields_ = [
        ("direct_base", ctypes.c_uint64),
        ("direct_tensor_map", ctypes.c_uint64),
        ("object_id", ctypes.c_uint64),
        ("offset", ctypes.c_uint64),
        ("object_slot", ctypes.c_uint32),
        ("object_version", ctypes.c_uint32),
        ("bytes", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
    ]


class _IndexedHostObject(ctypes.Structure):
    _fields_ = [
        ("object_id", ctypes.c_uint64),
        ("source_device_address", ctypes.c_uint64),
        ("staging_device_address", ctypes.c_uint64),
        ("source_indices_device_address", ctypes.c_uint64),
        ("staging_indices_device_address", ctypes.c_uint64),
        ("version", ctypes.c_uint32),
        ("index_count", ctypes.c_uint32),
        ("element_bytes", ctypes.c_uint32),
        ("source_stride_bytes", ctypes.c_uint32),
        ("staging_stride_bytes", ctypes.c_uint32),
        ("source_index_limit", ctypes.c_uint32),
        ("staging_index_limit", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


_INDEXED_HOST_OBJECT_PACKER = struct.Struct("@QQQQQIIIIIIII")
_ACQUIRE_REQUIREMENT_PACKER = struct.Struct("@QQQQIIII")
_WORK_ITEM_PACKER = struct.Struct("@IIIIIIIIIIIIQII")
_REQUEST_RANGE_PACKER = struct.Struct("@IIII")


class _IndexedHostIndexBinding(ctypes.Structure):
    _fields_ = [
        ("source_indices_device_address", ctypes.c_uint64),
        ("staging_indices_device_address", ctypes.c_uint64),
        ("index_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class _ContiguousCopyRun(ctypes.Structure):
    _fields_ = [
        ("source_first_row", ctypes.c_uint32),
        ("destination_first_row", ctypes.c_uint32),
        ("row_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class _StridedCopyGroup(ctypes.Structure):
    _fields_ = [
        ("source_address", ctypes.c_uint64),
        ("destination_address", ctypes.c_uint64),
        ("source_rows", ctypes.c_uint32),
        ("destination_rows", ctypes.c_uint32),
        ("row_bytes", ctypes.c_uint32),
        ("source_stride_bytes", ctypes.c_uint32),
        ("destination_stride_bytes", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class WorkItem(ctypes.Structure):
    _fields_ = [
        ("request_index", ctypes.c_uint32),
        ("request_slot", ctypes.c_uint32),
        ("generation", ctypes.c_uint32),
        ("logical_work", ctypes.c_uint32),
        ("dependency_begin", ctypes.c_uint32),
        ("dependency_count", ctypes.c_uint32),
        ("direct_dependency_count", ctypes.c_uint32),
        ("work_ticket", ctypes.c_uint32),
        ("reduction_group", ctypes.c_uint32),
        ("contributor_index", ctypes.c_uint32),
        ("contributor_count", ctypes.c_uint32),
        ("estimated_compute_ns", ctypes.c_uint32),
        ("ready_deadline_offset_ns", ctypes.c_uint64),
        ("reserved2", ctypes.c_uint32),
        ("reserved3", ctypes.c_uint32),
    ]


class RequestRange(ctypes.Structure):
    _fields_ = [
        ("work_begin", ctypes.c_uint32),
        ("work_count", ctypes.c_uint32),
        ("request_slot", ctypes.c_uint32),
        ("generation", ctypes.c_uint32),
    ]


class _NvmeOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("api_version", ctypes.c_uint32),
        ("endpoint", ctypes.c_char_p),
        ("device_ordinal", ctypes.c_int32),
        ("namespace_id", ctypes.c_uint32),
        ("queue_depth", ctypes.c_uint32),
        ("admin_timeout_ms", ctypes.c_uint32),
        ("media_policy", ctypes.c_uint32),
        ("dma_target", ctypes.c_uint32),
        ("hbm_mapping_policy", ctypes.c_uint32),
    ]


class _NvmeCapabilities(ctypes.Structure):
    _fields_ = [
        ("queue_depth", ctypes.c_uint32),
        ("controller_page_size", ctypes.c_uint32),
        ("lba_size", ctypes.c_uint32),
        ("max_transfer_bytes", ctypes.c_uint32),
        ("namespace_bytes", ctypes.c_uint64),
        ("queue_id", ctypes.c_uint32),
        ("queue_count", ctypes.c_uint32),
        ("device_ordinal", ctypes.c_int32),
        ("supports_hbm_peer_dma", ctypes.c_uint32),
        ("hbm_mapping_backend", ctypes.c_uint32),
        ("translated_iommu", ctypes.c_uint32),
        ("namespace_read_only", ctypes.c_uint32),
        ("gpu_doorbell_mapping_validated", ctypes.c_uint32),
    ]


class _NvmeQueueStats(ctypes.Structure):
    _fields_ = [
        ("submitted", ctypes.c_uint64),
        ("completed", ctypes.c_uint64),
        ("failed", ctypes.c_uint64),
        ("direct_submitted", ctypes.c_uint64),
        ("direct_fallbacks", ctypes.c_uint64),
        ("outstanding", ctypes.c_uint32),
        ("error", ctypes.c_uint32),
        ("sq_tail", ctypes.c_uint32),
        ("cq_head", ctypes.c_uint32),
        ("cq_phase", ctypes.c_uint32),
        ("next_completion_dword3", ctypes.c_uint32),
        ("hbm_region_registrations", ctypes.c_uint64),
        ("hbm_region_bytes", ctypes.c_uint64),
        ("hbm_transfer_views", ctypes.c_uint64),
    ]


class _NvmeHbmRegistrationRange(ctypes.Structure):
    _fields_ = [
        ("allocation_address", ctypes.c_uint64),
        ("allocation_bytes", ctypes.c_uint64),
        ("registration_address", ctypes.c_uint64),
        ("registration_bytes", ctypes.c_uint64),
    ]


class _RegisteredNvmeObject(ctypes.Structure):
    _fields_ = [
        ("object_id", ctypes.c_uint64),
        ("source_byte_offset", ctypes.c_uint64),
        ("bytes", ctypes.c_uint64),
        ("region", ctypes.c_void_p),
        ("destination_device_address", ctypes.c_uint64),
        ("prior_consumer_event", ctypes.c_uint64),
        ("slot", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
    ]


_REGISTERED_NVME_OBJECT_PACKER = struct.Struct("@QQQPQQII")


class _CxlOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("api_version", ctypes.c_uint32),
        ("endpoint", ctypes.c_char_p),
        ("window_bytes", ctypes.c_uint64),
        ("device_ordinal", ctypes.c_int32),
    ]


class _CxlCapabilities(ctypes.Structure):
    _fields_ = [
        ("window_bytes", ctypes.c_uint64),
        ("mapped_device_address", ctypes.c_uint64),
        ("device_ordinal", ctypes.c_int32),
        ("host_registered", ctypes.c_uint32),
        ("direct_device_visible", ctypes.c_uint32),
    ]


class _TierDescriptor(ctypes.Structure):
    _fields_ = [
        ("source_kind", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("device_state", ctypes.c_uint64),
        ("estimated_latency_ns", ctypes.c_uint64),
        ("estimated_bandwidth_bytes_per_second", ctypes.c_uint64),
        ("active", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("protocol_owner", ctypes.c_uint32),
        ("payload_owner", ctypes.c_uint32),
        ("transfer_destination_owner", ctypes.c_uint32),
        ("mapping_owner", ctypes.c_uint32),
        ("directory_owner", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class _EpochStatus(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_uint32),
        ("fresh", ctypes.c_uint32),
        ("pending", ctypes.c_uint32),
        ("ready", ctypes.c_uint32),
        ("done", ctypes.c_uint32),
        ("cancelled", ctypes.c_uint32),
        ("failed", ctypes.c_uint32),
        ("initializing", ctypes.c_uint32),
    ]


class _RequestProgress(ctypes.Structure):
    _fields_ = [
        ("request_id", ctypes.c_uint64),
        ("generation", ctypes.c_uint32),
        ("expected_work", ctypes.c_uint32),
        ("pending_work", ctypes.c_uint32),
        ("runnable_work", ctypes.c_uint32),
        ("completed_work", ctypes.c_uint32),
        ("failed_work", ctypes.c_uint32),
        ("cancelled_work", ctypes.c_uint32),
        ("epoch", ctypes.c_uint32),
        ("unavailable_bytes", ctypes.c_uint64),
        ("runnable_compute_ns", ctypes.c_uint64),
        ("completed_compute_ns", ctypes.c_uint64),
        ("pending_compute_ns", ctypes.c_uint64),
        ("expected_compute_ns", ctypes.c_uint64),
        ("dropped_attributions", ctypes.c_uint64),
        ("reserved", ctypes.c_uint64),
    ]


def _validate_abi_layouts() -> None:
    layouts = (
        ("OperatorContract", ctypes.sizeof(_OperatorContract), 80),
        ("TierDescriptor", ctypes.sizeof(_TierDescriptor), 64),
        ("OperatorPlan", ctypes.sizeof(_OperatorPlan), 72),
        ("AcquireRequirement", ctypes.sizeof(AcquireRequirement), 48),
        ("IndexedHostObject", ctypes.sizeof(_IndexedHostObject), 72),
        ("WorkItem", ctypes.sizeof(WorkItem), 64),
        ("RequestProgress", ctypes.sizeof(_RequestProgress), 96),
        ("RequestSpec", ctypes.sizeof(_RequestSpec), 40),
        ("RegisteredNvmeObject", ctypes.sizeof(_RegisteredNvmeObject), 56),
    )
    invalid = [
        f"{name}={observed} (expected {expected})"
        for name, observed, expected in layouts
        if observed != expected
    ]
    if invalid:
        raise RuntimeError("Python/native ABI layout mismatch: " + ", ".join(invalid))
    packed_layouts = (
        (_ACQUIRE_REQUIREMENT_PACKER, AcquireRequirement),
        (_INDEXED_HOST_OBJECT_PACKER, _IndexedHostObject),
        (_WORK_ITEM_PACKER, WorkItem),
        (_REQUEST_RANGE_PACKER, RequestRange),
        (_REGISTERED_NVME_OBJECT_PACKER, _RegisteredNvmeObject),
    )
    if any(packer.size != ctypes.sizeof(native) for packer, native in packed_layouts):
        raise RuntimeError("Python packed/native ABI layout mismatch")


_validate_abi_layouts()


@dataclasses.dataclass(frozen=True)
class RuntimeConfig:
    request_capacity: int
    object_capacity: int
    intent_capacity: int
    work_ticket_capacity: int
    max_replicas_per_object: int = 1
    max_dependencies_per_work_ticket: int = 8
    device_ordinal: int = -1
    enable_cta_nvme_try_issue: bool = False
    tenant_capacity: int = 0
    staging_byte_capacity: int = (1 << 64) - 1

    def __post_init__(self) -> None:
        for name in (
            "request_capacity",
            "object_capacity",
            "intent_capacity",
            "work_ticket_capacity",
            "max_replicas_per_object",
            "max_dependencies_per_work_ticket",
        ):
            object.__setattr__(
                self,
                name,
                _u32(getattr(self, name), name, positive=True),
            )
        object.__setattr__(self, "device_ordinal", _device_ordinal(self.device_ordinal))
        if type(self.enable_cta_nvme_try_issue) is not bool:
            raise ValueError("enable_cta_nvme_try_issue must be boolean")
        tenant_capacity = _u32(self.tenant_capacity, "tenant_capacity")
        staging_capacity = _u64(self.staging_byte_capacity, "staging_byte_capacity")
        # Mirror native normalization in the immutable Python image.  Policy
        # telemetry and callers must never observe zero while HostRuntime is
        # actually enforcing request-capacity tenants or an unlimited budget.
        object.__setattr__(
            self,
            "tenant_capacity",
            tenant_capacity or self.request_capacity,
        )
        object.__setattr__(
            self,
            "staging_byte_capacity",
            staging_capacity or (1 << 64) - 1,
        )

    def native(self) -> _RuntimeConfig:
        return _RuntimeConfig(
            ctypes.sizeof(_RuntimeConfig),
            API_VERSION,
            self.request_capacity,
            self.object_capacity,
            self.intent_capacity,
            self.work_ticket_capacity,
            self.max_replicas_per_object,
            self.max_dependencies_per_work_ticket,
            self.device_ordinal,
            int(self.enable_cta_nvme_try_issue),
            self.tenant_capacity,
            self.staging_byte_capacity,
        )


@dataclasses.dataclass(frozen=True)
class Replica:
    source_device_address: int
    placement: Placement
    tensor_map_address: int = 0
    estimated_latency_ns: int = 0
    estimated_bandwidth_bytes_per_second: int = 0

    def __post_init__(self) -> None:
        _u64(self.source_device_address, "replica source address", positive=True)
        _u64(self.tensor_map_address, "replica tensor-map address")
        _u64(self.estimated_latency_ns, "replica latency")
        _u64(
            self.estimated_bandwidth_bytes_per_second,
            "replica bandwidth",
        )
        try:
            Placement(self.placement)
        except (TypeError, ValueError) as error:
            raise ValueError("replica placement is invalid") from error

    def native(self) -> _Replica:
        return _Replica(
            self.source_device_address,
            self.tensor_map_address,
            self.estimated_latency_ns,
            self.estimated_bandwidth_bytes_per_second,
            int(self.placement),
            0,
        )


@dataclasses.dataclass(frozen=True)
class IndexedHostObject:
    object_id: int
    version: int
    source_device_address: int
    staging_device_address: int
    source_indices_device_address: int
    staging_indices_device_address: int
    index_count: int
    element_bytes: int
    source_stride_bytes: int
    staging_stride_bytes: int
    source_index_limit: int
    staging_index_limit: int

    def __post_init__(self) -> None:
        _u64(self.object_id, "indexed object id")
        _u32(self.version, "indexed object version")
        for name in (
            "source_device_address",
            "staging_device_address",
            "source_indices_device_address",
            "staging_indices_device_address",
        ):
            _u64(getattr(self, name), f"indexed {name}", positive=True)
        for name in (
            "index_count",
            "element_bytes",
            "source_stride_bytes",
            "staging_stride_bytes",
            "source_index_limit",
            "staging_index_limit",
        ):
            _u32(getattr(self, name), f"indexed {name}", positive=True)

    def native(self) -> _IndexedHostObject:
        return _IndexedHostObject(
            self.object_id,
            self.source_device_address,
            self.staging_device_address,
            self.source_indices_device_address,
            self.staging_indices_device_address,
            self.version,
            self.index_count,
            self.element_bytes,
            self.source_stride_bytes,
            self.staging_stride_bytes,
            self.source_index_limit,
            self.staging_index_limit,
            0,
        )


@dataclasses.dataclass(frozen=True)
class IndexedHostIndexBinding:
    source_indices_device_address: int
    staging_indices_device_address: int
    index_count: int

    def __post_init__(self) -> None:
        _u64(
            self.source_indices_device_address,
            "bound indexed source address",
            positive=True,
        )
        _u64(
            self.staging_indices_device_address,
            "bound indexed staging address",
            positive=True,
        )
        _u32(self.index_count, "bound indexed count", positive=True)

    def native(self) -> _IndexedHostIndexBinding:
        return _IndexedHostIndexBinding(
            self.source_indices_device_address,
            self.staging_indices_device_address,
            self.index_count,
            0,
        )


class IndexedAcquisitionPlan:
    """Native-ready directory and dependency image for acquisition groups.

    Framework adapters publish exact groups and lane geometry. This owner
    expands their Cartesian product directly into ctypes arrays, avoiding a
    per-object Python dataclass graph and retaining one immutable image for the
    corresponding work-plan upload. ``object_id_base`` plus ``object_version``
    gives every group/lane pair its resource identity. A physical group may
    serve several request generations (for example a vLLM shared-prefix
    block), so transfer ownership belongs to the resource/lease while
    ``group_consumers`` records every authorized request binding. Numerical
    WorkTickets retain the per-consumer slot and generation.
    """

    def __init__(
        self,
        topology: AcquisitionTopology,
        lanes: Iterable[IndexedTensorLane],
        *,
        work_bindings: Iterable[RequestBinding],
        source_indices_device_address: int,
        staging_indices_device_address: int,
        object_version: int,
        direct_base: int,
        first_slot: int = 0,
        object_id_base: int = 0x4E54410000000000,
    ) -> None:
        if not isinstance(topology, AcquisitionTopology):
            raise TypeError("indexed host plan requires AcquisitionTopology")
        lane_values = tuple(lanes)
        if not lane_values or any(
            not isinstance(lane, IndexedTensorLane) for lane in lane_values
        ):
            raise TypeError("indexed host plan requires typed tensor lanes")
        binding_values = tuple(work_bindings)
        if len(binding_values) != topology.work_count or any(
            not isinstance(binding, RequestBinding) for binding in binding_values
        ):
            raise ValueError("indexed work bindings do not match the topology")
        group_consumers: list[list[RequestBinding]] = [[] for _ in topology.groups]
        for binding, dependencies in zip(
            binding_values, topology.dependencies_by_work, strict=True
        ):
            for dependency in dependencies:
                consumers = group_consumers[dependency.group_index]
                if binding not in consumers:
                    consumers.append(binding)
        if any(not consumers for consumers in group_consumers):
            raise ValueError("indexed acquisition group has no request consumer")
        self.topology = topology
        self.lanes = lane_values
        self.work_bindings = binding_values
        self.group_consumers = tuple(tuple(consumers) for consumers in group_consumers)
        self.group_tenant_ids = tuple(
            tuple(sorted({consumer.tenant_id for consumer in consumers}))
            for consumers in self.group_consumers
        )
        self.first_slot = _u32(first_slot, "first indexed object slot")
        self.object_version = _u32(
            object_version, "indexed object version", positive=True
        )
        self.object_id_base = _u64(object_id_base, "indexed object ID base")
        self.direct_base = _u64(direct_base, "direct runtime base", positive=True)
        source_indices = _u64(
            source_indices_device_address,
            "source indices device address",
            positive=True,
        )
        staging_indices = _u64(
            staging_indices_device_address,
            "staging indices device address",
            positive=True,
        )
        if source_indices % 4 or staging_indices % 4:
            raise ValueError("indexed host plan index vectors must be int32-aligned")

        lane_count = len(lane_values)
        object_count = len(topology.groups) * lane_count
        if object_count == 0:
            raise ValueError("indexed host plan has no external transfer group")
        if self.first_slot + object_count >= 1 << 32:
            raise ValueError("indexed host plan object slots exceed uint32")
        if self.object_id_base + object_count > (1 << 64) - 1:
            raise ValueError("indexed host plan object IDs exceed uint64")
        self._objects = (_IndexedHostObject * object_count)()
        object_cursor = 0
        for group_index, group in enumerate(topology.groups):
            index_byte_offset = group.index_offset * 4
            for lane_index, lane in enumerate(lane_values):
                relative_slot = group_index * lane_count + lane_index
                _INDEXED_HOST_OBJECT_PACKER.pack_into(
                    self._objects,
                    object_cursor * _INDEXED_HOST_OBJECT_PACKER.size,
                    self.object_id_base + relative_slot,
                    lane.source_address,
                    lane.staging_address,
                    source_indices + index_byte_offset,
                    staging_indices + index_byte_offset,
                    self.object_version,
                    group.row_count,
                    lane.element_bytes,
                    lane.source_stride_bytes,
                    lane.staging_stride_bytes,
                    lane.source_index_limit,
                    lane.staging_index_limit,
                    0,
                )
                object_cursor += 1
        if object_cursor != object_count:
            raise RuntimeError("indexed host object materialization diverged")

        dependency_count = sum(
            lane_count * max(1, len(work)) for work in topology.dependencies_by_work
        )
        self._dependencies = (AcquireRequirement * dependency_count)()
        spans: list[WorkDependencySpan] = []
        external_slots: list[tuple[int, ...]] = []
        dependency_cursor = 0
        unresolved_counts: list[int] = []
        for work in topology.dependencies_by_work:
            begin = dependency_cursor
            work_slots: list[int] = []
            if not work:
                for _lane in lane_values:
                    _ACQUIRE_REQUIREMENT_PACKER.pack_into(
                        self._dependencies,
                        dependency_cursor * _ACQUIRE_REQUIREMENT_PACKER.size,
                        self.direct_base,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                    )
                    dependency_cursor += 1
                direct_count = lane_count
            else:
                for dependency in work:
                    group = topology.groups[dependency.group_index]
                    for lane_index, lane in enumerate(lane_values):
                        relative_slot = dependency.group_index * lane_count + lane_index
                        slot = self.first_slot + relative_slot
                        # A dependency names readiness of the shared transfer
                        # group, not a private copy of this work item's slice.
                        # The first claimant must therefore issue the complete
                        # ObjectEntry.  ``AcquisitionSlice`` remains the exact
                        # numerical coverage proof; every fan-out consumer
                        # waits for the containing group's single publication.
                        required_bytes = group.row_count * lane.element_bytes
                        if required_bytes >= 1 << 32:
                            raise ValueError(
                                "indexed work dependency bytes exceed uint32"
                            )
                        _ACQUIRE_REQUIREMENT_PACKER.pack_into(
                            self._dependencies,
                            dependency_cursor * _ACQUIRE_REQUIREMENT_PACKER.size,
                            0,
                            0,
                            self.object_id_base + relative_slot,
                            0,
                            slot,
                            self.object_version,
                            required_bytes,
                            0,
                        )
                        dependency_cursor += 1
                        work_slots.append(slot)
                direct_count = 0
                unresolved_counts.append(dependency_cursor - begin)
            spans.append(
                WorkDependencySpan(begin, dependency_cursor - begin, direct_count)
            )
            external_slots.append(tuple(work_slots))
        if dependency_cursor != dependency_count:
            raise RuntimeError("indexed host dependency materialization diverged")
        self.dependency_spans = tuple(spans)
        self.external_object_slots = tuple(external_slots)
        self.min_unresolved_dependencies = min(unresolved_counts, default=1)

    def require_single_tenant_groups(self) -> None:
        """Reject a shared transfer whose byte-credit owner is ambiguous.

        The device reserves one tenant credit for one physical acquisition.
        Sharing that acquisition across requests in the same tenant is exact
        and charged once.  A cross-tenant group would make the first CTA to
        discover it choose the charged tenant nondeterministically, so finite
        tenant isolation must reject such a topology before publication.
        """

        ambiguous = tuple(
            group_index
            for group_index, tenant_ids in enumerate(self.group_tenant_ids)
            if len(tenant_ids) != 1
        )
        if ambiguous:
            raise ValueError(
                "indexed acquisition groups cross tenant credit domains: "
                f"{ambiguous[:16]}"
            )

    @property
    def object_count(self) -> int:
        return len(self._objects)

    @property
    def dependencies(self) -> Any:
        return self._dependencies

    @property
    def native_objects(self) -> Any:
        return self._objects

    @property
    def transfer_bytes(self) -> int:
        lane_bytes = sum(lane.element_bytes for lane in self.lanes)
        return sum(group.row_count * lane_bytes for group in self.topology.groups)

    @property
    def object_transfer_bytes(self) -> tuple[int, ...]:
        """Physical payload owned by each directory object, in slot order.

        Indexed objects are emitted group-major and lane-minor.  Keeping the
        same exact byte vector beside the native image lets a consumer split a
        lookahead prefix from the remaining demand transfer without pretending
        that an already-completed wave was copied a second time.
        """

        return tuple(
            group.row_count * lane.element_bytes
            for group in self.topology.groups
            for lane in self.lanes
        )

    @property
    def max_object_fanout(self) -> int:
        return self.topology.max_group_fanout

    @property
    def direct_work_count(self) -> int:
        return self.topology.direct_work_count


@dataclasses.dataclass(frozen=True)
class NvmeOptions:
    endpoint: str
    device_ordinal: int = -1
    namespace_id: int = 1
    queue_depth: int = 64
    admin_timeout_ms: int = 10_000
    trust_read_only_device_code: bool = False
    dma_target: NvmeDmaTarget = NvmeDmaTarget.HBM_PEER
    hbm_mapping_policy: NvmeHbmMappingPolicy = NvmeHbmMappingPolicy.AUTO

    def __post_init__(self) -> None:
        if not self.endpoint:
            raise ValueError("NVMe endpoint must be explicit")
        _device_ordinal(self.device_ordinal)
        _u32(self.namespace_id, "NVMe namespace", positive=True)
        _u32(self.queue_depth, "NVMe queue depth", positive=True)
        _u32(self.admin_timeout_ms, "NVMe admin timeout", positive=True)
        if not isinstance(self.trust_read_only_device_code, bool):
            raise ValueError("NVMe read-only trust policy must be boolean")
        try:
            dma_target = NvmeDmaTarget(self.dma_target)
        except (TypeError, ValueError) as error:
            raise ValueError("NVMe DMA target is invalid") from error
        try:
            mapping_policy = NvmeHbmMappingPolicy(self.hbm_mapping_policy)
        except (TypeError, ValueError) as error:
            raise ValueError("NVMe HBM mapping policy is invalid") from error
        if (
            dma_target is not NvmeDmaTarget.HBM_PEER
            and mapping_policy is not NvmeHbmMappingPolicy.AUTO
        ):
            raise ValueError(
                "an explicit NVMe HBM mapping policy requires HBM peer DMA"
            )
        object.__setattr__(self, "dma_target", dma_target)
        object.__setattr__(self, "hbm_mapping_policy", mapping_policy)


@dataclasses.dataclass(frozen=True)
class NvmeCapabilities:
    queue_depth: int
    controller_page_size: int
    lba_size: int
    max_transfer_bytes: int
    namespace_bytes: int
    queue_id: int
    queue_count: int
    device_ordinal: int
    supports_hbm_peer_dma: bool
    hbm_mapping_backend: NvmeHbmMappingBackend
    translated_iommu: bool
    namespace_read_only: bool
    gpu_doorbell_mapping_validated: bool


@dataclasses.dataclass(frozen=True)
class NvmeQueueStats:
    submitted: int
    completed: int
    failed: int
    direct_submitted: int
    direct_fallbacks: int
    outstanding: int
    error: int
    sq_tail: int
    cq_head: int
    cq_phase: int
    next_completion_dword3: int
    hbm_region_registrations: int
    hbm_region_bytes: int
    hbm_transfer_views: int


@dataclasses.dataclass(frozen=True)
class NvmeHbmRegistrationRange:
    """Validated allocation identity and minimal peer-registration envelope."""

    allocation_address: int
    allocation_bytes: int
    registration_address: int
    registration_bytes: int

    def __post_init__(self) -> None:
        for name in (
            "allocation_address",
            "allocation_bytes",
            "registration_address",
            "registration_bytes",
        ):
            _u64(getattr(self, name), f"NVMe HBM {name}", positive=True)
        allocation_end = self.allocation_address + self.allocation_bytes
        registration_end = self.registration_address + self.registration_bytes
        if allocation_end >= 1 << 64 or registration_end >= 1 << 64:
            raise ValueError("NVMe HBM registration geometry exceeds uint64")
        if not (
            self.allocation_address <= self.registration_address
            and registration_end <= allocation_end
        ):
            raise ValueError("NVMe HBM registration exceeds its CUDA allocation")


@dataclasses.dataclass(frozen=True)
class CxlDaxOptions:
    endpoint: str
    window_bytes: int
    device_ordinal: int = -1

    def __post_init__(self) -> None:
        if not self.endpoint:
            raise ValueError("CXL DAX endpoint must be explicit")
        _u64(self.window_bytes, "CXL DAX window_bytes", positive=True)
        _device_ordinal(self.device_ordinal, "CXL DAX device_ordinal")


@dataclasses.dataclass(frozen=True)
class CxlDaxCapabilities:
    window_bytes: int
    mapped_device_address: int
    device_ordinal: int
    host_registered: bool
    direct_device_visible: bool


@dataclasses.dataclass(frozen=True)
class TierDescriptor:
    source_kind: TierKind
    capabilities: TierCapability
    device_state: int
    estimated_latency_ns: int
    estimated_bandwidth_bytes_per_second: int
    active: bool
    flags: int
    protocol_owner: ResourceOwner
    payload_owner: ResourceOwner
    transfer_destination_owner: ResourceOwner | None
    mapping_owner: ResourceOwner | None
    directory_owner: ResourceOwner

    def __post_init__(self) -> None:
        if not isinstance(self.protocol_owner, ResourceOwner):
            raise TypeError("tier protocol owner is not typed")
        if not isinstance(self.payload_owner, ResourceOwner):
            raise TypeError("tier payload owner is not typed")
        if self.transfer_destination_owner is not None and not isinstance(
            self.transfer_destination_owner, ResourceOwner
        ):
            raise TypeError("tier transfer destination owner is not typed")
        if self.mapping_owner is not None and not isinstance(
            self.mapping_owner, ResourceOwner
        ):
            raise TypeError("tier mapping owner is not typed")
        if not isinstance(self.directory_owner, ResourceOwner):
            raise TypeError("tier directory owner is not typed")


@dataclasses.dataclass(frozen=True)
class EpochStatus:
    total: int
    fresh: int
    pending: int
    ready: int
    done: int
    cancelled: int
    failed: int
    initializing: int

    @property
    def succeeded(self) -> bool:
        return self.done == self.total

    @property
    def has_failure(self) -> bool:
        return self.cancelled != 0 or self.failed != 0


@dataclasses.dataclass(frozen=True)
class RequestProgress:
    request_id: int
    generation: int
    expected_work: int
    pending_work: int
    runnable_work: int
    completed_work: int
    failed_work: int
    cancelled_work: int
    epoch: int
    unavailable_bytes: int
    runnable_compute_ns: int
    completed_compute_ns: int
    pending_compute_ns: int
    expected_compute_ns: int
    dropped_attributions: int

    @property
    def complete(self) -> bool:
        return (
            self.expected_work != 0
            and self.completed_work == self.expected_work
            and self.failed_work == 0
            and self.cancelled_work == 0
        )

    @property
    def remaining_compute_ns(self) -> int:
        return max(0, self.expected_compute_ns - self.completed_compute_ns)


def _request_progress_value(value: _RequestProgress) -> RequestProgress:
    return RequestProgress(
        value.request_id,
        value.generation,
        value.expected_work,
        value.pending_work,
        value.runnable_work,
        value.completed_work,
        value.failed_work,
        value.cancelled_work,
        value.epoch,
        value.unavailable_bytes,
        value.runnable_compute_ns,
        value.completed_compute_ns,
        value.pending_compute_ns,
        value.expected_compute_ns,
        value.dropped_attributions,
    )


def _library_candidates() -> Iterable[str]:
    configured = os.environ.get("NTA_RUNTIME_LIBRARY")
    if configured:
        yield configured
    root = pathlib.Path(__file__).resolve().parents[2]
    for build in ("build", "build-release", "build-local"):
        yield str(root / build / "libnta-runtime.so")
    discovered = ctypes.util.find_library("nta-runtime")
    if discovered:
        yield discovered


def _load() -> ctypes.CDLL:
    errors: list[str] = []
    for candidate in _library_candidates():
        try:
            return ctypes.CDLL(candidate)
        except OSError as error:
            errors.append(f"{candidate}: {error}")
    raise RuntimeError("cannot load libnta-runtime.so; " + "; ".join(errors))


_lib = _load()
_Handle = ctypes.c_void_p
_HandlePointer = ctypes.POINTER(_Handle)


def _function(name: str, restype: Any, *argtypes: Any) -> Any:
    function = getattr(_lib, name)
    function.restype = restype
    function.argtypes = list(argtypes)
    return function


_last_error = _function("nta_last_error", ctypes.c_char_p)
_api_version = _function("nta_runtime_c_api_version", ctypes.c_uint32)
_device_abi_version = _function("nta_runtime_device_abi_version", ctypes.c_uint32)
_nvme_create = _function(
    "nta_nvme_transport_create",
    ctypes.c_int,
    ctypes.POINTER(_NvmeOptions),
    _HandlePointer,
)
_nvme_destroy = _function("nta_nvme_transport_destroy", None, _Handle)
_nvme_capabilities = _function(
    "nta_nvme_transport_get_capabilities",
    ctypes.c_int,
    _Handle,
    ctypes.POINTER(_NvmeCapabilities),
)
_nvme_stats = _function(
    "nta_nvme_transport_read_stats",
    ctypes.c_int,
    _Handle,
    ctypes.POINTER(_NvmeQueueStats),
)
_nvme_describe_hbm_region = _function(
    "nta_nvme_transport_describe_hbm_region",
    ctypes.c_int,
    _Handle,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.POINTER(_NvmeHbmRegistrationRange),
)
_nvme_register_hbm_region = _function(
    "nta_nvme_transport_register_hbm_region",
    ctypes.c_int,
    _Handle,
    ctypes.c_uint64,
    ctypes.c_uint64,
    _HandlePointer,
)
_nvme_hbm_region_destroy = _function("nta_nvme_hbm_region_destroy", None, _Handle)
_cxl_create = _function(
    "nta_cxl_dax_transport_create",
    ctypes.c_int,
    ctypes.POINTER(_CxlOptions),
    _HandlePointer,
)
_cxl_destroy = _function("nta_cxl_dax_transport_destroy", None, _Handle)
_cxl_capabilities = _function(
    "nta_cxl_dax_transport_get_capabilities",
    ctypes.c_int,
    _Handle,
    ctypes.POINTER(_CxlCapabilities),
)
_runtime_tier_descriptor = _function(
    "nta_runtime_get_tier_descriptor",
    ctypes.c_int,
    _Handle,
    ctypes.c_uint32,
    ctypes.POINTER(_TierDescriptor),
)
_runtime_create = _function(
    "nta_runtime_create",
    ctypes.c_int,
    ctypes.POINTER(_RuntimeConfig),
    _Handle,
    _Handle,
    _HandlePointer,
)
_runtime_destroy = _function("nta_runtime_destroy", None, _Handle)
_runtime_set_request = _function(
    "nta_runtime_set_request",
    ctypes.c_int,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint64,
)
_runtime_publish_requests_async = _function(
    "nta_runtime_publish_requests_async",
    ctypes.c_int,
    _Handle,
    ctypes.POINTER(_RequestSpec),
    ctypes.c_uint32,
    ctypes.c_uint64,
)
_runtime_cancel_request = _function(
    "nta_runtime_cancel_request",
    ctypes.c_int,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint32,
)
_runtime_set_tenant_budget = _function(
    "nta_runtime_set_tenant_budget",
    ctypes.c_int,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint64,
)
_runtime_register_object = _function(
    "nta_runtime_register_object",
    ctypes.c_int,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.POINTER(_Replica),
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint64),
)
_runtime_register_indexed_host_object = _function(
    "nta_runtime_register_indexed_host_object",
    ctypes.c_int,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
)
_runtime_register_indexed_host_objects = _function(
    "nta_runtime_register_indexed_host_objects",
    ctypes.c_int,
    _Handle,
    ctypes.c_uint32,
    ctypes.POINTER(_IndexedHostObject),
    ctypes.c_uint32,
)
_runtime_register_indexed_host_objects_async = _function(
    "nta_runtime_register_indexed_host_objects_async",
    ctypes.c_int,
    _Handle,
    ctypes.c_uint32,
    ctypes.POINTER(_IndexedHostObject),
    ctypes.c_uint32,
    ctypes.c_uint64,
)
_runtime_register_indexed_host_objects_async_quiesced = _function(
    "nta_runtime_register_indexed_host_objects_async_quiesced",
    ctypes.c_int,
    _Handle,
    ctypes.c_uint32,
    ctypes.POINTER(_IndexedHostObject),
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint64,
)
_runtime_register_indexed_host_objects_async_bound = _function(
    "nta_runtime_register_indexed_host_objects_async_bound",
    ctypes.c_int,
    _Handle,
    ctypes.c_uint32,
    ctypes.POINTER(_IndexedHostObject),
    ctypes.c_uint32,
    ctypes.POINTER(_IndexedHostIndexBinding),
    ctypes.c_uint64,
    ctypes.c_uint64,
)
_runtime_bind_tensor_maps = _function(
    "nta_runtime_bind_tensor_maps",
    ctypes.c_int,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint64,
)
_runtime_install_nvme_object = _function(
    "nta_runtime_install_nvme_object",
    ctypes.c_int,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_uint64),
)
_runtime_install_nvme_object_async = _function(
    "nta_runtime_install_nvme_object_async",
    ctypes.c_int,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_uint64),
)
_runtime_install_registered_nvme_object = _function(
    "nta_runtime_install_registered_nvme_object",
    ctypes.c_int,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint64,
    _Handle,
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_uint64),
)
_runtime_install_registered_nvme_object_async = _function(
    "nta_runtime_install_registered_nvme_object_async",
    ctypes.c_int,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint64,
    _Handle,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_uint64),
)
_runtime_install_registered_nvme_objects_async = _function(
    "nta_runtime_install_registered_nvme_objects_async",
    ctypes.c_int,
    _Handle,
    ctypes.POINTER(_RegisteredNvmeObject),
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_uint64),
)
_runtime_pending_count = _function(
    "nta_runtime_read_pending_count",
    ctypes.c_int,
    _Handle,
    ctypes.POINTER(ctypes.c_uint32),
)
_runtime_epoch_status = _function(
    "nta_runtime_read_epoch_status",
    ctypes.c_int,
    _Handle,
    ctypes.c_uint32,
    ctypes.POINTER(_EpochStatus),
)
_runtime_sticky_failed_count = _function(
    "nta_runtime_read_sticky_failed_count",
    ctypes.c_int,
    _Handle,
    ctypes.POINTER(ctypes.c_uint32),
)
_runtime_request_progress = _function(
    "nta_runtime_read_request_progress",
    ctypes.c_int,
    _Handle,
    ctypes.c_uint32,
    ctypes.POINTER(_RequestProgress),
)
_runtime_request_progress_range = _function(
    "nta_runtime_read_request_progress_range",
    ctypes.c_int,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.POINTER(_RequestProgress),
)
_runtime_copy_request_progress_async = _function(
    "nta_runtime_copy_request_progress_async",
    ctypes.c_int,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint64,
)
_runtime_work_ticket_state = _function(
    "nta_runtime_read_work_ticket_state",
    ctypes.c_int,
    _Handle,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint32),
)
_runtime_work_runnable_ns = _function(
    "nta_runtime_read_work_runnable_ns",
    ctypes.c_int,
    _Handle,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint64),
)
_runtime_device_view = _function("nta_runtime_device_view", ctypes.c_uint64, _Handle)
_runtime_device_ordinal = _function(
    "nta_runtime_device_ordinal", ctypes.c_int32, _Handle
)
_runtime_wait_object_range_terminal = _function(
    "nta_runtime_wait_object_range_terminal",
    ctypes.c_int,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint64,
)
_plan_create = _function(
    "nta_device_work_plan_create",
    ctypes.c_int,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_int32,
    _HandlePointer,
)
_plan_destroy = _function("nta_device_work_plan_destroy", None, _Handle)
_plan_upload = _function(
    "nta_device_work_plan_upload",
    ctypes.c_int,
    _Handle,
    ctypes.POINTER(WorkItem),
    ctypes.c_uint32,
    ctypes.POINTER(AcquireRequirement),
    ctypes.c_uint32,
    ctypes.POINTER(RequestRange),
    ctypes.c_uint32,
    ctypes.c_uint64,
)
_plan_wait = _function(
    "nta_device_work_plan_wait_on", ctypes.c_int, _Handle, ctypes.c_uint64
)
_plan_mark_consumed = _function(
    "nta_device_work_plan_mark_consumed", ctypes.c_int, _Handle, ctypes.c_uint64
)
_plan_sync = _function("nta_device_work_plan_synchronize_upload", ctypes.c_int, _Handle)
_plan_work_items = _function(
    "nta_device_work_plan_work_items", ctypes.c_uint64, _Handle
)
_plan_dependencies = _function(
    "nta_device_work_plan_dependencies", ctypes.c_uint64, _Handle
)
_plan_work_count = _function(
    "nta_device_work_plan_work_item_count", ctypes.c_uint32, _Handle
)
_plan_dependency_count = _function(
    "nta_device_work_plan_dependency_count", ctypes.c_uint32, _Handle
)
_plan_device_ordinal = _function(
    "nta_device_work_plan_device_ordinal", ctypes.c_int32, _Handle
)
_device_pointer_dlpack = _function(
    "nta_device_pointer_dlpack",
    ctypes.c_int,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_void_p),
)
_dlpack_destroy = _function("nta_dlpack_managed_tensor_destroy", None, ctypes.c_void_p)
_stream_synchronize = _function("nta_stream_synchronize", ctypes.c_int, ctypes.c_uint64)
_copy_host_to_device = _function(
    "nta_copy_host_to_device_async",
    ctypes.c_int,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
)
_copy_strided_host_runs = _function(
    "nta_copy_strided_host_runs_async",
    ctypes.c_int,
    ctypes.POINTER(_StridedCopyGroup),
    ctypes.c_uint32,
    ctypes.POINTER(_ContiguousCopyRun),
    ctypes.c_uint32,
    ctypes.c_uint64,
)
_operator_module_create = _function(
    "nta_jit_operator_module_create", ctypes.c_int, ctypes.c_char_p, _HandlePointer
)
_operator_module_destroy = _function("nta_jit_operator_module_destroy", None, _Handle)
_operator_module_contract = _function(
    "nta_jit_operator_module_contract",
    ctypes.c_int,
    _Handle,
    ctypes.POINTER(_OperatorContract),
)
_operator_module_plan = _function(
    "nta_jit_operator_module_plan",
    ctypes.c_int,
    _Handle,
    ctypes.POINTER(_OperatorPlan),
)
_phase_create = _function(
    "nta_jit_phase_program_create", ctypes.c_int, ctypes.c_char_p, _HandlePointer
)
_phase_destroy = _function("nta_jit_phase_program_destroy", None, _Handle)
_phase_operator_contract = _function(
    "nta_jit_phase_operator_contract",
    ctypes.c_int,
    _Handle,
    ctypes.POINTER(_OperatorContract),
)
_phase_operator_plan = _function(
    "nta_jit_phase_operator_plan",
    ctypes.c_int,
    _Handle,
    ctypes.POINTER(_OperatorPlan),
)
_phase_reset = _function(
    "nta_jit_phase_reset",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint64,
)
_phase_discover = _function(
    "nta_jit_phase_discover",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_uint64,
)
_phase_discover_unqueued_host = _function(
    "nta_jit_phase_discover_unqueued_host",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_uint64,
)
_phase_discover_ordered_nvme = _function(
    "nta_jit_phase_discover_ordered_nvme",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint64,
)
_phase_invalidate_cached_objects = _function(
    "nta_jit_phase_invalidate_cached_objects",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint64,
)
_phase_validate_indexed_host_range = _function(
    "nta_jit_phase_validate_indexed_host_range",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint64,
)
_phase_warmup_indexed_host_validation = _function(
    "nta_jit_phase_warmup_indexed_host_validation",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint64,
)
_phase_rebind_indexed_host_pairs = _function(
    "nta_jit_phase_rebind_indexed_host_pairs",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
)
_phase_preload_host = _function(
    "nta_jit_phase_preload_host",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint64,
)
_phase_preload_host_pairs = _function(
    "nta_jit_phase_preload_host_pairs",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint64,
)
_phase_preload_host_pairs_ordered = _function(
    "nta_jit_phase_preload_host_pairs_ordered",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint64,
)
_phase_alias_preloaded = _function(
    "nta_jit_phase_alias_preloaded_objects",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_uint64,
)
_phase_host = _function(
    "nta_jit_phase_progress_host",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint64,
)
_phase_prepare_ready_window = _function(
    "nta_jit_phase_prepare_ready_window",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint64,
)
_phase_prepare_event_work_partition = _function(
    "nta_jit_phase_prepare_event_work_partition",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint64,
)
_phase_indexed_host_range = _function(
    "nta_jit_phase_progress_indexed_host_range",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint64,
)
_phase_validated_indexed_host_range = _function(
    "nta_jit_phase_progress_validated_indexed_host_range",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint64,
)
_phase_validated_indexed_host_range_parallel = _function(
    "nta_jit_phase_progress_validated_indexed_host_range_parallel",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint64,
)
_phase_set_indexed_row_counts = _function(
    "nta_jit_phase_set_indexed_row_counts",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint64,
)
_phase_prepare_selected_indexed_rows = _function(
    "nta_jit_phase_prepare_selected_indexed_rows",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint64,
)
_phase_prepare_bounded_selected_indexed_rows = _function(
    "nta_jit_phase_prepare_bounded_selected_indexed_rows",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint64,
)
_phase_reduce_mapped_indexed_key_pages = _function(
    "nta_jit_phase_reduce_mapped_indexed_key_pages",
    ctypes.c_int,
    _Handle,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint64,
    *([ctypes.c_uint32] * 5),
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
)
_phase_reduce_mapped_key_pages = _function(
    "nta_jit_phase_reduce_mapped_key_pages",
    ctypes.c_int,
    _Handle,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
)
_phase_nvme = _function(
    "nta_jit_phase_progress_nvme",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint64,
)
_phase_nvme_until_idle = _function(
    "nta_jit_phase_progress_nvme_until_idle",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint64,
)
_phase_nvme_ordered_until_idle = _function(
    "nta_jit_phase_progress_nvme_ordered_until_idle",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint64,
)
_phase_publish = _function(
    "nta_jit_phase_publish",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint64,
)
_phase_complete = _function(
    "nta_jit_phase_complete",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint64,
)
_phase_complete_stream_ordered = _function(
    "nta_jit_phase_complete_stream_ordered",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_uint64,
)


if _api_version() != API_VERSION:
    raise RuntimeError("libnta-runtime.so uses an incompatible C API")


def _check(status: int) -> None:
    if status != 0:
        message = _last_error()
        raise RuntimeError(
            message.decode("utf-8") if message else f"NTA status {status}"
        )


def _stream_address(value: Any) -> int:
    if value is None:
        return 0
    return int(getattr(value, "cuda_stream", value))


def _event_address(value: Any) -> int:
    if value is None:
        return 0
    return int(getattr(value, "cuda_event", value))


def synchronize_stream(stream: Any = None) -> None:
    _check(_stream_synchronize(_stream_address(stream)))


def copy_host_to_device_async(
    destination: int, source: int, bytes: int, stream: Any = None
) -> None:
    if min(destination, source, bytes) <= 0:
        raise ValueError("host-to-device copy needs addresses and bytes")
    _check(_copy_host_to_device(destination, source, bytes, _stream_address(stream)))


def copy_strided_host_runs_async(groups: Any, runs: Any, stream: Any = None) -> int:
    """Enqueue safe batches over a shared exact index-run layout.

    Returns the number of native batch submissions. Groups whose conservative
    destination spans overlap are partitioned into separate stream-ordered
    submissions, so the CUDA batch never contains dependent copies.
    """
    from nta_runtime.indexed_transfer import ContiguousPairRun, StridedCopyGroup

    group_values = tuple(groups)
    run_values = tuple(runs)
    if not group_values or not run_values:
        raise ValueError("strided host copy needs groups and runs")
    if any(not isinstance(group, StridedCopyGroup) for group in group_values):
        raise TypeError("strided host copy groups use StridedCopyGroup")
    if any(not isinstance(run, ContiguousPairRun) for run in run_values):
        raise TypeError("strided host copy runs use ContiguousPairRun")
    if len(group_values) >= 1 << 32 or len(run_values) >= 1 << 32:
        raise ValueError("strided host copy exceeds uint32 capacity")
    for run in run_values:
        if (
            min(run.source_first, run.destination_first) < 0
            or run.row_count <= 0
            or max(run.source_first, run.destination_first, run.row_count) >= 1 << 32
        ):
            raise ValueError("strided host copy run exceeds uint32 geometry")
    native_runs = (_ContiguousCopyRun * len(run_values))(
        *(
            _ContiguousCopyRun(
                run.source_first, run.destination_first, run.row_count, 0
            )
            for run in run_values
        )
    )
    maximum_copies = 1 << 16
    # Every group shares the same exact run layout.  Reduce its bounds once
    # instead of walking the Cartesian product in Python; the native boundary
    # still validates every concrete operation before submission.  Large
    # HiCache leases can contain hundreds of runs across dozens of K/V layer
    # groups, so the redundant O(groups * runs) interpreter loop otherwise
    # becomes visible in TTFT even though CUDA receives one batched copy.
    maximum_source_row = max(run.source_first + run.row_count for run in run_values)
    minimum_destination_row = min(run.destination_first for run in run_values)
    maximum_destination_row = max(
        run.destination_first + run.row_count for run in run_values
    )
    batches: list[list[tuple[StridedCopyGroup, tuple[int, int]]]] = []
    for group in group_values:
        if (
            maximum_source_row > group.source_rows
            or maximum_destination_row > group.destination_rows
        ):
            raise ValueError("strided host-copy run exceeds group geometry")
        span = (
            group.destination_address
            + minimum_destination_row * group.destination_stride_bytes,
            group.destination_address
            + (maximum_destination_row - 1) * group.destination_stride_bytes
            + group.row_bytes,
        )
        if span[1] >= 1 << 64:
            raise ValueError("strided host-copy address geometry exceeds uint64")
        for batch in batches:
            if (len(batch) + 1) * len(run_values) > maximum_copies:
                continue
            if all(span[1] <= other[0] or other[1] <= span[0] for _, other in batch):
                batch.append((group, span))
                break
        else:
            batches.append([(group, span)])

    stream_address = _stream_address(stream)
    for batch in batches:
        native_groups = (_StridedCopyGroup * len(batch))(
            *(
                _StridedCopyGroup(
                    group.source_address,
                    group.destination_address,
                    group.source_rows,
                    group.destination_rows,
                    group.row_bytes,
                    group.source_stride_bytes,
                    group.destination_stride_bytes,
                    0,
                )
                for group, _ in batch
            )
        )
        _check(
            _copy_strided_host_runs(
                native_groups,
                len(batch),
                native_runs,
                len(run_values),
                stream_address,
            )
        )
    return len(batches)


def _device_byte_tensor(address: int, device_ordinal: int):
    if address == 0:
        raise ValueError("cannot wrap a null CUDA address")
    managed = ctypes.c_void_p()
    _check(_device_pointer_dlpack(address, 1, device_ordinal, ctypes.byref(managed)))
    capsule_new = ctypes.pythonapi.PyCapsule_New
    capsule_new.restype = ctypes.py_object
    capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    try:
        capsule = capsule_new(managed, b"dltensor", None)
        import torch

        return torch.utils.dlpack.from_dlpack(capsule)
    except Exception:
        _dlpack_destroy(managed)
        raise


def device_abi_version() -> int:
    return int(_device_abi_version())


class _Owner:
    _handle: _Handle
    _destroy: Any

    def close(self) -> None:
        if self._handle:
            self._destroy(self._handle)
            self._handle = _Handle()

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class NvmeHbmRegion(_Owner):
    """Setup-time peer mapping for one stable caller-owned CUDA range."""

    _destroy = staticmethod(_nvme_hbm_region_destroy)

    def __init__(self, transport: "NvmeTransport", address: int, bytes: int):
        if address <= 0 or bytes <= 0:
            raise ValueError("NVMe HBM region address and bytes must be positive")
        self._handle = _Handle()
        self._transport = transport
        _check(
            _nvme_register_hbm_region(
                transport._handle,
                address,
                bytes,
                ctypes.byref(self._handle),
            )
        )
        self.address = address
        self.bytes = bytes


@dataclasses.dataclass(frozen=True, slots=True)
class RegisteredNvmeObjectInstall:
    """One typed entry in a stream-ordered registered-HBM publication."""

    slot: int
    object_id: int
    version: int
    source_byte_offset: int
    bytes: int
    region: NvmeHbmRegion
    destination_device_address: int
    prior_consumer_event: Any = dataclasses.field(
        default=None, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot", _u32(self.slot, "NVMe object slot"))
        object.__setattr__(
            self, "object_id", _u64(self.object_id, "NVMe object identity")
        )
        object.__setattr__(
            self,
            "version",
            _u32(self.version, "NVMe object version", positive=True),
        )
        object.__setattr__(
            self,
            "source_byte_offset",
            _u64(self.source_byte_offset, "NVMe object source offset"),
        )
        object.__setattr__(
            self, "bytes", _u64(self.bytes, "NVMe object bytes", positive=True)
        )
        object.__setattr__(
            self,
            "destination_device_address",
            _u64(
                self.destination_device_address,
                "NVMe object destination address",
                positive=True,
            ),
        )
        try:
            region_address_value = self.region.address
            region_bytes_value = self.region.bytes
        except AttributeError as error:
            raise TypeError(
                "registered NVMe object requires an HBM region contract"
            ) from error
        if self.source_byte_offset > (1 << 64) - 1 - self.bytes:
            raise ValueError("NVMe object source extent overflows uint64")
        region_address = _u64(
            region_address_value, "registered HBM region address", positive=True
        )
        region_bytes = _u64(
            region_bytes_value, "registered HBM region bytes", positive=True
        )
        if (
            self.destination_device_address < region_address
            or self.bytes > region_bytes
            or self.destination_device_address - region_address
            > region_bytes - self.bytes
        ):
            raise ValueError(
                "NVMe numerical destination exceeds its registered HBM region"
            )

    def native(self) -> _RegisteredNvmeObject:
        if not isinstance(self.region, NvmeHbmRegion) or not self.region._handle:
            raise ValueError("registered NVMe object requires a live HBM region")
        prior_consumer_event = _u64(
            _event_address(self.prior_consumer_event), "prior consumer event"
        )
        return _RegisteredNvmeObject(
            self.object_id,
            self.source_byte_offset,
            self.bytes,
            self.region._handle,
            self.destination_device_address,
            prior_consumer_event,
            self.slot,
            self.version,
        )


class NvmeTransport(_Owner):
    _destroy = staticmethod(_nvme_destroy)

    def __init__(self, options: NvmeOptions):
        self._handle = _Handle()
        endpoint = options.endpoint.encode("utf-8")
        native = _NvmeOptions(
            ctypes.sizeof(_NvmeOptions),
            API_VERSION,
            endpoint,
            options.device_ordinal,
            options.namespace_id,
            options.queue_depth,
            options.admin_timeout_ms,
            int(options.trust_read_only_device_code),
            int(options.dma_target),
            int(options.hbm_mapping_policy),
        )
        _check(_nvme_create(ctypes.byref(native), ctypes.byref(self._handle)))

    @property
    def capabilities(self) -> NvmeCapabilities:
        value = _NvmeCapabilities()
        _check(_nvme_capabilities(self._handle, ctypes.byref(value)))
        return NvmeCapabilities(
            value.queue_depth,
            value.controller_page_size,
            value.lba_size,
            value.max_transfer_bytes,
            value.namespace_bytes,
            value.queue_id,
            value.queue_count,
            value.device_ordinal,
            bool(value.supports_hbm_peer_dma),
            NvmeHbmMappingBackend(value.hbm_mapping_backend),
            bool(value.translated_iommu),
            bool(value.namespace_read_only),
            bool(value.gpu_doorbell_mapping_validated),
        )

    def register_hbm_region(self, address: int, bytes: int) -> NvmeHbmRegion:
        """Pin/map a stable CUDA range once for allocation-free transfer views."""

        return NvmeHbmRegion(self, address, bytes)

    def describe_hbm_region(self, address: int, bytes: int) -> NvmeHbmRegistrationRange:
        """Describe a slice without pinning it or changing IOMMU state."""

        _u64(address, "NVMe HBM address", positive=True)
        _u64(bytes, "NVMe HBM bytes", positive=True)
        native = _NvmeHbmRegistrationRange()
        _check(
            _nvme_describe_hbm_region(
                self._handle,
                address,
                bytes,
                ctypes.byref(native),
            )
        )
        return NvmeHbmRegistrationRange(
            native.allocation_address,
            native.allocation_bytes,
            native.registration_address,
            native.registration_bytes,
        )

    @property
    def stats(self) -> NvmeQueueStats:
        value = _NvmeQueueStats()
        _check(_nvme_stats(self._handle, ctypes.byref(value)))
        return NvmeQueueStats(
            value.submitted,
            value.completed,
            value.failed,
            value.direct_submitted,
            value.direct_fallbacks,
            value.outstanding,
            value.error,
            value.sq_tail,
            value.cq_head,
            value.cq_phase,
            value.next_completion_dword3,
            value.hbm_region_registrations,
            value.hbm_region_bytes,
            value.hbm_transfer_views,
        )


class CxlDaxTransport(_Owner):
    _destroy = staticmethod(_cxl_destroy)

    def __init__(self, options: CxlDaxOptions):
        self._handle = _Handle()
        self._endpoint = options.endpoint.encode("utf-8")
        native = _CxlOptions(
            ctypes.sizeof(_CxlOptions),
            API_VERSION,
            self._endpoint,
            options.window_bytes,
            options.device_ordinal,
        )
        _check(_cxl_create(ctypes.byref(native), ctypes.byref(self._handle)))

    @property
    def capabilities(self) -> CxlDaxCapabilities:
        value = _CxlCapabilities()
        _check(_cxl_capabilities(self._handle, ctypes.byref(value)))
        return CxlDaxCapabilities(
            value.window_bytes,
            value.mapped_device_address,
            value.device_ordinal,
            bool(value.host_registered),
            bool(value.direct_device_visible),
        )

    @property
    def device_address(self) -> int:
        return self.capabilities.mapped_device_address


class Runtime(_Owner):
    _destroy = staticmethod(_runtime_destroy)

    def __init__(
        self,
        config: RuntimeConfig,
        nvme: NvmeTransport | None = None,
        cxl: CxlDaxTransport | None = None,
    ):
        self._handle = _Handle()
        native = config.native()
        nvme_handle = nvme._handle if nvme is not None else _Handle()
        cxl_handle = cxl._handle if cxl is not None else _Handle()
        _check(
            _runtime_create(
                ctypes.byref(native),
                nvme_handle,
                cxl_handle,
                ctypes.byref(self._handle),
            )
        )
        self._nvme = nvme
        self._cxl = cxl
        self._config = config
        self._device_view_tensor = None

    def close(self) -> None:
        self._device_view_tensor = None
        super().close()

    @property
    def device_view(self) -> int:
        return int(_runtime_device_view(self._handle))

    @property
    def device_ordinal(self) -> int:
        return int(_runtime_device_ordinal(self._handle))

    @property
    def config(self) -> RuntimeConfig:
        """Return the immutable process-local runtime configuration."""
        return self._config

    def tier_descriptor(self, tier: TierKind) -> TierDescriptor:
        native = _TierDescriptor()
        _check(_runtime_tier_descriptor(self._handle, int(tier), ctypes.byref(native)))
        if native.reserved != 0:
            raise RuntimeError("native tier descriptor reserved bits are nonzero")

        def owner(value: int, *, optional: bool = False) -> ResourceOwner | None:
            if value == 0 and optional:
                return None
            values = {
                1: ResourceOwner.ENGINE,
                2: ResourceOwner.RUNTIME,
                3: ResourceOwner.TRANSPORT,
            }
            try:
                return values[int(value)]
            except KeyError as error:
                raise RuntimeError(
                    "native tier descriptor has invalid owner"
                ) from error

        return TierDescriptor(
            TierKind(native.source_kind),
            TierCapability(native.capabilities),
            native.device_state,
            native.estimated_latency_ns,
            native.estimated_bandwidth_bytes_per_second,
            bool(native.active),
            native.flags,
            owner(native.protocol_owner),
            owner(native.payload_owner),
            owner(native.transfer_destination_owner, optional=True),
            owner(native.mapping_owner, optional=True),
            owner(native.directory_owner),
        )

    @property
    def device_view_tensor(self):
        if self._device_view_tensor is None:
            self._device_view_tensor = _device_byte_tensor(
                self.device_view, self.device_ordinal
            )
        return self._device_view_tensor

    def set_request(
        self,
        slot: int,
        request_id: int,
        generation: int,
        *,
        tenant_id: int = 0,
        priority: int = 0,
        deadline_clock: int = 0,
        max_outstanding_bytes: int = (1 << 64) - 1,
    ) -> None:
        _check(
            _runtime_set_request(
                self._handle,
                slot,
                request_id,
                generation,
                tenant_id,
                priority,
                deadline_clock,
                max_outstanding_bytes,
            )
        )

    def publish_requests_async(
        self, requests: Iterable[RequestSpec], stream: Any = None
    ) -> None:
        """Publish changed request slots before later work on ``stream``.

        The stream must be ordered after every old generation that uses the
        supplied slots. This method does not synchronize the calling thread.
        """
        values = sorted(requests, key=lambda request: request.slot)
        if not values:
            raise ValueError("request publication batch cannot be empty")
        if len({request.slot for request in values}) != len(values):
            raise ValueError("request publication slots must be unique")
        native = (_RequestSpec * len(values))(*(request.native() for request in values))
        _check(
            _runtime_publish_requests_async(
                self._handle,
                native,
                len(values),
                _stream_address(stream),
            )
        )

    def cancel_request(self, slot: int, generation: int) -> None:
        _check(_runtime_cancel_request(self._handle, slot, generation))

    def set_tenant_budget(self, tenant_id: int, max_bytes: int) -> None:
        """Bound this tenant's concurrently acquired staging bytes."""
        _check(_runtime_set_tenant_budget(self._handle, tenant_id, max_bytes))

    def register_object(
        self,
        slot: int,
        object_id: int,
        version: int,
        bytes: int,
        replicas: Iterable[Replica],
        *,
        staging_device_address: int = 0,
    ) -> int:
        values = [replica.native() for replica in replicas]
        if not values:
            raise ValueError("at least one object replica is required")
        array = (_Replica * len(values))(*values)
        direct = ctypes.c_uint64()
        _check(
            _runtime_register_object(
                self._handle,
                slot,
                object_id,
                version,
                bytes,
                staging_device_address,
                array,
                len(values),
                ctypes.byref(direct),
            )
        )
        return int(direct.value)

    def register_indexed_host_object(
        self,
        slot: int,
        object_id: int,
        version: int,
        source_device_address: int,
        staging_device_address: int,
        source_indices_device_address: int,
        staging_indices_device_address: int,
        index_count: int,
        element_bytes: int,
        source_stride_bytes: int,
        staging_stride_bytes: int,
        source_index_limit: int,
        staging_index_limit: int,
    ) -> None:
        """Register a non-owning indexed pinned-host to HBM transfer."""
        _check(
            _runtime_register_indexed_host_object(
                self._handle,
                slot,
                object_id,
                version,
                source_device_address,
                staging_device_address,
                source_indices_device_address,
                staging_indices_device_address,
                index_count,
                element_bytes,
                source_stride_bytes,
                staging_stride_bytes,
                source_index_limit,
                staging_index_limit,
            )
        )

    def register_indexed_host_objects(
        self,
        first_slot: int,
        objects: Iterable[IndexedHostObject],
        stream: Any = None,
        quiescence_event: Any = None,
        index_binding: IndexedHostIndexBinding | None = None,
    ) -> None:
        """Bulk-publish a contiguous layer's indexed host objects."""
        values = [object_.native() for object_ in objects]
        if not values:
            raise ValueError("indexed host object batch cannot be empty")
        array = (_IndexedHostObject * len(values))(*values)
        self._register_indexed_host_native(
            first_slot,
            array,
            stream=stream,
            quiescence_event=quiescence_event,
            index_binding=index_binding,
        )

    def register_indexed_acquisition_plan(
        self,
        plan: IndexedAcquisitionPlan,
        *,
        stream: Any,
        quiescence_event: Any = None,
        index_binding: IndexedHostIndexBinding | None = None,
    ) -> None:
        """Publish one pre-materialized indexed resource/work image."""

        if not isinstance(plan, IndexedAcquisitionPlan):
            raise TypeError("indexed host publication requires IndexedAcquisitionPlan")
        self._register_indexed_host_native(
            plan.first_slot,
            plan.native_objects,
            stream=stream,
            quiescence_event=quiescence_event,
            index_binding=index_binding,
        )

    def _register_indexed_host_native(
        self,
        first_slot: int,
        array: Any,
        *,
        stream: Any,
        quiescence_event: Any,
        index_binding: IndexedHostIndexBinding | None,
    ) -> None:
        if (
            not isinstance(array, ctypes.Array)
            or getattr(type(array), "_type_", None) is not _IndexedHostObject
        ):
            raise TypeError("indexed host native objects have an invalid ABI")
        if not len(array):
            raise ValueError("indexed host object batch cannot be empty")
        if quiescence_event is not None and stream is None:
            raise ValueError("quiescence_event requires an asynchronous stream")
        if index_binding is not None:
            if stream is None:
                raise ValueError("index_binding requires an asynchronous stream")
            if not isinstance(index_binding, IndexedHostIndexBinding):
                raise TypeError("index_binding has an invalid type")
            native_binding = index_binding.native()
            _check(
                _runtime_register_indexed_host_objects_async_bound(
                    self._handle,
                    first_slot,
                    array,
                    len(array),
                    ctypes.byref(native_binding),
                    _stream_address(stream),
                    0 if quiescence_event is None else _event_address(quiescence_event),
                )
            )
        elif stream is None:
            _check(
                _runtime_register_indexed_host_objects(
                    self._handle, first_slot, array, len(array)
                )
            )
        elif quiescence_event is not None:
            _check(
                _runtime_register_indexed_host_objects_async_quiesced(
                    self._handle,
                    first_slot,
                    array,
                    len(array),
                    _stream_address(stream),
                    _event_address(quiescence_event),
                )
            )
        else:
            _check(
                _runtime_register_indexed_host_objects_async(
                    self._handle,
                    first_slot,
                    array,
                    len(array),
                    _stream_address(stream),
                )
            )

    def bind_tensor_maps(
        self,
        object_slot: int,
        relative_replica: int,
        replica_tensor_map: int,
        staging_tensor_map: int = 0,
    ) -> None:
        _check(
            _runtime_bind_tensor_maps(
                self._handle,
                object_slot,
                relative_replica,
                replica_tensor_map,
                staging_tensor_map,
            )
        )

    def wait_object_range_terminal(
        self, first_object_slot: int, object_count: int, stream: Any
    ) -> None:
        """Order ``stream`` after a contiguous Ready/Failed object range."""

        if (
            first_object_slot < 0
            or object_count <= 0
            or first_object_slot > self._config.object_capacity
            or object_count > self._config.object_capacity - first_object_slot
        ):
            raise ValueError("object terminal-wait range exceeds object capacity")
        if stream is None:
            raise ValueError("object terminal wait requires an explicit stream")
        _check(
            _runtime_wait_object_range_terminal(
                self._handle,
                first_object_slot,
                object_count,
                _stream_address(stream),
            )
        )

    def install_nvme_object(
        self,
        slot: int,
        object_id: int,
        version: int,
        source_byte_offset: int,
        bytes: int,
    ) -> int:
        """Republish an exact NVMe range, reusing the slot buffer when possible."""
        destination = ctypes.c_uint64()
        _check(
            _runtime_install_nvme_object(
                self._handle,
                slot,
                object_id,
                version,
                source_byte_offset,
                bytes,
                ctypes.byref(destination),
            )
        )
        return int(destination.value)

    def install_nvme_object_async(
        self,
        slot: int,
        object_id: int,
        version: int,
        source_byte_offset: int,
        bytes: int,
        stream: Any,
        prior_consumer_event: Any = None,
    ) -> int:
        """Publish an NVMe range without a device-wide replacement fence.

        The event must cover the previous forward's consumers when ``slot``
        already owns a destination.  Native retirement keeps that destination
        alive until the event-ordered replacement is safe.
        """
        if stream is None:
            raise ValueError("asynchronous NVMe installation requires a stream")
        destination = ctypes.c_uint64()
        _check(
            _runtime_install_nvme_object_async(
                self._handle,
                slot,
                object_id,
                version,
                source_byte_offset,
                bytes,
                _stream_address(stream),
                _event_address(prior_consumer_event),
                ctypes.byref(destination),
            )
        )
        return int(destination.value)

    def install_registered_nvme_object(
        self,
        slot: int,
        object_id: int,
        version: int,
        source_byte_offset: int,
        bytes: int,
        region: NvmeHbmRegion,
        destination_device_address: int,
    ) -> int:
        """Publish one transfer view of a setup-time registered HBM region."""
        if not isinstance(region, NvmeHbmRegion) or not region._handle:
            raise ValueError("registered NVMe object requires a live HBM region")
        if destination_device_address <= 0:
            raise ValueError("registered NVMe destination must be positive")
        destination = ctypes.c_uint64()
        _check(
            _runtime_install_registered_nvme_object(
                self._handle,
                slot,
                object_id,
                version,
                source_byte_offset,
                bytes,
                region._handle,
                destination_device_address,
                ctypes.byref(destination),
            )
        )
        return int(destination.value)

    def install_registered_nvme_object_async(
        self,
        slot: int,
        object_id: int,
        version: int,
        source_byte_offset: int,
        bytes: int,
        region: NvmeHbmRegion,
        destination_device_address: int,
        stream: Any,
        prior_consumer_event: Any = None,
    ) -> int:
        """Stream-order one view of a setup-time registered HBM region."""
        if not isinstance(region, NvmeHbmRegion) or not region._handle:
            raise ValueError("registered NVMe object requires a live HBM region")
        if destination_device_address <= 0:
            raise ValueError("registered NVMe destination must be positive")
        if stream is None:
            raise ValueError("asynchronous NVMe installation requires a stream")
        destination = ctypes.c_uint64()
        _check(
            _runtime_install_registered_nvme_object_async(
                self._handle,
                slot,
                object_id,
                version,
                source_byte_offset,
                bytes,
                region._handle,
                destination_device_address,
                _stream_address(stream),
                _event_address(prior_consumer_event),
                ctypes.byref(destination),
            )
        )
        return int(destination.value)

    def install_registered_nvme_objects_async(
        self,
        objects: Iterable[RegisteredNvmeObjectInstall],
        stream: Any,
    ) -> tuple[int, ...]:
        """Publish one contiguous registered-HBM directory transaction.

        All Python and native validation completes before either bulk H2D
        directory copy is enqueued. Repeated prior-consumer events are folded
        by the native runtime, while ownership remains field-scoped per slot.
        """

        values = tuple(objects)
        if not values or any(
            not isinstance(value, RegisteredNvmeObjectInstall) for value in values
        ):
            raise ValueError(
                "registered NVMe batch requires typed object installations"
            )
        if stream is None:
            raise ValueError("asynchronous NVMe installation requires a stream")
        first_slot = values[0].slot
        if tuple(value.slot for value in values) != tuple(
            range(first_slot, first_slot + len(values))
        ):
            raise ValueError(
                "registered NVMe batch slots must be contiguous and increasing"
            )
        if len({value.object_id for value in values}) != len(values):
            raise ValueError("registered NVMe batch repeats an object identity")
        if self._nvme is None or any(
            not isinstance(value.region, NvmeHbmRegion)
            or not value.region._handle
            or value.region._transport is not self._nvme
            for value in values
        ):
            raise ValueError(
                "registered NVMe batch regions belong to a different transport"
            )
        native = (_RegisteredNvmeObject * len(values))()
        for index, value in enumerate(values):
            _REGISTERED_NVME_OBJECT_PACKER.pack_into(
                native,
                index * _REGISTERED_NVME_OBJECT_PACKER.size,
                value.object_id,
                value.source_byte_offset,
                value.bytes,
                int(value.region._handle.value),
                value.destination_device_address,
                _u64(
                    _event_address(value.prior_consumer_event),
                    "prior consumer event",
                ),
                value.slot,
                value.version,
            )
        destinations = (ctypes.c_uint64 * len(values))()
        _check(
            _runtime_install_registered_nvme_objects_async(
                self._handle,
                native,
                len(values),
                _stream_address(stream),
                destinations,
            )
        )
        return tuple(int(address) for address in destinations)

    @property
    def pending_count(self) -> int:
        value = ctypes.c_uint32()
        _check(_runtime_pending_count(self._handle, ctypes.byref(value)))
        return int(value.value)

    def work_ticket_state(self, work_ticket: int) -> int:
        value = ctypes.c_uint32()
        _check(
            _runtime_work_ticket_state(self._handle, work_ticket, ctypes.byref(value))
        )
        return int(value.value)

    def work_runnable_ns(self, work_ticket_count: int) -> tuple[int, ...]:
        if work_ticket_count <= 0:
            raise ValueError("work ticket count must be positive")
        values = (ctypes.c_uint64 * work_ticket_count)()
        _check(_runtime_work_runnable_ns(self._handle, work_ticket_count, values))
        return tuple(int(value) for value in values)

    def epoch_status(self, work_ticket_count: int | None = None) -> EpochStatus:
        count = (
            self._config.work_ticket_capacity
            if work_ticket_count is None
            else work_ticket_count
        )
        value = _EpochStatus()
        _check(_runtime_epoch_status(self._handle, count, ctypes.byref(value)))
        return EpochStatus(
            value.total,
            value.fresh,
            value.pending,
            value.ready,
            value.done,
            value.cancelled,
            value.failed,
            value.initializing,
        )

    @property
    def sticky_failed_count(self) -> int:
        value = ctypes.c_uint32()
        _check(_runtime_sticky_failed_count(self._handle, ctypes.byref(value)))
        return int(value.value)

    def request_progress(self, request_slot: int) -> RequestProgress:
        value = _RequestProgress()
        _check(
            _runtime_request_progress(self._handle, request_slot, ctypes.byref(value))
        )
        return _request_progress_value(value)

    def request_progress_range(
        self, first_request_slot: int, request_count: int
    ) -> tuple[RequestProgress, ...]:
        if request_count <= 0:
            raise ValueError("request progress count must be positive")
        values = (_RequestProgress * request_count)()
        _check(
            _runtime_request_progress_range(
                self._handle, first_request_slot, request_count, values
            )
        )
        return tuple(_request_progress_value(value) for value in values)

    def request_progress_snapshot(
        self, capacity: int | None = None
    ) -> "RequestProgressSnapshot":
        return RequestProgressSnapshot(
            self,
            self._config.request_capacity if capacity is None else capacity,
        )


class RequestProgressSnapshot:
    """Reusable stream-ordered request-progress snapshot in pinned memory."""

    def __init__(self, runtime: Runtime, capacity: int) -> None:
        if capacity <= 0 or capacity > runtime._config.request_capacity:
            raise ValueError("request progress snapshot capacity is invalid")
        import torch

        self._runtime = runtime
        self._capacity = capacity
        self._storage = torch.empty(
            capacity * ctypes.sizeof(_RequestProgress),
            dtype=torch.uint8,
            pin_memory=True,
        )
        self._event = torch.cuda.Event()
        self._pending: tuple[int, int] | None = None

    @property
    def pending(self) -> bool:
        return self._pending is not None

    @property
    def ready(self) -> bool:
        return self._pending is not None and self._event.query()

    def capture(
        self, first_request_slot: int, request_count: int, stream: Any = None
    ) -> None:
        if request_count <= 0 or request_count > self._capacity:
            raise ValueError("request progress snapshot count is invalid")
        if self._pending is not None and not self._event.query():
            raise RuntimeError("request progress snapshot is still in flight")
        import torch

        selected_stream = (
            torch.cuda.current_stream(self._runtime.device_ordinal)
            if stream is None
            else stream
        )
        _check(
            _runtime_copy_request_progress_async(
                self._runtime._handle,
                first_request_slot,
                request_count,
                self._storage.data_ptr(),
                _stream_address(selected_stream),
            )
        )
        self._event.record(selected_stream)
        self._pending = (first_request_slot, request_count)

    def query(self) -> tuple[RequestProgress, ...] | None:
        if self._pending is None or not self._event.query():
            return None
        return self._consume()

    def wait(self) -> tuple[RequestProgress, ...]:
        if self._pending is None:
            raise RuntimeError("request progress snapshot has not been captured")
        self._event.synchronize()
        return self._consume()

    def _consume(self) -> tuple[RequestProgress, ...]:
        if self._pending is None:
            raise RuntimeError("request progress snapshot has not been captured")
        _, request_count = self._pending
        base = self._storage.data_ptr()
        stride = ctypes.sizeof(_RequestProgress)
        result = tuple(
            _request_progress_value(
                _RequestProgress.from_address(base + index * stride)
            )
            for index in range(request_count)
        )
        self._pending = None
        return result


class DeviceWorkPlan(_Owner):
    _destroy = staticmethod(_plan_destroy)

    def __init__(
        self,
        work_item_capacity: int,
        dependency_capacity: int,
        device_ordinal: int = -1,
    ):
        self._handle = _Handle()
        _check(
            _plan_create(
                work_item_capacity,
                dependency_capacity,
                device_ordinal,
                ctypes.byref(self._handle),
            )
        )
        self._work_items_tensor = None
        self._dependencies_tensor = None
        self._has_external = False

    def close(self) -> None:
        self._work_items_tensor = None
        self._dependencies_tensor = None
        super().close()

    def upload(
        self,
        work_items: Iterable[WorkItem],
        dependencies: Iterable[AcquireRequirement],
        requests: Iterable[RequestRange],
        stream: Any = None,
    ) -> None:
        work = list(work_items)
        dependency = list(dependencies)
        request = list(requests)
        if not work or not dependency or not request:
            raise ValueError("work-plan arrays must be non-empty")
        if any(item.dependency_count < item.direct_dependency_count for item in work):
            raise ValueError("direct dependency count exceeds total dependencies")
        work_array = (WorkItem * len(work))(*work)
        dependency_array = (AcquireRequirement * len(dependency))(*dependency)
        request_array = (RequestRange * len(request))(*request)
        self._upload_native(work_array, dependency_array, request_array, stream)

    def _upload_native(
        self,
        work_items: Any,
        dependencies: Any,
        requests: Any,
        stream: Any,
    ) -> None:
        """Upload already-materialized ctypes arrays without another host copy."""

        work_count = len(work_items)
        dependency_count = len(dependencies)
        request_count = len(requests)
        if not work_count or not dependency_count or not request_count:
            raise ValueError("work-plan native arrays must be non-empty")
        _check(
            _plan_upload(
                self._handle,
                work_items,
                work_count,
                dependencies,
                dependency_count,
                requests,
                request_count,
                _stream_address(stream),
            )
        )
        self._has_external = any(
            item.direct_dependency_count != item.dependency_count for item in work_items
        )

    def upload_exact(
        self,
        topology: ExactWorkTopology,
        dependency_spans: Iterable[WorkDependencySpan],
        dependencies: Iterable[AcquireRequirement],
        *,
        stream: Any = None,
    ) -> None:
        """Materialize a compact exact topology into the native ticket ABI.

        Request index, reduction ownership, contributor coordinates, and work
        tickets are canonical consequences of the request ranges.  Deriving
        them here avoids rebuilding a semantic object graph in every framework
        adapter while retaining the same native validation and generation
        checks as :meth:`upload`.
        """

        if not isinstance(topology, ExactWorkTopology):
            raise TypeError("exact work-plan upload requires ExactWorkTopology")
        spans = tuple(dependency_spans)
        if (
            isinstance(dependencies, ctypes.Array)
            and getattr(type(dependencies), "_type_", None) is AcquireRequirement
        ):
            native_dependencies = dependencies
        else:
            dependency_values = tuple(dependencies)
            native_dependencies = (AcquireRequirement * len(dependency_values))(
                *dependency_values
            )
        if len(spans) != topology.work_count:
            raise ValueError("exact topology and dependency spans must align")
        if any(not isinstance(span, WorkDependencySpan) for span in spans):
            raise TypeError("exact work-plan dependency spans must be typed")
        dependency_total = len(native_dependencies)
        if dependency_total == 0:
            raise ValueError("exact work-plan dependency array cannot be empty")
        for span in spans:
            if (
                span.begin > dependency_total
                or span.count > dependency_total - span.begin
            ):
                raise ValueError("exact work dependency span exceeds its array")

        work_items = (WorkItem * topology.work_count)()
        for request in topology.requests:
            for contributor_index in range(request.work_count):
                work_ticket = request.work_begin + contributor_index
                span = spans[work_ticket]
                _WORK_ITEM_PACKER.pack_into(
                    work_items,
                    work_ticket * _WORK_ITEM_PACKER.size,
                    request.request_index,
                    request.request_slot,
                    request.generation,
                    topology.logical_work[work_ticket],
                    span.begin,
                    span.count,
                    span.direct_count,
                    work_ticket,
                    request.request_index,
                    contributor_index,
                    request.work_count,
                    topology.estimated_compute_ns[work_ticket],
                    topology.ready_deadline_offset_ns[work_ticket],
                    0,
                    0,
                )
        native_requests = (RequestRange * len(topology.requests))()
        for index, request in enumerate(topology.requests):
            _REQUEST_RANGE_PACKER.pack_into(
                native_requests,
                index * _REQUEST_RANGE_PACKER.size,
                request.work_begin,
                request.work_count,
                request.request_slot,
                request.generation,
            )
        self._upload_native(
            work_items,
            native_dependencies,
            native_requests,
            stream,
        )

    def wait_on(self, stream: Any) -> None:
        _check(_plan_wait(self._handle, _stream_address(stream)))

    def mark_consumed(self, stream: Any) -> None:
        """Fence the last consumer launch before a future upload can reuse it."""
        _check(_plan_mark_consumed(self._handle, _stream_address(stream)))

    def synchronize_upload(self) -> None:
        _check(_plan_sync(self._handle))

    @property
    def work_items_address(self) -> int:
        return int(_plan_work_items(self._handle))

    @property
    def dependencies_address(self) -> int:
        return int(_plan_dependencies(self._handle))

    @property
    def work_item_count(self) -> int:
        return int(_plan_work_count(self._handle))

    @property
    def dependency_count(self) -> int:
        return int(_plan_dependency_count(self._handle))

    @property
    def device_ordinal(self) -> int:
        return int(_plan_device_ordinal(self._handle))

    @property
    def work_items_tensor(self):
        if self._work_items_tensor is None:
            self._work_items_tensor = _device_byte_tensor(
                self.work_items_address, self.device_ordinal
            )
        return self._work_items_tensor

    @property
    def dependencies_tensor(self):
        if self._dependencies_tensor is None:
            self._dependencies_tensor = _device_byte_tensor(
                self.dependencies_address, self.device_ordinal
            )
        return self._dependencies_tensor

    @property
    def has_external(self) -> bool:
        return self._has_external


def _verify_module_digest(
    module_path: pathlib.Path, expected_sha256: str | None
) -> None:
    if expected_sha256 is None:
        return
    normalized = expected_sha256.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("expected JIT module SHA-256 is invalid")
    digest = hashlib.sha256(module_path.read_bytes()).hexdigest()
    if digest != normalized:
        raise RuntimeError("JIT module content does not match its activation digest")


def _read_operator_metadata(
    handle: _Handle, contract_reader: Any, plan_reader: Any
) -> tuple[OperatorContract, OperatorPlan]:
    native = _OperatorContract()
    _check(contract_reader(handle, ctypes.byref(native)))
    if native.magic != 0x4F41544E or native.struct_bytes != ctypes.sizeof(native):
        raise RuntimeError("JIT module returned an invalid operator contract")
    fingerprint = (
        int(native.source_fingerprint_low).to_bytes(8, "little")
        + int(native.source_fingerprint_high).to_bytes(8, "little")
    ).hex()
    contract = OperatorContract(
        int(native.schema_version),
        int(native.runtime_abi_version),
        OperatorFamily(native.family),
        OperatorForm(native.form),
        OperatorCapability(native.capabilities),
        fingerprint,
        OperatorInstrumentation(native.instrumentation_flags),
        OperatorIdentityBinding(native.identity_binding),
        OperatorDemandBinding(native.demand_binding),
        OperatorAccessProof(native.access_proof),
        int(native.granularity_bytes),
        int(native.tier_mask),
    )
    native_plan = _OperatorPlan()
    _check(plan_reader(handle, ctypes.byref(native_plan)))
    if native_plan.magic != 0x5041544E or native_plan.struct_bytes != ctypes.sizeof(
        native_plan
    ):
        raise RuntimeError("JIT module returned an invalid operator plan")
    source_fingerprint = (
        int(native_plan.source_fingerprint_low).to_bytes(8, "little")
        + int(native_plan.source_fingerprint_high).to_bytes(8, "little")
    ).hex()
    plan_fingerprint = (
        int(native_plan.plan_fingerprint_low).to_bytes(8, "little")
        + int(native_plan.plan_fingerprint_high).to_bytes(8, "little")
    ).hex()
    plan = OperatorPlan(
        int(native_plan.schema_version),
        int(native_plan.runtime_abi_version),
        OperatorFamily(native_plan.family),
        int(native_plan.supported_forms),
        OperatorCoordinateMap(native_plan.coordinate_map),
        OperatorPartialState(native_plan.partial_state),
        OperatorReduction(native_plan.reduction),
        OperatorPlanFlag(native_plan.flags),
        source_fingerprint,
        plan_fingerprint,
    )
    if (
        plan.family != contract.family
        or not plan.supports(contract.form)
        or plan.runtime_abi_version != contract.runtime_abi_version
        or plan.source_fingerprint != contract.source_fingerprint
        or plan.plan_fingerprint == "0" * 32
    ):
        raise RuntimeError("JIT operator plan does not match the module contract")
    return contract, plan


class JitOperatorModule(_Owner):
    """Verified compiler contract owned independently of transport phases."""

    _destroy = staticmethod(_operator_module_destroy)

    def __init__(
        self,
        shared_object: os.PathLike[str] | str,
        *,
        expected_sha256: str | None = None,
    ) -> None:
        self._handle = _Handle()
        module_path = pathlib.Path(shared_object).resolve()
        _verify_module_digest(module_path, expected_sha256)
        _check(
            _operator_module_create(
                os.fsencode(module_path), ctypes.byref(self._handle)
            )
        )
        try:
            self._operator_contract, self._operator_plan = _read_operator_metadata(
                self._handle, _operator_module_contract, _operator_module_plan
            )
        except BaseException:
            self.close()
            raise

    @property
    def operator_contract(self) -> OperatorContract:
        return self._operator_contract

    @property
    def operator_plan(self) -> OperatorPlan:
        return self._operator_plan


class JitPhaseProgram(_Owner):
    _destroy = staticmethod(_phase_destroy)

    def __init__(
        self,
        shared_object: os.PathLike[str] | str,
        *,
        expected_sha256: str | None = None,
    ):
        self._handle = _Handle()
        module_path = pathlib.Path(shared_object).resolve()
        _verify_module_digest(module_path, expected_sha256)
        path = os.fsencode(module_path)
        _check(_phase_create(path, ctypes.byref(self._handle)))
        try:
            self._operator_contract, self._operator_plan = _read_operator_metadata(
                self._handle, _phase_operator_contract, _phase_operator_plan
            )
        except BaseException:
            self.close()
            raise

    @property
    def operator_contract(self) -> OperatorContract:
        return self._operator_contract

    @property
    def operator_plan(self) -> OperatorPlan:
        return self._operator_plan

    def reset(
        self,
        runtime: Runtime,
        object_count: int,
        work_ticket_count: int,
        stream: Any = None,
    ) -> None:
        if object_count < 0 or object_count > runtime.config.object_capacity:
            raise ValueError("phase object count exceeds runtime capacity")
        if (
            work_ticket_count <= 0
            or work_ticket_count > runtime.config.work_ticket_capacity
        ):
            raise ValueError("phase work-ticket count exceeds runtime capacity")
        _check(
            _phase_reset(
                self._handle,
                runtime._handle,
                object_count,
                work_ticket_count,
                _stream_address(stream),
            )
        )

    def discover(
        self, runtime: Runtime, plan: DeviceWorkPlan, stream: Any = None
    ) -> None:
        if runtime.device_ordinal != plan.device_ordinal:
            raise ValueError("runtime and work plan must own the same CUDA device")
        _check(
            _phase_discover(
                self._handle,
                runtime._handle,
                plan.work_items_address,
                plan.dependencies_address,
                plan.work_item_count,
                _stream_address(stream),
            )
        )

    def discover_unqueued_host(
        self, runtime: Runtime, plan: DeviceWorkPlan, stream: Any = None
    ) -> None:
        """Discover a statically safe Host range without building an EDF heap."""

        if runtime.device_ordinal != plan.device_ordinal:
            raise ValueError("runtime and work plan must own the same CUDA device")
        _check(
            _phase_discover_unqueued_host(
                self._handle,
                runtime._handle,
                plan.work_items_address,
                plan.dependencies_address,
                plan.work_item_count,
                _stream_address(stream),
            )
        )

    def discover_ordered_nvme(
        self,
        runtime: Runtime,
        plan: DeviceWorkPlan,
        first_intent: int,
        intent_count: int,
        stream: Any = None,
    ) -> None:
        """Discover a finite NVMe window with validated static EDF order."""
        if runtime.device_ordinal != plan.device_ordinal:
            raise ValueError("runtime and work plan must own the same CUDA device")
        if (
            first_intent < 0
            or intent_count <= 0
            or first_intent + intent_count > runtime.config.intent_capacity
        ):
            raise ValueError("ordered NVMe intent range exceeds runtime capacity")
        _check(
            _phase_discover_ordered_nvme(
                self._handle,
                runtime._handle,
                plan.work_items_address,
                plan.dependencies_address,
                plan.work_item_count,
                first_intent,
                intent_count,
                _stream_address(stream),
            )
        )

    def invalidate_cached_objects(
        self,
        runtime: Runtime,
        first_object: int,
        object_count: int,
        stream: Any = None,
    ) -> None:
        _check(
            _phase_invalidate_cached_objects(
                self._handle,
                runtime._handle,
                first_object,
                object_count,
                _stream_address(stream),
            )
        )

    def validate_indexed_host_range(
        self,
        runtime: Runtime,
        first_object: int,
        object_count: int,
        stream: Any = None,
    ) -> None:
        _check(
            _phase_validate_indexed_host_range(
                self._handle,
                runtime._handle,
                first_object,
                object_count,
                _stream_address(stream),
            )
        )

    def warmup_indexed_host_validation(
        self, runtime: Runtime, stream: Any = None
    ) -> None:
        """Materialize the exact validation kernel without touching state."""

        _check(
            _phase_warmup_indexed_host_validation(
                self._handle,
                runtime._handle,
                _stream_address(stream),
            )
        )

    def rebind_indexed_host_pairs(
        self,
        runtime: Runtime,
        first_object: int,
        pair_count: int,
        key_source: int,
        key_staging: int,
        value_source: int,
        value_staging: int,
        stream: Any = None,
    ) -> None:
        _check(
            _phase_rebind_indexed_host_pairs(
                self._handle,
                runtime._handle,
                first_object,
                pair_count,
                key_source,
                key_staging,
                value_source,
                value_staging,
                _stream_address(stream),
            )
        )

    def progress_host(self, runtime: Runtime, blocks: int, stream: Any = None) -> None:
        _check(
            _phase_host(self._handle, runtime._handle, blocks, _stream_address(stream))
        )

    def prepare_ready_window(
        self, runtime: Runtime, maximum_work: int, stream: Any = None
    ) -> None:
        if maximum_work <= 0 or maximum_work > runtime.config.work_ticket_capacity:
            raise ValueError("runnable-window work bound exceeds runtime capacity")
        _check(
            _phase_prepare_ready_window(
                self._handle,
                runtime._handle,
                maximum_work,
                _stream_address(stream),
            )
        )

    def prepare_event_work_partition(
        self,
        runtime: Runtime,
        plan: DeviceWorkPlan,
        direct_work_count: int,
        stream: Any = None,
    ) -> None:
        """Publish one exact direct/deferred order for an event-owned mover."""

        if runtime.device_ordinal != plan.device_ordinal:
            raise ValueError("runtime and work plan must own the same CUDA device")
        if (
            direct_work_count <= 0
            or direct_work_count >= plan.work_item_count
            or plan.work_item_count > runtime.config.work_ticket_capacity
        ):
            raise ValueError("event work partition requires mixed bounded work")
        _check(
            _phase_prepare_event_work_partition(
                self._handle,
                runtime._handle,
                plan.work_items_address,
                plan.work_item_count,
                direct_work_count,
                _stream_address(stream),
            )
        )

    def progress_indexed_host_range(
        self,
        runtime: Runtime,
        first_object: int,
        object_count: int,
        stream: Any = None,
    ) -> None:
        _check(
            _phase_indexed_host_range(
                self._handle,
                runtime._handle,
                first_object,
                object_count,
                _stream_address(stream),
            )
        )

    def progress_validated_indexed_host_range(
        self,
        runtime: Runtime,
        first_object: int,
        object_count: int,
        stream: Any = None,
    ) -> None:
        _check(
            _phase_validated_indexed_host_range(
                self._handle,
                runtime._handle,
                first_object,
                object_count,
                _stream_address(stream),
            )
        )

    def set_indexed_row_counts(
        self,
        runtime: Runtime,
        first_object: int,
        object_count: int,
        row_count: int,
        stream: Any = None,
    ) -> None:
        """Bound the next validated copies to each object's rewritten prefix."""
        _check(
            _phase_set_indexed_row_counts(
                self._handle,
                runtime._handle,
                first_object,
                object_count,
                row_count,
                _stream_address(stream),
            )
        )

    def prepare_selected_indexed_rows(
        self,
        runtime: Runtime,
        first_object: int,
        object_count: int,
        selected_pages: Any,
        page_tokens: int,
        token_count: int,
        host_rows: Any,
        device_rows: Any,
        staged_pages: Any,
        source_indices: Any,
        staging_indices: Any,
        copied_rows: Any,
        stream: Any = None,
    ) -> None:
        """Compact and validate device-selected misses in stream order."""
        import torch

        arrays = (
            selected_pages,
            host_rows,
            device_rows,
            staged_pages,
            source_indices,
            staging_indices,
            copied_rows,
        )
        if any(not tensor.is_cuda for tensor in arrays):
            raise ValueError("selected indexed-row arrays must reside on CUDA")
        if selected_pages.dtype != torch.int64 or selected_pages.numel() == 0:
            raise ValueError("selected pages must be a non-empty int64 tensor")
        if any(
            tensor.dtype != torch.int32
            for tensor in (
                host_rows,
                device_rows,
                staged_pages,
                source_indices,
                staging_indices,
            )
        ):
            raise ValueError("selected indexed-row mappings must use int32")
        if copied_rows.dtype != torch.int64 or copied_rows.numel() != 1:
            raise ValueError("copied-row accounting must be one int64 value")
        if host_rows.numel() != token_count or device_rows.numel() != token_count:
            raise ValueError("selected row maps disagree with the token count")
        capacity = int(source_indices.numel())
        if (
            staging_indices.numel() != capacity
            or min(object_count, page_tokens, token_count, capacity) <= 0
        ):
            raise ValueError("selected indexed-row geometry is invalid")
        _check(
            _phase_prepare_selected_indexed_rows(
                self._handle,
                runtime._handle,
                first_object,
                object_count,
                selected_pages.data_ptr(),
                selected_pages.numel(),
                page_tokens,
                token_count,
                host_rows.data_ptr(),
                device_rows.data_ptr(),
                staged_pages.data_ptr(),
                source_indices.data_ptr(),
                staging_indices.data_ptr(),
                capacity,
                copied_rows.data_ptr(),
                _stream_address(stream),
            )
        )

    def prepare_bounded_selected_indexed_rows(
        self,
        runtime: Runtime,
        first_object: int,
        object_count: int,
        selected_pages: Any,
        page_tokens: int,
        token_count: int,
        host_rows: Any,
        device_rows: Any,
        cached_pages: Any,
        selected_rows: Any,
        source_indices: Any,
        staging_indices: Any,
        copied_rows: Any,
        stream: Any = None,
    ) -> None:
        """Emit a physical selected table and compact only staging misses."""
        import torch

        arrays = (
            selected_pages,
            host_rows,
            device_rows,
            cached_pages,
            selected_rows,
            source_indices,
            staging_indices,
            copied_rows,
        )
        if any(
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cuda"
            or not tensor.is_contiguous()
            for tensor in arrays
        ):
            raise ValueError(
                "bounded selected indexed-row arrays must be contiguous CUDA tensors"
            )
        if any(tensor.device != selected_pages.device for tensor in arrays):
            raise ValueError("bounded selected indexed-row arrays must share a device")
        if selected_pages.dtype != torch.int64 or selected_pages.numel() == 0:
            raise ValueError("selected pages must be a non-empty int64 tensor")
        if cached_pages.dtype != torch.int64 or cached_pages.numel() == 0:
            raise ValueError("cached pages must be a non-empty int64 tensor")
        if any(
            tensor.dtype != torch.int32
            for tensor in (
                host_rows,
                device_rows,
                selected_rows,
                source_indices,
                staging_indices,
            )
        ):
            raise ValueError("bounded selected row mappings must use int32")
        if copied_rows.dtype != torch.int64 or copied_rows.numel() != 1:
            raise ValueError("copied-row accounting must be one int64 value")
        capacity = int(source_indices.numel())
        if (
            min(object_count, page_tokens, token_count, capacity) <= 0
            or host_rows.numel() != token_count
            or device_rows.numel() != capacity
            or selected_rows.numel() != capacity
            or staging_indices.numel() != capacity
            or capacity % page_tokens != 0
            or cached_pages.numel() != capacity // page_tokens
            or selected_pages.numel() > cached_pages.numel()
        ):
            raise ValueError("bounded selected indexed-row geometry is invalid")
        _check(
            _phase_prepare_bounded_selected_indexed_rows(
                self._handle,
                runtime._handle,
                first_object,
                object_count,
                selected_pages.data_ptr(),
                selected_pages.numel(),
                page_tokens,
                token_count,
                host_rows.data_ptr(),
                device_rows.data_ptr(),
                cached_pages.data_ptr(),
                cached_pages.numel(),
                selected_rows.data_ptr(),
                source_indices.data_ptr(),
                staging_indices.data_ptr(),
                capacity,
                copied_rows.data_ptr(),
                _stream_address(stream),
            )
        )

    def reduce_mapped_indexed_key_pages(
        self,
        source: Any,
        row_indices: Any,
        token_count: int,
        page_tokens: int,
        output_min: Any,
        output_max: Any,
        stream: Any = None,
    ) -> None:
        """Reduce fragmented pinned host K rows into CUDA page envelopes."""
        import torch

        if (
            not isinstance(source, torch.Tensor)
            or source.device.type != "cpu"
            or not source.is_pinned()
            or source.ndim != 3
            or source.dtype not in (torch.float16, torch.bfloat16)
            or source.stride(2) != 1
            or source.stride(1) != int(source.shape[2])
            or source.stride(0) < int(source.shape[1]) * int(source.shape[2])
        ):
            # Rows may be strided views into one pool allocation; only the
            # per-row [heads, dim] payload must be dense — the kernel walks
            # rows by the passed byte stride.
            raise ValueError(
                "mapped key reduction requires pinned fp16/bf16 dense rows"
            )
        if (
            not isinstance(row_indices, torch.Tensor)
            or row_indices.device.type != "cuda"
            or row_indices.dtype != torch.int32
            or not row_indices.is_contiguous()
            or row_indices.numel() < token_count
        ):
            raise ValueError("mapped key reduction requires device int32 row indices")
        pages = (token_count + page_tokens - 1) // page_tokens
        expected = (pages, int(source.shape[1]), int(source.shape[2]))
        if (
            token_count <= 0
            or page_tokens <= 0
            or any(
                not isinstance(output, torch.Tensor)
                or output.device.type != "cuda"
                or output.dtype != torch.float32
                or tuple(output.shape) != expected
                or not output.is_contiguous()
                for output in (output_min, output_max)
            )
        ):
            raise ValueError("mapped key reduction geometry is invalid")
        _check(
            _phase_reduce_mapped_indexed_key_pages(
                self._handle,
                source.data_ptr(),
                source.shape[0],
                source.stride(0) * source.element_size(),
                row_indices.data_ptr(),
                token_count,
                page_tokens,
                source.shape[1],
                source.shape[2],
                0 if source.dtype == torch.float16 else 1,
                output_min.data_ptr(),
                output_max.data_ptr(),
                _stream_address(stream),
            )
        )

    def reduce_mapped_key_pages(
        self,
        source: Any,
        first_row: int,
        token_count: int,
        page_tokens: int,
        output_min: Any,
        output_max: Any,
        stream: Any = None,
    ) -> None:
        """Reduce pinned host K rows directly into CUDA page envelopes."""
        import torch

        if (
            not isinstance(source, torch.Tensor)
            or source.device.type != "cpu"
            or not source.is_pinned()
            or source.ndim != 3
            or source.dtype not in (torch.float16, torch.bfloat16)
            or not source.is_contiguous()
        ):
            raise ValueError(
                "mapped key reduction requires contiguous pinned fp16/bf16 rows"
            )
        pages = (token_count + page_tokens - 1) // page_tokens
        expected = (pages, int(source.shape[1]), int(source.shape[2]))
        if (
            min(first_row, token_count, page_tokens) < 0
            or token_count == 0
            or page_tokens == 0
            or first_row + token_count > int(source.shape[0])
            or any(
                not isinstance(output, torch.Tensor)
                or output.device.type != "cuda"
                or output.dtype != torch.float32
                or tuple(output.shape) != expected
                or not output.is_contiguous()
                for output in (output_min, output_max)
            )
        ):
            raise ValueError("mapped key reduction geometry is invalid")
        _check(
            _phase_reduce_mapped_key_pages(
                self._handle,
                source.data_ptr(),
                source.shape[0],
                source.stride(0) * source.element_size(),
                first_row,
                token_count,
                page_tokens,
                source.shape[1],
                source.shape[2],
                0 if source.dtype == torch.float16 else 1,
                output_min.data_ptr(),
                output_max.data_ptr(),
                _stream_address(stream),
            )
        )

    def progress_validated_indexed_host_range_parallel(
        self,
        runtime: Runtime,
        first_object: int,
        object_count: int,
        copy_blocks_per_group: int,
        stream: Any = None,
    ) -> None:
        _check(
            _phase_validated_indexed_host_range_parallel(
                self._handle,
                runtime._handle,
                first_object,
                object_count,
                copy_blocks_per_group,
                _stream_address(stream),
            )
        )

    def preload_host(
        self,
        runtime: Runtime,
        first_object: int,
        object_count: int,
        stream: Any = None,
    ) -> None:
        _check(
            _phase_preload_host(
                self._handle,
                runtime._handle,
                first_object,
                object_count,
                _stream_address(stream),
            )
        )

    def preload_host_pairs(
        self,
        runtime: Runtime,
        first_object: int,
        pair_count: int,
        stream: Any = None,
    ) -> None:
        _check(
            _phase_preload_host_pairs(
                self._handle,
                runtime._handle,
                first_object,
                pair_count,
                _stream_address(stream),
            )
        )

    def preload_host_pairs_ordered(
        self,
        runtime: Runtime,
        first_object: int,
        pair_count: int,
        worker_blocks: int,
        task_head: Any,
        stream: Any = None,
    ) -> None:
        """Launch one bounded persistent gather in directory/EDF order."""

        task_head_address = int(task_head.data_ptr())
        if pair_count <= 0 or worker_blocks <= 0 or task_head_address == 0:
            raise ValueError("ordered paired preload geometry is invalid")
        _check(
            _phase_preload_host_pairs_ordered(
                self._handle,
                runtime._handle,
                first_object,
                pair_count,
                worker_blocks,
                task_head_address,
                _stream_address(stream),
            )
        )

    def alias_preloaded_objects(
        self,
        runtime: Runtime,
        source_first: int,
        destination_first: int,
        object_count: int,
        object_id_base: int,
        version: int,
        stream: Any = None,
    ) -> None:
        _check(
            _phase_alias_preloaded(
                self._handle,
                runtime._handle,
                source_first,
                destination_first,
                object_count,
                object_id_base,
                version,
                _stream_address(stream),
            )
        )

    def progress_nvme(
        self,
        runtime: Runtime,
        issue_budget: int,
        completion_budget: int,
        stream: Any = None,
    ) -> None:
        _check(
            _phase_nvme(
                self._handle,
                runtime._handle,
                issue_budget,
                completion_budget,
                _stream_address(stream),
            )
        )

    def progress_nvme_until_idle(
        self,
        runtime: Runtime,
        issue_budget: int,
        completion_budget: int,
        timeout_ns: int,
        stream: Any = None,
    ) -> None:
        if timeout_ns <= 0:
            raise ValueError("NVMe progress timeout must be positive")
        _check(
            _phase_nvme_until_idle(
                self._handle,
                runtime._handle,
                issue_budget,
                completion_budget,
                timeout_ns,
                _stream_address(stream),
            )
        )

    def progress_nvme_ordered_until_idle(
        self,
        runtime: Runtime,
        first_intent: int,
        intent_count: int,
        issue_budget: int,
        completion_budget: int,
        timeout_ns: int,
        stream: Any = None,
    ) -> None:
        """Advance one finite EDF window until transport is idle."""
        if (
            first_intent < 0
            or intent_count <= 0
            or first_intent + intent_count > runtime.config.intent_capacity
        ):
            raise ValueError("ordered NVMe intent range exceeds runtime capacity")
        if timeout_ns <= 0:
            raise ValueError("NVMe progress timeout must be positive")
        _check(
            _phase_nvme_ordered_until_idle(
                self._handle,
                runtime._handle,
                first_intent,
                intent_count,
                issue_budget,
                completion_budget,
                timeout_ns,
                _stream_address(stream),
            )
        )

    def publish(
        self, runtime: Runtime, pending_budget: int, stream: Any = None
    ) -> None:
        _check(
            _phase_publish(
                self._handle,
                runtime._handle,
                pending_budget,
                _stream_address(stream),
            )
        )

    def complete(
        self, runtime: Runtime, work_ticket_count: int, stream: Any = None
    ) -> None:
        _check(
            _phase_complete(
                self._handle,
                runtime._handle,
                work_ticket_count,
                _stream_address(stream),
            )
        )

    def complete_stream_ordered(
        self, runtime: Runtime, plan: DeviceWorkPlan, stream: Any = None
    ) -> None:
        if runtime.device_ordinal != plan.device_ordinal:
            raise ValueError("runtime and work plan must own the same CUDA device")
        _check(
            _phase_complete_stream_ordered(
                self._handle,
                runtime._handle,
                plan.work_items_address,
                plan.work_item_count,
                _stream_address(stream),
            )
        )
