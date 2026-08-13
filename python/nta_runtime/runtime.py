"""Owning Python bindings for the NTA engine/runtime boundary."""

from __future__ import annotations

import ctypes
import ctypes.util
import dataclasses
import enum
import os
import pathlib
from collections.abc import Iterable
from typing import Any


API_VERSION = 30


class RuntimeError(Exception):
    """An error returned by the native NTA runtime."""


class Placement(enum.IntEnum):
    HBM = 0
    HOST_MAPPED = 1
    HOST_STAGED = 2


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

    def require(
        self,
        *,
        family: OperatorFamily,
        form: OperatorForm,
        capabilities: OperatorCapability = OperatorCapability(0),
    ) -> None:
        if self.family != family or self.form != form:
            raise RuntimeError(
                f"JIT operator contract is {self.family.name}/{self.form.name}, "
                f"expected {family.name}/{form.name}"
            )
        missing = capabilities & ~self.capabilities
        if missing:
            raise RuntimeError(
                f"JIT operator contract lacks capabilities {missing!s}"
            )


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
    direct: "JitPhaseProgram", incremental: "JitPhaseProgram"
) -> OperatorPlan:
    """Return the common typed plan or reject an independently generated pair."""

    direct_contract = direct.operator_contract
    incremental_contract = incremental.operator_contract
    if (
        direct_contract.form != OperatorForm.DIRECT
        or incremental_contract.form != OperatorForm.INCREMENTAL
        or direct_contract.family != incremental_contract.family
        or direct_contract.source_fingerprint
        != incremental_contract.source_fingerprint
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
        ("flags", ctypes.c_uint32),
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
        ("reserved0", ctypes.c_uint32),
        ("reserved1", ctypes.c_uint32),
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
        ("supports_hbm_peer", ctypes.c_uint32),
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


class _RequestSpec(ctypes.Structure):
    _fields_ = [
        ("request_id", ctypes.c_uint64),
        ("deadline_clock", ctypes.c_uint64),
        ("max_outstanding_bytes", ctypes.c_uint64),
        ("slot", ctypes.c_uint32),
        ("generation", ctypes.c_uint32),
        ("tenant_id", ctypes.c_uint32),
        ("priority", ctypes.c_uint32),
    ]


def _validate_abi_layouts() -> None:
    layouts = (
        ("OperatorContract", ctypes.sizeof(_OperatorContract), 48),
        ("OperatorPlan", ctypes.sizeof(_OperatorPlan), 72),
        ("AcquireRequirement", ctypes.sizeof(AcquireRequirement), 48),
        ("WorkItem", ctypes.sizeof(WorkItem), 64),
        ("RequestProgress", ctypes.sizeof(_RequestProgress), 96),
        ("RequestSpec", ctypes.sizeof(_RequestSpec), 40),
    )
    invalid = [
        f"{name}={observed} (expected {expected})"
        for name, observed, expected in layouts
        if observed != expected
    ]
    if invalid:
        raise RuntimeError("Python/native ABI layout mismatch: " + ", ".join(invalid))


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
class RequestSpec:
    slot: int
    request_id: int
    generation: int
    tenant_id: int = 0
    priority: int = 0
    deadline_clock: int = 0
    max_outstanding_bytes: int = (1 << 64) - 1

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


@dataclasses.dataclass(frozen=True)
class Replica:
    source_device_address: int
    placement: Placement
    tensor_map_address: int = 0
    estimated_latency_ns: int = 0
    estimated_bandwidth_bytes_per_second: int = 0

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
    preacquired: bool = False

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
            int(self.preacquired),
        )


@dataclasses.dataclass(frozen=True)
class NvmeOptions:
    endpoint: str
    device_ordinal: int = -1
    namespace_id: int = 1
    queue_depth: int = 64
    admin_timeout_ms: int = 10_000
    trust_read_only_device_code: bool = False


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
    supports_hbm_peer: bool
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
_runtime_create = _function(
    "nta_runtime_create",
    ctypes.c_int,
    ctypes.POINTER(_RuntimeConfig),
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
    ctypes.c_uint32,
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
_phase_prepare_claim_table_selected_rows = _function(
    "nta_jit_phase_prepare_claim_table_selected_rows",
    ctypes.c_int,
    _Handle,
    _Handle,
    *([ctypes.c_uint64] * 13),
    *([ctypes.c_uint32] * 6),
    ctypes.c_uint64,
)
_phase_build_compact_plan = _function(
    "nta_jit_phase_build_compact_plan",
    ctypes.c_int,
    _Handle,
    *([ctypes.c_uint64] * 8),
    ctypes.c_uint32,
    ctypes.c_uint64,
)
_phase_select_prepare_claim_rows = _function(
    "nta_jit_phase_select_prepare_claim_rows",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint64,
    *([ctypes.c_uint32] * 4),
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_int64,
    ctypes.c_uint32,
    ctypes.c_uint64,
    *([ctypes.c_uint32] * 2),
    *([ctypes.c_uint64] * 3),
    ctypes.c_uint32,
    *([ctypes.c_uint64] * 3),
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


def synchronize_stream(stream: Any = None) -> None:
    _check(_stream_synchronize(_stream_address(stream)))


def copy_host_to_device_async(
    destination: int, source: int, bytes: int, stream: Any = None
) -> None:
    if min(destination, source, bytes) <= 0:
        raise ValueError("host-to-device copy needs addresses and bytes")
    _check(
        _copy_host_to_device(
            destination, source, bytes, _stream_address(stream)
        )
    )


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
            bool(value.supports_hbm_peer),
            bool(value.translated_iommu),
            bool(value.namespace_read_only),
            bool(value.gpu_doorbell_mapping_validated),
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
        )


class Runtime(_Owner):
    _destroy = staticmethod(_runtime_destroy)

    def __init__(self, config: RuntimeConfig, nvme: NvmeTransport | None = None):
        self._handle = _Handle()
        native = config.native()
        nvme_handle = nvme._handle if nvme is not None else _Handle()
        _check(
            _runtime_create(
                ctypes.byref(native), nvme_handle, ctypes.byref(self._handle)
            )
        )
        self._nvme = nvme
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

    def set_tenant_budget(
        self, tenant_id: int, max_bytes: int, weight: int = 1
    ) -> None:
        _check(_runtime_set_tenant_budget(self._handle, tenant_id, max_bytes, weight))

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
    ) -> None:
        """Bulk-publish a contiguous layer's indexed host objects."""
        values = [object_.native() for object_ in objects]
        if not values:
            raise ValueError("indexed host object batch cannot be empty")
        array = (_IndexedHostObject * len(values))(*values)
        if stream is None:
            _check(
                _runtime_register_indexed_host_objects(
                    self._handle, first_slot, array, len(values)
                )
            )
        else:
            _check(
                _runtime_register_indexed_host_objects_async(
                    self._handle,
                    first_slot,
                    array,
                    len(values),
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

    def install_nvme_object(
        self,
        slot: int,
        object_id: int,
        version: int,
        source_byte_offset: int,
        bytes: int,
    ) -> int:
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
        assert self._pending is not None
        _, request_count = self._pending
        base = self._storage.data_ptr()
        stride = ctypes.sizeof(_RequestProgress)
        result = tuple(
            _request_progress_value(_RequestProgress.from_address(base + index * stride))
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
        _check(
            _plan_upload(
                self._handle,
                work_array,
                len(work),
                dependency_array,
                len(dependency),
                request_array,
                len(request),
                _stream_address(stream),
            )
        )
        self._has_external = any(
            item.direct_dependency_count != item.dependency_count for item in work
        )

    def wait_on(self, stream: Any) -> None:
        _check(_plan_wait(self._handle, _stream_address(stream)))

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


class JitPhaseProgram(_Owner):
    _destroy = staticmethod(_phase_destroy)

    def __init__(self, shared_object: os.PathLike[str] | str):
        self._handle = _Handle()
        path = os.fsencode(shared_object)
        _check(_phase_create(path, ctypes.byref(self._handle)))
        native = _OperatorContract()
        _check(_phase_operator_contract(self._handle, ctypes.byref(native)))
        if native.magic != 0x4F41544E or native.struct_bytes != ctypes.sizeof(native):
            self.close()
            raise RuntimeError("JIT module returned an invalid operator contract")
        fingerprint = (
            int(native.source_fingerprint_low).to_bytes(8, "little")
            + int(native.source_fingerprint_high).to_bytes(8, "little")
        ).hex()
        self._operator_contract = OperatorContract(
            int(native.schema_version),
            int(native.runtime_abi_version),
            OperatorFamily(native.family),
            OperatorForm(native.form),
            OperatorCapability(native.capabilities),
            fingerprint,
        )
        native_plan = _OperatorPlan()
        _check(_phase_operator_plan(self._handle, ctypes.byref(native_plan)))
        if native_plan.magic != 0x5041544E or native_plan.struct_bytes != ctypes.sizeof(
            native_plan
        ):
            self.close()
            raise RuntimeError("JIT module returned an invalid operator plan")
        source_fingerprint = (
            int(native_plan.source_fingerprint_low).to_bytes(8, "little")
            + int(native_plan.source_fingerprint_high).to_bytes(8, "little")
        ).hex()
        plan_fingerprint = (
            int(native_plan.plan_fingerprint_low).to_bytes(8, "little")
            + int(native_plan.plan_fingerprint_high).to_bytes(8, "little")
        ).hex()
        self._operator_plan = OperatorPlan(
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
            self._operator_plan.family != self._operator_contract.family
            or not self._operator_plan.supports(self._operator_contract.form)
            or self._operator_plan.runtime_abi_version
            != self._operator_contract.runtime_abi_version
            or self._operator_plan.source_fingerprint
            != self._operator_contract.source_fingerprint
            or self._operator_plan.plan_fingerprint == "0" * 32
        ):
            self.close()
            raise RuntimeError(
                "JIT operator plan does not match the module contract"
            )

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
        if staging_indices.numel() != capacity or min(
            object_count, page_tokens, token_count, capacity
        ) <= 0:
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

    def prepare_claim_table_selected_rows(
        self,
        runtime: Runtime,
        table: Any,
        local_layer: int,
        stream: Any = None,
    ) -> None:
        """One fixed-shape launch preps every valid claim-table row."""
        import torch

        tensors = (
            table.valid,
            table.object_slots,
            table.capacity_words,
            table.selected_counts,
            table.token_counts,
            table.selected_pages,
            table.cached_pages,
            table.host_rows,
            table.staging_rows,
            table.selected_rows,
            table.source_indices,
            table.staging_indices,
            table.copied_rows,
        )
        if any(
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cuda"
            or not tensor.is_contiguous()
            for tensor in tensors
        ):
            raise ValueError(
                "claim-table launch requires contiguous CUDA table tensors"
            )
        if not 0 <= int(local_layer) < int(table.layer_count):
            raise ValueError("claim-table layer is out of range")
        _check(
            _phase_prepare_claim_table_selected_rows(
                self._handle,
                runtime._handle,
                *(int(tensor.data_ptr()) for tensor in tensors),
                int(table.max_claims),
                int(table.max_budget_pages),
                int(table.layer_count),
                int(local_layer),
                int(table.max_claim_tokens),
                int(table.page_tokens),
                _stream_address(stream),
            )
        )

    def build_compact_plan(
        self,
        dense_indices: Any,
        dense_offsets: Any,
        bound_lengths: Any,
        nonprefix_offsets: Any,
        nonprefix_indices: Any,
        claim_row_counts: Any,
        compact_offsets: Any,
        compact_indices: Any,
        batch_size: int,
        stream: Any = None,
    ) -> None:
        """Build the packed compact plan from per-request descriptors."""
        import torch

        tensors = (
            dense_indices,
            dense_offsets,
            bound_lengths,
            nonprefix_offsets,
            nonprefix_indices,
            claim_row_counts,
            compact_offsets,
            compact_indices,
        )
        if any(
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cuda"
            or tensor.dtype != torch.int32
            or not tensor.is_contiguous()
            for tensor in tensors
        ):
            raise ValueError(
                "compact-plan build requires contiguous int32 CUDA tensors"
            )
        if batch_size <= 0:
            raise ValueError("compact-plan batch size must be positive")
        _check(
            _phase_build_compact_plan(
                self._handle,
                *(int(tensor.data_ptr()) for tensor in tensors),
                int(batch_size),
                _stream_address(stream),
            )
        )

    def select_prepare_claim_rows(
        self,
        runtime: Runtime,
        first_object: int,
        queries: Any,
        layer_min: Any,
        layer_max: Any,
        page_scores: Any,
        full_forced_pages: Any,
        tail_page: int,
        free_budget: int,
        ordered_pages_out: Any,
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
        """One launch: score, select, order, and prep a claim layer."""
        import torch

        if (
            queries.dtype != torch.float16
            or queries.ndim != 3
            or not queries.is_contiguous()
            or layer_min.dtype != torch.float32
            or layer_min.shape != layer_max.shape
            or layer_min.ndim != 3
        ):
            raise ValueError("fused selection requires fp16 queries and fp32 envelopes")
        page_count = int(layer_min.shape[0])
        capacity = int(selected_rows.numel())
        cache_slots = int(cached_pages.numel())
        _check(
            _phase_select_prepare_claim_rows(
                self._handle,
                runtime._handle,
                int(first_object),
                int(queries.data_ptr()),
                int(queries.shape[0]),
                int(queries.shape[1]),
                int(layer_min.shape[1]),
                int(layer_min.shape[2]),
                int(layer_min.data_ptr()),
                int(layer_max.data_ptr()),
                int(page_scores.data_ptr()),
                page_count,
                int(full_forced_pages.data_ptr()),
                int(full_forced_pages.numel()),
                int(tail_page),
                int(free_budget),
                int(ordered_pages_out.data_ptr()),
                int(page_tokens),
                int(token_count),
                int(host_rows.data_ptr()),
                int(device_rows.data_ptr()),
                int(cached_pages.data_ptr()),
                cache_slots,
                int(selected_rows.data_ptr()),
                int(source_indices.data_ptr()),
                int(staging_indices.data_ptr()),
                capacity,
                int(copied_rows.data_ptr()),
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
            raise ValueError(
                "mapped key reduction requires device int32 row indices"
            )
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
