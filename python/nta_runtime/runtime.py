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


API_VERSION = 10


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
    ]


def _validate_abi_layouts() -> None:
    layouts = (
        ("AcquireRequirement", ctypes.sizeof(AcquireRequirement), 48),
        ("WorkItem", ctypes.sizeof(WorkItem), 64),
        ("RequestProgress", ctypes.sizeof(_RequestProgress), 64),
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
    enable_cta_nvme_try_issue: bool = True
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

    @property
    def complete(self) -> bool:
        return (
            self.expected_work != 0
            and self.completed_work == self.expected_work
            and self.failed_work == 0
            and self.cancelled_work == 0
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
_dlpack_destroy = _function(
    "nta_dlpack_managed_tensor_destroy", None, ctypes.c_void_p
)
_stream_synchronize = _function(
    "nta_stream_synchronize", ctypes.c_int, ctypes.c_uint64
)
_phase_create = _function(
    "nta_jit_phase_program_create", ctypes.c_int, ctypes.c_char_p, _HandlePointer
)
_phase_destroy = _function("nta_jit_phase_program_destroy", None, _Handle)
_phase_reset = _function(
    "nta_jit_phase_reset",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint32,
    ctypes.c_uint32,
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
_phase_host = _function(
    "nta_jit_phase_progress_host",
    ctypes.c_int,
    _Handle,
    _Handle,
    ctypes.c_uint32,
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
            _runtime_work_ticket_state(
                self._handle, work_ticket, ctypes.byref(value)
            )
        )
        return int(value.value)

    def work_runnable_ns(self, work_ticket_count: int) -> tuple[int, ...]:
        if work_ticket_count <= 0:
            raise ValueError("work ticket count must be positive")
        values = (ctypes.c_uint64 * work_ticket_count)()
        _check(
            _runtime_work_runnable_ns(
                self._handle, work_ticket_count, values
            )
        )
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

    def request_progress(self, request_slot: int) -> RequestProgress:
        value = _RequestProgress()
        _check(
            _runtime_request_progress(
                self._handle, request_slot, ctypes.byref(value)
            )
        )
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
        )

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
        return tuple(
            RequestProgress(
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
            )
            for value in values
        )


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

    def progress_host(self, runtime: Runtime, blocks: int, stream: Any = None) -> None:
        _check(
            _phase_host(self._handle, runtime._handle, blocks, _stream_address(stream))
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
