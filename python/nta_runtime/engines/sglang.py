"""SGLang 0.5.14 adapter for compiler-instrumented FlashInfer attention."""

from __future__ import annotations

import atexit
from collections import Counter
from dataclasses import dataclass, field
import json
import logging
import math
import operator
import os
import pathlib
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any

import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.memory_pool import KVWriteLoc

from nta_runtime.flashinfer import (
    BIND_CURRENT_GENERATION,
    FlashInferLayerEpoch,
    PREACQUIRED,
    attention_jit_args,
    enqueue_resident_attention,
    request_bound_attention_jit_args,
)
from nta_runtime.flashinfer_schedule import (
    Schedule,
    decode_schedule,
    paged_prefill_schedule,
    require_supported_version,
)
from nta_runtime.adapters.base import EngineBatch
from nta_runtime.adapters.sglang import SglangAdapter, SglangExecutionConfig
from nta_runtime.execution_core import ExecutionSession, ExecutionTile
from nta_runtime.execution_protocol import ProtocolKind
from nta_runtime.execution_planner import (
    conservative_resume_counts,
    HostCostModel,
    HostExecutionPlan,
    indexed_copy_blocks_per_group,
    plan_host_execution,
)
from nta_runtime.opportunity import OperatorArrival, TileArrival, append_json_line
from nta_runtime.requests import RequestBinding
from nta_runtime.engines.sglang_hicache import PendingHostLoad, SglangHiCacheBridge
from nta_runtime.runtime_resources import (
    RuntimeResourceConfig,
    ServingRuntimeResources,
)
from nta_runtime.tier import ServingTierConfig
from nta_runtime.runtime import (
    AcquireRequirement,
    DeviceWorkPlan,
    IndexedHostObject,
    JitPhaseProgram,
    OperatorCapability,
    OperatorAccessProof,
    OperatorContract,
    OperatorCoordinateMap,
    OperatorFamily,
    OperatorForm,
    OperatorDemandBinding,
    OperatorIdentityBinding,
    OperatorInstrumentation,
    OperatorPartialState,
    OperatorPlan,
    OperatorPlanFlag,
    OperatorReduction,
    RequestRange,
    TierKind,
    require_operator_pair,
)


_OBJECT_ID_BASE = 0x4E54410000000000
_LOOKAHEAD_ALIAS_ID_BASE = _OBJECT_ID_BASE | 0x00000000FFFF0000
_LOOKAHEAD_VERSION = 1
_MAX_ABI_BYTES = (1 << 32) - 1
# Per-forward timing, populated by the plugin's forward hooks. These samples
# are keyed by batch composition and measure the actual serving boundary seen
# by co-resident decode, rather than a transfer lifetime spanning forwards.
FORWARD_PROFILE: dict[str, float] = {}


def record_forward(kind: str, milliseconds: float) -> None:
    """Accumulate count/total/max for one forward-kind sample."""
    FORWARD_PROFILE[f"forward_{kind}_count"] = (
        FORWARD_PROFILE.get(f"forward_{kind}_count", 0.0) + 1.0
    )
    FORWARD_PROFILE[f"forward_{kind}_ms_total"] = (
        FORWARD_PROFILE.get(f"forward_{kind}_ms_total", 0.0) + milliseconds
    )
    FORWARD_PROFILE[f"forward_{kind}_ms_max"] = max(
        FORWARD_PROFILE.get(f"forward_{kind}_ms_max", 0.0), milliseconds
    )


# Incremented by the plugin's PrefillCudaGraphRunner patches (same scheduler
# process); exported through _stats_report so artifacts attest whether the
# breakable prefill graphs actually served batches or only captured.
PREFILL_GRAPH_COUNTERS: dict[str, int] = {
    "prefill_graph_served_batches": 0,
    "prefill_graph_capture_batches": 0,
}
logger = logging.getLogger(__name__)
_PagePair = tuple[tuple[int, ...], tuple[int, ...]]


class _StatsPublisher:
    """Coalesce evaluation snapshots and write them off the scheduler thread."""

    def __init__(self, path: pathlib.Path) -> None:
        self._path = path
        self._condition = threading.Condition()
        self._pending: tuple[int, dict[str, Any]] | None = None
        self._submitted = 0
        self._completed = 0
        self._error: Exception | None = None
        self._thread = threading.Thread(
            target=self._run, name="nta-stats-publisher", daemon=True
        )
        self._thread.start()

    def publish(self, report: dict[str, Any], *, wait: bool = False) -> None:
        with self._condition:
            self._submitted += 1
            sequence = self._submitted
            self._pending = (sequence, report)
            self._condition.notify()
            if wait:
                self._condition.wait_for(
                    lambda: self._completed >= sequence or self._error is not None
                )
                if self._error is not None:
                    raise RuntimeError(
                        "failed to publish NTA engine statistics"
                    ) from self._error

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending is not None)
                sequence, report = self._pending
                self._pending = None
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self._path.with_suffix(self._path.suffix + ".tmp")
                temporary.write_text(
                    json.dumps(report, sort_keys=True) + "\n", encoding="utf-8"
                )
                temporary.replace(self._path)
            except Exception as error:
                with self._condition:
                    self._error = error
                    self._condition.notify_all()
                return
            with self._condition:
                self._completed = sequence
                self._condition.notify_all()


@dataclass(frozen=True)
class _PrefetchedLayer:
    key_slot: int
    key_object_id: int
    value_slot: int
    value_object_id: int
    version: int
    key_bytes: int
    value_bytes: int
    ready_event: torch.cuda.Event
    transfer_first_slot: int


@dataclass(frozen=True)
class _FragmentLookahead:
    layer_id: int
    wrapper_id: int
    object_count: int
    preloaded_object_count: int
    key_source: int
    key_staging: int
    value_source: int
    value_staging: int
    ready_event: torch.cuda.Event


@dataclass
class _ActiveBatch:
    bindings: tuple[RequestBinding, ...]
    schedules: dict[int, Schedule]
    pending_host_load: PendingHostLoad | None
    page_pairs: dict[int, tuple[_PagePair, ...]]
    index_maps: dict[_PagePair, tuple[torch.Tensor, torch.Tensor]]
    prefetched_layers: dict[int, _PrefetchedLayer]
    prefetch_tensors: tuple[torch.Tensor, ...]
    host_execution: HostExecutionPlan | None = None
    grouping: str = "request"
    fragment_lookahead: dict[int, _FragmentLookahead] = field(default_factory=dict)
    execution: ExecutionSession | None = None


@dataclass
class _PlanAllocation:
    plan: DeviceWorkPlan
    work_capacity: int
    signature: tuple[Any, ...] | None = None
    object_count: int = 0
    index_tensors: tuple[torch.Tensor, ...] = ()
    host_execution: HostExecutionPlan | None = None
    object_version: int = 0
    transfer_bytes: int = 0
    indexed_geometry: tuple[int, ...] | None = None
    max_object_fanout: int = 1
    min_unresolved_dependencies: int = 1
    direct_work_count: int = 0
    external_object_slots: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class _DemandGraph:
    graph: torch.cuda.CUDAGraph
    query: torch.Tensor
    output: torch.Tensor
    retained_events: tuple[torch.cuda.Event, ...]
    wrapper_metadata: tuple[tuple[str, torch.Tensor], ...]


@dataclass(frozen=True)
class _DemandGraphKey:
    operator_family: str
    wrapper_id: int
    layer_id: int
    work_items_address: int
    dependencies_address: int
    runtime_address: int
    work_count: int
    object_count: int
    progress_blocks: tuple[int, ...]
    ready_work_counts: tuple[int, ...]
    initial_ready_work_count: int
    indexed_copy_blocks_per_group: int
    query_shape: tuple[int, ...]
    query_stride: tuple[int, ...]
    query_dtype: str
    query_device: str
    key_cache_address: int
    value_cache_address: int
    sm_scale: float
    k_scale: float | None
    v_scale: float | None
    causal: bool
    window_left: int
    wrapper_plan: Any
    wrapper_metadata_layout: tuple[
        tuple[str, tuple[int, ...], tuple[int, ...], str, str], ...
    ]


_GRAPH_WRAPPER_METADATA = (
    "_qo_indptr_buf",
    "_paged_kv_indptr_buf",
    "_paged_kv_indices_buf",
    "_paged_kv_last_page_len_buf",
    "_custom_mask_buf",
    "_mask_indptr_buf",
    "_prefix_len_ptr",
    "_token_pos_in_items_ptr",
    "_max_item_len_ptr",
    "_block_tables",
)


def _freeze_graph_plan(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    try:
        return operator.index(value)
    except TypeError:
        pass
    if isinstance(value, Iterable):
        return tuple(_freeze_graph_plan(item) for item in value)
    raise RuntimeError(
        f"FlashInfer graph plan contains unsupported {type(value).__name__} state"
    )


def _graph_wrapper_metadata(wrapper: Any) -> tuple[tuple[str, torch.Tensor], ...]:
    return tuple(
        (name, value)
        for name in _GRAPH_WRAPPER_METADATA
        if torch.is_tensor(value := getattr(wrapper, name, None))
    )


def _graph_wrapper_metadata_layout(
    wrapper: Any,
) -> tuple[tuple[str, tuple[int, ...], tuple[int, ...], str, str], ...]:
    return tuple(
        (
            name,
            tuple(int(extent) for extent in value.shape),
            tuple(int(stride) for stride in value.stride()),
            str(value.dtype),
            str(value.device),
        )
        for name, value in _graph_wrapper_metadata(wrapper)
    )


def _demand_graph_key(
    *,
    operator_family: str,
    wrapper: Any,
    layer_id: int,
    plan: DeviceWorkPlan,
    runtime_tensor: torch.Tensor,
    work_count: int,
    object_count: int,
    progress_blocks: tuple[int, ...],
    ready_work_counts: tuple[int, ...],
    initial_ready_work_count: int,
    indexed_copy_blocks_per_group: int,
    query: torch.Tensor,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
    sm_scale: float,
    k_scale: float | None,
    v_scale: float | None,
    causal: bool,
    window_left: int,
) -> _DemandGraphKey:
    """Describe every dynamic value baked into a demand graph launch."""
    if operator_family not in {"decode", "paged_prefill"}:
        raise ValueError("unsupported demand graph operator family")
    return _DemandGraphKey(
        operator_family,
        id(wrapper),
        int(layer_id),
        int(plan.work_items_address),
        int(plan.dependencies_address),
        int(runtime_tensor.data_ptr()),
        int(work_count),
        int(object_count),
        tuple(int(count) for count in progress_blocks),
        tuple(int(count) for count in ready_work_counts),
        int(initial_ready_work_count),
        int(indexed_copy_blocks_per_group),
        tuple(int(extent) for extent in query.shape),
        tuple(int(stride) for stride in query.stride()),
        str(query.dtype),
        str(query.device),
        int(kv_cache[0].data_ptr()),
        int(kv_cache[1].data_ptr()),
        float(sm_scale),
        None if k_scale is None else float(k_scale),
        None if v_scale is None else float(v_scale),
        bool(causal),
        int(window_left),
        _freeze_graph_plan(getattr(wrapper, "_plan_info", None)),
        _graph_wrapper_metadata_layout(wrapper),
    )


def _positive_environment(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _tenant_budget_specs() -> tuple[tuple[int, int, int], ...]:
    """Parse the optional process-level tenant quota policy once at startup.

    The runtime keeps an unlimited, active default for tenants that are not
    listed.  A deployment that needs isolation opts in with
    ``NTA_TENANT_BUDGETS=id:bytes[:weight],...``.  This keeps policy out of the
    per-forward path and makes a missing engine tenant annotation fail through
    the normal tenant-0 contract instead of silently changing quotas.
    """
    raw = os.environ.get("NTA_TENANT_BUDGETS", "").strip()
    if not raw:
        return ()
    specs: list[tuple[int, int, int]] = []
    seen: set[int] = set()
    for item in raw.split(","):
        fields = tuple(field.strip() for field in item.split(":") if field.strip())
        if len(fields) not in (2, 3):
            raise ValueError("NTA_TENANT_BUDGETS entries must be id:bytes[:weight]")
        tenant_id, max_bytes = (int(fields[0]), int(fields[1]))
        weight = 1 if len(fields) == 2 else int(fields[2])
        if tenant_id < 0 or max_bytes < 0 or weight <= 0:
            raise ValueError("NTA_TENANT_BUDGETS contains an invalid value")
        if tenant_id in seen:
            raise ValueError("NTA_TENANT_BUDGETS repeats a tenant")
        seen.add(tenant_id)
        specs.append((tenant_id, max_bytes, weight))
    return tuple(specs)


def _nonnegative_environment(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value


def _mover_stream_priority() -> int:
    value = int(os.environ.get("NTA_RUNTIME_MOVER_STREAM_PRIORITY", "0"))
    if value > 0:
        raise ValueError(
            "NTA_RUNTIME_MOVER_STREAM_PRIORITY must be zero or negative because "
            "CUDA stream priorities are non-positive"
        )
    return value


def _gain_environment(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if value < 1.0:
        raise ValueError(f"{name} must be at least one")
    return value


def _frontier_transfer_bytes(pending: PendingHostLoad) -> int:
    key_layers = pending.controller.mem_pool_host.k_data_refs
    value_layers = pending.controller.mem_pool_host.v_data_refs
    if len(key_layers) != len(value_layers):
        raise RuntimeError("HiCache host K/V layer counts disagree")
    page_count = int(pending.host_indices.numel())
    if page_count <= 0:
        raise RuntimeError("HiCache acquisition frontier has no host pages")
    return page_count * sum(
        int(key[0].numel()) * key.element_size()
        + int(value[0].numel()) * value.element_size()
        for key, value in zip(key_layers, value_layers, strict=True)
    )


def _pipeline_object_range(
    object_capacity: int, consumer_index: int, layer_count: int
) -> tuple[int, int]:
    """Reserve one producer's layer objects from the directory's high end."""
    if object_capacity <= 0 or consumer_index < 0 or layer_count <= 0:
        raise RuntimeError("HiCache layer-object geometry is invalid")
    object_count = 2 * layer_count
    end = object_capacity - consumer_index * object_count
    begin = end - object_count
    if begin < 2 or end > object_capacity:
        raise RuntimeError("HiCache layer objects exceed NTA directory capacity")
    return begin, end


def _dtype_tag(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.").replace("_", "")


def _plan_cache_signature(
    request_indices: tuple[int, ...],
    kv_tile_indices: tuple[int, ...],
    page_pairs: tuple[_PagePair, ...],
    request_slots: tuple[int, ...],
    key_bytes: int,
    value_bytes: int,
    prefetched_bytes: tuple[int, int] | None,
) -> tuple[Any, ...]:
    """Return the immutable identity of a reusable device-side plan.

    Request generations are dynamic directory state. Demand launches bind the
    current generation on device, so generation reuse must not rebuild the
    structural work and dependency arrays.
    """
    return (
        request_indices,
        kv_tile_indices,
        page_pairs,
        request_slots,
        key_bytes,
        value_bytes,
        prefetched_bytes,
    )


def _group_external_pages_by_request(
    schedule: Schedule, page_pairs: tuple[_PagePair, ...]
) -> tuple[_PagePair, ...]:
    """Share one exact indexed K/V acquisition group across request CTAs."""
    if schedule.work_count != len(page_pairs):
        raise RuntimeError("FlashInfer work and page-pair counts disagree")
    pages_by_request: dict[int, dict[int, int]] = {}
    for request_index, (host_pages, device_pages) in zip(
        schedule.request_indices, page_pairs
    ):
        if len(host_pages) != len(device_pages):
            raise RuntimeError("HiCache host/device page mappings disagree")
        request_pages = pages_by_request.setdefault(request_index, {})
        for host_page, device_page in zip(host_pages, device_pages):
            previous = request_pages.setdefault(device_page, host_page)
            if previous != host_page:
                raise RuntimeError(
                    "one device KV page maps to multiple host cache pages"
                )

    grouped = {
        request_index: (tuple(pages.values()), tuple(pages))
        for request_index, pages in pages_by_request.items()
        if pages
    }
    return tuple(
        grouped.get(request_index, ((), ())) if host_pages else ((), ())
        for request_index, (host_pages, _device_pages) in zip(
            schedule.request_indices, page_pairs
        )
    )


def _request_ranges(
    bindings: tuple[RequestBinding, ...], request_indices: tuple[int, ...]
) -> list[RequestRange]:
    """Build native ranges only for schedules grouped by request.

    ``DeviceWorkPlan`` requires one contiguous range per request because the
    reduction metadata and contributor ordinals are request-relative. A
    malformed or future FlashInfer schedule must fail closed here instead of
    producing zero-length or misbound ranges.
    """
    ranges: list[RequestRange] = []
    cursor = 0
    for binding in bindings:
        begin = cursor
        while (
            cursor < len(request_indices)
            and request_indices[cursor] == binding.request_index
        ):
            cursor += 1
        if cursor == begin:
            raise RuntimeError(
                f"FlashInfer schedule has no work for request {binding.request_index}"
            )
        ranges.append(
            RequestRange(
                begin,
                cursor - begin,
                binding.request_slot,
                binding.generation,
            )
        )
    if cursor != len(request_indices):
        raise RuntimeError("FlashInfer schedule is not grouped contiguously by request")
    return ranges


class NtaFlashInferAttnBackend(FlashInferAttnBackend):
    """FA2 backend carrying request semantics into every attention CTA."""

    def __init__(
        self,
        model_runner: Any,
        skip_prefill: bool = False,
        kv_indptr_buf: torch.Tensor | None = None,
        kv_last_page_len_buf: torch.Tensor | None = None,
        init_new_workspace: bool = False,
    ) -> None:
        require_supported_version()
        if model_runner.server_args.speculative_algorithm is not None:
            raise ValueError(
                "NTA's SGLang adapter does not support speculative decoding"
            )
        if model_runner.kv_cache_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "NTA's SGLang adapter currently supports float16 or bfloat16 KV"
            )
        super().__init__(
            model_runner,
            skip_prefill=skip_prefill,
            kv_indptr_buf=kv_indptr_buf,
            kv_last_page_len_buf=kv_last_page_len_buf,
            init_new_workspace=init_new_workspace,
        )
        if self.prefill_backend != "fa2" or self.decode_backend != "fa2":
            raise ValueError("NTA requires FlashInfer's FA2 attention kernels")

        # Keep the stock wrappers as an explicit resident reference.  NTA's
        # typed work-unit kernel is needed only when a forward contains an
        # external tier dependency; routing resident-only forwards through the
        # framework wrapper prevents instrumentation overhead from becoming a
        # regression for requests that do not exercise the mechanism.
        self._stock_decode_wrappers = tuple(self.decode_wrappers)
        self._stock_prefill_paged_wrappers = tuple(self.prefill_wrappers_paged)
        self._stock_prefill_verify_wrappers = tuple(self.prefill_wrappers_verify)

        self._hicache_enabled = bool(model_runner.server_args.enable_hierarchical_cache)
        self._model_runner = model_runner
        self._decode_jit_args: list[Any] | None = None
        self._prefill_jit_args: list[Any] | None = None
        self._install_instrumented_wrappers(model_runner, skip_prefill)

        request_capacity = int(model_runner.req_to_token_pool.req_to_token.shape[0])
        default_tickets = max(4096, request_capacity * 8)
        self._work_ticket_capacity = _positive_environment(
            "NTA_RUNTIME_MAX_WORK_TICKETS", default_tickets
        )
        self._object_capacity = 2 * self._work_ticket_capacity
        self._tenant_capacity = _positive_environment(
            "NTA_TENANT_CAPACITY", request_capacity
        )
        tenant_specs = _tenant_budget_specs()
        for tenant_id, _, _ in tenant_specs:
            if tenant_id >= self._tenant_capacity:
                raise RuntimeError(
                    f"tenant {tenant_id} exceeds NTA_TENANT_CAPACITY="
                    f"{self._tenant_capacity}"
                )
        try:
            self._execution_config = SglangExecutionConfig.from_environment()
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        resources: ServingRuntimeResources | None = None
        try:
            tier_config = ServingTierConfig.from_environment()
            resources = ServingRuntimeResources.open(
                tier_config=tier_config,
                runtime_config=RuntimeResourceConfig.with_environment_staging_limit(
                    request_capacity=request_capacity,
                    object_capacity=self._object_capacity,
                    intent_capacity=self._object_capacity,
                    work_ticket_capacity=self._work_ticket_capacity,
                    max_dependencies_per_work_ticket=2,
                    device_ordinal=torch.cuda.current_device(),
                    tenant_capacity=self._tenant_capacity,
                ),
            )
            if not resources.tier.is_host and not self._hicache_enabled:
                raise RuntimeError(
                    "a physical serving tier requires SGLang hierarchical cache metadata"
                )
        except (OSError, ValueError, RuntimeError) as error:
            if resources is not None:
                resources.close()
            raise RuntimeError(
                f"invalid NTA serving tier configuration: {error}"
            ) from error
        assert resources is not None
        self._resources = resources
        self._tier_service = resources.tier
        self._runtime = resources.runtime
        self._closed = False
        self._resources_closed = False
        self._configure_tenant_budgets(tenant_specs)
        self._request_adapter = SglangAdapter(self._runtime, request_capacity)
        self._hicache = SglangHiCacheBridge(
            self.token_to_kv_pool, work_capacity=max(4096, request_capacity * 4)
        )
        # CUDA priorities are inverted: acquisition movers always use the
        # lowest priority so they cannot preempt decode.
        mover_priority = _mover_stream_priority()
        self._prefetch_stream = torch.cuda.Stream(priority=mover_priority)
        self._progress_stream = torch.cuda.Stream(priority=mover_priority)
        self._host_cost_model = HostCostModel.from_environment()
        self._indexed_copy_target_bytes = _positive_environment(
            "NTA_EXECUTION_INDEXED_COPY_BYTES_PER_CTA", 1024 * 1024
        )
        self._indexed_copy_max_blocks = min(
            64,
            _positive_environment("NTA_EXECUTION_INDEXED_COPY_MAX_CTAS", 32),
        )
        self._frontier_layers_per_wave = min(
            64,
            _positive_environment("NTA_EXECUTION_FRONTIER_LAYERS_PER_WAVE", 4),
        )
        self._prefetch_enabled = (
            self._tier_service.is_host
            and self._hicache_enabled
            and self._execution_config.prefetch
        )
        self._incremental_enabled = (
            self._execution_config.protocol.kind is not ProtocolKind.CONVENTIONAL
        )
        self._overlap_enabled = self._execution_config.protocol.allow_overlap
        self._frontier_enabled = self._tier_service.is_host and self._overlap_enabled
        self._fragment_enabled = (
            self._tier_service.is_host
            and self._overlap_enabled
            and not self._prefetch_enabled
        )
        self._grouping = self._execution_config.grouping
        self._prefetch_ready_events: tuple[tuple[torch.cuda.Event, ...], ...] = ()
        self._bulk_events: tuple[torch.cuda.Event, ...] = ()
        layer_count = getattr(model_runner.model_config, "num_hidden_layers", None)
        if layer_count is None:
            layer_count = getattr(
                model_runner.model_config.hf_config, "num_hidden_layers"
            )
        self._model_layer_count = int(layer_count)
        self._model_start_layer = int(getattr(self.token_to_kv_pool, "start_layer", 0))
        self._cuda_graph_mode = False
        self._stock_forward = False
        self._execution_epoch = 0
        self._current_engine_batch: EngineBatch | None = None
        self._active_batch: _ActiveBatch | None = None
        self._plans: dict[tuple[int, int], _PlanAllocation] = {}
        self._phase_programs: dict[str, JitPhaseProgram] = {}
        self._operator_contracts: dict[
            tuple[OperatorFamily, OperatorForm], OperatorContract
        ] = {}
        self._operator_plans: dict[
            tuple[OperatorFamily, OperatorForm], OperatorPlan
        ] = {}
        self._operator_programs: dict[
            tuple[OperatorFamily, OperatorForm], JitPhaseProgram
        ] = {}
        self._demand_sync_events: dict[
            tuple[int, int, int], tuple[torch.cuda.Event, tuple[torch.cuda.Event, ...]]
        ] = {}
        self._demand_graphs: dict[_DemandGraphKey, _DemandGraph] = {}
        self._demand_graph_warmups: dict[_DemandGraphKey, None] = {}
        self._demand_graph_enabled = (
            self._tier_service.is_host
            and os.environ.get("NTA_EXECUTION_GRAPH", "1") != "0"
        )
        self._demand_graph_capacity = _positive_environment(
            "NTA_EXECUTION_GRAPH_CAPACITY", max(64, 4 * self._model_layer_count)
        )
        self._stats = {
            "schema": 1,
            "engine": "sglang",
            "backend": "nta_flashinfer",
            "execution_protocol": self._execution_config.protocol.kind.value,
            "work_granularity": self._execution_config.protocol.granularity.value,
            "protocol_max_inflight_units": self._execution_config.protocol.max_inflight_units,
            "runtime_tenant_capacity": self._resources.config.tenant_capacity,
            "runtime_staging_byte_capacity": self._resources.config.staging_byte_capacity,
            "execution_protocol_status": "native_work_unit",
            "execution_demand_semantics": "exact",
            "execution_session_scope": "attention_launch",
            "revision": os.environ.get("NTA_REVISION", "unknown"),
            "pid": os.getpid(),
            "prefetch_enabled": self._prefetch_enabled,
            "overlap_enabled": self._overlap_enabled,
            "frontier_enabled": self._frontier_enabled,
            "fragment_enabled": self._fragment_enabled,
            "sglang_mixed_chunk_enabled": bool(
                model_runner.server_args.enable_mixed_chunk
            ),
            "max_host_rounds": self._host_cost_model.max_rounds,
            "minimum_predicted_gain": self._host_cost_model.minimum_predicted_gain,
            "incremental_setup_ns": self._host_cost_model.incremental_setup_ns,
            "indexed_copy_target_bytes": self._indexed_copy_target_bytes,
            "indexed_copy_max_blocks": self._indexed_copy_max_blocks,
            "frontier_layers_per_wave": self._frontier_layers_per_wave,
            "batches": 0,
            "decode_launches": 0,
            "prefill_launches": 0,
            "cta_work_items": 0,
            "plan_uploads": 0,
            "request_rebindings": 0,
            "request_cancellations": 0,
            "external_launches": 0,
            "resident_reference_batches": 0,
            "hicache_external_batches": 0,
            "hicache_fallback_batches": 0,
            "indexed_host_objects": 0,
            "request_acquisition_groups": 0,
            "tile_acquisition_groups": 0,
            "adaptive_request_batches": 0,
            "adaptive_tile_batches": 0,
            "indexed_host_bytes": 0,
            "prefetched_layers": 0,
            "prefetched_host_bytes": 0,
            "lookahead_acquisition_layers": 0,
            "lookahead_acquisition_objects": 0,
            "lookahead_bound_launches": 0,
            "demand_host_layers": 0,
            "incremental_host_layers": 0,
            "request_overlap_layers": 0,
            "mixed_dependency_layers": 0,
            "mixed_forward_batches": 0,
            "mixed_forward_requests": 0,
            "mixed_scheduled_requests": 0,
            "mixed_direct_work_items": 0,
            "mixed_external_work_items": 0,
            "bulk_host_batches": 0,
            "transformed_direct_launches": 0,
            "ticketed_incremental_launches": 0,
            "stock_attention_launches": 0,
            "stock_resident_batches": 0,
            "stock_resident_attention_launches": 0,
            "stock_prefetched_external_batches": 0,
            "stock_prefetched_external_attention_launches": 0,
            "host_progress_rounds": 0,
            "parallel_indexed_progress_layers": 0,
            "fragment_lookahead_layers": 0,
            "fragment_lookahead_objects": 0,
            "fragment_remaining_rounds": 0,
            "compact_initial_launches": 0,
            "compact_initial_cta_bound": 0,
            "canonical_initial_cta_bound": 0,
            "compact_resume_launches": 0,
            "compact_resume_cta_bound": 0,
            "canonical_resume_cta_bound": 0,
            "predicted_atomic_ns": 0,
            "predicted_incremental_ns": 0,
            "progress_snapshots": 0,
            "request_work_completed": 0,
            "request_work_failed": 0,
            "request_compute_completed_ns": 0,
            "graph_captures": 0,
            "graph_replays": 0,
            "graph_external_batches": 0,
            "demand_graph_enabled": self._demand_graph_enabled,
            "demand_graph_capacity": self._demand_graph_capacity,
            "demand_graph_warmups": 0,
            "demand_graph_captures": 0,
            "demand_graph_replays": 0,
            "demand_graph_evictions": 0,
            "verified_operator_modules": 0,
            "started_unix_ns": time.time_ns(),
        }
        self._stats.update(self._tier_service.stats())
        self._stats.update(
            {
                "nvme_progress_rounds": 0,
                "nvme_bytes": 0,
                "nvme_epochs": 0,
                "tier_external_layers": 0,
                "cxl_direct_work_items": 0,
                "tier_host_proxy_bytes": 0,
            }
        )
        configured_stats = os.environ.get("NTA_ENGINE_STATS_FILE")
        self._stats_publisher: _StatsPublisher | None = None
        if configured_stats:
            stats_path = pathlib.Path(configured_stats)
            if stats_path.suffix:
                stats_path = stats_path.with_name(
                    f"{stats_path.stem}.{os.getpid()}{stats_path.suffix}"
                )
            else:
                stats_path = stats_path / f"nta-sglang-{os.getpid()}.json"
            self._stats_publisher = _StatsPublisher(stats_path)
        self._profile_cpu = os.environ.get("NTA_PROFILE_CPU") == "1"
        self._profile_transfer = os.environ.get("NTA_PROFILE_TRANSFER") == "1"
        self._profile_gpu = os.environ.get("NTA_PROFILE_GPU") == "1"
        # Barrier profiling measures how long the compute stream stalls at each
        # proactive layer-readiness wait. It is the opportunity signal the
        # RQ2/2A characterization consumes: stall > 0 means arrival, not
        # compute, bounded that layer.
        self._profile_barrier = os.environ.get("NTA_PROFILE_BARRIER") == "1"
        self._transfer_profiles: list[
            tuple[torch.cuda.Event, torch.cuda.Event, int, str]
        ] = []
        self._operator_profiles: list[
            tuple[torch.cuda.Event, torch.cuda.Event, str]
        ] = []
        self._barrier_profiles: list[
            tuple[torch.cuda.Event, torch.cuda.Event, int]
        ] = []
        self._barrier_stall_by_layer: dict[int, float] = {}
        trace_file = os.environ.get("NTA_OPPORTUNITY_TRACE_FILE")
        self._opportunity_trace = pathlib.Path(trace_file) if trace_file else None
        self._opportunity_revision = os.environ.get("NTA_REVISION", "")
        self._opportunity_model = os.environ.get(
            "NTA_OPPORTUNITY_MODEL",
            str(getattr(model_runner.model_config, "model_path", "unknown")),
        )
        self._opportunity_tier = os.environ.get("NTA_OPPORTUNITY_TIER", "host_staged")
        self._opportunity_batch = 0
        self._active_opportunity_batch = -1
        self._measure_opportunity_compute = (
            os.environ.get("NTA_OPPORTUNITY_MEASURE_COMPUTE") == "1"
        )
        device_properties = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        )
        self._opportunity_parallel_slots = _positive_environment(
            "NTA_OPPORTUNITY_PARALLEL_SLOTS",
            int(device_properties.multi_processor_count),
        )
        if self._opportunity_trace is not None:
            if not self._opportunity_revision:
                raise ValueError(
                    "NTA_REVISION is required when opportunity tracing is enabled"
                )
            if self._opportunity_tier != "host_staged":
                raise ValueError(
                    "the SGLang HiCache tracer only observes host_staged data"
                )
        if self._prefetch_enabled:
            self._hicache.set_prefetch_callback(self._prepare_host_pipeline)
        elif self._frontier_enabled and self._model_layer_count > 1:
            self._hicache.set_prefetch_callback(self._publish_cross_layer_frontier)
        atexit.register(self._write_stats)

    def _configure_tenant_budgets(
        self, specs: tuple[tuple[int, int, int], ...]
    ) -> None:
        for tenant_id, max_bytes, weight in specs:
            if tenant_id >= self._tenant_capacity:
                raise RuntimeError(
                    f"tenant {tenant_id} exceeds NTA_TENANT_CAPACITY="
                    f"{self._tenant_capacity}"
                )
            self._runtime.set_tenant_budget(tenant_id, max_bytes, weight)

    def cancel_requests(self, request_id_prefix: str, *, all: bool = False) -> int:
        cancelled = self._request_adapter.cancel_matching(request_id_prefix, all=all)
        self._stats["request_cancellations"] += cancelled
        return cancelled

    def close(self) -> None:
        """Flush observations and release CUDA/native tier resources."""
        if self._closed:
            return
        self._collect_transfer_profiles()
        self._collect_barrier_profiles()
        self._write_stats()

    def _close_resources(self) -> None:
        if self._resources_closed:
            return
        # Plans, graphs, and native runtime buffers all contain device pointers.
        # Quiesce every stream before releasing them, including direct NVMe HBM
        # destinations and CXL-backed mappings.
        torch.cuda.synchronize()
        self._hicache.close()
        self._demand_graphs.clear()
        self._demand_graph_warmups.clear()
        self._demand_sync_events.clear()
        for allocation in tuple(self._plans.values()):
            allocation.plan.close()
        self._plans.clear()
        for program in tuple(self._phase_programs.values()):
            program.close()
        self._phase_programs.clear()
        for program in tuple(self._operator_programs.values()):
            program.close()
        self._operator_programs.clear()
        self._resources.close()
        self._resources_closed = True

    def _install_instrumented_wrappers(
        self, model_runner: Any, skip_prefill: bool
    ) -> None:
        q_dtype = model_runner.dtype
        kv_dtype = model_runner.kv_cache_dtype
        head_dim = int(model_runner.model_config.head_dim)
        signature = f"h{head_dim}_{_dtype_tag(q_dtype)}_{_dtype_tag(kv_dtype)}"
        decode_base = (
            f"nta_sglang_decode_{{form}}_v11_"
            f"{'tc' if self.decode_use_tensor_cores else 'cc'}_"
            f"{signature}"
        )
        prefill_base = f"nta_sglang_prefill_{{form}}_v11_{signature}"
        self._wrapper_modules: dict[int, str] = {}

        def decode_wrappers(form: str) -> tuple[list[Any], list[Any]]:
            name = decode_base.format(form=form)
            jit_builder = (
                request_bound_attention_jit_args
                if form == "request_bound"
                else attention_jit_args
            )
            args = jit_builder(
                name,
                dtype_q=q_dtype,
                dtype_kv=kv_dtype,
                dtype_o=q_dtype,
                idtype=torch.int32,
                head_dim_qk=head_dim,
                head_dim_vo=head_dim,
            )
            wrappers = []
            for _ in range(self.num_wrappers):
                wrapper = BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    backend="fa2",
                    use_tensor_cores=self.decode_use_tensor_cores,
                    jit_args=args,
                )
                wrappers.append(wrapper)
                self._wrapper_modules[id(wrapper)] = name
            return wrappers, args

        self._nta_request_bound_decode_wrappers, self._decode_jit_args = (
            decode_wrappers("request_bound")
        )
        self._nta_demand_decode_wrappers, _ = decode_wrappers("demand_acquire")

        if skip_prefill:
            self._select_wrappers(False)
            return

        def prefill_wrappers(form: str) -> tuple[list[Any], list[Any]]:
            name = prefill_base.format(form=form)
            jit_builder = (
                request_bound_attention_jit_args
                if form == "request_bound"
                else attention_jit_args
            )
            args = jit_builder(
                name,
                dtype_q=q_dtype,
                dtype_kv=kv_dtype,
                dtype_o=q_dtype,
                idtype=torch.int32,
                head_dim_qk=head_dim,
                head_dim_vo=head_dim,
            )
            wrappers = []
            for _ in range(2 * self.num_wrappers):
                wrapper = BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    backend="fa2",
                    jit_args=args,
                )
                wrappers.append(wrapper)
                self._wrapper_modules[id(wrapper)] = name
            return wrappers, args

        request_prefill, self._prefill_jit_args = prefill_wrappers("request_bound")
        demand_prefill, _ = prefill_wrappers("demand_acquire")
        split = self.num_wrappers
        self._nta_request_bound_prefill_paged = request_prefill[:split]
        self._nta_request_bound_prefill_verify = request_prefill[split:]
        self._nta_demand_prefill_paged = demand_prefill[:split]
        self._nta_demand_prefill_verify = demand_prefill[split:]
        self._select_wrappers(False)

    def _select_wrappers(self, demand_acquire: bool) -> None:
        self.decode_wrappers = (
            self._nta_demand_decode_wrappers
            if demand_acquire
            else self._nta_request_bound_decode_wrappers
        )
        if self.skip_prefill:
            return
        self.prefill_wrappers_paged = (
            self._nta_demand_prefill_paged
            if demand_acquire
            else self._nta_request_bound_prefill_paged
        )
        self.prefill_wrappers_verify = (
            self._nta_demand_prefill_verify
            if demand_acquire
            else self._nta_request_bound_prefill_verify
        )

    def _select_stock_wrappers(self) -> None:
        self.decode_wrappers = list(self._stock_decode_wrappers)
        if self.skip_prefill:
            return
        self.prefill_wrappers_paged = list(self._stock_prefill_paged_wrappers)
        self.prefill_wrappers_verify = list(self._stock_prefill_verify_wrappers)

    def _create_decode_wrappers(self, bs: int, num_tokens: int) -> list[Any]:
        if self._decode_jit_args is None:
            raise RuntimeError("NTA decode JIT arguments are not initialized")
        wrappers = [
            BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer,
                "NHD",
                backend="fa2",
                use_cuda_graph=True,
                use_tensor_cores=self.decode_use_tensor_cores,
                paged_kv_indptr_buffer=self.kv_indptr[index][: num_tokens + 1],
                paged_kv_indices_buffer=self.cuda_graph_kv_indices[index],
                paged_kv_last_page_len_buffer=self.kv_last_page_len[:num_tokens],
                jit_args=self._decode_jit_args,
            )
            for index in range(self.num_wrappers)
        ]
        for wrapper in wrappers:
            self._wrapper_modules[id(wrapper)] = self._decode_jit_args[0]
        return wrappers

    def _create_prefill_wrappers(
        self, bs: int, use_custom_mask: bool = False
    ) -> list[Any]:
        if self._prefill_jit_args is None:
            raise RuntimeError("NTA prefill JIT arguments are not initialized")
        wrappers = []
        for index in range(self.num_wrappers):
            extra = (
                {
                    "custom_mask_buf": self.cuda_graph_custom_mask,
                    "mask_indptr_buf": self.cuda_graph_qk_indptr[index][: bs + 1],
                }
                if use_custom_mask
                else {}
            )
            wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer,
                "NHD",
                use_cuda_graph=True,
                backend="fa2",
                qo_indptr_buf=self.cuda_graph_qo_indptr[index][: bs + 1],
                paged_kv_indptr_buf=self.kv_indptr[index][: bs + 1],
                paged_kv_indices_buf=self.cuda_graph_kv_indices[index],
                paged_kv_last_page_len_buf=self.kv_last_page_len[:bs],
                jit_args=self._prefill_jit_args,
                **extra,
            )
            self._wrapper_modules[id(wrapper)] = self._prefill_jit_args[0]
            wrappers.append(wrapper)
        return wrappers

    def init_cuda_graph_state(self, *args: Any, **kwargs: Any) -> None:
        super().init_cuda_graph_state(*args, **kwargs)

    def _build_execution_session(
        self,
        *,
        bindings: tuple[RequestBinding, ...],
        schedule: Schedule,
        page_pairs: tuple[_PagePair, ...],
        layer: int,
        unit_bytes: int,
    ) -> ExecutionSession:
        """Translate one native attention launch into the work contract.

        FlashInfer wrappers are reused by every transformer layer.  The
        semantic session therefore describes one wrapper/layer launch instead
        of treating the wrapper array as model layers.
        """
        if self._current_engine_batch is None:
            raise RuntimeError("execution session has no engine batch epoch")
        tiles: list[ExecutionTile] = []
        work_id = 0
        contributor_counts = Counter(
            request_index for request_index in schedule.request_indices
        )
        contributor_indices = {request_index: 0 for request_index in contributor_counts}
        if page_pairs and len(page_pairs) != schedule.work_count:
            raise RuntimeError("execution page pairs do not match CTA schedule")
        for logical_work, request_index in enumerate(schedule.request_indices):
            if request_index < 0 or request_index >= len(bindings):
                raise RuntimeError("FlashInfer schedule referenced an invalid request")
            host_pages = page_pairs[logical_work][0] if page_pairs else ()
            candidate_units = max(1, len(host_pages))
            tiles.append(
                ExecutionTile(
                    work_id=work_id,
                    binding=bindings[request_index],
                    layer=layer,
                    logical_begin=int(schedule.kv_tile_indices[logical_work]),
                    candidate_units=candidate_units,
                    selected_ids=tuple(range(candidate_units)),
                    unit_bytes=unit_bytes,
                    # CXL dependencies are already device-visible direct
                    # addresses.  The HiCache mapping is still used to prove
                    # exact page identity, but it is not a readiness barrier
                    # or a host-data ownership signal for this tier.
                    ready=not host_pages or self._tier_service.is_cxl,
                    estimated_compute_ns=self._host_cost_model.tile_compute_ns,
                    reduction_group=request_index,
                    contributor_index=contributor_indices[request_index],
                    contributor_count=contributor_counts[request_index],
                )
            )
            contributor_indices[request_index] += 1
            work_id += 1
        if not tiles:
            raise RuntimeError("FlashInfer produced no execution work units")
        session = ExecutionSession.from_tiles(
            epoch=self._current_engine_batch.epoch,
            granularity=self._execution_config.protocol.granularity,
            protocol=self._execution_config.protocol,
            tiles=tiles,
        )
        self._stats.update(session.expose_stats())
        return session

    def _ensure_execution_session(
        self,
        wrapper: Any,
        layer: Any,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Create the semantic session immediately before native attention."""
        if self._active_batch is None:
            raise RuntimeError("cannot create execution session without active batch")
        batch = self._active_batch
        decode_wrappers = tuple(
            getattr(self.forward_metadata, "decode_wrappers", ()) or ()
        )
        if any(wrapper is candidate for candidate in decode_wrappers):
            schedule = decode_schedule(wrapper)
        else:
            schedule = paged_prefill_schedule(wrapper)
        page_pairs = batch.page_pairs.get(id(wrapper), ())
        if not page_pairs:
            page_pairs = tuple(((), ()) for _ in range(schedule.work_count))
            batch.page_pairs[id(wrapper)] = page_pairs
        batch.schedules[id(wrapper)] = schedule
        unit_bytes = int(
            kv_cache[0][0].numel() * kv_cache[0].element_size()
            + kv_cache[1][0].numel() * kv_cache[1].element_size()
        )
        batch.execution = self._build_execution_session(
            bindings=batch.bindings,
            schedule=schedule,
            page_pairs=page_pairs,
            layer=int(layer.layer_id) - self._model_start_layer,
            unit_bytes=unit_bytes,
        )

    def _upload_resident_plan(
        self,
        wrapper: Any,
        schedule: Schedule,
        execution: ExecutionSession,
    ) -> DeviceWorkPlan:
        """Materialize a direct demand plan for non-contiguous resident slots.

        The request-bound kernel takes one contiguous slot offset.  SGLang's
        pool allocator is free to return holes, so resident batches with
        non-contiguous slots use the same explicit per-ticket plan as the
        external path.  Dependencies are direct device views; no transfer or
        approximation is introduced.
        """
        batch = self._active_batch
        if batch is None:
            raise RuntimeError("resident work-plan upload has no active batch")
        plan = self._ensure_plan(wrapper, -1, schedule)
        allocation = self._plans[(id(wrapper), -1)]
        signature = (
            "resident",
            schedule.request_indices,
            schedule.kv_tile_indices,
            tuple(binding.request_slot for binding in batch.bindings),
            tuple(binding.generation for binding in batch.bindings),
        )
        if allocation.signature == signature:
            return plan

        semantic_units = []
        dependency_spans = []
        dependencies: list[AcquireRequirement] = []
        for work_ticket, (request_index, logical_begin) in enumerate(
            zip(schedule.request_indices, schedule.kv_tile_indices, strict=True)
        ):
            semantic_units.append(
                execution.unit_for_ticket(
                    work_id=work_ticket,
                    layer=execution.batch.units[work_ticket].layer,
                    logical_begin=int(logical_begin),
                    request_index=request_index,
                )
            )
            dependency_begin = len(dependencies)
            dependencies.extend(
                (
                    AcquireRequirement(
                        self._runtime.device_view,
                        0,
                        _OBJECT_ID_BASE | 0xFFFFFFF0,
                        0,
                        0,
                        1,
                        1,
                        0,
                    ),
                    AcquireRequirement(
                        self._runtime.device_view,
                        0,
                        _OBJECT_ID_BASE | 0xFFFFFFF1,
                        0,
                        1,
                        1,
                        1,
                        0,
                    ),
                )
            )
            dependency_spans.append((dependency_begin, 2, 2, work_ticket))

        ranges = _request_ranges(batch.bindings, schedule.request_indices)
        plan.upload_work_units(
            semantic_units,
            dependency_spans,
            dependencies,
            ranges,
            epoch=execution.epoch,
            stream=torch.cuda.current_stream(),
        )
        allocation.signature = signature
        allocation.object_count = 0
        allocation.direct_work_count = schedule.work_count
        allocation.external_object_slots = tuple(() for _ in range(schedule.work_count))
        return plan

    def _record_execution_layer(self, layer: Any) -> None:
        """Commit the semantic work boundary after native attention returns."""
        if self._active_batch is None or self._active_batch.execution is None:
            raise RuntimeError("attention returned without an execution session")
        local_layer = int(layer.layer_id) - self._model_start_layer
        self._stats.update(
            self._active_batch.execution.record_layer_completion(local_layer)
        )

    def _bind_forward_requests(
        self, forward_batch: Any, *, allow_capture_ids: bool
    ) -> tuple[RequestBinding, ...]:
        batch = self._request_adapter.bind_forward(
            forward_batch,
            allow_capture_ids=allow_capture_ids,
            stream=torch.cuda.current_stream(),
            epoch=self._execution_epoch,
            granularity=self._execution_config.protocol.granularity,
        )
        self._execution_epoch += 1
        self._current_engine_batch = batch
        self._stats["engine_batch_epoch"] = batch.epoch
        self._stats["engine_batch_size"] = len(batch.bindings)
        bindings = batch.bindings
        self._stats["request_rebindings"] += self._request_adapter.last_publish_count
        self._stats["request_metadata_updates"] = (
            self._stats.get("request_metadata_updates", 0)
            + self._request_adapter.last_metadata_publish_count
        )
        return bindings

    def init_forward_metadata_out_graph(
        self, forward_batch: Any, in_capture: bool = False
    ) -> None:
        self._cuda_graph_mode = True
        counter = getattr(self.token_to_kv_pool, "layer_transfer_counter", None)
        consumer_index = -1 if counter is None else int(counter.consumer_index)
        pending = self._hicache.get(consumer_index)
        self._stock_forward = pending is None
        if self._stock_forward:
            self._select_stock_wrappers()
        super().init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)
        bindings = self._bind_forward_requests(
            forward_batch, allow_capture_ids=in_capture
        )
        if pending is None:
            self._active_batch = _ActiveBatch(bindings, {}, None, {}, {}, {}, ())
            self._stats["resident_reference_batches"] += 1
            self._stats["stock_resident_batches"] += 1
        else:
            self._stock_forward = False
            if not self._prefetch_enabled or not pending.prefetched_layers:
                raise RuntimeError(
                    "CUDA graph replay requires the stream-ordered HiCache pipeline"
                )
            self._active_batch = _ActiveBatch(
                bindings,
                {},
                pending,
                {},
                {},
                pending.prefetched_layers,
                pending.prefetch_tensors,
            )
            final_layer = max(pending.prefetched_layers)
            torch.cuda.current_stream().wait_event(
                pending.prefetched_layers[final_layer].ready_event
            )
            self._hicache.handoff_prefetch(pending, self._prefetch_stream)
            self._stats["graph_external_batches"] += 1
        if in_capture:
            self._stats["graph_captures"] += 1
        else:
            self._stats["graph_replays"] += 1
            self._stats["batches"] += 1

    def init_forward_metadata_in_graph(self, forward_batch: Any) -> None:
        super().init_forward_metadata_in_graph(forward_batch)

    def init_forward_metadata(self, forward_batch: Any) -> None:
        self._cuda_graph_mode = False
        self._stock_forward = False
        if forward_batch.forward_mode.is_mixed():
            self._stats["mixed_forward_batches"] += 1
            self._stats["mixed_forward_requests"] += len(
                tuple(getattr(forward_batch, "rids", ()) or ())
            )
        counter = getattr(self.token_to_kv_pool, "layer_transfer_counter", None)
        consumer_index = -1 if counter is None else int(counter.consumer_index)
        pending = self._hicache.get(consumer_index)
        request_slots = getattr(forward_batch, "_nta_request_slots", None)
        if request_slots is None:
            request_slots = getattr(forward_batch, "req_pool_indices", None)
        if request_slots is not None and hasattr(request_slots, "tolist"):
            request_slots = request_slots.tolist()
        request_slots = tuple(int(slot) for slot in request_slots or ())
        contiguous_request_slots = bool(request_slots) and request_slots == tuple(
            range(request_slots[0], request_slots[0] + len(request_slots))
        )
        # A complete exact prefetch has already materialized every logical
        # HiCache page in SGLang's physical KV pool.  At that boundary the
        # acquisition mechanism has done its job; sending the ready pages
        # through a second transformed attention consumer only adds fixed
        # kernel and enqueue overhead.  Keep the typed path for partial or
        # still-arriving layers, and use the framework consumer only when the
        # readiness proof covers the whole model.
        stock_prefetched_external = (
            pending is not None
            and self._prefetch_enabled
            and len(pending.prefetched_layers) == self._model_layer_count
        )
        demand_acquire = (
            pending is not None
            and not self._prefetch_enabled
            and (
                self._incremental_enabled
                or self._overlap_enabled
                or self._fast_bulk_plan(pending).rounds > 1
            )
        ) or not contiguous_request_slots
        if pending is None:
            self._stock_forward = True
            self._select_stock_wrappers()
        elif stock_prefetched_external:
            self._stock_forward = True
            self._select_stock_wrappers()
        else:
            self._select_wrappers(demand_acquire)
        original_use_paged = self.use_paged
        self.use_paged = True
        try:
            super().init_forward_metadata(forward_batch)
            bind_started = time.perf_counter_ns() if self._profile_cpu else 0
            bindings = self._bind_forward_requests(
                forward_batch, allow_capture_ids=False
            )
            if self._profile_cpu:
                self._stats["request_bind_cpu_ns"] = self._stats.get(
                    "request_bind_cpu_ns", 0
                ) + (time.perf_counter_ns() - bind_started)
            if pending is None:
                self._active_batch = _ActiveBatch(bindings, {}, None, {}, {}, {}, ())
                self._stats["batches"] += 1
                self._stats["resident_reference_batches"] += 1
                self._stats["stock_resident_batches"] += 1
                return
            self._init_external_metadata(forward_batch, pending, bindings=bindings)
            if stock_prefetched_external:
                self._stats["stock_prefetched_external_batches"] += 1
        except Exception as error:
            self._active_batch = None
            self._stats["hicache_fallback_batches"] += 1
            self._stats["last_hicache_fallback"] = str(error)
            self._write_stats()
            raise RuntimeError(
                "NTA failed to bind the FlashInfer batch; stock fallback is "
                "disabled because it would bypass request-level semantics"
            ) from error
        finally:
            self.use_paged = original_use_paged

    def _init_external_metadata(
        self,
        forward_batch: Any,
        pending: PendingHostLoad,
        *,
        bindings: tuple[RequestBinding, ...] | None = None,
        count_batch: bool = True,
    ) -> HostExecutionPlan | None:
        if self._opportunity_trace is not None and count_batch:
            self._active_opportunity_batch = self._opportunity_batch
            self._opportunity_batch += 1
        metadata_started = time.perf_counter_ns() if self._profile_cpu else 0
        if forward_batch.forward_mode.is_decode_or_idle():
            wrappers = self.forward_metadata.decode_wrappers
            extractor = decode_schedule
        else:
            if self.forward_metadata.use_ragged:
                raise RuntimeError("NTA requires SGLang paged prefill metadata")
            wrappers = self.forward_metadata.prefill_wrappers
            extractor = paged_prefill_schedule
        if self._prefetch_enabled and not pending.prefetched_layers:
            raise RuntimeError("HiCache producer did not publish NTA lookahead objects")

        if bindings is None:
            bindings = self._bind_forward_requests(
                forward_batch, allow_capture_ids=False
            )

        schedules: dict[int, Schedule] = {}
        for wrapper in wrappers:
            schedule = extractor(wrapper)
            self._validate_schedule(schedule, bindings)
            schedules[id(wrapper)] = schedule
        pending_pages = set(pending.materialize_mapping())
        planned_pages: set[int] = set()
        tile_page_pairs: dict[int, tuple[_PagePair, ...]] = {}
        for wrapper in wrappers:
            planned_pages.update(self._wrapper_pages(wrapper))
            tile_page_pairs[id(wrapper)] = self._work_page_pairs(
                wrapper, schedules[id(wrapper)], pending
            )
        missing = pending_pages - planned_pages
        if missing:
            raise RuntimeError(
                f"attention metadata omits {len(missing)} promoted HiCache pages"
            )
        request_page_pairs = {
            wrapper_id: _group_external_pages_by_request(schedules[wrapper_id], pairs)
            for wrapper_id, pairs in tile_page_pairs.items()
        }
        if forward_batch.forward_mode.is_mixed():
            representative_id = next(iter(schedules))
            representative_schedule = schedules[representative_id]
            representative_pairs = request_page_pairs[representative_id]
            self._stats["mixed_scheduled_requests"] += len(
                set(representative_schedule.request_indices)
            )
            self._stats["mixed_direct_work_items"] += sum(
                not pair[0] for pair in representative_pairs
            )
            self._stats["mixed_external_work_items"] += sum(
                bool(pair[0]) for pair in representative_pairs
            )
        request_execution = self._metadata_execution_plan(
            schedules, request_page_pairs, pending
        )
        tile_execution = self._metadata_execution_plan(
            schedules, tile_page_pairs, pending
        )
        grouping = self._grouping
        if grouping == "adaptive":
            if self._incremental_enabled:
                grouping = "tile"
            else:
                grouping = (
                    "tile"
                    if tile_execution.predicted_incremental_ns
                    < request_execution.predicted_incremental_ns
                    else "request"
                )
            self._stats[f"adaptive_{grouping}_batches"] += 1
        if grouping == "tile":
            page_pairs = tile_page_pairs
            host_execution = tile_execution
        else:
            page_pairs = request_page_pairs
            host_execution = request_execution
        self._active_batch = _ActiveBatch(
            bindings,
            schedules,
            pending,
            page_pairs,
            {},
            (
                pending.prefetched_layers
                if self._prefetch_enabled or self._frontier_enabled
                else {}
            ),
            (
                pending.prefetch_tensors
                if self._prefetch_enabled or self._frontier_enabled
                else ()
            ),
            host_execution,
            grouping,
        )
        if self._profile_cpu:
            self._stats["metadata_cpu_ns"] = self._stats.get("metadata_cpu_ns", 0) + (
                time.perf_counter_ns() - metadata_started
            )
        if count_batch:
            self._stats["batches"] += 1
            self._stats["hicache_external_batches"] += 1
        return host_execution

    def _fast_bulk_plan(self, pending: PendingHostLoad) -> HostExecutionPlan:
        """Choose bulk execution without materializing GPU plan arrays.

        Padded work and one object pair per possible work item deliberately
        overestimate incremental opportunity. A one-round result is therefore
        safe to send through transformed direct attention after bulk transfer;
        uncertain batches continue through exact request/tile binding.
        """
        if self.forward_metadata is None:
            raise RuntimeError("bulk plan has no FlashInfer metadata")
        wrappers = (
            self.forward_metadata.decode_wrappers
            if hasattr(self.forward_metadata, "decode_wrappers")
            else self.forward_metadata.prefill_wrappers
        )
        padded_work = max(
            (int(wrapper._plan_info[0]) for wrapper in wrappers), default=0
        )
        transfer_count = int(pending.host_indices.numel())
        if padded_work <= 0 or transfer_count <= 0:
            raise RuntimeError("bulk plan observed empty FlashInfer or HiCache work")
        controller = pending.controller
        key_cache = controller.mem_pool_host.k_data_refs[0]
        value_cache = controller.mem_pool_host.v_data_refs[0]
        transfer_bytes = transfer_count * (
            key_cache[0].numel() * key_cache.element_size()
            + value_cache[0].numel() * value_cache.element_size()
        )
        possible_groups = min(padded_work, transfer_count)
        return plan_host_execution(
            object_count=2 * possible_groups,
            transfer_bytes=transfer_bytes,
            runnable_tiles=padded_work,
            model=self._host_cost_model,
            force_rounds=(
                self._host_cost_model.max_rounds if self._incremental_enabled else None
            ),
        )

    def _prepare_host_pipeline(
        self, pending: PendingHostLoad, *, first_local_layer: int = 0
    ) -> None:
        if self._profile_barrier:
            # Drain outstanding barrier measurements before this batch
            # re-records the shared per-layer ready events.
            self._collect_barrier_profiles()
        pipeline_started = time.perf_counter_ns() if self._profile_cpu else 0
        controller = pending.controller
        layer_count = int(controller.layer_num)
        if first_local_layer < 0 or first_local_layer >= layer_count:
            raise RuntimeError("HiCache acquisition frontier is outside the model")
        acquired_layer_count = layer_count - first_local_layer
        transfer_first_slot, _ = _pipeline_object_range(
            self._object_capacity, pending.consumer_index, layer_count
        )
        transfer_count = int(pending.host_indices.numel())
        if transfer_count <= 0 or transfer_count != int(pending.device_indices.numel()):
            raise RuntimeError("HiCache host pipeline has no promoted pages")
        device_pool = controller.mem_pool_device
        transfer_source_indices, transfer_staging_indices = controller.move_indices(
            pending.host_indices, pending.device_indices
        )
        # NTA's indexed-transfer ABI is uint32. SGLang currently publishes
        # int64 cache indices, so passing its storage through would interleave
        # every real index with a zero high word and corrupt the destination map.
        transfer_source_indices = transfer_source_indices.to(
            device=device_pool.device, dtype=torch.int32, non_blocking=True
        )
        transfer_staging_indices = transfer_staging_indices.to(
            device=device_pool.device, dtype=torch.int32, non_blocking=True
        )
        if not self._prefetch_ready_events:
            self._prefetch_ready_events = tuple(
                tuple(
                    torch.cuda.Event(enable_timing=self._profile_barrier)
                    for _ in range(layer_count)
                )
                for _ in controller.layer_done_counter.events
            )
        if pending.consumer_index >= len(self._prefetch_ready_events):
            raise RuntimeError("SGLang published an invalid HiCache producer slot")
        ready_events = self._prefetch_ready_events[pending.consumer_index]
        if layer_count > len(ready_events):
            raise RuntimeError(
                "SGLang HiCache layer count changed after initialization"
            )
        start_layer = int(getattr(device_pool, "start_layer", 0))
        layer_geometry: list[tuple[int, int]] = []
        transfer_objects: list[IndexedHostObject] = []
        paired_copy = True
        for local_layer in range(layer_count):
            layer_id = start_layer + local_layer
            key_cache = device_pool._get_key_buffer(layer_id)
            value_cache = device_pool._get_value_buffer(layer_id)
            host_key = controller.mem_pool_host.k_data_refs[local_layer]
            host_value = controller.mem_pool_host.v_data_refs[local_layer]
            if (
                host_key.dtype != key_cache.dtype
                or host_value.dtype != value_cache.dtype
            ):
                raise RuntimeError("HiCache host and device KV dtypes disagree")
            key_element_bytes = key_cache[0].numel() * key_cache.element_size()
            value_element_bytes = value_cache[0].numel() * value_cache.element_size()
            paired_copy &= (
                key_element_bytes == value_element_bytes
                and key_element_bytes in {128, 256, 512, 1024, 2048}
            )
            key_bytes = transfer_count * key_element_bytes
            value_bytes = transfer_count * value_element_bytes
            if max(key_bytes, value_bytes) > _MAX_ABI_BYTES:
                raise RuntimeError("HiCache layer transfer exceeds the NTA ABI limit")
            layer_geometry.append((key_bytes, value_bytes))
            first_slot = transfer_first_slot + 2 * local_layer
            object_id_base = (
                _OBJECT_ID_BASE
                + (1 << 44)
                + pending.consumer_index * 2 * layer_count
                + 2 * local_layer
            )
            transfer_objects.extend(
                (
                    IndexedHostObject(
                        object_id_base,
                        _LOOKAHEAD_VERSION,
                        host_key.data_ptr(),
                        key_cache.data_ptr(),
                        transfer_source_indices.data_ptr(),
                        transfer_staging_indices.data_ptr(),
                        transfer_count,
                        key_element_bytes,
                        host_key.stride(0) * host_key.element_size(),
                        key_cache.stride(0) * key_cache.element_size(),
                        int(host_key.shape[0]),
                        int(key_cache.shape[0]),
                    ),
                    IndexedHostObject(
                        object_id_base + 1,
                        _LOOKAHEAD_VERSION,
                        host_value.data_ptr(),
                        value_cache.data_ptr(),
                        transfer_source_indices.data_ptr(),
                        transfer_staging_indices.data_ptr(),
                        transfer_count,
                        value_element_bytes,
                        host_value.stride(0) * host_value.element_size(),
                        value_cache.stride(0) * value_cache.element_size(),
                        int(host_value.shape[0]),
                        int(value_cache.shape[0]),
                    ),
                )
            )

        prefetched_layers: dict[int, _PrefetchedLayer] = {}
        profile_start = (
            torch.cuda.Event(enable_timing=True) if self._profile_transfer else None
        )
        profile_finish = (
            torch.cuda.Event(enable_timing=True) if self._profile_transfer else None
        )
        try:
            producer_stream = torch.cuda.current_stream()
            self._runtime.register_indexed_host_objects(
                transfer_first_slot, transfer_objects, stream=producer_stream
            )
            pending.producer_event.start_event.record(producer_stream)
            phase_program = self._phase_program(self._nta_demand_decode_wrappers[0])
            with torch.cuda.stream(self._prefetch_stream):
                pending.producer_event.start_event.wait(self._prefetch_stream)
                if profile_start is not None:
                    profile_start.record(self._prefetch_stream)
                local_layer = first_local_layer
                while local_layer < layer_count:
                    wave_end = min(
                        layer_count, local_layer + self._frontier_layers_per_wave
                    )
                    first_slot = transfer_first_slot + 2 * local_layer
                    if paired_copy:
                        phase_program.preload_host_pairs(
                            self._runtime,
                            first_slot,
                            wave_end - local_layer,
                            self._prefetch_stream,
                        )
                    else:
                        phase_program.preload_host(
                            self._runtime,
                            first_slot,
                            2 * (wave_end - local_layer),
                            self._prefetch_stream,
                        )
                    if profile_finish is not None and wave_end == layer_count:
                        profile_finish.record(self._prefetch_stream)
                    for ready_layer in range(local_layer, wave_end):
                        key_bytes, value_bytes = layer_geometry[ready_layer]
                        ready_event = ready_events[ready_layer]
                        ready_event.record(self._prefetch_stream)
                        prefetched_layers[ready_layer] = _PrefetchedLayer(
                            0,
                            _LOOKAHEAD_ALIAS_ID_BASE,
                            1,
                            _LOOKAHEAD_ALIAS_ID_BASE + 1,
                            _LOOKAHEAD_VERSION,
                            key_bytes,
                            value_bytes,
                            ready_event,
                            transfer_first_slot + 2 * ready_layer,
                        )
                    self._stats["lookahead_copy_waves"] = (
                        self._stats.get("lookahead_copy_waves", 0) + 1
                    )
                    local_layer = wave_end
        except Exception:
            self._prefetch_stream.synchronize()
            self._stats["hicache_fallback_batches"] += 1
            raise
        pending.prefetched_layers = prefetched_layers
        pending.prefetch_tensors = (
            transfer_source_indices,
            transfer_staging_indices,
        )
        frontier_geometry = layer_geometry[first_local_layer:]
        if profile_start is not None and profile_finish is not None:
            transfer_bytes = sum(
                key_bytes + value_bytes for key_bytes, value_bytes in frontier_geometry
            )
            self._transfer_profiles.append(
                (profile_start, profile_finish, transfer_bytes, "pipeline")
            )
        self._stats["prefetched_layers"] += acquired_layer_count
        self._stats["prefetched_host_bytes"] += sum(
            key_bytes + value_bytes for key_bytes, value_bytes in frontier_geometry
        )
        self._stats["lookahead_acquisition_layers"] += acquired_layer_count
        self._stats["lookahead_acquisition_objects"] += 2 * acquired_layer_count
        if paired_copy:
            self._stats["paired_lookahead_layers"] = (
                self._stats.get("paired_lookahead_layers", 0) + acquired_layer_count
            )
        if self._profile_cpu:
            self._stats["pipeline_cpu_ns"] = self._stats.get("pipeline_cpu_ns", 0) + (
                time.perf_counter_ns() - pipeline_started
            )

    def _prepare_cross_layer_frontier(self, pending: PendingHostLoad) -> None:
        self._prepare_host_pipeline(pending, first_local_layer=1)
        self._stats["cross_layer_frontier_batches"] = (
            self._stats.get("cross_layer_frontier_batches", 0) + 1
        )
        self._stats["cross_layer_frontier_layers"] = self._stats.get(
            "cross_layer_frontier_layers", 0
        ) + len(pending.prefetched_layers)

    def _publish_cross_layer_frontier(self, pending: PendingHostLoad) -> None:
        transfer_bytes = _frontier_transfer_bytes(pending)
        self._prepare_cross_layer_frontier(pending)
        self._stats["frontier_proactive_batches"] = (
            self._stats.get("frontier_proactive_batches", 0) + 1
        )
        self._stats["frontier_published_bytes"] = (
            self._stats.get("frontier_published_bytes", 0) + transfer_bytes
        )

    def _wrapper_layout(
        self, wrapper: Any
    ) -> tuple[list[int], list[int], list[int], int]:
        batch_size = int(wrapper._batch_size)
        indptr = (
            wrapper._paged_kv_indptr_buf[: batch_size + 1]
            .detach()
            .to(device="cpu")
            .tolist()
        )
        page_count = int(indptr[-1])
        pages = (
            wrapper._paged_kv_indices_buf[:page_count]
            .detach()
            .to(device="cpu")
            .tolist()
        )
        last_page = (
            wrapper._paged_kv_last_page_len_buf[:batch_size]
            .detach()
            .to(device="cpu")
            .tolist()
        )
        page_size = int(getattr(wrapper, "_page_size", self.token_to_kv_pool.page_size))
        if page_size != 1:
            raise RuntimeError(
                "NTA's SGLang HiCache path currently requires page_size=1"
            )
        return indptr, pages, last_page, page_size

    def _wrapper_pages(self, wrapper: Any) -> tuple[int, ...]:
        _, pages, _, _ = self._wrapper_layout(wrapper)
        return tuple(pages)

    @staticmethod
    def _validate_schedule(
        schedule: Schedule, bindings: tuple[RequestBinding, ...]
    ) -> None:
        if schedule.work_count <= 0:
            raise RuntimeError("FlashInfer emitted no active CTA work")
        if schedule.work_count != len(schedule.kv_tile_indices):
            raise RuntimeError("FlashInfer schedule identity arrays disagree")
        cursor = 0
        for request_index in range(len(bindings)):
            begin = cursor
            while (
                cursor < schedule.work_count
                and schedule.request_indices[cursor] == request_index
            ):
                cursor += 1
            if cursor == begin:
                raise RuntimeError(
                    f"FlashInfer emitted no CTA work for request {request_index}"
                )
        if cursor != schedule.work_count:
            raise RuntimeError("FlashInfer CTA work is not request-contiguous")

    def _layer_execution_plan(
        self,
        wrapper: Any,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> HostExecutionPlan:
        batch = self._active_batch
        if batch is None or batch.pending_host_load is None:
            raise RuntimeError("host execution plan has no active HiCache load")
        if batch.host_execution is not None:
            return batch.host_execution
        schedule = batch.schedules.get(id(wrapper))
        pairs = batch.page_pairs.get(id(wrapper))
        if schedule is None or pairs is None:
            raise RuntimeError("host execution plan has no FlashInfer schedule")
        key_cache, value_cache = kv_cache
        key_element_bytes = key_cache[0].numel() * key_cache.element_size()
        value_element_bytes = value_cache[0].numel() * value_cache.element_size()
        return self._execution_plan(
            schedule, pairs, key_element_bytes, value_element_bytes
        )

    def _metadata_execution_plan(
        self,
        schedules: dict[int, Schedule],
        page_pairs: dict[int, tuple[_PagePair, ...]],
        pending: PendingHostLoad,
    ) -> HostExecutionPlan:
        controller = pending.controller
        if not controller.mem_pool_host.k_data_refs:
            raise RuntimeError("HiCache host pool has no K/V layers")
        key_cache = controller.mem_pool_host.k_data_refs[0]
        value_cache = controller.mem_pool_host.v_data_refs[0]
        key_element_bytes = key_cache[0].numel() * key_cache.element_size()
        value_element_bytes = value_cache[0].numel() * value_cache.element_size()
        plans = {
            self._execution_plan(
                schedule,
                page_pairs[wrapper_id],
                key_element_bytes,
                value_element_bytes,
            )
            for wrapper_id, schedule in schedules.items()
        }
        if len(plans) != 1:
            raise RuntimeError(
                "FlashInfer wrappers selected inconsistent host execution plans"
            )
        return plans.pop()

    def _execution_plan(
        self,
        schedule: Schedule,
        pairs: tuple[_PagePair, ...],
        key_element_bytes: int,
        value_element_bytes: int,
    ) -> HostExecutionPlan:
        unique_pairs = {pair for pair in pairs if pair[0]}
        if not unique_pairs:
            raise RuntimeError("external HiCache batch has no CTA dependency")
        transfer_bytes = sum(
            len(pair[0]) * (key_element_bytes + value_element_bytes)
            for pair in unique_pairs
        )
        if self._execution_config.protocol.kind is ProtocolKind.CONVENTIONAL:
            transfer_ns = math.ceil(
                transfer_bytes
                * 1_000_000_000
                / self._host_cost_model.bandwidth_bytes_per_second
            )
            compute_ns = schedule.work_count * self._host_cost_model.tile_compute_ns
            return HostExecutionPlan(
                (2 * len(unique_pairs),),
                transfer_ns + compute_ns,
                transfer_ns + compute_ns,
                False,
            )
        return plan_host_execution(
            object_count=2 * len(unique_pairs),
            transfer_bytes=transfer_bytes,
            runnable_tiles=schedule.work_count,
            model=self._host_cost_model,
            force_rounds=(
                self._host_cost_model.max_rounds if self._incremental_enabled else None
            ),
            initial_runnable_tiles=sum(not pair[0] for pair in pairs),
        )

    def _run_bulk_host_layer(
        self,
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        output: torch.Tensor,
        layer: Any,
        run_options: dict[str, Any],
        stream: torch.cuda.Stream,
    ) -> None:
        batch = self._active_batch
        if batch is None or batch.pending_host_load is None:
            raise RuntimeError("bulk host execution has no active HiCache load")
        pending = batch.pending_host_load
        local_layer = int(layer.layer_id) - int(
            getattr(pending.controller.mem_pool_device, "start_layer", 0)
        )
        if not pending.prefetched_layers:
            self._prepare_host_pipeline(pending)
            batch.prefetched_layers.update(pending.prefetched_layers)
        prefetched = pending.prefetched_layers.get(local_layer)
        if prefetched is None:
            raise RuntimeError(f"bulk host plan omitted layer {layer.layer_id}")
        if self._profile_barrier:
            arrive = torch.cuda.Event(enable_timing=True)
            arrive.record(stream)
            self._barrier_profiles.append(
                (arrive, prefetched.ready_event, int(layer.layer_id))
            )
        stream.wait_event(prefetched.ready_event)
        self._upload_plan(wrapper, int(layer.layer_id), kv_cache)
        self._run_preacquired_attention(
            wrapper, q, kv_cache, output, layer, run_options
        )
        self._bulk_events = (prefetched.ready_event,)

    def _run_preacquired_attention(
        self,
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        output: torch.Tensor,
        layer: Any,
        run_options: dict[str, Any],
    ) -> None:
        batch = self._active_batch
        if batch is None:
            raise RuntimeError("preacquired attention has no active batch")
        if id(wrapper) not in self._wrapper_modules:
            raise RuntimeError(
                "NTA direct attention requires a compiler-transformed "
                "FlashInfer wrapper"
            )
        # Loading the contract is a one-time dictionary hit after the first
        # launch. It prevents a fast direct module from bypassing source/form
        # pairing with the incremental module used by the same backend.
        self._phase_program(wrapper)
        runtime_tensor = self._runtime.device_view_tensor
        module_name = self._wrapper_modules[id(wrapper)]
        pending = batch.pending_host_load
        allocation = None
        schedule = None
        if pending is not None:
            schedule = batch.schedules.get(id(wrapper))
            allocation = self._plans.get((id(wrapper), -1))
            if (
                schedule is None
                or allocation is None
                or allocation.plan.work_item_count != schedule.work_count
            ):
                planned = sorted(
                    self._wrapper_modules.get(wrapper_id, str(wrapper_id))
                    for wrapper_id in batch.schedules
                )
                raise RuntimeError(
                    "preacquired external attention has no validated CTA work "
                    f"plan: wrapper={module_name} schedule={schedule is not None} "
                    f"allocation={allocation is not None} "
                    f"work_count={getattr(schedule, 'work_count', None)} "
                    f"plan_items="
                    f"{getattr(getattr(allocation, 'plan', None), 'work_item_count', None)} "
                    f"planned_wrappers={planned}"
                )
        elif "request_bound" not in module_name:
            schedule = batch.schedules.get(id(wrapper))
            execution = batch.execution
            if schedule is None or execution is None:
                raise RuntimeError(
                    "resident demand attention has no semantic CTA schedule"
                )
            self._upload_resident_plan(wrapper, schedule, execution)
            allocation = self._plans[(id(wrapper), -1)]
        if "request_bound" in module_name:
            request_slots = tuple(binding.request_slot for binding in batch.bindings)
            if not request_slots or request_slots != tuple(
                range(request_slots[0], request_slots[0] + len(request_slots))
            ):
                raise RuntimeError(
                    "NTA direct attention requires contiguous request slots"
                )
            wrapper.run(
                q,
                kv_cache,
                runtime_tensor,
                layer.scaling,
                request_slots[0],
                out=output,
                **run_options,
            )
        else:
            if allocation is None or schedule is None:
                raise RuntimeError(
                    "incremental FlashInfer wrapper requires a validated work plan"
                )
            wrapper.run(
                q,
                kv_cache,
                runtime_tensor,
                allocation.plan.work_items_tensor,
                allocation.plan.dependencies_tensor,
                layer.scaling,
                schedule.work_count,
                PREACQUIRED | BIND_CURRENT_GENERATION,
                out=output,
                **run_options,
            )
        self._stats["transformed_direct_launches"] += 1

    def _ensure_plan(
        self, wrapper: Any, layer_id: int, schedule: Schedule
    ) -> DeviceWorkPlan:
        key = (id(wrapper), layer_id)
        allocation = self._plans.get(key)
        if allocation is not None and schedule.work_count <= allocation.work_capacity:
            return allocation.plan
        if allocation is not None:
            torch.cuda.current_stream().synchronize()
            self._discard_demand_graphs(allocation.plan)
            allocation.plan.close()
        capacity = schedule.work_count
        plan = DeviceWorkPlan(capacity, 2 * capacity, self._runtime.device_ordinal)
        self._plans[key] = _PlanAllocation(plan, capacity)
        return plan

    def _discard_demand_graphs(self, plan: DeviceWorkPlan) -> None:
        """Drop graph executables before releasing their captured plan buffers."""
        work_items_address = int(plan.work_items_address)
        dependencies_address = int(plan.dependencies_address)
        stale = {
            key
            for key in self._demand_graph_warmups
            if key.work_items_address == work_items_address
            and key.dependencies_address == dependencies_address
        }
        for key in stale:
            self._demand_graphs.pop(key, None)
            self._demand_graph_warmups.pop(key, None)

    def _reserve_demand_graph_key(
        self, key: _DemandGraphKey, stream: torch.cuda.Stream
    ) -> None:
        """Reserve bounded graph-cache state, quiescing before pointer release."""
        if key in self._demand_graph_warmups:
            self._demand_graph_warmups.pop(key)
            self._demand_graph_warmups[key] = None
            return
        if len(self._demand_graph_warmups) >= self._demand_graph_capacity:
            stream.synchronize()
            stale = next(iter(self._demand_graph_warmups))
            self._demand_graph_warmups.pop(stale)
            self._demand_graphs.pop(stale, None)
            self._stats["demand_graph_evictions"] += 1
        self._demand_graph_warmups[key] = None

    def _record_demand_plan_stats(
        self,
        batch: _ActiveBatch,
        schedule: Schedule,
        object_count: int,
        transfer_bytes: int,
        host_execution: HostExecutionPlan,
    ) -> None:
        self._stats["demand_host_layers"] += 1
        self._stats["cta_work_items"] += schedule.work_count
        self._stats["indexed_host_objects"] += object_count
        group_counter = (
            "request_acquisition_groups"
            if batch.grouping == "request"
            else "tile_acquisition_groups"
        )
        self._stats[group_counter] += object_count // 2
        self._stats["indexed_host_bytes"] += transfer_bytes
        self._stats["host_progress_rounds"] += host_execution.rounds
        self._stats["predicted_atomic_ns"] += host_execution.predicted_atomic_ns
        self._stats["predicted_incremental_ns"] += (
            host_execution.predicted_incremental_ns
        )
        if host_execution.rounds > 1 or host_execution.overlap_initial:
            self._stats["incremental_host_layers"] += 1
        if host_execution.overlap_initial:
            self._stats["request_overlap_layers"] += 1

    def _upload_plan(
        self,
        wrapper: Any,
        layer_id: int,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[
        DeviceWorkPlan,
        Schedule,
        int,
        torch.cuda.Event | None,
        int,
        HostExecutionPlan | None,
    ]:
        profile_started = time.perf_counter_ns() if self._profile_cpu else 0
        batch = self._active_batch
        if batch is None:
            raise RuntimeError("NTA attention ran without forward metadata")
        schedule = batch.schedules.get(id(wrapper))
        if schedule is None:
            raise RuntimeError("NTA attention wrapper was not planned for this batch")
        if schedule.work_count > self._work_ticket_capacity:
            raise RuntimeError(
                f"FlashInfer needs {schedule.work_count} work tickets; configured "
                f"capacity is {self._work_ticket_capacity}"
            )
        key_cache, value_cache = kv_cache
        if not key_cache.is_cuda or not value_cache.is_cuda:
            raise RuntimeError("SGLang KV cache must be CUDA-addressable")
        key_bytes = min(int(key_cache.nbytes), _MAX_ABI_BYTES)
        value_bytes = min(int(value_cache.nbytes), _MAX_ABI_BYTES)
        if key_bytes == 0 or value_bytes == 0:
            raise RuntimeError("SGLang exposed an empty KV cache allocation")
        pending = batch.pending_host_load
        if pending is None:
            raise RuntimeError("demand plan has no HiCache transfer")
        controller = pending.controller
        device_pool = controller.mem_pool_device
        local_layer = layer_id - int(getattr(device_pool, "start_layer", 0))
        if local_layer < 0 or local_layer >= int(controller.layer_num):
            raise RuntimeError(f"SGLang layer {layer_id} is outside the HiCache pool")
        prefetched = batch.prefetched_layers.get(local_layer)
        if not self._tier_service.is_host and prefetched is not None:
            raise RuntimeError(
                "physical tiers cannot consume a host-prefetched HiCache layer"
            )
        page_pairs = batch.page_pairs[id(wrapper)]
        signature = _plan_cache_signature(
            schedule.request_indices,
            schedule.kv_tile_indices,
            page_pairs,
            tuple(binding.request_slot for binding in batch.bindings),
            key_bytes,
            value_bytes,
            None
            if prefetched is None
            else (prefetched.key_bytes, prefetched.value_bytes),
        )
        signature = signature + (
            self._tier_service.tier.value,
            self._tier_service.catalog_digest,
            layer_id if not self._tier_service.is_host else None,
        )
        # Work/ticket topology is layer invariant. Layer K/V addresses are
        # republished through the object directory on the consumer stream.
        plan = self._ensure_plan(wrapper, -1, schedule)
        allocation = self._plans[(id(wrapper), -1)]
        rebuild_plan = allocation.signature != signature
        if not self._tier_service.is_host and not rebuild_plan:
            self._stats["cta_work_items"] += schedule.work_count
            if self._profile_cpu:
                self._stats["plan_cpu_ns"] = self._stats.get("plan_cpu_ns", 0) + (
                    time.perf_counter_ns() - profile_started
                )
            return (
                plan,
                schedule,
                allocation.object_count,
                None,
                0,
                None,
            )
        if prefetched is not None and not rebuild_plan:
            if allocation.host_execution is None or allocation.object_count != 2:
                raise RuntimeError("cached HiCache plan is incomplete")
            self._stats["cta_work_items"] += schedule.work_count
            if self._profile_cpu:
                self._stats["plan_cpu_ns"] = self._stats.get("plan_cpu_ns", 0) + (
                    time.perf_counter_ns() - profile_started
                )
            return (
                plan,
                schedule,
                allocation.object_count,
                prefetched.ready_event,
                0,
                allocation.host_execution,
            )

        host_key = controller.mem_pool_host.k_data_refs[local_layer]
        host_value = controller.mem_pool_host.v_data_refs[local_layer]
        if host_key.dtype != key_cache.dtype or host_value.dtype != value_cache.dtype:
            raise RuntimeError("HiCache host and device KV dtypes disagree")
        key_element_bytes = key_cache[0].numel() * key_cache.element_size()
        value_element_bytes = value_cache[0].numel() * value_cache.element_size()
        if key_element_bytes <= 0 or value_element_bytes <= 0:
            raise RuntimeError("HiCache exposed an empty KV row")

        indexed_geometry = (
            key_element_bytes,
            value_element_bytes,
            host_key.stride(0) * host_key.element_size(),
            host_value.stride(0) * host_value.element_size(),
            key_cache.stride(0) * key_cache.element_size(),
            value_cache.stride(0) * value_cache.element_size(),
            int(host_key.shape[0]),
            int(host_value.shape[0]),
            int(key_cache.shape[0]),
            int(value_cache.shape[0]),
        )
        if (
            not rebuild_plan
            and allocation.indexed_geometry is not None
            and allocation.indexed_geometry != indexed_geometry
        ):
            rebuild_plan = True

        if not rebuild_plan and prefetched is None:
            host_execution = allocation.host_execution
            object_count = allocation.object_count
            if (
                host_execution is None
                or object_count == 0
                or object_count % 2 != 0
                or allocation.transfer_bytes == 0
                or allocation.indexed_geometry != indexed_geometry
            ):
                raise RuntimeError("cached demand plan is incomplete")
            stream = torch.cuda.current_stream()
            lookahead = batch.fragment_lookahead.pop(layer_id, None)
            preloaded_event = None
            preloaded_object_count = 0
            if lookahead is not None:
                expected = (
                    id(wrapper),
                    object_count,
                    host_key.data_ptr(),
                    key_cache.data_ptr(),
                    host_value.data_ptr(),
                    value_cache.data_ptr(),
                )
                actual = (
                    lookahead.wrapper_id,
                    lookahead.object_count,
                    lookahead.key_source,
                    lookahead.key_staging,
                    lookahead.value_source,
                    lookahead.value_staging,
                )
                if actual != expected:
                    raise RuntimeError(
                        "fragment lookahead no longer matches the next attention layer"
                    )
                preloaded_event = lookahead.ready_event
                preloaded_object_count = lookahead.preloaded_object_count
            else:
                self._phase_program(wrapper).rebind_indexed_host_pairs(
                    self._runtime,
                    0,
                    object_count // 2,
                    host_key.data_ptr(),
                    key_cache.data_ptr(),
                    host_value.data_ptr(),
                    value_cache.data_ptr(),
                    stream,
                )
            self._record_demand_plan_stats(
                batch,
                schedule,
                object_count,
                allocation.transfer_bytes,
                host_execution,
            )
            if self._profile_cpu:
                self._stats["plan_cpu_ns"] = self._stats.get("plan_cpu_ns", 0) + (
                    time.perf_counter_ns() - profile_started
                )
            return (
                plan,
                schedule,
                object_count,
                preloaded_event,
                preloaded_object_count,
                host_execution,
            )

        if rebuild_plan and prefetched is None:
            allocation.object_version = (allocation.object_version + 1) & 0xFFFFFFFF
            allocation.object_version = allocation.object_version or 1
        version = (
            prefetched.version if prefetched is not None else allocation.object_version
        )
        indexed_objects: list[IndexedHostObject] = []
        index_tensors: list[torch.Tensor] = []
        physical_object_bytes: list[int] = []
        pair_objects: dict[_PagePair, tuple[int, int, int, int]] = {}
        physical_extents: dict[_PagePair, tuple[Any, Any]] = {}
        if self._tier_service.is_nvme and prefetched is None:
            unique_physical_pairs = tuple(
                dict.fromkeys(pair for pair in page_pairs if pair[0])
            )
            if 2 * len(unique_physical_pairs) > self._object_capacity:
                raise RuntimeError(
                    "NVMe layer needs more HBM object slots than the runtime capacity"
                )
            # Resolve the complete catalog before installing any HBM object.
            # This keeps catalog/geometry errors transactional from the
            # engine's perspective and avoids partially publishing a layer.
            for pair in unique_physical_pairs:
                physical_extents[pair] = (
                    self._tier_service.extent(
                        layer_id, tuple(pair[1]), "key", key_element_bytes
                    ),
                    self._tier_service.extent(
                        layer_id, tuple(pair[1]), "value", value_element_bytes
                    ),
                )

        def objects_for(pair: _PagePair) -> tuple[int, int, int, int]:
            if prefetched is not None:
                return (
                    prefetched.key_slot,
                    prefetched.key_object_id,
                    prefetched.value_slot,
                    prefetched.value_object_id,
                )
            existing = pair_objects.get(pair)
            if existing is not None:
                return existing
            host_pages, device_pages = pair
            if self._tier_service.is_nvme:
                extents = physical_extents.get(pair)
                if extents is None:
                    raise RuntimeError("NVMe page pair was not catalog-validated")
                key_extent, value_extent = extents
                key_slot = len(physical_object_bytes)
                key_object_id = _OBJECT_ID_BASE | key_slot
                self._runtime.install_nvme_object(
                    key_slot,
                    key_object_id,
                    version,
                    key_extent.offset,
                    key_extent.bytes,
                )
                physical_object_bytes.append(key_extent.bytes)
                value_slot = len(physical_object_bytes)
                value_object_id = _OBJECT_ID_BASE | value_slot
                self._runtime.install_nvme_object(
                    value_slot,
                    value_object_id,
                    version,
                    value_extent.offset,
                    value_extent.bytes,
                )
                physical_object_bytes.append(value_extent.bytes)
                result = (key_slot, key_object_id, value_slot, value_object_id)
                pair_objects[pair] = result
                return result
            if self._tier_service.is_cxl:
                # CXL rows are direct dependencies, not runtime objects.  The
                # caller constructs the requirements from the same catalog so
                # the storage address never becomes an inferred/approximate
                # page-table mapping.
                raise AssertionError("CXL direct dependencies do not allocate objects")
            index_map = batch.index_maps.get(pair)
            if index_map is None:
                index_map = (
                    torch.tensor(
                        host_pages, dtype=torch.int32, device=key_cache.device
                    ),
                    torch.tensor(
                        device_pages, dtype=torch.int32, device=key_cache.device
                    ),
                )
                batch.index_maps[pair] = index_map
            source_indices, staging_indices = index_map
            index_tensors.extend((source_indices, staging_indices))
            key_slot = len(indexed_objects)
            key_object_id = _OBJECT_ID_BASE | key_slot
            indexed_objects.append(
                IndexedHostObject(
                    key_object_id,
                    version,
                    host_key.data_ptr(),
                    key_cache.data_ptr(),
                    source_indices.data_ptr(),
                    staging_indices.data_ptr(),
                    len(host_pages),
                    key_element_bytes,
                    host_key.stride(0) * host_key.element_size(),
                    key_cache.stride(0) * key_cache.element_size(),
                    int(host_key.shape[0]),
                    int(key_cache.shape[0]),
                )
            )
            value_slot = len(indexed_objects)
            value_object_id = _OBJECT_ID_BASE | value_slot
            indexed_objects.append(
                IndexedHostObject(
                    value_object_id,
                    version,
                    host_value.data_ptr(),
                    value_cache.data_ptr(),
                    source_indices.data_ptr(),
                    staging_indices.data_ptr(),
                    len(host_pages),
                    value_element_bytes,
                    host_value.stride(0) * host_value.element_size(),
                    value_cache.stride(0) * value_cache.element_size(),
                    int(host_value.shape[0]),
                    int(value_cache.shape[0]),
                )
            )
            result = (key_slot, key_object_id, value_slot, value_object_id)
            pair_objects[pair] = result
            return result

        semantic_units = []
        dependency_spans = []
        dependencies: list[AcquireRequirement] = []
        execution = batch.execution
        if execution is None:
            raise RuntimeError("native work-plan upload has no execution session")
        semantic_layer = int(layer_id) - self._model_start_layer
        if semantic_layer < 0:
            raise RuntimeError("native work-plan layer precedes the model partition")
        object_fanout: Counter[int] = Counter()
        unresolved_dependencies: list[int] = []
        direct_work_count = 0
        external_object_slots: list[tuple[int, ...]] = []
        for work_ticket, (request_index, kv_tile, pair) in enumerate(
            zip(schedule.request_indices, schedule.kv_tile_indices, page_pairs)
        ):
            binding = batch.bindings[request_index]
            semantic = execution.unit_for_ticket(
                work_id=work_ticket,
                layer=semantic_layer,
                logical_begin=int(kv_tile),
                request_index=request_index,
            )
            if semantic.binding != binding:
                raise RuntimeError(
                    "native work-plan identity diverged from semantic batch"
                )
            dependency_begin = len(dependencies)
            if pair[0]:
                if self._tier_service.is_cxl:
                    key_extent = self._tier_service.extent(
                        layer_id, tuple(pair[1]), "key", key_element_bytes
                    )
                    value_extent = self._tier_service.extent(
                        layer_id, tuple(pair[1]), "value", value_element_bytes
                    )
                    key_address = self._tier_service.device_address(key_extent)
                    value_address = self._tier_service.device_address(value_extent)
                    dependencies.extend(
                        (
                            AcquireRequirement(
                                key_address,
                                0,
                                _OBJECT_ID_BASE | 0xFFFFFFF0,
                                0,
                                0,
                                1,
                                key_extent.bytes,
                                0,
                            ),
                            AcquireRequirement(
                                value_address,
                                0,
                                _OBJECT_ID_BASE | 0xFFFFFFF1,
                                0,
                                1,
                                1,
                                value_extent.bytes,
                                0,
                            ),
                        )
                    )
                    direct_work_count += 1
                    external_object_slots.append(())
                    direct_dependencies = 2
                else:
                    key_slot, key_object_id, value_slot, value_object_id = objects_for(
                        pair
                    )
                    key_transfer_bytes = (
                        prefetched.key_bytes
                        if prefetched is not None
                        else len(pair[0]) * key_element_bytes
                    )
                    value_transfer_bytes = (
                        prefetched.value_bytes
                        if prefetched is not None
                        else len(pair[0]) * value_element_bytes
                    )
                    dependencies.extend(
                        (
                            AcquireRequirement(
                                0,
                                0,
                                key_object_id,
                                0,
                                key_slot,
                                version,
                                key_transfer_bytes,
                                0,
                            ),
                            AcquireRequirement(
                                0,
                                0,
                                value_object_id,
                                0,
                                value_slot,
                                version,
                                value_transfer_bytes,
                                0,
                            ),
                        )
                    )
                    object_fanout[key_slot] += 1
                    object_fanout[value_slot] += 1
                    unresolved_dependencies.append(2)
                    external_object_slots.append((key_slot, value_slot))
                    direct_dependencies = 0
            else:
                dependencies.extend(
                    (
                        AcquireRequirement(
                            self._runtime.device_view,
                            0,
                            _OBJECT_ID_BASE | 0xFFFFFFF0,
                            0,
                            0,
                            1,
                            1,
                            0,
                        ),
                        AcquireRequirement(
                            self._runtime.device_view,
                            0,
                            _OBJECT_ID_BASE | 0xFFFFFFF1,
                            0,
                            1,
                            1,
                            1,
                            0,
                        ),
                    )
                )
                direct_work_count += 1
                external_object_slots.append(())
                direct_dependencies = 2
            semantic_units.append(semantic)
            dependency_spans.append(
                (dependency_begin, 2, direct_dependencies, work_ticket)
            )

        ranges = _request_ranges(batch.bindings, schedule.request_indices)

        object_count = (
            2
            if prefetched is not None
            else len(indexed_objects)
            if self._tier_service.is_host
            else len(physical_object_bytes)
        )
        if object_count == 0 and self._tier_service.is_host:
            raise RuntimeError("external HiCache batch has no CTA dependency")
        if object_count > self._object_capacity:
            raise RuntimeError(
                f"HiCache layer needs {object_count} objects; configured capacity is "
                f"{self._object_capacity}"
            )
        if prefetched is None and pending.prefetched_layers:
            # key_slot/value_slot are the low alias slots used after a layer's
            # transfer completes. The proactive allocation itself remains in
            # the high range beginning at transfer_first_slot.
            pipeline_first_slot = min(
                layer.transfer_first_slot
                for layer in pending.prefetched_layers.values()
            )
            if object_count > pipeline_first_slot:
                raise RuntimeError("demand and proactive HiCache object ranges overlap")
        transfer_bytes = (
            prefetched.key_bytes + prefetched.value_bytes
            if prefetched is not None
            else sum(
                object_.index_count * object_.element_bytes
                for object_ in indexed_objects
            )
            if self._tier_service.is_host
            else sum(physical_object_bytes)
        )
        if self._tier_service.is_host:
            host_execution = plan_host_execution(
                object_count=object_count,
                transfer_bytes=transfer_bytes,
                runnable_tiles=schedule.work_count,
                initial_runnable_tiles=(
                    direct_work_count if self._overlap_enabled else 0
                ),
                model=self._host_cost_model,
                force_rounds=(
                    self._host_cost_model.max_rounds
                    if self._incremental_enabled
                    else None
                ),
            )
        else:
            host_execution = None
        stream = torch.cuda.current_stream()
        if self._tier_service.is_host and prefetched is None:
            self._runtime.register_indexed_host_objects(
                0, indexed_objects, stream=stream
            )
            self._phase_program(wrapper).validate_indexed_host_range(
                self._runtime, 0, object_count, stream
            )
        incremental = self._tier_service.is_host and (
            host_execution.rounds > 1
            or host_execution.overlap_initial
            or self._incremental_enabled
            or (self._frontier_enabled and local_layer == 0)
        )
        needs_plan = (
            not self._tier_service.is_host or prefetched is not None or incremental
        )
        if needs_plan and rebuild_plan:
            upload_started = time.perf_counter_ns() if self._profile_cpu else 0
            plan.upload_work_units(
                semantic_units,
                dependency_spans,
                dependencies,
                ranges,
                epoch=execution.epoch,
                stream=stream,
            )
            if self._profile_cpu:
                self._stats["native_plan_upload_cpu_ns"] = self._stats.get(
                    "native_plan_upload_cpu_ns", 0
                ) + (time.perf_counter_ns() - upload_started)
            self._stats["plan_uploads"] += 1
        if needs_plan and prefetched is not None:
            self._stats["cta_work_items"] += schedule.work_count
        allocation.signature = signature
        allocation.object_count = object_count
        allocation.index_tensors = tuple(index_tensors)
        allocation.host_execution = host_execution
        allocation.transfer_bytes = transfer_bytes
        allocation.indexed_geometry = indexed_geometry
        allocation.max_object_fanout = max(object_fanout.values(), default=1)
        allocation.min_unresolved_dependencies = min(unresolved_dependencies, default=1)
        allocation.direct_work_count = direct_work_count
        allocation.external_object_slots = tuple(external_object_slots)
        if self._tier_service.is_host and prefetched is None:
            transfer_bytes = sum(
                object_.index_count * object_.element_bytes
                for object_ in indexed_objects
            )
            self._record_demand_plan_stats(
                batch,
                schedule,
                object_count,
                transfer_bytes,
                host_execution,
            )
        elif self._tier_service.is_nvme:
            self._stats["cta_work_items"] += schedule.work_count
            self._stats["nvme_bytes"] += transfer_bytes
            self._stats["nvme_epochs"] += 1
        else:
            self._stats["cta_work_items"] += schedule.work_count
            self._stats["cxl_direct_work_items"] += direct_work_count
        if self._profile_cpu:
            self._stats["plan_cpu_ns"] = self._stats.get("plan_cpu_ns", 0) + (
                time.perf_counter_ns() - profile_started
            )
        return (
            plan,
            schedule,
            object_count,
            None if prefetched is None else prefetched.ready_event,
            0,
            host_execution,
        )

    def _work_page_pairs(
        self, wrapper: Any, schedule: Schedule, pending: PendingHostLoad
    ) -> tuple[_PagePair, ...]:
        indptr, pages, last_page, page_size = self._wrapper_layout(wrapper)
        host_by_device = pending.materialize_mapping()
        pairs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for request_index, kv_tile in zip(
            schedule.request_indices, schedule.kv_tile_indices
        ):
            if request_index < 0 or request_index + 1 >= len(indptr) or kv_tile < 0:
                raise RuntimeError("FlashInfer emitted an invalid KV tile coordinate")
            request_pages = pages[indptr[request_index] : indptr[request_index + 1]]
            if schedule.kv_chunk_tokens > 0:
                token_count = max(0, len(request_pages) - 1) * page_size + int(
                    last_page[request_index]
                )
                token_begin = kv_tile * schedule.kv_chunk_tokens
                token_end = min(token_count, token_begin + schedule.kv_chunk_tokens)
                page_begin = token_begin // page_size
                page_end = (token_end + page_size - 1) // page_size
                request_pages = request_pages[page_begin:page_end]
            device_pages = tuple(
                int(page) for page in request_pages if int(page) in host_by_device
            )
            host_pages = tuple(host_by_device[page] for page in device_pages)
            pairs.append((host_pages, device_pages))
        return tuple(pairs)

    def _enqueue_fragment_lookahead(
        self,
        wrapper: Any,
        layer_id: int,
        object_count: int,
        host_execution: HostExecutionPlan,
        stream: torch.cuda.Stream,
    ) -> None:
        """Stage one next-layer contributor wave during post-attention compute."""
        if (
            not self._fragment_enabled
            or self.num_wrappers != 1
            or host_execution.rounds <= 1
        ):
            return
        batch = self._active_batch
        if batch is None or batch.pending_host_load is None:
            return
        pending = batch.pending_host_load
        device_pool = pending.controller.mem_pool_device
        start_layer = int(getattr(device_pool, "start_layer", 0))
        next_layer_id = layer_id + 1
        next_local_layer = next_layer_id - start_layer
        if next_local_layer < 0 or next_local_layer >= int(
            pending.controller.layer_num
        ):
            return
        if next_layer_id in batch.fragment_lookahead:
            raise RuntimeError("duplicate fragment lookahead for one attention layer")

        allocation = self._plans.get((id(wrapper), -1))
        if (
            allocation is None
            or allocation.object_count != object_count
            or allocation.indexed_geometry is None
            or object_count % 2 != 0
        ):
            raise RuntimeError("fragment lookahead has no reusable indexed directory")
        first_object_count = host_execution.block_counts[0]
        if (
            first_object_count <= 0
            or first_object_count >= object_count
            or first_object_count % 2 != 0
        ):
            raise RuntimeError("fragment lookahead requires one complete K/V wave")

        host_key = pending.controller.mem_pool_host.k_data_refs[next_local_layer]
        host_value = pending.controller.mem_pool_host.v_data_refs[next_local_layer]
        key_cache = device_pool._get_key_buffer(next_layer_id)
        value_cache = device_pool._get_value_buffer(next_layer_id)
        geometry = (
            key_cache[0].numel() * key_cache.element_size(),
            value_cache[0].numel() * value_cache.element_size(),
            host_key.stride(0) * host_key.element_size(),
            host_value.stride(0) * host_value.element_size(),
            key_cache.stride(0) * key_cache.element_size(),
            value_cache.stride(0) * value_cache.element_size(),
            int(host_key.shape[0]),
            int(host_value.shape[0]),
            int(key_cache.shape[0]),
            int(value_cache.shape[0]),
        )
        if geometry != allocation.indexed_geometry:
            raise RuntimeError(
                "next-layer KV geometry changed during fragment lookahead"
            )

        attention_done = torch.cuda.Event()
        ready_event = torch.cuda.Event()
        attention_done.record(stream)
        phase_program = self._phase_program(wrapper)
        with torch.cuda.stream(self._prefetch_stream):
            self._prefetch_stream.wait_event(attention_done)
            phase_program.rebind_indexed_host_pairs(
                self._runtime,
                0,
                object_count // 2,
                host_key.data_ptr(),
                key_cache.data_ptr(),
                host_value.data_ptr(),
                value_cache.data_ptr(),
                self._prefetch_stream,
            )
            phase_program.preload_host_pairs(
                self._runtime,
                0,
                first_object_count // 2,
                self._prefetch_stream,
            )
            ready_event.record(self._prefetch_stream)
        batch.fragment_lookahead[next_layer_id] = _FragmentLookahead(
            next_layer_id,
            id(wrapper),
            object_count,
            first_object_count,
            host_key.data_ptr(),
            key_cache.data_ptr(),
            host_value.data_ptr(),
            value_cache.data_ptr(),
            ready_event,
        )
        self._stats["fragment_lookahead_layers"] += 1
        self._stats["fragment_lookahead_objects"] += first_object_count
        self._stats["fragment_remaining_rounds"] += host_execution.rounds - 1

    def _phase_program(self, wrapper: Any) -> JitPhaseProgram:
        module_name = self._wrapper_modules[id(wrapper)]
        program = self._phase_programs.get(module_name)
        if program is not None:
            return program
        workspace_value = os.environ.get("FLASHINFER_WORKSPACE_BASE")
        if not workspace_value:
            raise RuntimeError(
                "FLASHINFER_WORKSPACE_BASE is missing; run SGLang through "
                "tools/jit/activate.py --flashinfer-hook"
            )
        modules = sorted(pathlib.Path(workspace_value).rglob(f"{module_name}.so"))
        if len(modules) != 1:
            raise RuntimeError(
                f"expected one compiled FlashInfer module {module_name}.so; "
                f"found {len(modules)}"
            )
        program = JitPhaseProgram(modules[0])
        family = (
            OperatorFamily.FLASHINFER_DECODE
            if "decode" in module_name
            else OperatorFamily.FLASHINFER_PAGED_PREFILL
        )
        form = (
            OperatorForm.DIRECT
            if "request_bound" in module_name
            else OperatorForm.INCREMENTAL
        )
        required = (
            OperatorCapability.REQUEST_BINDING
            | OperatorCapability.TYPED_FLASHINFER_FRONTEND
        )
        if form == OperatorForm.DIRECT:
            required |= OperatorCapability.GRAPH_REPLAY
        else:
            required |= (
                OperatorCapability.OBJECT_DEPENDENCIES
                | OperatorCapability.FINITE_DEFERRAL
                | OperatorCapability.PARTIAL_PUBLICATION
                | OperatorCapability.COMPLETE_CONTRIBUTOR_MERGE
                | OperatorCapability.RUNNABLE_COMPACTION
            )
        contract = program.operator_contract
        contract.require(
            family=family,
            form=form,
            capabilities=required,
            instrumentation=(
                OperatorInstrumentation.TYPED_ACCESS_LOWERING
                | OperatorInstrumentation.EXACT_DEMAND
                | OperatorInstrumentation.GENERATION_SAFE_IDENTITY
                | OperatorInstrumentation.TIER_OWNERSHIP
            ),
            identity_binding=OperatorIdentityBinding.REQUEST_SLOT_GENERATION,
            demand_binding=OperatorDemandBinding.EXACT_WORK_UNIT,
            access_proof=OperatorAccessProof.TYPED_FRONTEND,
            tier_mask=(1 << 6) - 1,
        )
        plan = program.operator_plan
        plan.require(
            family=family,
            forms=(OperatorForm.DIRECT, OperatorForm.INCREMENTAL),
            coordinate_map=OperatorCoordinateMap.FLASHINFER_REQUEST_CONTIGUOUS,
            partial_state=OperatorPartialState.ONLINE_SOFTMAX_VALUE_LSE,
            reduction=OperatorReduction.ORDERED_MERGE_STATE,
            flags=(
                OperatorPlanFlag.FIXED_CAPACITY
                | OperatorPlanFlag.GRAPH_STABLE
                | OperatorPlanFlag.EXTERNAL_WAVE_SOURCES
                | OperatorPlanFlag.GENERATION_BOUND
                | OperatorPlanFlag.EXACT_COMPLETE_MERGE
            ),
        )
        peer_form = (
            OperatorForm.INCREMENTAL
            if form == OperatorForm.DIRECT
            else OperatorForm.DIRECT
        )
        peer = self._operator_contracts.get((family, peer_form))
        peer_program = self._operator_programs.get((family, peer_form))
        if peer is not None:
            if peer_program is None:
                program.close()
                raise RuntimeError("paired FlashInfer contract has no loaded program")
            try:
                if form == OperatorForm.DIRECT:
                    require_operator_pair(program, peer_program)
                else:
                    require_operator_pair(peer_program, program)
            except Exception:
                program.close()
                raise
        self._operator_contracts[(family, form)] = contract
        self._operator_plans[(family, form)] = plan
        self._operator_programs[(family, form)] = program
        self._stats["verified_operator_modules"] += 1
        self._phase_programs[module_name] = program
        return program

    def _layer_sync_events(
        self,
        layer_id: int,
        progress_rounds: int,
        stream: torch.cuda.Stream,
    ) -> tuple[torch.cuda.Event, tuple[torch.cuda.Event, ...]]:
        stream_address = int(stream.cuda_stream)
        progress_address = int(self._progress_stream.cuda_stream)
        key = (layer_id, stream_address, progress_address)
        existing = self._demand_sync_events.get(key)
        if existing is not None and len(existing[1]) == progress_rounds:
            return existing
        events = (
            torch.cuda.Event(),
            tuple(torch.cuda.Event() for _ in range(progress_rounds)),
        )
        self._demand_sync_events[key] = events
        return events

    def _enqueue_demand_graph(
        self,
        key: _DemandGraphKey,
        wrapper: Any,
        query: torch.Tensor,
        output: torch.Tensor,
        stream: torch.cuda.Stream,
        enqueue: Callable[
            [
                torch.Tensor,
                torch.Tensor,
                tuple[torch.cuda.Event, tuple[torch.cuda.Event, ...]],
                Callable[[Any], None] | None,
            ],
            None,
        ],
        eager_events: tuple[torch.cuda.Event, tuple[torch.cuda.Event, ...]],
        on_discovered: Callable[[Any], None] | None,
    ) -> torch.Tensor:
        """Warm, capture, or replay one finite incremental operator."""
        captured = self._demand_graphs.get(key)
        if captured is not None:
            self._reserve_demand_graph_key(key, stream)
            current_metadata = dict(_graph_wrapper_metadata(wrapper))
            for name, static_tensor in captured.wrapper_metadata:
                current = current_metadata.get(name)
                if current is None:
                    raise RuntimeError(
                        f"FlashInfer graph replay lost metadata tensor {name}"
                    )
                if (
                    current.shape != static_tensor.shape
                    or current.stride() != static_tensor.stride()
                    or current.dtype != static_tensor.dtype
                    or current.device != static_tensor.device
                ):
                    raise RuntimeError(
                        f"FlashInfer graph metadata layout changed for {name}"
                    )
                static_tensor.copy_(current, non_blocking=True)
            captured.query.copy_(query, non_blocking=True)
            captured.graph.replay()
            self._stats["demand_graph_replays"] += 1
            family_counter = f"demand_graph_{key.operator_family}_replays"
            self._stats[family_counter] = self._stats.get(family_counter, 0) + 1
            return captured.output

        if key not in self._demand_graph_warmups:
            enqueue(query, output, eager_events, on_discovered)
            self._reserve_demand_graph_key(key, stream)
            self._stats["demand_graph_warmups"] += 1
            family_counter = f"demand_graph_{key.operator_family}_warmups"
            self._stats[family_counter] = self._stats.get(family_counter, 0) + 1
            return output

        self._reserve_demand_graph_key(key, stream)
        static_query = torch.empty_like(query)
        static_output = torch.empty_like(output)
        static_query.copy_(query, non_blocking=True)
        discovery_done = torch.cuda.Event()
        arrival_events = tuple(
            torch.cuda.Event() for _ in range(len(key.progress_blocks))
        )
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream, capture_error_mode="global"):
                enqueue(
                    static_query,
                    static_output,
                    (discovery_done, arrival_events),
                    None,
                )
        except Exception as error:
            raise RuntimeError(
                "failed to capture the finite NTA demand operator graph"
            ) from error
        self._demand_graphs[key] = _DemandGraph(
            graph,
            static_query,
            static_output,
            (discovery_done, *arrival_events),
            _graph_wrapper_metadata(wrapper),
        )
        self._stats["demand_graph_captures"] += 1
        family_counter = f"demand_graph_{key.operator_family}_captures"
        self._stats[family_counter] = self._stats.get(family_counter, 0) + 1
        # Capture records the finite operator but does not produce this call's
        # output. Launch it once after instantiation, preserving stream order
        # with the current plan, directory, and static-query upload.
        graph.replay()
        self._stats["demand_graph_replays"] += 1
        family_counter = f"demand_graph_{key.operator_family}_replays"
        self._stats[family_counter] = self._stats.get(family_counter, 0) + 1
        return static_output

    def _run_attention(
        self,
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        layer: Any,
        *,
        causal: bool,
        window_left: int,
    ) -> torch.Tensor:
        if self._cuda_graph_mode:
            return self._run_graph_attention(
                wrapper,
                q,
                kv_cache,
                layer,
                causal=causal,
                window_left=window_left,
            )
        if layer.logit_cap not in (None, 0, 0.0):
            raise RuntimeError("NTA's FlashInfer adapter does not support logit caps")
        q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        output = torch.empty_like(q)
        verify_attention = os.environ.get("NTA_VERIFY_ATTENTION") == "1"
        if (
            verify_attention
            and os.environ.get("NTA_VERIFY_ATTENTION_MIXED_ONLY") == "1"
        ):
            verify_attention = len(self._active_batch.bindings) > 1
        verify_execution = (
            verify_attention or os.environ.get("NTA_VERIFY_EXECUTION") == "1"
        )
        if verify_execution:
            output.fill_(float("nan"))
        wrapper._causal = causal
        wrapper._window_left = window_left
        wrapper._logits_soft_cap = 0.0
        wrapper._sm_scale = layer.scaling
        batch = self._active_batch
        if batch is None:
            raise RuntimeError("NTA attention ran without request metadata")
        self._ensure_execution_session(wrapper, layer, kv_cache)
        pending = batch.pending_host_load
        stream = torch.cuda.current_stream()
        run_options = {
            "k_scale": layer.k_scale_float,
            "v_scale": layer.v_scale_float,
        }
        final_layer = (
            int(layer.layer_id) - self._model_start_layer + 1 == self._model_layer_count
        )
        enqueue_started = time.perf_counter_ns() if self._profile_cpu else 0
        gpu_profile = None
        if self._profile_gpu:
            gpu_profile = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            gpu_profile[0].record(stream)
        attention_form = "direct"
        epoch = None
        progress_rounds = 0
        if pending is None:
            self._run_preacquired_attention(
                wrapper, q, kv_cache, output, layer, run_options
            )
            local_layer = -1
        else:
            local_layer = int(layer.layer_id) - int(
                getattr(pending.controller.mem_pool_device, "start_layer", 0)
            )
            prefetched = batch.prefetched_layers.get(local_layer)
        if pending is not None and prefetched is not None:
            attention_form = "preloaded"
            if self._profile_barrier:
                arrive = torch.cuda.Event(enable_timing=True)
                arrive.record(stream)
                self._barrier_profiles.append(
                    (arrive, prefetched.ready_event, int(layer.layer_id))
                )
            stream.wait_event(prefetched.ready_event)
            # The preloaded form must not depend on a mixed/demand layer having
            # populated the structural plan first: batch composition is
            # timing-dependent, and a pure-preloaded batch is legal. The call
            # is a signature-checked cache hit after the first layer.
            self._upload_plan(wrapper, int(layer.layer_id), kv_cache)
            self._run_preacquired_attention(
                wrapper, q, kv_cache, output, layer, run_options
            )
            self._stats["lookahead_bound_launches"] += 1
        elif pending is not None and self._tier_service.is_cxl:
            attention_form = "cxl_direct"
            plan, schedule, object_count, _ready, _preloaded, _execution = (
                self._upload_plan(wrapper, int(layer.layer_id), kv_cache)
            )
            if plan.has_external or object_count != 0:
                raise RuntimeError(
                    "CXL direct plan unexpectedly contains staged objects"
                )
            enqueue_resident_attention(
                self._runtime,
                plan,
                wrapper,
                q,
                kv_cache,
                output,
                sm_scale=layer.scaling,
                run_options=run_options,
            )
            self._stats["request_work_completed"] += schedule.work_count
            self._stats["tier_external_layers"] += 1
            self._stats["transformed_direct_launches"] += 1
        elif pending is not None and self._tier_service.is_nvme:
            attention_form = "nvme"
            plan, schedule, object_count, _ready, _preloaded, _execution = (
                self._upload_plan(wrapper, int(layer.layer_id), kv_cache)
            )
            if not plan.has_external or object_count <= 0:
                raise RuntimeError(
                    "NVMe plan unexpectedly contains no external objects"
                )
            epoch = FlashInferLayerEpoch(
                self._runtime,
                plan,
                self._phase_program(wrapper),
                object_count=object_count,
                max_progress_rounds=self._tier_service.config.progress_rounds,
                wait_for_plan=False,
            )
            progress_rounds = epoch.enqueue_nvme(
                wrapper,
                q,
                kv_cache,
                output,
                issue_budget=self._tier_service.config.issue_budget,
                completion_budget=self._tier_service.config.completion_budget,
                timeout_ns=self._tier_service.config.progress_timeout_ns,
                sm_scale=layer.scaling,
                stream=stream,
                run_options=run_options,
            )
            self._stats["nvme_progress_rounds"] += progress_rounds
            self._stats["request_work_completed"] += schedule.work_count
            self._stats["tier_external_layers"] += 1
            self._stats["ticketed_incremental_launches"] += 1
            allocation = self._plans[(id(wrapper), -1)]
            if 0 < allocation.direct_work_count < schedule.work_count:
                self._stats["mixed_dependency_layers"] += 1
            if (
                final_layer
                or verify_execution
                or os.environ.get("NTA_VERIFY_TRANSFER") == "1"
            ):
                epoch.check(progress_rounds, stream)
            if final_layer and self._runtime.sticky_failed_count != 0:
                raise RuntimeError("an asynchronous NVMe acquisition epoch failed")
        elif pending is not None:
            execution_plan = self._layer_execution_plan(wrapper, kv_cache)
            if (
                execution_plan.rounds == 1
                and not execution_plan.overlap_initial
                and not self._incremental_enabled
                and not (self._frontier_enabled and local_layer == 0)
            ):
                attention_form = "bulk"
                self._run_bulk_host_layer(
                    wrapper,
                    q,
                    kv_cache,
                    output,
                    layer,
                    run_options,
                    stream,
                )
                self._stats["bulk_host_batches"] += 1
                self._stats["predicted_atomic_ns"] += execution_plan.predicted_atomic_ns
                self._stats["predicted_incremental_ns"] += (
                    execution_plan.predicted_incremental_ns
                )
            else:
                attention_form = "incremental"
                (
                    plan,
                    schedule,
                    object_count,
                    preloaded_event,
                    preloaded_object_count,
                    host_execution,
                ) = self._upload_plan(wrapper, int(layer.layer_id), kv_cache)
                if host_execution != execution_plan:
                    raise RuntimeError("host execution plan changed during planning")
                progress_blocks = host_execution.block_counts
                if preloaded_event is not None:
                    if preloaded_object_count != progress_blocks[0]:
                        raise RuntimeError(
                            "fragment lookahead does not match the execution plan"
                        )
                    progress_blocks = progress_blocks[1:]
                if not progress_blocks:
                    raise RuntimeError("incremental attention has no remaining wave")
                allocation = self._plans[(id(wrapper), -1)]
                if 0 < allocation.direct_work_count < schedule.work_count:
                    self._stats["mixed_dependency_layers"] += 1
                initial_ready_work_count = allocation.direct_work_count + sum(
                    1
                    for object_slots in allocation.external_object_slots
                    if object_slots
                    and all(
                        object_slot < preloaded_object_count
                        for object_slot in object_slots
                    )
                )
                if initial_ready_work_count >= schedule.work_count:
                    raise RuntimeError(
                        "incremental attention has no work after its initial fragment"
                    )
                self._stats["compact_initial_launches"] += int(
                    initial_ready_work_count != 0
                )
                self._stats["compact_initial_cta_bound"] += initial_ready_work_count
                self._stats["canonical_initial_cta_bound"] += schedule.work_count
                ready_work_counts = conservative_resume_counts(
                    block_counts=tuple(progress_blocks),
                    work_count=schedule.work_count - initial_ready_work_count,
                    max_object_fanout=allocation.max_object_fanout,
                    min_unresolved_dependencies=allocation.min_unresolved_dependencies,
                )
                self._stats["compact_resume_launches"] += len(ready_work_counts)
                self._stats["compact_resume_cta_bound"] += sum(ready_work_counts)
                self._stats["canonical_resume_cta_bound"] += (
                    len(ready_work_counts) * schedule.work_count
                )
                epoch = FlashInferLayerEpoch(
                    self._runtime,
                    plan,
                    self._phase_program(wrapper),
                    object_count=object_count,
                    max_progress_rounds=len(progress_blocks),
                    wait_for_plan=False,
                )
                progress_rounds = len(progress_blocks)
                transfer_profile = None
                if self._profile_transfer:
                    transfer_profile = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                on_discovered = None
                if local_layer == 0 and self._hicache.progress_publication_available():
                    request_slots = tuple(
                        binding.request_slot for binding in batch.bindings
                    )
                    first_request_slot = min(request_slots)
                    contiguous = request_slots == tuple(
                        range(
                            first_request_slot, first_request_slot + len(request_slots)
                        )
                    )
                    if contiguous:
                        progress_snapshot = self._runtime.request_progress_snapshot(
                            len(request_slots)
                        )

                        def publish_progress(discovery_stream: Any) -> None:
                            progress_snapshot.capture(
                                first_request_slot,
                                len(request_slots),
                                discovery_stream,
                            )
                            self._hicache.publish_request_progress(
                                progress_snapshot,
                                batch.bindings,
                                bandwidth_bytes_per_second=(
                                    self._host_cost_model.bandwidth_bytes_per_second
                                ),
                                fixed_latency_ns=self._host_cost_model.round_overhead_ns,
                            )
                            self._stats["progress_feedback_snapshots"] = (
                                self._stats.get("progress_feedback_snapshots", 0) + 1
                            )

                        on_discovered = publish_progress
                    else:
                        self._stats["progress_feedback_skipped_noncontiguous"] = (
                            self._stats.get(
                                "progress_feedback_skipped_noncontiguous", 0
                            )
                            + 1
                        )
                copy_blocks_per_group = indexed_copy_blocks_per_group(
                    transfer_bytes=allocation.transfer_bytes,
                    object_count=object_count,
                    target_bytes_per_block=self._indexed_copy_target_bytes,
                    maximum_blocks=self._indexed_copy_max_blocks,
                )

                def enqueue_demand(
                    query: torch.Tensor,
                    destination: torch.Tensor,
                    sync_events: tuple[torch.cuda.Event, tuple[torch.cuda.Event, ...]],
                    discovery_callback: Callable[[Any], None] | None,
                ) -> None:
                    epoch.enqueue_host(
                        wrapper,
                        query,
                        kv_cache,
                        destination,
                        progress_blocks=progress_blocks,
                        sm_scale=layer.scaling,
                        stream=stream,
                        progress_stream=self._progress_stream,
                        ready_event=preloaded_event,
                        ready_work_counts=ready_work_counts,
                        initial_ready_work_count=initial_ready_work_count,
                        indexed_host_first_object=preloaded_object_count,
                        indexed_host_prevalidated=True,
                        indexed_host_copy_blocks_per_group=copy_blocks_per_group,
                        sync_events=sync_events,
                        progress_profile=transfer_profile,
                        on_discovered=discovery_callback,
                        run_options=run_options,
                    )

                eager_events = self._layer_sync_events(
                    int(layer.layer_id), len(progress_blocks), stream
                )
                graph_eligible = (
                    self._demand_graph_enabled
                    and isinstance(
                        wrapper,
                        (
                            BatchDecodeWithPagedKVCacheWrapper,
                            BatchPrefillWithPagedKVCacheWrapper,
                        ),
                    )
                    and preloaded_event is None
                    and preloaded_object_count == 0
                    and transfer_profile is None
                )
                if graph_eligible:
                    graph_key = _demand_graph_key(
                        operator_family=(
                            "decode"
                            if isinstance(wrapper, BatchDecodeWithPagedKVCacheWrapper)
                            else "paged_prefill"
                        ),
                        wrapper=wrapper,
                        layer_id=int(layer.layer_id),
                        plan=plan,
                        runtime_tensor=self._runtime.device_view_tensor,
                        work_count=schedule.work_count,
                        object_count=object_count,
                        progress_blocks=tuple(progress_blocks),
                        ready_work_counts=tuple(ready_work_counts),
                        initial_ready_work_count=initial_ready_work_count,
                        indexed_copy_blocks_per_group=copy_blocks_per_group,
                        query=q,
                        kv_cache=kv_cache,
                        sm_scale=layer.scaling,
                        k_scale=layer.k_scale_float,
                        v_scale=layer.v_scale_float,
                        causal=causal,
                        window_left=window_left,
                    )
                    output = self._enqueue_demand_graph(
                        graph_key,
                        wrapper,
                        q,
                        output,
                        stream,
                        enqueue_demand,
                        eager_events,
                        on_discovered,
                    )
                else:
                    enqueue_demand(q, output, eager_events, on_discovered)
                self._stats["parallel_indexed_progress_layers"] += 1
                self._stats["prevalidated_indexed_progress_layers"] = (
                    self._stats.get("prevalidated_indexed_progress_layers", 0) + 1
                )
                if transfer_profile is not None:
                    self._transfer_profiles.append(
                        (*transfer_profile, allocation.transfer_bytes, "demand")
                    )
                if (
                    self._frontier_enabled
                    and local_layer == 0
                    and self._model_layer_count > 1
                    and host_execution.rounds == 1
                    and not pending.prefetched_layers
                ):
                    self._prepare_cross_layer_frontier(pending)
                    batch.prefetched_layers.update(pending.prefetched_layers)
                self._enqueue_fragment_lookahead(
                    wrapper,
                    int(layer.layer_id),
                    object_count,
                    host_execution,
                    stream,
                )
                self._stats["ticketed_incremental_launches"] += 1
                collect_progress = (
                    verify_execution or self._opportunity_trace is not None
                )
                verify_transfer = os.environ.get("NTA_VERIFY_TRANSFER") == "1"
                if final_layer or collect_progress or verify_transfer:
                    epoch.check(progress_rounds, stream)
                if final_layer and self._runtime.sticky_failed_count != 0:
                    raise RuntimeError(
                        "an earlier asynchronous acquisition epoch failed"
                    )
                if collect_progress:
                    request_slots = tuple(
                        binding.request_slot for binding in batch.bindings
                    )
                    first_request_slot = min(request_slots)
                    progress_range = self._runtime.request_progress_range(
                        first_request_slot,
                        max(request_slots) - first_request_slot + 1,
                    )
                    progress = tuple(
                        progress_range[request_slot - first_request_slot]
                        for request_slot in request_slots
                    )
                    external_requests = {
                        schedule.request_indices[index]
                        for index, object_slots in enumerate(
                            allocation.external_object_slots
                        )
                        if object_slots
                    }
                    if any(
                        item.failed_work != 0
                        or item.cancelled_work != 0
                        or item.dropped_attributions != 0
                        or item.completed_work != item.expected_work
                        or item.pending_work != 0
                        or item.runnable_work != 0
                        or item.unavailable_bytes != 0
                        or item.runnable_compute_ns != 0
                        or item.pending_compute_ns != 0
                        or item.completed_compute_ns != item.expected_compute_ns
                        for item in progress
                    ):
                        raise RuntimeError(
                            "request-level progress disagrees with the completed epoch"
                        )
                    if any(
                        progress[request_index].expected_work == 0
                        for request_index in external_requests
                    ):
                        raise RuntimeError(
                            "external request produced no progress attribution"
                        )
                    self._stats["progress_snapshots"] += 1
                    self._stats["request_work_completed"] += sum(
                        item.completed_work for item in progress
                    )
                    self._stats["request_work_failed"] += sum(
                        item.failed_work + item.cancelled_work for item in progress
                    )
                    self._stats["request_compute_completed_ns"] += sum(
                        item.completed_compute_ns for item in progress
                    )
                    self._stats["request_compute_expected_ns"] = self._stats.get(
                        "request_compute_expected_ns", 0
                    ) + sum(item.expected_compute_ns for item in progress)
                else:
                    self._stats["request_work_completed"] += schedule.work_count
                    self._stats["request_compute_completed_ns"] += (
                        schedule.work_count * self._host_cost_model.tile_compute_ns
                    )
                if self._opportunity_trace is not None:
                    tile_compute_ns = self._host_cost_model.tile_compute_ns
                    compute_source = "calibrated"
                    if self._measure_opportunity_compute:
                        tile_compute_ns = self._measure_flashinfer_tile_compute(
                            wrapper,
                            q,
                            kv_cache,
                            output,
                            layer,
                            run_options,
                            schedule.work_count,
                        )
                        compute_source = "measured"
                    runnable_ns = self._runtime.work_runnable_ns(schedule.work_count)
                    tiles = tuple(
                        TileArrival(
                            request_id=(
                                f"{batch.bindings[request_index].request_id:016x}"
                            ),
                            tile_id=work_ticket,
                            available_ns=runnable_ns[work_ticket],
                            compute_ns=tile_compute_ns,
                            logical_tile=schedule.kv_tile_indices[work_ticket],
                            availability_source=(
                                "resident_at_launch"
                                if runnable_ns[work_ticket] == 0
                                else "gpu_globaltimer"
                            ),
                            compute_source=compute_source,
                        )
                        for work_ticket, request_index in enumerate(
                            schedule.request_indices
                        )
                    )
                    append_json_line(
                        self._opportunity_trace,
                        OperatorArrival(
                            batch_id=(
                                f"{os.getpid()}:{self._active_opportunity_batch}"
                            ),
                            layer=int(layer.layer_id),
                            tiles=tiles,
                            revision=self._opportunity_revision,
                            engine="sglang",
                            model=self._opportunity_model,
                            tier=self._opportunity_tier,
                            observed_at_unix_ns=time.time_ns(),
                        ),
                    )
        if self._profile_cpu:
            elapsed_ns = time.perf_counter_ns() - enqueue_started
            self._stats["phase_enqueue_cpu_ns"] = (
                self._stats.get("phase_enqueue_cpu_ns", 0) + elapsed_ns
            )
            self._stats[f"{attention_form}_enqueue_cpu_ns"] = (
                self._stats.get(f"{attention_form}_enqueue_cpu_ns", 0) + elapsed_ns
            )
            self._stats[f"{attention_form}_enqueue_layers"] = (
                self._stats.get(f"{attention_form}_enqueue_layers", 0) + 1
            )
        if gpu_profile is not None:
            gpu_profile[1].record(stream)
            self._operator_profiles.append((*gpu_profile, attention_form))
        if (
            pending is not None
            and self._tier_service.is_host
            and os.environ.get("NTA_VERIFY_TRANSFER") == "1"
        ):
            self._verify_layer_transfer(int(layer.layer_id), kv_cache)
        if verify_execution:
            if epoch is None:
                stream.synchronize()
            if not torch.isfinite(output).all():
                raise RuntimeError(
                    f"instrumented FlashInfer did not write layer {layer.layer_id}"
                )
        if verify_attention:
            self._verify_attention_output(
                wrapper,
                q,
                kv_cache,
                output,
                layer,
                causal=causal,
                window_left=window_left,
            )
        self._record_execution_layer(layer)
        if pending is not None:
            self._stats["external_launches"] += 1
            self._hicache.complete_layer(pending, local_layer)
        if final_layer:
            self._publish_stats()
        return output.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _measure_flashinfer_tile_compute(
        self,
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        output: torch.Tensor,
        layer: Any,
        run_options: dict[str, Any],
        work_count: int,
    ) -> int:
        """Calibrate canonical tile cost after all data is resident.

        This evaluation-only launch contains no acquisition or progress work.
        Its kernel makespan is converted to a per-tile service cost for the
        analyzer's declared number of parallel CTA slots.
        """
        if work_count <= 0 or id(wrapper) not in self._wrapper_modules:
            raise RuntimeError("compute calibration requires instrumented CTA work")
        batch = self._active_batch
        if batch is None:
            raise RuntimeError("compute calibration has no active batch")
        start = torch.cuda.Event(enable_timing=True)
        finish = torch.cuda.Event(enable_timing=True)
        runtime_tensor = self._runtime.device_view_tensor
        start.record()
        wrapper.run(
            q,
            kv_cache,
            runtime_tensor,
            layer.scaling,
            batch.bindings[0].request_slot,
            out=output,
            **run_options,
        )
        finish.record()
        finish.synchronize()
        kernel_ns = max(1, math.ceil(start.elapsed_time(finish) * 1_000_000))
        active_slots = min(work_count, self._opportunity_parallel_slots)
        tile_ns = max(1, math.ceil(kernel_ns * active_slots / work_count))
        self._stats["opportunity_calibration_launches"] = (
            self._stats.get("opportunity_calibration_launches", 0) + 1
        )
        self._stats["opportunity_calibration_kernel_ns"] = (
            self._stats.get("opportunity_calibration_kernel_ns", 0) + kernel_ns
        )
        return tile_ns

    def _run_graph_attention(
        self,
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        layer: Any,
        *,
        causal: bool,
        window_left: int,
    ) -> torch.Tensor:
        if layer.logit_cap not in (None, 0, 0.0):
            raise RuntimeError("NTA's FlashInfer adapter does not support logit caps")
        batch = self._active_batch
        if batch is None:
            raise RuntimeError("NTA graph attention ran without request metadata")
        local_layer = int(layer.layer_id) - self._model_start_layer
        if local_layer < 0 or local_layer >= self._model_layer_count:
            raise RuntimeError(
                f"attention layer {layer.layer_id} is outside NTA graph state"
            )
        q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        output = torch.empty_like(q)
        wrapper._causal = causal
        wrapper._window_left = window_left
        wrapper._logits_soft_cap = 0.0
        wrapper._sm_scale = layer.scaling
        if id(wrapper) not in self._wrapper_modules:
            raise RuntimeError(
                "NTA graph replay requires a compiler-transformed FlashInfer wrapper"
            )
        self._phase_program(wrapper)
        runtime_tensor = self._runtime.device_view_tensor
        request_slots = tuple(binding.request_slot for binding in batch.bindings)
        if not request_slots or request_slots != tuple(
            range(request_slots[0], request_slots[0] + len(request_slots))
        ):
            raise RuntimeError("NTA graph attention requires contiguous request slots")
        self._ensure_execution_session(wrapper, layer, kv_cache)
        wrapper.run(
            q,
            kv_cache,
            runtime_tensor,
            layer.scaling,
            request_slots[0],
            out=output,
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )
        self._record_execution_layer(layer)
        self._stats["transformed_direct_launches"] += 1
        if local_layer + 1 == self._model_layer_count:
            self._publish_stats()
        return output.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _verify_attention_output(
        self,
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        actual: torch.Tensor,
        layer: Any,
        *,
        causal: bool,
        window_left: int,
    ) -> None:
        workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=q.device)
        batch_size = int(wrapper._batch_size)
        kv_indptr = wrapper._paged_kv_indptr_buf[: batch_size + 1]
        page_count = int(kv_indptr[-1].item())
        kv_indices = wrapper._paged_kv_indices_buf[:page_count]
        last_page_len = wrapper._paged_kv_last_page_len_buf[:batch_size]
        num_kv_heads = int(kv_cache[0].shape[-2])
        if isinstance(wrapper, BatchDecodeWithPagedKVCacheWrapper):
            reference_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                workspace,
                "NHD",
                backend="fa2",
                use_tensor_cores=self.decode_use_tensor_cores,
            )
            reference_wrapper.plan(
                kv_indptr,
                kv_indices,
                last_page_len,
                int(q.shape[1]),
                num_kv_heads,
                int(q.shape[2]),
                1,
                window_left=window_left,
                sm_scale=layer.scaling,
                q_data_type=q.dtype,
                kv_data_type=kv_cache[0].dtype,
            )
        else:
            qo_indptr = wrapper._qo_indptr_buf[: batch_size + 1]
            reference_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                workspace, "NHD", backend="fa2"
            )
            reference_wrapper.plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                last_page_len,
                int(q.shape[1]),
                num_kv_heads,
                int(q.shape[2]),
                1,
                causal=causal,
                window_left=window_left,
                sm_scale=layer.scaling,
                q_data_type=q.dtype,
                kv_data_type=kv_cache[0].dtype,
            )
        expected = reference_wrapper.run(
            q,
            kv_cache,
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )
        torch.cuda.current_stream().synchronize()
        difference = (actual.float() - expected.float()).abs()
        maximum = float(difference.max().item())
        mean = float(difference.mean().item())
        finite_fraction = float(torch.isfinite(actual).float().mean().item())
        actual_absmax = float(torch.nan_to_num(actual.float()).abs().max().item())
        expected_absmax = float(expected.float().abs().max().item())
        self._stats["last_attention_max_abs_error"] = maximum
        self._stats["last_attention_mean_abs_error"] = mean
        if not torch.allclose(actual, expected, rtol=2e-3, atol=2e-3):
            raise RuntimeError(
                "instrumented FlashInfer output differs from stock "
                f"(layer={layer.layer_id}, max={maximum:.6g}, mean={mean:.6g}, "
                f"finite={finite_fraction:.6g}, actual_absmax={actual_absmax:.6g}, "
                f"expected_absmax={expected_absmax:.6g})"
            )

    def _verify_layer_transfer(
        self, layer_id: int, kv_cache: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        pending = self._active_batch.pending_host_load
        if pending is None:
            raise RuntimeError("layer transfer verification has no HiCache transfer")
        controller = pending.controller
        local_layer = layer_id - int(
            getattr(controller.mem_pool_device, "start_layer", 0)
        )
        mapping = pending.materialize_mapping()
        device_pages = torch.tensor(
            tuple(mapping), dtype=torch.long, device=kv_cache[0].device
        )
        host_pages = torch.tensor(tuple(mapping.values()), dtype=torch.long)
        torch.cuda.current_stream().synchronize()
        expected_key = controller.mem_pool_host.k_data_refs[local_layer].index_select(
            0, host_pages
        )
        expected_value = controller.mem_pool_host.v_data_refs[local_layer].index_select(
            0, host_pages
        )
        actual_key = kv_cache[0].index_select(0, device_pages).cpu()
        actual_value = kv_cache[1].index_select(0, device_pages).cpu()
        for name, actual, expected in (
            ("key", actual_key, expected_key),
            ("value", actual_value, expected_value),
        ):
            unequal = actual != expected
            if unequal.any():
                bad_pages = unequal.flatten(1).any(1).nonzero().flatten().tolist()
                raise RuntimeError(
                    f"indexed {name} transfer mismatch on logical pages "
                    f"{bad_pages[:16]} ({len(bad_pages)}/{len(mapping)})"
                )

    def _wait_for_stock_external_layer(
        self, pending: PendingHostLoad, layer: Any
    ) -> int:
        """Join the producer event before stock attention consumes a page."""
        local_layer = int(layer.layer_id) - int(
            getattr(pending.controller.mem_pool_device, "start_layer", 0)
        )
        prefetched = pending.prefetched_layers.get(local_layer)
        if prefetched is None:
            raise RuntimeError(
                "stock external attention reached a layer without an exact "
                f"prefetch event: {layer.layer_id}"
            )
        stream = torch.cuda.current_stream()
        if self._profile_barrier:
            arrive = torch.cuda.Event(enable_timing=True)
            arrive.record(stream)
            self._barrier_profiles.append(
                (arrive, prefetched.ready_event, int(layer.layer_id))
            )
        stream.wait_event(prefetched.ready_event)
        return local_layer

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: Any,
        forward_batch: Any,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        if self._active_batch is None:
            raise RuntimeError(
                "NTA decode ran without transformed request metadata; stock "
                "dispatch is disabled"
            )
        if self._stock_forward:
            self._stats["stock_attention_launches"] += 1
            pending = self._active_batch.pending_host_load
            if pending is None:
                self._stats["stock_resident_attention_launches"] += 1
            else:
                self._stats["stock_prefetched_external_attention_launches"] += 1
                local_layer = self._wait_for_stock_external_layer(pending, layer)
            self._stats["decode_launches"] += 1
            output = FlashInferAttnBackend.forward_decode(
                self,
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
            )
            if pending is not None:
                self._hicache.complete_layer(pending, local_layer)
                if local_layer + 1 == self._model_layer_count:
                    # A complete external prefetch uses the framework's stock
                    # consumer.  It therefore bypasses _run_attention(),
                    # which is normally the point that publishes the NTA
                    # engine report.  Publish after the final layer so the
                    # paired harness can audit the exact acquisition path.
                    self._publish_stats()
            return output
        wrapper = self.forward_metadata.decode_wrappers[self._get_wrapper_idx(layer)]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        if k is not None:
            if v is None:
                raise ValueError("decode K and V must be supplied together")
            if save_kv_cache:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )
        kv_cache = (
            self.token_to_kv_pool._get_key_buffer(layer.layer_id),
            self.token_to_kv_pool._get_value_buffer(layer.layer_id),
        )
        self._stats["decode_launches"] += 1
        return self._run_attention(
            wrapper,
            q,
            kv_cache,
            layer,
            causal=False,
            window_left=layer.sliding_window_size,
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: Any,
        forward_batch: Any,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        if self._active_batch is None:
            raise RuntimeError(
                "NTA prefill ran without transformed request metadata; stock "
                "dispatch is disabled"
            )
        if self._stock_forward:
            self._stats["stock_attention_launches"] += 1
            pending = self._active_batch.pending_host_load
            if pending is None:
                self._stats["stock_resident_attention_launches"] += 1
            else:
                self._stats["stock_prefetched_external_attention_launches"] += 1
                local_layer = self._wait_for_stock_external_layer(pending, layer)
            self._stats["prefill_launches"] += 1
            output = FlashInferAttnBackend.forward_extend(
                self,
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
            )
            if pending is not None:
                self._hicache.complete_layer(pending, local_layer)
                if local_layer + 1 == self._model_layer_count:
                    self._publish_stats()
            return output
        if self.forward_metadata.use_ragged:
            raise RuntimeError("NTA requires paged FlashInfer prefill")
        wrapper = self.forward_metadata.prefill_wrappers[self._get_wrapper_idx(layer)]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        if k is not None:
            if v is None:
                raise ValueError("prefill K and V must be supplied together")
            if save_kv_cache:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )
        kv_cache = (
            self.token_to_kv_pool._get_key_buffer(layer.layer_id),
            self.token_to_kv_pool._get_value_buffer(layer.layer_id),
        )
        causal = (
            not layer.is_cross_attention
            and layer.attn_type != AttentionType.ENCODER_ONLY
        )
        window_left = (
            layer.sliding_window_size
            if not (
                self.forward_metadata.multi_item_params
                and self.forward_metadata.multi_item_params.is_enabled()
            )
            else -1
        )
        self._stats["prefill_launches"] += 1
        return self._run_attention(
            wrapper,
            q,
            kv_cache,
            layer,
            causal=causal,
            window_left=window_left,
        )

    def _collect_transfer_profiles(self) -> None:
        pending: list[tuple[torch.cuda.Event, torch.cuda.Event, int, str]] = []
        for start, finish, transfer_bytes, kind in self._transfer_profiles:
            if not finish.query():
                pending.append((start, finish, transfer_bytes, kind))
                continue
            milliseconds = start.elapsed_time(finish)
            self._stats["profiled_transfer_batches"] = (
                self._stats.get("profiled_transfer_batches", 0) + 1
            )
            self._stats["profiled_transfer_bytes"] = (
                self._stats.get("profiled_transfer_bytes", 0) + transfer_bytes
            )
            self._stats["profiled_transfer_gpu_ms"] = (
                self._stats.get("profiled_transfer_gpu_ms", 0.0) + milliseconds
            )
            prefix = f"profiled_{kind}_transfer"
            self._stats[f"{prefix}_batches"] = (
                self._stats.get(f"{prefix}_batches", 0) + 1
            )
            self._stats[f"{prefix}_bytes"] = (
                self._stats.get(f"{prefix}_bytes", 0) + transfer_bytes
            )
            self._stats[f"{prefix}_gpu_ms"] = (
                self._stats.get(f"{prefix}_gpu_ms", 0.0) + milliseconds
            )
        self._transfer_profiles = pending
        milliseconds = float(self._stats.get("profiled_transfer_gpu_ms", 0.0))
        if milliseconds > 0:
            self._stats["profiled_transfer_gib_per_second"] = (
                float(self._stats["profiled_transfer_bytes"])
                / (1 << 30)
                / (milliseconds / 1_000.0)
            )
        for kind in ("pipeline", "demand"):
            prefix = f"profiled_{kind}_transfer"
            kind_milliseconds = float(self._stats.get(f"{prefix}_gpu_ms", 0.0))
            if kind_milliseconds > 0:
                self._stats[f"{prefix}_gib_per_second"] = (
                    float(self._stats[f"{prefix}_bytes"])
                    / (1 << 30)
                    / (kind_milliseconds / 1_000.0)
                )
        pending_operators: list[tuple[torch.cuda.Event, torch.cuda.Event, str]] = []
        for start, finish, kind in self._operator_profiles:
            if not finish.query():
                pending_operators.append((start, finish, kind))
                continue
            milliseconds = start.elapsed_time(finish)
            prefix = f"profiled_{kind}_operator"
            self._stats[f"{prefix}_layers"] = self._stats.get(f"{prefix}_layers", 0) + 1
            self._stats[f"{prefix}_gpu_ms"] = (
                self._stats.get(f"{prefix}_gpu_ms", 0.0) + milliseconds
            )
        self._operator_profiles = pending_operators

    def _collect_barrier_profiles(self) -> None:
        if not self._barrier_profiles:
            return
        # Barrier pairs reuse the per-layer ready events across batches.
        # Profiling mode synchronizes before draining so every pair is final
        # and no event is re-recorded while a measurement is outstanding; the
        # sync cost is confined to NTA_PROFILE_BARRIER=1 runs, whose
        # host-side throughput is never an execution result.
        torch.cuda.synchronize()
        for arrive, ready, layer_id in self._barrier_profiles:
            stall_ms = max(0.0, arrive.elapsed_time(ready))
            self._stats["profiled_barrier_waits"] = (
                self._stats.get("profiled_barrier_waits", 0) + 1
            )
            self._stats["profiled_barrier_stall_gpu_ms"] = (
                self._stats.get("profiled_barrier_stall_gpu_ms", 0.0) + stall_ms
            )
            if stall_ms > 0.01:
                self._stats["profiled_barrier_stalled_waits"] = (
                    self._stats.get("profiled_barrier_stalled_waits", 0) + 1
                )
            self._stats["profiled_barrier_max_stall_gpu_ms"] = max(
                float(self._stats.get("profiled_barrier_max_stall_gpu_ms", 0.0)),
                stall_ms,
            )
            self._barrier_stall_by_layer[layer_id] = (
                self._barrier_stall_by_layer.get(layer_id, 0.0) + stall_ms
            )
        self._barrier_profiles = []

    def _stats_report(self) -> dict[str, Any]:
        self._collect_transfer_profiles()
        self._collect_barrier_profiles()
        report = dict(self._stats)
        if self._barrier_stall_by_layer:
            report["profiled_barrier_stall_by_layer_ms"] = {
                str(layer): round(stall, 4)
                for layer, stall in sorted(self._barrier_stall_by_layer.items())
            }
        contracts = sorted(
            self._operator_contracts.values(),
            key=lambda contract: (int(contract.family), int(contract.form)),
        )
        report["operator_contracts"] = [
            {
                "schema_version": contract.schema_version,
                "runtime_abi_version": contract.runtime_abi_version,
                "family": contract.family.name.lower(),
                "form": contract.form.name.lower(),
                "capabilities": int(contract.capabilities),
                "instrumentation_flags": int(contract.instrumentation_flags),
                "identity_binding": contract.identity_binding.name.lower(),
                "demand_binding": contract.demand_binding.name.lower(),
                "access_proof": contract.access_proof.name.lower(),
                "granularity_bytes": contract.granularity_bytes,
                "tier_mask": contract.tier_mask,
                "source_fingerprint": contract.source_fingerprint,
            }
            for contract in contracts
        ]
        report["operator_plans"] = [
            {
                "schema_version": plan.schema_version,
                "runtime_abi_version": plan.runtime_abi_version,
                "family": plan.family.name.lower(),
                "supported_forms": plan.supported_forms,
                "coordinate_map": plan.coordinate_map.name.lower(),
                "partial_state": plan.partial_state.name.lower(),
                "reduction": plan.reduction.name.lower(),
                "flags": int(plan.flags),
                "source_fingerprint": plan.source_fingerprint,
                "plan_fingerprint": plan.plan_fingerprint,
            }
            for plan in sorted(
                self._operator_plans.values(),
                key=lambda candidate: (
                    int(candidate.family),
                    candidate.plan_fingerprint,
                ),
            )
        ]
        report["tier_descriptors"] = [
            {
                "source_kind": descriptor.source_kind.name.lower(),
                "capabilities": int(descriptor.capabilities),
                "device_state": descriptor.device_state,
                "estimated_latency_ns": descriptor.estimated_latency_ns,
                "estimated_bandwidth_bytes_per_second": descriptor.estimated_bandwidth_bytes_per_second,
                "active": descriptor.active,
                "flags": descriptor.flags,
            }
            for descriptor in (self._runtime.tier_descriptor(tier) for tier in TierKind)
        ]
        families = {contract.family for contract in contracts}
        report["verified_operator_pairs"] = sum(
            (family, OperatorForm.DIRECT) in self._operator_contracts
            and (family, OperatorForm.INCREMENTAL) in self._operator_contracts
            for family in families
        )
        report["verified_operator_plan_pairs"] = sum(
            (family, OperatorForm.DIRECT) in self._operator_plans
            and (family, OperatorForm.INCREMENTAL) in self._operator_plans
            and self._operator_plans[(family, OperatorForm.DIRECT)]
            == self._operator_plans[(family, OperatorForm.INCREMENTAL)]
            for family in families
        )
        report.update(self._hicache.admission_stats())
        report.update(FORWARD_PROFILE)
        report.update(PREFILL_GRAPH_COUNTERS)
        report["finished_unix_ns"] = time.time_ns()
        return report

    def _publish_stats(self) -> None:
        if self._stats_publisher is None:
            return
        self._stats_publisher.publish(self._stats_report())

    def _write_stats(self) -> None:
        if self._closed:
            return
        if self._stats_publisher is not None:
            self._stats_publisher.publish(self._stats_report(), wait=True)
        self._close_resources()
        self._closed = True
