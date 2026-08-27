"""SGLang 0.5.16 adapter for compiler-instrumented FlashInfer attention."""

from __future__ import annotations

import atexit
from collections import Counter
from dataclasses import replace
import logging
import math
import os
import pathlib
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)
from flashinfer.decode import get_batch_decode_jit_module
from flashinfer.prefill import get_batch_prefill_jit_module
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.memory_pool import KVWriteLoc

from nta_runtime.flashinfer import (
    FlashInferLayerEpoch,
    PREACQUIRED_LAUNCH_FLAGS,
    adopt_planned_flashinfer_state,
    attention_jit_args,
    direct_requirement,
    object_requirement,
    request_bound_attention_jit_args,
)
from nta_runtime.flashinfer_schedule import (
    Schedule,
    decode_schedule,
    paged_prefill_schedule,
    require_supported_version,
)
from nta_runtime.indexed_transfer import (
    ContiguousPairRun,
    IndexedTensorLane,
    StridedCopyGroup,
    analyze_index_pairs,
)
from nta_runtime.indexed_transfer_torch import (
    TensorIndexedMoverPlan,
    plan_indexed_tensor_mover,
)
from nta_runtime.hbm_registration import HbmDestinationSlice
from nta_runtime.adapters.base import ConsumerContract, EngineBatch
from nta_runtime.adapters.sglang import (
    SglangAcquisitionSpan,
    SglangAdapter,
    SglangExecutionConfig,
    forward_metadata,
    validate_sglang_attention_tier,
)
from nta_runtime.execution_core import ExecutionPlan, ExecutionSession, ExecutionTile
from nta_runtime.execution_topology import ExactWorkTopology, WorkDependencySpan
from nta_runtime.execution_protocol import ProtocolKind
from nta_runtime.execution_planner import (
    conservative_resume_counts,
    HostCostModel,
    HostExecutionPlan,
    LayerDeadlineServiceCurve,
    indexed_copy_blocks_per_group,
    plan_host_execution,
    prove_atomic_host_execution,
)
from nta_runtime.opportunity import OperatorArrival, TileArrival, append_json_line
from nta_runtime.requests import RequestBinding
from nta_runtime.tenant import tenant_budget_specs
from nta_runtime.engines.sglang_hicache import (
    LeaseOperationTransfer,
    LeaseWorkDependency,
    PendingHostLoad,
    SglangHiCacheBridge,
    lease_indexed_transfer_topology,
)
from nta_runtime.engines.sglang_transfer import (
    HostMoverLeasePlan,
    host_mover_service_model_from_environment,
)
from nta_runtime.engines.sglang_state import (
    _ActiveBatch,
    _DemandGraph,
    _DemandGraphKey,
    _FragmentLookahead,
    _LayerServiceProfile,
    _MoverProfile,
    _PagePair,
    _PlanAllocation,
    _PrefetchedLayer,
    _StatsPublisher,
    _demand_graph_key,
    _graph_wrapper_metadata,
)
from nta_runtime.runtime_resources import (
    RuntimeResourceConfig,
    ServingRuntimeResources,
)
from nta_runtime.tier import ServingTierConfig
from nta_runtime.runtime import (
    AcquireRequirement,
    DeviceWorkPlan,
    IndexedHostIndexBinding,
    IndexedHostObject,
    IndexedHostPlan,
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
    TierKind,
    copy_strided_host_runs_async,
    require_operator_pair,
)


_OBJECT_ID_BASE = 0x4E54410000000000
_LOOKAHEAD_VERSION = 1
_MAX_ABI_BYTES = (1 << 32) - 1
OBSERVATION_MARKER_REQUEST_PREFIX = "nta-observation-marker:"
# Per-forward timing, populated by the plugin's forward hooks. These samples
# are keyed by batch composition and measure the actual serving boundary seen
# by co-resident decode, rather than a transfer lifetime spanning forwards.
FORWARD_PROFILE: dict[str, float] = {}


def _flag_value(value: Any) -> int:
    """Serialize both IntFlag and Flag values without losing their mask."""

    raw = getattr(value, "value", value)
    return int(raw)


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


def _consumer_contract_for_stats(
    stats: Mapping[str, Any], *, engine_version: str
) -> ConsumerContract:
    """Classify the numerical consumer actually represented by one report.

    Backend selection is not execution.  Native launch counters take
    precedence when a process served a mixed workload; otherwise a complete
    exact prefetch is a framework-reference consumer, and a resident-only
    process remains projection-only for evidence purposes.
    """
    native_launches = int(stats.get("transformed_direct_launches", 0)) + int(
        stats.get("ticketed_incremental_launches", 0)
    )
    stock_external_launches = int(
        stats.get("stock_prefetched_external_attention_launches", 0)
    ) + int(stats.get("graph_external_batches", 0))
    if native_launches:
        return ConsumerContract.native_work_unit(
            engine="sglang",
            backend="nta_flashinfer",
            engine_version=engine_version,
        )
    if stock_external_launches:
        return ConsumerContract.framework_reference(
            engine="sglang",
            backend="nta_flashinfer",
            engine_version=engine_version,
        )
    return ConsumerContract.projection_only(
        engine="sglang",
        backend="nta_flashinfer",
        engine_version=engine_version,
    )


# Incremented by the plugin's PrefillCudaGraphRunner patches (same scheduler
# process); exported through _stats_report so artifacts attest whether the
# breakable prefill graphs actually served batches or only captured.
PREFILL_GRAPH_COUNTERS: dict[str, int] = {
    "prefill_graph_served_batches": 0,
    "prefill_graph_capture_batches": 0,
}
logger = logging.getLogger(__name__)


def _positive_environment(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


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


def _host_mover_environment() -> str:
    value = os.environ.get("NTA_EXECUTION_HOST_MOVER", "auto").strip().lower()
    if value not in {"auto", "sm", "copy_engine"}:
        raise RuntimeError("NTA_EXECUTION_HOST_MOVER must be auto, sm, or copy_engine")
    return value


def _gain_environment(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if value < 1.0:
        raise ValueError(f"{name} must be at least one")
    return value


def _require_exact_prefetch_layers(
    prefetched_layers: Mapping[int, Any],
    layer_count: int,
    *,
    consumer: str,
) -> int:
    """Validate full-model readiness and return the final local layer."""
    expected_layers = set(range(layer_count))
    actual_layers = set(prefetched_layers)
    if actual_layers != expected_layers:
        missing = sorted(expected_layers - actual_layers)
        unexpected = sorted(actual_layers - expected_layers)
        raise RuntimeError(
            f"{consumer} requires an exact full-model prefetch "
            f"(missing={missing}, unexpected={unexpected})"
        )
    if layer_count <= 0:
        raise RuntimeError(f"{consumer} requires at least one model layer")
    return layer_count - 1


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


def _pipeline_object_id(consumer_index: int, layer_count: int, local_layer: int) -> int:
    """Return the stable K-object identity for one proactive layer pair."""

    if (
        consumer_index < 0
        or layer_count <= 0
        or local_layer < 0
        or local_layer >= layer_count
    ):
        raise RuntimeError("HiCache proactive object identity is out of range")
    return (
        _OBJECT_ID_BASE + (1 << 44) + consumer_index * 2 * layer_count + 2 * local_layer
    )


def _dtype_tag(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.").replace("_", "")


def _plan_cache_signature(
    request_indices: tuple[int, ...],
    kv_tile_indices: tuple[int, ...],
    dependency_geometry: Any,
    request_slots: tuple[int, ...],
    key_bytes: int,
    value_bytes: int,
    prefetched_bytes: tuple[int, ...] | None,
) -> tuple[Any, ...]:
    """Return the immutable identity of a reusable device-side plan.

    Request generations are dynamic directory state. Demand launches bind the
    current generation on device, so generation reuse must not rebuild the
    structural work and dependency arrays.
    """
    return (
        request_indices,
        kv_tile_indices,
        dependency_geometry,
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


def _page_pairs_for_schedule(
    schedule: Schedule,
    *,
    indptr: Sequence[int],
    pages: Sequence[int],
    last_page: Sequence[int],
    page_size: int,
    source_by_device: Mapping[int, int],
) -> tuple[_PagePair, ...]:
    """Bind exact external pages to FlashInfer work without context-size copies.

    ``pages[indptr[r]:indptr[r + 1]]`` is the complete request context. Taking
    that slice once per CTA makes metadata construction O(work * context) even
    though each CTA consumes only one bounded KV chunk. Compute absolute slice
    bounds first so every page is visited only by the work item that consumes it.
    """

    pairs: list[_PagePair] = []
    for request_index, kv_tile in zip(
        schedule.request_indices, schedule.kv_tile_indices, strict=True
    ):
        if request_index < 0 or request_index + 1 >= len(indptr) or kv_tile < 0:
            raise RuntimeError("FlashInfer emitted an invalid KV tile coordinate")
        request_begin = int(indptr[request_index])
        request_end = int(indptr[request_index + 1])
        if not 0 <= request_begin <= request_end <= len(pages):
            raise RuntimeError("FlashInfer emitted an invalid KV page range")
        page_begin = request_begin
        page_end = request_end
        if schedule.kv_chunk_tokens > 0:
            token_count = max(0, request_end - request_begin - 1) * page_size + int(
                last_page[request_index]
            )
            token_begin = kv_tile * schedule.kv_chunk_tokens
            token_end = min(token_count, token_begin + schedule.kv_chunk_tokens)
            page_begin += token_begin // page_size
            page_end = request_begin + (token_end + page_size - 1) // page_size
        device_pages = tuple(
            int(page)
            for page in pages[page_begin:page_end]
            if int(page) in source_by_device
        )
        source_pages = tuple(source_by_device[page] for page in device_pages)
        pairs.append((source_pages, device_pages))
    return tuple(pairs)


def _resolve_request_acquisitions(
    acquisitions: Sequence[SglangAcquisitionSpan],
    operation_transfers: Mapping[int, LeaseOperationTransfer],
    *,
    lease_transfer_rows: int,
) -> tuple[SglangAcquisitionSpan, ...]:
    """Join framework request identity to one exact acquisition lease.

    The adapter publishes the unmerged operation assigned to each request and
    the lease publishes the same operation before SGLang coalesces its physical
    transfer. Joining by operation identity is unambiguous even when requests
    share a radix node, and it requires no numerical page-table scan.
    """

    if lease_transfer_rows <= 0:
        raise RuntimeError("SGLang acquisition lease contains no rows")
    normalized: dict[int, LeaseOperationTransfer] = {}
    for raw_operation_id, transfer in operation_transfers.items():
        operation_id = int(raw_operation_id)
        if (
            not isinstance(transfer, LeaseOperationTransfer)
            or operation_id != transfer.operation_id
            or operation_id < 0
        ):
            raise RuntimeError(
                "SGLang acquisition lease has invalid operation identity"
            )
        if operation_id in normalized:
            raise RuntimeError("SGLang acquisition lease repeats an operation")
        normalized[operation_id] = transfer
    if (
        sum(transfer.row_count for transfer in normalized.values())
        != lease_transfer_rows
    ):
        raise RuntimeError("SGLang acquisition operations do not cover the lease rows")
    resolved = tuple(acquisitions)
    if any(not isinstance(item, SglangAcquisitionSpan) for item in resolved):
        raise RuntimeError("SGLang forward carries untyped acquisition metadata")
    referenced_operations: set[int] = set()
    for acquisition in resolved:
        if not acquisition.is_external:
            continue
        transfer = normalized.get(acquisition.operation_id)
        if transfer is None:
            raise RuntimeError(
                "SGLang request references an operation outside its acquisition lease"
            )
        if (
            acquisition.node_id != transfer.node_id
            or acquisition.row_count != transfer.row_count
        ):
            raise RuntimeError(
                "SGLang request span disagrees with its acquisition operation"
            )
        referenced_operations.add(acquisition.operation_id)
    missing_operations = set(normalized) - referenced_operations
    if missing_operations:
        raise RuntimeError(
            "SGLang forward metadata omits acquisition ownership for "
            f"{len(missing_operations)} operation(s): "
            f"lease={tuple(sorted(normalized))}, "
            f"requests={tuple(item.operation_id for item in resolved)}"
        )
    if not referenced_operations:
        raise RuntimeError(
            "SGLang forward has no request bound to its acquisition lease"
        )
    return resolved


def _cpu_sequence_lengths(forward_batch: Any, request_count: int) -> tuple[int, ...]:
    """Read SGLang's existing CPU mirror without introducing a GPU sync."""

    values = getattr(forward_batch, "seq_lens_cpu", None)
    if values is None:
        raise RuntimeError("SGLang forward omitted its CPU sequence-length mirror")
    if isinstance(values, torch.Tensor):
        if values.is_cuda:
            raise RuntimeError(
                "SGLang sequence-length mirror unexpectedly resides on GPU"
            )
        values = values.tolist()
    lengths = tuple(int(value) for value in values)
    if len(lengths) != request_count or any(length <= 0 for length in lengths):
        raise RuntimeError("SGLang sequence lengths do not match the request batch")
    return lengths


def _request_batch_heterogeneity(
    bindings: Sequence[RequestBinding],
    sequence_lengths: Sequence[int],
    acquisitions: Sequence[SglangAcquisitionSpan],
) -> tuple[str, ...]:
    """Return the exact axes that differ inside one engine ForwardBatch.

    This is deliberately computed from the framework's CPU metadata and the
    typed acquisition spans, never from request-level benchmark intent.  It
    therefore proves that heterogeneous requests reached the *same* scheduler
    forward instead of merely overlapping somewhere in the client trace.
    """

    size = len(bindings)
    if len(sequence_lengths) != size or len(acquisitions) != size:
        raise RuntimeError("SGLang batch heterogeneity vectors are misaligned")
    if size < 2:
        return ()
    axes: list[str] = []
    if len({int(value) for value in sequence_lengths}) > 1:
        axes.append("sequence_length")
    if len({item.is_external for item in acquisitions}) > 1:
        axes.append("availability")
    external_rows = {
        int(item.row_count) for item in acquisitions if item.is_external
    }
    if len(external_rows) > 1:
        axes.append("external_rows")
    for name, values in (
        ("tenant", (binding.tenant_id for binding in bindings)),
        ("priority", (binding.priority for binding in bindings)),
        ("deadline", (binding.deadline_clock for binding in bindings)),
    ):
        if len({int(value) for value in values}) > 1:
            axes.append(name)
    return tuple(axes)


def _project_work_acquisitions(
    schedule: Schedule,
    acquisitions: Sequence[SglangAcquisitionSpan],
    sequence_lengths: Sequence[int],
) -> tuple[LeaseWorkDependency | None, ...]:
    """Project exact operation-local spans onto FlashInfer work units."""

    if len(acquisitions) != len(sequence_lengths):
        raise RuntimeError("SGLang acquisition and sequence vectors are misaligned")
    intersections: list[list[tuple[int, int]]] = [[] for _ in range(len(acquisitions))]
    work_dependencies: list[LeaseWorkDependency | None] = []
    for request_index, kv_tile in zip(
        schedule.request_indices, schedule.kv_tile_indices, strict=True
    ):
        if request_index < 0 or request_index >= len(acquisitions) or kv_tile < 0:
            raise RuntimeError("FlashInfer emitted an invalid acquisition coordinate")
        sequence_end = int(sequence_lengths[request_index])
        if schedule.kv_chunk_tokens > 0:
            tile_begin = int(kv_tile) * schedule.kv_chunk_tokens
            tile_end = min(sequence_end, tile_begin + schedule.kv_chunk_tokens)
        else:
            tile_begin = 0
            tile_end = sequence_end
        if tile_begin < 0 or tile_begin >= tile_end:
            raise RuntimeError("FlashInfer emitted an empty KV work tile")
        acquisition = acquisitions[request_index]
        overlap_begin = max(tile_begin, acquisition.logical_begin)
        overlap_end = min(tile_end, acquisition.logical_end)
        rows = max(0, overlap_end - overlap_begin) if acquisition.is_external else 0
        work_dependencies.append(
            LeaseWorkDependency(
                acquisition.operation_id,
                overlap_begin - acquisition.logical_begin,
                rows,
            )
            if rows
            else None
        )
        if rows:
            intersections[request_index].append((overlap_begin, overlap_end))

    for request_index, acquisition in enumerate(acquisitions):
        if not acquisition.is_external:
            continue
        # FlashInfer repeats one (request, KV-tile) coordinate for independent
        # query/head work. Every repeated work item needs the dependency, but
        # the transport owns one shared interval. Validate exact coverage over
        # the unique KV intervals while retaining the per-work fan-out above.
        spans = sorted(set(intersections[request_index]))
        cursor = acquisition.logical_begin
        for begin, end in spans:
            if begin != cursor:
                raise RuntimeError(
                    "FlashInfer CTA schedule duplicates or omits acquired request rows: "
                    f"request={request_index}, acquisition="
                    f"[{acquisition.logical_begin},{acquisition.logical_end}), "
                    f"cursor={cursor}, next=[{begin},{end}), "
                    f"chunk={schedule.kv_chunk_tokens}, spans={spans[:16]}"
                )
            cursor = end
        if cursor != acquisition.logical_end:
            raise RuntimeError(
                "FlashInfer CTA schedule does not cover the acquired request span: "
                f"request={request_index}, acquisition="
                f"[{acquisition.logical_begin},{acquisition.logical_end}), "
                f"covered_end={cursor}, chunk={schedule.kv_chunk_tokens}, "
                f"spans={spans[:16]}"
            )
    return tuple(work_dependencies)


def _capacity_constrained_transfer_dependencies(
    dependencies: Sequence[LeaseWorkDependency | None],
    *,
    maximum_groups: int,
) -> tuple[LeaseWorkDependency | None, ...]:
    """Choose an exact, bounded completion granularity for one schedule.

    Numerical work retains its original interval.  The transport may merge
    adjacent intervals belonging to the same acquisition operation so its two
    K/V objects per group fit the runtime directory.  Greedily bisecting the
    largest remaining byte interval minimizes the worst completion quantum
    without a byte threshold or workload-specific policy.
    """

    if maximum_groups <= 0:
        raise ValueError("typed transfer grouping requires positive capacity")
    unique_by_operation: dict[int, list[LeaseWorkDependency]] = {}
    for dependency in dependencies:
        if dependency is None:
            continue
        values = unique_by_operation.setdefault(dependency.operation_id, [])
        if dependency not in values:
            values.append(dependency)
    if not unique_by_operation:
        raise RuntimeError("typed transfer grouping has no external dependency")
    if len(unique_by_operation) > maximum_groups:
        raise RuntimeError(
            "runtime object capacity cannot represent every acquisition operation"
        )

    partitions: list[list[LeaseWorkDependency]] = []
    for operation_id in sorted(unique_by_operation):
        intervals = sorted(
            unique_by_operation[operation_id], key=lambda item: item.row_begin
        )
        cursor = 0
        for interval in intervals:
            if interval.row_begin != cursor:
                raise RuntimeError(
                    "typed work intervals do not exactly partition an operation"
                )
            cursor = interval.row_end
        partitions.append(intervals)

    exact_group_count = sum(len(partition) for partition in partitions)
    target_groups = min(maximum_groups, exact_group_count)
    while len(partitions) < target_groups:
        candidates = [
            (index, partition)
            for index, partition in enumerate(partitions)
            if len(partition) > 1
        ]
        if not candidates:  # pragma: no cover - exact count invariant
            raise RuntimeError("typed transfer partition cannot reach its capacity")
        partition_index, partition = max(
            candidates,
            key=lambda item: (
                sum(interval.row_count for interval in item[1]),
                len(item[1]),
                -item[1][0].operation_id,
                -item[1][0].row_begin,
            ),
        )
        total_rows = sum(interval.row_count for interval in partition)
        prefix_rows = 0
        split_index = 1
        best_distance = total_rows
        for candidate_index, interval in enumerate(partition[:-1], start=1):
            prefix_rows += interval.row_count
            distance = abs(total_rows - 2 * prefix_rows)
            if distance < best_distance:
                best_distance = distance
                split_index = candidate_index
        partitions[partition_index : partition_index + 1] = (
            partition[:split_index],
            partition[split_index:],
        )

    grouped: dict[LeaseWorkDependency, LeaseWorkDependency] = {}
    for partition in partitions:
        first = partition[0]
        cursor = first.row_begin
        for interval in partition:
            if (
                interval.operation_id != first.operation_id
                or interval.row_begin != cursor
            ):
                raise RuntimeError("typed transfer group is not contiguous")
            cursor = interval.row_end
        group = LeaseWorkDependency(
            first.operation_id,
            first.row_begin,
            cursor - first.row_begin,
        )
        for interval in partition:
            grouped[interval] = group
    return tuple(
        None if dependency is None else grouped[dependency]
        for dependency in dependencies
    )


class _NvmeSlotLifetime:
    """Single-use stream proof for immutable mapping-view publication."""

    def __init__(self, consumer_event: Any) -> None:
        self._consumer_event = consumer_event
        self._destinations: dict[int, int] = {}
        self._consumer_proof = False

    def previous(self, slot: int) -> int | None:
        return self._destinations.get(slot)

    def prior_consumer_event(self, slot: int) -> Any | None:
        if slot not in self._destinations:
            return None
        if not self._consumer_proof:
            raise RuntimeError(
                f"NVMe slot {slot} replacement has no prior-consumer event"
            )
        return self._consumer_event

    def commit(self, publications: tuple[tuple[int, int], ...]) -> None:
        if not publications:
            raise ValueError("NVMe slot publication batch is empty")
        slots = tuple(slot for slot, _ in publications)
        if len(set(slots)) != len(slots) or any(
            address <= 0 for _, address in publications
        ):
            raise ValueError("NVMe slot publications must be unique and non-null")
        replaced = any(slot in self._destinations for slot in slots)
        if replaced and not self._consumer_proof:
            raise RuntimeError("NVMe replacement consumed no prior-consumer proof")
        self._destinations.update(publications)
        if replaced:
            self._consumer_proof = False

    def record_consumer(self, stream: Any) -> None:
        if not self._destinations:
            raise RuntimeError("cannot record an NVMe consumer before publication")
        self._consumer_event.record(stream)
        self._consumer_proof = True


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
        # Register cleanup state before any operation that may fail.  The
        # backend constructor opens the selected tier before creating several
        # CUDA streams and graph-side objects; a later configuration error
        # must still release that owner when Python destroys the partial
        # object.
        self._resources: ServingRuntimeResources | None = None
        self._resources_closed = True
        self._closed = True
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
        self._install_instrumented_wrappers(model_runner, skip_prefill)

        request_capacity = int(model_runner.req_to_token_pool.req_to_token.shape[0])
        default_tickets = max(4096, request_capacity * 8)
        self._work_ticket_capacity = _positive_environment(
            "NTA_RUNTIME_MAX_WORK_TICKETS", default_tickets
        )
        self._max_dependencies_per_work_ticket = _positive_environment(
            "NTA_RUNTIME_MAX_DEPENDENCIES_PER_WORK_TICKET", 16
        )
        self._object_capacity = 2 * self._work_ticket_capacity
        self._tenant_capacity = _positive_environment(
            "NTA_TENANT_CAPACITY", request_capacity
        )
        tenant_specs = tenant_budget_specs()
        for tenant_id, _ in tenant_specs:
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
            validate_sglang_attention_tier({"NTA_SERVING_TIER": tier_config.tier.value})
            resources = ServingRuntimeResources.open(
                tier_config=tier_config,
                runtime_config=RuntimeResourceConfig.with_environment_staging_limit(
                    request_capacity=request_capacity,
                    object_capacity=self._object_capacity,
                    intent_capacity=self._object_capacity,
                    work_ticket_capacity=self._work_ticket_capacity,
                    max_dependencies_per_work_ticket=(
                        self._max_dependencies_per_work_ticket
                    ),
                    device_ordinal=torch.cuda.current_device(),
                    tenant_capacity=self._tenant_capacity,
                ),
            )
            if resources.tier.is_host_staged and not self._hicache_enabled:
                raise RuntimeError(
                    "host_staged requires SGLang hierarchical-cache host payloads"
                )
            if resources.tier.is_hbm and self._hicache_enabled:
                raise RuntimeError(
                    "an HBM profile cannot label hierarchical-cache host transfers; "
                    "select host_staged or disable hierarchical cache"
                )
            if resources.tier.is_physical and not self._hicache_enabled:
                raise RuntimeError(
                    "a physical serving tier requires SGLang hierarchical cache metadata"
                )
            if (
                resources.tier.is_physical
                and getattr(model_runner.server_args, "hicache_storage_backend", None)
                != "dynamic"
            ):
                raise RuntimeError(
                    "a physical serving tier requires SGLang's dynamic "
                    "NtaSglangStorage stable-key connector"
                )
            if resources.tier.is_physical:
                catalog = resources.tier.catalog
                if catalog is None:
                    raise RuntimeError("a physical serving tier has no storage catalog")
                from nta_runtime.connectors.sglang_storage import (
                    validate_sglang_storage_backend,
                )

                validate_sglang_storage_backend(
                    model_runner.server_args,
                    expected_namespace=catalog.namespace,
                )
                if os.environ.get("NTA_EXECUTION_ALLOW_LOAD_FALLBACK", "0") == "1":
                    raise RuntimeError(
                        "physical metadata-only storage cannot fall back to a host "
                        "payload transfer"
                    )
        except (OSError, ValueError, RuntimeError) as error:
            if resources is not None:
                resources.close()
            raise RuntimeError(
                f"invalid NTA serving tier configuration: {error}"
            ) from error
        if resources is None:  # pragma: no cover - guarded by open()
            raise RuntimeError("serving runtime resources were not initialized")
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
        self._copy_stream = torch.cuda.Stream(priority=mover_priority)
        self._progress_stream = torch.cuda.Stream(priority=mover_priority)
        # The host-staged directory is reused across forwards.  After the
        # final attention layer records this event, the next registration can
        # wait on it in the CUDA stream instead of downloading every object
        # state to the CPU.  The conservative state probe remains the fallback
        # when a forward did not reach its normal completion edge.
        self._indexed_object_quiescence_event = torch.cuda.Event()
        self._indexed_object_quiescence_recorded = False
        # NVMe object slots have the same ownership hazard as indexed host
        # objects, but their old destination also owns a VFIO/IOMMU mapping.
        # Keep replacement stream-ordered so a changing physical page set does
        # not force the scheduler through a device-wide synchronization.
        self._nvme_slots = _NvmeSlotLifetime(torch.cuda.Event())
        self._nvme_regions: dict[tuple[int, str], Any] = {}
        self._host_cost_model = HostCostModel.from_environment()
        default_incremental_probes = (
            0 if self._host_cost_model.incremental_setup_ns is not None else 2
        )
        self._incremental_calibration_probes_remaining = _nonnegative_environment(
            "NTA_EXECUTION_CALIBRATION_PROBES", default_incremental_probes
        )
        self._incremental_initialization_probes_remaining = min(
            1, self._incremental_calibration_probes_remaining
        )
        self._incremental_setup_samples = 0
        self._host_mover = _host_mover_environment()
        self._host_mover_service_model = host_mover_service_model_from_environment()
        self._host_mover_calibration_samples = min(
            32,
            _positive_environment(
                "NTA_EXECUTION_HOST_MOVER_CALIBRATION_SAMPLES", 3
            ),
        )
        self._layer_service_minimum_samples = min(
            32,
            _positive_environment("NTA_EXECUTION_LAYER_SERVICE_MIN_SAMPLES", 4),
        )
        self._layer_service_maximum_samples = min(
            128,
            _positive_environment("NTA_EXECUTION_LAYER_SERVICE_MAX_SAMPLES", 32),
        )
        if (
            self._layer_service_maximum_samples
            < self._layer_service_minimum_samples
        ):
            raise RuntimeError(
                "NTA layer-service maximum samples are below its minimum"
            )
        self._layer_service_curves: dict[
            tuple[str, int, int], LayerDeadlineServiceCurve
        ] = {}
        self._copy_engine_max_operations = min(
            1 << 16,
            _positive_environment("NTA_EXECUTION_COPY_ENGINE_MAX_OPERATIONS", 4096),
        )
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
        self._admission_lead_layers = _positive_environment(
            "NTA_EXECUTION_ADMISSION_LEAD_LAYERS", 4
        )
        self._overlap_enabled = self._execution_config.protocol.allow_overlap
        self._frontier_enabled = (
            self._tier_service.is_host_staged and self._overlap_enabled
        )
        self._fragment_enabled = (
            self._tier_service.is_host_staged and self._overlap_enabled
        )
        self._grouping = self._execution_config.grouping
        self._prefetch_ready_events: tuple[tuple[torch.cuda.Event, ...], ...] = ()
        self._bulk_events: tuple[torch.cuda.Event, ...] = ()
        layer_count = getattr(model_runner.model_config, "num_hidden_layers", None)
        if layer_count is None:
            layer_count = getattr(
                model_runner.model_config.hf_config, "num_hidden_layers"
            )
        self._global_model_layer_count = int(layer_count)
        self._model_start_layer = int(getattr(self.token_to_kv_pool, "start_layer", 0))
        self._model_end_layer = int(
            getattr(
                self.token_to_kv_pool,
                "end_layer",
                self._global_model_layer_count,
            )
        )
        if not (
            0
            <= self._model_start_layer
            < self._model_end_layer
            <= self._global_model_layer_count
        ):
            raise RuntimeError("SGLang KV pool exposes an invalid local layer range")
        self._model_layer_count = self._model_end_layer - self._model_start_layer
        self._cuda_graph_mode = False
        self._stock_forward = False
        self._stock_wrapper_for_typed: dict[int, Any] = {}
        self._execution_epoch = 0
        self._current_engine_batch: EngineBatch | None = None
        self._active_batch: _ActiveBatch | None = None
        self._plans: dict[tuple[int, int], _PlanAllocation] = {}
        self._phase_programs: dict[str, JitPhaseProgram] = {}
        self._transport_program: JitPhaseProgram | None = None
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
            self._tier_service.is_host_staged
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
            "execution_protocol_status": "projection_only",
            "execution_demand_semantics": "exact",
            "execution_plan_scope": "attention_launch",
            "python_availability_state_machine": "verify_only",
            "consumer_contract": ConsumerContract.projection_only(
                engine="sglang",
                backend="nta_flashinfer",
                engine_version=os.environ.get("NTA_SGLANG_VERSION", "0.5.16"),
            ).as_dict(),
            "revision": os.environ.get("NTA_REVISION", "unknown"),
            "pid": os.getpid(),
            "host_execution_selection": "measured_direct_or_incremental",
            "host_direct_batches": 0,
            "host_incremental_batches": 0,
            "host_typed_mixed_batches": 0,
            "host_mixed_direct_batches": 0,
            "host_bound_after_full_publication_batches": 0,
            "host_selection_predicted_atomic_ns": 0,
            "host_selection_predicted_selected_ns": 0,
            "host_selection_bound_fastpath_batches": 0,
            "overlap_enabled": self._overlap_enabled,
            "frontier_enabled": self._frontier_enabled,
            "fragment_enabled": self._fragment_enabled,
            "sglang_mixed_chunk_enabled": bool(
                model_runner.server_args.enable_mixed_chunk
            ),
            "max_host_rounds": self._host_cost_model.max_rounds,
            "minimum_predicted_gain": self._host_cost_model.minimum_predicted_gain,
            "incremental_setup_ns": self._host_cost_model.incremental_setup_ns,
            "incremental_setup_calibrated": (
                self._host_cost_model.incremental_setup_ns is not None
            ),
            "incremental_setup_samples": 0,
            "incremental_initialization_samples": 0,
            "incremental_initialization_setup_ns": 0,
            "incremental_calibration_probes_remaining": (
                self._incremental_calibration_probes_remaining
            ),
            "cost_model_bandwidth_bps": (
                self._host_cost_model.bandwidth_bytes_per_second
            ),
            "cost_model_transfer_samples": 0,
            "host_mover": self._host_mover,
            "copy_engine_max_operations": self._copy_engine_max_operations,
            "host_mover_copy_calibrated": (
                self._host_mover_service_model.copy_calibrated
            ),
            "host_mover_calibration_samples_per_engine": (
                self._host_mover_calibration_samples
            ),
            "host_mover_sm_samples": self._host_mover_service_model.sm_samples,
            "host_mover_copy_samples": self._host_mover_service_model.copy_samples,
            "host_mover_calibration_probe_sm_batches": 0,
            "host_mover_calibration_probe_copy_batches": 0,
            "host_mover_profiled_sm_bytes": 0,
            "host_mover_profiled_sm_gpu_ms": 0.0,
            "host_mover_profiled_copy_bytes": 0,
            "host_mover_profiled_copy_gpu_ms": 0.0,
            "host_mover_sm_bandwidth_bps": (
                self._host_mover_service_model.sm_bandwidth_bytes_per_second
            ),
            "host_mover_copy_bandwidth_bps": (
                self._host_mover_service_model.copy_bandwidth_bytes_per_second
            ),
            "host_mover_copy_operation_ns": (
                self._host_mover_service_model.copy_operation_ns
            ),
            "host_mover_hybrid_join_ns": (
                self._host_mover_service_model.hybrid_join_ns
            ),
            "host_mover_minimum_gain": (self._host_mover_service_model.minimum_gain),
            "host_mover_predicted_sm_ns": 0,
            "host_mover_predicted_selected_ns": 0,
            "host_mover_overlap_compute_ns": 0,
            "layer_service_minimum_samples": self._layer_service_minimum_samples,
            "layer_service_maximum_samples": self._layer_service_maximum_samples,
            "layer_service_profiled_intervals": 0,
            "layer_service_calibrated_shapes": 0,
            "layer_service_conservative_ns": 0,
            "layer_service_plan_key_missing_batches": 0,
            "layer_service_plan_curve_missing_batches": 0,
            "layer_service_plan_curve_uncalibrated_batches": 0,
            "layer_service_plan_curve_calibrated_batches": 0,
            "layer_service_retirement_commits": 0,
            "host_mover_uncalibrated_copy_engine_batches": 0,
            "host_mover_insufficient_gain_batches": 0,
            "host_mover_service_cost_batches": 0,
            "host_mover_sm_batches": 0,
            "host_mover_copy_engine_batches": 0,
            "host_mover_hybrid_batches": 0,
            "copy_engine_waves": 0,
            "copy_engine_submissions": 0,
            "copy_engine_operations": 0,
            "copy_engine_issue_cpu_ns": 0,
            "hybrid_parallel_waves": 0,
            "copy_engine_selected_runs": 0,
            "copy_engine_selected_rows": 0,
            "copy_engine_bytes": 0,
            "copy_engine_layout_cpu_ns": 0,
            "sm_mover_bytes": 0,
            "sm_mover_rows": 0,
            "indexed_copy_target_bytes": self._indexed_copy_target_bytes,
            "indexed_copy_max_blocks": self._indexed_copy_max_blocks,
            "frontier_layers_per_wave": self._frontier_layers_per_wave,
            "admission_lead_layers": self._admission_lead_layers,
            "batches": 0,
            "decode_launches": 0,
            "prefill_launches": 0,
            "cta_work_items": 0,
            "plan_uploads": 0,
            "request_rebindings": 0,
            "request_cancellations": 0,
            "request_retirements": 0,
            "external_launches": 0,
            "native_external_attention_launches": 0,
            "resident_reference_batches": 0,
            "hicache_external_batches": 0,
            "hicache_fallback_batches": 0,
            "indexed_host_objects": 0,
            "request_acquisition_groups": 0,
            "tile_acquisition_groups": 0,
            "indexed_host_bytes": 0,
            "prefetched_layers": 0,
            "prefetched_host_bytes": 0,
            "tier_selected_leases": 0,
            "tier_selected_rows": 0,
            "tier_selected_bytes": 0,
            "tier_candidate_bytes": 0,
            "lookahead_acquisition_layers": 0,
            "lookahead_acquisition_objects": 0,
            "lookahead_bound_launches": 0,
            "arriving_prefetch_layers": 0,
            "arriving_prefetch_launches": 0,
            "typed_acquisition_batches": 0,
            "typed_acquisition_rows": 0,
            "typed_acquisition_work_items": 0,
            "demand_host_layers": 0,
            "incremental_host_layers": 0,
            "request_overlap_layers": 0,
            "mixed_dependency_layers": 0,
            "mixed_forward_batches": 0,
            "mixed_forward_requests": 0,
            "mixed_scheduled_requests": 0,
            "mixed_direct_work_items": 0,
            "mixed_external_work_items": 0,
            "multi_request_engine_batches": 0,
            "heterogeneous_engine_batches": 0,
            "multi_axis_heterogeneous_batches": 0,
            "sequence_length_heterogeneous_batches": 0,
            "availability_heterogeneous_batches": 0,
            "external_rows_heterogeneous_batches": 0,
            "tenant_heterogeneous_batches": 0,
            "priority_heterogeneous_batches": 0,
            "deadline_heterogeneous_batches": 0,
            "direct_host_layers": 0,
            "transformed_direct_launches": 0,
            "typed_bulk_attention_launches": 0,
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
            "transport_program_loaded": False,
            "work_topology_builds": 0,
            "work_topology_cache_hits": 0,
            "work_topology_cpu_ns": 0,
            "work_topology_cache_cpu_ns": 0,
            "work_topology_items": 0,
            "semantic_plan_builds": 0,
            "semantic_plan_cpu_ns": 0,
            "semantic_verifier_sessions": 0,
            "semantic_dense_tiles": 0,
            "started_unix_ns": time.time_ns(),
        }
        self._stats.update(self._tier_service.stats())
        self._stats.update(
            {
                "nvme_progress_rounds": 0,
                "nvme_bytes": 0,
                "nvme_epochs": 0,
                "tier_external_layers": 0,
                "tier_host_proxy_bytes": 0,
                "indexed_object_quiesced_registrations": 0,
                "indexed_object_lifetime_guard_fallbacks": 0,
                "nvme_view_publications": 0,
                "nvme_same_destination_installs": 0,
                "nvme_destination_rebinds": 0,
                "nvme_fresh_slot_installs": 0,
                "nvme_object_quiesced_replacements": 0,
                "nvme_region_prepare_ns": 0,
                "nvme_region_count": 0,
                "nvme_region_bytes": 0,
                "nvme_destination_slice_count": 0,
                "nvme_destination_slice_bytes": 0,
                "nvme_shared_region_slices": 0,
            }
        )
        if self._tier_service.is_nvme:
            self._prepare_nvme_regions()
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
            # SGLang terminates model workers with a signal on some otherwise
            # clean Engine shutdown paths, so Python atexit is not a reliable
            # setup-evidence boundary.  Persist a clearly typed setup snapshot
            # synchronously; the first served-batch/final report atomically
            # replaces it with numerical-consumer evidence.
            setup_report = dict(self._stats)
            setup_report.update(self._tier_service.stats())
            setup_report.update(
                {
                    "stats_lifecycle": "setup",
                    "stats_process_id": os.getpid(),
                    "snapshot_unix_ns": time.time_ns(),
                }
            )
            self._stats_publisher.publish(setup_report, wait=True)
        self._profile_cpu = os.environ.get("NTA_PROFILE_CPU") == "1"
        self._profile_transfer = os.environ.get("NTA_PROFILE_TRANSFER") == "1"
        self._profile_index_layout = os.environ.get("NTA_PROFILE_INDEX_LAYOUT") == "1"
        self._profile_index_min_bytes = _positive_environment(
            "NTA_PROFILE_INDEX_LAYOUT_MIN_BYTES", 64 * 1024
        )
        self._stats.update(
            {
                "indexed_layout_profile_enabled": self._profile_index_layout,
                "indexed_layout_min_copy_bytes": self._profile_index_min_bytes,
                "indexed_layout_profiles": 0,
                "indexed_layout_rows": 0,
                "indexed_layout_runs": 0,
                "indexed_layout_eligible_rows": 0,
                "indexed_layout_candidate_bytes": 0,
                "indexed_layout_profile_cpu_ns": 0,
                "indexed_layout_maximum_run_rows": 0,
            }
        )
        self._profile_gpu = os.environ.get("NTA_PROFILE_GPU") == "1"
        # Barrier profiling measures how long the compute stream stalls at each
        # proactive layer-readiness wait. It is the opportunity signal the
        # RQ2/2A characterization consumes: stall > 0 means arrival, not
        # compute, bounded that layer.
        self._profile_barrier = os.environ.get("NTA_PROFILE_BARRIER") == "1"
        self._transfer_profiles: list[
            tuple[torch.cuda.Event, torch.cuda.Event, int, str]
        ] = []
        self._mover_profiles: list[_MoverProfile] = []
        self._layer_service_profiles: list[_LayerServiceProfile] = []
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
        if self._tier_service.is_host_staged:
            # Capture ownership immediately, but delay the execution form until
            # the exact FlashInfer schedule is available in forward metadata.
            self._hicache.set_acquire_callback(self._hold_host_load)
            if (
                self._execution_config.protocol.kind is not ProtocolKind.CONVENTIONAL
                and self._host_cost_model.max_rounds > 1
                and (
                    self._host_cost_model.incremental_setup_ns is not None
                    or self._incremental_calibration_probes_remaining > 0
                )
            ):
                self._prepare_host_execution_modules()
        atexit.register(self._write_stats)

    def _hold_host_load(self, pending: PendingHostLoad) -> None:
        """Capture a HiCache lease before selecting its execution form.

        Ownership is captured by the hook before a batch can be admitted.
        Conventional execution publishes the complete producer immediately.
        Late-bound and partial execution defer transport until FlashInfer has
        exposed exact request/work dependencies; admission explicitly releases
        an unpublished lease for that binding step, so there is no producer /
        metadata cycle.
        """

        if pending.controller.mem_pool_device is not self.token_to_kv_pool:
            raise RuntimeError("HiCache lease belongs to a different device pool")
        layer_count = int(pending.controller.layer_num)
        if layer_count != self._model_layer_count:
            raise RuntimeError("HiCache load and model layer counts disagree")
        self._account_tier_selection(pending)
        initial_layers = (
            layer_count
            if self._execution_config.protocol.kind is ProtocolKind.CONVENTIONAL
            else 0
        )
        if initial_layers:
            self._prepare_host_pipeline(
                pending,
                first_local_layer=0,
                last_local_layer=initial_layers,
            )
        self._stats["initial_acquisition_batches"] = (
            self._stats.get("initial_acquisition_batches", 0) + 1
        )
        self._stats["initial_acquisition_layers"] = (
            self._stats.get("initial_acquisition_layers", 0) + initial_layers
        )
        self._stats["initial_typed_gap_layers"] = self._stats.get(
            "initial_typed_gap_layers", 0
        ) + (layer_count - initial_layers)
        if initial_layers == 0:
            self._stats["schedule_bound_acquisition_batches"] = (
                self._stats.get("schedule_bound_acquisition_batches", 0) + 1
            )

    def _account_tier_selection(self, pending: PendingHostLoad) -> None:
        """Count unique logical tier demand once per ownership lease."""

        if pending.selection_accounted:
            return
        row_count = int(pending.device_indices.numel())
        if row_count <= 0:
            raise RuntimeError("SGLang acquisition lease contains no selected rows")
        controller = pending.controller
        host_keys = tuple(controller.mem_pool_host.k_data_refs)
        host_values = tuple(controller.mem_pool_host.v_data_refs)
        layer_count = int(controller.layer_num)
        if (
            len(host_keys) != layer_count
            or len(host_values) != layer_count
            or not host_keys
        ):
            raise RuntimeError("SGLang acquisition lease has incomplete layer geometry")
        layer_bytes = tuple(
            row_count
            * (
                int(key[0].numel()) * key.element_size()
                + int(value[0].numel()) * value.element_size()
            )
            for key, value in zip(host_keys, host_values, strict=True)
        )
        if pending.layer_bytes and pending.layer_bytes != layer_bytes:
            raise RuntimeError("SGLang acquisition byte geometry changed after capture")
        pending.layer_bytes = layer_bytes
        pending.selection_accounted = True
        self._stats["tier_selected_leases"] += 1
        self._stats["tier_selected_rows"] += row_count
        self._stats["tier_selected_bytes"] += sum(layer_bytes)
        # SGLang HiCache load-back is an exact-dense source range. Sparse
        # demand providers may later publish a larger candidate set, but this
        # framework path neither approximates nor drops any candidate row.
        self._stats["tier_candidate_bytes"] += sum(layer_bytes)

    def _configure_tenant_budgets(self, specs: tuple[tuple[int, int], ...]) -> None:
        for tenant_id, max_bytes in specs:
            if tenant_id >= self._tenant_capacity:
                raise RuntimeError(
                    f"tenant {tenant_id} exceeds NTA_TENANT_CAPACITY="
                    f"{self._tenant_capacity}"
                )
            self._runtime.set_tenant_budget(tenant_id, max_bytes)

    def cancel_requests(self, request_id_prefix: str, *, all: bool = False) -> int:
        cancelled = self._request_adapter.cancel_matching(request_id_prefix, all=all)
        self._stats["request_cancellations"] += cancelled
        return cancelled

    def retire_request(self, request_id: str) -> bool:
        """Retire one SGLang-confirmed request generation.

        Completion and abort are separate framework lifecycle edges.  Both
        invalidate outstanding device work, but accounting them separately
        makes a missing completion hook visible in long-running serving tests.
        """
        # SGLang invokes this lifecycle edge only after the forward result has
        # reached the CPU.  CUDA timing events for that request's layer arrivals
        # are therefore eligible for a query-only commit here.  Deferring this
        # until the next transfer plan made behavior-matched warmup samples
        # invisible to the measured plan (and left shutdown as the first point
        # that committed them).
        committed_before = int(self._stats["layer_service_profiled_intervals"])
        self._collect_layer_service_profiles()
        self._stats["layer_service_retirement_commits"] += (
            int(self._stats["layer_service_profiled_intervals"]) - committed_before
        )
        retired = self._request_adapter.retire_request(request_id)
        if retired:
            self._stats["request_retirements"] += 1
        return retired

    def __del__(self) -> None:
        """Release resources if construction failed before normal close()."""
        try:
            if getattr(self, "_resources_closed", True):
                return
            hicache = getattr(self, "_hicache", None)
            if hicache is not None:
                try:
                    hicache.close()
                except BaseException:
                    pass
            resources = getattr(self, "_resources", None)
            if resources is not None:
                try:
                    resources.close()
                except BaseException:
                    pass
            self._resources_closed = True
        except BaseException:
            # Destructors cannot safely report or propagate errors. Explicit
            # close() remains strict; this path only prevents a partial
            # constructor from retaining a native transport indefinitely.
            pass

    def close(self) -> None:
        """Flush observations and release CUDA/native tier resources."""
        if self._closed:
            return
        self._collect_transfer_profiles()
        self._collect_barrier_profiles()
        self._write_stats(strict=True)

    def _close_resources(self) -> None:
        if self._resources_closed:
            return
        errors: list[BaseException] = []
        # Plans, graphs, and native runtime buffers all contain device pointers.
        # Quiesce every stream before releasing them, including direct NVMe HBM
        # destinations and CXL-backed mappings.  Teardown continues after an
        # individual owner fails so one bad lease cannot leak later owners.
        try:
            torch.cuda.synchronize()
        except BaseException as error:
            errors.append(error)
        try:
            self._hicache.close()
        except BaseException as error:
            errors.append(error)
        self._demand_graphs.clear()
        self._demand_graph_warmups.clear()
        self._demand_sync_events.clear()
        for allocation in tuple(self._plans.values()):
            try:
                allocation.plan.close()
            except BaseException as error:
                errors.append(error)
        self._plans.clear()
        for program in tuple(self._phase_programs.values()):
            try:
                program.close()
            except BaseException as error:
                errors.append(error)
        self._phase_programs.clear()
        if self._transport_program is not None:
            try:
                self._transport_program.close()
            except BaseException as error:
                errors.append(error)
            self._transport_program = None
        # _operator_programs is an index into _phase_programs for pair
        # validation, not a second owner of the native handles.
        self._operator_programs.clear()
        try:
            self._resources.close()
        except BaseException as error:
            errors.append(error)
        self._resources_closed = True
        if errors:
            raise RuntimeError(
                f"NTA resource teardown encountered {len(errors)} error(s)"
            ) from errors[0]

    def _install_instrumented_wrappers(
        self, model_runner: Any, skip_prefill: bool
    ) -> None:
        del skip_prefill
        self._instrumented_q_dtype = model_runner.dtype
        self._instrumented_kv_dtype = model_runner.kv_cache_dtype
        self._instrumented_head_dim = int(model_runner.model_config.head_dim)
        self._instrumented_signature = (
            f"h{self._instrumented_head_dim}_"
            f"{_dtype_tag(self._instrumented_q_dtype)}_"
            f"{_dtype_tag(self._instrumented_kv_dtype)}"
        )
        self._wrapper_modules: dict[int, str] = {}
        self._loaded_jit_modules: dict[str, Any] = {}
        self._typed_decode_wrappers: dict[str, list[Any]] = {}
        self._typed_prefill_wrappers: dict[str, tuple[list[Any], list[Any]]] = {}
        # Resident and complete-prefetch batches must not pay JIT cost for an
        # acquisition consumer they never execute.  Typed modules are built by
        # _select_wrappers() or the transport phase on first actual use.
        self._select_stock_wrappers()

    def _prepare_host_execution_modules(self) -> None:
        """Build and validate typed modules before accepting requests.

        FlashInfer compilation is deployment setup, not a request-period
        scheduling action. A calibrated selector or an explicit calibration
        probe can enter the incremental form, so both numerical families and
        the transport phase must be resident before the engine becomes ready.
        """

        started = time.perf_counter_ns()
        loaded: set[str] = set()
        try:
            self._select_wrappers(True)
            wrapper_groups: list[tuple[Any, ...]] = [tuple(self.decode_wrappers)]
            if not self.skip_prefill:
                wrapper_groups.extend(
                    (
                        tuple(self.prefill_wrappers_paged),
                        tuple(self.prefill_wrappers_verify),
                    )
                )
            for wrappers in wrapper_groups:
                if not wrappers:
                    continue
                wrapper = wrappers[0]
                module_name = self._wrapper_modules[id(wrapper)]
                if module_name in loaded:
                    continue
                self._phase_program(wrapper).warmup_indexed_host_validation(
                    self._runtime, torch.cuda.current_stream()
                )
                loaded.add(module_name)
            self._transport_phase_program()
            torch.cuda.current_stream().synchronize()
        finally:
            self._select_stock_wrappers()
        self._stats["typed_startup_precompiled"] = True
        self._stats["typed_startup_modules"] = len(loaded)
        self._stats["typed_startup_ns"] = time.perf_counter_ns() - started

    def _jit_arguments(self, name: str, form: str) -> list[Any]:
        if form not in {"request_bound", "demand_acquire"}:
            raise ValueError(f"unknown typed attention form {form!r}")
        jit_builder = (
            request_bound_attention_jit_args
            if form == "request_bound"
            else attention_jit_args
        )
        return jit_builder(
            name,
            dtype_q=self._instrumented_q_dtype,
            dtype_kv=self._instrumented_kv_dtype,
            dtype_o=self._instrumented_q_dtype,
            idtype=torch.int32,
            head_dim_qk=self._instrumented_head_dim,
            head_dim_vo=self._instrumented_head_dim,
        )

    def _load_prebuilt_attention_module(
        self,
        name: str,
        jit_args: list[Any],
        *,
        decode: bool,
        use_tensor_cores: bool = False,
    ) -> Any | None:
        """Load one content-addressed FlashInfer artifact without rebuilding it."""

        cached = self._loaded_jit_modules.get(name)
        if cached is not None:
            return cached
        workspace_value = os.environ.get("FLASHINFER_WORKSPACE_BASE")
        if not workspace_value:
            return None
        modules = sorted(pathlib.Path(workspace_value).rglob(f"{name}.so"))
        if not modules:
            return None
        if len(modules) != 1:
            raise RuntimeError(
                f"expected one prebuilt FlashInfer module {name}.so; "
                f"found {len(modules)}"
            )
        import tvm_ffi

        started = time.perf_counter_ns()
        raw_module = tvm_ffi.load_module(str(modules[0]))
        if decode and not use_tensor_cores:
            module = get_batch_decode_jit_module(name, raw_module)
        else:
            module = get_batch_prefill_jit_module(name, raw_module)
        self._loaded_jit_modules[name] = module
        self._stats["typed_prebuilt_module_loads"] = (
            self._stats.get("typed_prebuilt_module_loads", 0) + 1
        )
        self._stats["typed_prebuilt_module_load_ns"] = self._stats.get(
            "typed_prebuilt_module_load_ns", 0
        ) + (time.perf_counter_ns() - started)
        self._stats["typed_prebuilt_module_bytes"] = (
            self._stats.get("typed_prebuilt_module_bytes", 0)
            + modules[0].stat().st_size
        )
        if list(jit_args[7]) != ["nta_runtime", "nta_work_items", "nta_dependencies"]:
            raise RuntimeError("prebuilt attention module has unexpected tensor ABI")
        return module

    @staticmethod
    def _bind_prebuilt_attention_module(
        wrapper: Any, module: Any, jit_args: list[Any]
    ) -> None:
        wrapper._jit_module = module
        wrapper._jit_additional_tensor_names = list(jit_args[7])

    def _instrumented_decode(self, form: str) -> list[Any]:
        cached = self._typed_decode_wrappers.get(form)
        if cached is not None:
            return cached
        name = (
            f"nta_sglang_decode_{form}_v11_"
            f"{'tc' if self.decode_use_tensor_cores else 'cc'}_"
            f"{self._instrumented_signature}"
        )
        args = self._jit_arguments(name, form)
        prebuilt = self._load_prebuilt_attention_module(
            name,
            args,
            decode=True,
            use_tensor_cores=self.decode_use_tensor_cores,
        )
        wrappers = []
        for _ in range(self.num_wrappers):
            wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer,
                "NHD",
                backend="fa2",
                use_tensor_cores=self.decode_use_tensor_cores,
                jit_args=None if prebuilt is not None else args,
            )
            if prebuilt is not None:
                self._bind_prebuilt_attention_module(wrapper, prebuilt, args)
            wrappers.append(wrapper)
        for wrapper in wrappers:
            self._wrapper_modules[id(wrapper)] = name
        self._typed_decode_wrappers[form] = wrappers
        return wrappers

    def _instrumented_prefill(self, form: str) -> tuple[list[Any], list[Any]]:
        cached = self._typed_prefill_wrappers.get(form)
        if cached is not None:
            return cached
        name = f"nta_sglang_prefill_{form}_v11_{self._instrumented_signature}"
        args = self._jit_arguments(name, form)
        prebuilt = self._load_prebuilt_attention_module(
            name,
            args,
            decode=False,
        )
        wrappers = []
        for _ in range(2 * self.num_wrappers):
            wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer,
                "NHD",
                backend="fa2",
                jit_args=None if prebuilt is not None else args,
            )
            if prebuilt is not None:
                self._bind_prebuilt_attention_module(wrapper, prebuilt, args)
            wrappers.append(wrapper)
        for wrapper in wrappers:
            self._wrapper_modules[id(wrapper)] = name
        split = self.num_wrappers
        result = (wrappers[:split], wrappers[split:])
        self._typed_prefill_wrappers[form] = result
        return result

    def _select_wrappers(self, demand_acquire: bool) -> None:
        form = "demand_acquire" if demand_acquire else "request_bound"
        self.decode_wrappers = self._instrumented_decode(form)
        if self.skip_prefill:
            return
        (
            self.prefill_wrappers_paged,
            self.prefill_wrappers_verify,
        ) = self._instrumented_prefill(form)

    def _select_stock_wrappers(self) -> None:
        self.decode_wrappers = list(self._stock_decode_wrappers)
        if self.skip_prefill:
            return
        self.prefill_wrappers_paged = list(self._stock_prefill_paged_wrappers)
        self.prefill_wrappers_verify = list(self._stock_prefill_verify_wrappers)

    def _adopt_typed_forward_metadata(
        self, forward_batch: Any, stock_metadata: Any
    ) -> None:
        """Atomically project one validated stock plan onto typed wrappers.

        The active batch already owns the exact schedule and acquisition
        geometry extracted from ``stock_metadata``.  Typed wrappers share that
        plan's workspace after adoption, so only wrapper identity changes; a
        second extraction would repeat CUDA-to-host synchronization without
        adding evidence.
        """

        if hasattr(stock_metadata, "decode_wrappers"):
            sources = tuple(stock_metadata.decode_wrappers)
            targets = tuple(self.decode_wrappers)
            field = "decode_wrappers"
        else:
            if stock_metadata.use_ragged:
                raise RuntimeError("typed acquisition requires paged prefill")
            sources = tuple(stock_metadata.prefill_wrappers)
            targets = tuple(
                self.prefill_wrappers_verify
                if forward_batch.forward_mode.is_target_verify()
                else self.prefill_wrappers_paged
            )
            field = "prefill_wrappers"
        if not sources or len(sources) != len(targets):
            raise RuntimeError("stock and typed FlashInfer wrapper counts disagree")
        for target, source in zip(targets, sources, strict=True):
            adopt_planned_flashinfer_state(target, source)
        batch = self._active_batch
        if batch is None:
            raise RuntimeError("typed FlashInfer adoption has no validated batch")
        source_to_target = {
            id(source): id(target)
            for target, source in zip(targets, sources, strict=True)
        }
        batch.adopt_wrapper_identity(source_to_target)
        self.forward_metadata = replace(stock_metadata, **{field: list(targets)})
        self._stock_wrapper_for_typed = {
            id(target): source for target, source in zip(targets, sources, strict=True)
        }
        self._stats["reused_flashinfer_plans"] = self._stats.get(
            "reused_flashinfer_plans", 0
        ) + len(targets)
        self._stats["ready_stock_wrapper_pairs"] = self._stats.get(
            "ready_stock_wrapper_pairs", 0
        ) + len(targets)

    def _run_ready_stock_numerical(
        self,
        typed_wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        layer: Any,
        *,
        causal: bool,
        window_left: int,
    ) -> torch.Tensor:
        stock = self._stock_wrapper_for_typed.get(id(typed_wrapper))
        if stock is None:
            raise RuntimeError("event-complete layer has no verified stock wrapper")
        query = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        common = {
            "sm_scale": layer.scaling,
            "logits_soft_cap": 0.0,
            "k_scale": layer.k_scale_float,
            "v_scale": layer.v_scale_float,
        }
        if isinstance(stock, BatchDecodeWithPagedKVCacheWrapper):
            result = stock.forward(query, kv_cache, **common)
        else:
            result = stock.forward(
                query,
                kv_cache,
                causal=causal,
                window_left=window_left,
                **common,
            )
        self._stats["stock_ready_external_attention_launches"] = (
            self._stats.get("stock_ready_external_attention_launches", 0) + 1
        )
        return result

    def init_cuda_graph_state(self, *args: Any, **kwargs: Any) -> None:
        super().init_cuda_graph_state(*args, **kwargs)

    def _build_execution_plan(
        self,
        *,
        bindings: tuple[RequestBinding, ...],
        schedule: Schedule,
        page_pairs: tuple[_PagePair, ...],
        work_dependencies: tuple[LeaseWorkDependency | None, ...],
        layer: int,
        unit_bytes: int,
    ) -> ExecutionPlan:
        """Translate one native attention launch into the immutable contract.

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
        if work_dependencies:
            if len(work_dependencies) != schedule.work_count:
                raise RuntimeError("execution dependencies do not match CTA schedule")
        elif page_pairs and len(page_pairs) != schedule.work_count:
            raise RuntimeError("execution page pairs do not match CTA schedule")
        for logical_work, request_index in enumerate(schedule.request_indices):
            if request_index < 0 or request_index >= len(bindings):
                raise RuntimeError("FlashInfer schedule referenced an invalid request")
            external_rows = (
                work_dependencies[logical_work].row_count
                if work_dependencies and work_dependencies[logical_work] is not None
                else len(page_pairs[logical_work][0])
                if page_pairs
                else 0
            )
            candidate_units = max(1, external_rows)
            tiles.append(
                ExecutionTile(
                    work_id=work_id,
                    binding=bindings[request_index],
                    layer=layer,
                    logical_begin=int(schedule.kv_tile_indices[logical_work]),
                    candidate_units=candidate_units,
                    selected_ids=(),
                    unit_bytes=unit_bytes,
                    ready=external_rows == 0,
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
        self._stats["semantic_dense_tiles"] += len(tiles)
        plan = ExecutionPlan.from_tiles(
            epoch=self._current_engine_batch.epoch,
            granularity=self._execution_config.protocol.granularity,
            protocol=self._execution_config.protocol,
            tiles=tiles,
        )
        self._stats.update(plan.expose_stats())
        return plan

    def _ensure_execution_plan(
        self,
        wrapper: Any,
        layer: Any,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        *,
        verify: bool,
    ) -> tuple[bool, int]:
        """Create compact exact work and optionally instantiate its specification."""
        if self._active_batch is None:
            raise RuntimeError("cannot create execution session without active batch")
        batch = self._active_batch
        wrapper_id = id(wrapper)
        schedule = batch.schedules.get(wrapper_id)
        # SGLang plans each wrapper once per ForwardBatch. Production consumes
        # that already-validated snapshot for every model layer; rereading the
        # CUDA int-workspace here would introduce one D2H synchronization per
        # layer. Verification deliberately re-extracts and compares it.
        if schedule is None or verify:
            decode_wrappers = tuple(
                getattr(self.forward_metadata, "decode_wrappers", ()) or ()
            )
            extracted = (
                decode_schedule(wrapper)
                if any(wrapper is candidate for candidate in decode_wrappers)
                else paged_prefill_schedule(wrapper)
            )
            if schedule is not None and schedule != extracted:
                raise RuntimeError(
                    "FlashInfer wrapper schedule changed within a forward"
                )
            schedule = extracted
        page_pairs = batch.page_pairs.get(wrapper_id, ())
        work_dependencies = batch.work_dependencies.get(wrapper_id, ())
        if not page_pairs and not work_dependencies:
            page_pairs = tuple(((), ()) for _ in range(schedule.work_count))
            batch.page_pairs[wrapper_id] = page_pairs
        batch.schedules[wrapper_id] = schedule
        unit_bytes = int(
            kv_cache[0][0].numel() * kv_cache[0].element_size()
            + kv_cache[1][0].numel() * kv_cache[1].element_size()
        )
        topology = batch.work_topologies.get(wrapper_id)
        built = topology is None
        if topology is None:
            demand_units = tuple(
                max(
                    1,
                    work_dependencies[work_id].row_count
                    if work_dependencies and work_dependencies[work_id] is not None
                    else len(page_pairs[work_id][0])
                    if page_pairs
                    else 0,
                )
                for work_id in range(schedule.work_count)
            )
            topology = ExactWorkTopology.from_schedule(
                epoch=self._current_engine_batch.epoch,
                bindings=batch.bindings,
                request_indices=schedule.request_indices,
                logical_work=schedule.kv_tile_indices,
                demand_units=demand_units,
                unit_bytes=unit_bytes,
                estimated_compute_ns=self._host_cost_model.tile_compute_ns,
            )
            batch.work_topologies[wrapper_id] = topology
            self._stats["work_topology_items"] += topology.work_count
        elif topology.unit_bytes != unit_bytes:
            raise RuntimeError("SGLang KV row geometry changed within a forward")

        if verify:
            semantic_started = time.perf_counter_ns()
            batch.execution = self._build_execution_plan(
                bindings=batch.bindings,
                schedule=schedule,
                page_pairs=page_pairs,
                work_dependencies=work_dependencies,
                layer=int(layer.layer_id) - self._model_start_layer,
                unit_bytes=unit_bytes,
            )
            batch.verification_session = ExecutionSession.from_plan(batch.execution)
            semantic_elapsed = time.perf_counter_ns() - semantic_started
            self._stats["semantic_plan_builds"] += 1
            if self._profile_cpu:
                self._stats["semantic_plan_cpu_ns"] += semantic_elapsed
            self._stats["semantic_verifier_sessions"] += 1
        else:
            # Production retains only the compact topology. The semantic plan
            # is an opt-in specification, never a second serving state machine.
            batch.execution = None
            batch.verification_session = None
            semantic_elapsed = 0
        return built, semantic_elapsed

    def _upload_resident_plan(
        self,
        wrapper: Any,
        schedule: Schedule,
        topology: ExactWorkTopology,
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

        dependency_spans = []
        dependencies: list[AcquireRequirement] = []
        for work_ticket, (request_index, logical_begin) in enumerate(
            zip(schedule.request_indices, schedule.kv_tile_indices, strict=True)
        ):
            dependency_begin = len(dependencies)
            dependencies.extend(
                (
                    direct_requirement(self._runtime.device_view, 1),
                    direct_requirement(self._runtime.device_view, 1),
                )
            )
            dependency_spans.append(WorkDependencySpan(dependency_begin, 2, 2))

        plan.upload_exact(
            topology,
            dependency_spans,
            dependencies,
            stream=torch.cuda.current_stream(),
        )
        allocation.signature = signature
        allocation.object_count = 0
        allocation.direct_work_count = schedule.work_count
        allocation.external_object_slots = tuple(() for _ in range(schedule.work_count))
        return plan

    def _record_execution_layer(self, layer: Any, *, final_layer: bool) -> None:
        """Commit the semantic work boundary after native attention returns."""
        if self._active_batch is None:
            raise RuntimeError("attention returned without a typed work topology")
        local_layer = int(layer.layer_id) - self._model_start_layer
        verifier = self._active_batch.verification_session
        if verifier is not None:
            self._stats.update(verifier.record_layer_completion(local_layer))
        if self._tier_service.is_nvme:
            # NVMe object slots are reused between consecutive layers as well
            # as between forwards.  Recording at every layer gives the next
            # replacement the exact predecessor event; waiting for only the
            # final layer would leave a same-forward slot replacement unsafe.
            self._nvme_slots.record_consumer(torch.cuda.current_stream())
        elif self._tier_service.is_host_staged and final_layer:
            self._indexed_object_quiescence_event.record(torch.cuda.current_stream())
            self._indexed_object_quiescence_recorded = True

    def _prepare_nvme_regions(self) -> None:
        """Describe and coalesce stable framework KV allocations at startup."""

        started = time.perf_counter_ns()
        catalog = self._tier_service.catalog
        if catalog is None or catalog.layer_count != self._global_model_layer_count:
            raise RuntimeError(
                "NVMe catalog layer count does not match the SGLang model"
            )
        if catalog.page_tokens != 1:
            raise RuntimeError(
                "NVMe SGLang integration currently requires page_tokens=1"
            )
        destinations: list[HbmDestinationSlice] = []
        for local_layer in range(self._model_layer_count):
            layer_id = self._model_start_layer + local_layer
            tensors = (
                ("key", self.token_to_kv_pool._get_key_buffer(layer_id)),
                ("value", self.token_to_kv_pool._get_value_buffer(layer_id)),
            )
            for kind, tensor in tensors:
                if not tensor.is_cuda or int(tensor.nbytes) <= 0:
                    raise RuntimeError(
                        f"NVMe {kind} region for layer {layer_id} is not live CUDA HBM"
                    )
                destinations.append(
                    HbmDestinationSlice(
                        (layer_id, kind),
                        int(tensor.data_ptr()),
                        int(tensor.nbytes),
                    )
                )
        try:
            prepared = self._tier_service.prepare_nvme_hbm_destinations(
                tuple(destinations)
            )
        except BaseException as error:
            raise RuntimeError(
                "NVMe worker-prepare could not register the complete local KV "
                f"destination set (layers=[{self._model_start_layer}, "
                f"{self._model_end_layer}), tensors={len(destinations)})"
            ) from error
        self._nvme_regions.update(prepared.regions)
        self._stats["nvme_region_prepare_ns"] = time.perf_counter_ns() - started
        self._stats["nvme_region_count"] = prepared.registration_count
        self._stats["nvme_region_bytes"] = prepared.registration_bytes
        self._stats["nvme_destination_slice_count"] = prepared.destination_count
        self._stats["nvme_destination_slice_bytes"] = prepared.destination_bytes
        self._stats["nvme_shared_region_slices"] = (
            prepared.destination_count - prepared.registration_count
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
        if pending is not None:
            self._account_tier_selection(pending)
        # A framework CUDA graph has immutable kernel arguments.  Capturing a
        # request-bound NTA wrapper would freeze the capture-time request slot
        # and silently attribute later replays to the wrong generation.  Graph
        # execution therefore has one explicit contract: acquisition fully
        # materializes KV into SGLang's device pool before replay, and the
        # captured numerical consumer is stock FlashInfer.  Native typed
        # wrappers remain available for eager and NTA-owned finite graphs,
        # whose work-plan buffers are updated safely between replays.
        self._stock_forward = True
        self._select_stock_wrappers()
        super().init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)
        if in_capture:
            # SGLang's capture batch consists entirely of dummy rows, commonly
            # all using request-pool slot zero.  It has no serving identity and
            # must not consume generations or pollute the persistent registry.
            bindings: tuple[RequestBinding, ...] = ()
            self._stats["graph_capture_dummy_rows"] = self._stats.get(
                "graph_capture_dummy_rows", 0
            ) + int(getattr(forward_batch, "batch_size", 0) or 0)
        else:
            bindings = self._bind_forward_requests(
                forward_batch, allow_capture_ids=False
            )
        if pending is None:
            self._active_batch = _ActiveBatch(bindings, {}, None, {}, {}, {}, ())
            self._stats["resident_reference_batches"] += 1
            self._stats["stock_resident_batches"] += 1
        else:
            self._prepare_missing_host_layers(pending)
            final_layer = _require_exact_prefetch_layers(
                pending.prefetched_layers,
                self._model_layer_count,
                consumer="CUDA graph replay",
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
            if self._profile_barrier:
                arrive = torch.cuda.Event(enable_timing=True)
                arrive.record(torch.cuda.current_stream())
                self._barrier_profiles.append(
                    (
                        arrive,
                        pending.prefetched_layers[final_layer].ready_event,
                        self._model_start_layer + final_layer,
                    )
                )
                self._stats["profiled_graph_prefetch_waits"] = (
                    self._stats.get("profiled_graph_prefetch_waits", 0) + 1
                )
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

    def _activate_stock_prefetch(
        self,
        bindings: tuple[RequestBinding, ...],
        pending: PendingHostLoad,
        *,
        count_batch: bool = True,
    ) -> None:
        """Bind a complete exact prefetch without materializing an unused plan."""
        _require_exact_prefetch_layers(
            pending.prefetched_layers,
            self._model_layer_count,
            consumer="stock external attention",
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
        if count_batch:
            self._stats["batches"] += 1
            self._stats["hicache_external_batches"] += 1
        self._stats["stock_prefetched_external_batches"] += 1
        self._stats["stock_prefetch_metadata_fastpath_batches"] = (
            self._stats.get("stock_prefetch_metadata_fastpath_batches", 0) + 1
        )

    def init_forward_metadata(self, forward_batch: Any) -> None:
        self._cuda_graph_mode = False
        self._stock_forward = False
        self._stock_wrapper_for_typed.clear()
        if forward_batch.forward_mode.is_mixed():
            self._stats["mixed_forward_batches"] += 1
            self._stats["mixed_forward_requests"] += len(
                tuple(getattr(forward_batch, "rids", ()) or ())
            )
        counter = getattr(self.token_to_kv_pool, "layer_transfer_counter", None)
        consumer_index = -1 if counter is None else int(counter.consumer_index)
        pending = self._hicache.get(consumer_index)
        if pending is not None:
            self._account_tier_selection(pending)
        measured_host_selection = (
            pending is not None and self._tier_service.is_host_staged
        )
        mixed_host_batch = (
            measured_host_selection and forward_batch.forward_mode.is_mixed()
        )
        if pending is None:
            self._stock_forward = True
            self._select_stock_wrappers()
        elif measured_host_selection:
            # The stock plan supplies the exact schedule for a no-overhead
            # direct decision. A typed plan is built later only if measured
            # overlap justifies unresolved work.
            self._select_stock_wrappers()
        else:
            self._select_wrappers(True)
        original_use_paged = self.use_paged
        self.use_paged = True
        try:
            super().init_forward_metadata(forward_batch)
            if pending is None:
                # A resident stock forward has no acquisition identity to
                # publish and no native work to attribute.  Account its known
                # all-layer dispatch once here; the per-layer methods can then
                # be a thin call into SGLang's stock backend.
                stock_layers = self._model_layer_count
                self._stats["stock_attention_launches"] += stock_layers
                self._stats["stock_resident_attention_launches"] += stock_layers
                if forward_batch.forward_mode.is_decode_or_idle():
                    self._stats["decode_launches"] += stock_layers
                else:
                    self._stats["prefill_launches"] += stock_layers
                marker = any(
                    str(request_id).startswith(OBSERVATION_MARKER_REQUEST_PREFIX)
                    for request_id in tuple(getattr(forward_batch, "rids", ()) or ())
                )
                self._active_batch = _ActiveBatch(
                    (),
                    {},
                    None,
                    {},
                    {},
                    {},
                    (),
                    publish_stats_on_completion=marker,
                )
                self._stats["batches"] += 1
                self._stats["resident_reference_batches"] += 1
                self._stats["stock_resident_batches"] += 1
                return
            bind_started = time.perf_counter_ns() if self._profile_cpu else 0
            bindings = self._bind_forward_requests(
                forward_batch, allow_capture_ids=False
            )
            if self._profile_cpu:
                self._stats["request_bind_cpu_ns"] = self._stats.get(
                    "request_bind_cpu_ns", 0
                ) + (time.perf_counter_ns() - bind_started)
            if measured_host_selection:
                stock_metadata = self.forward_metadata
                selected = self._init_external_metadata(
                    forward_batch,
                    pending,
                    bindings=bindings,
                )
                if selected is None:
                    raise RuntimeError("host-staged batch has no execution decision")
                self._record_host_selection(selected)
                prefetch_fully_published = len(pending.prefetched_layers) == (
                    self._model_layer_count
                )
                if prefetch_fully_published or (
                    selected.rounds == 1 and not selected.overlap_initial
                ):
                    self._prepare_missing_host_layers(pending)
                    self._stock_forward = True
                    self._activate_stock_prefetch(bindings, pending, count_batch=False)
                    self._stats["host_direct_batches"] = (
                        self._stats.get("host_direct_batches", 0) + 1
                    )
                    if mixed_host_batch:
                        self._stats["host_mixed_direct_batches"] = (
                            self._stats.get("host_mixed_direct_batches", 0) + 1
                        )
                    if prefetch_fully_published:
                        self._stats["host_bound_after_full_publication_batches"] = (
                            self._stats.get(
                                "host_bound_after_full_publication_batches", 0
                            )
                            + 1
                        )
                    return

                incremental_setup_started = time.perf_counter_ns()
                wrapper_select_started = time.perf_counter_ns()
                self._select_wrappers(True)
                wrapper_select_ns = time.perf_counter_ns() - wrapper_select_started
                adoption_started = time.perf_counter_ns()
                self._adopt_typed_forward_metadata(forward_batch, stock_metadata)
                adoption_ns = time.perf_counter_ns() - adoption_started
                if self._active_batch is None:  # pragma: no cover - set above
                    raise RuntimeError("incremental host batch lost its metadata")
                self._active_batch.incremental_metadata_setup_ns = (
                    time.perf_counter_ns() - incremental_setup_started
                )
                for counter, elapsed in (
                    ("incremental_wrapper_select_cpu_ns", wrapper_select_ns),
                    ("incremental_metadata_adoption_cpu_ns", adoption_ns),
                    (
                        "incremental_metadata_setup_cpu_ns",
                        self._active_batch.incremental_metadata_setup_ns,
                    ),
                ):
                    self._stats[counter] = self._stats.get(counter, 0) + elapsed
                self._stats["host_incremental_batches"] = (
                    self._stats.get("host_incremental_batches", 0) + 1
                )
                if mixed_host_batch:
                    self._stats["host_typed_mixed_batches"] = (
                        self._stats.get("host_typed_mixed_batches", 0) + 1
                    )
                return
            self._init_external_metadata(forward_batch, pending, bindings=bindings)
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

    def _record_host_selection(self, selected: HostExecutionPlan) -> None:
        self._stats["host_selection_predicted_atomic_ns"] = (
            self._stats.get("host_selection_predicted_atomic_ns", 0)
            + selected.predicted_atomic_ns
        )
        self._stats["host_selection_predicted_selected_ns"] = (
            self._stats.get("host_selection_predicted_selected_ns", 0)
            + selected.predicted_incremental_ns
        )
        reason_key = f"host_selection_{selected.selection_reason}_batches"
        self._stats[reason_key] = self._stats.get(reason_key, 0) + 1

    def _record_request_batch_heterogeneity(
        self,
        bindings: Sequence[RequestBinding],
        sequence_lengths: Sequence[int],
        acquisitions: Sequence[SglangAcquisitionSpan],
    ) -> None:
        if len(bindings) < 2:
            return
        self._stats["multi_request_engine_batches"] += 1
        axes = _request_batch_heterogeneity(
            bindings, sequence_lengths, acquisitions
        )
        if not axes:
            return
        self._stats["heterogeneous_engine_batches"] += 1
        self._stats["multi_axis_heterogeneous_batches"] += int(len(axes) > 1)
        for axis in axes:
            self._stats[f"{axis}_heterogeneous_batches"] += 1

    def _init_external_metadata(
        self,
        forward_batch: Any,
        pending: PendingHostLoad,
        *,
        bindings: tuple[RequestBinding, ...] | None = None,
        count_batch: bool = True,
    ) -> HostExecutionPlan | None:
        self._account_tier_selection(pending)
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
        if bindings is None:
            bindings = self._bind_forward_requests(
                forward_batch, allow_capture_ids=False
            )

        schedules: dict[int, Schedule] = {}
        for wrapper in wrappers:
            schedule = extractor(wrapper)
            self._validate_schedule(schedule, bindings)
            schedules[id(wrapper)] = schedule
        bounded_direct = self._prove_direct_metadata_execution(
            schedules, pending, bindings
        )
        if bounded_direct is not None:
            self._active_batch = _ActiveBatch(
                bindings,
                schedules,
                pending,
                {},
                {},
                pending.prefetched_layers,
                pending.prefetch_tensors,
                bounded_direct,
                self._grouping,
            )
            self._stats["host_selection_bound_fastpath_batches"] += 1
            if self._profile_cpu:
                self._stats["metadata_cpu_ns"] = self._stats.get(
                    "metadata_cpu_ns", 0
                ) + (time.perf_counter_ns() - metadata_started)
            if count_batch:
                self._stats["batches"] += 1
                self._stats["hicache_external_batches"] += 1
            return bounded_direct
        metadata = forward_metadata(forward_batch)
        if len(metadata.acquisitions) != len(bindings):
            raise RuntimeError(
                "SGLang acquisition metadata does not match request bindings"
            )
        lease_rows = int(pending.device_indices.numel())
        acquisitions = _resolve_request_acquisitions(
            metadata.acquisitions,
            pending.transfers_by_operation(),
            lease_transfer_rows=lease_rows,
        )
        sequence_lengths = _cpu_sequence_lengths(forward_batch, len(bindings))
        self._record_request_batch_heterogeneity(
            bindings, sequence_lengths, acquisitions
        )
        if (
            self._tier_service.is_host_staged
            and os.environ.get("NTA_VERIFY_REQUEST_DEPENDENCIES") != "1"
        ):
            work_dependencies = {
                wrapper_id: _project_work_acquisitions(
                    schedule, acquisitions, sequence_lengths
                )
                for wrapper_id, schedule in schedules.items()
            }
            transfer_dependencies = {
                wrapper_id: _capacity_constrained_transfer_dependencies(
                    dependencies,
                    maximum_groups=self._object_capacity // 2,
                )
                for wrapper_id, dependencies in work_dependencies.items()
            }
            host_execution = self._typed_lease_execution_plan(
                schedules, work_dependencies, transfer_dependencies, pending
            )
            mixed_forward = forward_batch.forward_mode.is_mixed()
            if mixed_forward:
                representative_id = next(iter(schedules))
                representative_schedule = schedules[representative_id]
                representative_dependencies = work_dependencies[representative_id]
                self._stats["mixed_scheduled_requests"] += len(
                    set(representative_schedule.request_indices)
                )
                self._stats["mixed_direct_work_items"] += sum(
                    dependency is None for dependency in representative_dependencies
                )
                self._stats["mixed_external_work_items"] += sum(
                    dependency is not None for dependency in representative_dependencies
                )
            self._active_batch = _ActiveBatch(
                bindings,
                schedules,
                pending,
                {},
                {},
                pending.prefetched_layers,
                pending.prefetch_tensors,
                host_execution,
                "request",
                work_dependencies=work_dependencies,
                transfer_dependencies=transfer_dependencies,
                lease_transfer_rows=lease_rows,
            )
            self._stats["typed_acquisition_batches"] += 1
            self._stats["typed_acquisition_rows"] += lease_rows
            self._stats["typed_acquisition_work_items"] += sum(
                len(dependencies) for dependencies in work_dependencies.values()
            )
            exact_groups = sum(
                len({item for item in dependencies if item is not None})
                for dependencies in work_dependencies.values()
            )
            transfer_groups = sum(
                len({item for item in dependencies if item is not None})
                for dependencies in transfer_dependencies.values()
            )
            self._stats["typed_exact_dependency_groups"] = (
                self._stats.get("typed_exact_dependency_groups", 0) + exact_groups
            )
            self._stats["typed_transfer_groups"] = (
                self._stats.get("typed_transfer_groups", 0) + transfer_groups
            )
            self._stats["typed_granularity_constrained_batches"] = self._stats.get(
                "typed_granularity_constrained_batches", 0
            ) + int(transfer_groups < exact_groups)
            self._stats["typed_transfer_group_max_rows"] = max(
                self._stats.get("typed_transfer_group_max_rows", 0),
                max(
                    item.row_count
                    for dependencies in transfer_dependencies.values()
                    for item in dependencies
                    if item is not None
                ),
            )
            if self._profile_cpu:
                self._stats["metadata_cpu_ns"] = self._stats.get(
                    "metadata_cpu_ns", 0
                ) + (time.perf_counter_ns() - metadata_started)
            if count_batch:
                self._stats["batches"] += 1
                self._stats["hicache_external_batches"] += 1
            return host_execution
        pending_pages = set(pending.materialize_mapping())
        planned_pages: set[int] = set()
        tile_page_pairs: dict[int, tuple[_PagePair, ...]] = {}
        for wrapper in wrappers:
            layout = self._wrapper_layout(wrapper)
            planned_pages.update(layout[1])
            tile_page_pairs[id(wrapper)] = self._work_page_pairs(
                wrapper, schedules[id(wrapper)], pending, layout=layout
            )
        missing = pending_pages - planned_pages
        if missing:
            raise RuntimeError(
                f"attention metadata omits {len(missing)} promoted HiCache pages"
            )
        mixed_forward = forward_batch.forward_mode.is_mixed()
        request_page_pairs = (
            {
                wrapper_id: _group_external_pages_by_request(
                    schedules[wrapper_id], pairs
                )
                for wrapper_id, pairs in tile_page_pairs.items()
            }
            if self._grouping == "request" or mixed_forward
            else None
        )
        if mixed_forward:
            if request_page_pairs is None:  # pragma: no cover - construction invariant
                raise RuntimeError("mixed execution omitted request grouping")
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
        grouping = self._grouping
        if grouping == "tile":
            page_pairs = tile_page_pairs
            host_execution = self._metadata_execution_plan(
                schedules, tile_page_pairs, pending
            )
        else:
            if request_page_pairs is None:  # pragma: no cover - grouping invariant
                raise RuntimeError("request execution omitted request grouping")
            page_pairs = request_page_pairs
            host_execution = self._metadata_execution_plan(
                schedules, request_page_pairs, pending
            )
        self._active_batch = _ActiveBatch(
            bindings,
            schedules,
            pending,
            page_pairs,
            {},
            pending.prefetched_layers,
            pending.prefetch_tensors,
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

    def _collect_layer_service_profiles(self) -> None:
        """Retire completed attention-arrival gaps without synchronizing."""

        pending: list[_LayerServiceProfile] = []
        for profile in self._layer_service_profiles:
            if not profile.finish.query():
                pending.append(profile)
                continue
            elapsed_ns = max(
                1, round(profile.start.elapsed_time(profile.finish) * 1_000_000.0)
            )
            curve = self._layer_service_curves.get(
                profile.key,
                LayerDeadlineServiceCurve(
                    minimum_samples=self._layer_service_minimum_samples,
                    maximum_samples=self._layer_service_maximum_samples,
                ),
            ).with_observation(elapsed_ns)
            self._layer_service_curves[profile.key] = curve
            self._stats["layer_service_profiled_intervals"] += 1
        self._layer_service_profiles = pending
        calibrated = tuple(
            curve for curve in self._layer_service_curves.values() if curve.calibrated
        )
        self._stats["layer_service_calibrated_shapes"] = len(calibrated)

    def _record_layer_arrival(
        self, phase: str, query: torch.Tensor, layer: Any
    ) -> None:
        """Sample bounded per-layer compute slack for one exact forward shape."""

        batch = self._active_batch
        if (
            batch is None
            or batch.pending_host_load is None
            or not self._tier_service.is_host_staged
        ):
            return
        query_rows = int(query.shape[0])
        key = (phase, query_rows, len(batch.bindings))
        if min(query_rows, len(batch.bindings)) <= 0:
            raise RuntimeError("layer service calibration has an empty forward")
        if batch.layer_service_key is not None and batch.layer_service_key != key:
            raise RuntimeError("attention shape changed within one model forward")
        batch.layer_service_key = key
        curve = self._layer_service_curves.get(
            key,
            LayerDeadlineServiceCurve(
                minimum_samples=self._layer_service_minimum_samples,
                maximum_samples=self._layer_service_maximum_samples,
            ),
        )
        inflight = sum(profile.key == key for profile in self._layer_service_profiles)
        if len(curve.samples_ns) + inflight >= curve.maximum_samples:
            batch.layer_arrival_event = None
            return

        local_layer = int(layer.layer_id) - self._model_start_layer
        if not 0 <= local_layer < self._model_layer_count:
            raise RuntimeError("attention layer is outside the local model range")
        arrival = torch.cuda.Event(enable_timing=True)
        arrival.record(torch.cuda.current_stream())
        previous = batch.layer_arrival_event
        if previous is not None:
            if batch.layer_arrival_local_layer + 1 != local_layer:
                raise RuntimeError("attention layers did not arrive in model order")
            self._layer_service_profiles.append(
                _LayerServiceProfile(previous, arrival, key)
            )
        batch.layer_arrival_event = arrival
        batch.layer_arrival_local_layer = local_layer

    def _mover_profile_enabled(self, engine: str) -> bool:
        if engine not in {"sm", "copy_engine"}:
            raise ValueError("unknown host mover engine")
        if self._profile_transfer:
            return True
        if self._host_mover != "auto":
            return False
        samples = (
            self._host_mover_service_model.sm_samples
            if engine == "sm"
            else self._host_mover_service_model.copy_samples
        )
        inflight = sum(
            profile.engine == engine for profile in self._mover_profiles
        )
        return samples + inflight < self._host_mover_calibration_samples

    def _host_mover_lease_plan(
        self,
        pending: PendingHostLoad,
        row_bytes_by_layer: tuple[tuple[int, int], ...],
        transfer_count: int,
    ) -> HostMoverLeasePlan:
        cached = pending.mover_plan
        if cached is not None:
            if cached.row_count != transfer_count:
                raise RuntimeError("HiCache mover map changed during a lease")
            return cached

        # Retire completed CUDA-event samples before selecting a new lease.
        # This is query-only and never synchronizes the scheduler thread.
        self._collect_mover_profiles()
        self._collect_layer_service_profiles()
        controller = pending.controller
        moved_source_indices, moved_staging_indices = controller.move_indices(
            pending.host_indices, pending.device_indices
        )
        bytes_per_transferred_row = sum(
            key_bytes + value_bytes for key_bytes, value_bytes in row_bytes_by_layer
        )
        active = self._active_batch
        layer_service_key = None if active is None else active.layer_service_key
        layer_curve = (
            None
            if layer_service_key is None
            else self._layer_service_curves.get(layer_service_key)
        )
        self._stats["layer_service_last_plan_key"] = (
            None if layer_service_key is None else list(layer_service_key)
        )
        self._stats["layer_service_last_plan_samples"] = (
            0 if layer_curve is None else len(layer_curve.samples_ns)
        )
        if layer_service_key is None:
            self._stats["layer_service_plan_key_missing_batches"] += 1
        elif layer_curve is None:
            self._stats["layer_service_plan_curve_missing_batches"] += 1
        elif not layer_curve.calibrated:
            self._stats["layer_service_plan_curve_uncalibrated_batches"] += 1
        else:
            self._stats["layer_service_plan_curve_calibrated_batches"] += 1
        overlap_compute_ns = (
            0
            if layer_curve is None
            else layer_curve.overlap_budget_ns(max(0, len(row_bytes_by_layer) - 1))
        )
        self._stats["host_mover_overlap_compute_ns"] += overlap_compute_ns
        self._stats["layer_service_conservative_ns"] = (
            0 if layer_curve is None else layer_curve.conservative_layer_ns
        )
        auto_analysis = self._host_mover == "auto"
        analyze_layout = (
            self._host_mover == "copy_engine"
            or auto_analysis
            or self._profile_index_layout
        )
        if (
            self._host_mover == "copy_engine" or auto_analysis
        ) and self._copy_engine_max_operations < 2:
            raise RuntimeError(
                "copy-engine mover needs at least two K/V operations per layer"
            )

        tensor_plan: TensorIndexedMoverPlan | None = None
        layout_cpu_ns = 0
        planner_policy = self._host_mover
        calibration_probe_sm = False
        if self._host_mover == "auto":
            sm_inflight = any(
                profile.engine == "sm" for profile in self._mover_profiles
            )
            copy_inflight = any(
                profile.engine == "copy_engine" for profile in self._mover_profiles
            )
            if self._host_mover_service_model.sm_samples == 0 and not sm_inflight:
                planner_policy = "sm"
                calibration_probe_sm = True
            elif (
                self._host_mover_service_model.copy_samples == 0
                and not copy_inflight
            ):
                planner_policy = "probe_copy"
        if analyze_layout:
            layout_started = time.perf_counter_ns()
            tensor_plan = plan_indexed_tensor_mover(
                moved_source_indices,
                moved_staging_indices,
                row_bytes=bytes_per_transferred_row,
                copy_operations_per_run=2 * len(row_bytes_by_layer),
                maximum_copy_runs=self._copy_engine_max_operations // 2,
                service_model=self._host_mover_service_model,
                policy=planner_policy,
                overlap_compute_ns=overlap_compute_ns,
                # SGLang's allocator owns destination uniqueness. Rechecking
                # it requires a device sort and is therefore a verification
                # concern, not steady-state transport work.
                validate_unique_destinations=(
                    self._profile_index_layout
                    or os.environ.get("NTA_VERIFY_INDEX_MAP") == "1"
                ),
            )
            layout_cpu_ns = time.perf_counter_ns() - layout_started
            if self._host_mover != "sm":
                self._stats["copy_engine_layout_cpu_ns"] += layout_cpu_ns

        if tensor_plan is None:
            device = controller.mem_pool_device.device
            source = moved_source_indices.to(
                device=device, dtype=torch.int32, non_blocking=True
            )
            destination = moved_staging_indices.to(
                device=device, dtype=torch.int32, non_blocking=True
            )
            predicted_sm_ns = self._host_mover_service_model.candidate_ns(
                total_rows=transfer_count,
                copy_rows=0,
                copy_run_count=0,
                row_bytes=bytes_per_transferred_row,
                copy_operations_per_run=2 * len(row_bytes_by_layer),
                overlap_compute_ns=overlap_compute_ns,
            )
            if predicted_sm_ns is None:  # pragma: no cover - zero-copy invariant
                raise RuntimeError("SM mover produced no service estimate")
            plan = HostMoverLeasePlan(
                transfer_count,
                "sm",
                (),
                source,
                destination,
                None,
                layout_cpu_ns,
                predicted_sm_ns,
                predicted_sm_ns,
                (
                    "calibration_probe_sm"
                    if calibration_probe_sm
                    else "forced_sm"
                    if self._host_mover == "sm"
                    else "uncalibrated_copy_engine"
                ),
            )
        else:
            selection_reason = (
                "calibration_probe_sm"
                if calibration_probe_sm
                else tensor_plan.selection_reason
            )
            plan = HostMoverLeasePlan(
                transfer_count,
                tensor_plan.kind,
                tensor_plan.copy_runs,
                tensor_plan.sm_source_indices,
                tensor_plan.sm_destination_indices,
                tensor_plan.layout,
                layout_cpu_ns,
                tensor_plan.predicted_sm_ns,
                tensor_plan.predicted_selected_ns,
                selection_reason,
            )
        pending.mover_plan = plan
        pending.prefetch_tensors = plan.retained_tensors

        self._stats[f"host_mover_{plan.kind}_batches"] += 1
        self._stats["copy_engine_selected_runs"] += len(plan.copy_runs)
        self._stats["copy_engine_selected_rows"] += plan.copy_row_count
        self._stats["sm_mover_rows"] += plan.sm_row_count
        self._stats["host_mover_predicted_sm_ns"] += plan.predicted_sm_ns
        self._stats["host_mover_predicted_selected_ns"] += (
            plan.predicted_selected_ns or 0
        )
        reason_counter = {
            "calibration_probe_sm": "host_mover_calibration_probe_sm_batches",
            "calibration_probe_copy": "host_mover_calibration_probe_copy_batches",
            "uncalibrated_copy_engine": ("host_mover_uncalibrated_copy_engine_batches"),
            "insufficient_gain": "host_mover_insufficient_gain_batches",
            "service_cost": "host_mover_service_cost_batches",
        }.get(plan.selection_reason)
        if reason_counter is not None:
            self._stats[reason_counter] += 1
        if self._profile_index_layout:
            layout = plan.layout
            if layout is None:
                raise RuntimeError("indexed-layout profiling produced no layout")
            samples = self._stats.setdefault("indexed_layout_run_rows_samples", [])
            if len(samples) < 16:
                samples.append([run.row_count for run in layout.runs])
            eligible_rows = layout.eligible_rows(
                row_bytes=bytes_per_transferred_row,
                minimum_copy_bytes=self._profile_index_min_bytes,
            )
            candidate_bytes = eligible_rows * sum(
                key_bytes + value_bytes for key_bytes, value_bytes in row_bytes_by_layer
            )
            self._stats["indexed_layout_profiles"] += 1
            self._stats["indexed_layout_rows"] += layout.row_count
            self._stats["indexed_layout_runs"] += len(layout.runs)
            self._stats["indexed_layout_eligible_rows"] += eligible_rows
            self._stats["indexed_layout_candidate_bytes"] += candidate_bytes
            self._stats["indexed_layout_maximum_run_rows"] = max(
                self._stats["indexed_layout_maximum_run_rows"],
                layout.maximum_run_rows,
            )
            self._stats["indexed_layout_profile_cpu_ns"] += layout_cpu_ns
        return plan

    def _prepare_host_pipeline(
        self,
        pending: PendingHostLoad,
        *,
        first_local_layer: int = 0,
        last_local_layer: int | None = None,
    ) -> None:
        """Enqueue one non-overlapping, half-open model-layer frontier."""
        if self._profile_barrier:
            # Drain outstanding barrier measurements before this batch
            # re-records the shared per-layer ready events.
            self._collect_barrier_profiles()
        pipeline_started = time.perf_counter_ns() if self._profile_cpu else 0
        controller = pending.controller
        layer_count = int(controller.layer_num)
        if last_local_layer is None:
            last_local_layer = layer_count
        if not 0 <= first_local_layer < last_local_layer <= layer_count:
            raise RuntimeError("HiCache acquisition frontier is outside the model")
        overlapping_layers = sorted(
            set(range(first_local_layer, last_local_layer))
            & set(pending.prefetched_layers)
        )
        if overlapping_layers:
            raise RuntimeError(
                "HiCache acquisition frontier overlaps published layers "
                f"{overlapping_layers}"
            )
        acquired_layer_count = last_local_layer - first_local_layer
        transfer_first_slot, _ = _pipeline_object_range(
            self._object_capacity, pending.consumer_index, layer_count
        )
        transfer_count = int(pending.host_indices.numel())
        if transfer_count <= 0 or transfer_count != int(pending.device_indices.numel()):
            raise RuntimeError("HiCache host pipeline has no promoted pages")
        device_pool = controller.mem_pool_device
        host_keys = tuple(controller.mem_pool_host.k_data_refs)
        host_values = tuple(controller.mem_pool_host.v_data_refs)
        if len(host_keys) != layer_count or len(host_values) != layer_count:
            raise RuntimeError("HiCache host K/V layer geometry is incomplete")
        row_bytes_by_layer = tuple(
            (
                int(key[0].numel()) * key.element_size(),
                int(value[0].numel()) * value.element_size(),
            )
            for key, value in zip(host_keys, host_values, strict=True)
        )
        full_layer_bytes = tuple(
            transfer_count * (key_bytes + value_bytes)
            for key_bytes, value_bytes in row_bytes_by_layer
        )
        if pending.layer_bytes and pending.layer_bytes != full_layer_bytes:
            raise RuntimeError("HiCache layer geometry changed during acquisition")
        pending.layer_bytes = full_layer_bytes
        mover_plan = self._host_mover_lease_plan(
            pending, row_bytes_by_layer, transfer_count
        )
        copy_runs = mover_plan.copy_runs
        sm_transfer_count = mover_plan.sm_row_count
        use_copy_engine = bool(copy_runs)
        use_sm_mover = sm_transfer_count != 0
        transfer_source_indices = mover_plan.sm_source_indices if use_sm_mover else None
        transfer_staging_indices = (
            mover_plan.sm_destination_indices if use_sm_mover else None
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
        copy_groups: list[tuple[StridedCopyGroup, StridedCopyGroup]] = []
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
            if (key_element_bytes, value_element_bytes) != row_bytes_by_layer[
                local_layer
            ]:
                raise RuntimeError("HiCache host and device KV row geometry disagrees")
            key_source_stride = host_key.stride(0) * host_key.element_size()
            value_source_stride = host_value.stride(0) * host_value.element_size()
            key_destination_stride = key_cache.stride(0) * key_cache.element_size()
            value_destination_stride = (
                value_cache.stride(0) * value_cache.element_size()
            )
            paired_copy &= (
                key_element_bytes == value_element_bytes
                and key_element_bytes in {128, 256, 512, 1024, 2048}
            )
            key_bytes = transfer_count * key_element_bytes
            value_bytes = transfer_count * value_element_bytes
            if max(key_bytes, value_bytes) > _MAX_ABI_BYTES:
                raise RuntimeError("HiCache layer transfer exceeds the NTA ABI limit")
            layer_geometry.append((key_bytes, value_bytes))
            if use_copy_engine:
                copy_groups.append(
                    (
                        StridedCopyGroup(
                            host_key.data_ptr(),
                            key_cache.data_ptr(),
                            int(host_key.shape[0]),
                            int(key_cache.shape[0]),
                            key_element_bytes,
                            key_source_stride,
                            key_destination_stride,
                        ),
                        StridedCopyGroup(
                            host_value.data_ptr(),
                            value_cache.data_ptr(),
                            int(host_value.shape[0]),
                            int(value_cache.shape[0]),
                            value_element_bytes,
                            value_source_stride,
                            value_destination_stride,
                        ),
                    )
                )
            object_id_base = _pipeline_object_id(
                pending.consumer_index, layer_count, local_layer
            )
            if use_sm_mover:
                if transfer_source_indices is None or transfer_staging_indices is None:
                    raise RuntimeError("SM mover has no exact remainder index map")
                transfer_objects.extend(
                    (
                        IndexedHostObject(
                            object_id_base,
                            _LOOKAHEAD_VERSION,
                            host_key.data_ptr(),
                            key_cache.data_ptr(),
                            transfer_source_indices.data_ptr(),
                            transfer_staging_indices.data_ptr(),
                            sm_transfer_count,
                            key_element_bytes,
                            key_source_stride,
                            key_destination_stride,
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
                            sm_transfer_count,
                            value_element_bytes,
                            value_source_stride,
                            value_destination_stride,
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
        copy_engine_wave_layers = self._frontier_layers_per_wave
        if use_copy_engine:
            operations_per_layer = 2 * len(copy_runs)
            if operations_per_layer > self._copy_engine_max_operations:
                raise RuntimeError(
                    "copy-engine index map exceeds the configured operation bound"
                )
            copy_engine_wave_layers = max(
                1,
                min(
                    self._frontier_layers_per_wave,
                    self._copy_engine_max_operations // operations_per_layer,
                ),
            )
        try:
            producer_stream = torch.cuda.current_stream()
            if use_sm_mover:
                first_object = 2 * first_local_layer
                last_object = 2 * last_local_layer
                self._runtime.register_indexed_host_objects(
                    transfer_first_slot + first_object,
                    transfer_objects[first_object:last_object],
                    stream=producer_stream,
                )
            phase_start = (
                pending.producer_event.start_event
                if not pending.prefetched_layers
                else torch.cuda.Event()
            )
            if phase_start is not pending.producer_event.start_event:
                pending.transfer_events += (phase_start,)
            phase_start.record(producer_stream)
            phase_program = self._transport_phase_program() if use_sm_mover else None
            hybrid_parallel = use_copy_engine and use_sm_mover
            copy_stream = (
                self._copy_stream if hybrid_parallel else self._prefetch_stream
            )
            with torch.cuda.stream(self._prefetch_stream):
                phase_start.wait(self._prefetch_stream)
                if profile_start is not None:
                    profile_start.record(self._prefetch_stream)
            if hybrid_parallel:
                with torch.cuda.stream(copy_stream):
                    phase_start.wait(copy_stream)
            local_layer = first_local_layer
            while local_layer < last_local_layer:
                wave_end = min(
                    last_local_layer,
                    local_layer
                    + (
                        copy_engine_wave_layers
                        if use_copy_engine
                        else self._frontier_layers_per_wave
                    ),
                )
                wave_bytes = sum(
                    key_bytes + value_bytes
                    for key_bytes, value_bytes in layer_geometry[local_layer:wave_end]
                )
                wave_row_bytes = sum(
                    (key_bytes + value_bytes) // transfer_count
                    for key_bytes, value_bytes in layer_geometry[local_layer:wave_end]
                )
                copy_wave_bytes = mover_plan.copy_row_count * wave_row_bytes
                sm_wave_bytes = sm_transfer_count * wave_row_bytes
                if copy_wave_bytes + sm_wave_bytes != wave_bytes:
                    raise RuntimeError("host mover byte partition is not exact")
                if use_sm_mover:
                    profile_sm = self._mover_profile_enabled("sm")
                    sm_profile_start = (
                        torch.cuda.Event(enable_timing=True) if profile_sm else None
                    )
                    sm_profile_finish = (
                        torch.cuda.Event(enable_timing=True) if profile_sm else None
                    )
                    with torch.cuda.stream(self._prefetch_stream):
                        if sm_profile_start is not None:
                            sm_profile_start.record(self._prefetch_stream)
                        if use_sm_mover and paired_copy:
                            if phase_program is None:
                                raise RuntimeError(
                                    "SM host mover has no transport program"
                                )
                            first_slot = transfer_first_slot + 2 * local_layer
                            phase_program.preload_host_pairs(
                                self._runtime,
                                first_slot,
                                wave_end - local_layer,
                                self._prefetch_stream,
                            )
                        else:
                            if phase_program is None:
                                raise RuntimeError(
                                    "SM host mover has no transport program"
                                )
                            first_slot = transfer_first_slot + 2 * local_layer
                            phase_program.preload_host(
                                self._runtime,
                                first_slot,
                                2 * (wave_end - local_layer),
                                self._prefetch_stream,
                            )
                        if sm_profile_finish is not None:
                            sm_profile_finish.record(self._prefetch_stream)
                    if sm_profile_start is not None and sm_profile_finish is not None:
                        self._mover_profiles.append(
                            _MoverProfile(
                                sm_profile_start,
                                sm_profile_finish,
                                "sm",
                                sm_wave_bytes,
                                1,
                                0,
                            )
                        )
                    self._stats["sm_mover_bytes"] += sm_wave_bytes
                copy_done: torch.cuda.Event | None = None
                if use_copy_engine:
                    wave_groups = tuple(
                        group
                        for groups in copy_groups[local_layer:wave_end]
                        for group in groups
                    )
                    # ``cudaMemcpy3DBatchAsync`` is one native submission, but
                    # its host issue work scales with the number of batch ops.
                    # The service model is calibrated per op, matching the
                    # candidate cost ``runs * K/V-layer groups``; using the
                    # submission count here overcharges a later full lease by
                    # orders of magnitude.
                    copy_operation_count = len(wave_groups) * len(copy_runs)
                    if copy_operation_count <= 0:
                        raise RuntimeError(
                            "copy-engine wave has no physical copy operations"
                        )
                    profile_copy = self._mover_profile_enabled("copy_engine")
                    copy_profile_start = (
                        torch.cuda.Event(enable_timing=True) if profile_copy else None
                    )
                    copy_profile_finish = (
                        torch.cuda.Event(enable_timing=True) if profile_copy else None
                    )
                    with torch.cuda.stream(copy_stream):
                        if copy_profile_start is not None:
                            copy_profile_start.record(copy_stream)
                        copy_issue_started = time.perf_counter_ns()
                        copy_submissions = copy_strided_host_runs_async(
                            wave_groups,
                            copy_runs,
                            copy_stream,
                        )
                        copy_issue_ns = time.perf_counter_ns() - copy_issue_started
                        if copy_profile_finish is not None:
                            copy_profile_finish.record(copy_stream)
                        if hybrid_parallel:
                            copy_done = torch.cuda.Event()
                            copy_done.record(copy_stream)
                    self._stats["copy_engine_waves"] += 1
                    self._stats["copy_engine_submissions"] += copy_submissions
                    self._stats["copy_engine_issue_cpu_ns"] += copy_issue_ns
                    self._stats["copy_engine_operations"] += copy_operation_count
                    self._stats["copy_engine_bytes"] += copy_wave_bytes
                    if (
                        copy_profile_start is not None
                        and copy_profile_finish is not None
                    ):
                        self._mover_profiles.append(
                            _MoverProfile(
                                copy_profile_start,
                                copy_profile_finish,
                                "copy_engine",
                                copy_wave_bytes,
                                copy_operation_count,
                                copy_issue_ns,
                            )
                        )
                with torch.cuda.stream(self._prefetch_stream):
                    if copy_done is not None:
                        pending.transfer_events += (copy_done,)
                        copy_done.wait(self._prefetch_stream)
                        self._stats["hybrid_parallel_waves"] += 1
                    if profile_finish is not None and wave_end == last_local_layer:
                        profile_finish.record(self._prefetch_stream)
                    for ready_layer in range(local_layer, wave_end):
                        key_bytes, value_bytes = layer_geometry[ready_layer]
                        ready_event = ready_events[ready_layer]
                        ready_event.record(self._prefetch_stream)
                        prefetched_layers[ready_layer] = _PrefetchedLayer(
                            key_bytes,
                            value_bytes,
                            ready_event,
                            (
                                None
                                if not use_sm_mover
                                else transfer_first_slot + 2 * ready_layer
                            ),
                        )
                    self._stats["lookahead_copy_waves"] = (
                        self._stats.get("lookahead_copy_waves", 0) + 1
                    )
                local_layer = wave_end
        except Exception:
            self._prefetch_stream.synchronize()
            self._copy_stream.synchronize()
            self._stats["hicache_fallback_batches"] += 1
            raise
        pending.prefetched_layers.update(prefetched_layers)
        frontier_geometry = layer_geometry[first_local_layer:last_local_layer]
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
        self._stats["lookahead_acquisition_objects"] += (
            2 * acquired_layer_count if use_sm_mover else 0
        )
        if paired_copy and use_sm_mover:
            self._stats["paired_lookahead_layers"] = (
                self._stats.get("paired_lookahead_layers", 0) + acquired_layer_count
            )
        if self._profile_cpu:
            self._stats["pipeline_cpu_ns"] = self._stats.get("pipeline_cpu_ns", 0) + (
                time.perf_counter_ns() - pipeline_started
            )

    def _prepare_missing_host_layers(
        self,
        pending: PendingHostLoad,
        *,
        exclude: frozenset[int] = frozenset(),
    ) -> int:
        """Enqueue every unpublished layer except explicit typed-demand gaps."""
        layer_count = int(pending.controller.layer_num)
        if any(layer < 0 or layer >= layer_count for layer in exclude):
            raise RuntimeError("typed-demand exclusion is outside the model")
        missing = [
            layer
            for layer in range(layer_count)
            if layer not in pending.prefetched_layers and layer not in exclude
        ]
        if not missing:
            return 0
        range_begin = missing[0]
        previous = missing[0]
        ranges: list[tuple[int, int]] = []
        for layer in missing[1:]:
            if layer != previous + 1:
                ranges.append((range_begin, previous + 1))
                range_begin = layer
            previous = layer
        ranges.append((range_begin, previous + 1))
        for first_layer, last_layer in ranges:
            self._prepare_host_pipeline(
                pending,
                first_local_layer=first_layer,
                last_local_layer=last_layer,
            )
        return len(missing)

    def _prepare_cross_layer_frontier(self, pending: PendingHostLoad) -> int:
        missing = [
            layer
            for layer in range(int(pending.controller.layer_num))
            if layer not in pending.prefetched_layers
        ]
        if not missing:
            raise RuntimeError(
                "incremental host execution has no unresolved demand layer"
            )
        demand_layer = missing[0]
        acquired_layers = self._prepare_missing_host_layers(
            pending, exclude=frozenset((demand_layer,))
        )
        self._stats["cross_layer_frontier_batches"] = (
            self._stats.get("cross_layer_frontier_batches", 0) + 1
        )
        self._stats["cross_layer_frontier_layers"] = (
            self._stats.get("cross_layer_frontier_layers", 0) + acquired_layers
        )
        self._stats["typed_demand_gap_layers"] = (
            self._stats.get("typed_demand_gap_layers", 0) + 1
        )
        return demand_layer

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
        scope_units = int(pending.controller.layer_num)
        plans = {
            self._execution_plan(
                schedule,
                page_pairs[wrapper_id],
                key_element_bytes,
                value_element_bytes,
                scope_units=scope_units,
            )
            for wrapper_id, schedule in schedules.items()
        }
        if len(plans) != 1:
            raise RuntimeError(
                "FlashInfer wrappers selected inconsistent host execution plans"
            )
        return plans.pop()

    def _typed_lease_execution_plan(
        self,
        schedules: dict[int, Schedule],
        work_dependencies: dict[int, tuple[LeaseWorkDependency | None, ...]],
        transfer_dependencies: dict[int, tuple[LeaseWorkDependency | None, ...]],
        pending: PendingHostLoad,
    ) -> HostExecutionPlan:
        """Plan one proactive lease from exact request dependency classes."""

        if (
            not schedules
            or set(work_dependencies) != set(schedules)
            or set(transfer_dependencies) != set(schedules)
        ):
            raise RuntimeError("typed lease execution has no work dependencies")
        controller = pending.controller
        host_keys = tuple(controller.mem_pool_host.k_data_refs)
        host_values = tuple(controller.mem_pool_host.v_data_refs)
        if not host_keys or len(host_keys) != len(host_values):
            raise RuntimeError("HiCache host pool has incomplete K/V geometry")
        key_row_bytes = int(host_keys[0][0].numel()) * host_keys[0].element_size()
        value_row_bytes = int(host_values[0][0].numel()) * host_values[0].element_size()
        lease_rows = int(pending.device_indices.numel())
        transfer_bytes = lease_rows * (key_row_bytes + value_row_bytes)
        scope_units = int(controller.layer_num)

        def plan(wrapper_id: int, schedule: Schedule) -> HostExecutionPlan:
            dependencies = work_dependencies[wrapper_id]
            transfer_groups = transfer_dependencies[wrapper_id]
            if (
                len(dependencies) != schedule.work_count
                or len(transfer_groups) != schedule.work_count
            ):
                raise RuntimeError("typed lease dependencies do not match CTA work")
            initial_runnable = sum(item is None for item in dependencies)
            if not any(item is not None for item in dependencies):
                raise RuntimeError("typed lease schedule has no external work")
            object_count = 2 * len(
                {item for item in transfer_groups if item is not None}
            )
            if object_count <= 0 or object_count > self._object_capacity:
                raise RuntimeError("typed lease transfer groups exceed object capacity")
            if self._execution_config.protocol.kind is ProtocolKind.CONVENTIONAL:
                transfer_ns = math.ceil(
                    transfer_bytes
                    * 1_000_000_000
                    / self._host_cost_model.bandwidth_bytes_per_second
                )
                compute_ns = schedule.work_count * self._host_cost_model.tile_compute_ns
                total_ns = scope_units * (transfer_ns + compute_ns)
                return HostExecutionPlan(
                    (object_count,),
                    total_ns,
                    total_ns,
                    False,
                    "conventional",
                    scope_units,
                )
            return plan_host_execution(
                object_count=object_count,
                transfer_bytes=transfer_bytes,
                runnable_tiles=schedule.work_count,
                initial_runnable_tiles=initial_runnable,
                model=self._host_cost_model,
                calibration_probe=self._incremental_calibration_probes_remaining > 0,
                scope_units=scope_units,
            )

        plans = {
            plan(wrapper_id, schedule) for wrapper_id, schedule in schedules.items()
        }
        if len(plans) != 1:
            raise RuntimeError(
                "FlashInfer wrappers selected inconsistent typed lease plans"
            )
        return plans.pop()

    def _prove_direct_metadata_execution(
        self,
        schedules: dict[int, Schedule],
        pending: PendingHostLoad,
        bindings: tuple[RequestBinding, ...],
    ) -> HostExecutionPlan | None:
        """Skip exact graph construction when an ideal-overlap bound rejects it.

        The pending lease is the authoritative byte set for the materialized
        SGLang path. Schedule extraction supplies only numerical work counts;
        no page table or source map is downloaded by this proof.
        """

        if not schedules or not bindings:
            raise RuntimeError("host execution proof has no active schedule")
        controller = pending.controller
        host_keys = tuple(controller.mem_pool_host.k_data_refs)
        host_values = tuple(controller.mem_pool_host.v_data_refs)
        if not host_keys or len(host_keys) != len(host_values):
            raise RuntimeError("HiCache host pool has incomplete K/V geometry")
        transfer_count = int(pending.host_indices.numel())
        if transfer_count <= 0 or transfer_count != int(pending.device_indices.numel()):
            raise RuntimeError("HiCache host proof has no promoted pages")
        key_row_bytes = int(host_keys[0][0].numel()) * host_keys[0].element_size()
        value_row_bytes = int(host_values[0][0].numel()) * host_values[0].element_size()
        transfer_bytes = transfer_count * (key_row_bytes + value_row_bytes)
        object_count = 2 * len(bindings)
        proofs = tuple(
            prove_atomic_host_execution(
                object_count=object_count,
                transfer_bytes=transfer_bytes,
                runnable_tiles=schedule.work_count,
                model=self._host_cost_model,
                scope_units=int(pending.controller.layer_num),
            )
            for schedule in schedules.values()
        )
        if any(proof is None for proof in proofs):
            return None
        return max(
            (proof for proof in proofs if proof is not None),
            key=lambda proof: proof.predicted_atomic_ns,
        )

    def _execution_plan(
        self,
        schedule: Schedule,
        pairs: tuple[_PagePair, ...],
        key_element_bytes: int,
        value_element_bytes: int,
        *,
        scope_units: int = 1,
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
            total_ns = scope_units * (transfer_ns + compute_ns)
            return HostExecutionPlan(
                (2 * len(unique_pairs),),
                total_ns,
                total_ns,
                False,
                "conventional",
                scope_units,
            )
        return plan_host_execution(
            object_count=2 * len(unique_pairs),
            transfer_bytes=transfer_bytes,
            runnable_tiles=schedule.work_count,
            model=self._host_cost_model,
            initial_runnable_tiles=sum(not pair[0] for pair in pairs),
            calibration_probe=self._incremental_calibration_probes_remaining > 0,
            scope_units=scope_units,
        )

    def _run_bulk_host_layer(
        self,
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        output: torch.Tensor,
        layer: Any,
        stream: torch.cuda.Stream,
        *,
        causal: bool,
        window_left: int,
        run_options: dict[str, Any],
    ) -> torch.Tensor:
        batch = self._active_batch
        if batch is None or batch.pending_host_load is None:
            raise RuntimeError("bulk host execution has no active HiCache load")
        pending = batch.pending_host_load
        local_layer = int(layer.layer_id) - int(
            getattr(pending.controller.mem_pool_device, "start_layer", 0)
        )
        prefetched = pending.prefetched_layers.get(local_layer)
        if prefetched is None:
            (
                _external_plan,
                schedule,
                object_count,
                preloaded_event,
                preloaded_object_count,
                host_execution,
            ) = self._upload_plan(wrapper, int(layer.layer_id), kv_cache)
            if (
                preloaded_event is not None
                or preloaded_object_count != 0
                or object_count <= 0
                or host_execution is None
                or host_execution.rounds != 1
                or host_execution.overlap_initial
            ):
                raise RuntimeError("direct host staging received a non-direct plan")
            allocation = self._plans[(id(wrapper), -1)]
            # The runtime directory stores raw index pointers. PyTorch cannot
            # infer that the transport kernel consumes them, so retain the
            # tensors explicitly through the batch lifetime before replacing
            # the external work plan with its event-complete direct form.
            retained_indices = allocation.index_tensors
            pending.prefetch_tensors += retained_indices
            batch.prefetch_tensors += retained_indices
            self._transport_phase_program().preload_host(
                self._runtime, 0, object_count, stream
            )
            ready_event = torch.cuda.Event(enable_timing=self._profile_barrier)
            ready_event.record(stream)
            unique_pairs = {pair for pair in batch.page_pairs[id(wrapper)] if pair[0]}
            if object_count != 2 * len(unique_pairs):
                raise RuntimeError("direct host object pairing is not exact")
            row_count = sum(len(pair[0]) for pair in unique_pairs)
            key_cache, value_cache = kv_cache
            key_bytes = row_count * key_cache[0].numel() * key_cache.element_size()
            value_bytes = (
                row_count * value_cache[0].numel() * value_cache.element_size()
            )
            if key_bytes + value_bytes != allocation.transfer_bytes:
                raise RuntimeError("direct host staging byte accounting is not exact")
            prefetched = _PrefetchedLayer(key_bytes, value_bytes, ready_event, 0)
            pending.prefetched_layers[local_layer] = prefetched
            batch.prefetched_layers[local_layer] = prefetched
            self._stats["direct_staging_launches"] = (
                self._stats.get("direct_staging_launches", 0) + 1
            )
            self._stats["direct_staging_bytes"] = (
                self._stats.get("direct_staging_bytes", 0) + key_bytes + value_bytes
            )
            if id(wrapper) not in self._stock_wrapper_for_typed:
                # The first upload described the unresolved acquisition. Once
                # its stream event owns completion, republish the same exact
                # semantic work with direct requirements. Keeping the external
                # dependency array and merely changing launch flags would blur
                # the ownership transition that the preacquired contract is
                # designed to enforce.
                self._upload_plan(wrapper, int(layer.layer_id), kv_cache)
        if prefetched is None:  # pragma: no cover - established above
            raise RuntimeError("direct host staging produced no completion event")
        if self._profile_barrier:
            arrive = torch.cuda.Event(enable_timing=True)
            arrive.record(stream)
            self._barrier_profiles.append(
                (arrive, prefetched.ready_event, int(layer.layer_id))
            )
        stream.wait_event(prefetched.ready_event)
        if id(wrapper) in self._stock_wrapper_for_typed:
            result = self._run_ready_stock_numerical(
                wrapper,
                q,
                kv_cache,
                layer,
                causal=causal,
                window_left=window_left,
            )
            self._stats["stock_attention_launches"] += 1
            self._stats["stock_prefetched_external_attention_launches"] += 1
        else:
            self._run_preacquired_attention(
                wrapper, q, kv_cache, output, layer, run_options
            )
            result = output
            self._stats["typed_bulk_attention_launches"] = (
                self._stats.get("typed_bulk_attention_launches", 0) + 1
            )
        self._bulk_events = (prefetched.ready_event,)
        return result

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
            if allocation.plan.has_external:
                raise RuntimeError(
                    "event-complete attention retained transport dependencies"
                )
        elif "request_bound" not in module_name:
            schedule = batch.schedules.get(id(wrapper))
            topology = batch.work_topologies.get(id(wrapper))
            if schedule is None or topology is None:
                raise RuntimeError(
                    "resident demand attention has no exact CTA topology"
                )
            self._upload_resident_plan(wrapper, schedule, topology)
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
                PREACQUIRED_LAUNCH_FLAGS,
                out=output,
                **run_options,
            )
            allocation.plan.mark_consumed(torch.cuda.current_stream())
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
        plan = DeviceWorkPlan(
            capacity,
            self._max_dependencies_per_work_ticket * capacity,
            self._runtime.device_ordinal,
        )
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
        self._stats["predicted_atomic_ns"] += (
            host_execution.predicted_atomic_per_unit_ns
        )
        self._stats["predicted_incremental_ns"] += (
            host_execution.predicted_incremental_per_unit_ns
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
        *,
        arriving_prefetch: bool = False,
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
        if self._tier_service.is_cxl:
            raise RuntimeError(
                "CXL-DAX attention is deferred until acquired pages are "
                "materialized into the numerical KV view"
            )
        controller = pending.controller
        device_pool = controller.mem_pool_device
        local_layer = layer_id - int(getattr(device_pool, "start_layer", 0))
        if local_layer < 0 or local_layer >= int(controller.layer_num):
            raise RuntimeError(f"SGLang layer {layer_id} is outside the HiCache pool")
        prefetched = batch.prefetched_layers.get(local_layer)
        if arriving_prefetch and (
            not self._tier_service.is_host_staged
            or prefetched is None
            or prefetched.transfer_first_slot is None
        ):
            raise RuntimeError(
                "arriving host work requires a directory-backed proactive layer"
            )
        if not self._tier_service.is_host_staged and prefetched is not None:
            raise RuntimeError(
                "physical tiers cannot consume a host-prefetched HiCache layer"
            )
        page_pairs = batch.page_pairs.get(id(wrapper))
        if page_pairs is None:
            work_dependencies = batch.work_dependencies.get(id(wrapper))
            transfer_dependencies = batch.transfer_dependencies.get(id(wrapper))
            if work_dependencies is None or transfer_dependencies is None:
                raise RuntimeError("NTA attention plan omitted dependency geometry")
            if (
                len(work_dependencies) != schedule.work_count
                or len(transfer_dependencies) != schedule.work_count
            ):
                raise RuntimeError("NTA dependency geometry does not match CTA work")
            if not any(item is not None for item in work_dependencies):
                raise RuntimeError("typed acquisition plan has no external work")
            work_dependency_rows = tuple(
                0 if item is None else item.row_count for item in work_dependencies
            )
            dependency_geometry: Any = (
                "typed_lease",
                work_dependencies,
                transfer_dependencies,
                batch.lease_transfer_rows,
                pending.lease_id,
            )
        else:
            if len(page_pairs) != schedule.work_count:
                raise RuntimeError("NTA attention page dependencies are misaligned")
            work_dependencies = (None,) * schedule.work_count
            transfer_dependencies = (None,) * schedule.work_count
            work_dependency_rows = tuple(len(pair[0]) for pair in page_pairs)
            dependency_geometry = page_pairs
        signature = _plan_cache_signature(
            schedule.request_indices,
            schedule.kv_tile_indices,
            dependency_geometry,
            tuple(binding.request_slot for binding in batch.bindings),
            key_bytes,
            value_bytes,
            None
            if prefetched is None
            else (
                prefetched.key_bytes,
                prefetched.value_bytes,
                prefetched.transfer_first_slot if arriving_prefetch else -1,
            ),
        )
        signature = signature + (
            self._tier_service.tier.value,
            self._tier_service.catalog_digest,
            layer_id if not self._tier_service.is_host_staged else None,
        )
        # Work/ticket topology is layer invariant. Layer K/V addresses are
        # republished through the object directory on the consumer stream.
        plan = self._ensure_plan(wrapper, -1, schedule)
        allocation = self._plans[(id(wrapper), -1)]
        rebuild_plan = allocation.signature != signature
        if prefetched is not None and not rebuild_plan:
            if allocation.host_execution is None or allocation.object_count != 0:
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

        # A HiCache load is a compulsory payload acquisition even when its
        # structural work plan is unchanged.  Reusing the peer mapping is safe;
        # reusing ObjectState::Ready is not, because SGLang may have recycled
        # the same destination slot since the preceding load.
        if prefetched is None and (rebuild_plan or self._tier_service.is_nvme):
            allocation.object_version = (allocation.object_version + 1) & 0xFFFFFFFF
            allocation.object_version = allocation.object_version or 1
        # Object versions belong to directory-backed acquisitions. An
        # event-complete prefetch deliberately emits only direct requirements,
        # so coupling its state record to an otherwise-unused object version
        # creates a false ownership edge.
        version = allocation.object_version
        indexed_objects: list[IndexedHostObject] = []
        index_tensors: list[torch.Tensor] = []
        physical_object_bytes: list[int] = []
        physical_publications: list[tuple[int, int]] = []
        pair_objects: dict[_PagePair, tuple[tuple[int, int, int, int], ...]] = {}
        physical_runs: dict[_PagePair, tuple[ContiguousPairRun, ...]] = {}
        physical_extents: dict[ContiguousPairRun, tuple[Any, Any]] = {}
        lease_index_map = None
        indexed_host_plan = None
        if page_pairs is None and prefetched is None:
            if not self._tier_service.is_host_staged:
                raise RuntimeError(
                    "typed lease indices require the host-staged resource contract"
                )
            index_map_started = time.perf_counter_ns() if self._profile_cpu else 0
            lease_index_map = pending.materialize_device_index_map()
            if self._profile_cpu:
                self._stats["plan_index_map_cpu_ns"] = self._stats.get(
                    "plan_index_map_cpu_ns", 0
                ) + (time.perf_counter_ns() - index_map_started)
            if int(lease_index_map.source_indices.numel()) != batch.lease_transfer_rows:
                raise RuntimeError(
                    "typed lease index map changed after metadata binding"
                )
            index_tensors.extend(lease_index_map.retained_tensors)
            indexed_materialization_started = (
                time.perf_counter_ns() if self._profile_cpu else 0
            )
            indexed_topology = lease_indexed_transfer_topology(
                work_dependencies,
                transfer_dependencies,
                lease_index_map.operations,
                index_count=int(lease_index_map.source_indices.numel()),
            )
            indexed_host_plan = IndexedHostPlan(
                indexed_topology,
                (
                    IndexedTensorLane(
                        host_key.data_ptr(),
                        key_cache.data_ptr(),
                        key_element_bytes,
                        host_key.stride(0) * host_key.element_size(),
                        key_cache.stride(0) * key_cache.element_size(),
                        int(host_key.shape[0]),
                        int(key_cache.shape[0]),
                    ),
                    IndexedTensorLane(
                        host_value.data_ptr(),
                        value_cache.data_ptr(),
                        value_element_bytes,
                        host_value.stride(0) * host_value.element_size(),
                        value_cache.stride(0) * value_cache.element_size(),
                        int(host_value.shape[0]),
                        int(value_cache.shape[0]),
                    ),
                ),
                source_indices_device_address=lease_index_map.source_indices.data_ptr(),
                staging_indices_device_address=(
                    lease_index_map.destination_indices.data_ptr()
                ),
                object_version=version,
                direct_base=self._runtime.device_view,
                object_id_base=_OBJECT_ID_BASE,
            )
            if self._profile_cpu:
                self._stats["plan_indexed_materialization_cpu_ns"] = (
                    self._stats.get("plan_indexed_materialization_cpu_ns", 0)
                    + time.perf_counter_ns()
                    - indexed_materialization_started
                )
        if self._tier_service.is_nvme and prefetched is None:
            if page_pairs is None:
                raise RuntimeError(
                    "NVMe acquisition requires exact physical page pairs"
                )
            unique_physical_pairs = tuple(
                dict.fromkeys(pair for pair in page_pairs if pair[0])
            )
            key_stride_bytes = key_cache.stride(0) * key_cache.element_size()
            value_stride_bytes = value_cache.stride(0) * value_cache.element_size()
            if (
                key_stride_bytes != key_element_bytes
                or value_stride_bytes != value_element_bytes
            ):
                raise RuntimeError(
                    "direct NVMe numerical destinations require contiguous KV rows"
                )
            lba_size = self._tier_service.nvme_lba_size
            max_transfer_bytes = self._tier_service.nvme_max_transfer_bytes
            rows_per_lba = math.lcm(
                lba_size // math.gcd(lba_size, key_element_bytes),
                lba_size // math.gcd(lba_size, value_element_bytes),
            )
            maximum_rows = min(
                max_transfer_bytes // key_element_bytes,
                max_transfer_bytes // value_element_bytes,
            )
            maximum_rows -= maximum_rows % rows_per_lba
            if maximum_rows <= 0:
                raise RuntimeError(
                    "one LBA-aligned KV run exceeds the NVMe transfer limit"
                )
            for pair in unique_physical_pairs:
                source_ordinals, device_pages = pair
                if len(source_ordinals) != len(device_pages):
                    raise RuntimeError("physical HiCache page mappings disagree")
                layout = analyze_index_pairs(source_ordinals, device_pages)
                runs: list[ContiguousPairRun] = []
                for contiguous in layout.runs:
                    if contiguous.row_count % rows_per_lba:
                        raise RuntimeError(
                            "physical KV run is not exactly LBA materializable"
                        )
                    consumed = 0
                    while consumed < contiguous.row_count:
                        row_count = min(maximum_rows, contiguous.row_count - consumed)
                        runs.append(
                            ContiguousPairRun(
                                contiguous.source_first + consumed,
                                contiguous.destination_first + consumed,
                                row_count,
                            )
                        )
                        consumed += row_count
                physical_runs[pair] = tuple(runs)
            unique_physical_runs = tuple(
                dict.fromkeys(run for runs in physical_runs.values() for run in runs)
            )
            if 2 * len(unique_physical_runs) > self._object_capacity:
                raise RuntimeError(
                    "NVMe layer needs more HBM object slots than the runtime capacity"
                )
            # Resolve the complete catalog before installing any HBM object.
            # This keeps catalog/geometry errors transactional from the
            # engine's perspective and avoids partially publishing a layer.
            for run in unique_physical_runs:
                ordinals = tuple(
                    range(run.source_first, run.source_first + run.row_count)
                )
                if run.destination_first + run.row_count > min(
                    int(key_cache.shape[0]), int(value_cache.shape[0])
                ):
                    raise RuntimeError(
                        "NVMe destination run exceeds the numerical KV cache"
                    )
                physical_extents[run] = (
                    self._tier_service.extent(
                        layer_id, ordinals, "key", key_element_bytes
                    ),
                    self._tier_service.extent(
                        layer_id, ordinals, "value", value_element_bytes
                    ),
                )

        def objects_for(
            pair: _PagePair,
        ) -> tuple[tuple[int, int, int, int], ...]:
            existing = pair_objects.get(pair)
            if existing is not None:
                return existing
            host_pages, device_pages = pair
            if self._tier_service.is_nvme:
                runs = physical_runs.get(pair)
                if not runs:
                    raise RuntimeError("NVMe page pair was not run-validated")
                install_stream = torch.cuda.current_stream()
                result: list[tuple[int, int, int, int]] = []
                for run in runs:
                    extents = physical_extents.get(run)
                    if extents is None:
                        raise RuntimeError("NVMe page run was not catalog-validated")
                    key_extent, value_extent = extents
                    key_slot = len(physical_object_bytes)
                    key_object_id = _OBJECT_ID_BASE | key_slot
                    key_previous = self._nvme_slots.previous(key_slot)
                    key_quiescence = self._nvme_slots.prior_consumer_event(key_slot)
                    key_destination = (
                        key_cache.data_ptr()
                        + run.destination_first
                        * key_cache.stride(0)
                        * key_cache.element_size()
                    )
                    key_region = self._nvme_regions.get((layer_id, "key"))
                    if key_region is None:
                        raise RuntimeError("NVMe K allocation was not pre-registered")
                    key_address = self._runtime.install_registered_nvme_object_async(
                        key_slot,
                        key_object_id,
                        version,
                        key_extent.offset,
                        key_extent.bytes,
                        key_region,
                        key_destination,
                        install_stream,
                        key_quiescence,
                    )
                    if key_address != key_destination:
                        raise RuntimeError(
                            "NVMe K destination does not alias the numerical cache"
                        )
                    self._stats["nvme_view_publications"] += 1
                    if key_previous is None:
                        self._stats["nvme_fresh_slot_installs"] += 1
                    elif key_previous == key_address:
                        self._stats["nvme_same_destination_installs"] += 1
                    else:
                        self._stats["nvme_destination_rebinds"] += 1
                    if key_previous is not None:
                        self._stats["nvme_object_quiesced_replacements"] += 1
                    physical_publications.append((key_slot, key_address))
                    physical_object_bytes.append(key_extent.bytes)

                    value_slot = len(physical_object_bytes)
                    value_object_id = _OBJECT_ID_BASE | value_slot
                    value_previous = self._nvme_slots.previous(value_slot)
                    value_quiescence = self._nvme_slots.prior_consumer_event(value_slot)
                    value_destination = (
                        value_cache.data_ptr()
                        + run.destination_first
                        * value_cache.stride(0)
                        * value_cache.element_size()
                    )
                    value_region = self._nvme_regions.get((layer_id, "value"))
                    if value_region is None:
                        raise RuntimeError("NVMe V allocation was not pre-registered")
                    value_address = self._runtime.install_registered_nvme_object_async(
                        value_slot,
                        value_object_id,
                        version,
                        value_extent.offset,
                        value_extent.bytes,
                        value_region,
                        value_destination,
                        install_stream,
                        value_quiescence,
                    )
                    if value_address != value_destination:
                        raise RuntimeError(
                            "NVMe V destination does not alias the numerical cache"
                        )
                    self._stats["nvme_view_publications"] += 1
                    if value_previous is None:
                        self._stats["nvme_fresh_slot_installs"] += 1
                    elif value_previous == value_address:
                        self._stats["nvme_same_destination_installs"] += 1
                    else:
                        self._stats["nvme_destination_rebinds"] += 1
                    if value_previous is not None:
                        self._stats["nvme_object_quiesced_replacements"] += 1
                    physical_publications.append((value_slot, value_address))
                    physical_object_bytes.append(value_extent.bytes)
                    result.append(
                        (key_slot, key_object_id, value_slot, value_object_id)
                    )
                normalized = tuple(result)
                if not normalized:
                    raise RuntimeError("NVMe page pair produced no exact objects")
                pair_objects[pair] = normalized
                self._stats["nvme_numerical_alias_objects"] = self._stats.get(
                    "nvme_numerical_alias_objects", 0
                ) + 2 * len(normalized)
                self._stats["nvme_numerical_alias_bytes"] = self._stats.get(
                    "nvme_numerical_alias_bytes", 0
                ) + sum(
                    physical_extents[run][0].bytes + physical_extents[run][1].bytes
                    for run in runs
                )
                return normalized
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
            normalized = (result,)
            pair_objects[pair] = normalized
            return normalized

        dependency_build_started = time.perf_counter_ns() if self._profile_cpu else 0
        topology = batch.work_topologies.get(id(wrapper))
        if topology is None:
            raise RuntimeError("native work-plan upload has no exact topology")
        if indexed_host_plan is None:
            dependency_spans: list[WorkDependencySpan] = []
            dependencies: Any = []
            object_fanout: Counter[int] = Counter()
            unresolved_dependencies: list[int] = []
            direct_work_count = 0
            external_object_slots: list[tuple[int, ...]] = []
        else:
            dependency_spans = list(indexed_host_plan.dependency_spans)
            dependencies = indexed_host_plan.dependencies
            object_fanout = Counter(
                {0: indexed_host_plan.max_object_fanout}
            )
            unresolved_dependencies = [
                indexed_host_plan.min_unresolved_dependencies
            ]
            direct_work_count = indexed_host_plan.direct_work_count
            external_object_slots = list(
                indexed_host_plan.external_object_slots
            )
        physical_work_pairs = (
            page_pairs
            if page_pairs is not None
            else tuple(((), ()) for _ in range(schedule.work_count))
        )
        work_entries = () if indexed_host_plan is not None else enumerate(
            zip(
                schedule.request_indices,
                schedule.kv_tile_indices,
                physical_work_pairs,
                work_dependencies,
                transfer_dependencies,
                work_dependency_rows,
                strict=True,
            )
        )
        for work_ticket, (
            request_index,
            kv_tile,
            pair,
            dependency,
            transfer_dependency,
            external_rows,
        ) in work_entries:
            dependency_begin = len(dependencies)
            if external_rows > 0:
                if prefetched is not None:
                    if arriving_prefetch:
                        key_slot = prefetched.transfer_first_slot
                        if key_slot is None:  # pragma: no cover - validated above
                            raise RuntimeError(
                                "arriving host layer lost its directory slot"
                            )
                        value_slot = key_slot + 1
                        key_object_id = _pipeline_object_id(
                            pending.consumer_index,
                            int(pending.controller.layer_num),
                            local_layer,
                        )
                        value_object_id = key_object_id + 1
                        dependencies.extend(
                            (
                                object_requirement(
                                    object_slot=key_slot,
                                    object_id=key_object_id,
                                    object_version=_LOOKAHEAD_VERSION,
                                    bytes=external_rows * key_element_bytes,
                                ),
                                object_requirement(
                                    object_slot=value_slot,
                                    object_id=value_object_id,
                                    object_version=_LOOKAHEAD_VERSION,
                                    bytes=external_rows * value_element_bytes,
                                ),
                            )
                        )
                        object_fanout[key_slot] += 1
                        object_fanout[value_slot] += 1
                        unresolved_dependencies.append(2)
                        external_object_slots.append((key_slot, value_slot))
                        direct_dependencies = 0
                    else:
                        # A completed per-layer event is the sole completion
                        # edge; the numerical plan is therefore all-direct.
                        dependencies.extend(
                            (
                                direct_requirement(self._runtime.device_view, 1),
                                direct_requirement(self._runtime.device_view, 1),
                            )
                        )
                        direct_work_count += 1
                        external_object_slots.append(())
                        direct_dependencies = 2
                else:
                    if dependency is not None:
                        raise RuntimeError(
                            "typed lease work bypassed its compact indexed plan"
                        )
                    if not pair[0]:
                        raise RuntimeError(
                            "unpublished acquisition work omitted physical pages"
                        )
                    object_groups = objects_for(pair)
                    work_slots: list[int] = []
                    for group_index, (
                        key_slot,
                        key_object_id,
                        value_slot,
                        value_object_id,
                    ) in enumerate(object_groups):
                        if self._tier_service.is_nvme:
                            run = physical_runs[pair][group_index]
                            key_transfer_bytes = run.row_count * key_element_bytes
                            value_transfer_bytes = run.row_count * value_element_bytes
                        else:
                            key_transfer_bytes = external_rows * key_element_bytes
                            value_transfer_bytes = external_rows * value_element_bytes
                        dependencies.extend(
                            (
                                object_requirement(
                                    object_slot=key_slot,
                                    object_id=key_object_id,
                                    object_version=version,
                                    bytes=key_transfer_bytes,
                                ),
                                object_requirement(
                                    object_slot=value_slot,
                                    object_id=value_object_id,
                                    object_version=version,
                                    bytes=value_transfer_bytes,
                                ),
                            )
                        )
                        object_fanout[key_slot] += 1
                        object_fanout[value_slot] += 1
                        work_slots.extend((key_slot, value_slot))
                    unresolved_count = 2 * len(object_groups)
                    if unresolved_count > self._max_dependencies_per_work_ticket:
                        raise RuntimeError(
                            "NVMe run fragmentation exceeds the configured "
                            "per-work dependency capacity"
                        )
                    unresolved_dependencies.append(unresolved_count)
                    external_object_slots.append(tuple(work_slots))
                    direct_dependencies = 0
            else:
                if transfer_dependency is not None:
                    raise RuntimeError("direct work retained a transfer dependency")
                dependencies.extend(
                    (
                        direct_requirement(self._runtime.device_view, 1),
                        direct_requirement(self._runtime.device_view, 1),
                    )
                )
                direct_work_count += 1
                external_object_slots.append(())
                direct_dependencies = 2
            dependency_count = len(dependencies) - dependency_begin
            dependency_spans.append(
                WorkDependencySpan(
                    dependency_begin,
                    dependency_count,
                    direct_dependencies,
                )
            )

        if self._profile_cpu:
            self._stats["plan_dependency_build_cpu_ns"] = self._stats.get(
                "plan_dependency_build_cpu_ns", 0
            ) + (time.perf_counter_ns() - dependency_build_started)

        if prefetched is not None:
            object_count = 0
        elif indexed_host_plan is not None:
            object_count = indexed_host_plan.object_count
        elif self._tier_service.is_host_staged:
            object_count = len(indexed_objects)
        else:
            object_count = len(physical_object_bytes)
        if (
            object_count == 0
            and self._tier_service.is_host_staged
            and prefetched is None
        ):
            raise RuntimeError("external HiCache batch has no CTA dependency")
        if object_count > self._object_capacity:
            raise RuntimeError(
                f"HiCache layer needs {object_count} objects; configured capacity is "
                f"{self._object_capacity}"
            )
        if self._tier_service.is_nvme and physical_object_bytes:
            # Consume one predecessor proof only after every view publication
            # succeeded; the next proof is recorded after native attention.
            self._nvme_slots.commit(tuple(physical_publications))
        if prefetched is None and pending.prefetched_layers:
            # SM-driven proactive copies own high directory slots while the
            # current demand path owns the low range. Copy-engine layers have
            # no directory allocation and therefore need no overlap check.
            pipeline_slots = tuple(
                layer.transfer_first_slot
                for layer in pending.prefetched_layers.values()
                if layer.transfer_first_slot is not None
            )
            if pipeline_slots and object_count > min(pipeline_slots):
                raise RuntimeError("demand and proactive HiCache object ranges overlap")
        if prefetched is not None:
            transfer_bytes = prefetched.key_bytes + prefetched.value_bytes
        elif indexed_host_plan is not None:
            transfer_bytes = indexed_host_plan.transfer_bytes
        elif self._tier_service.is_host_staged:
            transfer_bytes = sum(
                object_.index_count * object_.element_bytes
                for object_ in indexed_objects
            )
        else:
            transfer_bytes = sum(physical_object_bytes)
        if self._tier_service.is_host_staged:
            if batch.host_execution is not None:
                # Cost decisions are immutable for one forward epoch. Reusing
                # the metadata-time plan prevents transfer/setup observations
                # from changing a later layer's execution form mid-batch. An
                # event-complete prefetch likewise remains causally attached to
                # the exact transfer decision that produced it.
                host_execution = batch.host_execution
            else:
                host_execution = plan_host_execution(
                    # Cost planning describes the K/V mover pair even though an
                    # event-complete prefetch contributes no acquisition objects
                    # to the numerical work plan.
                    object_count=2 if prefetched is not None else object_count,
                    transfer_bytes=transfer_bytes,
                    runnable_tiles=schedule.work_count,
                    initial_runnable_tiles=(
                        direct_work_count
                        if self._overlap_enabled and prefetched is None
                        else 0
                    ),
                    model=self._host_cost_model,
                )
        else:
            host_execution = None
        stream = torch.cuda.current_stream()
        if self._tier_service.is_host_staged and prefetched is None:
            registration_started = time.perf_counter_ns() if self._profile_cpu else 0
            quiescence_event = (
                self._indexed_object_quiescence_event
                if self._indexed_object_quiescence_recorded
                else None
            )
            if quiescence_event is None:
                self._stats["indexed_object_lifetime_guard_fallbacks"] += 1
            else:
                self._stats["indexed_object_quiesced_registrations"] += 1
            index_binding = (
                None
                if lease_index_map is None
                else IndexedHostIndexBinding(
                    lease_index_map.source_indices.data_ptr(),
                    lease_index_map.destination_indices.data_ptr(),
                    int(lease_index_map.source_indices.numel()),
                )
            )
            if indexed_host_plan is None:
                self._runtime.register_indexed_host_objects(
                    0,
                    indexed_objects,
                    stream=stream,
                    quiescence_event=quiescence_event,
                    index_binding=index_binding,
                )
            else:
                self._runtime.register_indexed_host_plan(
                    indexed_host_plan,
                    stream=stream,
                    quiescence_event=quiescence_event,
                    index_binding=index_binding,
                )
            if self._profile_cpu:
                published_ns = time.perf_counter_ns()
                self._stats["plan_directory_publish_cpu_ns"] = self._stats.get(
                    "plan_directory_publish_cpu_ns", 0
                ) + (published_ns - registration_started)
                validation_started = published_ns
            # The token is single-use: a new token is recorded only after the
            # just-published directory has completed its consumer forward.
            self._indexed_object_quiescence_recorded = False
            self._phase_program(wrapper).validate_indexed_host_range(
                self._runtime, 0, object_count, stream
            )
            if self._profile_cpu:
                validated_ns = time.perf_counter_ns()
                self._stats["plan_index_validation_cpu_ns"] = self._stats.get(
                    "plan_index_validation_cpu_ns", 0
                ) + (validated_ns - validation_started)
                self._stats["plan_registration_cpu_ns"] = self._stats.get(
                    "plan_registration_cpu_ns", 0
                ) + (validated_ns - registration_started)
        incremental = self._tier_service.is_host_staged and (
            host_execution.rounds > 1 or host_execution.overlap_initial
        )
        needs_plan = (
            not self._tier_service.is_host_staged
            or prefetched is not None
            or incremental
            or id(wrapper) not in self._stock_wrapper_for_typed
        )
        if needs_plan and rebuild_plan:
            upload_started = time.perf_counter_ns() if self._profile_cpu else 0
            plan.upload_exact(
                topology,
                dependency_spans,
                dependencies,
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
        allocation.exact_resume_windows = (
            self._tier_service.is_host_staged
            and prefetched is None
            and page_pairs is None
            and direct_work_count == 0
            and object_count == 2 * schedule.work_count
            and all(
                object_slots == (2 * work_ticket, 2 * work_ticket + 1)
                for work_ticket, object_slots in enumerate(external_object_slots)
            )
        )
        if self._tier_service.is_host_staged:
            if prefetched is None:
                self._record_demand_plan_stats(
                    batch,
                    schedule,
                    object_count,
                    transfer_bytes,
                    host_execution,
                )
            elif arriving_prefetch:
                self._record_demand_plan_stats(
                    batch,
                    schedule,
                    2,
                    transfer_bytes,
                    host_execution,
                )
                self._stats["arriving_prefetch_layers"] = (
                    self._stats.get("arriving_prefetch_layers", 0) + 1
                )
        elif self._tier_service.is_nvme:
            self._stats["cta_work_items"] += schedule.work_count
            self._stats["nvme_bytes"] += transfer_bytes
            self._stats["nvme_epochs"] += 1
        else:
            raise RuntimeError("external plan selected an unsupported serving tier")
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
        self,
        wrapper: Any,
        schedule: Schedule,
        pending: PendingHostLoad,
        *,
        layout: tuple[list[int], list[int], list[int], int] | None = None,
    ) -> tuple[_PagePair, ...]:
        indptr, pages, last_page, page_size = (
            self._wrapper_layout(wrapper) if layout is None else layout
        )
        if self._tier_service.is_host_staged:
            source_by_device = pending.materialize_mapping()
        else:
            catalog = self._tier_service.catalog
            if catalog is None:
                raise RuntimeError("physical HiCache load has no stable-key catalog")
            source_by_device = pending.materialize_storage_mapping(catalog)
        return _page_pairs_for_schedule(
            schedule,
            indptr=indptr,
            pages=pages,
            last_page=last_page,
            page_size=page_size,
            source_by_device=source_by_device,
        )

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

    def _transport_phase_program(self) -> JitPhaseProgram:
        """Return the runtime-owned, operator-independent transport phases."""
        if self._transport_program is not None:
            return self._transport_program
        configured = os.environ.get("NTA_TRANSPORT_PROGRAM")
        if not configured:
            raise RuntimeError(
                "NTA_TRANSPORT_PROGRAM is missing; activate the complete NTA "
                "runtime environment before constructing the SGLang backend"
            )
        path = pathlib.Path(configured).resolve()
        if not path.is_file():
            raise RuntimeError(f"NTA transport phase program does not exist: {path}")
        program = JitPhaseProgram(path)
        try:
            program.operator_contract.require(
                family=OperatorFamily.GENERIC,
                form=OperatorForm.INCREMENTAL,
                capabilities=(
                    OperatorCapability.OBJECT_DEPENDENCIES
                    | OperatorCapability.FINITE_DEFERRAL
                    | OperatorCapability.PARTIAL_PUBLICATION
                    | OperatorCapability.GRAPH_REPLAY
                ),
                instrumentation=OperatorInstrumentation.TIER_OWNERSHIP,
                identity_binding=OperatorIdentityBinding.NONE,
                demand_binding=OperatorDemandBinding.NONE,
                access_proof=OperatorAccessProof.NONE,
                tier_mask=(1 << 6) - 1,
            )
            program.operator_plan.require(
                family=OperatorFamily.GENERIC,
                forms=(OperatorForm.INCREMENTAL,),
                coordinate_map=OperatorCoordinateMap.UNSPECIFIED,
                partial_state=OperatorPartialState.NONE,
                reduction=OperatorReduction.NONE,
                flags=(
                    OperatorPlanFlag.FIXED_CAPACITY
                    | OperatorPlanFlag.GRAPH_STABLE
                    | OperatorPlanFlag.EXTERNAL_WAVE_SOURCES
                    | OperatorPlanFlag.GENERATION_BOUND
                ),
            )
        except Exception:
            program.close()
            raise
        self._transport_program = program
        self._stats["transport_program_loaded"] = True
        self._stats["transport_program_bytes"] = path.stat().st_size
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
        on_replayed: Callable[[], None] | None,
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
            if on_replayed is not None:
                on_replayed()
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
        if on_replayed is not None:
            on_replayed()
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
            raise RuntimeError(
                "framework CUDA graphs require fully materialized KV and the "
                "stock FlashInfer consumer"
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
        observe_setup = (
            batch.incremental_metadata_setup_ns > 0
            and not batch.incremental_setup_observed
        )
        measure_topology = self._profile_cpu or observe_setup
        topology_started = time.perf_counter_ns() if measure_topology else 0
        topology_built, semantic_elapsed = self._ensure_execution_plan(
            wrapper, layer, kv_cache, verify=verify_execution
        )
        execution_setup_elapsed = (
            time.perf_counter_ns() - topology_started if measure_topology else 0
        )
        topology_elapsed = max(0, execution_setup_elapsed - semantic_elapsed)
        if topology_built:
            self._stats["work_topology_builds"] += 1
        else:
            self._stats["work_topology_cache_hits"] += 1
        if self._profile_cpu:
            counter = (
                "work_topology_cpu_ns"
                if topology_built
                else "work_topology_cache_cpu_ns"
            )
            self._stats[counter] += topology_elapsed
        pending = batch.pending_host_load
        stream = torch.cuda.current_stream()
        run_options = {
            "k_scale": layer.k_scale_float,
            "v_scale": layer.v_scale_float,
        }
        final_layer = (
            int(layer.layer_id) - self._model_start_layer + 1 == self._model_layer_count
        )
        enqueue_started = time.perf_counter_ns()
        setup_dispatch_elapsed: int | None = None
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
        arriving_prefetch = False
        if (
            pending is not None
            and prefetched is not None
            and prefetched.transfer_first_slot is not None
        ):
            layer_execution = self._layer_execution_plan(wrapper, kv_cache)
            arriving_prefetch = (
                layer_execution.overlap_initial and not prefetched.ready_event.query()
            )
        if pending is not None and prefetched is not None and arriving_prefetch:
            attention_form = "arriving"
            if self._profile_barrier:
                arrive = torch.cuda.Event(enable_timing=True)
                arrive.record(stream)
                self._barrier_profiles.append(
                    (arrive, prefetched.ready_event, int(layer.layer_id))
                )
            (
                plan,
                schedule,
                object_count,
                ready_event,
                preloaded_object_count,
                host_execution,
            ) = self._upload_plan(
                wrapper,
                int(layer.layer_id),
                kv_cache,
                arriving_prefetch=True,
            )
            if (
                object_count != 0
                or ready_event is not prefetched.ready_event
                or preloaded_object_count != 0
                or host_execution is None
                or not host_execution.overlap_initial
            ):
                raise RuntimeError("arriving host layer has an inconsistent plan")
            allocation = self._plans[(id(wrapper), -1)]
            initial_ready_work_count = allocation.direct_work_count
            if not 0 < initial_ready_work_count < schedule.work_count:
                raise RuntimeError(
                    "arriving host layer requires direct and external work"
                )
            epoch = FlashInferLayerEpoch(
                self._runtime,
                plan,
                self._phase_program(wrapper),
                object_count=0,
                max_progress_rounds=1,
                wait_for_plan=False,
            )
            progress_rounds = epoch.enqueue_arriving_host(
                wrapper,
                q,
                kv_cache,
                output,
                ready_event=prefetched.ready_event,
                initial_ready_work_count=initial_ready_work_count,
                sm_scale=layer.scaling,
                stream=stream,
                run_options=run_options,
            )
            self._stats["mixed_dependency_layers"] += 1
            self._stats["compact_initial_launches"] += 1
            self._stats["compact_initial_cta_bound"] += initial_ready_work_count
            self._stats["canonical_initial_cta_bound"] += schedule.work_count
            deferred_work_count = schedule.work_count - initial_ready_work_count
            self._stats["compact_resume_launches"] += 1
            self._stats["compact_resume_cta_bound"] += deferred_work_count
            self._stats["canonical_resume_cta_bound"] += schedule.work_count
            self._stats["request_work_completed"] += schedule.work_count
            self._stats["request_compute_completed_ns"] += (
                schedule.work_count * self._host_cost_model.tile_compute_ns
            )
            self._stats["ticketed_incremental_launches"] += 1
            self._stats["arriving_prefetch_launches"] = (
                self._stats.get("arriving_prefetch_launches", 0) + 1
            )
            if (
                final_layer
                or verify_execution
                or os.environ.get("NTA_VERIFY_TRANSFER") == "1"
            ):
                epoch.check(progress_rounds, stream)
        elif pending is not None and prefetched is not None:
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
            if execution_plan.rounds == 1 and not execution_plan.overlap_initial:
                attention_form = "bulk"
                output = self._run_bulk_host_layer(
                    wrapper,
                    q,
                    kv_cache,
                    output,
                    layer,
                    stream,
                    causal=causal,
                    window_left=window_left,
                    run_options=run_options,
                )
                self._stats["direct_host_layers"] += 1
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
                orchestration_started = (
                    time.perf_counter_ns() if self._profile_cpu else 0
                )
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
                cumulative_ready_work_counts = conservative_resume_counts(
                    block_counts=tuple(progress_blocks),
                    work_count=schedule.work_count - initial_ready_work_count,
                    max_object_fanout=allocation.max_object_fanout,
                    min_unresolved_dependencies=allocation.min_unresolved_dependencies,
                )
                if allocation.exact_resume_windows:
                    previous_count = 0
                    ready_work_offsets_values = []
                    ready_work_counts_values = []
                    for cumulative_count in cumulative_ready_work_counts:
                        ready_work_offsets_values.append(
                            initial_ready_work_count + previous_count
                        )
                        ready_work_counts_values.append(
                            cumulative_count - previous_count
                        )
                        previous_count = cumulative_count
                    if previous_count != schedule.work_count - initial_ready_work_count:
                        raise RuntimeError(
                            "exact resume windows do not cover deferred work"
                        )
                    ready_work_offsets = tuple(ready_work_offsets_values)
                    ready_work_counts = tuple(ready_work_counts_values)
                    self._stats["exact_resume_window_layers"] = (
                        self._stats.get("exact_resume_window_layers", 0) + 1
                    )
                else:
                    ready_work_offsets = ()
                    ready_work_counts = cumulative_ready_work_counts
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
                        ready_work_offsets=(
                            ready_work_offsets if ready_work_offsets else None
                        ),
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
                if self._profile_cpu:
                    submission_started = time.perf_counter_ns()
                    self._stats["incremental_orchestration_cpu_ns"] = (
                        self._stats.get("incremental_orchestration_cpu_ns", 0)
                        + submission_started
                        - orchestration_started
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
                        ready_work_offsets=tuple(ready_work_offsets),
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
                        lambda: epoch.mark_consumed_after_replay(stream),
                    )
                else:
                    enqueue_demand(q, output, eager_events, on_discovered)
                if self._profile_cpu:
                    submitted_ns = time.perf_counter_ns()
                    self._stats["incremental_submission_cpu_ns"] = self._stats.get(
                        "incremental_submission_cpu_ns", 0
                    ) + (submitted_ns - submission_started)
                if observe_setup:
                    # Only work required to make the first native epoch
                    # runnable is recurring setup. Cross-layer lookahead below
                    # is issued after GPU work exists and is overlapped with
                    # that work; charging it here previously made a whole-model
                    # publication loop look like a first-dispatch dependency.
                    setup_dispatch_elapsed = time.perf_counter_ns() - enqueue_started
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
                    and not pending.prefetched_layers
                ):
                    frontier_started = time.perf_counter_ns()
                    retained_before = len(pending.prefetch_tensors)
                    self._prepare_cross_layer_frontier(pending)
                    batch.prefetched_layers.update(pending.prefetched_layers)
                    batch.prefetch_tensors += pending.prefetch_tensors[
                        retained_before:
                    ]
                    self._stats["incremental_frontier_cpu_ns"] = self._stats.get(
                        "incremental_frontier_cpu_ns", 0
                    ) + (time.perf_counter_ns() - frontier_started)
                lookahead_started = time.perf_counter_ns() if self._profile_cpu else 0
                self._enqueue_fragment_lookahead(
                    wrapper,
                    int(layer.layer_id),
                    object_count,
                    host_execution,
                    stream,
                )
                if self._profile_cpu:
                    self._stats["incremental_lookahead_cpu_ns"] = self._stats.get(
                        "incremental_lookahead_cpu_ns", 0
                    ) + (time.perf_counter_ns() - lookahead_started)
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
        elapsed_ns = time.perf_counter_ns() - enqueue_started
        if (
            attention_form in {"incremental", "arriving"}
            and batch.incremental_metadata_setup_ns > 0
            and not batch.incremental_setup_observed
        ):
            batch.incremental_setup_observation_ns = (
                batch.incremental_metadata_setup_ns
                + execution_setup_elapsed
                + (
                    elapsed_ns
                    if setup_dispatch_elapsed is None
                    else setup_dispatch_elapsed
                )
            )
            batch.incremental_setup_observed = True
        if self._profile_cpu:
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
            and self._tier_service.is_host_staged
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
        self._record_execution_layer(layer, final_layer=final_layer)
        if pending is not None:
            self._stats["external_launches"] += 1
            self._stats["native_external_attention_launches"] += 1
            self._hicache.complete_layer(pending, local_layer)
        if final_layer:
            self._commit_incremental_setup_observation(batch)
            self._publish_stats()
        return output.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _commit_incremental_setup_observation(self, batch: _ActiveBatch) -> None:
        """Publish one completed epoch's control cost for later decisions."""

        observed_ns = batch.incremental_setup_observation_ns
        if observed_ns <= 0:
            return
        is_probe = (
            batch.host_execution is not None
            and batch.host_execution.selection_reason == "calibration_probe"
        )
        if is_probe:
            self._incremental_calibration_probes_remaining = max(
                0, self._incremental_calibration_probes_remaining - 1
            )
        initialization_sample = (
            is_probe and self._incremental_initialization_probes_remaining > 0
        )
        if initialization_sample:
            self._incremental_initialization_probes_remaining -= 1
            self._stats["incremental_initialization_samples"] += 1
            self._stats["incremental_initialization_setup_ns"] += observed_ns
        else:
            first_sample = self._incremental_setup_samples == 0
            self._host_cost_model = (
                self._host_cost_model.with_incremental_setup_observation(
                    elapsed_ns=observed_ns,
                    alpha=1.0 if first_sample else 0.25,
                    maximum_step_ratio=64.0 if first_sample else 4.0,
                )
            )
            self._incremental_setup_samples += 1
        batch.incremental_setup_observation_ns = 0
        self._stats["incremental_setup_samples"] = self._incremental_setup_samples
        self._stats["incremental_setup_calibrated"] = (
            self._host_cost_model.incremental_setup_ns is not None
        )
        self._stats["incremental_setup_ns"] = self._host_cost_model.incremental_setup_ns
        self._stats["incremental_setup_observed_ns_total"] = (
            self._stats.get("incremental_setup_observed_ns_total", 0) + observed_ns
        )
        self._stats["incremental_setup_observed_ns_max"] = max(
            self._stats.get("incremental_setup_observed_ns_max", 0),
            observed_ns,
        )
        self._stats["incremental_calibration_probes_remaining"] = (
            self._incremental_calibration_probes_remaining
        )

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

    def _run_preloaded_stock_layer(
        self,
        typed_wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        layer: Any,
        *,
        causal: bool,
        window_left: int,
    ) -> torch.Tensor:
        batch = self._active_batch
        if batch is None or batch.pending_host_load is None:
            raise RuntimeError("preloaded stock layer has no external lease")
        pending = batch.pending_host_load
        local_layer = self._wait_for_stock_external_layer(pending, layer)
        profile = None
        stream = torch.cuda.current_stream()
        if self._profile_gpu:
            profile = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            profile[0].record(stream)
        output = self._run_ready_stock_numerical(
            typed_wrapper,
            q,
            kv_cache,
            layer,
            causal=causal,
            window_left=window_left,
        )
        if profile is not None:
            profile[1].record(stream)
            self._operator_profiles.append((*profile, "preloaded_stock"))
        self._stats["stock_attention_launches"] += 1
        self._stats["stock_prefetched_external_attention_launches"] += 1
        self._stats["lookahead_bound_launches"] += 1
        self._stats["external_launches"] += 1
        if os.environ.get("NTA_VERIFY_TRANSFER") == "1":
            self._verify_layer_transfer(int(layer.layer_id), kv_cache)
        final_layer = local_layer + 1 == self._model_layer_count
        if final_layer:
            self._indexed_object_quiescence_event.record(stream)
            self._indexed_object_quiescence_recorded = True
        self._hicache.complete_layer(pending, local_layer)
        if final_layer:
            self._commit_incremental_setup_observation(batch)
            self._publish_stats()
        return output.view(-1, layer.tp_q_head_num * layer.head_dim)

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
        self._record_layer_arrival("decode", q, layer)
        if self._stock_forward:
            pending = self._active_batch.pending_host_load
            if pending is None:
                output = FlashInferAttnBackend.forward_decode(
                    self,
                    q,
                    k,
                    v,
                    layer,
                    forward_batch,
                    save_kv_cache=save_kv_cache,
                )
                if (
                    self._active_batch.publish_stats_on_completion
                    and int(layer.layer_id) + 1 == self._model_end_layer
                ):
                    self._publish_stats()
                return output
            self._stats["stock_attention_launches"] += 1
            self._stats["stock_prefetched_external_attention_launches"] += 1
            self._stats["external_launches"] += 1
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
            self._hicache.complete_layer(pending, local_layer)
            if local_layer + 1 == self._model_layer_count:
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
        pending = self._active_batch.pending_host_load
        local_layer = int(layer.layer_id) - self._model_start_layer
        if (
            pending is not None
            and local_layer in self._active_batch.prefetched_layers
            and id(wrapper) in self._stock_wrapper_for_typed
        ):
            return self._run_preloaded_stock_layer(
                wrapper,
                q,
                kv_cache,
                layer,
                causal=False,
                window_left=layer.sliding_window_size,
            )
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
        self._record_layer_arrival("extend", q, layer)
        if self._stock_forward:
            pending = self._active_batch.pending_host_load
            if pending is None:
                output = FlashInferAttnBackend.forward_extend(
                    self,
                    q,
                    k,
                    v,
                    layer,
                    forward_batch,
                    save_kv_cache=save_kv_cache,
                )
                if (
                    self._active_batch.publish_stats_on_completion
                    and int(layer.layer_id) + 1 == self._model_end_layer
                ):
                    self._publish_stats()
                return output
            self._stats["stock_attention_launches"] += 1
            self._stats["stock_prefetched_external_attention_launches"] += 1
            self._stats["external_launches"] += 1
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
        pending = self._active_batch.pending_host_load
        local_layer = int(layer.layer_id) - self._model_start_layer
        if (
            pending is not None
            and local_layer in self._active_batch.prefetched_layers
            and id(wrapper) in self._stock_wrapper_for_typed
        ):
            return self._run_preloaded_stock_layer(
                wrapper,
                q,
                kv_cache,
                layer,
                causal=causal,
                window_left=window_left,
            )
        return self._run_attention(
            wrapper,
            q,
            kv_cache,
            layer,
            causal=causal,
            window_left=window_left,
        )

    def _collect_mover_profiles(self) -> None:
        pending: list[_MoverProfile] = []
        for profile in self._mover_profiles:
            if not profile.finish.query():
                pending.append(profile)
                continue
            milliseconds = profile.start.elapsed_time(profile.finish)
            elapsed_ns = max(1, round(milliseconds * 1_000_000.0))
            previous = self._host_mover_service_model
            if profile.engine == "sm":
                self._host_mover_service_model = previous.with_sm_observation(
                    transfer_bytes=profile.transfer_bytes,
                    elapsed_ns=elapsed_ns,
                )
            else:
                self._host_mover_service_model = previous.with_copy_observation(
                    transfer_bytes=profile.transfer_bytes,
                    elapsed_ns=elapsed_ns,
                    operation_count=profile.operation_count,
                    issue_cpu_ns=profile.issue_cpu_ns,
                )
            profile_label = "sm" if profile.engine == "sm" else "copy"
            prefix = f"host_mover_profiled_{profile_label}"
            self._stats[f"{prefix}_bytes"] += profile.transfer_bytes
            self._stats[f"{prefix}_gpu_ms"] += milliseconds

        self._mover_profiles = pending
        model = self._host_mover_service_model
        self._stats["host_mover_sm_samples"] = model.sm_samples
        self._stats["host_mover_copy_samples"] = model.copy_samples
        self._stats["host_mover_copy_calibrated"] = model.copy_calibrated
        self._stats["host_mover_sm_bandwidth_bps"] = (
            model.sm_bandwidth_bytes_per_second
        )
        self._stats["host_mover_copy_bandwidth_bps"] = (
            model.copy_bandwidth_bytes_per_second
        )
        self._stats["host_mover_copy_operation_ns"] = model.copy_operation_ns

    def _collect_transfer_profiles(self) -> None:
        self._collect_mover_profiles()
        pending: list[tuple[torch.cuda.Event, torch.cuda.Event, int, str]] = []
        for start, finish, transfer_bytes, kind in self._transfer_profiles:
            if not finish.query():
                pending.append((start, finish, transfer_bytes, kind))
                continue
            milliseconds = start.elapsed_time(finish)
            elapsed_ns = max(1, round(milliseconds * 1_000_000.0))
            previous_bandwidth = self._host_cost_model.bandwidth_bytes_per_second
            self._host_cost_model = self._host_cost_model.with_transfer_observation(
                transfer_bytes=transfer_bytes,
                elapsed_ns=elapsed_ns,
            )
            if self._host_cost_model.bandwidth_bytes_per_second != previous_bandwidth:
                self._stats["cost_model_transfer_samples"] += 1
                self._stats["cost_model_bandwidth_bps"] = (
                    self._host_cost_model.bandwidth_bytes_per_second
                )
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

    def _stats_report(self, *, lifecycle: str = "served") -> dict[str, Any]:
        self._collect_transfer_profiles()
        self._collect_layer_service_profiles()
        self._collect_barrier_profiles()
        report = dict(self._stats)
        report.update(self._tier_service.stats())
        report["layer_service_curves"] = [
            {
                "phase": key[0],
                "query_rows": key[1],
                "batch_size": key[2],
                "samples": len(curve.samples_ns),
                "conservative_layer_ns": curve.conservative_layer_ns,
            }
            for key, curve in sorted(self._layer_service_curves.items())
        ]
        consumer_contract = _consumer_contract_for_stats(
            report,
            engine_version=os.environ.get("NTA_SGLANG_VERSION", "0.5.16"),
        )
        report["consumer_contract"] = consumer_contract.as_dict()
        report["execution_protocol_status"] = consumer_contract.kind.value
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
        if self._transport_program is not None:
            transport_contract = self._transport_program.operator_contract
            transport_plan = self._transport_program.operator_plan
            report["transport_contract"] = {
                "schema_version": transport_contract.schema_version,
                "runtime_abi_version": transport_contract.runtime_abi_version,
                "family": transport_contract.family.name.lower(),
                "form": transport_contract.form.name.lower(),
                "capabilities": int(transport_contract.capabilities),
                "instrumentation_flags": int(transport_contract.instrumentation_flags),
                "tier_mask": transport_contract.tier_mask,
                "supported_forms": transport_plan.supported_forms,
                "flags": int(transport_plan.flags),
                "source_fingerprint": transport_contract.source_fingerprint,
                "plan_fingerprint": transport_plan.plan_fingerprint,
            }
        report["tier_descriptors"] = [
            {
                "source_kind": descriptor.source_kind.name.lower(),
                "capabilities": _flag_value(descriptor.capabilities),
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
        report["stats_lifecycle"] = lifecycle
        report["snapshot_unix_ns"] = time.time_ns()
        report["finished_unix_ns"] = report["snapshot_unix_ns"]
        return report

    def _publish_stats(self) -> None:
        if self._stats_publisher is None:
            return
        self._stats_publisher.publish(self._stats_report())

    def _write_stats(self, *, strict: bool = False) -> None:
        if self._closed:
            return
        shutdown_error: BaseException | None = None
        try:
            if self._stats_publisher is not None:
                self._stats_publisher.publish(
                    self._stats_report(lifecycle="shutdown"), wait=True
                )
        except BaseException as error:
            shutdown_error = error
        finally:
            try:
                if self._stats_publisher is not None:
                    self._stats_publisher.close()
            except BaseException as error:
                if shutdown_error is None:
                    shutdown_error = error
            try:
                self._close_resources()
            except BaseException as error:
                if shutdown_error is None:
                    shutdown_error = error
            self._closed = True
        if shutdown_error is not None and strict:
            raise RuntimeError(
                "NTA engine shutdown completed with a statistics or resource error"
            ) from shutdown_error
