"""Typed per-layer dispatch and numerical execution for SGLang attention.

The framework backend owns request lifecycle and SGLang API adaptation.  This
module owns the mutually-exclusive numerical/transport paths selected for one
model layer.  Keeping those responsibilities separate prevents framework
hooks from becoming a second execution state machine.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
import math
import os
import pathlib
import time
from typing import Any

import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)

from nta_runtime.flashinfer import (
    FlashInferLayerEpoch,
    PREACQUIRED_LAUNCH_FLAGS,
    VALIDATE_RUNTIME_HEALTH,
    enqueue_event_partitioned_attention,
)
from nta_runtime.execution_planner import (
    HostCostModel,
    HostExecutionPlan,
    plan_host_layer_execution,
)
from nta_runtime.opportunity import OperatorArrival, TileArrival, append_json_line
from nta_runtime.engines.sglang_graphs import DemandGraphCache, demand_graph_key
from nta_runtime.engines.sglang_planning import requires_feasible_edf
from nta_runtime.engines.sglang_state import SglangForwardEpoch, _BarrierProfile
from nta_runtime.engines.sglang_nvme import SglangNvmeAcquisitionPipeline


class AttentionDispatchKind(str, Enum):
    PREACQUIRED = "direct"
    ARRIVING_PREFETCH = "arriving"
    PRELOADED = "preloaded"
    NVME = "nvme"
    HOST_INCREMENTAL = "incremental"
    HOST_DEVICE_BULK = "device_bulk"


@dataclass(frozen=True, slots=True)
class AttentionDispatch:
    """One validated numerical/transport form for a model layer."""

    kind: AttentionDispatchKind
    local_layer: int
    prefetched: Any | None
    host_execution: HostExecutionPlan | None


@dataclass(frozen=True, slots=True)
class AttentionDispatchOutcome:
    """Execution facts consumed by the framework-owned layer lifecycle."""

    epoch: FlashInferLayerEpoch | None = None
    indexed_object_count: int = 0
    progress_rounds: int = 0
    progressive_consumer: bool = False
    output: torch.Tensor | None = None
    deadline_fragment: DeadlineFragment | None = None
    setup_dispatch_elapsed_ns: int | None = None


@dataclass(frozen=True, slots=True)
class DeadlineFragment:
    """A demand fragment offered to the framework-owned deadline frontier."""

    wrapper: Any
    object_count: int
    host_execution: HostExecutionPlan
    stream: torch.cuda.Stream


@dataclass(frozen=True, slots=True)
class SglangAttentionExecutionConfig:
    """Immutable execution settings consumed below the framework boundary."""

    tenant_isolation_enabled: bool
    indexed_copy_target_bytes: int
    indexed_copy_max_blocks: int
    stream_ordered_retirement: bool
    demand_graph_enabled: bool
    profile_barrier: bool
    profile_cpu: bool
    profile_gpu: bool
    profile_transfer: bool
    measure_opportunity_compute: bool
    opportunity_parallel_slots: int
    opportunity_trace: pathlib.Path | None
    opportunity_revision: str
    opportunity_model: str
    opportunity_tier: str

    def __post_init__(self) -> None:
        if (
            min(
                self.indexed_copy_target_bytes,
                self.indexed_copy_max_blocks,
                self.opportunity_parallel_slots,
            )
            <= 0
        ):
            raise ValueError("SGLang attention execution geometry must be positive")
        if self.opportunity_trace is not None and (
            not self.opportunity_revision or not self.opportunity_model
        ):
            raise ValueError("opportunity tracing requires revision and model identity")


_EventPair = tuple[torch.cuda.Event, torch.cuda.Event]


@dataclass(frozen=True, slots=True)
class _PreparedHostLayer:
    """Validated physical and launch state for one host-demand layer."""

    host_execution: HostExecutionPlan
    device_bulk: bool
    plan: Any
    schedule: Any
    allocation: Any
    object_count: int
    preloaded_event: torch.cuda.Event | None
    preloaded_object_count: int
    template: Any
    epoch: FlashInferLayerEpoch
    collect_progress: bool
    coalesce_stream_retirement: bool
    transfer_profile: _EventPair | None
    discovery_profile: _EventPair | None
    consumer_profile: _EventPair | None
    retirement_profile: _EventPair | None
    orchestration_started_ns: int

    @property
    def progress_rounds(self) -> int:
        return len(self.template.progress_blocks)


def select_attention_dispatch(
    *,
    pending: Any | None,
    host_execution: HostExecutionPlan | None,
    tier_is_nvme: bool,
    layer_id: int,
    prefetch_event_ordered: bool = False,
    modeled_ready_by_attention: bool = False,
) -> AttentionDispatch:
    """Select exactly one layer path without submitting CUDA work."""

    if pending is None:
        if host_execution is not None:
            raise RuntimeError("resident attention retained a host execution plan")
        return AttentionDispatch(
            AttentionDispatchKind.PREACQUIRED,
            -1,
            None,
            None,
        )

    start_layer = int(getattr(pending.controller.mem_pool_device, "start_layer", 0))
    local_layer = int(layer_id) - start_layer
    if local_layer < 0:
        raise RuntimeError("attention layer precedes its HiCache owner partition")
    prefetched = pending.prefetched_layers.get(local_layer)
    if prefetched is not None:
        if prefetched.transfer_first_slot is not None and host_execution is None:
            raise RuntimeError("SM-prefetched attention has no host execution decision")
        arriving = (
            prefetched.transfer_first_slot is not None
            and host_execution.uses_dependency_protocol
            and host_execution.overlap_initial
            and not prefetch_event_ordered
            and not modeled_ready_by_attention
            and not prefetched.ready_event.query()
        )
        return AttentionDispatch(
            AttentionDispatchKind.ARRIVING_PREFETCH
            if arriving
            else AttentionDispatchKind.PRELOADED,
            local_layer,
            prefetched,
            host_execution,
        )

    if tier_is_nvme:
        if host_execution is not None:
            raise RuntimeError("NVMe attention retained a host execution plan")
        return AttentionDispatch(
            AttentionDispatchKind.NVME,
            local_layer,
            None,
            None,
        )

    if host_execution is None:
        raise RuntimeError("host-staged attention has no execution decision")
    if not host_execution.uses_dependency_protocol:
        raise RuntimeError(
            "direct host execution reached a typed wrapper; metadata selection "
            "must dispatch the stock preacquired consumer"
        )
    return AttentionDispatch(
        AttentionDispatchKind.HOST_DEVICE_BULK
        if host_execution.uses_device_bulk
        else AttentionDispatchKind.HOST_INCREMENTAL,
        local_layer,
        None,
        host_execution,
    )


def use_preloaded_stock_alias(
    dispatch: AttentionDispatch,
    *,
    alias_available: bool,
    typed_observation_required: bool = False,
) -> bool:
    """Return whether a typed wrapper may use its stock numerical alias.

    Submission and readiness are intentionally distinct: an arriving layer
    already has a transport fence, but only PRELOADED proves that the fence is
    complete.  Keeping this decision beside the pure dispatcher prevents the
    framework adapter from turning an in-flight layer into a blocking stock
    launch merely because its descriptor has been published.
    """

    return (
        alias_available
        and not typed_observation_required
        and dispatch.kind is AttentionDispatchKind.PRELOADED
    )


class SglangAttentionExecutor:
    """Execute typed attention forms without owning framework lifecycle.

    Dependencies are the concrete runtime owners needed by numerical
    execution.  There is deliberately no reference to
    ``NtaFlashInferAttnBackend``: forward admission, request retirement, and
    SGLang metadata remain at the framework boundary.
    """

    def __init__(
        self,
        *,
        runtime: Any,
        tier_service: Any,
        hicache: Any,
        materializer: Any,
        nvme_pipeline: SglangNvmeAcquisitionPipeline | None,
        kernels: Any,
        demand_graph_cache: DemandGraphCache,
        progress_stream: torch.cuda.Stream,
        stats: dict[str, Any],
        stock_wrapper: Callable[[int], Any | None],
        transfer_profiles: list[tuple[torch.cuda.Event, torch.cuda.Event, int, str]],
        operator_profiles: list[tuple[torch.cuda.Event, torch.cuda.Event, str, int]],
        barrier_profiles: list[_BarrierProfile],
        config: SglangAttentionExecutionConfig,
    ) -> None:
        self._runtime = runtime
        self._tier_service = tier_service
        self._hicache = hicache
        self._materializer = materializer
        self._nvme_pipeline = nvme_pipeline
        self._kernels = kernels
        self._demand_graph_cache = demand_graph_cache
        self._progress_stream = progress_stream
        self._stats = stats
        self._stock_wrapper = stock_wrapper
        self._transfer_profiles = transfer_profiles
        self._operator_profiles = operator_profiles
        self._barrier_profiles = barrier_profiles
        self._config = config
        self._demand_sync_events: dict[
            tuple[int, int, int],
            tuple[torch.cuda.Event, tuple[torch.cuda.Event, ...]],
        ] = {}

    def prepare_nvme_batch(
        self,
        *,
        batch: SglangForwardEpoch,
        wrappers: tuple[Any, ...],
        ordering_stream: torch.cuda.Stream,
        kv_cache_for_layer: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
        tile_compute_ns: int,
    ) -> None:
        """Bind exact consumers and enqueue every layer's NVMe producer."""

        pipeline = self._nvme_pipeline
        if pipeline is None or batch.nvme_acquisition is not None:
            raise RuntimeError("NVMe acquisition pipeline is unavailable or reused")
        if not wrappers or {id(wrapper) for wrapper in wrappers} != set(
            batch.semantic_plans
        ):
            raise RuntimeError("NVMe consumer wrappers do not cover semantic plans")
        if tile_compute_ns <= 0:
            raise RuntimeError("NVMe acquisition has no positive compute estimate")
        inter_layer_compute_ns = max(
            semantic.schedule.work_count
            for semantic in batch.semantic_plans.values()
        ) * tile_compute_ns

        def prepare_consumers(stream: torch.cuda.Stream) -> None:
            for wrapper in wrappers:
                semantic = batch.semantic_plans[id(wrapper)]
                plan = self._materializer.upload_preacquired_plan(
                    batch,
                    wrapper,
                    semantic.schedule,
                    semantic.topology,
                    stream=stream,
                )
                allocation = self._materializer.require_allocation(wrapper)
                if plan.has_external or allocation.object_count != 0:
                    raise RuntimeError(
                        "event-ready NVMe consumer retained transport ownership"
                    )

        batch.nvme_acquisition = pipeline.prepare(
            semantic_plans=batch.semantic_plans,
            bindings=batch.bindings,
            ordering_stream=ordering_stream,
            prepare_consumers=prepare_consumers,
            kv_cache_for_layer=kv_cache_for_layer,
            inter_layer_compute_ns=inter_layer_compute_ns,
        )

    def clear(self) -> None:
        """Release reusable event topology after all CUDA work is quiescent."""

        self._demand_sync_events.clear()

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

    def _upload_plan(
        self,
        batch: SglangForwardEpoch,
        wrapper: Any,
        layer_id: int,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        *,
        arriving_prefetch: bool = False,
    ) -> tuple[Any, Any, int, torch.cuda.Event | None, int, HostExecutionPlan | None]:
        return self._materializer.upload_plan(
            batch,
            wrapper,
            layer_id,
            kv_cache,
            arriving_prefetch=arriving_prefetch,
        )

    def prepare_arriving_plans(
        self,
        *,
        batch: SglangForwardEpoch,
        wrappers: tuple[Any, ...],
        layer_id: int,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Materialize layer-invariant event partitions during batch binding."""

        started = time.perf_counter_ns() if self._config.profile_cpu else 0
        prepared = 0
        for wrapper in wrappers:
            if id(wrapper) not in batch.semantic_plans:
                raise RuntimeError(
                    "arriving-plan preparation found an unplanned wrapper"
                )
            self._materializer.prepare_arriving_plan(batch, wrapper, layer_id, kv_cache)
            prepared += 1
        if prepared == 0:
            raise RuntimeError("arriving-plan preparation has no typed wrappers")
        self._stats["arriving_plan_preparations"] = (
            self._stats.get("arriving_plan_preparations", 0) + prepared
        )
        if self._config.profile_cpu:
            self._stats["arriving_plan_prepare_cpu_ns"] = self._stats.get(
                "arriving_plan_prepare_cpu_ns", 0
            ) + (time.perf_counter_ns() - started)

    def run_preacquired(
        self,
        batch: SglangForwardEpoch,
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        output: torch.Tensor,
        layer: Any,
        run_options: dict[str, Any],
        *,
        validate_runtime_health: bool = False,
    ) -> None:
        """Run exact compiler-verified work after all dependencies are ready."""

        if not self._kernels.is_instrumented(wrapper):
            raise RuntimeError(
                "NTA preacquired attention requires a compiler-transformed "
                "FlashInfer wrapper"
            )
        # The same compiler-verified incremental module consumes unresolved
        # and already-ready plans. PREACQUIRED_LAUNCH_FLAGS removes transport
        # work without introducing a second numerical module or tensor ABI.
        self._kernels.operator_module(wrapper)
        runtime_tensor = self._runtime.device_view_tensor
        module_name = self._kernels.module_name(wrapper)
        allocation = None
        schedule = None
        if batch.pending_host_load is not None:
            semantic = batch.semantic_plans.get(id(wrapper))
            schedule = None if semantic is None else semantic.schedule
            allocation = self._materializer.allocation(wrapper)
            if (
                schedule is None
                or allocation is None
                or allocation.plan.work_item_count != schedule.work_count
            ):
                planned = sorted(
                    self._kernels.describe_wrapper_id(wrapper_id)
                    for wrapper_id in batch.semantic_plans
                )
                raise RuntimeError(
                    "preacquired external attention has no validated CTA work "
                    f"plan: wrapper={module_name} schedule={schedule is not None} "
                    f"allocation={allocation is not None} "
                    f"work_count={getattr(schedule, 'work_count', None)} "
                    "plan_items="
                    f"{getattr(getattr(allocation, 'plan', None), 'work_item_count', None)} "
                    f"planned_wrappers={planned}"
                )
            if allocation.plan.has_external:
                raise RuntimeError(
                    "event-complete attention retained transport dependencies"
                )
        else:
            semantic = batch.semantic_plans.get(id(wrapper))
            if semantic is None:
                raise RuntimeError(
                    "resident demand attention has no exact CTA topology"
                )
            schedule = semantic.schedule
            self._materializer.upload_preacquired_plan(
                batch,
                wrapper,
                schedule,
                semantic.topology,
                stream=torch.cuda.current_stream(),
            )
            allocation = self._materializer.require_allocation(wrapper)
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
            PREACQUIRED_LAUNCH_FLAGS
            | (VALIDATE_RUNTIME_HEALTH if validate_runtime_health else 0),
            out=output,
            **run_options,
        )
        allocation.plan.mark_consumed(torch.cuda.current_stream())
        self._stats["transformed_direct_launches"] += 1

    def execute_non_host(
        self,
        *,
        dispatch: AttentionDispatch,
        batch: SglangForwardEpoch,
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        output: torch.Tensor,
        layer: Any,
        stream: torch.cuda.Stream,
        run_options: dict[str, Any],
        final_layer: bool,
        verify_execution: bool,
        verify_transfer: bool,
        tile_compute_ns: int,
    ) -> AttentionDispatchOutcome:
        """Execute a non-host-demand form selected by the pure dispatcher."""

        if dispatch.kind is AttentionDispatchKind.PREACQUIRED:
            self.run_preacquired(
                batch, wrapper, q, kv_cache, output, layer, run_options
            )
            return AttentionDispatchOutcome()
        if dispatch.kind is AttentionDispatchKind.ARRIVING_PREFETCH:
            return self._execute_arriving_prefetch(
                dispatch=dispatch,
                batch=batch,
                wrapper=wrapper,
                q=q,
                kv_cache=kv_cache,
                output=output,
                layer=layer,
                stream=stream,
                run_options=run_options,
                final_layer=final_layer,
                verify_execution=verify_execution,
                verify_transfer=verify_transfer,
                tile_compute_ns=tile_compute_ns,
            )
        if dispatch.kind is AttentionDispatchKind.PRELOADED:
            return self._execute_preloaded(
                dispatch=dispatch,
                batch=batch,
                wrapper=wrapper,
                q=q,
                kv_cache=kv_cache,
                output=output,
                layer=layer,
                stream=stream,
                run_options=run_options,
            )
        if dispatch.kind is AttentionDispatchKind.NVME:
            return self._execute_nvme(
                batch=batch,
                wrapper=wrapper,
                q=q,
                kv_cache=kv_cache,
                output=output,
                layer=layer,
                stream=stream,
                run_options=run_options,
                final_layer=final_layer,
                verify_execution=verify_execution,
                verify_transfer=verify_transfer,
            )
        raise RuntimeError("host-demand attention reached the non-host executor")

    def _execute_arriving_prefetch(
        self,
        *,
        dispatch: AttentionDispatch,
        batch: SglangForwardEpoch,
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        output: torch.Tensor,
        layer: Any,
        stream: torch.cuda.Stream,
        run_options: dict[str, Any],
        final_layer: bool,
        verify_execution: bool,
        verify_transfer: bool,
        tile_compute_ns: int,
    ) -> AttentionDispatchOutcome:
        pending = batch.pending_host_load
        prefetched = dispatch.prefetched
        if pending is None or prefetched is None:
            raise RuntimeError("arriving dispatch lost its HiCache readiness")
        self._record_barrier_arrival(prefetched.ready_event, layer, stream)
        (
            plan,
            schedule,
            object_count,
            ready_event,
            preloaded_object_count,
            host_execution,
        ) = self._upload_plan(
            batch,
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
            or not host_execution.uses_dependency_protocol
            or not host_execution.overlap_initial
        ):
            raise RuntimeError("arriving host layer has an inconsistent plan")
        allocation = self._materializer.require_allocation(wrapper)
        initial_ready_work_count = allocation.direct_work_count
        wave_work_counts = allocation.event_wave_work_counts
        nonempty_wave_count = sum(count > 0 for count in wave_work_counts)
        if (
            not allocation.event_partitioned
            or not 0 < initial_ready_work_count < schedule.work_count
            or len(wave_work_counts) != prefetched.wave_count
            or initial_ready_work_count + sum(wave_work_counts) != schedule.work_count
            or nonempty_wave_count <= 0
        ):
            raise RuntimeError(
                "arriving host layer requires an exact direct/wave partition"
            )
        partition_key = (
            id(wrapper),
            id(plan),
            schedule.work_count,
            initial_ready_work_count,
            wave_work_counts,
        )
        prepare_partition = batch.arriving_partition_key != partition_key
        enqueue_event_partitioned_attention(
            self._runtime,
            plan,
            self._kernels.transport_program(),
            wrapper,
            q,
            kv_cache,
            output,
            ready_events=prefetched.wave_events,
            ready_object_slots=prefetched.wave_object_slots,
            registration_event=prefetched.registration_event,
            direct_work_count=initial_ready_work_count,
            wave_work_counts=wave_work_counts,
            prepare_partition=prepare_partition,
            sm_scale=layer.scaling,
            stream=stream,
            run_options=run_options,
        )
        batch.arriving_partition_key = partition_key
        partition_counter = (
            "arriving_partition_preparations"
            if prepare_partition
            else "arriving_partition_reuses"
        )
        self._stats[partition_counter] = self._stats.get(partition_counter, 0) + 1
        self._stats["mixed_dependency_layers"] += 1
        self._stats["compact_initial_launches"] += 1
        self._stats["compact_initial_cta_bound"] += initial_ready_work_count
        self._stats["canonical_initial_cta_bound"] += schedule.work_count
        deferred_work_count = schedule.work_count - initial_ready_work_count
        self._stats["compact_resume_launches"] += nonempty_wave_count
        self._stats["compact_resume_cta_bound"] += deferred_work_count
        self._stats["canonical_resume_cta_bound"] += (
            schedule.work_count * nonempty_wave_count
        )
        self._stats["request_work_completed"] += schedule.work_count
        self._stats["request_compute_completed_ns"] += (
            schedule.work_count * tile_compute_ns
        )
        self._stats["event_ordered_incremental_launches"] = (
            self._stats.get("event_ordered_incremental_launches", 0) + 1
        )
        self._stats["event_ordered_wave_launches"] = (
            self._stats.get("event_ordered_wave_launches", 0) + nonempty_wave_count
        )
        self._stats["arriving_prefetch_launches"] = (
            self._stats.get("arriving_prefetch_launches", 0) + 1
        )
        return AttentionDispatchOutcome(
            progress_rounds=nonempty_wave_count,
            progressive_consumer=True,
        )

    def _execute_preloaded(
        self,
        *,
        dispatch: AttentionDispatch,
        batch: SglangForwardEpoch,
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        output: torch.Tensor,
        layer: Any,
        stream: torch.cuda.Stream,
        run_options: dict[str, Any],
    ) -> AttentionDispatchOutcome:
        pending = batch.pending_host_load
        prefetched = dispatch.prefetched
        if pending is None or prefetched is None:
            raise RuntimeError("preloaded dispatch lost its HiCache readiness")
        self._record_barrier_arrival(prefetched.ready_event, layer, stream)
        stream.wait_event(prefetched.ready_event)
        # A pure-preloaded batch is legal, so materialization cannot rely on
        # an earlier mixed layer having populated this structural plan.
        self._upload_plan(batch, wrapper, int(layer.layer_id), kv_cache)
        self.run_preacquired(batch, wrapper, q, kv_cache, output, layer, run_options)
        self._stats["lookahead_bound_launches"] += 1
        return AttentionDispatchOutcome()

    def _execute_nvme(
        self,
        *,
        batch: SglangForwardEpoch,
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        output: torch.Tensor,
        layer: Any,
        stream: torch.cuda.Stream,
        run_options: dict[str, Any],
        final_layer: bool,
        verify_execution: bool,
        verify_transfer: bool,
    ) -> AttentionDispatchOutcome:
        pipeline = self._nvme_pipeline
        acquisition = batch.nvme_acquisition
        if batch.pending_host_load is None or pipeline is None or acquisition is None:
            raise RuntimeError("NVMe dispatch lost its proactive acquisition")
        local_layer = int(layer.layer_id) - acquisition.layers[0].layer_id
        acquired = acquisition.layer(local_layer)
        self._record_barrier_arrival(acquired.ready_event, layer, stream)
        pipeline.wait_layer(acquisition, acquired, stream)
        allocation = self._materializer.require_allocation(wrapper)
        semantic = batch.semantic_plans.get(id(wrapper))
        if (
            semantic is None
            or allocation.plan.has_external
            or allocation.object_count != 0
            or allocation.plan.work_item_count != semantic.schedule.work_count
        ):
            raise RuntimeError("NVMe event-ready consumer plan is incomplete")
        self.run_preacquired(
            batch,
            wrapper,
            q,
            kv_cache,
            output,
            layer,
            run_options,
            validate_runtime_health=True,
        )
        schedule = semantic.schedule
        self._stats["request_work_completed"] += schedule.work_count
        self._stats["tier_external_layers"] += 1
        self._stats["nvme_preacquired_launches"] = (
            self._stats.get("nvme_preacquired_launches", 0) + 1
        )
        if final_layer:
            pipeline.record_consumer(acquisition, stream)
        if final_layer and self._runtime.sticky_failed_count != 0:
            raise RuntimeError("the proactive NVMe acquisition pipeline failed")
        return AttentionDispatchOutcome()

    def _record_barrier_arrival(
        self,
        ready_event: torch.cuda.Event,
        layer: Any,
        stream: torch.cuda.Stream,
    ) -> None:
        if not self._config.profile_barrier:
            return
        arrive = torch.cuda.Event(enable_timing=True)
        arrive.record(stream)
        self._barrier_profiles.append(
            _BarrierProfile(
                arrive,
                ready_event,
                int(layer.layer_id),
                "attention_layer",
            )
        )

    def execute_host(
        self,
        *,
        dispatch: AttentionDispatch,
        batch: SglangForwardEpoch,
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        output: torch.Tensor,
        layer: Any,
        stream: torch.cuda.Stream,
        run_options: dict[str, Any],
        causal: bool,
        window_left: int,
        final_layer: bool,
        verify_execution: bool,
        verify_transfer: bool,
        observe_setup: bool,
        enqueue_started_ns: int,
        host_cost_model: HostCostModel,
        active_opportunity_batch: int,
    ) -> AttentionDispatchOutcome:
        """Execute one host-demand layer through prepare, submit, and account."""

        prepared = self._prepare_host_layer(
            dispatch=dispatch,
            batch=batch,
            wrapper=wrapper,
            kv_cache=kv_cache,
            layer=layer,
            verify_execution=verify_execution,
            verify_transfer=verify_transfer,
        )
        on_discovered = self._progress_publisher(batch, dispatch.local_layer)
        output, setup_dispatch_elapsed_ns = self._submit_host_layer(
            prepared=prepared,
            batch=batch,
            wrapper=wrapper,
            q=q,
            kv_cache=kv_cache,
            output=output,
            layer=layer,
            stream=stream,
            run_options=run_options,
            causal=causal,
            window_left=window_left,
            on_discovered=on_discovered,
            observe_setup=observe_setup,
            enqueue_started_ns=enqueue_started_ns,
        )

        deadline_fragment = (
            None
            if prepared.device_bulk
            else DeadlineFragment(
                wrapper,
                prepared.object_count,
                prepared.host_execution,
                stream,
            )
        )
        self._stats["ticketed_incremental_launches"] += 1
        if (
            final_layer or prepared.collect_progress or verify_transfer
        ) and not prepared.coalesce_stream_retirement:
            prepared.epoch.check(prepared.progress_rounds, stream)
        if (
            final_layer
            and not prepared.coalesce_stream_retirement
            and self._runtime.sticky_failed_count != 0
        ):
            raise RuntimeError("an earlier asynchronous acquisition epoch failed")
        self._account_host_progress(prepared, batch, host_cost_model)
        self._record_opportunity(
            prepared,
            batch,
            wrapper,
            q,
            kv_cache,
            output,
            layer,
            run_options,
            host_cost_model,
            active_opportunity_batch,
        )
        return AttentionDispatchOutcome(
            epoch=prepared.epoch,
            indexed_object_count=prepared.object_count,
            progress_rounds=prepared.progress_rounds,
            progressive_consumer=prepared.template.progressive_consumer,
            output=output,
            deadline_fragment=deadline_fragment,
            setup_dispatch_elapsed_ns=setup_dispatch_elapsed_ns,
        )

    def _prepare_host_layer(
        self,
        *,
        dispatch: AttentionDispatch,
        batch: SglangForwardEpoch,
        wrapper: Any,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        layer: Any,
        verify_execution: bool,
        verify_transfer: bool,
    ) -> _PreparedHostLayer:
        if dispatch.kind not in {
            AttentionDispatchKind.HOST_INCREMENTAL,
            AttentionDispatchKind.HOST_DEVICE_BULK,
        }:
            raise RuntimeError("non-host attention reached the host executor")
        pending = batch.pending_host_load
        execution_plan = dispatch.host_execution
        if pending is None or execution_plan is None:
            raise RuntimeError("host dispatch lost its execution lease")
        device_bulk = dispatch.kind is AttentionDispatchKind.HOST_DEVICE_BULK
        if device_bulk != execution_plan.uses_device_bulk:
            raise RuntimeError("host dispatch changed after selection")

        (
            plan,
            schedule,
            object_count,
            preloaded_event,
            preloaded_object_count,
            host_execution,
        ) = self._upload_plan(batch, wrapper, int(layer.layer_id), kv_cache)
        orchestration_started_ns = (
            time.perf_counter_ns() if self._config.profile_cpu else 0
        )
        if host_execution != execution_plan:
            raise RuntimeError("host execution plan changed during planning")
        allocation = self._materializer.require_allocation(wrapper)
        if (preloaded_event is None) != (preloaded_object_count == 0):
            raise RuntimeError("host fragment readiness and object ownership disagree")
        if 0 < allocation.direct_work_count < schedule.work_count:
            self._stats["mixed_dependency_layers"] += 1

        queued_feasible_edf = requires_feasible_edf(
            batch.bindings,
            tenant_isolation=self._config.tenant_isolation_enabled,
        )
        template_key = (
            id(wrapper),
            preloaded_object_count,
            int(queued_feasible_edf),
        )
        template = batch.host_layer_templates.get(template_key)
        expected_demand_bytes = allocation.transfer_bytes - sum(
            allocation.object_transfer_bytes[:preloaded_object_count]
        )
        if template is None:
            template = plan_host_layer_execution(
                host_execution=host_execution,
                object_count=object_count,
                work_count=schedule.work_count,
                transfer_bytes=allocation.transfer_bytes,
                object_transfer_bytes=allocation.object_transfer_bytes,
                external_object_slots=allocation.external_object_slots,
                direct_work_count=allocation.direct_work_count,
                max_object_fanout=allocation.max_object_fanout,
                min_unresolved_dependencies=allocation.min_unresolved_dependencies,
                preloaded_object_count=preloaded_object_count,
                queued_feasible_edf=queued_feasible_edf,
                indexed_copy_target_bytes=self._config.indexed_copy_target_bytes,
                indexed_copy_max_blocks=self._config.indexed_copy_max_blocks,
            )
            batch.host_layer_templates[template_key] = template
            self._stats["execution_template_builds"] += 1
        else:
            if (
                template.object_count != object_count
                or template.work_count != schedule.work_count
                or template.direct_work_count != allocation.direct_work_count
                or template.demand_transfer_bytes != expected_demand_bytes
                or template.queued_feasible_edf != queued_feasible_edf
            ):
                raise RuntimeError(
                    "cached host execution template changed within a forward"
                )
            self._stats["execution_template_reuses"] += 1

        self._stats["compact_initial_launches"] += int(
            template.initial_ready_work_count != 0
        )
        self._stats["compact_initial_cta_bound"] += template.initial_ready_work_count
        self._stats["canonical_initial_cta_bound"] += schedule.work_count
        if device_bulk:
            self._stats["device_bulk_layers"] += 1
        elif queued_feasible_edf:
            self._stats["queued_feasible_edf_layers"] += 1
            self._stats["dynamic_runnable_window_layers"] = (
                self._stats.get("dynamic_runnable_window_layers", 0) + 1
            )
            self._stats["unique_resume_work_bound"] = self._stats.get(
                "unique_resume_work_bound", 0
            ) + (schedule.work_count - template.initial_ready_work_count)
        else:
            self._stats["indexed_range_fastpath_layers"] += 1
            self._stats["unqueued_host_discovery_layers"] += int(
                template.indexed_host_order_prevalidated
            )
        self._stats["exact_resume_window_layers"] += int(template.exact_resume_windows)
        self._stats["compact_resume_launches"] += template.nonempty_resume_windows
        self._stats["compact_resume_cta_bound"] += sum(template.ready_work_counts)
        self._stats["canonical_resume_cta_bound"] += (
            len(template.ready_work_counts) * schedule.work_count
        )

        epoch = FlashInferLayerEpoch(
            self._runtime,
            plan,
            self._kernels.transport_program(),
            object_count=object_count,
            max_progress_rounds=len(template.progress_blocks),
            wait_for_plan=False,
            stream_ordered_retirement=self._config.stream_ordered_retirement,
        )
        if self._config.stream_ordered_retirement:
            self._stats["stream_ordered_retirement_layers"] += 1
        collect_progress = (
            verify_execution or self._config.opportunity_trace is not None
        )
        coalesce_stream_retirement = (
            self._config.stream_ordered_retirement
            # Native indexed acquisition owns per-layer object issue counts,
            # work tickets, and request/tenant byte credits. Those states must
            # complete before the directory slots are rebound. Only an
            # event-owned arriving consumer has no object state to retire.
            and object_count == 0
            and not collect_progress
            and not verify_transfer
        )
        transfer_profile = (
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            if self._config.profile_transfer
            else None
        )
        discovery_profile = (
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            if self._config.profile_gpu
            else None
        )
        consumer_profile = (
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            if self._config.profile_gpu and self._config.stream_ordered_retirement
            else None
        )
        retirement_profile = (
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            if self._config.profile_gpu
            and self._config.stream_ordered_retirement
            and not coalesce_stream_retirement
            else None
        )
        return _PreparedHostLayer(
            host_execution=host_execution,
            device_bulk=device_bulk,
            plan=plan,
            schedule=schedule,
            allocation=allocation,
            object_count=object_count,
            preloaded_event=preloaded_event,
            preloaded_object_count=preloaded_object_count,
            template=template,
            epoch=epoch,
            collect_progress=collect_progress,
            coalesce_stream_retirement=coalesce_stream_retirement,
            transfer_profile=transfer_profile,
            discovery_profile=discovery_profile,
            consumer_profile=consumer_profile,
            retirement_profile=retirement_profile,
            orchestration_started_ns=orchestration_started_ns,
        )

    def _progress_publisher(
        self,
        batch: SglangForwardEpoch,
        local_layer: int,
    ) -> Callable[[Any], None] | None:
        if local_layer != 0 or not self._hicache.progress_publication_available():
            return None
        request_slots = tuple(binding.request_slot for binding in batch.bindings)
        first_request_slot = min(request_slots)
        if request_slots != tuple(
            range(first_request_slot, first_request_slot + len(request_slots))
        ):
            self._stats["progress_feedback_skipped_noncontiguous"] = (
                self._stats.get("progress_feedback_skipped_noncontiguous", 0) + 1
            )
            return None
        progress_snapshot = self._runtime.request_progress_snapshot(len(request_slots))

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

        return publish_progress

    def _submit_host_layer(
        self,
        *,
        prepared: _PreparedHostLayer,
        batch: SglangForwardEpoch,
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        output: torch.Tensor,
        layer: Any,
        stream: torch.cuda.Stream,
        run_options: dict[str, Any],
        causal: bool,
        window_left: int,
        on_discovered: Callable[[Any], None] | None,
        observe_setup: bool,
        enqueue_started_ns: int,
    ) -> tuple[torch.Tensor, int | None]:
        template = prepared.template

        def enqueue_demand(
            query: torch.Tensor,
            destination: torch.Tensor,
            sync_events: tuple[torch.cuda.Event, tuple[torch.cuda.Event, ...]],
            discovery_callback: Callable[[Any], None] | None,
        ) -> None:
            prepared.epoch.enqueue_host(
                wrapper,
                query,
                kv_cache,
                destination,
                progress_blocks=template.progress_blocks,
                sm_scale=layer.scaling,
                stream=stream,
                progress_stream=self._progress_stream,
                ready_event=prepared.preloaded_event,
                ready_work_counts=template.ready_work_counts,
                ready_work_offsets=template.ready_work_offsets,
                initial_ready_work_count=template.initial_ready_work_count,
                indexed_host_first_object=template.indexed_host_first_object,
                indexed_host_range_prevalidated=(
                    template.indexed_host_range_prevalidated
                ),
                indexed_host_order_prevalidated=(
                    template.indexed_host_order_prevalidated
                ),
                indexed_host_copy_blocks_per_group=(
                    template.indexed_copy_blocks_per_group
                ),
                sync_events=sync_events,
                discovery_profile=prepared.discovery_profile,
                progress_profile=prepared.transfer_profile,
                consumer_profile=prepared.consumer_profile,
                retirement_profile=prepared.retirement_profile,
                complete_stream_ordered=(not prepared.coalesce_stream_retirement),
                on_discovered=discovery_callback,
                run_options=run_options,
            )

        eager_events = self._layer_sync_events(
            int(layer.layer_id),
            prepared.progress_rounds,
            stream,
        )
        graph_eligible = (
            self._config.demand_graph_enabled
            and isinstance(
                wrapper,
                (
                    BatchDecodeWithPagedKVCacheWrapper,
                    BatchPrefillWithPagedKVCacheWrapper,
                ),
            )
            and prepared.preloaded_event is None
            and prepared.preloaded_object_count == 0
            and prepared.transfer_profile is None
            and prepared.discovery_profile is None
        )
        if self._config.profile_cpu:
            submission_started_ns = time.perf_counter_ns()
            self._stats["incremental_orchestration_cpu_ns"] = (
                self._stats.get("incremental_orchestration_cpu_ns", 0)
                + submission_started_ns
                - prepared.orchestration_started_ns
            )
        if graph_eligible:
            graph_key = demand_graph_key(
                operator_family=(
                    "decode"
                    if isinstance(wrapper, BatchDecodeWithPagedKVCacheWrapper)
                    else "paged_prefill"
                ),
                wrapper=wrapper,
                layer_id=int(layer.layer_id),
                stream_address=int(stream.cuda_stream),
                plan=prepared.plan,
                runtime_tensor=self._runtime.device_view_tensor,
                work_count=prepared.schedule.work_count,
                object_count=prepared.object_count,
                progress_blocks=tuple(template.progress_blocks),
                ready_work_counts=tuple(template.ready_work_counts),
                ready_work_offsets=tuple(template.ready_work_offsets),
                initial_ready_work_count=template.initial_ready_work_count,
                indexed_copy_blocks_per_group=(template.indexed_copy_blocks_per_group),
                query=q,
                kv_cache=kv_cache,
                sm_scale=layer.scaling,
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
                causal=causal,
                window_left=window_left,
            )

            def after_graph_replay() -> None:
                self._complete_graph_replay(
                    prepared.epoch,
                    stream,
                    on_discovered,
                )

            output = self._demand_graph_cache.enqueue(
                graph_key,
                wrapper,
                q,
                output,
                stream,
                enqueue_demand,
                eager_events,
                on_discovered,
                after_graph_replay,
            )
        else:
            enqueue_demand(q, output, eager_events, on_discovered)

        if prepared.coalesce_stream_retirement:
            prior_epoch = batch.stream_ordered_epoch
            if prior_epoch is not None and prior_epoch.plan is not prepared.plan:
                raise RuntimeError(
                    "one forward cannot defer retirement across work plans"
                )
            batch.stream_ordered_epoch = prepared.epoch
            batch.stream_ordered_progress_rounds = prepared.progress_rounds
            batch.stream_ordered_layers += 1
        elif self._config.stream_ordered_retirement:
            self._stats["stream_ordered_retirement_launches"] += 1
        if self._config.profile_cpu:
            submitted_ns = time.perf_counter_ns()
            self._stats["incremental_submission_cpu_ns"] = self._stats.get(
                "incremental_submission_cpu_ns", 0
            ) + (submitted_ns - submission_started_ns)

        setup_dispatch_elapsed_ns = (
            time.perf_counter_ns() - enqueue_started_ns if observe_setup else None
        )
        self._stats["parallel_indexed_progress_layers"] += 1
        self._stats["prevalidated_indexed_progress_layers"] = (
            self._stats.get("prevalidated_indexed_progress_layers", 0) + 1
        )
        if prepared.transfer_profile is not None:
            self._transfer_profiles.append(
                (
                    *prepared.transfer_profile,
                    template.demand_transfer_bytes,
                    "demand",
                )
            )
        if prepared.discovery_profile is not None:
            self._operator_profiles.append(
                (*prepared.discovery_profile, "work_discovery", 1)
            )
        if prepared.consumer_profile is not None:
            self._operator_profiles.append(
                (*prepared.consumer_profile, "incremental_ready_consumer", 1)
            )
        if prepared.retirement_profile is not None:
            self._operator_profiles.append(
                (*prepared.retirement_profile, "stream_retirement", 1)
            )
        return output, setup_dispatch_elapsed_ns

    def _account_host_progress(
        self,
        prepared: _PreparedHostLayer,
        batch: SglangForwardEpoch,
        host_cost_model: HostCostModel,
    ) -> None:
        if not prepared.collect_progress:
            self._stats["request_work_completed"] += prepared.schedule.work_count
            self._stats["request_compute_completed_ns"] += (
                prepared.schedule.work_count * host_cost_model.tile_compute_ns
            )
            return

        request_slots = tuple(binding.request_slot for binding in batch.bindings)
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
            prepared.schedule.request_indices[index]
            for index, object_slots in enumerate(
                prepared.allocation.external_object_slots
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
            raise RuntimeError("external request produced no progress attribution")
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

    def _record_opportunity(
        self,
        prepared: _PreparedHostLayer,
        batch: SglangForwardEpoch,
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        output: torch.Tensor,
        layer: Any,
        run_options: dict[str, Any],
        host_cost_model: HostCostModel,
        active_opportunity_batch: int,
    ) -> None:
        if self._config.opportunity_trace is None:
            return
        tile_compute_ns = host_cost_model.tile_compute_ns
        compute_source = "calibrated"
        if self._config.measure_opportunity_compute:
            tile_compute_ns = self._measure_flashinfer_tile_compute(
                wrapper,
                q,
                kv_cache,
                output,
                run_options,
                prepared.schedule.work_count,
            )
            compute_source = "measured"
        runnable_ns = self._runtime.work_runnable_ns(prepared.schedule.work_count)
        tiles = tuple(
            TileArrival(
                request_id=f"{batch.bindings[request_index].request_id:016x}",
                tile_id=work_ticket,
                available_ns=runnable_ns[work_ticket],
                compute_ns=tile_compute_ns,
                logical_tile=prepared.schedule.kv_tile_indices[work_ticket],
                availability_source=(
                    "resident_at_launch"
                    if runnable_ns[work_ticket] == 0
                    else "gpu_globaltimer"
                ),
                compute_source=compute_source,
            )
            for work_ticket, request_index in enumerate(
                prepared.schedule.request_indices
            )
        )
        append_json_line(
            self._config.opportunity_trace,
            OperatorArrival(
                batch_id=f"{os.getpid()}:{active_opportunity_batch}",
                layer=int(layer.layer_id),
                tiles=tiles,
                revision=self._config.opportunity_revision,
                engine="sglang",
                model=self._config.opportunity_model,
                tier=self._config.opportunity_tier,
                observed_at_unix_ns=time.time_ns(),
            ),
        )

    def _complete_graph_replay(
        self,
        epoch: FlashInferLayerEpoch,
        stream: torch.cuda.Stream,
        on_discovered: Callable[[Any], None] | None,
    ) -> None:
        """Publish completion feedback that cannot run during graph capture."""

        epoch.mark_consumed_after_replay(stream)
        if on_discovered is None:
            return
        on_discovered(stream)
        key = "progress_feedback_graph_completion_snapshots"
        self._stats[key] = self._stats.get(key, 0) + 1

    def _measure_flashinfer_tile_compute(
        self,
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        output: torch.Tensor,
        run_options: dict[str, Any],
        work_count: int,
    ) -> int:
        """Measure stock numerical service without acquisition attribution."""

        if work_count <= 0 or not self._kernels.is_instrumented(wrapper):
            raise RuntimeError("compute calibration requires instrumented CTA work")
        stock = self._stock_wrapper(id(wrapper))
        if stock is None:
            raise RuntimeError(
                "compute calibration requires the paired stock numerical wrapper"
            )
        start = torch.cuda.Event(enable_timing=True)
        finish = torch.cuda.Event(enable_timing=True)
        calibration_output = torch.empty_like(output)
        stream = torch.cuda.current_stream(q.device)
        start.record(stream)
        stock.run(q, kv_cache, out=calibration_output, **run_options)
        finish.record(stream)
        finish.synchronize()
        kernel_ns = max(1, math.ceil(start.elapsed_time(finish) * 1_000_000))
        active_slots = min(work_count, self._config.opportunity_parallel_slots)
        tile_ns = max(1, math.ceil(kernel_ns * active_slots / work_count))
        self._stats["opportunity_calibration_launches"] = (
            self._stats.get("opportunity_calibration_launches", 0) + 1
        )
        self._stats["opportunity_calibration_kernel_ns"] = (
            self._stats.get("opportunity_calibration_kernel_ns", 0) + kernel_ns
        )
        return tile_ns
