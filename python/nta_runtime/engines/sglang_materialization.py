"""Physical plan materialization for SGLang attention.

This component owns reusable DeviceWorkPlan storage, runtime directory
publication, and directory lifetime tokens. It does not own framework request
binding, numerical dispatch, scheduling policy, or report publication.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Any

import torch

from nta_runtime.flashinfer import direct_requirement
from nta_runtime.engines.sglang_planning import (
    DEMAND_OBJECT_ID_BASE,
    MAX_ABI_BYTES,
)
from nta_runtime.engines.sglang_state import SglangForwardEpoch
from nta_runtime.execution_planner import HostExecutionPlan
from nta_runtime.execution_topology import WorkDependencySpan
from nta_runtime.execution_topology import ExactWorkTopology
from nta_runtime.flashinfer_schedule import Schedule
from nta_runtime.indexed_transfer import IndexedTensorLane
from nta_runtime.runtime import (
    DeviceWorkPlan,
    INVALID_INDEX,
    IndexedAcquisitionPlan,
    IndexedHostIndexBinding,
    JitPhaseProgram,
    MAX_EVENT_COMPLETION_CLASSES,
)


@dataclass(frozen=True)
class MaterializedAttentionPlan:
    """Immutable numerical view of one materialized attention plan.

    Mutable signatures, object versions, retained index tensors, and capacity
    bookkeeping remain private to :class:`SglangPlanMaterializer`.  Framework
    dispatch may inspect this snapshot but cannot mutate physical ownership.
    """

    plan: DeviceWorkPlan
    object_count: int
    transfer_bytes: int
    object_transfer_bytes: tuple[int, ...]
    indexed_geometry: tuple[int, ...] | None
    max_object_fanout: int
    min_unresolved_dependencies: int
    direct_work_count: int
    event_partitioned: bool
    event_wave_work_counts: tuple[int, ...]
    external_object_slots: tuple[tuple[int, ...], ...]
    external_work_mask: tuple[bool, ...]


@dataclass
class _PlanAllocation:
    plan: DeviceWorkPlan
    work_capacity: int
    signature: tuple[Any, ...] | None = None
    object_count: int = 0
    index_tensors: tuple[torch.Tensor, ...] = ()
    object_version: int = 0
    transfer_bytes: int = 0
    # Exact physical payload per indexed directory object, in object-slot
    # order. Host fragment lookahead consumes a prefix of this vector.
    object_transfer_bytes: tuple[int, ...] = ()
    indexed_geometry: tuple[int, ...] | None = None
    max_object_fanout: int = 1
    min_unresolved_dependencies: int = 1
    direct_work_count: int = 0
    event_partitioned: bool = False
    event_wave_work_counts: tuple[int, ...] = ()
    external_object_slots: tuple[tuple[int, ...], ...] = ()
    external_work_mask: tuple[bool, ...] = ()

    def view(self) -> MaterializedAttentionPlan:
        return MaterializedAttentionPlan(
            self.plan,
            self.object_count,
            self.transfer_bytes,
            self.object_transfer_bytes,
            self.indexed_geometry,
            self.max_object_fanout,
            self.min_unresolved_dependencies,
            self.direct_work_count,
            self.event_partitioned,
            self.event_wave_work_counts,
            self.external_object_slots,
            self.external_work_mask,
        )


class SglangPlanMaterializer:
    """Own physical plan buffers and tier directory publication."""

    def __init__(
        self,
        *,
        runtime: Any,
        tier_service: Any,
        max_dependencies_per_work_ticket: int,
        work_ticket_capacity: int,
        object_capacity: int,
        tenant_isolation_enabled: bool,
        profile_cpu: bool,
        stats: dict[str, Any],
        stock_wrapper_available: Callable[[int], bool],
        transport_program: Callable[[], JitPhaseProgram],
        discard_plan: Callable[[DeviceWorkPlan], None],
    ) -> None:
        if (
            min(
                max_dependencies_per_work_ticket,
                work_ticket_capacity,
                object_capacity,
            )
            <= 0
        ):
            raise ValueError("SGLang materializer capacities must be positive")
        self._runtime = runtime
        self._tier_service = tier_service
        self._max_dependencies_per_work_ticket = max_dependencies_per_work_ticket
        self._work_ticket_capacity = work_ticket_capacity
        self._object_capacity = object_capacity
        self._tenant_isolation_enabled = tenant_isolation_enabled
        self._profile_cpu = profile_cpu
        self._stats = stats
        self._stock_wrapper_available = stock_wrapper_available
        self._transport_program_callback = transport_program
        self._discard_plan_callback = discard_plan
        self._plans: dict[tuple[int, int], _PlanAllocation] = {}
        self._indexed_object_quiescence_event = (
            torch.cuda.Event() if self._tier_service.is_host_staged else None
        )
        self._indexed_object_quiescence_recorded = False

    def allocation(
        self, wrapper: Any, layer_id: int = -1
    ) -> MaterializedAttentionPlan | None:
        allocation = self._plans.get((id(wrapper), layer_id))
        return None if allocation is None else allocation.view()

    def require_allocation(
        self, wrapper: Any, layer_id: int = -1
    ) -> MaterializedAttentionPlan:
        allocation = self.allocation(wrapper, layer_id)
        if allocation is None:
            raise RuntimeError("SGLang attention has no materialized work plan")
        return allocation

    def _require_owned_allocation(
        self, wrapper: Any, layer_id: int = -1
    ) -> _PlanAllocation:
        allocation = self._plans.get((id(wrapper), layer_id))
        if allocation is None:
            raise RuntimeError("SGLang attention has no owned work-plan allocation")
        return allocation

    def ensure_plan(
        self, wrapper: Any, layer_id: int, schedule: Schedule
    ) -> DeviceWorkPlan:
        key = (id(wrapper), layer_id)
        allocation = self._plans.get(key)
        if allocation is not None and schedule.work_count <= allocation.work_capacity:
            return allocation.plan
        if allocation is not None:
            torch.cuda.current_stream().synchronize()
            self._discard_plan_callback(allocation.plan)
            allocation.plan.close()
        capacity = schedule.work_count
        plan = DeviceWorkPlan(
            capacity,
            self._max_dependencies_per_work_ticket * capacity,
            self._runtime.device_ordinal,
        )
        self._plans[key] = _PlanAllocation(plan, capacity)
        return plan

    def upload_preacquired_plan(
        self,
        batch: SglangForwardEpoch,
        wrapper: Any,
        schedule: Schedule,
        topology: ExactWorkTopology,
        *,
        completion_classes: tuple[int, ...] | None = None,
        stream: torch.cuda.Stream,
    ) -> DeviceWorkPlan:
        """Publish exact numerical work whose transport is producer-owned.

        Dependencies are always the resident runtime view, so the numerical
        plan never owns a tier transfer. Optional completion classes retain the
        compiler-verified work-to-acquisition-group mapping for a producer that
        publishes finite readiness waves.
        """

        plan = self.ensure_plan(wrapper, -1, schedule)
        allocation = self._require_owned_allocation(wrapper)
        class_values = (
            None
            if completion_classes is None
            else tuple(int(value) for value in completion_classes)
        )
        if class_values is not None and (
            len(class_values) != schedule.work_count
            or any(
                value != INVALID_INDEX
                and not 0 <= value < MAX_EVENT_COMPLETION_CLASSES
                for value in class_values
            )
            or all(value == INVALID_INDEX for value in class_values)
        ):
            raise ValueError("preacquired completion classes are inconsistent")
        signature = (
            "preacquired",
            schedule.request_indices,
            schedule.kv_tile_indices,
            tuple(binding.request_slot for binding in batch.bindings),
            tuple(binding.generation for binding in batch.bindings),
            class_values,
        )
        if allocation.signature == signature:
            return plan

        dependency_spans: list[WorkDependencySpan] = []
        dependencies = []
        for _work_ticket in range(schedule.work_count):
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
            completion_classes=class_values,
            stream=stream,
        )
        allocation.signature = signature
        allocation.object_count = 0
        allocation.direct_work_count = (
            schedule.work_count
            if class_values is None
            else sum(value == INVALID_INDEX for value in class_values)
        )
        allocation.event_partitioned = class_values is not None
        if class_values is None:
            allocation.event_wave_work_counts = ()
            allocation.external_work_mask = (False,) * schedule.work_count
        else:
            wave_counts = [0] * (1 + max(
                value for value in class_values if value != INVALID_INDEX
            ))
            for value in class_values:
                if value != INVALID_INDEX:
                    wave_counts[value] += 1
            allocation.event_wave_work_counts = tuple(wave_counts)
            allocation.external_work_mask = tuple(
                value != INVALID_INDEX for value in class_values
            )
        allocation.external_object_slots = tuple(() for _ in range(schedule.work_count))
        return plan

    def close(self) -> tuple[BaseException, ...]:
        """Close every plan after the owning backend has quiesced CUDA."""

        errors: list[BaseException] = []
        for allocation in tuple(self._plans.values()):
            try:
                allocation.plan.close()
            except BaseException as error:
                errors.append(error)
        self._plans.clear()
        return tuple(errors)

    def record_host_consumer(
        self,
        stream: torch.cuda.Stream,
        *,
        indexed_objects: bool,
        final_layer: bool,
    ) -> None:
        """Publish stream-ordered indexed-object reuse and final lifetime."""

        if indexed_objects or final_layer:
            self._require_indexed_quiescence_event().record(stream)
            self._indexed_object_quiescence_recorded = True

    def _transport_program(self) -> JitPhaseProgram:
        return self._transport_program_callback()

    def _require_indexed_quiescence_event(self) -> torch.cuda.Event:
        event = self._indexed_object_quiescence_event
        if event is None:
            raise RuntimeError("host-staged materialization has no quiescence event")
        return event

    def _record_demand_plan_stats(
        self,
        batch: SglangForwardEpoch,
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
        self._stats["native_demand_sm_bytes"] += transfer_bytes
        self._stats["host_progress_rounds"] += host_execution.rounds
        self._stats["predicted_atomic_ns"] += (
            host_execution.predicted_atomic_per_unit_ns
        )
        self._stats["predicted_incremental_ns"] += (
            host_execution.predicted_incremental_per_unit_ns
        )
        if host_execution.uses_dependency_protocol:
            self._stats["incremental_host_layers"] += 1
        if host_execution.overlap_initial:
            self._stats["request_overlap_layers"] += 1

    def _record_arriving_consumer_stats(
        self,
        host_execution: HostExecutionPlan,
    ) -> None:
        """Account an in-flight proactive acquisition without charging it twice.

        An arriving consumer publishes dependencies on the directory objects
        owned by ``LayerAcquisition`` and waits on that owner's readiness
        fence.  It neither registers an indexed demand acquisition nor moves a
        second copy of the K/V bytes.  In particular, physical-byte, object,
        group, and CTA accounting belongs to the proactive owner (CTA work is
        already counted once when the arriving plan is uploaded).
        """

        self._stats["demand_host_layers"] += 1
        if host_execution.uses_dependency_protocol:
            self._stats["incremental_host_layers"] += 1
        if host_execution.overlap_initial:
            self._stats["request_overlap_layers"] += 1

    def upload_plan(
        self,
        batch: SglangForwardEpoch,
        wrapper: Any,
        layer_id: int,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        *,
        arriving_prefetch: bool = False,
        preparing_arriving: bool = False,
    ) -> tuple[
        DeviceWorkPlan,
        Schedule,
        int,
        torch.cuda.Event | None,
        int,
        HostExecutionPlan | None,
    ]:
        if preparing_arriving and not arriving_prefetch:
            raise ValueError("only an arriving plan can be prepared ahead of use")
        if not self._tier_service.is_host_staged:
            raise RuntimeError(
                "physical tiers use their proactive acquisition owner; the "
                "attention materializer only owns host-staged plans"
            )
        profile_started = time.perf_counter_ns() if self._profile_cpu else 0
        wrapper_id = id(wrapper)
        semantic = batch.semantic_plans.get(wrapper_id)
        if semantic is None:
            raise RuntimeError("NTA attention wrapper was not planned for this batch")
        schedule = semantic.schedule
        if schedule.work_count > self._work_ticket_capacity:
            raise RuntimeError(
                f"FlashInfer needs {schedule.work_count} work tickets; configured "
                f"capacity is {self._work_ticket_capacity}"
            )
        key_cache, value_cache = kv_cache
        if not key_cache.is_cuda or not value_cache.is_cuda:
            raise RuntimeError("SGLang KV cache must be CUDA-addressable")
        key_bytes = min(int(key_cache.nbytes), MAX_ABI_BYTES)
        value_bytes = min(int(value_cache.nbytes), MAX_ABI_BYTES)
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
        prefetched = pending.prefetched_layers.get(local_layer)
        if (
            self._tenant_isolation_enabled
            and prefetched is not None
            and not getattr(pending, "shared_acquisition_registered", False)
        ):
            raise RuntimeError(
                "unaccounted proactive Host publication cannot bypass finite "
                "tenant budgets"
            )
        if arriving_prefetch and (
            prefetched is None or prefetched.transfer_first_slot is None
        ):
            raise RuntimeError(
                "arriving host work requires a directory-backed proactive layer"
            )
        if semantic.dependency_kind != "typed_lease":
            raise RuntimeError("host-staged attention requires typed lease semantics")
        acquisition_slices = semantic.acquisition_slices
        acquisition_groups = semantic.acquisition_groups
        if not any(item is not None for item in acquisition_slices):
            raise RuntimeError("typed acquisition plan has no external work")

        prefetched_signature = None
        if prefetched is not None:
            prefetched_signature = (
                prefetched.key_bytes,
                prefetched.value_bytes,
                # An arriving consumer never discovers these directory objects.
                # Its producer event owns readiness, while this plan retains only
                # the layer-invariant direct/deferred work partition.  Excluding
                # the per-layer slot is therefore both safe and required for one
                # forward-scoped upload.
                "event_partition" if arriving_prefetch else "event_complete",
                prefetched.wave_row_ends,
            )
        signature = semantic.signature_prefix + (
            key_bytes,
            value_bytes,
            prefetched_signature,
            self._tier_service.tier.value,
            self._tier_service.catalog_digest,
            None,
        )
        work_dependency_rows = semantic.work_dependency_rows
        # Work/ticket topology is layer invariant. Layer K/V addresses are
        # republished through the object directory on the consumer stream.
        plan = self.ensure_plan(wrapper, -1, schedule)
        allocation = self._plans[(id(wrapper), -1)]
        rebuild_plan = allocation.signature != signature
        if prefetched is not None and not rebuild_plan:
            host_execution = batch.host_execution
            if (
                host_execution is None
                or allocation.object_count != 0
                or allocation.event_partitioned != arriving_prefetch
                or len(allocation.external_work_mask) != schedule.work_count
                or (
                    arriving_prefetch
                    and sum(allocation.event_wave_work_counts)
                    + allocation.direct_work_count
                    != schedule.work_count
                )
            ):
                raise RuntimeError("cached HiCache plan is incomplete")
            if not preparing_arriving:
                self._stats["cta_work_items"] += schedule.work_count
                if arriving_prefetch:
                    self._record_arriving_consumer_stats(host_execution)
                    self._stats["arriving_prefetch_layers"] = (
                        self._stats.get("arriving_prefetch_layers", 0) + 1
                    )
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
                host_execution,
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
            host_execution = batch.host_execution
            object_count = allocation.object_count
            if (
                host_execution is None
                or object_count == 0
                or object_count % 2 != 0
                or allocation.transfer_bytes == 0
                or len(allocation.object_transfer_bytes) != object_count
                or sum(allocation.object_transfer_bytes) != allocation.transfer_bytes
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
                self._transport_program().rebind_indexed_host_pairs(
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
        if prefetched is None and rebuild_plan:
            allocation.object_version = (allocation.object_version + 1) & 0xFFFFFFFF
            allocation.object_version = allocation.object_version or 1
        # Object versions belong to directory-backed acquisitions. An
        # event-complete prefetch deliberately emits only direct requirements,
        # so coupling its state record to an otherwise-unused object version
        # creates a false ownership edge.
        version = allocation.object_version
        index_tensors: list[torch.Tensor] = []
        lease_index_map = None
        indexed_acquisition_plan = None
        if prefetched is None:
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
            indexed_topology = semantic.indexed_topology
            if indexed_topology is None:
                raise RuntimeError("typed lease lost its semantic acquisition topology")
            indexed_acquisition_plan = IndexedAcquisitionPlan(
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
                work_bindings=tuple(
                    batch.bindings[request_index]
                    for request_index in schedule.request_indices
                ),
                source_indices_device_address=lease_index_map.source_indices.data_ptr(),
                staging_indices_device_address=(
                    lease_index_map.destination_indices.data_ptr()
                ),
                object_version=version,
                direct_base=self._runtime.device_view,
                object_id_base=DEMAND_OBJECT_ID_BASE,
            )
            # Request-owned KV is tenant-local independently of whether finite
            # byte-budget enforcement is enabled.  A policy toggle may change
            # admission/accounting, never resource ownership.
            indexed_acquisition_plan.require_single_tenant_groups()
            if self._profile_cpu:
                self._stats["plan_indexed_materialization_cpu_ns"] = (
                    self._stats.get("plan_indexed_materialization_cpu_ns", 0)
                    + time.perf_counter_ns()
                    - indexed_materialization_started
                )
        dependency_build_started = time.perf_counter_ns() if self._profile_cpu else 0
        topology = semantic.topology
        if indexed_acquisition_plan is None:
            dependency_spans: list[WorkDependencySpan] = []
            dependencies: Any = []
            object_fanout: Counter[int] = Counter()
            unresolved_dependencies: list[int] = []
            direct_work_count = 0
            external_object_slots: list[tuple[int, ...]] = []
            external_work_mask: list[bool] = []
        else:
            dependency_spans = list(indexed_acquisition_plan.dependency_spans)
            dependencies = indexed_acquisition_plan.dependencies
            object_fanout = Counter({0: indexed_acquisition_plan.max_object_fanout})
            unresolved_dependencies = [
                indexed_acquisition_plan.min_unresolved_dependencies
            ]
            direct_work_count = indexed_acquisition_plan.direct_work_count
            external_object_slots = list(indexed_acquisition_plan.external_object_slots)
            external_work_mask = [bool(slots) for slots in external_object_slots]
        event_wave_work_counts: list[int] = []
        event_completion_classes: list[int] | None = None
        operation_by_id = {}
        if arriving_prefetch:
            if prefetched is None or prefetched.transfer_first_slot is None:
                raise RuntimeError("arriving plan lost its proactive SM owner")
            if prefetched.wave_count <= 0 or prefetched.wave_row_ends[-1] != int(
                pending.device_indices.numel()
            ):
                raise RuntimeError("arriving plan has incomplete wave readiness")
            event_wave_work_counts = [0] * prefetched.wave_count
            event_completion_classes = [INVALID_INDEX] * schedule.work_count
            operation_by_id = {
                operation.operation_id: operation
                for operation in pending.operation_ranges()
            }
        work_entries = (
            ()
            if indexed_acquisition_plan is not None
            else enumerate(
                zip(
                    schedule.request_indices,
                    schedule.kv_tile_indices,
                    acquisition_slices,
                    acquisition_groups,
                    work_dependency_rows,
                    strict=True,
                )
            )
        )
        for work_ticket, (
            _request_index,
            _kv_tile,
            dependency,
            transfer_dependency,
            external_rows,
        ) in work_entries:
            dependency_begin = len(dependencies)
            if external_rows > 0:
                external_work_mask.append(True)
                if prefetched is None:
                    raise RuntimeError(
                        "host demand work bypassed its compact indexed plan"
                    )
                if arriving_prefetch:
                    if dependency is None:
                        raise RuntimeError(
                            "arriving external work lost its exact lease slice"
                        )
                    operation = operation_by_id.get(dependency.operation_id)
                    if operation is None:
                        raise RuntimeError(
                            "arriving work names an unknown lease operation"
                        )
                    absolute_begin = operation.row_begin + dependency.row_begin
                    absolute_end = operation.row_begin + dependency.row_end
                    if (
                        absolute_begin < 0
                        or absolute_end <= absolute_begin
                        or absolute_end > prefetched.wave_row_ends[-1]
                    ):
                        raise RuntimeError(
                            "arriving work exceeds its wave-partitioned lease"
                        )
                    wave_indices: list[int] = []
                    wave_begin = 0
                    for wave, wave_end in enumerate(prefetched.wave_row_ends):
                        if absolute_begin < wave_end and wave_begin < absolute_end:
                            wave_indices.append(wave)
                        wave_begin = wave_end
                    if not wave_indices:
                        raise RuntimeError("arriving work has no completion wave")
                    # Transfer ownership and its physical object slots belong
                    # to LayerAcquisition. The numerical plan is preacquired
                    # and reusable across layers; only completionClass carries
                    # the stable producer/consumer ordering relation.
                    dependencies.extend(
                        (
                            direct_requirement(self._runtime.device_view, 1),
                            direct_requirement(self._runtime.device_view, 1),
                        )
                    )
                    completion_wave = wave_indices[-1]
                    event_wave_work_counts[completion_wave] += 1
                    if event_completion_classes is None:  # pragma: no cover
                        raise RuntimeError(
                            "arriving completion classes were not initialized"
                        )
                    event_completion_classes[work_ticket] = completion_wave
                    unresolved_dependencies.append(2 * len(wave_indices))
                    external_object_slots.append(())
                    direct_dependencies = 2
                else:
                    # A completed per-layer event is the sole completion edge;
                    # the numerical plan is therefore all-direct.
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
                external_work_mask.append(False)
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

        if len(external_work_mask) != schedule.work_count:
            raise RuntimeError("materialized external-work identity is incomplete")

        if self._profile_cpu:
            self._stats["plan_dependency_build_cpu_ns"] = self._stats.get(
                "plan_dependency_build_cpu_ns", 0
            ) + (time.perf_counter_ns() - dependency_build_started)

        if prefetched is not None:
            object_count = 0
        elif indexed_acquisition_plan is not None:
            object_count = indexed_acquisition_plan.object_count
        else:
            raise RuntimeError("host-staged layer omitted its typed acquisition plan")
        if object_count == 0 and prefetched is None:
            raise RuntimeError("external HiCache batch has no CTA dependency")
        if object_count > self._object_capacity:
            raise RuntimeError(
                f"HiCache layer needs {object_count} objects; configured capacity is "
                f"{self._object_capacity}"
            )
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
        elif indexed_acquisition_plan is not None:
            transfer_bytes = indexed_acquisition_plan.transfer_bytes
        else:
            raise RuntimeError("host-staged transfer has no typed byte geometry")
        # Selection is immutable for one forward epoch and belongs to the
        # engine planner. Materialization consumes that decision; it must
        # never silently invent a second policy from physical geometry.
        host_execution = batch.host_execution
        if host_execution is None:
            raise RuntimeError("host-staged batch has no execution decision")
        stream = torch.cuda.current_stream()
        if prefetched is None:
            registration_started = time.perf_counter_ns() if self._profile_cpu else 0
            quiescence_event = (
                self._require_indexed_quiescence_event()
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
            if indexed_acquisition_plan is None or index_binding is None:
                raise RuntimeError("host-staged publication is not lease-owned")
            self._runtime.register_indexed_acquisition_plan(
                indexed_acquisition_plan,
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
            self._transport_program().validate_indexed_host_range(
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
        incremental = host_execution.uses_dependency_protocol
        needs_plan = (
            prefetched is not None
            or incremental
            or not self._stock_wrapper_available(id(wrapper))
        )
        if needs_plan and rebuild_plan:
            upload_started = time.perf_counter_ns() if self._profile_cpu else 0
            plan.upload_exact(
                topology,
                dependency_spans,
                dependencies,
                completion_classes=event_completion_classes,
                stream=stream,
            )
            if self._profile_cpu:
                self._stats["native_plan_upload_cpu_ns"] = self._stats.get(
                    "native_plan_upload_cpu_ns", 0
                ) + (time.perf_counter_ns() - upload_started)
            self._stats["plan_uploads"] += 1
        if needs_plan and prefetched is not None and not preparing_arriving:
            self._stats["cta_work_items"] += schedule.work_count
        allocation.signature = signature
        allocation.object_count = object_count
        allocation.index_tensors = tuple(index_tensors)
        allocation.transfer_bytes = transfer_bytes
        allocation.object_transfer_bytes = (
            indexed_acquisition_plan.object_transfer_bytes
            if indexed_acquisition_plan is not None
            else ()
        )
        allocation.indexed_geometry = indexed_geometry
        allocation.max_object_fanout = max(object_fanout.values(), default=1)
        allocation.min_unresolved_dependencies = min(unresolved_dependencies, default=1)
        allocation.direct_work_count = direct_work_count
        allocation.event_partitioned = arriving_prefetch
        allocation.event_wave_work_counts = tuple(event_wave_work_counts)
        allocation.external_object_slots = tuple(external_object_slots)
        allocation.external_work_mask = tuple(external_work_mask)
        if prefetched is None:
            self._record_demand_plan_stats(
                batch,
                schedule,
                object_count,
                transfer_bytes,
                host_execution,
            )
        elif arriving_prefetch and not preparing_arriving:
            self._record_arriving_consumer_stats(host_execution)
            self._stats["arriving_prefetch_layers"] = (
                self._stats.get("arriving_prefetch_layers", 0) + 1
            )
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

    def prepare_arriving_plan(
        self,
        batch: SglangForwardEpoch,
        wrapper: Any,
        layer_id: int,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> DeviceWorkPlan:
        """Upload one event-partitioned structural plan before layer dispatch."""

        plan, schedule, object_count, _event, preloaded, host_execution = (
            self.upload_plan(
                batch,
                wrapper,
                layer_id,
                kv_cache,
                arriving_prefetch=True,
                preparing_arriving=True,
            )
        )
        allocation = self.require_allocation(wrapper)
        if (
            object_count != 0
            or preloaded != 0
            or host_execution is None
            or not allocation.event_partitioned
            or not 0 < allocation.direct_work_count < schedule.work_count
            or sum(allocation.event_wave_work_counts) + allocation.direct_work_count
            != schedule.work_count
        ):
            raise RuntimeError(
                "prepared arriving plan has inconsistent ownership "
                f"(objects={object_count}, preloaded={preloaded}, "
                f"host_execution={host_execution is not None}, "
                f"event_partitioned={allocation.event_partitioned}, "
                f"direct_work={allocation.direct_work_count}, "
                f"total_work={schedule.work_count})"
            )
        return plan
