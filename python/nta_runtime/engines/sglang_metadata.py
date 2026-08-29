"""Forward-scoped semantic planning for SGLang FlashInfer metadata.

This component translates one framework metadata snapshot into an immutable
``_ActiveBatch``.  It owns no CUDA stream, runtime directory, transport, or
framework lifecycle state, so the resulting plan can be inspected and tested
without entering numerical execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from nta_runtime.adapters.base import EngineBatch
from nta_runtime.adapters.sglang import (
    SglangAcquisitionSpan,
    SglangExecutionConfig,
    forward_metadata,
)
from nta_runtime.execution_planner import HostCostModel, HostExecutionMode, HostExecutionPlan
from nta_runtime.flashinfer_schedule import (
    Schedule,
    decode_schedule,
    paged_prefill_schedule,
)
from nta_runtime.requests import RequestBinding
from nta_runtime.engines.sglang_contracts import (
    LeaseAcquisitionGroup,
    LeaseAcquisitionSlice,
    PagePair,
)
from nta_runtime.engines.sglang_hicache import PendingHostLoad
from nta_runtime.engines.sglang_planning import cpu_sequence_lengths
from nta_runtime.engines.sglang_semantics import (
    build_semantic_wrapper_plan,
    plan_typed_lease_execution,
    prove_direct_metadata_execution,
    validate_schedule,
    work_page_pairs,
    wrapper_page_layout,
)
from nta_runtime.engines.sglang_state import _ActiveBatch, _SemanticWrapperPlan
from nta_runtime.engines.sglang_topology import (
    capacity_constrained_acquisition_groups,
    group_external_pages_by_request,
    project_acquisition_slices,
    request_batch_heterogeneity,
    resolve_request_acquisitions,
)


@dataclass(frozen=True, slots=True)
class SglangMetadataPlan:
    batch: _ActiveBatch
    host_execution: HostExecutionPlan | None


class SglangMetadataPlanner:
    """Build exact layer-invariant work topology from one forward snapshot."""

    def __init__(
        self,
        *,
        tier_service: Any,
        execution_config: SglangExecutionConfig,
        tenant_isolation_enabled: bool,
        object_capacity: int,
        grouping: str,
        page_size: int,
        profile_cpu: bool,
        stats: dict[str, Any],
    ) -> None:
        if object_capacity <= 0 or page_size <= 0:
            raise ValueError("SGLang metadata planner geometry must be positive")
        if grouping not in {"request", "tile"}:
            raise ValueError("SGLang metadata grouping must be request or tile")
        self._tier_service = tier_service
        self._execution_config = execution_config
        self._tenant_isolation_enabled = tenant_isolation_enabled
        self._object_capacity = object_capacity
        self._grouping = grouping
        self._page_size = page_size
        self._profile_cpu = profile_cpu
        self._stats = stats

    def plan(
        self,
        *,
        forward_batch: Any,
        wrapper_metadata: Any,
        pending: PendingHostLoad,
        bindings: tuple[RequestBinding, ...],
        engine_batch: EngineBatch,
        host_cost_model: HostCostModel,
        calibration_probe: bool,
        count_batch: bool,
    ) -> SglangMetadataPlan:
        if not bindings or engine_batch.bindings != bindings:
            raise RuntimeError("external metadata has inconsistent request bindings")
        metadata_started = time.perf_counter_ns() if self._profile_cpu else 0
        if forward_batch.forward_mode.is_decode_or_idle():
            wrappers = wrapper_metadata.decode_wrappers
            extractor = decode_schedule
        else:
            if wrapper_metadata.use_ragged:
                raise RuntimeError("NTA requires SGLang paged prefill metadata")
            wrappers = wrapper_metadata.prefill_wrappers
            extractor = paged_prefill_schedule

        def semantic_plan(
            schedule: Schedule,
            dependency_kind: str,
            *,
            page_pairs: tuple[PagePair, ...] = (),
            acquisition_slices: tuple[LeaseAcquisitionSlice | None, ...] = (),
            acquisition_groups: tuple[LeaseAcquisitionGroup | None, ...] = (),
        ) -> _SemanticWrapperPlan:
            started = time.perf_counter_ns() if self._profile_cpu else 0
            result = build_semantic_wrapper_plan(
                engine_batch=engine_batch,
                tile_compute_ns=host_cost_model.tile_compute_ns,
                bindings=bindings,
                schedule=schedule,
                pending=pending,
                dependency_kind=dependency_kind,
                page_pairs=page_pairs,
                acquisition_slices=acquisition_slices,
                acquisition_groups=acquisition_groups,
            )
            self._stats["semantic_wrapper_plan_builds"] += 1
            self._stats["semantic_wrapper_plan_items"] += result.topology.work_count
            if self._profile_cpu:
                self._stats["semantic_wrapper_plan_cpu_ns"] += (
                    time.perf_counter_ns() - started
                )
            return result

        schedules: dict[int, Schedule] = {}
        for wrapper in wrappers:
            schedule = extractor(wrapper)
            validate_schedule(schedule, bindings)
            schedules[id(wrapper)] = schedule
        bounded_direct = self._bounded_direct_plan(
            schedules,
            pending,
            bindings,
            host_cost_model=host_cost_model,
            calibration_probe=calibration_probe,
        )
        if bounded_direct is not None:
            self._record_direct_mixed_heterogeneity(forward_batch, bindings)
            result = SglangMetadataPlan(
                _ActiveBatch(
                    bindings=bindings,
                    semantic_plans={},
                    pending_host_load=pending,
                    host_execution=bounded_direct,
                    grouping=self._grouping,
                ),
                bounded_direct,
            )
            self._stats["host_selection_bound_fastpath_batches"] += 1
            self._finish_profile(metadata_started, count_batch)
            return result

        metadata = forward_metadata(forward_batch)
        if len(metadata.acquisitions) != len(bindings):
            raise RuntimeError(
                "SGLang acquisition metadata does not match request bindings"
            )
        lease_rows = int(pending.device_indices.numel())
        acquisitions = resolve_request_acquisitions(
            metadata.acquisitions,
            pending.transfers_by_operation(),
            lease_transfer_rows=lease_rows,
        )
        sequence_lengths = cpu_sequence_lengths(forward_batch, len(bindings))
        self._record_heterogeneity(bindings, sequence_lengths, acquisitions)
        if self._tier_service.is_host_staged:
            result = self._plan_host_staged(
                forward_batch=forward_batch,
                pending=pending,
                bindings=bindings,
                schedules=schedules,
                sequence_lengths=sequence_lengths,
                acquisitions=acquisitions,
                semantic_plan=semantic_plan,
                lease_rows=lease_rows,
                host_cost_model=host_cost_model,
                calibration_probe=calibration_probe,
            )
        else:
            result = self._plan_physical(
                forward_batch=forward_batch,
                pending=pending,
                bindings=bindings,
                wrappers=wrappers,
                schedules=schedules,
                semantic_plan=semantic_plan,
            )
        self._finish_profile(metadata_started, count_batch)
        return result

    def _bounded_direct_plan(
        self,
        schedules: dict[int, Schedule],
        pending: PendingHostLoad,
        bindings: tuple[RequestBinding, ...],
        *,
        host_cost_model: HostCostModel,
        calibration_probe: bool,
    ) -> HostExecutionPlan | None:
        """Apply the zero-topology fast path unless AUTO is sampling it.

        A calibration epoch must construct and execute one typed observation;
        otherwise an early, fully-published prefetch can permanently bypass
        the only path capable of closing the online setup-cost model.
        """

        if not self._tier_service.is_host_staged or calibration_probe:
            return None
        return prove_direct_metadata_execution(
            schedules,
            pending,
            bindings,
            host_staged=True,
            tenant_isolation=self._tenant_isolation_enabled,
            model=host_cost_model,
            mode=self._execution_config.host_execution_mode,
        )

    def _record_direct_mixed_heterogeneity(
        self,
        forward_batch: Any,
        bindings: tuple[RequestBinding, ...],
    ) -> None:
        """Record mixed direct geometry without constructing typed topology.

        In SGLang mixed mode, decode rows already own resident HBM KV while the
        HiCache lease belongs to the concurrent extend rows.  That framework
        invariant proves availability heterogeneity.  Sequence geometry comes
        from SGLang's existing CPU mirror, so the no-overhead direct path adds
        neither a CUDA readback nor exact acquisition projection.
        """

        if not forward_batch.forward_mode.is_mixed() or len(bindings) < 2:
            return
        sequence_lengths = cpu_sequence_lengths(forward_batch, len(bindings))
        sequence_heterogeneous = len(set(sequence_lengths)) > 1
        self._stats["multi_request_engine_batches"] += 1
        self._stats["heterogeneous_engine_batches"] += 1
        self._stats["availability_heterogeneous_batches"] += 1
        self._stats["sequence_length_heterogeneous_batches"] += int(
            sequence_heterogeneous
        )
        self._stats["multi_axis_heterogeneous_batches"] += int(
            sequence_heterogeneous
        )

    def _plan_host_staged(
        self,
        *,
        forward_batch: Any,
        pending: PendingHostLoad,
        bindings: tuple[RequestBinding, ...],
        schedules: dict[int, Schedule],
        sequence_lengths: tuple[int, ...],
        acquisitions: tuple[SglangAcquisitionSpan, ...],
        semantic_plan: Any,
        lease_rows: int,
        host_cost_model: HostCostModel,
        calibration_probe: bool,
    ) -> SglangMetadataPlan:
        acquisition_slices = {
            wrapper_id: project_acquisition_slices(
                schedule, acquisitions, sequence_lengths
            )
            for wrapper_id, schedule in schedules.items()
        }
        acquisition_groups = {
            wrapper_id: capacity_constrained_acquisition_groups(
                dependencies,
                maximum_groups=self._object_capacity // 2,
            )
            for wrapper_id, dependencies in acquisition_slices.items()
        }
        host_execution = plan_typed_lease_execution(
            schedules,
            acquisition_slices,
            acquisition_groups,
            pending,
            object_capacity=self._object_capacity,
            model=host_cost_model,
            mode=self._execution_config.host_execution_mode,
            calibration_probe=(
                self._execution_config.host_execution_mode
                is HostExecutionMode.AUTO
                and calibration_probe
            ),
            tenant_isolation=self._tenant_isolation_enabled,
        )
        semantic_plans = (
            {
                wrapper_id: semantic_plan(
                    schedule=schedule,
                    dependency_kind="typed_lease",
                    acquisition_slices=acquisition_slices[wrapper_id],
                    acquisition_groups=acquisition_groups[wrapper_id],
                )
                for wrapper_id, schedule in schedules.items()
            }
            if host_execution.uses_dependency_protocol
            else {}
        )
        if forward_batch.forward_mode.is_mixed():
            representative_id = next(iter(schedules))
            schedule = schedules[representative_id]
            dependencies = acquisition_slices[representative_id]
            self._stats["mixed_scheduled_requests"] += len(
                set(schedule.request_indices)
            )
            self._stats["mixed_direct_work_items"] += sum(
                dependency is None for dependency in dependencies
            )
            self._stats["mixed_external_work_items"] += sum(
                dependency is not None for dependency in dependencies
            )
        self._stats["typed_acquisition_batches"] += 1
        self._stats["typed_acquisition_rows"] += lease_rows
        self._stats["typed_acquisition_work_items"] += sum(
            len(dependencies) for dependencies in acquisition_slices.values()
        )
        exact_groups = sum(
            len({item for item in dependencies if item is not None})
            for dependencies in acquisition_slices.values()
        )
        transfer_groups = sum(
            len({item for item in dependencies if item is not None})
            for dependencies in acquisition_groups.values()
        )
        self._stats["typed_exact_dependency_groups"] = self._stats.get(
            "typed_exact_dependency_groups", 0
        ) + exact_groups
        self._stats["typed_transfer_groups"] = self._stats.get(
            "typed_transfer_groups", 0
        ) + transfer_groups
        self._stats["typed_granularity_constrained_batches"] = self._stats.get(
            "typed_granularity_constrained_batches", 0
        ) + int(transfer_groups < exact_groups)
        rows = [
            item.row_count
            for dependencies in acquisition_groups.values()
            for item in dependencies
            if item is not None
        ]
        if not rows:
            raise RuntimeError("typed host plan contains no acquisition groups")
        self._stats["typed_transfer_group_max_rows"] = max(
            self._stats.get("typed_transfer_group_max_rows", 0), max(rows)
        )
        return SglangMetadataPlan(
            _ActiveBatch(
                bindings=bindings,
                semantic_plans=semantic_plans,
                pending_host_load=pending,
                host_execution=host_execution,
                grouping="request",
                lease_transfer_rows=lease_rows,
            ),
            host_execution,
        )

    def _plan_physical(
        self,
        *,
        forward_batch: Any,
        pending: PendingHostLoad,
        bindings: tuple[RequestBinding, ...],
        wrappers: tuple[Any, ...] | list[Any],
        schedules: dict[int, Schedule],
        semantic_plan: Any,
    ) -> SglangMetadataPlan:
        pending_pages = set(pending.materialize_mapping())
        planned_pages: set[int] = set()
        tile_page_pairs: dict[int, tuple[PagePair, ...]] = {}
        for wrapper in wrappers:
            layout = wrapper_page_layout(wrapper, default_page_size=self._page_size)
            planned_pages.update(layout[1])
            tile_page_pairs[id(wrapper)] = work_page_pairs(
                schedules[id(wrapper)],
                pending,
                layout=layout,
                host_staged=False,
                physical_catalog=self._tier_service.catalog,
            )
        missing = pending_pages - planned_pages
        if missing:
            raise RuntimeError(
                f"attention metadata omits {len(missing)} promoted HiCache pages"
            )
        mixed_forward = forward_batch.forward_mode.is_mixed()
        request_page_pairs = (
            {
                wrapper_id: group_external_pages_by_request(
                    schedules[wrapper_id], pairs
                )
                for wrapper_id, pairs in tile_page_pairs.items()
            }
            if self._grouping == "request" or mixed_forward
            else None
        )
        if mixed_forward:
            if request_page_pairs is None:  # pragma: no cover
                raise RuntimeError("mixed execution omitted request grouping")
            representative_id = next(iter(schedules))
            schedule = schedules[representative_id]
            pairs = request_page_pairs[representative_id]
            self._stats["mixed_scheduled_requests"] += len(
                set(schedule.request_indices)
            )
            self._stats["mixed_direct_work_items"] += sum(not pair[0] for pair in pairs)
            self._stats["mixed_external_work_items"] += sum(bool(pair[0]) for pair in pairs)
        if self._grouping == "tile":
            page_pairs = tile_page_pairs
        else:
            if request_page_pairs is None:  # pragma: no cover
                raise RuntimeError("request execution omitted request grouping")
            page_pairs = request_page_pairs
        return SglangMetadataPlan(
            _ActiveBatch(
                bindings=bindings,
                semantic_plans={
                    wrapper_id: semantic_plan(
                        schedule=schedule,
                        dependency_kind="physical_pages",
                        page_pairs=page_pairs[wrapper_id],
                    )
                    for wrapper_id, schedule in schedules.items()
                },
                pending_host_load=pending,
                host_execution=None,
                grouping=self._grouping,
            ),
            None,
        )

    def _record_heterogeneity(
        self,
        bindings: tuple[RequestBinding, ...],
        sequence_lengths: tuple[int, ...],
        acquisitions: tuple[SglangAcquisitionSpan, ...],
    ) -> None:
        if len(bindings) < 2:
            return
        self._stats["multi_request_engine_batches"] += 1
        axes = request_batch_heterogeneity(bindings, sequence_lengths, acquisitions)
        if not axes:
            return
        self._stats["heterogeneous_engine_batches"] += 1
        self._stats["multi_axis_heterogeneous_batches"] += int(len(axes) > 1)
        for axis in axes:
            self._stats[f"{axis}_heterogeneous_batches"] += 1

    def _finish_profile(self, started_ns: int, count_batch: bool) -> None:
        if self._profile_cpu:
            self._stats["metadata_cpu_ns"] = self._stats.get(
                "metadata_cpu_ns", 0
            ) + (time.perf_counter_ns() - started_ns)
        if count_batch:
            self._stats["batches"] += 1
            self._stats["hicache_external_batches"] += 1
