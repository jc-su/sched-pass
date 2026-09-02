"""Telemetry ownership for the SGLang integration.

The numerical adapter records counters, but it does not define their schema or
own report I/O. This module keeps immutable setup identity, mutable counter
initialization, consumer classification, and asynchronous publication behind
one boundary. It deliberately imports neither SGLang nor CUDA.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
import pathlib
import threading
import time
from typing import Any

from nta_runtime.adapters.base import ConsumerContract


class _ProcessHookTelemetry:
    """Thread-safe ownership for counters produced outside one backend.

    SGLang hooks are installed once per worker process and may observe several
    attention backend instances. Their counters are therefore explicitly
    process-scoped; backend reports consume immutable snapshots rather than
    sharing mutable dictionaries with hook callbacks.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, float | int] = {
            "prefill_graph_served_batches": 0,
            "prefill_graph_capture_batches": 0,
        }

    def increment(self, key: str, value: float | int = 1) -> float | int:
        if not key or value < 0:
            raise ValueError("hook telemetry increments must be nonnegative")
        with self._lock:
            updated = self._values.get(key, 0) + value
            self._values[key] = updated
            return updated

    def maximum(self, key: str, value: float) -> None:
        if not key or value < 0:
            raise ValueError("hook telemetry maxima must be nonnegative")
        with self._lock:
            self._values[key] = max(float(self._values.get(key, 0.0)), value)

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            return dict(self._values)


_PROCESS_HOOK_TELEMETRY = _ProcessHookTelemetry()


@dataclass(frozen=True, slots=True)
class SglangTelemetryConfig:
    """Immutable deployment identity recorded in every engine report."""

    model_layer_count: int
    execution_protocol: str
    host_execution_mode: str
    work_granularity: str
    protocol_max_inflight_units: int
    runtime_tenant_capacity: int
    runtime_staging_byte_capacity: int
    tenant_isolation_enabled: bool
    overlap_enabled: bool
    frontier_enabled: bool
    fragment_enabled: bool
    demand_overlap_policy: str
    stream_ordered_retirement_enabled: bool
    sglang_mixed_chunk_enabled: bool
    max_host_rounds: int
    minimum_predicted_gain: float
    incremental_setup_ns: int | None
    incremental_service_scale: float | None
    incremental_calibration_probes_remaining: int
    cost_model_bandwidth_bps: int
    host_mover: str
    copy_engine_max_operations: int
    host_mover_copy_calibrated: bool
    host_mover_calibration_samples_per_engine: int
    host_mover_sm_samples: int
    host_mover_copy_samples: int
    host_mover_sm_bandwidth_bps: int
    host_mover_copy_bandwidth_bps: int | None
    host_mover_copy_operation_ns: int | None
    host_mover_hybrid_join_ns: int
    host_mover_minimum_gain: float
    layer_service_minimum_samples: int
    layer_service_maximum_samples: int
    indexed_copy_target_bytes: int
    indexed_copy_max_blocks: int
    frontier_layers_per_wave: int
    sm_acquisition_waves: int
    sm_mover_max_worker_ctas: int
    demand_graph_enabled: bool
    demand_graph_capacity: int
    engine_version: str
    revision: str

    def __post_init__(self) -> None:
        positive = (
            self.model_layer_count,
            self.protocol_max_inflight_units,
            self.runtime_tenant_capacity,
            self.cost_model_bandwidth_bps,
            self.copy_engine_max_operations,
            self.host_mover_calibration_samples_per_engine,
            self.layer_service_minimum_samples,
            self.layer_service_maximum_samples,
            self.indexed_copy_target_bytes,
            self.indexed_copy_max_blocks,
            self.frontier_layers_per_wave,
            self.sm_acquisition_waves,
            self.sm_mover_max_worker_ctas,
            self.demand_graph_capacity,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("SGLang telemetry geometry must be positive")
        if self.layer_service_maximum_samples < self.layer_service_minimum_samples:
            raise ValueError("SGLang telemetry service sample bounds are invalid")
        if (
            min(
                self.runtime_staging_byte_capacity,
                self.max_host_rounds,
                self.incremental_calibration_probes_remaining,
                self.host_mover_sm_samples,
                self.host_mover_copy_samples,
                self.host_mover_hybrid_join_ns,
            )
            < 0
        ):
            raise ValueError("SGLang telemetry counters cannot be negative")
        if not all(
            value
            for value in (
                self.execution_protocol,
                self.host_execution_mode,
                self.work_granularity,
                self.demand_overlap_policy,
                self.host_mover,
                self.engine_version,
                self.revision,
            )
        ):
            raise ValueError("SGLang telemetry identity cannot be empty")


_ZERO_COUNTERS = (
    "host_direct_batches",
    "host_scheduled_bulk_batches",
    "stock_scheduled_frontier_batches",
    "shared_acquisition_registered_groups",
    "shared_acquisition_registered_cohorts",
    "shared_acquisition_ready_groups",
    "shared_acquisition_ready_cohorts",
    "shared_acquisition_resource_skipped_cohorts",
    "shared_acquisition_resource_blocked_cohorts",
    "shared_acquisition_submitted_packets",
    "shared_acquisition_submitted_groups",
    "shared_acquisition_submitted_cohorts",
    "shared_acquisition_retired_cohorts",
    "shared_acquisition_cancelled_leases",
    "shared_acquisition_publication_waits",
    "shared_acquisition_publication_wait_rounds",
    "host_device_bulk_batches",
    "host_incremental_batches",
    "host_typed_mixed_batches",
    "host_mixed_direct_batches",
    "host_mixed_scheduled_bulk_batches",
    "host_bound_after_full_ready_batches",
    "host_typed_after_full_publication_batches",
    "host_acquisition_jobs_prepared",
    "host_acquisition_jobs_submitted",
    "host_acquisition_submission_calls",
    "host_acquisition_models_bound",
    "host_acquisition_structural_owners",
    "host_acquisition_layers_consumed",
    "host_acquisition_refill_jobs",
    "host_acquisition_shape_uncalibrated",
    "host_acquisition_mover_uncalibrated",
    "host_acquisition_model_rejected",
    "lease_acquisition_groups_prepared",
    "lease_acquisition_groups_started",
    "metadata_acquisition_groups_prepared",
    "host_selection_predicted_atomic_ns",
    "host_selection_predicted_selected_ns",
    "host_selection_bound_fastpath_batches",
    "stream_ordered_retirement_layers",
    "stream_ordered_retirement_launches",
    "stream_ordered_retirement_batches",
    "stream_ordered_prefetch_events",
    "stream_ordered_prefetch_event_reuses",
    "incremental_setup_samples",
    "incremental_initialization_samples",
    "incremental_initialization_setup_ns",
    "incremental_service_samples",
    "cost_model_transfer_samples",
    "native_demand_sm_bytes",
    "prefetch_mover_plan_calibration_probe_sm_leases",
    "prefetch_mover_plan_calibration_probe_copy_leases",
    "host_mover_profiled_sm_bytes",
    "host_mover_profiled_copy_bytes",
    "host_mover_sm_calibrated_buckets",
    "host_mover_copy_calibrated_buckets",
    "host_mover_sm_max_sample_bytes",
    "host_mover_copy_max_sample_bytes",
    "host_mover_predicted_sm_ns",
    "host_mover_predicted_selected_ns",
    "host_mover_complete_calibration_frontiers",
    "host_mover_complete_calibration_wave_samples",
    "host_mover_overlap_compute_ns",
    "host_mover_overlap_profiled_leases",
    "host_mover_overlap_profiled_copy_ns",
    "host_mover_overlap_profiled_compute_ns",
    "host_mover_overlap_profiled_concurrent_ns",
    "layer_service_profiled_intervals",
    "layer_service_calibrated_shapes",
    "layer_service_conservative_ns",
    "layer_service_plan_key_missing_batches",
    "layer_service_plan_curve_missing_batches",
    "layer_service_plan_curve_uncalibrated_batches",
    "layer_service_plan_curve_calibrated_batches",
    "layer_service_retirement_commits",
    "prefetch_mover_plan_uncalibrated_copy_engine_leases",
    "prefetch_mover_plan_frozen_uncalibrated_sm_leases",
    "prefetch_mover_plan_frozen_uncalibrated_copy_engine_leases",
    "prefetch_mover_plan_frozen_uncalibrated_overlap_leases",
    "prefetch_mover_plan_insufficient_gain_leases",
    "prefetch_mover_plan_service_cost_leases",
    "prefetch_mover_plan_execution_context_unbound_leases",
    "prefetch_mover_plan_sm_leases",
    "prefetch_mover_plan_copy_engine_leases",
    "prefetch_mover_plan_hybrid_leases",
    "copy_engine_waves",
    "copy_engine_submissions",
    "copy_engine_operations",
    "copy_engine_issue_cpu_ns",
    "hybrid_parallel_waves",
    "prefetch_mover_plan_copy_runs",
    "prefetch_mover_plan_copy_rows",
    "copy_engine_bytes",
    "copy_engine_layout_cpu_ns",
    "sm_mover_bytes",
    "sm_acquisition_wave_submissions",
    "sm_mover_worker_ctas",
    "sm_mover_throttled_submissions",
    "prefetch_mover_plan_sm_rows",
    "batches",
    "decode_launches",
    "prefill_launches",
    "cta_work_items",
    "plan_uploads",
    "request_rebindings",
    "request_cancellations",
    "request_retirements",
    "forward_lifecycle_completions",
    "forward_lifecycle_aborts",
    "external_launches",
    "native_external_attention_launches",
    "native_dispatch_prefix_observations",
    "native_dispatch_nonprefix_batches",
    "progressive_consumer_batch_observations",
    "progressive_consumer_batches",
    "progressive_consumer_layers",
    "device_bulk_layers",
    "exact_resume_window_layers",
    "execution_template_builds",
    "execution_template_reuses",
    "deadline_frontier_model_builds",
    "deadline_frontier_model_reuses",
    "deadline_frontier_plan_builds",
    "deadline_frontier_plan_reuses",
    "deadline_frontier_full_recomputes",
    "deadline_frontier_plans",
    "deadline_frontier_noop_calls",
    "deadline_frontier_published_layers",
    "deadline_frontier_modeled_ready_layers",
    "deadline_frontier_modeled_stock_dispatches",
    "deadline_frontier_fragment_layers",
    "partial_consumer_unproven_layers",
    "partial_consumer_planned_layers",
    "consumer_policy_profiled_leases",
    "consumer_policy_probe_leases",
    "consumer_policy_frozen_profile_leases",
    "consumer_policy_frozen_conservative_leases",
    "consumer_policy_probe_misses",
    "consumer_policy_rejected_shapes",
    "consumer_policy_planned_layers",
    "consumer_policy_arrival_samples",
    "consumer_policy_stock_samples",
    "consumer_policy_partial_samples",
    "consumer_policy_partial_setup_samples",
    "consumer_policy_partial_reuse_samples",
    "calibration_frozen_setup_observations",
    "resident_reference_batches",
    "resident_reference_metadata_calls",
    "resident_reference_metadata_cpu_ns",
    "resident_reference_metadata_stock_cpu_ns",
    "resident_reference_metadata_overhead_cpu_ns",
    "hicache_external_batches",
    "hicache_fallback_batches",
    "indexed_host_objects",
    "request_acquisition_groups",
    "tile_acquisition_groups",
    "indexed_host_bytes",
    "prefetched_layers",
    "prefetched_host_bytes",
    "tier_selected_leases",
    "tier_selected_rows",
    "tier_selected_bytes",
    "tier_candidate_bytes",
    "lookahead_acquisition_layers",
    "lookahead_acquisition_objects",
    "lookahead_bound_launches",
    "arriving_prefetch_layers",
    "arriving_prefetch_launches",
    "arriving_plan_preparations",
    "arriving_partition_preparations",
    "arriving_partition_reuses",
    "event_ordered_incremental_launches",
    "event_ordered_wave_launches",
    "typed_acquisition_batches",
    "typed_acquisition_rows",
    "typed_acquisition_work_items",
    "demand_host_layers",
    "incremental_host_layers",
    "request_overlap_layers",
    "mixed_dependency_layers",
    "mixed_forward_batches",
    "mixed_forward_requests",
    "mixed_scheduled_requests",
    "mixed_direct_work_items",
    "mixed_external_work_items",
    "multi_request_engine_batches",
    "heterogeneous_engine_batches",
    "multi_axis_heterogeneous_batches",
    "sequence_length_heterogeneous_batches",
    "availability_heterogeneous_batches",
    "external_rows_heterogeneous_batches",
    "tenant_heterogeneous_batches",
    "priority_heterogeneous_batches",
    "deadline_heterogeneous_batches",
    "transformed_direct_launches",
    "ticketed_incremental_launches",
    "stock_attention_launches",
    "stock_resident_batches",
    "stock_resident_attention_launches",
    "stock_prefetched_external_batches",
    "stock_prefetched_external_attention_launches",
    "host_progress_rounds",
    "parallel_indexed_progress_layers",
    "indexed_range_fastpath_layers",
    "unqueued_host_discovery_layers",
    "queued_feasible_edf_layers",
    "fragment_lookahead_layers",
    "fragment_lookahead_objects",
    "fragment_lookahead_bytes",
    "fragment_remaining_rounds",
    "compact_initial_launches",
    "compact_initial_cta_bound",
    "canonical_initial_cta_bound",
    "compact_resume_launches",
    "compact_resume_cta_bound",
    "canonical_resume_cta_bound",
    "predicted_atomic_ns",
    "predicted_incremental_ns",
    "progress_snapshots",
    "request_work_completed",
    "request_work_failed",
    "request_compute_completed_ns",
    "graph_captures",
    "graph_replays",
    "graph_external_batches",
    "demand_graph_warmups",
    "demand_graph_captures",
    "demand_graph_replays",
    "demand_graph_evictions",
    "verified_operator_modules",
    "semantic_wrapper_plan_builds",
    "semantic_wrapper_plan_lookups",
    "semantic_wrapper_plan_cpu_ns",
    "semantic_wrapper_plan_lookup_cpu_ns",
    "semantic_wrapper_plan_items",
    "semantic_verifier_plan_builds",
    "semantic_plan_cpu_ns",
    "semantic_verifier_sessions",
    "semantic_dense_tiles",
    "nvme_progress_rounds",
    "nvme_bytes",
    "nvme_epochs",
    "tier_external_layers",
    "tier_host_proxy_bytes",
    "indexed_object_quiesced_registrations",
    "indexed_object_lifetime_guard_fallbacks",
    "nvme_view_publications",
    "nvme_same_destination_installs",
    "nvme_destination_rebinds",
    "nvme_fresh_slot_installs",
    "nvme_object_quiesced_replacements",
    "nvme_region_prepare_ns",
    "nvme_region_count",
    "nvme_region_bytes",
    "nvme_destination_slice_count",
    "nvme_destination_slice_bytes",
    "nvme_shared_region_slices",
)

_ZERO_FLOAT_COUNTERS = (
    "host_mover_profiled_sm_gpu_ms",
    "host_mover_profiled_copy_gpu_ms",
)

# These fields are current capabilities or model state, not work performed in
# an interval. Every other zero-initialized field is a monotone cumulative
# counter whose ownership belongs to this producer schema.
_ENGINE_GAUGE_FIELDS = frozenset(
    {
        "host_mover_sm_calibrated_buckets",
        "host_mover_copy_calibrated_buckets",
        "host_mover_sm_max_sample_bytes",
        "host_mover_copy_max_sample_bytes",
        "layer_service_calibrated_shapes",
        "layer_service_conservative_ns",
        "verified_operator_modules",
    }
)

_CUMULATIVE_COUNTER_FIELDS = tuple(
    field
    for field in (*_ZERO_COUNTERS, *_ZERO_FLOAT_COUNTERS)
    if field not in _ENGINE_GAUGE_FIELDS
)


def initial_engine_stats(
    config: SglangTelemetryConfig, tier_stats: Mapping[str, Any]
) -> dict[str, Any]:
    """Create one complete, duplicate-free report schema."""

    if len(_ZERO_COUNTERS) != len(set(_ZERO_COUNTERS)):
        raise RuntimeError("SGLang telemetry counter schema contains duplicates")
    zero_fields = set(_ZERO_COUNTERS) | set(_ZERO_FLOAT_COUNTERS)
    if not _ENGINE_GAUGE_FIELDS.issubset(zero_fields):
        raise RuntimeError("SGLang telemetry gauge schema contains unknown fields")
    result: dict[str, Any] = {
        "schema": 1,
        "engine": "sglang",
        "backend": "nta_flashinfer",
        "model_layer_count": config.model_layer_count,
        "execution_protocol": config.execution_protocol,
        "host_execution_mode": config.host_execution_mode,
        "work_granularity": config.work_granularity,
        "protocol_max_inflight_units": config.protocol_max_inflight_units,
        "runtime_tenant_capacity": config.runtime_tenant_capacity,
        "runtime_staging_byte_capacity": config.runtime_staging_byte_capacity,
        "tenant_isolation_enabled": config.tenant_isolation_enabled,
        "execution_protocol_status": "projection_only",
        "execution_demand_semantics": "exact",
        "execution_plan_scope": "attention_launch",
        "python_availability_state_machine": "verify_only",
        "consumer_contract": ConsumerContract.projection_only(
            engine="sglang",
            backend="nta_flashinfer",
            engine_version=config.engine_version,
        ).as_dict(),
        "revision": config.revision,
        "pid": os.getpid(),
        "host_execution_selection": "measured_direct_or_incremental",
        "overlap_enabled": config.overlap_enabled,
        "frontier_enabled": config.frontier_enabled,
        # Layer deadlines are strictly ordered by transformer execution, so
        # EDF proves feasibility but does not create a different layer order.
        # Fine-grained request/group arbitration lives in the typed runtime.
        "layer_scheduler": "structural_layer_order",
        "layer_feasibility_test": "simultaneous_release_edf",
        "fragment_enabled": config.fragment_enabled,
        "demand_overlap_policy": config.demand_overlap_policy,
        "stream_ordered_retirement_enabled": config.stream_ordered_retirement_enabled,
        "sglang_mixed_chunk_enabled": config.sglang_mixed_chunk_enabled,
        "max_host_rounds": config.max_host_rounds,
        "minimum_predicted_gain": config.minimum_predicted_gain,
        "incremental_setup_ns": config.incremental_setup_ns,
        "incremental_setup_calibrated": config.incremental_setup_ns is not None,
        "incremental_service_scale": config.incremental_service_scale,
        "incremental_service_calibrated": (
            config.incremental_service_scale is not None
        ),
        "incremental_calibration_probes_remaining": (
            config.incremental_calibration_probes_remaining
        ),
        "cost_model_bandwidth_bps": config.cost_model_bandwidth_bps,
        # Proactive publication and demand acquisition are separate planes.
        "host_mover": config.host_mover,
        "prefetch_mover_policy": config.host_mover,
        "native_demand_mover": "sm_indexed_device",
        "copy_engine_max_operations": config.copy_engine_max_operations,
        "host_mover_copy_calibrated": config.host_mover_copy_calibrated,
        "host_mover_calibration_samples_per_engine": (
            config.host_mover_calibration_samples_per_engine
        ),
        "host_mover_sm_samples": config.host_mover_sm_samples,
        "host_mover_copy_samples": config.host_mover_copy_samples,
        "host_mover_sm_bandwidth_bps": config.host_mover_sm_bandwidth_bps,
        "host_mover_copy_bandwidth_bps": config.host_mover_copy_bandwidth_bps,
        "host_mover_copy_operation_ns": config.host_mover_copy_operation_ns,
        "host_mover_hybrid_join_ns": config.host_mover_hybrid_join_ns,
        "host_mover_minimum_gain": config.host_mover_minimum_gain,
        "host_mover_service_curves": [],
        "layer_service_minimum_samples": config.layer_service_minimum_samples,
        "layer_service_maximum_samples": config.layer_service_maximum_samples,
        "indexed_copy_target_bytes": config.indexed_copy_target_bytes,
        "indexed_copy_max_blocks": config.indexed_copy_max_blocks,
        "frontier_layers_per_wave": config.frontier_layers_per_wave,
        "sm_acquisition_waves": config.sm_acquisition_waves,
        "sm_mover_max_worker_ctas": config.sm_mover_max_worker_ctas,
        "demand_graph_enabled": config.demand_graph_enabled,
        "demand_graph_capacity": config.demand_graph_capacity,
        "transport_program_loaded": False,
        # Measurement consumers subtract exactly these fields at timed-window
        # boundaries. Publishing the schema with the values prevents a second,
        # drifting allow-list in every benchmark harness.
        "cumulative_counter_fields": list(_CUMULATIVE_COUNTER_FIELDS),
        "started_unix_ns": time.time_ns(),
    }
    result.update(dict.fromkeys(_ZERO_COUNTERS, 0))
    result.update(dict.fromkeys(_ZERO_FLOAT_COUNTERS, 0.0))
    overlap = set(result).intersection(tier_stats)
    if overlap:
        raise RuntimeError(
            f"tier telemetry collides with engine fields: {sorted(overlap)}"
        )
    result.update(tier_stats)
    return result


class StatsPublisher:
    """Coalesce evaluation snapshots and write them off the scheduler thread."""

    def __init__(self, path: pathlib.Path) -> None:
        self._path = path
        self._condition = threading.Condition()
        self._pending: tuple[int, dict[str, Any]] | None = None
        self._submitted = 0
        self._completed = 0
        self._error: Exception | None = None
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run, name="nta-stats-publisher", daemon=True
        )
        self._thread.start()

    def publish(self, report: dict[str, Any], *, wait: bool = False) -> None:
        with self._condition:
            if self._stopping:
                raise RuntimeError("NTA engine statistics publisher is closed")
            if self._error is not None:
                raise RuntimeError(
                    "NTA engine statistics publisher failed"
                ) from self._error
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

    def close(self) -> None:
        """Stop the writer thread after all already-published work settles."""

        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._pending is not None or self._stopping
                )
                if self._pending is None and self._stopping:
                    return
                assert self._pending is not None
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


def flag_value(value: Any) -> int:
    """Serialize both ``IntFlag`` and ``Flag`` values as one mask."""

    return int(getattr(value, "value", value))


def record_forward(kind: str, milliseconds: float) -> None:
    """Accumulate count, total, and maximum for one forward-kind sample."""

    if not kind or milliseconds < 0:
        raise ValueError("forward telemetry sample is invalid")
    _PROCESS_HOOK_TELEMETRY.increment(f"forward_{kind}_count", 1.0)
    _PROCESS_HOOK_TELEMETRY.increment(f"forward_{kind}_ms_total", float(milliseconds))
    _PROCESS_HOOK_TELEMETRY.maximum(f"forward_{kind}_ms_max", float(milliseconds))


def record_prefill_graph(kind: str) -> None:
    """Record a process-wide prefill graph hook observation."""

    if kind not in {"served", "capture"}:
        raise ValueError(f"unknown prefill graph observation {kind!r}")
    _PROCESS_HOOK_TELEMETRY.increment(f"prefill_graph_{kind}_batches")


def record_observability_degraded(site: str) -> int:
    """Count an observation failure and return its one-based occurrence."""

    return int(_PROCESS_HOOK_TELEMETRY.increment(f"observability_degraded_{site}"))


def process_hook_stats() -> dict[str, float | int]:
    """Return an immutable snapshot of process-scoped hook counters."""

    return _PROCESS_HOOK_TELEMETRY.snapshot()


def consumer_contract_for_stats(
    stats: Mapping[str, Any], *, engine_version: str
) -> ConsumerContract:
    """Classify the numerical consumer represented by one engine report."""

    native_launches = (
        int(stats.get("transformed_direct_launches", 0))
        + int(stats.get("ticketed_incremental_launches", 0))
        + int(stats.get("event_ordered_incremental_launches", 0))
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
