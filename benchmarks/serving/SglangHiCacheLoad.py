#!/usr/bin/env python3
"""Run a placement-proven mixed HiCache load through an in-process SGLang engine."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, replace
import hashlib
import importlib.metadata
import json
import math
import os
import pathlib
import platform
import random
import subprocess
import sys
import time
from typing import Any, Protocol, Sequence

try:
    from experiments.bailian import (
        demand_trace_digest,
        input_page_ids,
        read_jsonl,
        unique_input_page_ids,
    )
    from experiments.cache_identity import effective_cached_prefixes
    from experiments.cache_evolution import annotate_timed_cache_bindings
    from experiments.atomic_io import atomic_write_json
    from experiments.queueing import finite_window_system_accounting
    from experiments.serving_metrics import joint_slo_goodput
    from experiments.validate_workload import validate as validate_workload
    from experiments.workload_heterogeneity import serving_batch_heterogeneity
except ModuleNotFoundError:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from experiments.bailian import (
        demand_trace_digest,
        input_page_ids,
        read_jsonl,
        unique_input_page_ids,
    )
    from experiments.cache_identity import effective_cached_prefixes
    from experiments.cache_evolution import annotate_timed_cache_bindings
    from experiments.atomic_io import atomic_write_json
    from experiments.queueing import finite_window_system_accounting
    from experiments.serving_metrics import joint_slo_goodput
    from experiments.validate_workload import validate as validate_workload
    from experiments.workload_heterogeneity import serving_batch_heterogeneity

from SglangHiCache import (
    configure_environment,
    device_cached_tokens,
    generated_text,
    generation_results,
    git_value,
    host_cached_tokens,
    make_prompt,
)


_HICACHE_WRITE_POLICY = "write_through_selective"
_CALIBRATION_CPU_AFFINITY_ENV = "NTA_EXECUTION_CALIBRATION_CPU_AFFINITY"


_TIMED_AUTO_CALIBRATION_COUNTERS = (
    "host_selection_calibration_probe_batches",
    "host_selection_consumer_policy_probe_batches",
    "consumer_policy_profiled_leases",
    "consumer_policy_probe_leases",
    "consumer_policy_arrival_samples",
    "consumer_policy_stock_samples",
    "consumer_policy_partial_samples",
    "consumer_policy_partial_reuse_samples",
    "consumer_policy_partial_setup_samples",
    "layer_service_profiled_intervals",
    "incremental_initialization_samples",
    "incremental_setup_samples",
    "incremental_service_samples",
    "cost_model_transfer_samples",
    "prefetch_mover_plan_calibration_probe_copy_leases",
    "prefetch_mover_plan_calibration_probe_sm_leases",
    "host_mover_overlap_profiled_leases",
)


_MEASUREMENT_COUNTERS = frozenset(
    {
        "batches",
        "canonical_initial_cta_bound",
        "canonical_resume_cta_bound",
        "compact_initial_cta_bound",
        "compact_initial_launches",
        "compact_resume_cta_bound",
        "compact_resume_launches",
        "copy_engine_bytes",
        "copy_engine_issue_cpu_ns",
        "copy_engine_layout_cpu_ns",
        "copy_engine_operations",
        "copy_engine_submissions",
        "copy_engine_waves",
        "cost_model_transfer_samples",
        "admission_acquisition_groups_prepared",
        "admission_acquisition_groups_started",
        "cta_work_items",
        "decode_launches",
        "demand_graph_captures",
        "demand_graph_evictions",
        "demand_graph_paged_prefill_warmups",
        "demand_graph_replays",
        "demand_graph_warmups",
        "demand_host_layers",
        "direct_staging_bytes",
        "direct_staging_launches",
        "demand_graph_paged_prefill_captures",
        "demand_graph_paged_prefill_replays",
        "exact_resume_window_layers",
        "external_launches",
        "fragment_remaining_rounds",
        "native_dispatch_prefix_observations",
        "native_dispatch_nonprefix_batches",
        "native_demand_sm_bytes",
        "progressive_consumer_batch_observations",
        "progressive_consumer_batches",
        "progressive_consumer_layers",
        "resident_reference_metadata_calls",
        "resident_reference_metadata_cpu_ns",
        "resident_reference_metadata_stock_cpu_ns",
        "resident_reference_metadata_overhead_cpu_ns",
        "fragment_lookahead_layers",
        "fragment_lookahead_objects",
        "fragment_lookahead_bytes",
        "graph_capture_dummy_rows",
        "graph_captures",
        "graph_external_batches",
        "graph_replays",
        "hicache_external_batches",
        "hicache_fallback_batches",
        "host_direct_batches",
        "host_scheduled_bulk_batches",
        "host_device_bulk_batches",
        "host_bound_after_full_ready_batches",
        "host_typed_after_full_publication_batches",
        "host_acquisition_jobs_prepared",
        "host_acquisition_jobs_submitted",
        "host_acquisition_submission_calls",
        "host_acquisition_models_bound",
        "host_acquisition_layers_consumed",
        "host_acquisition_refill_jobs",
        "host_acquisition_shape_uncalibrated",
        "host_acquisition_mover_uncalibrated",
        "host_acquisition_model_rejected",
        "lease_acquisition_groups_prepared",
        "lease_acquisition_groups_started",
        "metadata_acquisition_groups_prepared",
        "host_incremental_batches",
        "host_mixed_direct_batches",
        "host_mixed_scheduled_bulk_batches",
        "host_typed_mixed_batches",
        "host_progress_rounds",
        "hybrid_parallel_waves",
        "indexed_host_bytes",
        "indexed_host_objects",
        "indexed_range_fastpath_layers",
        "unqueued_host_discovery_layers",
        "indexed_layout_candidate_bytes",
        "indexed_layout_eligible_rows",
        "indexed_layout_profile_cpu_ns",
        "indexed_layout_profiles",
        "indexed_layout_rows",
        "indexed_layout_runs",
        "indexed_object_lifetime_guard_fallbacks",
        "indexed_object_quiesced_registrations",
        "initial_acquisition_batches",
        "initial_acquisition_layers",
        "initial_typed_gap_layers",
        "incremental_host_layers",
        "incremental_initialization_samples",
        "incremental_initialization_setup_ns",
        "incremental_setup_observed_ns_total",
        "incremental_setup_samples",
        "lookahead_acquisition_layers",
        "lookahead_acquisition_objects",
        "lookahead_bound_launches",
        "lookahead_copy_waves",
        "metadata_cpu_ns",
        "mixed_dependency_layers",
        "mixed_direct_work_items",
        "mixed_external_work_items",
        "mixed_forward_batches",
        "mixed_forward_requests",
        "mixed_scheduled_requests",
        "multi_request_engine_batches",
        "heterogeneous_engine_batches",
        "multi_axis_heterogeneous_batches",
        "sequence_length_heterogeneous_batches",
        "availability_heterogeneous_batches",
        "external_rows_heterogeneous_batches",
        "tenant_heterogeneous_batches",
        "priority_heterogeneous_batches",
        "queued_feasible_edf_layers",
        "deadline_heterogeneous_batches",
        "native_external_attention_launches",
        "nvme_numerical_alias_bytes",
        "nvme_numerical_alias_objects",
        "nvme_bytes",
        "nvme_destination_rebinds",
        "nvme_destination_slice_bytes",
        "nvme_destination_slice_count",
        "nvme_epochs",
        "nvme_fresh_slot_installs",
        "nvme_object_quiesced_replacements",
        "nvme_progress_rounds",
        "nvme_region_bytes",
        "nvme_region_count",
        "nvme_region_prepare_ns",
        "nvme_same_destination_installs",
        "nvme_shared_region_slices",
        "nvme_view_publications",
        "paired_lookahead_layers",
        "parallel_indexed_progress_layers",
        "pipeline_cpu_ns",
        "plan_cpu_ns",
        "plan_uploads",
        "prefetched_host_bytes",
        "prefetched_layers",
        "prefetch_mover_plan_calibration_probe_copy_leases",
        "prefetch_mover_plan_calibration_probe_sm_leases",
        "prefetch_mover_plan_copy_engine_leases",
        "prefetch_mover_plan_copy_rows",
        "prefetch_mover_plan_copy_runs",
        "prefetch_mover_plan_hybrid_leases",
        "prefetch_mover_plan_insufficient_gain_leases",
        "prefetch_mover_plan_service_cost_leases",
        "prefetch_mover_plan_sm_leases",
        "prefetch_mover_plan_sm_rows",
        "prefetch_mover_plan_uncalibrated_copy_engine_leases",
        "prefill_graph_capture_batches",
        "prefill_graph_served_batches",
        "prefill_launches",
        "profiled_attention_arrivals",
        "profiled_attention_materially_stalled_arrivals",
        "profiled_attention_not_ready_at_arrival",
        "profiled_attention_ready_at_arrival",
        "profiled_attention_stall_gpu_ms",
        "prevalidated_indexed_progress_layers",
        "progress_feedback_skipped_noncontiguous",
        "progress_feedback_snapshots",
        "progress_snapshots",
        "ready_stock_wrapper_pairs",
        "request_acquisition_groups",
        "request_cancellations",
        "request_compute_expected_ns",
        "request_retirements",
        "request_compute_completed_ns",
        "request_metadata_updates",
        "request_overlap_layers",
        "request_rebindings",
        "request_work_completed",
        "request_work_failed",
        "deadline_frontier_plans",
        "deadline_frontier_published_layers",
        "deadline_frontier_fragment_layers",
        "deadline_frontier_model_builds",
        "deadline_frontier_model_reuses",
        "deadline_frontier_uncalibrated",
        "deadline_frontier_cpu_ns",
        "layer_service_plan_key_missing_batches",
        "layer_service_plan_curve_missing_batches",
        "layer_service_plan_curve_uncalibrated_batches",
        "layer_service_plan_curve_calibrated_batches",
        "layer_service_retirement_commits",
        "layer_service_profiled_intervals",
        "resident_reference_batches",
        "semantic_wrapper_plan_builds",
        "semantic_wrapper_plan_lookups",
        "semantic_wrapper_plan_cpu_ns",
        "semantic_wrapper_plan_lookup_cpu_ns",
        "semantic_wrapper_plan_items",
        "semantic_dense_tiles",
        "semantic_verifier_plan_builds",
        "semantic_plan_cpu_ns",
        "semantic_verifier_sessions",
        "schedule_bound_acquisition_batches",
        "sm_mover_bytes",
        "stock_attention_launches",
        "stock_prefetch_metadata_fastpath_batches",
        "stock_prefetched_external_attention_launches",
        "stream_ordered_prefetch_events",
        "stream_ordered_prefetch_event_reuses",
        "stock_prefetched_external_batches",
        "stock_ready_external_attention_launches",
        "stock_resident_attention_launches",
        "stock_resident_batches",
        "stream_ordered_retirement_batches",
        "stream_ordered_retirement_layers",
        "stream_ordered_retirement_launches",
        "ticketed_incremental_launches",
        "event_ordered_incremental_launches",
        "event_ordered_wave_launches",
        "sm_acquisition_wave_submissions",
        "tier_external_layers",
        "tier_host_proxy_bytes",
        "tier_candidate_bytes",
        "tier_selected_bytes",
        "tier_selected_leases",
        "tier_selected_rows",
        "tile_acquisition_groups",
        "transformed_direct_launches",
        "typed_acquisition_batches",
        "typed_acquisition_rows",
        "typed_acquisition_work_items",
        "typed_exact_dependency_groups",
        "typed_granularity_constrained_batches",
        "typed_transfer_groups",
        "reused_flashinfer_plans",
        "arriving_prefetch_launches",
        "arriving_prefetch_layers",
        "arriving_plan_preparations",
        "arriving_partition_preparations",
        "arriving_partition_reuses",
        "opportunity_calibration_kernel_ns",
        "opportunity_calibration_launches",
        "predicted_atomic_ns",
        "predicted_incremental_ns",
    }
)


def _read_engine_stats(
    workspace: pathlib.Path, prior_paths: set[pathlib.Path]
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for path in sorted(set(workspace.glob("nta-engine.*.json")) - prior_paths):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(report, dict):
            reports[path.name] = report
    return reports


def _wait_for_engine_stats(
    workspace: pathlib.Path,
    prior_paths: set[pathlib.Path],
    *,
    after_unix_ns: int,
    timeout_seconds: float = 10.0,
) -> dict[str, dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        reports = _read_engine_stats(workspace, prior_paths)
        if reports and all(
            int(report.get("snapshot_unix_ns", 0)) >= after_unix_ns
            for report in reports.values()
        ):
            return reports
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "NTA engine statistics did not publish the measurement snapshot"
            )
        time.sleep(0.01)


def _publish_engine_stats_snapshot(
    engine: Any,
    workspace: pathlib.Path,
    prior_paths: set[pathlib.Path],
) -> dict[str, dict[str, Any]]:
    """Take an out-of-band, CUDA-quiescent NTA statistics snapshot."""

    from nta_runtime.plugins.sglang import STATS_SNAPSHOT_RPC_METHOD

    snapshot_started_ns = time.time_ns()
    engine.collective_rpc(STATS_SNAPSHOT_RPC_METHOD)
    return _wait_for_engine_stats(
        workspace,
        prior_paths,
        after_unix_ns=snapshot_started_ns,
    )


def _require_closed_auto_calibration(
    reports: dict[str, dict[str, Any]],
    *,
    calibration_training_run: bool = False,
) -> None:
    """Keep AUTO learning outside a serving measurement window.

    A writable profile run is explicitly a training process whose complete
    output is excluded by the paired harness. Every actual measurement must
    load a read-only profile. Frozen policy maps an unseen shape to the
    conservative scheduled-whole-layer path, so dynamic batching cannot turn
    an exact-shape warmup miss into a user-visible exploration probe.
    """

    auto_reports = [
        report
        for report in reports.values()
        if report.get("backend") == "nta_flashinfer"
        and report.get("serving_tier") == "host_staged"
        and report.get("host_execution_mode") == "auto"
    ]
    if calibration_training_run and not auto_reports:
        raise RuntimeError(
            "AUTO calibration training requires host-staged AUTO execution"
        )
    failures: list[str] = []
    for report in auto_reports:
        if calibration_training_run:
            if (
                report.get("calibration_profile_enabled") is not True
                or report.get("calibration_profile_read_only") is not False
            ):
                failures.append("writable calibration profile")
            continue
        if (
            report.get("incremental_setup_calibrated") is not True
            or int(report.get("incremental_calibration_probes_remaining", -1)) != 0
        ):
            failures.append("execution-form setup")
        if report.get("host_mover_overlap_calibrated") is not True:
            failures.append("copy/compute overlap")
        if any(
            int(report.get(name, 0)) != 0
            for name in (
                "prefetch_mover_plan_frozen_uncalibrated_sm_leases",
                "prefetch_mover_plan_frozen_uncalibrated_copy_engine_leases",
                "prefetch_mover_plan_frozen_uncalibrated_overlap_leases",
            )
        ):
            failures.append("mover scale coverage")
        consumer = report.get("consumer_policy_calibration")
        if (
            not isinstance(consumer, dict)
            or consumer.get("mode") != "frozen"
            or report.get("calibration_profile_enabled") is not True
            or report.get("calibration_profile_status") != "loaded_read_only"
            or report.get("calibration_profile_read_only") is not True
        ):
            failures.append("read-only partial-consumer policy")
    if failures:
        raise RuntimeError(
            "AUTO serving measurement is not calibration-frozen: "
            + ", ".join(sorted(set(failures)))
            + "; prepare and reopen a compatibility-bound read-only profile"
        )


def _require_no_timed_auto_calibration(
    reports: Sequence[dict[str, Any]],
    *,
    calibration_training_run: bool = False,
) -> None:
    """Reject any AUTO learning action counted inside the timed delta."""

    if calibration_training_run:
        return
    actions: dict[str, int] = {}
    for name in _TIMED_AUTO_CALIBRATION_COUNTERS:
        count = sum(
            int(report.get(name, 0))
            for report in reports
            if report.get("backend") == "nta_flashinfer"
            and report.get("serving_tier") == "host_staged"
            and report.get("host_execution_mode") == "auto"
        )
        if count:
            actions[name] = count
    if actions:
        raise RuntimeError(
            "AUTO serving measurement performed online calibration: "
            + json.dumps(actions, sort_keys=True)
        )


def _consumer_contract(kind: str) -> dict[str, Any]:
    if kind not in {"native_work_unit", "framework_reference", "projection_only"}:
        raise ValueError(f"unknown measured consumer kind {kind!r}")
    native = kind == "native_work_unit"
    return {
        "schema": 1,
        "engine": "sglang",
        "backend": "nta_flashinfer",
        "kind": kind,
        "exact_demand": True,
        "typed_work_plan": native,
        "native_submission": native,
        "numerical_consumer": kind != "projection_only",
        "engine_version": os.environ.get("NTA_SGLANG_VERSION", "0.5.16"),
    }


def _measured_consumer_contracts(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Describe every numerical path used in the timed counter window."""

    native = (
        int(report.get("transformed_direct_launches", 0))
        + int(report.get("ticketed_incremental_launches", 0))
        + int(report.get("event_ordered_incremental_launches", 0))
    )
    stock = int(report.get("stock_prefetched_external_attention_launches", 0))
    kinds = [
        kind
        for kind, active in (
            ("native_work_unit", native),
            ("framework_reference", stock),
        )
        if active
    ]
    if not kinds:
        kinds.append("projection_only")
    return [_consumer_contract(kind) for kind in kinds]


def _measured_consumer_contract(report: dict[str, Any]) -> dict[str, Any]:
    """Return the strongest aggregate contract for compatibility."""

    return _measured_consumer_contracts(report)[0]


def _execution_dispatch(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate and classify the timed selection-to-consumer boundary.

    Typed metadata may prove that a mixed batch has no profitable overlap.
    That decision must dispatch stock FlashInfer; entering the native
    work-unit consumer in a direct-only window is pure mechanism overhead and
    invalidates a no-regression result.  Mixed direct/incremental windows are
    represented explicitly instead of guessing which batch owned a launch.
    """

    counters = {
        name: sum(int(report.get(name, 0)) for report in reports)
        for name in (
            "host_direct_batches",
            "host_scheduled_bulk_batches",
            "host_device_bulk_batches",
            "host_incremental_batches",
            "host_mixed_direct_batches",
            "host_mixed_scheduled_bulk_batches",
            "host_typed_mixed_batches",
            "stock_prefetched_external_attention_launches",
            "transformed_direct_launches",
            "ticketed_incremental_launches",
            "event_ordered_incremental_launches",
            "plan_uploads",
            "semantic_wrapper_plan_builds",
            "host_acquisition_jobs_submitted",
        )
    }
    direct = counters["host_direct_batches"]
    scheduled = counters["host_scheduled_bulk_batches"]
    device_bulk = counters["host_device_bulk_batches"]
    incremental = counters["host_incremental_batches"]
    stock_launches = counters["stock_prefetched_external_attention_launches"]
    native_launches = (
        counters["transformed_direct_launches"]
        + counters["ticketed_incremental_launches"]
        + counters["event_ordered_incremental_launches"]
    )
    active_forms = sum(
        value > 0 for value in (direct, scheduled, device_bulk, incremental)
    )
    if direct and active_forms == 1:
        residual = {
            name: counters[name]
            for name in (
                "transformed_direct_launches",
                "ticketed_incremental_launches",
                "event_ordered_incremental_launches",
                "plan_uploads",
                "semantic_wrapper_plan_builds",
            )
            if counters[name]
        }
        if residual:
            raise RuntimeError(
                "direct-only host selection entered the native work-unit path: "
                + json.dumps(residual, sort_keys=True)
            )
        if stock_launches == 0:
            raise RuntimeError(
                "direct-only host selection did not execute stock FlashInfer"
            )
        kind = "stock_direct"
    elif scheduled and active_forms == 1:
        if native_launches or stock_launches == 0:
            raise RuntimeError(
                "scheduled bulk did not remain on the stock numerical consumer"
            )
        kind = "scheduled_bulk"
    elif device_bulk and active_forms == 1:
        if native_launches == 0 or stock_launches:
            raise RuntimeError("device bulk did not remain on its native bulk consumer")
        kind = "device_bulk_diagnostic"
    elif incremental and active_forms == 1:
        if native_launches:
            kind = "native_incremental"
        elif counters["host_acquisition_jobs_submitted"] > 0 and stock_launches > 0:
            # Deadline scheduling may complete every exact job before its
            # numerical arrival.  The optimized ready-stock consumer is then
            # the intended consume decision, not a mechanism fallback.
            kind = "scheduled_preacquired"
        else:
            raise RuntimeError(
                "incremental host selection executed neither native work nor "
                "a scheduled preacquired consumer"
            )
    elif active_forms > 1:
        kind = "mixed_dispatch"
    else:
        kind = "unclassified"
    return {"kind": kind, **counters}


def _measurement_delta(
    final: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    """Project cumulative engine counters onto the timed request window."""
    measured = dict(final)
    counter_names = set(_MEASUREMENT_COUNTERS)
    final_schema = _cumulative_counter_schema(final)
    baseline_schema = _cumulative_counter_schema(baseline)
    if final_schema != baseline_schema:
        raise RuntimeError(
            "engine cumulative-counter schema changed during measurement"
        )
    counter_names.update(final_schema)
    counter_names.update(
        name for name in set(final) | set(baseline) if name.startswith("admission_")
    )
    counter_names.update(
        name
        for name in set(final) | set(baseline)
        if name.startswith("host_selection_") and name != "host_selection"
    )
    counter_names.update(
        name
        for name in set(final) | set(baseline)
        if name.startswith("native_dispatch_prefix_layers_")
        and name.endswith("_batches")
    )
    counter_names.update(
        name
        for name in set(final) | set(baseline)
        if name.startswith("progressive_consumer_layers_") and name.endswith("_batches")
    )
    # Frontier fields are all monotone work/accounting counters. Keep this
    # family-based so adding one scheduler observation cannot silently leak
    # warmup history into a timed serving window.
    counter_names.update(
        name
        for name in set(final) | set(baseline)
        if name.startswith("deadline_frontier_")
    )
    counter_names.update(
        name
        for name in set(final) | set(baseline)
        if name.startswith("host_mover_")
        and (
            name.endswith("_batches")
            or name.startswith("host_mover_overlap_profiled_")
            or (
                name.startswith("host_mover_profiled_")
                and name.endswith(("_bytes", "_gpu_ms"))
            )
            or name
            in {
                "host_mover_overlap_compute_ns",
                "host_mover_predicted_sm_ns",
                "host_mover_predicted_selected_ns",
            }
        )
    )
    counter_names.update(
        name
        for name in set(final) | set(baseline)
        if name.startswith("forward_") and name.endswith(("_count", "_ms_total"))
    )
    counter_names.update(
        name
        for name in set(final) | set(baseline)
        if name.startswith("observability_degraded_")
    )
    # Profiling and CPU accounting fields are created lazily, so a static
    # allow-list cannot know every operator form in advance.  Only cumulative
    # quantities are projected; rates and maxima are derived/gauge values and
    # must never be subtracted as counters.
    counter_names.update(
        name for name in set(final) | set(baseline) if name.endswith("_cpu_ns")
    )
    counter_names.update(
        name for name in set(final) | set(baseline) if name.endswith("_enqueue_layers")
    )
    counter_names.update(
        name
        for name in set(final) | set(baseline)
        if name.startswith("profiled_")
        and "_max_" not in name
        and name.endswith(
            ("_batches", "_bytes", "_gpu_ms", "_layers", "_launches", "_waits")
        )
    )
    for name in counter_names:
        end = final.get(name, 0)
        begin = baseline.get(name, 0)
        if isinstance(end, bool) or isinstance(begin, bool):
            raise RuntimeError(f"measurement counter {name} is boolean")
        if not isinstance(end, (int, float)) or not isinstance(begin, (int, float)):
            continue
        value = end - begin
        if value < 0:
            raise RuntimeError(f"measurement counter {name} decreased")
        measured[name] = value
    for map_name in ("profiled_attention_stall_by_layer_ms",):
        final_map = final.get(map_name, {})
        baseline_map = baseline.get(map_name, {})
        if not isinstance(final_map, dict) or not isinstance(baseline_map, dict):
            raise RuntimeError(f"measurement field {map_name} is not a mapping")
        projected: dict[str, float] = {}
        for layer in set(final_map) | set(baseline_map):
            end = final_map.get(layer, 0.0)
            begin = baseline_map.get(layer, 0.0)
            if (
                isinstance(end, bool)
                or isinstance(begin, bool)
                or not isinstance(end, (int, float))
                or not isinstance(begin, (int, float))
            ):
                raise RuntimeError(
                    f"measurement field {map_name}[{layer!r}] is not numeric"
                )
            value = float(end) - float(begin)
            if value < 0.0:
                raise RuntimeError(f"measurement field {map_name}[{layer!r}] decreased")
            if value > 0.0:
                projected[str(layer)] = value
        measured[map_name] = dict(
            sorted(projected.items(), key=lambda item: int(item[0]))
        )
    # A cumulative maximum cannot be projected by subtraction. Per-window
    # totals and event-order counts remain exact; omit maxima rather than
    # leaking warmup history into measured evidence.
    measured.pop("profiled_barrier_max_stall_gpu_ms", None)
    measured.pop("profiled_attention_max_stall_gpu_ms", None)
    for prefix in (
        "profiled_transfer",
        "profiled_pipeline_transfer",
        "profiled_fragment_transfer",
        "profiled_demand_transfer",
    ):
        elapsed_ms = float(measured.get(f"{prefix}_gpu_ms", 0.0))
        transfer_bytes = int(measured.get(f"{prefix}_bytes", 0))
        rate_name = f"{prefix}_gib_per_second"
        if elapsed_ms > 0.0 and transfer_bytes > 0:
            measured[rate_name] = transfer_bytes / (1 << 30) / (elapsed_ms / 1_000.0)
        else:
            measured.pop(rate_name, None)
    measured["measurement_scope"] = "timed_load_delta"
    measured["measurement_baseline_unix_ns"] = int(baseline.get("snapshot_unix_ns", 0))
    measured["measurement_counter_fields"] = sorted(counter_names)
    measured["consumer_contracts"] = _measured_consumer_contracts(measured)
    measured["consumer_contract"] = measured["consumer_contracts"][0]
    measured["execution_protocol_status"] = measured["consumer_contract"]["kind"]
    return measured


def _cumulative_counter_schema(report: dict[str, Any]) -> frozenset[str]:
    """Validate the counter ownership schema published by an engine report."""

    fields = report.get("cumulative_counter_fields")
    if fields is None:
        # Stock/upstream reports predate and do not need the NTA producer schema.
        return frozenset()
    if (
        not isinstance(fields, list)
        or any(not isinstance(field, str) or not field for field in fields)
        or len(fields) != len(set(fields))
    ):
        raise RuntimeError("engine cumulative-counter schema is invalid")
    return frozenset(fields)


ROOT = pathlib.Path(__file__).resolve().parents[2]

# SGLang 0.5.16 publishes ``max_req_input_len`` as ``context_len - 6``
# (``max_req_len`` is ``context_len - 1`` and the scheduler reserves another
# five tokens), then rejects inputs at the published bound.  Keep a small
# adapter margin so generated pressure requests remain valid across that
# tokenizer/scheduler boundary.  This is a request-envelope constraint, not a
# change to the normalized workload's exact token counts.
SGLANG_INPUT_MARGIN_TOKENS = 8


def _max_request_input_tokens(context_length: int, max_total_tokens: int) -> int:
    """Return the input envelope enforced by the configured SGLang engine.

    SGLang bounds one request by both the model context and the engine's KV
    token pool.  The latter matters for placement-pressure requests: using
    only ``context_length`` can generate a request larger than a deliberately
    small ``max_total_tokens`` pool before the timed workload even starts.
    """

    return min(context_length, max_total_tokens) - SGLANG_INPUT_MARGIN_TOKENS


def _machine_metadata() -> dict[str, Any]:
    def command(argv: list[str]) -> str | None:
        try:
            result = subprocess.run(
                argv,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "python": sys.version,
        "gpu": command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,pci.bus_id",
                "--format=csv,noheader",
            ]
        ),
    }


def _parse_cpu_affinity(value: str | None) -> frozenset[int] | None:
    """Parse a Linux CPU-list without delegating benchmark policy to a shell."""

    if value is None:
        return None
    cpus: set[int] = set()
    for segment in value.split(","):
        fields = segment.strip().split("-", 1)
        try:
            first = int(fields[0])
            last = first if len(fields) == 1 else int(fields[1])
        except ValueError as error:
            raise ValueError(f"invalid CPU affinity segment {segment!r}") from error
        if first < 0 or last < first:
            raise ValueError(f"invalid CPU affinity segment {segment!r}")
        cpus.update(range(first, last + 1))
    if not cpus:
        raise ValueError("CPU affinity cannot be empty")
    return frozenset(cpus)


def _apply_engine_cpu_affinity(
    engine: Any, requested: frozenset[int] | None
) -> dict[str, Any]:
    """Pin and verify the complete in-process SGLang engine tree.

    SGLang may replace the launcher's affinity while constructing scheduler
    subprocesses.  Applying the artifact contract after construction and
    before every setup/model request makes the measured control-plane and
    host-memory path reproducible.  Every existing thread is included; a
    process-level check alone can miss CUDA or framework worker threads with a
    wider inherited mask.
    """

    if requested is None:
        return {
            "requested": None,
            "verified": False,
            "scope": "uncontrolled",
            "processes": [],
        }
    os.sched_setaffinity(0, requested)
    child_pids = tuple(int(pid) for pid in engine.get_all_child_pids())
    process_pids = (os.getpid(), *child_pids)
    for pid in process_pids:
        task_root = pathlib.Path(f"/proc/{pid}/task")
        try:
            tids = tuple(int(entry.name) for entry in task_root.iterdir())
        except (FileNotFoundError, ProcessLookupError) as error:
            raise RuntimeError(
                f"SGLang process {pid} exited while applying CPU affinity"
            ) from error
        for tid in tids:
            try:
                os.sched_setaffinity(tid, requested)
            except ProcessLookupError:
                # A transient framework thread may retire between the task
                # directory snapshot and sched_setaffinity. The process-level
                # verification below remains authoritative.
                continue

    processes: list[dict[str, Any]] = []
    for pid in process_pids:
        try:
            actual = frozenset(os.sched_getaffinity(pid))
            name = pathlib.Path(f"/proc/{pid}/comm").read_text().strip()
        except (FileNotFoundError, ProcessLookupError) as error:
            raise RuntimeError(
                f"SGLang process {pid} exited before CPU-affinity verification"
            ) from error
        if actual != requested:
            raise RuntimeError(
                "SGLang CPU-affinity contract was not preserved: "
                f"process={name}, expected={sorted(requested)}, "
                f"actual={sorted(actual)}"
            )
        processes.append(
            {
                "role": "client" if pid == os.getpid() else name,
                "cpus": sorted(actual),
            }
        )
    return {
        "requested": sorted(requested),
        "verified": True,
        "scope": "client_and_engine_children",
        "processes": processes,
    }


def _engine_byte_accounting(
    stats: list[dict[str, Any]],
) -> tuple[int | None, int | None, str]:
    """Project NTA's physical transfer counters into the serving report.

    ``tier_selected_bytes`` counts each ownership lease's exact logical payload
    once. Per-work-unit demand is deliberately separate because many CTAs may
    share one acquired object. The physical bar is the bytes actually staged
    into device-side destinations: the sum of mutually exclusive host-pipeline,
    indexed-host, and NVMe counters. Keeping the projection here avoids both
    CTA-fanout overcounting and inference from token/cache-size estimates.
    """
    nta_stats = [
        entry
        for entry in stats
        if isinstance(entry, dict) and entry.get("backend") == "nta_flashinfer"
    ]
    if not nta_stats:
        return None, None, "not exposed by SGLang engine metadata"
    if any("tier_selected_bytes" not in entry for entry in nta_stats):
        raise RuntimeError("NTA engine omitted typed tier-selection accounting")
    selected = sum(int(entry["tier_selected_bytes"]) for entry in nta_stats)
    physical = sum(
        int(entry.get("prefetched_host_bytes", 0))
        + int(entry.get("indexed_host_bytes", 0))
        + int(entry.get("nvme_bytes", 0))
        for entry in nta_stats
    )
    if selected < 0 or physical < 0:
        raise RuntimeError("NTA engine published negative byte accounting")
    if physical == 0:
        return None, None, "not exposed by SGLang engine metadata"
    if physical < selected:
        raise RuntimeError(
            "NTA engine physical transfer bytes are below selected demand bytes"
        )
    return selected, physical, "exact_engine_transfer_counters"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument(
        "--attention-backend",
        choices=("flashinfer", "nta_flashinfer"),
        required=True,
    )
    parser.add_argument("--external-requests", type=int, default=3)
    parser.add_argument("--external-tokens", type=int, default=8192)
    parser.add_argument(
        "--external-suffix-tokens",
        type=int,
        default=0,
        help=(
            "uncached tokens appended to each host-resident prefix so the timed "
            "request executes chunked prefill instead of an exact-prefix decode"
        ),
    )
    parser.add_argument("--resident-requests", type=int, default=1)
    parser.add_argument("--resident-tokens", type=int, default=8192)
    parser.add_argument("--resident-output-tokens", type=int, default=128)
    parser.add_argument("--external-output-tokens", type=int, default=32)
    parser.add_argument("--request-rate", type=float, default=12.0)
    parser.add_argument("--churn-tokens", type=int, default=12000)
    parser.add_argument("--max-total-tokens", type=int, default=18000)
    parser.add_argument("--context-length", type=int, default=32768)
    # 0 keeps the historical setting (chunk == context length, i.e. a
    # 16K prefill runs as one unchunked forward). Smaller values are
    # the standard decode-protection configuration and apply to both
    # arms identically.
    parser.add_argument("--chunked-prefill-size", type=int, default=0)
    parser.add_argument("--mem-fraction-static", type=float, default=0.35)
    parser.add_argument("--hicache-ratio", type=float, default=8.0)
    parser.add_argument("--max-running-requests", type=int, default=16)
    parser.add_argument(
        "--numa-node",
        type=int,
        help=(
            "explicit SGLang scheduler and HiCache host-allocation NUMA node; "
            "use the node local to the measured GPU"
        ),
    )
    parser.add_argument(
        "--cpu-affinity",
        help=(
            "Linux CPU-list applied fail-closed to the SGLang client and all "
            "engine children (for example 0-15)"
        ),
    )
    parser.add_argument(
        "--eviction-rounds",
        type=int,
        help=(
            "explicit cache-churn rounds; zero disables churn for a capacity-fit "
            "workload, while the default derives rounds from max_total_tokens"
        ),
    )
    parser.add_argument(
        "--batch-mode",
        choices=("coalesced", "separate"),
        default="coalesced",
        help=(
            "coalesced enables SGLang mixed-chunk batching so resident decode and "
            "external-prefix work share the paged FlashInfer launch; separate is "
            "the scheduler-level ablation"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--workload-manifest",
        type=pathlib.Path,
        help=(
            "normalized Bailian workload manifest; the same manifest is used "
            "for both stock and NTA arms"
        ),
    )
    parser.add_argument(
        "--scale-workload-arrivals-to-request-rate",
        action="store_true",
        help=(
            "uniformly time-dilate a manifest with a recorded target rate so "
            "its exact request order/shape is replayed at --request-rate"
        ),
    )
    parser.add_argument(
        "--allow-oversubscribed-pool",
        action="store_true",
        help=(
            "admit timed contexts whose dense KV exceeds the device pool; "
            "this is the capacity experiment's operating condition — the "
            "dense arm honestly queues and retracts under pressure while "
            "the sidecar arm holds only bounded staging"
        ),
    )
    parser.add_argument("--flashinfer-workspace-base", type=pathlib.Path, required=True)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="optional persistent JSON report path",
    )
    parser.add_argument(
        "--cuda-graph-decode", choices=("disabled", "full"), default="disabled"
    )
    parser.add_argument(
        "--cuda-graph-prefill",
        choices=("disabled", "breakable"),
        default="disabled",
        help=(
            "prefill-phase CUDA graph backend for BOTH arms; breakable "
            "captures the dense per-layer compute piecewise and leaves "
            "attention and exact staging remain eager between "
            "pieces, shrinking the extend forward's launch-overhead span"
        ),
    )
    parser.add_argument(
        "--load-warmup-iterations",
        type=int,
        default=8,
        help=(
            "one or more performance-excluded exact-shape mixed arrivals; "
            "the final occurrence proves that timed cache placement and query "
            "geometry are reproducible"
        ),
    )
    parser.add_argument(
        "--auto-calibration-training-run",
        action="store_true",
        help=(
            "mark this entire NTA AUTO process as excluded profile training; "
            "requires a writable compatibility-bound calibration profile"
        ),
    )
    parser.add_argument(
        "--setup-idle-timeout-seconds",
        type=float,
        default=120.0,
        help=(
            "setup-only timeout for SGLang requests and asynchronous HiCache "
            "I/O to retire before deterministic cache reconstruction"
        ),
    )
    parser.add_argument("--slo-ttft-seconds", type=float, default=8.0)
    parser.add_argument("--slo-tpot-seconds", type=float, default=0.050)
    parser.add_argument("--slo-p99-itl-seconds", type=float, default=0.100)
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    integer_fields = (
        args.external_requests,
        args.external_tokens,
        args.resident_requests,
        args.resident_tokens,
        args.resident_output_tokens,
        args.external_output_tokens,
        args.churn_tokens,
        args.max_total_tokens,
        args.context_length,
        args.max_running_requests,
    )
    if min(integer_fields) <= 0:
        parser.error("request and token counts must be positive")
    if args.external_suffix_tokens < 0:
        parser.error("external suffix token count cannot be negative")
    if min(args.context_length, args.max_total_tokens) <= SGLANG_INPUT_MARGIN_TOKENS:
        parser.error("configured token capacity is too small for a request")
    max_request_input_tokens = _max_request_input_tokens(
        args.context_length, args.max_total_tokens
    )
    if args.churn_tokens >= max_request_input_tokens:
        parser.error("churn token count exceeds the SGLang request input budget")
    if args.load_warmup_iterations <= 0:
        parser.error(
            "at least one load warmup iteration is required for the exact "
            "placement calibration"
        )
    if args.setup_idle_timeout_seconds <= 0.0:
        parser.error("setup idle timeout must be positive")
    if args.eviction_rounds is not None and args.eviction_rounds < 0:
        parser.error("eviction rounds cannot be negative")
    if args.numa_node is not None and args.numa_node < 0:
        parser.error("NUMA node cannot be negative")
    try:
        _parse_cpu_affinity(args.cpu_affinity)
    except ValueError as error:
        parser.error(str(error))
    if (
        min(
            args.slo_ttft_seconds,
            args.slo_tpot_seconds,
            args.slo_p99_itl_seconds,
        )
        <= 0
    ):
        parser.error("SLO thresholds must be positive")
    if args.request_rate <= 0:
        parser.error("request rate must be positive")
    if args.scale_workload_arrivals_to_request_rate and args.workload_manifest is None:
        parser.error("arrival scaling requires --workload-manifest")
    if not 0.0 < args.mem_fraction_static < 1.0:
        parser.error("--mem-fraction-static must be between zero and one")
    if args.workload_manifest is None:
        if (
            args.external_tokens + args.external_suffix_tokens
            >= max_request_input_tokens
        ):
            parser.error("external prompt exceeds the SGLang request input budget")
        if args.resident_tokens >= max_request_input_tokens:
            parser.error("resident prompt exceeds the SGLang request input budget")
    if args.hicache_ratio <= 1:
        parser.error("HiCache ratio must exceed device cache capacity")
    if args.workload_manifest is None:
        active_tokens = (
            args.resident_requests * args.resident_tokens
            + args.external_requests
            * (args.external_tokens + args.external_suffix_tokens)
        )
        if (
            active_tokens >= args.max_total_tokens
            and not args.allow_oversubscribed_pool
        ):
            parser.error(
                "all timed resident and external contexts must fit together "
                "(pass --allow-oversubscribed-pool for capacity-pressure runs)"
            )
        if (
            args.external_requests * args.external_tokens + args.churn_tokens
            <= args.max_total_tokens
        ):
            parser.error(
                "external contexts and churn must exceed the device token pool"
            )
    return args


TokenInput = tuple[int, ...]


class _AsyncGate(Protocol):
    async def wait(self) -> Any: ...


class _Signal(Protocol):
    def set(self) -> Any: ...


def _required_placement_pressure_tokens(
    *,
    device_pool_tokens: int,
    page_tokens: int,
    external_cache_tokens: int,
    largest_external_object_tokens: int,
    exact_manifest: bool,
    eviction_rounds: int | None,
    churn_tokens: int,
) -> int:
    """Return unique setup pressure that proves the requested cache split.

    A pool-sized window is sufficient only when every old radix page is
    immediately evictable. HiCache write-through temporarily pins freshly
    materialized external pages. For an exact manifest, append one complete
    external working-set window so any page released after the eviction front
    first passes it is still made older than enough unique pages. This work is
    setup-only and is excluded from all serving measurements.
    """

    if min(device_pool_tokens, page_tokens, churn_tokens) <= 0:
        raise ValueError("cache placement geometry must be positive")
    if min(external_cache_tokens, largest_external_object_tokens) < 0:
        raise ValueError("external cache working set cannot be negative")
    if eviction_rounds is not None and eviction_rounds < 0:
        raise ValueError("cache placement eviction rounds cannot be negative")

    pool_frontier = device_pool_tokens + page_tokens
    if exact_manifest:
        # Radix eviction is object-granular: a leaf cannot be partially
        # displaced to satisfy a token-count target. One largest-object slack
        # turns the continuous pool budget into a safe discrete frontier.
        required = (
            pool_frontier + external_cache_tokens + largest_external_object_tokens
        )
        if eviction_rounds is not None:
            required = max(required, eviction_rounds * churn_tokens)
        return required
    if eviction_rounds is None:
        return pool_frontier
    return eviction_rounds * churn_tokens


def _structure_token_inputs(
    tokenizer: Any, rows: Sequence[dict[str, Any]], block_size: int
) -> tuple[tuple[TokenInput, ...], str]:
    """Map exact content-block identity to tokenizer-valid input IDs.

    Text decode/encode is not identity preserving under BPE.  The serving API
    accepts token IDs directly, so each distinct block receives a unique first
    token and deterministic filler tokens.  This makes equal block hashes
    byte-for-byte equal while distinct hashes diverge at their first position.
    """

    vocabulary_size = int(len(tokenizer))
    special = {int(value) for value in getattr(tokenizer, "all_special_ids", ())}
    safe_tokens = [
        token_id for token_id in range(vocabulary_size) if token_id not in special
    ]
    page_ids = sorted(
        {
            page_id
            for row in rows
            for page_id in input_page_ids(row, block_size=block_size)
        }
    )
    if not safe_tokens or len(page_ids) > len(safe_tokens):
        raise RuntimeError(
            "Bailian cohort has more distinct pages than collision-free first "
            "tokens in the tokenizer vocabulary"
        )
    block_tokens: dict[str, TokenInput] = {}
    identity_digest = hashlib.sha256()
    for ordinal, page_id in enumerate(page_ids):
        values = [safe_tokens[ordinal]]
        for position in range(1, block_size):
            digest = hashlib.sha256(f"{page_id}:{position}".encode("utf-8")).digest()
            values.append(
                safe_tokens[int.from_bytes(digest[:8], "big") % len(safe_tokens)]
            )
        encoded = tuple(values)
        block_tokens[page_id] = encoded
        identity_digest.update(page_id.encode("utf-8"))
        identity_digest.update(b"\0")
        for token_id in encoded:
            identity_digest.update(int(token_id).to_bytes(8, "little"))

    inputs: list[TokenInput] = []
    for row in rows:
        token_count = int(row["input_length"])
        values = tuple(
            token_id
            for page_id in input_page_ids(row, block_size=block_size)
            for token_id in block_tokens[page_id]
        )[:token_count]
        if len(values) != token_count:
            raise RuntimeError("Bailian token adapter did not cover the full input")
        inputs.append(values)
    return tuple(inputs), identity_digest.hexdigest()


def _append_request_unique_suffixes(
    tokenizer: Any,
    request_ids: Sequence[str],
    inputs: Sequence[TokenInput],
    token_count: int,
) -> tuple[tuple[TokenInput, ...], str | None]:
    """Append deterministic request-local token rows to manifest inputs.

    The Bailian manifest describes the source trace.  A mechanism-envelope
    experiment may add uncached compute rows, but those rows must be present in
    the actual token inputs rather than merely in report metadata.  Distinct
    first suffix tokens prevent two equal (or prefix-related) source prompts
    from accidentally sharing the synthetic continuation in SGLang's radix
    cache.  The full suffix is deterministic so paired arms replay identical
    demand.
    """

    if token_count < 0:
        raise ValueError("external suffix token count cannot be negative")
    normalized_inputs = tuple(
        tuple(int(value) for value in prompt) for prompt in inputs
    )
    if len(request_ids) != len(normalized_inputs):
        raise RuntimeError("external suffix request identities and inputs disagree")
    if token_count == 0:
        return normalized_inputs, None

    vocabulary_size = int(len(tokenizer))
    special = {int(value) for value in getattr(tokenizer, "all_special_ids", ())}
    safe_tokens = tuple(
        token_id for token_id in range(vocabulary_size) if token_id not in special
    )
    if len(safe_tokens) < len(normalized_inputs):
        raise RuntimeError(
            "tokenizer has too few non-special tokens for request-unique suffixes"
        )

    # If one source prompt is a prefix of another, do not choose the latter's
    # next token as the former's branch token.  Equal prompts are separated by
    # the globally unique generated first-token set below.
    forbidden_by_index: list[set[int]] = []
    for prompt in normalized_inputs:
        forbidden_by_index.append(
            {
                other[len(prompt)]
                for other in normalized_inputs
                if len(other) > len(prompt) and other[: len(prompt)] == prompt
            }
        )

    used_first_tokens: set[int] = set()
    suffix_identity = hashlib.sha256(b"nta-request-unique-suffix-v1\0")
    extended: list[TokenInput] = []
    for index, (request_id, prompt) in enumerate(
        zip(request_ids, normalized_inputs, strict=True)
    ):
        request_key = str(request_id).encode("utf-8")
        seed = hashlib.sha256(b"nta-suffix-first\0" + request_key).digest()
        start = int.from_bytes(seed[:8], "big") % len(safe_tokens)
        first_token: int | None = None
        for offset in range(len(safe_tokens)):
            candidate = safe_tokens[(start + offset) % len(safe_tokens)]
            if (
                candidate not in used_first_tokens
                and candidate not in forbidden_by_index[index]
            ):
                first_token = candidate
                break
        if first_token is None:  # pragma: no cover - guarded by vocabulary check
            raise RuntimeError("could not allocate a request-unique suffix branch")
        used_first_tokens.add(first_token)

        suffix = [first_token]
        for position in range(1, token_count):
            digest = hashlib.sha256(
                b"nta-suffix-row\0" + request_key + int(position).to_bytes(8, "little")
            ).digest()
            suffix.append(
                safe_tokens[int.from_bytes(digest[:8], "big") % len(safe_tokens)]
            )
        suffix_identity.update(request_key)
        suffix_identity.update(b"\0")
        for token_id in suffix:
            suffix_identity.update(int(token_id).to_bytes(8, "little"))
        extended.append(prompt + tuple(suffix))
    return tuple(extended), suffix_identity.hexdigest()


@dataclass(frozen=True)
class LoadedWorkload:
    """Role-partitioned replay inputs sharing one exact arrival timebase."""

    resident_request_ids: tuple[str, ...]
    external_request_ids: tuple[str, ...]
    resident_inputs: tuple[TokenInput, ...]
    external_inputs: tuple[TokenInput, ...]
    resident_arrival_offsets: tuple[float, ...]
    external_arrival_offsets: tuple[float, ...]
    resident_output_tokens: tuple[int, ...]
    external_output_tokens: tuple[int, ...]
    metadata: dict[str, Any]


def _configure_workload_arrivals(
    workload: LoadedWorkload,
    *,
    target_rate: float,
    scale_to_target: bool,
) -> LoadedWorkload:
    """Prove or explicitly transform one manifest arrival schedule."""

    if not math.isfinite(target_rate) or target_rate <= 0.0:
        raise ValueError("workload target arrival rate must be positive")
    metadata = dict(workload.metadata)
    arrival = metadata.get("arrival")
    if not isinstance(arrival, dict):
        raise RuntimeError("workload manifest has no arrival contract")
    source_value = arrival.get("target_rate_per_second")
    source_rate = (
        float(source_value)
        if isinstance(source_value, (int, float))
        and not isinstance(source_value, bool)
        and math.isfinite(float(source_value))
        and float(source_value) > 0.0
        else None
    )
    if scale_to_target:
        if source_rate is None:
            raise RuntimeError(
                "arrival scaling requires a manifest with a positive "
                "target_rate_per_second"
            )
        scale = source_rate / target_rate
        resident_offsets = tuple(
            float(offset) * scale for offset in workload.resident_arrival_offsets
        )
        external_offsets = tuple(
            float(offset) * scale for offset in workload.external_arrival_offsets
        )
        request_offsets = {
            request_id: float(offset) * scale
            for request_id, offset in metadata["request_arrival_offsets"].items()
        }
        method = "uniform_manifest_time_dilation"
    else:
        if source_rate is not None and not math.isclose(
            source_rate, target_rate, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise RuntimeError(
                "--request-rate disagrees with the manifest arrival rate; "
                "pass --scale-workload-arrivals-to-request-rate to transform it"
            )
        scale = 1.0
        resident_offsets = workload.resident_arrival_offsets
        external_offsets = workload.external_arrival_offsets
        request_offsets = dict(metadata["request_arrival_offsets"])
        method = "manifest_exact"
    metadata["request_arrival_offsets"] = request_offsets
    metadata["runtime_arrival"] = {
        "method": method,
        "source_mode": str(arrival.get("mode")),
        "source_target_rate_per_second": source_rate,
        "target_rate_per_second": target_rate if source_rate is not None else None,
        "uniform_time_scale": scale,
        "request_order_preserved": True,
    }
    return replace(
        workload,
        resident_arrival_offsets=resident_offsets,
        external_arrival_offsets=external_offsets,
        metadata=metadata,
    )


def _load_workload(
    path: pathlib.Path,
    tokenizer: Any,
    *,
    external_suffix_tokens: int = 0,
) -> LoadedWorkload:
    manifest = validate_workload(path.resolve())
    records_path = path.resolve().parent / str(manifest["records_file"])
    rows = read_jsonl(records_path)
    if not rows:
        raise RuntimeError("normalized workload contains no requests")
    computed_demand_digest = demand_trace_digest(rows)
    if computed_demand_digest != manifest["demand_trace_digest"]:
        raise RuntimeError(
            "normalized workload demand digest does not match its records"
        )
    explicit_states = [row.get("request_state") for row in rows]
    if any(state not in {"resident", "external"} for state in explicit_states):
        raise RuntimeError(
            "serving replay requires explicit resident/external placement for "
            "every request"
        )
    resident_rows = [row for row in rows if row["request_state"] == "resident"]
    external_rows = [row for row in rows if row["request_state"] == "external"]
    if not resident_rows or not external_rows:
        raise RuntimeError(
            "serving replay needs at least one resident and one external request; "
            "add request_state to the normalized workload"
        )

    origin = min(float(row["arrival_seconds"]) for row in rows)
    resident_offsets = tuple(
        float(row["arrival_seconds"]) - origin for row in resident_rows
    )
    external_offsets = tuple(
        float(row["arrival_seconds"]) - origin for row in external_rows
    )
    request_arrival_offsets = {
        str(row["request_id"]): float(row["arrival_seconds"]) - origin for row in rows
    }
    block_size = int(manifest["block_size"])
    inputs, token_identity_digest = _structure_token_inputs(tokenizer, rows, block_size)
    source_input_by_request = {
        str(row["request_id"]): values for row, values in zip(rows, inputs, strict=True)
    }
    external_request_ids = tuple(str(row["request_id"]) for row in external_rows)
    source_external_inputs = tuple(
        source_input_by_request[request_id] for request_id in external_request_ids
    )
    effective_external_inputs, suffix_identity_digest = _append_request_unique_suffixes(
        tokenizer,
        external_request_ids,
        source_external_inputs,
        external_suffix_tokens,
    )
    input_by_request = dict(source_input_by_request)
    input_by_request.update(
        zip(external_request_ids, effective_external_inputs, strict=True)
    )
    source_identity_digest = token_identity_digest
    if suffix_identity_digest is not None:
        identity = hashlib.sha256(b"nta-effective-token-input-v1\0")
        identity.update(source_identity_digest.encode("ascii"))
        identity.update(b"\0")
        identity.update(suffix_identity_digest.encode("ascii"))
        token_identity_digest = identity.hexdigest()
    resident_page_ids = unique_input_page_ids(resident_rows, block_size=block_size)
    external_cached_prefix_tokens = [
        int(row["cached_prefix_tokens"]) for row in external_rows
    ]
    initial_cached_objects = [
        (
            source_input_by_request[str(row["request_id"])],
            (
                int(row["cached_prefix_tokens"])
                if row["request_state"] == "external"
                else int(row["input_length"]) - 1
            ),
        )
        for row in rows
    ]
    external_effective_cached_prefix_tokens = list(
        effective_cached_prefixes(
            [(values, len(values)) for values in effective_external_inputs],
            initial_cached_objects,
        )
    )
    declared_effective = [
        row.get("effective_cached_prefix_tokens") for row in external_rows
    ]
    if any(value is not None for value in declared_effective) and (
        any(value is None for value in declared_effective)
        or [int(value) for value in declared_effective]
        != external_effective_cached_prefix_tokens
    ):
        raise RuntimeError(
            "manifest effective cache prefixes disagree with token-level identity"
        )
    external_cached_page_ids = frozenset(
        page_id
        for row, prefix_tokens in zip(
            external_rows, external_cached_prefix_tokens, strict=True
        )
        for page_id in input_page_ids(row, block_size=block_size)[
            : (prefix_tokens + block_size - 1) // block_size
        ]
    )
    metadata = {
        "manifest": str(path.resolve()),
        "manifest_digest": hashlib.sha256(path.resolve().read_bytes()).hexdigest(),
        "records_digest": str(manifest["records_digest"]),
        "demand_trace_digest": str(manifest["demand_trace_digest"]),
        "block_size": block_size,
        "arrival": manifest["arrival"],
        "prompt": manifest["prompt"],
        "claims": manifest["claims"],
        "selection": manifest["selection"],
        "state_mapping": "explicit_request_state",
        "cache_placement": manifest["cache_placement"],
        "request_count": len(rows),
        "request_id_order": [str(row["request_id"]) for row in rows],
        "request_arrival_offsets": request_arrival_offsets,
        "tokenization_errors": 0,
        "token_input_adapter": "collision_free_content_block_tokens_v1",
        "token_input_identity_digest": token_identity_digest,
        "source_token_input_identity_digest": source_identity_digest,
        "token_suffix_adapter": (
            "deterministic_request_unique_token_suffix_v1"
            if external_suffix_tokens
            else "none"
        ),
        "token_suffix_identity_digest": suffix_identity_digest,
        "external_suffix_tokens": external_suffix_tokens,
        "resident_input_tokens": [int(row["input_length"]) for row in resident_rows],
        "source_external_input_tokens": [
            int(row["input_length"]) for row in external_rows
        ],
        "external_input_tokens": [
            len(input_by_request[str(row["request_id"])]) for row in external_rows
        ],
        # Each row declares the object materialized for that request.  The
        # effective prefix below is derived separately from the union of all
        # resident and external objects in the shared radix cache.
        "external_cached_prefix_tokens": external_cached_prefix_tokens,
        "external_effective_cached_prefix_tokens": (
            external_effective_cached_prefix_tokens
        ),
        "resident_output_tokens": [int(row["output_length"]) for row in resident_rows],
        "external_output_tokens": [int(row["output_length"]) for row in external_rows],
        # These are logical page counts, not raw request-length sums.  Shared
        # Bailian prefix pages occur once in the combined pressure budget.
        "resident_input_cache_pages": len(resident_page_ids),
        "external_input_cache_pages": len(external_cached_page_ids),
        "combined_input_cache_pages": len(resident_page_ids | external_cached_page_ids),
        "shared_input_cache_pages": len(resident_page_ids & external_cached_page_ids),
    }
    return LoadedWorkload(
        resident_request_ids=tuple(str(row["request_id"]) for row in resident_rows),
        external_request_ids=external_request_ids,
        resident_inputs=tuple(
            input_by_request[str(row["request_id"])] for row in resident_rows
        ),
        external_inputs=tuple(
            input_by_request[str(row["request_id"])] for row in external_rows
        ),
        resident_arrival_offsets=resident_offsets,
        external_arrival_offsets=external_offsets,
        resident_output_tokens=tuple(metadata["resident_output_tokens"]),
        external_output_tokens=tuple(metadata["external_output_tokens"]),
        metadata=metadata,
    )


def _meta(result: dict[str, Any]) -> dict[str, Any]:
    meta = result.get("meta_info")
    if not isinstance(meta, dict):
        raise RuntimeError("SGLang omitted request metadata")
    return meta


def _token_input(tokenizer: Any, prompt: str) -> TokenInput:
    token_ids = tuple(
        int(value) for value in tokenizer.encode(prompt, add_special_tokens=False)
    )
    if not token_ids:
        raise RuntimeError("serving request tokenized to an empty input")
    return token_ids


def _unique_pressure_input(
    tokenizer: Any,
    *,
    label: str,
    token_count: int,
    forbidden_first_tokens: set[int],
) -> TokenInput:
    """Build one exact-length pressure object with a disjoint radix root."""

    values = list(_token_input(tokenizer, make_prompt(tokenizer, label, token_count)))
    special = {int(value) for value in getattr(tokenizer, "all_special_ids", ())}
    vocabulary_size = int(len(tokenizer))
    seed = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")
    for offset in range(vocabulary_size):
        candidate = (seed + offset) % vocabulary_size
        if candidate not in special and candidate not in forbidden_first_tokens:
            values[0] = candidate
            forbidden_first_tokens.add(candidate)
            return tuple(values)
    raise RuntimeError("tokenizer has no disjoint placement-pressure root token")


def _reusable_prefix_tokens(prefix: TokenInput, prompt: TokenInput) -> int:
    """Return the exact radix rows reusable by the timed SGLang request.

    SGLang retains the final input token as a query when the request is an
    exact-prefix hit.  A continuation-bearing request can reuse the complete
    materialized prefix because its first suffix token supplies that query.
    """

    if not prefix or len(prefix) > len(prompt) or prompt[: len(prefix)] != prefix:
        raise RuntimeError("external prompt does not extend its materialized prefix")
    reusable = len(prefix) - int(len(prefix) == len(prompt))
    if reusable <= 0:
        raise RuntimeError("external request has no reusable radix prefix")
    return reusable


def _placement_probe_groups(
    prefixes: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    """Group equal objects and order destructive probes by radix inclusion.

    A probe promotes the observed prefix.  Probing a longer shared prefix
    before a shorter one can therefore make the shorter request appear
    device-resident even though their shared object was host-backed before
    verification.  Equal prefixes are one content object and need one probe;
    shortest-first ordering proves every distinct radix frontier before a
    strict superset can promote it.
    """

    groups: dict[TokenInput, list[int]] = {}
    for index, prefix in enumerate(prefixes):
        identity = tuple(int(value) for value in prefix)
        if not identity:
            raise ValueError("placement probe prefix cannot be empty")
        groups.setdefault(identity, []).append(index)
    return tuple(
        tuple(indices)
        for _, indices in sorted(
            groups.items(),
            key=lambda item: (len(item[0]), item[0]),
        )
    )


def _generate_one(
    engine: Any,
    input_ids: TokenInput,
    sampling: dict[str, Any],
    *,
    rid: str | None = None,
) -> Any:
    return engine.generate(
        input_ids=list(input_ids),
        sampling_params=sampling,
        rid=rid,
    )


def _generate_many(
    engine: Any,
    inputs: Sequence[TokenInput],
    samplings: Sequence[dict[str, Any]],
) -> Any:
    if len(inputs) != len(samplings):
        raise RuntimeError("batched serving inputs and sampling parameters disagree")
    return engine.generate(
        input_ids=[list(input_ids) for input_ids in inputs],
        sampling_params=list(samplings),
    )


def _flush_cache_when_idle(
    engine: Any,
    *,
    timeout_seconds: float,
    reason: str,
) -> float:
    """Flush SGLang after requests and asynchronous HiCache I/O retire.

    ``Engine.generate`` returning is not a HiCache lifetime boundary: with
    write-through enabled, the scheduler may still own a D-to-H operation for
    the completed request.  SGLang's immediate ``Engine.flush_cache`` wrapper
    therefore races legitimately with setup traffic.  The tokenizer-manager
    contract carries a bounded timeout to the scheduler, whose deferred flush
    waits for request queues and all HiCache I/O to become idle.
    """

    if timeout_seconds <= 0.0:
        raise ValueError("SGLang idle-flush timeout must be positive")
    manager = getattr(engine, "tokenizer_manager", None)
    event_loop = getattr(engine, "loop", None)
    flush_cache = getattr(manager, "flush_cache", None)
    run_until_complete = getattr(event_loop, "run_until_complete", None)
    if manager is None or not callable(flush_cache) or not callable(run_until_complete):
        raise RuntimeError(
            f"SGLang engine lacks the deferred idle-flush contract required to {reason}"
        )

    started = time.perf_counter()
    result = run_until_complete(flush_cache(timeout_s=timeout_seconds))
    elapsed = time.perf_counter() - started
    if not bool(getattr(result, "success", False)):
        message = str(getattr(result, "message", "")) or "unspecified failure"
        raise RuntimeError(
            f"failed to {reason} after waiting at most "
            f"{timeout_seconds:.3f}s for SGLang/HiCache retirement: {message}"
        )
    return elapsed


def _distinct_prefix_branch_input(
    tokenizer: Any,
    prefix: Sequence[int],
    measured: Sequence[int],
    *,
    label: str,
    forbidden_first_tokens: set[int],
) -> TokenInput:
    """Build an exact-prefix branch with a distinct continuation.

    Prefix materialization must not insert the timed continuation into the
    radix cache. Text-level length matching is insufficient because tokenizer
    merges can alter the boundary. This harness submits token IDs directly:
    the cached prefix is byte-for-byte identical and the continuation has the
    requested row count, while its first token is unique across materialized
    and measured inputs.
    """
    prefix_ids = tuple(int(value) for value in prefix)
    measured_ids = tuple(int(value) for value in measured)
    if not prefix_ids or measured_ids[: len(prefix_ids)] != prefix_ids:
        raise RuntimeError("distinct branch does not retain the exact cached prefix")
    query_rows = len(measured_ids) - len(prefix_ids)
    if query_rows <= 0:
        raise RuntimeError("distinct branch has no uncached continuation")
    suffix = list(_token_input(tokenizer, make_prompt(tokenizer, label, query_rows)))
    if len(suffix) != query_rows:
        raise RuntimeError("distinct branch changed the exact query-row count")
    special = {int(value) for value in getattr(tokenizer, "all_special_ids", ())}
    vocabulary_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    if vocabulary_size <= len(special) + len(forbidden_first_tokens):
        raise RuntimeError("tokenizer has no distinct branch token")
    if suffix[0] in forbidden_first_tokens or suffix[0] in special:
        seed = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")
        for offset in range(vocabulary_size):
            candidate = (seed + offset) % vocabulary_size
            if candidate not in special and candidate not in forbidden_first_tokens:
                suffix[0] = candidate
                break
        else:  # pragma: no cover - guarded by the vocabulary-size check above
            raise RuntimeError("could not select a distinct branch token")
    forbidden_first_tokens.add(suffix[0])
    candidate = prefix_ids + tuple(suffix)
    if len(candidate) != len(measured_ids) or candidate == measured_ids:
        raise RuntimeError("distinct branch did not preserve the exact request shape")
    return candidate


def _exact_prefix_materialization_inputs(
    tokenizer: Any,
    prefixes: Sequence[Sequence[int]],
    measured_inputs: Sequence[Sequence[int]],
) -> tuple[TokenInput, ...]:
    """Create one-token branches that materialize every exact timed prefix.

    Submitting a prefix by itself can leave its final input token outside the
    reusable radix boundary.  A distinct one-token continuation makes the
    complete prefix reusable without inserting the timed continuation.  One
    shared forbidden set prevents materialization branches from colliding
    across requests.
    """

    if len(prefixes) != len(measured_inputs):
        raise RuntimeError("prefix materialization inputs disagree")
    forbidden_first_tokens: set[int] = set()
    for prefix, measured in zip(prefixes, measured_inputs, strict=True):
        prefix_ids = tuple(int(value) for value in prefix)
        measured_ids = tuple(int(value) for value in measured)
        if (
            not prefix_ids
            or measured_ids[: len(prefix_ids)] != prefix_ids
            or len(measured_ids) <= len(prefix_ids)
        ):
            raise RuntimeError(
                "prefix materialization requires an exact prefix and continuation"
            )
        forbidden_first_tokens.add(measured_ids[len(prefix_ids)])

    materialization_inputs: list[TokenInput] = []
    for index, (prefix, measured) in enumerate(
        zip(prefixes, measured_inputs, strict=True)
    ):
        prefix_ids = tuple(int(value) for value in prefix)
        measured_ids = tuple(int(value) for value in measured)
        one_row_shape = prefix_ids + (measured_ids[len(prefix_ids)],)
        materialization_inputs.append(
            _distinct_prefix_branch_input(
                tokenizer,
                prefix_ids,
                one_row_shape,
                label=f"prefix-materialization-{index}",
                forbidden_first_tokens=forbidden_first_tokens,
            )
        )
    return tuple(materialization_inputs)


async def _stream_request(
    engine: Any,
    input_ids: TokenInput,
    sampling: dict[str, Any],
    *,
    kind: str,
    index: int,
    request_id: str | None,
    gate: _AsyncGate | None,
    first_token_event: _Signal | None,
    offset_seconds: float,
    load_start_seconds: float,
) -> dict[str, Any]:
    # Natural-trace arrivals are scheduled from the load origin.  The
    # synthetic mixed-batch workload is different: its external cohort is not
    # released until every resident request has produced a first token.  That
    # barrier defines the synthetic cohort's arrival, rather than client or
    # engine admission delay.  Keeping the old load-origin timestamp made the
    # deliberate workload gate look like tens of milliseconds of admission
    # queueing and corrupted finite-window queueing/accounting evidence.
    schedule_origin = load_start_seconds
    workload_gate_wait_seconds = 0.0
    if gate is not None:
        await gate.wait()
        schedule_origin = time.perf_counter()
        workload_gate_wait_seconds = max(0.0, schedule_origin - load_start_seconds)
    scheduled_arrival = schedule_origin + offset_seconds
    # ``asyncio.sleep(delay)`` is allowed to wake at the event-loop clock's
    # resolution boundary.  On this host that occasionally returned up to
    # 0.7 ms before the requested deadline, which made a formally open-loop
    # request appear to be submitted before its registered arrival.  Wait on
    # the absolute deadline and recheck after every wakeup so the load
    # generator, not a validator tolerance, owns the arrival contract.
    while True:
        remaining = scheduled_arrival - time.perf_counter()
        if remaining <= 0.0:
            break
        await asyncio.sleep(remaining)
    submitted = time.perf_counter()
    stream = await engine.async_generate(
        input_ids=list(input_ids),
        sampling_params=sampling,
        stream=True,
        rid=request_id or f"nta-load-{kind}-{index}",
    )
    first = 0.0
    token_times: list[float] = []
    previous_completion_tokens = 0
    duplicate_stream_events = 0
    final: dict[str, Any] | None = None
    async for result in stream:
        now = time.perf_counter()
        meta = _meta(result)
        completion_tokens = int(meta.get("completion_tokens", 0))
        token_delta = completion_tokens - previous_completion_tokens
        if token_delta < 0:
            raise RuntimeError("SGLang stream completion token count decreased")
        if token_delta > 1:
            raise RuntimeError(
                "SGLang stream coalesced multiple tokens despite stream_interval=1; "
                "token-level ITL would be invalid"
            )
        if token_delta == 1:
            if first == 0.0:
                first = now
                if first_token_event is not None:
                    first_token_event.set()
            token_times.append(now)
            previous_completion_tokens = completion_tokens
        else:
            duplicate_stream_events += 1
        final = result
    finished = time.perf_counter()
    if final is None or first == 0.0:
        raise RuntimeError(f"SGLang returned no streamed output for {kind}-{index}")
    meta = _meta(final)
    completion_tokens = int(meta.get("completion_tokens", len(token_times)))
    if completion_tokens != len(token_times):
        raise RuntimeError(
            "SGLang stream events do not provide one timestamp per completion token"
        )
    intervals = [
        current - previous for previous, current in zip(token_times, token_times[1:])
    ]
    return {
        "kind": kind,
        "index": index,
        "request_id": request_id or f"nta-load-{kind}-{index}",
        "arrival_offset_seconds": scheduled_arrival - load_start_seconds,
        "arrival_seconds": scheduled_arrival - load_start_seconds,
        "workload_gate_wait_seconds": workload_gate_wait_seconds,
        "submitted_offset_seconds": submitted - load_start_seconds,
        "first_token_offset_seconds": first - load_start_seconds,
        "finished_offset_seconds": finished - load_start_seconds,
        "submitted_seconds": submitted,
        "first_token_seconds": first,
        "finished_seconds": finished,
        "ttft_seconds": first - submitted,
        "e2e_seconds": finished - submitted,
        "admission_delay_seconds": max(0.0, submitted - scheduled_arrival),
        "system_time_seconds": max(0.0, finished - scheduled_arrival),
        "tpot_seconds": (
            (token_times[-1] - first) / (completion_tokens - 1)
            if completion_tokens > 1
            else 0.0
        ),
        "inter_token_seconds": intervals,
        "itl_sample_count": len(intervals),
        "token_timestamps_exact": True,
        "token_timestamp_source": "sglang_stream_interval_1_completion_delta",
        "duplicate_stream_events": duplicate_stream_events,
        "p99_itl_seconds": _percentile(intervals, 0.99),
        "completion_tokens": completion_tokens,
        "input_tokens": len(input_ids),
        "device_cached_tokens": device_cached_tokens(final),
        "host_cached_tokens": host_cached_tokens(final),
        "text": generated_text(final),
    }


class _FirstTokenBarrier:
    """Release synthetic external arrivals after every resident is active.

    A single ``asyncio.Event`` set by the first resident made the measured
    mixed-forward shape depend on which response happened to reach the client
    first.  That scheduler race changed the number of engine forwards across
    otherwise identical arms.  Natural-trace replay uses explicit timestamps
    and bypasses this barrier; the synthetic mechanism workload uses it to
    define one reproducible state transition.
    """

    __slots__ = ("_event", "_remaining")

    def __init__(self, parties: int) -> None:
        if parties <= 0:
            raise ValueError("first-token barrier requires resident parties")
        self._remaining = parties
        self._event = asyncio.Event()

    def set(self) -> None:
        if self._remaining <= 0:
            return
        self._remaining -= 1
        if self._remaining == 0:
            self._event.set()

    async def wait(self) -> None:
        await self._event.wait()


async def _run_load(
    engine: Any,
    resident_prompts: list[TokenInput],
    external_prompts: list[TokenInput],
    args: argparse.Namespace,
    resident_offsets: Sequence[float] | None = None,
    external_offsets: Sequence[float] | None = None,
    resident_output_tokens: Sequence[int] | None = None,
    external_output_tokens: Sequence[int] | None = None,
    resident_request_ids: Sequence[str] | None = None,
    external_request_ids: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    residents_started = _FirstTokenBarrier(len(resident_prompts))
    resident_sampling = {
        "temperature": 0,
        # A performance-excluded warmup must preserve the timed concurrency
        # shape. Truncating resident decode to one token lets it retire before
        # the external request joins, so graphs and service curves calibrate a
        # different single-request forward and the measured mixed shape stays
        # cold.
        "max_new_tokens": args.resident_output_tokens,
        "ignore_eos": True,
        "stream_interval": 1,
    }
    external_sampling = {
        "temperature": 0,
        "max_new_tokens": args.external_output_tokens,
        "ignore_eos": True,
        "stream_interval": 1,
    }

    async def resident(index: int, prompt: TokenInput) -> dict[str, Any]:
        sampling = dict(resident_sampling)
        if resident_output_tokens is not None:
            sampling["max_new_tokens"] = max(1, resident_output_tokens[index])
        record = await _stream_request(
            engine,
            prompt,
            sampling,
            kind="resident",
            index=index,
            request_id=(
                resident_request_ids[index]
                if resident_request_ids is not None
                else None
            ),
            gate=None,
            first_token_event=residents_started,
            offset_seconds=(
                float(resident_offsets[index]) if resident_offsets is not None else 0.0
            ),
            load_start_seconds=started,
        )
        return record

    if resident_offsets is not None and len(resident_offsets) != len(resident_prompts):
        raise RuntimeError("workload arrival count does not match resident prompts")
    if external_offsets is not None:
        if len(external_offsets) != len(external_prompts):
            raise RuntimeError("workload arrival count does not match external prompts")
        offsets = external_offsets
    else:
        rng = random.Random(args.seed)
        offsets = [0.0] if external_prompts else []
        arrival = 0.0
        for _ in external_prompts[1:]:
            arrival += rng.expovariate(args.request_rate)
            offsets.append(arrival)

    resident_tasks = [
        asyncio.create_task(resident(index, prompt))
        for index, prompt in enumerate(resident_prompts)
    ]
    external_tasks = [
        asyncio.create_task(
            _stream_request(
                engine,
                prompt,
                {
                    **external_sampling,
                    "max_new_tokens": max(
                        1,
                        external_output_tokens[index]
                        if external_output_tokens is not None
                        else args.external_output_tokens,
                    ),
                },
                kind="external",
                index=index,
                request_id=(
                    external_request_ids[index]
                    if external_request_ids is not None
                    else None
                ),
                gate=(
                    None
                    if resident_offsets is not None and external_offsets is not None
                    else residents_started
                ),
                first_token_event=None,
                offset_seconds=offsets[index],
                load_start_seconds=started,
            )
        )
        for index, prompt in enumerate(external_prompts)
    ]
    records = await asyncio.gather(*(resident_tasks + external_tasks))
    return records, time.perf_counter() - started


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _latency_percentiles(records: list[dict[str, Any]], field: str) -> dict[str, float]:
    values = [float(record[field]) for record in records]
    return {
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


def _itl_values(records: list[dict[str, Any]]) -> list[float]:
    return [
        float(interval)
        for record in records
        for interval in record["inter_token_seconds"]
    ] or [0.0]


def main() -> int:
    args = parse_args()
    requested_cpu_affinity = _parse_cpu_affinity(args.cpu_affinity)
    if requested_cpu_affinity is not None:
        os.sched_setaffinity(0, requested_cpu_affinity)
        os.environ[_CALIBRATION_CPU_AFFINITY_ENV] = ",".join(
            str(cpu) for cpu in sorted(requested_cpu_affinity)
        )
    if args.numa_node is not None:
        requested_numa_node = str(args.numa_node)
        configured_numa_node = os.environ.get("SGLANG_HICACHE_HOST_NUMA_NODE")
        if configured_numa_node not in {None, requested_numa_node}:
            raise RuntimeError(
                "SGLang scheduler and HiCache allocator NUMA nodes disagree: "
                f"scheduler={requested_numa_node}, "
                f"allocator={configured_numa_node}"
            )
        os.environ["SGLANG_HICACHE_HOST_NUMA_NODE"] = requested_numa_node
    workspace = configure_environment(args)
    prior_stats_paths = set(workspace.glob("nta-engine.*.json"))
    import sglang as sgl
    from transformers import AutoTokenizer

    workload_metadata: dict[str, Any] | None = None
    resident_offsets: Sequence[float] | None = None
    external_offsets: Sequence[float] | None = None
    resident_output_tokens: Sequence[int] | None = None
    external_output_tokens: Sequence[int] | None = None
    resident_request_ids: Sequence[str] | None = None
    external_request_ids: Sequence[str] | None = None
    tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()))
    if args.workload_manifest is not None:
        loaded_workload = _load_workload(
            args.workload_manifest,
            tokenizer,
            external_suffix_tokens=args.external_suffix_tokens,
        )
        loaded_workload = _configure_workload_arrivals(
            loaded_workload,
            target_rate=args.request_rate,
            scale_to_target=args.scale_workload_arrivals_to_request_rate,
        )
        resident_request_ids = loaded_workload.resident_request_ids
        external_request_ids = loaded_workload.external_request_ids
        resident_prompts = list(loaded_workload.resident_inputs)
        external_prompts = list(loaded_workload.external_inputs)
        resident_offsets = loaded_workload.resident_arrival_offsets
        external_offsets = loaded_workload.external_arrival_offsets
        resident_output_tokens = loaded_workload.resident_output_tokens
        external_output_tokens = loaded_workload.external_output_tokens
        workload_metadata = loaded_workload.metadata
        if workload_metadata["tokenization_errors"]:
            raise RuntimeError(
                "Bailian structure prompt could not preserve exact tokenizer lengths; "
                "use a tokenizer-compatible prompt adapter before claiming serving evidence"
            )
        prompt_lengths = [
            *map(len, resident_prompts),
            *map(len, external_prompts),
        ]
        output_lengths = [
            *workload_metadata["resident_output_tokens"],
            *workload_metadata["external_output_tokens"],
        ]
        max_request_input_tokens = _max_request_input_tokens(
            args.context_length, args.max_total_tokens
        )
        if any(length >= max_request_input_tokens for length in prompt_lengths):
            raise RuntimeError(
                "Bailian prompt exceeds the SGLang request input budget "
                f"({max_request_input_tokens} tokens for context length "
                f"{args.context_length})"
            )
        if any(
            input_length + output_length > args.context_length
            for input_length, output_length in zip(prompt_lengths, output_lengths)
        ):
            raise RuntimeError(
                "Bailian input and output lengths exceed the configured context "
                f"length ({args.context_length})"
            )
        args.resident_requests = len(resident_prompts)
        args.external_requests = len(external_prompts)
        args.resident_tokens = max(workload_metadata["resident_input_tokens"])
        args.external_tokens = max(workload_metadata["source_external_input_tokens"])
        cached_prefix_lengths = [
            int(value) for value in workload_metadata["external_cached_prefix_tokens"]
        ]
        if len(cached_prefix_lengths) != len(external_prompts) or any(
            length <= 0 or length >= len(prompt)
            for length, prompt in zip(
                cached_prefix_lengths, external_prompts, strict=True
            )
        ):
            raise RuntimeError(
                "Bailian serving replay requires a nonempty exact shared prefix "
                "and at least one uncached query token for every external request"
            )
        external_prefixes = [
            prompt[:length]
            for prompt, length in zip(
                external_prompts, cached_prefix_lengths, strict=True
            )
        ]
        # Shape setup must never insert a timed request's continuation into the
        # radix cache. Only the token count is shared with the measured input.
        shape_prompt = _token_input(
            tokenizer,
            make_prompt(tokenizer, "load-shape-workload", len(external_prompts[0])),
        )
    else:
        external_prefixes = [
            _token_input(
                tokenizer,
                make_prompt(tokenizer, f"load-external-{index}", args.external_tokens),
            )
            for index in range(args.external_requests)
        ]
        external_prompts = [
            (
                prefix
                if args.external_suffix_tokens == 0
                else prefix
                + _token_input(
                    tokenizer,
                    make_prompt(
                        tokenizer,
                        f"load-external-suffix-{index}",
                        args.external_suffix_tokens,
                    ),
                )
            )
            for index, prefix in enumerate(external_prefixes)
        ]
        resident_prompts = [
            _token_input(
                tokenizer,
                make_prompt(tokenizer, f"load-resident-{index}", args.resident_tokens),
            )
            for index in range(args.resident_requests)
        ]
        shape_prompt = _token_input(
            tokenizer,
            make_prompt(
                tokenizer,
                "load-shape",
                max(len(prompt) for prompt in external_prompts),
            ),
        )
    external_cached_prefix_lengths = [
        _reusable_prefix_tokens(prefix, prompt)
        for prefix, prompt in zip(external_prefixes, external_prompts, strict=True)
    ]
    initial_cached_objects = [
        *zip(external_prefixes, external_cached_prefix_lengths, strict=True),
        *((prompt, len(prompt) - 1) for prompt in resident_prompts),
    ]
    effective_external_cached_prefix_lengths = list(
        effective_cached_prefixes(
            [(prompt, len(prompt)) for prompt in external_prompts],
            initial_cached_objects,
        )
    )
    if workload_metadata is not None and (
        effective_external_cached_prefix_lengths
        != [
            int(value)
            for value in workload_metadata["external_effective_cached_prefix_tokens"]
        ]
    ):
        raise RuntimeError(
            "runtime token inputs disagree with the workload cache-union contract"
        )
    external_query_rows = [
        len(prompt) - cached
        for cached, prompt in zip(
            effective_external_cached_prefix_lengths, external_prompts, strict=True
        )
    ]
    external_materialization_prompts = _exact_prefix_materialization_inputs(
        tokenizer, external_prefixes, external_prompts
    )
    placement_probe_groups = _placement_probe_groups(external_prefixes)
    resident_probe_groups = _placement_probe_groups(resident_prompts)
    external_materialization_representatives = tuple(
        group[0] for group in placement_probe_groups
    )
    if any(
        len(prompt)
        >= _max_request_input_tokens(args.context_length, args.max_total_tokens)
        for prompt in [*resident_prompts, *external_prompts]
    ):
        raise RuntimeError(
            "a generated prompt exceeds the SGLang request input budget "
            f"({_max_request_input_tokens(args.context_length, args.max_total_tokens)} "
            "tokens for "
            f"context length {args.context_length})"
        )
    eviction_rounds = (
        args.eviction_rounds
        if args.eviction_rounds is not None
        else args.max_total_tokens // args.churn_tokens + 1
    )
    setup_sampling = {"temperature": 0, "max_new_tokens": 1}
    resident_cache_tokens = 0
    external_cache_tokens = 0
    combined_cache_tokens = 0
    shared_cache_tokens = 0
    placement_page_tokens = (
        int(workload_metadata["block_size"]) if workload_metadata is not None else 1
    )
    placement_eviction_token_counts: list[int] = []
    if workload_metadata is not None:
        block_size = int(workload_metadata["block_size"])
        resident_cache_tokens = (
            int(workload_metadata["resident_input_cache_pages"]) * block_size
        )
        external_cache_tokens = (
            int(workload_metadata["external_input_cache_pages"]) * block_size
        )
        combined_cache_tokens = (
            int(workload_metadata["combined_input_cache_pages"]) * block_size
        )
        shared_cache_tokens = (
            int(workload_metadata["shared_input_cache_pages"]) * block_size
        )
        if resident_cache_tokens > args.max_total_tokens:
            raise RuntimeError(
                "resident input working set cannot fit in the configured SGLang "
                f"KV pool ({resident_cache_tokens} > {args.max_total_tokens} tokens)"
            )
        # SGLang's reusable radix/HiCache prefix is the request input. The
        # generated completion is not itself a prefix of the next request, so
        # it does not provide reliable pressure for this placement phase.
        # The cohort labels every external request as a host-tier dependency.
        # Merely crossing the combined working-set size evicts an arbitrary LRU
        # suffix and can leave a short external prefix entirely on device.  A
        # pool-sized unique pressure window starts that eviction. The trailing
        # external-working-set window covers pages temporarily pinned by
        # asynchronous write-through. Warming residents afterwards then
        # restores only the exact pages shared with the resident set.
    required_placement_pressure = _required_placement_pressure_tokens(
        device_pool_tokens=args.max_total_tokens,
        page_tokens=placement_page_tokens,
        external_cache_tokens=external_cache_tokens,
        largest_external_object_tokens=(
            max(workload_metadata["external_cached_prefix_tokens"])
            if workload_metadata is not None
            else 0
        ),
        exact_manifest=workload_metadata is not None,
        eviction_rounds=args.eviction_rounds,
        churn_tokens=args.churn_tokens,
    )
    remaining = required_placement_pressure
    maximum_pressure_prompt_tokens = min(
        args.churn_tokens,
        _max_request_input_tokens(args.context_length, args.max_total_tokens) - 1,
    )
    while remaining > 0:
        token_count = min(maximum_pressure_prompt_tokens, remaining)
        placement_eviction_token_counts.append(token_count)
        remaining -= token_count

    measurement_baseline: dict[str, dict[str, Any]] = {}
    cpu_affinity_contract: dict[str, Any] = {}
    setup_cache_flush_wait_seconds: list[float] = []
    placement_pressure_applications = 0
    placement_pressure_total_tokens = 0
    placement_probe_history: list[dict[str, Any]] = []
    final_placement_proof: dict[str, Any] = {}
    resident_placement_history: list[dict[str, Any]] = []
    pressure_forbidden_first_tokens = {
        int(prompt[0])
        for prompt in (
            *resident_prompts,
            *external_prompts,
            *external_materialization_prompts,
            shape_prompt,
        )
    }
    load_started = time.perf_counter()
    with sgl.Engine(
        model_path=str(args.model.resolve()),
        attention_backend=args.attention_backend,
        dtype="float16",
        mem_fraction_static=args.mem_fraction_static,
        context_length=args.context_length,
        max_total_tokens=args.max_total_tokens,
        max_running_requests=args.max_running_requests,
        cuda_graph_backend_decode=args.cuda_graph_decode,
        cuda_graph_backend_prefill=args.cuda_graph_prefill,
        chunked_prefill_size=(
            args.chunked_prefill_size
            if args.chunked_prefill_size > 0
            else args.context_length
        ),
        enable_mixed_chunk=args.batch_mode == "coalesced",
        enable_hierarchical_cache=True,
        hicache_ratio=args.hicache_ratio,
        # Exact external prefixes are touched twice; one-shot placement
        # pressure is not. Selective write-through therefore persists demand
        # without polluting L2 with setup-only eviction objects.
        hicache_write_policy=_HICACHE_WRITE_POLICY,
        hicache_io_backend="kernel",
        hicache_mem_layout="page_first",
        numa_node=None if args.numa_node is None else [args.numa_node],
    ) as engine:
        cpu_affinity_contract = _apply_engine_cpu_affinity(
            engine, requested_cpu_affinity
        )

        def warm_residents(*, reason: str) -> None:
            """Materialize, then exactly verify the resident working set."""

            observations: list[dict[str, int]] = []
            for begin in range(
                0, len(resident_probe_groups), args.max_running_requests
            ):
                end = begin + args.max_running_requests
                groups = resident_probe_groups[begin:end]
                prompts = [resident_prompts[group[0]] for group in groups]
                samplings = [dict(setup_sampling)] * len(prompts)
                results = generation_results(_generate_many(engine, prompts, samplings))
                if any(
                    device_cached_tokens(result) != len(prompt) - 1
                    or host_cached_tokens(result) != 0
                    for prompt, result in zip(prompts, results, strict=True)
                ):
                    # A cold first access creates the radix entry; it is not
                    # evidence that placement failed. The second access is the
                    # exact cache-state observation used by the setup gate.
                    results = generation_results(
                        _generate_many(
                            engine,
                            prompts,
                            [dict(setup_sampling)] * len(prompts),
                        )
                    )
                for group, prompt, result in zip(groups, prompts, results, strict=True):
                    actual_device = device_cached_tokens(result)
                    actual_host = host_cached_tokens(result)
                    if actual_device != len(prompt) - 1 or actual_host != 0:
                        raise RuntimeError(
                            "resident warmup did not establish its exact device "
                            "prefix: "
                            f"expected={len(prompt) - 1}, "
                            f"device={actual_device}, host={actual_host}"
                        )
                    for index in group:
                        if resident_prompts[index] != prompt:
                            raise RuntimeError(
                                "equal resident setup identities disagree"
                            )
                        observations.append(
                            {
                                "index": index,
                                "expected": len(prompt) - 1,
                                "device": actual_device,
                                "host": actual_host,
                                "representative_index": group[0],
                                "shared_identity_group_size": len(group),
                            }
                        )
            observations.sort(key=lambda value: value["index"])
            resident_placement_history.append(
                {"reason": reason, "observations": observations}
            )

        load_seconds = time.perf_counter() - load_started
        generated_text(_generate_one(engine, shape_prompt, setup_sampling))
        generated_text(_generate_one(engine, shape_prompt, setup_sampling))
        # Shape/JIT warmup uses a deliberately disjoint prompt so it cannot
        # pre-populate the measured request.  It must not remain in HiCache,
        # either: retaining that otherwise-dead prefix adds one full-context
        # object to the setup working set and can evict the exact external
        # prefix from a host pool that is large enough for the measured
        # placement.  JIT modules and allocator state survive a radix-cache
        # flush, so clear only cache contents before constructing placement.
        setup_cache_flush_wait_seconds.append(
            _flush_cache_when_idle(
                engine,
                timeout_seconds=args.setup_idle_timeout_seconds,
                reason="clear shape warmup before external placement",
            )
        )

        def warm_external_prefixes() -> None:
            """Create a reusable, write-through-backed external prefix.

            SGLang may mark a long prefill as ``chunked`` and deliberately
            defer its write-through hit-count update.  A second setup hit is
            therefore part of the placement protocol: it makes the external
            prefix eligible for D→H backup before the pressure phase.  This
            is setup-only and never contributes to the timed records.
            """

            inputs = [
                external_materialization_prompts[index]
                for index in external_materialization_representatives
            ]
            generation_results(
                _generate_many(
                    engine,
                    inputs,
                    [dict(setup_sampling)] * len(inputs),
                )
            )
            generation_results(
                _generate_many(
                    engine,
                    inputs,
                    [dict(setup_sampling)] * len(inputs),
                )
            )

        def apply_placement_pressure() -> None:
            """Apply one disjoint, L2-ineligible pressure window."""

            nonlocal placement_pressure_applications
            nonlocal placement_pressure_total_tokens
            application = placement_pressure_applications
            for index, token_count in enumerate(placement_eviction_token_counts):
                prompt = _unique_pressure_input(
                    tokenizer,
                    label=f"load-placement-pressure-{application}-{index}",
                    token_count=token_count,
                    forbidden_first_tokens=pressure_forbidden_first_tokens,
                )
                generated_text(_generate_one(engine, prompt, setup_sampling))
                placement_pressure_total_tokens += len(prompt)
            placement_pressure_applications += 1

        def construct_external_placement(*, reason: str) -> None:
            """Build and destructively verify the exact host-backed frontier."""

            nonlocal final_placement_proof
            warm_external_prefixes()
            apply_placement_pressure()
            for attempt in range(1, 4):
                observations: list[dict[str, int]] = []
                missing: list[dict[str, int]] = []
                # A probe is itself a promotion. Verify one representative of
                # each content object, shortest prefix first, so a longer
                # shared radix object cannot invalidate a later observation
                # of its strict prefix. Equal prefixes share one physical
                # placement and therefore one source observation.
                for group in placement_probe_groups:
                    representative = group[0]
                    prompt = external_materialization_prompts[representative]
                    expected = external_cached_prefix_lengths[representative]
                    result = _generate_one(engine, prompt, setup_sampling)
                    device = device_cached_tokens(result)
                    host = host_cached_tokens(result)
                    group_missing = host <= 0 or host + device != expected
                    for index in group:
                        if external_cached_prefix_lengths[index] != expected:
                            raise RuntimeError(
                                "equal placement identities disagree on token length"
                            )
                        observation = {
                            "index": index,
                            "expected": int(expected),
                            "device": device,
                            "host": host,
                            "representative_index": representative,
                            "shared_identity_group_size": len(group),
                        }
                        observations.append(observation)
                        if group_missing:
                            missing.append(observation)
                    if group_missing:
                        # One repair hit closes selective write-through for a
                        # cold or still-device-only radix leaf.
                        generated_text(_generate_one(engine, prompt, setup_sampling))
                observations.sort(key=lambda value: value["index"])
                missing.sort(key=lambda value: value["index"])
                placement_probe_history.append(
                    {
                        "reason": reason,
                        "attempt": attempt,
                        "missing": missing,
                    }
                )
                # The probe is destructive because it promotes host pages.
                # A fresh disjoint pressure window replays the proven split.
                apply_placement_pressure()
                if not missing:
                    final_placement_proof = {
                        "reason": reason,
                        "attempt": attempt,
                        "observations": observations,
                        "destructive_probe_followed_by_disjoint_replay": True,
                        "probe_order": "shortest_distinct_prefix_first",
                        "unique_prefix_groups": len(placement_probe_groups),
                    }
                    return
            raise RuntimeError(
                "external placement did not converge after bounded backing probes: "
                + json.dumps(missing, sort_keys=True)
            )

        construct_external_placement(reason="jit_warmup")
        warm_residents(reason="jit_warmup")

        def establish_final_placement() -> None:
            """Rebuild the requested device/host split from an empty cache.

            Warmup requests intentionally exercise the same mixed scheduler
            path as the timed load, but every exact calibration suffix creates
            a distinct radix branch. Merely applying more LRU pressure leaves
            those branches resident and makes TPOT depend on the configured
            warmup count. Flush and deterministically reconstruct placement
            after every excluded warmup so the measured cache topology, not
            only its token totals, is identical across causal arms.

            A normalized Bailian cohort has an exact placement contract.  LRU
            pressure alone cannot reconstruct that contract; its registered
            eviction prompts remain the authoritative sequence. Synthetic
            workloads reuse one fixed cold churn window after each flush.
            This setup is outside the timed window and is identical for every
            causal arm.
            """

            setup_cache_flush_wait_seconds.append(
                _flush_cache_when_idle(
                    engine,
                    timeout_seconds=args.setup_idle_timeout_seconds,
                    reason="reset cache before rebuilding measured placement",
                )
            )
            construct_external_placement(reason="measured_reconstruction")
            warm_residents(reason="measured_reconstruction")

        # The placement above was built from the same empty-cache boundary as
        # ``establish_final_placement`` and is already destructively verified.
        # Rebuilding it again before the first excluded warmup performed no
        # semantic work and accounted for one quarter of setup reconstruction
        # in the formal two-warmup campaign. With no warmup, retain the rebuild
        # so diagnostic runs still emit the measured-reconstruction proof
        # required by the serving report contract.
        if args.load_warmup_iterations == 0:
            establish_final_placement()

        calibration_shape_records: list[list[dict[str, int]]] = []
        for _warmup in range(args.load_warmup_iterations):
            # Demand graphs warm on the first occurrence and capture on the
            # second. Both are excluded so the measured occurrence is replay.
            # Replaying the exact token inputs also preserves cross-request
            # content sharing and controlled-cycle cache evolution. Every
            # excluded occurrence is followed by a cache flush and exact
            # placement reconstruction, so no warmup radix entry reaches the
            # timed window.
            warmup_records, _ = engine.loop.run_until_complete(
                _run_load(
                    engine,
                    resident_prompts,
                    external_prompts,
                    args,
                    resident_offsets,
                    external_offsets,
                    resident_output_tokens,
                    external_output_tokens,
                    resident_request_ids,
                    external_request_ids,
                )
            )
            calibration_shape_records.append(
                [
                    {
                        "index": int(record["index"]),
                        "input_tokens": int(record["input_tokens"]),
                        "host_cached_tokens": int(record["host_cached_tokens"]),
                        "device_cached_tokens": int(record["device_cached_tokens"]),
                    }
                    for record in warmup_records
                    if record["kind"] == "external"
                ]
            )
            establish_final_placement()

        expected_setup_flushes = (
            2 if args.load_warmup_iterations == 0 else 1 + args.load_warmup_iterations
        )
        if len(setup_cache_flush_wait_seconds) != expected_setup_flushes:
            raise RuntimeError(
                "serving setup performed a redundant cache reconstruction: "
                f"expected {expected_setup_flushes} idle flushes, observed "
                f"{len(setup_cache_flush_wait_seconds)}"
            )

        # SGLang's startup and cache-management workers may replace process
        # affinity after Engine construction. Re-apply and verify the declared
        # contract at the final quiescent boundary immediately before timing;
        # this is the affinity that the serving result is allowed to claim.
        cpu_affinity_contract = _apply_engine_cpu_affinity(
            engine, requested_cpu_affinity
        )
        # Delimit the timed counter window with a control-plane snapshot. It
        # quiesces prior CUDA observations without issuing a model request or
        # changing cache placement.
        measurement_baseline = (
            _publish_engine_stats_snapshot(engine, workspace, prior_stats_paths)
            if args.attention_backend == "nta_flashinfer"
            else {}
        )
        if args.attention_backend == "nta_flashinfer":
            _require_closed_auto_calibration(
                measurement_baseline,
                calibration_training_run=args.auto_calibration_training_run,
            )
        records, elapsed = engine.loop.run_until_complete(
            _run_load(
                engine,
                resident_prompts,
                external_prompts,
                args,
                resident_offsets,
                external_offsets,
                resident_output_tokens,
                external_output_tokens,
                resident_request_ids,
                external_request_ids,
            )
        )
        # The timed wall-clock stops before this out-of-band completion
        # boundary, so statistics publication is excluded from latency.
        measurement_final = (
            _publish_engine_stats_snapshot(engine, workspace, prior_stats_paths)
            if args.attention_backend == "nta_flashinfer"
            else {}
        )

    external = [record for record in records if record["kind"] == "external"]
    resident = [record for record in records if record["kind"] == "resident"]
    expected_external_prefixes = (
        [
            int(value)
            for value in workload_metadata["external_effective_cached_prefix_tokens"]
        ]
        if workload_metadata is not None
        else effective_external_cached_prefix_lengths
    )
    try:
        cache_binding_contract, cache_state_transitions = annotate_timed_cache_bindings(
            records,
            resident_inputs=resident_prompts,
            external_inputs=external_prompts,
            external_materialized_prefix_tokens=external_cached_prefix_lengths,
            external_initial_prefix_tokens=expected_external_prefixes,
        )
    except ValueError as error:
        raise RuntimeError(
            f"timed cache-binding evidence is invalid: {error}"
        ) from error
    minimum_host_prefix = min(record["host_cached_tokens"] for record in external)
    execution_ready_external = [
        int(record["index"])
        for record in external
        if record["host_cached_tokens"] == 0 and record["device_cached_tokens"] > 0
    ]
    uncached_external = [
        int(record["index"])
        for record in external
        if record["host_cached_tokens"] == 0 and record["device_cached_tokens"] == 0
    ]
    if not resident_placement_history or (
        resident_placement_history[-1]["reason"] != "measured_reconstruction"
    ):
        raise RuntimeError("final resident placement was not reconstructed")

    def external_shapes(values: Sequence[dict[str, Any]]) -> list[dict[str, int]]:
        return sorted(
            (
                {
                    "index": int(record["index"]),
                    "input_tokens": int(record["input_tokens"]),
                    "host_cached_tokens": int(record["host_cached_tokens"]),
                    "device_cached_tokens": int(record["device_cached_tokens"]),
                }
                for record in values
                if record["kind"] == "external"
            ),
            key=lambda value: value["index"],
        )

    timed_external_shapes = external_shapes(records)
    # The first excluded occurrence is a stabilization pass: CUDA graph/JIT
    # state and the framework's asynchronous device/host placement converge
    # there.  It must preserve token identity and query geometry, but its
    # instantaneous tier split is not a calibration claim.  The final
    # excluded occurrence is the shape calibration and must match the timed
    # device/host split exactly.  Treating both passes as calibrations made a
    # harmless first-touch page placement (with the same total exact prefix)
    # invalidate an otherwise shape-identical trial.
    stabilization_shapes = calibration_shape_records[:-1]
    shape_calibrations = calibration_shape_records[-1:]
    mismatched_stabilizations = [
        {
            "warmup": warmup,
            "stabilization": sorted(shape, key=lambda value: value["index"]),
            "timed": timed_external_shapes,
        }
        for warmup, shape in enumerate(stabilization_shapes)
        if sorted(shape, key=lambda value: value["index"]) != timed_external_shapes
    ]
    mismatched_calibrations = [
        {
            "warmup": len(stabilization_shapes) + calibration,
            "calibration": sorted(shape, key=lambda value: value["index"]),
            "timed": timed_external_shapes,
        }
        for calibration, shape in enumerate(shape_calibrations)
        if sorted(shape, key=lambda value: value["index"]) != timed_external_shapes
    ]
    if not shape_calibrations or mismatched_calibrations:
        raise RuntimeError(
            "final performance calibration did not preserve exact cached-prefix "
            "placement and uncached query rows: "
            + json.dumps(mismatched_calibrations, sort_keys=True)
        )
    calibration_contract = {
        "kind": "exact_token_content_graph_and_query_rows",
        "cache_composition": "initial_object_union_longest_common_prefix",
        "content_graph_preserved": True,
        "cache_reset_after_each_warmup": True,
        "stabilization_iterations": len(stabilization_shapes),
        "stabilization_tier_split_mismatches": mismatched_stabilizations,
        "shape_calibration_iterations": len(shape_calibrations),
        "final_calibration_matches_timed": not mismatched_calibrations,
        "tier_split_preserved": not mismatched_calibrations,
        "verified": len(shape_calibrations) == 1 and not mismatched_calibrations,
        "warmup_iterations": len(calibration_shape_records),
        "materialized_prefix_tokens": external_cached_prefix_lengths,
        "cached_prefix_tokens": effective_external_cached_prefix_lengths,
        "uncached_query_rows": external_query_rows,
        "timed_shapes": timed_external_shapes,
    }
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda value: (value["kind"], value["index"])):
        text = record.pop("text").encode("utf-8")
        record["text_sha256"] = hashlib.sha256(text).hexdigest()
        digest.update(text)
        digest.update(b"\0")
    cumulative_stats_by_name = (
        measurement_final
        if args.attention_backend == "nta_flashinfer"
        else _read_engine_stats(workspace, prior_stats_paths)
    )
    cumulative_stats = list(cumulative_stats_by_name.values())
    if args.attention_backend == "nta_flashinfer":
        if not measurement_baseline or (
            set(measurement_baseline) != set(cumulative_stats_by_name)
        ):
            raise RuntimeError(
                "NTA measurement baseline and final worker statistics disagree"
            )
        stats = [
            _measurement_delta(report, measurement_baseline[name])
            for name, report in sorted(cumulative_stats_by_name.items())
        ]
    else:
        stats = []
    _require_no_timed_auto_calibration(
        stats,
        calibration_training_run=args.auto_calibration_training_run,
    )
    execution_dispatch = (
        _execution_dispatch(stats)
        if args.attention_backend == "nta_flashinfer"
        else {"kind": "framework_reference"}
    )
    engine_version = importlib.metadata.version("sglang")
    consumer_contract: dict[str, Any] | None = None
    consumer_contracts: list[dict[str, Any]] = []
    if args.attention_backend == "nta_flashinfer":
        contracts_by_kind: dict[str, dict[str, Any]] = {}
        for entry in stats:
            if not isinstance(entry, dict) or entry.get("backend") != "nta_flashinfer":
                continue
            path_contracts = entry.get("consumer_contracts")
            if not isinstance(path_contracts, list):
                path_contracts = [entry.get("consumer_contract")]
            for contract in path_contracts:
                if isinstance(contract, dict) and isinstance(contract.get("kind"), str):
                    contracts_by_kind[contract["kind"]] = contract
        consumer_contracts = [
            contracts_by_kind[kind]
            for kind in ("native_work_unit", "framework_reference", "projection_only")
            if kind in contracts_by_kind
        ]
        # Prefer proof that the native work-unit consumer launched.  If the
        # report only contains a projection/reference contract, preserve it so
        # the formal evaluator can reject the trial with an actionable reason
        # instead of silently manufacturing evidence.
        consumer_contract = consumer_contracts[0] if consumer_contracts else None
    else:
        # Stock FlashInfer is the numerical framework reference.  It is not a
        # typed NTA work-unit consumer, but it still consumes the same exact
        # demand and must be represented explicitly in a formal arm.
        consumer_contract = {
            "schema": 1,
            "engine": "sglang",
            "backend": "flashinfer",
            "kind": "framework_reference",
            "exact_demand": True,
            "typed_work_plan": False,
            "native_submission": False,
            "numerical_consumer": True,
            "engine_version": engine_version,
        }
        consumer_contracts = [consumer_contract]
    tier_entries = {
        str(entry["serving_tier"])
        for entry in stats
        if isinstance(entry, dict) and "serving_tier" in entry
    }
    if len(tier_entries) > 1:
        raise RuntimeError("NTA worker processes disagree on the serving tier")
    serving_tier = next(iter(tier_entries), "host_staged")
    tier_catalog_digests = {
        str(entry["tier_catalog_digest"])
        for entry in stats
        if isinstance(entry, dict) and entry.get("tier_catalog_digest")
    }
    if len(tier_catalog_digests) > 1:
        raise RuntimeError("NTA worker processes disagree on the tier catalog")
    total_tokens = sum(record["completion_tokens"] for record in records)
    selected_tokens = sum(
        record["host_cached_tokens"] + record["device_cached_tokens"]
        for record in records
    )
    physical_tokens = sum(
        record["host_cached_tokens"] + record["device_cached_tokens"]
        for record in records
    )
    admission_delays = [record["admission_delay_seconds"] for record in records]
    finite_window_accounting = finite_window_system_accounting(records, elapsed)
    ttft = _latency_percentiles(records, "ttft_seconds")
    tpot = _latency_percentiles(records, "tpot_seconds")
    itl_values = _itl_values(records)
    itl = {
        "p50": _percentile(itl_values, 0.50),
        "p95": _percentile(itl_values, 0.95),
        "p99": _percentile(itl_values, 0.99),
    }
    slo_goodput = joint_slo_goodput(
        {"records": records, "elapsed_seconds": elapsed},
        ttft_seconds=args.slo_ttft_seconds,
        tpot_seconds=args.slo_tpot_seconds,
        p99_itl_seconds=args.slo_p99_itl_seconds,
    )
    correctness = {
        "verification_failures": 0,
        "placement_proven": True,
        "placement_reset_between_warmups": True,
        "generated_text_sha256": digest.hexdigest(),
        "demand_trace_digest": (
            workload_metadata["demand_trace_digest"]
            if workload_metadata is not None
            else None
        ),
    }
    selected_bytes, physical_bytes, byte_accounting_status = _engine_byte_accounting(
        stats
    )
    batch_heterogeneity = serving_batch_heterogeneity(records, stats)
    if (
        args.attention_backend == "nta_flashinfer"
        and not args.auto_calibration_training_run
        and args.batch_mode == "coalesced"
        and resident
        and external
        and not batch_heterogeneity["proven"]
    ):
        raise RuntimeError(
            "coalesced workload did not produce a measured heterogeneous "
            "SGLang ForwardBatch; increase resident decode lifetime or fix "
            f"arrival shaping: {batch_heterogeneity!r}"
        )
    report = {
        "schema": 1,
        "classification": "sglang-hicache-load",
        "revision": os.environ.get("NTA_REVISION") or git_value("rev-parse", "HEAD"),
        "dirty": bool(git_value("status", "--porcelain")),
        "engine": "sglang",
        "machine": _machine_metadata(),
        "engine_version": engine_version,
        "flashinfer_version": importlib.metadata.version("flashinfer-python"),
        "attention_backend": args.attention_backend,
        "serving_tier": serving_tier,
        "tier_catalog_digest": next(iter(tier_catalog_digests), None),
        "model": str(args.model.resolve()),
        "seed": args.seed,
        "request_rate": args.request_rate,
        "arrival_schedule": (
            workload_metadata["runtime_arrival"]
            if workload_metadata is not None
            else {
                "method": "seeded_exponential",
                "source_mode": "synthetic_open_loop",
                "source_target_rate_per_second": args.request_rate,
                "target_rate_per_second": args.request_rate,
                "uniform_time_scale": 1.0,
                "request_order_preserved": True,
            }
        ),
        "external_requests": args.external_requests,
        "external_tokens": args.external_tokens,
        "external_suffix_tokens": args.external_suffix_tokens,
        "minimum_external_host_cached_tokens": minimum_host_prefix,
        "execution_ready_external_requests": execution_ready_external,
        "uncached_external_requests": uncached_external,
        "resident_requests": args.resident_requests,
        "resident_tokens": args.resident_tokens,
        "resident_output_tokens": args.resident_output_tokens,
        "external_output_tokens": args.external_output_tokens,
        "eviction_rounds": eviction_rounds,
        "placement_eviction_rounds": len(placement_eviction_token_counts),
        "placement_eviction_tokens": placement_pressure_total_tokens,
        "placement_pressure_applications": placement_pressure_applications,
        "placement_probe_history": placement_probe_history,
        "initial_placement_proof": final_placement_proof,
        "initial_resident_placement_proof": resident_placement_history[-1],
        "cache_state_transitions": cache_state_transitions,
        "resident_input_cache_tokens": resident_cache_tokens,
        "external_input_cache_tokens": external_cache_tokens,
        "combined_input_cache_tokens": combined_cache_tokens,
        "shared_input_cache_tokens": shared_cache_tokens,
        "required_placement_pressure_tokens": required_placement_pressure,
        "churn_tokens": args.churn_tokens,
        "max_total_tokens": args.max_total_tokens,
        "context_length": args.context_length,
        "mem_fraction_static": args.mem_fraction_static,
        "max_running_requests": args.max_running_requests,
        "numa_node": args.numa_node,
        "cpu_affinity": cpu_affinity_contract,
        "batch_mode": args.batch_mode,
        "mixed_chunk_enabled": args.batch_mode == "coalesced",
        "chunked_prefill_size": (
            args.chunked_prefill_size
            if args.chunked_prefill_size > 0
            else args.context_length
        ),
        "hicache_ratio": args.hicache_ratio,
        "hicache_write_policy": _HICACHE_WRITE_POLICY,
        "cuda_graph_decode": args.cuda_graph_decode,
        "cuda_graph_prefill": args.cuda_graph_prefill,
        "auto_calibration_training_run": args.auto_calibration_training_run,
        "load_warmup_iterations": args.load_warmup_iterations,
        "setup_cache_flush": {
            "contract": "sglang_deferred_fully_idle",
            "timeout_seconds": args.setup_idle_timeout_seconds,
            "count": len(setup_cache_flush_wait_seconds),
            "wait_seconds_total": sum(setup_cache_flush_wait_seconds),
            "wait_seconds_max": max(setup_cache_flush_wait_seconds, default=0.0),
            "excluded_from_timed_window": True,
        },
        "load_warmup_excluded": (
            args.load_warmup_iterations >= 2 and calibration_contract["verified"]
        ),
        "calibration_input_contract": calibration_contract,
        "cache_binding_contract": cache_binding_contract,
        "batch_heterogeneity": batch_heterogeneity,
        "workload": workload_metadata,
        "demand_semantics": "exact",
        "execution_dispatch": execution_dispatch,
        "demand_trace_digest": (
            workload_metadata["demand_trace_digest"]
            if workload_metadata is not None
            else None
        ),
        "selected_bytes": selected_bytes,
        "physical_bytes": physical_bytes,
        "byte_accounting_status": byte_accounting_status,
        "selected_kv_tokens": selected_tokens,
        "physical_kv_tokens": physical_tokens,
        "load_seconds": load_seconds,
        "elapsed_seconds": elapsed,
        "request_throughput": len(records) / elapsed,
        "output_token_throughput": total_tokens / elapsed,
        "resident_output_token_throughput": (
            sum(int(record["completion_tokens"]) for record in resident) / elapsed
        ),
        "external_output_token_throughput": (
            sum(int(record["completion_tokens"]) for record in external) / elapsed
        ),
        "p50_ttft_seconds": ttft["p50"],
        "p95_ttft_seconds": ttft["p95"],
        "p99_ttft_seconds": ttft["p99"],
        "p50_tpot_seconds": tpot["p50"],
        "p95_tpot_seconds": tpot["p95"],
        "p99_tpot_seconds": tpot["p99"],
        "p99_itl_seconds": itl["p99"],
        "latency_percentiles": {
            "ttft_seconds": ttft,
            "tpot_seconds": tpot,
            "inter_token_seconds": itl,
        },
        "slo_goodput": slo_goodput,
        "generated_text_sha256": digest.hexdigest(),
        "placement_proven": True,
        "placement_reset_between_warmups": True,
        "verification_failures": 0,
        "correctness": correctness,
        "finite_window_accounting": finite_window_accounting,
        "admission_delay_seconds": {
            "mean": sum(admission_delays) / len(admission_delays),
            "p95": _percentile(admission_delays, 0.95),
            "scope": "client_admission_delay",
        },
        "records": records,
        "resident_p95_ttft_seconds": _percentile(
            [record["ttft_seconds"] for record in resident], 0.95
        ),
        "resident_p95_tpot_seconds": _percentile(
            [record["tpot_seconds"] for record in resident], 0.95
        ),
        "resident_p99_itl_seconds": _percentile(
            [
                interval
                for record in resident
                for interval in record["inter_token_seconds"]
            ],
            0.99,
        ),
        "external_p95_ttft_seconds": _percentile(
            [record["ttft_seconds"] for record in external], 0.95
        ),
        "engine_stats": stats,
        "engine_stats_cumulative": cumulative_stats,
    }
    if consumer_contract is not None:
        report["consumer_contract"] = consumer_contract
        report["consumer_contracts"] = consumer_contracts
    if args.output is not None:
        atomic_write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
