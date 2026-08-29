#!/usr/bin/env python3
"""Validate physical serving-path evidence without initializing CUDA."""

from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.serving_path_evidence import (  # noqa: E402
    require_exercised_paths,
    require_frontier_shape,
)
from experiments.mechanism_arms import validate_arm_result  # noqa: E402


def _arm_report(arm: str) -> dict:
    framework = arm == "A0"
    consumer = "framework_reference" if arm in {"A0", "A1"} else "native_work_unit"
    report = {
        "attention_backend": "flashinfer" if framework else "nta_flashinfer",
        "consumer_contract": {"kind": consumer},
        "records": [{"kind": "external", "host_cached_tokens": 8}],
        "engine_stats": [],
        "batch_heterogeneity": {
            "proven": arm == "A3",
            "native_mixed_consumer_proven": arm == "A3",
        },
    }
    if framework:
        return report
    counters = {
        "backend": "nta_flashinfer",
        "execution_protocol": "late_bound",
        "host_execution_mode": {
            "A1": "direct",
            "A2": "device_bulk",
            "A3": "dependency_aware",
        }[arm],
        "hicache_fallback_batches": 0,
        "hicache_external_batches": 1,
        "host_direct_batches": int(arm == "A1"),
        "host_device_bulk_batches": int(arm == "A2"),
        "host_incremental_batches": int(arm == "A3"),
        "external_launches": 36,
        "native_external_attention_launches": (
            0 if arm == "A1" else 1 if arm == "A3" else 36
        ),
        "stock_prefetched_external_attention_launches": (
            36 if arm == "A1" else 35 if arm == "A3" else 0
        ),
        "ticketed_incremental_launches": (
            0 if arm == "A1" else 1 if arm == "A3" else 36
        ),
        "event_ordered_incremental_launches": 1 if arm == "A3" else 0,
        "request_acquisition_groups": 0 if arm == "A1" else 1,
        "mixed_dependency_layers": 1 if arm == "A3" else 0,
        "progressive_consumer_batch_observations": 1,
        "progressive_consumer_batches": 1 if arm == "A3" else 0,
        "progressive_consumer_layers": 1 if arm == "A3" else 0,
        "prefetch_mover_plan_calibration_probe_sm_leases": 0,
        "prefetch_mover_plan_calibration_probe_copy_leases": 0,
        "verified_operator_modules": 0 if arm == "A1" else 2,
    }
    report["engine_stats"] = [counters]
    return report


def main() -> None:
    for arm in ("A0", "A1", "A2", "A3"):
        proof = validate_arm_result(_arm_report(arm), arm)
        assert proof["arm"] == arm
    mislabeled = _arm_report("A3")
    try:
        validate_arm_result(mislabeled, "A2")
    except ValueError as error:
        assert "A2" in str(error)
    else:
        raise AssertionError("a progressive run passed as device-bulk")
    ungrouped_bulk = _arm_report("A2")
    ungrouped_bulk["engine_stats"][0]["request_acquisition_groups"] = 0
    try:
        validate_arm_result(ungrouped_bulk, "A2")
    except ValueError as error:
        assert "A2" in str(error)
    else:
        raise AssertionError("an ungrouped bulk launch passed as exact A2")
    event_only = _arm_report("A3")
    event_only["engine_stats"][0]["ticketed_incremental_launches"] = 0
    event_only["engine_stats"][0]["request_acquisition_groups"] = 0
    try:
        validate_arm_result(event_only, "A3")
    except ValueError as error:
        assert "A3" in str(error)
    else:
        raise AssertionError("an event-only proactive prefetch passed as A3")
    calibration = _arm_report("A3")
    calibration["engine_stats"][0][
        "prefetch_mover_plan_calibration_probe_sm_leases"
    ] = 1
    try:
        validate_arm_result(calibration, "A3")
    except ValueError as error:
        assert "calibration probe" in str(error)
    else:
        raise AssertionError("a timed host-mover probe passed a causal arm")

    exercised_stats = [
        {
            "native_demand_sm_bytes": 64,
            "native_external_attention_launches": 1,
            "sm_mover_bytes": 128,
            "copy_engine_bytes": 256,
            "copy_engine_operations": 4,
            "copy_engine_submissions": 1,
            "hybrid_parallel_waves": 1,
            "progressive_consumer_layers": 1,
            "exact_resume_window_layers": 1,
            "stock_ready_external_attention_launches": 35,
            "host_progress_rounds": 4,
        }
    ]
    execution = require_exercised_paths(
        exercised_stats,
        [
            "native_demand_sm",
            "prefetch_sm",
            "prefetch_copy_engine",
            "prefetch_hybrid",
            "partial_consumer",
        ],
    )
    assert execution["frontier"] == {
        "native_layers": 1,
        "ready_stock_layers": 35,
        "progress_rounds": 4,
    }
    require_frontier_shape(
        execution,
        native_layers=1,
        ready_stock_layers=35,
        progressive_layers=1,
    )
    try:
        require_frontier_shape(
            execution,
            native_layers=36,
            ready_stock_layers=0,
            progressive_layers=36,
        )
    except ValueError as error:
        assert "frontier shape mismatch" in str(error)
    else:
        raise AssertionError("a forced-36 run masqueraded as the 1/35 frontier")

    mislabeled_copy_arm = [
        {
            "native_demand_sm_bytes": 256,
            "native_external_attention_launches": 36,
            "copy_engine_bytes": 0,
            "copy_engine_operations": 0,
            "copy_engine_submissions": 0,
        }
    ]
    try:
        require_exercised_paths(mislabeled_copy_arm, ["prefetch_copy_engine"])
    except ValueError as error:
        assert "prefetch_copy_engine" in str(error)
    else:
        raise AssertionError("a zero-byte copy-engine arm was accepted")


if __name__ == "__main__":
    main()
