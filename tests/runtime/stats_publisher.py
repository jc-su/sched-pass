#!/usr/bin/env python3
"""Test asynchronous statistics publication and shutdown ownership."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import threading


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from nta_runtime.engines.sglang_telemetry import (  # noqa: E402
    SglangTelemetryConfig,
    StatsPublisher,
    initial_engine_stats,
    process_hook_stats,
    record_forward,
    record_observability_degraded,
    record_prefill_graph,
)


def _telemetry_config() -> SglangTelemetryConfig:
    return SglangTelemetryConfig(
        model_layer_count=36,
        execution_protocol="late_bound",
        host_execution_mode="auto",
        work_granularity="page_group",
        protocol_max_inflight_units=8,
        runtime_tenant_capacity=64,
        runtime_staging_byte_capacity=1 << 30,
        tenant_isolation_enabled=True,
        overlap_enabled=True,
        frontier_enabled=False,
        fragment_enabled=False,
        demand_overlap_policy="none",
        stream_ordered_retirement_enabled=False,
        sglang_mixed_chunk_enabled=True,
        max_host_rounds=4,
        minimum_predicted_gain=1.03,
        incremental_setup_ns=None,
        incremental_service_scale=None,
        incremental_calibration_probes_remaining=0,
        cost_model_bandwidth_bps=30_000_000_000,
        host_mover="auto",
        copy_engine_max_operations=4096,
        host_mover_copy_calibrated=False,
        host_mover_calibration_samples_per_engine=3,
        host_mover_sm_samples=0,
        host_mover_copy_samples=0,
        host_mover_sm_bandwidth_bps=30_000_000_000,
        host_mover_copy_bandwidth_bps=None,
        host_mover_copy_operation_ns=None,
        host_mover_hybrid_join_ns=0,
        host_mover_minimum_gain=1.03,
        layer_service_minimum_samples=4,
        layer_service_maximum_samples=32,
        indexed_copy_target_bytes=1 << 20,
        indexed_copy_max_blocks=32,
        frontier_layers_per_wave=4,
        sm_acquisition_waves=4,
        sm_mover_max_worker_ctas=8,
        demand_graph_enabled=False,
        demand_graph_capacity=144,
        engine_version="test",
        revision="fixture",
    )


def main() -> None:
    before = process_hook_stats()

    def record_samples() -> None:
        for _ in range(100):
            record_forward("threaded_test", 0.25)

    workers = tuple(threading.Thread(target=record_samples) for _ in range(8))
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    record_prefill_graph("served")
    degraded = record_observability_degraded("threaded_test")
    after = process_hook_stats()
    assert (
        after["forward_threaded_test_count"]
        - before.get("forward_threaded_test_count", 0.0)
        == 800.0
    )
    assert (
        after["forward_threaded_test_ms_total"]
        - before.get("forward_threaded_test_ms_total", 0.0)
        == 200.0
    )
    assert after["forward_threaded_test_ms_max"] == 0.25
    assert after["prefill_graph_served_batches"] == (
        before["prefill_graph_served_batches"] + 1
    )
    assert degraded >= 1

    stats = initial_engine_stats(
        _telemetry_config(),
        {"serving_tier": "host_staged", "tier_fallback": False},
    )
    assert stats["consumer_contract"]["kind"] == "projection_only"
    assert stats["native_demand_sm_bytes"] == 0
    assert stats["serving_tier"] == "host_staged"
    counter_fields = set(stats["cumulative_counter_fields"])
    assert "host_device_bulk_batches" in counter_fields
    assert "forward_lifecycle_aborts" in counter_fields
    assert "host_mover_profiled_sm_gpu_ms" in counter_fields
    assert "verified_operator_modules" not in counter_fields
    assert "layer_service_conservative_ns" not in counter_fields
    try:
        initial_engine_stats(_telemetry_config(), {"engine": "collision"})
    except RuntimeError as error:
        assert "collides" in str(error)
    else:
        raise AssertionError("tier telemetry overwrote engine identity")

    with tempfile.TemporaryDirectory(prefix="nta-stats-publisher-") as directory:
        root = Path(directory)
        output = root / "stats.json"
        publisher = StatsPublisher(output)
        publisher.publish({"sequence": 1})
        publisher.publish({"sequence": 2})
        publisher.close()
        assert json.loads(output.read_text(encoding="utf-8")) == {"sequence": 2}
        publisher.close()
        try:
            publisher.publish({"sequence": 3})
        except RuntimeError:
            pass
        else:
            raise AssertionError("closed statistics publisher accepted a report")

        blocked_parent = root / "blocked"
        blocked_parent.write_text("not a directory", encoding="utf-8")
        failed = StatsPublisher(blocked_parent / "stats.json")
        try:
            failed.publish({"sequence": 1}, wait=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError("statistics write failure was not reported")
        failed.close()
    print("stats_publisher=pass")


if __name__ == "__main__":
    main()
