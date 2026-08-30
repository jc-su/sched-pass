#!/usr/bin/env python3
"""Validate bounded, retry-safe SGLang layer-service calibration."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from nta_runtime.engines.sglang_calibration import (  # noqa: E402
    SglangConsumerPolicyCalibration,
    SglangLayerServiceCalibration,
)


class Event:
    def __init__(self, timestamp_ms: float) -> None:
        self.timestamp_ms = timestamp_ms
        self.fail_elapsed = False
        self.recorded_stream = None

    def record(self, stream) -> None:
        self.recorded_stream = stream

    def query(self) -> bool:
        return True

    def elapsed_time(self, finish: "Event") -> float:
        if self.fail_elapsed:
            raise RuntimeError("elapsed-time failure")
        return finish.timestamp_ms - self.timestamp_ms


def main() -> None:
    assert SglangLayerServiceCalibration.shape_key(
        SimpleNamespace(reqs=(object(), object()), extend_num_tokens=7)
    ) == ("extend", 7, 2)
    assert SglangLayerServiceCalibration.shape_key(
        SimpleNamespace(batch_size=0, rids=(1, 2), extend_num_tokens=7)
    ) == ("extend", 7, 2)
    assert (
        SglangLayerServiceCalibration.shape_key(
            SimpleNamespace(batch_size=2, extend_num_tokens=0)
        )
        is None
    )

    stats = {
        "layer_service_profiled_intervals": 0,
        "layer_service_calibrated_shapes": 0,
    }
    calibration = SglangLayerServiceCalibration(
        enabled=True,
        minimum_samples=2,
        maximum_samples=4,
        model_start_layer=10,
        model_layer_count=4,
        stats=stats,
    )
    batch = SimpleNamespace(
        pending_host_load=object(),
        bindings=(object(), object()),
        layer_service_key=None,
        layer_arrival_event=None,
        layer_arrival_local_layer=-1,
    )
    events = [Event(0.0), Event(0.1), Event(0.3)]
    events[1].fail_elapsed = True
    with (
        patch(
            "nta_runtime.engines.sglang_calibration.torch.cuda.Event",
            side_effect=events,
        ),
        patch(
            "nta_runtime.engines.sglang_calibration.torch.cuda.current_stream",
            return_value="compute-stream",
        ),
    ):
        for layer in range(10, 13):
            calibration.record(
                batch=batch,
                phase="extend",
                query=SimpleNamespace(shape=(32,)),
                global_layer=layer,
            )
    assert calibration.pending_count == 2
    assert all(event.recorded_stream == "compute-stream" for event in events)

    # The first interval commits before the second raises. A retry must process
    # only the retained suffix, never count the committed sample twice.
    try:
        calibration.collect()
    except RuntimeError as error:
        assert "elapsed-time failure" in str(error)
    else:
        raise AssertionError("layer calibration swallowed an event failure")
    assert stats["layer_service_profiled_intervals"] == 1
    assert calibration.pending_count == 1
    events[1].fail_elapsed = False
    calibration.collect()
    assert stats["layer_service_profiled_intervals"] == 2
    assert stats["layer_service_calibrated_shapes"] == 1
    assert calibration.pending_count == 0
    curve = calibration.curve(("extend", 32, 2), calibrated_only=True)
    assert curve is not None and curve.samples_ns == (100_000, 200_000)
    assert calibration.report() == [
        {
            "phase": "extend",
            "query_rows": 32,
            "batch_size": 2,
            "samples": 2,
            "conservative_interval_ns": 100_000,
        }
    ]

    disabled = SglangLayerServiceCalibration(
        enabled=False,
        minimum_samples=1,
        maximum_samples=1,
        model_start_layer=0,
        model_layer_count=1,
        stats={
            "layer_service_profiled_intervals": 0,
            "layer_service_calibrated_shapes": 0,
        },
    )
    disabled.record(
        batch=batch,
        phase="extend",
        query=SimpleNamespace(shape=(32,)),
        global_layer=0,
    )
    assert disabled.pending_count == 0 and disabled.report() == []

    policy_stats = {
        "consumer_policy_profiled_leases": 0,
        "consumer_policy_probe_leases": 0,
        "consumer_policy_probe_misses": 0,
        "consumer_policy_rejected_shapes": 0,
        "consumer_policy_planned_layers": 0,
        "consumer_policy_arrival_samples": 0,
        "consumer_policy_stock_samples": 0,
        "consumer_policy_partial_samples": 0,
    }
    policy = SglangConsumerPolicyCalibration(
        enabled=True,
        model_start_layer=10,
        model_layer_count=2,
        minimum_samples=2,
        maximum_samples=4,
        stats=policy_stats,
    )
    pending = SimpleNamespace(
        device_indices=SimpleNamespace(numel=lambda: 8),
        layer_bytes=(1024, 1024),
        arrival_profile_key=None,
        arrival_profiling=False,
        arrival_profile_active=False,
        consumer_policy_probe=False,
        partial_profile_recorded=False,
        planned_progressive_layers=frozenset(),
    )
    key = policy.bind_lease(
        pending,
        layer_service_key=("extend", 32, 2),
        mover_kind="sm",
        layers_per_submission=2,
        sm_waves_per_layer=1,
        minimum_gain=1.03,
    )
    assert key is not None and pending.arrival_profiling
    assert not pending.arrival_profile_active
    assert not pending.consumer_policy_probe
    assert not pending.planned_progressive_layers

    # Two ordinary stock forwards establish a conservative signed margin for
    # every layer. Layer zero is always at least 0.5 ms late; layer one is
    # already ready. Probe execution is deliberately a later phase.
    ready_events = (
        (Event(1.5), Event(1.0)),
        (Event(1.6), Event(1.5)),
    )
    arrival_events = [Event(1.0), Event(2.0), Event(1.0), Event(2.0)]
    with (
        patch(
            "nta_runtime.engines.sglang_calibration.torch.cuda.Event",
            side_effect=arrival_events,
        ),
        patch(
            "nta_runtime.engines.sglang_calibration.torch.cuda.current_stream",
            return_value="compute-stream",
        ),
    ):
        pending.arrival_profile_active = True
        for forward, layer_ready in enumerate(ready_events):
            acquisition = SimpleNamespace(
                layer=lambda layer, events=layer_ready: SimpleNamespace(
                    partial_publication=SimpleNamespace(
                        profile_ready_event=events[layer]
                    )
                )
            )
            policy_batch = SimpleNamespace(
                pending_host_load=pending,
                bindings=(object(), object()),
                acquisition=acquisition,
            )
            for layer in range(2):
                policy.record_arrival(
                    batch=policy_batch,
                    phase="extend",
                    query=SimpleNamespace(shape=(32,)),
                    global_layer=10 + layer,
                )
                policy.record_stock_profile(
                    pending=pending,
                    global_layer=10 + layer,
                    start=Event(float(layer)),
                    finish=Event(float(layer) + 0.1),
                )
    policy.collect()
    assert policy.pending_count == 0
    assert policy_stats["consumer_policy_arrival_samples"] == 4
    assert policy_stats["consumer_policy_stock_samples"] == 4
    assert policy_stats["consumer_policy_partial_samples"] == 0
    assert not policy.profitable_layers(key, minimum_gain=1.03)

    # Persistent stock lateness unlocks a finite partial-consumer probe. Two
    # independent leases establish a worst-case 0.22 ms operator cost.
    for forward, cost in enumerate((0.2, 0.22)):
        pending.partial_profile_recorded = False
        rebound = policy.bind_lease(
            pending,
            layer_service_key=("extend", 32, 2),
            mover_kind="sm",
            layers_per_submission=2,
            sm_waves_per_layer=1,
            minimum_gain=1.03,
        )
        assert rebound == key
        assert not pending.arrival_profiling
        assert pending.consumer_policy_probe
        policy.record_partial_profile(
            pending=pending,
            start=Event(float(forward)),
            finish=Event(float(forward) + cost),
        )
    policy.collect()
    assert policy_stats["consumer_policy_partial_samples"] == 2
    assert policy.profitable_layers(key, minimum_gain=1.03) == frozenset({0})

    pending.partial_profile_recorded = False
    rebound = policy.bind_lease(
        pending,
        layer_service_key=("extend", 32, 2),
        mover_kind="sm",
        layers_per_submission=2,
        sm_waves_per_layer=1,
        minimum_gain=1.03,
    )
    assert rebound == key
    assert not pending.arrival_profiling
    assert not pending.consumer_policy_probe
    assert pending.planned_progressive_layers == frozenset({0})
    assert policy.report() == {
        "minimum_samples": 2,
        "maximum_samples": 4,
        "maximum_probe_misses": 2,
        "arrival_shapes": 1,
        "calibrated_arrival_layers": 2,
        "calibrated_stock_layers": 2,
        "partial_shapes": 1,
        "calibrated_partial_shapes": 1,
        "probe_rejected_shapes": 0,
        "shapes": [
            {
                "phase": "extend",
                "query_rows_bucket": 5,
                "batch_size_bucket": 1,
                "transfer_rows_bucket": 3,
                "transfer_bytes_bucket": 11,
                "mover_kind": "sm",
                "layers_per_submission": 2,
                "sm_waves_per_layer": 1,
                "calibrated_arrival_layers": 2,
                "calibrated_stock_layers": 2,
                "maximum_conservative_lateness_ns": 500000,
                "minimum_stock_service_ns": 100000,
                "maximum_partial_service_ns": 220000,
                "profitable_layers": 1,
                "probe_misses": 0,
            }
        ],
    }

    rejected_stats = {
        "consumer_policy_profiled_leases": 0,
        "consumer_policy_probe_leases": 0,
        "consumer_policy_probe_misses": 0,
        "consumer_policy_rejected_shapes": 0,
        "consumer_policy_planned_layers": 0,
        "consumer_policy_arrival_samples": 0,
        "consumer_policy_stock_samples": 0,
        "consumer_policy_partial_samples": 0,
    }
    rejected = SglangConsumerPolicyCalibration(
        enabled=True,
        model_start_layer=0,
        model_layer_count=1,
        maximum_probe_misses=2,
        stats=rejected_stats,
    )
    missed = SimpleNamespace(
        arrival_profile_key=key,
        partial_profile_recorded=False,
    )
    rejected.retire_lease(missed, probe_executed=True)
    rejected.retire_lease(missed, probe_executed=True)
    assert rejected_stats["consumer_policy_probe_misses"] == 2
    assert rejected_stats["consumer_policy_rejected_shapes"] == 1


if __name__ == "__main__":
    main()
