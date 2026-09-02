#!/usr/bin/env python3
"""Validate bounded, retry-safe SGLang layer-service calibration."""

from __future__ import annotations

import copy
from collections import defaultdict
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
    layer_state = calibration.export_state()
    restored_layer_stats = {
        "layer_service_profiled_intervals": 0,
        "layer_service_calibrated_shapes": 0,
    }
    restored_layer = SglangLayerServiceCalibration(
        enabled=True,
        minimum_samples=2,
        maximum_samples=4,
        model_start_layer=10,
        model_layer_count=4,
        stats=restored_layer_stats,
    )
    assert restored_layer.import_state(layer_state) == 2
    assert restored_layer.export_state() == layer_state
    assert restored_layer.curve(("extend", 32, 2), calibrated_only=True) == curve

    frozen_layer = SglangLayerServiceCalibration(
        enabled=True,
        frozen=True,
        minimum_samples=2,
        maximum_samples=4,
        model_start_layer=10,
        model_layer_count=4,
        stats={
            "layer_service_profiled_intervals": 0,
            "layer_service_calibrated_shapes": 0,
        },
    )
    assert frozen_layer.import_state(layer_state) == 2
    frozen_layer.record(
        batch=batch,
        phase="extend",
        query=SimpleNamespace(shape=(64,)),
        global_layer=10,
    )
    assert frozen_layer.pending_count == 0
    assert frozen_layer.export_state() == layer_state

    incompatible_layer = copy.deepcopy(layer_state)
    incompatible_layer["model_layer_count"] = 5
    try:
        SglangLayerServiceCalibration(
            enabled=True,
            minimum_samples=2,
            maximum_samples=4,
            model_start_layer=10,
            model_layer_count=4,
            stats={
                "layer_service_profiled_intervals": 0,
                "layer_service_calibrated_shapes": 0,
            },
        ).import_state(incompatible_layer)
    except ValueError as error:
        assert "incompatible" in str(error)
    else:
        raise AssertionError("layer-service calibration accepted stale geometry")

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
        "consumer_policy_partial_setup_samples": 0,
        "consumer_policy_partial_reuse_samples": 0,
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
        prefetched_layers={},
    )
    key = policy.bind_lease(
        pending,
        layer_service_key=("extend", 32, 2),
        producer_kind="sm",
        layers_per_submission=2,
        sm_waves_per_layer=1,
        minimum_gain=1.03,
    )
    assert key is not None and pending.arrival_profiling
    assert not policy.shape_closed(key)
    assert not pending.arrival_profile_active
    assert not pending.consumer_policy_probe
    assert not pending.planned_progressive_layers

    # A first unseen-shape lease can have a finite admission frontier before
    # its batch-derived consumer key exists.  It is skipped as one complete
    # profiling unit; a missing marker is never reconstructed after the fact.
    skip_stats = defaultdict(int)
    skip_policy = SglangConsumerPolicyCalibration(
        enabled=True,
        model_start_layer=10,
        model_layer_count=2,
        minimum_samples=2,
        maximum_samples=4,
        stats=skip_stats,
    )
    prepublished = SimpleNamespace(
        device_indices=SimpleNamespace(numel=lambda: 8),
        layer_bytes=(1024, 1024),
        arrival_profile_key=None,
        arrival_profiling=False,
        arrival_profile_active=False,
        consumer_policy_probe=False,
        partial_profile_recorded=False,
        planned_progressive_layers=frozenset(),
        prefetched_layers={
            0: SimpleNamespace(profile_ready_event=None),
        },
    )
    prepublished_key = skip_policy.bind_lease(
        prepublished,
        layer_service_key=("extend", 64, 2),
        producer_kind="sm",
        layers_per_submission=2,
        sm_waves_per_layer=1,
        minimum_gain=1.03,
    )
    assert prepublished_key is not None
    assert not prepublished.arrival_profiling
    assert skip_stats["consumer_policy_prepublication_skipped_leases"] == 1

    # A scheduler-bound layer may reach attention before the shared producer
    # has published its fence. Record the arrival before any wait, then pair it
    # with the real producer marker; the resulting signed lateness is evidence,
    # not an error or a reason to manufacture a zero-lateness timestamp.
    deferred_stats = defaultdict(int)
    deferred_policy = SglangConsumerPolicyCalibration(
        enabled=True,
        model_start_layer=10,
        model_layer_count=1,
        minimum_samples=1,
        maximum_samples=2,
        stats=deferred_stats,
    )
    deferred_pending = SimpleNamespace(
        device_indices=SimpleNamespace(numel=lambda: 8),
        layer_bytes=(1024,),
        arrival_profile_key=None,
        arrival_profiling=False,
        arrival_profile_active=False,
        consumer_policy_probe=False,
        partial_profile_recorded=False,
        planned_progressive_layers=frozenset(),
        prefetched_layers={},
    )
    deferred_key = deferred_policy.bind_lease(
        deferred_pending,
        layer_service_key=("extend", 32, 2),
        producer_kind="sm",
        layers_per_submission=1,
        sm_waves_per_layer=1,
        minimum_gain=1.03,
    )
    assert deferred_key is not None
    deferred_pending.arrival_profile_active = True
    deferred_acquisition = SimpleNamespace(layer=lambda _layer: None)
    deferred_batch = SimpleNamespace(
        pending_host_load=deferred_pending,
        bindings=(object(), object()),
        acquisition=deferred_acquisition,
    )
    with (
        patch(
            "nta_runtime.engines.sglang_calibration.torch.cuda.Event",
            return_value=Event(1.0),
        ),
        patch(
            "nta_runtime.engines.sglang_calibration.torch.cuda.current_stream",
            return_value="compute-stream",
        ),
    ):
        deferred_policy.record_arrival(
            batch=deferred_batch,
            phase="extend",
            query=SimpleNamespace(shape=(32,)),
            global_layer=10,
        )
    assert deferred_policy.pending_count == 1
    assert deferred_stats["consumer_policy_deferred_arrival_markers"] == 1
    ready = Event(1.5)
    deferred_pending.prefetched_layers[0] = SimpleNamespace(profile_ready_event=ready)
    deferred_acquisition.layer = lambda _layer: SimpleNamespace(
        profile_ready_event=ready
    )
    deferred_policy.bind_arrival_ready(batch=deferred_batch, global_layer=10)
    assert deferred_stats["consumer_policy_deferred_arrivals_bound"] == 1
    deferred_policy.collect()
    assert deferred_policy.pending_count == 0
    deferred_report = deferred_policy.report()
    assert deferred_report["shapes"][0]["maximum_conservative_lateness_ns"] == 500000

    nvme_pending = SimpleNamespace(
        device_indices=SimpleNamespace(numel=lambda: 1),
        layer_bytes=(1,),
        arrival_profile_key=None,
        arrival_profiling=False,
        arrival_profile_active=False,
        consumer_policy_probe=False,
        partial_profile_recorded=False,
        planned_progressive_layers=frozenset(),
    )
    nvme_policy = SglangConsumerPolicyCalibration(
        enabled=True,
        model_start_layer=10,
        model_layer_count=2,
        minimum_samples=2,
        maximum_samples=4,
        stats=defaultdict(int),
    )
    nvme_key = nvme_policy.bind_lease(
        nvme_pending,
        layer_service_key=("extend", 32, 2),
        producer_kind="nvme_direct",
        layers_per_submission=2,
        sm_waves_per_layer=4,
        minimum_gain=1.03,
        transfer_rows=16,
        transfer_bytes=131072,
    )
    assert nvme_key is not None
    assert nvme_key.producer_kind == "nvme_direct"
    assert nvme_key.sm_waves_per_layer == 4

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
                    profile_ready_event=events[layer]
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
    assert not policy.shape_closed(key)

    # Persistent stock lateness unlocks a finite partial-consumer probe. Two
    # independent leases separate the first-layer partition cost from the
    # forward-local reusable critical path.
    for forward, (cold_cost, reuse_cost) in enumerate(((0.2, 0.15), (0.22, 0.16))):
        pending.partial_profile_recorded = False
        rebound = policy.bind_lease(
            pending,
            layer_service_key=("extend", 32, 2),
            producer_kind="sm",
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
            dispatch_ready=Event(float(forward) + 0.01),
            finish=Event(float(forward) + cold_cost),
            partition_prepared=True,
        )
        policy.record_partial_profile(
            pending=pending,
            start=Event(float(forward) + 1.0),
            dispatch_ready=Event(float(forward) + 1.01),
            finish=Event(float(forward) + 1.0 + reuse_cost),
            partition_prepared=False,
        )
    policy.collect()
    assert policy_stats["consumer_policy_partial_samples"] == 4
    assert policy_stats["consumer_policy_partial_setup_samples"] == 2
    assert policy_stats["consumer_policy_partial_reuse_samples"] == 2
    assert policy.profitable_layers(key, minimum_gain=1.03) == frozenset({0})

    pending.partial_profile_recorded = False
    rebound = policy.bind_lease(
        pending,
        layer_service_key=("extend", 32, 2),
        producer_kind="sm",
        layers_per_submission=2,
        sm_waves_per_layer=1,
        minimum_gain=1.03,
    )
    assert rebound == key
    assert not pending.arrival_profiling
    assert not pending.consumer_policy_probe
    assert pending.planned_progressive_layers == frozenset({0})
    assert policy.shape_closed(key)
    policy_report = policy.report()
    assert policy_report == {
        "mode": "learning",
        "minimum_samples": 2,
        "maximum_samples": 4,
        "maximum_probe_misses": 2,
        "arrival_shapes": 1,
        "calibrated_arrival_layers": 2,
        "calibrated_stock_layers": 2,
        "partial_shapes": 1,
        "calibrated_partial_shapes": 1,
        "calibrated_partial_reuse_shapes": 1,
        "probe_rejected_shapes": 0,
        "closed_shapes": 1,
        "open_shapes": 0,
        "last_shape_closed": True,
        "last_shape_calibrated": True,
        "last_shape_decision": "profile",
        "shapes": [
            {
                "phase": "extend",
                "query_rows_bucket": 5,
                "batch_size_bucket": 1,
                "transfer_rows_bucket": 3,
                "transfer_bytes_bucket": 11,
                "producer_kind": "sm",
                "layers_per_submission": 2,
                "sm_waves_per_layer": 1,
                "calibrated_arrival_layers": 2,
                "calibrated_stock_layers": 2,
                "maximum_conservative_lateness_ns": 500000,
                "minimum_stock_service_ns": 100000,
                "maximum_partial_cold_critical_path_ns": 220000,
                "maximum_partial_reuse_critical_path_ns": 160000,
                "maximum_partial_cold_dispatch_ns": 10000,
                "maximum_partial_reuse_dispatch_ns": 10000,
                "maximum_partial_cold_device_ns": 210000,
                "maximum_partial_reuse_device_ns": 150000,
                "estimated_partial_fixed_setup_ns": 60000,
                "profitable_layers": 1,
                "probe_attempts": 2,
                "probe_misses": 0,
                "closed": True,
            }
        ],
    }, policy_report
    policy_state = policy.export_state()
    restored_policy_stats = {name: 0 for name in policy_stats}
    restored_policy = SglangConsumerPolicyCalibration(
        enabled=True,
        model_start_layer=10,
        model_layer_count=2,
        minimum_samples=2,
        maximum_samples=4,
        stats=restored_policy_stats,
    )
    assert restored_policy.import_state(policy_state) == 20
    assert restored_policy.export_state() == policy_state
    assert restored_policy.shape_closed(key)
    pending.partial_profile_recorded = False
    assert (
        restored_policy.bind_lease(
            pending,
            layer_service_key=("extend", 32, 2),
            producer_kind="sm",
            layers_per_submission=2,
            sm_waves_per_layer=1,
            minimum_gain=1.03,
        )
        == key
    )
    assert not pending.arrival_profiling
    assert not pending.consumer_policy_probe
    assert pending.planned_progressive_layers == frozenset({0})

    frozen_stats = {name: 0 for name in policy_stats}
    frozen_policy = SglangConsumerPolicyCalibration(
        enabled=True,
        frozen=True,
        model_start_layer=10,
        model_layer_count=2,
        minimum_samples=2,
        maximum_samples=4,
        stats=frozen_stats,
    )
    assert frozen_policy.import_state(policy_state) == 20
    frozen_state = frozen_policy.export_state()
    assert (
        frozen_policy.bind_lease(
            pending,
            layer_service_key=("extend", 32, 2),
            producer_kind="sm",
            layers_per_submission=2,
            sm_waves_per_layer=1,
            minimum_gain=1.03,
        )
        == key
    )
    assert not pending.arrival_profiling
    assert not pending.consumer_policy_probe
    assert pending.planned_progressive_layers == frozenset({0})
    assert frozen_stats["consumer_policy_frozen_profile_leases"] == 1

    unknown = SimpleNamespace(
        device_indices=SimpleNamespace(numel=lambda: 128),
        layer_bytes=(1 << 20, 1 << 20),
        arrival_profile_key=None,
        arrival_profiling=False,
        arrival_profile_active=False,
        consumer_policy_probe=False,
        partial_profile_recorded=False,
        planned_progressive_layers=frozenset(),
    )
    unknown_key = frozen_policy.bind_lease(
        unknown,
        layer_service_key=("extend", 4096, 8),
        producer_kind="copy_engine",
        layers_per_submission=4,
        sm_waves_per_layer=1,
        minimum_gain=1.03,
    )
    assert unknown_key is not None and unknown_key != key
    assert not unknown.arrival_profiling
    assert not unknown.consumer_policy_probe
    assert not unknown.planned_progressive_layers
    assert frozen_stats["consumer_policy_frozen_conservative_leases"] == 1
    assert frozen_policy.export_state() == frozen_state
    frozen_report = frozen_policy.report()
    assert frozen_report["mode"] == "frozen"
    assert frozen_report["last_shape_closed"] is True
    assert frozen_report["last_shape_calibrated"] is False
    assert frozen_report["last_shape_decision"] == "conservative_stock"

    inconsistent_partial = copy.deepcopy(policy_state)
    inconsistent_partial["partial_device_curves"] = []
    try:
        SglangConsumerPolicyCalibration(
            enabled=True,
            model_start_layer=10,
            model_layer_count=2,
            minimum_samples=2,
            maximum_samples=4,
            stats={name: 0 for name in policy_stats},
        ).import_state(inconsistent_partial)
    except ValueError as error:
        assert "inconsistent" in str(error)
    else:
        raise AssertionError("consumer calibration accepted a torn partial sample")

    rejected_stats = {
        "consumer_policy_profiled_leases": 0,
        "consumer_policy_probe_leases": 0,
        "consumer_policy_probe_misses": 0,
        "consumer_policy_rejected_shapes": 0,
        "consumer_policy_planned_layers": 0,
        "consumer_policy_arrival_samples": 0,
        "consumer_policy_stock_samples": 0,
        "consumer_policy_partial_samples": 0,
        "consumer_policy_partial_setup_samples": 0,
        "consumer_policy_partial_reuse_samples": 0,
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
