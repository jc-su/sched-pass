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


if __name__ == "__main__":
    main()
