#!/usr/bin/env python3
"""Validate explicit manifest arrival-rate transformations."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "serving"))

from SglangHiCacheLoad import (  # noqa: E402
    LoadedWorkload,
    _configure_workload_arrivals,
)


def workload(rate: float | None = 10.0) -> LoadedWorkload:
    return LoadedWorkload(
        resident_request_ids=("resident",),
        external_request_ids=("external-0", "external-1"),
        resident_inputs=((1,),),
        external_inputs=((2,), (3,)),
        resident_arrival_offsets=(0.0,),
        external_arrival_offsets=(0.1, 0.2),
        resident_output_tokens=(2,),
        external_output_tokens=(2, 2),
        metadata={
            "arrival": {
                "mode": "calibrated_open_loop",
                "target_rate_per_second": rate,
            },
            "request_arrival_offsets": {
                "resident": 0.0,
                "external-0": 0.1,
                "external-1": 0.2,
            },
        },
    )


def expect_failure(function, text: str) -> None:
    try:
        function()
    except RuntimeError as error:
        assert text in str(error)
    else:
        raise AssertionError("invalid workload arrival contract was accepted")


def main() -> None:
    scaled = _configure_workload_arrivals(
        workload(), target_rate=20.0, scale_to_target=True
    )
    assert scaled.resident_arrival_offsets == (0.0,)
    assert scaled.external_arrival_offsets == (0.05, 0.1)
    assert scaled.metadata["request_arrival_offsets"] == {
        "resident": 0.0,
        "external-0": 0.05,
        "external-1": 0.1,
    }
    assert scaled.metadata["runtime_arrival"] == {
        "method": "uniform_manifest_time_dilation",
        "source_mode": "calibrated_open_loop",
        "source_target_rate_per_second": 10.0,
        "target_rate_per_second": 20.0,
        "uniform_time_scale": 0.5,
        "request_order_preserved": True,
    }

    exact = _configure_workload_arrivals(
        workload(), target_rate=10.0, scale_to_target=False
    )
    assert exact.external_arrival_offsets == (0.1, 0.2)
    assert exact.metadata["runtime_arrival"]["method"] == "manifest_exact"

    expect_failure(
        lambda: _configure_workload_arrivals(
            workload(), target_rate=20.0, scale_to_target=False
        ),
        "disagrees",
    )
    expect_failure(
        lambda: _configure_workload_arrivals(
            workload(None), target_rate=20.0, scale_to_target=True
        ),
        "positive target_rate_per_second",
    )
    print("workload_arrival_scaling=pass")


if __name__ == "__main__":
    main()
