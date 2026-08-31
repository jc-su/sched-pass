#!/usr/bin/env python3
"""Pure contract tests for repeated serving-trial qualification gates."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "serving"))

import CompareSglangHiCacheLoadTrials as gates  # noqa: E402


def _record(request_id: str, *, kind: str = "external") -> dict[str, object]:
    return {"kind": kind, "request_id": request_id}


def _report(
    request_ids: list[str],
    *,
    nta_request_ids: list[str] | None = None,
    native_launches: int = 4,
    stock_launches: int = 0,
) -> dict[str, object]:
    external_launches = native_launches + stock_launches
    return {
        "stock": {"records": [_record(value) for value in request_ids]},
        "nta": {
            "records": [
                _record(value)
                for value in (
                    request_ids if nta_request_ids is None else nta_request_ids
                )
            ]
        },
        "mechanism_activation": {
            "external_launches": external_launches,
            "transformed_external_launches": native_launches,
            "stock_prefetched_external_launches": stock_launches,
            "native_work_unit_active": native_launches > 0,
            "external_attention_transformed": (
                external_launches > 0 and stock_launches == 0
            ),
        },
    }


def _expect_runtime_error(function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


def _expect_argument_error(function, value: str) -> None:
    try:
        function(value)
    except argparse.ArgumentTypeError:
        return
    raise AssertionError("expected argparse.ArgumentTypeError")


def main() -> None:
    gates._validate_worktree_status(0, "")
    _expect_runtime_error(gates._validate_worktree_status, 0, " M tracked.py\n")
    _expect_runtime_error(gates._validate_worktree_status, 128, "", "git failed")
    with mock.patch.object(
        gates.subprocess,
        "run",
        side_effect=AssertionError("diagnostic preflight invoked git"),
    ):
        gates._require_clean_worktree(diagnostic=True)

    assert gates._positive_int("100") == 100
    assert gates._nonnegative_int("0") == 0
    _expect_argument_error(gates._positive_int, "0")
    _expect_argument_error(gates._nonnegative_int, "-1")

    saved_argv = sys.argv
    sys.argv = ["CompareSglangHiCacheLoadTrials.py", "--diagnostic", "--", "child"]
    try:
        parsed = gates.parse_args()
    finally:
        sys.argv = saved_argv
    assert parsed.diagnostic is True
    assert parsed.min_external_observations == 100
    assert parsed.min_distinct_external_requests == 0

    reports = [_report(["a", "b"]), _report(["a", "c"])]
    evidence = gates._external_request_evidence(
        reports,
        min_observations=4,
        min_distinct_requests=3,
    )
    assert evidence["passes"] is True
    assert evidence["per_arm"] == {
        "stock": {
            "external_observations": 4,
            "distinct_external_request_ids": 3,
            "observations_threshold_met": True,
            "distinct_requests_threshold_met": True,
        },
        "nta": {
            "external_observations": 4,
            "distinct_external_request_ids": 3,
            "observations_threshold_met": True,
            "distinct_requests_threshold_met": True,
        },
    }
    gates._validate_external_request_evidence(evidence)

    insufficient = gates._external_request_evidence(
        reports,
        min_observations=5,
        min_distinct_requests=0,
    )
    assert insufficient["passes"] is False
    assert all(
        arm["observations_threshold_met"] is False
        for arm in insufficient["per_arm"].values()
    )

    insufficient_distinct = gates._external_request_evidence(
        reports,
        min_observations=4,
        min_distinct_requests=4,
    )
    assert insufficient_distinct["passes"] is False
    assert all(
        arm["distinct_requests_threshold_met"] is False
        for arm in insufficient_distinct["per_arm"].values()
    )

    _expect_runtime_error(
        gates._external_request_evidence,
        [_report(["a", "b"], nta_request_ids=["a", "c"])],
        min_observations=2,
        min_distinct_requests=0,
    )

    missing_id = _report(["a"])
    missing_id["stock"]["records"][0].pop("request_id")
    _expect_runtime_error(
        gates._external_request_evidence,
        [missing_id],
        min_observations=1,
        min_distinct_requests=0,
    )

    stale = copy.deepcopy(evidence)
    stale["passes"] = False
    _expect_runtime_error(gates._validate_external_request_evidence, stale)

    native = _report(["a"])
    assert gates._native_consumer_evidence(native)[
        "all_external_launches_native"
    ] is True
    gates._require_native_consumer(native, trial=0)

    stock_consumer = _report(["a"], native_launches=3, stock_launches=1)
    assert gates._native_consumer_evidence(stock_consumer)[
        "all_external_launches_native"
    ] is False
    _expect_runtime_error(gates._require_native_consumer, stock_consumer, trial=1)

    ratio_reports = []
    for value in (1.1, 1.2, 1.3):
        ratio_report = {field: 1.0 for field in gates.RATIO_FIELDS}
        ratio_report["resident_output_throughput_ratio"] = value
        ratio_report["external_output_throughput_ratio"] = value + 0.1
        ratio_reports.append(ratio_report)
    ratios = gates._aggregate(ratio_reports, seed=17)
    assert ratios["resident_output_throughput_ratio"]["paired_values"] == [
        1.1,
        1.2,
        1.3,
    ]
    assert [
        round(value, 10)
        for value in ratios["external_output_throughput_ratio"]["paired_values"]
    ] == [1.2, 1.3, 1.4]
    assert ratios["preregistered_goodput_ratio"]["paired_values"] == [
        1.0,
        1.0,
        1.0,
    ]
    assert ratios["preregistered_joint_goodput_ratio"]["paired_values"] == [
        1.0,
        1.0,
        1.0,
    ]

    zero_joint_reports = copy.deepcopy(ratio_reports)
    zero_joint_reports[0]["preregistered_joint_goodput_ratio"] = 0.0
    zero_joint = gates._aggregate(zero_joint_reports, seed=19)[
        "preregistered_joint_goodput_ratio"
    ]
    assert zero_joint["zero_goodput_trials"] == 1
    assert zero_joint["paired_values"] == [0.0, 1.0, 1.0]

    passing_bar = gates._registered_goodput_bar(
        {
            "geometric_mean": 1.6,
            "bootstrap_95_percent_ci": [1.1, 1.8],
        },
        token_level_eligible=True,
    )
    assert passing_bar["passes"] is True
    assert passing_bar["all_requests_have_token_level_itl"] is True
    ineligible_bar = gates._registered_goodput_bar(
        {
            "geometric_mean": 1.6,
            "bootstrap_95_percent_ci": [1.1, 1.8],
        },
        token_level_eligible=False,
    )
    assert ineligible_bar["passes"] is False
    zero_joint_bar = gates._registered_goodput_bar(
        zero_joint,
        token_level_eligible=True,
    )
    assert zero_joint_bar["passes"] is False

    assert gates._ratio_bar(
        {"geometric_mean": 1.05}, threshold=1.05, at_most=True
    )["passes"] is True
    assert gates._ratio_bar(
        {"geometric_mean": 1.051}, threshold=1.05, at_most=True
    )["passes"] is False
    assert gates._ratio_bar(
        {"geometric_mean": 0.95}, threshold=0.95, at_most=False
    )["passes"] is True
    assert gates._ratio_bar(
        {"geometric_mean": 0.949}, threshold=0.95, at_most=False
    )["passes"] is False
    assert gates._ratio_bar({}, threshold=0.95, at_most=False)["passes"] is False

    tpot_equivalent = gates._ratio_bar(
        {
            "geometric_mean": 1.01,
            "bootstrap_95_percent_ci": [0.98, 1.05],
        },
        threshold=1.05,
        at_most=True,
        bootstrap_bound="upper",
    )
    assert tpot_equivalent["passes"] is True
    assert tpot_equivalent["bootstrap_95_percent_ci_upper"] == 1.05
    assert gates._ratio_bar(
        {
            "geometric_mean": 1.01,
            "bootstrap_95_percent_ci": [0.98, 1.051],
        },
        threshold=1.05,
        at_most=True,
        bootstrap_bound="upper",
    )["passes"] is False

    throughput_equivalent = gates._ratio_bar(
        {
            "geometric_mean": 1.02,
            "bootstrap_95_percent_ci": [0.95, 1.08],
        },
        threshold=0.95,
        at_most=False,
        bootstrap_bound="lower",
    )
    assert throughput_equivalent["passes"] is True
    assert throughput_equivalent["bootstrap_95_percent_ci_lower"] == 0.95
    assert gates._ratio_bar(
        {
            "geometric_mean": 1.02,
            "bootstrap_95_percent_ci": [0.949, 1.08],
        },
        threshold=0.95,
        at_most=False,
        bootstrap_bound="lower",
    )["passes"] is False
    assert gates._ratio_bar(
        {"geometric_mean": 1.02},
        threshold=0.95,
        at_most=False,
        bootstrap_bound="lower",
    )["passes"] is False

    print("serving_trial_gates=passed")


if __name__ == "__main__":
    main()
