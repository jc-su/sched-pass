#!/usr/bin/env python3
"""Validate a repeated paired-serving qualification result.

This contract distinguishes a completed diagnostic from formal qualification.
Formal evidence is qualified only when every registered correctness, mechanism,
and performance bar passes on one revision, machine, and consumed workload.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any, name: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"serving trial aggregate has no finite {name}",
    )
    return float(value)


def _validate_external_request_evidence(
    evidence: Any, *, formal: bool
) -> None:
    _require(
        isinstance(evidence, dict) and evidence.get("schema") == 1,
        "serving trial aggregate has no external-request evidence",
    )
    minimum = evidence.get("minimum_external_observations_per_arm")
    minimum_distinct = evidence.get(
        "minimum_distinct_external_request_ids_per_arm"
    )
    _require(
        isinstance(minimum, int)
        and not isinstance(minimum, bool)
        and minimum > 0
        and isinstance(minimum_distinct, int)
        and not isinstance(minimum_distinct, bool)
        and minimum_distinct >= 0,
        "serving trial external-request thresholds are invalid",
    )
    _require(
        not formal or minimum >= 100,
        "formal serving evidence requires at least 100 external observations",
    )
    per_arm = evidence.get("per_arm")
    _require(
        isinstance(per_arm, dict) and set(per_arm) == {"stock", "nta"},
        "serving trial external-request arm evidence is invalid",
    )
    arm_passes: list[bool] = []
    for name in ("stock", "nta"):
        arm = per_arm[name]
        _require(isinstance(arm, dict), f"external-request arm {name} is invalid")
        observations = arm.get("external_observations")
        distinct = arm.get("distinct_external_request_ids")
        _require(
            isinstance(observations, int)
            and not isinstance(observations, bool)
            and observations >= 0
            and isinstance(distinct, int)
            and not isinstance(distinct, bool)
            and 0 <= distinct <= observations,
            f"external-request arm {name} has invalid counts",
        )
        observations_met = observations >= minimum
        distinct_met = distinct >= minimum_distinct
        _require(
            arm.get("observations_threshold_met") is observations_met
            and arm.get("distinct_requests_threshold_met") is distinct_met,
            f"external-request arm {name} has a stale threshold verdict",
        )
        arm_passes.append(observations_met and distinct_met)
    counts_match = (
        per_arm["stock"]["external_observations"]
        == per_arm["nta"]["external_observations"]
    )
    _require(
        evidence.get("paired_observation_counts_match") is counts_match
        and isinstance(evidence.get("paired_distinct_request_ids_match"), bool),
        "external-request paired evidence is inconsistent",
    )
    expected = bool(
        counts_match
        and evidence["paired_distinct_request_ids_match"]
        and all(arm_passes)
    )
    _require(
        evidence.get("passes") is expected,
        "external-request aggregate verdict is inconsistent",
    )


def validate(report: dict[str, Any]) -> dict[str, Any]:
    _require(report.get("schema") == 2, "unsupported serving trial schema")
    _require(
        report.get("classification") == "sglang-hicache-load-qualification",
        "result is not an SGLang serving qualification aggregate",
    )
    mode = report.get("mode")
    _require(mode in {"formal", "diagnostic"}, "unknown serving trial mode")
    _validate_external_request_evidence(
        report.get("external_request_evidence"), formal=mode == "formal"
    )
    trial_count = report.get("trial_count")
    _require(
        isinstance(trial_count, int)
        and not isinstance(trial_count, bool)
        and trial_count >= 3,
        "serving trial aggregate has fewer than three paired trials",
    )
    artifacts = report.get("trial_artifacts")
    _require(
        isinstance(artifacts, list) and len(artifacts) == trial_count,
        "serving trial artifact count does not match trial count",
    )
    for index, artifact in enumerate(artifacts):
        _require(
            isinstance(artifact, dict), f"serving trial artifact {index} is invalid"
        )
        for field in ("path", "sha256", "revision", "machine_digest", "demand_digest"):
            value = artifact.get(field)
            _require(
                isinstance(value, str) and value,
                f"serving trial artifact {index} lacks {field}",
            )
        _require(
            len(artifact["sha256"]) == 64
            and all(character in "0123456789abcdef" for character in artifact["sha256"]),
            f"serving trial artifact {index} has an invalid sha256",
        )

    revisions = report.get("revisions")
    machine_digests = report.get("machine_digests")
    demand_digests = report.get("demand_digests")
    for values, name in (
        (revisions, "revision"),
        (machine_digests, "machine"),
        (demand_digests, "demand"),
    ):
        _require(
            isinstance(values, list)
            and values
            and values == sorted(set(values))
            and all(isinstance(value, str) and value for value in values),
            f"serving trial aggregate has invalid {name} identities",
        )
    _require(
        revisions == sorted({artifact["revision"] for artifact in artifacts}),
        "serving trial revision identities do not match artifacts",
    )
    _require(
        machine_digests
        == sorted({artifact["machine_digest"] for artifact in artifacts}),
        "serving trial machine identities do not match artifacts",
    )
    _require(
        demand_digests == sorted({artifact["demand_digest"] for artifact in artifacts}),
        "serving trial demand identities do not match artifacts",
    )

    bars = report.get("bars")
    _require(isinstance(bars, dict) and bars, "serving trial aggregate has no bars")
    required_bars = {
        "registered_goodput",
        "registered_joint_goodput",
        "resident_p99_itl",
        "resident_p95_tpot",
        "resident_output_throughput",
        "output_throughput",
        "outputs",
        "mechanism",
        "physical_bytes",
        "provenance",
    }
    _require(
        set(bars) == required_bars,
        "serving trial aggregate has an unexpected registered-bar set",
    )
    for name, bar in bars.items():
        _require(
            isinstance(bar, dict) and isinstance(bar.get("passes"), bool),
            f"serving trial bar {name} has no boolean verdict",
        )
    for key, label in (
        ("registered_goodput", "registered-goodput"),
        ("registered_joint_goodput", "registered joint-goodput"),
    ):
        registered = bars[key]
        threshold = _finite(registered.get("bar"), f"{label} bar")
        _require(
            isinstance(registered.get("all_requests_have_token_level_itl"), bool),
            f"{label} bar lacks token-level ITL eligibility",
        )
        _require(
            registered["all_requests_have_token_level_itl"]
            or registered.get("passes") is False,
            f"{label} bar passed with requests lacking token-level ITL",
        )
        if (
            registered.get("geometric_mean") is None
            or registered.get("ci_floor") is None
        ):
            _require(
                registered.get("passes") is False,
                f"{label} bar passed without a finite estimate",
            )
        else:
            geometric_mean = _finite(
                registered.get("geometric_mean"), f"{label} geometric mean"
            )
            ci_floor = _finite(registered.get("ci_floor"), f"{label} CI floor")
            expected = bool(
                registered["all_requests_have_token_level_itl"]
                and geometric_mean >= threshold
                and ci_floor > 1.0
            )
            _require(
                registered.get("passes") is expected,
                f"{label} verdict is inconsistent",
            )

    ratio_bars = (
        ("resident_p99_itl", 1.05, True, None),
        ("resident_p95_tpot", 1.05, True, "upper"),
        ("resident_output_throughput", 0.95, False, "lower"),
        ("output_throughput", 0.95, False, "lower"),
    )
    for name, expected_threshold, at_most, bound in ratio_bars:
        bar = bars[name]
        threshold = _finite(bar.get("bar"), f"{name} bar")
        _require(
            threshold == expected_threshold,
            f"{name} threshold does not match the registered contract",
        )
        geometric_mean = _finite(
            bar.get("geometric_mean"), f"{name} geometric mean"
        )
        compared = geometric_mean
        if bound is not None:
            compared = _finite(
                bar.get(f"bootstrap_95_percent_ci_{bound}"),
                f"{name} bootstrap CI {bound}",
            )
        expected = compared <= threshold if at_most else compared >= threshold
        _require(
            bar.get("passes") is expected,
            f"{name} verdict is inconsistent",
        )

    all_bars_pass = all(bool(bar["passes"]) for bar in bars.values())
    _require(
        report.get("all_bars_pass") is all_bars_pass,
        "serving trial aggregate bar summary is inconsistent",
    )
    formal_identity = (
        len(revisions) == len(machine_digests) == len(demand_digests) == 1
    )
    formally_qualified = trial_count >= 10 and formal_identity and all_bars_pass
    expected_grade = (
        "qualified"
        if mode == "formal" and formally_qualified
        else "failed"
        if mode == "formal"
        else "diagnostic"
    )
    _require(
        report.get("evidence_grade") == expected_grade,
        "serving trial evidence grade overstates or understates its verdict",
    )
    _require(
        report.get("qualified") is (mode == "formal" and formally_qualified),
        "serving trial qualified flag disagrees with registered evidence",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    validate(json.loads(args.report.resolve().read_text(encoding="utf-8")))
    print("serving_trials=valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
