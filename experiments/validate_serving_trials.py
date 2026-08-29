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


def validate(report: dict[str, Any]) -> dict[str, Any]:
    _require(report.get("schema") == 2, "unsupported serving trial schema")
    _require(
        report.get("classification") == "sglang-hicache-load-qualification",
        "result is not an SGLang serving qualification aggregate",
    )
    mode = report.get("mode")
    _require(mode in {"formal", "diagnostic"}, "unknown serving trial mode")
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
        "resident_p99_itl",
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
    registered = bars["registered_goodput"]
    _finite(registered.get("bar"), "registered-goodput bar")
    _require(
        isinstance(registered.get("all_requests_have_token_level_itl"), bool),
        "registered-goodput bar lacks token-level ITL eligibility",
    )
    _require(
        registered["all_requests_have_token_level_itl"]
        or registered.get("passes") is False,
        "registered-goodput bar passed with requests lacking token-level ITL",
    )
    if registered.get("geometric_mean") is None or registered.get("ci_floor") is None:
        _require(
            registered.get("passes") is False,
            "registered-goodput bar passed without a finite estimate",
        )
    else:
        _finite(
            registered.get("geometric_mean"),
            "registered-goodput geometric mean",
        )
        _finite(registered.get("ci_floor"), "registered-goodput CI floor")
    resident = bars["resident_p99_itl"]
    _finite(resident.get("bar"), "resident-ITL bar")
    _finite(resident.get("geometric_mean"), "resident-ITL geometric mean")

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
