#!/usr/bin/env python3
"""Compare a measured JSON report with a captured, machine-specific baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _get(document: dict[str, Any], path: str) -> float:
    value: Any = document
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            raise ValueError(f"metric path is missing: {path}")
        value = value[component]
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"metric is not finite: {path}")
    return float(value)


def validate_baseline(baseline: dict[str, Any]) -> None:
    """Validate the portable shape of a captured machine-specific baseline."""
    if not isinstance(baseline, dict):
        raise ValueError("performance baseline must be an object")
    if (
        baseline.get("schema") != 1
        or baseline.get("classification") != "nta-performance-baseline"
    ):
        raise ValueError("baseline is not a captured NTA performance baseline")
    report = baseline.get("report")
    metrics = baseline.get("metrics")
    if not isinstance(report, dict) or not isinstance(metrics, list) or not metrics:
        raise ValueError("baseline must contain a report and at least one metric")
    if int(report.get("verification_failures", 0)) != 0:
        raise ValueError("baseline contains correctness failures")
    for metric in metrics:
        if (
            not isinstance(metric, dict)
            or not metric.get("name")
            or not metric.get("path")
        ):
            raise ValueError("baseline metric must have a name and path")
        direction = metric.get("direction", "lower_is_better")
        if direction not in {"lower_is_better", "higher_is_better"}:
            raise ValueError(f"unsupported performance direction: {direction}")
        tolerance = float(metric.get("relative_tolerance", 0.05))
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError(
                "performance relative tolerance must be finite and nonnegative"
            )
        if "absolute_tolerance" in metric:
            absolute_tolerance = float(metric["absolute_tolerance"])
            if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0:
                raise ValueError(
                    "performance absolute tolerance must be finite and nonnegative"
                )
        _get(report, str(metric["path"]))


def compare(baseline: dict[str, Any], measured: dict[str, Any]) -> dict[str, Any]:
    validate_baseline(baseline)
    if not isinstance(measured, dict):
        raise ValueError("measured performance report must be an object")
    if int(measured.get("verification_failures", 0)) != 0:
        raise ValueError("measured report contains correctness failures")
    checks = []
    failures = []
    for metric in baseline.get("metrics", []):
        name = str(metric["name"])
        base = _get(baseline["report"], str(metric["path"]))
        current = _get(measured, str(metric["path"]))
        tolerance = float(metric.get("relative_tolerance", 0.05))
        direction = metric.get("direction", "lower_is_better")
        if direction not in {"lower_is_better", "higher_is_better"}:
            raise ValueError(f"unsupported performance direction: {direction}")
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError(
                "performance relative tolerance must be finite and nonnegative"
            )
        if base == 0.0:
            absolute_tolerance = float(metric.get("absolute_tolerance", 0.0))
            relative_change = None
            violation = abs(current - base) > absolute_tolerance
        else:
            relative_change = (current - base) / base
            violation = (
                relative_change > tolerance
                if direction == "lower_is_better"
                else relative_change < -tolerance
            )
        item = {
            "name": name,
            "baseline": base,
            "measured": current,
            "relative_change": relative_change,
            "relative_tolerance": tolerance,
            "absolute_tolerance": float(metric.get("absolute_tolerance", 0.0)),
            "direction": direction,
            "pass": not violation,
        }
        checks.append(item)
        if violation:
            failures.append(name)
    return {
        "schema": 1,
        "classification": "nta-performance-regression",
        "pass": not failures,
        "failures": failures,
        "checks": checks,
        "baseline_revision": baseline.get("revision"),
        "measured_revision": measured.get("revision"),
        "baseline_machine": baseline.get("machine"),
        "measured_machine": measured.get("machine"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--measured", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        json.loads(args.baseline.read_text(encoding="utf-8")),
        json.loads(args.measured.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
