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


def compare(baseline: dict[str, Any], measured: dict[str, Any]) -> dict[str, Any]:
    if baseline.get("schema") != 1 or baseline.get("classification") != "nta-performance-baseline":
        raise ValueError("baseline is not a captured NTA performance baseline")
    if int(baseline["report"].get("verification_failures", 0)) != 0:
        raise ValueError("baseline contains correctness failures")
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
        relative_change = (current - base) / base if base else 0.0
        violation = relative_change > tolerance if direction == "lower_is_better" else relative_change < -tolerance
        item = {"name": name, "baseline": base, "measured": current, "relative_change": relative_change, "relative_tolerance": tolerance, "direction": direction, "pass": not violation}
        checks.append(item)
        if violation:
            failures.append(name)
    return {"schema": 1, "classification": "nta-performance-regression", "pass": not failures, "failures": failures, "checks": checks, "baseline_revision": baseline.get("revision"), "measured_revision": measured.get("revision")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--measured", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.baseline.read_text(encoding="utf-8")), json.loads(args.measured.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
