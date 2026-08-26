#!/usr/bin/env python3
"""Validate profiler, baseline, measurement, and regression evidence together."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .check_regression import validate_baseline
except ImportError:  # Direct script execution.
    from check_regression import validate_baseline


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(output: Path) -> dict[str, Any]:
    profile = json.loads((output / "profile.json").read_text(encoding="utf-8"))
    baseline = json.loads((output / "baseline.json").read_text(encoding="utf-8"))
    measured = json.loads((output / "measured.json").read_text(encoding="utf-8"))
    regression = json.loads((output / "regression.json").read_text(encoding="utf-8"))
    capture = json.loads((output / "capture.json").read_text(encoding="utf-8"))

    _require(isinstance(profile, dict), "profiler artifact is not an object")
    _require(isinstance(baseline, dict), "performance baseline is not an object")
    _require(isinstance(regression, dict), "regression artifact is not an object")
    _require(isinstance(capture, dict), "performance capture is not an object")
    _require(profile.get("schema") == 1, "unsupported profiler artifact schema")
    _require(
        profile.get("classification") == "nta-profile", "invalid profiler artifact"
    )
    _require(profile.get("status") == "complete", "profiler evidence is not complete")
    _require(profile.get("tool") in {"nsys", "ncu", "perf"}, "invalid profiler tool")
    _require(profile.get("returncode") == 0, "profiler command failed")
    _require(
        isinstance(profile.get("command"), list) and profile["command"],
        "profiler command is missing",
    )
    outputs = profile.get("outputs")
    _require(
        isinstance(outputs, list) and outputs,
        "profiler produced no raw output artifact",
    )
    for name in outputs:
        relative = Path(str(name))
        _require(
            not relative.is_absolute() and ".." not in relative.parts,
            "profiler output path escapes the artifact directory",
        )
        output_path = output / relative
        _require(
            output_path.is_file() and output_path.stat().st_size > 0,
            f"profiler output is missing or empty: {name}",
        )
    _require(
        (output / "stdout.log").is_file(),
        "profiler stdout log is missing",
    )
    _require(
        capture.get("schema") == 1
        and capture.get("classification") == "nta-performance-capture",
        "invalid performance capture metadata",
    )
    for field, filename in (
        ("profile_digest", "profile.json"),
        ("baseline_digest", "baseline.json"),
        ("measured_digest", "measured.json"),
        ("regression_digest", "regression.json"),
    ):
        _require(
            isinstance(capture.get(field), str)
            and capture[field] == _digest(output / filename),
            f"performance capture digest mismatch: {filename}",
        )

    validate_baseline(baseline)
    _require(isinstance(measured, dict), "measured performance report is not an object")
    baseline_machine = baseline.get("machine")
    measured_machine = measured.get("machine")
    _require(
        isinstance(baseline_machine, str) and bool(baseline_machine),
        "performance baseline has no machine identity",
    )
    _require(
        isinstance(measured_machine, str) and bool(measured_machine),
        "measured performance report has no machine identity",
    )
    _require(
        baseline_machine == measured_machine,
        "baseline and measured performance ran on different machines",
    )
    _require(
        isinstance(baseline.get("revision"), str) and bool(baseline["revision"]),
        "performance baseline has no revision",
    )
    _require(
        isinstance(measured.get("revision"), str) and bool(measured["revision"]),
        "measured performance report has no revision",
    )
    _require(
        capture.get("machine") == measured_machine,
        "performance capture machine differs from measured report",
    )
    _require(
        capture.get("revision") == measured.get("revision"),
        "performance capture revision differs from measured report",
    )
    _require(
        measured.get("verification_failures", 0) == 0,
        "measured report has correctness failures",
    )
    _require(regression.get("schema") == 1, "unsupported regression artifact schema")
    _require(
        regression.get("classification") == "nta-performance-regression",
        "invalid regression artifact",
    )
    _require(regression.get("pass") is True, "performance regression gate did not pass")
    _require(
        isinstance(regression.get("checks"), list) and regression["checks"],
        "regression has no metric checks",
    )
    _require(
        all(check.get("pass") is True for check in regression["checks"]),
        "regression contains a failed metric check",
    )
    _require(not regression.get("failures"), "regression artifact lists failures")
    _require(
        regression.get("baseline_revision") == baseline.get("revision"),
        "regression does not identify the captured baseline",
    )
    _require(
        regression.get("measured_revision") == measured.get("revision"),
        "regression does not identify the measured revision",
    )
    return {
        "schema": 1,
        "classification": "nta-performance-evidence",
        "profiler": profile.get("tool"),
        "baseline_revision": baseline.get("revision"),
        "measured_revision": measured.get("revision"),
        "metric_count": len(regression["checks"]),
        "pass": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = validate(args.output.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
