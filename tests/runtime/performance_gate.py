#!/usr/bin/env python3
"""Exercise the combined performance evidence gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

from experiments.check_regression import compare
from experiments.capture_performance import compose
from experiments.validate_performance_artifact import validate


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="nta-performance-gate-") as directory:
        root = Path(directory)
        sources = root / "sources"
        profile_source = sources / "profile"
        profile_source.mkdir(parents=True)
        profile = {
            "schema": 1,
            "classification": "nta-profile",
            "status": "complete",
            "tool": "perf",
            "command": ["perf", "stat", "./trial"],
            "returncode": 0,
            "outputs": ["perf-stat.csv"],
        }
        baseline = {
            "schema": 1,
            "classification": "nta-performance-baseline",
            "revision": "baseline-revision",
            "machine": "fixture-machine",
            "report": {"verification_failures": 0, "graph_ms": 1.0},
            "metrics": [
                {
                    "name": "graph",
                    "path": "graph_ms",
                    "direction": "lower_is_better",
                    "relative_tolerance": 0.05,
                }
            ],
        }
        measured = {
            "revision": "measured-revision",
            "machine": "fixture-machine",
            "verification_failures": 0,
            "graph_ms": 1.01,
        }
        (profile_source / "profile.json").write_text(
            json.dumps(profile, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (sources / "baseline.json").write_text(
            json.dumps(baseline, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (sources / "measured.json").write_text(
            json.dumps(measured, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (profile_source / "perf-stat.csv").write_text("cycles,100\n", encoding="utf-8")
        (profile_source / "stdout.log").write_text("perf complete\n", encoding="utf-8")
        evidence = root / "captured"
        result = compose(
            profile_source=profile_source,
            baseline_source=sources / "baseline.json",
            measured_source=sources / "measured.json",
            output=evidence,
        )
        assert result["valid"] is True
        assert validate(evidence)["pass"] is True
        regression = json.loads((evidence / "regression.json").read_text())
        invalid = dict(regression)
        invalid["pass"] = False
        (evidence / "regression.json").write_text(json.dumps(invalid), encoding="utf-8")
        capture = json.loads((evidence / "capture.json").read_text(encoding="utf-8"))
        capture["regression_digest"] = hashlib.sha256(
            (evidence / "regression.json").read_bytes()
        ).hexdigest()
        (evidence / "capture.json").write_text(json.dumps(capture), encoding="utf-8")
        try:
            validate(evidence)
        except ValueError as error:
            assert "did not pass" in str(error)
        else:
            raise AssertionError("failed performance evidence was accepted")
        zero_baseline = dict(baseline)
        zero_baseline["report"] = {"verification_failures": 0, "graph_ms": 0.0}
        zero_baseline["metrics"] = [
            {
                "name": "zero_metric",
                "path": "graph_ms",
                "direction": "lower_is_better",
                "relative_tolerance": 0.05,
            }
        ]
        assert compare(zero_baseline, {"verification_failures": 0, "graph_ms": 0.0})[
            "pass"
        ]
        assert not compare(
            zero_baseline, {"verification_failures": 0, "graph_ms": 1.0}
        )["pass"]
    print("performance_gate=pass")


if __name__ == "__main__":
    main()
