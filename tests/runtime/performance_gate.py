#!/usr/bin/env python3
"""Exercise the combined performance evidence gate."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from experiments.check_regression import compare
from experiments.validate_performance_artifact import validate


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="nta-performance-gate-") as directory:
        root = Path(directory)
        profile = {
            "schema": 1,
            "classification": "nta-profile",
            "status": "complete",
            "tool": "perf",
            "command": ["perf", "stat", "./trial"],
            "returncode": 0,
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
        regression = compare(baseline, measured)
        for name, document in {
            "profile.json": profile,
            "baseline.json": baseline,
            "measured.json": measured,
            "regression.json": regression,
        }.items():
            (root / name).write_text(
                json.dumps(document, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        assert validate(root)["pass"] is True
        invalid = dict(regression)
        invalid["pass"] = False
        (root / "regression.json").write_text(json.dumps(invalid), encoding="utf-8")
        try:
            validate(root)
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
        assert not compare(zero_baseline, {"verification_failures": 0, "graph_ms": 1.0})[
            "pass"
        ]
    print("performance_gate=pass")


if __name__ == "__main__":
    main()
