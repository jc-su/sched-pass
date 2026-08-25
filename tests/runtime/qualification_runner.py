#!/usr/bin/env python3
"""Smoke-test the qualification trial runner and confidence summary."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        spec = {
            "schema": 1,
            "repetitions": 2,
            "seed": 7,
            "experiments": [
                {
                    "name": "runner-smoke",
                    "variant": "mechanism",
                    "command": [
                        sys.executable,
                        "-c",
                        "import json; print(json.dumps({'latency_ms': 2.5}))",
                    ],
                    "metrics": ["latency_ms"],
                },
                {
                    "name": "runner-smoke",
                    "variant": "baseline",
                    "command": [
                        sys.executable,
                        "-c",
                        "import json; print(json.dumps({'latency_ms': 5.0}))",
                    ],
                    "metrics": ["latency_ms"],
                },
            ],
            "comparisons": [
                {
                    "name": "mechanism-speedup",
                    "experiment": "runner-smoke",
                    "numerator_variant": "baseline",
                    "denominator_variant": "mechanism",
                    "metric": "latency_ms",
                }
            ],
        }
        spec_path = root / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        output = root / "output"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/run-qualified-trials.py"),
                "--spec",
                str(spec_path),
                "--output-dir",
                str(output),
                "--allow-dirty",
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        records = (output / "trials.jsonl").read_text(encoding="utf-8").splitlines()
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert len(records) == 4
        assert metadata["spec"] == spec
        assert "spec_sha256" not in metadata
        assert all("log_sha256" not in json.loads(record) for record in records)
        metric = next(
            item for item in summary["summaries"] if item["variant"] == "mechanism"
        )["metrics"]["latency_ms"]
        assert metric["count"] == 2
        assert metric["mean"] == 2.5
        comparison = summary["comparisons"][0]
        assert comparison["ratio"] == "baseline/mechanism"
        assert comparison["interval"]["mean"] == 2.0

    print("qualification_runner=pass")


if __name__ == "__main__":
    main()
