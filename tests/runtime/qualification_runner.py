#!/usr/bin/env python3
"""Smoke-test the qualification trial runner and confidence summary."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from experiments.bailian import normalize, write_workload  # noqa: E402


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

        occupied = root / "occupied"
        occupied.mkdir()
        (occupied / "sentinel").write_text("keep\n", encoding="utf-8")
        refused = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/run-qualified-trials.py"),
                "--spec",
                str(spec_path),
                "--output-dir",
                str(occupied),
                "--allow-dirty",
            ],
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert refused.returncode != 0
        assert "not empty" in refused.stderr
        assert (occupied / "sentinel").read_text(encoding="utf-8") == "keep\n"

        workload_manifest, workload_records = normalize(
            [
                {
                    "chat_id": "qualification-workload",
                    "input_length": 16,
                    "output_length": 1,
                }
            ]
        )
        workload_path = root / "workload" / "manifest.json"
        records_path = root / "workload" / "records.jsonl"
        write_workload(workload_path, records_path, workload_manifest, workload_records)
        malformed_manifest = root / "malformed-workload.json"
        malformed_document = json.loads(workload_path.read_text(encoding="utf-8"))
        malformed_document["request_count"] = 2
        malformed_manifest.write_text(json.dumps(malformed_document), encoding="utf-8")
        invalid_spec = dict(spec)
        invalid_spec["workload_manifest"] = str(malformed_manifest)
        invalid_spec_path = root / "invalid-spec.json"
        invalid_spec_path.write_text(json.dumps(invalid_spec), encoding="utf-8")
        refused_manifest = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/run-qualified-trials.py"),
                "--spec",
                str(invalid_spec_path),
                "--output-dir",
                str(root / "invalid-output"),
                "--allow-dirty",
            ],
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert refused_manifest.returncode != 0
        assert "failed validation" in refused_manifest.stderr

    print("qualification_runner=pass")


if __name__ == "__main__":
    main()
