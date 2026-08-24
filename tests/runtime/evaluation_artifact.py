#!/usr/bin/env python3
"""Exercise the complete dependency-free paired evaluation artifact gate."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.bailian import normalize, write_workload  # noqa: E402


STRATUM = {
    "request_state": "mixed",
    "granularity": "page_group",
    "load_ratio": "balanced",
    "availability_skew": "medium",
    "staging_pressure": "near_capacity",
    "arrival": "batch_release",
}


def _command(goodput: float) -> list[str]:
    result = {
        "classification": "nta-evaluation-fixture",
        "verification_failures": 0,
        "slo_goodput": goodput,
        "p95_ttft_seconds": 0.1,
        "p99_itl_seconds": 0.01,
        "littles_law": {
            "method": "finite_window_arrival_departure_accounting",
            "residual": 0.0,
        },
    }
    code = f"import json; print(json.dumps({result!r}))"
    return [sys.executable, "-c", code]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="nta-evaluation-artifact-") as directory:
        root = Path(directory)
        manifest, rows = normalize(
            [
                {
                    "request_id": "fixture-a",
                    "input_length": 32,
                    "output_length": 4,
                    "hash_ids": ["prefix-0", "prefix-1"],
                },
                {
                    "request_id": "fixture-b",
                    "input_length": 48,
                    "output_length": 5,
                    "hash_ids": ["prefix-0", "prefix-1", "prefix-2"],
                },
            ],
            arrival_mode="batch_release",
            synthesize_prompts=True,
        )
        workload = root / "workload.json"
        records = root / "records.jsonl"
        write_workload(workload, records, manifest, rows)
        spec = {
            "schema": 1,
            "classification": "nta-paired-evaluation",
            "workload_manifest": str(workload),
            "repetitions": 5,
            "seed": 7,
            "experiments": [
                {
                    "name": "fixture-pair",
                    "variant": "B0",
                    "arm": "B0",
                    "tier": "hbm",
                    "demand_semantics": "exact",
                    "stratum": STRATUM,
                    "command": _command(1.0),
                    "metrics": [
                        "slo_goodput",
                        "p95_ttft_seconds",
                        "p99_itl_seconds",
                    ],
                },
                {
                    "name": "fixture-pair",
                    "variant": "B1",
                    "arm": "B1",
                    "tier": "hbm",
                    "demand_semantics": "exact",
                    "stratum": STRATUM,
                    "command": _command(1.2),
                    "metrics": [
                        "slo_goodput",
                        "p95_ttft_seconds",
                        "p99_itl_seconds",
                    ],
                },
            ],
            "comparisons": [
                {
                    "name": "fixture-goodput",
                    "experiment": "fixture-pair",
                    "numerator_variant": "B1",
                    "denominator_variant": "B0",
                    "metric": "slo_goodput",
                }
            ],
        }
        spec_path = root / "spec.json"
        spec_path.write_text(json.dumps(spec) + "\n", encoding="utf-8")
        output = root / "evaluation"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments" / "run_evaluation.py"),
                "--spec",
                str(spec_path),
                "--output-dir",
                str(output),
                "--allow-dirty",
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments" / "validate_evaluation_artifact.py"),
                str(output),
            ],
            cwd=ROOT,
            check=True,
        )
        metadata = json.loads(
            (output / "evaluation-metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["workload_demand_digest"] == manifest["demand_trace_digest"]
    print("evaluation_artifact=pass")


if __name__ == "__main__":
    main()
