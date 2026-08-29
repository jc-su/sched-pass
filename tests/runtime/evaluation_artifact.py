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
from experiments.workload_scenario import describe_workload_scenario  # noqa: E402


def _command(goodput: float) -> list[str]:
    result = {
        "classification": "nta-evaluation-fixture",
        "verification_failures": 0,
        "slo_goodput": goodput,
        "p95_ttft_seconds": 0.1,
        "p99_itl_seconds": 0.01,
        "finite_window_accounting": {
            "method": "finite_window_arrival_departure_accounting",
            "arrival_rate_per_second": 5.0,
            "completion_rate_per_second": 5.0,
            "mean_in_system": 0.5,
            "mean_system_time_seconds": 0.1,
            "occupancy_area_request_seconds": 0.1,
            "sum_residence_seconds": 0.1,
            "interpretation": "descriptive_client_timestamp_accounting",
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
        scenario = describe_workload_scenario("fixture", workload)
        manifest_b, rows_b = normalize(
            [
                {
                    "request_id": "fixture-c",
                    "input_length": 96,
                    "output_length": 8,
                    "hash_ids": ["prefix-b0", "prefix-b1"],
                },
                {
                    "request_id": "fixture-d",
                    "input_length": 160,
                    "output_length": 13,
                    "hash_ids": ["prefix-b0", "prefix-b1", "prefix-b2"],
                },
            ],
            arrival_mode="batch_release",
            synthesize_prompts=True,
        )
        workload_b = root / "workload-b.json"
        records_b = root / "records-b.jsonl"
        write_workload(workload_b, records_b, manifest_b, rows_b)
        scenario_b = describe_workload_scenario("fixture-b", workload_b)
        spec = {
            "schema": 1,
            "classification": "nta-paired-evaluation",
            "workload_manifests": [str(workload), str(workload_b)],
            "repetitions": 5,
            "seed": 7,
            "experiments": [
                {
                    "name": "fixture-pair",
                    "variant": "A0",
                    "arm": "A0",
                    "tier": "hbm",
                    "demand_semantics": "exact",
                    "stratum": scenario,
                    "workload_manifest": str(workload),
                    "command": _command(1.0),
                    "metrics": [
                        "slo_goodput",
                        "p95_ttft_seconds",
                        "p99_itl_seconds",
                    ],
                },
                {
                    "name": "fixture-pair",
                    "variant": "A1",
                    "arm": "A1",
                    "tier": "hbm",
                    "demand_semantics": "exact",
                    "stratum": scenario,
                    "workload_manifest": str(workload),
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
                    "numerator_variant": "A1",
                    "denominator_variant": "A0",
                    "metric": "slo_goodput",
                }
            ],
        }
        for experiment in list(spec["experiments"]):
            clone = json.loads(json.dumps(experiment))
            clone["name"] = "fixture-pair-b"
            clone["stratum"] = scenario_b
            clone["workload_manifest"] = str(workload_b)
            spec["experiments"].append(clone)
        comparison_b = dict(spec["comparisons"][0])
        comparison_b["name"] = "fixture-goodput-b"
        comparison_b["experiment"] = "fixture-pair-b"
        spec["comparisons"].append(comparison_b)
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
        assert len(metadata["workloads"]) == 2
        assert {
            entry["demand_trace_digest"] for entry in metadata["workloads"]
        } == {
            manifest["demand_trace_digest"],
            manifest_b["demand_trace_digest"],
        }
        artifact = root / "artifact"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments" / "reproduce.py"),
                "--profile",
                "evaluation",
                "--spec",
                str(spec_path),
                "--output",
                str(artifact),
                "--allow-dirty",
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments" / "validate_bundle.py"),
                str(artifact),
            ],
            cwd=ROOT,
            check=True,
        )
    print("evaluation_artifact=pass")


if __name__ == "__main__":
    main()
