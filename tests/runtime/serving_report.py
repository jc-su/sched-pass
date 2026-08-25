#!/usr/bin/env python3
"""Test the strict serving-evidence report contract without SGLang."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.validate_serving_report import validate  # noqa: E402


def single() -> dict[str, object]:
    records = [
        {
            "kind": "resident",
            "request_id": "request-0",
            "arrival_offset_seconds": 0.0,
            "submitted_offset_seconds": 0.1,
            "finished_offset_seconds": 0.2,
            "ttft_seconds": 0.1,
            "tpot_seconds": 0.01,
            "p99_itl_seconds": 0.01,
            "admission_delay_seconds": 0.0,
            "system_time_seconds": 0.2,
            "completion_tokens": 2,
            "text_sha256": "resident-digest",
            "inter_token_seconds": [0.01],
        }
    ]
    return {
        "schema": 1,
        "classification": "sglang-hicache-load",
        "revision": "test-revision",
        "machine": {"hostname": "test"},
        "demand_semantics": "exact",
        "placement_proven": True,
        "verification_failures": 0,
        "correctness": {"verification_failures": 0, "generated_text_sha256": "all"},
        "generated_text_sha256": "all",
        "records": records,
        "engine_stats": [{"backend": "nta_flashinfer"}],
        "p50_ttft_seconds": 0.1,
        "p95_ttft_seconds": 0.1,
        "p99_ttft_seconds": 0.1,
        "p50_tpot_seconds": 0.01,
        "p95_tpot_seconds": 0.01,
        "p99_tpot_seconds": 0.01,
        "p99_itl_seconds": 0.01,
        "slo_goodput": {
            "qualified_requests": 1,
            "total_requests": 1,
            "goodput_requests_per_second": 5.0,
        },
        "littles_law": {
            "method": "finite_window_arrival_departure_accounting",
            "residual": 0.0,
        },
        "selected_bytes": None,
        "physical_bytes": None,
        "byte_accounting_status": "not exposed by SGLang engine metadata",
    }


def main() -> None:
    stock = single()
    nta = copy.deepcopy(stock)
    nta["correctness"] = {"verification_failures": 0, "generated_text_sha256": "nta"}
    comparison = {
        "schema": 1,
        "classification": "sglang-hicache-load-comparison",
        "outputs_diverge": False,
        "stock": stock,
        "nta": nta,
        "mechanism_activation": {
            "external_launches": 1,
            "external_attention_accounted": True,
            "external_attention_transformed": True,
            "external_attention_stock_consumer": False,
            "transformed_external_launches": 1,
            "stock_prefetched_external_launches": 0,
            "fallback_batches": 0,
            "transformed_direct_launches": 0,
            "ticketed_incremental_launches": 1,
        },
        "stock_slo_goodput": 5.0,
        "nta_slo_goodput": 6.0,
        "goodput_ratio": 1.2,
        "output_throughput_ratio": 1.1,
        "external_p95_ttft_ratio": 0.9,
    }
    validate(comparison)
    invalid = copy.deepcopy(comparison)
    invalid["outputs_diverge"] = True
    try:
        validate(invalid)
    except ValueError as error:
        assert "divergent" in str(error)
    else:
        raise AssertionError("divergent serving output was accepted")
    invalid_single = copy.deepcopy(stock)
    invalid_single["records"][0]["request_id"] = "duplicate"
    invalid_single["workload"] = {
        "manifest_digest": "manifest",
        "records_digest": "records",
        "demand_trace_digest": "demand",
        "tokenization_errors": 0,
        "request_id_order": ["request-0"],
    }
    invalid_single["demand_trace_digest"] = "demand"
    try:
        validate(invalid_single)
    except ValueError as error:
        assert "request identities" in str(error)
    else:
        raise AssertionError(
            "serving report with an unknown request identity was accepted"
        )
    comparison["stock"]["engine_stats"] = []
    validate(comparison)
    with tempfile.TemporaryDirectory(prefix="nta-serving-artifact-") as directory:
        root = Path(directory)
        result = root / "serving.json"
        result.write_text(json.dumps(single()) + "\n", encoding="utf-8")
        bundle = root / "bundle"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments" / "reproduce.py"),
                "--profile",
                "serving",
                "--output",
                str(bundle),
                "--result",
                str(result),
                "--allow-dirty",
                "--",
                sys.executable,
                "-c",
                "print('serving-fixture')",
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments" / "validate_bundle.py"),
                str(bundle),
            ],
            cwd=ROOT,
            check=True,
        )
    print("serving_report=pass")


if __name__ == "__main__":
    main()
