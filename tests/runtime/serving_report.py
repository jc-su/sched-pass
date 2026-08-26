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
        "cotenant_gpu_samples": 0,
        "gpu_samples": 1,
        "gpu_sampling_errors": 0,
        "gpu_sampling_complete": True,
        "cotenant_pids_seen": [],
        "verification_failures": 0,
        "correctness": {"verification_failures": 0, "generated_text_sha256": "all"},
        "generated_text_sha256": "all",
        "records": records,
        "engine_stats": [
            {
                "backend": "nta_flashinfer",
                "consumer_contract": {
                    "schema": 1,
                    "engine": "sglang",
                    "backend": "nta_flashinfer",
                    "kind": "native_work_unit",
                    "exact_demand": True,
                    "typed_work_plan": True,
                    "native_submission": True,
                    "numerical_consumer": True,
                    "engine_version": "0.5.16",
                },
            }
        ],
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
    framework_reference = copy.deepcopy(nta)
    framework_reference["engine_stats"][0]["consumer_contract"] = {
        "schema": 1,
        "engine": "sglang",
        "backend": "nta_flashinfer",
        "kind": "framework_reference",
        "exact_demand": True,
        "typed_work_plan": False,
        "native_submission": False,
        "numerical_consumer": True,
        "engine_version": "0.5.16",
    }
    framework_reference["engine_stats"][0][
        "stock_prefetched_external_attention_launches"
    ] = 1
    validate(framework_reference)
    invalid_reference = copy.deepcopy(framework_reference)
    invalid_reference["engine_stats"][0][
        "stock_prefetched_external_attention_launches"
    ] = 0
    try:
        validate(invalid_reference)
    except ValueError as error:
        assert "external exact prefetch" in str(error)
    else:
        raise AssertionError("unfenced framework-reference evidence was accepted")
    invalid_contract = copy.deepcopy(comparison)
    invalid_contract["nta"]["engine_stats"][0]["consumer_contract"]["kind"] = (
        "projection_only"
    )
    invalid_contract["nta"]["engine_stats"][0]["consumer_contract"][
        "native_submission"
    ] = False
    invalid_contract["nta"]["engine_stats"][0]["consumer_contract"][
        "typed_work_plan"
    ] = False
    invalid_contract["nta"]["engine_stats"][0]["consumer_contract"][
        "numerical_consumer"
    ] = False
    try:
        validate(invalid_contract)
    except ValueError as error:
        assert "projection-only" in str(error)
    else:
        raise AssertionError("projection-only serving evidence was accepted")

    host_tier = copy.deepcopy(nta)
    host_tier["engine_stats"][0].update(
        {"serving_tier": "host_staged", "tier_fallback": False}
    )
    validate(host_tier)
    invalid_host_tier = copy.deepcopy(host_tier)
    invalid_host_tier["engine_stats"][0]["cxl_direct_work_items"] = 1
    try:
        validate(invalid_host_tier)
    except ValueError as error:
        assert "CXL direct work" in str(error)
    else:
        raise AssertionError("host-staged evidence reported CXL work")

    invalid_type = copy.deepcopy(comparison)
    invalid_type["nta"]["engine_stats"][0]["consumer_contract"][
        "numerical_consumer"
    ] = 1
    try:
        validate(invalid_type)
    except ValueError as error:
        assert "not boolean" in str(error)
    else:
        raise AssertionError("non-boolean consumer evidence was accepted")
    invalid_backend = copy.deepcopy(comparison)
    invalid_backend["nta"]["engine_stats"] = [
        {"backend": "stock_flashinfer", "latency_ms": 1.0}
    ]
    try:
        validate(invalid_backend)
    except ValueError as error:
        assert "numerical consumer" in str(error)
    else:
        raise AssertionError("non-NTA engine statistics were accepted as NTA evidence")
    invalid = copy.deepcopy(comparison)
    invalid["outputs_diverge"] = True
    try:
        validate(invalid)
    except ValueError as error:
        assert "divergent" in str(error)
    else:
        raise AssertionError("divergent serving output was accepted")
    invalid_environment = copy.deepcopy(comparison)
    invalid_environment["nta"]["cotenant_gpu_samples"] = 1
    invalid_environment["nta"]["cotenant_pids_seen"] = [12345]
    try:
        validate(invalid_environment)
    except ValueError as error:
        assert "contaminated" in str(error)
    else:
        raise AssertionError("co-tenant-contaminated serving evidence was accepted")
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
