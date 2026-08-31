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
from experiments.validate_serving_trials import (  # noqa: E402
    validate as validate_trials,
)
from experiments.bailian import normalize, write_workload  # noqa: E402
from experiments.validate_workload import validate as validate_workload  # noqa: E402
from experiments.serving_metrics import (  # noqa: E402
    preregistered_goodput,
    preregistered_joint_goodput,
    relative_goodput,
    relative_thresholds,
)
from experiments.workload_heterogeneity import (  # noqa: E402
    serving_batch_heterogeneity,
)
from nta_runtime.resource_contract import resource_contract  # noqa: E402


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
            "admission_delay_seconds": 0.1,
            "system_time_seconds": 0.2,
            "completion_tokens": 2,
            "itl_sample_count": 1,
            "token_timestamps_exact": True,
            "token_timestamp_source": "sglang_stream_interval_1_completion_delta",
            "input_tokens": 32,
            "device_cached_tokens": 31,
            "host_cached_tokens": 0,
            "text_sha256": "resident-digest",
            "inter_token_seconds": [0.01],
        }
    ]
    report = {
        "schema": 1,
        "classification": "sglang-hicache-load",
        "revision": "test-revision",
        "attention_backend": "nta_flashinfer",
        "machine": {"hostname": "test"},
        "demand_semantics": "exact",
        "placement_proven": True,
        "initial_placement_proof": {
            "reason": "measured_reconstruction",
            "attempt": 1,
            "destructive_probe_followed_by_disjoint_replay": True,
            "observations": [
                {"index": 0, "expected": 32, "device": 0, "host": 32}
            ],
        },
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
                "ticketed_incremental_launches": 1,
                "stock_prefetched_external_attention_launches": 0,
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
        "resident_p95_ttft_seconds": 0.1,
        "resident_p95_tpot_seconds": 0.01,
        "resident_p99_itl_seconds": 0.01,
        "external_p95_ttft_seconds": 0.1,
        "elapsed_seconds": 0.2,
        "request_throughput": 5.0,
        "output_token_throughput": 10.0,
        "resident_output_token_throughput": 10.0,
        "external_output_token_throughput": 0.0,
        "slo_goodput": {
            "qualified_requests": 1,
            "total_requests": 1,
            "requests_with_token_level_itl": 1,
            "attainment": 1.0,
            "goodput_requests_per_second": 5.0,
            "thresholds_seconds": {
                "ttft": 8.0,
                "tpot": 0.05,
                "p99_itl": 0.1,
            },
        },
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
        "selected_bytes": None,
        "physical_bytes": None,
        "byte_accounting_status": "not exposed by SGLang engine metadata",
    }
    report["batch_heterogeneity"] = serving_batch_heterogeneity(
        records, report["engine_stats"]
    )
    return report


def main() -> None:
    heterogeneous_records = [
        {
            "kind": "resident",
            "input_tokens": 32,
            "completion_tokens": 8,
            "host_cached_tokens": 0,
            "device_cached_tokens": 31,
            "arrival_offset_seconds": 0.0,
            "submitted_offset_seconds": 0.0,
            "finished_offset_seconds": 0.2,
        },
        {
            "kind": "external",
            "input_tokens": 128,
            "completion_tokens": 2,
            "host_cached_tokens": 95,
            "device_cached_tokens": 1,
            "arrival_offset_seconds": 0.05,
            "submitted_offset_seconds": 0.05,
            "finished_offset_seconds": 0.15,
        },
    ]
    heterogeneity = serving_batch_heterogeneity(
        heterogeneous_records,
        [
            {
                "multi_request_engine_batches": 1,
                "heterogeneous_engine_batches": 1,
                "multi_axis_heterogeneous_batches": 1,
                "sequence_length_heterogeneous_batches": 1,
                "availability_heterogeneous_batches": 1,
                "mixed_dependency_layers": 1,
            }
        ],
    )
    assert heterogeneity["proven"] is True
    assert heterogeneity["native_mixed_consumer_proven"] is True
    assert heterogeneity["heterogeneous_axis_count"] == 6
    assert abs(
        heterogeneity["client_overlap"]["resident_external_overlap_seconds"]
        - 0.1
    ) < 1e-12

    direct_heterogeneity = serving_batch_heterogeneity(
        heterogeneous_records,
        [
            {
                "multi_request_engine_batches": 1,
                "heterogeneous_engine_batches": 1,
                "sequence_length_heterogeneous_batches": 1,
                "availability_heterogeneous_batches": 1,
                "mixed_dependency_layers": 0,
            }
        ],
    )
    assert direct_heterogeneity["proven"] is True
    assert direct_heterogeneity["native_mixed_consumer_proven"] is False

    stock = single()
    stock["attention_backend"] = "flashinfer"
    stock["engine_stats"] = []
    joint_boundary = copy.deepcopy(stock)
    joint_boundary["records"][0]["tpot_seconds"] = 0.060
    assert preregistered_goodput(joint_boundary)["qualified_requests"] == 1
    assert (
        preregistered_joint_goodput(joint_boundary)["qualified_requests"] == 0
    )
    nta = copy.deepcopy(stock)
    nta["attention_backend"] = "nta_flashinfer"
    nta["engine_stats"] = single()["engine_stats"]
    nta["correctness"] = {"verification_failures": 0, "generated_text_sha256": "nta"}
    comparison = {
        "schema": 1,
        "classification": "sglang-hicache-load-comparison",
        "revision": "test-revision",
        "harness_args": {
            "fixture": True,
            "require_exercised_path": [],
            "require_native_frontier_layers": None,
            "require_ready_stock_layers": None,
            "require_progressive_layers": None,
        },
        "execution_order": ["flashinfer", "nta_flashinfer"],
        "evidence_scope": "native_work_unit",
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
            "native_work_unit_active": True,
            "heterogeneous_work_unit_active": False,
            "batch_heterogeneity_proven": False,
            "transport_only": False,
        },
        "slo_scale": 1.5,
        "transport_execution": {
            "schema": 1,
            "native_demand_sm": {"bytes": 0, "layers": 0, "exercised": False},
            "prefetch_sm": {"bytes": 0, "exercised": False},
            "prefetch_copy_engine": {
                "bytes": 0,
                "operations": 0,
                "submissions": 0,
                "exercised": False,
            },
            "prefetch_hybrid": {"parallel_waves": 0, "exercised": False},
            "partial_consumer": {
                "layers": 0,
                "exact_window_layers": 0,
                "exercised": False,
            },
            "frontier": {
                "native_layers": 0,
                "ready_stock_layers": 0,
                "progress_rounds": 0,
            },
        },
    }
    thresholds = relative_thresholds(stock, comparison["slo_scale"])
    stock_goodput = relative_goodput(stock, thresholds)
    nta_goodput = relative_goodput(nta, thresholds)
    stock_preregistered = preregistered_goodput(stock)
    nta_preregistered = preregistered_goodput(nta)
    stock_joint_preregistered = preregistered_joint_goodput(stock)
    nta_joint_preregistered = preregistered_joint_goodput(nta)
    comparison.update(
        {
            "slo_thresholds_seconds": thresholds,
            "stock_goodput": stock_goodput,
            "nta_goodput": nta_goodput,
            "stock_preregistered_goodput": stock_preregistered,
            "nta_preregistered_goodput": nta_preregistered,
            "stock_preregistered_joint_goodput": stock_joint_preregistered,
            "nta_preregistered_joint_goodput": nta_joint_preregistered,
            "stock_slo_goodput": 5.0,
            "nta_slo_goodput": 5.0,
            "stock_p50_ttft_seconds": 0.1,
            "stock_p95_ttft_seconds": 0.1,
            "stock_p99_ttft_seconds": 0.1,
            "stock_p99_itl_seconds": 0.01,
            "nta_p50_ttft_seconds": 0.1,
            "nta_p95_ttft_seconds": 0.1,
            "nta_p99_ttft_seconds": 0.1,
            "nta_p99_itl_seconds": 0.01,
            "goodput_ratio": 1.0,
            "preregistered_goodput_ratio": 1.0,
            "preregistered_joint_goodput_ratio": 1.0,
            "output_throughput_ratio": 1.0,
            "resident_output_throughput_ratio": 1.0,
            "external_output_throughput_ratio": 1.0,
            "resident_p95_ttft_ratio": 1.0,
            "resident_p95_tpot_ratio": 1.0,
            "resident_p99_itl_ratio": 1.0,
            "external_p95_ttft_ratio": 1.0,
        }
    )
    validate(comparison)
    standalone = copy.deepcopy(nta)
    for field in (
        "cotenant_gpu_samples",
        "gpu_samples",
        "gpu_sampling_errors",
        "gpu_sampling_complete",
        "cotenant_pids_seen",
    ):
        standalone.pop(field)
    validate(standalone)
    tampered_goodput = copy.deepcopy(comparison)
    tampered_goodput["nta_preregistered_goodput"][
        "goodput_requests_per_second"
    ] *= 2
    try:
        validate(tampered_goodput)
    except ValueError as error:
        assert "preregistered goodput" in str(error)
    else:
        raise AssertionError("tampered preregistered goodput passed validation")
    mislabeled_transport = copy.deepcopy(comparison)
    mislabeled_transport["transport_execution"]["prefetch_copy_engine"].update(
        {"bytes": 4096, "operations": 2, "submissions": 1, "exercised": True}
    )
    try:
        validate(mislabeled_transport)
    except ValueError as error:
        assert "timed counters" in str(error)
    else:
        raise AssertionError("self-reported transport labels bypassed timed counters")
    wrong_frontier = copy.deepcopy(comparison)
    wrong_frontier["harness_args"]["require_native_frontier_layers"] = 1
    try:
        validate(wrong_frontier)
    except ValueError as error:
        assert "frontier" in str(error)
    else:
        raise AssertionError("a mismatched serving frontier passed artifact validation")
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
    framework_reference["engine_stats"][0]["ticketed_incremental_launches"] = 0
    validate(framework_reference)
    invalid_reference = copy.deepcopy(framework_reference)
    invalid_reference["engine_stats"][0][
        "stock_prefetched_external_attention_launches"
    ] = 0
    try:
        validate(invalid_reference)
    except ValueError as error:
        assert "timed numerical launches" in str(error)
    else:
        raise AssertionError("unfenced framework-reference evidence was accepted")
    mixed_consumer = copy.deepcopy(nta)
    framework_contract = framework_reference["engine_stats"][0]["consumer_contract"]
    mixed_consumer["engine_stats"][0].update(
        {
            "consumer_contracts": [
                mixed_consumer["engine_stats"][0]["consumer_contract"],
                framework_contract,
            ],
            "stock_prefetched_external_attention_launches": 1,
        }
    )
    validate(mixed_consumer)
    missing_mixed_contract = copy.deepcopy(mixed_consumer)
    missing_mixed_contract["engine_stats"][0]["consumer_contracts"] = [
        missing_mixed_contract["engine_stats"][0]["consumer_contract"]
    ]
    try:
        validate(missing_mixed_contract)
    except ValueError as error:
        assert "timed numerical launches" in str(error)
    else:
        raise AssertionError("mixed serving evidence hid its framework consumer")
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
    host_contract = resource_contract("host_staged")
    host_tier["engine_stats"][0].update(
        {
            "serving_tier": "host_staged",
            "tier_fallback": False,
            "tier_data_path": host_contract.steady_state_path,
            "resource_contract": host_contract.as_dict(),
            "tier_host_proxy_bytes": 0,
        }
    )
    validate(host_tier)
    invalid_host_tier = copy.deepcopy(host_tier)
    invalid_host_tier["engine_stats"][0]["tier_data_path"] = (
        "cuda_visible_cxl_direct"
    )
    try:
        validate(invalid_host_tier)
    except ValueError as error:
        assert "data path" in str(error)
    else:
        raise AssertionError("host-staged evidence reported the CXL data path")

    cxl_tier = copy.deepcopy(nta)
    cxl_contract = resource_contract("cxl_dax")
    cxl_tier["engine_stats"][0].update(
        {
            "serving_tier": "cxl_dax",
            "tier_fallback": False,
            "tier_data_path": cxl_contract.steady_state_path,
            "resource_contract": cxl_contract.as_dict(),
            "tier_host_proxy_bytes": 0,
            "tier_catalog_digest": "native-cxl-catalog",
            "tier_capabilities": {},
        }
    )
    try:
        validate(cxl_tier)
    except ValueError as error:
        assert "no SGLang FlashInfer numerical route" in str(error)
    else:
        raise AssertionError("native CXL evidence was mislabeled as SGLang serving")

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
    missing_initial_placement = copy.deepcopy(comparison)
    del missing_initial_placement["nta"]["initial_placement_proof"]
    try:
        validate(missing_initial_placement)
    except ValueError as error:
        assert "initial placement proof" in str(error)
    else:
        raise AssertionError("serving evidence without initial placement passed")
    inexact_initial_placement = copy.deepcopy(comparison)
    inexact_initial_placement["nta"]["initial_placement_proof"]["observations"][0][
        "host"
    ] = 31
    try:
        validate(inexact_initial_placement)
    except ValueError as error:
        assert "not exact" in str(error)
    else:
        raise AssertionError("inexact initial placement evidence passed")
    invalid_single = copy.deepcopy(stock)
    invalid_single["records"][0]["request_id"] = "duplicate"
    invalid_single["workload"] = {
        "manifest_digest": "manifest",
        "records_digest": "records",
        "demand_trace_digest": "demand",
        "tokenization_errors": 0,
        "token_input_adapter": "collision_free_content_block_tokens_v1",
        "token_input_identity_digest": "tokens",
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

    trial_artifact = {
        "path": "trials/trial-00.json",
        "sha256": "a" * 64,
        "revision": "test-revision",
        "machine_digest": "test-machine",
        "demand_digest": "test-demand",
    }
    passing_bars = {
        "registered_goodput": {
            "bar": 1.5,
            "geometric_mean": 1.6,
            "ci_floor": 1.1,
            "all_requests_have_token_level_itl": True,
            "passes": True,
        },
        "registered_joint_goodput": {
            "bar": 1.5,
            "geometric_mean": 1.6,
            "ci_floor": 1.1,
            "all_requests_have_token_level_itl": True,
            "passes": True,
        },
        "resident_p99_itl": {
            "bar": 1.05,
            "geometric_mean": 1.0,
            "passes": True,
        },
        "resident_p95_tpot": {
            "bar": 1.05,
            "geometric_mean": 1.0,
            "bootstrap_95_percent_ci_upper": 1.02,
            "passes": True,
        },
        "resident_output_throughput": {
            "bar": 0.95,
            "geometric_mean": 1.0,
            "bootstrap_95_percent_ci_lower": 0.98,
            "passes": True,
        },
        "output_throughput": {
            "bar": 0.95,
            "geometric_mean": 1.0,
            "bootstrap_95_percent_ci_lower": 0.98,
            "passes": True,
        },
        "outputs": {"passes": True},
        "mechanism": {"passes": True},
        "physical_bytes": {"passes": True},
        "provenance": {"passes": True},
    }
    qualification = {
        "schema": 2,
        "classification": "sglang-hicache-load-qualification",
        "mode": "formal",
        "trial_count": 10,
        "trial_artifacts": [copy.deepcopy(trial_artifact) for _ in range(10)],
        "revisions": ["test-revision"],
        "machine_digests": ["test-machine"],
        "demand_digests": ["test-demand"],
        "external_request_evidence": {
            "schema": 1,
            "minimum_external_observations_per_arm": 100,
            "minimum_distinct_external_request_ids_per_arm": 0,
            "per_arm": {
                arm: {
                    "external_observations": 100,
                    "distinct_external_request_ids": 10,
                    "observations_threshold_met": True,
                    "distinct_requests_threshold_met": True,
                }
                for arm in ("stock", "nta")
            },
            "paired_observation_counts_match": True,
            "paired_distinct_request_ids_match": True,
            "passes": True,
        },
        "bars": passing_bars,
        "all_bars_pass": True,
        "evidence_grade": "qualified",
        "qualified": True,
    }
    validate_trials(qualification)
    ineligible = copy.deepcopy(qualification)
    ineligible["bars"]["registered_goodput"][
        "all_requests_have_token_level_itl"
    ] = False
    try:
        validate_trials(ineligible)
    except ValueError as error:
        assert "lacking token-level ITL" in str(error)
    else:
        raise AssertionError("ITL-ineligible workload passed the goodput bar")
    overstated = copy.deepcopy(qualification)
    overstated["bars"]["registered_goodput"]["geometric_mean"] = 1.4
    overstated["bars"]["registered_goodput"]["passes"] = False
    overstated["all_bars_pass"] = False
    try:
        validate_trials(overstated)
    except ValueError as error:
        assert "evidence grade" in str(error)
    else:
        raise AssertionError("a failed performance bar remained qualified")
    diagnostic = copy.deepcopy(overstated)
    diagnostic["mode"] = "diagnostic"
    diagnostic["evidence_grade"] = "diagnostic"
    diagnostic["qualified"] = False
    validate_trials(diagnostic)
    with tempfile.TemporaryDirectory(prefix="nta-serving-artifact-") as directory:
        root = Path(directory)
        result = root / "serving.json"
        result.write_text(json.dumps(single()) + "\n", encoding="utf-8")
        workload_manifest, workload_rows = normalize(
            [
                {
                    "chat_id": "artifact-fixture",
                    "timestamp": 1.0,
                    "input_length": 32,
                    "output_length": 2,
                    "hash_ids": ["page-a", "page-b"],
                    "turn": 0,
                }
            ],
            arrival_mode="trace",
            synthesize_prompts=True,
        )
        source_manifest = root / "source-workload" / "source-manifest.json"
        source_records = root / "source-workload" / "unusual-records-name.jsonl"
        write_workload(
            source_manifest,
            source_records,
            workload_manifest,
            workload_rows,
        )
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
                "--workload-manifest",
                str(source_manifest),
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
        copied_manifest = bundle / "workload" / "manifest.json"
        copied_document = validate_workload(copied_manifest)
        assert copied_document["records_file"] == "unusual-records-name.jsonl"
        assert copied_manifest.read_bytes() == source_manifest.read_bytes()
        assert (bundle / "workload" / "unusual-records-name.jsonl").is_file()
    print("serving_report=pass")


if __name__ == "__main__":
    main()
