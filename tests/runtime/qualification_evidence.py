#!/usr/bin/env python3
"""Ensure release gates reject evidence for the superseded mechanism."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_qualifier():
    path = ROOT / "scripts" / "qualify-release.py"
    spec = importlib.util.spec_from_file_location("nta_qualify_release", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load release qualifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    qualifier = load_qualifier()
    with tempfile.TemporaryDirectory() as temporary:
        evidence = pathlib.Path(temporary)
        old = {"schema": 2, "revision": "revision"}
        (evidence / "production-evidence.json").write_text(
            json.dumps(old), encoding="utf-8"
        )
        production = qualifier.production_checks(evidence, "revision")
        assert len(production) == 1 and not production[0]["passed"]
        assert "schema 3" in production[0]["detail"]

        current = {"schema": 3, "revision": "revision", "artifacts": []}
        (evidence / "production-evidence.json").write_text(
            json.dumps(current), encoding="utf-8"
        )
        production = qualifier.production_checks(evidence, "revision")
        production_names = {item["name"] for item in production}
        assert {
            "serving integration",
            "serving graph path",
            "serving performance bounds",
            "serving tier coverage",
        }.issubset(production_names)
        assert not any(item["passed"] for item in production)

        original_verify_artifacts = qualifier.verify_artifacts
        qualifier.verify_artifacts = lambda *_args: (True, "test artifacts")
        current["serving"] = {
            "engine": "sglang",
            "mechanism_integrated": True,
            "mechanism_mode": "request_aware_dual_form",
            "correctness": True,
            "transfer_verification": True,
            "all_attention_layers_executed": True,
            "baseline_and_mechanism": True,
            "zero_fallback": True,
            "all_attention_transformed": True,
            "bounded_hbm_tier_streaming": True,
            "generation_safe_request_completion": True,
            "jit_cache_primed": True,
            "compiler_contract_verified": True,
            "compiler_plan_verified": True,
            "verified_operator_modules": 4,
            "verified_operator_pairs": 2,
            "verified_operator_plan_pairs": 2,
            "transformed_direct_launches": 10,
            "ticketed_incremental_launches": 5,
            "stock_attention_launches": 0,
            "matched_cache_and_admission": True,
            "decode_cuda_graph_replay": True,
            "paged_prefill_integrated": True,
            "demand_operator_graph_replay": True,
            "demand_graph_captures": 5,
            "demand_graph_replays": 10,
        }
        (evidence / "production-evidence.json").write_text(
            json.dumps(current), encoding="utf-8"
        )
        production = qualifier.production_checks(evidence, "revision")
        integration = next(
            item for item in production if item["name"] == "serving integration"
        )
        assert integration["passed"] is True
        current["serving"]["compiler_contract_verified"] = False
        (evidence / "production-evidence.json").write_text(
            json.dumps(current), encoding="utf-8"
        )
        production = qualifier.production_checks(evidence, "revision")
        integration = next(
            item for item in production if item["name"] == "serving integration"
        )
        assert integration["passed"] is False
        current["serving"]["compiler_contract_verified"] = True
        current["serving"]["compiler_plan_verified"] = False
        (evidence / "production-evidence.json").write_text(
            json.dumps(current), encoding="utf-8"
        )
        production = qualifier.production_checks(evidence, "revision")
        integration = next(
            item for item in production if item["name"] == "serving integration"
        )
        assert integration["passed"] is False
        current["serving"]["compiler_plan_verified"] = True
        current["serving"]["stock_attention_launches"] = 1
        (evidence / "production-evidence.json").write_text(
            json.dumps(current), encoding="utf-8"
        )
        production = qualifier.production_checks(evidence, "revision")
        integration = next(
            item for item in production if item["name"] == "serving integration"
        )
        assert integration["passed"] is False
        qualifier.verify_artifacts = original_verify_artifacts

        (evidence / "osdi-evidence.json").write_text(
            json.dumps(current), encoding="utf-8"
        )
        osdi = qualifier.osdi_checks(evidence, "revision")
        osdi_names = {item["name"] for item in osdi}
        assert {
            "measured dense opportunity",
            "compiler-generated forms",
            "real FlashInfer incremental execution",
            "heterogeneous serving workload",
            "unified scheduler and engine feedback",
            "real GPU-selected FlashInfer acquisition",
            "mechanism performance bounds",
        }.issubset(osdi_names)
        assert not any(item["passed"] for item in osdi)

        qualifier.verify_artifacts = lambda *_args: (True, "test artifacts")
        current["incremental_execution"] = {
            "decode": True,
            "paged_prefill": True,
            "canonical_flashinfer_attention": True,
            "custom_attention_kernel": False,
            "partial_before_last_arrival": True,
            "generation_safe_request_completion": True,
            "stock_output_parity": True,
            "demand_cuda_graph_replay": True,
            "demand_graph_families": ["decode", "paged_prefill"],
            "demand_graph_captures": 10,
            "demand_graph_replays": 20,
            "all_attention_transformed": True,
            "transformed_direct_launches": 10,
            "ticketed_incremental_launches": 10,
            "stock_attention_launches": 0,
            "bounded_hbm_staging_reduction": 4.0,
            "speedup_over_atomic_promotion": 1.15,
            "speedup_95ci_lower": 1.01,
        }
        (evidence / "osdi-evidence.json").write_text(
            json.dumps(current), encoding="utf-8"
        )
        osdi = qualifier.osdi_checks(evidence, "revision")
        incremental = next(
            item
            for item in osdi
            if item["name"] == "real FlashInfer incremental execution"
        )
        assert incremental["passed"] is True
        current["incremental_execution"]["custom_attention_kernel"] = True
        (evidence / "osdi-evidence.json").write_text(
            json.dumps(current), encoding="utf-8"
        )
        osdi = qualifier.osdi_checks(evidence, "revision")
        incremental = next(
            item
            for item in osdi
            if item["name"] == "real FlashInfer incremental execution"
        )
        assert incremental["passed"] is False
        current["incremental_execution"]["custom_attention_kernel"] = False

        current["sparse_flashinfer"] = {
            "gpu_selected_pages": True,
            "nta_hot_path_host_identity_round_trips": 0,
            "real_flashinfer_selector": True,
            "real_flashinfer_attention": True,
            "all_policy_attention_transformed": True,
            "paired_operator_contract_verified": True,
            "stock_output_parity": True,
            "candidate_sweep_points": 5,
            "selectivity_crossover_measured": True,
            "peak_speedup_over_overfetch": 2.0,
            "peak_speedup_bootstrap_95_percent_ci": [1.9, 2.1],
            "maximum_online_policy_regret": 1.01,
            "policy_regret_definition": "same_trial_chosen_over_best",
            "candidate_retained_baseline": True,
            "minimum_cold_indexed_latency_ratio_to_candidate_retained": 3.0,
            "maximum_cold_indexed_latency_ratio_to_candidate_retained": 4.0,
            "no_selectivity_policy_mode": "bulk",
            "no_selectivity_speedup": 0.99,
            "no_selectivity_forced_indexed_throughput_ratio": 0.61,
            "maximum_regret_to_offline_oracle": 2.0,
        }
        (evidence / "osdi-evidence.json").write_text(
            json.dumps(current), encoding="utf-8"
        )
        osdi = qualifier.osdi_checks(evidence, "revision")
        sparse = next(
            item
            for item in osdi
            if item["name"] == "real GPU-selected FlashInfer acquisition"
        )
        assert sparse["passed"] is True

        current["sparse_flashinfer"]["maximum_regret_to_offline_oracle"] = 2.01
        (evidence / "osdi-evidence.json").write_text(
            json.dumps(current), encoding="utf-8"
        )
        osdi = qualifier.osdi_checks(evidence, "revision")
        sparse = next(
            item
            for item in osdi
            if item["name"] == "real GPU-selected FlashInfer acquisition"
        )
        assert sparse["passed"] is False

    print("qualification_evidence=pass")


if __name__ == "__main__":
    main()
