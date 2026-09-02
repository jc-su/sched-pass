#!/usr/bin/env python3
"""Validate the machine-readable OSDI evaluation contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from nta_runtime.resource_contract import resource_contract  # noqa: E402
from experiments.mechanism_arms import (  # noqa: E402
    ARMS,
    ARM_DEFINITIONS,
    CAUSAL_PAIRS,
)


def validate(document: dict[str, Any]) -> None:
    if (
        document.get("schema") != 2
        or document.get("classification") != "nta-osdi-evaluation-contract"
    ):
        raise ValueError("unsupported evaluation manifest")
    if (
        document.get("selection_semantics") != "exact-demand-only"
        or document.get("approximate_attention_claim") is not False
    ):
        raise ValueError("evaluation contract must be exact and non-approximate")
    tier_ids = {tier.get("id") for tier in document.get("tiers", [])}
    if not {"hbm", "host_mem", "nvme", "dax"} <= tier_ids:
        raise ValueError("evaluation contract lacks HBM/host/NVMe/DAX tiers")
    expected_resources = {
        tier_id: (resource_kind, resource_contract(resource_kind).steady_state_path)
        for tier_id, resource_kind in {
            "hbm": "hbm",
            "host_mem": "host_staged",
            "nvme": "nvme",
            "dax": "cxl_dax",
        }.items()
    }
    tiers = {tier["id"]: tier for tier in document["tiers"]}
    for tier_id, (resource_kind, steady_state_path) in expected_resources.items():
        tier = tiers.get(tier_id)
        if not isinstance(tier, dict):
            raise ValueError(f"evaluation contract lacks tier object: {tier_id}")
        if tier.get("resource_kind") != resource_kind:
            raise ValueError(f"{tier_id} must name resource kind {resource_kind!r}")
        if tier.get("steady_state_path") != steady_state_path:
            raise ValueError(
                f"{tier_id} must name steady-state path {steady_state_path!r}"
            )
    if document.get("matched_baselines") != [
        {
            "id": "host_mapped",
            "resource_kind": "host_mapped",
            "steady_state_path": "gpu_mapped_host_load",
            "scope": "nvme_dma_destination",
            "serving_tier": False,
        }
    ]:
        raise ValueError(
            "evaluation contract must keep host-mapped as an explicit matched "
            "baseline, not an ambiguous host serving tier"
        )
    if document.get("correctness_role") != (
        "mandatory_validity_gate_not_research_question"
    ):
        raise ValueError("correctness must be a validity gate, not a research question")

    research_questions = document.get("research_questions")
    if not isinstance(research_questions, list) or len(research_questions) != 4:
        raise ValueError("evaluation contract must define exactly four research questions")
    expected_rq_roles = {
        "RQ1": "end_to_end_effectiveness",
        "RQ2": "mechanism_attribution",
        "RQ3": "opportunity_envelope",
        "RQ4": "deployment_cost_and_isolation",
    }
    actual_rq_roles = {
        rq.get("id"): rq.get("role")
        for rq in research_questions
        if isinstance(rq, dict)
    }
    if actual_rq_roles != expected_rq_roles:
        raise ValueError("evaluation contract must use the canonical four-RQ structure")
    if any(
        not isinstance(rq.get("metrics"), list) or not rq["metrics"]
        for rq in research_questions
    ):
        raise ValueError("every research question must declare measurable outcomes")

    systems = document.get("systems")
    if not isinstance(systems, list):
        raise ValueError("evaluation contract must declare compared systems")
    system_ids = {
        system.get("id") for system in systems if isinstance(system, dict)
    }
    required_systems = {
        "nta-full",
        "sglang-hicache-kernel",
        "vllm-lmcache",
        "gpu-only-reference",
    }
    if not required_systems <= system_ids:
        raise ValueError("evaluation contract lacks required current competitors")

    campaign = document.get("paper_campaign_contract")
    if not isinstance(campaign, dict):
        raise ValueError("evaluation contract lacks a paper campaign contract")
    minimums = {
        "minimum_models": 3,
        "minimum_model_families": 2,
        "minimum_long_context_models": 2,
        "minimum_workload_families": 4,
        "minimum_natural_trace_workloads": 2,
        "minimum_load_points_per_headline_curve": 4,
        "minimum_independent_requests_per_load_point": 1000,
        "minimum_independent_requests_for_p99": 10000,
        "minimum_repetitions": 5,
    }
    for field, lower_bound in minimums.items():
        value = campaign.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < lower_bound:
            raise ValueError(f"paper campaign field {field!r} must be >= {lower_bound}")
    if campaign.get("require_short_context_control") is not True:
        raise ValueError("paper campaign must include a short-context no-regression control")
    competitor_groups = campaign.get("required_competitor_groups")
    if not isinstance(competitor_groups, list) or {
        group.get("id") for group in competitor_groups if isinstance(group, dict)
    } != {"closest_upstream", "cross_framework", "non_hierarchical_reference"}:
        raise ValueError("paper campaign must define all required competitor groups")
    required_evidence = campaign.get("required_rq_evidence")
    if not isinstance(required_evidence, dict) or set(required_evidence) != set(
        expected_rq_roles
    ):
        raise ValueError("paper campaign must define evidence for every research question")
    if any(
        not isinstance(evidence, list) or not evidence
        for evidence in required_evidence.values()
    ):
        raise ValueError("paper campaign evidence sets must be non-empty")

    mechanism = document.get("mechanism_study")
    if not isinstance(mechanism, dict):
        raise ValueError("evaluation contract lacks the matched mechanism study")
    if mechanism.get("profile") != "mechanism-study":
        raise ValueError("causal arms must be labeled mechanism-study, not OSDI-complete")
    arms = mechanism.get("arms", [])
    if [arm.get("id") for arm in arms] != list(ARMS):
        raise ValueError("evaluation contract must define every canonical arm in order")
    if not all(arm.get("exact_demand") is True for arm in arms):
        raise ValueError("every arm must use exact demand")
    if any(
        arm.get("name") != ARM_DEFINITIONS[arm["id"]]["name"] for arm in arms
    ):
        raise ValueError("evaluation arm names diverge from the executable forms")
    expected_pairs = tuple(
        (numerator, denominator)
        for numerator, denominator, _role in CAUSAL_PAIRS
    )
    causal_pairs = mechanism.get("causal_pairs")
    actual_pairs = (
        tuple((pair.get("numerator"), pair.get("denominator")) for pair in causal_pairs)
        if isinstance(causal_pairs, list)
        else ()
    )
    if actual_pairs != expected_pairs:
        raise ValueError(
            "evaluation contract must include every executable causal boundary"
        )
    if (
        mechanism.get("minimum_distinct_scenarios", 0) < 6
        or mechanism.get("same_scenario_required_within_pair") is not True
        or set(mechanism.get("identity", ()))
        != {"manifest_sha256", "records_digest", "demand_trace_digest"}
    ):
        raise ValueError("evaluation contract lacks typed workload scenarios")
    observations = document.get("mechanism_observations")
    if (
        not isinstance(observations, dict)
        or observations.get("require_result_emitted_evidence") is not True
        or observations.get("ablation_or_continuous_axis_only") is not True
        or set(observations.get("orthogonal_axes", ()))
        != {
            "frontier_depth",
            "work_unit_granularity",
            "transport_engine",
            "tier",
            "batch_heterogeneity",
        }
    ):
        raise ValueError("evaluation contract misclassifies mechanism observations")
    statistics = document.get("statistical_protocol", {})
    if statistics.get("minimum_repetitions", 0) < 5 or not statistics.get(
        "randomized_paired_arm_order"
    ):
        raise ValueError("evaluation contract has no defensible paired-trial protocol")
    if statistics.get("replay_cycles_are_independent_samples") is not False:
        raise ValueError("replayed identities cannot be counted as independent samples")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=ROOT / "experiments" / "evaluation-manifest.json",
    )
    args = parser.parse_args()
    validate(json.loads(args.manifest.read_text(encoding="utf-8")))
    print("evaluation_manifest=valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
