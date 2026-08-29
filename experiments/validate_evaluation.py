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
        document.get("schema") != 1
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
    arms = document.get("arms", [])
    if [arm.get("id") for arm in arms] != list(ARMS):
        raise ValueError("evaluation contract must define A0-A3 in order")
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
    causal_pairs = document.get("causal_pairs")
    actual_pairs = (
        tuple((pair.get("numerator"), pair.get("denominator")) for pair in causal_pairs)
        if isinstance(causal_pairs, list)
        else ()
    )
    if actual_pairs != expected_pairs:
        raise ValueError(
            "evaluation contract must include every executable causal boundary"
        )
    scenarios = document.get("workload_scenario_contract")
    if (
        not isinstance(scenarios, dict)
        or scenarios.get("minimum_distinct_scenarios", 0) < 6
        or scenarios.get("free_form_stratum_labels") is not False
        or scenarios.get("same_scenario_required_within_pair") is not True
        or set(scenarios.get("identity", ()))
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
