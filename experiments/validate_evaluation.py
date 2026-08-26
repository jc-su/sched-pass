#!/usr/bin/env python3
"""Validate the machine-readable OSDI evaluation contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


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
        "hbm": ("hbm", "gpu_hbm_load"),
        "host_mem": ("host_staged", "host_indexed_copy"),
        "nvme": ("nvme", "gpu_owned_nvme_to_hbm"),
        "dax": ("cxl_dax", "cuda_visible_cxl_direct"),
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
    if [arm.get("id") for arm in arms] != [f"B{index}" for index in range(7)]:
        raise ValueError("evaluation contract must define B0-B6 in order")
    if not all(arm.get("exact_demand") is True for arm in arms):
        raise ValueError("every arm must use exact demand")
    expected_pairs = tuple((f"B{index}", f"B{index - 1}") for index in range(1, 7)) + (
        ("B3", "B1"),
        ("B5", "B3"),
    )
    causal_pairs = document.get("causal_pairs")
    actual_pairs = (
        tuple((pair.get("numerator"), pair.get("denominator")) for pair in causal_pairs)
        if isinstance(causal_pairs, list)
        else ()
    )
    if actual_pairs != expected_pairs:
        raise ValueError(
            "evaluation contract must include adjacent and decisive cross-boundary pairs"
        )
    strata = document.get("strata", {})
    for name in (
        "request_state",
        "granularity",
        "load_ratio",
        "availability_skew",
        "staging_pressure",
        "arrival",
    ):
        if not strata.get(name):
            raise ValueError(f"missing evaluation stratum {name}")
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
