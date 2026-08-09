#!/usr/bin/env python3
"""Validate serving evidence gates independently from an SGLang installation."""

from __future__ import annotations

import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_comparison_module():
    path = ROOT / "benchmarks" / "serving" / "CompareSglangHiCache.py"
    spec = importlib.util.spec_from_file_location("nta_compare_hicache", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the HiCache comparison harness")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def report(*, compact_ctas: int, canonical_ctas: int) -> dict:
    return {
        "engine_stats": [
            {
                "backend": "nta_flashinfer",
                "hicache_fallback_batches": 0,
                "hicache_claimed_batches": 1,
                "transformed_direct_launches": 0,
                "stock_attention_launches": 0,
                "external_launches": 1,
                "prefetched_layers": 0,
                "demand_host_layers": 1,
                "pipeline_host_enabled": False,
                "ticketed_incremental_launches": 1,
                "decode_launches": 0,
                "prefill_launches": 1,
                "verified_operator_modules": 1,
                "verified_operator_pairs": 1,
                "operator_contracts": [{"abi": 1}],
                "compact_resume_launches": 1,
                "compact_resume_cta_bound": compact_ctas,
                "canonical_resume_cta_bound": canonical_ctas,
                "demand_graph_warmups": 1,
                "demand_graph_captures": 1,
                "demand_graph_replays": 1,
            }
        ]
    }


def main() -> None:
    module = load_comparison_module()
    one_cta = report(compact_ctas=1, canonical_ctas=1)
    activation = module.require_clean_mechanism(
        one_cta, require_demand_graph=True
    )
    assert activation["compact_resume_cta_ratio"] == 1.0

    try:
        module.require_clean_mechanism(
            one_cta,
            require_demand_graph=True,
            require_physical_compaction=True,
        )
    except RuntimeError as error:
        assert "physically compact" in str(error)
    else:
        raise AssertionError("one-CTA evidence satisfied the compaction gate")

    compact = report(compact_ctas=1, canonical_ctas=2)
    activation = module.require_clean_mechanism(
        compact,
        require_demand_graph=True,
        require_physical_compaction=True,
    )
    assert activation["compact_resume_cta_ratio"] == 0.5


if __name__ == "__main__":
    main()
