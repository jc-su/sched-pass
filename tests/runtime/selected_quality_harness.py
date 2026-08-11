#!/usr/bin/env python3
"""Checks for the selected-quality comparator's evidence normalization."""

from __future__ import annotations

import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[2]
HARNESS_DIR = ROOT / "benchmarks" / "serving"
HARNESS = HARNESS_DIR / "SglangSelectedQuality.py"


def load_harness():
    sys.path.insert(0, str(HARNESS_DIR))
    spec = importlib.util.spec_from_file_location("nta_selected_quality", HARNESS)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load selected-quality harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_harness()
    report = {
        "engine_stats": [
            {
                "tiered_claims": 1,
                "tiered_device_compaction_launches": 36,
                "tiered_bounded_cache_launches": 36,
                "tiered_rows_copied": 0,
                "tiered_rows_copied_released": 18072,
                "tiered_rows_rehit": 0,
                "tiered_rows_rehit_released": 0,
                "tiered_selection_reuse_layers": 432,
                "tiered_tokens_total": 12472920,
                "tiered_tokens_kept": 1000000,
                "external_prefix_claims": 1,
                "external_dense_slots_avoided": 15094,
                "external_staging_slots": 1024,
                "external_admission_credit_rows": 15606,
                "external_dense_high_water_rows": 15606,
                "external_staging_high_water_rows": 512,
                "selected_compiler_launches": 468,
                "hicache_fallback_batches": 0,
                "stock_attention_launches": 0,
            }
        ]
    }
    normalized = module.normalize_finished_tiered_counters(report)
    counters = normalized["engine_stats"][0]
    assert counters["tiered_rows_copied"] == 18072
    _, activation = module.validate_tiered_activation(
        normalized,
        require_overlap=False,
    )
    assert activation["rows_copied"] == 18072
    assert activation["selection_reuse_layers"] == 432
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
