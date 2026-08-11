#!/usr/bin/env python3
"""The selected-serving report must prove that its mechanisms executed."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
from types import SimpleNamespace


ROOT = pathlib.Path(__file__).resolve().parents[2]
HARNESS = ROOT / "benchmarks" / "serving" / "SglangSelectedLoad.py"


def load_harness():
    spec = importlib.util.spec_from_file_location("nta_selected_load", HARNESS)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load selected-serving harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report(**updates):
    counters = {
        "tiered_claims": 2,
        "tiered_device_compaction_launches": 72,
        "tiered_bounded_cache_launches": 72,
        "tiered_request_overlap_layers": 36,
        "tiered_request_overlap_peer_requests": 36,
        "selected_compiler_launches": 72,
        "tiered_rows_copied": 4096,
        "tiered_rows_rehit": 2048,
        "tiered_tokens_total": 16384,
        "tiered_tokens_kept": 2048,
        "external_prefix_claims": 2,
        "external_dense_slots_avoided": 14336,
        "external_staging_slots": 2048,
        "external_admission_credit_rows": 16384,
        "external_dense_high_water_rows": 16384,
        "external_staging_high_water_rows": 2048,
        "hicache_fallback_batches": 0,
        "stock_attention_launches": 0,
    }
    counters.update(updates)
    return {"engine_stats": [counters]}


def reject(module, expected: str, **updates) -> None:
    try:
        module.validate_tiered_activation(report(**updates))
    except RuntimeError as error:
        if expected not in str(error):
            raise AssertionError(f"unexpected rejection: {error}") from error
    else:
        raise AssertionError(f"selected evidence accepted invalid {expected}")


def main() -> int:
    module = load_harness()
    counters, activation = module.validate_tiered_activation(report())
    assert counters["tiered_tokens_kept"] == 2048
    assert activation["compiler_attention_launches"] == 72
    compact_counters, compact_activation = module.validate_tiered_activation(
        report(
            tiered_request_overlap_layers=0,
            tiered_request_overlap_peer_requests=0,
            tiered_rows_rehit=0,
            tiered_selection_reuse_layers=612,
        ),
        require_overlap=False,
    )
    assert compact_counters["tiered_selection_reuse_layers"] == 612
    assert compact_activation["selection_reuse_layers"] == 612

    reject(module, "compiler_attention_launches", selected_compiler_launches=0)
    reject(module, "device_compaction_launches", tiered_device_compaction_launches=0)
    reject(module, "bounded_cache_launches", tiered_bounded_cache_launches=0)
    reject(module, "request_overlap_layers", tiered_request_overlap_layers=0)
    reject(
        module,
        "overlapped_peer_requests",
        tiered_request_overlap_peer_requests=0,
    )
    try:
        module.validate_tiered_activation(
            report(
                tiered_request_overlap_layers=0,
                tiered_request_overlap_peer_requests=0,
                tiered_rows_rehit=0,
                tiered_selection_reuse_layers=612,
            )
        )
    except RuntimeError as error:
        assert "request_overlap_layers" in str(error)
    else:
        raise AssertionError("overlap-required gate accepted compact evidence")
    reject(module, "rows_copied", tiered_rows_copied=0)
    reject(module, "rows_reused", tiered_rows_rehit=0)
    reject(module, "external_prefix_claims", external_prefix_claims=0)
    reject(module, "dense_slots_avoided", external_dense_slots_avoided=0)
    reject(module, "staging_slots", external_staging_slots=0)
    reject(module, "admission_credit_rows", external_admission_credit_rows=0)
    reject(module, "dense_high_water_rows", external_dense_high_water_rows=0)
    reject(module, "staging_high_water_rows", external_staging_high_water_rows=0)
    reject(
        module,
        "reduce live KV allocation",
        external_staging_high_water_rows=16384,
    )
    reject(module, "fallback", hicache_fallback_batches=1)
    reject(module, "compiler-generated", stock_attention_launches=1)
    reject(module, "selective attention", tiered_tokens_kept=16384)
    with tempfile.TemporaryDirectory() as temporary:
        quality_path = pathlib.Path(temporary) / "quality.json"
        quality_path.write_text(
            json.dumps(
                {
                    "classification": "quest-attention-mass-recall",
                    "model": str(ROOT),
                    "prompt_tokens": 16384,
                    "page_tokens": 16,
                    "aggregate": {
                        "quest_recall_at_32": {
                            "mean": 0.8134,
                            "min_layer": 0.4394,
                        },
                        "oracle_recall_at_32": {
                            "mean": 0.8612,
                            "min_layer": 0.6180,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        quality = module.selector_quality_gate(
            SimpleNamespace(
                selector_quality_report=quality_path,
                model=ROOT,
                external_tokens=16384,
                selected_budget=32,
                selected_page_tokens=16,
                min_selector_mean_recall=0.80,
                min_selector_min_layer_recall=0.40,
                max_selector_oracle_gap=0.05,
            )
        )
        assert quality["quest_mean_recall"] == 0.8134
        try:
            module.selector_quality_gate(
                SimpleNamespace(
                    selector_quality_report=quality_path,
                    model=ROOT,
                    external_tokens=16384,
                    selected_budget=32,
                    selected_page_tokens=16,
                    min_selector_mean_recall=0.90,
                    min_selector_min_layer_recall=None,
                    max_selector_oracle_gap=None,
                )
            )
        except RuntimeError as error:
            assert "mean recall" in str(error)
        else:
            raise AssertionError("quality gate accepted low selector recall")
        try:
            module.selector_quality_gate(
                SimpleNamespace(
                    selector_quality_report=quality_path,
                    model=ROOT,
                    external_tokens=16384,
                    selected_budget=64,
                    selected_page_tokens=16,
                    min_selector_mean_recall=None,
                    min_selector_min_layer_recall=None,
                    max_selector_oracle_gap=None,
                )
            )
        except RuntimeError as error:
            assert "budget 64" in str(error)
        else:
            raise AssertionError("quality gate accepted wrong budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
