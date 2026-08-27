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


def load_serving_module():
    directory = ROOT / "benchmarks" / "serving"
    path = directory / "SglangHiCacheLoad.py"
    spec = importlib.util.spec_from_file_location("nta_sglang_load", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the SGLang load harness")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(directory))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def report(*, compact_ctas: int, canonical_ctas: int) -> dict:
    return {
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
                "hicache_fallback_batches": 0,
                "hicache_external_batches": 1,
                "transformed_direct_launches": 0,
                "stock_attention_launches": 0,
                "stock_resident_attention_launches": 0,
                "stock_prefetched_external_attention_launches": 0,
                "external_launches": 1,
                "native_external_attention_launches": 1,
                "prefetched_layers": 0,
                "demand_host_layers": 1,
                "pipeline_host_enabled": False,
                "ticketed_incremental_launches": 1,
                "decode_launches": 0,
                "prefill_launches": 1,
                "verified_operator_modules": 1,
                "verified_operator_pairs": 1,
                "operator_contracts": [{"abi": 1}],
                "mixed_dependency_layers": 1,
                "compact_resume_launches": 1,
                "compact_resume_cta_bound": compact_ctas,
                "canonical_resume_cta_bound": canonical_ctas,
                "demand_graph_warmups": 1,
                "demand_graph_captures": 1,
                "demand_graph_replays": 1,
                "execution_protocol": "late_bound",
            }
        ]
    }


def main() -> None:
    module = load_comparison_module()
    one_cta = report(compact_ctas=1, canonical_ctas=1)
    activation = module.require_clean_mechanism(one_cta, require_demand_graph=True)
    assert activation["compact_resume_cta_ratio"] == 1.0
    assert activation["native_work_unit_active"]
    assert activation["heterogeneous_work_unit_active"]
    assert not activation["transport_only"]

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

    physical = report(compact_ctas=1, canonical_ctas=1)
    physical["engine_stats"][0].update(
        {
            "serving_tier": "nvme",
            "tier_external_layers": 1,
            "demand_host_layers": 0,
            "compact_resume_launches": 0,
            "compact_resume_cta_bound": 0,
            "canonical_resume_cta_bound": 0,
            "ticketed_incremental_launches": 1,
        }
    )
    activation = module.require_clean_mechanism(
        physical,
        require_physical_compaction=True,
    )
    assert activation["external_attention_transformed"]
    assert not activation["physical_compaction_applicable"]

    conventional = report(compact_ctas=1, canonical_ctas=1)
    conventional["engine_stats"][0].update(
        {"execution_protocol": "conventional", "mixed_dependency_layers": 0}
    )
    activation = module.require_clean_mechanism(conventional)
    assert activation["execution_protocol"] == "conventional"

    # A complete exact prefetch is allowed to use the framework consumer after
    # the acquisition fence.  It must still be counted as external work, but
    # it is not falsely reported as a transformed NTA attention launch.
    prefetched = report(compact_ctas=1, canonical_ctas=1)
    prefetched["engine_stats"][0].update(
        {
            "stock_attention_launches": 2,
            "stock_resident_attention_launches": 1,
            "stock_prefetched_external_attention_launches": 1,
            "external_launches": 1,
            "native_external_attention_launches": 0,
            "ticketed_incremental_launches": 0,
            "prefetched_layers": 1,
            "demand_host_layers": 0,
            "decode_launches": 2,
            "prefill_launches": 0,
            "mixed_dependency_layers": 0,
        }
    )
    activation = module.require_clean_mechanism(prefetched)
    assert activation["external_attention_transformed"] is False
    assert activation["external_attention_stock_consumer"] is True
    assert activation["external_attention_accounted"] is True
    assert not activation["native_work_unit_active"]
    assert not activation["heterogeneous_work_unit_active"]
    assert activation["transport_only"]

    serving = load_serving_module()

    class _Tokenizer:
        vocab_size = 256
        all_special_ids: tuple[int, ...] = ()

        @staticmethod
        def encode(text: str, *, add_special_tokens: bool) -> list[int]:
            assert not add_special_tokens
            return [ord(value) for value in text]

        @staticmethod
        def decode(values, *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens
            return "".join(chr(int(value)) for value in values)

        def __len__(self) -> int:
            return self.vocab_size

    prefix = (11, 12, 13)
    measured_input = prefix + (14, 15)
    forbidden = {14}
    calibration = serving._exact_calibration_input(
        _Tokenizer(),
        prefix,
        measured_input,
        label="calibration-a",
        forbidden_first_tokens=forbidden,
    )
    second_calibration = serving._exact_calibration_input(
        _Tokenizer(),
        prefix,
        measured_input,
        label="calibration-b",
        forbidden_first_tokens=forbidden,
    )
    assert calibration[: len(prefix)] == second_calibration[: len(prefix)] == prefix
    assert len(calibration) == len(second_calibration) == len(measured_input)
    assert (
        len(
            {
                measured_input[len(prefix)],
                calibration[len(prefix)],
                second_calibration[len(prefix)],
            }
        )
        == 3
    )

    baseline = {
        "backend": "nta_flashinfer",
        "snapshot_unix_ns": 100,
        "verified_operator_modules": 2,
        "ticketed_incremental_launches": 1,
        "native_external_attention_launches": 1,
        "stock_prefetched_external_attention_launches": 287,
        "external_launches": 288,
        "prefetched_layers": 287,
        "demand_host_layers": 1,
        "hicache_external_batches": 8,
        "host_direct_batches": 8,
        "host_incremental_batches": 0,
        "host_mixed_direct_batches": 2,
        "host_typed_mixed_batches": 2,
        "host_bound_after_full_publication_batches": 7,
        "plan_uploads": 1,
        "work_topology_builds": 1,
        "admission_considered_batches": 8,
        "admission_lead_layers": 4,
        "hybrid_parallel_waves": 59,
        "phase_enqueue_cpu_ns": 90_000,
        "profiled_pipeline_transfer_batches": 15,
        "profiled_pipeline_transfer_bytes": 7_133_669_376,
        "profiled_pipeline_transfer_gpu_ms": 168.0,
        "profiled_pipeline_transfer_gib_per_second": 39.5,
        "tier_selected_leases": 2,
        "tier_selected_rows": 16_380,
        "tier_selected_bytes": 603_832_320,
        "tier_candidate_bytes": 603_832_320,
        "prefetched_host_bytes": 603_832_320,
        "typed_acquisition_batches": 2,
        "typed_acquisition_rows": 16_380,
        "typed_acquisition_work_items": 252,
    }
    final = {
        **baseline,
        "snapshot_unix_ns": 200,
        "stock_prefetched_external_attention_launches": 323,
        "external_launches": 324,
        "prefetched_layers": 323,
        "hicache_external_batches": 9,
        "host_direct_batches": 9,
        "host_mixed_direct_batches": 3,
        "host_typed_mixed_batches": 3,
        "host_bound_after_full_publication_batches": 8,
        "admission_considered_batches": 9,
        "hybrid_parallel_waves": 69,
        "phase_enqueue_cpu_ns": 117_000,
        "profiled_pipeline_transfer_batches": 16,
        "profiled_pipeline_transfer_bytes": 8_322_680_832,
        "profiled_pipeline_transfer_gpu_ms": 196.0,
        "profiled_pipeline_transfer_gib_per_second": 39.0,
        "tier_selected_leases": 3,
        "tier_selected_rows": 24_570,
        "tier_selected_bytes": 905_748_480,
        "tier_candidate_bytes": 905_748_480,
        "prefetched_host_bytes": 905_748_480,
        "typed_acquisition_batches": 3,
        "typed_acquisition_rows": 24_570,
        "typed_acquisition_work_items": 378,
    }
    measured = serving._measurement_delta(final, baseline)
    assert measured["measurement_scope"] == "timed_load_delta"
    assert measured["verified_operator_modules"] == 2
    assert measured["native_external_attention_launches"] == 0
    assert measured["stock_prefetched_external_attention_launches"] == 36
    assert measured["external_launches"] == 36
    assert measured["prefetched_layers"] == 36
    assert measured["hicache_external_batches"] == 1
    assert measured["host_direct_batches"] == 1
    assert measured["host_incremental_batches"] == 0
    assert measured["host_mixed_direct_batches"] == 1
    assert measured["host_typed_mixed_batches"] == 1
    assert measured["host_bound_after_full_publication_batches"] == 1
    assert measured["admission_considered_batches"] == 1
    assert measured["admission_lead_layers"] == 4
    assert measured["hybrid_parallel_waves"] == 10
    assert measured["phase_enqueue_cpu_ns"] == 27_000
    assert measured["profiled_pipeline_transfer_batches"] == 1
    assert measured["profiled_pipeline_transfer_bytes"] == 1_189_011_456
    assert measured["profiled_pipeline_transfer_gpu_ms"] == 28.0
    assert measured["tier_selected_leases"] == 1
    assert measured["tier_selected_rows"] == 8_190
    assert measured["tier_selected_bytes"] == 301_916_160
    assert measured["tier_candidate_bytes"] == 301_916_160
    assert measured["prefetched_host_bytes"] == 301_916_160
    assert measured["typed_acquisition_batches"] == 1
    assert measured["typed_acquisition_rows"] == 8_190
    assert measured["typed_acquisition_work_items"] == 126
    expected_gib_per_second = 1_189_011_456 / (1 << 30) / 0.028
    assert (
        abs(
            measured["profiled_pipeline_transfer_gib_per_second"]
            - expected_gib_per_second
        )
        < 1e-9
    )
    assert measured["consumer_contract"]["kind"] == "framework_reference"
    dispatch = serving._execution_dispatch([measured])
    assert dispatch["kind"] == "stock_direct"
    assert dispatch["plan_uploads"] == 0
    assert dispatch["work_topology_builds"] == 0
    invalid_dispatch = dict(measured, transformed_direct_launches=1)
    try:
        serving._execution_dispatch([invalid_dispatch])
    except RuntimeError as error:
        assert "direct-only" in str(error)
    else:
        raise AssertionError("direct-only execution accepted a native launch")
    selected, physical_bytes, status = serving._engine_byte_accounting([measured])
    assert selected == physical_bytes == 301_916_160
    assert status == "exact_engine_transfer_counters"


if __name__ == "__main__":
    main()
