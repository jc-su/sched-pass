#!/usr/bin/env python3
"""Validate serving evidence gates independently from an SGLang installation."""

from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import sys
import time


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
                "verified_dual_form_operator_plans": 1,
                "operator_contracts": [{"abi": 1}],
                "transport_program_loaded": True,
                "transport_program_sha256": "a" * 64,
                "transport_contract": {
                    "family": "generic",
                    "form": "incremental",
                    "tier_mask": 63,
                },
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

    calibrated_auto = report(compact_ctas=1, canonical_ctas=1)
    calibrated_auto["engine_stats"][0].update(
        {
            "serving_tier": "host_staged",
            "host_execution_mode": "auto",
            "incremental_setup_calibrated": True,
            "incremental_calibration_probes_remaining": 0,
            "consumer_policy_calibration": {"last_shape_closed": True},
        }
    )
    activation = module.require_clean_mechanism(calibrated_auto)
    assert activation["auto_calibration_applicable"]
    assert activation["auto_calibration_closed"]
    calibrated_auto["engine_stats"][0].update(
        {
            "calibration_profile_status": "loaded_read_only",
            "calibration_profile_read_only": True,
            "calibration_profile_loaded_samples": 17,
            "calibration_profile_sha256": "b" * 64,
            "calibration_profile_save_count": 0,
            "calibration_profile_checkpoint_failures": 0,
            "calibration_profile_deferred_checkpoints": 0,
        }
    )
    activation = module.require_clean_mechanism(
        calibrated_auto, require_read_only_calibration_profile=True
    )
    assert activation["calibration_profile_digests"] == ["b" * 64]
    writable_profile = report(compact_ctas=1, canonical_ctas=1)
    writable_profile["engine_stats"][0].update(calibrated_auto["engine_stats"][0])
    writable_profile["engine_stats"][0]["calibration_profile_read_only"] = False
    try:
        module.require_clean_mechanism(
            writable_profile, require_read_only_calibration_profile=True
        )
    except RuntimeError as error:
        assert "profile is writable" in str(error)
    else:
        raise AssertionError("a writable calibration profile passed the timed gate")
    uncalibrated_auto = report(compact_ctas=1, canonical_ctas=1)
    uncalibrated_auto["engine_stats"][0].update(
        {
            "serving_tier": "host_staged",
            "host_execution_mode": "auto",
            "incremental_setup_calibrated": False,
            "incremental_calibration_probes_remaining": 1,
            "consumer_policy_calibration": {"last_shape_closed": False},
        }
    )
    try:
        module.require_clean_mechanism(uncalibrated_auto)
    except RuntimeError as error:
        assert "calibration closed" in str(error)
    else:
        raise AssertionError("uncalibrated host AUTO trial passed the evidence gate")

    timed_probe = report(compact_ctas=1, canonical_ctas=1)
    timed_probe["engine_stats"][0].update(
        {
            "serving_tier": "host_staged",
            "host_execution_mode": "auto",
            "incremental_setup_calibrated": True,
            "incremental_calibration_probes_remaining": 0,
            "consumer_policy_calibration": {"last_shape_closed": True},
            "consumer_policy_probe_leases": 1,
        }
    )
    try:
        module.require_clean_mechanism(timed_probe)
    except RuntimeError as error:
        assert "timed window for calibration" in str(error)
    else:
        raise AssertionError("timed AUTO calibration passed the evidence gate")

    # A complete exact prefetch is allowed to use the framework consumer after
    # the acquisition fence.  It must still be counted as external work, but
    # it is not falsely reported as a transformed NTA attention launch.
    prefetched = report(compact_ctas=1, canonical_ctas=1)
    prefetched["engine_stats"][0].update(
        {
            "consumer_contract": {
                "schema": 1,
                "engine": "sglang",
                "backend": "nta_flashinfer",
                "kind": "framework_reference",
                "exact_demand": True,
                "typed_work_plan": False,
                "native_submission": False,
                "numerical_consumer": True,
                "engine_version": "0.5.16",
            },
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
            "verified_operator_modules": 0,
            "verified_dual_form_operator_plans": 0,
            "operator_contracts": [],
        }
    )
    activation = module.require_clean_mechanism(prefetched)
    assert activation["external_attention_transformed"] is False
    assert activation["external_attention_stock_consumer"] is True
    assert activation["external_attention_accounted"] is True
    assert not activation["native_work_unit_active"]
    assert not activation["heterogeneous_work_unit_active"]
    assert activation["transport_only"]
    assert activation["transport_program_verified"]
    assert activation["framework_preacquired_verified"]
    assert not activation["compiler_verification_required"]
    assert activation["verification_domain"] == "framework_exact_preacquired"

    # One timed forward may use the native consumer at an unresolved frontier
    # and stock FlashInfer after later layers become fully preacquired.  The
    # aggregate contract remains native, while the path set proves both
    # numerical consumers instead of forcing one label to describe both.
    mixed = report(compact_ctas=1, canonical_ctas=2)
    native_contract = dict(mixed["engine_stats"][0]["consumer_contract"])
    framework_contract = dict(prefetched["engine_stats"][0]["consumer_contract"])
    mixed["engine_stats"][0].update(
        {
            "consumer_contracts": [native_contract, framework_contract],
            "stock_attention_launches": 1,
            "stock_prefetched_external_attention_launches": 1,
            "external_launches": 2,
            "native_external_attention_launches": 1,
            "prefetched_layers": 2,
            "demand_host_layers": 1,
            "host_acquisition_layers_consumed": 2,
            "prefill_launches": 2,
        }
    )
    activation = module.require_clean_mechanism(
        mixed, require_physical_compaction=True
    )
    assert activation["native_work_unit_active"]
    assert activation["external_attention_stock_consumer"]
    assert activation["framework_preacquired_verified"]
    missing_mixed_path = report(compact_ctas=1, canonical_ctas=2)
    missing_mixed_path["engine_stats"][0].update(mixed["engine_stats"][0])
    missing_mixed_path["engine_stats"][0]["consumer_contracts"] = [native_contract]
    try:
        module.require_clean_mechanism(missing_mixed_path)
    except RuntimeError as error:
        assert "framework contract" in str(error)
    else:
        raise AssertionError("mixed execution hid its stock numerical contract")

    missing_transport = report(compact_ctas=1, canonical_ctas=1)
    missing_transport["engine_stats"][0].pop("transport_contract")
    try:
        module.require_clean_mechanism(missing_transport)
    except RuntimeError as error:
        assert "transport program" in str(error)
    else:
        raise AssertionError("an unverified transport program passed the gate")

    missing_compiler = report(compact_ctas=1, canonical_ctas=1)
    missing_compiler["engine_stats"][0].update(
        {"verified_operator_modules": 0, "operator_contracts": []}
    )
    try:
        module.require_clean_mechanism(missing_compiler)
    except RuntimeError as error:
        assert "compiler contracts" in str(error)
    else:
        raise AssertionError("a native path without compiler proof passed the gate")

    serving = load_serving_module()
    closed_snapshot = {
        "scheduler": {
            "backend": "nta_flashinfer",
            "serving_tier": "host_staged",
            "host_execution_mode": "auto",
            "incremental_setup_calibrated": True,
            "incremental_calibration_probes_remaining": 0,
            "consumer_policy_calibration": {"last_shape_closed": True},
        }
    }
    serving._require_closed_auto_calibration(closed_snapshot)
    open_snapshot = {"scheduler": dict(closed_snapshot["scheduler"])}
    open_snapshot["scheduler"]["consumer_policy_calibration"] = {
        "last_shape_closed": False
    }
    try:
        serving._require_closed_auto_calibration(open_snapshot)
    except RuntimeError as error:
        assert "warmup ended before AUTO calibration closed" in str(error)
    else:
        raise AssertionError("an open consumer policy entered timed serving")
    assert serving._max_request_input_tokens(32_768, 18_000) == 17_992
    assert serving._max_request_input_tokens(16_000, 18_000) == 15_992
    assert serving._reusable_prefix_tokens((1, 2, 3), (1, 2, 3)) == 2
    assert serving._reusable_prefix_tokens((1, 2, 3), (1, 2, 3, 4)) == 3
    try:
        serving._reusable_prefix_tokens((1, 2), (1, 3, 4))
    except RuntimeError as error:
        assert "does not extend" in str(error)
    else:
        raise AssertionError("serving placement accepted a mismatched prefix")

    class _FlushResult:
        def __init__(self, success: bool, message: str = "") -> None:
            self.success = success
            self.message = message

    class _FlushManager:
        def __init__(self, result: _FlushResult) -> None:
            self.result = result
            self.timeout_seconds: float | None = None

        async def flush_cache(self, *, timeout_s: float) -> _FlushResult:
            self.timeout_seconds = timeout_s
            await asyncio.sleep(0)
            return self.result

    class _FlushLoop:
        @staticmethod
        def run_until_complete(awaitable):
            return asyncio.run(awaitable)

    class _FlushEngine:
        def __init__(self, result: _FlushResult) -> None:
            self.tokenizer_manager = _FlushManager(result)
            self.loop = _FlushLoop()

    flush_engine = _FlushEngine(_FlushResult(True))
    flush_wait = serving._flush_cache_when_idle(
        flush_engine,
        timeout_seconds=17.0,
        reason="exercise fixture placement",
    )
    assert flush_wait >= 0.0
    assert flush_engine.tokenizer_manager.timeout_seconds == 17.0
    failed_flush_engine = _FlushEngine(_FlushResult(False, "still busy"))
    try:
        serving._flush_cache_when_idle(
            failed_flush_engine,
            timeout_seconds=3.0,
            reason="exercise failed fixture placement",
        )
    except RuntimeError as error:
        assert "exercise failed fixture placement" in str(error)
        assert "still busy" in str(error)
    else:
        raise AssertionError("failed deferred cache flush passed the setup gate")
    try:
        serving._flush_cache_when_idle(
            object(), timeout_seconds=3.0, reason="exercise missing fixture contract"
        )
    except RuntimeError as error:
        assert "lacks the deferred idle-flush contract" in str(error)
    else:
        raise AssertionError("missing deferred cache-flush API passed the setup gate")

    class _StreamingEngine:
        def __init__(self, completion_counts: tuple[int, ...]) -> None:
            self._completion_counts = completion_counts

        async def async_generate(self, **kwargs):
            assert kwargs["sampling_params"]["stream_interval"] == 1

            async def events():
                for completion_tokens in self._completion_counts:
                    await asyncio.sleep(0)
                    yield {
                        "text": "x" * completion_tokens,
                        "meta_info": {
                            "completion_tokens": completion_tokens,
                            "cached_tokens_details": {"device": 3, "host": 0},
                        },
                    }

            return events()

    streamed = asyncio.run(
        serving._stream_request(
            _StreamingEngine((1, 2, 3)),
            (1, 2, 3, 4),
            {
                "temperature": 0,
                "max_new_tokens": 3,
                "ignore_eos": True,
                "stream_interval": 1,
            },
            kind="fixture",
            index=0,
            request_id="fixture-stream",
            gate=None,
            first_token_event=None,
            offset_seconds=0.0,
            load_start_seconds=time.perf_counter(),
        )
    )
    assert streamed["completion_tokens"] == 3
    assert streamed["itl_sample_count"] == 2
    assert len(streamed["inter_token_seconds"]) == 2
    assert streamed["token_timestamps_exact"] is True

    async def exercise_first_token_barrier() -> None:
        barrier = serving._FirstTokenBarrier(2)
        released = asyncio.create_task(barrier.wait())
        await asyncio.sleep(0)
        assert not released.done()
        barrier.set()
        await asyncio.sleep(0)
        assert not released.done()
        barrier.set()
        await released
        # A duplicate stream notification cannot underflow or re-open the
        # already-completed state transition.
        barrier.set()

    asyncio.run(exercise_first_token_barrier())

    async def exercise_gated_arrival_accounting() -> None:
        barrier = serving._FirstTokenBarrier(1)
        started = time.perf_counter()
        request = asyncio.create_task(
            serving._stream_request(
                _StreamingEngine((1,)),
                (1, 2, 3, 4),
                {
                    "temperature": 0,
                    "max_new_tokens": 1,
                    "ignore_eos": True,
                    "stream_interval": 1,
                },
                kind="fixture",
                index=0,
                request_id="fixture-gated-arrival",
                gate=barrier,
                first_token_event=None,
                offset_seconds=0.0,
                load_start_seconds=started,
            )
        )
        await asyncio.sleep(0.01)
        barrier.set()
        record = await request
        assert record["workload_gate_wait_seconds"] >= 0.005
        assert abs(
            record["arrival_offset_seconds"]
            - record["workload_gate_wait_seconds"]
        ) < 1e-6
        assert abs(
            record["admission_delay_seconds"]
            - (
                record["submitted_offset_seconds"]
                - record["arrival_offset_seconds"]
            )
        ) < 1e-6
        assert record["admission_delay_seconds"] < record[
            "workload_gate_wait_seconds"
        ]

    asyncio.run(exercise_gated_arrival_accounting())
    try:
        serving._FirstTokenBarrier(0)
    except ValueError as error:
        assert "resident parties" in str(error)
    else:
        raise AssertionError("empty synthetic resident barrier was accepted")
    try:
        asyncio.run(
            serving._stream_request(
                _StreamingEngine((1, 3)),
                (1, 2, 3, 4),
                {
                    "temperature": 0,
                    "max_new_tokens": 3,
                    "ignore_eos": True,
                    "stream_interval": 1,
                },
                kind="fixture",
                index=0,
                request_id="fixture-coalesced",
                gate=None,
                first_token_event=None,
                offset_seconds=0.0,
                load_start_seconds=time.perf_counter(),
            )
        )
    except RuntimeError as error:
        assert "coalesced multiple tokens" in str(error)
    else:
        raise AssertionError("coalesced stream events were accepted as token ITL")

    sys.path.insert(0, str(ROOT / "python"))
    from nta_runtime.engines.sglang_lifecycle import SglangForwardLifecycle
    from nta_runtime.engines.sglang_state import (
        SglangForwardEpoch,
        SglangForwardPlan,
    )

    dispatch_stats = {
        "native_dispatch_prefix_observations": 0,
        "native_dispatch_nonprefix_batches": 0,
        "progressive_consumer_batch_observations": 0,
        "progressive_consumer_layers": 0,
    }
    frontier = SglangForwardEpoch(
        plan=SglangForwardPlan(
            bindings=(),
            semantic_plans={},
            pending_host_load=None,
        ),
    )
    frontier_lifecycle = SglangForwardLifecycle(
        request_adapter=object(),
        hicache=object(),
        granularity=object(),
        model_layer_count=4,
        stats=dispatch_stats,
    )
    frontier_lifecycle.activate(frontier)
    for layer, native in enumerate((True, True, False, False)):
        frontier_lifecycle.record_external_dispatch(
            frontier,
            layer,
            native_dispatch=native,
            progressive_consumer=native,
            final_layer=layer == 3,
        )
    assert dispatch_stats["native_dispatch_prefix_observations"] == 1
    assert dispatch_stats["native_dispatch_prefix_layers_2_batches"] == 1
    assert dispatch_stats["progressive_consumer_batch_observations"] == 1
    assert dispatch_stats["progressive_consumer_layers"] == 2
    assert dispatch_stats["progressive_consumer_layers_2_batches"] == 1
    nonprefix = SglangForwardEpoch(
        plan=SglangForwardPlan(
            bindings=(),
            semantic_plans={},
            pending_host_load=None,
        ),
    )
    nonprefix_lifecycle = SglangForwardLifecycle(
        request_adapter=object(),
        hicache=object(),
        granularity=object(),
        model_layer_count=4,
        stats=dispatch_stats,
    )
    nonprefix_lifecycle.activate(nonprefix)
    for layer, native in enumerate((False, True, False, True)):
        nonprefix_lifecycle.record_external_dispatch(
            nonprefix,
            layer,
            native_dispatch=native,
            progressive_consumer=native,
            final_layer=layer == 3,
        )
    assert dispatch_stats["native_dispatch_nonprefix_batches"] == 1
    assert dispatch_stats["native_dispatch_nonprefix_layers_2_batches"] == 1
    assert dispatch_stats["native_dispatch_prefix_observations"] == 1

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

    materialized, materialization_forbidden = (
        serving._exact_prefix_materialization_inputs(
            _Tokenizer(),
            (prefix, (21, 22)),
            (measured_input, (21, 22, 99, 14)),
        )
    )
    assert tuple(len(value) for value in materialized) == (len(prefix) + 1, 3)
    assert materialized[0][: len(prefix)] == prefix
    assert materialized[1][:2] == (21, 22)
    assert materialized[0][len(prefix)] not in {14, 99}
    assert materialized[1][2] not in {14, 99}
    assert materialized[0][len(prefix)] != materialized[1][2]
    assert {14, 99}.issubset(materialization_forbidden)

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
        "host_bound_after_full_ready_batches": 7,
        "plan_uploads": 1,
        "semantic_wrapper_plan_builds": 1,
        "admission_considered_batches": 8,
        "hybrid_parallel_waves": 59,
        "indexed_range_fastpath_layers": 140,
        "unqueued_host_discovery_layers": 140,
        "queued_feasible_edf_layers": 4,
        "stream_ordered_retirement_layers": 280,
        "stream_ordered_retirement_launches": 8,
        "stream_ordered_retirement_batches": 8,
        "profiled_stream_retirement_operator_layers": 280,
        "profiled_stream_retirement_operator_launches": 8,
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
        "host_mover_profiled_sm_bytes": 64,
        "host_mover_profiled_sm_gpu_ms": 1.0,
        "demand_graph_paged_prefill_captures": 2,
        "demand_graph_paged_prefill_replays": 5,
        "schedule_bound_acquisition_batches": 8,
        "forward_mixed_count": 4.0,
        "forward_mixed_ms_total": 20.0,
        "forward_mixed_ms_max": 7.0,
        "native_dispatch_prefix_observations": 2,
        "native_dispatch_nonprefix_batches": 0,
        "native_dispatch_prefix_layers_1_batches": 2,
        "progressive_consumer_batch_observations": 2,
        "progressive_consumer_batches": 2,
        "progressive_consumer_layers": 2,
        "progressive_consumer_layers_1_batches": 2,
        "cumulative_counter_fields": [
            "host_device_bulk_batches",
            "forward_lifecycle_aborts",
        ],
        "host_device_bulk_batches": 4,
        "forward_lifecycle_aborts": 2,
        "deadline_frontier_modeled_ready_layers": 70,
        "deadline_frontier_modeled_stock_dispatches": 70,
        "deadline_frontier_first_missed_layer_sum": 140,
        "profiled_attention_stall_by_layer_ms": {"0": 1.0, "1": 2.0},
        "profiled_attention_max_stall_gpu_ms": 2.0,
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
        "host_bound_after_full_ready_batches": 8,
        "admission_considered_batches": 9,
        "hybrid_parallel_waves": 69,
        "indexed_range_fastpath_layers": 175,
        "unqueued_host_discovery_layers": 175,
        "queued_feasible_edf_layers": 5,
        "stream_ordered_retirement_layers": 316,
        "stream_ordered_retirement_launches": 9,
        "stream_ordered_retirement_batches": 9,
        "profiled_stream_retirement_operator_layers": 316,
        "profiled_stream_retirement_operator_launches": 9,
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
        "host_mover_profiled_sm_bytes": 96,
        "host_mover_profiled_sm_gpu_ms": 1.5,
        "demand_graph_paged_prefill_captures": 3,
        "demand_graph_paged_prefill_replays": 9,
        "schedule_bound_acquisition_batches": 11,
        "forward_mixed_count": 6.0,
        "forward_mixed_ms_total": 31.0,
        "forward_mixed_ms_max": 8.0,
        "native_dispatch_prefix_observations": 3,
        "native_dispatch_prefix_layers_1_batches": 3,
        "progressive_consumer_batch_observations": 3,
        "progressive_consumer_batches": 3,
        "progressive_consumer_layers": 3,
        "progressive_consumer_layers_1_batches": 3,
        "host_device_bulk_batches": 7,
        "forward_lifecycle_aborts": 3,
        "deadline_frontier_modeled_ready_layers": 105,
        "deadline_frontier_modeled_stock_dispatches": 105,
        "deadline_frontier_first_missed_layer_sum": 210,
        "profiled_attention_stall_by_layer_ms": {"0": 1.5, "1": 2.0},
        "profiled_attention_max_stall_gpu_ms": 2.0,
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
    assert measured["host_bound_after_full_ready_batches"] == 1
    assert measured["admission_considered_batches"] == 1
    assert measured["hybrid_parallel_waves"] == 10
    assert measured["indexed_range_fastpath_layers"] == 35
    assert measured["unqueued_host_discovery_layers"] == 35
    assert measured["queued_feasible_edf_layers"] == 1
    assert measured["stream_ordered_retirement_layers"] == 36
    assert measured["stream_ordered_retirement_launches"] == 1
    assert measured["stream_ordered_retirement_batches"] == 1
    assert measured["progressive_consumer_batches"] == 1
    assert measured["profiled_stream_retirement_operator_layers"] == 36
    assert measured["profiled_stream_retirement_operator_launches"] == 1
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
    assert measured["host_mover_profiled_sm_bytes"] == 32
    assert measured["host_mover_profiled_sm_gpu_ms"] == 0.5
    assert measured["demand_graph_paged_prefill_captures"] == 1
    assert measured["demand_graph_paged_prefill_replays"] == 4
    assert measured["schedule_bound_acquisition_batches"] == 3
    assert measured["forward_mixed_count"] == 2.0
    assert measured["forward_mixed_ms_total"] == 11.0
    assert measured["forward_mixed_ms_max"] == 8.0
    assert measured["native_dispatch_prefix_observations"] == 1
    assert measured["native_dispatch_nonprefix_batches"] == 0
    assert measured["native_dispatch_prefix_layers_1_batches"] == 1
    assert measured["progressive_consumer_batch_observations"] == 1
    assert measured["progressive_consumer_layers"] == 1
    assert measured["progressive_consumer_layers_1_batches"] == 1
    assert measured["host_device_bulk_batches"] == 3
    assert measured["forward_lifecycle_aborts"] == 1
    assert measured["deadline_frontier_modeled_ready_layers"] == 35
    assert measured["deadline_frontier_modeled_stock_dispatches"] == 35
    assert measured["deadline_frontier_first_missed_layer_sum"] == 70
    assert measured["profiled_attention_stall_by_layer_ms"] == {"0": 0.5}
    assert "profiled_attention_max_stall_gpu_ms" not in measured
    expected_gib_per_second = 1_189_011_456 / (1 << 30) / 0.028
    assert (
        abs(
            measured["profiled_pipeline_transfer_gib_per_second"]
            - expected_gib_per_second
        )
        < 1e-9
    )
    assert measured["consumer_contract"]["kind"] == "framework_reference"
    assert [
        contract["kind"] for contract in measured["consumer_contracts"]
    ] == ["framework_reference"]
    assert [
        contract["kind"]
        for contract in serving._measured_consumer_contracts(
            {
                "ticketed_incremental_launches": 1,
                "stock_prefetched_external_attention_launches": 35,
            }
        )
    ] == ["native_work_unit", "framework_reference"]
    dispatch = serving._execution_dispatch([measured])
    assert dispatch["kind"] == "stock_direct"
    assert dispatch["plan_uploads"] == 0
    assert dispatch["semantic_wrapper_plan_builds"] == 0
    invalid_dispatch = dict(measured, transformed_direct_launches=1)
    try:
        serving._execution_dispatch([invalid_dispatch])
    except RuntimeError as error:
        assert "direct-only" in str(error)
    else:
        raise AssertionError("direct-only execution accepted a native launch")
    scheduled = {
        **measured,
        "host_direct_batches": 0,
        "host_incremental_batches": 1,
        "host_acquisition_jobs_submitted": 36,
        "stock_prefetched_external_attention_launches": 36,
        "transformed_direct_launches": 0,
        "ticketed_incremental_launches": 0,
    }
    dispatch = serving._execution_dispatch([scheduled])
    assert dispatch["kind"] == "scheduled_preacquired"
    selected, physical_bytes, status = serving._engine_byte_accounting([measured])
    assert selected == physical_bytes == 301_916_160
    assert status == "exact_engine_transfer_counters"


if __name__ == "__main__":
    main()
