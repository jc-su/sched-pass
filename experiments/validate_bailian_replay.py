#!/usr/bin/env python3
"""Validate natural Bailian replay reports and paired comparisons."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .validate_serving_report import (
        _finite,
        _require,
        _validate_byte_accounting,
        _validate_consumer_contract,
        _validate_environment,
        _validate_finite_window_accounting,
        _validate_tier_provenance,
    )
except ImportError:  # pragma: no cover - direct script execution
    from validate_serving_report import (
        _finite,
        _require,
        _validate_byte_accounting,
        _validate_consumer_contract,
        _validate_environment,
        _validate_finite_window_accounting,
        _validate_tier_provenance,
    )


def _records(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = report.get("records")
    _require(isinstance(records, list) and records, "natural replay has no records")
    for index, record in enumerate(records):
        _require(isinstance(record, dict), f"natural replay record {index} is invalid")
        for field in (
            "source_index",
            "source_request_id",
            "source_input_tokens",
            "source_output_tokens",
            "replay_output_tokens",
            "replayable_prefix_tokens",
            "input_tokens",
            "completion_tokens",
            "arrival_offset_seconds",
            "submitted_offset_seconds",
            "finished_offset_seconds",
            "ttft_seconds",
            "tpot_seconds",
            "p99_itl_seconds",
            "itl_sample_count",
            "inter_token_seconds",
            "token_timestamps_exact",
            "token_timestamp_source",
            "host_cached_tokens",
            "device_cached_tokens",
            "observed_cache_state",
            "text_sha256",
        ):
            _require(field in record, f"natural replay record {index} lacks {field}")
        for field in (
            "source_index",
            "source_input_tokens",
            "source_output_tokens",
            "replay_output_tokens",
            "replayable_prefix_tokens",
            "input_tokens",
            "completion_tokens",
            "itl_sample_count",
            "host_cached_tokens",
            "device_cached_tokens",
        ):
            value = record[field]
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                f"natural replay record {index} has invalid {field}",
            )
        _require(
            record["source_input_tokens"] + 1 == record["input_tokens"],
            f"natural replay record {index} changed source input geometry",
        )
        _require(
            record["replay_output_tokens"] == record["completion_tokens"] > 0,
            f"natural replay record {index} changed output demand",
        )
        observed = record["host_cached_tokens"] + record["device_cached_tokens"]
        _require(
            observed <= record["replayable_prefix_tokens"],
            f"natural replay record {index} manufactured prefix reuse",
        )
        expected_state = (
            "device_and_host"
            if record["host_cached_tokens"] and record["device_cached_tokens"]
            else "host"
            if record["host_cached_tokens"]
            else "device"
            if record["device_cached_tokens"]
            else "cold"
        )
        _require(
            record["observed_cache_state"] == expected_state,
            f"natural replay record {index} cache state is inconsistent",
        )
        for field in (
            "arrival_offset_seconds",
            "submitted_offset_seconds",
            "finished_offset_seconds",
            "ttft_seconds",
            "tpot_seconds",
            "p99_itl_seconds",
        ):
            _require(
                _finite(record[field], f"record {index} {field}") >= 0,
                f"natural replay record {index} has negative {field}",
            )
        _require(
            record["arrival_offset_seconds"] <= record["submitted_offset_seconds"]
            <= record["finished_offset_seconds"],
            f"natural replay record {index} timestamps are not ordered",
        )
        intervals = record["inter_token_seconds"]
        _require(
            isinstance(intervals, list)
            and record["itl_sample_count"]
            == max(0, record["completion_tokens"] - 1)
            == len(intervals),
            f"natural replay record {index} has invalid token intervals",
        )
        _require(
            record["token_timestamps_exact"] is True
            and record["token_timestamp_source"]
            == "sglang_stream_interval_1_completion_delta",
            f"natural replay record {index} lacks exact token timestamps",
        )
        _require(
            isinstance(record["text_sha256"], str) and record["text_sha256"],
            f"natural replay record {index} has no output digest",
        )
    return records


def _validate_workload(report: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> None:
    workload = report.get("workload")
    _require(isinstance(workload, dict), "natural replay lacks workload provenance")
    _require(
        workload.get("selection") == "contiguous_source_window"
        and workload.get("cache_state_source") == "observed_during_engine_replay"
        and workload.get("production_cache_state_claim") is False,
        "natural replay workload makes an invalid cache-state claim",
    )
    for field in (
        "manifest_digest",
        "records_digest",
        "source_demand_trace_digest",
        "selected_demand_trace_digest",
    ):
        _require(
            isinstance(workload.get(field), str) and bool(workload[field]),
            f"natural replay workload lacks {field}",
        )
    encoding = workload.get("token_encoding")
    boundary = workload.get("cache_boundary")
    _require(
        isinstance(encoding, dict)
        and encoding.get("kind") == "collision_free_content_block_tokens_v1"
        and encoding.get("distinct_pages_diverge_at_first_token") is True
        and isinstance(encoding.get("identity_digest"), str)
        and bool(encoding["identity_digest"]),
        "natural replay lacks collision-free content identity",
    )
    _require(
        isinstance(boundary, dict)
        and boundary.get("kind") == "reserved_special_token_cache_boundary_v1"
        and boundary.get("source_prefix_identity_preserved") is True
        and boundary.get("synthetic_output_alias_prevented") is True,
        "natural replay cache boundary is not exact",
    )
    opportunity = workload.get("mechanism_opportunity")
    _require(
        isinstance(opportunity, dict)
        and opportunity.get("selection_uses_measured_performance") is False
        and opportunity.get("observed_tier_placement_required") is True,
        "natural replay opportunity selection is outcome-dependent",
    )
    axes = workload.get("measured_axes")
    _require(isinstance(axes, dict), "natural replay lacks measured demand axes")
    for name in ("input_tokens", "output_tokens", "uncached_query_rows"):
        _require(
            isinstance(axes.get(name), dict) and axes[name].get("heterogeneous") is True,
            f"natural replay workload is not heterogeneous in {name}",
        )
    expected_indices = list(
        range(int(workload["measured_start"]), int(workload["measured_end"]))
    )
    actual_indices = [int(record["source_index"]) for record in records]
    _require(
        actual_indices == expected_indices,
        "natural replay records are not the selected contiguous source window",
    )


def _validate_native_dispatch(report: Mapping[str, Any], *, nta: bool) -> None:
    dispatch = report.get("native_dispatch")
    _require(
        isinstance(dispatch, dict),
        "natural replay lacks native-dispatch evidence",
    )
    _require(
        dispatch.get("definition")
        == "native_numerical_dispatch_layers_per_external_batch",
        "natural replay mislabels native dispatch as readiness evidence",
    )
    histogram_value = dispatch.get("histogram")
    _require(
        isinstance(histogram_value, dict),
        "native-dispatch histogram is invalid",
    )
    histogram: dict[int, int] = {}
    for depth, count in histogram_value.items():
        _require(
            isinstance(depth, str)
            and depth.isdigit()
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count >= 0,
            "native-dispatch histogram contains an invalid entry",
        )
        histogram[int(depth)] = count
    observations = int(dispatch.get("observations", -1))
    layers = dispatch.get("model_layer_count")
    _require(
        observations == sum(histogram.values()) and observations >= 0,
        "native-dispatch observations disagree with the histogram",
    )
    prefix_histogram_value = dispatch.get("monotone_prefix_histogram")
    dynamic_histogram_value = dispatch.get("dynamic_histogram")
    _require(
        isinstance(prefix_histogram_value, dict)
        and isinstance(dynamic_histogram_value, dict),
        "native-dispatch trajectory histograms are invalid",
    )
    prefix_histogram = {
        int(depth): int(count) for depth, count in prefix_histogram_value.items()
    }
    dynamic_histogram = {
        int(depth): int(count) for depth, count in dynamic_histogram_value.items()
    }
    _require(
        Counter(histogram) == Counter(prefix_histogram) + Counter(dynamic_histogram),
        "native-dispatch trajectory histograms disagree with the total",
    )
    prefix_observations = sum(prefix_histogram.values())
    dynamic_observations = sum(dynamic_histogram.values())
    _require(
        dispatch.get("monotone_prefix_observations") == prefix_observations
        and dispatch.get("dynamic_observations") == dynamic_observations,
        "native-dispatch trajectory counters are inconsistent",
    )
    expected_monotone_fraction = (
        prefix_observations / observations if observations else None
    )
    _require(
        dispatch.get("monotone_prefix_fraction") == expected_monotone_fraction,
        "native-dispatch monotone-prefix fraction is inconsistent",
    )
    if nta and observations:
        _require(
            isinstance(layers, int)
            and not isinstance(layers, bool)
            and layers > 0
            and all(0 <= depth <= layers for depth in histogram),
            "NTA native dispatch exceeds model geometry",
        )
    native_layers = sum(depth * count for depth, count in histogram.items())
    mixed = (
        sum(count for depth, count in histogram.items() if 0 < depth < layers)
        if isinstance(layers, int) and layers > 0
        else 0
    )
    expected_fraction = (
        native_layers / (observations * layers)
        if observations and isinstance(layers, int) and layers > 0
        else None
    )
    _require(
        dispatch.get("native_layer_observations") == native_layers
        and dispatch.get("mixed_dispatch_observations") == mixed
        and dispatch.get("framework_only_observations") == histogram.get(0, 0)
        and dispatch.get("native_only_observations")
        == (histogram.get(layers, 0) if isinstance(layers, int) else 0),
        "native-dispatch derived counters are inconsistent",
    )
    if expected_fraction is None:
        _require(
            dispatch.get("native_layer_fraction") is None,
            "empty native dispatch has a fabricated layer fraction",
        )
    else:
        _require(
            math.isclose(
                float(dispatch.get("native_layer_fraction", math.nan)),
                expected_fraction,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "native-dispatch layer fraction is inconsistent",
        )


def _validate_progressive_consumer(
    report: Mapping[str, Any], *, nta: bool
) -> None:
    consumer = report.get("progressive_consumer")
    _require(
        isinstance(consumer, dict),
        "natural replay lacks progressive-consumer evidence",
    )
    _require(
        consumer.get("definition")
        == "layers_executed_by_ticketed_progressive_work_unit_consumer",
        "natural replay has an unknown progressive-consumer definition",
    )
    histogram_value = consumer.get("histogram")
    _require(
        isinstance(histogram_value, dict),
        "progressive-consumer histogram is invalid",
    )
    histogram: dict[int, int] = {}
    for layers, count in histogram_value.items():
        _require(
            isinstance(layers, str)
            and layers.isdigit()
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count >= 0,
            "progressive-consumer histogram contains an invalid entry",
        )
        histogram[int(layers)] = count
    observations = int(consumer.get("observations", -1))
    model_layers = consumer.get("model_layer_count")
    _require(
        observations == sum(histogram.values()) and observations >= 0,
        "progressive-consumer observations disagree with the histogram",
    )
    if nta and observations:
        _require(
            isinstance(model_layers, int)
            and not isinstance(model_layers, bool)
            and model_layers > 0
            and all(0 <= layers <= model_layers for layers in histogram),
            "NTA progressive-consumer coverage exceeds model geometry",
        )
    layer_observations = sum(
        layers * count for layers, count in histogram.items()
    )
    active = sum(count for layers, count in histogram.items() if layers > 0)
    expected_fraction = (
        layer_observations / (observations * model_layers)
        if observations and isinstance(model_layers, int) and model_layers > 0
        else None
    )
    _require(
        consumer.get("layer_observations") == layer_observations
        and consumer.get("active_observations") == active
        and consumer.get("inactive_observations") == histogram.get(0, 0),
        "progressive-consumer derived counters are inconsistent",
    )
    if expected_fraction is None:
        _require(
            consumer.get("layer_fraction") is None,
            "empty progressive-consumer evidence fabricates a layer fraction",
        )
    else:
        _require(
            math.isclose(
                float(consumer.get("layer_fraction", math.nan)),
                expected_fraction,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "progressive-consumer layer fraction is inconsistent",
        )


def _validate_prefetch_arrival_readiness(report: Mapping[str, Any]) -> None:
    readiness = report.get("prefetch_arrival_readiness")
    _require(
        isinstance(readiness, dict),
        "natural replay lacks prefetch arrival-readiness evidence",
    )
    _require(
        readiness.get("definition")
        == "cuda_event_order_at_proactive_prefetch_attention_wait",
        "natural replay has an unknown arrival-readiness definition",
    )
    _require(
        readiness.get("status") in {"profiled", "not_profiled"},
        "natural replay has an invalid arrival-readiness status",
    )
    arrivals = int(readiness.get("arrivals", -1))
    ready = int(readiness.get("ready_at_arrival", -1))
    not_ready = int(readiness.get("not_ready_at_arrival", -1))
    material = int(readiness.get("materially_stalled_arrivals", -1))
    stall_ms = _finite(readiness.get("stall_gpu_ms"), "prefetch arrival stall")
    _require(
        min(arrivals, ready, not_ready, material) >= 0
        and ready + not_ready == arrivals
        and material <= not_ready
        and stall_ms >= 0.0,
        "natural replay arrival-readiness counters are inconsistent",
    )
    _require(
        math.isclose(
            _finite(
                readiness.get("material_stall_threshold_ms"),
                "prefetch material-stall threshold",
            ),
            0.01,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "natural replay changed its material-stall threshold",
    )
    if readiness["status"] == "not_profiled":
        _require(
            arrivals == ready == not_ready == material == 0 and stall_ms == 0.0,
            "unprofiled natural replay fabricated arrival-readiness evidence",
        )


def _validate_single(report: dict[str, Any], *, require_formal: bool) -> dict[str, Any]:
    _require(report.get("schema") == 1, "unsupported natural replay schema")
    _require(
        report.get("classification") == "sglang-bailian-natural-replay",
        "report is not a natural Bailian replay",
    )
    _require(
        isinstance(report.get("revision"), str)
        and bool(report["revision"])
        and isinstance(report.get("machine"), dict)
        and bool(report["machine"]),
        "natural replay lacks revision or machine provenance",
    )
    _require(
        report.get("demand_semantics") == "exact_content_block_prefix_replay"
        and report.get("cache_policy") == "natural_observed_no_forced_placement"
        and report.get("placement_proven") is False
        and report.get("cache_state_observed") is True,
        "natural replay misstates demand or placement semantics",
    )
    _validate_environment(report, require_complete=require_formal)
    _require(
        report.get("verification_failures") == 0
        and report.get("prefix_fidelity_violations") == 0,
        "natural replay has correctness failures",
    )
    correctness = report.get("correctness")
    _require(
        isinstance(correctness, dict)
        and correctness.get("verification_failures") == 0
        and correctness.get("prefix_fidelity_violations") == 0
        and correctness.get("source_prefix_identity_preserved") is True
        and correctness.get("generated_text_sha256")
        == report.get("generated_text_sha256"),
        "natural replay correctness contract is incomplete",
    )
    records = _records(report)
    _validate_workload(report, records)
    for field in (
        "request_throughput",
        "output_token_throughput",
        "p50_ttft_seconds",
        "p95_ttft_seconds",
        "p99_ttft_seconds",
        "p50_tpot_seconds",
        "p95_tpot_seconds",
        "p99_tpot_seconds",
        "p99_itl_seconds",
    ):
        _require(_finite(report.get(field), field) >= 0, f"negative natural {field}")
    _require(
        report["p50_ttft_seconds"]
        <= report["p95_ttft_seconds"]
        <= report["p99_ttft_seconds"]
        and report["p50_tpot_seconds"]
        <= report["p95_tpot_seconds"]
        <= report["p99_tpot_seconds"],
        "natural replay latency percentiles are not monotonic",
    )
    goodput = report.get("slo_goodput")
    _require(isinstance(goodput, dict), "natural replay lacks SLO goodput")
    total = int(_finite(goodput.get("total_requests"), "goodput total requests"))
    qualified = int(
        _finite(goodput.get("qualified_requests"), "goodput qualified requests")
    )
    _require(
        total == len(records)
        and 0 <= qualified <= total
        and _finite(
            goodput.get("goodput_requests_per_second"), "goodput rate"
        )
        >= 0,
        "natural replay goodput is inconsistent",
    )
    _require(
        sum(int(record["itl_sample_count"]) for record in records) > 0,
        "natural replay ITL is vacuous",
    )
    warmup = report.get("warmup")
    expected_boundary = (
        "out_of_band_scheduler_control_rpc"
        if report.get("attention_backend") == "nta_flashinfer"
        else "drained_warmup_host_timer"
    )
    _require(
        isinstance(warmup, dict)
        and warmup.get("performance_excluded") is True
        and warmup.get("measurement_boundary") == expected_boundary
        and warmup.get("marker_perturbation") == "none",
        "natural replay measurement boundary perturbs model/cache state",
    )
    heterogeneity = report.get("heterogeneity")
    _require(
        isinstance(heterogeneity, dict)
        and int(heterogeneity.get("heterogeneous_axis_count", 0)) >= 3,
        "natural replay lacks request-set heterogeneity",
    )
    if require_formal:
        _require(
            int(heterogeneity.get("maximum_concurrent_requests", 0)) > 1,
            "formal natural replay never had concurrent requests",
        )
    nta = report.get("attention_backend") == "nta_flashinfer"
    engine_stats = report.get("engine_stats")
    _require(isinstance(engine_stats, list), "natural replay engine stats are invalid")
    if nta and require_formal:
        _require(engine_stats, "formal NTA natural replay lacks engine statistics")
    if not nta:
        _require(not engine_stats, "stock natural replay contains NTA engine stats")
    _validate_native_dispatch(report, nta=nta)
    _validate_progressive_consumer(report, nta=nta)
    _validate_prefetch_arrival_readiness(report)
    _validate_finite_window_accounting(report)
    _validate_byte_accounting(report)
    _validate_tier_provenance(report)
    _validate_consumer_contract(
        report, require_formal_execution=nta and require_formal
    )
    return report


def _identity(report: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    return [
        (
            record.get("source_index"),
            record.get("source_request_id"),
            record.get("source_input_tokens"),
            record.get("replay_output_tokens"),
            record.get("arrival_offset_seconds"),
            record.get("replayable_prefix_tokens"),
        )
        for record in report.get("records", ())
    ]


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def validate(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("classification") == "sglang-bailian-natural-replay":
        return _validate_single(report, require_formal=False)
    _require(
        report.get("classification")
        == "sglang-bailian-natural-replay-comparison",
        "unsupported Bailian replay classification",
    )
    stock = report.get("stock")
    nta = report.get("nta")
    _require(
        isinstance(stock, dict) and isinstance(nta, dict),
        "natural comparison lacks paired arms",
    )
    _validate_single(stock, require_formal=True)
    _validate_single(nta, require_formal=True)
    _require(
        report.get("outputs_diverge") is False
        and stock.get("generated_text_sha256") == nta.get("generated_text_sha256"),
        "natural replay paired outputs diverge",
    )
    _require(
        report.get("request_demand_matched") is True and _identity(stock) == _identity(nta),
        "natural replay paired demand differs",
    )
    for field in (
        "manifest_digest",
        "records_digest",
        "selected_demand_trace_digest",
    ):
        _require(
            stock["workload"].get(field) == nta["workload"].get(field),
            f"natural replay paired {field} differs",
        )
    recomputed = {
        "output_throughput_ratio": _ratio(
            float(nta["output_token_throughput"]),
            float(stock["output_token_throughput"]),
        ),
        "request_throughput_ratio": _ratio(
            float(nta["request_throughput"]), float(stock["request_throughput"])
        ),
        "goodput_ratio": _ratio(
            float(nta["slo_goodput"]["goodput_requests_per_second"]),
            float(stock["slo_goodput"]["goodput_requests_per_second"]),
        ),
        "p95_ttft_ratio": _ratio(
            float(nta["p95_ttft_seconds"]), float(stock["p95_ttft_seconds"])
        ),
        "p95_tpot_ratio": _ratio(
            float(nta["p95_tpot_seconds"]), float(stock["p95_tpot_seconds"])
        ),
        "p99_itl_ratio": _ratio(
            float(nta["p99_itl_seconds"]), float(stock["p99_itl_seconds"])
        ),
    }
    for field, expected in recomputed.items():
        _require(
            math.isclose(
                float(report.get(field, math.nan)),
                expected,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ),
            f"natural replay comparison has an invalid {field}",
        )
    _require(
        report.get("native_dispatch") == nta.get("native_dispatch")
        and report.get("progressive_consumer") == nta.get("progressive_consumer")
        and report.get("prefetch_arrival_readiness")
        == nta.get("prefetch_arrival_readiness"),
        "natural comparison mechanism coverage differs from its NTA arm",
    )
    activation = report.get("mechanism_activation")
    if int(report.get("nta_host_requests", 0)) > 0:
        _require(
            isinstance(activation, dict)
            and activation.get("fallback_batches") == 0
            and activation.get("external_attention_accounted") is True,
            "natural replay host demand lacks a clean NTA mechanism",
        )
    else:
        _require(
            activation is None,
            "natural replay fabricated mechanism activation without NTA host demand",
        )
    eligibility = report.get("claim_eligibility")
    _require(
        isinstance(eligibility, dict)
        and isinstance(eligibility.get("matched_causal_serving"), bool)
        and isinstance(eligibility.get("blockers"), list)
        and eligibility["matched_causal_serving"] == (not eligibility["blockers"]),
        "natural replay claim eligibility is inconsistent",
    )
    return report


def validate_formal_arm(report: dict[str, Any]) -> dict[str, Any]:
    """Validate one non-degenerate natural-replay causal arm."""

    _require(
        report.get("classification") == "sglang-bailian-natural-replay",
        "formal natural replay must be one arm, not a nested comparison",
    )
    _validate_single(report, require_formal=True)
    _require(report.get("dirty") is False, "formal natural replay used a dirty revision")
    records = report.get("records", [])
    _require(
        bool(records)
        and all(int(record.get("completion_tokens", 0)) >= 2 for record in records),
        "formal natural replay has vacuous one-token TPOT/ITL measurements",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    validate(json.loads(args.report.resolve().read_text(encoding="utf-8")))
    print("bailian_replay=valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
