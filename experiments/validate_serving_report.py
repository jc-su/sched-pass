#!/usr/bin/env python3
"""Validate structured serving evidence before it enters an artifact bundle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from nta_runtime.resource_contract import resource_contract

try:
    from .consumer_contract import validate_consumer_contract
    from .serving_metrics import (
        preregistered_goodput,
        relative_goodput,
        relative_thresholds,
        safe_ratio,
    )
    from .serving_path_evidence import (
        EXERCISED_PATHS,
        require_frontier_shape,
        summarize_transport_execution,
    )
    from .workload_heterogeneity import serving_batch_heterogeneity
except ImportError:  # pragma: no cover - direct script execution
    from consumer_contract import validate_consumer_contract
    from serving_metrics import (
        preregistered_goodput,
        relative_goodput,
        relative_thresholds,
        safe_ratio,
    )
    from serving_path_evidence import (
        EXERCISED_PATHS,
        require_frontier_shape,
        summarize_transport_execution,
    )
    from workload_heterogeneity import serving_batch_heterogeneity


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any, name: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"serving report has no finite {name}",
    )
    return float(value)


def _nonnegative_integer(value: Any, name: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"serving report has invalid {name}",
    )
    return value


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _close(actual: Any, expected: float, name: str) -> None:
    value = _finite(actual, name)
    _require(
        math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12),
        f"serving report {name} does not match request records",
    )


def _validate_derived_object(actual: Any, expected: dict[str, Any], name: str) -> None:
    """Require a derived metric object to match its request-record recomputation."""

    _require(isinstance(actual, dict), f"serving report has no {name} object")
    _require(
        set(actual) == set(expected),
        f"serving report {name} has an unexpected field set",
    )
    for field, expected_value in expected.items():
        value = actual[field]
        field_name = f"{name} {field}"
        if isinstance(expected_value, dict):
            _validate_derived_object(value, expected_value, field_name)
        elif isinstance(expected_value, int):
            _require(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value == expected_value,
                f"serving report {field_name} does not match request records",
            )
        elif isinstance(expected_value, float):
            _close(value, expected_value, field_name)
        else:  # pragma: no cover - metric schema is intentionally closed
            _require(value == expected_value, f"serving report has invalid {field_name}")


def _validate_ratio(actual: Any, expected: float | None, name: str) -> None:
    if expected is None:
        _require(
            actual is None,
            f"serving report {name} invents a finite ratio with a zero baseline",
        )
    else:
        _close(actual, expected, name)


def _validate_environment(report: dict[str, Any], *, require_complete: bool) -> None:
    """Validate run-level GPU occupancy evidence.

    A direct worker report may be used for debugging without the comparison
    driver's sampler. A formal paired result cannot: it must show at least one
    successful sample, zero sampler errors, and no foreign compute PID.
    """

    fields = {
        "cotenant_gpu_samples",
        "gpu_samples",
        "gpu_sampling_errors",
        "gpu_sampling_complete",
        "cotenant_pids_seen",
    }
    if not fields & report.keys():
        _require(
            not require_complete,
            "formal serving evidence has no GPU environment sampler",
        )
        return
    _require(
        fields <= report.keys(),
        "serving environment sampler evidence is incomplete",
    )
    foreign_samples = _nonnegative_integer(
        report["cotenant_gpu_samples"], "co-tenant GPU sample count"
    )
    samples = _nonnegative_integer(report["gpu_samples"], "GPU sample count")
    errors = _nonnegative_integer(
        report["gpu_sampling_errors"], "GPU sampler error count"
    )
    pids = report["cotenant_pids_seen"]
    _require(
        isinstance(pids, list)
        and all(
            isinstance(pid, int) and not isinstance(pid, bool) and pid > 0
            for pid in pids
        )
        and len(set(pids)) == len(pids),
        "serving report has invalid co-tenant PID evidence",
    )
    _require(foreign_samples <= samples, "co-tenant samples exceed total samples")
    _require(
        (foreign_samples == 0) == (len(pids) == 0),
        "co-tenant sample and PID evidence disagree",
    )
    if require_complete:
        _require(samples > 0, "formal serving evidence has no successful GPU sample")
        _require(errors == 0, "formal serving evidence lost GPU samples")
        _require(
            report["gpu_sampling_complete"] is True,
            "formal serving evidence has incomplete GPU sampling",
        )
        _require(
            foreign_samples == 0,
            "formal serving evidence was contaminated by a foreign GPU process",
        )


def _validate_finite_window_accounting(report: dict[str, Any]) -> None:
    accounting = report.get("finite_window_accounting")
    _require(
        isinstance(accounting, dict),
        "serving report has no finite-window client timestamp accounting",
    )
    _require(
        accounting.get("method") == "finite_window_arrival_departure_accounting",
        "serving report uses an unknown finite-window accounting method",
    )
    _require(
        accounting.get("interpretation")
        == "descriptive_client_timestamp_accounting",
        "serving report mislabels finite-window accounting as queueing evidence",
    )
    for field in (
        "arrival_rate_per_second",
        "completion_rate_per_second",
        "mean_in_system",
        "mean_system_time_seconds",
        "occupancy_area_request_seconds",
        "sum_residence_seconds",
    ):
        _finite(accounting.get(field), f"finite-window accounting {field}")


def _validate_workload(report: dict[str, Any]) -> None:
    workload = report.get("workload")
    if workload is None:
        return
    _require(isinstance(workload, dict), "serving workload provenance is not an object")
    for field in ("manifest_digest", "records_digest", "demand_trace_digest"):
        value = workload.get(field)
        _require(isinstance(value, str) and value, f"serving workload lacks {field}")
    _require(
        workload.get("tokenization_errors") == 0,
        "serving workload changed tokenizer length",
    )
    _require(
        workload.get("token_input_adapter")
        == "collision_free_content_block_tokens_v1"
        and isinstance(workload.get("token_input_identity_digest"), str)
        and bool(workload["token_input_identity_digest"]),
        "serving workload lacks exact token-input identity",
    )
    _require(
        report.get("demand_trace_digest") == workload["demand_trace_digest"],
        "serving demand digest diverges from workload provenance",
    )
    expected_ids = workload.get("request_id_order")
    actual_ids = [record.get("request_id") for record in report.get("records", [])]
    _require(
        isinstance(expected_ids, list)
        and all(isinstance(value, str) and value for value in expected_ids)
        and len(set(expected_ids)) == len(expected_ids)
        and len(actual_ids) == len(expected_ids)
        and all(isinstance(value, str) and value for value in actual_ids)
        and set(actual_ids) == set(expected_ids),
        "serving records do not preserve normalized request identities",
    )
    expected_arrivals = workload.get("request_arrival_offsets")
    _require(
        isinstance(expected_arrivals, dict)
        and set(expected_arrivals) == set(expected_ids),
        "serving workload lacks the request arrival mapping",
    )
    for record in report.get("records", []):
        expected = _finite(
            expected_arrivals.get(record["request_id"]),
            f"arrival offset for {record['request_id']}",
        )
        _require(
            expected >= 0
            and math.isclose(
                float(record["arrival_offset_seconds"]),
                expected,
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
            f"serving record arrival diverges for {record['request_id']}",
        )

    # Keep old diagnostic fixtures readable. New reports make a requested
    # suffix auditable against the exact input lengths seen by SGLang.
    if "external_suffix_tokens" not in workload:
        return
    suffix_tokens = workload.get("external_suffix_tokens")
    _require(
        isinstance(suffix_tokens, int)
        and not isinstance(suffix_tokens, bool)
        and suffix_tokens >= 0
        and report.get("external_suffix_tokens") == suffix_tokens,
        "serving workload has invalid external suffix provenance",
    )
    source_identity = workload.get("source_token_input_identity_digest")
    effective_identity = workload.get("token_input_identity_digest")
    _require(
        isinstance(source_identity, str)
        and bool(source_identity)
        and isinstance(effective_identity, str)
        and bool(effective_identity),
        "serving workload lacks source/effective token identity",
    )
    resident_lengths = workload.get("resident_input_tokens")
    source_external_lengths = workload.get("source_external_input_tokens")
    effective_external_lengths = workload.get("external_input_tokens")
    _require(
        isinstance(resident_lengths, list)
        and isinstance(source_external_lengths, list)
        and isinstance(effective_external_lengths, list)
        and all(isinstance(value, int) and value > 0 for value in resident_lengths)
        and all(
            isinstance(value, int) and value > 0 for value in source_external_lengths
        )
        and all(
            isinstance(value, int) and value > 0 for value in effective_external_lengths
        )
        and [
            effective - source
            for source, effective in zip(
                source_external_lengths, effective_external_lengths, strict=True
            )
        ]
        == [suffix_tokens] * len(source_external_lengths),
        "serving workload suffix does not match effective external input lengths",
    )
    records_by_kind = {
        kind: sorted(
            (record for record in report.get("records", []) if record["kind"] == kind),
            key=lambda record: int(record["index"]),
        )
        for kind in ("resident", "external")
    }
    _require(
        [record["input_tokens"] for record in records_by_kind["resident"]]
        == resident_lengths
        and [record["input_tokens"] for record in records_by_kind["external"]]
        == effective_external_lengths,
        "serving timed inputs diverge from effective workload lengths",
    )
    if suffix_tokens == 0:
        _require(
            workload.get("token_suffix_adapter") == "none"
            and workload.get("token_suffix_identity_digest") is None
            and source_identity == effective_identity,
            "zero-suffix replay changed token identity",
        )
    else:
        _require(
            workload.get("token_suffix_adapter")
            == "deterministic_request_unique_token_suffix_v1"
            and isinstance(workload.get("token_suffix_identity_digest"), str)
            and bool(workload["token_suffix_identity_digest"])
            and source_identity != effective_identity,
            "nonzero suffix replay lacks collision-free token identity",
        )


def _validate_byte_accounting(report: dict[str, Any]) -> None:
    selected = report.get("selected_bytes")
    physical = report.get("physical_bytes")
    status = report.get("byte_accounting_status")
    if selected is None or physical is None:
        _require(
            status == "not exposed by SGLang engine metadata",
            "serving report hides byte accounting without an explicit status",
        )
        return
    _finite(selected, "selected bytes")
    _finite(physical, "physical bytes")
    _require(
        float(selected) >= 0 and float(physical) >= float(selected),
        "serving byte accounting is inconsistent",
    )


def _validate_batch_heterogeneity(report: dict[str, Any]) -> None:
    try:
        expected = serving_batch_heterogeneity(
            report.get("records", ()), report.get("engine_stats", ())
        )
    except ValueError as error:
        raise ValueError(f"serving batch heterogeneity is invalid: {error}") from error
    _require(
        report.get("batch_heterogeneity") == expected,
        "serving batch heterogeneity does not match request and engine evidence",
    )


def _validate_tier_provenance(report: dict[str, Any]) -> None:
    """Validate tier metadata when the NTA engine publishes it.

    Older stock/reference reports intentionally have no NTA engine stream.
    Any NTA stream that does publish tier metadata must be self-consistent and
    may not hide a physical-tier fallback behind a host label.
    """
    stats = report.get("engine_stats", [])
    declarations = {
        str(entry["serving_tier"])
        for entry in stats
        if isinstance(entry, dict) and "serving_tier" in entry
    }
    if not declarations:
        return
    _require(len(declarations) == 1, "serving engine stats disagree on serving tier")
    tier = declarations.pop()
    _require(
        tier in {"host_staged", "nvme", "cxl_dax"}, "serving report has an unknown tier"
    )
    _require(
        tier != "cxl_dax",
        "CXL-DAX has a qualified native direct consumer but no SGLang "
        "FlashInfer numerical route; it cannot enter a serving artifact",
    )
    for entry in stats:
        if not isinstance(entry, dict) or "serving_tier" not in entry:
            continue
        _require(
            entry.get("tier_fallback") is False,
            "serving tier fallback was not fail-closed",
        )
        expected_contract = resource_contract(tier)
        _require(
            entry.get("resource_contract") == expected_contract.as_dict(),
            "serving tier resource contract diverges from the runtime contract",
        )
        _require(
            entry.get("tier_data_path") == expected_contract.steady_state_path,
            "serving tier data path diverges from the runtime contract",
        )
        if tier == "host_staged":
            _require(
                int(entry.get("tier_host_proxy_bytes", 0)) >= 0,
                "host-staged serving report has invalid proxy-byte accounting",
            )
        if tier in {"nvme", "cxl_dax"}:
            _require(
                int(entry.get("tier_host_proxy_bytes", 0)) == 0,
                "physical-tier serving report used host memory as a data proxy",
            )
            _require(
                isinstance(entry.get("tier_catalog_digest"), str)
                and entry["tier_catalog_digest"],
                "physical-tier serving report has no catalog digest",
            )
            _require(
                int(entry.get("nvme_bytes", 0)) > 0
                and int(entry.get("nvme_epochs", 0)) > 0,
                "NVMe serving report has no peer-DMA execution evidence",
            )
            _require(
                isinstance(entry.get("tier_capabilities"), dict),
                "physical-tier serving report has no capability evidence",
            )


def _validate_consumer_contract(
    report: dict[str, Any], *, require_formal_execution: bool
) -> None:
    """Validate the numerical consumer, not only the scheduler projection."""
    contracts: list[dict[str, Any]] = []
    for entry in report.get("engine_stats", []):
        if not isinstance(entry, dict) or entry.get("backend") != "nta_flashinfer":
            continue
        try:
            primary = validate_consumer_contract(
                entry.get("consumer_contract"),
                expected_engine="sglang",
                expected_backend=entry.get("backend"),
                require_formal_execution=require_formal_execution,
            )
            raw_paths = entry.get("consumer_contracts")
            if raw_paths is None:
                raw_paths = [entry.get("consumer_contract")]
            if not isinstance(raw_paths, list) or not raw_paths:
                raise ValueError("consumer path contracts must be a nonempty list")
            paths = [
                validate_consumer_contract(
                    value,
                    expected_engine="sglang",
                    expected_backend=entry.get("backend"),
                    require_formal_execution=require_formal_execution,
                )
                for value in raw_paths
            ]
        except ValueError as error:
            raise ValueError(str(error)) from error
        path_kinds = [contract["kind"] for contract in paths]
        native_launches = sum(
            int(entry.get(name, 0))
            for name in (
                "transformed_direct_launches",
                "ticketed_incremental_launches",
                "event_ordered_incremental_launches",
            )
        )
        stock_launches = int(
            entry.get("stock_prefetched_external_attention_launches", 0)
        )
        expected_kinds = [
            kind
            for kind, launches in (
                ("native_work_unit", native_launches),
                ("framework_reference", stock_launches),
            )
            if launches > 0
        ] or ["projection_only"]
        _require(
            path_kinds == expected_kinds and primary["kind"] == expected_kinds[0],
            "consumer path contracts diverge from timed numerical launches",
        )
        contracts.extend(paths)
    if require_formal_execution:
        _require(
            contracts,
            "formal NTA serving evidence has no numerical consumer",
        )
        if any(contract.get("kind") == "framework_reference" for contract in contracts):
            stock_external_launches = sum(
                int(entry.get("stock_prefetched_external_attention_launches", 0))
                for entry in report.get("engine_stats", [])
                if isinstance(entry, dict) and entry.get("backend") == "nta_flashinfer"
            )
            _require(
                stock_external_launches > 0,
                "framework-reference consumer was not backed by an external "
                "exact prefetch",
            )


def _validate_single(
    report: dict[str, Any],
    *,
    require_engine_stats: bool = True,
    require_complete_environment: bool = False,
) -> None:
    _require(report.get("schema") == 1, "unsupported serving report schema")
    _require(
        report.get("classification") == "sglang-hicache-load",
        "serving result is not an SGLang load report",
    )
    _require(
        isinstance(report.get("revision"), str) and report["revision"],
        "serving report has no revision",
    )
    _require(
        isinstance(report.get("machine"), dict) and report["machine"],
        "serving report has no machine metadata",
    )
    backend = report.get("attention_backend")
    _require(
        backend in {"flashinfer", "nta_flashinfer"},
        "serving report has no supported attention backend",
    )
    _require(report.get("demand_semantics") == "exact", "serving demand is not exact")
    _require(report.get("placement_proven") is True, "serving placement was not proven")
    _validate_environment(report, require_complete=require_complete_environment)
    _require(
        report.get("verification_failures") == 0,
        "serving report has verification failures",
    )
    correctness = report.get("correctness")
    _require(
        isinstance(correctness, dict) and correctness.get("verification_failures") == 0,
        "serving correctness contract is incomplete",
    )
    _require(
        isinstance(correctness.get("generated_text_sha256"), str)
        and correctness["generated_text_sha256"],
        "serving report has no output correctness digest",
    )
    _require(
        isinstance(report.get("generated_text_sha256"), str)
        and report["generated_text_sha256"],
        "serving report has no aggregate output digest",
    )
    records = report.get("records")
    _require(
        isinstance(records, list) and records, "serving report has no request records"
    )
    for index, record in enumerate(records):
        _require(isinstance(record, dict), f"serving record {index} is not an object")
        _require(
            record.get("kind") in {"resident", "external"},
            f"serving record {index} has an unknown request kind",
        )
        for field in (
            "kind",
            "arrival_offset_seconds",
            "submitted_offset_seconds",
            "finished_offset_seconds",
            "ttft_seconds",
            "tpot_seconds",
            "p99_itl_seconds",
            "admission_delay_seconds",
            "system_time_seconds",
            "completion_tokens",
            "itl_sample_count",
            "token_timestamps_exact",
            "token_timestamp_source",
            "input_tokens",
            "host_cached_tokens",
            "device_cached_tokens",
            "text_sha256",
            "request_id",
        ):
            _require(field in record, f"serving record {index} lacks {field}")
        for field in (
            "arrival_offset_seconds",
            "submitted_offset_seconds",
            "finished_offset_seconds",
            "ttft_seconds",
            "tpot_seconds",
            "p99_itl_seconds",
            "admission_delay_seconds",
            "system_time_seconds",
        ):
            _require(
                _finite(record[field], f"record {index} {field}") >= 0,
                f"record {index} {field} is negative",
            )
        _require(
            float(record["submitted_offset_seconds"])
            >= float(record["arrival_offset_seconds"]),
            f"serving record {index} was submitted before arrival",
        )
        _require(
            float(record["finished_offset_seconds"])
            >= float(record["submitted_offset_seconds"]),
            f"serving record {index} finished before submission",
        )
        _close(
            record["admission_delay_seconds"],
            float(record["submitted_offset_seconds"])
            - float(record["arrival_offset_seconds"]),
            f"record {index} admission delay",
        )
        _close(
            record["system_time_seconds"],
            float(record["finished_offset_seconds"])
            - float(record["arrival_offset_seconds"]),
            f"record {index} system time",
        )
        _require(
            isinstance(record["completion_tokens"], int)
            and not isinstance(record["completion_tokens"], bool)
            and record["completion_tokens"] > 0,
            f"serving record {index} has invalid completion token count",
        )
        _require(
            record["token_timestamps_exact"] is True
            and record["token_timestamp_source"]
            == "sglang_stream_interval_1_completion_delta",
            f"serving record {index} lacks exact token timestamp evidence",
        )
        _require(
            isinstance(record["itl_sample_count"], int)
            and not isinstance(record["itl_sample_count"], bool)
            and record["itl_sample_count"]
            == max(0, record["completion_tokens"] - 1)
            == len(record.get("inter_token_seconds", [])),
            f"serving record {index} ITL samples do not match completion tokens",
        )
        _require(
            isinstance(record["text_sha256"], str) and record["text_sha256"],
            f"serving record {index} has no output digest",
        )
        intervals = [float(value) for value in record.get("inter_token_seconds", [])]
        _require(
            all(math.isfinite(value) and value >= 0.0 for value in intervals),
            f"serving record {index} contains an invalid ITL",
        )
        _close(
            record["p99_itl_seconds"],
            _percentile(intervals, 0.99),
            f"record {index} p99 ITL",
        )
    engine_stats = report.get("engine_stats")
    _require(
        isinstance(engine_stats, list),
        "serving report engine statistics are not a list",
    )
    if require_engine_stats:
        _require(engine_stats, "serving report has no engine statistics")
    if backend == "flashinfer":
        _require(
            not engine_stats,
            "stock serving report contains NTA engine statistics",
        )
    else:
        _require(engine_stats, "NTA serving report has no engine statistics")
    for field in (
        "p50_ttft_seconds",
        "p95_ttft_seconds",
        "p99_ttft_seconds",
        "p50_tpot_seconds",
        "p95_tpot_seconds",
        "p99_tpot_seconds",
        "p99_itl_seconds",
        "slo_goodput",
    ):
        _require(field in report, f"serving report lacks {field}")
    for field in (
        "p50_ttft_seconds",
        "p95_ttft_seconds",
        "p99_ttft_seconds",
        "p50_tpot_seconds",
        "p95_tpot_seconds",
        "p99_tpot_seconds",
        "p99_itl_seconds",
    ):
        _finite(report[field], field)
    _require(
        report["p50_ttft_seconds"]
        <= report["p95_ttft_seconds"]
        <= report["p99_ttft_seconds"]
        and report["p50_tpot_seconds"]
        <= report["p95_tpot_seconds"]
        <= report["p99_tpot_seconds"],
        "serving latency percentiles are not monotonic",
    )
    goodput = report["slo_goodput"]
    _require(isinstance(goodput, dict), "serving report has no SLO goodput object")
    for field in (
        "qualified_requests",
        "total_requests",
        "goodput_requests_per_second",
    ):
        _finite(goodput.get(field), f"SLO goodput {field}")
    _require(
        int(goodput["total_requests"]) == len(records),
        "SLO goodput request count diverges from records",
    )
    _require(
        0 <= int(goodput["qualified_requests"]) <= int(goodput["total_requests"])
        and goodput["goodput_requests_per_second"] >= 0,
        "SLO goodput counts or rate are inconsistent",
    )
    elapsed = _finite(report.get("elapsed_seconds"), "elapsed seconds")
    _require(elapsed > 0.0, "serving elapsed time is not positive")
    _close(
        report.get("request_throughput"),
        len(records) / elapsed,
        "request throughput",
    )
    _close(
        report.get("output_token_throughput"),
        sum(int(record["completion_tokens"]) for record in records) / elapsed,
        "output token throughput",
    )
    for field, source, fraction in (
        ("p50_ttft_seconds", "ttft_seconds", 0.50),
        ("p95_ttft_seconds", "ttft_seconds", 0.95),
        ("p99_ttft_seconds", "ttft_seconds", 0.99),
        ("p50_tpot_seconds", "tpot_seconds", 0.50),
        ("p95_tpot_seconds", "tpot_seconds", 0.95),
        ("p99_tpot_seconds", "tpot_seconds", 0.99),
    ):
        _close(
            report[field],
            _percentile([float(record[source]) for record in records], fraction),
            field,
        )
    all_intervals = [
        float(interval)
        for record in records
        for interval in record["inter_token_seconds"]
    ] or [0.0]
    _close(
        report["p99_itl_seconds"],
        _percentile(all_intervals, 0.99),
        "p99_itl_seconds",
    )
    thresholds = goodput.get("thresholds_seconds")
    _require(
        isinstance(thresholds, dict),
        "serving SLO goodput has no threshold identity",
    )
    ttft_threshold = _finite(thresholds.get("ttft"), "SLO TTFT threshold")
    itl_threshold = _finite(thresholds.get("p99_itl"), "SLO ITL threshold")
    _require(
        ttft_threshold > 0.0 and itl_threshold > 0.0,
        "serving SLO thresholds are not positive",
    )
    token_level_requests = sum(
        int(record["itl_sample_count"]) > 0
        and record["token_timestamps_exact"] is True
        for record in records
    )
    qualified = sum(
        float(record["ttft_seconds"]) <= ttft_threshold
        and int(record["itl_sample_count"]) > 0
        and record["token_timestamps_exact"] is True
        and float(record["p99_itl_seconds"]) <= itl_threshold
        for record in records
    )
    _require(
        int(goodput["qualified_requests"]) == qualified
        and goodput.get("requests_with_token_level_itl") == token_level_requests,
        "serving SLO goodput was not recomputed from exact token timestamps",
    )
    _close(goodput.get("attainment"), qualified / len(records), "SLO attainment")
    _close(
        goodput["goodput_requests_per_second"],
        qualified / elapsed,
        "SLO goodput rate",
    )
    _validate_finite_window_accounting(report)
    _validate_workload(report)
    _validate_byte_accounting(report)
    _validate_batch_heterogeneity(report)
    _validate_tier_provenance(report)
    _validate_consumer_contract(report, require_formal_execution=require_engine_stats)


def validate(report: dict[str, Any]) -> dict[str, Any]:
    classification = report.get("classification")
    if classification == "sglang-hicache-load":
        _validate_single(
            report,
            require_engine_stats=report.get("attention_backend") == "nta_flashinfer",
        )
        return report
    _require(
        classification == "sglang-hicache-load-comparison",
        "unsupported serving comparison classification",
    )
    _require(
        report.get("outputs_diverge") is False,
        "serving comparison has divergent outputs",
    )
    stock = report.get("stock")
    nta = report.get("nta")
    _require(
        isinstance(stock, dict) and isinstance(nta, dict),
        "serving comparison lacks stock or NTA report",
    )
    # The stock arm intentionally has no NTA plugin statistics.  Its timing,
    # correctness, placement, and finite-window accounting remain mandatory; only
    # the implementation-specific engine-stat stream is absent.
    _validate_single(
        stock,
        require_engine_stats=False,
        require_complete_environment=True,
    )
    _validate_single(nta, require_complete_environment=True)
    comparison_revision = report.get("revision")
    _require(
        isinstance(comparison_revision, str)
        and comparison_revision
        and comparison_revision == stock.get("revision") == nta.get("revision"),
        "serving comparison arms do not share the recorded revision",
    )
    _require(
        stock.get("machine") == nta.get("machine"),
        "serving comparison arms do not share the recorded machine",
    )
    _require(
        isinstance(report.get("harness_args"), dict),
        "serving comparison has no harness argument identity",
    )
    execution_order = report.get("execution_order")
    _require(
        isinstance(execution_order, list)
        and sorted(execution_order) == ["flashinfer", "nta_flashinfer"],
        "serving comparison has an invalid arm execution order",
    )
    _require(
        stock.get("demand_trace_digest") == nta.get("demand_trace_digest"),
        "stock and NTA demand digests differ",
    )
    _require(
        stock.get("generated_text_sha256") == nta.get("generated_text_sha256"),
        "stock and NTA output digests differ despite outputs_diverge=false",
    )
    activation = report.get("mechanism_activation")
    _require(
        isinstance(activation, dict) and activation.get("external_launches", 0) > 0,
        "serving comparison has no mechanism activation",
    )
    evidence_scope = report.get("evidence_scope")
    _require(
        evidence_scope
        in {
            "heterogeneous_work_unit",
            "native_work_unit",
            "transport_only",
            "exact_execution_only",
        },
        "serving comparison has no typed evidence scope",
    )
    expected_scope = (
        "heterogeneous_work_unit"
        if activation.get("heterogeneous_work_unit_active") is True
        and activation.get("batch_heterogeneity_proven") is True
        else "native_work_unit"
        if activation.get("native_work_unit_active") is True
        else "transport_only"
        if activation.get("transport_only") is True
        else "exact_execution_only"
    )
    _require(
        evidence_scope == expected_scope,
        "serving comparison evidence scope disagrees with activation counters",
    )
    if evidence_scope == "heterogeneous_work_unit":
        _require(
            isinstance(nta.get("batch_heterogeneity"), dict)
            and nta["batch_heterogeneity"].get("proven") is True,
            "heterogeneous-work-unit evidence lacks batch-internal workload proof",
        )
    _require(
        activation.get("external_attention_accounted") is True,
        "serving comparison did not account for exact external attention work",
    )
    _require(
        activation.get("external_attention_transformed") is True
        or activation.get("external_attention_stock_consumer") is True,
        "serving comparison did not prove an exact external attention consumer",
    )
    _require(
        activation.get("fallback_batches") == 0,
        "serving comparison used a fallback batch",
    )
    _require(
        int(activation.get("external_launches", 0))
        == int(activation.get("transformed_external_launches", 0))
        + int(activation.get("stock_prefetched_external_launches", 0)),
        "serving comparison external attention accounting is not exact",
    )
    nta_stats = nta.get("engine_stats")
    _require(isinstance(nta_stats, list), "NTA serving arm has no engine statistics")
    try:
        expected_transport_execution = summarize_transport_execution(nta_stats)
    except ValueError as error:
        raise ValueError(f"invalid physical execution counters: {error}") from error
    _require(
        report.get("transport_execution") == expected_transport_execution,
        "serving transport execution evidence disagrees with timed counters",
    )
    harness_args = report["harness_args"]
    required_paths = harness_args.get("require_exercised_path")
    _require(
        isinstance(required_paths, list)
        and all(isinstance(name, str) for name in required_paths)
        and len(required_paths) == len(set(required_paths))
        and set(required_paths).issubset(EXERCISED_PATHS),
        "serving comparison has invalid required execution paths",
    )
    _require(
        all(
            expected_transport_execution[name]["exercised"] is True
            for name in required_paths
        ),
        "serving comparison did not exercise every required physical path",
    )
    try:
        require_frontier_shape(
            expected_transport_execution,
            native_layers=harness_args.get("require_native_frontier_layers"),
            ready_stock_layers=harness_args.get("require_ready_stock_layers"),
            progressive_layers=harness_args.get("require_progressive_layers"),
        )
    except ValueError as error:
        raise ValueError(f"invalid serving frontier evidence: {error}") from error
    scale = _finite(report.get("slo_scale"), "relative SLO scale")
    _require(scale > 0.0, "serving comparison has a nonpositive SLO scale")
    thresholds = relative_thresholds(stock, scale)
    _validate_derived_object(
        report.get("slo_thresholds_seconds"),
        thresholds,
        "relative SLO thresholds",
    )
    stock_goodput = relative_goodput(stock, thresholds)
    nta_goodput = relative_goodput(nta, thresholds)
    stock_preregistered = preregistered_goodput(stock)
    nta_preregistered = preregistered_goodput(nta)
    _validate_derived_object(
        report.get("stock_goodput"), stock_goodput, "stock relative goodput"
    )
    _validate_derived_object(
        report.get("nta_goodput"), nta_goodput, "NTA relative goodput"
    )
    _validate_derived_object(
        report.get("stock_preregistered_goodput"),
        stock_preregistered,
        "stock preregistered goodput",
    )
    _validate_derived_object(
        report.get("nta_preregistered_goodput"),
        nta_preregistered,
        "NTA preregistered goodput",
    )
    _close(
        report.get("stock_slo_goodput"),
        float(stock["slo_goodput"]["goodput_requests_per_second"]),
        "stock SLO goodput",
    )
    _close(
        report.get("nta_slo_goodput"),
        float(nta["slo_goodput"]["goodput_requests_per_second"]),
        "NTA SLO goodput",
    )
    for prefix, arm in (("stock", stock), ("nta", nta)):
        for field in (
            "p50_ttft_seconds",
            "p95_ttft_seconds",
            "p99_ttft_seconds",
            "p99_itl_seconds",
        ):
            _close(
                report.get(f"{prefix}_{field}"),
                float(arm[field]),
                f"{prefix} {field}",
            )
    _validate_ratio(
        report.get("output_throughput_ratio"),
        safe_ratio(
            float(nta["output_token_throughput"]),
            float(stock["output_token_throughput"]),
        ),
        "output throughput ratio",
    )
    _validate_ratio(
        report.get("goodput_ratio"),
        safe_ratio(
            float(nta_goodput["goodput_requests_per_second"]),
            float(stock_goodput["goodput_requests_per_second"]),
        ),
        "relative-goodput ratio",
    )
    _validate_ratio(
        report.get("preregistered_goodput_ratio"),
        safe_ratio(
            float(nta_preregistered["goodput_requests_per_second"]),
            float(stock_preregistered["goodput_requests_per_second"]),
        ),
        "preregistered-goodput ratio",
    )
    for field, numerator, denominator in (
        (
            "resident_p95_ttft_ratio",
            nta["resident_p95_ttft_seconds"],
            stock["resident_p95_ttft_seconds"],
        ),
        (
            "resident_p95_tpot_ratio",
            nta["resident_p95_tpot_seconds"],
            stock["resident_p95_tpot_seconds"],
        ),
        (
            "resident_p99_itl_ratio",
            nta["resident_p99_itl_seconds"],
            stock["resident_p99_itl_seconds"],
        ),
        (
            "external_p95_ttft_ratio",
            nta["external_p95_ttft_seconds"],
            stock["external_p95_ttft_seconds"],
        ),
    ):
        _validate_ratio(
            report.get(field),
            safe_ratio(float(numerator), float(denominator)),
            field,
        )
    return report


def validate_formal_arm(report: dict[str, Any]) -> dict[str, Any]:
    """Validate one independently timed serving arm for formal evaluation.

    Comparison reports remain useful diagnostics, but they are not one causal
    arm and therefore cannot satisfy this contract.  Formal TPOT/ITL evidence
    also requires at least two generated tokens for every measured request;
    accepting one-token trials would make both metrics structurally vacuous.
    """

    _require(
        report.get("classification") == "sglang-hicache-load",
        "formal serving arm must be one load report, not a nested comparison",
    )
    _validate_single(
        report,
        require_engine_stats=report.get("attention_backend") == "nta_flashinfer",
        require_complete_environment=True,
    )
    _require(report.get("dirty") is False, "formal serving arm used a dirty revision")
    records = report.get("records", [])
    _require(
        bool(records)
        and all(int(record.get("completion_tokens", 0)) >= 2 for record in records),
        "formal serving arm has vacuous one-token TPOT/ITL measurements",
    )
    _require(
        sum(int(record.get("itl_sample_count", 0)) for record in records) > 0,
        "formal serving arm contains no token-level ITL samples",
    )
    _require(
        report.get("load_warmup_excluded") is True,
        "formal serving arm did not exclude a shape-faithful load warmup",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    validate(json.loads(args.report.resolve().read_text(encoding="utf-8")))
    print("serving_report=valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
