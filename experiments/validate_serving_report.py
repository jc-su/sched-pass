#!/usr/bin/env python3
"""Validate structured serving evidence before it enters an artifact bundle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from .consumer_contract import validate_consumer_contract
except ImportError:  # pragma: no cover - direct script execution
    from consumer_contract import validate_consumer_contract


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


def _validate_littles_law(report: dict[str, Any]) -> None:
    little = report.get("littles_law")
    _require(isinstance(little, dict), "serving report has no Little's Law report")
    _require(
        little.get("method") == "finite_window_arrival_departure_accounting",
        "serving report uses an unknown Little's Law method",
    )
    _finite(little.get("residual"), "Little's Law residual")


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
    for entry in stats:
        if not isinstance(entry, dict) or "serving_tier" not in entry:
            continue
        _require(
            entry.get("tier_fallback") is False,
            "serving tier fallback was not fail-closed",
        )
        if tier in {"nvme", "cxl_dax"}:
            expected_path = (
                "gpu_owned_nvme_to_hbm" if tier == "nvme" else "cuda_visible_cxl_direct"
            )
            _require(
                entry.get("tier_data_path") == expected_path,
                "physical-tier serving report has an unexpected data path",
            )
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
                isinstance(entry.get("tier_capabilities"), dict),
                "physical-tier serving report has no capability evidence",
            )


def _validate_consumer_contract(
    report: dict[str, Any], *, require_formal_execution: bool
) -> None:
    """Validate the numerical consumer, not only the scheduler projection."""
    for entry in report.get("engine_stats", []):
        if not isinstance(entry, dict) or entry.get("backend") != "nta_flashinfer":
            continue
        try:
            validate_consumer_contract(
                entry.get("consumer_contract"),
                expected_engine="sglang",
                expected_backend=entry.get("backend"),
                require_formal_execution=require_formal_execution,
            )
        except ValueError as error:
            raise ValueError(str(error)) from error


def _validate_single(
    report: dict[str, Any], *, require_engine_stats: bool = True
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
    _require(report.get("demand_semantics") == "exact", "serving demand is not exact")
    _require(report.get("placement_proven") is True, "serving placement was not proven")
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
        _require(
            isinstance(record["completion_tokens"], int)
            and not isinstance(record["completion_tokens"], bool)
            and record["completion_tokens"] > 0,
            f"serving record {index} has invalid completion token count",
        )
        _require(
            isinstance(record["text_sha256"], str) and record["text_sha256"],
            f"serving record {index} has no output digest",
        )
    engine_stats = report.get("engine_stats")
    _require(
        isinstance(engine_stats, list),
        "serving report engine statistics are not a list",
    )
    if require_engine_stats:
        _require(engine_stats, "serving report has no engine statistics")
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
    _validate_littles_law(report)
    _validate_workload(report)
    _validate_byte_accounting(report)
    _validate_tier_provenance(report)
    _validate_consumer_contract(
        report, require_formal_execution=require_engine_stats
    )


def validate(report: dict[str, Any]) -> dict[str, Any]:
    classification = report.get("classification")
    if classification == "sglang-hicache-load":
        _validate_single(report)
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
    # correctness, placement, and Little's Law fields remain mandatory; only
    # the implementation-specific engine-stat stream is absent.
    _validate_single(stock, require_engine_stats=False)
    _validate_single(nta)
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
    for field in (
        "nta_slo_goodput",
        "stock_slo_goodput",
        "goodput_ratio",
        "output_throughput_ratio",
        "external_p95_ttft_ratio",
    ):
        _finite(report.get(field), field)
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
