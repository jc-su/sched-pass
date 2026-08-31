"""Pure serving metrics shared by producers and artifact validators.

Metric definitions must not live only in the benchmark that emits them.  The
artifact validator imports the same functions and recomputes every value from
request records, so a stale or hand-edited aggregate cannot pass by remaining
finite.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


PREREGISTERED_TTFT_SECONDS = 8.0
PREREGISTERED_TPOT_SECONDS = 0.050
PREREGISTERED_P99_ITL_SECONDS = 0.100


def _records(report: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    records = report.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("serving metric input has no request records")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("serving metric input contains a non-object record")
    return records


def _elapsed(report: Mapping[str, Any]) -> float:
    elapsed = report.get("elapsed_seconds")
    if (
        not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or not math.isfinite(float(elapsed))
        or float(elapsed) <= 0.0
    ):
        raise ValueError("serving metric input has no positive elapsed time")
    return float(elapsed)


def safe_ratio(numerator: float, denominator: float) -> float | None:
    """Return a finite ratio, or ``None`` when no finite ratio exists."""

    if not math.isfinite(numerator) or not math.isfinite(denominator):
        raise ValueError("serving metric ratio inputs must be finite")
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else None
    return numerator / denominator


def relative_thresholds(
    stock: Mapping[str, Any], scale: float
) -> dict[str, float]:
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("relative SLO scale must be positive")
    return {
        "resident_ttft": scale * float(stock["resident_p95_ttft_seconds"]),
        "resident_tpot": scale * float(stock["resident_p95_tpot_seconds"]),
        "resident_itl": scale * float(stock["resident_p99_itl_seconds"]),
        "external_ttft": scale * float(stock["external_p95_ttft_seconds"]),
    }


def relative_goodput(
    report: Mapping[str, Any], thresholds: Mapping[str, float]
) -> dict[str, Any]:
    records = _records(report)
    resident_ok: list[bool] = []
    external_ok: list[bool] = []
    token_level = 0
    for record in records:
        exact_itl = (
            int(record.get("itl_sample_count", 0)) > 0
            and record.get("token_timestamps_exact") is True
        )
        token_level += int(exact_itl)
        kind = record.get("kind")
        if kind == "resident":
            resident_ok.append(
                float(record["ttft_seconds"]) <= thresholds["resident_ttft"]
                and exact_itl
                and float(record["tpot_seconds"]) <= thresholds["resident_tpot"]
                and float(record["p99_itl_seconds"])
                <= thresholds["resident_itl"]
            )
        elif kind == "external":
            external_ok.append(
                float(record["ttft_seconds"]) <= thresholds["external_ttft"]
                and exact_itl
            )
        else:
            raise ValueError("serving metric record has an unknown request kind")
    passed = sum(resident_ok) + sum(external_ok)
    total = len(records)
    return {
        "passed_requests": passed,
        "total_requests": total,
        "requests_with_token_level_itl": token_level,
        "slo_attainment": passed / total,
        "goodput_requests_per_second": passed / _elapsed(report),
        "resident_slo_attainment": (
            sum(resident_ok) / len(resident_ok) if resident_ok else 0.0
        ),
        "external_slo_attainment": (
            sum(external_ok) / len(external_ok) if external_ok else 0.0
        ),
    }


def joint_slo_goodput(
    report: Mapping[str, Any],
    *,
    ttft_seconds: float,
    tpot_seconds: float,
    p99_itl_seconds: float,
) -> dict[str, Any]:
    """Compute request goodput under one fixed TTFT/TPOT/ITL contract."""

    thresholds = (ttft_seconds, tpot_seconds, p99_itl_seconds)
    if any(not math.isfinite(value) or value <= 0.0 for value in thresholds):
        raise ValueError("joint serving SLO thresholds must be positive and finite")

    records = _records(report)
    token_level = sum(
        int(record.get("itl_sample_count", 0)) > 0
        and record.get("token_timestamps_exact") is True
        for record in records
    )
    qualified = sum(
        float(record["ttft_seconds"]) <= ttft_seconds
        and int(record.get("itl_sample_count", 0)) > 0
        and record.get("token_timestamps_exact") is True
        and float(record["tpot_seconds"]) <= tpot_seconds
        and float(record["p99_itl_seconds"]) <= p99_itl_seconds
        for record in records
    )
    total = len(records)
    return {
        "qualified_requests": qualified,
        "total_requests": total,
        "requests_with_token_level_itl": token_level,
        "attainment": qualified / total,
        "goodput_requests_per_second": qualified / _elapsed(report),
        "thresholds_seconds": {
            "ttft": ttft_seconds,
            "tpot": tpot_seconds,
            "p99_itl": p99_itl_seconds,
        },
    }


def preregistered_goodput(report: Mapping[str, Any]) -> dict[str, Any]:
    """Compute the legacy fixed TTFT-and-token-ITL goodput.

    Keep this historical metric stable so banked artifacts retain one meaning.
    New formal evaluations use :func:`preregistered_joint_goodput`.
    """

    records = _records(report)
    token_level = sum(
        int(record.get("itl_sample_count", 0)) > 0
        and record.get("token_timestamps_exact") is True
        for record in records
    )
    qualified = sum(
        float(record["ttft_seconds"]) <= PREREGISTERED_TTFT_SECONDS
        and int(record.get("itl_sample_count", 0)) > 0
        and record.get("token_timestamps_exact") is True
        and float(record["p99_itl_seconds"])
        <= PREREGISTERED_P99_ITL_SECONDS
        for record in records
    )
    total = len(records)
    return {
        "qualified_requests": qualified,
        "total_requests": total,
        "requests_with_token_level_itl": token_level,
        "attainment": qualified / total,
        "goodput_requests_per_second": qualified / _elapsed(report),
        "thresholds_seconds": {
            "ttft": PREREGISTERED_TTFT_SECONDS,
            "p99_itl": PREREGISTERED_P99_ITL_SECONDS,
        },
    }


def preregistered_joint_goodput(report: Mapping[str, Any]) -> dict[str, Any]:
    """Compute the fixed joint TTFT/TPOT/p99-ITL serving goodput."""

    return joint_slo_goodput(
        report,
        ttft_seconds=PREREGISTERED_TTFT_SECONDS,
        tpot_seconds=PREREGISTERED_TPOT_SECONDS,
        p99_itl_seconds=PREREGISTERED_P99_ITL_SECONDS,
    )
