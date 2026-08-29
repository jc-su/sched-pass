"""Typed, outcome-independent workload scenario identity for paired evaluation."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
from typing import Any, Mapping

try:
    from .validate_workload import validate as validate_workload
except ImportError:  # pragma: no cover - direct script execution
    from validate_workload import validate as validate_workload


_SUMMARY_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cached_prefix_tokens",
    "uncached_query_rows",
)


def _summary(value: Any, name: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"workload scenario lacks {name} statistics")
    result: dict[str, float] = {}
    for field in ("min", "mean", "max"):
        number = value.get(field)
        if (
            not isinstance(number, (int, float))
            or isinstance(number, bool)
            or not math.isfinite(float(number))
            or float(number) < 0.0
        ):
            raise ValueError(f"workload scenario has invalid {name}.{field}")
        result[field] = float(number)
    if not result["min"] <= result["mean"] <= result["max"]:
        raise ValueError(f"workload scenario has non-monotonic {name} statistics")
    return result


def describe_workload_scenario(label: str, manifest_path: Path) -> dict[str, Any]:
    """Derive the complete scenario descriptor from a validated manifest."""

    if re.fullmatch(r"[A-Za-z0-9_.-]+", label) is None:
        raise ValueError(f"workload scenario id is invalid: {label!r}")
    path = manifest_path.resolve()
    manifest = validate_workload(path)
    statistics = manifest.get("statistics")
    if not isinstance(statistics, Mapping):
        raise ValueError("workload manifest has no statistics")
    arrival = manifest.get("arrival")
    if not isinstance(arrival, Mapping):
        raise ValueError("workload manifest has no arrival contract")
    state_counts = statistics.get("request_state_counts", {})
    if not isinstance(state_counts, Mapping) or not all(
        isinstance(name, str)
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
        for name, count in state_counts.items()
    ):
        raise ValueError("workload manifest has invalid request-state counts")
    summaries = {
        name: _summary(statistics.get(name), name) for name in _SUMMARY_FIELDS
    }
    return {
        "schema": 1,
        "id": label,
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "demand_trace_digest": str(manifest["demand_trace_digest"]),
        "records_digest": str(manifest["records_digest"]),
        "request_count": int(manifest["request_count"]),
        "arrival": {
            "mode": arrival.get("mode"),
            "source": arrival.get("source"),
            "target_rate_per_second": arrival.get("target_rate_per_second"),
            "time_scale": arrival.get("time_scale"),
            "production_arrival_claim": arrival.get(
                "production_arrival_claim", False
            ),
        },
        "request_state_counts": dict(sorted(state_counts.items())),
        "statistics": summaries,
        "heterogeneous_axes": sorted(
            name
            for name, summary in summaries.items()
            if summary["min"] != summary["max"]
        ),
    }


def validate_workload_scenario(
    descriptor: Mapping[str, Any], manifest_path: Path
) -> dict[str, Any]:
    """Re-derive a descriptor so labels cannot diverge from workload data."""

    label = descriptor.get("id")
    if not isinstance(label, str):
        raise ValueError("workload scenario has no id")
    expected = describe_workload_scenario(label, manifest_path)
    if dict(descriptor) != expected:
        raise ValueError(
            f"workload scenario {label!r} does not match its validated manifest"
        )
    return expected
