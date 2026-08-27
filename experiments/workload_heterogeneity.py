"""Measured heterogeneity evidence for serving workloads.

Request diversity in a manifest is not proof that a scheduler co-batched the
requests.  This module keeps three scopes separate: the request set's observed
shape, client-side lifetime overlap, and typed counters emitted from one
engine ForwardBatch.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _integer(record: Mapping[str, Any], name: str) -> int:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"serving record {name} must be a non-negative integer")
    return value


def _number(record: Mapping[str, Any], name: str) -> float:
    value = record.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"serving record {name} must be finite")
    return float(value)


def _axis(values: Sequence[int | float | str]) -> dict[str, Any]:
    distinct = len(set(values))
    result: dict[str, Any] = {
        "distinct": distinct,
        "heterogeneous": distinct > 1,
    }
    if values and all(isinstance(value, (int, float)) for value in values):
        result.update({"min": min(values), "max": max(values)})
    else:
        result["values"] = sorted({str(value) for value in values})
    return result


def _overlap(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    intervals: list[tuple[float, float, str]] = []
    for record in records:
        begin = _number(record, "submitted_offset_seconds")
        end = _number(record, "finished_offset_seconds")
        kind = str(record.get("kind"))
        if kind not in {"resident", "external"}:
            raise ValueError("serving record kind must be resident or external")
        if begin < 0 or end < begin:
            raise ValueError("serving record lifetime is invalid")
        intervals.append((begin, end, kind))

    # Group equal timestamps.  The active set describes the open interval
    # between the previous event and this one, so zero-width contact is never
    # mistaken for actual concurrency.
    events: dict[float, list[tuple[int, str]]] = {}
    for begin, end, kind in intervals:
        events.setdefault(begin, []).append((1, kind))
        events.setdefault(end, []).append((-1, kind))
    active = {"resident": 0, "external": 0}
    maximum = 0
    mixed_seconds = 0.0
    previous: float | None = None
    for timestamp in sorted(events):
        if previous is not None and active["resident"] and active["external"]:
            mixed_seconds += timestamp - previous
        # Apply equal-time events as one net transition. A zero-duration
        # request or simultaneous retire/admit contributes no overlap.
        for kind in ("resident", "external"):
            active[kind] += sum(
                delta for delta, event_kind in events[timestamp] if event_kind == kind
            )
            if active[kind] < 0:
                raise ValueError("serving lifetimes retire an inactive request")
        maximum = max(maximum, active["resident"] + active["external"])
        previous = timestamp
    if any(active.values()):
        raise ValueError("serving lifetimes do not retire every request")
    return {
        "maximum_concurrent_requests": maximum,
        "resident_external_overlap_seconds": mixed_seconds,
        "resident_external_overlap": mixed_seconds > 0.0,
    }


def _counter(stats: Sequence[Mapping[str, Any]], name: str) -> int:
    total = 0
    for report in stats:
        value = report.get(name, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"engine heterogeneity counter {name} is invalid")
        total += value
    return total


def serving_batch_heterogeneity(
    records: Sequence[Mapping[str, Any]],
    engine_stats: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a fail-closed proof of actual batch-internal heterogeneity."""

    if not records:
        raise ValueError("heterogeneity evidence requires request records")
    kinds: list[str] = []
    input_tokens: list[int] = []
    output_tokens: list[int] = []
    cached_prefix_tokens: list[int] = []
    query_rows: list[int] = []
    arrivals: list[float] = []
    for record in records:
        kind = str(record.get("kind"))
        if kind not in {"resident", "external"}:
            raise ValueError("serving record kind must be resident or external")
        input_count = _integer(record, "input_tokens")
        output_count = _integer(record, "completion_tokens")
        cached = _integer(record, "host_cached_tokens") + _integer(
            record, "device_cached_tokens"
        )
        if input_count <= 0 or cached > input_count:
            raise ValueError("serving record cached-prefix geometry is invalid")
        kinds.append(kind)
        input_tokens.append(input_count)
        output_tokens.append(output_count)
        cached_prefix_tokens.append(cached)
        query_rows.append(input_count - cached)
        arrivals.append(_number(record, "arrival_offset_seconds"))

    axes = {
        "request_state": _axis(kinds),
        "input_tokens": _axis(input_tokens),
        "cached_prefix_tokens": _axis(cached_prefix_tokens),
        "uncached_query_rows": _axis(query_rows),
        "output_tokens": _axis(output_tokens),
        "arrival_offset_seconds": _axis(arrivals),
    }
    overlap = _overlap(records)
    engine = {
        name: _counter(engine_stats, name)
        for name in (
            "multi_request_engine_batches",
            "heterogeneous_engine_batches",
            "multi_axis_heterogeneous_batches",
            "sequence_length_heterogeneous_batches",
            "availability_heterogeneous_batches",
            "external_rows_heterogeneous_batches",
            "tenant_heterogeneous_batches",
            "priority_heterogeneous_batches",
            "deadline_heterogeneous_batches",
            "mixed_dependency_layers",
        )
    }
    batch_internal = (
        engine["heterogeneous_engine_batches"] > 0
        and engine["sequence_length_heterogeneous_batches"] > 0
        and engine["availability_heterogeneous_batches"] > 0
        and engine["mixed_dependency_layers"] > 0
    )
    proven = bool(overlap["resident_external_overlap"] and batch_internal)
    return {
        "schema": 1,
        "request_count": len(records),
        "axes": axes,
        "heterogeneous_axis_count": sum(
            bool(value["heterogeneous"]) for value in axes.values()
        ),
        "client_overlap": overlap,
        "engine_forward": engine,
        "batch_internal_geometry_proven": (
            engine["sequence_length_heterogeneous_batches"] > 0
        ),
        "batch_internal_availability_proven": (
            engine["availability_heterogeneous_batches"] > 0
        ),
        "batch_internal_proven": batch_internal,
        "proven": proven,
        "scope": "batch_internal" if proven else "request_set_only",
    }
