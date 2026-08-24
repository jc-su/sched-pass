"""Dependency-free finite-window queueing accounting."""

from __future__ import annotations

from typing import Any


def finite_window_littles_law(
    records: list[dict[str, Any]], elapsed_seconds: float
) -> dict[str, Any]:
    """Compute measured arrival/departure accounting and its residual.

    Both sides use the same finite observation window.  The residual is a
    timestamp-consistency check, not an independent queueing performance
    estimate.  Callers must identify the queue scope they actually observe.
    """

    if not records:
        return {
            "method": "finite_window_arrival_departure_accounting",
            "request_count": 0,
            "window_seconds": 0.0,
            "arrival_rate_per_second": 0.0,
            "mean_in_system": 0.0,
            "mean_system_time_seconds": 0.0,
            "residual": 0.0,
        }
    window_start = min(float(record["arrival_seconds"]) for record in records)
    window_end = max(float(record["finished_offset_seconds"]) for record in records)
    window = max(window_end - window_start, float(elapsed_seconds), 1.0e-12)
    events = [(float(record["arrival_seconds"]), 1) for record in records]
    events.extend((float(record["finished_offset_seconds"]), -1) for record in records)
    events.sort(key=lambda event: (event[0], event[1]))
    area = 0.0
    occupancy = 0
    previous = window_start
    for timestamp, delta in events:
        clipped = min(max(timestamp, window_start), window_end)
        if clipped >= previous:
            area += occupancy * (clipped - previous)
        occupancy += delta
        previous = clipped
    mean_system_time = sum(
        float(record["system_time_seconds"]) for record in records
    ) / len(records)
    arrival_rate = len(records) / window
    mean_in_system = area / window
    return {
        "method": "finite_window_arrival_departure_accounting",
        "request_count": len(records),
        "window_start_seconds": window_start,
        "window_end_seconds": window_end,
        "window_seconds": window,
        "arrival_rate_per_second": arrival_rate,
        "mean_in_system": mean_in_system,
        "mean_system_time_seconds": mean_system_time,
        "lhs": mean_in_system,
        "rhs": arrival_rate * mean_system_time,
        "residual": abs(mean_in_system - arrival_rate * mean_system_time),
        "queue_metric_scope": "client_admission_delay; internal engine queue is not exposed",
    }
