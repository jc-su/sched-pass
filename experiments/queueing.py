"""Dependency-free finite-window queueing accounting."""

from __future__ import annotations

from typing import Any


def modeled_blocked_cohort_accounting(
    blocked_units: int, pending_window_us: float
) -> dict[str, Any]:
    """Account for a synthetic blocked cohort over one availability window.

    All units enter the blocked population at the beginning of the modeled
    window and become available at deterministic interval midpoints.  The
    occupancy and residence time are integrated from the release events.  This
    is deterministic cohort accounting, not a Little's-law test and not a
    stationary queueing measurement.
    """

    if blocked_units < 0:
        raise ValueError("blocked unit count cannot be negative")
    if pending_window_us < 0:
        raise ValueError("pending window cannot be negative")

    release_count = int(blocked_units)
    window_us = float(pending_window_us)
    if release_count == 0 or window_us == 0.0:
        return {
            "method": "finite_window_synthetic_release_accounting",
            "release_process": "uniform_midpoint_over_availability_window",
            "pending_release_count": release_count,
            "pending_window_us": window_us,
            "pending_area_unit_us": 0.0,
            "release_rate_per_second": 0.0,
            "mean_pending_units": 0.0,
            "mean_pending_us": 0.0,
            "interpretation": "cohort_accounting_not_stationary_queueing",
        }

    release_offsets_us = (
        window_us * (index + 0.5) / release_count for index in range(release_count)
    )
    pending_area_unit_us = sum(release_offsets_us)
    completion_rate = release_count / window_us * 1_000_000
    mean_pending_units = pending_area_unit_us / window_us
    mean_pending_us = pending_area_unit_us / release_count
    return {
        "method": "finite_window_synthetic_release_accounting",
        "release_process": "uniform_midpoint_over_availability_window",
        "pending_release_count": release_count,
        "pending_window_us": window_us,
        "pending_area_unit_us": pending_area_unit_us,
        "release_rate_per_second": completion_rate,
        "mean_pending_units": mean_pending_units,
        "mean_pending_us": mean_pending_us,
        "interpretation": "cohort_accounting_not_stationary_queueing",
    }


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
