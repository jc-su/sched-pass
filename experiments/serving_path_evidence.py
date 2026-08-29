"""Physical execution evidence derived from timed serving counters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


EXERCISED_PATHS = (
    "native_demand_sm",
    "prefetch_sm",
    "prefetch_copy_engine",
    "prefetch_hybrid",
    "partial_consumer",
)


def summarize_transport_execution(
    stats: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize physical execution, never planner intent, for a timed window."""

    def total(name: str) -> int:
        result = 0
        for entry in stats:
            value = entry.get(name, 0)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"serving counter {name!r} is not nonnegative")
            result += value
        return result

    native_sm_bytes = total("native_demand_sm_bytes")
    prefetch_sm_bytes = total("sm_mover_bytes")
    prefetch_copy_bytes = total("copy_engine_bytes")
    prefetch_copy_operations = total("copy_engine_operations")
    prefetch_copy_submissions = total("copy_engine_submissions")
    hybrid_waves = total("hybrid_parallel_waves")
    progressive_layers = total("progressive_consumer_layers")
    return {
        "schema": 1,
        "native_demand_sm": {
            "bytes": native_sm_bytes,
            "layers": total("native_external_attention_launches"),
            "exercised": native_sm_bytes > 0,
        },
        "prefetch_sm": {
            "bytes": prefetch_sm_bytes,
            "exercised": prefetch_sm_bytes > 0,
        },
        "prefetch_copy_engine": {
            "bytes": prefetch_copy_bytes,
            "operations": prefetch_copy_operations,
            "submissions": prefetch_copy_submissions,
            "exercised": (
                prefetch_copy_bytes > 0
                and prefetch_copy_operations > 0
                and prefetch_copy_submissions > 0
            ),
        },
        "prefetch_hybrid": {
            "parallel_waves": hybrid_waves,
            "exercised": (
                prefetch_sm_bytes > 0
                and prefetch_copy_bytes > 0
                and hybrid_waves > 0
            ),
        },
        "partial_consumer": {
            "layers": progressive_layers,
            "exact_window_layers": total("exact_resume_window_layers"),
            "exercised": progressive_layers > 0,
        },
        "frontier": {
            "native_layers": total("native_external_attention_launches"),
            "ready_stock_layers": total("stock_ready_external_attention_launches"),
            "progress_rounds": total("host_progress_rounds"),
        },
    }


def require_exercised_paths(
    stats: Sequence[Mapping[str, Any]], required: Sequence[str]
) -> dict[str, Any]:
    unknown = [name for name in required if name not in EXERCISED_PATHS]
    if unknown:
        raise ValueError("unknown serving execution path(s): " + ", ".join(unknown))
    evidence = summarize_transport_execution(stats)
    missing = [
        name
        for name in dict.fromkeys(required)
        if not bool(evidence[name]["exercised"])
    ]
    if missing:
        raise ValueError(
            "timed serving arm did not exercise required physical path(s): "
            + ", ".join(missing)
        )
    return evidence


def require_frontier_shape(
    evidence: Mapping[str, Any],
    *,
    native_layers: int | None,
    ready_stock_layers: int | None,
    progressive_layers: int | None,
) -> None:
    required = {
        "native_layers": native_layers,
        "ready_stock_layers": ready_stock_layers,
        "progressive_layers": progressive_layers,
    }
    if any(
        value is not None
        and (not isinstance(value, int) or isinstance(value, bool) or value < 0)
        for value in required.values()
    ):
        raise ValueError("required serving frontier counts must be nonnegative")
    frontier = evidence.get("frontier")
    partial = evidence.get("partial_consumer")
    if not isinstance(frontier, Mapping) or not isinstance(partial, Mapping):
        raise ValueError("serving execution evidence has no frontier shape")
    observed = {
        "native_layers": frontier.get("native_layers"),
        "ready_stock_layers": frontier.get("ready_stock_layers"),
        "progressive_layers": partial.get("layers"),
    }
    mismatches = {
        name: {"required": value, "observed": observed[name]}
        for name, value in required.items()
        if value is not None and observed[name] != value
    }
    if mismatches:
        raise ValueError(f"timed serving frontier shape mismatch: {mismatches}")
