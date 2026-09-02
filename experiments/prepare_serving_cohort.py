#!/usr/bin/env python3
"""Select a bounded, explicitly heterogeneous serving cohort from Bailian.

The source trace characterizes production request structure.  Cache placement
and load intensity are machine-specific experimental controls, so this tool
records them as synthetic and never preserves a production-arrival claim for
the diversity-conditioned subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

try:
    from .bailian import (
        demand_trace_digest,
        read_jsonl,
        workload_statistics,
        write_workload,
    )
    from .validate_workload import validate
    from .cache_identity import effective_cached_prefixes
except ImportError:  # pragma: no cover - direct CLI execution
    from bailian import (
        demand_trace_digest,
        read_jsonl,
        workload_statistics,
        write_workload,
    )
    from validate_workload import validate
    from cache_identity import effective_cached_prefixes


ARRIVAL_MODES = (
    "batch_release",
    "calibrated_open_loop",
    "burst",
    "resident_then_burst",
    "trace_scaled",
)
EXTERNAL_SOURCES = ("reuse", "followup")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_followup(row: Mapping[str, Any]) -> bool:
    parent = row.get("parent_chat_id")
    return parent not in (None, "", -1, "-1")


def _candidate(
    row: Mapping[str, Any],
    *,
    role: str,
    block_size: int,
    context_length: int,
    min_input_tokens: int,
    max_input_tokens: int,
    max_output_tokens: int,
    min_resident_output_tokens: int,
    min_external_output_tokens: int,
    min_external_cached_tokens: int,
    min_external_query_rows: int,
    max_external_query_rows: int | None,
    external_source: str,
) -> dict[str, Any] | None:
    input_tokens = int(row["input_length"])
    output_tokens = max(1, int(row["output_length"]))
    if (
        input_tokens < max(2, min_input_tokens)
        or input_tokens > max_input_tokens
        or output_tokens > max_output_tokens
        or input_tokens + output_tokens > context_length
    ):
        return None
    if role == "resident":
        if output_tokens < min_resident_output_tokens:
            return None
        cached = input_tokens - 1
    elif role == "external":
        if output_tokens < min_external_output_tokens:
            return None
        if external_source == "followup" and not _is_followup(row):
            return None
        cached = min(
            input_tokens - 1,
            int(row.get("shared_prefix_blocks", 0)) * block_size,
        )
        if cached <= 0:
            return None
        query_rows = input_tokens - cached
        if (
            cached < min_external_cached_tokens
            or query_rows < min_external_query_rows
            or (
                max_external_query_rows is not None
                and query_rows > max_external_query_rows
            )
        ):
            return None
    else:  # pragma: no cover - internal caller invariant
        raise ValueError(f"unknown serving role {role}")
    candidate = dict(row)
    candidate["source_request_state"] = row.get("request_state")
    candidate["request_state"] = role
    candidate["cached_prefix_tokens"] = cached
    candidate["cohort_active_tokens"] = input_tokens + output_tokens
    return candidate


def _shape_score(rows: Sequence[Mapping[str, Any]]) -> float:
    def spread(values: Sequence[int]) -> float:
        return math.log1p(max(values)) - math.log1p(min(values))

    inputs = [int(row["input_length"]) for row in rows]
    outputs = [max(1, int(row["output_length"])) for row in rows]
    cached = [
        int(row.get("effective_cached_prefix_tokens", row["cached_prefix_tokens"]))
        for row in rows
    ]
    queries = [
        int(row["input_length"])
        - int(row.get("effective_cached_prefix_tokens", row["cached_prefix_tokens"]))
        for row in rows
    ]
    modalities = {str(row.get("modality", "unknown")) for row in rows}
    followup_kinds = {_is_followup(row) for row in rows}
    return (
        spread(inputs)
        + spread(outputs)
        + spread(cached)
        + spread(queries)
        + 0.25 * (len(modalities) - 1)
        + 0.10 * (len(followup_kinds) - 1)
    )


def _cached_union_rows(
    rows: Sequence[Mapping[str, Any]], block_size: int
) -> list[dict[str, Any]]:
    """Attach the exact prefix visible from the initial radix-object union."""

    objects = [
        (
            tuple(str(value) for value in row["hash_ids"]),
            int(row["cached_prefix_tokens"]),
        )
        for row in rows
    ]
    effective = effective_cached_prefixes(
        [
            (
                tuple(str(value) for value in row["hash_ids"]),
                int(row["input_length"]),
            )
            for row in rows
        ],
        objects,
        tokens_per_identity_unit=block_size,
    )
    result: list[dict[str, Any]] = []
    for row, cached in zip(rows, effective, strict=True):
        value = dict(row)
        value["effective_cached_prefix_tokens"] = cached
        result.append(value)
    return result


def _candidate_preserves_external_query_rows(
    selected: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    *,
    block_size: int,
    min_external_query_rows: int,
) -> bool:
    """Check only the new pairwise edges in the cache-union constraint."""

    candidate_object = (
        tuple(str(value) for value in candidate["hash_ids"]),
        int(candidate["cached_prefix_tokens"]),
    )
    external_targets = [row for row in selected if row["request_state"] == "external"]
    for target in external_targets:
        effective = effective_cached_prefixes(
            [
                (
                    tuple(str(value) for value in target["hash_ids"]),
                    int(target["input_length"]),
                )
            ],
            [candidate_object],
            tokens_per_identity_unit=block_size,
        )[0]
        if int(target["input_length"]) - effective < min_external_query_rows:
            return False
    if candidate["request_state"] != "external":
        return True
    candidate_target = (
        tuple(str(value) for value in candidate["hash_ids"]),
        int(candidate["input_length"]),
    )
    objects = [
        (
            tuple(str(value) for value in row["hash_ids"]),
            int(row["cached_prefix_tokens"]),
        )
        for row in selected
    ]
    objects.append(candidate_object)
    effective = effective_cached_prefixes(
        [candidate_target],
        objects,
        tokens_per_identity_unit=block_size,
    )[0]
    return int(candidate["input_length"]) - effective >= min_external_query_rows


def _select_diverse(
    rows: Sequence[Mapping[str, Any]],
    *,
    resident_requests: int,
    external_requests: int,
    context_length: int,
    active_token_budget: int,
    block_size: int,
    min_input_tokens: int,
    max_input_tokens: int,
    max_output_tokens: int,
    min_resident_output_tokens: int,
    min_external_output_tokens: int,
    min_external_cached_tokens: int,
    min_external_query_rows: int,
    max_external_query_rows: int | None,
    external_source: str,
) -> tuple[list[dict[str, Any]], float]:
    if resident_requests <= 0 or external_requests <= 0:
        raise ValueError("serving cohort needs resident and external requests")
    pools = {
        role: [
            candidate
            for row in rows
            if (
                candidate := _candidate(
                    row,
                    role=role,
                    block_size=block_size,
                    context_length=context_length,
                    min_input_tokens=min_input_tokens,
                    max_input_tokens=max_input_tokens,
                    max_output_tokens=max_output_tokens,
                    min_resident_output_tokens=min_resident_output_tokens,
                    min_external_output_tokens=min_external_output_tokens,
                    min_external_cached_tokens=min_external_cached_tokens,
                    min_external_query_rows=min_external_query_rows,
                    max_external_query_rows=max_external_query_rows,
                    external_source=external_source,
                )
            )
            is not None
        ]
        for role in ("resident", "external")
    }
    counts = {"resident": resident_requests, "external": external_requests}
    for role, count in counts.items():
        if len(pools[role]) < count:
            raise ValueError(
                f"source workload has only {len(pools[role])} eligible {role} rows"
            )

    # Start from the minimum-cost feasible cohort.  Deterministic replacement
    # then maximizes joint geometry spread while respecting the exact active
    # token budget and role counts.
    selected: list[dict[str, Any]] = []
    roles: list[str] = []
    used: set[str] = set()
    for role in ("external", "resident"):
        ordered = sorted(
            pools[role],
            key=lambda row: (
                int(row["cohort_active_tokens"]),
                int(row.get("source_row", 0)),
                str(row["request_id"]),
            ),
        )
        for candidate in ordered:
            request_id = str(candidate["request_id"])
            if request_id in used:
                continue
            if not _candidate_preserves_external_query_rows(
                selected,
                candidate,
                block_size=block_size,
                min_external_query_rows=min_external_query_rows,
            ):
                continue
            selected.append(candidate)
            roles.append(role)
            used.add(request_id)
            if roles.count(role) == counts[role]:
                break
        selected_count = roles.count(role)
        if selected_count != counts[role]:
            raise ValueError(
                "deterministic cache-union selection found only "
                f"{selected_count} compatible {role} rows for the requested "
                f"{counts[role]}"
            )
    total = sum(int(row["cohort_active_tokens"]) for row in selected)
    if total > active_token_budget:
        raise ValueError(
            "minimum-cost serving cohort exceeds the active token budget "
            f"({total} > {active_token_budget})"
        )

    score = _shape_score(selected)
    for _ in range(3):
        changed = False
        for index, role in enumerate(roles):
            old = selected[index]
            old_id = str(old["request_id"])
            base_total = total - int(old["cohort_active_tokens"])
            proposal_base = [
                value for position, value in enumerate(selected) if position != index
            ]
            best = old
            best_score = score
            for candidate in pools[role]:
                candidate_id = str(candidate["request_id"])
                if candidate_id != old_id and candidate_id in used:
                    continue
                if not _candidate_preserves_external_query_rows(
                    proposal_base,
                    candidate,
                    block_size=block_size,
                    min_external_query_rows=min_external_query_rows,
                ):
                    continue
                candidate_total = base_total + int(candidate["cohort_active_tokens"])
                if candidate_total > active_token_budget:
                    continue
                proposal = [*selected]
                proposal[index] = candidate
                candidate_score = _shape_score(proposal)
                candidate_key = (
                    candidate_score,
                    -candidate_total,
                    -int(candidate.get("source_row", 0)),
                )
                best_total = base_total + int(best["cohort_active_tokens"])
                best_key = (
                    best_score,
                    -best_total,
                    -int(best.get("source_row", 0)),
                )
                if candidate_key > best_key:
                    best = candidate
                    best_score = candidate_score
            if best is old:
                continue
            used.remove(old_id)
            used.add(str(best["request_id"]))
            total = base_total + int(best["cohort_active_tokens"])
            selected[index] = best
            score = best_score
            changed = True
        if not changed:
            break

    selected.sort(
        key=lambda row: (
            float(row.get("timestamp_seconds", row.get("arrival_seconds", 0.0))),
            int(row.get("source_row", 0)),
        )
    )
    selected = _cached_union_rows(selected, block_size)
    if any(
        row["request_state"] == "external"
        and int(row["input_length"]) - int(row["effective_cached_prefix_tokens"])
        < min_external_query_rows
        for row in selected
    ):
        raise ValueError(
            "selected cache-object union violates the external query-row bound"
        )
    return selected, _shape_score(selected)


def _repeat_cohort(
    rows: Sequence[Mapping[str, Any]], replay_cycles: int
) -> list[dict[str, Any]]:
    """Repeat one exact content working set without claiming independent rows.

    A serving load experiment often needs more request observations than a
    bounded host tier can hold as distinct KV objects.  Replaying the same
    content working set is valid only when that reuse is explicit: request IDs
    remain unique, source IDs and cycle numbers remain recoverable, and source
    ordering is preserved within every cycle.  The manifest records that the
    cycles are not statistically independent source samples.
    """

    if replay_cycles <= 0:
        raise ValueError("serving cohort replay cycles must be positive")
    materialized = [dict(row) for row in rows]
    if replay_cycles == 1:
        return materialized
    if not materialized:
        raise ValueError("cannot repeat an empty serving cohort")

    source_ids = [str(row["request_id"]) for row in materialized]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("base serving cohort request IDs are not unique")
    timestamps = [
        float(row.get("timestamp_seconds", row.get("arrival_seconds", 0.0)))
        for row in materialized
    ]
    if any(right < left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("base serving cohort is not source ordered")
    origin = timestamps[0]
    relative = [value - origin for value in timestamps]
    positive_gaps = [
        right - left for left, right in zip(relative, relative[1:]) if right > left
    ]
    wrap_gap = statistics.median(positive_gaps) if positive_gaps else 1.0
    cycle_span = relative[-1] + wrap_gap

    repeated: list[dict[str, Any]] = []
    for cycle in range(replay_cycles):
        for source_id, order_seconds, row in zip(
            source_ids, relative, materialized, strict=True
        ):
            value = dict(row)
            value["source_request_id"] = source_id
            value["request_id"] = f"{source_id}::replay-cycle-{cycle}"
            value["replay_cycle"] = cycle
            value["cohort_order_seconds"] = order_seconds + cycle * cycle_span
            repeated.append(value)
    return repeated


def _assign_arrivals(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    target_rate: float | None,
    time_scale: float,
    burst_size: int,
) -> dict[str, Any]:
    if mode not in ARRIVAL_MODES:
        raise ValueError(f"unknown serving cohort arrival mode {mode}")
    if time_scale <= 0:
        raise ValueError("time_scale must be positive")

    def source_timestamp(row: Mapping[str, Any]) -> float:
        return float(
            row.get(
                "cohort_order_seconds",
                row.get("timestamp_seconds", row["arrival_seconds"]),
            )
        )

    def controlled_burst_offsets(
        count: int, *, first_group: int = 0
    ) -> tuple[list[float], float]:
        """Return reproducible near-simultaneous arrivals at the target rate.

        Giving every request in a burst the same timestamp leaves coroutine
        wakeup order, framework batch formation, and therefore cache evolution
        undefined.  Spread one burst over exactly one tenth of its group
        interval while retaining ``burst_size / target_rate`` between group
        starts.  The controlled workload remains bursty and preserves its
        long-run offered rate, but repeated causal arms now have one declared
        request order rather than an event-loop race.
        """

        if target_rate is None or target_rate <= 0:
            raise ValueError("burst mode requires a positive target rate")
        if burst_size < 2:
            raise ValueError("burst mode requires burst_size >= 2")
        group_gap = burst_size / target_rate
        intra_burst_spacing = group_gap / (10 * burst_size)
        return (
            [
                (first_group + index // burst_size) * group_gap
                + (index % burst_size) * intra_burst_spacing
                for index in range(count)
            ],
            intra_burst_spacing,
        )

    if mode == "resident_then_burst":
        # This controlled regime first establishes useful resident decode, then
        # injects external requests in bounded bursts. Keep source-time order
        # within each role so only the explicitly declared role boundary is
        # synthetic; content, shape, and within-role trace order are unchanged.
        rows.sort(
            key=lambda row: (
                0 if row.get("request_state") == "resident" else 1,
                source_timestamp(row),
                str(row.get("request_id", "")),
            )
        )
    timestamps = [source_timestamp(row) for row in rows]
    if mode != "resident_then_burst" and any(
        right < left for left, right in zip(timestamps, timestamps[1:])
    ):
        raise ValueError("selected serving cohort is not timestamp ordered")
    if mode == "batch_release":
        offsets = [0.0] * len(rows)
        source = "controlled_batch_release"
    elif mode == "trace_scaled":
        origin = timestamps[0]
        offsets = [(value - origin) * time_scale for value in timestamps]
        source = "selection_conditioned_trace_gaps"
    elif mode == "calibrated_open_loop":
        if target_rate is None or target_rate <= 0:
            raise ValueError("calibrated_open_loop requires a positive target rate")
        gaps = [max(0.0, b - a) for a, b in zip(timestamps, timestamps[1:])]
        positive = [gap for gap in gaps if gap > 0]
        if not positive:
            gaps = [1.0] * max(0, len(rows) - 1)
            positive = gaps
        scale = (1.0 / target_rate) / statistics.fmean(positive)
        offsets = [0.0]
        for gap in gaps:
            offsets.append(offsets[-1] + gap * scale)
        source = "selection_conditioned_gaps_rate_calibrated"
    elif mode == "burst":
        offsets, intra_burst_spacing = controlled_burst_offsets(len(rows))
        source = "controlled_microordered_burst"
    else:
        resident_count = sum(row.get("request_state") == "resident" for row in rows)
        if resident_count <= 0 or resident_count == len(rows):
            raise ValueError(
                "resident_then_burst requires resident and external requests"
            )
        external_offsets, intra_burst_spacing = controlled_burst_offsets(
            len(rows) - resident_count, first_group=1
        )
        offsets = [0.0] * resident_count + external_offsets
        source = "controlled_resident_then_microordered_external_burst"
    for row, offset in zip(rows, offsets):
        row["arrival_seconds"] = float(offset)
        row["arrival_source"] = source
        row.pop("cohort_order_seconds", None)
    return {
        "mode": mode,
        "source": source,
        "time_scale": time_scale,
        "target_rate_per_second": target_rate,
        "has_original_timestamps": all("timestamp_seconds" in row for row in rows),
        "production_arrival_claim": False,
        "selection_conditioned": True,
        "offline_order_is_arrival": mode == "resident_then_burst",
        "burst_size": (
            burst_size if mode in {"burst", "resident_then_burst"} else None
        ),
        "intra_burst_spacing_seconds": (
            intra_burst_spacing
            if mode in {"burst", "resident_then_burst"}
            else None
        ),
        "simultaneous_arrivals": mode == "batch_release",
    }


def _axis(values: Sequence[int]) -> dict[str, Any]:
    return {
        "min": min(values),
        "max": max(values),
        "distinct": len(set(values)),
        "heterogeneous": len(set(values)) > 1,
    }


def _cohort_contract(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    axes = {
        "input_tokens": _axis([int(row["input_length"]) for row in rows]),
        "output_tokens": _axis([int(row["output_length"]) for row in rows]),
        "cached_prefix_tokens": _axis(
            [int(row["cached_prefix_tokens"]) for row in rows]
        ),
        "effective_cached_prefix_tokens": _axis(
            [int(row["effective_cached_prefix_tokens"]) for row in rows]
        ),
        "uncached_query_rows": _axis(
            [
                int(row["input_length"]) - int(row["effective_cached_prefix_tokens"])
                for row in rows
            ]
        ),
    }
    states = sorted({str(row["request_state"]) for row in rows})
    required = (
        states == ["external", "resident"]
        and axes["input_tokens"]["heterogeneous"]
        and axes["cached_prefix_tokens"]["heterogeneous"]
        and axes["uncached_query_rows"]["heterogeneous"]
        and axes["output_tokens"]["heterogeneous"]
    )
    if not required:
        raise ValueError("selected cohort does not satisfy joint shape heterogeneity")
    return {
        "schema": 1,
        "request_states": states,
        "axes": axes,
        "joint_shape_heterogeneity": True,
    }


def build_cohort(
    source_manifest_path: Path,
    *,
    resident_requests: int,
    external_requests: int,
    context_length: int,
    active_token_budget: int,
    min_input_tokens: int = 2,
    max_input_tokens: int | None = None,
    max_output_tokens: int | None = None,
    min_resident_output_tokens: int = 1,
    min_external_output_tokens: int = 1,
    min_external_cached_tokens: int = 1,
    min_external_query_rows: int = 1,
    max_external_query_rows: int | None = None,
    arrival_mode: str,
    target_rate: float | None = None,
    time_scale: float = 1.0,
    burst_size: int = 4,
    external_source: str = "reuse",
    replay_cycles: int = 1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_manifest_path = source_manifest_path.resolve()
    source = validate(source_manifest_path)
    records_path = source_manifest_path.parent / str(source["records_file"])
    rows = read_jsonl(records_path)
    block_size = int(source["block_size"])
    max_input_tokens = (
        context_length - 1 if max_input_tokens is None else max_input_tokens
    )
    max_output_tokens = (
        context_length - 1 if max_output_tokens is None else max_output_tokens
    )
    if (
        context_length <= 1
        or min_input_tokens < 2
        or max_input_tokens <= 0
        or max_output_tokens <= 0
        or min_resident_output_tokens <= 0
        or min_external_output_tokens <= 0
        or min_external_cached_tokens <= 0
        or min_external_query_rows <= 0
        or (
            max_external_query_rows is not None
            and max_external_query_rows < min_external_query_rows
        )
        or min_input_tokens > max_input_tokens
        or max_input_tokens >= context_length
        or max_output_tokens >= context_length
    ):
        raise ValueError("serving cohort token envelopes are invalid")
    if replay_cycles <= 0:
        raise ValueError("serving cohort replay cycles must be positive")
    per_cycle_active_budget = active_token_budget // replay_cycles
    if per_cycle_active_budget <= 0:
        raise ValueError("active token budget cannot cover one replay cycle")
    selected, score = _select_diverse(
        rows,
        resident_requests=resident_requests,
        external_requests=external_requests,
        context_length=context_length,
        active_token_budget=per_cycle_active_budget,
        block_size=block_size,
        min_input_tokens=min_input_tokens,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        min_resident_output_tokens=min_resident_output_tokens,
        min_external_output_tokens=min_external_output_tokens,
        min_external_cached_tokens=min_external_cached_tokens,
        min_external_query_rows=min_external_query_rows,
        max_external_query_rows=max_external_query_rows,
        external_source=external_source,
    )
    base_request_count = len(selected)
    selected = _repeat_cohort(selected, replay_cycles)
    arrival = _assign_arrivals(
        selected,
        mode=arrival_mode,
        target_rate=target_rate,
        time_scale=time_scale,
        burst_size=burst_size,
    )
    contract = _cohort_contract(selected)
    total_active = sum(int(row["cohort_active_tokens"]) for row in selected)
    for row in selected:
        row.pop("cohort_active_tokens", None)
    manifest = {
        "schema": 2,
        "classification": "bailian-structure-replay",
        "source_format": source["source_format"],
        "block_size": block_size,
        "request_count": len(selected),
        "selection": {
            "mode": "diverse_serving_cohort",
            "max_requests": len(selected),
            "source_request_count": int(source["request_count"]),
            "resident_requests": resident_requests * replay_cycles,
            "external_requests": external_requests * replay_cycles,
            "external_source": external_source,
            "context_length": context_length,
            "min_input_tokens": min_input_tokens,
            "max_input_tokens": max_input_tokens,
            "max_output_tokens": max_output_tokens,
            "min_resident_output_tokens": min_resident_output_tokens,
            "min_external_output_tokens": min_external_output_tokens,
            "min_external_cached_tokens": min_external_cached_tokens,
            "min_external_query_rows": min_external_query_rows,
            "max_external_query_rows": max_external_query_rows,
            "active_token_budget": active_token_budget,
            "active_tokens": total_active,
            "algorithm": "deterministic_union_aware_shape_spread_v2",
            "shape_score": score,
            "distribution_representative_claim": False,
            **(
                {
                    "controlled_replay": {
                        "cycles": replay_cycles,
                        "base_request_count": base_request_count,
                        "base_resident_requests": resident_requests,
                        "base_external_requests": external_requests,
                        "content_identity_reused_across_cycles": True,
                        "statistical_independence_claim": False,
                        "schedule": "sequential_cycles_preserving_source_order",
                    }
                }
                if replay_cycles > 1
                else {}
            ),
        },
        "lineage": {
            "source_manifest": source_manifest_path.name,
            "source_manifest_digest": _digest(source_manifest_path),
            "source_records_digest": source["records_digest"],
            "source_demand_trace_digest": source["demand_trace_digest"],
        },
        "arrival": arrival,
        "prompt": dict(source["prompt"]),
        "serving_state": {
            "policy": "diverse_serving_cohort",
            "synthetic": True,
            "source": "controlled_resident_external_assignment",
            "counts": {
                "resident": resident_requests * replay_cycles,
                "external": external_requests * replay_cycles,
            },
        },
        "cache_placement": {
            "source": "exact_hash_reuse_controlled_placement",
            "synthetic": True,
            "identity_field": "cached_prefix_tokens",
            "effective_identity_field": "effective_cached_prefix_tokens",
            "composition": "initial_object_union_longest_common_prefix",
        },
        "cohort_heterogeneity": contract,
        "statistics": workload_statistics(selected),
        "claims": {
            "arrival_is_production_trace": False,
            "prompt_semantics_are_representative": False,
            "hash_block_identity_is_exact": True,
            "offline_row_order_is_arrival": False,
            "serving_state_is_production_cache_state": False,
            "cache_placement_is_production": False,
            "selection_is_distribution_representative": False,
        },
    }
    manifest["demand_trace_digest"] = demand_trace_digest(selected)
    return manifest, selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--resident-requests", type=int, default=2)
    parser.add_argument("--external-requests", type=int, default=6)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--min-input-tokens", type=int, default=2)
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        help="framework request-input envelope; defaults to context_length - 1",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        help="bound trace selection without truncating any completion",
    )
    parser.add_argument("--min-resident-output-tokens", type=int, default=1)
    parser.add_argument("--min-external-output-tokens", type=int, default=1)
    parser.add_argument("--min-external-cached-tokens", type=int, default=1)
    parser.add_argument("--min-external-query-rows", type=int, default=1)
    parser.add_argument(
        "--max-external-query-rows",
        type=int,
        help=(
            "optional pre-execution frontier-opportunity bound; records the "
            "selection as controlled rather than distribution representative"
        ),
    )
    parser.add_argument("--active-token-budget", type=int, required=True)
    parser.add_argument("--arrival-mode", choices=ARRIVAL_MODES, required=True)
    parser.add_argument("--target-rate", type=float)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--burst-size", type=int, default=4)
    parser.add_argument(
        "--replay-cycles",
        type=int,
        default=1,
        help=(
            "repeat the selected exact content working set with unique request "
            "IDs; cycles are recorded as non-independent controlled replay"
        ),
    )
    parser.add_argument("--external-source", choices=EXTERNAL_SOURCES, default="reuse")
    args = parser.parse_args(argv)
    try:
        manifest, records = build_cohort(
            args.source_manifest,
            resident_requests=args.resident_requests,
            external_requests=args.external_requests,
            context_length=args.context_length,
            active_token_budget=args.active_token_budget,
            min_input_tokens=args.min_input_tokens,
            max_input_tokens=args.max_input_tokens,
            max_output_tokens=args.max_output_tokens,
            min_resident_output_tokens=args.min_resident_output_tokens,
            min_external_output_tokens=args.min_external_output_tokens,
            min_external_cached_tokens=args.min_external_cached_tokens,
            min_external_query_rows=args.min_external_query_rows,
            max_external_query_rows=args.max_external_query_rows,
            arrival_mode=args.arrival_mode,
            target_rate=args.target_rate,
            time_scale=args.time_scale,
            burst_size=args.burst_size,
            external_source=args.external_source,
            replay_cycles=args.replay_cycles,
        )
        write_workload(args.manifest, args.records, manifest, records)
        validate(args.manifest.resolve())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"prepare_serving_cohort failed: {error}", file=sys.stderr)
        return 2
    print(args.manifest.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
