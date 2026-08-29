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
except ImportError:  # pragma: no cover - direct CLI execution
    from bailian import (
        demand_trace_digest,
        read_jsonl,
        workload_statistics,
        write_workload,
    )
    from validate_workload import validate


ARRIVAL_MODES = ("batch_release", "calibrated_open_loop", "burst", "trace_scaled")
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
    min_external_cached_tokens: int,
    min_external_query_rows: int,
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
        if external_source == "followup" and not _is_followup(row):
            return None
        cached = min(
            input_tokens - 1,
            int(row.get("shared_prefix_blocks", 0)) * block_size,
        )
        if cached <= 0:
            return None
        if (
            cached < min_external_cached_tokens
            or input_tokens - cached < min_external_query_rows
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
    cached = [int(row["cached_prefix_tokens"]) for row in rows]
    queries = [
        int(row["input_length"]) - int(row["cached_prefix_tokens"])
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
    min_external_cached_tokens: int,
    min_external_query_rows: int,
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
                    min_external_cached_tokens=min_external_cached_tokens,
                    min_external_query_rows=min_external_query_rows,
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
            selected.append(candidate)
            roles.append(role)
            used.add(request_id)
            if roles.count(role) == counts[role]:
                break
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
            best = old
            best_score = score
            for candidate in pools[role]:
                candidate_id = str(candidate["request_id"])
                if candidate_id != old_id and candidate_id in used:
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
    return selected, score


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
    timestamps = [
        float(row.get("timestamp_seconds", row["arrival_seconds"])) for row in rows
    ]
    if any(right < left for left, right in zip(timestamps, timestamps[1:])):
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
    else:
        if target_rate is None or target_rate <= 0:
            raise ValueError("burst mode requires a positive target rate")
        if burst_size < 2:
            raise ValueError("burst mode requires burst_size >= 2")
        group_gap = burst_size / target_rate
        offsets = [(index // burst_size) * group_gap for index in range(len(rows))]
        source = "controlled_burst"
    for row, offset in zip(rows, offsets):
        row["arrival_seconds"] = float(offset)
        row["arrival_source"] = source
    return {
        "mode": mode,
        "source": source,
        "time_scale": time_scale,
        "target_rate_per_second": target_rate,
        "has_original_timestamps": all("timestamp_seconds" in row for row in rows),
        "production_arrival_claim": False,
        "selection_conditioned": True,
        "offline_order_is_arrival": False,
        "burst_size": burst_size if mode == "burst" else None,
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
        "uncached_query_rows": _axis(
            [
                int(row["input_length"]) - int(row["cached_prefix_tokens"])
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
    min_external_cached_tokens: int = 1,
    min_external_query_rows: int = 1,
    arrival_mode: str,
    target_rate: float | None = None,
    time_scale: float = 1.0,
    burst_size: int = 4,
    external_source: str = "reuse",
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
        or min_external_cached_tokens <= 0
        or min_external_query_rows <= 0
        or min_input_tokens > max_input_tokens
        or max_input_tokens >= context_length
        or max_output_tokens >= context_length
    ):
        raise ValueError("serving cohort token envelopes are invalid")
    selected, score = _select_diverse(
        rows,
        resident_requests=resident_requests,
        external_requests=external_requests,
        context_length=context_length,
        active_token_budget=active_token_budget,
        block_size=block_size,
        min_input_tokens=min_input_tokens,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        min_resident_output_tokens=min_resident_output_tokens,
        min_external_cached_tokens=min_external_cached_tokens,
        min_external_query_rows=min_external_query_rows,
        external_source=external_source,
    )
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
            "resident_requests": resident_requests,
            "external_requests": external_requests,
            "external_source": external_source,
            "context_length": context_length,
            "min_input_tokens": min_input_tokens,
            "max_input_tokens": max_input_tokens,
            "max_output_tokens": max_output_tokens,
            "min_resident_output_tokens": min_resident_output_tokens,
            "min_external_cached_tokens": min_external_cached_tokens,
            "min_external_query_rows": min_external_query_rows,
            "active_token_budget": active_token_budget,
            "active_tokens": total_active,
            "algorithm": "deterministic_joint_shape_spread_v1",
            "shape_score": score,
            "distribution_representative_claim": False,
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
                "resident": resident_requests,
                "external": external_requests,
            },
        },
        "cache_placement": {
            "source": "exact_hash_reuse_controlled_placement",
            "synthetic": True,
            "identity_field": "cached_prefix_tokens",
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
    parser.add_argument("--min-external-cached-tokens", type=int, default=1)
    parser.add_argument("--min-external-query-rows", type=int, default=1)
    parser.add_argument("--active-token-budget", type=int, required=True)
    parser.add_argument("--arrival-mode", choices=ARRIVAL_MODES, required=True)
    parser.add_argument("--target-rate", type=float)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--burst-size", type=int, default=4)
    parser.add_argument(
        "--external-source", choices=EXTERNAL_SOURCES, default="reuse"
    )
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
            min_external_cached_tokens=args.min_external_cached_tokens,
            min_external_query_rows=args.min_external_query_rows,
            arrival_mode=args.arrival_mode,
            target_rate=args.target_rate,
            time_scale=args.time_scale,
            burst_size=args.burst_size,
            external_source=args.external_source,
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
