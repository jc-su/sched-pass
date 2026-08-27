#!/usr/bin/env python3
"""Produce the RQ0 opportunity report for a normalized workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

try:
    from .bailian import demand_trace_digest, read_jsonl
    from .validate_workload import validate
except ImportError:  # pragma: no cover - direct CLI execution
    from bailian import demand_trace_digest, read_jsonl
    from validate_workload import validate


def _percentile(values: list[float | int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _quantiles(values: list[float | int], fractions: tuple[float, ...]) -> dict[str, float]:
    return {
        f"p{round(fraction * 100)}": _percentile(values, fraction)
        for fraction in fractions
    }


def _followup_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    timestamp_by_request: dict[str, float] = {}
    gaps: list[float] = []
    followups = 0
    unresolved = 0
    for row in rows:
        request_id = str(row["request_id"])
        timestamp = float(row["arrival_seconds"])
        parent = row.get("parent_chat_id")
        if parent not in (None, "", -1, "-1"):
            followups += 1
            parent_timestamp = timestamp_by_request.get(str(parent))
            if parent_timestamp is None or parent_timestamp > timestamp:
                unresolved += 1
            else:
                gaps.append(timestamp - parent_timestamp)
        timestamp_by_request[request_id] = timestamp
    return {
        "requests": followups,
        "request_fraction": followups / len(rows),
        "resolved_parent_links": len(gaps),
        "unresolved_parent_links": unresolved,
        "reuse_gap_seconds": _quantiles(gaps, (0.50, 0.90, 0.99)),
    }


def _block_repetition_statistics(
    rows: list[dict[str, Any]], block_size: int
) -> dict[str, Any]:
    seen: set[str] = set()
    total_occurrences = 0
    repeated_occurrences = 0
    covered_tokens = 0
    input_tokens = 0
    for row in rows:
        hashes = [str(value) for value in row["hash_ids"]]
        total_occurrences += len(hashes)
        repeated_occurrences += sum(value in seen for value in hashes)
        seen.update(hashes)
        request_tokens = int(row["input_length"])
        input_tokens += request_tokens
        covered_tokens += min(request_tokens, len(hashes) * block_size)
    return {
        "total_block_occurrences": total_occurrences,
        "distinct_block_ids": len(seen),
        "cross_request_repeated_occurrences": repeated_occurrences,
        "cross_request_repeat_fraction": (
            repeated_occurrences / total_occurrences if total_occurrences else 0.0
        ),
        "input_token_hash_coverage": (
            covered_tokens / input_tokens if input_tokens else 0.0
        ),
    }


def analyze(path: Path) -> dict[str, Any]:
    path = path.resolve()
    manifest = validate(path)
    records_path = path.parent / str(manifest["records_file"])
    rows = read_jsonl(records_path)
    input_lengths = [int(row["input_length"]) for row in rows]
    output_lengths = [int(row["output_length"]) for row in rows]
    shared = [int(row.get("shared_prefix_blocks", 0)) for row in rows]
    candidate_blocks = [
        math.ceil(int(row["input_length"]) / int(manifest["block_size"]))
        for row in rows
    ]
    arrivals = [float(row["arrival_seconds"]) for row in rows]
    gaps = [right - left for left, right in zip(arrivals, arrivals[1:])]
    positive_gaps = [gap for gap in gaps if gap > 0]
    states = [
        str(row["request_state"])
        for row in rows
        if row.get("request_state") is not None
    ]
    state_counts = {state: states.count(state) for state in sorted(set(states))}
    entropy = 0.0
    if states:
        for count in state_counts.values():
            probability = count / len(states)
            entropy -= probability * math.log2(probability)
    mean_gap = statistics.fmean(positive_gaps) if positive_gaps else 0.0
    all_gap_mean = statistics.fmean(gaps) if gaps else 0.0
    block_size = int(manifest["block_size"])
    reused_prefix_tokens = [value * block_size for value in shared if value > 0]
    trace_span = arrivals[-1] - arrivals[0] if len(arrivals) > 1 else 0.0
    block_repetition = _block_repetition_statistics(rows, block_size)
    report = {
        "schema": 2,
        "classification": "bailian-rq0-opportunity-report",
        "provenance": {
            "manifest": str(path),
            "manifest_digest": hashlib.sha256(path.read_bytes()).hexdigest(),
            "records_digest": manifest["records_digest"],
            "demand_trace_digest": demand_trace_digest(rows),
        },
        "claims": manifest["claims"],
        "arrival": {
            **manifest["arrival"],
            "request_count": len(rows),
            "zero_gap_fraction": gaps.count(0.0) / len(gaps) if gaps else 0.0,
            "positive_gap_mean_seconds": mean_gap,
            "positive_gap_cv": (
                statistics.pstdev(positive_gaps) / mean_gap
                if len(positive_gaps) > 1 and mean_gap > 0
                else 0.0
            ),
            "trace_span_seconds": trace_span,
            "mean_request_rate_per_second": (
                len(rows) / trace_span if trace_span > 0 else 0.0
            ),
            "interarrival_cv": (
                statistics.pstdev(gaps) / all_gap_mean
                if len(gaps) > 1 and all_gap_mean > 0
                else 0.0
            ),
        },
        "lengths": {
            "input_tokens": {
                "min": min(input_lengths),
                **_quantiles(input_lengths, (0.50, 0.90, 0.99)),
                "mean": statistics.fmean(input_lengths),
                "max": max(input_lengths),
                "share_at_least_4k": sum(value >= 4096 for value in input_lengths)
                / len(rows),
                "share_at_least_8k": sum(value >= 8192 for value in input_lengths)
                / len(rows),
                "share_at_least_16k": sum(value >= 16384 for value in input_lengths)
                / len(rows),
            },
            "output_tokens": {
                "min": min(output_lengths),
                **_quantiles(output_lengths, (0.50, 0.90, 0.99)),
                "mean": statistics.fmean(output_lengths),
                "max": max(output_lengths),
            },
        },
        "prefix_reuse": {
            "requests_with_shared_prefix": sum(value > 0 for value in shared),
            "shared_request_fraction": sum(value > 0 for value in shared) / len(rows),
            "mean_shared_prefix_blocks": statistics.fmean(shared),
            "max_shared_prefix_blocks": max(shared),
            "reused_prefix_tokens": _quantiles(
                reused_prefix_tokens, (0.50, 0.90, 0.99)
            ),
            "exact_prefix_block_occurrence_fraction": (
                sum(shared) / block_repetition["total_block_occurrences"]
                if block_repetition["total_block_occurrences"]
                else 0.0
            ),
        },
        "sessions": _followup_statistics(rows),
        "block_identity": block_repetition,
        "request_shape_heterogeneity": {
            "distinct_input_lengths": len(set(input_lengths)),
            "distinct_output_lengths": len(set(output_lengths)),
            "distinct_input_output_prefix_shapes": len(
                set(zip(input_lengths, output_lengths, shared))
            ),
            "joint_shape_heterogeneous": len(
                set(zip(input_lengths, output_lengths, shared))
            )
            > 1,
        },
        "state_heterogeneity": {
            "counts": state_counts,
            "annotated_request_fraction": len(states) / len(rows),
            "shannon_entropy_bits": entropy,
        },
        "exact_demand_shape": {
            "candidate_kv_blocks": {
                "min": min(candidate_blocks),
                "median": statistics.median(candidate_blocks),
                "mean": statistics.fmean(candidate_blocks),
                "max": max(candidate_blocks),
                "total": sum(candidate_blocks),
            },
            "shared_prefix_blocks_total": sum(shared),
        },
        "compute_transfer_regime": {
            "status": "trace_only_not_identifiable",
            "reason": (
                "an anonymized structure trace lacks model KV byte geometry, "
                "measured layer compute, and qualified tier bandwidth"
            ),
            "measured_in": ["native_tier_report", "serving_profile"],
        },
        "semantic_prompt_claim": manifest["prompt"][
            "semantic_representativeness_claim"
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.manifest)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
