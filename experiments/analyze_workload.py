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
    states = [str(row["request_state"]) for row in rows if row.get("request_state") is not None]
    state_counts = {state: states.count(state) for state in sorted(set(states))}
    entropy = 0.0
    if states:
        for count in state_counts.values():
            probability = count / len(states)
            entropy -= probability * math.log2(probability)
    mean_gap = statistics.fmean(positive_gaps) if positive_gaps else 0.0
    report = {
        "schema": 1,
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
        },
        "lengths": {
            "input_tokens": {
                "min": min(input_lengths),
                "median": statistics.median(input_lengths),
                "mean": statistics.fmean(input_lengths),
                "max": max(input_lengths),
            },
            "output_tokens": {
                "min": min(output_lengths),
                "median": statistics.median(output_lengths),
                "mean": statistics.fmean(output_lengths),
                "max": max(output_lengths),
            },
        },
        "prefix_reuse": {
            "requests_with_shared_prefix": sum(value > 0 for value in shared),
            "shared_request_fraction": sum(value > 0 for value in shared) / len(rows),
            "mean_shared_prefix_blocks": statistics.fmean(shared),
            "max_shared_prefix_blocks": max(shared),
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
        "semantic_prompt_claim": manifest["prompt"]["semantic_representativeness_claim"],
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
