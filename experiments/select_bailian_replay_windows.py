#!/usr/bin/env python3
"""Pre-register Bailian replay windows by mechanism opportunity, not outcome.

The census scans disjoint measured windows in source order. It reports both a
source-order representative sample and a trace-only stress sample for each
stratum. Neither selection reads latency, throughput, cache placement, or any
NTA counter; observed host/device state remains an output of the serving run.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import math
from pathlib import Path
import sys
from typing import Any

try:
    from .atomic_io import atomic_write_json
    from .bailian import input_page_ids, read_jsonl
    from .bailian_replay import build_replay_window, opportunity_stratum
    from .validate_workload import validate as validate_workload
except ImportError:  # pragma: no cover - direct CLI execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiments.atomic_io import atomic_write_json
    from experiments.bailian import input_page_ids, read_jsonl
    from experiments.bailian_replay import build_replay_window, opportunity_stratum
    from experiments.validate_workload import validate as validate_workload


def _descriptor(window) -> dict[str, Any]:
    opportunity = dict(window.metadata["mechanism_opportunity"])
    return {
        "measured_start": int(window.metadata["measured_start"]),
        "measured_end": int(window.metadata["measured_end"]),
        "source_arrival_span_seconds": float(
            window.metadata["source_arrival_span_seconds"]
        ),
        "measured_arrival_span_seconds": float(
            window.metadata["measured_arrival_span_seconds"]
        ),
        "measured_followup_requests": int(
            window.metadata["measured_followup_requests"]
        ),
        "measured_axes": window.metadata["measured_axes"],
        "selected_demand_trace_digest": str(
            window.metadata["selected_demand_trace_digest"]
        ),
        "opportunity": opportunity,
        "stratum": opportunity_stratum(opportunity),
    }


def _stress_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    opportunity = candidate["opportunity"]
    # Every term is fixed before execution and monotone in the hypothesized
    # transfer/overlap opportunity. Source index breaks ties deterministically.
    return (
        -float(opportunity["tier_overlap_candidate_prefix_tokens"]),
        -float(opportunity["input_capacity_crossing_prefix_tokens"]),
        -float(opportunity["maximum_input_pressure_capacity_ratio"]),
        -float(opportunity["replayable_prefix_tokens"]),
        float(candidate["measured_start"]),
    )


def select_windows(
    manifest_path: Path,
    *,
    warmup_requests: int,
    measured_requests: int,
    stride_requests: int,
    context_length: int,
    input_margin_tokens: int,
    input_adapter_tokens: int,
    max_output_tokens: int,
    output_length_scale: float,
    device_token_capacity: int,
    consumer_wave_tokens: int,
    per_stratum: int,
) -> dict[str, Any]:
    """Build the complete candidate census and deterministic selections."""

    positive = (
        measured_requests,
        stride_requests,
        context_length,
        max_output_tokens,
        device_token_capacity,
        consumer_wave_tokens,
        per_stratum,
    )
    if warmup_requests < 0 or any(value <= 0 for value in positive):
        raise ValueError("window-selection counts and capacities must be positive")
    if not math.isfinite(output_length_scale) or output_length_scale <= 0.0:
        raise ValueError("window-selection output scale must be finite and positive")
    path = manifest_path.resolve()
    manifest = validate_workload(path)
    if manifest["serving_state"].get("policy") != "preserve_absent":
        raise ValueError("natural replay selection requires absent source placement")
    records_path = path.parent / str(manifest["records_file"])
    rows = read_jsonl(records_path)
    block_size = int(manifest["block_size"])
    # A full census revisits every source row in several neighboring windows.
    # Materialize exact logical pages once so each candidate does not allocate
    # and stringify the same potentially long block sequence again.  The cache
    # contains tuples of references to the normalized identities, not another
    # copy of their strings.
    page_id_rows = tuple(
        input_page_ids(row, block_size=block_size) for row in rows
    )
    candidates: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    last_start = len(rows) - measured_requests
    for measured_start in range(
        warmup_requests, last_start + 1, stride_requests
    ):
        try:
            window = build_replay_window(
                rows,
                measured_start=measured_start,
                warmup_requests=warmup_requests,
                measured_requests=measured_requests,
                context_length=context_length,
                input_margin_tokens=input_margin_tokens,
                input_adapter_tokens=input_adapter_tokens,
                max_output_tokens=max_output_tokens,
                output_length_scale=output_length_scale,
                device_token_capacity=device_token_capacity,
                consumer_wave_tokens=consumer_wave_tokens,
                page_id_rows=page_id_rows,
            )
        except ValueError as error:
            reason = str(error).split(":", 1)[0]
            rejected[reason] += 1
            continue
        candidates.append(_descriptor(window))
    if not candidates:
        raise ValueError("no replay window satisfies the configured envelope")

    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_stratum[str(candidate["stratum"])].append(candidate)
    representative = {
        name: values[:per_stratum]
        for name, values in sorted(by_stratum.items())
    }
    stress = {
        name: sorted(values, key=_stress_key)[:per_stratum]
        for name, values in sorted(by_stratum.items())
    }
    census = {
        name: {
            "windows": len(values),
            "fraction": len(values) / len(candidates),
        }
        for name, values in sorted(by_stratum.items())
    }
    return {
        "schema": 1,
        "classification": "bailian-natural-replay-opportunity-census",
        "provenance": {
            "manifest": str(path),
            "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "records_digest": str(manifest["records_digest"]),
            "demand_trace_digest": str(manifest["demand_trace_digest"]),
        },
        "selection_contract": {
            "uses_measured_performance": False,
            "uses_observed_cache_placement": False,
            "representative": "first_n_in_source_order_per_stratum",
            "stress": "top_n_by_preregistered_trace_opportunity_per_stratum",
            "serving_run_must_report_observed_placement": True,
        },
        "configuration": {
            "warmup_requests": warmup_requests,
            "measured_requests": measured_requests,
            "stride_requests": stride_requests,
            "context_length": context_length,
            "input_margin_tokens": input_margin_tokens,
            "input_adapter_tokens": input_adapter_tokens,
            "max_output_tokens": max_output_tokens,
            "output_length_scale": output_length_scale,
            "device_token_capacity": device_token_capacity,
            "consumer_wave_tokens": consumer_wave_tokens,
            "per_stratum": per_stratum,
        },
        "candidate_windows": len(candidates),
        "rejected_windows": sum(rejected.values()),
        "rejected_reasons": dict(sorted(rejected.items())),
        "census": census,
        "representative_windows": representative,
        "stress_windows": stress,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--warmup-requests", type=int, default=64)
    parser.add_argument("--measured-requests", type=int, default=32)
    parser.add_argument("--stride-requests", type=int)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--input-margin-tokens", type=int, default=8)
    parser.add_argument("--input-adapter-tokens", type=int, default=1)
    parser.add_argument("--max-output-tokens", type=int, default=32)
    parser.add_argument("--output-length-scale", type=float, default=0.1)
    parser.add_argument("--device-token-capacity", type=int, required=True)
    parser.add_argument("--consumer-wave-tokens", type=int, default=2048)
    parser.add_argument("--per-stratum", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stride = args.stride_requests or args.measured_requests
    report = select_windows(
        args.manifest,
        warmup_requests=args.warmup_requests,
        measured_requests=args.measured_requests,
        stride_requests=stride,
        context_length=args.context_length,
        input_margin_tokens=args.input_margin_tokens,
        input_adapter_tokens=args.input_adapter_tokens,
        max_output_tokens=args.max_output_tokens,
        output_length_scale=args.output_length_scale,
        device_token_capacity=args.device_token_capacity,
        consumer_wave_tokens=args.consumer_wave_tokens,
        per_stratum=args.per_stratum,
    )
    atomic_write_json(args.output, report)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
