#!/usr/bin/env python3
"""Validate a normalized Bailian workload and its provenance claims."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .bailian import demand_trace_digest, read_jsonl
except ImportError:  # pragma: no cover - supports direct CLI execution
    from bailian import demand_trace_digest, read_jsonl


def validate(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != 1 or manifest.get("classification") != "bailian-structure-replay":
        raise ValueError("unsupported workload manifest")
    records_path = path.parent / str(manifest.get("records_file", ""))
    if not records_path.is_file():
        raise ValueError("workload records file is missing")
    digest = hashlib.sha256(records_path.read_bytes()).hexdigest()
    if digest != manifest.get("records_digest"):
        raise ValueError("workload records digest mismatch")
    rows = read_jsonl(records_path)
    if len(rows) != manifest.get("request_count") or not rows:
        raise ValueError("workload request count is inconsistent")
    selection = manifest.get("selection")
    if not isinstance(selection, dict) or selection.get("mode") not in {"all_rows", "source_prefix"}:
        raise ValueError("workload manifest lacks selection provenance")
    source_count = selection.get("source_request_count")
    if not isinstance(source_count, int) or source_count < len(rows):
        raise ValueError("workload selection source count is inconsistent")
    selected_max = selection.get("max_requests")
    if selection["mode"] == "all_rows" and selected_max is not None:
        raise ValueError("all_rows selection cannot carry max_requests")
    if selection["mode"] == "source_prefix" and selected_max != len(rows):
        raise ValueError("source_prefix selection does not match request count")
    required = {"request_id", "input_length", "output_length", "hash_ids", "arrival_seconds"}
    if any(not required <= set(row) for row in rows):
        raise ValueError("workload rows do not contain the normalized demand fields")
    request_ids = [str(row["request_id"]) for row in rows]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("workload request IDs are not unique")
    block_size = int(manifest["block_size"])
    if block_size <= 0:
        raise ValueError("workload block size must be positive")
    arrivals = [float(row["arrival_seconds"]) for row in rows]
    if any(value < 0 for value in arrivals) or any(right < left for left, right in zip(arrivals, arrivals[1:])):
        raise ValueError("arrival offsets are not monotonic and non-negative")
    arrival = manifest["arrival"]
    if not arrival["production_arrival_claim"] and manifest["claims"]["arrival_is_production_trace"]:
        raise ValueError("manifest makes a false production-arrival claim")
    if arrival["mode"] == "batch_release" and any(value != 0 for value in arrivals):
        raise ValueError("batch release must publish all requests at time zero")
    if arrival["mode"] == "trace" and not arrival["has_original_timestamps"]:
        raise ValueError("trace mode lacks original timestamps")
    state = manifest.get("serving_state")
    if not isinstance(state, dict) or not isinstance(state.get("policy"), str):
        raise ValueError("workload manifest lacks serving-state provenance")
    request_states = {row.get("request_state") for row in rows}
    if any(value is not None and value not in {"resident", "external"} for value in request_states):
        raise ValueError("workload request_state must be resident or external")
    if state.get("synthetic"):
        if request_states != {"resident", "external"}:
            raise ValueError("synthetic serving state must contain resident and external rows")
        if manifest.get("claims", {}).get("serving_state_is_production_cache_state"):
            raise ValueError("synthetic serving state was marked as production state")
    prompt = manifest["prompt"]
    for row in rows:
        if int(row["input_length"]) <= 0 or int(row["output_length"]) < 0:
            raise ValueError("workload lengths are out of range")
        if not isinstance(row["hash_ids"], list) or any(not str(value) for value in row["hash_ids"]):
            raise ValueError("workload hash_ids are not a non-empty-string list")
        if prompt["enabled"] and int(row["prompt_token_count"]) != int(row["input_length"]):
            raise ValueError("synthetic prompt token count does not match input length")
        if len(row["hash_ids"]) > (int(row["input_length"]) + block_size - 1) // block_size:
            raise ValueError("hash prefix is longer than the request")
    expected_demand_digest = demand_trace_digest(rows)
    if manifest.get("demand_trace_digest") != expected_demand_digest:
        raise ValueError("workload demand trace digest mismatch")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    validate(args.manifest.resolve())
    print("bailian_workload=valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
