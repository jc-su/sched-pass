#!/usr/bin/env python3
"""Validate a normalized Bailian workload and its provenance claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    from .bailian import demand_trace_digest, read_jsonl
except ImportError:  # pragma: no cover - supports direct CLI execution
    from bailian import demand_trace_digest, read_jsonl


def validate(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read workload manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError("workload manifest must be a JSON object")
    if (
        manifest.get("schema") != 1
        or manifest.get("classification") != "bailian-structure-replay"
    ):
        raise ValueError("unsupported workload manifest")
    records_name = manifest.get("records_file")
    if not isinstance(records_name, str) or not records_name:
        raise ValueError("workload manifest has no records_file")
    records_relative = Path(records_name)
    if records_relative.is_absolute() or ".." in records_relative.parts:
        raise ValueError("workload records_file escapes the manifest directory")
    records_path = (path.parent / records_relative).resolve()
    try:
        records_path.relative_to(path.parent.resolve())
    except ValueError as error:
        raise ValueError(
            "workload records_file escapes the manifest directory"
        ) from error
    if not records_path.is_file():
        raise ValueError("workload records file is missing")
    digest_state = hashlib.sha256()
    try:
        with records_path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest_state.update(block)
    except OSError as error:
        raise ValueError(f"cannot read workload records: {error}") from error
    digest = digest_state.hexdigest()
    if digest != manifest.get("records_digest"):
        raise ValueError("workload records digest mismatch")
    try:
        rows = read_jsonl(records_path)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot decode workload records: {error}") from error
    request_count = manifest.get("request_count")
    if (
        not isinstance(request_count, int)
        or isinstance(request_count, bool)
        or request_count <= 0
        or len(rows) != request_count
    ):
        raise ValueError("workload request count is inconsistent")
    selection = manifest.get("selection")
    if not isinstance(selection, dict) or selection.get("mode") not in {
        "all_rows",
        "source_prefix",
    }:
        raise ValueError("workload manifest lacks selection provenance")
    source_count = selection.get("source_request_count")
    if (
        not isinstance(source_count, int)
        or isinstance(source_count, bool)
        or source_count < len(rows)
    ):
        raise ValueError("workload selection source count is inconsistent")
    selected_max = selection.get("max_requests")
    if selection["mode"] == "all_rows" and selected_max is not None:
        raise ValueError("all_rows selection cannot carry max_requests")
    if selection["mode"] == "all_rows" and source_count != len(rows):
        raise ValueError("all_rows selection does not cover the source")
    if selection["mode"] == "source_prefix":
        if (
            not isinstance(selected_max, int)
            or isinstance(selected_max, bool)
            or selected_max <= 0
            or selected_max != len(rows)
        ):
            raise ValueError("source_prefix selection does not match request count")
    required = {
        "request_id",
        "input_length",
        "output_length",
        "hash_ids",
        "arrival_seconds",
    }
    if any(not required <= set(row) for row in rows):
        raise ValueError("workload rows do not contain the normalized demand fields")
    request_ids = [row["request_id"] for row in rows]
    if any(not isinstance(value, str) or not value for value in request_ids):
        raise ValueError("workload request IDs must be non-empty strings")
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("workload request IDs are not unique")
    block_size = manifest.get("block_size")
    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size <= 0
    ):
        raise ValueError("workload block size must be positive")
    arrivals: list[float] = []
    for row in rows:
        value = row["arrival_seconds"]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError("workload arrival offsets must be finite and non-negative")
        arrivals.append(float(value))
    if any(right < left for left, right in zip(arrivals, arrivals[1:])):
        raise ValueError("arrival offsets are not monotonic")
    arrival = manifest["arrival"]
    if not isinstance(arrival, dict):
        raise ValueError("workload manifest lacks arrival provenance")
    for field in ("mode", "source"):
        if not isinstance(arrival.get(field), str) or not arrival[field]:
            raise ValueError(f"workload arrival lacks {field}")
    if not isinstance(arrival.get("production_arrival_claim"), bool):
        raise ValueError("workload arrival claim is not boolean")
    claims = manifest.get("claims")
    if not isinstance(claims, dict):
        raise ValueError("workload manifest lacks claims")
    if not isinstance(claims.get("arrival_is_production_trace"), bool):
        raise ValueError("workload production-arrival claim is not boolean")
    if claims["arrival_is_production_trace"] != arrival["production_arrival_claim"]:
        raise ValueError("workload arrival claims disagree")
    if arrival["mode"] == "batch_release" and any(value != 0 for value in arrivals):
        raise ValueError("batch release must publish all requests at time zero")
    if (
        arrival["mode"] == "trace"
        and arrival.get("has_original_timestamps") is not True
    ):
        raise ValueError("trace mode lacks original timestamps")
    state = manifest.get("serving_state")
    if (
        not isinstance(state, dict)
        or not isinstance(state.get("policy"), str)
        or not state["policy"]
        or not isinstance(state.get("synthetic"), bool)
    ):
        raise ValueError("workload manifest lacks serving-state provenance")
    request_states = {row.get("request_state") for row in rows}
    if any(
        value is not None and value not in {"resident", "external"}
        for value in request_states
    ):
        raise ValueError("workload request_state must be resident or external")
    if state["synthetic"]:
        if request_states != {"resident", "external"}:
            raise ValueError(
                "synthetic serving state must contain resident and external rows"
            )
        if claims.get("serving_state_is_production_cache_state"):
            raise ValueError("synthetic serving state was marked as production state")
    state_claim = claims.get("serving_state_is_production_cache_state")
    if not isinstance(state_claim, bool):
        raise ValueError("workload serving-state claim is not boolean")
    if state["policy"] == "preserve_absent":
        if request_states != {None} or state_claim:
            raise ValueError(
                "absent serving state cannot be claimed as production cache state"
            )
    elif state["policy"] == "preserve_existing":
        if not request_states <= {"resident", "external"} or None in request_states:
            raise ValueError(
                "preserved serving state must label every request as resident or external"
            )
        if not state_claim:
            raise ValueError(
                "preserved serving state must retain its production-state claim"
            )
    elif state["policy"] == "root_resident":
        if not state["synthetic"] or state_claim:
            raise ValueError("root_resident state provenance is inconsistent")
    else:
        raise ValueError(f"unknown serving-state policy: {state['policy']}")
    prompt = manifest.get("prompt")
    if (
        not isinstance(prompt, dict)
        or not isinstance(prompt.get("enabled"), bool)
        or prompt.get("semantic_representativeness_claim") is not False
    ):
        raise ValueError("workload manifest lacks non-semantic prompt provenance")
    for row in rows:
        input_length = row["input_length"]
        output_length = row["output_length"]
        if (
            not isinstance(input_length, int)
            or isinstance(input_length, bool)
            or input_length <= 0
            or not isinstance(output_length, int)
            or isinstance(output_length, bool)
            or output_length < 0
        ):
            raise ValueError("workload lengths are out of range")
        if not isinstance(row["hash_ids"], list) or any(
            not isinstance(value, str) or not value for value in row["hash_ids"]
        ):
            raise ValueError("workload hash_ids are not a non-empty-string list")
        if prompt["enabled"]:
            prompt_count = row.get("prompt_token_count")
            if (
                not isinstance(prompt_count, int)
                or isinstance(prompt_count, bool)
                or prompt_count != input_length
            ):
                raise ValueError(
                    "synthetic prompt token count does not match input length"
                )
        if len(row["hash_ids"]) > (input_length + block_size - 1) // block_size:
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
