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
        manifest.get("schema") != 2
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
        "diverse_serving_cohort",
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
    if selection["mode"] == "diverse_serving_cohort":
        if (
            not isinstance(selected_max, int)
            or isinstance(selected_max, bool)
            or selected_max != len(rows)
            or selection.get("algorithm")
            != "deterministic_joint_shape_spread_v1"
            or selection.get("distribution_representative_claim") is not False
        ):
            raise ValueError("diverse serving cohort selection is inconsistent")
        resident_count = selection.get("resident_requests")
        external_count = selection.get("external_requests")
        if (
            not isinstance(resident_count, int)
            or isinstance(resident_count, bool)
            or resident_count <= 0
            or not isinstance(external_count, int)
            or isinstance(external_count, bool)
            or external_count <= 0
            or resident_count + external_count != len(rows)
        ):
            raise ValueError("diverse serving cohort role counts are invalid")
        active_budget = selection.get("active_token_budget")
        active_tokens = selection.get("active_tokens")
        context_length = selection.get("context_length")
        max_input_tokens = selection.get("max_input_tokens")
        max_output_tokens = selection.get("max_output_tokens")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (context_length, max_input_tokens, max_output_tokens)
        ) or not (
            max_input_tokens < context_length
            and max_output_tokens < context_length
        ):
            raise ValueError("diverse serving cohort token envelope is invalid")
        expected_active = sum(
            int(row["input_length"]) + max(1, int(row["output_length"]))
            for row in rows
        )
        if (
            not isinstance(active_budget, int)
            or isinstance(active_budget, bool)
            or active_budget <= 0
            or not isinstance(active_tokens, int)
            or isinstance(active_tokens, bool)
            or active_tokens != expected_active
            or active_tokens > active_budget
        ):
            raise ValueError("diverse serving cohort exceeds its active token budget")
        if any(
            int(row["input_length"]) > max_input_tokens
            or max(1, int(row["output_length"])) > max_output_tokens
            or int(row["input_length"]) + max(1, int(row["output_length"]))
            > context_length
            for row in rows
        ):
            raise ValueError("diverse serving cohort violates its token envelope")
        lineage = manifest.get("lineage")
        if not isinstance(lineage, dict) or any(
            not isinstance(lineage.get(field), str) or not lineage[field]
            for field in (
                "source_manifest_digest",
                "source_records_digest",
                "source_demand_trace_digest",
            )
        ):
            raise ValueError("diverse serving cohort lacks source lineage")
    required = {
        "request_id",
        "input_length",
        "output_length",
        "hash_ids",
        "cached_prefix_tokens",
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
    elif state["policy"] == "diverse_serving_cohort":
        if (
            not state["synthetic"]
            or state_claim
            or request_states != {"resident", "external"}
            or state.get("counts")
            != {
                "resident": sum(
                    row.get("request_state") == "resident" for row in rows
                ),
                "external": sum(
                    row.get("request_state") == "external" for row in rows
                ),
            }
        ):
            raise ValueError("diverse serving cohort state provenance is inconsistent")
    else:
        raise ValueError(f"unknown serving-state policy: {state['policy']}")
    cache_placement = manifest.get("cache_placement")
    if (
        not isinstance(cache_placement, dict)
        or cache_placement.get("identity_field") != "cached_prefix_tokens"
        or not isinstance(cache_placement.get("source"), str)
        or not cache_placement["source"]
        or not isinstance(cache_placement.get("synthetic"), bool)
    ):
        raise ValueError("workload manifest lacks cache-placement provenance")
    cache_claim = claims.get("cache_placement_is_production")
    if not isinstance(cache_claim, bool) or cache_claim == cache_placement["synthetic"]:
        raise ValueError("workload cache-placement claim is inconsistent")
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
        cached_prefix_tokens = row["cached_prefix_tokens"]
        if (
            not isinstance(cached_prefix_tokens, int)
            or isinstance(cached_prefix_tokens, bool)
            or cached_prefix_tokens < 0
            or cached_prefix_tokens >= input_length
        ):
            raise ValueError("workload cached-prefix length is out of range")
        request_state = row.get("request_state")
        if request_state == "resident" and cached_prefix_tokens != input_length - 1:
            raise ValueError("resident workload row is not fully prefix-cached")
        if request_state == "external" and cached_prefix_tokens <= 0:
            raise ValueError("external workload row has no cached prefix")
        if request_state is None and cached_prefix_tokens != 0:
            raise ValueError("unplaced workload row carries cached-prefix state")
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
    if selection["mode"] == "diverse_serving_cohort":
        cohort = manifest.get("cohort_heterogeneity")
        if (
            not isinstance(cohort, dict)
            or cohort.get("schema") != 1
            or cohort.get("joint_shape_heterogeneity") is not True
            or cohort.get("request_states") != ["external", "resident"]
            or not isinstance(cohort.get("axes"), dict)
        ):
            raise ValueError("diverse serving cohort lacks heterogeneity evidence")
        expected_axes = {
            "input_tokens": [int(row["input_length"]) for row in rows],
            "output_tokens": [int(row["output_length"]) for row in rows],
            "cached_prefix_tokens": [
                int(row["cached_prefix_tokens"]) for row in rows
            ],
            "uncached_query_rows": [
                int(row["input_length"]) - int(row["cached_prefix_tokens"])
                for row in rows
            ],
        }
        for name, values in expected_axes.items():
            expected = {
                "min": min(values),
                "max": max(values),
                "distinct": len(set(values)),
                "heterogeneous": len(set(values)) > 1,
            }
            if cohort["axes"].get(name) != expected or not expected["heterogeneous"]:
                raise ValueError(
                    f"diverse serving cohort {name} evidence is inconsistent"
                )
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
