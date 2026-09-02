"""Deterministic windows for natural-cache Bailian serving replay.

The normalized Bailian trace describes demand identity and arrival order, but
it does not observe this machine's HBM/host placement.  A natural replay must
therefore keep workload selection separate from cache outcome: this module
selects an immediately preceding warmup window plus a contiguous measured
window and computes only the prefix that those replayed requests can create.
The serving engine remains solely responsible for reporting where that prefix
actually resides at request time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .bailian import demand_trace_digest, input_page_ids, read_jsonl
from .validate_workload import validate as validate_workload


TokenInput = tuple[int, ...]


@dataclass(frozen=True)
class ReplayWindow:
    """A source-contiguous replay split into drained warmup and measurement."""

    warmup_rows: tuple[dict[str, Any], ...]
    measured_rows: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]

    @property
    def rows(self) -> tuple[dict[str, Any], ...]:
        return self.warmup_rows + self.measured_rows


@dataclass
class _PrefixNode:
    children: dict[str, "_PrefixNode"] = field(default_factory=dict)
    last_position: int | None = None


def _is_followup(row: Mapping[str, Any]) -> bool:
    return row.get("parent_chat_id") not in (None, "", -1, "-1")


def _finite_nonnegative(row: Mapping[str, Any], field: str) -> float:
    value = row.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"Bailian replay row {field} must be finite and non-negative")
    return float(value)


def _annotate_replayable_prefixes(
    rows: list[dict[str, Any]],
    block_size: int,
    *,
    page_rows: Sequence[Sequence[str]] | None = None,
) -> None:
    """Compute exact local reuse and an input-only reuse-distance lower bound.

    ``replayable_prefix_tokens`` is expressed in source-input tokens.  The
    serving adapter appends one request-boundary query row, so an identical
    prior source input can make all of the current source input reusable; the
    appended row, rather than the final source token, remains the query.  The
    provider-length bound is also required for a target that extends a shorter
    cached object whose final (partial) content block matches.
    """

    if page_rows is None:
        materialized_page_rows = [
            input_page_ids(row, block_size=block_size) for row in rows
        ]
    else:
        if len(page_rows) != len(rows):
            raise ValueError("precomputed replay pages do not match the selected rows")
        materialized_page_rows = page_rows
    root = _PrefixNode()
    for position, (row, page_ids) in enumerate(
        zip(rows, materialized_page_rows, strict=True)
    ):
        node = root
        blocks = 0
        for page_id in page_ids:
            child = node.children.get(page_id)
            if child is None:
                break
            node = child
            blocks += 1
        provider_position = node.last_position if blocks else None
        row["replayable_prefix_blocks"] = blocks
        provider_tokens = (
            int(rows[provider_position]["input_length"])
            if provider_position is not None
            else 0
        )
        row["replayable_prefix_tokens"] = min(
            int(row["input_length"]), provider_tokens, blocks * block_size
        )
        if provider_position is None:
            row["replayable_prefix_provider_source_index"] = None
            row["replayable_reuse_gap_seconds"] = None
            row["replayable_intervening_unique_input_pages"] = 0
        else:
            provider = rows[provider_position]
            row["replayable_prefix_provider_source_index"] = int(
                provider["source_index"]
            )
            reuse_gap = _finite_nonnegative(
                row, "arrival_seconds"
            ) - _finite_nonnegative(provider, "arrival_seconds")
            if reuse_gap < 0.0:
                raise ValueError("Bailian replay reuse precedes its prefix provider")
            row["replayable_reuse_gap_seconds"] = reuse_gap
            intervening = {
                page_id
                for intervening_pages in materialized_page_rows[
                    provider_position + 1 : position
                ]
                for page_id in intervening_pages
            }
            row["replayable_intervening_unique_input_pages"] = len(intervening)

        node = root
        for page_id in page_ids:
            child = node.children.get(page_id)
            if child is None:
                child = _PrefixNode()
                node.children[page_id] = child
            node = child
            node.last_position = position


def _mechanism_opportunity(
    measured: Sequence[Mapping[str, Any]],
    *,
    block_size: int,
    device_token_capacity: int | None,
    consumer_wave_tokens: int | None,
) -> dict[str, Any]:
    """Describe cache-pressure/overlap opportunity without using performance.

    Intervening input pages omit generated output pages, so their count is a
    lower bound on reuse distance. Crossing the configured device capacity is
    evidence of cache pressure, not a claim that a particular prefix landed in
    host memory; the serving run must still report observed placement.
    """

    reused = [row for row in measured if int(row["replayable_prefix_tokens"]) > 0]
    base: dict[str, Any] = {
        "schema": 1,
        "selection_signal": "source_identity_arrival_and_capacity_only",
        "selection_uses_measured_performance": False,
        "replayable_prefix_requests": len(reused),
        "replayable_prefix_tokens": sum(
            int(row["replayable_prefix_tokens"]) for row in reused
        ),
        "reuse_distance_kind": "intervening_unique_input_pages_lower_bound",
        "observed_tier_placement_required": True,
    }
    if device_token_capacity is None or consumer_wave_tokens is None:
        return {
            **base,
            "status": "unparameterized",
            "reason": "device capacity and consumer-wave tokens were not supplied",
        }
    if device_token_capacity <= 0 or consumer_wave_tokens <= 0:
        raise ValueError("opportunity capacity and wave tokens must be positive")
    capacity_pages = max(1, device_token_capacity // block_size)
    capacity_crossing = [
        row
        for row in reused
        if int(row["replayable_intervening_unique_input_pages"]) >= capacity_pages
    ]
    overlap_candidates = [
        row
        for row in capacity_crossing
        if int(row["input_length"]) - int(row["replayable_prefix_tokens"])
        >= consumer_wave_tokens
    ]
    pressure_ratios = [
        int(row["replayable_intervening_unique_input_pages"]) / capacity_pages
        for row in reused
    ]
    return {
        **base,
        "status": "trace_constructed_not_placement_claim",
        "device_token_capacity": device_token_capacity,
        "device_capacity_pages": capacity_pages,
        "consumer_wave_tokens": consumer_wave_tokens,
        "input_capacity_crossing_requests": len(capacity_crossing),
        "input_capacity_crossing_prefix_tokens": sum(
            int(row["replayable_prefix_tokens"]) for row in capacity_crossing
        ),
        "tier_overlap_candidate_requests": len(overlap_candidates),
        "tier_overlap_candidate_prefix_tokens": sum(
            int(row["replayable_prefix_tokens"]) for row in overlap_candidates
        ),
        "maximum_input_pressure_capacity_ratio": max(pressure_ratios, default=0.0),
        "definition": (
            "replayable exact prefix whose input-only reuse distance crosses "
            "device capacity; overlap also requires at least one uncached "
            "consumer wave"
        ),
    }


def opportunity_stratum(opportunity: Mapping[str, Any]) -> str:
    """Classify one replay window using only pre-execution opportunity data."""

    if opportunity.get("status") == "unparameterized":
        raise ValueError("cannot stratify an unparameterized replay window")
    if int(opportunity.get("replayable_prefix_requests", 0)) == 0:
        return "no_replayable_prefix"
    if int(opportunity.get("input_capacity_crossing_requests", 0)) == 0:
        return "reuse_without_capacity_crossing"
    if int(opportunity.get("tier_overlap_candidate_requests", 0)) == 0:
        return "capacity_crossing_without_compute_wave"
    return "capacity_crossing_with_compute_wave"


def _phase_offsets(rows: Sequence[dict[str, Any]], time_scale: float) -> None:
    if not rows:
        return
    origin = _finite_nonnegative(rows[0], "arrival_seconds")
    previous = origin
    for row in rows:
        source = _finite_nonnegative(row, "arrival_seconds")
        if source < previous:
            raise ValueError("Bailian replay arrivals must be non-decreasing")
        row["replay_arrival_seconds"] = (source - origin) * time_scale
        previous = source


def _axis(values: Sequence[int]) -> dict[str, Any]:
    return {
        "min": min(values),
        "max": max(values),
        "distinct": len(set(values)),
        "heterogeneous": len(set(values)) > 1,
    }


def build_replay_window(
    rows: Sequence[Mapping[str, Any]],
    *,
    measured_start: int,
    warmup_requests: int,
    measured_requests: int,
    context_length: int,
    input_margin_tokens: int,
    max_output_tokens: int,
    input_adapter_tokens: int = 0,
    device_token_capacity: int | None = None,
    consumer_wave_tokens: int | None = None,
    output_length_scale: float = 1.0,
    time_scale: float = 1.0,
    page_id_rows: Sequence[Sequence[str]] | None = None,
) -> ReplayWindow:
    """Build a deterministic, source-contiguous natural-cache replay window.

    Output length is the sole transformed demand dimension.  The source value
    remains in ``output_length`` and ``replay_output_tokens`` records the exact
    bounded generation used by the serving run.  Input tokens, content hashes,
    request order, and scaled inter-arrival geometry are otherwise unchanged.
    """

    integer_options = (
        measured_start,
        warmup_requests,
        measured_requests,
        context_length,
        input_margin_tokens,
        input_adapter_tokens,
        max_output_tokens,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in integer_options
    ):
        raise ValueError("Bailian replay window options must be integers")
    if any(
        value is not None and (isinstance(value, bool) or not isinstance(value, int))
        for value in (device_token_capacity, consumer_wave_tokens)
    ):
        raise ValueError("Bailian opportunity parameters must be integers")
    if measured_start < 0 or warmup_requests < 0 or measured_requests <= 0:
        raise ValueError("Bailian replay window bounds are invalid")
    if page_id_rows is not None and len(page_id_rows) != len(rows):
        raise ValueError("precomputed replay pages do not match the source rows")
    if (
        context_length <= input_margin_tokens + input_adapter_tokens
        or input_margin_tokens < 0
        or input_adapter_tokens < 0
    ):
        raise ValueError("Bailian replay context envelope is invalid")
    if max_output_tokens <= 0:
        raise ValueError("Bailian replay output cap must be positive")
    if not math.isfinite(output_length_scale) or output_length_scale <= 0.0:
        raise ValueError("Bailian replay output scale must be finite and positive")
    if not math.isfinite(time_scale) or time_scale <= 0.0:
        raise ValueError("Bailian replay time scale must be finite and positive")
    warmup_begin = measured_start - warmup_requests
    measured_end = measured_start + measured_requests
    if warmup_begin < 0 or measured_end > len(rows):
        raise ValueError("Bailian replay window exceeds the source trace")

    selected = [dict(row) for row in rows[warmup_begin:measured_end]]
    block_sizes = {int(row.get("block_size", 0)) for row in selected}
    if len(block_sizes) != 1 or next(iter(block_sizes), 0) <= 0:
        raise ValueError("Bailian replay rows disagree on block size")
    block_size = next(iter(block_sizes))
    max_input_tokens = context_length - input_margin_tokens - input_adapter_tokens
    truncated = 0
    context_capped = 0
    for relative_index, row in enumerate(selected):
        input_tokens = int(row["input_length"])
        source_output = max(1, int(row["output_length"]))
        if input_tokens <= 0 or input_tokens >= max_input_tokens:
            raise ValueError(
                "Bailian replay input exceeds the serving envelope: "
                f"request={row.get('request_id')} input={input_tokens} "
                f"limit={max_input_tokens - 1}"
            )
        available_output = context_length - input_tokens - input_adapter_tokens
        if available_output <= 0:
            raise ValueError("Bailian replay request leaves no generation capacity")
        scaled_output = max(1, math.ceil(source_output * output_length_scale))
        replay_output = min(scaled_output, max_output_tokens, available_output)
        truncated += replay_output < source_output
        context_capped += available_output < min(scaled_output, max_output_tokens)
        row["source_index"] = warmup_begin + relative_index
        row["replay_phase"] = (
            "warmup" if relative_index < warmup_requests else "measurement"
        )
        row["replay_output_tokens"] = replay_output

    selected_page_rows = (
        page_id_rows[warmup_begin:measured_end]
        if page_id_rows is not None
        else None
    )
    _annotate_replayable_prefixes(
        selected,
        block_size,
        page_rows=selected_page_rows,
    )
    warmup = selected[:warmup_requests]
    measured = selected[warmup_requests:]
    _phase_offsets(warmup, time_scale)
    _phase_offsets(measured, time_scale)
    unique_pages = {
        page_id
        for page_ids in (
            selected_page_rows
            if selected_page_rows is not None
            else [input_page_ids(row, block_size=block_size) for row in selected]
        )
        for page_id in page_ids
    }
    measured_inputs = [int(row["input_length"]) for row in measured]
    measured_outputs = [int(row["replay_output_tokens"]) for row in measured]
    measured_prefixes = [int(row["replayable_prefix_tokens"]) for row in measured]
    measured_queries = [
        int(row["input_length"]) - int(row["replayable_prefix_tokens"])
        for row in measured
    ]
    metadata = {
        "schema": 1,
        "selection": "contiguous_source_window",
        "warmup_begin": warmup_begin,
        "measured_start": measured_start,
        "measured_end": measured_end,
        "warmup_requests": len(warmup),
        "measured_requests": len(measured),
        "time_scale": time_scale,
        "source_arrival_span_seconds": (
            _finite_nonnegative(selected[-1], "arrival_seconds")
            - _finite_nonnegative(selected[0], "arrival_seconds")
        ),
        "measured_arrival_span_seconds": (
            float(measured[-1]["replay_arrival_seconds"])
            if len(measured) > 1
            else 0.0
        ),
        "context_length": context_length,
        "input_margin_tokens": input_margin_tokens,
        "input_adapter_tokens": input_adapter_tokens,
        "max_output_tokens": max_output_tokens,
        "output_length_scale": output_length_scale,
        "output_truncated_requests": truncated,
        "output_context_capped_requests": context_capped,
        "output_transform": (
            "min(max(1, ceil(source * configured_scale)), configured_cap, "
            "remaining_context)"
        ),
        "block_size": block_size,
        "unique_input_pages": len(unique_pages),
        "selected_demand_trace_digest": demand_trace_digest(selected),
        "measured_followup_requests": sum(_is_followup(row) for row in measured),
        "measured_reused_prefix_requests": sum(
            value > 0 for value in measured_prefixes
        ),
        "measured_axes": {
            "input_tokens": _axis(measured_inputs),
            "output_tokens": _axis(measured_outputs),
            "replayable_prefix_tokens": _axis(measured_prefixes),
            "uncached_query_rows": _axis(measured_queries),
        },
        "cache_state_source": "observed_during_engine_replay",
        "warmup_policy": "immediately_preceding_source_requests",
        "production_cache_state_claim": False,
        "mechanism_opportunity": _mechanism_opportunity(
            measured,
            block_size=block_size,
            device_token_capacity=device_token_capacity,
            consumer_wave_tokens=consumer_wave_tokens,
        ),
    }
    return ReplayWindow(tuple(warmup), tuple(measured), metadata)


def load_replay_window(
    manifest_path: Path,
    **window_options: Any,
) -> ReplayWindow:
    """Validate a normalized workload and select one natural replay window."""

    path = manifest_path.resolve()
    manifest = validate_workload(path)
    if manifest["serving_state"].get("policy") != "preserve_absent":
        raise ValueError(
            "natural Bailian replay requires absent source placement; use the "
            "placement harness for synthetic resident/external assignments"
        )
    records_path = path.parent / str(manifest["records_file"])
    window = build_replay_window(read_jsonl(records_path), **window_options)
    metadata = dict(window.metadata)
    metadata.update(
        {
            "manifest": str(path),
            "manifest_digest": hashlib.sha256(path.read_bytes()).hexdigest(),
            "records_digest": str(manifest["records_digest"]),
            "source_demand_trace_digest": str(manifest["demand_trace_digest"]),
            "source_request_count": int(manifest["request_count"]),
            "source_arrival": manifest["arrival"],
            "source_claims": manifest["claims"],
        }
    )
    return ReplayWindow(window.warmup_rows, window.measured_rows, metadata)


def encode_content_blocks(
    rows: Iterable[Mapping[str, Any]],
    *,
    block_size: int,
    vocabulary_token_ids: Iterable[int],
    special_token_ids: Iterable[int] = (),
) -> tuple[tuple[TokenInput, ...], dict[str, Any]]:
    """Encode logical blocks without introducing token-level prefix aliases.

    Each distinct page receives a globally unique first token.  Equal page IDs
    therefore map to equal token blocks, while unequal pages diverge at their
    first token.  The function fails when a replay window exceeds that exact
    encoding capacity instead of silently hashing multiple pages together.
    """

    if block_size <= 0:
        raise ValueError("content-block encoding requires a positive block size")
    materialized = [dict(row) for row in rows]
    page_ids = sorted(
        {
            page_id
            for row in materialized
            for page_id in input_page_ids(row, block_size=block_size)
        }
    )
    special = {int(value) for value in special_token_ids}
    safe_tokens = sorted(
        {
            int(value)
            for value in vocabulary_token_ids
            if int(value) >= 0 and int(value) not in special
        }
    )
    if len(page_ids) > len(safe_tokens):
        raise ValueError(
            "natural replay has more distinct pages than collision-free first "
            f"tokens ({len(page_ids)} > {len(safe_tokens)}); select a smaller "
            "window or a tokenizer with a larger vocabulary"
        )
    if page_ids and not safe_tokens:
        raise ValueError("tokenizer exposes no safe content tokens")

    encoded_pages: dict[str, TokenInput] = {}
    digest = hashlib.sha256()
    for ordinal, page_id in enumerate(page_ids):
        values = [safe_tokens[ordinal]]
        for position in range(1, block_size):
            seed = hashlib.sha256(f"{page_id}:{position}".encode("utf-8")).digest()
            values.append(
                safe_tokens[int.from_bytes(seed[:8], "big") % len(safe_tokens)]
            )
        block = tuple(values)
        encoded_pages[page_id] = block
        digest.update(page_id.encode("utf-8"))
        digest.update(b"\0")
        for token_id in block:
            digest.update(token_id.to_bytes(8, "little", signed=False))

    inputs: list[TokenInput] = []
    for row in materialized:
        token_count = int(row["input_length"])
        values = tuple(
            token_id
            for page_id in input_page_ids(row, block_size=block_size)
            for token_id in encoded_pages[page_id]
        )[:token_count]
        if len(values) != token_count:
            raise ValueError("content-block encoding did not cover the full input")
        inputs.append(values)
    return tuple(inputs), {
        "schema": 1,
        "kind": "collision_free_content_block_tokens_v1",
        "block_size": block_size,
        "unique_pages": len(page_ids),
        "safe_first_token_capacity": len(safe_tokens),
        "identity_digest": digest.hexdigest(),
        "distinct_pages_diverge_at_first_token": True,
    }


def tokenizer_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """Return the tokenizer's concrete vocabulary IDs, including added tokens."""

    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocabulary = get_vocab()
        if isinstance(vocabulary, dict) and vocabulary:
            return tuple(sorted({int(value) for value in vocabulary.values()}))
    size = int(len(tokenizer))
    if size <= 0:
        raise ValueError("tokenizer exposes an empty vocabulary")
    return tuple(range(size))


def encode_window(
    window: ReplayWindow, tokenizer: Any
) -> tuple[tuple[TokenInput, ...], dict[str, Any]]:
    """Encode a replay window with the concrete serving tokenizer."""

    return encode_content_blocks(
        window.rows,
        block_size=int(window.metadata["block_size"]),
        vocabulary_token_ids=tokenizer_token_ids(tokenizer),
        special_token_ids=getattr(tokenizer, "all_special_ids", ()),
    )


def append_cache_boundary(
    inputs: Sequence[TokenInput], tokenizer: Any
) -> tuple[tuple[TokenInput, ...], dict[str, Any]]:
    """Append one reserved token that isolates replayed input from output.

    Bailian exposes input content hashes but not the original token IDs or model
    outputs. Without a boundary, a synthetic completion can accidentally match
    the next replayed request's continuation and manufacture cache reuse absent
    from the trace. ``encode_content_blocks`` excludes every special token, so
    appending one special ID preserves each source input-prefix relationship and
    guarantees generated suffixes sit behind a divergent cache edge.
    """

    materialized = tuple(tuple(int(value) for value in item) for item in inputs)
    if not materialized or any(not item for item in materialized):
        raise ValueError("cache-boundary adaptation requires non-empty token inputs")
    special = sorted(
        {
            int(value)
            for value in getattr(tokenizer, "all_special_ids", ())
            if int(value) >= 0
        }
    )
    if not special:
        raise ValueError("tokenizer exposes no reserved cache-boundary token")
    preferred = getattr(tokenizer, "eos_token_id", None)
    boundary = int(preferred) if preferred in special else special[0]
    if any(boundary in item for item in materialized):
        raise ValueError("content-block encoding used the reserved boundary token")
    adapted = tuple(item + (boundary,) for item in materialized)
    digest = hashlib.sha256()
    for item in adapted:
        for token_id in item:
            digest.update(token_id.to_bytes(8, "little", signed=False))
        digest.update(b"\0")
    return adapted, {
        "schema": 1,
        "kind": "reserved_special_token_cache_boundary_v1",
        "boundary_token_id": boundary,
        "added_query_rows_per_request": 1,
        "source_prefix_identity_preserved": True,
        "synthetic_output_alias_prevented": True,
        "adapted_input_digest": digest.hexdigest(),
    }
