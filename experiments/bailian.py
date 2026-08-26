"""Bailian-derived workload normalization for reproducible experiments.

The public Bailian data is anonymized.  This module therefore treats block
hashes, lengths, session links, and (when present) timestamps as the source
trace.  It never invents production arrival times from offline row order and
never claims that synthetic prompt text is semantically representative.

The module is deliberately dependency-free.  A normalized manifest can be
consumed by a serving adapter, while the exact block-token representation is
also available for adapters that can bypass a tokenizer.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = 1
DEFAULT_BLOCK_SIZE = 16
STATE_POLICIES = ("preserve", "root_resident")
_MISSING = object()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def demand_trace_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash only the exact demand identity consumed by a serving adapter.

    Prompt text is intentionally excluded: structure-only prompts are an
    adapter vehicle, while request order, block identity, lengths, state, and
    release time define the workload that must be shared by paired arms.
    """

    canonical = [
        {
            "request_id": str(row["request_id"]),
            "input_length": int(row["input_length"]),
            "output_length": int(row["output_length"]),
            "hash_ids": [str(value) for value in row.get("hash_ids", ())],
            "block_size": int(row.get("block_size", DEFAULT_BLOCK_SIZE)),
            "request_state": row.get("request_state"),
            "arrival_seconds": float(row["arrival_seconds"]),
        }
        for row in rows
    ]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _first(row: Mapping[str, Any], *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    if default is not _MISSING:
        return default
    raise ValueError(f"row is missing one of {names}")


def _number(value: Any, field: str, *, integer: bool = False) -> float | int:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    if integer:
        rounded = int(parsed)
        if parsed != rounded:
            raise ValueError(f"{field} must be an integer")
        return rounded
    return parsed


def _hash_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = value.split()
        value = decoded
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise ValueError("hash_ids must be a JSON array or whitespace-separated string")
    result = [str(item) for item in value]
    if any(not item for item in result):
        raise ValueError("hash_ids cannot contain empty values")
    return result


def _timestamp_seconds(value: Any, unit: str) -> float:
    timestamp = float(_number(value, "timestamp"))
    if unit == "seconds":
        return timestamp
    if unit == "milliseconds":
        return timestamp / 1000.0
    if unit != "auto":
        raise ValueError(f"unknown timestamp unit {unit}")
    # Bailian online timestamps are documented in seconds.  Auto is only a
    # convenience for exports that clearly use epoch milliseconds; callers
    # should record the explicit unit in the manifest when possible.
    return timestamp / 1000.0 if timestamp > 100_000_000_000 else timestamp


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL (or a JSON array) without silently dropping malformed rows."""

    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list) or not all(
            isinstance(row, dict) for row in value
        ):
            raise ValueError("JSON workload input must be an array of objects")
        return [dict(row) for row in value]
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at line {line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"line {line_number} is not a JSON object")
        rows.append(row)
    return rows


def _normalize_row(
    row: Mapping[str, Any], index: int, timestamp_unit: str
) -> dict[str, Any]:
    request_id = str(
        _first(row, "request_id", "id", "chat_id", default=f"request-{index:08d}")
    )
    input_length = int(
        _number(
            _first(row, "input_length", "prompt_tokens", "input_tokens"),
            "input_length",
            integer=True,
        )
    )
    output_length = int(
        _number(
            _first(
                row, "output_length", "completion_tokens", "output_tokens", default=0
            ),
            "output_length",
            integer=True,
        )
    )
    hashes = _hash_ids(_first(row, "hash_ids", "prefix_hashes", default=[]))
    timestamp = _first(row, "timestamp", "arrival", "arrival_seconds", default=None)
    normalized: dict[str, Any] = {
        "request_id": request_id,
        "input_length": input_length,
        "output_length": output_length,
        "hash_ids": hashes,
        "block_size": DEFAULT_BLOCK_SIZE,
        "chat_id": str(_first(row, "chat_id", "session_id", default=request_id)),
        "parent_chat_id": _first(row, "parent_chat_id", "parent_id", default=None),
        "turn": int(_number(_first(row, "turn", default=index), "turn", integer=True)),
        "modality": str(_first(row, "type", "modality", default="text")),
        "request_state": _first(row, "request_state", "state", default=None),
        "source_row": index,
    }
    if timestamp is not None:
        normalized["timestamp_seconds"] = _timestamp_seconds(timestamp, timestamp_unit)
        normalized["arrival_source"] = "trace_timestamp"
    else:
        normalized["arrival_source"] = "absent_offline_trace"
    return normalized


def _prefix_reuse(rows: list[dict[str, Any]]) -> None:
    """Annotate exact shared prefixes against previously seen requests.

    A trie makes this linear in the total number of hash IDs.  The previous
    tuple-prefix set rebuilt a tuple of length 1..L for every row, which made
    long anonymized traces needlessly quadratic in both copying and hashing.
    """

    prefix_trie: dict[str, Any] = {}
    for row in rows:
        hashes = row["hash_ids"]
        node = prefix_trie
        longest = 0
        for block_id in hashes:
            child = node.get(block_id)
            if child is None:
                break
            node = child
            longest += 1

        node = prefix_trie
        for block_id in hashes:
            child = node.get(block_id)
            if child is None:
                child = {}
                node[block_id] = child
            node = child
        row["shared_prefix_blocks"] = longest
        row["unique_blocks"] = max(
            0, math.ceil(row["input_length"] / row["block_size"]) - longest
        )


def input_page_ids(
    row: Mapping[str, Any], *, block_size: int = DEFAULT_BLOCK_SIZE
) -> tuple[str, ...]:
    """Return the exact logical input-page identities for one normalized row.

    Bailian may expose only the shared prefix hashes.  The remaining pages are
    deterministic request-local identities, matching :func:`synthesize_prompt`.
    Keeping this rule in the workload layer prevents serving harnesses from
    estimating cache pressure by summing request lengths and accidentally
    counting shared pages multiple times.
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    token_count = int(row["input_length"])
    if token_count <= 0:
        raise ValueError("input_length must be positive")
    page_count = math.ceil(token_count / block_size)
    page_ids = [str(value) for value in row.get("hash_ids", ())]
    if len(page_ids) > page_count:
        raise ValueError("hash prefix is longer than the request")
    page_ids.extend(
        f"{row['request_id']}:unique:{index}"
        for index in range(len(page_ids), page_count)
    )
    return tuple(page_ids)


def unique_input_page_ids(
    rows: Iterable[Mapping[str, Any]], *, block_size: int = DEFAULT_BLOCK_SIZE
) -> frozenset[str]:
    """Return the deduplicated input-page set for a request collection."""

    return frozenset(
        page_id
        for row in rows
        for page_id in input_page_ids(row, block_size=block_size)
    )


def _assign_serving_states(rows: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    """Attach an explicit serving-state construction policy to a trace.

    Bailian exports describe session structure but do not label whether a
    request was resident in this experiment's device pool.  ``root_resident``
    turns that missing runtime state into a reproducible experimental setup:
    root/turn-zero requests are warmed into the device tier and follow-up
    turns are driven through the external-tier path.  This is a setup label,
    not a claim about the production cache state.
    """

    if policy not in STATE_POLICIES:
        raise ValueError(f"unknown serving state policy: {policy}")
    explicit = any(row.get("request_state") is not None for row in rows)
    if policy == "preserve" or explicit:
        return {
            "policy": "preserve_existing" if explicit else "preserve_absent",
            "synthetic": False,
            "source": "trace_request_state" if explicit else "not_provided",
        }
    if policy == "root_resident":
        for row in rows:
            parent = row.get("parent_chat_id")
            turn = int(row.get("turn", 0))
            is_root = parent in (None, "", -1, "-1") or turn <= 0
            row["request_state"] = "resident" if is_root else "external"
        counts = {
            state: sum(row["request_state"] == state for row in rows)
            for state in ("resident", "external")
        }
        if not counts["resident"] or not counts["external"]:
            raise ValueError(
                "root_resident state policy did not produce both resident and "
                "external requests"
            )
        return {
            "policy": policy,
            "synthetic": True,
            "source": "parent_chat_id_or_turn",
            "counts": counts,
        }
    raise ValueError(f"unhandled serving state policy {policy}")


def _normalize_arrivals(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    time_scale: float,
    target_rate: float | None,
    seed: int,
    reference_rows: Sequence[dict[str, Any]] | None,
) -> dict[str, Any]:
    if time_scale <= 0:
        raise ValueError("time_scale must be positive")
    timestamps = [row.get("timestamp_seconds") for row in rows]
    has_timestamps = all(value is not None for value in timestamps) and bool(rows)
    if mode == "trace":
        if not has_timestamps:
            raise ValueError("trace arrival mode requires timestamps on every row")
        raw = [float(value) for value in timestamps]
        if any(right < left for left, right in zip(raw, raw[1:])):
            raise ValueError("trace timestamps must be non-decreasing")
        origin = raw[0]
        offsets = [(value - origin) * time_scale for value in raw]
        source = "trace_timestamp"
        if target_rate is not None:
            if target_rate <= 0:
                raise ValueError("target_rate must be positive")
            raw_gaps = [
                max(0.0, right - left) * time_scale for left, right in zip(raw, raw[1:])
            ]
            positive = [gap for gap in raw_gaps if gap > 0]
            if positive:
                scale = (1.0 / target_rate) / mean(positive)
                offsets = [0.0]
                for gap in raw_gaps:
                    offsets.append(offsets[-1] + gap * scale)
            source = "trace_timestamp_scaled"
    elif mode == "batch_release":
        offsets = [0.0 for _ in rows]
        source = "batch_release_no_arrival_claim"
    elif mode == "calibrated_open_loop":
        if target_rate is None or target_rate <= 0:
            raise ValueError("calibrated_open_loop requires a positive target_rate")
        if reference_rows:
            ref = [float(row["arrival_seconds"]) for row in reference_rows]
            ref_gaps = [max(0.0, right - left) for left, right in zip(ref, ref[1:])]
            positive = [gap for gap in ref_gaps if gap > 0]
            if not positive:
                raise ValueError("arrival reference has no positive inter-arrival gaps")
            rng = random.Random(seed)
            shuffled = list(positive)
            rng.shuffle(shuffled)
            scale = (1.0 / target_rate) / mean(positive)
            gaps = [
                shuffled[index % len(shuffled)] * scale
                for index in range(max(0, len(rows) - 1))
            ]
            offsets = [0.0]
            for gap in gaps:
                offsets.append(offsets[-1] + gap)
            source = "calibrated_from_timestamped_reference"
        else:
            gap = 1.0 / target_rate
            offsets = [index * gap for index in range(len(rows))]
            source = "calibrated_constant_rate"
    else:
        raise ValueError(f"unknown arrival mode {mode}")
    for row, offset in zip(rows, offsets):
        row["arrival_seconds"] = float(offset)
        if mode == "trace" and row["arrival_source"] == "trace_timestamp":
            row["arrival_source"] = source
        elif mode != "trace":
            row["arrival_source"] = source
    return {
        "mode": mode,
        "source": source,
        "time_scale": time_scale,
        "target_rate_per_second": target_rate,
        "has_original_timestamps": has_timestamps,
        # A trace replay at its original time scale is the only form that can
        # retain a production-arrival claim.  Scaling the gaps (or replacing
        # them with a target-rate calibration) creates a controlled replay,
        # even though its ordering and gap shape still come from the trace.
        "production_arrival_claim": (
            mode == "trace"
            and has_timestamps
            and time_scale == 1.0
            and target_rate is None
        ),
        "offline_order_is_arrival": False,
        "seed": seed,
    }


def _token_for(block_id: str, position: int) -> int:
    digest = hashlib.sha256(f"{block_id}:{position}".encode()).digest()
    # Keep IDs away from common special-token ranges while remaining stable.
    return 1000 + int.from_bytes(digest[:4], "big") % 900_000


def synthesize_prompt(
    row: Mapping[str, Any], *, block_size: int = DEFAULT_BLOCK_SIZE
) -> dict[str, Any]:
    """Create structure-preserving token IDs and a tokenizer-independent text form."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    token_count = int(row["input_length"])
    ids = input_page_ids(row, block_size=block_size)
    token_ids: list[int] = []
    for block_index, block_id in enumerate(ids):
        for position in range(block_size):
            if len(token_ids) >= token_count:
                break
            token_ids.append(_token_for(str(block_id), position))
        if len(token_ids) >= token_count:
            break
    words = " ".join(f"t{token_id}" for token_id in token_ids)
    return {
        "prompt_text": words,
        "prompt_token_ids": token_ids,
        "prompt_token_count": len(token_ids),
        "prompt_source": "deterministic_structure_only",
        "semantic_representativeness_claim": False,
        "shared_prefix_blocks": int(row.get("shared_prefix_blocks", 0)),
    }


def normalize(
    rows: Iterable[Mapping[str, Any]],
    *,
    arrival_mode: str = "batch_release",
    timestamp_unit: str = "auto",
    time_scale: float = 1.0,
    target_rate: float | None = None,
    seed: int = 20260824,
    reference_rows: Sequence[Mapping[str, Any]] | None = None,
    synthesize_prompts: bool = False,
    state_policy: str = "preserve",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized = [
        _normalize_row(row, index, timestamp_unit) for index, row in enumerate(rows)
    ]
    if not normalized:
        raise ValueError("workload input contains no requests")
    _prefix_reuse(normalized)
    state = _assign_serving_states(normalized, state_policy)
    reference = None
    if reference_rows is not None:
        reference = [
            _normalize_row(row, index, timestamp_unit)
            for index, row in enumerate(reference_rows)
        ]
        _prefix_reuse(reference)
        reference_timestamps = [row.get("timestamp_seconds") for row in reference]
        if not reference_timestamps or any(
            value is None for value in reference_timestamps
        ):
            raise ValueError("arrival reference must contain timestamps on every row")
        for row, timestamp in zip(reference, reference_timestamps):
            row["arrival_seconds"] = float(timestamp) - float(reference_timestamps[0])
    arrival = _normalize_arrivals(
        normalized,
        mode=arrival_mode,
        time_scale=time_scale,
        target_rate=target_rate,
        seed=seed,
        reference_rows=reference,
    )
    if synthesize_prompts:
        for row in normalized:
            row.update(synthesize_prompt(row))
    arrival_gaps = [
        right - left
        for left, right in zip(
            (row["arrival_seconds"] for row in normalized),
            (row["arrival_seconds"] for row in normalized[1:]),
        )
    ]
    manifest = {
        "schema": SCHEMA,
        "classification": "bailian-structure-replay",
        "source_format": "online_timestamped_or_offline_anonymized",
        "block_size": DEFAULT_BLOCK_SIZE,
        "request_count": len(normalized),
        "selection": {
            "mode": "all_rows",
            "max_requests": None,
            "source_request_count": len(normalized),
        },
        "arrival": arrival,
        "prompt": {
            "enabled": synthesize_prompts,
            "source": "deterministic_structure_only" if synthesize_prompts else None,
            "semantic_representativeness_claim": False,
        },
        "serving_state": state,
        "statistics": {
            "input_tokens": {
                "min": min(row["input_length"] for row in normalized),
                "max": max(row["input_length"] for row in normalized),
                "mean": mean(row["input_length"] for row in normalized),
            },
            "output_tokens": {
                "min": min(row["output_length"] for row in normalized),
                "max": max(row["output_length"] for row in normalized),
                "mean": mean(row["output_length"] for row in normalized),
            },
            "shared_prefix_blocks": sum(
                row["shared_prefix_blocks"] for row in normalized
            ),
            "requests_with_shared_prefix": sum(
                row["shared_prefix_blocks"] > 0 for row in normalized
            ),
            "modalities": sorted({row["modality"] for row in normalized}),
            "request_state_counts": {
                str(state): sum(row["request_state"] == state for row in normalized)
                for state in sorted(
                    {
                        row["request_state"]
                        for row in normalized
                        if row["request_state"] is not None
                    }
                )
            },
            "interarrival_seconds": {
                "min": min(arrival_gaps) if arrival_gaps else 0.0,
                "max": max(arrival_gaps) if arrival_gaps else 0.0,
                "mean": mean(arrival_gaps) if arrival_gaps else 0.0,
                "positive_fraction": (
                    sum(gap > 0 for gap in arrival_gaps) / len(arrival_gaps)
                    if arrival_gaps
                    else 0.0
                ),
            },
        },
        "claims": {
            "arrival_is_production_trace": arrival["production_arrival_claim"],
            "prompt_semantics_are_representative": False,
            "hash_block_identity_is_exact": True,
            "offline_row_order_is_arrival": False,
            "serving_state_is_production_cache_state": not state["synthetic"],
        },
    }
    manifest["demand_trace_digest"] = demand_trace_digest(normalized)
    return manifest, normalized


def write_workload(
    manifest_path: Path,
    records_path: Path,
    manifest: dict[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_text(
        "".join(json.dumps(dict(record), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = dict(manifest)
    manifest["records_file"] = records_path.name
    manifest["records_digest"] = file_digest(records_path)
    manifest["demand_trace_digest"] = demand_trace_digest(records)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
