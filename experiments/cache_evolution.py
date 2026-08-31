"""Exact per-arrival cache binding evidence for stateful serving replays.

Initial placement and timed placement are different facts.  A setup probe can
prove where an exact prefix resides before the timed window, but earlier timed
requests may subsequently promote, extend, or evict that prefix.  This module
keeps those facts separate while retaining an exact token-identity chain for
every observation.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence


TokenInput = Sequence[int]


def token_prefix_digest(tokens: TokenInput, prefix_tokens: int) -> str:
    """Return a domain-separated digest for one exact token prefix."""

    if prefix_tokens < 0 or prefix_tokens > len(tokens):
        raise ValueError("token-prefix digest length is outside the input")
    digest = hashlib.sha256(b"nta-exact-token-prefix-v1\0")
    digest.update(int(prefix_tokens).to_bytes(8, "little", signed=False))
    for token in tokens[:prefix_tokens]:
        value = int(token)
        if value < 0:
            raise ValueError("token-prefix identity contains a negative token")
        digest.update(value.to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def arrival_cache_state(host_tokens: int, device_tokens: int) -> str:
    """Classify the exact prefix visible when a request reaches the engine."""

    if host_tokens < 0 or device_tokens < 0:
        raise ValueError("cache-prefix token counts cannot be negative")
    if host_tokens == 0 and device_tokens == 0:
        return "uncached"
    if host_tokens == 0:
        return "device"
    if device_tokens == 0:
        return "host"
    return "split"


def annotate_timed_cache_bindings(
    records: Sequence[dict[str, Any]],
    *,
    resident_inputs: Sequence[TokenInput],
    external_inputs: Sequence[TokenInput],
    external_materialized_prefix_tokens: Sequence[int],
    external_initial_prefix_tokens: Sequence[int],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Annotate timed records without freezing cache state at setup time.

    The first timed access to each exact initial prefix must still match the
    setup-derived prefix length.  Later accesses are observations of a stateful
    cache and may legally see promotion, extension, eviction, or a complete
    miss.  Access ordinals are assigned by the declared arrival schedule, not
    by coroutine completion order.
    """

    if len(external_inputs) != len(external_materialized_prefix_tokens) or len(
        external_inputs
    ) != len(external_initial_prefix_tokens):
        raise ValueError("external cache-binding vectors have different lengths")

    inputs_by_kind = {
        "resident": resident_inputs,
        "external": external_inputs,
    }
    role_records: dict[str, list[dict[str, Any]]] = {}
    entries: list[dict[str, Any]] = []
    for kind, inputs in inputs_by_kind.items():
        selected = sorted(
            (record for record in records if record.get("kind") == kind),
            key=lambda record: int(record.get("index", -1)),
        )
        if [int(record.get("index", -1)) for record in selected] != list(
            range(len(inputs))
        ):
            raise ValueError(f"timed {kind} cache-binding indices are not contiguous")
        role_records[kind] = selected
        for index, (record, raw_input) in enumerate(zip(selected, inputs, strict=True)):
            tokens = tuple(int(token) for token in raw_input)
            if len(tokens) <= 1 or int(record.get("input_tokens", -1)) != len(tokens):
                raise ValueError(f"timed {kind} input shape disagrees at index {index}")
            if kind == "external":
                materialized = int(external_materialized_prefix_tokens[index])
                initial = int(external_initial_prefix_tokens[index])
            else:
                materialized = len(tokens) - 1
                initial = materialized
            if not (0 < materialized <= initial < len(tokens)):
                raise ValueError(
                    f"timed {kind} initial prefix is invalid at index {index}"
                )

            host = int(record.get("host_cached_tokens", -1))
            device = int(record.get("device_cached_tokens", -1))
            state = arrival_cache_state(host, device)
            observed = host + device
            if observed >= len(tokens):
                raise ValueError(
                    f"timed {kind} cache prefix consumes the query at index {index}"
                )
            identity = token_prefix_digest(tokens, initial)
            observed_identity = token_prefix_digest(tokens, observed)
            record.update(
                {
                    "materialized_cached_prefix_tokens": materialized,
                    "effective_initial_cached_prefix_tokens": initial,
                    "observed_cached_prefix_tokens": observed,
                    "initial_cache_state": kind,
                    "observed_cache_state": state,
                    "cache_binding_identity_sha256": identity,
                    "observed_cache_binding_sha256": observed_identity,
                }
            )
            arrival = record.get("arrival_offset_seconds")
            if (
                not isinstance(arrival, (int, float))
                or isinstance(arrival, bool)
                or not math.isfinite(float(arrival))
                or float(arrival) < 0.0
            ):
                raise ValueError(f"timed {kind} arrival is invalid at index {index}")
            entries.append(
                {
                    "record": record,
                    "initial": initial,
                    "identity": identity,
                    "arrival": float(arrival),
                }
            )

    identity_accesses: dict[str, int] = {}
    first_access_failures: list[dict[str, Any]] = []
    for entry in sorted(
        entries,
        key=lambda value: (
            value["arrival"],
            str(value["record"].get("request_id", "")),
            str(value["record"]["kind"]),
            int(value["record"]["index"]),
        ),
    ):
        identity = str(entry["identity"])
        ordinal = identity_accesses.get(identity, 0)
        identity_accesses[identity] = ordinal + 1
        record = entry["record"]
        matches_initial = int(record["observed_cached_prefix_tokens"]) == int(
            entry["initial"]
        )
        record.update(
            {
                "cache_identity_access_ordinal": ordinal,
                "cache_identity_first_access": ordinal == 0,
                "initial_cache_contract_match": matches_initial,
            }
        )
        if ordinal == 0 and not matches_initial:
            first_access_failures.append(
                {
                    "kind": record["kind"],
                    "index": int(record["index"]),
                    "request_id": str(record.get("request_id", "")),
                    "expected": int(entry["initial"]),
                    "observed": int(record["observed_cached_prefix_tokens"]),
                    "device": int(record["device_cached_tokens"]),
                    "host": int(record["host_cached_tokens"]),
                }
            )
    if first_access_failures:
        raise ValueError(
            "a first timed content access diverged from the exact initial cache "
            f"contract: {first_access_failures!r}"
        )

    transitions: dict[str, int] = {}
    for record in records:
        transition = (
            f"{record['initial_cache_state']}_to_{record['observed_cache_state']}"
        )
        transitions[transition] = transitions.get(transition, 0) + 1

    external = role_records["external"]
    contract = {
        "schema": 2,
        "composition": "initial_object_union_longest_common_prefix",
        "identity_source": "exact_token_ids",
        "timed_observation": "sglang_exact_radix_prefix_at_request_arrival",
        "state_model": "initial_proof_then_per_arrival_cache_evolution",
        "materialized_prefix_tokens": [
            int(record["materialized_cached_prefix_tokens"]) for record in external
        ],
        "effective_initial_prefix_tokens": [
            int(record["effective_initial_cached_prefix_tokens"]) for record in external
        ],
        "observed_prefix_tokens": [
            int(record["observed_cached_prefix_tokens"]) for record in external
        ],
        "initial_prefix_identity_sha256": [
            str(record["cache_binding_identity_sha256"]) for record in external
        ],
        "observed_prefix_identity_sha256": [
            str(record["observed_cache_binding_sha256"]) for record in external
        ],
        "identity_access_ordinals": [
            int(record["cache_identity_access_ordinal"]) for record in external
        ],
        "first_identity_access": [
            bool(record["cache_identity_first_access"]) for record in external
        ],
        "matches_initial_contract": [
            bool(record["initial_cache_contract_match"]) for record in external
        ],
        "first_identity_accesses_verified": True,
        "later_accesses_may_evolve": True,
        "unique_content_identities": len(identity_accesses),
        "timed_content_accesses": len(entries),
        "exact": True,
    }
    return contract, transitions
