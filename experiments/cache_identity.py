"""Exact cache-prefix composition over a content-addressed object union.

A request's materialized parent prefix is an object boundary, not necessarily
the longest prefix visible in a shared radix cache.  Another resident or
host-backed object can extend the same content path.  These helpers derive the
effective prefix from content identity alone, before execution timing can
affect the observation.
"""

from __future__ import annotations

from typing import Hashable, Sequence


Identity = Sequence[Hashable]
CachedObject = tuple[Identity, int]
Target = tuple[Identity, int]


def common_prefix_units(left: Identity, right: Identity) -> int:
    """Return the number of equal leading identity units."""

    count = 0
    for left_value, right_value in zip(left, right):
        if left_value != right_value:
            break
        count += 1
    return count


def effective_cached_prefix_tokens(
    target_identity: Identity,
    target_tokens: int,
    cached_objects: Sequence[CachedObject],
    *,
    tokens_per_identity_unit: int = 1,
) -> int:
    """Derive one request's reusable prefix from an initial object union.

    ``cached_objects`` contains each object's content identity and reusable
    token length.  The result is capped at ``target_tokens - 1`` because
    SGLang retains one input token as the query on an exact-prefix hit.
    """

    if target_tokens <= 1:
        raise ValueError("cache target must retain at least one query token")
    if tokens_per_identity_unit <= 0:
        raise ValueError("identity-unit width must be positive")
    if not cached_objects:
        raise ValueError("effective cache composition needs an object union")

    reusable_limit = target_tokens - 1
    best = 0
    for object_identity, object_tokens in cached_objects:
        if object_tokens <= 0:
            raise ValueError("cached object length must be positive")
        shared_tokens = (
            common_prefix_units(target_identity, object_identity)
            * tokens_per_identity_unit
        )
        best = max(best, min(reusable_limit, object_tokens, shared_tokens))
    return best


def effective_cached_prefixes(
    targets: Sequence[Target],
    cached_objects: Sequence[CachedObject],
    *,
    tokens_per_identity_unit: int = 1,
) -> tuple[int, ...]:
    """Vector form of :func:`effective_cached_prefix_tokens`."""

    return tuple(
        effective_cached_prefix_tokens(
            identity,
            token_count,
            cached_objects,
            tokens_per_identity_unit=tokens_per_identity_unit,
        )
        for identity, token_count in targets
    )
