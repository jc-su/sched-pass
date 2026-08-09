"""Query-aware page selection via Quest-style key envelopes.

Quest (Tang et al., MLSys 2024) ranks KV pages by an upper bound on the
attention logit any token inside the page can produce for the current query:
for per-page elementwise key minima and maxima, ``sum_d max(q_d * kmin_d,
q_d * kmax_d)`` bounds ``max_token q . k`` because each coordinate of a page
token lies inside ``[kmin_d, kmax_d]``. Selecting the top-k pages by this
bound is training free and works on any GQA checkpoint.

This module is the device-side identity producer for the selected-demand
workload: envelopes live wherever the caller maintains them, scoring runs on
the GPU against the live query, and the resulting indices feed the existing
selected-page acquisition machinery without any host round trip.
"""

from __future__ import annotations

from typing import Any


def page_key_envelopes(key_pages: Any) -> tuple[Any, Any]:
    """Per-page, per-head elementwise key minima and maxima.

    ``key_pages`` is ``(pages, tokens, kv_heads, head_dim)``; the reduction is
    over tokens. Returns ``(kmin, kmax)`` shaped
    ``(pages, kv_heads, head_dim)``.
    """
    if key_pages.ndim != 4:
        raise ValueError(
            "key pages must be (pages, tokens, kv_heads, head_dim); got "
            f"{tuple(key_pages.shape)}"
        )
    if min(key_pages.shape) <= 0:
        raise ValueError("key pages must be non-empty in every dimension")
    return key_pages.amin(dim=1), key_pages.amax(dim=1)


def quest_page_scores(
    query: Any, kmin: Any, kmax: Any, *, group_size: int
) -> Any:
    """Upper-bound attention-logit score for every page.

    ``query`` is ``(batch, query_heads, head_dim)``; ``kmin``/``kmax`` are
    ``(pages, kv_heads, head_dim)`` with ``query_heads == kv_heads *
    group_size`` (GQA). The result is ``(batch, pages)``: per query head the
    coordinatewise maximum of ``q * kmin`` and ``q * kmax`` summed over the
    head dimension, then summed over the head's group. Computation stays on
    the tensors' device; float32 accumulation keeps the bound exact for
    half-precision inputs.
    """
    if query.ndim != 3 or kmin.ndim != 3 or kmax.ndim != 3:
        raise ValueError("query and envelopes must be rank-3 tensors")
    if kmin.shape != kmax.shape:
        raise ValueError("envelope minima and maxima shapes disagree")
    batch, query_heads, head_dim = query.shape
    pages, kv_heads, envelope_dim = kmin.shape
    if envelope_dim != head_dim:
        raise ValueError(
            f"head dimension disagrees: query {head_dim}, envelope "
            f"{envelope_dim}"
        )
    if group_size <= 0 or query_heads != kv_heads * group_size:
        raise ValueError(
            f"{query_heads} query heads are not {kv_heads} KV heads times "
            f"group size {group_size}"
        )
    import torch

    grouped = query.to(torch.float32).view(batch, kv_heads, group_size, head_dim)
    # The bound needs max(q*kmin, q*kmax) per coordinate before any
    # contraction, so the product cannot be folded into a matmul.
    # (batch, kv_heads, group, 1, head_dim) x (1, kv_heads, 1, pages, head_dim)
    envelope_min = kmin.to(torch.float32).permute(1, 0, 2)[None, :, None]
    envelope_max = kmax.to(torch.float32).permute(1, 0, 2)[None, :, None]
    queries = grouped.unsqueeze(3)
    per_coordinate = torch.maximum(
        queries * envelope_min, queries * envelope_max
    )
    return per_coordinate.sum(dim=-1).sum(dim=2).sum(dim=1)


def budgeted_page_selection(
    scores: Any,
    page_count: int,
    budget: int,
    *,
    sink_pages: int,
    recent_pages: int,
) -> Any:
    """Deployed-form selection: retention first, envelopes for the rest.

    Returns the sorted page indices for one request: the recent window and
    attention sinks are always retained (recent outranks sink when the budget
    cannot hold both, because the local window carries the most mass), and
    the remaining budget goes to the highest-scoring other pages. The result
    is deterministic for tied scores via index order.
    """
    import torch

    if page_count <= 0 or budget <= 0:
        raise ValueError("page count and budget must be positive")
    if sink_pages < 0 or recent_pages < 0:
        raise ValueError("retention counts cannot be negative")
    if scores.ndim != 1 or scores.shape[0] < page_count:
        raise ValueError(
            f"scores must cover all {page_count} pages; got "
            f"{tuple(scores.shape)}"
        )
    recent = list(range(max(0, page_count - recent_pages), page_count))
    sink = [
        page for page in range(min(sink_pages, page_count))
        if page not in set(recent)
    ]
    reserved = (recent + sink)[:budget]
    reserved_set = set(reserved)
    remaining = budget - len(reserved)
    if remaining > 0:
        order = torch.argsort(
            scores[:page_count], descending=True, stable=True
        )
        for page in order.tolist():
            if page in reserved_set:
                continue
            reserved.append(page)
            reserved_set.add(page)
            remaining -= 1
            if remaining == 0:
                break
    result = torch.tensor(sorted(reserved), dtype=torch.int64,
                          device=scores.device)
    if result.numel() != min(budget, page_count):
        raise AssertionError("selection failed to fill the available budget")
    return result


def quest_candidate_scores(
    query: Any, candidate_key_pages: Any, *, group_size: int
) -> Any:
    """Score each request's own candidate pages.

    ``candidate_key_pages`` is ``(batch, candidates, tokens, kv_heads,
    head_dim)`` — every request ranks only its own candidate pool, matching
    the per-request page tables the acquisition plan consumes. Returns
    ``(batch, candidates)``.
    """
    if candidate_key_pages.ndim != 5:
        raise ValueError(
            "candidate key pages must be (batch, candidates, tokens, "
            f"kv_heads, head_dim); got {tuple(candidate_key_pages.shape)}"
        )
    import torch

    batch, candidates = candidate_key_pages.shape[:2]
    scores = torch.empty(
        (batch, candidates),
        dtype=torch.float32,
        device=query.device,
    )
    for request in range(batch):
        kmin, kmax = page_key_envelopes(candidate_key_pages[request])
        scores[request] = quest_page_scores(
            query[request : request + 1], kmin, kmax, group_size=group_size
        )[0]
    return scores
