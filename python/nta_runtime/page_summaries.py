"""Incremental key-envelope maintenance for a mutating paged KV pool.

The per-step selector needs page envelopes that stay correct while the pool
mutates underneath it: decode appends one token per request per step to each
request's tail page, promotion materializes whole pages, and eviction reuses
page slots for new occupants. Envelope maintenance must therefore be
incremental (an append touches only its page, never a rescan), batched (one
kernel-side update per decode step, not one per request), and
generation-tagged so a reused slot can never serve a stale envelope.

All state lives on the caller's device; scoring gathers envelopes for a
candidate set and delegates to the verified Quest scorer.
"""

from __future__ import annotations

from typing import Any

from .quest_selector import quest_page_scores


class PageSummaryTable:
    """Per-page key envelopes with generation-safe slot reuse."""

    def __init__(
        self,
        page_count: int,
        page_tokens: int,
        kv_heads: int,
        head_dim: int,
        *,
        device: Any = "cpu",
    ) -> None:
        import torch

        if min(page_count, page_tokens, kv_heads, head_dim) <= 0:
            raise ValueError("page summary geometry must be positive")
        self.page_tokens = int(page_tokens)
        self.kv_heads = int(kv_heads)
        self.head_dim = int(head_dim)
        shape = (page_count, kv_heads, head_dim)
        self.kmin = torch.full(shape, float("inf"), device=device)
        self.kmax = torch.full(shape, float("-inf"), device=device)
        self.filled = torch.zeros(page_count, dtype=torch.int32, device=device)
        self.generation = torch.zeros(
            page_count, dtype=torch.int64, device=device
        )

    @property
    def page_count(self) -> int:
        return int(self.filled.shape[0])

    def _check_pages(self, page_indices: Any) -> Any:
        import torch

        if not torch.is_tensor(page_indices):
            page_indices = torch.as_tensor(
                page_indices, dtype=torch.int64, device=self.filled.device
            )
        if page_indices.ndim != 1 or page_indices.numel() == 0:
            raise ValueError("page indices must be a non-empty vector")
        if bool((page_indices < 0).any()) or bool(
            (page_indices >= self.page_count).any()
        ):
            raise ValueError("page index out of range")
        return page_indices.to(dtype=torch.int64, device=self.filled.device)

    def _check_tokens(self, key_tokens: Any, expected_rank: int) -> None:
        if key_tokens.ndim != expected_rank or key_tokens.shape[-2:] != (
            self.kv_heads,
            self.head_dim,
        ):
            raise ValueError(
                "key tokens must end in (kv_heads, head_dim) = "
                f"({self.kv_heads}, {self.head_dim}); got "
                f"{tuple(key_tokens.shape)}"
            )

    def reset_pages(self, page_indices: Any) -> None:
        """Invalidate slots for reuse and advance their generations."""
        pages = self._check_pages(page_indices)
        self.kmin[pages] = float("inf")
        self.kmax[pages] = float("-inf")
        self.filled[pages] = 0
        self.generation[pages] += 1

    def append_tokens(self, page_indices: Any, key_tokens: Any) -> None:
        """Fold one new token per listed page into its envelope.

        ``page_indices`` is ``(batch,)`` and ``key_tokens`` is
        ``(batch, kv_heads, head_dim)`` — the decode-step shape: every
        request contributes one key to its tail page. Duplicate pages within
        one call are rejected because their fill accounting would race.
        """

        pages = self._check_pages(page_indices)
        self._check_tokens(key_tokens, 3)
        if key_tokens.shape[0] != pages.numel():
            raise ValueError("one key token is required per listed page")
        if pages.unique().numel() != pages.numel():
            raise ValueError("duplicate pages in one append call")
        if bool((self.filled[pages] >= self.page_tokens).any()):
            raise ValueError("append exceeds page capacity")
        source = key_tokens.to(self.kmin.dtype)
        self.kmin.index_reduce_(0, pages, source, "amin")
        self.kmax.index_reduce_(0, pages, source, "amax")
        self.filled[pages] += 1

    def write_pages(self, page_indices: Any, key_pages: Any) -> None:
        """Materialize whole pages (promotion), replacing any prior state.

        ``key_pages`` is ``(batch, tokens, kv_heads, head_dim)`` with
        ``tokens`` in ``[1, page_tokens]`` (the final page of a sequence may
        be partial).
        """
        pages = self._check_pages(page_indices)
        self._check_tokens(key_pages, 4)
        tokens = key_pages.shape[1]
        if key_pages.shape[0] != pages.numel():
            raise ValueError("one key page is required per listed page")
        if not 1 <= tokens <= self.page_tokens:
            raise ValueError("page token count is outside the page geometry")
        source = key_pages.to(self.kmin.dtype)
        self.kmin[pages] = source.amin(dim=1)
        self.kmax[pages] = source.amax(dim=1)
        self.filled[pages] = tokens

    def scores(self, query: Any, page_indices: Any, *, group_size: int) -> Any:
        """Quest scores for a candidate set; unfilled pages score -inf."""

        pages = self._check_pages(page_indices)
        scores = quest_page_scores(
            query, self.kmin[pages], self.kmax[pages], group_size=group_size
        )
        empty = self.filled[pages] == 0
        if bool(empty.any()):
            scores = scores.masked_fill(
                empty.unsqueeze(0), float("-inf")
            )
        return scores

    def envelopes(self, page_indices: Any) -> tuple[Any, Any]:
        pages = self._check_pages(page_indices)
        return self.kmin[pages], self.kmax[pages]
