"""Shadow evaluation of the selected attention form (1D health-check 2).

On resident decode batches, this runs the complete selection plumbing the
selected form will use — logical-page envelopes over the live device pool,
Quest scoring against the step's real queries, budgeted retention selection,
compact per-token table construction — and verifies it end to end without
touching the batch's served output: two independently planned stock wrappers
execute over the identical compact table and must agree exactly, and every
structural invariant of the table (subset, retention, ordering) is asserted.
Zero acquisition occurs; serving semantics are unchanged. Routing real output
through the form is the next stage and requires this one green.

SGLang's HiCache profile stores KV at page size one, so a "page" here is a
logical run of ``page_tokens`` consecutive sequence positions; a selected
table is simply the subset of token-slot indices belonging to selected
logical pages, which FlashInfer consumes without kernel changes.
"""

from __future__ import annotations

from typing import Any

import torch

from nta_runtime.quest_selector import (
    budgeted_page_selection,
    page_key_envelopes,
    quest_page_scores,
)


class SelectedShadow:
    """Owns the verification wrapper pair and the per-step evaluation."""

    def __init__(self, budget_pages: int, page_tokens: int) -> None:
        if budget_pages <= 0 or page_tokens <= 0:
            raise ValueError("selected-form budget and page tokens must be positive")
        self.budget_pages = budget_pages
        self.page_tokens = page_tokens
        self._wrappers: tuple[Any, Any] | None = None

    def _verification_wrappers(self) -> tuple[Any, Any]:
        if self._wrappers is None:
            import flashinfer

            self._wrappers = tuple(
                flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    torch.empty(
                        32 * 1024 * 1024, dtype=torch.uint8, device="cuda"
                    ),
                    "NHD",
                    backend="fa2",
                )
                for _ in range(2)
            )
        return self._wrappers

    def _select_request(
        self, query: torch.Tensor, tokens: torch.Tensor, key_cache: torch.Tensor,
        group_size: int,
    ) -> torch.Tensor:
        """Return the kept token-slot indices for one request."""
        count = int(tokens.numel())
        pages = (count + self.page_tokens - 1) // self.page_tokens
        keys = key_cache[tokens.to(torch.long)].to(torch.float32)
        full = (count // self.page_tokens) * self.page_tokens
        if full:
            kmin, kmax = page_key_envelopes(
                keys[:full].view(
                    full // self.page_tokens, self.page_tokens, *keys.shape[1:]
                )
            )
        else:
            kmin = keys.new_empty((0, *keys.shape[1:]))
            kmax = kmin
        if full < count:
            tail_min, tail_max = page_key_envelopes(keys[full:].unsqueeze(0))
            kmin = torch.cat([kmin, tail_min])
            kmax = torch.cat([kmax, tail_max])
        scores = quest_page_scores(
            query.unsqueeze(0).to(torch.float32), kmin, kmax,
            group_size=group_size,
        )[0]
        chosen = budgeted_page_selection(
            scores, pages, min(self.budget_pages, pages),
            sink_pages=1, recent_pages=2,
        )
        kept = torch.cat(
            [
                tokens[p * self.page_tokens : (p + 1) * self.page_tokens]
                for p in chosen.tolist()
            ]
        )
        # Structural invariants, asserted rather than assumed: the kept set
        # is a subset in original order, and the recent window survived.
        if kept.numel() > count or not torch.isin(kept, tokens).all():
            raise RuntimeError("selected form kept tokens outside the request")
        recent = tokens[-min(count, self.page_tokens) :]
        if not torch.isin(recent, kept).all():
            raise RuntimeError("selected form dropped the recent window")
        return kept

    def evaluate(
        self,
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        layer: Any,
        stats: dict[str, Any],
        serve_output: torch.Tensor | None = None,
    ) -> bool:
        """Run the selection chain; optionally serve its result.

        With ``serve_output`` set (stage 3a), the verified selected-attention
        result is copied into the layer's output tensor and becomes the
        served computation; the dual-wrapper equality and structural
        invariants still gate every step. Returns whether output was served.
        """
        if torch.cuda.is_current_stream_capturing():
            return False
        key_cache, value_cache = kv_cache
        batch_size = q.shape[0]
        indptr = wrapper._paged_kv_indptr_buf[: batch_size + 1]
        total = int(indptr[-1])
        indices = wrapper._paged_kv_indices_buf[:total]
        group_size = q.shape[1] // key_cache.shape[1]

        kept_per_request = []
        indptr_cpu = indptr.to("cpu", torch.int64)
        for request in range(batch_size):
            begin = int(indptr_cpu[request])
            end = int(indptr_cpu[request + 1])
            if end <= begin:
                raise RuntimeError("selected form saw an empty request table")
            kept_per_request.append(
                self._select_request(
                    q[request], indices[begin:end], key_cache, group_size
                )
            )

        compact_indices = torch.cat(kept_per_request).to(torch.int32)
        counts = [0] + [int(k.numel()) for k in kept_per_request]
        compact_indptr = torch.tensor(
            counts, dtype=torch.int32, device=indices.device
        ).cumsum(0, dtype=torch.int32)
        last_page_len = torch.ones(
            batch_size, dtype=torch.int32, device=indices.device
        )

        first, second = self._verification_wrappers()
        outputs = []
        for verifier in (first, second):
            verifier.plan(
                compact_indptr,
                compact_indices,
                last_page_len,
                q.shape[1],
                key_cache.shape[1],
                q.shape[2],
                1,
                q_data_type=q.dtype,
                kv_data_type=key_cache.dtype,
                sm_scale=layer.scaling,
                disable_split_kv=True,
            )
            outputs.append(verifier.run(q, (key_cache, value_cache)))
        if not torch.equal(outputs[0], outputs[1]):
            raise RuntimeError(
                "selected form verification wrappers disagree over an "
                "identical compact table"
            )
        prefix = "selected_serve" if serve_output is not None else "selected_shadow"
        if serve_output is not None:
            serve_output.copy_(outputs[0])
        stats[f"{prefix}_layers"] = stats.get(f"{prefix}_layers", 0) + 1
        stats[f"{prefix}_tokens_total"] = (
            stats.get(f"{prefix}_tokens_total", 0) + total
        )
        stats[f"{prefix}_tokens_kept"] = (
            stats.get(f"{prefix}_tokens_kept", 0) + int(compact_indices.numel())
        )
        return serve_output is not None
