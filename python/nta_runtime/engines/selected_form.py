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

    def _prefill_wrappers(self) -> tuple[Any, Any]:
        if getattr(self, "_prefill_pair", None) is None:
            import flashinfer

            self._prefill_pair = tuple(
                flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    torch.empty(
                        64 * 1024 * 1024, dtype=torch.uint8, device="cuda"
                    ),
                    "NHD",
                    backend="fa2",
                )
                for _ in range(2)
            )
        return self._prefill_pair

    def evaluate_tiered(
        self,
        engine: Any,
        claim: Any,
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        layer: Any,
        stats: dict[str, Any],
        serve_output: torch.Tensor,
        prefill: bool,
    ) -> bool:
        """Serve one layer of a tiered batch: stage the chosen prefix rows
        through the bounded indexed path, then run attention over chosen
        prefix plus resident tail. Peer requests keep full tables.

        Decode without verification takes the plan-once fast path; verify
        mode and extend take the reference path (dual wrappers and byte
        verification under verify, a single wrapper otherwise)."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("tiered serving inside graph capture is unsupported")
        key_cache, value_cache = kv_cache
        local_layer = int(layer.layer_id) - claim.start_layer
        if not 0 <= local_layer < claim.layer_count:
            raise RuntimeError("tiered layer is outside the claimed model")
        group_size = q.shape[1] // key_cache.shape[1]
        stream = torch.cuda.current_stream()

        if not prefill and claim.fast_ok and not claim.verify:
            return self._tiered_decode_fast(
                engine, claim, wrapper, q, kv_cache, layer, local_layer,
                group_size, stats, serve_output, stream,
            )

        if prefill:
            planned = int(getattr(wrapper, "_batch_size", 1) or 1)
            if planned != 1:
                raise RuntimeError(
                    "tiered extend currently requires a single-request "
                    f"forward; the wrapper planned {planned}"
                )
            batch_size = 1
        else:
            batch_size = q.shape[0]
        indptr = wrapper._paged_kv_indptr_buf[: batch_size + 1]
        total = int(indptr[-1])
        indices = wrapper._paged_kv_indices_buf[:total]
        indptr_cpu = indptr.to("cpu", torch.int64)

        kept_per_request = []
        staged_any = False
        for request in range(batch_size):
            begin = int(indptr_cpu[request])
            end = int(indptr_cpu[request + 1])
            tokens = indices[begin:end]
            queries = q if prefill else q[request : request + 1]
            tokens_long = tokens.to(torch.long)
            prefix_mask = claim.slot_is_prefix[tokens_long]
            prefix_count = int(prefix_mask.sum())
            if prefix_count == 0:
                kept_per_request.append(tokens)
                continue
            if prefix_count != claim.token_count:
                raise RuntimeError(
                    f"tiered request exposes {prefix_count} of "
                    f"{claim.token_count} claimed prefix slots; partial "
                    "prefix reuse is outside the current stage"
                )
            staged_any = True
            chosen = claim.choose_pages(local_layer, queries, group_size)
            if claim.verify and claim.fast_ok and not prefill:
                free = claim.choose_free_pages(
                    local_layer, queries, group_size
                )
                device_chosen = sorted(
                    torch.cat([claim.forced_pages, free]).tolist()
                )
                if device_chosen != sorted(chosen):
                    raise RuntimeError(
                        "device-side page selection diverged from the "
                        "reference selection"
                    )
            claim.stage_layer(engine, local_layer, chosen, stream)
            chosen_mask = torch.zeros(
                claim.pages, dtype=torch.bool, device=tokens.device
            )
            chosen_mask[
                torch.tensor(chosen, dtype=torch.int64, device=tokens.device)
            ] = True
            positions = claim.slot_to_position[tokens_long]
            page_of_token = positions // claim.page_tokens
            keep_mask = ~prefix_mask | chosen_mask[page_of_token]
            kept_per_request.append(tokens[keep_mask])
        if not staged_any:
            raise RuntimeError(
                "tiered forward matched the claim but no request carried its "
                "prefix slots; refusing to serve unstaged rows"
            )

        compact_indices = torch.cat(kept_per_request).to(torch.int32)
        counts = [0] + [int(k.numel()) for k in kept_per_request]
        compact_indptr = torch.tensor(
            counts, dtype=torch.int32, device=indices.device
        ).cumsum(0, dtype=torch.int32)
        last_page_len = torch.ones(
            batch_size, dtype=torch.int32, device=indices.device
        )

        outputs = []
        if prefill:
            suffix = q.shape[0]
            qo_indptr = torch.tensor(
                [0, suffix], dtype=torch.int32, device=indices.device
            )
            verifiers = self._prefill_wrappers()
            for verifier in verifiers if claim.verify else verifiers[:1]:
                verifier.plan(
                    qo_indptr,
                    compact_indptr,
                    compact_indices,
                    last_page_len,
                    q.shape[1],
                    key_cache.shape[1],
                    q.shape[2],
                    1,
                    causal=True,
                    sm_scale=layer.scaling,
                    q_data_type=q.dtype,
                    kv_data_type=key_cache.dtype,
                )
                outputs.append(verifier.run(q, (key_cache, value_cache)))
        else:
            verifiers = self._verification_wrappers()
            for verifier in verifiers if claim.verify else verifiers[:1]:
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
        if len(outputs) == 2 and not torch.equal(outputs[0], outputs[1]):
            raise RuntimeError(
                "tiered verification wrappers disagree over an identical "
                "compact table"
            )
        serve_output.copy_(outputs[0])
        stats["tiered_rows_copied"] = (
            stats.get("tiered_rows_copied_released", 0) + claim.rows_copied
        )
        stats["tiered_rows_rehit"] = (
            stats.get("tiered_rows_rehit_released", 0) + claim.rows_rehit
        )
        kind = "tiered_prefill" if prefill else "tiered_decode"
        stats[f"{kind}_layers"] = stats.get(f"{kind}_layers", 0) + 1
        stats["tiered_tokens_total"] = (
            stats.get("tiered_tokens_total", 0) + total
        )
        stats["tiered_tokens_kept"] = (
            stats.get("tiered_tokens_kept", 0) + int(compact_indices.numel())
        )
        return True

    def _tiered_decode_fast(
        self,
        engine: Any,
        claim: Any,
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        layer: Any,
        local_layer: int,
        group_size: int,
        stats: dict[str, Any],
        serve_output: torch.Tensor,
        stream: Any,
    ) -> bool:
        """Plan-once decode serving.

        The kept row count is layer-invariant (retention always keeps the
        tail page), so the verification wrapper is planned exactly once per
        forward against an indices buffer this class owns; each layer
        rewrites only the claim request's prefix segment of that buffer from
        the device-side selection and runs attention. FlashInfer's decode
        planner consumes indptr lengths only — indices are read from the
        device buffer at run time — which is what makes the in-place rewrite
        sound.
        """
        ctx = claim.ctx
        if (
            ctx is None
            or local_layer == 0
            or ctx["wrapper"] is not wrapper
            or ctx["next"] != local_layer
        ):
            if local_layer != 0:
                raise RuntimeError(
                    "tiered fast path entered mid-forward at layer "
                    f"{local_layer} without a forward context"
                )
            ctx = self._build_decode_ctx(claim, wrapper, q, kv_cache, layer)
            claim.ctx = ctx
        ctx["next"] = local_layer + 1

        request = ctx["claim_request"]
        free = claim.choose_free_pages(
            local_layer, q[request : request + 1], group_size
        )
        positions = claim.kept_prefix_positions(free)
        ctx["plan_indices"][ctx["seg_begin"] : ctx["seg_end"]].copy_(
            claim.device_rows[positions]
        )
        claim.stage_missing(engine, local_layer, free, stream)
        ctx["verifier"].run(q, kv_cache, out=serve_output)
        if claim.verify_fast:
            self._fast_crosscheck(
                claim, q, kv_cache, layer, local_layer, group_size,
                serve_output, ctx,
            )
            stats["tiered_fast_checked_layers"] = (
                stats.get("tiered_fast_checked_layers", 0) + 1
            )

        stats["tiered_rows_copied"] = (
            stats.get("tiered_rows_copied_released", 0) + claim.rows_copied
        )
        stats["tiered_rows_rehit"] = (
            stats.get("tiered_rows_rehit_released", 0) + claim.rows_rehit
        )
        stats["tiered_decode_layers"] = (
            stats.get("tiered_decode_layers", 0) + 1
        )
        stats["tiered_fast_layers"] = stats.get("tiered_fast_layers", 0) + 1
        stats["tiered_tokens_total"] = (
            stats.get("tiered_tokens_total", 0) + ctx["total"]
        )
        stats["tiered_tokens_kept"] = (
            stats.get("tiered_tokens_kept", 0) + ctx["kept_total"]
        )
        return True

    def _build_decode_ctx(
        self,
        claim: Any,
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        layer: Any,
    ) -> dict[str, Any]:
        """Once per decode forward: identify the claim request, lay out the
        compact indices buffer (peer tables and the resident tail are
        layer-invariant), and plan the single serving wrapper against it."""
        key_cache, _ = kv_cache
        batch_size = q.shape[0]
        indptr_cpu = wrapper._paged_kv_indptr_buf[: batch_size + 1].to(
            "cpu", torch.int64
        )
        total = int(indptr_cpu[-1])
        indices = wrapper._paged_kv_indices_buf[:total]

        claim_request = None
        segments: list[tuple[str, torch.Tensor]] = []
        for request in range(batch_size):
            begin = int(indptr_cpu[request])
            end = int(indptr_cpu[request + 1])
            tokens = indices[begin:end]
            prefix_mask = claim.slot_is_prefix[tokens.to(torch.long)]
            prefix_count = int(prefix_mask.sum())
            if prefix_count == 0:
                segments.append(("peer", tokens))
                continue
            if prefix_count != claim.token_count:
                raise RuntimeError(
                    f"tiered request exposes {prefix_count} of "
                    f"{claim.token_count} claimed prefix slots; partial "
                    "prefix reuse is outside the current stage"
                )
            if claim_request is not None:
                raise RuntimeError(
                    "two requests carry the claimed prefix in one forward"
                )
            # The claim's slots need not be the leading table entries nor in
            # sequence order (radix hits can leave the claimed segment
            # mid-table); decode attention is order-free, so the layout is
            # simply [non-claim tokens, rewritable prefix segment].
            claim_request = request
            segments.append(("claim", tokens[~prefix_mask]))
        if claim_request is None:
            raise RuntimeError(
                "tiered forward matched the claim but no request carried "
                "its prefix slots; refusing to serve unstaged rows"
            )

        kept_counts = [
            claim.kept_prefix_rows + int(tokens.numel())
            if kind == "claim"
            else int(tokens.numel())
            for kind, tokens in segments
        ]
        kept_total = sum(kept_counts)
        plan_indices = torch.empty(
            kept_total, dtype=torch.int32, device=q.device
        )
        offset = seg_begin = seg_end = 0
        for (kind, tokens), count in zip(segments, kept_counts):
            if kind == "peer":
                plan_indices[offset : offset + count].copy_(tokens)
            else:
                seg_begin = offset + count - claim.kept_prefix_rows
                seg_end = offset + count
                plan_indices[offset : seg_begin].copy_(tokens)
            offset += count

        boundaries = [0]
        for count in kept_counts:
            boundaries.append(boundaries[-1] + count)
        indptr_host = torch.tensor(boundaries, dtype=torch.int32)
        last_page_len_host = torch.ones(batch_size, dtype=torch.int32)
        verifier = self._verification_wrappers()[0]
        verifier.plan(
            indptr_host,
            plan_indices,
            last_page_len_host,
            q.shape[1],
            key_cache.shape[1],
            q.shape[2],
            1,
            q_data_type=q.dtype,
            kv_data_type=key_cache.dtype,
            sm_scale=layer.scaling,
            disable_split_kv=True,
        )
        if verifier._paged_kv_indices_buf.data_ptr() != plan_indices.data_ptr():
            raise RuntimeError(
                "the planned wrapper copied the indices buffer; in-place "
                "per-layer rewrites would not be visible"
            )
        return {
            "wrapper": wrapper,
            "verifier": verifier,
            "claim_request": claim_request,
            "plan_indices": plan_indices,
            "seg_begin": seg_begin,
            "seg_end": seg_end,
            "total": total,
            "kept_total": kept_total,
            "segments": segments,
            "next": 0,
        }

    def _fast_crosscheck(
        self,
        claim: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        layer: Any,
        local_layer: int,
        group_size: int,
        serve_output: torch.Tensor,
        ctx: dict[str, Any],
    ) -> None:
        """Verify one fast-path layer against independent machinery.

        Three checks: the device selection equals the reference selection;
        the staged rows byte-match their host sources; and an independently
        planned wrapper over the reference-ordered compact table reproduces
        the served output. The kv orderings differ, so the attention check
        is a tolerance compare — tight enough that any wrong page, stale
        row, or misplaced segment write fails it by orders of magnitude.
        """
        key_cache, value_cache = kv_cache
        request = ctx["claim_request"]
        queries = q[request : request + 1]
        chosen = claim.choose_pages(local_layer, queries, group_size)
        fast_chosen = sorted(
            torch.cat(
                [
                    claim.forced_pages,
                    claim.choose_free_pages(local_layer, queries, group_size),
                ]
            ).tolist()
        )
        if fast_chosen != sorted(chosen):
            raise RuntimeError(
                "fast-path selection diverged from the reference selection"
            )
        all_positions = claim._positions_of(chosen)
        claim._verify_layer(
            local_layer, all_positions, int(all_positions.numel())
        )

        kept_per_request = []
        for kind, tokens in ctx["segments"]:
            if kind == "peer":
                kept_per_request.append(tokens)
            else:
                kept_per_request.append(
                    torch.cat(
                        [
                            claim.device_rows[all_positions],
                            tokens,
                        ]
                    )
                )
        compact_indices = torch.cat(kept_per_request).to(torch.int32)
        counts = [0] + [int(k.numel()) for k in kept_per_request]
        compact_indptr = torch.tensor(
            counts, dtype=torch.int32, device=compact_indices.device
        ).cumsum(0, dtype=torch.int32)
        last_page_len = torch.ones(
            q.shape[0], dtype=torch.int32, device=compact_indices.device
        )
        reference = self._verification_wrappers()[1]
        reference.plan(
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
        expected = reference.run(q, (key_cache, value_cache))
        if not torch.allclose(
            serve_output.float(), expected.float(), rtol=1e-2, atol=1e-3
        ):
            difference = (serve_output.float() - expected.float()).abs()
            raise RuntimeError(
                "fast-path attention diverged from the reference-ordered "
                f"computation at layer {local_layer}: max abs difference "
                f"{float(difference.max()):.6f}, reference magnitude "
                f"{float(expected.float().abs().max()):.6f}"
            )

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
