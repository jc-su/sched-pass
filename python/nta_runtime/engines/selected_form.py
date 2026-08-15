"""Selected FlashInfer execution for resident and host-tiered KV tables.

The resident verification mode builds compact per-token tables and compares two
independent wrappers. The tiered mode combines device-side Quest selection,
validated miss compaction, indexed host acquisition, and request-bound
compiler-generated FlashInfer wrappers. It preserves request generation and
supports multiple claimed requests in one batch.

SGLang's HiCache profile stores KV at page size one, so a "page" here is a
logical run of ``page_tokens`` consecutive sequence positions; a selected
table is simply the subset of token-slot indices belonging to selected
logical pages, which FlashInfer consumes without kernel changes.

External prefixes use virtual request-table identities and bounded physical
staging rows. FlashInfer receives only compact physical tables.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from nta_runtime.quest_selector import (
    budgeted_page_selection,
    page_key_envelopes,
    quest_page_scores,
)


class SelectedAttentionExecutor:
    """Own selected-table planning and compiler-generated FlashInfer launches."""

    def __init__(
        self,
        budget_pages: int,
        page_tokens: int,
        *,
        decode_jit_args: list[Any] | None = None,
        prefill_jit_args: list[Any] | None = None,
        register_wrapper: Any = None,
        use_tensor_cores: bool = False,
    ) -> None:
        if budget_pages <= 0 or page_tokens <= 0:
            raise ValueError("selected-form budget and page tokens must be positive")
        self.budget_pages = budget_pages
        self.page_tokens = page_tokens
        self._decode_jit_args = decode_jit_args
        self._prefill_jit_args = prefill_jit_args
        self._register_wrapper = register_wrapper
        self._use_tensor_cores = use_tensor_cores
        configured = (
            decode_jit_args is not None,
            prefill_jit_args is not None,
            register_wrapper is not None,
        )
        if any(configured) and not all(configured):
            raise ValueError(
                "selected compiler execution requires both JIT forms and registration"
            )
        self.compiler_transformed = all(configured)
        self._wrappers: tuple[Any, Any] | None = None
        self._overlap_decode_wrappers: dict[int, tuple[Any, torch.Tensor]] = {}
        self._tiered_batch_contexts: dict[int, dict[str, Any]] = {}
        self._compact_plan_state: dict[str, Any] | None = None
        self._compact_plan_verify = (
            os.environ.get("NTA_SGLANG_COMPACT_PLAN_VERIFY") == "1"
        )
        self._device_plan_enabled = (
            os.environ.get("NTA_SGLANG_DEVICE_PLAN", "1") != "0"
        )

    def begin_tiered_forward(self) -> None:
        """Invalidate wrapper plans at the engine's forward boundary."""
        self._tiered_batch_contexts.clear()

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
                    use_tensor_cores=self._use_tensor_cores,
                    **(
                        {"jit_args": self._decode_jit_args}
                        if self._decode_jit_args is not None
                        else {}
                    ),
                )
                for _ in range(2)
            )
            if self._register_wrapper is not None and self._decode_jit_args is not None:
                for wrapper in self._wrappers:
                    self._register_wrapper(wrapper, self._decode_jit_args[0])
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
                    **(
                        {"jit_args": self._prefill_jit_args}
                        if self._prefill_jit_args is not None
                        else {}
                    ),
                )
                for _ in range(2)
            )
            if self._register_wrapper is not None and self._prefill_jit_args is not None:
                for wrapper in self._prefill_pair:
                    self._register_wrapper(wrapper, self._prefill_jit_args[0])
        return self._prefill_pair

    def _overlap_decode_wrapper(self, owner: Any) -> Any:
        """Return a compiler-generated peer-group wrapper for one owner."""
        key = id(owner)
        cached = self._overlap_decode_wrappers.get(key)
        if cached is not None:
            return cached[0]
        import flashinfer

        workspace = torch.empty(
            32 * 1024 * 1024, dtype=torch.uint8, device="cuda"
        )
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace,
            "NHD",
            backend="fa2",
            use_tensor_cores=bool(owner.use_tensor_cores),
            **(
                {"jit_args": self._decode_jit_args}
                if self._decode_jit_args is not None
                else {}
            ),
        )
        if self._register_wrapper is not None and self._decode_jit_args is not None:
            self._register_wrapper(wrapper, self._decode_jit_args[0])
        self._overlap_decode_wrappers[key] = (wrapper, workspace)
        return wrapper

    def _run_paged(
        self,
        engine: Any,
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        layer: Any,
        *,
        out: torch.Tensor | None = None,
        request_positions: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if not self.compiler_transformed:
            return wrapper.run(q, kv_cache, out=out)
        batch = getattr(engine, "_active_batch", None)
        bindings = () if batch is None else batch.bindings
        request_slots = tuple(
            bindings[position].request_slot
            for position in (
                range(len(bindings))
                if request_positions is None
                else request_positions
            )
        )
        if not request_slots or request_slots != tuple(
            range(request_slots[0], request_slots[0] + len(request_slots))
        ):
            raise RuntimeError(
                "selected attention requires contiguous request bindings"
            )
        engine._phase_program(wrapper)
        result = wrapper.run(
            q,
            kv_cache,
            engine._runtime.device_view_tensor,
            layer.scaling,
            request_slots[0],
            out=out,
        )
        engine._stats["selected_compiler_launches"] = (
            engine._stats.get("selected_compiler_launches", 0) + 1
        )
        return result

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

        if claim.fast_ok and not claim.verify:
            return self._tiered_fast(
                engine, claim, wrapper, q, kv_cache, layer, local_layer,
                group_size, stats, serve_output, stream, prefill,
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
            prefix_mask = claim.table_prefix_mask(tokens)
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
            claim.note_serving(engine, request, prefix_mask)
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
            if claim.external_sidecar:
                selected_rows = claim.page_row_count(chosen)
                kept_per_request.append(
                    torch.cat(
                        [
                            claim.device_rows[:selected_rows],
                            tokens[~prefix_mask],
                        ]
                    )
                )
                continue
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
                outputs.append(
                    self._run_paged(
                        engine, verifier, q, (key_cache, value_cache), layer
                    )
                )
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
                outputs.append(
                    self._run_paged(
                        engine, verifier, q, (key_cache, value_cache), layer
                    )
                )
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

    def evaluate_tiered_claims(
        self,
        engine: Any,
        claims: tuple[Any, ...],
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        layer: Any,
        stats: dict[str, Any],
        serve_output: torch.Tensor,
        prefill: bool,
    ) -> bool:
        """Serve every external-prefix request through one compact plan.

        A dense peer table is unsafe when that peer belongs to another live
        claim, so matching and compaction are batch-wide.  The plan is built
        once at layer zero; every later layer only updates fixed-size claim
        segments and stages newly selected rows.
        """
        if not claims:
            return False
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("tiered serving inside graph capture is unsupported")
        if any(not claim.fast_ok or claim.verify for claim in claims):
            if len(claims) == 1:
                return self.evaluate_tiered(
                    engine,
                    claims[0],
                    wrapper,
                    q,
                    kv_cache,
                    layer,
                    stats,
                    serve_output,
                    prefill,
                )
            raise RuntimeError(
                "verification modes and non-fixed-shape claims are "
                "unsupported with concurrent tiered claims; run the gate "
                "workload with a single external prefix"
            )

        local_layer = int(layer.layer_id) - claims[0].start_layer
        if any(
            claim.start_layer != claims[0].start_layer
            or claim.layer_count != claims[0].layer_count
            for claim in claims
        ):
            raise RuntimeError("concurrent tiered claims disagree on model geometry")
        if not 0 <= local_layer < claims[0].layer_count:
            raise RuntimeError("tiered layer is outside the claimed model")

        claim_ids = tuple(claim.claim_id for claim in claims)
        context_key = id(wrapper)
        ctx = self._tiered_batch_contexts.get(context_key)
        if (
            ctx is None
            or ctx["wrapper"] is not wrapper
            or ctx["prefill"] is not prefill
            or ctx["claim_ids"] != claim_ids
        ):
            ctx = self._build_multi_claim_ctx(
                engine, claims, wrapper, q, kv_cache, layer, prefill
            )
            if ctx is None:
                return False
            self._tiered_batch_contexts[context_key] = ctx

        key_cache, _ = kv_cache
        group_size = q.shape[1] // key_cache.shape[1]
        stream = torch.cuda.current_stream()
        verify_targets: list[tuple[Any, torch.Tensor, torch.Tensor]] = []
        if ctx.get("request_overlap", False):
            copy_events = []
            layer_rows: list[torch.Tensor] = []
            for entry in ctx["claim_entries"]:
                claim = entry["claim"]
                request = entry["request"]
                selected_rows = None if prefill else claim.cached_selected_rows(
                    local_layer
                )
                if selected_rows is None:
                    queries = q[request : request + 1]
                    free = claim.choose_free_pages(
                        local_layer, queries, group_size
                    )
                    selected_rows, copied = claim.stage_missing_async(
                        engine, local_layer, free, stream
                    )
                    claim.remember_selected_rows(local_layer, selected_rows)
                    copy_events.append(copied)
                    if claim.verify_fast:
                        verify_targets.append((claim, free, selected_rows))
                else:
                    stats["tiered_selection_reuse_layers"] = (
                        stats.get("tiered_selection_reuse_layers", 0) + 1
                    )
                layer_rows.append(selected_rows)
                claim.layers_served += 1
            self._write_claim_segments(ctx, layer_rows)
            self._run_paged(
                engine,
                ctx["peer_wrapper"],
                q[ctx["peer_slice"]],
                kv_cache,
                layer,
                out=ctx["peer_output"],
                request_positions=ctx["peer_positions"],
            )
            serve_output[ctx["peer_slice"]].copy_(ctx["peer_output"])
            for copied in copy_events:
                stream.wait_event(copied)
            self._run_paged(
                engine,
                ctx["verifier"],
                q[ctx["claim_slice"]],
                kv_cache,
                layer,
                out=ctx["external_output"],
                request_positions=ctx["claim_positions"],
            )
            serve_output[ctx["claim_slice"]].copy_(ctx["external_output"])
            stats["tiered_request_overlap_layers"] = (
                stats.get("tiered_request_overlap_layers", 0) + 1
            )
            stats["tiered_request_overlap_peer_requests"] = (
                stats.get("tiered_request_overlap_peer_requests", 0)
                + len(ctx["peer_positions"])
            )
        else:
            if local_layer == 0:
                for entry in ctx["claim_entries"]:
                    entry["claim"].advance_decode_step()
                state = self._compact_plan_state
                fast_step = bool(
                    ctx.get("device_plan")
                    and state is not None
                    and ctx["claim_entries"]
                    and all(
                        entry["claim"].reuse_cache_complete
                        and not entry["claim"].refresh_due()
                        for entry in ctx["claim_entries"]
                    )
                )
                ctx["fast_reuse_step"] = fast_step
                if fast_step:
                    for entry in ctx["claim_entries"]:
                        entry["claim"].layers_served += entry[
                            "claim"
                        ].layer_count
            if ctx.get("fast_reuse_step"):
                # Full-reuse step: every claim's per-layer rows are cached
                # in stable buffers, so the layer's claim block is one
                # index_copy of a concatenation reused until the next
                # refresh or staging invalidates it. No per-claim Python
                # runs between layer zero's check and attention.
                state = self._compact_plan_state
                cats = state.setdefault("layer_cats", {})
                cat = cats.get(local_layer)
                if cat is None:
                    cat = torch.cat(
                        [
                            entry["claim"]._selected_row_cache[local_layer]
                            for entry in ctx["claim_entries"]
                        ]
                    )
                    cats[local_layer] = cat
                positions = ctx.get("claim_row_positions")
                if positions is None:
                    entry = ctx["claim_entries"][0]
                    ctx["plan_indices"][entry["begin"] : entry["end"]].copy_(
                        cat
                    )
                else:
                    ctx["plan_indices"].index_copy_(0, positions, cat)
                self._run_paged(
                    engine, ctx["verifier"], q, kv_cache, layer,
                    out=serve_output,
                )
                stats["tiered_selection_reuse_layers"] = stats.get(
                    "tiered_selection_reuse_layers", 0
                ) + len(ctx["claim_entries"])
                stats["tiered_fast_reuse_layers"] = (
                    stats.get("tiered_fast_reuse_layers", 0) + 1
                )
                kind = "tiered_decode"
                stats[f"{kind}_layers"] = stats.get(f"{kind}_layers", 0) + 1
                stats["tiered_fast_layers"] = (
                    stats.get("tiered_fast_layers", 0) + 1
                )
                stats["tiered_concurrent_claims_max"] = max(
                    stats.get("tiered_concurrent_claims_max", 0),
                    len(ctx["claim_entries"]),
                )
                stats["tiered_tokens_total"] = (
                    stats.get("tiered_tokens_total", 0) + ctx["total"]
                )
                stats["tiered_tokens_kept"] = (
                    stats.get("tiered_tokens_kept", 0) + ctx["kept_total"]
                )
                return True
            layer_rows: list[torch.Tensor | None] = []
            staging: list[tuple[Any, torch.Tensor]] = []
            staged_positions: list[int] = []
            for position, entry in enumerate(ctx["claim_entries"]):
                claim = entry["claim"]
                request = entry["request"]
                if prefill:
                    query_begin, query_end = ctx["query_ranges"][request]
                    queries = q[query_begin:query_end]
                else:
                    queries = q[request : request + 1]
                selected_rows = None if prefill else claim.cached_selected_rows(
                    local_layer
                )
                host_orchestrated = bool(
                    getattr(engine, "_host_orchestrated", False)
                )
                if selected_rows is None and (
                    prefill
                    and not host_orchestrated
                    and claim.external_sidecar
                    and not claim.verify
                    and not claim.verify_fast
                    and claim.selection_refresh_interval > 1
                    and claim.free_budget > 0
                    and claim.selected_rows is not None
                    and claim.copy_stream is not None
                ):
                    # Wavefront extend: each layer's selection, prep, and
                    # transfer are issued on the claim's prep stream the
                    # moment that layer's queries exist — selection stays
                    # on per-layer queries (layer-zero queries scored
                    # against other layers' envelopes proved to be noise:
                    # quality 0.0 versus stock 1.0), and the compute
                    # stream pays one event record per layer instead of
                    # the old synchronous host round trip or the invalid
                    # all-layers burst.
                    if local_layer == 0:
                        span = getattr(claim, "_extend_span", None)
                        if span is None:
                            span = (torch.cuda.Event(True), torch.cuda.Event(True))
                            claim._extend_span = span
                        span[0].record(stream)
                    selected_rows = claim.stage_layer_offstream(
                        engine, local_layer, queries, group_size, stream
                    )
                elif selected_rows is None:
                    free = claim.choose_free_pages(local_layer, queries, group_size)
                    if host_orchestrated and claim.external_sidecar:
                        # RQ3 baseline B1: identical selection and transfer,
                        # orchestration through the host control edge. The
                        # device claim chain (fused selection, table prep,
                        # wavefront extend) is bypassed entirely.
                        selected_rows = claim.stage_layer_host_orchestrated(
                            engine, local_layer, free, stream
                        )
                        fallback_state = self._compact_plan_state
                        if fallback_state is not None:
                            fallback_state.get("layer_cats", {}).pop(
                                local_layer, None
                            )
                        claim.remember_selected_rows(local_layer, selected_rows)
                        if claim.verify:
                            positions = claim.kept_prefix_positions(free)
                            claim._verify_layer(
                                local_layer,
                                positions,
                                int(positions.numel()),
                                physical_rows=selected_rows,
                            )
                        elif claim.verify_fast:
                            verify_targets.append((claim, free, selected_rows))
                    elif (
                        not prefill
                        and claim.external_sidecar
                        and getattr(claim, "table_backed", False)
                        and getattr(claim, "table_slot", None) is not None
                    ):
                        # Deferred to the single table-driven prep launch
                        # once every staging claim's pages are known.
                        staging.append((claim, free))
                        staged_positions.append(position)
                    else:
                        selected_rows = claim.stage_missing(
                            engine, local_layer, free, stream
                        )
                        fallback_state = self._compact_plan_state
                        if fallback_state is not None:
                            fallback_state.get("layer_cats", {}).pop(
                                local_layer, None
                            )
                        claim.remember_selected_rows(local_layer, selected_rows)
                        if claim.verify_fast:
                            verify_targets.append((claim, free, selected_rows))
                else:
                    stats["tiered_selection_reuse_layers"] = (
                        stats.get("tiered_selection_reuse_layers", 0) + 1
                    )
                layer_rows.append(selected_rows)
                claim.layers_served += 1
            if staging:
                copy_events = self._stage_claims_via_table(
                    engine, ctx, staging, local_layer, stream
                )
                for (claim, free), position in zip(
                    staging, staged_positions, strict=True
                ):
                    selected_rows = claim.selected_rows[: claim.kept_prefix_rows]
                    claim.remember_selected_rows(local_layer, selected_rows)
                    if claim.verify_fast:
                        verify_targets.append((claim, free, selected_rows))
                    layer_rows[position] = selected_rows
                for copied in copy_events:
                    stream.wait_event(copied)
            self._write_claim_segments(ctx, layer_rows)
            self._run_paged(
                engine, ctx["verifier"], q, kv_cache, layer, out=serve_output
            )
            if prefill and local_layer == claims[0].layer_count - 1:
                for entry in ctx["claim_entries"]:
                    span = getattr(entry["claim"], "_extend_span", None)
                    if span is not None:
                        span[1].record(stream)
                        entry["claim"]._extend_span_armed = True

        if verify_targets:
            # VERIFY=fast on the live path: every freshly staged layer is
            # byte-verified against its pinned host source after all copy
            # waits have been enqueued. Reuse layers were verified when
            # first staged; the bounded cache owns their slots until
            # eviction, so their bytes cannot change underneath.
            for verified_claim, free, physical_rows in verify_targets:
                positions = verified_claim.kept_prefix_positions(free)
                verified_claim._verify_layer(
                    local_layer,
                    positions,
                    int(positions.numel()),
                    physical_rows=physical_rows,
                )
                stats["tiered_fast_checked_layers"] = (
                    stats.get("tiered_fast_checked_layers", 0) + 1
                )
        stats["tiered_rows_copied"] = stats.get(
            "tiered_rows_copied_released", 0
        ) + sum(claim.rows_copied for claim in claims)
        stats["tiered_rows_rehit"] = stats.get(
            "tiered_rows_rehit_released", 0
        ) + sum(claim.rows_rehit for claim in claims)
        kind = "tiered_prefill" if prefill else "tiered_decode"
        stats[f"{kind}_layers"] = stats.get(f"{kind}_layers", 0) + 1
        stats["tiered_fast_layers"] = stats.get("tiered_fast_layers", 0) + 1
        stats["tiered_concurrent_claims_max"] = max(
            stats.get("tiered_concurrent_claims_max", 0),
            len(ctx["claim_entries"]),
        )
        stats["tiered_tokens_total"] = (
            stats.get("tiered_tokens_total", 0) + ctx["total"]
        )
        stats["tiered_tokens_kept"] = (
            stats.get("tiered_tokens_kept", 0) + ctx["kept_total"]
        )
        return True

    def _write_claim_segments(
        self, ctx: dict[str, Any], layer_rows: list[torch.Tensor]
    ) -> None:
        """Write every claim's selected rows for one layer.

        At high claim concurrency, per-claim slice copies dominated decode
        (~540 launches per step at fifteen claims); one concatenation and
        one index_copy through the per-forward destination map replace
        them with two launches per layer.
        """
        entries = ctx["claim_entries"]
        if not entries:
            return
        if len(entries) == 1 or ctx.get("claim_row_positions") is None:
            for entry, rows in zip(entries, layer_rows, strict=True):
                ctx["plan_indices"][entry["begin"] : entry["end"]].copy_(rows)
            return
        ctx["plan_indices"].index_copy_(
            0, ctx["claim_row_positions"], torch.cat(layer_rows)
        )

    def _stage_claims_via_table(
        self,
        engine: Any,
        ctx: dict[str, Any],
        staging: list[tuple[Any, torch.Tensor]],
        local_layer: int,
        stream: Any,
    ) -> list[Any]:
        """Stage every claim missing this layer with one prep launch.

        The per-claim ``prepare_bounded`` launches were the measured
        host-bound cost (~0.56 ms per claim per staging layer): each is a
        small kernel whose launch latency serializes on the CPU. Here the
        page sets and selection counts land in the claim table through a
        fixed number of batched writes, one fixed-shape kernel walks every
        valid row, and only the per-claim host-transfer progress launches
        remain on the claims' copy streams. Returns the per-claim copy
        events the compute stream must wait on before attention.
        """
        table = engine._claim_table
        state = self._compact_plan_state
        if state is not None:
            state.get("layer_cats", {}).pop(local_layer, None)
        pieces: list[torch.Tensor] = []
        dests: list[torch.Tensor] = []
        slots: list[torch.Tensor] = []
        counts: list[torch.Tensor] = []
        for claim, free in staging:
            expected = (
                int(claim.full_forced_pages.numel()) + int(free.numel()) + 1
            )
            if getattr(claim, "_table_page_count", None) is None:
                index = claim.table_slot.index
                device = table.selected_pages.device
                claim._table_page_count = expected
                claim._table_slot_dev = torch.tensor(
                    [index], dtype=torch.int64, device=device
                )
                claim._table_count_dev = torch.tensor(
                    [expected], dtype=torch.int32, device=device
                )
                claim._table_page_dest = index * table.max_budget_pages + (
                    torch.arange(expected, dtype=torch.int64, device=device)
                )
            elif claim._table_page_count != expected:
                raise RuntimeError(
                    f"claim {claim.claim_id} changed its selection shape: "
                    f"{claim._table_page_count} pages planned, {expected} "
                    "offered; fixed-shape staging cannot follow"
                )
            pieces.extend((claim.full_forced_pages, free, claim.forced_pages[-1:]))
            dests.append(claim._table_page_dest)
            slots.append(claim._table_slot_dev)
            counts.append(claim._table_count_dev)
        # Rows not staging this layer must present a zero count: their
        # per-layer bounded caches would otherwise chase a stale page set.
        table.selected_counts.zero_()
        table.selected_counts.index_copy_(0, torch.cat(slots), torch.cat(counts))
        table.selected_pages.view(-1).index_copy_(
            0, torch.cat(dests), torch.cat(pieces)
        )
        phases = engine._phase_program(engine._nta_demand_decode_wrappers[0])
        phases.prepare_claim_table_selected_rows(
            engine._runtime, table, local_layer, stream=stream
        )
        ready = ctx.get("table_prep_ready")
        if ready is None:
            ready = torch.cuda.Event()
            ctx["table_prep_ready"] = ready
        ready.record(stream)
        events: list[Any] = []
        for claim, _ in staging:
            base = claim.first_object_slot + 2 * local_layer
            claim.copy_stream.wait_event(ready)
            phases.progress_validated_indexed_host_range(
                engine._runtime, base, 2, stream=claim.copy_stream
            )
            copied = claim.copy_ready[local_layer]
            copied.record(claim.copy_stream)
            events.append(copied)
            claim.device_accounting = True
            claim.requested_rows += claim.kept_prefix_rows
        stats = engine._stats
        stats["tiered_device_compaction_launches"] = (
            stats.get("tiered_device_compaction_launches", 0) + 1
        )
        stats["tiered_bounded_cache_launches"] = (
            stats.get("tiered_bounded_cache_launches", 0) + 1
        )
        stats["tiered_device_selected_pages"] = stats.get(
            "tiered_device_selected_pages", 0
        ) + sum(claim._table_page_count for claim, _ in staging)
        stats["tiered_table_prep_launches"] = (
            stats.get("tiered_table_prep_launches", 0) + 1
        )
        stats["tiered_table_staged_claims"] = (
            stats.get("tiered_table_staged_claims", 0) + len(staging)
        )
        return events


    def _device_compact_plan(
        self,
        engine: Any,
        wrapper: Any,
        segments: list[tuple[Any | None, torch.Tensor | None, int]],
        kv_lengths: tuple[int, ...],
        kept_counts: list[int],
        total: int,
        batch_size: int,
        device: Any,
    ) -> torch.Tensor:
        """Build the packed compact plan with one descriptor-driven launch.

        The Python plan build cost three launches per claim request per
        step (index_select, cat, segment copy) plus a fresh allocation;
        here descriptors persist across steps for a fixed batch
        composition, the per-step host work is two cumulative sums and
        one pinned-buffer upload, and one kernel writes every remainder.
        Claim kept-prefix blocks stay per-layer writes. The output buffer
        is persistent so the retained FlashInfer plan pointer — and later
        a captured decode step — survive across steps.
        """
        key = (
            id(wrapper),
            tuple(getattr(engine, "_current_request_ids", ())),
            tuple(
                claim.claim_id if claim is not None else 0
                for claim, _, _ in segments
            ),
        )
        state = self._compact_plan_state
        if state is None or state["key"] != key:
            bound_lengths = []
            claim_rows = []
            np_pieces = []
            np_offsets = [0]
            for claim, _, _ in segments:
                if claim is None:
                    bound_lengths.append(0)
                    claim_rows.append(0)
                    np_offsets.append(np_offsets[-1])
                else:
                    bound_lengths.append(claim.bound_length)
                    claim_rows.append(claim.kept_prefix_rows)
                    np_pieces.append(
                        claim.bound_nonprefix_index.to(torch.int32)
                    )
                    np_offsets.append(
                        np_offsets[-1]
                        + int(claim.bound_nonprefix_index.numel())
                    )
            np_indices = (
                torch.cat(np_pieces).contiguous()
                if np_pieces and np_offsets[-1] > 0
                else torch.zeros(1, dtype=torch.int32, device=device)
            )
            state = {
                "key": key,
                "bound_lengths": torch.tensor(
                    bound_lengths, dtype=torch.int32, device=device
                ),
                "claim_rows": torch.tensor(
                    claim_rows, dtype=torch.int32, device=device
                ),
                "np_offsets": torch.tensor(
                    np_offsets, dtype=torch.int32, device=device
                ),
                "np_indices": np_indices,
                "pinned": torch.empty(
                    2 * batch_size + 2, dtype=torch.int32, pin_memory=True
                ),
                "offsets": torch.empty(
                    2 * batch_size + 2, dtype=torch.int32, device=device
                ),
                "compact": torch.empty(
                    sum(kept_counts) + 1024 * batch_size,
                    dtype=torch.int32,
                    device=device,
                ),
            }
            self._compact_plan_state = state
        kept_total = sum(kept_counts)
        if state["compact"].numel() < kept_total:
            state["compact"] = torch.empty(
                kept_total + 1024 * batch_size,
                dtype=torch.int32,
                device=device,
            )
        pinned = state["pinned"]
        pinned[0] = 0
        pinned[batch_size + 1] = 0
        dense = 0
        compact = 0
        for index in range(batch_size):
            dense += int(kv_lengths[index])
            compact += kept_counts[index]
            pinned[index + 1] = dense
            pinned[batch_size + 2 + index] = compact
        if dense != total:
            raise RuntimeError("compact-plan dense lengths disagree with the plan")
        # Synchronous for the same pinned-reuse race the graph path hit:
        # the buffer is rewritten next step while an async upload may
        # still be in flight.
        state["offsets"].copy_(pinned)
        phases = engine._phase_program(engine._nta_demand_decode_wrappers[0])
        phases.build_compact_plan(
            wrapper._paged_kv_indices_buf,
            state["offsets"][: batch_size + 1],
            state["bound_lengths"],
            state["np_offsets"],
            state["np_indices"],
            state["claim_rows"],
            state["offsets"][batch_size + 1 :],
            state["compact"],
            batch_size,
            stream=torch.cuda.current_stream(),
        )
        stats = engine._stats
        stats["tiered_device_plan_builds"] = (
            stats.get("tiered_device_plan_builds", 0) + 1
        )
        if self._compact_plan_verify:
            # Debug dual build: reconstruct every remainder with the
            # reference Python ops and require byte equality. Synchronizes;
            # never enable on a measured run.
            reference = torch.zeros(
                kept_total, dtype=torch.int32, device=device
            )
            indices = wrapper._paged_kv_indices_buf
            dense_begin = 0
            offset = 0
            for (claim, _, request), count in zip(
                segments, kept_counts, strict=True
            ):
                length = int(kv_lengths[request])
                tokens = indices[dense_begin : dense_begin + length]
                if claim is None:
                    reference[offset : offset + count].copy_(tokens)
                else:
                    remainder = torch.cat(
                        [
                            tokens[: claim.bound_length].index_select(
                                0, claim.bound_nonprefix_index
                            ),
                            tokens[claim.bound_length :],
                        ]
                    )
                    begin = offset + claim.kept_prefix_rows
                    reference[begin : offset + count].copy_(remainder)
                    reference[offset : begin].copy_(
                        state["compact"][offset : begin]
                    )
                dense_begin += length
                offset += count
            if not torch.equal(reference, state["compact"][:kept_total]):
                raise RuntimeError(
                    "device compact plan disagrees with the reference build"
                )
            stats["tiered_device_plan_verified"] = (
                stats.get("tiered_device_plan_verified", 0) + 1
            )
        return state["compact"]

    def _build_multi_claim_ctx(
        self,
        engine: Any,
        claims: tuple[Any, ...],
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        layer: Any,
        prefill: bool,
    ) -> dict[str, Any] | None:
        key_cache, _ = kv_cache
        kv_lengths = tuple(getattr(engine, "_current_kv_lengths", ()))
        query_lengths = tuple(getattr(engine, "_current_query_lengths", ()))
        if prefill:
            batch_size = int(getattr(wrapper, "_batch_size", 1) or 1)
            qo_indptr = wrapper._qo_indptr_buf[: batch_size + 1]
            if len(query_lengths) != batch_size or sum(query_lengths) != q.shape[0]:
                raise RuntimeError("tiered extend query offsets disagree with Q")
            query_boundaries = [0]
            for length in query_lengths:
                query_boundaries.append(query_boundaries[-1] + length)
            query_ranges = tuple(
                (query_boundaries[index], query_boundaries[index + 1])
                for index in range(batch_size)
            )
        else:
            batch_size = q.shape[0]
            qo_indptr = None
            query_ranges = tuple((index, index + 1) for index in range(batch_size))
        if len(kv_lengths) != batch_size:
            raise RuntimeError("tiered batch omitted CPU KV lengths")
        boundaries = [0]
        for length in kv_lengths:
            boundaries.append(boundaries[-1] + length)
        total = boundaries[-1]
        indices = wrapper._paged_kv_indices_buf[:total]
        segments: list[tuple[Any | None, torch.Tensor | None, int]] = []
        matched_claims: set[int] = set()
        request_ids = getattr(engine, "_current_request_ids", ())
        if len(request_ids) != batch_size:
            raise RuntimeError("tiered batch omitted request identities")
        # Once every live claim is bound, remainders are gathered by the
        # compact-plan kernel in one launch; the Python cat/index_select
        # per claim request per step exists only for first-serve steps.
        split_overlap = not any(
            int(getattr(claim, "selection_refresh_interval", 1)) > 1
            for claim in claims
        )
        device_plan = (
            self._device_plan_enabled
            and not prefill
            and not split_overlap
            and all(
                getattr(claim, "bound_nonprefix_index", None) is not None
                for claim in claims
            )
        )
        for request in range(batch_size):
            begin = boundaries[request]
            end = boundaries[request + 1]
            tokens = indices[begin:end]
            matched = None
            remainder = tokens
            for claim in claims:
                if (
                    claim.request_id is not None
                    and claim.request_id != request_ids[request]
                ):
                    continue
                if claim.request_id is not None and getattr(
                    claim, "bound_nonprefix_index", None
                ) is not None:
                    # Bound claim: remainder sizes are known analytically —
                    # frozen non-prefix positions plus everything appended
                    # past the bound length. No masked_select, no host
                    # synchronization on the per-forward path.
                    if tokens.numel() < claim.bound_length:
                        raise RuntimeError(
                            "tiered request table shrank below its bound prefix"
                        )
                    if matched is not None:
                        raise RuntimeError(
                            "one request matched two live tiered claims"
                        )
                    if claim.claim_id in matched_claims:
                        raise RuntimeError(
                            "one tiered claim matched two requests"
                        )
                    matched = claim
                    matched_claims.add(claim.claim_id)
                    if device_plan:
                        # The compact-plan kernel gathers this remainder on
                        # device; its length is analytic.
                        remainder = None
                    else:
                        remainder = torch.cat(
                            [
                                tokens[: claim.bound_length].index_select(
                                    0, claim.bound_nonprefix_index
                                ),
                                tokens[claim.bound_length :],
                            ]
                        )
                    claim.note_serving(engine, request)
                    continue
                prefix_mask = claim.table_prefix_mask(tokens)
                if claim.request_id is None:
                    prefix_count = int(prefix_mask.sum())
                    if prefix_count == 0:
                        continue
                    if prefix_count != claim.token_count:
                        # Before first use, unrelated requests can contain a
                        # few recycled slot numbers from this claim. Identity
                        # becomes authoritative after the complete match.
                        continue
                if matched is not None:
                    raise RuntimeError("one request matched two live tiered claims")
                if claim.claim_id in matched_claims:
                    raise RuntimeError("one tiered claim matched two requests")
                matched = claim
                matched_claims.add(claim.claim_id)
                remainder = tokens[~prefix_mask]
                claim.note_serving(engine, request, prefix_mask)
            segments.append((matched, remainder, request))
        if not matched_claims:
            return None

        claim_positions = tuple(
            request for claim, _, request in segments if claim is not None
        )
        peer_positions = tuple(
            request for claim, _, request in segments if claim is None
        )
        contiguous_claims = bool(claim_positions) and claim_positions == tuple(
            range(claim_positions[0], claim_positions[-1] + 1)
        )
        contiguous_peers = bool(peer_positions) and peer_positions == tuple(
            range(peer_positions[0], peer_positions[-1] + 1)
        )
        if not prefill and contiguous_claims and contiguous_peers and split_overlap:
            return self._build_request_overlap_ctx(
                claims,
                wrapper,
                q,
                key_cache,
                layer,
                segments,
                claim_positions,
                peer_positions,
                total,
            )

        kept_counts = []
        for (claim, tokens, request) in segments:
            if claim is None:
                kept_counts.append(int(kv_lengths[request]))
            elif device_plan:
                kept_counts.append(
                    claim.kept_prefix_rows
                    + int(claim.bound_nonprefix_index.numel())
                    + int(kv_lengths[request])
                    - claim.bound_length
                )
            else:
                kept_counts.append(
                    int(tokens.numel()) + claim.kept_prefix_rows
                )
        kept_total = sum(kept_counts)
        claim_entries = []
        offset = 0
        if device_plan:
            plan_indices = self._device_compact_plan(
                engine, wrapper, segments, kv_lengths, kept_counts, total,
                batch_size, q.device,
            )[:kept_total]
            for (claim, _, request), count in zip(
                segments, kept_counts, strict=True
            ):
                if claim is not None:
                    claim_entries.append(
                        {
                            "claim": claim,
                            "request": request,
                            "begin": offset,
                            "end": offset + claim.kept_prefix_rows,
                        }
                    )
                offset += count
        else:
            plan_indices = torch.empty(
                kept_total, dtype=torch.int32, device=q.device
            )
            for (claim, tokens, request), count in zip(
                segments, kept_counts, strict=True
            ):
                if claim is None:
                    plan_indices[offset : offset + count].copy_(tokens)
                else:
                    claim_end = offset + claim.kept_prefix_rows
                    plan_indices[claim_end : offset + count].copy_(tokens)
                    claim_entries.append(
                        {
                            "claim": claim,
                            "request": request,
                            "begin": offset,
                            "end": claim_end,
                        }
                    )
                offset += count

        boundaries = [0]
        for count in kept_counts:
            boundaries.append(boundaries[-1] + count)
        indptr = torch.tensor(boundaries, dtype=torch.int32)
        last_page_len = torch.ones(batch_size, dtype=torch.int32)
        verifier = wrapper
        if prefill:
            verifier.plan(
                qo_indptr,
                indptr,
                plan_indices,
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
        else:
            verifier.plan(
                indptr,
                plan_indices,
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
        if verifier._paged_kv_indices_buf.data_ptr() != plan_indices.data_ptr():
            raise RuntimeError("FlashInfer did not retain the compact indices buffer")
        claim_row_positions = (
            torch.cat(
                [
                    torch.arange(
                        entry["begin"], entry["end"],
                        dtype=torch.int64, device=q.device,
                    )
                    for entry in claim_entries
                ]
            )
            if len(claim_entries) > 1
            else None
        )
        return {
            "wrapper": wrapper,
            "verifier": verifier,
            "device_plan": device_plan,
            "claim_ids": tuple(claim.claim_id for claim in claims),
            "claim_entries": claim_entries,
            "claim_row_positions": claim_row_positions,
            "query_ranges": query_ranges,
            "plan_indices": plan_indices,
            "total": total,
            "kept_total": kept_total,
            "prefill": prefill,
        }

    def _build_request_overlap_ctx(
        self,
        claims: tuple[Any, ...],
        wrapper: Any,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        layer: Any,
        segments: list[tuple[Any | None, torch.Tensor, int]],
        claim_positions: tuple[int, ...],
        peer_positions: tuple[int, ...],
        total: int,
    ) -> dict[str, Any]:
        """Plan compiler-generated peer and external request groups."""
        claim_segments = [
            (claim, tokens, request)
            for claim, tokens, request in segments
            if claim is not None
        ]
        peer_segments = [
            (tokens, request)
            for claim, tokens, request in segments
            if claim is None
        ]
        external_counts = [
            claim.kept_prefix_rows + int(tokens.numel())
            for claim, tokens, _ in claim_segments
        ]
        peer_counts = [int(tokens.numel()) for tokens, _ in peer_segments]
        if min(external_counts + peer_counts) <= 0:
            raise RuntimeError(
                "request overlap requires non-empty peer and external tables"
            )

        external_total = sum(external_counts)
        external_indices = torch.empty(
            external_total, dtype=torch.int32, device=q.device
        )
        claim_entries = []
        offset = 0
        for (claim, tokens, request), count in zip(
            claim_segments, external_counts, strict=True
        ):
            selected_end = offset + claim.kept_prefix_rows
            external_indices[selected_end : offset + count].copy_(tokens)
            claim_entries.append(
                {
                    "claim": claim,
                    "request": request,
                    "begin": offset,
                    "end": selected_end,
                }
            )
            offset += count
        external_boundaries = [0]
        for count in external_counts:
            external_boundaries.append(external_boundaries[-1] + count)
        wrapper.plan(
            torch.tensor(external_boundaries, dtype=torch.int32),
            external_indices,
            torch.ones(len(claim_segments), dtype=torch.int32),
            q.shape[1],
            key_cache.shape[1],
            q.shape[2],
            1,
            q_data_type=q.dtype,
            kv_data_type=key_cache.dtype,
            sm_scale=layer.scaling,
            disable_split_kv=True,
        )
        if wrapper._paged_kv_indices_buf.data_ptr() != external_indices.data_ptr():
            raise RuntimeError(
                "FlashInfer did not retain the external-group indices buffer"
            )

        peer_indices = torch.cat([tokens for tokens, _ in peer_segments]).to(
            torch.int32
        )
        peer_boundaries = [0]
        for count in peer_counts:
            peer_boundaries.append(peer_boundaries[-1] + count)
        peer_wrapper = self._overlap_decode_wrapper(wrapper)
        peer_wrapper.plan(
            torch.tensor(peer_boundaries, dtype=torch.int32),
            peer_indices,
            torch.ones(len(peer_segments), dtype=torch.int32),
            q.shape[1],
            key_cache.shape[1],
            q.shape[2],
            1,
            q_data_type=q.dtype,
            kv_data_type=key_cache.dtype,
            sm_scale=layer.scaling,
            disable_split_kv=True,
        )
        if peer_wrapper._paged_kv_indices_buf.data_ptr() != peer_indices.data_ptr():
            raise RuntimeError(
                "FlashInfer did not retain the peer-group indices buffer"
            )

        return {
            "wrapper": wrapper,
            "verifier": wrapper,
            "peer_wrapper": peer_wrapper,
            "claim_ids": tuple(claim.claim_id for claim in claims),
            "claim_entries": claim_entries,
            "claim_positions": claim_positions,
            "peer_positions": peer_positions,
            "claim_slice": slice(claim_positions[0], claim_positions[-1] + 1),
            "peer_slice": slice(peer_positions[0], peer_positions[-1] + 1),
            "plan_indices": external_indices,
            "peer_indices": peer_indices,
            "external_output": torch.empty_like(
                q[claim_positions[0] : claim_positions[-1] + 1]
            ),
            "peer_output": torch.empty_like(
                q[peer_positions[0] : peer_positions[-1] + 1]
            ),
            "total": total,
            "kept_total": external_total + sum(peer_counts),
            "prefill": False,
            "request_overlap": True,
        }

    def _tiered_fast(
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
        prefill: bool,
    ) -> bool:
        """Plan-once serving for both extend and decode.

        The kept row count is layer-invariant (retention always keeps the
        tail page), so the serving wrapper is planned exactly once per
        forward against an indices buffer this class owns; each layer
        rewrites only the claim's segment of that buffer from the
        device-side selection and runs attention. FlashInfer's planners
        consume indptr lengths only — indices are read from the device
        buffer at run time — which is what makes the in-place rewrite
        sound. The claim segment leads each request's table: extend is
        causal and only the trailing ``qo_len`` kv positions are
        progressively masked, so every prefix row must precede the suffix,
        while decode attention is order-free either way.
        """
        ctx = claim.ctx
        if (
            ctx is None
            or local_layer == 0
            or ctx["wrapper"] is not wrapper
            or ctx["prefill"] is not prefill
            or ctx["next"] != local_layer
        ):
            if local_layer != 0:
                raise RuntimeError(
                    "tiered fast path entered mid-forward at layer "
                    f"{local_layer} without a forward context"
                )
            ctx = self._build_fast_ctx(
                engine, claim, wrapper, q, kv_cache, layer, prefill
            )
            claim.ctx = ctx
        ctx["next"] = local_layer + 1

        request = ctx["claim_request"]
        queries = q if prefill else q[request : request + 1]
        free = claim.choose_free_pages(local_layer, queries, group_size)
        selected_rows = claim.stage_missing(engine, local_layer, free, stream)
        ctx["plan_indices"][ctx["seg_begin"] : ctx["seg_end"]].copy_(
            selected_rows
        )
        self._run_paged(
            engine, ctx["verifier"], q, kv_cache, layer, out=serve_output
        )
        if claim.verify_fast:
            self._fast_crosscheck(
                engine, claim, q, kv_cache, layer, local_layer, group_size,
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
        kind = "tiered_prefill" if prefill else "tiered_decode"
        stats[f"{kind}_layers"] = stats.get(f"{kind}_layers", 0) + 1
        stats["tiered_fast_layers"] = stats.get("tiered_fast_layers", 0) + 1
        stats["tiered_tokens_total"] = (
            stats.get("tiered_tokens_total", 0) + ctx["total"]
        )
        stats["tiered_tokens_kept"] = (
            stats.get("tiered_tokens_kept", 0) + ctx["kept_total"]
        )
        return True

    def _build_fast_ctx(
        self,
        engine: Any,
        claim: Any,
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        layer: Any,
        prefill: bool,
    ) -> dict[str, Any]:
        """Once per forward: identify the claim request, lay out the compact
        indices buffer (peer tables and non-claim tokens are layer-invariant),
        and plan the single serving wrapper against it."""
        key_cache, _ = kv_cache
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
            prefix_mask = claim.table_prefix_mask(tokens)
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
            # mid-table). The non-claim remainder keeps its original order,
            # which for extend leaves the causal suffix trailing as
            # causality requires.
            claim_request = request
            claim.note_serving(engine, request, prefix_mask)
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
                seg_begin = offset
                seg_end = offset + claim.kept_prefix_rows
                plan_indices[seg_end : offset + count].copy_(tokens)
            offset += count

        boundaries = [0]
        for count in kept_counts:
            boundaries.append(boundaries[-1] + count)
        indptr_host = torch.tensor(boundaries, dtype=torch.int32)
        last_page_len_host = torch.ones(batch_size, dtype=torch.int32)
        if prefill:
            verifier = self._prefill_wrappers()[0]
            qo_indptr_host = torch.tensor(
                [0, q.shape[0]], dtype=torch.int32
            )
            verifier.plan(
                qo_indptr_host,
                indptr_host,
                plan_indices,
                last_page_len_host,
                q.shape[1],
                key_cache.shape[1],
                q.shape[2],
                1,
                causal=True,
                sm_scale=layer.scaling,
                q_data_type=q.dtype,
                kv_data_type=key_cache.dtype,
            )
        else:
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
            "prefill": prefill,
            "next": 0,
        }

    def _fast_crosscheck(
        self,
        engine: Any,
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
        planned wrapper over a differently-ordered compact table reproduces
        the served output. The kv orderings differ, so the attention check
        is a tolerance compare — tight enough that any wrong page, stale
        row, or misplaced segment write fails it by orders of magnitude.
        """
        key_cache, value_cache = kv_cache
        prefill = ctx["prefill"]
        request = ctx["claim_request"]
        queries = q if prefill else q[request : request + 1]
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
        physical_rows = claim.physical_rows_for_pages(local_layer, chosen)
        claim._verify_layer(
            local_layer,
            all_positions,
            int(all_positions.numel()),
            physical_rows,
        )

        kept_per_request = []
        for kind, tokens in ctx["segments"]:
            if kind == "peer":
                kept_per_request.append(tokens)
            else:
                kept_per_request.append(
                    torch.cat(
                        [
                            physical_rows,
                            tokens,
                        ]
                    )
                )
        compact_indices = torch.cat(kept_per_request).to(torch.int32)
        counts = [0] + [int(k.numel()) for k in kept_per_request]
        compact_indptr = torch.tensor(
            counts, dtype=torch.int32, device=compact_indices.device
        ).cumsum(0, dtype=torch.int32)
        batch_size = 1 if prefill else q.shape[0]
        last_page_len = torch.ones(
            batch_size, dtype=torch.int32, device=compact_indices.device
        )
        if prefill:
            reference = self._prefill_wrappers()[1]
            qo_indptr = torch.tensor(
                [0, q.shape[0]], dtype=torch.int32,
                device=compact_indices.device,
            )
            reference.plan(
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
        else:
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
        expected = self._run_paged(
            engine, reference, q, (key_cache, value_cache), layer
        )
        # Tolerance is calibrated to fp16 reduction reordering, which the
        # differing kv orders legitimately produce: a 16K-term extend was
        # observed to differ by exactly one ulp at the output magnitude
        # (3.9e-3 at 6.7). Real staging or selection errors diverge at the
        # output's own scale, orders of magnitude above this bound.
        if not torch.allclose(
            serve_output.float(), expected.float(), rtol=2e-2, atol=8e-3
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
        engine: Any,
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
            outputs.append(
                self._run_paged(
                    engine, verifier, q, (key_cache, value_cache), layer
                )
            )
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
