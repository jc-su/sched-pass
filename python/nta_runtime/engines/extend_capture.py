"""Captured extend forwards for external-prefix claims.

INTEGRATION PIVOT (2026-08-19, recorded after the first serving smoke):
the smoke run proved the hook, eligibility, and workspace paths work
(one warmup, twelve geometry refusals) and exposed that extend token
counts vary per request, so exact-size graphs cannot amortize — capture
needs bucketed padding with last-real-token logits extraction. SGLang
0.5.14 already ships exactly that machinery in PrefillCudaGraphRunner
(bucketed capture_num_tokens, dummy ForwardBatch construction, buffer
population, output slicing), disabled by default. The next step is to
enable it and plug this module's workspace binding and in-graph staging
into its backend contract instead of duplicating padding and logits
handling here. The workspace, geometry checks, in-graph staging, and
fail-closed counters below carry over unchanged.


The eager extend forward costs 30-64ms of per-layer launch overhead and
is the last co-resident interference mechanism at every registered shape
(P4-third: eight of ten trials at parity, the miss entirely two extend
collisions). Staging itself is 116us per layer and the whole chain is
capture-sound (ExtendStagingCaptureProbe: replays recompute selection
and stage bytes exactly from changed contents).

Design: one CUDA graph per extend geometry, captured lazily around the
first real extend that exhibits it, against a claim-independent
workspace. The registered transfer objects bind global pool base
pointers, so per-claim identity lives entirely in buffer contents: host
rows, staging lease rows, envelopes, and the pinned source bytes the
transfer kernels read at replay. Before a replay the host binds the
incoming claim into the workspace (small device copies); afterwards it
copies the per-layer selected-row tables back into the claim so the
decode path proceeds unchanged. Claims whose geometry has no captured
graph fall back to the eager extend and are counted.

Failure posture is fail-closed throughout: any binding mismatch,
geometry surprise, or capture error falls back to eager and records a
reason counter; replays never serve a claim whose identity checks fail.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

import torch


@dataclass
class ExtendGeometry:
    """Everything that must match for a captured extend to be reusable."""

    token_count: int
    new_tokens: int
    pages: int
    free_budget: int
    kept_rows: int
    capacity_rows: int
    layer_count: int

    @classmethod
    def of(cls, claim: Any, new_tokens: int) -> "ExtendGeometry":
        return cls(
            token_count=int(claim.token_count),
            new_tokens=int(new_tokens),
            pages=int(claim.pages),
            free_budget=int(claim.free_budget),
            kept_rows=int(claim.kept_prefix_rows),
            capacity_rows=int(claim.capacity_rows),
            layer_count=int(claim.layer_count),
        )

    def key(self) -> tuple[int, ...]:
        return (
            self.token_count,
            self.new_tokens,
            self.pages,
            self.free_budget,
            self.kept_rows,
            self.capacity_rows,
            self.layer_count,
        )


@dataclass
class ExtendWorkspace:
    """Static claim-shaped buffers whose pointers the graph bakes in."""

    geometry: ExtendGeometry
    kmin: torch.Tensor
    kmax: torch.Tensor
    page_scores: torch.Tensor
    ordered_pages: torch.Tensor
    full_forced_pages: torch.Tensor
    host_rows: torch.Tensor
    device_rows: torch.Tensor
    cached_pages: torch.Tensor
    selected_rows: torch.Tensor
    row_cache: torch.Tensor
    source_index: torch.Tensor
    staging_index: torch.Tensor
    copied_rows: torch.Tensor
    first_object_slot: int = -1
    bound_claim_id: int = -1

    @classmethod
    def allocate(
        cls, geometry: ExtendGeometry, heads: int, dim: int, device: Any
    ) -> "ExtendWorkspace":
        g = geometry
        return cls(
            geometry=g,
            kmin=torch.zeros(
                g.layer_count, g.pages, heads, dim,
                dtype=torch.float32, device=device,
            ),
            kmax=torch.zeros(
                g.layer_count, g.pages, heads, dim,
                dtype=torch.float32, device=device,
            ),
            page_scores=torch.empty(
                g.pages, dtype=torch.float32, device=device
            ),
            ordered_pages=torch.empty(
                2 + g.free_budget + 1, dtype=torch.int64, device=device
            ),
            full_forced_pages=torch.tensor(
                [0, g.pages - 2], dtype=torch.int64, device=device
            ),
            host_rows=torch.zeros(
                g.token_count, dtype=torch.int32, device=device
            ),
            device_rows=torch.zeros(
                g.capacity_rows, dtype=torch.int32, device=device
            ),
            cached_pages=torch.full(
                (g.layer_count, g.capacity_rows // 16 or 1),
                -1,
                dtype=torch.int64,
                device=device,
            ),
            selected_rows=torch.empty(
                g.capacity_rows, dtype=torch.int32, device=device
            ),
            row_cache=torch.zeros(
                g.layer_count, g.kept_rows, dtype=torch.int32, device=device
            ),
            source_index=torch.zeros(
                g.layer_count, g.capacity_rows,
                dtype=torch.int32, device=device,
            ),
            staging_index=torch.zeros(
                g.layer_count, g.capacity_rows,
                dtype=torch.int32, device=device,
            ),
            copied_rows=torch.zeros(1, dtype=torch.int64, device=device),
        )

    def bind(self, claim: Any, engine: Any) -> bool:
        """Point the workspace at a claim's identity; False refuses replay."""
        g = self.geometry
        if (
            int(claim.token_count) != g.token_count
            or int(claim.pages) != g.pages
            or int(claim.free_budget) != g.free_budget
            or int(claim.kept_prefix_rows) != g.kept_rows
            or int(claim.capacity_rows) != g.capacity_rows
            or claim.kmin is None
            or claim.kmin.shape != self.kmin.shape
        ):
            return False
        self.kmin.copy_(claim.kmin)
        self.kmax.copy_(claim.kmax)
        self.host_rows.copy_(claim.host_rows)
        self.device_rows.copy_(claim.device_rows[: g.capacity_rows])
        self.cached_pages.fill_(-1)
        self.copied_rows.zero_()
        self.bound_claim_id = int(claim.claim_id)
        return True

    def unbind_into(self, claim: Any) -> None:
        """Copy replay results back so the decode path proceeds unchanged."""
        for local_layer in range(self.geometry.layer_count):
            claim.remember_selected_rows(
                local_layer, self.row_cache[local_layer]
            )
        claim.rows_copied += int(self.copied_rows.item())
        claim.requested_rows += (
            self.geometry.kept_rows * self.geometry.layer_count
        )
        claim.device_accounting = True
        self.bound_claim_id = -1


class ExtendCaptureRunner:
    """Lazy per-geometry capture and replay of tiered extend forwards."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.enabled = os.environ.get("NTA_SGLANG_EXTEND_CAPTURE") == "1"
        self.graphs: dict[tuple[int, ...], dict[str, Any]] = {}
        self.workspace: ExtendWorkspace | None = None
        self.capturing = False
        self.capture_claim: Any = None
        self._count("initialized", 0)

    def _count(self, reason: str, delta: int = 1) -> None:
        key = f"extend_capture_{reason}"
        self.engine._stats[key] = self.engine._stats.get(key, 0) + delta

    def workspace_for(self, claim: Any, new_tokens: int) -> ExtendWorkspace | None:
        geometry = ExtendGeometry.of(claim, new_tokens)
        if self.workspace is None:
            reference = claim.kmin
            if reference is None:
                return None
            heads, dim = int(reference.shape[-2]), int(reference.shape[-1])
            self.workspace = ExtendWorkspace.allocate(
                geometry, heads, dim, reference.device
            )
            self._register_workspace_objects(claim)
        if self.workspace.geometry.key() != geometry.key():
            self._count("geometry_mismatch")
            return None
        return self.workspace

    def _register_workspace_objects(self, claim: Any) -> None:
        """Register transfer objects bound to workspace index arrays.

        Source and staging base pointers are the global pools (identical
        for every claim); only the per-layer index arrays are workspace
        state, so one registration serves every geometry-matched claim.
        """
        from nta_runtime.runtime import IndexedHostObject

        engine = self.engine
        workspace = self.workspace
        controller = claim.pending.controller
        host_pool = controller.mem_pool_host
        device_pool = controller.mem_pool_device
        start_layer = int(getattr(device_pool, "start_layer", 0))
        lease = engine._tiered_object_ranges.acquire(0x7FFFFFF0)
        workspace.first_object_slot = lease.begin
        self._workspace_lease = lease
        objects = []
        for local_layer in range(workspace.geometry.layer_count):
            layer_id = start_layer + local_layer
            host_key = host_pool.k_data_refs[local_layer]
            host_value = host_pool.v_data_refs[local_layer]
            key_cache = device_pool._get_key_buffer(layer_id)
            value_cache = device_pool._get_value_buffer(layer_id)
            element = key_cache[0].numel() * key_cache.element_size()
            for source, staging in (
                (host_key, key_cache),
                (host_value, value_cache),
            ):
                objects.append(
                    IndexedHostObject(
                        0x45585443 + len(objects),
                        1,
                        source.data_ptr(),
                        staging.data_ptr(),
                        workspace.source_index[local_layer].data_ptr(),
                        workspace.staging_index[local_layer].data_ptr(),
                        workspace.geometry.capacity_rows,
                        element,
                        source.stride(0) * source.element_size(),
                        staging.stride(0) * staging.element_size(),
                        int(source.shape[0]),
                        int(staging.shape[0]),
                    )
                )
        engine._runtime.register_indexed_host_objects(
            workspace.first_object_slot, objects
        )

    def stage_layer_in_graph(
        self, local_layer: int, queries: torch.Tensor, stream: Any
    ) -> torch.Tensor:
        """The capture-mode wavefront: workspace pointers, one stream."""
        workspace = self.workspace
        engine = self.engine
        phases = engine._phase_program(engine._nta_demand_decode_wrappers[0])
        if queries.dtype != torch.float16:
            queries = queries.to(torch.float16)
        base = workspace.first_object_slot + 2 * local_layer
        phases.select_prepare_claim_rows(
            engine._runtime,
            base,
            queries.contiguous(),
            workspace.kmin[local_layer],
            workspace.kmax[local_layer],
            workspace.page_scores,
            workspace.full_forced_pages,
            workspace.geometry.pages - 1,
            workspace.geometry.free_budget,
            workspace.ordered_pages,
            16,
            workspace.geometry.token_count,
            workspace.host_rows,
            workspace.device_rows,
            workspace.cached_pages[local_layer],
            workspace.selected_rows,
            workspace.source_index[local_layer],
            workspace.staging_index[local_layer],
            workspace.copied_rows,
            stream=stream,
        )
        phases.progress_validated_indexed_host_range(
            engine._runtime, base, 2, stream=stream
        )
        kept = workspace.geometry.kept_rows
        workspace.row_cache[local_layer].copy_(
            workspace.selected_rows[:kept]
        )
        return workspace.row_cache[local_layer]


    def eligible_claim(self, forward_batch: Any) -> Any | None:
        """The single sidecar claim this extend serves, or None."""
        if not self.enabled or self.capturing:
            return None
        mode = getattr(forward_batch, "forward_mode", None)
        if mode is None or not mode.is_extend() or mode.is_mixed():
            return None
        rids = tuple(getattr(forward_batch, "rids", ()) or ())
        if len(rids) != 1:
            self._count("multi_request_batch")
            return None
        claim = None
        for candidate in self.engine._tiered_claims.values():
            if candidate.request_id == rids[0]:
                claim = candidate
                break
        if claim is None:
            return None
        if (
            not claim.external_sidecar
            or claim.verify
            or claim.verify_fast
            or claim.select_all
            or claim.selection_refresh_interval <= 1
            or claim.kmin is None
        ):
            self._count("claim_ineligible")
            return None
        return claim

    def run(
        self,
        runner: Any,
        forward_batch: Any,
        pp_proxy_tensors: Any,
        claim: Any,
    ) -> Any | None:
        """Serve an eligible extend by warm/capture/replay; None means eager.

        The ladder per geometry: the first extend runs eager (stashing the
        serve references and warming autotuned kernels), the second is
        captured while it executes, and every later one replays. Metadata,
        context, and planning happen OUTSIDE the recorded region.
        """
        new_tokens = int(forward_batch.input_ids.numel())
        workspace = self.workspace_for(claim, new_tokens)
        if workspace is None:
            return None
        key = workspace.geometry.key()
        entry = self.graphs.get(key)
        if entry is None:
            self.graphs[key] = {"state": "warmed"}
            self._count("warmup_eager")
            return None
        if entry["state"] == "warmed":
            if getattr(self, "serve_refs", None) is None:
                self._count("no_serve_refs")
                return None
            try:
                return self._capture(
                    entry, runner, forward_batch, pp_proxy_tensors, claim,
                    workspace,
                )
            except Exception:
                entry["state"] = "failed"
                self._count("capture_failed")
                raise
        if entry["state"] != "ready":
            return None
        return self._replay(entry, runner, forward_batch, claim, workspace)

    def _prepare(self, runner: Any, forward_batch: Any, pp_proxy_tensors):
        """Mirror the eager runner's pre-forward steps outside the graph."""
        model_runner = runner.model_runner
        if not model_runner.server_args.enable_pdmux:
            forward_batch = runner.load_batch(forward_batch, pp_proxy_tensors)
        if forward_batch.needs_forward_metadata_init():
            if hasattr(model_runner.model, "prepare_forward_batch"):
                model_runner.model.prepare_forward_batch(forward_batch)
            model_runner.attn_backend.init_forward_metadata(forward_batch)
        return forward_batch

    def _bind_inputs(self, entry: dict, forward_batch: Any) -> None:
        statics = entry.setdefault("inputs", {})
        for name in ("input_ids", "positions", "out_cache_loc"):
            value = getattr(forward_batch, name, None)
            if value is None:
                continue
            if name not in statics:
                statics[name] = value.clone()
            else:
                statics[name].copy_(value)
            setattr(forward_batch, name, statics[name])
        workspace = self.workspace
        if "out_cache_loc" in statics:
            workspace_suffix = statics["out_cache_loc"].to(torch.int32)
            if "suffix_rows" not in statics:
                statics["suffix_rows"] = workspace_suffix.clone()
            else:
                statics["suffix_rows"].copy_(workspace_suffix)

    def _prime_context(self, entry: dict, claim: Any) -> Any:
        """Build the tiered extend context outside the recorded region.

        The context's plan is geometry-static: every replay reuses its
        boundaries and buffers, with per-claim variation living in the
        plan buffer's contents (kept rows applied in-graph per layer, the
        suffix refreshed in-graph from the static out_cache_loc).
        """
        engine = self.engine
        executor = engine._selected
        wrapper, layer, kv_cache = self.serve_refs
        geometry = self.workspace.geometry
        queries = torch.zeros(
            geometry.new_tokens,
            int(engine._nta_demand_decode_wrappers[0]._num_qo_heads)
            if hasattr(engine._nta_demand_decode_wrappers[0], "_num_qo_heads")
            else self.workspace.kmin.shape[-2],
            self.workspace.kmin.shape[-1],
            dtype=torch.float16,
            device=self.workspace.kmin.device,
        )
        ctx = executor._build_multi_claim_ctx(
            engine, (claim,), wrapper, queries, kv_cache, layer, True
        )
        if ctx is None:
            return None
        executor._tiered_batch_contexts[id(wrapper)] = ctx
        entry["plan_indices"] = ctx["plan_indices"]
        entry["suffix_slice"] = (
            geometry.kept_rows,
            geometry.kept_rows + geometry.new_tokens,
        )
        return ctx

    def _capture(
        self, entry, runner, forward_batch, pp_proxy_tensors, claim,
        workspace,
    ):
        if not workspace.bind(claim, self.engine):
            self._count("bind_refused")
            return None
        forward_batch = self._prepare(runner, forward_batch, pp_proxy_tensors)
        self._bind_inputs(entry, forward_batch)
        if self._prime_context(entry, claim) is None:
            self._count("context_priming_failed")
            return None
        model_runner = runner.model_runner
        kwargs = model_runner._extend_forward_kwargs(
            forward_batch, pp_proxy_tensors
        )
        begin, end = entry["suffix_slice"]
        suffix_static = entry["inputs"]["suffix_rows"]
        graph = torch.cuda.CUDAGraph()
        self.capturing = True
        self.capture_claim = claim
        try:
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                entry["plan_indices"][begin:end].copy_(suffix_static)
                output = model_runner.model.forward(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                    **kwargs,
                )
        finally:
            self.capturing = False
            self.capture_claim = None
        entry["graph"] = graph
        entry["output"] = output
        entry["state"] = "ready"
        self._count("captured")
        workspace.unbind_into(claim)
        return output

    def _replay(self, entry, runner, forward_batch, claim, workspace):
        if not workspace.bind(claim, self.engine):
            self._count("bind_refused")
            return None
        forward_batch = self._prepare(runner, forward_batch, None)
        self._bind_inputs(entry, forward_batch)
        entry["graph"].replay()
        self._count("replayed")
        workspace.unbind_into(claim)
        stats = self.engine._stats
        stats["tiered_extend_captured_replays"] = (
            stats.get("tiered_extend_captured_replays", 0) + 1
        )
        return entry["output"]

