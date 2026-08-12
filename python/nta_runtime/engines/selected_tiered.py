"""Tiered selected-serving diagnostic integrated with SGLang HiCache.

A claimed host prefix is never bulk-promoted. At claim time the engine builds
per-layer key envelopes either by streaming each host-key layer through one
reused CUDA scratch tensor or by reducing mapped pinned-host rows directly.
It registers per-layer K/V indexed objects whose index arrays are engine-owned
device tensors. External sidecars have no HiCache producer event; legacy
claimed loads complete their owned producer events. At every attention layer
-- extend and decode -- the
selector scores the prefix's envelopes against the live queries, retention
keeps sinks and the recent window, and only the chosen pages' rows move
host-to-device through the validated indexed path, bounded by the new
per-step row-count seam, into the sidecar's bounded physical staging rows.
Attention then runs over the chosen prefix rows plus the
request's resident tail; peer requests in a mixed decode batch keep their
full tables.

Decode serving takes the fixed-shape fast path: retention always forces the
sink and recent pages, so the kept row count is layer-invariant and the
attention wrapper is planned once per forward — each layer only rewrites
the planned indices buffer in place from a device-side selection whose
shapes never depend on data. A GPU kernel validates selected identities,
compacts only misses, and updates indexed-object row counts without a host
identity or count round trip. Verify mode runs the reference path instead —
host selection, dual
independently planned wrappers, byte-verification of staged rows, and a
cross-check that the device selection equals the reference selection.

The external-prefix hook runs before dense load-back allocation and reports
live/high-water dense-equivalent and staging rows. Cross-request radix sharing
of one claim is unsupported.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

import torch

from nta_runtime.quest_selector import budgeted_page_selection, quest_page_scores
from nta_runtime.runtime import IndexedHostObject

_TIERED_OBJECT_ID_BASE = 0x5449_000000000000


class TieredClaim:
    """Per-claim state: envelopes, registrations, mappings, accounting."""

    def __init__(
        self,
        pending: Any,
        engine: Any,
        *,
        budget_pages: int,
        page_tokens: int,
        first_object_slot: int,
        claim_id: int,
        verify: bool,
        verify_fast: bool = False,
        table_views: dict | None = None,
    ) -> None:
        self.pending = pending
        self.budget_pages = budget_pages
        self.page_tokens = page_tokens
        self.first_object_slot = first_object_slot
        self.claim_id = claim_id
        self.verify = verify
        self.verify_fast = verify_fast
        if not 0 < claim_id <= 0xFFFFFFFF:
            raise RuntimeError("tiered claim ID exceeds the object namespace")
        controller = pending.controller
        self.external_sidecar = bool(
            getattr(pending, "external_sidecar", False)
        )
        self.layer_count = int(controller.layer_num)
        host_indices, device_indices = controller.move_indices(
            pending.host_indices, pending.device_indices
        )
        self.token_count = int(host_indices.numel())
        if self.token_count <= 0:
            raise RuntimeError("tiered claim has no prefix tokens")
        self.pages = (
            self.token_count + page_tokens - 1
        ) // page_tokens
        device = torch.device("cuda")
        self.host_rows = host_indices.to(device, torch.int32)
        self.virtual_rows = device_indices.to(device, torch.int64)
        self.device_rows = (
            pending.staging_rows.to(device, torch.int32)
            if self.external_sidecar
            else device_indices.to(device, torch.int32)
        )
        self.host_rows_cpu = host_indices.to("cpu", torch.int64)

        host_pool = controller.mem_pool_host
        device_pool = controller.mem_pool_device
        start_layer = int(getattr(device_pool, "start_layer", 0))
        self.start_layer = start_layer

        # Key envelopes are immutable for an external prefix. Reuse a prior
        # probe's summaries when the exact host mapping and layer geometry
        # repeat; otherwise build them without promoting KV into dense slots.
        self.contiguous_host = self.token_count == 1 or bool(
            (self.host_rows_cpu.diff() == 1).all()
        )
        summary_path = os.environ.get("NTA_SGLANG_SUMMARY_PATH", "copy")
        if summary_path not in ("copy", "mapped"):
            raise RuntimeError(
                "NTA_SGLANG_SUMMARY_PATH must be 'copy' or 'mapped'"
            )
        self.summary_path = summary_path
        self.summary_scratch_bytes = 0
        reference_key = host_pool.k_data_refs[0]
        summary_bytes = (
            self.layer_count
            * self.token_count
            * reference_key[0].numel()
            * reference_key.element_size()
        )
        cache_key = self._summary_cache_key(
            pending, host_pool, start_layer, summary_path
        )
        cache = getattr(engine, "_tiered_summary_cache", None)
        cache_entry = None if (verify or verify_fast or cache is None) else cache.get(
            cache_key
        )
        summary_stream = (
            None
            if (
                verify
                or verify_fast
                or os.environ.get("NTA_SGLANG_SUMMARY_ASYNC", "0") != "1"
            )
            else getattr(engine, "_summary_stream", None)
        )
        self.summaries_ready: Any = None
        if cache_entry is not None:
            self.kmin, self.kmax, ready, cached_bytes = cache_entry
            if summary_stream is not None:
                # Defer the readiness wait to first selection: waiting
                # here would stall the compute stream behind a scan that
                # may still be in flight on the summary stream.
                self.summaries_ready = ready
            else:
                torch.cuda.current_stream(device).wait_event(ready)
            engine._stats["tiered_summary_cache_hits"] = (
                engine._stats.get("tiered_summary_cache_hits", 0) + 1
            )
            engine._stats["tiered_summary_source_bytes_avoided"] = (
                engine._stats.get("tiered_summary_source_bytes_avoided", 0)
                + summary_bytes
            )
            engine._stats["tiered_summary_cache_hit_bytes"] = (
                engine._stats.get("tiered_summary_cache_hit_bytes", 0)
                + int(cached_bytes)
            )
        else:
            # The envelope scan is ~13.5GB of pinned-host reads for a 16K
            # prefix; on the compute stream it queues half a second of
            # work ahead of every live decode. Non-verify claims scan on
            # the engine's summary stream and publish a readiness event;
            # admission holds the claim's batch until it fires and the
            # first selection waits on it as the fail-safe.
            scan_context = (
                torch.cuda.stream(summary_stream)
                if summary_stream is not None
                else contextlib.nullcontext()
            )
            with scan_context:
                self.kmin, self.kmax = self._build_envelopes(
                    engine, host_pool, device, verify, verify_fast
                )
            counter = f"tiered_summary_{summary_path}_claims"
            engine._stats[counter] = engine._stats.get(counter, 0) + 1
            engine._stats["tiered_summary_source_bytes"] = (
                engine._stats.get("tiered_summary_source_bytes", 0) + summary_bytes
            )
            engine._stats["tiered_summary_scratch_high_water_bytes"] = max(
                engine._stats.get("tiered_summary_scratch_high_water_bytes", 0),
                self.summary_scratch_bytes,
            )
            if summary_stream is not None:
                done = torch.cuda.Event()
                done.record(summary_stream)
                self.summaries_ready = done
            if cache is not None and not (verify or verify_fast):
                ready = self.summaries_ready or torch.cuda.Event()
                if self.summaries_ready is None:
                    ready.record(torch.cuda.current_stream(device))
                entry_bytes = (
                    self.kmin.numel() * self.kmin.element_size()
                    + self.kmax.numel() * self.kmax.element_size()
                )
                cache[cache_key] = (self.kmin, self.kmax, ready, entry_bytes)
                order = getattr(engine, "_tiered_summary_cache_order", None)
                if order is not None:
                    order.append(cache_key)
                    capacity = int(
                        getattr(engine, "_tiered_summary_cache_capacity", 0)
                    )
                    while capacity >= 0 and len(order) > capacity:
                        evicted = order.pop(0)
                        if cache.pop(evicted, None) is not None:
                            engine._stats["tiered_summary_cache_evictions"] = (
                                engine._stats.get(
                                    "tiered_summary_cache_evictions", 0
                                )
                                + 1
                            )
                engine._stats["tiered_summary_cache_misses"] = (
                    engine._stats.get("tiered_summary_cache_misses", 0) + 1
                )
                engine._stats["tiered_summary_cache_bytes"] = sum(
                    int(entry[3]) for entry in cache.values()
                )

        # Shared per-step index arrays: every layer's objects point at the
        # same device tensors; layers rewrite the prefix sequentially on the
        # compute stream, so the copy consumes each layer's own contents.
        # Capacity is page-aligned: the device cache and compaction kernel
        # require whole slots, and the bridge leases staging identically, so
        # a sub-page claim still owns one full page of rows.
        self.capacity_rows = (
            min(budget_pages, self.pages) * page_tokens
            if self.external_sidecar
            else min(budget_pages * page_tokens, self.token_count)
        )
        self.table_backed = table_views is not None
        if table_views is not None:
            # Claim-table rows: pointer-stable slices of one allocation
            # per field. Transfer index words are per (claim, layer) so a
            # pipelined extend's in-flight transfers never share a row
            # with a later layer's prep.
            self.source_index = table_views["source_indices"][
                :, : self.capacity_rows
            ]
            self.staging_index = table_views["staging_indices"][
                :, : self.capacity_rows
            ]
        else:
            self.source_index = torch.zeros(
                (self.layer_count, self.capacity_rows),
                dtype=torch.int32, device=device,
            )
            self.staging_index = torch.zeros(
                (self.layer_count, self.capacity_rows),
                dtype=torch.int32, device=device,
            )
        self.cached_pages: torch.Tensor | None = None
        self.selected_rows: torch.Tensor | None = None
        self.copy_stream: torch.cuda.Stream | None = None
        self.selection_ready: tuple[torch.cuda.Event, ...] = ()
        self.copy_ready: tuple[torch.cuda.Event, ...] = ()
        if self.external_sidecar:
            cache_slots = self.capacity_rows // page_tokens
            if cache_slots > 0:
                if table_views is not None:
                    self.cached_pages = table_views["cached_pages"][
                        : self.layer_count, :cache_slots
                    ]
                    self.cached_pages.fill_(-1)
                    self.selected_rows = table_views["selected_rows"][
                        : self.capacity_rows
                    ]
                else:
                    self.cached_pages = torch.full(
                        (self.layer_count, cache_slots),
                        -1,
                        dtype=torch.int64,
                        device=device,
                    )
                    self.selected_rows = torch.empty(
                        self.capacity_rows, dtype=torch.int32, device=device
                    )
                self.copy_stream = torch.cuda.Stream(device=device)
                self.selection_ready = tuple(
                    torch.cuda.Event() for _ in range(self.layer_count)
                )
                self.copy_ready = tuple(
                    torch.cuda.Event() for _ in range(self.layer_count)
                )

        stream = torch.cuda.current_stream()
        objects = []
        for local_layer in range(self.layer_count):
            layer_id = start_layer + local_layer
            host_key = host_pool.k_data_refs[local_layer]
            host_value = host_pool.v_data_refs[local_layer]
            key_cache = device_pool._get_key_buffer(layer_id)
            value_cache = device_pool._get_value_buffer(layer_id)
            element = key_cache[0].numel() * key_cache.element_size()
            # Every source and destination must agree on the per-row byte
            # count the indexed copy uses; checking only one pairing lets an
            # asymmetric host layout stage silently corrupted KV.
            for label, tensor in (
                ("host key", host_key),
                ("host value", host_value),
                ("device value", value_cache),
            ):
                if tensor[0].numel() * tensor.element_size() != element:
                    raise RuntimeError(
                        f"tiered K/V row geometry disagrees: {label} rows "
                        f"are {tensor[0].numel() * tensor.element_size()} "
                        f"bytes, device key rows are {element}"
                    )
            for source, staging in (
                (host_key, key_cache),
                (host_value, value_cache),
            ):
                objects.append(
                    IndexedHostObject(
                        _TIERED_OBJECT_ID_BASE
                        + (claim_id << 16)
                        + len(objects),
                        1,
                        source.data_ptr(),
                        staging.data_ptr(),
                        self.source_index[local_layer].data_ptr(),
                        self.staging_index[local_layer].data_ptr(),
                        self.capacity_rows,
                        element,
                        source.stride(0) * source.element_size(),
                        staging.stride(0) * staging.element_size(),
                        int(source.shape[0]),
                        int(staging.shape[0]),
                    )
                )
        engine._runtime.register_indexed_host_objects(
            first_object_slot, objects, stream=stream
        )
        self.staged = (
            None
            if self.external_sidecar
            else torch.zeros(
                (self.layer_count, self.pages),
                dtype=torch.int32,
                device=device,
            )
        )
        self.copied_rows_device = (
            table_views["copied_rows"]
            if table_views is not None
            else torch.zeros(1, dtype=torch.int64, device=device)
        )
        self._copied_rows_host = torch.empty(1, dtype=torch.int64, pin_memory=True)
        self._accounting_event = torch.cuda.Event()
        self.device_accounting = False
        self.requested_rows = 0
        # Order-independent request matching: the claim's mapping order is
        # the radix/controller order, not sequence order, so requests are
        # recognized by slot membership and every table token maps back to
        # its claim position through a pool-sized lookup.
        self.virtual_begin = (
            int(self.virtual_rows[0]) if self.external_sidecar else None
        )
        if self.external_sidecar:
            self.slot_is_prefix = None
            self.slot_to_position = None
        else:
            pool_rows = int(
                device_pool._get_key_buffer(start_layer).shape[0]
            )
            self.slot_is_prefix = torch.zeros(
                pool_rows, dtype=torch.bool, device=device
            )
            self.slot_to_position = torch.zeros(
                pool_rows, dtype=torch.int64, device=device
            )
            rows_long = self.device_rows.to(torch.long)
            self.slot_is_prefix[rows_long] = True
            self.slot_to_position[rows_long] = torch.arange(
                self.token_count, dtype=torch.int64, device=device
            )
        self.rows_copied = 0
        self.rows_rehit = 0
        self.layers_served = 0
        self.request_id: str | None = getattr(pending, "request_id", None)
        self.request_generation: int | None = None
        self.bound_prefix_mask: torch.Tensor | None = None

        # Fixed-shape decode selection: with sink and recent retention the
        # forced set is {0, pages-2, pages-1}, the tail page is always kept,
        # and the free budget is constant — so every per-layer tensor has a
        # data-independent shape and the kept row count never changes.
        self.free_budget = budget_pages - 3
        self.select_all = not (
            self.pages > budget_pages and self.free_budget > 0
        )
        # A claim smaller than the budget needs no selection: keeping every
        # page is itself a fixed shape (empty free set, kept rows equal to
        # the token count), so small re-promotion claims under churn serve
        # through the same concurrent path instead of refusing the batch.
        self.fast_ok = True
        self.page_arange = torch.arange(
            page_tokens, dtype=torch.int64, device=device
        )
        self.tail_positions = torch.arange(
            (self.pages - 1) * page_tokens, self.token_count,
            dtype=torch.int64, device=device,
        )
        if self.select_all:
            self.free_budget = 0
            self.forced_pages = torch.arange(
                self.pages, dtype=torch.int64, device=device
            )
            self.full_forced_pages = self.forced_pages[: self.pages - 1]
            self.kept_prefix_rows = self.token_count
        else:
            self.forced_pages = torch.tensor(
                [0, self.pages - 2, self.pages - 1],
                dtype=torch.int64, device=device,
            )
            self.full_forced_pages = self.forced_pages[:2]
            self.kept_prefix_rows = (
                (self.free_budget + 2) * page_tokens
                + int(self.tail_positions.numel())
            )
        self.ctx: dict[str, Any] | None = None
        self.selection_refresh_interval = int(
            os.environ.get("NTA_SGLANG_SELECTED_REFRESH_INTERVAL", "1") or 1
        )
        if self.selection_refresh_interval <= 0:
            raise RuntimeError("NTA_SGLANG_SELECTED_REFRESH_INTERVAL must be positive")
        self._selected_row_cache: list[torch.Tensor | None] = [
            None for _ in range(self.layer_count)
        ]
        # Refresh bookkeeping in decode steps, not per-layer visits: all
        # layers advance together each step, so one step counter and a
        # per-layer last-staged step express the same cadence with O(1)
        # work per claim per step. ``_earliest_stage_step`` lets the serve
        # loop rule out a whole step's refreshes with one comparison.
        self._decode_step = 0
        self._layer_stage_step: list[int | None] = [
            None for _ in range(self.layer_count)
        ]
        self._earliest_stage_step: int | None = None
        self.reuse_cache_complete = False

    def _summary_cache_key(
        self, pending: Any, host_pool: Any, start_layer: int, summary_path: str
    ) -> tuple[Any, ...]:
        if self.contiguous_host:
            host_signature: tuple[Any, ...] = (
                "contiguous",
                int(self.host_rows_cpu[0]),
                int(self.host_rows_cpu[-1]) + 1,
            )
        else:
            host_signature = (
                "indexed",
                tuple(int(row) for row in self.host_rows_cpu.tolist()),
            )
        layer_geometry = tuple(
            (
                int(layer.data_ptr()),
                tuple(int(value) for value in layer.shape),
                tuple(int(value) for value in layer.stride()),
                str(layer.dtype),
            )
            for layer in host_pool.k_data_refs[: self.layer_count]
        )
        return (
            "summary-v1",
            tuple(int(value) for value in getattr(pending, "node_ids", ())),
            int(start_layer),
            int(self.layer_count),
            int(self.token_count),
            int(self.page_tokens),
            summary_path,
            host_signature,
            layer_geometry,
        )

    def _build_envelopes(
        self,
        engine: Any,
        host_pool: Any,
        device: torch.device,
        verify: bool,
        verify_fast: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.contiguous_host and not (verify or verify_fast):
            if self.summary_path == "mapped":
                return self._mapped_envelopes(engine, host_pool, device)
            return self._streamed_envelopes(host_pool, device)
        kmin, kmax = self._gathered_envelopes(host_pool, device)
        if self.contiguous_host:
            if self.summary_path == "mapped":
                streamed_kmin, streamed_kmax = self._mapped_envelopes(
                    engine, host_pool, device
                )
            else:
                streamed_kmin, streamed_kmax = self._streamed_envelopes(
                    host_pool, device
                )
            if not (
                torch.equal(streamed_kmin, kmin)
                and torch.equal(streamed_kmax, kmax)
            ):
                raise RuntimeError(
                    "streamed envelope scan diverged from the CPU "
                    "reference; min/max are order-invariant, so this "
                    "is a transfer or indexing defect"
                )
        return kmin, kmax

    def _page_reduce(
        self, rows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Reduce in the storage dtype and cast the result: min/max select
        # existing values, so fp16 reduction followed by a lossless cast is
        # bit-identical to reducing in fp32 at half the traffic.
        heads, dim = rows.shape[-2], rows.shape[-1]
        token_count = int(rows.shape[0])
        full = (token_count // self.page_tokens) * self.page_tokens
        page_view = rows[:full].view(-1, self.page_tokens, heads, dim)
        kmin = page_view.amin(dim=1)
        kmax = page_view.amax(dim=1)
        if full < token_count:
            tail = rows[full:]
            kmin = torch.cat([kmin, tail.amin(dim=0, keepdim=True)])
            kmax = torch.cat([kmax, tail.amax(dim=0, keepdim=True)])
        return kmin.to(torch.float32), kmax.to(torch.float32)

    def _gathered_envelopes(
        self, host_pool: Any, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kmins, kmaxs = [], []
        for local_layer in range(self.layer_count):
            rows = host_pool.k_data_refs[local_layer][self.host_rows_cpu]
            kmin, kmax = self._page_reduce(rows)
            kmins.append(kmin)
            kmaxs.append(kmax)
        return torch.stack(kmins).to(device), torch.stack(kmaxs).to(device)

    def _streamed_envelopes(
        self, host_pool: Any, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        begin = int(self.host_rows_cpu[0])
        reference = host_pool.k_data_refs[0]
        configured_rows = int(
            os.environ.get("NTA_SGLANG_SUMMARY_CHUNK_ROWS", "4096") or 0
        )
        if configured_rows <= 0:
            raise RuntimeError("NTA_SGLANG_SUMMARY_CHUNK_ROWS must be positive")
        chunk_rows = max(
            self.page_tokens,
            configured_rows // self.page_tokens * self.page_tokens,
        )
        scratch_rows = min(self.token_count, chunk_rows)
        scratch = torch.empty(
            (scratch_rows, *reference.shape[1:]),
            dtype=reference.dtype, device=device,
        )
        self.summary_scratch_bytes = scratch.numel() * scratch.element_size()
        kmins, kmaxs = [], []
        for local_layer in range(self.layer_count):
            host_key = host_pool.k_data_refs[local_layer]
            if (
                host_key.shape[1:] != reference.shape[1:]
                or host_key.dtype != reference.dtype
            ):
                raise RuntimeError("host key layers disagree in geometry")
            layer_mins, layer_maxs = [], []
            for offset in range(0, self.token_count, scratch_rows):
                count = min(scratch_rows, self.token_count - offset)
                view = scratch[:count]
                view.copy_(
                    host_key[begin + offset : begin + offset + count],
                    non_blocking=True,
                )
                kmin, kmax = self._page_reduce(view)
                layer_mins.append(kmin)
                layer_maxs.append(kmax)
            kmins.append(torch.cat(layer_mins))
            kmaxs.append(torch.cat(layer_maxs))
        return torch.stack(kmins), torch.stack(kmaxs)

    def _mapped_envelopes(
        self, engine: Any, host_pool: Any, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse mapped-host key reads and page reduction without K staging."""
        begin = int(self.host_rows_cpu[0])
        reference = host_pool.k_data_refs[0]
        heads, dim = int(reference.shape[-2]), int(reference.shape[-1])
        shape = (self.layer_count, self.pages, heads, dim)
        kmin = torch.empty(shape, dtype=torch.float32, device=device)
        kmax = torch.empty_like(kmin)
        phases = engine._phase_program(engine._nta_demand_decode_wrappers[0])
        stream = torch.cuda.current_stream(device)
        for local_layer in range(self.layer_count):
            host_key = host_pool.k_data_refs[local_layer]
            if (
                host_key.shape[1:] != reference.shape[1:]
                or host_key.dtype != reference.dtype
            ):
                raise RuntimeError("host key layers disagree in geometry")
            phases.reduce_mapped_key_pages(
                host_key,
                begin,
                self.token_count,
                self.page_tokens,
                kmin[local_layer],
                kmax[local_layer],
                stream=stream,
            )
        return kmin, kmax

    def page_row_count(self, chosen: list[int]) -> int:
        tail_rows = self.token_count - (self.pages - 1) * self.page_tokens
        count = 0
        for page in chosen:
            count += tail_rows if page == self.pages - 1 else self.page_tokens
        return count

    def _positions_of(self, pages: list[int]) -> torch.Tensor:
        return torch.cat(
            [
                torch.arange(
                    page * self.page_tokens,
                    min((page + 1) * self.page_tokens, self.token_count),
                    device=self.host_rows.device,
                    dtype=torch.int64,
                )
                for page in pages
            ]
        )

    def _stage_rows(
        self, engine: Any, local_layer: int, positions: torch.Tensor,
        count: int, stream: Any,
    ) -> None:
        if count <= 0 or count > self.capacity_rows:
            raise RuntimeError(
                f"tiered staging asked for {count} rows outside capacity "
                f"{self.capacity_rows}"
            )
        self.source_index[local_layer, :count] = self.host_rows[positions]
        if self.external_sidecar:
            self.staging_index[local_layer, :count] = self.device_rows[:count]
        else:
            self.staging_index[local_layer, :count] = self.device_rows[
                positions
            ]
        phases = engine._phase_program(engine._nta_demand_decode_wrappers[0])
        base = self.first_object_slot + 2 * local_layer
        phases.set_indexed_row_counts(
            engine._runtime, base, 2, count, stream=stream
        )
        phases.progress_validated_indexed_host_range(
            engine._runtime, base, 2, stream=stream
        )
        self.rows_copied += count

    def stage_layer(
        self, engine: Any, local_layer: int, chosen: list[int],
        stream: Any,
    ) -> None:
        """Reference staging: copy the chosen pages' rows that are not
        already staged for this layer; verify the whole chosen set."""
        if self.external_sidecar:
            positions = self._positions_of(chosen)
            count = int(positions.numel())
            self._stage_rows(engine, local_layer, positions, count, stream)
            self.copied_rows_device.add_(count)
            self.device_accounting = True
            self.requested_rows += count
            if self.verify or self.verify_fast:
                self._verify_layer(local_layer, positions, count)
            return
        chosen_tensor = torch.tensor(
            chosen, dtype=torch.int64, device=self.staged.device
        )
        if self.staged is None:
            raise RuntimeError("legacy tiered staging bitmap is unavailable")
        already = self.staged[local_layer, chosen_tensor].tolist()
        new_pages = [p for p, hit in zip(chosen, already) if not hit]
        self.rows_rehit += self.page_row_count(
            [p for p, hit in zip(chosen, already) if hit]
        )
        if new_pages:
            positions = self._positions_of(new_pages)
            self._stage_rows(
                engine, local_layer, positions, int(positions.numel()), stream
            )
            self.staged[
                local_layer,
                torch.tensor(
                    new_pages, dtype=torch.int64, device=self.staged.device
                ),
            ] = True
        if self.verify or self.verify_fast:
            all_positions = self._positions_of(chosen)
            self._verify_layer(
                local_layer, all_positions, int(all_positions.numel())
            )

    def table_prefix_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return the external-prefix positions for this request's table.

        Sidecar claims recompute by identity on every forward: virtual
        token ids are never valid physical rows, so the range test is
        exact under any table rebuild or reorder — whereas replaying a
        positional snapshot after a retract/resume would mark the wrong
        positions and feed virtual ids into the planned physical indices.
        The positional snapshot is kept only for the legacy dense-slot
        mode, where recycled physical slot ids can alias between requests
        and the first observation is the authoritative one.
        """
        if getattr(self, "external_sidecar", False):
            return (tokens >= self.virtual_begin) & (
                tokens < self.virtual_begin + self.token_count
            )
        if self.bound_prefix_mask is None:
            return self.slot_is_prefix[tokens.to(torch.long)]
        if tokens.numel() < self.bound_prefix_mask.numel():
            raise RuntimeError("tiered request table shrank below its bound prefix")
        mask = torch.zeros(tokens.numel(), dtype=torch.bool, device=tokens.device)
        mask[: self.bound_prefix_mask.numel()].copy_(self.bound_prefix_mask)
        return mask

    def note_serving(
        self,
        engine: Any,
        request_position: int,
        prefix_mask: torch.Tensor | None = None,
    ) -> None:
        """Bind the claim to the lifetime identity of its first consumer.

        Identity is (request id, req_pool_idx): the pool row is stable for
        a request's whole life, unlike slot-tracker generations, which
        count reuse of *batch positions* and legitimately change whenever
        batch composition changes. A matching id on a different pool row
        is a genuinely new request epoch touching a live claim's rows —
        the cross-arrival case — and is refused.
        """
        request_ids = getattr(engine, "_current_request_ids", ())
        pool_indices = getattr(engine, "_current_req_pool_indices", ())
        if (
            request_position >= len(request_ids)
            or request_position >= len(pool_indices)
        ):
            raise RuntimeError("tiered forward omitted request identity")
        request_id = request_ids[request_position]
        pool_index = pool_indices[request_position]
        if self.request_generation is None:
            if prefix_mask is None:
                raise RuntimeError("first tiered binding omitted its prefix mask")
            if self.request_id is not None and self.request_id != request_id:
                raise RuntimeError("external prefix reached the wrong request")
            self.request_id = request_id
            self.request_generation = pool_index
            self.bound_prefix_mask = prefix_mask.clone()
            # One synchronization at bind time buys sync-free remainders on
            # every later forward: prefix positions are frozen once bound,
            # and everything the table appends after this length is
            # non-prefix by construction.
            self.bound_nonprefix_index = (
                (~self.bound_prefix_mask).nonzero().squeeze(1)
            )
            self.bound_length = int(self.bound_prefix_mask.numel())
        elif self.request_id != request_id or self.request_generation != pool_index:
            raise RuntimeError(
                "tiered claim served outside its bound request: claim "
                f"{self.claim_id} bound (request={self.request_id!r}, "
                f"pool_row={self.request_generation}) but table position "
                f"{request_position} carries (request={request_id!r}, "
                f"pool_row={pool_index}). A matching id on a different pool "
                "row means a later arrival's request is touching this live "
                "claim's rows — cross-arrival reuse of a tiered prefix is "
                "unsupported until claims can rebind"
            )

    def choose_free_pages(
        self, local_layer: int, query: torch.Tensor, group_size: int
    ) -> torch.Tensor:
        """Device-side selection with data-independent shapes.

        Select-all claims skip scoring entirely: every page is forced, the
        free set is empty, and the shape is as fixed as any budgeted one.

        Masking the forced pages to -inf and taking the first ``free_budget``
        entries of a stable descending argsort ranks the non-forced pages
        exactly as the reference selection does (stable order among unmasked
        pages is unchanged by re-keying the masked ones), so the union with
        the forced set equals ``budgeted_page_selection`` — asserted by the
        verify path. No host synchronization occurs.
        """
        if self.select_all:
            return torch.empty(
                0, dtype=torch.int64, device=self.forced_pages.device
            )
        if self.summaries_ready is not None:
            # One-shot ordering between the summary stream's scan and the
            # first live selection; a no-op when admission already held
            # the batch until the event fired.
            torch.cuda.current_stream().wait_event(self.summaries_ready)
            self.summaries_ready = None
        scores = quest_page_scores(
            query.to(torch.float32),
            self.kmin[local_layer],
            self.kmax[local_layer],
            group_size=group_size,
        ).sum(dim=0)
        scores = scores.index_fill(
            0, self.forced_pages, float("-inf")
        )
        order = torch.argsort(scores, descending=True, stable=True)
        return order[: self.free_budget]

    def kept_prefix_positions(self, free_pages: torch.Tensor) -> torch.Tensor:
        """Claim positions kept this layer: full forced + free + tail."""
        full = torch.cat([self.full_forced_pages, free_pages])
        body = (
            full.unsqueeze(1) * self.page_tokens + self.page_arange
        ).reshape(-1)
        return torch.cat([body, self.tail_positions])

    def selected_device_rows(self, positions: torch.Tensor) -> torch.Tensor:
        if getattr(self, "external_sidecar", False):
            return self.device_rows[: positions.numel()]
        return self.device_rows[positions]

    def stage_missing(
        self, engine: Any, local_layer: int, free_pages: torch.Tensor,
        stream: Any,
    ) -> torch.Tensor:
        """Acquire misses and return the selected physical table on device."""
        chosen = torch.cat([self.forced_pages, free_pages])
        if self.external_sidecar:
            rows, ready = self.stage_missing_async(
                engine, local_layer, free_pages, stream
            )
            stream.wait_event(ready)
            return rows
        if self.staged is None:
            raise RuntimeError("legacy tiered staging bitmap is unavailable")
        phases = engine._phase_program(engine._nta_demand_decode_wrappers[0])
        base = self.first_object_slot + 2 * local_layer
        phases.prepare_selected_indexed_rows(
            engine._runtime,
            base,
            2,
            chosen,
            self.page_tokens,
            self.token_count,
            self.host_rows,
            self.device_rows,
            self.staged[local_layer],
            self.source_index[local_layer],
            self.staging_index[local_layer],
            self.copied_rows_device,
            stream=stream,
        )
        phases.progress_validated_indexed_host_range(
            engine._runtime, base, 2, stream=stream
        )
        engine._stats["tiered_device_compaction_launches"] = (
            engine._stats.get("tiered_device_compaction_launches", 0) + 1
        )
        engine._stats["tiered_device_selected_pages"] = (
            engine._stats.get("tiered_device_selected_pages", 0)
            + int(chosen.numel())
        )
        self.device_accounting = True
        self.requested_rows += self.kept_prefix_rows
        positions = self.kept_prefix_positions(free_pages)
        return self.selected_device_rows(positions)

    def stage_missing_async(
        self,
        engine: Any,
        local_layer: int,
        free_pages: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> tuple[torch.Tensor, torch.cuda.Event]:
        """Launch miss-only copies on the claim transfer stream."""
        if (
            not self.external_sidecar
            or self.cached_pages is None
            or self.selected_rows is None
            or self.copy_stream is None
            or not self.selection_ready
            or not self.copy_ready
        ):
            raise RuntimeError("external claim has no bounded staging cache")
        if not 0 <= local_layer < self.layer_count:
            raise RuntimeError("external staging layer is out of range")
        if self.capacity_rows % self.page_tokens != 0:
            raise RuntimeError("external staging capacity is not page aligned")
        ordered_pages = torch.cat(
            [self.full_forced_pages, free_pages, self.forced_pages[-1:]]
        )
        phases = engine._phase_program(engine._nta_demand_decode_wrappers[0])
        base = self.first_object_slot + 2 * local_layer
        phases.prepare_bounded_selected_indexed_rows(
            engine._runtime,
            base,
            2,
            ordered_pages,
            self.page_tokens,
            self.token_count,
            self.host_rows,
            self.device_rows,
            self.cached_pages[local_layer],
            self.selected_rows,
            self.source_index[local_layer],
            self.staging_index[local_layer],
            self.copied_rows_device,
            stream=stream,
        )
        selected = self.selection_ready[local_layer]
        copied = self.copy_ready[local_layer]
        selected.record(stream)
        self.copy_stream.wait_event(selected)
        phases.progress_validated_indexed_host_range(
            engine._runtime, base, 2, stream=self.copy_stream
        )
        copied.record(self.copy_stream)
        engine._stats["tiered_device_compaction_launches"] = (
            engine._stats.get("tiered_device_compaction_launches", 0) + 1
        )
        engine._stats["tiered_bounded_cache_launches"] = (
            engine._stats.get("tiered_bounded_cache_launches", 0) + 1
        )
        engine._stats["tiered_device_selected_pages"] = (
            engine._stats.get("tiered_device_selected_pages", 0)
            + int(ordered_pages.numel())
        )
        self.device_accounting = True
        self.requested_rows += self.kept_prefix_rows
        return self.selected_rows[: self.kept_prefix_rows], copied

    def advance_decode_step(self) -> None:
        """Mark one decode step; the serve loop calls this at layer zero."""
        self._decode_step += 1

    def refresh_due(self) -> bool:
        """Whether any layer's selection must be rebuilt this step."""
        if self._earliest_stage_step is None:
            return True
        return (
            self._decode_step - self._earliest_stage_step
            >= self.selection_refresh_interval
        )

    def stage_all_layers_async(
        self,
        engine: Any,
        frees: list[torch.Tensor],
        stream: torch.cuda.Stream,
    ) -> list[torch.cuda.Event]:
        """Issue every layer's selection, staging, and transfer up front.

        A tiered extend's queries are fixed for the whole forward, so all
        selections are known at layer zero. The serialized path paid one
        synchronous copy-wait per layer (~36 round trips per extend — the
        measured ~1.5/s extend ceiling); here every prep launch lands on
        the compute stream back to back, transfers follow on the claim's
        copy stream gated per layer, and each layer's attention waits
        just-in-time. Transfer index words live in per-layer arrays so a
        later layer's prep can never overwrite indices a transfer still
        reads; the shared selected-row table is cloned into the per-layer
        cache between preps, stream-ordered.
        """
        if (
            not self.external_sidecar
            or self.cached_pages is None
            or self.selected_rows is None
            or self.copy_stream is None
            or not self.selection_ready
            or not self.copy_ready
        ):
            raise RuntimeError("external claim has no bounded staging cache")
        if len(frees) != self.layer_count:
            raise RuntimeError("pipelined staging requires one page set per layer")
        phases = engine._phase_program(engine._nta_demand_decode_wrappers[0])
        events: list[torch.cuda.Event] = []
        for local_layer, free_pages in enumerate(frees):
            ordered_pages = torch.cat(
                [self.full_forced_pages, free_pages, self.forced_pages[-1:]]
            )
            base = self.first_object_slot + 2 * local_layer
            phases.prepare_bounded_selected_indexed_rows(
                engine._runtime,
                base,
                2,
                ordered_pages,
                self.page_tokens,
                self.token_count,
                self.host_rows,
                self.device_rows,
                self.cached_pages[local_layer],
                self.selected_rows,
                self.source_index[local_layer],
                self.staging_index[local_layer],
                self.copied_rows_device,
                stream=stream,
            )
            # Clone before the next layer's prep rewrites the shared
            # selected-row table; the cache buffer is this layer's stable
            # source for the segment write.
            self.remember_selected_rows(
                local_layer, self.selected_rows[: self.kept_prefix_rows]
            )
            selected = self.selection_ready[local_layer]
            selected.record(stream)
            self.copy_stream.wait_event(selected)
            phases.progress_validated_indexed_host_range(
                engine._runtime, base, 2, stream=self.copy_stream
            )
            copied = self.copy_ready[local_layer]
            copied.record(self.copy_stream)
            events.append(copied)
        engine._stats["tiered_pipelined_extends"] = (
            engine._stats.get("tiered_pipelined_extends", 0) + 1
        )
        engine._stats["tiered_device_compaction_launches"] = (
            engine._stats.get("tiered_device_compaction_launches", 0)
            + self.layer_count
        )
        engine._stats["tiered_bounded_cache_launches"] = (
            engine._stats.get("tiered_bounded_cache_launches", 0)
            + self.layer_count
        )
        self.device_accounting = True
        self.requested_rows += self.kept_prefix_rows * self.layer_count
        return events

    def cached_selected_rows(self, local_layer: int) -> torch.Tensor | None:
        """Return a reusable selected table for this decode layer, if legal."""
        if self.selection_refresh_interval <= 1:
            return None
        if not 0 <= local_layer < self.layer_count:
            raise RuntimeError("selected-row cache layer is out of range")
        rows = self._selected_row_cache[local_layer]
        if rows is None:
            return None
        staged = self._layer_stage_step[local_layer]
        if (
            staged is None
            or self._decode_step - staged >= self.selection_refresh_interval
        ):
            return None
        return rows

    def remember_selected_rows(
        self, local_layer: int, selected_rows: torch.Tensor
    ) -> None:
        if self.selection_refresh_interval <= 1:
            return
        if not 0 <= local_layer < self.layer_count:
            raise RuntimeError("selected-row cache layer is out of range")
        buffer = self._selected_row_cache[local_layer]
        if buffer is None:
            # Allocated once per layer, then refreshed in place: a stable
            # pointer is the precondition for replaying a captured decode
            # step across refresh boundaries (phase 3 of the operator
            # build) — a re-cloned cache would strand the graph on the
            # retired allocation.
            self._selected_row_cache[local_layer] = (
                selected_rows.detach().clone()
            )
        elif (
            buffer.shape != selected_rows.shape
            or buffer.dtype != selected_rows.dtype
        ):
            raise RuntimeError(
                f"claim {self.claim_id} layer {local_layer} changed its "
                f"selected-row shape from {tuple(buffer.shape)}/"
                f"{buffer.dtype} to {tuple(selected_rows.shape)}/"
                f"{selected_rows.dtype}; fixed-shape reuse cannot follow"
            )
        else:
            buffer.copy_(selected_rows)
        self._layer_stage_step[local_layer] = self._decode_step
        stage_steps = self._layer_stage_step
        self._earliest_stage_step = (
            None
            if any(step is None for step in stage_steps)
            else min(stage_steps)
        )
        self.reuse_cache_complete = all(
            rows is not None for rows in self._selected_row_cache
        )

    def physical_rows_for_pages(
        self, local_layer: int, pages: list[int]
    ) -> torch.Tensor:
        """Resolve cached physical rows for the synchronization-heavy verifier."""
        if not self.external_sidecar or self.cached_pages is None:
            return self.device_rows[self._positions_of(pages)]
        torch.cuda.current_stream().synchronize()
        cached = self.cached_pages[local_layer].cpu().tolist()
        rows: list[torch.Tensor] = []
        for page in pages:
            try:
                slot = cached.index(page)
            except ValueError as error:
                raise RuntimeError(
                    f"selected page {page} is absent from the bounded cache"
                ) from error
            count = min(
                self.page_tokens,
                self.token_count - page * self.page_tokens,
            )
            begin = slot * self.page_tokens
            rows.append(self.device_rows[begin : begin + count])
        return torch.cat(rows)

    def enqueue_accounting_snapshot(self, stream: Any) -> tuple[Any, ...]:
        """Retain device counters until their nonblocking snapshot completes."""
        if not self.device_accounting:
            return ()
        self._copied_rows_host.copy_(self.copied_rows_device, non_blocking=True)
        self._accounting_event.record(stream)
        return (
            self._accounting_event,
            self._copied_rows_host,
            self.copied_rows_device,
            self.requested_rows,
        )

    def _verify_layer(
        self,
        local_layer: int,
        positions: torch.Tensor,
        count: int,
        physical_rows: torch.Tensor | None = None,
    ) -> None:
        controller = self.pending.controller
        device_pool = controller.mem_pool_device
        layer_id = self.start_layer + local_layer
        torch.cuda.current_stream().synchronize()
        for host_ref, device_buffer in (
            (controller.mem_pool_host.k_data_refs[local_layer],
             device_pool._get_key_buffer(layer_id)),
            (controller.mem_pool_host.v_data_refs[local_layer],
             device_pool._get_value_buffer(layer_id)),
        ):
            host_rows = host_ref[self.host_rows_cpu[positions.cpu()]]
            selected_rows = (
                self.selected_device_rows(positions)
                if physical_rows is None
                else physical_rows
            )
            staged_rows = device_buffer[selected_rows.to(torch.long)].cpu()
            staged_flat = staged_rows.view(count, -1)
            host_flat = host_rows.view(count, -1)
            if not torch.equal(staged_flat, host_flat):
                mismatch = (staged_flat != host_flat).any(dim=1)
                first = int(mismatch.nonzero()[0])
                raise RuntimeError(
                    f"tiered staging diverged from host rows at layer "
                    f"{local_layer}: first bad row {first}/{count} "
                    f"(claim position {int(positions[first])}), staged head "
                    f"{staged_flat[first, :4].tolist()}, host head "
                    f"{host_flat[first, :4].tolist()}, staged all-zero="
                    f"{bool((staged_flat[first] == 0).all())}"
                )

    def release_resources(self) -> None:
        release = getattr(self.pending, "release_resources", None)
        if callable(release):
            release()

    def choose_pages(
        self, local_layer: int, query: torch.Tensor, group_size: int
    ) -> list[int]:
        scores = quest_page_scores(
            query.to(torch.float32),
            self.kmin[local_layer],
            self.kmax[local_layer],
            group_size=group_size,
        ).sum(dim=0)
        chosen = budgeted_page_selection(
            scores, self.pages, min(self.budget_pages, self.pages),
            sink_pages=1, recent_pages=2,
        )
        return chosen.tolist()
