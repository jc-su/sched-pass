"""Tiered selected serving (1D stage 3b).

A claimed host prefix is never bulk-promoted. At claim time the engine builds
per-layer key envelopes from the host rows on the CPU (one scan, no device
copies), registers per-layer K/V indexed objects whose index arrays are
engine-owned device tensors, and completes SGLang's producer events so the
scheduler proceeds. At every attention layer — extend and decode — the
selector scores the prefix's envelopes against the live queries, retention
keeps sinks and the recent window, and only the chosen pages' rows move
host-to-device through the validated indexed path, bounded by the new
per-step row-count seam, into the device slots SGLang already allocated for
the prefix. Attention then runs over the chosen prefix rows plus the
request's resident tail; peer requests in a mixed decode batch keep their
full tables.

Stage-3b boundaries, recorded: selection is host-orchestrated (device
scoring, host argsort/index assembly — the device-resident tightening is a
pre-campaign optimization); copies are not yet hit-skipped (the staged
bitmap measures reuse for the economics without changing correctness); one
tiered claim at a time; cross-request radix reuse of a tiered prefix is
unsupported until the cache integration stage.
"""

from __future__ import annotations

from typing import Any

import torch

from nta_runtime.quest_selector import budgeted_page_selection, quest_page_scores
from nta_runtime.runtime import IndexedHostObject

_TIERED_OBJECT_ID_BASE = 0x54494552_00000000


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
        verify: bool,
    ) -> None:
        self.pending = pending
        self.budget_pages = budget_pages
        self.page_tokens = page_tokens
        self.first_object_slot = first_object_slot
        self.verify = verify
        controller = pending.controller
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
        self.device_rows = device_indices.to(device, torch.int32)
        self.host_rows_cpu = host_indices.to("cpu", torch.int64)

        host_pool = controller.mem_pool_host
        device_pool = controller.mem_pool_device
        start_layer = int(getattr(device_pool, "start_layer", 0))
        self.start_layer = start_layer

        # Claim-time envelopes from host keys: one CPU pass, no promotion.
        kmins, kmaxs = [], []
        for local_layer in range(self.layer_count):
            host_key = host_pool.k_data_refs[local_layer]
            rows = host_key[self.host_rows_cpu].to(torch.float32)
            heads, dim = rows.shape[-2], rows.shape[-1]
            full = (self.token_count // page_tokens) * page_tokens
            page_view = rows[:full].view(-1, page_tokens, heads, dim)
            kmin = page_view.amin(dim=1)
            kmax = page_view.amax(dim=1)
            if full < self.token_count:
                tail = rows[full:]
                kmin = torch.cat([kmin, tail.amin(dim=0, keepdim=True)])
                kmax = torch.cat([kmax, tail.amax(dim=0, keepdim=True)])
            kmins.append(kmin)
            kmaxs.append(kmax)
        self.kmin = torch.stack(kmins).to(device)
        self.kmax = torch.stack(kmaxs).to(device)

        # Shared per-step index arrays: every layer's objects point at the
        # same device tensors; layers rewrite the prefix sequentially on the
        # compute stream, so the copy consumes each layer's own contents.
        capacity = budget_pages * page_tokens
        self.capacity_rows = min(capacity, self.token_count)
        self.source_index = torch.zeros(
            self.capacity_rows, dtype=torch.int32, device=device
        )
        self.staging_index = torch.zeros(
            self.capacity_rows, dtype=torch.int32, device=device
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
            if host_value[0].numel() * host_value.element_size() != element:
                raise RuntimeError("tiered K/V row geometry disagrees")
            for source, staging in (
                (host_key, key_cache),
                (host_value, value_cache),
            ):
                objects.append(
                    IndexedHostObject(
                        _TIERED_OBJECT_ID_BASE + len(objects),
                        1,
                        source.data_ptr(),
                        staging.data_ptr(),
                        self.source_index.data_ptr(),
                        self.staging_index.data_ptr(),
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
        self.staged = torch.zeros(
            (self.layer_count, self.pages), dtype=torch.bool, device=device
        )
        # Order-independent request matching: the claim's mapping order is
        # the radix/controller order, not sequence order, so requests are
        # recognized by slot membership and every table token maps back to
        # its claim position through a pool-sized lookup.
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

    def page_row_count(self, chosen: list[int]) -> int:
        tail_rows = self.token_count - (self.pages - 1) * self.page_tokens
        count = 0
        for page in chosen:
            count += tail_rows if page == self.pages - 1 else self.page_tokens
        return count

    def stage_layer(
        self, engine: Any, local_layer: int, chosen: list[int],
        stream: Any,
    ) -> torch.Tensor:
        """Copy the chosen pages' rows for one layer; return kept positions."""
        positions = torch.cat(
            [
                torch.arange(
                    page * self.page_tokens,
                    min((page + 1) * self.page_tokens, self.token_count),
                    device=self.host_rows.device,
                    dtype=torch.int64,
                )
                for page in chosen
            ]
        )
        count = int(positions.numel())
        if count == 0 or count > self.capacity_rows:
            raise RuntimeError(
                f"tiered selection produced {count} rows outside capacity "
                f"{self.capacity_rows}"
            )
        self.source_index[:count] = self.host_rows[positions]
        self.staging_index[:count] = self.device_rows[positions]
        phases = engine._phase_program(engine._nta_demand_decode_wrappers[0])
        base = self.first_object_slot + 2 * local_layer
        phases.set_indexed_row_counts(
            engine._runtime, base, 2, count, stream=stream
        )
        phases.progress_validated_indexed_host_range(
            engine._runtime, base, 2, stream=stream
        )
        chosen_mask = torch.zeros(
            self.pages, dtype=torch.bool, device=self.staged.device
        )
        chosen_tensor = torch.tensor(
            chosen, dtype=torch.int64, device=self.staged.device
        )
        chosen_mask[chosen_tensor] = True
        previously = self.staged[local_layer] & chosen_mask
        self.rows_rehit += int(previously.sum()) * self.page_tokens
        self.staged[local_layer] |= chosen_mask
        self.rows_copied += count

        if self.verify:
            self._verify_layer(local_layer, positions, count)
        return positions

    def _verify_layer(
        self, local_layer: int, positions: torch.Tensor, count: int
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
            staged_rows = device_buffer[
                self.device_rows[positions].to(torch.long)
            ].cpu()
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
