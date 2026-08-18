"""Writeback-time page envelopes for external-prefix claims.

Claim-creation envelope scans are the largest avoidable resident
interference: each new claim reads its full prefix's host K rows (~13 GB
per load trial) exactly when an arrival lands. The bytes already flowed
once — at HiCache writeback, while the KV still sits in device memory. The
store reduces each written node's K rows into per-(layer, page) min/max
envelopes at that moment, keeps them in host memory bounded by an LRU
byte budget, and hands claims their envelopes for the price of one small
host-to-device copy.

Correctness never depends on the store: a gather succeeds only when every
node is present, page-aligned within the prefix, and its recorded host
rows match the claim's exactly; any gap falls back to the existing scan
and is counted. Envelopes for forced pages (sinks, recency, tail) are
masked before ranking, so a partial trailing page contributes zeros
harmlessly.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import torch

logger = logging.getLogger(__name__)


class WritebackSummaryStore:
    """Per-node K envelopes recorded at device-to-host writeback."""

    def __init__(
        self, page_tokens: int, capacity_bytes: int, device: str = "cuda"
    ) -> None:
        self.device = device
        if page_tokens <= 0:
            raise RuntimeError("summary store needs a positive page size")
        if capacity_bytes <= 0:
            raise RuntimeError("summary store needs a positive byte budget")
        self.page_tokens = page_tokens
        self.capacity_bytes = capacity_bytes
        self._entries: dict[int, tuple[torch.Tensor, int]] = {}
        # Vectorized location: a flat host-row -> pool-slot table plus one
        # preallocated envelope pool make a claim's whole gather two torch
        # ops (an indexed load and a min/max reduction over each page's
        # sixteen rows' covering envelopes) instead of a Python loop that
        # stalled the scheduler ~90ms per claim in the P4-second campaign.
        # The per-row union equals the per-covering-page union bound.
        self._row_slots: torch.Tensor | None = None
        self._pool_min: torch.Tensor | None = None
        self._pool_max: torch.Tensor | None = None
        self._pool_free: list[int] = []
        self._pool_capacity = 0
        self._node_slots: dict[int, list[int]] = {}
        self._table_rows = 1 << 21
        self._order: list[int] = []
        self._bytes = 0
        self._lock = threading.Lock()
        self.recorded_nodes = 0
        self.evicted_nodes = 0
        self.miss_reasons: dict[str, int] = {}
        self.offset_counts: dict[int, int] = {}
        self.gather_ms_total = 0.0
        self.record_ms_total = 0.0
        self.record_calls = 0

    def record(
        self,
        node_id: int,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        device_pool: Any,
        layer_ids: tuple[int, ...],
        ancestor_host_rows: list[Any] | None = None,
        host_pool: Any = None,
    ) -> None:
        """Reduce a backed-up node's K rows into global-grid page envelopes.

        Incremental backups chunk at arbitrary token boundaries, but the
        backup invariant makes every ancestor's host rows available, so
        pages align to the sequence's global grid: pages fully inside this
        node reduce from its device rows, and the single page straddling
        the chunk boundary combines a handful of already-backed ancestor
        rows read from the pinned host pool. Keys are the pages' host-row
        tuples — content identity that survives radix splits.

        Runs on the caller's stream after the KV writes it summarizes; the
        device rows stay allocated until the backup acks.
        """
        import time as _time

        _record_began = _time.monotonic()
        tokens = int(device_indices.numel())
        if tokens <= 0 or node_id < 0:
            return
        offset = sum(int(len(rows)) for rows in (ancestor_host_rows or []))
        self.offset_counts[offset % self.page_tokens] = (
            self.offset_counts.get(offset % self.page_tokens, 0) + 1
        )
        first_page = offset // self.page_tokens
        boundary = offset % self.page_tokens
        end = offset + tokens
        last_full_page = end // self.page_tokens
        pages = last_full_page - first_page
        if pages <= 0:
            return
        node_host = host_indices.to("cpu", torch.int64)
        boundary_rows: torch.Tensor | None = None
        if boundary:
            if host_pool is None or not ancestor_host_rows:
                return
            tail: list[int] = []
            for rows in reversed(ancestor_host_rows):
                needed = boundary - len(tail)
                if needed <= 0:
                    break
                chunk = rows.to("cpu", torch.int64).tolist()
                tail = chunk[-needed:] + tail
            if len(tail) != boundary:
                return
            boundary_rows = torch.tensor(tail, dtype=torch.int64)
        covered = pages * self.page_tokens - boundary
        pool_device = device_pool._get_key_buffer(layer_ids[0]).device
        rows = device_indices[:covered].to(
            device=pool_device, dtype=torch.long, non_blocking=False
        )
        minima: list[torch.Tensor] = []
        maxima: list[torch.Tensor] = []
        for index, layer_id in enumerate(layer_ids):
            key = device_pool._get_key_buffer(layer_id)[rows]
            if boundary_rows is not None:
                host_key = host_pool.k_data_refs[index][boundary_rows].to(
                    key.device
                )
                key = torch.cat([host_key.to(key.dtype), key])
            paged = key.view(pages, self.page_tokens, *key.shape[1:])
            minima.append(paged.amin(dim=1))
            maxima.append(paged.amax(dim=1))
        # The pool lives on the device in fp16: min and max of fp16 data
        # are exact in fp16, record is a device-to-device write with no
        # host copy, and gathers return device tensors directly.
        kmin = torch.stack(minima).to(torch.float16)
        kmax = torch.stack(maxima).to(torch.float16)
        if boundary_rows is not None:
            host_rows = torch.cat([boundary_rows, node_host[:covered]])
        else:
            host_rows = node_host[:covered]
        entry_bytes = (
            kmin.numel() * kmin.element_size()
            + kmax.numel() * kmax.element_size()
            + host_rows.numel() * host_rows.element_size()
        )
        if int(host_rows.max()) >= self._table_rows or int(host_rows.min()) < 0:
            self.miss_reasons["record_row_out_of_range"] = (
                self.miss_reasons.get("record_row_out_of_range", 0) + 1
            )
            return
        with self._lock:
            self._drop(node_id)
            while self._bytes + entry_bytes > self.capacity_bytes and self._order:
                self._drop(self._order[0])
                self.evicted_nodes += 1
            self._ensure_pool(kmin.shape[0], kmin.shape[2:], pages)
            slots = [self._pool_free.pop() for _ in range(pages)]
            slot_tensor = torch.tensor(slots, dtype=torch.int64, device=self.device)
            self._pool_min[:, slot_tensor] = kmin
            self._pool_max[:, slot_tensor] = kmax
            page_rows = host_rows.view(pages, self.page_tokens).to(self.device)
            self._row_slots[page_rows.reshape(-1)] = slot_tensor.to(
                torch.int32
            ).repeat_interleave(self.page_tokens)
            self._entries[node_id] = (host_rows, entry_bytes)
            self._order.append(node_id)
            self._bytes += entry_bytes
            self._node_slots[node_id] = slots
            self.recorded_nodes += 1
        self.record_ms_total += (_time.monotonic() - _record_began) * 1_000.0
        self.record_calls += 1

    def _ensure_pool(
        self,
        layer_count: int,
        envelope_shape: torch.Size,
        pages_needed: int,
    ) -> None:
        if self._row_slots is None:
            self._row_slots = torch.full(
                (self._table_rows,), -1, dtype=torch.int32, device=self.device
            )
        while (
            self._pool_min is None
            or len(self._pool_free) < pages_needed
        ):
            # A growth copies the whole pool on the writeback path; start
            # large enough that a load-shape run never grows.
            grown = 32768 if self._pool_min is None else self._pool_capacity
            begin = self._pool_capacity
            new_min = torch.zeros(
                (layer_count, self._pool_capacity + grown, *envelope_shape),
                dtype=torch.float16,
                device=self.device,
            )
            new_max = torch.zeros_like(new_min)
            if self._pool_min is not None:
                new_min[:, : self._pool_capacity] = self._pool_min
                new_max[:, : self._pool_capacity] = self._pool_max
            self._pool_min = new_min
            self._pool_max = new_max
            self._pool_capacity += grown
            self._pool_free.extend(range(begin, self._pool_capacity))

    def _drop(self, node_id: int) -> None:
        entry = self._entries.pop(node_id, None)
        if entry is None:
            return
        self._bytes -= entry[1]
        self._order.remove(node_id)
        slots = self._node_slots.pop(node_id, [])
        if slots and self._row_slots is not None:
            rows = entry[0].to(self.device)
            slot_tensor = torch.tensor(slots, dtype=torch.int32, device=self.device)
            current = self._row_slots[rows]
            owned = (
                current.view(len(slots), self.page_tokens)
                == slot_tensor.unsqueeze(1)
            ).reshape(-1)
            cleared = torch.where(
                owned, torch.full_like(current, -1), current
            )
            self._row_slots[rows] = cleared
        self._pool_free.extend(slots)

    @property
    def stored_bytes(self) -> int:
        with self._lock:
            return self._bytes

    def gather(
        self,
        node_ids: tuple[int, ...],
        host_rows_cpu: torch.Tensor,
        layer_count: int,
        pages: int,
        token_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Assemble a claim's envelopes; None means fall back to the scan.

        Two torch ops: the slot table maps every full-page row to its
        newest covering envelope, and a min/max reduction over each claim
        page's sixteen rows takes their union — a valid, slightly looser
        bound at grid-phase boundaries, quality-gated by the scored
        battery. Any unmapped row misses the whole claim, fail-closed.
        """
        import time as _time

        _gather_began = _time.monotonic()
        del node_ids  # node identity does not survive radix splits
        if pages <= 0:
            self.miss_reasons["no_pages"] = (
                self.miss_reasons.get("no_pages", 0) + 1
            )
            return None
        full_pages = token_count // self.page_tokens
        if full_pages <= 0:
            self.miss_reasons["no_full_pages"] = (
                self.miss_reasons.get("no_full_pages", 0) + 1
            )
            return None
        with self._lock:
            if self._row_slots is None or self._pool_min is None:
                self.miss_reasons["store_empty"] = (
                    self.miss_reasons.get("store_empty", 0) + 1
                )
                return None
            rows = host_rows_cpu[: full_pages * self.page_tokens].to(
                device=self.device, dtype=torch.int64
            )
            if int(rows.max()) >= self._row_slots.numel() or int(rows.min()) < 0:
                self.miss_reasons["row_out_of_range"] = (
                    self.miss_reasons.get("row_out_of_range", 0) + 1
                )
                return None
            slots = self._row_slots[rows]
            if bool((slots < 0).any()):
                self.miss_reasons["page_rows_never_recorded"] = (
                    self.miss_reasons.get("page_rows_never_recorded", 0) + 1
                )
                return None
            if self._pool_min.shape[0] != layer_count:
                self.miss_reasons["layer_mismatch"] = (
                    self.miss_reasons.get("layer_mismatch", 0) + 1
                )
                return None
            page_slots = slots.to(torch.int64).view(
                full_pages, self.page_tokens
            )
            low = page_slots.min(dim=1).values
            high = page_slots.max(dim=1).values
            two_slot = bool(
                (
                    (page_slots == low.unsqueeze(1))
                    | (page_slots == high.unsqueeze(1))
                ).all()
            )
            if two_slot:
                # The common shapes: an aligned page maps to one recorded
                # slot (low == high) and a phase-shifted page straddles
                # exactly two consecutive ones. Gathering two slots per
                # page keeps the intermediate at two envelope copies
                # instead of sixteen — the sixteen-fold materialization
                # stalled claim prep ~890ms in the first probe.
                low_min = self._pool_min[:, low]
                high_min = self._pool_min[:, high]
                low_max = self._pool_max[:, low]
                high_max = self._pool_max[:, high]
                paged_min = torch.minimum(low_min, high_min)
                paged_max = torch.maximum(low_max, high_max)
            else:
                covering_min = self._pool_min[:, page_slots.reshape(-1)]
                covering_max = self._pool_max[:, page_slots.reshape(-1)]
                shape = covering_min.shape
                paged_min = covering_min.view(
                    layer_count, full_pages, self.page_tokens, *shape[2:]
                ).amin(dim=2)
                paged_max = covering_max.view(
                    layer_count, full_pages, self.page_tokens, *shape[2:]
                ).amax(dim=2)
        shape = paged_min.shape
        kmin = torch.zeros(
            (layer_count, pages, *shape[2:]),
            dtype=torch.float32,
            device=paged_min.device,
        )
        kmax = torch.zeros_like(kmin)
        kmin[:, :full_pages] = paged_min.to(torch.float32)
        kmax[:, :full_pages] = paged_max.to(torch.float32)
        self.gather_ms_total += (_time.monotonic() - _gather_began) * 1_000.0
        return kmin, kmax
