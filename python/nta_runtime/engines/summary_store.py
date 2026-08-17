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

    def __init__(self, page_tokens: int, capacity_bytes: int) -> None:
        if page_tokens <= 0:
            raise RuntimeError("summary store needs a positive page size")
        if capacity_bytes <= 0:
            raise RuntimeError("summary store needs a positive byte budget")
        self.page_tokens = page_tokens
        self.capacity_bytes = capacity_bytes
        self._entries: dict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]
        ] = {}
        # Content index: each recorded page is reachable by the tuple of
        # its host-row ids. Radix node ids do not survive tree splits, but
        # a page's host rows do — the key is simultaneously the identity
        # check, so a host-pool reshuffle simply stops matching.
        self._page_index: dict[tuple[int, ...], tuple[int, int]] = {}
        # Row locator: any recorded host row resolves to its (node, page).
        # A claim's page grid can be phase-shifted from the sequence grid
        # (its host rows start after the device-resident prefix), so a
        # claim page's sixteen consecutive rows fall inside at most two
        # consecutive recorded pages; the elementwise min/max of those two
        # envelopes is a valid, slightly looser bound for the page.
        self._row_locator: dict[int, tuple[int, int]] = {}
        self._order: list[int] = []
        self._bytes = 0
        self._lock = threading.Lock()
        self.recorded_nodes = 0
        self.evicted_nodes = 0
        self.miss_reasons: dict[str, int] = {}
        self.offset_counts: dict[int, int] = {}
        self._rows_recorded: set[int] = set()

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
        rows = device_indices[:covered].to(
            device="cuda", dtype=torch.long, non_blocking=False
        )
        minima: list[torch.Tensor] = []
        maxima: list[torch.Tensor] = []
        for index, layer_id in enumerate(layer_ids):
            key = device_pool._get_key_buffer(layer_id)[rows].float()
            if boundary_rows is not None:
                host_key = (
                    host_pool.k_data_refs[index][boundary_rows]
                    .to(key.device)
                    .float()
                )
                key = torch.cat([host_key, key])
            paged = key.view(pages, self.page_tokens, *key.shape[1:])
            minima.append(paged.amin(dim=1))
            maxima.append(paged.amax(dim=1))
        kmin = torch.stack(minima).to("cpu")
        kmax = torch.stack(maxima).to("cpu")
        if boundary_rows is not None:
            host_rows = torch.cat([boundary_rows, node_host[:covered]])
        else:
            host_rows = node_host[:covered]
        entry_bytes = (
            kmin.numel() * kmin.element_size()
            + kmax.numel() * kmax.element_size()
            + host_rows.numel() * host_rows.element_size()
        )
        host_rows_list = host_rows.tolist()
        with self._lock:
            self._drop(node_id)
            while self._bytes + entry_bytes > self.capacity_bytes and self._order:
                self._drop(self._order[0])
                self.evicted_nodes += 1
            self._entries[node_id] = (host_rows, kmin, kmax, entry_bytes)
            self._order.append(node_id)
            self._bytes += entry_bytes
            for page in range(pages):
                key = tuple(
                    host_rows_list[
                        page * self.page_tokens : (page + 1) * self.page_tokens
                    ]
                )
                self._page_index[key] = (node_id, page)
                self._rows_recorded.update(key)
                for row in key:
                    self._row_locator[row] = (node_id, page)
            self.recorded_nodes += 1

    def _drop(self, node_id: int) -> None:
        entry = self._entries.pop(node_id, None)
        if entry is None:
            return
        self._bytes -= entry[3]
        self._order.remove(node_id)
        host_rows_list = entry[0].tolist()
        pages = len(host_rows_list) // self.page_tokens
        for page in range(pages):
            key = tuple(
                host_rows_list[
                    page * self.page_tokens : (page + 1) * self.page_tokens
                ]
            )
            if self._page_index.get(key, (None, None))[0] == node_id:
                self._page_index.pop(key, None)
            for row in key:
                if self._row_locator.get(row, (None, None))[0] == node_id:
                    self._row_locator.pop(row, None)

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

        Each claim page resolves through the row locator to the one or two
        consecutive recorded pages containing its rows; a single covering
        page yields the exact envelope, two yield their elementwise union
        — a valid, looser bound whose quality effect the scored battery
        gates. Content identity is enforced by requiring every claim row
        to sit inside the covering pages' recorded rows.
        """
        del node_ids  # node identity does not survive radix splits
        if pages <= 0:
            self.miss_reasons["no_pages"] = (
                self.miss_reasons.get("no_pages", 0) + 1
            )
            return None
        host_rows_list = host_rows_cpu.tolist()
        full_pages = token_count // self.page_tokens
        kmin: torch.Tensor | None = None
        kmax: torch.Tensor | None = None
        with self._lock:
            for page in range(full_pages):
                rows = host_rows_list[
                    page * self.page_tokens : (page + 1) * self.page_tokens
                ]
                covering: list[tuple[int, int]] = []
                located_all = True
                for row in rows:
                    located = self._row_locator.get(row)
                    if located is None:
                        located_all = False
                        break
                    if located not in covering:
                        covering.append(located)
                if not located_all:
                    self.miss_reasons["page_rows_never_recorded"] = (
                        self.miss_reasons.get("page_rows_never_recorded", 0)
                        + 1
                    )
                    return None
                for node_id, _ in covering:
                    entry = self._entries.get(node_id)
                    if entry is None or entry[1].shape[0] != layer_count:
                        self.miss_reasons["entry_shape"] = (
                            self.miss_reasons.get("entry_shape", 0) + 1
                        )
                        return None
                if kmin is None:
                    reference = self._entries[covering[0][0]][1]
                    kmin = torch.zeros(
                        (layer_count, pages, *reference.shape[2:]),
                        dtype=torch.float32,
                    )
                    kmax = torch.zeros_like(kmin)
                page_min: torch.Tensor | None = None
                page_max: torch.Tensor | None = None
                for node_id, node_page in covering:
                    _, node_min, node_max, _ = self._entries[node_id]
                    part_min = node_min[:, node_page]
                    part_max = node_max[:, node_page]
                    page_min = (
                        part_min
                        if page_min is None
                        else torch.minimum(page_min, part_min)
                    )
                    page_max = (
                        part_max
                        if page_max is None
                        else torch.maximum(page_max, part_max)
                    )
                kmin[:, page] = page_min
                kmax[:, page] = page_max
        if kmin is None:
            self.miss_reasons["no_full_pages"] = (
                self.miss_reasons.get("no_full_pages", 0) + 1
            )
            return None
        return kmin, kmax
