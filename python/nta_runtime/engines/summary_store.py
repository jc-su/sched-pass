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

import threading
from typing import Any

import torch


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
        self._order: list[int] = []
        self._bytes = 0
        self._lock = threading.Lock()
        self.recorded_nodes = 0
        self.evicted_nodes = 0

    def record(
        self,
        node_id: int,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        device_pool: Any,
        layer_ids: tuple[int, ...],
    ) -> None:
        """Reduce a written node's K rows into page envelopes.

        Runs on the caller's stream after the KV writes it summarizes, so
        ordering is inherited; the device rows stay allocated until the
        backup acks, which happens strictly later.
        """
        tokens = int(device_indices.numel())
        pages = tokens // self.page_tokens
        if pages == 0 or node_id < 0:
            return
        aligned = pages * self.page_tokens
        rows = device_indices[:aligned].to(
            device="cuda", dtype=torch.long, non_blocking=False
        )
        minima: list[torch.Tensor] = []
        maxima: list[torch.Tensor] = []
        for layer_id in layer_ids:
            key = device_pool._get_key_buffer(layer_id)[rows].float()
            paged = key.view(pages, self.page_tokens, *key.shape[1:])
            minima.append(paged.amin(dim=1))
            maxima.append(paged.amax(dim=1))
        kmin = torch.stack(minima).to("cpu")
        kmax = torch.stack(maxima).to("cpu")
        host_rows = host_indices[:aligned].to("cpu", torch.int64)
        entry_bytes = (
            kmin.numel() * kmin.element_size()
            + kmax.numel() * kmax.element_size()
            + host_rows.numel() * host_rows.element_size()
        )
        with self._lock:
            stale = self._entries.pop(node_id, None)
            if stale is not None:
                self._bytes -= stale[3]
                self._order.remove(node_id)
            while self._bytes + entry_bytes > self.capacity_bytes and self._order:
                evicted = self._order.pop(0)
                dropped = self._entries.pop(evicted, None)
                if dropped is not None:
                    self._bytes -= dropped[3]
                    self.evicted_nodes += 1
            self._entries[node_id] = (host_rows, kmin, kmax, entry_bytes)
            self._order.append(node_id)
            self._bytes += entry_bytes
            self.recorded_nodes += 1

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

        Every node must be recorded, start on a page boundary within the
        prefix, and carry exactly the host rows the claim sees now — a
        reused node id or a host-pool reshuffle fails the identity check
        and the caller scans instead.
        """
        if not node_ids or pages <= 0:
            return None
        with self._lock:
            entries = [self._entries.get(node_id) for node_id in node_ids]
        if any(entry is None for entry in entries):
            return None
        reference_min = entries[0][1]
        kmin = torch.zeros(
            (layer_count, pages, *reference_min.shape[2:]),
            dtype=torch.float32,
        )
        kmax = torch.zeros_like(kmin)
        offset = 0
        for entry in entries:
            host_rows, node_min, node_max, _ = entry
            if offset % self.page_tokens != 0:
                return None
            if node_min.shape[0] != layer_count:
                return None
            node_tokens = int(host_rows.numel())
            end = offset + node_tokens
            if end > token_count:
                return None
            if not torch.equal(
                host_rows, host_rows_cpu[offset:end].to(torch.int64)
            ):
                return None
            page_begin = offset // self.page_tokens
            node_pages = node_tokens // self.page_tokens
            kmin[:, page_begin : page_begin + node_pages] = node_min
            kmax[:, page_begin : page_begin + node_pages] = node_max
            offset = end
        if offset != token_count:
            # A trailing partial page (or an uncovered suffix) is legal
            # only if everything covered is page-aligned and the gap is
            # smaller than one page — the tail page is force-kept and its
            # envelope is masked before ranking.
            if token_count - offset >= self.page_tokens:
                return None
        return kmin, kmax
