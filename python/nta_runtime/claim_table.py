"""Fixed-shape device metadata for concurrent tiered claims.

Phase 1 of the graph-capturable multi-claim operator (design of record in
``docs/SELECTED_DEMAND.md``): every tensor is allocated once at
construction with a stable pointer and a fixed maximum shape — the
precondition for CUDA graph capture — and per-claim state lives in row
slices of those allocations, so a captured kernel can iterate claims from
device-resident words without any host round trip.

Slot lifecycle mirrors the bounded staging pool's discipline: ``acquire``
binds a slot to a (claim id, generation) and returns stable views;
``retire`` invalidates the slot immediately (kernels observe the cleared
validity word) and defers reuse behind a completion fence, so in-flight
work referencing the retired slot can never alias its successor.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any


@dataclass(frozen=True)
class ClaimSlot:
    """One bound claim row; identity changes whenever the row is reused."""

    index: int
    claim_id: int
    generation: int


class ClaimTable:
    """Own the per-claim device words the generated operator consumes."""

    def __init__(
        self,
        max_claims: int,
        max_budget_pages: int,
        page_tokens: int,
        *,
        layer_count: int = 1,
        max_claim_tokens: int = 32768,
        device: Any,
    ) -> None:
        import torch

        if (
            max_claims <= 0
            or max_budget_pages <= 0
            or page_tokens <= 0
            or layer_count <= 0
            or max_claim_tokens <= 0
        ):
            raise ValueError("claim-table geometry must be positive")
        self.max_claims = max_claims
        self.max_budget_pages = max_budget_pages
        self.page_tokens = page_tokens
        self.layer_count = layer_count
        self.max_claim_tokens = max_claim_tokens
        self.capacity_rows = max_budget_pages * page_tokens
        zeros = lambda *shape, dtype: torch.zeros(  # noqa: E731
            shape, dtype=dtype, device=device
        )
        # Claim-lifetime words. ``valid`` is the device-side truth kernels
        # branch on; ``active_count`` bounds device-side iteration.
        self.claim_ids = zeros(max_claims, dtype=torch.int64)
        self.generations = zeros(max_claims, dtype=torch.int32)
        self.valid = zeros(max_claims, dtype=torch.int32)
        self.active_count = zeros(1, dtype=torch.int32)
        self.page_counts = zeros(max_claims, dtype=torch.int32)
        self.token_counts = zeros(max_claims, dtype=torch.int32)
        self.kept_rows = zeros(max_claims, dtype=torch.int32)
        # Per-claim sub-tables as row slices of single allocations.
        self.selected_pages = zeros(
            max_claims, max_budget_pages, dtype=torch.int64
        )
        # The bounded page cache is per (claim, layer): each layer of a
        # claim owns its own slot-to-page mapping.
        self.cached_pages = torch.full(
            (max_claims, layer_count, max_budget_pages),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.staging_rows = zeros(
            max_claims, self.capacity_rows, dtype=torch.int32
        )
        self.selected_rows = zeros(
            max_claims, self.capacity_rows, dtype=torch.int32
        )
        self.source_indices = zeros(
            max_claims, self.capacity_rows, dtype=torch.int32
        )
        self.staging_indices = zeros(
            max_claims, self.capacity_rows, dtype=torch.int32
        )
        self.copied_rows = zeros(max_claims, dtype=torch.int64)
        # Geometry and dispatch words the table-driven prep kernel reads
        # per claim: the object-range base, the page-aligned staging
        # capacity, and the number of selected pages written for the
        # current layer.
        self.object_slots = zeros(max_claims, dtype=torch.int32)
        # Host-row identities per claim, so the table-driven kernel can
        # build transfer source indices without touching claim-owned
        # allocations.
        self.host_rows = zeros(max_claims, max_claim_tokens, dtype=torch.int32)
        self.capacity_words = zeros(max_claims, dtype=torch.int32)
        self.selected_counts = zeros(max_claims, dtype=torch.int32)
        self.layer_words = zeros(max_claims, dtype=torch.int32)
        self._generation_counter = [0] * max_claims
        self._bound: dict[int, ClaimSlot] = {}
        self._free = list(range(max_claims))
        self._retired: list[tuple[ClaimSlot, Any]] = []
        self._high_watermark = 0
        self._lock = threading.Lock()

    @property
    def high_watermark(self) -> int:
        with self._lock:
            return self._high_watermark

    def acquire(self, claim_id: int) -> ClaimSlot:
        """Bind a free row to a claim; the row's tensors remain stable."""
        if claim_id <= 0:
            raise ValueError("claim id must be positive")
        with self._lock:
            if claim_id in self._bound:
                raise RuntimeError("claim already owns a table row")
            if not self._free:
                raise RuntimeError("claim table is exhausted")
            index = self._free.pop()
            generation = (self._generation_counter[index] + 1) & 0x7FFFFFFF
            generation = generation or 1
            self._generation_counter[index] = generation
            slot = ClaimSlot(index, claim_id, generation)
            self._bound[claim_id] = slot
            self._high_watermark = max(self._high_watermark, len(self._bound))
        self.claim_ids[index] = claim_id
        self.generations[index] = generation
        self.copied_rows[index] = 0
        self.cached_pages[index].fill_(-1)
        return slot

    def activate(self, slot: ClaimSlot) -> None:
        """Publish the row to device iteration after its fields are set."""
        self._check(slot)
        self.valid[slot.index] = 1
        self._publish_count()

    def retire(self, slot: ClaimSlot, completion: Any) -> None:
        """Invalidate now; reuse only after ``completion`` reports done."""
        if not callable(getattr(completion, "query", None)):
            raise TypeError("claim retirement requires a queryable fence")
        with self._lock:
            if self._bound.get(slot.claim_id) != slot:
                raise RuntimeError("stale or foreign claim slot")
            self._bound.pop(slot.claim_id)
            self._retired.append((slot, completion))
        self.valid[slot.index] = 0
        self._publish_count()

    def reclaim(self) -> int:
        """Return fenced retired rows to the free list."""
        with self._lock:
            retired = self._retired
            self._retired = []
        pending: list[tuple[ClaimSlot, Any]] = []
        reclaimed = 0
        for slot, completion in retired:
            if completion.query():
                with self._lock:
                    self._free.append(slot.index)
                reclaimed += 1
            else:
                pending.append((slot, completion))
        if pending:
            with self._lock:
                self._retired.extend(pending)
        return reclaimed

    def views(self, slot: ClaimSlot) -> dict[str, Any]:
        """Stable per-claim tensor slices for staging and consumption."""
        self._check(slot)
        index = slot.index
        return {
            "selected_pages": self.selected_pages[index],
            "cached_pages": self.cached_pages[index],
            "staging_rows": self.staging_rows[index],
            "selected_rows": self.selected_rows[index],
            "source_indices": self.source_indices[index],
            "staging_indices": self.staging_indices[index],
            "copied_rows": self.copied_rows[index : index + 1],
            "host_rows": self.host_rows[index],
            "selected_count": self.selected_counts[index : index + 1],
            "layer_word": self.layer_words[index : index + 1],
        }

    def _check(self, slot: ClaimSlot) -> None:
        with self._lock:
            if self._bound.get(slot.claim_id) != slot:
                raise RuntimeError("stale or foreign claim slot")

    def _publish_count(self) -> None:
        with self._lock:
            count = len(self._bound)
        self.active_count[0] = count
