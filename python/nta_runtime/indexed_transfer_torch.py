"""Device-side run analysis for indexed transfer maps.

The pure-Python planner in :mod:`nta_runtime.indexed_transfer` remains the
portable reference and artifact-analysis implementation. Serving adapters use
this module so an O(rows) device map is never downloaded and expanded into
Python objects merely to discover O(runs) contiguous regions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nta_runtime.indexed_transfer import (
    ContiguousPairRun,
    IndexedMoverServiceModel,
    IndexedPairLayout,
    select_indexed_mover_runs,
)


@dataclass(frozen=True)
class TensorIndexedMoverPlan:
    """Exact copy-engine/SM partition retaining the SM remainder on device."""

    layout: IndexedPairLayout
    copy_runs: tuple[ContiguousPairRun, ...]
    sm_source_indices: torch.Tensor
    sm_destination_indices: torch.Tensor
    predicted_sm_ns: int
    predicted_selected_ns: int | None
    selection_reason: str

    def __post_init__(self) -> None:
        if self.sm_source_indices.device != self.sm_destination_indices.device:
            raise ValueError("indexed mover SM maps must share one device")
        if self.sm_source_indices.dtype is not torch.int32 or (
            self.sm_destination_indices.dtype is not torch.int32
        ):
            raise ValueError("indexed mover SM maps must use the uint32 ABI storage")
        if self.sm_source_indices.ndim != 1 or self.sm_destination_indices.ndim != 1:
            raise ValueError("indexed mover SM maps must be vectors")
        if self.sm_source_indices.numel() != self.sm_destination_indices.numel():
            raise ValueError("indexed mover SM maps disagree")
        if self.copy_row_count + self.sm_row_count != self.layout.row_count:
            raise ValueError("indexed mover plan does not cover every input row")
        if self.predicted_sm_ns <= 0 or (
            self.predicted_selected_ns is not None
            and self.predicted_selected_ns <= 0
        ):
            raise ValueError("indexed tensor mover predictions must be positive")

    @property
    def row_count(self) -> int:
        return self.layout.row_count

    @property
    def copy_row_count(self) -> int:
        return sum(run.row_count for run in self.copy_runs)

    @property
    def sm_row_count(self) -> int:
        return int(self.sm_source_indices.numel())

    @property
    def kind(self) -> str:
        if not self.copy_runs:
            return "sm"
        if self.sm_row_count == 0:
            return "copy_engine"
        return "hybrid"


def _require_index_vector(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if tensor.ndim != 1 or tensor.numel() <= 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if tensor.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must use int32 or int64 storage")
    return tensor.detach().contiguous()


def plan_indexed_tensor_mover(
    source_indices: torch.Tensor,
    destination_indices: torch.Tensor,
    *,
    row_bytes: int,
    copy_operations_per_run: int,
    maximum_copy_runs: int,
    service_model: IndexedMoverServiceModel,
    policy: str = "auto",
    overlap_compute_ns: int = 0,
    validate_unique_destinations: bool = True,
) -> TensorIndexedMoverPlan:
    """Partition an index map while downloading only maximal-run descriptors.

    Adjacency, run IDs, and the disjoint SM remainder are computed on the map's
    device. The CPU receives three integers per maximal run, independent of
    the number of rows in long contiguous regions.
    """

    source = _require_index_vector(source_indices, "source indices")
    destination = _require_index_vector(destination_indices, "destination indices")
    if source.device != destination.device or source.numel() != destination.numel():
        raise ValueError("indexed-transfer maps must share device and length")
    bounds = torch.stack(
        (source.min(), destination.min(), source.max(), destination.max())
    ).to(device="cpu")
    source_min, destination_min, source_max, destination_max = (
        int(value) for value in bounds.tolist()
    )
    if min(source_min, destination_min) < 0:
        raise ValueError("indexed-transfer indices cannot be negative")
    if max(source_max, destination_max) >= 1 << 32:
        raise ValueError("indexed-transfer indices exceed the uint32 ABI")
    if validate_unique_destinations and destination.numel() > 1:
        ordered_destination = torch.sort(destination).values
        if bool(torch.any(ordered_destination[1:] == ordered_destination[:-1]).item()):
            raise ValueError("indexed-transfer destinations must be unique")

    row_count = int(source.numel())
    run_starts_mask = torch.empty(row_count, dtype=torch.bool, device=source.device)
    run_starts_mask[0] = True
    if row_count > 1:
        run_starts_mask[1:] = (source[1:] != source[:-1] + 1) | (
            destination[1:] != destination[:-1] + 1
        )
    run_starts = torch.nonzero(run_starts_mask, as_tuple=False).flatten()
    run_ends = torch.cat(
        (
            run_starts[1:],
            torch.tensor((row_count,), dtype=run_starts.dtype, device=source.device),
        )
    )
    descriptors = torch.stack(
        (
            source.index_select(0, run_starts).to(dtype=torch.int64),
            destination.index_select(0, run_starts).to(dtype=torch.int64),
            run_ends - run_starts,
        ),
        dim=1,
    ).to(device="cpu")
    runs = tuple(
        ContiguousPairRun(int(source_first), int(destination_first), int(length))
        for source_first, destination_first, length in descriptors.tolist()
    )
    layout = IndexedPairLayout(row_count, runs)
    selection = select_indexed_mover_runs(
        layout,
        row_bytes=row_bytes,
        copy_operations_per_run=copy_operations_per_run,
        maximum_copy_runs=maximum_copy_runs,
        service_model=service_model,
        policy=policy,
        overlap_compute_ns=overlap_compute_ns,
    )
    selected_indices = set(selection.selected_run_indices)

    copy_runs = tuple(
        run for index, run in enumerate(runs) if index in selected_indices
    )
    if not selected_indices:
        sm_source = source.to(dtype=torch.int32)
        sm_destination = destination.to(dtype=torch.int32)
    elif len(selected_indices) == len(runs):
        sm_source = torch.empty(0, dtype=torch.int32, device=source.device)
        sm_destination = torch.empty(0, dtype=torch.int32, device=source.device)
    else:
        selected_run_flags = torch.tensor(
            tuple(index in selected_indices for index in range(len(runs))),
            dtype=torch.bool,
            device=source.device,
        )
        run_ids = torch.cumsum(run_starts_mask.to(dtype=torch.int64), dim=0) - 1
        sm_mask = ~selected_run_flags.index_select(0, run_ids)
        sm_source = source.masked_select(sm_mask).to(dtype=torch.int32)
        sm_destination = destination.masked_select(sm_mask).to(dtype=torch.int32)
    return TensorIndexedMoverPlan(
        layout,
        copy_runs,
        sm_source.contiguous(),
        sm_destination.contiguous(),
        selection.predicted_sm_ns,
        selection.predicted_selected_ns,
        selection.reason,
    )
