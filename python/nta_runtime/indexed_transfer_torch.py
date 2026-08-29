"""Device-side run analysis for indexed transfer maps.

The pure-Python planner in :mod:`nta_runtime.indexed_transfer` remains the
portable reference and artifact-analysis implementation. Serving adapters use
this module so an O(rows) device map is never downloaded and expanded into
Python objects merely to discover O(runs) contiguous regions.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time

import torch

from nta_runtime.indexed_transfer import (
    ContiguousPairRun,
    IndexedMoverServiceModel,
    IndexedPairLayout,
    select_indexed_mover_candidates,
)


@dataclass(frozen=True)
class TensorIndexedMoverPlan:
    """Exact copy-engine/SM partition retaining the SM remainder on device."""

    total_rows: int
    total_run_count: int
    layout: IndexedPairLayout | None
    copy_runs: tuple[ContiguousPairRun, ...]
    sm_source_indices: torch.Tensor
    sm_destination_indices: torch.Tensor
    predicted_sm_ns: int
    predicted_selected_ns: int | None
    selection_reason: str

    def __post_init__(self) -> None:
        if min(self.total_rows, self.total_run_count) <= 0:
            raise ValueError("indexed mover plan requires rows and exact runs")
        if self.layout is not None and (
            self.layout.row_count != self.total_rows
            or len(self.layout.runs) != self.total_run_count
        ):
            raise ValueError("indexed mover diagnostic layout is incomplete")
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
        if self.copy_row_count + self.sm_row_count != self.total_rows:
            raise ValueError("indexed mover plan does not cover every input row")
        if self.predicted_sm_ns <= 0 or (
            self.predicted_selected_ns is not None and self.predicted_selected_ns <= 0
        ):
            raise ValueError("indexed tensor mover predictions must be positive")

    @property
    def row_count(self) -> int:
        return self.total_rows

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


_WARMED_PLANNERS: set[tuple[str, int, int]] = set()
_WARMING_PLANNERS: dict[tuple[str, int, int], threading.Event] = {}
_WARMED_PLANNERS_LOCK = threading.Lock()


def warm_indexed_tensor_mover(
    device: torch.device | str,
    *,
    maximum_rows: int,
    maximum_copy_runs: int,
) -> int:
    """Pay one-time tensor-kernel initialization before serving requests.

    PyTorch lazily initializes the CUDA implementations used by run discovery
    and top-k selection. On a loaded serving process that one-time cost can be
    two orders of magnitude larger than steady-state planning. Exercise the
    production bounded-candidate path at the deployment's largest row shape
    and return its visible setup cost. The cache is process-local because CUDA
    initialization is process-local; callers expose the cost in setup
    telemetry rather than hiding it in a request warmup.
    """

    if maximum_rows <= 0 or maximum_copy_runs <= 0:
        raise ValueError("indexed mover warmup geometry must be positive")
    target = torch.device(device)
    row_count = min(maximum_rows, 1 << 16)
    copy_run_count = min(maximum_copy_runs, row_count)
    key = (str(target), row_count, copy_run_count)
    while True:
        with _WARMED_PLANNERS_LOCK:
            if key in _WARMED_PLANNERS:
                return 0
            completion = _WARMING_PLANNERS.get(key)
            owns_warmup = completion is None
            if owns_warmup:
                completion = threading.Event()
                _WARMING_PLANNERS[key] = completion
        if owns_warmup:
            break
        # Concurrent engine construction must not mistake an in-progress CUDA
        # warmup for a completed one. If the owner fails, one waiter retries.
        completion.wait()
    started = time.perf_counter_ns()
    try:
        source = torch.arange(row_count, dtype=torch.int32, device=target)
        # Every destination discontinuity is an exact one-row run. This
        # exercises the maximum bounded candidate set and the SM complement,
        # a strict superset of the kernels needed by dense maps.
        destination = source * 2
        service_model = IndexedMoverServiceModel(
            sm_bandwidth_bytes_per_second=1_000_000_000,
            copy_bandwidth_bytes_per_second=1_000_000_000,
            copy_operation_ns=1,
            sm_samples=3,
            copy_samples=3,
        )
        plan = plan_indexed_tensor_mover(
            source,
            destination,
            row_bytes=1,
            copy_operations_per_run=1,
            maximum_copy_runs=copy_run_count,
            service_model=service_model,
            policy="probe_copy",
            validate_unique_destinations=False,
            capture_full_layout=False,
        )
        if plan.total_run_count != row_count:
            raise RuntimeError("indexed mover warmup produced an invalid layout")
        if target.type == "cuda":
            torch.cuda.synchronize(target)
    except BaseException:
        with _WARMED_PLANNERS_LOCK:
            completion = _WARMING_PLANNERS.pop(key)
            completion.set()
        raise
    with _WARMED_PLANNERS_LOCK:
        _WARMED_PLANNERS.add(key)
        completion = _WARMING_PLANNERS.pop(key)
        completion.set()
    return time.perf_counter_ns() - started


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
    service_scale_bytes: int | None = None,
    validate_unique_destinations: bool = True,
    capture_full_layout: bool = True,
) -> TensorIndexedMoverPlan:
    """Partition an index map while downloading only maximal-run descriptors.

    Adjacency, run IDs, and the disjoint SM remainder are computed on the map's
    device. Production planning downloads at most ``maximum_copy_runs`` exact
    descriptors; a complete decomposition is materialized only for explicit
    layout profiling. ``service_scale_bytes`` optionally attests the physical
    wave bucket from which the caller selected its deployment curve; it must
    not be replaced by aggregate bytes spanning repeated waves.
    """

    source = _require_index_vector(source_indices, "source indices")
    destination = _require_index_vector(destination_indices, "destination indices")
    if source.device != destination.device or source.numel() != destination.numel():
        raise ValueError("indexed-transfer maps must share device and length")
    if validate_unique_destinations and destination.numel() > 1:
        ordered_destination = torch.sort(destination).values
        if bool(torch.any(ordered_destination[1:] == ordered_destination[:-1]).item()):
            raise ValueError("indexed-transfer destinations must be unique")

    row_count = int(source.numel())
    run_starts_mask = torch.empty(row_count, dtype=torch.bool, device=source.device)
    run_starts_mask[0] = True
    if row_count > 1:
        run_starts_mask[1:] = (source[1:] - source[:-1] != 1) | (
            destination[1:] - destination[:-1] != 1
        )
    run_starts = torch.nonzero(run_starts_mask, as_tuple=False).flatten()
    run_ends = torch.cat(
        (
            run_starts[1:],
            torch.tensor((row_count,), dtype=run_starts.dtype, device=source.device),
        )
    )
    run_lengths = run_ends - run_starts
    total_run_count = int(run_starts.numel())
    if capture_full_layout:
        candidate_indices = torch.arange(
            total_run_count, dtype=run_starts.dtype, device=source.device
        )
    elif policy == "sm":
        candidate_indices = torch.empty(0, dtype=run_starts.dtype, device=source.device)
    else:
        candidate_count = min(total_run_count, maximum_copy_runs)
        candidate_indices = torch.topk(
            run_lengths,
            candidate_count,
            largest=True,
            sorted=True,
        ).indices
    candidate_starts = run_starts.index_select(0, candidate_indices)
    candidate_descriptors = torch.stack(
        (
            candidate_indices.to(dtype=torch.int64),
            source.index_select(0, candidate_starts).to(dtype=torch.int64),
            destination.index_select(0, candidate_starts).to(dtype=torch.int64),
            run_lengths.index_select(0, candidate_indices).to(dtype=torch.int64),
        ),
        dim=1,
    )
    bounds = torch.stack(
        (source.min(), destination.min(), source.max(), destination.max())
    ).to(dtype=torch.int64, device=source.device)
    # Bounds and every descriptor the CPU selector can possibly choose share
    # one bounded D2H synchronization. A second selected-descriptor download
    # is redundant because selection is restricted to these exact candidates.
    metadata = torch.cat((bounds.reshape(1, 4), candidate_descriptors), dim=0).to(
        device="cpu"
    )
    metadata_rows = metadata.tolist()
    source_min, destination_min, source_max, destination_max = (
        int(value) for value in metadata_rows[0]
    )
    if min(source_min, destination_min) < 0:
        raise ValueError("indexed-transfer indices cannot be negative")
    if max(source_max, destination_max) >= 1 << 31:
        raise ValueError("indexed-transfer indices exceed signed int32 storage")
    descriptor_by_index = {
        int(run_index): ContiguousPairRun(
            int(source_first), int(destination_first), int(length)
        )
        for run_index, source_first, destination_first, length in metadata_rows[1:]
    }
    candidate_runs = tuple(
        (run_index, run.row_count) for run_index, run in descriptor_by_index.items()
    )
    selection = select_indexed_mover_candidates(
        total_rows=row_count,
        total_run_count=total_run_count,
        candidate_runs=candidate_runs,
        row_bytes=row_bytes,
        copy_operations_per_run=copy_operations_per_run,
        maximum_copy_runs=maximum_copy_runs,
        service_model=service_model,
        policy=policy,
        overlap_compute_ns=overlap_compute_ns,
        service_scale_bytes=service_scale_bytes,
    )
    selected_indices = set(selection.selected_run_indices)
    layout = (
        IndexedPairLayout(
            row_count,
            tuple(descriptor_by_index[index] for index in range(total_run_count)),
        )
        if capture_full_layout
        else None
    )
    copy_runs = tuple(descriptor_by_index[index] for index in sorted(selected_indices))
    if not selected_indices:
        sm_source = source.to(dtype=torch.int32)
        sm_destination = destination.to(dtype=torch.int32)
    elif sum(run.row_count for run in copy_runs) == row_count:
        sm_source = torch.empty(0, dtype=torch.int32, device=source.device)
        sm_destination = torch.empty(0, dtype=torch.int32, device=source.device)
    else:
        # Selection is bounded by ``maximum_copy_runs``.  Expanding it into a
        # Python bool tuple reintroduced O(total runs) interpreter work for a
        # fully scattered map even though descriptor D2H was bounded. Scatter
        # only the selected IDs into the device mask and keep the complement
        # construction on the GPU.
        selected_run_flags = torch.zeros(
            total_run_count, dtype=torch.bool, device=source.device
        )
        selected_run_flags.index_fill_(
            0,
            torch.tensor(
                tuple(sorted(selected_indices)),
                dtype=run_starts.dtype,
                device=source.device,
            ),
            True,
        )
        run_ids = torch.cumsum(run_starts_mask.to(dtype=torch.int64), dim=0) - 1
        sm_mask = ~selected_run_flags.index_select(0, run_ids)
        sm_source = source.masked_select(sm_mask).to(dtype=torch.int32)
        sm_destination = destination.masked_select(sm_mask).to(dtype=torch.int32)
    return TensorIndexedMoverPlan(
        row_count,
        total_run_count,
        layout,
        copy_runs,
        sm_source.contiguous(),
        sm_destination.contiguous(),
        selection.predicted_sm_ns,
        selection.predicted_selected_ns,
        selection.reason,
    )
