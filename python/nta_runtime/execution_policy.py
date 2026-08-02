"""Cost model for finite request-aware host-staging rounds."""

from __future__ import annotations

import dataclasses
import math


@dataclasses.dataclass(frozen=True)
class HostCostModel:
    bandwidth_bytes_per_second: int = 30_000_000_000
    round_overhead_ns: int = 15_000
    tile_compute_ns: int = 3_000
    max_rounds: int = 4
    minimum_predicted_gain: float = 1.03
    dependency_width: int = 2

    def validate(self) -> None:
        if (
            min(
                self.bandwidth_bytes_per_second,
                self.tile_compute_ns,
                self.max_rounds,
                self.dependency_width,
            )
            <= 0
        ):
            raise ValueError("host execution cost parameters must be positive")
        if self.round_overhead_ns < 0 or self.minimum_predicted_gain < 1.0:
            raise ValueError("host execution overhead and gain are invalid")


@dataclasses.dataclass(frozen=True)
class HostExecutionPlan:
    block_counts: tuple[int, ...]
    predicted_atomic_ns: int
    predicted_incremental_ns: int

    @property
    def rounds(self) -> int:
        return len(self.block_counts)

    @property
    def predicted_gain(self) -> float:
        if self.predicted_incremental_ns == 0:
            return 1.0
        return self.predicted_atomic_ns / self.predicted_incremental_ns


@dataclasses.dataclass(frozen=True)
class DeviceDemandCostModel:
    """Calibrated costs for bulk candidates versus device-indexed demand."""

    bulk_bandwidth_bytes_per_second: int = 55_000_000_000
    indexed_bandwidth_bytes_per_second: int = 30_000_000_000
    bulk_setup_ns: int = 10_000
    indexed_setup_ns: int = 50_000
    indexed_page_overhead_ns: int = 100
    minimum_predicted_gain: float = 1.10

    def validate(self) -> None:
        if (
            min(
                self.bulk_bandwidth_bytes_per_second,
                self.indexed_bandwidth_bytes_per_second,
            )
            <= 0
        ):
            raise ValueError("device-demand bandwidths must be positive")
        if (
            min(
                self.bulk_setup_ns,
                self.indexed_setup_ns,
                self.indexed_page_overhead_ns,
            )
            < 0
            or self.minimum_predicted_gain < 1.0
        ):
            raise ValueError("device-demand overhead and gain are invalid")


@dataclasses.dataclass(frozen=True)
class DeviceDemandPlan:
    mode: str
    predicted_bulk_ns: int
    predicted_indexed_ns: int

    @property
    def predicted_gain(self) -> float:
        selected = (
            self.predicted_indexed_ns
            if self.mode == "indexed"
            else self.predicted_bulk_ns
        )
        return self.predicted_bulk_ns / selected


def plan_device_demand(
    *,
    candidate_bytes: int,
    selected_bytes: int,
    selected_pages: int,
    model: DeviceDemandCostModel,
) -> DeviceDemandPlan:
    """Choose bulk or indexed transfer without observing selected identities."""

    model.validate()
    if min(candidate_bytes, selected_bytes, selected_pages) <= 0:
        raise ValueError("device-demand planning needs non-empty transfer geometry")
    if selected_bytes > candidate_bytes:
        raise ValueError("selected bytes cannot exceed candidate bytes")
    bulk_ns = model.bulk_setup_ns + math.ceil(
        candidate_bytes * 1_000_000_000 / model.bulk_bandwidth_bytes_per_second
    )
    indexed_ns = (
        model.indexed_setup_ns
        + selected_pages * model.indexed_page_overhead_ns
        + math.ceil(
            selected_bytes * 1_000_000_000 / model.indexed_bandwidth_bytes_per_second
        )
    )
    mode = "indexed" if bulk_ns / indexed_ns >= model.minimum_predicted_gain else "bulk"
    return DeviceDemandPlan(mode, bulk_ns, indexed_ns)


def _round_width(object_count: int, rounds: int, dependency_width: int) -> int:
    width = math.ceil(object_count / rounds)
    return min(
        object_count,
        math.ceil(width / dependency_width) * dependency_width,
    )


def _block_counts(object_count: int, width: int) -> tuple[int, ...]:
    return tuple(
        min(width, object_count - first) for first in range(0, object_count, width)
    )


def _pipeline_ns(
    transfer_ns: int,
    compute_ns: int,
    rounds: int,
    round_overhead_ns: int,
) -> int:
    if rounds <= 1:
        return transfer_ns + compute_ns
    transfer_stage = math.ceil(transfer_ns / rounds)
    compute_stage = math.ceil(compute_ns / rounds)
    return (
        transfer_stage
        + compute_stage
        + (rounds - 1) * max(transfer_stage, compute_stage)
        + (rounds - 1) * round_overhead_ns
    )


def plan_host_execution(
    *,
    object_count: int,
    transfer_bytes: int,
    runnable_tiles: int,
    model: HostCostModel,
) -> HostExecutionPlan:
    """Choose one atomic round or a bounded transfer/compute pipeline.

    The model has no future oracle. It uses calibrated bandwidth, per-round
    launch cost, and compiler/runtime tile cost. Urgency affects queue order;
    this function controls only how much active work one finite round admits.
    """

    model.validate()
    if min(object_count, transfer_bytes, runnable_tiles) <= 0:
        raise ValueError("host execution planning needs non-empty active work")
    transfer_ns = math.ceil(
        transfer_bytes * 1_000_000_000 / model.bandwidth_bytes_per_second
    )
    compute_ns = runnable_tiles * model.tile_compute_ns
    atomic_ns = transfer_ns + compute_ns
    best_counts = (object_count,)
    best_ns = atomic_ns

    max_rounds = min(model.max_rounds, math.ceil(object_count / model.dependency_width))
    for requested_rounds in range(2, max_rounds + 1):
        width = _round_width(object_count, requested_rounds, model.dependency_width)
        counts = _block_counts(object_count, width)
        candidate_ns = _pipeline_ns(
            transfer_ns, compute_ns, len(counts), model.round_overhead_ns
        )
        if candidate_ns < best_ns:
            best_counts = counts
            best_ns = candidate_ns

    if atomic_ns / best_ns < model.minimum_predicted_gain:
        best_counts = (object_count,)
        best_ns = atomic_ns
    return HostExecutionPlan(best_counts, atomic_ns, best_ns)
