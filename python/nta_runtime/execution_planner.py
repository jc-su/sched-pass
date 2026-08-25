"""Measured planner for the unified late-bound execution mechanism.

This module contains cost estimation only.  It does not own request identity,
availability, or protocol state; those belong to ``execution_core`` and
``execution_protocol``.  A plan is a bounded execution choice, never a data
quality selector.
"""

from __future__ import annotations

import dataclasses
import math


@dataclasses.dataclass(frozen=True)
class HostCostModel:
    bandwidth_bytes_per_second: int = 30_000_000_000
    round_overhead_ns: int = 15_000
    incremental_setup_ns: int = 0
    tile_compute_ns: int = 3_000
    max_rounds: int = 4
    minimum_predicted_gain: float = 1.03
    dependency_width: int = 2

    @classmethod
    def from_environment(
        cls, environ: dict[str, str] | None = None, *, prefix: str = "NTA_EXECUTION"
    ) -> "HostCostModel":
        import os

        values = os.environ if environ is None else environ
        calibrated_bandwidth = values.get(
            "NTA_TIER_HOST_STAGED_BANDWIDTH_BPS", cls.bandwidth_bytes_per_second
        )
        return cls(
            bandwidth_bytes_per_second=int(
                values.get(
                    f"{prefix}_HOST_BANDWIDTH_BPS", calibrated_bandwidth
                )
            ),
            round_overhead_ns=int(
                values.get(f"{prefix}_ROUND_OVERHEAD_NS", cls.round_overhead_ns)
            ),
            incremental_setup_ns=int(
                values.get(f"{prefix}_INCREMENTAL_SETUP_NS", cls.incremental_setup_ns)
            ),
            tile_compute_ns=int(
                values.get(f"{prefix}_TILE_COMPUTE_NS", cls.tile_compute_ns)
            ),
            max_rounds=int(values.get(f"{prefix}_MAX_ROUNDS", cls.max_rounds)),
            minimum_predicted_gain=float(
                values.get(f"{prefix}_MIN_PREDICTED_GAIN", cls.minimum_predicted_gain)
            ),
        )

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
        if (
            min(self.round_overhead_ns, self.incremental_setup_ns) < 0
            or self.minimum_predicted_gain < 1.0
        ):
            raise ValueError("host execution overhead and gain are invalid")


@dataclasses.dataclass(frozen=True)
class HostExecutionPlan:
    block_counts: tuple[int, ...]
    predicted_atomic_ns: int
    predicted_incremental_ns: int
    overlap_initial: bool = False

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
    bulk_bandwidth_bytes_per_second: int = 55_000_000_000
    indexed_bandwidth_bytes_per_second: int = 55_000_000_000
    bulk_setup_ns: int = 10_000
    indexed_setup_ns: int = 80_000
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


def indexed_copy_blocks_per_group(
    *,
    transfer_bytes: int,
    object_count: int,
    target_bytes_per_block: int = 1024 * 1024,
    maximum_blocks: int = 32,
) -> int:
    if min(transfer_bytes, object_count, target_bytes_per_block, maximum_blocks) <= 0:
        raise ValueError("indexed copy geometry must be positive")
    object_groups = math.ceil(object_count / 2)
    bytes_per_group = math.ceil(transfer_bytes / object_groups)
    return min(
        maximum_blocks, max(1, math.ceil(bytes_per_group / target_bytes_per_block))
    )


def plan_device_demand(
    *,
    candidate_bytes: int,
    selected_bytes: int,
    selected_units: int,
    model: DeviceDemandCostModel,
) -> DeviceDemandPlan:
    model.validate()
    if min(candidate_bytes, selected_bytes, selected_units) <= 0:
        raise ValueError("device-demand planning needs non-empty transfer geometry")
    if selected_bytes > candidate_bytes:
        raise ValueError("selected bytes cannot exceed candidate bytes")
    bulk_ns = model.bulk_setup_ns + math.ceil(
        candidate_bytes * 1_000_000_000 / model.bulk_bandwidth_bytes_per_second
    )
    indexed_ns = (
        model.indexed_setup_ns
        + selected_units * model.indexed_page_overhead_ns
        + math.ceil(
            selected_bytes * 1_000_000_000 / model.indexed_bandwidth_bytes_per_second
        )
    )
    mode = "indexed" if bulk_ns / indexed_ns >= model.minimum_predicted_gain else "bulk"
    return DeviceDemandPlan(mode, bulk_ns, indexed_ns)


def conservative_resume_counts(
    *,
    block_counts: tuple[int, ...],
    work_count: int,
    max_object_fanout: int,
    min_unresolved_dependencies: int,
) -> tuple[int, ...]:
    if (
        not block_counts
        or min(
            *block_counts,
            work_count,
            max_object_fanout,
            min_unresolved_dependencies,
        )
        <= 0
    ):
        raise ValueError("resume-count geometry must be positive")
    cumulative_objects = 0
    result = []
    for blocks in block_counts:
        cumulative_objects += blocks
        result.append(
            min(
                work_count,
                math.ceil(
                    cumulative_objects * max_object_fanout / min_unresolved_dependencies
                ),
            )
        )
    return tuple(result)


def _round_width(object_count: int, rounds: int, dependency_width: int) -> int:
    width = math.ceil(object_count / rounds)
    return min(object_count, math.ceil(width / dependency_width) * dependency_width)


def _block_counts(object_count: int, width: int) -> tuple[int, ...]:
    return tuple(
        min(width, object_count - first) for first in range(0, object_count, width)
    )


def _pipeline_ns(
    transfer_ns: int,
    compute_ns: int,
    rounds: int,
    round_overhead_ns: int,
    incremental_setup_ns: int,
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
        + incremental_setup_ns
    )


def plan_host_execution(
    *,
    object_count: int,
    transfer_bytes: int,
    runnable_tiles: int,
    model: HostCostModel,
    force_rounds: int | None = None,
    initial_runnable_tiles: int = 0,
) -> HostExecutionPlan:
    model.validate()
    if min(object_count, transfer_bytes, runnable_tiles) <= 0:
        raise ValueError("host execution planning needs non-empty active work")
    if not 0 <= initial_runnable_tiles < runnable_tiles:
        raise ValueError("initial runnable tiles must be a proper work subset")
    if force_rounds is not None and force_rounds <= 0:
        raise ValueError("forced host execution needs a positive round bound")
    transfer_ns = math.ceil(
        transfer_bytes * 1_000_000_000 / model.bandwidth_bytes_per_second
    )
    compute_ns = runnable_tiles * model.tile_compute_ns
    initial_compute_ns = initial_runnable_tiles * model.tile_compute_ns
    deferred_compute_ns = compute_ns - initial_compute_ns
    atomic_ns = transfer_ns + compute_ns
    best_counts = (object_count,)
    best_ns = atomic_ns
    overlap_initial = False

    if initial_runnable_tiles:
        one_wave_ns = (
            max(transfer_ns, initial_compute_ns)
            + deferred_compute_ns
            + model.incremental_setup_ns
        )
        if (
            force_rounds is not None
            or atomic_ns / one_wave_ns >= model.minimum_predicted_gain
        ):
            best_ns = one_wave_ns
            overlap_initial = True

    max_rounds = min(model.max_rounds, math.ceil(object_count / model.dependency_width))
    if force_rounds is not None:
        selected_rounds = min(force_rounds, max_rounds)
        if selected_rounds <= 1:
            return HostExecutionPlan(best_counts, atomic_ns, best_ns, overlap_initial)
        width = _round_width(object_count, selected_rounds, model.dependency_width)
        counts = _block_counts(object_count, width)
        pipeline_ns = _pipeline_ns(
            transfer_ns,
            deferred_compute_ns,
            len(counts),
            model.round_overhead_ns,
            model.incremental_setup_ns,
        )
        return HostExecutionPlan(
            counts,
            atomic_ns,
            max(initial_compute_ns, pipeline_ns),
            initial_runnable_tiles != 0,
        )
    for requested_rounds in range(2, max_rounds + 1):
        width = _round_width(object_count, requested_rounds, model.dependency_width)
        counts = _block_counts(object_count, width)
        candidate_ns = max(
            initial_compute_ns,
            _pipeline_ns(
                transfer_ns,
                deferred_compute_ns,
                len(counts),
                model.round_overhead_ns,
                model.incremental_setup_ns,
            ),
        )
        if candidate_ns < best_ns:
            best_counts = counts
            best_ns = candidate_ns
            overlap_initial = initial_runnable_tiles != 0

    if atomic_ns / best_ns < model.minimum_predicted_gain:
        best_counts = (object_count,)
        best_ns = atomic_ns
        overlap_initial = False
    return HostExecutionPlan(best_counts, atomic_ns, best_ns, overlap_initial)
