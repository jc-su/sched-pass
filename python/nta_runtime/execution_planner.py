"""Measured planner for the unified late-bound execution mechanism.

This module contains cost estimation only.  It does not own request identity,
availability, or protocol state; those belong to ``execution_core`` and
``execution_protocol``.  A plan is a bounded execution choice, never a data
quality selector.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Literal


@dataclasses.dataclass(frozen=True)
class LayerDeadlineServiceCurve:
    """Measured lower envelope of useful compute between attention deadlines.

    A transport wave for layer ``i + 1`` can overlap the stream work between
    the attention arrivals of layers ``i`` and ``i + 1``.  The curve retains a
    bounded deployment-local sample window and exposes no slack until enough
    completed CUDA-event intervals exist.  Using the minimum completed sample
    is deliberately conservative: scheduler pauses may increase an interval,
    but can never manufacture optimistic transport slack.
    """

    samples_ns: tuple[int, ...] = ()
    minimum_samples: int = 4
    maximum_samples: int = 32

    def __post_init__(self) -> None:
        if self.minimum_samples <= 0 or self.maximum_samples < self.minimum_samples:
            raise ValueError("layer service-curve sample bounds are invalid")
        if len(self.samples_ns) > self.maximum_samples or any(
            sample <= 0 for sample in self.samples_ns
        ):
            raise ValueError("layer service-curve samples are invalid")

    @property
    def calibrated(self) -> bool:
        return len(self.samples_ns) >= self.minimum_samples

    @property
    def conservative_layer_ns(self) -> int:
        return min(self.samples_ns) if self.calibrated else 0

    def with_observation(self, elapsed_ns: int) -> "LayerDeadlineServiceCurve":
        if elapsed_ns <= 0:
            raise ValueError("layer service observation must be positive")
        samples = (*self.samples_ns, elapsed_ns)[-self.maximum_samples :]
        return dataclasses.replace(self, samples_ns=samples)

    def overlap_budget_ns(self, layer_intervals: int) -> int:
        if layer_intervals < 0:
            raise ValueError("layer service interval count cannot be negative")
        return self.conservative_layer_ns * layer_intervals


@dataclasses.dataclass(frozen=True)
class HostCostModel:
    bandwidth_bytes_per_second: int = 30_000_000_000
    round_overhead_ns: int = 15_000
    # ``None`` is deliberately different from zero.  A missing measurement
    # must not make the automatic selector assume that typed metadata,
    # discovery, and the first incremental dispatch are free.
    incremental_setup_ns: int | None = None
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
        setup_value = values.get(f"{prefix}_INCREMENTAL_SETUP_NS")
        return cls(
            bandwidth_bytes_per_second=int(
                values.get(f"{prefix}_HOST_BANDWIDTH_BPS", calibrated_bandwidth)
            ),
            round_overhead_ns=int(
                values.get(f"{prefix}_ROUND_OVERHEAD_NS", cls.round_overhead_ns)
            ),
            incremental_setup_ns=None if setup_value is None else int(setup_value),
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
        if self.round_overhead_ns < 0 or self.minimum_predicted_gain < 1.0:
            raise ValueError("host execution overhead and gain are invalid")
        if self.incremental_setup_ns is not None and self.incremental_setup_ns < 0:
            raise ValueError("incremental setup cost cannot be negative")

    def with_transfer_observation(
        self,
        *,
        transfer_bytes: int,
        elapsed_ns: int,
        alpha: float = 0.25,
        minimum_sample_bytes: int = 64 * 1024,
        maximum_step_ratio: float = 2.0,
    ) -> "HostCostModel":
        """Return a bounded EWMA calibration from one completed GPU transfer.

        The configured bandwidth is the cold-start prior.  Completed CUDA-event
        measurements update subsequent plans in-process, closing the old
        profile/model loop without letting one tiny or noisy sample swing a
        scheduling decision arbitrarily.
        """

        if min(transfer_bytes, elapsed_ns, minimum_sample_bytes) <= 0:
            raise ValueError("transfer calibration geometry must be positive")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("transfer calibration alpha must be in (0, 1]")
        if maximum_step_ratio < 1.0:
            raise ValueError("transfer calibration step ratio must be at least one")
        if transfer_bytes < minimum_sample_bytes:
            return self
        observed = max(1, transfer_bytes * 1_000_000_000 // elapsed_ns)
        lower = max(1, int(self.bandwidth_bytes_per_second / maximum_step_ratio))
        upper = max(lower, int(self.bandwidth_bytes_per_second * maximum_step_ratio))
        bounded = min(max(observed, lower), upper)
        calibrated = max(
            1,
            round(
                (1.0 - alpha) * self.bandwidth_bytes_per_second
                + alpha * bounded
            ),
        )
        return dataclasses.replace(self, bandwidth_bytes_per_second=calibrated)

    def with_incremental_setup_observation(
        self,
        *,
        elapsed_ns: int,
        alpha: float = 0.25,
        maximum_step_ratio: float = 4.0,
    ) -> "HostCostModel":
        """Return a bounded EWMA of recurring incremental control cost.

        The first observation establishes the deployment-local calibration.
        Later observations are step-bounded so one scheduler interruption does
        not permanently disable an otherwise useful execution form.
        """

        if elapsed_ns <= 0:
            raise ValueError("incremental setup observation must be positive")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("incremental setup alpha must be in (0, 1]")
        if maximum_step_ratio < 1.0:
            raise ValueError("incremental setup step ratio must be at least one")
        previous = self.incremental_setup_ns
        if previous is None or previous == 0:
            return dataclasses.replace(self, incremental_setup_ns=elapsed_ns)
        lower = max(1, int(previous / maximum_step_ratio))
        upper = max(lower, int(previous * maximum_step_ratio))
        bounded = min(max(elapsed_ns, lower), upper)
        calibrated = max(1, round((1.0 - alpha) * previous + alpha * bounded))
        return dataclasses.replace(self, incremental_setup_ns=calibrated)


@dataclasses.dataclass(frozen=True)
class HostExecutionPlan:
    """One execution-form decision over one or more equivalent layer units.

    ``predicted_*_ns`` are costs for the complete decision scope.  Transport,
    compute, and round overhead repeat for every unit; metadata/setup is paid
    once.  The selected ``block_counts`` remain the per-unit wave geometry.
    """

    block_counts: tuple[int, ...]
    predicted_atomic_ns: int
    predicted_incremental_ns: int
    overlap_initial: bool = False
    selection_reason: Literal[
        "conventional",
        "uncalibrated_setup",
        "calibration_probe",
        "insufficient_gain",
        "predicted_gain",
    ] = "predicted_gain"
    scope_units: int = 1

    def __post_init__(self) -> None:
        if (
            self.scope_units <= 0
            or self.predicted_atomic_ns < 0
            or self.predicted_incremental_ns < 0
        ):
            raise ValueError("host execution decision scope is invalid")

    @property
    def rounds(self) -> int:
        return len(self.block_counts)

    @property
    def predicted_gain(self) -> float:
        if self.predicted_incremental_ns == 0:
            return 1.0
        return self.predicted_atomic_ns / self.predicted_incremental_ns

    @property
    def predicted_atomic_per_unit_ns(self) -> int:
        return math.ceil(self.predicted_atomic_ns / self.scope_units)

    @property
    def predicted_incremental_per_unit_ns(self) -> int:
        return math.ceil(self.predicted_incremental_ns / self.scope_units)


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


def prove_atomic_host_execution(
    *,
    object_count: int,
    transfer_bytes: int,
    runnable_tiles: int,
    model: HostCostModel,
    scope_units: int = 1,
) -> HostExecutionPlan | None:
    """Prove that incremental execution cannot meet the configured gain.

    The proof deliberately assumes a stronger mechanism than the runtime can
    implement: transfer and all numerical work overlap perfectly, there is no
    round overhead, and dependencies never delay either resource. The only
    unavoidable incremental cost is the measured setup cost. If even this
    optimistic lower bound misses ``minimum_predicted_gain``, constructing an
    exact page/dependency graph cannot change the decision and is pure control
    overhead.

    ``None`` means only that the bound is inconclusive. It is not evidence
    that incremental execution is profitable; the exact planner must decide.
    """

    model.validate()
    if min(object_count, transfer_bytes, runnable_tiles, scope_units) <= 0:
        raise ValueError("host execution proof needs non-empty active work")
    if model.incremental_setup_ns is None:
        return None
    transfer_ns = math.ceil(
        transfer_bytes * 1_000_000_000 / model.bandwidth_bytes_per_second
    )
    compute_ns = runnable_tiles * model.tile_compute_ns
    atomic_ns = scope_units * (transfer_ns + compute_ns)
    optimistic_incremental_ns = (
        scope_units * max(transfer_ns, compute_ns) + model.incremental_setup_ns
    )
    if atomic_ns / optimistic_incremental_ns >= model.minimum_predicted_gain:
        return None
    return HostExecutionPlan(
        (object_count,),
        atomic_ns,
        atomic_ns,
        False,
        "insufficient_gain",
        scope_units,
    )


def plan_host_execution(
    *,
    object_count: int,
    transfer_bytes: int,
    runnable_tiles: int,
    model: HostCostModel,
    initial_runnable_tiles: int = 0,
    calibration_probe: bool = False,
    scope_units: int = 1,
) -> HostExecutionPlan:
    model.validate()
    if min(object_count, transfer_bytes, runnable_tiles, scope_units) <= 0:
        raise ValueError("host execution planning needs non-empty active work")
    if not 0 <= initial_runnable_tiles < runnable_tiles:
        raise ValueError("initial runnable tiles must be a proper work subset")
    transfer_ns = math.ceil(
        transfer_bytes * 1_000_000_000 / model.bandwidth_bytes_per_second
    )
    compute_ns = runnable_tiles * model.tile_compute_ns
    initial_compute_ns = initial_runnable_tiles * model.tile_compute_ns
    deferred_compute_ns = compute_ns - initial_compute_ns
    atomic_unit_ns = transfer_ns + compute_ns
    atomic_ns = scope_units * atomic_unit_ns
    best_counts = (object_count,)
    best_ns = atomic_ns
    overlap_initial = False

    setup_ns = model.incremental_setup_ns
    if setup_ns is None and not calibration_probe:
        return HostExecutionPlan(
            best_counts,
            atomic_ns,
            atomic_ns,
            False,
            "uncalibrated_setup",
            scope_units,
        )
    if setup_ns is None:
        setup_ns = 0

    if initial_runnable_tiles:
        one_wave_unit_ns = (
            max(transfer_ns, initial_compute_ns)
            + deferred_compute_ns
        )
        one_wave_ns = scope_units * one_wave_unit_ns + setup_ns
        if atomic_ns / one_wave_ns >= model.minimum_predicted_gain:
            best_ns = one_wave_ns
            overlap_initial = True

    max_rounds = min(model.max_rounds, math.ceil(object_count / model.dependency_width))
    for requested_rounds in range(2, max_rounds + 1):
        width = _round_width(object_count, requested_rounds, model.dependency_width)
        counts = _block_counts(object_count, width)
        candidate_unit_ns = max(
            initial_compute_ns,
            _pipeline_ns(
                transfer_ns,
                deferred_compute_ns,
                len(counts),
                model.round_overhead_ns,
                0,
            ),
        )
        candidate_ns = scope_units * candidate_unit_ns + setup_ns
        if candidate_ns < best_ns:
            best_counts = counts
            best_ns = candidate_ns
            overlap_initial = initial_runnable_tiles != 0

    if atomic_ns / best_ns < model.minimum_predicted_gain:
        best_counts = (object_count,)
        best_ns = atomic_ns
        overlap_initial = False
        reason = "insufficient_gain"
    else:
        reason = "calibration_probe" if calibration_probe else "predicted_gain"
    return HostExecutionPlan(
        best_counts,
        atomic_ns,
        best_ns,
        overlap_initial,
        reason,
        scope_units,
    )
