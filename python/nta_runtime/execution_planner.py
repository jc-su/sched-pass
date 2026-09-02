"""Measured planner for the unified late-bound execution mechanism.

This module contains cost estimation only.  It does not own request identity,
availability, or protocol state; those belong to ``execution_core`` and
``execution_protocol``.  A plan is a bounded execution choice, never a data
quality selector.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
import math
from typing import Literal


@dataclasses.dataclass(frozen=True)
class HostCostModel:
    bandwidth_bytes_per_second: int = 30_000_000_000
    round_overhead_ns: int = 15_000
    # ``None`` is deliberately different from zero.  A missing measurement
    # must not make the automatic selector assume that typed metadata,
    # discovery, and the first incremental dispatch are free.
    incremental_setup_ns: int | None = None
    # Closed-loop correction for the complete device-side incremental
    # operator (discovery, transfer progress, numerical windows, and merge).
    # ``None`` is fail-closed for AUTO: transfer bandwidth and Python setup
    # alone cannot predict the cost of the typed numerical consumer.
    incremental_service_scale: float | None = None
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
        service_scale = values.get(f"{prefix}_INCREMENTAL_SERVICE_SCALE")
        return cls(
            bandwidth_bytes_per_second=int(
                values.get(f"{prefix}_HOST_BANDWIDTH_BPS", calibrated_bandwidth)
            ),
            round_overhead_ns=int(
                values.get(f"{prefix}_ROUND_OVERHEAD_NS", cls.round_overhead_ns)
            ),
            incremental_setup_ns=None if setup_value is None else int(setup_value),
            incremental_service_scale=(
                None if service_scale is None else float(service_scale)
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
        if self.round_overhead_ns < 0 or self.minimum_predicted_gain < 1.0:
            raise ValueError("host execution overhead and gain are invalid")
        if self.incremental_setup_ns is not None and self.incremental_setup_ns < 0:
            raise ValueError("incremental setup cost cannot be negative")
        if (
            self.incremental_service_scale is not None
            and not 0.125 <= self.incremental_service_scale <= 64.0
        ):
            raise ValueError("incremental service scale is outside its safe range")

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
            round((1.0 - alpha) * self.bandwidth_bytes_per_second + alpha * bounded),
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

    def with_incremental_service_observation(
        self,
        *,
        predicted_ns: int,
        predicted_scale: float,
        elapsed_ns: int,
        alpha: float = 0.25,
        maximum_step_ratio: float = 4.0,
    ) -> "HostCostModel":
        """Calibrate the device-side incremental service prediction.

        ``predicted_ns`` is the already-scaled prediction attached to the
        immutable execution decision, and ``predicted_scale`` is the scale
        that decision used. The pair reconstructs an absolute correction
        against the analytical base model even when CUDA completion is
        collected after a later decision has updated this model. The update
        is bounded because one CUDA scheduling interruption must not
        permanently disable an otherwise useful execution form.
        """

        if min(predicted_ns, elapsed_ns) <= 0:
            raise ValueError("incremental service observation must be positive")
        if not 0.125 <= predicted_scale <= 64.0:
            raise ValueError("incremental prediction scale is outside its safe range")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("incremental service alpha must be in (0, 1]")
        if maximum_step_ratio < 1.0:
            raise ValueError(
                "incremental service step ratio must be at least one"
            )
        previous = self.incremental_service_scale
        observed = predicted_scale * elapsed_ns / predicted_ns
        observed = min(64.0, max(0.125, observed))
        if previous is None:
            calibrated = observed
        else:
            lower = max(0.125, previous / maximum_step_ratio)
            upper = min(64.0, previous * maximum_step_ratio)
            bounded = min(max(observed, lower), upper)
            calibrated = (1.0 - alpha) * previous + alpha * bounded
        return dataclasses.replace(
            self,
            incremental_service_scale=min(64.0, max(0.125, calibrated)),
        )


class HostExecutionForm(str, Enum):
    """Explicit control path; wave count alone cannot identify ownership."""

    DIRECT = "direct"
    EAGER_PROGRESSIVE = "eager_progressive"
    SCHEDULED_BULK = "scheduled_bulk"
    DEVICE_BULK = "device_bulk"
    DEPENDENCY_AWARE = "dependency_aware"


class HostExecutionMode(str, Enum):
    """Selection mode, kept separate from the form ultimately executed.

    ``AUTO`` is the production policy.  The other values are explicit causal
    experiment controls and correctness/debugging tools; they are not hidden
    fallbacks and are reported in the resulting plan.
    """

    AUTO = "auto"
    DIRECT = "direct"
    EAGER_PROGRESSIVE = "eager_progressive"
    SCHEDULED_BULK = "scheduled_bulk"
    DEVICE_BULK = "device_bulk"
    DEPENDENCY_AWARE = "dependency_aware"


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
    form: HostExecutionForm
    overlap_initial: bool = False
    selection_reason: Literal[
        "conventional",
        "uncalibrated_setup",
        "calibration_probe",
        "consumer_policy_probe",
        "insufficient_gain",
        "predicted_gain",
        "tenant_isolation",
        "scheduled_preacquired",
        "forced_direct",
        "forced_eager_progressive",
        "forced_scheduled_bulk",
        "forced_device_bulk",
        "forced_dependency_aware",
    ] = "predicted_gain"
    scope_units: int = 1
    # Scale used when ``predicted_incremental_ns`` was constructed. Keeping
    # this on the immutable decision makes delayed CUDA observations
    # unambiguous even if a newer forward has recalibrated the model.
    incremental_service_scale: float = 1.0

    def __post_init__(self) -> None:
        if (
            self.scope_units <= 0
            or self.predicted_atomic_ns < 0
            or self.predicted_incremental_ns < 0
            or not 0.125 <= self.incremental_service_scale <= 64.0
        ):
            raise ValueError("host execution decision scope is invalid")
        if not isinstance(self.form, HostExecutionForm):
            raise TypeError("host execution form has an invalid type")
        if self.form in {
            HostExecutionForm.DIRECT,
            HostExecutionForm.SCHEDULED_BULK,
            HostExecutionForm.DEVICE_BULK,
        } and (len(self.block_counts) != 1 or self.overlap_initial):
            raise ValueError("bulk execution must be one non-overlapped wave")

    @property
    def rounds(self) -> int:
        return len(self.block_counts)

    @property
    def uses_dependency_protocol(self) -> bool:
        return self.form in {
            HostExecutionForm.EAGER_PROGRESSIVE,
            HostExecutionForm.DEVICE_BULK,
            HostExecutionForm.DEPENDENCY_AWARE,
        }

    @property
    def uses_scheduler_bound_acquisition(self) -> bool:
        """Whether transport ownership is bound at the scheduler shape edge."""

        return self.form in {
            HostExecutionForm.SCHEDULED_BULK,
            HostExecutionForm.DEPENDENCY_AWARE,
        }

    @property
    def uses_device_bulk(self) -> bool:
        return self.form is HostExecutionForm.DEVICE_BULK

    @property
    def uses_progressive_consumer(self) -> bool:
        return self.form in {
            HostExecutionForm.EAGER_PROGRESSIVE,
            HostExecutionForm.DEPENDENCY_AWARE,
        }

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


@dataclasses.dataclass(frozen=True)
class RunnableWorkWindows:
    """Exact disjoint slices of the device-published runnable queue.

    ``initial_count`` is the prefix published by discovery (resident work and
    dependencies completed by a preloaded fragment).  Every later progress
    wave appends ``counts[i]`` tickets, so ``offsets[i]`` is the exact queue
    prefix already consumed by earlier numerical launches.  Counts may be zero
    when a transport wave completes no work; callers must skip that numerical
    launch instead of inventing a duplicate one.
    """

    initial_count: int
    offsets: tuple[int, ...]
    counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.initial_count < 0
            or len(self.offsets) != len(self.counts)
            or any(count < 0 for count in self.counts)
        ):
            raise ValueError("runnable work windows have invalid geometry")
        cursor = self.initial_count
        for offset, count in zip(self.offsets, self.counts, strict=True):
            if offset != cursor:
                raise ValueError("runnable work windows are not contiguous")
            cursor += count

    @property
    def work_count(self) -> int:
        return self.initial_count + sum(self.counts)

    @property
    def launch_count(self) -> int:
        return int(self.initial_count != 0) + sum(count != 0 for count in self.counts)


def plan_exact_runnable_windows(
    *,
    external_object_slots: tuple[tuple[int, ...], ...],
    first_unresolved_object: int,
    block_counts: tuple[int, ...],
) -> RunnableWorkWindows:
    """Project sequential object completion into exact one-shot work launches.

    The prevalidated indexed-host path completes the contiguous object ranges
    in ``block_counts`` order. A work ticket becomes runnable in the first wave
    whose exclusive object boundary covers all of its dependencies. Shared
    acquisition groups and arbitrary fan-out remain exact because each work
    ticket is counted once, at its latest dependency. This removes the old
    one-group-per-work restriction without reading ``readyCount`` on the host.
    """

    if first_unresolved_object < 0 or not external_object_slots or not block_counts:
        raise ValueError("exact runnable-window geometry is empty")
    if any(count <= 0 for count in block_counts):
        raise ValueError("exact runnable-window progress must be positive")

    boundaries: list[int] = []
    cursor = first_unresolved_object
    for count in block_counts:
        cursor += count
        boundaries.append(cursor)

    initial_count = 0
    counts = [0] * len(block_counts)
    for raw_slots in external_object_slots:
        slots = tuple(int(slot) for slot in raw_slots)
        if any(slot < 0 for slot in slots):
            raise ValueError("runnable work names a negative object slot")
        unresolved = tuple(slot for slot in slots if slot >= first_unresolved_object)
        if not unresolved:
            initial_count += 1
            continue
        final_dependency = max(unresolved)
        for wave, boundary in enumerate(boundaries):
            if final_dependency < boundary:
                counts[wave] += 1
                break
        else:
            raise ValueError(
                "runnable work dependency is not covered by the progress waves"
            )

    offsets: list[int] = []
    cursor = initial_count
    for count in counts:
        offsets.append(cursor)
        cursor += count
    result = RunnableWorkWindows(initial_count, tuple(offsets), tuple(counts))
    if result.work_count != len(external_object_slots):  # pragma: no cover
        raise RuntimeError("runnable-window planning lost work")
    return result


@dataclasses.dataclass(frozen=True)
class HostLayerExecutionTemplate:
    """Immutable launch geometry shared by equivalent transformer layers.

    Request identity, acquisition-group ownership, and object addresses remain
    in the uploaded work plan and runtime directory.  This template contains
    only shape-stable orchestration: finite progress waves, numerical launch
    windows, and indexed-copy geometry.  Reusing it therefore removes Python
    planning from each layer without extending any resource lifetime.
    """

    object_count: int
    work_count: int
    direct_work_count: int
    preloaded_object_count: int
    queued_feasible_edf: bool
    progress_blocks: tuple[int, ...]
    ready_work_counts: tuple[int, ...]
    ready_work_offsets: tuple[int, ...] | None
    initial_ready_work_count: int
    demand_transfer_bytes: int
    indexed_host_first_object: int | None
    indexed_host_range_prevalidated: bool
    indexed_host_order_prevalidated: bool
    indexed_copy_blocks_per_group: int
    progressive_consumer: bool
    exact_resume_windows: bool

    def __post_init__(self) -> None:
        if (
            self.object_count <= 0
            or self.work_count <= 0
            or not self.progress_blocks
            or len(self.progress_blocks) != len(self.ready_work_counts)
            or self.demand_transfer_bytes <= 0
            or self.initial_ready_work_count < 0
            or not 0 <= self.direct_work_count < self.work_count
            or not 0 <= self.preloaded_object_count < self.object_count
            or self.indexed_copy_blocks_per_group <= 0
        ):
            raise ValueError("host layer execution template has invalid geometry")
        if self.ready_work_offsets is not None and len(self.ready_work_offsets) != len(
            self.progress_blocks
        ):
            raise ValueError("host layer execution offsets do not match progress")

    @property
    def progress_rounds(self) -> int:
        return len(self.progress_blocks)

    @property
    def nonempty_resume_windows(self) -> int:
        return sum(count != 0 for count in self.ready_work_counts)


def plan_host_layer_execution(
    *,
    host_execution: HostExecutionPlan,
    object_count: int,
    work_count: int,
    transfer_bytes: int,
    object_transfer_bytes: tuple[int, ...],
    external_object_slots: tuple[tuple[int, ...], ...],
    direct_work_count: int,
    max_object_fanout: int,
    min_unresolved_dependencies: int,
    preloaded_object_count: int,
    queued_feasible_edf: bool,
    indexed_copy_target_bytes: int,
    indexed_copy_max_blocks: int,
) -> HostLayerExecutionTemplate:
    """Build one exact host-acquisition/numerical launch template.

    ``DEVICE_BULK`` deliberately withholds all numerical work until its sole
    acquisition wave completes.  ``DEPENDENCY_AWARE`` exposes either exact
    disjoint windows for a prevalidated range or conservative device-queue
    prefixes when EDF/tenant ordering must remain on the GPU.
    """

    if (
        min(
            object_count,
            work_count,
            transfer_bytes,
            max_object_fanout,
            min_unresolved_dependencies,
            indexed_copy_target_bytes,
            indexed_copy_max_blocks,
        )
        <= 0
        or not 0 <= direct_work_count < work_count
        or not 0 <= preloaded_object_count < object_count
        or len(object_transfer_bytes) != object_count
        or sum(object_transfer_bytes) != transfer_bytes
        or len(external_object_slots) != work_count
    ):
        raise ValueError("host layer execution input geometry is incomplete")

    progress_blocks = host_execution.block_counts
    if preloaded_object_count:
        if host_execution.uses_device_bulk:
            raise ValueError("device-bulk execution cannot consume a fragment")
        if preloaded_object_count != progress_blocks[0]:
            raise ValueError("preloaded fragment does not match the first wave")
        progress_blocks = progress_blocks[1:]
    if not progress_blocks:
        raise ValueError("host layer execution has no unresolved acquisition wave")
    if host_execution.uses_device_bulk and (
        len(progress_blocks) != 1 or progress_blocks[0] != object_count
    ):
        raise ValueError("device-bulk execution requires one complete wave")

    demand_transfer_bytes = transfer_bytes - sum(
        object_transfer_bytes[:preloaded_object_count]
    )
    if demand_transfer_bytes <= 0:
        raise ValueError("host layer execution has no unresolved payload")

    initial_ready_work_count = (
        0
        if host_execution.uses_device_bulk
        else direct_work_count
        + sum(
            1
            for object_slots in external_object_slots
            if object_slots
            and all(slot < preloaded_object_count for slot in object_slots)
        )
    )
    if not host_execution.uses_device_bulk and initial_ready_work_count >= work_count:
        raise ValueError("dependency-aware execution has no deferred work")

    exact_resume_windows = False
    if host_execution.uses_device_bulk:
        ready_work_counts = (work_count,)
        ready_work_offsets: tuple[int, ...] | None = (0,)
    elif queued_feasible_edf:
        ready_work_counts = conservative_resume_counts(
            block_counts=progress_blocks,
            work_count=work_count - initial_ready_work_count,
            max_object_fanout=max_object_fanout,
            min_unresolved_dependencies=min_unresolved_dependencies,
        )
        ready_work_offsets = None
    else:
        windows = plan_exact_runnable_windows(
            external_object_slots=external_object_slots,
            first_unresolved_object=preloaded_object_count,
            block_counts=progress_blocks,
        )
        if (
            windows.initial_count != initial_ready_work_count
            or windows.work_count != work_count
        ):
            raise ValueError("exact runnable windows disagree with discovered work")
        ready_work_counts = windows.counts
        ready_work_offsets = windows.offsets
        exact_resume_windows = True

    progressive_consumer = host_execution.uses_progressive_consumer and (
        initial_ready_work_count != 0
        or sum(count != 0 for count in ready_work_counts) > 1
    )
    return HostLayerExecutionTemplate(
        object_count=object_count,
        work_count=work_count,
        direct_work_count=direct_work_count,
        preloaded_object_count=preloaded_object_count,
        queued_feasible_edf=queued_feasible_edf,
        progress_blocks=progress_blocks,
        ready_work_counts=ready_work_counts,
        ready_work_offsets=ready_work_offsets,
        initial_ready_work_count=initial_ready_work_count,
        demand_transfer_bytes=demand_transfer_bytes,
        indexed_host_first_object=(
            None if queued_feasible_edf else preloaded_object_count
        ),
        indexed_host_range_prevalidated=not queued_feasible_edf,
        indexed_host_order_prevalidated=not queued_feasible_edf,
        indexed_copy_blocks_per_group=indexed_copy_blocks_per_group(
            transfer_bytes=demand_transfer_bytes,
            object_count=object_count - preloaded_object_count,
            target_bytes_per_block=indexed_copy_target_bytes,
            maximum_blocks=indexed_copy_max_blocks,
        ),
        progressive_consumer=progressive_consumer,
        exact_resume_windows=exact_resume_windows,
    )


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


def _repeated_stage_pipeline_ns(
    transfer_ns: int,
    compute_ns: int,
    units: int,
) -> int:
    """Lower-bound a conventional transfer/compute pipeline.

    SGLang publishes one readiness event per transformer layer on a mover
    stream while numerical attention runs on the framework stream.  Once the
    first layer is available, transfer for layer ``i + 1`` can overlap
    attention for layer ``i``.  Treating every layer as ``transfer + compute``
    serial work invents ``units - 1`` pipeline bubbles and makes an
    incremental consumer appear profitable when it is not.

    This max-plus expression is intentionally favorable to the conventional
    path.  SM movers can contend with attention and real layers are not
    identical, so measured execution can be slower; an automatic selector
    must not choose the more expensive dependency protocol by assuming such
    contention creates free headroom.
    """

    if min(transfer_ns, compute_ns, units) <= 0:
        raise ValueError("repeated pipeline geometry must be positive")
    return transfer_ns + compute_ns + (units - 1) * max(transfer_ns, compute_ns)


def _incremental_service_ns(base_ns: int, model: HostCostModel) -> int:
    """Apply a completed device-service calibration to an analytical cost."""

    scale = (
        1.0
        if model.incremental_service_scale is None
        else model.incremental_service_scale
    )
    return max(1, math.ceil(base_ns * scale))


def _incremental_service_scale(model: HostCostModel) -> float:
    return (
        1.0
        if model.incremental_service_scale is None
        else model.incremental_service_scale
    )


def prove_atomic_host_execution(
    *,
    object_count: int,
    transfer_bytes: int,
    runnable_tiles: int,
    model: HostCostModel,
    scope_units: int = 1,
    mode: HostExecutionMode = HostExecutionMode.AUTO,
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
    if not isinstance(mode, HostExecutionMode):
        raise TypeError("host execution mode has an invalid type")
    if min(object_count, transfer_bytes, runnable_tiles, scope_units) <= 0:
        raise ValueError("host execution proof needs non-empty active work")
    transfer_ns = math.ceil(
        transfer_bytes * 1_000_000_000 / model.bandwidth_bytes_per_second
    )
    compute_ns = runnable_tiles * model.tile_compute_ns
    atomic_ns = _repeated_stage_pipeline_ns(
        transfer_ns,
        compute_ns,
        scope_units,
    )
    if mode in {
        HostExecutionMode.EAGER_PROGRESSIVE,
        HostExecutionMode.DEVICE_BULK,
        HostExecutionMode.DEPENDENCY_AWARE,
    }:
        return None
    if mode in {HostExecutionMode.DIRECT, HostExecutionMode.SCHEDULED_BULK}:
        scheduled = mode is HostExecutionMode.SCHEDULED_BULK
        return HostExecutionPlan(
            block_counts=(object_count,),
            predicted_atomic_ns=atomic_ns,
            predicted_incremental_ns=atomic_ns,
            form=(
                HostExecutionForm.SCHEDULED_BULK
                if scheduled
                else HostExecutionForm.DIRECT
            ),
            overlap_initial=False,
            selection_reason=(
                "forced_scheduled_bulk" if scheduled else "forced_direct"
            ),
            scope_units=scope_units,
            incremental_service_scale=_incremental_service_scale(model),
        )
    if (
        model.incremental_setup_ns is None
        or model.incremental_service_scale is None
    ):
        # AUTO cannot soundly select a consumer whose recurring setup has not
        # been observed.  Returning the direct proof here also avoids building
        # and uploading an exact dependency graph solely to reach the same
        # fail-closed decision later.
        return HostExecutionPlan(
            block_counts=(object_count,),
            predicted_atomic_ns=atomic_ns,
            predicted_incremental_ns=atomic_ns,
            form=HostExecutionForm.DIRECT,
            overlap_initial=False,
            selection_reason="uncalibrated_setup",
            scope_units=scope_units,
            incremental_service_scale=_incremental_service_scale(model),
        )
    optimistic_incremental_ns = (
        scope_units
        * _incremental_service_ns(max(transfer_ns, compute_ns), model)
        + model.incremental_setup_ns
    )
    if atomic_ns / optimistic_incremental_ns >= model.minimum_predicted_gain:
        return None
    return HostExecutionPlan(
        block_counts=(object_count,),
        predicted_atomic_ns=atomic_ns,
        predicted_incremental_ns=optimistic_incremental_ns,
        form=HostExecutionForm.DIRECT,
        overlap_initial=False,
        selection_reason="insufficient_gain",
        scope_units=scope_units,
        incremental_service_scale=_incremental_service_scale(model),
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
    require_dependency_protocol: bool = False,
    mode: HostExecutionMode = HostExecutionMode.AUTO,
) -> HostExecutionPlan:
    model.validate()
    if not isinstance(mode, HostExecutionMode):
        raise TypeError("host execution mode has an invalid type")
    if require_dependency_protocol and mode in {
        HostExecutionMode.DIRECT,
        HostExecutionMode.SCHEDULED_BULK,
    }:
        raise ValueError("tenant isolation requires typed host execution")
    if calibration_probe and mode is not HostExecutionMode.AUTO:
        raise ValueError("calibration probes require automatic host execution")
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
    atomic_ns = _repeated_stage_pipeline_ns(
        transfer_ns,
        compute_ns,
        scope_units,
    )
    if mode in {HostExecutionMode.DIRECT, HostExecutionMode.SCHEDULED_BULK}:
        scheduled = mode is HostExecutionMode.SCHEDULED_BULK
        return HostExecutionPlan(
            block_counts=(object_count,),
            predicted_atomic_ns=atomic_ns,
            predicted_incremental_ns=atomic_ns,
            form=(
                HostExecutionForm.SCHEDULED_BULK
                if scheduled
                else HostExecutionForm.DIRECT
            ),
            overlap_initial=False,
            selection_reason=(
                "forced_scheduled_bulk" if scheduled else "forced_direct"
            ),
            scope_units=scope_units,
            incremental_service_scale=_incremental_service_scale(model),
        )
    if mode is HostExecutionMode.DEVICE_BULK:
        setup_ns = (
            0 if model.incremental_setup_ns is None else model.incremental_setup_ns
        )
        return HostExecutionPlan(
            block_counts=(object_count,),
            predicted_atomic_ns=atomic_ns,
            predicted_incremental_ns=atomic_ns + setup_ns,
            form=HostExecutionForm.DEVICE_BULK,
            overlap_initial=False,
            selection_reason="forced_device_bulk",
            scope_units=scope_units,
            incremental_service_scale=_incremental_service_scale(model),
        )
    setup_ns = model.incremental_setup_ns
    if (
        (setup_ns is None or model.incremental_service_scale is None)
        and mode is HostExecutionMode.AUTO
        and not calibration_probe
        and not require_dependency_protocol
    ):
        return HostExecutionPlan(
            block_counts=(object_count,),
            predicted_atomic_ns=atomic_ns,
            predicted_incremental_ns=atomic_ns,
            form=HostExecutionForm.DIRECT,
            overlap_initial=False,
            selection_reason="uncalibrated_setup",
            scope_units=scope_units,
            incremental_service_scale=_incremental_service_scale(model),
        )
    if setup_ns is None:
        setup_ns = 0

    # A one-wave dependency-aware form is semantically distinct from direct
    # bulk staging: it uses object ownership, request generation, and tenant
    # credits even when the geometry cannot expose a second wave.
    one_wave_unit_ns = max(transfer_ns, initial_compute_ns) + deferred_compute_ns
    best_counts = (object_count,)
    best_ns = scope_units * _incremental_service_ns(one_wave_unit_ns, model) + setup_ns
    overlap_initial = initial_runnable_tiles != 0

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
        candidate_ns = (
            scope_units * _incremental_service_ns(candidate_unit_ns, model) + setup_ns
        )
        if candidate_ns < best_ns:
            best_counts = counts
            best_ns = candidate_ns
            overlap_initial = initial_runnable_tiles != 0

    selected = (
        calibration_probe
        or mode is HostExecutionMode.EAGER_PROGRESSIVE
        or mode is HostExecutionMode.DEPENDENCY_AWARE
        or require_dependency_protocol
        or atomic_ns / best_ns >= model.minimum_predicted_gain
    )
    # A probe is an explicitly bounded data-collection epoch, not a normal
    # policy decision. It must execute the path whose recurring setup cost is
    # unknown even when the cold prior predicts a loss; otherwise the probe
    # budget can never retire and AUTO can never become measured.
    if calibration_probe:
        reason = "calibration_probe"
    elif mode is HostExecutionMode.EAGER_PROGRESSIVE:
        reason = "forced_eager_progressive"
    elif mode is HostExecutionMode.DEPENDENCY_AWARE:
        reason = "forced_dependency_aware"
    elif require_dependency_protocol:
        reason = "tenant_isolation"
    elif selected:
        reason = "predicted_gain"
    else:
        reason = "insufficient_gain"
    return HostExecutionPlan(
        block_counts=best_counts if selected else (object_count,),
        predicted_atomic_ns=atomic_ns,
        predicted_incremental_ns=best_ns,
        form=(
            HostExecutionForm.EAGER_PROGRESSIVE
            if mode is HostExecutionMode.EAGER_PROGRESSIVE
            else HostExecutionForm.DEPENDENCY_AWARE
            if selected
            else HostExecutionForm.DIRECT
        ),
        overlap_initial=overlap_initial if selected else False,
        selection_reason=reason,
        scope_units=scope_units,
        incremental_service_scale=_incremental_service_scale(model),
    )
