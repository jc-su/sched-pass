"""Tier-neutral deadline scheduling for exact acquisition jobs.

The compiler and framework adapters prove *what* bytes a work unit consumes.
Transport backends own *how* those bytes move.  This module is the narrow
boundary between them: measured service demand is scheduled against numerical
consumer deadlines without importing CUDA, a framework, or a tier backend.

The core EDF theorem used here is intentionally explicit.  Jobs passed to
``schedule_acquisition_jobs`` are all available at time zero and share
one serialized link.  EDF minimizes maximum lateness for that model, so the
cumulative inequalities are an exact feasibility test.  Backends must not use
the result as a proof when jobs have different release times, hidden setup
work, or an independently contended link; those cases require a different
model rather than an optimistic flag.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from enum import Enum


@dataclass(frozen=True, slots=True)
class AcquisitionServiceCurve:
    """Conservative deployment-local compute service between deadlines."""

    samples_ns: tuple[int, ...] = ()
    minimum_samples: int = 4
    maximum_samples: int = 32

    def __post_init__(self) -> None:
        if self.minimum_samples <= 0 or self.maximum_samples < self.minimum_samples:
            raise ValueError("acquisition service-curve sample bounds are invalid")
        if len(self.samples_ns) > self.maximum_samples or any(
            sample <= 0 for sample in self.samples_ns
        ):
            raise ValueError("acquisition service-curve samples are invalid")

    @property
    def calibrated(self) -> bool:
        return len(self.samples_ns) >= self.minimum_samples

    @property
    def conservative_interval_ns(self) -> int:
        return min(self.samples_ns) if self.calibrated else 0

    def with_observation(self, elapsed_ns: int) -> "AcquisitionServiceCurve":
        if elapsed_ns <= 0:
            raise ValueError("acquisition service observation must be positive")
        samples = (*self.samples_ns, elapsed_ns)[-self.maximum_samples :]
        return replace(self, samples_ns=samples)

    def overlap_budget_ns(self, intervals: int) -> int:
        if intervals < 0:
            raise ValueError("acquisition service interval count cannot be negative")
        return self.conservative_interval_ns * intervals


@dataclass(frozen=True, slots=True)
class AcquisitionWork:
    """Identity and payload owned by one acquisition lifecycle.

    This descriptor deliberately has no timing fields.  Exact work can become
    transport-ready before a deployment-calibrated deadline model exists; the
    lifecycle queue must not manufacture service estimates merely to represent
    that state.
    """

    job_id: int
    payload_bytes: int

    def __post_init__(self) -> None:
        if isinstance(self.job_id, bool) or not isinstance(self.job_id, int):
            raise TypeError("acquisition work ID must be an integer")
        if self.job_id < 0:
            raise ValueError("acquisition work ID cannot be negative")
        if self.payload_bytes <= 0:
            raise ValueError("acquisition work payload must be positive")


@dataclass(frozen=True, slots=True)
class AcquisitionJob(AcquisitionWork):
    """One non-preemptive transfer job in a simultaneous-release schedule.

    ``job_id`` is an execution-local ordinal.  Exact request generation,
    segment, and resource-version identity stays in the acquisition topology;
    the scheduler never hashes or reconstructs semantic identity.
    """

    service_ns: int
    deadline_ns: int

    def __post_init__(self) -> None:
        AcquisitionWork.__post_init__(self)
        if self.service_ns <= 0:
            raise ValueError("EDF job service must be positive")
        if self.deadline_ns < 0:
            raise ValueError("EDF job deadline cannot be negative")


@dataclass(frozen=True, slots=True)
class AcquisitionSchedule:
    """Auditable result of one exact simultaneous-release EDF test."""

    ordered_job_ids: tuple[int, ...]
    completion_ns: tuple[int, ...]
    deadlines_ns: tuple[int, ...]
    first_missed_job_id: int | None
    required_initial_slack_ns: int

    def __post_init__(self) -> None:
        count = len(self.ordered_job_ids)
        if (
            len(set(self.ordered_job_ids)) != count
            or len(self.completion_ns) != count
            or len(self.deadlines_ns) != count
            or any(value <= 0 for value in self.completion_ns)
            or any(value < 0 for value in self.deadlines_ns)
            or self.required_initial_slack_ns < 0
        ):
            raise ValueError("EDF schedule is internally inconsistent")
        if self.first_missed_job_id is not None and (
            self.first_missed_job_id not in self.ordered_job_ids
        ):
            raise ValueError("EDF missed job is outside the schedule")
        if self.feasible != (self.required_initial_slack_ns == 0):
            raise ValueError("EDF feasibility and required slack disagree")

    @property
    def feasible(self) -> bool:
        return self.first_missed_job_id is None


def schedule_acquisition_jobs(
    jobs: Iterable[AcquisitionJob],
) -> AcquisitionSchedule:
    """Schedule one serialized link and test every cumulative EDF deadline.

    The caller may provide jobs in any order.  Equal deadlines retain their
    explicit ``job_id`` order so both execution and evidence are deterministic.
    """

    values = tuple(jobs)
    if len({job.job_id for job in values}) != len(values):
        raise ValueError("EDF jobs must have unique IDs")
    ordered = tuple(sorted(values, key=lambda job: (job.deadline_ns, job.job_id)))
    elapsed_ns = 0
    maximum_lateness_ns = 0
    first_missed_job_id: int | None = None
    completion_ns: list[int] = []
    for job in ordered:
        elapsed_ns += job.service_ns
        completion_ns.append(elapsed_ns)
        lateness_ns = elapsed_ns - job.deadline_ns
        if lateness_ns > 0:
            maximum_lateness_ns = max(maximum_lateness_ns, lateness_ns)
            if first_missed_job_id is None:
                first_missed_job_id = job.job_id
    return AcquisitionSchedule(
        ordered_job_ids=tuple(job.job_id for job in ordered),
        completion_ns=tuple(completion_ns),
        deadlines_ns=tuple(job.deadline_ns for job in ordered),
        first_missed_job_id=first_missed_job_id,
        required_initial_slack_ns=maximum_lateness_ns,
    )


class AcquisitionJobState(str, Enum):
    """Control-plane lifecycle of one exact acquisition job.

    ``FENCE_PUBLISHED`` means the backend has submitted the transfer and
    published the readiness primitive consumed by numerical execution.  It does
    not claim that the bytes are already resident; the backend-specific fence
    or native object state remains the source of that fact.
    """

    PLANNED = "planned"
    SUBMITTED = "submitted"
    FENCE_PUBLISHED = "fence_published"
    CONSUMED = "consumed"
    CANCELLED = "cancelled"
    FAILED = "failed"


_TERMINAL_JOB_STATES = frozenset(
    {
        AcquisitionJobState.CONSUMED,
        AcquisitionJobState.CANCELLED,
        AcquisitionJobState.FAILED,
    }
)


class AcquisitionQueue:
    """Bounded work-conserving lifecycle for one ordered transport queue.

    Scheduling policy is intentionally outside this state machine.  A caller
    may pass calibrated EDF order, or a structural consumer order when work is
    ready before timing calibration.  Capacity limits outstanding submissions,
    not total workload size: retiring or cancelling a job immediately exposes
    the next planned job.  CUDA events, NVMe commands, HBM addresses, and
    framework leases deliberately remain outside this dependency-free owner.
    """

    def __init__(
        self,
        jobs: Iterable[AcquisitionWork],
        *,
        ordered_job_ids: Iterable[int],
        max_inflight_jobs: int,
    ) -> None:
        values = tuple(jobs)
        if isinstance(max_inflight_jobs, bool) or not isinstance(
            max_inflight_jobs, int
        ):
            raise TypeError("acquisition in-flight capacity must be an integer")
        if max_inflight_jobs <= 0:
            raise ValueError("acquisition in-flight capacity must be positive")
        if len({job.job_id for job in values}) != len(values):
            raise ValueError("acquisition work must have unique IDs")
        self._jobs = {job.job_id: job for job in values}
        order = tuple(ordered_job_ids)
        if len(order) != len(values) or set(order) != set(self._jobs):
            raise ValueError(
                "acquisition execution order must cover every job exactly once"
            )
        self._ordered_job_ids = order
        self._states = {
            job_id: AcquisitionJobState.PLANNED
            for job_id in self._ordered_job_ids
        }
        self._state_counts = {
            state: (len(values) if state is AcquisitionJobState.PLANNED else 0)
            for state in AcquisitionJobState
        }
        self._max_inflight_jobs = max_inflight_jobs
        self._next_job = 0

    @classmethod
    def from_edf(
        cls,
        jobs: Iterable[AcquisitionJob],
        *,
        max_inflight_jobs: int,
    ) -> "AcquisitionQueue":
        """Create a lifecycle queue from one explicit calibrated EDF result."""

        values = tuple(jobs)
        schedule = schedule_acquisition_jobs(values)
        return cls(
            values,
            ordered_job_ids=schedule.ordered_job_ids,
            max_inflight_jobs=max_inflight_jobs,
        )

    @property
    def max_inflight_jobs(self) -> int:
        return self._max_inflight_jobs

    @property
    def job_ids(self) -> tuple[int, ...]:
        """Return the immutable execution order owned by this queue."""

        return self._ordered_job_ids

    def state(self, job_id: int) -> AcquisitionJobState:
        try:
            return self._states[job_id]
        except KeyError as error:
            raise KeyError(f"unknown acquisition job {job_id}") from error

    @property
    def inflight_count(self) -> int:
        return self.count_states(
            AcquisitionJobState.SUBMITTED,
            AcquisitionJobState.FENCE_PUBLISHED,
        )

    @property
    def terminal(self) -> bool:
        return self.count_states(*_TERMINAL_JOB_STATES) == len(self._states)

    def count_states(self, *states: AcquisitionJobState) -> int:
        """Return an O(number-of-states) lifecycle count.

        Admission and per-layer retirement query this queue on a serving hot
        path.  Maintaining exact state cardinalities avoids repeatedly scanning
        every acquisition job as model depth or frontier granularity grows.
        """

        if not states or len(set(states)) != len(states):
            raise ValueError("acquisition state query must be unique and non-empty")
        return sum(self._state_counts[state] for state in states)

    def _set_state(self, job_id: int, target: AcquisitionJobState) -> None:
        current = self._states[job_id]
        if current is target:
            return
        self._state_counts[current] -= 1
        self._state_counts[target] += 1
        self._states[job_id] = target

    def cancel_unfinished(self) -> None:
        """Cancel every nonterminal job at an exceptional lifetime boundary."""

        for job_id in self._ordered_job_ids:
            if self._states[job_id] not in _TERMINAL_JOB_STATES:
                self._set_state(job_id, AcquisitionJobState.CANCELLED)

    def claim(self) -> tuple[AcquisitionWork, ...]:
        """Fill every available submission slot in the bound execution order."""

        available = self._max_inflight_jobs - self.inflight_count
        claimed: list[AcquisitionWork] = []
        order = self._ordered_job_ids
        while available > 0 and self._next_job < len(order):
            job_id = order[self._next_job]
            self._next_job += 1
            state = self._states[job_id]
            if state is not AcquisitionJobState.PLANNED:
                continue
            self._set_state(job_id, AcquisitionJobState.SUBMITTED)
            claimed.append(self._jobs[job_id])
            available -= 1
        return tuple(claimed)

    def publish_fence(self, job_id: int) -> None:
        """Publish the backend readiness primitive after successful submission."""

        self._transition(
            job_id,
            AcquisitionJobState.SUBMITTED,
            AcquisitionJobState.FENCE_PUBLISHED,
        )

    def retire(self, job_id: int) -> None:
        """Retire one job after its final numerical consumer is ordered."""

        self._transition(
            job_id,
            AcquisitionJobState.FENCE_PUBLISHED,
            AcquisitionJobState.CONSUMED,
        )

    def cancel(self, job_id: int) -> None:
        self._finish(job_id, AcquisitionJobState.CANCELLED)

    def fail(self, job_id: int) -> None:
        self._finish(job_id, AcquisitionJobState.FAILED)

    def _finish(self, job_id: int, target: AcquisitionJobState) -> None:
        state = self.state(job_id)
        if state in _TERMINAL_JOB_STATES:
            raise ValueError(
                f"terminal acquisition job {job_id} cannot become {target.value}"
            )
        self._set_state(job_id, target)

    def _transition(
        self,
        job_id: int,
        expected: AcquisitionJobState,
        target: AcquisitionJobState,
    ) -> None:
        state = self.state(job_id)
        if state is not expected:
            raise ValueError(
                f"acquisition job {job_id} cannot transition "
                f"{state.value} -> {target.value}"
            )
        self._set_state(job_id, target)


@dataclass(frozen=True, slots=True)
class LayerAcquisitionModel:
    """Transformer projection of exact acquisition jobs onto one tier link.

    Every unresolved layer is available when the external-resource lease is
    captured.  ``initial_compute_ns`` is useful work before layer-zero
    attention; ``inter_layer_compute_ns`` is the conservative useful interval
    between subsequent attention arrivals.  Transfer service comes from the
    selected backend's measured model, never from a hard-coded byte threshold.
    """

    layer_bytes: tuple[int, ...]
    transfer_service_ns: tuple[int, ...]
    initial_compute_ns: int
    inter_layer_compute_ns: int

    def __post_init__(self) -> None:
        if (
            not self.layer_bytes
            or len(self.layer_bytes) != len(self.transfer_service_ns)
            or any(value <= 0 for value in self.layer_bytes)
            or any(value <= 0 for value in self.transfer_service_ns)
            or self.initial_compute_ns < 0
            or self.inter_layer_compute_ns <= 0
        ):
            raise ValueError("layer acquisition model has invalid service geometry")

    def analyze_admission(
        self, *, ready_prefix_layers: int
    ) -> "LayerAcquisitionFeasibility":
        """Test unresolved jobs from the forward's admission time origin."""

        return self._analyze(
            ready_prefix_layers=ready_prefix_layers,
            deadline_ns=lambda layer: (
                self.initial_compute_ns + layer * self.inter_layer_compute_ns
            ),
        )

    def admission_jobs(self) -> tuple[AcquisitionJob, ...]:
        """Project every layer onto the forward's admission-time EDF axis."""

        return tuple(
            AcquisitionJob(
                job_id=layer,
                payload_bytes=payload_bytes,
                service_ns=self.transfer_service_ns[layer],
                deadline_ns=(
                    self.initial_compute_ns + layer * self.inter_layer_compute_ns
                ),
            )
            for layer, payload_bytes in enumerate(self.layer_bytes)
        )

    def analyze_after_attention(
        self,
        *,
        completed_layer: int,
        ready_prefix_layers: int | None = None,
    ) -> "LayerAcquisitionFeasibility":
        """Test suffix feasibility from one completed attention arrival."""

        layer_count = len(self.layer_bytes)
        if not 0 <= completed_layer < layer_count:
            raise ValueError("completed attention layer is outside the model")
        ready = (
            completed_layer + 1 if ready_prefix_layers is None else ready_prefix_layers
        )
        if not completed_layer + 1 <= ready <= layer_count:
            raise ValueError("ready prefix precedes the completed attention layer")
        return self._analyze(
            ready_prefix_layers=ready,
            deadline_ns=lambda layer: (
                (layer - completed_layer) * self.inter_layer_compute_ns
            ),
        )

    def minimum_admission_ready_prefix(self) -> int:
        """Return the smallest completed prefix that makes the suffix feasible.

        This is a proof result, not an admission heuristic.  A backend may submit
        more work, but a framework need not release the numerical forward before
        this prefix is actually ready.
        """

        for ready_prefix in range(len(self.layer_bytes) + 1):
            if self.analyze_admission(ready_prefix_layers=ready_prefix).feasible:
                return ready_prefix
        raise RuntimeError("a fully ready acquisition model must be feasible")

    def compile_after_attention_frontier(self) -> "LayerAcquisitionFrontier":
        """Compile every suffix's first EDF miss once for a forward.

        Transformer-layer deadlines are strictly increasing and every job is
        simultaneously released, so EDF order is the layer order.  Prefix sums
        reproduce :meth:`analyze_after_attention` exactly without rebuilding
        and sorting ``AcquisitionJob`` objects at every layer arrival.
        """

        layer_count = len(self.layer_bytes)
        feasible_ends: list[int] = []
        for completed_layer in range(layer_count):
            elapsed_ns = 0
            feasible_end = layer_count
            for layer in range(completed_layer + 1, layer_count):
                elapsed_ns += self.transfer_service_ns[layer]
                deadline_ns = (layer - completed_layer) * self.inter_layer_compute_ns
                if elapsed_ns > deadline_ns:
                    feasible_end = layer
                    break
            feasible_ends.append(feasible_end)
        return LayerAcquisitionFrontier(tuple(feasible_ends))

    def _analyze(
        self,
        *,
        ready_prefix_layers: int,
        deadline_ns: Callable[[int], int],
    ) -> "LayerAcquisitionFeasibility":
        layer_count = len(self.layer_bytes)
        if not 0 <= ready_prefix_layers <= layer_count:
            raise ValueError("ready layer prefix is outside the acquisition model")
        admission_jobs = self.admission_jobs()
        schedule = schedule_acquisition_jobs(
            replace(admission_jobs[layer], deadline_ns=deadline_ns(layer))
            for layer in range(ready_prefix_layers, layer_count)
        )
        return LayerAcquisitionFeasibility(
            ready_prefix_layers=ready_prefix_layers,
            layer_count=layer_count,
            schedule=schedule,
        )


@dataclass(frozen=True, slots=True)
class LayerAcquisitionFeasibility:
    """Layer-indexed view of a tier-neutral EDF schedule."""

    ready_prefix_layers: int
    layer_count: int
    schedule: AcquisitionSchedule

    def __post_init__(self) -> None:
        if (
            self.layer_count <= 0
            or not 0 <= self.ready_prefix_layers <= self.layer_count
        ):
            raise ValueError("layer acquisition feasibility geometry is invalid")
        expected = set(range(self.ready_prefix_layers, self.layer_count))
        if set(self.schedule.ordered_job_ids) != expected:
            raise ValueError("EDF schedule does not cover the unresolved layer suffix")

    @property
    def feasible(self) -> bool:
        return self.schedule.feasible

    @property
    def first_missed_layer(self) -> int | None:
        return self.schedule.first_missed_job_id

    @property
    def required_initial_slack_ns(self) -> int:
        return self.schedule.required_initial_slack_ns

    @property
    def cumulative_completion_ns(self) -> tuple[int, ...]:
        return self.schedule.completion_ns

    @property
    def deadlines_ns(self) -> tuple[int, ...]:
        return self.schedule.deadlines_ns


@dataclass(frozen=True, slots=True)
class LayerAcquisitionFrontier:
    """O(1) lookup table for a frozen layer-acquisition service model.

    Entry ``i`` is the exclusive ready prefix that can be published after
    attention layer ``i``.  A value equal to the model layer count means that
    the complete suffix is feasible; otherwise the value is the first missed
    layer and must remain demand-driven.
    """

    feasible_end_by_completed_layer: tuple[int, ...]

    def __post_init__(self) -> None:
        layer_count = len(self.feasible_end_by_completed_layer)
        if layer_count == 0 or any(
            not completed_layer + 1 <= feasible_end <= layer_count
            for completed_layer, feasible_end in enumerate(
                self.feasible_end_by_completed_layer
            )
        ):
            raise ValueError("layer acquisition frontier has invalid geometry")

    @property
    def layer_count(self) -> int:
        return len(self.feasible_end_by_completed_layer)

    def feasible_end_after_attention(self, completed_layer: int) -> int:
        if not 0 <= completed_layer < self.layer_count:
            raise ValueError("completed layer is outside the acquisition frontier")
        return self.feasible_end_by_completed_layer[completed_layer]
