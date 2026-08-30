"""Framework-neutral analysis for indexed host-to-device transfers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import ceil, isfinite
from typing import Any, Iterable

from .abi import u32, u64


@dataclass(frozen=True)
class IndexedHostResource:
    """One pinned-host/HBM row resource with explicit packed geometry.

    ``source_tensor`` owns the pinned backing allocation.  Its first byte for
    this logical layer is ``source_offset_bytes`` into every physical row.
    ``destination_tensor`` names the exact framework view consumed by the
    numerical kernel.  Keeping row bytes separate from row strides is
    essential for vLLM's packed, cross-layer block allocation.
    """

    source_tensor: Any
    destination_tensor: Any
    source_offset_bytes: int
    row_bytes: int
    source_stride_bytes: int
    destination_stride_bytes: int
    source_rows: int
    destination_rows: int

    def __post_init__(self) -> None:
        if self.source_offset_bytes < 0:
            raise ValueError("indexed host source offset cannot be negative")
        geometry = (
            self.row_bytes,
            self.source_stride_bytes,
            self.destination_stride_bytes,
            self.source_rows,
            self.destination_rows,
        )
        if min(geometry) <= 0:
            raise ValueError("indexed host resources require positive geometry")
        if self.row_bytes > min(
            self.source_stride_bytes, self.destination_stride_bytes
        ):
            raise ValueError("indexed host row bytes exceed the physical stride")
        if self.source_offset_bytes + self.row_bytes > self.source_stride_bytes:
            raise ValueError("indexed host layer exceeds its packed source row")
        if max(geometry) >= 1 << 32:
            raise ValueError("indexed host resource geometry exceeds uint32")


@dataclass(frozen=True, slots=True)
class AcquisitionGroup:
    """One shared transfer/readiness owner in a retained device index map."""

    index_offset: int
    row_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "index_offset", u32(self.index_offset, "transfer index offset")
        )
        object.__setattr__(
            self, "row_count", u32(self.row_count, "transfer row count", positive=True)
        )

    @property
    def index_end(self) -> int:
        return self.index_offset + self.row_count


@dataclass(frozen=True, slots=True)
class AcquisitionSlice:
    """The exact rows one numerical work item consumes from a group."""

    group_index: int
    row_offset: int
    row_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "group_index", u32(self.group_index, "transfer group index")
        )
        object.__setattr__(self, "row_offset", u32(self.row_offset, "work row offset"))
        object.__setattr__(
            self, "row_count", u32(self.row_count, "work row count", positive=True)
        )

    @property
    def row_end(self) -> int:
        return self.row_offset + self.row_count


@dataclass(frozen=True, slots=True)
class AcquisitionTopology:
    """Framework-neutral exact mapping from work to acquisition groups.

    Acquisition groups own transfer, readiness, and resource accounting;
    slices retain the exact numerical subset consumed by each work item. A
    group may fan out to many work items, and a work item may reference many
    groups. Empty slice tuples are the canonical direct/resident form.
    """

    index_count: int
    groups: tuple[AcquisitionGroup, ...]
    dependencies_by_work: tuple[tuple[AcquisitionSlice, ...], ...]
    _group_fanout: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "index_count", u32(self.index_count, "index count", positive=True)
        )
        if not self.dependencies_by_work:
            raise ValueError("indexed transfer topology has no work")
        ordered = sorted(enumerate(self.groups), key=lambda item: item[1].index_offset)
        previous_end = 0
        for position, (_group_index, group) in enumerate(ordered):
            if group.index_end > self.index_count:
                raise ValueError("indexed transfer group exceeds its retained map")
            if position and group.index_offset < previous_end:
                raise ValueError("indexed transfer groups overlap")
            previous_end = group.index_end
        fanout = [0] * len(self.groups)
        for work in self.dependencies_by_work:
            if len({dependency.group_index for dependency in work}) != len(work):
                raise ValueError("one work item repeats an indexed transfer group")
            for dependency in work:
                if dependency.group_index >= len(self.groups):
                    raise ValueError("work dependency names an invalid transfer group")
                if dependency.row_end > self.groups[dependency.group_index].row_count:
                    raise ValueError("work dependency exceeds its transfer group")
                fanout[dependency.group_index] += 1
        if any(count == 0 for count in fanout):
            raise ValueError("indexed transfer group has no numerical consumer")
        object.__setattr__(self, "_group_fanout", tuple(fanout))

    @property
    def work_count(self) -> int:
        return len(self.dependencies_by_work)

    @property
    def direct_work_count(self) -> int:
        return sum(not dependencies for dependencies in self.dependencies_by_work)

    @property
    def max_group_fanout(self) -> int:
        return max(self._group_fanout, default=1)


@dataclass(frozen=True, slots=True)
class IndexedTensorLane:
    """One tensor lane expanded over every indexed transfer group."""

    source_address: int
    staging_address: int
    element_bytes: int
    source_stride_bytes: int
    staging_stride_bytes: int
    source_index_limit: int
    staging_index_limit: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_address",
            u64(self.source_address, "indexed source address", positive=True),
        )
        object.__setattr__(
            self,
            "staging_address",
            u64(self.staging_address, "indexed staging address", positive=True),
        )
        for name in (
            "element_bytes",
            "source_stride_bytes",
            "staging_stride_bytes",
            "source_index_limit",
            "staging_index_limit",
        ):
            object.__setattr__(
                self,
                name,
                u32(getattr(self, name), f"indexed lane {name}", positive=True),
            )
        if self.element_bytes > min(
            self.source_stride_bytes, self.staging_stride_bytes
        ):
            raise ValueError("indexed lane element bytes exceed its row stride")


@dataclass(frozen=True)
class ContiguousPairRun:
    """A maximal run contiguous in both the source and destination."""

    source_first: int
    destination_first: int
    row_count: int

    def bytes(self, row_bytes: int) -> int:
        if row_bytes <= 0:
            raise ValueError("indexed-transfer row bytes must be positive")
        return self.row_count * row_bytes


@dataclass(frozen=True)
class IndexedPairLayout:
    """Exact run decomposition of one paired source/destination index map."""

    row_count: int
    runs: tuple[ContiguousPairRun, ...]

    @property
    def maximum_run_rows(self) -> int:
        return max((run.row_count for run in self.runs), default=0)

    def eligible_rows(self, *, row_bytes: int, minimum_copy_bytes: int) -> int:
        if row_bytes <= 0 or minimum_copy_bytes <= 0:
            raise ValueError("copy-engine thresholds must be positive")
        return sum(
            run.row_count
            for run in self.runs
            if run.bytes(row_bytes) >= minimum_copy_bytes
        )

    def eligible_runs(self, *, row_bytes: int, minimum_copy_bytes: int) -> int:
        if row_bytes <= 0 or minimum_copy_bytes <= 0:
            raise ValueError("copy-engine thresholds must be positive")
        return sum(run.bytes(row_bytes) >= minimum_copy_bytes for run in self.runs)


@dataclass(frozen=True)
class IndexedDependencyRun:
    """One contiguous transfer run shared by an exact set of work items."""

    pair_offset: int
    source_first: int
    destination_first: int
    row_count: int
    work_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if min(self.pair_offset, self.source_first, self.destination_first) < 0:
            raise ValueError("indexed dependency run offsets cannot be negative")
        if self.row_count <= 0 or not self.work_ids:
            raise ValueError("indexed dependency runs require rows and consumers")
        if tuple(sorted(set(self.work_ids))) != self.work_ids:
            raise ValueError(
                "indexed dependency run work IDs must be unique and sorted"
            )


@dataclass(frozen=True)
class IndexedDependencyLayout:
    """Exact transfer objects and per-work dependencies for one layer."""

    source_indices: tuple[int, ...]
    destination_indices: tuple[int, ...]
    runs: tuple[IndexedDependencyRun, ...]
    run_indices_by_work: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if len(self.source_indices) != len(self.destination_indices):
            raise ValueError("indexed dependency source/destination maps disagree")
        if len(set(self.destination_indices)) != len(self.destination_indices):
            raise ValueError("indexed dependency destinations must be unique")
        covered = sum(run.row_count for run in self.runs)
        if covered != len(self.source_indices):
            raise ValueError("indexed dependency runs do not cover every transfer")
        for run_index, run in enumerate(self.runs):
            if run.pair_offset + run.row_count > len(self.source_indices):
                raise ValueError("indexed dependency run exceeds its pair map")
            for work_id in run.work_ids:
                if (
                    work_id >= len(self.run_indices_by_work)
                    or run_index not in self.run_indices_by_work[work_id]
                ):
                    raise ValueError("indexed dependency reverse map is incomplete")


@dataclass(frozen=True)
class IndexedMoverPlan:
    """Disjoint, exact partition between copy-engine runs and SM rows."""

    row_count: int
    copy_runs: tuple[ContiguousPairRun, ...]
    sm_source_indices: tuple[int, ...]
    sm_destination_indices: tuple[int, ...]
    predicted_sm_ns: int
    predicted_selected_ns: int | None
    selection_reason: str

    def __post_init__(self) -> None:
        copy_rows = sum(run.row_count for run in self.copy_runs)
        if (
            self.row_count <= 0
            or copy_rows + len(self.sm_source_indices) != self.row_count
        ):
            raise ValueError("indexed mover plan does not cover every input row")
        if len(self.sm_source_indices) != len(self.sm_destination_indices):
            raise ValueError("indexed mover SM maps disagree")
        if len(set(self.sm_destination_indices)) != len(self.sm_destination_indices):
            raise ValueError("indexed mover SM destinations overlap")
        if self.predicted_sm_ns <= 0 or (
            self.predicted_selected_ns is not None and self.predicted_selected_ns <= 0
        ):
            raise ValueError("indexed mover predictions must be positive")
        if self.selection_reason not in {
            "forced_sm",
            "forced_copy_engine",
            "calibration_probe_sm",
            "calibration_probe_copy",
            "uncalibrated_sm_reference",
            "uncalibrated_copy_engine",
            "uncalibrated_transfer_scale",
            "insufficient_gain",
            "service_cost",
        }:
            raise ValueError("indexed mover selection reason is invalid")

    @property
    def copy_row_count(self) -> int:
        return sum(run.row_count for run in self.copy_runs)

    @property
    def sm_row_count(self) -> int:
        return len(self.sm_source_indices)

    @property
    def kind(self) -> str:
        if not self.copy_runs:
            return "sm"
        if not self.sm_source_indices:
            return "copy_engine"
        return "hybrid"

    def copy_bytes(self, row_bytes: int) -> int:
        if row_bytes <= 0:
            raise ValueError("indexed mover row bytes must be positive")
        return self.copy_row_count * row_bytes


@dataclass(frozen=True)
class IndexedMoverServiceModel:
    """Deployment-calibrated service costs for one shared host link.

    Copy-engine DMA and SM gathers are different issuers, but both consume the
    same host-to-GPU link. SM gather also consumes execution capacity needed by
    the numerical kernel. The model therefore serializes SM and copy link
    service, serializes SM gather and compute, and overlaps copy with compute
    only after deployment observations calibrate that overlap. Copy CUDA-event
    service already contains descriptor-submission starvation; scheduler issue
    demand is a second resource bound, not an additive copy of the same time.

    One instance is either an externally supplied deployment curve
    (``calibration_scale_bucket is None``) or an online curve for exactly one
    power-of-two total-wave bucket. Auto selection requires enough SM and copy
    samples at that scale. Forced policies remain diagnostic arms and may use
    uncalibrated estimates without authorizing ``auto``.
    """

    sm_bandwidth_bytes_per_second: int
    copy_bandwidth_bytes_per_second: int | None = None
    copy_operation_ns: int | None = None
    hybrid_join_ns: int = 0
    minimum_gain: float = 1.03
    sm_samples: int = 0
    copy_samples: int = 0
    minimum_calibration_samples: int = 3
    calibration_scale_bucket: int | None = None
    copy_compute_overlap_efficiency: float = 0.0
    overlap_samples: int = 0

    def __post_init__(self) -> None:
        if self.sm_bandwidth_bytes_per_second <= 0:
            raise ValueError("SM mover bandwidth must be positive")
        if (self.copy_bandwidth_bytes_per_second is None) != (
            self.copy_operation_ns is None
        ):
            raise ValueError(
                "copy-engine bandwidth and operation cost must be calibrated together"
            )
        if self.copy_bandwidth_bytes_per_second is not None and (
            self.copy_bandwidth_bytes_per_second <= 0
            or self.copy_operation_ns is None
            or self.copy_operation_ns < 0
        ):
            raise ValueError("copy-engine mover calibration is invalid")
        if (
            self.hybrid_join_ns < 0
            or not isfinite(self.minimum_gain)
            or self.minimum_gain < 1.0
        ):
            raise ValueError("mover join cost and minimum gain are invalid")
        if min(self.sm_samples, self.copy_samples, self.overlap_samples) < 0:
            raise ValueError("mover calibration sample counts cannot be negative")
        if self.minimum_calibration_samples <= 0:
            raise ValueError("mover minimum calibration samples must be positive")
        if self.calibration_scale_bucket is not None and (
            self.calibration_scale_bucket < 0
        ):
            raise ValueError("mover calibration scale bucket cannot be negative")
        if self.copy_samples != 0 and not self.copy_estimate_available:
            raise ValueError("copy-engine samples require a complete calibration")
        if not isfinite(self.copy_compute_overlap_efficiency) or not (
            0.0 <= self.copy_compute_overlap_efficiency <= 1.0
        ):
            raise ValueError("copy/compute overlap efficiency must be in [0, 1]")

    @property
    def copy_estimate_available(self) -> bool:
        return self.copy_bandwidth_bytes_per_second is not None

    @property
    def sm_calibrated(self) -> bool:
        return self.sm_samples >= self.minimum_calibration_samples

    @property
    def copy_calibrated(self) -> bool:
        return self.copy_estimate_available and (
            self.copy_samples >= self.minimum_calibration_samples
        )

    @property
    def overlap_calibrated(self) -> bool:
        return self.overlap_samples >= self.minimum_calibration_samples

    @property
    def effective_copy_compute_overlap(self) -> float:
        return self.copy_compute_overlap_efficiency if self.overlap_calibrated else 0.0

    @staticmethod
    def _scale_bucket(transfer_bytes: int) -> int:
        if transfer_bytes <= 0:
            raise ValueError("mover calibration bytes must be positive")
        return transfer_bytes.bit_length() - 1

    def _observation_scale_bucket(self, transfer_bytes: int) -> int:
        bucket = self._scale_bucket(transfer_bytes)
        if (
            self.calibration_scale_bucket is not None
            and self.calibration_scale_bucket != bucket
        ):
            raise ValueError("one mover service model cannot mix transfer-size buckets")
        return bucket

    def supports_transfer_scale(self, transfer_bytes: int) -> bool:
        """Return whether this curve covers one total-wave size class."""

        bucket = self._scale_bucket(transfer_bytes)
        return self.calibration_scale_bucket in {None, bucket}

    @staticmethod
    def _bounded_ewma(
        previous: int,
        observed: int,
        *,
        alpha: float,
        maximum_step_ratio: float,
    ) -> int:
        if min(previous, observed) <= 0:
            raise ValueError("mover observations must be positive")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("mover calibration alpha must be in (0, 1]")
        if maximum_step_ratio < 1.0:
            raise ValueError("mover calibration step ratio must be at least one")
        lower = max(1, int(previous / maximum_step_ratio))
        upper = max(lower, int(previous * maximum_step_ratio))
        bounded = min(max(observed, lower), upper)
        return max(1, round((1.0 - alpha) * previous + alpha * bounded))

    def with_sm_observation(
        self,
        *,
        transfer_bytes: int,
        service_scale_bytes: int | None = None,
        elapsed_ns: int,
        alpha: float = 0.25,
        minimum_sample_bytes: int = 64 * 1024,
        maximum_step_ratio: float = 2.0,
    ) -> "IndexedMoverServiceModel":
        """Calibrate effective SM-mover service from one completed wave.

        The event interval includes launch and memory-system cost, so the
        resulting rate is an effective service curve at the observed wave
        geometry rather than a nominal PCIe bandwidth claim. Small waves are
        ignored because timer and launch noise dominate them.
        """

        if min(transfer_bytes, elapsed_ns, minimum_sample_bytes) <= 0:
            raise ValueError("SM mover calibration geometry must be positive")
        if transfer_bytes < minimum_sample_bytes:
            return self
        scale_bytes = (
            transfer_bytes if service_scale_bytes is None else service_scale_bytes
        )
        if scale_bytes < transfer_bytes:
            raise ValueError("SM mover service scale is below its physical transfer")
        bucket = self._observation_scale_bucket(scale_bytes)
        observed = max(1, transfer_bytes * 1_000_000_000 // elapsed_ns)
        bandwidth = (
            observed
            if self.sm_samples == 0
            else self._bounded_ewma(
                self.sm_bandwidth_bytes_per_second,
                observed,
                alpha=alpha,
                maximum_step_ratio=maximum_step_ratio,
            )
        )
        return replace(
            self,
            sm_bandwidth_bytes_per_second=bandwidth,
            sm_samples=self.sm_samples + 1,
            calibration_scale_bucket=bucket,
        )

    def with_copy_observation(
        self,
        *,
        transfer_bytes: int,
        service_scale_bytes: int | None = None,
        elapsed_ns: int,
        operation_count: int,
        issue_cpu_ns: int,
        alpha: float = 0.25,
        minimum_sample_bytes: int = 64 * 1024,
        maximum_step_ratio: float = 2.0,
    ) -> "IndexedMoverServiceModel":
        """Calibrate end-to-end copy service and descriptor issue cost.

        The start event executes before the host finishes submitting a batch,
        so its stream-visible interval includes both DMA and any device-idle
        submission gap. Subtracting the complete CPU interval is unsound when
        descriptor issue pipelines with earlier copies: it can manufacture a
        bandwidth above the physical link. Keep the observed end-to-end rate
        and model scheduler-thread issue separately and conservatively.
        """

        if (
            min(
                transfer_bytes,
                elapsed_ns,
                operation_count,
                minimum_sample_bytes,
            )
            <= 0
            or issue_cpu_ns < 0
        ):
            raise ValueError("copy-engine calibration geometry must be positive")
        if transfer_bytes < minimum_sample_bytes:
            return self
        scale_bytes = (
            transfer_bytes if service_scale_bytes is None else service_scale_bytes
        )
        if scale_bytes < transfer_bytes:
            raise ValueError("copy-engine service scale is below its physical transfer")
        bucket = self._observation_scale_bucket(scale_bytes)
        observed_bandwidth = max(1, transfer_bytes * 1_000_000_000 // elapsed_ns)
        observed_operation_ns = max(0, issue_cpu_ns // operation_count)
        if self.copy_samples == 0:
            # A prior is not a measurement. Once the minimum sample geometry
            # is met, the first deployment observation establishes the curve;
            # later samples are bounded to reject transient outliers.
            bandwidth = observed_bandwidth
            operation_ns = observed_operation_ns
        else:
            if not self.copy_estimate_available:
                raise RuntimeError("copy samples exist without a service estimate")
            assert self.copy_operation_ns is not None
            bandwidth = self._bounded_ewma(
                self.copy_bandwidth_bytes_per_second,
                observed_bandwidth,
                alpha=alpha,
                maximum_step_ratio=maximum_step_ratio,
            )
            operation_ns = (
                observed_operation_ns
                if self.copy_operation_ns == 0
                else self._bounded_ewma(
                    self.copy_operation_ns,
                    max(1, observed_operation_ns),
                    alpha=alpha,
                    maximum_step_ratio=maximum_step_ratio,
                )
            )
        return replace(
            self,
            copy_bandwidth_bytes_per_second=bandwidth,
            copy_operation_ns=operation_ns,
            copy_samples=self.copy_samples + 1,
            calibration_scale_bucket=bucket,
        )

    def with_copy_compute_overlap_observation(
        self,
        *,
        transfer_bytes: int,
        isolated_copy_ns: int,
        isolated_compute_ns: int,
        concurrent_ns: int,
        alpha: float = 0.25,
    ) -> "IndexedMoverServiceModel":
        """Update overlap from isolated and concurrent deployment timings.

        Efficiency is the fraction of the ideal overlap window actually saved:
        ``(copy + compute - concurrent) / min(copy, compute)``. Zero denotes
        serial execution and one denotes perfect overlap. A concurrent sample
        faster than the physical ``max(copy, compute)`` lower bound is rejected;
        contention slower than serial execution conservatively records zero.
        """

        if (
            min(transfer_bytes, isolated_copy_ns, isolated_compute_ns, concurrent_ns)
            <= 0
        ):
            raise ValueError("mover overlap observations must be positive")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("mover overlap alpha must be in (0, 1]")
        bucket = self._observation_scale_bucket(transfer_bytes)
        ideal_lower_bound = max(isolated_copy_ns, isolated_compute_ns)
        if concurrent_ns < ideal_lower_bound:
            raise ValueError("copy/compute overlap sample violates its lower bound")
        overlap_window = min(isolated_copy_ns, isolated_compute_ns)
        saved_ns = max(0, isolated_copy_ns + isolated_compute_ns - concurrent_ns)
        observed = min(1.0, saved_ns / overlap_window)
        efficiency = (
            observed
            if self.overlap_samples == 0
            else (1.0 - alpha) * self.copy_compute_overlap_efficiency + alpha * observed
        )
        return replace(
            self,
            copy_compute_overlap_efficiency=efficiency,
            overlap_samples=self.overlap_samples + 1,
            calibration_scale_bucket=bucket,
        )

    @staticmethod
    def _transfer_ns(transfer_bytes: int, bandwidth_bytes_per_second: int) -> int:
        if transfer_bytes <= 0 or bandwidth_bytes_per_second <= 0:
            raise ValueError("mover service geometry must be positive")
        return max(
            1,
            (transfer_bytes * 1_000_000_000 + bandwidth_bytes_per_second - 1)
            // bandwidth_bytes_per_second,
        )

    def sm_only_ns(self, transfer_bytes: int) -> int:
        return self._transfer_ns(transfer_bytes, self.sm_bandwidth_bytes_per_second)

    def _copy_service_ns(
        self, *, transfer_bytes: int, operation_count: int
    ) -> int | None:
        """Return copy completion without double charging descriptor issue.

        CUDA-event elapsed time is an end-to-end copy-stream observation: when
        host submission starves the stream, that delay is already reflected in
        effective bandwidth. Independently measured scheduler demand can still
        dominate a more fragmented candidate, so the two resource bounds are
        combined with ``max`` rather than added.
        """

        if transfer_bytes <= 0 or operation_count <= 0:
            raise ValueError("copy service geometry must be positive")
        if not self.copy_estimate_available:
            return None
        assert self.copy_bandwidth_bytes_per_second is not None
        assert self.copy_operation_ns is not None
        transfer_ns = self._transfer_ns(
            transfer_bytes, self.copy_bandwidth_bytes_per_second
        )
        issue_ns = operation_count * self.copy_operation_ns
        return max(transfer_ns, issue_ns)

    def _copy_compute_ns(self, copy_ns: int, compute_ns: int) -> int:
        if copy_ns <= 0 or compute_ns < 0:
            raise ValueError("copy/compute service geometry is invalid")
        if compute_ns == 0:
            return copy_ns
        saved_ns = int(self.effective_copy_compute_overlap * min(copy_ns, compute_ns))
        return copy_ns + compute_ns - saved_ns

    def meets_selection_margin(self, reference_ns: int, candidate_ns: int) -> bool:
        """Apply ``minimum_gain`` after optimization, in units of time.

        This is a multiplicative policy safety margin, not a transfer-size
        cutoff: it never changes candidate costs or the candidate optimum.
        """

        if min(reference_ns, candidate_ns) <= 0:
            raise ValueError("mover selection predictions must be positive")
        return reference_ns >= ceil(candidate_ns * self.minimum_gain)

    def candidate_ns(
        self,
        *,
        total_rows: int,
        copy_rows: int,
        copy_run_count: int,
        row_bytes: int,
        copy_operations_per_run: int,
        overlap_compute_ns: int = 0,
    ) -> int | None:
        if min(total_rows, row_bytes, copy_operations_per_run) <= 0:
            raise ValueError("indexed mover candidate geometry must be positive")
        if overlap_compute_ns < 0:
            raise ValueError("indexed mover overlap compute cannot be negative")
        if not 0 <= copy_rows <= total_rows or copy_run_count < 0:
            raise ValueError("indexed mover candidate partition is invalid")
        if (copy_rows == 0) != (copy_run_count == 0):
            raise ValueError("copy rows and runs must both be empty or non-empty")
        if copy_rows == 0:
            # Conservatively treat SM gather and numerical work as sharing one
            # execution resource. This is the no-copy reference makespan, not
            # merely the mover's isolated service time.
            return self.sm_only_ns(total_rows * row_bytes) + overlap_compute_ns
        if not self.copy_estimate_available:
            return None
        copy_service_ns = self._copy_service_ns(
            transfer_bytes=copy_rows * row_bytes,
            operation_count=copy_run_count * copy_operations_per_run,
        )
        if copy_service_ns is None:  # pragma: no cover - availability invariant
            raise RuntimeError("copy estimate disappeared during candidate analysis")
        sm_rows = total_rows - copy_rows
        if sm_rows == 0:
            return self._copy_compute_ns(copy_service_ns, overlap_compute_ns)
        # SM and copy consume one host link and are therefore serialized. SM
        # gather also serializes with numerical compute. Only the copy/compute
        # pair may overlap, and only by its measured efficiency.
        return (
            self._transfer_ns(sm_rows * row_bytes, self.sm_bandwidth_bytes_per_second)
            + self._copy_compute_ns(copy_service_ns, overlap_compute_ns)
            + self.hybrid_join_ns
        )

    def ideal_copy_can_qualify(
        self,
        *,
        total_rows: int,
        row_bytes: int,
        copy_operations_per_run: int,
        overlap_compute_ns: int = 0,
        service_scale_bytes: int | None = None,
    ) -> bool:
        """Whether an optimistic mover lower bound can meet the gain contract.

        Before downloading run descriptors, pretend every byte uses the faster
        calibrated link rate, pays no descriptor or join cost, and enjoys the
        measured copy/compute overlap. This is deliberately more optimistic
        than every realizable SM/copy partition. If even this lower bound cannot
        beat the SM reference by ``minimum_gain``, layout analysis cannot change
        the decision and safely stays off the scheduler hot path.
        """

        if not (
            self.sm_calibrated
            and self.copy_calibrated
        ):
            return False
        if service_scale_bytes is not None and not self.supports_transfer_scale(
            service_scale_bytes
        ):
            return False
        predicted_sm_ns = self.candidate_ns(
            total_rows=total_rows,
            copy_rows=0,
            copy_run_count=0,
            row_bytes=row_bytes,
            copy_operations_per_run=copy_operations_per_run,
            overlap_compute_ns=overlap_compute_ns,
        )
        assert self.copy_bandwidth_bytes_per_second is not None
        ideal_link_ns = self._transfer_ns(
            total_rows * row_bytes,
            max(
                self.sm_bandwidth_bytes_per_second,
                self.copy_bandwidth_bytes_per_second,
            ),
        )
        ideal_copy_ns = self._copy_compute_ns(ideal_link_ns, overlap_compute_ns)
        return (
            ideal_copy_ns is not None
            and predicted_sm_ns is not None
            and (self.meets_selection_margin(predicted_sm_ns, ideal_copy_ns))
        )


@dataclass(frozen=True)
class IndexedMoverSelection:
    selected_run_indices: tuple[int, ...]
    predicted_sm_ns: int
    predicted_selected_ns: int | None
    reason: str


def select_indexed_mover_candidates(
    *,
    total_rows: int,
    total_run_count: int,
    candidate_runs: Iterable[tuple[int, int]],
    row_bytes: int,
    copy_operations_per_run: int,
    maximum_copy_runs: int,
    service_model: IndexedMoverServiceModel,
    policy: str = "auto",
    overlap_compute_ns: int = 0,
    service_scale_bytes: int | None = None,
) -> IndexedMoverSelection:
    """Select from the longest exact runs without materializing every run.

    ``candidate_runs`` contains ``(run_index, row_count)`` pairs. For auto and
    probe policies it must contain the longest ``maximum_copy_runs`` runs.
    The bounded longest-run prefixes are a candidate family, not a claim of a
    globally optimal subset: each selected exact partition must independently
    beat the all-SM reference by the calibrated policy margin. This keeps the
    scheduler cost bounded while avoiding the exponential subset problem.
    ``service_scale_bytes`` is the physical wave geometry used to pick an online
    curve; it is intentionally distinct from aggregate lease bytes.
    Forced copy execution must provide the complete decomposition.
    """

    if (
        min(
            total_rows,
            total_run_count,
            row_bytes,
            copy_operations_per_run,
            maximum_copy_runs,
        )
        <= 0
    ):
        raise ValueError("indexed mover service geometry must be positive")
    if overlap_compute_ns < 0:
        raise ValueError("indexed mover overlap compute cannot be negative")
    if service_scale_bytes is not None and service_scale_bytes <= 0:
        raise ValueError("indexed mover service scale must be positive")
    if policy not in {"auto", "sm", "copy_engine", "probe_copy"}:
        raise ValueError(
            "indexed mover policy must be auto, sm, copy_engine, or probe_copy"
        )
    candidates = tuple((int(index), int(rows)) for index, rows in candidate_runs)
    candidate_rows = sum(rows for _, rows in candidates)
    if (
        total_run_count > total_rows
        or any(
            index < 0 or index >= total_run_count or rows <= 0
            for index, rows in candidates
        )
        or len({index for index, _ in candidates}) != len(candidates)
        or candidate_rows > total_rows
        or (len(candidates) == total_run_count and candidate_rows != total_rows)
        or (len(candidates) < total_run_count and candidate_rows >= total_rows)
    ):
        raise ValueError("indexed mover candidate runs are invalid")
    predicted_sm_ns = service_model.candidate_ns(
        total_rows=total_rows,
        copy_rows=0,
        copy_run_count=0,
        row_bytes=row_bytes,
        copy_operations_per_run=copy_operations_per_run,
        overlap_compute_ns=overlap_compute_ns,
    )
    if predicted_sm_ns is None:  # pragma: no cover - zero-copy is always modeled
        raise RuntimeError("SM-only mover produced no service estimate")
    if policy == "sm":
        return IndexedMoverSelection((), predicted_sm_ns, predicted_sm_ns, "forced_sm")

    ordered = tuple(
        index
        for index, _ in sorted(candidates, key=lambda item: (-item[1], item[0]))[
            :maximum_copy_runs
        ]
    )
    rows_by_index = dict(candidates)
    if policy == "copy_engine":
        if total_run_count > maximum_copy_runs:
            raise ValueError("copy-engine index map exceeds the operation bound")
        if (
            len(candidates) != total_run_count
            or sum(rows_by_index.values()) != total_rows
        ):
            raise ValueError("forced copy-engine plan requires every exact run")
        selected = tuple(sorted(rows_by_index))
        return IndexedMoverSelection(
            selected,
            predicted_sm_ns,
            service_model.candidate_ns(
                total_rows=total_rows,
                copy_rows=total_rows,
                copy_run_count=len(selected),
                row_bytes=row_bytes,
                copy_operations_per_run=copy_operations_per_run,
                overlap_compute_ns=overlap_compute_ns,
            ),
            "forced_copy_engine",
        )
    if policy == "probe_copy":
        if not ordered:
            raise RuntimeError("copy-engine calibration has no candidate run")
        required_candidates = min(total_run_count, maximum_copy_runs)
        if len(candidates) < required_candidates:
            raise ValueError(
                "indexed mover selection is missing longest-run candidates"
            )
        probe_rows = sum(rows_by_index[index] for index in ordered)
        return IndexedMoverSelection(
            tuple(sorted(ordered)),
            predicted_sm_ns,
            service_model.candidate_ns(
                total_rows=total_rows,
                copy_rows=probe_rows,
                copy_run_count=len(ordered),
                row_bytes=row_bytes,
                copy_operations_per_run=copy_operations_per_run,
                overlap_compute_ns=overlap_compute_ns,
            ),
            "calibration_probe_copy",
        )
    required_candidates = min(total_run_count, maximum_copy_runs)
    if len(candidates) < required_candidates:
        raise ValueError("indexed mover selection is missing longest-run candidates")
    if not service_model.sm_calibrated:
        return IndexedMoverSelection(
            (), predicted_sm_ns, predicted_sm_ns, "uncalibrated_sm_reference"
        )
    if service_scale_bytes is not None and not service_model.supports_transfer_scale(
        service_scale_bytes
    ):
        return IndexedMoverSelection(
            (), predicted_sm_ns, predicted_sm_ns, "uncalibrated_transfer_scale"
        )
    if not service_model.copy_calibrated:
        return IndexedMoverSelection(
            (),
            predicted_sm_ns,
            predicted_sm_ns,
            "uncalibrated_copy_engine",
        )
    selected_rows = 0
    prefix_indices: tuple[int, ...] = ()
    best_indices: tuple[int, ...] = ()
    best_ns = predicted_sm_ns
    for run_index in ordered:
        selected_rows += rows_by_index[run_index]
        prefix_indices = (*prefix_indices, run_index)
        candidate_ns = service_model.candidate_ns(
            total_rows=total_rows,
            copy_rows=selected_rows,
            copy_run_count=len(prefix_indices),
            row_bytes=row_bytes,
            copy_operations_per_run=copy_operations_per_run,
            overlap_compute_ns=overlap_compute_ns,
        )
        if candidate_ns is None:  # pragma: no cover - calibrated invariant
            raise RuntimeError("calibrated mover produced no service estimate")
        # A fixed hybrid join can make the first run lose even when a longer
        # prefix amortizes it, so every bounded prefix must be evaluated.
        if candidate_ns < best_ns:
            best_indices = prefix_indices
            best_ns = candidate_ns
    if not best_indices or not service_model.meets_selection_margin(
        predicted_sm_ns, best_ns
    ):
        return IndexedMoverSelection(
            (), predicted_sm_ns, predicted_sm_ns, "insufficient_gain"
        )
    return IndexedMoverSelection(
        tuple(sorted(best_indices)), predicted_sm_ns, best_ns, "service_cost"
    )


def select_indexed_mover_runs(
    layout: IndexedPairLayout,
    *,
    row_bytes: int,
    copy_operations_per_run: int,
    maximum_copy_runs: int,
    service_model: IndexedMoverServiceModel,
    policy: str = "auto",
    overlap_compute_ns: int = 0,
    service_scale_bytes: int | None = None,
) -> IndexedMoverSelection:
    """Choose the exact partition minimizing resource-aware stage makespan."""
    return select_indexed_mover_candidates(
        total_rows=layout.row_count,
        total_run_count=len(layout.runs),
        candidate_runs=tuple(
            (index, run.row_count) for index, run in enumerate(layout.runs)
        ),
        row_bytes=row_bytes,
        copy_operations_per_run=copy_operations_per_run,
        maximum_copy_runs=maximum_copy_runs,
        service_model=service_model,
        policy=policy,
        overlap_compute_ns=overlap_compute_ns,
        service_scale_bytes=service_scale_bytes,
    )


@dataclass(frozen=True)
class StridedCopyGroup:
    """One strided-row tensor pair that shares an exact index-run layout."""

    source_address: int
    destination_address: int
    source_rows: int
    destination_rows: int
    row_bytes: int
    source_stride_bytes: int
    destination_stride_bytes: int

    def __post_init__(self) -> None:
        if (
            min(
                self.source_address,
                self.destination_address,
                self.source_rows,
                self.destination_rows,
                self.row_bytes,
                self.source_stride_bytes,
                self.destination_stride_bytes,
            )
            <= 0
        ):
            raise ValueError("strided copy groups require positive geometry")
        if (
            min(self.source_stride_bytes, self.destination_stride_bytes)
            < self.row_bytes
        ):
            raise ValueError("strided copy pitches cannot be smaller than one row")
        if (
            max(
                self.source_rows,
                self.destination_rows,
                self.row_bytes,
                self.source_stride_bytes,
                self.destination_stride_bytes,
            )
            >= 1 << 32
        ):
            raise ValueError("strided copy group geometry exceeds uint32")
        if max(self.source_address, self.destination_address) >= 1 << 64:
            raise ValueError("strided copy group address exceeds uint64")


def analyze_index_pairs(
    source_indices: Iterable[int], destination_indices: Iterable[int]
) -> IndexedPairLayout:
    """Decompose an exact index map without reordering either side.

    Copy engines can replace an indexed gather only when both addresses advance
    by one row. Reordering is forbidden because destination ownership and
    duplicate detection belong to the framework producer.
    """

    source = tuple(int(value) for value in source_indices)
    destination = tuple(int(value) for value in destination_indices)
    if not source or len(source) != len(destination):
        raise ValueError("indexed-transfer maps must be non-empty and equal length")
    if min(source) < 0 or min(destination) < 0:
        raise ValueError("indexed-transfer indices cannot be negative")
    if len(set(destination)) != len(destination):
        raise ValueError("indexed-transfer destinations must be unique")

    runs: list[ContiguousPairRun] = []
    run_begin = 0
    for index in range(1, len(source)):
        if (
            source[index] != source[index - 1] + 1
            or destination[index] != destination[index - 1] + 1
        ):
            runs.append(
                ContiguousPairRun(
                    source[run_begin], destination[run_begin], index - run_begin
                )
            )
            run_begin = index
    runs.append(
        ContiguousPairRun(
            source[run_begin], destination[run_begin], len(source) - run_begin
        )
    )
    return IndexedPairLayout(len(source), tuple(runs))


def plan_indexed_dependencies(
    work_pairs: Iterable[Iterable[tuple[int, int]]],
) -> IndexedDependencyLayout:
    """Build maximal exact runs without widening any work dependency.

    A source/destination pair may be shared by multiple attention work items.
    Adjacent pairs are coalesced only when both indices are contiguous *and*
    the set of consuming work items is identical.  This prevents a transfer
    optimization from making one work item wait for unrelated bytes.
    """

    rows = tuple(
        tuple((int(source), int(destination)) for source, destination in pairs)
        for pairs in work_pairs
    )
    consumers_by_destination: dict[int, set[int]] = {}
    source_by_destination: dict[int, int] = {}
    for work_id, pairs in enumerate(rows):
        if len(set(pairs)) != len(pairs):
            raise ValueError("one work item contains duplicate indexed transfers")
        for source, destination in pairs:
            if min(source, destination) < 0:
                raise ValueError("indexed dependency indices cannot be negative")
            previous = source_by_destination.setdefault(destination, source)
            if previous != source:
                raise ValueError("one destination is bound to conflicting sources")
            consumers_by_destination.setdefault(destination, set()).add(work_id)

    ordered = tuple(
        (source_by_destination[destination], destination)
        for destination in sorted(source_by_destination)
    )
    if not ordered:
        return IndexedDependencyLayout((), (), (), tuple(() for _ in rows))
    source_indices = tuple(source for source, _ in ordered)
    destination_indices = tuple(destination for _, destination in ordered)
    runs: list[IndexedDependencyRun] = []
    run_begin = 0
    previous_consumers = tuple(sorted(consumers_by_destination[destination_indices[0]]))
    for index in range(1, len(ordered)):
        consumers = tuple(sorted(consumers_by_destination[destination_indices[index]]))
        if (
            source_indices[index] != source_indices[index - 1] + 1
            or destination_indices[index] != destination_indices[index - 1] + 1
            or consumers != previous_consumers
        ):
            runs.append(
                IndexedDependencyRun(
                    run_begin,
                    source_indices[run_begin],
                    destination_indices[run_begin],
                    index - run_begin,
                    previous_consumers,
                )
            )
            run_begin = index
            previous_consumers = consumers
    runs.append(
        IndexedDependencyRun(
            run_begin,
            source_indices[run_begin],
            destination_indices[run_begin],
            len(ordered) - run_begin,
            previous_consumers,
        )
    )
    by_work: list[list[int]] = [[] for _ in rows]
    for run_index, run in enumerate(runs):
        for work_id in run.work_ids:
            by_work[work_id].append(run_index)
    return IndexedDependencyLayout(
        source_indices,
        destination_indices,
        tuple(runs),
        tuple(tuple(indices) for indices in by_work),
    )


def plan_indexed_mover(
    source_indices: Iterable[int],
    destination_indices: Iterable[int],
    *,
    row_bytes: int,
    copy_operations_per_run: int,
    maximum_copy_runs: int,
    service_model: IndexedMoverServiceModel,
    policy: str = "auto",
    overlap_compute_ns: int = 0,
    service_scale_bytes: int | None = None,
) -> IndexedMoverPlan:
    """Partition one exact map without changing row or destination ownership."""

    source = tuple(int(value) for value in source_indices)
    destination = tuple(int(value) for value in destination_indices)
    layout = analyze_index_pairs(source, destination)
    selection = select_indexed_mover_runs(
        layout,
        row_bytes=row_bytes,
        copy_operations_per_run=copy_operations_per_run,
        maximum_copy_runs=maximum_copy_runs,
        service_model=service_model,
        policy=policy,
        overlap_compute_ns=overlap_compute_ns,
        service_scale_bytes=service_scale_bytes,
    )
    selected_indices = set(selection.selected_run_indices)

    copy_runs = tuple(
        run for index, run in enumerate(layout.runs) if index in selected_indices
    )
    sm_source: list[int] = []
    sm_destination: list[int] = []
    cursor = 0
    for index, run in enumerate(layout.runs):
        next_cursor = cursor + run.row_count
        if index not in selected_indices:
            sm_source.extend(source[cursor:next_cursor])
            sm_destination.extend(destination[cursor:next_cursor])
        cursor = next_cursor
    if cursor != layout.row_count:
        raise RuntimeError("indexed mover layout did not consume its input map")
    return IndexedMoverPlan(
        layout.row_count,
        copy_runs,
        tuple(sm_source),
        tuple(sm_destination),
        selection.predicted_sm_ns,
        selection.predicted_selected_ns,
        selection.reason,
    )
