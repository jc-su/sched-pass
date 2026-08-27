"""Framework-neutral analysis for indexed host-to-device transfers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
class IndexedTransferGroup:
    """One exact transport object as a slice of a retained device index map."""

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
class IndexedWorkDependency:
    """The exact rows one work item consumes within a transfer group."""

    group_index: int
    row_offset: int
    row_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "group_index", u32(self.group_index, "transfer group index")
        )
        object.__setattr__(
            self, "row_offset", u32(self.row_offset, "work row offset")
        )
        object.__setattr__(
            self, "row_count", u32(self.row_count, "work row count", positive=True)
        )

    @property
    def row_end(self) -> int:
        return self.row_offset + self.row_count


@dataclass(frozen=True, slots=True)
class IndexedTransferTopology:
    """Framework-neutral exact mapping from work to indexed transfer groups.

    Transfer groups own readiness granularity; work dependencies retain the
    exact numerical subset inside each group.  A group may serve many work
    items, and a work item may require many groups. Empty work dependencies
    are the canonical direct/resident representation.
    """

    index_count: int
    groups: tuple[IndexedTransferGroup, ...]
    dependencies_by_work: tuple[tuple[IndexedWorkDependency, ...], ...]
    _group_fanout: tuple[int, ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "index_count", u32(self.index_count, "index count", positive=True)
        )
        if not self.dependencies_by_work:
            raise ValueError("indexed transfer topology has no work")
        ordered = sorted(
            enumerate(self.groups), key=lambda item: item[1].index_offset
        )
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
                self, name, u32(getattr(self, name), f"indexed lane {name}", positive=True)
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
            raise ValueError("indexed dependency run work IDs must be unique and sorted")


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
        if self.row_count <= 0 or copy_rows + len(self.sm_source_indices) != self.row_count:
            raise ValueError("indexed mover plan does not cover every input row")
        if len(self.sm_source_indices) != len(self.sm_destination_indices):
            raise ValueError("indexed mover SM maps disagree")
        if len(set(self.sm_destination_indices)) != len(self.sm_destination_indices):
            raise ValueError("indexed mover SM destinations overlap")
        if self.predicted_sm_ns <= 0 or (
            self.predicted_selected_ns is not None
            and self.predicted_selected_ns <= 0
        ):
            raise ValueError("indexed mover predictions must be positive")
        if self.selection_reason not in {
            "forced_sm",
            "forced_copy_engine",
            "calibration_probe_sm",
            "calibration_probe_copy",
            "uncalibrated_copy_engine",
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
    same host-to-GPU link. SM gathers additionally consume execution capacity
    needed by the numerical kernel, while copy-engine work can overlap useful
    compute. Candidate makespan therefore serializes SM service with compute
    and overlaps copy service with compute; a hybrid never pretends the two
    movers are independent physical links. Missing copy-engine calibration
    fails closed to the SM path instead of reviving a hidden byte threshold.
    """

    sm_bandwidth_bytes_per_second: int
    copy_bandwidth_bytes_per_second: int | None = None
    copy_operation_ns: int | None = None
    hybrid_join_ns: int = 0
    minimum_gain: float = 1.03
    sm_samples: int = 0
    copy_samples: int = 0

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
        if self.hybrid_join_ns < 0 or self.minimum_gain < 1.0:
            raise ValueError("mover join cost and minimum gain are invalid")
        if min(self.sm_samples, self.copy_samples) < 0:
            raise ValueError("mover calibration sample counts cannot be negative")
        if self.copy_samples != 0 and not self.copy_calibrated:
            raise ValueError("copy-engine samples require a complete calibration")

    @property
    def copy_calibrated(self) -> bool:
        return self.copy_bandwidth_bytes_per_second is not None

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
        observed = max(1, transfer_bytes * 1_000_000_000 // elapsed_ns)
        bandwidth = self._bounded_ewma(
            self.sm_bandwidth_bytes_per_second,
            observed,
            alpha=alpha,
            maximum_step_ratio=maximum_step_ratio,
        )
        return replace(
            self,
            sm_bandwidth_bytes_per_second=bandwidth,
            sm_samples=self.sm_samples + 1,
        )

    def with_copy_observation(
        self,
        *,
        transfer_bytes: int,
        elapsed_ns: int,
        operation_count: int,
        issue_cpu_ns: int,
        alpha: float = 0.25,
        minimum_sample_bytes: int = 64 * 1024,
        maximum_step_ratio: float = 2.0,
    ) -> "IndexedMoverServiceModel":
        """Calibrate copy-DMA byte service and per-operation issue cost.

        CUDA events bound the stream-visible completion interval while the CPU
        timer measures descriptor issue.  Separating them prevents a fragmented
        layout from looking like low link bandwidth and lets the same model
        compare bulk, scattered, and hybrid candidates without a byte cutoff.
        """

        if min(
            transfer_bytes,
            elapsed_ns,
            operation_count,
            minimum_sample_bytes,
        ) <= 0 or issue_cpu_ns < 0:
            raise ValueError("copy-engine calibration geometry must be positive")
        if transfer_bytes < minimum_sample_bytes:
            return self
        transfer_ns = max(1, elapsed_ns - min(issue_cpu_ns, elapsed_ns - 1))
        observed_bandwidth = max(
            1, transfer_bytes * 1_000_000_000 // transfer_ns
        )
        observed_operation_ns = max(0, issue_cpu_ns // operation_count)
        if self.copy_bandwidth_bytes_per_second is None:
            # The SM service rate is a deployment-local prior for the same
            # physical link. It bounds a noisy first DMA observation without
            # assuming that the two issuers have equal service.
            bandwidth = self._bounded_ewma(
                self.sm_bandwidth_bytes_per_second,
                observed_bandwidth,
                alpha=1.0,
                maximum_step_ratio=4.0,
            )
            operation_ns = observed_operation_ns
        else:
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
        )

    @staticmethod
    def _transfer_ns(bytes: int, bandwidth_bytes_per_second: int) -> int:
        if bytes <= 0 or bandwidth_bytes_per_second <= 0:
            raise ValueError("mover service geometry must be positive")
        return max(
            1,
            (bytes * 1_000_000_000 + bandwidth_bytes_per_second - 1)
            // bandwidth_bytes_per_second,
        )

    def sm_only_ns(self, transfer_bytes: int) -> int:
        return self._transfer_ns(
            transfer_bytes, self.sm_bandwidth_bytes_per_second
        )

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
            return (
                self.sm_only_ns(total_rows * row_bytes) + overlap_compute_ns
            )
        if not self.copy_calibrated:
            return None
        assert self.copy_bandwidth_bytes_per_second is not None
        assert self.copy_operation_ns is not None
        copy_ns = self._transfer_ns(
            copy_rows * row_bytes, self.copy_bandwidth_bytes_per_second
        ) + copy_run_count * copy_operations_per_run * self.copy_operation_ns
        sm_rows = total_rows - copy_rows
        if sm_rows == 0:
            return max(copy_ns, overlap_compute_ns)
        # Copy DMA may cover useful compute, but the SM remainder competes for
        # execution capacity. The additive link bound also prevents a hybrid
        # from claiming the sum of two independently calibrated bandwidths.
        return (
            self._transfer_ns(
                sm_rows * row_bytes, self.sm_bandwidth_bytes_per_second
            )
            + max(copy_ns, overlap_compute_ns)
            + self.hybrid_join_ns
        )


@dataclass(frozen=True)
class IndexedMoverSelection:
    selected_run_indices: tuple[int, ...]
    predicted_sm_ns: int
    predicted_selected_ns: int | None
    reason: str


def select_indexed_mover_runs(
    layout: IndexedPairLayout,
    *,
    row_bytes: int,
    copy_operations_per_run: int,
    maximum_copy_runs: int,
    service_model: IndexedMoverServiceModel,
    policy: str = "auto",
    overlap_compute_ns: int = 0,
) -> IndexedMoverSelection:
    """Choose the exact partition minimizing resource-aware stage makespan."""

    if min(row_bytes, copy_operations_per_run, maximum_copy_runs) <= 0:
        raise ValueError("indexed mover service geometry must be positive")
    if overlap_compute_ns < 0:
        raise ValueError("indexed mover overlap compute cannot be negative")
    if policy not in {"auto", "sm", "copy_engine", "probe_copy"}:
        raise ValueError(
            "indexed mover policy must be auto, sm, copy_engine, or probe_copy"
        )
    predicted_sm_ns = service_model.candidate_ns(
        total_rows=layout.row_count,
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
    if policy == "copy_engine":
        if len(layout.runs) > maximum_copy_runs:
            raise ValueError("copy-engine index map exceeds the operation bound")
        selected = tuple(range(len(layout.runs)))
        return IndexedMoverSelection(
            selected,
            predicted_sm_ns,
            service_model.candidate_ns(
                total_rows=layout.row_count,
                copy_rows=layout.row_count,
                copy_run_count=len(selected),
                row_bytes=row_bytes,
                copy_operations_per_run=copy_operations_per_run,
                overlap_compute_ns=overlap_compute_ns,
            ),
            "forced_copy_engine",
        )
    if policy == "probe_copy":
        # Probe the longest representable runs so the copy engine contributes
        # a measurable byte interval while every omitted row remains on the SM
        # mover. Per-engine CUDA events make the hybrid observation separable.
        ordered = sorted(
            range(len(layout.runs)),
            key=lambda index: (-layout.runs[index].row_count, index),
        )[:maximum_copy_runs]
        if not ordered:  # pragma: no cover - non-empty layout invariant
            raise RuntimeError("copy-engine calibration has no candidate run")
        return IndexedMoverSelection(
            tuple(sorted(ordered)),
            predicted_sm_ns,
            predicted_sm_ns,
            "calibration_probe_copy",
        )
    if not service_model.copy_calibrated:
        return IndexedMoverSelection(
            (),
            predicted_sm_ns,
            predicted_sm_ns,
            "uncalibrated_copy_engine",
        )

    ordered = sorted(
        range(len(layout.runs)),
        key=lambda index: (-layout.runs[index].row_count, index),
    )[:maximum_copy_runs]
    selected_rows = 0
    best_indices: tuple[int, ...] = ()
    best_ns = predicted_sm_ns
    for run_index in ordered:
        selected_rows += layout.runs[run_index].row_count
        candidate_indices = (*best_indices, run_index)
        candidate_ns = service_model.candidate_ns(
            total_rows=layout.row_count,
            copy_rows=selected_rows,
            copy_run_count=len(candidate_indices),
            row_bytes=row_bytes,
            copy_operations_per_run=copy_operations_per_run,
            overlap_compute_ns=overlap_compute_ns,
        )
        if candidate_ns is None:  # pragma: no cover - calibrated invariant
            raise RuntimeError("calibrated mover produced no service estimate")
        # For the additive model, runs are ordered by non-increasing byte
        # benefit and every run has the same operation cost. Once a prefix no
        # longer improves the objective, no shorter remaining run can do so.
        if candidate_ns >= best_ns:
            break
        best_indices = candidate_indices
        best_ns = candidate_ns
    if not best_indices or predicted_sm_ns < best_ns * service_model.minimum_gain:
        return IndexedMoverSelection(
            (), predicted_sm_ns, predicted_sm_ns, "insufficient_gain"
        )
    return IndexedMoverSelection(
        tuple(sorted(best_indices)), predicted_sm_ns, best_ns, "service_cost"
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

    rows = tuple(tuple((int(source), int(destination)) for source, destination in pairs) for pairs in work_pairs)
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
