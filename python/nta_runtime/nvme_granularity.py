"""Exact NVMe materialization granularity planning.

The numerical demand is never approximated.  A plan either DMA-writes exact
contiguous source/destination runs directly into the engine cache, or reads a
larger contiguous source span into a bounded HBM scratch arena and compacts
only the requested rows into their exact destinations.  Selection is made by
an explicit deployment service model; an uncalibrated deployment remains on
the direct path.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from math import ceil, gcd, isfinite, lcm
from types import MappingProxyType
from typing import Any, Iterable


IndexPair = tuple[tuple[int, ...], tuple[int, ...]]


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


class NvmeGranularity(str, Enum):
    DIRECT = "direct"
    SPAN_COMPACT = "span_compact"


def plan_nvme_scratch_capacity(*, queue_depth: int, max_transfer_bytes: int) -> int:
    """Bound scratch to one usable controller-queue residency."""

    if min(queue_depth, max_transfer_bytes) <= 0:
        raise ValueError("NVMe scratch planning requires positive queue geometry")
    return max(1, queue_depth - 1) * max_transfer_bytes


@dataclass(frozen=True, slots=True)
class NvmeTransferServiceModel:
    """Measured steady-state costs used by the granularity optimizer.

    ``command_service_ns`` is the effective per-command queue service at the
    deployment queue depth, not the device's unloaded read latency.
    ``read_bandwidth_bytes_per_second`` captures the byte-dependent part of
    service after command cost.  Compaction is a separate HBM-to-HBM kernel.
    All three values must be measured before automatic span selection is
    allowed.
    """

    command_service_ns: int | None = None
    read_bandwidth_bytes_per_second: int | None = None
    compaction_bandwidth_bytes_per_second: int | None = None
    compaction_launch_ns: int = 0
    minimum_gain: float = 1.03

    def __post_init__(self) -> None:
        values = (
            self.command_service_ns,
            self.read_bandwidth_bytes_per_second,
            self.compaction_bandwidth_bytes_per_second,
        )
        if any(value is not None and value <= 0 for value in values):
            raise ValueError("NVMe service measurements must be positive")
        if self.compaction_launch_ns < 0:
            raise ValueError("NVMe compaction launch cost cannot be negative")
        if not isfinite(self.minimum_gain) or self.minimum_gain < 1.0:
            raise ValueError("NVMe minimum predicted gain must be at least one")

    @property
    def calibrated(self) -> bool:
        return all(
            value is not None
            for value in (
                self.command_service_ns,
                self.read_bandwidth_bytes_per_second,
                self.compaction_bandwidth_bytes_per_second,
            )
        )

    @staticmethod
    def _bytes_ns(bytes_: int, bandwidth: int) -> int:
        if bytes_ <= 0 or bandwidth <= 0:
            raise ValueError("NVMe service geometry must be positive")
        return max(1, ceil(bytes_ * 1_000_000_000 / bandwidth))

    def transfer_ns(self, *, command_count: int, transfer_bytes: int) -> int:
        if not self.calibrated or command_count <= 0 or transfer_bytes <= 0:
            raise RuntimeError("NVMe transfer prediction requires calibration")
        assert self.command_service_ns is not None
        assert self.read_bandwidth_bytes_per_second is not None
        return command_count * self.command_service_ns + self._bytes_ns(
            transfer_bytes, self.read_bandwidth_bytes_per_second
        )

    def compaction_ns(self, *, exact_bytes: int, launch_count: int = 1) -> int:
        if not self.calibrated or min(exact_bytes, launch_count) <= 0:
            raise RuntimeError("NVMe compaction prediction requires calibration")
        assert self.compaction_bandwidth_bytes_per_second is not None
        # Every exact byte is read once from scratch and written once to its
        # framework destination.
        return launch_count * self.compaction_launch_ns + self._bytes_ns(
            2 * exact_bytes, self.compaction_bandwidth_bytes_per_second
        )


@dataclass(frozen=True, slots=True)
class NvmeSpanSelection:
    source_row_offset: int
    destination_row: int

    def __post_init__(self) -> None:
        if min(self.source_row_offset, self.destination_row) < 0:
            raise ValueError("NVMe span selections cannot use negative rows")


@dataclass(frozen=True, slots=True)
class NvmeSourceSpan:
    source_first: int
    source_row_count: int
    scratch_offsets: tuple[int, ...]
    selections: tuple[NvmeSpanSelection, ...]

    def __post_init__(self) -> None:
        if (
            self.source_first < 0
            or self.source_row_count <= 0
            or not self.scratch_offsets
            or min(self.scratch_offsets) < 0
            or not self.selections
            or any(
                selection.source_row_offset >= self.source_row_count
                for selection in self.selections
            )
        ):
            raise ValueError("NVMe source span has invalid exact geometry")
        if tuple(sorted(set(self.scratch_offsets))) != self.scratch_offsets:
            raise ValueError("NVMe span scratch lanes must be unique and ordered")


@dataclass(frozen=True, slots=True)
class NvmeSpanPlan:
    pair_span_indices: tuple[tuple[IndexPair, tuple[int, ...]], ...]
    spans: tuple[NvmeSourceSpan, ...]
    lane_element_bytes: tuple[int, ...]
    physical_bytes: int
    exact_bytes: int
    scratch_bytes: int
    rows_per_lba: int
    _lookup: Any = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        if (
            not self.spans
            or not self.lane_element_bytes
            or min(self.lane_element_bytes) <= 0
            or min(
                self.physical_bytes,
                self.exact_bytes,
                self.scratch_bytes,
                self.rows_per_lba,
            )
            <= 0
        ):
            raise ValueError("NVMe span plan has invalid resource geometry")
        lookup = dict(self.pair_span_indices)
        if len(lookup) != len(self.pair_span_indices) or not lookup:
            raise ValueError("NVMe span plan repeats or omits an index pair")
        if any(
            not indices
            or tuple(sorted(set(indices))) != indices
            or indices[-1] >= len(self.spans)
            for indices in lookup.values()
        ):
            raise ValueError("NVMe pair-to-span ownership is invalid")
        object.__setattr__(self, "_lookup", MappingProxyType(lookup))

    @property
    def object_count(self) -> int:
        return len(self.spans) * len(self.lane_element_bytes)

    @property
    def command_count(self) -> int:
        return self.object_count

    @property
    def selected_row_copy_count(self) -> int:
        return len(self.lane_element_bytes) * sum(
            len(span.selections) for span in self.spans
        )

    def spans_for(self, pair: IndexPair) -> tuple[NvmeSourceSpan, ...]:
        try:
            return tuple(self.spans[index] for index in self._lookup[pair])
        except KeyError as error:
            raise RuntimeError("NVMe index pair was not span-validated") from error


@dataclass(frozen=True, slots=True)
class NvmeGranularityDecision:
    kind: NvmeGranularity
    reason: str
    direct_predicted_ns: int | None
    span_predicted_ns: int | None


def plan_nvme_spans(
    index_pairs: Iterable[IndexPair],
    *,
    lane_element_bytes: tuple[int, ...],
    lba_size: int,
    max_transfer_bytes: int,
    scratch_alignment: int,
    service_model: NvmeTransferServiceModel,
) -> NvmeSpanPlan:
    """Minimize measured transport cost over exact source-span partitions."""

    if (
        not lane_element_bytes
        or min(lane_element_bytes) <= 0
        or min(lba_size, max_transfer_bytes, scratch_alignment) <= 0
    ):
        raise ValueError("NVMe span planning requires positive resource geometry")
    if not service_model.calibrated:
        raise RuntimeError("NVMe span planning requires a calibrated service model")
    pairs = tuple(dict.fromkeys(pair for pair in index_pairs if pair[0]))
    if not pairs:
        raise RuntimeError("NVMe span planning has no external index pair")

    destination_source: dict[int, int] = {}
    source_destinations: dict[int, set[int]] = {}
    for source_rows, destination_rows in pairs:
        if len(source_rows) != len(destination_rows):
            raise RuntimeError("NVMe source and destination index counts disagree")
        if min(source_rows) < 0 or min(destination_rows) < 0:
            raise RuntimeError("NVMe span rows cannot be negative")
        if len(set(destination_rows)) != len(destination_rows):
            raise RuntimeError("one NVMe demand repeats a destination row")
        for source, destination in zip(source_rows, destination_rows, strict=True):
            previous = destination_source.setdefault(destination, source)
            if previous != source:
                raise RuntimeError("one NVMe destination is bound to two sources")
            source_destinations.setdefault(source, set()).add(destination)

    sources = tuple(sorted(source_destinations))
    rows_per_lba = lcm(
        *(
            lba_size // gcd(lba_size, element_bytes)
            for element_bytes in lane_element_bytes
        )
    )
    maximum_span_rows = min(
        max_transfer_bytes // element_bytes for element_bytes in lane_element_bytes
    )
    maximum_span_rows -= maximum_span_rows % rows_per_lba
    if maximum_span_rows <= 0:
        raise RuntimeError("one LBA-aligned row group exceeds the NVMe transfer limit")

    # For a span [i, j), the calibrated objective is a fixed command cost plus
    # a linear byte service. Quantize byte service upward once per source row;
    # this is conservative by less than one nanosecond per physical row and
    # makes the recurrence separable:
    #
    #   dp[j] = fixed + row_ns * (source[j-1] + 1)
    #           + min_i(dp[i] - row_ns * source[i]).
    #
    # A monotone deque per LBA residue therefore computes the globally optimal
    # partition under this conservative model in O(selected_rows), rather than
    # copying O(n)-length predecessor tuples across an O(n^2) search. Physical
    # bytes and command count remain deterministic lexicographic tie breakers;
    # there is no empirical row-gap threshold hidden in the planner.
    lane_bytes = sum(lane_element_bytes)
    assert service_model.command_service_ns is not None
    assert service_model.read_bandwidth_bytes_per_second is not None
    fixed_service_ns = (
        len(lane_element_bytes) * service_model.command_service_ns
    )
    row_service_ns = service_model._bytes_ns(
        lane_bytes, service_model.read_bandwidth_bytes_per_second
    )
    count = len(sources)
    best_service_ns = [0] * (count + 1)
    best_physical_bytes = [0] * (count + 1)
    best_command_count = [0] * (count + 1)
    predecessor = [-1] * (count + 1)
    reachable = [False] * (count + 1)
    reachable[0] = True
    # Entries are (adjusted lexicographic objective, selected-row index).
    candidates: dict[int, deque[tuple[tuple[int, int, int], int]]] = defaultdict(
        deque
    )
    for end in range(count):
        begin = end
        if reachable[begin]:
            adjusted = (
                best_service_ns[begin] - row_service_ns * sources[begin],
                best_physical_bytes[begin] - lane_bytes * sources[begin],
                best_command_count[begin],
            )
            queue = candidates[sources[begin] % rows_per_lba]
            # A later no-worse candidate remains valid for at least as long.
            while queue and queue[-1][0] >= adjusted:
                queue.pop()
            queue.append((adjusted, begin))

        target_residue = (sources[end] + 1) % rows_per_lba
        queue = candidates[target_residue]
        minimum_source = sources[end] - maximum_span_rows + 1
        while queue and sources[queue[0][1]] < minimum_source:
            queue.popleft()
        if not queue:
            continue
        adjusted, begin = queue[0]
        endpoint = sources[end] + 1
        best_service_ns[end + 1] = (
            fixed_service_ns + row_service_ns * endpoint + adjusted[0]
        )
        best_physical_bytes[end + 1] = lane_bytes * endpoint + adjusted[1]
        best_command_count[end + 1] = len(lane_element_bytes) + adjusted[2]
        predecessor[end + 1] = begin
        reachable[end + 1] = True
    if not reachable[-1]:
        raise RuntimeError("NVMe exact rows cannot form bounded source spans")

    partition: list[tuple[int, int]] = []
    end = count
    while end != 0:
        begin = predecessor[end]
        if begin < 0 or begin >= end:
            raise RuntimeError("NVMe span planner produced an invalid predecessor")
        partition.append((begin, end))
        end = begin
    partition.reverse()

    spans: list[NvmeSourceSpan] = []
    source_span_index: dict[int, int] = {}
    scratch_cursor = 0
    for begin, end in partition:
        source_first = sources[begin]
        source_row_count = sources[end - 1] - source_first + 1
        offsets: list[int] = []
        for element_bytes in lane_element_bytes:
            scratch_cursor = _round_up(scratch_cursor, scratch_alignment)
            offsets.append(scratch_cursor)
            scratch_cursor += source_row_count * element_bytes
        selections: list[NvmeSpanSelection] = []
        span_index = len(spans)
        for source in sources[begin:end]:
            source_span_index[source] = span_index
            for destination in sorted(source_destinations[source]):
                selections.append(NvmeSpanSelection(source - source_first, destination))
        spans.append(
            NvmeSourceSpan(
                source_first,
                source_row_count,
                tuple(offsets),
                tuple(selections),
            )
        )

    pair_span_indices = tuple(
        (
            pair,
            tuple(sorted({source_span_index[source] for source in pair[0]})),
        )
        for pair in pairs
    )
    exact_mapping_count = len(destination_source)
    return NvmeSpanPlan(
        pair_span_indices,
        tuple(spans),
        tuple(lane_element_bytes),
        best_physical_bytes[-1],
        exact_mapping_count * lane_bytes,
        scratch_cursor,
        rows_per_lba,
    )


def choose_nvme_granularity(
    *,
    direct_command_count: int,
    direct_transfer_bytes: int,
    direct_work_item_count: int,
    span_command_count: int,
    span_transfer_bytes: int,
    span_exact_bytes: int,
    span_work_item_count: int,
    span_scratch_bytes: int,
    compaction_launch_count: int,
    object_capacity: int,
    work_ticket_capacity: int,
    scratch_capacity_bytes: int,
    service_model: NvmeTransferServiceModel,
) -> NvmeGranularityDecision:
    """Choose an exact physical plan; never choose data quality."""

    if (
        min(
            direct_command_count,
            direct_transfer_bytes,
            direct_work_item_count,
            object_capacity,
            work_ticket_capacity,
        )
        <= 0
        or min(
            span_command_count,
            span_transfer_bytes,
            span_exact_bytes,
            span_work_item_count,
            span_scratch_bytes,
            compaction_launch_count,
        )
        < 0
    ):
        raise ValueError("NVMe granularity comparison requires positive geometry")
    if scratch_capacity_bytes < 0:
        raise ValueError("NVMe scratch capacity cannot be negative")
    if not service_model.calibrated:
        return NvmeGranularityDecision(
            NvmeGranularity.DIRECT, "uncalibrated", None, None
        )
    if (
        min(
            span_command_count,
            span_transfer_bytes,
            span_exact_bytes,
            span_work_item_count,
            span_scratch_bytes,
        )
        <= 0
    ):
        raise ValueError("calibrated NVMe span geometry must be positive")
    direct_ns = service_model.transfer_ns(
        command_count=direct_command_count,
        transfer_bytes=direct_transfer_bytes,
    )
    span_ns = service_model.transfer_ns(
        command_count=span_command_count,
        transfer_bytes=span_transfer_bytes,
    ) + service_model.compaction_ns(
        exact_bytes=span_exact_bytes,
        launch_count=compaction_launch_count,
    )
    direct_feasible = (
        direct_command_count <= object_capacity
        and direct_work_item_count <= work_ticket_capacity
    )
    span_runtime_feasible = (
        span_command_count <= object_capacity
        and span_work_item_count <= work_ticket_capacity
    )
    span_feasible = (
        span_runtime_feasible and span_scratch_bytes <= scratch_capacity_bytes
    )
    if not direct_feasible and span_feasible:
        return NvmeGranularityDecision(
            NvmeGranularity.SPAN_COMPACT,
            "direct_capacity",
            direct_ns,
            span_ns,
        )
    if not span_feasible:
        reason = (
            "scratch_capacity"
            if span_scratch_bytes > scratch_capacity_bytes
            else "runtime_capacity"
        )
        return NvmeGranularityDecision(
            NvmeGranularity.DIRECT, reason, direct_ns, span_ns
        )
    if direct_ns < ceil(span_ns * service_model.minimum_gain):
        return NvmeGranularityDecision(
            NvmeGranularity.DIRECT,
            "insufficient_gain",
            direct_ns,
            span_ns,
        )
    return NvmeGranularityDecision(
        NvmeGranularity.SPAN_COMPACT,
        "service_cost",
        direct_ns,
        span_ns,
    )
