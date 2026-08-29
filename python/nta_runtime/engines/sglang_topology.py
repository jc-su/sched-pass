"""Pure request-to-work topology construction for the SGLang adapter.

This module translates typed framework metadata into exact acquisition groups
and runtime dependencies.  It owns neither HiCache leases nor CUDA resources,
which keeps policy, lifetime, and numerical execution independently testable.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
import heapq

from nta_runtime.adapters.sglang import SglangAcquisitionSpan
from nta_runtime.engines.sglang_contracts import (
    LeaseAcquisitionGroup,
    LeaseAcquisitionSlice,
    LeaseOperationRange,
    LeaseOperationTransfer,
    PagePair,
)
from nta_runtime.flashinfer_schedule import Schedule
from nta_runtime.indexed_transfer import (
    AcquisitionGroup,
    AcquisitionSlice,
    AcquisitionTopology,
)
from nta_runtime.requests import RequestBinding


def group_external_pages_by_request(
    schedule: Schedule, page_pairs: tuple[PagePair, ...]
) -> tuple[PagePair, ...]:
    """Share one exact indexed K/V acquisition group across request CTAs."""

    if schedule.work_count != len(page_pairs):
        raise RuntimeError("FlashInfer work and page-pair counts disagree")
    pages_by_request: dict[int, dict[int, int]] = {}
    for request_index, (host_pages, device_pages) in zip(
        schedule.request_indices, page_pairs, strict=True
    ):
        if len(host_pages) != len(device_pages):
            raise RuntimeError("HiCache host/device page mappings disagree")
        request_pages = pages_by_request.setdefault(request_index, {})
        for host_page, device_page in zip(host_pages, device_pages, strict=True):
            previous = request_pages.setdefault(device_page, host_page)
            if previous != host_page:
                raise RuntimeError(
                    "one device KV page maps to multiple host cache pages"
                )

    grouped = {
        request_index: (tuple(pages.values()), tuple(pages))
        for request_index, pages in pages_by_request.items()
        if pages
    }
    return tuple(
        grouped.get(request_index, ((), ())) if host_pages else ((), ())
        for request_index, (host_pages, _device_pages) in zip(
            schedule.request_indices, page_pairs, strict=True
        )
    )


def page_pairs_for_schedule(
    schedule: Schedule,
    *,
    indptr: Sequence[int],
    pages: Sequence[int],
    last_page: Sequence[int],
    page_size: int,
    source_by_device: Mapping[int, int],
) -> tuple[PagePair, ...]:
    """Bind exact external pages to work without context-sized copies.

    ``pages[indptr[r]:indptr[r + 1]]`` is the complete request context. Taking
    that slice once per CTA makes metadata construction O(work * context) even
    though each CTA consumes only one bounded KV chunk. Absolute bounds ensure
    each page is visited only by the work item that consumes it.
    """

    if page_size <= 0:
        raise RuntimeError("FlashInfer KV page size must be positive")
    normalized_indptr = tuple(int(value) for value in indptr)
    normalized_last_page = tuple(int(value) for value in last_page)
    if (
        len(normalized_indptr) != len(normalized_last_page) + 1
        or not normalized_indptr
        or normalized_indptr[0] != 0
        or normalized_indptr[-1] != len(pages)
        or any(
            begin < 0 or begin >= end
            for begin, end in zip(normalized_indptr, normalized_indptr[1:])
        )
    ):
        raise RuntimeError("FlashInfer emitted an invalid KV page table")
    if any(value <= 0 or value > page_size for value in normalized_last_page):
        raise RuntimeError("FlashInfer emitted an invalid last-page length")

    pairs: list[PagePair] = []
    for request_index, kv_tile in zip(
        schedule.request_indices, schedule.kv_tile_indices, strict=True
    ):
        if (
            request_index < 0
            or request_index >= len(normalized_last_page)
            or kv_tile < 0
        ):
            raise RuntimeError("FlashInfer emitted an invalid KV tile coordinate")
        request_begin = normalized_indptr[request_index]
        request_end = normalized_indptr[request_index + 1]
        page_begin = request_begin
        page_end = request_end
        if schedule.kv_chunk_tokens > 0:
            token_count = (request_end - request_begin - 1) * page_size + (
                normalized_last_page[request_index]
            )
            token_begin = kv_tile * schedule.kv_chunk_tokens
            if token_begin >= token_count:
                raise RuntimeError("FlashInfer emitted an out-of-range KV tile")
            token_end = min(token_count, token_begin + schedule.kv_chunk_tokens)
            page_begin += token_begin // page_size
            page_end = request_begin + (token_end + page_size - 1) // page_size
        device_pages = tuple(
            int(page)
            for page in pages[page_begin:page_end]
            if int(page) in source_by_device
        )
        source_pages = tuple(source_by_device[page] for page in device_pages)
        pairs.append((source_pages, device_pages))
    return tuple(pairs)


def resolve_request_acquisitions(
    acquisitions: Sequence[SglangAcquisitionSpan],
    operation_transfers: Mapping[int, LeaseOperationTransfer],
    *,
    lease_transfer_rows: int,
) -> tuple[SglangAcquisitionSpan, ...]:
    """Join framework request identity to one exact acquisition lease."""

    if lease_transfer_rows <= 0:
        raise RuntimeError("SGLang acquisition lease contains no rows")
    normalized: dict[int, LeaseOperationTransfer] = {}
    for raw_operation_id, transfer in operation_transfers.items():
        operation_id = int(raw_operation_id)
        if (
            not isinstance(transfer, LeaseOperationTransfer)
            or operation_id != transfer.operation_id
            or operation_id < 0
        ):
            raise RuntimeError(
                "SGLang acquisition lease has invalid operation identity"
            )
        if operation_id in normalized:
            raise RuntimeError("SGLang acquisition lease repeats an operation")
        normalized[operation_id] = transfer
    if sum(item.row_count for item in normalized.values()) != lease_transfer_rows:
        raise RuntimeError("SGLang acquisition operations do not cover the lease rows")

    resolved = tuple(acquisitions)
    if any(not isinstance(item, SglangAcquisitionSpan) for item in resolved):
        raise RuntimeError("SGLang forward carries untyped acquisition metadata")
    referenced_operations: set[int] = set()
    for acquisition in resolved:
        if not acquisition.is_external:
            continue
        transfer = normalized.get(acquisition.operation_id)
        if transfer is None:
            raise RuntimeError(
                "SGLang request references an operation outside its acquisition lease"
            )
        if (
            acquisition.node_id != transfer.node_id
            or acquisition.row_count != transfer.row_count
        ):
            raise RuntimeError(
                "SGLang request span disagrees with its acquisition operation"
            )
        if acquisition.operation_id in referenced_operations:
            raise RuntimeError(
                "one SGLang acquisition operation cannot be owned by multiple "
                "requests"
            )
        referenced_operations.add(acquisition.operation_id)

    missing_operations = set(normalized) - referenced_operations
    if missing_operations:
        raise RuntimeError(
            "SGLang forward metadata omits acquisition ownership for "
            f"{len(missing_operations)} operation(s): "
            f"lease={tuple(sorted(normalized))}, "
            f"requests={tuple(item.operation_id for item in resolved)}"
        )
    if not referenced_operations:
        raise RuntimeError("SGLang forward has no request bound to its acquisition lease")
    return resolved


def request_batch_heterogeneity(
    bindings: Sequence[RequestBinding],
    sequence_lengths: Sequence[int],
    acquisitions: Sequence[SglangAcquisitionSpan],
) -> tuple[str, ...]:
    """Return the exact axes that differ inside one engine ForwardBatch."""

    size = len(bindings)
    if len(sequence_lengths) != size or len(acquisitions) != size:
        raise RuntimeError("SGLang batch heterogeneity vectors are misaligned")
    if size < 2:
        return ()
    axes: list[str] = []
    if len({int(value) for value in sequence_lengths}) > 1:
        axes.append("sequence_length")
    if len({item.is_external for item in acquisitions}) > 1:
        axes.append("availability")
    external_rows = {
        int(item.row_count) for item in acquisitions if item.is_external
    }
    if len(external_rows) > 1:
        axes.append("external_rows")
    for name, values in (
        ("tenant", (binding.tenant_id for binding in bindings)),
        ("priority", (binding.priority for binding in bindings)),
        ("deadline", (binding.deadline_clock for binding in bindings)),
    ):
        if len({int(value) for value in values}) > 1:
            axes.append(name)
    return tuple(axes)


def project_acquisition_slices(
    schedule: Schedule,
    acquisitions: Sequence[SglangAcquisitionSpan],
    sequence_lengths: Sequence[int],
) -> tuple[LeaseAcquisitionSlice | None, ...]:
    """Project exact operation-local spans onto FlashInfer work units."""

    if len(acquisitions) != len(sequence_lengths):
        raise RuntimeError("SGLang acquisition and sequence vectors are misaligned")
    intersections: list[list[tuple[int, int]]] = [[] for _ in acquisitions]
    acquisition_slices: list[LeaseAcquisitionSlice | None] = []
    for request_index, kv_tile in zip(
        schedule.request_indices, schedule.kv_tile_indices, strict=True
    ):
        if request_index < 0 or request_index >= len(acquisitions) or kv_tile < 0:
            raise RuntimeError("FlashInfer emitted an invalid acquisition coordinate")
        sequence_end = int(sequence_lengths[request_index])
        if schedule.kv_chunk_tokens > 0:
            tile_begin = int(kv_tile) * schedule.kv_chunk_tokens
            tile_end = min(sequence_end, tile_begin + schedule.kv_chunk_tokens)
        else:
            tile_begin = 0
            tile_end = sequence_end
        if tile_begin < 0 or tile_begin >= tile_end:
            raise RuntimeError("FlashInfer emitted an empty KV work tile")
        acquisition = acquisitions[request_index]
        overlap_begin = max(tile_begin, acquisition.logical_begin)
        overlap_end = min(tile_end, acquisition.logical_end)
        rows = max(0, overlap_end - overlap_begin) if acquisition.is_external else 0
        acquisition_slices.append(
            LeaseAcquisitionSlice(
                acquisition.operation_id,
                overlap_begin - acquisition.logical_begin,
                rows,
            )
            if rows
            else None
        )
        if rows:
            intersections[request_index].append((overlap_begin, overlap_end))

    for request_index, acquisition in enumerate(acquisitions):
        if not acquisition.is_external:
            continue
        # FlashInfer repeats a request/tile coordinate for independent
        # query/head work. Retain fan-out above, but validate exact coverage
        # over the unique transport intervals.
        spans = sorted(set(intersections[request_index]))
        cursor = acquisition.logical_begin
        for begin, end in spans:
            if begin != cursor:
                raise RuntimeError(
                    "FlashInfer CTA schedule duplicates or omits acquired request "
                    f"rows: request={request_index}, acquisition="
                    f"[{acquisition.logical_begin},{acquisition.logical_end}), "
                    f"cursor={cursor}, next=[{begin},{end}), "
                    f"chunk={schedule.kv_chunk_tokens}, spans={spans[:16]}"
                )
            cursor = end
        if cursor != acquisition.logical_end:
            raise RuntimeError(
                "FlashInfer CTA schedule does not cover the acquired request span: "
                f"request={request_index}, acquisition="
                f"[{acquisition.logical_begin},{acquisition.logical_end}), "
                f"covered_end={cursor}, chunk={schedule.kv_chunk_tokens}, "
                f"spans={spans[:16]}"
            )
    return tuple(acquisition_slices)


def capacity_constrained_acquisition_groups(
    dependencies: Sequence[LeaseAcquisitionSlice | None],
    *,
    maximum_groups: int,
) -> tuple[LeaseAcquisitionGroup | None, ...]:
    """Choose an exact, bounded completion granularity for one schedule.

    Numerical work retains its original interval. The transport may merge
    adjacent intervals from the same acquisition operation so its two K/V
    objects per group fit the runtime directory. Greedily bisecting the largest
    remaining interval minimizes the worst completion quantum without a byte
    threshold or workload-specific policy.
    """

    if maximum_groups <= 0:
        raise ValueError("typed transfer grouping requires positive capacity")
    unique_by_operation: dict[int, dict[LeaseAcquisitionSlice, None]] = {}
    for dependency in dependencies:
        if dependency is None:
            continue
        unique_by_operation.setdefault(dependency.operation_id, {}).setdefault(
            dependency, None
        )
    if not unique_by_operation:
        raise RuntimeError("typed transfer grouping has no external dependency")
    if len(unique_by_operation) > maximum_groups:
        raise RuntimeError(
            "runtime object capacity cannot represent every acquisition operation"
        )

    intervals_by_operation: dict[int, tuple[LeaseAcquisitionSlice, ...]] = {}
    prefix_rows_by_operation: dict[int, tuple[int, ...]] = {}
    for operation_id in sorted(unique_by_operation):
        intervals = sorted(
            unique_by_operation[operation_id], key=lambda item: item.row_begin
        )
        cursor = 0
        prefix_rows = [0]
        for interval in intervals:
            if interval.row_begin != cursor:
                raise RuntimeError(
                    "typed work intervals do not exactly partition an operation"
                )
            cursor = interval.row_end
            prefix_rows.append(prefix_rows[-1] + interval.row_count)
        intervals_by_operation[operation_id] = tuple(intervals)
        prefix_rows_by_operation[operation_id] = tuple(prefix_rows)

    exact_group_count = sum(len(items) for items in intervals_by_operation.values())
    target_groups = min(maximum_groups, exact_group_count)
    partitions: dict[int, tuple[int, int, int]] = {}
    candidates: list[tuple[int, int, int, int, int]] = []
    next_partition_id = 0

    def add_partition(operation_id: int, first: int, last: int) -> None:
        nonlocal next_partition_id
        intervals = intervals_by_operation[operation_id]
        prefix_rows = prefix_rows_by_operation[operation_id]
        if not 0 <= first < last <= len(intervals):
            raise RuntimeError("typed transfer partition range is invalid")
        partition_id = next_partition_id
        next_partition_id += 1
        partitions[partition_id] = (operation_id, first, last)
        count = last - first
        if count > 1:
            heapq.heappush(
                candidates,
                (
                    -(prefix_rows[last] - prefix_rows[first]),
                    -count,
                    operation_id,
                    intervals[first].row_begin,
                    partition_id,
                ),
            )

    for operation_id, intervals in intervals_by_operation.items():
        add_partition(operation_id, 0, len(intervals))
    while len(partitions) < target_groups:
        if not candidates:  # pragma: no cover - exact count invariant
            raise RuntimeError("typed transfer partition cannot reach its capacity")
        _rows, _count, _operation, _begin, partition_id = heapq.heappop(candidates)
        operation_id, first, last = partitions.pop(partition_id)
        prefix_rows = prefix_rows_by_operation[operation_id]
        base_rows = prefix_rows[first]
        total_rows = prefix_rows[last] - base_rows
        insertion = bisect.bisect_left(
            prefix_rows,
            base_rows + (total_rows + 1) // 2,
            lo=first + 1,
            hi=last,
        )
        split_candidates = {
            max(first + 1, min(last - 1, insertion)),
            max(first + 1, min(last - 1, insertion - 1)),
        }
        split_index = min(
            split_candidates,
            key=lambda candidate: (
                abs(total_rows - 2 * (prefix_rows[candidate] - base_rows)),
                candidate,
            ),
        )
        add_partition(operation_id, first, split_index)
        add_partition(operation_id, split_index, last)

    grouped: dict[LeaseAcquisitionSlice, LeaseAcquisitionGroup] = {}
    for operation_id, first_index, last_index in partitions.values():
        intervals = intervals_by_operation[operation_id]
        first = intervals[first_index]
        last = intervals[last_index - 1]
        group = LeaseAcquisitionGroup(
            operation_id,
            first.row_begin,
            last.row_end - first.row_begin,
        )
        for interval_index in range(first_index, last_index):
            grouped[intervals[interval_index]] = group
    return tuple(
        None if dependency is None else grouped[dependency]
        for dependency in dependencies
    )


def lease_acquisition_topology(
    acquisition_slices: tuple[LeaseAcquisitionSlice | None, ...],
    acquisition_groups: tuple[LeaseAcquisitionGroup | None, ...],
    operations: tuple[LeaseOperationRange, ...],
    *,
    index_count: int,
) -> AcquisitionTopology:
    """Translate lease-local exactness into the shared acquisition contract."""

    if not acquisition_slices or len(acquisition_slices) != len(acquisition_groups):
        raise ValueError("lease indexed topology requires aligned work dependencies")
    operation_by_id = {operation.operation_id: operation for operation in operations}
    if len(operation_by_id) != len(operations):
        raise ValueError("lease indexed topology repeats an operation identity")
    group_by_identity: dict[LeaseAcquisitionGroup, int] = {}
    groups: list[AcquisitionGroup] = []

    def group_index(group: LeaseAcquisitionGroup) -> int:
        existing = group_by_identity.get(group)
        if existing is not None:
            return existing
        operation = operation_by_id.get(group.operation_id)
        if operation is None or group.row_end > operation.row_count:
            raise ValueError("lease transfer dependency exceeds its operation")
        result = len(groups)
        group_by_identity[group] = result
        groups.append(
            AcquisitionGroup(
                operation.row_begin + group.row_begin,
                group.row_count,
            )
        )
        return result

    dependencies_by_work: list[tuple[AcquisitionSlice, ...]] = []
    for exact, group in zip(acquisition_slices, acquisition_groups, strict=True):
        if exact is None:
            if group is not None:
                raise ValueError("direct lease work retained a transfer dependency")
            dependencies_by_work.append(())
            continue
        if (
            group is None
            or group.operation_id != exact.operation_id
            or group.row_begin > exact.row_begin
            or group.row_end < exact.row_end
        ):
            raise ValueError("lease transfer group does not contain exact work")
        index = group_index(group)
        dependencies_by_work.append(
            (
                AcquisitionSlice(
                    index,
                    exact.row_begin - group.row_begin,
                    exact.row_count,
                ),
            )
        )
    if not groups:
        raise ValueError("lease indexed topology has no external transfer")
    return AcquisitionTopology(
        index_count,
        tuple(groups),
        tuple(dependencies_by_work),
    )
