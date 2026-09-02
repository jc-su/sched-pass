"""Pure exact-demand geometry planning for SGLang NVMe acquisition."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil
from typing import Any

from nta_runtime.engines.sglang_contracts import PagePair
from nta_runtime.indexed_transfer import ContiguousPairRun
from nta_runtime.nvme_materialization import (
    NvmeRunPlan,
    NvmeRunSummary,
    materialize_nvme_run_plan,
    summarize_nvme_runs,
)
from nta_runtime.nvme_granularity import (
    NvmeGranularity,
    NvmeGranularityDecision,
    NvmeSourceSpan,
    NvmeSpanPlan,
    NvmeTransferServiceModel,
    choose_nvme_granularity,
    plan_nvme_spans,
)
from nta_runtime.requests import RequestBinding


@dataclass(frozen=True, slots=True)
class NvmeTransferPacketPlan:
    """One physical K/V packet shared by generation-bound consumers.

    This is deliberately not an exact acquisition group.  A direct run may
    coalesce adjacent exact demand, while a source span may additionally read
    rows that are discarded by compaction.  ``packet_index`` is a forward-local
    join key used by exact groups; request-generation identity is attached only
    after the layer resource version is known.
    """

    packet_index: int
    materialization: ContiguousPairRun | NvmeSourceSpan
    source_first: int
    destination_first: int | None
    row_count: int
    consumer_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.packet_index < 0
            or self.source_first < 0
            or (self.destination_first is not None and self.destination_first < 0)
            or self.row_count <= 0
            or not self.consumer_indices
            or tuple(sorted(set(self.consumer_indices))) != self.consumer_indices
        ):
            raise ValueError("NVMe transfer packet has invalid physical ownership")


@dataclass(frozen=True, slots=True)
class NvmeExactAcquisitionGroupPlan:
    """One compiler-visible exact demand segment and its physical packets."""

    group_index: int
    request_index: int
    segment_begin: int
    segment_count: int
    packet_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            min(self.group_index, self.request_index, self.segment_begin) < 0
            or self.segment_count <= 0
            or not self.packet_indices
            or tuple(sorted(set(self.packet_indices))) != self.packet_indices
        ):
            raise ValueError("NVMe exact acquisition group is invalid")

    @property
    def segment_end(self) -> int:
        return self.segment_begin + self.segment_count


@dataclass(frozen=True, slots=True)
class NvmeScopedMaterializationPlan:
    """Physical materialization deduplicated within one tenant scope."""

    tenant_id: int
    plan: NvmeRunPlan | NvmeSpanPlan
    packets: tuple[NvmeTransferPacketPlan, ...]
    span_scratch_offset: int = 0


@dataclass(frozen=True, slots=True)
class _ScopedMaterializationCandidate:
    """Allocation-light alternatives retained until policy is decided."""

    tenant_id: int
    consumers: tuple[tuple[PagePair, tuple[int, ...]], ...]
    direct: NvmeRunSummary
    direct_owners: tuple[tuple[tuple[int, int, int], tuple[int, ...]], ...]
    span: NvmeSpanPlan | None = None
    span_owners: tuple[tuple[int, ...], ...] = ()
    span_scratch_offset: int = 0


class _PagePairKey:
    """Cache the O(rows) hash while exact pairs move through batch planning."""

    __slots__ = ("pair", "_hash")

    def __init__(self, pair: PagePair) -> None:
        self.pair = pair
        self._hash = hash(pair)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _PagePairKey) and (
            self.pair is other.pair or self.pair == other.pair
        )


@dataclass(frozen=True, slots=True)
class NvmeBatchGeometry:
    """Layer-invariant exact-demand and physical geometry for one forward."""

    scopes: tuple[NvmeScopedMaterializationPlan, ...]
    exact_groups: tuple[NvmeExactAcquisitionGroupPlan, ...]
    work_group_indices: tuple[tuple[int, tuple[int | None, ...]], ...]
    row_bytes: tuple[int, int]
    object_count: int
    work_item_count: int
    logical_transfer_bytes: int
    scoped_exact_transfer_bytes: int
    unique_source_transfer_bytes: int
    physical_transfer_bytes: int
    granularity: NvmeGranularity
    granularity_reason: str
    scratch_bytes_per_layer: int
    direct_predicted_ns: int | None
    selected_predicted_ns: int | None

    def groups_for_wrapper(self, wrapper_id: int) -> tuple[int | None, ...]:
        for candidate, groups in self.work_group_indices:
            if candidate == wrapper_id:
                return groups
        raise KeyError("NVMe geometry has no exact groups for wrapper")


def plan_nvme_window_layer_capacity(
    *,
    layer_count: int,
    objects_per_layer: int,
    capacity_layer_limit: int,
    queue_depth: int,
    explicit_limit: int | None = None,
) -> int:
    """Size one producer window from queue geometry, not a device threshold.

    One queue residency only measures fill and drain. Two usable queue depths
    permit completion-driven refill and therefore reach the steady-state data
    path; one additional layer keeps useful producer work available while the
    next compact descriptor image is enqueued. Capacity and model bounds remain
    hard limits. An explicit limit exists only for mechanism-envelope tests.
    """

    if min(layer_count, objects_per_layer, capacity_layer_limit, queue_depth) <= 0:
        raise ValueError("NVMe window planning requires positive resource geometry")
    if explicit_limit is not None and (
        explicit_limit <= 0 or explicit_limit > layer_count
    ):
        raise ValueError("NVMe acquisition window limit exceeds model layers")
    if explicit_limit is not None:
        return min(capacity_layer_limit, layer_count, explicit_limit)
    usable_queue_entries = max(1, queue_depth - 1)
    refill_layers = (
        2 * usable_queue_entries + objects_per_layer - 1
    ) // objects_per_layer
    producer_lookahead = 1 if refill_layers < layer_count else 0
    return min(
        capacity_layer_limit,
        layer_count,
        refill_layers + producer_lookahead,
    )


def plan_nvme_batch_geometry(
    *,
    semantic_plans: Mapping[int, Any],
    bindings: tuple[RequestBinding, ...],
    row_bytes: tuple[int, int],
    lba_size: int,
    max_transfer_bytes: int,
    object_capacity: int,
    work_ticket_capacity: int,
    tenant_isolation: bool,
    service_model: NvmeTransferServiceModel | None = None,
    scratch_capacity_bytes: int = 0,
    scratch_alignment: int = 4096,
) -> NvmeBatchGeometry:
    """Factor exact numerical demand into shared transport acquisition groups.

    KV objects are request-owned, so identical runs are deduplicated only among
    consumers with the same tenant ID.  ``tenant_isolation`` controls budget
    policy outside this geometry planner; disabling that policy never widens
    object ownership or permits cross-tenant fan-out.  Immutable resources may
    be shared globally only through an explicit ``GLOBAL_SHARED`` contract and
    therefore do not enter this request-KV planner.
    """

    if (
        not semantic_plans
        or not bindings
        or len(row_bytes) != 2
        or min(row_bytes) <= 0
        or min(lba_size, max_transfer_bytes, object_capacity, work_ticket_capacity) <= 0
        or scratch_capacity_bytes < 0
        or scratch_alignment <= 0
    ):
        raise ValueError("NVMe batch geometry requires non-empty bounded inputs")
    if not isinstance(tenant_isolation, bool):
        raise TypeError("NVMe tenant isolation policy must be boolean")
    binding_by_index = {binding.request_index: binding for binding in bindings}
    if len(binding_by_index) != len(bindings):
        raise ValueError("NVMe request bindings repeat an engine request index")

    pair_consumers: dict[_PagePairKey, set[int]] = defaultdict(set)
    request_pairs: dict[int, dict[_PagePairKey, None]] = defaultdict(dict)
    work_pairs: dict[int, tuple[tuple[int, _PagePairKey | None], ...]] = {}
    for wrapper_id, semantic in semantic_plans.items():
        if getattr(semantic, "dependency_kind", None) != "physical_pages":
            raise RuntimeError("NVMe acquisition requires physical-page semantics")
        schedule = semantic.schedule
        if len(schedule.request_indices) != len(semantic.page_pairs):
            raise RuntimeError("NVMe schedule and physical-page demand disagree")
        wrapper_pairs: list[tuple[int, _PagePairKey | None]] = []
        for request_index, pair in zip(
            schedule.request_indices, semantic.page_pairs, strict=True
        ):
            if request_index not in binding_by_index:
                raise RuntimeError("NVMe work references an unbound request")
            if pair[0]:
                if len(pair[0]) != len(pair[1]):
                    raise RuntimeError("NVMe source/destination pair is malformed")
                key = _PagePairKey(pair)
                pair_consumers[key].add(int(request_index))
                request_pairs[int(request_index)].setdefault(key, None)
                wrapper_pairs.append((int(request_index), key))
            else:
                if pair[1]:
                    raise RuntimeError("NVMe direct work has destination-only demand")
                wrapper_pairs.append((int(request_index), None))
        work_pairs[int(wrapper_id)] = tuple(wrapper_pairs)
    if not pair_consumers:
        raise RuntimeError("NVMe external batch contains no physical demand")

    scoped_pairs: dict[int, dict[_PagePairKey, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for pair_key, consumers in pair_consumers.items():
        for request_index in consumers:
            tenant = binding_by_index[request_index].tenant_id
            scoped_pairs[tenant][pair_key].add(request_index)

    model = service_model or NvmeTransferServiceModel()
    if not isinstance(model, NvmeTransferServiceModel):
        raise TypeError("NVMe batch geometry requires a typed service model")
    lane_bytes = sum(row_bytes)
    logical_mappings = {
        (source, destination)
        for pair_key in pair_consumers
        for pair in (pair_key.pair,)
        for source, destination in zip(pair[0], pair[1], strict=True)
    }
    logical_bytes = len(logical_mappings) * lane_bytes
    candidates: list[_ScopedMaterializationCandidate] = []
    direct_objects = 0
    direct_work_items = 0
    direct_physical_bytes = 0
    scoped_exact_bytes = 0
    unique_source_bytes = 0
    span_objects = 0
    span_work_items = 0
    span_physical_bytes = 0
    span_exact_bytes = 0
    scratch_cursor = 0
    for tenant_id in sorted(scoped_pairs):
        consumers = scoped_pairs[tenant_id]
        normalized_consumers = tuple(
            (pair_key.pair, tuple(sorted(indices)))
            for pair_key, indices in consumers.items()
        )
        # Keep the direct alternative allocation-light until policy is known.
        # Sparse long-context demand can contain thousands of one-row runs;
        # constructing typed run/group objects for an unselected alternative
        # would put tens of milliseconds back on the TTFT path.
        direct = summarize_nvme_runs(
            (pair for pair, _indices in normalized_consumers),
            lane_element_bytes=row_bytes,
            lba_size=lba_size,
            max_transfer_bytes=max_transfer_bytes,
        )
        run_consumers: dict[tuple[int, int, int], set[int]] = defaultdict(set)
        for (planned_pair, runs), (pair, indices) in zip(
            direct.pair_runs, normalized_consumers, strict=True
        ):
            if planned_pair is not pair and planned_pair != pair:
                raise RuntimeError("NVMe direct summary reordered exact demand")
            for run in runs:
                run_consumers[run].update(indices)
        direct_owners = tuple(
            (run, tuple(sorted(run_consumers[run]))) for run in direct.unique_runs
        )
        if any(not owners for _run, owners in direct_owners):
            raise RuntimeError("NVMe run planning produced an unowned transfer group")
        direct_objects += direct.object_count
        direct_work_items += sum(len(owners) for _run, owners in direct_owners)
        direct_physical_bytes += direct.physical_rows * lane_bytes
        scope_mappings = {
            (source, destination)
            for pair, _indices in normalized_consumers
            for source, destination in zip(pair[0], pair[1], strict=True)
        }
        scoped_exact_bytes += len(scope_mappings) * lane_bytes
        unique_source_bytes += (
            len({source for source, _ in scope_mappings}) * lane_bytes
        )

        candidates.append(
            _ScopedMaterializationCandidate(
                tenant_id,
                normalized_consumers,
                direct,
                direct_owners,
            )
        )

    if direct_physical_bytes < unique_source_bytes:
        raise RuntimeError("direct NVMe planning undercounted unique source bytes")
    direct_feasible = (
        direct_objects <= object_capacity and direct_work_items <= work_ticket_capacity
    )
    direct_lower_bound_ns: int | None = None
    span_lower_bound_ns: int | None = None
    span_planning_warranted = False
    if model.calibrated:
        direct_lower_bound_ns = model.transfer_ns(
            command_count=direct_objects,
            transfer_bytes=direct_physical_bytes,
        )
        span_lower_bound_ns = model.transfer_ns(
            command_count=len(candidates) * len(row_bytes),
            transfer_bytes=unique_source_bytes,
        ) + model.compaction_ns(
            exact_bytes=scoped_exact_bytes,
            launch_count=len(set(row_bytes)),
        )
        span_planning_warranted = not direct_feasible or direct_lower_bound_ns >= ceil(
            span_lower_bound_ns * model.minimum_gain
        )
    if span_planning_warranted:
        planned_candidates: list[_ScopedMaterializationCandidate] = []
        for candidate in candidates:
            span_plan = plan_nvme_spans(
                (pair for pair, _indices in candidate.consumers),
                lane_element_bytes=row_bytes,
                lba_size=lba_size,
                max_transfer_bytes=max_transfer_bytes,
                scratch_alignment=scratch_alignment,
                service_model=model,
            )
            owners_by_span = [set() for _span in span_plan.spans]
            for (planned_pair, span_indices), (pair, indices) in zip(
                span_plan.pair_span_indices, candidate.consumers, strict=True
            ):
                if planned_pair is not pair and planned_pair != pair:
                    raise RuntimeError("NVMe span plan reordered exact demand")
                for span_index in span_indices:
                    owners_by_span[span_index].update(indices)
            span_owners = tuple(
                tuple(sorted(owners)) for owners in owners_by_span
            )
            if any(not owners for owners in span_owners):
                raise RuntimeError("NVMe source span has no request owner")
            scratch_cursor = (
                (scratch_cursor + scratch_alignment - 1) // scratch_alignment
            ) * scratch_alignment
            planned_candidates.append(
                _ScopedMaterializationCandidate(
                    candidate.tenant_id,
                    candidate.consumers,
                    candidate.direct,
                    candidate.direct_owners,
                    span_plan,
                    span_owners,
                    scratch_cursor,
                )
            )
            scratch_cursor += span_plan.scratch_bytes
            span_objects += span_plan.object_count
            span_work_items += sum(len(owners) for owners in span_owners)
            span_physical_bytes += span_plan.physical_bytes
            span_exact_bytes += span_plan.exact_bytes
        candidates = planned_candidates
        if (
            span_exact_bytes != scoped_exact_bytes
            or span_physical_bytes < unique_source_bytes
        ):
            raise RuntimeError("span NVMe planning changed exact materialization")

    decision = (
        NvmeGranularityDecision(
            NvmeGranularity.DIRECT,
            "cost_lower_bound",
            direct_lower_bound_ns,
            span_lower_bound_ns,
        )
        if model.calibrated and not span_planning_warranted
        else choose_nvme_granularity(
            direct_command_count=direct_objects,
            direct_transfer_bytes=direct_physical_bytes,
            direct_work_item_count=direct_work_items,
            span_command_count=span_objects,
            span_transfer_bytes=span_physical_bytes,
            span_exact_bytes=span_exact_bytes,
            span_work_item_count=span_work_items,
            span_scratch_bytes=scratch_cursor,
            compaction_launch_count=len(set(row_bytes)),
            object_capacity=object_capacity,
            work_ticket_capacity=work_ticket_capacity,
            scratch_capacity_bytes=scratch_capacity_bytes,
            service_model=model,
        )
    )
    granularity = decision.kind
    granularity_reason = decision.reason
    selected_objects = direct_objects
    selected_work_items = direct_work_items
    selected_physical_bytes = direct_physical_bytes
    scratch_bytes = 0
    selected_predicted_ns = decision.direct_predicted_ns
    if granularity is NvmeGranularity.SPAN_COMPACT:
        selected_objects = span_objects
        selected_work_items = span_work_items
        selected_physical_bytes = span_physical_bytes
        scratch_bytes = scratch_cursor
        selected_predicted_ns = decision.span_predicted_ns
    if granularity is NvmeGranularity.DIRECT and not direct_feasible:
        reason = (
            "acquisition objects"
            if direct_objects > object_capacity
            else "generation-bound work items"
        )
        raise RuntimeError(
            f"NVMe forward needs more concurrent {reason} than the runtime "
            "directory; increase NTA_RUNTIME_MAX_WORK_TICKETS or calibrate "
            "exact span materialization"
        )
    scopes: list[NvmeScopedMaterializationPlan] = []
    packet_index = 0
    for candidate in candidates:
        if granularity is NvmeGranularity.DIRECT:
            plan = materialize_nvme_run_plan(
                candidate.direct, object_capacity=object_capacity
            )
            run_by_geometry = dict(
                zip(candidate.direct.unique_runs, plan.unique_runs, strict=True)
            )
            packets = tuple(
                NvmeTransferPacketPlan(
                    packet_index + offset,
                    run_by_geometry[geometry],
                    geometry[0],
                    geometry[1],
                    geometry[2],
                    owners,
                )
                for offset, (geometry, owners) in enumerate(candidate.direct_owners)
            )
            packet_index += len(packets)
            scopes.append(
                NvmeScopedMaterializationPlan(candidate.tenant_id, plan, packets)
            )
            continue
        if candidate.span is None or len(candidate.span_owners) != len(
            candidate.span.spans
        ):
            raise RuntimeError("selected NVMe span plan is incomplete")
        packets = tuple(
            NvmeTransferPacketPlan(
                packet_index + offset,
                span,
                span.source_first,
                None,
                span.source_row_count,
                owners,
            )
            for offset, (span, owners) in enumerate(
                zip(candidate.span.spans, candidate.span_owners, strict=True)
            )
        )
        packet_index += len(packets)
        scopes.append(
            NvmeScopedMaterializationPlan(
                candidate.tenant_id,
                candidate.span,
                packets,
                candidate.span_scratch_offset,
            )
        )

    all_packets = tuple(packet for scope in scopes for packet in scope.packets)
    if tuple(packet.packet_index for packet in all_packets) != tuple(
        range(len(all_packets))
    ):
        raise RuntimeError("NVMe transfer packets lost dense forward-local identity")

    exact_groups: list[NvmeExactAcquisitionGroupPlan] = []
    group_by_request_pair: dict[tuple[int, _PagePairKey], int] = {}
    for request_index in sorted(request_pairs):
        segment_begin = 0
        for pair_key in sorted(request_pairs[request_index], key=lambda item: item.pair):
            pair = pair_key.pair
            mappings = set(zip(pair[0], pair[1], strict=True))
            packet_indices: list[int] = []
            covered: set[tuple[int, int]] = set()
            for packet in all_packets:
                if request_index not in packet.consumer_indices:
                    continue
                packet_end = packet.source_first + packet.row_count
                packet_mappings = {
                    (source, destination)
                    for source, destination in mappings
                    if packet.source_first <= source < packet_end
                    and (
                        packet.destination_first is None
                        or destination - packet.destination_first
                        == source - packet.source_first
                    )
                }
                if not packet_mappings:
                    continue
                packet_indices.append(packet.packet_index)
                covered.update(packet_mappings)
            if covered != mappings:
                raise RuntimeError(
                    "NVMe physical packets do not cover one exact demand group"
                )
            group_index = len(exact_groups)
            exact_groups.append(
                NvmeExactAcquisitionGroupPlan(
                    group_index,
                    request_index,
                    segment_begin,
                    len(mappings),
                    tuple(packet_indices),
                )
            )
            group_by_request_pair[(request_index, pair_key)] = group_index
            segment_begin += len(mappings)

    work_group_indices = tuple(
        (
            wrapper_id,
            tuple(
                None
                if pair_key is None
                else group_by_request_pair[(request_index, pair_key)]
                for request_index, pair_key in wrapper_pairs
            ),
        )
        for wrapper_id, wrapper_pairs in work_pairs.items()
    )
    if not exact_groups or any(
        len(groups) != len(work_pairs[wrapper_id])
        for wrapper_id, groups in work_group_indices
    ):
        raise RuntimeError("NVMe exact demand groups do not cover compiler work")
    return NvmeBatchGeometry(
        tuple(scopes),
        tuple(exact_groups),
        work_group_indices,
        tuple(row_bytes),
        selected_objects,
        selected_work_items,
        logical_bytes,
        scoped_exact_bytes,
        unique_source_bytes,
        selected_physical_bytes,
        granularity,
        granularity_reason,
        scratch_bytes,
        decision.direct_predicted_ns,
        selected_predicted_ns,
    )
