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
class NvmeAcquisitionGroupPlan:
    """One physical K/V object pair and its generation-bound consumers."""

    materialization: ContiguousPairRun | NvmeSourceSpan
    source_first: int
    destination_first: int | None
    row_count: int
    consumer_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.source_first < 0
            or (self.destination_first is not None and self.destination_first < 0)
            or self.row_count <= 0
            or not self.consumer_indices
            or tuple(sorted(set(self.consumer_indices))) != self.consumer_indices
        ):
            raise ValueError("NVMe acquisition group has invalid exact ownership")


@dataclass(frozen=True, slots=True)
class NvmeScopedMaterializationPlan:
    """Materialization deduplicated within one accounting/isolation scope."""

    tenant_id: int | None
    plan: NvmeRunPlan | NvmeSpanPlan
    groups: tuple[NvmeAcquisitionGroupPlan, ...]
    span_scratch_offset: int = 0


@dataclass(frozen=True, slots=True)
class _ScopedMaterializationCandidate:
    """Allocation-light alternatives retained until policy is decided."""

    tenant_id: int | None
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
    """Layer-invariant exact source/destination geometry for one forward."""

    scopes: tuple[NvmeScopedMaterializationPlan, ...]
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

    Without tenant isolation, identical runs are transferred once across the
    batch.  With isolation, deduplication is scoped to a tenant: this prevents
    one tenant from consuming another tenant's finite byte budget.  Cross-
    tenant sharing may therefore duplicate physical bytes, an explicit and
    conservative isolation cost rather than nondeterministic first-CTA
    charging.
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
    binding_by_index = {binding.request_index: binding for binding in bindings}
    if len(binding_by_index) != len(bindings):
        raise ValueError("NVMe request bindings repeat an engine request index")

    pair_consumers: dict[_PagePairKey, set[int]] = defaultdict(set)
    for semantic in semantic_plans.values():
        if getattr(semantic, "dependency_kind", None) != "physical_pages":
            raise RuntimeError("NVMe acquisition requires physical-page semantics")
        schedule = semantic.schedule
        if len(schedule.request_indices) != len(semantic.page_pairs):
            raise RuntimeError("NVMe schedule and physical-page demand disagree")
        for request_index, pair in zip(
            schedule.request_indices, semantic.page_pairs, strict=True
        ):
            if request_index not in binding_by_index:
                raise RuntimeError("NVMe work references an unbound request")
            if pair[0]:
                if len(pair[0]) != len(pair[1]):
                    raise RuntimeError("NVMe source/destination pair is malformed")
                pair_consumers[_PagePairKey(pair)].add(int(request_index))
    if not pair_consumers:
        raise RuntimeError("NVMe external batch contains no physical demand")

    scoped_pairs: dict[int | None, dict[_PagePairKey, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for pair_key, consumers in pair_consumers.items():
        if tenant_isolation:
            for request_index in consumers:
                tenant = binding_by_index[request_index].tenant_id
                scoped_pairs[tenant][pair_key].add(request_index)
        else:
            scoped_pairs[None][pair_key].update(consumers)

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
    for tenant_id in sorted(
        scoped_pairs, key=lambda value: -1 if value is None else value
    ):
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
    for candidate in candidates:
        if granularity is NvmeGranularity.DIRECT:
            plan = materialize_nvme_run_plan(
                candidate.direct, object_capacity=object_capacity
            )
            run_by_geometry = dict(
                zip(candidate.direct.unique_runs, plan.unique_runs, strict=True)
            )
            groups = tuple(
                NvmeAcquisitionGroupPlan(
                    run_by_geometry[geometry],
                    geometry[0],
                    geometry[1],
                    geometry[2],
                    owners,
                )
                for geometry, owners in candidate.direct_owners
            )
            scopes.append(
                NvmeScopedMaterializationPlan(candidate.tenant_id, plan, groups)
            )
            continue
        if candidate.span is None or len(candidate.span_owners) != len(
            candidate.span.spans
        ):
            raise RuntimeError("selected NVMe span plan is incomplete")
        groups = tuple(
            NvmeAcquisitionGroupPlan(
                span,
                span.source_first,
                None,
                span.source_row_count,
                owners,
            )
            for span, owners in zip(
                candidate.span.spans, candidate.span_owners, strict=True
            )
        )
        scopes.append(
            NvmeScopedMaterializationPlan(
                candidate.tenant_id,
                candidate.span,
                groups,
                candidate.span_scratch_offset,
            )
        )
    return NvmeBatchGeometry(
        tuple(scopes),
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
