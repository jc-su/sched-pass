"""Proactive NVMe acquisition for SGLang attention.

The runtime directory is a transport control structure.  Numerical attention
does not own its NVMe commands and does not keep transient directory slots
alive.  Instead, this owner publishes capacity-bounded cross-layer acquisition
windows on a dedicated progress stream. One discovery builds the window's EDF
order and one persistent completion-driven launch keeps the controller queue
deep. A typed layer-major window uses a device-validated O(1) EDF cursor; any
non-monotonic dynamic image falls back to the generic heap without a host
round trip. Finite progress epochs publish one CUDA event at each layer's
terminal frontier; numerical dispatch waits on that event with no host poll,
per-object driver calls, or resident attention waiter. The compiler-verified
attention plan consumes the ordinary SGLang HBM cache only after that edge.

This separation is what permits cross-layer pipelining: while attention
consumes layer *i*, the NVMe queue can materialize layer *i + 1*.  It also
keeps setup-only VFIO/IOMMU mappings out of the per-forward data path.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from nta_runtime.engines.sglang_contracts import PagePair
from nta_runtime.engines.sglang_acquisition_contract import (
    AcquisitionConsumerPlan,
    AcquisitionTier,
    SglangForwardAcquisition,
    SglangLayerAcquisition,
)
from nta_runtime.execution_topology import (
    ExactWorkTopology,
    RequestWorkTopology,
    WorkDependencySpan,
)
from nta_runtime.flashinfer import object_requirement
from nta_runtime.indexed_transfer import ContiguousPairRun
from nta_runtime.nvme_materialization import (
    NvmeRunPlan,
    NvmeRunPublication,
    NvmeSlotLifetime,
    NvmeTensorLane,
    PreparedNvmeRunPublication,
    plan_nvme_runs,
    prepare_nvme_runs,
    publish_prepared_nvme_runs,
)
from nta_runtime.requests import RequestBinding
from nta_runtime.runtime import DeviceWorkPlan, JitPhaseProgram, Runtime


_OBJECT_ID_BASE = 0x4E54410000000000


@dataclass(frozen=True, slots=True)
class NvmeAcquisitionGroupKey:
    """Stable identity of one K/V run acquired for one request generation."""

    request_slot: int
    generation: int
    tenant_id: int
    layer_id: int
    source_first: int
    destination_first: int
    row_count: int
    resource_version: int


@dataclass(frozen=True, slots=True)
class _ScopedRunPlan:
    """Runs deduplicated within one accounting/isolation scope."""

    tenant_id: int | None
    plan: NvmeRunPlan
    consumers: Mapping[PagePair, tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class NvmeBatchGeometry:
    """Layer-invariant exact source/destination geometry for one forward."""

    scopes: tuple[_ScopedRunPlan, ...]
    row_bytes: tuple[int, int]
    object_count: int
    work_item_count: int
    logical_transfer_bytes: int


@dataclass(frozen=True, slots=True)
class NvmeLayerAcquisition:
    """One layer's transport ownership and completion edge."""

    layer_id: int
    ready_event: torch.cuda.Event
    first_object_slot: int
    object_count: int
    group_count: int
    work_item_count: int
    transfer_bytes: int
    logical_transfer_bytes: int
    resource_version: int
    ready_deadline_offset_ns: int
    groups: tuple[NvmeAcquisitionGroupKey, ...]


@dataclass(frozen=True, slots=True)
class NvmeBatchAcquisition:
    """Immutable layer-ready map installed before the first attention layer."""

    acquisition_id: int
    layers: tuple[NvmeLayerAcquisition, ...]
    window_count: int

    def layer(self, local_layer: int) -> NvmeLayerAcquisition:
        if local_layer < 0 or local_layer >= len(self.layers):
            raise RuntimeError("NVMe attention layer is outside its acquisition")
        layer = self.layers[local_layer]
        if layer.layer_id != self.layers[0].layer_id + local_layer:
            raise RuntimeError("NVMe acquisition layers are not contiguous")
        return layer


@dataclass(slots=True)
class _TransportPlanAllocation:
    plan: DeviceWorkPlan
    work_capacity: int
    dependency_capacity: int


@dataclass(frozen=True, slots=True)
class _PublishedRun:
    """One physical K/V transfer group and all generation-bound consumers."""

    run: ContiguousPairRun
    objects: tuple[Any, ...]
    consumers: tuple[RequestBinding, ...]


@dataclass(frozen=True, slots=True)
class _PublishedLayer:
    """One layer's directory image inside a cross-layer acquisition window."""

    local_layer: int
    layer_id: int
    version: int
    first_object_slot: int
    groups: tuple[_PublishedRun, ...]
    transfer_bytes: int

    @property
    def object_count(self) -> int:
        return sum(len(group.objects) for group in self.groups)


@dataclass(frozen=True, slots=True)
class _PreparedScope:
    scope: _ScopedRunPlan
    publication: PreparedNvmeRunPublication


@dataclass(frozen=True, slots=True)
class _PreparedLayer:
    local_layer: int
    layer_id: int
    version: int
    first_object_slot: int
    scopes: tuple[_PreparedScope, ...]

    @property
    def object_count(self) -> int:
        return sum(scope.publication.object_count for scope in self.scopes)


def _deadline_key(binding: RequestBinding) -> tuple[int, int, int, int]:
    # deadline==0 is explicitly best effort in the native EDF queue.
    return (
        int(binding.deadline_clock == 0),
        binding.deadline_clock if binding.deadline_clock else (1 << 64) - 1,
        -binding.priority,
        binding.request_index,
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
    ):
        raise ValueError("NVMe batch geometry requires non-empty bounded inputs")
    binding_by_index = {binding.request_index: binding for binding in bindings}
    if len(binding_by_index) != len(bindings):
        raise ValueError("NVMe request bindings repeat an engine request index")

    pair_consumers: dict[PagePair, set[int]] = defaultdict(set)
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
                pair_consumers[pair].add(int(request_index))
    if not pair_consumers:
        raise RuntimeError("NVMe external batch contains no physical demand")

    scoped_pairs: dict[int | None, dict[PagePair, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for pair, consumers in pair_consumers.items():
        if tenant_isolation:
            for request_index in consumers:
                tenant = binding_by_index[request_index].tenant_id
                scoped_pairs[tenant][pair].add(request_index)
        else:
            scoped_pairs[None][pair].update(consumers)

    scopes: list[_ScopedRunPlan] = []
    total_objects = 0
    total_work_items = 0
    physical_bytes = 0
    for tenant_id in sorted(
        scoped_pairs, key=lambda value: -1 if value is None else value
    ):
        consumers = scoped_pairs[tenant_id]
        plan = plan_nvme_runs(
            consumers,
            lane_element_bytes=row_bytes,
            lba_size=lba_size,
            max_transfer_bytes=max_transfer_bytes,
            object_capacity=object_capacity,
        )
        total_objects += plan.object_count
        run_consumers: dict[ContiguousPairRun, set[int]] = defaultdict(set)
        for pair, indices in consumers.items():
            for run in plan.runs_for(pair):
                run_consumers[run].update(indices)
        if any(not run_consumers[run] for run in plan.unique_runs):
            raise RuntimeError("NVMe run planning produced an unowned transfer group")
        total_work_items += sum(len(run_consumers[run]) for run in plan.unique_runs)
        physical_bytes += sum(
            run.row_count * sum(row_bytes) for run in plan.unique_runs
        )
        scopes.append(
            _ScopedRunPlan(
                tenant_id,
                plan,
                {pair: tuple(sorted(indices)) for pair, indices in consumers.items()},
            )
        )
    if total_objects > object_capacity:
        raise RuntimeError(
            "NVMe forward needs more concurrent acquisition objects than the "
            "runtime directory; increase NTA_RUNTIME_MAX_WORK_TICKETS"
        )
    if total_work_items > work_ticket_capacity:
        raise RuntimeError(
            "NVMe forward needs more generation-bound work items than the runtime "
            "ticket directory; increase NTA_RUNTIME_MAX_WORK_TICKETS"
        )
    logical_runs = {run for scope in scopes for run in scope.plan.unique_runs}
    logical_bytes = sum(run.row_count * sum(row_bytes) for run in logical_runs)
    if physical_bytes < logical_bytes:
        raise RuntimeError("NVMe isolation accounting undercounted physical bytes")
    return NvmeBatchGeometry(
        tuple(scopes),
        tuple(row_bytes),
        total_objects,
        total_work_items,
        logical_bytes,
    )


class SglangNvmeAcquisitionPipeline:
    """Own proactive cross-layer NVMe transport and its CUDA event frontier."""

    def __init__(
        self,
        *,
        runtime: Runtime,
        tier_service: Any,
        transport_program: Callable[[], JitPhaseProgram],
        progress_stream: torch.cuda.Stream,
        layer_start: int,
        layer_count: int,
        object_capacity: int,
        work_ticket_capacity: int,
        tenant_isolation: bool,
        regions: Mapping[tuple[int, str], Any],
        stats: dict[str, Any],
    ) -> None:
        if (
            min(layer_start, layer_count, object_capacity, work_ticket_capacity) < 0
            or layer_count == 0
            or object_capacity == 0
            or work_ticket_capacity == 0
        ):
            raise ValueError("NVMe acquisition pipeline has invalid capacity")
        if not regions:
            raise ValueError("NVMe acquisition pipeline has no registered HBM regions")
        self._runtime = runtime
        self._tier_service = tier_service
        self._transport_program = transport_program
        self._progress_stream = progress_stream
        self._layer_start = layer_start
        self._layer_count = layer_count
        self._object_capacity = object_capacity
        self._work_ticket_capacity = work_ticket_capacity
        self._intent_capacity = int(runtime.config.intent_capacity)
        self._tenant_isolation = tenant_isolation
        self._regions = dict(regions)
        self._stats = stats
        self._slot_lifetime = NvmeSlotLifetime(torch.cuda.Event())
        self._ready_events = tuple(torch.cuda.Event() for _ in range(layer_count))
        self._binding_event = torch.cuda.Event()
        self._consumer_event = torch.cuda.Event()
        self._consumer_recorded = False
        self._transport_plans: dict[int, _TransportPlanAllocation] = {}
        self._retired_plans: list[DeviceWorkPlan] = []
        self._next_resource_version = 1
        self._next_acquisition_id = 1
        self._active_acquisition_id: int | None = None
        self._waited_layers: set[int] = set()

    def _next_version(self) -> int:
        version = self._next_resource_version
        if version >= (1 << 32) - 1:
            raise RuntimeError("NVMe resource-version space is exhausted")
        self._next_resource_version += 1
        return version

    def _transport_plan(
        self, window_index: int, work_count: int, dependency_count: int
    ) -> DeviceWorkPlan:
        if (
            work_count <= 0
            or work_count > self._work_ticket_capacity
            or dependency_count != 2 * work_count
        ):
            raise RuntimeError(
                "NVMe acquisition-group geometry exceeds runtime capacity"
            )
        existing = self._transport_plans.get(window_index)
        if (
            existing is not None
            and work_count <= existing.work_capacity
            and dependency_count <= existing.dependency_capacity
        ):
            return existing.plan
        if existing is not None:
            # The old plan may still be referenced by already-enqueued work.
            # Retain it until pipeline close instead of introducing a resize
            # synchronization into request admission.
            self._retired_plans.append(existing.plan)
        plan = DeviceWorkPlan(
            work_count, dependency_count, self._runtime.device_ordinal
        )
        self._transport_plans[window_index] = _TransportPlanAllocation(
            plan, work_count, dependency_count
        )
        return plan

    def _prepare_layer(
        self,
        *,
        geometry: NvmeBatchGeometry,
        local_layer: int,
        layer_id: int,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        version: int,
        first_object_slot: int,
    ) -> _PreparedLayer:
        key_cache, value_cache = kv_cache
        if not key_cache.is_cuda or not value_cache.is_cuda:
            raise RuntimeError("NVMe acquisition destinations must be CUDA tensors")
        lanes = (
            NvmeTensorLane(
                "key",
                int(key_cache.data_ptr()),
                int(key_cache.shape[0]),
                geometry.row_bytes[0],
                int(key_cache.stride(0) * key_cache.element_size()),
                self._regions[(layer_id, "key")],
            ),
            NvmeTensorLane(
                "value",
                int(value_cache.data_ptr()),
                int(value_cache.shape[0]),
                geometry.row_bytes[1],
                int(value_cache.stride(0) * value_cache.element_size()),
                self._regions[(layer_id, "value")],
            ),
        )
        prepared: list[_PreparedScope] = []
        first_slot = first_object_slot
        for scope in geometry.scopes:
            publication = prepare_nvme_runs(
                scope.plan,
                lanes,
                extent_resolver=self._tier_service.extent,
                layer_id=layer_id,
                object_version=version,
                object_id_base=_OBJECT_ID_BASE + first_slot,
                first_object_slot=first_slot,
                lifetime=self._slot_lifetime,
            )
            prepared.append(_PreparedScope(scope, publication))
            first_slot += publication.object_count
        if first_slot != first_object_slot + geometry.object_count:
            raise RuntimeError("NVMe layer preparation changed object geometry")
        return _PreparedLayer(
            local_layer,
            layer_id,
            version,
            first_object_slot,
            tuple(prepared),
        )

    def _finalize_layer(
        self,
        *,
        geometry: NvmeBatchGeometry,
        bindings: tuple[RequestBinding, ...],
        prepared: _PreparedLayer,
        publications: tuple[NvmeRunPublication, ...],
    ) -> _PublishedLayer:
        if len(publications) != len(prepared.scopes):
            raise RuntimeError("NVMe layer publication changed its scope count")
        binding_by_index = {binding.request_index: binding for binding in bindings}
        groups: list[_PublishedRun] = []
        transfer_bytes = 0
        for prepared_scope, publication in zip(
            prepared.scopes, publications, strict=True
        ):
            scope = prepared_scope.scope
            counters = publication.counters
            self._stats["nvme_view_publications"] = (
                self._stats.get("nvme_view_publications", 0) + publication.object_count
            )
            for name, value in (
                ("nvme_fresh_slot_installs", counters.fresh_slots),
                ("nvme_same_destination_installs", counters.same_destination_slots),
                ("nvme_destination_rebinds", counters.destination_rebinds),
                (
                    "nvme_object_quiesced_replacements",
                    counters.quiesced_replacements,
                ),
            ):
                self._stats[name] = self._stats.get(name, 0) + value
            by_run = dict(publication.objects_by_run)
            run_consumers: dict[ContiguousPairRun, set[int]] = defaultdict(set)
            for pair, consumers in scope.consumers.items():
                for run in scope.plan.runs_for(pair):
                    run_consumers[run].update(consumers)
            for run in scope.plan.unique_runs:
                consumer_indices = run_consumers[run]
                # Keep one generation-bound work ticket per consumer while all
                # tickets name the same two directory objects.  The object
                # protocol still issues each K/V DMA once, but cancellation or
                # slot reuse of one request cannot invalidate another request's
                # readiness proof.
                consumers = tuple(
                    sorted(
                        (binding_by_index[index] for index in consumer_indices),
                        key=_deadline_key,
                    )
                )
                if not consumers:
                    raise RuntimeError("NVMe acquisition run has no live consumer")
                if scope.tenant_id is not None and any(
                    consumer.tenant_id != scope.tenant_id for consumer in consumers
                ):
                    raise RuntimeError("NVMe isolated run escaped its tenant scope")
                objects = by_run[run]
                if len(objects) != 2:
                    raise RuntimeError("NVMe acquisition group must own one K/V pair")
                groups.append(_PublishedRun(run, objects, consumers))
            transfer_bytes += publication.transfer_bytes
        published = _PublishedLayer(
            prepared.local_layer,
            prepared.layer_id,
            prepared.version,
            prepared.first_object_slot,
            tuple(groups),
            transfer_bytes,
        )
        if published.object_count != geometry.object_count:
            raise RuntimeError("NVMe layer publication changed object geometry")
        return published

    def _upload_transport_window(
        self,
        *,
        window_index: int,
        row_bytes: tuple[int, int],
        layers: tuple[_PublishedLayer, ...],
        inter_layer_compute_ns: int,
    ) -> tuple[
        DeviceWorkPlan,
        Mapping[int, tuple[NvmeAcquisitionGroupKey, ...]],
    ]:
        if not layers or inter_layer_compute_ns <= 0:
            raise RuntimeError("NVMe transport window has invalid deadline geometry")
        grouped: dict[int, list[tuple[_PublishedLayer, _PublishedRun]]] = defaultdict(
            list
        )
        binding_by_request: dict[int, RequestBinding] = {}
        for layer in layers:
            for group in layer.groups:
                for consumer in group.consumers:
                    grouped[consumer.request_index].append((layer, group))
                    binding_by_request[consumer.request_index] = consumer
        dependencies = []
        spans: list[WorkDependencySpan] = []
        logical_work: list[int] = []
        demand_units: list[int] = []
        ready_deadline_offsets: list[int] = []
        requests: list[RequestWorkTopology] = []
        keys_by_layer: dict[int, list[NvmeAcquisitionGroupKey]] = {
            layer.local_layer: [] for layer in layers
        }
        cursor = 0
        first_local_layer = layers[0].local_layer
        for topology_request_index, request_index in enumerate(sorted(grouped)):
            owner = binding_by_request[request_index]
            owned = grouped[request_index]
            requests.append(
                RequestWorkTopology(
                    topology_request_index,
                    cursor,
                    len(owned),
                    owner.request_slot,
                    owner.generation,
                )
            )
            for layer, group in owned:
                begin = len(dependencies)
                for object_ in group.objects:
                    dependencies.append(
                        object_requirement(
                            object_slot=object_.slot,
                            object_id=object_.object_id,
                            object_version=layer.version,
                            bytes=object_.bytes,
                        )
                    )
                spans.append(WorkDependencySpan(begin, len(group.objects), 0))
                logical_work.append(cursor)
                demand_units.append(group.run.row_count)
                # Offset zero means "inherit the request deadline" in the
                # native ABI, so the first layer uses one nanosecond. Relative
                # offsets keep every window in the GPU global-timer domain.
                deadline_offset = 1 + (
                    layer.local_layer - first_local_layer
                ) * inter_layer_compute_ns
                if deadline_offset >= 1 << 64:
                    raise RuntimeError("NVMe layer deadline exceeds the native ABI")
                ready_deadline_offsets.append(deadline_offset)
                key = NvmeAcquisitionGroupKey(
                    owner.request_slot,
                    owner.generation,
                    owner.tenant_id,
                    layer.layer_id,
                    group.run.source_first,
                    group.run.destination_first,
                    group.run.row_count,
                    layer.version,
                )
                keys_by_layer[layer.local_layer].append(key)
                cursor += 1
        expected_work = sum(
            len(group.consumers) for layer in layers for group in layer.groups
        )
        if cursor != expected_work or not dependencies:
            raise RuntimeError("NVMe transport topology omitted an acquisition group")
        topology = ExactWorkTopology(
            layers[0].version,
            tuple(logical_work),
            tuple(demand_units),
            sum(row_bytes),
            (0,) * cursor,
            tuple(requests),
            tuple(ready_deadline_offsets),
        )
        plan = self._transport_plan(window_index, cursor, len(dependencies))
        plan.upload_exact(topology, spans, dependencies, stream=self._progress_stream)
        return plan, {
            local_layer: tuple(keys) for local_layer, keys in keys_by_layer.items()
        }

    def prepare(
        self,
        *,
        semantic_plans: Mapping[int, Any],
        bindings: tuple[RequestBinding, ...],
        ordering_stream: torch.cuda.Stream,
        prepare_consumers: Callable[[torch.cuda.Stream], None],
        kv_cache_for_layer: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
        inter_layer_compute_ns: int,
    ) -> NvmeBatchAcquisition:
        """Enqueue a complete cross-layer producer pipeline without synchronizing."""

        if self._active_acquisition_id is not None:
            raise RuntimeError("the preceding NVMe acquisition has no consumer fence")
        if inter_layer_compute_ns <= 0:
            raise RuntimeError("NVMe acquisition needs positive inter-layer compute")
        first_cache = kv_cache_for_layer(self._layer_start)
        row_bytes = tuple(
            int(tensor[0].numel() * tensor.element_size()) for tensor in first_cache
        )
        geometry = plan_nvme_batch_geometry(
            semantic_plans=semantic_plans,
            bindings=bindings,
            row_bytes=(row_bytes[0], row_bytes[1]),
            lba_size=self._tier_service.nvme_lba_size,
            max_transfer_bytes=self._tier_service.nvme_max_transfer_bytes,
            object_capacity=self._object_capacity,
            work_ticket_capacity=self._work_ticket_capacity,
            tenant_isolation=self._tenant_isolation,
        )

        # Request-directory publication and any preceding numerical consumer
        # are explicit stream edges.  No host synchronization is required.
        self._binding_event.record(ordering_stream)
        if self._consumer_recorded:
            self._progress_stream.wait_event(self._consumer_event)
        self._progress_stream.wait_event(self._binding_event)
        prepare_consumers(self._progress_stream)

        phases = self._transport_program()
        layers: list[NvmeLayerAcquisition] = []
        capacity_layer_limit = min(
            self._object_capacity // geometry.object_count,
            self._intent_capacity // geometry.object_count,
            self._work_ticket_capacity // geometry.work_item_count,
        )
        if capacity_layer_limit <= 0:
            raise RuntimeError("NVMe acquisition cannot fit one layer in the runtime")
        # Keep enough layer-major commands to fill the controller's usable
        # queue, plus one producer lookahead layer while the next window's
        # descriptors are prepared. This is a resource-geometry bound, not a
        # device-specific threshold; large per-layer demand naturally uses a
        # smaller window and fine-grained demand naturally uses a larger one.
        issue_width = max(1, int(self._tier_service.config.issue_budget) - 1)
        queue_fill_layers = (
            issue_width + geometry.object_count - 1
        ) // geometry.object_count
        window_layer_capacity = min(
            capacity_layer_limit,
            self._layer_count,
            queue_fill_layers + (1 if self._layer_count > 1 else 0),
        )
        window_count = (
            self._layer_count + window_layer_capacity - 1
        ) // window_layer_capacity
        for window_index in range(window_count):
            first_local_layer = window_index * window_layer_capacity
            last_local_layer = min(
                self._layer_count, first_local_layer + window_layer_capacity
            )
            prepared_layers: list[_PreparedLayer] = []
            for local_layer in range(first_local_layer, last_local_layer):
                layer_id = self._layer_start + local_layer
                version = self._next_version()
                first_object_slot = (
                    local_layer - first_local_layer
                ) * geometry.object_count
                prepared_layers.append(
                    self._prepare_layer(
                        geometry=geometry,
                        local_layer=local_layer,
                        layer_id=layer_id,
                        kv_cache=kv_cache_for_layer(layer_id),
                        version=version,
                        first_object_slot=first_object_slot,
                    )
                )

            prepared_window = tuple(prepared_layers)
            flat_preparations = tuple(
                scope.publication
                for layer in prepared_window
                for scope in layer.scopes
            )
            flat_publications = publish_prepared_nvme_runs(
                flat_preparations,
                runtime=self._runtime,
                stream=self._progress_stream,
                lifetime=self._slot_lifetime,
            )
            published_layers: list[_PublishedLayer] = []
            publication_cursor = 0
            for prepared in prepared_window:
                scope_count = len(prepared.scopes)
                published_layers.append(
                    self._finalize_layer(
                        geometry=geometry,
                        bindings=bindings,
                        prepared=prepared,
                        publications=flat_publications[
                            publication_cursor : publication_cursor + scope_count
                        ],
                    )
                )
                publication_cursor += scope_count
            if publication_cursor != len(flat_publications):
                raise RuntimeError("NVMe window publication changed its scope image")

            published_window = tuple(published_layers)

            plan, keys_by_layer = self._upload_transport_window(
                window_index=window_index,
                row_bytes=geometry.row_bytes,
                layers=published_window,
                inter_layer_compute_ns=inter_layer_compute_ns,
            )
            object_count = geometry.object_count * len(published_window)
            work_count = geometry.work_item_count * len(published_window)
            phases.reset(
                self._runtime,
                object_count,
                work_count,
                self._progress_stream,
            )
            phases.discover_ordered_nvme(
                self._runtime,
                plan,
                0,
                object_count,
                self._progress_stream,
            )
            plan.mark_consumed(self._progress_stream)
            for published in published_window:
                phases.progress_nvme_ordered_until_range_terminal(
                    self._runtime,
                    0,
                    object_count,
                    published.first_object_slot,
                    geometry.object_count,
                    self._tier_service.config.issue_budget,
                    self._tier_service.config.completion_budget,
                    self._tier_service.config.progress_timeout_ns,
                    self._progress_stream,
                )
                self._ready_events[published.local_layer].record(
                    self._progress_stream
                )
            phases.publish(self._runtime, work_count, self._progress_stream)
            phases.complete(self._runtime, work_count, self._progress_stream)
            self._slot_lifetime.record_retirement(self._progress_stream)

            for published in published_window:
                keys = keys_by_layer[published.local_layer]
                deadline_offset = 1 + (
                    published.local_layer - first_local_layer
                ) * inter_layer_compute_ns
                layers.append(
                    NvmeLayerAcquisition(
                        published.layer_id,
                        self._ready_events[published.local_layer],
                        published.first_object_slot,
                        geometry.object_count,
                        len(published.groups),
                        len(keys),
                        published.transfer_bytes,
                        geometry.logical_transfer_bytes,
                        published.version,
                        deadline_offset,
                        keys,
                    )
                )

        acquisition_id = self._next_acquisition_id
        self._next_acquisition_id += 1
        self._active_acquisition_id = acquisition_id
        self._consumer_recorded = False
        self._waited_layers.clear()
        acquisition = NvmeBatchAcquisition(
            acquisition_id, tuple(layers), window_count
        )
        physical_bytes = sum(layer.transfer_bytes for layer in layers)
        logical_bytes = sum(layer.logical_transfer_bytes for layer in layers)
        self._stats["nvme_pipeline_batches"] = (
            self._stats.get("nvme_pipeline_batches", 0) + 1
        )
        self._stats["nvme_pipeline_layers"] = self._stats.get(
            "nvme_pipeline_layers", 0
        ) + len(layers)
        self._stats["nvme_pipeline_groups"] = self._stats.get(
            "nvme_pipeline_groups", 0
        ) + sum(layer.group_count for layer in layers)
        self._stats["nvme_pipeline_work_items"] = self._stats.get(
            "nvme_pipeline_work_items", 0
        ) + sum(layer.work_item_count for layer in layers)
        self._stats["nvme_pipeline_physical_bytes"] = (
            self._stats.get("nvme_pipeline_physical_bytes", 0) + physical_bytes
        )
        self._stats["nvme_pipeline_logical_bytes"] = (
            self._stats.get("nvme_pipeline_logical_bytes", 0) + logical_bytes
        )
        self._stats["nvme_pipeline_isolation_bytes"] = self._stats.get(
            "nvme_pipeline_isolation_bytes", 0
        ) + (physical_bytes - logical_bytes)
        self._stats["nvme_bytes"] = self._stats.get("nvme_bytes", 0) + physical_bytes
        self._stats["nvme_pipeline_windows"] = self._stats.get(
            "nvme_pipeline_windows", 0
        ) + window_count
        self._stats["nvme_epochs"] = self._stats.get("nvme_epochs", 0) + window_count
        self._stats["nvme_progress_rounds"] = self._stats.get(
            "nvme_progress_rounds", 0
        ) + window_count
        return acquisition

    def wait_layer(
        self,
        acquisition: NvmeBatchAcquisition,
        layer: NvmeLayerAcquisition,
        stream: torch.cuda.Stream,
    ) -> None:
        """Bind one device-published layer terminal state to its consumer."""

        if acquisition.acquisition_id != self._active_acquisition_id:
            raise RuntimeError("NVMe layer wait does not own the active acquisition")
        local_layer = layer.layer_id - acquisition.layers[0].layer_id
        if acquisition.layer(local_layer) is not layer:
            raise RuntimeError("NVMe layer wait uses a foreign acquisition record")
        if local_layer in self._waited_layers:
            raise RuntimeError("NVMe attention layer was consumed more than once")
        stream.wait_event(layer.ready_event)
        self._waited_layers.add(local_layer)

    def record_consumer(
        self, acquisition: NvmeBatchAcquisition, stream: torch.cuda.Stream
    ) -> None:
        """Fence the final numerical consumer before destination reuse."""

        if acquisition.acquisition_id != self._active_acquisition_id:
            raise RuntimeError("NVMe consumer does not own the active acquisition")
        if len(self._waited_layers) != len(acquisition.layers):
            raise RuntimeError("NVMe consumer retired before every layer was bound")
        self._consumer_event.record(stream)
        self._consumer_recorded = True
        self._active_acquisition_id = None
        self._waited_layers.clear()

    def abort(self, acquisition: NvmeBatchAcquisition) -> None:
        """Release an acquisition after the caller has synchronized CUDA."""

        if acquisition.acquisition_id != self._active_acquisition_id:
            return
        self._active_acquisition_id = None
        self._consumer_recorded = False
        self._waited_layers.clear()

    def close(self) -> tuple[BaseException, ...]:
        errors: list[BaseException] = []
        plans = [item.plan for item in self._transport_plans.values()]
        plans.extend(self._retired_plans)
        self._transport_plans.clear()
        self._retired_plans.clear()
        for plan in plans:
            try:
                plan.close()
            except BaseException as error:
                errors.append(error)
        return tuple(errors)


class NvmeForwardAcquisition(SglangForwardAcquisition):
    """Expose proactive NVMe windows through the common consumer contract."""

    def __init__(
        self,
        pipeline: SglangNvmeAcquisitionPipeline,
        acquisition: NvmeBatchAcquisition,
    ) -> None:
        if not acquisition.layers:
            raise ValueError("NVMe forward acquisition has no layers")
        self._pipeline = pipeline
        self._acquisition = acquisition
        self._finished = False

    @property
    def backend_acquisition(self) -> NvmeBatchAcquisition:
        return self._acquisition

    @property
    def tier(self) -> AcquisitionTier:
        return AcquisitionTier.NVME

    def layer(self, local_layer: int) -> SglangLayerAcquisition:
        acquired = self._acquisition.layer(local_layer)
        return SglangLayerAcquisition(
            self,
            local_layer,
            acquired.layer_id,
            acquired.ready_event,
            AcquisitionTier.NVME,
            AcquisitionConsumerPlan.PREACQUIRED,
            backend_record=acquired,
        )

    def consume_layer(
        self,
        layer: SglangLayerAcquisition,
        stream: torch.cuda.Stream,
        *,
        wait_for_ready: bool,
    ) -> None:
        if self._finished or layer.owner is not self:
            raise RuntimeError("NVMe consumer uses an inactive acquisition")
        if not wait_for_ready:
            raise RuntimeError("NVMe layer has no partial consumer publication")
        acquired = layer.backend_record
        if not isinstance(acquired, NvmeLayerAcquisition):
            raise RuntimeError("NVMe consumer lost its backend layer record")
        self._pipeline.wait_layer(self._acquisition, acquired, stream)

    def finish(self, stream: torch.cuda.Stream) -> None:
        if self._finished:
            raise RuntimeError("NVMe acquisition was finished more than once")
        self._pipeline.record_consumer(self._acquisition, stream)
        self._finished = True

    def abort_after_quiescence(self) -> None:
        if self._finished:
            return
        self._pipeline.abort(self._acquisition)
        self._finished = True
