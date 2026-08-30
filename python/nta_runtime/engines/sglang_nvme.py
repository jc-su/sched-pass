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
import time
from typing import Any

import torch

from nta_runtime.engines.sglang_acquisition_contract import (
    AcquisitionConsumerPlan,
    AcquisitionTier,
    SglangForwardAcquisition,
    SglangLayerAcquisition,
)
from nta_runtime.engines.sglang_nvme_planning import (
    NvmeAcquisitionGroupPlan,
    NvmeBatchGeometry,
    NvmeScopedMaterializationPlan,
    plan_nvme_batch_geometry,
    plan_nvme_window_layer_capacity,
)
from nta_runtime.execution_topology import (
    ExactWorkTopology,
    RequestWorkTopology,
    WorkDependencySpan,
)
from nta_runtime.flashinfer import object_requirement
from nta_runtime.indexed_transfer import ContiguousPairRun
from nta_runtime.nvme_materialization import (
    NvmeScratchArena,
    NvmeRunPlan,
    NvmeRunPublication,
    NvmeSpanPublication,
    NvmeSlotLifetime,
    NvmeTensorLane,
    PreparedNvmeRunPublication,
    PreparedNvmeSpanPublication,
    prepare_nvme_runs,
    prepare_nvme_spans,
    publish_prepared_nvme_runs,
    publish_prepared_nvme_spans,
)
from nta_runtime.nvme_granularity import (
    NvmeGranularity,
    NvmeSourceSpan,
    NvmeSpanPlan,
    NvmeTransferServiceModel,
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
    destination_first: int | None
    row_count: int
    resource_version: int


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
    compaction_images: tuple[_CompactionImage, ...] = ()

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
class _PublishedGroup:
    """One physical K/V transfer group and all generation-bound consumers."""

    source_first: int
    destination_first: int | None
    row_count: int
    objects: tuple[Any, ...]
    consumers: tuple[RequestBinding, ...]


@dataclass(frozen=True, slots=True)
class _PublishedLayer:
    """One layer's directory image inside a cross-layer acquisition window."""

    local_layer: int
    layer_id: int
    version: int
    first_object_slot: int
    groups: tuple[_PublishedGroup, ...]
    transfer_bytes: int

    @property
    def object_count(self) -> int:
        return sum(len(group.objects) for group in self.groups)


@dataclass(frozen=True, slots=True)
class _PreparedScope:
    scope: NvmeScopedMaterializationPlan
    publication: PreparedNvmeRunPublication | PreparedNvmeSpanPublication


@dataclass(frozen=True, slots=True)
class _CompactionLaunch:
    local_layer: int
    address_offset: int
    row_count: int
    row_bytes: int


@dataclass(frozen=True, slots=True)
class _CompactionImage:
    rows: torch.Tensor
    launches: tuple[_CompactionLaunch, ...]


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
        scratch: NvmeScratchArena | None = None,
        stats: dict[str, Any],
        window_layer_limit: int | None = None,
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
        if window_layer_limit is not None and (
            window_layer_limit <= 0 or window_layer_limit > layer_count
        ):
            raise ValueError("NVMe acquisition window limit exceeds model layers")
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
        self._scratch = scratch
        configured_model = getattr(
            self._tier_service.config, "nvme_service_model", None
        )
        self._service_model = (
            configured_model
            if isinstance(configured_model, NvmeTransferServiceModel)
            else NvmeTransferServiceModel()
        )
        self._scratch_alignment = int(
            getattr(self._tier_service, "nvme_controller_page_size", 4096)
        )
        self._stats = stats
        self._window_layer_limit = window_layer_limit
        self._slot_lifetime = NvmeSlotLifetime(torch.cuda.Event())
        self._ready_events = tuple(torch.cuda.Event() for _ in range(layer_count))
        # A stream-memory wait consumes no SM residency.  Keep it separate from
        # the transport worker so one ordered worker can keep the controller
        # busy while each transformer layer independently publishes a CUDA-
        # visible ready event to its numerical consumer.
        self._readiness_stream = torch.cuda.Stream(device=runtime.device_ordinal)
        self._window_armed_events = tuple(
            torch.cuda.Event() for _ in range(layer_count)
        )
        self._window_observed_events = tuple(
            torch.cuda.Event() for _ in range(layer_count)
        )
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

    @staticmethod
    def _bind_group_consumers(
        geometry: NvmeBatchGeometry,
        bindings: tuple[RequestBinding, ...],
    ) -> Mapping[NvmeAcquisitionGroupPlan, tuple[RequestBinding, ...]]:
        binding_by_index = {binding.request_index: binding for binding in bindings}
        if len(binding_by_index) != len(bindings):
            raise RuntimeError("NVMe forward repeats an engine request index")
        result: dict[NvmeAcquisitionGroupPlan, tuple[RequestBinding, ...]] = {}
        for scope in geometry.scopes:
            for group_plan in scope.groups:
                consumers = tuple(
                    sorted(
                        (
                            binding_by_index[index]
                            for index in group_plan.consumer_indices
                        ),
                        key=_deadline_key,
                    )
                )
                if not consumers:
                    raise RuntimeError("NVMe acquisition group has no live consumer")
                if scope.tenant_id is not None and any(
                    consumer.tenant_id != scope.tenant_id for consumer in consumers
                ):
                    raise RuntimeError("NVMe acquisition escaped its tenant scope")
                result[group_plan] = consumers
        return result

    def _prepare_layer(
        self,
        *,
        geometry: NvmeBatchGeometry,
        local_layer: int,
        layer_id: int,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        version: int,
        first_object_slot: int,
        scratch_base: int,
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
            if geometry.granularity is NvmeGranularity.DIRECT:
                if not isinstance(scope.plan, NvmeRunPlan):
                    raise RuntimeError("direct materialization lost its run plan")
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
            else:
                if self._scratch is None or not isinstance(scope.plan, NvmeSpanPlan):
                    raise RuntimeError(
                        "span materialization lost its registered scratch owner"
                    )
                publication = prepare_nvme_spans(
                    scope.plan,
                    lanes,
                    scratch=self._scratch,
                    scratch_base=scratch_base + scope.span_scratch_offset,
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
        bound_consumers: Mapping[
            NvmeAcquisitionGroupPlan, tuple[RequestBinding, ...]
        ],
        prepared: _PreparedLayer,
        publications: tuple[NvmeRunPublication | NvmeSpanPublication, ...],
    ) -> _PublishedLayer:
        if len(publications) != len(prepared.scopes):
            raise RuntimeError("NVMe layer publication changed its scope count")
        groups: list[_PublishedGroup] = []
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
            if geometry.granularity is NvmeGranularity.DIRECT:
                if not isinstance(publication, NvmeRunPublication):
                    raise RuntimeError("direct NVMe scope published a source span")
            else:
                if not isinstance(publication, NvmeSpanPublication):
                    raise RuntimeError("span NVMe scope published a direct run")
            for group_plan in scope.groups:
                # Keep one generation-bound work ticket per consumer while all
                # tickets name the same two directory objects.  The object
                # protocol still issues each K/V DMA once, but cancellation or
                # slot reuse of one request cannot invalidate another request's
                # readiness proof.
                consumers = bound_consumers[group_plan]
                if isinstance(publication, NvmeRunPublication):
                    if not isinstance(group_plan.materialization, ContiguousPairRun):
                        raise RuntimeError(
                            "direct publication lost its transfer-run identity"
                        )
                    objects = publication.objects_for_run(group_plan.materialization)
                else:
                    if not isinstance(group_plan.materialization, NvmeSourceSpan):
                        raise RuntimeError(
                            "span publication lost its source-span identity"
                        )
                    objects = publication.objects_for(group_plan.materialization)
                if len(objects) != 2:
                    raise RuntimeError("NVMe acquisition group must own one K/V pair")
                groups.append(
                    _PublishedGroup(
                        group_plan.source_first,
                        group_plan.destination_first,
                        group_plan.row_count,
                        objects,
                        consumers,
                    )
                )
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

    def _build_compaction_image(
        self,
        *,
        publications_by_layer: tuple[
            tuple[NvmeRunPublication | NvmeSpanPublication, ...], ...
        ],
        first_local_layer: int,
    ) -> _CompactionImage | None:
        """Build one immutable device address image for a transport window."""

        rows: list[tuple[int, int, int]] = []
        launches: list[_CompactionLaunch] = []
        for layer_offset, publications in enumerate(publications_by_layer):
            by_row_bytes: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
            for publication in publications:
                if not isinstance(publication, NvmeSpanPublication):
                    if isinstance(publication, NvmeRunPublication):
                        continue
                    raise RuntimeError("NVMe publication has an unknown exact form")
                for row_bytes, addresses in publication.compaction_addresses:
                    by_row_bytes[row_bytes].extend(addresses)
            for row_bytes, addresses in by_row_bytes.items():
                if not addresses:
                    raise RuntimeError("NVMe span produced an empty compaction wave")
                address_offset = len(rows)
                rows.extend(addresses)
                launches.append(
                    _CompactionLaunch(
                        first_local_layer + layer_offset,
                        address_offset,
                        len(addresses),
                        row_bytes,
                    )
                )
        if not launches:
            return None
        device = torch.device("cuda", self._runtime.device_ordinal)
        with torch.cuda.stream(self._progress_stream):
            row_image = torch.tensor(
                rows,
                dtype=torch.int64,
                device=device,
            )
        row_image.record_stream(self._readiness_stream)
        return _CompactionImage(
            row_image,
            tuple(launches),
        )

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
        grouped: dict[int, list[tuple[_PublishedLayer, _PublishedGroup]]] = defaultdict(
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
                demand_units.append(group.row_count)
                # Offset zero means "inherit the request deadline" in the
                # native ABI, so the first layer uses one nanosecond. Relative
                # offsets keep every window in the GPU global-timer domain.
                deadline_offset = (
                    1 + (layer.local_layer - first_local_layer) * inter_layer_compute_ns
                )
                if deadline_offset >= 1 << 64:
                    raise RuntimeError("NVMe layer deadline exceeds the native ABI")
                ready_deadline_offsets.append(deadline_offset)
                key = NvmeAcquisitionGroupKey(
                    owner.request_slot,
                    owner.generation,
                    owner.tenant_id,
                    layer.layer_id,
                    group.source_first,
                    group.destination_first,
                    group.row_count,
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

        prepare_started_ns = time.perf_counter_ns()
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
            service_model=self._service_model,
            scratch_capacity_bytes=(
                0 if self._scratch is None else self._scratch.bytes
            ),
            scratch_alignment=self._scratch_alignment,
        )
        bound_consumers = self._bind_group_consumers(geometry, bindings)
        geometry_finished_ns = time.perf_counter_ns()
        descriptor_cpu_ns = 0
        publication_cpu_ns = 0
        finalization_cpu_ns = 0
        topology_cpu_ns = 0

        # Request-directory publication and any preceding numerical consumer
        # are explicit stream edges.  No host synchronization is required.
        self._binding_event.record(ordering_stream)
        if self._consumer_recorded:
            self._progress_stream.wait_event(self._consumer_event)
        self._progress_stream.wait_event(self._binding_event)
        prepare_consumers(self._progress_stream)

        phases = self._transport_program()
        layers: list[NvmeLayerAcquisition] = []
        compaction_images: list[_CompactionImage] = []
        capacity_layer_limit = min(
            self._object_capacity // geometry.object_count,
            self._intent_capacity // geometry.object_count,
            self._work_ticket_capacity // geometry.work_item_count,
        )
        if geometry.granularity is NvmeGranularity.SPAN_COMPACT:
            if self._scratch is None or geometry.scratch_bytes_per_layer <= 0:
                raise RuntimeError("span materialization has no bounded scratch arena")
            capacity_layer_limit = min(
                capacity_layer_limit,
                self._scratch.bytes // geometry.scratch_bytes_per_layer,
            )
        if capacity_layer_limit <= 0:
            raise RuntimeError("NVMe acquisition cannot fit one layer in the runtime")
        window_layer_capacity = plan_nvme_window_layer_capacity(
            layer_count=self._layer_count,
            objects_per_layer=geometry.object_count,
            capacity_layer_limit=capacity_layer_limit,
            queue_depth=int(self._tier_service.config.queue_depth),
            explicit_limit=self._window_layer_limit,
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
            stage_started_ns = time.perf_counter_ns()
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
                        scratch_base=(local_layer - first_local_layer)
                        * geometry.scratch_bytes_per_layer,
                    )
                )
            descriptor_cpu_ns += time.perf_counter_ns() - stage_started_ns

            prepared_window = tuple(prepared_layers)
            flat_preparations = tuple(
                scope.publication for layer in prepared_window for scope in layer.scopes
            )
            stage_started_ns = time.perf_counter_ns()
            if geometry.granularity is NvmeGranularity.DIRECT:
                if not all(
                    isinstance(item, PreparedNvmeRunPublication)
                    for item in flat_preparations
                ):
                    raise RuntimeError("direct NVMe preparation changed form")
                flat_publications = publish_prepared_nvme_runs(
                    flat_preparations,
                    runtime=self._runtime,
                    stream=self._progress_stream,
                    lifetime=self._slot_lifetime,
                )
            else:
                if not all(
                    isinstance(item, PreparedNvmeSpanPublication)
                    for item in flat_preparations
                ):
                    raise RuntimeError("span NVMe preparation changed form")
                flat_publications = publish_prepared_nvme_spans(
                    flat_preparations,
                    runtime=self._runtime,
                    stream=self._progress_stream,
                    lifetime=self._slot_lifetime,
                )
            publication_cpu_ns += time.perf_counter_ns() - stage_started_ns
            published_layers: list[_PublishedLayer] = []
            publications_by_layer: list[
                tuple[NvmeRunPublication | NvmeSpanPublication, ...]
            ] = []
            publication_cursor = 0
            stage_started_ns = time.perf_counter_ns()
            for prepared in prepared_window:
                scope_count = len(prepared.scopes)
                layer_publications = flat_publications[
                    publication_cursor : publication_cursor + scope_count
                ]
                publications_by_layer.append(layer_publications)
                published_layers.append(
                    self._finalize_layer(
                        geometry=geometry,
                        bound_consumers=bound_consumers,
                        prepared=prepared,
                        publications=layer_publications,
                    )
                )
                publication_cursor += scope_count
            if publication_cursor != len(flat_publications):
                raise RuntimeError("NVMe window publication changed its scope image")

            published_window = tuple(published_layers)
            compaction_image = self._build_compaction_image(
                publications_by_layer=tuple(publications_by_layer),
                first_local_layer=first_local_layer,
            )
            if geometry.granularity is NvmeGranularity.SPAN_COMPACT:
                if compaction_image is None:
                    raise RuntimeError("span NVMe window lost its compaction image")
                compaction_images.append(compaction_image)
            elif compaction_image is not None:
                raise RuntimeError("direct NVMe window acquired scratch compaction")
            finalization_cpu_ns += time.perf_counter_ns() - stage_started_ns

            stage_started_ns = time.perf_counter_ns()
            plan, keys_by_layer = self._upload_transport_window(
                window_index=window_index,
                row_bytes=geometry.row_bytes,
                layers=published_window,
                inter_layer_compute_ns=inter_layer_compute_ns,
            )
            topology_cpu_ns += time.perf_counter_ns() - stage_started_ns
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

            # Reset/discovery must become CUDA-visible before terminal waits
            # inspect reused directory slots; otherwise an old Ready value can
            # satisfy a new acquisition (ABA).  One persistent ordered worker
            # then advances the complete window.  Batched stream-memory waits
            # publish true per-layer readiness without a resident polling CTA.
            armed_event = self._window_armed_events[window_index]
            observed_event = self._window_observed_events[window_index]
            armed_event.record(self._progress_stream)
            phases.progress_nvme_ordered_until_idle(
                self._runtime,
                0,
                object_count,
                self._tier_service.config.issue_budget,
                self._tier_service.config.completion_budget,
                self._tier_service.config.progress_timeout_ns,
                self._progress_stream,
            )
            self._readiness_stream.wait_event(armed_event)
            launches_by_layer: dict[int, list[_CompactionLaunch]] = defaultdict(list)
            if compaction_image is not None:
                for launch in compaction_image.launches:
                    launches_by_layer[launch.local_layer].append(launch)
            for published in published_window:
                self._runtime.wait_object_range_terminal(
                    published.first_object_slot,
                    geometry.object_count,
                    self._readiness_stream,
                )
                layer_launches = launches_by_layer[published.local_layer]
                if compaction_image is not None and not layer_launches:
                    raise RuntimeError("span NVMe layer lost its compaction launch")
                if not layer_launches:
                    phases.require_ready_objects(
                        self._runtime,
                        published.first_object_slot,
                        geometry.object_count,
                        self._readiness_stream,
                    )
                for launch in layer_launches:
                    phases.compact_ready_hbm_rows(
                        self._runtime,
                        compaction_image.rows,
                        launch.row_bytes,
                        self._readiness_stream,
                        first_row=launch.address_offset,
                        row_count=launch.row_count,
                    )
                self._ready_events[published.local_layer].record(self._readiness_stream)
            observed_event.record(self._readiness_stream)
            # Directory slots may be reused by the next window only after its
            # terminal states have been captured in layer-owned CUDA events.
            self._progress_stream.wait_event(observed_event)
            phases.publish(self._runtime, work_count, self._progress_stream)
            phases.complete(self._runtime, work_count, self._progress_stream)
            self._slot_lifetime.record_retirement(self._progress_stream)

            for published in published_window:
                keys = keys_by_layer[published.local_layer]
                deadline_offset = (
                    1
                    + (published.local_layer - first_local_layer)
                    * inter_layer_compute_ns
                )
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
            acquisition_id,
            tuple(layers),
            window_count,
            tuple(compaction_images),
        )
        physical_bytes = sum(layer.transfer_bytes for layer in layers)
        logical_bytes = sum(layer.logical_transfer_bytes for layer in layers)
        scoped_exact_bytes = geometry.scoped_exact_transfer_bytes * len(layers)
        unique_source_bytes = geometry.unique_source_transfer_bytes * len(layers)
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
        ) + (scoped_exact_bytes - logical_bytes)
        self._stats["nvme_pipeline_unique_source_bytes"] = (
            self._stats.get("nvme_pipeline_unique_source_bytes", 0)
            + unique_source_bytes
        )
        self._stats["nvme_pipeline_extra_source_bytes"] = self._stats.get(
            "nvme_pipeline_extra_source_bytes", 0
        ) + (physical_bytes - unique_source_bytes)
        granularity_key = f"nvme_granularity_{geometry.granularity.value}_batches"
        self._stats[granularity_key] = self._stats.get(granularity_key, 0) + 1
        reason_key = f"nvme_granularity_reason_{geometry.granularity_reason}"
        self._stats[reason_key] = self._stats.get(reason_key, 0) + 1
        if geometry.direct_predicted_ns is not None:
            self._stats["nvme_direct_predicted_ns"] = self._stats.get(
                "nvme_direct_predicted_ns", 0
            ) + geometry.direct_predicted_ns * len(layers)
        if geometry.selected_predicted_ns is not None:
            self._stats["nvme_selected_predicted_ns"] = self._stats.get(
                "nvme_selected_predicted_ns", 0
            ) + geometry.selected_predicted_ns * len(layers)
        if compaction_images:
            compaction_rows = sum(
                launch.row_count
                for image in compaction_images
                for launch in image.launches
            )
            self._stats["nvme_compaction_rows"] = (
                self._stats.get("nvme_compaction_rows", 0) + compaction_rows
            )
            self._stats["nvme_compaction_launches"] = self._stats.get(
                "nvme_compaction_launches", 0
            ) + sum(len(image.launches) for image in compaction_images)
            self._stats["nvme_scratch_high_water_bytes"] = max(
                self._stats.get("nvme_scratch_high_water_bytes", 0),
                geometry.scratch_bytes_per_layer
                * min(window_layer_capacity, len(layers)),
            )
        self._stats["nvme_bytes"] = self._stats.get("nvme_bytes", 0) + physical_bytes
        self._stats["nvme_pipeline_windows"] = (
            self._stats.get("nvme_pipeline_windows", 0) + window_count
        )
        self._stats["nvme_epochs"] = self._stats.get("nvme_epochs", 0) + window_count
        self._stats["nvme_progress_rounds"] = (
            self._stats.get("nvme_progress_rounds", 0) + window_count
        )
        for name, value in (
            (
                "nvme_geometry_cpu_ns",
                geometry_finished_ns - prepare_started_ns,
            ),
            ("nvme_descriptor_cpu_ns", descriptor_cpu_ns),
            ("nvme_publication_cpu_ns", publication_cpu_ns),
            ("nvme_finalization_cpu_ns", finalization_cpu_ns),
            ("nvme_topology_cpu_ns", topology_cpu_ns),
            (
                "nvme_prepare_cpu_ns",
                time.perf_counter_ns() - prepare_started_ns,
            ),
        ):
            self._stats[name] = self._stats.get(name, 0) + value
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
