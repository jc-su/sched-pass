"""Pure semantic planning for the SGLang FlashInfer integration.

This module translates a framework snapshot into immutable work topology and
execution decisions.  It owns no CUDA stream, runtime directory slot, HiCache
lease lifetime, numerical wrapper, or telemetry sink.  Those ownership
boundaries remain in the framework backend and the physical materializer.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from nta_runtime.adapters.base import EngineBatch
from nta_runtime.engines.sglang_contracts import (
    LeaseAcquisitionGroup,
    LeaseAcquisitionSlice,
    PagePair,
)
from nta_runtime.engines.sglang_hicache import PendingHostLoad
from nta_runtime.engines.sglang_planning import semantic_plan_signature_prefix
from nta_runtime.engines.sglang_state import _SemanticWrapperPlan
from nta_runtime.engines.sglang_topology import (
    lease_acquisition_topology,
    page_pairs_for_schedule,
)
from nta_runtime.execution_core import ExecutionPlan, ExecutionTile
from nta_runtime.execution_planner import (
    HostCostModel,
    HostExecutionMode,
    HostExecutionPlan,
    plan_host_execution,
    prove_atomic_host_execution,
)
from nta_runtime.execution_protocol import ExecutionProtocolConfig
from nta_runtime.execution_topology import ExactWorkTopology
from nta_runtime.flashinfer_schedule import Schedule
from nta_runtime.requests import RequestBinding


def validate_schedule(schedule: Schedule, bindings: tuple[RequestBinding, ...]) -> None:
    """Prove the request-contiguous FlashInfer coordinate contract."""

    if schedule.work_count <= 0:
        raise RuntimeError("FlashInfer emitted no active CTA work")
    if schedule.work_count != len(schedule.kv_tile_indices):
        raise RuntimeError("FlashInfer schedule identity arrays disagree")
    cursor = 0
    for request_index in range(len(bindings)):
        begin = cursor
        while (
            cursor < schedule.work_count
            and schedule.request_indices[cursor] == request_index
        ):
            cursor += 1
        if cursor == begin:
            raise RuntimeError(
                f"FlashInfer emitted no CTA work for request {request_index}"
            )
    if cursor != schedule.work_count:
        raise RuntimeError("FlashInfer CTA work is not request-contiguous")


def wrapper_page_layout(
    wrapper: Any, *, default_page_size: int
) -> tuple[list[int], list[int], list[int], int]:
    """Capture one forward-scoped page-table snapshot from FlashInfer."""

    batch_size = int(wrapper._batch_size)
    indptr = (
        wrapper._paged_kv_indptr_buf[: batch_size + 1]
        .detach()
        .to(device="cpu")
        .tolist()
    )
    page_count = int(indptr[-1])
    pages = (
        wrapper._paged_kv_indices_buf[:page_count].detach().to(device="cpu").tolist()
    )
    last_page = (
        wrapper._paged_kv_last_page_len_buf[:batch_size]
        .detach()
        .to(device="cpu")
        .tolist()
    )
    page_size = int(getattr(wrapper, "_page_size", default_page_size))
    if page_size != 1:
        raise RuntimeError("NTA's SGLang HiCache path currently requires page_size=1")
    return indptr, pages, last_page, page_size


def work_page_pairs(
    schedule: Schedule,
    pending: PendingHostLoad,
    *,
    layout: tuple[list[int], list[int], list[int], int],
    host_staged: bool,
    physical_catalog: Any | None,
) -> tuple[PagePair, ...]:
    """Resolve exact work demand against one lease-owned source map."""

    indptr, pages, last_page, page_size = layout
    if host_staged:
        source_by_device = pending.materialize_mapping()
    else:
        if physical_catalog is None:
            raise RuntimeError("physical HiCache load has no stable-key catalog")
        source_by_device = pending.materialize_storage_mapping(physical_catalog)
    return page_pairs_for_schedule(
        schedule,
        indptr=indptr,
        pages=pages,
        last_page=last_page,
        page_size=page_size,
        source_by_device=source_by_device,
    )


def build_execution_plan(
    *,
    engine_batch: EngineBatch,
    protocol: ExecutionProtocolConfig,
    tile_compute_ns: int,
    bindings: tuple[RequestBinding, ...],
    schedule: Schedule,
    page_pairs: tuple[PagePair, ...],
    acquisition_slices: tuple[LeaseAcquisitionSlice | None, ...],
    layer: int,
    unit_bytes: int,
) -> ExecutionPlan:
    """Translate one native launch into the executable specification."""

    tiles: list[ExecutionTile] = []
    contributor_counts = Counter(schedule.request_indices)
    contributor_indices = {request: 0 for request in contributor_counts}
    if acquisition_slices:
        if len(acquisition_slices) != schedule.work_count:
            raise RuntimeError("execution dependencies do not match CTA schedule")
    elif page_pairs and len(page_pairs) != schedule.work_count:
        raise RuntimeError("execution page pairs do not match CTA schedule")
    for work_id, request_index in enumerate(schedule.request_indices):
        if request_index < 0 or request_index >= len(bindings):
            raise RuntimeError("FlashInfer schedule referenced an invalid request")
        external_rows = (
            acquisition_slices[work_id].row_count
            if acquisition_slices and acquisition_slices[work_id] is not None
            else len(page_pairs[work_id][0])
            if page_pairs
            else 0
        )
        tiles.append(
            ExecutionTile(
                work_id=work_id,
                binding=bindings[request_index],
                layer=layer,
                logical_begin=int(schedule.kv_tile_indices[work_id]),
                candidate_units=max(1, external_rows),
                selected_ids=(),
                unit_bytes=unit_bytes,
                ready=external_rows == 0,
                estimated_compute_ns=tile_compute_ns,
                reduction_group=request_index,
                contributor_index=contributor_indices[request_index],
                contributor_count=contributor_counts[request_index],
            )
        )
        contributor_indices[request_index] += 1
    if not tiles:
        raise RuntimeError("FlashInfer produced no execution work units")
    return ExecutionPlan.from_tiles(
        epoch=engine_batch.epoch,
        granularity=protocol.granularity,
        protocol=protocol,
        tiles=tiles,
    )


def build_semantic_wrapper_plan(
    *,
    engine_batch: EngineBatch,
    tile_compute_ns: int,
    bindings: tuple[RequestBinding, ...],
    schedule: Schedule,
    pending: PendingHostLoad,
    dependency_kind: str,
    page_pairs: tuple[PagePair, ...] = (),
    acquisition_slices: tuple[LeaseAcquisitionSlice | None, ...] = (),
    acquisition_groups: tuple[LeaseAcquisitionGroup | None, ...] = (),
) -> _SemanticWrapperPlan:
    """Build the layer-invariant exact plan before numerical execution."""

    if engine_batch.bindings != bindings:
        raise RuntimeError("semantic wrapper plan bindings changed after publish")
    row_bytes = {
        key_bytes + value_bytes for key_bytes, value_bytes in pending.row_bytes_by_layer
    }
    if len(row_bytes) != 1:
        raise RuntimeError(
            "one FlashInfer wrapper cannot represent layer-varying KV row bytes"
        )
    unit_bytes = row_bytes.pop()
    if dependency_kind == "typed_lease":
        dependency_rows = tuple(
            0 if item is None else item.row_count for item in acquisition_slices
        )
        indexed_topology = lease_acquisition_topology(
            acquisition_slices,
            acquisition_groups,
            pending.operation_ranges(),
            index_count=int(pending.device_indices.numel()),
        )
        dependency_geometry: Any = (
            "typed_lease",
            acquisition_slices,
            acquisition_groups,
            int(pending.device_indices.numel()),
            pending.lease_id,
        )
    elif dependency_kind == "physical_pages":
        dependency_rows = tuple(len(pair[0]) for pair in page_pairs)
        indexed_topology = None
        dependency_geometry = page_pairs
    elif dependency_kind == "direct":
        dependency_rows = (0,) * schedule.work_count
        indexed_topology = None
        dependency_geometry = ("direct",)
    else:
        raise ValueError(f"unknown semantic dependency kind {dependency_kind!r}")
    topology = ExactWorkTopology.from_schedule(
        epoch=engine_batch.epoch,
        bindings=bindings,
        request_indices=schedule.request_indices,
        logical_work=schedule.kv_tile_indices,
        demand_units=tuple(max(1, rows) for rows in dependency_rows),
        unit_bytes=unit_bytes,
        estimated_compute_ns=tile_compute_ns,
    )
    return _SemanticWrapperPlan(
        schedule=schedule,
        topology=topology,
        dependency_kind=dependency_kind,
        work_dependency_rows=dependency_rows,
        signature_prefix=semantic_plan_signature_prefix(
            schedule.request_indices,
            schedule.kv_tile_indices,
            dependency_geometry,
            tuple((binding.request_slot, binding.generation) for binding in bindings),
        ),
        page_pairs=page_pairs,
        acquisition_slices=acquisition_slices,
        acquisition_groups=acquisition_groups,
        indexed_topology=indexed_topology,
    )


def plan_typed_lease_execution(
    schedules: Mapping[int, Schedule],
    acquisition_slices: Mapping[int, tuple[LeaseAcquisitionSlice | None, ...]],
    acquisition_groups: Mapping[int, tuple[LeaseAcquisitionGroup | None, ...]],
    pending: PendingHostLoad,
    *,
    object_capacity: int,
    model: HostCostModel,
    mode: HostExecutionMode,
    calibration_probe: bool,
    tenant_isolation: bool,
) -> HostExecutionPlan:
    """Choose one immutable host execution form from exact lease geometry."""

    if (
        not schedules
        or set(acquisition_slices) != set(schedules)
        or set(acquisition_groups) != set(schedules)
    ):
        raise RuntimeError("typed lease execution has no work dependencies")
    controller = pending.controller
    host_keys = tuple(controller.mem_pool_host.k_data_refs)
    host_values = tuple(controller.mem_pool_host.v_data_refs)
    if not host_keys or len(host_keys) != len(host_values):
        raise RuntimeError("HiCache host pool has incomplete K/V geometry")
    key_row_bytes = int(host_keys[0][0].numel()) * host_keys[0].element_size()
    value_row_bytes = int(host_values[0][0].numel()) * host_values[0].element_size()
    transfer_bytes = int(pending.device_indices.numel()) * (
        key_row_bytes + value_row_bytes
    )

    decisions: set[HostExecutionPlan] = set()
    for wrapper_id, schedule in schedules.items():
        dependencies = acquisition_slices[wrapper_id]
        transfer_groups = acquisition_groups[wrapper_id]
        if (
            len(dependencies) != schedule.work_count
            or len(transfer_groups) != schedule.work_count
        ):
            raise RuntimeError("typed lease dependencies do not match CTA work")
        initial_runnable = sum(item is None for item in dependencies)
        if not any(item is not None for item in dependencies):
            raise RuntimeError("typed lease schedule has no external work")
        object_count = 2 * len({item for item in transfer_groups if item is not None})
        if object_count <= 0 or object_count > object_capacity:
            raise RuntimeError("typed lease transfer groups exceed object capacity")
        decisions.add(
            plan_host_execution(
                object_count=object_count,
                transfer_bytes=transfer_bytes,
                runnable_tiles=schedule.work_count,
                initial_runnable_tiles=initial_runnable,
                model=model,
                calibration_probe=calibration_probe,
                scope_units=int(controller.layer_num),
                require_dependency_protocol=tenant_isolation,
                mode=mode,
            )
        )
    if len(decisions) != 1:
        raise RuntimeError(
            "FlashInfer wrappers selected inconsistent typed lease plans"
        )
    return decisions.pop()


def prove_direct_metadata_execution(
    schedules: Mapping[int, Schedule],
    pending: PendingHostLoad,
    bindings: tuple[RequestBinding, ...],
    *,
    host_staged: bool,
    tenant_isolation: bool,
    model: HostCostModel,
    mode: HostExecutionMode,
) -> HostExecutionPlan | None:
    """Reject dependency construction when even ideal overlap cannot win."""

    if not host_staged:
        raise RuntimeError("direct metadata proof is host-staged only")
    if tenant_isolation:
        return None
    if not schedules or not bindings:
        raise RuntimeError("host execution proof has no active schedule")
    controller = pending.controller
    host_keys = tuple(controller.mem_pool_host.k_data_refs)
    host_values = tuple(controller.mem_pool_host.v_data_refs)
    if not host_keys or len(host_keys) != len(host_values):
        raise RuntimeError("HiCache host pool has incomplete K/V geometry")
    transfer_count = int(pending.host_indices.numel())
    if transfer_count <= 0 or transfer_count != int(pending.device_indices.numel()):
        raise RuntimeError("HiCache host proof has no promoted pages")
    key_row_bytes = int(host_keys[0][0].numel()) * host_keys[0].element_size()
    value_row_bytes = int(host_values[0][0].numel()) * host_values[0].element_size()
    transfer_bytes = transfer_count * (key_row_bytes + value_row_bytes)
    proofs = tuple(
        prove_atomic_host_execution(
            object_count=2 * len(bindings),
            transfer_bytes=transfer_bytes,
            runnable_tiles=schedule.work_count,
            model=model,
            scope_units=int(controller.layer_num),
            mode=mode,
        )
        for schedule in schedules.values()
    )
    if any(proof is None for proof in proofs):
        return None
    return max(
        (proof for proof in proofs if proof is not None),
        key=lambda proof: proof.predicted_atomic_ns,
    )
