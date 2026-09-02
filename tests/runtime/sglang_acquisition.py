#!/usr/bin/env python3
"""Validate SGLang Host acquisition ownership and submission boundaries."""

from __future__ import annotations

import types

import torch

from nta_runtime.acquisition_scheduler import (
    LayerAcquisition,
    LayerAcquisitionModel,
    SharedAcquisitionState,
)
from nta_runtime.engines.sglang_acquisition import (
    SglangHostAcquisitionCoordinator,
)
from nta_runtime.engines.sglang_contracts import (
    LeaseAcquisitionGroup,
    LeaseOperationDemand,
    LeaseOperationRange,
)
from nta_runtime.engines.sglang_state import SglangForwardEpoch, SglangForwardPlan
from nta_runtime.execution_planner import HostExecutionMode
from nta_runtime.execution_protocol import ProtocolKind
from nta_runtime.requests import RequestBinding


class FakeTransport:
    def __init__(self) -> None:
        self.ranges: list[tuple[int, int]] = []
        self.events: dict[int, types.SimpleNamespace] = {}
        self.wave_events: dict[tuple[int, int], types.SimpleNamespace] = {}
        self.submitted_waves: dict[tuple[int, int], set[int]] = {}

    def prepare(
        self,
        pending,
        transfer_plan,
        *,
        first_local_layer: int,
        last_local_layer: int,
    ) -> None:
        assert transfer_plan is pending.transfer_plan
        self.ranges.append((first_local_layer, last_local_layer))
        for local_layer in range(first_local_layer, last_local_layer):
            event = types.SimpleNamespace(ready=False)
            event.query = lambda item=event: item.ready
            event.synchronize = lambda item=event: setattr(item, "ready", True)
            self.events[local_layer] = event
            pending.prefetched_layers[local_layer] = types.SimpleNamespace(
                key_bytes=1,
                value_bytes=1,
                ready_event=event,
                transfer_first_slot=None,
                wave_events=(),
                wave_row_ends=(),
            )

    def prepare_sm_group_wave(
        self, pending, transfer_plan, *, local_layer: int, wave_index: int
    ):
        assert transfer_plan is pending.transfer_plan
        event = types.SimpleNamespace(ready=False)
        event.query = lambda item=event: item.ready
        event.synchronize = lambda item=event: setattr(item, "ready", True)
        self.wave_events[(local_layer, wave_index)] = event
        key = (pending.lease_id, local_layer)
        submitted = self.submitted_waves.setdefault(key, set())
        submitted.add(wave_index)
        wave_ends = transfer_plan.layers[local_layer].wave_row_ends
        published = len(submitted) == len(wave_ends)
        if published:
            events = tuple(
                self.wave_events[(local_layer, wave)]
                for wave in range(len(wave_ends))
            )
            pending.prefetched_layers[local_layer] = types.SimpleNamespace(
                key_bytes=1,
                value_bytes=1,
                ready_event=events[-1],
                transfer_first_slot=0,
                wave_events=events,
                wave_row_ends=wave_ends,
            )
        return event, published


def fake_plan(*, row_count: int, wave_ends: tuple[int, ...] = ()):
    return types.SimpleNamespace(
        mover=types.SimpleNamespace(row_count=row_count),
        layers=tuple(
            types.SimpleNamespace(wave_row_ends=wave_ends) for _ in range(4)
        ),
        sm_waves_per_layer=len(wave_ends),
    )


def coordinator(
    *,
    isolated: bool = False,
    mode: HostExecutionMode = HostExecutionMode.AUTO,
    layer_count: int = 4,
    frontier_enabled: bool = True,
):
    pool = object()
    transport = FakeTransport()

    def bind(request_ids, request_slots, *, tenant_ids, **_kwargs):
        return tuple(
            RequestBinding(
                index,
                int(slot),
                1,
                index + 1,
                tenant_id=int(tenant),
            )
            for index, (slot, tenant) in enumerate(
                zip(request_slots, tenant_ids, strict=True)
            )
        )

    owner = SglangHostAcquisitionCoordinator(
        device_pool=pool,
        execution_config=types.SimpleNamespace(
            protocol=types.SimpleNamespace(kind=ProtocolKind.LATE_BOUND),
            host_execution_mode=mode,
        ),
        tenant_isolation_enabled=isolated,
        model_layer_count=layer_count,
        sm_acquisition_waves=1,
        frontier_enabled=frontier_enabled,
        frontier_layers_per_wave=4,
        movers=types.SimpleNamespace(),
        calibration=types.SimpleNamespace(),
        consumer_calibration=types.SimpleNamespace(
            bind_lease=lambda *_args, **_kwargs: None
        ),
        minimum_consumer_gain=1.03,
        transport=transport,
        request_adapter=types.SimpleNamespace(bind=bind),
        staging_capacity_bytes=(1 << 64) - 1,
        tenant_specs=(),
        max_inflight_groups=4096,
        stats={},
    )
    return owner, pool, transport


def pending(pool):
    return types.SimpleNamespace(
        lease_id=1,
        controller=types.SimpleNamespace(mem_pool_device=pool, layer_num=4),
        layer_bytes=(),
        prefetched_layers={},
        transfer_plan=None,
        acquisition=None,
        operation_demands=(LeaseOperationDemand(1, "request-1", 0, 1, 0),),
        operation_requests=(),
        operation_bindings={},
        scheduled_acquisition_groups=(LeaseAcquisitionGroup(1, 0, 1),),
        acquisition_group_identities={},
        shared_acquisition_registered=False,
        operation_ranges=lambda: (LeaseOperationRange(1, 0, 1),),
    )


def main() -> None:
    # AUTO captures ownership but waits for an exact scheduler shape before it
    # freezes mover descriptors. This prevents the earlier shape-free edge from
    # permanently selecting the SM issuer.
    owner, pool, transport = coordinator()
    lease = pending(pool)

    def account(item) -> None:
        item.layer_bytes = (1024,) * 4

    plan = object()

    def freeze(item, **_kwargs):
        item.transfer_plan = plan
        return plan

    owner.account_selection = account
    owner.transfer_plan = freeze
    owner.capture(lease)
    assert transport.ranges == []
    assert lease.acquisition is None
    assert lease.transfer_plan is None
    assert owner._stats["initial_acquisition_layers"] == 0
    assert owner._stats["initial_typed_gap_layers"] == 4
    assert owner._stats["schedule_bound_acquisition_batches"] == 1

    # DIRECT remains the explicit batch-agnostic eager baseline.
    direct_owner, direct_pool, direct_transport = coordinator(
        mode=HostExecutionMode.DIRECT
    )
    direct = pending(direct_pool)
    direct_owner.account_selection = account
    direct_owner.transfer_plan = freeze
    direct_owner.capture(direct)
    assert direct_transport.ranges == [(0, 4)]
    assert direct.acquisition.started
    assert direct.acquisition.model is None
    assert direct.acquisition.fully_published
    assert direct_owner._stats["initial_acquisition_layers"] == 4
    assert direct_owner._stats["initial_typed_gap_layers"] == 0
    assert direct_owner._stats["lease_acquisition_groups_started"] == 1

    # A1P owns the same eager producer as A1 while exposing the typed partial
    # consumer. Acquisition timing and consumer release are independent axes.
    eager_typed_owner, eager_typed_pool, eager_typed_transport = coordinator(
        mode=HostExecutionMode.EAGER_PROGRESSIVE
    )
    eager_typed = pending(eager_typed_pool)
    eager_typed_owner.account_selection = account
    eager_typed_owner.transfer_plan = freeze
    eager_typed_owner.capture(eager_typed)
    assert eager_typed_transport.ranges == [(0, 4)]
    assert eager_typed.acquisition.started
    assert eager_typed.acquisition.fully_published
    assert eager_typed_owner.eager_capture_enabled
    assert eager_typed_owner._stats["initial_acquisition_layers"] == 4
    assert eager_typed_owner._stats.get("schedule_bound_acquisition_batches", 0) == 0

    # A2 and A3 share the scheduler-bound producer queue. Neither freezes the
    # transfer plan at lease capture; their only causal difference is whether
    # the numerical consumer may run from partial event-wave readiness.
    scheduled_owner, scheduled_pool, scheduled_transport = coordinator(
        mode=HostExecutionMode.SCHEDULED_BULK
    )
    scheduled = pending(scheduled_pool)
    scheduled_owner.account_selection = account
    scheduled_owner.transfer_plan = freeze
    scheduled_owner.capture(scheduled)
    assert scheduled.acquisition is None
    assert not scheduled.prefetched_layers
    assert not scheduled_transport.ranges
    assert scheduled_owner._stats["initial_acquisition_layers"] == 0
    assert scheduled_owner._stats["schedule_bound_acquisition_batches"] == 1
    assert scheduled_owner.proactive_layer_queue_enabled
    assert not scheduled_owner.eager_capture_enabled

    typed_owner, typed_pool, typed_transport = coordinator(
        mode=HostExecutionMode.DEPENDENCY_AWARE
    )
    typed = pending(typed_pool)
    typed_owner.account_selection = account
    typed_owner.transfer_plan = freeze
    typed_owner.capture(typed)
    assert typed.acquisition is None
    assert not typed.prefetched_layers
    assert not typed_transport.ranges
    assert typed_owner._stats["initial_acquisition_layers"] == 0
    assert typed_owner._stats["schedule_bound_acquisition_batches"] == 1
    assert typed_owner.proactive_layer_queue_enabled

    # Tenant accounting must be bound before transport, so capture remains
    # claim-free and publishes no layer for an isolated lease.
    isolated_owner, isolated_pool, isolated_transport = coordinator(isolated=True)
    isolated = pending(isolated_pool)
    isolated_owner.account_selection = account
    isolated_owner.transfer_plan = freeze
    isolated_owner.capture(isolated)
    assert isolated_owner.proactive_layer_queue_enabled
    assert isolated.acquisition is None
    assert not isolated.prefetched_layers
    assert not isolated_transport.ranges
    assert isolated_owner._stats["schedule_bound_acquisition_batches"] == 1

    # Admission starts exactly one complete finite queue. Layer zero is the
    # earliest deadline, not a bootstrap-only transport special case.
    admission_owner, _pool, admission_transport = coordinator()
    model = LayerAcquisitionModel(
        layer_bytes=(1,) * 4,
        transfer_service_ns=(50,) * 4,
        initial_compute_ns=0,
        inter_layer_compute_ns=100,
    )
    admission = types.SimpleNamespace(
        controller=types.SimpleNamespace(layer_num=4),
        transfer_plan=object(),
        prefetched_layers={},
        acquisition=LayerAcquisition(model.layer_bytes),
        shared_acquisition_registered=False,
    )
    admission.acquisition.bind_model(model)
    admission_owner.transfer_plan = lambda item, **_kwargs: item.transfer_plan
    admission_owner.deadline_model = lambda item, _batch: item.acquisition.model
    admission_owner.start_admission(admission, object())
    assert admission_transport.ranges == [(0, 4)]
    assert admission.acquisition.fully_published
    assert admission_owner._stats["host_acquisition_jobs_submitted"] == 4

    # Feasibility is optional; physical ownership is not. An unseen dynamic
    # batch shape receives a structural layer queue, submits in transformer
    # order, and retires every numerical consumer without claiming an EDF
    # model or learning optimistically in the serving window.
    structural_owner, structural_pool, structural_transport = coordinator()
    structural = pending(structural_pool)
    structural.arrival_profile_key = None
    structural_owner.account_selection = account
    structural_owner._calibration = types.SimpleNamespace(
        shape_key=lambda _batch: ("extend", 64, 2),
        curve_for_batch=lambda _batch: None,
    )
    structural_owner._movers = types.SimpleNamespace(collect_profiles=lambda: None)
    structural_plan = types.SimpleNamespace(
        mover=types.SimpleNamespace(kind="sm"),
        sm_waves_per_layer=1,
    )
    structural_owner.transfer_plan = (
        lambda item, **_kwargs: setattr(item, "transfer_plan", structural_plan)
        or structural_plan
    )

    def bind_structural(item, **_kwargs):
        item.arrival_profile_key = object()

    structural_owner._consumer_calibration = types.SimpleNamespace(
        bind_lease=bind_structural
    )
    structural_owner.capture(structural)
    structural_batch = types.SimpleNamespace(
        reqs=(types.SimpleNamespace(rid="request-1", req_pool_idx=3),)
    )
    assert not structural_owner.prepare_owner(structural, structural_batch)
    assert structural.operation_bindings[1].request_slot == 3
    assert structural.operation_bindings[1].generation == 1
    assert len(structural.acquisition_group_identities) == 4
    assert structural.acquisition_group_identities[0][0].request_slot == 3
    assert structural.acquisition_group_identities[0][0].resource_version == 1
    assert structural.acquisition is not None
    assert structural.acquisition.model is None
    assert structural_owner._stats["host_acquisition_jobs_prepared"] == 4
    assert structural_owner._stats["host_acquisition_structural_owners"] == 1
    assert structural_owner.submit(structural) == 4
    assert structural_transport.ranges == [(0, 4)]
    for layer in range(4):
        structural_owner.retire_layer(structural, layer)
    assert structural.acquisition.queue.terminal
    assert structural_owner._stats["host_acquisition_layers_consumed"] == 4

    # The production shared-link path keeps complete request-generation group
    # identity while issuing only a finite layer cohort. Completing its fence
    # opens the next EDF slot; registration never queues the whole model.
    shared_owner, shared_pool, shared_transport = coordinator(
        layer_count=4, frontier_enabled=True
    )
    shared_owner._shared_dispatch_horizon = 1
    shared_owner._shared_layers_per_dispatch = 1
    shared = pending(shared_pool)
    shared.device_indices = torch.tensor((7,), dtype=torch.int32)
    shared.row_bytes_by_layer = ((1, 1),) * 4
    shared.layer_bytes = (2,) * 4
    shared.transfer_plan = fake_plan(row_count=1)
    shared_owner.transfer_plan = lambda item, **_kwargs: item.transfer_plan
    shared_owner._bind_group_identities(shared, structural_batch)
    shared_model = LayerAcquisitionModel(
        layer_bytes=shared.layer_bytes,
        transfer_service_ns=(50,) * 4,
        initial_compute_ns=0,
        inter_layer_compute_ns=100,
    )
    shared.acquisition = LayerAcquisition(shared.layer_bytes)
    shared.acquisition.bind_model(shared_model)
    shared_owner._register_shared_acquisition(shared, shared_model)
    shared_owner._pump_shared_acquisition()
    assert shared_transport.ranges == [(0, 1)]
    assert len(shared.prefetched_layers) == 1
    shared_transport.events[0].ready = True
    shared_owner.progress_shared_acquisition()
    shared_owner._pump_shared_acquisition()
    assert shared_transport.ranges == [(0, 1), (1, 2)]
    shared_owner.retire_layer(shared, 0)
    assert shared_owner._stats["shared_acquisition_retired_cohorts"] == 1

    # A later mixed batch can reach metadata while an older lease owns the
    # finite dispatch horizon.  Its whole-layer stock consumer waits for EDF
    # to publish layer zero; it must not bypass the queue with eager full-model
    # publication.
    blocked = pending(shared_pool)
    blocked.lease_id = 2
    blocked.device_indices = torch.tensor((9,), dtype=torch.int32)
    blocked.row_bytes_by_layer = ((1, 1),) * 4
    blocked.layer_bytes = (2,) * 4
    blocked.transfer_plan = fake_plan(row_count=1)
    shared_owner._bind_group_identities(blocked, structural_batch)
    shared_owner._register_shared_acquisition(blocked, shared_model)
    assert 0 not in blocked.prefetched_layers
    shared_owner.ensure_layer_published(blocked, 0)
    assert 0 in blocked.prefetched_layers
    assert shared_owner._stats["shared_acquisition_publication_waits"] == 1
    assert shared_owner._stats["shared_acquisition_publication_wait_rounds"] > 0
    blocked.prefetched_layers[0].transfer_first_slot = 0
    progressive_batch = types.SimpleNamespace(
        pending_host_load=blocked,
        host_execution=types.SimpleNamespace(
            uses_progressive_consumer=True,
            overlap_initial=True,
            selection_reason="forced_dependency_aware",
        ),
        modeled_ready_by_attention_layers=set(),
        planned_progressive_consumer_layers=set(),
    )
    assert shared_owner.plan_published_consumer_layer(
        blocked, progressive_batch, 0
    )
    assert progressive_batch.planned_progressive_consumer_layers == {0}
    # Re-observing the same late publication is idempotent and cannot inflate
    # activation evidence.
    assert shared_owner.plan_published_consumer_layer(
        blocked, progressive_batch, 0
    )
    assert shared_owner._stats["partial_consumer_planned_layers"] == 1
    blocked_identity = blocked.acquisition_group_identities[0][0]
    assert (
        shared_owner._shared_queue.state(blocked_identity)
        is SharedAcquisitionState.FENCE_PUBLISHED
    )
    shared_owner.retire_layer(blocked, 0)
    blocked.prefetched_layers[0].ready_event.synchronize()
    shared_owner.progress_shared_acquisition()
    assert not any(key[:2] == (blocked.lease_id, 0) for key in shared_owner._shared_cohorts)

    # Readiness and resource lifetime are group-scoped even when one finite
    # physical layer packet coalesces several request segments.  The first
    # wave releases only its own tenant/staging reservation and can be
    # consumed without waiting for the second request's wave.
    wave_owner, wave_pool, wave_transport = coordinator(
        layer_count=4, frontier_enabled=True
    )
    wave_owner._sm_acquisition_waves = 2
    wave_owner._shared_dispatch_horizon = 1
    wave_owner._shared_layers_per_dispatch = 1
    wave_pending = pending(wave_pool)
    wave_pending.device_indices = torch.tensor((7, 8), dtype=torch.int32)
    wave_pending.operation_demands = (
        LeaseOperationDemand(1, "wave-request-1", 0, 1, 0),
        LeaseOperationDemand(2, "wave-request-2", 0, 1, 0),
    )
    wave_pending.scheduled_acquisition_groups = (
        LeaseAcquisitionGroup(1, 0, 1),
        LeaseAcquisitionGroup(2, 0, 1),
    )
    wave_pending.operation_ranges = lambda: (
        LeaseOperationRange(1, 0, 1),
        LeaseOperationRange(2, 1, 1),
    )
    wave_pending.row_bytes_by_layer = ((1, 1),) * 4
    wave_pending.layer_bytes = (4,) * 4
    wave_pending.transfer_plan = fake_plan(row_count=2, wave_ends=(1, 2))
    wave_owner.transfer_plan = lambda item, **_kwargs: item.transfer_plan
    wave_owner._bind_group_identities(
        wave_pending,
        types.SimpleNamespace(
            reqs=(
                types.SimpleNamespace(rid="wave-request-1", req_pool_idx=3),
                types.SimpleNamespace(rid="wave-request-2", req_pool_idx=4),
            )
        ),
    )
    wave_model = LayerAcquisitionModel(
        layer_bytes=wave_pending.layer_bytes,
        transfer_service_ns=(100,) * 4,
        initial_compute_ns=0,
        inter_layer_compute_ns=200,
    )
    wave_owner._register_shared_acquisition(wave_pending, wave_model)
    wave_owner._pump_shared_acquisition()
    first_wave = wave_transport.wave_events[(0, 0)]
    identities = wave_pending.acquisition_group_identities[0]
    first_wave.ready = True
    wave_owner.progress_shared_acquisition()
    assert wave_owner._shared_queue.state(identities[0]) is SharedAcquisitionState.READY
    assert (
        wave_owner._shared_queue.state(identities[1])
        is SharedAcquisitionState.PLANNED
    )
    assert wave_owner._shared_queue.staging_outstanding_bytes == 0
    wave_owner._pump_shared_acquisition()
    second_wave = wave_transport.wave_events[(0, 1)]
    assert 0 in wave_pending.prefetched_layers
    assert (
        wave_owner._shared_queue.state(identities[1])
        is SharedAcquisitionState.FENCE_PUBLISHED
    )
    assert wave_owner._shared_queue.staging_outstanding_bytes == 2
    wave_owner.retire_layer(wave_pending, 0)
    second_wave.ready = True
    wave_owner.progress_shared_acquisition()
    assert (wave_pending.lease_id, 0) not in wave_owner._shared_cohorts
    assert wave_owner._shared_queue.staging_outstanding_bytes == 0

    # A progressive producer capability is not itself a partial-consumer
    # decision. Normal AUTO selection must wait for a per-layer arrival/cost
    # proof; bounded calibration may explicitly measure the path, while an EDF
    # ready prediction revokes that layer from the probe.
    publications = {
        layer: types.SimpleNamespace(transfer_first_slot=2 * layer)
        for layer in range(4)
    }

    def consumer_batch(reason: str) -> SglangForwardEpoch:
        lease = types.SimpleNamespace(
            prefetched_layers=publications,
            planned_progressive_layers=frozenset(),
        )
        execution = types.SimpleNamespace(
            uses_progressive_consumer=True,
            overlap_initial=True,
            selection_reason=reason,
        )
        return SglangForwardEpoch(
            plan=SglangForwardPlan(
                bindings=(),
                semantic_plans={},
                pending_host_load=lease,
                host_execution=execution,
            )
        )

    unproven_batch = consumer_batch("predicted_gain")
    admission_owner.plan_published_consumers(
        unproven_batch.pending_host_load,
        unproven_batch,
    )
    assert not unproven_batch.planned_progressive_consumer_layers
    assert admission_owner._stats["partial_consumer_unproven_layers"] == 4

    probe_batch = consumer_batch("calibration_probe")
    probe_batch.modeled_ready_by_attention_layers.add(3)
    admission_owner.plan_published_consumers(
        probe_batch.pending_host_load,
        probe_batch,
    )
    assert probe_batch.planned_progressive_consumer_layers == {0, 1, 2}
    assert admission_owner._stats["partial_consumer_planned_layers"] == 3

    # Frontier policy belongs to the acquisition owner, not the framework
    # adapter. Before calibration it emits one bounded probe; a frozen model
    # can publish the complete feasible suffix in one transition.
    def exercise_frontier(
        *, calibrated: bool, frozen: bool = False
    ) -> tuple[list[tuple[int, int]], dict[str, int], set[int]]:
        frontier_owner, _pool, frontier_transport = coordinator(layer_count=36)
        frontier_owner._movers = types.SimpleNamespace(
            collect_profiles=lambda: None,
            lease_calibrated=lambda _pending: False,
            calibration_frozen=frozen,
        )
        frontier_owner._calibration = types.SimpleNamespace(
            collect=lambda: None,
            curve=lambda _key: None,
        )
        frontier_pending = types.SimpleNamespace(
            controller=types.SimpleNamespace(layer_num=36),
            mover_plan=object(),
            prefetched_layers={},
            transfer_plan=object(),
            acquisition=None,
        )
        model = (
            LayerAcquisitionModel(
                layer_bytes=(1,) * 36,
                transfer_service_ns=(50,) * 36,
                initial_compute_ns=0,
                inter_layer_compute_ns=100,
            )
            if calibrated
            else None
        )
        batch = SglangForwardEpoch(
            plan=SglangForwardPlan(
                bindings=(),
                semantic_plans={},
                pending_host_load=frontier_pending,
            ),
            deadline_model=model,
            deadline_model_initialized=calibrated,
        )
        frontier_owner.transfer_plan = lambda item, **_kwargs: item.transfer_plan
        frontier_owner.advance_after_attention(frontier_pending, batch, 0)
        return (
            frontier_transport.ranges,
            frontier_owner._stats,
            batch.modeled_ready_by_attention_layers,
        )

    probe_ranges, probe_stats, probe_modeled = exercise_frontier(calibrated=False)
    assert probe_ranges == [(1, 5)]
    assert probe_stats["deadline_frontier_calibration_layers"] == 4
    assert not probe_modeled
    frozen_ranges, frozen_stats, frozen_modeled = exercise_frontier(
        calibrated=False, frozen=True
    )
    assert frozen_ranges == []
    assert frozen_stats.get("deadline_frontier_calibration_layers", 0) == 0
    assert not frozen_modeled
    modeled_ranges, modeled_stats, modeled_layers = exercise_frontier(calibrated=True)
    assert modeled_ranges == [(1, 36)]
    assert modeled_stats["deadline_frontier_published_layers"] == 35
    assert modeled_stats["deadline_frontier_modeled_ready_layers"] == 35
    assert modeled_layers == set(range(1, 36))

    # A fully published direct/eager lease must not enter calibration or EDF
    # analysis on every layer.  This is the resident launch thread's stock
    # external fast path, so even collecting a profile here is a regression.
    full_owner, _pool, full_transport = coordinator(layer_count=4)
    full_owner._movers = types.SimpleNamespace(
        collect_profiles=lambda: (_ for _ in ()).throw(
            AssertionError("fully published lease entered frontier analysis")
        )
    )
    fully_published = types.SimpleNamespace(
        controller=types.SimpleNamespace(layer_num=4),
        prefetched_layers={layer: object() for layer in range(4)},
        acquisition=None,
    )
    full_batch = SglangForwardEpoch(
        plan=SglangForwardPlan(
            bindings=(),
            semantic_plans={},
            pending_host_load=fully_published,
        )
    )
    for layer in range(4):
        full_owner.advance_after_attention(fully_published, full_batch, layer)
    assert full_transport.ranges == []
    assert full_owner._stats == {}

    # Every public publication edge passes the exact frozen plan into the data
    # path; the transport never calls back into planning or allocation.
    range_owner, _pool, range_transport = coordinator()
    ranged = types.SimpleNamespace(
        controller=types.SimpleNamespace(layer_num=4),
        transfer_plan=object(),
        prefetched_layers={0: object()},
    )
    range_owner.transfer_plan = lambda item, **_kwargs: item.transfer_plan
    range_owner.publish_range(ranged, 1, 3)
    assert range_owner.publish_missing(ranged, exclude=frozenset({3})) == 0
    assert range_transport.ranges == [(1, 3)]
    disjoint = types.SimpleNamespace(
        controller=types.SimpleNamespace(layer_num=4),
        transfer_plan=object(),
        prefetched_layers={1: object()},
    )
    assert range_owner.publish_missing(disjoint, exclude=frozenset({3})) == 2
    assert range_transport.ranges[-2:] == [(0, 1), (2, 3)]


if __name__ == "__main__":
    main()
