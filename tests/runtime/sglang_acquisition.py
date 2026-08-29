#!/usr/bin/env python3
"""Validate SGLang Host acquisition ownership and submission boundaries."""

from __future__ import annotations

import types

from nta_runtime.acquisition_scheduler import LayerAcquisition, LayerAcquisitionModel
from nta_runtime.engines.sglang_acquisition import (
    SglangHostAcquisitionCoordinator,
)
from nta_runtime.engines.sglang_state import _ActiveBatch
from nta_runtime.execution_planner import HostExecutionMode
from nta_runtime.execution_protocol import ProtocolKind


class FakeTransport:
    def __init__(self) -> None:
        self.ranges: list[tuple[int, int]] = []

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
            pending.prefetched_layers[local_layer] = object()

def coordinator(
    *,
    isolated: bool = False,
    mode: HostExecutionMode = HostExecutionMode.AUTO,
    layer_count: int = 4,
    frontier_enabled: bool = True,
):
    pool = object()
    transport = FakeTransport()
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
        transport=transport,
        stats={},
    )
    return owner, pool, transport


def pending(pool):
    return types.SimpleNamespace(
        controller=types.SimpleNamespace(mem_pool_device=pool, layer_num=4),
        layer_bytes=(),
        prefetched_layers={},
        transfer_plan=None,
        acquisition=None,
    )


def main() -> None:
    # Capture freezes physical descriptors before one dense finite submission.
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
    assert transport.ranges == [(0, 4)]
    assert lease.acquisition.started
    assert lease.acquisition.model is None
    assert lease.acquisition.fully_published
    assert owner._stats["initial_acquisition_layers"] == 4
    assert owner._stats["initial_typed_gap_layers"] == 0
    assert owner._stats["lease_acquisition_groups_started"] == 1

    # The explicit dependency-aware causal form cannot be preempted by an
    # eager whole-layer producer. Its first exact groups are bound only after
    # typed FlashInfer metadata exists.
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
    assert not typed_owner.prepare_admission(typed, object())

    # Tenant accounting must be bound before transport, so capture remains
    # claim-free and publishes no layer for an isolated lease.
    isolated_owner, isolated_pool, isolated_transport = coordinator(isolated=True)
    isolated = pending(isolated_pool)
    isolated_owner.account_selection = account
    isolated_owner.transfer_plan = freeze
    isolated_owner.capture(isolated)
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
    )
    admission.acquisition.bind_model(model)
    admission_owner.transfer_plan = lambda item, **_kwargs: item.transfer_plan
    admission_owner.deadline_model = lambda item, _batch: item.acquisition.model
    admission_owner.start_admission(admission, object())
    assert admission_transport.ranges == [(0, 4)]
    assert admission.acquisition.fully_published
    assert admission_owner._stats["host_acquisition_jobs_submitted"] == 4

    # Frontier policy belongs to the acquisition owner, not the framework
    # adapter. Before calibration it emits one bounded probe; a frozen model
    # can publish the complete feasible suffix in one transition.
    def exercise_frontier(
        *, calibrated: bool
    ) -> tuple[list[tuple[int, int]], dict[str, int]]:
        frontier_owner, _pool, frontier_transport = coordinator(layer_count=36)
        frontier_owner._movers = types.SimpleNamespace(
            collect_profiles=lambda: None,
            lease_calibrated=lambda _pending: False,
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
        batch = _ActiveBatch(
            bindings=(),
            semantic_plans={},
            pending_host_load=frontier_pending,
            deadline_model=model,
            deadline_model_initialized=calibrated,
        )
        frontier_owner.transfer_plan = (
            lambda item, **_kwargs: item.transfer_plan
        )
        frontier_owner.advance_after_attention(frontier_pending, batch, 0)
        return frontier_transport.ranges, frontier_owner._stats

    probe_ranges, probe_stats = exercise_frontier(calibrated=False)
    assert probe_ranges == [(1, 5)]
    assert probe_stats["deadline_frontier_calibration_layers"] == 4
    modeled_ranges, modeled_stats = exercise_frontier(calibrated=True)
    assert modeled_ranges == [(1, 36)]
    assert modeled_stats["deadline_frontier_published_layers"] == 35

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
