#!/usr/bin/env python3
"""Validate exact paired-run decomposition for transfer profiling/planning."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from itertools import combinations
from types import SimpleNamespace

import torch

from nta_runtime.indexed_transfer import (
    ContiguousPairRun,
    IndexedMoverServiceModel,
    StridedCopyGroup,
    analyze_index_pairs,
    plan_indexed_dependencies,
    plan_indexed_mover,
    select_indexed_mover_candidates,
)
from nta_runtime.indexed_transfer_torch import (
    plan_indexed_tensor_mover,
    warm_indexed_tensor_mover,
)
from nta_runtime.engines.sglang_transfer import (
    HostMoverController,
    HostMoverLeasePlan,
    build_host_transfer_lease_plan,
    host_mover_service_model_from_environment,
)


def _apply_plan(source, destination, plan):
    output = [None] * (max(destination) + 1)
    for run in plan.copy_runs:
        for row in range(run.row_count):
            output[run.destination_first + row] = source[run.source_first + row]
    for source_row, destination_row in zip(
        plan.sm_source_indices, plan.sm_destination_indices, strict=True
    ):
        output[destination_row] = source[source_row]
    return output


def _calibrated_model(
    *,
    sm_bandwidth: int = 1_000_000_000,
    copy_bandwidth: int = 4_000_000_000,
    copy_operation_ns: int = 800,
    minimum_gain: float = 1.03,
    hybrid_join_ns: int = 0,
) -> IndexedMoverServiceModel:
    return IndexedMoverServiceModel(
        sm_bandwidth_bytes_per_second=sm_bandwidth,
        copy_bandwidth_bytes_per_second=copy_bandwidth,
        copy_operation_ns=copy_operation_ns,
        hybrid_join_ns=hybrid_join_ns,
        minimum_gain=minimum_gain,
        sm_samples=3,
        copy_samples=3,
    )


def _assert_prefix_matches_exhaustive_optimum(
    run_rows: tuple[int, ...],
    *,
    maximum_copy_runs: int,
    service_model: IndexedMoverServiceModel,
    overlap_compute_ns: int,
) -> None:
    """Compare bounded-prefix selection with every representable subset."""

    if service_model.minimum_gain != 1.0:
        raise ValueError("the exhaustive cost oracle requires a unit policy margin")
    total_rows = sum(run_rows)
    row_bytes = 4_096
    operations_per_run = 3
    predicted_sm_ns = service_model.candidate_ns(
        total_rows=total_rows,
        copy_rows=0,
        copy_run_count=0,
        row_bytes=row_bytes,
        copy_operations_per_run=operations_per_run,
        overlap_compute_ns=overlap_compute_ns,
    )
    assert predicted_sm_ns is not None
    exhaustive_ns = predicted_sm_ns
    for subset_size in range(1, min(len(run_rows), maximum_copy_runs) + 1):
        for subset in combinations(range(len(run_rows)), subset_size):
            candidate_ns = service_model.candidate_ns(
                total_rows=total_rows,
                copy_rows=sum(run_rows[index] for index in subset),
                copy_run_count=subset_size,
                row_bytes=row_bytes,
                copy_operations_per_run=operations_per_run,
                overlap_compute_ns=overlap_compute_ns,
            )
            assert candidate_ns is not None
            exhaustive_ns = min(exhaustive_ns, candidate_ns)
    selected = select_indexed_mover_candidates(
        total_rows=total_rows,
        total_run_count=len(run_rows),
        candidate_runs=tuple(enumerate(run_rows)),
        row_bytes=row_bytes,
        copy_operations_per_run=operations_per_run,
        maximum_copy_runs=maximum_copy_runs,
        service_model=service_model,
        overlap_compute_ns=overlap_compute_ns,
    )
    assert selected.predicted_selected_ns == exhaustive_ns


def main() -> None:
    host_keys = tuple(torch.empty((8, 2, 4), dtype=torch.float16) for _ in range(2))
    host_values = tuple(torch.empty((8, 2, 4), dtype=torch.float16) for _ in range(2))
    device_keys = tuple(torch.empty((8, 2, 4), dtype=torch.float16) for _ in range(2))
    device_values = tuple(torch.empty((8, 2, 4), dtype=torch.float16) for _ in range(2))

    class DevicePool:
        start_layer = 0

        @staticmethod
        def _get_key_buffer(layer: int) -> torch.Tensor:
            return device_keys[layer]

        @staticmethod
        def _get_value_buffer(layer: int) -> torch.Tensor:
            return device_values[layer]

    controller = SimpleNamespace(
        layer_num=2,
        mem_pool_host=SimpleNamespace(
            k_data_refs=host_keys,
            v_data_refs=host_values,
        ),
        mem_pool_device=DevicePool(),
    )
    source_map = torch.tensor((0, 1, 2), dtype=torch.int32)
    destination_map = torch.tensor((3, 4, 5), dtype=torch.int32)
    sm_lease = HostMoverLeasePlan(
        3,
        "sm",
        (),
        source_map,
        destination_map,
        None,
        0,
        100,
        100,
        "forced_sm",
    )
    row_geometry = ((16, 16), (16, 16))
    sm_transfer = build_host_transfer_lease_plan(
        controller,
        sm_lease,
        row_geometry,
        object_id_bases=(100, 200),
        object_version=1,
        sm_acquisition_waves=2,
    )
    assert sm_transfer.layer_geometry == ((48, 48), (48, 48))
    assert sm_transfer.sm_waves_per_layer == 2
    assert len(sm_transfer.indexed_objects) == 8
    assert not any(sm_transfer.copy_groups)
    assert tuple(item.index_count for item in sm_transfer.layers[0].indexed_objects) == (
        2,
        2,
        1,
        1,
    )
    copy_lease = HostMoverLeasePlan(
        3,
        "copy_engine",
        (ContiguousPairRun(0, 3, 3),),
        torch.empty(0, dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        None,
        0,
        100,
        80,
        "forced_copy_engine",
    )
    copy_transfer = build_host_transfer_lease_plan(
        controller,
        copy_lease,
        row_geometry,
        object_id_bases=(100, 200),
        object_version=1,
        sm_acquisition_waves=4,
    )
    assert not copy_transfer.indexed_objects
    assert tuple(len(groups) for groups in copy_transfer.copy_groups) == (2, 2)

    mover_stats: dict[str, object] = {}
    mover_controller = HostMoverController(
        policy="sm",
        default_service_model=_calibrated_model(),
        calibration_samples=3,
        copy_engine_max_operations=64,
        frontier_layers_per_wave=2,
        profile_transfer=False,
        frontier_enabled=False,
        profile_index_layout=False,
        profile_index_min_bytes=64 * 1024,
        verify_index_map=False,
        stats=mover_stats,
    )
    materializations = 0

    class PendingMoverLease:
        mover_plan = None
        prefetch_tensors = ()
        device_indices = destination_map
        row_bytes_by_layer = row_geometry

        @staticmethod
        def materialize_device_index_map():
            nonlocal materializations
            materializations += 1
            return SimpleNamespace(
                source_indices=source_map,
                destination_indices=destination_map,
            )

    pending_mover = PendingMoverLease()
    mover_plan = mover_controller.plan(
        pending_mover,
        row_geometry,
        3,
        layer_service_key=None,
        layer_curve=None,
        collect_layer_profiles=lambda: None,
    )
    assert mover_plan.kind == "sm"
    assert mover_plan.selection_reason == "forced_sm"
    assert mover_controller.admission_calibrated(pending_mover)
    assert mover_controller.lease_calibrated(pending_mover)
    assert (
        mover_controller.plan(
            pending_mover,
            row_geometry,
            3,
            layer_service_key=None,
            layer_curve=None,
            collect_layer_profiles=lambda: None,
        )
        is mover_plan
    )
    assert materializations == 1
    assert mover_stats["prefetch_mover_plan_sm_leases"] == 1

    uncalibrated_environment = host_mover_service_model_from_environment({})
    assert not uncalibrated_environment.copy_calibrated
    assert uncalibrated_environment.sm_samples == 0
    assert uncalibrated_environment.copy_samples == 0
    calibrated_environment = host_mover_service_model_from_environment(
        {
            "NTA_EXECUTION_HOST_SM_BANDWIDTH_BPS": "10",
            "NTA_EXECUTION_HOST_COPY_BANDWIDTH_BPS": "20",
            "NTA_EXECUTION_HOST_COPY_OPERATION_NS": "30",
        }
    )
    assert calibrated_environment.copy_calibrated
    assert calibrated_environment.sm_calibrated
    assert calibrated_environment.sm_samples == 3
    assert calibrated_environment.copy_samples == 3
    assert calibrated_environment.ideal_copy_can_qualify(
        total_rows=1024,
        row_bytes=4096,
        copy_operations_per_run=2,
    )
    assert not _calibrated_model(
        sm_bandwidth=40_000_000_000,
        copy_bandwidth=10_000_000_000,
        copy_operation_ns=1_000,
    ).ideal_copy_can_qualify(
        total_rows=1024,
        row_bytes=4096,
        copy_operations_per_run=2,
    )
    scale_scoped_environment = host_mover_service_model_from_environment(
        {
            "NTA_EXECUTION_HOST_SM_BANDWIDTH_BPS": "10",
            "NTA_EXECUTION_HOST_COPY_BANDWIDTH_BPS": "20",
            "NTA_EXECUTION_HOST_COPY_OPERATION_NS": "30",
            "NTA_EXECUTION_HOST_MOVER_CALIBRATION_BYTES": str(1 << 20),
            "NTA_EXECUTION_HOST_COPY_COMPUTE_OVERLAP_EFFICIENCY": "0.5",
        }
    )
    assert scale_scoped_environment.supports_transfer_scale(1 << 20)
    assert not scale_scoped_environment.supports_transfer_scale(1 << 21)
    assert scale_scoped_environment.effective_copy_compute_overlap == 0.5
    try:
        host_mover_service_model_from_environment(
            {"NTA_EXECUTION_HOST_COPY_BANDWIDTH_BPS": "20"}
        )
    except ValueError as error:
        assert "requires both" in str(error)
    else:
        raise AssertionError("partial mover calibration was accepted")

    service_model = _calibrated_model()
    observed_sm = IndexedMoverServiceModel(sm_bandwidth_bytes_per_second=10)
    for sample in range(3):
        observed_sm = observed_sm.with_sm_observation(
            transfer_bytes=1 << 20,
            elapsed_ns=1_000_000,
            alpha=1.0,
        )
        assert observed_sm.sm_calibrated == (sample == 2)
    assert observed_sm.sm_samples == 3
    assert observed_sm.sm_bandwidth_bytes_per_second == 1_048_576_000
    observed_copy = observed_sm
    for sample in range(3):
        observed_copy = observed_copy.with_copy_observation(
            transfer_bytes=1 << 20,
            elapsed_ns=400_000,
            operation_count=8,
            issue_cpu_ns=80_000,
            alpha=1.0,
        )
        assert observed_copy.copy_calibrated == (sample == 2)
    assert observed_copy.copy_samples == 3
    assert observed_copy.copy_calibrated
    assert observed_copy.copy_operation_ns == 10_000
    assert observed_copy.copy_bandwidth_bytes_per_second == 2_621_440_000
    assert observed_copy.copy_bandwidth_bytes_per_second > (
        observed_copy.sm_bandwidth_bytes_per_second
    )
    assert observed_copy.calibration_scale_bucket == 20
    hybrid_scale = 1 << 23
    hybrid_curve = IndexedMoverServiceModel(
        sm_bandwidth_bytes_per_second=10,
        minimum_calibration_samples=1,
    ).with_sm_observation(
        transfer_bytes=1 << 20,
        service_scale_bytes=hybrid_scale,
        elapsed_ns=1_000_000,
        alpha=1.0,
    )
    hybrid_curve = hybrid_curve.with_copy_observation(
        transfer_bytes=7 << 20,
        service_scale_bytes=hybrid_scale,
        elapsed_ns=1_000_000,
        operation_count=8,
        issue_cpu_ns=80_000,
        alpha=1.0,
    )
    assert hybrid_curve.sm_calibrated and hybrid_curve.copy_calibrated
    assert hybrid_curve.supports_transfer_scale(hybrid_scale)
    assert not hybrid_curve.supports_transfer_scale(1 << 20)
    try:
        hybrid_curve.with_sm_observation(
            transfer_bytes=1 << 20,
            service_scale_bytes=1 << 19,
            elapsed_ns=1_000_000,
        )
    except ValueError as error:
        assert "service scale" in str(error)
    else:
        raise AssertionError("a mover component exceeded its service-scale wave")
    try:
        observed_copy.with_sm_observation(
            transfer_bytes=1 << 21,
            elapsed_ns=1_000_000,
        )
    except ValueError as error:
        assert "size buckets" in str(error)
    else:
        raise AssertionError("one mover curve mixed unrelated transfer scales")
    for invalid in (
        {"transfer_bytes": 0, "elapsed_ns": 1},
        {"transfer_bytes": 1, "elapsed_ns": 0},
    ):
        try:
            service_model.with_sm_observation(**invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid SM mover observation was accepted")
    layout = analyze_index_pairs((4, 5, 6, 20, 21, 8), (40, 41, 42, 3, 4, 9))
    assert layout.row_count == 6
    assert tuple(
        (run.source_first, run.destination_first, run.row_count) for run in layout.runs
    ) == ((4, 40, 3), (20, 3, 2), (8, 9, 1))
    assert layout.maximum_run_rows == 3
    assert layout.eligible_rows(row_bytes=1024, minimum_copy_bytes=2048) == 5
    assert layout.eligible_runs(row_bytes=1024, minimum_copy_bytes=2048) == 2

    source_values = tuple(range(32))
    source_indices = (4, 5, 6, 20, 21, 8)
    destination_indices = (10, 11, 12, 3, 4, 9)
    auto = plan_indexed_mover(
        source_indices,
        destination_indices,
        row_bytes=1024,
        copy_operations_per_run=1,
        maximum_copy_runs=8,
        service_model=service_model,
    )
    assert auto.kind == "copy_engine"
    assert auto.copy_row_count == 6 and auto.sm_row_count == 0
    assert auto.predicted_sm_ns == 6_144
    assert auto.predicted_selected_ns == 2_400
    compute_aware = plan_indexed_mover(
        source_indices,
        destination_indices,
        row_bytes=1024,
        copy_operations_per_run=1,
        maximum_copy_runs=8,
        service_model=service_model,
        overlap_compute_ns=5_000,
    )
    # No overlap observation means conservative serialization with compute.
    assert compute_aware.kind == "copy_engine"
    assert compute_aware.copy_row_count == 6 and compute_aware.sm_row_count == 0
    assert compute_aware.predicted_selected_ns == 7_400
    assert compute_aware.predicted_sm_ns == 11_144

    measured_overlap = service_model
    for sample in range(3):
        measured_overlap = measured_overlap.with_copy_compute_overlap_observation(
            transfer_bytes=6 * 1024,
            isolated_copy_ns=2_400,
            isolated_compute_ns=5_000,
            concurrent_ns=5_000,
            alpha=1.0,
        )
        assert measured_overlap.overlap_calibrated == (sample == 2)
    overlap_aware = plan_indexed_mover(
        source_indices,
        destination_indices,
        row_bytes=1024,
        copy_operations_per_run=1,
        maximum_copy_runs=8,
        service_model=measured_overlap,
        overlap_compute_ns=5_000,
    )
    assert overlap_aware.kind == "copy_engine"
    assert overlap_aware.predicted_selected_ns == 5_000
    try:
        service_model.with_copy_compute_overlap_observation(
            transfer_bytes=6 * 1024,
            isolated_copy_ns=2_400,
            isolated_compute_ns=5_000,
            concurrent_ns=4_999,
        )
    except ValueError as error:
        assert "lower bound" in str(error)
    else:
        raise AssertionError("an impossible overlap observation was accepted")

    # Copy CUDA elapsed already includes stream starvation from descriptor
    # issue. The model takes the maximum resource bound, not their sum.
    assert (
        service_model.candidate_ns(
            total_rows=4,
            copy_rows=4,
            copy_run_count=1,
            row_bytes=1024,
            copy_operations_per_run=1,
        )
        == 1_024
    )

    # Three layouts exercise all auto outcomes under one measured service
    # contract: all-copy, a strict hybrid, and SM-only.
    hybrid_source = (*range(8), 100, 101, 102, 200)
    hybrid_destination = (*range(20, 28), 40, 41, 42, 60)
    hybrid_model = _calibrated_model(copy_operation_ns=3_000)
    hybrid = plan_indexed_mover(
        hybrid_source,
        hybrid_destination,
        row_bytes=1_000,
        copy_operations_per_run=1,
        maximum_copy_runs=3,
        service_model=hybrid_model,
    )
    assert hybrid.kind == "hybrid"
    assert hybrid.copy_row_count == 8 and hybrid.sm_row_count == 4
    assert hybrid.predicted_sm_ns == 12_000
    assert hybrid.predicted_selected_ns == 7_000

    exhaustive_models = (
        _calibrated_model(copy_operation_ns=0, minimum_gain=1.0),
        _calibrated_model(
            copy_operation_ns=2_000,
            minimum_gain=1.0,
            hybrid_join_ns=700,
        ),
        replace(
            _calibrated_model(copy_operation_ns=2_000, minimum_gain=1.0),
            copy_compute_overlap_efficiency=0.5,
            overlap_samples=3,
        ),
    )
    for run_rows in (
        (1,),
        (1, 2),
        (4, 1, 3),
        (2, 5, 1, 4),
        (8, 3, 2, 1, 1),
    ):
        for maximum_runs in range(1, min(3, len(run_rows)) + 1):
            for exhaustive_model in exhaustive_models:
                _assert_prefix_matches_exhaustive_optimum(
                    run_rows,
                    maximum_copy_runs=maximum_runs,
                    service_model=exhaustive_model,
                    overlap_compute_ns=9_000,
                )
    issue_bound = plan_indexed_mover(
        source_indices,
        destination_indices,
        row_bytes=1024,
        copy_operations_per_run=1,
        maximum_copy_runs=8,
        service_model=_calibrated_model(
            copy_bandwidth=1_000_000_000_000,
            copy_operation_ns=5_000,
        ),
        overlap_compute_ns=5_000,
    )
    assert issue_bound.kind == "sm"
    assert issue_bound.selection_reason == "insufficient_gain"

    # A fixed hybrid join cost can make the first copy run lose while a longer
    # prefix amortizes that cost. The planner must evaluate every representable
    # prefix rather than stopping at the first non-improving candidate.
    join_amortized = plan_indexed_mover(
        (0, 1, 100, 101, 200, 201),
        (10, 11, 20, 21, 30, 31),
        row_bytes=50_000,
        copy_operations_per_run=1,
        maximum_copy_runs=2,
        service_model=_calibrated_model(
            copy_operation_ns=0,
            hybrid_join_ns=100_000,
        ),
    )
    assert join_amortized.kind == "hybrid"
    assert join_amortized.copy_row_count == 4
    assert join_amortized.sm_row_count == 2
    assert join_amortized.predicted_sm_ns == 300_000
    assert join_amortized.predicted_selected_ns == 250_000

    # ``minimum_gain`` is a post-optimization safety margin. It changes only
    # policy acceptance, never the modeled costs or a byte eligibility cutoff.
    margin_permissive = _calibrated_model(
        copy_bandwidth=1_050_000_000,
        copy_operation_ns=0,
        minimum_gain=1.0,
    )
    margin_strict = replace(margin_permissive, minimum_gain=1.10)
    permissive = plan_indexed_mover(
        tuple(range(100)),
        tuple(range(200, 300)),
        row_bytes=1_000,
        copy_operations_per_run=1,
        maximum_copy_runs=1,
        service_model=margin_permissive,
    )
    strict = plan_indexed_mover(
        tuple(range(100)),
        tuple(range(200, 300)),
        row_bytes=1_000,
        copy_operations_per_run=1,
        maximum_copy_runs=1,
        service_model=margin_strict,
    )
    assert permissive.kind == "copy_engine"
    assert strict.kind == "sm" and strict.selection_reason == "insufficient_gain"
    assert margin_permissive.candidate_ns(
        total_rows=100,
        copy_rows=100,
        copy_run_count=1,
        row_bytes=1_000,
        copy_operations_per_run=1,
    ) == margin_strict.candidate_ns(
        total_rows=100,
        copy_rows=100,
        copy_run_count=1,
        row_bytes=1_000,
        copy_operations_per_run=1,
    )
    forced_strict = plan_indexed_mover(
        tuple(range(100)),
        tuple(range(200, 300)),
        row_bytes=1_000,
        copy_operations_per_run=1,
        maximum_copy_runs=1,
        service_model=margin_strict,
        policy="copy_engine",
    )
    assert forced_strict.kind == "copy_engine"

    # A smaller contiguous map can use copy while a larger fragmented map uses
    # SM. Selection is service/layout based, not ``bytes >= hidden_threshold``.
    layout_sensitive = _calibrated_model(copy_operation_ns=10_000)
    small_contiguous = plan_indexed_mover(
        tuple(range(100)),
        tuple(range(300, 400)),
        row_bytes=1_000,
        copy_operations_per_run=1,
        maximum_copy_runs=200,
        service_model=layout_sensitive,
    )
    large_scattered = plan_indexed_mover(
        tuple(range(0, 400, 2)),
        tuple(range(1_000, 1_400, 2)),
        row_bytes=1_000,
        copy_operations_per_run=1,
        maximum_copy_runs=200,
        service_model=layout_sensitive,
    )
    assert small_contiguous.kind == "copy_engine"
    assert large_scattered.kind == "sm"

    scale_scoped = replace(service_model, calibration_scale_bucket=20)
    same_scale = plan_indexed_mover(
        (0,),
        (1,),
        row_bytes=1 << 20,
        copy_operations_per_run=1,
        maximum_copy_runs=1,
        service_model=scale_scoped,
        service_scale_bytes=1 << 20,
    )
    extrapolated_scale = plan_indexed_mover(
        (0,),
        (1,),
        row_bytes=1 << 21,
        copy_operations_per_run=1,
        maximum_copy_runs=1,
        service_model=scale_scoped,
        service_scale_bytes=1 << 21,
    )
    assert same_scale.kind == "copy_engine"
    assert extrapolated_scale.kind == "sm"
    assert extrapolated_scale.selection_reason == "uncalibrated_transfer_scale"

    # With measured copy/compute overlap but slower copy byte service, subset
    # cost can be non-monotone. A bounded longest-run view cannot prove the
    # global optimum, so auto fails closed while forced diagnostics remain.
    nonmonotone = _calibrated_model(
        sm_bandwidth=4_000_000_000,
        copy_bandwidth=1_000_000_000,
        copy_operation_ns=0,
    )
    nonmonotone = replace(
        nonmonotone,
        copy_compute_overlap_efficiency=1.0,
        overlap_samples=3,
    )
    unproven = plan_indexed_mover(
        source_indices,
        destination_indices,
        row_bytes=1024,
        copy_operations_per_run=1,
        maximum_copy_runs=3,
        service_model=nonmonotone,
        overlap_compute_ns=100_000,
    )
    assert unproven.kind == "sm"
    assert unproven.selection_reason == "candidate_optimality_unproven"
    output = _apply_plan(source_values, destination_indices, auto)
    for source_row, destination_row in zip(
        source_indices, destination_indices, strict=True
    ):
        assert output[destination_row] == source_values[source_row]

    capped = plan_indexed_mover(
        source_indices,
        destination_indices,
        row_bytes=1024,
        copy_operations_per_run=1,
        maximum_copy_runs=1,
        service_model=service_model,
    )
    assert tuple(
        (run.source_first, run.destination_first, run.row_count)
        for run in capped.copy_runs
    ) == ((4, 10, 3),)
    assert capped.copy_row_count == 3 and capped.sm_row_count == 3
    entirely_uncalibrated = plan_indexed_mover(
        (1, 3, 5),
        (2, 4, 6),
        row_bytes=4096,
        copy_operations_per_run=1,
        maximum_copy_runs=8,
        service_model=IndexedMoverServiceModel(
            sm_bandwidth_bytes_per_second=1_000_000_000
        ),
    )
    assert entirely_uncalibrated.kind == "sm"
    assert entirely_uncalibrated.selection_reason == "uncalibrated_sm_reference"
    copy_uncalibrated = plan_indexed_mover(
        (1, 3, 5),
        (2, 4, 6),
        row_bytes=4096,
        copy_operations_per_run=1,
        maximum_copy_runs=8,
        service_model=IndexedMoverServiceModel(
            sm_bandwidth_bytes_per_second=1_000_000_000,
            sm_samples=3,
        ),
    )
    assert copy_uncalibrated.kind == "sm"
    assert copy_uncalibrated.selection_reason == "uncalibrated_copy_engine"
    seeded_but_unmeasured = plan_indexed_mover(
        (1, 3, 5),
        (2, 4, 6),
        row_bytes=4096,
        copy_operations_per_run=1,
        maximum_copy_runs=8,
        service_model=IndexedMoverServiceModel(
            sm_bandwidth_bytes_per_second=1_000_000_000,
            copy_bandwidth_bytes_per_second=4_000_000_000,
            copy_operation_ns=0,
            sm_samples=3,
            copy_samples=0,
        ),
    )
    assert seeded_but_unmeasured.kind == "sm"
    assert seeded_but_unmeasured.selection_reason == "uncalibrated_copy_engine"
    forced_sm = plan_indexed_mover(
        source_indices,
        destination_indices,
        row_bytes=1,
        copy_operations_per_run=1,
        maximum_copy_runs=3,
        service_model=service_model,
        policy="sm",
    )
    assert forced_sm.kind == "sm"
    assert forced_sm.selection_reason == "forced_sm"
    forced = plan_indexed_mover(
        source_indices,
        destination_indices,
        row_bytes=1,
        copy_operations_per_run=1,
        maximum_copy_runs=3,
        service_model=IndexedMoverServiceModel(
            sm_bandwidth_bytes_per_second=1_000_000_000
        ),
        policy="copy_engine",
    )
    assert forced.kind == "copy_engine" and forced.copy_row_count == 6
    assert forced.predicted_selected_ns is None
    probe = plan_indexed_mover(
        source_indices,
        destination_indices,
        row_bytes=1,
        copy_operations_per_run=1,
        maximum_copy_runs=1,
        service_model=IndexedMoverServiceModel(
            sm_bandwidth_bytes_per_second=1_000_000_000
        ),
        policy="probe_copy",
    )
    assert probe.kind == "hybrid"
    assert probe.copy_row_count == 3 and probe.sm_row_count == 3
    assert probe.selection_reason == "calibration_probe_copy"

    tensor_plan = plan_indexed_tensor_mover(
        torch.tensor(source_indices, dtype=torch.int32),
        torch.tensor(destination_indices, dtype=torch.int64),
        row_bytes=1024,
        copy_operations_per_run=1,
        maximum_copy_runs=8,
        service_model=service_model,
    )
    assert tensor_plan.kind == auto.kind
    assert tensor_plan.layout == analyze_index_pairs(
        source_indices, destination_indices
    )
    assert tensor_plan.copy_runs == auto.copy_runs
    assert tuple(tensor_plan.sm_source_indices.tolist()) == auto.sm_source_indices
    assert (
        tuple(tensor_plan.sm_destination_indices.tolist())
        == auto.sm_destination_indices
    )
    bounded_tensor_plan = plan_indexed_tensor_mover(
        torch.tensor(source_indices, dtype=torch.int32),
        torch.tensor(destination_indices, dtype=torch.int64),
        row_bytes=1024,
        copy_operations_per_run=1,
        maximum_copy_runs=8,
        service_model=service_model,
        capture_full_layout=False,
    )
    assert bounded_tensor_plan.layout is None
    assert bounded_tensor_plan.total_run_count == len(tensor_plan.layout.runs)
    assert bounded_tensor_plan.copy_runs == tensor_plan.copy_runs
    assert torch.equal(
        bounded_tensor_plan.sm_source_indices, tensor_plan.sm_source_indices
    )
    assert torch.equal(
        bounded_tensor_plan.sm_destination_indices,
        tensor_plan.sm_destination_indices,
    )
    bounded_hybrid_tensor_plan = plan_indexed_tensor_mover(
        torch.tensor(source_indices, dtype=torch.int32),
        torch.tensor(destination_indices, dtype=torch.int64),
        row_bytes=1024,
        copy_operations_per_run=1,
        maximum_copy_runs=1,
        service_model=service_model,
        policy="probe_copy",
        capture_full_layout=False,
    )
    assert bounded_hybrid_tensor_plan.kind == "hybrid"
    assert bounded_hybrid_tensor_plan.copy_row_count == probe.copy_row_count
    assert bounded_hybrid_tensor_plan.sm_row_count == probe.sm_row_count
    assert warm_indexed_tensor_mover(
        "cpu", maximum_rows=8, maximum_copy_runs=3
    ) > 0
    assert (
        warm_indexed_tensor_mover("cpu", maximum_rows=8, maximum_copy_runs=3)
        == 0
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent_warmups = tuple(
            pool.map(
                lambda _index: warm_indexed_tensor_mover(
                    "cpu", maximum_rows=9, maximum_copy_runs=2
                ),
                range(4),
            )
        )
    assert sum(elapsed_ns > 0 for elapsed_ns in concurrent_warmups) == 1
    compute_aware_tensor = plan_indexed_tensor_mover(
        torch.tensor(source_indices, dtype=torch.int32),
        torch.tensor(destination_indices, dtype=torch.int64),
        row_bytes=1024,
        copy_operations_per_run=1,
        maximum_copy_runs=8,
        service_model=service_model,
        overlap_compute_ns=5_000,
    )
    assert compute_aware_tensor.kind == "copy_engine"
    assert compute_aware_tensor.copy_row_count == 6
    assert compute_aware_tensor.sm_row_count == 0
    try:
        plan_indexed_tensor_mover(
            torch.tensor((1, 2), dtype=torch.int64),
            torch.tensor((3, 3), dtype=torch.int64),
            row_bytes=1,
            copy_operations_per_run=1,
            maximum_copy_runs=2,
            service_model=service_model,
        )
    except ValueError as error:
        assert "destinations must be unique" in str(error)
    else:
        raise AssertionError("tensor mover accepted overlapping destinations")
    try:
        plan_indexed_tensor_mover(
            torch.tensor((1 << 31,), dtype=torch.int64),
            torch.tensor((0,), dtype=torch.int64),
            row_bytes=1,
            copy_operations_per_run=1,
            maximum_copy_runs=1,
            service_model=service_model,
        )
    except ValueError as error:
        assert "signed int32" in str(error)
    else:
        raise AssertionError("tensor mover accepted an int32-wrapping index")

    dependencies = plan_indexed_dependencies(
        (
            ((4, 10), (5, 11), (20, 30)),
            ((4, 10), (5, 11), (6, 12)),
            (),
        )
    )
    assert dependencies.source_indices == (4, 5, 6, 20)
    assert dependencies.destination_indices == (10, 11, 12, 30)
    assert tuple(
        (
            run.pair_offset,
            run.source_first,
            run.destination_first,
            run.row_count,
            run.work_ids,
        )
        for run in dependencies.runs
    ) == (
        (0, 4, 10, 2, (0, 1)),
        (2, 6, 12, 1, (1,)),
        (3, 20, 30, 1, (0,)),
    )
    assert dependencies.run_indices_by_work == ((0, 2), (0, 1), ())
    for invalid in (
        (((1, 3), (1, 3)),),
        (((1, 3),), ((2, 3),)),
        (((-1, 3),),),
    ):
        try:
            plan_indexed_dependencies(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid indexed dependency map was accepted")
    try:
        plan_indexed_mover(
            source_indices,
            destination_indices,
            row_bytes=1,
            copy_operations_per_run=1,
            maximum_copy_runs=2,
            service_model=service_model,
            policy="copy_engine",
        )
    except ValueError as error:
        assert "operation bound" in str(error)
    else:
        raise AssertionError("forced copy-engine policy exceeded its operation bound")

    group = StridedCopyGroup(1, 2, 64, 32, 256, 1024, 512)
    assert group.source_stride_bytes == 1024
    for arguments in (
        (0, 2, 64, 32, 256, 1024, 512),
        (1, 2, 64, 32, 256, 128, 512),
        (1, 2, 1 << 32, 32, 256, 1024, 512),
    ):
        try:
            StridedCopyGroup(*arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid strided-copy geometry was accepted")

    for source, destination in (
        ((), ()),
        ((0,), (0, 1)),
        ((0,), (-1,)),
        ((0, 1), (3, 3)),
    ):
        try:
            analyze_index_pairs(source, destination)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid indexed-transfer map was accepted")
    print("indexed_transfer=pass")


if __name__ == "__main__":
    main()
