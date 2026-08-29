#!/usr/bin/env python3

from dataclasses import replace
import math

from nta_runtime.execution_planner import (
    conservative_resume_counts,
    HostCostModel,
    HostExecutionForm,
    HostExecutionMode,
    HostExecutionPlan,
    indexed_copy_blocks_per_group,
    plan_exact_runnable_windows,
    plan_host_layer_execution,
    plan_host_execution,
    prove_atomic_host_execution,
)


def main() -> None:
    calibrated = HostCostModel.from_environment(
        {"NTA_TIER_HOST_STAGED_BANDWIDTH_BPS": "123456789"}
    )
    assert calibrated.bandwidth_bytes_per_second == 123456789
    updated = calibrated.with_transfer_observation(
        transfer_bytes=1 << 20,
        elapsed_ns=40_000_000,
        alpha=0.5,
        maximum_step_ratio=2.0,
    )
    assert updated.bandwidth_bytes_per_second < calibrated.bandwidth_bytes_per_second
    assert (
        updated.bandwidth_bytes_per_second >= calibrated.bandwidth_bytes_per_second // 2
    )
    assert calibrated.incremental_setup_ns is None
    setup = calibrated.with_incremental_setup_observation(elapsed_ns=2_000_000)
    assert setup.incremental_setup_ns == 2_000_000
    setup = setup.with_incremental_setup_observation(elapsed_ns=4_000_000, alpha=0.5)
    assert setup.incremental_setup_ns == 3_000_000
    assert (
        calibrated.with_transfer_observation(
            transfer_bytes=1024,
            elapsed_ns=1_000,
        )
        is calibrated
    )
    for invalid in (
        {"transfer_bytes": 0, "elapsed_ns": 1},
        {"transfer_bytes": 1, "elapsed_ns": 0},
        {"transfer_bytes": 1, "elapsed_ns": 1, "alpha": 0.0},
    ):
        try:
            calibrated.with_transfer_observation(**invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid transfer calibration was accepted")
    assert (
        indexed_copy_blocks_per_group(transfer_bytes=4 * 1024 * 1024, object_count=2)
        == 4
    )
    assert indexed_copy_blocks_per_group(transfer_bytes=64 * 1024, object_count=2) == 1
    assert (
        indexed_copy_blocks_per_group(transfer_bytes=128 * 1024 * 1024, object_count=2)
        == 32
    )
    assert conservative_resume_counts(
        block_counts=(8, 8),
        work_count=8,
        max_object_fanout=1,
        min_unresolved_dependencies=2,
    ) == (4, 8)
    assert conservative_resume_counts(
        block_counts=(2, 2),
        work_count=32,
        max_object_fanout=32,
        min_unresolved_dependencies=2,
    ) == (32, 32)
    try:
        conservative_resume_counts(
            block_counts=(0,),
            work_count=1,
            max_object_fanout=1,
            min_unresolved_dependencies=1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("an empty resume wave was accepted")

    shared_windows = plan_exact_runnable_windows(
        external_object_slots=((0, 1), (0, 1), (2, 3), (4, 5), ()),
        first_unresolved_object=0,
        block_counts=(2, 2, 2),
    )
    assert shared_windows.initial_count == 1
    assert shared_windows.offsets == (1, 3, 4)
    assert shared_windows.counts == (2, 1, 1)
    assert shared_windows.work_count == 5
    assert shared_windows.launch_count == 4
    preloaded_windows = plan_exact_runnable_windows(
        external_object_slots=((0, 1), (0, 1), (2, 3), (4, 5), ()),
        first_unresolved_object=2,
        block_counts=(2, 2),
    )
    assert preloaded_windows.initial_count == 3
    assert preloaded_windows.offsets == (3, 4)
    assert preloaded_windows.counts == (1, 1)
    empty_early_waves = plan_exact_runnable_windows(
        external_object_slots=((4, 5),),
        first_unresolved_object=0,
        block_counts=(2, 2, 2),
    )
    assert empty_early_waves.offsets == (0, 0, 0)
    assert empty_early_waves.counts == (0, 0, 1)
    assert empty_early_waves.launch_count == 1
    try:
        plan_exact_runnable_windows(
            external_object_slots=((6, 7),),
            first_unresolved_object=0,
            block_counts=(2, 2, 2),
        )
    except ValueError as error:
        assert "not covered" in str(error)
    else:
        raise AssertionError("an uncovered runnable dependency was accepted")

    model = HostCostModel(
        bandwidth_bytes_per_second=20_000_000_000,
        round_overhead_ns=20_000,
        incremental_setup_ns=0,
        tile_compute_ns=4_000,
        max_rounds=4,
        minimum_predicted_gain=1.03,
    )
    pipelined = plan_host_execution(
        object_count=16, transfer_bytes=4 * 1024 * 1024, runnable_tiles=64, model=model
    )
    assert pipelined.rounds > 1
    assert sum(pipelined.block_counts) == 16
    assert all(count % 2 == 0 for count in pipelined.block_counts)
    assert pipelined.predicted_gain >= model.minimum_predicted_gain
    assert pipelined.form is HostExecutionForm.DEPENDENCY_AWARE
    assert pipelined.uses_dependency_protocol

    atomic = plan_host_execution(
        object_count=2, transfer_bytes=64 * 1024, runnable_tiles=1, model=model
    )
    assert atomic.block_counts == (2,)
    assert atomic.predicted_gain == 1.0
    assert not atomic.overlap_initial
    assert atomic.form is HostExecutionForm.DIRECT
    assert not atomic.uses_dependency_protocol

    forced_direct = plan_host_execution(
        object_count=16,
        transfer_bytes=4 * 1024 * 1024,
        runnable_tiles=64,
        model=model,
        mode=HostExecutionMode.DIRECT,
    )
    assert forced_direct.form is HostExecutionForm.DIRECT
    assert forced_direct.selection_reason == "forced_direct"
    forced_device_bulk = plan_host_execution(
        object_count=16,
        transfer_bytes=4 * 1024 * 1024,
        runnable_tiles=64,
        initial_runnable_tiles=17,
        model=replace(model, incremental_setup_ns=None),
        mode=HostExecutionMode.DEVICE_BULK,
    )
    assert forced_device_bulk.form is HostExecutionForm.DEVICE_BULK
    assert forced_device_bulk.block_counts == (16,)
    assert not forced_device_bulk.overlap_initial
    assert forced_device_bulk.uses_dependency_protocol
    assert forced_device_bulk.uses_device_bulk
    assert not forced_device_bulk.uses_progressive_consumer
    assert forced_device_bulk.selection_reason == "forced_device_bulk"
    bulk_template = plan_host_layer_execution(
        host_execution=HostExecutionPlan(
            block_counts=(6,),
            predicted_atomic_ns=10,
            predicted_incremental_ns=11,
            form=HostExecutionForm.DEVICE_BULK,
            selection_reason="forced_device_bulk",
        ),
        object_count=6,
        work_count=5,
        transfer_bytes=600,
        object_transfer_bytes=(100,) * 6,
        external_object_slots=((), (0, 1), (0, 1), (2, 3), (4, 5)),
        direct_work_count=1,
        max_object_fanout=2,
        min_unresolved_dependencies=2,
        preloaded_object_count=0,
        queued_feasible_edf=False,
        indexed_copy_target_bytes=1024,
        indexed_copy_max_blocks=8,
    )
    assert bulk_template.progress_blocks == (6,)
    assert bulk_template.ready_work_counts == (5,)
    assert bulk_template.ready_work_offsets == (0,)
    assert bulk_template.initial_ready_work_count == 0
    assert not bulk_template.progressive_consumer
    assert not bulk_template.exact_resume_windows

    dependency_form = HostExecutionPlan(
        block_counts=(2, 2, 2),
        predicted_atomic_ns=20,
        predicted_incremental_ns=10,
        form=HostExecutionForm.DEPENDENCY_AWARE,
        selection_reason="forced_dependency_aware",
    )
    exact_template = plan_host_layer_execution(
        host_execution=dependency_form,
        object_count=6,
        work_count=5,
        transfer_bytes=600,
        object_transfer_bytes=(100,) * 6,
        external_object_slots=((), (0, 1), (0, 1), (2, 3), (4, 5)),
        direct_work_count=1,
        max_object_fanout=2,
        min_unresolved_dependencies=2,
        preloaded_object_count=0,
        queued_feasible_edf=False,
        indexed_copy_target_bytes=1024,
        indexed_copy_max_blocks=8,
    )
    assert exact_template.ready_work_offsets == (1, 3, 4)
    assert exact_template.ready_work_counts == (2, 1, 1)
    assert exact_template.progressive_consumer
    assert exact_template.exact_resume_windows

    queued_template = plan_host_layer_execution(
        host_execution=dependency_form,
        object_count=6,
        work_count=5,
        transfer_bytes=600,
        object_transfer_bytes=(100,) * 6,
        external_object_slots=((), (0, 1), (0, 1), (2, 3), (4, 5)),
        direct_work_count=1,
        max_object_fanout=2,
        min_unresolved_dependencies=2,
        preloaded_object_count=0,
        queued_feasible_edf=True,
        indexed_copy_target_bytes=1024,
        indexed_copy_max_blocks=8,
    )
    assert queued_template.ready_work_offsets is None
    assert queued_template.progressive_consumer
    assert not queued_template.indexed_host_prevalidated
    forced_dependency = plan_host_execution(
        object_count=2,
        transfer_bytes=64 * 1024,
        runnable_tiles=1,
        model=replace(model, incremental_setup_ns=None),
        mode=HostExecutionMode.DEPENDENCY_AWARE,
    )
    assert forced_dependency.form is HostExecutionForm.DEPENDENCY_AWARE
    assert forced_dependency.selection_reason == "forced_dependency_aware"

    uncalibrated = plan_host_execution(
        object_count=16,
        transfer_bytes=4 * 1024 * 1024,
        runnable_tiles=64,
        model=HostCostModel(
            bandwidth_bytes_per_second=20_000_000_000,
            tile_compute_ns=4_000,
        ),
    )
    assert uncalibrated.rounds == 1
    assert uncalibrated.selection_reason == "uncalibrated_setup"
    assert uncalibrated.form is HostExecutionForm.DIRECT
    probe = plan_host_execution(
        object_count=16,
        transfer_bytes=4 * 1024 * 1024,
        runnable_tiles=64,
        model=HostCostModel(
            bandwidth_bytes_per_second=20_000_000_000,
            tile_compute_ns=4_000,
        ),
        calibration_probe=True,
    )
    assert probe.rounds > 1
    assert probe.selection_reason == "calibration_probe"
    assert probe.form is HostExecutionForm.DEPENDENCY_AWARE

    request_overlap = plan_host_execution(
        object_count=2,
        transfer_bytes=2_000_000,
        runnable_tiles=50,
        initial_runnable_tiles=30,
        model=model,
    )
    assert request_overlap.block_counts == (2,)
    assert request_overlap.overlap_initial
    assert request_overlap.predicted_gain >= model.minimum_predicted_gain
    assert request_overlap.form is HostExecutionForm.DEPENDENCY_AWARE

    overhead_dominated = plan_host_execution(
        object_count=16,
        transfer_bytes=64 * 1024,
        runnable_tiles=8,
        model=model,
    )
    assert overhead_dominated.rounds == 1
    assert overhead_dominated.block_counts == (16,)
    assert overhead_dominated.predicted_gain == 1.0
    assert overhead_dominated.form is HostExecutionForm.DIRECT
    overhead_probe = plan_host_execution(
        object_count=16,
        transfer_bytes=64 * 1024,
        runnable_tiles=8,
        model=model,
        calibration_probe=True,
    )
    assert overhead_probe.form is HostExecutionForm.DEPENDENCY_AWARE
    assert overhead_probe.selection_reason == "calibration_probe"

    # Finite tenant budgets are a correctness contract, not a profitability
    # hint.  Even an overhead-dominated one-wave transfer must retain native
    # request generation, object ownership, and credit accounting.
    isolated = plan_host_execution(
        object_count=16,
        transfer_bytes=64 * 1024,
        runnable_tiles=8,
        model=model,
        require_dependency_protocol=True,
    )
    assert isolated.rounds == 1
    assert isolated.block_counts == (16,)
    assert isolated.form is HostExecutionForm.DEPENDENCY_AWARE
    assert isolated.uses_dependency_protocol
    assert isolated.selection_reason == "tenant_isolation"

    # The conventional arm is itself a cross-layer pipeline: while attention
    # consumes layer i, the mover can prepare layer i + 1.  Multiplying the
    # one-layer transfer+compute sum by 36 invents 35 serialization bubbles
    # and used to produce a false incremental win.
    expensive_setup = replace(model, incremental_setup_ns=2_000_000)
    single_layer = plan_host_execution(
        object_count=16,
        transfer_bytes=4 * 1024 * 1024,
        runnable_tiles=64,
        model=expensive_setup,
    )
    whole_forward = plan_host_execution(
        object_count=16,
        transfer_bytes=4 * 1024 * 1024,
        runnable_tiles=64,
        model=expensive_setup,
        scope_units=36,
    )
    assert single_layer.rounds == 1
    assert single_layer.form is HostExecutionForm.DIRECT
    assert whole_forward.rounds == 1
    assert whole_forward.form is HostExecutionForm.DIRECT
    assert whole_forward.scope_units == 36
    transfer_ns = math.ceil(
        4 * 1024 * 1024 * 1_000_000_000 / expensive_setup.bandwidth_bytes_per_second
    )
    compute_ns = 64 * expensive_setup.tile_compute_ns
    assert whole_forward.predicted_atomic_ns == (
        transfer_ns + compute_ns + 35 * max(transfer_ns, compute_ns)
    )
    assert whole_forward.predicted_atomic_ns < 36 * (transfer_ns + compute_ns)

    # This fast-path is a proof, not another empirical threshold: it may skip
    # graph construction only when perfect overlap still cannot pay setup.
    proved_atomic = prove_atomic_host_execution(
        object_count=16,
        transfer_bytes=64 * 1024,
        runnable_tiles=8,
        model=replace(model, incremental_setup_ns=2_000_000),
    )
    assert proved_atomic is not None
    assert proved_atomic.rounds == 1
    assert proved_atomic.form is HostExecutionForm.DIRECT
    assert proved_atomic.selection_reason == "insufficient_gain"
    assert (
        prove_atomic_host_execution(
            object_count=16,
            transfer_bytes=4 * 1024 * 1024,
            runnable_tiles=64,
            model=model,
        )
        is None
    )
    assert (
        prove_atomic_host_execution(
            object_count=16,
            transfer_bytes=4 * 1024 * 1024,
            runnable_tiles=64,
            model=model,
            mode=HostExecutionMode.DEVICE_BULK,
        )
        is None
    )
    pipelined_proof = prove_atomic_host_execution(
        object_count=16,
        transfer_bytes=4 * 1024 * 1024,
        runnable_tiles=64,
        model=expensive_setup,
        scope_units=36,
    )
    assert pipelined_proof is not None
    assert pipelined_proof.form is HostExecutionForm.DIRECT
    assert pipelined_proof.selection_reason == "insufficient_gain"
    uncalibrated_proof = prove_atomic_host_execution(
        object_count=16,
        transfer_bytes=4 * 1024 * 1024,
        runnable_tiles=64,
        model=replace(model, incremental_setup_ns=None),
    )
    assert uncalibrated_proof is not None
    assert uncalibrated_proof.form is HostExecutionForm.DIRECT
    assert uncalibrated_proof.selection_reason == "uncalibrated_setup"
    try:
        plan_host_execution(
            object_count=0, transfer_bytes=1, runnable_tiles=1, model=model
        )
    except ValueError:
        pass
    else:
        raise AssertionError("empty host work was accepted")


if __name__ == "__main__":
    main()
