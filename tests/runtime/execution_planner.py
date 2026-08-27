#!/usr/bin/env python3

from dataclasses import replace

from nta_runtime.execution_planner import (
    conservative_resume_counts,
    HostCostModel,
    LayerDeadlineServiceCurve,
    indexed_copy_blocks_per_group,
    plan_host_execution,
    prove_atomic_host_execution,
)


def main() -> None:
    deadline_curve = LayerDeadlineServiceCurve(
        minimum_samples=3, maximum_samples=4
    )
    for sample in (2_000, 1_500):
        deadline_curve = deadline_curve.with_observation(sample)
    assert not deadline_curve.calibrated
    assert deadline_curve.overlap_budget_ns(35) == 0
    deadline_curve = deadline_curve.with_observation(1_750)
    assert deadline_curve.calibrated
    assert deadline_curve.conservative_layer_ns == 1_500
    assert deadline_curve.overlap_budget_ns(35) == 52_500
    deadline_curve = deadline_curve.with_observation(1_600)
    deadline_curve = deadline_curve.with_observation(1_700)
    assert deadline_curve.samples_ns == (1_500, 1_750, 1_600, 1_700)

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
    assert updated.bandwidth_bytes_per_second >= calibrated.bandwidth_bytes_per_second // 2
    assert calibrated.incremental_setup_ns is None
    setup = calibrated.with_incremental_setup_observation(elapsed_ns=2_000_000)
    assert setup.incremental_setup_ns == 2_000_000
    setup = setup.with_incremental_setup_observation(
        elapsed_ns=4_000_000, alpha=0.5
    )
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

    atomic = plan_host_execution(
        object_count=2, transfer_bytes=64 * 1024, runnable_tiles=1, model=model
    )
    assert atomic.block_counts == (2,)
    assert atomic.predicted_gain == 1.0
    assert not atomic.overlap_initial

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

    overhead_dominated = plan_host_execution(
        object_count=16,
        transfer_bytes=64 * 1024,
        runnable_tiles=8,
        model=model,
    )
    assert overhead_dominated.rounds == 1
    assert overhead_dominated.block_counts == (16,)
    assert overhead_dominated.predicted_gain == 1.0

    # Metadata/setup is a batch cost, not a per-layer toll.  One layer cannot
    # amortize this setup, while a 36-layer model can select the same per-layer
    # wave geometry from a profitable whole-forward decision.
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
    assert whole_forward.rounds > 1
    assert whole_forward.scope_units == 36
    assert whole_forward.predicted_gain >= model.minimum_predicted_gain
    assert whole_forward.predicted_atomic_per_unit_ns == (
        single_layer.predicted_atomic_ns
    )
    assert whole_forward.predicted_incremental_per_unit_ns < (
        whole_forward.predicted_atomic_per_unit_ns
    )

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
            model=expensive_setup,
            scope_units=36,
        )
        is None
    )
    assert (
        prove_atomic_host_execution(
            object_count=16,
            transfer_bytes=4 * 1024 * 1024,
            runnable_tiles=64,
            model=replace(model, incremental_setup_ns=None),
        )
        is None
    )
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
