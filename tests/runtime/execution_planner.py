#!/usr/bin/env python3

from nta_runtime.execution_planner import (
    conservative_resume_counts,
    DeviceDemandCostModel,
    HostCostModel,
    indexed_copy_blocks_per_group,
    plan_device_demand,
    plan_host_execution,
)


def main() -> None:
    calibrated = HostCostModel.from_environment(
        {"NTA_TIER_HOST_STAGED_BANDWIDTH_BPS": "123456789"}
    )
    assert calibrated.bandwidth_bytes_per_second == 123456789
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

    forced = plan_host_execution(
        object_count=16,
        transfer_bytes=64 * 1024,
        runnable_tiles=8,
        model=model,
        force_rounds=2,
    )
    assert forced.rounds == 2
    assert forced.block_counts == (8, 8)
    assert forced.predicted_incremental_ns > forced.predicted_atomic_ns
    one_round = plan_host_execution(
        object_count=16,
        transfer_bytes=64 * 1024,
        runnable_tiles=8,
        model=model,
        force_rounds=1,
    )
    assert one_round.rounds == 1
    assert one_round.block_counts == (16,)
    try:
        plan_host_execution(
            object_count=16,
            transfer_bytes=64 * 1024,
            runnable_tiles=8,
            model=model,
            force_rounds=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("zero forced host rounds were accepted")
    try:
        plan_host_execution(
            object_count=0, transfer_bytes=1, runnable_tiles=1, model=model
        )
    except ValueError:
        pass
    else:
        raise AssertionError("empty host work was accepted")

    device_model = DeviceDemandCostModel()
    bulk = plan_device_demand(
        candidate_bytes=8 * 1024 * 1024,
        selected_bytes=8 * 1024 * 1024,
        selected_units=512,
        model=device_model,
    )
    assert bulk.mode == "bulk"
    crossover = plan_device_demand(
        candidate_bytes=16 * 1024 * 1024,
        selected_bytes=8 * 1024 * 1024,
        selected_units=512,
        model=device_model,
    )
    assert crossover.mode == "indexed"
    selective = plan_device_demand(
        candidate_bytes=128 * 1024 * 1024,
        selected_bytes=8 * 1024 * 1024,
        selected_units=512,
        model=device_model,
    )
    assert selective.mode == "indexed"
    assert selective.predicted_gain >= device_model.minimum_predicted_gain
    try:
        plan_device_demand(
            candidate_bytes=1024,
            selected_bytes=2048,
            selected_units=1,
            model=device_model,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("oversized selected demand was accepted")


if __name__ == "__main__":
    main()
