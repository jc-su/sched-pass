#!/usr/bin/env python3

from nta_runtime.execution_policy import HostCostModel, plan_host_execution


def main() -> None:
    model = HostCostModel(
        bandwidth_bytes_per_second=20_000_000_000,
        round_overhead_ns=20_000,
        tile_compute_ns=4_000,
        max_rounds=4,
        minimum_predicted_gain=1.03,
    )
    pipelined = plan_host_execution(
        object_count=16,
        transfer_bytes=4 * 1024 * 1024,
        runnable_tiles=64,
        model=model,
    )
    assert pipelined.rounds > 1
    assert sum(pipelined.block_counts) == 16
    assert all(count % 2 == 0 for count in pipelined.block_counts)
    assert pipelined.predicted_gain >= model.minimum_predicted_gain

    atomic = plan_host_execution(
        object_count=2,
        transfer_bytes=64 * 1024,
        runnable_tiles=1,
        model=model,
    )
    assert atomic.block_counts == (2,)
    assert atomic.predicted_gain == 1.0

    try:
        plan_host_execution(
            object_count=0,
            transfer_bytes=1,
            runnable_tiles=1,
            model=model,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("empty host work was accepted")


if __name__ == "__main__":
    main()
