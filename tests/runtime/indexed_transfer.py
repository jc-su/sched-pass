#!/usr/bin/env python3
"""Validate exact paired-run decomposition for transfer profiling/planning."""

import torch

from nta_runtime.indexed_transfer import (
    IndexedMoverServiceModel,
    StridedCopyGroup,
    analyze_index_pairs,
    plan_indexed_dependencies,
    plan_indexed_mover,
)
from nta_runtime.indexed_transfer_torch import plan_indexed_tensor_mover
from nta_runtime.engines.sglang_transfer import (
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


def main() -> None:
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
    assert calibrated_environment.sm_samples == 1
    assert calibrated_environment.copy_samples == 1
    try:
        host_mover_service_model_from_environment(
            {"NTA_EXECUTION_HOST_COPY_BANDWIDTH_BPS": "20"}
        )
    except ValueError as error:
        assert "requires both" in str(error)
    else:
        raise AssertionError("partial mover calibration was accepted")

    service_model = IndexedMoverServiceModel(
        sm_bandwidth_bytes_per_second=1_000_000_000,
        copy_bandwidth_bytes_per_second=4_000_000_000,
        copy_operation_ns=800,
        minimum_gain=1.03,
    )
    observed_sm = service_model.with_sm_observation(
        transfer_bytes=1 << 20,
        elapsed_ns=1_000_000,
        alpha=1.0,
    )
    assert observed_sm.sm_samples == 1
    assert observed_sm.sm_bandwidth_bytes_per_second == 1_048_576_000
    observed_copy = IndexedMoverServiceModel(
        sm_bandwidth_bytes_per_second=observed_sm.sm_bandwidth_bytes_per_second,
        sm_samples=observed_sm.sm_samples,
    ).with_copy_observation(
        transfer_bytes=1 << 20,
        elapsed_ns=400_000,
        operation_count=8,
        issue_cpu_ns=80_000,
        alpha=1.0,
    )
    assert observed_copy.copy_samples == 1
    assert observed_copy.copy_calibrated
    assert observed_copy.copy_operation_ns == 10_000
    assert observed_copy.copy_bandwidth_bytes_per_second > (
        observed_copy.sm_bandwidth_bytes_per_second
    )
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
    assert auto.kind == "hybrid"
    assert auto.copy_row_count == 5 and auto.sm_row_count == 1
    compute_aware = plan_indexed_mover(
        source_indices,
        destination_indices,
        row_bytes=1024,
        copy_operations_per_run=1,
        maximum_copy_runs=8,
        service_model=service_model,
        overlap_compute_ns=5_000,
    )
    assert compute_aware.kind == "copy_engine"
    assert compute_aware.copy_row_count == 6 and compute_aware.sm_row_count == 0
    assert compute_aware.predicted_selected_ns == 5_000
    assert compute_aware.predicted_sm_ns == 11_144
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
    scattered = plan_indexed_mover(
        (1, 3, 5),
        (2, 4, 6),
        row_bytes=4096,
        copy_operations_per_run=1,
        maximum_copy_runs=8,
        service_model=IndexedMoverServiceModel(
            sm_bandwidth_bytes_per_second=1_000_000_000
        ),
    )
    assert scattered.kind == "sm" and scattered.sm_row_count == 3
    assert scattered.selection_reason == "uncalibrated_copy_engine"
    forced = plan_indexed_mover(
        source_indices,
        destination_indices,
        row_bytes=1,
        copy_operations_per_run=1,
        maximum_copy_runs=3,
        service_model=service_model,
        policy="copy_engine",
    )
    assert forced.kind == "copy_engine" and forced.copy_row_count == 6
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
        (run.pair_offset, run.source_first, run.destination_first, run.row_count, run.work_ids)
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
