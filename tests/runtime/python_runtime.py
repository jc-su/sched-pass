#!/usr/bin/env python3
"""Exercise the owning Python engine boundary against real CUDA allocations."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from nta_runtime import (
    AcquireRequirement,
    DeviceWorkPlan,
    IndexedHostIndexBinding,
    IndexedHostObject,
    JitPhaseProgram,
    CxlDaxOptions,
    NvmeOptions,
    OperatorCapability,
    OperatorContract,
    OperatorFamily,
    OperatorForm,
    OperatorInstrumentation,
    OperatorPartialState,
    OperatorPlan,
    OperatorPlanFlag,
    OperatorReduction,
    Placement,
    Replica,
    RequestRange,
    Runtime,
    RuntimeConfig,
    RequestSpec,
    WorkItem,
    copy_strided_host_runs_async,
    device_abi_version,
)
from nta_runtime.indexed_transfer import (
    StridedCopyGroup,
    analyze_index_pairs,
)


def main() -> None:
    for factory in (
        lambda: RuntimeConfig(1 << 32, 1, 1, 1),
        lambda: RequestSpec(0, 1 << 64, 0),
        lambda: RequestSpec(0, 17, 0),
        lambda: RequestSpec(0, 17, 1, priority=8),
        lambda: Replica(1, Placement.HBM, estimated_latency_ns=1 << 64),
        lambda: NvmeOptions("vfio:0000:00:00.0", queue_depth=1 << 32),
        lambda: CxlDaxOptions("/dev/dax0.0", 1 << 64),
    ):
        try:
            factory()
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-range native ABI input was accepted")

    # Reset is a phase-boundary operation, so an oversized structural schedule
    # must fail in Python before the native reset kernel can silently truncate
    # it to the runtime directory capacity.
    phase_program = object.__new__(JitPhaseProgram)
    phase_program._handle = None
    bounded_runtime = SimpleNamespace(
        config=SimpleNamespace(object_capacity=3, work_ticket_capacity=4)
    )
    for object_count, work_ticket_count in ((-1, 1), (4, 1), (0, 0), (0, 5)):
        try:
            phase_program.reset(
                bounded_runtime,
                object_count=object_count,
                work_ticket_count=work_ticket_count,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-capacity phase reset was accepted")
    try:
        OperatorContract(
            1,
            29,
            OperatorFamily.GENERIC,
            OperatorForm.DIRECT,
            OperatorCapability(0),
            "0" * 32,
            OperatorInstrumentation(1 << 8),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unknown operator instrumentation was accepted")
    try:
        OperatorPlan(
            1,
            29,
            OperatorFamily.GENERIC,
            1 << 8,
            0,
            OperatorPartialState.NONE,
            OperatorReduction.NONE,
            OperatorPlanFlag(0),
            "0" * 32,
            "1" * 32,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unknown operator plan form was accepted")
    assert device_abi_version() > 0
    source = torch.empty(4096, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()
    with Runtime(
        RuntimeConfig(
            request_capacity=2,
            object_capacity=2,
            intent_capacity=4,
            work_ticket_capacity=2,
            max_dependencies_per_work_ticket=1,
        )
    ) as runtime:
        runtime.set_tenant_budget(0, 4096)
        runtime.set_request(0, 17, 3, priority=4)
        direct = runtime.register_object(
            0,
            101,
            7,
            source.numel(),
            [Replica(source.data_ptr(), Placement.HBM)],
        )
        assert direct == source.data_ptr()
        with DeviceWorkPlan(2, 2, runtime.device_ordinal) as plan:
            plan.upload(
                [WorkItem(0, 0, 3, 11, 0, 1, 1, 0, 0, 0, 1, 2500)],
                [AcquireRequirement(direct, 0, 101, 0, 0, 7, 4096, 0)],
                [RequestRange(0, 1, 0, 3)],
                stream,
            )
            plan.wait_on(torch.cuda.current_stream())
            torch.cuda.synchronize()
            plan.mark_consumed(torch.cuda.current_stream())
            assert plan.work_items_address != 0
            assert plan.dependencies_address != 0
            assert plan.work_item_count == 1
            assert plan.dependency_count == 1
            assert plan.device_ordinal == runtime.device_ordinal
            assert plan.work_items_tensor.data_ptr() == plan.work_items_address
            assert plan.dependencies_tensor.data_ptr() == plan.dependencies_address
            assert not plan.has_external
        assert runtime.device_view_tensor.data_ptr() == runtime.device_view
        host_backing = torch.arange(128, dtype=torch.uint8, pin_memory=True).view(
            4, 2, 16
        )
        host_rows = host_backing[:, 0, :]
        staging_backing = torch.full((4, 2, 16), 0xA5, dtype=torch.uint8, device="cuda")
        staging_rows = staging_backing[:, 0, :]
        source_indices = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
        staging_indices = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
        runtime.register_indexed_host_objects(
            1,
            [
                IndexedHostObject(
                    object_id=102,
                    version=8,
                    source_device_address=host_rows.data_ptr(),
                    staging_device_address=staging_rows.data_ptr(),
                    source_indices_device_address=source_indices.data_ptr(),
                    staging_indices_device_address=staging_indices.data_ptr(),
                    index_count=2,
                    element_bytes=16,
                    source_stride_bytes=host_rows.stride(0) * host_rows.element_size(),
                    staging_stride_bytes=staging_rows.stride(0)
                    * staging_rows.element_size(),
                    source_index_limit=4,
                    staging_index_limit=4,
                )
            ],
            stream=stream,
            index_binding=IndexedHostIndexBinding(
                source_indices.data_ptr(),
                staging_indices.data_ptr(),
                int(source_indices.numel()),
            ),
        )
        layout = analyze_index_pairs((3, 1), (0, 2))
        submissions = copy_strided_host_runs_async(
            (
                StridedCopyGroup(
                    source_address=host_rows.data_ptr(),
                    destination_address=staging_rows.data_ptr(),
                    source_rows=4,
                    destination_rows=4,
                    row_bytes=16,
                    source_stride_bytes=host_rows.stride(0) * host_rows.element_size(),
                    destination_stride_bytes=staging_rows.stride(0)
                    * staging_rows.element_size(),
                ),
            ),
            layout.runs,
            stream,
        )
        assert submissions == 1
        stream.synchronize()
        expected_staging = torch.full((4, 16), 0xA5, dtype=torch.uint8)
        expected_staging[0].copy_(host_rows[3])
        expected_staging[2].copy_(host_rows[1])
        assert torch.equal(staging_rows.cpu(), expected_staging)
        assert torch.all(staging_backing[:, 1, :] == 0xA5)
        assert runtime.pending_count == 0
        progress = runtime.request_progress(0)
        assert progress.expected_work == 0
        assert not progress.complete
        assert progress.unavailable_bytes == 0
        assert progress.runnable_compute_ns == 0
        assert progress.completed_compute_ns == 0
        assert progress.pending_compute_ns == 0
        assert progress.expected_compute_ns == 0
        assert progress.remaining_compute_ns == 0
        assert runtime.work_runnable_ns(2) == (0, 0)
        progress_range = runtime.request_progress_range(0, 2)
        assert len(progress_range) == 2
        assert progress_range[0].request_id == 17
        assert progress_range[0].generation == 3
        snapshot = runtime.request_progress_snapshot()
        snapshot.capture(0, 2, stream)
        captured = snapshot.wait()
        assert not snapshot.pending
        assert snapshot.query() is None
        assert tuple((item.request_id, item.generation) for item in captured) == (
            (17, 3),
            (0, 0),
        )
        epoch = runtime.epoch_status(2)
        assert epoch.total == 2
        assert epoch.fresh == 2
        assert not epoch.succeeded
        assert not epoch.has_failure

    print("python_runtime=pass")


if __name__ == "__main__":
    main()
