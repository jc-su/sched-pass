#!/usr/bin/env python3
"""Exercise the owning Python engine boundary against real CUDA allocations."""

from __future__ import annotations

import torch

from nta_runtime import (
    AcquireRequirement,
    DeviceWorkPlan,
    IndexedHostObject,
    Placement,
    Replica,
    RequestRange,
    Runtime,
    RuntimeConfig,
    WorkItem,
    build_selected_page_work_plan,
    device_abi_version,
    register_selected_host_pages,
)


def main() -> None:
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
            assert plan.work_items_address != 0
            assert plan.dependencies_address != 0
            assert plan.work_item_count == 1
            assert plan.dependency_count == 1
            assert plan.device_ordinal == runtime.device_ordinal
            assert plan.work_items_tensor.data_ptr() == plan.work_items_address
            assert plan.dependencies_tensor.data_ptr() == plan.dependencies_address
            assert not plan.has_external
        assert runtime.device_view_tensor.data_ptr() == runtime.device_view
        host_rows = torch.arange(64, dtype=torch.uint8, pin_memory=True).view(4, 16)
        staging_rows = torch.zeros_like(host_rows, device="cuda")
        source_indices = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
        staging_indices = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
        runtime.register_indexed_host_objects(
            1,
            [
                IndexedHostObject(
                    102,
                    8,
                    host_rows.data_ptr(),
                    staging_rows.data_ptr(),
                    source_indices.data_ptr(),
                    staging_indices.data_ptr(),
                    2,
                    16,
                    16,
                    16,
                    4,
                    4,
                )
            ],
            stream=stream,
        )
        stream.synchronize()
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

    selected_source = torch.arange(8 * 16, dtype=torch.uint8, pin_memory=True).view(
        8, 16
    )
    selected_staging = torch.zeros((4, 16), dtype=torch.uint8, device="cuda")
    selected_indices = torch.tensor([[7, 1], [4, 2]], dtype=torch.int32, device="cuda")
    with Runtime(
        RuntimeConfig(
            request_capacity=2,
            object_capacity=2,
            intent_capacity=4,
            work_ticket_capacity=4,
            max_dependencies_per_work_ticket=1,
        )
    ) as runtime:
        for request in range(2):
            runtime.set_request(request, 100 + request, 1)
        acquisition = register_selected_host_pages(
            runtime,
            selected_source,
            selected_staging,
            selected_indices,
            stream=stream,
        )
        with build_selected_page_work_plan(
            runtime, acquisition, (0, 0, 1, 1), stream=stream
        ) as plan:
            plan.wait_on(torch.cuda.current_stream())
            torch.cuda.synchronize()
            assert acquisition.request_count == 2
            assert acquisition.pages_per_request == 2
            assert plan.work_item_count == 4
            assert plan.dependency_count == 4
            assert plan.has_external
    print("python_runtime=pass")


if __name__ == "__main__":
    main()
