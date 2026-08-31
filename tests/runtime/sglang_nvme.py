from types import SimpleNamespace

from nta_runtime.engines.sglang_nvme_planning import (
    plan_nvme_batch_geometry,
    plan_nvme_window_layer_capacity,
)
from nta_runtime.nvme_granularity import (
    NvmeGranularity,
    NvmeTransferServiceModel,
)
from nta_runtime.requests import RequestBinding


PAIR = ((0, 1, 2, 3), (16, 17, 18, 19))


def semantic(*request_indices: int):
    return SimpleNamespace(
        dependency_kind="physical_pages",
        schedule=SimpleNamespace(request_indices=request_indices),
        page_pairs=(PAIR,) * len(request_indices),
    )


def geometry(
    *,
    isolated: bool,
    capacity: int = 32,
    work_ticket_capacity: int = 32,
    tenant_ids: tuple[int, int] = (4, 9),
):
    return plan_nvme_batch_geometry(
        semantic_plans={7: semantic(0, 1)},
        bindings=(
            RequestBinding(
                0, 3, 11, 101, deadline_clock=900, tenant_id=tenant_ids[0]
            ),
            RequestBinding(
                1, 8, 17, 202, deadline_clock=500, tenant_id=tenant_ids[1]
            ),
        ),
        row_bytes=(4096, 4096),
        lba_size=4096,
        max_transfer_bytes=8192,
        object_capacity=capacity,
        work_ticket_capacity=work_ticket_capacity,
        tenant_isolation=isolated,
    )


def test_request_owned_kv_never_fans_out_across_tenants() -> None:
    planned = geometry(isolated=False)
    assert tuple(scope.tenant_id for scope in planned.scopes) == (4, 9)
    assert planned.object_count == 8
    assert planned.work_item_count == 4
    assert planned.logical_transfer_bytes == 4 * 8192
    assert planned.scoped_exact_transfer_bytes == 8 * 8192
    assert planned.unique_source_transfer_bytes == 8 * 8192
    for scope in planned.scopes:
        expected_consumer = 0 if scope.tenant_id == 4 else 1
        assert all(
            group.consumer_indices == (expected_consumer,) for group in scope.groups
        )


def test_budget_isolation_flag_does_not_change_object_scope() -> None:
    unbudgeted = geometry(isolated=False)
    isolated = geometry(isolated=True)
    assert unbudgeted == isolated


def test_same_tenant_deduplication_is_preserved() -> None:
    planned = geometry(isolated=False, tenant_ids=(4, 4))
    assert tuple(scope.tenant_id for scope in planned.scopes) == (4,)
    assert planned.object_count == 4
    assert planned.work_item_count == 4
    assert planned.logical_transfer_bytes == 4 * 8192
    assert planned.scoped_exact_transfer_bytes == planned.logical_transfer_bytes
    assert planned.unique_source_transfer_bytes == planned.logical_transfer_bytes
    assert all(
        group.consumer_indices == (0, 1)
        for group in planned.scopes[0].groups
    )


def test_tenant_scoped_geometry_fails_before_partial_publication() -> None:
    try:
        geometry(isolated=False, capacity=7)
    except RuntimeError as error:
        assert "more concurrent acquisition objects" in str(error)
    else:
        raise AssertionError("an over-capacity tenant-scoped plan was accepted")


def test_fanout_ticket_capacity_fails_before_publication() -> None:
    try:
        geometry(isolated=False, work_ticket_capacity=3)
    except RuntimeError as error:
        assert "generation-bound work items" in str(error)
    else:
        raise AssertionError("NVMe fan-out exceeded the work-ticket directory")


def test_nonphysical_semantics_fail_closed() -> None:
    bad = semantic(0)
    bad.dependency_kind = "typed_lease"
    try:
        plan_nvme_batch_geometry(
            semantic_plans={1: bad},
            bindings=(RequestBinding(0, 0, 1, 7),),
            row_bytes=(4096, 4096),
            lba_size=4096,
            max_transfer_bytes=8192,
            object_capacity=8,
            work_ticket_capacity=8,
            tenant_isolation=False,
        )
    except RuntimeError as error:
        assert "physical-page semantics" in str(error)
    else:
        raise AssertionError("nonphysical NVMe semantics were accepted")


def test_measured_service_cost_selects_exact_span_compaction() -> None:
    sparse = (tuple(range(0, 32, 2)), tuple(range(16)))
    sparse_semantic = SimpleNamespace(
        dependency_kind="physical_pages",
        schedule=SimpleNamespace(request_indices=(0, 1)),
        page_pairs=(sparse, sparse),
    )
    model = NvmeTransferServiceModel(
        command_service_ns=20_000,
        read_bandwidth_bytes_per_second=6_000_000_000,
        compaction_bandwidth_bytes_per_second=600_000_000_000,
        compaction_launch_ns=10_000,
    )
    planned = plan_nvme_batch_geometry(
        semantic_plans={7: sparse_semantic},
        bindings=(
            RequestBinding(0, 3, 11, 101, deadline_clock=900, tenant_id=4),
            RequestBinding(1, 8, 17, 202, deadline_clock=500, tenant_id=4),
        ),
        row_bytes=(4096, 4096),
        lba_size=4096,
        max_transfer_bytes=2 * 1024 * 1024,
        object_capacity=64,
        work_ticket_capacity=64,
        tenant_isolation=False,
        service_model=model,
        scratch_capacity_bytes=1 << 20,
        scratch_alignment=4096,
    )
    assert planned.granularity is NvmeGranularity.SPAN_COMPACT
    assert planned.granularity_reason == "service_cost"
    assert planned.object_count == 2
    assert planned.work_item_count == 2
    assert planned.logical_transfer_bytes == 16 * 8192
    assert planned.scoped_exact_transfer_bytes == planned.logical_transfer_bytes
    assert planned.unique_source_transfer_bytes == planned.logical_transfer_bytes
    assert planned.physical_transfer_bytes == 31 * 8192
    assert planned.scratch_bytes_per_layer == planned.physical_transfer_bytes
    assert planned.selected_predicted_ns < planned.direct_predicted_ns


def test_uncalibrated_geometry_never_selects_span_optimistically() -> None:
    planned = geometry(isolated=False)
    assert planned.granularity is NvmeGranularity.DIRECT
    assert planned.granularity_reason == "uncalibrated"
    assert planned.scratch_bytes_per_layer == 0
    assert planned.physical_transfer_bytes >= planned.unique_source_transfer_bytes


def test_window_planner_reaches_refill_steady_state() -> None:
    assert (
        plan_nvme_window_layer_capacity(
            layer_count=36,
            objects_per_layer=32,
            capacity_layer_limit=36,
            queue_depth=64,
        )
        == 5
    )
    # Larger per-layer demand naturally needs fewer layers, while a small
    # runtime remains a hard bound independent of the queue target.
    assert (
        plan_nvme_window_layer_capacity(
            layer_count=36,
            objects_per_layer=96,
            capacity_layer_limit=36,
            queue_depth=64,
        )
        == 3
    )
    assert (
        plan_nvme_window_layer_capacity(
            layer_count=36,
            objects_per_layer=4,
            capacity_layer_limit=7,
            queue_depth=64,
        )
        == 7
    )
    assert (
        plan_nvme_window_layer_capacity(
            layer_count=36,
            objects_per_layer=32,
            capacity_layer_limit=36,
            queue_depth=64,
            explicit_limit=4,
        )
        == 4
    )


def main() -> None:
    test_request_owned_kv_never_fans_out_across_tenants()
    test_budget_isolation_flag_does_not_change_object_scope()
    test_same_tenant_deduplication_is_preserved()
    test_tenant_scoped_geometry_fails_before_partial_publication()
    test_fanout_ticket_capacity_fails_before_publication()
    test_nonphysical_semantics_fail_closed()
    test_measured_service_cost_selects_exact_span_compaction()
    test_uncalibrated_geometry_never_selects_span_optimistically()
    test_window_planner_reaches_refill_steady_state()
    print("sglang_nvme=pass")


if __name__ == "__main__":
    main()
