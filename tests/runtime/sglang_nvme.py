from types import SimpleNamespace

from nta_runtime.engines.sglang_nvme import (
    plan_nvme_batch_geometry,
    plan_nvme_window_layer_capacity,
)
from nta_runtime.requests import RequestBinding


PAIR = ((0, 1, 2, 3), (16, 17, 18, 19))


def semantic(*request_indices: int):
    return SimpleNamespace(
        dependency_kind="physical_pages",
        schedule=SimpleNamespace(request_indices=request_indices),
        page_pairs=(PAIR,) * len(request_indices),
    )


def geometry(*, isolated: bool, capacity: int = 32, work_ticket_capacity: int = 32):
    return plan_nvme_batch_geometry(
        semantic_plans={7: semantic(0, 1)},
        bindings=(
            RequestBinding(0, 3, 11, 101, deadline_clock=900, tenant_id=4),
            RequestBinding(1, 8, 17, 202, deadline_clock=500, tenant_id=9),
        ),
        row_bytes=(4096, 4096),
        lba_size=4096,
        max_transfer_bytes=8192,
        object_capacity=capacity,
        work_ticket_capacity=work_ticket_capacity,
        tenant_isolation=isolated,
    )


def test_batch_deduplicates_shared_runs_without_isolation() -> None:
    planned = geometry(isolated=False)
    assert len(planned.scopes) == 1
    assert planned.object_count == 4
    # Two shared transfer groups fan out to both request generations.  DMA is
    # deduplicated, but cancellation-safe readiness remains per consumer.
    assert planned.work_item_count == 4
    assert planned.logical_transfer_bytes == 4 * 8192


def test_tenant_isolation_scopes_shared_physical_bytes() -> None:
    planned = geometry(isolated=True)
    assert tuple(scope.tenant_id for scope in planned.scopes) == (4, 9)
    assert planned.object_count == 8
    assert planned.work_item_count == 4
    assert planned.logical_transfer_bytes == 4 * 8192


def test_isolated_geometry_fails_before_partial_publication() -> None:
    try:
        geometry(isolated=True, capacity=7)
    except RuntimeError as error:
        assert "more concurrent acquisition objects" in str(error)
    else:
        raise AssertionError("an over-capacity isolated plan was accepted")


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
    test_batch_deduplicates_shared_runs_without_isolation()
    test_tenant_isolation_scopes_shared_physical_bytes()
    test_isolated_geometry_fails_before_partial_publication()
    test_fanout_ticket_capacity_fails_before_publication()
    test_nonphysical_semantics_fail_closed()
    test_window_planner_reaches_refill_steady_state()
    print("sglang_nvme=pass")


if __name__ == "__main__":
    main()
