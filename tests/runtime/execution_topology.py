from nta_runtime.execution_topology import (
    ExactWorkTopology,
    WorkDependencySpan,
)
from nta_runtime.requests import RequestBinding
from nta_runtime.runtime import (
    INVALID_INDEX,
    AcquireRequirement,
    DeviceWorkPlan,
    WorkItemFlag,
)


def main() -> None:
    bindings = (
        RequestBinding(0, 7, 2, 101),
        RequestBinding(1, 11, 5, 202),
    )
    topology = ExactWorkTopology.from_schedule(
        epoch=9,
        bindings=bindings,
        request_indices=(0, 0, 1),
        logical_work=(3, 8, 13),
        demand_units=(4, 2, 7),
        unit_bytes=128,
        estimated_compute_ns=(100, 200, 300),
    )
    assert topology.work_count == 3
    assert topology.request_count == 2
    assert topology.selected_bytes == 13 * 128

    captured = {}
    plan = DeviceWorkPlan.__new__(DeviceWorkPlan)
    plan._handle = None
    plan._upload_native = lambda work, dependencies, requests, stream=None: captured.update(
        work=tuple(work),
        dependencies=tuple(dependencies),
        requests=tuple(requests),
        stream=stream,
    )
    dependencies = tuple(
        AcquireRequirement(0, 0, 100 + index, 0, index, 1, 128, 0)
        for index in range(3)
    )
    plan.upload_exact(
        topology,
        (
            WorkDependencySpan(0, 1, 0),
            WorkDependencySpan(1, 1, 0),
            WorkDependencySpan(2, 1, 0),
        ),
        dependencies,
    )
    work = captured["work"]
    assert tuple(item.work_ticket for item in work) == (0, 1, 2)
    assert tuple(item.request_index for item in work) == (0, 0, 1)
    assert tuple(item.request_slot for item in work) == (7, 7, 11)
    assert tuple(item.generation for item in work) == (2, 2, 5)
    assert tuple(item.logical_work for item in work) == (3, 8, 13)
    assert tuple(item.contributor_index for item in work) == (0, 1, 0)
    assert tuple(item.contributor_count for item in work) == (2, 2, 1)
    assert tuple(item.estimated_compute_ns for item in work) == (100, 200, 300)

    plan.upload_exact(
        topology,
        (
            WorkDependencySpan(0, 1, 0),
            WorkDependencySpan(1, 1, 0),
            WorkDependencySpan(2, 1, 0),
        ),
        dependencies,
        work_ticket_base=17,
        deadline_relative_to_discovery=True,
    )
    shared_work = captured["work"]
    assert tuple(item.work_ticket for item in shared_work) == (17, 18, 19)
    assert tuple(item.reduction_group for item in shared_work) == (17, 17, 18)
    assert all(
        item.flags == int(WorkItemFlag.DEADLINE_RELATIVE_TO_DISCOVERY)
        for item in shared_work
    )

    direct_dependencies = tuple(
        AcquireRequirement(1, 0, 0, 0, 0, 0, 1, 0) for _ in range(3)
    )
    direct_spans = tuple(WorkDependencySpan(index, 1, 1) for index in range(3))
    plan.upload_exact(
        topology,
        direct_spans,
        direct_dependencies,
        completion_classes=(INVALID_INDEX, 1, 0),
    )
    event_work = captured["work"]
    assert tuple(item.completion_class for item in event_work) == (
        INVALID_INDEX,
        1,
        0,
    )
    assert all(
        item.flags == int(WorkItemFlag.EVENT_PARTITION) for item in event_work
    )

    try:
        plan.upload_exact(
            topology,
            (
                WorkDependencySpan(0, 1, 0),
                WorkDependencySpan(1, 1, 0),
                WorkDependencySpan(2, 1, 0),
            ),
            dependencies,
            completion_classes=(INVALID_INDEX, 1, 0),
        )
    except ValueError as error:
        assert "must be preacquired" in str(error)
    else:  # pragma: no cover - fail-closed contract
        raise AssertionError("event partition retained physical dependencies")

    try:
        ExactWorkTopology.from_schedule(
            epoch=1,
            bindings=bindings,
            request_indices=(0, 1, 0),
            logical_work=(0, 1, 2),
            demand_units=(1, 1, 1),
            unit_bytes=1,
            estimated_compute_ns=1,
        )
    except ValueError as error:
        assert "contiguous" in str(error)
    else:  # pragma: no cover - fail-closed contract
        raise AssertionError("interleaved request work was accepted")
    print("execution_topology=pass")


if __name__ == "__main__":
    main()
