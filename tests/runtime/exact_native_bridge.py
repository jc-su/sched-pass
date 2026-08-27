from nta_runtime.execution_topology import ExactWorkTopology, WorkDependencySpan
from nta_runtime.requests import RequestBinding
from nta_runtime.runtime import AcquireRequirement, DeviceWorkPlan


def main() -> None:
    binding = RequestBinding(0, 4, 2, 99)
    topology = ExactWorkTopology.from_schedule(
        epoch=7,
        bindings=(binding,),
        request_indices=(0,),
        logical_work=(7,),
        demand_units=(2,),
        unit_bytes=128,
        estimated_compute_ns=321,
    )
    captured = {}
    plan = DeviceWorkPlan.__new__(DeviceWorkPlan)
    plan._handle = None
    plan._upload_native = (
        lambda work, dependencies, requests, stream: captured.update(
            work=tuple(work),
            dependencies=tuple(dependencies),
            requests=tuple(requests),
            stream=stream,
        )
    )
    requirement = AcquireRequirement(0, 0, 11, 0, 2, 1, 128, 0)
    plan.upload_exact(
        topology,
        (WorkDependencySpan(0, 1, 0),),
        (requirement,),
    )
    native = captured["work"][0]
    assert native.request_index == 0
    assert native.request_slot == 4
    assert native.generation == 2
    assert native.logical_work == 7
    assert native.work_ticket == 0
    assert native.reduction_group == 0
    assert native.contributor_index == 0
    assert native.contributor_count == 1
    assert native.estimated_compute_ns == 321
    request = captured["requests"][0]
    assert request.work_begin == 0
    assert request.work_count == 1
    assert request.request_slot == 4
    assert request.generation == 2
    print("exact_native_bridge=pass")


if __name__ == "__main__":
    main()
