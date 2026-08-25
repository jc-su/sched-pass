from nta_runtime.requests import RequestBinding
from nta_runtime.runtime import AcquireRequirement, DeviceWorkPlan, RequestRange
from nta_runtime.work_unit import (
    DemandDescriptor,
    DemandSemantics,
    Granularity,
    WorkUnit,
)


def main() -> None:
    binding = RequestBinding(0, 4, 2, 99)
    unit = WorkUnit(
        work_id=3,
        binding=binding,
        layer=2,
        logical_begin=7,
        logical_count=1,
        demand=DemandDescriptor(
            candidate_units=4,
            selected_units=2,
            unit_bytes=128,
            granularity=Granularity.PAGE_GROUP,
            semantics=DemandSemantics.EXACT_SPARSE,
            provider="test.trace",
            epoch=7,
            selected_ids=(0, 2),
        ),
        estimated_compute_ns=321,
        reduction_group=5,
        contributor_index=1,
        contributor_count=2,
    )
    captured = {}
    plan = DeviceWorkPlan.__new__(DeviceWorkPlan)
    plan._handle = None
    plan.upload = lambda work, dependencies, requests, stream=None: captured.update(
        work=tuple(work),
        dependencies=tuple(dependencies),
        requests=tuple(requests),
        stream=stream,
    )
    requirement = AcquireRequirement(0, 0, 11, 0, 2, 1, 128, 0)
    plan.upload_work_units(
        (unit,),
        ((0, 1, 0, 3),),
        (requirement,),
        (RequestRange(0, 1, binding.request_slot, binding.generation),),
        epoch=7,
    )
    native = captured["work"][0]
    assert native.request_slot == 4
    assert native.generation == 2
    assert native.logical_work == 7
    assert native.work_ticket == 3
    assert native.reduction_group == 5
    assert native.contributor_index == 1
    assert native.estimated_compute_ns == 321
    print("semantic_native_bridge=pass")


if __name__ == "__main__":
    main()
