from nta_runtime.indexed_transfer import (
    IndexedTensorLane,
    IndexedTransferGroup,
    IndexedTransferTopology,
    IndexedWorkDependency,
)
from nta_runtime.runtime import IndexedHostPlan


def main() -> None:
    topology = IndexedTransferTopology(
        6,
        (
            IndexedTransferGroup(0, 4),
            IndexedTransferGroup(4, 2),
        ),
        (
            (IndexedWorkDependency(0, 1, 2),),
            (
                IndexedWorkDependency(0, 0, 4),
                IndexedWorkDependency(1, 0, 2),
            ),
            (),
        ),
    )
    lanes = (
        IndexedTensorLane(0x1000, 0x2000, 16, 16, 16, 32, 32),
        IndexedTensorLane(0x3000, 0x4000, 32, 32, 32, 32, 32),
    )
    plan = IndexedHostPlan(
        topology,
        lanes,
        source_indices_device_address=0x5000,
        staging_indices_device_address=0x6000,
        object_version=3,
        direct_base=0x7000,
        first_slot=5,
        object_id_base=100,
    )
    assert plan.object_count == 4
    assert plan.transfer_bytes == 6 * (16 + 32)
    assert plan.max_object_fanout == 2
    assert plan.direct_work_count == 1
    assert not plan.exact_resume_windows
    assert plan.external_object_slots == ((5, 6), (5, 6, 7, 8), ())
    assert tuple(
        (span.begin, span.count, span.direct_count)
        for span in plan.dependency_spans
    ) == ((0, 2, 0), (2, 4, 0), (6, 2, 2))
    assert plan.native_objects[0].source_indices_device_address == 0x5000
    assert plan.native_objects[2].source_indices_device_address == 0x5010
    assert plan.dependencies[0].bytes == 2 * 16
    assert plan.dependencies[1].bytes == 2 * 32
    assert plan.dependencies[6].direct_base == 0x7000
    assert plan.dependencies[7].direct_base == 0x7000
    print("indexed_host_plan=pass")


if __name__ == "__main__":
    main()
