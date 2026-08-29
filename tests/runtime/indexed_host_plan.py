from nta_runtime.indexed_transfer import (
    AcquisitionGroup,
    AcquisitionSlice,
    AcquisitionTopology,
    IndexedTensorLane,
)
from nta_runtime.runtime import IndexedAcquisitionPlan
from nta_runtime.requests import RequestBinding


def main() -> None:
    topology = AcquisitionTopology(
        6,
        (
            AcquisitionGroup(0, 4),
            AcquisitionGroup(4, 2),
        ),
        (
            (AcquisitionSlice(0, 1, 2),),
            (
                AcquisitionSlice(0, 0, 4),
                AcquisitionSlice(1, 0, 2),
            ),
            (),
        ),
    )
    lanes = (
        IndexedTensorLane(0x1000, 0x2000, 16, 16, 16, 32, 32),
        IndexedTensorLane(0x3000, 0x4000, 32, 32, 32, 32, 32),
    )
    request_zero = RequestBinding(0, 3, 1, 100)
    request_one = RequestBinding(1, 4, 1, 101)
    plan = IndexedAcquisitionPlan(
        topology,
        lanes,
        work_bindings=(request_zero, request_zero, request_one),
        source_indices_device_address=0x5000,
        staging_indices_device_address=0x6000,
        object_version=3,
        direct_base=0x7000,
        first_slot=5,
        object_id_base=100,
    )
    assert plan.object_count == 4
    assert plan.group_consumers == ((request_zero,), (request_zero,))
    assert plan.transfer_bytes == 6 * (16 + 32)
    assert plan.object_transfer_bytes == (4 * 16, 4 * 32, 2 * 16, 2 * 32)
    assert sum(plan.object_transfer_bytes) == plan.transfer_bytes
    assert plan.max_object_fanout == 2
    assert plan.direct_work_count == 1
    assert plan.external_object_slots == ((5, 6), (5, 6, 7, 8), ())
    assert tuple(
        (span.begin, span.count, span.direct_count)
        for span in plan.dependency_spans
    ) == ((0, 2, 0), (2, 4, 0), (6, 2, 2))
    assert plan.native_objects[0].source_indices_device_address == 0x5000
    assert plan.native_objects[2].source_indices_device_address == 0x5010
    # Work 0 consumes only rows [1, 3) of a four-row group, but the shared
    # acquisition owner issues and publishes the complete group exactly once.
    assert plan.dependencies[0].bytes == 4 * 16
    assert plan.dependencies[1].bytes == 4 * 32
    assert plan.dependencies[6].direct_base == 0x7000
    assert plan.dependencies[7].direct_base == 0x7000
    shared = IndexedAcquisitionPlan(
        topology,
        lanes,
        work_bindings=(request_zero, request_one, request_one),
        source_indices_device_address=0x5000,
        staging_indices_device_address=0x6000,
        object_version=3,
        direct_base=0x7000,
    )
    assert shared.group_consumers == (
        (request_zero, request_one),
        (request_one,),
    )
    assert shared.group_tenant_ids == ((0,), (0,))
    shared.require_single_tenant_groups()
    tenant_one = RequestBinding(1, 4, 1, 101, tenant_id=1)
    cross_tenant = IndexedAcquisitionPlan(
        topology,
        lanes,
        work_bindings=(request_zero, tenant_one, tenant_one),
        source_indices_device_address=0x5000,
        staging_indices_device_address=0x6000,
        object_version=3,
        direct_base=0x7000,
    )
    assert cross_tenant.group_tenant_ids == ((0, 1), (1,))
    try:
        cross_tenant.require_single_tenant_groups()
    except ValueError as error:
        assert "cross tenant credit domains" in str(error)
    else:
        raise AssertionError("cross-tenant acquisition group bypassed isolation")
    print("indexed_acquisition_plan=pass")


if __name__ == "__main__":
    main()
