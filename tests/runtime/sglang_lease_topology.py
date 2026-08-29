import time

from nta_runtime.adapters.sglang import SglangAcquisitionSpan
from nta_runtime.engines.sglang_contracts import (
    LeaseAcquisitionGroup,
    LeaseAcquisitionSlice,
    LeaseOperationRange,
    LeaseOperationTransfer,
)
from nta_runtime.engines.sglang_topology import (
    capacity_constrained_acquisition_groups,
    group_external_pages_by_request,
    lease_acquisition_topology,
    page_pairs_for_schedule,
    project_acquisition_slices,
    request_batch_heterogeneity,
    resolve_request_acquisitions,
)
from nta_runtime.flashinfer_schedule import Schedule
from nta_runtime.requests import RequestBinding, stable_request_id


def main() -> None:
    topology = lease_acquisition_topology(
        (
            LeaseAcquisitionSlice(7, 2, 3),
            LeaseAcquisitionSlice(7, 5, 2),
            None,
            LeaseAcquisitionSlice(9, 1, 2),
        ),
        (
            LeaseAcquisitionGroup(7, 0, 7),
            LeaseAcquisitionGroup(7, 0, 7),
            None,
            LeaseAcquisitionGroup(9, 0, 4),
        ),
        (
            LeaseOperationRange(7, 0, 7),
            LeaseOperationRange(9, 7, 4),
        ),
        index_count=11,
    )
    assert tuple((group.index_offset, group.row_count) for group in topology.groups) == (
        (0, 7),
        (7, 4),
    )
    assert topology.direct_work_count == 1
    assert topology.max_group_fanout == 2
    assert tuple(
        tuple((item.group_index, item.row_offset, item.row_count) for item in work)
        for work in topology.dependencies_by_work
    ) == (((0, 2, 3),), ((0, 5, 2),), (), ((1, 1, 2),))

    acquisitions = (
        SglangAcquisitionSpan(71, 41, 8, 32),
        SglangAcquisitionSpan.direct(),
        SglangAcquisitionSpan(72, 52, 0, 8),
    )
    transfers = {
        71: LeaseOperationTransfer(71, 41, 32),
        72: LeaseOperationTransfer(72, 52, 8),
    }
    assert (
        resolve_request_acquisitions(
            acquisitions, transfers, lease_transfer_rows=40
        )
        == acquisitions
    )
    try:
        resolve_request_acquisitions(
            (acquisitions[0], acquisitions[0], acquisitions[2]),
            transfers,
            lease_transfer_rows=40,
        )
    except RuntimeError as error:
        assert "multiple requests" in str(error)
    else:
        raise AssertionError("two requests shared one acquisition operation")

    heterogeneous_bindings = (
        RequestBinding(0, 10, 1, stable_request_id("heterogeneous-0")),
        RequestBinding(
            1,
            11,
            1,
            stable_request_id("heterogeneous-1"),
            priority=2,
            tenant_id=7,
        ),
        RequestBinding(2, 12, 1, stable_request_id("heterogeneous-2")),
    )
    assert request_batch_heterogeneity(
        heterogeneous_bindings, (40, 8, 16), acquisitions
    ) == (
        "sequence_length",
        "availability",
        "external_rows",
        "tenant",
        "priority",
    )
    assert request_batch_heterogeneity(
        (heterogeneous_bindings[0],), (40,), (acquisitions[0],)
    ) == ()

    projected = project_acquisition_slices(
        Schedule(
            (0, 0, 0, 0, 0, 0, 1, 2),
            (0, 0, 1, 1, 2, 2, 0, 0),
            16,
            8,
        ),
        acquisitions,
        (40, 8, 8),
    )
    assert projected == (
        LeaseAcquisitionSlice(71, 0, 8),
        LeaseAcquisitionSlice(71, 0, 8),
        LeaseAcquisitionSlice(71, 8, 16),
        LeaseAcquisitionSlice(71, 8, 16),
        LeaseAcquisitionSlice(71, 24, 8),
        LeaseAcquisitionSlice(71, 24, 8),
        None,
        LeaseAcquisitionSlice(72, 0, 8),
    )
    assert capacity_constrained_acquisition_groups(
        projected, maximum_groups=2
    ) == (
        LeaseAcquisitionGroup(71, 0, 32),
        LeaseAcquisitionGroup(71, 0, 32),
        LeaseAcquisitionGroup(71, 0, 32),
        LeaseAcquisitionGroup(71, 0, 32),
        LeaseAcquisitionGroup(71, 0, 32),
        LeaseAcquisitionGroup(71, 0, 32),
        None,
        LeaseAcquisitionGroup(72, 0, 8),
    )

    scale_dependencies = tuple(
        LeaseAcquisitionSlice(99, index, 1) for index in range(8192)
    )
    scale_started = time.perf_counter()
    scale_groups = capacity_constrained_acquisition_groups(
        scale_dependencies, maximum_groups=4096
    )
    scale_seconds = time.perf_counter() - scale_started
    assert len(set(scale_groups)) == 4096
    assert scale_seconds < 2.0, (
        "acquisition grouping regressed from its O(W log W) control bound: "
        f"{scale_seconds:.3f}s"
    )

    schedule = Schedule((0, 0, 1, 1), (0, 1, 0, 1), 4, 128)
    grouped = group_external_pages_by_request(
        schedule,
        (
            ((10, 11), (20, 21)),
            ((11, 12), (21, 22)),
            ((), ()),
            ((30,), (40,)),
        ),
    )
    assert grouped == (
        ((10, 11, 12), (20, 21, 22)),
        ((10, 11, 12), (20, 21, 22)),
        ((), ()),
        ((30,), (40,)),
    )
    chunked_pairs = page_pairs_for_schedule(
        Schedule((0, 0, 1), (0, 1, 0), 4, 3),
        indptr=(0, 8, 12),
        pages=(10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23),
        last_page=(1, 1),
        page_size=1,
        source_by_device={11: 101, 12: 102, 16: 106, 20: 200, 21: 201},
    )
    assert chunked_pairs == (
        ((101, 102), (11, 12)),
        ((106,), (16,)),
        ((200, 201), (20, 21)),
    )
    for invalid_schedule, invalid_indptr, invalid_last_page, message in (
        (Schedule((0,), (2,), 4, 3), (0, 2), (1,), "out-of-range"),
        (Schedule((0,), (0,), 4, 3), (1, 2), (1,), "page table"),
        (Schedule((0,), (0,), 4, 3), (0, 2), (0,), "last-page"),
    ):
        try:
            page_pairs_for_schedule(
                invalid_schedule,
                indptr=invalid_indptr,
                pages=(10, 11),
                last_page=invalid_last_page,
                page_size=1,
                source_by_device={},
            )
        except RuntimeError as error:
            assert message in str(error)
        else:
            raise AssertionError("invalid FlashInfer page geometry was accepted")
    try:
        group_external_pages_by_request(
            schedule,
            (
                ((10,), (20,)),
                ((11,), (20,)),
                ((), ()),
                ((30,), (40,)),
            ),
        )
    except RuntimeError as error:
        assert "multiple host cache pages" in str(error)
    else:
        raise AssertionError("request grouping accepted an inconsistent page map")
    print("sglang_lease_topology=pass")


if __name__ == "__main__":
    main()
