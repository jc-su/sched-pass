from nta_runtime.engines.sglang_hicache import (
    LeaseOperationRange,
    LeaseWorkDependency,
    lease_indexed_transfer_topology,
)


def main() -> None:
    topology = lease_indexed_transfer_topology(
        (
            LeaseWorkDependency(7, 2, 3),
            LeaseWorkDependency(7, 5, 2),
            None,
            LeaseWorkDependency(9, 1, 2),
        ),
        (
            LeaseWorkDependency(7, 0, 7),
            LeaseWorkDependency(7, 0, 7),
            None,
            LeaseWorkDependency(9, 0, 4),
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
    print("sglang_lease_topology=pass")


if __name__ == "__main__":
    main()
