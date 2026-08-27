from nta_runtime.execution_protocol import (
    ExecutionProtocolConfig,
    WorkLedger,
)
from nta_runtime.requests import RequestBinding
from nta_runtime.work_unit import (
    Availability,
    DemandDescriptor,
    DemandSemantics,
    Granularity,
    WorkBatch,
    WorkUnit,
)


def binding(index: int, slot: int, generation: int) -> RequestBinding:
    return RequestBinding(index, slot, generation, 1000 + index)


def demand(*, selected: int, candidate: int, epoch: int = 7) -> DemandDescriptor:
    return DemandDescriptor(
        candidate_units=candidate,
        selected_units=selected,
        unit_bytes=128,
        granularity=Granularity.PAGE_GROUP,
        semantics=DemandSemantics.EXACT_SPARSE,
        provider="fixture-exact-mask",
        epoch=epoch,
        selected_ids=tuple(range(selected)),
    )


def test_exact_demand_and_heterogeneous_batch() -> None:
    units = (
        WorkUnit(0, binding(0, 4, 2), 0, 0, 1, demand(selected=2, candidate=8)),
        WorkUnit(1, binding(1, 9, 4), 0, 0, 1, demand(selected=8, candidate=8)),
    )
    batch = WorkBatch(7, Granularity.PAGE_GROUP, units)
    assert batch.is_heterogeneous
    assert batch.request_identities == ((4, 2), (9, 4))
    assert units[0].demand.selected_bytes == 256
    assert units[0].demand.is_exact


def test_generation_checked_partial_protocol() -> None:
    unit = WorkUnit(
        0,
        binding(0, 4, 2),
        0,
        0,
        1,
        demand(selected=2, candidate=8),
    )
    batch = WorkBatch(7, Granularity.PAGE_GROUP, (unit,))
    ledger = WorkLedger(
        batch,
        ExecutionProtocolConfig.partial(
            granularity=Granularity.PAGE_GROUP,
            max_inflight_units=1,
        ),
    )
    ledger.discover(0, ready=False, binding=unit.binding, epoch=7)
    ledger.transition(0, Availability.READY, binding=unit.binding, epoch=7)
    ledger.transition(0, Availability.RUNNING, binding=unit.binding, epoch=7)
    ledger.transition(0, Availability.PARTIAL, binding=unit.binding, epoch=7)
    assert ledger.runnable_groups() == ()
    ledger.transition(0, Availability.READY, binding=unit.binding, epoch=7)
    assert ledger.runnable_groups() == ((0,),)
    try:
        ledger.transition(
            0,
            Availability.READY,
            binding=binding(0, 4, 3),
            epoch=7,
        )
    except ValueError as error:
        assert "stale request generation" in str(error)
    else:
        raise AssertionError("stale generation advanced a work unit")


def test_conventional_protocol_rejects_partial_execution() -> None:
    unit = WorkUnit(
        0,
        binding(0, 4, 2),
        0,
        0,
        1,
        demand(selected=2, candidate=8),
    )
    batch = WorkBatch(7, Granularity.PAGE_GROUP, (unit,))
    config = ExecutionProtocolConfig.conventional(
        granularity=Granularity.PAGE_GROUP,
        max_inflight_units=1,
    )
    ledger = WorkLedger(batch, config)
    ledger.discover(0, ready=True, binding=unit.binding, epoch=7)
    ledger.transition(0, Availability.RUNNING, binding=unit.binding, epoch=7)
    try:
        ledger.transition(0, Availability.PARTIAL, binding=unit.binding, epoch=7)
    except ValueError as error:
        assert "does not support partial" in str(error)
    else:
        raise AssertionError("conventional protocol accepted partial work")


def test_conventional_protocol_has_a_readiness_boundary() -> None:
    units = tuple(
        WorkUnit(
            index,
            binding(index, index, 1),
            0,
            index,
            1,
            demand(selected=1, candidate=2),
        )
        for index in range(3)
    )
    batch = WorkBatch(7, Granularity.PAGE_GROUP, units)
    ledger = WorkLedger(
        batch,
        ExecutionProtocolConfig.conventional(
            granularity=Granularity.PAGE_GROUP,
            max_inflight_units=2,
        ),
    )
    for unit in units[:2]:
        ledger.discover(unit.work_id, ready=True, binding=unit.binding, epoch=7)
    assert ledger.runnable_groups() == ()
    ledger.discover(2, ready=True, binding=units[2].binding, epoch=7)
    assert ledger.runnable_groups() == ((0, 1), (2,))


def test_inflight_bound_counts_running_units() -> None:
    units = tuple(
        WorkUnit(
            index,
            binding(index, index, 1),
            0,
            index,
            1,
            demand(selected=1, candidate=2),
        )
        for index in range(2)
    )
    batch = WorkBatch(7, Granularity.PAGE_GROUP, units)
    ledger = WorkLedger(
        batch,
        ExecutionProtocolConfig.late_bound(
            granularity=Granularity.PAGE_GROUP,
            max_inflight_units=1,
        ),
    )
    for unit in units:
        ledger.discover(unit.work_id, ready=True, binding=unit.binding, epoch=7)
    ledger.transition(0, Availability.RUNNING, binding=units[0].binding, epoch=7)
    assert ledger.runnable_groups() == ()
    try:
        ledger.transition(1, Availability.RUNNING, binding=units[1].binding, epoch=7)
    except ValueError as error:
        assert "in-flight capacity" in str(error)
    else:
        raise AssertionError("ledger exceeded its in-flight capacity")
    assert ledger.state(1) is Availability.READY


def test_protocol_configuration_is_framework_neutral() -> None:
    config = ExecutionProtocolConfig.from_environment(
        {
            "NTA_EXECUTION_PROTOCOL": "late_bound",
            "NTA_EXECUTION_GRANULARITY": "page_group",
            "NTA_EXECUTION_MAX_INFLIGHT_UNITS": "8",
        }
    )
    assert config.kind.value == "late_bound"
    assert config.max_inflight_units == 8


def main() -> None:
    test_exact_demand_and_heterogeneous_batch()
    test_generation_checked_partial_protocol()
    test_conventional_protocol_rejects_partial_execution()
    test_conventional_protocol_has_a_readiness_boundary()
    test_inflight_bound_counts_running_units()
    test_protocol_configuration_is_framework_neutral()
    print("work_unit=pass")


if __name__ == "__main__":
    main()
