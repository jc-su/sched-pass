from nta_runtime.execution_core import ExecutionSession, ExecutionTile
from nta_runtime.execution_protocol import ExecutionProtocolConfig
from nta_runtime.requests import RequestBinding
from nta_runtime.work_unit import Availability, Granularity


def main() -> None:
    bindings = (
        RequestBinding(0, 2, 1, 10),
        RequestBinding(1, 3, 4, 11),
    )
    session = ExecutionSession.from_tiles(
        epoch=5,
        granularity=Granularity.PAGE_GROUP,
        protocol=ExecutionProtocolConfig.late_bound(
            granularity=Granularity.PAGE_GROUP,
            max_inflight_units=1,
        ),
        tiles=(
            ExecutionTile(0, bindings[0], 0, 0, 2, (0, 1), 128, True, 100, 0),
            # Multiple native contributors may share a logical KV coordinate;
            # the schedule work ticket remains their unique semantic identity.
            ExecutionTile(1, bindings[1], 0, 0, 4, (1, 3), 128, False, 100, 1),
        ),
    )
    assert session.runnable_groups() == ((0,),)
    session.launch_group((0,))
    session.complete_group((0,))
    assert session.ledger.state(0) is Availability.COMPLETE
    session.make_ready((1,))
    assert session.runnable_groups() == ((1,),)
    assert (
        session.unit_for_ticket(
            work_id=1, layer=0, logical_begin=0, request_index=1
        ).work_id
        == 1
    )
    try:
        session.unit_for_ticket(work_id=1, layer=0, logical_begin=1, request_index=1)
    except RuntimeError as error:
        assert "coordinates diverged" in str(error)
    else:
        raise AssertionError("stale native coordinates were accepted")
    stats = session.expose_stats()
    assert stats["work_complete"] == 1
    assert stats["work_is_heterogeneous"]

    bounded = ExecutionSession.from_tiles(
        epoch=5,
        granularity=Granularity.PAGE_GROUP,
        protocol=ExecutionProtocolConfig.late_bound(
            granularity=Granularity.PAGE_GROUP,
            max_inflight_units=1,
        ),
        tiles=(
            ExecutionTile(0, bindings[0], 0, 0, 1, (0,), 128, True, 100, 0),
            ExecutionTile(1, bindings[1], 0, 1, 1, (0,), 128, True, 100, 1),
        ),
    )
    try:
        bounded.launch_group((0, 1))
    except RuntimeError as error:
        assert "in-flight capacity" in str(error)
    else:
        raise AssertionError("launch group exceeded its in-flight capacity")
    assert bounded.ready_work == (0, 1)
    try:
        bounded.launch_group((0, 0))
    except RuntimeError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate launch work unit was accepted")
    assert bounded.ready_work == (0, 1)
    print("execution_core=pass")


if __name__ == "__main__":
    main()
