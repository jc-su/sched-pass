#!/usr/bin/env python3
"""Validate causal request-level acquisition/compute policy."""

from __future__ import annotations

from dataclasses import dataclass

from nta_runtime.critical_work import (
    RequestWork,
    ServiceModel,
    estimate_critical_work,
    plan_critical_work,
)
from nta_runtime.critical_work import _urgency


@dataclass(frozen=True)
class Progress:
    request_id: int
    generation: int
    expected_work: int
    pending_work: int
    runnable_work: int
    completed_work: int
    failed_work: int
    cancelled_work: int
    unavailable_bytes: int
    runnable_compute_ns: int
    completed_compute_ns: int
    pending_compute_ns: int
    expected_compute_ns: int
    dropped_attributions: int = 0


def work(
    request_id: int,
    *,
    pending: int = 0,
    runnable: int = 0,
    completed: int = 0,
    pending_ns: int = 0,
    runnable_ns: int = 0,
    completed_ns: int = 0,
    unavailable_bytes: int = 0,
    priority: int = 0,
    deadline_ns: int = 0,
    generation: int = 1,
    virtual_time_ns: int = 0,
) -> RequestWork:
    progress = Progress(
        request_id,
        generation,
        pending + runnable + completed,
        pending,
        runnable,
        completed,
        0,
        0,
        unavailable_bytes,
        runnable_ns,
        completed_ns,
        pending_ns,
        pending_ns + runnable_ns + completed_ns,
    )
    return RequestWork.from_progress(
        progress,
        priority=priority,
        deadline_ns=deadline_ns,
        tenant_virtual_time_ns=virtual_time_ns,
    )


def main() -> None:
    assert _urgency(0, 0, None) == 0
    assert _urgency(0, 100_000, None) == 3
    assert _urgency(0, 500_000, None) == 2
    assert _urgency(0, 2_000_000, None) == 1
    assert _urgency(4, 10_000_000, None) == 4
    assert _urgency(0, 200, -1) == 7
    assert _urgency(0, 200, 0) == 7
    assert _urgency(0, 200, 200) == 6
    assert _urgency(0, 200, 600) == 5
    assert _urgency(0, 200, 4_999_800) == 4
    assert _urgency(0, 200, 5_000_000) == 0
    model = ServiceModel(
        bandwidth_bytes_per_second=10_000_000_000,
        fixed_latency_ns=20_000,
        queue_delay_ns=10_000,
        reduction_ns=2_000,
    )
    urgent = work(
        11,
        pending=2,
        runnable=1,
        pending_ns=24_000,
        runnable_ns=8_000,
        unavailable_bytes=1_000_000,
        deadline_ns=1_180_000,
    )
    resident = work(12, runnable=4, runnable_ns=32_000, priority=5)
    background = work(
        13,
        pending=1,
        pending_ns=4_000,
        unavailable_bytes=64 * 1024,
        virtual_time_ns=50_000,
    )
    terminal = work(14, completed=1, completed_ns=4_000, priority=7)

    estimate = estimate_critical_work(urgent, now_ns=1_000_000, model=model)
    assert estimate.acquisition_ns == 130_000
    assert estimate.critical_path_ns == 156_000
    assert estimate.slack_ns == 24_000
    assert estimate.urgency == 6

    plan = plan_critical_work(
        (background, resident, terminal, urgent), now_ns=1_000_000, model=model
    )
    assert plan.data_order == ((11, 1), (13, 1))
    assert plan.compute_order == ((11, 1), (12, 1))
    assert plan.requests[-1].request.request_id == 14

    # A completion changes the online frontier without a future-arrival trace.
    arrived = work(11, runnable=3, runnable_ns=32_000, deadline_ns=1_180_000)
    arrived_plan = plan_critical_work(
        (arrived, background), now_ns=1_020_000, model=model
    )
    assert arrived_plan.data_order == ((13, 1),)
    assert arrived_plan.compute_order == ((11, 1),)

    try:
        plan_critical_work((urgent, urgent), now_ns=0, model=model)
    except ValueError as error:
        assert "repeats" in str(error)
    else:
        raise AssertionError("duplicate request generation was accepted")

    invalid = work(99, runnable=1, runnable_ns=1)
    invalid = RequestWork(
        **{
            **invalid.__dict__,
            "pending_compute_ns": 1,
            "expected_compute_ns": 2,
        }
    )
    try:
        estimate_critical_work(invalid, now_ns=0, model=model)
    except ValueError as error:
        assert "without pending work" in str(error)
    else:
        raise AssertionError("inconsistent request progress was accepted")

    dropped = RequestWork(**{**urgent.__dict__, "dropped_attributions": 1})
    try:
        estimate_critical_work(dropped, now_ns=0, model=model)
    except ValueError as error:
        assert "dropped" in str(error)
    else:
        raise AssertionError("dropped request attribution was accepted")


if __name__ == "__main__":
    main()
