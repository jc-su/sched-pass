#!/usr/bin/env python3
"""Validate the compiler-progress/framework-admission state contract."""

from __future__ import annotations

from dataclasses import dataclass, replace

from nta_runtime.progress_frontier import (
    FrontierState,
    RequestFrontierEntry,
    build_request_frontier,
)


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


def entry(request_id: int, *, pending: int = 0, runnable: int = 0):
    progress = Progress(
        request_id,
        1,
        pending + runnable,
        pending,
        runnable,
        0,
        0,
        0,
        4096 if pending else 0,
        3000 if runnable else 0,
        0,
        3000 if pending else 0,
        3000 * (pending + runnable),
    )
    return RequestFrontierEntry.from_progress(progress)


def main() -> None:
    blocked = entry(11, pending=1)
    runnable = entry(12, runnable=1)
    mixed = build_request_frontier((blocked, runnable))
    assert mixed.state is FrontierState.EXECUTABLE
    assert mixed.executable == ((12, 1),)
    assert mixed.data_blocked == ((11, 1),)

    assert build_request_frontier((blocked,)).state is FrontierState.DATA_BLOCKED
    complete = replace(
        entry(13, runnable=1),
        runnable_work=0,
        runnable_compute_ns=0,
        completed_work=1,
        completed_compute_ns=3000,
    )
    assert build_request_frontier((complete,)).state is FrontierState.QUIESCENT

    for invalid, message in (
        ((blocked, blocked), "repeats"),
        ((replace(blocked, dropped_attributions=1),), "dropped"),
        ((replace(blocked, pending_work=0),), "without pending"),
    ):
        try:
            build_request_frontier(invalid)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError("invalid request frontier was accepted")


if __name__ == "__main__":
    main()
