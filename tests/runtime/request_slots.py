#!/usr/bin/env python3
"""Validate engine-neutral request identity and slot reuse."""

from __future__ import annotations

from nta_runtime.requests import RequestSlotTracker, stable_request_id


class Runtime:
    def __init__(self) -> None:
        self.updates: list[tuple[int, int, int, int, int]] = []
        self.cancellations: list[tuple[int, int]] = []

    def set_request(
        self,
        slot: int,
        request_id: int,
        generation: int,
        *,
        priority: int,
        deadline_clock: int,
    ) -> None:
        self.updates.append(
            (slot, request_id, generation, priority, deadline_clock)
        )

    def cancel_request(self, slot: int, generation: int) -> None:
        self.cancellations.append((slot, generation))


runtime = Runtime()
tracker = RequestSlotTracker(runtime, 4)
first = tracker.bind(["request-a", "request-b"], [2, 0])
assert [binding.generation for binding in first] == [1, 1]
assert tracker.last_publish_count == 2
assert runtime.updates == [
    (2, stable_request_id("request-a"), 1, 0, 0),
    (0, stable_request_id("request-b"), 1, 0, 0),
]

same = tracker.bind(["request-a", "request-b"], [2, 0])
assert same == first
assert tracker.last_publish_count == 0
assert len(runtime.updates) == 2

reprioritized = tracker.bind(
    ["request-a", "request-b"], [2, 0], priorities=[7, 2]
)
assert [binding.priority for binding in reprioritized] == [7, 2]
assert tracker.last_publish_count == 0
assert tracker.last_policy_publish_count == 2
assert len(runtime.updates) == 4

replacement = tracker.bind(["request-c"], [2])
assert replacement[0].generation == 2
assert tracker.last_publish_count == 1
assert tracker.cancel("request-c")
assert runtime.cancellations == [(2, 2)]
assert not tracker.cancel("missing")
tracker.bind(["agent/branch-a", "agent/branch-b", "other"], [0, 1, 3])
assert tracker.cancel_matching("agent/") == 2
assert runtime.cancellations[-2:] == [(0, 2), (1, 1)]
assert tracker.cancel_matching(all=True) == 4
