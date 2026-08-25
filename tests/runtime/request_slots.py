#!/usr/bin/env python3
"""Validate engine-neutral request identity and slot reuse."""

from __future__ import annotations

import nta_runtime.requests as requests_module
from nta_runtime.requests import RequestIdentityRegistry, stable_request_id


class Runtime:
    def __init__(self) -> None:
        self.updates: list[tuple[int, int, int, int, int]] = []
        self.tenant_updates: list[int] = []
        self.cancellations: list[tuple[int, int]] = []
        self.async_updates = []

    def set_request(
        self,
        slot: int,
        request_id: int,
        generation: int,
        *,
        tenant_id: int,
        priority: int,
        deadline_clock: int,
    ) -> None:
        self.updates.append((slot, request_id, generation, priority, deadline_clock))
        self.tenant_updates.append(tenant_id)

    def publish_requests_async(self, requests, stream) -> None:
        self.async_updates.append((tuple(requests), stream))

    def cancel_request(self, slot: int, generation: int) -> None:
        self.cancellations.append((slot, generation))


runtime = Runtime()
registry = RequestIdentityRegistry(runtime, 4)
first = registry.bind(["request-a", "request-b"], [2, 0])
assert [binding.generation for binding in first] == [1, 1]
assert registry.last_publish_count == 2
assert runtime.updates == [
    (2, stable_request_id("request-a"), 1, 0, 0),
    (0, stable_request_id("request-b"), 1, 0, 0),
]

tenant_runtime = Runtime()
tenant_registry = RequestIdentityRegistry(tenant_runtime, 4)
multi_tenant = tenant_registry.bind(
    ["request-a", "request-b"], [2, 0], tenant_ids=[4, 9]
)
assert [binding.tenant_id for binding in multi_tenant] == [4, 9]
assert tenant_runtime.tenant_updates == [4, 9]

try:
    registry.bind(["request-a", "request-a"], [1, 3])
except ValueError as error:
    assert "multiple slots" in str(error)
else:
    raise AssertionError("one request ID was accepted in multiple slots")

same = registry.bind(["request-a", "request-b"], [2, 0])
assert same == first
assert registry.last_publish_count == 0
assert len(runtime.updates) == 2

reprioritized = registry.bind(["request-a", "request-b"], [2, 0], priorities=[7, 2])
assert [binding.priority for binding in reprioritized] == [7, 2]
assert registry.last_publish_count == 0
assert registry.last_metadata_publish_count == 2
assert len(runtime.updates) == 4

replacement = registry.bind(["request-c"], [2])
assert replacement[0].generation == 2
assert registry.last_publish_count == 1
assert registry.cancel("request-c")
assert runtime.cancellations == [(2, 2)]
same_id_after_cancel = registry.bind(["request-c"], [2])
assert same_id_after_cancel[0].generation == 3
assert registry.last_publish_count == 1
assert not registry.cancel("missing")
registry.bind(["agent/branch-a", "agent/branch-b", "other"], [0, 1, 3])
assert registry.cancel_matching("agent/") == 2
assert runtime.cancellations[-2:] == [(0, 2), (1, 1)]
assert registry.cancel_matching(all=True) == 2

collision_runtime = Runtime()
collision_registry = RequestIdentityRegistry(collision_runtime, 2)
original_stable_request_id = requests_module.stable_request_id
requests_module.stable_request_id = lambda _: 17
try:
    collision_registry.bind(["collision-a"], [0])
    try:
        collision_registry.bind(["collision-b"], [1])
    except ValueError as error:
        assert "hash collision" in str(error)
    else:
        raise AssertionError("request identity collision did not fail closed")
finally:
    requests_module.stable_request_id = original_stable_request_id

async_runtime = Runtime()
async_tracker = RequestIdentityRegistry(async_runtime, 4)
async_bindings = async_tracker.bind(["request-z", "request-y"], [1, 0], stream=1234)
assert [binding.request_slot for binding in async_bindings] == [1, 0]
assert async_runtime.updates == []
assert len(async_runtime.async_updates) == 1
updates, stream = async_runtime.async_updates[0]
assert stream == 1234
assert [update.slot for update in updates] == [1, 0]
