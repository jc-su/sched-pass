#!/usr/bin/env python3
"""Unit tests for request-aware tier-streaming wave planning."""

from __future__ import annotations

from nta_runtime.tier_streaming import (
    TierStreamingCostModel,
    TierStreamingRequest,
    build_tier_streaming_schedule,
    plan_tier_streaming_execution,
)


def expect_error(callable_object, message: str) -> None:
    try:
        callable_object()
    except ValueError:
        return
    raise AssertionError(message)


def main() -> int:
    requests = [
        TierStreamingRequest(
            40,
            64,
            1024,
            768,
            priority=1,
            deadline_ns=900,
            generation=4,
            tenant_id=2,
        ),
        TierStreamingRequest(
            10, 256, 1024, 0, priority=2, deadline_ns=300, generation=7
        ),
        TierStreamingRequest(
            30, 128, 1024, 512, priority=7, deadline_ns=400, tenant_id=3
        ),
        TierStreamingRequest(20, 32, 1024, 1024, priority=0, deadline_ns=100),
    ]
    schedule = build_tier_streaming_schedule(requests, 256)

    assert [request.request_id for request in schedule.requests] == [10, 30, 40, 20]
    assert [wave.active_request_count for wave in schedule.waves] == [3, 2, 1, 1]
    assert [wave.token_count for wave in schedule.waves] == [768, 512, 256, 256]
    assert [wave.completed_request_ids for wave in schedule.waves] == [
        (40,),
        (30,),
        (),
        (10,),
    ]
    assert [wave.completed_request_keys for wave in schedule.waves] == [
        ((40, 4),),
        ((30, 0),),
        (),
        ((10, 7),),
    ]
    assert schedule.waves[0].segments[0].request_generation == 7
    assert schedule.waves[0].segments[1].tenant_id == 3
    assert schedule.external_tokens == 1792
    assert schedule.maximum_wave_tokens == 768
    assert schedule.staging_tokens(2) == 1536
    assert schedule.requests[-1].external_tokens == 0

    cancelled = TierStreamingRequest(50, 32, 1024, 0, generation=9, cancelled=True)
    without_cancelled = build_tier_streaming_schedule(requests + [cancelled], 256)
    assert without_cancelled.cancelled_request_keys == ((50, 9),)
    assert all(request.request_id != 50 for request in without_cancelled.requests)
    cancelled_only = build_tier_streaming_schedule([cancelled], 256)
    assert not cancelled_only.requests and not cancelled_only.waves

    kv_bytes_per_token = 4096
    streaming = plan_tier_streaming_execution(
        requests,
        candidate_group_tokens=(64, 128, 256),
        slot_count=2,
        kv_bytes_per_token=kv_bytes_per_token,
        staging_budget_bytes=2 * 768 * kv_bytes_per_token,
        model=TierStreamingCostModel(),
    )
    assert streaming.mode == "stream"
    assert not streaming.bulk_capacity_feasible
    assert streaming.staging_bytes <= 2 * 768 * kv_bytes_per_token

    bulk = plan_tier_streaming_execution(
        requests,
        candidate_group_tokens=(64, 128, 256),
        slot_count=2,
        kv_bytes_per_token=kv_bytes_per_token,
        staging_budget_bytes=8 * 1024 * kv_bytes_per_token,
        model=TierStreamingCostModel(
            partial_attention_pairs_per_second=1,
            minimum_predicted_speedup=1.01,
        ),
    )
    assert bulk.mode == "bulk"
    assert bulk.bulk_capacity_feasible
    assert len(bulk.schedule.waves) == 1

    resident = plan_tier_streaming_execution(
        [TierStreamingRequest(1, 1, 128, 128)],
        candidate_group_tokens=(64,),
        slot_count=2,
        kv_bytes_per_token=kv_bytes_per_token,
        staging_budget_bytes=kv_bytes_per_token,
        model=TierStreamingCostModel(),
    )
    assert resident.mode == "direct"

    expect_error(
        lambda: build_tier_streaming_schedule([], 1), "empty schedule accepted"
    )
    expect_error(
        lambda: build_tier_streaming_schedule(requests, 0),
        "zero-sized group accepted",
    )
    expect_error(
        lambda: build_tier_streaming_schedule([requests[0], requests[0]], 1),
        "duplicate request IDs accepted",
    )
    expect_error(
        lambda: TierStreamingRequest(1, 1, 8, 9),
        "invalid resident range accepted",
    )
    expect_error(
        lambda: plan_tier_streaming_execution(
            requests,
            candidate_group_tokens=(256,),
            slot_count=2,
            kv_bytes_per_token=kv_bytes_per_token,
            staging_budget_bytes=1,
            model=TierStreamingCostModel(),
        ),
        "impossible staging budget accepted",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
