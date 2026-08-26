"""Request-aware planning for bounded-HBM tier-streaming operators."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from .abi import bounded_integer as _bounded_integer
from .abi import u32 as _u32
from .abi import u64 as _u64


_INT32_MAX = (1 << 31) - 1
_MAX_REQUEST_PRIORITY = 7


@dataclass(frozen=True)
class TierStreamingRequest:
    """Static per-step facts retained across engine, compiler, and transport."""

    request_id: int
    query_tokens: int
    context_tokens: int
    resident_tokens: int
    priority: int = 0
    deadline_ns: int = 0
    generation: int = 1
    tenant_id: int = 0
    cancelled: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_id",
            _u64(self.request_id, "tier-streaming request ID"),
        )
        object.__setattr__(
            self,
            "query_tokens",
            _bounded_integer(
                self.query_tokens,
                "tier-streaming query tokens",
                minimum=1,
                maximum=_INT32_MAX,
            ),
        )
        context_tokens = _bounded_integer(
            self.context_tokens,
            "tier-streaming context tokens",
            minimum=1,
            maximum=_INT32_MAX,
        )
        object.__setattr__(self, "context_tokens", context_tokens)
        resident_tokens = _bounded_integer(
            self.resident_tokens,
            "tier-streaming resident tokens",
            minimum=0,
            maximum=context_tokens,
        )
        if resident_tokens > context_tokens:
            raise ValueError("resident tokens must be within the context")
        object.__setattr__(self, "resident_tokens", resident_tokens)
        object.__setattr__(
            self,
            "priority",
            _bounded_integer(
                self.priority,
                "tier-streaming priority",
                minimum=0,
                maximum=_MAX_REQUEST_PRIORITY,
            ),
        )
        object.__setattr__(
            self,
            "deadline_ns",
            _u64(self.deadline_ns, "tier-streaming deadline"),
        )
        object.__setattr__(
            self,
            "generation",
            _u32(
                self.generation,
                "tier-streaming generation",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "tenant_id",
            _u32(self.tenant_id, "tier-streaming tenant"),
        )
        if not isinstance(self.cancelled, bool):
            raise ValueError("tier-streaming cancelled flag must be boolean")

    @property
    def external_tokens(self) -> int:
        return self.context_tokens - self.resident_tokens

    @property
    def key(self) -> tuple[int, int]:
        return (self.request_id, self.generation)


@dataclass(frozen=True)
class TierStreamingSegment:
    """One request-owned external KV segment in a transfer wave."""

    request_index: int
    request_id: int
    request_generation: int
    tenant_id: int
    source_token_offset: int
    token_count: int


@dataclass(frozen=True)
class TierStreamingWave:
    """A coalesced transfer and partial-compute wave."""

    index: int
    segments: tuple[TierStreamingSegment, ...]
    completed_request_keys: tuple[tuple[int, int], ...]

    @property
    def active_request_count(self) -> int:
        return len(self.segments)

    @property
    def token_count(self) -> int:
        return sum(segment.token_count for segment in self.segments)

    @property
    def completed_request_ids(self) -> tuple[int, ...]:
        return tuple(
            request_id for request_id, _generation in self.completed_request_keys
        )


@dataclass(frozen=True)
class TierStreamingSchedule:
    """Finite wave schedule with compact active-request prefixes."""

    requests: tuple[TierStreamingRequest, ...]
    waves: tuple[TierStreamingWave, ...]
    group_tokens: int
    cancelled_request_keys: tuple[tuple[int, int], ...] = ()

    @property
    def external_tokens(self) -> int:
        return sum(request.external_tokens for request in self.requests)

    @property
    def maximum_wave_tokens(self) -> int:
        return max((wave.token_count for wave in self.waves), default=0)

    def staging_tokens(self, slot_count: int) -> int:
        if slot_count <= 0:
            raise ValueError("slot count must be positive")
        return slot_count * self.maximum_wave_tokens


@dataclass(frozen=True)
class TierStreamingCostModel:
    """Online inputs used to compare bulk and bounded-HBM execution."""

    transfer_bandwidth_bytes_per_second: int = 56_000_000_000
    attention_pairs_per_second: int = 20_000_000_000
    partial_attention_pairs_per_second: int = 10_000_000_000
    base_launch_ns: int = 20_000
    transfer_setup_ns: int = 10_000
    partial_launch_and_merge_ns: int = 45_000
    minimum_predicted_speedup: float = 1.03

    def validate(self) -> None:
        if (
            min(
                self.transfer_bandwidth_bytes_per_second,
                self.attention_pairs_per_second,
                self.partial_attention_pairs_per_second,
                self.base_launch_ns,
            )
            <= 0
        ):
            raise ValueError(
                "tier-streaming rates and base launch cost must be positive"
            )
        if min(self.transfer_setup_ns, self.partial_launch_and_merge_ns) < 0:
            raise ValueError("tier-streaming setup costs must be nonnegative")
        if self.minimum_predicted_speedup < 1.0:
            raise ValueError("minimum predicted speedup must be at least one")


@dataclass(frozen=True)
class TierStreamingExecutionPlan:
    """One dual-form decision; bulk is the largest legal streaming group."""

    mode: Literal["direct", "bulk", "stream"]
    schedule: TierStreamingSchedule
    group_tokens: int
    staging_bytes: int
    predicted_bulk_ns: int
    predicted_stream_ns: int
    bulk_capacity_feasible: bool

    @property
    def predicted_speedup(self) -> float:
        selected = (
            self.predicted_stream_ns
            if self.mode == "stream"
            else self.predicted_bulk_ns
        )
        if selected == 0:
            return 1.0
        return self.predicted_bulk_ns / selected


def _request_order(request: TierStreamingRequest) -> tuple[int, int, int, int]:
    # Long external spans first make every wave's active set a compact prefix.
    # Deadline and priority retain deterministic policy order for equal spans.
    deadline = request.deadline_ns if request.deadline_ns != 0 else 2**63 - 1
    return (-request.external_tokens, deadline, -request.priority, request.request_id)


def build_tier_streaming_schedule(
    requests: list[TierStreamingRequest] | tuple[TierStreamingRequest, ...],
    group_tokens: int,
) -> TierStreamingSchedule:
    """Build finite coalesced waves for heterogeneous request placements."""

    if not requests:
        raise ValueError("at least one request is required")
    if group_tokens <= 0:
        raise ValueError("group token count must be positive")
    request_ids = [request.request_id for request in requests]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("request IDs must be unique")

    cancelled = tuple(request.key for request in requests if request.cancelled)
    ordered = tuple(
        sorted(
            (request for request in requests if not request.cancelled),
            key=_request_order,
        )
    )
    if not ordered:
        return TierStreamingSchedule((), (), group_tokens, cancelled)
    max_external = max(request.external_tokens for request in ordered)
    waves: list[TierStreamingWave] = []
    for wave_index, source_offset in enumerate(range(0, max_external, group_tokens)):
        segments: list[TierStreamingSegment] = []
        completed: list[tuple[int, int]] = []
        for request_index, request in enumerate(ordered):
            remaining = request.external_tokens - source_offset
            if remaining <= 0:
                break
            token_count = min(group_tokens, remaining)
            segments.append(
                TierStreamingSegment(
                    request_index=request_index,
                    request_id=request.request_id,
                    request_generation=request.generation,
                    tenant_id=request.tenant_id,
                    source_token_offset=source_offset,
                    token_count=token_count,
                )
            )
            if remaining <= group_tokens:
                completed.append(request.key)
        if segments:
            waves.append(
                TierStreamingWave(
                    index=wave_index,
                    segments=tuple(segments),
                    completed_request_keys=tuple(completed),
                )
            )

    return TierStreamingSchedule(
        requests=ordered,
        waves=tuple(waves),
        group_tokens=group_tokens,
        cancelled_request_keys=cancelled,
    )


def _transfer_ns(byte_count: int, model: TierStreamingCostModel) -> int:
    return model.transfer_setup_ns + math.ceil(
        byte_count * 1_000_000_000 / model.transfer_bandwidth_bytes_per_second
    )


def _attention_ns(pair_count: int, pair_rate: int, launch_ns: int) -> int:
    return launch_ns + math.ceil(pair_count * 1_000_000_000 / pair_rate)


def _predict_execution_ns(
    schedule: TierStreamingSchedule,
    *,
    kv_bytes_per_token: int,
    model: TierStreamingCostModel,
) -> tuple[int, int]:
    base_pairs = sum(
        request.query_tokens * (request.resident_tokens + request.query_tokens)
        for request in schedule.requests
    )
    external_pairs = sum(
        request.query_tokens * request.external_tokens for request in schedule.requests
    )
    external_bytes = schedule.external_tokens * kv_bytes_per_token
    base_ns = _attention_ns(
        base_pairs, model.attention_pairs_per_second, model.base_launch_ns
    )
    bulk_transfer_ns = _transfer_ns(external_bytes, model)
    bulk_partial_ns = _attention_ns(
        external_pairs,
        model.attention_pairs_per_second,
        model.partial_launch_and_merge_ns,
    )
    bulk_ns = max(base_ns, bulk_transfer_ns) + bulk_partial_ns

    copy_finish_ns = 0
    compute_finish_ns = base_ns
    for wave in schedule.waves:
        copy_finish_ns += _transfer_ns(wave.token_count * kv_bytes_per_token, model)
        wave_pairs = sum(
            schedule.requests[segment.request_index].query_tokens * segment.token_count
            for segment in wave.segments
        )
        wave_compute_ns = _attention_ns(
            wave_pairs,
            model.partial_attention_pairs_per_second,
            model.partial_launch_and_merge_ns,
        )
        compute_finish_ns = max(compute_finish_ns, copy_finish_ns) + wave_compute_ns
    return bulk_ns, compute_finish_ns


def plan_tier_streaming_execution(
    requests: list[TierStreamingRequest] | tuple[TierStreamingRequest, ...],
    *,
    candidate_group_tokens: tuple[int, ...],
    slot_count: int,
    kv_bytes_per_token: int,
    staging_budget_bytes: int,
    model: TierStreamingCostModel,
) -> TierStreamingExecutionPlan:
    """Select a finite direct, bulk, or bounded-streaming form without an oracle."""

    model.validate()
    if not candidate_group_tokens or any(
        value <= 0 for value in candidate_group_tokens
    ):
        raise ValueError("candidate group sizes must be positive")
    if min(slot_count, kv_bytes_per_token, staging_budget_bytes) <= 0:
        raise ValueError("streaming capacity geometry must be positive")

    schedules = tuple(
        build_tier_streaming_schedule(requests, group_tokens)
        for group_tokens in sorted(set(candidate_group_tokens))
    )
    if not schedules[0].requests or schedules[0].external_tokens == 0:
        return TierStreamingExecutionPlan(
            "direct", schedules[0], schedules[0].group_tokens, 0, 0, 0, True
        )

    external_bytes = schedules[0].external_tokens * kv_bytes_per_token
    candidates: list[tuple[int, int, TierStreamingSchedule]] = []
    bulk_ns = 0
    for schedule in schedules:
        staging_bytes = schedule.staging_tokens(slot_count) * kv_bytes_per_token
        if staging_bytes > staging_budget_bytes:
            continue
        predicted_bulk_ns, predicted_stream_ns = _predict_execution_ns(
            schedule, kv_bytes_per_token=kv_bytes_per_token, model=model
        )
        bulk_ns = predicted_bulk_ns
        candidates.append((predicted_stream_ns, staging_bytes, schedule))
    if not candidates:
        raise ValueError("no tier-streaming group fits the HBM staging budget")

    stream_ns, staging_bytes, schedule = min(candidates, key=lambda value: value[:2])
    bulk_feasible = external_bytes <= staging_budget_bytes
    use_stream = not bulk_feasible or (
        bulk_ns / stream_ns >= model.minimum_predicted_speedup
    )
    if use_stream:
        return TierStreamingExecutionPlan(
            "stream",
            schedule,
            schedule.group_tokens,
            staging_bytes,
            bulk_ns,
            stream_ns,
            bulk_feasible,
        )
    bulk_schedule = build_tier_streaming_schedule(
        requests,
        max(request.external_tokens for request in schedule.requests),
    )
    return TierStreamingExecutionPlan(
        "bulk",
        bulk_schedule,
        bulk_schedule.group_tokens,
        external_bytes,
        bulk_ns,
        stream_ns,
        True,
    )
