"""Acquisition-aware SGLang admission for external KV batches."""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any, Callable

from nta_runtime.engines.sglang_hicache import SglangHiCacheBridge, find_bridge
from nta_runtime.requests import stable_request_id


_STATE_ATTRIBUTE = "_nta_acquisition_admission"


def _nonnegative_environment(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 0:
        raise RuntimeError(f"{name} must be nonnegative")
    return value


def _positive_environment(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise RuntimeError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class AdmissionConfig:
    enabled: bool
    lead_layers: int
    max_delay_ns: int
    minimum_bytes: int

    @classmethod
    def from_environment(cls) -> "AdmissionConfig":
        return cls(
            enabled=os.environ.get("NTA_SGLANG_ACQUISITION_ADMISSION", "1") != "0",
            lead_layers=_positive_environment("NTA_SGLANG_ADMISSION_LEAD_LAYERS", 4),
            max_delay_ns=1_000
            * _nonnegative_environment("NTA_SGLANG_ADMISSION_MAX_DELAY_US", 10_000),
            minimum_bytes=_nonnegative_environment(
                "NTA_SGLANG_ADMISSION_MIN_BYTES", 1 << 20
            ),
        )


@dataclass
class _StagedBatch:
    batch: Any
    bridge: SglangHiCacheBridge
    consumer_index: int
    started_ns: int
    request_count: int
    external_bytes: int
    force_release: bool = False


def _has_runnable_decode(scheduler: Any) -> bool:
    running = getattr(scheduler, "running_batch", None)
    for request in tuple(getattr(running, "reqs", ()) or ()):
        if not request.finished():
            return True
    return False


def _running_request_ids(scheduler: Any) -> set[int]:
    running = getattr(scheduler, "running_batch", None)
    return {
        stable_request_id(str(request.rid))
        for request in tuple(getattr(running, "reqs", ()) or ())
        if not request.finished() and getattr(request, "rid", None)
    }


def _compiler_feedback_reason(
    scheduler: Any, bridge: SglangHiCacheBridge
) -> str | None:
    plan = bridge.poll_critical_work(_running_request_ids(scheduler))
    if plan is None:
        return None
    if plan.compute_order:
        bridge.record_admission(admission_feedback_executable=1)
        return None
    if plan.data_order:
        bridge.record_admission(admission_feedback_data_blocked=1)
        return "data_blocked"
    bridge.record_admission(admission_feedback_terminal=1)
    return "terminal"


def _bridge_for_batch(batch: Any) -> SglangHiCacheBridge | None:
    tree_cache = getattr(batch, "tree_cache", None)
    for _ in range(3):
        controller = getattr(tree_cache, "cache_controller", None)
        if controller is not None:
            device_pool = getattr(controller, "mem_pool_device", None)
            if device_pool is not None:
                return find_bridge(device_pool)
        tree_cache = getattr(tree_cache, "inner", None)
        if tree_cache is None:
            break
    return None


class AcquisitionAdmission:
    """Overlap an allocated external batch with useful resident decode work."""

    def __init__(
        self,
        config: AdmissionConfig,
        *,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._config = config
        self._clock = clock
        self._staged: _StagedBatch | None = None

    @property
    def has_staged_batch(self) -> bool:
        return self._staged is not None

    def consider(
        self,
        scheduler: Any,
        batch: Any,
        bridge: SglangHiCacheBridge | None,
    ) -> Any | None:
        if batch is None or not self._config.enabled or bridge is None:
            return batch
        if self._staged is not None:
            raise RuntimeError("acquisition admission already owns a batch")
        consumer_index = int(getattr(batch, "hicache_consumer_index", -1))
        requests = tuple(getattr(batch, "reqs", ()) or ())
        external_bytes = bridge.transfer_bytes(consumer_index)
        if getattr(batch, "decoding_reqs", None) is not None:
            bridge.record_admission(
                admission_considered_batches=1,
                admission_considered_requests=len(requests),
                admission_external_bytes=external_bytes,
                admission_released_mixed_batches=1,
            )
            return batch
        progress = bridge.progress(consumer_index)
        if progress is None:
            return batch
        bridge.record_admission(
            admission_considered_batches=1,
            admission_considered_requests=len(requests),
            admission_external_bytes=external_bytes or progress.total_bytes,
        )
        if progress.total_bytes < self._config.minimum_bytes:
            bridge.record_admission(admission_released_small_batches=1)
            return batch
        if not _has_runnable_decode(scheduler):
            bridge.record_admission(admission_released_without_decode=1)
            return batch
        feedback_reason = _compiler_feedback_reason(scheduler, bridge)
        if feedback_reason is not None:
            bridge.record_admission(
                **{f"admission_released_feedback_{feedback_reason}": 1}
            )
            return batch
        if self._has_lead(progress.leading_layers, progress.total_layers):
            bridge.record_admission(admission_released_with_initial_lead=1)
            return batch

        now = self._clock()
        self._staged = _StagedBatch(
            batch,
            bridge,
            consumer_index,
            now,
            len(requests),
            progress.total_bytes,
        )
        bridge.record_admission(
            admission_delayed_batches=1,
            admission_delayed_requests=len(requests),
        )
        return None

    def poll(self, scheduler: Any) -> Any | None:
        staged = self._staged
        if staged is None:
            raise RuntimeError("acquisition admission has no staged batch")
        now = self._clock()
        elapsed = now - staged.started_ns
        progress = staged.bridge.progress(staged.consumer_index)
        staged.bridge.record_admission(admission_delay_polls=1)

        reason = ""
        if staged.force_release:
            reason = "cancelled"
        elif progress is None:
            reason = "lost"
        elif self._has_lead(progress.leading_layers, progress.total_layers):
            reason = "lead"
        elif elapsed >= self._config.max_delay_ns:
            reason = "deadline"
        elif not _has_runnable_decode(scheduler):
            reason = "no_decode"
        else:
            feedback_reason = _compiler_feedback_reason(scheduler, staged.bridge)
            if feedback_reason is not None:
                reason = f"feedback_{feedback_reason}"
        if not reason:
            staged.bridge.record_admission(admission_hidden_decode_steps=1)
            return None

        staged.bridge.record_admission(
            admission_delay_ns=elapsed,
            **{f"admission_released_{reason}": 1},
        )
        self._staged = None
        return staged.batch

    def cancel(self, request_id: str, *, all: bool) -> None:
        staged = self._staged
        if staged is None:
            return
        matches = [
            request
            for request in tuple(getattr(staged.batch, "reqs", ()) or ())
            if all or str(getattr(request, "rid", "")).startswith(request_id)
        ]
        if not matches:
            return
        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        for request in matches:
            if not request.finished():
                request.to_finish = FINISH_ABORT()
        staged.force_release = True
        staged.bridge.record_admission(admission_cancelled_requests=len(matches))

    def _has_lead(self, leading_layers: int, total_layers: int) -> bool:
        return total_layers > 0 and leading_layers >= min(
            self._config.lead_layers, total_layers
        )


def _state(scheduler: Any) -> AcquisitionAdmission:
    state = getattr(scheduler, _STATE_ATTRIBUTE, None)
    if state is None:
        state = AcquisitionAdmission(AdmissionConfig.from_environment())
        setattr(scheduler, _STATE_ATTRIBUTE, state)
    return state


def route_prefill_admission(
    original: Callable[..., Any], scheduler: Any, *args: Any, **kwargs: Any
) -> Any | None:
    state = _state(scheduler)
    if state.has_staged_batch:
        return state.poll(scheduler)
    batch = original(scheduler, *args, **kwargs)
    return state.consider(scheduler, batch, _bridge_for_batch(batch))


def cancel_staged_batch(scheduler: Any, recv_req: Any) -> None:
    state = getattr(scheduler, _STATE_ATTRIBUTE, None)
    if state is None:
        return
    state.cancel(
        str(getattr(recv_req, "rid", "") or ""),
        all=bool(getattr(recv_req, "abort_all", False)),
    )
