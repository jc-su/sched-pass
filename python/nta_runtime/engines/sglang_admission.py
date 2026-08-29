"""Acquisition-aware SGLang admission for external KV batches."""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any, Callable

from nta_runtime.engines.sglang_hicache import SglangHiCacheBridge, find_bridge
from nta_runtime.progress_frontier import FrontierState
from nta_runtime.requests import stable_request_id


_STATE_ATTRIBUTE = "_nta_acquisition_admission"


def _nonnegative_environment(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 0:
        raise RuntimeError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True)
class AdmissionConfig:
    enabled: bool
    max_delay_ns: int

    @classmethod
    def from_environment(cls) -> "AdmissionConfig":
        return cls(
            enabled=os.environ.get("NTA_EXECUTION_ADMISSION", "1") != "0",
            max_delay_ns=1_000
            * _nonnegative_environment("NTA_EXECUTION_ADMISSION_MAX_DELAY_US", 10_000),
        )


@dataclass
class _StagedBatch:
    batch: Any
    bridge: SglangHiCacheBridge
    consumer_index: int
    started_ns: int
    force_release: bool = False


def _has_runnable_decode(running: Any) -> bool:
    for request in tuple(getattr(running, "reqs", ()) or ()):
        if not request.finished():
            return True
    return False


def _running_request_ids(running: Any) -> set[int]:
    return {
        stable_request_id(str(request.rid))
        for request in tuple(getattr(running, "reqs", ()) or ())
        if not request.finished() and getattr(request, "rid", None)
    }


def _compiler_feedback_reason(running: Any, bridge: SglangHiCacheBridge) -> str | None:
    frontier = bridge.poll_request_frontier(_running_request_ids(running))
    if frontier is None:
        return None
    if frontier.state is FrontierState.EXECUTABLE:
        bridge.record_admission(admission_feedback_executable=1)
        return None
    if frontier.state is FrontierState.DATA_BLOCKED:
        bridge.record_admission(admission_feedback_data_blocked=1)
        return "data_blocked"
    bridge.record_admission(admission_feedback_quiescent=1)
    return "quiescent"


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
        *,
        running_batch: Any | None = None,
    ) -> Any | None:
        if batch is None or not self._config.enabled or bridge is None:
            return batch
        if self._staged is not None:
            raise RuntimeError("acquisition admission already owns a batch")
        running = (
            running_batch
            if running_batch is not None
            else getattr(scheduler, "running_batch", None)
        )
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
        if progress.complete:
            bridge.record_admission(admission_released_complete=1)
            return batch
        if not _has_runnable_decode(running):
            bridge.record_admission(admission_released_without_decode=1)
            return batch
        feedback_reason = _compiler_feedback_reason(running, bridge)
        if feedback_reason is not None:
            bridge.record_admission(
                **{f"admission_released_feedback_{feedback_reason}": 1}
            )
            return batch
        unpublished = progress.published_layers == 0
        if unpublished and not bridge.prepare_admission_acquisition(
            consumer_index, batch
        ):
            # Exact semantic WorkItems are not available until forward metadata.
            # An uncalibrated physical group therefore stays descriptor-free and
            # reaches the typed partial path instead of being held speculatively.
            bridge.record_admission(admission_released_for_binding=1)
            return batch
        feasibility = self._feasibility(bridge, batch, progress)
        if feasibility is None:
            bridge.record_admission(admission_released_uncalibrated=1)
            return batch
        self._record_feasibility(bridge, feasibility)
        if unpublished and (
            feasibility.required_initial_slack_ns > self._config.max_delay_ns
        ):
            # Starting a finite queue that cannot recover inside the policy's
            # SLO cap would turn exact metadata binding into avoidable delay.
            # Leave the link untouched and bind exact work immediately.
            bridge.record_admission(admission_released_partial_slo=1)
            return batch
        if unpublished:
            bridge.start_admission_acquisition(consumer_index, batch)
            progress = bridge.progress(consumer_index)
            if progress is None or progress.published_layers == 0:
                raise RuntimeError(
                    "HiCache admission frontier did not publish transfer progress"
                )
        if feasibility.feasible:
            bridge.record_admission(admission_released_feasible=1)
            return batch
        if self._config.max_delay_ns == 0:
            bridge.record_admission(admission_released_slo_cap=1)
            return batch

        now = self._clock()
        self._staged = _StagedBatch(
            batch,
            bridge,
            consumer_index,
            now,
        )
        bridge.record_admission(
            admission_delayed_batches=1,
            admission_delayed_requests=len(requests),
        )
        return None

    def poll(self, scheduler: Any, *, running_batch: Any | None = None) -> Any | None:
        staged = self._staged
        if staged is None:
            raise RuntimeError("acquisition admission has no staged batch")
        running = (
            running_batch
            if running_batch is not None
            else getattr(scheduler, "running_batch", None)
        )
        now = self._clock()
        elapsed = now - staged.started_ns
        staged.bridge.record_admission(admission_delay_polls=1)

        progress = staged.bridge.progress(staged.consumer_index)

        reason = ""
        if staged.force_release:
            reason = "cancelled"
        elif progress is None:
            reason = "lost"
        elif progress.complete:
            reason = "complete"
        elif elapsed >= self._config.max_delay_ns:
            reason = "slo_cap"
        elif not _has_runnable_decode(running):
            reason = "no_decode"
        else:
            feedback_reason = _compiler_feedback_reason(running, staged.bridge)
            if feedback_reason is not None:
                reason = f"feedback_{feedback_reason}"
            else:
                feasibility = self._feasibility(
                    staged.bridge, staged.batch, progress
                )
                if feasibility is None:
                    reason = "uncalibrated"
                else:
                    self._record_feasibility(staged.bridge, feasibility)
                    if feasibility.feasible:
                        reason = "feasible"
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

    @staticmethod
    def _feasibility(bridge: Any, batch: Any, progress: Any) -> Any | None:
        model = bridge.deadline_model(progress.consumer_index, batch)
        if model is None:
            return None
        return model.analyze_admission(ready_prefix_layers=progress.leading_layers)

    @staticmethod
    def _record_feasibility(bridge: Any, feasibility: Any) -> None:
        increments = {
            "admission_feasibility_tests": 1,
            "admission_feasibility_ready_prefix_layers": (
                feasibility.ready_prefix_layers
            ),
            "admission_feasibility_required_slack_ns": (
                feasibility.required_initial_slack_ns
            ),
        }
        if feasibility.feasible:
            increments["admission_feasibility_feasible"] = 1
        else:
            increments["admission_feasibility_infeasible"] = 1
            increments["admission_feasibility_first_missed_layer"] = int(
                feasibility.first_missed_layer
            )
        bridge.record_admission(**increments)


def _state(scheduler: Any) -> AcquisitionAdmission:
    state = getattr(scheduler, _STATE_ATTRIBUTE, None)
    if state is None:
        state = AcquisitionAdmission(AdmissionConfig.from_environment())
        setattr(scheduler, _STATE_ATTRIBUTE, state)
    return state


def _prefill_running_batch(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Extract the pinned SGLang 0.5.16 prefill input from the hook call."""
    if "running_batch" in kwargs:
        return kwargs["running_batch"]
    # The target's pinned signature is
    # (self, prefill_delayer_single_pass, running_batch).  Keep this explicit:
    # silently guessing another framework signature would make admission act
    # on the wrong running set.
    if len(args) == 2:
        return args[1]
    raise RuntimeError(
        "SGLang prefill admission hook received an unsupported argument shape"
    )


def _split_prefill_result(result: Any) -> tuple[Any | None, Any]:
    """Validate the pinned SGLang raw-prefill return contract."""
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(
            "SGLang 0.5.16 prefill hook returned neither "
            "(batch_to_run, running_batch) nor the pinned tuple shape"
        )
    return result[0], result[1]


def route_prefill_admission(
    original: Callable[..., Any], scheduler: Any, *args: Any, **kwargs: Any
) -> tuple[Any | None, Any]:
    running_batch = _prefill_running_batch(args, kwargs)
    state = _state(scheduler)
    if state.has_staged_batch:
        return state.poll(scheduler, running_batch=running_batch), running_batch
    batch, next_running_batch = _split_prefill_result(
        original(scheduler, *args, **kwargs)
    )
    admitted = state.consider(
        scheduler,
        batch,
        _bridge_for_batch(batch),
        running_batch=next_running_batch,
    )
    return admitted, next_running_batch


def cancel_staged_batch(scheduler: Any, recv_req: Any) -> None:
    state = getattr(scheduler, _STATE_ATTRIBUTE, None)
    if state is None:
        return
    state.cancel(
        str(getattr(recv_req, "rid", "") or ""),
        all=bool(getattr(recv_req, "abort_all", False)),
    )
