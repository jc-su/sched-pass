"""vLLM 0.26 V1 plugin bootstrap.

vLLM's general-plugin group is intentionally used only as a registration
point.  The numerical path is the registered V1 ``AttentionBackend``; the
small worker patch exists solely to publish scheduler identity after
``GPUModelRunner._update_states`` has refreshed the persistent input batch.
"""

from __future__ import annotations

from collections.abc import Mapping
import functools
import importlib.metadata
from typing import Any

from nta_runtime.adapters.vllm_v1 import (
    current_vllm_v1_forward_state,
    vllm_v1_forward_state,
)


SUPPORTED_VLLM_VERSION = "0.26.0"
_REGISTERED = False
_PATCHED = False


def _check_version() -> None:
    try:
        installed = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("vLLM is not installed") from error
    if installed != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(
            f"NTA vLLM plugin requires {SUPPORTED_VLLM_VERSION}, found {installed}"
        )


def _has_scheduled_work(scheduler_output: Any) -> bool:
    scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
    return isinstance(scheduled, Mapping) and bool(scheduled)


def _patch_worker() -> None:
    global _PATCHED
    if _PATCHED:
        return
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original_update = GPUModelRunner._update_states
    original_execute = GPUModelRunner.execute_model

    @functools.wraps(original_update)
    def update_states(self, scheduler_output, *args, **kwargs):
        result = original_update(self, scheduler_output, *args, **kwargs)
        state = current_vllm_v1_forward_state()
        if state is None or not _has_scheduled_work(scheduler_output):
            return result
        from nta_runtime.engines.vllm import _controller

        controller = _controller(self)
        state.scheduler_output = scheduler_output
        state.input_batch = self.input_batch
        state.batch = controller.bind(scheduler_output)
        state.hook = controller.hook
        state.page_size = controller.page_size
        return result

    @functools.wraps(original_execute)
    def execute_model(self, scheduler_output, intermediate_tensors=None):
        with vllm_v1_forward_state(scheduler_output):
            return original_execute(self, scheduler_output, intermediate_tensors)

    GPUModelRunner._update_states = update_states
    GPUModelRunner.execute_model = execute_model
    _PATCHED = True


def register() -> None:
    """Register the custom backend and install the worker sidecar bridge."""
    global _REGISTERED
    if _REGISTERED:
        return
    _check_version()
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.CUSTOM,
        class_path="nta_runtime.engines.vllm.NtaVllmFlashInferBackend",
    )
    _REGISTERED = True
    # The vLLM general-plugin loader runs in every engine process that can
    # execute a worker.  Do not swallow an import or patching error here:
    # without the sidecar, CUSTOM backend selection would leave identity and
    # exact demand unbound and fail much later inside attention.  Version
    # checking above makes this strict patch safe for the pinned profile.
    _patch_worker()
