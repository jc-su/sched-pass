"""vLLM 0.26 plugin bootstrap.

vLLM's general-plugin group is intentionally used only as a registration
point.  The numerical path is the registered ``AttentionBackend``; the
worker bridge publishes scheduler identity at the pinned vLLM ``v1`` worker
boundary. The current profile uses its V2 model runner.
"""

from __future__ import annotations

from collections.abc import Mapping
import functools
import importlib.metadata
from typing import Any

from nta_runtime.adapters.vllm_v1 import (
    current_vllm_v1_forward_state,
    vllm_v1_reference_warmup_state,
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


def _abort_failed_forward(state: Any, failure: BaseException) -> None:
    """Run every NTA abort hook without hiding the numerical failure."""
    from nta_runtime.engines.vllm import _abort_forward

    try:
        _abort_forward(state)
    except BaseException as abort_failure:
        failure.add_note(f"vLLM NTA forward abort also failed: {abort_failure!r}")


def _patch_v1_runner(runner_class: type[Any]) -> None:
    if getattr(runner_class, "_nta_bridge_patched", False):
        return
    original_update = runner_class._update_states
    original_execute = runner_class.execute_model
    original_dummy_run = runner_class._dummy_run
    original_shutdown = runner_class.shutdown

    @functools.wraps(original_update)
    def update_states(self, scheduler_output, *args, **kwargs):
        result = original_update(self, scheduler_output, *args, **kwargs)
        state = current_vllm_v1_forward_state()
        if (
            state is None
            or state.reference_warmup
            or not _has_scheduled_work(scheduler_output)
        ):
            return result
        from nta_runtime.engines.vllm import _controller

        controller = _controller(self)
        state.scheduler_output = scheduler_output
        state.input_batch = self.input_batch
        state.execution_owner = controller
        controller.begin_forward(state)
        state.batch = controller.bind(scheduler_output)
        controller.prepare_physical_destinations()
        state.hook = controller.hook
        state.tier_service = controller.tier_service
        state.tenant_isolation_enabled = controller.tenant_isolation_enabled
        state.request_bindings_tensor = controller.request_bindings_tensor
        state.page_size = controller.page_size
        return result

    @functools.wraps(original_execute)
    def execute_model(self, scheduler_output, intermediate_tensors=None):
        state = current_vllm_v1_forward_state()
        if state is not None and state.reference_warmup:
            return original_execute(self, scheduler_output, intermediate_tensors)
        with vllm_v1_forward_state(scheduler_output) as forward_state:
            try:
                result = original_execute(self, scheduler_output, intermediate_tensors)
                from nta_runtime.engines.vllm import _commit_forward

                _commit_forward(forward_state)
                return result
            except BaseException as failure:
                _abort_failed_forward(forward_state, failure)
                raise

    @functools.wraps(original_dummy_run)
    def dummy_run(self, *args, **kwargs):
        with vllm_v1_reference_warmup_state():
            return original_dummy_run(self, *args, **kwargs)

    @functools.wraps(original_shutdown)
    def shutdown(self, *args, **kwargs):
        try:
            return original_shutdown(self, *args, **kwargs)
        finally:
            controller = getattr(self, "_nta_vllm_controller", None)
            if controller is not None:
                controller.close()

    runner_class._update_states = update_states
    runner_class.execute_model = execute_model
    runner_class._dummy_run = dummy_run
    runner_class.shutdown = shutdown
    runner_class._nta_bridge_patched = True


def _patch_v2_runner(runner_class: type[Any]) -> None:
    if getattr(runner_class, "_nta_bridge_patched", False):
        return
    original_execute = runner_class.execute_model
    original_dummy_run = runner_class._dummy_run
    original_prepare_attn = runner_class.prepare_attn
    original_shutdown = runner_class.shutdown

    @functools.wraps(original_execute)
    def execute_model(
        self,
        scheduler_output,
        intermediate_tensors=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        is_profile=False,
    ):
        state = current_vllm_v1_forward_state()
        if dummy_run or (state is not None and state.reference_warmup):
            if state is not None and state.reference_warmup:
                return original_execute(
                    self,
                    scheduler_output,
                    intermediate_tensors,
                    dummy_run,
                    skip_attn_for_dummy_run,
                    is_profile,
                )
            with vllm_v1_reference_warmup_state():
                return original_execute(
                    self,
                    scheduler_output,
                    intermediate_tensors,
                    dummy_run,
                    skip_attn_for_dummy_run,
                    is_profile,
                )
        with vllm_v1_forward_state(scheduler_output) as forward_state:
            try:
                result = original_execute(
                    self,
                    scheduler_output,
                    intermediate_tensors,
                    dummy_run,
                    skip_attn_for_dummy_run,
                    is_profile,
                )
                from nta_runtime.engines.vllm import _commit_forward

                _commit_forward(forward_state)
                return result
            except BaseException as failure:
                _abort_failed_forward(forward_state, failure)
                raise

    @functools.wraps(original_prepare_attn)
    def prepare_attn(self, input_batch):
        state = current_vllm_v1_forward_state()
        if state is None or state.reference_warmup or state.batch is not None:
            return original_prepare_attn(self, input_batch)
        from nta_runtime.connectors.vllm import NtaVllmConnectorMetadata

        metadata = getattr(state.scheduler_output, "kv_connector_metadata", None)
        if not isinstance(metadata, NtaVllmConnectorMetadata):
            raise RuntimeError(
                "NTA vLLM requires NtaVllmConnector through kv_transfer_config"
            )
        from nta_runtime.engines.vllm import _controller

        controller = _controller(self)
        metadata = metadata.aligned_to(input_batch.req_ids)
        state.input_batch = input_batch
        state.connector_metadata = metadata
        state.execution_owner = controller
        controller.begin_forward(state)
        state.batch = controller.bind_connector(metadata, input_batch)
        controller.prepare_physical_destinations()
        state.hook = controller.hook
        state.tier_service = controller.tier_service
        state.tenant_isolation_enabled = controller.tenant_isolation_enabled
        state.request_bindings_tensor = controller.request_bindings_tensor
        state.page_size = controller.page_size
        # FlashInferMetadataBuilder now sees the immutable EngineBatch and can
        # select the request-bound wrapper before vLLM performs its one native
        # plan.  Publication remains stream ordered before that plan and the
        # subsequent model launch.
        return original_prepare_attn(self, input_batch)

    @functools.wraps(original_dummy_run)
    def dummy_run(self, *args, **kwargs):
        with vllm_v1_reference_warmup_state():
            return original_dummy_run(self, *args, **kwargs)

    @functools.wraps(original_shutdown)
    def shutdown(self, *args, **kwargs):
        try:
            return original_shutdown(self, *args, **kwargs)
        finally:
            controller = getattr(self, "_nta_vllm_controller", None)
            if controller is not None:
                controller.close()

    runner_class.execute_model = execute_model
    runner_class.prepare_attn = prepare_attn
    runner_class._dummy_run = dummy_run
    runner_class.shutdown = shutdown
    runner_class._nta_bridge_patched = True


def _patch_framework_warmup() -> None:
    """Keep vLLM's synthetic kernel warmup outside the NTA contract."""
    import vllm.v1.worker.gpu_worker as gpu_worker

    if getattr(gpu_worker, "_nta_warmup_patched", False):
        return
    original_warmup = gpu_worker.warmup_kernels

    @functools.wraps(original_warmup)
    def warmup(*args, **kwargs):
        with vllm_v1_reference_warmup_state():
            return original_warmup(*args, **kwargs)

    gpu_worker.warmup_kernels = warmup
    gpu_worker._nta_warmup_patched = True


def _patch_worker() -> None:
    global _PATCHED
    if _PATCHED:
        return
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as V2ModelRunner

    _patch_v2_runner(V2ModelRunner)
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as V1ModelRunner
    except ImportError:
        V1ModelRunner = None
    if V1ModelRunner is not None:
        _patch_v1_runner(V1ModelRunner)
    _patch_framework_warmup()
    _PATCHED = True


def ensure_worker_bridge() -> None:
    """Install the vLLM worker bridge in the process owning the runner.

    The backend class can be resolved independently of the general-plugin
    bootstrap.  The native backend therefore calls this function when it is
    constructed in the worker, making the actual execution boundary explicit
    and keeping installation idempotent across vLLM's processes.
    """
    _check_version()
    _patch_worker()


def register() -> None:
    """Register the backend without importing private worker internals."""
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
    # The backend class is resolved in the worker process.  Installing the
    # private bridge here would import GPU worker internals in the frontend or
    # engine process and make registration itself a lifecycle side effect.
    # ``get_impl_cls``/``__init__`` install it at the execution edge.
