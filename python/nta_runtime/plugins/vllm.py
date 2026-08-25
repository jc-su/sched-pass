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

import numpy as np

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


def _patch_v2_block_tables() -> None:
    """Keep a CPU ownership mirror for vLLM V2 block allocation writes."""
    from vllm.v1.worker.gpu.block_table import BlockTables

    if getattr(BlockTables, "_nta_bridge_patched", False):
        return
    original_init = BlockTables.__init__
    original_append = BlockTables.append_block_ids

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._nta_cpu_block_tables = [
            np.zeros(tuple(table.gpu.shape), dtype=np.int32)
            for table in self.block_tables
        ]
        self._nta_cpu_num_blocks = np.zeros(
            (self.num_kv_cache_groups, self.max_num_reqs), dtype=np.int32
        )

    @functools.wraps(original_append)
    def append(self, req_index, new_block_ids, overwrite):
        result = original_append(self, req_index, new_block_ids, overwrite)
        tables = self._nta_cpu_block_tables
        counts = self._nta_cpu_num_blocks
        for group_id, group_block_ids in enumerate(new_block_ids):
            expanded = list(group_block_ids)
            blocks_per_kv_block = self.blocks_per_kv_block[group_id]
            if blocks_per_kv_block > 1:
                expanded = [
                    block * blocks_per_kv_block + offset
                    for block in expanded
                    for offset in range(blocks_per_kv_block)
                ]
            start = 0 if overwrite else int(counts[group_id, req_index])
            if overwrite:
                tables[group_id][req_index, :].fill(0)
            end = start + len(expanded)
            if end > tables[group_id].shape[1]:
                raise RuntimeError("vLLM V2 CPU block-table mirror capacity exhausted")
            tables[group_id][req_index, start:end] = expanded
            counts[group_id, req_index] = end
        return result

    BlockTables.__init__ = init
    BlockTables.append_block_ids = append
    BlockTables._nta_bridge_patched = True


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
        state.batch = controller.bind(scheduler_output)
        state.hook = controller.hook
        state.tier_service = controller.tier_service
        state.page_size = controller.page_size
        return result

    @functools.wraps(original_execute)
    def execute_model(self, scheduler_output, intermediate_tensors=None):
        state = current_vllm_v1_forward_state()
        if state is not None and state.reference_warmup:
            return original_execute(self, scheduler_output, intermediate_tensors)
        with vllm_v1_forward_state(scheduler_output):
            return original_execute(self, scheduler_output, intermediate_tensors)

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
        with vllm_v1_forward_state(scheduler_output):
            return original_execute(
                self,
                scheduler_output,
                intermediate_tensors,
                dummy_run,
                skip_attn_for_dummy_run,
                is_profile,
            )

    @functools.wraps(original_prepare_attn)
    def prepare_attn(self, input_batch):
        result = original_prepare_attn(self, input_batch)
        state = current_vllm_v1_forward_state()
        if state is None or state.reference_warmup or state.batch is not None:
            return result
        tables = getattr(self.block_tables, "_nta_cpu_block_tables", None)
        counts = getattr(self.block_tables, "_nta_cpu_num_blocks", None)
        if tables is None or counts is None:
            raise RuntimeError("vLLM V2 NTA block-table mirror is not initialized")
        from nta_runtime.engines.vllm import _controller

        controller = _controller(self)
        state.input_batch = input_batch
        state.batch = controller.bind_v2(
            state.scheduler_output,
            input_batch,
            block_tables=tables,
            num_blocks=counts,
        )
        state.hook = controller.hook
        state.tier_service = controller.tier_service
        state.page_size = controller.page_size
        return result

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
    _patch_v2_block_tables()
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
