"""SGLang plugin registration for NTA's FlashInfer attention backend."""

from __future__ import annotations

import importlib.metadata


SUPPORTED_SGLANG_VERSION = "0.5.14"
BACKEND_NAME = "nta_flashinfer"
_RELEASE_TARGET = "sglang.srt.managers.scheduler.Scheduler.release_host_resources"
_HICACHE_LOAD_TARGET = (
    "sglang.srt.managers.cache_controller.HiCacheController.start_loading"
)
_WRITE_BACKUP_TARGETS = (
    "sglang.srt.mem_cache.hiradix_cache.HiRadixCache.write_backup",
    "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache.write_backup",
)
_ABORT_TARGET = "sglang.srt.managers.scheduler.Scheduler.abort_request"
_REQUEST_FINISH_TARGET = (
    "sglang.srt.managers.scheduler_components.batch_result_processor."
    "SchedulerBatchResultProcessor._handle_finish_state_updated_req"
)
_FORWARD_BATCH_TARGET = (
    "sglang.srt.model_executor.forward_batch_info.ForwardBatch.init_new"
)
_PREFILL_ADMISSION_TARGET = (
    "sglang.srt.managers.scheduler.Scheduler._get_new_batch_prefill_raw"
)
_EXTERNAL_ADMISSION_TARGET = (
    "sglang.srt.managers.schedule_policy.PrefillAdder.add_one_req"
)
_CUDA_GRAPH_ELIGIBILITY_TARGET = (
    "sglang.srt.model_executor.runner.decode_cuda_graph_runner."
    "DecodeCudaGraphRunner.can_run_graph"
)
_EXTERNAL_PREFIX_TARGETS = (
    "sglang.srt.mem_cache.hiradix_cache.HiRadixCache.init_load_back",
    "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache.init_load_back",
)
_CACHE_UNFINISHED_TARGETS = (
    "sglang.srt.mem_cache.hiradix_cache.HiRadixCache.cache_unfinished_req",
    "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache.cache_unfinished_req",
)
_CACHE_FINISHED_TARGETS = (
    "sglang.srt.mem_cache.hiradix_cache.HiRadixCache.cache_finished_req",
    "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache.cache_finished_req",
)
_ALLOCATOR_FREE_TARGETS = (
    "sglang.srt.mem_cache.allocator.token.TokenToKVPoolAllocator.free",
    "sglang.srt.mem_cache.allocator.paged.PagedTokenToKVPoolAllocator.free",
)


def _preserve_graph_request_metadata() -> None:
    """Carry host request identity through SGLang's padded replay view."""
    from sglang.srt.model_executor.runner import decode_cuda_graph_runner

    current = decode_cuda_graph_runner.build_replay_fb_view
    if getattr(current, "_nta_preserves_request_metadata", False):
        return

    def build_replay_fb_view(*args, **kwargs):
        view = current(*args, **kwargs)
        forward_batch = kwargs.get("forward_batch", args[0] if args else None)
        raw_bs = int(kwargs.get("raw_bs", args[3] if len(args) > 3 else 0))
        padded_bs = int(kwargs.get("bs", args[2] if len(args) > 2 else raw_bs))
        request_ids = list(getattr(forward_batch, "rids", ()) or ())
        if len(request_ids) != raw_bs:
            raise RuntimeError("SGLang graph replay omitted live request IDs")
        request_ids.extend(
            f"__nta_graph_padding_{index}" for index in range(raw_bs, padded_bs)
        )
        priorities = list(
            getattr(forward_batch, "_nta_request_priorities", (0,) * raw_bs)
        )
        if len(priorities) != raw_bs:
            raise RuntimeError("SGLang graph replay omitted request priorities")
        priorities.extend(0 for _ in range(raw_bs, padded_bs))
        view.rids = request_ids
        view._nta_request_priorities = tuple(priorities)
        view._nta_raw_batch_size = raw_bs
        return view

    build_replay_fb_view._nta_preserves_request_metadata = True
    decode_cuda_graph_runner.build_replay_fb_view = build_replay_fb_view


def _walk_attention_backends(scheduler):
    worker = getattr(scheduler, "tp_worker", None)
    if worker is None:
        worker = getattr(scheduler, "model_worker", None)
    runner = getattr(worker, "model_runner", None)
    pending = [getattr(runner, "attn_backend", None)]
    visited: set[int] = set()
    while pending:
        backend = pending.pop()
        if backend is None or id(backend) in visited:
            continue
        visited.add(id(backend))
        yield backend
        pending.extend(
            getattr(backend, name, None)
            for name in ("prefill_backend", "decode_backend", "full_attn_backend")
        )


def _flush_backend_stats(scheduler, *args, **kwargs) -> None:
    del args, kwargs
    for backend in _walk_attention_backends(scheduler):
        close = getattr(backend, "close", None)
        if callable(close):
            close()
        writer = getattr(backend, "_write_stats", None)
        if callable(writer):
            writer()


def _cancel_backend_requests(scheduler, recv_req, *args, **kwargs) -> None:
    del args, kwargs
    from nta_runtime.engines.sglang_admission import cancel_staged_batch

    cancel_staged_batch(scheduler, recv_req)
    request_id = getattr(recv_req, "rid", "") or ""
    abort_all = bool(getattr(recv_req, "abort_all", False))
    for backend in _walk_attention_backends(scheduler):
        cancel = getattr(backend, "cancel_requests", None)
        if callable(cancel):
            cancel(request_id, all=abort_all)


def _retire_finished_request(processor, req, *args, **kwargs) -> None:
    del args, kwargs
    if not req.finished():
        return
    request_id = getattr(req, "rid", "") or ""
    if not request_id:
        raise RuntimeError("finished SGLang request omitted its request ID")
    handle = getattr(req, "_nta_external_prefix", None)
    if handle is not None:
        if not handle._released and not handle.retire("finished"):
            raise RuntimeError("finished external prefix lost its runtime claim")
        return
    for backend in _walk_attention_backends(processor):
        finish = getattr(backend, "finish_requests", None)
        if callable(finish):
            finish((request_id,))


def _attach_request_priorities(forward_batch, cls, batch, model_runner):
    del cls
    requests = tuple(getattr(batch, "reqs", ()))
    request_ids = tuple(getattr(forward_batch, "rids", ()) or ())
    if len(requests) != len(request_ids):
        raise RuntimeError("SGLang request policy and forward batch disagree")
    server_args = model_runner.server_args
    if not bool(getattr(server_args, "enable_priority_scheduling", False)):
        priorities = (0,) * len(requests)
    else:
        raw = tuple(int(getattr(request, "priority", 0) or 0) for request in requests)
        low_first = bool(
            getattr(server_args, "schedule_low_priority_values_first", False)
        )
        ordered = sorted(set(raw), reverse=not low_first)
        rank = {value: index for index, value in enumerate(ordered)}
        denominator = max(1, len(ordered) - 1)
        priorities = tuple(
            7 - round(rank[value] * 7 / denominator) for value in raw
        )
    forward_batch._nta_request_priorities = priorities


def _route_cuda_graph_eligibility(original, runner, forward_batch):
    request_ids = tuple(str(value) for value in getattr(forward_batch, "rids", ()))
    pending = [getattr(runner, "attn_backend", None)]
    pending.extend(getattr(runner.model_runner, "decode_attn_backend_group", ()) or ())
    visited: set[int] = set()
    for backend in pending:
        if backend is None or id(backend) in visited:
            continue
        visited.add(id(backend))
        requires_eager = getattr(backend, "requires_eager_requests", None)
        if callable(requires_eager) and requires_eager(request_ids):
            epoch_ready = getattr(backend, "tiered_graph_epoch_ready", None)
            if callable(epoch_ready) and epoch_ready(request_ids):
                backend._stats["tiered_graph_epoch_batches"] = (
                    backend._stats.get("tiered_graph_epoch_batches", 0) + 1
                )
                continue
            backend._stats["tiered_graph_eager_batches"] = (
                backend._stats.get("tiered_graph_eager_batches", 0) + 1
            )
            return False
    return original(runner, forward_batch)


def register() -> None:
    version = importlib.metadata.version("sglang")
    if version != SUPPORTED_SGLANG_VERSION:
        raise RuntimeError(
            f"NTA supports SGLang {SUPPORTED_SGLANG_VERSION}; found {version}"
        )

    from sglang.srt.layers.attention.attention_registry import (
        ATTENTION_BACKENDS,
        register_attention_backend,
    )
    from sglang.srt.server_args import (
        ATTENTION_BACKEND_CHOICES,
        add_attention_backend_choices,
    )
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from nta_runtime.engines.sglang_hicache import (
        route_start_loading,
        route_write_backup,
    )
    from nta_runtime.engines.sglang_external import (
        route_allocator_free,
        route_cache_finished,
        route_cache_unfinished,
        route_external_admission_credit,
        route_init_load_back,
    )
    from nta_runtime.engines.sglang_admission import route_prefill_admission

    _preserve_graph_request_metadata()

    if BACKEND_NAME not in ATTENTION_BACKEND_CHOICES:
        add_attention_backend_choices([BACKEND_NAME])
    hooks = HookRegistry._hooks[_RELEASE_TARGET]
    if not any(hook is _flush_backend_stats for _, hook, _ in hooks):
        HookRegistry.register(_RELEASE_TARGET, _flush_backend_stats, HookType.BEFORE)
    hicache_hooks = HookRegistry._hooks[_HICACHE_LOAD_TARGET]
    if not any(hook is route_start_loading for _, hook, _ in hicache_hooks):
        HookRegistry.register(
            _HICACHE_LOAD_TARGET, route_start_loading, HookType.AROUND
        )
    for write_target in _WRITE_BACKUP_TARGETS:
        write_hooks = HookRegistry._hooks[write_target]
        if not any(hook is route_write_backup for _, hook, _ in write_hooks):
            HookRegistry.register(
                write_target, route_write_backup, HookType.AROUND
            )
    abort_hooks = HookRegistry._hooks[_ABORT_TARGET]
    if not any(hook is _cancel_backend_requests for _, hook, _ in abort_hooks):
        HookRegistry.register(_ABORT_TARGET, _cancel_backend_requests, HookType.BEFORE)
    finish_hooks = HookRegistry._hooks[_REQUEST_FINISH_TARGET]
    if not any(hook is _retire_finished_request for _, hook, _ in finish_hooks):
        HookRegistry.register(
            _REQUEST_FINISH_TARGET, _retire_finished_request, HookType.BEFORE
        )
    forward_hooks = HookRegistry._hooks[_FORWARD_BATCH_TARGET]
    if not any(hook is _attach_request_priorities for _, hook, _ in forward_hooks):
        HookRegistry.register(
            _FORWARD_BATCH_TARGET, _attach_request_priorities, HookType.AFTER
        )
    admission_hooks = HookRegistry._hooks[_PREFILL_ADMISSION_TARGET]
    if not any(hook is route_prefill_admission for _, hook, _ in admission_hooks):
        HookRegistry.register(
            _PREFILL_ADMISSION_TARGET,
            route_prefill_admission,
            HookType.AROUND,
        )
    graph_hooks = HookRegistry._hooks[_CUDA_GRAPH_ELIGIBILITY_TARGET]
    if not any(hook is _route_cuda_graph_eligibility for _, hook, _ in graph_hooks):
        HookRegistry.register(
            _CUDA_GRAPH_ELIGIBILITY_TARGET,
            _route_cuda_graph_eligibility,
            HookType.AROUND,
        )
    external_admission_hooks = HookRegistry._hooks[_EXTERNAL_ADMISSION_TARGET]
    if not any(
        hook is route_external_admission_credit
        for _, hook, _ in external_admission_hooks
    ):
        HookRegistry.register(
            _EXTERNAL_ADMISSION_TARGET,
            route_external_admission_credit,
            HookType.AROUND,
        )
    for target in _EXTERNAL_PREFIX_TARGETS:
        hooks = HookRegistry._hooks[target]
        if not any(hook is route_init_load_back for _, hook, _ in hooks):
            HookRegistry.register(
                target, route_init_load_back, HookType.AROUND
            )
    for target in _CACHE_UNFINISHED_TARGETS:
        hooks = HookRegistry._hooks[target]
        if not any(hook is route_cache_unfinished for _, hook, _ in hooks):
            HookRegistry.register(
                target, route_cache_unfinished, HookType.AROUND
            )
    for target in _CACHE_FINISHED_TARGETS:
        hooks = HookRegistry._hooks[target]
        if not any(hook is route_cache_finished for _, hook, _ in hooks):
            HookRegistry.register(
                target, route_cache_finished, HookType.AROUND
            )
    for target in _ALLOCATOR_FREE_TARGETS:
        hooks = HookRegistry._hooks[target]
        if not any(hook is route_allocator_free for _, hook, _ in hooks):
            HookRegistry.register(
                target, route_allocator_free, HookType.AROUND
            )
    if BACKEND_NAME in ATTENTION_BACKENDS:
        return

    @register_attention_backend(BACKEND_NAME)
    def create_backend(model_runner):
        if model_runner.use_mla_backend:
            raise ValueError("NTA's SGLang backend does not support MLA models")
        from nta_runtime.engines.sglang import NtaFlashInferAttnBackend

        return NtaFlashInferAttnBackend(
            model_runner, init_new_workspace=model_runner.init_new_workspace
        )
