"""SGLang plugin registration for NTA's FlashInfer attention backend."""

from __future__ import annotations

import importlib.metadata


SUPPORTED_SGLANG_VERSION = "0.5.14"
BACKEND_NAME = "nta_flashinfer"
_RELEASE_TARGET = "sglang.srt.managers.scheduler.Scheduler.release_host_resources"
_HICACHE_LOAD_TARGET = (
    "sglang.srt.managers.cache_controller.HiCacheController.start_loading"
)
_ABORT_TARGET = "sglang.srt.managers.scheduler.Scheduler.abort_request"
_FORWARD_BATCH_TARGET = (
    "sglang.srt.model_executor.forward_batch_info.ForwardBatch.init_new"
)
_PREFILL_ADMISSION_TARGET = (
    "sglang.srt.managers.scheduler.Scheduler._get_new_batch_prefill_raw"
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
    from nta_runtime.engines.sglang_hicache import route_start_loading
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
    abort_hooks = HookRegistry._hooks[_ABORT_TARGET]
    if not any(hook is _cancel_backend_requests for _, hook, _ in abort_hooks):
        HookRegistry.register(_ABORT_TARGET, _cancel_backend_requests, HookType.BEFORE)
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
