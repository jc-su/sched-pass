"""SGLang plugin registration for NTA's FlashInfer attention backend."""

from __future__ import annotations

import importlib.metadata


SUPPORTED_SGLANG_VERSION = "0.5.14"
BACKEND_NAME = "nta_flashinfer"
_RELEASE_TARGET = "sglang.srt.managers.scheduler.Scheduler.release_host_resources"
_HICACHE_LOAD_TARGET = (
    "sglang.srt.managers.cache_controller.HiCacheController.start_loading"
)
_EXECUTE_EXTEND_TARGET = (
    "sglang.srt.model_executor.runner.eager_runner.EagerRunner._execute_extend"
)
_EXECUTE_DECODE_TARGET = (
    "sglang.srt.model_executor.runner.eager_runner.EagerRunner._execute_decode"
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


_OBSERVABILITY_DEGRADED: dict[str, int] = {}


def _observability_degraded(site: str, error: Exception) -> None:
    """First failure per observation site is loud; the rest are counted.

    Mechanism guards raise (CapKV stance); observation never does (eKV
    stance) — but silent degradation is worse than loud, so the count is
    exported through the engine stats as observability_degraded_<site>.
    """
    from nta_runtime.engines.sglang import FORWARD_PROFILE

    key = f"observability_degraded_{site}"
    count = _OBSERVABILITY_DEGRADED.get(site, 0)
    _OBSERVABILITY_DEGRADED[site] = count + 1
    FORWARD_PROFILE[key] = float(_OBSERVABILITY_DEGRADED[site])
    if count == 0:
        import logging

        logging.getLogger(__name__).exception(
            "observation hook %s degraded (serving continues): %r",
            site,
            error,
        )


def _profile_forward(original, runner, forward_batch, *args, **kwargs):
    """Time one forward and attribute it to a batch-composition class.

    Enabled by NTA_PROFILE_FORWARD=1. Two CUDA events per forward is
    the cheapest measurement that answers what a co-resident decode waits
    behind; the per-lease staging spans cannot answer it because a chunked
    prefill's staging is spread over several forwards.
    """
    import os

    if os.environ.get("NTA_PROFILE_FORWARD") != "1":
        return original(runner, forward_batch, *args, **kwargs)

    import torch

    from nta_runtime.engines.sglang import record_forward

    backend = getattr(runner.model_runner, "attn_backend", None)
    active_batch = getattr(backend, "_active_batch", None)
    staging = int(getattr(active_batch, "pending_host_load", None) is not None)
    mode = getattr(forward_batch, "forward_mode", None)
    is_mixed = bool(mode is not None and mode.is_mixed())
    if staging and is_mixed:
        kind = "staging_mixed"
    elif staging:
        kind = "staging_pure"
    elif is_mixed:
        kind = "mixed_nostage"
    else:
        kind = "plain"

    # An observation hook must never take down serving, and event synchronize
    # is illegal inside a captured graph. Degrade loudly once, count the
    # rest, keep serving; the counter rides the stats report.
    try:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    except Exception as error:
        _observability_degraded("forward_profile", error)
        return original(runner, forward_batch, *args, **kwargs)
    try:
        return original(runner, forward_batch, *args, **kwargs)
    finally:
        try:
            end.record()
            end.synchronize()
            record_forward(kind, start.elapsed_time(end))
        except Exception as error:
            _observability_degraded("forward_profile", error)


def _preserve_prefill_graph_request_metadata() -> None:
    """Carry request identity through the prefill BCG static batch.

    PrefillCudaGraphRunner.load_batch builds a static ForwardBatch for
    piecewise replay without copying ``rids``; the NTA backend refuses
    batches without request identity, so replays would fail closed.
    Prefill pads tokens, not requests, so identity carries one-to-one.
    """
    from sglang.srt.model_executor.runner import prefill_cuda_graph_runner

    runner_cls = prefill_cuda_graph_runner.PrefillCudaGraphRunner
    current = runner_cls.load_batch
    if getattr(current, "_nta_preserves_request_metadata", False):
        return

    def load_batch(self, forward_batch, **kwargs):
        from nta_runtime.engines.sglang import PREFILL_GRAPH_COUNTERS

        static_batch = current(self, forward_batch, **kwargs)
        static_batch.rids = getattr(forward_batch, "rids", None)
        static_batch._nta_request_slots = getattr(
            forward_batch, "_nta_request_slots", None
        )
        if static_batch._nta_request_slots is None:
            slots = getattr(forward_batch, "req_pool_indices", None)
            if slots is not None and hasattr(slots, "tolist"):
                slots = slots.tolist()
            static_batch._nta_request_slots = slots
        priorities = getattr(forward_batch, "_nta_request_priorities", None)
        if priorities is not None:
            static_batch._nta_request_priorities = priorities
        PREFILL_GRAPH_COUNTERS["prefill_graph_served_batches"] += 1
        return static_batch

    load_batch._nta_preserves_request_metadata = True
    runner_cls.load_batch = load_batch

    # Startup capture builds dummy batches with no request identity; give
    # them capture placeholders (mirroring the decode runner's padding
    # ids) so the fail-closed identity guard admits warmup and capture
    # without weakening it for real batches.
    current_prepare = runner_cls.capture_prepare
    if not getattr(current_prepare, "_nta_preserves_request_metadata", False):

        def capture_prepare(self, num_tokens):
            from nta_runtime.engines.sglang import PREFILL_GRAPH_COUNTERS

            result = current_prepare(self, num_tokens)
            PREFILL_GRAPH_COUNTERS["prefill_graph_capture_batches"] += 1
            forward_batch = result[0] if isinstance(result, tuple) else result
            if not getattr(forward_batch, "rids", None):
                batch_size = int(getattr(forward_batch, "batch_size", 1) or 1)
                forward_batch.rids = tuple(
                    f"__nta_graph_padding_{index}" for index in range(batch_size)
                )
                forward_batch._nta_request_priorities = (0,) * batch_size
            return result

        capture_prepare._nta_preserves_request_metadata = True
        runner_cls.capture_prepare = capture_prepare


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
        request_slots = getattr(forward_batch, "_nta_request_slots", None)
        if request_slots is None:
            request_slots = getattr(forward_batch, "req_pool_indices", None)
        if request_slots is not None and hasattr(request_slots, "tolist"):
            request_slots = request_slots.tolist()
        if request_slots is None or len(request_slots) != raw_bs:
            raise RuntimeError("SGLang graph replay omitted request-pool slots")
        view._nta_request_slots = tuple(int(slot) for slot in request_slots)
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
        priorities = tuple(7 - round(rank[value] * 7 / denominator) for value in raw)
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
    _preserve_prefill_graph_request_metadata()

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
    for forward_target in (_EXECUTE_EXTEND_TARGET, _EXECUTE_DECODE_TARGET):
        forward_hooks = HookRegistry._hooks[forward_target]
        if not any(hook is _profile_forward for _, hook, _ in forward_hooks):
            HookRegistry.register(forward_target, _profile_forward, HookType.AROUND)
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
