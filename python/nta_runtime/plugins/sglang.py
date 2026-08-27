"""SGLang plugin registration for NTA's FlashInfer attention backend."""

from __future__ import annotations

import importlib.metadata
import os


SUPPORTED_SGLANG_VERSION = "0.5.16"
BACKEND_NAME = "nta_flashinfer"
_RELEASE_TARGET = "sglang.srt.managers.scheduler.Scheduler.release_host_resources"
_HICACHE_LOAD_TARGET = (
    "sglang.srt.managers.cache_controller.HiCacheController.start_loading"
)
_PREFILL_REQUEST_BIND_TARGET = (
    "sglang.srt.managers.schedule_policy.PrefillAdder.add_one_req"
)
_EXECUTE_EXTEND_TARGET = (
    "sglang.srt.model_executor.runner.eager_runner.EagerRunner._execute_extend"
)
_EXECUTE_DECODE_TARGET = (
    "sglang.srt.model_executor.runner.eager_runner.EagerRunner._execute_decode"
)
_EAGER_LOAD_BATCH_TARGET = (
    "sglang.srt.model_executor.runner.eager_runner.EagerRunner.load_batch"
)
_ABORT_TARGET = "sglang.srt.managers.scheduler.Scheduler.abort_request"
_REQUEST_FINISH_TARGET = (
    "sglang.srt.managers.scheduler_components.batch_result_processor."
    "SchedulerBatchResultProcessor._handle_finish_state_updated_req"
)
_PREFILL_FINISH_TARGET = (
    "sglang.srt.managers.scheduler_components.batch_result_processor."
    "SchedulerBatchResultProcessor.process_batch_result_prefill"
)
_PREBUILT_FINISH_TARGET = (
    "sglang.srt.managers.scheduler_components.batch_result_processor."
    "SchedulerBatchResultProcessor.process_batch_result_prebuilt"
)
_FORWARD_BATCH_TARGET = (
    "sglang.srt.model_executor.forward_batch_info.ForwardBatch.init_new"
)
_PREFILL_ADMISSION_TARGET = (
    "sglang.srt.managers.scheduler.Scheduler._get_new_batch_prefill_raw"
)
_PREFILL_GRAPH_LOAD_BATCH_TARGET = (
    "sglang.srt.model_executor.runner.prefill_cuda_graph_runner."
    "PrefillCudaGraphRunner.load_batch"
)
_PREFILL_GRAPH_CAPTURE_PREPARE_TARGET = (
    "sglang.srt.model_executor.runner.prefill_cuda_graph_runner."
    "PrefillCudaGraphRunner.capture_prepare"
)
_DECODE_GRAPH_REPLAY_VIEW_TARGET = (
    "sglang.srt.model_executor.runner.decode_cuda_graph_runner.build_replay_fb_view"
)
_REQUIRED_LIFECYCLE_HOOK_TARGETS = (
    _RELEASE_TARGET,
    _HICACHE_LOAD_TARGET,
    _PREFILL_REQUEST_BIND_TARGET,
    _ABORT_TARGET,
    _REQUEST_FINISH_TARGET,
    _PREFILL_FINISH_TARGET,
    _PREBUILT_FINISH_TARGET,
    _FORWARD_BATCH_TARGET,
    _EAGER_LOAD_BATCH_TARGET,
    _PREFILL_ADMISSION_TARGET,
    _PREFILL_GRAPH_LOAD_BATCH_TARGET,
    _PREFILL_GRAPH_CAPTURE_PREPARE_TARGET,
    _DECODE_GRAPH_REPLAY_VIEW_TARGET,
)

_ACQUISITION_ATTRIBUTE = "_nta_acquisition_span"

# HookRegistry owns ordering and application.  This local set only makes a
# repeated ``register()`` call idempotent; it deliberately does not inspect or
# mutate HookRegistry's private hook list.
_REGISTERED_HOOKS: set[tuple[str, int, str]] = set()
_PROFILE_FORWARD_ENABLED: bool | None = None


_OBSERVABILITY_DEGRADED: dict[str, int] = {}


def _required_hook_targets() -> tuple[str, ...]:
    """Return hooks required by this process's enabled instrumentation.

    Forward timing is diagnostic opt-in.  It must not become a hidden
    correctness requirement when its hot-path wrappers are intentionally not
    registered.
    """
    targets = list(_REQUIRED_LIFECYCLE_HOOK_TARGETS)
    if _profile_forward_enabled():
        targets.extend((_EXECUTE_EXTEND_TARGET, _EXECUTE_DECODE_TARGET))
    return tuple(targets)


def _profile_forward_enabled() -> bool:
    """Freeze the diagnostic choice at the first plugin registration."""
    global _PROFILE_FORWARD_ENABLED
    if _PROFILE_FORWARD_ENABLED is None:
        _PROFILE_FORWARD_ENABLED = os.environ.get("NTA_PROFILE_FORWARD") == "1"
    return _PROFILE_FORWARD_ENABLED


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

    This function is registered only when ``NTA_PROFILE_FORWARD=1`` at
    process startup. Two CUDA events per forward is the cheapest measurement
    that answers what a co-resident decode waits behind; the per-lease staging
    spans cannot answer it because a chunked prefill's staging is spread over
    several forwards.
    """
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
    """Register official AROUND hooks for the prefill graph metadata views.

    PrefillCudaGraphRunner builds a static ForwardBatch without copying
    ``rids``.  The NTA backend refuses batches without request identity, so
    replay would fail closed.  These hooks are registered through SGLang's
    HookRegistry rather than replacing methods directly; that keeps plugin
    provenance, ordering, duplicate detection, and application lifecycle in
    one mechanism.
    """
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    _register_hook(
        HookRegistry,
        _PREFILL_GRAPH_LOAD_BATCH_TARGET,
        _preserve_prefill_load_batch,
        HookType.AROUND,
    )
    _register_hook(
        HookRegistry,
        _PREFILL_GRAPH_CAPTURE_PREPARE_TARGET,
        _preserve_prefill_capture_prepare,
        HookType.AROUND,
    )


def _preserve_prefill_load_batch(original, self, forward_batch, **kwargs):
    """Copy live request metadata into a static prefill graph batch."""
    from nta_runtime.engines.sglang import PREFILL_GRAPH_COUNTERS
    from nta_runtime.adapters.sglang import (
        FORWARD_METADATA_ATTRIBUTE,
        forward_metadata,
    )

    static_batch = original(self, forward_batch, **kwargs)
    static_batch.rids = getattr(forward_batch, "rids", None)
    metadata = forward_metadata(forward_batch)
    setattr(static_batch, FORWARD_METADATA_ATTRIBUTE, metadata)
    PREFILL_GRAPH_COUNTERS["prefill_graph_served_batches"] += 1
    return static_batch


def _preserve_eager_load_batch(
    original, runner, forward_batch, *args, **kwargs
):
    """Carry NTA's immutable sidecar through SGLang's eager buffer view."""

    from nta_runtime.adapters.sglang import (
        FORWARD_METADATA_ATTRIBUTE,
        forward_metadata,
    )

    metadata = forward_metadata(forward_batch)
    view = original(runner, forward_batch, *args, **kwargs)
    setattr(view, FORWARD_METADATA_ATTRIBUTE, metadata)
    raw_batch_size = getattr(forward_batch, "_nta_raw_batch_size", None)
    if raw_batch_size is not None:
        view._nta_raw_batch_size = raw_batch_size
    return view


def _preserve_prefill_capture_prepare(original, self, num_tokens, *args, **kwargs):
    """Give graph-capture-only prefill batches stable placeholder identity."""
    from nta_runtime.engines.sglang import PREFILL_GRAPH_COUNTERS
    from nta_runtime.adapters.sglang import (
        FORWARD_METADATA_ATTRIBUTE,
        SglangAcquisitionSpan,
        SglangForwardMetadata,
    )

    result = original(self, num_tokens, *args, **kwargs)
    PREFILL_GRAPH_COUNTERS["prefill_graph_capture_batches"] += 1
    forward_batch = result[0] if isinstance(result, tuple) else result
    if not getattr(forward_batch, "rids", None):
        batch_size = int(getattr(forward_batch, "batch_size", 1) or 1)
        forward_batch.rids = tuple(
            f"__nta_graph_padding_{index}" for index in range(batch_size)
        )
        forward_batch_metadata = SglangForwardMetadata(
            tuple(range(batch_size)),
            (0,) * batch_size,
            (0,) * batch_size,
            (SglangAcquisitionSpan.direct(),) * batch_size,
        )
        setattr(
            forward_batch,
            FORWARD_METADATA_ATTRIBUTE,
            forward_batch_metadata,
        )
    return result


def _preserve_graph_request_metadata() -> None:
    """Register the decode graph replay metadata hook with SGLang."""
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    _register_hook(
        HookRegistry,
        _DECODE_GRAPH_REPLAY_VIEW_TARGET,
        _preserve_decode_replay_view,
        HookType.AROUND,
    )


def _preserve_decode_replay_view(original, *args, **kwargs):
    """Carry live identity and tenant metadata through padded replay views."""
    view = original(*args, **kwargs)
    from nta_runtime.adapters.sglang import (
        FORWARD_METADATA_ATTRIBUTE,
        forward_metadata,
    )

    forward_batch = kwargs.get("forward_batch", args[0] if args else None)
    raw_bs = int(kwargs.get("raw_bs", args[3] if len(args) > 3 else 0))
    padded_bs = int(kwargs.get("bs", args[2] if len(args) > 2 else raw_bs))
    request_ids = list(getattr(forward_batch, "rids", ()) or ())
    if len(request_ids) != raw_bs:
        raise RuntimeError("SGLang graph replay omitted live request IDs")
    request_ids.extend(
        f"__nta_graph_padding_{index}" for index in range(raw_bs, padded_bs)
    )
    metadata = forward_metadata(forward_batch)
    if len(metadata.request_slots) != raw_bs:
        raise RuntimeError("SGLang graph replay omitted request-pool slots")
    padded_slots = getattr(view, "req_pool_indices", None)
    if padded_slots is None:
        raise RuntimeError("SGLang graph replay omitted padded request-pool slots")
    padded_metadata = metadata.pad(padded_slots)
    if len(padded_metadata.request_slots) != padded_bs:
        raise RuntimeError("SGLang graph replay padded slot count disagrees with bs")
    view.rids = request_ids
    setattr(view, FORWARD_METADATA_ATTRIBUTE, padded_metadata)
    view._nta_raw_batch_size = raw_bs
    return view


def _register_hook(registry, target, hook, hook_type) -> None:
    """Register one hook idempotently through SGLang's public API."""
    patched = getattr(registry, "_patched", None)
    if not isinstance(patched, set):
        raise RuntimeError(
            "unsupported SGLang HookRegistry: missing its pinned applied-target state"
        )
    if target in patched:
        raise RuntimeError(
            f"cannot register NTA hook after SGLang already applied {target}"
        )
    key = (target, id(hook), hook_type.value)
    if key in _REGISTERED_HOOKS:
        return
    registry.register(target, hook, hook_type)
    _REGISTERED_HOOKS.add(key)


def _require_hooks_installed(registry) -> None:
    """Reject construction after SGLang skipped a required hook.

    SGLang 0.5.16 exposes registration publicly but has no public query for
    whether ``apply_hooks`` succeeded; its loader also logs-and-continues on
    application errors.  The pinned adapter therefore reads only the
    version-qualified applied-target set here, and fails closed if that
    private compatibility seam changes.
    """
    patched = getattr(registry, "_patched", None)
    if not isinstance(patched, set):
        raise RuntimeError(
            "unsupported SGLang HookRegistry: cannot verify applied hooks"
        )
    missing = tuple(
        target for target in _required_hook_targets() if target not in patched
    )
    if missing:
        raise RuntimeError(
            "NTA SGLang plugin did not install required lifecycle hooks: "
            + ", ".join(missing)
        )


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
    # This hook runs after SGLang has updated the request's finish state and
    # immediately before it releases the request-pool slot.  Close NTA's
    # generation at that same lifecycle boundary.  Merely observing finish here
    # leaves the old request active in the identity registry; a later request
    # with the same rid in a recycled SGLang slot would then be rejected as a
    # simultaneous binding even though the generations do not overlap.
    for backend in _walk_attention_backends(processor):
        retire = getattr(backend, "retire_request", None)
        if callable(retire):
            retire(request_id)


def _retire_prefill_finished_requests(
    result, processor, batch, *args, **kwargs
) -> None:
    """Close requests that SGLang finishes directly in prefill.

    ``max_new_tokens == 1`` never enters the decode helper hooked above:
    SGLang updates its finish state and releases KV directly inside
    ``process_batch_result_prefill``.  Run after that method so every branch
    (generation, embedding, and mixed prefill) has a single common retirement
    test.  The registry operation is idempotent for resident-only requests that
    never entered NTA's acquisition path.
    """
    del result, args, kwargs
    for req in tuple(getattr(batch, "reqs", ())):
        if req.finished():
            _retire_finished_request(processor, req)


def _capture_prefill_request_binding(original, adder, request, *args, **kwargs):
    """Capture SGLang's exact request→load-operation ownership edge."""

    from nta_runtime.adapters.sglang import SglangAcquisitionSpan

    request_id = str(getattr(request, "rid", "") or "")
    if not request_id:
        raise RuntimeError("SGLang prefill request omitted its request identity")
    needs_load = bool(request.needs_host_load_back())
    expected_node_id = int(getattr(request.best_match_node, "id", -1))
    prefix_begin = len(request.prefix_indices)
    tree_cache = getattr(adder, "tree_cache", None)
    controller = getattr(tree_cache, "cache_controller", None)
    load_queue = getattr(controller, "load_queue", None)
    if needs_load and load_queue is None:
        raise RuntimeError("SGLang host-load request has no cache-controller queue")
    queued_operation_ids = (
        set()
        if load_queue is None
        else {int(getattr(operation, "id", -1)) for operation in load_queue}
    )
    if -1 in queued_operation_ids:
        raise RuntimeError("SGLang load queue contains an untyped operation")
    result = original(adder, request, *args, **kwargs)
    loaded_rows = len(request.prefix_indices) - prefix_begin
    if loaded_rows < 0:
        raise RuntimeError("SGLang host loading shrank the request prefix")
    loaded_node_id = int(getattr(request.last_node, "id", -1))
    if loaded_rows:
        if not needs_load or expected_node_id < 0 or loaded_node_id != expected_node_id:
            raise RuntimeError("SGLang loaded rows without a stable radix identity")
        new_operations = tuple(
            operation
            for operation in load_queue
            if int(getattr(operation, "id", -1)) not in queued_operation_ids
        )
        if len(new_operations) != 1:
            raise RuntimeError(
                "one SGLang request must create exactly one unmerged load operation"
            )
        operation = new_operations[0]
        operation_id = int(getattr(operation, "id", -1))
        operation_nodes = tuple(int(node) for node in operation.node_ids)
        operation_rows = int(operation.device_indices.numel())
        if operation_id < 0 or operation_nodes != (expected_node_id,):
            raise RuntimeError("SGLang load operation identity disagrees with request")
        if operation_rows != loaded_rows:
            raise RuntimeError("SGLang load operation rows disagree with request span")
        acquisition = SglangAcquisitionSpan(
            operation_id,
            expected_node_id,
            prefix_begin,
            loaded_rows,
        )
    else:
        previous = getattr(request, _ACQUISITION_ATTRIBUTE, None)
        if previous is not None and not isinstance(previous, SglangAcquisitionSpan):
            raise RuntimeError("SGLang request carries malformed acquisition metadata")
        # A request may be revisited after add_one_req loaded its rows but did
        # not admit it. Preserve that one-shot edge until ForwardBatch.init_new
        # consumes it; otherwise this is a resident request.
        acquisition = (
            previous
            if isinstance(previous, SglangAcquisitionSpan) and previous.is_external
            else SglangAcquisitionSpan.direct()
        )
    setattr(request, _ACQUISITION_ATTRIBUTE, acquisition)
    return result


def _attach_request_priorities(
    forward_batch, cls, batch, model_runner, *hook_args, **hook_kwargs
):
    # SGLang extends ForwardBatch.init_new's hook signature across releases
    # (for example, capture_hidden_mode was added after the plugin was first
    # integrated).  The sidecar only depends on the stable batch and runner
    # objects; accepting and deliberately ignoring extension arguments keeps
    # the integration version-tolerant without hiding required state.
    del cls, hook_args, hook_kwargs
    from nta_runtime.adapters.base import _integer_vector
    from nta_runtime.adapters.sglang import (
        FORWARD_METADATA_ATTRIBUTE,
        SglangAcquisitionSpan,
        SglangForwardMetadata,
    )

    requests = tuple(getattr(batch, "reqs", ()))
    request_ids = tuple(getattr(forward_batch, "rids", ()) or ())
    if len(requests) != len(request_ids):
        raise RuntimeError("SGLang request policy and forward batch disagree")
    server_args = model_runner.server_args
    if not bool(getattr(server_args, "enable_priority_scheduling", False)):
        priorities = (0,) * len(requests)
    else:
        try:
            raw = _integer_vector(
                tuple(
                    0 if (value := getattr(request, "priority", 0)) is None else value
                    for request in requests
                ),
                "SGLang scheduler priorities",
                minimum=-(1 << 31),
                maximum=(1 << 31) - 1,
            )
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        low_first = bool(
            getattr(server_args, "schedule_low_priority_values_first", False)
        )
        ordered = sorted(set(raw), reverse=not low_first)
        rank = {value: index for index, value in enumerate(ordered)}
        denominator = max(1, len(ordered) - 1)
        priorities = tuple(7 - round(rank[value] * 7 / denominator) for value in raw)
    request_slots = getattr(forward_batch, "req_pool_indices", None)
    if request_slots is None:
        raise RuntimeError("SGLang forward batch omitted request-pool slots")
    if hasattr(request_slots, "tolist"):
        request_slots = request_slots.tolist()
    existing = getattr(forward_batch, FORWARD_METADATA_ATTRIBUTE, None)
    if existing is not None and not isinstance(existing, SglangForwardMetadata):
        raise RuntimeError("SGLang forward metadata has an invalid sidecar type")
    tenant_ids = (
        (0,) * len(requests)
        if existing is None
        else tuple(int(tenant_id) for tenant_id in existing.tenant_ids)
    )
    acquisitions: list[SglangAcquisitionSpan] = []
    for request in requests:
        acquisition = getattr(
            request, _ACQUISITION_ATTRIBUTE, SglangAcquisitionSpan.direct()
        )
        if not isinstance(acquisition, SglangAcquisitionSpan):
            raise RuntimeError("SGLang request carries malformed acquisition metadata")
        acquisitions.append(acquisition)
    for request in requests:
        # The sidecar now owns this forward's immutable copy. Clearing the
        # mutable request attribute prevents a later decode or unrelated load
        # lease from reusing stale acquisition ownership.
        setattr(request, _ACQUISITION_ATTRIBUTE, SglangAcquisitionSpan.direct())
    setattr(
        forward_batch,
        FORWARD_METADATA_ATTRIBUTE,
        SglangForwardMetadata(
            tuple(int(slot) for slot in request_slots),
            priorities,
            tenant_ids,
            tuple(acquisitions),
        ),
    )


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
    _register_hook(HookRegistry, _RELEASE_TARGET, _flush_backend_stats, HookType.BEFORE)
    _register_hook(
        HookRegistry, _HICACHE_LOAD_TARGET, route_start_loading, HookType.AROUND
    )
    _register_hook(
        HookRegistry,
        _PREFILL_REQUEST_BIND_TARGET,
        _capture_prefill_request_binding,
        HookType.AROUND,
    )
    # Forward timing is diagnostic instrumentation, not part of the serving
    # contract.  Do not leave an AROUND wrapper on the hottest framework
    # methods when profiling is disabled: even a fast wrapper adds a Python
    # call and an environment lookup to every prefill/decode step.  The
    # process-start setting is intentional; changing it after plugin loading
    # cannot safely remove an already-applied monkey patch.
    if _profile_forward_enabled():
        for forward_target in (_EXECUTE_EXTEND_TARGET, _EXECUTE_DECODE_TARGET):
            _register_hook(
                HookRegistry, forward_target, _profile_forward, HookType.AROUND
            )
    _register_hook(
        HookRegistry, _ABORT_TARGET, _cancel_backend_requests, HookType.BEFORE
    )
    _register_hook(
        HookRegistry, _REQUEST_FINISH_TARGET, _retire_finished_request, HookType.BEFORE
    )
    _register_hook(
        HookRegistry,
        _PREFILL_FINISH_TARGET,
        _retire_prefill_finished_requests,
        HookType.AFTER,
    )
    _register_hook(
        HookRegistry,
        _PREBUILT_FINISH_TARGET,
        _retire_prefill_finished_requests,
        HookType.AFTER,
    )
    _register_hook(
        HookRegistry, _FORWARD_BATCH_TARGET, _attach_request_priorities, HookType.AFTER
    )
    _register_hook(
        HookRegistry,
        _EAGER_LOAD_BATCH_TARGET,
        _preserve_eager_load_batch,
        HookType.AROUND,
    )
    _register_hook(
        HookRegistry,
        _PREFILL_ADMISSION_TARGET,
        route_prefill_admission,
        HookType.AROUND,
    )
    if BACKEND_NAME not in ATTENTION_BACKENDS:

        @register_attention_backend(BACKEND_NAME)
        def create_backend(model_runner):
            _require_hooks_installed(HookRegistry)
            if model_runner.use_mla_backend:
                raise ValueError("NTA's SGLang backend does not support MLA models")
            from nta_runtime.engines.sglang import NtaFlashInferAttnBackend

            return NtaFlashInferAttnBackend(
                model_runner, init_new_workspace=model_runner.init_new_workspace
            )
