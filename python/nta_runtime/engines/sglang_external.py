"""SGLang external-prefix ownership before dense device-slot allocation."""

from __future__ import annotations

import os
import time

from dataclasses import dataclass
from typing import Any, Callable

import torch

from nta_runtime.virtual_namespace import (
    VIRTUAL_TOKEN_BASE,
    VIRTUAL_TOKEN_LIMIT,
)


@dataclass
class ExternalPrefixHandle:
    """One request-owned host prefix and its bounded physical staging rows."""

    claim_id: int
    request_id: str
    consumer_index: int
    host_indices: torch.Tensor
    device_indices: torch.Tensor
    staging_rows: torch.Tensor
    controller: Any
    node_ids: tuple[int, ...]
    resident_prefix_len: int
    source_release: Callable[[], None]
    registry_release: Callable[["ExternalPrefixHandle"], None]
    retire_callback: Callable[[str], bool] | None = None
    external_sidecar: bool = True
    _released: bool = False

    @property
    def virtual_begin(self) -> int:
        return int(self.device_indices[0])

    def retire(self, reason: str) -> bool:
        if self._released:
            return False
        if not callable(self.retire_callback):
            raise RuntimeError("external prefix lost its runtime retirement edge")
        return self.retire_callback(reason)

    def release_resources(self) -> None:
        """Release staging and host ownership after the GPU completion fence."""
        if self._released:
            raise RuntimeError("external prefix resources were released twice")
        self._released = True
        self.retire_callback = None
        allocator = getattr(
            self.controller.mem_pool_device_allocator,
            "full_attn_allocator",
            self.controller.mem_pool_device_allocator,
        )
        allocator.free(self.staging_rows)
        self.source_release()
        self.registry_release(self)


def _controller(cache: Any) -> Any | None:
    return getattr(cache, "cache_controller", None)


def _bridge(cache: Any) -> Any | None:
    controller = _controller(cache)
    if controller is None:
        return None
    from nta_runtime.engines.sglang_hicache import find_bridge

    return find_bridge(controller.mem_pool_device)


def _external_cache(cache: Any) -> tuple[Any, Any] | tuple[None, None]:
    """Find the enabled concrete cache below optional session wrappers."""
    current = cache
    for _ in range(4):
        bridge = _bridge(current)
        if bridge is not None and bridge.external_prefix_enabled:
            return current, bridge
        current = getattr(current, "inner", None)
        if current is None:
            break
    return None, None


def _validate_external_cache(cache: Any, request: Any) -> None:
    controller = _controller(cache)
    needs_host_load = getattr(request, "needs_host_load_back", None)
    if controller is None or not callable(needs_host_load) or not needs_host_load():
        raise RuntimeError("external-prefix request omitted a host load")
    if not bool(getattr(cache, "page_size", 0) == 1):
        raise RuntimeError("external-prefix ownership requires page size one")
    if getattr(request, "swa_host_hit_length", 0) or getattr(
        request, "mamba_host_hit_length", 0
    ):
        raise RuntimeError("external-prefix ownership supports full attention only")
    components = getattr(cache, "_components_tuple", None)
    if components is not None:
        from sglang.srt.mem_cache.unified_cache_components import BASE_COMPONENT_TYPE

        if len(components) != 1 or components[0].component_type != BASE_COMPONENT_TYPE:
            raise RuntimeError(
                "external-prefix ownership requires a full-attention-only "
                "unified cache"
            )


def _legacy_host_rows(cache: Any, best_match_node: Any) -> tuple[Any, Callable[[], None]]:
    nodes = []
    node = best_match_node
    while bool(getattr(node, "evicted", False)):
        host_value = getattr(node, "host_value", None)
        if host_value is None:
            raise RuntimeError("evicted SGLang node omitted host KV indices")
        nodes.insert(0, node)
        node = node.parent
    if not nodes:
        raise RuntimeError("external-prefix hook received no evicted host nodes")
    best_match_node.protect_host()

    def release() -> None:
        best_match_node.release_host()

    return torch.cat([entry.host_value for entry in nodes]), release


def _unified_host_rows(cache: Any, best_match_node: Any) -> tuple[Any, Callable[[], None]]:
    from sglang.srt.mem_cache.unified_cache_components import (
        BASE_COMPONENT_TYPE,
        CacheTransferPhase,
    )

    transfers = cache.components[BASE_COMPONENT_TYPE].build_hicache_transfers(
        best_match_node, CacheTransferPhase.LOAD_BACK
    )
    if not transfers or len(transfers) != 1:
        raise RuntimeError("unified SGLang prefix produced no unique KV transfer")
    transfer = transfers[0]
    host_indices = transfer.host_indices
    if host_indices is None or host_indices.numel() == 0:
        raise RuntimeError("unified SGLang prefix omitted host KV indices")
    lock_params = cache.inc_host_lock_ref(best_match_node).to_dec_params()

    def release() -> None:
        cache.dec_host_lock_ref(best_match_node, lock_params)

    return host_indices, release


def route_init_load_back(
    original: Any, cache: Any, params: Any, *args: Any, **kwargs: Any
) -> Any:
    """Publish an external handle instead of promoting every host token."""
    bridge = _bridge(cache)
    if bridge is None or not bridge.external_prefix_enabled:
        return original(cache, params, *args, **kwargs)
    if args or kwargs:
        raise RuntimeError("unsupported SGLang init_load_back calling convention")
    request = getattr(params, "req", None)
    if request is None:
        raise RuntimeError("external-prefix load omitted request identity")
    _validate_external_cache(cache, request)
    if getattr(request, "_nta_external_prefix", None) is not None:
        raise RuntimeError("request already owns an external prefix")
    best_match_node = params.best_match_node
    if hasattr(best_match_node, "component_data"):
        host_indices, source_release = _unified_host_rows(cache, best_match_node)
    else:
        host_indices, source_release = _legacy_host_rows(cache, best_match_node)
    expected = int(params.host_hit_length)
    configured = os.environ.get("NTA_SGLANG_EXTERNAL_MIN_TOKENS")
    minimum = (
        int(configured)
        if configured
        else int(getattr(bridge, "_external_prefix_page_tokens", 1) or 1)
    )
    if expected < max(1, minimum):
        # Claims exist to avoid promoting large prefixes; a sub-page host
        # hit costs nothing to serve densely, and claiming it attaches
        # external-prefix semantics — including suppressed radix
        # insertion — to requests that are not external at all. Observed
        # failure: churn evicts the tree, every later request gets a
        # trivial one-token host hit, the sidecar claims it, and radix
        # caching silently dies for the whole workload.
        source_release()
        return original(cache, params)
    if expected <= 0 or int(host_indices.numel()) != expected:
        source_release()
        raise RuntimeError(
            "external-prefix host rows disagree with SGLang's logical hit length"
        )
    try:
        handle = bridge.claim_external_prefix(
            request,
            host_indices,
            source_release,
            _controller(cache),
            cache,
            node_ids=(int(getattr(best_match_node, "id", -1)),),
        )
    except Exception:
        source_release()
        raise
    request._nta_external_prefix = handle
    return handle.device_indices, request.last_node


def _isolation_defer_ns() -> int:
    """Deferral bound in ns; zero disables the isolation gate entirely."""
    if os.environ.get("NTA_SGLANG_ISOLATION_ADMISSION") != "1":
        return 0
    raw = os.environ.get("NTA_SGLANG_ISOLATION_MAX_DEFER_US", "20000")
    try:
        bound = int(raw)
    except ValueError as error:
        raise RuntimeError(
            "NTA_SGLANG_ISOLATION_MAX_DEFER_US must be an integer "
            f"microsecond bound, got {raw!r}"
        ) from error
    if bound <= 0:
        raise RuntimeError(
            "isolation admission requires a positive deferral bound so "
            "external requests cannot starve"
        )
    return bound * 1000


def _defer_for_isolation(adder: Any, request: Any, bridge: Any) -> bool:
    """True when admitting this external would batch it over a live decode.

    A host-prefix external stages its claim inside the extend forward, and
    ``enable_mixed_chunk`` merges the running decode batch into that same
    forward, so every co-resident decode token waits the whole staging
    span (measured 37-47ms; 65-77% of claim-staging extends are mixed).
    Withholding the external from *this* prefill batch lets the decode
    batch run alone; the request stays in the waiting queue and is
    admitted on a later iteration, or unconditionally once its deferral
    bound expires, so external TTFT damage is bounded by construction.
    """
    bound_ns = _isolation_defer_ns()
    if bound_ns == 0:
        return False
    running = getattr(adder, "running_batch", None)
    if running is None:
        return False
    live_decodes = [
        candidate
        for candidate in getattr(running, "reqs", ()) or ()
        if not candidate.finished()
    ]
    if not live_decodes:
        return False
    now = time.monotonic_ns()
    since = getattr(request, "_nta_isolation_deferred_since", None)
    if since is None:
        request._nta_isolation_deferred_since = now
        bridge.record_admission(
            admission_isolation_deferred_requests=1,
            admission_isolation_deferred_decodes=len(live_decodes),
        )
        return True
    if now - int(since) < bound_ns:
        bridge.record_admission(admission_isolation_deferred_again=1)
        return True
    bridge.record_admission(admission_isolation_released_on_bound=1)
    return False


def route_external_admission_credit(
    original: Any, adder: Any, request: Any, *args: Any, **kwargs: Any
) -> Any:
    """Do not charge a host-only prefix against dense device admission.

    SGLang checks ``extend_input_len`` before ``init_load_back`` turns the host
    prefix into external identities. Temporarily credit exactly those rows in
    the conservative total-token check. The normal budget update runs after
    publication and therefore charges the physical suffix; the external hook
    separately allocates the bounded staging rows from the real allocator.
    """
    cache, bridge = _external_cache(getattr(adder, "tree_cache", None))
    needs_host_load = getattr(request, "needs_host_load_back", None)
    if bridge is None or not callable(needs_host_load) or not needs_host_load():
        return original(adder, request, *args, **kwargs)
    _validate_external_cache(cache, request)
    if _defer_for_isolation(adder, request, bridge):
        from sglang.srt.managers.schedule_policy import AddReqResult

        return AddReqResult.OTHER
    credit = int(getattr(request, "host_hit_length", 0))
    if credit <= 0:
        raise RuntimeError("external-prefix admission omitted its host-token credit")
    before = int(adder.rem_total_token_offset)
    adder.rem_total_token_offset = before - credit
    bridge.record_admission(
        external_admission_credit_rows=credit,
        external_admission_credit_requests=1,
    )
    try:
        return original(adder, request, *args, **kwargs)
    finally:
        expected = before - credit
        delta = int(adder.rem_total_token_offset) - expected
        adder.rem_total_token_offset = before + delta


def route_allocator_free(
    original: Any, allocator: Any, indices: torch.Tensor, *args: Any, **kwargs: Any
) -> Any:
    """Never return virtual external IDs to SGLang's physical allocator."""
    if args or kwargs:
        return original(allocator, indices, *args, **kwargs)
    if not torch.is_tensor(indices) or indices.numel() == 0:
        return original(allocator, indices)
    virtual = indices >= VIRTUAL_TOKEN_BASE
    if not bool(virtual.any()):
        return original(allocator, indices)
    from nta_runtime.engines.sglang_hicache import find_bridge

    bridge = find_bridge(getattr(allocator, "_kvcache", None))
    if bridge is None or not bridge.external_prefix_enabled:
        raise RuntimeError("virtual external token reached an unowned allocator")
    bridge.record_admission(virtual_token_releases=int(virtual.sum()))
    physical = indices[~virtual]
    if physical.numel() == 0:
        return None
    return original(allocator, physical)


def route_cache_unfinished(
    original: Any, cache: Any, request: Any, *args: Any, **kwargs: Any
) -> Any:
    """Keep an external request private instead of inserting virtual IDs."""
    handle = getattr(request, "_nta_external_prefix", None)
    if os.environ.get("NTA_DEBUG_CACHE_ROUTES") == "1":
        print(
            f"[nta-cache] unfinished rid={getattr(request, 'rid', '?')} "
            f"handle={handle is not None}",
            flush=True,
        )
    if handle is None:
        return original(cache, request, *args, **kwargs)
    token_count = len(request.get_fill_ids())
    indices = cache.req_to_token_pool.req_to_token[
        request.req_pool_idx, :token_count
    ].to(dtype=torch.int64, copy=True)
    request.prefix_indices = indices
    request.cache_protected_len = len(indices)
    return None


def route_cache_finished(
    original: Any, cache: Any, request: Any, *args: Any, **kwargs: Any
) -> Any:
    """Free only physical suffix rows; preserve the pre-existing host prefix."""
    handle = getattr(request, "_nta_external_prefix", None)
    if os.environ.get("NTA_DEBUG_CACHE_ROUTES") == "1":
        print(
            f"[nta-cache] finished rid={getattr(request, 'rid', '?')} "
            f"handle={handle is not None} args={args} kwargs={kwargs}",
            flush=True,
        )
    if handle is None:
        return original(cache, request, *args, **kwargs)
    if not handle._released and not handle.retire("finished"):
        raise RuntimeError("finished external prefix lost its runtime claim")
    saved = request.cache_protected_len
    request.cache_protected_len = handle.resident_prefix_len
    try:
        positional = list(args)
        if positional:
            positional[0] = False
        else:
            kwargs = dict(kwargs)
            kwargs["is_insert"] = False
        return original(cache, request, *positional, **kwargs)
    finally:
        request.cache_protected_len = saved
