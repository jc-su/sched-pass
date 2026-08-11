"""SGLang HiCache ownership bridge used by the plugin hook."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
import threading
import time
import weakref
from typing import Any

import torch

from nta_runtime.critical_work import (
    CriticalWorkPlan,
    RequestWork,
    ServiceModel,
    plan_critical_work,
)


@dataclass
class PendingHostLoad:
    claim_id: int
    consumer_index: int
    host_indices: torch.Tensor
    device_indices: torch.Tensor
    producer_event: Any
    controller: Any
    node_ids: tuple[int, ...]
    held_ack: Any = None
    host_by_device: dict[int, int] = field(default_factory=dict)
    completed_layers: int = 0
    prefetched_layers: dict[int, Any] = field(default_factory=dict)
    prefetch_tensors: tuple[torch.Tensor, ...] = ()

    def materialize_mapping(self) -> dict[int, int]:
        if not self.host_by_device:
            host = self.host_indices.detach().to(device="cpu").tolist()
            device = self.device_indices.detach().to(device="cpu").tolist()
            if len(host) != len(device) or len(set(device)) != len(device):
                raise RuntimeError("SGLang HiCache published an invalid page map")
            self.host_by_device = dict(zip(device, host))
        return self.host_by_device


@dataclass(frozen=True)
class HostLoadProgress:
    """Nonblocking snapshot of an NTA-owned HiCache transfer."""

    consumer_index: int
    leading_layers: int
    total_layers: int
    leading_bytes: int
    total_bytes: int

    @property
    def complete(self) -> bool:
        return self.total_layers > 0 and self.leading_layers == self.total_layers


@dataclass(frozen=True)
class _ProgressPublication:
    snapshot: Any
    bindings: tuple[Any, ...]
    model: ServiceModel


class SglangHiCacheBridge:
    """Own intercepted HiCache loads until the final attention layer retires."""

    def __init__(self, device_pool: Any) -> None:
        self.device_pool = device_pool
        self._pending: dict[int, PendingHostLoad] = {}
        self._owned: dict[int, PendingHostLoad] = {}
        self._next_claim_id = 1
        self._next_virtual_token = 1 << 30
        self._external_prefix_capacity_rows = 0
        self._external_prefixes: dict[str, Any] = {}
        self._external_live_dense_rows = 0
        self._external_live_staging_rows = 0
        self._external_dense_high_water_rows = 0
        self._external_staging_high_water_rows = 0
        self._lock = threading.Lock()
        self._prefetch_callback: Any = None
        self._admission_stats: dict[str, int] = {}
        self._progress_publications: list[_ProgressPublication] = []
        self._latest_request_work: dict[
            tuple[int, int], tuple[RequestWork, ServiceModel]
        ] = {}
        _register_bridge(device_pool, self)

    def set_prefetch_callback(self, callback: Any) -> None:
        self._prefetch_callback = callback

    @property
    def external_prefix_enabled(self) -> bool:
        return self._external_prefix_capacity_rows > 0

    def enable_external_prefixes(self, capacity_rows: int, callback: Any) -> None:
        if capacity_rows <= 0 or not callable(callback):
            raise ValueError("external-prefix ownership requires capacity and callback")
        self._external_prefix_capacity_rows = capacity_rows
        self._prefetch_callback = callback

    def claim_external_prefix(
        self,
        request: Any,
        host_indices: torch.Tensor,
        source_release: Any,
        controller: Any,
        cache: Any,
        *,
        node_ids: tuple[int, ...],
    ) -> Any:
        """Allocate virtual identity plus bounded physical staging rows."""
        from nta_runtime.engines.sglang_external import (
            ExternalPrefixHandle,
            VIRTUAL_TOKEN_BASE,
            VIRTUAL_TOKEN_LIMIT,
        )

        if not self.external_prefix_enabled:
            raise RuntimeError("external-prefix ownership is disabled")
        request_id = str(getattr(request, "rid", "") or "")
        if not request_id:
            raise RuntimeError("external-prefix request omitted its ID")
        with self._lock:
            if request_id in self._external_prefixes:
                raise RuntimeError("request already has a live external prefix")
        token_count = int(host_indices.numel())
        if token_count <= 0:
            raise RuntimeError("external-prefix request has no host rows")
        if controller is None or controller.mem_pool_device is not self.device_pool:
            raise RuntimeError("external-prefix request omitted its cache controller")
        allocator = getattr(
            controller.mem_pool_device_allocator,
            "full_attn_allocator",
            controller.mem_pool_device_allocator,
        )
        physical_capacity = min(self._external_prefix_capacity_rows, token_count)
        staging_rows = allocator.alloc(physical_capacity)
        evicted_rows = 0
        if staging_rows is None:
            from sglang.srt.mem_cache.base_prefix_cache import EvictParams

            available = int(allocator.available_size())
            needed = max(1, physical_capacity - available)
            eviction = cache.evict(EvictParams(num_tokens=needed))
            evicted_rows = int(eviction.num_tokens_evicted)
            staging_rows = allocator.alloc(physical_capacity)
        if staging_rows is None:
            raise RuntimeError("bounded external-prefix staging pool is exhausted")
        if int(staging_rows.numel()) != physical_capacity:
            allocator.free(staging_rows)
            raise RuntimeError("bounded staging allocator returned malformed rows")
        with self._lock:
            virtual_begin = self._next_virtual_token
            virtual_end = virtual_begin + token_count
            namespace_exhausted = (
                virtual_begin < VIRTUAL_TOKEN_BASE
                or virtual_end > VIRTUAL_TOKEN_LIMIT + 1
            )
            if not namespace_exhausted:
                self._next_virtual_token = virtual_end
                claim_id = self._next_claim_id
                self._next_claim_id += 1
        if namespace_exhausted:
            allocator.free(staging_rows)
            raise RuntimeError("external virtual-token namespace is exhausted")
        virtual = torch.arange(
            virtual_begin,
            virtual_end,
            dtype=torch.int64,
            device=controller.device,
        )
        def registry_release(handle: Any) -> None:
            with self._lock:
                current = self._external_prefixes.get(handle.request_id)
                if current is not handle:
                    raise RuntimeError("external-prefix registry lost ownership")
                self._external_prefixes.pop(handle.request_id)
                self._external_live_dense_rows -= token_count
                self._external_live_staging_rows -= physical_capacity
                if (
                    self._external_live_dense_rows < 0
                    or self._external_live_staging_rows < 0
                ):
                    raise RuntimeError("external-prefix capacity accounting underflow")

        handle = ExternalPrefixHandle(
            claim_id=claim_id,
            request_id=request_id,
            consumer_index=-1,
            host_indices=host_indices,
            device_indices=virtual,
            staging_rows=staging_rows,
            controller=controller,
            node_ids=node_ids,
            resident_prefix_len=int(request.prefix_indices.numel()),
            source_release=source_release,
            registry_release=registry_release,
        )
        with self._lock:
            duplicate = request_id in self._external_prefixes
            if not duplicate:
                self._external_prefixes[request_id] = handle
                self._external_live_dense_rows += token_count
                self._external_live_staging_rows += physical_capacity
                self._external_dense_high_water_rows = max(
                    self._external_dense_high_water_rows,
                    self._external_live_dense_rows,
                )
                self._external_staging_high_water_rows = max(
                    self._external_staging_high_water_rows,
                    self._external_live_staging_rows,
                )
        if duplicate:
            allocator.free(staging_rows)
            raise RuntimeError("request acquired two external-prefix claims")
        try:
            self._prefetch_callback(handle)
        except Exception:
            with self._lock:
                self._external_prefixes.pop(request_id, None)
                self._external_live_dense_rows -= token_count
                self._external_live_staging_rows -= physical_capacity
            allocator.free(staging_rows)
            raise
        self.record_admission(
            external_prefix_claims=1,
            external_dense_slots_avoided=token_count - physical_capacity,
            external_staging_slots=physical_capacity,
            external_staging_evicted_rows=evicted_rows,
        )
        return handle

    def external_prefix(self, request_id: str) -> Any | None:
        with self._lock:
            return self._external_prefixes.get(request_id)

    @staticmethod
    def supports(controller: Any) -> bool:
        host_pool = controller.mem_pool_host
        device_pool = controller.mem_pool_device
        return (
            not controller.has_draft
            and controller.io_backend == "kernel"
            and getattr(host_pool, "layout", None) in ("layer_first", "page_first")
            and getattr(host_pool, "pin_memory", False)
            and hasattr(host_pool, "k_data_refs")
            and hasattr(host_pool, "v_data_refs")
            and hasattr(device_pool, "_get_key_buffer")
            and hasattr(device_pool, "_get_value_buffer")
            and int(controller.page_size) == 1
        )

    def claim(self, controller: Any) -> int | None:
        if controller.mem_pool_device is not self.device_pool:
            return None
        if not self.supports(controller):
            return None
        if not controller.load_queue:
            return -1

        from sglang.srt.managers.cache_controller import CacheOperation, HiCacheAck

        producer_id = controller.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(controller.load_queue)
        controller.load_queue.clear()
        event = controller.layer_done_counter.events[producer_id]
        with self._lock:
            claim_id = self._next_claim_id
            self._next_claim_id += 1
        pending = PendingHostLoad(
            claim_id=claim_id,
            consumer_index=producer_id,
            host_indices=op.host_indices,
            device_indices=op.device_indices,
            producer_event=event,
            controller=controller,
            node_ids=tuple(op.node_ids),
        )
        ack = HiCacheAck(event.start_event, event.finish_event, op.node_ids)
        if self._prefetch_callback is None:
            controller.ack_load_queue.append(ack)
        else:
            # A claimed load completes its producer events immediately, which
            # would fire the ack and let loading_check unlock the host radix
            # nodes while the claim still stages from them — churn write-back
            # then recycles the host rows out from under the claim. Holding
            # the ack until retire() pins the host source for the claim's
            # whole lifetime. Retirement records a new finish event after the
            # final copy or attention consumer on the actual CUDA stream.
            pending.held_ack = ack
        with self._lock:
            self._pending[producer_id] = pending
            self._owned[claim_id] = pending
        if self._prefetch_callback is not None:
            try:
                self._prefetch_callback(pending)
            except Exception as error:
                # A prefetch failure must not silently revert a claimed load to
                # stock transfer: that bypasses request-level semantics and is
                # invisible to the zero-fallback measurement gates. Restoring
                # SGLang's own transfer is an explicit, counted opt-in for
                # resilience deployments only.
                if os.environ.get("NTA_SGLANG_ALLOW_PREFETCH_FALLBACK", "0") != "1":
                    raise RuntimeError(
                        "NTA early HiCache prefetch failed for a claimed load; "
                        "set NTA_SGLANG_ALLOW_PREFETCH_FALLBACK=1 to restore "
                        "SGLang transfer instead of failing"
                    ) from error
                logging.getLogger(__name__).exception(
                    "NTA early HiCache prefetch failed; restoring SGLang transfer"
                )
                with self._lock:
                    self._admission_stats["prefetch_fallback_loads"] = (
                        self._admission_stats.get("prefetch_fallback_loads", 0) + 1
                    )
                pending.held_ack = None
                controller.ack_load_queue.append(ack)
                self.fallback(pending)
                return producer_id
        return producer_id

    def get(self, consumer_index: int) -> PendingHostLoad | None:
        if consumer_index < 0:
            return None
        with self._lock:
            return self._pending.get(consumer_index)

    def progress(self, consumer_index: int) -> HostLoadProgress | None:
        """Query ordered layer progress without synchronizing a CUDA stream."""
        pending = self.get(consumer_index)
        if pending is None or not pending.prefetched_layers:
            return None
        ordered = tuple(
            pending.prefetched_layers[index]
            for index in sorted(pending.prefetched_layers)
        )
        total_bytes = sum(
            int(layer.key_bytes) + int(layer.value_bytes) for layer in ordered
        )
        leading_layers = 0
        leading_bytes = 0
        for layer in ordered:
            if not layer.ready_event.query():
                break
            leading_layers += 1
            leading_bytes += int(layer.key_bytes) + int(layer.value_bytes)
        return HostLoadProgress(
            consumer_index,
            leading_layers,
            len(ordered),
            leading_bytes,
            total_bytes,
        )

    def transfer_bytes(self, consumer_index: int) -> int:
        """Return pending K/V bytes from shape metadata without touching page data."""
        pending = self.get(consumer_index)
        if pending is None:
            return 0
        page_count = int(pending.host_indices.numel())
        controller = pending.controller
        keys = tuple(controller.mem_pool_host.k_data_refs)
        values = tuple(controller.mem_pool_host.v_data_refs)
        if page_count <= 0 or len(keys) != len(values):
            return 0
        return page_count * sum(
            int(key[0].numel()) * key.element_size()
            + int(value[0].numel()) * value.element_size()
            for key, value in zip(keys, values)
        )

    def record_admission(self, **increments: int) -> None:
        with self._lock:
            for name, value in increments.items():
                if value < 0:
                    raise ValueError("admission counters cannot decrease")
                self._admission_stats[name] = self._admission_stats.get(name, 0) + int(
                    value
                )

    def admission_stats(self) -> dict[str, int]:
        with self._lock:
            result = dict(self._admission_stats)
            result.update(
                external_live_dense_rows=self._external_live_dense_rows,
                external_live_staging_rows=self._external_live_staging_rows,
                external_dense_high_water_rows=self._external_dense_high_water_rows,
                external_staging_high_water_rows=(
                    self._external_staging_high_water_rows
                ),
            )
            return result

    def progress_publication_available(self) -> bool:
        """Return whether another nonblocking compiler snapshot can be retained."""
        self._drain_progress_publications()
        with self._lock:
            return len(self._progress_publications) < 8

    def publish_request_progress(
        self,
        snapshot: Any,
        bindings: tuple[Any, ...],
        *,
        bandwidth_bytes_per_second: int,
        fixed_latency_ns: int = 0,
    ) -> None:
        """Publish one post-discovery snapshot for serving admission.

        The snapshot owns pinned storage until its stream event completes. A
        bounded number may be in flight; callers must check
        ``progress_publication_available`` before capturing a new one.
        """
        if not bindings or not getattr(snapshot, "pending", False):
            raise ValueError("request-progress publication is empty")
        model = ServiceModel(
            bandwidth_bytes_per_second,
            fixed_latency_ns=fixed_latency_ns,
        )
        model.validate()
        self._drain_progress_publications()
        with self._lock:
            if len(self._progress_publications) >= 8:
                raise RuntimeError("request-progress publication ring is full")
            self._progress_publications.append(
                _ProgressPublication(snapshot, tuple(bindings), model)
            )
            self._admission_stats["progress_feedback_published"] = (
                self._admission_stats.get("progress_feedback_published", 0) + 1
            )

    def poll_critical_work(
        self, request_ids: set[int]
    ) -> CriticalWorkPlan | None:
        """Consume current-generation compiler work for active engine requests."""
        if not request_ids:
            return None
        self._drain_progress_publications()
        with self._lock:
            selected_keys = [
                key
                for key, (work, _) in self._latest_request_work.items()
                if work.request_id in request_ids
            ]
            selected = [self._latest_request_work.pop(key) for key in selected_keys]
        if not selected:
            return None
        models = {model for _, model in selected}
        if len(models) != 1:
            raise RuntimeError("request-progress feedback mixed service models")
        plan = plan_critical_work(
            (work for work, _ in selected),
            now_ns=time.monotonic_ns(),
            model=models.pop(),
        )
        self.record_admission(
            progress_feedback_consumed=1,
            progress_feedback_requests=len(plan.requests),
        )
        return plan

    def _drain_progress_publications(self) -> None:
        with self._lock:
            publications = tuple(self._progress_publications)
            self._progress_publications.clear()
        incomplete: list[_ProgressPublication] = []
        completed: list[tuple[RequestWork, ServiceModel]] = []
        stale = 0
        for publication in publications:
            progress = publication.snapshot.query()
            if progress is None:
                incomplete.append(publication)
                continue
            if len(progress) != len(publication.bindings):
                raise RuntimeError("request-progress snapshot changed row count")
            for binding, item in zip(publication.bindings, progress):
                if (
                    item.request_id != binding.request_id
                    or item.generation != binding.generation
                ):
                    stale += 1
                    continue
                completed.append(
                    (
                        RequestWork.from_progress(
                            item,
                            priority=binding.priority,
                            deadline_ns=0,
                        ),
                        publication.model,
                    )
                )
        with self._lock:
            self._progress_publications.extend(incomplete)
            for work, model in completed:
                self._latest_request_work[(work.request_id, work.generation)] = (
                    work,
                    model,
                )
            if stale:
                self._admission_stats["progress_feedback_stale_rows"] = (
                    self._admission_stats.get("progress_feedback_stale_rows", 0)
                    + stale
                )

    def complete_layer(self, pending: PendingHostLoad, local_layer: int) -> None:
        if local_layer != pending.completed_layers:
            raise RuntimeError(
                "SGLang attention layers reached HiCache out of order "
                f"(expected {pending.completed_layers}, got {local_layer})"
            )
        pending.producer_event.complete(local_layer)
        pending.completed_layers += 1
        if pending.completed_layers == pending.controller.layer_num:
            stream = torch.cuda.current_stream()
            if pending.host_indices.is_cuda:
                pending.host_indices.record_stream(stream)
            if pending.device_indices.is_cuda:
                pending.device_indices.record_stream(stream)
            self._drop_ownership(pending)

    def _drop_ownership(self, pending: PendingHostLoad) -> bool:
        with self._lock:
            current = self._owned.get(pending.claim_id)
            if current is not pending:
                return False
            self._owned.pop(pending.claim_id)
            if self._pending.get(pending.consumer_index) is pending:
                self._pending.pop(pending.consumer_index)
            return True

    def retire(
        self,
        pending: PendingHostLoad,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> bool:
        """Drop a claim whose completion happened outside the layer flow.

        The tiered selected path completes producer events at claim time and
        serves layers itself; the pending entry must still be released when
        the claim ends so producer-slot reuse stays fail-closed for genuinely
        live entries.  A held acknowledgement must describe NTA's last use of
        the pinned host rows, not SGLang's already-completed producer event.
        Recording a replacement finish event on the consuming stream keeps
        reclamation asynchronous while preventing completion DMA from racing
        host-row reuse.
        """
        if not self._drop_ownership(pending):
            return False
        ack = pending.held_ack
        if ack is not None:
            if stream is not None:
                finish_event = torch.cuda.Event()
                finish_event.record(stream)
                ack = type(ack)(ack.start_event, finish_event, ack.node_ids)
                if pending.host_indices.is_cuda:
                    pending.host_indices.record_stream(stream)
                if pending.device_indices.is_cuda:
                    pending.device_indices.record_stream(stream)
            pending.controller.ack_load_queue.append(ack)
            pending.held_ack = None
        return True

    def handoff_prefetch(
        self, pending: PendingHostLoad, stream: torch.cuda.Stream
    ) -> None:
        """Publish an enqueued transfer to graph replay without CPU callbacks."""
        if pending.completed_layers != 0:
            raise RuntimeError("HiCache transfer was already handed off")
        with torch.cuda.stream(stream):
            for local_layer in range(int(pending.controller.layer_num)):
                pending.producer_event.complete(local_layer)
            if pending.host_indices.is_cuda:
                pending.host_indices.record_stream(stream)
            if pending.device_indices.is_cuda:
                pending.device_indices.record_stream(stream)
        pending.completed_layers = int(pending.controller.layer_num)
        if not self._drop_ownership(pending):
            raise RuntimeError("HiCache claim changed before graph handoff")

    def fallback(self, pending: PendingHostLoad) -> None:
        """Resume SGLang's original layer-wise transfer after a planning miss."""
        from sglang.srt.utils import get_device_module

        controller = pending.controller
        device_module = get_device_module()
        host_indices, device_indices = controller.move_indices(
            pending.host_indices, pending.device_indices
        )
        pending.producer_event.start_event.record()
        with device_module.stream(controller.load_stream):
            pending.producer_event.start_event.wait(controller.load_stream)
            for local_layer in range(controller.layer_num):
                controller.mem_pool_host.load_to_device_per_layer(
                    controller.mem_pool_device,
                    host_indices,
                    device_indices,
                    local_layer,
                    controller.io_backend,
                )
                pending.producer_event.complete(local_layer)
            if host_indices.is_cuda:
                host_indices.record_stream(controller.load_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(controller.load_stream)
        self._drop_ownership(pending)


_BRIDGES: dict[int, tuple[weakref.ReferenceType[Any], weakref.ReferenceType[Any]]] = {}
_BRIDGES_LOCK = threading.Lock()


def _register_bridge(device_pool: Any, bridge: SglangHiCacheBridge) -> None:
    key = id(device_pool)

    def remove(_reference: Any) -> None:
        with _BRIDGES_LOCK:
            _BRIDGES.pop(key, None)

    with _BRIDGES_LOCK:
        _BRIDGES[key] = (weakref.ref(device_pool, remove), weakref.ref(bridge, remove))


def find_bridge(device_pool: Any) -> SglangHiCacheBridge | None:
    with _BRIDGES_LOCK:
        entry = _BRIDGES.get(id(device_pool))
        if entry is None or entry[0]() is not device_pool:
            return None
        return entry[1]()


def route_start_loading(
    original: Any, controller: Any, *args: Any, **kwargs: Any
) -> int:
    bridge = find_bridge(controller.mem_pool_device)
    if bridge is not None:
        result = bridge.claim(controller)
        if result is not None:
            return result
    return original(controller, *args, **kwargs)
