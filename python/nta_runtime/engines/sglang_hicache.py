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
    lease_id: int
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

    def __init__(self, device_pool: Any, *, work_capacity: int = 4096) -> None:
        if work_capacity <= 0:
            raise ValueError("HiCache progress work capacity must be positive")
        self.device_pool = device_pool
        self._work_capacity = work_capacity
        self._pending: dict[int, PendingHostLoad] = {}
        self._owned: dict[int, PendingHostLoad] = {}
        self._next_lease_id = 1
        self._lock = threading.Lock()
        self._prefetch_callback: Any = None
        self._admission_stats: dict[str, int] = {}
        self._progress_publications: list[_ProgressPublication] = []
        self._latest_request_work: dict[
            tuple[int, int], tuple[RequestWork, ServiceModel]
        ] = {}
        self._latest_request_key: dict[int, tuple[int, int]] = {}
        _register_bridge(device_pool, self)

    def set_prefetch_callback(self, callback: Any) -> None:
        self._prefetch_callback = callback

    def close(self) -> None:
        """Retire every outstanding lease during orderly engine teardown."""
        with self._lock:
            pending = tuple(self._owned.values())
        if not pending:
            return
        stream = torch.cuda.current_stream()
        for lease in pending:
            self.retire(lease, stream=stream)

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

    def acquire_load(self, controller: Any) -> int | None:
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
            lease_id = self._next_lease_id
            self._next_lease_id += 1
        pending = PendingHostLoad(
            lease_id=lease_id,
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
            # A leased load completes its producer events immediately, which
            # would fire the ack and let loading_check unlock the host radix
            # nodes while the lease still stages from them — churn write-back
            # then recycles the host rows out from under the lease. Holding
            # the ack until retire() pins the host source for the lease's
            # whole lifetime. Retirement records a new finish event after the
            # final copy or attention consumer on the actual CUDA stream.
            pending.held_ack = ack
        with self._lock:
            self._pending[producer_id] = pending
            self._owned[lease_id] = pending
        if self._prefetch_callback is not None:
            try:
                self._prefetch_callback(pending)
            except Exception as error:
                # A prefetch failure must not silently revert a leased load to
                # stock transfer: that bypasses request-level semantics and is
                # invisible to the zero-fallback measurement gates. Restoring
                # SGLang's own transfer is an explicit, counted opt-in for
                # resilience deployments only.
                if os.environ.get("NTA_EXECUTION_ALLOW_PREFETCH_FALLBACK", "0") != "1":
                    # Fail closed, but not dirty: the dead lease must not
                    # keep the producer slot occupied or the host nodes
                    # pinned for whoever survives this exception.
                    self._drop_ownership(pending)
                    raise RuntimeError(
                        "NTA early HiCache prefetch failed for a leased load; "
                        "set NTA_EXECUTION_ALLOW_PREFETCH_FALLBACK=1 to enable "
                        "SGLang transfer instead of failing"
                    ) from error
                logging.getLogger(__name__).exception(
                    "NTA early HiCache prefetch failed; restoring SGLang transfer"
                )
                with self._lock:
                    self._admission_stats["prefetch_fallback_loads"] = (
                        self._admission_stats.get("prefetch_fallback_loads", 0) + 1
                    )
                # fallback() replays the transfer and its tail drops
                # ownership, which now delivers the held acknowledgement
                # with the producer-finish semantics a real transfer has.
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
            return dict(self._admission_stats)

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

    def poll_critical_work(self, request_ids: set[int]) -> CriticalWorkPlan | None:
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
            selected = []
            for key in selected_keys:
                selected.append(self._latest_request_work.pop(key))
                if self._latest_request_key.get(key[0]) == key:
                    self._latest_request_key.pop(key[0], None)
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
        position = 0
        try:
            for position, publication in enumerate(publications):
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
        except Exception:
            # Publications own pinned host storage until their stream
            # events complete; dropping the untraversed tail on an
            # exception would free buffers under in-flight D2H copies.
            # Requeue the failing entry, the tail, and the incomplete
            # snapshots, then let the failure propagate. Work already in
            # ``completed`` is lost with the raise, which only costs
            # feedback freshness — its buffers are already quiesced.
            with self._lock:
                self._progress_publications.extend(publications[position:])
                self._progress_publications.extend(incomplete)
            raise
        with self._lock:
            self._progress_publications.extend(incomplete)
            for work, model in completed:
                # Keep only the newest generation for a request.  Feedback is
                # advisory admission state, not an ownership record; stale
                # generations must not accumulate until a long-lived server
                # exhausts memory.
                previous_key = self._latest_request_key.pop(work.request_id, None)
                if previous_key is not None:
                    self._latest_request_work.pop(previous_key, None)
                if len(self._latest_request_work) >= self._work_capacity:
                    evicted_key = next(iter(self._latest_request_work))
                    self._latest_request_work.pop(evicted_key)
                    if self._latest_request_key.get(evicted_key[0]) == evicted_key:
                        self._latest_request_key.pop(evicted_key[0], None)
                    self._admission_stats["progress_feedback_evictions"] = (
                        self._admission_stats.get("progress_feedback_evictions", 0)
                        + 1
                    )
                key = (work.request_id, work.generation)
                self._latest_request_work[key] = (
                    work,
                    model,
                )
                self._latest_request_key[work.request_id] = key
            if stale:
                self._admission_stats["progress_feedback_stale_rows"] = (
                    self._admission_stats.get("progress_feedback_stale_rows", 0) + stale
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
            current = self._owned.get(pending.lease_id)
            if current is not pending:
                return False
            self._owned.pop(pending.lease_id)
            if self._pending.get(pending.consumer_index) is pending:
                self._pending.pop(pending.consumer_index)
        # Ownership ends exactly once, and every ending path must deliver
        # the held acknowledgement or SGLang never unlocks the host radix
        # nodes — a silent one-pinned-prefix-per-load leak. Callers needing
        # a stream-fenced finish event replace ``held_ack`` before dropping.
        ack = pending.held_ack
        if ack is not None:
            pending.held_ack = None
            pending.controller.ack_load_queue.append(ack)
        return True

    def retire(
        self,
        pending: PendingHostLoad,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> bool:
        """Drop an external load whose completion happened outside the layer flow.

        The pending entry must still be released when the request ends so
        producer-slot reuse stays fail-closed for genuinely live entries. A
        held acknowledgement must describe NTA's last use of
        the pinned host rows, not SGLang's already-completed producer event.
        Recording a replacement finish event on the consuming stream keeps
        reclamation asynchronous while preventing completion DMA from racing
        host-row reuse.
        """
        ack = pending.held_ack
        if ack is not None and stream is not None:
            finish_event = torch.cuda.Event()
            finish_event.record(stream)
            pending.held_ack = type(ack)(ack.start_event, finish_event, ack.node_ids)
            if pending.host_indices.is_cuda:
                pending.host_indices.record_stream(stream)
            if pending.device_indices.is_cuda:
                pending.device_indices.record_stream(stream)
        return self._drop_ownership(pending)

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
            raise RuntimeError("HiCache lease changed before graph handoff")

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
        result = bridge.acquire_load(controller)
        if result is not None:
            return result
    return original(controller, *args, **kwargs)
