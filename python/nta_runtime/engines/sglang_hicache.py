"""SGLang HiCache ownership bridge used by the plugin hook."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import threading
import weakref
from typing import Any

import torch

from nta_runtime.engines.sglang_transfer import (
    MoverCopyInterval,
    HostMoverLeasePlan,
    HostTransferLeasePlan,
)
from nta_runtime.engines.sglang_contracts import (
    LeaseAcquisitionGroup,
    LeaseDeviceIndexMap,
    LeaseOperationRange,
    LeaseOperationRequest,
    LeaseOperationTransfer,
)
from nta_runtime.acquisition_scheduler import LayerAcquisition, LayerAcquisitionModel
from nta_runtime.acquisition_scheduler import AcquisitionGroupIdentity
from nta_runtime.engines.sglang_acquisition_contract import AcquisitionArrivalProfileKey

from nta_runtime.progress_frontier import (
    RequestFrontier,
    RequestFrontierEntry,
    build_request_frontier,
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
    # Exact scheduler-side decomposition captured before SGLang merges load
    # operations. The operation ID, not merely a radix node ID, is the unique
    # control-plane join key: separate requests may legitimately name the same
    # node. This avoids downloading and reverse-engineering the numerical page
    # table to recover request ownership.
    operation_transfers: tuple[LeaseOperationTransfer, ...]
    operation_requests: tuple[LeaseOperationRequest, ...]
    # Only operations whose requests survived PrefillAdder's complete budget
    # checks are execution dependencies of the current forward.  SGLang can
    # still include rejected operations in the same physical load queue; those
    # rows are exact speculative prefetch and intentionally own no request or
    # tenant charge in this lease.
    demand_operation_ids: frozenset[int] | None = None
    storage_keys: tuple[str, ...] | None = None
    held_ack: Any = None
    host_by_device: dict[int, int] = field(default_factory=dict)
    completed_layers: int = 0
    prefetched_layers: dict[int, Any] = field(default_factory=dict)
    prefetch_tensors: tuple[torch.Tensor, ...] = ()
    # Immutable byte geometry for the whole local model.  The acquisition
    # owner publishes this before enqueueing the first frontier so admission
    # can distinguish "four of thirty-six layers ready" from "all four
    # published layers ready" without synchronizing or inspecting page data.
    layer_bytes: tuple[int, ...] = ()
    row_bytes_by_layer: tuple[tuple[int, int], ...] = ()
    mover_plan: HostMoverLeasePlan | None = None
    transfer_plan: HostTransferLeasePlan | None = None
    device_index_map: LeaseDeviceIndexMap | None = None
    transfer_events: tuple[Any, ...] = ()
    # Calibration-only timing ownership for one copy-engine probe lease.  The
    # transport publishes the physical copy interval; the attention adapter
    # later binds the first/last compute arrivals from the same forward.  No
    # event is queried on the serving hot path.
    mover_overlap_copy_intervals: list[MoverCopyInterval] = field(default_factory=list)
    mover_overlap_compute_start: Any = None
    selection_accounted: bool = False
    acquisition: LayerAcquisition | None = None
    operation_bindings: dict[int, Any] = field(default_factory=dict)
    # Canonical, operation-local partition shared by the scheduler identity,
    # physical SM completion waves, and numerical dependency projection.
    scheduled_acquisition_groups: tuple[LeaseAcquisitionGroup, ...] = ()
    acquisition_group_identities: dict[
        int, tuple[AcquisitionGroupIdentity, ...]
    ] = field(default_factory=dict)
    shared_acquisition_registered: bool = False
    shared_deadline_model: LayerAcquisitionModel | None = None
    arrival_profile_key: AcquisitionArrivalProfileKey | None = None
    arrival_profiling: bool = False
    # Set only after metadata selected the ordinary stock numerical path.
    # Typed calibration/probe setup must not bias the arrival-margin sample.
    arrival_profile_active: bool = False
    consumer_policy_probe: bool = False
    partial_profile_recorded: bool = False
    planned_progressive_layers: frozenset[int] = frozenset()

    def transfers_by_operation(self) -> dict[int, LeaseOperationTransfer]:
        transfers: dict[int, LeaseOperationTransfer] = {}
        for transfer in self.operation_transfers:
            if not isinstance(transfer, LeaseOperationTransfer):
                raise RuntimeError("SGLang HiCache published an untyped operation")
            if transfer.operation_id in transfers:
                raise RuntimeError("SGLang HiCache repeated a load operation identity")
            transfers[transfer.operation_id] = transfer
        if sum(item.row_count for item in transfers.values()) != int(
            self.device_indices.numel()
        ):
            raise RuntimeError("SGLang HiCache operations do not cover the lease")
        return transfers

    def materialize_mapping(self) -> dict[int, int]:
        if not self.host_by_device:
            host = self.host_indices.detach().to(device="cpu").tolist()
            device = self.device_indices.detach().to(device="cpu").tolist()
            if len(host) != len(device) or len(set(device)) != len(device):
                raise RuntimeError("SGLang HiCache published an invalid page map")
            self.host_by_device = dict(zip(device, host))
        return self.host_by_device

    def operation_ranges(self) -> tuple[LeaseOperationRange, ...]:
        """Return the exact merge-order partition without inspecting page data."""

        cursor = 0
        ranges: list[LeaseOperationRange] = []
        operation_ids: set[int] = set()
        for transfer in self.operation_transfers:
            if not isinstance(transfer, LeaseOperationTransfer):
                raise RuntimeError("SGLang HiCache published an untyped operation")
            if transfer.operation_id in operation_ids:
                raise RuntimeError("SGLang HiCache repeated a load operation identity")
            operation_ids.add(transfer.operation_id)
            ranges.append(
                LeaseOperationRange(
                    transfer.operation_id,
                    cursor,
                    transfer.row_count,
                )
            )
            cursor += transfer.row_count
        if cursor != int(self.device_indices.numel()):
            raise RuntimeError("SGLang HiCache operations do not partition the lease")
        return tuple(ranges)

    def materialize_device_index_map(self) -> LeaseDeviceIndexMap:
        """Publish the merged lease map in the runtime's int32 CUDA ABI.

        ``CacheOperation.merge_ops`` concatenates unmerged operations in queue
        order.  Retaining that typed partition avoids downloading the page map
        merely to recover request ownership, and the returned tensors remain
        owned by this lease until its final numerical consumer retires it.
        """

        cached = self.device_index_map
        if cached is not None:
            if int(cached.source_indices.numel()) != int(self.device_indices.numel()):
                raise RuntimeError("SGLang lease index geometry changed after binding")
            return cached
        source, destination = self.controller.move_indices(
            self.host_indices, self.device_indices
        )
        if (
            not isinstance(source, torch.Tensor)
            or not isinstance(destination, torch.Tensor)
            or source.ndim != 1
            or destination.ndim != 1
            or source.numel() <= 0
            or source.numel() != destination.numel()
        ):
            raise RuntimeError("SGLang produced an invalid merged lease index map")
        device = torch.device(self.controller.mem_pool_device.device)
        if device.type != "cuda":
            raise RuntimeError("SGLang indexed acquisition requires a CUDA device")
        source = (
            source.detach()
            .to(device=device, dtype=torch.int32, non_blocking=True)
            .contiguous()
        )
        destination = (
            destination.detach()
            .to(device=device, dtype=torch.int32, non_blocking=True)
            .contiguous()
        )
        result = LeaseDeviceIndexMap(
            source,
            destination,
            self.operation_ranges(),
        )
        self.device_index_map = result
        return result

    def materialize_storage_mapping(self, catalog: Any) -> dict[int, int]:
        """Bind stable physical ordinals to this lease's GPU destinations."""

        if self.storage_keys is None:
            raise RuntimeError(
                "physical HiCache load has no stable storage-key connector binding"
            )
        device = tuple(
            int(value)
            for value in self.device_indices.detach().to(device="cpu").tolist()
        )
        if len(device) != len(self.storage_keys) or len(set(device)) != len(device):
            raise RuntimeError("physical HiCache storage/device bindings disagree")
        return {
            destination: int(catalog.ordinal(storage_key))
            for destination, storage_key in zip(device, self.storage_keys, strict=True)
        }


@dataclass(frozen=True)
class HostLoadProgress:
    """Nonblocking snapshot of an NTA-owned HiCache transfer."""

    consumer_index: int
    published_layers: int
    leading_layers: int
    total_layers: int
    leading_bytes: int
    total_bytes: int

    def __post_init__(self) -> None:
        if self.consumer_index < 0 or self.total_layers <= 0 or self.total_bytes <= 0:
            raise ValueError("HiCache progress geometry is invalid")
        if not 0 <= self.leading_layers <= self.published_layers <= self.total_layers:
            raise ValueError("HiCache progress frontiers are not ordered")
        if not 0 <= self.leading_bytes <= self.total_bytes:
            raise ValueError("HiCache progress bytes are invalid")

    @property
    def complete(self) -> bool:
        return self.total_layers > 0 and self.leading_layers == self.total_layers


@dataclass(frozen=True)
class _ProgressPublication:
    snapshot: Any
    bindings: tuple[Any, ...]


class SglangHiCacheBridge:
    """Own intercepted HiCache loads until the final attention layer retires."""

    def __init__(
        self,
        device_pool: Any,
        *,
        work_capacity: int = 4096,
        allow_load_fallback: bool = False,
    ) -> None:
        if work_capacity <= 0:
            raise ValueError("HiCache progress work capacity must be positive")
        self.device_pool = device_pool
        self._work_capacity = work_capacity
        self._allow_load_fallback = bool(allow_load_fallback)
        self._pending: dict[int, PendingHostLoad] = {}
        self._owned: dict[int, PendingHostLoad] = {}
        # ``PrefillAdder.add_one_req`` can enqueue a host-load operation and
        # then reject that request on a later token/chunk budget check.  The
        # framework queue therefore is not, by itself, an admission boundary.
        # Keep the scheduler's explicit decision keyed by SGLang's monotone
        # operation identity so acquire_load() can form a lease containing
        # exactly the requests in the current forward.
        self._operation_admission: dict[int, bool] = {}
        self._operation_requests: dict[int, LeaseOperationRequest] = {}
        self._next_lease_id = 1
        self._lock = threading.Lock()
        self._acquire_callback: Any = None
        self._deadline_model_callback: Any = None
        self._admission_prepare_callback: Any = None
        self._admission_start_callback: Any = None
        self._admission_feasibility_callback: Any = None
        self._progress_callback: Any = None
        self._retire_callback: Any = None
        self._admission_stats: dict[str, int] = {}
        self._progress_publications: list[_ProgressPublication] = []
        self._latest_request_work: dict[tuple[int, int], RequestFrontierEntry] = {}
        self._latest_request_key: dict[int, tuple[int, int]] = {}
        self._closed = False
        _register_bridge(device_pool, self)

    def set_acquire_callback(self, callback: Any) -> None:
        """Install the owner invoked after a host-load lease is captured.

        The callback may defer, directly stage, or incrementally stage the
        payload. Its primary contract is ownership: while it is installed the
        bridge holds SGLang's acknowledgement until the NTA consumer retires.
        """

        self._acquire_callback = callback

    def set_deadline_model_callback(self, callback: Any) -> None:
        """Install the deployment-local transfer/deadline model provider.

        The bridge owns physical lease state but deliberately does not own a
        scheduling policy.  The attention backend supplies completed CUDA-event
        calibration through this callback; admission treats ``None`` as an
        uncalibrated state and must not make an optimistic delay decision.
        """

        self._deadline_model_callback = callback

    def set_admission_acquisition_callbacks(self, *, prepare: Any, start: Any) -> None:
        """Install descriptor-only and transfer-publication admission edges.

        Keeping these as two operations is essential: the scheduler first
        verifies calibrated delay bounds, then starts the finite
        work-conserving queue.  A batch outside the delay bound reaches exact
        metadata binding before transport is committed.
        """

        self._admission_prepare_callback = prepare
        self._admission_start_callback = start

    def set_acquisition_progress_callback(self, callback: Any) -> None:
        """Install the nonblocking shared-link progress owner."""

        self._progress_callback = callback

    def set_admission_feasibility_callback(self, callback: Any) -> None:
        """Install the global resource/schedule feasibility provider."""

        self._admission_feasibility_callback = callback

    def set_acquisition_retire_callback(self, callback: Any) -> None:
        """Install cleanup for shared jobs when a framework lease aborts."""

        self._retire_callback = callback

    def admission_feasibility(
        self, consumer_index: int, batch: Any, progress: HostLoadProgress
    ) -> Any | None:
        pending = self.get(consumer_index)
        callback = self._admission_feasibility_callback
        if pending is None or callback is None:
            model = self.deadline_model(consumer_index, batch)
            return (
                None
                if model is None
                else model.analyze_admission(
                    ready_prefix_layers=progress.leading_layers
                )
            )
        return callback(pending, batch, progress)

    def prepare_admission_acquisition(self, consumer_index: int, batch: Any) -> bool:
        """Prepare immutable lease descriptors without starting transport."""

        pending = self.get(consumer_index)
        callback = self._admission_prepare_callback
        return bool(
            pending is not None and callback is not None and callback(pending, batch)
        )

    def start_admission_acquisition(self, consumer_index: int, batch: Any) -> None:
        """Start the already-prepared finite acquisition queue."""

        pending = self.get(consumer_index)
        callback = self._admission_start_callback
        if pending is None or callback is None:
            raise RuntimeError("HiCache admission acquisition is not configured")
        callback(pending, batch)

    def deadline_model(
        self, consumer_index: int, batch: Any
    ) -> LayerAcquisitionModel | None:
        """Return a calibrated model without inspecting payload data."""

        pending = self.get(consumer_index)
        callback = self._deadline_model_callback
        if pending is None or callback is None:
            return None
        model = callback(pending, batch)
        if model is not None and model.layer_bytes != pending.layer_bytes:
            raise RuntimeError(
                "HiCache deadline model disagrees with captured layer bytes"
            )
        return model

    def close(self) -> None:
        """Retire leases and release feedback snapshots during engine teardown.

        The backend synchronizes CUDA before calling this method.  Keeping the
        synchronization at the backend owner avoids a hidden device-wide sync
        in the steady-state hook while making it safe to drop snapshots that
        retain pinned staging buffers.  Every lease is attempted even when one
        retirement fails, so a single malformed acknowledgement cannot leak
        the remaining host ownership records.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pending = tuple(self._owned.values())
        first_error: BaseException | None = None
        if pending:
            stream = torch.cuda.current_stream()
            for lease in pending:
                try:
                    self.retire(lease, stream=stream)
                except BaseException as error:
                    if first_error is None:
                        first_error = error
        with self._lock:
            self._progress_publications.clear()
            self._latest_request_work.clear()
            self._latest_request_key.clear()
            self._operation_admission.clear()
            self._operation_requests.clear()
        if first_error is not None:
            raise RuntimeError(
                "SGLang HiCache bridge failed to retire one or more leases"
            ) from first_error

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

    def record_operation_admission(self, operation_id: int, admitted: bool) -> None:
        """Publish the scheduler decision for one queued load operation.

        A rejected request can be reconsidered later, so ``False -> True`` is
        the only valid state transition.  Reversing an admitted operation
        would detach an already-formed forward from its physical lease.
        """

        operation_id = int(operation_id)
        if operation_id < 0:
            raise ValueError("SGLang load operation identity is invalid")
        admitted = bool(admitted)
        with self._lock:
            previous = self._operation_admission.get(operation_id)
            if previous is True and not admitted:
                raise RuntimeError(
                    "SGLang revoked an admitted HiCache load operation"
                )
            self._operation_admission[operation_id] = admitted

    def record_operation_request(self, request: LeaseOperationRequest) -> None:
        """Bind one unmerged load operation to its scheduler request owner."""

        if not isinstance(request, LeaseOperationRequest):
            raise TypeError("SGLang operation ownership must use the typed contract")
        with self._lock:
            previous = self._operation_requests.get(request.operation_id)
            if previous is not None and previous != request:
                raise RuntimeError(
                    "SGLang load operation changed request ownership before capture"
                )
            self._operation_requests[request.operation_id] = request

    def acquire_load(self, controller: Any) -> int | None:
        with self._lock:
            if self._closed:
                raise RuntimeError("SGLang HiCache bridge is closed")
        if controller.mem_pool_device is not self.device_pool:
            return None
        if not self.supports(controller):
            return None
        if not controller.load_queue:
            return -1

        from sglang.srt.managers.cache_controller import CacheOperation, HiCacheAck

        queued_operations = tuple(controller.load_queue)
        queued_operation_ids = frozenset(
            int(getattr(operation, "id", -1)) for operation in queued_operations
        )
        with self._lock:
            for operation_id in set(self._operation_admission) - queued_operation_ids:
                self._operation_admission.pop(operation_id, None)
                self._operation_requests.pop(operation_id, None)
            missing = tuple(
                int(getattr(operation, "id", -1))
                for operation in queued_operations
                if int(getattr(operation, "id", -1))
                not in self._operation_admission
            )
            if missing:
                raise RuntimeError(
                    "SGLang HiCache load queue omitted admission ownership for "
                    f"operation(s) {missing}"
                )
            demand_operation_ids = frozenset(
                int(operation.id)
                for operation in queued_operations
                if self._operation_admission[int(operation.id)]
            )
            missing_requests = tuple(
                operation_id
                for operation_id in demand_operation_ids
                if operation_id not in self._operation_requests
            )
            if missing_requests:
                raise RuntimeError(
                    "SGLang admitted load operation omitted request ownership for "
                    f"operation(s) {missing_requests}"
                )
            # Preserve CacheOperation.merge_ops() order.  Logical operation
            # ranges, physical descriptor ranges, and request identities must
            # describe the same partition; sorting opaque operation IDs would
            # silently break that invariant after priority queueing.
            operation_requests = tuple(
                self._operation_requests[int(operation.id)]
                for operation in queued_operations
                if int(operation.id) in demand_operation_ids
            )
        # SGLang may enqueue speculative host prefetch before its budget check
        # admits a request.  A flush can therefore contain both exact demand
        # and unowned speculation.  Never absorb the latter into a
        # scheduler-owned lease: doing so would transfer and reserve bytes for
        # which no request-generation or tenant exists.  Leave it on SGLang's
        # queue so a later request retry can claim it, or SGLang can execute it
        # as an explicitly unowned prefetch on a subsequent start_loading().
        leased_operations = (
            tuple(
                operation
                for operation in queued_operations
                if int(operation.id) in demand_operation_ids
            )
            if self._acquire_callback is not None and demand_operation_ids
            else queued_operations
        )
        leased_operation_ids = frozenset(
            int(operation.id) for operation in leased_operations
        )
        producer_id = controller.layer_done_counter.update_producer()
        operation_transfers: list[LeaseOperationTransfer] = []
        for queued in leased_operations:
            if len(queued.node_ids) != 1:
                raise RuntimeError(
                    "unmerged SGLang HiCache operation has ambiguous node ownership"
                )
            operation_transfers.append(
                LeaseOperationTransfer(
                    int(getattr(queued, "id", -1)),
                    int(queued.node_ids[0]),
                    int(queued.device_indices.numel()),
                )
            )
        op = CacheOperation.merge_ops(list(leased_operations))
        controller.load_queue[:] = [
            operation
            for operation in queued_operations
            if int(operation.id) not in leased_operation_ids
        ]
        with self._lock:
            for operation_id in leased_operation_ids:
                self._operation_admission.pop(operation_id, None)
                self._operation_requests.pop(operation_id, None)
        event = controller.layer_done_counter.events[producer_id]
        from nta_runtime.connectors.sglang_storage import (
            maybe_resolve_sglang_storage_keys,
        )

        storage_keys = maybe_resolve_sglang_storage_keys(
            controller.mem_pool_host, op.host_indices
        )
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
            operation_transfers=tuple(operation_transfers),
            operation_requests=operation_requests,
            demand_operation_ids=demand_operation_ids,
            storage_keys=storage_keys,
        )
        ack = HiCacheAck(event.start_event, event.finish_event, op.node_ids)
        if self._acquire_callback is None:
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
        commit_error: RuntimeError | None = None
        with self._lock:
            if self._closed:
                commit_error = RuntimeError(
                    "SGLang HiCache bridge closed while acquiring a load"
                )
            elif producer_id in self._pending:
                commit_error = RuntimeError(
                    "SGLang reused a live HiCache producer slot"
                )
            elif lease_id in self._owned:  # pragma: no cover - monotone ID guard
                commit_error = RuntimeError("SGLang repeated a HiCache lease identity")
            else:
                self._pending[producer_id] = pending
                self._owned[lease_id] = pending
        if commit_error is not None:
            # close() snapshots ownership while holding the same lock.  A load
            # that loses the final commit race is therefore not in that
            # snapshot and must release its held acknowledgement itself.
            held = pending.held_ack
            if held is not None:
                pending.held_ack = None
                controller.ack_load_queue.append(held)
            raise commit_error
        if self._acquire_callback is not None and not demand_operation_ids:
            # SGLang allocated destinations before rejecting these requests.
            # Complete that framework prefetch so its locks/slots can retire,
            # but do not manufacture request ownership for the unrelated
            # forward that happened to flush the queue.
            self.record_admission(unowned_prefetch_loads=1)
            self.fallback(pending)
        elif self._acquire_callback is not None:
            try:
                self._acquire_callback(pending)
            except Exception as error:
                # An acquisition failure must not silently revert a leased load to
                # stock transfer: that bypasses request-level semantics and is
                # invisible to the zero-fallback measurement gates. Restoring
                # SGLang's own transfer is an explicit, counted opt-in for
                # resilience deployments only.
                if not self._allow_load_fallback:
                    # Fail closed, but not dirty: the dead lease must not
                    # keep the producer slot occupied or the host nodes
                    # pinned for whoever survives this exception.
                    self._drop_ownership(pending)
                    raise RuntimeError(
                        "NTA HiCache acquisition failed for a leased load; "
                        "set NTA_EXECUTION_ALLOW_LOAD_FALLBACK=1 to enable "
                        "SGLang transfer instead of failing"
                    ) from error
                logging.getLogger(__name__).exception(
                    "NTA HiCache acquisition failed; restoring SGLang transfer"
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
        if pending is None or not pending.layer_bytes:
            return None
        if self._progress_callback is not None:
            self._progress_callback(pending)
        total_layers = len(pending.layer_bytes)
        if total_layers != int(pending.controller.layer_num):
            raise RuntimeError("HiCache progress geometry changed during a lease")
        leading_layers = 0
        leading_bytes = 0
        for local_layer, expected_bytes in enumerate(pending.layer_bytes):
            layer = pending.prefetched_layers.get(local_layer)
            if layer is None:
                break
            actual_bytes = int(layer.key_bytes) + int(layer.value_bytes)
            if actual_bytes != expected_bytes:
                raise RuntimeError("HiCache progress byte geometry is inconsistent")
            if not layer.ready_event.query():
                break
            leading_layers += 1
            leading_bytes += actual_bytes
        return HostLoadProgress(
            consumer_index=consumer_index,
            published_layers=len(pending.prefetched_layers),
            leading_layers=leading_layers,
            total_layers=total_layers,
            leading_bytes=leading_bytes,
            total_bytes=sum(pending.layer_bytes),
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
    ) -> None:
        """Publish one post-discovery snapshot for serving admission.

        The snapshot owns pinned storage until its stream event completes. A
        bounded number may be in flight; callers must check
        ``progress_publication_available`` before capturing a new one.
        """
        if not bindings or not getattr(snapshot, "pending", False):
            raise ValueError("request-progress publication is empty")
        self._drain_progress_publications()
        with self._lock:
            if self._closed:
                raise RuntimeError("SGLang HiCache bridge is closed")
            if len(self._progress_publications) >= 8:
                raise RuntimeError("request-progress publication ring is full")
            self._progress_publications.append(
                _ProgressPublication(snapshot, tuple(bindings))
            )
            self._admission_stats["progress_feedback_published"] = (
                self._admission_stats.get("progress_feedback_published", 0) + 1
            )

    def poll_request_frontier(self, request_ids: set[int]) -> RequestFrontier | None:
        """Consume current-generation compiler progress for active requests."""
        if not request_ids:
            return None
        self._drain_progress_publications()
        with self._lock:
            selected_keys = [
                key
                for key, work in self._latest_request_work.items()
                if work.request_id in request_ids
            ]
            selected = []
            for key in selected_keys:
                selected.append(self._latest_request_work.pop(key))
                if self._latest_request_key.get(key[0]) == key:
                    self._latest_request_key.pop(key[0], None)
        if not selected:
            return None
        frontier = build_request_frontier(selected)
        self.record_admission(
            progress_feedback_consumed=1,
            progress_feedback_requests=len(frontier.requests),
        )
        return frontier

    def _drain_progress_publications(self) -> None:
        with self._lock:
            publications = tuple(self._progress_publications)
            self._progress_publications.clear()
        incomplete: list[_ProgressPublication] = []
        completed: list[RequestFrontierEntry] = []
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
                    completed.append(RequestFrontierEntry.from_progress(item))
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
            for work in completed:
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
                        self._admission_stats.get("progress_feedback_evictions", 0) + 1
                    )
                key = (work.request_id, work.generation)
                self._latest_request_work[key] = work
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
            acquisition = getattr(pending, "acquisition", None)
            if acquisition is not None and not acquisition.queue.terminal:
                raise RuntimeError("HiCache lease completed with live acquisition jobs")
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
        acquisition = getattr(pending, "acquisition", None)
        if acquisition is not None:
            acquisition.cancel_unfinished()
        if self._retire_callback is not None:
            self._retire_callback(pending)
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
        acquisition = getattr(pending, "acquisition", None)
        if acquisition is not None:
            acquisition.retire_published()
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
