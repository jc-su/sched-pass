"""Pinned vLLM 0.26 CPU-cache ownership behind NTA's typed data path.

This module is the only NTA component that knows vLLM's
``SimpleCPUOffloadScheduler`` and ``SimpleCPUOffloadWorker`` internals.  vLLM
continues to own prefix-cache allocation, hashing, eviction, and pinned CPU
payload lifetime.  NTA deliberately replaces only the load execution edge:
the scheduler keeps a hit in the current step and the instrumented attention
consumer materializes exact blocks from the worker's pinned tensors.

The upstream connector normally suspends a request and performs a conventional
gather after the forward.  Returning that completion through
``finished_recving`` for an NTA in-forward load would corrupt vLLM's request
state.  We therefore report stream-ordered load quiescence through private
worker metadata consumed only by this connector's scheduler-side owner.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import time
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorWorkerMetadata,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.simple_kv_offload.manager import (
    LoadRequestState,
    SimpleCPUOffloadScheduler,
    StoreRequestState,
    TransferMeta,
)
from vllm.v1.simple_kv_offload.metadata import (
    INVALID_JOB_ID,
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker
from nta_runtime.indexed_transfer import (
    IndexedHostResource,
    StridedCopyGroup,
    analyze_index_pairs,
)
from nta_runtime.acquisition_scheduler import (
    LayerAcquisition,
    TenantCreditCharge,
    TenantCreditLease,
    TenantCreditLedger,
)
from nta_runtime.engines.vllm_config import vllm_host_layers_per_wave
from nta_runtime.runtime import copy_strided_host_runs_async
from nta_runtime.tenant import tenant_budget_specs

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


DEFAULT_CPU_CAPACITY_BYTES = 8 * 1024**3


class VllmHostAcquisition:
    """One exact layer queue backed by contiguous transport-wave fences."""

    def __init__(
        self,
        layer_names: tuple[str, ...],
        layer_fences: tuple[int, ...],
        events: tuple[torch.cuda.Event, ...],
        lifecycle: LayerAcquisition,
        *,
        transfer_blocks: int,
        transfer_bytes: int,
        transfer_runs: int,
        copy_operations: int,
        copy_submissions: int,
        submission_cpu_ns: int,
        credit_lease: TenantCreditLease | None,
    ) -> None:
        if (
            not layer_names
            or len(set(layer_names)) != len(layer_names)
            or len(layer_fences) != len(layer_names)
            or not events
            or set(layer_fences) != set(range(len(events)))
            or any(
                current > following
                for current, following in zip(layer_fences, layer_fences[1:])
            )
            or not lifecycle.fully_published
            or transfer_blocks <= 0
            or transfer_bytes <= 0
            or transfer_runs <= 0
            or copy_operations <= 0
            or copy_submissions <= 0
            or submission_cpu_ns <= 0
        ):
            raise ValueError("vLLM Host acquisition is incomplete")
        self._layer_names = layer_names
        self._ordinals = {name: index for index, name in enumerate(layer_names)}
        self._layer_fences = layer_fences
        self._events = events
        self._lifecycle = lifecycle
        self.transfer_blocks = transfer_blocks
        self.transfer_bytes = transfer_bytes
        self.transfer_runs = transfer_runs
        self.copy_operations = copy_operations
        self.copy_submissions = copy_submissions
        self.submission_cpu_ns = submission_cpu_ns
        self._credit_lease = credit_lease

    @property
    def layer_count(self) -> int:
        return len(self._layer_names)

    @property
    def wave_count(self) -> int:
        return len(self._events)

    @property
    def terminal(self) -> bool:
        return self._lifecycle.queue.terminal

    @property
    def tenant_accounted(self) -> bool:
        return self._credit_lease is not None

    @property
    def tenant_charge_bytes(self) -> int:
        lease = self._credit_lease
        return 0 if lease is None else sum(charge.bytes for charge in lease.charges)

    def fence_for(self, layer_name: str) -> tuple[int, torch.cuda.Event]:
        try:
            layer = self._ordinals[layer_name]
        except KeyError:
            raise RuntimeError(
                f"vLLM Host acquisition does not own layer {layer_name!r}"
            ) from None
        fence = self._layer_fences[layer]
        return fence, self._events[fence]

    def retire(self, layer_name: str) -> None:
        try:
            layer = self._ordinals[layer_name]
        except KeyError:
            raise RuntimeError(
                f"vLLM Host acquisition cannot retire layer {layer_name!r}"
            ) from None
        self._lifecycle.retire(layer)

    def cancel_unfinished(self) -> None:
        self._lifecycle.cancel_unfinished()

    def synchronize(self) -> None:
        self._events[-1].synchronize()

    def completion_ready(self) -> bool:
        return bool(self._events[-1].query())

    def take_credit_release(
        self,
    ) -> tuple[torch.cuda.Event, TenantCreditLease] | None:
        lease = self._credit_lease
        self._credit_lease = None
        return None if lease is None else (self._events[-1], lease)


@dataclass(frozen=True)
class _SynchronousMatch:
    """The scheduler facts upstream otherwise tries to infer from block hashes."""

    local_tokens: int
    external_tokens: int


def _resolve_synchronous_load(
    *,
    local_tokens: int,
    external_tokens: int,
    block_ids_by_group: tuple[list[int], ...],
    cpu_hit_blocks: tuple[list[Any], ...],
    kv_cache_groups: tuple[Any, ...],
    cp_world_size: int,
) -> tuple[list[int], list[Any]]:
    """Map an exact external token interval to GPU and pinned-CPU blocks.

    vLLM's stock offload manager derives ``local_tokens`` by counting hashed
    GPU blocks.  That is valid for its delayed-cache asynchronous path, but in
    NTA's synchronous path the allocator has already hashed the newly admitted
    external blocks.  Carrying the scheduler fact explicitly avoids counting
    that interval twice.
    """

    if local_tokens < 0 or external_tokens <= 0 or cp_world_size <= 0:
        raise RuntimeError("invalid synchronous vLLM host token interval")
    num_groups = len(block_ids_by_group)
    if (
        len(cpu_hit_blocks) != num_groups
        or len(kv_cache_groups) != num_groups
        or num_groups == 0
    ):
        raise RuntimeError("vLLM host cache groups disagree during load binding")

    gpu_block_ids: list[int] = []
    selected_cpu_blocks: list[Any] = []
    for group_index, (group_gpu_ids, group_cpu_blocks, group) in enumerate(
        zip(block_ids_by_group, cpu_hit_blocks, kv_cache_groups, strict=True)
    ):
        block_size = int(group.kv_cache_spec.block_size) * cp_world_size
        if block_size <= 0 or local_tokens % block_size or external_tokens % block_size:
            raise RuntimeError(
                f"vLLM host token interval is not aligned to cache group {group_index}"
            )
        first = local_tokens // block_size
        count = external_tokens // block_size
        end = first + count
        if end > len(group_gpu_ids) or count > len(group_cpu_blocks):
            raise RuntimeError(
                "vLLM host allocation omits an admitted external cache block"
            )
        for gpu_block_id, cpu_block in zip(
            group_gpu_ids[first:end], group_cpu_blocks[:count], strict=True
        ):
            if cpu_block.is_null:
                continue
            if int(gpu_block_id) < 0 or int(cpu_block.block_id) < 0:
                raise RuntimeError("vLLM host load contains a negative block ID")
            gpu_block_ids.append(int(gpu_block_id))
            selected_cpu_blocks.append(cpu_block)

    if not gpu_block_ids or len(gpu_block_ids) != len(selected_cpu_blocks):
        raise RuntimeError("vLLM host hit did not resolve to a numerical load")
    if len(set(gpu_block_ids)) != len(gpu_block_ids):
        raise RuntimeError("vLLM host load aliases a GPU destination block")
    return gpu_block_ids, selected_cpu_blocks


def _dense_row_bytes(tensor: torch.Tensor, layer_name: str) -> int:
    """Return one block's byte span for any dense positive-stride permutation."""

    expected_stride = 1
    dimensions = sorted(
        (
            (int(extent), int(stride))
            for extent, stride in zip(
                tensor.shape[1:], tensor.stride()[1:], strict=True
            )
            if int(extent) > 1
        ),
        key=lambda item: item[1],
    )
    for extent, stride in dimensions:
        if stride != expected_stride:
            raise RuntimeError(
                f"vLLM layer {layer_name!r} has a sparse or overlapping KV row"
            )
        expected_stride *= extent
    return expected_stride * int(tensor.element_size())


def build_indexed_host_resources(
    kv_caches: dict[str, torch.Tensor],
    gpu_backings: dict[str, torch.Tensor],
    cpu_backings: dict[str, torch.Tensor],
) -> dict[str, IndexedHostResource]:
    """Resolve vLLM layer views into their pinned packed-row resources."""

    resources: dict[str, IndexedHostResource] = {}
    for layer_name, layer_tensor in kv_caches.items():
        if not isinstance(layer_tensor, torch.Tensor):
            raise RuntimeError("vLLM host_staged supports attention tensors only")
        if (
            not layer_tensor.is_cuda
            or layer_tensor.ndim < 2
            or int(layer_tensor.shape[0]) <= 0
            or any(int(stride) <= 0 for stride in layer_tensor.stride())
        ):
            raise RuntimeError(
                f"vLLM layer {layer_name!r} has unsupported packed KV geometry"
            )
        storage_address = int(layer_tensor.untyped_storage().data_ptr())
        matches = [
            (name, tensor)
            for name, tensor in gpu_backings.items()
            if int(tensor.untyped_storage().data_ptr()) == storage_address
        ]
        if len(matches) != 1:
            raise RuntimeError(f"vLLM layer {layer_name!r} has ambiguous host backing")
        backing_name, gpu_backing = matches[0]
        try:
            cpu_backing = cpu_backings[backing_name]
        except KeyError as error:
            raise RuntimeError(
                f"vLLM layer {layer_name!r} has no pinned CPU backing"
            ) from error
        if (
            not isinstance(cpu_backing, torch.Tensor)
            or cpu_backing.is_cuda
            or not cpu_backing.is_pinned()
            or not cpu_backing.is_contiguous()
        ):
            raise RuntimeError(
                f"vLLM layer {layer_name!r} has no contiguous pinned CPU backing"
            )
        source_stride = int(cpu_backing.stride(0)) * int(cpu_backing.element_size())
        destination_stride = int(layer_tensor.stride(0)) * int(
            layer_tensor.element_size()
        )
        row_bytes = _dense_row_bytes(layer_tensor, layer_name)
        source_offset = int(layer_tensor.data_ptr()) - int(gpu_backing.data_ptr())
        if (
            int(gpu_backing.shape[0]) != int(layer_tensor.shape[0])
            or int(cpu_backing.shape[0]) <= 0
            or source_stride
            != int(gpu_backing.stride(0)) * int(gpu_backing.element_size())
            or destination_stride != source_stride
        ):
            raise RuntimeError(
                f"vLLM layer {layer_name!r} disagrees with its packed backing"
            )
        resources[layer_name] = IndexedHostResource(
            source_tensor=cpu_backing,
            destination_tensor=layer_tensor,
            source_offset_bytes=source_offset,
            row_bytes=row_bytes,
            source_stride_bytes=source_stride,
            destination_stride_bytes=destination_stride,
            source_rows=int(cpu_backing.shape[0]),
            destination_rows=int(layer_tensor.shape[0]),
        )
    if set(resources) != set(kv_caches):
        raise RuntimeError("vLLM host resource registration lost a layer")
    return resources


@dataclass(frozen=True)
class NtaVllmHostWorkerMetadata(KVConnectorWorkerMetadata):
    """Completions that must not enter vLLM's request-resume state machine."""

    completed_load_requests: dict[str, int]
    completed_store_events: dict[int, int]

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "NtaVllmHostWorkerMetadata":
        if not isinstance(other, NtaVllmHostWorkerMetadata):
            raise TypeError("cannot aggregate foreign vLLM host metadata")
        loads = dict(self.completed_load_requests)
        for request_id, count in other.completed_load_requests.items():
            loads[request_id] = loads.get(request_id, 0) + count
        stores = dict(self.completed_store_events)
        for event, count in other.completed_store_events.items():
            stores[event] = stores.get(event, 0) + count
        return NtaVllmHostWorkerMetadata(loads, stores)


@dataclass(frozen=True)
class VllmHostRequestTransfer:
    """Exact pinned-source/HBM-destination pairs owned by one request."""

    request_id: str
    pairs: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if (
            not self.request_id
            or not self.pairs
            or any(min(source, destination) < 0 for source, destination in self.pairs)
        ):
            raise ValueError("vLLM Host request transfer is incomplete")
        if len({destination for _, destination in self.pairs}) != len(self.pairs):
            raise ValueError("vLLM Host request transfer repeats a destination")


@dataclass(frozen=True)
class NtaVllmHostTransferMetadata:
    """Upstream store metadata plus typed request ownership for Host loads."""

    upstream: SimpleCPUOffloadMetadata
    requests: tuple[VllmHostRequestTransfer, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.upstream, SimpleCPUOffloadMetadata):
            raise TypeError("vLLM Host transfer metadata has a foreign payload")
        flattened = tuple(pair for request in self.requests for pair in request.pairs)
        upstream_pairs = tuple(
            zip(
                self.upstream.load_cpu_blocks,
                self.upstream.load_gpu_blocks,
                strict=True,
            )
        )
        if flattened != upstream_pairs:
            raise ValueError("vLLM Host request ownership changed the transfer map")
        if len({request.request_id for request in self.requests}) != len(self.requests):
            raise ValueError("vLLM Host transfer repeats a request")
        if upstream_pairs:
            expected_requests = tuple(
                self.upstream.load_event_to_reqs.get(self.upstream.load_event, ())
            )
            if tuple(request.request_id for request in self.requests) != (
                expected_requests
            ):
                raise ValueError("vLLM Host load event changed request ownership")
        elif self.requests:
            raise ValueError("empty vLLM Host metadata owns request transfers")


def _capacity_per_rank(vllm_config: "VllmConfig") -> int:
    transfer = vllm_config.kv_transfer_config
    extra = {} if transfer is None else transfer.kv_connector_extra_config or {}
    total = int(extra.get("cpu_bytes_to_use", DEFAULT_CPU_CAPACITY_BYTES))
    world_size = int(vllm_config.parallel_config.world_size)
    if total <= 0 or world_size <= 0:
        raise RuntimeError("vLLM host cache capacity and world size must be positive")
    per_rank = int(extra.get("cpu_bytes_to_use_per_rank", total // world_size))
    if per_rank <= 0:
        raise RuntimeError("vLLM host cache capacity per rank must be positive")
    return per_rank


class VllmHostScheduler:
    """Scheduler-side vLLM cache policy with synchronous-in-forward admission."""

    def __init__(
        self, vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"
    ) -> None:
        if not vllm_config.cache_config.enable_prefix_caching:
            raise RuntimeError("vLLM host_staged requires prefix caching")
        from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

        scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self._manager = SimpleCPUOffloadScheduler(
            vllm_config,
            kv_cache_config,
            _capacity_per_rank(vllm_config),
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            lazy_offload=bool(extra.get("lazy_offload", False)),
        )
        cleanup = getattr(self._manager, "_cleanup_load_request", None)
        if not callable(cleanup):
            raise RuntimeError(
                "vLLM host cache no longer exposes load ownership cleanup"
            )
        self._expected_workers = int(vllm_config.parallel_config.world_size)
        self._load_completion_counts: dict[str, int] = {}
        self._synchronous_matches: dict[str, _SynchronousMatch] = {}

    def bind_gpu_block_pool(self, block_pool: "BlockPool") -> None:
        self._manager.bind_gpu_block_pool(block_pool)

    def matched_tokens(self, request: "Request", computed_tokens: int) -> int:
        request_id = str(request.request_id)
        self._synchronous_matches.pop(request_id, None)
        count, asynchronous = self._manager.get_num_new_matched_tokens(
            request, computed_tokens
        )
        if count is None:
            raise RuntimeError("vLLM host cache returned an indeterminate local hit")
        if count > 0 and not asynchronous:
            raise RuntimeError("vLLM host cache changed its pinned-load contract")
        if count > 0:
            self._synchronous_matches[request_id] = _SynchronousMatch(
                local_tokens=int(computed_tokens), external_tokens=int(count)
            )
        return int(count)

    def update_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        external_tokens: int,
    ) -> None:
        request_id = str(request.request_id)
        match = self._synchronous_matches.pop(request_id, None)
        if external_tokens == 0:
            if match is not None:
                # Let upstream release the temporary CPU pin before rejecting
                # the scheduler contradiction.
                self._manager.update_state_after_alloc(request, blocks, 0)
                raise RuntimeError("vLLM discarded an admitted host cache hit")
            self._manager.update_state_after_alloc(request, blocks, 0)
            return
        if match is None or int(external_tokens) != match.external_tokens:
            pending = self._manager._pending_cpu_hits.pop(request_id, None)
            if pending is not None:
                self._manager._free_pending_cpu_hit(pending)
            raise RuntimeError("vLLM host allocation has no exact pinned cache hit")

        pending = self._manager._pending_cpu_hits.pop(request_id, None)
        if pending is None:
            raise RuntimeError("vLLM host allocation lost its pinned CPU payload")
        cpu_hit_blocks, hit_length = pending

        cpu_load_pinned = False
        gpu_load_pinned = False
        selected_cpu_blocks: list[Any] = []
        selected_gpu_blocks: list[Any] = []
        try:
            block_ids_by_group = blocks.get_block_ids()
            if block_ids_by_group is None:
                raise RuntimeError("vLLM host allocation has no destination blocks")
            if int(hit_length) != match.external_tokens:
                raise RuntimeError("vLLM host hit length changed before allocation")
            gpu_block_ids, selected_cpu_blocks = _resolve_synchronous_load(
                local_tokens=match.local_tokens,
                external_tokens=match.external_tokens,
                block_ids_by_group=block_ids_by_group,
                cpu_hit_blocks=cpu_hit_blocks,
                kv_cache_groups=tuple(
                    self._manager.cpu_kv_cache_config.kv_cache_groups
                ),
                cp_world_size=int(self._manager.cp_world_size),
            )
            gpu_pool = self._manager._gpu_block_pool
            if gpu_pool is None:
                raise RuntimeError("vLLM host scheduler has no bound GPU block pool")
            selected_gpu_blocks = [gpu_pool.blocks[block] for block in gpu_block_ids]

            self._manager.cpu_block_pool.touch(selected_cpu_blocks)
            cpu_load_pinned = True
            gpu_pool.touch(selected_gpu_blocks)
            gpu_load_pinned = True

            if self._manager._reqs_to_load.get(request_id) is not None:
                raise RuntimeError("vLLM host request already owns an active load")
            self._manager._reqs_to_load[request_id] = LoadRequestState(
                request=request,
                transfer_meta=TransferMeta(
                    gpu_block_ids=gpu_block_ids,
                    cpu_block_ids=[
                        int(block.block_id) for block in selected_cpu_blocks
                    ],
                ),
            )
            if (
                not self._manager._lazy_mode
                and request_id not in self._manager._reqs_to_store
            ):
                num_groups = len(block_ids_by_group)
                self._manager._reqs_to_store[request_id] = StoreRequestState(
                    request=request,
                    block_ids=tuple([] for _ in range(num_groups)),
                    num_stored_blocks=[0] * num_groups,
                )
        except Exception:
            if gpu_load_pinned:
                self._manager._gpu_block_pool.free_blocks(selected_gpu_blocks)
            if cpu_load_pinned:
                self._manager.cpu_block_pool.free_blocks(selected_cpu_blocks)
            self._manager._reqs_to_load.pop(request_id, None)
            raise
        finally:
            # Drop the temporary lookup pin.  The selected load blocks keep the
            # extra ref acquired above until stream completion; unselected
            # suffix blocks become evictable immediately.
            self._manager._free_pending_cpu_hit(pending)

    def build_metadata(self, scheduler_output: Any) -> NtaVllmHostTransferMetadata:
        requests: list[VllmHostRequestTransfer] = []
        for request_id, state in self._manager._reqs_to_load.items():
            if state.load_event is not None:
                continue
            transfer = state.transfer_meta
            if transfer is None:
                raise RuntimeError("vLLM Host load lost its exact transfer map")
            requests.append(
                VllmHostRequestTransfer(
                    str(request_id),
                    tuple(
                        zip(
                            transfer.cpu_block_ids,
                            transfer.gpu_block_ids,
                            strict=True,
                        )
                    ),
                )
            )
        upstream = self._manager.build_connector_meta(scheduler_output)
        return NtaVllmHostTransferMetadata(upstream, tuple(requests))

    def update_output(self, output: KVConnectorOutput) -> None:
        metadata = output.kv_connector_worker_meta
        if not isinstance(metadata, NtaVllmHostWorkerMetadata):
            return
        for request_id, count in metadata.completed_load_requests.items():
            total = self._load_completion_counts.get(request_id, 0) + int(count)
            if total >= self._expected_workers:
                self._load_completion_counts.pop(request_id, None)
                self._manager._cleanup_load_request(request_id)
            else:
                self._load_completion_counts[request_id] = total
        if metadata.completed_store_events:
            self._manager.update_connector_output(
                KVConnectorOutput(
                    kv_connector_worker_meta=SimpleCPUOffloadWorkerMetadata(
                        completed_store_events=dict(metadata.completed_store_events)
                    )
                )
            )

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        self._synchronous_matches.pop(str(request.request_id), None)
        return self._manager.request_finished(request, block_ids)

    def request_finished_all_groups(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        self._synchronous_matches.pop(str(request.request_id), None)
        return self._manager.request_finished_all_groups(request, block_ids)

    def has_pending_stores(self) -> bool:
        return self._manager.has_pending_stores()

    def take_events(self) -> Any:
        return self._manager.take_events()

    def reset(self) -> bool:
        self._synchronous_matches.clear()
        return bool(self._manager.reset())


class VllmHostWorker:
    """Worker-side pinned payload owner; NTA, not vLLM, executes loads."""

    def __init__(
        self, vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"
    ) -> None:
        self._worker = SimpleCPUOffloadWorker(
            vllm_config, kv_cache_config, _capacity_per_rank(vllm_config)
        )
        self._metadata: NtaVllmHostTransferMetadata | None = None
        self._load_events: list[tuple[torch.cuda.Event, tuple[str, ...]]] = []
        self._completed_loads: dict[str, int] = {}
        self._resources: dict[str, IndexedHostResource] = {}
        self._requested_layers_per_wave = vllm_host_layers_per_wave()
        self._wave_groups: tuple[
            tuple[int, int, tuple[StridedCopyGroup, ...]], ...
        ] = ()
        self._acquisition: VllmHostAcquisition | None = None
        self._credits = TenantCreditLedger(tenant_budget_specs())
        # A normal completion is a CUDA event. An exceptional submission whose
        # event publication failed retains the load stream itself as the only
        # safe completion primitive. Both expose query()/synchronize().
        self._pending_credit_releases: list[tuple[Any, TenantCreditLease]] = []

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._worker.register_kv_caches(kv_caches)
        gpu_backings = self._worker.gpu_kv_caches
        cpu_backings = self._worker.cpu_kv_caches
        if gpu_backings is None or cpu_backings is None:
            raise RuntimeError("vLLM host worker did not allocate transfer backings")
        self._resources = build_indexed_host_resources(
            kv_caches, gpu_backings, cpu_backings
        )
        layer_names = tuple(self._resources)
        if tuple(cpu_backings) != layer_names or tuple(gpu_backings) != layer_names:
            raise RuntimeError("vLLM Host backing order disagrees with KV resources")
        layers_per_wave = min(
            self._requested_layers_per_wave or len(layer_names), len(layer_names)
        )
        self._wave_groups = tuple(
            (
                begin,
                min(begin + layers_per_wave, len(layer_names)),
                tuple(
                    StridedCopyGroup(
                        int(self._resources[layer_name].source_tensor.data_ptr())
                        + self._resources[layer_name].source_offset_bytes,
                        int(self._resources[layer_name].destination_tensor.data_ptr()),
                        self._resources[layer_name].source_rows,
                        self._resources[layer_name].destination_rows,
                        self._resources[layer_name].row_bytes,
                        self._resources[layer_name].source_stride_bytes,
                        self._resources[layer_name].destination_stride_bytes,
                    )
                    for layer_name in layer_names[
                        begin : min(begin + layers_per_wave, len(layer_names))
                    ]
                ),
            )
            for begin in range(0, len(layer_names), layers_per_wave)
        )

    @property
    def resources(self) -> dict[str, IndexedHostResource]:
        if not self._resources:
            raise RuntimeError("vLLM host payloads were not registered")
        return self._resources

    def bind(self, metadata: NtaVllmHostTransferMetadata) -> None:
        if not isinstance(metadata, NtaVllmHostTransferMetadata):
            raise TypeError("vLLM Host worker requires typed transfer metadata")
        if self._metadata is not None:
            raise RuntimeError("vLLM host metadata was rebound before finalization")
        upstream = metadata.upstream
        if len(upstream.load_cpu_blocks) != len(upstream.load_gpu_blocks):
            raise RuntimeError("vLLM host source/destination block maps disagree")
        if len(set(upstream.load_gpu_blocks)) != len(upstream.load_gpu_blocks):
            raise RuntimeError("vLLM host load destinations are not unique")
        self._metadata = metadata
        # Preserve vLLM's store path but prevent its conventional CPU->GPU
        # gather from racing NTA's typed in-forward materialization.
        store_only = replace(
            upstream,
            load_event=INVALID_JOB_ID,
            load_gpu_blocks=[],
            load_cpu_blocks=[],
            load_event_to_reqs={},
        )
        self._worker.bind_connector_metadata(store_only)

    @property
    def transfer_pairs(self) -> tuple[tuple[int, int], ...]:
        metadata = self._metadata
        if metadata is None:
            raise RuntimeError("vLLM host transfer metadata is not bound")
        return tuple(
            zip(
                metadata.upstream.load_cpu_blocks,
                metadata.upstream.load_gpu_blocks,
                strict=True,
            )
        )

    @property
    def request_transfers(self) -> tuple[VllmHostRequestTransfer, ...]:
        metadata = self._metadata
        if metadata is None:
            raise RuntimeError("vLLM host transfer metadata is not bound")
        return metadata.requests

    def _collect_credit_releases(self) -> None:
        pending: list[tuple[Any, TenantCreditLease]] = []
        for event, lease in self._pending_credit_releases:
            if event.query():
                self._credits.release(lease)
            else:
                pending.append((event, lease))
        self._pending_credit_releases = pending

    def _reserve_tenant_credits(
        self, request_tenants: Mapping[str, int] | None
    ) -> TenantCreditLease | None:
        if not self._credits.finite:
            return None
        if request_tenants is None:
            raise RuntimeError("finite tenant budgets require request ownership")
        requests = self.request_transfers
        transfer_requests = {request.request_id for request in requests}
        if not transfer_requests.issubset(request_tenants):
            raise RuntimeError("vLLM Host request and tenant ownership disagree")
        layer_bytes = sum(resource.row_bytes for resource in self._resources.values())
        return self._credits.try_reserve(
            TenantCreditCharge(
                int(request_tenants[request.request_id]),
                len(request.pairs) * layer_bytes,
            )
            for request in requests
        )

    def submit_exact_layers(
        self, request_tenants: Mapping[str, int] | None = None
    ) -> VllmHostAcquisition | None:
        """Queue exact layers in transport waves and publish layer fences."""

        pairs = self.transfer_pairs
        if not pairs:
            return None
        if (
            not self._wave_groups
            or self._wave_groups[0][0] != 0
            or self._wave_groups[-1][1] != len(self._resources)
            or any(
                first_end != second_begin
                for (_, first_end, _), (second_begin, _, _) in zip(
                    self._wave_groups, self._wave_groups[1:]
                )
            )
        ):
            raise RuntimeError(
                "vLLM Host acquisition has no complete transport-wave plan"
            )
        if self._acquisition is not None:
            raise RuntimeError("vLLM Host acquisition was submitted twice")
        self._collect_credit_releases()
        credit_lease = self._reserve_tenant_credits(request_tenants)
        if self._credits.finite and credit_lease is None:
            return None
        ordered_pairs = tuple(sorted(pairs, key=lambda pair: pair[1]))
        source_blocks, destination_blocks = zip(*ordered_pairs, strict=True)
        layout = analyze_index_pairs(source_blocks, destination_blocks)
        layer_names = tuple(self._resources)
        lifecycle = LayerAcquisition(
            tuple(
                len(pairs) * self._resources[layer_name].row_bytes
                for layer_name in layer_names
            )
        )
        published: dict[int, torch.cuda.Event] = {}
        layer_fences: dict[int, int] = {}
        events: list[torch.cuda.Event] = []
        copy_submissions = 0
        transport_attempted = False

        def publish_range(begin: int, end: int) -> None:
            nonlocal copy_submissions, transport_attempted
            for wave_begin, wave_end, groups in self._wave_groups:
                if wave_begin < begin or wave_end > end:
                    continue
                # Set this before entering the native helper: a later native
                # batch or CUDA-event publication may fail after earlier DMA
                # has already entered the stream.
                transport_attempted = True
                copy_submissions += copy_strided_host_runs_async(
                    groups, layout.runs, self._worker.load_stream
                )
                event = torch.cuda.Event()
                event.record(self._worker.load_stream)
                fence = len(events)
                events.append(event)
                for layer in range(wave_begin, wave_end):
                    published[layer] = event
                    layer_fences[layer] = fence

        try:
            submission_begin_ns = time.perf_counter_ns()
            submission = lifecycle.submit_available(
                publish_range=publish_range,
                published_layers=published,
            )
            submission_cpu_ns = time.perf_counter_ns() - submission_begin_ns
            if submission.job_count != len(layer_names):
                raise RuntimeError(
                    "vLLM Host acquisition did not fill its finite layer queue"
                )
            acquisition = VllmHostAcquisition(
                layer_names,
                tuple(layer_fences[layer] for layer in range(len(layer_names))),
                tuple(events),
                lifecycle,
                transfer_blocks=len(pairs) * len(layer_names),
                transfer_bytes=len(pairs)
                * sum(resource.row_bytes for resource in self._resources.values()),
                transfer_runs=len(layout.runs),
                copy_operations=len(layout.runs) * len(layer_names),
                copy_submissions=copy_submissions,
                submission_cpu_ns=max(1, submission_cpu_ns),
                credit_lease=credit_lease,
            )
        except BaseException as error:
            completion = self._worker.load_stream if transport_attempted else None
            cleanup_completed = completion is None
            if completion is not None:
                try:
                    completion.synchronize()
                    cleanup_completed = True
                except BaseException as synchronization_error:
                    error.add_note(
                        "vLLM Host acquisition cleanup also failed: "
                        f"{synchronization_error!r}"
                    )
            lifecycle.cancel_unfinished()
            if credit_lease is not None:
                if cleanup_completed:
                    self._credits.release(credit_lease)
                else:
                    self._pending_credit_releases.append((completion, credit_lease))
            raise
        self._acquisition = acquisition
        return acquisition

    def handle_preemptions(self, metadata: NtaVllmHostTransferMetadata) -> None:
        if not isinstance(metadata, NtaVllmHostTransferMetadata):
            raise TypeError("vLLM Host preemption metadata is not typed")
        self._worker.handle_preemptions(metadata.upstream)

    def finish(self, finished_request_ids: set[str]) -> None:
        metadata = self._metadata
        if metadata is None:
            return
        upstream = metadata.upstream
        sending, receiving = self._worker.get_finished(finished_request_ids)
        if sending or receiving:
            raise RuntimeError("store-only vLLM host worker reported a load completion")
        if upstream.load_cpu_blocks:
            request_ids = tuple(
                upstream.load_event_to_reqs.get(upstream.load_event, ())
            )
            if not request_ids:
                raise RuntimeError("vLLM host load has no owning request IDs")
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
            self._load_events.append((event, request_ids))
        while self._load_events and self._load_events[0][0].query():
            _, request_ids = self._load_events.pop(0)
            for request_id in request_ids:
                self._completed_loads[request_id] = 1

    def build_worker_metadata(self) -> NtaVllmHostWorkerMetadata | None:
        store_metadata = self._worker.build_connector_worker_meta()
        stores = (
            {}
            if store_metadata is None
            else dict(store_metadata.completed_store_events)
        )
        if not self._completed_loads and not stores:
            return None
        result = NtaVllmHostWorkerMetadata(dict(self._completed_loads), stores)
        self._completed_loads.clear()
        return result

    def clear(self) -> None:
        if self._acquisition is not None and not self._acquisition.terminal:
            raise RuntimeError(
                "vLLM Host acquisition cleared before every layer was consumed"
            )
        if self._acquisition is not None:
            release = self._acquisition.take_credit_release()
            if release is not None:
                event, lease = release
                if event.query():
                    self._credits.release(lease)
                else:
                    self._pending_credit_releases.append(release)
        self._worker.clear_connector_metadata()
        self._metadata = None
        self._acquisition = None

    def abort(self) -> None:
        failure: BaseException | None = None
        if self._acquisition is not None:
            self._acquisition.cancel_unfinished()
            try:
                self._acquisition.synchronize()
                release = self._acquisition.take_credit_release()
                if release is not None:
                    self._credits.release(release[1])
            except BaseException as error:
                failure = error
        try:
            self._worker.clear_connector_metadata()
        except BaseException as error:
            if failure is None:
                failure = error
            else:
                failure.add_note(f"vLLM Host metadata cleanup also failed: {error!r}")
        self._metadata = None
        self._acquisition = None
        if failure is not None:
            raise failure

    def shutdown(self) -> None:
        if self._acquisition is not None:
            self._acquisition.cancel_unfinished()
            self._acquisition.synchronize()
            release = self._acquisition.take_credit_release()
            if release is not None:
                self._credits.release(release[1])
            self._acquisition = None
        for event, lease in self._pending_credit_releases:
            event.synchronize()
            self._credits.release(lease)
        self._pending_credit_releases.clear()
        for event, _ in self._load_events:
            event.synchronize()
        self._load_events.clear()
        self._worker._flush_and_sync_all()
        self._resources.clear()
        self._wave_groups = ()
        if self._credits.active_lease_count:
            raise RuntimeError("vLLM Host worker leaked tenant credit leases")
