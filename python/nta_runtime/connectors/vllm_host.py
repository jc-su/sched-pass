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

from dataclasses import dataclass, replace
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
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
    BatchMemcpyParams,
    build_params,
    copy_blocks,
)

from nta_runtime.indexed_transfer import IndexedHostResource

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


DEFAULT_CPU_CAPACITY_BYTES = 8 * 1024**3


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
        if (
            block_size <= 0
            or local_tokens % block_size
            or external_tokens % block_size
        ):
            raise RuntimeError(
                "vLLM host token interval is not aligned to cache group "
                f"{group_index}"
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
            for extent, stride in zip(tensor.shape[1:], tensor.stride()[1:], strict=True)
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
            raise RuntimeError(
                f"vLLM layer {layer_name!r} has ambiguous host backing"
            )
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
            raise RuntimeError("vLLM host cache no longer exposes load ownership cleanup")
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
                    cpu_block_ids=[int(block.block_id) for block in selected_cpu_blocks],
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

    def build_metadata(self, scheduler_output: Any) -> SimpleCPUOffloadMetadata:
        return self._manager.build_connector_meta(scheduler_output)

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
        self._metadata: SimpleCPUOffloadMetadata | None = None
        self._load_events: list[tuple[torch.cuda.Event, tuple[str, ...]]] = []
        self._completed_loads: dict[str, int] = {}
        self._resources: dict[str, IndexedHostResource] = {}
        self._preload_params: BatchMemcpyParams | None = None
        self._preload_event: torch.cuda.Event | None = None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._worker.register_kv_caches(kv_caches)
        gpu_backings = self._worker.gpu_kv_caches
        cpu_backings = self._worker.cpu_kv_caches
        if gpu_backings is None or cpu_backings is None:
            raise RuntimeError("vLLM host worker did not allocate transfer backings")
        self._resources = build_indexed_host_resources(
            kv_caches, gpu_backings, cpu_backings
        )
        self._preload_params = build_params(
            cpu_backings,
            gpu_backings,
            self._worker.load_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
        )

    @property
    def resources(self) -> dict[str, IndexedHostResource]:
        if not self._resources:
            raise RuntimeError("vLLM host payloads were not registered")
        return self._resources

    def bind(self, metadata: SimpleCPUOffloadMetadata) -> None:
        if self._metadata is not None:
            raise RuntimeError("vLLM host metadata was rebound before finalization")
        if len(metadata.load_cpu_blocks) != len(metadata.load_gpu_blocks):
            raise RuntimeError("vLLM host source/destination block maps disagree")
        if len(set(metadata.load_gpu_blocks)) != len(metadata.load_gpu_blocks):
            raise RuntimeError("vLLM host load destinations are not unique")
        self._metadata = metadata
        # Preserve vLLM's store path but prevent its conventional CPU->GPU
        # gather from racing NTA's typed in-forward materialization.
        store_only = replace(
            metadata,
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
            zip(metadata.load_cpu_blocks, metadata.load_gpu_blocks, strict=True)
        )

    def preload_exact(self) -> torch.cuda.Event | None:
        """Submit one exact all-layer CPU-to-HBM batch before model execution."""

        pairs = self.transfer_pairs
        if not pairs:
            return None
        if self._preload_params is None or not self._resources:
            raise RuntimeError("vLLM host preload has no registered typed resources")
        if self._preload_event is not None:
            raise RuntimeError("vLLM host preload was submitted twice")
        source_blocks, destination_blocks = zip(*pairs, strict=True)
        copy_blocks(
            list(source_blocks),
            list(destination_blocks),
            self._preload_params,
        )
        event = torch.cuda.Event()
        event.record(self._worker.load_stream)
        self._preload_event = event
        return event

    def handle_preemptions(self, metadata: SimpleCPUOffloadMetadata) -> None:
        self._worker.handle_preemptions(metadata)

    def finish(self, finished_request_ids: set[str]) -> None:
        metadata = self._metadata
        if metadata is None:
            return
        sending, receiving = self._worker.get_finished(finished_request_ids)
        if sending or receiving:
            raise RuntimeError("store-only vLLM host worker reported a load completion")
        if metadata.load_cpu_blocks:
            request_ids = tuple(metadata.load_event_to_reqs.get(metadata.load_event, ()))
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
        self._worker.clear_connector_metadata()
        self._metadata = None
        self._preload_event = None

    def shutdown(self) -> None:
        if self._preload_event is not None:
            self._preload_event.synchronize()
            self._preload_event = None
        for event, _ in self._load_events:
            event.synchronize()
        self._load_events.clear()
        self._worker._flush_and_sync_all()
        self._resources.clear()
        self._preload_params = None
