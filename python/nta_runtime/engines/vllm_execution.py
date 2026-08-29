"""Typed preparation objects for the vLLM attention execution boundary.

This module contains framework-shape validation and exact host dependency
construction.  Native runtime publication remains in ``engines.vllm`` so the
allocation/lifetime owner and the steady-state submission owner stay explicit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from nta_runtime.adapters.base import EngineBatch
from nta_runtime.execution_core import ExecutionSession
from nta_runtime.execution_topology import ExactWorkTopology
from nta_runtime.indexed_transfer import (
    AcquisitionGroup,
    AcquisitionSlice,
    AcquisitionTopology,
    IndexedDependencyLayout,
    IndexedHostResource,
    plan_indexed_dependencies,
)


PhysicalPages = Callable[[EngineBatch, Any, int, int, int], tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class VllmHostPreparation:
    """Validated source geometry and exact dependencies for one layer."""

    layer_name: str
    resource: IndexedHostResource
    source: torch.Tensor
    layout: IndexedDependencyLayout
    source_indices: torch.Tensor | None
    destination_indices: torch.Tensor | None

    def __post_init__(self) -> None:
        has_layout = bool(self.layout.runs)
        if has_layout != (self.source_indices is not None):
            raise ValueError("vLLM host source index ownership is incomplete")
        if has_layout != (self.destination_indices is not None):
            raise ValueError("vLLM host destination index ownership is incomplete")

    @property
    def transfer_blocks(self) -> int:
        return len(self.layout.source_indices)

    @property
    def transfer_bytes(self) -> int:
        return self.transfer_blocks * self.resource.row_bytes

    def acquisition_topology(self) -> AcquisitionTopology:
        if not self.layout.runs:
            raise RuntimeError("resident vLLM host work has no acquisition topology")
        return AcquisitionTopology(
            len(self.layout.source_indices),
            tuple(
                AcquisitionGroup(run.pair_offset, run.row_count)
                for run in self.layout.runs
            ),
            tuple(
                tuple(
                    AcquisitionSlice(
                        run_index,
                        0,
                        self.layout.runs[run_index].row_count,
                    )
                    for run_index in run_indices
                )
                for run_indices in self.layout.run_indices_by_work
            ),
        )


@dataclass(frozen=True, slots=True)
class VllmScheduleContext:
    """Validated identity/topology state shared by upload and submission."""

    physical: bool
    host_staged: bool
    owner: Any
    destination: Any | None
    semantic_layer: int
    topology: ExactWorkTopology
    verifier: ExecutionSession | None


@dataclass(frozen=True, slots=True)
class VllmSchedulePublication:
    """One uploaded device plan and its external-publication ownership."""

    plan: Any
    object_count: int
    has_external_transfer: bool
    tier: Any | None = None

    def __post_init__(self) -> None:
        if self.object_count < 0:
            raise ValueError("vLLM schedule object count cannot be negative")
        if self.has_external_transfer != (self.object_count > 0):
            raise ValueError("vLLM external publication has inconsistent object count")


@dataclass(frozen=True, slots=True)
class _VllmHostResourceBinding:
    layer_name: str
    resource: IndexedHostResource
    source: torch.Tensor


def _require_host_resource(
    state: Any, layer: Any, kv_cache: torch.Tensor
) -> _VllmHostResourceBinding:
    layer_name = getattr(layer, "layer_name", None)
    resources = getattr(state, "host_resources", None)
    if not isinstance(layer_name, str) or not layer_name:
        raise RuntimeError("vLLM host-staged attention has no stable layer name")
    if not isinstance(resources, dict):
        raise RuntimeError("vLLM host-staged attention has no resource directory")
    resource = resources.get(layer_name)
    if not isinstance(resource, IndexedHostResource):
        raise RuntimeError(
            f"vLLM host cache has no typed payload for layer {layer_name!r}"
        )
    source = resource.source_tensor
    destination = resource.destination_tensor
    if (
        not isinstance(source, torch.Tensor)
        or source.is_cuda
        or not source.is_pinned()
        or source.ndim < 2
        or not source.is_contiguous()
    ):
        raise RuntimeError("vLLM host source must own contiguous pinned rows")
    if (
        not isinstance(destination, torch.Tensor)
        or not destination.is_cuda
        or not isinstance(kv_cache, torch.Tensor)
        or not kv_cache.is_cuda
        or int(destination.data_ptr()) != int(kv_cache.data_ptr())
        or int(destination.shape[0]) != int(kv_cache.shape[0])
        or int(kv_cache.stride(0)) * int(kv_cache.element_size())
        != resource.destination_stride_bytes
    ):
        raise RuntimeError(
            "vLLM host resource does not name the numerical KV destination"
        )
    return _VllmHostResourceBinding(layer_name, resource, source)


def _host_dependency_layout(
    *,
    state: Any,
    batch: EngineBatch,
    schedule: Any,
    page_size: int,
    binding: _VllmHostResourceBinding,
    physical_pages: PhysicalPages,
) -> IndexedDependencyLayout:
    pairs = tuple(getattr(state, "host_transfer_pairs", ()))
    resource = binding.resource
    source_by_destination = {
        destination_index: source_index for source_index, destination_index in pairs
    }
    if len(source_by_destination) != len(pairs):
        raise RuntimeError("vLLM host transfer destinations are not unique")
    if any(
        source_index < 0
        or source_index >= resource.source_rows
        or destination_index < 0
        or destination_index >= resource.destination_rows
        for destination_index, source_index in source_by_destination.items()
    ):
        raise RuntimeError("vLLM host transfer exceeds source/destination blocks")

    consumed = state.consumed_host_destinations(binding.layer_name)
    work_pairs: list[tuple[tuple[int, int], ...]] = []
    selected_pages_by_work: list[tuple[int, ...]] = []
    for request_index, kv_tile in zip(
        schedule.request_indices, schedule.kv_tile_indices, strict=True
    ):
        pages = physical_pages(
            batch, schedule, int(request_index), int(kv_tile), page_size
        )
        selected_pages_by_work.append(pages)
        work_pairs.append(
            tuple(
                (source_by_destination[page], page)
                for page in pages
                if page in source_by_destination and page not in consumed
            )
        )
    state.record_host_schedule(
        binding.layer_name,
        int(getattr(schedule, "kv_chunk_tokens", 0)),
        tuple(int(value) for value in schedule.request_indices),
        tuple(int(value) for value in schedule.kv_tile_indices),
        tuple(selected_pages_by_work),
    )
    return plan_indexed_dependencies(work_pairs)


def _host_index_tensors(
    state: Any, layout: IndexedDependencyLayout, device: torch.device
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not layout.runs:
        return None, None
    device_index = (
        torch.cuda.current_device() if device.index is None else int(device.index)
    )
    index_key = (device_index, layout.source_indices, layout.destination_indices)
    tensors = state.host_index_tensors.get(index_key)
    if tensors is None:
        tensors = (
            torch.tensor(layout.source_indices, dtype=torch.int32, device=device),
            torch.tensor(layout.destination_indices, dtype=torch.int32, device=device),
        )
        state.host_index_tensors[index_key] = tensors
    return tensors


def prepare_host_layer(
    *,
    state: Any,
    batch: EngineBatch,
    schedule: Any,
    layer: Any,
    kv_cache: torch.Tensor,
    page_size: int,
    object_capacity: int,
    physical_pages: PhysicalPages,
) -> VllmHostPreparation:
    """Validate one host resource and construct its exact work dependencies."""

    binding = _require_host_resource(state, layer, kv_cache)
    layout = _host_dependency_layout(
        state=state,
        batch=batch,
        schedule=schedule,
        page_size=page_size,
        binding=binding,
        physical_pages=physical_pages,
    )
    if len(layout.runs) > object_capacity:
        raise RuntimeError("vLLM host layer exceeds runtime object capacity")
    source_indices, destination_indices = _host_index_tensors(
        state, layout, kv_cache.device
    )
    return VllmHostPreparation(
        binding.layer_name,
        binding.resource,
        binding.source,
        layout,
        source_indices,
        destination_indices,
    )


@dataclass(frozen=True, slots=True)
class VllmPrefillBuffers:
    qo_indptr: torch.Tensor
    indptr: torch.Tensor
    indices: torch.Tensor
    last_page_len: torch.Tensor


def require_prefill_buffers(wrapper: Any, request_count: int) -> VllmPrefillBuffers:
    buffers = tuple(
        getattr(wrapper, name, None)
        for name in (
            "_qo_indptr_buf",
            "_paged_kv_indptr_buf",
            "_paged_kv_indices_buf",
            "_paged_kv_last_page_len_buf",
        )
    )
    if not all(isinstance(tensor, torch.Tensor) for tensor in buffers):
        raise RuntimeError(
            "vLLM FlashInfer prefill metadata has no typed paged-KV buffers"
        )
    qo_indptr, indptr, indices, last_page_len = buffers
    if any(
        tensor.dtype != torch.int32
        or not tensor.is_cuda
        or not tensor.is_contiguous()
        for tensor in buffers
    ):
        raise RuntimeError(
            "vLLM FlashInfer prefill buffers must be contiguous CUDA int32 tensors"
        )
    if qo_indptr.numel() != request_count + 1:
        raise RuntimeError("vLLM FlashInfer prefill has the wrong request count")
    return VllmPrefillBuffers(qo_indptr, indptr, indices, last_page_len)


@dataclass(frozen=True, slots=True)
class VllmDecodeBuffers:
    indptr: torch.Tensor
    indices: torch.Tensor
    last_page_len: torch.Tensor


def require_decode_buffers(wrapper: Any, request_count: int) -> VllmDecodeBuffers:
    buffers = tuple(
        getattr(wrapper, name, None)
        for name in (
            "_paged_kv_indptr_buf",
            "_paged_kv_indices_buf",
            "_paged_kv_last_page_len_buf",
        )
    )
    if not all(isinstance(tensor, torch.Tensor) for tensor in buffers):
        raise RuntimeError(
            "vLLM FlashInfer metadata has no typed paged-KV device buffers"
        )
    indptr, indices, last_page_len = buffers
    if indptr.numel() != request_count + 1:
        raise RuntimeError("vLLM FlashInfer page indptr has the wrong request count")
    if any(
        tensor.dtype != torch.int32
        or not tensor.is_cuda
        or not tensor.is_contiguous()
        for tensor in buffers
    ):
        raise RuntimeError(
            "vLLM FlashInfer paged-KV buffers must be contiguous CUDA int32 tensors"
        )
    if indices.numel() <= 0:
        raise RuntimeError("vLLM FlashInfer page-index buffer is incomplete")
    return VllmDecodeBuffers(
        indptr, indices, last_page_len[:request_count]
    )


__all__ = [
    "VllmDecodeBuffers",
    "VllmHostPreparation",
    "VllmPrefillBuffers",
    "VllmScheduleContext",
    "VllmSchedulePublication",
    "prepare_host_layer",
    "require_decode_buffers",
    "require_prefill_buffers",
]
