"""Version-checked extraction of CTA identity from FlashInfer plans."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
from typing import Any


SUPPORTED_VERSION = "0.6.12"


@dataclass(frozen=True)
class Schedule:
    request_indices: tuple[int, ...]
    kv_tile_indices: tuple[int, ...]
    kv_chunk_tokens: int
    padded_work_count: int

    @property
    def work_count(self) -> int:
        return len(self.request_indices)


def require_supported_version() -> None:
    version = importlib.metadata.version("flashinfer-python")
    if version != SUPPORTED_VERSION:
        raise RuntimeError(
            f"unsupported FlashInfer {version}; expected {SUPPORTED_VERSION}"
        )


def _read_i32(workspace: Any, byte_offset: int, count: int) -> list[int]:
    import torch

    if byte_offset < 0 or byte_offset % 4 != 0 or count < 0:
        raise RuntimeError("invalid FlashInfer int-workspace range")
    end = byte_offset + count * 4
    if workspace.dtype != torch.uint8 or end > workspace.numel():
        raise RuntimeError("FlashInfer int-workspace range is out of bounds")
    return workspace[byte_offset:end].view(torch.int32).cpu().tolist()


def _extract(
    wrapper: object,
    expected_size: int,
    request_offset_index: int,
    kv_tile_offset_index: int,
    chunk_offset_index: int,
    mask_offset_index: int,
    graph_index: int,
    split_index: int,
    *,
    allow_zero_chunk: bool = False,
) -> Schedule:
    require_supported_version()
    plan = list(getattr(wrapper, "_plan_info"))
    workspace = getattr(wrapper, "_int_workspace_buffer")
    if len(plan) != expected_size:
        raise RuntimeError(
            f"unexpected FlashInfer PlanInfo length {len(plan)}; expected {expected_size}"
        )
    padded = int(plan[0])
    if padded <= 0:
        raise RuntimeError("FlashInfer produced an empty scheduler grid")
    requests = _read_i32(workspace, int(plan[request_offset_index]), padded)
    kv_tiles = _read_i32(workspace, int(plan[kv_tile_offset_index]), padded)
    active = [True] * padded
    if bool(plan[graph_index]) and bool(plan[split_index]):
        mask_offset = int(plan[mask_offset_index])
        if mask_offset < 0 or mask_offset + padded > workspace.numel():
            raise RuntimeError("FlashInfer block-valid mask is out of bounds")
        active = [
            bool(value)
            for value in workspace[mask_offset : mask_offset + padded].cpu()
        ]
    request_indices = tuple(
        value for value, valid in zip(requests, active) if valid
    )
    kv_tile_indices = tuple(
        value for value, valid in zip(kv_tiles, active) if valid
    )
    chunk_tokens = _read_i32(workspace, int(plan[chunk_offset_index]), 1)[0]
    if allow_zero_chunk and not bool(plan[split_index]):
        chunk_tokens = 0
    elif chunk_tokens <= 0:
        raise RuntimeError("FlashInfer KV chunk size must be positive")
    return Schedule(request_indices, kv_tile_indices, chunk_tokens, padded)


def decode_schedule(wrapper: object) -> Schedule:
    if bool(getattr(wrapper, "use_tensor_cores")):
        return paged_prefill_schedule(wrapper)
    return _extract(wrapper, 10, 3, 4, 7, 6, 8, 9)


def paged_prefill_schedule(wrapper: object) -> Schedule:
    # With split-K disabled, 0.6.12 narrows INT64_MAX * page_size to IdType
    # zero. The scheduler still emits one KV tile for each Q tile.
    return _extract(
        wrapper, 15, 4, 6, 9, 12, 13, 14, allow_zero_chunk=True
    )
