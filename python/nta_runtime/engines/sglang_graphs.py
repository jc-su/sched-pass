"""Finite CUDA graph ownership for SGLang's typed attention operator."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import operator
from typing import Any

import torch

from nta_runtime.runtime import DeviceWorkPlan


@dataclass(frozen=True)
class DemandGraphKey:
    operator_family: str
    wrapper_id: int
    layer_id: int
    stream_address: int
    work_items_address: int
    dependencies_address: int
    runtime_address: int
    work_count: int
    object_count: int
    progress_blocks: tuple[int, ...]
    ready_work_counts: tuple[int, ...]
    ready_work_offsets: tuple[int, ...]
    initial_ready_work_count: int
    indexed_copy_blocks_per_group: int
    query_shape: tuple[int, ...]
    query_stride: tuple[int, ...]
    query_dtype: str
    query_device: str
    key_cache_address: int
    value_cache_address: int
    sm_scale: float
    k_scale: float | None
    v_scale: float | None
    causal: bool
    window_left: int
    wrapper_plan: Any
    wrapper_metadata_layout: tuple[
        tuple[str, tuple[int, ...], tuple[int, ...], str, str], ...
    ]


@dataclass(frozen=True)
class _DemandGraph:
    graph: torch.cuda.CUDAGraph
    query: torch.Tensor
    output: torch.Tensor
    retained_events: tuple[torch.cuda.Event, ...]
    wrapper_metadata: tuple[tuple[str, torch.Tensor], ...]


_WRAPPER_METADATA = (
    "_qo_indptr_buf",
    "_paged_kv_indptr_buf",
    "_paged_kv_indices_buf",
    "_paged_kv_last_page_len_buf",
    "_custom_mask_buf",
    "_mask_indptr_buf",
    "_prefix_len_ptr",
    "_token_pos_in_items_ptr",
    "_max_item_len_ptr",
    "_block_tables",
)


def _freeze_plan(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    try:
        return operator.index(value)
    except TypeError:
        pass
    if isinstance(value, Iterable):
        return tuple(_freeze_plan(item) for item in value)
    raise RuntimeError(
        f"FlashInfer graph plan contains unsupported {type(value).__name__} state"
    )


def wrapper_metadata(wrapper: Any) -> tuple[tuple[str, torch.Tensor], ...]:
    return tuple(
        (name, value)
        for name in _WRAPPER_METADATA
        if torch.is_tensor(value := getattr(wrapper, name, None))
    )


def demand_graph_key(
    *,
    operator_family: str,
    wrapper: Any,
    layer_id: int,
    stream_address: int,
    plan: DeviceWorkPlan,
    runtime_tensor: torch.Tensor,
    work_count: int,
    object_count: int,
    progress_blocks: tuple[int, ...],
    ready_work_counts: tuple[int, ...],
    ready_work_offsets: tuple[int, ...],
    initial_ready_work_count: int,
    indexed_copy_blocks_per_group: int,
    query: torch.Tensor,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
    sm_scale: float,
    k_scale: float | None,
    v_scale: float | None,
    causal: bool,
    window_left: int,
) -> DemandGraphKey:
    """Describe every dynamic value baked into one graph launch."""

    if operator_family not in {"decode", "paged_prefill"}:
        raise ValueError("unsupported demand graph operator family")
    metadata_layout = tuple(
        (
            name,
            tuple(int(extent) for extent in value.shape),
            tuple(int(stride) for stride in value.stride()),
            str(value.dtype),
            str(value.device),
        )
        for name, value in wrapper_metadata(wrapper)
    )
    return DemandGraphKey(
        operator_family,
        id(wrapper),
        int(layer_id),
        int(stream_address),
        int(plan.work_items_address),
        int(plan.dependencies_address),
        int(runtime_tensor.data_ptr()),
        int(work_count),
        int(object_count),
        tuple(int(count) for count in progress_blocks),
        tuple(int(count) for count in ready_work_counts),
        tuple(int(offset) for offset in ready_work_offsets),
        int(initial_ready_work_count),
        int(indexed_copy_blocks_per_group),
        tuple(int(extent) for extent in query.shape),
        tuple(int(stride) for stride in query.stride()),
        str(query.dtype),
        str(query.device),
        int(kv_cache[0].data_ptr()),
        int(kv_cache[1].data_ptr()),
        float(sm_scale),
        None if k_scale is None else float(k_scale),
        None if v_scale is None else float(v_scale),
        bool(causal),
        int(window_left),
        _freeze_plan(getattr(wrapper, "_plan_info", None)),
        metadata_layout,
    )


class DemandGraphCache:
    """Own a bounded LRU-like set of stream-qualified CUDA graphs."""

    def __init__(self, *, capacity: int, stats: dict[str, Any]) -> None:
        if capacity <= 0:
            raise ValueError("demand graph capacity must be positive")
        self._capacity = capacity
        self._stats = stats
        self._graphs: dict[DemandGraphKey, _DemandGraph] = {}
        self._warmups: dict[DemandGraphKey, None] = {}

    def clear(self) -> None:
        self._graphs.clear()
        self._warmups.clear()

    def contains(self, key: DemandGraphKey) -> bool:
        return key in self._graphs

    def tracks(self, key: DemandGraphKey) -> bool:
        """Return whether a key owns either a warmup or captured entry."""

        return key in self._warmups

    def captured(self, key: DemandGraphKey) -> _DemandGraph | None:
        return self._graphs.get(key)

    def discard_plan(self, plan: DeviceWorkPlan) -> None:
        """Drop executables before releasing their captured plan buffers."""

        work_items_address = int(plan.work_items_address)
        dependencies_address = int(plan.dependencies_address)
        stale = {
            key
            for key in self._warmups
            if key.work_items_address == work_items_address
            and key.dependencies_address == dependencies_address
        }
        for key in stale:
            self._graphs.pop(key, None)
            self._warmups.pop(key, None)

    def reserve(self, key: DemandGraphKey, stream: torch.cuda.Stream) -> None:
        """Reserve cache state, quiescing before captured pointers are freed."""

        if key in self._warmups:
            self._warmups.pop(key)
            self._warmups[key] = None
            return
        if len(self._warmups) >= self._capacity:
            stream.synchronize()
            stale = next(iter(self._warmups))
            self._warmups.pop(stale)
            self._graphs.pop(stale, None)
            self._stats["demand_graph_evictions"] += 1
        self._warmups[key] = None

    def enqueue(
        self,
        key: DemandGraphKey,
        wrapper: Any,
        query: torch.Tensor,
        output: torch.Tensor,
        stream: torch.cuda.Stream,
        enqueue: Callable[
            [
                torch.Tensor,
                torch.Tensor,
                tuple[torch.cuda.Event, tuple[torch.cuda.Event, ...]],
                Callable[[Any], None] | None,
            ],
            None,
        ],
        eager_events: tuple[torch.cuda.Event, tuple[torch.cuda.Event, ...]],
        on_discovered: Callable[[Any], None] | None,
        on_replayed: Callable[[], None] | None,
    ) -> torch.Tensor:
        """Warm, capture, or replay one finite incremental operator."""

        captured = self._graphs.get(key)
        if captured is not None:
            self.reserve(key, stream)
            current_metadata = dict(wrapper_metadata(wrapper))
            for name, static_tensor in captured.wrapper_metadata:
                current = current_metadata.get(name)
                if current is None:
                    raise RuntimeError(
                        f"FlashInfer graph replay lost metadata tensor {name}"
                    )
                if (
                    current.shape != static_tensor.shape
                    or current.stride() != static_tensor.stride()
                    or current.dtype != static_tensor.dtype
                    or current.device != static_tensor.device
                ):
                    raise RuntimeError(
                        f"FlashInfer graph metadata layout changed for {name}"
                    )
                static_tensor.copy_(current, non_blocking=True)
            captured.query.copy_(query, non_blocking=True)
            captured.graph.replay()
            if on_replayed is not None:
                on_replayed()
            self._record("replays", key.operator_family)
            return captured.output

        if key not in self._warmups:
            enqueue(query, output, eager_events, on_discovered)
            self.reserve(key, stream)
            self._record("warmups", key.operator_family)
            return output

        self.reserve(key, stream)
        static_query = torch.empty_like(query)
        static_output = torch.empty_like(output)
        static_query.copy_(query, non_blocking=True)
        discovery_done = torch.cuda.Event()
        arrival_events = tuple(
            torch.cuda.Event() for _ in range(len(key.progress_blocks))
        )
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream, capture_error_mode="global"):
                enqueue(
                    static_query,
                    static_output,
                    (discovery_done, arrival_events),
                    None,
                )
        except Exception as error:
            raise RuntimeError(
                "failed to capture the finite NTA demand operator graph"
            ) from error
        captured = _DemandGraph(
            graph,
            static_query,
            static_output,
            (discovery_done, *arrival_events),
            wrapper_metadata(wrapper),
        )
        self._graphs[key] = captured
        self._record("captures", key.operator_family)
        # Capture instantiates the executable but does not produce this call's
        # output, so replay it once in the current stream order.
        graph.replay()
        if on_replayed is not None:
            on_replayed()
        self._record("replays", key.operator_family)
        return static_output

    def _record(self, action: str, family: str) -> None:
        self._stats[f"demand_graph_{action}"] += 1
        family_counter = f"demand_graph_{family}_{action}"
        self._stats[family_counter] = self._stats.get(family_counter, 0) + 1
