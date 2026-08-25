"""Engine-neutral state records used by the SGLang attention adapter.

This module contains no SGLang imports.  It owns only lifetime-safe state
records, graph-cache keys, and the asynchronous statistics publisher used by
the framework-facing attention implementation.  Keeping these records out of
the numerical adapter makes the framework boundary auditable and prevents a
state object from accidentally becoming a second execution protocol.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
import json
import operator
import pathlib
import threading
from typing import Any

import torch

from nta_runtime.engines.sglang_hicache import PendingHostLoad
from nta_runtime.execution_core import ExecutionSession
from nta_runtime.execution_planner import HostExecutionPlan
from nta_runtime.flashinfer_schedule import Schedule
from nta_runtime.requests import RequestBinding
from nta_runtime.runtime import DeviceWorkPlan


_PagePair = tuple[tuple[int, ...], tuple[int, ...]]


class _StatsPublisher:
    """Coalesce evaluation snapshots and write them off the scheduler thread."""

    def __init__(self, path: pathlib.Path) -> None:
        self._path = path
        self._condition = threading.Condition()
        self._pending: tuple[int, dict[str, Any]] | None = None
        self._submitted = 0
        self._completed = 0
        self._error: Exception | None = None
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run, name="nta-stats-publisher", daemon=True
        )
        self._thread.start()

    def publish(self, report: dict[str, Any], *, wait: bool = False) -> None:
        with self._condition:
            if self._stopping:
                raise RuntimeError("NTA engine statistics publisher is closed")
            if self._error is not None:
                raise RuntimeError("NTA engine statistics publisher failed") from self._error
            self._submitted += 1
            sequence = self._submitted
            self._pending = (sequence, report)
            self._condition.notify()
            if wait:
                self._condition.wait_for(
                    lambda: self._completed >= sequence or self._error is not None
                )
                if self._error is not None:
                    raise RuntimeError(
                        "failed to publish NTA engine statistics"
                    ) from self._error

    def close(self) -> None:
        """Stop the writer thread after all already-published work settles."""
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._pending is not None or self._stopping
                )
                if self._pending is None and self._stopping:
                    return
                sequence, report = self._pending
                self._pending = None
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self._path.with_suffix(self._path.suffix + ".tmp")
                temporary.write_text(
                    json.dumps(report, sort_keys=True) + "\n", encoding="utf-8"
                )
                temporary.replace(self._path)
            except Exception as error:
                with self._condition:
                    self._error = error
                    self._condition.notify_all()
                return
            with self._condition:
                self._completed = sequence
                self._condition.notify_all()


@dataclass(frozen=True)
class _PrefetchedLayer:
    key_slot: int
    key_object_id: int
    value_slot: int
    value_object_id: int
    version: int
    key_bytes: int
    value_bytes: int
    ready_event: torch.cuda.Event
    transfer_first_slot: int


@dataclass(frozen=True)
class _FragmentLookahead:
    layer_id: int
    wrapper_id: int
    object_count: int
    preloaded_object_count: int
    key_source: int
    key_staging: int
    value_source: int
    value_staging: int
    ready_event: torch.cuda.Event


@dataclass
class _ActiveBatch:
    bindings: tuple[RequestBinding, ...]
    schedules: dict[int, Schedule]
    pending_host_load: PendingHostLoad | None
    page_pairs: dict[int, tuple[_PagePair, ...]]
    index_maps: dict[_PagePair, tuple[torch.Tensor, torch.Tensor]]
    prefetched_layers: dict[int, _PrefetchedLayer]
    prefetch_tensors: tuple[torch.Tensor, ...]
    host_execution: HostExecutionPlan | None = None
    grouping: str = "request"
    fragment_lookahead: dict[int, _FragmentLookahead] = field(default_factory=dict)
    execution: ExecutionSession | None = None


@dataclass
class _PlanAllocation:
    plan: DeviceWorkPlan
    work_capacity: int
    signature: tuple[Any, ...] | None = None
    object_count: int = 0
    index_tensors: tuple[torch.Tensor, ...] = ()
    host_execution: HostExecutionPlan | None = None
    object_version: int = 0
    transfer_bytes: int = 0
    indexed_geometry: tuple[int, ...] | None = None
    max_object_fanout: int = 1
    min_unresolved_dependencies: int = 1
    direct_work_count: int = 0
    external_object_slots: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class _DemandGraph:
    graph: torch.cuda.CUDAGraph
    query: torch.Tensor
    output: torch.Tensor
    retained_events: tuple[torch.cuda.Event, ...]
    wrapper_metadata: tuple[tuple[str, torch.Tensor], ...]


@dataclass(frozen=True)
class _DemandGraphKey:
    operator_family: str
    wrapper_id: int
    layer_id: int
    work_items_address: int
    dependencies_address: int
    runtime_address: int
    work_count: int
    object_count: int
    progress_blocks: tuple[int, ...]
    ready_work_counts: tuple[int, ...]
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


_GRAPH_WRAPPER_METADATA = (
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


def _freeze_graph_plan(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    try:
        return operator.index(value)
    except TypeError:
        pass
    if isinstance(value, Iterable):
        return tuple(_freeze_graph_plan(item) for item in value)
    raise RuntimeError(
        f"FlashInfer graph plan contains unsupported {type(value).__name__} state"
    )


def _graph_wrapper_metadata(wrapper: Any) -> tuple[tuple[str, torch.Tensor], ...]:
    return tuple(
        (name, value)
        for name in _GRAPH_WRAPPER_METADATA
        if torch.is_tensor(value := getattr(wrapper, name, None))
    )


def _graph_wrapper_metadata_layout(
    wrapper: Any,
) -> tuple[tuple[str, tuple[int, ...], tuple[int, ...], str, str], ...]:
    return tuple(
        (
            name,
            tuple(int(extent) for extent in value.shape),
            tuple(int(stride) for stride in value.stride()),
            str(value.dtype),
            str(value.device),
        )
        for name, value in _graph_wrapper_metadata(wrapper)
    )


def _demand_graph_key(
    *,
    operator_family: str,
    wrapper: Any,
    layer_id: int,
    plan: DeviceWorkPlan,
    runtime_tensor: torch.Tensor,
    work_count: int,
    object_count: int,
    progress_blocks: tuple[int, ...],
    ready_work_counts: tuple[int, ...],
    initial_ready_work_count: int,
    indexed_copy_blocks_per_group: int,
    query: torch.Tensor,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
    sm_scale: float,
    k_scale: float | None,
    v_scale: float | None,
    causal: bool,
    window_left: int,
) -> _DemandGraphKey:
    """Describe every dynamic value baked into a demand graph launch."""
    if operator_family not in {"decode", "paged_prefill"}:
        raise ValueError("unsupported demand graph operator family")
    return _DemandGraphKey(
        operator_family,
        id(wrapper),
        int(layer_id),
        int(plan.work_items_address),
        int(plan.dependencies_address),
        int(runtime_tensor.data_ptr()),
        int(work_count),
        int(object_count),
        tuple(int(count) for count in progress_blocks),
        tuple(int(count) for count in ready_work_counts),
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
        _freeze_graph_plan(getattr(wrapper, "_plan_info", None)),
        _graph_wrapper_metadata_layout(wrapper),
    )
