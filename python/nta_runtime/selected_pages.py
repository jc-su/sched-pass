"""GPU-selected page acquisition without a host identity round trip."""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Sequence
from typing import Any

from .runtime import (
    AcquireRequirement,
    DeviceWorkPlan,
    IndexedHostObject,
    RequestRange,
    Runtime,
    WorkItem,
)


@dataclasses.dataclass(frozen=True)
class SelectedPageAcquisition:
    """Own device index tables and their bounded object bindings."""

    source_indices: Any
    destination_indices: Any
    object_slots: tuple[int, ...]
    object_ids: tuple[int, ...]
    version: int
    page_bytes: int

    @property
    def request_count(self) -> int:
        return len(self.object_slots)

    @property
    def pages_per_request(self) -> int:
        return int(self.source_indices.shape[1])

    def requirement(self, request_index: int) -> AcquireRequirement:
        if not 0 <= request_index < self.request_count:
            raise IndexError("selected-page request index is out of range")
        return AcquireRequirement(
            0,
            0,
            self.object_ids[request_index],
            0,
            self.object_slots[request_index],
            self.version,
            self.pages_per_request * self.page_bytes,
            0,
        )


def register_selected_host_pages(
    runtime: Runtime,
    host_pages: Any,
    staging_pages: Any,
    selected_source_indices: Any,
    *,
    first_object_slot: int = 0,
    object_id_base: int = 0x53454C4543540000,
    version: int = 1,
    stream: Any = None,
) -> SelectedPageAcquisition:
    """Bind one indexed host object to each request's GPU index row."""
    import torch

    if (
        version <= 0
        or version > 0xFFFFFFFF
        or not 0 <= first_object_slot <= 0xFFFFFFFF
        or not 0 <= object_id_base <= 0xFFFFFFFFFFFFFFFF
    ):
        raise ValueError("selected-page object version and first slot are invalid")
    if (
        not isinstance(host_pages, torch.Tensor)
        or host_pages.device.type != "cpu"
        or not host_pages.is_pinned()
        or not host_pages.is_contiguous()
    ):
        raise ValueError("selected-page source must be contiguous pinned CPU memory")
    if (
        not isinstance(staging_pages, torch.Tensor)
        or staging_pages.device.type != "cuda"
        or not staging_pages.is_contiguous()
    ):
        raise ValueError("selected-page staging must be contiguous CUDA memory")
    if (
        host_pages.ndim < 2
        or staging_pages.ndim != host_pages.ndim
        or host_pages.shape[1:] != staging_pages.shape[1:]
        or host_pages.dtype != staging_pages.dtype
    ):
        raise ValueError("selected-page source and staging row geometry disagree")
    if (
        not isinstance(selected_source_indices, torch.Tensor)
        or selected_source_indices.device != staging_pages.device
        or selected_source_indices.dtype != torch.int32
        or selected_source_indices.ndim != 2
        or not selected_source_indices.is_contiguous()
        or min(selected_source_indices.shape) <= 0
    ):
        raise ValueError(
            "selected-page indices must be a non-empty contiguous CUDA int32 matrix"
        )

    request_count, pages_per_request = map(int, selected_source_indices.shape)
    selected_pages = request_count * pages_per_request
    if selected_pages > int(staging_pages.shape[0]):
        raise ValueError("selected-page staging has too few destination rows")
    if first_object_slot + request_count > 1 << 32:
        raise ValueError("selected-page object slots overflow uint32")
    page_bytes = host_pages[0].numel() * host_pages.element_size()
    if page_bytes <= 0 or page_bytes > 0xFFFFFFFF:
        raise ValueError("selected-page row size is outside the runtime ABI")

    destination_indices = torch.arange(
        selected_pages,
        dtype=torch.int32,
        device=selected_source_indices.device,
    ).view(request_count, pages_per_request)
    source_stride = host_pages.stride(0) * host_pages.element_size()
    staging_stride = staging_pages.stride(0) * staging_pages.element_size()
    objects = []
    object_slots = []
    object_ids = []
    for request_index in range(request_count):
        object_slot = first_object_slot + request_index
        object_id = object_id_base + request_index
        if object_id > 0xFFFFFFFFFFFFFFFF:
            raise ValueError("selected-page object IDs overflow uint64")
        objects.append(
            IndexedHostObject(
                object_id,
                version,
                host_pages.data_ptr(),
                staging_pages.data_ptr(),
                selected_source_indices[request_index].data_ptr(),
                destination_indices[request_index].data_ptr(),
                pages_per_request,
                page_bytes,
                source_stride,
                staging_stride,
                int(host_pages.shape[0]),
                int(staging_pages.shape[0]),
            )
        )
        object_slots.append(object_slot)
        object_ids.append(object_id)
    runtime.register_indexed_host_objects(first_object_slot, objects, stream=stream)
    return SelectedPageAcquisition(
        selected_source_indices,
        destination_indices,
        tuple(object_slots),
        tuple(object_ids),
        version,
        page_bytes,
    )


def build_selected_page_work_plan(
    runtime: Runtime,
    acquisition: SelectedPageAcquisition,
    schedule_request_indices: Sequence[int],
    *,
    request_slots: Sequence[int] | None = None,
    generations: Sequence[int] | None = None,
    estimated_compute_ns: int = 1,
    stream: Any = None,
) -> DeviceWorkPlan:
    """Bind each compiler-visible work item to its request's selected pages."""
    schedule = tuple(int(index) for index in schedule_request_indices)
    if not schedule or min(schedule) < 0 or max(schedule) >= acquisition.request_count:
        raise ValueError("selected-page schedule contains an invalid request index")
    if estimated_compute_ns <= 0 or estimated_compute_ns > 0xFFFFFFFF:
        raise ValueError("selected-page compute estimate is outside uint32")
    if request_slots is None:
        request_slots = tuple(range(acquisition.request_count))
    if generations is None:
        generations = (acquisition.version,) * acquisition.request_count
    slots = tuple(int(value) for value in request_slots)
    epochs = tuple(int(value) for value in generations)
    if len(slots) != acquisition.request_count or len(epochs) != len(slots):
        raise ValueError("selected-page request bindings have the wrong width")
    if any(value < 0 or value > 0xFFFFFFFF for value in slots) or any(
        value <= 0 or value > 0xFFFFFFFF for value in epochs
    ):
        raise ValueError("selected-page request bindings are outside uint32")

    contributor_counts = Counter(schedule)
    contributor_indices = {request: 0 for request in contributor_counts}
    dependencies = []
    work_items = []
    for work_ticket, request in enumerate(schedule):
        dependencies.append(acquisition.requirement(request))
        work_items.append(
            WorkItem(
                request,
                slots[request],
                epochs[request],
                work_ticket,
                work_ticket,
                1,
                0,
                work_ticket,
                request,
                contributor_indices[request],
                contributor_counts[request],
                estimated_compute_ns,
            )
        )
        contributor_indices[request] += 1

    ranges = []
    observed_requests = set()
    begin = 0
    while begin < len(schedule):
        request = schedule[begin]
        if request in observed_requests:
            raise ValueError("selected-page request work must be contiguous")
        observed_requests.add(request)
        end = begin + 1
        while end < len(schedule) and schedule[end] == request:
            end += 1
        ranges.append(RequestRange(begin, end - begin, slots[request], epochs[request]))
        begin = end

    plan = DeviceWorkPlan(len(work_items), len(dependencies), runtime.device_ordinal)
    plan.upload(work_items, dependencies, ranges, stream=stream)
    return plan
