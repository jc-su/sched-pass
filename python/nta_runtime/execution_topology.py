"""Compact exact work topology shared by framework adapters and the runtime.

The semantic execution model is valuable as an executable specification, but
constructing a graph of ``DemandDescriptor`` and ``WorkUnit`` objects on every
serving request duplicates facts already proved by an engine schedule.  This
module contains the smaller production contract: exact demand geometry,
request generation ownership, and the logical coordinates needed by native
work tickets.  It deliberately contains no availability state machine.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .abi import u32, u64
from .requests import RequestBinding


@dataclass(frozen=True, slots=True)
class RequestWorkTopology:
    """One request's contiguous contributor range in an exact work topology."""

    request_index: int
    work_begin: int
    work_count: int
    request_slot: int
    generation: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "request_index", u32(self.request_index, "request index")
        )
        object.__setattr__(self, "work_begin", u32(self.work_begin, "work begin"))
        object.__setattr__(
            self, "work_count", u32(self.work_count, "work count", positive=True)
        )
        object.__setattr__(
            self, "request_slot", u32(self.request_slot, "request slot")
        )
        object.__setattr__(
            self,
            "generation",
            u32(self.generation, "request generation", positive=True),
        )


@dataclass(frozen=True, slots=True)
class WorkDependencySpan:
    """One work ticket's bounded slice of the batch dependency array."""

    begin: int
    count: int
    direct_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "begin", u32(self.begin, "dependency begin"))
        object.__setattr__(
            self, "count", u32(self.count, "dependency count", positive=True)
        )
        object.__setattr__(
            self,
            "direct_count",
            u32(self.direct_count, "direct dependency count"),
        )
        if self.direct_count > self.count:
            raise ValueError("direct dependency count exceeds dependency count")


@dataclass(frozen=True, slots=True)
class ExactWorkTopology:
    """Minimal production proof for one exact, request-owned work schedule.

    ``demand_units`` records the exact dense units consumed by each work item;
    selected sparse IDs remain a compiler/frontend proof and are not copied
    into the native ticket ABI.  Requests must own contiguous contributors,
    matching the FlashInfer merge contract and native reduction layout.
    """

    epoch: int
    logical_work: tuple[int, ...]
    demand_units: tuple[int, ...]
    unit_bytes: int
    estimated_compute_ns: tuple[int, ...]
    requests: tuple[RequestWorkTopology, ...]
    ready_deadline_offset_ns: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "epoch", u32(self.epoch, "work epoch"))
        object.__setattr__(
            self, "unit_bytes", u64(self.unit_bytes, "unit bytes", positive=True)
        )
        work_count = len(self.logical_work)
        if work_count == 0:
            raise ValueError("exact work topology cannot be empty")
        if (
            len(self.demand_units) != work_count
            or len(self.estimated_compute_ns) != work_count
        ):
            raise ValueError("exact work topology arrays must align")
        logical = tuple(u32(value, "logical work") for value in self.logical_work)
        demand = tuple(
            u32(value, "exact demand units", positive=True)
            for value in self.demand_units
        )
        compute = tuple(
            u32(value, "estimated compute ns")
            for value in self.estimated_compute_ns
        )
        object.__setattr__(self, "logical_work", logical)
        object.__setattr__(self, "demand_units", demand)
        object.__setattr__(self, "estimated_compute_ns", compute)
        deadline_values = self.ready_deadline_offset_ns or (0,) * work_count
        if len(deadline_values) != work_count:
            raise ValueError("exact work topology deadline array must align")
        object.__setattr__(
            self,
            "ready_deadline_offset_ns",
            tuple(u64(value, "ready deadline offset") for value in deadline_values),
        )
        if not self.requests:
            raise ValueError("exact work topology has no request ownership")
        cursor = 0
        for expected_index, request in enumerate(self.requests):
            if request.request_index != expected_index or request.work_begin != cursor:
                raise ValueError("exact request work ranges are not canonical")
            cursor += request.work_count
        if cursor != work_count:
            raise ValueError("exact request work ranges do not cover the topology")

    @classmethod
    def from_schedule(
        cls,
        *,
        epoch: int,
        bindings: Sequence[RequestBinding],
        request_indices: Sequence[int],
        logical_work: Sequence[int],
        demand_units: Sequence[int],
        unit_bytes: int,
        estimated_compute_ns: int | Sequence[int],
    ) -> "ExactWorkTopology":
        """Validate an engine schedule once and retain only its native facts."""

        binding_values = tuple(bindings)
        request_values = tuple(int(value) for value in request_indices)
        logical_values = tuple(int(value) for value in logical_work)
        demand_values = tuple(int(value) for value in demand_units)
        if not binding_values:
            raise ValueError("exact work topology requires request bindings")
        if len(request_values) != len(logical_values) or len(request_values) != len(
            demand_values
        ):
            raise ValueError("engine schedule and exact demand arrays must align")
        if isinstance(estimated_compute_ns, int):
            compute_values = (estimated_compute_ns,) * len(request_values)
        else:
            compute_values = tuple(int(value) for value in estimated_compute_ns)
        ranges: list[RequestWorkTopology] = []
        cursor = 0
        for expected_index, binding in enumerate(binding_values):
            if binding.request_index != expected_index:
                raise ValueError("request bindings are not in canonical engine order")
            begin = cursor
            while (
                cursor < len(request_values)
                and request_values[cursor] == expected_index
            ):
                cursor += 1
            if cursor == begin:
                raise ValueError(
                    f"engine schedule has no work for request {expected_index}"
                )
            ranges.append(
                RequestWorkTopology(
                    expected_index,
                    begin,
                    cursor - begin,
                    binding.request_slot,
                    binding.generation,
                )
            )
        if cursor != len(request_values):
            raise ValueError("engine schedule is not contiguous in request order")
        return cls(
            epoch,
            logical_values,
            demand_values,
            unit_bytes,
            compute_values,
            tuple(ranges),
        )

    @property
    def work_count(self) -> int:
        return len(self.logical_work)

    @property
    def request_count(self) -> int:
        return len(self.requests)

    @property
    def selected_units(self) -> int:
        return sum(self.demand_units)

    @property
    def selected_bytes(self) -> int:
        return self.selected_units * self.unit_bytes
