"""Engine-neutral semantic contract for late-bound execution.

The native ABI already carries :class:`WorkItem` and dependency records.  This
module is the higher-level contract used by frontends and protocol planners:
one work unit has one request-generation identity, one demand descriptor, and
one availability state.  It deliberately contains no SGLang, vLLM, or
transport types.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .requests import RequestBinding


class Granularity(str, Enum):
    """The unit at which demand and availability may differ."""

    REQUEST = "request"
    LAYER = "layer"
    PAGE_GROUP = "page_group"
    CTA_TILE = "cta_tile"


class DemandSemantics(str, Enum):
    """What the demand provider promises to the numerical consumer."""

    EXACT_DENSE = "exact_dense"
    EXACT_SPARSE = "exact_sparse"


class Availability(str, Enum):
    """Lifecycle state of one work unit in an execution epoch."""

    UNBOUND = "unbound"
    BLOCKED = "blocked"
    READY = "ready"
    RUNNING = "running"
    PARTIAL = "partial"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class DemandDescriptor:
    """A demand result independent of how it was produced.

    ``selected_units`` describes the exact amount consumed by the work unit.
    Demand quality is outside this contract: the provider must supply the
    exact IDs consumed by the numerical operator.
    """

    candidate_units: int
    selected_units: int
    unit_bytes: int
    granularity: Granularity
    semantics: DemandSemantics
    provider: str
    epoch: int
    selected_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.semantics, DemandSemantics):
            raise ValueError("demand semantics must be an exact contract")
        if self.candidate_units <= 0:
            raise ValueError("candidate_units must be positive")
        if not 0 < self.selected_units <= self.candidate_units:
            raise ValueError("selected_units must be in (0, candidate_units]")
        if self.unit_bytes <= 0:
            raise ValueError("unit_bytes must be positive")
        if not self.provider:
            raise ValueError("demand provider must be named")
        if self.epoch < 0:
            raise ValueError("demand epoch cannot be negative")
        if self.selected_ids:
            if len(self.selected_ids) != self.selected_units:
                raise ValueError("selected_ids must match selected_units")
            if len(set(self.selected_ids)) != len(self.selected_ids):
                raise ValueError("selected_ids must be unique")
            if (
                min(self.selected_ids) < 0
                or max(self.selected_ids) >= self.candidate_units
            ):
                raise ValueError("selected_ids must be candidate-relative indices")
        if self.semantics is DemandSemantics.EXACT_SPARSE and not self.selected_ids:
            raise ValueError("exact sparse demand must identify its selected units")
        if self.semantics is DemandSemantics.EXACT_DENSE:
            if self.selected_units != self.candidate_units:
                raise ValueError("exact dense demand must select every candidate")
            if self.selected_ids and tuple(self.selected_ids) != tuple(
                range(self.candidate_units)
            ):
                raise ValueError("exact dense demand must preserve candidate order")

    @property
    def candidate_bytes(self) -> int:
        return self.candidate_units * self.unit_bytes

    @property
    def selected_bytes(self) -> int:
        return self.selected_units * self.unit_bytes

    @property
    def reduction_fraction(self) -> float:
        return 1.0 - (self.selected_units / self.candidate_units)

    @property
    def is_exact(self) -> bool:
        return self.semantics in (
            DemandSemantics.EXACT_DENSE,
            DemandSemantics.EXACT_SPARSE,
        )


@dataclass(frozen=True)
class WorkUnit:
    """One schedulable, request-bound piece of numerical work."""

    work_id: int
    binding: RequestBinding
    layer: int
    logical_begin: int
    logical_count: int
    demand: DemandDescriptor
    dependency_ids: tuple[int, ...] = ()
    estimated_compute_ns: int = 0
    reduction_group: int = 0
    contributor_index: int = 0
    contributor_count: int = 1
    availability: Availability = Availability.UNBOUND

    def __post_init__(self) -> None:
        if self.work_id < 0:
            raise ValueError("work_id cannot be negative")
        if self.layer < 0:
            raise ValueError("layer cannot be negative")
        if self.logical_begin < 0 or self.logical_count <= 0:
            raise ValueError("logical work range is invalid")
        if self.estimated_compute_ns < 0:
            raise ValueError("estimated compute time cannot be negative")
        if self.reduction_group < 0:
            raise ValueError("reduction_group cannot be negative")
        if not 0 <= self.contributor_index < self.contributor_count:
            raise ValueError("contributor index is outside its reduction group")
        if len(set(self.dependency_ids)) != len(self.dependency_ids):
            raise ValueError("dependency_ids must be unique")
        if any(dependency < 0 for dependency in self.dependency_ids):
            raise ValueError("dependency_ids cannot be negative")
        if self.demand.epoch < 0:
            raise ValueError("work unit epoch cannot be negative")

    @property
    def identity(self) -> tuple[int, int]:
        """The slot/generation pair that makes stale completion impossible."""

        return (self.binding.request_slot, self.binding.generation)


@dataclass(frozen=True)
class WorkBatch:
    """A validated heterogeneous batch for one execution epoch."""

    epoch: int
    granularity: Granularity
    units: tuple[WorkUnit, ...]

    def __post_init__(self) -> None:
        if self.epoch < 0:
            raise ValueError("batch epoch cannot be negative")
        if not self.units:
            raise ValueError("a work batch must contain at least one unit")
        work_ids = [unit.work_id for unit in self.units]
        if len(set(work_ids)) != len(work_ids):
            raise ValueError("work_id must be unique within a batch")
        request_indices: dict[int, tuple[int, int]] = {}
        for unit in self.units:
            if unit.demand.granularity is not self.granularity:
                raise ValueError("all work units must use the batch granularity")
            if unit.demand.epoch != self.epoch:
                raise ValueError("work demand epoch does not match the batch")
            identity = unit.identity
            previous = request_indices.setdefault(unit.binding.request_index, identity)
            if previous != identity:
                raise ValueError(
                    "one request index cannot change generation in a batch"
                )

    @property
    def request_identities(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted({unit.identity for unit in self.units}))

    @property
    def ready_fraction(self) -> float:
        return sum(
            unit.availability is Availability.READY for unit in self.units
        ) / len(self.units)

    @property
    def blocked_fraction(self) -> float:
        return sum(
            unit.availability is Availability.BLOCKED for unit in self.units
        ) / len(self.units)

    @property
    def is_heterogeneous(self) -> bool:
        return (
            len({unit.availability for unit in self.units}) > 1
            or len({unit.demand.selected_units for unit in self.units}) > 1
        )
