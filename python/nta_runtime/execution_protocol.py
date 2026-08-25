"""Protocol and granularity planning for heterogeneous late-bound work."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping

from .requests import RequestBinding
from .work_unit import Availability, Granularity, WorkBatch


class ProtocolKind(str, Enum):
    """Execution forms of one late-bound work-unit mechanism."""

    CONVENTIONAL = "conventional"
    LATE_BOUND = "late_bound"
    PARTIAL = "partial"


@dataclass(frozen=True)
class ExecutionProtocolConfig:
    """Explicit protocol configuration replacing scattered environment flags."""

    kind: ProtocolKind
    granularity: Granularity
    max_inflight_units: int
    allow_overlap: bool
    allow_partial: bool

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        prefix: str = "NTA_EXECUTION",
    ) -> "ExecutionProtocolConfig":
        """Parse the one framework-neutral protocol configuration."""
        import os

        values = os.environ if environ is None else environ
        raw_kind = values.get(f"{prefix}_PROTOCOL", "late_bound").strip().lower()
        try:
            kind = ProtocolKind(raw_kind)
        except ValueError as error:
            raise ValueError(
                f"{prefix}_PROTOCOL must be conventional, late_bound, or partial"
            ) from error
        try:
            granularity = Granularity(
                values.get(f"{prefix}_GRANULARITY", Granularity.PAGE_GROUP.value)
                .strip()
                .lower()
            )
        except ValueError as error:
            raise ValueError(
                f"{prefix}_GRANULARITY must be request, layer, page_group, or cta_tile"
            ) from error
        try:
            max_inflight = int(values.get(f"{prefix}_MAX_INFLIGHT_UNITS", "4096"))
        except ValueError as error:
            raise ValueError(
                f"{prefix}_MAX_INFLIGHT_UNITS must be an integer"
            ) from error
        if kind is ProtocolKind.CONVENTIONAL:
            return cls.conventional(
                granularity=granularity, max_inflight_units=max_inflight
            )
        if kind is ProtocolKind.PARTIAL:
            return cls.partial(granularity=granularity, max_inflight_units=max_inflight)
        return cls.late_bound(granularity=granularity, max_inflight_units=max_inflight)

    def __post_init__(self) -> None:
        if self.max_inflight_units <= 0:
            raise ValueError("max_inflight_units must be positive")
        if self.kind is ProtocolKind.CONVENTIONAL and (
            self.allow_overlap or self.allow_partial
        ):
            raise ValueError("conventional execution cannot overlap or resume work")
        if self.kind is ProtocolKind.PARTIAL and not self.allow_partial:
            raise ValueError("partial execution must allow partial work")
        if self.kind is not ProtocolKind.PARTIAL and self.allow_partial:
            raise ValueError("only the partial protocol may resume partial work")

    @classmethod
    def conventional(
        cls, *, granularity: Granularity, max_inflight_units: int
    ) -> "ExecutionProtocolConfig":
        return cls(
            ProtocolKind.CONVENTIONAL,
            granularity,
            max_inflight_units,
            allow_overlap=False,
            allow_partial=False,
        )

    @classmethod
    def late_bound(
        cls, *, granularity: Granularity, max_inflight_units: int
    ) -> "ExecutionProtocolConfig":
        return cls(
            ProtocolKind.LATE_BOUND,
            granularity,
            max_inflight_units,
            allow_overlap=True,
            allow_partial=False,
        )

    @classmethod
    def partial(
        cls, *, granularity: Granularity, max_inflight_units: int
    ) -> "ExecutionProtocolConfig":
        return cls(
            ProtocolKind.PARTIAL,
            granularity,
            max_inflight_units,
            allow_overlap=True,
            allow_partial=True,
        )


_TRANSITIONS: Mapping[Availability, frozenset[Availability]] = {
    Availability.UNBOUND: frozenset(
        {
            Availability.BLOCKED,
            Availability.READY,
            Availability.CANCELLED,
            Availability.FAILED,
        }
    ),
    Availability.BLOCKED: frozenset(
        {Availability.READY, Availability.CANCELLED, Availability.FAILED}
    ),
    Availability.READY: frozenset(
        {Availability.RUNNING, Availability.CANCELLED, Availability.FAILED}
    ),
    Availability.RUNNING: frozenset(
        {
            Availability.PARTIAL,
            Availability.COMPLETE,
            Availability.CANCELLED,
            Availability.FAILED,
        }
    ),
    Availability.PARTIAL: frozenset(
        {
            Availability.READY,
            Availability.RUNNING,
            Availability.COMPLETE,
            Availability.CANCELLED,
            Availability.FAILED,
        }
    ),
    Availability.COMPLETE: frozenset(),
    Availability.CANCELLED: frozenset(),
    Availability.FAILED: frozenset(),
}


class WorkLedger:
    """Generation-checked state machine shared by direct and partial forms."""

    def __init__(self, batch: WorkBatch, config: ExecutionProtocolConfig) -> None:
        if batch.granularity is not config.granularity:
            raise ValueError("batch and execution protocol use different granularities")
        self.batch = batch
        self.config = config
        self._units = {unit.work_id: unit for unit in batch.units}
        self._states = {unit.work_id: unit.availability for unit in batch.units}

    def state(self, work_id: int) -> Availability:
        try:
            return self._states[work_id]
        except KeyError as error:
            raise KeyError(f"unknown work unit {work_id}") from error

    def transition(
        self,
        work_id: int,
        target: Availability,
        *,
        binding: RequestBinding,
        epoch: int,
    ) -> None:
        unit = self._units.get(work_id)
        if unit is None:
            raise KeyError(f"unknown work unit {work_id}")
        if epoch != self.batch.epoch or epoch != unit.demand.epoch:
            raise ValueError("stale execution epoch")
        if (
            binding.request_slot != unit.binding.request_slot
            or binding.generation != unit.binding.generation
        ):
            raise ValueError("stale request generation cannot advance work")
        current = self._states[work_id]
        if target not in _TRANSITIONS[current]:
            raise ValueError(
                f"invalid work transition {current.value} -> {target.value}"
            )
        if target is Availability.PARTIAL and not self.config.allow_partial:
            raise ValueError(
                "the selected execution protocol does not support partial work"
            )
        self._states[work_id] = target

    def discover(
        self,
        work_id: int,
        *,
        ready: bool,
        binding: RequestBinding,
        epoch: int,
    ) -> None:
        self.transition(
            work_id,
            Availability.READY if ready else Availability.BLOCKED,
            binding=binding,
            epoch=epoch,
        )

    def units_in(self, *states: Availability) -> tuple[int, ...]:
        allowed = set(states)
        return tuple(
            work_id for work_id, state in self._states.items() if state in allowed
        )

    def runnable_groups(self) -> tuple[tuple[int, ...], ...]:
        """Return bounded groups that the selected protocol may launch now.

        Conventional execution has a batch readiness boundary.  The other
        forms expose ready work immediately, but never exceed the configured
        in-flight bound.  Keeping this rule in the ledger prevents each engine
        adapter from inventing its own interpretation of the configuration.
        """
        ready = self.units_in(Availability.READY)
        if self.config.kind is ProtocolKind.CONVENTIONAL and len(ready) != len(
            self._units
        ):
            return ()
        in_flight = len(self.units_in(Availability.RUNNING))
        width = self.config.max_inflight_units - in_flight
        if width <= 0:
            return ()
        return tuple(
            ready[start : start + width] for start in range(0, len(ready), width)
        )

    @property
    def is_complete(self) -> bool:
        return all(state is Availability.COMPLETE for state in self._states.values())

    @property
    def state_counts(self) -> dict[Availability, int]:
        counts = {state: 0 for state in Availability}
        for state in self._states.values():
            counts[state] += 1
        return counts


@dataclass(frozen=True)
class GranularityEstimate:
    units_per_group: int
    groups: int
    transfer_ns: int
    compute_ns: int
    control_ns: int
    availability_ns: int

    @property
    def total_ns(self) -> int:
        return (
            self.transfer_ns + self.compute_ns + self.control_ns + self.availability_ns
        )


@dataclass(frozen=True)
class GranularityCostModel:
    """Transparent cost model for choosing a work-unit grouping.

    The model is deliberately a planner, not a performance oracle.  It makes
    the tradeoff explicit: finer groups reduce availability exposure but add
    per-group control cost.  Parameters must come from measured hardware and
    transport profiles in an experiment, never from a selector's quality data.
    """

    bandwidth_bytes_per_second: int = 55_000_000_000
    group_overhead_ns: int = 80_000
    compute_per_unit_ns: int = 3_000

    def __post_init__(self) -> None:
        if self.bandwidth_bytes_per_second <= 0:
            raise ValueError("bandwidth must be positive")
        if self.group_overhead_ns < 0 or self.compute_per_unit_ns < 0:
            raise ValueError("cost parameters cannot be negative")

    def estimate(
        self,
        *,
        selected_units: int,
        unit_bytes: int,
        units_per_group: int,
        availability_skew_ns: int,
    ) -> GranularityEstimate:
        if min(selected_units, unit_bytes, units_per_group) <= 0:
            raise ValueError("granularity estimate dimensions must be positive")
        if availability_skew_ns < 0:
            raise ValueError("availability skew cannot be negative")
        groups = math.ceil(selected_units / units_per_group)
        transfer_ns = math.ceil(
            selected_units
            * unit_bytes
            * 1_000_000_000
            / self.bandwidth_bytes_per_second
        )
        compute_ns = selected_units * self.compute_per_unit_ns
        control_ns = groups * self.group_overhead_ns
        availability_ns = math.ceil(availability_skew_ns / groups)
        return GranularityEstimate(
            units_per_group,
            groups,
            transfer_ns,
            compute_ns,
            control_ns,
            availability_ns,
        )

    def choose(
        self,
        *,
        selected_units: int,
        unit_bytes: int,
        candidate_group_sizes: tuple[int, ...],
        availability_skew_ns: int,
    ) -> GranularityEstimate:
        if not candidate_group_sizes:
            raise ValueError("at least one candidate group size is required")
        estimates = tuple(
            self.estimate(
                selected_units=selected_units,
                unit_bytes=unit_bytes,
                units_per_group=group_size,
                availability_skew_ns=availability_skew_ns,
            )
            for group_size in candidate_group_sizes
        )
        return min(estimates, key=lambda estimate: estimate.total_ns)
