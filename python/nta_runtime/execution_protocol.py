"""Protocol and granularity planning for heterogeneous late-bound work."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
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
        # Keep insertion-ordered membership buckets instead of deriving every
        # query by scanning ``_states``.  A forward may expose the ledger once
        # per layer; the old representation made that O(layers * work_units)
        # even when only a small ready/blocked frontier changed.  Dicts give
        # deterministic iteration and O(1) state-count/membership updates.
        self._state_members: dict[Availability, dict[int, None]] = {
            state: {} for state in Availability
        }
        for unit in batch.units:
            self._state_members[unit.availability][unit.work_id] = None

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
        if (
            target is Availability.RUNNING
            and len(self._state_members[Availability.RUNNING])
            >= self.config.max_inflight_units
        ):
            raise ValueError("execution protocol in-flight capacity would be exceeded")
        self._state_members[current].pop(work_id)
        self._state_members[target][work_id] = None
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
        if not states:
            return ()
        if len(states) == 1:
            return tuple(self._state_members[states[0]])
        # Preserve the old set-like API semantics for callers that pass the
        # same state twice, while retaining deterministic batch order.  The
        # state buckets are insertion ordered, so a scan over the unit order
        # is both bounded and duplicate-free.
        allowed = set(states)
        return tuple(
            work_id for work_id, state in self._states.items() if state in allowed
        )

    def runnable_groups(self) -> tuple[tuple[int, ...], ...]:
        """Return bounded groups that the selected protocol may launch now.

        Conventional execution has a batch readiness boundary.  The other
        forms expose ready work immediately, but never exceed the configured
        in-flight bound per group. Groups are sequential launch windows: a
        caller must complete or partially release one group before launching
        the next. Keeping this rule in the ledger prevents each engine adapter
        from inventing its own interpretation of the configuration.
        """
        ready = self.units_in(Availability.READY)
        if self.config.kind is ProtocolKind.CONVENTIONAL and len(ready) != len(
            self._units
        ):
            return ()
        in_flight = len(self._state_members[Availability.RUNNING])
        width = self.config.max_inflight_units - in_flight
        if width <= 0:
            return ()
        return tuple(
            ready[start : start + width] for start in range(0, len(ready), width)
        )

    @property
    def is_complete(self) -> bool:
        return len(self._state_members[Availability.COMPLETE]) == len(self._units)

    @property
    def state_counts(self) -> dict[Availability, int]:
        return {state: len(members) for state, members in self._state_members.items()}
