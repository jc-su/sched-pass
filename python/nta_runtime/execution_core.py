"""Engine-neutral execution session for one heterogeneous forward epoch.

The engine adapter owns request metadata.  The compiler/runtime owns the
native storage ABI.  This module is the semantic bridge between them: it
turns an engine-independent list of logical tiles into one validated
``WorkBatch`` and keeps every availability transition generation- and
epoch-checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .execution_protocol import ExecutionProtocolConfig, WorkLedger
from .requests import RequestBinding
from .work_unit import (
    Availability,
    DemandDescriptor,
    DemandSemantics,
    Granularity,
    WorkBatch,
    WorkUnit,
)


@dataclass(frozen=True)
class ExecutionTile:
    """Engine-neutral description of one logical consumer tile.

    ``selected_ids=()`` is the canonical exact-dense representation: every
    candidate is selected in candidate order, so materializing an O(pages)
    identity tuple would carry no information. Exact-sparse tiles must provide
    their candidate-relative IDs explicitly.
    """

    work_id: int
    binding: RequestBinding
    layer: int
    logical_begin: int
    candidate_units: int
    selected_ids: tuple[int, ...]
    unit_bytes: int
    ready: bool
    estimated_compute_ns: int
    reduction_group: int
    contributor_index: int = 0
    contributor_count: int = 1

    def __post_init__(self) -> None:
        if self.candidate_units <= 0:
            raise ValueError("execution tile must have candidate units")
        if self.unit_bytes <= 0:
            raise ValueError("execution tile unit size must be positive")
        if self.selected_ids:
            if len(set(self.selected_ids)) != len(self.selected_ids):
                raise ValueError("execution tile selected units must be unique")
            if (
                min(self.selected_ids) < 0
                or max(self.selected_ids) >= self.candidate_units
            ):
                raise ValueError(
                    "execution tile selected ID is outside its candidate set"
                )

    @property
    def selected_units(self) -> int:
        return len(self.selected_ids) if self.selected_ids else self.candidate_units

    @property
    def demand_semantics(self) -> DemandSemantics:
        if not self.selected_ids:
            return DemandSemantics.EXACT_DENSE
        if len(self.selected_ids) != self.candidate_units:
            return DemandSemantics.EXACT_SPARSE
        return (
            DemandSemantics.EXACT_DENSE
            if all(
                selected == index for index, selected in enumerate(self.selected_ids)
            )
            else DemandSemantics.EXACT_SPARSE
        )

    @property
    def canonical_selected_ids(self) -> tuple[int, ...]:
        if self.demand_semantics is DemandSemantics.EXACT_DENSE:
            return ()
        return self.selected_ids


def _work_batch_from_tiles(
    *,
    epoch: int,
    granularity: Granularity,
    tiles: Iterable[ExecutionTile],
) -> WorkBatch:
    tile_values = tuple(tiles)
    units = tuple(
        WorkUnit(
            work_id=tile.work_id,
            binding=tile.binding,
            layer=tile.layer,
            logical_begin=tile.logical_begin,
            logical_count=1,
            demand=DemandDescriptor(
                candidate_units=tile.candidate_units,
                selected_units=tile.selected_units,
                unit_bytes=tile.unit_bytes,
                granularity=granularity,
                semantics=tile.demand_semantics,
                provider="engine.schedule",
                epoch=epoch,
                selected_ids=tile.canonical_selected_ids,
            ),
            estimated_compute_ns=tile.estimated_compute_ns,
            reduction_group=tile.reduction_group,
            contributor_index=tile.contributor_index,
            contributor_count=tile.contributor_count,
            availability=(
                Availability.READY if tile.ready else Availability.BLOCKED
            ),
        )
        for tile in tile_values
    )
    return WorkBatch(epoch, granularity, units)


@dataclass(frozen=True)
class ExecutionPlan:
    """Immutable typed work description consumed by the native runtime.

    Serving needs identity, exact demand, and schedule-coordinate validation,
    but native tickets—not Python—own availability transitions.  Keeping this
    contract separate from :class:`ExecutionSession` prevents a CI
    specification ledger from becoming a second state machine on the serving
    hot path.
    """

    batch: WorkBatch
    protocol: ExecutionProtocolConfig
    _units_by_work_id: dict[int, WorkUnit] = field(
        init=False, repr=False, compare=False
    )
    _request_count: int = field(init=False, repr=False, compare=False)
    _selected_units: int = field(init=False, repr=False, compare=False)
    _candidate_units: int = field(init=False, repr=False, compare=False)
    _selected_bytes: int = field(init=False, repr=False, compare=False)
    _candidate_bytes: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.batch.granularity is not self.protocol.granularity:
            raise ValueError("batch and execution protocol use different granularities")
        object.__setattr__(
            self,
            "_units_by_work_id",
            {unit.work_id: unit for unit in self.batch.units},
        )
        object.__setattr__(
            self, "_request_count", len(self.batch.request_identities)
        )
        object.__setattr__(
            self,
            "_selected_units",
            sum(unit.demand.selected_units for unit in self.batch.units),
        )
        object.__setattr__(
            self,
            "_candidate_units",
            sum(unit.demand.candidate_units for unit in self.batch.units),
        )
        object.__setattr__(
            self,
            "_selected_bytes",
            sum(unit.demand.selected_bytes for unit in self.batch.units),
        )
        object.__setattr__(
            self,
            "_candidate_bytes",
            sum(unit.demand.candidate_bytes for unit in self.batch.units),
        )

    @classmethod
    def from_tiles(
        cls,
        *,
        epoch: int,
        granularity: Granularity,
        protocol: ExecutionProtocolConfig,
        tiles: Iterable[ExecutionTile],
    ) -> "ExecutionPlan":
        return cls(
            _work_batch_from_tiles(epoch=epoch, granularity=granularity, tiles=tiles),
            protocol,
        )

    @property
    def epoch(self) -> int:
        return self.batch.epoch

    @property
    def request_identities(self) -> tuple[tuple[int, int], ...]:
        return self.batch.request_identities

    def expose_stats(self) -> dict[str, int | float | bool]:
        counts = {state: 0 for state in Availability}
        for unit in self.batch.units:
            counts[unit.availability] += 1
        return {
            "work_epoch": self.epoch,
            "work_units": len(self.batch.units),
            "work_requests": self._request_count,
            "work_ready": counts[Availability.READY],
            "work_blocked": counts[Availability.BLOCKED],
            "work_running": counts[Availability.RUNNING],
            "work_partial": counts[Availability.PARTIAL],
            "work_complete": counts[Availability.COMPLETE],
            "work_cancelled": counts[Availability.CANCELLED],
            "work_failed": counts[Availability.FAILED],
            "work_selected_units": self._selected_units,
            "work_candidate_units": self._candidate_units,
            "work_selected_bytes": self._selected_bytes,
            "work_candidate_bytes": self._candidate_bytes,
            "work_is_heterogeneous": self.batch.is_heterogeneous,
            "work_ready_fraction": self.batch.ready_fraction,
        }

    def unit_for_ticket(
        self,
        *,
        work_id: int,
        layer: int,
        logical_begin: int,
        request_index: int,
    ) -> WorkUnit:
        unit = self._units_by_work_id.get(work_id)
        if unit is None:
            raise RuntimeError(
                "native schedule has no unique semantic work ticket: "
                f"work_id={work_id} layer={layer} logical={logical_begin} "
                f"request={request_index}"
            )
        if (
            unit.layer != layer
            or unit.logical_begin != logical_begin
            or unit.binding.request_index != request_index
        ):
            raise RuntimeError(
                "native schedule semantic coordinates diverged for work ticket: "
                f"work_id={work_id} expected=(layer={layer}, logical={logical_begin}, "
                f"request={request_index}) actual=(layer={unit.layer}, "
                f"logical={unit.logical_begin}, "
                f"request={unit.binding.request_index})"
            )
        return unit

    def unit_for_topology(
        self,
        *,
        work_id: int,
        logical_begin: int,
        request_index: int,
    ) -> WorkUnit:
        """Resolve a layer-invariant native work-plan coordinate.

        SGLang reuses one FlashInfer wrapper topology across transformer
        layers, and the native ``WorkItem`` ABI intentionally contains no
        layer field. This lookup retains the ticket/request/logical checks
        needed by that ABI without forcing Python to rebuild identical typed
        work for every layer. Opt-in semantic verification continues to use
        :meth:`unit_for_ticket` with an exact layer coordinate.
        """

        unit = self._units_by_work_id.get(work_id)
        if unit is None:
            raise RuntimeError(
                "native schedule has no unique semantic topology ticket: "
                f"work_id={work_id} logical={logical_begin} request={request_index}"
            )
        if (
            unit.logical_begin != logical_begin
            or unit.binding.request_index != request_index
        ):
            raise RuntimeError(
                "native schedule topology diverged for work ticket: "
                f"work_id={work_id} expected=(logical={logical_begin}, "
                f"request={request_index}) actual=(logical={unit.logical_begin}, "
                f"request={unit.binding.request_index})"
            )
        return unit


@dataclass
class ExecutionSession:
    """Executable specification of typed work-unit transitions.

    Unit tests, modeled experiments, and opt-in serving verification use this
    ledger. Production serving consumes :class:`ExecutionPlan` directly and
    lets native tickets remain the sole availability state machine.
    """

    batch: WorkBatch
    protocol: ExecutionProtocolConfig
    ledger: WorkLedger
    _units_by_work_id: dict[int, WorkUnit] = field(
        init=False, repr=False, compare=False
    )
    _units_by_layer: dict[int, tuple[int, ...]] = field(
        init=False, repr=False, compare=False
    )
    _request_count: int = field(init=False, repr=False, compare=False)
    _selected_units: int = field(init=False, repr=False, compare=False)
    _candidate_units: int = field(init=False, repr=False, compare=False)
    _selected_bytes: int = field(init=False, repr=False, compare=False)
    _candidate_bytes: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Native plan upload resolves every schedule ticket through this
        # boundary.  Keep that lookup O(1): a linear scan here turns a large
        # heterogeneous schedule into an accidental O(work_units^2) Python
        # control-plane cost before the GPU can overlap any transfer.
        self._units_by_work_id = {unit.work_id: unit for unit in self.batch.units}
        by_layer: dict[int, list[int]] = {}
        for unit in self.batch.units:
            by_layer.setdefault(unit.layer, []).append(unit.work_id)
        self._units_by_layer = {
            layer: tuple(work_ids) for layer, work_ids in by_layer.items()
        }
        self._request_count = len(self.batch.request_identities)
        self._selected_units = sum(
            unit.demand.selected_units for unit in self.batch.units
        )
        self._candidate_units = sum(
            unit.demand.candidate_units for unit in self.batch.units
        )
        self._selected_bytes = sum(
            unit.demand.selected_bytes for unit in self.batch.units
        )
        self._candidate_bytes = sum(
            unit.demand.candidate_bytes for unit in self.batch.units
        )

    @classmethod
    def from_tiles(
        cls,
        *,
        epoch: int,
        granularity: Granularity,
        protocol: ExecutionProtocolConfig,
        tiles: Iterable[ExecutionTile],
    ) -> "ExecutionSession":
        return cls.from_plan(
            ExecutionPlan.from_tiles(
                epoch=epoch,
                granularity=granularity,
                protocol=protocol,
                tiles=tiles,
            )
        )

    @classmethod
    def from_plan(cls, plan: ExecutionPlan) -> "ExecutionSession":
        return cls(
            plan.batch,
            plan.protocol,
            WorkLedger(plan.batch, plan.protocol),
        )

    @property
    def epoch(self) -> int:
        return self.batch.epoch

    @property
    def request_identities(self) -> tuple[tuple[int, int], ...]:
        return self.batch.request_identities

    @property
    def ready_work(self) -> tuple[int, ...]:
        return self.ledger.units_in(Availability.READY)

    @property
    def blocked_work(self) -> tuple[int, ...]:
        return self.ledger.units_in(Availability.BLOCKED)

    def runnable_groups(self) -> tuple[tuple[int, ...], ...]:
        return self.ledger.runnable_groups()

    def _binding(self, work_id: int) -> RequestBinding:
        try:
            return self._units_by_work_id[work_id].binding
        except KeyError as error:
            raise KeyError(f"unknown work unit {work_id}") from error

    def make_ready(self, work_ids: Iterable[int]) -> None:
        for work_id in work_ids:
            state = self.ledger.state(work_id)
            if state is Availability.BLOCKED:
                self.ledger.transition(
                    work_id,
                    Availability.READY,
                    binding=self._binding(work_id),
                    epoch=self.epoch,
                )

    def launch_group(self, work_ids: Iterable[int]) -> None:
        """Record a bounded native launch for already-runnable work."""
        values = tuple(work_ids)
        if len(set(values)) != len(values):
            raise RuntimeError("a launch group cannot contain duplicate work units")
        for work_id in values:
            if self.ledger.state(work_id) is not Availability.READY:
                raise RuntimeError(f"work unit {work_id} is not runnable")
        running = self.ledger.state_counts[Availability.RUNNING]
        if running + len(values) > self.protocol.max_inflight_units:
            raise RuntimeError(
                "launch group exceeds execution protocol in-flight capacity"
            )
        for work_id in values:
            self.ledger.transition(
                work_id,
                Availability.RUNNING,
                binding=self._binding(work_id),
                epoch=self.epoch,
            )

    def complete_group(self, work_ids: Iterable[int]) -> None:
        for work_id in work_ids:
            if self.ledger.state(work_id) is not Availability.RUNNING:
                raise RuntimeError(f"work unit {work_id} is not running")
            self.ledger.transition(
                work_id,
                Availability.COMPLETE,
                binding=self._binding(work_id),
                epoch=self.epoch,
            )

    def partial_group(self, work_ids: Iterable[int]) -> None:
        """Publish an exact partial result before continuation."""
        if not self.protocol.allow_partial:
            raise RuntimeError("the execution protocol does not support partial work")
        for work_id in work_ids:
            if self.ledger.state(work_id) is not Availability.RUNNING:
                raise RuntimeError(f"work unit {work_id} is not running")
            self.ledger.transition(
                work_id,
                Availability.PARTIAL,
                binding=self._binding(work_id),
                epoch=self.epoch,
            )
            self.ledger.transition(
                work_id,
                Availability.READY,
                binding=self._binding(work_id),
                epoch=self.epoch,
            )

    def complete_layer(self, layer: int) -> None:
        """Retire the semantic work for one completed consumer layer.

        Transport and CUDA completion remain native operations.  The engine
        calls this only after the layer's stream-ordered consumer has passed
        its availability boundary, so the semantic ledger and native launch
        cannot diverge silently.
        """
        layer_ids = self._units_by_layer.get(layer, ())
        for work_id in layer_ids:
            state = self.ledger.state(work_id)
            if state is Availability.BLOCKED:
                self.make_ready((work_id,))
                state = Availability.READY
            if state is Availability.READY:
                self.launch_group((work_id,))
                self.complete_group((work_id,))
            elif state is Availability.RUNNING:
                self.complete_group((work_id,))

    def record_layer_completion(self, layer: int) -> dict[str, int | float | bool]:
        self.complete_layer(layer)
        return self.expose_stats()

    def expose_stats(self) -> dict[str, int | float | bool]:
        counts = self.ledger.state_counts
        return {
            "work_epoch": self.epoch,
            "work_units": len(self.batch.units),
            "work_requests": self._request_count,
            "work_ready": counts[Availability.READY],
            "work_blocked": counts[Availability.BLOCKED],
            "work_running": counts[Availability.RUNNING],
            "work_partial": counts[Availability.PARTIAL],
            "work_complete": counts[Availability.COMPLETE],
            "work_cancelled": counts[Availability.CANCELLED],
            "work_failed": counts[Availability.FAILED],
            "work_selected_units": self._selected_units,
            "work_candidate_units": self._candidate_units,
            "work_selected_bytes": self._selected_bytes,
            "work_candidate_bytes": self._candidate_bytes,
            "work_is_heterogeneous": self.batch.is_heterogeneous,
            "work_ready_fraction": self.batch.ready_fraction,
        }

    def unit_for_ticket(
        self,
        *,
        work_id: int,
        layer: int,
        logical_begin: int,
        request_index: int,
    ) -> WorkUnit:
        """Resolve one native ticket and validate its semantic coordinates.

        A logical KV coordinate is not necessarily unique inside a native
        schedule: split-K and other multi-contributor plans may emit several
        CTAs for one request and coordinate.  The schedule ordinal is the
        engine's canonical work-ticket identity; the remaining fields are
        checked as an invariant so a stale or reordered schedule fails closed.
        """
        unit = self._units_by_work_id.get(work_id)
        if unit is None:
            raise RuntimeError(
                "native schedule has no unique semantic work ticket: "
                f"work_id={work_id} layer={layer} logical={logical_begin} "
                f"request={request_index}"
            )
        if (
            unit.layer != layer
            or unit.logical_begin != logical_begin
            or unit.binding.request_index != request_index
        ):
            raise RuntimeError(
                "native schedule semantic coordinates diverged for work ticket: "
                f"work_id={work_id} expected=(layer={layer}, logical={logical_begin}, "
                f"request={request_index}) actual=(layer={unit.layer}, "
                f"logical={unit.logical_begin}, "
                f"request={unit.binding.request_index})"
            )
        return unit
