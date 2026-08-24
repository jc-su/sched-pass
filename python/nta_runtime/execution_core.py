"""Engine-neutral execution session for one heterogeneous forward epoch.

The engine adapter owns request metadata.  The compiler/runtime owns the
native storage ABI.  This module is the semantic bridge between them: it
turns an engine-independent list of logical tiles into one validated
``WorkBatch`` and keeps every availability transition generation- and
epoch-checked.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    """Engine-neutral description of one logical consumer tile."""

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
        if not self.selected_ids:
            raise ValueError("execution tile must identify selected units")
        if len(set(self.selected_ids)) != len(self.selected_ids):
            raise ValueError("execution tile selected units must be unique")
        if min(self.selected_ids) < 0 or max(self.selected_ids) >= self.candidate_units:
            raise ValueError("execution tile selected ID is outside its candidate set")


@dataclass
class ExecutionSession:
    """The sole semantic execution state for one engine forward.

    The native runtime may execute the concrete transport and CUDA work, but
    it receives its identity from this session.  No engine-specific code is
    allowed to maintain a second availability or generation state machine.
    """

    batch: WorkBatch
    protocol: ExecutionProtocolConfig
    ledger: WorkLedger

    @classmethod
    def from_tiles(
        cls,
        *,
        epoch: int,
        granularity: Granularity,
        protocol: ExecutionProtocolConfig,
        tiles: Iterable[ExecutionTile],
    ) -> "ExecutionSession":
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
                    selected_units=len(tile.selected_ids),
                    unit_bytes=tile.unit_bytes,
                    granularity=granularity,
                    semantics=(
                        DemandSemantics.EXACT_DENSE
                        if len(tile.selected_ids) == tile.candidate_units
                        else DemandSemantics.EXACT_SPARSE
                    ),
                    provider="engine.schedule",
                    epoch=epoch,
                    selected_ids=tile.selected_ids,
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
        batch = WorkBatch(epoch, granularity, units)
        return cls(batch, protocol, WorkLedger(batch, protocol))

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
            return next(unit.binding for unit in self.batch.units if unit.work_id == work_id)
        except StopIteration as error:
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
        for work_id in work_ids:
            if self.ledger.state(work_id) is not Availability.READY:
                raise RuntimeError(f"work unit {work_id} is not runnable")
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
        layer_ids = tuple(
            unit.work_id for unit in self.batch.units if unit.layer == layer
        )
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
        selected = sum(unit.demand.selected_units for unit in self.batch.units)
        candidates = sum(unit.demand.candidate_units for unit in self.batch.units)
        selected_bytes = sum(unit.demand.selected_bytes for unit in self.batch.units)
        candidate_bytes = sum(unit.demand.candidate_bytes for unit in self.batch.units)
        return {
            "work_epoch": self.epoch,
            "work_units": len(self.batch.units),
            "work_requests": len(self.request_identities),
            "work_ready": counts[Availability.READY],
            "work_blocked": counts[Availability.BLOCKED],
            "work_running": counts[Availability.RUNNING],
            "work_partial": counts[Availability.PARTIAL],
            "work_complete": counts[Availability.COMPLETE],
            "work_cancelled": counts[Availability.CANCELLED],
            "work_failed": counts[Availability.FAILED],
            "work_selected_units": selected,
            "work_candidate_units": candidates,
            "work_selected_bytes": selected_bytes,
            "work_candidate_bytes": candidate_bytes,
            "work_is_heterogeneous": self.batch.is_heterogeneous,
            "work_ready_fraction": self.batch.ready_fraction,
        }

    def unit_for(
        self, *, layer: int, logical_begin: int, request_index: int
    ) -> WorkUnit:
        matches = tuple(
            unit
            for unit in self.batch.units
            if unit.layer == layer
            and unit.logical_begin == logical_begin
            and unit.binding.request_index == request_index
        )
        if len(matches) != 1:
            raise RuntimeError(
                "native schedule has no unique semantic work unit: "
                f"layer={layer} logical={logical_begin} request={request_index}"
            )
        return matches[0]
