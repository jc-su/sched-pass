"""Bounded completion policy for finite GPU acquisition epochs."""

from __future__ import annotations

import dataclasses
from typing import Any

from .runtime import (
    EpochStatus,
    JitPhaseProgram,
    Runtime,
    RuntimeError,
    synchronize_stream,
)


@dataclasses.dataclass(frozen=True)
class EpochResult:
    status: EpochStatus
    progress_rounds: int


class BoundedEpoch:
    """Run a layer epoch with a finite number of GPU progress launches."""

    def __init__(
        self,
        phases: JitPhaseProgram,
        runtime: Runtime,
        *,
        object_count: int,
        work_ticket_count: int,
        max_progress_rounds: int,
    ) -> None:
        if object_count < 0 or min(work_ticket_count, max_progress_rounds) <= 0:
            raise ValueError(
                "bounded epoch object count cannot be negative and ticket/round "
                "counts must be positive"
            )
        self.phases = phases
        self.runtime = runtime
        self.object_count = object_count
        self.work_ticket_count = work_ticket_count
        self.max_progress_rounds = max_progress_rounds

    def _status(self, stream: Any) -> EpochStatus:
        synchronize_stream(stream)
        status = self.runtime.epoch_status(self.work_ticket_count)
        if status.has_failure:
            raise RuntimeError(
                "acquisition epoch failed "
                f"(failed={status.failed}, cancelled={status.cancelled})"
            )
        return status

    def _exhausted(self, status: EpochStatus) -> RuntimeError:
        return RuntimeError(
            f"acquisition epoch exhausted {self.max_progress_rounds} progress rounds "
            f"(new={status.fresh}, pending={status.pending}, ready={status.ready}, "
            f"initializing={status.initializing})"
        )

    def check(self, progress_rounds: int, stream: Any = None) -> EpochResult:
        """Check a completed eager launch or graph replay and fail closed."""
        status = self._status(stream)
        if not status.succeeded:
            raise self._exhausted(status)
        return EpochResult(status, progress_rounds)
