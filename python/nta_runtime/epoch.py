"""Bounded completion policy for finite GPU acquisition epochs."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from .runtime import EpochStatus, JitPhaseProgram, Runtime, RuntimeError, synchronize_stream


@dataclasses.dataclass(frozen=True)
class EpochResult:
    status: EpochStatus
    progress_passes: int


class BoundedEpoch:
    """Run a layer epoch with a finite number of GPU progress launches."""

    def __init__(
        self,
        phases: JitPhaseProgram,
        runtime: Runtime,
        *,
        object_count: int,
        work_ticket_count: int,
        max_progress_passes: int,
    ) -> None:
        if min(object_count, work_ticket_count, max_progress_passes) <= 0:
            raise ValueError("bounded epoch counts must be positive")
        self.phases = phases
        self.runtime = runtime
        self.object_count = object_count
        self.work_ticket_count = work_ticket_count
        self.max_progress_passes = max_progress_passes

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
            f"acquisition epoch exhausted {self.max_progress_passes} progress passes "
            f"(new={status.fresh}, pending={status.pending}, ready={status.ready}, "
            f"initializing={status.initializing})"
        )

    def run_host(
        self,
        initial: Callable[[], None],
        ready: Callable[[], None],
        *,
        progress_blocks: int,
        stream: Any = None,
    ) -> EpochResult:
        if progress_blocks <= 0:
            raise ValueError("host progress block count must be positive")
        self.phases.reset(
            self.runtime, self.object_count, self.work_ticket_count, stream
        )
        initial()
        self.phases.complete(self.runtime, self.work_ticket_count, stream)
        status = self._status(stream)
        if status.succeeded:
            return EpochResult(status, 0)

        for progress_pass in range(1, self.max_progress_passes + 1):
            self.phases.progress_host(self.runtime, progress_blocks, stream)
            ready()
            self.phases.complete(self.runtime, self.work_ticket_count, stream)
            status = self._status(stream)
            if status.succeeded:
                return EpochResult(status, progress_pass)
        raise self._exhausted(status)

    def run_host_fixed(
        self,
        initial: Callable[[], None],
        ready: Callable[[int, bool], None],
        *,
        progress_blocks: int,
        stream: Any = None,
    ) -> EpochResult:
        """Enqueue every bounded host round and check once at the boundary."""
        self.enqueue_host_fixed(
            initial, ready, progress_blocks=progress_blocks, stream=stream
        )
        return self.check(self.max_progress_passes, stream)

    def enqueue_host_fixed(
        self,
        initial: Callable[[], None],
        ready: Callable[[int, bool], None],
        *,
        progress_blocks: int,
        stream: Any = None,
    ) -> None:
        """Enqueue a graph-capturable fixed host epoch without synchronizing."""
        if progress_blocks <= 0:
            raise ValueError("host progress block count must be positive")
        self.phases.reset(
            self.runtime, self.object_count, self.work_ticket_count, stream
        )
        initial()
        self.phases.complete(self.runtime, self.work_ticket_count, stream)
        for progress_pass in range(1, self.max_progress_passes + 1):
            self.phases.progress_host(self.runtime, progress_blocks, stream)
            ready(progress_pass, progress_pass == self.max_progress_passes)
            self.phases.complete(self.runtime, self.work_ticket_count, stream)

    def run_nvme(
        self,
        initial: Callable[[], None],
        ready: Callable[[], None],
        *,
        issue_budget: int,
        completion_budget: int,
        stream: Any = None,
    ) -> EpochResult:
        if issue_budget <= 0 or completion_budget <= 0:
            raise ValueError("NVMe issue and completion budgets must be positive")
        self.phases.reset(
            self.runtime, self.object_count, self.work_ticket_count, stream
        )
        initial()
        self.phases.complete(self.runtime, self.work_ticket_count, stream)
        status = self._status(stream)
        if status.succeeded:
            return EpochResult(status, 0)

        for progress_pass in range(1, self.max_progress_passes + 1):
            self.phases.progress_nvme(
                self.runtime, issue_budget, completion_budget, stream
            )
            ready()
            self.phases.complete(self.runtime, self.work_ticket_count, stream)
            status = self._status(stream)
            if status.succeeded:
                return EpochResult(status, progress_pass)
        raise self._exhausted(status)

    def run_nvme_fixed(
        self,
        initial: Callable[[], None],
        ready: Callable[[int, bool], None],
        *,
        issue_budget: int,
        completion_budget: int,
        stream: Any = None,
    ) -> EpochResult:
        """Enqueue every bounded NVMe round and check once at the boundary."""
        self.enqueue_nvme_fixed(
            initial,
            ready,
            issue_budget=issue_budget,
            completion_budget=completion_budget,
            stream=stream,
        )
        return self.check(self.max_progress_passes, stream)

    def enqueue_nvme_fixed(
        self,
        initial: Callable[[], None],
        ready: Callable[[int, bool], None],
        *,
        issue_budget: int,
        completion_budget: int,
        stream: Any = None,
    ) -> None:
        """Enqueue a graph-capturable fixed NVMe epoch without synchronizing."""
        if issue_budget <= 0 or completion_budget <= 0:
            raise ValueError("NVMe issue and completion budgets must be positive")
        self.phases.reset(
            self.runtime, self.object_count, self.work_ticket_count, stream
        )
        initial()
        self.phases.complete(self.runtime, self.work_ticket_count, stream)
        for progress_pass in range(1, self.max_progress_passes + 1):
            self.phases.progress_nvme(
                self.runtime, issue_budget, completion_budget, stream
            )
            ready(progress_pass, progress_pass == self.max_progress_passes)
            self.phases.complete(self.runtime, self.work_ticket_count, stream)

    def check(self, progress_passes: int, stream: Any = None) -> EpochResult:
        """Check a completed eager launch or graph replay and fail closed."""
        status = self._status(stream)
        if not status.succeeded:
            raise self._exhausted(status)
        return EpochResult(status, progress_passes)
