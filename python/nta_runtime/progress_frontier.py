"""Generation-safe compiler progress at the framework admission boundary.

This module intentionally describes state, not a scheduling policy.  The
compiler/runtime publish contributor progress; an engine may use the resulting
frontier to decide whether resident work can run while another batch acquires
data.  No ordering is promised unless an engine actually consumes one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Protocol


class Progress(Protocol):
    request_id: int
    generation: int
    expected_work: int
    pending_work: int
    runnable_work: int
    completed_work: int
    failed_work: int
    cancelled_work: int
    unavailable_bytes: int
    runnable_compute_ns: int
    completed_compute_ns: int
    pending_compute_ns: int
    expected_compute_ns: int
    dropped_attributions: int


class FrontierState(str, Enum):
    EXECUTABLE = "executable"
    DATA_BLOCKED = "data_blocked"
    QUIESCENT = "quiescent"


@dataclass(frozen=True)
class RequestFrontierEntry:
    """One compiler-attributed request generation in a progress snapshot."""

    request_id: int
    generation: int
    expected_work: int
    pending_work: int
    runnable_work: int
    completed_work: int
    failed_work: int
    cancelled_work: int
    unavailable_bytes: int
    pending_compute_ns: int
    runnable_compute_ns: int
    completed_compute_ns: int
    expected_compute_ns: int
    dropped_attributions: int = 0

    @classmethod
    def from_progress(cls, progress: Progress) -> "RequestFrontierEntry":
        return cls(
            progress.request_id,
            progress.generation,
            progress.expected_work,
            progress.pending_work,
            progress.runnable_work,
            progress.completed_work,
            progress.failed_work,
            progress.cancelled_work,
            progress.unavailable_bytes,
            progress.pending_compute_ns,
            progress.runnable_compute_ns,
            progress.completed_compute_ns,
            progress.expected_compute_ns,
            progress.dropped_attributions,
        )

    @property
    def identity(self) -> tuple[int, int]:
        return self.request_id, self.generation

    @property
    def terminal(self) -> bool:
        return self.expected_work != 0 and (
            self.completed_work + self.failed_work + self.cancelled_work
            == self.expected_work
        )

    def validate(self) -> None:
        if (
            min(
                self.request_id,
                self.generation,
                self.expected_work,
                self.pending_work,
                self.runnable_work,
                self.completed_work,
                self.failed_work,
                self.cancelled_work,
                self.unavailable_bytes,
                self.pending_compute_ns,
                self.runnable_compute_ns,
                self.completed_compute_ns,
                self.expected_compute_ns,
                self.dropped_attributions,
            )
            < 0
        ):
            raise ValueError("request frontier contains a negative counter")
        if self.dropped_attributions != 0:
            raise ValueError("request frontier contains dropped progress attribution")
        terminal = self.completed_work + self.failed_work + self.cancelled_work
        if self.pending_work + self.runnable_work + terminal > self.expected_work:
            raise ValueError("request frontier counters exceed expected contributors")
        accounted_compute = (
            self.pending_compute_ns
            + self.runnable_compute_ns
            + self.completed_compute_ns
        )
        if accounted_compute > self.expected_compute_ns:
            raise ValueError("request frontier compute exceeds expected service")
        if self.pending_work == 0 and (
            self.unavailable_bytes != 0 or self.pending_compute_ns != 0
        ):
            raise ValueError("request without pending work carries blocked service")
        if self.runnable_work == 0 and self.runnable_compute_ns != 0:
            raise ValueError("request without runnable work carries runnable service")


@dataclass(frozen=True)
class RequestFrontier:
    """A causal state summary consumed by framework admission."""

    requests: tuple[RequestFrontierEntry, ...]

    def __post_init__(self) -> None:
        if not self.requests:
            raise ValueError("request frontier cannot be empty")
        for request in self.requests:
            request.validate()
        identities = tuple(request.identity for request in self.requests)
        if len(set(identities)) != len(identities):
            raise ValueError("request frontier repeats a request generation")

    @property
    def executable(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            request.identity
            for request in self.requests
            if request.runnable_work != 0 and not request.terminal
        )

    @property
    def data_blocked(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            request.identity
            for request in self.requests
            if request.pending_work != 0 and not request.terminal
        )

    @property
    def state(self) -> FrontierState:
        # Runnable resident work wins over blocked contributors because it is
        # useful overlap for admission even when the same request also waits.
        if self.executable:
            return FrontierState.EXECUTABLE
        if self.data_blocked:
            return FrontierState.DATA_BLOCKED
        return FrontierState.QUIESCENT


def build_request_frontier(
    requests: Iterable[RequestFrontierEntry],
) -> RequestFrontier:
    return RequestFrontier(tuple(requests))
