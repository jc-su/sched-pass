"""Request-level service frontier shared by acquisition and compute policy."""

from __future__ import annotations

from dataclasses import dataclass
import math
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


@dataclass(frozen=True)
class ServiceModel:
    """Online estimate for one request's currently selected data source."""

    bandwidth_bytes_per_second: int
    fixed_latency_ns: int = 0
    queue_delay_ns: int = 0
    reduction_ns: int = 0

    def validate(self) -> None:
        if self.bandwidth_bytes_per_second <= 0:
            raise ValueError("service bandwidth must be positive")
        if min(self.fixed_latency_ns, self.queue_delay_ns, self.reduction_ns) < 0:
            raise ValueError("service latency estimates cannot be negative")


@dataclass(frozen=True)
class RequestWork:
    """Generation-stamped work visible at the compiler/runtime boundary."""

    request_id: int
    generation: int
    priority: int
    deadline_ns: int
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
    tenant_virtual_time_ns: int = 0
    dropped_attributions: int = 0

    @classmethod
    def from_progress(
        cls,
        progress: Progress,
        *,
        priority: int = 0,
        deadline_ns: int = 0,
        tenant_virtual_time_ns: int = 0,
    ) -> "RequestWork":
        return cls(
            progress.request_id,
            progress.generation,
            priority,
            deadline_ns,
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
            tenant_virtual_time_ns,
            progress.dropped_attributions,
        )

    def validate(self) -> None:
        values = (
            self.request_id,
            self.generation,
            self.deadline_ns,
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
            self.tenant_virtual_time_ns,
            self.dropped_attributions,
        )
        if min(values) < 0 or self.priority < 0 or self.priority > 7:
            raise ValueError("request work contains an invalid counter or policy value")
        if self.dropped_attributions != 0:
            raise ValueError("request work contains dropped progress attribution")
        terminal = self.completed_work + self.failed_work + self.cancelled_work
        if self.pending_work + self.runnable_work + terminal > self.expected_work:
            raise ValueError("request work counters exceed expected contributors")
        accounted_compute = (
            self.pending_compute_ns
            + self.runnable_compute_ns
            + self.completed_compute_ns
        )
        if accounted_compute > self.expected_compute_ns:
            raise ValueError("request compute counters exceed expected service")
        if self.pending_work == 0 and (
            self.unavailable_bytes != 0 or self.pending_compute_ns != 0
        ):
            raise ValueError("request without pending work carries blocked service")

    @property
    def terminal(self) -> bool:
        return self.expected_work != 0 and (
            self.completed_work + self.failed_work + self.cancelled_work
            == self.expected_work
        )


@dataclass(frozen=True)
class CriticalWork:
    """Online request service estimate; this is a bound, not an oracle."""

    request: RequestWork
    acquisition_ns: int
    critical_path_ns: int
    slack_ns: int | None
    urgency: int

    @property
    def needs_data(self) -> bool:
        return self.request.pending_work != 0

    @property
    def has_executable_work(self) -> bool:
        return self.request.runnable_work != 0


@dataclass(frozen=True)
class CriticalWorkPlan:
    """One ordering contract for transport credits and executable contributors."""

    requests: tuple[CriticalWork, ...]
    data_order: tuple[tuple[int, int], ...]
    compute_order: tuple[tuple[int, int], ...]


def _acquisition_ns(request: RequestWork, model: ServiceModel) -> int:
    if request.pending_work == 0:
        return 0
    transfer_ns = math.ceil(
        request.unavailable_bytes
        * 1_000_000_000
        / model.bandwidth_bytes_per_second
    )
    return model.fixed_latency_ns + model.queue_delay_ns + transfer_ns


def _urgency(priority: int, critical_path_ns: int, slack_ns: int | None) -> int:
    """Mirror ``device::urgencyBucket`` for one request snapshot.

    ``slack_ns`` excludes the predicted critical path. Reconstructing the
    remaining deadline budget keeps host admission and device transport on
    the same quantized policy without publishing a second policy field.
    """
    if slack_ns is None:
        service_urgency = (
            0
            if critical_path_ns == 0
            else 3
            if critical_path_ns <= 100_000
            else 2
            if critical_path_ns <= 500_000
            else 1
            if critical_path_ns <= 2_000_000
            else 0
        )
        return max(priority, service_urgency)
    remaining_ns = max(0, slack_ns + critical_path_ns)
    deadline_urgency = (
        7
        if remaining_ns == 0
        or (critical_path_ns != 0 and remaining_ns <= critical_path_ns)
        else 6
        if critical_path_ns != 0 and remaining_ns <= 2 * critical_path_ns
        else 5
        if critical_path_ns != 0 and remaining_ns <= 4 * critical_path_ns
        else 4
        if remaining_ns <= 5_000_000
        else 0
    )
    return max(priority, deadline_urgency)


def estimate_critical_work(
    request: RequestWork, *, now_ns: int, model: ServiceModel
) -> CriticalWork:
    """Estimate remaining request service from current, observable state only."""

    request.validate()
    model.validate()
    if now_ns < 0:
        raise ValueError("current time cannot be negative")
    acquisition_ns = _acquisition_ns(request, model)
    # Executable contributors can overlap the selected transport. Contributors
    # still blocked on data execute afterwards; final reduction follows both.
    critical_path_ns = (
        max(acquisition_ns, request.runnable_compute_ns)
        + request.pending_compute_ns
        + (model.reduction_ns if not request.terminal else 0)
    )
    slack_ns = (
        None
        if request.deadline_ns == 0
        else request.deadline_ns - now_ns - critical_path_ns
    )
    return CriticalWork(
        request,
        acquisition_ns,
        critical_path_ns,
        slack_ns,
        _urgency(request.priority, critical_path_ns, slack_ns),
    )


def plan_critical_work(
    requests: Iterable[RequestWork], *, now_ns: int, model: ServiceModel
) -> CriticalWorkPlan:
    """Rank finite request work for data service and GPU contributor dispatch.

    The policy is causal: every input is current request state or an online
    service calibration. Request generation is part of every returned identity.
    """

    estimates = tuple(
        estimate_critical_work(request, now_ns=now_ns, model=model)
        for request in requests
    )
    identities = [
        (item.request.request_id, item.request.generation) for item in estimates
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("critical-work input repeats a request generation")

    def key(item: CriticalWork) -> tuple[int, int, int, int, int, int, int]:
        request = item.request
        deadline_class = 1 if item.slack_ns is None else 0
        slack = item.slack_ns if item.slack_ns is not None else 0
        return (
            int(request.terminal),
            -item.urgency,
            deadline_class,
            slack,
            request.tenant_virtual_time_ns,
            -item.critical_path_ns,
            request.request_id,
        )

    ordered = tuple(sorted(estimates, key=key))
    data_order = tuple(
        (item.request.request_id, item.request.generation)
        for item in ordered
        if item.needs_data and not item.request.terminal
    )
    compute_order = tuple(
        (item.request.request_id, item.request.generation)
        for item in ordered
        if item.has_executable_work and not item.request.terminal
    )
    return CriticalWorkPlan(ordered, data_order, compute_order)
