"""Tier-neutral deadline scheduling for exact acquisition jobs.

The compiler and framework adapters prove *what* bytes a work unit consumes.
Transport backends own *how* those bytes move.  This module is the narrow
boundary between them: measured service demand is scheduled against numerical
consumer deadlines without importing CUDA, a framework, or a tier backend.

The core EDF theorem used here is intentionally explicit.  Jobs passed to
``schedule_acquisition_jobs`` are all available at time zero and share
one serialized link.  EDF minimizes maximum lateness for that model, so the
cumulative inequalities are an exact feasibility test.  Backends must not use
the result as a proof when jobs have different release times, hidden setup
work, or an independently contended link; those cases require a different
model rather than an optimistic flag.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
import heapq
from threading import Lock


_UINT64_MAX = (1 << 64) - 1


@dataclass(frozen=True, slots=True)
class TenantCreditCharge:
    """Exact bytes one transport lease owns on behalf of a tenant."""

    tenant_id: int
    bytes: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.tenant_id, bool)
            or isinstance(self.bytes, bool)
            or not isinstance(self.tenant_id, int)
            or not isinstance(self.bytes, int)
            or self.tenant_id < 0
            or self.tenant_id >= 1 << 32
            or self.bytes <= 0
            or self.bytes > _UINT64_MAX
        ):
            raise ValueError("tenant acquisition charge is outside the ABI")


@dataclass(frozen=True, slots=True)
class TenantCreditLease:
    """Opaque all-or-nothing reservation returned by a credit ledger."""

    lease_id: int
    charges: tuple[TenantCreditCharge, ...]

    def __post_init__(self) -> None:
        if self.lease_id <= 0 or not self.charges:
            raise ValueError("tenant credit lease is empty")
        if tuple(sorted(self.charges, key=lambda charge: charge.tenant_id)) != (
            self.charges
        ) or len({charge.tenant_id for charge in self.charges}) != len(self.charges):
            raise ValueError("tenant credit lease charges are not canonical")


class TenantCreditLedger:
    """Thread-safe control-plane credits for externally issued transfers.

    Device-initiated paths continue to use runtime atomics.  Copy-engine paths
    have no device acquisition instruction, so their transport owner reserves
    the same finite byte resource before submission and releases it only after
    the readiness fence completes. Missing tenant entries retain the runtime's
    unlimited default.
    """

    def __init__(self, budgets: Iterable[tuple[int, int]]) -> None:
        values = tuple((int(tenant), int(limit)) for tenant, limit in budgets)
        if len({tenant for tenant, _ in values}) != len(values) or any(
            tenant < 0 or tenant >= 1 << 32 or limit < 0 or limit > _UINT64_MAX
            for tenant, limit in values
        ):
            raise ValueError("tenant credit budgets are invalid")
        self._budgets = dict(values)
        self._outstanding: dict[int, int] = {}
        # Leases are process-local capabilities, not serializable identifiers.
        # Retaining the returned object lets release reject a lease minted by
        # another ledger even when both ledgers happen to use the same numeric
        # sequence and charges.
        self._leases: dict[int, TenantCreditLease] = {}
        self._next_lease_id = 1
        self._lock = Lock()

    @property
    def finite(self) -> bool:
        return any(limit != _UINT64_MAX for limit in self._budgets.values())

    @property
    def active_lease_count(self) -> int:
        with self._lock:
            return len(self._leases)

    def outstanding_bytes(self, tenant_id: int) -> int:
        if tenant_id < 0 or tenant_id >= 1 << 32:
            raise ValueError("tenant ID is outside the runtime ABI")
        with self._lock:
            return self._outstanding.get(tenant_id, 0)

    @staticmethod
    def _canonical_charges(
        charges: Iterable[TenantCreditCharge],
    ) -> tuple[TenantCreditCharge, ...]:
        totals: dict[int, int] = {}
        for charge in charges:
            if not isinstance(charge, TenantCreditCharge):
                raise TypeError("tenant credit reservations require typed charges")
            total = totals.get(charge.tenant_id, 0) + charge.bytes
            if total > _UINT64_MAX:
                raise OverflowError("tenant acquisition charge exceeds uint64")
            totals[charge.tenant_id] = total
        if not totals:
            raise ValueError("tenant credit reservation is empty")
        return tuple(
            TenantCreditCharge(tenant, total)
            for tenant, total in sorted(totals.items())
        )

    def try_reserve(
        self, charges: Iterable[TenantCreditCharge]
    ) -> TenantCreditLease | None:
        canonical = self._canonical_charges(charges)
        with self._lock:
            for charge in canonical:
                outstanding = self._outstanding.get(charge.tenant_id, 0)
                limit = self._budgets.get(charge.tenant_id, _UINT64_MAX)
                if charge.bytes > limit - outstanding:
                    return None
            lease_id = self._next_lease_id
            self._next_lease_id += 1
            for charge in canonical:
                self._outstanding[charge.tenant_id] = (
                    self._outstanding.get(charge.tenant_id, 0) + charge.bytes
                )
            lease = TenantCreditLease(lease_id, canonical)
            self._leases[lease_id] = lease
            return lease

    def release(self, lease: TenantCreditLease) -> None:
        if not isinstance(lease, TenantCreditLease):
            raise TypeError("tenant credit release requires a typed lease")
        with self._lock:
            owned = self._leases.get(lease.lease_id)
            if owned is not lease:
                raise RuntimeError("tenant credit lease is stale or foreign")
            charges = owned.charges
            # Validate the complete transaction before mutating any tenant.
            # An invariant failure must not leave a partially released lease.
            for charge in charges:
                outstanding = self._outstanding.get(charge.tenant_id, 0)
                if outstanding < charge.bytes:
                    raise RuntimeError("tenant credit accounting underflow")
            for charge in charges:
                outstanding = self._outstanding[charge.tenant_id]
                remaining = outstanding - charge.bytes
                if remaining:
                    self._outstanding[charge.tenant_id] = remaining
                else:
                    self._outstanding.pop(charge.tenant_id, None)
            self._leases.pop(lease.lease_id)


@dataclass(frozen=True, order=True, slots=True)
class AcquisitionGroupIdentity:
    """Stable semantic identity of one exact K/V acquisition group.

    A transport submission ID is deliberately not an identity.  Slot reuse,
    layer reuse, and tier-object replacement can all preserve the same local
    ordinal while naming different bytes.  Every scheduler and consumer join
    therefore carries the complete request-generation, layer, exact segment,
    and resource-version tuple.
    """

    request_slot: int
    request_generation: int
    layer_id: int
    segment_begin: int
    segment_count: int
    resource_version: int

    def __post_init__(self) -> None:
        fields = (
            self.request_slot,
            self.request_generation,
            self.layer_id,
            self.segment_begin,
            self.segment_count,
            self.resource_version,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in fields):
            raise TypeError("acquisition-group identity fields must be integers")
        if not 0 <= self.request_slot < 1 << 32:
            raise ValueError("acquisition-group request slot exceeds uint32")
        if not 0 < self.request_generation < 1 << 32:
            raise ValueError("acquisition-group generation must be positive uint32")
        if not 0 <= self.layer_id < 1 << 32:
            raise ValueError("acquisition-group layer exceeds uint32")
        if self.segment_begin < 0 or self.segment_count <= 0:
            raise ValueError("acquisition-group segment geometry is invalid")
        if not 0 < self.resource_version <= _UINT64_MAX:
            raise ValueError("acquisition-group resource version is invalid")
        if self.segment_begin + self.segment_count > _UINT64_MAX:
            raise ValueError("acquisition-group segment exceeds uint64")

    @property
    def segment_end(self) -> int:
        return self.segment_begin + self.segment_count


@dataclass(frozen=True, slots=True)
class SharedAcquisitionJob:
    """One finite, non-preemptive group on a shared transport link.

    Times are absolute values in one monotonic clock domain.  ``staging_bytes``
    is the amount retained from dispatch until physical readiness; it can
    differ from payload bytes for a backend that materializes compressed or
    packed objects.  Priority is only an equal-deadline tie break: the policy
    remains EDF and does not silently become framework priority scheduling.
    """

    identity: AcquisitionGroupIdentity
    tenant_id: int
    payload_bytes: int
    staging_bytes: int
    release_ns: int
    service_ns: int
    deadline_ns: int
    priority: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.identity, AcquisitionGroupIdentity):
            raise TypeError("shared acquisition requires a typed group identity")
        values = (
            self.tenant_id,
            self.payload_bytes,
            self.staging_bytes,
            self.release_ns,
            self.service_ns,
            self.deadline_ns,
            self.priority,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("shared acquisition job fields must be integers")
        if not 0 <= self.tenant_id < 1 << 32:
            raise ValueError("shared acquisition tenant exceeds uint32")
        if not 0 < self.payload_bytes <= _UINT64_MAX:
            raise ValueError("shared acquisition payload is invalid")
        if not 0 < self.staging_bytes <= _UINT64_MAX:
            raise ValueError("shared acquisition staging charge is invalid")
        if self.release_ns < 0 or self.service_ns <= 0 or self.deadline_ns < 0:
            raise ValueError("shared acquisition timing is invalid")
        if not 0 <= self.priority <= 7:
            raise ValueError("shared acquisition priority is outside the runtime ABI")


@dataclass(frozen=True, slots=True)
class SharedAcquisitionSchedule:
    """Auditable work-conserving EDF schedule with arbitrary release times."""

    ordered_identities: tuple[AcquisitionGroupIdentity, ...]
    start_ns: tuple[int, ...]
    completion_ns: tuple[int, ...]
    deadlines_ns: tuple[int, ...]
    first_missed_identity: AcquisitionGroupIdentity | None
    maximum_lateness_ns: int

    def __post_init__(self) -> None:
        count = len(self.ordered_identities)
        if (
            len(set(self.ordered_identities)) != count
            or len(self.start_ns) != count
            or len(self.completion_ns) != count
            or len(self.deadlines_ns) != count
            or any(start < 0 for start in self.start_ns)
            or any(
                completion <= start
                for start, completion in zip(self.start_ns, self.completion_ns, strict=True)
            )
            or any(deadline < 0 for deadline in self.deadlines_ns)
            or self.maximum_lateness_ns < 0
        ):
            raise ValueError("shared acquisition schedule is inconsistent")
        if self.first_missed_identity is not None and (
            self.first_missed_identity not in self.ordered_identities
        ):
            raise ValueError("shared acquisition miss is outside the schedule")
        if self.feasible != (self.maximum_lateness_ns == 0):
            raise ValueError("shared acquisition feasibility disagrees with lateness")

    @property
    def feasible(self) -> bool:
        return self.first_missed_identity is None


def schedule_shared_acquisition_jobs(
    jobs: Iterable[SharedAcquisitionJob],
    *,
    available_ns: int = 0,
) -> SharedAcquisitionSchedule:
    """Simulate the exact non-preemptive EDF policy used by the shared link.

    EDF is optimal for the simultaneous-release model used by
    :func:`schedule_acquisition_jobs`; it is not generally optimal with release
    times.  This function intentionally makes the narrower claim needed by the
    implementation: the returned feasibility is exact for *our work-conserving
    non-preemptive EDF policy*.  A paper result must not relabel it as a global
    schedulability theorem.
    """

    if isinstance(available_ns, bool) or not isinstance(available_ns, int):
        raise TypeError("shared-link availability must be an integer")
    if available_ns < 0:
        raise ValueError("shared-link availability cannot be negative")
    pending = sorted(
        tuple(jobs), key=lambda job: (job.release_ns, job.identity)
    )
    if len({job.identity for job in pending}) != len(pending):
        raise ValueError("shared acquisition jobs must have unique identities")

    ready: list[tuple[int, int, AcquisitionGroupIdentity, SharedAcquisitionJob]] = []
    cursor = 0
    now = available_ns
    ordered: list[AcquisitionGroupIdentity] = []
    starts: list[int] = []
    completions: list[int] = []
    deadlines: list[int] = []
    first_missed: AcquisitionGroupIdentity | None = None
    maximum_lateness = 0
    while cursor < len(pending) or ready:
        if not ready and cursor < len(pending) and pending[cursor].release_ns > now:
            now = pending[cursor].release_ns
        while cursor < len(pending) and pending[cursor].release_ns <= now:
            job = pending[cursor]
            heapq.heappush(
                ready,
                (job.deadline_ns, -job.priority, job.identity, job),
            )
            cursor += 1
        if not ready:
            continue
        _deadline, _priority, _identity, job = heapq.heappop(ready)
        start = now
        now += job.service_ns
        lateness = now - job.deadline_ns
        if lateness > 0:
            maximum_lateness = max(maximum_lateness, lateness)
            if first_missed is None:
                first_missed = job.identity
        ordered.append(job.identity)
        starts.append(start)
        completions.append(now)
        deadlines.append(job.deadline_ns)
    return SharedAcquisitionSchedule(
        tuple(ordered),
        tuple(starts),
        tuple(completions),
        tuple(deadlines),
        first_missed,
        maximum_lateness,
    )


class SharedAcquisitionState(str, Enum):
    PLANNED = "planned"
    SUBMITTED = "submitted"
    FENCE_PUBLISHED = "fence_published"
    READY = "ready"
    CONSUMED = "consumed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(slots=True)
class _SharedAcquisitionRecord:
    job: SharedAcquisitionJob
    state: SharedAcquisitionState = SharedAcquisitionState.PLANNED
    submitted_ns: int | None = None
    predicted_completion_ns: int | None = None
    tenant_lease: TenantCreditLease | None = None
    consumer_ordered: bool = False
    cancel_requested: bool = False


class SharedAcquisitionQueue:
    """Dynamic multi-request queue for one serialized transport link.

    The queue accepts groups from multiple framework batches.  Dispatch is
    deliberately bounded: only ``max_inflight_groups`` finite groups may be
    irrevocably ordered on the backend at once, so a later urgent request is
    delayed by at most that chunk horizon rather than by an entire model.  Both
    staging bytes and tenant credits are reserved atomically before a group is
    returned to the backend.
    """

    def __init__(
        self,
        *,
        staging_capacity_bytes: int,
        tenant_credits: TenantCreditLedger,
        max_inflight_groups: int = 1,
    ) -> None:
        if (
            isinstance(staging_capacity_bytes, bool)
            or not isinstance(staging_capacity_bytes, int)
            or not 0 < staging_capacity_bytes <= _UINT64_MAX
        ):
            raise ValueError("shared acquisition staging capacity is invalid")
        if (
            isinstance(max_inflight_groups, bool)
            or not isinstance(max_inflight_groups, int)
            or max_inflight_groups <= 0
        ):
            raise ValueError("shared acquisition dispatch horizon is invalid")
        if not isinstance(tenant_credits, TenantCreditLedger):
            raise TypeError("shared acquisition requires a tenant credit ledger")
        self._staging_capacity_bytes = staging_capacity_bytes
        self._tenant_credits = tenant_credits
        self._max_inflight_groups = max_inflight_groups
        self._records: dict[AcquisitionGroupIdentity, _SharedAcquisitionRecord] = {}
        self._staging_outstanding_bytes = 0
        self._predicted_link_tail_ns = 0

    @property
    def staging_outstanding_bytes(self) -> int:
        return self._staging_outstanding_bytes

    @property
    def group_count(self) -> int:
        return len(self._records)

    @property
    def inflight_count(self) -> int:
        return sum(
            record.state
            in {
                SharedAcquisitionState.SUBMITTED,
                SharedAcquisitionState.FENCE_PUBLISHED,
            }
            for record in self._records.values()
        )

    def state(self, identity: AcquisitionGroupIdentity) -> SharedAcquisitionState:
        try:
            return self._records[identity].state
        except KeyError as error:
            raise KeyError("unknown shared acquisition group") from error

    def add(self, jobs: Iterable[SharedAcquisitionJob]) -> None:
        values = tuple(jobs)
        if len({job.identity for job in values}) != len(values):
            raise ValueError("shared acquisition batch repeats a group identity")
        duplicates = tuple(job.identity for job in values if job.identity in self._records)
        if duplicates:
            raise ValueError(f"shared acquisition group already exists: {duplicates[0]!r}")
        oversized = next(
            (job for job in values if job.staging_bytes > self._staging_capacity_bytes),
            None,
        )
        if oversized is not None:
            raise ValueError("one acquisition group exceeds the staging capacity")
        self._records.update(
            (job.identity, _SharedAcquisitionRecord(job)) for job in values
        )

    def analyze(self, *, now_ns: int) -> SharedAcquisitionSchedule:
        """Analyze fixed in-flight work plus every dynamically ordered group."""

        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
            raise ValueError("shared acquisition analysis time is invalid")
        fixed = sorted(
            (
                record
                for record in self._records.values()
                if record.state
                in {
                    SharedAcquisitionState.SUBMITTED,
                    SharedAcquisitionState.FENCE_PUBLISHED,
                }
            ),
            key=lambda record: (
                record.predicted_completion_ns
                if record.predicted_completion_ns is not None
                else _UINT64_MAX,
                record.job.identity,
            ),
        )
        if any(
            record.predicted_completion_ns is None or record.submitted_ns is None
            for record in fixed
        ):
            raise RuntimeError("shared acquisition in-flight prediction is incomplete")
        jobs = tuple(
            record.job
            for record in self._records.values()
            if record.state is SharedAcquisitionState.PLANNED
        )
        dynamic = schedule_shared_acquisition_jobs(
            jobs,
            available_ns=max(now_ns, self._predicted_link_tail_ns),
        )
        fixed_identities = tuple(record.job.identity for record in fixed)
        fixed_starts = tuple(
            int(record.predicted_completion_ns) - record.job.service_ns
            for record in fixed
        )
        fixed_completions = tuple(
            int(record.predicted_completion_ns) for record in fixed
        )
        fixed_deadlines = tuple(record.job.deadline_ns for record in fixed)
        first_missed = next(
            (
                record.job.identity
                for record in fixed
                if int(record.predicted_completion_ns) > record.job.deadline_ns
            ),
            dynamic.first_missed_identity,
        )
        maximum_lateness = max(
            dynamic.maximum_lateness_ns,
            max(
                (
                    int(record.predicted_completion_ns) - record.job.deadline_ns
                    for record in fixed
                ),
                default=0,
            ),
            0,
        )
        return SharedAcquisitionSchedule(
            fixed_identities + dynamic.ordered_identities,
            fixed_starts + dynamic.start_ns,
            fixed_completions + dynamic.completion_ns,
            fixed_deadlines + dynamic.deadlines_ns,
            first_missed,
            maximum_lateness,
        )

    def claim(self, *, now_ns: int) -> tuple[SharedAcquisitionJob, ...]:
        """Reserve and return every currently available dispatch slot."""

        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
            raise ValueError("shared acquisition dispatch time is invalid")
        available_slots = self._max_inflight_groups - self.inflight_count
        if available_slots <= 0:
            return ()
        candidates = sorted(
            (
                record.job
                for record in self._records.values()
                if record.state is SharedAcquisitionState.PLANNED
                and record.job.release_ns <= now_ns
            ),
            key=lambda job: (
                job.deadline_ns,
                -job.priority,
                job.identity,
            ),
        )
        claimed: list[SharedAcquisitionJob] = []
        for job in candidates:
            if len(claimed) == available_slots:
                break
            if (
                job.staging_bytes
                > self._staging_capacity_bytes - self._staging_outstanding_bytes
            ):
                continue
            tenant_lease = self._tenant_credits.try_reserve(
                (TenantCreditCharge(job.tenant_id, job.staging_bytes),)
            )
            if tenant_lease is None:
                continue
            record = self._records[job.identity]
            record.state = SharedAcquisitionState.SUBMITTED
            record.submitted_ns = now_ns
            predicted_start = max(now_ns, self._predicted_link_tail_ns)
            record.predicted_completion_ns = predicted_start + job.service_ns
            self._predicted_link_tail_ns = record.predicted_completion_ns
            record.tenant_lease = tenant_lease
            self._staging_outstanding_bytes += job.staging_bytes
            claimed.append(job)
        return tuple(claimed)

    def next_released_identity(
        self, *, now_ns: int
    ) -> AcquisitionGroupIdentity | None:
        """Return the next policy choice without reserving or mutating it."""

        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
            raise ValueError("shared acquisition dispatch time is invalid")
        candidates = (
            record.job
            for record in self._records.values()
            if record.state is SharedAcquisitionState.PLANNED
            and record.job.release_ns <= now_ns
        )
        selected = min(
            candidates,
            key=lambda job: (job.deadline_ns, -job.priority, job.identity),
            default=None,
        )
        return None if selected is None else selected.identity

    def released_identities(
        self, *, now_ns: int
    ) -> tuple[AcquisitionGroupIdentity, ...]:
        """Return resource-unfiltered policy order without mutating state."""

        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
            raise ValueError("shared acquisition dispatch time is invalid")
        return tuple(
            job.identity
            for job in sorted(
                (
                    record.job
                    for record in self._records.values()
                    if record.state is SharedAcquisitionState.PLANNED
                    and record.job.release_ns <= now_ns
                ),
                key=lambda job: (job.deadline_ns, -job.priority, job.identity),
            )
        )

    def claim_cohort(
        self,
        identities: Iterable[AcquisitionGroupIdentity],
        *,
        now_ns: int,
    ) -> tuple[SharedAcquisitionJob, ...]:
        """Atomically reserve one backend-coalesced finite dispatch cohort.

        Some engines can issue several exact request segments more efficiently
        as one layer packet.  Coalescing is legal only after EDF chooses one
        member: the caller supplies that member's complete physical cohort and
        every constituent retains independent identity, credit, readiness, and
        consumption state.
        """

        values = tuple(identities)
        if not values or len(set(values)) != len(values):
            raise ValueError("shared acquisition cohort must be unique and non-empty")
        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
            raise ValueError("shared acquisition dispatch time is invalid")
        if len(values) > self._max_inflight_groups - self.inflight_count:
            return ()
        records: list[_SharedAcquisitionRecord] = []
        for identity in values:
            record = self._records.get(identity)
            if (
                record is None
                or record.state is not SharedAcquisitionState.PLANNED
                or record.job.release_ns > now_ns
            ):
                return ()
            records.append(record)
        staging_bytes = sum(record.job.staging_bytes for record in records)
        if staging_bytes > self._staging_capacity_bytes - self._staging_outstanding_bytes:
            return ()

        reservations: list[tuple[_SharedAcquisitionRecord, TenantCreditLease]] = []
        for record in records:
            lease = self._tenant_credits.try_reserve(
                (
                    TenantCreditCharge(
                        record.job.tenant_id, record.job.staging_bytes
                    ),
                )
            )
            if lease is None:
                for _record, reserved in reversed(reservations):
                    self._tenant_credits.release(reserved)
                return ()
            reservations.append((record, lease))

        predicted_start = max(now_ns, self._predicted_link_tail_ns)
        for record, lease in reservations:
            job = record.job
            record.state = SharedAcquisitionState.SUBMITTED
            record.submitted_ns = now_ns
            predicted_start += job.service_ns
            record.predicted_completion_ns = predicted_start
            record.tenant_lease = lease
            self._staging_outstanding_bytes += job.staging_bytes
        self._predicted_link_tail_ns = predicted_start
        return tuple(record.job for record in records)

    def publish_fence(self, identity: AcquisitionGroupIdentity) -> None:
        record = self._record_in_state(identity, SharedAcquisitionState.SUBMITTED)
        record.state = SharedAcquisitionState.FENCE_PUBLISHED

    def mark_ready(self, identity: AcquisitionGroupIdentity) -> None:
        record = self._record_in_state(
            identity, SharedAcquisitionState.FENCE_PUBLISHED
        )
        self._release_reservation(record)
        record.state = (
            SharedAcquisitionState.CANCELLED
            if record.cancel_requested
            else SharedAcquisitionState.CONSUMED
            if record.consumer_ordered
            else SharedAcquisitionState.READY
        )

    def mark_link_idle(self, *, now_ns: int) -> None:
        """Close conservative prediction drift after every submitted group is ready."""

        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
            raise ValueError("shared acquisition idle time is invalid")
        if self.inflight_count:
            raise RuntimeError("shared acquisition link is not idle")
        self._predicted_link_tail_ns = now_ns

    def consume(self, identity: AcquisitionGroupIdentity) -> None:
        record = self._records.get(identity)
        if record is None:
            raise KeyError("unknown shared acquisition group")
        if record.state is SharedAcquisitionState.FENCE_PUBLISHED:
            if record.consumer_ordered:
                raise RuntimeError("shared acquisition consumer was ordered twice")
            record.consumer_ordered = True
            return
        if record.state is SharedAcquisitionState.READY:
            record.state = SharedAcquisitionState.CONSUMED
            return
        raise RuntimeError(
            f"shared acquisition cannot be consumed in state {record.state.value}"
        )

    def cancel_request(self, request_slot: int, generation: int) -> int:
        cancelled = 0
        for identity, record in self._records.items():
            if (
                identity.request_slot == request_slot
                and identity.request_generation == generation
                and record.state is SharedAcquisitionState.PLANNED
            ):
                record.state = SharedAcquisitionState.CANCELLED
                cancelled += 1
        return cancelled

    def cancel(self, identity: AcquisitionGroupIdentity) -> None:
        """Cancel planned work or defer in-flight retirement until readiness."""

        record = self._records.get(identity)
        if record is None:
            raise KeyError("unknown shared acquisition group")
        if record.state is SharedAcquisitionState.PLANNED:
            record.state = SharedAcquisitionState.CANCELLED
            return
        if record.state in {
            SharedAcquisitionState.SUBMITTED,
            SharedAcquisitionState.FENCE_PUBLISHED,
        }:
            record.cancel_requested = True
            return
        if record.state is SharedAcquisitionState.READY:
            record.state = SharedAcquisitionState.CANCELLED
            return
        raise RuntimeError(
            f"shared acquisition cannot be cancelled in state {record.state.value}"
        )

    def fail(self, identity: AcquisitionGroupIdentity) -> None:
        record = self._records.get(identity)
        if record is None:
            raise KeyError("unknown shared acquisition group")
        if record.state not in {
            SharedAcquisitionState.SUBMITTED,
            SharedAcquisitionState.FENCE_PUBLISHED,
        }:
            raise RuntimeError(
                f"shared acquisition cannot fail in state {record.state.value}"
            )
        self._release_reservation(record)
        record.state = SharedAcquisitionState.FAILED

    def forget_terminal(
        self, identities: Iterable[AcquisitionGroupIdentity]
    ) -> None:
        """Drop completed semantic records after their framework lease retires."""

        values = tuple(identities)
        if len(set(values)) != len(values):
            raise ValueError("shared acquisition retirement repeats an identity")
        for identity in values:
            record = self._records.get(identity)
            if record is None:
                raise KeyError("unknown shared acquisition group")
            if record.state not in {
                SharedAcquisitionState.CONSUMED,
                SharedAcquisitionState.CANCELLED,
                SharedAcquisitionState.FAILED,
            }:
                raise RuntimeError(
                    "shared acquisition cannot be forgotten before retirement"
                )
        for identity in values:
            self._records.pop(identity)

    def _record_in_state(
        self,
        identity: AcquisitionGroupIdentity,
        expected: SharedAcquisitionState,
    ) -> _SharedAcquisitionRecord:
        record = self._records.get(identity)
        if record is None:
            raise KeyError("unknown shared acquisition group")
        if record.state is not expected:
            raise RuntimeError(
                f"shared acquisition is {record.state.value}, expected {expected.value}"
            )
        return record

    def _release_reservation(self, record: _SharedAcquisitionRecord) -> None:
        lease = record.tenant_lease
        if lease is None or self._staging_outstanding_bytes < record.job.staging_bytes:
            raise RuntimeError("shared acquisition reservation is incomplete")
        self._tenant_credits.release(lease)
        record.tenant_lease = None
        self._staging_outstanding_bytes -= record.job.staging_bytes


@dataclass(frozen=True, slots=True)
class AcquisitionServiceCurve:
    """Conservative deployment-local compute service between deadlines."""

    samples_ns: tuple[int, ...] = ()
    minimum_samples: int = 4
    maximum_samples: int = 32

    def __post_init__(self) -> None:
        if self.minimum_samples <= 0 or self.maximum_samples < self.minimum_samples:
            raise ValueError("acquisition service-curve sample bounds are invalid")
        if len(self.samples_ns) > self.maximum_samples or any(
            sample <= 0 for sample in self.samples_ns
        ):
            raise ValueError("acquisition service-curve samples are invalid")

    @property
    def calibrated(self) -> bool:
        return len(self.samples_ns) >= self.minimum_samples

    @property
    def conservative_interval_ns(self) -> int:
        return min(self.samples_ns) if self.calibrated else 0

    def with_observation(self, elapsed_ns: int) -> "AcquisitionServiceCurve":
        if elapsed_ns <= 0:
            raise ValueError("acquisition service observation must be positive")
        samples = (*self.samples_ns, elapsed_ns)[-self.maximum_samples :]
        return replace(self, samples_ns=samples)

    def overlap_budget_ns(self, intervals: int) -> int:
        if intervals < 0:
            raise ValueError("acquisition service interval count cannot be negative")
        return self.conservative_interval_ns * intervals


@dataclass(frozen=True, slots=True)
class AcquisitionWork:
    """Identity and payload owned by one acquisition lifecycle.

    This descriptor deliberately has no timing fields.  Exact work can become
    transport-ready before a deployment-calibrated deadline model exists; the
    lifecycle queue must not manufacture service estimates merely to represent
    that state.
    """

    job_id: int
    payload_bytes: int

    def __post_init__(self) -> None:
        if isinstance(self.job_id, bool) or not isinstance(self.job_id, int):
            raise TypeError("acquisition work ID must be an integer")
        if self.job_id < 0:
            raise ValueError("acquisition work ID cannot be negative")
        if self.payload_bytes <= 0:
            raise ValueError("acquisition work payload must be positive")


@dataclass(frozen=True, slots=True)
class AcquisitionJob(AcquisitionWork):
    """One non-preemptive transfer job in a simultaneous-release schedule.

    ``job_id`` is an execution-local ordinal.  Exact request generation,
    segment, and resource-version identity stays in the acquisition topology;
    the scheduler never hashes or reconstructs semantic identity.
    """

    service_ns: int
    deadline_ns: int

    def __post_init__(self) -> None:
        AcquisitionWork.__post_init__(self)
        if self.service_ns <= 0:
            raise ValueError("EDF job service must be positive")
        if self.deadline_ns < 0:
            raise ValueError("EDF job deadline cannot be negative")


@dataclass(frozen=True, slots=True)
class AcquisitionSchedule:
    """Auditable result of one exact simultaneous-release EDF test."""

    ordered_job_ids: tuple[int, ...]
    completion_ns: tuple[int, ...]
    deadlines_ns: tuple[int, ...]
    first_missed_job_id: int | None
    required_initial_slack_ns: int

    def __post_init__(self) -> None:
        count = len(self.ordered_job_ids)
        if (
            len(set(self.ordered_job_ids)) != count
            or len(self.completion_ns) != count
            or len(self.deadlines_ns) != count
            or any(value <= 0 for value in self.completion_ns)
            or any(value < 0 for value in self.deadlines_ns)
            or self.required_initial_slack_ns < 0
        ):
            raise ValueError("EDF schedule is internally inconsistent")
        if self.first_missed_job_id is not None and (
            self.first_missed_job_id not in self.ordered_job_ids
        ):
            raise ValueError("EDF missed job is outside the schedule")
        if self.feasible != (self.required_initial_slack_ns == 0):
            raise ValueError("EDF feasibility and required slack disagree")

    @property
    def feasible(self) -> bool:
        return self.first_missed_job_id is None


def schedule_acquisition_jobs(
    jobs: Iterable[AcquisitionJob],
) -> AcquisitionSchedule:
    """Schedule one serialized link and test every cumulative EDF deadline.

    The caller may provide jobs in any order.  Equal deadlines retain their
    explicit ``job_id`` order so both execution and evidence are deterministic.
    """

    values = tuple(jobs)
    if len({job.job_id for job in values}) != len(values):
        raise ValueError("EDF jobs must have unique IDs")
    ordered = tuple(sorted(values, key=lambda job: (job.deadline_ns, job.job_id)))
    elapsed_ns = 0
    maximum_lateness_ns = 0
    first_missed_job_id: int | None = None
    completion_ns: list[int] = []
    for job in ordered:
        elapsed_ns += job.service_ns
        completion_ns.append(elapsed_ns)
        lateness_ns = elapsed_ns - job.deadline_ns
        if lateness_ns > 0:
            maximum_lateness_ns = max(maximum_lateness_ns, lateness_ns)
            if first_missed_job_id is None:
                first_missed_job_id = job.job_id
    return AcquisitionSchedule(
        ordered_job_ids=tuple(job.job_id for job in ordered),
        completion_ns=tuple(completion_ns),
        deadlines_ns=tuple(job.deadline_ns for job in ordered),
        first_missed_job_id=first_missed_job_id,
        required_initial_slack_ns=maximum_lateness_ns,
    )


class AcquisitionJobState(str, Enum):
    """Control-plane lifecycle of one exact acquisition job.

    ``FENCE_PUBLISHED`` means the backend has submitted the transfer and
    published the readiness primitive consumed by numerical execution.  It does
    not claim that the bytes are already resident; the backend-specific fence
    or native object state remains the source of that fact.
    """

    PLANNED = "planned"
    SUBMITTED = "submitted"
    FENCE_PUBLISHED = "fence_published"
    CONSUMED = "consumed"
    CANCELLED = "cancelled"
    FAILED = "failed"


_TERMINAL_JOB_STATES = frozenset(
    {
        AcquisitionJobState.CONSUMED,
        AcquisitionJobState.CANCELLED,
        AcquisitionJobState.FAILED,
    }
)


class AcquisitionQueue:
    """Bounded work-conserving lifecycle for one ordered transport queue.

    Scheduling policy is intentionally outside this state machine.  A caller
    may pass calibrated EDF order, or a structural consumer order when work is
    ready before timing calibration.  Capacity limits outstanding submissions,
    not total workload size: retiring or cancelling a job immediately exposes
    the next planned job.  CUDA events, NVMe commands, HBM addresses, and
    framework leases deliberately remain outside this dependency-free owner.
    """

    def __init__(
        self,
        jobs: Iterable[AcquisitionWork],
        *,
        ordered_job_ids: Iterable[int],
        max_inflight_jobs: int,
    ) -> None:
        values = tuple(jobs)
        if isinstance(max_inflight_jobs, bool) or not isinstance(
            max_inflight_jobs, int
        ):
            raise TypeError("acquisition in-flight capacity must be an integer")
        if max_inflight_jobs <= 0:
            raise ValueError("acquisition in-flight capacity must be positive")
        if len({job.job_id for job in values}) != len(values):
            raise ValueError("acquisition work must have unique IDs")
        self._jobs = {job.job_id: job for job in values}
        order = tuple(ordered_job_ids)
        if len(order) != len(values) or set(order) != set(self._jobs):
            raise ValueError(
                "acquisition execution order must cover every job exactly once"
            )
        self._ordered_job_ids = order
        self._states = {
            job_id: AcquisitionJobState.PLANNED for job_id in self._ordered_job_ids
        }
        self._state_counts = {
            state: (len(values) if state is AcquisitionJobState.PLANNED else 0)
            for state in AcquisitionJobState
        }
        self._max_inflight_jobs = max_inflight_jobs
        self._next_job = 0

    @classmethod
    def from_edf(
        cls,
        jobs: Iterable[AcquisitionJob],
        *,
        max_inflight_jobs: int,
    ) -> "AcquisitionQueue":
        """Create a lifecycle queue from one explicit calibrated EDF result."""

        values = tuple(jobs)
        schedule = schedule_acquisition_jobs(values)
        return cls(
            values,
            ordered_job_ids=schedule.ordered_job_ids,
            max_inflight_jobs=max_inflight_jobs,
        )

    @property
    def max_inflight_jobs(self) -> int:
        return self._max_inflight_jobs

    @property
    def job_ids(self) -> tuple[int, ...]:
        """Return the immutable execution order owned by this queue."""

        return self._ordered_job_ids

    def state(self, job_id: int) -> AcquisitionJobState:
        try:
            return self._states[job_id]
        except KeyError as error:
            raise KeyError(f"unknown acquisition job {job_id}") from error

    @property
    def inflight_count(self) -> int:
        return self.count_states(
            AcquisitionJobState.SUBMITTED,
            AcquisitionJobState.FENCE_PUBLISHED,
        )

    @property
    def terminal(self) -> bool:
        return self.count_states(*_TERMINAL_JOB_STATES) == len(self._states)

    def count_states(self, *states: AcquisitionJobState) -> int:
        """Return an O(number-of-states) lifecycle count.

        Admission and per-layer retirement query this queue on a serving hot
        path.  Maintaining exact state cardinalities avoids repeatedly scanning
        every acquisition job as model depth or frontier granularity grows.
        """

        if not states or len(set(states)) != len(states):
            raise ValueError("acquisition state query must be unique and non-empty")
        return sum(self._state_counts[state] for state in states)

    def _set_state(self, job_id: int, target: AcquisitionJobState) -> None:
        current = self._states[job_id]
        if current is target:
            return
        self._state_counts[current] -= 1
        self._state_counts[target] += 1
        self._states[job_id] = target

    def cancel_unfinished(self) -> None:
        """Cancel every nonterminal job at an exceptional lifetime boundary."""

        for job_id in self._ordered_job_ids:
            if self._states[job_id] not in _TERMINAL_JOB_STATES:
                self._set_state(job_id, AcquisitionJobState.CANCELLED)

    def claim(self) -> tuple[AcquisitionWork, ...]:
        """Fill every available submission slot in the bound execution order."""

        available = self._max_inflight_jobs - self.inflight_count
        claimed: list[AcquisitionWork] = []
        order = self._ordered_job_ids
        while available > 0 and self._next_job < len(order):
            job_id = order[self._next_job]
            self._next_job += 1
            state = self._states[job_id]
            if state is not AcquisitionJobState.PLANNED:
                continue
            self._set_state(job_id, AcquisitionJobState.SUBMITTED)
            claimed.append(self._jobs[job_id])
            available -= 1
        return tuple(claimed)

    def publish_fence(self, job_id: int) -> None:
        """Publish the backend readiness primitive after successful submission."""

        self._transition(
            job_id,
            AcquisitionJobState.SUBMITTED,
            AcquisitionJobState.FENCE_PUBLISHED,
        )

    def retire(self, job_id: int) -> None:
        """Retire one job after its final numerical consumer is ordered."""

        self._transition(
            job_id,
            AcquisitionJobState.FENCE_PUBLISHED,
            AcquisitionJobState.CONSUMED,
        )

    def cancel(self, job_id: int) -> None:
        self._finish(job_id, AcquisitionJobState.CANCELLED)

    def fail(self, job_id: int) -> None:
        self._finish(job_id, AcquisitionJobState.FAILED)

    def _finish(self, job_id: int, target: AcquisitionJobState) -> None:
        state = self.state(job_id)
        if state in _TERMINAL_JOB_STATES:
            raise ValueError(
                f"terminal acquisition job {job_id} cannot become {target.value}"
            )
        self._set_state(job_id, target)

    def _transition(
        self,
        job_id: int,
        expected: AcquisitionJobState,
        target: AcquisitionJobState,
    ) -> None:
        state = self.state(job_id)
        if state is not expected:
            raise ValueError(
                f"acquisition job {job_id} cannot transition "
                f"{state.value} -> {target.value}"
            )
        self._set_state(job_id, target)


@dataclass(frozen=True, slots=True)
class LayerAcquisitionSubmission:
    """One work-conserving layer submission result."""

    job_count: int
    ranges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.job_count < 0 or any(
            begin < 0 or end <= begin for begin, end in self.ranges
        ):
            raise ValueError("layer acquisition submission geometry is invalid")
        if self.job_count != sum(end - begin for begin, end in self.ranges):
            raise ValueError("layer acquisition ranges do not cover the claimed jobs")


class LayerAcquisition:
    """Own one finite, layer-ordered exact-acquisition lifecycle.

    The owner is independent of CUDA, framework metadata, and tier transport.
    A backend publishes one readiness primitive for every claimed layer; the
    numerical adapter retires that layer only after ordering its final
    consumer. Optional calibrated EDF analysis proves that the structural
    transformer order remains the correct simultaneous-release order.
    """

    def __init__(self, layer_bytes: tuple[int, ...]) -> None:
        if not layer_bytes or any(value <= 0 for value in layer_bytes):
            raise ValueError("layer acquisition requires positive layer bytes")
        self._layer_bytes = tuple(layer_bytes)
        self._model: LayerAcquisitionModel | None = None
        self.queue = AcquisitionQueue(
            tuple(
                AcquisitionWork(layer, payload_bytes)
                for layer, payload_bytes in enumerate(self._layer_bytes)
            ),
            ordered_job_ids=range(len(self._layer_bytes)),
            # The backend link is already serialized. Publish the complete
            # finite queue so it cannot idle between framework callbacks;
            # readiness remains independently observable per layer.
            max_inflight_jobs=len(self._layer_bytes),
        )

    @property
    def model(self) -> "LayerAcquisitionModel | None":
        return self._model

    def bind_model(self, model: "LayerAcquisitionModel") -> bool:
        """Attach calibrated feasibility without changing byte ownership."""

        if model.layer_bytes != self._layer_bytes:
            raise RuntimeError("EDF model changed layer acquisition byte ownership")
        if schedule_acquisition_jobs(model.admission_jobs()).ordered_job_ids != (
            self.queue.job_ids
        ):
            raise RuntimeError("EDF order disagrees with numerical layer order")
        if self._model is None:
            self._model = model
            return True
        if self._model != model:
            raise RuntimeError("layer acquisition changed its calibrated EDF model")
        return False

    @property
    def started(self) -> bool:
        return self.queue.count_states(AcquisitionJobState.PLANNED) != len(
            self.queue.job_ids
        )

    @property
    def fully_published(self) -> bool:
        return self.queue.count_states(
            AcquisitionJobState.FENCE_PUBLISHED,
            AcquisitionJobState.CONSUMED,
        ) == len(self.queue.job_ids)

    def submit_available(
        self,
        *,
        publish_range: Callable[[int, int], None],
        published_layers: Mapping[int, object],
    ) -> LayerAcquisitionSubmission:
        """Fill available link slots and publish each claimed readiness fence."""

        claimed = self.queue.claim()
        if not claimed:
            return LayerAcquisitionSubmission(0, ())
        claimed_ids = tuple(job.job_id for job in claimed)
        ranges = _contiguous_ranges(claimed_ids)
        try:
            for begin, end in ranges:
                publish_range(begin, end)
                for layer in range(begin, end):
                    if layer not in published_layers:
                        raise RuntimeError(
                            "transport returned without publishing layer "
                            f"{layer}'s readiness fence"
                        )
                    self.queue.publish_fence(layer)
        except BaseException:
            for job_id in claimed_ids:
                if self.queue.state(job_id) is AcquisitionJobState.SUBMITTED:
                    self.queue.fail(job_id)
            raise
        return LayerAcquisitionSubmission(len(claimed), ranges)

    def retire(self, layer: int) -> None:
        """Retire one layer after its numerical consumer has been ordered."""

        if layer not in self.queue.job_ids:
            raise RuntimeError(f"layer acquisition does not own layer {layer}")
        state = self.queue.state(layer)
        if state is not AcquisitionJobState.FENCE_PUBLISHED:
            raise RuntimeError(
                f"layer acquisition {layer} reached its consumer in state {state.value}"
            )
        self.queue.retire(layer)

    def retire_published(self) -> None:
        """Retire a fully published graph batch at its stream handoff."""

        for job_id in self.queue.job_ids:
            state = self.queue.state(job_id)
            if state is AcquisitionJobState.FENCE_PUBLISHED:
                self.queue.retire(job_id)
            elif state is not AcquisitionJobState.CONSUMED:
                raise RuntimeError(
                    "graph handoff contains an unpublished acquisition job"
                )

    def cancel_unfinished(self) -> None:
        self.queue.cancel_unfinished()


def _contiguous_ranges(job_ids: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Coalesce adjacent acquisition jobs without changing scheduler order."""

    if not job_ids:
        return ()
    ranges: list[tuple[int, int]] = []
    begin = previous = job_ids[0]
    for job_id in job_ids[1:]:
        if job_id == previous + 1:
            previous = job_id
            continue
        ranges.append((begin, previous + 1))
        begin = previous = job_id
    ranges.append((begin, previous + 1))
    return tuple(ranges)


@dataclass(frozen=True, slots=True)
class LayerAcquisitionModel:
    """Transformer projection of exact acquisition jobs onto one tier link.

    Every unresolved layer is available when the external-resource lease is
    captured.  ``initial_compute_ns`` is useful work before layer-zero
    attention; ``inter_layer_compute_ns`` is the conservative useful interval
    between subsequent attention arrivals.  Transfer service comes from the
    selected backend's measured model, never from a hard-coded byte threshold.
    """

    layer_bytes: tuple[int, ...]
    transfer_service_ns: tuple[int, ...]
    initial_compute_ns: int
    inter_layer_compute_ns: int

    def __post_init__(self) -> None:
        if (
            not self.layer_bytes
            or len(self.layer_bytes) != len(self.transfer_service_ns)
            or any(value <= 0 for value in self.layer_bytes)
            or any(value <= 0 for value in self.transfer_service_ns)
            or self.initial_compute_ns < 0
            or self.inter_layer_compute_ns <= 0
        ):
            raise ValueError("layer acquisition model has invalid service geometry")

    def analyze_admission(
        self, *, ready_prefix_layers: int
    ) -> "LayerAcquisitionFeasibility":
        """Test unresolved jobs from the forward's admission time origin."""

        return self._analyze(
            ready_prefix_layers=ready_prefix_layers,
            deadline_ns=lambda layer: (
                self.initial_compute_ns + layer * self.inter_layer_compute_ns
            ),
        )

    def admission_jobs(self) -> tuple[AcquisitionJob, ...]:
        """Project every layer onto the forward's admission-time EDF axis."""

        return tuple(
            AcquisitionJob(
                job_id=layer,
                payload_bytes=payload_bytes,
                service_ns=self.transfer_service_ns[layer],
                deadline_ns=(
                    self.initial_compute_ns + layer * self.inter_layer_compute_ns
                ),
            )
            for layer, payload_bytes in enumerate(self.layer_bytes)
        )

    def analyze_after_attention(
        self,
        *,
        completed_layer: int,
        ready_prefix_layers: int | None = None,
    ) -> "LayerAcquisitionFeasibility":
        """Test suffix feasibility from one completed attention arrival."""

        layer_count = len(self.layer_bytes)
        if not 0 <= completed_layer < layer_count:
            raise ValueError("completed attention layer is outside the model")
        ready = (
            completed_layer + 1 if ready_prefix_layers is None else ready_prefix_layers
        )
        if not completed_layer + 1 <= ready <= layer_count:
            raise ValueError("ready prefix precedes the completed attention layer")
        return self._analyze(
            ready_prefix_layers=ready,
            deadline_ns=lambda layer: (
                (layer - completed_layer) * self.inter_layer_compute_ns
            ),
        )

    def minimum_admission_ready_prefix(self) -> int:
        """Return the smallest completed prefix that makes the suffix feasible.

        This is a proof result, not an admission heuristic.  A backend may submit
        more work, but a framework need not release the numerical forward before
        this prefix is actually ready.
        """

        for ready_prefix in range(len(self.layer_bytes) + 1):
            if self.analyze_admission(ready_prefix_layers=ready_prefix).feasible:
                return ready_prefix
        raise RuntimeError("a fully ready acquisition model must be feasible")

    def compile_after_attention_frontier(self) -> "LayerAcquisitionFrontier":
        """Compile every suffix's first EDF miss once for a forward.

        Transformer-layer deadlines are strictly increasing and every job is
        simultaneously released, so EDF order is the layer order.  Prefix sums
        reproduce :meth:`analyze_after_attention` exactly without rebuilding
        and sorting ``AcquisitionJob`` objects at every layer arrival.
        """

        layer_count = len(self.layer_bytes)
        feasible_ends: list[int] = []
        for completed_layer in range(layer_count):
            elapsed_ns = 0
            feasible_end = layer_count
            for layer in range(completed_layer + 1, layer_count):
                elapsed_ns += self.transfer_service_ns[layer]
                deadline_ns = (layer - completed_layer) * self.inter_layer_compute_ns
                if elapsed_ns > deadline_ns:
                    feasible_end = layer
                    break
            feasible_ends.append(feasible_end)
        return LayerAcquisitionFrontier(tuple(feasible_ends))

    def _analyze(
        self,
        *,
        ready_prefix_layers: int,
        deadline_ns: Callable[[int], int],
    ) -> "LayerAcquisitionFeasibility":
        layer_count = len(self.layer_bytes)
        if not 0 <= ready_prefix_layers <= layer_count:
            raise ValueError("ready layer prefix is outside the acquisition model")
        admission_jobs = self.admission_jobs()
        schedule = schedule_acquisition_jobs(
            replace(admission_jobs[layer], deadline_ns=deadline_ns(layer))
            for layer in range(ready_prefix_layers, layer_count)
        )
        return LayerAcquisitionFeasibility(
            ready_prefix_layers=ready_prefix_layers,
            layer_count=layer_count,
            schedule=schedule,
        )


@dataclass(frozen=True, slots=True)
class LayerAcquisitionFeasibility:
    """Layer-indexed view of a tier-neutral EDF schedule."""

    ready_prefix_layers: int
    layer_count: int
    schedule: AcquisitionSchedule

    def __post_init__(self) -> None:
        if (
            self.layer_count <= 0
            or not 0 <= self.ready_prefix_layers <= self.layer_count
        ):
            raise ValueError("layer acquisition feasibility geometry is invalid")
        expected = set(range(self.ready_prefix_layers, self.layer_count))
        if set(self.schedule.ordered_job_ids) != expected:
            raise ValueError("EDF schedule does not cover the unresolved layer suffix")

    @property
    def feasible(self) -> bool:
        return self.schedule.feasible

    @property
    def first_missed_layer(self) -> int | None:
        return self.schedule.first_missed_job_id

    @property
    def required_initial_slack_ns(self) -> int:
        return self.schedule.required_initial_slack_ns

    @property
    def cumulative_completion_ns(self) -> tuple[int, ...]:
        return self.schedule.completion_ns

    @property
    def deadlines_ns(self) -> tuple[int, ...]:
        return self.schedule.deadlines_ns


@dataclass(frozen=True, slots=True)
class LayerAcquisitionFrontier:
    """O(1) lookup table for a frozen layer-acquisition service model.

    Entry ``i`` is the exclusive ready prefix that can be published after
    attention layer ``i``.  A value equal to the model layer count means that
    the complete suffix is feasible; otherwise the value is the first missed
    layer and must remain demand-driven.
    """

    feasible_end_by_completed_layer: tuple[int, ...]

    def __post_init__(self) -> None:
        layer_count = len(self.feasible_end_by_completed_layer)
        if layer_count == 0 or any(
            not completed_layer + 1 <= feasible_end <= layer_count
            for completed_layer, feasible_end in enumerate(
                self.feasible_end_by_completed_layer
            )
        ):
            raise ValueError("layer acquisition frontier has invalid geometry")

    @property
    def layer_count(self) -> int:
        return len(self.feasible_end_by_completed_layer)

    def feasible_end_after_attention(self, completed_layer: int) -> int:
        if not 0 <= completed_layer < self.layer_count:
            raise ValueError("completed layer is outside the acquisition frontier")
        return self.feasible_end_by_completed_layer[completed_layer]
