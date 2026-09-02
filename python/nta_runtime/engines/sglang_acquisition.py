"""Lease-scoped Host acquisition coordination for the SGLang integration.

This component is the sole owner of the control path from an exact HiCache
lease to immutable transfer descriptors, a calibrated deadline model, finite
submission, and layer retirement.  The transport receives an already-frozen
plan and therefore cannot allocate or invoke scheduling policy from its
steady-state submission path.
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
import bisect
import time
from typing import TYPE_CHECKING, Any

from nta_runtime.acquisition_scheduler import (
    AcquisitionGroupIdentity,
    AcquisitionServiceCurve,
    LayerAcquisition,
    LayerAcquisitionModel,
    SharedAcquisitionJob,
    SharedAcquisitionQueue,
    SharedAcquisitionState,
    TenantCreditLedger,
)
from nta_runtime.adapters.sglang import SglangExecutionConfig
from nta_runtime.engines.sglang_calibration import (
    LayerServiceKey,
    SglangConsumerPolicyCalibration,
    SglangLayerServiceCalibration,
)
from nta_runtime.engines.sglang_hicache import PendingHostLoad
from nta_runtime.engines.sglang_contracts import (
    LeaseAcquisitionGroup,
    LeaseOperationRequest,
)
from nta_runtime.engines.sglang_pipeline import SglangHostTransport
from nta_runtime.engines.sglang_planning import (
    calibration_probe_end,
    pipeline_object_id,
)
from nta_runtime.engines.sglang_transfer import (
    HostMoverController,
    HostTransferLeasePlan,
    build_host_transfer_lease_plan,
)
from nta_runtime.execution_planner import HostExecutionMode
from nta_runtime.execution_protocol import ProtocolKind

if TYPE_CHECKING:
    from nta_runtime.engines.sglang_state import SglangForwardEpoch


@dataclass(slots=True)
class _SharedHostCohort:
    pending: PendingHostLoad
    local_layer: int
    identities: tuple[AcquisitionGroupIdentity, ...]
    absolute_row_ends: tuple[int, ...]
    consumer_ordered: bool = False
    cancel_requested: bool = False


@dataclass(frozen=True, slots=True)
class SharedHostAdmissionFeasibility:
    """SGLang view of one global shared-link policy simulation."""

    ready_prefix_layers: int
    feasible: bool
    first_missed_layer: int | None
    required_initial_slack_ns: int


class SglangHostAcquisitionCoordinator:
    """Own one exact Host lease from capture through numerical retirement."""

    def __init__(
        self,
        *,
        device_pool: Any,
        execution_config: SglangExecutionConfig,
        tenant_isolation_enabled: bool,
        model_layer_count: int,
        sm_acquisition_waves: int,
        frontier_enabled: bool,
        frontier_layers_per_wave: int,
        movers: HostMoverController,
        calibration: SglangLayerServiceCalibration,
        consumer_calibration: SglangConsumerPolicyCalibration,
        minimum_consumer_gain: float,
        transport: SglangHostTransport,
        request_adapter: Any,
        staging_capacity_bytes: int,
        tenant_specs: tuple[tuple[int, int], ...],
        max_inflight_groups: int,
        stats: MutableMapping[str, Any],
    ) -> None:
        if (
            min(
                model_layer_count,
                sm_acquisition_waves,
                frontier_layers_per_wave,
            )
            <= 0
        ):
            raise ValueError("SGLang Host acquisition geometry must be positive")
        if minimum_consumer_gain < 1.0:
            raise ValueError("SGLang consumer gain must be at least one")
        self._device_pool = device_pool
        self._execution_config = execution_config
        self._tenant_isolation_enabled = bool(tenant_isolation_enabled)
        self._model_layer_count = model_layer_count
        self._sm_acquisition_waves = sm_acquisition_waves
        self._frontier_enabled = bool(frontier_enabled)
        self._frontier_layers_per_wave = frontier_layers_per_wave
        self._movers = movers
        self._calibration = calibration
        self._consumer_calibration = consumer_calibration
        self._minimum_consumer_gain = minimum_consumer_gain
        self._transport = transport
        self._request_adapter = request_adapter
        self._shared_queue = SharedAcquisitionQueue(
            staging_capacity_bytes=staging_capacity_bytes,
            tenant_credits=TenantCreditLedger(tenant_specs),
            max_inflight_groups=max_inflight_groups,
        )
        # This is an irrevocable CUDA submission horizon, not a byte/shape
        # heuristic. It is kept finite so a newly admitted earlier deadline is
        # blocked by at most a small number of already-issued layer cohorts.
        self._shared_layers_per_dispatch = frontier_layers_per_wave
        self._shared_dispatch_horizon = 2
        self._shared_cohorts: dict[
            tuple[int, int], _SharedHostCohort
        ] = {}
        self._shared_identity_owner: dict[
            AcquisitionGroupIdentity, tuple[int, int]
        ] = {}
        self._shared_active: list[tuple[tuple[int, int], ...]] = []
        self._stats = stats

    @property
    def proactive_layer_queue_enabled(self) -> bool:
        """Return whether a scheduler-bound batch may submit Host transfers.

        SCHEDULED_BULK and DEPENDENCY_AWARE admit the same finite,
        scheduler-bound queue; their consumer policy differs, not their
        ownership or transport path. AUTO may select either consumer after
        calibration. DIRECT is the explicit eager diagnostic arm and may
        submit at capture. DEVICE_BULK retains device-discovered acquisition as
        a diagnostic. Scheduled Host transport binds request generations and
        reserves tenant credits before publication, so isolation no longer
        disables overlap.
        """

        return (
            self._execution_config.protocol.kind is ProtocolKind.LATE_BOUND
            and self._execution_config.host_execution_mode
            in {
                HostExecutionMode.AUTO,
                HostExecutionMode.DIRECT,
                HostExecutionMode.SCHEDULED_BULK,
                HostExecutionMode.DEPENDENCY_AWARE,
            }
        )

    @property
    def eager_capture_enabled(self) -> bool:
        """Return whether transport selection is intentionally batch-agnostic."""

        return (
            self.proactive_layer_queue_enabled
            and not self._tenant_isolation_enabled
            and self._execution_config.host_execution_mode is HostExecutionMode.DIRECT
        )

    def capture(self, pending: PendingHostLoad) -> None:
        """Capture physical ownership and start any unconditional acquisition."""

        if pending.controller.mem_pool_device is not self._device_pool:
            raise RuntimeError("HiCache lease belongs to a different device pool")
        layer_count = int(pending.controller.layer_num)
        if layer_count != self._model_layer_count:
            raise RuntimeError("HiCache load and model layer counts disagree")
        groups = tuple(
            LeaseAcquisitionGroup(request.operation_id, 0, request.row_count)
            for request in pending.operation_demands
        )
        if not groups:
            raise RuntimeError("HiCache demand lease has no exact request segments")
        if pending.scheduled_acquisition_groups not in {(), groups}:
            raise RuntimeError("HiCache acquisition groups changed after capture")
        pending.scheduled_acquisition_groups = groups
        self.account_selection(pending)
        conventional = self._execution_config.protocol.kind is ProtocolKind.CONVENTIONAL
        initial_layers = 0
        if conventional:
            self.publish_range(pending, 0, layer_count)
            initial_layers = layer_count
        elif self.eager_capture_enabled:
            # DIRECT is the explicit batch-agnostic baseline. AUTO must wait for
            # the scheduler batch: freezing its mover here would permanently
            # discard the already-calibrated layer-service shape and make the
            # resource-aware copy-engine/SM decision impossible.
            self.transfer_plan(pending)
            pending.acquisition = LayerAcquisition(pending.layer_bytes)
            self._add("host_acquisition_jobs_prepared", layer_count)
            self._add("lease_acquisition_groups_prepared")
            initial_layers = self.submit(pending)
            if initial_layers != layer_count:
                raise RuntimeError(
                    "dense HiCache lease did not fill its finite acquisition queue"
                )
            self._add("lease_acquisition_groups_started")
        self._add("initial_acquisition_batches")
        self._add("initial_acquisition_layers", initial_layers)
        self._add("initial_typed_gap_layers", layer_count - initial_layers)
        if initial_layers == 0:
            self._add("schedule_bound_acquisition_batches")

    def deadline_model(
        self, pending: PendingHostLoad, batch: Any
    ) -> LayerAcquisitionModel | None:
        """Return a calibrated EDF model without synchronizing CUDA."""

        acquisition = pending.acquisition
        shared_model = getattr(pending, "shared_deadline_model", None)
        if shared_model is not None:
            return shared_model
        if acquisition is not None and acquisition.model is not None:
            return acquisition.model
        self._movers.collect_profiles()
        self._calibration.collect()
        if (
            pending.mover_plan is None
            or not pending.layer_bytes
            or not pending.row_bytes_by_layer
        ):
            return None
        curve = self._calibration.curve_for_batch(batch)
        if curve is None:
            return None
        model = self.deadline_model_for_curve(pending, curve)
        if model is not None and acquisition is not None:
            acquisition.bind_model(model)
        return model

    def deadline_model_for_curve(
        self,
        pending: PendingHostLoad,
        curve: AcquisitionServiceCurve,
    ) -> LayerAcquisitionModel | None:
        mover = pending.mover_plan
        if (
            mover is None
            or not curve.calibrated
            or not pending.layer_bytes
            or not pending.row_bytes_by_layer
        ):
            return None
        transfer_count = int(pending.device_indices.numel())
        if transfer_count <= 0 or transfer_count != mover.row_count:
            raise RuntimeError("HiCache deadline mover geometry changed")
        if not self._movers.lease_calibrated(pending):
            return None
        representative_bytes = self._movers.representative_wave_bytes(
            pending.row_bytes_by_layer, transfer_count
        )
        service_model = self._movers.service_model(representative_bytes)
        layer_service: list[int] = []
        for key_row_bytes, value_row_bytes in pending.row_bytes_by_layer:
            service_ns = service_model.candidate_ns(
                total_rows=transfer_count,
                copy_rows=mover.copy_row_count,
                copy_run_count=len(mover.copy_runs),
                row_bytes=key_row_bytes + value_row_bytes,
                copy_operations_per_run=2,
            )
            if service_ns is None:
                return None
            layer_service.append(service_ns)
        return LayerAcquisitionModel(
            layer_bytes=pending.layer_bytes,
            transfer_service_ns=tuple(layer_service),
            # SGLang exposes no calibrated useful-compute interval between
            # admission and first attention.  Zero is a conservative deadline
            # origin, not a layer-zero transport special case.
            initial_compute_ns=0,
            inter_layer_compute_ns=curve.conservative_interval_ns,
        )

    def prepare_owner(
        self,
        pending: PendingHostLoad,
        batch: Any,
        *,
        active_batch: SglangForwardEpoch | None = None,
    ) -> bool:
        """Bind one calibrated EDF proof to immutable physical ownership."""

        acquisition = pending.acquisition
        if getattr(pending, "shared_deadline_model", None) is not None:
            return True
        if acquisition is not None and acquisition.model is not None:
            return True
        if acquisition is None and pending.prefetched_layers:
            return False
        self._bind_group_identities(pending, batch)
        shape_key = self._calibration.shape_key(batch)
        curve = self._calibration.curve_for_batch(batch)
        if shape_key is None or curve is None:
            self._add("host_acquisition_shape_uncalibrated")
        if active_batch is not None and active_batch.pending_host_load is pending:
            active_batch.layer_service_key = shape_key
        self._movers.collect_profiles()
        transfer_plan = self.transfer_plan(
            pending,
            layer_service_key=shape_key,
            layer_curve=curve,
        )
        if (
            shape_key is not None
            and getattr(pending, "arrival_profile_key", None) is None
        ):
            self._consumer_calibration.bind_lease(
                pending,
                layer_service_key=shape_key,
                producer_kind=transfer_plan.mover.kind,
                layers_per_submission=self._frontier_layers_per_wave,
                sm_waves_per_layer=transfer_plan.sm_waves_per_layer,
                minimum_gain=self._minimum_consumer_gain,
            )
        if acquisition is None:
            acquisition = LayerAcquisition(pending.layer_bytes)
            pending.acquisition = acquisition
            self._add("host_acquisition_jobs_prepared", len(pending.layer_bytes))
            self._add("host_acquisition_structural_owners")
        if curve is None:
            return False
        model = self.deadline_model_for_curve(pending, curve)
        if model is None:
            self._add(
                "host_acquisition_mover_uncalibrated"
                if not self._movers.lease_calibrated(pending)
                else "host_acquisition_model_rejected"
            )
            return False
        if acquisition.bind_model(model):
            self._add("host_acquisition_models_bound")
        self._register_shared_acquisition(pending, model)
        # The dynamic shared queue is now the sole lifecycle owner. Retaining a
        # second layer queue would leave never-submitted PLANNED jobs alive at
        # final HiCache retirement and make cancellation/accounting ambiguous.
        pending.shared_deadline_model = model
        pending.acquisition = None
        return True

    def _bind_group_identities(self, pending: PendingHostLoad, batch: Any) -> None:
        """Assign request generations before scheduler-owned transport starts."""

        demands = tuple(pending.operation_demands)
        if not demands:
            raise RuntimeError(
                "scheduled HiCache acquisition has no request-operation ownership"
            )
        operation_ids = tuple(demand.operation_id for demand in demands)
        if len(set(operation_ids)) != len(operation_ids):
            raise RuntimeError("HiCache acquisition repeats request-operation identity")
        if pending.operation_bindings:
            if set(pending.operation_bindings) != set(operation_ids):
                raise RuntimeError("HiCache acquisition request bindings changed")
            return
        batch_requests = tuple(getattr(batch, "reqs", ()) or ())
        slots_by_request: dict[str, int] = {}
        for request in batch_requests:
            request_id = str(getattr(request, "rid", "") or "")
            request_slot = getattr(request, "req_pool_idx", None)
            if not request_id or request_slot is None:
                raise RuntimeError(
                    "scheduled HiCache batch omitted allocated request identity"
                )
            if request_id in slots_by_request:
                raise RuntimeError("scheduled HiCache batch repeated request identity")
            slots_by_request[request_id] = int(request_slot)
        missing = tuple(
            demand.request_id
            for demand in demands
            if demand.request_id not in slots_by_request
        )
        if missing:
            raise RuntimeError(
                "scheduled HiCache demand is absent from the allocated batch: "
                f"{missing}"
            )
        requests = tuple(
            LeaseOperationRequest.bind(demand, slots_by_request[demand.request_id])
            for demand in demands
        )
        pending.operation_requests = requests
        bindings = self._request_adapter.bind(
            tuple(request.request_id for request in requests),
            tuple(request.request_slot for request in requests),
            tenant_ids=tuple(request.tenant_id for request in requests),
        )
        pending.operation_bindings = {
            request.operation_id: binding
            for request, binding in zip(requests, bindings, strict=True)
        }
        groups = pending.scheduled_acquisition_groups
        if not groups:
            raise RuntimeError(
                "HiCache acquisition identities require frozen semantic groups"
            )
        request_by_operation = {
            request.operation_id: request for request in requests
        }
        start_layer = int(getattr(self._device_pool, "start_layer", 0))
        identities: dict[int, tuple[AcquisitionGroupIdentity, ...]] = {}
        for local_layer in range(self._model_layer_count):
            identities[local_layer] = tuple(
                AcquisitionGroupIdentity(
                    request_slot=binding.request_slot,
                    request_generation=binding.generation,
                    layer_id=start_layer + local_layer,
                    segment_begin=request.logical_begin + group.row_begin,
                    segment_count=group.row_count,
                    resource_version=pending.lease_id,
                )
                for group in groups
                for request in (request_by_operation[group.operation_id],)
                for binding in (pending.operation_bindings[group.operation_id],)
            )
        pending.acquisition_group_identities = identities
        self._add("host_acquisition_request_bindings", len(bindings))
        self._add(
            "host_acquisition_semantic_groups",
            len(groups) * self._model_layer_count,
        )

    def _register_shared_acquisition(
        self, pending: PendingHostLoad, model: LayerAcquisitionModel
    ) -> None:
        """Publish one dynamic shared-link job per exact request/layer group."""

        if getattr(pending, "shared_acquisition_registered", False):
            return
        if set(pending.acquisition_group_identities) != set(
            range(self._model_layer_count)
        ):
            raise RuntimeError("Host acquisition has incomplete semantic groups")
        groups = pending.scheduled_acquisition_groups
        total_rows = sum(group.row_count for group in groups)
        if total_rows != int(pending.device_indices.numel()):
            raise RuntimeError(
                "scheduled Host groups do not cover the physical acquisition lease"
            )
        release_ns = time.monotonic_ns()
        jobs: list[SharedAcquisitionJob] = []
        cohorts: list[_SharedHostCohort] = []
        operation_by_id = {
            operation.operation_id: operation
            for operation in pending.operation_ranges()
        }
        absolute_row_ends = tuple(
            operation_by_id[group.operation_id].row_begin + group.row_end
            for group in groups
        )
        if tuple(sorted(absolute_row_ends)) != absolute_row_ends or (
            not absolute_row_ends
            or absolute_row_ends[-1] != int(pending.device_indices.numel())
        ):
            raise RuntimeError("scheduled Host groups do not partition lease order")
        for local_layer in range(self._model_layer_count):
            identities = pending.acquisition_group_identities[local_layer]
            if len(identities) != len(groups):
                raise RuntimeError("Host acquisition group cardinality changed")
            row_bytes = sum(pending.row_bytes_by_layer[local_layer])
            layer_service_ns = model.transfer_service_ns[local_layer]
            cumulative_rows = 0
            cumulative_service_ns = 0
            for group, identity in zip(groups, identities, strict=True):
                cumulative_rows += group.row_count
                target_service_ns = (
                    layer_service_ns * cumulative_rows // total_rows
                )
                service_ns = max(1, target_service_ns - cumulative_service_ns)
                cumulative_service_ns += service_ns
                binding = pending.operation_bindings[group.operation_id]
                payload_bytes = group.row_count * row_bytes
                jobs.append(
                    SharedAcquisitionJob(
                        identity=identity,
                        tenant_id=binding.tenant_id,
                        payload_bytes=payload_bytes,
                        staging_bytes=payload_bytes,
                        release_ns=release_ns,
                        service_ns=service_ns,
                        deadline_ns=(
                            release_ns
                            + model.initial_compute_ns
                            + local_layer * model.inter_layer_compute_ns
                        ),
                        priority=binding.priority,
                    )
                )
            cohort = _SharedHostCohort(
                pending, local_layer, identities, absolute_row_ends
            )
            key = (pending.lease_id, local_layer)
            if key in self._shared_cohorts or any(
                identity in self._shared_identity_owner for identity in identities
            ):
                raise RuntimeError("Host acquisition repeated a shared-link identity")
            cohorts.append(cohort)
        self._shared_queue.add(jobs)
        for cohort in cohorts:
            key = (pending.lease_id, cohort.local_layer)
            self._shared_cohorts[key] = cohort
            for identity in cohort.identities:
                self._shared_identity_owner[identity] = key
        pending.shared_acquisition_registered = True
        self._add("shared_acquisition_registered_groups", len(jobs))
        self._add("shared_acquisition_registered_cohorts", len(cohorts))

    def admission_feasibility(
        self,
        pending: PendingHostLoad,
        batch: Any,
        progress: Any,
    ) -> SharedHostAdmissionFeasibility | None:
        """Analyze the actual cross-batch queue, including fixed submissions."""

        model = self.deadline_model(pending, batch)
        if model is None:
            return None
        if not getattr(pending, "shared_acquisition_registered", False):
            local = model.analyze_admission(
                ready_prefix_layers=progress.leading_layers
            )
            return SharedHostAdmissionFeasibility(
                local.ready_prefix_layers,
                local.feasible,
                local.first_missed_layer,
                local.required_initial_slack_ns,
            )
        self.progress_shared_acquisition()
        schedule = self._shared_queue.analyze(now_ns=time.monotonic_ns())
        missed = schedule.first_missed_identity
        start_layer = int(getattr(self._device_pool, "start_layer", 0))
        first_missed_layer = None if missed is None else missed.layer_id - start_layer
        if first_missed_layer is not None and not (
            0 <= first_missed_layer < self._model_layer_count
        ):
            raise RuntimeError("shared EDF miss lies outside the model partition")
        return SharedHostAdmissionFeasibility(
            ready_prefix_layers=progress.leading_layers,
            feasible=schedule.feasible,
            first_missed_layer=first_missed_layer,
            required_initial_slack_ns=schedule.maximum_lateness_ns,
        )

    def prepare_admission(self, pending: PendingHostLoad, batch: Any) -> bool:
        """Prepare descriptors only when admission has a usable model."""

        if not self.proactive_layer_queue_enabled:
            return False
        already_prepared = (
            pending.acquisition is not None
            or pending.shared_acquisition_registered
        )
        ready = self.prepare_owner(pending, batch)
        if ready and not already_prepared:
            self._add("admission_acquisition_groups_prepared")
        return ready

    def start_admission(self, pending: PendingHostLoad, batch: Any) -> None:
        """Start the finite queue after admission has bounded its delay."""

        if pending.shared_acquisition_registered:
            if (
                pending.transfer_plan is None
                or pending.prefetched_layers
                or pending.shared_deadline_model is None
            ):
                raise RuntimeError(
                    "HiCache shared acquisition was not prepared exactly once"
                )
            # The finite horizon is global. A legal pump may submit only an
            # older lease, so local publication growth is not a valid start
            # invariant. Follow EDF until this admitted lease owns its first
            # layer fence; never bypass older work with a direct publication.
            self.ensure_layer_published(pending, 0)
            self._add("admission_acquisition_groups_started")
            return
        acquisition = pending.acquisition
        if (
            pending.transfer_plan is None
            or pending.prefetched_layers
            or acquisition is None
            or acquisition.started
        ):
            raise RuntimeError(
                "HiCache admission acquisition was not prepared exactly once"
            )
        if self.deadline_model(pending, batch) is None or acquisition.model is None:
            raise RuntimeError("HiCache admission acquisition lost its calibration")
        submitted = self.submit(pending)
        if submitted != int(pending.controller.layer_num):
            raise RuntimeError("HiCache admission did not fill its finite link queue")
        self._add("admission_acquisition_groups_started")

    def submit(self, pending: PendingHostLoad) -> int:
        """Fill one lease's finite queue and publish exact layer fences."""

        if getattr(pending, "shared_acquisition_registered", False):
            before = len(pending.prefetched_layers)
            self.progress_shared_acquisition()
            self._pump_shared_acquisition()
            return len(pending.prefetched_layers) - before
        acquisition = pending.acquisition
        if acquisition is None:
            raise RuntimeError("HiCache lease has no prepared acquisition owner")
        plan = self.transfer_plan(pending)
        submission = acquisition.submit_available(
            publish_range=lambda begin, end: self._transport.prepare(
                pending,
                plan,
                first_local_layer=begin,
                last_local_layer=end,
            ),
            published_layers=pending.prefetched_layers,
        )
        if submission.job_count:
            self._add("host_acquisition_submission_calls", len(submission.ranges))
            self._add("host_acquisition_jobs_submitted", submission.job_count)
        return submission.job_count

    def progress_shared_acquisition(self) -> None:
        """Retire completed cohorts without synchronizing a CUDA stream."""

        if not self._shared_active:
            return
        now_ns = time.monotonic_ns()
        still_active: list[tuple[tuple[int, int], ...]] = []
        for packet in self._shared_active:
            incomplete: list[tuple[int, int]] = []
            for key in packet:
                cohort = self._shared_cohorts.get(key)
                if cohort is None:
                    raise RuntimeError("shared Host cohort lost semantic ownership")
                publication = cohort.pending.prefetched_layers.get(cohort.local_layer)
                if publication is None:
                    raise RuntimeError(
                        "shared Host cohort has no readiness publication"
                    )
                newly_ready = 0
                cohort_incomplete = False
                for identity, absolute_row_end in zip(
                    cohort.identities, cohort.absolute_row_ends, strict=True
                ):
                    state = self._shared_queue.state(identity)
                    if state is SharedAcquisitionState.FENCE_PUBLISHED:
                        ready_event = publication.ready_event
                        if publication.wave_events:
                            wave = bisect.bisect_left(
                                publication.wave_row_ends, absolute_row_end
                            )
                            if wave >= len(publication.wave_events):
                                raise RuntimeError(
                                    "shared Host group exceeds its completion waves"
                                )
                            ready_event = publication.wave_events[wave]
                        if not ready_event.query():
                            cohort_incomplete = True
                            continue
                        self._shared_queue.mark_ready(identity)
                        newly_ready += 1
                    elif state not in {
                        SharedAcquisitionState.READY,
                        SharedAcquisitionState.CONSUMED,
                        SharedAcquisitionState.CANCELLED,
                    }:
                        raise RuntimeError(
                            "shared Host cohort completed from an invalid lifecycle state"
                        )
                if newly_ready:
                    self._add("shared_acquisition_ready_groups", newly_ready)
                if cohort_incomplete:
                    incomplete.append(key)
                    continue
                self._add("shared_acquisition_ready_cohorts")
                if cohort.consumer_ordered or cohort.cancel_requested:
                    self._forget_shared_cohort(key, cohort)
            if incomplete:
                still_active.append(tuple(incomplete))
        self._shared_active = still_active
        if self._shared_queue.inflight_count == 0:
            self._shared_queue.mark_link_idle(now_ns=now_ns)

    def _pump_shared_acquisition(self) -> int:
        """Fill the finite global link horizon in dynamic EDF order."""

        submitted = 0
        while len(self._shared_active) < self._shared_dispatch_horizon:
            now_ns = time.monotonic_ns()
            key: tuple[int, int] | None = None
            cohort: _SharedHostCohort | None = None
            claimed: list[SharedAcquisitionJob] = []
            visited: set[tuple[int, int]] = set()
            # Choose EDF among resource-admissible packets. An exhausted
            # tenant or a temporarily full staging reservation must not
            # head-of-line block an unrelated tenant whose packet fits.
            for identity in self._shared_queue.released_identities(now_ns=now_ns):
                candidate_key = self._shared_identity_owner.get(identity)
                if candidate_key is None or candidate_key in visited:
                    continue
                visited.add(candidate_key)
                candidate = self._shared_cohorts.get(candidate_key)
                if candidate is None:
                    raise RuntimeError("shared EDF selected an unowned Host group")
                candidate_claim = self._shared_queue.claim_cohort(
                    candidate.identities, now_ns=now_ns
                )
                if candidate_claim:
                    key = candidate_key
                    cohort = candidate
                    claimed = list(candidate_claim)
                    break
                self._add("shared_acquisition_resource_skipped_cohorts")
            if key is None or cohort is None:
                break
            if not claimed:  # pragma: no cover - guarded above
                self._add("shared_acquisition_resource_blocked_cohorts")
                break
            packet_keys = [key]
            packet_cohorts = [cohort]
            # Preserve the mover's occupancy-derived adjacent-layer packet only
            # while every next layer is also the next global EDF choice. This
            # is coalescing after scheduling, not a layer-order shortcut.
            while len(packet_keys) < self._shared_layers_per_dispatch:
                next_identity = self._shared_queue.next_released_identity(
                    now_ns=now_ns
                )
                if next_identity is None:
                    break
                next_key = self._shared_identity_owner.get(next_identity)
                next_cohort = (
                    None if next_key is None else self._shared_cohorts.get(next_key)
                )
                previous = packet_cohorts[-1]
                if (
                    next_key is None
                    or next_cohort is None
                    or next_cohort.pending is not cohort.pending
                    or next_cohort.local_layer != previous.local_layer + 1
                ):
                    break
                next_claimed = self._shared_queue.claim_cohort(
                    next_cohort.identities, now_ns=now_ns
                )
                if not next_claimed:
                    break
                claimed.extend(next_claimed)
                packet_keys.append(next_key)
                packet_cohorts.append(next_cohort)
            try:
                plan = self.transfer_plan(cohort.pending)
                self._transport.prepare(
                    cohort.pending,
                    plan,
                    first_local_layer=cohort.local_layer,
                    last_local_layer=packet_cohorts[-1].local_layer + 1,
                )
                for packet_cohort in packet_cohorts:
                    publication = packet_cohort.pending.prefetched_layers.get(
                        packet_cohort.local_layer
                    )
                    if publication is None:
                        raise RuntimeError(
                            "shared Host transport returned without a layer fence"
                        )
                for group in claimed:
                    self._shared_queue.publish_fence(group.identity)
            except BaseException:
                for group in claimed:
                    state = self._shared_queue.state(group.identity)
                    if state in {
                        SharedAcquisitionState.SUBMITTED,
                        SharedAcquisitionState.FENCE_PUBLISHED,
                    }:
                        self._shared_queue.fail(group.identity)
                raise
            self._shared_active.append(tuple(packet_keys))
            submitted += len(packet_keys)
            self._add("shared_acquisition_submitted_packets")
            self._add("shared_acquisition_submitted_cohorts", len(packet_keys))
            self._add("shared_acquisition_submitted_groups", len(claimed))
        return submitted

    def _forget_shared_cohort(
        self, key: tuple[int, int], cohort: _SharedHostCohort
    ) -> None:
        self._shared_queue.forget_terminal(cohort.identities)
        self._shared_cohorts.pop(key, None)
        for identity in cohort.identities:
            self._shared_identity_owner.pop(identity, None)
        self._add("shared_acquisition_retired_cohorts")

    def poll_and_pump_shared_acquisition(self, pending: PendingHostLoad) -> None:
        """Bridge callback used by admission progress for every pending lease."""

        if not getattr(pending, "shared_acquisition_registered", False):
            return
        self.progress_shared_acquisition()
        self._pump_shared_acquisition()

    def ensure_layer_published(
        self, pending: PendingHostLoad, local_layer: int
    ) -> None:
        """Wait until shared EDF publishes one required whole-layer fence.

        A mixed SGLang batch cannot always be held at admission: an older
        acquisition may occupy the finite global-link horizon while resident
        decode in the new batch remains runnable.  The A2 whole-layer stock
        consumer nevertheless needs a concrete layer fence before numerical
        execution starts.  Advance only by waiting for the oldest already-
        submitted packet; never bypass EDF by publishing the target directly.
        """

        if not 0 <= local_layer < self._model_layer_count:
            raise ValueError("required Host acquisition layer is outside the model")
        if not getattr(pending, "shared_acquisition_registered", False):
            raise RuntimeError("required Host layer has no shared acquisition owner")
        if local_layer in pending.prefetched_layers:
            return
        self._add("shared_acquisition_publication_waits")
        # No new jobs can be registered from this scheduler thread while it is
        # inside metadata binding.  The current finite record count therefore
        # bounds the number of completion edges needed to reach the target.
        maximum_rounds = self._shared_queue.group_count + 1
        for _ in range(maximum_rounds):
            self.progress_shared_acquisition()
            self._pump_shared_acquisition()
            if local_layer in pending.prefetched_layers:
                return
            if not self._shared_active:
                raise RuntimeError(
                    "shared EDF has no in-flight packet for a required Host layer"
                )
            first_packet = self._shared_active[0]
            if not first_packet:
                raise RuntimeError("shared EDF retained an empty transport packet")
            cohort = self._shared_cohorts.get(first_packet[0])
            if cohort is None:
                raise RuntimeError("shared EDF lost its oldest transport cohort")
            publication = cohort.pending.prefetched_layers.get(cohort.local_layer)
            if publication is None:
                raise RuntimeError("shared EDF oldest cohort has no published fence")
            publication.ready_event.synchronize()
            self._add("shared_acquisition_publication_wait_rounds")
        raise RuntimeError("shared EDF did not reach a required Host layer")

    def cancel_shared_acquisition(self, pending: PendingHostLoad) -> None:
        """Stop undispatched work and fence in-flight reservations before reuse."""

        if not getattr(pending, "shared_acquisition_registered", False):
            return
        for local_layer in range(self._model_layer_count):
            key = (pending.lease_id, local_layer)
            cohort = self._shared_cohorts.get(key)
            if cohort is None:
                continue
            cohort.cancel_requested = True
            for identity in cohort.identities:
                state = self._shared_queue.state(identity)
                if state not in {
                    SharedAcquisitionState.CANCELLED,
                    SharedAcquisitionState.FAILED,
                }:
                    self._shared_queue.cancel(identity)
            if all(
                self._shared_queue.state(identity)
                in {
                    SharedAcquisitionState.CANCELLED,
                    SharedAcquisitionState.FAILED,
                }
                for identity in cohort.identities
            ):
                self._forget_shared_cohort(key, cohort)
        self.progress_shared_acquisition()
        self._pump_shared_acquisition()
        self._add("shared_acquisition_cancelled_leases")

    def plan_published_consumers(
        self,
        pending: PendingHostLoad,
        batch: SglangForwardEpoch,
    ) -> None:
        """Bind partial consumers only at an explicit calibration/causal edge.

        A published progressive layer merely exposes wave readiness; it does
        not prove that the GPU will reach attention before the complete layer
        fence.  AUTO's normal ``predicted_gain`` result is currently a
        whole-forward execution estimate, not that per-layer arrival proof, so
        it must not turn a racing CPU ``Event.query`` into partial execution.
        Bounded probes and the explicit dependency-aware experiment retain the
        path needed to measure and validate a future closed-loop policy.
        """

        if batch.pending_host_load is not pending:
            raise RuntimeError("consumer planning lost its active HiCache lease")
        execution = batch.host_execution
        if (
            execution is None
            or not execution.uses_progressive_consumer
            or not execution.overlap_initial
        ):
            return
        progressive_layers = {
            local_layer
            for local_layer, publication in pending.prefetched_layers.items()
            if getattr(publication, "transfer_first_slot", None) is not None
        }
        if not progressive_layers:
            return
        explicit_measurement = execution.selection_reason in {
            "calibration_probe",
            "consumer_policy_probe",
            "forced_dependency_aware",
        }
        planned = (
            progressive_layers
            if explicit_measurement
            else progressive_layers & set(pending.planned_progressive_layers)
        ) - batch.modeled_ready_by_attention_layers
        if not planned:
            self._add("partial_consumer_unproven_layers", len(progressive_layers))
            return
        batch.planned_progressive_consumer_layers.update(planned)
        self._add("partial_consumer_planned_layers", len(planned))

    def plan_published_consumer_layer(
        self,
        pending: PendingHostLoad,
        batch: SglangForwardEpoch,
        local_layer: int,
    ) -> bool:
        """Authorize one layer published after the metadata-time snapshot.

        The global queue can publish a later layer only when attention is
        about to reach it.  Forced A3 and calibrated production policy must
        apply the same authorization rule at that edge; otherwise a late
        publication silently loses partial consumption and falls back to a
        blocking whole-layer wait.
        """

        if batch.pending_host_load is not pending:
            raise RuntimeError("consumer planning lost its active HiCache lease")
        if not 0 <= local_layer < self._model_layer_count:
            raise ValueError("progressive consumer layer is outside the model")
        execution = batch.host_execution
        publication = pending.prefetched_layers.get(local_layer)
        if (
            execution is None
            or not execution.uses_progressive_consumer
            or not execution.overlap_initial
            or publication is None
            or getattr(publication, "transfer_first_slot", None) is None
            or local_layer in batch.modeled_ready_by_attention_layers
        ):
            return False
        explicit_measurement = execution.selection_reason in {
            "calibration_probe",
            "consumer_policy_probe",
            "forced_dependency_aware",
        }
        if not explicit_measurement and local_layer not in pending.planned_progressive_layers:
            self._add("partial_consumer_unproven_layers")
            return False
        if local_layer in batch.planned_progressive_consumer_layers:
            return True
        batch.planned_progressive_consumer_layers.add(local_layer)
        self._add("partial_consumer_planned_layers")
        return True

    def publish_range(
        self,
        pending: PendingHostLoad,
        first_local_layer: int,
        last_local_layer: int,
    ) -> None:
        """Publish one explicit range from an immutable lease plan."""

        plan = self.transfer_plan(pending)
        self._transport.prepare(
            pending,
            plan,
            first_local_layer=first_local_layer,
            last_local_layer=last_local_layer,
        )

    def publish_missing(
        self,
        pending: PendingHostLoad,
        *,
        exclude: frozenset[int] = frozenset(),
    ) -> int:
        """Publish every missing layer from an immutable lease plan."""

        layer_count = int(pending.controller.layer_num)
        if any(layer < 0 or layer >= layer_count for layer in exclude):
            raise RuntimeError("typed-demand exclusion is outside the model")
        missing = tuple(
            layer
            for layer in range(layer_count)
            if layer not in pending.prefetched_layers and layer not in exclude
        )
        if not missing:
            return 0
        plan = self.transfer_plan(pending)
        ranges: list[tuple[int, int]] = []
        range_begin = previous = missing[0]
        for layer in missing[1:]:
            if layer != previous + 1:
                ranges.append((range_begin, previous + 1))
                range_begin = layer
            previous = layer
        ranges.append((range_begin, previous + 1))
        for first_layer, last_layer in ranges:
            self._transport.prepare(
                pending,
                plan,
                first_local_layer=first_layer,
                last_local_layer=last_layer,
            )
        return len(missing)

    def transfer_plan(
        self,
        pending: PendingHostLoad,
        *,
        layer_service_key: LayerServiceKey | None = None,
        layer_curve: AcquisitionServiceCurve | None = None,
    ) -> HostTransferLeasePlan:
        """Build immutable K/V descriptors once, before frontier slicing."""

        cached = pending.transfer_plan
        layer_count = int(pending.controller.layer_num)
        transfer_count = int(pending.device_indices.numel())
        if cached is not None:
            if (
                cached.mover.row_count != transfer_count
                or len(cached.layers) != layer_count
            ):
                raise RuntimeError("HiCache transfer geometry changed during a lease")
            self._add("host_transfer_plan_reuses")
            return cached
        if transfer_count <= 0 or transfer_count != int(pending.host_indices.numel()):
            raise RuntimeError("HiCache host pipeline has no promoted pages")
        if not pending.row_bytes_by_layer:
            self.account_selection(pending)
        if len(pending.row_bytes_by_layer) != layer_count:
            raise RuntimeError("HiCache acquisition has incomplete row geometry")
        mover = self._movers.plan(
            pending,
            pending.row_bytes_by_layer,
            transfer_count,
            layer_service_key=layer_service_key,
            layer_curve=layer_curve,
            collect_layer_profiles=self._calibration.collect,
        )
        if pending.lease_id <= 0 or pending.lease_id > 0xFFFFFFFF:
            raise RuntimeError("HiCache lease version exceeds the runtime ABI")
        result = build_host_transfer_lease_plan(
            pending.controller,
            mover,
            pending.row_bytes_by_layer,
            object_id_bases=tuple(
                pipeline_object_id(
                    pending.consumer_index,
                    layer_count,
                    local_layer,
                    self._sm_acquisition_waves,
                )
                for local_layer in range(layer_count)
            ),
            object_version=pending.lease_id,
            sm_acquisition_waves=self._sm_acquisition_waves,
            preferred_wave_row_ends=self._preferred_wave_row_ends(pending),
        )
        layer_bytes = tuple(
            key_bytes + value_bytes for key_bytes, value_bytes in result.layer_geometry
        )
        if pending.layer_bytes != layer_bytes:
            raise RuntimeError(
                "HiCache transfer plan changed its captured byte geometry"
            )
        pending.transfer_plan = result
        self._add("host_transfer_plan_builds")
        return result

    def _preferred_wave_row_ends(
        self, pending: PendingHostLoad
    ) -> tuple[int, ...]:
        """Align finite SM readiness waves to scheduler acquisition groups."""

        groups = pending.scheduled_acquisition_groups
        if not groups:
            return ()
        operation_by_id = {
            operation.operation_id: operation
            for operation in pending.operation_ranges()
        }
        exact_ends = tuple(
            operation_by_id[group.operation_id].row_begin + group.row_end
            for group in groups
        )
        maximum = self._sm_acquisition_waves
        if len(exact_ends) <= maximum:
            return exact_ends
        # Keep the number of physical events bounded while cutting only between
        # complete request operations. The ceil-index rule is deterministic and
        # balances operation count without pretending a coalesced wave is one
        # request's independent readiness record.
        count = len(exact_ends)
        ends = tuple(
            exact_ends[min(count, (wave * count + maximum - 1) // maximum) - 1]
            for wave in range(1, maximum + 1)
        )
        if len(set(ends)) != len(ends) or ends[-1] != exact_ends[-1]:
            raise RuntimeError("Host operation waves could not be bounded exactly")
        return ends

    def account_selection(self, pending: PendingHostLoad) -> None:
        """Count unique exact tier demand once per ownership lease."""

        if pending.selection_accounted:
            return
        row_count = int(pending.device_indices.numel())
        if row_count <= 0:
            raise RuntimeError("SGLang acquisition lease contains no selected rows")
        controller = pending.controller
        host_keys = tuple(controller.mem_pool_host.k_data_refs)
        host_values = tuple(controller.mem_pool_host.v_data_refs)
        layer_count = int(controller.layer_num)
        if (
            len(host_keys) != layer_count
            or len(host_values) != layer_count
            or not host_keys
        ):
            raise RuntimeError("SGLang acquisition lease has incomplete layer geometry")
        row_bytes_by_layer = tuple(
            (
                int(key[0].numel()) * key.element_size(),
                int(value[0].numel()) * value.element_size(),
            )
            for key, value in zip(host_keys, host_values, strict=True)
        )
        layer_bytes = tuple(
            row_count * (key_bytes + value_bytes)
            for key_bytes, value_bytes in row_bytes_by_layer
        )
        if pending.layer_bytes and pending.layer_bytes != layer_bytes:
            raise RuntimeError("SGLang acquisition byte geometry changed after capture")
        if (
            pending.row_bytes_by_layer
            and pending.row_bytes_by_layer != row_bytes_by_layer
        ):
            raise RuntimeError("SGLang acquisition row geometry changed after capture")
        pending.layer_bytes = layer_bytes
        pending.row_bytes_by_layer = row_bytes_by_layer
        pending.selection_accounted = True
        self._add("tier_selected_leases")
        self._add("tier_selected_rows", row_count)
        self._add("tier_selected_bytes", sum(layer_bytes))
        # HiCache load-back is exact-dense: this framework path neither
        # approximates nor drops a candidate row.
        self._add("tier_candidate_bytes", sum(layer_bytes))

    def advance_after_attention(
        self,
        pending: PendingHostLoad,
        batch: SglangForwardEpoch,
        completed_local_layer: int,
        *,
        enqueue_fragment: Callable[[], bool] | None = None,
    ) -> None:
        """Publish the maximal calibrated layer prefix after one consumer.

        Transformer layer deadlines are structurally ordered.  The compiled
        simultaneous-release EDF test therefore proves how far whole-layer
        transport may run ahead; it does not pretend to reorder layers.  A
        modeled miss remains unpublished so the typed consumer can acquire
        exact request groups, optionally seeding one fragment lookahead.
        """

        if batch.pending_host_load is not pending:
            raise RuntimeError("deadline frontier lost its active HiCache lease")
        if getattr(pending, "shared_acquisition_registered", False):
            self.retire_layer(pending, completed_local_layer)
            self.progress_shared_acquisition()
            self._pump_shared_acquisition()
            return
        acquisition = pending.acquisition
        if acquisition is not None:
            self.retire_layer(pending, completed_local_layer)
            # The current Host backend normally fills its finite queue at
            # capture. A bounded-inflight backend can refill through this same
            # ownership edge without putting policy in the transport.
            if not acquisition.fully_published:
                submitted = self.submit(pending)
                self._add("host_acquisition_refill_jobs", submitted)
            return
        # A direct/eager producer has already published every exact layer and
        # owns no finite acquisition queue. There is nothing for EDF to admit
        # or refill; rebuilding the same model once per attention layer is pure
        # launch-thread overhead on the stock fast path.
        if len(pending.prefetched_layers) == self._model_layer_count:
            return
        if not self._frontier_enabled:
            return

        layer_count = int(pending.controller.layer_num)
        if layer_count != self._model_layer_count:
            raise RuntimeError("deadline frontier and model layer counts disagree")
        ready_prefix = completed_local_layer + 1
        if not 0 < ready_prefix <= layer_count:
            raise RuntimeError("deadline frontier received an invalid layer prefix")
        if ready_prefix == layer_count:
            return

        model = batch.deadline_model
        frontier_plan_built = False
        if not batch.deadline_model_initialized:
            self._movers.collect_profiles()
            self._calibration.collect()
            if pending.mover_plan is None:
                # Descriptor preparation follows the current attention launch,
                # so it is outside the next layer's first-dispatch dependency.
                self.transfer_plan(pending)
            curve = (
                None
                if batch.layer_service_key is None
                else self._calibration.curve(batch.layer_service_key)
            )
            model = (
                None if curve is None else self.deadline_model_for_curve(pending, curve)
            )
            if model is not None:
                batch.deadline_model = model
                batch.deadline_model_initialized = True
                self._add("deadline_frontier_model_builds")
        else:
            self._add("deadline_frontier_model_reuses")

        if model is None:
            self._add("deadline_frontier_uncalibrated")
            # A read-only deployment profile is a hard learning boundary, not
            # merely a persistence policy.  An unseen mover scale must use the
            # structural, work-conserving submission path without issuing a
            # bounded calibration frontier inside the measured request.
            calibration_probe = not getattr(
                self._movers, "calibration_frozen", False
            ) and not self._movers.lease_calibrated(pending)
            if calibration_probe and ready_prefix not in pending.prefetched_layers:
                probe_end = calibration_probe_end(
                    ready_prefix,
                    layer_count,
                    self._frontier_layers_per_wave,
                )
                self.publish_range(pending, ready_prefix, probe_end)
                self._add(
                    "deadline_frontier_calibration_layers",
                    probe_end - ready_prefix,
                )
            elif (
                enqueue_fragment is not None
                and ready_prefix not in pending.prefetched_layers
            ):
                enqueue_fragment()
            return

        if batch.deadline_frontier is None:
            batch.deadline_frontier = model.compile_after_attention_frontier()
            frontier_plan_built = True
            self._add("deadline_frontier_plan_builds")
        frontier = batch.deadline_frontier
        if frontier is None or frontier.layer_count != layer_count:
            raise RuntimeError("deadline frontier has no compiled service plan")

        feasible_end = frontier.feasible_end_after_attention(completed_local_layer)
        self._add("deadline_frontier_plans")
        self._add("deadline_frontier_plan_reuses", int(not frontier_plan_built))
        if feasible_end != layer_count:
            self._add("deadline_frontier_first_missed_layer_sum", feasible_end)

        publish_begin = ready_prefix
        while (
            publish_begin < layer_count and publish_begin in pending.prefetched_layers
        ):
            publish_begin += 1
        if publish_begin < feasible_end:
            self.publish_range(pending, publish_begin, feasible_end)
            self._add(
                "deadline_frontier_published_layers",
                feasible_end - publish_begin,
            )

        # The frozen service model says this whole prefix completes before its
        # attention deadlines even if it starts at the current consumer edge.
        # Previously the dispatcher discarded that result and queried the event
        # from a CPU thread running many layers ahead of the GPU. Preserve the
        # model decision explicitly; execution still waits on each producer
        # fence, so this can affect only consumer form and performance.
        modeled_ready = set(range(ready_prefix, feasible_end))
        if not modeled_ready.issubset(pending.prefetched_layers):
            raise RuntimeError(
                "deadline-feasible acquisition prefix was not fully published"
            )
        newly_modeled = modeled_ready - batch.modeled_ready_by_attention_layers
        batch.modeled_ready_by_attention_layers.update(modeled_ready)
        self._add("deadline_frontier_modeled_ready_layers", len(newly_modeled))

        fragment_enqueued = False
        if (
            feasible_end == ready_prefix
            and ready_prefix not in pending.prefetched_layers
            and enqueue_fragment is not None
        ):
            fragment_enqueued = enqueue_fragment()
            if fragment_enqueued:
                self._add("deadline_frontier_fragment_layers")
        if publish_begin >= feasible_end and not fragment_enqueued:
            self._add("deadline_frontier_noop_calls")

    def retire_layer(self, pending: PendingHostLoad, local_layer: int) -> None:
        """Retire transport ownership after a numerical consumer is ordered."""

        if getattr(pending, "shared_acquisition_registered", False):
            key = (pending.lease_id, local_layer)
            cohort = self._shared_cohorts.get(key)
            if cohort is None:
                raise RuntimeError("shared Host layer has no acquisition cohort")
            if cohort.consumer_ordered:
                raise RuntimeError("shared Host layer was consumed more than once")
            for identity in cohort.identities:
                self._shared_queue.consume(identity)
            cohort.consumer_ordered = True
            if all(
                self._shared_queue.state(identity)
                is SharedAcquisitionState.CONSUMED
                for identity in cohort.identities
            ):
                self._forget_shared_cohort(key, cohort)
            self._add("host_acquisition_layers_consumed")
            return
        acquisition = pending.acquisition
        if acquisition is None:
            return
        acquisition.retire(local_layer)
        self._add("host_acquisition_layers_consumed")

    def _add(self, name: str, value: int = 1) -> None:
        self._stats[name] = self._stats.get(name, 0) + value
