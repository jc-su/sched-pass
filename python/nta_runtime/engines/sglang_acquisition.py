"""Lease-scoped Host acquisition coordination for the SGLang integration.

This component is the sole owner of the control path from an exact HiCache
lease to immutable transfer descriptors, a calibrated deadline model, finite
submission, and layer retirement.  The transport receives an already-frozen
plan and therefore cannot allocate or invoke scheduling policy from its
steady-state submission path.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any

from nta_runtime.acquisition_scheduler import (
    AcquisitionServiceCurve,
    LayerAcquisition,
    LayerAcquisitionModel,
)
from nta_runtime.adapters.sglang import SglangExecutionConfig
from nta_runtime.engines.sglang_calibration import (
    LayerServiceKey,
    SglangLayerServiceCalibration,
)
from nta_runtime.engines.sglang_hicache import PendingHostLoad
from nta_runtime.engines.sglang_pipeline import SglangHostTransport
from nta_runtime.engines.sglang_planning import pipeline_object_id
from nta_runtime.engines.sglang_transfer import (
    HostMoverController,
    HostTransferLeasePlan,
    build_host_transfer_lease_plan,
)
from nta_runtime.execution_planner import HostExecutionMode
from nta_runtime.execution_protocol import ProtocolKind

if TYPE_CHECKING:
    from nta_runtime.engines.sglang_state import _ActiveBatch


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
        movers: HostMoverController,
        calibration: SglangLayerServiceCalibration,
        transport: SglangHostTransport,
        stats: MutableMapping[str, Any],
    ) -> None:
        if model_layer_count <= 0 or sm_acquisition_waves <= 0:
            raise ValueError("SGLang Host acquisition geometry must be positive")
        self._device_pool = device_pool
        self._execution_config = execution_config
        self._tenant_isolation_enabled = bool(tenant_isolation_enabled)
        self._model_layer_count = model_layer_count
        self._sm_acquisition_waves = sm_acquisition_waves
        self._movers = movers
        self._calibration = calibration
        self._transport = transport
        self._stats = stats

    def capture(self, pending: PendingHostLoad) -> None:
        """Capture physical ownership and start any unconditional acquisition."""

        if pending.controller.mem_pool_device is not self._device_pool:
            raise RuntimeError("HiCache lease belongs to a different device pool")
        layer_count = int(pending.controller.layer_num)
        if layer_count != self._model_layer_count:
            raise RuntimeError("HiCache load and model layer counts disagree")
        self.account_selection(pending)
        conventional = self._execution_config.protocol.kind is ProtocolKind.CONVENTIONAL
        eager_dense = (
            not conventional
            and not self._tenant_isolation_enabled
            and self._execution_config.host_execution_mode
            is not HostExecutionMode.DEVICE_BULK
        )
        initial_layers = 0
        if conventional:
            self.publish_range(pending, 0, layer_count)
            initial_layers = layer_count
        elif eager_dense:
            # Descriptor ownership is frozen before submission.  No active
            # ForwardBatch exists at this edge, so mover selection can use only
            # deployment-local observations and never stale batch state.
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
        active_batch: _ActiveBatch | None = None,
    ) -> bool:
        """Bind one calibrated EDF proof to immutable physical ownership."""

        acquisition = pending.acquisition
        if acquisition is not None and acquisition.model is not None:
            return True
        if acquisition is None and pending.prefetched_layers:
            return False
        shape_key = self._calibration.shape_key(batch)
        curve = self._calibration.curve_for_batch(batch)
        if shape_key is None or curve is None:
            self._add("host_acquisition_shape_uncalibrated")
            return False
        if active_batch is not None and active_batch.pending_host_load is pending:
            active_batch.layer_service_key = shape_key
        self._movers.collect_profiles()
        self.transfer_plan(
            pending,
            layer_service_key=shape_key,
            layer_curve=curve,
        )
        model = self.deadline_model_for_curve(pending, curve)
        if model is None:
            self._add(
                "host_acquisition_mover_uncalibrated"
                if not self._movers.lease_calibrated(pending)
                else "host_acquisition_model_rejected"
            )
            return False
        if acquisition is None:
            acquisition = LayerAcquisition(pending.layer_bytes)
            pending.acquisition = acquisition
            self._add("host_acquisition_jobs_prepared", len(model.layer_bytes))
        if acquisition.bind_model(model):
            self._add("host_acquisition_models_bound")
        return True

    def prepare_admission(self, pending: PendingHostLoad, batch: Any) -> bool:
        """Prepare descriptors only when admission has a usable model."""

        if (
            self._tenant_isolation_enabled
            or self._execution_config.host_execution_mode
            is HostExecutionMode.DEVICE_BULK
        ):
            return False
        already_prepared = pending.acquisition is not None
        ready = self.prepare_owner(pending, batch)
        if ready and not already_prepared:
            self._add("admission_acquisition_groups_prepared")
        return ready

    def start_admission(self, pending: PendingHostLoad, batch: Any) -> None:
        """Start the finite queue after admission has bounded its delay."""

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

    def retire_layer(self, pending: PendingHostLoad, local_layer: int) -> None:
        """Retire transport ownership after a numerical consumer is ordered."""

        acquisition = pending.acquisition
        if acquisition is None:
            return
        acquisition.retire(local_layer)
        self._add("host_acquisition_layers_consumed")

    def _add(self, name: str, value: int = 1) -> None:
        self._stats[name] = self._stats.get(name, 0) + value
