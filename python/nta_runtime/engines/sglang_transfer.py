"""Typed, lease-scoped transfer planning for SGLang HiCache loads."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, replace
import os
import time
from typing import Any

import torch

from nta_runtime.indexed_transfer import (
    ContiguousPairRun,
    IndexedMoverServiceModel,
    IndexedPairLayout,
    StridedCopyGroup,
)
from nta_runtime.indexed_transfer_torch import plan_indexed_tensor_mover
from nta_runtime.runtime import IndexedHostObject
from nta_runtime.engines.sglang_planning import (
    byte_scale_bucket,
    maximum_mover_wave_bytes,
    mover_layout_required,
)
from nta_runtime.acquisition_scheduler import AcquisitionServiceCurve


_HOST_MOVER_STATE_SCHEMA = 1


def _profile_int(
    value: Any,
    owner: str,
    *,
    minimum: int = 0,
    maximum: int = (1 << 63) - 1,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{owner} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{owner} is outside its bound")
    return value


def _profile_mapping(value: Any, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{owner} must be a string-keyed object")
    return value


def _profile_fields(
    value: Mapping[str, Any], expected: frozenset[str], owner: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{owner} fields disagree "
            f"(missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)})"
        )


def host_mover_service_model_from_environment(
    environ: dict[str, str] | None = None,
) -> IndexedMoverServiceModel:
    """Load an explicit deployment calibration, failing closed to SM.

    A copy-engine bandwidth without its per-operation issue cost (or vice
    versa) is not a usable calibration. Keeping both optional makes the
    uncalibrated state representable instead of silently substituting a byte
    threshold. ``NTA_EXECUTION_HOST_MOVER_CALIBRATION_BYTES`` can bind an
    externally measured curve to its physical power-of-two wave class, and
    copy/compute overlap remains zero unless an explicit measured efficiency
    is supplied.
    """

    values = os.environ if environ is None else environ
    host_bandwidth = int(
        values.get("NTA_TIER_HOST_STAGED_BANDWIDTH_BPS", 30_000_000_000)
    )
    sm_bandwidth_text = values.get("NTA_EXECUTION_HOST_SM_BANDWIDTH_BPS")
    sm_bandwidth = int(
        host_bandwidth if sm_bandwidth_text is None else sm_bandwidth_text
    )
    copy_bandwidth_text = values.get("NTA_EXECUTION_HOST_COPY_BANDWIDTH_BPS")
    copy_operation_text = values.get("NTA_EXECUTION_HOST_COPY_OPERATION_NS")
    if (copy_bandwidth_text is None) != (copy_operation_text is None):
        raise ValueError(
            "NTA host-copy calibration requires both COPY_BANDWIDTH_BPS "
            "and COPY_OPERATION_NS"
        )
    minimum_samples = min(
        32, int(values.get("NTA_EXECUTION_HOST_MOVER_CALIBRATION_SAMPLES", 3))
    )
    calibration_bytes_text = values.get("NTA_EXECUTION_HOST_MOVER_CALIBRATION_BYTES")
    calibration_scale_bucket: int | None = None
    if calibration_bytes_text is not None:
        calibration_bytes = int(calibration_bytes_text)
        if calibration_bytes <= 0:
            raise ValueError("NTA host-mover calibration bytes must be positive")
        calibration_scale_bucket = calibration_bytes.bit_length() - 1
    overlap_text = values.get("NTA_EXECUTION_HOST_COPY_COMPUTE_OVERLAP_EFFICIENCY")
    return IndexedMoverServiceModel(
        sm_bandwidth_bytes_per_second=sm_bandwidth,
        copy_bandwidth_bytes_per_second=(
            None if copy_bandwidth_text is None else int(copy_bandwidth_text)
        ),
        copy_operation_ns=(
            None if copy_operation_text is None else int(copy_operation_text)
        ),
        hybrid_join_ns=int(values.get("NTA_EXECUTION_HOST_HYBRID_JOIN_NS", 0)),
        minimum_gain=float(values.get("NTA_EXECUTION_HOST_MOVER_MIN_GAIN", 1.03)),
        # Explicit deployment calibrations represent a completed calibration
        # campaign. Defaults remain priors and trigger bounded in-process probes
        # before ``auto`` compares the two issuers.
        sm_samples=0 if sm_bandwidth_text is None else minimum_samples,
        copy_samples=0 if copy_bandwidth_text is None else minimum_samples,
        minimum_calibration_samples=minimum_samples,
        calibration_scale_bucket=calibration_scale_bucket,
        copy_compute_overlap_efficiency=(
            0.0 if overlap_text is None else float(overlap_text)
        ),
        overlap_samples=0 if overlap_text is None else minimum_samples,
    )


@dataclass(frozen=True)
class HostMoverLeasePlan:
    """One exact index partition reused by every layer of a host-load lease.

    SGLang publishes one source/destination page map for the whole model.  Run
    decomposition is therefore a lease property, not a layer-wave property;
    retaining it here prevents O(pages) analysis and descriptor D2H syncs from
    recurring when a later frontier is enqueued.
    """

    row_count: int
    kind: str
    copy_runs: tuple[ContiguousPairRun, ...]
    sm_source_indices: torch.Tensor
    sm_destination_indices: torch.Tensor
    layout: IndexedPairLayout | None
    layout_cpu_ns: int
    predicted_sm_ns: int
    predicted_selected_ns: int | None
    selection_reason: str

    def __post_init__(self) -> None:
        if self.row_count <= 0:
            raise ValueError("host mover lease plan has no rows")
        if self.kind not in {"sm", "copy_engine", "hybrid"}:
            raise ValueError("host mover lease plan has an invalid kind")
        if self.layout_cpu_ns < 0:
            raise ValueError("host mover lease planning time cannot be negative")
        if self.predicted_sm_ns <= 0 or (
            self.predicted_selected_ns is not None and self.predicted_selected_ns <= 0
        ):
            raise ValueError("host mover lease predictions must be positive")
        if self.sm_source_indices.device != self.sm_destination_indices.device:
            raise ValueError("host mover SM maps must share one device")
        if self.sm_source_indices.dtype is not torch.int32 or (
            self.sm_destination_indices.dtype is not torch.int32
        ):
            raise ValueError("host mover SM maps must use ABI int32 storage")
        if self.sm_source_indices.ndim != 1 or (self.sm_destination_indices.ndim != 1):
            raise ValueError("host mover SM maps must be vectors")
        if self.sm_source_indices.numel() != self.sm_destination_indices.numel():
            raise ValueError("host mover SM maps disagree")
        if self.copy_row_count + self.sm_row_count != self.row_count:
            raise ValueError("host mover lease plan does not cover every row")
        expected_kind = (
            "sm"
            if not self.copy_runs
            else "copy_engine"
            if self.sm_row_count == 0
            else "hybrid"
        )
        if self.kind != expected_kind:
            raise ValueError("host mover lease kind disagrees with its partition")

    @property
    def copy_row_count(self) -> int:
        return sum(run.row_count for run in self.copy_runs)

    @property
    def sm_row_count(self) -> int:
        return int(self.sm_source_indices.numel())

    @property
    def retained_tensors(self) -> tuple[torch.Tensor, ...]:
        if self.sm_row_count == 0:
            return ()
        return (self.sm_source_indices, self.sm_destination_indices)


@dataclass(frozen=True)
class MoverProfile:
    """One separable mover-wave observation retired off the hot path."""

    start: torch.cuda.Event
    finish: torch.cuda.Event
    engine: str
    transfer_bytes: int
    service_scale_bytes: int
    operation_count: int
    issue_cpu_ns: int
    calibration: bool = False

    def __post_init__(self) -> None:
        if self.engine not in {"sm", "copy_engine"}:
            raise ValueError("mover profile has an invalid engine")
        if (
            self.transfer_bytes <= 0
            or self.service_scale_bytes < self.transfer_bytes
            or self.operation_count <= 0
        ):
            raise ValueError("mover profile requires positive service geometry")
        if self.issue_cpu_ns < 0:
            raise ValueError("mover profile issue time cannot be negative")
        if self.engine == "sm" and self.issue_cpu_ns != 0:
            raise ValueError("SM mover profiles cannot carry CPU issue time")


class HostMoverController:
    """Own mover selection, calibration, and lease-scoped index analysis.

    This is the only owner of SM/copy-engine service curves and outstanding
    mover observations.  The framework adapter supplies an exact HiCache lease
    and an optional calibrated layer curve; it does not inspect or mutate the
    controller's calibration state.
    """

    def __init__(
        self,
        *,
        policy: str,
        default_service_model: IndexedMoverServiceModel,
        calibration_samples: int,
        copy_engine_max_operations: int,
        frontier_layers_per_wave: int,
        profile_transfer: bool,
        frontier_enabled: bool,
        profile_index_layout: bool,
        profile_index_min_bytes: int,
        verify_index_map: bool,
        stats: MutableMapping[str, Any],
        frozen: bool = False,
    ) -> None:
        if policy not in {"auto", "sm", "copy_engine"}:
            raise ValueError("host mover policy is invalid")
        if (
            min(
                calibration_samples,
                copy_engine_max_operations,
                frontier_layers_per_wave,
                profile_index_min_bytes,
            )
            <= 0
        ):
            raise ValueError("host mover configuration must be positive")
        self._policy = policy
        self._frozen = bool(frozen)
        self._default_service_model = default_service_model
        self._calibration_samples = calibration_samples
        self._copy_engine_max_operations = copy_engine_max_operations
        self._frontier_layers_per_wave = frontier_layers_per_wave
        self._profile_transfer = profile_transfer
        self._frontier_enabled = frontier_enabled
        self._profile_index_layout = profile_index_layout
        self._profile_index_min_bytes = profile_index_min_bytes
        self._verify_index_map = verify_index_map
        self._stats = stats
        self._service_models: dict[int, IndexedMoverServiceModel] = {}
        self._last_service_bucket: int | None = None
        self._profiles: list[MoverProfile] = []
        self._profile_buckets: dict[str, dict[int, int]] = {
            "sm": {},
            "copy_engine": {},
        }
        self._profile_max_sample_bytes = {"sm": 0, "copy_engine": 0}

    @property
    def policy(self) -> str:
        return self._policy

    @property
    def pending_profile_count(self) -> int:
        return len(self._profiles)

    def representative_wave_bytes(
        self,
        row_bytes_by_layer: tuple[tuple[int, int], ...],
        transfer_count: int,
    ) -> int:
        return maximum_mover_wave_bytes(
            row_bytes_by_layer,
            transfer_count,
            self._frontier_layers_per_wave,
        )

    def _add_stat(self, name: str, value: int | float = 1) -> None:
        self._stats[name] = self._stats.get(name, 0) + value

    def _bucket_sample_count(self, engine: str, transfer_bytes: int) -> int:
        bucket = byte_scale_bucket(transfer_bytes)
        completed = self._profile_buckets[engine].get(bucket, 0)
        inflight = sum(
            profile.engine == engine
            and byte_scale_bucket(profile.transfer_bytes) == bucket
            for profile in self._profiles
        )
        return completed + inflight

    def _engine_precalibrated(self, engine: str, transfer_bytes: int) -> bool:
        model = self._default_service_model
        if not model.supports_transfer_scale(transfer_bytes):
            return False
        return model.sm_calibrated if engine == "sm" else model.copy_calibrated

    def _uncalibrated_prior(self) -> IndexedMoverServiceModel:
        return replace(
            self._default_service_model,
            sm_samples=0,
            copy_samples=0,
            copy_compute_overlap_efficiency=0.0,
            overlap_samples=0,
            calibration_scale_bucket=None,
        )

    def service_model(self, transfer_bytes: int) -> IndexedMoverServiceModel:
        bucket = byte_scale_bucket(transfer_bytes)
        self._last_service_bucket = bucket
        measured = self._service_models.get(bucket)
        if measured is not None:
            return measured
        configured = self._default_service_model
        return (
            configured
            if configured.supports_transfer_scale(transfer_bytes)
            else self._uncalibrated_prior()
        )

    def profile_enabled(
        self,
        engine: str,
        transfer_bytes: int,
        *,
        complete_calibration: bool = False,
    ) -> bool:
        if engine not in {"sm", "copy_engine"}:
            raise ValueError("unknown host mover engine")
        if transfer_bytes <= 0:
            raise ValueError("host mover profile bytes must be positive")
        if self._frozen:
            return False
        # A calibration decision applies to the complete acquisition frontier,
        # not whichever early waves happen to satisfy the minimum sample
        # count. Later waves can overlap resident compute and observe a very
        # different effective service rate. Retain every wave from the bounded
        # probe lease, then aggregate them before authorizing auto selection.
        if complete_calibration:
            return True
        if self._profile_transfer:
            return True
        if self._policy != "auto" and not self._frontier_enabled:
            return False
        if self._engine_precalibrated(engine, transfer_bytes):
            return False
        return (
            self._bucket_sample_count(engine, transfer_bytes)
            < self._calibration_samples
        )

    def scale_calibrated(self, engine: str, transfer_bytes: int) -> bool:
        if engine not in {"sm", "copy_engine"}:
            raise ValueError("unknown host mover engine")
        model = self.service_model(transfer_bytes)
        return model.sm_calibrated if engine == "sm" else model.copy_calibrated

    def admission_calibrated(self, pending: Any) -> bool:
        if not pending.row_bytes_by_layer:
            return False
        transfer_count = int(pending.device_indices.numel())
        representative_bytes = self.representative_wave_bytes(
            pending.row_bytes_by_layer, transfer_count
        )
        if self._policy == "sm":
            return self.scale_calibrated("sm", representative_bytes)
        if self._policy == "copy_engine":
            return self.scale_calibrated("copy_engine", representative_bytes)
        return self.scale_calibrated(
            "sm", representative_bytes
        ) and self.scale_calibrated("copy_engine", representative_bytes)

    def lease_calibrated(self, pending: Any) -> bool:
        mover = pending.mover_plan
        if mover is None or not pending.row_bytes_by_layer:
            return False
        transfer_count = int(pending.device_indices.numel())
        if transfer_count <= 0 or transfer_count != mover.row_count:
            raise RuntimeError("HiCache mover calibration geometry changed")
        representative_bytes = self.representative_wave_bytes(
            pending.row_bytes_by_layer, transfer_count
        )
        return not (
            mover.sm_row_count != 0
            and not self.scale_calibrated("sm", representative_bytes)
        ) and not (
            bool(mover.copy_runs)
            and not self.scale_calibrated("copy_engine", representative_bytes)
        )

    def record_profile(self, profile: MoverProfile) -> None:
        if self._frozen:
            raise RuntimeError("frozen host-mover policy cannot record calibration")
        self._profiles.append(profile)

    @staticmethod
    def _service_model_state(
        bucket: int, model: IndexedMoverServiceModel
    ) -> dict[str, Any]:
        return {
            "scale_bucket": bucket,
            "sm_bandwidth_bytes_per_second": model.sm_bandwidth_bytes_per_second,
            "copy_bandwidth_bytes_per_second": (
                model.copy_bandwidth_bytes_per_second
            ),
            "copy_operation_ns": model.copy_operation_ns,
            "hybrid_join_ns": model.hybrid_join_ns,
            "minimum_gain": model.minimum_gain,
            "sm_samples": model.sm_samples,
            "copy_samples": model.copy_samples,
            "minimum_calibration_samples": model.minimum_calibration_samples,
            "calibration_scale_bucket": model.calibration_scale_bucket,
            "copy_compute_overlap_efficiency": (
                model.copy_compute_overlap_efficiency
            ),
            "overlap_samples": model.overlap_samples,
        }

    def export_state(self) -> dict[str, Any]:
        """Return deployment-local mover curves without event ownership."""

        if self._profiles:
            raise RuntimeError("host mover cannot snapshot pending CUDA events")
        return {
            "schema": _HOST_MOVER_STATE_SCHEMA,
            "policy": self._policy,
            "calibration_samples": self._calibration_samples,
            "copy_engine_max_operations": self._copy_engine_max_operations,
            "frontier_layers_per_wave": self._frontier_layers_per_wave,
            "curves": [
                self._service_model_state(bucket, curve)
                for bucket, curve in sorted(self._service_models.items())
            ],
            "maximum_sample_bytes": dict(self._profile_max_sample_bytes),
        }

    def import_state(self, value: Any) -> int:
        """Restore validated mover curves before any lease is planned."""

        if self._profiles or self._service_models:
            raise RuntimeError("host mover calibration is not empty")
        state = _profile_mapping(value, "host-mover calibration")
        _profile_fields(
            state,
            frozenset(
                {
                    "schema",
                    "policy",
                    "calibration_samples",
                    "copy_engine_max_operations",
                    "frontier_layers_per_wave",
                    "curves",
                    "maximum_sample_bytes",
                }
            ),
            "host-mover calibration",
        )
        policy = state["policy"]
        if not isinstance(policy, str) or policy != self._policy:
            raise ValueError("host-mover calibration policy is incompatible")
        expected = (
            _HOST_MOVER_STATE_SCHEMA,
            self._calibration_samples,
            self._copy_engine_max_operations,
            self._frontier_layers_per_wave,
        )
        actual = tuple(
            _profile_int(state[name], f"host-mover {name}")
            for name in (
                "schema",
                "calibration_samples",
                "copy_engine_max_operations",
                "frontier_layers_per_wave",
            )
        )
        if actual != expected:
            raise ValueError("host-mover calibration geometry is incompatible")
        rows = state["curves"]
        if not isinstance(rows, list) or len(rows) > 1024:
            raise ValueError("host-mover service curves exceed their bound")
        curve_fields = frozenset(
            {
                "scale_bucket",
                "sm_bandwidth_bytes_per_second",
                "copy_bandwidth_bytes_per_second",
                "copy_operation_ns",
                "hybrid_join_ns",
                "minimum_gain",
                "sm_samples",
                "copy_samples",
                "minimum_calibration_samples",
                "calibration_scale_bucket",
                "copy_compute_overlap_efficiency",
                "overlap_samples",
            }
        )
        restored: dict[int, IndexedMoverServiceModel] = {}
        for raw in rows:
            row = _profile_mapping(raw, "host-mover curve")
            _profile_fields(row, curve_fields, "host-mover curve")
            bucket = _profile_int(row["scale_bucket"], "host-mover bucket", maximum=62)
            if bucket in restored:
                raise ValueError("host-mover profile repeats a scale bucket")
            optional_ints: dict[str, int | None] = {}
            for name in (
                "copy_bandwidth_bytes_per_second",
                "copy_operation_ns",
                "calibration_scale_bucket",
            ):
                raw_value = row[name]
                optional_ints[name] = (
                    None
                    if raw_value is None
                    else _profile_int(raw_value, f"host-mover {name}")
                )
            raw_gain = row["minimum_gain"]
            raw_overlap = row["copy_compute_overlap_efficiency"]
            if (
                isinstance(raw_gain, bool)
                or not isinstance(raw_gain, (int, float))
                or isinstance(raw_overlap, bool)
                or not isinstance(raw_overlap, (int, float))
            ):
                raise ValueError("host-mover floating calibration is invalid")
            curve = IndexedMoverServiceModel(
                sm_bandwidth_bytes_per_second=_profile_int(
                    row["sm_bandwidth_bytes_per_second"],
                    "host-mover SM bandwidth",
                    minimum=1,
                ),
                copy_bandwidth_bytes_per_second=optional_ints[
                    "copy_bandwidth_bytes_per_second"
                ],
                copy_operation_ns=optional_ints["copy_operation_ns"],
                hybrid_join_ns=_profile_int(
                    row["hybrid_join_ns"], "host-mover hybrid join"
                ),
                minimum_gain=float(raw_gain),
                sm_samples=_profile_int(
                    row["sm_samples"], "host-mover SM samples", maximum=1 << 20
                ),
                copy_samples=_profile_int(
                    row["copy_samples"],
                    "host-mover copy samples",
                    maximum=1 << 20,
                ),
                minimum_calibration_samples=_profile_int(
                    row["minimum_calibration_samples"],
                    "host-mover minimum samples",
                    minimum=1,
                ),
                calibration_scale_bucket=optional_ints[
                    "calibration_scale_bucket"
                ],
                copy_compute_overlap_efficiency=float(raw_overlap),
                overlap_samples=_profile_int(
                    row["overlap_samples"],
                    "host-mover overlap samples",
                    maximum=1 << 20,
                ),
            )
            default = self._default_service_model
            if (
                curve.calibration_scale_bucket != bucket
                or curve.minimum_calibration_samples != self._calibration_samples
                or curve.minimum_gain != default.minimum_gain
                or curve.hybrid_join_ns != default.hybrid_join_ns
            ):
                raise ValueError("host-mover curve contract is incompatible")
            restored[bucket] = curve
        maximum_bytes = _profile_mapping(
            state["maximum_sample_bytes"], "host-mover maximum sample bytes"
        )
        _profile_fields(
            maximum_bytes,
            frozenset({"sm", "copy_engine"}),
            "host-mover maximum sample bytes",
        )
        self._service_models = restored
        self._profile_buckets = {
            "sm": {bucket: curve.sm_samples for bucket, curve in restored.items()},
            "copy_engine": {
                bucket: curve.copy_samples for bucket, curve in restored.items()
            },
        }
        self._profile_max_sample_bytes = {
            "sm": _profile_int(maximum_bytes["sm"], "host-mover SM maximum bytes"),
            "copy_engine": _profile_int(
                maximum_bytes["copy_engine"], "host-mover copy maximum bytes"
            ),
        }
        self._publish_service_stats()
        return sum(
            curve.sm_samples + curve.copy_samples + curve.overlap_samples
            for curve in restored.values()
        )

    def plan(
        self,
        pending: Any,
        row_bytes_by_layer: tuple[tuple[int, int], ...],
        transfer_count: int,
        *,
        layer_service_key: tuple[str, int, int] | None,
        layer_curve: AcquisitionServiceCurve | None,
        collect_layer_profiles: Callable[[], None],
    ) -> HostMoverLeasePlan:
        cached = pending.mover_plan
        if cached is not None:
            if cached.row_count != transfer_count:
                raise RuntimeError("HiCache mover map changed during a lease")
            return cached

        self.collect_profiles()
        collect_layer_profiles()
        device_index_map = pending.materialize_device_index_map()
        moved_source_indices = device_index_map.source_indices
        moved_staging_indices = device_index_map.destination_indices
        bytes_per_transferred_row = sum(
            key_bytes + value_bytes for key_bytes, value_bytes in row_bytes_by_layer
        )
        representative_sm_wave_bytes = self.representative_wave_bytes(
            row_bytes_by_layer, transfer_count
        )
        service_model = self.service_model(representative_sm_wave_bytes)
        self._stats["layer_service_last_plan_key"] = (
            None if layer_service_key is None else list(layer_service_key)
        )
        self._stats["layer_service_last_plan_samples"] = (
            0 if layer_curve is None else len(layer_curve.samples_ns)
        )
        if layer_service_key is None:
            self._add_stat("layer_service_plan_key_missing_batches")
        elif layer_curve is None:
            self._add_stat("layer_service_plan_curve_missing_batches")
        elif not layer_curve.calibrated:
            self._add_stat("layer_service_plan_curve_uncalibrated_batches")
        else:
            self._add_stat("layer_service_plan_curve_calibrated_batches")
        overlap_compute_ns = (
            0
            if layer_curve is None
            else layer_curve.overlap_budget_ns(max(0, len(row_bytes_by_layer) - 1))
        )
        self._add_stat("host_mover_overlap_compute_ns", overlap_compute_ns)
        self._stats["layer_service_conservative_ns"] = (
            0 if layer_curve is None else layer_curve.conservative_interval_ns
        )
        auto_analysis = self._policy == "auto"
        if (
            self._policy == "copy_engine" or auto_analysis
        ) and self._copy_engine_max_operations < 2:
            raise RuntimeError(
                "copy-engine mover needs at least two K/V operations per layer"
            )

        tensor_plan = None
        layout_cpu_ns = 0
        planner_policy = self._policy
        calibration_probe_sm = False
        layout_free_insufficient_gain = False
        execution_context_unbound = False
        if self._policy == "auto":
            if not self.scale_calibrated("sm", representative_sm_wave_bytes):
                planner_policy = "sm"
                calibration_probe_sm = True
            elif not service_model.copy_calibrated and not any(
                profile.engine == "copy_engine" for profile in self._profiles
            ):
                planner_policy = "probe_copy"
            elif layer_curve is None or not layer_curve.calibrated:
                # Dense HiCache transport may start before SGLang binds the
                # eventual ForwardBatch. An isolated copy-engine curve cannot
                # predict interference from resident numerical work that has
                # not been named yet. Keep the safe SM issuer unless a
                # same-shape compute curve closes that context boundary.
                planner_policy = "sm"
                execution_context_unbound = True
            elif not service_model.ideal_copy_can_qualify(
                total_rows=transfer_count,
                row_bytes=bytes_per_transferred_row,
                copy_operations_per_run=2 * len(row_bytes_by_layer),
                overlap_compute_ns=overlap_compute_ns,
                service_scale_bytes=representative_sm_wave_bytes,
            ):
                planner_policy = "sm"
                layout_free_insufficient_gain = True
        analyze_layout = mover_layout_required(
            planner_policy, self._profile_index_layout
        )
        if analyze_layout:
            layout_started = time.perf_counter_ns()
            tensor_plan = plan_indexed_tensor_mover(
                moved_source_indices,
                moved_staging_indices,
                row_bytes=bytes_per_transferred_row,
                copy_operations_per_run=2 * len(row_bytes_by_layer),
                maximum_copy_runs=self._copy_engine_max_operations // 2,
                service_model=service_model,
                policy=planner_policy,
                overlap_compute_ns=overlap_compute_ns,
                service_scale_bytes=representative_sm_wave_bytes,
                validate_unique_destinations=(
                    self._profile_index_layout or self._verify_index_map
                ),
                capture_full_layout=self._profile_index_layout,
            )
            layout_cpu_ns = time.perf_counter_ns() - layout_started
            if self._policy != "sm":
                self._add_stat("copy_engine_layout_cpu_ns", layout_cpu_ns)

        if tensor_plan is None:
            predicted_sm_ns = service_model.candidate_ns(
                total_rows=transfer_count,
                copy_rows=0,
                copy_run_count=0,
                row_bytes=bytes_per_transferred_row,
                copy_operations_per_run=2 * len(row_bytes_by_layer),
                overlap_compute_ns=overlap_compute_ns,
            )
            if predicted_sm_ns is None:  # pragma: no cover - positive rows
                raise RuntimeError("SM mover produced no service estimate")
            plan = HostMoverLeasePlan(
                transfer_count,
                "sm",
                (),
                moved_source_indices,
                moved_staging_indices,
                None,
                layout_cpu_ns,
                predicted_sm_ns,
                predicted_sm_ns,
                (
                    "calibration_probe_sm"
                    if calibration_probe_sm
                    else "execution_context_unbound"
                    if execution_context_unbound
                    else "insufficient_gain"
                    if layout_free_insufficient_gain
                    else "forced_sm"
                    if self._policy == "sm"
                    else "uncalibrated_copy_engine"
                ),
            )
        else:
            selection_reason = (
                "calibration_probe_sm"
                if calibration_probe_sm
                else tensor_plan.selection_reason
            )
            plan = HostMoverLeasePlan(
                transfer_count,
                tensor_plan.kind,
                tensor_plan.copy_runs,
                tensor_plan.sm_source_indices,
                tensor_plan.sm_destination_indices,
                tensor_plan.layout,
                layout_cpu_ns,
                tensor_plan.predicted_sm_ns,
                tensor_plan.predicted_selected_ns,
                selection_reason,
            )
        pending.mover_plan = plan
        pending.prefetch_tensors = plan.retained_tensors

        self._add_stat(f"prefetch_mover_plan_{plan.kind}_leases")
        self._add_stat("prefetch_mover_plan_copy_runs", len(plan.copy_runs))
        self._add_stat("prefetch_mover_plan_copy_rows", plan.copy_row_count)
        self._add_stat("prefetch_mover_plan_sm_rows", plan.sm_row_count)
        self._add_stat("host_mover_predicted_sm_ns", plan.predicted_sm_ns)
        self._add_stat(
            "host_mover_predicted_selected_ns", plan.predicted_selected_ns or 0
        )
        reason_counter = {
            "calibration_probe_sm": "prefetch_mover_plan_calibration_probe_sm_leases",
            "calibration_probe_copy": (
                "prefetch_mover_plan_calibration_probe_copy_leases"
            ),
            "uncalibrated_copy_engine": (
                "prefetch_mover_plan_uncalibrated_copy_engine_leases"
            ),
            "insufficient_gain": "prefetch_mover_plan_insufficient_gain_leases",
            "service_cost": "prefetch_mover_plan_service_cost_leases",
            "execution_context_unbound": (
                "prefetch_mover_plan_execution_context_unbound_leases"
            ),
        }.get(plan.selection_reason)
        if reason_counter is not None:
            self._add_stat(reason_counter)
        if self._profile_index_layout:
            layout = plan.layout
            if layout is None:
                raise RuntimeError("indexed-layout profiling produced no layout")
            samples = self._stats.setdefault("indexed_layout_run_rows_samples", [])
            if len(samples) < 16:
                samples.append([run.row_count for run in layout.runs])
            eligible_rows = layout.eligible_rows(
                row_bytes=bytes_per_transferred_row,
                minimum_copy_bytes=self._profile_index_min_bytes,
            )
            candidate_bytes = eligible_rows * bytes_per_transferred_row
            self._add_stat("indexed_layout_profiles")
            self._add_stat("indexed_layout_rows", layout.row_count)
            self._add_stat("indexed_layout_runs", len(layout.runs))
            self._add_stat("indexed_layout_eligible_rows", eligible_rows)
            self._add_stat("indexed_layout_candidate_bytes", candidate_bytes)
            self._stats["indexed_layout_maximum_run_rows"] = max(
                self._stats.get("indexed_layout_maximum_run_rows", 0),
                layout.maximum_run_rows,
            )
            self._add_stat("indexed_layout_profile_cpu_ns", layout_cpu_ns)
        return plan

    def collect_profiles(self) -> None:
        pending: list[MoverProfile] = []
        completed: list[tuple[MoverProfile, float, int]] = []
        for profile in self._profiles:
            if not profile.finish.query():
                pending.append(profile)
                continue
            milliseconds = profile.start.elapsed_time(profile.finish)
            elapsed_ns = max(1, round(milliseconds * 1_000_000.0))
            self._profile_max_sample_bytes[profile.engine] = max(
                self._profile_max_sample_bytes[profile.engine],
                profile.service_scale_bytes,
            )
            profile_label = "sm" if profile.engine == "sm" else "copy"
            prefix = f"host_mover_profiled_{profile_label}"
            self._add_stat(f"{prefix}_bytes", profile.transfer_bytes)
            self._add_stat(f"{prefix}_gpu_ms", milliseconds)
            completed.append((profile, milliseconds, elapsed_ns))

        self._profiles = pending

        def previous_curve(
            engine: str, service_scale_bytes: int
        ) -> tuple[int, IndexedMoverServiceModel]:
            bucket = byte_scale_bucket(service_scale_bytes)
            previous = self._service_models.get(bucket)
            if previous is None:
                configured = self._default_service_model
                previous = (
                    configured
                    if configured.supports_transfer_scale(service_scale_bytes)
                    else self._uncalibrated_prior()
                )
            return bucket, previous

        def commit_curve(
            engine: str,
            bucket: int,
            previous_samples: int,
            updated: IndexedMoverServiceModel,
            sample_count: int,
        ) -> None:
            self._service_models[bucket] = updated
            updated_samples = (
                updated.sm_samples if engine == "sm" else updated.copy_samples
            )
            if updated_samples > previous_samples:
                samples = self._profile_buckets[engine]
                samples[bucket] = samples.get(bucket, 0) + sample_count

        calibration_groups: dict[
            tuple[str, int], list[tuple[MoverProfile, int]]
        ] = {}
        for profile, _milliseconds, elapsed_ns in completed:
            if profile.calibration:
                key = (profile.engine, byte_scale_bucket(profile.service_scale_bytes))
                calibration_groups.setdefault(key, []).append((profile, elapsed_ns))
                continue
            bucket, previous = previous_curve(
                profile.engine, profile.service_scale_bytes
            )
            if profile.engine == "sm":
                previous_samples = previous.sm_samples
                updated = previous.with_sm_observation(
                    transfer_bytes=profile.transfer_bytes,
                    service_scale_bytes=profile.service_scale_bytes,
                    elapsed_ns=elapsed_ns,
                    alpha=0.25,
                )
            else:
                previous_samples = previous.copy_samples
                updated = previous.with_copy_observation(
                    transfer_bytes=profile.transfer_bytes,
                    service_scale_bytes=profile.service_scale_bytes,
                    elapsed_ns=elapsed_ns,
                    operation_count=profile.operation_count,
                    issue_cpu_ns=profile.issue_cpu_ns,
                    alpha=0.25,
                )
            commit_curve(
                profile.engine, bucket, previous_samples, updated, 1
            )

        for (engine, bucket), observations in calibration_groups.items():
            # Collapse one complete probe frontier to its aggregate service
            # rate. This weights short final waves correctly and prevents an
            # unusually fast first wave from authorizing a slower issuer.
            total_bytes = sum(profile.transfer_bytes for profile, _ in observations)
            total_elapsed_ns = sum(elapsed_ns for _, elapsed_ns in observations)
            service_scale_bytes = max(
                profile.service_scale_bytes for profile, _ in observations
            )
            equivalent_elapsed_ns = max(
                1,
                round(
                    service_scale_bytes * total_elapsed_ns / total_bytes
                ),
            )
            _resolved_bucket, previous = previous_curve(engine, service_scale_bytes)
            if _resolved_bucket != bucket:  # pragma: no cover - grouping invariant
                raise RuntimeError("mover calibration bucket changed")
            sample_count = len(observations)
            if engine == "sm":
                previous_samples = previous.sm_samples
                updated = previous.with_sm_observation(
                    transfer_bytes=service_scale_bytes,
                    service_scale_bytes=service_scale_bytes,
                    elapsed_ns=equivalent_elapsed_ns,
                    alpha=1.0,
                )
                if updated.sm_samples == previous_samples:
                    # The model deliberately rejects timer-dominated transfer
                    # scales. A complete frontier may still consist entirely
                    # of such small waves; do not manufacture calibration
                    # samples for an observation the model did not accept.
                    continue
                if updated.sm_samples != previous_samples + 1:
                    raise RuntimeError("SM mover observation changed sample weight")
                updated = replace(
                    updated, sm_samples=previous_samples + sample_count
                )
            else:
                previous_samples = previous.copy_samples
                updated = previous.with_copy_observation(
                    transfer_bytes=service_scale_bytes,
                    service_scale_bytes=service_scale_bytes,
                    elapsed_ns=equivalent_elapsed_ns,
                    operation_count=sum(
                        profile.operation_count for profile, _ in observations
                    ),
                    issue_cpu_ns=sum(
                        profile.issue_cpu_ns for profile, _ in observations
                    ),
                    alpha=1.0,
                )
                if updated.copy_samples == previous_samples:
                    # In particular, retaining a nonzero sample count without
                    # a bandwidth/operation estimate violates the typed model
                    # and used to abort auto calibration at small wave scales.
                    continue
                if updated.copy_samples != previous_samples + 1:
                    raise RuntimeError("copy mover observation changed sample weight")
                updated = replace(
                    updated, copy_samples=previous_samples + sample_count
                )
            commit_curve(
                engine,
                bucket,
                previous_samples,
                updated,
                sample_count,
            )
            self._add_stat("host_mover_complete_calibration_frontiers")
            self._add_stat(
                "host_mover_complete_calibration_wave_samples", sample_count
            )

        self._publish_service_stats()

    def _publish_service_stats(self) -> None:
        model = (
            self._default_service_model
            if self._last_service_bucket is None
            else self._service_models.get(
                self._last_service_bucket, self._default_service_model
            )
        )
        self._stats["host_mover_sm_samples"] = model.sm_samples
        self._stats["host_mover_copy_samples"] = model.copy_samples
        self._stats["host_mover_copy_calibrated"] = model.copy_calibrated
        self._stats["host_mover_sm_bandwidth_bps"] = model.sm_bandwidth_bytes_per_second
        self._stats["host_mover_copy_bandwidth_bps"] = (
            model.copy_bandwidth_bytes_per_second
        )
        self._stats["host_mover_copy_operation_ns"] = model.copy_operation_ns
        for engine, label in (("sm", "sm"), ("copy_engine", "copy")):
            buckets = self._profile_buckets[engine]
            self._stats[f"host_mover_{label}_calibrated_buckets"] = sum(
                count >= self._calibration_samples for count in buckets.values()
            )
            self._stats[f"host_mover_{label}_max_sample_bytes"] = max(
                self._profile_max_sample_bytes[engine],
                max(
                    (
                        profile.service_scale_bytes
                        for profile in self._profiles
                        if profile.engine == engine
                    ),
                    default=0,
                ),
            )
        self._stats["host_mover_service_curves"] = [
            {
                "scale_bucket": bucket,
                "minimum_bytes": 1 << bucket,
                "maximum_bytes": (1 << (bucket + 1)) - 1,
                "sm_bandwidth_bps": curve.sm_bandwidth_bytes_per_second,
                "copy_bandwidth_bps": curve.copy_bandwidth_bytes_per_second,
                "copy_operation_ns": curve.copy_operation_ns,
                "sm_samples": curve.sm_samples,
                "copy_samples": curve.copy_samples,
            }
            for bucket, curve in sorted(self._service_models.items())
        ]


@dataclass(frozen=True)
class HostTransferLayer:
    """Immutable K/V transport descriptors for one local model layer."""

    key_bytes: int
    value_bytes: int
    indexed_objects: tuple[IndexedHostObject, ...]
    copy_groups: tuple[StridedCopyGroup, ...]
    wave_row_ends: tuple[int, ...]

    def __post_init__(self) -> None:
        if min(self.key_bytes, self.value_bytes) <= 0:
            raise ValueError("host transfer layer byte geometry must be positive")
        if len(self.indexed_objects) != 2 * len(self.wave_row_ends):
            raise ValueError("host transfer layer SM waves are incomplete")
        if self.wave_row_ends and (
            tuple(sorted(set(self.wave_row_ends))) != self.wave_row_ends
            or self.wave_row_ends[0] <= 0
        ):
            raise ValueError("host transfer layer wave boundaries are invalid")
        if len(self.copy_groups) not in {0, 2}:
            raise ValueError("host transfer layer must own zero or two copy groups")


@dataclass(frozen=True)
class HostTransferLeasePlan:
    """Lease-scoped descriptors reused by every acquisition frontier."""

    mover: HostMoverLeasePlan
    layers: tuple[HostTransferLayer, ...]
    paired_indexed_copy: bool

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("host transfer lease plan has no model layers")
        needs_sm = self.mover.sm_row_count != 0
        needs_copy = bool(self.mover.copy_runs)
        if any(bool(layer.indexed_objects) != needs_sm for layer in self.layers):
            raise ValueError("host transfer lease SM descriptors are incomplete")
        if any(bool(layer.copy_groups) != needs_copy for layer in self.layers):
            raise ValueError("host transfer lease copy descriptors are incomplete")
        wave_boundaries = {layer.wave_row_ends for layer in self.layers}
        if len(wave_boundaries) != 1:
            raise ValueError("host transfer lease layers disagree on wave geometry")
        boundaries = next(iter(wave_boundaries))
        if needs_sm and (not boundaries or boundaries[-1] != self.mover.row_count):
            raise ValueError("host transfer lease waves do not cover the lease")
        if not needs_sm and boundaries:
            raise ValueError("copy-only host transfer retained SM waves")
        object_ids = tuple(
            item.object_id for layer in self.layers for item in layer.indexed_objects
        )
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("host transfer lease repeats an object identity")

    @property
    def layer_geometry(self) -> tuple[tuple[int, int], ...]:
        return tuple((layer.key_bytes, layer.value_bytes) for layer in self.layers)

    @property
    def indexed_objects(self) -> tuple[IndexedHostObject, ...]:
        return tuple(item for layer in self.layers for item in layer.indexed_objects)

    @property
    def copy_groups(self) -> tuple[tuple[StridedCopyGroup, ...], ...]:
        return tuple(layer.copy_groups for layer in self.layers)

    @property
    def sm_waves_per_layer(self) -> int:
        return len(self.layers[0].wave_row_ends)

    @property
    def objects_per_layer(self) -> int:
        return 2 * self.sm_waves_per_layer


def sm_acquisition_wave_count(
    mover: HostMoverLeasePlan, maximum_waves: int
) -> int:
    """Return the exact completion-wave count supported by one mover plan."""

    if maximum_waves <= 0:
        raise ValueError("SM acquisition wave bound must be positive")
    if mover.sm_row_count == 0:
        return 0
    # A hybrid plan has disjoint SM/copy row orders. Until both engines share
    # one indexed completion map, only their joined layer event is sound.
    return 1 if mover.copy_runs else min(maximum_waves, mover.sm_row_count)


def build_host_transfer_lease_plan(
    controller: object,
    mover: HostMoverLeasePlan,
    row_bytes_by_layer: tuple[tuple[int, int], ...],
    *,
    object_id_bases: tuple[int, ...],
    object_version: int,
    sm_acquisition_waves: int,
) -> HostTransferLeasePlan:
    """Materialize stable K/V descriptors exactly once for one HiCache lease."""

    layer_count = int(getattr(controller, "layer_num"))
    if (
        layer_count <= 0
        or len(row_bytes_by_layer) != layer_count
        or len(object_id_bases) != layer_count
        or object_version <= 0
        or sm_acquisition_waves <= 0
    ):
        raise ValueError("host transfer lease layer geometry is invalid")
    transfer_count = mover.row_count
    device_pool = getattr(controller, "mem_pool_device")
    host_pool = getattr(controller, "mem_pool_host")
    host_keys = tuple(host_pool.k_data_refs)
    host_values = tuple(host_pool.v_data_refs)
    if len(host_keys) != layer_count or len(host_values) != layer_count:
        raise RuntimeError("HiCache host K/V layer geometry is incomplete")

    use_sm = mover.sm_row_count != 0
    use_copy = bool(mover.copy_runs)
    source_indices = mover.sm_source_indices
    destination_indices = mover.sm_destination_indices
    if use_sm and (
        int(source_indices.numel()) != mover.sm_row_count
        or int(destination_indices.numel()) != mover.sm_row_count
    ):
        raise RuntimeError("SM mover index geometry changed during lease planning")
    # Sub-layer completion is enabled only when one SM index stream owns the
    # complete lease. A hybrid/copy wave has a different row order per engine;
    # it therefore retains one joined layer event until a shared ordering proof
    # is available rather than publishing optimistic partial readiness.
    wave_count = sm_acquisition_wave_count(mover, sm_acquisition_waves)
    wave_row_ends: tuple[int, ...] = ()
    if wave_count:
        covered_rows = mover.row_count
        quotient, remainder = divmod(covered_rows, wave_count)
        cursor = 0
        ends: list[int] = []
        for wave in range(wave_count):
            cursor += quotient + int(wave < remainder)
            ends.append(cursor)
        wave_row_ends = tuple(ends)

    start_layer = int(getattr(device_pool, "start_layer", 0))
    paired_indexed_copy = True
    layers: list[HostTransferLayer] = []
    for local_layer, (host_key, host_value) in enumerate(
        zip(host_keys, host_values, strict=True)
    ):
        layer_id = start_layer + local_layer
        key_cache = device_pool._get_key_buffer(layer_id)
        value_cache = device_pool._get_value_buffer(layer_id)
        if host_key.dtype != key_cache.dtype or host_value.dtype != value_cache.dtype:
            raise RuntimeError("HiCache host and device KV dtypes disagree")

        key_element_bytes = int(key_cache[0].numel()) * key_cache.element_size()
        value_element_bytes = int(value_cache[0].numel()) * value_cache.element_size()
        if (key_element_bytes, value_element_bytes) != row_bytes_by_layer[local_layer]:
            raise RuntimeError("HiCache host and device KV row geometry disagrees")
        key_source_stride = host_key.stride(0) * host_key.element_size()
        value_source_stride = host_value.stride(0) * host_value.element_size()
        key_destination_stride = key_cache.stride(0) * key_cache.element_size()
        value_destination_stride = value_cache.stride(0) * value_cache.element_size()
        paired_indexed_copy &= (
            key_element_bytes == value_element_bytes
            and key_element_bytes in {128, 256, 512, 1024, 2048}
        )
        key_bytes = transfer_count * key_element_bytes
        value_bytes = transfer_count * value_element_bytes
        if max(key_bytes, value_bytes) >= 1 << 32:
            raise RuntimeError("HiCache layer transfer exceeds the NTA ABI limit")

        indexed_objects: tuple[IndexedHostObject, ...] = ()
        if use_sm:
            object_id_base = object_id_bases[local_layer]
            wave_objects: list[IndexedHostObject] = []
            begin = 0
            for wave, row_end in enumerate(wave_row_ends):
                # Hybrid execution has one joined event and one SM descriptor
                # covering only its disjoint SM partition. Pure SM waves map
                # directly onto the complete lease index vector.
                row_count = (
                    mover.sm_row_count
                    if use_copy
                    else row_end - begin
                )
                index_offset = 0 if use_copy else begin * 4
                wave_objects.extend(
                    (
                        IndexedHostObject(
                            object_id_base + 2 * wave,
                            object_version,
                            host_key.data_ptr(),
                            key_cache.data_ptr(),
                            source_indices.data_ptr() + index_offset,
                            destination_indices.data_ptr() + index_offset,
                            row_count,
                            key_element_bytes,
                            key_source_stride,
                            key_destination_stride,
                            int(host_key.shape[0]),
                            int(key_cache.shape[0]),
                        ),
                        IndexedHostObject(
                            object_id_base + 2 * wave + 1,
                            object_version,
                            host_value.data_ptr(),
                            value_cache.data_ptr(),
                            source_indices.data_ptr() + index_offset,
                            destination_indices.data_ptr() + index_offset,
                            row_count,
                            value_element_bytes,
                            value_source_stride,
                            value_destination_stride,
                            int(host_value.shape[0]),
                            int(value_cache.shape[0]),
                        ),
                    )
                )
                begin = row_end
            indexed_objects = tuple(wave_objects)

        copy_groups: tuple[StridedCopyGroup, ...] = ()
        if use_copy:
            copy_groups = (
                StridedCopyGroup(
                    host_key.data_ptr(),
                    key_cache.data_ptr(),
                    int(host_key.shape[0]),
                    int(key_cache.shape[0]),
                    key_element_bytes,
                    key_source_stride,
                    key_destination_stride,
                ),
                StridedCopyGroup(
                    host_value.data_ptr(),
                    value_cache.data_ptr(),
                    int(host_value.shape[0]),
                    int(value_cache.shape[0]),
                    value_element_bytes,
                    value_source_stride,
                    value_destination_stride,
                ),
            )
        layers.append(
            HostTransferLayer(
                key_bytes,
                value_bytes,
                indexed_objects,
                copy_groups,
                wave_row_ends,
            )
        )
    return HostTransferLeasePlan(mover, tuple(layers), paired_indexed_copy)
