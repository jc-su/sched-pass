"""Bounded deployment calibration for SGLang layer-arrival service.

This owner contains only observations that may influence future scheduling.
It does not own request identity, transfer submission, attention dispatch, or
artifact publication.  CUDA events remain pending until their elapsed time is
queryable; completed samples update one conservative curve per stable forward
shape without synchronizing the serving stream.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import torch

from nta_runtime.acquisition_scheduler import AcquisitionServiceCurve
from nta_runtime.engines.sglang_acquisition_contract import HostArrivalProfileKey
from nta_runtime.engines.sglang_planning import byte_scale_bucket
from nta_runtime.engines.sglang_state import SglangForwardEpoch


LayerServiceKey = tuple[str, int, int]
_CALIBRATION_STATE_SCHEMA = 1


def _state_mapping(value: Any, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{owner} calibration state must be a string-keyed object")
    return value


def _state_keys(
    value: Mapping[str, Any], expected: frozenset[str], owner: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{owner} calibration state fields disagree "
            f"(missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)})"
        )


def _state_int(
    value: Any,
    owner: str,
    *,
    minimum: int = 0,
    maximum: int = (1 << 63) - 1,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{owner} calibration value must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{owner} calibration value is outside its bound")
    return value


def _state_samples(
    value: Any,
    owner: str,
    *,
    maximum_samples: int,
    signed: bool = False,
    allow_zero: bool = False,
) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) > maximum_samples:
        raise ValueError(f"{owner} calibration samples exceed their bound")
    minimum = -(1 << 63) + 1 if signed else 0 if allow_zero else 1
    return tuple(
        _state_int(sample, owner, minimum=minimum) for sample in value
    )


def _state_list(value: Any, owner: str, *, maximum_items: int) -> list[Any]:
    if not isinstance(value, list) or len(value) > maximum_items:
        raise ValueError(f"{owner} calibration rows exceed their bound")
    return value


def _arrival_key_state(key: HostArrivalProfileKey) -> dict[str, int | str]:
    return {
        "phase": key.phase,
        "query_rows_bucket": key.query_rows_bucket,
        "batch_size_bucket": key.batch_size_bucket,
        "transfer_rows_bucket": key.transfer_rows_bucket,
        "transfer_bytes_bucket": key.transfer_bytes_bucket,
        "mover_kind": key.mover_kind,
        "layers_per_submission": key.layers_per_submission,
        "sm_waves_per_layer": key.sm_waves_per_layer,
    }


def _arrival_key_from_state(value: Any) -> HostArrivalProfileKey:
    state = _state_mapping(value, "consumer-policy key")
    fields = frozenset(
        {
            "phase",
            "query_rows_bucket",
            "batch_size_bucket",
            "transfer_rows_bucket",
            "transfer_bytes_bucket",
            "mover_kind",
            "layers_per_submission",
            "sm_waves_per_layer",
        }
    )
    _state_keys(state, fields, "consumer-policy key")
    phase = state["phase"]
    mover_kind = state["mover_kind"]
    if not isinstance(phase, str) or not isinstance(mover_kind, str):
        raise ValueError("consumer-policy key strings are invalid")
    return HostArrivalProfileKey(
        phase=phase,
        query_rows_bucket=_state_int(
            state["query_rows_bucket"], "query-row bucket"
        ),
        batch_size_bucket=_state_int(
            state["batch_size_bucket"], "batch-size bucket"
        ),
        transfer_rows_bucket=_state_int(
            state["transfer_rows_bucket"], "transfer-row bucket"
        ),
        transfer_bytes_bucket=_state_int(
            state["transfer_bytes_bucket"], "transfer-byte bucket"
        ),
        mover_kind=mover_kind,
        layers_per_submission=_state_int(
            state["layers_per_submission"],
            "layers per submission",
            minimum=1,
        ),
        sm_waves_per_layer=_state_int(
            state["sm_waves_per_layer"], "SM waves per layer", minimum=1
        ),
    )


@dataclass(frozen=True, slots=True)
class _LayerServiceProfile:
    """One adjacent attention-arrival interval for a stable forward shape."""

    start: torch.cuda.Event
    finish: torch.cuda.Event
    key: LayerServiceKey

    def __post_init__(self) -> None:
        phase, query_rows, batch_size = self.key
        if phase not in {"decode", "extend"} or min(query_rows, batch_size) <= 0:
            raise ValueError("layer service profile has an invalid shape key")


class SglangLayerServiceCalibration:
    """Own bounded per-shape compute-service observations and CUDA events."""

    def __init__(
        self,
        *,
        enabled: bool,
        minimum_samples: int,
        maximum_samples: int,
        model_start_layer: int,
        model_layer_count: int,
        stats: dict[str, Any],
        frozen: bool = False,
    ) -> None:
        if (
            minimum_samples <= 0
            or maximum_samples < minimum_samples
            or model_start_layer < 0
            or model_layer_count <= 0
        ):
            raise ValueError("SGLang layer-service calibration geometry is invalid")
        self._enabled = bool(enabled)
        self._frozen = bool(frozen)
        self._minimum_samples = minimum_samples
        self._maximum_samples = maximum_samples
        self._model_start_layer = model_start_layer
        self._model_layer_count = model_layer_count
        self._stats = stats
        self._curves: dict[LayerServiceKey, AcquisitionServiceCurve] = {}
        self._profiles: list[_LayerServiceProfile] = []
        self._inflight_by_key: dict[LayerServiceKey, int] = {}

    @property
    def pending_count(self) -> int:
        return len(self._profiles)

    @staticmethod
    def shape_key(batch: Any) -> LayerServiceKey | None:
        """Resolve the same extend key at scheduler and ForwardBatch seams."""

        requests = tuple(getattr(batch, "reqs", ()) or ())
        query_rows = getattr(batch, "extend_num_tokens", None)
        batch_size = len(requests)
        if batch_size == 0:
            batch_size = int(getattr(batch, "batch_size", 0) or 0)
        if batch_size == 0:
            batch_size = len(tuple(getattr(batch, "rids", ()) or ()))
        if batch_size <= 0 or query_rows is None or int(query_rows) <= 0:
            return None
        return ("extend", int(query_rows), batch_size)

    def curve(
        self, key: LayerServiceKey | None, *, calibrated_only: bool = False
    ) -> AcquisitionServiceCurve | None:
        if key is None:
            return None
        curve = self._curves.get(key)
        if calibrated_only and (curve is None or not curve.calibrated):
            return None
        return curve

    def curve_for_batch(self, batch: Any) -> AcquisitionServiceCurve | None:
        return self.curve(self.shape_key(batch), calibrated_only=True)

    def collect(self) -> None:
        """Retire completed observations without synchronizing CUDA.

        Profiles are removed immediately after their curve update. If a later
        event fails, already committed samples cannot be counted a second time
        on retry; the failing profile and untouched suffix remain pending.
        """

        pending: list[_LayerServiceProfile] = []
        profiles = self._profiles
        for index, profile in enumerate(profiles):
            if not profile.finish.query():
                pending.append(profile)
                continue
            try:
                elapsed_ns = max(
                    1,
                    round(profile.start.elapsed_time(profile.finish) * 1_000_000.0),
                )
                curve = self._curves.get(
                    profile.key,
                    AcquisitionServiceCurve(
                        minimum_samples=self._minimum_samples,
                        maximum_samples=self._maximum_samples,
                    ),
                ).with_observation(elapsed_ns)
            except BaseException:
                pending.extend(profiles[index:])
                self._profiles = pending
                self._rebuild_inflight_counts()
                raise
            self._curves[profile.key] = curve
            self._stats["layer_service_profiled_intervals"] += 1
        self._profiles = pending
        self._rebuild_inflight_counts()
        self._stats["layer_service_calibrated_shapes"] = sum(
            curve.calibrated for curve in self._curves.values()
        )

    def record(
        self,
        *,
        batch: SglangForwardEpoch | None,
        phase: str,
        query: torch.Tensor,
        global_layer: int,
    ) -> None:
        """Record one adjacent attention arrival for a Host-backed forward."""

        if (
            not self._enabled
            or self._frozen
            or batch is None
            or batch.pending_host_load is None
        ):
            return
        query_rows = int(query.shape[0])
        key = (phase, query_rows, len(batch.bindings))
        if (
            phase not in {"decode", "extend"}
            or min(query_rows, len(batch.bindings)) <= 0
        ):
            raise RuntimeError("layer service calibration has an invalid forward")
        if batch.layer_service_key is not None and batch.layer_service_key != key:
            raise RuntimeError("attention shape changed within one model forward")
        batch.layer_service_key = key

        curve = self._curves.get(
            key,
            AcquisitionServiceCurve(
                minimum_samples=self._minimum_samples,
                maximum_samples=self._maximum_samples,
            ),
        )
        if (
            len(curve.samples_ns) + self._inflight_by_key.get(key, 0)
            >= curve.maximum_samples
        ):
            batch.layer_arrival_event = None
            batch.layer_arrival_local_layer = -1
            return

        local_layer = int(global_layer) - self._model_start_layer
        if not 0 <= local_layer < self._model_layer_count:
            raise RuntimeError("attention layer is outside the local model range")
        arrival = torch.cuda.Event(enable_timing=True)
        arrival.record(torch.cuda.current_stream())
        previous = batch.layer_arrival_event
        if previous is not None:
            if batch.layer_arrival_local_layer + 1 != local_layer:
                raise RuntimeError("attention layers did not arrive in model order")
            self._profiles.append(_LayerServiceProfile(previous, arrival, key))
            self._inflight_by_key[key] = self._inflight_by_key.get(key, 0) + 1
        batch.layer_arrival_event = arrival
        batch.layer_arrival_local_layer = local_layer

    def report(self) -> list[dict[str, int | str]]:
        return [
            {
                "phase": key[0],
                "query_rows": key[1],
                "batch_size": key[2],
                "samples": len(curve.samples_ns),
                "conservative_interval_ns": curve.conservative_interval_ns,
            }
            for key, curve in sorted(self._curves.items())
        ]

    def export_state(self) -> dict[str, Any]:
        """Return pointer-free bounded state suitable for deployment reuse."""

        if self._profiles:
            raise RuntimeError(
                "layer-service calibration cannot snapshot pending CUDA events"
            )
        return {
            "schema": _CALIBRATION_STATE_SCHEMA,
            "model_start_layer": self._model_start_layer,
            "model_layer_count": self._model_layer_count,
            "minimum_samples": self._minimum_samples,
            "maximum_samples": self._maximum_samples,
            "curves": [
                {
                    "phase": key[0],
                    "query_rows": key[1],
                    "batch_size": key[2],
                    "samples_ns": list(curve.samples_ns),
                }
                for key, curve in sorted(self._curves.items())
            ],
        }

    def import_state(self, value: Any) -> int:
        """Restore a complete state transaction before the first forward."""

        if self._profiles or self._curves:
            raise RuntimeError("layer-service calibration is not empty")
        state = _state_mapping(value, "layer-service")
        _state_keys(
            state,
            frozenset(
                {
                    "schema",
                    "model_start_layer",
                    "model_layer_count",
                    "minimum_samples",
                    "maximum_samples",
                    "curves",
                }
            ),
            "layer-service",
        )
        expected = (
            _CALIBRATION_STATE_SCHEMA,
            self._model_start_layer,
            self._model_layer_count,
            self._minimum_samples,
            self._maximum_samples,
        )
        actual = tuple(
            _state_int(state[name], f"layer-service {name}")
            for name in (
                "schema",
                "model_start_layer",
                "model_layer_count",
                "minimum_samples",
                "maximum_samples",
            )
        )
        if actual != expected:
            raise ValueError("layer-service calibration geometry is incompatible")
        rows = state["curves"]
        if not isinstance(rows, list):
            raise ValueError("layer-service curves must be a list")
        restored: dict[LayerServiceKey, AcquisitionServiceCurve] = {}
        for raw in rows:
            row = _state_mapping(raw, "layer-service curve")
            _state_keys(
                row,
                frozenset(
                    {"phase", "query_rows", "batch_size", "samples_ns"}
                ),
                "layer-service curve",
            )
            phase = row["phase"]
            if not isinstance(phase, str) or phase not in {"decode", "extend"}:
                raise ValueError("layer-service curve has an invalid phase")
            key = (
                phase,
                _state_int(row["query_rows"], "layer query rows", minimum=1),
                _state_int(row["batch_size"], "layer batch size", minimum=1),
            )
            if key in restored:
                raise ValueError("layer-service profile repeats a shape")
            restored[key] = AcquisitionServiceCurve(
                samples_ns=_state_samples(
                    row["samples_ns"],
                    "layer-service",
                    maximum_samples=self._maximum_samples,
                ),
                minimum_samples=self._minimum_samples,
                maximum_samples=self._maximum_samples,
            )
        self._curves = restored
        self._stats["layer_service_calibrated_shapes"] = sum(
            curve.calibrated for curve in restored.values()
        )
        return sum(len(curve.samples_ns) for curve in restored.values())

    def _rebuild_inflight_counts(self) -> None:
        counts: dict[LayerServiceKey, int] = {}
        for profile in self._profiles:
            counts[profile.key] = counts.get(profile.key, 0) + 1
        self._inflight_by_key = counts


@dataclass(frozen=True, slots=True)
class _BoundedTimingCurve:
    """Finite deployment samples used by the conservative consumer policy."""

    samples_ns: tuple[int, ...] = ()
    minimum_samples: int = 2
    maximum_samples: int = 8

    def __post_init__(self) -> None:
        if self.minimum_samples <= 0 or self.maximum_samples < self.minimum_samples:
            raise ValueError("consumer-policy sample bounds are invalid")
        if len(self.samples_ns) > self.maximum_samples:
            raise ValueError("consumer-policy sample curve exceeds its bound")

    @property
    def calibrated(self) -> bool:
        return len(self.samples_ns) >= self.minimum_samples

    def with_observation(self, elapsed_ns: int) -> "_BoundedTimingCurve":
        return _BoundedTimingCurve(
            (*self.samples_ns, int(elapsed_ns))[-self.maximum_samples :],
            self.minimum_samples,
            self.maximum_samples,
        )


@dataclass(frozen=True, slots=True)
class _LayerArrivalProfile:
    arrival: torch.cuda.Event
    ready: torch.cuda.Event
    key: HostArrivalProfileKey
    local_layer: int


@dataclass(frozen=True, slots=True)
class _PartialConsumerProfile:
    start: torch.cuda.Event
    dispatch_ready: torch.cuda.Event
    finish: torch.cuda.Event
    key: HostArrivalProfileKey
    partition_prepared: bool


@dataclass(frozen=True, slots=True)
class _StockConsumerProfile:
    start: torch.cuda.Event
    finish: torch.cuda.Event
    key: HostArrivalProfileKey
    local_layer: int


class SglangConsumerPolicyCalibration:
    """Closed-loop, bounded selector for the optional partial consumer.

    The policy compares two directly measured quantities for one stable
    deployment shape: signed producer lateness at each GPU attention arrival,
    stock numerical service, and the complete partial-consumer critical path.
    The latter is split into a one-time direct/deferred partition preparation
    and forward-local partition reuse.  AUTO chooses a set of layers only when
    their aggregate conservative stock cost amortizes that fixed preparation
    plus every selected reuse by the configured gain factor. Unknown or noisy
    shapes therefore use the scheduled-preacquired stock specialization.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        model_start_layer: int,
        model_layer_count: int,
        minimum_samples: int = 2,
        maximum_samples: int = 8,
        maximum_probe_misses: int = 2,
        stats: dict[str, Any],
        frozen: bool = False,
    ) -> None:
        if (
            model_start_layer < 0
            or model_layer_count <= 0
            or minimum_samples <= 0
            or maximum_samples < minimum_samples
            or maximum_probe_misses <= 0
        ):
            raise ValueError("SGLang consumer-policy calibration is invalid")
        self._enabled = bool(enabled)
        self._frozen = bool(frozen)
        self._model_start_layer = model_start_layer
        self._model_layer_count = model_layer_count
        self._minimum_samples = minimum_samples
        self._maximum_samples = maximum_samples
        self._maximum_probe_misses = maximum_probe_misses
        self._stats = stats
        self._arrival_curves: dict[
            tuple[HostArrivalProfileKey, int], _BoundedTimingCurve
        ] = {}
        self._stock_curves: dict[
            tuple[HostArrivalProfileKey, int], _BoundedTimingCurve
        ] = {}
        # ``True`` keys are the first partial layer in a forward and include
        # one-time partition setup. ``False`` keys reuse that partition.
        self._partial_total_curves: dict[
            tuple[HostArrivalProfileKey, bool], _BoundedTimingCurve
        ] = {}
        self._partial_dispatch_curves: dict[
            tuple[HostArrivalProfileKey, bool], _BoundedTimingCurve
        ] = {}
        self._partial_device_curves: dict[
            tuple[HostArrivalProfileKey, bool], _BoundedTimingCurve
        ] = {}
        self._arrival_profiles: list[_LayerArrivalProfile] = []
        self._stock_profiles: list[_StockConsumerProfile] = []
        self._partial_profiles: list[_PartialConsumerProfile] = []
        self._arrival_inflight: dict[tuple[HostArrivalProfileKey, int], int] = {}
        self._stock_inflight: dict[tuple[HostArrivalProfileKey, int], int] = {}
        self._partial_inflight: dict[tuple[HostArrivalProfileKey, bool], int] = {}
        self._probe_misses: dict[HostArrivalProfileKey, int] = {}
        self._probe_attempts: dict[HostArrivalProfileKey, int] = {}
        self._minimum_gain_by_key: dict[HostArrivalProfileKey, float] = {}
        self._last_key: HostArrivalProfileKey | None = None

    @property
    def pending_count(self) -> int:
        return (
            len(self._arrival_profiles)
            + len(self._stock_profiles)
            + len(self._partial_profiles)
        )

    def bind_lease(
        self,
        pending: Any,
        *,
        layer_service_key: LayerServiceKey,
        mover_kind: str,
        layers_per_submission: int,
        sm_waves_per_layer: int,
        minimum_gain: float,
    ) -> HostArrivalProfileKey | None:
        """Bind profiling and a frozen progressive-layer plan before transport."""

        if not self._enabled:
            return None
        self.collect()
        phase, query_rows, batch_size = layer_service_key
        transfer_rows = int(pending.device_indices.numel())
        transfer_bytes = sum(int(value) for value in pending.layer_bytes)
        key = HostArrivalProfileKey(
            phase=phase,
            query_rows_bucket=byte_scale_bucket(query_rows),
            batch_size_bucket=byte_scale_bucket(batch_size),
            transfer_rows_bucket=byte_scale_bucket(transfer_rows),
            transfer_bytes_bucket=byte_scale_bucket(transfer_bytes),
            mover_kind=mover_kind,
            layers_per_submission=layers_per_submission,
            sm_waves_per_layer=max(1, sm_waves_per_layer),
        )
        self._last_key = key
        pending.arrival_profile_key = key
        pending.arrival_profile_active = False
        if self._frozen:
            calibrated = self.shape_closed(key)
            pending.arrival_profiling = False
            pending.consumer_policy_probe = False
            pending.planned_progressive_layers = (
                self.profitable_layers(key, minimum_gain=minimum_gain)
                if calibrated
                else frozenset()
            )
            counter = (
                "consumer_policy_frozen_profile_leases"
                if calibrated
                else "consumer_policy_frozen_conservative_leases"
            )
            self._stats[counter] = self._stats.get(counter, 0) + 1
            self._stats["consumer_policy_planned_layers"] += len(
                pending.planned_progressive_layers
            )
            return key
        self._minimum_gain_by_key[key] = minimum_gain
        pending.arrival_profiling = any(
            min(
                self._sample_count(
                    self._arrival_curves,
                    self._arrival_inflight,
                    (key, layer),
                ),
                self._sample_count(
                    self._stock_curves,
                    self._stock_inflight,
                    (key, layer),
                ),
            )
            < self._minimum_samples
            for layer in range(self._model_layer_count)
        )
        potentially_late = (
            not pending.arrival_profiling
            and any(
                curve.calibrated and min(curve.samples_ns) > 0
                for (curve_key, _layer), curve in self._arrival_curves.items()
                if curve_key == key
            )
        )
        cold_samples = self._sample_count(
            self._partial_total_curves, self._partial_inflight, (key, True)
        )
        reuse_samples = self._sample_count(
            self._partial_total_curves, self._partial_inflight, (key, False)
        )
        needs_partial_observation = (
            cold_samples < self._minimum_samples
            or reuse_samples < self._minimum_samples
        )
        pending.consumer_policy_probe = (
            potentially_late
            and needs_partial_observation
            and self._probe_misses.get(key, 0) < self._maximum_probe_misses
            and self._probe_attempts.get(key, 0) < self._maximum_samples
        )
        if pending.consumer_policy_probe:
            self._probe_attempts[key] = self._probe_attempts.get(key, 0) + 1
        planned = self.profitable_layers(key, minimum_gain=minimum_gain)
        pending.planned_progressive_layers = planned
        self._stats["consumer_policy_profiled_leases"] += int(
            pending.arrival_profiling
        )
        self._stats["consumer_policy_probe_leases"] += int(
            pending.consumer_policy_probe
        )
        self._stats["consumer_policy_planned_layers"] += len(planned)
        return key

    def record_arrival(
        self,
        *,
        batch: SglangForwardEpoch | None,
        phase: str,
        query: torch.Tensor,
        global_layer: int,
    ) -> None:
        if not self._enabled or batch is None or batch.pending_host_load is None:
            return
        pending = batch.pending_host_load
        key = pending.arrival_profile_key
        if key is None or not pending.arrival_profile_active:
            return
        local_layer = int(global_layer) - self._model_start_layer
        if not 0 <= local_layer < self._model_layer_count:
            raise RuntimeError("consumer-policy arrival is outside the model")
        if (
            key.phase != phase
            or key.query_rows_bucket != byte_scale_bucket(int(query.shape[0]))
            or key.batch_size_bucket != byte_scale_bucket(len(batch.bindings))
        ):
            raise RuntimeError("consumer-policy forward shape changed after binding")
        curve_key = (key, local_layer)
        if (
            self._sample_count(
                self._arrival_curves,
                self._arrival_inflight,
                curve_key,
            )
            >= self._minimum_samples
        ):
            return
        acquisition = (
            None if batch.acquisition is None else batch.acquisition.layer(local_layer)
        )
        publication = (
            None if acquisition is None else acquisition.partial_publication
        )
        ready = None if publication is None else publication.profile_ready_event
        if ready is None:
            raise RuntimeError("profiled Host acquisition omitted its timing marker")
        arrival = torch.cuda.Event(enable_timing=True)
        arrival.record(torch.cuda.current_stream())
        self._arrival_profiles.append(
            _LayerArrivalProfile(arrival, ready, key, local_layer)
        )
        self._arrival_inflight[curve_key] = (
            self._arrival_inflight.get(curve_key, 0) + 1
        )

    def record_stock_profile(
        self,
        *,
        pending: Any,
        global_layer: int,
        start: torch.cuda.Event,
        finish: torch.cuda.Event,
    ) -> None:
        """Record stock numerical service after its producer wait."""

        key = pending.arrival_profile_key
        if not self._enabled or key is None or not pending.arrival_profile_active:
            return
        local_layer = int(global_layer) - self._model_start_layer
        if not 0 <= local_layer < self._model_layer_count:
            raise RuntimeError("stock consumer profile is outside the model")
        curve_key = (key, local_layer)
        if (
            self._sample_count(self._stock_curves, self._stock_inflight, curve_key)
            >= self._minimum_samples
        ):
            return
        self._stock_profiles.append(
            _StockConsumerProfile(start, finish, key, local_layer)
        )
        self._stock_inflight[curve_key] = self._stock_inflight.get(curve_key, 0) + 1

    def record_partial_profile(
        self,
        *,
        pending: Any,
        start: torch.cuda.Event,
        dispatch_ready: torch.cuda.Event,
        finish: torch.cuda.Event,
        partition_prepared: bool,
    ) -> None:
        key = pending.arrival_profile_key
        if (
            not self._enabled
            or key is None
            or not pending.consumer_policy_probe
        ):
            return
        curve_key = (key, bool(partition_prepared))
        if (
            self._sample_count(
                self._partial_total_curves,
                self._partial_inflight,
                curve_key,
            )
            >= self._minimum_samples
        ):
            return
        self._partial_profiles.append(
            _PartialConsumerProfile(
                start,
                dispatch_ready,
                finish,
                key,
                bool(partition_prepared),
            )
        )
        self._partial_inflight[curve_key] = (
            self._partial_inflight.get(curve_key, 0) + 1
        )
        pending.partial_profile_recorded = True

    def retire_lease(self, pending: Any, *, probe_executed: bool) -> None:
        """Bound probes that reached no arriving consumer.

        Typed setup can itself delay attention until the producer is complete.
        Repeating that losing probe forever would turn a fail-closed policy
        into a steady-state regression.  A small finite miss budget records
        that this shape has no executable partial opportunity under the
        current deployment; later unseen geometry classes calibrate
        independently.
        """

        key = pending.arrival_profile_key
        if (
            not self._enabled
            or key is None
            or not probe_executed
            or pending.partial_profile_recorded
        ):
            return
        self._probe_misses[key] = self._probe_misses.get(key, 0) + 1
        self._stats["consumer_policy_probe_misses"] += 1
        if self._probe_misses[key] == self._maximum_probe_misses:
            self._stats["consumer_policy_rejected_shapes"] += 1

    def collect(self) -> None:
        """Commit completed CUDA observations without synchronizing serving."""

        pending_arrivals: list[_LayerArrivalProfile] = []
        arrival_profiles = self._arrival_profiles
        for index, profile in enumerate(arrival_profiles):
            try:
                if not profile.arrival.query() or not profile.ready.query():
                    pending_arrivals.append(profile)
                    continue
                signed_ns = round(
                    profile.arrival.elapsed_time(profile.ready) * 1_000_000.0
                )
                curve_key = (profile.key, profile.local_layer)
                self._arrival_curves[curve_key] = self._curve(
                    self._arrival_curves.get(curve_key)
                ).with_observation(signed_ns)
                self._stats["consumer_policy_arrival_samples"] += 1
            except BaseException:
                pending_arrivals.extend(arrival_profiles[index:])
                self._arrival_profiles = pending_arrivals
                self._rebuild_inflight()
                raise
        self._arrival_profiles = pending_arrivals

        pending_stock: list[_StockConsumerProfile] = []
        stock_profiles = self._stock_profiles
        for index, profile in enumerate(stock_profiles):
            try:
                if not profile.finish.query():
                    pending_stock.append(profile)
                    continue
                elapsed_ns = max(
                    1,
                    round(profile.start.elapsed_time(profile.finish) * 1_000_000.0),
                )
                curve_key = (profile.key, profile.local_layer)
                self._stock_curves[curve_key] = self._curve(
                    self._stock_curves.get(curve_key)
                ).with_observation(elapsed_ns)
                self._stats["consumer_policy_stock_samples"] += 1
            except BaseException:
                pending_stock.extend(stock_profiles[index:])
                self._stock_profiles = pending_stock
                self._rebuild_inflight()
                raise
        self._stock_profiles = pending_stock

        pending_partial: list[_PartialConsumerProfile] = []
        partial_profiles = self._partial_profiles
        for index, profile in enumerate(partial_profiles):
            try:
                if (
                    not profile.start.query()
                    or not profile.dispatch_ready.query()
                    or not profile.finish.query()
                ):
                    pending_partial.append(profile)
                    continue
                total_ns = max(
                    1,
                    round(profile.start.elapsed_time(profile.finish) * 1_000_000.0),
                )
                dispatch_ns = max(
                    0,
                    round(
                        profile.start.elapsed_time(profile.dispatch_ready)
                        * 1_000_000.0
                    ),
                )
                device_ns = max(
                    1,
                    round(
                        profile.dispatch_ready.elapsed_time(profile.finish)
                        * 1_000_000.0
                    ),
                )
                curve_key = (profile.key, profile.partition_prepared)
                total_curve = self._curve(
                    self._partial_total_curves.get(curve_key)
                ).with_observation(total_ns)
                dispatch_curve = self._curve(
                    self._partial_dispatch_curves.get(curve_key)
                ).with_observation(dispatch_ns)
                device_curve = self._curve(
                    self._partial_device_curves.get(curve_key)
                ).with_observation(device_ns)
                # Commit the three views as one logical observation. Profile
                # persistence requires equal sample histories; a failed curve
                # construction must not leave only the total path updated.
                self._partial_total_curves[curve_key] = total_curve
                self._partial_dispatch_curves[curve_key] = dispatch_curve
                self._partial_device_curves[curve_key] = device_curve
                self._stats["consumer_policy_partial_samples"] += 1
                suffix = "setup" if profile.partition_prepared else "reuse"
                self._stats[f"consumer_policy_partial_{suffix}_samples"] += 1
            except BaseException:
                pending_partial.extend(partial_profiles[index:])
                self._partial_profiles = pending_partial
                self._rebuild_inflight()
                raise
        self._partial_profiles = pending_partial
        self._rebuild_inflight()

    def profitable_layers(
        self,
        key: HostArrivalProfileKey,
        *,
        minimum_gain: float,
    ) -> frozenset[int]:
        if minimum_gain < 1.0:
            raise ValueError("consumer-policy minimum gain must be at least one")
        cold = self._partial_total_curves.get((key, True))
        if cold is None or not cold.calibrated:
            return frozenset()
        stock_costs = {
            layer: min(arrival.samples_ns) + min(stock.samples_ns)
            for layer in range(self._model_layer_count)
            if (arrival := self._arrival_curves.get((key, layer))) is not None
            and arrival.calibrated
            and min(arrival.samples_ns) > 0
            and (stock := self._stock_curves.get((key, layer))) is not None
            and stock.calibrated
        }
        if not stock_costs:
            return frozenset()
        cold_cost_ns = max(cold.samples_ns)
        reuse = self._partial_total_curves.get((key, False))
        if reuse is None or not reuse.calibrated:
            # One calibrated cold path proves at most one selection.  Do not
            # extrapolate an unobserved reuse cost across a transformer.
            layer, stock_cost_ns = max(
                stock_costs.items(), key=lambda item: (item[1], -item[0])
            )
            return (
                frozenset({layer})
                if stock_cost_ns >= math.ceil(cold_cost_ns * minimum_gain)
                else frozenset()
            )

        reuse_cost_ns = max(reuse.samples_ns)
        candidates = {
            layer: cost
            for layer, cost in stock_costs.items()
            if cost >= math.ceil(reuse_cost_ns * minimum_gain)
        }
        if not candidates:
            return frozenset()
        fixed_setup_ns = max(0, cold_cost_ns - reuse_cost_ns)
        stock_total_ns = sum(candidates.values())
        partial_total_ns = fixed_setup_ns + reuse_cost_ns * len(candidates)
        if stock_total_ns < math.ceil(partial_total_ns * minimum_gain):
            return frozenset()
        return frozenset(candidates)

    def shape_closed(self, key: HostArrivalProfileKey) -> bool:
        """Return whether a shape has a production consumer decision.

        Arrival and stock service must first be calibrated for every model
        layer. A shape whose producer is never late needs no partial probe. A
        late shape closes after both first-use and reusable partial costs are
        known, or after the bounded probe budget rejects the optional path.
        Exact CUDA-event ordering is independent of this policy state.
        """

        arrivals = tuple(
            self._arrival_curves.get((key, layer))
            for layer in range(self._model_layer_count)
        )
        stocks = tuple(
            self._stock_curves.get((key, layer))
            for layer in range(self._model_layer_count)
        )
        if any(curve is None or not curve.calibrated for curve in arrivals) or any(
            curve is None or not curve.calibrated for curve in stocks
        ):
            return False
        if not any(
            min(curve.samples_ns) > 0 for curve in arrivals if curve is not None
        ):
            return True
        cold = self._partial_total_curves.get((key, True))
        reuse = self._partial_total_curves.get((key, False))
        if (
            cold is not None
            and cold.calibrated
            and reuse is not None
            and reuse.calibrated
        ):
            return True
        return (
            self._probe_misses.get(key, 0) >= self._maximum_probe_misses
            or self._probe_attempts.get(key, 0) >= self._maximum_samples
        )

    @staticmethod
    def _key_order(key: HostArrivalProfileKey) -> tuple[Any, ...]:
        return (
            key.phase,
            key.query_rows_bucket,
            key.batch_size_bucket,
            key.transfer_rows_bucket,
            key.transfer_bytes_bucket,
            key.mover_kind,
            key.layers_per_submission,
            key.sm_waves_per_layer,
        )

    def export_state(self) -> dict[str, Any]:
        """Return bounded numerical policy state without request/CUDA ownership."""

        if self.pending_count:
            raise RuntimeError(
                "consumer-policy calibration cannot snapshot pending CUDA events"
            )

        def layer_rows(
            curves: dict[tuple[HostArrivalProfileKey, int], _BoundedTimingCurve]
        ) -> list[dict[str, Any]]:
            return [
                {
                    "key": _arrival_key_state(key),
                    "layer": layer,
                    "samples_ns": list(curve.samples_ns),
                }
                for (key, layer), curve in sorted(
                    curves.items(),
                    key=lambda item: (self._key_order(item[0][0]), item[0][1]),
                )
            ]

        def partial_rows(
            curves: dict[tuple[HostArrivalProfileKey, bool], _BoundedTimingCurve]
        ) -> list[dict[str, Any]]:
            return [
                {
                    "key": _arrival_key_state(key),
                    "partition_prepared": prepared,
                    "samples_ns": list(curve.samples_ns),
                }
                for (key, prepared), curve in sorted(
                    curves.items(),
                    key=lambda item: (self._key_order(item[0][0]), item[0][1]),
                )
            ]

        def counter_rows(values: dict[HostArrivalProfileKey, int]) -> list[dict[str, Any]]:
            return [
                {"key": _arrival_key_state(key), "value": count}
                for key, count in sorted(values.items(), key=lambda item: self._key_order(item[0]))
            ]

        return {
            "schema": _CALIBRATION_STATE_SCHEMA,
            "model_start_layer": self._model_start_layer,
            "model_layer_count": self._model_layer_count,
            "minimum_samples": self._minimum_samples,
            "maximum_samples": self._maximum_samples,
            "maximum_probe_misses": self._maximum_probe_misses,
            "arrival_curves": layer_rows(self._arrival_curves),
            "stock_curves": layer_rows(self._stock_curves),
            "partial_total_curves": partial_rows(self._partial_total_curves),
            "partial_dispatch_curves": partial_rows(self._partial_dispatch_curves),
            "partial_device_curves": partial_rows(self._partial_device_curves),
            "probe_misses": counter_rows(self._probe_misses),
            "probe_attempts": counter_rows(self._probe_attempts),
            "minimum_gains": [
                {"key": _arrival_key_state(key), "value": gain}
                for key, gain in sorted(
                    self._minimum_gain_by_key.items(),
                    key=lambda item: self._key_order(item[0]),
                )
            ],
        }

    def import_state(self, value: Any) -> int:
        """Restore one validated policy transaction before request admission."""

        if self.pending_count or any(
            (
                self._arrival_curves,
                self._stock_curves,
                self._partial_total_curves,
                self._partial_dispatch_curves,
                self._partial_device_curves,
                self._probe_misses,
                self._probe_attempts,
                self._minimum_gain_by_key,
            )
        ):
            raise RuntimeError("consumer-policy calibration is not empty")
        state = _state_mapping(value, "consumer-policy")
        fields = frozenset(
            {
                "schema",
                "model_start_layer",
                "model_layer_count",
                "minimum_samples",
                "maximum_samples",
                "maximum_probe_misses",
                "arrival_curves",
                "stock_curves",
                "partial_total_curves",
                "partial_dispatch_curves",
                "partial_device_curves",
                "probe_misses",
                "probe_attempts",
                "minimum_gains",
            }
        )
        _state_keys(state, fields, "consumer-policy")
        expected = (
            _CALIBRATION_STATE_SCHEMA,
            self._model_start_layer,
            self._model_layer_count,
            self._minimum_samples,
            self._maximum_samples,
            self._maximum_probe_misses,
        )
        actual = tuple(
            _state_int(state[name], f"consumer-policy {name}")
            for name in (
                "schema",
                "model_start_layer",
                "model_layer_count",
                "minimum_samples",
                "maximum_samples",
                "maximum_probe_misses",
            )
        )
        if actual != expected:
            raise ValueError("consumer-policy calibration geometry is incompatible")

        maximum_layer_rows = max(1024, self._model_layer_count * 4096)

        def parse_layer_curves(
            raw_rows: Any, owner: str, *, signed: bool
        ) -> dict[tuple[HostArrivalProfileKey, int], _BoundedTimingCurve]:
            result: dict[tuple[HostArrivalProfileKey, int], _BoundedTimingCurve] = {}
            for raw in _state_list(
                raw_rows, owner, maximum_items=maximum_layer_rows
            ):
                row = _state_mapping(raw, owner)
                _state_keys(
                    row, frozenset({"key", "layer", "samples_ns"}), owner
                )
                key = _arrival_key_from_state(row["key"])
                layer = _state_int(
                    row["layer"],
                    f"{owner} layer",
                    maximum=self._model_layer_count - 1,
                )
                curve_key = (key, layer)
                if curve_key in result:
                    raise ValueError(f"{owner} repeats a curve")
                result[curve_key] = _BoundedTimingCurve(
                    samples_ns=_state_samples(
                        row["samples_ns"],
                        owner,
                        maximum_samples=self._maximum_samples,
                        signed=signed,
                    ),
                    minimum_samples=self._minimum_samples,
                    maximum_samples=self._maximum_samples,
                )
            return result

        def parse_partial_curves(
            raw_rows: Any, owner: str, *, allow_zero: bool = False
        ) -> dict[tuple[HostArrivalProfileKey, bool], _BoundedTimingCurve]:
            result: dict[tuple[HostArrivalProfileKey, bool], _BoundedTimingCurve] = {}
            for raw in _state_list(raw_rows, owner, maximum_items=8192):
                row = _state_mapping(raw, owner)
                _state_keys(
                    row,
                    frozenset({"key", "partition_prepared", "samples_ns"}),
                    owner,
                )
                prepared = row["partition_prepared"]
                if not isinstance(prepared, bool):
                    raise ValueError(f"{owner} partition flag must be boolean")
                curve_key = (_arrival_key_from_state(row["key"]), prepared)
                if curve_key in result:
                    raise ValueError(f"{owner} repeats a curve")
                result[curve_key] = _BoundedTimingCurve(
                    samples_ns=_state_samples(
                        row["samples_ns"],
                        owner,
                        maximum_samples=self._maximum_samples,
                        allow_zero=allow_zero,
                    ),
                    minimum_samples=self._minimum_samples,
                    maximum_samples=self._maximum_samples,
                )
            return result

        def parse_counters(
            raw_rows: Any, owner: str, *, maximum: int
        ) -> dict[HostArrivalProfileKey, int]:
            result: dict[HostArrivalProfileKey, int] = {}
            for raw in _state_list(raw_rows, owner, maximum_items=8192):
                row = _state_mapping(raw, owner)
                _state_keys(row, frozenset({"key", "value"}), owner)
                key = _arrival_key_from_state(row["key"])
                if key in result:
                    raise ValueError(f"{owner} repeats a shape")
                result[key] = _state_int(
                    row["value"], owner, maximum=maximum
                )
            return result

        arrival_curves = parse_layer_curves(
            state["arrival_curves"], "arrival curve", signed=True
        )
        stock_curves = parse_layer_curves(
            state["stock_curves"], "stock curve", signed=False
        )
        partial_total = parse_partial_curves(
            state["partial_total_curves"], "partial-total curve"
        )
        partial_dispatch = parse_partial_curves(
            state["partial_dispatch_curves"],
            "partial-dispatch curve",
            allow_zero=True,
        )
        partial_device = parse_partial_curves(
            state["partial_device_curves"], "partial-device curve"
        )
        if not (
            partial_total.keys()
            == partial_dispatch.keys()
            == partial_device.keys()
        ) or any(
            not (
                len(partial_total[key].samples_ns)
                == len(partial_dispatch[key].samples_ns)
                == len(partial_device[key].samples_ns)
            )
            for key in partial_total
        ):
            raise ValueError("partial-consumer calibration views are inconsistent")
        probe_misses = parse_counters(
            state["probe_misses"],
            "probe misses",
            maximum=self._maximum_probe_misses,
        )
        probe_attempts = parse_counters(
            state["probe_attempts"],
            "probe attempts",
            maximum=self._maximum_samples,
        )
        minimum_gains: dict[HostArrivalProfileKey, float] = {}
        for raw in _state_list(
            state["minimum_gains"], "minimum gains", maximum_items=8192
        ):
            row = _state_mapping(raw, "minimum gain")
            _state_keys(row, frozenset({"key", "value"}), "minimum gain")
            key = _arrival_key_from_state(row["key"])
            raw_gain = row["value"]
            if isinstance(raw_gain, bool) or not isinstance(raw_gain, (int, float)):
                raise ValueError("consumer-policy minimum gain is not numeric")
            gain = float(raw_gain)
            if not math.isfinite(gain) or gain < 1.0 or key in minimum_gains:
                raise ValueError("consumer-policy minimum gain is invalid")
            minimum_gains[key] = gain
        curve_keys = {
            *(key for key, _layer in arrival_curves),
            *(key for key, _layer in stock_curves),
            *(key for key, _prepared in partial_total),
        }
        if not curve_keys.issubset(minimum_gains):
            raise ValueError("consumer-policy curve has no selection margin")

        self._arrival_curves = arrival_curves
        self._stock_curves = stock_curves
        self._partial_total_curves = partial_total
        self._partial_dispatch_curves = partial_dispatch
        self._partial_device_curves = partial_device
        self._probe_misses = probe_misses
        self._probe_attempts = probe_attempts
        self._minimum_gain_by_key = minimum_gains
        return sum(
            len(curve.samples_ns)
            for curves in (
                arrival_curves,
                stock_curves,
                partial_total,
                partial_dispatch,
                partial_device,
            )
            for curve in curves.values()
        )

    def report(self) -> dict[str, Any]:
        calibrated_arrivals = sum(
            curve.calibrated for curve in self._arrival_curves.values()
        )
        calibrated_stock = sum(curve.calibrated for curve in self._stock_curves.values())
        keys = sorted(
            {
                *self._minimum_gain_by_key,
                *(key for key, _layer in self._arrival_curves),
                *(key for key, _prepared in self._partial_total_curves),
            },
            key=lambda key: (
                key.phase,
                key.query_rows_bucket,
                key.batch_size_bucket,
                key.transfer_bytes_bucket,
                key.mover_kind,
                key.layers_per_submission,
                key.sm_waves_per_layer,
            ),
        )
        shapes = []
        for key in keys:
            arrivals = {
                layer: curve
                for (curve_key, layer), curve in self._arrival_curves.items()
                if curve_key == key and curve.calibrated
            }
            stocks = {
                layer: curve
                for (curve_key, layer), curve in self._stock_curves.items()
                if curve_key == key and curve.calibrated
            }
            cold_key = (key, True)
            reuse_key = (key, False)
            cold = self._partial_total_curves.get(cold_key)
            reuse = self._partial_total_curves.get(reuse_key)
            cold_dispatch = self._partial_dispatch_curves.get(cold_key)
            reuse_dispatch = self._partial_dispatch_curves.get(reuse_key)
            cold_device = self._partial_device_curves.get(cold_key)
            reuse_device = self._partial_device_curves.get(reuse_key)
            gain = self._minimum_gain_by_key.get(key, 1.0)
            shapes.append(
                {
                    "phase": key.phase,
                    "query_rows_bucket": key.query_rows_bucket,
                    "batch_size_bucket": key.batch_size_bucket,
                    "transfer_rows_bucket": key.transfer_rows_bucket,
                    "transfer_bytes_bucket": key.transfer_bytes_bucket,
                    "mover_kind": key.mover_kind,
                    "layers_per_submission": key.layers_per_submission,
                    "sm_waves_per_layer": key.sm_waves_per_layer,
                    "calibrated_arrival_layers": len(arrivals),
                    "calibrated_stock_layers": len(stocks),
                    "maximum_conservative_lateness_ns": max(
                        (min(curve.samples_ns) for curve in arrivals.values()),
                        default=None,
                    ),
                    "minimum_stock_service_ns": min(
                        (min(curve.samples_ns) for curve in stocks.values()),
                        default=None,
                    ),
                    "maximum_partial_cold_critical_path_ns": (
                        max(cold.samples_ns)
                        if cold is not None and cold.calibrated
                        else None
                    ),
                    "maximum_partial_reuse_critical_path_ns": (
                        max(reuse.samples_ns)
                        if reuse is not None and reuse.calibrated
                        else None
                    ),
                    "maximum_partial_cold_dispatch_ns": (
                        max(cold_dispatch.samples_ns)
                        if cold_dispatch is not None and cold_dispatch.calibrated
                        else None
                    ),
                    "maximum_partial_reuse_dispatch_ns": (
                        max(reuse_dispatch.samples_ns)
                        if reuse_dispatch is not None and reuse_dispatch.calibrated
                        else None
                    ),
                    "maximum_partial_cold_device_ns": (
                        max(cold_device.samples_ns)
                        if cold_device is not None and cold_device.calibrated
                        else None
                    ),
                    "maximum_partial_reuse_device_ns": (
                        max(reuse_device.samples_ns)
                        if reuse_device is not None and reuse_device.calibrated
                        else None
                    ),
                    "estimated_partial_fixed_setup_ns": (
                        max(0, max(cold.samples_ns) - max(reuse.samples_ns))
                        if cold is not None
                        and cold.calibrated
                        and reuse is not None
                        and reuse.calibrated
                        else None
                    ),
                    "profitable_layers": len(
                        self.profitable_layers(key, minimum_gain=gain)
                    ),
                    "probe_attempts": self._probe_attempts.get(key, 0),
                    "probe_misses": self._probe_misses.get(key, 0),
                    "closed": self.shape_closed(key),
                }
            )
        closed_shapes = sum(self.shape_closed(key) for key in keys)
        return {
            "mode": "frozen" if self._frozen else "learning",
            "minimum_samples": self._minimum_samples,
            "maximum_samples": self._maximum_samples,
            "maximum_probe_misses": self._maximum_probe_misses,
            "arrival_shapes": len({key for key, _layer in self._arrival_curves}),
            "calibrated_arrival_layers": calibrated_arrivals,
            "calibrated_stock_layers": calibrated_stock,
            "partial_shapes": len(
                {key for key, _prepared in self._partial_total_curves}
            ),
            "calibrated_partial_shapes": sum(
                curve.calibrated
                for (key, prepared), curve in self._partial_total_curves.items()
                if prepared
            ),
            "calibrated_partial_reuse_shapes": sum(
                curve.calibrated
                for (key, prepared), curve in self._partial_total_curves.items()
                if not prepared
            ),
            "probe_rejected_shapes": sum(
                misses >= self._maximum_probe_misses
                for misses in self._probe_misses.values()
            ),
            "closed_shapes": closed_shapes,
            "open_shapes": len(keys) - closed_shapes,
            "last_shape_closed": (
                None
                if self._last_key is None
                else self._frozen or self.shape_closed(self._last_key)
            ),
            "last_shape_calibrated": (
                None if self._last_key is None else self.shape_closed(self._last_key)
            ),
            "last_shape_decision": (
                None
                if self._last_key is None
                else "profile"
                if self.shape_closed(self._last_key)
                else "conservative_stock"
                if self._frozen
                else "learning"
            ),
            "shapes": shapes,
        }

    def _curve(self, curve: _BoundedTimingCurve | None) -> _BoundedTimingCurve:
        return curve or _BoundedTimingCurve(
            minimum_samples=self._minimum_samples,
            maximum_samples=self._maximum_samples,
        )

    @staticmethod
    def _sample_count(curves: dict[Any, Any], inflight: dict[Any, int], key: Any) -> int:
        curve = curves.get(key)
        return (0 if curve is None else len(curve.samples_ns)) + inflight.get(key, 0)

    def _rebuild_inflight(self) -> None:
        arrivals: dict[tuple[HostArrivalProfileKey, int], int] = {}
        for profile in self._arrival_profiles:
            key = (profile.key, profile.local_layer)
            arrivals[key] = arrivals.get(key, 0) + 1
        stocks: dict[tuple[HostArrivalProfileKey, int], int] = {}
        for profile in self._stock_profiles:
            key = (profile.key, profile.local_layer)
            stocks[key] = stocks.get(key, 0) + 1
        partial: dict[tuple[HostArrivalProfileKey, bool], int] = {}
        for profile in self._partial_profiles:
            key = (profile.key, profile.partition_prepared)
            partial[key] = partial.get(key, 0) + 1
        self._arrival_inflight = arrivals
        self._stock_inflight = stocks
        self._partial_inflight = partial
