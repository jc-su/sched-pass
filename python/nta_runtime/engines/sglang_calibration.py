"""Bounded deployment calibration for SGLang layer-arrival service.

This owner contains only observations that may influence future scheduling.
It does not own request identity, transfer submission, attention dispatch, or
artifact publication.  CUDA events remain pending until their elapsed time is
queryable; completed samples update one conservative curve per stable forward
shape without synchronizing the serving stream.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from nta_runtime.acquisition_scheduler import AcquisitionServiceCurve
from nta_runtime.engines.sglang_acquisition_contract import HostArrivalProfileKey
from nta_runtime.engines.sglang_planning import byte_scale_bucket
from nta_runtime.engines.sglang_state import SglangForwardEpoch


LayerServiceKey = tuple[str, int, int]


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
    ) -> None:
        if (
            minimum_samples <= 0
            or maximum_samples < minimum_samples
            or model_start_layer < 0
            or model_layer_count <= 0
        ):
            raise ValueError("SGLang layer-service calibration geometry is invalid")
        self._enabled = bool(enabled)
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

        if not self._enabled or batch is None or batch.pending_host_load is None:
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
    finish: torch.cuda.Event
    key: HostArrivalProfileKey


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
    and complete partial-consumer service.  AUTO enables a layer only when the
    conservative observed stock path (producer wait plus numerical attention)
    exceeds the largest observed partial-consumer service cost by the
    configured gain factor. Unknown or noisy shapes therefore use the
    scheduled-preacquired stock specialization.
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
        self._partial_curves: dict[HostArrivalProfileKey, _BoundedTimingCurve] = {}
        self._arrival_profiles: list[_LayerArrivalProfile] = []
        self._stock_profiles: list[_StockConsumerProfile] = []
        self._partial_profiles: list[_PartialConsumerProfile] = []
        self._arrival_inflight: dict[tuple[HostArrivalProfileKey, int], int] = {}
        self._stock_inflight: dict[tuple[HostArrivalProfileKey, int], int] = {}
        self._partial_inflight: dict[HostArrivalProfileKey, int] = {}
        self._probe_misses: dict[HostArrivalProfileKey, int] = {}
        self._minimum_gain_by_key: dict[HostArrivalProfileKey, float] = {}

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
        pending.arrival_profile_key = key
        pending.arrival_profile_active = False
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
        pending.consumer_policy_probe = potentially_late and (
            self._sample_count(self._partial_curves, self._partial_inflight, key)
            < self._minimum_samples
        ) and self._probe_misses.get(key, 0) < self._maximum_probe_misses
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
        finish: torch.cuda.Event,
    ) -> None:
        key = pending.arrival_profile_key
        if (
            not self._enabled
            or key is None
            or not pending.consumer_policy_probe
            or pending.partial_profile_recorded
        ):
            return
        if (
            self._sample_count(self._partial_curves, self._partial_inflight, key)
            >= self._minimum_samples
        ):
            return
        self._partial_profiles.append(_PartialConsumerProfile(start, finish, key))
        self._partial_inflight[key] = self._partial_inflight.get(key, 0) + 1
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
                if not profile.start.query() or not profile.finish.query():
                    pending_partial.append(profile)
                    continue
                elapsed_ns = max(
                    1,
                    round(profile.start.elapsed_time(profile.finish) * 1_000_000.0),
                )
                self._partial_curves[profile.key] = self._curve(
                    self._partial_curves.get(profile.key)
                ).with_observation(elapsed_ns)
                self._stats["consumer_policy_partial_samples"] += 1
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
        partial = self._partial_curves.get(key)
        if partial is None or not partial.calibrated:
            return frozenset()
        conservative_cost_ns = max(partial.samples_ns)
        return frozenset(
            layer
            for layer in range(self._model_layer_count)
            if (arrival := self._arrival_curves.get((key, layer))) is not None
            and arrival.calibrated
            and min(arrival.samples_ns) > 0
            and (stock := self._stock_curves.get((key, layer))) is not None
            and stock.calibrated
            and (
                min(arrival.samples_ns) + min(stock.samples_ns)
                >= math.ceil(conservative_cost_ns * minimum_gain)
            )
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
                *self._partial_curves,
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
            partial = self._partial_curves.get(key)
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
                    "maximum_partial_service_ns": (
                        max(partial.samples_ns)
                        if partial is not None and partial.calibrated
                        else None
                    ),
                    "profitable_layers": len(
                        self.profitable_layers(key, minimum_gain=gain)
                    ),
                    "probe_misses": self._probe_misses.get(key, 0),
                }
            )
        return {
            "minimum_samples": self._minimum_samples,
            "maximum_samples": self._maximum_samples,
            "maximum_probe_misses": self._maximum_probe_misses,
            "arrival_shapes": len({key for key, _layer in self._arrival_curves}),
            "calibrated_arrival_layers": calibrated_arrivals,
            "calibrated_stock_layers": calibrated_stock,
            "partial_shapes": len(self._partial_curves),
            "calibrated_partial_shapes": sum(
                curve.calibrated for curve in self._partial_curves.values()
            ),
            "probe_rejected_shapes": sum(
                misses >= self._maximum_probe_misses
                for misses in self._probe_misses.values()
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
        partial: dict[HostArrivalProfileKey, int] = {}
        for profile in self._partial_profiles:
            partial[profile.key] = partial.get(profile.key, 0) + 1
        self._arrival_inflight = arrivals
        self._stock_inflight = stocks
        self._partial_inflight = partial
