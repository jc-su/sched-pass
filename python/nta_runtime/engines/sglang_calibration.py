"""Bounded deployment calibration for SGLang layer-arrival service.

This owner contains only observations that may influence future scheduling.
It does not own request identity, transfer submission, attention dispatch, or
artifact publication.  CUDA events remain pending until their elapsed time is
queryable; completed samples update one conservative curve per stable forward
shape without synchronizing the serving stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from nta_runtime.acquisition_scheduler import AcquisitionServiceCurve
from nta_runtime.engines.sglang_state import _ActiveBatch


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
        batch: _ActiveBatch | None,
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
