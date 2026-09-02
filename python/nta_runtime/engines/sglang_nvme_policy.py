"""Closed-loop NVMe consumer policy at the framework/runtime boundary."""

from __future__ import annotations

from typing import Any

from nta_runtime.engines.sglang_calibration import (
    SglangConsumerPolicyCalibration,
    SglangLayerServiceCalibration,
)
from nta_runtime.engines.sglang_hicache import PendingHostLoad
from nta_runtime.engines.sglang_nvme import (
    NvmeBatchGeometry,
)
from nta_runtime.engines.sglang_state import SglangForwardEpoch


def plan_nvme_consumer_policy(
    *,
    forward_batch: Any,
    pending: PendingHostLoad,
    batch: SglangForwardEpoch,
    geometry: NvmeBatchGeometry,
    window_count: int,
    wave_count: int,
    layer_calibration: SglangLayerServiceCalibration,
    consumer_calibration: SglangConsumerPolicyCalibration,
    model_layer_count: int,
    minimum_gain: float,
    stats: dict[str, Any],
) -> frozenset[int]:
    """Bind one measured whole-layer/partial decision to an NVMe forward.

    Event partitions are structural and prepared before attention. This policy
    controls whether they execute: uncalibrated shapes collect producer/stock
    observations, bounded probes measure the partial path, and only a
    conservative aggregate gain installs progressive layers in production.
    """

    if (
        model_layer_count <= 0
        or minimum_gain < 1.0
        or window_count <= 0
        or wave_count < 0
    ):
        raise ValueError("NVMe consumer policy has invalid deployment geometry")
    layer_service_key = layer_calibration.shape_key(forward_batch)
    batch.layer_service_key = layer_service_key
    if layer_service_key is None:
        stats["nvme_partial_consumer_unclassified_batches"] = stats.get(
            "nvme_partial_consumer_unclassified_batches", 0
        ) + 1
        return frozenset()

    consumer_calibration.bind_lease(
        pending,
        layer_service_key=layer_service_key,
        producer_kind=f"nvme_{geometry.granularity.value}",
        layers_per_submission=max(
            1, (model_layer_count + window_count - 1) // window_count
        ),
        sm_waves_per_layer=max(1, wave_count),
        minimum_gain=minimum_gain,
        transfer_rows=max(
            1, geometry.logical_transfer_bytes // sum(geometry.row_bytes)
        ),
        transfer_bytes=geometry.logical_transfer_bytes,
    )
    pending.arrival_profile_active = bool(pending.arrival_profiling)
    progressive_layers = set(range(model_layer_count)) if wave_count else set()
    planned = (
        progressive_layers
        if pending.consumer_policy_probe
        else progressive_layers & set(pending.planned_progressive_layers)
    )
    batch.planned_progressive_consumer_layers.update(planned)
    stats["nvme_partial_consumer_planned_layers"] = stats.get(
        "nvme_partial_consumer_planned_layers", 0
    ) + len(planned)
    return frozenset(planned)
