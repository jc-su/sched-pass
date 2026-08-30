"""Setup-owned SGLang HBM resources for direct NVMe acquisition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import time
from types import MappingProxyType
from typing import Any

import torch

from nta_runtime.hbm_registration import HbmDestinationSlice
from nta_runtime.nvme_granularity import plan_nvme_scratch_capacity
from nta_runtime.nvme_materialization import NvmeScratchArena


_SCRATCH_RESOURCE_KEY = ("nta", "nvme-span-scratch")


@dataclass(frozen=True, slots=True)
class SglangNvmeResources:
    """Setup-owned HBM resources borrowed by the steady-state pipeline."""

    regions: Mapping[tuple[int, str], Any]
    scratch: NvmeScratchArena | None


def prepare_sglang_nvme_resources(
    *,
    tier_service: Any,
    token_to_kv_pool: Any,
    model_start_layer: int,
    model_end_layer: int,
    global_model_layer_count: int,
    stats: dict[str, Any],
) -> SglangNvmeResources:
    """Register stable KV destinations and bounded scratch exactly once."""

    started = time.perf_counter_ns()
    catalog = tier_service.catalog
    if catalog is None or catalog.layer_count != global_model_layer_count:
        raise RuntimeError("NVMe catalog layer count does not match the SGLang model")
    if catalog.page_tokens != 1:
        raise RuntimeError("NVMe SGLang integration currently requires page_tokens=1")
    destinations: list[HbmDestinationSlice] = []
    for layer_id in range(model_start_layer, model_end_layer):
        tensors = (
            ("key", token_to_kv_pool._get_key_buffer(layer_id)),
            ("value", token_to_kv_pool._get_value_buffer(layer_id)),
        )
        for kind, tensor in tensors:
            if not tensor.is_cuda or int(tensor.nbytes) <= 0:
                raise RuntimeError(
                    f"NVMe {kind} region for layer {layer_id} is not live CUDA HBM"
                )
            destinations.append(
                HbmDestinationSlice(
                    (layer_id, kind),
                    int(tensor.data_ptr()),
                    int(tensor.nbytes),
                )
            )

    scratch_tensor: torch.Tensor | None = None
    if tier_service.config.nvme_service_model.calibrated:
        scratch_bytes = plan_nvme_scratch_capacity(
            queue_depth=int(tier_service.config.queue_depth),
            max_transfer_bytes=tier_service.nvme_max_transfer_bytes,
        )
        scratch_tensor = torch.empty(
            scratch_bytes,
            dtype=torch.uint8,
            device=token_to_kv_pool.device,
        )
        destinations.append(
            HbmDestinationSlice(
                _SCRATCH_RESOURCE_KEY,
                int(scratch_tensor.data_ptr()),
                int(scratch_tensor.numel()),
            )
        )
    try:
        prepared = tier_service.prepare_nvme_hbm_destinations(tuple(destinations))
    except BaseException as error:
        raise RuntimeError(
            "NVMe worker-prepare could not register the complete local KV "
            f"destination set (layers=[{model_start_layer}, {model_end_layer}), "
            f"tensors={len(destinations)})"
        ) from error

    stats["nvme_region_prepare_ns"] = time.perf_counter_ns() - started
    stats["nvme_region_count"] = prepared.registration_count
    stats["nvme_region_bytes"] = prepared.registration_bytes
    stats["nvme_destination_slice_count"] = prepared.destination_count
    stats["nvme_destination_slice_bytes"] = prepared.destination_bytes
    stats["nvme_shared_region_slices"] = (
        prepared.destination_count - prepared.registration_count
    )
    regions = dict(prepared.regions)
    scratch = (
        None
        if scratch_tensor is None
        else NvmeScratchArena(
            scratch_tensor,
            regions.pop(_SCRATCH_RESOURCE_KEY),
        )
    )
    stats["nvme_span_scratch_bytes"] = 0 if scratch is None else scratch.bytes
    stats["nvme_span_service_model_calibrated"] = (
        tier_service.config.nvme_service_model.calibrated
    )
    return SglangNvmeResources(MappingProxyType(regions), scratch)
