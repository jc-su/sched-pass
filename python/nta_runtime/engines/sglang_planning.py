"""Pure SGLang planning, geometry, and deployment-option helpers.

No function in this module owns a CUDA stream, a HiCache lease, or framework
lifecycle state.  Keeping these decisions pure makes them reusable by the
admission bridge and testable without constructing the  attention backend.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Any

import torch

from nta_runtime.requests import RequestBinding


# Stable namespace for SGLang acquisition objects.  Publication code imports
# this value directly; keeping it public prevents a refactor from leaving a
# runtime-only undefined private name in the materialization path.
DEMAND_OBJECT_ID_BASE = 0x4E54410000000000
LOOKAHEAD_OBJECT_VERSION = 1
MAX_ABI_BYTES = (1 << 32) - 1


def byte_scale_bucket(transfer_bytes: int) -> int:
    """Return the power-of-two service-geometry class for a transfer."""

    if transfer_bytes <= 0:
        raise ValueError("mover calibration bytes must be positive")
    return transfer_bytes.bit_length() - 1


def maximum_mover_wave_bytes(
    row_bytes_by_layer: tuple[tuple[int, int], ...],
    transfer_count: int,
    layers_per_wave: int,
) -> int:
    if not row_bytes_by_layer or min(transfer_count, layers_per_wave) <= 0:
        raise ValueError("mover wave geometry must be positive")
    return max(
        transfer_count
        * sum(
            key_bytes + value_bytes
            for key_bytes, value_bytes in row_bytes_by_layer[
                begin : begin + layers_per_wave
            ]
        )
        for begin in range(0, len(row_bytes_by_layer), layers_per_wave)
    )


def calibration_probe_end(
    ready_prefix: int, layer_count: int, layers_per_wave: int
) -> int:
    """Return one complete same-scale mover wave for calibration."""

    if (
        layer_count <= 0
        or layers_per_wave <= 0
        or ready_prefix < 0
        or ready_prefix >= layer_count
    ):
        raise ValueError("mover calibration frontier is invalid")
    return min(layer_count, ready_prefix + layers_per_wave)


def mover_layout_required(policy: str, profile_layout: bool) -> bool:
    """Return whether the selected mover needs a host run decomposition."""

    if policy not in {"auto", "sm", "copy_engine", "probe_copy"}:
        raise ValueError("unknown indexed mover policy")
    return profile_layout or policy != "sm"


def requires_feasible_edf(
    bindings: tuple[RequestBinding, ...], *, tenant_isolation: bool
) -> bool:
    """Return whether descriptor order cannot replace device EDF ordering.

    Equal request deadlines and priorities form one EDF-equivalence class, so
    canonical request/segment order is a valid deterministic tie break.  The
    indexed Host path may then avoid materializing a queue that it never
    consumes.  Distinct policy keys, or tenant credit isolation, retain the
    dynamic queue so issue order and feasibility remain device governed.
    """

    if not bindings:
        raise ValueError("host acquisition has no request bindings")
    policy_keys = {
        (binding.deadline_clock, binding.priority) for binding in bindings
    }
    return tenant_isolation or len(policy_keys) > 1


def positive_environment(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def nonnegative_environment(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value


def boolean_environment(name: str, default: bool) -> bool:
    """Read one fail-closed 0/1 deployment switch."""

    raw = os.environ.get(name, "1" if default else "0").strip()
    if raw not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1")
    return raw == "1"


def demand_overlap_policy(
    *, host_staged: bool, frontier_enabled: bool, graph_requested: bool
) -> tuple[bool, bool, str]:
    """Select one mutually exclusive launch-amortization strategy."""

    demand_graph = host_staged and graph_requested
    fragment_lookahead = frontier_enabled and not demand_graph
    policy = (
        "finite_demand_graph"
        if demand_graph
        else "first_wave_lookahead"
        if fragment_lookahead
        else "none"
    )
    return demand_graph, fragment_lookahead, policy


def mover_stream_priority() -> int:
    value = int(os.environ.get("NTA_RUNTIME_MOVER_STREAM_PRIORITY", "0"))
    if value > 0:
        raise ValueError(
            "NTA_RUNTIME_MOVER_STREAM_PRIORITY must be zero or negative because "
            "CUDA stream priorities are non-positive"
        )
    return value


def host_mover_environment() -> str:
    value = os.environ.get("NTA_EXECUTION_HOST_MOVER", "auto").strip().lower()
    if value not in {"auto", "sm", "copy_engine"}:
        raise RuntimeError("NTA_EXECUTION_HOST_MOVER must be auto, sm, or copy_engine")
    return value


def require_exact_prefetch_layers(
    prefetched_layers: Mapping[int, Any],
    layer_count: int,
    *,
    consumer: str,
) -> int:
    """Validate full-model readiness and return the final local layer."""

    expected_layers = set(range(layer_count))
    actual_layers = set(prefetched_layers)
    if actual_layers != expected_layers:
        missing = sorted(expected_layers - actual_layers)
        unexpected = sorted(actual_layers - expected_layers)
        raise RuntimeError(
            f"{consumer} requires an exact full-model prefetch "
            f"(missing={missing}, unexpected={unexpected})"
        )
    if layer_count <= 0:
        raise RuntimeError(f"{consumer} requires at least one model layer")
    return layer_count - 1


def pipeline_object_range(
    object_capacity: int,
    consumer_index: int,
    layer_count: int,
    reserved_waves_per_layer: int,
) -> tuple[int, int]:
    """Reserve one producer's wave objects from the directory's high end."""

    if (
        object_capacity <= 0
        or consumer_index < 0
        or layer_count <= 0
        or reserved_waves_per_layer <= 0
    ):
        raise RuntimeError("HiCache layer-object geometry is invalid")
    object_count = 2 * layer_count * reserved_waves_per_layer
    end = object_capacity - consumer_index * object_count
    begin = end - object_count
    if begin < 2 or end > object_capacity:
        raise RuntimeError("HiCache layer objects exceed NTA directory capacity")
    return begin, end


def pipeline_object_id(
    consumer_index: int,
    layer_count: int,
    local_layer: int,
    reserved_waves_per_layer: int,
) -> int:
    """Return the first stable K-object identity for one proactive layer."""

    if (
        consumer_index < 0
        or layer_count <= 0
        or local_layer < 0
        or local_layer >= layer_count
        or reserved_waves_per_layer <= 0
    ):
        raise RuntimeError("HiCache proactive object identity is out of range")
    return (
        DEMAND_OBJECT_ID_BASE
        + (1 << 44)
        + consumer_index * 2 * layer_count * reserved_waves_per_layer
        + 2 * reserved_waves_per_layer * local_layer
    )


def dtype_tag(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.").replace("_", "")


def semantic_plan_signature_prefix(
    request_indices: tuple[int, ...],
    kv_tile_indices: tuple[int, ...],
    dependency_geometry: Any,
    request_identities: tuple[tuple[int, int], ...],
) -> tuple[Any, ...]:
    """Return generation-complete identity for one wrapper plan.

    Native work items bind both request slot and generation.  A cache key that
    names only slots can otherwise replay a stale generation after the engine
    reuses its request pool.
    """

    if not request_identities or any(
        slot < 0 or generation <= 0 for slot, generation in request_identities
    ):
        raise ValueError("semantic plan request identities are invalid")

    return (
        request_indices,
        kv_tile_indices,
        dependency_geometry,
        request_identities,
    )


def cpu_sequence_lengths(forward_batch: Any, request_count: int) -> tuple[int, ...]:
    """Read SGLang's existing CPU mirror without introducing a GPU sync."""

    values = getattr(forward_batch, "seq_lens_cpu", None)
    if values is None:
        raise RuntimeError("SGLang forward omitted its CPU sequence-length mirror")
    if isinstance(values, torch.Tensor):
        if values.is_cuda:
            raise RuntimeError(
                "SGLang sequence-length mirror unexpectedly resides on GPU"
            )
        values = values.tolist()
    lengths = tuple(int(value) for value in values)
    if len(lengths) != request_count or any(length <= 0 for length in lengths):
        raise RuntimeError("SGLang sequence lengths do not match the request batch")
    return lengths
