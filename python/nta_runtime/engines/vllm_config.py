"""Immutable deployment configuration for the vLLM execution boundary.

vLLM constructs its worker controller, metadata builder, and attention
implementation at different framework lifecycle points.  Each owner parses
its process-start configuration exactly once and carries typed values into the
forward path; request binding and per-layer attention never consult the
environment.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import os
from pathlib import Path

from nta_runtime.adapters.vllm_v1 import validate_vllm_attention_tier
from nta_runtime.tenant import (
    tenant_budget_specs,
    tenant_isolation_required,
    tenant_mapper_from_environment,
)
from nta_runtime.tier import ServingTierConfig


_UINT64_MAX = (1 << 64) - 1
SUPPORTED_VLLM_VERSION = "0.26.0"


def _boolean(values: Mapping[str, str], name: str, *, default: bool = False) -> bool:
    raw = values.get(name, "1" if default else "0").strip()
    if raw not in {"0", "1"}:
        raise RuntimeError(f"{name} must be 0 or 1")
    return raw == "1"


def _positive(values: Mapping[str, str], name: str, default: int) -> int:
    raw = values.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def _nonnegative(values: Mapping[str, str], name: str, default: int) -> int:
    raw = values.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a nonnegative integer") from error
    if value < 0 or value > _UINT64_MAX:
        raise RuntimeError(f"{name} must be a nonnegative 64-bit integer")
    return value


def vllm_host_layers_per_wave(
    environ: Mapping[str, str] | None = None,
) -> int | None:
    """Return the explicit Host transport-wave override, if configured.

    An unset value selects one maximally coalesced model-layer wave, the
    measured production default. The override exists only for deployment
    calibration and mechanism sweeps; request-time code never consults the
    environment and no uncalibrated "auto" policy is implied.
    """

    values = os.environ if environ is None else environ
    raw = values.get("NTA_VLLM_HOST_LAYERS_PER_WAVE", "").strip().lower()
    if not raw:
        return None
    return _positive(values, "NTA_VLLM_HOST_LAYERS_PER_WAVE", 1)


@dataclass(frozen=True, slots=True)
class VllmWorkerConfig:
    """One worker's resource ownership and bridge configuration."""

    work_ticket_capacity: int
    max_dependencies_per_work_ticket: int
    object_capacity: int
    tenant_capacity: int
    tenant_specs: tuple[tuple[int, int], ...]
    tenant_isolation_enabled: bool
    tenant_for_request: Callable[[str], int] | None
    staging_byte_capacity: int
    tier: ServingTierConfig
    profile_cpu: bool

    @classmethod
    def from_environment(
        cls,
        *,
        request_capacity: int,
        max_batched_tokens: int,
        environ: Mapping[str, str] | None = None,
    ) -> "VllmWorkerConfig":
        if request_capacity <= 0:
            raise ValueError("vLLM request capacity must be positive")
        if max_batched_tokens < 0:
            raise ValueError("vLLM maximum batched tokens cannot be negative")
        values = os.environ if environ is None else environ
        default_work_capacity = max(256, 64 * request_capacity, max_batched_tokens)
        work_capacity = _positive(
            values, "NTA_VLLM_WORK_TICKET_CAPACITY", default_work_capacity
        )
        max_dependencies = _positive(values, "NTA_VLLM_MAX_DEPENDENCIES_PER_WORK", 32)
        tenant_capacity = _positive(
            values, "NTA_TENANT_CAPACITY", max(1, request_capacity)
        )
        tenant_specs = tenant_budget_specs(values)
        for tenant_id, _maximum_bytes in tenant_specs:
            if tenant_id >= tenant_capacity:
                raise RuntimeError(
                    f"tenant {tenant_id} exceeds NTA_TENANT_CAPACITY={tenant_capacity}"
                )
        staging_limit = _nonnegative(values, "NTA_STAGING_BYTE_CAPACITY", _UINT64_MAX)
        # Validate both the engine's numerical address-space contract and the
        # transport owner's concrete endpoint/catalog configuration now.
        selected_tier = validate_vllm_attention_tier(values)
        tier = ServingTierConfig.from_environment(values)
        if tier.tier.value != selected_tier:
            raise RuntimeError("vLLM attention and resource tiers disagree")
        return cls(
            work_ticket_capacity=work_capacity,
            max_dependencies_per_work_ticket=max_dependencies,
            object_capacity=max(2, max_dependencies * work_capacity),
            tenant_capacity=tenant_capacity,
            tenant_specs=tenant_specs,
            tenant_isolation_enabled=tenant_isolation_required(tenant_specs),
            tenant_for_request=tenant_mapper_from_environment(values),
            # Native zero means unlimited.
            staging_byte_capacity=staging_limit or _UINT64_MAX,
            tier=tier,
            profile_cpu=_boolean(values, "NTA_PROFILE_CPU"),
        )


@dataclass(frozen=True, slots=True)
class VllmAttentionConfig:
    """One metadata-builder/attention-implementation execution contract."""

    native_enabled: bool
    serving_tier: str
    profile_cpu: bool
    verify_execution: bool
    verify_transfer: bool
    compare_stock: bool
    allow_stock_fallback: bool
    workspace_base: Path | None
    workspace_bytes: int
    decode_module_override: str | None
    prefill_module_override: str | None
    host_copy_blocks_per_group: int

    @classmethod
    def from_environment(
        cls,
        *,
        default_workspace_bytes: int,
        environ: Mapping[str, str] | None = None,
    ) -> "VllmAttentionConfig":
        if default_workspace_bytes <= 0:
            raise ValueError("vLLM default FlashInfer workspace must be positive")
        values = os.environ if environ is None else environ
        native_enabled = _boolean(values, "NTA_VLLM_NATIVE")
        serving_tier = validate_vllm_attention_tier(values)
        raw_workspace = values.get("FLASHINFER_WORKSPACE_BASE", "").strip()
        workspace = (
            Path(raw_workspace).expanduser().resolve() if raw_workspace else None
        )
        if native_enabled and workspace is None:
            raise RuntimeError(
                "NTA vLLM native attention requires FLASHINFER_WORKSPACE_BASE; "
                "run tools/jit/activate.py --flashinfer-hook first"
            )
        return cls(
            native_enabled=native_enabled,
            serving_tier=serving_tier,
            profile_cpu=_boolean(values, "NTA_PROFILE_CPU"),
            verify_execution=_boolean(values, "NTA_VERIFY_EXECUTION"),
            verify_transfer=_boolean(values, "NTA_VLLM_VERIFY_TRANSFER"),
            compare_stock=_boolean(values, "NTA_VLLM_COMPARE_STOCK"),
            allow_stock_fallback=_boolean(values, "NTA_VLLM_ALLOW_STOCK_FALLBACK"),
            workspace_base=workspace,
            workspace_bytes=_positive(
                values,
                "NTA_VLLM_FLASHINFER_WORKSPACE_BYTES",
                default_workspace_bytes,
            ),
            decode_module_override=(
                values.get("NTA_VLLM_DECODE_MODULE", "").strip() or None
            ),
            prefill_module_override=(
                values.get("NTA_VLLM_PREFILL_MODULE", "").strip() or None
            ),
            host_copy_blocks_per_group=_positive(
                values, "NTA_VLLM_HOST_COPY_BLOCKS_PER_GROUP", 2
            ),
        )

    def require_workspace(self) -> Path:
        if self.workspace_base is None:
            raise RuntimeError(
                "NTA vLLM native attention has no configured FlashInfer workspace"
            )
        return self.workspace_base

    def module_override(self, kind: str) -> str | None:
        if kind == "decode":
            return self.decode_module_override
        if kind == "prefill":
            return self.prefill_module_override
        raise ValueError(f"unknown vLLM attention phase {kind!r}")
