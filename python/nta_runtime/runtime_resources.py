"""Engine-neutral owner for a serving tier and its native runtime.

Framework adapters provide capacities and a device ordinal.  This module owns
the construction and destruction order of the selected tier transport and the
HostRuntime that borrows it.  A framework cannot accidentally close a
transport before the native runtime has quiesced.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

from .runtime import Runtime, RuntimeConfig, TierKind
from .tier import ServingTier, ServingTierConfig, ServingTierService


_UINT64_MAX = (1 << 64) - 1


def _nonnegative_environment(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a nonnegative integer") from error
    if value < 0 or value > _UINT64_MAX:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class RuntimeResourceConfig:
    request_capacity: int
    object_capacity: int
    intent_capacity: int
    work_ticket_capacity: int
    tenant_capacity: int
    device_ordinal: int
    max_dependencies_per_work_ticket: int = 8
    staging_byte_capacity: int = _UINT64_MAX

    def __post_init__(self) -> None:
        for name in (
            "request_capacity",
            "object_capacity",
            "intent_capacity",
            "work_ticket_capacity",
            "tenant_capacity",
            "max_dependencies_per_work_ticket",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.device_ordinal < -1:
            raise ValueError("device_ordinal must be -1 or nonnegative")
        if not 0 <= self.staging_byte_capacity <= _UINT64_MAX:
            raise ValueError("staging_byte_capacity is outside uint64")

    def native(self) -> RuntimeConfig:
        return RuntimeConfig(
            request_capacity=self.request_capacity,
            object_capacity=self.object_capacity,
            intent_capacity=self.intent_capacity,
            work_ticket_capacity=self.work_ticket_capacity,
            max_dependencies_per_work_ticket=self.max_dependencies_per_work_ticket,
            device_ordinal=self.device_ordinal,
            tenant_capacity=self.tenant_capacity,
            staging_byte_capacity=self.staging_byte_capacity,
        )

    @classmethod
    def with_environment_staging_limit(
        cls, *,
        request_capacity: int,
        object_capacity: int,
        intent_capacity: int,
        work_ticket_capacity: int,
        tenant_capacity: int,
        device_ordinal: int,
        max_dependencies_per_work_ticket: int = 8,
    ) -> "RuntimeResourceConfig":
        staging_limit = _nonnegative_environment(
            "NTA_STAGING_BYTE_CAPACITY", _UINT64_MAX
        )
        return cls(
            request_capacity=request_capacity,
            object_capacity=object_capacity,
            intent_capacity=intent_capacity,
            work_ticket_capacity=work_ticket_capacity,
            tenant_capacity=tenant_capacity,
            device_ordinal=device_ordinal,
            max_dependencies_per_work_ticket=max_dependencies_per_work_ticket,
            # Native zero means unlimited; expose the normalized value in
            # stats so policy provenance matches the actual runtime contract.
            staging_byte_capacity=staging_limit or _UINT64_MAX,
        )


class ServingRuntimeResources:
    """Own one selected tier service and the runtime that uses it."""

    def __init__(
        self,
        tier: ServingTierService,
        runtime: Runtime,
        config: RuntimeResourceConfig,
    ) -> None:
        self.tier = tier
        self.runtime = runtime
        self.config = config
        self._closed = False

    @classmethod
    def open(
        cls,
        *,
        tier_config: ServingTierConfig,
        runtime_config: RuntimeResourceConfig,
    ) -> "ServingRuntimeResources":
        tier = ServingTierService(tier_config)
        try:
            runtime = Runtime(
                runtime_config.native(),
                nvme=tier.nvme,
                cxl=tier.cxl,
            )
        except BaseException:
            tier.close()
            raise
        native_kind = {
            ServingTier.HOST_STAGED: TierKind.HOST_STAGED,
            ServingTier.NVME: TierKind.NVME,
            ServingTier.CXL_DAX: TierKind.CXL,
        }[tier_config.tier]
        descriptor = runtime.tier_descriptor(native_kind)
        if (
            not descriptor.active
            or descriptor.capabilities != tier.contract.capabilities
        ):
            try:
                runtime.close()
            finally:
                tier.close()
            raise RuntimeError(
                "native tier descriptor diverges from the selected resource contract"
            )
        return cls(tier, runtime, runtime_config)

    def close(self) -> None:
        """Quiesce the consumer runtime before releasing tier transports."""
        if self._closed:
            return
        self._closed = True
        try:
            self.runtime.close()
        finally:
            self.tier.close()

    def __del__(self) -> None:
        # NtaFlashInferAttnBackend can fail during partially initialized
        # construction after assigning this owner.  Preserve the same
        # runtime-before-transport order on that exceptional path.
        try:
            self.close()
        except Exception:
            pass
