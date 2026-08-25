"""Engine-neutral owner for a serving tier and its native runtime.

Framework adapters provide capacities and a device ordinal.  This module owns
the construction and destruction order of the selected tier transport and the
HostRuntime that borrows it.  A framework cannot accidentally close a
transport before the native runtime has quiesced.
"""

from __future__ import annotations

from dataclasses import dataclass
import numbers
import os

from .runtime import Runtime, RuntimeConfig, TierKind
from .tier import ServingTier, ServingTierConfig, ServingTierService


_UINT64_MAX = (1 << 64) - 1
_UINT32_MAX = (1 << 32) - 1
_INT32_MAX = (1 << 31) - 1


def _bounded_integer(value: int, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or result > maximum:
        raise ValueError(f"{name} is outside [{minimum}, {maximum}]")
    return result


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
            _bounded_integer(
                getattr(self, name), name, minimum=1, maximum=_UINT32_MAX
            )
        _bounded_integer(
            self.device_ordinal,
            "device_ordinal",
            minimum=-1,
            maximum=_INT32_MAX,
        )
        _bounded_integer(
            self.staging_byte_capacity,
            "staging_byte_capacity",
            minimum=0,
            maximum=_UINT64_MAX,
        )

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
        runtime: Runtime | None = None
        try:
            runtime = Runtime(
                runtime_config.native(),
                nvme=tier.nvme,
                cxl=tier.cxl,
            )
            native_kind = {
                ServingTier.HOST_STAGED: TierKind.HOST_STAGED,
                ServingTier.NVME: TierKind.NVME,
                ServingTier.CXL_DAX: TierKind.CXL,
            }[tier_config.tier]
            descriptor = runtime.tier_descriptor(native_kind)
            if (
                not descriptor.active
                or descriptor.capabilities != tier.contract.capabilities
                or descriptor.protocol_owner is not tier.contract.protocol_owner
                or descriptor.payload_owner is not tier.contract.payload_owner
                or descriptor.transfer_destination_owner
                is not tier.contract.transfer_destination_owner
            ):
                raise RuntimeError(
                    "native tier descriptor diverges from the selected resource contract"
                )
        except BaseException:
            try:
                if runtime is not None:
                    runtime.close()
            finally:
                tier.close()
            raise
        if runtime is None:
            raise RuntimeError("runtime construction returned no owner")
        return cls(tier, runtime, runtime_config)

    def close(self) -> None:
        """Quiesce the consumer runtime before releasing tier transports."""
        if self._closed:
            return
        self._closed = True
        first_error: BaseException | None = None
        try:
            self.runtime.close()
        except BaseException as error:
            first_error = error
        try:
            self.tier.close()
        except BaseException as error:
            if first_error is None:
                first_error = error
        if first_error is not None:
            raise RuntimeError("serving runtime resource teardown failed") from first_error

    def __del__(self) -> None:
        # NtaFlashInferAttnBackend can fail during partially initialized
        # construction after assigning this owner.  Preserve the same
        # runtime-before-transport order on that exceptional path.
        try:
            self.close()
        except Exception:
            pass
