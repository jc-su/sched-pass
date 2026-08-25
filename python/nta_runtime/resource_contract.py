"""Dependency-free contract for the memory/tier resource boundary.

The native ABI and the framework adapters must agree on two different classes
of facts: what a resource is allowed to do, and its explicit protocol/payload/
destination ownership. Keeping this contract independent of CUDA bindings lets
catalog validation, adapters, and artifact checks use the same vocabulary
without opening a transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Flag, Enum, auto


class ResourceKind(str, Enum):
    HBM = "hbm"
    HOST_MAPPED = "host_mapped"
    HOST_STAGED = "host_staged"
    NVME = "nvme"
    CXL_DAX = "cxl_dax"


class ResourceCapability(Flag):
    DIRECT_ADDRESS = auto()
    DEVICE_INITIATED = auto()
    HOST_REGISTERED = auto()
    PERSISTENT_STORAGE = auto()
    INDEXED_TRANSFER = auto()


class ResourceOwner(str, Enum):
    ENGINE = "engine"
    RUNTIME = "runtime"
    TRANSPORT = "transport"


@dataclass(frozen=True)
class ResourceContract:
    """Immutable setup/data-path contract for one resource kind.

    ``requires_catalog`` and ``requires_endpoint`` describe setup-time
    identity.  ``steady_state_path`` describes the actual consumer path and
    is deliberately not allowed to imply a fallback through host memory.
    """

    kind: ResourceKind
    capabilities: ResourceCapability
    # These are deliberately separate ownership facts.  ``protocol_owner``
    # controls the backend protocol; ``payload_owner`` owns the bytes in the
    # selected tier; ``transfer_destination_owner`` owns a temporary/device
    # destination when the data path materializes one.  ``None`` means the
    # selected resource is consumed directly and has no runtime-owned staging
    # destination.  This is more precise than a set of possible allocators,
    # which cannot describe the owner of one actual transfer.
    protocol_owner: ResourceOwner
    payload_owner: ResourceOwner
    transfer_destination_owner: ResourceOwner | None
    directory_owner: ResourceOwner
    steady_state_path: str
    requires_catalog: bool
    requires_endpoint: bool
    direct_device_visible: bool
    uses_host_proxy: bool

    def __post_init__(self) -> None:
        if not isinstance(self.protocol_owner, ResourceOwner):
            raise TypeError("resource protocol owner must be typed")
        if not isinstance(self.payload_owner, ResourceOwner):
            raise TypeError("resource payload owner must be typed")
        if self.transfer_destination_owner is not None and not isinstance(
            self.transfer_destination_owner, ResourceOwner
        ):
            raise TypeError("resource transfer destination owner must be typed")
        if not self.steady_state_path:
            raise ValueError("resource steady-state path must be named")
        if self.directory_owner is not ResourceOwner.RUNTIME:
            raise ValueError("native resource directories must be runtime-owned")
        if self.requires_catalog != self.requires_endpoint:
            raise ValueError(
                "physical resource catalog and endpoint requirements must match"
            )
        direct = bool(self.capabilities & ResourceCapability.DIRECT_ADDRESS)
        if direct != self.direct_device_visible:
            raise ValueError(
                "direct-address capability must match device visibility"
            )
        if self.kind is ResourceKind.HOST_STAGED:
            if not self.uses_host_proxy:
                raise ValueError("host-staged resources require a host proxy")
            if not (self.capabilities & ResourceCapability.INDEXED_TRANSFER):
                raise ValueError("host-staged resources require indexed transfer")
        elif self.uses_host_proxy:
            raise ValueError("only host-staged resources may use a host proxy")
        if self.kind in (ResourceKind.NVME, ResourceKind.CXL_DAX):
            if (
                not self.requires_catalog
                or self.protocol_owner is not ResourceOwner.TRANSPORT
                or self.payload_owner is not ResourceOwner.TRANSPORT
            ):
                raise ValueError("physical resources require transport ownership")
        if self.kind is ResourceKind.HOST_STAGED:
            if self.protocol_owner is not ResourceOwner.RUNTIME:
                raise ValueError("host-staged protocol must be runtime-owned")
            if self.payload_owner is not ResourceOwner.ENGINE:
                raise ValueError("host-staged payload must be engine-owned")
            if self.transfer_destination_owner is not ResourceOwner.RUNTIME:
                raise ValueError("host-staged destination must be runtime-owned")
        if self.kind in (ResourceKind.HBM, ResourceKind.HOST_MAPPED):
            if self.protocol_owner is not ResourceOwner.ENGINE:
                raise ValueError("resident resources must be engine-protocol-owned")
            if self.payload_owner is not ResourceOwner.ENGINE:
                raise ValueError("resident resources must be engine-owned")
            if self.transfer_destination_owner is not None:
                raise ValueError("direct resident resources cannot have a staging destination")
        if self.kind is ResourceKind.CXL_DAX:
            if self.transfer_destination_owner is not None:
                raise ValueError("direct CXL resources cannot have a staging destination")
        if self.kind is ResourceKind.NVME:
            if self.transfer_destination_owner is not ResourceOwner.TRANSPORT:
                raise ValueError("NVMe HBM destination must be transport-owned")

    @property
    def physical(self) -> bool:
        return self.requires_endpoint

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "capabilities": [
                capability.name.lower()
                for capability in ResourceCapability
                if self.capabilities & capability
            ],
            "protocol_owner": self.protocol_owner.value,
            "payload_owner": self.payload_owner.value,
            "transfer_destination_owner": (
                None
                if self.transfer_destination_owner is None
                else self.transfer_destination_owner.value
            ),
            "directory_owner": self.directory_owner.value,
            "steady_state_path": self.steady_state_path,
            "requires_catalog": self.requires_catalog,
            "requires_endpoint": self.requires_endpoint,
            "direct_device_visible": self.direct_device_visible,
            "uses_host_proxy": self.uses_host_proxy,
        }


_CONTRACTS = {
    ResourceKind.HBM: ResourceContract(
        ResourceKind.HBM,
        ResourceCapability.DIRECT_ADDRESS,
        ResourceOwner.ENGINE,
        ResourceOwner.ENGINE,
        None,
        ResourceOwner.RUNTIME,
        "gpu_hbm_load",
        False,
        False,
        True,
        False,
    ),
    ResourceKind.HOST_MAPPED: ResourceContract(
        ResourceKind.HOST_MAPPED,
        ResourceCapability.DIRECT_ADDRESS | ResourceCapability.HOST_REGISTERED,
        ResourceOwner.ENGINE,
        ResourceOwner.ENGINE,
        None,
        ResourceOwner.RUNTIME,
        "gpu_mapped_host_load",
        False,
        False,
        True,
        False,
    ),
    ResourceKind.HOST_STAGED: ResourceContract(
        ResourceKind.HOST_STAGED,
        ResourceCapability.INDEXED_TRANSFER,
        ResourceOwner.RUNTIME,
        ResourceOwner.ENGINE,
        ResourceOwner.RUNTIME,
        ResourceOwner.RUNTIME,
        "host_indexed_copy",
        False,
        False,
        False,
        True,
    ),
    ResourceKind.NVME: ResourceContract(
        ResourceKind.NVME,
        ResourceCapability.DEVICE_INITIATED | ResourceCapability.PERSISTENT_STORAGE,
        ResourceOwner.TRANSPORT,
        ResourceOwner.TRANSPORT,
        ResourceOwner.TRANSPORT,
        ResourceOwner.RUNTIME,
        "gpu_owned_nvme_to_hbm",
        True,
        True,
        False,
        False,
    ),
    ResourceKind.CXL_DAX: ResourceContract(
        ResourceKind.CXL_DAX,
        ResourceCapability.DIRECT_ADDRESS
        | ResourceCapability.HOST_REGISTERED,
        ResourceOwner.TRANSPORT,
        ResourceOwner.TRANSPORT,
        None,
        ResourceOwner.RUNTIME,
        "cuda_visible_cxl_direct",
        True,
        True,
        True,
        False,
    ),
}


def resource_contract(kind: ResourceKind | str) -> ResourceContract:
    """Return the immutable contract for a resource kind."""

    try:
        resolved = kind if isinstance(kind, ResourceKind) else ResourceKind(kind)
        return _CONTRACTS[resolved]
    except (KeyError, ValueError) as error:
        raise ValueError(f"unsupported resource kind: {kind!r}") from error
