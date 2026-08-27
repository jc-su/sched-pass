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


class ResourceAddressSpace(str, Enum):
    HBM = "hbm"
    HOST_MAPPED = "host_mapped"
    NVME = "nvme"
    CXL_DAX = "cxl_dax"


class ResourcePath(str, Enum):
    RESIDENT = "resident"
    DIRECT = "direct"
    MATERIALIZED = "materialized"


@dataclass(frozen=True)
class ResourceContract:
    """Immutable setup/data-path contract for one resource kind.

    ``requires_catalog`` and ``requires_endpoint`` describe setup-time
    identity.  ``steady_state_path`` describes the actual consumer path and
    is deliberately not allowed to imply a fallback through host memory.
    """

    kind: ResourceKind
    capabilities: ResourceCapability
    source_address_space: ResourceAddressSpace
    numerical_address_space: ResourceAddressSpace
    path: ResourcePath
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
    mapping_owner: ResourceOwner | None
    # Owner of the address that the numerical consumer dereferences after the
    # dependency becomes ready.  This is intentionally distinct from payload
    # and mapping ownership: NVMe owns persistent bytes and its IOMMU mapping,
    # but FlashInfer consumes an engine-owned HBM destination.  CXL-DAX is the
    # opposite shape: the transport-owned mapping is itself the numerical
    # address.  Framework adapters must validate this field before opening a
    # physical resource; readiness for one address must never gate loads from
    # another.
    numerical_address_owner: ResourceOwner
    directory_owner: ResourceOwner
    steady_state_path: str
    requires_catalog: bool
    requires_endpoint: bool
    direct_device_visible: bool
    uses_host_proxy: bool

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ResourceKind):
            raise TypeError("resource kind must be a ResourceKind value")
        if not isinstance(self.capabilities, ResourceCapability):
            raise TypeError("resource capabilities must be ResourceCapability flags")
        if not isinstance(self.source_address_space, ResourceAddressSpace):
            raise TypeError("resource source address space must be typed")
        if not isinstance(self.numerical_address_space, ResourceAddressSpace):
            raise TypeError("resource numerical address space must be typed")
        if not isinstance(self.path, ResourcePath):
            raise TypeError("resource path must be typed")
        if not isinstance(self.protocol_owner, ResourceOwner):
            raise TypeError("resource protocol owner must be typed")
        if not isinstance(self.payload_owner, ResourceOwner):
            raise TypeError("resource payload owner must be typed")
        if self.transfer_destination_owner is not None and not isinstance(
            self.transfer_destination_owner, ResourceOwner
        ):
            raise TypeError("resource transfer destination owner must be typed")
        if self.mapping_owner is not None and not isinstance(
            self.mapping_owner, ResourceOwner
        ):
            raise TypeError("resource mapping owner must be typed")
        if not isinstance(self.numerical_address_owner, ResourceOwner):
            raise TypeError("resource numerical address owner must be typed")
        for field in (
            "requires_catalog",
            "requires_endpoint",
            "direct_device_visible",
            "uses_host_proxy",
        ):
            if type(getattr(self, field)) is not bool:
                raise TypeError(f"resource contract {field} must be bool")
        if not isinstance(self.steady_state_path, str):
            raise TypeError("resource steady-state path must be a string")
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
            raise ValueError("direct-address capability must match device visibility")
        if self.path is ResourcePath.RESIDENT:
            if (
                self.source_address_space is not self.numerical_address_space
                or self.transfer_destination_owner is not None
                or self.requires_endpoint
            ):
                raise ValueError("resident resources must already be the numerical bytes")
        elif self.path is ResourcePath.DIRECT:
            if (
                self.source_address_space is not self.numerical_address_space
                or self.transfer_destination_owner is not None
                or not self.direct_device_visible
            ):
                raise ValueError("direct resources must expose their source numerically")
        elif self.path is ResourcePath.MATERIALIZED:
            if (
                self.source_address_space is self.numerical_address_space
                or self.transfer_destination_owner is None
                or self.direct_device_visible
            ):
                raise ValueError(
                    "materialized resources require a distinct non-direct source"
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
                or self.mapping_owner is not ResourceOwner.TRANSPORT
            ):
                raise ValueError("physical resources require transport ownership")
        if self.kind is ResourceKind.HOST_STAGED:
            if self.protocol_owner is not ResourceOwner.RUNTIME:
                raise ValueError("host-staged protocol must be runtime-owned")
            if self.payload_owner is not ResourceOwner.ENGINE:
                raise ValueError("host-staged payload must be engine-owned")
            if self.transfer_destination_owner is not ResourceOwner.ENGINE:
                raise ValueError("host-staged destination must be engine-owned")
            if self.mapping_owner is not None:
                raise ValueError("host-staged resources have no mapping lease")
        if self.kind in (ResourceKind.HBM, ResourceKind.HOST_MAPPED):
            if self.protocol_owner is not ResourceOwner.ENGINE:
                raise ValueError("resident resources must be engine-protocol-owned")
            if self.payload_owner is not ResourceOwner.ENGINE:
                raise ValueError("resident resources must be engine-owned")
            if self.transfer_destination_owner is not None:
                raise ValueError(
                    "direct resident resources cannot have a staging destination"
                )
        if self.kind is ResourceKind.HBM and self.mapping_owner is not None:
            raise ValueError("HBM resources have no separate mapping lease")
        if (
            self.kind is ResourceKind.HOST_MAPPED
            and self.mapping_owner is not ResourceOwner.ENGINE
        ):
            raise ValueError("host-mapped addressability must be engine-owned")
        if self.kind is ResourceKind.CXL_DAX:
            if self.transfer_destination_owner is not None:
                raise ValueError(
                    "direct CXL resources cannot have a staging destination"
                )
            if self.numerical_address_owner is not ResourceOwner.TRANSPORT:
                raise ValueError(
                    "direct CXL numerical addresses must be transport-owned"
                )
        if self.kind is ResourceKind.NVME:
            if self.transfer_destination_owner is not ResourceOwner.ENGINE:
                raise ValueError("NVMe HBM destination must be engine-owned")
        if self.transfer_destination_owner is not None:
            if self.numerical_address_owner is not self.transfer_destination_owner:
                raise ValueError(
                    "the numerical address owner must own the transfer destination"
                )
        elif self.kind is not ResourceKind.CXL_DAX:
            if self.numerical_address_owner is not self.payload_owner:
                raise ValueError(
                    "direct numerical addresses must be owned by the payload owner"
                )

    @property
    def physical(self) -> bool:
        return self.requires_endpoint

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "source_address_space": self.source_address_space.value,
            "numerical_address_space": self.numerical_address_space.value,
            "path": self.path.value,
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
            "mapping_owner": (
                None if self.mapping_owner is None else self.mapping_owner.value
            ),
            "numerical_address_owner": self.numerical_address_owner.value,
            "directory_owner": self.directory_owner.value,
            "steady_state_path": self.steady_state_path,
            "requires_catalog": self.requires_catalog,
            "requires_endpoint": self.requires_endpoint,
            "direct_device_visible": self.direct_device_visible,
            "uses_host_proxy": self.uses_host_proxy,
        }


_CONTRACTS = {
    ResourceKind.HBM: ResourceContract(
        kind=ResourceKind.HBM,
        capabilities=ResourceCapability.DIRECT_ADDRESS,
        source_address_space=ResourceAddressSpace.HBM,
        numerical_address_space=ResourceAddressSpace.HBM,
        path=ResourcePath.RESIDENT,
        protocol_owner=ResourceOwner.ENGINE,
        payload_owner=ResourceOwner.ENGINE,
        transfer_destination_owner=None,
        mapping_owner=None,
        numerical_address_owner=ResourceOwner.ENGINE,
        directory_owner=ResourceOwner.RUNTIME,
        steady_state_path="gpu_hbm_load",
        requires_catalog=False,
        requires_endpoint=False,
        direct_device_visible=True,
        uses_host_proxy=False,
    ),
    ResourceKind.HOST_MAPPED: ResourceContract(
        kind=ResourceKind.HOST_MAPPED,
        capabilities=ResourceCapability.DIRECT_ADDRESS
        | ResourceCapability.HOST_REGISTERED,
        source_address_space=ResourceAddressSpace.HOST_MAPPED,
        numerical_address_space=ResourceAddressSpace.HOST_MAPPED,
        path=ResourcePath.DIRECT,
        protocol_owner=ResourceOwner.ENGINE,
        payload_owner=ResourceOwner.ENGINE,
        transfer_destination_owner=None,
        mapping_owner=ResourceOwner.ENGINE,
        numerical_address_owner=ResourceOwner.ENGINE,
        directory_owner=ResourceOwner.RUNTIME,
        steady_state_path="gpu_mapped_host_load",
        requires_catalog=False,
        requires_endpoint=False,
        direct_device_visible=True,
        uses_host_proxy=False,
    ),
    ResourceKind.HOST_STAGED: ResourceContract(
        kind=ResourceKind.HOST_STAGED,
        capabilities=ResourceCapability.INDEXED_TRANSFER,
        source_address_space=ResourceAddressSpace.HOST_MAPPED,
        numerical_address_space=ResourceAddressSpace.HBM,
        path=ResourcePath.MATERIALIZED,
        protocol_owner=ResourceOwner.RUNTIME,
        payload_owner=ResourceOwner.ENGINE,
        transfer_destination_owner=ResourceOwner.ENGINE,
        mapping_owner=None,
        numerical_address_owner=ResourceOwner.ENGINE,
        directory_owner=ResourceOwner.RUNTIME,
        steady_state_path="host_indexed_copy_to_engine_hbm",
        requires_catalog=False,
        requires_endpoint=False,
        direct_device_visible=False,
        uses_host_proxy=True,
    ),
    ResourceKind.NVME: ResourceContract(
        kind=ResourceKind.NVME,
        capabilities=ResourceCapability.DEVICE_INITIATED
        | ResourceCapability.PERSISTENT_STORAGE,
        source_address_space=ResourceAddressSpace.NVME,
        numerical_address_space=ResourceAddressSpace.HBM,
        path=ResourcePath.MATERIALIZED,
        protocol_owner=ResourceOwner.TRANSPORT,
        payload_owner=ResourceOwner.TRANSPORT,
        transfer_destination_owner=ResourceOwner.ENGINE,
        mapping_owner=ResourceOwner.TRANSPORT,
        numerical_address_owner=ResourceOwner.ENGINE,
        directory_owner=ResourceOwner.RUNTIME,
        steady_state_path="nvme_peer_dma_to_engine_hbm",
        requires_catalog=True,
        requires_endpoint=True,
        direct_device_visible=False,
        uses_host_proxy=False,
    ),
    ResourceKind.CXL_DAX: ResourceContract(
        kind=ResourceKind.CXL_DAX,
        capabilities=ResourceCapability.DIRECT_ADDRESS
        | ResourceCapability.HOST_REGISTERED,
        source_address_space=ResourceAddressSpace.CXL_DAX,
        numerical_address_space=ResourceAddressSpace.CXL_DAX,
        path=ResourcePath.DIRECT,
        protocol_owner=ResourceOwner.TRANSPORT,
        payload_owner=ResourceOwner.TRANSPORT,
        transfer_destination_owner=None,
        mapping_owner=ResourceOwner.TRANSPORT,
        numerical_address_owner=ResourceOwner.TRANSPORT,
        directory_owner=ResourceOwner.RUNTIME,
        steady_state_path="cuda_visible_cxl_direct",
        requires_catalog=True,
        requires_endpoint=True,
        direct_device_visible=True,
        uses_host_proxy=False,
    ),
}


def resource_contract(kind: ResourceKind | str) -> ResourceContract:
    """Return the immutable contract for a resource kind."""

    try:
        resolved = kind if isinstance(kind, ResourceKind) else ResourceKind(kind)
        return _CONTRACTS[resolved]
    except (KeyError, ValueError) as error:
        raise ValueError(f"unsupported resource kind: {kind!r}") from error


def require_numerical_binding(
    contract: ResourceContract,
    expected_owner: ResourceOwner,
    expected_address_space: ResourceAddressSpace,
    allowed_paths: frozenset[ResourcePath],
    *,
    consumer: str,
) -> None:
    """Reject a resource whose ready address is not the consumer's pointer.

    A dependency protocol can be internally correct while still guarding the
    wrong numerical address.  Framework paged-KV tables currently name
    engine-owned storage; accepting a direct transport mapping would therefore
    produce a false readiness proof and potentially stale numerical results.
    """

    if not isinstance(contract, ResourceContract):
        raise TypeError("numerical consumer validation requires a resource contract")
    if not isinstance(expected_owner, ResourceOwner):
        raise TypeError("expected numerical address owner must be typed")
    if not isinstance(expected_address_space, ResourceAddressSpace):
        raise TypeError("expected numerical address space must be typed")
    if not isinstance(allowed_paths, frozenset) or not allowed_paths or any(
        not isinstance(path, ResourcePath) for path in allowed_paths
    ):
        raise TypeError("allowed numerical resource paths must be a non-empty frozenset")
    if not isinstance(consumer, str) or not consumer.strip():
        raise ValueError("numerical consumer must be named")
    if contract.numerical_address_owner is not expected_owner:
        raise RuntimeError(
            f"{consumer} cannot consume {contract.kind.value}: its numerical "
            f"pointer is {expected_owner.value}-owned, but the ready address is "
            f"{contract.numerical_address_owner.value}-owned"
        )
    if contract.numerical_address_space is not expected_address_space:
        raise RuntimeError(
            f"{consumer} cannot consume {contract.kind.value}: its numerical "
            f"pointer names {expected_address_space.value}, but readiness names "
            f"{contract.numerical_address_space.value}"
        )
    if contract.path not in allowed_paths:
        raise RuntimeError(
            f"{consumer} has no {contract.path.value} implementation for "
            f"{contract.kind.value}"
        )
