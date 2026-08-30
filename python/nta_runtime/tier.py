"""Engine-neutral serving tier selection and exact page ownership.

The serving adapters do not infer storage locations from a framework's host
cache.  A tier catalog is an explicit, immutable contract produced by the
storage preparation step.  This keeps the compiler/runtime mechanism exact:
the operator supplies requested page identities, while this module resolves
those identities to byte extents owned by one physical tier.

There is intentionally no implicit fallback between tiers.  Resident HBM is
the safe default because it is the only source already named by an unmodified
framework KV table.  Host-staged is a distinct materialization path and must
never be used as a label for resident HBM.  Selecting a physical tier requires
both its endpoint and a validated catalog.  A missing capability is therefore
an operator/configuration error, not a silent change in the measured path.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
import enum
import hashlib
import json
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

from .abi import bounded_integer
from .hbm_registration import (
    DescribedHbmDestination,
    HbmDestinationSlice,
    coalesce_hbm_destinations,
)
from .nvme_granularity import NvmeTransferServiceModel
from .resource_contract import (
    ResourceKind,
    resource_contract,
)


_UINT64_MAX = (1 << 64) - 1
_UINT32_MAX = (1 << 32) - 1
_INT32_MAX = (1 << 31) - 1


class ServingTier(str, enum.Enum):
    HBM = "hbm"
    HOST_MAPPED = "host_mapped"
    HOST_STAGED = "host_staged"
    NVME = "nvme"
    CXL_DAX = "cxl_dax"


class NvmeHbmBackendRequirement(str, enum.Enum):
    """Setup-plane mapping backend accepted by one NVMe deployment.

    This is deliberately independent of the native CUDA binding so frontend
    configuration and artifact inspection remain import-safe.  ``AUTO`` means
    either direct-HBM backend is acceptable; it never permits host-mapped DMA.
    """

    AUTO = "auto"
    CUDA_DMA_BUF_IOAS = "cuda-dmabuf-ioas"
    NVIDIA_PEER_PAGES = "nvidia-peer-pages"


_RESOURCE_KIND_FOR_SERVING_TIER = {
    ServingTier.HBM: ResourceKind.HBM,
    ServingTier.HOST_MAPPED: ResourceKind.HOST_MAPPED,
    ServingTier.HOST_STAGED: ResourceKind.HOST_STAGED,
    ServingTier.NVME: ResourceKind.NVME,
    ServingTier.CXL_DAX: ResourceKind.CXL_DAX,
}

PHYSICAL_SERVING_TIERS = frozenset((ServingTier.NVME, ServingTier.CXL_DAX))


def _require_nvme_hbm_backend(
    requirement: NvmeHbmBackendRequirement, selected: str
) -> None:
    if not isinstance(requirement, NvmeHbmBackendRequirement):
        raise TypeError("NVMe HBM backend requirement must be typed")
    valid_selected = {
        NvmeHbmBackendRequirement.CUDA_DMA_BUF_IOAS.value,
        NvmeHbmBackendRequirement.NVIDIA_PEER_PAGES.value,
    }
    if selected not in valid_selected:
        raise RuntimeError(f"NVMe selected an unknown HBM mapping backend: {selected}")
    if (
        requirement is not NvmeHbmBackendRequirement.AUTO
        and selected != requirement.value
    ):
        raise RuntimeError(
            "NVMe HBM mapping backend does not satisfy the deployment contract: "
            f"required={requirement.value} selected={selected}"
        )


@dataclass(frozen=True)
class PageExtent:
    """A contiguous exact byte range in a tier's address space."""

    offset: int
    bytes: int

    def __post_init__(self) -> None:
        offset = bounded_integer(
            self.offset,
            "tier extent offset",
            minimum=0,
            maximum=_UINT64_MAX,
        )
        size = bounded_integer(
            self.bytes,
            "tier extent bytes",
            minimum=1,
            maximum=(1 << 32) - 1,
        )
        if size > _UINT64_MAX - offset:
            raise ValueError("tier extent exceeds the native address-space limit")
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "bytes", size)


@dataclass(frozen=True)
class PageTransferRun:
    """One maximal source/destination-contiguous physical page transfer.

    ``destination_first`` is a framework-owned page/block index.  The tier
    owns only the immutable source extent; translating the destination index
    to an HBM address remains the framework adapter's responsibility.
    """

    source: PageExtent
    destination_first: int
    row_count: int

    def __post_init__(self) -> None:
        destination = bounded_integer(
            self.destination_first,
            "page-transfer destination",
            minimum=0,
            maximum=_UINT32_MAX,
        )
        rows = bounded_integer(
            self.row_count,
            "page-transfer row count",
            minimum=1,
            maximum=_UINT32_MAX,
        )
        if destination > _UINT32_MAX - (rows - 1):
            raise ValueError("page-transfer destination range exceeds uint32")
        object.__setattr__(self, "destination_first", destination)
        object.__setattr__(self, "row_count", rows)


@dataclass(frozen=True)
class NvmeHbmPreparation:
    """Owned setup result for disjoint framework destination registrations."""

    regions: Mapping[Hashable, Any]
    destination_count: int
    destination_bytes: int
    registration_count: int
    registration_bytes: int


@dataclass(frozen=True)
class _PageRecord:
    storage_key: str
    ordinal: int
    layer: int
    components: Mapping[str, PageExtent]


class TierPageCatalog:
    """Stable storage-key to physical-byte directory.

    Framework device page numbers are destination slots and may change on
    every cache load.  They are deliberately absent from this catalog.  A
    connector binds a stable content/storage key to the current destination;
    the catalog maps that key to a dense physical ordinal and exact per-layer
    typed component extents. Consecutive ordinals may be coalesced into one
    transfer only when the selected component extents are also contiguous.
    """

    SCHEMA = 2
    FORMAT = "typed-components-v1"

    def __init__(
        self,
        *,
        tier: ServingTier,
        records: tuple[_PageRecord, ...],
        namespace: str,
        page_tokens: int,
        layer_count: int,
        components: tuple[str, ...],
        alignment_bytes: int = 4096,
        window_bytes: int | None = None,
        digest: str = "",
    ) -> None:
        if not isinstance(tier, ServingTier):
            raise TypeError("tier catalog tier must be a ServingTier value")
        if tier not in PHYSICAL_SERVING_TIERS:
            raise ValueError("tier page catalogs are only valid for physical tiers")
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("tier catalog namespace must be non-empty")
        if len(namespace.encode("utf-8")) > 1024:
            raise ValueError("tier catalog namespace is too long")
        page_tokens = _positive_int(page_tokens, "catalog page_tokens")
        layer_count = _positive_int(layer_count, "catalog layer_count")
        if (
            not components
            or len(set(components)) != len(components)
            or any(
                not isinstance(component, str)
                or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", component) is None
                for component in components
            )
        ):
            raise ValueError(
                "tier catalog components must be unique lower-case identifiers"
            )
        component_set = set(components)
        alignment_bytes = _catalog_uint64(
            alignment_bytes, "catalog alignment_bytes", minimum=1
        )
        if alignment_bytes & (alignment_bytes - 1):
            raise ValueError("catalog alignment_bytes must be a positive power of two")
        if window_bytes is not None:
            window_bytes = _catalog_uint64(
                window_bytes, "catalog window_bytes", minimum=1
            )
        if not records:
            raise ValueError("tier page catalog cannot be empty")
        by_coordinate: dict[tuple[int, int], _PageRecord] = {}
        ordinal_by_storage_key: dict[str, int] = {}
        storage_key_by_ordinal: dict[int, str] = {}
        layers_by_ordinal: dict[int, set[int]] = {}
        ranges: list[tuple[int, int, tuple[int, int], str]] = []
        for record in records:
            if not isinstance(record.storage_key, str) or not record.storage_key:
                raise ValueError("catalog storage keys must be non-empty strings")
            if len(record.storage_key.encode("utf-8")) > 4096:
                raise ValueError("catalog storage key is too long")
            ordinal = bounded_integer(
                record.ordinal,
                "catalog ordinal",
                minimum=0,
                maximum=_UINT32_MAX,
            )
            layer = bounded_integer(
                record.layer,
                "catalog layer",
                minimum=0,
                maximum=_UINT32_MAX,
            )
            if layer >= layer_count:
                raise ValueError("catalog entry layer exceeds layer_count")
            coordinate = (layer, ordinal)
            if coordinate in by_coordinate:
                raise ValueError(f"catalog contains duplicate coordinate {coordinate}")
            previous_ordinal = ordinal_by_storage_key.setdefault(
                record.storage_key, ordinal
            )
            previous_key = storage_key_by_ordinal.setdefault(
                ordinal, record.storage_key
            )
            if previous_ordinal != ordinal or previous_key != record.storage_key:
                raise ValueError("catalog storage keys and ordinals are not one-to-one")
            layers_by_ordinal.setdefault(ordinal, set()).add(layer)
            if set(record.components) != component_set:
                raise ValueError(
                    f"catalog entry {coordinate} does not define every component"
                )
            for extent in record.components.values():
                if extent.offset % alignment_bytes:
                    raise ValueError(
                        "catalog extent "
                        f"{coordinate} is not aligned to {alignment_bytes} bytes"
                    )
                if (
                    window_bytes is not None
                    and extent.offset + extent.bytes > window_bytes
                ):
                    raise ValueError(
                        f"catalog extent {coordinate} exceeds its tier window"
                    )
            ranges.extend(
                (
                    extent.offset,
                    extent.offset + extent.bytes,
                    coordinate,
                    component,
                )
                for component, extent in record.components.items()
            )
            by_coordinate[coordinate] = record
        expected_ordinals = set(range(len(storage_key_by_ordinal)))
        if set(storage_key_by_ordinal) != expected_ordinals:
            raise ValueError("catalog physical ordinals must be dense from zero")
        expected_layers = set(range(layer_count))
        incomplete = {
            ordinal: sorted(expected_layers - layers)
            for ordinal, layers in layers_by_ordinal.items()
            if layers != expected_layers
        }
        if incomplete:
            raise ValueError(
                f"catalog pages do not cover every model layer: {incomplete}"
            )
        previous: tuple[int, int, tuple[int, int], str] | None = None
        for current in sorted(ranges):
            if previous is not None and current[0] < previous[1]:
                raise ValueError(
                    "catalog extents overlap: "
                    f"layer/ordinal={previous[2]} {previous[3]} and "
                    f"layer/ordinal={current[2]} {current[3]}"
                )
            previous = current
        self.tier = tier
        self._records = by_coordinate
        self._ordinal_by_storage_key = ordinal_by_storage_key
        self._storage_key_by_ordinal = storage_key_by_ordinal
        self.namespace = namespace
        self.page_tokens = page_tokens
        self.layer_count = layer_count
        self.components = components
        self.alignment_bytes = alignment_bytes
        self.window_bytes = window_bytes
        self.digest = digest

    def validate_nvme_geometry(
        self,
        *,
        lba_size: int,
        max_transfer_bytes: int,
        namespace_bytes: int,
    ) -> None:
        """Qualify every immutable extent against one opened namespace.

        This is a setup-plane gate.  A catalog that could fail alignment,
        command-size, or namespace bounds must be rejected before any request
        can publish a partial runtime directory image.
        """

        lba_size = _catalog_uint64(lba_size, "NVMe LBA size", minimum=1)
        max_transfer_bytes = _catalog_uint64(
            max_transfer_bytes, "NVMe max transfer bytes", minimum=1
        )
        namespace_bytes = _catalog_uint64(
            namespace_bytes, "NVMe namespace bytes", minimum=1
        )
        if self.tier is not ServingTier.NVME:
            raise ValueError("NVMe geometry can only qualify an NVMe catalog")
        if self.window_bytes is not None and self.window_bytes > namespace_bytes:
            raise ValueError("NVMe catalog window exceeds the opened namespace")
        for (layer, ordinal), record in self._records.items():
            for component, extent in record.components.items():
                if (
                    extent.offset > namespace_bytes
                    or extent.bytes > namespace_bytes - extent.offset
                ):
                    raise ValueError(
                        "NVMe catalog extent exceeds the opened namespace: "
                        f"layer={layer}, ordinal={ordinal}, component={component}"
                    )
                try:
                    _validate_nvme_extent(
                        extent,
                        lba_size=lba_size,
                        max_transfer_bytes=max_transfer_bytes,
                        kind=component,
                    )
                except RuntimeError as error:
                    raise ValueError(
                        "NVMe catalog is incompatible with the opened controller: "
                        f"layer={layer}, ordinal={ordinal}, component={component}: "
                        f"{error}"
                    ) from error

    @property
    def page_count(self) -> int:
        return len(self._storage_key_by_ordinal)

    def has_storage_key(self, storage_key: str) -> bool:
        return storage_key in self._ordinal_by_storage_key

    def ordinal(self, storage_key: str) -> int:
        try:
            return self._ordinal_by_storage_key[storage_key]
        except KeyError:
            raise KeyError(f"tier catalog has no storage key {storage_key!r}") from None

    def storage_key(self, ordinal: int) -> str:
        ordinal = bounded_integer(
            ordinal, "catalog ordinal", minimum=0, maximum=_UINT32_MAX
        )
        try:
            return self._storage_key_by_ordinal[ordinal]
        except KeyError:
            raise KeyError(f"tier catalog has no ordinal {ordinal}") from None

    @classmethod
    def load(
        cls, path: str | os.PathLike[str], *, expected_tier: ServingTier
    ) -> "TierPageCatalog":
        catalog_path = Path(path)
        try:
            raw = catalog_path.read_bytes()
            document = json.loads(raw)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"cannot read tier catalog {catalog_path}: {error}"
            ) from error
        if not isinstance(document, dict):
            raise ValueError("tier catalog root must be an object")
        unknown = set(document) - {
            "schema",
            "tier",
            "format",
            "namespace",
            "page_tokens",
            "layer_count",
            "components",
            "alignment_bytes",
            "window_bytes",
            "entries",
        }
        if unknown:
            raise ValueError(f"tier catalog has unknown fields: {sorted(unknown)}")
        if document.get("schema") != cls.SCHEMA:
            raise ValueError("unsupported tier catalog schema")
        if document.get("format") != cls.FORMAT:
            raise ValueError("unsupported tier catalog payload format")
        if document.get("tier") != expected_tier.value:
            raise ValueError(
                f"tier catalog declares {document.get('tier')!r}, expected {expected_tier.value!r}"
            )
        alignment = _positive_int(
            document.get("alignment_bytes", 4096), "alignment_bytes"
        )
        window = document.get("window_bytes")
        if window is not None:
            window = _positive_int(window, "window_bytes")
        namespace = document.get("namespace")
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("tier catalog namespace must be non-empty")
        page_tokens = _positive_int(document.get("page_tokens"), "page_tokens")
        layer_count = _positive_int(document.get("layer_count"), "layer_count")
        raw_components = document.get("components")
        if not isinstance(raw_components, list) or not raw_components:
            raise ValueError("tier catalog components must be a non-empty list")
        components = tuple(raw_components)
        raw_records = document.get("entries")
        if not isinstance(raw_records, list) or not raw_records:
            raise ValueError("tier catalog must contain a non-empty entries list")
        records: list[_PageRecord] = []
        for index, item in enumerate(raw_records):
            if not isinstance(item, dict):
                raise ValueError(f"tier catalog entry {index} is not an object")
            unknown = set(item) - {
                "storage_key",
                "ordinal",
                "layer",
                "components",
            }
            if unknown:
                raise ValueError(
                    f"tier catalog entry {index} has unknown fields: {sorted(unknown)}"
                )
            storage_key = item.get("storage_key")
            if not isinstance(storage_key, str) or not storage_key:
                raise ValueError(f"entries[{index}].storage_key must be non-empty")
            ordinal = _nonnegative_int(item.get("ordinal"), f"entries[{index}].ordinal")
            layer = _nonnegative_int(item.get("layer"), f"entries[{index}].layer")
            raw_entry_components = item.get("components")
            if not isinstance(raw_entry_components, dict):
                raise ValueError(f"entries[{index}].components must be an object")
            records.append(
                _PageRecord(
                    storage_key,
                    ordinal,
                    layer,
                    MappingProxyType(
                        {
                            component: _parse_extent(
                                extent,
                                f"entries[{index}].components.{component}",
                            )
                            for component, extent in raw_entry_components.items()
                        }
                    ),
                )
            )
        return cls(
            tier=expected_tier,
            records=tuple(records),
            namespace=namespace,
            page_tokens=page_tokens,
            layer_count=layer_count,
            components=components,
            alignment_bytes=alignment,
            window_bytes=window,
            digest=hashlib.sha256(raw).hexdigest(),
        )

    def span(
        self,
        *,
        layer: int,
        ordinals: tuple[int, ...],
        component: str,
        row_bytes: int,
    ) -> PageExtent:
        layer = bounded_integer(
            layer, "catalog span layer", minimum=0, maximum=_UINT32_MAX
        )
        row_bytes = bounded_integer(
            row_bytes, "catalog span row size", minimum=1, maximum=_UINT32_MAX
        )
        if component not in self.components:
            raise ValueError(f"catalog has no component {component!r}")
        if not ordinals:
            raise ValueError("catalog span cannot be empty")
        extents: list[PageExtent] = []
        for ordinal in ordinals:
            ordinal = bounded_integer(
                ordinal, "catalog span ordinal", minimum=0, maximum=_UINT32_MAX
            )
            record = self._records.get((layer, ordinal))
            if record is None:
                raise KeyError(f"tier catalog has no layer={layer}, ordinal={ordinal}")
            extent = record.components[component]
            if extent.bytes != row_bytes:
                raise ValueError(
                    "catalog row size mismatch for "
                    f"layer={layer}, ordinal={ordinal}, {component}: "
                    f"{extent.bytes} != {row_bytes}"
                )
            extents.append(extent)
        expected_bytes = row_bytes * len(extents)
        if any(
            current.offset != previous.offset + previous.bytes
            for previous, current in zip(extents, extents[1:])
        ):
            raise ValueError(
                "tier catalog ordinals are not contiguous for "
                f"layer={layer}, {component}"
            )
        return PageExtent(extents[0].offset, expected_bytes)

    def transfer_runs(
        self,
        *,
        layer: int,
        storage_keys: tuple[str, ...],
        destination_indices: tuple[int, ...],
        component: str,
        row_bytes: int,
        max_transfer_bytes: int,
    ) -> tuple[PageTransferRun, ...]:
        """Resolve exact identities into maximal bounded physical runs.

        Coalescing is legal only when both the immutable source byte extent and
        the framework destination block advance contiguously.  Input order is
        irrelevant; destination ownership defines the canonical order.  This
        lets SGLang and vLLM share one directory rule without treating their
        transient page numbers as storage identities.
        """

        if not storage_keys or len(storage_keys) != len(destination_indices):
            raise ValueError(
                "page-transfer identities and destinations must be non-empty "
                "and aligned"
            )
        row_bytes = bounded_integer(
            row_bytes,
            "page-transfer row size",
            minimum=1,
            maximum=_UINT32_MAX,
        )
        max_transfer_bytes = bounded_integer(
            max_transfer_bytes,
            "page-transfer byte limit",
            minimum=row_bytes,
            maximum=_UINT32_MAX,
        )
        if max_transfer_bytes < row_bytes:
            raise ValueError("page-transfer byte limit is smaller than one row")

        resolved: list[tuple[int, PageExtent]] = []
        seen_destinations: set[int] = set()
        for storage_key, destination in zip(
            storage_keys, destination_indices, strict=True
        ):
            if not isinstance(storage_key, str) or not storage_key:
                raise ValueError("page-transfer storage keys must be non-empty")
            destination = bounded_integer(
                destination,
                "page-transfer destination",
                minimum=0,
                maximum=_UINT32_MAX,
            )
            if destination in seen_destinations:
                raise ValueError("page-transfer destinations must be unique")
            seen_destinations.add(destination)
            ordinal = self.ordinal(storage_key)
            extent = self.span(
                layer=layer,
                ordinals=(ordinal,),
                component=component,
                row_bytes=row_bytes,
            )
            resolved.append((destination, extent))

        ordered = sorted(resolved, key=lambda item: item[0])
        runs: list[PageTransferRun] = []
        run_destination, run_extent = ordered[0]
        run_rows = 1
        for destination, extent in ordered[1:]:
            contiguous = (
                destination == run_destination + run_rows
                and extent.offset == run_extent.offset + run_extent.bytes
                and run_extent.bytes + row_bytes <= max_transfer_bytes
            )
            if contiguous:
                run_extent = PageExtent(run_extent.offset, run_extent.bytes + row_bytes)
                run_rows += 1
                continue
            runs.append(PageTransferRun(run_extent, run_destination, run_rows))
            run_destination = destination
            run_extent = extent
            run_rows = 1
        runs.append(PageTransferRun(run_extent, run_destination, run_rows))
        return tuple(runs)


@dataclass(frozen=True)
class ServingTierConfig:
    tier: ServingTier = ServingTier.HBM
    endpoint: str | None = None
    catalog_path: Path | None = None
    namespace_id: int = 1
    queue_depth: int = 64
    window_bytes: int = 0
    device_ordinal: int = -1
    issue_budget: int = 64
    completion_budget: int = 64
    progress_rounds: int = 1
    progress_timeout_ns: int = 100_000_000
    trust_read_only_device_code: bool = False
    nvme_hbm_backend: NvmeHbmBackendRequirement = NvmeHbmBackendRequirement.AUTO
    nvme_service_model: NvmeTransferServiceModel = field(
        default_factory=NvmeTransferServiceModel
    )

    _NVME_DEFAULTS: ClassVar[dict[str, int]] = {
        "namespace_id": 1,
        "queue_depth": 64,
        "issue_budget": 64,
        "completion_budget": 64,
        "progress_rounds": 1,
        "progress_timeout_ns": 100_000_000,
    }

    def __post_init__(self) -> None:
        if not isinstance(self.tier, ServingTier):
            raise ValueError("tier must be a ServingTier value")
        if self.endpoint is not None:
            if not isinstance(self.endpoint, str) or not self.endpoint:
                raise ValueError("tier endpoint must be a non-empty string")
        if self.catalog_path is not None:
            if not isinstance(self.catalog_path, (str, os.PathLike)):
                raise ValueError("tier catalog_path must be path-like")
            object.__setattr__(self, "catalog_path", Path(self.catalog_path))
        for name in (
            "namespace_id",
            "queue_depth",
            "issue_budget",
            "completion_budget",
            "progress_rounds",
        ):
            object.__setattr__(
                self,
                name,
                bounded_integer(
                    getattr(self, name), name, minimum=1, maximum=_UINT32_MAX
                ),
            )
        object.__setattr__(
            self,
            "progress_timeout_ns",
            bounded_integer(
                self.progress_timeout_ns,
                "progress_timeout_ns",
                minimum=1,
                maximum=_UINT64_MAX,
            ),
        )
        object.__setattr__(
            self,
            "window_bytes",
            bounded_integer(
                self.window_bytes,
                "window_bytes",
                minimum=0,
                maximum=_UINT64_MAX,
            ),
        )
        object.__setattr__(
            self,
            "device_ordinal",
            bounded_integer(
                self.device_ordinal,
                "device_ordinal",
                minimum=-1,
                maximum=_INT32_MAX,
            ),
        )
        if not isinstance(self.trust_read_only_device_code, bool):
            raise ValueError("trust_read_only_device_code must be boolean")
        if not isinstance(self.nvme_hbm_backend, NvmeHbmBackendRequirement):
            raise ValueError(
                "nvme_hbm_backend must be a NvmeHbmBackendRequirement value"
            )
        if not isinstance(self.nvme_service_model, NvmeTransferServiceModel):
            raise ValueError("nvme_service_model must be a typed service model")
        if self.tier is ServingTier.CXL_DAX:
            if self.window_bytes <= 0:
                raise ValueError("CXL DAX window_bytes must be positive")
        elif self.window_bytes != 0:
            raise ValueError("only CXL DAX accepts a window_bytes value")
        if (
            self.tier is not ServingTier.NVME
            and self.nvme_hbm_backend is not NvmeHbmBackendRequirement.AUTO
        ):
            raise ValueError("an NVMe HBM backend requirement needs the NVMe tier")
        if self.tier is not ServingTier.NVME and self.nvme_service_model.calibrated:
            raise ValueError("an NVMe service model needs the NVMe tier")
        if self.tier is not ServingTier.NVME:
            changed_nvme_fields = [
                name
                for name, default in self._NVME_DEFAULTS.items()
                if getattr(self, name) != default
            ]
            if self.trust_read_only_device_code:
                changed_nvme_fields.append("trust_read_only_device_code")
            if changed_nvme_fields:
                raise ValueError(
                    "NVMe-only configuration was applied to "
                    f"{self.tier.value}: {', '.join(changed_nvme_fields)}"
                )
        if self.tier in PHYSICAL_SERVING_TIERS and (
            not self.endpoint or self.catalog_path is None
        ):
            raise ValueError(
                "physical serving tiers require an endpoint and page catalog"
            )
        if self.tier not in PHYSICAL_SERVING_TIERS and (
            self.endpoint is not None or self.catalog_path is not None
        ):
            raise ValueError("non-physical tiers cannot own an endpoint or catalog")

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> "ServingTierConfig":
        values = os.environ if environ is None else environ
        raw_tier = values.get("NTA_SERVING_TIER", ServingTier.HBM.value).strip().lower()
        try:
            tier = ServingTier(raw_tier)
        except ValueError as error:
            raise ValueError(
                "NTA_SERVING_TIER must be hbm, host_mapped, host_staged, "
                "nvme, or cxl_dax"
            ) from error
        if tier not in PHYSICAL_SERVING_TIERS:
            return cls(tier=tier)
        endpoint_name = (
            "NTA_NVME_ENDPOINT" if tier is ServingTier.NVME else "NTA_CXL_DAX_DEVICE"
        )
        endpoint = values.get(endpoint_name, "").strip()
        if not endpoint:
            raise ValueError(f"{endpoint_name} is required for {tier.value}")
        catalog_value = values.get("NTA_TIER_CATALOG", "").strip()
        if not catalog_value:
            raise ValueError("NTA_TIER_CATALOG is required for a physical serving tier")
        window_bytes = 0
        if tier is ServingTier.CXL_DAX:
            raw_window = values.get("NTA_CXL_DAX_WINDOW_BYTES", "").strip()
            if raw_window:
                window_bytes = _positive_int(raw_window, "NTA_CXL_DAX_WINDOW_BYTES")
            else:
                window_mib = _positive_int(
                    values.get("NTA_CXL_DAX_WINDOW_MIB", "0"),
                    "NTA_CXL_DAX_WINDOW_MIB",
                )
                window_bytes = window_mib * 1024 * 1024
        device_ordinal = int(values.get("NTA_TIER_DEVICE_ORDINAL", "-1"))
        if device_ordinal < -1:
            raise ValueError("NTA_TIER_DEVICE_ORDINAL must be -1 or nonnegative")
        raw_hbm_backend = values.get("NTA_NVME_HBM_BACKEND", "auto").strip().lower()
        try:
            hbm_backend = NvmeHbmBackendRequirement(raw_hbm_backend)
        except ValueError as error:
            raise ValueError(
                "NTA_NVME_HBM_BACKEND must be auto, cuda-dmabuf-ioas, or "
                "nvidia-peer-pages"
            ) from error
        service_names = (
            "NTA_NVME_COMMAND_SERVICE_NS",
            "NTA_NVME_READ_BANDWIDTH_BPS",
            "NTA_NVME_COMPACTION_BANDWIDTH_BPS",
        )
        service_values = tuple(values.get(name, "").strip() for name in service_names)
        service_tuning_present = any(
            values.get(name, "").strip()
            for name in (
                "NTA_NVME_COMPACTION_LAUNCH_NS",
                "NTA_NVME_MINIMUM_GAIN",
            )
        )
        if (any(service_values) and not all(service_values)) or (
            service_tuning_present and not all(service_values)
        ):
            raise ValueError(
                "NVMe granularity calibration requires command, read-bandwidth, "
                "and compaction-bandwidth measurements together"
            )
        service_model = (
            NvmeTransferServiceModel()
            if not any(service_values)
            else NvmeTransferServiceModel(
                command_service_ns=_positive_int(service_values[0], service_names[0]),
                read_bandwidth_bytes_per_second=_positive_int(
                    service_values[1], service_names[1]
                ),
                compaction_bandwidth_bytes_per_second=_positive_int(
                    service_values[2], service_names[2]
                ),
                compaction_launch_ns=int(
                    values.get("NTA_NVME_COMPACTION_LAUNCH_NS", "0")
                ),
                minimum_gain=float(values.get("NTA_NVME_MINIMUM_GAIN", "1.03")),
            )
        )
        return cls(
            tier=tier,
            endpoint=endpoint,
            catalog_path=Path(catalog_value),
            namespace_id=_positive_int(
                values.get("NTA_NVME_NAMESPACE", "1"), "NTA_NVME_NAMESPACE"
            ),
            queue_depth=_positive_int(
                values.get("NTA_NVME_QUEUE_DEPTH", "64"), "NTA_NVME_QUEUE_DEPTH"
            ),
            window_bytes=window_bytes,
            device_ordinal=device_ordinal,
            issue_budget=_positive_int(
                values.get("NTA_NVME_ISSUE_BUDGET", "64"), "NTA_NVME_ISSUE_BUDGET"
            ),
            completion_budget=_positive_int(
                values.get("NTA_NVME_COMPLETION_BUDGET", "64"),
                "NTA_NVME_COMPLETION_BUDGET",
            ),
            progress_rounds=_positive_int(
                values.get("NTA_NVME_PROGRESS_ROUNDS", "1"), "NTA_NVME_PROGRESS_ROUNDS"
            ),
            progress_timeout_ns=_positive_int(
                values.get("NTA_NVME_PROGRESS_TIMEOUT_NS", "100000000"),
                "NTA_NVME_PROGRESS_TIMEOUT_NS",
            ),
            trust_read_only_device_code=_binary_flag(
                values.get("NTA_NVME_TRUST_READ_ONLY_DEVICE_CODE", "0"),
                "NTA_NVME_TRUST_READ_ONLY_DEVICE_CODE",
            ),
            nvme_hbm_backend=hbm_backend,
            nvme_service_model=service_model,
        )


class ServingTierService:
    """Own one selected tier, its catalog, and its native transport."""

    def __init__(self, config: ServingTierConfig) -> None:
        if not isinstance(config, ServingTierConfig):
            raise TypeError("serving tier service requires a typed configuration")
        self.config = config
        self.contract = resource_contract(_RESOURCE_KIND_FOR_SERVING_TIER[config.tier])
        self.catalog = (
            None
            if config.catalog_path is None
            else TierPageCatalog.load(config.catalog_path, expected_tier=config.tier)
        )
        self.nvme: NvmeTransport | None = None
        self.cxl: CxlDaxTransport | None = None
        self._nvme_hbm_regions: dict[tuple[int, int], Any] = {}
        self._nvme_hbm_prepared = False
        self._nvme_lba_size: int | None = None
        self._nvme_controller_page_size: int | None = None
        self._nvme_max_transfer_bytes: int | None = None
        self._nvme_namespace_bytes: int | None = None
        # Keep catalog validation and experiment tooling independent of CUDA
        # and libnta-runtime.  Native bindings are loaded only when a serving
        # process explicitly opens a physical tier.
        if config.tier in PHYSICAL_SERVING_TIERS:
            from .runtime import (
                CxlDaxOptions,
                CxlDaxTransport,
                NvmeHbmMappingPolicy,
                NvmeOptions,
                NvmeTransport,
            )

        if config.tier is ServingTier.NVME:
            if config.endpoint is None or self.catalog is None:
                raise ValueError("NVMe service requires an endpoint and page catalog")
            mapping_policy = {
                NvmeHbmBackendRequirement.AUTO: NvmeHbmMappingPolicy.AUTO,
                NvmeHbmBackendRequirement.CUDA_DMA_BUF_IOAS: (
                    NvmeHbmMappingPolicy.CUDA_DMA_BUF_IOAS
                ),
                NvmeHbmBackendRequirement.NVIDIA_PEER_PAGES: (
                    NvmeHbmMappingPolicy.NVIDIA_PEER_PAGES
                ),
            }[config.nvme_hbm_backend]
            try:
                self.nvme = NvmeTransport(
                    NvmeOptions(
                        endpoint=config.endpoint,
                        device_ordinal=config.device_ordinal,
                        namespace_id=config.namespace_id,
                        queue_depth=config.queue_depth,
                        trust_read_only_device_code=(
                            config.trust_read_only_device_code
                        ),
                        hbm_mapping_policy=mapping_policy,
                    )
                )
                capabilities = self.nvme.capabilities
                self._nvme_lba_size = int(capabilities.lba_size)
                self._nvme_controller_page_size = int(capabilities.controller_page_size)
                self._nvme_max_transfer_bytes = int(capabilities.max_transfer_bytes)
                self._nvme_namespace_bytes = int(capabilities.namespace_bytes)
                self.catalog.validate_nvme_geometry(
                    lba_size=self._nvme_lba_size,
                    max_transfer_bytes=self._nvme_max_transfer_bytes,
                    namespace_bytes=self._nvme_namespace_bytes,
                )
                if not capabilities.supports_hbm_peer_dma:
                    raise RuntimeError(
                        "NVMe endpoint does not support direct HBM peer DMA"
                    )
                if not capabilities.translated_iommu:
                    raise RuntimeError(
                        "NVMe endpoint is not attached through a translated IOMMU"
                    )
                if not capabilities.gpu_doorbell_mapping_validated:
                    raise RuntimeError("NVMe GPU doorbell mapping was not qualified")
                selected_backend = capabilities.hbm_mapping_backend.artifact_name
                _require_nvme_hbm_backend(config.nvme_hbm_backend, selected_backend)
            except BaseException:
                if self.nvme is not None:
                    self.nvme.close()
                self.nvme = None
                raise
        elif config.tier is ServingTier.CXL_DAX:
            if (
                config.endpoint is None
                or config.catalog_path is None
                or config.window_bytes <= 0
            ):
                raise ValueError(
                    "CXL-DAX service requires an endpoint, catalog, and window"
                )
            if self.catalog is None or self.catalog.window_bytes != config.window_bytes:
                raise ValueError(
                    "CXL catalog window_bytes must match the configured DAX window"
                )
            self.cxl = CxlDaxTransport(
                CxlDaxOptions(
                    endpoint=config.endpoint,
                    window_bytes=config.window_bytes,
                    device_ordinal=config.device_ordinal,
                )
            )
            if not self.cxl.capabilities.direct_device_visible:
                self.cxl.close()
                self.cxl = None
                raise RuntimeError("CXL DAX endpoint is not CUDA device-visible")

    def close(self) -> None:
        """Release the selected transport after its consumer runtime is quiesced."""
        nvme, cxl = self.nvme, self.cxl
        self.nvme = None
        self.cxl = None
        first_error: BaseException | None = None
        regions = getattr(self, "_nvme_hbm_regions", {})
        for region in regions.values():
            try:
                region.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        regions.clear()
        if nvme is not None:
            try:
                nvme.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if cxl is not None:
            try:
                cxl.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise RuntimeError(
                "serving tier transport teardown failed"
            ) from first_error

    def __del__(self) -> None:
        # A physical transport can be created before a later capability check
        # rejects the service.  Keep the same best-effort owner cleanup on that
        # exceptional construction path as on normal runtime shutdown.
        try:
            self.close()
        except BaseException:
            pass

    @property
    def tier(self) -> ServingTier:
        return self.config.tier

    @property
    def is_hbm(self) -> bool:
        return self.tier is ServingTier.HBM

    @property
    def is_host_staged(self) -> bool:
        return self.tier is ServingTier.HOST_STAGED

    @property
    def is_physical(self) -> bool:
        return self.tier in PHYSICAL_SERVING_TIERS

    @property
    def is_nvme(self) -> bool:
        return self.tier is ServingTier.NVME

    @property
    def is_cxl(self) -> bool:
        return self.tier is ServingTier.CXL_DAX

    @property
    def catalog_digest(self) -> str | None:
        return None if self.catalog is None else self.catalog.digest

    @property
    def nvme_lba_size(self) -> int:
        if not self.is_nvme or self._nvme_lba_size is None:
            raise RuntimeError("NVMe LBA size is only available for an NVMe tier")
        return self._nvme_lba_size

    @property
    def nvme_controller_page_size(self) -> int:
        if not self.is_nvme or self._nvme_controller_page_size is None:
            raise RuntimeError(
                "NVMe controller page size is only available for an NVMe tier"
            )
        return self._nvme_controller_page_size

    @property
    def nvme_max_transfer_bytes(self) -> int:
        if not self.is_nvme or self._nvme_max_transfer_bytes is None:
            raise RuntimeError("NVMe transfer limit is only available for an NVMe tier")
        return self._nvme_max_transfer_bytes

    def prepare_nvme_hbm_destinations(
        self, destinations: tuple[HbmDestinationSlice, ...]
    ) -> NvmeHbmPreparation:
        """Validate, coalesce, and register all framework HBM slices once.

        The describe phase is mutation-free.  Only after every tensor has a
        consistent native allocation/envelope description are disjoint peer
        mappings installed.  This prevents CUDA allocator suballocations from
        acquiring overlapping IOMMU PTE ownership.
        """

        if not self.is_nvme or self.nvme is None:
            raise RuntimeError("HBM destination preparation requires an NVMe tier")
        if self._nvme_hbm_prepared or self._nvme_hbm_regions:
            raise RuntimeError("NVMe HBM destinations were already prepared")
        if not destinations:
            raise ValueError("NVMe HBM destination preparation cannot be empty")

        descriptions = tuple(
            DescribedHbmDestination.from_native(
                destination,
                self.nvme.describe_hbm_region(destination.address, destination.bytes),
            )
            for destination in destinations
        )
        groups = coalesce_hbm_destinations(descriptions)
        registered: list[tuple[Any, Any]] = []
        try:
            for group in groups:
                region = self.nvme.register_hbm_region(
                    group.registration_address, group.registration_bytes
                )
                registered.append((group, region))
        except BaseException:
            for _group, region in reversed(registered):
                region.close()
            raise

        by_destination: dict[Hashable, Any] = {}
        for group, region in registered:
            self._nvme_hbm_regions[
                (group.registration_address, group.registration_bytes)
            ] = region
            for destination in group.destinations:
                by_destination[destination.key] = region
        if len(by_destination) != len(destinations):
            raise RuntimeError("NVMe HBM preparation lost a destination binding")
        self._nvme_hbm_prepared = True
        return NvmeHbmPreparation(
            regions=MappingProxyType(by_destination),
            destination_count=len(destinations),
            destination_bytes=sum(item.bytes for item in destinations),
            registration_count=len(groups),
            registration_bytes=sum(item.registration_bytes for item in groups),
        )

    def extent(
        self, layer: int, ordinals: tuple[int, ...], component: str, row_bytes: int
    ) -> PageExtent:
        if self.catalog is None:
            raise RuntimeError(f"{self.tier.value} has no page catalog")
        extent = self.catalog.span(
            layer=layer,
            ordinals=ordinals,
            component=component,
            row_bytes=row_bytes,
        )
        if self.is_nvme:
            lba_size = self._nvme_lba_size
            max_transfer_bytes = self._nvme_max_transfer_bytes
            if lba_size is None or max_transfer_bytes is None:
                raise RuntimeError("NVMe tier has no cached controller capabilities")
            _validate_nvme_extent(
                extent,
                lba_size=lba_size,
                max_transfer_bytes=max_transfer_bytes,
                kind=component,
            )
        return extent

    def device_address(self, extent: PageExtent) -> int:
        if not self.is_cxl or self.cxl is None:
            raise RuntimeError("device_address is only valid for CXL direct mappings")
        address = self.cxl.device_address + extent.offset
        if (
            not self.cxl.capabilities.direct_device_visible
            or not (extent.offset + extent.bytes <= self.cxl.capabilities.window_bytes)
            or address > _UINT64_MAX - extent.bytes
        ):
            raise RuntimeError("CXL catalog extent is outside the mapped device window")
        return address

    def stats(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "serving_tier": self.tier.value,
            "tier_catalog_digest": self.catalog_digest,
            "tier_data_path": self.contract.steady_state_path,
            "resource_contract": self.contract.as_dict(),
            "tier_fallback": False,
            "required_nvme_hbm_backend": self.config.nvme_hbm_backend.value,
        }
        service_model = self.config.nvme_service_model
        result["nvme_granularity_service_model"] = {
            "calibrated": service_model.calibrated,
            "command_service_ns": service_model.command_service_ns,
            "read_bandwidth_bytes_per_second": (
                service_model.read_bandwidth_bytes_per_second
            ),
            "compaction_bandwidth_bytes_per_second": (
                service_model.compaction_bandwidth_bytes_per_second
            ),
            "compaction_launch_ns": service_model.compaction_launch_ns,
            "minimum_gain": service_model.minimum_gain,
        }
        if self.nvme is not None:
            capabilities = self.nvme.capabilities
            queue_stats = self.nvme.stats
            result["tier_capabilities"] = {
                "translated_iommu": capabilities.translated_iommu,
                "hbm_peer_dma": capabilities.supports_hbm_peer_dma,
                "gpu_doorbell_validated": capabilities.gpu_doorbell_mapping_validated,
                "namespace_read_only": capabilities.namespace_read_only,
                "mapping_backend": capabilities.hbm_mapping_backend.artifact_name,
                "lba_size": capabilities.lba_size,
                "max_transfer_bytes": capabilities.max_transfer_bytes,
                "namespace_bytes": capabilities.namespace_bytes,
                "hbm_region_registrations": queue_stats.hbm_region_registrations,
                "hbm_region_bytes": queue_stats.hbm_region_bytes,
                "hbm_transfer_views": queue_stats.hbm_transfer_views,
            }
        elif self.cxl is not None:
            capabilities = self.cxl.capabilities
            result["tier_capabilities"] = {
                "direct_device_visible": capabilities.direct_device_visible,
                "host_registered": capabilities.host_registered,
                "window_bytes": capabilities.window_bytes,
            }
        return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if isinstance(value, float) and result != value:
        raise ValueError(f"{name} must be a positive integer")
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _binary_flag(value: Any, name: str) -> bool:
    if not isinstance(value, str) or value.strip() not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1")
    return value.strip() == "1"


def _catalog_uint64(value: Any, name: str, *, minimum: int) -> int:
    try:
        return bounded_integer(value, name, minimum=minimum, maximum=_UINT64_MAX)
    except ValueError as error:
        if isinstance(value, int) and value > _UINT64_MAX:
            raise ValueError(f"{name} exceeds the native uint64 limit") from error
        raise


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a nonnegative integer") from error
    if isinstance(value, float) and result != value:
        raise ValueError(f"{name} must be a nonnegative integer")
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _parse_extent(value: Any, name: str) -> PageExtent:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    unknown = set(value) - {"offset", "bytes"}
    if unknown:
        raise ValueError(f"{name} has unknown fields: {sorted(unknown)}")
    return PageExtent(
        _nonnegative_int(value.get("offset"), f"{name}.offset"),
        _positive_int(value.get("bytes"), f"{name}.bytes"),
    )


def _validate_nvme_extent(
    extent: PageExtent, *, lba_size: int, max_transfer_bytes: int, kind: str
) -> None:
    if lba_size <= 0 or max_transfer_bytes <= 0:
        raise RuntimeError("NVMe controller reported invalid transfer capabilities")
    if extent.offset % lba_size:
        raise RuntimeError(
            f"NVMe {kind} extent offset is not LBA aligned: {extent.offset} bytes "
            f"(LBA size {lba_size})"
        )
    if extent.bytes % lba_size:
        raise RuntimeError(
            f"NVMe {kind} extent is not LBA aligned: {extent.bytes} bytes "
            f"(LBA size {lba_size})"
        )
    if extent.bytes > max_transfer_bytes:
        raise RuntimeError(
            f"NVMe {kind} extent is {extent.bytes} bytes, exceeding the "
            f"controller max transfer {max_transfer_bytes} bytes; "
            "use a smaller exact FlashInfer KV chunk"
        )
