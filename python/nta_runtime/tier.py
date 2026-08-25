"""Engine-neutral serving tier selection and exact page ownership.

The serving adapters do not infer storage locations from a framework's host
cache.  A tier catalog is an explicit, immutable contract produced by the
storage preparation step.  This keeps the compiler/runtime mechanism exact:
the operator supplies requested page identities, while this module resolves
those identities to byte extents owned by one physical tier.

There is intentionally no implicit fallback between tiers.  Host-staged is
the default deployment profile; selecting a physical tier requires both its
endpoint and a validated catalog.  A missing capability is therefore an
operator/configuration error, not a silent change in the measured path.
"""

from __future__ import annotations

from dataclasses import dataclass
import enum
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


_UINT64_MAX = (1 << 64) - 1


class ServingTier(str, enum.Enum):
    HOST_STAGED = "host_staged"
    NVME = "nvme"
    CXL_DAX = "cxl_dax"


@dataclass(frozen=True)
class PageExtent:
    """A contiguous exact byte range in a tier's address space."""

    offset: int
    bytes: int

    def __post_init__(self) -> None:
        if self.offset < 0 or self.bytes <= 0:
            raise ValueError(
                "tier extents require a nonnegative offset and positive bytes"
            )
        if self.bytes > (1 << 32) - 1:
            raise ValueError("tier extent exceeds the native dependency byte limit")
        if self.offset > _UINT64_MAX or self.bytes > _UINT64_MAX - self.offset:
            raise ValueError("tier extent exceeds the native address-space limit")


@dataclass(frozen=True)
class _PageRecord:
    layer: int
    page: int
    key: PageExtent
    value: PageExtent


class TierPageCatalog:
    """Validated logical-page to physical-byte mapping.

    Catalog entries are keyed by the device page identity emitted by the
    framework's page table.  Each entry describes one complete K/V row.  A
    multi-page request is accepted only when all requested rows are present,
    have the expected exact row size, and form one contiguous byte span.  The
    latter rule is required by the current NVMe object ABI and also prevents a
    CXL direct dependency from silently changing into a gather operation.
    """

    SCHEMA = 1

    def __init__(
        self,
        *,
        tier: ServingTier,
        records: tuple[_PageRecord, ...],
        alignment_bytes: int = 4096,
        window_bytes: int | None = None,
        digest: str = "",
    ) -> None:
        if alignment_bytes <= 0 or alignment_bytes & (alignment_bytes - 1):
            raise ValueError("catalog alignment_bytes must be a positive power of two")
        if window_bytes is not None and window_bytes <= 0:
            raise ValueError("catalog window_bytes must be positive")
        by_key: dict[tuple[int, int], _PageRecord] = {}
        ranges: list[tuple[int, int, tuple[int, int], str]] = []
        for record in records:
            key = (record.layer, record.page)
            if key in by_key:
                raise ValueError(f"catalog contains duplicate page {key}")
            for extent in (record.key, record.value):
                if extent.offset % alignment_bytes:
                    raise ValueError(
                        f"catalog extent {key} is not aligned to {alignment_bytes} bytes"
                    )
                if (
                    window_bytes is not None
                    and extent.offset + extent.bytes > window_bytes
                ):
                    raise ValueError(f"catalog extent {key} exceeds its tier window")
            if (
                record.key.offset < record.value.offset + record.value.bytes
                and record.value.offset < record.key.offset + record.key.bytes
            ):
                raise ValueError(f"catalog K/V extents overlap for page {key}")
            ranges.extend(
                (
                    extent.offset,
                    extent.offset + extent.bytes,
                    key,
                    kind,
                )
                for kind, extent in (("key", record.key), ("value", record.value))
            )
            by_key[key] = record
        previous: tuple[int, int, tuple[int, int], str] | None = None
        for current in sorted(ranges):
            if previous is not None and current[0] < previous[1]:
                raise ValueError(
                    "catalog extents overlap: "
                    f"layer/page={previous[2]} {previous[3]} and "
                    f"layer/page={current[2]} {current[3]}"
                )
            previous = current
        self.tier = tier
        self._records = by_key
        self.alignment_bytes = alignment_bytes
        self.window_bytes = window_bytes
        self.digest = digest

    @property
    def page_count(self) -> int:
        return len(self._records)

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
            "alignment_bytes",
            "window_bytes",
            "pages",
        }
        if unknown:
            raise ValueError(f"tier catalog has unknown fields: {sorted(unknown)}")
        if document.get("schema") != cls.SCHEMA:
            raise ValueError("unsupported tier catalog schema")
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
        raw_records = document.get("pages")
        if not isinstance(raw_records, list) or not raw_records:
            raise ValueError("tier catalog must contain a non-empty pages list")
        records: list[_PageRecord] = []
        for index, item in enumerate(raw_records):
            if not isinstance(item, dict):
                raise ValueError(f"tier catalog page {index} is not an object")
            unknown = set(item) - {"layer", "page", "key", "value"}
            if unknown:
                raise ValueError(
                    f"tier catalog page {index} has unknown fields: {sorted(unknown)}"
                )
            layer = _nonnegative_int(item.get("layer"), f"pages[{index}].layer")
            page = _nonnegative_int(item.get("page"), f"pages[{index}].page")
            records.append(
                _PageRecord(
                    layer,
                    page,
                    _parse_extent(item.get("key"), f"pages[{index}].key"),
                    _parse_extent(item.get("value"), f"pages[{index}].value"),
                )
            )
        return cls(
            tier=expected_tier,
            records=tuple(records),
            alignment_bytes=alignment,
            window_bytes=window,
            digest=hashlib.sha256(raw).hexdigest(),
        )

    def span(
        self,
        *,
        layer: int,
        pages: tuple[int, ...],
        kind: str,
        row_bytes: int,
    ) -> PageExtent:
        if layer < 0 or row_bytes <= 0:
            raise ValueError(
                "catalog span requires a nonnegative layer and positive row size"
            )
        if kind not in {"key", "value"}:
            raise ValueError("catalog span kind must be key or value")
        if not pages:
            raise ValueError("catalog span cannot be empty")
        extents: list[PageExtent] = []
        for page in pages:
            if page < 0:
                raise ValueError("catalog page IDs cannot be negative")
            record = self._records.get((layer, page))
            if record is None:
                raise KeyError(f"tier catalog has no layer={layer}, page={page}")
            extent = record.key if kind == "key" else record.value
            if extent.bytes != row_bytes:
                raise ValueError(
                    f"catalog row size mismatch for layer={layer}, page={page}, {kind}: "
                    f"{extent.bytes} != {row_bytes}"
                )
            extents.append(extent)
        expected_bytes = row_bytes * len(extents)
        if any(
            current.offset != previous.offset + previous.bytes
            for previous, current in zip(extents, extents[1:])
        ):
            raise ValueError(
                f"tier catalog pages are not contiguous for layer={layer}, {kind}"
            )
        return PageExtent(extents[0].offset, expected_bytes)


@dataclass(frozen=True)
class ServingTierConfig:
    tier: ServingTier = ServingTier.HOST_STAGED
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

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> "ServingTierConfig":
        values = os.environ if environ is None else environ
        raw_tier = (
            values.get("NTA_SERVING_TIER", ServingTier.HOST_STAGED.value)
            .strip()
            .lower()
        )
        aliases = {
            "host": ServingTier.HOST_STAGED.value,
            "cxl": ServingTier.CXL_DAX.value,
        }
        raw_tier = aliases.get(raw_tier, raw_tier)
        try:
            tier = ServingTier(raw_tier)
        except ValueError as error:
            raise ValueError(
                "NTA_SERVING_TIER must be host_staged, nvme, or cxl_dax"
            ) from error
        if tier is ServingTier.HOST_STAGED:
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
            trust_read_only_device_code=values.get(
                "NTA_NVME_TRUST_READ_ONLY_DEVICE_CODE", "0"
            )
            == "1",
        )


class ServingTierService:
    """Own one selected tier, its catalog, and its native transport."""

    def __init__(self, config: ServingTierConfig) -> None:
        self.config = config
        self.catalog = (
            None
            if config.catalog_path is None
            else TierPageCatalog.load(config.catalog_path, expected_tier=config.tier)
        )
        self.nvme: NvmeTransport | None = None
        self.cxl: CxlDaxTransport | None = None
        self._nvme_lba_size: int | None = None
        self._nvme_max_transfer_bytes: int | None = None
        # Keep catalog validation and experiment tooling independent of CUDA
        # and libnta-runtime.  Native bindings are loaded only when a serving
        # process explicitly opens a physical tier.
        if config.tier is not ServingTier.HOST_STAGED:
            from .runtime import (
                CxlDaxOptions,
                CxlDaxTransport,
                NvmeOptions,
                NvmeTransport,
            )

        if config.tier is ServingTier.NVME:
            assert config.endpoint is not None
            assert self.catalog is not None
            self.nvme = NvmeTransport(
                NvmeOptions(
                    endpoint=config.endpoint,
                    device_ordinal=config.device_ordinal,
                    namespace_id=config.namespace_id,
                    queue_depth=config.queue_depth,
                    trust_read_only_device_code=config.trust_read_only_device_code,
                )
            )
            capabilities = self.nvme.capabilities
            self._nvme_lba_size = int(capabilities.lba_size)
            self._nvme_max_transfer_bytes = int(capabilities.max_transfer_bytes)
            if not capabilities.supports_hbm_peer_dma:
                self.nvme.close()
                self.nvme = None
                raise RuntimeError("NVMe endpoint does not support direct HBM peer DMA")
            if not capabilities.translated_iommu:
                self.nvme.close()
                self.nvme = None
                raise RuntimeError(
                    "NVMe endpoint is not attached through a translated IOMMU"
                )
            if not capabilities.gpu_doorbell_mapping_validated:
                self.nvme.close()
                self.nvme = None
                raise RuntimeError("NVMe GPU doorbell mapping was not qualified")
        elif config.tier is ServingTier.CXL_DAX:
            assert config.endpoint is not None
            assert config.catalog_path is not None
            assert config.window_bytes > 0
            self.cxl = CxlDaxTransport(
                CxlDaxOptions(
                    endpoint=config.endpoint,
                    window_bytes=config.window_bytes,
                    device_ordinal=config.device_ordinal,
                )
            )
            if self.catalog is None or self.catalog.window_bytes != config.window_bytes:
                self.cxl.close()
                self.cxl = None
                raise ValueError(
                    "CXL catalog window_bytes must match the configured DAX window"
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
        if nvme is not None:
            nvme.close()
        if cxl is not None:
            cxl.close()

    @property
    def tier(self) -> ServingTier:
        return self.config.tier

    @property
    def is_host(self) -> bool:
        return self.tier is ServingTier.HOST_STAGED

    @property
    def is_nvme(self) -> bool:
        return self.tier is ServingTier.NVME

    @property
    def is_cxl(self) -> bool:
        return self.tier is ServingTier.CXL_DAX

    @property
    def catalog_digest(self) -> str | None:
        return None if self.catalog is None else self.catalog.digest

    def extent(
        self, layer: int, pages: tuple[int, ...], kind: str, row_bytes: int
    ) -> PageExtent:
        if self.catalog is None:
            raise RuntimeError(f"{self.tier.value} has no page catalog")
        extent = self.catalog.span(
            layer=layer, pages=pages, kind=kind, row_bytes=row_bytes
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
                kind=kind,
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
            "tier_data_path": (
                "host_indexed_copy"
                if self.is_host
                else "gpu_owned_nvme_to_hbm"
                if self.is_nvme
                else "cuda_visible_cxl_direct"
            ),
            "tier_fallback": False,
        }
        if self.nvme is not None:
            capabilities = self.nvme.capabilities
            result["tier_capabilities"] = {
                "translated_iommu": capabilities.translated_iommu,
                "hbm_peer_dma": capabilities.supports_hbm_peer_dma,
                "gpu_doorbell_validated": capabilities.gpu_doorbell_mapping_validated,
                "namespace_read_only": capabilities.namespace_read_only,
                "mapping_backend": capabilities.hbm_mapping_backend.name.lower(),
                "lba_size": capabilities.lba_size,
                "max_transfer_bytes": capabilities.max_transfer_bytes,
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
