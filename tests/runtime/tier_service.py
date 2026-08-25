"""Contract tests for explicit serving-tier selection and exact page spans."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from nta_runtime.tier import (  # noqa: E402
    ServingTier,
    ServingTierConfig,
    ServingTierService,
    TierPageCatalog,
    _validate_nvme_extent,
)
from nta_runtime.resource_contract import (  # noqa: E402
    ResourceKind,
    ResourceOwner,
    resource_contract,
)
from nta_runtime.runtime_resources import RuntimeResourceConfig  # noqa: E402


def main() -> None:
    hbm = resource_contract(ResourceKind.HBM)
    host_mapped = resource_contract(ResourceKind.HOST_MAPPED)
    host_staged = resource_contract(ResourceKind.HOST_STAGED)
    nvme = resource_contract(ResourceKind.NVME)
    cxl = resource_contract(ResourceKind.CXL_DAX)
    assert (
        hbm.direct_device_visible
        and hbm.protocol_owner is ResourceOwner.ENGINE
        and hbm.payload_owner is ResourceOwner.ENGINE
        and hbm.transfer_destination_owner is None
    )
    assert host_mapped.direct_device_visible
    assert host_staged.uses_host_proxy and not host_staged.physical
    assert nvme.physical and not nvme.direct_device_visible
    assert cxl.physical and cxl.direct_device_visible
    assert host_staged.directory_owner is ResourceOwner.RUNTIME
    assert host_staged.protocol_owner is ResourceOwner.RUNTIME
    assert host_staged.payload_owner is ResourceOwner.ENGINE
    assert host_staged.transfer_destination_owner is ResourceOwner.RUNTIME
    assert nvme.protocol_owner is ResourceOwner.TRANSPORT
    assert nvme.payload_owner is ResourceOwner.TRANSPORT
    assert nvme.transfer_destination_owner is ResourceOwner.TRANSPORT
    assert nvme.as_dict()["steady_state_path"] == "gpu_owned_nvme_to_hbm"
    assert nvme.as_dict()["transfer_destination_owner"] == "transport"
    config = RuntimeResourceConfig.with_environment_staging_limit(
        request_capacity=4,
        object_capacity=8,
        intent_capacity=8,
        work_ticket_capacity=8,
        tenant_capacity=4,
        device_ordinal=-1,
    )
    assert config.staging_byte_capacity == (1 << 64) - 1
    try:
        RuntimeResourceConfig(
            request_capacity=0,
            object_capacity=8,
            intent_capacity=8,
            work_ticket_capacity=8,
            tenant_capacity=4,
            device_ordinal=-1,
        )
    except ValueError as error:
        assert "request_capacity" in str(error)
    else:
        raise AssertionError("invalid runtime resource capacity was accepted")
    assert ServingTierConfig.from_environment().tier is ServingTier.HOST_STAGED
    try:
        ServingTierConfig.from_environment({"NTA_SERVING_TIER": "nvme"})
    except ValueError as error:
        assert "NTA_NVME_ENDPOINT" in str(error)
    else:
        raise AssertionError("NVMe selection silently accepted a missing endpoint")
    try:
        ServingTierConfig(tier=ServingTier.NVME)
    except ValueError as error:
        assert "endpoint" in str(error)
    else:
        raise AssertionError("invalid physical tier config was accepted")
    try:
        ServingTierConfig(window_bytes=1 << 64)
    except ValueError as error:
        assert "uint64" in str(error)
    else:
        raise AssertionError("an out-of-range tier window was accepted")

    document = {
        "schema": 1,
        "tier": "nvme",
        "alignment_bytes": 4096,
        "pages": [
            {
                "layer": 3,
                "page": 7,
                "key": {"offset": 0, "bytes": 4096},
                "value": {"offset": 8192, "bytes": 4096},
            },
            {
                "layer": 3,
                "page": 8,
                "key": {"offset": 4096, "bytes": 4096},
                "value": {"offset": 12288, "bytes": 4096},
            },
        ],
    }
    with tempfile.TemporaryDirectory(prefix="nta-tier-catalog-") as directory:
        path = Path(directory) / "catalog.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        catalog = TierPageCatalog.load(path, expected_tier=ServingTier.NVME)
        assert (
            catalog.span(layer=3, pages=(7, 8), kind="key", row_bytes=4096).bytes
            == 8192
        )
        assert (
            catalog.span(layer=3, pages=(7, 8), kind="value", row_bytes=4096).offset
            == 8192
        )
        try:
            catalog.span(layer=3, pages=(8, 7), kind="key", row_bytes=4096)
        except ValueError as error:
            assert "contiguous" in str(error)
        else:
            raise AssertionError("non-contiguous catalog pages were accepted")
        overlap = json.loads(json.dumps(document))
        overlap["pages"][1]["value"]["offset"] = 0
        overlap_path = Path(directory) / "overlap.json"
        overlap_path.write_text(json.dumps(overlap), encoding="utf-8")
        try:
            TierPageCatalog.load(overlap_path, expected_tier=ServingTier.NVME)
        except ValueError as error:
            assert "overlap" in str(error)
        else:
            raise AssertionError("cross-page catalog overlap was accepted")

        fractional = json.loads(json.dumps(document))
        fractional["pages"][0]["key"]["offset"] = 0.5
        fractional_path = Path(directory) / "fractional.json"
        fractional_path.write_text(json.dumps(fractional), encoding="utf-8")
        try:
            TierPageCatalog.load(fractional_path, expected_tier=ServingTier.NVME)
        except ValueError as error:
            assert "integer" in str(error)
        else:
            raise AssertionError("fractional catalog extent was accepted")

        config = ServingTierConfig.from_environment(
            {
                "NTA_SERVING_TIER": "nvme",
                "NTA_NVME_ENDPOINT": "/dev/nvme-test",
                "NTA_TIER_CATALOG": str(path),
            }
        )
        assert config.catalog_path == path
        service = ServingTierService(ServingTierConfig())
        assert service.stats()["tier_fallback"] is False
        assert service.stats()["resource_contract"]["kind"] == "host_staged"
        service.close()
        _validate_nvme_extent(
            catalog.span(layer=3, pages=(7,), kind="key", row_bytes=4096),
            lba_size=4096,
            max_transfer_bytes=8192,
            kind="key",
        )
        try:
            _validate_nvme_extent(
                catalog.span(layer=3, pages=(7, 8), kind="key", row_bytes=4096),
                lba_size=4096,
                max_transfer_bytes=4096,
                kind="key",
            )
        except RuntimeError as error:
            assert "max transfer" in str(error)
        else:
            raise AssertionError("oversized NVMe extent was not rejected")
        huge_window = json.loads(json.dumps(document))
        huge_window["window_bytes"] = 1 << 64
        huge_window_path = Path(directory) / "huge-window.json"
        huge_window_path.write_text(json.dumps(huge_window), encoding="utf-8")
        try:
            TierPageCatalog.load(huge_window_path, expected_tier=ServingTier.NVME)
        except ValueError as error:
            assert "uint64" in str(error)
        else:
            raise AssertionError("an out-of-range catalog window was accepted")
    print("tier_service=pass")


if __name__ == "__main__":
    main()
