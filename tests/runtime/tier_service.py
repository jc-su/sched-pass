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


def main() -> None:
    assert ServingTierConfig.from_environment().tier is ServingTier.HOST_STAGED
    try:
        ServingTierConfig.from_environment({"NTA_SERVING_TIER": "nvme"})
    except ValueError as error:
        assert "NTA_NVME_ENDPOINT" in str(error)
    else:
        raise AssertionError("NVMe selection silently accepted a missing endpoint")

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
        assert ServingTierService(ServingTierConfig()).stats()["tier_fallback"] is False
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
    print("tier_service=pass")


if __name__ == "__main__":
    main()
