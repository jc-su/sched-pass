"""Contract tests for explicit serving-tier selection and exact page spans."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch


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
    ResourceCapability,
    ResourceKind,
    ResourceOwner,
    resource_contract,
)
from nta_runtime.runtime_resources import RuntimeResourceConfig  # noqa: E402
import nta_runtime.runtime_resources as runtime_resources  # noqa: E402


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
    assert not (cxl.capabilities & ResourceCapability.PERSISTENT_STORAGE)
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
    class FakeTier:
        contract = host_staged
        nvme = None
        cxl = None
        closed = False

        def close(self) -> None:
            self.closed = True

    class FailingRuntime:
        closed = False

        def __init__(self, *_args, **_kwargs) -> None:
            self.constructor_arguments = (_args, _kwargs)

        def tier_descriptor(self, _tier) -> object:
            raise RuntimeError("descriptor query failed")

        def close(self) -> None:
            self.closed = True

    class CloseProbe:
        def __init__(self, log: list[str], name: str, fail: bool = False) -> None:
            self.log = log
            self.name = name
            self.fail = fail

        def close(self) -> None:
            self.log.append(self.name)
            if self.fail:
                raise RuntimeError("synthetic transport close failure")

    close_log: list[str] = []
    partial_service = ServingTierService.__new__(ServingTierService)
    partial_service.nvme = CloseProbe(close_log, "nvme", fail=True)
    partial_service.cxl = CloseProbe(close_log, "cxl")
    try:
        partial_service.close()
    except RuntimeError as error:
        assert "teardown" in str(error)
    else:
        raise AssertionError("tier close failure was silently accepted")
    assert close_log == ["nvme", "cxl"]
    assert partial_service.nvme is None and partial_service.cxl is None
    resource_log: list[str] = []
    partial_resources = runtime_resources.ServingRuntimeResources.__new__(
        runtime_resources.ServingRuntimeResources
    )
    partial_resources.runtime = CloseProbe(resource_log, "runtime", fail=True)
    partial_resources.tier = CloseProbe(resource_log, "tier", fail=True)
    partial_resources._closed = False
    try:
        partial_resources.close()
    except RuntimeError as error:
        assert "resource teardown" in str(error)
    else:
        raise AssertionError("runtime resource close failure was silently accepted")
    assert resource_log == ["runtime", "tier"]

    fake_tier = FakeTier()
    failing_runtime = FailingRuntime()
    with (
        patch.object(runtime_resources, "ServingTierService", return_value=fake_tier),
        patch.object(
            runtime_resources, "_create_runtime", return_value=failing_runtime
        ),
    ):
        try:
            runtime_resources.ServingRuntimeResources.open(
                tier_config=ServingTierConfig(), runtime_config=config
            )
        except RuntimeError as error:
            assert "descriptor query failed" in str(error)
        else:
            raise AssertionError("descriptor failure was silently accepted")
    assert failing_runtime.closed and fake_tier.closed
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
        assert "window_bytes" in str(error)
    else:
        raise AssertionError("an out-of-range tier window was accepted")
    for field in ("namespace_id", "queue_depth", "issue_budget", "progress_rounds"):
        try:
            ServingTierConfig(**{field: 1 << 32})
        except ValueError as error:
            assert field in str(error)
        else:
            raise AssertionError(f"{field} overflow was accepted")
    try:
        ServingTierConfig(device_ordinal=1 << 31)
    except ValueError as error:
        assert "device_ordinal" in str(error)
    else:
        raise AssertionError("device ordinal overflow was accepted")
    try:
        RuntimeResourceConfig(
            request_capacity=1 << 32,
            object_capacity=8,
            intent_capacity=8,
            work_ticket_capacity=8,
            tenant_capacity=4,
            device_ordinal=-1,
        )
    except ValueError as error:
        assert "request_capacity" in str(error)
    else:
        raise AssertionError("runtime uint32 capacity overflow was accepted")

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
