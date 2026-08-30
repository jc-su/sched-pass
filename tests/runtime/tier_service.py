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
    NvmeHbmBackendRequirement,
    PageExtent,
    ServingTier,
    ServingTierConfig,
    ServingTierService,
    TierPageCatalog,
    _require_nvme_hbm_backend,
    _validate_nvme_extent,
)
from nta_runtime.resource_contract import (  # noqa: E402
    ResourceAddressSpace,
    ResourceCapability,
    ResourceKind,
    ResourceOwner,
    ResourcePath,
    require_numerical_binding,
    resource_contract,
)
from nta_runtime.storage_identity import (  # noqa: E402
    sglang_storage_key,
    vllm_storage_key,
)
from nta_runtime.runtime_resources import RuntimeResourceConfig  # noqa: E402
import nta_runtime.runtime_resources as runtime_resources  # noqa: E402


def main() -> None:
    assert sglang_storage_key("same") != vllm_storage_key(b"same")
    assert sglang_storage_key("same") == sglang_storage_key("same")
    hbm = resource_contract(ResourceKind.HBM)
    host_mapped = resource_contract(ResourceKind.HOST_MAPPED)
    host_staged = resource_contract(ResourceKind.HOST_STAGED)
    nvme = resource_contract(ResourceKind.NVME)
    cxl = resource_contract(ResourceKind.CXL_DAX)
    assert (
        hbm.direct_numerical_path
        and hbm.protocol_owner is ResourceOwner.ENGINE
        and hbm.payload_owner is ResourceOwner.ENGINE
        and hbm.transfer_destination_owner is None
    )
    assert host_mapped.direct_numerical_path
    assert host_staged.uses_host_proxy and not host_staged.physical
    assert host_staged.capabilities & ResourceCapability.HOST_REGISTERED
    assert host_staged.capabilities & ResourceCapability.INDEXED_TRANSFER
    assert nvme.physical and not nvme.direct_numerical_path
    assert cxl.physical and cxl.direct_numerical_path
    assert not (cxl.capabilities & ResourceCapability.PERSISTENT_STORAGE)
    assert host_staged.directory_owner is ResourceOwner.RUNTIME
    assert host_staged.protocol_owner is ResourceOwner.RUNTIME
    assert host_staged.payload_owner is ResourceOwner.ENGINE
    assert host_staged.transfer_destination_owner is ResourceOwner.ENGINE
    assert host_staged.mapping_owner is None
    assert nvme.protocol_owner is ResourceOwner.TRANSPORT
    assert nvme.payload_owner is ResourceOwner.TRANSPORT
    assert nvme.transfer_destination_owner is ResourceOwner.ENGINE
    assert nvme.mapping_owner is ResourceOwner.TRANSPORT
    assert nvme.numerical_address_owner is ResourceOwner.ENGINE
    assert cxl.numerical_address_owner is ResourceOwner.TRANSPORT
    assert nvme.as_dict()["steady_state_path"] == "nvme_peer_dma_to_engine_hbm"
    assert nvme.as_dict()["transfer_destination_owner"] == "engine"
    assert nvme.as_dict()["mapping_owner"] == "transport"
    assert nvme.as_dict()["numerical_address_owner"] == "engine"
    require_numerical_binding(
        nvme,
        ResourceOwner.ENGINE,
        ResourceAddressSpace.HBM,
        frozenset((ResourcePath.MATERIALIZED,)),
        consumer="test paged-KV consumer",
    )
    try:
        require_numerical_binding(
            cxl,
            ResourceOwner.ENGINE,
            ResourceAddressSpace.HBM,
            frozenset((ResourcePath.MATERIALIZED,)),
            consumer="test paged-KV consumer",
        )
    except RuntimeError as error:
        assert "ready address" in str(error)
    else:
        raise AssertionError("mismatched numerical address ownership was accepted")
    config = RuntimeResourceConfig.with_environment_staging_limit(
        request_capacity=4,
        object_capacity=8,
        intent_capacity=8,
        work_ticket_capacity=8,
        tenant_capacity=4,
        device_ordinal=-1,
    )
    assert config.staging_byte_capacity == (1 << 64) - 1
    assert RuntimeResourceConfig(
        request_capacity=4,
        object_capacity=8,
        intent_capacity=8,
        work_ticket_capacity=8,
        tenant_capacity=4,
        device_ordinal=-1,
        staging_byte_capacity=0,
    ).staging_byte_capacity == (1 << 64) - 1

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
    partial_resources._runtime = CloseProbe(resource_log, "runtime", fail=True)
    partial_resources._tier = CloseProbe(resource_log, "tier", fail=True)
    partial_resources._config = config
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
    assert ServingTierConfig.from_environment().tier is ServingTier.HBM
    assert (
        ServingTierConfig.from_environment({"NTA_SERVING_TIER": "host_staged"}).tier
        is ServingTier.HOST_STAGED
    )
    assert (
        ServingTierConfig.from_environment({"NTA_SERVING_TIER": "host_mapped"}).tier
        is ServingTier.HOST_MAPPED
    )
    nvme_environment = {
        "NTA_SERVING_TIER": "nvme",
        "NTA_NVME_ENDPOINT": "vfio:0000:01:00.0",
        "NTA_TIER_CATALOG": "/tmp/catalog.json",
        "NTA_NVME_HBM_BACKEND": "cuda-dmabuf-ioas",
    }
    assert (
        ServingTierConfig.from_environment(nvme_environment).nvme_hbm_backend
        is NvmeHbmBackendRequirement.CUDA_DMA_BUF_IOAS
    )
    calibrated = ServingTierConfig.from_environment(
        {
            **nvme_environment,
            "NTA_NVME_COMMAND_SERVICE_NS": "13000",
            "NTA_NVME_READ_BANDWIDTH_BPS": "6700000000",
            "NTA_NVME_COMPACTION_BANDWIDTH_BPS": "600000000000",
            "NTA_NVME_COMPACTION_LAUNCH_NS": "20000",
            "NTA_NVME_MINIMUM_GAIN": "1.05",
        }
    ).nvme_service_model
    assert calibrated.calibrated
    assert calibrated.command_service_ns == 13_000
    assert calibrated.compaction_launch_ns == 20_000
    assert calibrated.minimum_gain == 1.05
    try:
        ServingTierConfig.from_environment(
            {**nvme_environment, "NTA_NVME_COMMAND_SERVICE_NS": "13000"}
        )
    except ValueError as error:
        assert "requires command" in str(error)
    else:
        raise AssertionError("a partial NVMe service calibration was accepted")
    _require_nvme_hbm_backend(
        NvmeHbmBackendRequirement.AUTO, "nvidia-peer-pages"
    )
    _require_nvme_hbm_backend(
        NvmeHbmBackendRequirement.CUDA_DMA_BUF_IOAS, "cuda-dmabuf-ioas"
    )
    try:
        _require_nvme_hbm_backend(
            NvmeHbmBackendRequirement.CUDA_DMA_BUF_IOAS,
            "nvidia-peer-pages",
        )
    except RuntimeError as error:
        assert "required=cuda-dmabuf-ioas" in str(error)
    else:
        raise AssertionError("a peer-pages fallback satisfied a module-free contract")
    try:
        ServingTierConfig.from_environment(
            {**nvme_environment, "NTA_NVME_HBM_BACKEND": "guess"}
        )
    except ValueError as error:
        assert "NTA_NVME_HBM_BACKEND" in str(error)
    else:
        raise AssertionError("an unknown NVMe HBM backend was accepted")
    try:
        ServingTierConfig(
            nvme_hbm_backend=NvmeHbmBackendRequirement.CUDA_DMA_BUF_IOAS
        )
    except ValueError as error:
        assert "NVMe tier" in str(error)
    else:
        raise AssertionError("an NVMe backend requirement was applied to HBM")
    try:
        ServingTierConfig(issue_budget=32)
    except ValueError as error:
        assert "NVMe-only" in str(error) and "issue_budget" in str(error)
    else:
        raise AssertionError("an NVMe progress budget was applied to HBM")
    try:
        ServingTierConfig(window_bytes=4096)
    except ValueError as error:
        assert "only CXL" in str(error)
    else:
        raise AssertionError("a CXL window was applied to HBM")
    try:
        ServingTierConfig.from_environment(
            {
                **nvme_environment,
                "NTA_NVME_TRUST_READ_ONLY_DEVICE_CODE": "true",
            }
        )
    except ValueError as error:
        assert "must be 0 or 1" in str(error)
    else:
        raise AssertionError("an ambiguous NVMe trust policy was accepted")
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
    for invalid_extent in ((0.5, 4096), ("0", 4096), (0, "4096")):
        try:
            PageExtent(*invalid_extent)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("non-integer tier extent was accepted")
    try:
        ServingTierConfig(queue_depth="64")  # type: ignore[arg-type]
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("string tier configuration was retained")
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
        "schema": 2,
        "tier": "nvme",
        "format": "typed-components-v1",
        "namespace": "test-model/tp0",
        "page_tokens": 1,
        "layer_count": 1,
        "components": ["key", "value"],
        "alignment_bytes": 4096,
        "entries": [
            {
                "storage_key": "key-a",
                "ordinal": 0,
                "layer": 0,
                "components": {
                    "key": {"offset": 0, "bytes": 4096},
                    "value": {"offset": 8192, "bytes": 4096},
                },
            },
            {
                "storage_key": "key-b",
                "ordinal": 1,
                "layer": 0,
                "components": {
                    "key": {"offset": 4096, "bytes": 4096},
                    "value": {"offset": 12288, "bytes": 4096},
                },
            },
        ],
    }
    with tempfile.TemporaryDirectory(prefix="nta-tier-catalog-") as directory:
        path = Path(directory) / "catalog.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        catalog = TierPageCatalog.load(path, expected_tier=ServingTier.NVME)
        assert catalog.ordinal("key-a") == 0
        assert catalog.storage_key(1) == "key-b"
        assert catalog.namespace == "test-model/tp0"
        assert (
            catalog.span(
                layer=0, ordinals=(0, 1), component="key", row_bytes=4096
            ).bytes
            == 8192
        )
        catalog.validate_nvme_geometry(
            lba_size=4096,
            max_transfer_bytes=8192,
            namespace_bytes=16384,
        )
        try:
            catalog.validate_nvme_geometry(
                lba_size=4096,
                max_transfer_bytes=8192,
                namespace_bytes=12287,
            )
        except ValueError as error:
            assert "opened namespace" in str(error)
        else:
            raise AssertionError("an out-of-namespace catalog was accepted")
        try:
            catalog.validate_nvme_geometry(
                lba_size=8192,
                max_transfer_bytes=8192,
                namespace_bytes=16384,
            )
        except ValueError as error:
            assert "opened controller" in str(error) and "LBA aligned" in str(error)
        else:
            raise AssertionError("a controller-incompatible catalog was accepted")
        try:
            catalog.validate_nvme_geometry(
                lba_size=4096,
                max_transfer_bytes=2048,
                namespace_bytes=16384,
            )
        except ValueError as error:
            assert "max transfer" in str(error)
        else:
            raise AssertionError("an oversized catalog row was accepted at setup")
        assert (
            catalog.span(
                layer=0, ordinals=(0, 1), component="value", row_bytes=4096
            ).offset
            == 8192
        )
        runs = catalog.transfer_runs(
            layer=0,
            storage_keys=("key-b", "key-a"),
            destination_indices=(11, 10),
            component="key",
            row_bytes=4096,
            max_transfer_bytes=8192,
        )
        assert len(runs) == 1
        assert runs[0].source == PageExtent(0, 8192)
        assert runs[0].destination_first == 10 and runs[0].row_count == 2
        bounded_runs = catalog.transfer_runs(
            layer=0,
            storage_keys=("key-a", "key-b"),
            destination_indices=(10, 11),
            component="key",
            row_bytes=4096,
            max_transfer_bytes=4096,
        )
        assert tuple(run.row_count for run in bounded_runs) == (1, 1)
        fragmented_runs = catalog.transfer_runs(
            layer=0,
            storage_keys=("key-a", "key-b"),
            destination_indices=(10, 12),
            component="key",
            row_bytes=4096,
            max_transfer_bytes=8192,
        )
        assert tuple(run.destination_first for run in fragmented_runs) == (10, 12)
        try:
            catalog.transfer_runs(
                layer=0,
                storage_keys=("key-a", "key-b"),
                destination_indices=(10, 10),
                component="key",
                row_bytes=4096,
                max_transfer_bytes=8192,
            )
        except ValueError as error:
            assert "unique" in str(error)
        else:
            raise AssertionError("duplicate transfer destinations were accepted")
        try:
            catalog.span(layer=0, ordinals=(1, 0), component="key", row_bytes=4096)
        except ValueError as error:
            assert "contiguous" in str(error)
        else:
            raise AssertionError("non-contiguous catalog ordinals were accepted")
        overlap = json.loads(json.dumps(document))
        overlap["entries"][1]["components"]["value"]["offset"] = 0
        overlap_path = Path(directory) / "overlap.json"
        overlap_path.write_text(json.dumps(overlap), encoding="utf-8")
        try:
            TierPageCatalog.load(overlap_path, expected_tier=ServingTier.NVME)
        except ValueError as error:
            assert "overlap" in str(error)
        else:
            raise AssertionError("cross-page catalog overlap was accepted")

        fractional = json.loads(json.dumps(document))
        fractional["entries"][0]["components"]["key"]["offset"] = 0.5
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
        assert service.stats()["resource_contract"]["kind"] == "hbm"
        service.close()
        _validate_nvme_extent(
            catalog.span(layer=0, ordinals=(0,), component="key", row_bytes=4096),
            lba_size=4096,
            max_transfer_bytes=8192,
            kind="key",
        )
        try:
            _validate_nvme_extent(
                PageExtent(512, 4096),
                lba_size=4096,
                max_transfer_bytes=8192,
                kind="key",
            )
        except RuntimeError as error:
            assert "offset" in str(error) and "LBA aligned" in str(error)
        else:
            raise AssertionError("misaligned NVMe catalog offset was accepted")
        try:
            _validate_nvme_extent(
                catalog.span(
                    layer=0,
                    ordinals=(0, 1),
                    component="key",
                    row_bytes=4096,
                ),
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
