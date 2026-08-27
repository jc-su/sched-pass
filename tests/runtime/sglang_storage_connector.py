#!/usr/bin/env python3
"""Stable-key and host-row lifetime gates for the SGLang storage connector."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile

import torch
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)

from nta_runtime.connectors.sglang_storage import (
    NtaSglangStorage,
    resolve_sglang_storage_keys,
    validate_sglang_storage_backend,
)
from nta_runtime.storage_identity import sglang_storage_key


class _HostPool:
    page_size = 1
    layer_num = 1

    def __init__(self) -> None:
        self.next_rows = torch.tensor([3], dtype=torch.int64)

    def alloc(self, _count: int) -> torch.Tensor:
        return self.next_rows.clone()


def _catalog(path: Path) -> None:
    document = {
        "schema": 2,
        "tier": "nvme",
        "format": "typed-components-v1",
        "namespace": "connector-test/tp0",
        "page_tokens": 1,
        "layer_count": 1,
        "components": ["key", "value"],
        "alignment_bytes": 512,
        "window_bytes": 2048,
        "entries": [
            {
                "storage_key": sglang_storage_key("key-a"),
                "ordinal": 0,
                "layer": 0,
                "components": {
                    "key": {"offset": 0, "bytes": 512},
                    "value": {"offset": 1024, "bytes": 512},
                },
            },
            {
                "storage_key": sglang_storage_key("key-b"),
                "ordinal": 1,
                "layer": 0,
                "components": {
                    "key": {"offset": 512, "bytes": 512},
                    "value": {"offset": 1536, "bytes": 512},
                },
            },
        ],
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def _extra() -> dict[str, object]:
    return {
        "backend_name": "nta",
        "module_path": "nta_runtime.connectors.sglang_storage",
        "class_name": "NtaSglangStorage",
        "namespace": "connector-test/tp0",
        "interface_v1": 1,
    }


def _storage_config(extra: dict[str, object]) -> HiCacheStorageConfig:
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=True,
        is_page_first_layout=True,
        model_name="connector-test",
        extra_config=extra,
    )


def main() -> None:
    previous_tier = os.environ.get("NTA_SERVING_TIER")
    previous_catalog = os.environ.get("NTA_TIER_CATALOG")
    with tempfile.TemporaryDirectory(prefix="nta-sglang-storage-") as directory:
        catalog_path = Path(directory) / "catalog.json"
        _catalog(catalog_path)
        os.environ["NTA_SERVING_TIER"] = "nvme"
        os.environ["NTA_TIER_CATALOG"] = str(catalog_path)
        extra = _extra()
        args = SimpleNamespace(
            hicache_storage_backend="dynamic",
            hicache_storage_backend_extra_config=json.dumps(extra),
        )
        validate_sglang_storage_backend(args, expected_namespace="connector-test/tp0")
        wrong = dict(extra)
        wrong["interface_v1"] = 0
        args.hicache_storage_backend_extra_config = json.dumps(wrong)
        try:
            validate_sglang_storage_backend(
                args, expected_namespace="connector-test/tp0"
            )
        except RuntimeError as error:
            assert "interface_v1" in str(error)
        else:
            raise AssertionError("generic SGLang payload API was accepted")

        storage = NtaSglangStorage(_storage_config(extra))
        pool = _HostPool()
        original_alloc = pool.alloc
        storage.register_mem_pool_host(pool)
        assert storage.batch_exists(["key-a", "key-b", "missing"]) == 2
        hit = storage.batch_exists_v2(["key-a", "key-b"])
        assert hit.kv_hit_pages == 2

        rows = torch.tensor([3, 4], dtype=torch.int64)
        results = storage.batch_get_v1(["key-a", "missing"], rows)
        assert results == [True, False]
        assert resolve_sglang_storage_keys(pool, rows[:1]) == (
            sglang_storage_key("key-a"),
        )
        try:
            resolve_sglang_storage_keys(pool, rows[1:])
        except RuntimeError as error:
            assert "no stable storage-key" in str(error)
        else:
            raise AssertionError("a storage miss retained a host-row identity")

        # Host slots are recyclable. Allocation must erase an old content
        # identity before the row can be repurposed by any L2 path.
        allocated = pool.alloc(1)
        assert allocated.tolist() == [3]
        try:
            resolve_sglang_storage_keys(pool, allocated)
        except RuntimeError as error:
            assert "no stable storage-key" in str(error)
        else:
            raise AssertionError("a recycled host row retained a stale identity")

        storage.batch_get_v2(
            [
                PoolTransfer(
                    name=PoolName.KV,
                    host_indices=torch.tensor([3], dtype=torch.int64),
                    keys=["key-b"],
                )
            ]
        )
        assert resolve_sglang_storage_keys(pool, allocated) == (
            sglang_storage_key("key-b"),
        )
        storage.close()
        assert pool.alloc == original_alloc
        try:
            resolve_sglang_storage_keys(pool, allocated)
        except RuntimeError as error:
            assert "no NTA storage connector" in str(error)
        else:
            raise AssertionError("connector close leaked a host-row registry")

    if previous_tier is None:
        os.environ.pop("NTA_SERVING_TIER", None)
    else:
        os.environ["NTA_SERVING_TIER"] = previous_tier
    if previous_catalog is None:
        os.environ.pop("NTA_TIER_CATALOG", None)
    else:
        os.environ["NTA_TIER_CATALOG"] = previous_catalog
    print("sglang_storage_connector=pass")


if __name__ == "__main__":
    main()
