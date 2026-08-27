"""Stable SGLang HiCache storage-key connector for physical NTA tiers.

The SGLang storage API already computes content-stable prefix-page keys.  This
backend deliberately moves metadata only: it proves which keys exist in an
immutable NTA physical-tier catalog and binds each temporary host-cache row to
its stable key.  The NTA attention consumer then resolves that key directly to
NVMe/CXL extents and transfers into the current GPU destination.  No payload is
read through host memory in serving mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import threading
import tomllib
import weakref
from typing import Any

import torch
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)

from nta_runtime.tier import ServingTier, TierPageCatalog
from nta_runtime.storage_identity import sglang_storage_key


@dataclass
class _HostRowBindings:
    page_tokens: int
    rows: dict[int, str] = field(default_factory=dict)
    generation: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    original_alloc: Any = None
    wrapped_alloc: Any = None

    def bind(
        self,
        keys: tuple[str, ...],
        indices: tuple[int, ...],
        present: tuple[bool, ...],
    ) -> None:
        if len(indices) != len(keys) * self.page_tokens:
            raise RuntimeError("SGLang storage keys and host rows disagree")
        if len(present) != len(keys):
            raise RuntimeError("SGLang storage results and keys disagree")
        with self.lock:
            for key_index, key in enumerate(keys):
                begin = key_index * self.page_tokens
                for row in indices[begin : begin + self.page_tokens]:
                    if present[key_index]:
                        self.rows[row] = key
                    else:
                        self.rows.pop(row, None)
            self.generation += 1

    def invalidate(self, indices: tuple[int, ...]) -> None:
        with self.lock:
            for row in indices:
                self.rows.pop(row, None)
            self.generation += 1

    def attach_allocator(self, host_pool: Any) -> None:
        if self.wrapped_alloc is not None:
            return
        original = getattr(host_pool, "alloc", None)
        if not callable(original):
            raise RuntimeError("SGLang host pool has no callable allocator")

        def allocate(*args: Any, **kwargs: Any) -> Any:
            allocated = original(*args, **kwargs)
            if allocated is not None:
                self.invalidate(_indices(allocated))
            return allocated

        self.original_alloc = original
        self.wrapped_alloc = allocate
        try:
            host_pool.alloc = allocate
        except (AttributeError, TypeError) as error:
            self.original_alloc = None
            self.wrapped_alloc = None
            raise RuntimeError(
                "SGLang host-pool allocator cannot publish lifecycle invalidations"
            ) from error

    def detach_allocator(self, host_pool: Any) -> None:
        wrapped, original = self.wrapped_alloc, self.original_alloc
        if wrapped is not None and getattr(host_pool, "alloc", None) is wrapped:
            host_pool.alloc = original
        self.wrapped_alloc = None
        self.original_alloc = None
        with self.lock:
            self.rows.clear()
            self.generation += 1

    def resolve(self, indices: tuple[int, ...]) -> tuple[str, ...]:
        with self.lock:
            missing = tuple(row for row in indices if row not in self.rows)
            if missing:
                raise RuntimeError(
                    "physical HiCache rows have no stable storage-key binding: "
                    f"{missing[:8]}"
                )
            return tuple(self.rows[row] for row in indices)


_BINDINGS_LOCK = threading.Lock()
_BINDINGS: dict[int, tuple[weakref.ReferenceType[Any], _HostRowBindings]] = {}


def _indices(values: torch.Tensor) -> tuple[int, ...]:
    if not isinstance(values, torch.Tensor) or values.ndim != 1:
        raise RuntimeError("SGLang storage indices must be a one-dimensional tensor")
    result = tuple(int(value) for value in values.detach().to(device="cpu").tolist())
    if any(value < 0 for value in result):
        raise RuntimeError("SGLang storage indices must be nonnegative")
    return result


def _bindings_for(
    host_pool: Any, *, page_tokens: int | None = None
) -> _HostRowBindings:
    identity = id(host_pool)
    with _BINDINGS_LOCK:
        existing = _BINDINGS.get(identity)
        if existing is not None and existing[0]() is host_pool:
            bindings = existing[1]
            if page_tokens is not None and bindings.page_tokens != page_tokens:
                raise RuntimeError("SGLang host-pool page geometry changed")
            return bindings
        if page_tokens is None:
            raise RuntimeError("SGLang host pool has no NTA storage connector")
        bindings = _HostRowBindings(page_tokens)
        bindings.attach_allocator(host_pool)

        def discard(_reference: weakref.ReferenceType[Any]) -> None:
            with _BINDINGS_LOCK:
                current = _BINDINGS.get(identity)
                if current is not None and current[0] is _reference:
                    _BINDINGS.pop(identity, None)

        reference = weakref.ref(host_pool, discard)
        _BINDINGS[identity] = (reference, bindings)
        return bindings


def _release_bindings_for(host_pool: Any) -> None:
    identity = id(host_pool)
    with _BINDINGS_LOCK:
        existing = _BINDINGS.get(identity)
        if existing is None or existing[0]() is not host_pool:
            return
        _BINDINGS.pop(identity, None)
        bindings = existing[1]
    bindings.detach_allocator(host_pool)


def _load_storage_extra_config(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str) or not raw.strip():
        raise RuntimeError(
            "physical NTA storage requires SGLang backend extra configuration"
        )
    value = raw.strip()
    if not value.startswith("@"):
        try:
            document = json.loads(value)
        except json.JSONDecodeError as error:
            raise RuntimeError("invalid SGLang storage backend JSON") from error
    else:
        path = Path(value[1:]).expanduser()
        try:
            suffix = path.suffix.lower()
            if suffix == ".json":
                document = json.loads(path.read_text(encoding="utf-8"))
            elif suffix == ".toml":
                with path.open("rb") as stream:
                    document = tomllib.load(stream)
            elif suffix in {".yaml", ".yml"}:
                import yaml

                document = yaml.safe_load(path.read_text(encoding="utf-8"))
            else:
                raise RuntimeError(
                    "SGLang storage config files must be JSON, TOML, or YAML"
                )
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"cannot load SGLang storage backend config {path}"
            ) from error
    if not isinstance(document, dict):
        raise RuntimeError("SGLang storage backend config must be an object")
    return document


def _validate_storage_extra_config(
    extra: dict[str, Any], *, expected_namespace: str
) -> None:
    expected = {
        "backend_name": "nta",
        "module_path": "nta_runtime.connectors.sglang_storage",
        "class_name": "NtaSglangStorage",
        "namespace": expected_namespace,
    }
    for name, value in expected.items():
        if extra.get(name) != value:
            raise RuntimeError(f"SGLang physical storage {name} must be {value!r}")
    if extra.get("interface_v1") != 1:
        raise RuntimeError(
            "SGLang physical storage requires interface_v1=1 so metadata binds "
            "replace generic host-payload reads"
        )


def validate_sglang_storage_backend(
    server_args: Any, *, expected_namespace: str
) -> None:
    """Validate that SGLang will instantiate the exact metadata connector."""

    if getattr(server_args, "hicache_storage_backend", None) != "dynamic":
        raise RuntimeError("SGLang physical storage backend must be dynamic")
    extra = _load_storage_extra_config(
        getattr(server_args, "hicache_storage_backend_extra_config", None)
    )
    _validate_storage_extra_config(extra, expected_namespace=expected_namespace)


def resolve_sglang_storage_keys(
    host_pool: Any, host_indices: torch.Tensor
) -> tuple[str, ...]:
    """Resolve leased host rows to immutable physical storage keys."""

    return _bindings_for(host_pool).resolve(_indices(host_indices))


def maybe_resolve_sglang_storage_keys(
    host_pool: Any, host_indices: torch.Tensor
) -> tuple[str, ...] | None:
    """Return no binding for ordinary L2 HiCache; fail on partial NTA state."""

    identity = id(host_pool)
    with _BINDINGS_LOCK:
        existing = _BINDINGS.get(identity)
        if existing is None or existing[0]() is not host_pool:
            return None
        bindings = existing[1]
    return bindings.resolve(_indices(host_indices))


class NtaSglangStorage(HiCacheStorage):
    """Metadata-only dynamic HiCache backend for immutable physical payloads."""

    def __init__(
        self, storage_config: HiCacheStorageConfig, _factory_kwargs: Any = None
    ) -> None:
        del _factory_kwargs
        raw_tier = os.environ.get("NTA_SERVING_TIER", "").strip().lower()
        raw_tier = {"cxl": "cxl_dax"}.get(raw_tier, raw_tier)
        if raw_tier not in {ServingTier.NVME.value, ServingTier.CXL_DAX.value}:
            raise RuntimeError(
                "NtaSglangStorage is valid only for an explicit physical NTA tier"
            )
        catalog_path = os.environ.get("NTA_TIER_CATALOG", "").strip()
        if not catalog_path:
            raise RuntimeError("NtaSglangStorage requires NTA_TIER_CATALOG")
        self.catalog = TierPageCatalog.load(
            catalog_path, expected_tier=ServingTier(raw_tier)
        )
        extra = storage_config.extra_config or {}
        _validate_storage_extra_config(extra, expected_namespace=self.catalog.namespace)
        if storage_config.is_mla_model:
            raise RuntimeError("NTA physical SGLang storage currently requires MHA")
        if not storage_config.is_page_first_layout:
            raise RuntimeError(
                "NTA physical SGLang storage requires page_first host layout"
            )
        self.storage_config = storage_config
        self.registered_pools: dict[Any, Any] = {}
        self._binding_pools: dict[int, Any] = {}
        self._stats: dict[str, int] = {
            "exists_queries": 0,
            "metadata_get_pages": 0,
            "catalog_set_hits": 0,
            "catalog_set_misses": 0,
        }
        self._stats_lock = threading.Lock()

    def _register_pool(self, host_pool: Any, name: Any) -> None:
        page_tokens = int(getattr(host_pool, "page_size", 0))
        if page_tokens != self.catalog.page_tokens or page_tokens != 1:
            raise RuntimeError(
                "NTA physical SGLang storage requires catalog and host page_tokens=1"
            )
        layer_count = int(getattr(host_pool, "layer_num", 0))
        if layer_count != self.catalog.layer_count:
            raise RuntimeError(
                "SGLang host-pool layer count does not match the physical catalog"
            )
        self.registered_pools[name] = host_pool
        if id(host_pool) not in self._binding_pools:
            _bindings_for(host_pool, page_tokens=page_tokens)
            self._binding_pools[id(host_pool)] = host_pool

    def register_mem_pool_host(self, mem_pool_host: Any) -> None:
        self.mem_pool_host = mem_pool_host
        self._register_pool(mem_pool_host, PoolName.KV)

    def register_mem_host_pool_v2(self, host_pool: Any, host_pool_name: Any) -> None:
        self._register_pool(host_pool, host_pool_name)

    def _pool(self, name: Any) -> Any:
        try:
            return self.registered_pools[name]
        except KeyError:
            raise RuntimeError(
                f"SGLang storage pool {name!s} was not registered"
            ) from None

    def _bind_transfer(self, transfer: PoolTransfer) -> list[bool]:
        keys = tuple(sglang_storage_key(str(key)) for key in (transfer.keys or ()))
        if transfer.host_indices is None:
            return [False] * len(keys)
        rows = _indices(transfer.host_indices)
        results = tuple(self.catalog.has_storage_key(key) for key in keys)
        pool = self._pool(transfer.name)
        _bindings_for(pool).bind(keys, rows, results)
        return list(results)

    def batch_exists_v2(
        self,
        keys: list[str],
        pool_transfers: list[PoolTransfer] | None = None,
        extra_info: Any = None,
    ) -> PoolTransferResult:
        del extra_info
        if pool_transfers:
            raise RuntimeError("NTA physical catalog has no auxiliary SGLang pools")
        with self._stats_lock:
            self._stats["exists_queries"] += len(keys)
        count = 0
        for key in keys:
            if not self.catalog.has_storage_key(sglang_storage_key(str(key))):
                break
            count += 1
        return PoolTransferResult(count, {})

    def batch_get_v2(
        self, transfers: list[PoolTransfer], extra_info: Any = None
    ) -> dict[Any, list[bool]]:
        del extra_info
        result: dict[Any, list[bool]] = {}
        for transfer in transfers:
            values = self._bind_transfer(transfer)
            result[transfer.name] = values
            with self._stats_lock:
                self._stats["metadata_get_pages"] += sum(values)
        return result

    def batch_get_v1(
        self, keys: list[str], host_indices: torch.Tensor, extra_info: Any = None
    ) -> list[bool]:
        del extra_info
        transfer = PoolTransfer(
            name=PoolName.KV,
            host_indices=host_indices,
            keys=list(keys),
        )
        return self.batch_get_v2([transfer])[PoolName.KV]

    def batch_set_v2(
        self, transfers: list[PoolTransfer], extra_info: Any = None
    ) -> dict[Any, list[bool]]:
        del extra_info
        result: dict[Any, list[bool]] = {}
        for transfer in transfers:
            values = self._bind_transfer(transfer)
            result[transfer.name] = values
            with self._stats_lock:
                self._stats["catalog_set_hits"] += sum(values)
                self._stats["catalog_set_misses"] += len(values) - sum(values)
        return result

    def batch_set_v1(
        self, keys: list[str], host_indices: torch.Tensor, extra_info: Any = None
    ) -> list[bool]:
        del extra_info
        transfer = PoolTransfer(
            name=PoolName.KV,
            host_indices=host_indices,
            keys=list(keys),
        )
        return self.batch_set_v2([transfer])[PoolName.KV]

    def get(
        self, key: str, target_location: Any = None, target_sizes: Any = None
    ) -> None:
        del key, target_location, target_sizes
        # Payload reads through this generic API would traverse host memory and
        # violate the selected tier contract. Only batch_get_v2 metadata binds
        # are legal; the typed attention consumer performs the actual read.
        return None

    def batch_get(
        self,
        keys: list[str],
        target_locations: Any = None,
        target_sizes: Any = None,
    ) -> list[None]:
        del target_locations, target_sizes
        return [None] * len(keys)

    def set(
        self,
        key: str,
        value: Any = None,
        target_location: Any = None,
        target_sizes: Any = None,
    ) -> bool:
        del value, target_location, target_sizes
        return self.catalog.has_storage_key(sglang_storage_key(str(key)))

    def batch_set(
        self,
        keys: list[str],
        values: Any = None,
        target_locations: Any = None,
        target_sizes: Any = None,
    ) -> bool:
        del values, target_locations, target_sizes
        return all(
            self.catalog.has_storage_key(sglang_storage_key(str(key))) for key in keys
        )

    def exists(self, key: str) -> bool:
        with self._stats_lock:
            self._stats["exists_queries"] += 1
        return self.catalog.has_storage_key(sglang_storage_key(str(key)))

    def clear(self) -> None:
        raise RuntimeError(
            "immutable NTA physical storage cannot be cleared at runtime"
        )

    def close(self) -> None:
        pools = tuple(self._binding_pools.values())
        self._binding_pools.clear()
        self.registered_pools.clear()
        for pool in pools:
            _release_bindings_for(pool)

    def get_stats(self) -> dict[str, int]:
        with self._stats_lock:
            return dict(self._stats)
