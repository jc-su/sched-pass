"""Official vLLM 0.26 scheduler/worker lifecycle seam for NTA.

The connector carries exact request IDs and block ownership to the worker.  It
does not copy KV through vLLM's layer hooks: NTA's typed attention consumer owns
availability and transport.  Clearing connector metadata after validation
keeps those generic per-layer hooks out of the steady-state path.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import os
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)

from nta_runtime.adapters.vllm_v1 import current_vllm_v1_forward_state
from nta_runtime.requests import stable_request_id
from nta_runtime.storage_identity import vllm_storage_key
from nta_runtime.tier import ServingTier, TierPageCatalog

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_interface import KVCacheConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request
    from vllm.config import VllmConfig


SUPPORTED_VLLM_VERSION = "0.26.0"


@dataclass(frozen=True)
class NtaVllmRequestMetadata:
    request_id: str
    block_ids: tuple[int, ...]
    storage_keys: tuple[str | None, ...]
    num_scheduled_tokens: int

    def __post_init__(self) -> None:
        if not self.request_id or self.num_scheduled_tokens <= 0:
            raise ValueError("NTA vLLM metadata requires scheduled request identity")
        if not self.block_ids or any(block < 0 for block in self.block_ids):
            raise ValueError("NTA vLLM metadata requires nonnegative exact blocks")
        if len(set(self.block_ids)) != len(self.block_ids):
            raise ValueError("NTA vLLM metadata block IDs must be unique")
        if len(self.storage_keys) != len(self.block_ids):
            raise ValueError("NTA vLLM storage keys must align with block IDs")
        if any(key is not None and not key for key in self.storage_keys):
            raise ValueError("NTA vLLM storage keys must be non-empty")


@dataclass(frozen=True)
class NtaVllmConnectorMetadata(KVConnectorMetadata):
    requests: tuple[NtaVllmRequestMetadata, ...]
    finished_request_ids: tuple[str, ...]
    host_transfer: Any | None = None

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(request.request_id for request in self.requests)

    @property
    def block_tables(self) -> tuple[tuple[int, ...], ...]:
        return tuple(request.block_ids for request in self.requests)

    @property
    def storage_key_tables(self) -> tuple[tuple[str | None, ...], ...]:
        return tuple(request.storage_keys for request in self.requests)


@dataclass(frozen=True)
class _ExternalMatch:
    first_block: int
    storage_keys: tuple[str, ...]
    token_count: int


class NtaVllmConnector(KVConnectorBase_V1, SupportsHMA):
    """Carry exact lifecycle metadata; numerical acquisition remains NTA-owned."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        if importlib.metadata.version("vllm") != SUPPORTED_VLLM_VERSION:
            raise RuntimeError(f"NTA connector requires vLLM {SUPPORTED_VLLM_VERSION}")
        super().__init__(vllm_config, role, kv_cache_config)
        groups = tuple(getattr(kv_cache_config, "kv_cache_groups", ()))
        if len(groups) != 1:
            raise RuntimeError("NTA vLLM connector requires exactly one KV group")
        if getattr(vllm_config, "speculative_config", None) is not None:
            raise RuntimeError(
                "NTA vLLM connector does not yet support speculative decoding"
            )
        self._block_ids: dict[str, tuple[int, ...]] = {}
        self._storage_keys_by_block: dict[str, dict[int, str]] = {}
        self._external_matches: dict[str, _ExternalMatch] = {}
        self._pending_finished: set[str] = set()
        raw_tier = os.environ.get("NTA_SERVING_TIER", "hbm").strip().lower()
        try:
            tier = ServingTier(raw_tier)
        except ValueError as error:
            raise RuntimeError(f"unsupported NTA vLLM tier {raw_tier!r}") from error
        if tier not in {ServingTier.HBM, ServingTier.HOST_STAGED, ServingTier.NVME}:
            raise RuntimeError(
                f"NTA vLLM does not have a numerical {tier.value} payload path"
            )
        self._host_scheduler: Any | None = None
        self._host_worker: Any | None = None
        if tier is ServingTier.HOST_STAGED:
            from nta_runtime.connectors.vllm_host import (
                VllmHostScheduler,
                VllmHostWorker,
            )

            if role is KVConnectorRole.SCHEDULER:
                self._host_scheduler = VllmHostScheduler(
                    vllm_config, kv_cache_config
                )
            elif role is KVConnectorRole.WORKER:
                self._host_worker = VllmHostWorker(vllm_config, kv_cache_config)
            else:  # pragma: no cover - pinned enum has exactly two roles
                raise RuntimeError(f"unsupported vLLM connector role {role!r}")
        self.catalog: TierPageCatalog | None = None
        self._block_size = 0
        if tier is ServingTier.NVME:
            catalog_path = os.environ.get("NTA_TIER_CATALOG", "").strip()
            if not catalog_path:
                raise RuntimeError("vLLM physical storage requires NTA_TIER_CATALOG")
            self.catalog = TierPageCatalog.load(catalog_path, expected_tier=tier)
            spec = getattr(groups[0], "kv_cache_spec", None)
            self._block_size = int(getattr(spec, "block_size", 0))
            layer_names = tuple(getattr(groups[0], "layer_names", ()))
            if self._block_size <= 0 or not layer_names:
                raise RuntimeError("vLLM physical storage has incomplete KV geometry")
            if self.catalog.page_tokens != self._block_size:
                raise RuntimeError(
                    "vLLM block size does not match the physical catalog"
                )
            if self.catalog.layer_count != len(layer_names):
                raise RuntimeError(
                    "vLLM layer count does not match the physical catalog"
                )
            if self.catalog.components != ("packed_kv",):
                raise RuntimeError(
                    "vLLM physical catalog must contain only packed_kv components"
                )
            transfer = getattr(vllm_config, "kv_transfer_config", None)
            extra = getattr(transfer, "kv_connector_extra_config", None) or {}
            if extra.get("namespace") != self.catalog.namespace:
                raise RuntimeError(
                    "vLLM connector namespace does not match the physical catalog"
                )
            if not os.environ.get("PYTHONHASHSEED"):
                raise RuntimeError(
                    "vLLM physical storage requires PYTHONHASHSEED for stable "
                    "cross-process block identities"
                )

    @staticmethod
    def _one_group(block_ids: tuple[list[int], ...]) -> tuple[int, ...]:
        if len(block_ids) != 1 or not block_ids[0]:
            raise RuntimeError("NTA vLLM connector requires one non-empty block row")
        row = tuple(int(block) for block in block_ids[0])
        if any(block < 0 for block in row) or len(set(row)) != len(row):
            raise RuntimeError("NTA vLLM connector received invalid block ownership")
        return row

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        request_id = str(request.request_id)
        self._external_matches.pop(request_id, None)
        if self._host_scheduler is not None:
            return self._host_scheduler.matched_tokens(request, num_computed_tokens), False
        if self.catalog is None:
            return 0, False
        if num_computed_tokens < 0 or num_computed_tokens % self._block_size:
            return 0, False
        hashes = tuple(getattr(request, "block_hashes", ()))
        if any(not isinstance(value, bytes) or not value for value in hashes):
            raise RuntimeError("vLLM physical block hashes must be non-empty bytes")
        first_block = num_computed_tokens // self._block_size
        matched: list[str] = []
        for value in hashes[first_block:]:
            key = vllm_storage_key(value)
            if not self.catalog.has_storage_key(key):
                break
            matched.append(key)
        # External ownership is block-granular.  Returning a partial final
        # block would let vLLM write the current token and then let NTA's
        # whole-page DMA overwrite it in the same layer.  Keep one token for
        # local computation and admit only complete external blocks.
        remaining = max(0, int(request.num_tokens) - num_computed_tokens - 1)
        matched_blocks = min(len(matched), remaining // self._block_size)
        if matched_blocks <= 0:
            return 0, False
        matched = matched[:matched_blocks]
        token_count = matched_blocks * self._block_size
        self._external_matches[request_id] = _ExternalMatch(
            first_block, tuple(matched), token_count
        )
        # NTA loads synchronously inside the typed attention launch. Returning
        # False keeps the request in this scheduler step; no host-side wait or
        # completion-driven request suspension is introduced.
        return token_count, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        values = blocks.get_block_ids()
        assert values is not None
        request_id = str(request.request_id)
        row = self._one_group(values)
        self._block_ids[request_id] = row
        if self._host_scheduler is not None:
            self._host_scheduler.update_after_alloc(
                request, blocks, int(num_external_tokens)
            )
            return
        match = self._external_matches.pop(request_id, None)
        if num_external_tokens == 0:
            return
        if match is None or int(num_external_tokens) != match.token_count:
            raise RuntimeError("vLLM external allocation has no exact catalog match")
        end = match.first_block + len(match.storage_keys)
        if end > len(row):
            raise RuntimeError("vLLM external allocation omits destination blocks")
        self._storage_keys_by_block[request_id] = {
            block: key
            for block, key in zip(
                row[match.first_block : end], match.storage_keys, strict=True
            )
        }

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> NtaVllmConnectorMetadata:
        host_transfer = (
            None
            if self._host_scheduler is None
            else self._host_scheduler.build_metadata(scheduler_output)
        )
        for request in scheduler_output.scheduled_new_reqs:
            self._block_ids[str(request.req_id)] = self._one_group(request.block_ids)

        cached = scheduler_output.scheduled_cached_reqs
        for request_id, new_blocks in zip(
            cached.req_ids, cached.new_block_ids, strict=True
        ):
            request_id = str(request_id)
            if new_blocks is None:
                continue
            appended = self._one_group(new_blocks)
            if request_id in cached.resumed_req_ids:
                self._block_ids[request_id] = appended
                continue
            previous = self._block_ids.get(request_id, ())
            # update_state_after_alloc may already have published the complete
            # row for this step.  Append only when the delta is not its suffix.
            if len(previous) < len(appended) or previous[-len(appended) :] != appended:
                self._block_ids[request_id] = previous + appended

        scheduled = scheduler_output.num_scheduled_tokens
        ordered_ids = sorted(
            (str(request_id) for request_id in scheduled),
            key=lambda request_id: (
                int(scheduled[request_id]) != 1,
                int(scheduled[request_id]),
            ),
        )
        requests = []
        for request_id in ordered_ids:
            row = self._block_ids.get(request_id)
            if row is None:
                raise RuntimeError(
                    f"NTA vLLM connector has no block ownership for {request_id!r}"
                )
            requests.append(
                NtaVllmRequestMetadata(
                    request_id,
                    row,
                    tuple(
                        self._storage_keys_by_block.get(request_id, {}).get(block)
                        for block in row
                    ),
                    int(scheduled[request_id]),
                )
            )
        finished = set(str(value) for value in scheduler_output.finished_req_ids)
        preempted = getattr(scheduler_output, "preempted_req_ids", None)
        if preempted:
            finished.update(str(value) for value in preempted)
        finished.update(self._pending_finished)
        metadata = NtaVllmConnectorMetadata(
            tuple(requests), tuple(sorted(finished)), host_transfer
        )
        # A scheduler step may retire requests without launching a worker
        # forward.  Connector metadata from such a step is never observed by
        # the runtime controller, so retain the notifications until the next
        # non-empty batch.  retire_request() is idempotent, making replay safe.
        if requests:
            self._pending_finished.difference_update(finished)
            for request in requests:
                self._storage_keys_by_block.pop(request.request_id, None)
        return metadata

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        del forward_context, kwargs
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, NtaVllmConnectorMetadata):
            raise RuntimeError("NTA vLLM worker received foreign connector metadata")
        if self._host_worker is not None:
            if metadata.host_transfer is None:
                raise RuntimeError("vLLM host worker received no host-transfer metadata")
            self._host_worker.bind(metadata.host_transfer)
        elif metadata.host_transfer is not None:
            raise RuntimeError("non-host vLLM worker received host-transfer metadata")
        if not metadata.requests:
            if self._host_worker is not None and self._host_worker.transfer_pairs:
                raise RuntimeError("vLLM host loads require a numerical forward")
            return
        state = current_vllm_v1_forward_state()
        if state is None or state.reference_warmup:
            raise RuntimeError("NTA connector ran outside a real vLLM forward")
        if metadata.requests:
            if state.batch is None:
                raise RuntimeError(
                    "NTA connector metadata was not bound before planning"
                )
            expected_ids = tuple(
                stable_request_id(value) for value in metadata.request_ids
            )
            if state.batch.request_ids != expected_ids:
                raise RuntimeError("NTA connector and attention request order disagree")
            exact = state.batch.exact_demand
            if exact is None or exact.request_unit_ids != metadata.block_tables:
                raise RuntimeError(
                    "NTA connector and attention block ownership disagree"
                )
            state.storage_key_tables = metadata.storage_key_tables
            if self._host_worker is not None:
                pairs = tuple(
                    sorted(self._host_worker.transfer_pairs, key=lambda pair: pair[1])
                )
                resources = dict(self._host_worker.resources)
                state.host_transfer_pairs = pairs
                state.host_resources = resources
                if pairs:
                    event = self._host_worker.preload_exact()
                    if event is None:
                        raise RuntimeError(
                            "vLLM host preload omitted an admitted transfer"
                        )
                    state.host_preload_event = event
                    state.host_preload_blocks = len(pairs) * len(resources)
                    state.host_preload_bytes = len(pairs) * sum(
                        resource.row_bytes for resource in resources.values()
                    )
        state.connector_validated = True

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        if self._host_worker is not None:
            self._host_worker.register_kv_caches(kv_caches)

    def bind_gpu_block_pool(self, gpu_block_pool: Any) -> None:
        if self._host_scheduler is not None:
            self._host_scheduler.bind_gpu_block_pool(gpu_block_pool)

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        if self._host_worker is None:
            return
        if not isinstance(kv_connector_metadata, NtaVllmConnectorMetadata):
            raise RuntimeError("vLLM host preemption received foreign metadata")
        if kv_connector_metadata.host_transfer is None:
            raise RuntimeError("vLLM host preemption has no transfer metadata")
        self._host_worker.handle_preemptions(kv_connector_metadata.host_transfer)

    def wait_for_layer_load(self, layer_name: str) -> None:
        del layer_name

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        del layer_name, kv_layer, attn_metadata, kwargs

    def wait_for_save(self) -> None:
        return

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        if self._host_worker is not None:
            self._host_worker.finish(finished_req_ids)
        return None, None

    def build_connector_worker_meta(self) -> Any | None:
        return (
            None
            if self._host_worker is None
            else self._host_worker.build_worker_metadata()
        )

    def update_connector_output(self, connector_output: Any) -> None:
        if self._host_scheduler is not None:
            self._host_scheduler.update_output(connector_output)

    def has_pending_push_work(self) -> bool:
        return bool(
            self._host_scheduler is not None
            and self._host_scheduler.has_pending_stores()
        )

    def take_events(self) -> Any:
        return () if self._host_scheduler is None else self._host_scheduler.take_events()

    def reset_cache(self) -> bool | None:
        return None if self._host_scheduler is None else self._host_scheduler.reset()

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()
        if self._host_worker is not None:
            self._host_worker.clear()

    def shutdown(self) -> None:
        if self._host_worker is not None:
            self._host_worker.shutdown()

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        request_id = str(request.request_id)
        self._block_ids.pop(request_id, None)
        self._storage_keys_by_block.pop(request_id, None)
        self._external_matches.pop(request_id, None)
        self._pending_finished.add(request_id)
        if self._host_scheduler is not None:
            return self._host_scheduler.request_finished(request, block_ids)
        return False, None

    def request_finished_all_groups(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        request_id = str(request.request_id)
        self._block_ids.pop(request_id, None)
        self._storage_keys_by_block.pop(request_id, None)
        self._external_matches.pop(request_id, None)
        self._pending_finished.add(request_id)
        if self._host_scheduler is not None:
            return self._host_scheduler.request_finished_all_groups(
                request, block_ids
            )
        return False, None
