"""Official vLLM 0.26 scheduler/worker lifecycle seam for NTA.

The connector carries exact request IDs and block ownership to the worker.  It
does not copy KV through vLLM's layer hooks: NTA's typed attention consumer owns
availability and transport.  Clearing connector metadata after validation
keeps those generic per-layer hooks out of the steady-state path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import importlib.metadata
import os
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
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
        if self.storage_keys and len(self.storage_keys) != len(self.block_ids):
            raise ValueError("NTA vLLM storage keys must align with block IDs")
        if any(key is not None and not key for key in self.storage_keys):
            raise ValueError("NTA vLLM storage keys must be non-empty")


@dataclass(frozen=True)
class NtaVllmConnectorMetadata(KVConnectorMetadata):
    requests: tuple[NtaVllmRequestMetadata, ...]
    finished_request_ids: tuple[str, ...]
    external_lease: "NtaVllmExternalLease | None" = None
    host_transfer: Any | None = None

    def __post_init__(self) -> None:
        request_ids = self.request_ids
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("NTA vLLM metadata request IDs must be unique")
        external_entries = NtaVllmExternalLease.entries_for(self.requests)
        if external_entries:
            if self.external_lease is None:
                raise ValueError("external vLLM demand requires a forward lease")
            if self.external_lease.entries != external_entries:
                raise ValueError("vLLM forward lease does not match exact metadata")
        elif self.external_lease is not None:
            raise ValueError("resident vLLM metadata cannot carry an external lease")

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(request.request_id for request in self.requests)

    @property
    def block_tables(self) -> tuple[tuple[int, ...], ...]:
        return tuple(request.block_ids for request in self.requests)

    @property
    def storage_key_tables(self) -> tuple[tuple[str | None, ...], ...]:
        return tuple(request.storage_keys for request in self.requests)

    def aligned_to(self, request_ids: Sequence[str]) -> "NtaVllmConnectorMetadata":
        """Return exact metadata in the authoritative numerical row order."""
        if not isinstance(request_ids, Sequence) or isinstance(
            request_ids, (str, bytes)
        ):
            raise RuntimeError("vLLM InputBatch.req_ids must be a request sequence")
        actual_ids = tuple(str(request_id) for request_id in request_ids)
        if any(not request_id for request_id in actual_ids):
            raise RuntimeError("vLLM InputBatch.req_ids must be non-empty")
        if len(set(actual_ids)) != len(actual_ids):
            raise RuntimeError("vLLM InputBatch.req_ids contains duplicate rows")
        by_request = {request.request_id: request for request in self.requests}
        if len(by_request) != len(self.requests) or set(by_request) != set(actual_ids):
            raise RuntimeError(
                "vLLM connector metadata and InputBatch.req_ids disagree"
            )
        aligned_requests = tuple(by_request[request_id] for request_id in actual_ids)
        if aligned_requests == self.requests:
            return self
        aligned_entries = NtaVllmExternalLease.entries_for(aligned_requests)
        aligned_lease = (
            None
            if self.external_lease is None
            else NtaVllmExternalLease(self.external_lease.lease_id, aligned_entries)
        )
        return NtaVllmConnectorMetadata(
            requests=aligned_requests,
            finished_request_ids=self.finished_request_ids,
            external_lease=aligned_lease,
            host_transfer=self.host_transfer,
        )


@dataclass(frozen=True)
class _ExternalMatch:
    first_block: int
    storage_keys: tuple[str, ...]
    token_count: int


@dataclass(frozen=True)
class NtaVllmExternalLease:
    """Exact external identities retained until worker execution commits."""

    lease_id: int
    entries: tuple[tuple[str, tuple[tuple[int, str], ...]], ...]

    def __post_init__(self) -> None:
        if self.lease_id <= 0 or not self.entries:
            raise ValueError("NTA vLLM external lease must be positive and non-empty")
        request_ids = tuple(request_id for request_id, _ in self.entries)
        if any(not request_id for request_id in request_ids):
            raise ValueError("NTA vLLM external lease request IDs must be non-empty")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("NTA vLLM external lease request IDs must be unique")
        for _request_id, bindings in self.entries:
            if not bindings:
                raise ValueError("NTA vLLM external lease rows must be non-empty")
            blocks = tuple(block for block, _key in bindings)
            if any(block < 0 for block in blocks) or len(set(blocks)) != len(blocks):
                raise ValueError("NTA vLLM external lease block IDs are invalid")
            if any(not key for _block, key in bindings):
                raise ValueError("NTA vLLM external lease keys must be non-empty")

    @staticmethod
    def entries_for(
        requests: tuple[NtaVllmRequestMetadata, ...],
    ) -> tuple[tuple[str, tuple[tuple[int, str], ...]], ...]:
        entries = []
        for request in requests:
            if not request.storage_keys:
                continue
            bindings = tuple(
                (block, key)
                for block, key in zip(
                    request.block_ids, request.storage_keys, strict=True
                )
                if key is not None
            )
            if bindings:
                entries.append((request.request_id, bindings))
        return tuple(entries)


@dataclass(frozen=True)
class NtaVllmForwardAck(KVConnectorWorkerMetadata):
    """Worker-to-scheduler proof that every leased row was numerically used."""

    lease: NtaVllmExternalLease
    worker_count: int = 1

    def __post_init__(self) -> None:
        if self.worker_count <= 0:
            raise ValueError("vLLM forward ACK worker count must be positive")

    def aggregate(self, other: KVConnectorWorkerMetadata) -> "NtaVllmForwardAck":
        if not isinstance(other, NtaVllmForwardAck) or other.lease != self.lease:
            raise RuntimeError(
                "vLLM workers disagree on the committed external forward lease"
            )
        return NtaVllmForwardAck(self.lease, self.worker_count + other.worker_count)


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
        self._external_lease_by_block: dict[str, dict[int, int]] = {}
        self._pending_finished: set[str] = set()
        self._next_external_lease_id = 1
        parallel = getattr(vllm_config, "parallel_config", None)
        self._expected_worker_acks = int(getattr(parallel, "world_size", 1) or 1)
        if self._expected_worker_acks <= 0:
            raise RuntimeError("vLLM connector worker world size must be positive")
        self._forward_active = False
        self._active_external_lease: NtaVllmExternalLease | None = None
        self._committed_external_lease: NtaVllmExternalLease | None = None
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
                self._host_scheduler = VllmHostScheduler(vllm_config, kv_cache_config)
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
            return self._host_scheduler.matched_tokens(
                request, num_computed_tokens
            ), False
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
        self._external_lease_by_block.pop(request_id, None)

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
        requests = []
        for raw_request_id in scheduled:
            request_id = str(raw_request_id)
            row = self._block_ids.get(request_id)
            if row is None:
                raise RuntimeError(
                    f"NTA vLLM connector has no block ownership for {request_id!r}"
                )
            storage = self._storage_keys_by_block.get(request_id)
            storage_keys = ()
            if storage:
                candidate_keys = tuple(storage.get(block) for block in row)
                if any(key is not None for key in candidate_keys):
                    storage_keys = candidate_keys
            requests.append(
                NtaVllmRequestMetadata(
                    request_id,
                    row,
                    storage_keys,
                    int(scheduled[raw_request_id]),
                )
            )
        finished = set(str(value) for value in scheduler_output.finished_req_ids)
        preempted = {
            str(value)
            for value in (getattr(scheduler_output, "preempted_req_ids", ()) or ())
        }
        finished.update(preempted)
        # A preemption ends the current runtime generation even when vLLM
        # later resumes the same logical request ID. Its old physical row and
        # content-key bindings cannot be carried across that reuse boundary.
        for request_id in preempted:
            self._block_ids.pop(request_id, None)
            self._storage_keys_by_block.pop(request_id, None)
            self._external_lease_by_block.pop(request_id, None)
            self._external_matches.pop(request_id, None)
        self._pending_finished.update(finished)
        finished = set(self._pending_finished)
        requests_tuple = tuple(requests)
        external_entries = NtaVllmExternalLease.entries_for(requests_tuple)
        external_lease = None
        if external_entries:
            external_lease = NtaVllmExternalLease(
                self._next_external_lease_id, external_entries
            )
            self._next_external_lease_id += 1
            for request_id, bindings in external_entries:
                versions = self._external_lease_by_block.setdefault(request_id, {})
                for block, _key in bindings:
                    versions[block] = external_lease.lease_id
        metadata = NtaVllmConnectorMetadata(
            requests=requests_tuple,
            finished_request_ids=tuple(sorted(finished)),
            external_lease=external_lease,
            host_transfer=host_transfer,
        )
        # A scheduler step may retire requests without launching a worker
        # forward.  Connector metadata from such a step is never observed by
        # the runtime controller, so retain the notifications until the next
        # non-empty batch.  retire_request() is idempotent, making replay safe.
        if requests:
            self._pending_finished.difference_update(finished)
        return metadata

    def begin_forward(self, metadata: NtaVllmConnectorMetadata) -> None:
        """Lease one worker metadata object without consuming scheduler state."""
        if self._forward_active:
            raise RuntimeError("vLLM connector forward lifetimes overlap")
        if self._committed_external_lease is not None:
            raise RuntimeError("vLLM external ACK was not collected before reuse")
        self._forward_active = True
        self._active_external_lease = metadata.external_lease

    def validate_forward_commit(self) -> None:
        if not self._forward_active:
            raise RuntimeError("vLLM connector has no active forward to commit")

    def commit_forward(self) -> None:
        """Publish an ACK only after the complete numerical forward succeeds."""
        self.validate_forward_commit()
        self._committed_external_lease = self._active_external_lease
        self._active_external_lease = None
        self._forward_active = False

    def abort_forward(self) -> None:
        """Release worker metadata without leaving a reportable ACK."""
        self._active_external_lease = None
        self._committed_external_lease = None
        self._forward_active = False
        if self.has_connector_metadata():
            super().clear_connector_metadata()
        if self._host_worker is not None:
            self._host_worker.abort()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        del forward_context, kwargs
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, NtaVllmConnectorMetadata):
            self.abort_forward()
            raise RuntimeError("NTA vLLM worker received foreign connector metadata")
        if self._host_worker is not None:
            if metadata.host_transfer is None:
                self.abort_forward()
                raise RuntimeError(
                    "vLLM host worker received no host-transfer metadata"
                )
            try:
                self._host_worker.bind(metadata.host_transfer)
            except BaseException:
                self.abort_forward()
                raise
        elif metadata.host_transfer is not None:
            self.abort_forward()
            raise RuntimeError("non-host vLLM worker received host-transfer metadata")
        if not metadata.requests:
            if self._host_worker is not None and self._host_worker.transfer_pairs:
                self.abort_forward()
                raise RuntimeError("vLLM host loads require a numerical forward")
            return
        state = current_vllm_v1_forward_state()
        if state is None or state.reference_warmup:
            self.abort_forward()
            raise RuntimeError("NTA connector ran outside a real vLLM forward")
        state.connector_owner = self
        try:
            input_rows = getattr(state.input_batch, "req_ids", None)
            if not isinstance(input_rows, Sequence) or isinstance(
                input_rows, (str, bytes)
            ):
                raise RuntimeError(
                    "NTA connector has no InputBatch.req_ids row contract"
                )
            scheduled_ids = set(metadata.request_ids)
            input_request_ids = tuple(
                str(request_id)
                for request_id in input_rows
                if str(request_id) in scheduled_ids
            )
            metadata = metadata.aligned_to(input_request_ids)
            planned = state.connector_metadata
            if planned is not None:
                if not isinstance(planned, NtaVllmConnectorMetadata) or (
                    planned.requests,
                    planned.finished_request_ids,
                    planned.external_lease,
                ) != (
                    metadata.requests,
                    metadata.finished_request_ids,
                    metadata.external_lease,
                ):
                    raise RuntimeError(
                        "NTA connector metadata changed after attention planning"
                    )
            state.connector_metadata = metadata
            self.begin_forward(metadata)
        except BaseException:
            self.abort_forward()
            raise
        try:
            if state.batch is None:
                raise RuntimeError(
                    "NTA connector metadata was not bound before planning"
                )
            actual_ids = tuple(str(value) for value in input_request_ids)
            if metadata.request_ids != actual_ids:
                raise RuntimeError(
                    "NTA connector metadata and InputBatch.req_ids disagree "
                    "item-by-item"
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
                    if state.batch is None or len(metadata.request_ids) != len(
                        state.batch.bindings
                    ):
                        raise RuntimeError(
                            "vLLM Host tenant ownership is not aligned to the batch"
                        )
                    request_tenants = {
                        request_id: binding.tenant_id
                        for request_id, binding in zip(
                            metadata.request_ids,
                            state.batch.bindings,
                            strict=True,
                        )
                    }
                    acquisition = self._host_worker.submit_exact_layers(request_tenants)
                    if acquisition is None:
                        if not state.tenant_isolation_enabled:
                            raise RuntimeError(
                                "vLLM Host scheduler omitted an admitted acquisition"
                            )
                        state.record_evidence("host_acquisition_credit_rejections")
                        state.connector_validated = True
                        return
                    state.host_acquisition = acquisition
                    state.record_evidence("host_acquisition_batches")
                    state.record_evidence(
                        "host_acquisition_jobs", acquisition.layer_count
                    )
                    state.record_evidence(
                        "host_acquisition_waves", acquisition.wave_count
                    )
                    state.record_profile_ns(
                        "host_acquisition_submission_cpu_ns",
                        acquisition.submission_cpu_ns,
                    )
                    state.record_evidence(
                        "host_transfer_blocks", acquisition.transfer_blocks
                    )
                    state.record_evidence(
                        "host_transfer_bytes", acquisition.transfer_bytes
                    )
                    state.record_evidence(
                        "host_transfer_runs", acquisition.transfer_runs
                    )
                    state.record_evidence(
                        "host_copy_operations", acquisition.copy_operations
                    )
                    state.record_evidence(
                        "host_copy_submissions", acquisition.copy_submissions
                    )
                    if acquisition.tenant_accounted:
                        state.record_evidence(
                            "host_acquisition_tenant_accounted_batches"
                        )
                        state.record_evidence(
                            "host_acquisition_tenant_charge_bytes",
                            acquisition.tenant_charge_bytes,
                        )
            state.connector_validated = True
        except BaseException:
            self.abort_forward()
            raise

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
        host_metadata = (
            None
            if self._host_worker is None
            else self._host_worker.build_worker_metadata()
        )
        lease = self._committed_external_lease
        if lease is not None and host_metadata is not None:
            raise RuntimeError("vLLM forward produced two worker metadata owners")
        if lease is None:
            return host_metadata
        self._committed_external_lease = None
        return NtaVllmForwardAck(lease)

    def update_connector_output(self, connector_output: Any) -> None:
        worker_metadata = getattr(connector_output, "kv_connector_worker_meta", None)
        if isinstance(worker_metadata, NtaVllmForwardAck):
            if self._host_scheduler is not None:
                raise RuntimeError("host vLLM connector received an NVMe lease ACK")
            if worker_metadata.worker_count != self._expected_worker_acks:
                raise RuntimeError(
                    "vLLM external lease ACK did not cover every worker: "
                    f"expected={self._expected_worker_acks}, "
                    f"received={worker_metadata.worker_count}"
                )
            actions: list[tuple[str, int, bool]] = []
            for request_id, bindings in worker_metadata.lease.entries:
                current = self._storage_keys_by_block.get(request_id)
                versions = self._external_lease_by_block.get(request_id)
                if current is None:
                    # The request was retired before its delayed ACK reached the
                    # scheduler; its identity has already been discarded.
                    continue
                for block, key in bindings:
                    if (
                        versions is None
                        or versions.get(block) != worker_metadata.lease.lease_id
                    ):
                        # A retry or request-generation reuse superseded this
                        # ACK before it reached the scheduler.
                        continue
                    present = current.get(block)
                    if present is None:
                        # ACK delivery is idempotent across worker aggregation.
                        actions.append((request_id, block, False))
                        continue
                    if present != key:
                        raise RuntimeError(
                            "vLLM external ACK conflicts with current storage identity"
                        )
                    actions.append((request_id, block, True))
            touched_requests: set[str] = set()
            for request_id, block, remove_storage_key in actions:
                touched_requests.add(request_id)
                current = self._storage_keys_by_block.get(request_id)
                versions = self._external_lease_by_block.get(request_id)
                if remove_storage_key and current is not None:
                    current.pop(block, None)
                if versions is not None:
                    versions.pop(block, None)
            for request_id in touched_requests:
                current = self._storage_keys_by_block.get(request_id)
                versions = self._external_lease_by_block.get(request_id)
                if not current:
                    self._storage_keys_by_block.pop(request_id, None)
                if not versions:
                    self._external_lease_by_block.pop(request_id, None)
        elif worker_metadata is not None and self._host_scheduler is None:
            raise RuntimeError("NTA vLLM scheduler received foreign worker metadata")
        if self._host_scheduler is not None:
            self._host_scheduler.update_output(connector_output)

    def has_pending_push_work(self) -> bool:
        return bool(
            self._host_scheduler is not None
            and self._host_scheduler.has_pending_stores()
        )

    def take_events(self) -> Any:
        return (
            () if self._host_scheduler is None else self._host_scheduler.take_events()
        )

    def reset_cache(self) -> bool | None:
        return None if self._host_scheduler is None else self._host_scheduler.reset()

    def clear_connector_metadata(self) -> None:
        if self._forward_active:
            raise RuntimeError("vLLM connector metadata cleared before forward end")
        super().clear_connector_metadata()
        if self._host_worker is not None:
            self._host_worker.clear()

    def shutdown(self) -> None:
        failure: BaseException | None = None
        try:
            self.abort_forward()
        except BaseException as error:
            failure = error
        self._committed_external_lease = None
        if self._host_worker is not None:
            try:
                self._host_worker.shutdown()
            except BaseException as error:
                if failure is None:
                    raise
                failure.add_note(f"vLLM host shutdown also failed: {error!r}")
        if failure is not None:
            raise failure

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        request_id = str(request.request_id)
        self._block_ids.pop(request_id, None)
        self._storage_keys_by_block.pop(request_id, None)
        self._external_lease_by_block.pop(request_id, None)
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
        self._external_lease_by_block.pop(request_id, None)
        self._external_matches.pop(request_id, None)
        self._pending_finished.add(request_id)
        if self._host_scheduler is not None:
            return self._host_scheduler.request_finished_all_groups(request, block_ids)
        return False, None
