"""SGLang 0.5.14 adapter for compiler-instrumented FlashInfer attention."""

from __future__ import annotations

import atexit
from collections import Counter
from dataclasses import dataclass
import json
import logging
import os
import pathlib
import time
from typing import Any

import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.memory_pool import KVWriteLoc

from nta_runtime.flashinfer import (
    FlashInferLayerEpoch,
    attention_jit_args,
)
from nta_runtime.flashinfer_schedule import (
    Schedule,
    decode_schedule,
    paged_prefill_schedule,
    require_supported_version,
)
from nta_runtime.execution_policy import (
    HostCostModel,
    HostExecutionPlan,
    plan_host_execution,
)
from nta_runtime.opportunity import OperatorArrival, TileArrival, append_json_line
from nta_runtime.requests import RequestBinding, RequestSlotTracker
from nta_runtime.engines.sglang_hicache import PendingHostLoad, SglangHiCacheBridge
from nta_runtime.runtime import (
    AcquireRequirement,
    DeviceWorkPlan,
    IndexedHostObject,
    JitPhaseProgram,
    RequestRange,
    Runtime,
    RuntimeConfig,
    WorkItem,
)


_OBJECT_ID_BASE = 0x4E54410000000000
_MAX_ABI_BYTES = (1 << 32) - 1
logger = logging.getLogger(__name__)
_PagePair = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class _PrefetchedLayer:
    key_slot: int
    key_object_id: int
    value_slot: int
    value_object_id: int
    version: int
    key_bytes: int
    value_bytes: int
    ready_event: torch.cuda.Event


@dataclass(frozen=True)
class _ActiveBatch:
    bindings: tuple[RequestBinding, ...]
    schedules: dict[int, Schedule]
    pending_host_load: PendingHostLoad | None
    page_pairs: dict[int, tuple[_PagePair, ...]]
    index_maps: dict[_PagePair, tuple[torch.Tensor, torch.Tensor]]
    prefetched_layers: dict[int, _PrefetchedLayer]
    prefetch_tensors: tuple[torch.Tensor, ...]


@dataclass
class _PlanAllocation:
    plan: DeviceWorkPlan
    work_capacity: int
    signature: tuple[Any, ...] | None = None
    object_count: int = 0
    index_tensors: tuple[torch.Tensor, ...] = ()
    host_execution: HostExecutionPlan | None = None
    object_version: int = 0


def _positive_environment(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_environment(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value


def _gain_environment(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if value < 1.0:
        raise ValueError(f"{name} must be at least one")
    return value


def _dtype_tag(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.").replace("_", "")


def _plan_cache_signature(
    request_indices: tuple[int, ...],
    kv_tile_indices: tuple[int, ...],
    page_pairs: tuple[_PagePair, ...],
    request_slots: tuple[int, ...],
    key_pointer: int,
    value_pointer: int,
    key_bytes: int,
    value_bytes: int,
    prefetched_bytes: tuple[int, int] | None,
) -> tuple[Any, ...]:
    """Return an exact identity for every device-side plan input."""
    return (
        request_indices,
        kv_tile_indices,
        page_pairs,
        request_slots,
        key_pointer,
        value_pointer,
        key_bytes,
        value_bytes,
        prefetched_bytes,
    )


class NtaFlashInferAttnBackend(FlashInferAttnBackend):
    """FA2 backend carrying request semantics into every attention CTA."""

    def __init__(
        self,
        model_runner: Any,
        skip_prefill: bool = False,
        kv_indptr_buf: torch.Tensor | None = None,
        kv_last_page_len_buf: torch.Tensor | None = None,
        init_new_workspace: bool = False,
    ) -> None:
        require_supported_version()
        if model_runner.server_args.speculative_algorithm is not None:
            raise ValueError(
                "NTA's SGLang adapter does not support speculative decoding"
            )
        if model_runner.kv_cache_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "NTA's SGLang adapter currently supports float16 or bfloat16 KV"
            )
        super().__init__(
            model_runner,
            skip_prefill=skip_prefill,
            kv_indptr_buf=kv_indptr_buf,
            kv_last_page_len_buf=kv_last_page_len_buf,
            init_new_workspace=init_new_workspace,
        )
        if self.prefill_backend != "fa2" or self.decode_backend != "fa2":
            raise ValueError("NTA requires FlashInfer's FA2 attention kernels")

        self._stock_decode_wrappers = self.decode_wrappers
        self._stock_prefill_wrappers_paged = self.prefill_wrappers_paged
        self._stock_prefill_wrappers_verify = self.prefill_wrappers_verify
        self._hicache_enabled = bool(model_runner.server_args.enable_hierarchical_cache)
        self._decode_jit_args: list[Any] | None = None
        self._prefill_jit_args: list[Any] | None = None
        if self._hicache_enabled:
            self._install_instrumented_wrappers(model_runner, skip_prefill)
        else:
            self._nta_decode_wrappers = self._stock_decode_wrappers
            self._nta_prefill_wrappers_paged = self._stock_prefill_wrappers_paged
            self._nta_prefill_wrappers_verify = self._stock_prefill_wrappers_verify
            self._wrapper_modules = {}

        request_capacity = int(model_runner.req_to_token_pool.req_to_token.shape[0])
        default_tickets = max(4096, request_capacity * 8)
        self._work_ticket_capacity = _positive_environment(
            "NTA_SGLANG_MAX_WORK_TICKETS", default_tickets
        )
        self._object_capacity = 2 * self._work_ticket_capacity
        self._runtime = Runtime(
            RuntimeConfig(
                request_capacity=request_capacity,
                object_capacity=self._object_capacity,
                intent_capacity=self._object_capacity,
                work_ticket_capacity=self._work_ticket_capacity,
                max_dependencies_per_work_ticket=2,
                device_ordinal=torch.cuda.current_device(),
                enable_cta_nvme_try_issue=False,
            )
        )
        self._runtime.set_tenant_budget(0, (1 << 64) - 1)
        self._request_slots = RequestSlotTracker(self._runtime, request_capacity)
        self._hicache = SglangHiCacheBridge(self.token_to_kv_pool)
        self._prefetch_stream = torch.cuda.Stream(priority=0)
        self._host_cost_model = HostCostModel(
            bandwidth_bytes_per_second=_positive_environment(
                "NTA_SGLANG_HOST_BANDWIDTH_BPS", 30_000_000_000
            ),
            round_overhead_ns=_nonnegative_environment(
                "NTA_SGLANG_ROUND_OVERHEAD_NS", 15_000
            ),
            tile_compute_ns=_positive_environment(
                "NTA_SGLANG_TILE_COMPUTE_NS", 3_000
            ),
            max_rounds=_positive_environment("NTA_SGLANG_MAX_HOST_ROUNDS", 4),
            minimum_predicted_gain=_gain_environment(
                "NTA_SGLANG_MIN_PREDICTED_GAIN", 1.03
            ),
        )
        self._pipeline_host = (
            self._hicache_enabled
            and os.environ.get("NTA_SGLANG_PIPELINE_HOST", "1") != "0"
        )
        self._allow_fallback = os.environ.get("NTA_SGLANG_ALLOW_FALLBACK") == "1"
        self._force_incremental = (
            os.environ.get("NTA_SGLANG_FORCE_INCREMENTAL") == "1"
        )
        self._prefetch_ready_events: tuple[tuple[torch.cuda.Event, ...], ...] = ()
        self._bulk_events: tuple[torch.cuda.Event, ...] = ()
        layer_count = getattr(model_runner.model_config, "num_hidden_layers", None)
        if layer_count is None:
            layer_count = getattr(
                model_runner.model_config.hf_config, "num_hidden_layers"
            )
        self._model_layer_count = int(layer_count)
        self._model_start_layer = int(
            getattr(self.token_to_kv_pool, "start_layer", 0)
        )
        self._cuda_graph_mode = False
        self._active_batch: _ActiveBatch | None = None
        self._plans: dict[tuple[int, int], _PlanAllocation] = {}
        self._phase_programs: dict[str, JitPhaseProgram] = {}
        self._stats = {
            "schema": 1,
            "engine": "sglang",
            "backend": "nta_flashinfer",
            "revision": os.environ.get("NTA_REVISION", "unknown"),
            "pid": os.getpid(),
            "batches": 0,
            "decode_launches": 0,
            "prefill_launches": 0,
            "cta_work_items": 0,
            "plan_uploads": 0,
            "request_rebindings": 0,
            "request_cancellations": 0,
            "external_launches": 0,
            "resident_stock_batches": 0,
            "hicache_claimed_batches": 0,
            "hicache_fallback_batches": 0,
            "indexed_host_objects": 0,
            "indexed_host_bytes": 0,
            "prefetched_layers": 0,
            "prefetched_host_bytes": 0,
            "demand_host_layers": 0,
            "incremental_host_layers": 0,
            "bulk_host_batches": 0,
            "host_progress_rounds": 0,
            "predicted_atomic_ns": 0,
            "predicted_incremental_ns": 0,
            "progress_snapshots": 0,
            "request_work_completed": 0,
            "request_work_failed": 0,
            "request_compute_completed_ns": 0,
            "graph_captures": 0,
            "graph_replays": 0,
            "graph_external_batches": 0,
            "started_unix_ns": time.time_ns(),
        }
        self._profile_cpu = os.environ.get("NTA_SGLANG_PROFILE_CPU") == "1"
        trace_file = os.environ.get("NTA_OPPORTUNITY_TRACE_FILE")
        self._opportunity_trace = pathlib.Path(trace_file) if trace_file else None
        self._opportunity_revision = os.environ.get("NTA_REVISION", "")
        self._opportunity_model = os.environ.get(
            "NTA_OPPORTUNITY_MODEL",
            str(getattr(model_runner.model_config, "model_path", "unknown")),
        )
        self._opportunity_tier = os.environ.get(
            "NTA_OPPORTUNITY_TIER", "host_staged"
        )
        self._opportunity_batch = 0
        self._active_opportunity_batch = -1
        if self._opportunity_trace is not None:
            if not self._opportunity_revision:
                raise ValueError(
                    "NTA_REVISION is required when opportunity tracing is enabled"
                )
            if self._opportunity_tier != "host_staged":
                raise ValueError(
                    "the SGLang HiCache tracer only observes host_staged data"
                )
        if self._pipeline_host:
            self._hicache.set_prefetch_callback(self._prepare_host_pipeline)
        atexit.register(self._write_stats)

    def cancel_requests(self, request_id_prefix: str, *, all: bool = False) -> int:
        cancelled = self._request_slots.cancel_matching(request_id_prefix, all=all)
        self._stats["request_cancellations"] += cancelled
        return cancelled

    def _install_instrumented_wrappers(
        self, model_runner: Any, skip_prefill: bool
    ) -> None:
        q_dtype = model_runner.dtype
        kv_dtype = model_runner.kv_cache_dtype
        head_dim = int(model_runner.model_config.head_dim)
        signature = f"h{head_dim}_{_dtype_tag(q_dtype)}_{_dtype_tag(kv_dtype)}"
        decode_name = (
            f"nta_sglang_decode_default_v2_"
            f"{'tc' if self.decode_use_tensor_cores else 'cc'}_"
            f"{signature}"
        )
        prefill_name = f"nta_sglang_prefill_default_v2_{signature}"
        self._wrapper_modules: dict[int, str] = {}

        decode_args = attention_jit_args(
            decode_name,
            dtype_q=q_dtype,
            dtype_kv=kv_dtype,
            dtype_o=q_dtype,
            idtype=torch.int32,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
        )
        self._decode_jit_args = decode_args
        self._nta_decode_wrappers = []
        for _ in range(self.num_wrappers):
            wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer,
                "NHD",
                backend="fa2",
                use_tensor_cores=self.decode_use_tensor_cores,
                jit_args=decode_args,
            )
            self._nta_decode_wrappers.append(wrapper)
            self._wrapper_modules[id(wrapper)] = decode_name

        if skip_prefill:
            return
        prefill_args = attention_jit_args(
            prefill_name,
            dtype_q=q_dtype,
            dtype_kv=kv_dtype,
            dtype_o=q_dtype,
            idtype=torch.int32,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
        )
        self._prefill_jit_args = prefill_args

        def make_prefill() -> BatchPrefillWithPagedKVCacheWrapper:
            wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer,
                "NHD",
                backend="fa2",
                jit_args=prefill_args,
            )
            self._wrapper_modules[id(wrapper)] = prefill_name
            return wrapper

        self._nta_prefill_wrappers_paged = [
            make_prefill() for _ in range(self.num_wrappers)
        ]
        self._nta_prefill_wrappers_verify = [
            make_prefill() for _ in range(self.num_wrappers)
        ]
        self._select_wrappers(False)

    def _select_wrappers(self, external: bool) -> None:
        self.decode_wrappers = (
            self._nta_decode_wrappers if external else self._stock_decode_wrappers
        )
        if self.skip_prefill:
            return
        self.prefill_wrappers_paged = (
            self._nta_prefill_wrappers_paged
            if external
            else self._stock_prefill_wrappers_paged
        )
        self.prefill_wrappers_verify = (
            self._nta_prefill_wrappers_verify
            if external
            else self._stock_prefill_wrappers_verify
        )

    def _create_decode_wrappers(self, bs: int, num_tokens: int) -> list[Any]:
        if not self._hicache_enabled:
            return super()._create_decode_wrappers(bs, num_tokens)
        if self._decode_jit_args is None:
            raise RuntimeError("NTA decode JIT arguments are not initialized")
        wrappers = [
            BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer,
                "NHD",
                backend="fa2",
                use_cuda_graph=True,
                use_tensor_cores=self.decode_use_tensor_cores,
                paged_kv_indptr_buffer=self.kv_indptr[index][: num_tokens + 1],
                paged_kv_indices_buffer=self.cuda_graph_kv_indices[index],
                paged_kv_last_page_len_buffer=self.kv_last_page_len[:num_tokens],
                jit_args=self._decode_jit_args,
            )
            for index in range(self.num_wrappers)
        ]
        for wrapper in wrappers:
            self._wrapper_modules[id(wrapper)] = self._decode_jit_args[0]
        return wrappers

    def _create_prefill_wrappers(
        self, bs: int, use_custom_mask: bool = False
    ) -> list[Any]:
        if not self._hicache_enabled:
            return super()._create_prefill_wrappers(bs, use_custom_mask)
        if self._prefill_jit_args is None:
            raise RuntimeError("NTA prefill JIT arguments are not initialized")
        wrappers = []
        for index in range(self.num_wrappers):
            extra = (
                {
                    "custom_mask_buf": self.cuda_graph_custom_mask,
                    "mask_indptr_buf": self.cuda_graph_qk_indptr[index][: bs + 1],
                }
                if use_custom_mask
                else {}
            )
            wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer,
                "NHD",
                use_cuda_graph=True,
                backend="fa2",
                qo_indptr_buf=self.cuda_graph_qo_indptr[index][: bs + 1],
                paged_kv_indptr_buf=self.kv_indptr[index][: bs + 1],
                paged_kv_indices_buf=self.cuda_graph_kv_indices[index],
                paged_kv_last_page_len_buf=self.kv_last_page_len[:bs],
                jit_args=self._prefill_jit_args,
                **extra,
            )
            self._wrapper_modules[id(wrapper)] = self._prefill_jit_args[0]
            wrappers.append(wrapper)
        return wrappers

    def init_cuda_graph_state(self, *args: Any, **kwargs: Any) -> None:
        super().init_cuda_graph_state(*args, **kwargs)

    def _bind_forward_requests(
        self, forward_batch: Any, *, allow_capture_ids: bool
    ) -> tuple[RequestBinding, ...]:
        request_ids = getattr(forward_batch, "rids", None)
        batch_size = int(forward_batch.batch_size)
        if request_ids is None:
            if not allow_capture_ids:
                raise RuntimeError(
                    "SGLang CUDA replay omitted request IDs from its metadata view"
                )
            request_ids = [
                f"__nta_graph_capture_{index}" for index in range(batch_size)
            ]
        if len(request_ids) != batch_size:
            raise RuntimeError("SGLang request IDs do not match the padded graph batch")
        priorities = tuple(
            int(priority)
            for priority in getattr(
                forward_batch, "_nta_request_priorities", (0,) * batch_size
            )
        )
        bindings = self._request_slots.bind(
            request_ids, range(batch_size), priorities=priorities
        )
        self._stats["request_rebindings"] += self._request_slots.last_publish_count
        self._stats["request_policy_updates"] = self._stats.get(
            "request_policy_updates", 0
        ) + self._request_slots.last_policy_publish_count
        return bindings

    def init_forward_metadata_out_graph(
        self, forward_batch: Any, in_capture: bool = False
    ) -> None:
        self._cuda_graph_mode = True
        super().init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)
        if not self._hicache_enabled:
            self._active_batch = None
            return

        bindings = self._bind_forward_requests(
            forward_batch, allow_capture_ids=in_capture
        )
        counter = getattr(self.token_to_kv_pool, "layer_transfer_counter", None)
        consumer_index = -1 if counter is None else int(counter.consumer_index)
        pending = self._hicache.get(consumer_index)
        if pending is None:
            self._active_batch = _ActiveBatch(bindings, {}, None, {}, {}, {}, ())
        else:
            if not self._pipeline_host or not pending.prefetched_layers:
                raise RuntimeError(
                    "CUDA graph replay requires the stream-ordered HiCache pipeline"
                )
            self._active_batch = _ActiveBatch(
                bindings,
                {},
                pending,
                {},
                {},
                pending.prefetched_layers,
                pending.prefetch_tensors,
            )
            final_layer = max(pending.prefetched_layers)
            torch.cuda.current_stream().wait_event(
                pending.prefetched_layers[final_layer].ready_event
            )
            self._hicache.handoff_prefetch(pending, self._prefetch_stream)
            self._stats["graph_external_batches"] += 1
        if in_capture:
            self._stats["graph_captures"] += 1
        else:
            self._stats["graph_replays"] += 1
            self._stats["batches"] += 1

    def init_forward_metadata_in_graph(self, forward_batch: Any) -> None:
        super().init_forward_metadata_in_graph(forward_batch)

    def init_forward_metadata(self, forward_batch: Any) -> None:
        self._cuda_graph_mode = False
        counter = getattr(self.token_to_kv_pool, "layer_transfer_counter", None)
        consumer_index = -1 if counter is None else int(counter.consumer_index)
        pending = self._hicache.get(consumer_index)
        if pending is None:
            self._select_wrappers(False)
            self._active_batch = None
            super().init_forward_metadata(forward_batch)
            self._stats["batches"] += 1
            self._stats["resident_stock_batches"] += 1
            return

        self._select_wrappers(True)
        original_use_paged = self.use_paged
        self.use_paged = True
        try:
            super().init_forward_metadata(forward_batch)
            self._init_external_metadata(forward_batch, pending)
        except Exception as error:
            self._active_batch = None
            self._stats["hicache_fallback_batches"] += 1
            self._stats["last_hicache_fallback"] = str(error)
            self._write_stats()
            if not self._allow_fallback:
                raise RuntimeError(
                    "NTA failed to bind a claimed HiCache batch; set "
                    "NTA_SGLANG_ALLOW_FALLBACK=1 only for availability testing"
                ) from error
            logger.error(
                "NTA could not bind a HiCache batch; explicit stock fallback: %s",
                error,
            )
            self._hicache.fallback(pending)
            self._select_wrappers(False)
            self.use_paged = original_use_paged
            super().init_forward_metadata(forward_batch)
            self._stats["batches"] += 1
            return
        finally:
            self.use_paged = original_use_paged

    def _init_external_metadata(
        self, forward_batch: Any, pending: PendingHostLoad
    ) -> None:
        if self._opportunity_trace is not None:
            self._active_opportunity_batch = self._opportunity_batch
            self._opportunity_batch += 1
        metadata_started = time.perf_counter_ns() if self._profile_cpu else 0
        request_ids = forward_batch.rids
        if request_ids is None:
            raise RuntimeError("SGLang did not publish request IDs to NTA")
        # NTA owns compact batch-local request slots. Engine allocator slots
        # are intentionally volatile and would force identical CTA plans to
        # be rebuilt whenever SGLang recycles its request pool.
        bind_started = time.perf_counter_ns() if self._profile_cpu else 0
        bindings = self._bind_forward_requests(
            forward_batch, allow_capture_ids=False
        )
        if self._profile_cpu:
            self._stats["request_bind_cpu_ns"] = self._stats.get(
                "request_bind_cpu_ns", 0
            ) + (time.perf_counter_ns() - bind_started)
        if forward_batch.forward_mode.is_decode_or_idle():
            wrappers = self.forward_metadata.decode_wrappers
            extractor = decode_schedule
        else:
            if self.forward_metadata.use_ragged:
                raise RuntimeError("NTA requires SGLang paged prefill metadata")
            wrappers = self.forward_metadata.prefill_wrappers
            extractor = paged_prefill_schedule
        if self._pipeline_host:
            if not pending.prefetched_layers:
                raise RuntimeError("HiCache producer did not publish prefetched layers")
            self._active_batch = _ActiveBatch(
                bindings,
                {},
                pending,
                {},
                {},
                pending.prefetched_layers,
                pending.prefetch_tensors,
            )
            if self._profile_cpu:
                self._stats["metadata_cpu_ns"] = self._stats.get(
                    "metadata_cpu_ns", 0
                ) + (time.perf_counter_ns() - metadata_started)
            self._stats["batches"] += 1
            self._stats["hicache_claimed_batches"] += 1
            return

        schedules: dict[int, Schedule] = {}
        for wrapper in wrappers:
            schedule = extractor(wrapper)
            self._validate_schedule(schedule, bindings)
            schedules[id(wrapper)] = schedule
        pending_pages = set(pending.materialize_mapping())
        planned_pages: set[int] = set()
        page_pairs: dict[int, tuple[_PagePair, ...]] = {}
        for wrapper in wrappers:
            planned_pages.update(self._wrapper_pages(wrapper))
            page_pairs[id(wrapper)] = self._work_page_pairs(
                wrapper, schedules[id(wrapper)], pending
            )
        missing = pending_pages - planned_pages
        if missing:
            raise RuntimeError(
                f"attention metadata omits {len(missing)} promoted HiCache pages"
            )
        self._active_batch = _ActiveBatch(
            bindings, schedules, pending, page_pairs, {}, {}, ()
        )
        if self._profile_cpu:
            self._stats["metadata_cpu_ns"] = self._stats.get(
                "metadata_cpu_ns", 0
            ) + (time.perf_counter_ns() - metadata_started)
        self._stats["batches"] += 1
        self._stats["hicache_claimed_batches"] += 1

    def _prepare_host_pipeline(self, pending: PendingHostLoad) -> None:
        pipeline_started = time.perf_counter_ns() if self._profile_cpu else 0
        controller = pending.controller
        layer_count = int(controller.layer_num)
        if layer_count <= 0 or 2 * layer_count > self._object_capacity:
            raise RuntimeError("HiCache layer objects exceed NTA directory capacity")
        transfer_count = int(pending.host_indices.numel())
        if transfer_count <= 0 or transfer_count != int(pending.device_indices.numel()):
            raise RuntimeError("HiCache host pipeline has no promoted pages")
        transfer_source_indices, transfer_staging_indices = controller.move_indices(
            pending.host_indices, pending.device_indices
        )
        pending.producer_event.start_event.record()
        if not self._prefetch_ready_events:
            self._prefetch_ready_events = tuple(
                tuple(torch.cuda.Event() for _ in range(layer_count))
                for _ in controller.layer_done_counter.events
            )
        if pending.consumer_index >= len(self._prefetch_ready_events):
            raise RuntimeError("SGLang published an invalid HiCache producer slot")
        ready_events = self._prefetch_ready_events[pending.consumer_index]
        if layer_count > len(ready_events):
            raise RuntimeError("SGLang HiCache layer count changed after initialization")

        device_pool = controller.mem_pool_device
        start_layer = int(getattr(device_pool, "start_layer", 0))
        # Slots identify stable layer objects. Stream ordering and the recorded
        # ready event version each batch's contents, so plans can remain cached
        # across engine request-generation changes.
        version = 1
        layer_geometry: list[tuple[int, int]] = []
        for local_layer in range(layer_count):
            layer_id = start_layer + local_layer
            key_cache = device_pool._get_key_buffer(layer_id)
            value_cache = device_pool._get_value_buffer(layer_id)
            host_key = controller.mem_pool_host.k_data_refs[local_layer]
            host_value = controller.mem_pool_host.v_data_refs[local_layer]
            if (
                host_key.dtype != key_cache.dtype
                or host_value.dtype != value_cache.dtype
            ):
                raise RuntimeError("HiCache host and device KV dtypes disagree")
            key_element_bytes = key_cache[0].numel() * key_cache.element_size()
            value_element_bytes = value_cache[0].numel() * value_cache.element_size()
            key_bytes = transfer_count * key_element_bytes
            value_bytes = transfer_count * value_element_bytes
            if max(key_bytes, value_bytes) > _MAX_ABI_BYTES:
                raise RuntimeError("HiCache layer transfer exceeds the NTA ABI limit")
            layer_geometry.append((key_bytes, value_bytes))

        prefetched_layers: dict[int, _PrefetchedLayer] = {}
        try:
            with torch.cuda.stream(self._prefetch_stream):
                pending.producer_event.start_event.wait(self._prefetch_stream)
                for local_layer, (key_bytes, value_bytes) in enumerate(
                    layer_geometry
                ):
                    first_slot = 2 * local_layer
                    controller.mem_pool_host.load_to_device_per_layer(
                        device_pool,
                        transfer_source_indices,
                        transfer_staging_indices,
                        local_layer,
                        controller.io_backend,
                    )
                    ready_event = ready_events[local_layer]
                    ready_event.record(self._prefetch_stream)
                    prefetched_layers[local_layer] = _PrefetchedLayer(
                        first_slot,
                        _OBJECT_ID_BASE | (local_layer << 32),
                        first_slot + 1,
                        (_OBJECT_ID_BASE | (local_layer << 32)) | 1,
                        version,
                        key_bytes,
                        value_bytes,
                        ready_event,
                    )
        except Exception:
            # Stock fallback uses another stream. Drain any earlier layer
            # writes before it reuses the same destination rows.
            self._prefetch_stream.synchronize()
            raise
        pending.prefetched_layers = prefetched_layers
        pending.prefetch_tensors = (
            transfer_source_indices,
            transfer_staging_indices,
        )
        self._stats["prefetched_layers"] += layer_count
        self._stats["prefetched_host_bytes"] += sum(
            key_bytes + value_bytes for key_bytes, value_bytes in layer_geometry
        )
        if self._profile_cpu:
            self._stats["pipeline_cpu_ns"] = self._stats.get(
                "pipeline_cpu_ns", 0
            ) + (time.perf_counter_ns() - pipeline_started)

    def _wrapper_layout(
        self, wrapper: Any
    ) -> tuple[list[int], list[int], list[int], int]:
        batch_size = int(wrapper._batch_size)
        indptr = (
            wrapper._paged_kv_indptr_buf[: batch_size + 1]
            .detach()
            .to(device="cpu")
            .tolist()
        )
        page_count = int(indptr[-1])
        pages = (
            wrapper._paged_kv_indices_buf[:page_count]
            .detach()
            .to(device="cpu")
            .tolist()
        )
        last_page = (
            wrapper._paged_kv_last_page_len_buf[:batch_size]
            .detach()
            .to(device="cpu")
            .tolist()
        )
        page_size = int(getattr(wrapper, "_page_size", self.token_to_kv_pool.page_size))
        if page_size != 1:
            raise RuntimeError(
                "NTA's SGLang HiCache path currently requires page_size=1"
            )
        return indptr, pages, last_page, page_size

    def _wrapper_pages(self, wrapper: Any) -> tuple[int, ...]:
        _, pages, _, _ = self._wrapper_layout(wrapper)
        return tuple(pages)

    @staticmethod
    def _validate_schedule(
        schedule: Schedule, bindings: tuple[RequestBinding, ...]
    ) -> None:
        if schedule.work_count <= 0:
            raise RuntimeError("FlashInfer emitted no active CTA work")
        if schedule.work_count != len(schedule.kv_tile_indices):
            raise RuntimeError("FlashInfer schedule identity arrays disagree")
        cursor = 0
        for request_index in range(len(bindings)):
            begin = cursor
            while (
                cursor < schedule.work_count
                and schedule.request_indices[cursor] == request_index
            ):
                cursor += 1
            if cursor == begin:
                raise RuntimeError(
                    f"FlashInfer emitted no CTA work for request {request_index}"
                )
        if cursor != schedule.work_count:
            raise RuntimeError("FlashInfer CTA work is not request-contiguous")

    def _layer_execution_policy(
        self,
        wrapper: Any,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> HostExecutionPlan:
        batch = self._active_batch
        if batch is None or batch.pending_host_load is None:
            raise RuntimeError("host execution policy has no active HiCache load")
        schedule = batch.schedules.get(id(wrapper))
        pairs = batch.page_pairs.get(id(wrapper))
        if schedule is None or pairs is None:
            raise RuntimeError("host execution policy has no FlashInfer schedule")
        unique_pairs = {pair for pair in pairs if pair[0]}
        if not unique_pairs:
            raise RuntimeError("claimed HiCache batch has no external CTA dependency")
        key_cache, value_cache = kv_cache
        key_element_bytes = key_cache[0].numel() * key_cache.element_size()
        value_element_bytes = value_cache[0].numel() * value_cache.element_size()
        transfer_bytes = sum(
            len(pair[0]) * (key_element_bytes + value_element_bytes)
            for pair in unique_pairs
        )
        return plan_host_execution(
            object_count=2 * len(unique_pairs),
            transfer_bytes=transfer_bytes,
            runnable_tiles=schedule.work_count,
            model=self._host_cost_model,
        )

    def _run_bulk_host_layer(
        self,
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        output: torch.Tensor,
        layer: Any,
        run_options: dict[str, Any],
        stream: torch.cuda.Stream,
    ) -> None:
        batch = self._active_batch
        if batch is None or batch.pending_host_load is None:
            raise RuntimeError("bulk host execution has no active HiCache load")
        pending = batch.pending_host_load
        local_layer = int(layer.layer_id) - int(
            getattr(pending.controller.mem_pool_device, "start_layer", 0)
        )
        if not pending.prefetched_layers:
            self._prepare_host_pipeline(pending)
            batch.prefetched_layers.update(pending.prefetched_layers)
        prefetched = pending.prefetched_layers.get(local_layer)
        if prefetched is None:
            raise RuntimeError(f"bulk host policy omitted layer {layer.layer_id}")
        stream.wait_event(prefetched.ready_event)
        runtime_tensor = self._runtime.device_view_tensor
        wrapper.run(
            q,
            kv_cache,
            runtime_tensor,
            runtime_tensor,
            runtime_tensor,
            layer.scaling,
            len(batch.bindings),
            14,
            out=output,
            **run_options,
        )
        self._bulk_events = (prefetched.ready_event,)

    def _ensure_plan(
        self, wrapper: Any, layer_id: int, schedule: Schedule
    ) -> DeviceWorkPlan:
        key = (id(wrapper), layer_id)
        allocation = self._plans.get(key)
        if allocation is not None and schedule.work_count <= allocation.work_capacity:
            return allocation.plan
        if allocation is not None:
            torch.cuda.current_stream().synchronize()
            allocation.plan.close()
        capacity = schedule.work_count
        plan = DeviceWorkPlan(capacity, 2 * capacity, self._runtime.device_ordinal)
        self._plans[key] = _PlanAllocation(plan, capacity)
        return plan

    def _upload_plan(
        self,
        wrapper: Any,
        layer_id: int,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[
        DeviceWorkPlan,
        Schedule,
        int,
        torch.cuda.Event | None,
        HostExecutionPlan,
    ]:
        profile_started = time.perf_counter_ns() if self._profile_cpu else 0
        batch = self._active_batch
        if batch is None:
            raise RuntimeError("NTA attention ran without forward metadata")
        schedule = batch.schedules.get(id(wrapper))
        if schedule is None:
            raise RuntimeError("NTA attention wrapper was not planned for this batch")
        if schedule.work_count > self._work_ticket_capacity:
            raise RuntimeError(
                f"FlashInfer needs {schedule.work_count} work tickets; configured "
                f"capacity is {self._work_ticket_capacity}"
            )
        key_cache, value_cache = kv_cache
        if not key_cache.is_cuda or not value_cache.is_cuda:
            raise RuntimeError("SGLang KV cache must be CUDA-addressable")
        key_bytes = min(int(key_cache.nbytes), _MAX_ABI_BYTES)
        value_bytes = min(int(value_cache.nbytes), _MAX_ABI_BYTES)
        if key_bytes == 0 or value_bytes == 0:
            raise RuntimeError("SGLang exposed an empty KV cache allocation")
        pending = batch.pending_host_load
        if pending is None:
            raise RuntimeError("demand plan has no HiCache transfer")
        controller = pending.controller
        device_pool = controller.mem_pool_device
        local_layer = layer_id - int(getattr(device_pool, "start_layer", 0))
        if local_layer < 0 or local_layer >= int(controller.layer_num):
            raise RuntimeError(f"SGLang layer {layer_id} is outside the HiCache pool")
        prefetched = batch.prefetched_layers.get(local_layer)
        page_pairs = batch.page_pairs[id(wrapper)]
        signature = _plan_cache_signature(
            schedule.request_indices,
            schedule.kv_tile_indices,
            page_pairs,
            tuple(binding.request_slot for binding in batch.bindings),
            key_cache.data_ptr(),
            value_cache.data_ptr(),
            key_bytes,
            value_bytes,
            None
            if prefetched is None
            else (prefetched.key_bytes, prefetched.value_bytes),
        )
        plan = self._ensure_plan(wrapper, layer_id, schedule)
        allocation = self._plans[(id(wrapper), layer_id)]
        if allocation.signature == signature:
            if allocation.host_execution is None:
                raise RuntimeError("cached demand plan has no host execution policy")
            if self._profile_cpu:
                self._stats["plan_cpu_ns"] = self._stats.get("plan_cpu_ns", 0) + (
                    time.perf_counter_ns() - profile_started
                )
            return (
                plan,
                schedule,
                allocation.object_count,
                None if prefetched is None else prefetched.ready_event,
                allocation.host_execution,
            )

        host_key = controller.mem_pool_host.k_data_refs[local_layer]
        host_value = controller.mem_pool_host.v_data_refs[local_layer]
        if host_key.dtype != key_cache.dtype or host_value.dtype != value_cache.dtype:
            raise RuntimeError("HiCache host and device KV dtypes disagree")
        key_element_bytes = key_cache[0].numel() * key_cache.element_size()
        value_element_bytes = value_cache[0].numel() * value_cache.element_size()
        if key_element_bytes <= 0 or value_element_bytes <= 0:
            raise RuntimeError("HiCache exposed an empty KV row")

        allocation.object_version = (allocation.object_version + 1) & 0xFFFFFFFF
        allocation.object_version = allocation.object_version or 1
        version = allocation.object_version
        indexed_objects: list[IndexedHostObject] = []
        index_tensors: list[torch.Tensor] = []
        pair_objects: dict[_PagePair, tuple[int, int, int, int]] = {}

        def objects_for(pair: _PagePair) -> tuple[int, int, int, int]:
            if prefetched is not None:
                return (
                    prefetched.key_slot,
                    prefetched.key_object_id,
                    prefetched.value_slot,
                    prefetched.value_object_id,
                )
            existing = pair_objects.get(pair)
            if existing is not None:
                return existing
            host_pages, device_pages = pair
            index_map = batch.index_maps.get(pair)
            if index_map is None:
                index_map = (
                    torch.tensor(
                        host_pages, dtype=torch.int32, device=key_cache.device
                    ),
                    torch.tensor(
                        device_pages, dtype=torch.int32, device=key_cache.device
                    ),
                )
                batch.index_maps[pair] = index_map
            source_indices, staging_indices = index_map
            index_tensors.extend((source_indices, staging_indices))
            key_slot = len(indexed_objects)
            key_object_id = _OBJECT_ID_BASE | (local_layer << 32) | key_slot
            indexed_objects.append(
                IndexedHostObject(
                    key_object_id,
                    version,
                    host_key.data_ptr(),
                    key_cache.data_ptr(),
                    source_indices.data_ptr(),
                    staging_indices.data_ptr(),
                    len(host_pages),
                    key_element_bytes,
                    host_key.stride(0) * host_key.element_size(),
                    key_cache.stride(0) * key_cache.element_size(),
                )
            )
            value_slot = len(indexed_objects)
            value_object_id = _OBJECT_ID_BASE | (local_layer << 32) | value_slot
            indexed_objects.append(
                IndexedHostObject(
                    value_object_id,
                    version,
                    host_value.data_ptr(),
                    value_cache.data_ptr(),
                    source_indices.data_ptr(),
                    staging_indices.data_ptr(),
                    len(host_pages),
                    value_element_bytes,
                    host_value.stride(0) * host_value.element_size(),
                    value_cache.stride(0) * value_cache.element_size(),
                )
            )
            result = (key_slot, key_object_id, value_slot, value_object_id)
            pair_objects[pair] = result
            return result

        work_items: list[WorkItem] = []
        dependencies: list[AcquireRequirement] = []
        contributor_counts = Counter(schedule.request_indices)
        contributor_indices = {request_index: 0 for request_index in contributor_counts}
        for work_ticket, (request_index, kv_tile, pair) in enumerate(
            zip(schedule.request_indices, schedule.kv_tile_indices, page_pairs)
        ):
            binding = batch.bindings[request_index]
            dependency_begin = len(dependencies)
            if pair[0]:
                key_slot, key_object_id, value_slot, value_object_id = objects_for(pair)
                key_transfer_bytes = (
                    prefetched.key_bytes
                    if prefetched is not None
                    else len(pair[0]) * key_element_bytes
                )
                value_transfer_bytes = (
                    prefetched.value_bytes
                    if prefetched is not None
                    else len(pair[0]) * value_element_bytes
                )
                dependencies.extend(
                    (
                        AcquireRequirement(
                            0,
                            0,
                            key_object_id,
                            0,
                            key_slot,
                            version,
                            key_transfer_bytes,
                            0,
                        ),
                        AcquireRequirement(
                            0,
                            0,
                            value_object_id,
                            0,
                            value_slot,
                            version,
                            value_transfer_bytes,
                            0,
                        ),
                    )
                )
                direct_dependencies = 0
            else:
                dependencies.extend(
                    (
                        AcquireRequirement(
                            key_cache.data_ptr(),
                            0,
                            _OBJECT_ID_BASE | (layer_id << 1),
                            0,
                            0,
                            1,
                            key_bytes,
                            0,
                        ),
                        AcquireRequirement(
                            value_cache.data_ptr(),
                            0,
                            _OBJECT_ID_BASE | (layer_id << 1) | 1,
                            0,
                            1,
                            1,
                            value_bytes,
                            0,
                        ),
                    )
                )
                direct_dependencies = 2
            work_items.append(
                WorkItem(
                    request_index,
                    binding.request_slot,
                    binding.generation,
                    kv_tile,
                    dependency_begin,
                    2,
                    direct_dependencies,
                    work_ticket,
                    request_index,
                    contributor_indices[request_index],
                    contributor_counts[request_index],
                    self._host_cost_model.tile_compute_ns,
                )
            )
            contributor_indices[request_index] += 1

        ranges: list[RequestRange] = []
        cursor = 0
        for binding in batch.bindings:
            begin = cursor
            while (
                cursor < schedule.work_count
                and schedule.request_indices[cursor] == binding.request_index
            ):
                cursor += 1
            ranges.append(
                RequestRange(
                    begin,
                    cursor - begin,
                    binding.request_slot,
                    binding.generation,
                )
            )

        object_count = 2 if prefetched is not None else len(indexed_objects)
        if object_count == 0:
            raise RuntimeError("claimed HiCache batch has no external CTA dependency")
        if object_count > self._object_capacity:
            raise RuntimeError(
                f"HiCache layer needs {object_count} objects; configured capacity is "
                f"{self._object_capacity}"
            )
        transfer_bytes = (
            prefetched.key_bytes + prefetched.value_bytes
            if prefetched is not None
            else sum(
                object_.index_count * object_.element_bytes
                for object_ in indexed_objects
            )
        )
        host_execution = plan_host_execution(
            object_count=object_count,
            transfer_bytes=transfer_bytes,
            runnable_tiles=schedule.work_count,
            model=self._host_cost_model,
        )
        stream = torch.cuda.current_stream()
        if prefetched is None:
            self._runtime.register_indexed_host_objects(
                0, indexed_objects, stream=stream
            )
            self._stats["demand_host_layers"] += 1
        incremental = host_execution.rounds > 1 or self._force_incremental
        if incremental:
            upload_started = time.perf_counter_ns() if self._profile_cpu else 0
            plan.upload(work_items, dependencies, ranges, stream)
            if self._profile_cpu:
                self._stats["native_plan_upload_cpu_ns"] = self._stats.get(
                    "native_plan_upload_cpu_ns", 0
                ) + (time.perf_counter_ns() - upload_started)
            self._stats["plan_uploads"] += 1
            self._stats["cta_work_items"] += schedule.work_count
        allocation.signature = signature
        allocation.object_count = object_count
        allocation.index_tensors = tuple(index_tensors)
        allocation.host_execution = host_execution
        if prefetched is None:
            self._stats["indexed_host_objects"] += object_count
            self._stats["indexed_host_bytes"] += sum(
                object_.index_count * object_.element_bytes
                for object_ in indexed_objects
            )
            self._stats["host_progress_rounds"] += host_execution.rounds
            self._stats["predicted_atomic_ns"] += (
                host_execution.predicted_atomic_ns
            )
            self._stats["predicted_incremental_ns"] += (
                host_execution.predicted_incremental_ns
            )
            if host_execution.rounds > 1:
                self._stats["incremental_host_layers"] += 1
        if self._profile_cpu:
            self._stats["plan_cpu_ns"] = self._stats.get("plan_cpu_ns", 0) + (
                time.perf_counter_ns() - profile_started
            )
        return (
            plan,
            schedule,
            object_count,
            None if prefetched is None else prefetched.ready_event,
            host_execution,
        )

    def _work_page_pairs(
        self, wrapper: Any, schedule: Schedule, pending: PendingHostLoad
    ) -> tuple[_PagePair, ...]:
        indptr, pages, last_page, page_size = self._wrapper_layout(wrapper)
        host_by_device = pending.materialize_mapping()
        pairs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for request_index, kv_tile in zip(
            schedule.request_indices, schedule.kv_tile_indices
        ):
            if request_index < 0 or request_index + 1 >= len(indptr) or kv_tile < 0:
                raise RuntimeError("FlashInfer emitted an invalid KV tile coordinate")
            request_pages = pages[indptr[request_index] : indptr[request_index + 1]]
            if schedule.kv_chunk_tokens > 0:
                token_count = max(0, len(request_pages) - 1) * page_size + int(
                    last_page[request_index]
                )
                token_begin = kv_tile * schedule.kv_chunk_tokens
                token_end = min(token_count, token_begin + schedule.kv_chunk_tokens)
                page_begin = token_begin // page_size
                page_end = (token_end + page_size - 1) // page_size
                request_pages = request_pages[page_begin:page_end]
            device_pages = tuple(
                int(page) for page in request_pages if int(page) in host_by_device
            )
            host_pages = tuple(host_by_device[page] for page in device_pages)
            pairs.append((host_pages, device_pages))
        return tuple(pairs)

    def _phase_program(self, wrapper: Any) -> JitPhaseProgram:
        module_name = self._wrapper_modules[id(wrapper)]
        program = self._phase_programs.get(module_name)
        if program is not None:
            return program
        workspace_value = os.environ.get("FLASHINFER_WORKSPACE_BASE")
        if not workspace_value:
            raise RuntimeError(
                "FLASHINFER_WORKSPACE_BASE is missing; run SGLang through "
                "tools/jit/activate.py --flashinfer-hook"
            )
        modules = sorted(pathlib.Path(workspace_value).rglob(f"{module_name}.so"))
        if len(modules) != 1:
            raise RuntimeError(
                f"expected one compiled FlashInfer module {module_name}.so; "
                f"found {len(modules)}"
            )
        program = JitPhaseProgram(modules[0])
        self._phase_programs[module_name] = program
        return program

    def _run_attention(
        self,
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        layer: Any,
        *,
        causal: bool,
        window_left: int,
    ) -> torch.Tensor:
        if self._cuda_graph_mode:
            return self._run_graph_attention(
                wrapper,
                q,
                kv_cache,
                layer,
                causal=causal,
                window_left=window_left,
            )
        if layer.logit_cap not in (None, 0, 0.0):
            raise RuntimeError("NTA's FlashInfer adapter does not support logit caps")
        q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        output = torch.empty_like(q)
        verify_attention = os.environ.get("NTA_SGLANG_VERIFY_ATTENTION") == "1"
        verify_execution = (
            verify_attention or os.environ.get("NTA_SGLANG_VERIFY_EXECUTION") == "1"
        )
        if verify_execution:
            output.fill_(float("nan"))
        wrapper._causal = causal
        wrapper._window_left = window_left
        wrapper._logits_soft_cap = 0.0
        wrapper._sm_scale = layer.scaling
        batch = self._active_batch
        pending = batch.pending_host_load
        if pending is None:
            raise RuntimeError("eager external attention has no HiCache transfer")
        local_layer = int(layer.layer_id) - int(
            getattr(pending.controller.mem_pool_device, "start_layer", 0)
        )
        prefetched = batch.prefetched_layers.get(local_layer)
        stream = torch.cuda.current_stream()
        run_options = {
            "k_scale": layer.k_scale_float,
            "v_scale": layer.v_scale_float,
        }
        enqueue_started = time.perf_counter_ns() if self._profile_cpu else 0
        epoch = None
        progress_passes = 0
        if prefetched is not None:
            stream.wait_event(prefetched.ready_event)
            runtime_tensor = self._runtime.device_view_tensor
            wrapper.run(
                q,
                kv_cache,
                runtime_tensor,
                runtime_tensor,
                runtime_tensor,
                layer.scaling,
                len(batch.bindings),
                14,
                out=output,
                **run_options,
            )
            self._stats["planless_preacquired_launches"] = self._stats.get(
                "planless_preacquired_launches", 0
            ) + 1
        else:
            selected_policy = self._layer_execution_policy(wrapper, kv_cache)
            if selected_policy.rounds == 1 and not self._force_incremental:
                self._run_bulk_host_layer(
                    wrapper,
                    q,
                    kv_cache,
                    output,
                    layer,
                    run_options,
                    stream,
                )
                self._stats["bulk_host_batches"] += 1
                self._stats["predicted_atomic_ns"] += (
                    selected_policy.predicted_atomic_ns
                )
                self._stats["predicted_incremental_ns"] += (
                    selected_policy.predicted_incremental_ns
                )
            else:
                plan, schedule, object_count, _, host_execution = self._upload_plan(
                    wrapper, int(layer.layer_id), kv_cache
                )
                if host_execution != selected_policy:
                    raise RuntimeError("host execution policy changed during planning")
                epoch = FlashInferLayerEpoch(
                    self._runtime,
                    plan,
                    self._phase_program(wrapper),
                    object_count=object_count,
                    max_progress_passes=host_execution.rounds,
                    wait_for_plan=False,
                )
                progress_passes = host_execution.rounds
                epoch.enqueue_host(
                    wrapper,
                    q,
                    kv_cache,
                    output,
                    progress_blocks=host_execution.block_counts,
                    sm_scale=layer.scaling,
                    stream=stream,
                    progress_stream=self._prefetch_stream,
                    run_options=run_options,
                )
                epoch.check(progress_passes, stream)
                progress = self._runtime.request_progress_range(
                    0, len(batch.bindings)
                )
                if any(
                    item.failed_work != 0
                    or item.cancelled_work != 0
                    or item.completed_work != item.expected_work
                    or item.pending_work != 0
                    or item.runnable_work != 0
                    or item.unavailable_bytes != 0
                    or item.runnable_compute_ns != 0
                    for item in progress
                ):
                    raise RuntimeError(
                        "request-level progress disagrees with the completed epoch"
                    )
                self._stats["progress_snapshots"] += 1
                self._stats["request_work_completed"] += sum(
                    item.completed_work for item in progress
                )
                self._stats["request_work_failed"] += sum(
                    item.failed_work + item.cancelled_work for item in progress
                )
                self._stats["request_compute_completed_ns"] += sum(
                    item.completed_compute_ns for item in progress
                )
                if self._opportunity_trace is not None:
                    runnable_ns = self._runtime.work_runnable_ns(schedule.work_count)
                    tiles = tuple(
                        TileArrival(
                            request_id=(
                                f"{batch.bindings[request_index].request_id:016x}"
                            ),
                            tile_id=work_ticket,
                            available_ns=runnable_ns[work_ticket],
                            compute_ns=self._host_cost_model.tile_compute_ns,
                            logical_tile=schedule.kv_tile_indices[work_ticket],
                            availability_source=(
                                "resident_at_launch"
                                if runnable_ns[work_ticket] == 0
                                else "gpu_globaltimer"
                            ),
                        )
                        for work_ticket, request_index in enumerate(
                            schedule.request_indices
                        )
                    )
                    append_json_line(
                        self._opportunity_trace,
                        OperatorArrival(
                            batch_id=(
                                f"{os.getpid()}:{self._active_opportunity_batch}"
                            ),
                            layer=int(layer.layer_id),
                            tiles=tiles,
                            revision=self._opportunity_revision,
                            engine="sglang",
                            model=self._opportunity_model,
                            tier=self._opportunity_tier,
                            observed_at_unix_ns=time.time_ns(),
                        ),
                    )
        if self._profile_cpu:
            self._stats["phase_enqueue_cpu_ns"] = self._stats.get(
                "phase_enqueue_cpu_ns", 0
            ) + (time.perf_counter_ns() - enqueue_started)
        if os.environ.get("NTA_SGLANG_VERIFY_TRANSFER") == "1":
            self._verify_layer_transfer(int(layer.layer_id), kv_cache)
        if verify_execution:
            if epoch is None:
                stream.synchronize()
            if not torch.isfinite(output).all():
                raise RuntimeError(
                    f"instrumented FlashInfer did not write layer {layer.layer_id}"
                )
        if verify_attention:
            self._verify_attention_output(
                wrapper,
                q,
                kv_cache,
                output,
                layer,
                causal=causal,
                window_left=window_left,
            )
        self._stats["external_launches"] += 1
        self._hicache.complete_layer(pending, local_layer)
        if local_layer + 1 == int(pending.controller.layer_num):
            self._write_stats()
        return output.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _run_graph_attention(
        self,
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        layer: Any,
        *,
        causal: bool,
        window_left: int,
    ) -> torch.Tensor:
        if layer.logit_cap not in (None, 0, 0.0):
            raise RuntimeError("NTA's FlashInfer adapter does not support logit caps")
        batch = self._active_batch
        if batch is None:
            raise RuntimeError("NTA graph attention ran without request metadata")
        local_layer = int(layer.layer_id) - self._model_start_layer
        if local_layer < 0 or local_layer >= self._model_layer_count:
            raise RuntimeError(
                f"attention layer {layer.layer_id} is outside NTA graph state"
            )
        q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        output = torch.empty_like(q)
        wrapper._causal = causal
        wrapper._window_left = window_left
        wrapper._logits_soft_cap = 0.0
        wrapper._sm_scale = layer.scaling
        runtime_tensor = self._runtime.device_view_tensor
        wrapper.run(
            q,
            kv_cache,
            runtime_tensor,
            runtime_tensor,
            runtime_tensor,
            layer.scaling,
            len(batch.bindings),
            14,
            out=output,
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )
        return output.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _verify_attention_output(
        self,
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        actual: torch.Tensor,
        layer: Any,
        *,
        causal: bool,
        window_left: int,
    ) -> None:
        workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=q.device)
        batch_size = int(wrapper._batch_size)
        kv_indptr = wrapper._paged_kv_indptr_buf[: batch_size + 1]
        page_count = int(kv_indptr[-1].item())
        kv_indices = wrapper._paged_kv_indices_buf[:page_count]
        last_page_len = wrapper._paged_kv_last_page_len_buf[:batch_size]
        num_kv_heads = int(kv_cache[0].shape[-2])
        if isinstance(wrapper, BatchDecodeWithPagedKVCacheWrapper):
            reference_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                workspace,
                "NHD",
                backend="fa2",
                use_tensor_cores=self.decode_use_tensor_cores,
            )
            reference_wrapper.plan(
                kv_indptr,
                kv_indices,
                last_page_len,
                int(q.shape[1]),
                num_kv_heads,
                int(q.shape[2]),
                1,
                window_left=window_left,
                sm_scale=layer.scaling,
                q_data_type=q.dtype,
                kv_data_type=kv_cache[0].dtype,
            )
        else:
            qo_indptr = wrapper._qo_indptr_buf[: batch_size + 1]
            reference_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                workspace, "NHD", backend="fa2"
            )
            reference_wrapper.plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                last_page_len,
                int(q.shape[1]),
                num_kv_heads,
                int(q.shape[2]),
                1,
                causal=causal,
                window_left=window_left,
                sm_scale=layer.scaling,
                q_data_type=q.dtype,
                kv_data_type=kv_cache[0].dtype,
            )
        expected = reference_wrapper.run(
            q,
            kv_cache,
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )
        torch.cuda.current_stream().synchronize()
        difference = (actual.float() - expected.float()).abs()
        maximum = float(difference.max().item())
        mean = float(difference.mean().item())
        finite_fraction = float(torch.isfinite(actual).float().mean().item())
        actual_absmax = float(torch.nan_to_num(actual.float()).abs().max().item())
        expected_absmax = float(expected.float().abs().max().item())
        self._stats["last_attention_max_abs_error"] = maximum
        self._stats["last_attention_mean_abs_error"] = mean
        if not torch.allclose(actual, expected, rtol=2e-3, atol=2e-3):
            raise RuntimeError(
                "instrumented FlashInfer output differs from stock "
                f"(layer={layer.layer_id}, max={maximum:.6g}, mean={mean:.6g}, "
                f"finite={finite_fraction:.6g}, actual_absmax={actual_absmax:.6g}, "
                f"expected_absmax={expected_absmax:.6g})"
            )

    def _verify_layer_transfer(
        self, layer_id: int, kv_cache: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        pending = self._active_batch.pending_host_load
        if pending is None:
            raise RuntimeError("layer transfer verification has no HiCache transfer")
        controller = pending.controller
        local_layer = layer_id - int(
            getattr(controller.mem_pool_device, "start_layer", 0)
        )
        mapping = pending.materialize_mapping()
        device_pages = torch.tensor(
            tuple(mapping), dtype=torch.long, device=kv_cache[0].device
        )
        host_pages = torch.tensor(tuple(mapping.values()), dtype=torch.long)
        torch.cuda.current_stream().synchronize()
        expected_key = controller.mem_pool_host.k_data_refs[local_layer].index_select(
            0, host_pages
        )
        expected_value = controller.mem_pool_host.v_data_refs[local_layer].index_select(
            0, host_pages
        )
        actual_key = kv_cache[0].index_select(0, device_pages).cpu()
        actual_value = kv_cache[1].index_select(0, device_pages).cpu()
        for name, actual, expected in (
            ("key", actual_key, expected_key),
            ("value", actual_value, expected_value),
        ):
            unequal = actual != expected
            if unequal.any():
                bad_pages = unequal.flatten(1).any(1).nonzero().flatten().tolist()
                raise RuntimeError(
                    f"indexed {name} transfer mismatch on logical pages "
                    f"{bad_pages[:16]} ({len(bad_pages)}/{len(mapping)})"
                )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: Any,
        forward_batch: Any,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        if self._active_batch is None:
            return super().forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache
            )
        wrapper = self.forward_metadata.decode_wrappers[self._get_wrapper_idx(layer)]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        if k is not None:
            if v is None:
                raise ValueError("decode K and V must be supplied together")
            if save_kv_cache:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )
        kv_cache = (
            self.token_to_kv_pool._get_key_buffer(layer.layer_id),
            self.token_to_kv_pool._get_value_buffer(layer.layer_id),
        )
        self._stats["decode_launches"] += 1
        return self._run_attention(
            wrapper,
            q,
            kv_cache,
            layer,
            causal=False,
            window_left=layer.sliding_window_size,
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: Any,
        forward_batch: Any,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        if self._active_batch is None:
            return super().forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache
            )
        if self.forward_metadata.use_ragged:
            raise RuntimeError("NTA requires paged FlashInfer prefill")
        wrapper = self.forward_metadata.prefill_wrappers[self._get_wrapper_idx(layer)]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        if k is not None:
            if v is None:
                raise ValueError("prefill K and V must be supplied together")
            if save_kv_cache:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )
        kv_cache = (
            self.token_to_kv_pool._get_key_buffer(layer.layer_id),
            self.token_to_kv_pool._get_value_buffer(layer.layer_id),
        )
        causal = (
            not layer.is_cross_attention
            and layer.attn_type != AttentionType.ENCODER_ONLY
        )
        window_left = (
            layer.sliding_window_size
            if not (
                self.forward_metadata.multi_item_params
                and self.forward_metadata.multi_item_params.is_enabled()
            )
            else -1
        )
        self._stats["prefill_launches"] += 1
        return self._run_attention(
            wrapper,
            q,
            kv_cache,
            layer,
            causal=causal,
            window_left=window_left,
        )

    def _write_stats(self) -> None:
        configured = os.environ.get("NTA_ENGINE_STATS_FILE")
        if not configured:
            return
        path = pathlib.Path(configured)
        if path.suffix:
            path = path.with_name(f"{path.stem}.{os.getpid()}{path.suffix}")
        else:
            path = path / f"nta-sglang-{os.getpid()}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        report = dict(self._stats)
        report["finished_unix_ns"] = time.time_ns()
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
