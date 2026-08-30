"""vLLM 0.26 composition root for instrumented FlashInfer attention.

``vllm_worker`` owns the framework-private runner lifecycle, ``vllm_modules``
owns setup-time module materialization, and ``vllm_execution`` validates typed
preparations.  This module contains the registered metadata/attention backend
and submits the same NTA work-plan ABI used by the SGLang adapter.

The qualified profile is one KV group, FA2 prefill and single-token decode,
including heterogeneous mixed batches, without CUDA graph capture. HBM uses
resident CUDA KV; host-staged materializes exact pinned rows into vLLM's packed
HBM allocation; NVMe binds stable storage identities to the same framework-
owned blocks. CXL-DAX remains explicitly fail-closed until vLLM's numerical
block-table pointer can name that address space. Unsupported resident-HBM
features may use the reference implementation only when explicitly enabled;
external tiers always fail closed after admission because a partial transfer
cannot safely fall back to resident attention.
"""

from __future__ import annotations

import atexit
from collections.abc import Callable
import json
import os
import pathlib
import time
from typing import Any

import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)
from vllm.v1.attention.backends.flashinfer import (
    FIPrefill,
    FlashInferBackend,
    FlashInferMetadataBuilder,
    FlashInferImpl,
    FlashInferMetadata,
)
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides

from nta_runtime.adapters.base import ConsumerContract, EngineBatch
from nta_runtime.adapters.vllm_v1 import (
    current_vllm_v1_forward_state,
)
from nta_runtime.execution_core import ExecutionPlan, ExecutionSession, ExecutionTile
from nta_runtime.execution_topology import ExactWorkTopology, WorkDependencySpan
from nta_runtime.execution_protocol import ExecutionProtocolConfig
from nta_runtime.engines.vllm_config import (
    SUPPORTED_VLLM_VERSION,
    VllmAttentionConfig,
)
from nta_runtime.engines.vllm_execution import (
    VllmDecodeBuffers,
    VllmPrefillBuffers,
    VllmScheduleContext,
    VllmSchedulePublication,
    prepare_host_layer,
    require_decode_buffers,
    require_prefill_buffers,
)
from nta_runtime.engines.vllm_modules import (
    VLLM_STATS,
    _DEFAULT_MODULES,
    _REQUEST_BOUND_MODULES,
    _default_workspace_bytes,
    _find_module,
    _new_request_bound_wrapper,
    _operator_module,
    _transport_program,
    _prepare_attention_modules,
)
from nta_runtime.engines.vllm_worker import (
    AttentionWorkspaceContract,
    VllmV1WorkerController,
    _PhysicalLayerDestination,
    _abort_forward,
    _commit_forward,
    _controller,
    _physical_transfer_layout,
)
from nta_runtime.indexed_transfer import (
    IndexedTensorLane,
)
from nta_runtime.nvme_materialization import (
    RegisteredNvmeObjectBinding,
    publish_registered_nvme_objects,
)
from nta_runtime.flashinfer import (
    FlashInferLayerEpoch,
    PREACQUIRED_LAUNCH_FLAGS,
    attention_jit_args,
    direct_requirement,
    enqueue_resident_attention,
    object_requirement,
)
from nta_runtime.flashinfer_schedule import decode_schedule, paged_prefill_schedule
from nta_runtime.runtime import (
    AcquireRequirement,
    DeviceWorkPlan,
    IndexedHostIndexBinding,
    IndexedAcquisitionPlan,
    JitPhaseProgram,
    Runtime,
)


__all__ = [
    "NtaVllmFlashInferBackend",
    "NtaVllmFlashInferImpl",
    "NtaVllmFlashInferMetadataBuilder",
    "VLLM_STATS",
    "VllmV1WorkerController",
    "_abort_forward",
    "_commit_forward",
    "_controller",
    "_physical_transfer_layout",
    "consumer_contract",
]


class NtaVllmFlashInferMetadataBuilder(FlashInferMetadataBuilder):
    """Let vLLM plan the typed direct wrapper at its native ownership edge."""

    _cudagraph_support = AttentionCGSupport.NEVER

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._nta_config = VllmAttentionConfig.from_environment(
            default_workspace_bytes=_default_workspace_bytes()
        )
        _prepare_attention_modules(
            self._nta_config,
            (self.q_data_type_prefill, self.q_data_type_decode),
            self.head_dim,
        )
        self._nta_direct_prefill_wrapper: Any | None = None
        self._nta_direct_decode_wrapper: Any | None = None

    def build(self, *args: Any, **kwargs: Any) -> Any:
        if not self._nta_config.profile_cpu:
            return super().build(*args, **kwargs)
        started = time.perf_counter_ns()
        try:
            return super().build(*args, **kwargs)
        finally:
            VLLM_STATS["metadata_build_cpu_ns"] += time.perf_counter_ns() - started
            VLLM_STATS["metadata_build_calls"] += 1

    def _direct_batch(self) -> EngineBatch | None:
        # Differential mode deliberately keeps the stock metadata wrapper so
        # the attention consumer can execute both implementations.  It is a
        # correctness diagnostic and must never contaminate performance data.
        # The request-bound wrapper has no acquisition edge: it is therefore a
        # resident-HBM fast path only. External tiers must preserve FlashInfer's
        # work schedule so their exact transfer dependencies can be attached.
        if (
            not self._nta_config.native_enabled
            or self._nta_config.compare_stock
            or self._nta_config.serving_tier != "hbm"
        ):
            return None
        state = current_vllm_v1_forward_state()
        if (
            state is None
            or state.reference_warmup
            or not isinstance(state.batch, EngineBatch)
        ):
            return None
        return state.batch

    def _get_prefill_wrapper(self, causal: bool = True) -> Any:
        batch = self._direct_batch()
        if batch is None:
            return super()._get_prefill_wrapper(causal=causal)
        if self.use_dcp or self.is_kvcache_nvfp4:
            raise RuntimeError("vLLM NTA direct prefill does not support DCP or NVFP4")
        if self._nta_direct_prefill_wrapper is None:
            self._nta_direct_prefill_wrapper = _new_request_bound_wrapper(
                "prefill",
                self._get_workspace_buffer(),
                query_dtype=self.q_data_type_prefill,
                kv_dtype=self.kv_cache_spec.dtype,
                head_size=self.head_dim,
                workspace_base=self._nta_config.require_workspace(),
            )
        VLLM_STATS["framework_direct_prefill_plans"] += 1
        return self._nta_direct_prefill_wrapper

    def _get_decode_wrapper(self, batch_size: int, use_cudagraph: bool = False) -> Any:
        batch = self._direct_batch()
        if batch is None or use_cudagraph:
            return super()._get_decode_wrapper(batch_size, use_cudagraph)
        if self.use_dcp or self.is_kvcache_nvfp4:
            raise RuntimeError("vLLM NTA direct decode does not support DCP or NVFP4")
        if self._nta_direct_decode_wrapper is None:
            self._nta_direct_decode_wrapper = _new_request_bound_wrapper(
                "decode",
                self._get_workspace_buffer(),
                query_dtype=self.q_data_type_decode,
                kv_dtype=self.kv_cache_spec.dtype,
                head_size=self.head_dim,
                workspace_base=self._nta_config.require_workspace(),
            )
        VLLM_STATS["framework_direct_decode_plans"] += 1
        return self._nta_direct_decode_wrapper

    @classmethod
    def get_cudagraph_support(
        cls, vllm_config: Any, kv_cache_spec: Any
    ) -> AttentionCGSupport:
        return AttentionCGSupport.NEVER


class NtaVllmFlashInferBackend(FlashInferBackend):
    """vLLM registry backend whose ``impl.forward`` launches NTA work units."""

    @staticmethod
    def get_impl_cls() -> type["NtaVllmFlashInferImpl"]:
        from nta_runtime.plugins.vllm import ensure_worker_bridge

        ensure_worker_bridge()
        return NtaVllmFlashInferImpl

    @staticmethod
    def get_builder_cls() -> type[NtaVllmFlashInferMetadataBuilder]:
        return NtaVllmFlashInferMetadataBuilder


class NtaVllmFlashInferImpl(FlashInferImpl):
    """Native NTA consumer for exact FlashInfer prefill and decode."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        from nta_runtime.plugins.vllm import ensure_worker_bridge

        ensure_worker_bridge()
        super().__init__(*args, **kwargs)
        self._nta_wrapper: BatchDecodeWithPagedKVCacheWrapper | None = None
        self._nta_prefill_wrapper: BatchPrefillWithPagedKVCacheWrapper | None = None
        self._nta_plan: DeviceWorkPlan | None = None
        self._nta_plan_capacity = 0
        self._nta_dependency_capacity = 0
        self._nta_program: JitPhaseProgram | None = None
        # Native mode is explicit. Physical tiers are additionally checked by
        # the worker resource owner and are never silently routed through the
        # resident framework path.
        self._nta_config = VllmAttentionConfig.from_environment(
            default_workspace_bytes=_default_workspace_bytes()
        )
        self._native_enabled = self._nta_config.native_enabled
        self._serving_tier = self._nta_config.serving_tier
        self._profile_cpu = self._nta_config.profile_cpu
        self._verify_semantics = self._nta_config.verify_execution

    def _request_bound_wrapper(
        self,
        kind: str,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        workspace: torch.Tensor,
    ) -> Any:
        if isinstance(self.kv_cache_dtype, str) and self.kv_cache_dtype not in {
            "auto",
            "float16",
            "bfloat16",
        }:
            raise RuntimeError(
                "native vLLM direct attention does not support quantized KV "
                f"cache dtype {self.kv_cache_dtype!r}"
            )
        return _new_request_bound_wrapper(
            kind,
            workspace,
            query_dtype=query.dtype,
            kv_dtype=kv_cache.dtype,
            head_size=self.head_size,
            workspace_base=self._nta_config.require_workspace(),
        )

    @staticmethod
    def _framework_workspace_storage(
        stock_wrapper: Any, query: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        """Validate and return the framework's worker-lifetime storage."""

        workspace = getattr(stock_wrapper, "_float_workspace_buffer", None)
        if not isinstance(workspace, torch.Tensor):
            raise RuntimeError("vLLM stock wrapper has no FlashInfer workspace")
        if (
            workspace.dtype != torch.uint8
            or not workspace.is_cuda
            or not workspace.is_contiguous()
            or workspace.device != query.device
        ):
            raise RuntimeError(
                "vLLM stock wrapper workspace is not contiguous uint8 CUDA storage"
            )
        capacity = workspace.numel() * workspace.element_size()
        return workspace, capacity

    @classmethod
    def _framework_workspace(
        cls,
        stock_wrapper: Any,
        query: torch.Tensor,
        required_bytes: int,
    ) -> tuple[torch.Tensor, AttentionWorkspaceContract]:
        """Borrow a sufficiently large framework FlashInfer workspace."""

        workspace, capacity = cls._framework_workspace_storage(stock_wrapper, query)
        if capacity < required_bytes:
            raise RuntimeError(
                "vLLM stock wrapper workspace is smaller than the NTA contract: "
                f"{capacity} < {required_bytes}"
            )
        return workspace, AttentionWorkspaceContract.framework_owned(
            capacity, workspace.data_ptr()
        )

    def _request_bound_available(self, state: Any, owner: Any) -> bool:
        """Return whether exact KV has a direct-consumer readiness contract.

        HBM is resident by construction. The Host connector publishes one
        readiness fence per layer; the direct consumer orders itself on that
        fence without waiting for the remaining model layers.
        """

        if not isinstance(owner, VllmV1WorkerController):
            return False
        if self._serving_tier == "hbm":
            return True
        if self._serving_tier != "host_staged":
            return False
        pairs = tuple(getattr(state, "host_transfer_pairs", ()))
        acquisition = getattr(state, "host_acquisition", None)
        isolation = bool(getattr(state, "tenant_isolation_enabled", False))
        if isolation != owner.tenant_isolation_enabled:
            raise RuntimeError("vLLM tenant-isolation state disagrees with its owner")
        if isolation and pairs:
            if acquisition is not None and not acquisition.tenant_accounted:
                raise RuntimeError(
                    "vLLM Host acquisition bypassed finite tenant byte credits"
                )
            return acquisition is not None
        if acquisition is not None:
            if not pairs:
                raise RuntimeError(
                    "vLLM Host acquisition has no exact transfer ownership"
                )
            return True
        return not pairs

    def _phase_signature(
        self,
        kind: str,
        module_name: str,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        *,
        page_size: int,
        causal: bool,
        workspace_bytes: int,
    ) -> tuple[Any, ...]:
        """Return every invariant that can change a FlashInfer phase plan."""
        return (
            kind,
            module_name,
            str(self._nta_config.workspace_base or ""),
            str(query.device),
            query.dtype,
            kv_cache.dtype,
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            page_size,
            get_kv_cache_layout(),
            float(self.scale),
            bool(causal),
            int(self.window_left),
            float(self.logits_soft_cap or 0.0),
            workspace_bytes,
        )

    @staticmethod
    def _kv_cache_tuple(kv_cache: torch.Tensor) -> tuple[torch.Tensor, ...]:
        permuted = kv_cache.permute(*FlashInferBackend.get_kv_cache_stride_order())
        permuted = canonicalize_singleton_dim_strides(permuted)
        return tuple(permuted.split(permuted.shape[-1] // 2, dim=-1))

    def _compare_stock(
        self,
        stock_wrapper: Any,
        query: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        output: torch.Tensor,
        *,
        custom_wrapper: Any | None = None,
        schedule: Any | None = None,
        kind: str = "attention",
        diagnostic_context: dict[str, Any] | None = None,
    ) -> None:
        if not self._nta_config.compare_stock:
            return
        stock_output = torch.empty_like(output)
        stock_wrapper.run(query, kv_cache, out=stock_output)
        torch.cuda.synchronize()
        absolute = torch.nan_to_num(
            (output.float() - stock_output.float()).abs(), nan=float("inf")
        )
        difference = absolute.max().item()
        reference = torch.nan_to_num(stock_output.float().abs(), nan=float("inf"))
        tolerance = 2e-3 + 2e-3 * reference.max().item()
        VLLM_STATS["native_stock_diff_max_e9"] = max(
            VLLM_STATS["native_stock_diff_max_e9"], int(difference * 1_000_000_000)
        )
        if not difference <= tolerance:
            row_error = absolute.reshape(absolute.shape[0], -1).amax(dim=1)
            worst_row = int(row_error.argmax().item())
            row_nonzero = torch.count_nonzero(
                output.reshape(output.shape[0], -1), dim=1
            )
            nonzero_rows = torch.nonzero(row_nonzero, as_tuple=False).view(-1)

            def wrapper_summary(wrapper: Any) -> dict[str, Any]:
                summary: dict[str, Any] = {
                    "plan": tuple(int(value) for value in wrapper._plan_info),
                }
                for name in (
                    "_qo_indptr_buf",
                    "_paged_kv_indptr_buf",
                    "_paged_kv_last_page_len_buf",
                    "_request_indices_buf",
                    "_qo_tile_indices_buf",
                    "_kv_tile_indices_buf",
                ):
                    tensor = getattr(wrapper, name, None)
                    if isinstance(tensor, torch.Tensor):
                        values = tensor.detach().view(-1)[:16].cpu().tolist()
                        summary[name] = tuple(int(value) for value in values)
                return summary

            detail = {
                "kind": kind,
                "query_shape": tuple(int(value) for value in query.shape),
                "kv_shapes": tuple(
                    tuple(int(value) for value in tensor.shape) for tensor in kv_cache
                ),
                "worst_row": worst_row,
                "worst_row_abs": float(row_error[worst_row].item()),
                "output_abs_max": float(
                    torch.nan_to_num(output.float().abs(), nan=float("inf"))
                    .max()
                    .item()
                ),
                "stock_abs_max": float(reference.max().item()),
                "output_nonzero": int(torch.count_nonzero(output).item()),
                "nonzero_row_range": (
                    (
                        int(nonzero_rows[0].item()),
                        int(nonzero_rows[-1].item()),
                        int(nonzero_rows.numel()),
                    )
                    if nonzero_rows.numel()
                    else None
                ),
                "window_left": int(self.window_left),
                "logits_soft_cap": float(self.logits_soft_cap or 0.0),
                "scale": float(self.scale),
                "stock": wrapper_summary(stock_wrapper),
                "custom": (
                    wrapper_summary(custom_wrapper)
                    if custom_wrapper is not None
                    else None
                ),
                "schedule_requests": (
                    tuple(int(value) for value in schedule.request_indices[:16])
                    if schedule is not None
                    else None
                ),
                "schedule_kv_tiles": (
                    tuple(int(value) for value in schedule.kv_tile_indices[:16])
                    if schedule is not None
                    else None
                ),
                "schedule_work_count": (
                    int(schedule.work_count) if schedule is not None else None
                ),
                "context": diagnostic_context,
            }
            raise RuntimeError(
                "vLLM NTA attention disagrees with stock FlashInfer: "
                f"max_abs={difference:.8g}, tolerance={tolerance:.8g}, "
                f"detail={detail!r}"
            )

    def _run_request_bound(
        self,
        state: Any,
        batch: EngineBatch,
        wrapper: Any,
        stock_wrapper: Any,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        *,
        kind: str,
        framework_owned: bool,
        phase_start: int,
    ) -> torch.Tensor:
        started = time.perf_counter_ns() if self._profile_cpu else 0
        request_bindings = state.phase_request_bindings(
            phase_start, len(batch.bindings)
        )
        if (
            not isinstance(request_bindings, torch.Tensor)
            or request_bindings.dtype != torch.int64
            or not request_bindings.is_cuda
            or not request_bindings.is_contiguous()
            or request_bindings.numel() != 2 * len(batch.bindings)
        ):
            raise RuntimeError("vLLM direct attention has no typed request bindings")
        if self._serving_tier == "host_staged":
            state.wait_for_host_layer(torch.cuda.current_stream(query.device))
        kv_cache_for_flashinfer = self._kv_cache_tuple(kv_cache)
        owner = getattr(state, "execution_owner", None)
        record_consumer = getattr(owner, "record_request_binding_consumer", None)
        if not callable(record_consumer):
            raise RuntimeError("vLLM direct attention has no binding-table owner")
        stream = torch.cuda.current_stream(query.device)
        try:
            wrapper.run(
                query,
                kv_cache_for_flashinfer,
                state.hook.runtime.device_view_tensor,
                request_bindings,
                self.scale,
                out=output,
            )
        finally:
            record_consumer(request_bindings, stream)
        self._compare_stock(
            stock_wrapper,
            query,
            kv_cache_for_flashinfer,
            output,
            custom_wrapper=wrapper,
            kind=kind,
        )
        state.record_native_launch(
            kind,
            len(batch.bindings),
            form="request_bound",
            framework_owned=framework_owned,
            serving_tier=self._serving_tier,
        )
        if self._profile_cpu:
            state.record_profile_ns(
                "direct_submit_cpu_ns", time.perf_counter_ns() - started
            )
        return output

    @staticmethod
    def _physical_pages(
        batch: EngineBatch,
        schedule: Any,
        request_index: int,
        kv_tile: int,
        page_size: int,
    ) -> tuple[int, ...]:
        if batch.exact_demand is None:
            raise RuntimeError("vLLM physical plan has no exact page demand")
        if request_index < 0 or request_index >= len(
            batch.exact_demand.request_unit_ids
        ):
            raise RuntimeError("vLLM physical plan referenced an invalid request")
        pages = batch.exact_demand.request_unit_ids[request_index]
        chunk_tokens = int(getattr(schedule, "kv_chunk_tokens", 0))
        if chunk_tokens == 0:
            if kv_tile != 0:
                raise RuntimeError(
                    "unsplit vLLM FlashInfer schedule emitted a nonzero KV tile"
                )
            selected = pages
        else:
            if chunk_tokens % page_size != 0:
                raise RuntimeError(
                    "vLLM physical acquisition requires page-aligned FlashInfer "
                    "KV chunks"
                )
            pages_per_tile = chunk_tokens // page_size
            begin = kv_tile * pages_per_tile
            selected = pages[begin : begin + pages_per_tile]
        if not selected:
            raise RuntimeError("vLLM physical schedule selected no KV pages")
        return tuple(int(page) for page in selected)

    @classmethod
    def _external_page_bindings(
        cls,
        batch: EngineBatch,
        schedule: Any,
        request_index: int,
        kv_tile: int,
        page_size: int,
        storage_key_tables: tuple[tuple[str | None, ...], ...],
    ) -> tuple[tuple[str, int], ...]:
        """Select external stable identities for one exact FlashInfer tile."""

        if batch.exact_demand is None:
            raise RuntimeError("vLLM physical plan has no exact page demand")
        if request_index < 0 or request_index >= len(storage_key_tables):
            raise RuntimeError("vLLM physical plan has no storage-key row")
        pages = tuple(
            int(page) for page in batch.exact_demand.request_unit_ids[request_index]
        )
        keys = storage_key_tables[request_index]
        if len(keys) != len(pages):
            raise RuntimeError("vLLM storage identities do not align with block IDs")
        selected = cls._physical_pages(
            batch, schedule, request_index, kv_tile, page_size
        )
        by_page = dict(zip(pages, keys, strict=True))
        if len(by_page) != len(pages):
            raise RuntimeError("vLLM exact block row contains duplicate destinations")
        bindings: list[tuple[str, int]] = []
        for page in selected:
            key = by_page[page]
            if key is not None:
                bindings.append((key, page))
        return tuple(bindings)

    def _ensure_work_plan(
        self, runtime: Runtime, work_count: int, dependency_capacity: int
    ) -> DeviceWorkPlan:
        if work_count <= 0 or dependency_capacity <= 0:
            raise ValueError("vLLM work-plan capacities must be positive")
        if (
            self._nta_plan is None
            or work_count > self._nta_plan_capacity
            or dependency_capacity > self._nta_dependency_capacity
        ):
            if self._nta_plan is not None:
                torch.cuda.current_stream().synchronize()
                self._nta_plan.close()
            self._nta_plan = DeviceWorkPlan(
                work_count, dependency_capacity, runtime.device_ordinal
            )
            self._nta_plan_capacity = work_count
            self._nta_dependency_capacity = dependency_capacity
        return self._nta_plan

    def _incremental_module_name(
        self, kind: str, query: torch.Tensor, kv_cache: torch.Tensor
    ) -> str:
        if kind not in {"decode", "prefill"}:
            raise ValueError(f"unknown vLLM attention phase {kind!r}")
        if query.dtype not in _DEFAULT_MODULES or kv_cache.dtype != query.dtype:
            raise RuntimeError(
                f"native vLLM NTA {kind} requires matching float16 or "
                "bfloat16 query and KV-cache dtypes"
            )
        if isinstance(self.kv_cache_dtype, str) and self.kv_cache_dtype not in {
            "auto",
            "float16",
            "bfloat16",
        }:
            raise RuntimeError(
                f"native vLLM NTA {kind} does not support quantized KV cache "
                f"dtype {self.kv_cache_dtype!r}"
            )
        override = self._nta_config.module_override(kind)
        return override or _DEFAULT_MODULES[query.dtype]

    def _build_incremental_wrapper(
        self,
        kind: str,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        module_name: str,
        workspace: torch.Tensor,
    ) -> Any:
        if self._nta_config.module_override(kind) is not None:
            module_path = _find_module(
                module_name, self._nta_config.require_workspace()
            )
        else:
            module_path = _find_module(
                module_name, self._nta_config.require_workspace()
            )
        operator_module = _operator_module(module_path)
        transport_program = _transport_program()
        jit_args = attention_jit_args(
            module_name,
            dtype_q=query.dtype,
            dtype_kv=kv_cache.dtype,
            dtype_o=query.dtype,
            idtype=torch.int32,
            head_dim_qk=self.head_size,
            head_dim_vo=self.head_size,
        )
        if kind == "decode":
            wrapper = BatchDecodeWithPagedKVCacheWrapper(
                workspace,
                get_kv_cache_layout(),
                backend="fa2",
                use_tensor_cores=True,
                jit_args=jit_args,
            )
        else:
            wrapper = BatchPrefillWithPagedKVCacheWrapper(
                workspace,
                get_kv_cache_layout(),
                backend="fa2",
                jit_args=jit_args,
            )
        # The wrapper is the lifetime owner retained by the worker phase.  Keep
        # its verified module program adjacent so every layer can submit the
        # same resource without rescanning the JIT workspace.
        wrapper._nta_operator_module = operator_module
        wrapper._nta_transport_program = transport_program
        wrapper._nta_module_path = module_path
        return wrapper

    def _ensure_local_incremental_wrapper(
        self, kind: str, query: torch.Tensor, kv_cache: torch.Tensor
    ) -> Any:
        attribute = "_nta_wrapper" if kind == "decode" else "_nta_prefill_wrapper"
        wrapper = getattr(self, attribute)
        if wrapper is None:
            module_name = self._incremental_module_name(kind, query, kv_cache)
            workspace = torch.empty(
                self._nta_config.workspace_bytes,
                dtype=torch.uint8,
                device=query.device,
            )
            wrapper = self._build_incremental_wrapper(
                kind,
                query,
                kv_cache,
                module_name,
                workspace,
            )
            setattr(self, attribute, wrapper)
            VLLM_STATS["local_incremental_wrapper_builds"] += 1
        self._nta_program = wrapper._nta_transport_program
        return wrapper

    def _build_plan(
        self,
        batch: EngineBatch,
        schedule: Any,
        *,
        layer: int = 0,
    ) -> ExecutionPlan:
        if batch.exact_demand is None:
            raise RuntimeError("vLLM NTA attention has no exact engine batch")
        bindings = batch.bindings
        if any(
            request_index < 0 or request_index >= len(bindings)
            for request_index in schedule.request_indices
        ):
            raise RuntimeError("FlashInfer schedule referenced an invalid vLLM request")
        if len(batch.exact_demand.request_unit_ids) != len(bindings):
            raise RuntimeError("vLLM exact demand does not match request bindings")
        from collections import Counter

        contributor_counts = Counter(schedule.request_indices)
        contributor_indices = Counter()
        tiles: list[ExecutionTile] = []
        for work_id, (request_index, kv_tile) in enumerate(
            zip(schedule.request_indices, schedule.kv_tile_indices, strict=True)
        ):
            binding = bindings[request_index]
            candidates = len(batch.exact_demand.request_unit_ids[request_index])
            tiles.append(
                ExecutionTile(
                    work_id=work_id,
                    binding=binding,
                    layer=layer,
                    logical_begin=int(kv_tile),
                    candidate_units=candidates,
                    selected_ids=(),
                    unit_bytes=batch.exact_demand.unit_bytes,
                    ready=True,
                    estimated_compute_ns=1,
                    reduction_group=request_index,
                    contributor_index=contributor_indices[request_index],
                    contributor_count=contributor_counts[request_index],
                )
            )
            contributor_indices[request_index] += 1
        if not tiles:
            raise RuntimeError("FlashInfer produced no vLLM decode work units")
        return ExecutionPlan.from_tiles(
            epoch=batch.epoch,
            granularity=batch.granularity,
            protocol=ExecutionProtocolConfig.late_bound(
                granularity=batch.granularity,
                max_inflight_units=max(1, len(tiles)),
            ),
            tiles=tiles,
        )

    def _build_topology(
        self,
        batch: EngineBatch,
        schedule: Any,
    ) -> ExactWorkTopology:
        """Build the production exact contract without a semantic state graph."""

        if batch.exact_demand is None:
            raise RuntimeError("vLLM NTA attention has no exact engine batch")
        demand = batch.exact_demand
        if any(
            request_index < 0 or request_index >= len(batch.bindings)
            for request_index in schedule.request_indices
        ):
            raise RuntimeError("FlashInfer schedule referenced an invalid vLLM request")
        return ExactWorkTopology.from_schedule(
            epoch=batch.epoch,
            bindings=batch.bindings,
            request_indices=schedule.request_indices,
            logical_work=schedule.kv_tile_indices,
            demand_units=tuple(
                len(demand.request_unit_ids[request_index])
                for request_index in schedule.request_indices
            ),
            unit_bytes=demand.unit_bytes,
            estimated_compute_ns=1,
        )

    def _upload_physical_plan(
        self,
        state: Any,
        batch: EngineBatch,
        schedule: Any,
        topology: ExactWorkTopology,
        destination: _PhysicalLayerDestination,
        page_size: int,
    ) -> tuple[DeviceWorkPlan, int, bool]:
        """Bind stable storage keys to framework-owned packed KV blocks.

        The FlashInfer overlay gates numerical execution but deliberately
        keeps its original paged-KV pointers.  Every transport object must
        therefore DMA into the exact vLLM block that those pointers address;
        runtime-private staging would publish readiness for the wrong bytes.
        """
        tier = getattr(state, "tier_service", None)
        if tier is None or not tier.is_nvme or tier.catalog is None:
            raise RuntimeError("vLLM physical plan requires an NVMe tier catalog")
        runtime = state.hook.runtime
        work_count = schedule.work_count
        if work_count <= 0:
            raise RuntimeError("vLLM physical schedule is empty")
        max_dependencies = int(runtime.config.max_dependencies_per_work_ticket)
        plan = self._ensure_work_plan(
            runtime, work_count, max(1, max_dependencies * work_count)
        )
        if batch.exact_demand is None:
            raise RuntimeError("vLLM physical plan has no exact demand")
        if batch.exact_demand.unit_bytes != destination.block_bytes:
            raise RuntimeError(
                "vLLM physical payload bytes do not match the registered KV block"
            )
        storage_key_tables = state.storage_keys_for(batch)

        stream = torch.cuda.current_stream()
        work_bindings: list[tuple[tuple[str, int], ...]] = []
        for request_index, kv_tile in zip(
            schedule.request_indices, schedule.kv_tile_indices, strict=True
        ):
            bindings = self._external_page_bindings(
                batch,
                schedule,
                int(request_index),
                int(kv_tile),
                page_size,
                storage_key_tables,
            )
            work_bindings.append(bindings)
        layout = _physical_transfer_layout(
            tier.catalog,
            catalog_layer=destination.catalog_layer,
            work_bindings=tuple(work_bindings),
            row_bytes=destination.block_bytes,
            max_transfer_bytes=tier.nvme_max_transfer_bytes,
        )
        runs = layout.runs
        if len(runs) > runtime.config.object_capacity:
            raise RuntimeError("vLLM physical layer exceeds runtime object capacity")
        owner = getattr(state, "execution_owner", None)
        if not isinstance(owner, VllmV1WorkerController):
            raise RuntimeError("vLLM physical publication has no worker owner")
        version, prior_consumer = (
            owner.begin_external_publication(stream) if runs else (0, None)
        )

        bindings: list[RegisteredNvmeObjectBinding] = []
        for slot, run in enumerate(runs):
            object_id = 0x4E54415600000000 + slot
            destination_address = destination.address(
                run.destination_first, run.row_count
            )
            bindings.append(
                RegisteredNvmeObjectBinding(
                    slot,
                    object_id,
                    version,
                    run.source.offset,
                    run.source.bytes,
                    destination.region,
                    destination_address,
                    prior_consumer,
                )
            )
        if bindings:
            publish_registered_nvme_objects(
                tuple(bindings),
                runtime=runtime,
                stream=stream,
            )

        requirement_by_run: list[AcquireRequirement] = []
        for slot, run in enumerate(runs):
            object_id = 0x4E54415600000000 + slot
            requirement_by_run.append(
                object_requirement(
                    object_slot=slot,
                    object_id=object_id,
                    object_version=version,
                    bytes=run.source.bytes,
                )
            )
        dependencies: list[AcquireRequirement] = []
        spans: list[WorkDependencySpan] = []
        for work_id, run_slots in enumerate(layout.run_indices_by_work):
            begin = len(dependencies)
            if len(run_slots) > max_dependencies:
                raise RuntimeError(
                    "vLLM physical fragmentation exceeds "
                    "NTA_VLLM_MAX_DEPENDENCIES_PER_WORK"
                )
            if run_slots:
                dependencies.extend(requirement_by_run[slot] for slot in run_slots)
                dependency_count = len(run_slots)
                direct_count = 0
            else:
                # Native plan upload keeps a non-empty dependency array.  This
                # typed direct sentinel carries no payload and lets resident
                # work take the acquire-set direct fast path.
                dependencies.append(direct_requirement(runtime.device_view, 1))
                dependency_count = 1
                direct_count = 1
            spans.append(WorkDependencySpan(begin, dependency_count, direct_count))

        plan.upload_exact(
            topology,
            spans,
            dependencies,
            stream=stream,
        )
        state.record_evidence("physical_transfer_runs", len(runs))
        state.record_evidence("physical_transfer_blocks", layout.unique_block_count)
        state.record_evidence(
            "physical_transfer_bytes", sum(run.source.bytes for run in runs)
        )
        return plan, len(runs), bool(runs)

    def _upload_plan(
        self,
        topology: ExactWorkTopology,
        schedule: Any,
        runtime: Runtime,
    ) -> DeviceWorkPlan:
        work_count = schedule.work_count
        plan = self._ensure_work_plan(runtime, work_count, 2 * work_count)
        dependencies: list[AcquireRequirement] = []
        spans = []
        for work_id in range(work_count):
            begin = len(dependencies)
            dependencies.extend(
                (
                    direct_requirement(runtime.device_view, 1),
                    direct_requirement(runtime.device_view, 1),
                )
            )
            spans.append(WorkDependencySpan(begin, 2, 2))
        plan.upload_exact(
            topology,
            spans,
            dependencies,
            stream=torch.cuda.current_stream(),
        )
        return plan

    def _upload_host_plan(
        self,
        state: Any,
        batch: EngineBatch,
        schedule: Any,
        topology: ExactWorkTopology,
        layer: Any,
        kv_cache: torch.Tensor,
        page_size: int,
    ) -> tuple[DeviceWorkPlan, int, bool]:
        """Bind one packed CPU layer slice to exact framework HBM blocks."""

        runtime = state.hook.runtime
        owner = getattr(state, "execution_owner", None)
        if not isinstance(owner, VllmV1WorkerController):
            raise RuntimeError("vLLM host publication has no worker owner")
        prepared = prepare_host_layer(
            state=state,
            batch=batch,
            schedule=schedule,
            layer=layer,
            kv_cache=kv_cache,
            page_size=page_size,
            object_capacity=int(runtime.config.object_capacity),
            physical_pages=self._physical_pages,
        )
        layout = prepared.layout
        layer_name = prepared.layer_name
        resource = prepared.resource
        source = prepared.source
        source_indices = prepared.source_indices
        destination_indices = prepared.destination_indices
        work_count = schedule.work_count
        max_dependencies = int(runtime.config.max_dependencies_per_work_ticket)
        plan = self._ensure_work_plan(
            runtime, work_count, max(1, max_dependencies * work_count)
        )
        stream = torch.cuda.current_stream()
        version, prior_consumer = (
            owner.begin_external_publication(stream) if layout.runs else (0, None)
        )
        if layout.runs:
            assert source_indices is not None and destination_indices is not None
            indexed_plan = IndexedAcquisitionPlan(
                prepared.acquisition_topology(),
                (
                    IndexedTensorLane(
                        int(source.data_ptr()) + resource.source_offset_bytes,
                        int(kv_cache.data_ptr()),
                        resource.row_bytes,
                        resource.source_stride_bytes,
                        resource.destination_stride_bytes,
                        resource.source_rows,
                        resource.destination_rows,
                    ),
                ),
                work_bindings=tuple(
                    batch.bindings[request_index]
                    for request_index in schedule.request_indices
                ),
                source_indices_device_address=int(source_indices.data_ptr()),
                staging_indices_device_address=int(destination_indices.data_ptr()),
                object_version=version,
                direct_base=runtime.device_view,
                object_id_base=0x4E54414800000000,
            )
            if owner.tenant_isolation_enabled:
                indexed_plan.require_single_tenant_groups()
            if any(
                span.count > max_dependencies for span in indexed_plan.dependency_spans
            ):
                raise RuntimeError(
                    "vLLM host fragmentation exceeds NTA_VLLM_MAX_DEPENDENCIES_PER_WORK"
                )
            plan.upload_exact(
                topology,
                indexed_plan.dependency_spans,
                indexed_plan.dependencies,
                stream=stream,
            )
            runtime.register_indexed_acquisition_plan(
                indexed_plan,
                stream=stream,
                quiescence_event=prior_consumer,
                index_binding=IndexedHostIndexBinding(
                    int(source_indices.data_ptr()),
                    int(destination_indices.data_ptr()),
                    int(source_indices.numel()),
                ),
            )
            object_count = indexed_plan.object_count
        else:
            dependencies = [
                direct_requirement(runtime.device_view, 1) for _ in range(work_count)
            ]
            plan.upload_exact(
                topology,
                tuple(
                    WorkDependencySpan(work_id, 1, 1) for work_id in range(work_count)
                ),
                dependencies,
                stream=torch.cuda.current_stream(),
            )
            object_count = 0
        if object_count:
            if self._nta_program is None:
                raise RuntimeError("vLLM host attention has no phase program")
            self._nta_program.validate_indexed_host_range(
                runtime, 0, object_count, stream
            )
        state.record_host_destinations(layer_name, layout.destination_indices)
        state.record_evidence("host_transfer_runs", object_count)
        state.record_evidence("host_transfer_blocks", prepared.transfer_blocks)
        state.record_evidence("host_transfer_bytes", prepared.transfer_bytes)
        return plan, object_count, bool(object_count)

    def _prepare_native_schedule(
        self, state: Any, batch: EngineBatch, schedule: Any, layer: Any
    ) -> VllmScheduleContext:
        """Validate identity and build one immutable numerical topology."""

        if self._nta_program is None:
            raise RuntimeError("vLLM NTA attention has no validated phase program")
        if batch.exact_demand is None:
            raise RuntimeError("vLLM NTA attention has no exact engine batch")
        physical = self._serving_tier in {"nvme", "cxl_dax"}
        host_staged = self._serving_tier == "host_staged"
        owner = getattr(state, "execution_owner", None)
        destination = None
        if physical:
            if not isinstance(owner, VllmV1WorkerController):
                raise RuntimeError("vLLM physical attention has no worker owner")
            destination = owner.physical_destination(layer)
        semantic_layer = (
            destination.catalog_layer
            if destination is not None
            else owner.semantic_layer(layer)
            if isinstance(owner, VllmV1WorkerController)
            else 0
        )
        topology_started = time.perf_counter_ns() if self._profile_cpu else 0
        topology = self._build_topology(batch, schedule)
        state.record_evidence("work_topology_builds")
        state.record_evidence("work_topology_items", topology.work_count)
        if self._profile_cpu:
            state.record_evidence(
                "work_topology_cpu_ns", time.perf_counter_ns() - topology_started
            )
        verifier = None
        if self._verify_semantics:
            semantic_started = time.perf_counter_ns() if self._profile_cpu else 0
            verifier = ExecutionSession.from_plan(
                self._build_plan(batch, schedule, layer=semantic_layer)
            )
            state.record_evidence("semantic_plan_builds")
            state.record_evidence("semantic_verifier_sessions")
            if self._profile_cpu:
                state.record_evidence(
                    "semantic_plan_cpu_ns", time.perf_counter_ns() - semantic_started
                )
        return VllmScheduleContext(
            physical,
            host_staged,
            owner,
            destination,
            semantic_layer,
            topology,
            verifier,
        )

    def _upload_native_schedule(
        self,
        context: VllmScheduleContext,
        state: Any,
        batch: EngineBatch,
        schedule: Any,
        layer: Any,
        kv_cache: torch.Tensor,
    ) -> VllmSchedulePublication:
        if context.physical:
            tier = getattr(state, "tier_service", None)
            if tier is None or tier.tier.value != self._serving_tier:
                raise RuntimeError(
                    "vLLM forward tier does not match the worker resource owner"
                )
            destination = context.destination
            if destination is None:
                raise RuntimeError("vLLM physical schedule has no destination")
            plan, object_count, has_external_transfer = self._upload_physical_plan(
                state,
                batch,
                schedule,
                context.topology,
                destination,
                int(state.page_size),
            )
            return VllmSchedulePublication(
                plan, object_count, has_external_transfer, tier
            )
        if context.host_staged:
            plan, object_count, has_external_transfer = self._upload_host_plan(
                state,
                batch,
                schedule,
                context.topology,
                layer,
                kv_cache,
                int(state.page_size),
            )
            return VllmSchedulePublication(plan, object_count, has_external_transfer)
        return VllmSchedulePublication(
            self._upload_plan(context.topology, schedule, state.hook.runtime),
            0,
            False,
        )

    def _submit_native_schedule(
        self,
        context: VllmScheduleContext,
        publication: VllmSchedulePublication,
        state: Any,
        schedule: Any,
        wrapper: Any,
        query: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        """Submit transport and numerical work under one consumption fence."""

        plan = publication.plan
        stream = torch.cuda.current_stream()
        try:
            if context.physical and publication.has_external_transfer:
                tier = publication.tier
                if tier is None:
                    raise RuntimeError("vLLM physical schedule lost its tier owner")
                epoch = FlashInferLayerEpoch(
                    state.hook.runtime,
                    plan,
                    self._nta_program,
                    object_count=publication.object_count,
                    max_progress_rounds=tier.config.progress_rounds,
                    wait_for_plan=False,
                )
                progress_rounds = epoch.enqueue_nvme(
                    wrapper,
                    query,
                    kv_cache,
                    output,
                    issue_budget=tier.config.issue_budget,
                    completion_budget=tier.config.completion_budget,
                    timeout_ns=tier.config.progress_timeout_ns,
                    sm_scale=self.scale,
                    stream=stream,
                )
                if self._nta_config.verify_transfer:
                    epoch.check(progress_rounds, stream)
            elif context.physical:
                enqueue_resident_attention(
                    state.hook.runtime,
                    plan,
                    wrapper,
                    query,
                    kv_cache,
                    output,
                    sm_scale=self.scale,
                )
            elif context.host_staged and publication.has_external_transfer:
                epoch = FlashInferLayerEpoch(
                    state.hook.runtime,
                    plan,
                    self._nta_program,
                    object_count=publication.object_count,
                    max_progress_rounds=1,
                    wait_for_plan=False,
                )
                passes = epoch.enqueue_host(
                    wrapper,
                    query,
                    kv_cache,
                    output,
                    progress_blocks=publication.object_count,
                    sm_scale=self.scale,
                    stream=stream,
                    indexed_host_first_object=0,
                    indexed_host_range_prevalidated=True,
                    indexed_host_copy_blocks_per_group=(
                        self._nta_config.host_copy_blocks_per_group
                    ),
                )
                if self._nta_config.verify_transfer:
                    epoch.check(passes, stream)
            elif context.host_staged:
                enqueue_resident_attention(
                    state.hook.runtime,
                    plan,
                    wrapper,
                    query,
                    kv_cache,
                    output,
                    sm_scale=self.scale,
                )
            else:
                wrapper.run(
                    query,
                    kv_cache,
                    state.hook.runtime.device_view_tensor,
                    plan.work_items_tensor,
                    plan.dependencies_tensor,
                    self.scale,
                    schedule.work_count,
                    PREACQUIRED_LAUNCH_FLAGS,
                    out=output,
                )
        finally:
            try:
                plan.mark_consumed(stream)
            finally:
                if publication.has_external_transfer:
                    if not isinstance(context.owner, VllmV1WorkerController):
                        raise RuntimeError(
                            "vLLM external attention has no worker owner"
                        )
                    context.owner.record_external_consumer(stream)

    def _record_native_schedule(
        self,
        context: VllmScheduleContext,
        state: Any,
        batch: EngineBatch,
        schedule: Any,
        wrapper: Any,
        stock_wrapper: Any,
        query: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        output: torch.Tensor,
        *,
        kind: str,
    ) -> None:
        """Run opt-in verification and stage committed-path evidence."""

        self._compare_stock(
            stock_wrapper,
            query,
            kv_cache,
            output,
            custom_wrapper=wrapper,
            schedule=schedule,
            kind=kind,
            diagnostic_context={
                "runtime_work_capacity": int(
                    state.hook.runtime.config.work_ticket_capacity
                ),
                "bindings": tuple(
                    (
                        int(binding.request_index),
                        int(binding.request_slot),
                        int(binding.generation),
                    )
                    for binding in batch.bindings
                ),
                "units": tuple(
                    (
                        int(request.work_begin + contributor_index),
                        int(request.request_index),
                        int(contributor_index),
                        int(request.work_count),
                    )
                    for request in context.topology.requests
                    for contributor_index in range(request.work_count)
                )[:32],
            },
        )
        if context.verifier is not None:
            context.verifier.record_layer_completion(context.semantic_layer)
        state.record_native_launch(
            kind,
            schedule.work_count,
            form="incremental",
            framework_owned=False,
            serving_tier=self._serving_tier,
        )

    def _run_native_schedule(
        self,
        state: Any,
        batch: EngineBatch,
        schedule: Any,
        wrapper: Any,
        stock_wrapper: Any,
        layer: Any,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        *,
        kind: str,
    ) -> torch.Tensor:
        """Execute one exact FlashInfer schedule for either attention phase."""

        context = self._prepare_native_schedule(state, batch, schedule, layer)
        publication = self._upload_native_schedule(
            context, state, batch, schedule, layer, kv_cache
        )
        kv_cache_for_flashinfer = self._kv_cache_tuple(kv_cache)
        self._submit_native_schedule(
            context,
            publication,
            state,
            schedule,
            wrapper,
            query,
            kv_cache_for_flashinfer,
            output,
        )
        self._record_native_schedule(
            context,
            state,
            batch,
            schedule,
            wrapper,
            stock_wrapper,
            query,
            kv_cache_for_flashinfer,
            output,
            kind=kind,
        )
        return output

    def _request_bound_phase(
        self,
        *,
        owner: Any,
        kind: str,
        batch: EngineBatch,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        stock_wrapper: Any,
        page_size: int,
        causal: bool,
        plan: Callable[[Any], None],
    ) -> Any:
        if not isinstance(owner, VllmV1WorkerController):
            raise RuntimeError("vLLM request-bound phase has no worker owner")
        workspace_bytes = self._nta_config.workspace_bytes
        module_name = _REQUEST_BOUND_MODULES.get(query.dtype, "unsupported")
        key = self._phase_signature(
            kind,
            module_name,
            query,
            kv_cache,
            page_size=page_size,
            causal=causal,
            workspace_bytes=workspace_bytes,
        )
        if self._nta_config.compare_stock:
            workspace = None
            workspace_contract = AttentionWorkspaceContract.worker_owned(
                workspace_bytes
            )
        else:
            workspace, workspace_contract = self._framework_workspace(
                stock_wrapper, query, workspace_bytes
            )

        def build_wrapper() -> Any:
            phase_workspace = workspace
            if phase_workspace is None:
                phase_workspace = torch.empty(
                    workspace_bytes, dtype=torch.uint8, device=query.device
                )
            return self._request_bound_wrapper(kind, query, kv_cache, phase_workspace)

        wrapper, _ = owner.attention_phase(
            "request_bound",
            key,
            batch.epoch,
            build_wrapper,
            plan,
            workspace=workspace_contract,
        )
        return wrapper

    def _incremental_phase(
        self,
        *,
        owner: Any,
        kind: str,
        batch: EngineBatch,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        stock_wrapper: Any,
        page_size: int,
        causal: bool,
        plan: Callable[[Any], Any],
    ) -> tuple[Any, Any]:
        workspace_bytes = self._nta_config.workspace_bytes
        module_name = self._incremental_module_name(kind, query, kv_cache)
        if isinstance(owner, VllmV1WorkerController):
            key = self._phase_signature(
                kind,
                module_name,
                query,
                kv_cache,
                page_size=page_size,
                causal=causal,
                workspace_bytes=workspace_bytes,
            )
            if self._nta_config.compare_stock:
                workspace = None
                workspace_contract = AttentionWorkspaceContract.worker_owned(
                    workspace_bytes
                )
            else:
                candidate, capacity = self._framework_workspace_storage(
                    stock_wrapper, query
                )
                if capacity >= workspace_bytes:
                    workspace = candidate
                    workspace_contract = AttentionWorkspaceContract.framework_owned(
                        capacity, candidate.data_ptr()
                    )
                else:
                    # Some unit-sized framework profiles intentionally reserve
                    # less memory than the deployed NTA module contract. Only
                    # those profiles need an independent worker allocation.
                    workspace = None
                    workspace_contract = AttentionWorkspaceContract.worker_owned(
                        workspace_bytes
                    )

            def build_wrapper() -> Any:
                phase_workspace = workspace
                if phase_workspace is None:
                    phase_workspace = torch.empty(
                        workspace_bytes, dtype=torch.uint8, device=query.device
                    )
                return self._build_incremental_wrapper(
                    kind, query, kv_cache, module_name, phase_workspace
                )

            wrapper, schedule = owner.attention_phase(
                "incremental",
                key,
                batch.epoch,
                build_wrapper,
                plan,
                workspace=workspace_contract,
            )
            self._nta_program = wrapper._nta_transport_program
            return wrapper, schedule
        wrapper = self._ensure_local_incremental_wrapper(kind, query, kv_cache)
        return wrapper, plan(wrapper)

    def _plan_prefill_wrapper(
        self,
        wrapper: Any,
        buffers: VllmPrefillBuffers,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        page_size: int,
        causal: bool,
        *,
        extract_schedule: bool,
    ) -> Any | None:
        wrapper.plan(
            buffers.qo_indptr,
            buffers.indptr,
            buffers.indices,
            buffers.last_page_len,
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            page_size,
            q_data_type=query.dtype,
            kv_data_type=kv_cache.dtype,
            sm_scale=self.scale,
            causal=causal,
            window_left=self.window_left,
            logits_soft_cap=self.logits_soft_cap or 0.0,
            # Preserve FlashInfer's canonical planner choice. NTA binds
            # readiness to emitted work, not to a different schedule.
            disable_split_kv=False,
        )
        if not extract_schedule:
            return None
        schedule = paged_prefill_schedule(wrapper)
        if schedule.work_count <= 0:
            raise RuntimeError("vLLM FlashInfer prefill produced no work units")
        return schedule

    def _plan_decode_wrapper(
        self,
        wrapper: Any,
        buffers: VllmDecodeBuffers,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        page_size: int,
        *,
        extract_schedule: bool,
    ) -> Any | None:
        wrapper.plan(
            buffers.indptr,
            buffers.indices,
            buffers.last_page_len,
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            page_size,
            q_data_type=query.dtype,
            kv_data_type=kv_cache.dtype,
            sm_scale=self.scale,
            window_left=self.window_left,
            logits_soft_cap=self.logits_soft_cap or 0.0,
            disable_split_kv=False,
        )
        if not extract_schedule:
            return None
        schedule = decode_schedule(wrapper)
        if schedule.work_count <= 0:
            raise RuntimeError("vLLM FlashInfer decode produced no work units")
        return schedule

    def _native_prefill_forward(
        self,
        layer: Any,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        state = current_vllm_v1_forward_state()
        if state is None or state.batch is None or state.hook is None:
            raise RuntimeError("vLLM NTA prefill ran without a worker sidecar")
        if attn_metadata.use_cascade:
            raise RuntimeError("native vLLM prefill does not support cascade attention")
        prefill_metadata = attn_metadata.prefill
        if not isinstance(prefill_metadata, FIPrefill):
            raise RuntimeError(
                "native vLLM prefill requires the FlashInfer FA2 prefill metadata"
            )
        stock_wrapper = prefill_metadata.wrapper
        batch = state.batch
        if not isinstance(batch, EngineBatch):
            raise RuntimeError("vLLM prefill has no engine batch")
        expected_rows = attn_metadata.num_decodes + attn_metadata.num_prefills
        if len(batch.bindings) < expected_rows or batch.exact_demand is None:
            raise RuntimeError(
                "native vLLM prefill has fewer exact rows than the scheduled batch"
            )
        prefill_batch = state.phase_batch(
            attn_metadata.num_decodes, attn_metadata.num_prefills
        )
        # The metadata builder already validated and planned this typed wrapper
        # once for the forward.  Do not repeat its paged-buffer validation in
        # every transformer layer; the request-bound launch needs only the
        # immutable phase binding and the framework-owned plan.
        if getattr(stock_wrapper, "_nta_request_bound", False):
            return self._run_request_bound(
                state,
                prefill_batch,
                stock_wrapper,
                stock_wrapper,
                query,
                kv_cache,
                output,
                kind="prefill",
                framework_owned=True,
                phase_start=attn_metadata.num_decodes,
            )
        buffers = require_prefill_buffers(stock_wrapper, attn_metadata.num_prefills)
        page_size = int(getattr(state, "page_size", 0) or 0)
        if page_size <= 0:
            raise RuntimeError("vLLM forward sidecar has no token page size")
        owner = getattr(state, "execution_owner", None)
        if self._request_bound_available(state, owner):
            direct_wrapper = self._request_bound_phase(
                owner=owner,
                kind="prefill",
                batch=prefill_batch,
                query=query,
                kv_cache=kv_cache,
                stock_wrapper=stock_wrapper,
                page_size=page_size,
                causal=attn_metadata.causal,
                plan=lambda wrapper: self._plan_prefill_wrapper(
                    wrapper,
                    buffers,
                    query,
                    kv_cache,
                    page_size,
                    attn_metadata.causal,
                    extract_schedule=False,
                ),
            )
            return self._run_request_bound(
                state,
                prefill_batch,
                direct_wrapper,
                stock_wrapper,
                query,
                kv_cache,
                output,
                kind="prefill",
                framework_owned=False,
                phase_start=attn_metadata.num_decodes,
            )

        wrapper, schedule = self._incremental_phase(
            owner=owner,
            kind="prefill",
            batch=prefill_batch,
            query=query,
            kv_cache=kv_cache,
            stock_wrapper=stock_wrapper,
            page_size=page_size,
            causal=attn_metadata.causal,
            plan=lambda wrapper: self._plan_prefill_wrapper(
                wrapper,
                buffers,
                query,
                kv_cache,
                page_size,
                attn_metadata.causal,
                extract_schedule=True,
            ),
        )
        return self._run_native_schedule(
            state,
            prefill_batch,
            schedule,
            wrapper,
            stock_wrapper,
            layer,
            query,
            kv_cache,
            output,
            kind="prefill",
        )

    def _native_forward(
        self,
        layer: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        state = current_vllm_v1_forward_state()
        if state is None or state.batch is None or state.hook is None:
            raise RuntimeError("vLLM NTA attention ran without a worker sidecar")
        batch = state.batch
        if not isinstance(batch, EngineBatch):
            raise RuntimeError("vLLM decode has no engine batch")
        if len(batch.bindings) < attn_metadata.num_decodes:
            raise RuntimeError(
                "native vLLM decode has fewer exact rows than the scheduled batch"
            )
        if attn_metadata.use_cascade:
            raise RuntimeError("native vLLM decode does not support cascade attention")
        if getattr(attn_metadata, "decode_use_trtllm", False):
            raise RuntimeError(
                "native vLLM path requires FlashInfer FA2 decode, not TRTLLM"
            )
        if attn_metadata.num_decode_tokens != attn_metadata.num_decodes:
            raise RuntimeError("native vLLM path requires one query token per request")
        if batch.exact_demand is None or len(
            batch.exact_demand.request_unit_ids
        ) != len(batch.bindings):
            raise RuntimeError("native vLLM exact-demand rows are misaligned")
        decode_batch = state.phase_batch(0, attn_metadata.num_decodes)

        decode_metadata = getattr(attn_metadata, "decode", None)
        stock_wrapper = getattr(decode_metadata, "wrapper", None)
        if stock_wrapper is None:
            raise RuntimeError(
                "native vLLM decode requires the planned FlashInfer decode wrapper"
            )
        if getattr(stock_wrapper, "_nta_request_bound", False):
            return self._run_request_bound(
                state,
                decode_batch,
                stock_wrapper,
                stock_wrapper,
                query,
                kv_cache,
                output,
                kind="decode",
                framework_owned=True,
                phase_start=0,
            )
        buffers = require_decode_buffers(stock_wrapper, attn_metadata.num_decodes)
        # The helper validates host-known tensor lengths without reading
        # ``indptr[-1]`` back to Python, avoiding a per-layer synchronization.
        page_size = int(getattr(state, "page_size", 0) or 0)
        if page_size <= 0:
            raise RuntimeError("vLLM forward sidecar has no token page size")

        owner = getattr(state, "execution_owner", None)
        if self._request_bound_available(state, owner):
            direct_wrapper = self._request_bound_phase(
                owner=owner,
                kind="decode",
                batch=decode_batch,
                query=query,
                kv_cache=kv_cache,
                stock_wrapper=stock_wrapper,
                page_size=page_size,
                causal=False,
                plan=lambda wrapper: self._plan_decode_wrapper(
                    wrapper,
                    buffers,
                    query,
                    kv_cache,
                    page_size,
                    extract_schedule=False,
                ),
            )
            return self._run_request_bound(
                state,
                decode_batch,
                direct_wrapper,
                stock_wrapper,
                query,
                kv_cache,
                output,
                kind="decode",
                framework_owned=False,
                phase_start=0,
            )

        wrapper, schedule = self._incremental_phase(
            owner=owner,
            kind="decode",
            batch=decode_batch,
            query=query,
            kv_cache=kv_cache,
            stock_wrapper=stock_wrapper,
            page_size=page_size,
            causal=False,
            plan=lambda wrapper: self._plan_decode_wrapper(
                wrapper,
                buffers,
                query,
                kv_cache,
                page_size,
                extract_schedule=True,
            ),
        )
        return self._run_native_schedule(
            state,
            decode_batch,
            schedule,
            wrapper,
            stock_wrapper,
            layer,
            query,
            kv_cache,
            output,
            kind="decode",
        )

    def forward(
        self,
        layer: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output is None:
            raise RuntimeError("NTA vLLM attention requires an output buffer")
        original_query = query
        original_key = key
        original_value = value
        original_output = output
        state = current_vllm_v1_forward_state()
        if state is not None and state.reference_warmup:
            VLLM_STATS["reference_warmup_launches"] += 1
            return super().forward(
                layer,
                original_query,
                original_key,
                original_value,
                kv_cache,
                attn_metadata,
                output=original_output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )
        if attn_metadata is None or not self._native_enabled:
            if state is None:
                VLLM_STATS["reference_attention_launches"] += 1
            else:
                state.record_evidence("reference_attention_launches")
            return super().forward(
                layer,
                original_query,
                original_key,
                original_value,
                kv_cache,
                attn_metadata,
                output=original_output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )
        if state is None or not state.connector_validated:
            raise RuntimeError(
                "vLLM NTA attention requires validated KVConnector lifecycle metadata"
            )
        num_actual_tokens = attn_metadata.num_actual_tokens
        host_layer_name: str | None = None
        if self._serving_tier == "host_staged":
            value = getattr(layer, "layer_name", None)
            if not isinstance(value, str) or not value:
                raise RuntimeError("vLLM host-staged attention has no layer identity")
            host_layer_name = value
            state.begin_host_layer(host_layer_name)
        try:
            if attn_metadata.num_prefill_tokens:
                self._native_prefill_forward(
                    layer,
                    query[attn_metadata.num_decode_tokens : num_actual_tokens],
                    kv_cache,
                    attn_metadata,
                    output[attn_metadata.num_decode_tokens : num_actual_tokens],
                )
            if attn_metadata.num_decode_tokens:
                if attn_metadata.num_decode_tokens != attn_metadata.num_decodes:
                    raise RuntimeError(
                        "native vLLM decode requires one query token per request"
                    )
                self._native_forward(
                    layer,
                    query[: attn_metadata.num_decode_tokens],
                    key[: attn_metadata.num_decode_tokens],
                    value[: attn_metadata.num_decode_tokens],
                    kv_cache,
                    attn_metadata,
                    output[: attn_metadata.num_decode_tokens],
                )
            if host_layer_name is not None:
                state.finish_host_layer(host_layer_name)
            return original_output
        except RuntimeError:
            if host_layer_name is not None:
                state.abort_host_layer(host_layer_name)
            if self._serving_tier != "hbm" or not self._nta_config.allow_stock_fallback:
                raise
            state.record_evidence("reference_fallback_launches")
            return super().forward(
                layer,
                original_query,
                original_key,
                original_value,
                kv_cache,
                attn_metadata,
                output=original_output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )


def consumer_contract() -> dict[str, Any]:
    """Return process-local evidence for artifact collectors."""
    if VLLM_STATS["reference_fallback_launches"]:
        # A process that executed even one reference fallback did not provide
        # an all-native numerical boundary.  Report the conservative contract;
        # formal native validators additionally require this counter to be zero.
        return ConsumerContract.framework_reference(
            engine="vllm",
            backend="flashinfer_fallback",
            engine_version=SUPPORTED_VLLM_VERSION,
        ).as_dict()
    native = (
        VLLM_STATS["native_decode_launches"] + VLLM_STATS["native_prefill_launches"]
    )
    if native:
        contract = ConsumerContract.native_work_unit(
            engine="vllm",
            backend="nta_flashinfer",
            engine_version=SUPPORTED_VLLM_VERSION,
        ).as_dict()
        contract.update(
            {
                "native_launches": native,
                "serving_tier": os.environ.get("NTA_SERVING_TIER", "hbm"),
                "resident_only": (
                    VLLM_STATS["physical_decode_launches"]
                    + VLLM_STATS["physical_prefill_launches"]
                    + VLLM_STATS["host_decode_launches"]
                    + VLLM_STATS["host_prefill_launches"]
                    == 0
                ),
                "physical_decode_launches": VLLM_STATS["physical_decode_launches"],
                "physical_prefill_launches": VLLM_STATS["physical_prefill_launches"],
                "host_decode_launches": VLLM_STATS["host_decode_launches"],
                "host_prefill_launches": VLLM_STATS["host_prefill_launches"],
                "host_transfer_blocks": VLLM_STATS["host_transfer_blocks"],
                "host_transfer_bytes": VLLM_STATS["host_transfer_bytes"],
            }
        )
        return contract
    if VLLM_STATS["reference_attention_launches"]:
        return ConsumerContract.framework_reference(
            engine="vllm",
            backend="nta_flashinfer",
            engine_version=SUPPORTED_VLLM_VERSION,
        ).as_dict()
    return ConsumerContract.projection_only(
        engine="vllm",
        backend="nta_flashinfer",
        engine_version=SUPPORTED_VLLM_VERSION,
    ).as_dict()


@atexit.register
def _publish_vllm_evidence() -> None:
    """Publish worker-local evidence for a parent-process serving harness."""
    if not any(VLLM_STATS.values()):
        return
    workspace = os.environ.get("FLASHINFER_WORKSPACE_BASE")
    if not workspace:
        return
    token = os.environ.get("NTA_VLLM_EVIDENCE_TOKEN", "default")
    path = pathlib.Path(workspace) / f"nta-vllm-engine.{token}.{os.getpid()}.json"
    report = {
        "schema": 1,
        "engine": "vllm",
        "backend": "nta_flashinfer",
        "engine_version": SUPPORTED_VLLM_VERSION,
        "consumer_contract": consumer_contract(),
        "stats": dict(VLLM_STATS),
        "native_enabled": os.environ.get("NTA_VLLM_NATIVE", "0") == "1",
        "stock_fallback_enabled": os.environ.get("NTA_VLLM_ALLOW_STOCK_FALLBACK", "0")
        == "1",
        "serving_tier": os.environ.get("NTA_SERVING_TIER", "hbm"),
        "physical_decode_launches": VLLM_STATS["physical_decode_launches"],
        "physical_prefill_launches": VLLM_STATS["physical_prefill_launches"],
        "host_decode_launches": VLLM_STATS["host_decode_launches"],
        "host_prefill_launches": VLLM_STATS["host_prefill_launches"],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    except OSError:
        # Evidence publication cannot change inference correctness during
        # interpreter teardown. The serving harness treats missing evidence as
        # a failed native-verification gate.
        return
