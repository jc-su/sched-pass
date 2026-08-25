"""vLLM 0.26 V1 consumer for instrumented FlashInfer decode attention.

The worker controller is the only framework-private bridge in this module.
It publishes an engine-neutral :class:`EngineBatch` after vLLM updates its
persistent input batch.  ``NtaVllmFlashInferImpl`` then consumes that batch
through a real vLLM ``AttentionImpl`` call and submits the same typed NTA
work-plan ABI used by the SGLang adapter.

This first consumer is deliberately narrow: resident CUDA KV, one KV group,
non-TRTLLM single-token decode, and no CUDA graph capture.  Unsupported
features use vLLM's reference implementation only when explicitly enabled;
the native path otherwise fails closed so an artifact cannot silently claim
NTA execution for a stock launch.
"""

from __future__ import annotations

import atexit
from collections import Counter
import contextlib
import os
import pathlib
import threading
from typing import Any
import weakref

import torch
from flashinfer import BatchDecodeWithPagedKVCacheWrapper
from vllm import envs
from vllm.v1.attention.backends.flashinfer import (
    FlashInferBackend,
    FlashInferMetadataBuilder,
    FlashInferImpl,
    FlashInferMetadata,
)
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.utils import get_kv_cache_layout

from nta_runtime.adapters.base import EngineBatch
from nta_runtime.adapters.vllm_v1 import (
    VllmV1Hook,
    current_vllm_v1_forward_state,
)
from nta_runtime.execution_core import ExecutionSession, ExecutionTile
from nta_runtime.execution_protocol import ExecutionProtocolConfig
from nta_runtime.flashinfer import (
    BIND_CURRENT_GENERATION,
    PREACQUIRED,
    attention_jit_args,
    direct_requirement,
    request_ranges_for_schedule,
)
from nta_runtime.flashinfer_schedule import decode_schedule
from nta_runtime.tenant import tenant_budget_specs, tenant_mapper_from_environment
from nta_runtime.runtime import (
    AcquireRequirement,
    DeviceWorkPlan,
    JitPhaseProgram,
    OperatorCapability,
    OperatorAccessProof,
    OperatorDemandBinding,
    OperatorFamily,
    OperatorForm,
    OperatorIdentityBinding,
    OperatorInstrumentation,
    OperatorPartialState,
    OperatorPlanFlag,
    OperatorCoordinateMap,
    OperatorReduction,
    Runtime,
    RuntimeConfig,
)
from nta_runtime.work_unit import Granularity


SUPPORTED_VLLM_VERSION = "0.26.0"
_DEFAULT_MODULES = {
    torch.float16: "nta_batch_prefill_default_v2_hooked",
    torch.bfloat16: "nta_batch_prefill_default_v2_hooked_bf16",
}
_MODULE_LOCK = threading.Lock()
_PHASE_PROGRAMS: dict[pathlib.Path, JitPhaseProgram] = {}
VLLM_STATS: Counter[str] = Counter()


def _positive_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def _find_module(name: str) -> pathlib.Path:
    workspace = os.environ.get("FLASHINFER_WORKSPACE_BASE")
    if not workspace:
        raise RuntimeError(
            "NTA vLLM native attention requires FLASHINFER_WORKSPACE_BASE; "
            "run tools/jit/activate.py --flashinfer-hook first"
        )
    matches = tuple(pathlib.Path(workspace).rglob(f"{name}.so"))
    if len(matches) != 1:
        raise RuntimeError(
            f"NTA vLLM native attention expected one {name}.so in {workspace}, "
            f"found {len(matches)}"
        )
    return matches[0].resolve()


def _phase_program(path: pathlib.Path) -> JitPhaseProgram:
    with _MODULE_LOCK:
        existing = _PHASE_PROGRAMS.get(path)
        if existing is not None:
            return existing
        program = JitPhaseProgram(path)
        family = (
            OperatorFamily.FLASHINFER_PAGED_PREFILL
            if "prefill" in path.name
            else OperatorFamily.FLASHINFER_DECODE
        )
        program.operator_contract.require(
            family=family,
            form=OperatorForm.INCREMENTAL,
            capabilities=(
                OperatorCapability.REQUEST_BINDING
                | OperatorCapability.OBJECT_DEPENDENCIES
                | OperatorCapability.FINITE_DEFERRAL
                | OperatorCapability.PARTIAL_PUBLICATION
                | OperatorCapability.COMPLETE_CONTRIBUTOR_MERGE
                | OperatorCapability.RUNNABLE_COMPACTION
                | OperatorCapability.TYPED_FLASHINFER_FRONTEND
            ),
            instrumentation=(
                OperatorInstrumentation.TYPED_ACCESS_LOWERING
                | OperatorInstrumentation.EXACT_DEMAND
                | OperatorInstrumentation.GENERATION_SAFE_IDENTITY
                | OperatorInstrumentation.TIER_OWNERSHIP
            ),
            identity_binding=OperatorIdentityBinding.REQUEST_SLOT_GENERATION,
            demand_binding=OperatorDemandBinding.EXACT_WORK_UNIT,
            access_proof=OperatorAccessProof.TYPED_FRONTEND,
            tier_mask=(1 << 6) - 1,
        )
        program.operator_plan.require(
            family=family,
            forms=(OperatorForm.DIRECT, OperatorForm.INCREMENTAL),
            coordinate_map=OperatorCoordinateMap.FLASHINFER_REQUEST_CONTIGUOUS,
            partial_state=OperatorPartialState.ONLINE_SOFTMAX_VALUE_LSE,
            reduction=OperatorReduction.ORDERED_MERGE_STATE,
            flags=(
                OperatorPlanFlag.FIXED_CAPACITY
                | OperatorPlanFlag.GRAPH_STABLE
                | OperatorPlanFlag.EXTERNAL_WAVE_SOURCES
                | OperatorPlanFlag.GENERATION_BOUND
                | OperatorPlanFlag.EXACT_COMPLETE_MERGE
            ),
        )
        _PHASE_PROGRAMS[path] = program
        VLLM_STATS["verified_operator_modules"] += 1
        return program


@atexit.register
def _close_phase_programs() -> None:
    for program in tuple(_PHASE_PROGRAMS.values()):
        with contextlib.suppress(Exception):
            program.close()
    _PHASE_PROGRAMS.clear()


def _build_runtime(runner: Any, request_capacity: int, work_capacity: int) -> Runtime:
    device = getattr(runner, "device", None)
    device_ordinal = (
        int(device.index)
        if isinstance(device, torch.device) and device.index is not None
        else int(torch.cuda.current_device())
    )
    tenant_capacity = _positive_env("NTA_TENANT_CAPACITY", max(1, request_capacity))
    return Runtime(
        RuntimeConfig(
            request_capacity=request_capacity,
            object_capacity=max(2, 2 * request_capacity),
            intent_capacity=max(2, 2 * request_capacity),
            work_ticket_capacity=work_capacity,
            max_dependencies_per_work_ticket=2,
            device_ordinal=device_ordinal,
            tenant_capacity=tenant_capacity,
        )
    )


class VllmV1WorkerController:
    """Own the runtime and identity hook for one vLLM GPU model runner."""

    def __init__(self, runner: Any) -> None:
        self._runner_ref = weakref.ref(runner)
        self._runtime: Runtime | None = None
        self._hook: VllmV1Hook | None = None
        self._page_size = 0
        self._epoch = 0

    def bind(self, scheduler_output: Any) -> EngineBatch:
        runner = self._runner_ref()
        if runner is None:
            raise RuntimeError("vLLM V1 model runner was destroyed")
        input_batch = getattr(runner, "input_batch", None)
        request_capacity = int(getattr(runner, "max_num_reqs", 0))
        if input_batch is None or request_capacity <= 0:
            raise RuntimeError("vLLM V1 runner is not initialized with InputBatch")
        groups = getattr(
            getattr(runner, "kv_cache_config", None), "kv_cache_groups", ()
        )
        if len(groups) != 1:
            raise RuntimeError(
                "NTA vLLM V1 currently requires exactly one KV cache group"
            )
        spec = groups[0].kv_cache_spec
        self._page_size = int(getattr(spec, "block_size", 0))
        if self._page_size <= 0:
            raise RuntimeError("vLLM V1 KV cache spec has no positive block_size")
        page_bytes = int(getattr(spec, "page_size_bytes", 0))
        if page_bytes <= 0:
            raise RuntimeError("vLLM V1 KV cache spec has no page_size_bytes")
        work_capacity = _positive_env(
            "NTA_VLLM_WORK_TICKET_CAPACITY",
            max(request_capacity, 4 * request_capacity),
        )
        if self._runtime is None:
            self._runtime = _build_runtime(runner, request_capacity, work_capacity)
            tenant_specs = tenant_budget_specs()
            tenant_capacity = int(self._runtime.config.tenant_capacity)
            for tenant_id, max_bytes, weight in tenant_specs:
                if tenant_id >= tenant_capacity:
                    raise RuntimeError(
                        f"tenant {tenant_id} exceeds NTA_TENANT_CAPACITY="
                        f"{tenant_capacity}"
                    )
                self._runtime.set_tenant_budget(tenant_id, max_bytes, weight)
            self._hook = VllmV1Hook(
                self._runtime,
                request_capacity,
                page_bytes=page_bytes,
                expected_vllm_version=SUPPORTED_VLLM_VERSION,
                tenant_for_request=tenant_mapper_from_environment(),
            )
        assert self._hook is not None
        batch = self._hook.bind_forward(
            scheduler_output,
            input_batch,
            epoch=self._epoch,
            stream=torch.cuda.current_stream(),
            granularity=Granularity.PAGE_GROUP,
        )
        self._epoch += 1
        return batch

    @property
    def hook(self) -> VllmV1Hook:
        if self._hook is None:
            raise RuntimeError("vLLM NTA worker controller has not bound a forward")
        return self._hook

    @property
    def page_size(self) -> int:
        if self._page_size <= 0:
            raise RuntimeError("vLLM V1 worker controller has no page size")
        return self._page_size


def _controller(runner: Any) -> VllmV1WorkerController:
    controller = getattr(runner, "_nta_vllm_controller", None)
    if controller is None:
        controller = VllmV1WorkerController(runner)
        setattr(runner, "_nta_vllm_controller", controller)
    return controller


class NtaVllmFlashInferMetadataBuilder(FlashInferMetadataBuilder):
    """Metadata builder matching stock FlashInfer but disabling unsafe graphs."""

    _cudagraph_support = AttentionCGSupport.NEVER

    @classmethod
    def get_cudagraph_support(cls, vllm_config: Any, kv_cache_spec: Any) -> AttentionCGSupport:
        return AttentionCGSupport.NEVER


class NtaVllmFlashInferBackend(FlashInferBackend):
    """vLLM registry backend whose ``impl.forward`` launches NTA work units."""

    @staticmethod
    def get_impl_cls() -> type["NtaVllmFlashInferImpl"]:
        return NtaVllmFlashInferImpl

    @staticmethod
    def get_builder_cls() -> type[NtaVllmFlashInferMetadataBuilder]:
        return NtaVllmFlashInferMetadataBuilder


class NtaVllmFlashInferImpl(FlashInferImpl):
    """Native NTA consumer for resident single-token FlashInfer decode."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._nta_wrapper: BatchDecodeWithPagedKVCacheWrapper | None = None
        self._nta_plan: DeviceWorkPlan | None = None
        self._nta_plan_capacity = 0
        self._nta_program: JitPhaseProgram | None = None
        # Resident-only vLLM requests do not exercise a remote-tier
        # dependency.  Keep the framework reference as the no-regression
        # default; the native work-unit consumer is an explicit integration
        # profile until an external KVConnector supplies the same exact
        # demand/lifetime boundary.
        self._native_enabled = os.environ.get("NTA_VLLM_NATIVE", "0") == "1"

    def _ensure_wrapper(self, query: torch.Tensor, kv_cache: torch.Tensor) -> None:
        if self._nta_wrapper is not None:
            return
        if query.dtype not in _DEFAULT_MODULES or kv_cache.dtype != query.dtype:
            raise RuntimeError(
                "native vLLM NTA attention requires matching float16 or "
                "bfloat16 query and resident KV cache dtypes"
            )
        if isinstance(self.kv_cache_dtype, str) and self.kv_cache_dtype not in {
            "auto",
            "float16",
            "bfloat16",
        }:
            raise RuntimeError(
                "native vLLM NTA attention does not support quantized KV cache "
                f"dtype {self.kv_cache_dtype!r}"
            )
        module_name = os.environ.get(
            "NTA_VLLM_DECODE_MODULE", _DEFAULT_MODULES[query.dtype]
        )
        module_path = _find_module(module_name)
        self._nta_program = _phase_program(module_path)
        jit_args = attention_jit_args(
            module_name,
            dtype_q=query.dtype,
            dtype_kv=kv_cache.dtype,
            dtype_o=query.dtype,
            idtype=torch.int32,
            head_dim_qk=self.head_size,
            head_dim_vo=self.head_size,
        )
        workspace_bytes = _positive_env(
            "NTA_VLLM_FLASHINFER_WORKSPACE_BYTES",
            int(getattr(envs, "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE", 64 * 1024 * 1024)),
        )
        workspace = torch.zeros(
            workspace_bytes, dtype=torch.uint8, device=query.device
        )
        self._nta_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace,
            get_kv_cache_layout(),
            backend="fa2",
            use_tensor_cores=True,
            jit_args=jit_args,
        )

    def _build_plan(
        self,
        state: Any,
        schedule: Any,
    ) -> ExecutionSession:
        batch = state.batch
        if not isinstance(batch, EngineBatch) or batch.exact_demand is None:
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
                    layer=0,
                    logical_begin=int(kv_tile),
                    candidate_units=candidates,
                    selected_ids=tuple(range(candidates)),
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
        return ExecutionSession.from_tiles(
            epoch=batch.epoch,
            granularity=batch.granularity,
            protocol=ExecutionProtocolConfig.late_bound(
                granularity=batch.granularity,
                max_inflight_units=max(1, len(tiles)),
            ),
            tiles=tiles,
        )

    def _upload_plan(
        self,
        execution: ExecutionSession,
        schedule: Any,
        runtime: Runtime,
        bindings: tuple[Any, ...],
    ) -> DeviceWorkPlan:
        work_count = schedule.work_count
        if work_count > self._nta_plan_capacity:
            if self._nta_plan is not None:
                torch.cuda.current_stream().synchronize()
                self._nta_plan.close()
            self._nta_plan = DeviceWorkPlan(
                work_count,
                2 * work_count,
                runtime.device_ordinal,
            )
            self._nta_plan_capacity = work_count
        assert self._nta_plan is not None
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
            spans.append((begin, 2, 2, work_id))
        ranges = request_ranges_for_schedule(
            bindings, schedule.request_indices
        )
        self._nta_plan.upload_work_units(
            execution.batch.units,
            spans,
            dependencies,
            ranges,
            epoch=execution.epoch,
            stream=torch.cuda.current_stream(),
        )
        return self._nta_plan

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
        if len(batch.bindings) != attn_metadata.num_decodes:
            raise RuntimeError(
                "native vLLM decode requires one scheduled token per request"
            )
        if attn_metadata.num_prefill_tokens or attn_metadata.use_cascade:
            raise RuntimeError("native vLLM path only supports pure decode batches")
        if getattr(attn_metadata, "decode_use_trtllm", False):
            raise RuntimeError(
                "native vLLM path requires FlashInfer FA2 decode, not TRTLLM"
            )
        if attn_metadata.num_decode_tokens != attn_metadata.num_decodes:
            raise RuntimeError("native vLLM path requires one query token per request")
        if len(batch.exact_demand.request_unit_ids) != len(batch.bindings):
            raise RuntimeError("native vLLM exact-demand rows are misaligned")

        decode_metadata = getattr(attn_metadata, "decode", None)
        stock_wrapper = getattr(decode_metadata, "wrapper", None)
        if stock_wrapper is None:
            raise RuntimeError(
                "native vLLM decode requires the planned FlashInfer decode wrapper"
            )
        indptr = getattr(stock_wrapper, "_paged_kv_indptr_buf", None)
        indices = getattr(stock_wrapper, "_paged_kv_indices_buf", None)
        last_page_len = getattr(stock_wrapper, "_paged_kv_last_page_len_buf", None)
        if not all(isinstance(tensor, torch.Tensor) for tensor in (indptr, indices, last_page_len)):
            raise RuntimeError(
                "vLLM FlashInfer metadata has no typed paged-KV device buffers"
            )
        if indptr.numel() != attn_metadata.num_decodes + 1:
            raise RuntimeError("vLLM FlashInfer page indptr has the wrong request count")
        page_count = int(indptr[-1].item())
        if page_count <= 0 or indices.numel() < page_count:
            raise RuntimeError("vLLM FlashInfer page-index buffer is incomplete")
        indptr = indptr.to(dtype=torch.int32)
        indices = indices[:page_count].to(dtype=torch.int32)
        last_page_len = last_page_len[: attn_metadata.num_decodes].to(dtype=torch.int32)
        page_size = int(getattr(state, "page_size", 0) or 0)
        if page_size <= 0:
            raise RuntimeError("vLLM forward sidecar has no token page size")

        self._ensure_wrapper(query, kv_cache)
        assert self._nta_wrapper is not None
        self._nta_wrapper.plan(
            indptr,
            indices,
            last_page_len,
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            page_size,
            q_data_type=query.dtype,
            kv_data_type=kv_cache.dtype,
            sm_scale=self.scale,
            disable_split_kv=True,
        )
        schedule = decode_schedule(self._nta_wrapper)
        execution = self._build_plan(state, schedule)
        plan = self._upload_plan(
            execution,
            schedule,
            state.hook.runtime,
            batch.bindings,
        )
        kv_cache_permute = kv_cache.permute(
            *FlashInferBackend.get_kv_cache_stride_order()
        )
        kv_cache_for_flashinfer = kv_cache_permute.split(self.head_size, dim=-1)
        self._nta_wrapper.run(
            query,
            kv_cache_for_flashinfer,
            state.hook.runtime.device_view_tensor,
            plan.work_items_tensor,
            plan.dependencies_tensor,
            self.scale,
            schedule.work_count,
            PREACQUIRED | BIND_CURRENT_GENERATION,
            out=output,
        )
        plan.mark_consumed(torch.cuda.current_stream())
        execution.record_layer_completion(0)
        state.hook.record_native_launch()
        VLLM_STATS["native_decode_launches"] += 1
        VLLM_STATS["native_decode_work_items"] += schedule.work_count
        return output

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
        if attn_metadata is None or not self._native_enabled:
            VLLM_STATS["reference_attention_launches"] += 1
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
        if self.kv_sharing_target_layer_name is None:
            torch.ops._C_cache_ops.reshape_and_cache_flash(
                key,
                value,
                kv_cache[:, 0],
                kv_cache[:, 1],
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )
        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        output = output[:num_actual_tokens]
        key = key[:num_actual_tokens]
        value = value[:num_actual_tokens]
        try:
            self._native_forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
            )
            return original_output
        except RuntimeError:
            if os.environ.get("NTA_VLLM_ALLOW_STOCK_FALLBACK") != "1":
                raise
            VLLM_STATS["reference_fallback_launches"] += 1
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
    native = VLLM_STATS["native_decode_launches"]
    if native:
        return {
            "engine": "vllm",
            "backend": "nta_flashinfer",
            "engine_version": SUPPORTED_VLLM_VERSION,
            "kind": "native_work_unit",
            "native_launches": native,
        }
    if VLLM_STATS["reference_attention_launches"]:
        return {
            "engine": "vllm",
            "backend": "nta_flashinfer",
            "engine_version": SUPPORTED_VLLM_VERSION,
            "kind": "framework_reference",
        }
    return {
        "engine": "vllm",
        "backend": "nta_flashinfer",
        "engine_version": SUPPORTED_VLLM_VERSION,
        "kind": "projection_only",
    }
