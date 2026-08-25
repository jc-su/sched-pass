"""vLLM 0.26 V1 consumer for instrumented FlashInfer attention.

The worker controller is the only framework-private bridge in this module.
It publishes an engine-neutral :class:`EngineBatch` after vLLM updates its
persistent input batch.  ``NtaVllmFlashInferImpl`` then consumes that batch
through a real vLLM ``AttentionImpl`` call and submits the same typed NTA
work-plan ABI used by the SGLang adapter.

The qualified profile is one KV group, FA2 pure prefill or single-token
decode, and no CUDA graph capture. Host-staged uses resident CUDA KV; NVMe and
CXL-DAX use the same exact work-plan/phase protocol as SGLang. Mixed
prefill/decode batches and unsupported features use vLLM's reference
implementation only when explicitly enabled; the native path otherwise fails
closed so an artifact cannot silently claim NTA execution for a stock launch.
"""

from __future__ import annotations

import atexit
from collections import Counter
import contextlib
import json
import os
import pathlib
import threading
from typing import Any
import weakref

import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)
from vllm import envs
from vllm.v1.attention.backends.flashinfer import (
    FIPrefill,
    FlashInferBackend,
    FlashInferMetadataBuilder,
    FlashInferImpl,
    FlashInferMetadata,
)
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.utils import get_kv_cache_layout

from nta_runtime.adapters.base import ConsumerContract, EngineBatch
from nta_runtime.adapters.vllm_v1 import (
    VllmV1Hook,
    current_vllm_v1_forward_state,
    validate_vllm_attention_tier,
)
from nta_runtime.execution_core import ExecutionSession, ExecutionTile
from nta_runtime.execution_protocol import ExecutionProtocolConfig
from nta_runtime.runtime_resources import (
    RuntimeResourceConfig,
    ServingRuntimeResources,
)
from nta_runtime.tier import ServingTierConfig
from nta_runtime.flashinfer import (
    BIND_CURRENT_GENERATION,
    FlashInferLayerEpoch,
    PREACQUIRED,
    attention_jit_args,
    direct_requirement,
    enqueue_resident_attention,
    request_ranges_for_schedule,
)
from nta_runtime.flashinfer_schedule import decode_schedule, paged_prefill_schedule
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
)
from nta_runtime.work_unit import Granularity


SUPPORTED_VLLM_VERSION = "0.26.0"
_DEFAULT_MODULES = {
    # FlashInfer's tensor-core decode wrapper consumes a paged-prefill JIT
    # module for its FA2 plan/run interface.
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


def _ensure_default_attention_module(
    name: str, dtype: torch.dtype, head_size: int
) -> pathlib.Path:
    """Build the pinned NTA tensor-core module during backend initialization."""
    if head_size != 128:
        raise RuntimeError(
            "native vLLM NTA attention currently requires head_size=128; "
            "provide NTA_VLLM_DECODE_MODULE for another qualified module"
        )
    workspace = os.environ.get("FLASHINFER_WORKSPACE_BASE")
    if not workspace:
        raise RuntimeError(
            "NTA vLLM native attention requires FLASHINFER_WORKSPACE_BASE; "
            "run tools/jit/activate.py --flashinfer-hook first"
        )
    matches = tuple(pathlib.Path(workspace).rglob(f"{name}.so"))
    if not matches:
        from flashinfer.jit.attention.modules import gen_customize_batch_prefill_module

        specification = gen_customize_batch_prefill_module(
            "fa2",
            name,
            dtype,
            dtype,
            dtype,
            torch.int32,
            head_size,
            head_size,
            ["nta_runtime", "nta_work_items", "nta_dependencies"],
            ["uint8_t", "uint8_t", "uint8_t"],
            ["sm_scale", "nta_work_count", "nta_skip_merge"],
            ["double", "int64_t", "int64_t"],
            "DefaultAttention<false, false, false, false>",
            "#include <flashinfer/attention/variants.cuh>",
        )
        specification.build_and_load()
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


def _build_resources(
    runner: Any, request_capacity: int, work_capacity: int
) -> ServingRuntimeResources:
    device = getattr(runner, "device", None)
    device_ordinal = (
        int(device.index)
        if isinstance(device, torch.device) and device.index is not None
        else int(torch.cuda.current_device())
    )
    tenant_capacity = _positive_env("NTA_TENANT_CAPACITY", max(1, request_capacity))
    return ServingRuntimeResources.open(
        tier_config=ServingTierConfig.from_environment(),
        runtime_config=RuntimeResourceConfig.with_environment_staging_limit(
            request_capacity=request_capacity,
            # A physical work item owns a K/V object pair. Keep object
            # capacity proportional to the ticket bound instead of the
            # mutable vLLM batch row count.
            object_capacity=max(2, 2 * work_capacity),
            intent_capacity=max(2, 2 * work_capacity),
            work_ticket_capacity=work_capacity,
            max_dependencies_per_work_ticket=2,
            device_ordinal=device_ordinal,
            tenant_capacity=tenant_capacity,
        ),
    )


class VllmV1WorkerController:
    """Own one worker-local runtime and identity hook.

    The class name retains the vLLM ``v1`` API namespace distinction; the
    current 0.26 profile uses the ``v1.worker.gpu.model_runner`` V2 runner.
    The controller owns the native runtime for exactly that runner lifetime.
    """

    def __init__(self, runner: Any) -> None:
        self._runner_ref = weakref.ref(runner)
        self._runtime: Runtime | None = None
        self._resources: ServingRuntimeResources | None = None
        self._hook: VllmV1Hook | None = None
        self._page_size = 0
        self._page_bytes = 0
        self._request_capacity = 0
        self._epoch = 0

    @staticmethod
    def _cache_geometry(runner: Any) -> tuple[int, int]:
        groups = getattr(
            getattr(runner, "kv_cache_config", None), "kv_cache_groups", ()
        )
        if len(groups) != 1:
            raise RuntimeError(
                "NTA vLLM currently requires exactly one KV cache group"
            )
        spec = groups[0].kv_cache_spec
        page_size = int(getattr(spec, "block_size", 0))
        if page_size <= 0:
            raise RuntimeError("vLLM KV cache spec has no positive block_size")
        page_bytes = int(getattr(spec, "page_size_bytes", 0))
        if page_bytes <= 0:
            raise RuntimeError("vLLM KV cache spec has no page_size_bytes")
        return page_size, page_bytes

    def _ensure_hook(
        self,
        runner: Any,
        request_capacity: int,
        page_size: int,
        page_bytes: int,
    ) -> VllmV1Hook:
        if self._runtime is None:
            work_capacity = _positive_env(
                "NTA_VLLM_WORK_TICKET_CAPACITY",
                max(request_capacity, 4 * request_capacity),
            )
            resources = _build_resources(runner, request_capacity, work_capacity)
            runtime = resources.runtime
            try:
                tenant_specs = tenant_budget_specs()
                tenant_capacity = int(runtime.config.tenant_capacity)
                for tenant_id, max_bytes, weight in tenant_specs:
                    if tenant_id >= tenant_capacity:
                        raise RuntimeError(
                            f"tenant {tenant_id} exceeds NTA_TENANT_CAPACITY="
                            f"{tenant_capacity}"
                        )
                    runtime.set_tenant_budget(tenant_id, max_bytes, weight)
                hook = VllmV1Hook(
                    runtime,
                    request_capacity,
                    page_bytes=page_bytes,
                    expected_vllm_version=SUPPORTED_VLLM_VERSION,
                    tenant_for_request=tenant_mapper_from_environment(),
                )
            except BaseException:
                try:
                    resources.close()
                except BaseException:
                    pass
                raise
            self._resources = resources
            self._runtime = runtime
            self._hook = hook
            self._request_capacity = request_capacity
            self._page_bytes = page_bytes
        elif (
            self._request_capacity != request_capacity
            or self._page_bytes != page_bytes
        ):
            raise RuntimeError(
                "vLLM KV cache geometry changed while the worker runtime was live"
            )
        self._page_size = page_size
        assert self._hook is not None
        return self._hook

    def bind(self, scheduler_output: Any) -> EngineBatch:
        runner = self._runner_ref()
        if runner is None:
            raise RuntimeError("vLLM V1 model runner was destroyed")
        input_batch = getattr(runner, "input_batch", None)
        request_capacity = int(getattr(runner, "max_num_reqs", 0))
        if input_batch is None or request_capacity <= 0:
            raise RuntimeError("vLLM V1 runner is not initialized with InputBatch")
        page_size, page_bytes = self._cache_geometry(runner)
        hook = self._ensure_hook(
            runner, request_capacity, page_size, page_bytes
        )
        batch = hook.bind_forward(
            scheduler_output,
            input_batch,
            epoch=self._epoch,
            stream=torch.cuda.current_stream(),
            granularity=Granularity.PAGE_GROUP,
        )
        self._epoch += 1
        return batch

    def bind_v2(
        self,
        scheduler_output: Any,
        input_batch: Any,
        *,
        block_tables: Any,
        num_blocks: Any,
    ) -> EngineBatch:
        runner = self._runner_ref()
        if runner is None:
            raise RuntimeError("vLLM V2 model runner was destroyed")
        request_capacity = int(getattr(runner, "max_num_reqs", 0))
        if request_capacity <= 0:
            raise RuntimeError("vLLM V2 runner has no positive request capacity")
        page_size, page_bytes = self._cache_geometry(runner)
        hook = self._ensure_hook(
            runner, request_capacity, page_size, page_bytes
        )
        batch = hook.bind_v2_forward(
            scheduler_output,
            input_batch,
            block_tables=block_tables,
            num_blocks=num_blocks,
            epoch=self._epoch,
            stream=torch.cuda.current_stream(),
            granularity=Granularity.PAGE_GROUP,
        )
        self._epoch += 1
        return batch

    def close(self) -> None:
        """Close the runtime after the framework has stopped using the runner."""
        runtime, self._runtime = self._runtime, None
        resources, self._resources = self._resources, None
        self._hook = None
        self._page_size = 0
        self._page_bytes = 0
        self._request_capacity = 0
        if resources is not None:
            resources.close()
        elif runtime is not None:
            runtime.close()

    def __del__(self) -> None:
        # vLLM may abandon a worker during initialization (for example after
        # a tenant policy or KV geometry rejection).  The normal shutdown hook
        # is not guaranteed to run for that partial worker, so retain the same
        # best-effort runtime ownership fallback as the serving adapters.
        try:
            self.close()
        except BaseException:
            pass

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

    @property
    def tier_service(self) -> Any:
        if self._resources is None:
            raise RuntimeError("vLLM worker controller has no serving tier")
        return self._resources.tier


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
        self._nta_program: JitPhaseProgram | None = None
        self._physical_quiescence_event: torch.cuda.Event | None = None
        self._physical_quiescence_recorded = False
        # Native mode is explicit. Physical tiers are additionally checked by
        # the worker resource owner and are never silently routed through the
        # resident framework path.
        self._native_enabled = os.environ.get("NTA_VLLM_NATIVE", "0") == "1"
        self._serving_tier = validate_vllm_attention_tier()

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
        if os.environ.get("NTA_VLLM_DECODE_MODULE"):
            module_path = _find_module(module_name)
        else:
            module_path = _ensure_default_attention_module(
                module_name, query.dtype, self.head_size
            )
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

    def _ensure_prefill_wrapper(
        self, query: torch.Tensor, kv_cache: torch.Tensor
    ) -> None:
        if self._nta_prefill_wrapper is not None:
            return
        if query.dtype not in _DEFAULT_MODULES or kv_cache.dtype != query.dtype:
            raise RuntimeError(
                "native vLLM NTA prefill requires matching float16 or "
                "bfloat16 query and KV-cache dtypes"
            )
        if isinstance(self.kv_cache_dtype, str) and self.kv_cache_dtype not in {
            "auto",
            "float16",
            "bfloat16",
        }:
            raise RuntimeError(
                "native vLLM NTA prefill does not support quantized KV cache "
                f"dtype {self.kv_cache_dtype!r}"
            )
        module_name = os.environ.get(
            "NTA_VLLM_PREFILL_MODULE", _DEFAULT_MODULES[query.dtype]
        )
        if os.environ.get("NTA_VLLM_PREFILL_MODULE"):
            module_path = _find_module(module_name)
        else:
            module_path = _ensure_default_attention_module(
                module_name, query.dtype, self.head_size
            )
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
            int(
                getattr(
                    envs,
                    "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE",
                    64 * 1024 * 1024,
                )
            ),
        )
        workspace = torch.zeros(
            workspace_bytes, dtype=torch.uint8, device=query.device
        )
        self._nta_prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            get_kv_cache_layout(),
            backend="fa2",
            jit_args=jit_args,
        )

    def _build_plan(
        self,
        state: Any,
        schedule: Any,
        *,
        layer: int = 0,
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
                    layer=layer,
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

    def _upload_physical_plan(
        self,
        state: Any,
        schedule: Any,
        execution: ExecutionSession,
        layer: int,
        page_size: int,
        key_bytes: int,
        value_bytes: int,
    ) -> tuple[DeviceWorkPlan, int, bool]:
        """Upload exact CXL/NVMe dependencies for one vLLM layer.

        vLLM's packed resident KV layout is not a safe destination for a
        separate K/V physical extent.  The NTA attention module therefore
        consumes the two typed dependency addresses/staging objects directly,
        exactly as SGLang's physical path does.  This keeps NVMe DMA in HBM
        and leaves CXL as a device-visible direct dependency.
        """
        tier = getattr(state, "tier_service", None)
        if tier is None or tier.is_host:
            raise RuntimeError("vLLM physical plan has no physical tier service")
        batch = state.batch
        if not isinstance(batch, EngineBatch):
            raise RuntimeError("vLLM physical plan has no engine batch")
        runtime = state.hook.runtime
        work_count = schedule.work_count
        if work_count <= 0:
            raise RuntimeError("vLLM physical schedule is empty")
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

        stream = torch.cuda.current_stream()
        version = int(batch.epoch) + 1
        object_slots: dict[tuple[int, ...], tuple[int, int]] = {}
        dependencies: list[AcquireRequirement] = []
        spans: list[tuple[int, int, int, int]] = []
        object_count = 0
        for work_id, (request_index, kv_tile) in enumerate(
            zip(schedule.request_indices, schedule.kv_tile_indices, strict=True)
        ):
            pages = self._physical_pages(
                batch, schedule, int(request_index), int(kv_tile), page_size
            )
            begin = len(dependencies)
            if tier.is_cxl:
                key_extent = tier.extent(layer, pages, "key", key_bytes)
                value_extent = tier.extent(layer, pages, "value", value_bytes)
                dependencies.extend(
                    (
                        direct_requirement(
                            tier.device_address(key_extent), key_extent.bytes
                        ),
                        direct_requirement(
                            tier.device_address(value_extent), value_extent.bytes
                        ),
                    )
                )
                direct_count = 2
            elif tier.is_nvme:
                slots = object_slots.get(pages)
                if slots is None:
                    key_extent = tier.extent(layer, pages, "key", key_bytes)
                    value_extent = tier.extent(layer, pages, "value", value_bytes)
                    if object_count + 2 > runtime.config.object_capacity:
                        raise RuntimeError(
                            "vLLM physical layer exceeds the runtime object capacity"
                        )
                    key_slot = object_count
                    key_address = runtime.install_nvme_object_async(
                        key_slot,
                        0x4E54410000000000 | key_slot,
                        version,
                        key_extent.offset,
                        key_extent.bytes,
                        stream,
                        self._physical_quiescence_event
                        if self._physical_quiescence_recorded
                        else None,
                    )
                    value_slot = key_slot + 1
                    value_address = runtime.install_nvme_object_async(
                        value_slot,
                        0x4E54410000000000 | value_slot,
                        version,
                        value_extent.offset,
                        value_extent.bytes,
                        stream,
                        self._physical_quiescence_event
                        if self._physical_quiescence_recorded
                        else None,
                    )
                    del key_address, value_address
                    slots = (key_slot, value_slot)
                    object_slots[pages] = slots
                    object_count += 2
                key_slot, value_slot = slots
                dependencies.extend(
                    (
                        AcquireRequirement(
                            0,
                            0,
                            0x4E54410000000000 | key_slot,
                            0,
                            key_slot,
                            version,
                            len(pages) * key_bytes,
                            0,
                        ),
                        AcquireRequirement(
                            0,
                            0,
                            0x4E54410000000000 | value_slot,
                            0,
                            value_slot,
                            version,
                            len(pages) * value_bytes,
                            0,
                        ),
                    )
                )
                direct_count = 0
            else:
                raise RuntimeError(f"unsupported vLLM physical tier {tier.tier}")
            spans.append((begin, 2, direct_count, work_id))

        ranges = request_ranges_for_schedule(batch.bindings, schedule.request_indices)
        self._nta_plan.upload_work_units(
            execution.batch.units,
            spans,
            dependencies,
            ranges,
            epoch=execution.epoch,
            stream=stream,
        )
        return self._nta_plan, object_count, tier.is_nvme

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

    def _run_native_schedule(
        self,
        state: Any,
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
        physical = self._serving_tier != "host_staged"
        raw_layer_id = getattr(layer, "layer_id", None)
        if physical and raw_layer_id is None:
            raise RuntimeError(
                "vLLM physical attention requires a stable transformer layer_id"
            )
        layer_id = 0 if raw_layer_id is None else int(raw_layer_id)
        execution = self._build_plan(
            state,
            schedule,
            layer=layer_id if physical else 0,
        )
        if self._nta_program is None:
            raise RuntimeError("vLLM NTA attention has no validated phase program")
        batch = state.batch
        if not isinstance(batch, EngineBatch) or batch.exact_demand is None:
            raise RuntimeError("vLLM NTA attention has no exact engine batch")
        if physical:
            tier = getattr(state, "tier_service", None)
            if tier is None or tier.tier.value != self._serving_tier:
                raise RuntimeError(
                    "vLLM forward tier does not match the worker resource owner"
                )
            if batch.exact_demand.unit_bytes % 2:
                raise RuntimeError(
                    "vLLM physical KV page bytes must split evenly into K/V"
                )
            plan, object_count, is_nvme = self._upload_physical_plan(
                state,
                schedule,
                execution,
                layer_id,
                int(state.page_size),
                batch.exact_demand.unit_bytes // 2,
                batch.exact_demand.unit_bytes // 2,
            )
            self._nta_program.reset(
                state.hook.runtime,
                object_count=object_count,
                work_ticket_count=schedule.work_count,
                stream=torch.cuda.current_stream(),
            )
        else:
            is_nvme = False
            self._nta_program.reset(
                state.hook.runtime,
                object_count=0,
                work_ticket_count=schedule.work_count,
                stream=torch.cuda.current_stream(),
            )
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
        if physical and is_nvme:
            epoch = FlashInferLayerEpoch(
                state.hook.runtime,
                plan,
                self._nta_program,
                object_count=object_count,
                max_progress_rounds=tier.config.progress_rounds,
                wait_for_plan=False,
            )
            progress_rounds = epoch.enqueue_nvme(
                wrapper,
                query,
                kv_cache_for_flashinfer,
                output,
                issue_budget=tier.config.issue_budget,
                completion_budget=tier.config.completion_budget,
                timeout_ns=tier.config.progress_timeout_ns,
                sm_scale=self.scale,
                stream=torch.cuda.current_stream(),
            )
            if os.environ.get("NTA_VLLM_VERIFY_TRANSFER") == "1":
                epoch.check(progress_rounds, torch.cuda.current_stream())
        elif physical:
            enqueue_resident_attention(
                state.hook.runtime,
                plan,
                wrapper,
                query,
                kv_cache_for_flashinfer,
                output,
                sm_scale=self.scale,
            )
        else:
            wrapper.run(
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
        if os.environ.get("NTA_VLLM_COMPARE_STOCK") == "1":
            stock_output = torch.empty_like(output)
            stock_wrapper.run(query, kv_cache_for_flashinfer, out=stock_output)
            torch.cuda.synchronize()
            difference = torch.nan_to_num(
                (output.float() - stock_output.float()).abs(),
                nan=float("inf"),
            ).max().item()
            VLLM_STATS["native_stock_diff_max_milli"] = max(
                VLLM_STATS["native_stock_diff_max_milli"],
                int(difference * 1000),
            )
        if physical and is_nvme:
            if self._physical_quiescence_event is None:
                self._physical_quiescence_event = torch.cuda.Event()
            self._physical_quiescence_event.record(torch.cuda.current_stream())
            self._physical_quiescence_recorded = True
        elif not physical:
            plan.mark_consumed(torch.cuda.current_stream())
        execution.record_layer_completion(0)
        state.hook.record_native_launch()
        VLLM_STATS[f"native_{kind}_launches"] += 1
        VLLM_STATS[f"physical_{kind}_launches"] += int(physical)
        VLLM_STATS[f"native_{kind}_work_items"] += schedule.work_count
        return output

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
        if attn_metadata.num_decodes or attn_metadata.use_cascade:
            raise RuntimeError("native vLLM prefill requires a pure prefill batch")
        prefill_metadata = attn_metadata.prefill
        if not isinstance(prefill_metadata, FIPrefill):
            raise RuntimeError(
                "native vLLM prefill requires the FlashInfer FA2 prefill metadata"
            )
        stock_wrapper = prefill_metadata.wrapper
        buffers = tuple(
            getattr(stock_wrapper, name, None)
            for name in (
                "_qo_indptr_buf",
                "_paged_kv_indptr_buf",
                "_paged_kv_indices_buf",
                "_paged_kv_last_page_len_buf",
            )
        )
        if not all(isinstance(tensor, torch.Tensor) for tensor in buffers):
            raise RuntimeError(
                "vLLM FlashInfer prefill metadata has no typed paged-KV buffers"
            )
        qo_indptr, indptr, indices, last_page_len = buffers
        if any(
            tensor.dtype != torch.int32
            or not tensor.is_cuda
            or not tensor.is_contiguous()
            for tensor in buffers
        ):
            raise RuntimeError(
                "vLLM FlashInfer prefill buffers must be contiguous CUDA int32 tensors"
            )
        if qo_indptr.numel() != attn_metadata.num_prefills + 1:
            raise RuntimeError("vLLM FlashInfer prefill has the wrong request count")
        batch = state.batch
        if not isinstance(batch, EngineBatch):
            raise RuntimeError("vLLM prefill has no engine batch")
        if len(batch.bindings) != attn_metadata.num_prefills:
            raise RuntimeError(
                "native vLLM prefill requires one exact row per scheduled request"
            )
        page_size = int(getattr(state, "page_size", 0) or 0)
        if page_size <= 0:
            raise RuntimeError("vLLM forward sidecar has no token page size")
        self._ensure_prefill_wrapper(query, kv_cache)
        assert self._nta_prefill_wrapper is not None
        self._nta_prefill_wrapper.plan(
            qo_indptr,
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
            causal=attn_metadata.causal,
            disable_split_kv=True,
        )
        schedule = paged_prefill_schedule(self._nta_prefill_wrapper)
        if schedule.work_count <= 0:
            raise RuntimeError("vLLM FlashInfer prefill produced no work units")
        return self._run_native_schedule(
            state,
            schedule,
            self._nta_prefill_wrapper,
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
        if any(
            tensor.dtype != torch.int32 or not tensor.is_cuda or not tensor.is_contiguous()
            for tensor in (indptr, indices, last_page_len)
        ):
            raise RuntimeError(
                "vLLM FlashInfer paged-KV buffers must be contiguous CUDA int32 tensors"
            )
        # The non-graph FlashInfer wrapper stores exactly the ``indices`` tensor
        # supplied to ``plan``.  Use its host-known length instead of reading
        # ``indptr[-1]`` back to Python, which would add a synchronization to
        # every attention layer.  The wrapper itself has already validated the
        # indptr/indices geometry while planning.
        page_count = indices.numel()
        if page_count <= 0:
            raise RuntimeError("vLLM FlashInfer page-index buffer is incomplete")
        last_page_len = last_page_len[: attn_metadata.num_decodes]
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
        return self._run_native_schedule(
            state,
            schedule,
            self._nta_wrapper,
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
        num_actual_tokens = attn_metadata.num_actual_tokens
        try:
            if attn_metadata.num_prefill_tokens:
                if attn_metadata.num_decodes:
                    raise RuntimeError(
                        "native vLLM attention requires a pure prefill or pure "
                        "decode batch; mixed batches use the explicit reference"
                    )
                self._native_prefill_forward(
                    layer,
                    query[:num_actual_tokens],
                    kv_cache,
                    attn_metadata,
                    output[:num_actual_tokens],
                )
            else:
                if attn_metadata.num_decode_tokens != attn_metadata.num_decodes:
                    raise RuntimeError(
                        "native vLLM decode requires one query token per request"
                    )
                self._native_forward(
                    layer,
                    query[:num_actual_tokens],
                    key[:num_actual_tokens],
                    value[:num_actual_tokens],
                    kv_cache,
                    attn_metadata,
                    output[:num_actual_tokens],
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
    native = VLLM_STATS["native_decode_launches"] + VLLM_STATS[
        "native_prefill_launches"
    ]
    if native:
        contract = ConsumerContract.native_work_unit(
            engine="vllm",
            backend="nta_flashinfer",
            engine_version=SUPPORTED_VLLM_VERSION,
        ).as_dict()
        contract.update(
            {
                "native_launches": native,
                "serving_tier": os.environ.get("NTA_SERVING_TIER", "host_staged"),
                "resident_only": (
                    VLLM_STATS["physical_decode_launches"]
                    + VLLM_STATS["physical_prefill_launches"]
                    == 0
                ),
                "physical_decode_launches": VLLM_STATS["physical_decode_launches"],
                "physical_prefill_launches": VLLM_STATS[
                    "physical_prefill_launches"
                ],
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
        "stock_fallback_enabled": os.environ.get(
            "NTA_VLLM_ALLOW_STOCK_FALLBACK", "0"
        )
        == "1",
        "serving_tier": os.environ.get("NTA_SERVING_TIER", "host_staged"),
        "physical_decode_launches": VLLM_STATS["physical_decode_launches"],
        "physical_prefill_launches": VLLM_STATS["physical_prefill_launches"],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    except OSError:
        # Evidence publication cannot change inference correctness during
        # interpreter teardown. The serving harness treats missing evidence as
        # a failed native-verification gate.
        return
