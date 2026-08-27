"""vLLM 0.26 V1 consumer for instrumented FlashInfer attention.

The worker controller is the only framework-private bridge in this module.
It publishes an engine-neutral :class:`EngineBatch` after vLLM updates its
persistent input batch.  ``NtaVllmFlashInferImpl`` then consumes that batch
through a real vLLM ``AttentionImpl`` call and submits the same typed NTA
work-plan ABI used by the SGLang adapter.

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
from collections import Counter
from collections.abc import Callable
import contextlib
from dataclasses import dataclass
import json
import os
import pathlib
import threading
import time
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
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides

from nta_runtime.adapters.base import ConsumerContract, EngineBatch
from nta_runtime.adapters.vllm_v1 import (
    VllmV1Hook,
    current_vllm_v1_forward_state,
    validate_vllm_attention_tier,
)
from nta_runtime.execution_core import ExecutionPlan, ExecutionSession, ExecutionTile
from nta_runtime.execution_topology import ExactWorkTopology, WorkDependencySpan
from nta_runtime.execution_protocol import ExecutionProtocolConfig
from nta_runtime.runtime_resources import (
    RuntimeResourceConfig,
    ServingRuntimeResources,
)
from nta_runtime.hbm_registration import HbmDestinationSlice
from nta_runtime.indexed_transfer import (
    IndexedHostResource,
    IndexedTensorLane,
    IndexedTransferGroup,
    IndexedTransferTopology,
    IndexedWorkDependency,
    plan_indexed_dependencies,
)
from nta_runtime.tier import PageTransferRun, ServingTierConfig, TierPageCatalog
from nta_runtime.flashinfer import (
    FlashInferLayerEpoch,
    PREACQUIRED_LAUNCH_FLAGS,
    attention_jit_args,
    direct_requirement,
    enqueue_resident_attention,
    mapped_request_bound_attention_jit_args,
    object_requirement,
)
from nta_runtime.flashinfer_schedule import decode_schedule, paged_prefill_schedule
from nta_runtime.tenant import tenant_budget_specs, tenant_mapper_from_environment
from nta_runtime.runtime import (
    AcquireRequirement,
    DeviceWorkPlan,
    IndexedHostIndexBinding,
    IndexedHostPlan,
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
_REQUEST_BOUND_MODULES = {
    torch.float16: "nta_batch_prefill_vllm_request_bound_v2_mapped_fp16",
    torch.bfloat16: "nta_batch_prefill_vllm_request_bound_v2_mapped_bf16",
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
    name: str,
    dtype: torch.dtype,
    head_size: int,
    *,
    request_bound: bool = False,
    mapped_request_slots: bool = False,
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

        if mapped_request_slots:
            if not request_bound:
                raise RuntimeError("mapped request slots require a direct module")
            tensor_names = ["nta_runtime", "nta_request_slots"]
            tensor_dtypes = ["uint8_t", "int32_t"]
            scalar_names = ["sm_scale"]
            scalar_dtypes = ["double"]
        elif request_bound:
            tensor_names = ["nta_runtime"]
            tensor_dtypes = ["uint8_t"]
            scalar_names = ["sm_scale", "nta_request_slot_offset"]
            scalar_dtypes = ["double", "int64_t"]
        else:
            tensor_names = ["nta_runtime", "nta_work_items", "nta_dependencies"]
            tensor_dtypes = ["uint8_t", "uint8_t", "uint8_t"]
            scalar_names = ["sm_scale", "nta_work_count", "nta_skip_merge"]
            scalar_dtypes = ["double", "int64_t", "int64_t"]
        specification = gen_customize_batch_prefill_module(
            "fa2",
            name,
            dtype,
            dtype,
            dtype,
            torch.int32,
            head_size,
            head_size,
            tensor_names,
            tensor_dtypes,
            scalar_names,
            scalar_dtypes,
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
        request_bound = "request_bound" in path.name
        form = OperatorForm.DIRECT if request_bound else OperatorForm.INCREMENTAL
        capabilities = (
            OperatorCapability.REQUEST_BINDING
            | OperatorCapability.TYPED_FLASHINFER_FRONTEND
        )
        if request_bound:
            capabilities |= OperatorCapability.GRAPH_REPLAY
        else:
            capabilities |= (
                OperatorCapability.OBJECT_DEPENDENCIES
                | OperatorCapability.FINITE_DEFERRAL
                | OperatorCapability.PARTIAL_PUBLICATION
                | OperatorCapability.COMPLETE_CONTRIBUTOR_MERGE
                | OperatorCapability.RUNNABLE_COMPACTION
            )
        program.operator_contract.require(
            family=family,
            form=form,
            capabilities=capabilities,
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


def _new_request_bound_wrapper(
    kind: str,
    workspace: torch.Tensor,
    *,
    query_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    head_size: int,
) -> Any:
    """Build and validate one direct wrapper for a framework-owned plan."""
    if kind not in {"decode", "prefill"}:
        raise ValueError(f"unknown vLLM attention phase {kind!r}")
    if query_dtype not in _REQUEST_BOUND_MODULES or kv_dtype != query_dtype:
        raise RuntimeError(
            "native vLLM direct attention requires matching float16 or "
            "bfloat16 query and resident KV cache dtypes"
        )
    module_name = _REQUEST_BOUND_MODULES[query_dtype]
    module_path = _ensure_default_attention_module(
        module_name,
        query_dtype,
        head_size,
        request_bound=True,
        mapped_request_slots=True,
    )
    _phase_program(module_path)
    jit_args = mapped_request_bound_attention_jit_args(
        module_name,
        dtype_q=query_dtype,
        dtype_kv=kv_dtype,
        dtype_o=query_dtype,
        idtype=torch.int32,
        head_dim_qk=head_size,
        head_dim_vo=head_size,
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
    # This object is process-local and owned by the metadata builder.  The
    # marker distinguishes a framework-planned direct module from a stock
    # wrapper that still needs the controller's incremental fallback.
    wrapper._nta_request_bound = True
    VLLM_STATS["framework_direct_wrapper_builds"] += 1
    return wrapper


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
    max_dependencies = _positive_env(
        "NTA_VLLM_MAX_DEPENDENCIES_PER_WORK", 32
    )
    return ServingRuntimeResources.open(
        tier_config=ServingTierConfig.from_environment(),
        runtime_config=RuntimeResourceConfig.with_environment_staging_limit(
            request_capacity=request_capacity,
            # Physical objects are maximal source/destination-contiguous runs.
            # Keep the bound explicit: normal sequential prefixes collapse to
            # one run, while fragmented catalogs fail closed instead of
            # silently dropping dependencies.
            object_capacity=max(2, max_dependencies * work_capacity),
            intent_capacity=max(2, max_dependencies * work_capacity),
            work_ticket_capacity=work_capacity,
            max_dependencies_per_work_ticket=max_dependencies,
            device_ordinal=device_ordinal,
            tenant_capacity=tenant_capacity,
        ),
    )


@dataclass
class _WorkerAttentionPhase:
    """One worker-owned FlashInfer resource and its epoch-scoped plan result."""

    resource: Any
    workspace_bytes: int
    planned_epoch: int | None = None
    plan_result: Any = None


@dataclass(frozen=True)
class _PhysicalLayerDestination:
    """One framework-owned, setup-time registered packed KV tensor."""

    layer_name: str
    catalog_layer: int
    tensor_address: int
    block_count: int
    block_bytes: int
    region: Any

    def address(self, first_block: int, block_count: int) -> int:
        if first_block < 0 or block_count <= 0:
            raise ValueError("vLLM physical destination range must be positive")
        if first_block > self.block_count - block_count:
            raise RuntimeError("vLLM physical destination exceeds its KV tensor")
        return self.tensor_address + first_block * self.block_bytes


@dataclass(frozen=True)
class _PhysicalTransferLayout:
    """Pure directory result consumed by the runtime publication step."""

    runs: tuple[PageTransferRun, ...]
    run_indices_by_work: tuple[tuple[int, ...], ...]
    unique_block_count: int


def _physical_transfer_layout(
    catalog: TierPageCatalog,
    *,
    catalog_layer: int,
    work_bindings: tuple[tuple[tuple[str, int], ...], ...],
    row_bytes: int,
    max_transfer_bytes: int,
) -> _PhysicalTransferLayout:
    """Coalesce one layer while preserving every work item's ready frontier."""

    key_by_destination: dict[int, str] = {}
    for bindings in work_bindings:
        for storage_key, block in bindings:
            previous = key_by_destination.setdefault(block, storage_key)
            if previous != storage_key:
                raise RuntimeError(
                    "vLLM destination block is bound to conflicting storage keys"
                )
    ordered = tuple(sorted(key_by_destination.items()))
    runs = (
        catalog.transfer_runs(
            layer=catalog_layer,
            storage_keys=tuple(storage_key for _, storage_key in ordered),
            destination_indices=tuple(block for block, _ in ordered),
            component="packed_kv",
            row_bytes=row_bytes,
            max_transfer_bytes=max_transfer_bytes,
        )
        if ordered
        else ()
    )
    run_by_destination: dict[int, int] = {}
    for run_index, run in enumerate(runs):
        for block in range(
            run.destination_first, run.destination_first + run.row_count
        ):
            if block in run_by_destination:
                raise RuntimeError("vLLM physical transfer runs overlap")
            run_by_destination[block] = run_index
    run_indices_by_work = tuple(
        tuple(dict.fromkeys(run_by_destination[block] for _, block in bindings))
        for bindings in work_bindings
    )
    return _PhysicalTransferLayout(runs, run_indices_by_work, len(ordered))


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
        # FlashInfer planner state and its large workspace are worker resources,
        # not transformer-layer resources.  vLLM invokes one AttentionImpl per
        # layer with the same batch metadata, so a per-impl owner multiplies
        # workspace residency and planner/readback cost by the layer count.
        self._attention_phases: dict[
            tuple[Any, ...], _WorkerAttentionPhase
        ] = {}
        self._attention_workspace_bytes = 0
        self._request_slots_host: torch.Tensor | None = None
        self._request_slots_host_numpy: Any | None = None
        self._request_slots_device: torch.Tensor | None = None
        self._layer_ordinals: dict[str, int] = {}
        self._physical_destinations: dict[str, _PhysicalLayerDestination] = {}
        self._physical_destinations_prepared = False
        # The runtime object directory is worker-global, while vLLM constructs
        # one AttentionImpl per transformer layer.  Directory generations and
        # consumer quiescence must therefore live here, not in an impl.
        self._external_object_version = 0
        self._external_consumer_event: torch.cuda.Event | None = None

    @staticmethod
    def _cache_geometry(runner: Any) -> tuple[int, int]:
        groups = getattr(
            getattr(runner, "kv_cache_config", None), "kv_cache_groups", ()
        )
        if len(groups) != 1:
            raise RuntimeError("NTA vLLM currently requires exactly one KV cache group")
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
            scheduler_config = getattr(
                getattr(runner, "vllm_config", None), "scheduler_config", None
            )
            max_batched_tokens = int(
                getattr(scheduler_config, "max_num_batched_tokens", 0) or 0
            )
            work_capacity = _positive_env(
                "NTA_VLLM_WORK_TICKET_CAPACITY",
                # FlashInfer's canonical x-coordinate is finer than a request:
                # causal prefill emits Q tiles and may split each across KV.
                # Bound the runtime by the framework token envelope, while a
                # user override remains available for unusually fragmented
                # custom planners.
                max(256, 64 * request_capacity, max_batched_tokens),
            )
            resources = _build_resources(runner, request_capacity, work_capacity)
            runtime = resources.runtime
            try:
                tenant_specs = tenant_budget_specs()
                tenant_capacity = int(runtime.config.tenant_capacity)
                for tenant_id, max_bytes in tenant_specs:
                    if tenant_id >= tenant_capacity:
                        raise RuntimeError(
                            f"tenant {tenant_id} exceeds NTA_TENANT_CAPACITY="
                            f"{tenant_capacity}"
                        )
                    runtime.set_tenant_budget(tenant_id, max_bytes)
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
            self._request_capacity != request_capacity or self._page_bytes != page_bytes
        ):
            raise RuntimeError(
                "vLLM KV cache geometry changed while the worker runtime was live"
            )
        self._page_size = page_size
        assert self._hook is not None
        return self._hook

    def _publish_request_slots(self, batch: EngineBatch) -> None:
        """Publish one stable request-index to runtime-slot map per forward."""
        if self._request_slots_host is None:
            runner = self._runner_ref()
            if runner is None:
                raise RuntimeError("vLLM model runner was destroyed")
            device = getattr(runner, "device", torch.device("cuda"))
            self._request_slots_host = torch.empty(
                self._request_capacity,
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            self._request_slots_host.fill_(-1)
            self._request_slots_host_numpy = self._request_slots_host.numpy()
            self._request_slots_device = torch.empty(
                self._request_capacity, dtype=torch.int32, device=device
            )
        assert self._request_slots_host_numpy is not None
        assert self._request_slots_device is not None
        self._request_slots_host_numpy[: len(batch.bindings)] = batch.request_slots
        self._request_slots_device.copy_(self._request_slots_host, non_blocking=True)

    def bind(self, scheduler_output: Any) -> EngineBatch:
        runner = self._runner_ref()
        if runner is None:
            raise RuntimeError("vLLM V1 model runner was destroyed")
        input_batch = getattr(runner, "input_batch", None)
        request_capacity = int(getattr(runner, "max_num_reqs", 0))
        if input_batch is None or request_capacity <= 0:
            raise RuntimeError("vLLM V1 runner is not initialized with InputBatch")
        page_size, page_bytes = self._cache_geometry(runner)
        hook = self._ensure_hook(runner, request_capacity, page_size, page_bytes)
        started = time.perf_counter_ns()
        batch = hook.bind_forward(
            scheduler_output,
            input_batch,
            epoch=self._epoch,
            stream=torch.cuda.current_stream(),
            granularity=Granularity.PAGE_GROUP,
        )
        self._publish_request_slots(batch)
        if os.environ.get("NTA_PROFILE_CPU") == "1":
            VLLM_STATS["bridge_bind_cpu_ns"] += time.perf_counter_ns() - started
            VLLM_STATS["bridge_bind_calls"] += 1
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
        hook = self._ensure_hook(runner, request_capacity, page_size, page_bytes)
        started = time.perf_counter_ns()
        batch = hook.bind_v2_forward(
            scheduler_output,
            input_batch,
            block_tables=block_tables,
            num_blocks=num_blocks,
            epoch=self._epoch,
            stream=torch.cuda.current_stream(),
            granularity=Granularity.PAGE_GROUP,
        )
        self._publish_request_slots(batch)
        if os.environ.get("NTA_PROFILE_CPU") == "1":
            VLLM_STATS["bridge_bind_cpu_ns"] += time.perf_counter_ns() - started
            VLLM_STATS["bridge_bind_calls"] += 1
            VLLM_STATS.update(hook.last_bind_profile)
        self._epoch += 1
        return batch

    def bind_connector(self, metadata: Any) -> EngineBatch:
        """Bind one official KVConnector metadata object before FI planning."""
        runner = self._runner_ref()
        if runner is None:
            raise RuntimeError("vLLM V2 model runner was destroyed")
        request_capacity = int(getattr(runner, "max_num_reqs", 0))
        if request_capacity <= 0:
            raise RuntimeError("vLLM V2 runner has no positive request capacity")
        page_size, page_bytes = self._cache_geometry(runner)
        hook = self._ensure_hook(runner, request_capacity, page_size, page_bytes)
        batch = hook.bind_connector_forward(
            metadata.request_ids,
            metadata.block_tables,
            metadata.finished_request_ids,
            epoch=self._epoch,
            stream=torch.cuda.current_stream(),
            granularity=Granularity.PAGE_GROUP,
        )
        self._publish_request_slots(batch)
        self._epoch += 1
        return batch

    def prepare_physical_destinations(self) -> None:
        """Register every packed vLLM layer destination exactly once.

        vLLM owns allocation and numerical lifetime.  NTA owns only peer
        mapping views over those tensors, and installs per-transfer object
        views later without allocation, registration, or ioctl work in the
        forward path.
        """

        if self._resources is None:
            raise RuntimeError("vLLM physical setup ran before runtime binding")
        tier = self._resources.tier
        if tier.is_hbm or tier.is_host_staged:
            return
        if not tier.is_nvme:
            raise RuntimeError(
                f"vLLM {tier.tier.value} numerical consumption is deferred; "
                "the profile fails closed instead of preparing the wrong address space"
            )
        if self._physical_destinations_prepared:
            return
        runner = self._runner_ref()
        if runner is None:
            raise RuntimeError("vLLM model runner was destroyed")
        catalog = tier.catalog
        if catalog is None or catalog.components != ("packed_kv",):
            raise RuntimeError("vLLM NVMe setup requires a packed_kv catalog")
        groups = tuple(getattr(runner.kv_cache_config, "kv_cache_groups", ()))
        if len(groups) != 1:
            raise RuntimeError("vLLM NVMe setup requires exactly one KV group")
        layer_names = tuple(str(value) for value in groups[0].layer_names)
        if len(layer_names) != catalog.layer_count:
            raise RuntimeError("vLLM KV layers do not match the physical catalog")
        context = getattr(
            getattr(runner, "compilation_config", None),
            "static_forward_context",
            None,
        )
        if not isinstance(context, dict):
            raise RuntimeError("vLLM runner has no static attention-layer directory")

        destinations: list[HbmDestinationSlice] = []
        geometry: dict[str, tuple[int, int]] = {}
        addresses: set[int] = set()
        for layer_name in layer_names:
            layer = context.get(layer_name)
            tensor = getattr(layer, "kv_cache", None)
            if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
                raise RuntimeError(
                    f"vLLM layer {layer_name!r} has no CUDA KV destination"
                )
            if tensor.ndim < 1 or tensor.shape[0] <= 0:
                raise RuntimeError(f"vLLM layer {layer_name!r} has invalid KV shape")
            block_count = int(tensor.shape[0])
            block_stride = int(tensor.stride(0)) * int(tensor.element_size())
            if block_stride != self._page_bytes:
                raise RuntimeError(
                    f"vLLM layer {layer_name!r} block stride {block_stride} "
                    f"does not match catalog payload bytes {self._page_bytes}"
                )
            address = int(tensor.data_ptr())
            if address <= 0 or address in addresses:
                raise RuntimeError(
                    "vLLM physical profile does not support shared/aliased layer "
                    "KV destinations"
                )
            addresses.add(address)
            total_bytes = block_count * self._page_bytes
            destinations.append(
                HbmDestinationSlice(layer_name, address, total_bytes)
            )
            geometry[layer_name] = (address, block_count)

        preparation = tier.prepare_nvme_hbm_destinations(tuple(destinations))
        prepared: dict[str, _PhysicalLayerDestination] = {}
        for catalog_layer, layer_name in enumerate(layer_names):
            address, block_count = geometry[layer_name]
            region = preparation.regions.get(layer_name)
            if region is None:
                raise RuntimeError("vLLM NVMe setup lost a registered layer mapping")
            prepared[layer_name] = _PhysicalLayerDestination(
                layer_name,
                catalog_layer,
                address,
                block_count,
                self._page_bytes,
                region,
            )
        self._physical_destinations = prepared
        self._physical_destinations_prepared = True
        VLLM_STATS["physical_destination_layers"] = len(prepared)
        VLLM_STATS["physical_destination_registrations"] = (
            preparation.registration_count
        )
        VLLM_STATS["physical_destination_bytes"] = preparation.destination_bytes
        VLLM_STATS["physical_registration_bytes"] = preparation.registration_bytes

    def physical_destination(self, layer: Any) -> _PhysicalLayerDestination:
        """Resolve an Attention layer through vLLM's stable layer-name seam."""

        if not self._physical_destinations_prepared:
            raise RuntimeError("vLLM physical destinations were not prepared")
        layer_name = getattr(layer, "layer_name", None)
        if not isinstance(layer_name, str) or not layer_name:
            raise RuntimeError("vLLM attention layer has no stable layer_name")
        try:
            return self._physical_destinations[layer_name]
        except KeyError:
            raise RuntimeError(
                f"vLLM layer {layer_name!r} is absent from the physical directory"
            ) from None

    def semantic_layer(self, layer: Any) -> int:
        """Resolve one framework layer to a stable model-local ordinal."""

        layer_name = getattr(layer, "layer_name", None)
        if not isinstance(layer_name, str) or not layer_name:
            raise RuntimeError("vLLM attention layer has no stable layer_name")
        if not self._layer_ordinals:
            runner = self._runner_ref()
            groups = tuple(
                getattr(getattr(runner, "kv_cache_config", None), "kv_cache_groups", ())
            )
            if len(groups) != 1:
                raise RuntimeError("vLLM semantic layer directory requires one KV group")
            names = tuple(str(value) for value in groups[0].layer_names)
            if not names or len(set(names)) != len(names):
                raise RuntimeError("vLLM KV layer directory is empty or ambiguous")
            self._layer_ordinals = {
                name: ordinal for ordinal, name in enumerate(names)
            }
        try:
            return self._layer_ordinals[layer_name]
        except KeyError:
            raise RuntimeError(
                f"vLLM layer {layer_name!r} is absent from the semantic directory"
            ) from None

    def attention_phase(
        self,
        form: str,
        key: tuple[Any, ...],
        epoch: int,
        build: Callable[[], Any],
        plan: Callable[[Any], Any],
        *,
        workspace_bytes: int,
    ) -> tuple[Any, Any]:
        """Acquire one worker-shared wrapper and plan it once per batch epoch.

        vLLM shares one FlashInfer metadata plan across all transformer layers.
        NTA must preserve the same ownership boundary: planning a custom wrapper
        in every ``AttentionImpl`` adds repeated planner work, device-to-host
        schedule readback, and a full workspace per layer.  The worker executes
        layers serially, so one resource per form/phase/signature is the
        narrowest safe lifetime.  ``plan_result`` lets incremental consumers
        cache the extracted structural schedule alongside the wrapper.
        """
        if form not in {"request_bound", "incremental"}:
            raise ValueError(f"unknown vLLM attention form {form!r}")
        if epoch < 0:
            raise ValueError("vLLM attention phase epoch cannot be negative")
        if workspace_bytes <= 0:
            raise ValueError("vLLM attention workspace must be positive")
        phase_key = (form, *key)
        phase = self._attention_phases.get(phase_key)
        if phase is None:
            phase = _WorkerAttentionPhase(build(), workspace_bytes)
            self._attention_phases[phase_key] = phase
            self._attention_workspace_bytes += workspace_bytes
            prefix = f"worker_{form}"
            VLLM_STATS[f"{prefix}_wrapper_builds"] += 1
            VLLM_STATS[f"{prefix}_workspace_allocated_bytes"] += workspace_bytes
            VLLM_STATS["worker_attention_workspace_peak_bytes"] = max(
                VLLM_STATS["worker_attention_workspace_peak_bytes"],
                self._attention_workspace_bytes,
            )
        if phase.planned_epoch == epoch:
            VLLM_STATS[f"worker_{form}_plan_reuses"] += 1
            return phase.resource, phase.plan_result
        if phase.planned_epoch is not None and epoch < phase.planned_epoch:
            raise RuntimeError("vLLM attention phase epoch moved backwards")
        phase.plan_result = plan(phase.resource)
        phase.planned_epoch = epoch
        VLLM_STATS[f"worker_{form}_plan_builds"] += 1
        return phase.resource, phase.plan_result

    def begin_external_publication(self) -> tuple[int, torch.cuda.Event | None]:
        """Allocate one worker-global directory generation and consume its fence."""

        self._external_object_version = (self._external_object_version + 1) & 0xFFFFFFFF
        self._external_object_version = self._external_object_version or 1
        prior, self._external_consumer_event = self._external_consumer_event, None
        return self._external_object_version, prior

    def record_external_consumer(self, stream: torch.cuda.Stream) -> None:
        """Publish completion ordering for the next directory replacement."""

        if self._external_consumer_event is not None:
            raise RuntimeError(
                "vLLM external directory was consumed twice without publication"
            )
        event = torch.cuda.Event()
        event.record(stream)
        self._external_consumer_event = event

    def close(self) -> None:
        """Close the runtime after the framework has stopped using the runner."""
        runtime, self._runtime = self._runtime, None
        resources, self._resources = self._resources, None
        self._hook = None
        self._page_size = 0
        self._page_bytes = 0
        self._request_capacity = 0
        self._attention_phases.clear()
        self._attention_workspace_bytes = 0
        self._request_slots_host = None
        self._request_slots_host_numpy = None
        self._request_slots_device = None
        self._layer_ordinals.clear()
        self._physical_destinations.clear()
        self._physical_destinations_prepared = False
        self._external_consumer_event = None
        self._external_object_version = 0
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

    @property
    def request_slots_tensor(self) -> torch.Tensor:
        if self._request_slots_device is None:
            raise RuntimeError("vLLM worker has no published request-slot map")
        return self._request_slots_device


def _controller(runner: Any) -> VllmV1WorkerController:
    controller = getattr(runner, "_nta_vllm_controller", None)
    if controller is None:
        controller = VllmV1WorkerController(runner)
        setattr(runner, "_nta_vllm_controller", controller)
    return controller


def _commit_forward_evidence(state: Any) -> None:
    """Publish direct-launch evidence once the framework forward succeeds."""
    counters = state.commit_direct_evidence()
    native_launches = counters.get("native_decode_launches", 0) + counters.get(
        "native_prefill_launches", 0
    )
    if native_launches:
        if state.hook is None:
            raise RuntimeError("vLLM direct forward completed without an identity hook")
        state.hook.record_native_launch(native_launches)
    VLLM_STATS.update(counters)


class NtaVllmFlashInferMetadataBuilder(FlashInferMetadataBuilder):
    """Let vLLM plan the typed direct wrapper at its native ownership edge."""

    _cudagraph_support = AttentionCGSupport.NEVER

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._nta_direct_prefill_wrapper: Any | None = None
        self._nta_direct_decode_wrapper: Any | None = None

    def build(self, *args: Any, **kwargs: Any) -> Any:
        if os.environ.get("NTA_PROFILE_CPU") != "1":
            return super().build(*args, **kwargs)
        started = time.perf_counter_ns()
        try:
            return super().build(*args, **kwargs)
        finally:
            VLLM_STATS["metadata_build_cpu_ns"] += time.perf_counter_ns() - started
            VLLM_STATS["metadata_build_calls"] += 1

    @staticmethod
    def _direct_batch() -> EngineBatch | None:
        # Differential mode deliberately keeps the stock metadata wrapper so
        # the attention consumer can execute both implementations.  It is a
        # correctness diagnostic and must never contaminate performance data.
        # The request-bound wrapper has no acquisition edge: it is therefore a
        # resident-HBM fast path only. External tiers must preserve FlashInfer's
        # work schedule so their exact transfer dependencies can be attached.
        if (
            os.environ.get("NTA_VLLM_COMPARE_STOCK") == "1"
            or os.environ.get("NTA_SERVING_TIER", "hbm").strip().lower() != "hbm"
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
            raise RuntimeError(
                "vLLM NTA direct prefill does not support DCP or NVFP4"
            )
        if self._nta_direct_prefill_wrapper is None:
            self._nta_direct_prefill_wrapper = _new_request_bound_wrapper(
                "prefill",
                self._get_workspace_buffer(),
                query_dtype=self.q_data_type_prefill,
                kv_dtype=self.kv_cache_spec.dtype,
                head_size=self.head_dim,
            )
        VLLM_STATS["framework_direct_prefill_plans"] += 1
        return self._nta_direct_prefill_wrapper

    def _get_decode_wrapper(
        self, batch_size: int, use_cudagraph: bool = False
    ) -> Any:
        batch = self._direct_batch()
        if batch is None or use_cudagraph:
            return super()._get_decode_wrapper(batch_size, use_cudagraph)
        if self.use_dcp or self.is_kvcache_nvfp4:
            raise RuntimeError(
                "vLLM NTA direct decode does not support DCP or NVFP4"
            )
        if self._nta_direct_decode_wrapper is None:
            self._nta_direct_decode_wrapper = _new_request_bound_wrapper(
                "decode",
                self._get_workspace_buffer(),
                query_dtype=self.q_data_type_decode,
                kv_dtype=self.kv_cache_spec.dtype,
                head_size=self.head_dim,
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
        self._native_enabled = os.environ.get("NTA_VLLM_NATIVE", "0") == "1"
        self._serving_tier = validate_vllm_attention_tier()
        self._profile_cpu = os.environ.get("NTA_PROFILE_CPU") == "1"
        self._verify_semantics = os.environ.get("NTA_VERIFY_EXECUTION") == "1"

    def _request_bound_wrapper(
        self,
        kind: str,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        workspace_bytes: int,
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
        workspace = torch.empty(
            workspace_bytes, dtype=torch.uint8, device=query.device
        )
        return _new_request_bound_wrapper(
            kind,
            workspace,
            query_dtype=query.dtype,
            kv_dtype=kv_cache.dtype,
            head_size=self.head_size,
        )

    @staticmethod
    def _configured_workspace_bytes() -> int:
        return _positive_env(
            "NTA_VLLM_FLASHINFER_WORKSPACE_BYTES",
            int(
                getattr(
                    envs,
                    "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE",
                    64 * 1024 * 1024,
                )
            ),
        )

    def _request_bound_ready(self, state: Any, owner: Any) -> bool:
        """Return whether exact KV is ready for one direct consumer launch.

        HBM is resident by construction.  The host connector can establish the
        same precondition with one typed all-layer batch copy; a host forward
        with admitted pairs but no completion event must retain the incremental
        consumer instead of silently reading an incomplete destination.
        """

        if not isinstance(owner, VllmV1WorkerController):
            return False
        if self._serving_tier == "hbm":
            return True
        if self._serving_tier != "host_staged":
            return False
        pairs = tuple(getattr(state, "host_transfer_pairs", ()))
        event = getattr(state, "host_preload_event", None)
        if event is not None:
            if not pairs:
                raise RuntimeError(
                    "vLLM host preload event has no exact transfer ownership"
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
            os.environ.get("FLASHINFER_WORKSPACE_BASE", ""),
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
        if os.environ.get("NTA_VLLM_COMPARE_STOCK") != "1":
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
                    tuple(int(value) for value in tensor.shape)
                    for tensor in kv_cache
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
    ) -> torch.Tensor:
        started = time.perf_counter_ns() if self._profile_cpu else 0
        request_slots = getattr(state, "request_slots_tensor", None)
        if (
            not isinstance(request_slots, torch.Tensor)
            or request_slots.dtype != torch.int32
            or not request_slots.is_cuda
            or not request_slots.is_contiguous()
            or request_slots.numel() < len(batch.bindings)
        ):
            raise RuntimeError("vLLM direct attention has no typed request-slot map")
        if self._serving_tier == "host_staged":
            state.wait_for_host_preload(torch.cuda.current_stream(query.device))
        kv_cache_for_flashinfer = self._kv_cache_tuple(kv_cache)
        wrapper.run(
            query,
            kv_cache_for_flashinfer,
            state.hook.runtime.device_view_tensor,
            request_slots,
            self.scale,
            out=output,
        )
        self._compare_stock(
            stock_wrapper,
            query,
            kv_cache_for_flashinfer,
            output,
            custom_wrapper=wrapper,
            kind=kind,
        )
        state.record_direct_launch(
            kind,
            len(batch.bindings),
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
        environment_name = (
            "NTA_VLLM_DECODE_MODULE"
            if kind == "decode"
            else "NTA_VLLM_PREFILL_MODULE"
        )
        return os.environ.get(environment_name, _DEFAULT_MODULES[query.dtype])

    def _build_incremental_wrapper(
        self,
        kind: str,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        module_name: str,
        workspace_bytes: int,
    ) -> Any:
        environment_name = (
            "NTA_VLLM_DECODE_MODULE"
            if kind == "decode"
            else "NTA_VLLM_PREFILL_MODULE"
        )
        if os.environ.get(environment_name):
            module_path = _find_module(module_name)
        else:
            module_path = _ensure_default_attention_module(
                module_name, query.dtype, self.head_size
            )
        program = _phase_program(module_path)
        jit_args = attention_jit_args(
            module_name,
            dtype_q=query.dtype,
            dtype_kv=kv_cache.dtype,
            dtype_o=query.dtype,
            idtype=torch.int32,
            head_dim_qk=self.head_size,
            head_dim_vo=self.head_size,
        )
        # Planning initializes every workspace region consumed by FlashInfer;
        # zero-filling hundreds of MiB here is pure startup latency.
        workspace = torch.empty(
            workspace_bytes, dtype=torch.uint8, device=query.device
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
        wrapper._nta_phase_program = program
        wrapper._nta_module_path = module_path
        return wrapper

    def _ensure_local_incremental_wrapper(
        self, kind: str, query: torch.Tensor, kv_cache: torch.Tensor
    ) -> Any:
        attribute = "_nta_wrapper" if kind == "decode" else "_nta_prefill_wrapper"
        wrapper = getattr(self, attribute)
        if wrapper is None:
            module_name = self._incremental_module_name(kind, query, kv_cache)
            wrapper = self._build_incremental_wrapper(
                kind,
                query,
                kv_cache,
                module_name,
                self._configured_workspace_bytes(),
            )
            setattr(self, attribute, wrapper)
            VLLM_STATS["local_incremental_wrapper_builds"] += 1
        self._nta_program = wrapper._nta_phase_program
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
            owner.begin_external_publication() if runs else (0, None)
        )

        requirement_by_run: list[AcquireRequirement] = []
        for slot, run in enumerate(runs):
            object_id = 0x4E54415600000000 + slot
            destination_address = destination.address(
                run.destination_first, run.row_count
            )
            installed_address = runtime.install_registered_nvme_object_async(
                slot,
                object_id,
                version,
                run.source.offset,
                run.source.bytes,
                destination.region,
                destination_address,
                stream,
                prior_consumer if slot == 0 else None,
            )
            if installed_address != destination_address:
                raise RuntimeError("NVMe runtime rebound the vLLM HBM destination")
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
        VLLM_STATS["physical_transfer_runs"] += len(runs)
        VLLM_STATS["physical_transfer_blocks"] += layout.unique_block_count
        VLLM_STATS["physical_transfer_bytes"] += sum(
            run.source.bytes for run in runs
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
        pairs = tuple(getattr(state, "host_transfer_pairs", ()))
        layer_name = getattr(layer, "layer_name", None)
        resources = getattr(state, "host_resources", None)
        if not isinstance(layer_name, str) or not layer_name:
            raise RuntimeError("vLLM host-staged attention has no stable layer name")
        if not isinstance(resources, dict):
            raise RuntimeError("vLLM host-staged attention has no resource directory")
        resource = resources.get(layer_name)
        if not isinstance(resource, IndexedHostResource):
            raise RuntimeError(
                f"vLLM host cache has no typed payload for layer {layer_name!r}"
            )
        source = resource.source_tensor
        destination = resource.destination_tensor
        if (
            not isinstance(source, torch.Tensor)
            or source.is_cuda
            or source.ndim < 2
            or not source.is_contiguous()
        ):
            raise RuntimeError("vLLM host source must own contiguous pinned rows")
        if (
            not isinstance(destination, torch.Tensor)
            or not destination.is_cuda
            or not isinstance(kv_cache, torch.Tensor)
            or not kv_cache.is_cuda
            or int(destination.data_ptr()) != int(kv_cache.data_ptr())
            or int(destination.shape[0]) != int(kv_cache.shape[0])
            or int(kv_cache.stride(0)) * int(kv_cache.element_size())
            != resource.destination_stride_bytes
        ):
            raise RuntimeError(
                "vLLM host resource does not name the numerical KV destination"
            )
        source_by_destination = {
            destination_index: source_index
            for source_index, destination_index in pairs
        }
        if len(source_by_destination) != len(pairs):
            raise RuntimeError("vLLM host transfer destinations are not unique")
        if any(
            source_index < 0
            or source_index >= resource.source_rows
            or destination_index < 0
            or destination_index >= resource.destination_rows
            for destination_index, source_index in source_by_destination.items()
        ):
            raise RuntimeError("vLLM host transfer exceeds source/destination blocks")

        consumed = state.consumed_host_destinations(layer_name)
        work_pairs = []
        selected_pages_by_work = []
        for request_index, kv_tile in zip(
            schedule.request_indices, schedule.kv_tile_indices, strict=True
        ):
            pages = self._physical_pages(
                batch, schedule, int(request_index), int(kv_tile), page_size
            )
            selected_pages_by_work.append(tuple(pages))
            work_pairs.append(
                tuple(
                    (source_by_destination[page], page)
                    for page in pages
                    if page in source_by_destination and page not in consumed
                )
            )
        state.record_host_schedule(
            layer_name,
            int(getattr(schedule, "kv_chunk_tokens", 0)),
            tuple(int(value) for value in schedule.request_indices),
            tuple(int(value) for value in schedule.kv_tile_indices),
            tuple(selected_pages_by_work),
        )
        layout = plan_indexed_dependencies(tuple(work_pairs))
        if len(layout.runs) > runtime.config.object_capacity:
            raise RuntimeError("vLLM host layer exceeds runtime object capacity")
        source_indices: torch.Tensor | None = None
        destination_indices: torch.Tensor | None = None
        if layout.runs:
            device_index = (
                torch.cuda.current_device()
                if kv_cache.device.index is None
                else int(kv_cache.device.index)
            )
            index_key = (
                device_index,
                layout.source_indices,
                layout.destination_indices,
            )
            tensors = state.host_index_tensors.get(index_key)
            if tensors is None:
                tensors = (
                    torch.tensor(
                        layout.source_indices,
                        dtype=torch.int32,
                        device=kv_cache.device,
                    ),
                    torch.tensor(
                        layout.destination_indices,
                        dtype=torch.int32,
                        device=kv_cache.device,
                    ),
                )
                state.host_index_tensors[index_key] = tensors
            source_indices, destination_indices = tensors
        work_count = schedule.work_count
        max_dependencies = int(runtime.config.max_dependencies_per_work_ticket)
        plan = self._ensure_work_plan(
            runtime, work_count, max(1, max_dependencies * work_count)
        )
        version, prior_consumer = (
            owner.begin_external_publication() if layout.runs else (0, None)
        )
        if layout.runs:
            assert source_indices is not None and destination_indices is not None
            transfer_topology = IndexedTransferTopology(
                len(layout.source_indices),
                tuple(
                    IndexedTransferGroup(run.pair_offset, run.row_count)
                    for run in layout.runs
                ),
                tuple(
                    tuple(
                        IndexedWorkDependency(
                            run_index,
                            0,
                            layout.runs[run_index].row_count,
                        )
                        for run_index in run_indices
                    )
                    for run_indices in layout.run_indices_by_work
                ),
            )
            indexed_plan = IndexedHostPlan(
                transfer_topology,
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
                source_indices_device_address=int(source_indices.data_ptr()),
                staging_indices_device_address=int(destination_indices.data_ptr()),
                object_version=version,
                direct_base=runtime.device_view,
                object_id_base=0x4E54414800000000,
            )
            if any(
                span.count > max_dependencies
                for span in indexed_plan.dependency_spans
            ):
                raise RuntimeError(
                    "vLLM host fragmentation exceeds "
                    "NTA_VLLM_MAX_DEPENDENCIES_PER_WORK"
                )
            plan.upload_exact(
                topology,
                indexed_plan.dependency_spans,
                indexed_plan.dependencies,
                stream=torch.cuda.current_stream(),
            )
            runtime.register_indexed_host_plan(
                indexed_plan,
                stream=torch.cuda.current_stream(),
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
                direct_requirement(runtime.device_view, 1)
                for _ in range(work_count)
            ]
            plan.upload_exact(
                topology,
                tuple(
                    WorkDependencySpan(work_id, 1, 1)
                    for work_id in range(work_count)
                ),
                dependencies,
                stream=torch.cuda.current_stream(),
            )
            object_count = 0
        if object_count:
            if self._nta_program is None:
                raise RuntimeError("vLLM host attention has no phase program")
            self._nta_program.validate_indexed_host_range(
                runtime, 0, object_count, torch.cuda.current_stream()
            )
        state.record_host_destinations(layer_name, layout.destination_indices)
        VLLM_STATS["host_transfer_runs"] += object_count
        VLLM_STATS["host_transfer_blocks"] += len(layout.source_indices)
        VLLM_STATS["host_transfer_bytes"] += (
            len(layout.source_indices) * resource.row_bytes
        )
        return plan, object_count, bool(object_count)

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
        VLLM_STATS["work_topology_builds"] += 1
        VLLM_STATS["work_topology_items"] += topology.work_count
        if self._profile_cpu:
            VLLM_STATS["work_topology_cpu_ns"] += (
                time.perf_counter_ns() - topology_started
            )
        execution = None
        verifier = None
        if self._verify_semantics:
            semantic_started = time.perf_counter_ns() if self._profile_cpu else 0
            execution = self._build_plan(
                batch,
                schedule,
                layer=semantic_layer,
            )
            verifier = ExecutionSession.from_plan(execution)
            VLLM_STATS["semantic_plan_builds"] += 1
            VLLM_STATS["semantic_dense_tiles"] += schedule.work_count
            VLLM_STATS["semantic_verifier_sessions"] += 1
            if self._profile_cpu:
                VLLM_STATS["semantic_plan_cpu_ns"] += (
                    time.perf_counter_ns() - semantic_started
                )
        if self._nta_program is None:
            raise RuntimeError("vLLM NTA attention has no validated phase program")
        if batch.exact_demand is None:
            raise RuntimeError("vLLM NTA attention has no exact engine batch")
        if physical:
            tier = getattr(state, "tier_service", None)
            if tier is None or tier.tier.value != self._serving_tier:
                raise RuntimeError(
                    "vLLM forward tier does not match the worker resource owner"
                )
            assert destination is not None
            plan, object_count, has_external_transfer = self._upload_physical_plan(
                state,
                batch,
                schedule,
                topology,
                destination,
                int(state.page_size),
            )
        elif host_staged:
            plan, object_count, has_external_transfer = self._upload_host_plan(
                state,
                batch,
                schedule,
                topology,
                layer,
                kv_cache,
                int(state.page_size),
            )
        else:
            has_external_transfer = False
            plan = self._upload_plan(
                topology,
                schedule,
                state.hook.runtime,
            )

        kv_cache_for_flashinfer = self._kv_cache_tuple(kv_cache)
        if physical and has_external_transfer:
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
        elif host_staged and has_external_transfer:
            epoch = FlashInferLayerEpoch(
                state.hook.runtime,
                plan,
                self._nta_program,
                object_count=object_count,
                max_progress_rounds=1,
                wait_for_plan=False,
            )
            passes = epoch.enqueue_host(
                wrapper,
                query,
                kv_cache_for_flashinfer,
                output,
                progress_blocks=object_count,
                sm_scale=self.scale,
                stream=torch.cuda.current_stream(),
                indexed_host_first_object=0,
                indexed_host_prevalidated=True,
                indexed_host_copy_blocks_per_group=_positive_env(
                    "NTA_VLLM_HOST_COPY_BLOCKS_PER_GROUP", 2
                ),
            )
            if os.environ.get("NTA_VLLM_VERIFY_TRANSFER") == "1":
                epoch.check(passes, torch.cuda.current_stream())
        elif host_staged:
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
                PREACQUIRED_LAUNCH_FLAGS,
                out=output,
            )
        self._compare_stock(
            stock_wrapper,
            query,
            kv_cache_for_flashinfer,
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
                    for request in topology.requests
                    for contributor_index in range(request.work_count)
                )[:32],
            },
        )
        if (physical or host_staged) and has_external_transfer:
            if not isinstance(owner, VllmV1WorkerController):
                raise RuntimeError("vLLM external attention has no worker owner")
            owner.record_external_consumer(torch.cuda.current_stream())
        elif not physical and not host_staged:
            plan.mark_consumed(torch.cuda.current_stream())
        if verifier is not None:
            verifier.record_layer_completion(semantic_layer)
        state.hook.record_native_launch()
        VLLM_STATS[f"native_{kind}_launches"] += 1
        VLLM_STATS[f"physical_{kind}_launches"] += int(physical)
        VLLM_STATS[f"host_{kind}_launches"] += int(host_staged)
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
            )
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
        page_size = int(getattr(state, "page_size", 0) or 0)
        if page_size <= 0:
            raise RuntimeError("vLLM forward sidecar has no token page size")
        owner = getattr(state, "execution_owner", None)
        workspace_bytes = self._configured_workspace_bytes()
        if self._request_bound_ready(state, owner):
            module_name = _REQUEST_BOUND_MODULES.get(query.dtype, "unsupported")
            key = self._phase_signature(
                "prefill",
                module_name,
                query,
                kv_cache,
                page_size=page_size,
                causal=attn_metadata.causal,
                workspace_bytes=workspace_bytes,
            )

            def plan_direct(wrapper: Any) -> None:
                wrapper.plan(
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
                    window_left=self.window_left,
                    logits_soft_cap=self.logits_soft_cap or 0.0,
                    disable_split_kv=False,
                )

            direct_wrapper, _ = owner.attention_phase(
                "request_bound",
                key,
                prefill_batch.epoch,
                lambda: self._request_bound_wrapper(
                    "prefill", query, kv_cache, workspace_bytes
                ),
                plan_direct,
                workspace_bytes=workspace_bytes,
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
            )

        module_name = self._incremental_module_name("prefill", query, kv_cache)

        def plan_incremental(wrapper: Any) -> Any:
            wrapper.plan(
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
                window_left=self.window_left,
                logits_soft_cap=self.logits_soft_cap or 0.0,
                # Preserve FlashInfer's canonical planner choice. NTA binds
                # readiness to the emitted work, not to a different schedule.
                disable_split_kv=False,
            )
            planned = paged_prefill_schedule(wrapper)
            if planned.work_count <= 0:
                raise RuntimeError("vLLM FlashInfer prefill produced no work units")
            return planned

        if isinstance(owner, VllmV1WorkerController):
            key = self._phase_signature(
                "prefill",
                module_name,
                query,
                kv_cache,
                page_size=page_size,
                causal=attn_metadata.causal,
                workspace_bytes=workspace_bytes,
            )
            wrapper, schedule = owner.attention_phase(
                "incremental",
                key,
                prefill_batch.epoch,
                lambda: self._build_incremental_wrapper(
                    "prefill",
                    query,
                    kv_cache,
                    module_name,
                    workspace_bytes,
                ),
                plan_incremental,
                workspace_bytes=workspace_bytes,
            )
            self._nta_program = wrapper._nta_phase_program
        else:
            wrapper = self._ensure_local_incremental_wrapper(
                "prefill", query, kv_cache
            )
            schedule = plan_incremental(wrapper)
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
        if (
            batch.exact_demand is None
            or len(batch.exact_demand.request_unit_ids) != len(batch.bindings)
        ):
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
            )
        indptr = getattr(stock_wrapper, "_paged_kv_indptr_buf", None)
        indices = getattr(stock_wrapper, "_paged_kv_indices_buf", None)
        last_page_len = getattr(stock_wrapper, "_paged_kv_last_page_len_buf", None)
        if not all(
            isinstance(tensor, torch.Tensor)
            for tensor in (indptr, indices, last_page_len)
        ):
            raise RuntimeError(
                "vLLM FlashInfer metadata has no typed paged-KV device buffers"
            )
        if indptr.numel() != attn_metadata.num_decodes + 1:
            raise RuntimeError(
                "vLLM FlashInfer page indptr has the wrong request count"
            )
        if any(
            tensor.dtype != torch.int32
            or not tensor.is_cuda
            or not tensor.is_contiguous()
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

        owner = getattr(state, "execution_owner", None)
        workspace_bytes = self._configured_workspace_bytes()
        if self._request_bound_ready(state, owner):
            module_name = _REQUEST_BOUND_MODULES.get(query.dtype, "unsupported")
            key = self._phase_signature(
                "decode",
                module_name,
                query,
                kv_cache,
                page_size=page_size,
                causal=False,
                workspace_bytes=workspace_bytes,
            )

            def plan_direct(wrapper: Any) -> None:
                wrapper.plan(
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
                    window_left=self.window_left,
                    logits_soft_cap=self.logits_soft_cap or 0.0,
                    disable_split_kv=False,
                )

            direct_wrapper, _ = owner.attention_phase(
                "request_bound",
                key,
                decode_batch.epoch,
                lambda: self._request_bound_wrapper(
                    "decode", query, kv_cache, workspace_bytes
                ),
                plan_direct,
                workspace_bytes=workspace_bytes,
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
            )

        module_name = self._incremental_module_name("decode", query, kv_cache)

        def plan_incremental(wrapper: Any) -> Any:
            wrapper.plan(
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
                window_left=self.window_left,
                logits_soft_cap=self.logits_soft_cap or 0.0,
                disable_split_kv=False,
            )
            planned = decode_schedule(wrapper)
            if planned.work_count <= 0:
                raise RuntimeError("vLLM FlashInfer decode produced no work units")
            return planned

        if isinstance(owner, VllmV1WorkerController):
            key = self._phase_signature(
                "decode",
                module_name,
                query,
                kv_cache,
                page_size=page_size,
                causal=False,
                workspace_bytes=workspace_bytes,
            )
            wrapper, schedule = owner.attention_phase(
                "incremental",
                key,
                decode_batch.epoch,
                lambda: self._build_incremental_wrapper(
                    "decode", query, kv_cache, module_name, workspace_bytes
                ),
                plan_incremental,
                workspace_bytes=workspace_bytes,
            )
            self._nta_program = wrapper._nta_phase_program
        else:
            wrapper = self._ensure_local_incremental_wrapper(
                "decode", query, kv_cache
            )
            schedule = plan_incremental(wrapper)
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
            if (
                self._serving_tier != "hbm"
                or os.environ.get("NTA_VLLM_ALLOW_STOCK_FALLBACK") != "1"
            ):
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
