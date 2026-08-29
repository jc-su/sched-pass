"""Setup-time ownership for typed vLLM FlashInfer modules."""

from __future__ import annotations

import atexit
from collections import Counter
import contextlib
import pathlib
import threading
from typing import Any

import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)
from vllm import envs
from vllm.v1.attention.backends.utils import get_kv_cache_layout

from nta_runtime.engines.vllm_config import VllmAttentionConfig
from nta_runtime.flashinfer import (
    mapped_request_bound_attention_jit_args,
)
from nta_runtime.runtime import (
    JitOperatorModule,
    JitPhaseProgram,
    OperatorAccessProof,
    OperatorCapability,
    OperatorCoordinateMap,
    OperatorDemandBinding,
    OperatorFamily,
    OperatorForm,
    OperatorIdentityBinding,
    OperatorInstrumentation,
    OperatorPartialState,
    OperatorPlanFlag,
    OperatorReduction,
)
from nta_runtime.transport_program import load_activated_transport_program


_DEFAULT_MODULES = {
    # FlashInfer's tensor-core decode wrapper consumes a paged-prefill JIT
    # module for its FA2 plan/run interface.
    torch.float16: "nta_batch_prefill_default_v2_hooked",
    torch.bfloat16: "nta_batch_prefill_default_v2_hooked_bf16",
}
_REQUEST_BOUND_MODULES = {
    torch.float16: "nta_batch_prefill_vllm_request_bound_v3_binding_fp16",
    torch.bfloat16: "nta_batch_prefill_vllm_request_bound_v3_binding_bf16",
}
_MODULE_LOCK = threading.Lock()
_OPERATOR_MODULES: dict[pathlib.Path, JitOperatorModule] = {}
_TRANSPORT_PROGRAM: JitPhaseProgram | None = None
VLLM_STATS: Counter[str] = Counter()


def _default_workspace_bytes() -> int:
    return int(
        getattr(
            envs,
            "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE",
            64 * 1024 * 1024,
        )
    )


def _find_module(name: str, workspace: pathlib.Path) -> pathlib.Path:
    matches = tuple(workspace.rglob(f"{name}.so"))
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
    workspace: pathlib.Path,
    request_bound: bool = False,
    mapped_request_slots: bool = False,
) -> pathlib.Path:
    """Build the pinned NTA tensor-core module during backend initialization."""
    if head_size != 128:
        raise RuntimeError(
            "native vLLM NTA attention currently requires head_size=128"
        )
    matches = tuple(workspace.rglob(f"{name}.so"))
    if not matches:
        from flashinfer.jit.attention.modules import gen_customize_batch_prefill_module

        if mapped_request_slots:
            if not request_bound:
                raise RuntimeError("mapped request slots require a direct module")
            tensor_names = ["nta_runtime", "nta_request_bindings"]
            tensor_dtypes = ["uint8_t", "int64_t"]
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
        matches = tuple(workspace.rglob(f"{name}.so"))
    if len(matches) != 1:
        raise RuntimeError(
            f"NTA vLLM native attention expected one {name}.so in {workspace}, "
            f"found {len(matches)}"
        )
    return matches[0].resolve()


def _prepare_attention_modules(
    config: VllmAttentionConfig,
    query_dtypes: tuple[torch.dtype, ...],
    head_size: int,
) -> None:
    """Materialize every module that the selected profile can request."""

    if not config.native_enabled:
        return
    workspace = config.require_workspace()
    dtypes = tuple(dict.fromkeys(query_dtypes))
    for dtype in dtypes:
        if dtype not in _DEFAULT_MODULES:
            raise RuntimeError(
                "native vLLM NTA attention requires float16 or bfloat16 queries"
            )

    needs_direct = config.serving_tier in {"hbm", "host_staged"}
    needs_incremental = config.serving_tier in {"host_staged", "nvme"}
    for dtype in dtypes:
        if needs_direct:
            _ensure_default_attention_module(
                _REQUEST_BOUND_MODULES[dtype],
                dtype,
                head_size,
                workspace=workspace,
                request_bound=True,
                mapped_request_slots=True,
            )
        if not needs_incremental:
            continue
        prepared_names: set[str] = set()
        for kind in ("decode", "prefill"):
            override = config.module_override(kind)
            module_name = override or _DEFAULT_MODULES[dtype]
            if module_name in prepared_names:
                continue
            prepared_names.add(module_name)
            if override is not None:
                _find_module(module_name, workspace)
            else:
                _ensure_default_attention_module(
                    module_name,
                    dtype,
                    head_size,
                    workspace=workspace,
                )


def _operator_module(path: pathlib.Path) -> JitOperatorModule:
    with _MODULE_LOCK:
        existing = _OPERATOR_MODULES.get(path)
        if existing is not None:
            return existing
        module = JitOperatorModule(path)
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
        module.operator_contract.require(
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
        module.operator_plan.require(
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
        _OPERATOR_MODULES[path] = module
        VLLM_STATS["verified_operator_modules"] += 1
        return module


def _transport_program() -> JitPhaseProgram:
    global _TRANSPORT_PROGRAM
    with _MODULE_LOCK:
        if _TRANSPORT_PROGRAM is None:
            program, path, _digest = load_activated_transport_program()
            _TRANSPORT_PROGRAM = program
            VLLM_STATS["transport_program_loads"] += 1
            VLLM_STATS["transport_program_bytes"] = path.stat().st_size
        return _TRANSPORT_PROGRAM


def _new_request_bound_wrapper(
    kind: str,
    workspace: torch.Tensor,
    *,
    query_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    head_size: int,
    workspace_base: pathlib.Path,
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
    module_path = _find_module(module_name, workspace_base)
    _operator_module(module_path)
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
    wrapper._nta_request_bound = True
    VLLM_STATS["framework_direct_wrapper_builds"] += 1
    return wrapper


@atexit.register
def _close_native_modules() -> None:
    global _TRANSPORT_PROGRAM
    for module in tuple(_OPERATOR_MODULES.values()):
        with contextlib.suppress(Exception):
            module.close()
    _OPERATOR_MODULES.clear()
    if _TRANSPORT_PROGRAM is not None:
        with contextlib.suppress(Exception):
            _TRANSPORT_PROGRAM.close()
        _TRANSPORT_PROGRAM = None


__all__ = [
    "VLLM_STATS",
    "_DEFAULT_MODULES",
    "_REQUEST_BOUND_MODULES",
    "_default_workspace_bytes",
    "_find_module",
    "_new_request_bound_wrapper",
    "_operator_module",
    "_transport_program",
    "_prepare_attention_modules",
]
