"""Owned FlashInfer kernel resources for the SGLang engine adapter.

This module is the composition boundary between SGLang's Python backend and
the compiler-produced NTA modules.  It owns wrapper construction, module
lookup, ABI verification, phase-program caches, and teardown.  It deliberately
depends only on immutable construction inputs and a statistics sink; it never
retains the SGLang backend that uses it.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
import time
from typing import Any

import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)
from flashinfer.decode import (
    gen_customize_batch_decode_module,
    get_batch_decode_jit_module,
)
from flashinfer.prefill import (
    gen_customize_batch_prefill_module,
    get_batch_prefill_jit_module,
)

from nta_runtime.engines.sglang_planning import dtype_tag
from nta_runtime.flashinfer import attention_jit_args
from nta_runtime.flashinfer_jit import (
    FlashInferMaterializationOrigin,
    FlashInferMaterializationProvenance,
    materialize_typed_flashinfer_module,
)
from nta_runtime.runtime import (
    JitOperatorModule,
    JitPhaseProgram,
    OperatorAccessProof,
    OperatorCapability,
    OperatorContract,
    OperatorCoordinateMap,
    OperatorDemandBinding,
    OperatorFamily,
    OperatorForm,
    OperatorIdentityBinding,
    OperatorInstrumentation,
    OperatorPartialState,
    OperatorPlan,
    OperatorPlanFlag,
    OperatorReduction,
)
from nta_runtime.transport_program import load_activated_transport_program


_ATTENTION_TENSOR_ABI = (
    "nta_runtime",
    "nta_work_items",
    "nta_dependencies",
)


@dataclass(frozen=True, slots=True)
class SglangWrapperSet:
    """One complete, immutable selection of SGLang FlashInfer wrappers."""

    decode: tuple[Any, ...]
    prefill_paged: tuple[Any, ...]
    prefill_verify: tuple[Any, ...]

    @classmethod
    def capture(
        cls,
        *,
        decode: list[Any] | tuple[Any, ...],
        prefill_paged: list[Any] | tuple[Any, ...],
        prefill_verify: list[Any] | tuple[Any, ...],
    ) -> SglangWrapperSet:
        return cls(tuple(decode), tuple(prefill_paged), tuple(prefill_verify))


@dataclass(frozen=True, slots=True)
class SglangKernelConfig:
    """Construction inputs that determine the instrumented module identity."""

    dtype_q: torch.dtype
    dtype_kv: torch.dtype
    head_dim: int
    num_wrappers: int
    skip_prefill: bool
    decode_use_tensor_cores: bool
    stream_ordered_retirement: bool
    workspace_buffer: torch.Tensor


@dataclass(frozen=True, slots=True)
class SglangKernelContractReport:
    """Serializable snapshot of every ABI contract accepted by the owner."""

    operator_contracts: tuple[dict[str, Any], ...]
    operator_plans: tuple[dict[str, Any], ...]
    transport_contract: dict[str, Any] | None
    verified_dual_form_operator_plans: int


class SglangKernelResources:
    """Own instrumented wrappers and their verified native programs.

    The owner is intentionally compositional: callers pass the runtime and
    stream only to the setup operation that needs them.  No method reaches
    back into the SGLang backend.
    """

    def __init__(
        self,
        *,
        config: SglangKernelConfig,
        stock_wrappers: SglangWrapperSet,
        stats: MutableMapping[str, Any],
    ) -> None:
        self._config: SglangKernelConfig | None = config
        self._stock_wrappers = stock_wrappers
        self._stats: MutableMapping[str, Any] | None = stats
        self._wrapper_modules: dict[int, str] = {}
        self._loaded_jit_modules: dict[str, Any] = {}
        self._typed_materializations: dict[
            str, FlashInferMaterializationProvenance
        ] = {}
        self._typed_decode_wrappers: tuple[Any, ...] | None = None
        self._typed_prefill_wrappers: tuple[tuple[Any, ...], tuple[Any, ...]] | None = (
            None
        )
        self._operator_modules: dict[str, JitOperatorModule] = {}
        self._transport_program: JitPhaseProgram | None = None
        self._operator_contracts: dict[OperatorFamily, OperatorContract] = {}
        self._operator_plans: dict[OperatorFamily, OperatorPlan] = {}
        self._closed = False

    def _require_config(self) -> SglangKernelConfig:
        if self._closed or self._config is None:
            raise RuntimeError("SGLang kernel resources are closed")
        return self._config

    def _require_stats(self) -> MutableMapping[str, Any]:
        if self._closed or self._stats is None:
            raise RuntimeError("SGLang kernel resources are closed")
        return self._stats

    @property
    def stock_wrappers(self) -> SglangWrapperSet:
        self._require_config()
        return self._stock_wrappers

    def typed_wrappers(self) -> SglangWrapperSet:
        config = self._require_config()
        decode = self._instrumented_decode()
        if config.skip_prefill:
            return SglangWrapperSet(decode, (), ())
        prefill_paged, prefill_verify = self._instrumented_prefill()
        return SglangWrapperSet(decode, prefill_paged, prefill_verify)

    def is_instrumented(self, wrapper: Any) -> bool:
        self._require_config()
        return id(wrapper) in self._wrapper_modules

    def module_name(self, wrapper: Any) -> str:
        self._require_config()
        try:
            return self._wrapper_modules[id(wrapper)]
        except KeyError as error:
            raise RuntimeError(
                "FlashInfer wrapper is not owned by the instrumented kernel set"
            ) from error

    def describe_wrapper_id(self, wrapper_id: int) -> str:
        self._require_config()
        return self._wrapper_modules.get(wrapper_id, str(wrapper_id))

    def prepare_typed_execution_modules(
        self,
        *,
        runtime: Any,
        host_staged: bool,
        stream: torch.cuda.Stream,
    ) -> None:
        """Build and validate every request-capable module before admission."""

        config = self._require_config()
        stats = self._require_stats()
        started = time.perf_counter_ns()
        loaded: set[str] = set()
        wrappers = self.typed_wrappers()
        wrapper_groups = [wrappers.decode]
        if not config.skip_prefill:
            wrapper_groups.extend((wrappers.prefill_paged, wrappers.prefill_verify))
        for group in wrapper_groups:
            if not group:
                continue
            wrapper = group[0]
            module_name = self.module_name(wrapper)
            if module_name in loaded:
                continue
            self.operator_module(wrapper)
            loaded.add(module_name)
        transport = self.transport_program()
        if host_staged:
            transport.warmup_indexed_host_validation(runtime, stream)
        stream.synchronize()
        materializations = tuple(
            self._typed_materializations[module_name] for module_name in sorted(loaded)
        )
        if len(materializations) != len(loaded):
            raise RuntimeError("typed startup lacks exact JIT materialization provenance")
        cold_builds = sum(
            item.origin is FlashInferMaterializationOrigin.COLD_BUILD_OWNER
            for item in materializations
        )
        stats["typed_startup_precompiled"] = cold_builds == 0
        stats["typed_startup_cold_builds"] = cold_builds
        stats["typed_startup_modules"] = len(loaded)
        stats["typed_startup_ns"] = time.perf_counter_ns() - started

    def _jit_arguments(self, name: str) -> list[Any]:
        config = self._require_config()
        return attention_jit_args(
            name,
            dtype_q=config.dtype_q,
            dtype_kv=config.dtype_kv,
            dtype_o=config.dtype_q,
            idtype=torch.int32,
            head_dim_qk=config.head_dim,
            head_dim_vo=config.head_dim,
        )

    @staticmethod
    def _require_attention_tensor_abi(jit_args: list[Any]) -> None:
        if tuple(jit_args[7]) != _ATTENTION_TENSOR_ABI:
            raise RuntimeError("prebuilt attention module has unexpected tensor ABI")

    def _materialize_attention_module(
        self,
        name: str,
        jit_args: list[Any],
        *,
        decode: bool,
        use_tensor_cores: bool = False,
    ) -> Any:
        """Materialize and bind one exact, content-addressed JIT artifact."""

        self._require_attention_tensor_abi(jit_args)
        cached = self._loaded_jit_modules.get(name)
        if cached is not None:
            return cached
        spec = (
            gen_customize_batch_decode_module(*jit_args)
            if decode and not use_tensor_cores
            else gen_customize_batch_prefill_module("fa2", *jit_args)
        )
        materialized = materialize_typed_flashinfer_module(spec)
        stats = self._require_stats()
        raw_module = materialized.module
        if decode and not use_tensor_cores:
            module = get_batch_decode_jit_module(name, raw_module)
        else:
            module = get_batch_prefill_jit_module(name, raw_module)
        self._loaded_jit_modules[name] = module
        provenance = materialized.provenance
        self._typed_materializations[name] = provenance
        origin_counter = {
            FlashInferMaterializationOrigin.COLD_BUILD_OWNER: "typed_module_cold_builds",
            FlashInferMaterializationOrigin.DISK_CACHE_LOAD: "typed_module_disk_loads",
            FlashInferMaterializationOrigin.PROCESS_CACHE_HIT: (
                "typed_module_process_cache_hits"
            ),
        }[provenance.origin]
        stats[origin_counter] = stats.get(origin_counter, 0) + 1
        stats["typed_module_build_ns"] = stats.get("typed_module_build_ns", 0) + (
            provenance.build_ns
        )
        stats["typed_module_load_ns"] = stats.get("typed_module_load_ns", 0) + (
            provenance.load_ns
        )
        stats["typed_module_lock_wait_ns"] = stats.get(
            "typed_module_lock_wait_ns", 0
        ) + provenance.lock_wait_ns
        stats["typed_module_bytes"] = stats.get("typed_module_bytes", 0) + (
            provenance.artifact_bytes
        )
        stats["typed_module_materializations"] = [
            self._typed_materializations[key].as_dict()
            for key in sorted(self._typed_materializations)
        ]
        return module

    @staticmethod
    def _bind_prebuilt_attention_module(
        wrapper: Any, module: Any, jit_args: list[Any]
    ) -> None:
        SglangKernelResources._require_attention_tensor_abi(jit_args)
        wrapper._jit_module = module
        wrapper._jit_additional_tensor_names = list(_ATTENTION_TENSOR_ABI)

    def _instrumented_decode(self) -> tuple[Any, ...]:
        if self._typed_decode_wrappers is not None:
            return self._typed_decode_wrappers
        config = self._require_config()
        signature = (
            f"h{config.head_dim}_{dtype_tag(config.dtype_q)}_"
            f"{dtype_tag(config.dtype_kv)}"
        )
        variant = "tc" if config.decode_use_tensor_cores else "cc"
        name = (
            f"nta_sglang_decode_stream_ordered_v1_demand_acquire_{variant}_{signature}"
            if config.stream_ordered_retirement
            else f"nta_sglang_decode_demand_acquire_v11_{variant}_{signature}"
        )
        args = self._jit_arguments(name)
        prebuilt = self._materialize_attention_module(
            name,
            args,
            decode=True,
            use_tensor_cores=config.decode_use_tensor_cores,
        )
        wrappers = tuple(
            BatchDecodeWithPagedKVCacheWrapper(
                config.workspace_buffer,
                "NHD",
                backend="fa2",
                use_tensor_cores=config.decode_use_tensor_cores,
                jit_args=None,
            )
            for _ in range(config.num_wrappers)
        )
        for wrapper in wrappers:
            self._bind_prebuilt_attention_module(wrapper, prebuilt, args)
        for wrapper in wrappers:
            self._wrapper_modules[id(wrapper)] = name
        self._typed_decode_wrappers = wrappers
        return wrappers

    def _instrumented_prefill(
        self,
    ) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        if self._typed_prefill_wrappers is not None:
            return self._typed_prefill_wrappers
        config = self._require_config()
        signature = (
            f"h{config.head_dim}_{dtype_tag(config.dtype_q)}_"
            f"{dtype_tag(config.dtype_kv)}"
        )
        name = (
            f"nta_sglang_prefill_stream_ordered_v1_demand_acquire_{signature}"
            if config.stream_ordered_retirement
            else f"nta_sglang_prefill_demand_acquire_v11_{signature}"
        )
        args = self._jit_arguments(name)
        prebuilt = self._materialize_attention_module(name, args, decode=False)
        wrappers = tuple(
            BatchPrefillWithPagedKVCacheWrapper(
                config.workspace_buffer,
                "NHD",
                backend="fa2",
                jit_args=None,
            )
            for _ in range(2 * config.num_wrappers)
        )
        for wrapper in wrappers:
            self._bind_prebuilt_attention_module(wrapper, prebuilt, args)
        for wrapper in wrappers:
            self._wrapper_modules[id(wrapper)] = name
        split = config.num_wrappers
        result = (wrappers[:split], wrappers[split:])
        self._typed_prefill_wrappers = result
        return result

    def operator_module(self, wrapper: Any) -> JitOperatorModule:
        """Load and verify one numerical module without transport coupling."""

        config = self._require_config()
        stats = self._require_stats()
        module_name = self.module_name(wrapper)
        cached = self._operator_modules.get(module_name)
        if cached is not None:
            return cached
        provenance = self._typed_materializations.get(module_name)
        if provenance is None:
            raise RuntimeError(
                "typed wrapper lacks exact FlashInfer materialization provenance"
            )
        module_path = provenance.library_path
        if not module_path.is_file() or module_path.name != f"{module_name}.so":
            raise RuntimeError(
                "typed FlashInfer materialization no longer names its exact artifact"
            )
        module = JitOperatorModule(module_path)
        family = (
            OperatorFamily.FLASHINFER_DECODE
            if "decode" in module_name
            else OperatorFamily.FLASHINFER_PAGED_PREFILL
        )
        try:
            required = (
                OperatorCapability.REQUEST_BINDING
                | OperatorCapability.TYPED_FLASHINFER_FRONTEND
                | OperatorCapability.OBJECT_DEPENDENCIES
                | OperatorCapability.FINITE_DEFERRAL
                | OperatorCapability.PARTIAL_PUBLICATION
                | OperatorCapability.COMPLETE_CONTRIBUTOR_MERGE
                | OperatorCapability.RUNNABLE_COMPACTION
            )
            if config.stream_ordered_retirement:
                required |= OperatorCapability.STREAM_ORDERED_RETIREMENT
            contract = module.operator_contract
            contract.require(
                family=family,
                form=OperatorForm.INCREMENTAL,
                capabilities=required,
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
            if (
                bool(
                    contract.capabilities & OperatorCapability.STREAM_ORDERED_RETIREMENT
                )
                != config.stream_ordered_retirement
            ):
                raise RuntimeError(
                    "typed module stream-ordered retirement capability does not "
                    "match the selected execution contract"
                )
            plan = module.operator_plan
            plan.require(
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
            previous_contract = self._operator_contracts.get(family)
            previous_plan = self._operator_plans.get(family)
            if (previous_contract is not None and previous_contract != contract) or (
                previous_plan is not None and previous_plan != plan
            ):
                raise RuntimeError(
                    "SGLang loaded incompatible typed modules for one operator family"
                )
        except BaseException:
            module.close()
            raise
        self._operator_contracts[family] = contract
        self._operator_plans[family] = plan
        stats["verified_operator_modules"] += 1
        self._operator_modules[module_name] = module
        return module

    def transport_program(self) -> JitPhaseProgram:
        """Return the runtime-owned, operator-independent transport phases."""

        self._require_config()
        stats = self._require_stats()
        if self._transport_program is not None:
            return self._transport_program
        program, path, expected_digest = load_activated_transport_program()
        self._transport_program = program
        stats["transport_program_loaded"] = True
        stats["transport_program_bytes"] = path.stat().st_size
        stats["transport_program_sha256"] = expected_digest.lower()
        return program

    def contract_report(self) -> SglangKernelContractReport:
        self._require_config()
        contracts = sorted(
            self._operator_contracts.values(),
            key=lambda contract: (int(contract.family), int(contract.form)),
        )
        operator_contracts = tuple(
            {
                "schema_version": contract.schema_version,
                "runtime_abi_version": contract.runtime_abi_version,
                "family": contract.family.name.lower(),
                "form": contract.form.name.lower(),
                "capabilities": int(contract.capabilities),
                "instrumentation_flags": int(contract.instrumentation_flags),
                "identity_binding": contract.identity_binding.name.lower(),
                "demand_binding": contract.demand_binding.name.lower(),
                "access_proof": contract.access_proof.name.lower(),
                "granularity_bytes": contract.granularity_bytes,
                "tier_mask": contract.tier_mask,
                "source_fingerprint": contract.source_fingerprint,
            }
            for contract in contracts
        )
        plans = sorted(
            self._operator_plans.values(),
            key=lambda plan: (int(plan.family), plan.plan_fingerprint),
        )
        operator_plans = tuple(
            {
                "schema_version": plan.schema_version,
                "runtime_abi_version": plan.runtime_abi_version,
                "family": plan.family.name.lower(),
                "supported_forms": plan.supported_forms,
                "coordinate_map": plan.coordinate_map.name.lower(),
                "partial_state": plan.partial_state.name.lower(),
                "reduction": plan.reduction.name.lower(),
                "flags": int(plan.flags),
                "source_fingerprint": plan.source_fingerprint,
                "plan_fingerprint": plan.plan_fingerprint,
            }
            for plan in plans
        )
        transport_contract = None
        if self._transport_program is not None:
            contract = self._transport_program.operator_contract
            plan = self._transport_program.operator_plan
            transport_contract = {
                "schema_version": contract.schema_version,
                "runtime_abi_version": contract.runtime_abi_version,
                "family": contract.family.name.lower(),
                "form": contract.form.name.lower(),
                "capabilities": int(contract.capabilities),
                "instrumentation_flags": int(contract.instrumentation_flags),
                "tier_mask": contract.tier_mask,
                "supported_forms": plan.supported_forms,
                "flags": int(plan.flags),
                "source_fingerprint": contract.source_fingerprint,
                "plan_fingerprint": plan.plan_fingerprint,
            }
        verified_dual_form = sum(
            plan.supports(OperatorForm.DIRECT)
            and plan.supports(OperatorForm.INCREMENTAL)
            for plan in plans
        )
        return SglangKernelContractReport(
            operator_contracts=operator_contracts,
            operator_plans=operator_plans,
            transport_contract=transport_contract,
            verified_dual_form_operator_plans=verified_dual_form,
        )

    def close(self) -> tuple[BaseException, ...]:
        """Close all native programs while preserving fail-complete teardown."""

        if self._closed:
            return ()
        errors: list[BaseException] = []
        for module in tuple(self._operator_modules.values()):
            try:
                module.close()
            except BaseException as error:
                errors.append(error)
        self._operator_modules.clear()
        if self._transport_program is not None:
            try:
                self._transport_program.close()
            except BaseException as error:
                errors.append(error)
            self._transport_program = None
        self._wrapper_modules.clear()
        self._loaded_jit_modules.clear()
        self._typed_materializations.clear()
        self._typed_decode_wrappers = None
        self._typed_prefill_wrappers = None
        self._operator_contracts.clear()
        self._operator_plans.clear()
        self._stock_wrappers = SglangWrapperSet((), (), ())
        self._config = None
        self._stats = None
        self._closed = True
        return tuple(errors)
