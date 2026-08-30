"""SGLang 0.5.16 adapter for compiler-instrumented FlashInfer attention."""

from __future__ import annotations

import atexit
from dataclasses import replace
import math
import os
import time
from typing import Any

import torch
from flashinfer import BatchDecodeWithPagedKVCacheWrapper
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.memory_pool import KVWriteLoc

from nta_runtime.flashinfer import adopt_planned_flashinfer_state
from nta_runtime.flashinfer_schedule import (
    decode_schedule,
    paged_prefill_schedule,
    require_supported_version,
)
from nta_runtime.hbm_registration import HbmDestinationSlice
from nta_runtime.indexed_transfer_torch import warm_indexed_tensor_mover
from nta_runtime.adapters.sglang import (
    SglangAdapter,
    validate_sglang_attention_tier,
)
from nta_runtime.execution_core import ExecutionSession
from nta_runtime.execution_protocol import ProtocolKind
from nta_runtime.execution_planner import (
    HostExecutionForm,
    HostExecutionMode,
    HostExecutionPlan,
)
from nta_runtime.requests import RequestBinding
from nta_runtime.engines.sglang_hicache import (
    PendingHostLoad,
    SglangHiCacheBridge,
)
from nta_runtime.engines.sglang_acquisition import (
    SglangHostAcquisitionCoordinator,
)
from nta_runtime.engines.sglang_transfer import HostMoverController
from nta_runtime.engines.sglang_pipeline import SglangHostTransport
from nta_runtime.engines.sglang_nvme import SglangNvmeAcquisitionPipeline
from nta_runtime.engines.sglang_acquisition_contract import (
    AcquisitionTier,
    HostForwardAcquisition,
)
from nta_runtime.engines.sglang_config import (
    AUTO_INCREMENTAL_INITIALIZATION_PROBES,
    SglangBootstrapConfig,
    SglangExecutionTuning,
)
from nta_runtime.engines.sglang_metadata import SglangMetadataPlanner
from nta_runtime.engines.sglang_lifecycle import SglangForwardLifecycle
from nta_runtime.engines.sglang_state import (
    SglangForwardEpoch,
    SglangForwardPlan,
    _BarrierProfile,
    _FragmentLookahead,
    _OperatorProfile,
)
from nta_runtime.engines.sglang_calibration import (
    SglangConsumerPolicyCalibration,
    SglangLayerServiceCalibration,
)
from nta_runtime.engines.sglang_graphs import DemandGraphCache
from nta_runtime.engines.sglang_execution import (
    AttentionDispatchKind,
    DeadlineFragment,
    SglangAttentionExecutionConfig,
    SglangAttentionExecutor,
    select_attention_dispatch,
    use_preloaded_stock_alias,
)
from nta_runtime.engines.sglang_kernels import (
    SglangKernelConfig,
    SglangKernelResources,
    SglangWrapperSet,
)
from nta_runtime.engines.sglang_materialization import SglangPlanMaterializer
from nta_runtime.engines.sglang_verification import SglangAttentionVerifier
from nta_runtime.engines.sglang_semantics import (
    build_execution_plan,
)
from nta_runtime.engines.sglang_planning import (
    require_exact_prefetch_layers as _require_exact_prefetch_layers,
)
from nta_runtime.engines.sglang_telemetry import (
    SglangTelemetryConfig,
    StatsPublisher,
    consumer_contract_for_stats as _consumer_contract_for_stats,
    flag_value as _flag_value,
    initial_engine_stats,
    process_hook_stats,
)
from nta_runtime.runtime_resources import (
    RuntimeResourceConfig,
    ServingRuntimeResources,
)
from nta_runtime.tier import ServingTierConfig
from nta_runtime.runtime import TierKind


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
        # Register cleanup state before any operation that may fail.  The
        # backend constructor opens the selected tier before creating several
        # CUDA streams and graph-side objects; a later configuration error
        # must still release that owner when Python destroys the partial
        # object.
        self._resources: ServingRuntimeResources | None = None
        self._materializer: SglangPlanMaterializer | None = None
        self._kernels: SglangKernelResources | None = None
        self._attention_executor: SglangAttentionExecutor | None = None
        self._attention_verifier: SglangAttentionVerifier | None = None
        self._resources_closed = True
        self._closed = True
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

        # Keep the stock wrappers as an explicit resident reference.  NTA's
        # typed work-unit kernel is needed only when a forward contains an
        # external tier dependency; routing resident-only forwards through the
        # framework wrapper prevents instrumentation overhead from becoming a
        # regression for requests that do not exercise the mechanism.
        stock_wrappers = SglangWrapperSet.capture(
            decode=self.decode_wrappers,
            prefill_paged=self.prefill_wrappers_paged,
            prefill_verify=self.prefill_wrappers_verify,
        )

        self._hicache_enabled = bool(model_runner.server_args.enable_hierarchical_cache)
        self._model_runner = model_runner

        request_capacity = int(model_runner.req_to_token_pool.req_to_token.shape[0])
        bootstrap = SglangBootstrapConfig.from_environment(request_capacity)
        self._work_ticket_capacity = bootstrap.work_ticket_capacity
        self._max_dependencies_per_work_ticket = (
            bootstrap.max_dependencies_per_work_ticket
        )
        self._object_capacity = bootstrap.object_capacity
        self._tenant_capacity = bootstrap.tenant_capacity
        self._tenant_isolation_enabled = bootstrap.tenant_isolation_enabled
        self._execution_config = bootstrap.execution
        resources: ServingRuntimeResources | None = None
        try:
            tier_config = ServingTierConfig.from_environment()
            validate_sglang_attention_tier({"NTA_SERVING_TIER": tier_config.tier.value})
            resources = ServingRuntimeResources.open(
                tier_config=tier_config,
                runtime_config=RuntimeResourceConfig.with_environment_staging_limit(
                    request_capacity=request_capacity,
                    object_capacity=self._object_capacity,
                    intent_capacity=self._object_capacity,
                    work_ticket_capacity=self._work_ticket_capacity,
                    max_dependencies_per_work_ticket=(
                        self._max_dependencies_per_work_ticket
                    ),
                    device_ordinal=torch.cuda.current_device(),
                    tenant_capacity=self._tenant_capacity,
                ),
            )
            if resources.tier.is_host_staged and not self._hicache_enabled:
                raise RuntimeError(
                    "host_staged requires SGLang hierarchical-cache host payloads"
                )
            if resources.tier.is_hbm and self._hicache_enabled:
                raise RuntimeError(
                    "an HBM profile cannot label hierarchical-cache host transfers; "
                    "select host_staged or disable hierarchical cache"
                )
            if resources.tier.is_physical and not self._hicache_enabled:
                raise RuntimeError(
                    "a physical serving tier requires SGLang hierarchical cache metadata"
                )
            if (
                resources.tier.is_physical
                and getattr(model_runner.server_args, "hicache_storage_backend", None)
                != "dynamic"
            ):
                raise RuntimeError(
                    "a physical serving tier requires SGLang's dynamic "
                    "NtaSglangStorage stable-key connector"
                )
            if resources.tier.is_physical:
                catalog = resources.tier.catalog
                if catalog is None:
                    raise RuntimeError("a physical serving tier has no storage catalog")
                from nta_runtime.connectors.sglang_storage import (
                    validate_sglang_storage_backend,
                )

                validate_sglang_storage_backend(
                    model_runner.server_args,
                    expected_namespace=catalog.namespace,
                )
                if bootstrap.allow_load_fallback:
                    raise RuntimeError(
                        "physical metadata-only storage cannot fall back to a host "
                        "payload transfer"
                    )
        except (OSError, ValueError, RuntimeError) as error:
            if resources is not None:
                resources.close()
            raise RuntimeError(
                f"invalid NTA serving tier configuration: {error}"
            ) from error
        if resources is None:  # pragma: no cover - guarded by open()
            raise RuntimeError("serving runtime resources were not initialized")
        self._resources = resources
        self._tier_service = resources.tier
        self._runtime = resources.runtime
        self._closed = False
        self._resources_closed = False
        if (
            self._tenant_isolation_enabled
            and self._tier_service.is_host_staged
            and self._execution_config.protocol.kind is ProtocolKind.CONVENTIONAL
        ):
            self._resources.close()
            self._resources_closed = True
            self._closed = True
            raise RuntimeError(
                "finite tenant budgets require the dependency-aware host protocol"
            )
        self._configure_tenant_budgets(bootstrap.tenant_specs)
        self._request_adapter = SglangAdapter(self._runtime, request_capacity)
        self._hicache = SglangHiCacheBridge(
            self.token_to_kv_pool,
            work_capacity=max(4096, request_capacity * 4),
            allow_load_fallback=bootstrap.allow_load_fallback,
        )
        opportunity_parallel_slots = int(
            torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
        )
        tuning = SglangExecutionTuning.from_environment(
            model_runner=model_runner,
            token_pool=self.token_to_kv_pool,
            tier=self._tier_service,
            bootstrap=bootstrap,
            opportunity_parallel_slots=opportunity_parallel_slots,
        )
        observability = tuning.observability
        mover_priority = tuning.mover_stream_priority
        self._progress_stream = torch.cuda.Stream(priority=mover_priority)
        self._host_cost_model = tuning.host_cost_model
        self._incremental_calibration_probes_remaining = (
            tuning.incremental_calibration_probes
        )
        self._incremental_initialization_probes_remaining = min(
            AUTO_INCREMENTAL_INITIALIZATION_PROBES,
            self._incremental_calibration_probes_remaining,
        )
        self._incremental_setup_samples = 0
        self._incremental_service_samples = 0
        host_mover_policy = tuning.host_mover_policy
        host_mover_default_service_model = tuning.host_mover_default_service_model
        host_mover_calibration_samples = tuning.host_mover_calibration_samples
        layer_service_minimum_samples = tuning.layer_service_minimum_samples
        layer_service_maximum_samples = tuning.layer_service_maximum_samples
        self._copy_engine_max_operations = tuning.copy_engine_max_operations
        self._indexed_copy_target_bytes = tuning.indexed_copy_target_bytes
        self._indexed_copy_max_blocks = tuning.indexed_copy_max_blocks
        self._frontier_layers_per_wave = tuning.frontier_layers_per_wave
        self._sm_acquisition_waves = tuning.sm_acquisition_waves
        self._sm_mover_max_worker_ctas = tuning.sm_mover_max_worker_ctas
        self._overlap_enabled = tuning.overlap_enabled
        self._frontier_enabled = tuning.frontier_enabled
        self._demand_graph_enabled = tuning.demand_graph_enabled
        self._fragment_enabled = tuning.fragment_enabled
        self._demand_overlap_policy = tuning.demand_overlap_policy
        self._stream_ordered_retirement = tuning.stream_ordered_retirement
        self._verification = tuning.verification
        self._grouping = tuning.grouping
        self._global_model_layer_count = tuning.model.global_layer_count
        self._model_start_layer = tuning.model.first_layer
        self._model_end_layer = tuning.model.end_layer
        self._model_layer_count = tuning.model.layer_count
        self._cuda_graph_mode = False
        self._stock_forward = False
        self._resident_reference_forward = False
        demand_graph_capacity = tuning.demand_graph_capacity
        mover_model = host_mover_default_service_model
        self._stats = initial_engine_stats(
            SglangTelemetryConfig(
                model_layer_count=self._model_layer_count,
                execution_protocol=self._execution_config.protocol.kind.value,
                host_execution_mode=self._execution_config.host_execution_mode.value,
                work_granularity=self._execution_config.protocol.granularity.value,
                protocol_max_inflight_units=(
                    self._execution_config.protocol.max_inflight_units
                ),
                runtime_tenant_capacity=self._resources.config.tenant_capacity,
                runtime_staging_byte_capacity=(
                    self._resources.config.staging_byte_capacity
                ),
                tenant_isolation_enabled=self._tenant_isolation_enabled,
                overlap_enabled=self._overlap_enabled,
                frontier_enabled=self._frontier_enabled,
                fragment_enabled=self._fragment_enabled,
                demand_overlap_policy=self._demand_overlap_policy,
                stream_ordered_retirement_enabled=(self._stream_ordered_retirement),
                sglang_mixed_chunk_enabled=bool(
                    model_runner.server_args.enable_mixed_chunk
                ),
                max_host_rounds=self._host_cost_model.max_rounds,
                minimum_predicted_gain=self._host_cost_model.minimum_predicted_gain,
                incremental_setup_ns=self._host_cost_model.incremental_setup_ns,
                incremental_service_scale=(
                    self._host_cost_model.incremental_service_scale
                ),
                incremental_calibration_probes_remaining=(
                    self._incremental_calibration_probes_remaining
                ),
                cost_model_bandwidth_bps=(
                    self._host_cost_model.bandwidth_bytes_per_second
                ),
                host_mover=host_mover_policy,
                copy_engine_max_operations=self._copy_engine_max_operations,
                host_mover_copy_calibrated=mover_model.copy_calibrated,
                host_mover_calibration_samples_per_engine=(
                    host_mover_calibration_samples
                ),
                host_mover_sm_samples=mover_model.sm_samples,
                host_mover_copy_samples=mover_model.copy_samples,
                host_mover_sm_bandwidth_bps=(mover_model.sm_bandwidth_bytes_per_second),
                host_mover_copy_bandwidth_bps=(
                    mover_model.copy_bandwidth_bytes_per_second
                ),
                host_mover_copy_operation_ns=mover_model.copy_operation_ns,
                host_mover_hybrid_join_ns=mover_model.hybrid_join_ns,
                host_mover_minimum_gain=mover_model.minimum_gain,
                layer_service_minimum_samples=layer_service_minimum_samples,
                layer_service_maximum_samples=layer_service_maximum_samples,
                indexed_copy_target_bytes=self._indexed_copy_target_bytes,
                indexed_copy_max_blocks=self._indexed_copy_max_blocks,
                frontier_layers_per_wave=self._frontier_layers_per_wave,
                sm_acquisition_waves=self._sm_acquisition_waves,
                sm_mover_max_worker_ctas=self._sm_mover_max_worker_ctas,
                demand_graph_enabled=self._demand_graph_enabled,
                demand_graph_capacity=demand_graph_capacity,
                engine_version=observability.engine_version,
                revision=observability.revision,
            ),
            self._tier_service.stats(),
        )
        self._forward_lifecycle = SglangForwardLifecycle(
            request_adapter=self._request_adapter,
            hicache=self._hicache,
            granularity=self._execution_config.protocol.granularity,
            model_layer_count=self._model_layer_count,
            stats=self._stats,
        )
        self._layer_calibration = SglangLayerServiceCalibration(
            enabled=self._tier_service.is_host_staged,
            minimum_samples=layer_service_minimum_samples,
            maximum_samples=layer_service_maximum_samples,
            model_start_layer=self._model_start_layer,
            model_layer_count=self._model_layer_count,
            stats=self._stats,
        )
        self._consumer_calibration = SglangConsumerPolicyCalibration(
            enabled=self._tier_service.is_host_staged,
            model_start_layer=self._model_start_layer,
            model_layer_count=self._model_layer_count,
            stats=self._stats,
        )
        self._attention_verifier = SglangAttentionVerifier(
            decode_use_tensor_cores=self.decode_use_tensor_cores,
            stats=self._stats,
        )
        self._kernels = SglangKernelResources(
            config=SglangKernelConfig(
                dtype_q=model_runner.dtype,
                dtype_kv=model_runner.kv_cache_dtype,
                head_dim=int(model_runner.model_config.head_dim),
                num_wrappers=self.num_wrappers,
                skip_prefill=self.skip_prefill,
                decode_use_tensor_cores=self.decode_use_tensor_cores,
                stream_ordered_retirement=self._stream_ordered_retirement,
                workspace_buffer=self.workspace_buffer,
            ),
            stock_wrappers=stock_wrappers,
            stats=self._stats,
        )
        self._activate_wrapper_set(self._kernels.stock_wrappers)
        self._demand_graph_cache = DemandGraphCache(
            capacity=demand_graph_capacity,
            stats=self._stats,
        )
        nvme_regions = (
            self._prepare_nvme_regions() if self._tier_service.is_nvme else {}
        )
        self._profile_cpu = observability.profile_cpu
        self._metadata_planner = SglangMetadataPlanner(
            tier_service=self._tier_service,
            execution_config=self._execution_config,
            tenant_isolation_enabled=self._tenant_isolation_enabled,
            object_capacity=self._object_capacity,
            grouping=self._grouping,
            page_size=int(self.token_to_kv_pool.page_size),
            profile_cpu=self._profile_cpu,
            stats=self._stats,
        )
        self._materializer = SglangPlanMaterializer(
            runtime=self._runtime,
            tier_service=self._tier_service,
            max_dependencies_per_work_ticket=(self._max_dependencies_per_work_ticket),
            work_ticket_capacity=self._work_ticket_capacity,
            object_capacity=self._object_capacity,
            tenant_isolation_enabled=self._tenant_isolation_enabled,
            profile_cpu=self._profile_cpu,
            stats=self._stats,
            stock_wrapper_available=self._has_stock_wrapper,
            transport_program=self._kernels.transport_program,
            discard_plan=self._demand_graph_cache.discard_plan,
        )
        self._nvme_pipeline = (
            SglangNvmeAcquisitionPipeline(
                runtime=self._runtime,
                tier_service=self._tier_service,
                transport_program=self._require_kernels().transport_program,
                progress_stream=self._progress_stream,
                layer_start=self._model_start_layer,
                layer_count=self._model_layer_count,
                object_capacity=self._object_capacity,
                work_ticket_capacity=self._work_ticket_capacity,
                tenant_isolation=self._tenant_isolation_enabled,
                regions=nvme_regions,
                stats=self._stats,
            )
            if self._tier_service.is_nvme
            else None
        )
        configured_stats = observability.stats_file
        self._stats_publisher: StatsPublisher | None = None
        self._profile_transfer = observability.profile_transfer
        self._profile_index_layout = observability.profile_index_layout
        self._profile_index_min_bytes = observability.profile_index_min_bytes
        self._stats.update(
            {
                "indexed_layout_profile_enabled": self._profile_index_layout,
                "indexed_layout_min_copy_bytes": self._profile_index_min_bytes,
                "indexed_layout_profiles": 0,
                "indexed_layout_rows": 0,
                "indexed_layout_runs": 0,
                "indexed_layout_eligible_rows": 0,
                "indexed_layout_candidate_bytes": 0,
                "indexed_layout_profile_cpu_ns": 0,
                "indexed_layout_maximum_run_rows": 0,
            }
        )
        self._profile_gpu = observability.profile_gpu
        # Barrier profiling measures how long the compute stream stalls at each
        # proactive layer-readiness wait. It is the opportunity signal the
        # RQ2/2A characterization consumes: stall > 0 means arrival, not
        # compute, bounded that layer.
        self._profile_barrier = observability.profile_barrier
        self._transfer_profiles: list[
            tuple[torch.cuda.Event, torch.cuda.Event, int, str]
        ] = []
        self._host_movers = HostMoverController(
            policy=host_mover_policy,
            default_service_model=host_mover_default_service_model,
            calibration_samples=host_mover_calibration_samples,
            copy_engine_max_operations=self._copy_engine_max_operations,
            frontier_layers_per_wave=self._frontier_layers_per_wave,
            profile_transfer=self._profile_transfer,
            frontier_enabled=self._frontier_enabled,
            profile_index_layout=self._profile_index_layout,
            profile_index_min_bytes=self._profile_index_min_bytes,
            verify_index_map=self._verification.index_map,
            stats=self._stats,
        )
        self._host_transport = SglangHostTransport(
            runtime=self._runtime,
            host_movers=self._host_movers,
            object_capacity=self._object_capacity,
            stream_priority=mover_priority,
            frontier_layers_per_wave=self._frontier_layers_per_wave,
            sm_acquisition_waves=self._sm_acquisition_waves,
            sm_mover_max_worker_ctas=self._sm_mover_max_worker_ctas,
            copy_engine_max_operations=self._copy_engine_max_operations,
            profile_barrier=self._profile_barrier,
            profile_cpu=self._profile_cpu,
            profile_transfer=self._profile_transfer,
            stats=self._stats,
            transfer_profiles=self._transfer_profiles,
            transport_program=self._require_kernels().transport_program,
            collect_barrier_profiles=self._collect_barrier_profiles,
        )
        self._host_acquisition = SglangHostAcquisitionCoordinator(
            device_pool=self.token_to_kv_pool,
            execution_config=self._execution_config,
            tenant_isolation_enabled=self._tenant_isolation_enabled,
            model_layer_count=self._model_layer_count,
            sm_acquisition_waves=self._sm_acquisition_waves,
            frontier_enabled=self._frontier_enabled,
            frontier_layers_per_wave=self._frontier_layers_per_wave,
            movers=self._host_movers,
            calibration=self._layer_calibration,
            consumer_calibration=self._consumer_calibration,
            minimum_consumer_gain=self._host_cost_model.minimum_predicted_gain,
            transport=self._host_transport,
            stats=self._stats,
        )
        self._operator_profiles: list[_OperatorProfile] = []
        self._barrier_profiles: list[_BarrierProfile] = []
        self._barrier_stall_by_layer: dict[int, float] = {}
        self._stats.update(
            {
                "barrier_profile_enabled": self._profile_barrier,
                "profiled_attention_arrivals": 0,
                "profiled_attention_ready_at_arrival": 0,
                "profiled_attention_not_ready_at_arrival": 0,
                "profiled_attention_materially_stalled_arrivals": 0,
                "profiled_attention_stall_gpu_ms": 0.0,
                "profiled_attention_max_stall_gpu_ms": 0.0,
            }
        )
        self._opportunity_trace = observability.opportunity_trace
        self._opportunity_revision = observability.revision
        self._opportunity_model = observability.opportunity_model
        self._opportunity_tier = observability.opportunity_tier
        self._opportunity_batch = 0
        self._active_opportunity_batch = -1
        self._measure_opportunity_compute = observability.measure_opportunity_compute
        self._opportunity_parallel_slots = observability.opportunity_parallel_slots
        self._engine_version = observability.engine_version
        self._attention_executor = SglangAttentionExecutor(
            runtime=self._runtime,
            tier_service=self._tier_service,
            hicache=self._hicache,
            materializer=self._require_materializer(),
            nvme_pipeline=self._nvme_pipeline,
            kernels=self._require_kernels(),
            demand_graph_cache=self._demand_graph_cache,
            progress_stream=self._progress_stream,
            stats=self._stats,
            stock_wrapper=self._forward_lifecycle.stock_wrapper,
            transfer_profiles=self._transfer_profiles,
            operator_profiles=self._operator_profiles,
            barrier_profiles=self._barrier_profiles,
            config=SglangAttentionExecutionConfig(
                tenant_isolation_enabled=self._tenant_isolation_enabled,
                indexed_copy_target_bytes=self._indexed_copy_target_bytes,
                indexed_copy_max_blocks=self._indexed_copy_max_blocks,
                stream_ordered_retirement=self._stream_ordered_retirement,
                demand_graph_enabled=self._demand_graph_enabled,
                profile_barrier=self._profile_barrier,
                profile_cpu=self._profile_cpu,
                profile_gpu=self._profile_gpu,
                profile_transfer=self._profile_transfer,
                measure_opportunity_compute=self._measure_opportunity_compute,
                opportunity_parallel_slots=self._opportunity_parallel_slots,
                opportunity_trace=self._opportunity_trace,
                opportunity_revision=self._opportunity_revision,
                opportunity_model=self._opportunity_model,
                opportunity_tier=self._opportunity_tier,
            ),
        )
        if self._tier_service.is_host_staged:
            # Capture ownership immediately, but delay the execution form until
            # the exact FlashInfer schedule is available in forward metadata.
            self._hicache.set_acquire_callback(self._host_acquisition.capture)
            self._hicache.set_deadline_model_callback(
                self._host_acquisition.deadline_model
            )
            self._hicache.set_admission_acquisition_callbacks(
                prepare=self._host_acquisition.prepare_admission,
                start=self._host_acquisition.start_admission,
            )
            if tuning.requires_typed_host_modules(bootstrap):
                self._require_kernels().prepare_typed_execution_modules(
                    runtime=self._runtime,
                    host_staged=True,
                    stream=torch.cuda.current_stream(),
                )
            if host_mover_policy != "sm":
                planner_warmup_ns = warm_indexed_tensor_mover(
                    self.token_to_kv_pool.device,
                    maximum_rows=int(self.token_to_kv_pool.size),
                    maximum_copy_runs=self._copy_engine_max_operations // 2,
                )
                self._stats["indexed_mover_setup_warmup_ns"] = planner_warmup_ns
                self._stats["indexed_mover_setup_warmup_rows"] = min(
                    int(self.token_to_kv_pool.size), 1 << 16
                )
                self._stats["indexed_mover_setup_warmup_runs"] = min(
                    self._copy_engine_max_operations // 2,
                    int(self.token_to_kv_pool.size),
                )
        elif self._tier_service.is_nvme:
            # NVMe has no framework-reference numerical fallback: every
            # external batch requires the compiler-verified native consumer.
            # Prepare it before engine readiness so the first user request can
            # never become a 95-second compilation probe.
            self._require_kernels().prepare_typed_execution_modules(
                runtime=self._runtime,
                host_staged=False,
                stream=torch.cuda.current_stream(),
            )
        if configured_stats is not None:
            stats_path = configured_stats
            if stats_path.suffix:
                stats_path = stats_path.with_name(
                    f"{stats_path.stem}.{os.getpid()}{stats_path.suffix}"
                )
            else:
                stats_path = stats_path / f"nta-sglang-{os.getpid()}.json"
            self._stats_publisher = StatsPublisher(stats_path)
            # Persist setup evidence only after every setup action, including
            # typed-module loading and mover-kernel initialization, has either
            # completed or failed construction.
            setup_report = dict(self._stats)
            setup_report.update(self._tier_service.stats())
            setup_report.update(
                {
                    "stats_lifecycle": "setup",
                    "stats_process_id": os.getpid(),
                    "snapshot_unix_ns": time.time_ns(),
                }
            )
            self._stats_publisher.publish(setup_report, wait=True)
        atexit.register(self._write_stats)

    def _configure_tenant_budgets(self, specs: tuple[tuple[int, int], ...]) -> None:
        for tenant_id, max_bytes in specs:
            if tenant_id >= self._tenant_capacity:
                raise RuntimeError(
                    f"tenant {tenant_id} exceeds NTA_TENANT_CAPACITY="
                    f"{self._tenant_capacity}"
                )
            self._runtime.set_tenant_budget(tenant_id, max_bytes)

    def cancel_requests(self, request_id_prefix: str, *, all: bool = False) -> int:
        cancelled = self._request_adapter.cancel_matching(request_id_prefix, all=all)
        self._stats["request_cancellations"] += cancelled
        return cancelled

    def retire_request(self, request_id: str) -> bool:
        """Retire one SGLang-confirmed request generation.

        Completion and abort are separate framework lifecycle edges.  Both
        invalidate outstanding device work, but accounting them separately
        makes a missing completion hook visible in long-running serving tests.
        """
        # SGLang invokes this lifecycle edge only after the forward result has
        # reached the CPU.  CUDA timing events for that request's layer arrivals
        # are therefore eligible for a query-only commit here.  Deferring this
        # until the next transfer plan made behavior-matched warmup samples
        # invisible to the measured plan (and left shutdown as the first point
        # that committed them).
        committed_before = int(self._stats["layer_service_profiled_intervals"])
        self._layer_calibration.collect()
        self._consumer_calibration.collect()
        self._stats["layer_service_retirement_commits"] += (
            int(self._stats["layer_service_profiled_intervals"]) - committed_before
        )
        retired = self._request_adapter.retire_request(request_id)
        if retired:
            self._stats["request_retirements"] += 1
        return retired

    def __del__(self) -> None:
        """Release resources if construction failed before normal close()."""
        try:
            publisher = getattr(self, "_stats_publisher", None)
            if publisher is not None:
                try:
                    publisher.close()
                except BaseException:
                    pass
                self._stats_publisher = None
            if not getattr(self, "_resources_closed", True):
                try:
                    self._close_resources()
                except BaseException:
                    pass
        except BaseException:
            # Destructors cannot safely report or propagate errors. Explicit
            # close() remains strict; this path only prevents a partial
            # constructor from retaining a native transport indefinitely.
            pass

    def close(self) -> None:
        """Flush observations and release CUDA/native tier resources."""
        if self._closed:
            return
        self._write_stats(strict=True)

    def _quiesce_observation_boundary(self) -> None:
        """Retire every CUDA observation at an explicit control boundary.

        Ordinary forwards only query completed events, so profiling cannot
        serialize the serving path. Artifact control snapshots and engine
        shutdown are different: their snapshot must describe all work issued
        before the boundary and none issued after it.  Synchronize there, then
        fail closed if any event-backed observation remains unretired.
        """

        torch.cuda.synchronize()
        self._collect_transfer_profiles()
        self._layer_calibration.collect()
        self._consumer_calibration.collect()
        self._collect_barrier_profiles(already_synchronized=True)
        pending = {
            "mover": self._host_movers.pending_profile_count,
            "transfer": len(self._transfer_profiles),
            "operator": len(self._operator_profiles),
            "layer_service": self._layer_calibration.pending_count,
            "consumer_policy": self._consumer_calibration.pending_count,
            "barrier": len(self._barrier_profiles),
        }
        pending = {name: count for name, count in pending.items() if count}
        if pending:
            raise RuntimeError(
                "CUDA observations remained pending after a synchronized "
                f"measurement boundary: {pending}"
            )

    def _close_resources(self) -> None:
        if self._resources_closed:
            return
        errors: list[BaseException] = []
        # Plans, graphs, and native runtime buffers all contain device pointers.
        # Quiesce every stream before releasing them, including direct NVMe HBM
        # destinations and CXL-backed mappings.  Teardown continues after an
        # individual owner fails so one bad lease cannot leak later owners.
        try:
            torch.cuda.synchronize()
        except BaseException as error:
            errors.append(error)
        hicache = getattr(self, "_hicache", None)
        if hicache is not None:
            try:
                hicache.close()
            except BaseException as error:
                errors.append(error)
        lifecycle = getattr(self, "_forward_lifecycle", None)
        if lifecycle is not None:
            lifecycle.reset_after_quiescence()
        graph_cache = getattr(self, "_demand_graph_cache", None)
        if graph_cache is not None:
            graph_cache.clear()
        attention_executor = getattr(self, "_attention_executor", None)
        if attention_executor is not None:
            attention_executor.clear()
            self._attention_executor = None
        self._attention_verifier = None
        nvme_pipeline = getattr(self, "_nvme_pipeline", None)
        if nvme_pipeline is not None:
            errors.extend(nvme_pipeline.close())
            self._nvme_pipeline = None
        materializer = getattr(self, "_materializer", None)
        if materializer is not None:
            errors.extend(materializer.close())
            self._materializer = None
        kernels = getattr(self, "_kernels", None)
        if kernels is not None:
            # The inherited backend fields borrow the active wrapper set.
            # Return those fields to the stock set before the owner drops its
            # instrumented wrappers and loaded-module references.
            try:
                self._activate_wrapper_set(kernels.stock_wrappers)
            except BaseException as error:
                errors.append(error)
            errors.extend(kernels.close())
            self._kernels = None
        resources = getattr(self, "_resources", None)
        if resources is not None:
            try:
                resources.close()
            except BaseException as error:
                errors.append(error)
        self._resources_closed = True
        if errors:
            raise RuntimeError(
                f"NTA resource teardown encountered {len(errors)} error(s)"
            ) from errors[0]

    def _require_materializer(self) -> SglangPlanMaterializer:
        materializer = self._materializer
        if materializer is None:
            raise RuntimeError("SGLang plan materializer is not available")
        return materializer

    def _has_stock_wrapper(self, wrapper_id: int) -> bool:
        return self._forward_lifecycle.has_wrapper_alias(wrapper_id)

    def forward_profile_cursor(self) -> int:
        """Snapshot lifecycle activation for an optional around-forward probe."""

        return self._forward_lifecycle.profile_cursor()

    def external_forward_since(self, cursor: int) -> bool:
        """Classify the measured forward after its epoch may have completed."""

        return self._forward_lifecycle.external_since(cursor)

    def _require_kernels(self) -> SglangKernelResources:
        kernels = self._kernels
        if kernels is None:
            raise RuntimeError("SGLang kernel resources are not available")
        return kernels

    def _require_attention_executor(self) -> SglangAttentionExecutor:
        executor = self._attention_executor
        if executor is None:
            raise RuntimeError("SGLang attention executor is not available")
        return executor

    def _require_attention_verifier(self) -> SglangAttentionVerifier:
        verifier = self._attention_verifier
        if verifier is None:
            raise RuntimeError("SGLang attention verifier is not available")
        return verifier

    def _activate_wrapper_set(self, wrappers: SglangWrapperSet) -> None:
        self.decode_wrappers = list(wrappers.decode)
        if self.skip_prefill:
            return
        self.prefill_wrappers_paged = list(wrappers.prefill_paged)
        self.prefill_wrappers_verify = list(wrappers.prefill_verify)

    def _adopt_typed_forward_metadata(
        self, forward_batch: Any, stock_metadata: Any
    ) -> tuple[Any, ...]:
        """Atomically project one validated stock plan onto typed wrappers.

        The active batch already owns the exact schedule and acquisition
        geometry extracted from ``stock_metadata``.  Typed wrappers share that
        plan's workspace after adoption, so only wrapper identity changes; a
        second extraction would repeat CUDA-to-host synchronization without
        adding evidence.
        """

        if hasattr(stock_metadata, "decode_wrappers"):
            sources = tuple(stock_metadata.decode_wrappers)
            targets = tuple(self.decode_wrappers)
            field = "decode_wrappers"
        else:
            if stock_metadata.use_ragged:
                raise RuntimeError("typed acquisition requires paged prefill")
            sources = tuple(stock_metadata.prefill_wrappers)
            targets = tuple(
                self.prefill_wrappers_verify
                if forward_batch.forward_mode.is_target_verify()
                else self.prefill_wrappers_paged
            )
            field = "prefill_wrappers"
        if not sources or len(sources) != len(targets):
            raise RuntimeError("stock and typed FlashInfer wrapper counts disagree")
        for target, source in zip(targets, sources, strict=True):
            adopt_planned_flashinfer_state(target, source)
        batch = self._forward_lifecycle.active
        if batch is None:
            raise RuntimeError("typed FlashInfer adoption has no validated batch")
        source_to_target = {
            id(source): id(target)
            for target, source in zip(targets, sources, strict=True)
        }
        target_to_source = {
            id(target): source
            for target, source in zip(targets, sources, strict=True)
        }
        self._forward_lifecycle.adopt_wrapper_aliases(
            batch,
            source_to_target,
            target_to_source,
        )
        self.forward_metadata = replace(stock_metadata, **{field: list(targets)})
        self._stats["reused_flashinfer_plans"] = self._stats.get(
            "reused_flashinfer_plans", 0
        ) + len(targets)
        self._stats["ready_stock_wrapper_pairs"] = self._stats.get(
            "ready_stock_wrapper_pairs", 0
        ) + len(targets)
        return targets

    def _run_ready_stock_numerical(
        self,
        typed_wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        layer: Any,
        *,
        causal: bool,
        window_left: int,
    ) -> torch.Tensor:
        stock = self._forward_lifecycle.stock_wrapper(id(typed_wrapper))
        if stock is None:
            raise RuntimeError("event-complete layer has no verified stock wrapper")
        query = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        common = {
            "sm_scale": layer.scaling,
            "logits_soft_cap": 0.0,
            "k_scale": layer.k_scale_float,
            "v_scale": layer.v_scale_float,
        }
        if isinstance(stock, BatchDecodeWithPagedKVCacheWrapper):
            result = stock.forward(query, kv_cache, **common)
        else:
            result = stock.forward(
                query,
                kv_cache,
                causal=causal,
                window_left=window_left,
                **common,
            )
        self._stats["stock_ready_external_attention_launches"] = (
            self._stats.get("stock_ready_external_attention_launches", 0) + 1
        )
        return result

    def init_cuda_graph_state(self, *args: Any, **kwargs: Any) -> None:
        super().init_cuda_graph_state(*args, **kwargs)

    def _validate_semantic_wrapper_plan(
        self,
        wrapper: Any,
        layer: Any,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        *,
        verify: bool,
    ) -> int:
        """Check physical row geometry and optionally run the specification."""
        if self._forward_lifecycle.active is None:
            raise RuntimeError("cannot validate execution without active batch")
        batch = self._forward_lifecycle.active
        wrapper_id = id(wrapper)
        semantic = batch.semantic_plans.get(wrapper_id)
        if semantic is None:
            raise RuntimeError("attention wrapper has no forward-scoped semantic plan")
        schedule = semantic.schedule
        # SGLang plans each wrapper once per ForwardBatch. Production consumes
        # that already-validated snapshot for every model layer; rereading the
        # CUDA int-workspace here would introduce one D2H synchronization per
        # layer. Verification deliberately re-extracts and compares it.
        if verify:
            decode_wrappers = tuple(
                getattr(self.forward_metadata, "decode_wrappers", ()) or ()
            )
            extracted = (
                decode_schedule(wrapper)
                if any(wrapper is candidate for candidate in decode_wrappers)
                else paged_prefill_schedule(wrapper)
            )
            if schedule != extracted:
                raise RuntimeError(
                    "FlashInfer wrapper schedule changed within a forward"
                )
        unit_bytes = int(
            kv_cache[0][0].numel() * kv_cache[0].element_size()
            + kv_cache[1][0].numel() * kv_cache[1].element_size()
        )
        if semantic.topology.unit_bytes != unit_bytes:
            raise RuntimeError("SGLang KV row geometry changed within a forward")

        if verify:
            semantic_started = time.perf_counter_ns()
            engine_batch = self._forward_lifecycle.engine_batch
            if engine_batch is None:
                raise RuntimeError("execution verification has no engine batch epoch")
            batch.execution = build_execution_plan(
                engine_batch=engine_batch,
                protocol=self._execution_config.protocol,
                tile_compute_ns=self._host_cost_model.tile_compute_ns,
                bindings=batch.bindings,
                schedule=schedule,
                page_pairs=semantic.page_pairs,
                acquisition_slices=semantic.acquisition_slices,
                layer=int(layer.layer_id) - self._model_start_layer,
                unit_bytes=unit_bytes,
            )
            self._stats["semantic_dense_tiles"] += len(batch.execution.batch.units)
            self._stats.update(batch.execution.expose_stats())
            batch.verification_session = ExecutionSession.from_plan(batch.execution)
            semantic_elapsed = time.perf_counter_ns() - semantic_started
            self._stats["semantic_verifier_plan_builds"] += 1
            if self._profile_cpu:
                self._stats["semantic_plan_cpu_ns"] += semantic_elapsed
            self._stats["semantic_verifier_sessions"] += 1
        else:
            # Production retains only the compact topology. The semantic plan
            # is an opt-in specification, never a second serving state machine.
            batch.execution = None
            batch.verification_session = None
            semantic_elapsed = 0
        self._stats["semantic_wrapper_plan_lookups"] += 1
        return semantic_elapsed

    def _record_execution_layer(
        self,
        layer: Any,
        *,
        indexed_object_count: int,
        final_layer: bool,
    ) -> None:
        """Commit the semantic work boundary after native attention returns."""
        if self._forward_lifecycle.active is None:
            raise RuntimeError("attention returned without a typed work topology")
        if indexed_object_count < 0:
            raise ValueError("indexed object count cannot be negative")
        local_layer = int(layer.layer_id) - self._model_start_layer
        verifier = self._forward_lifecycle.active.verification_session
        if verifier is not None:
            self._stats.update(verifier.record_layer_completion(local_layer))
        # Native host objects publish a consumer edge before their directory
        # identity can be replaced. Event-owned layers need only the final
        # lease edge. The physical-plan owner records exactly that lifetime.
        acquisition = self._forward_lifecycle.active.acquisition
        if (
            acquisition is not None
            and acquisition.tier is AcquisitionTier.HOST_STAGED
        ):
            self._require_materializer().record_host_consumer(
                torch.cuda.current_stream(),
                indexed_objects=indexed_object_count != 0,
                final_layer=final_layer,
            )

    def _commit_external_layer(
        self,
        *,
        batch: SglangForwardEpoch,
        pending: PendingHostLoad,
        layer: Any,
        local_layer: int,
        native_dispatch: bool,
        progressive_consumer: bool,
        indexed_object_count: int = 0,
        record_semantic: bool = False,
        fragment: DeadlineFragment | None = None,
    ) -> None:
        """Commit one external consumer in a single ordered transaction.

        The numerical launch is already enqueued. Publish its directory
        quiescence edge first, check final asynchronous status, account the
        dispatch, advance acquisition, release the HiCache layer, and finally
        retire forward-scoped state.
        This ordering is shared by native, preloaded, decode, and extend paths.
        """

        if self._forward_lifecycle.active is not batch:
            raise RuntimeError("external layer commit lost its forward epoch")
        acquisition = batch.acquisition
        if acquisition is None:
            raise RuntimeError("external layer commit has no acquisition owner")
        try:
            final_layer = local_layer + 1 == self._model_layer_count
            validated_dispatch = self._forward_lifecycle.validate_external_dispatch(
                batch,
                local_layer,
                native_dispatch=native_dispatch,
                progressive_consumer=progressive_consumer,
                final_layer=final_layer,
            )
            if record_semantic:
                self._record_execution_layer(
                    layer,
                    indexed_object_count=indexed_object_count,
                    final_layer=final_layer,
                )
            elif acquisition.tier is AcquisitionTier.HOST_STAGED:
                self._require_materializer().record_host_consumer(
                    torch.cuda.current_stream(),
                    indexed_objects=False,
                    final_layer=final_layer,
                )

            # The final stream epoch owns asynchronous status for every native
            # acquisition in this forward. Check it before HiCache can publish its
            # held acknowledgement and unlock framework-owned host rows.
            if final_layer:
                self._finalize_stream_ordered_batch(
                    batch,
                    torch.cuda.current_stream(),
                )
                acquisition.finish(torch.cuda.current_stream())
                if (
                    acquisition.tier is AcquisitionTier.NVME
                    and self._runtime.sticky_failed_count != 0
                ):
                    raise RuntimeError("the proactive NVMe acquisition pipeline failed")

            self._forward_lifecycle.commit_external_dispatch(validated_dispatch)
            self._stats["external_launches"] += 1
            dispatch_counter = (
                "native_external_attention_launches"
                if native_dispatch
                else "stock_prefetched_external_attention_launches"
            )
            self._stats[dispatch_counter] += 1

            if acquisition.tier is AcquisitionTier.HOST_STAGED:
                frontier_started = time.perf_counter_ns() if self._profile_cpu else 0
                self._advance_deadline_frontier(
                    pending,
                    local_layer,
                    fragment=fragment,
                )
                if self._profile_cpu:
                    self._stats["deadline_frontier_cpu_ns"] = self._stats.get(
                        "deadline_frontier_cpu_ns", 0
                    ) + (time.perf_counter_ns() - frontier_started)

            self._hicache.complete_layer(pending, local_layer)
            if not final_layer:
                return
            self._commit_incremental_setup_observation(batch)
            self._finish_forward(batch)
        except BaseException as error:
            # Attention has already been enqueued when this transaction starts.
            # Any failed publication stage therefore needs exceptional global
            # quiescence before framework rows, NVMe ownership, or aliases can
            # be reused.  Cleanup is idempotent and preserves the primary fault.
            try:
                self.abort_active_forward(pending)
            except BaseException as cleanup_error:
                error.add_note(
                    "external layer cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise

    def _begin_forward(self) -> None:
        self._forward_lifecycle.begin()

    def _finish_forward(self, batch: SglangForwardEpoch) -> None:
        pending = batch.pending_host_load
        if pending is not None:
            execution = batch.host_execution
            self._consumer_calibration.retire_lease(
                pending,
                probe_executed=(
                    execution is not None
                    and execution.selection_reason == "consumer_policy_probe"
                ),
            )
        self._forward_lifecycle.finish(
            batch,
            retain_for_graph=self._cuda_graph_mode,
        )

    def abort_active_forward(self, pending: PendingHostLoad | None = None) -> bool:
        """Quiesce and retire an abnormal forward without leaking its lease.

        This is an exceptional control boundary, so a device synchronization
        is intentional: proactive copies may have been issued on auxiliary
        streams, and returning SGLang's host-row acknowledgement before all of
        them finish would permit source reuse under DMA.
        """

        return self._forward_lifecycle.abort(pending)

    def _finalize_stream_ordered_batch(
        self, batch: SglangForwardEpoch, stream: torch.cuda.Stream
    ) -> None:
        """Retire one immutable typed plan after its final forward consumer."""

        epoch = batch.stream_ordered_epoch
        if epoch is None:
            return
        layers = batch.stream_ordered_layers
        rounds = batch.stream_ordered_progress_rounds
        if layers <= 0 or rounds <= 0:
            raise RuntimeError("deferred stream retirement lost its finite geometry")
        profile = None
        if self._profile_gpu:
            profile = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            profile[0].record(stream)
        epoch.retire_stream_ordered(stream)
        if profile is not None:
            profile[1].record(stream)
        # This is the forward's normal synchronization/error boundary. The
        # completion kernel validates request generation/cancellation and the
        # exact final runnable window before DeviceWorkPlan storage can move.
        epoch.check(rounds, stream)
        if self._runtime.sticky_failed_count != 0:
            raise RuntimeError("an asynchronous acquisition epoch failed")
        if profile is not None:
            self._operator_profiles.append(
                _OperatorProfile(*profile, "stream_retirement", layers)
            )
        self._stats["stream_ordered_retirement_launches"] += 1
        self._stats["stream_ordered_retirement_batches"] += 1
        batch.stream_ordered_epoch = None
        batch.stream_ordered_progress_rounds = 0
        batch.stream_ordered_layers = 0

    def _record_external_layer_execution(
        self,
        batch: SglangForwardEpoch,
        local_layer: int,
        *,
        native_dispatch: bool,
        progressive_consumer: bool,
        final_layer: bool,
    ) -> None:
        self._forward_lifecycle.record_external_dispatch(
            batch,
            local_layer,
            native_dispatch=native_dispatch,
            progressive_consumer=progressive_consumer,
            final_layer=final_layer,
        )

    def _prepare_nvme_regions(self) -> dict[tuple[int, str], Any]:
        """Describe and coalesce stable framework KV allocations at startup."""

        started = time.perf_counter_ns()
        catalog = self._tier_service.catalog
        if catalog is None or catalog.layer_count != self._global_model_layer_count:
            raise RuntimeError(
                "NVMe catalog layer count does not match the SGLang model"
            )
        if catalog.page_tokens != 1:
            raise RuntimeError(
                "NVMe SGLang integration currently requires page_tokens=1"
            )
        destinations: list[HbmDestinationSlice] = []
        for local_layer in range(self._model_layer_count):
            layer_id = self._model_start_layer + local_layer
            tensors = (
                ("key", self.token_to_kv_pool._get_key_buffer(layer_id)),
                ("value", self.token_to_kv_pool._get_value_buffer(layer_id)),
            )
            for kind, tensor in tensors:
                if not tensor.is_cuda or int(tensor.nbytes) <= 0:
                    raise RuntimeError(
                        f"NVMe {kind} region for layer {layer_id} is not live CUDA HBM"
                    )
                destinations.append(
                    HbmDestinationSlice(
                        (layer_id, kind),
                        int(tensor.data_ptr()),
                        int(tensor.nbytes),
                    )
                )
        try:
            prepared = self._tier_service.prepare_nvme_hbm_destinations(
                tuple(destinations)
            )
        except BaseException as error:
            raise RuntimeError(
                "NVMe worker-prepare could not register the complete local KV "
                f"destination set (layers=[{self._model_start_layer}, "
                f"{self._model_end_layer}), tensors={len(destinations)})"
            ) from error
        self._stats["nvme_region_prepare_ns"] = time.perf_counter_ns() - started
        self._stats["nvme_region_count"] = prepared.registration_count
        self._stats["nvme_region_bytes"] = prepared.registration_bytes
        self._stats["nvme_destination_slice_count"] = prepared.destination_count
        self._stats["nvme_destination_slice_bytes"] = prepared.destination_bytes
        self._stats["nvme_shared_region_slices"] = (
            prepared.destination_count - prepared.registration_count
        )
        return dict(prepared.regions)

    def _bind_forward_requests(
        self, forward_batch: Any, *, allow_capture_ids: bool
    ) -> tuple[RequestBinding, ...]:
        return self._forward_lifecycle.bind_requests(
            forward_batch,
            allow_capture_ids=allow_capture_ids,
        )

    def init_forward_metadata_out_graph(
        self, forward_batch: Any, in_capture: bool = False
    ) -> None:
        self._begin_forward()
        self._cuda_graph_mode = True
        self._resident_reference_forward = False
        counter = getattr(self.token_to_kv_pool, "layer_transfer_counter", None)
        consumer_index = -1 if counter is None else int(counter.consumer_index)
        pending = self._hicache.get(consumer_index)
        if pending is not None:
            self._host_acquisition.account_selection(pending)
        # A framework CUDA graph has immutable kernel arguments.  Capturing a
        # request-bound NTA wrapper would freeze the capture-time request slot
        # and silently attribute later replays to the wrong generation.  Graph
        # execution therefore has one explicit contract: acquisition fully
        # materializes KV into SGLang's device pool before replay, and the
        # captured numerical consumer is stock FlashInfer.  Native typed
        # wrappers remain available for eager and NTA-owned finite graphs,
        # whose work-plan buffers are updated safely between replays.
        self._stock_forward = True
        self._activate_wrapper_set(self._require_kernels().stock_wrappers)
        super().init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)
        if in_capture:
            # SGLang's capture batch consists entirely of dummy rows, commonly
            # all using request-pool slot zero.  It has no serving identity and
            # must not consume generations or pollute the persistent registry.
            bindings: tuple[RequestBinding, ...] = ()
            self._stats["graph_capture_dummy_rows"] = self._stats.get(
                "graph_capture_dummy_rows", 0
            ) + int(getattr(forward_batch, "batch_size", 0) or 0)
        else:
            bindings = self._bind_forward_requests(
                forward_batch, allow_capture_ids=False
            )
        if pending is None:
            self._forward_lifecycle.activate(
                SglangForwardEpoch(
                    plan=SglangForwardPlan(
                        bindings=bindings,
                        semantic_plans={},
                        pending_host_load=None,
                    ),
                )
            )
            self._stats["resident_reference_batches"] += 1
            self._stats["stock_resident_batches"] += 1
        else:
            if self._tenant_isolation_enabled:
                raise RuntimeError(
                    "external CUDA-graph prefetch cannot bypass finite tenant budgets"
                )
            if pending.acquisition is None:
                self._host_acquisition.publish_missing(pending)
            else:
                self._host_acquisition.submit(pending)
            final_layer = _require_exact_prefetch_layers(
                pending.prefetched_layers,
                self._model_layer_count,
                consumer="CUDA graph replay",
            )
            self._forward_lifecycle.activate(
                SglangForwardEpoch(
                    plan=SglangForwardPlan(
                        bindings=bindings,
                        semantic_plans={},
                        pending_host_load=pending,
                    ),
                    acquisition=HostForwardAcquisition(pending),
                )
            )
            if self._profile_barrier:
                arrive = torch.cuda.Event(enable_timing=True)
                arrive.record(torch.cuda.current_stream())
                self._barrier_profiles.append(
                    _BarrierProfile(
                        arrive,
                        pending.prefetched_layers[final_layer].ready_event,
                        self._model_start_layer + final_layer,
                        "graph_batch",
                    )
                )
                self._stats["profiled_graph_prefetch_waits"] = (
                    self._stats.get("profiled_graph_prefetch_waits", 0) + 1
                )
            torch.cuda.current_stream().wait_event(
                pending.prefetched_layers[final_layer].ready_event
            )
            self._hicache.handoff_prefetch(pending, self._host_transport.stream)
            self._stats["graph_external_batches"] += 1
        if in_capture:
            self._stats["graph_captures"] += 1
        else:
            self._stats["graph_replays"] += 1
            self._stats["batches"] += 1

    def init_forward_metadata_in_graph(self, forward_batch: Any) -> None:
        super().init_forward_metadata_in_graph(forward_batch)

    def _activate_stock_prefetch(
        self,
        bindings: tuple[RequestBinding, ...],
        pending: PendingHostLoad,
        *,
        count_batch: bool = True,
    ) -> None:
        """Bind a complete exact prefetch without materializing an unused plan."""
        _require_exact_prefetch_layers(
            pending.prefetched_layers,
            self._model_layer_count,
            consumer="stock external attention",
        )
        replacement = SglangForwardEpoch(
            plan=SglangForwardPlan(
                bindings=bindings,
                semantic_plans={},
                pending_host_load=pending,
            ),
            acquisition=HostForwardAcquisition(pending),
        )
        active = self._forward_lifecycle.active
        if active is None:
            self._forward_lifecycle.activate(replacement)
        else:
            self._forward_lifecycle.replace_unstarted_epoch(active, replacement)
        if count_batch:
            self._stats["batches"] += 1
            self._stats["hicache_external_batches"] += 1
        self._stats["stock_prefetched_external_batches"] += 1
        self._stats["stock_prefetch_metadata_fastpath_batches"] = (
            self._stats.get("stock_prefetch_metadata_fastpath_batches", 0) + 1
        )

    def init_forward_metadata(self, forward_batch: Any) -> None:
        metadata_profile_started = time.perf_counter_ns() if self._profile_cpu else 0
        self._begin_forward()
        self._cuda_graph_mode = False
        self._stock_forward = False
        self._resident_reference_forward = False
        if forward_batch.forward_mode.is_mixed():
            self._stats["mixed_forward_batches"] += 1
            self._stats["mixed_forward_requests"] += len(
                tuple(getattr(forward_batch, "rids", ()) or ())
            )
        counter = getattr(self.token_to_kv_pool, "layer_transfer_counter", None)
        consumer_index = -1 if counter is None else int(counter.consumer_index)
        pending = self._hicache.get(consumer_index)
        if pending is not None:
            self._host_acquisition.account_selection(pending)
        measured_host_selection = (
            pending is not None and self._tier_service.is_host_staged
        )
        mixed_host_batch = (
            measured_host_selection and forward_batch.forward_mode.is_mixed()
        )
        if pending is None:
            self._stock_forward = True
            self._activate_wrapper_set(self._require_kernels().stock_wrappers)
        elif measured_host_selection or self._tier_service.is_nvme:
            # The stock plan supplies the exact schedule for a no-overhead
            # direct decision. A typed plan is built later only if measured
            # overlap justifies unresolved work.
            self._activate_wrapper_set(self._require_kernels().stock_wrappers)
        else:
            self._activate_wrapper_set(self._require_kernels().typed_wrappers())
        original_use_paged = self.use_paged
        # A resident batch is a true framework-reference fast path: preserve
        # SGLang's own ragged-vs-paged choice.  Only an external batch may
        # later adopt compiler-typed wrappers, whose prefill contract is
        # paged, so forcing ``use_paged`` is scoped to that case.
        if pending is not None:
            self.use_paged = True
        try:
            stock_metadata_started = (
                time.perf_counter_ns() if self._profile_cpu else 0
            )
            super().init_forward_metadata(forward_batch)
            stock_metadata_finished = (
                time.perf_counter_ns() if self._profile_cpu else 0
            )
            if pending is None:
                # A resident stock forward has no acquisition identity,
                # resource lifetime, or native work to attribute.  Publish
                # only the observation serial needed by the optional around-
                # forward profiler.  In particular, do not allocate a typed
                # epoch and do not send every layer through its state machine.
                stock_layers = self._model_layer_count
                self._stats["stock_attention_launches"] += stock_layers
                self._stats["stock_resident_attention_launches"] += stock_layers
                if forward_batch.forward_mode.is_decode_or_idle():
                    self._stats["decode_launches"] += stock_layers
                else:
                    self._stats["prefill_launches"] += stock_layers
                self._forward_lifecycle.record_reference_forward()
                self._resident_reference_forward = True
                self._stats["batches"] += 1
                self._stats["resident_reference_batches"] += 1
                self._stats["stock_resident_batches"] += 1
                if self._profile_cpu:
                    metadata_profile_finished = time.perf_counter_ns()
                    total_ns = metadata_profile_finished - metadata_profile_started
                    stock_ns = stock_metadata_finished - stock_metadata_started
                    self._stats["resident_reference_metadata_calls"] += 1
                    self._stats["resident_reference_metadata_cpu_ns"] += total_ns
                    self._stats["resident_reference_metadata_stock_cpu_ns"] += stock_ns
                    self._stats[
                        "resident_reference_metadata_overhead_cpu_ns"
                    ] += max(0, total_ns - stock_ns)
                return
            bind_started = time.perf_counter_ns() if self._profile_cpu else 0
            bindings = self._bind_forward_requests(
                forward_batch, allow_capture_ids=False
            )
            if self._profile_cpu:
                self._stats["request_bind_cpu_ns"] = self._stats.get(
                    "request_bind_cpu_ns", 0
                ) + (time.perf_counter_ns() - bind_started)
            if measured_host_selection:
                stock_metadata = self.forward_metadata
                selected = self._init_external_metadata(
                    forward_batch,
                    pending,
                    bindings=bindings,
                )
                if selected is None:
                    raise RuntimeError("host-staged batch has no execution decision")
                self._record_host_selection(selected)
                if (
                    (pending.acquisition is None or pending.acquisition.model is None)
                    and selected.uses_dependency_protocol
                    and self._host_acquisition.proactive_layer_queue_enabled
                    and self._host_acquisition.prepare_owner(
                        pending,
                        forward_batch,
                        active_batch=self._forward_lifecycle.active,
                    )
                ):
                    self._stats["metadata_acquisition_groups_prepared"] = (
                        self._stats.get("metadata_acquisition_groups_prepared", 0) + 1
                    )
                if (
                    pending.acquisition is not None
                    and not pending.acquisition.fully_published
                ):
                    self._host_acquisition.submit(pending)
                prefetch_fully_published = len(pending.prefetched_layers) == (
                    self._model_layer_count
                )
                progress = self._hicache.progress(consumer_index)
                prefetch_fully_ready = progress is not None and progress.complete
                active_batch = self._forward_lifecycle.active
                if active_batch is None:  # pragma: no cover - activated above
                    raise RuntimeError("host-staged batch lost its execution epoch")
                ready_stock_fastpath = (
                    prefetch_fully_ready
                    and self._execution_config.host_execution_mode
                    is HostExecutionMode.AUTO
                    and selected.selection_reason
                    not in {"calibration_probe", "consumer_policy_probe"}
                )
                if ready_stock_fastpath or not selected.uses_dependency_protocol:
                    self._host_acquisition.publish_missing(pending)
                    self._stock_forward = True
                    self._activate_stock_prefetch(bindings, pending, count_batch=False)
                    self._stats["host_direct_batches"] = (
                        self._stats.get("host_direct_batches", 0) + 1
                    )
                    if mixed_host_batch:
                        self._stats["host_mixed_direct_batches"] = (
                            self._stats.get("host_mixed_direct_batches", 0) + 1
                        )
                    if ready_stock_fastpath:
                        self._stats["host_bound_after_full_ready_batches"] = (
                            self._stats.get("host_bound_after_full_ready_batches", 0)
                            + 1
                        )
                    return

                if prefetch_fully_published:
                    self._stats["host_typed_after_full_publication_batches"] = (
                        self._stats.get("host_typed_after_full_publication_batches", 0)
                        + 1
                    )

                incremental_setup_started = time.perf_counter_ns()
                wrapper_select_started = time.perf_counter_ns()
                self._activate_wrapper_set(self._require_kernels().typed_wrappers())
                wrapper_select_ns = time.perf_counter_ns() - wrapper_select_started
                adoption_started = time.perf_counter_ns()
                typed_wrappers = self._adopt_typed_forward_metadata(
                    forward_batch, stock_metadata
                )
                adoption_ns = time.perf_counter_ns() - adoption_started
                if self._forward_lifecycle.active is None:  # pragma: no cover - set above
                    raise RuntimeError("incremental host batch lost its metadata")
                # Wrapper identity is part of the immutable semantic plan.
                # Publish the mutable per-layer consumer decision only after
                # that identity transition; doing it in the opposite order
                # correctly trips SglangForwardEpoch.require_unstarted.
                self._host_acquisition.plan_published_consumers(
                    pending,
                    self._forward_lifecycle.active,
                )
                event_partition_layer = next(
                    (
                        local_layer
                        for local_layer, layer_prefetch in sorted(
                            pending.prefetched_layers.items()
                        )
                        if layer_prefetch.transfer_first_slot is not None
                    ),
                    None,
                )
                # A producer descriptor alone does not imply a partial
                # consumer.  Cache-placement and external-only forwards have
                # no direct numerical work (overlap_initial is false); they
                # must wait for the producer event and use the preacquired
                # stock alias without allocating a runnable partition.
                if event_partition_layer is not None and selected.overlap_initial:
                    partition_layer_id = self._model_start_layer + event_partition_layer
                    self._require_attention_executor().prepare_arriving_plans(
                        batch=self._forward_lifecycle.active,
                        wrappers=typed_wrappers,
                        layer_id=partition_layer_id,
                        kv_cache=(
                            self.token_to_kv_pool._get_key_buffer(partition_layer_id),
                            self.token_to_kv_pool._get_value_buffer(partition_layer_id),
                        ),
                    )
                self._forward_lifecycle.active.incremental_metadata_setup_ns = (
                    time.perf_counter_ns() - incremental_setup_started
                )
                for counter, elapsed in (
                    ("incremental_wrapper_select_cpu_ns", wrapper_select_ns),
                    ("incremental_metadata_adoption_cpu_ns", adoption_ns),
                    (
                        "incremental_metadata_setup_cpu_ns",
                        self._forward_lifecycle.active.incremental_metadata_setup_ns,
                    ),
                ):
                    self._stats[counter] = self._stats.get(counter, 0) + elapsed
                selected_counter = (
                    "host_device_bulk_batches"
                    if selected.uses_device_bulk
                    else "host_incremental_batches"
                )
                self._stats[selected_counter] = self._stats.get(selected_counter, 0) + 1
                if mixed_host_batch:
                    self._stats["host_typed_mixed_batches"] = (
                        self._stats.get("host_typed_mixed_batches", 0) + 1
                    )
                return
            stock_metadata = self.forward_metadata
            self._init_external_metadata(forward_batch, pending, bindings=bindings)
            if self._tier_service.is_nvme:
                batch = self._forward_lifecycle.active
                if batch is None:
                    raise RuntimeError("NVMe metadata produced no active batch")
                self._activate_wrapper_set(self._require_kernels().typed_wrappers())
                typed_wrappers = self._adopt_typed_forward_metadata(
                    forward_batch, stock_metadata
                )
                self._require_attention_executor().prepare_nvme_batch(
                    batch=batch,
                    wrappers=typed_wrappers,
                    ordering_stream=torch.cuda.current_stream(),
                    tile_compute_ns=self._host_cost_model.tile_compute_ns,
                    kv_cache_for_layer=lambda layer_id: (
                        self.token_to_kv_pool._get_key_buffer(layer_id),
                        self.token_to_kv_pool._get_value_buffer(layer_id),
                    ),
                )
        except Exception as error:
            try:
                self.abort_active_forward(pending)
            except BaseException as cleanup_error:
                error.add_note(
                    "abnormal forward cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            self._stats["hicache_fallback_batches"] += 1
            self._stats["last_hicache_fallback"] = str(error)
            self._write_stats()
            raise RuntimeError(
                "NTA failed to bind the FlashInfer batch; stock fallback is "
                "disabled because it would bypass request-level semantics"
            ) from error
        finally:
            self.use_paged = original_use_paged

    def _record_host_selection(self, selected: HostExecutionPlan) -> None:
        configured = self._execution_config.host_execution_mode
        forced_forms = {
            HostExecutionMode.DIRECT: HostExecutionForm.DIRECT,
            HostExecutionMode.DEVICE_BULK: HostExecutionForm.DEVICE_BULK,
            HostExecutionMode.DEPENDENCY_AWARE: HostExecutionForm.DEPENDENCY_AWARE,
        }
        expected = forced_forms.get(configured)
        if expected is not None and selected.form is not expected:
            raise RuntimeError(
                "forced host execution form disagrees with the selected plan"
            )
        self._stats["host_selection_predicted_atomic_ns"] = (
            self._stats.get("host_selection_predicted_atomic_ns", 0)
            + selected.predicted_atomic_ns
        )
        self._stats["host_selection_predicted_selected_ns"] = (
            self._stats.get("host_selection_predicted_selected_ns", 0)
            + selected.predicted_incremental_ns
        )
        reason_key = f"host_selection_{selected.selection_reason}_batches"
        self._stats[reason_key] = self._stats.get(reason_key, 0) + 1

    def _init_external_metadata(
        self,
        forward_batch: Any,
        pending: PendingHostLoad,
        *,
        bindings: tuple[RequestBinding, ...] | None = None,
        count_batch: bool = True,
    ) -> HostExecutionPlan | None:
        self._host_acquisition.account_selection(pending)
        if self._opportunity_trace is not None and count_batch:
            self._active_opportunity_batch = self._opportunity_batch
            self._opportunity_batch += 1
        if bindings is None:
            bindings = self._bind_forward_requests(
                forward_batch, allow_capture_ids=False
            )
        engine_batch = self._forward_lifecycle.engine_batch
        if engine_batch is None:
            raise RuntimeError("external metadata has no engine batch epoch")
        planned = self._metadata_planner.plan(
            forward_batch=forward_batch,
            wrapper_metadata=self.forward_metadata,
            pending=pending,
            bindings=bindings,
            engine_batch=engine_batch,
            host_cost_model=self._host_cost_model,
            calibration_probe=(
                self._execution_config.host_execution_mode is HostExecutionMode.AUTO
                and self._incremental_calibration_probes_remaining > 0
            ),
            count_batch=count_batch,
        )
        if self._tier_service.is_host_staged:
            execution = planned.host_execution
            pending.arrival_profile_active = bool(
                pending.arrival_profiling
                and execution is not None
                and not execution.uses_dependency_protocol
            )
        if self._tier_service.is_host_staged:
            planned.batch.acquisition = HostForwardAcquisition(pending)
        self._forward_lifecycle.activate(planned.batch)
        return planned.host_execution

    def _advance_deadline_frontier(
        self,
        pending: PendingHostLoad,
        completed_local_layer: int,
        *,
        fragment: DeadlineFragment | None = None,
    ) -> None:
        """Hand one framework consumer edge to the Host acquisition owner."""
        batch = self._forward_lifecycle.active
        if batch is None or batch.pending_host_load is not pending:
            raise RuntimeError("deadline frontier lost its active HiCache lease")
        enqueue_fragment = (
            None
            if fragment is None
            else lambda: self._enqueue_fragment_lookahead(
                fragment.wrapper,
                self._model_start_layer + completed_local_layer,
                fragment.object_count,
                fragment.host_execution,
                fragment.stream,
            )
        )
        self._host_acquisition.advance_after_attention(
            pending,
            batch,
            completed_local_layer,
            enqueue_fragment=enqueue_fragment,
        )

    def _require_host_execution_plan(self) -> HostExecutionPlan:
        batch = self._forward_lifecycle.active
        if batch is None or batch.pending_host_load is None:
            raise RuntimeError("host execution plan has no active HiCache load")
        if batch.host_execution is None:
            raise RuntimeError("host-staged batch has no execution decision")
        return batch.host_execution

    def _enqueue_fragment_lookahead(
        self,
        wrapper: Any,
        layer_id: int,
        object_count: int,
        host_execution: HostExecutionPlan,
        stream: torch.cuda.Stream,
    ) -> bool:
        """Stage one next-layer contributor wave during post-attention compute."""
        if (
            not self._fragment_enabled
            or self.num_wrappers != 1
            or not host_execution.uses_dependency_protocol
            or host_execution.rounds <= 1
        ):
            return False
        batch = self._forward_lifecycle.active
        if batch is None or batch.pending_host_load is None:
            return False
        pending = batch.pending_host_load
        device_pool = pending.controller.mem_pool_device
        start_layer = int(getattr(device_pool, "start_layer", 0))
        next_layer_id = layer_id + 1
        next_local_layer = next_layer_id - start_layer
        if next_local_layer < 0 or next_local_layer >= int(
            pending.controller.layer_num
        ):
            return False
        if next_layer_id in batch.fragment_lookahead:
            raise RuntimeError("duplicate fragment lookahead for one attention layer")

        allocation = self._require_materializer().allocation(wrapper)
        if (
            allocation is None
            or allocation.object_count != object_count
            or allocation.indexed_geometry is None
            or object_count % 2 != 0
        ):
            raise RuntimeError("fragment lookahead has no reusable indexed directory")
        first_object_count = host_execution.block_counts[0]
        if (
            first_object_count <= 0
            or first_object_count >= object_count
            or first_object_count % 2 != 0
        ):
            raise RuntimeError("fragment lookahead requires one complete K/V wave")
        if (
            len(allocation.object_transfer_bytes) != object_count
            or sum(allocation.object_transfer_bytes) != allocation.transfer_bytes
        ):
            raise RuntimeError("fragment lookahead has incomplete byte ownership")
        fragment_transfer_bytes = sum(
            allocation.object_transfer_bytes[:first_object_count]
        )
        if fragment_transfer_bytes <= 0:
            raise RuntimeError("fragment lookahead has no physical payload")

        host_key = pending.controller.mem_pool_host.k_data_refs[next_local_layer]
        host_value = pending.controller.mem_pool_host.v_data_refs[next_local_layer]
        key_cache = device_pool._get_key_buffer(next_layer_id)
        value_cache = device_pool._get_value_buffer(next_layer_id)
        geometry = (
            key_cache[0].numel() * key_cache.element_size(),
            value_cache[0].numel() * value_cache.element_size(),
            host_key.stride(0) * host_key.element_size(),
            host_value.stride(0) * host_value.element_size(),
            key_cache.stride(0) * key_cache.element_size(),
            value_cache.stride(0) * value_cache.element_size(),
            int(host_key.shape[0]),
            int(host_value.shape[0]),
            int(key_cache.shape[0]),
            int(value_cache.shape[0]),
        )
        if geometry != allocation.indexed_geometry:
            raise RuntimeError(
                "next-layer KV geometry changed during fragment lookahead"
            )

        attention_done = torch.cuda.Event()
        ready_event = torch.cuda.Event()
        attention_done.record(stream)
        phase_program = self._require_kernels().transport_program()
        transfer_profile = (
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            if self._profile_transfer
            else None
        )
        prefetch_stream = self._host_transport.stream
        with torch.cuda.stream(prefetch_stream):
            prefetch_stream.wait_event(attention_done)
            phase_program.rebind_indexed_host_pairs(
                self._runtime,
                0,
                object_count // 2,
                host_key.data_ptr(),
                key_cache.data_ptr(),
                host_value.data_ptr(),
                value_cache.data_ptr(),
                prefetch_stream,
            )
            if transfer_profile is not None:
                transfer_profile[0].record(prefetch_stream)
            phase_program.preload_host_pairs(
                self._runtime,
                0,
                first_object_count // 2,
                prefetch_stream,
            )
            if transfer_profile is not None:
                transfer_profile[1].record(prefetch_stream)
            ready_event.record(prefetch_stream)
        if transfer_profile is not None:
            self._transfer_profiles.append(
                (*transfer_profile, fragment_transfer_bytes, "fragment")
            )
        batch.fragment_lookahead[next_layer_id] = _FragmentLookahead(
            next_layer_id,
            id(wrapper),
            object_count,
            first_object_count,
            host_key.data_ptr(),
            key_cache.data_ptr(),
            host_value.data_ptr(),
            value_cache.data_ptr(),
            ready_event,
        )
        self._stats["fragment_lookahead_layers"] += 1
        self._stats["fragment_lookahead_objects"] += first_object_count
        self._stats["fragment_lookahead_bytes"] = (
            self._stats.get("fragment_lookahead_bytes", 0) + fragment_transfer_bytes
        )
        self._stats["fragment_remaining_rounds"] += host_execution.rounds - 1
        return True

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
            raise RuntimeError(
                "framework CUDA graphs require fully materialized KV and the "
                "stock FlashInfer consumer"
            )
        if layer.logit_cap not in (None, 0, 0.0):
            raise RuntimeError("NTA's FlashInfer adapter does not support logit caps")
        batch = self._forward_lifecycle.active
        if batch is None:
            raise RuntimeError("NTA attention ran without request metadata")

        pending = batch.pending_host_load
        local_layer = int(layer.layer_id) - self._model_start_layer
        acquisition = (
            None if batch.acquisition is None else batch.acquisition.layer(local_layer)
        )
        prefetch_event_ordered = (
            acquisition is not None
            and id(acquisition.ready_event) in batch.ordered_prefetch_event_ids
        )
        modeled_ready_by_attention = (
            acquisition is not None
            and local_layer in batch.modeled_ready_by_attention_layers
        )
        progressive_consumer_planned = (
            acquisition is not None
            and local_layer in batch.planned_progressive_consumer_layers
            and not modeled_ready_by_attention
        )
        dispatch = select_attention_dispatch(
            pending=pending,
            host_execution=batch.host_execution,
            acquisition=acquisition,
            layer_id=int(layer.layer_id),
            prefetch_event_ordered=prefetch_event_ordered,
            progressive_consumer_planned=progressive_consumer_planned,
        )
        if modeled_ready_by_attention:
            self._stats["deadline_frontier_modeled_stock_dispatches"] = (
                self._stats.get("deadline_frontier_modeled_stock_dispatches", 0) + 1
            )
        if dispatch.acquisition is not None:
            event_id = id(dispatch.acquisition.ready_event)
            if prefetch_event_ordered:
                self._stats["stream_ordered_prefetch_event_reuses"] = (
                    self._stats.get("stream_ordered_prefetch_event_reuses", 0) + 1
                )
            else:
                batch.ordered_prefetch_event_ids.add(event_id)
                self._stats["stream_ordered_prefetch_events"] = (
                    self._stats.get("stream_ordered_prefetch_events", 0) + 1
                )
        typed_observation_required = (
            batch.host_execution is not None
            and batch.host_execution.selection_reason == "calibration_probe"
            and not batch.incremental_setup_observed
        )
        # The typed wrapper may alias a stock numerical wrapper after adopting
        # the same validated FlashInfer plan.  Use that zero-overhead alias only
        # after the transport event is complete.  An ARRIVING_PREFETCH owns an
        # in-flight acquisition and must reach the typed partial consumer so
        # direct work can run before its external dependencies become ready.
        if use_preloaded_stock_alias(
            dispatch,
            alias_available=self._forward_lifecycle.has_wrapper_alias(id(wrapper)),
            typed_observation_required=typed_observation_required,
        ):
            return self._run_preloaded_stock_layer(
                wrapper,
                q,
                kv_cache,
                layer,
                causal=causal,
                window_left=window_left,
            )

        q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        output = torch.empty_like(q)
        verify_attention = self._verification.attention
        if verify_attention and self._verification.attention_mixed_only:
            verify_attention = len(batch.bindings) > 1
        verify_execution = verify_attention or self._verification.execution
        verify_transfer = self._verification.transfer
        if verify_execution:
            output.fill_(float("nan"))
        wrapper._causal = causal
        wrapper._window_left = window_left
        wrapper._logits_soft_cap = 0.0
        wrapper._sm_scale = layer.scaling

        observe_setup = (
            batch.incremental_metadata_setup_ns > 0
            and not batch.incremental_setup_observed
        )
        measure_topology = self._profile_cpu or observe_setup
        topology_started = time.perf_counter_ns() if measure_topology else 0
        semantic_elapsed = self._validate_semantic_wrapper_plan(
            wrapper, layer, kv_cache, verify=verify_execution
        )
        execution_setup_elapsed = (
            time.perf_counter_ns() - topology_started if measure_topology else 0
        )
        topology_elapsed = max(0, execution_setup_elapsed - semantic_elapsed)
        if self._profile_cpu:
            self._stats["semantic_wrapper_plan_lookup_cpu_ns"] += topology_elapsed

        stream = torch.cuda.current_stream()
        run_options = {
            "k_scale": layer.k_scale_float,
            "v_scale": layer.v_scale_float,
        }
        final_layer = (
            int(layer.layer_id) - self._model_start_layer + 1 == self._model_layer_count
        )
        enqueue_started = time.perf_counter_ns()
        service_probe = (
            batch.host_execution is not None
            and batch.host_execution.selection_reason == "calibration_probe"
            and not batch.incremental_setup_observed
            and dispatch.kind
            in {
                AttentionDispatchKind.HOST_INCREMENTAL,
                AttentionDispatchKind.ARRIVING_PREFETCH,
            }
        )
        partial_policy_probe = (
            dispatch.kind is AttentionDispatchKind.ARRIVING_PREFETCH
            and batch.host_execution is not None
            and batch.host_execution.selection_reason == "consumer_policy_probe"
            and pending is not None
            and pending.arrival_profile_key is not None
        )
        gpu_profile = None
        if self._profile_gpu or service_probe or partial_policy_probe:
            gpu_profile = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            gpu_profile[0].record(stream)

        executor = self._require_attention_executor()
        if dispatch.kind in {
            AttentionDispatchKind.HOST_INCREMENTAL,
            AttentionDispatchKind.HOST_DEVICE_BULK,
        }:
            outcome = executor.execute_host(
                dispatch=dispatch,
                batch=batch,
                wrapper=wrapper,
                q=q,
                kv_cache=kv_cache,
                output=output,
                layer=layer,
                stream=stream,
                run_options=run_options,
                causal=causal,
                window_left=window_left,
                final_layer=final_layer,
                verify_execution=verify_execution,
                verify_transfer=verify_transfer,
                observe_setup=observe_setup,
                enqueue_started_ns=enqueue_started,
                host_cost_model=self._host_cost_model,
                active_opportunity_batch=self._active_opportunity_batch,
            )
            if outcome.output is None:
                raise RuntimeError("host attention executor returned no output")
            output = outcome.output
        else:
            outcome = executor.execute_non_host(
                dispatch=dispatch,
                batch=batch,
                wrapper=wrapper,
                q=q,
                kv_cache=kv_cache,
                output=output,
                layer=layer,
                stream=stream,
                run_options=run_options,
                final_layer=final_layer,
                verify_execution=verify_execution,
                verify_transfer=verify_transfer,
                tile_compute_ns=self._host_cost_model.tile_compute_ns,
            )

        elapsed_ns = time.perf_counter_ns() - enqueue_started
        if (
            dispatch.kind
            in {
                AttentionDispatchKind.HOST_INCREMENTAL,
                AttentionDispatchKind.ARRIVING_PREFETCH,
                AttentionDispatchKind.PRELOADED,
            }
            and batch.incremental_metadata_setup_ns > 0
            and not batch.incremental_setup_observed
        ):
            dispatch_elapsed = outcome.setup_dispatch_elapsed_ns
            batch.incremental_setup_observation_ns = (
                batch.incremental_metadata_setup_ns
                + execution_setup_elapsed
                + (elapsed_ns if dispatch_elapsed is None else dispatch_elapsed)
            )
            batch.incremental_setup_observed = True
        if self._profile_cpu:
            attention_form = dispatch.kind.value
            self._stats["phase_enqueue_cpu_ns"] = (
                self._stats.get("phase_enqueue_cpu_ns", 0) + elapsed_ns
            )
            self._stats[f"{attention_form}_enqueue_cpu_ns"] = (
                self._stats.get(f"{attention_form}_enqueue_cpu_ns", 0) + elapsed_ns
            )
            self._stats[f"{attention_form}_enqueue_layers"] = (
                self._stats.get(f"{attention_form}_enqueue_layers", 0) + 1
            )
        if gpu_profile is not None:
            gpu_profile[1].record(stream)
            if partial_policy_probe:
                if pending is None:  # pragma: no cover - guarded above
                    raise RuntimeError("partial policy probe lost its Host lease")
                self._consumer_calibration.record_partial_profile(
                    pending=pending,
                    start=gpu_profile[0],
                    finish=gpu_profile[1],
                )
            service_prediction_ns = None
            service_prediction_scale = None
            if service_probe:
                execution = batch.host_execution
                if execution is None:  # pragma: no cover - guarded above
                    raise RuntimeError("service probe lost its execution plan")
                setup_per_unit_ns = math.ceil(
                    (self._host_cost_model.incremental_setup_ns or 0)
                    / execution.scope_units
                )
                service_prediction_ns = max(
                    1,
                    execution.predicted_incremental_per_unit_ns
                    - setup_per_unit_ns,
                )
                service_prediction_scale = execution.incremental_service_scale
            self._operator_profiles.append(
                _OperatorProfile(
                    *gpu_profile,
                    dispatch.kind.value,
                    1,
                    service_prediction_ns,
                    service_prediction_scale,
                )
            )

        if (
            pending is not None
            and batch.acquisition is not None
            and batch.acquisition.tier is AcquisitionTier.HOST_STAGED
            and verify_transfer
        ):
            self._require_attention_verifier().verify_layer_transfer(
                batch,
                int(layer.layer_id),
                kv_cache,
            )
        if verify_execution:
            if outcome.epoch is None:
                stream.synchronize()
            if not torch.isfinite(output).all():
                raise RuntimeError(
                    f"instrumented FlashInfer did not write layer {layer.layer_id}"
                )
        if verify_attention:
            self._require_attention_verifier().verify_attention_output(
                wrapper,
                q,
                kv_cache,
                output,
                layer,
                causal=causal,
                window_left=window_left,
            )

        if pending is not None:
            self._commit_external_layer(
                batch=batch,
                pending=pending,
                layer=layer,
                local_layer=dispatch.local_layer,
                native_dispatch=True,
                progressive_consumer=outcome.progressive_consumer,
                indexed_object_count=outcome.indexed_object_count,
                record_semantic=True,
                fragment=outcome.deadline_fragment,
            )
        else:
            self._record_execution_layer(
                layer,
                indexed_object_count=outcome.indexed_object_count,
                final_layer=final_layer,
            )
            if final_layer:
                self._commit_incremental_setup_observation(batch)
                self._finish_forward(batch)
        return output.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _commit_incremental_setup_observation(self, batch: SglangForwardEpoch) -> None:
        """Publish one completed epoch's control cost for later decisions."""

        observed_ns = batch.incremental_setup_observation_ns
        if observed_ns <= 0:
            return
        is_probe = (
            batch.host_execution is not None
            and batch.host_execution.selection_reason == "calibration_probe"
        )
        if is_probe:
            self._incremental_calibration_probes_remaining = max(
                0, self._incremental_calibration_probes_remaining - 1
            )
        initialization_sample = (
            is_probe and self._incremental_initialization_probes_remaining > 0
        )
        if initialization_sample:
            self._incremental_initialization_probes_remaining -= 1
            self._stats["incremental_initialization_samples"] += 1
            self._stats["incremental_initialization_setup_ns"] += observed_ns
        else:
            first_sample = self._incremental_setup_samples == 0
            self._host_cost_model = (
                self._host_cost_model.with_incremental_setup_observation(
                    elapsed_ns=observed_ns,
                    alpha=1.0 if first_sample else 0.25,
                    maximum_step_ratio=64.0 if first_sample else 4.0,
                )
            )
            self._incremental_setup_samples += 1
        batch.incremental_setup_observation_ns = 0
        self._stats["incremental_setup_samples"] = self._incremental_setup_samples
        self._stats["incremental_setup_calibrated"] = (
            self._host_cost_model.incremental_setup_ns is not None
        )
        self._stats["incremental_setup_ns"] = self._host_cost_model.incremental_setup_ns
        self._stats["incremental_setup_observed_ns_total"] = (
            self._stats.get("incremental_setup_observed_ns_total", 0) + observed_ns
        )
        self._stats["incremental_setup_observed_ns_max"] = max(
            self._stats.get("incremental_setup_observed_ns_max", 0),
            observed_ns,
        )
        self._stats["incremental_calibration_probes_remaining"] = (
            self._incremental_calibration_probes_remaining
        )

    def _wait_for_stock_external_layer(
        self, batch: SglangForwardEpoch, layer: Any
    ) -> int:
        """Join the producer event before stock attention consumes a page."""

        acquisition = batch.acquisition
        if acquisition is None:
            raise RuntimeError("stock external attention has no acquisition owner")
        local_layer = int(layer.layer_id) - self._model_start_layer
        published = acquisition.layer(local_layer)
        if published is None:
            raise RuntimeError(
                "stock external attention reached a layer without an exact "
                f"prefetch event: {layer.layer_id}"
            )
        stream = torch.cuda.current_stream()
        if self._profile_barrier:
            arrive = torch.cuda.Event(enable_timing=True)
            arrive.record(stream)
            self._barrier_profiles.append(
                _BarrierProfile(
                    arrive,
                    published.ready_event,
                    int(layer.layer_id),
                    "attention_layer",
                )
            )
        acquisition.consume_layer(published, stream, wait_for_ready=True)
        return local_layer

    def _run_preloaded_stock_layer(
        self,
        typed_wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        layer: Any,
        *,
        causal: bool,
        window_left: int,
    ) -> torch.Tensor:
        batch = self._forward_lifecycle.active
        if batch is None or batch.pending_host_load is None:
            raise RuntimeError("preloaded stock layer has no external lease")
        pending = batch.pending_host_load
        local_layer = self._wait_for_stock_external_layer(batch, layer)
        profile = None
        stream = torch.cuda.current_stream()
        if self._profile_gpu:
            profile = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            profile[0].record(stream)
        output = self._run_ready_stock_numerical(
            typed_wrapper,
            q,
            kv_cache,
            layer,
            causal=causal,
            window_left=window_left,
        )
        if profile is not None:
            profile[1].record(stream)
            self._operator_profiles.append(
                _OperatorProfile(*profile, "preloaded_stock", 1)
            )
        self._stats["stock_attention_launches"] += 1
        self._stats["lookahead_bound_launches"] += 1
        if (
            batch.acquisition is not None
            and batch.acquisition.tier is AcquisitionTier.NVME
        ):
            semantic = batch.semantic_plans.get(id(typed_wrapper))
            if semantic is None:
                raise RuntimeError("NVMe stock alias lost its exact semantic plan")
            self._stats["request_work_completed"] += semantic.schedule.work_count
            self._stats["tier_external_layers"] += 1
            self._stats["nvme_preacquired_launches"] = (
                self._stats.get("nvme_preacquired_launches", 0) + 1
            )
            self._stats["nvme_ready_stock_launches"] = (
                self._stats.get("nvme_ready_stock_launches", 0) + 1
            )
        if (
            self._verification.transfer
            and batch.acquisition is not None
            and batch.acquisition.tier is AcquisitionTier.HOST_STAGED
        ):
            self._require_attention_verifier().verify_layer_transfer(
                batch,
                int(layer.layer_id),
                kv_cache,
            )
        self._commit_external_layer(
            batch=batch,
            pending=pending,
            layer=layer,
            local_layer=local_layer,
            native_dispatch=False,
            progressive_consumer=False,
        )
        return output.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: Any,
        forward_batch: Any,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        if self._resident_reference_forward:
            return FlashInferAttnBackend.forward_decode(
                self,
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
            )
        batch = self._forward_lifecycle.active
        if batch is None:
            raise RuntimeError(
                "NTA decode ran without transformed request metadata; stock "
                "dispatch is disabled"
            )
        if batch.pending_host_load is not None:
            self._layer_calibration.record(
                batch=self._forward_lifecycle.active,
                phase="decode",
                query=q,
                global_layer=int(layer.layer_id),
            )
            self._consumer_calibration.record_arrival(
                batch=batch,
                phase="decode",
                query=q,
                global_layer=int(layer.layer_id),
            )
        if self._stock_forward:
            pending = batch.pending_host_load
            if pending is None:  # pragma: no cover - resident path returns above
                raise RuntimeError("resident decode entered an external epoch")
            self._stats["stock_attention_launches"] += 1
            local_layer = self._wait_for_stock_external_layer(batch, layer)
            self._stats["decode_launches"] += 1
            stock_profile = None
            if pending.arrival_profile_active:
                stream = torch.cuda.current_stream()
                stock_profile = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                stock_profile[0].record(stream)
            output = FlashInferAttnBackend.forward_decode(
                self,
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
            )
            if stock_profile is not None:
                stock_profile[1].record(stream)
                self._consumer_calibration.record_stock_profile(
                    pending=pending,
                    global_layer=int(layer.layer_id),
                    start=stock_profile[0],
                    finish=stock_profile[1],
                )
            self._commit_external_layer(
                batch=batch,
                pending=pending,
                layer=layer,
                local_layer=local_layer,
                native_dispatch=False,
                progressive_consumer=False,
            )
            return output
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
        if self._resident_reference_forward:
            return FlashInferAttnBackend.forward_extend(
                self,
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
            )
        batch = self._forward_lifecycle.active
        if batch is None:
            raise RuntimeError(
                "NTA prefill ran without transformed request metadata; stock "
                "dispatch is disabled"
            )
        if batch.pending_host_load is not None:
            self._layer_calibration.record(
                batch=self._forward_lifecycle.active,
                phase="extend",
                query=q,
                global_layer=int(layer.layer_id),
            )
            self._consumer_calibration.record_arrival(
                batch=batch,
                phase="extend",
                query=q,
                global_layer=int(layer.layer_id),
            )
        if self._stock_forward:
            pending = batch.pending_host_load
            if pending is None:  # pragma: no cover - resident path returns above
                raise RuntimeError("resident prefill entered an external epoch")
            self._stats["stock_attention_launches"] += 1
            local_layer = self._wait_for_stock_external_layer(batch, layer)
            self._stats["prefill_launches"] += 1
            stock_profile = None
            if pending.arrival_profile_active:
                stream = torch.cuda.current_stream()
                stock_profile = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                stock_profile[0].record(stream)
            output = FlashInferAttnBackend.forward_extend(
                self,
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
            )
            if stock_profile is not None:
                stock_profile[1].record(stream)
                self._consumer_calibration.record_stock_profile(
                    pending=pending,
                    global_layer=int(layer.layer_id),
                    start=stock_profile[0],
                    finish=stock_profile[1],
                )
            self._commit_external_layer(
                batch=batch,
                pending=pending,
                layer=layer,
                local_layer=local_layer,
                native_dispatch=False,
                progressive_consumer=False,
            )
            return output
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

    def _collect_transfer_profiles(self) -> None:
        self._host_movers.collect_profiles()
        pending: list[tuple[torch.cuda.Event, torch.cuda.Event, int, str]] = []
        for start, finish, transfer_bytes, kind in self._transfer_profiles:
            if not finish.query():
                pending.append((start, finish, transfer_bytes, kind))
                continue
            milliseconds = start.elapsed_time(finish)
            elapsed_ns = max(1, round(milliseconds * 1_000_000.0))
            previous_bandwidth = self._host_cost_model.bandwidth_bytes_per_second
            self._host_cost_model = self._host_cost_model.with_transfer_observation(
                transfer_bytes=transfer_bytes,
                elapsed_ns=elapsed_ns,
            )
            if self._host_cost_model.bandwidth_bytes_per_second != previous_bandwidth:
                self._stats["cost_model_transfer_samples"] += 1
                self._stats["cost_model_bandwidth_bps"] = (
                    self._host_cost_model.bandwidth_bytes_per_second
                )
            self._stats["profiled_transfer_batches"] = (
                self._stats.get("profiled_transfer_batches", 0) + 1
            )
            self._stats["profiled_transfer_bytes"] = (
                self._stats.get("profiled_transfer_bytes", 0) + transfer_bytes
            )
            self._stats["profiled_transfer_gpu_ms"] = (
                self._stats.get("profiled_transfer_gpu_ms", 0.0) + milliseconds
            )
            prefix = f"profiled_{kind}_transfer"
            self._stats[f"{prefix}_batches"] = (
                self._stats.get(f"{prefix}_batches", 0) + 1
            )
            self._stats[f"{prefix}_bytes"] = (
                self._stats.get(f"{prefix}_bytes", 0) + transfer_bytes
            )
            self._stats[f"{prefix}_gpu_ms"] = (
                self._stats.get(f"{prefix}_gpu_ms", 0.0) + milliseconds
            )
        self._transfer_profiles[:] = pending
        milliseconds = float(self._stats.get("profiled_transfer_gpu_ms", 0.0))
        if milliseconds > 0:
            self._stats["profiled_transfer_gib_per_second"] = (
                float(self._stats["profiled_transfer_bytes"])
                / (1 << 30)
                / (milliseconds / 1_000.0)
            )
        for kind in ("pipeline", "fragment", "demand"):
            prefix = f"profiled_{kind}_transfer"
            kind_milliseconds = float(self._stats.get(f"{prefix}_gpu_ms", 0.0))
            if kind_milliseconds > 0:
                self._stats[f"{prefix}_gib_per_second"] = (
                    float(self._stats[f"{prefix}_bytes"])
                    / (1 << 30)
                    / (kind_milliseconds / 1_000.0)
                )
        pending_operators: list[_OperatorProfile] = []
        for profile in self._operator_profiles:
            if not profile.finish.query():
                pending_operators.append(profile)
                continue
            milliseconds = profile.start.elapsed_time(profile.finish)
            if profile.service_prediction_ns is not None:
                elapsed_ns = max(1, round(milliseconds * 1_000_000.0))
                first_sample = self._incremental_service_samples == 0
                self._host_cost_model = (
                    self._host_cost_model.with_incremental_service_observation(
                        predicted_ns=profile.service_prediction_ns,
                        predicted_scale=profile.service_prediction_scale,
                        elapsed_ns=elapsed_ns,
                        alpha=1.0 if first_sample else 0.25,
                        maximum_step_ratio=64.0 if first_sample else 4.0,
                    )
                )
                self._incremental_service_samples += 1
                self._stats["incremental_service_samples"] = (
                    self._incremental_service_samples
                )
                self._stats["incremental_service_scale"] = (
                    self._host_cost_model.incremental_service_scale
                )
                self._stats["incremental_service_calibrated"] = True
                self._stats["incremental_service_observed_ns_total"] = (
                    self._stats.get("incremental_service_observed_ns_total", 0)
                    + elapsed_ns
                )
            prefix = f"profiled_{profile.kind}_operator"
            self._stats[f"{prefix}_layers"] = (
                self._stats.get(f"{prefix}_layers", 0) + profile.covered_layers
            )
            self._stats[f"{prefix}_launches"] = (
                self._stats.get(f"{prefix}_launches", 0) + 1
            )
            self._stats[f"{prefix}_gpu_ms"] = (
                self._stats.get(f"{prefix}_gpu_ms", 0.0) + milliseconds
            )
        self._operator_profiles[:] = pending_operators
        self._consumer_calibration.collect()

    def _collect_barrier_profiles(self, *, already_synchronized: bool = False) -> None:
        if not self._barrier_profiles:
            return
        # Barrier pairs reuse the per-layer ready events across batches.
        # Profiling mode synchronizes before draining so every pair is final
        # and no event is re-recorded while a measurement is outstanding; the
        # sync cost is confined to NTA_PROFILE_BARRIER=1 runs, whose
        # host-side throughput is never an execution result.
        if not already_synchronized:
            torch.cuda.synchronize()
        for profile in self._barrier_profiles:
            signed_ready_after_arrival_ms = profile.arrive.elapsed_time(profile.ready)
            stall_ms = max(0.0, signed_ready_after_arrival_ms)
            self._stats["profiled_barrier_waits"] = (
                self._stats.get("profiled_barrier_waits", 0) + 1
            )
            self._stats["profiled_barrier_stall_gpu_ms"] = (
                self._stats.get("profiled_barrier_stall_gpu_ms", 0.0) + stall_ms
            )
            if stall_ms > 0.01:
                self._stats["profiled_barrier_stalled_waits"] = (
                    self._stats.get("profiled_barrier_stalled_waits", 0) + 1
                )
            self._stats["profiled_barrier_max_stall_gpu_ms"] = max(
                float(self._stats.get("profiled_barrier_max_stall_gpu_ms", 0.0)),
                stall_ms,
            )
            if profile.scope == "attention_layer":
                self._stats["profiled_attention_arrivals"] += 1
                self._stats["profiled_attention_stall_gpu_ms"] += stall_ms
                self._stats["profiled_attention_max_stall_gpu_ms"] = max(
                    self._stats["profiled_attention_max_stall_gpu_ms"],
                    stall_ms,
                )
                readiness = (
                    "profiled_attention_not_ready_at_arrival"
                    if signed_ready_after_arrival_ms > 0.0
                    else "profiled_attention_ready_at_arrival"
                )
                self._stats[readiness] += 1
                if stall_ms > 0.01:
                    self._stats["profiled_attention_materially_stalled_arrivals"] += 1
                self._barrier_stall_by_layer[profile.layer_id] = (
                    self._barrier_stall_by_layer.get(profile.layer_id, 0.0) + stall_ms
                )
        self._barrier_profiles.clear()

    def _stats_report(self, *, lifecycle: str = "served") -> dict[str, Any]:
        self._collect_transfer_profiles()
        self._layer_calibration.collect()
        self._collect_barrier_profiles()
        report = dict(self._stats)
        report.update(self._tier_service.stats())
        report["layer_service_curves"] = self._layer_calibration.report()
        report["consumer_policy_calibration"] = self._consumer_calibration.report()
        consumer_contract = _consumer_contract_for_stats(
            report,
            engine_version=self._engine_version,
        )
        report["consumer_contract"] = consumer_contract.as_dict()
        report["execution_protocol_status"] = consumer_contract.kind.value
        if self._barrier_stall_by_layer:
            report["profiled_attention_stall_by_layer_ms"] = {
                str(layer): round(stall, 4)
                for layer, stall in sorted(self._barrier_stall_by_layer.items())
            }
        kernel_report = self._require_kernels().contract_report()
        report["operator_contracts"] = list(kernel_report.operator_contracts)
        report["operator_plans"] = list(kernel_report.operator_plans)
        if kernel_report.transport_contract is not None:
            report["transport_contract"] = kernel_report.transport_contract
        report["tier_descriptors"] = [
            {
                "source_kind": descriptor.source_kind.name.lower(),
                "capabilities": _flag_value(descriptor.capabilities),
                "device_state": descriptor.device_state,
                "estimated_latency_ns": descriptor.estimated_latency_ns,
                "estimated_bandwidth_bytes_per_second": descriptor.estimated_bandwidth_bytes_per_second,
                "active": descriptor.active,
                "flags": descriptor.flags,
            }
            for descriptor in (self._runtime.tier_descriptor(tier) for tier in TierKind)
        ]
        report["verified_dual_form_operator_plans"] = (
            kernel_report.verified_dual_form_operator_plans
        )
        report.update(self._hicache.admission_stats())
        report.update(process_hook_stats())
        report["stats_lifecycle"] = lifecycle
        report["snapshot_unix_ns"] = time.time_ns()
        report["finished_unix_ns"] = report["snapshot_unix_ns"]
        return report

    def _publish_stats(
        self, *, observation_boundary: bool = False, wait: bool = False
    ) -> None:
        if self._stats_publisher is None:
            return
        if observation_boundary:
            self._quiesce_observation_boundary()
        self._stats_publisher.publish(self._stats_report(), wait=wait)

    def _write_stats(self, *, strict: bool = False) -> None:
        if self._closed:
            return
        shutdown_error: BaseException | None = None
        try:
            self._quiesce_observation_boundary()
        except BaseException as error:
            shutdown_error = error
        try:
            if self._stats_publisher is not None:
                self._stats_publisher.publish(
                    self._stats_report(lifecycle="shutdown"), wait=True
                )
        except BaseException as error:
            if shutdown_error is None:
                shutdown_error = error
        finally:
            try:
                if self._stats_publisher is not None:
                    self._stats_publisher.close()
            except BaseException as error:
                if shutdown_error is None:
                    shutdown_error = error
            try:
                self._close_resources()
            except BaseException as error:
                if shutdown_error is None:
                    shutdown_error = error
            self._closed = True
        if shutdown_error is not None and strict:
            raise RuntimeError(
                "NTA engine shutdown completed with a statistics or resource error"
            ) from shutdown_error
