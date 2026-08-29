"""SGLang 0.5.16 adapter for compiler-instrumented FlashInfer attention."""

from __future__ import annotations

import atexit
from dataclasses import replace
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
from nta_runtime.adapters.base import EngineBatch
from nta_runtime.adapters.sglang import (
    SglangAdapter,
    validate_sglang_attention_tier,
)
from nta_runtime.execution_core import ExecutionSession
from nta_runtime.execution_protocol import ProtocolKind
from nta_runtime.acquisition_scheduler import (
    AcquisitionServiceCurve,
    LayerAcquisitionModel,
)
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
from nta_runtime.engines.sglang_acquisition import HostLayerAcquisition
from nta_runtime.engines.sglang_transfer import (
    HostMoverController,
    HostMoverLeasePlan,
    HostTransferLeasePlan,
    build_host_transfer_lease_plan,
)
from nta_runtime.engines.sglang_pipeline import SglangHostTransport
from nta_runtime.engines.sglang_nvme import SglangNvmeAcquisitionPipeline
from nta_runtime.engines.sglang_config import (
    AUTO_INCREMENTAL_INITIALIZATION_PROBES,
    SglangBootstrapConfig,
    SglangExecutionTuning,
)
from nta_runtime.engines.sglang_metadata import SglangMetadataPlanner
from nta_runtime.engines.sglang_state import (
    _ActiveBatch,
    _BarrierProfile,
    _FragmentLookahead,
    _LayerServiceProfile,
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
    calibration_probe_end as _calibration_probe_end,
    pipeline_object_id as _pipeline_object_id,
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
        host_mover_policy = tuning.host_mover_policy
        host_mover_default_service_model = tuning.host_mover_default_service_model
        host_mover_calibration_samples = tuning.host_mover_calibration_samples
        self._layer_service_minimum_samples = tuning.layer_service_minimum_samples
        self._layer_service_maximum_samples = tuning.layer_service_maximum_samples
        self._layer_service_curves: dict[
            tuple[str, int, int], AcquisitionServiceCurve
        ] = {}
        self._copy_engine_max_operations = tuning.copy_engine_max_operations
        self._indexed_copy_target_bytes = tuning.indexed_copy_target_bytes
        self._indexed_copy_max_blocks = tuning.indexed_copy_max_blocks
        self._frontier_layers_per_wave = tuning.frontier_layers_per_wave
        self._sm_acquisition_waves = tuning.sm_acquisition_waves
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
        self._stock_wrapper_for_typed: dict[int, Any] = {}
        self._execution_epoch = 0
        self._current_engine_batch: EngineBatch | None = None
        self._active_batch: _ActiveBatch | None = None
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
                layer_service_minimum_samples=self._layer_service_minimum_samples,
                layer_service_maximum_samples=self._layer_service_maximum_samples,
                indexed_copy_target_bytes=self._indexed_copy_target_bytes,
                indexed_copy_max_blocks=self._indexed_copy_max_blocks,
                frontier_layers_per_wave=self._frontier_layers_per_wave,
                sm_acquisition_waves=self._sm_acquisition_waves,
                demand_graph_enabled=self._demand_graph_enabled,
                demand_graph_capacity=demand_graph_capacity,
                engine_version=observability.engine_version,
                revision=observability.revision,
            ),
            self._tier_service.stats(),
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
        if configured_stats is not None:
            stats_path = configured_stats
            if stats_path.suffix:
                stats_path = stats_path.with_name(
                    f"{stats_path.stem}.{os.getpid()}{stats_path.suffix}"
                )
            else:
                stats_path = stats_path / f"nta-sglang-{os.getpid()}.json"
            self._stats_publisher = StatsPublisher(stats_path)
            # SGLang terminates model workers with a signal on some otherwise
            # clean Engine shutdown paths, so Python atexit is not a reliable
            # setup-evidence boundary.  Persist a clearly typed setup snapshot
            # synchronously; the first served-batch/final report atomically
            # replaces it with numerical-consumer evidence.
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
            copy_engine_max_operations=self._copy_engine_max_operations,
            profile_barrier=self._profile_barrier,
            profile_cpu=self._profile_cpu,
            profile_transfer=self._profile_transfer,
            stats=self._stats,
            transfer_profiles=self._transfer_profiles,
            transfer_plan=self._host_transfer_lease_plan,
            transport_program=self._require_kernels().transport_program,
            collect_barrier_profiles=self._collect_barrier_profiles,
        )
        self._layer_service_profiles: list[_LayerServiceProfile] = []
        self._operator_profiles: list[
            tuple[torch.cuda.Event, torch.cuda.Event, str, int]
        ] = []
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
            stock_wrapper_for_typed=self._stock_wrapper_for_typed,
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
            self._hicache.set_acquire_callback(self._hold_host_load)
            self._hicache.set_deadline_model_callback(self._admission_deadline_model)
            self._hicache.set_admission_acquisition_callbacks(
                prepare=self._prepare_admission_acquisition,
                start=self._start_admission_acquisition,
            )
            if tuning.requires_typed_host_modules(bootstrap):
                self._require_kernels().prepare_typed_execution_modules(
                    runtime=self._runtime,
                    host_staged=True,
                    stream=torch.cuda.current_stream(),
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
        atexit.register(self._write_stats)

    def _hold_host_load(self, pending: PendingHostLoad) -> None:
        """Capture a HiCache lease before selecting its execution form.

        Ownership is captured by the hook before a batch can be admitted.
        Conventional execution publishes the complete producer immediately.
        A dense late-bound lease also names every byte that the forward will
        consume, so its finite transport queue starts immediately rather than
        waiting for FlashInfer metadata.  The later batch binding supplies only
        calibrated deadlines and work-unit mappings; it cannot expand physical
        ownership or restart transport.  Tenant-isolated and device-bulk causal
        arms still defer until their request/accounting contract is available.
        """

        if pending.controller.mem_pool_device is not self.token_to_kv_pool:
            raise RuntimeError("HiCache lease belongs to a different device pool")
        layer_count = int(pending.controller.layer_num)
        if layer_count != self._model_layer_count:
            raise RuntimeError("HiCache load and model layer counts disagree")
        self._account_tier_selection(pending)
        conventional = (
            self._execution_config.protocol.kind is ProtocolKind.CONVENTIONAL
        )
        eager_dense = (
            not conventional
            and not self._tenant_isolation_enabled
            and self._execution_config.host_execution_mode
            is not HostExecutionMode.DEVICE_BULK
        )
        initial_layers = 0
        if conventional:
            self._host_transport.prepare(
                pending,
                first_local_layer=0,
                last_local_layer=layer_count,
            )
            initial_layers = layer_count
        elif eager_dense:
            # Build exact physical descriptors at lease capture.  No active
            # ForwardBatch exists here; mover choice therefore uses only its
            # deployment-local link curve, never stale batch state.
            self._host_transfer_lease_plan(pending)
            pending.acquisition = HostLayerAcquisition(pending.layer_bytes)
            self._stats["host_acquisition_jobs_prepared"] = self._stats.get(
                "host_acquisition_jobs_prepared", 0
            ) + layer_count
            self._stats["lease_acquisition_groups_prepared"] = self._stats.get(
                "lease_acquisition_groups_prepared", 0
            ) + 1
            initial_layers = self._submit_host_acquisition(pending)
            if initial_layers != layer_count:
                raise RuntimeError(
                    "dense HiCache lease did not fill its finite acquisition queue"
                )
            self._stats["lease_acquisition_groups_started"] = self._stats.get(
                "lease_acquisition_groups_started", 0
            ) + 1
        self._stats["initial_acquisition_batches"] = (
            self._stats.get("initial_acquisition_batches", 0) + 1
        )
        self._stats["initial_acquisition_layers"] = (
            self._stats.get("initial_acquisition_layers", 0) + initial_layers
        )
        self._stats["initial_typed_gap_layers"] = self._stats.get(
            "initial_typed_gap_layers", 0
        ) + (layer_count - initial_layers)
        if initial_layers == 0:
            self._stats["schedule_bound_acquisition_batches"] = (
                self._stats.get("schedule_bound_acquisition_batches", 0) + 1
            )

    def _admission_deadline_model(
        self, pending: PendingHostLoad, batch: Any
    ) -> LayerAcquisitionModel | None:
        """Build a calibrated EDF model without synchronizing CUDA.

        Admission is allowed to delay a prefill only when both sides of the
        inequality are deployment observations: completed mover waves provide
        link service and completed attention-arrival intervals provide compute
        deadlines.  Missing shape or mover calibration returns ``None`` and the
        framework releases the batch to the typed partial consumer.
        """

        acquisition = getattr(pending, "acquisition", None)
        if acquisition is not None and acquisition.model is not None:
            return acquisition.model
        self._host_movers.collect_profiles()
        self._collect_layer_service_profiles()
        mover = pending.mover_plan
        if mover is None or not pending.layer_bytes or not pending.row_bytes_by_layer:
            return None
        curve = self._admission_shape_curve(batch)
        if curve is None:
            return None
        model = self._deadline_model_for_curve(pending, curve)
        if model is not None and acquisition is not None:
            acquisition.bind_model(model)
        return model

    def _deadline_model_for_curve(
        self,
        pending: PendingHostLoad,
        curve: AcquisitionServiceCurve,
    ) -> LayerAcquisitionModel | None:
        mover = pending.mover_plan
        if (
            mover is None
            or not curve.calibrated
            or not pending.layer_bytes
            or not pending.row_bytes_by_layer
        ):
            return None
        transfer_count = int(pending.device_indices.numel())
        if transfer_count <= 0 or transfer_count != mover.row_count:
            raise RuntimeError("HiCache deadline mover geometry changed")
        if not self._host_movers.lease_calibrated(pending):
            return None
        representative_bytes = self._host_movers.representative_wave_bytes(
            pending.row_bytes_by_layer, transfer_count
        )
        service_model = self._host_movers.service_model(representative_bytes)
        layer_service: list[int] = []
        for key_row_bytes, value_row_bytes in pending.row_bytes_by_layer:
            service_ns = service_model.candidate_ns(
                total_rows=transfer_count,
                copy_rows=mover.copy_row_count,
                copy_run_count=len(mover.copy_runs),
                row_bytes=key_row_bytes + value_row_bytes,
                copy_operations_per_run=2,
            )
            if service_ns is None:
                return None
            layer_service.append(service_ns)
        return LayerAcquisitionModel(
            layer_bytes=pending.layer_bytes,
            transfer_service_ns=tuple(layer_service),
            # The framework exposes no calibrated useful-compute interval
            # between batch admission and the first attention arrival.  Zero
            # is the conservative deadline origin; it does not make layer zero
            # a transport or consumer special case.
            initial_compute_ns=0,
            inter_layer_compute_ns=curve.conservative_interval_ns,
        )

    def _admission_shape_curve(self, batch: Any) -> AcquisitionServiceCurve | None:
        key = self._acquisition_shape_key(batch)
        if key is None:
            return None
        curve = self._layer_service_curves.get(key)
        return curve if curve is not None and curve.calibrated else None

    @staticmethod
    def _acquisition_shape_key(batch: Any) -> tuple[str, int, int] | None:
        """Resolve the same extend key at scheduler and ForwardBatch seams."""

        requests = tuple(getattr(batch, "reqs", ()) or ())
        query_rows = getattr(batch, "extend_num_tokens", None)
        batch_size = len(requests)
        if batch_size == 0:
            batch_size = int(getattr(batch, "batch_size", 0) or 0)
        if batch_size == 0:
            batch_size = len(tuple(getattr(batch, "rids", ()) or ()))
        if batch_size <= 0 or query_rows is None or int(query_rows) <= 0:
            return None
        return ("extend", int(query_rows), batch_size)

    def _prepare_host_acquisition_owner(
        self, pending: PendingHostLoad, batch: Any
    ) -> bool:
        """Bind one calibrated EDF proof to immutable physical ownership."""

        acquisition = getattr(pending, "acquisition", None)
        if acquisition is not None and acquisition.model is not None:
            return True
        if acquisition is None and pending.prefetched_layers:
            return False
        shape_key = self._acquisition_shape_key(batch)
        curve = self._admission_shape_curve(batch)
        if shape_key is None or curve is None:
            self._stats["host_acquisition_shape_uncalibrated"] = self._stats.get(
                "host_acquisition_shape_uncalibrated", 0
            ) + 1
            return False
        active = self._active_batch
        if active is not None and active.pending_host_load is pending:
            active.layer_service_key = shape_key
        self._host_movers.collect_profiles()
        self._host_transfer_lease_plan(
            pending,
            layer_service_key=shape_key,
            layer_curve=curve,
        )
        model = self._deadline_model_for_curve(pending, curve)
        if model is None:
            counter = (
                "host_acquisition_mover_uncalibrated"
                if not self._host_movers.lease_calibrated(pending)
                else "host_acquisition_model_rejected"
            )
            self._stats[counter] = self._stats.get(counter, 0) + 1
            return False
        if acquisition is None:
            acquisition = HostLayerAcquisition(pending.layer_bytes)
            pending.acquisition = acquisition
            self._stats["host_acquisition_jobs_prepared"] = self._stats.get(
                "host_acquisition_jobs_prepared", 0
            ) + len(model.layer_bytes)
        if acquisition.bind_model(model):
            self._stats["host_acquisition_models_bound"] = self._stats.get(
                "host_acquisition_models_bound", 0
            ) + 1
        return True

    def _prepare_admission_acquisition(
        self, pending: PendingHostLoad, batch: Any
    ) -> bool:
        """Build the physical acquisition group only when its model is usable."""

        if (
            self._tenant_isolation_enabled
            or self._execution_config.host_execution_mode
            is HostExecutionMode.DEVICE_BULK
        ):
            return False
        already_prepared = getattr(pending, "acquisition", None) is not None
        ready = self._prepare_host_acquisition_owner(pending, batch)
        if ready and not already_prepared:
            self._stats["admission_acquisition_groups_prepared"] = (
                self._stats.get("admission_acquisition_groups_prepared", 0) + 1
            )
        return ready

    def _submit_host_acquisition(self, pending: PendingHostLoad) -> int:
        """Fill the lease's Host link queue and publish exact layer fences."""

        acquisition = getattr(pending, "acquisition", None)
        if acquisition is None:
            raise RuntimeError("HiCache lease has no prepared acquisition owner")
        submission = acquisition.submit_available(
            publish_range=lambda begin, end: self._host_transport.prepare(
                pending,
                first_local_layer=begin,
                last_local_layer=end,
            ),
            published_layers=pending.prefetched_layers,
        )
        if submission.job_count:
            self._stats["host_acquisition_submission_calls"] = self._stats.get(
                "host_acquisition_submission_calls", 0
            ) + len(submission.ranges)
            self._stats["host_acquisition_jobs_submitted"] = self._stats.get(
                "host_acquisition_jobs_submitted", 0
            ) + submission.job_count
        return submission.job_count

    def _start_admission_acquisition(
        self, pending: PendingHostLoad, batch: Any
    ) -> None:
        """Start the finite EDF queue after admission has bounded its delay."""

        if (
            pending.transfer_plan is None
            or pending.prefetched_layers
            or pending.acquisition is None
            or pending.acquisition.started
        ):
            raise RuntimeError(
                "HiCache admission acquisition was not prepared exactly once"
            )
        if (
            self._admission_deadline_model(pending, batch) is None
            or pending.acquisition.model is None
        ):
            raise RuntimeError("HiCache admission acquisition lost its calibration")
        submitted = self._submit_host_acquisition(pending)
        if submitted != int(pending.controller.layer_num):
            raise RuntimeError("HiCache admission did not fill its finite link queue")
        self._stats["admission_acquisition_groups_started"] = (
            self._stats.get("admission_acquisition_groups_started", 0) + 1
        )

    def _account_tier_selection(self, pending: PendingHostLoad) -> None:
        """Count unique logical tier demand once per ownership lease."""

        if pending.selection_accounted:
            return
        row_count = int(pending.device_indices.numel())
        if row_count <= 0:
            raise RuntimeError("SGLang acquisition lease contains no selected rows")
        controller = pending.controller
        host_keys = tuple(controller.mem_pool_host.k_data_refs)
        host_values = tuple(controller.mem_pool_host.v_data_refs)
        layer_count = int(controller.layer_num)
        if (
            len(host_keys) != layer_count
            or len(host_values) != layer_count
            or not host_keys
        ):
            raise RuntimeError("SGLang acquisition lease has incomplete layer geometry")
        row_bytes_by_layer = tuple(
            (
                int(key[0].numel()) * key.element_size(),
                int(value[0].numel()) * value.element_size(),
            )
            for key, value in zip(host_keys, host_values, strict=True)
        )
        layer_bytes = tuple(
            row_count * (key_bytes + value_bytes)
            for key_bytes, value_bytes in row_bytes_by_layer
        )
        if pending.layer_bytes and pending.layer_bytes != layer_bytes:
            raise RuntimeError("SGLang acquisition byte geometry changed after capture")
        if (
            pending.row_bytes_by_layer
            and pending.row_bytes_by_layer != row_bytes_by_layer
        ):
            raise RuntimeError("SGLang acquisition row geometry changed after capture")
        pending.layer_bytes = layer_bytes
        pending.row_bytes_by_layer = row_bytes_by_layer
        pending.selection_accounted = True
        self._stats["tier_selected_leases"] += 1
        self._stats["tier_selected_rows"] += row_count
        self._stats["tier_selected_bytes"] += sum(layer_bytes)
        # SGLang HiCache load-back is an exact-dense source range. Sparse
        # demand providers may later publish a larger candidate set, but this
        # framework path neither approximates nor drops any candidate row.
        self._stats["tier_candidate_bytes"] += sum(layer_bytes)

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
        self._collect_layer_service_profiles()
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
        self._collect_layer_service_profiles()
        self._collect_barrier_profiles(already_synchronized=True)
        pending = {
            "mover": self._host_movers.pending_profile_count,
            "transfer": len(self._transfer_profiles),
            "operator": len(self._operator_profiles),
            "layer_service": len(self._layer_service_profiles),
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
        self._active_batch = None
        self._current_engine_batch = None
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
        return wrapper_id in self._stock_wrapper_for_typed

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

    def _current_forward_wrappers(self) -> tuple[Any, ...]:
        """Return the exact wrapper set represented by ForwardBatch metadata."""

        metadata = self.forward_metadata
        if hasattr(metadata, "decode_wrappers"):
            wrappers = tuple(metadata.decode_wrappers)
        else:
            if metadata.use_ragged:
                raise RuntimeError("NVMe acquisition requires paged prefill")
            wrappers = tuple(metadata.prefill_wrappers)
        if not wrappers:
            raise RuntimeError("FlashInfer forward metadata contains no wrappers")
        return wrappers

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
        batch = self._active_batch
        if batch is None:
            raise RuntimeError("typed FlashInfer adoption has no validated batch")
        source_to_target = {
            id(source): id(target)
            for target, source in zip(targets, sources, strict=True)
        }
        batch.adopt_wrapper_identity(source_to_target)
        self.forward_metadata = replace(stock_metadata, **{field: list(targets)})
        self._stock_wrapper_for_typed.clear()
        self._stock_wrapper_for_typed.update(
            {
                id(target): source
                for target, source in zip(targets, sources, strict=True)
            }
        )
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
        stock = self._stock_wrapper_for_typed.get(id(typed_wrapper))
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
        if self._active_batch is None:
            raise RuntimeError("cannot validate execution without active batch")
        batch = self._active_batch
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
            engine_batch = self._current_engine_batch
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

    def _record_execution_layer(self, layer: Any, *, final_layer: bool) -> None:
        """Commit the semantic work boundary after native attention returns."""
        if self._active_batch is None:
            raise RuntimeError("attention returned without a typed work topology")
        local_layer = int(layer.layer_id) - self._model_start_layer
        verifier = self._active_batch.verification_session
        if verifier is not None:
            self._stats.update(verifier.record_layer_completion(local_layer))
        # NVMe slots need a predecessor event after every layer; host-indexed
        # objects need the final consumer edge.  The physical-plan owner
        # publishes the exact token required by its selected resource.
        if self._tier_service.is_host_staged:
            self._require_materializer().record_host_consumer(
                torch.cuda.current_stream(), final_layer=final_layer
            )

    def _begin_forward(self) -> None:
        """Retire completed Python state and reject an unfinished predecessor."""

        batch = self._active_batch
        if batch is None:
            self._current_engine_batch = None
            return
        if batch.stream_ordered_epoch is not None:
            raise RuntimeError(
                "the preceding typed forward did not retire its stream-ordered "
                "work window"
            )
        pending = batch.pending_host_load
        if pending is not None and self._hicache.get(pending.consumer_index) is pending:
            raise RuntimeError(
                "the preceding forward did not retire its HiCache acquisition lease"
            )
        self._active_batch = None
        self._current_engine_batch = None
        self._stock_wrapper_for_typed.clear()

    def _finish_forward(self, batch: _ActiveBatch) -> None:
        """Release forward-scoped Python ownership after its final consumer."""

        if self._cuda_graph_mode:
            return
        if self._active_batch is not batch:
            raise RuntimeError("SGLang forward completion lost its active batch")
        if batch.stream_ordered_epoch is not None:
            raise RuntimeError("SGLang forward completed with an unretired work epoch")
        pending = batch.pending_host_load
        if pending is not None and self._hicache.get(pending.consumer_index) is pending:
            raise RuntimeError("SGLang forward completed with a live acquisition lease")
        self._active_batch = None
        self._current_engine_batch = None
        self._stock_wrapper_for_typed.clear()
        self._stats["forward_lifecycle_completions"] += 1

    def abort_active_forward(self, pending: PendingHostLoad | None = None) -> bool:
        """Quiesce and retire an abnormal forward without leaking its lease.

        This is an exceptional control boundary, so a device synchronization
        is intentional: proactive copies may have been issued on auxiliary
        streams, and returning SGLang's host-row acknowledgement before all of
        them finish would permit source reuse under DMA.
        """

        batch = self._active_batch
        target = batch.pending_host_load if batch is not None else pending
        if batch is None and target is None:
            self._current_engine_batch = None
            return False
        torch.cuda.synchronize()
        if batch is not None and batch.nvme_acquisition is not None:
            pipeline = self._nvme_pipeline
            if pipeline is not None:
                pipeline.abort(batch.nvme_acquisition)
        retired = False
        if target is not None:
            retired = self._hicache.retire(target, stream=torch.cuda.current_stream())
        self._active_batch = None
        self._current_engine_batch = None
        self._stock_wrapper_for_typed.clear()
        self._stats["forward_lifecycle_aborts"] += 1
        return batch is not None or retired

    def _finalize_stream_ordered_batch(
        self, batch: _ActiveBatch, stream: torch.cuda.Stream
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
            self._operator_profiles.append((*profile, "stream_retirement", layers))
        self._stats["stream_ordered_retirement_launches"] += 1
        self._stats["stream_ordered_retirement_batches"] += 1
        batch.stream_ordered_epoch = None
        batch.stream_ordered_progress_rounds = 0
        batch.stream_ordered_layers = 0

    def _record_external_layer_execution(
        self,
        batch: _ActiveBatch,
        local_layer: int,
        *,
        native_dispatch: bool,
        progressive_consumer: bool,
        final_layer: bool,
    ) -> None:
        """Record exact external dispatch and progressive-consumer coverage.

        Native-vs-framework is a numerical dispatch choice, not a readiness
        observation. Progressive coverage is recorded only for an execution
        epoch that releases work as acquisition groups become runnable. These
        facts remain separate so dispatch form cannot masquerade as partial
        arrival evidence.
        """

        if batch.external_dispatch_recorded:
            raise RuntimeError("external dispatch received a layer after completion")
        if final_layer:
            self._finalize_stream_ordered_batch(batch, torch.cuda.current_stream())
        if local_layer != batch.external_last_local_layer + 1:
            raise RuntimeError("external dispatch layers are not contiguous")
        batch.external_last_local_layer = local_layer
        if native_dispatch:
            if batch.framework_dispatch_seen:
                batch.native_dispatch_nonprefix_seen = True
            batch.native_dispatch_external_layers += 1
        else:
            batch.framework_dispatch_seen = True
            batch.framework_dispatch_external_layers += 1
        if progressive_consumer:
            if not native_dispatch:
                raise RuntimeError(
                    "framework external dispatch cannot claim progressive work"
                )
            batch.progressive_consumer_external_layers += 1
        if not final_layer:
            return
        observed_layers = (
            batch.native_dispatch_external_layers
            + batch.framework_dispatch_external_layers
        )
        if observed_layers != self._model_layer_count:
            raise RuntimeError(
                "external dispatch did not account for every model layer"
            )
        native_layers = batch.native_dispatch_external_layers
        if batch.native_dispatch_nonprefix_seen:
            self._stats["native_dispatch_nonprefix_batches"] += 1
            key = f"native_dispatch_nonprefix_layers_{native_layers}_batches"
        else:
            self._stats["native_dispatch_prefix_observations"] += 1
            key = f"native_dispatch_prefix_layers_{native_layers}_batches"
        self._stats[key] = self._stats.get(key, 0) + 1
        progressive_layers = batch.progressive_consumer_external_layers
        self._stats["progressive_consumer_batch_observations"] += 1
        self._stats["progressive_consumer_layers"] += progressive_layers
        self._stats["progressive_consumer_batches"] = self._stats.get(
            "progressive_consumer_batches", 0
        ) + int(progressive_layers > 0)
        progressive_key = f"progressive_consumer_layers_{progressive_layers}_batches"
        self._stats[progressive_key] = self._stats.get(progressive_key, 0) + 1
        batch.external_dispatch_recorded = True

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
        batch = self._request_adapter.bind_forward(
            forward_batch,
            allow_capture_ids=allow_capture_ids,
            stream=torch.cuda.current_stream(),
            epoch=self._execution_epoch,
            granularity=self._execution_config.protocol.granularity,
        )
        self._execution_epoch += 1
        self._current_engine_batch = batch
        self._stats["engine_batch_epoch"] = batch.epoch
        self._stats["engine_batch_size"] = len(batch.bindings)
        bindings = batch.bindings
        self._stats["request_rebindings"] += self._request_adapter.last_publish_count
        self._stats["request_metadata_updates"] = (
            self._stats.get("request_metadata_updates", 0)
            + self._request_adapter.last_metadata_publish_count
        )
        return bindings

    def init_forward_metadata_out_graph(
        self, forward_batch: Any, in_capture: bool = False
    ) -> None:
        self._begin_forward()
        self._cuda_graph_mode = True
        counter = getattr(self.token_to_kv_pool, "layer_transfer_counter", None)
        consumer_index = -1 if counter is None else int(counter.consumer_index)
        pending = self._hicache.get(consumer_index)
        if pending is not None:
            self._account_tier_selection(pending)
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
            self._active_batch = _ActiveBatch(
                bindings=bindings,
                semantic_plans={},
                pending_host_load=None,
            )
            self._stats["resident_reference_batches"] += 1
            self._stats["stock_resident_batches"] += 1
        else:
            if self._tenant_isolation_enabled:
                raise RuntimeError(
                    "external CUDA-graph prefetch cannot bypass finite tenant budgets"
                )
            if pending.acquisition is None:
                self._host_transport.prepare_missing(pending)
            else:
                self._submit_host_acquisition(pending)
            final_layer = _require_exact_prefetch_layers(
                pending.prefetched_layers,
                self._model_layer_count,
                consumer="CUDA graph replay",
            )
            self._active_batch = _ActiveBatch(
                bindings=bindings,
                semantic_plans={},
                pending_host_load=pending,
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
        self._active_batch = _ActiveBatch(
            bindings=bindings,
            semantic_plans={},
            pending_host_load=pending,
        )
        if count_batch:
            self._stats["batches"] += 1
            self._stats["hicache_external_batches"] += 1
        self._stats["stock_prefetched_external_batches"] += 1
        self._stats["stock_prefetch_metadata_fastpath_batches"] = (
            self._stats.get("stock_prefetch_metadata_fastpath_batches", 0) + 1
        )

    def init_forward_metadata(self, forward_batch: Any) -> None:
        self._begin_forward()
        self._cuda_graph_mode = False
        self._stock_forward = False
        self._stock_wrapper_for_typed.clear()
        if forward_batch.forward_mode.is_mixed():
            self._stats["mixed_forward_batches"] += 1
            self._stats["mixed_forward_requests"] += len(
                tuple(getattr(forward_batch, "rids", ()) or ())
            )
        counter = getattr(self.token_to_kv_pool, "layer_transfer_counter", None)
        consumer_index = -1 if counter is None else int(counter.consumer_index)
        pending = self._hicache.get(consumer_index)
        if pending is not None:
            self._account_tier_selection(pending)
        measured_host_selection = (
            pending is not None and self._tier_service.is_host_staged
        )
        mixed_host_batch = (
            measured_host_selection and forward_batch.forward_mode.is_mixed()
        )
        if pending is None:
            self._stock_forward = True
            self._activate_wrapper_set(self._require_kernels().stock_wrappers)
        elif measured_host_selection:
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
            super().init_forward_metadata(forward_batch)
            if pending is None:
                # A resident stock forward has no acquisition identity to
                # publish and no native work to attribute.  Account its known
                # all-layer dispatch once here; the per-layer methods can then
                # be a thin call into SGLang's stock backend.
                stock_layers = self._model_layer_count
                self._stats["stock_attention_launches"] += stock_layers
                self._stats["stock_resident_attention_launches"] += stock_layers
                if forward_batch.forward_mode.is_decode_or_idle():
                    self._stats["decode_launches"] += stock_layers
                else:
                    self._stats["prefill_launches"] += stock_layers
                self._active_batch = _ActiveBatch(
                    bindings=(),
                    semantic_plans={},
                    pending_host_load=None,
                )
                self._stats["batches"] += 1
                self._stats["resident_reference_batches"] += 1
                self._stats["stock_resident_batches"] += 1
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
                    (
                        pending.acquisition is None
                        or pending.acquisition.model is None
                    )
                    and selected.uses_dependency_protocol
                    and not self._tenant_isolation_enabled
                    and self._execution_config.host_execution_mode
                    is not HostExecutionMode.DEVICE_BULK
                    and self._prepare_host_acquisition_owner(pending, forward_batch)
                ):
                    self._stats["metadata_acquisition_groups_prepared"] = (
                        self._stats.get("metadata_acquisition_groups_prepared", 0) + 1
                    )
                if (
                    pending.acquisition is not None
                    and not pending.acquisition.fully_published
                ):
                    self._submit_host_acquisition(pending)
                prefetch_fully_published = len(pending.prefetched_layers) == (
                    self._model_layer_count
                )
                progress = self._hicache.progress(consumer_index)
                prefetch_fully_ready = progress is not None and progress.complete
                ready_stock_fastpath = (
                    prefetch_fully_ready
                    and self._execution_config.host_execution_mode
                    is HostExecutionMode.AUTO
                    and selected.selection_reason != "calibration_probe"
                )
                if ready_stock_fastpath or not selected.uses_dependency_protocol:
                    self._host_transport.prepare_missing(pending)
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
                            self._stats.get(
                                "host_bound_after_full_ready_batches", 0
                            )
                            + 1
                        )
                    return

                if prefetch_fully_published:
                    self._stats["host_typed_after_full_publication_batches"] = (
                        self._stats.get(
                            "host_typed_after_full_publication_batches", 0
                        )
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
                if self._active_batch is None:  # pragma: no cover - set above
                    raise RuntimeError("incremental host batch lost its metadata")
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
                        batch=self._active_batch,
                        wrappers=typed_wrappers,
                        layer_id=partition_layer_id,
                        kv_cache=(
                            self.token_to_kv_pool._get_key_buffer(partition_layer_id),
                            self.token_to_kv_pool._get_value_buffer(partition_layer_id),
                        ),
                    )
                self._active_batch.incremental_metadata_setup_ns = (
                    time.perf_counter_ns() - incremental_setup_started
                )
                for counter, elapsed in (
                    ("incremental_wrapper_select_cpu_ns", wrapper_select_ns),
                    ("incremental_metadata_adoption_cpu_ns", adoption_ns),
                    (
                        "incremental_metadata_setup_cpu_ns",
                        self._active_batch.incremental_metadata_setup_ns,
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
            self._init_external_metadata(forward_batch, pending, bindings=bindings)
            if self._tier_service.is_nvme:
                batch = self._active_batch
                if batch is None:
                    raise RuntimeError("NVMe metadata produced no active batch")
                self._require_attention_executor().prepare_nvme_batch(
                    batch=batch,
                    wrappers=self._current_forward_wrappers(),
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
        self._account_tier_selection(pending)
        if self._opportunity_trace is not None and count_batch:
            self._active_opportunity_batch = self._opportunity_batch
            self._opportunity_batch += 1
        if bindings is None:
            bindings = self._bind_forward_requests(
                forward_batch, allow_capture_ids=False
            )
        engine_batch = self._current_engine_batch
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
        self._active_batch = planned.batch
        return planned.host_execution

    def _collect_layer_service_profiles(self) -> None:
        """Retire completed attention-arrival gaps without synchronizing."""

        pending: list[_LayerServiceProfile] = []
        for profile in self._layer_service_profiles:
            if not profile.finish.query():
                pending.append(profile)
                continue
            elapsed_ns = max(
                1, round(profile.start.elapsed_time(profile.finish) * 1_000_000.0)
            )
            curve = self._layer_service_curves.get(
                profile.key,
                AcquisitionServiceCurve(
                    minimum_samples=self._layer_service_minimum_samples,
                    maximum_samples=self._layer_service_maximum_samples,
                ),
            ).with_observation(elapsed_ns)
            self._layer_service_curves[profile.key] = curve
            self._stats["layer_service_profiled_intervals"] += 1
        self._layer_service_profiles = pending
        calibrated = tuple(
            curve for curve in self._layer_service_curves.values() if curve.calibrated
        )
        self._stats["layer_service_calibrated_shapes"] = len(calibrated)

    def _record_layer_arrival(
        self, phase: str, query: torch.Tensor, layer: Any
    ) -> None:
        """Sample bounded per-layer compute slack for one exact forward shape."""

        batch = self._active_batch
        if (
            batch is None
            or batch.pending_host_load is None
            or not self._tier_service.is_host_staged
        ):
            return
        query_rows = int(query.shape[0])
        key = (phase, query_rows, len(batch.bindings))
        if min(query_rows, len(batch.bindings)) <= 0:
            raise RuntimeError("layer service calibration has an empty forward")
        if batch.layer_service_key is not None and batch.layer_service_key != key:
            raise RuntimeError("attention shape changed within one model forward")
        batch.layer_service_key = key
        curve = self._layer_service_curves.get(
            key,
            AcquisitionServiceCurve(
                minimum_samples=self._layer_service_minimum_samples,
                maximum_samples=self._layer_service_maximum_samples,
            ),
        )
        inflight = sum(profile.key == key for profile in self._layer_service_profiles)
        if len(curve.samples_ns) + inflight >= curve.maximum_samples:
            batch.layer_arrival_event = None
            return

        local_layer = int(layer.layer_id) - self._model_start_layer
        if not 0 <= local_layer < self._model_layer_count:
            raise RuntimeError("attention layer is outside the local model range")
        arrival = torch.cuda.Event(enable_timing=True)
        arrival.record(torch.cuda.current_stream())
        previous = batch.layer_arrival_event
        if previous is not None:
            if batch.layer_arrival_local_layer + 1 != local_layer:
                raise RuntimeError("attention layers did not arrive in model order")
            self._layer_service_profiles.append(
                _LayerServiceProfile(previous, arrival, key)
            )
        batch.layer_arrival_event = arrival
        batch.layer_arrival_local_layer = local_layer

    def _host_mover_lease_plan(
        self,
        pending: PendingHostLoad,
        row_bytes_by_layer: tuple[tuple[int, int], ...],
        transfer_count: int,
        *,
        layer_service_key: tuple[str, int, int] | None = None,
        layer_curve: AcquisitionServiceCurve | None = None,
    ) -> HostMoverLeasePlan:
        return self._host_movers.plan(
            pending,
            row_bytes_by_layer,
            transfer_count,
            layer_service_key=layer_service_key,
            layer_curve=layer_curve,
            collect_layer_profiles=self._collect_layer_service_profiles,
        )

    def _host_transfer_lease_plan(
        self,
        pending: PendingHostLoad,
        *,
        layer_service_key: tuple[str, int, int] | None = None,
        layer_curve: AcquisitionServiceCurve | None = None,
    ) -> HostTransferLeasePlan:
        """Build immutable K/V descriptors once, before any frontier slicing."""

        cached = pending.transfer_plan
        layer_count = int(pending.controller.layer_num)
        transfer_count = int(pending.device_indices.numel())
        if cached is not None:
            if (
                cached.mover.row_count != transfer_count
                or len(cached.layers) != layer_count
            ):
                raise RuntimeError("HiCache transfer geometry changed during a lease")
            self._stats["host_transfer_plan_reuses"] = (
                self._stats.get("host_transfer_plan_reuses", 0) + 1
            )
            return cached
        if transfer_count <= 0 or transfer_count != int(pending.host_indices.numel()):
            raise RuntimeError("HiCache host pipeline has no promoted pages")
        if not pending.row_bytes_by_layer:
            self._account_tier_selection(pending)
        if len(pending.row_bytes_by_layer) != layer_count:
            raise RuntimeError("HiCache acquisition has incomplete row geometry")
        mover = self._host_mover_lease_plan(
            pending,
            pending.row_bytes_by_layer,
            transfer_count,
            layer_service_key=layer_service_key,
            layer_curve=layer_curve,
        )
        if pending.lease_id <= 0 or pending.lease_id > 0xFFFFFFFF:
            raise RuntimeError("HiCache lease version exceeds the runtime ABI")
        result = build_host_transfer_lease_plan(
            pending.controller,
            mover,
            pending.row_bytes_by_layer,
            object_id_bases=tuple(
                _pipeline_object_id(
                    pending.consumer_index,
                    layer_count,
                    local_layer,
                    self._sm_acquisition_waves,
                )
                for local_layer in range(layer_count)
            ),
            object_version=pending.lease_id,
            sm_acquisition_waves=self._sm_acquisition_waves,
        )
        layer_bytes = tuple(
            key_bytes + value_bytes for key_bytes, value_bytes in result.layer_geometry
        )
        if pending.layer_bytes != layer_bytes:
            raise RuntimeError(
                "HiCache transfer plan changed its captured byte geometry"
            )
        pending.transfer_plan = result
        self._stats["host_transfer_plan_builds"] = (
            self._stats.get("host_transfer_plan_builds", 0) + 1
        )
        return result

    def _advance_deadline_frontier(
        self,
        pending: PendingHostLoad,
        completed_local_layer: int,
        *,
        fragment: DeadlineFragment | None = None,
    ) -> None:
        """Publish the maximal calibrated EDF-feasible layer prefix.

        Full-layer acquisition and numerical work are separate ownership
        objects.  Layers whose cumulative mover service meets every attention
        deadline are published in deadline order and later use the preacquired
        numerical kernel.  The first modeled miss remains unpublished so the
        compiler-verified partial consumer can acquire it at work-unit
        granularity instead of waiting for an already-committed whole layer.
        """

        batch = self._active_batch
        if batch is None or batch.pending_host_load is not pending:
            raise RuntimeError("deadline frontier lost its active HiCache lease")
        acquisition = getattr(pending, "acquisition", None)
        if acquisition is not None:
            self._retire_host_acquisition_layer(pending, completed_local_layer)
            # The current Host backend publishes its complete finite queue at
            # lease capture.  Do not rescan the queue and call the transport
            # submitter after every transformer layer merely to rediscover that
            # no work remains.  A future bounded-inflight backend retains the
            # refill path through the same lifecycle predicate.
            if not acquisition.fully_published:
                submitted = self._submit_host_acquisition(pending)
                self._stats["host_acquisition_refill_jobs"] = self._stats.get(
                    "host_acquisition_refill_jobs", 0
                ) + submitted
            return
        if not self._frontier_enabled:
            return
        layer_count = int(pending.controller.layer_num)
        ready_prefix = completed_local_layer + 1
        if not 0 < ready_prefix <= layer_count:
            raise RuntimeError("deadline frontier received an invalid layer prefix")
        if ready_prefix == layer_count:
            return

        model = batch.deadline_model
        frontier_plan_built = False
        if not batch.deadline_model_initialized:
            self._host_movers.collect_profiles()
            self._collect_layer_service_profiles()
            if pending.mover_plan is None:
                # Descriptor preparation is issued only after current attention
                # is queued, so it is outside the next layer's first-dispatch
                # dependency. Auto may choose one bounded calibration probe
                # here; admission itself never runs a probe.
                self._host_transfer_lease_plan(pending)
            curve = (
                None
                if batch.layer_service_key is None
                else self._layer_service_curves.get(batch.layer_service_key)
            )
            model = (
                None
                if curve is None
                else self._deadline_model_for_curve(pending, curve)
            )
            if model is not None:
                batch.deadline_model = model
                batch.deadline_model_initialized = True
                self._stats["deadline_frontier_model_builds"] = (
                    self._stats.get("deadline_frontier_model_builds", 0) + 1
                )
        else:
            self._stats["deadline_frontier_model_reuses"] = (
                self._stats.get("deadline_frontier_model_reuses", 0) + 1
            )
        if model is None:
            self._stats["deadline_frontier_uncalibrated"] = (
                self._stats.get("deadline_frontier_uncalibrated", 0) + 1
            )
            calibration_probe = not self._host_movers.lease_calibrated(pending)
            if calibration_probe and ready_prefix not in pending.prefetched_layers:
                probe_end = _calibration_probe_end(
                    ready_prefix, layer_count, self._frontier_layers_per_wave
                )
                self._host_transport.prepare(
                    pending,
                    first_local_layer=ready_prefix,
                    last_local_layer=probe_end,
                )
                self._stats["deadline_frontier_calibration_layers"] = (
                    self._stats.get("deadline_frontier_calibration_layers", 0)
                    + probe_end
                    - ready_prefix
                )
            elif fragment is not None and ready_prefix not in pending.prefetched_layers:
                self._enqueue_fragment_lookahead(
                    fragment.wrapper,
                    self._model_start_layer + completed_local_layer,
                    fragment.object_count,
                    fragment.host_execution,
                    fragment.stream,
                )
            return

        if batch.deadline_frontier is None:
            batch.deadline_frontier = model.compile_after_attention_frontier()
            frontier_plan_built = True
            self._stats["deadline_frontier_plan_builds"] = (
                self._stats.get("deadline_frontier_plan_builds", 0) + 1
            )

        frontier = batch.deadline_frontier
        if frontier is None or frontier.layer_count != layer_count:
            raise RuntimeError("deadline frontier has no compiled service plan")
        feasible_end = frontier.feasible_end_after_attention(completed_local_layer)
        self._stats["deadline_frontier_plans"] = (
            self._stats.get("deadline_frontier_plans", 0) + 1
        )
        self._stats["deadline_frontier_plan_reuses"] = self._stats.get(
            "deadline_frontier_plan_reuses", 0
        ) + int(not frontier_plan_built)
        first_missed_layer = None if feasible_end == layer_count else feasible_end
        if first_missed_layer is not None:
            self._stats["deadline_frontier_first_missed_layer_sum"] = (
                self._stats.get("deadline_frontier_first_missed_layer_sum", 0)
                + first_missed_layer
            )
        publish_begin = ready_prefix
        while (
            publish_begin < layer_count and publish_begin in pending.prefetched_layers
        ):
            publish_begin += 1
        if publish_begin < feasible_end:
            self._host_transport.prepare(
                pending,
                first_local_layer=publish_begin,
                last_local_layer=feasible_end,
            )
            self._stats["deadline_frontier_published_layers"] = (
                self._stats.get("deadline_frontier_published_layers", 0)
                + feasible_end
                - publish_begin
            )
        fragment_enqueued = False
        if (
            feasible_end == ready_prefix
            and ready_prefix not in pending.prefetched_layers
            and fragment is not None
        ):
            fragment_enqueued = self._enqueue_fragment_lookahead(
                fragment.wrapper,
                self._model_start_layer + completed_local_layer,
                fragment.object_count,
                fragment.host_execution,
                fragment.stream,
            )
            if fragment_enqueued:
                self._stats["deadline_frontier_fragment_layers"] = (
                    self._stats.get("deadline_frontier_fragment_layers", 0) + 1
                )
        if publish_begin >= feasible_end and not fragment_enqueued:
            self._stats["deadline_frontier_noop_calls"] = (
                self._stats.get("deadline_frontier_noop_calls", 0) + 1
            )

    def _retire_host_acquisition_layer(
        self,
        pending: PendingHostLoad,
        local_layer: int,
    ) -> None:
        """Retire transport ownership after any numerical consumer is ordered."""

        acquisition = getattr(pending, "acquisition", None)
        if acquisition is None:
            return
        acquisition.retire(local_layer)
        self._stats["host_acquisition_layers_consumed"] = self._stats.get(
            "host_acquisition_layers_consumed", 0
        ) + 1

    def _require_host_execution_plan(self) -> HostExecutionPlan:
        batch = self._active_batch
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
        batch = self._active_batch
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
            return
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
        batch = self._active_batch
        if batch is None:
            raise RuntimeError("NTA attention ran without request metadata")

        pending = batch.pending_host_load
        dispatch = select_attention_dispatch(
            pending=pending,
            host_execution=batch.host_execution,
            tier_is_nvme=self._tier_service.is_nvme,
            layer_id=int(layer.layer_id),
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
            alias_available=id(wrapper) in self._stock_wrapper_for_typed,
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
        gpu_profile = None
        if self._profile_gpu:
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

        if pending is not None:
            frontier_started = time.perf_counter_ns() if self._profile_cpu else 0
            self._advance_deadline_frontier(
                pending,
                dispatch.local_layer,
                fragment=outcome.deadline_fragment,
            )
            if self._profile_cpu:
                self._stats["deadline_frontier_cpu_ns"] = self._stats.get(
                    "deadline_frontier_cpu_ns", 0
                ) + (time.perf_counter_ns() - frontier_started)

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
            self._operator_profiles.append((*gpu_profile, dispatch.kind.value, 1))

        if (
            pending is not None
            and self._tier_service.is_host_staged
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

        self._record_execution_layer(layer, final_layer=final_layer)
        if pending is not None:
            self._stats["external_launches"] += 1
            self._stats["native_external_attention_launches"] += 1
            self._record_external_layer_execution(
                batch,
                dispatch.local_layer,
                native_dispatch=True,
                progressive_consumer=outcome.progressive_consumer,
                final_layer=final_layer,
            )
            self._hicache.complete_layer(pending, dispatch.local_layer)
        if final_layer:
            self._commit_incremental_setup_observation(batch)
            self._publish_stats()
            self._finish_forward(batch)
        return output.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _commit_incremental_setup_observation(self, batch: _ActiveBatch) -> None:
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
        self, pending: PendingHostLoad, layer: Any
    ) -> int:
        """Join the producer event before stock attention consumes a page."""

        local_layer = int(layer.layer_id) - int(
            getattr(pending.controller.mem_pool_device, "start_layer", 0)
        )
        prefetched = pending.prefetched_layers.get(local_layer)
        if prefetched is None:
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
                    prefetched.ready_event,
                    int(layer.layer_id),
                    "attention_layer",
                )
            )
        stream.wait_event(prefetched.ready_event)
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
        batch = self._active_batch
        if batch is None or batch.pending_host_load is None:
            raise RuntimeError("preloaded stock layer has no external lease")
        pending = batch.pending_host_load
        local_layer = self._wait_for_stock_external_layer(pending, layer)
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
            self._operator_profiles.append((*profile, "preloaded_stock", 1))
        self._stats["stock_attention_launches"] += 1
        self._stats["stock_prefetched_external_attention_launches"] += 1
        self._stats["lookahead_bound_launches"] += 1
        self._stats["external_launches"] += 1
        if self._verification.transfer:
            self._require_attention_verifier().verify_layer_transfer(
                batch,
                int(layer.layer_id),
                kv_cache,
            )
        final_layer = local_layer + 1 == self._model_layer_count
        self._record_external_layer_execution(
            batch,
            local_layer,
            native_dispatch=False,
            progressive_consumer=False,
            final_layer=final_layer,
        )
        self._advance_deadline_frontier(pending, local_layer)
        self._require_materializer().record_host_consumer(
            stream, final_layer=final_layer
        )
        self._hicache.complete_layer(pending, local_layer)
        if final_layer:
            self._commit_incremental_setup_observation(batch)
            self._publish_stats()
            self._finish_forward(batch)
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
        batch = self._active_batch
        if batch is None:
            raise RuntimeError(
                "NTA decode ran without transformed request metadata; stock "
                "dispatch is disabled"
            )
        if batch.pending_host_load is not None:
            self._record_layer_arrival("decode", q, layer)
        if self._stock_forward:
            pending = batch.pending_host_load
            if pending is None:
                output = FlashInferAttnBackend.forward_decode(
                    self,
                    q,
                    k,
                    v,
                    layer,
                    forward_batch,
                    save_kv_cache=save_kv_cache,
                )
                if (
                    int(layer.layer_id) - self._model_start_layer + 1
                    == self._model_layer_count
                ):
                    self._finish_forward(batch)
                return output
            self._stats["stock_attention_launches"] += 1
            self._stats["stock_prefetched_external_attention_launches"] += 1
            self._stats["external_launches"] += 1
            local_layer = self._wait_for_stock_external_layer(pending, layer)
            self._stats["decode_launches"] += 1
            output = FlashInferAttnBackend.forward_decode(
                self,
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
            )
            final_layer = local_layer + 1 == self._model_layer_count
            self._record_external_layer_execution(
                batch,
                local_layer,
                native_dispatch=False,
                progressive_consumer=False,
                final_layer=final_layer,
            )
            self._require_materializer().record_host_consumer(
                torch.cuda.current_stream(), final_layer=final_layer
            )
            self._retire_host_acquisition_layer(pending, local_layer)
            self._hicache.complete_layer(pending, local_layer)
            if final_layer:
                self._finish_forward(batch)
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
        batch = self._active_batch
        if batch is None:
            raise RuntimeError(
                "NTA prefill ran without transformed request metadata; stock "
                "dispatch is disabled"
            )
        if batch.pending_host_load is not None:
            self._record_layer_arrival("extend", q, layer)
        if self._stock_forward:
            pending = batch.pending_host_load
            if pending is None:
                output = FlashInferAttnBackend.forward_extend(
                    self,
                    q,
                    k,
                    v,
                    layer,
                    forward_batch,
                    save_kv_cache=save_kv_cache,
                )
                if (
                    int(layer.layer_id) - self._model_start_layer + 1
                    == self._model_layer_count
                ):
                    self._finish_forward(batch)
                return output
            self._stats["stock_attention_launches"] += 1
            self._stats["stock_prefetched_external_attention_launches"] += 1
            self._stats["external_launches"] += 1
            local_layer = self._wait_for_stock_external_layer(pending, layer)
            self._stats["prefill_launches"] += 1
            output = FlashInferAttnBackend.forward_extend(
                self,
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
            )
            final_layer = local_layer + 1 == self._model_layer_count
            self._record_external_layer_execution(
                batch,
                local_layer,
                native_dispatch=False,
                progressive_consumer=False,
                final_layer=final_layer,
            )
            self._require_materializer().record_host_consumer(
                torch.cuda.current_stream(), final_layer=final_layer
            )
            self._retire_host_acquisition_layer(pending, local_layer)
            self._hicache.complete_layer(pending, local_layer)
            if final_layer:
                self._finish_forward(batch)
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
        pending_operators: list[
            tuple[torch.cuda.Event, torch.cuda.Event, str, int]
        ] = []
        for start, finish, kind, covered_layers in self._operator_profiles:
            if not finish.query():
                pending_operators.append((start, finish, kind, covered_layers))
                continue
            milliseconds = start.elapsed_time(finish)
            prefix = f"profiled_{kind}_operator"
            self._stats[f"{prefix}_layers"] = (
                self._stats.get(f"{prefix}_layers", 0) + covered_layers
            )
            self._stats[f"{prefix}_launches"] = (
                self._stats.get(f"{prefix}_launches", 0) + 1
            )
            self._stats[f"{prefix}_gpu_ms"] = (
                self._stats.get(f"{prefix}_gpu_ms", 0.0) + milliseconds
            )
        self._operator_profiles[:] = pending_operators

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
        self._collect_layer_service_profiles()
        self._collect_barrier_profiles()
        report = dict(self._stats)
        report.update(self._tier_service.stats())
        report["layer_service_curves"] = [
            {
                "phase": key[0],
                "query_rows": key[1],
                "batch_size": key[2],
                "samples": len(curve.samples_ns),
                "conservative_interval_ns": curve.conservative_interval_ns,
            }
            for key, curve in sorted(self._layer_service_curves.items())
        ]
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
