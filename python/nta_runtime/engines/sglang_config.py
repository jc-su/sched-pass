"""Validated deployment configuration for the SGLang engine boundary.

Environment parsing is a setup-plane concern.  Keeping it out of the forward
adapter prevents lifecycle code from becoming an implicit configuration
registry and gives tests one immutable object to validate before CUDA work is
submitted.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import pathlib
from typing import Any

from nta_runtime.adapters.sglang import SglangExecutionConfig
from nta_runtime.execution_planner import HostCostModel, HostExecutionMode
from nta_runtime.execution_protocol import ProtocolKind
from nta_runtime.engines.sglang_planning import (
    boolean_environment,
    demand_overlap_policy,
    host_mover_environment,
    mover_stream_priority,
    minimum_saturating_pair_layers,
    nonnegative_environment,
    positive_environment,
)
from nta_runtime.engines.sglang_calibration_profile import (
    SglangCalibrationProfileConfig,
)
from nta_runtime.engines.sglang_transfer import (
    host_mover_service_model_from_environment,
)
from nta_runtime.indexed_transfer import IndexedMoverServiceModel
from nta_runtime.tenant import tenant_budget_specs, tenant_isolation_required


# One dependency-aware sample is deliberately consumed as an initialization
# observation. AUTO therefore needs one additional steady-state sample before
# it can make a measured execution-form decision. Keeping these values beside
# the parser prevents a one-probe deployment from staying uncalibrated forever.
AUTO_INCREMENTAL_INITIALIZATION_PROBES = 1
DEFAULT_AUTO_INCREMENTAL_CALIBRATION_PROBES = 2


def incremental_calibration_probe_count(
    *, execution: SglangExecutionConfig, host_staged: bool
) -> int:
    """Return a closed-loop AUTO calibration budget.

    Zero is an explicit opt-out. Any enabled budget must cover both the
    initialization observation and at least one steady-state observation, and
    is meaningful only for the late-bound host-staged AUTO path.
    """

    applicable = (
        host_staged
        and execution.protocol.kind is not ProtocolKind.CONVENTIONAL
        and execution.host_execution_mode is HostExecutionMode.AUTO
    )
    probes = nonnegative_environment(
        "NTA_EXECUTION_CALIBRATION_PROBES",
        DEFAULT_AUTO_INCREMENTAL_CALIBRATION_PROBES if applicable else 0,
    )
    if probes == 0:
        return 0
    if not applicable:
        raise RuntimeError(
            "incremental calibration probes require late-bound host-staged AUTO execution"
        )
    if probes <= AUTO_INCREMENTAL_INITIALIZATION_PROBES:
        raise RuntimeError(
            "NTA_EXECUTION_CALIBRATION_PROBES must be 0 or at least 2: "
            "one initialization and one steady-state observation are required"
        )
    return probes


@dataclass(frozen=True, slots=True)
class SglangBootstrapConfig:
    """Settings needed before the native tier/runtime owner can open."""

    request_capacity: int
    work_ticket_capacity: int
    max_dependencies_per_work_ticket: int
    object_capacity: int
    tenant_capacity: int
    tenant_specs: tuple[tuple[int, int], ...]
    tenant_isolation_enabled: bool
    allow_load_fallback: bool
    execution: SglangExecutionConfig

    @classmethod
    def from_environment(cls, request_capacity: int) -> "SglangBootstrapConfig":
        if request_capacity <= 0:
            raise ValueError("SGLang request capacity must be positive")
        work_ticket_capacity = positive_environment(
            "NTA_RUNTIME_MAX_WORK_TICKETS", max(4096, request_capacity * 8)
        )
        max_dependencies = positive_environment(
            "NTA_RUNTIME_MAX_DEPENDENCIES_PER_WORK_TICKET", 16
        )
        object_capacity = 2 * work_ticket_capacity
        tenant_capacity = positive_environment("NTA_TENANT_CAPACITY", request_capacity)
        tenant_specs = tenant_budget_specs()
        for tenant_id, _maximum_bytes in tenant_specs:
            if tenant_id >= tenant_capacity:
                raise RuntimeError(
                    f"tenant {tenant_id} exceeds NTA_TENANT_CAPACITY={tenant_capacity}"
                )
        try:
            execution = SglangExecutionConfig.from_environment()
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        return cls(
            request_capacity=request_capacity,
            work_ticket_capacity=work_ticket_capacity,
            max_dependencies_per_work_ticket=max_dependencies,
            object_capacity=object_capacity,
            tenant_capacity=tenant_capacity,
            tenant_specs=tenant_specs,
            tenant_isolation_enabled=tenant_isolation_required(tenant_specs),
            allow_load_fallback=boolean_environment(
                "NTA_EXECUTION_ALLOW_LOAD_FALLBACK", False
            ),
            execution=execution,
        )


@dataclass(frozen=True, slots=True)
class SglangModelPartition:
    global_layer_count: int
    first_layer: int
    end_layer: int

    @property
    def layer_count(self) -> int:
        return self.end_layer - self.first_layer

    @classmethod
    def from_runner(cls, model_runner: Any, token_pool: Any) -> "SglangModelPartition":
        layer_count = getattr(model_runner.model_config, "num_hidden_layers", None)
        if layer_count is None:
            layer_count = getattr(
                model_runner.model_config.hf_config, "num_hidden_layers"
            )
        global_layer_count = int(layer_count)
        first_layer = int(getattr(token_pool, "start_layer", 0))
        end_layer = int(getattr(token_pool, "end_layer", global_layer_count))
        if not 0 <= first_layer < end_layer <= global_layer_count:
            raise RuntimeError("SGLang KV pool exposes an invalid local layer range")
        return cls(global_layer_count, first_layer, end_layer)


@dataclass(frozen=True, slots=True)
class SglangVerificationConfig:
    """Immutable opt-in correctness checks for one worker lifetime."""

    attention: bool
    attention_mixed_only: bool
    attention_layer: int | None
    attention_first_partial: bool
    execution: bool
    transfer: bool
    transfer_layer: int | None
    index_map: bool

    @classmethod
    def from_environment(cls) -> "SglangVerificationConfig":
        attention = boolean_environment("NTA_VERIFY_ATTENTION", False)
        mixed_only = boolean_environment("NTA_VERIFY_ATTENTION_MIXED_ONLY", False)
        if mixed_only and not attention:
            raise RuntimeError(
                "NTA_VERIFY_ATTENTION_MIXED_ONLY requires NTA_VERIFY_ATTENTION"
            )
        layer_text = os.environ.get("NTA_VERIFY_ATTENTION_LAYER", "").strip()
        attention_layer = None if not layer_text else int(layer_text)
        if attention_layer is not None and (attention_layer < 0 or not attention):
            raise RuntimeError(
                "NTA_VERIFY_ATTENTION_LAYER requires NTA_VERIFY_ATTENTION and "
                "a nonnegative global layer id"
            )
        first_partial = boolean_environment(
            "NTA_VERIFY_ATTENTION_FIRST_PARTIAL", False
        )
        if first_partial and (not attention or attention_layer is not None):
            raise RuntimeError(
                "NTA_VERIFY_ATTENTION_FIRST_PARTIAL requires "
                "NTA_VERIFY_ATTENTION and excludes NTA_VERIFY_ATTENTION_LAYER"
            )
        transfer = boolean_environment("NTA_VERIFY_TRANSFER", False)
        transfer_layer_text = os.environ.get(
            "NTA_VERIFY_TRANSFER_LAYER", ""
        ).strip()
        transfer_layer = (
            None if not transfer_layer_text else int(transfer_layer_text)
        )
        if transfer_layer is not None and (transfer_layer < 0 or not transfer):
            raise RuntimeError(
                "NTA_VERIFY_TRANSFER_LAYER requires NTA_VERIFY_TRANSFER and "
                "a nonnegative global layer id"
            )
        return cls(
            attention=attention,
            attention_mixed_only=mixed_only,
            attention_layer=attention_layer,
            attention_first_partial=first_partial,
            execution=boolean_environment("NTA_VERIFY_EXECUTION", False),
            transfer=transfer,
            transfer_layer=transfer_layer,
            index_map=boolean_environment("NTA_VERIFY_INDEX_MAP", False),
        )


@dataclass(frozen=True, slots=True)
class SglangObservabilityConfig:
    """Process-start profiling and artifact-output configuration."""

    profile_cpu: bool
    profile_transfer: bool
    profile_index_layout: bool
    profile_index_min_bytes: int
    profile_gpu: bool
    profile_barrier: bool
    stats_file: pathlib.Path | None
    engine_version: str
    revision: str
    opportunity_trace: pathlib.Path | None
    opportunity_model: str
    opportunity_tier: str
    measure_opportunity_compute: bool
    opportunity_parallel_slots: int

    @classmethod
    def from_environment(
        cls,
        *,
        model_runner: Any,
        tier: Any,
        opportunity_parallel_slots: int,
    ) -> "SglangObservabilityConfig":
        if opportunity_parallel_slots <= 0:
            raise ValueError("SGLang opportunity parallelism must be positive")
        configured_stats = os.environ.get("NTA_ENGINE_STATS_FILE", "").strip()
        trace_file = os.environ.get("NTA_OPPORTUNITY_TRACE_FILE", "").strip()
        revision = os.environ.get("NTA_REVISION", "").strip()
        model = os.environ.get(
            "NTA_OPPORTUNITY_MODEL",
            str(getattr(model_runner.model_config, "model_path", "unknown")),
        ).strip()
        if trace_file:
            if not revision:
                raise ValueError(
                    "NTA_REVISION is required when opportunity tracing is enabled"
                )
            if not tier.is_host_staged:
                raise ValueError(
                    "the SGLang opportunity tracer requires the host_staged tier"
                )
        return cls(
            profile_cpu=boolean_environment("NTA_PROFILE_CPU", False),
            profile_transfer=boolean_environment("NTA_PROFILE_TRANSFER", False),
            profile_index_layout=boolean_environment("NTA_PROFILE_INDEX_LAYOUT", False),
            profile_index_min_bytes=positive_environment(
                "NTA_PROFILE_INDEX_LAYOUT_MIN_BYTES", 64 * 1024
            ),
            profile_gpu=boolean_environment("NTA_PROFILE_GPU", False),
            profile_barrier=boolean_environment("NTA_PROFILE_BARRIER", False),
            stats_file=(pathlib.Path(configured_stats) if configured_stats else None),
            engine_version=os.environ.get("NTA_SGLANG_VERSION", "0.5.16").strip(),
            revision=revision or "unknown",
            opportunity_trace=pathlib.Path(trace_file) if trace_file else None,
            opportunity_model=model,
            opportunity_tier=tier.tier.value,
            measure_opportunity_compute=boolean_environment(
                "NTA_OPPORTUNITY_MEASURE_COMPUTE", False
            ),
            opportunity_parallel_slots=positive_environment(
                "NTA_OPPORTUNITY_PARALLEL_SLOTS", opportunity_parallel_slots
            ),
        )

    def __post_init__(self) -> None:
        if min(self.profile_index_min_bytes, self.opportunity_parallel_slots) <= 0:
            raise ValueError("SGLang observability geometry must be positive")
        if not self.engine_version or not self.revision or not self.opportunity_model:
            raise ValueError("SGLang observability identity cannot be empty")


@dataclass(frozen=True, slots=True)
class SglangExecutionTuning:
    """Immutable host acquisition and overlap configuration."""

    host_cost_model: HostCostModel
    mover_stream_priority: int
    incremental_calibration_probes: int
    host_mover_policy: str
    host_mover_default_service_model: IndexedMoverServiceModel
    host_mover_calibration_samples: int
    layer_service_minimum_samples: int
    layer_service_maximum_samples: int
    copy_engine_max_operations: int
    indexed_copy_target_bytes: int
    indexed_copy_max_blocks: int
    frontier_layers_per_wave: int
    sm_acquisition_waves: int
    sm_mover_max_worker_ctas: int
    overlap_enabled: bool
    frontier_enabled: bool
    demand_graph_enabled: bool
    fragment_enabled: bool
    demand_overlap_policy: str
    stream_ordered_retirement: bool
    grouping: str
    demand_graph_capacity: int
    model: SglangModelPartition
    calibration_profile: SglangCalibrationProfileConfig
    verification: SglangVerificationConfig
    observability: SglangObservabilityConfig

    @classmethod
    def from_environment(
        cls,
        *,
        model_runner: Any,
        token_pool: Any,
        tier: Any,
        bootstrap: SglangBootstrapConfig,
        opportunity_parallel_slots: int,
    ) -> "SglangExecutionTuning":
        host_cost_model = HostCostModel.from_environment()
        calibration_probes = incremental_calibration_probe_count(
            execution=bootstrap.execution,
            host_staged=tier.is_host_staged,
        )
        mover_policy = host_mover_environment()
        mover_model = host_mover_service_model_from_environment()
        mover_samples = min(
            32,
            positive_environment("NTA_EXECUTION_HOST_MOVER_CALIBRATION_SAMPLES", 3),
        )
        layer_minimum = min(
            32, positive_environment("NTA_EXECUTION_LAYER_SERVICE_MIN_SAMPLES", 4)
        )
        layer_maximum = min(
            128,
            positive_environment("NTA_EXECUTION_LAYER_SERVICE_MAX_SAMPLES", 32),
        )
        if layer_maximum < layer_minimum:
            raise RuntimeError(
                "NTA layer-service maximum samples are below its minimum"
            )
        copy_maximum = min(
            1 << 16,
            positive_environment("NTA_EXECUTION_COPY_ENGINE_MAX_OPERATIONS", 4096),
        )
        indexed_target = positive_environment(
            "NTA_EXECUTION_INDEXED_COPY_BYTES_PER_CTA", 1024 * 1024
        )
        indexed_blocks = min(
            64, positive_environment("NTA_EXECUTION_INDEXED_COPY_MAX_CTAS", 32)
        )
        sm_mover_max_worker_ctas = min(
            64,
            positive_environment("NTA_EXECUTION_HOST_SM_MAX_WORKER_CTAS", 8),
        )
        frontier_wave = min(
            64,
            positive_environment(
                "NTA_EXECUTION_FRONTIER_LAYERS_PER_WAVE",
                minimum_saturating_pair_layers(sm_mover_max_worker_ctas),
            ),
        )
        overlap_enabled = bootstrap.execution.protocol.allow_overlap
        frontier_enabled = (
            tier.is_host_staged
            and overlap_enabled
            and bootstrap.execution.host_execution_mode
            is not HostExecutionMode.DEVICE_BULK
        )
        # Work-ticket progress rounds and physical SM completion waves are
        # different decisions.  The former does not predict the extra
        # FlashInfer launches or SM contention caused by splitting one K/V
        # transfer.  Keep the production default at one lossless wave; larger
        # values are explicit mechanism/evaluation arms until a measured
        # cross-resource selector owns this choice.
        requested_sm_waves = min(
            16,
            positive_environment(
                "NTA_EXECUTION_SM_ACQUISITION_WAVES",
                1,
            ),
        )
        sm_acquisition_waves = (
            requested_sm_waves
            if overlap_enabled
            and bootstrap.execution.host_execution_mode
            not in {
                HostExecutionMode.DIRECT,
                HostExecutionMode.SCHEDULED_BULK,
            }
            else 1
        )
        graph_enabled, fragment_enabled, overlap_policy = demand_overlap_policy(
            host_staged=tier.is_host_staged,
            frontier_enabled=frontier_enabled,
            graph_requested=boolean_environment("NTA_EXECUTION_GRAPH", False),
        )
        stream_ordered = boolean_environment(
            "NTA_EXECUTION_STREAM_ORDERED_RETIREMENT", False
        )
        if stream_ordered and (not tier.is_host_staged or graph_enabled):
            raise RuntimeError(
                "stream-ordered retirement currently requires host-staged, "
                "eager execution"
            )
        model = SglangModelPartition.from_runner(model_runner, token_pool)
        calibration_profile = SglangCalibrationProfileConfig.from_environment(
            model_runner=model_runner,
            applicable=(
                tier.is_host_staged
                and bootstrap.execution.protocol.kind is not ProtocolKind.CONVENTIONAL
                and bootstrap.execution.host_execution_mode is HostExecutionMode.AUTO
            ),
        )
        graph_capacity = positive_environment(
            "NTA_EXECUTION_GRAPH_CAPACITY", max(64, 4 * model.layer_count)
        )
        return cls(
            host_cost_model=host_cost_model,
            mover_stream_priority=mover_stream_priority(),
            incremental_calibration_probes=calibration_probes,
            host_mover_policy=mover_policy,
            host_mover_default_service_model=mover_model,
            host_mover_calibration_samples=mover_samples,
            layer_service_minimum_samples=layer_minimum,
            layer_service_maximum_samples=layer_maximum,
            copy_engine_max_operations=copy_maximum,
            indexed_copy_target_bytes=indexed_target,
            indexed_copy_max_blocks=indexed_blocks,
            frontier_layers_per_wave=frontier_wave,
            sm_acquisition_waves=sm_acquisition_waves,
            sm_mover_max_worker_ctas=sm_mover_max_worker_ctas,
            overlap_enabled=overlap_enabled,
            frontier_enabled=frontier_enabled,
            demand_graph_enabled=graph_enabled,
            fragment_enabled=fragment_enabled,
            demand_overlap_policy=overlap_policy,
            stream_ordered_retirement=stream_ordered,
            grouping=bootstrap.execution.grouping,
            demand_graph_capacity=graph_capacity,
            model=model,
            calibration_profile=calibration_profile,
            verification=SglangVerificationConfig.from_environment(),
            observability=SglangObservabilityConfig.from_environment(
                model_runner=model_runner,
                tier=tier,
                opportunity_parallel_slots=opportunity_parallel_slots,
            ),
        )

    def requires_typed_host_modules(
        self,
        bootstrap: SglangBootstrapConfig,
        *,
        host_cost_model: HostCostModel | None = None,
        calibration_probes_remaining: int | None = None,
    ) -> bool:
        """Whether setup must build typed modules before engine readiness."""

        model = self.host_cost_model if host_cost_model is None else host_cost_model
        probes = (
            self.incremental_calibration_probes
            if calibration_probes_remaining is None
            else calibration_probes_remaining
        )
        if probes < 0:
            raise ValueError("remaining calibration probes cannot be negative")

        return (
            bootstrap.execution.protocol.kind is not ProtocolKind.CONVENTIONAL
            and (
                bootstrap.execution.host_execution_mode
                in {HostExecutionMode.DEVICE_BULK, HostExecutionMode.DEPENDENCY_AWARE}
                or model.max_rounds > 1
                or bootstrap.tenant_isolation_enabled
            )
            and (
                bootstrap.execution.host_execution_mode
                in {HostExecutionMode.DEVICE_BULK, HostExecutionMode.DEPENDENCY_AWARE}
                or model.incremental_setup_ns is not None
                or probes > 0
                or bootstrap.tenant_isolation_enabled
            )
        )
