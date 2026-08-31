"""Engine-neutral state records used by the SGLang attention adapter.

This module contains no SGLang imports.  It owns only lifetime-safe state
records, graph-cache keys, and the asynchronous statistics publisher used by
the framework-facing attention implementation.  Keeping these records out of
the numerical adapter makes the framework boundary auditable and prevents a
state object from accidentally becoming a second execution protocol.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch

from nta_runtime.engines.sglang_contracts import (
    LeaseAcquisitionGroup,
    LeaseAcquisitionSlice,
    PagePair,
)
from nta_runtime.engines.sglang_acquisition_contract import (
    SglangForwardAcquisition,
)
from nta_runtime.engines.sglang_hicache import PendingHostLoad
from nta_runtime.execution_core import ExecutionPlan, ExecutionSession
from nta_runtime.execution_topology import ExactWorkTopology
from nta_runtime.acquisition_scheduler import (
    LayerAcquisitionFrontier,
    LayerAcquisitionModel,
)
from nta_runtime.execution_planner import HostExecutionPlan, HostLayerExecutionTemplate
from nta_runtime.flashinfer_schedule import Schedule
from nta_runtime.indexed_transfer import AcquisitionTopology
from nta_runtime.requests import RequestBinding

if TYPE_CHECKING:
    from nta_runtime.flashinfer import FlashInferLayerEpoch


@dataclass(frozen=True)
class _BarrierProfile:
    """CUDA event ordering at an explicit producer/consumer wait."""

    arrive: torch.cuda.Event
    ready: torch.cuda.Event
    layer_id: int
    scope: str

    def __post_init__(self) -> None:
        if self.scope not in {"attention_layer", "graph_batch"}:
            raise ValueError("barrier profile has an invalid scope")


@dataclass(frozen=True)
class _OperatorProfile:
    """One asynchronously collected CUDA operator observation.

    Service predictions are attached only to bounded AUTO calibration probes.
    Their scale is frozen with the decision so delayed collection cannot
    accidentally compound a newer model update.
    """

    start: torch.cuda.Event
    finish: torch.cuda.Event
    kind: str
    covered_layers: int
    service_prediction_ns: int | None = None
    service_prediction_scale: float | None = None

    def __post_init__(self) -> None:
        if not self.kind or self.covered_layers <= 0:
            raise ValueError("operator profile geometry is invalid")
        has_prediction = self.service_prediction_ns is not None
        if has_prediction != (self.service_prediction_scale is not None):
            raise ValueError("operator service prediction is incomplete")
        if has_prediction and (
            self.service_prediction_ns <= 0
            or not 0.125 <= self.service_prediction_scale <= 64.0
        ):
            raise ValueError("operator service prediction is invalid")


@dataclass(frozen=True)
class _FragmentLookahead:
    layer_id: int
    wrapper_id: int
    object_count: int
    preloaded_object_count: int
    key_source: int
    key_staging: int
    value_source: int
    value_staging: int
    ready_event: torch.cuda.Event


@dataclass(frozen=True)
class _SemanticWrapperPlan:
    """Immutable exact plan built once from one FlashInfer ForwardBatch.

    This record owns request/work identity and dependency geometry only.  It
    deliberately carries no layer tensor address, object version, CUDA event,
    or runtime slot lifetime; those are physical materialization state and are
    rebound by the numerical layer that consumes them.
    """

    schedule: Schedule
    topology: ExactWorkTopology
    dependency_kind: str
    work_dependency_rows: tuple[int, ...]
    signature_prefix: tuple[Any, ...]
    page_pairs: tuple[PagePair, ...] = ()
    acquisition_slices: tuple[LeaseAcquisitionSlice | None, ...] = ()
    acquisition_groups: tuple[LeaseAcquisitionGroup | None, ...] = ()
    indexed_topology: AcquisitionTopology | None = None

    def __post_init__(self) -> None:
        work_count = self.schedule.work_count
        if (
            self.dependency_kind not in {"typed_lease", "physical_pages", "direct"}
            or self.topology.work_count != work_count
            or len(self.work_dependency_rows) != work_count
            or len(self.signature_prefix) != 4
        ):
            raise ValueError("SGLang semantic wrapper plan geometry is invalid")
        if self.dependency_kind == "typed_lease":
            if (
                len(self.acquisition_slices) != work_count
                or len(self.acquisition_groups) != work_count
                or self.page_pairs
                or self.indexed_topology is None
                or self.indexed_topology.work_count != work_count
            ):
                raise ValueError("typed lease semantic plan is incomplete")
            expected_rows = tuple(
                0 if item is None else item.row_count
                for item in self.acquisition_slices
            )
        elif self.dependency_kind == "physical_pages":
            if (
                len(self.page_pairs) != work_count
                or self.acquisition_slices
                or self.acquisition_groups
                or self.indexed_topology is not None
            ):
                raise ValueError("physical semantic plan is incomplete")
            expected_rows = tuple(len(pair[0]) for pair in self.page_pairs)
        else:
            if (
                self.page_pairs
                or self.acquisition_slices
                or self.acquisition_groups
                or self.indexed_topology is not None
            ):
                raise ValueError("direct semantic plan retained dependencies")
            expected_rows = (0,) * work_count
        if self.work_dependency_rows != expected_rows:
            raise ValueError("semantic dependency rows disagree with exact demand")
        if self.topology.demand_units != tuple(max(1, rows) for rows in expected_rows):
            raise ValueError("semantic work topology disagrees with dependencies")


@dataclass(frozen=True, slots=True, kw_only=True)
class SglangForwardPlan:
    """Immutable semantic and resource identity for one framework forward."""

    bindings: tuple[RequestBinding, ...]
    semantic_plans: Mapping[int, _SemanticWrapperPlan]
    pending_host_load: PendingHostLoad | None
    host_execution: HostExecutionPlan | None = None
    grouping: str = "request"
    # Exact operation-local interval required by each FlashInfer work unit.
    # Transport ownership remains one lease; repeated CTA/head work shares the
    # same typed interval.
    lease_transfer_rows: int = 0

    def __post_init__(self) -> None:
        if self.grouping not in {"request", "tile"}:
            raise ValueError("SGLang forward plan has an invalid grouping")
        if self.lease_transfer_rows < 0:
            raise ValueError("SGLang forward plan has negative lease rows")
        object.__setattr__(self, "bindings", tuple(self.bindings))
        object.__setattr__(
            self,
            "semantic_plans",
            MappingProxyType(dict(self.semantic_plans)),
        )

    def with_wrapper_identity(
        self, source_to_target: Mapping[int, int]
    ) -> SglangForwardPlan:
        """Return the same semantic plan keyed by adopted wrapper identity."""

        mapping = dict(source_to_target)
        source_ids = set(mapping)
        target_ids = set(mapping.values())
        if not mapping or len(target_ids) != len(mapping):
            raise RuntimeError(
                "FlashInfer wrapper adoption must be non-empty and injective"
            )
        if set(self.semantic_plans) != source_ids:
            raise RuntimeError(
                "FlashInfer wrapper adoption does not cover its semantic plans"
            )
        remapped = {
            mapping[source]: value for source, value in self.semantic_plans.items()
        }
        return SglangForwardPlan(
            bindings=self.bindings,
            semantic_plans=remapped,
            pending_host_load=self.pending_host_load,
            host_execution=self.host_execution,
            grouping=self.grouping,
            lease_transfer_rows=self.lease_transfer_rows,
        )


@dataclass(kw_only=True, slots=True)
class SglangForwardEpoch:
    """Sole mutable execution state for one immutable forward plan."""

    plan: SglangForwardPlan
    acquisition: SglangForwardAcquisition | None = None
    fragment_lookahead: dict[int, _FragmentLookahead] = field(default_factory=dict)
    # One transport submission can publish the same completion fence for
    # several adjacent layers. Once that fence is ordered on the numerical
    # stream, a racing CPU Event.query must not create another partial
    # consumer for a later layer in the group.
    ordered_prefetch_event_ids: set[int] = field(default_factory=set)
    # Layers published by the calibrated EDF frontier are predicted to finish
    # before their numerical deadline.  This is a performance contract, not a
    # readiness assertion: the stock consumer still orders the producer event,
    # so model error can add a wait but can never expose incomplete KV.  Keeping
    # this decision on the forward epoch prevents a launch-ahead CPU
    # ``Event.query`` from selecting an expensive partial consumer for work that
    # will be ready when the GPU actually reaches attention.
    modeled_ready_by_attention_layers: set[int] = field(default_factory=set)
    # A partial consumer is a planned execution form, never the absence of a
    # readiness proof.  AUTO leaves this set empty until a calibrated
    # arrival/cost policy proves that consuming waves is cheaper than waiting
    # for the complete producer fence.  Bounded calibration and explicit
    # causal modes may populate it to measure that execution form.
    planned_progressive_consumer_layers: set[int] = field(default_factory=set)
    # Pure orchestration geometry is invariant across layers for one FlashInfer
    # wrapper.  Object addresses and versions are rebound separately by the
    # materializer, so caching this template does not retain tier resources.
    host_layer_templates: dict[tuple[int, int, int], HostLayerExecutionTemplate] = (
        field(default_factory=dict)
    )
    # RuntimeView owns one reusable runnable queue.  This key names the exact
    # wrapper/plan partition currently published there; switching wrappers
    # republishes on the same consumer stream, while repeated transformer layers
    # reuse the immutable order without reset or dependency discovery.
    arriving_partition_key: tuple[int, int, int, int, tuple[int, ...]] | None = None
    # Expensive page-table/lease projection checks are verification-only and
    # run once per adopted wrapper, never once per transformer layer.
    verified_projection_wrappers: set[int] = field(default_factory=set)
    execution: ExecutionPlan | None = None
    verification_session: ExecutionSession | None = None
    # Measured recurring control work between the direct/incremental decision
    # and the first typed attention dispatch. It is combined with that first
    # dispatch's CPU issue time exactly once to calibrate later batches.
    incremental_metadata_setup_ns: int = 0
    incremental_setup_observed: bool = False
    incremental_setup_observation_ns: int = 0
    # A bounded calibration records adjacent attention arrivals for this
    # forward shape. It is performance state only and never participates in
    # request identity, demand exactness, or numerical ordering.
    layer_service_key: tuple[str, int, int] | None = None
    layer_arrival_event: torch.cuda.Event | None = None
    layer_arrival_local_layer: int = -1
    # Freeze one calibrated EDF model for this forward. Observations completed
    # during the forward calibrate later batches; rebuilding the same service
    # vector at every layer changes no decision and taxes the launch thread.
    deadline_model: LayerAcquisitionModel | None = None
    deadline_model_initialized: bool = False
    deadline_frontier: LayerAcquisitionFrontier | None = None
    # Per-forward dispatch composition. This is deliberately separate from
    # readiness: choosing the native numerical form does not prove that data
    # was incomplete when attention arrived. Calibration may temporarily make
    # native/framework layers non-prefix; the production evidence gate records
    # that composition instead of turning a legal per-layer choice into a
    # serving failure.
    native_dispatch_external_layers: int = 0
    framework_dispatch_external_layers: int = 0
    progressive_consumer_external_layers: int = 0
    external_last_local_layer: int = -1
    framework_dispatch_seen: bool = False
    native_dispatch_nonprefix_seen: bool = False
    external_dispatch_recorded: bool = False
    # Event-owned numerical kernels can consume one immutable work topology for
    # the whole forward without rebinding runtime objects. Native indexed
    # acquisition instead retires per layer before its directory slots are
    # reused, so it never installs an epoch here.
    stream_ordered_epoch: FlashInferLayerEpoch | None = None
    stream_ordered_progress_rounds: int = 0
    stream_ordered_layers: int = 0

    @property
    def bindings(self) -> tuple[RequestBinding, ...]:
        return self.plan.bindings

    @property
    def semantic_plans(self) -> Mapping[int, _SemanticWrapperPlan]:
        return self.plan.semantic_plans

    @property
    def pending_host_load(self) -> PendingHostLoad | None:
        return self.plan.pending_host_load

    @property
    def host_execution(self) -> HostExecutionPlan | None:
        return self.plan.host_execution

    @property
    def grouping(self) -> str:
        return self.plan.grouping

    @property
    def lease_transfer_rows(self) -> int:
        return self.plan.lease_transfer_rows

    def require_unstarted(self, operation: str) -> None:
        """Reject an identity transition after numerical execution can observe it."""

        if not operation:
            raise ValueError("forward epoch operation must be named")
        if (
            self.fragment_lookahead
            or self.ordered_prefetch_event_ids
            or self.modeled_ready_by_attention_layers
            or self.planned_progressive_consumer_layers
            or self.host_layer_templates
            or self.arriving_partition_key is not None
            or self.verified_projection_wrappers
            or self.execution is not None
            or self.verification_session is not None
            or self.incremental_metadata_setup_ns != 0
            or self.incremental_setup_observed
            or self.incremental_setup_observation_ns != 0
            or self.layer_arrival_event is not None
            or self.layer_arrival_local_layer != -1
            or self.deadline_model is not None
            or self.deadline_model_initialized
            or self.deadline_frontier is not None
            or self.native_dispatch_external_layers != 0
            or self.framework_dispatch_external_layers != 0
            or self.progressive_consumer_external_layers != 0
            or self.external_last_local_layer != -1
            or self.framework_dispatch_seen
            or self.native_dispatch_nonprefix_seen
            or self.external_dispatch_recorded
            or self.stream_ordered_epoch is not None
            or self.stream_ordered_progress_rounds != 0
            or self.stream_ordered_layers != 0
        ):
            raise RuntimeError(f"SGLang cannot {operation} after execution began")

    def adopt_wrapper_identity(self, source_to_target: Mapping[int, int]) -> None:
        """Re-key a validated plan after zero-copy numerical-module adoption.

        FlashInfer's stock and typed wrappers share one immutable plan and its
        workspace after adoption.  Schedule and acquisition metadata belong to
        that plan, not to either Python wrapper object, so rebuilding them from
        the CUDA workspace would add a redundant D2H synchronization.  Wrapper
        replacement is legal only before any layer-specific execution state is
        created; fail closed if that lifecycle boundary has already passed.
        """

        self.require_unstarted("adopt FlashInfer wrapper identity")
        self.plan = self.plan.with_wrapper_identity(source_to_target)
