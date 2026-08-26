"""Reusable bounded-HBM execution over canonical FlashInfer partials."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import os
import pathlib
from typing import Any

import flashinfer
import torch

from .bounded_staging import BoundedStagingPool
from .flashinfer import (
    BIND_CURRENT_GENERATION,
    attention_jit_args,
    direct_requirement,
    pack_work_metadata,
    request_bound_attention_jit_args,
)
from .flashinfer_schedule import paged_prefill_schedule
from .runtime import (
    AcquireRequirement,
    DeviceWorkPlan,
    IndexedHostObject,
    JitPhaseProgram,
    OperatorCapability,
    OperatorAccessProof,
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
    RequestRange,
    Runtime,
    RuntimeConfig,
    WorkItem,
    copy_host_to_device_async,
    require_operator_pair,
)
from .tier_streaming import TierStreamingSchedule


RequestKey = tuple[int, int]
_COMPILED_PLAN_FLAGS = (
    OperatorPlanFlag.FIXED_CAPACITY
    | OperatorPlanFlag.GRAPH_STABLE
    | OperatorPlanFlag.EXTERNAL_WAVE_SOURCES
    | OperatorPlanFlag.GENERATION_BOUND
    | OperatorPlanFlag.EXACT_COMPLETE_MERGE
)


def _cumulative(values: Sequence[int]) -> tuple[int, ...]:
    result = [0]
    for value in values:
        result.append(result[-1] + int(value))
    return tuple(result)


@dataclass(frozen=True)
class _ResidentRun:
    query_begin: int
    query_end: int
    kv_begin: int
    kv_end: int
    wrapper: Any


@dataclass(frozen=True)
class _CompiledWorkPlan:
    plan: DeviceWorkPlan
    work_ticket_begin: int
    reduction_group_begin: int


@dataclass(frozen=True)
class FlashInferHostWave:
    """Pinned host KV and a planned canonical FlashInfer partial."""

    key: torch.Tensor
    value: torch.Tensor
    query_rows: int
    wrapper: Any

    def validate(self) -> None:
        if self.key.device.type != "cpu" or self.value.device.type != "cpu":
            raise ValueError("FlashInfer tier waves must originate in host memory")
        if not self.key.is_pinned() or not self.value.is_pinned():
            raise ValueError("FlashInfer tier waves require pinned host tensors")
        if self.key.shape != self.value.shape or self.key.dtype != self.value.dtype:
            raise ValueError("FlashInfer tier-wave K/V tensors must match")
        if self.key.ndim < 2 or self.key.shape[0] <= 0 or self.query_rows <= 0:
            raise ValueError("FlashInfer tier-wave geometry must be positive")
        if self.wrapper is None:
            raise ValueError("FlashInfer tier wave requires a planned wrapper")

    @property
    def token_count(self) -> int:
        return int(self.key.shape[0])


@dataclass(frozen=True)
class _ExecutorEvents:
    start: torch.cuda.Event
    ready: tuple[torch.cuda.Event, ...]
    released: tuple[torch.cuda.Event, ...]
    finished: torch.cuda.Event


class FlashInferTierStreamingExecutor:
    """Double-buffer host KV while canonical partial attention consumes it."""

    def __init__(
        self,
        schedule: TierStreamingSchedule,
        waves: Sequence[FlashInferHostWave],
        *,
        slot_count: int = 2,
        device: torch.device | str | int | None = None,
        copy_stream: torch.cuda.Stream | None = None,
        compute_stream: torch.cuda.Stream | None = None,
        transfer_wave: Callable[[int, torch.cuda.Stream], None] | None = None,
        max_inflight_epochs: int = 8,
    ) -> None:
        if slot_count < 2:
            raise ValueError("tier streaming requires at least two staging slots")
        if len(waves) != len(schedule.waves) or not waves:
            raise ValueError("FlashInfer host waves must match the finite schedule")
        if max_inflight_epochs <= 0:
            raise ValueError("in-flight epoch capacity must be positive")
        self.schedule = schedule
        self.waves = tuple(waves)
        for planned, wave in zip(schedule.waves, self.waves, strict=True):
            wave.validate()
            if wave.token_count != planned.token_count:
                raise ValueError("FlashInfer host-wave tokens disagree with the plan")

        first = self.waves[0]
        if any(
            wave.key.shape[1:] != first.key.shape[1:]
            or wave.key.dtype != first.key.dtype
            for wave in self.waves[1:]
        ):
            raise ValueError("FlashInfer tier waves require one KV shape and dtype")
        target = (
            torch.device("cuda", device)
            if isinstance(device, int)
            else torch.device("cuda" if device is None else device)
        )
        if target.type != "cuda":
            raise ValueError("FlashInfer tier streaming requires a CUDA device")
        self.slot_count = slot_count
        self.compute_stream = compute_stream or torch.cuda.current_stream(target)
        self.copy_stream = copy_stream or torch.cuda.Stream(device=target, priority=0)
        self.transfer_wave = transfer_wave
        self._staging_pool = BoundedStagingPool.allocate(
            1,
            slot_count * schedule.maximum_wave_tokens,
            schedule.maximum_wave_tokens,
            tuple(first.key.shape[1:]),
            dtype=first.key.dtype,
            device=target,
        )
        self._staging_leases = tuple(
            self._staging_pool.acquire(slot + 1) for slot in range(slot_count)
        )
        staging = tuple(
            self._staging_pool.view(lease, 0) for lease in self._staging_leases
        )
        self.staging_key = [pair[0] for pair in staging]
        self.staging_value = [pair[1] for pair in staging]
        self._event_sets = tuple(
            _ExecutorEvents(
                torch.cuda.Event(),
                tuple(torch.cuda.Event() for _ in self.waves),
                tuple(torch.cuda.Event() for _ in self.waves),
                torch.cuda.Event(),
            )
            for _ in range(max_inflight_epochs)
        )
        self._event_set_used = [False] * max_inflight_epochs
        self._event_cursor = -1
        first_events = self._event_sets[0]
        self._start = first_events.start
        self._ready = list(first_events.ready)
        self._slot_free = list(first_events.released)
        self._finished = first_events.finished
        self._serialize_eager_runs = True

    def _begin_eager_epoch(self) -> None:
        previous = (
            None if self._event_cursor < 0 else self._event_sets[self._event_cursor]
        )
        candidate = (self._event_cursor + 1) % len(self._event_sets)
        events = self._event_sets[candidate]
        if self._event_set_used[candidate] and not events.finished.query():
            raise RuntimeError("tier-streaming in-flight epoch capacity was exhausted")
        if previous is not None:
            self.copy_stream.wait_event(previous.finished)
        self._event_cursor = candidate
        self._event_set_used[candidate] = True
        self._start = events.start
        self._ready = list(events.ready)
        self._slot_free = list(events.released)
        self._finished = events.finished

    @property
    def staging_tokens(self) -> int:
        return self.slot_count * self.schedule.maximum_wave_tokens

    @property
    def staging_bytes(self) -> int:
        return self._staging_pool.capacity_bytes

    def _enqueue_copy(self, wave_index: int) -> None:
        slot = wave_index % self.slot_count
        wave = self.waves[wave_index]
        previous_wave = wave_index - self.slot_count
        if previous_wave >= 0:
            self.copy_stream.wait_event(self._slot_free[previous_wave])
        if self.transfer_wave is None:
            bytes = wave.key.numel() * wave.key.element_size()
            copy_host_to_device_async(
                self.staging_key[slot].data_ptr(),
                wave.key.data_ptr(),
                bytes,
                self.copy_stream,
            )
            copy_host_to_device_async(
                self.staging_value[slot].data_ptr(),
                wave.value.data_ptr(),
                bytes,
                self.copy_stream,
            )
        else:
            self.transfer_wave(wave_index, self.copy_stream)
        self._ready[wave_index].record(self.copy_stream)

    def run(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        lse: torch.Tensor,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        *,
        run_base: Callable[[], None],
        run_partial: Callable[
            [
                FlashInferHostWave,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
            None,
        ]
        | None = None,
        completion_events: dict[RequestKey, torch.cuda.Event] | None = None,
    ) -> None:
        """Enqueue one finite streaming operator on the configured streams."""

        if not all(
            tensor.is_cuda
            for tensor in (query, output, lse, partial_output, partial_lse)
        ):
            raise ValueError("FlashInfer tier-streaming compute tensors must be CUDA")
        if output.shape != partial_output.shape or lse.shape != partial_lse.shape:
            raise ValueError("FlashInfer partial and accumulated buffers must match")

        if self._serialize_eager_runs:
            self._begin_eager_epoch()
        self._start.record(self.compute_stream)
        self.copy_stream.wait_event(self._start)
        for wave_index in range(min(self.slot_count, len(self.waves))):
            self._enqueue_copy(wave_index)
        run_base()

        if completion_events is not None:
            for request in self.schedule.requests:
                if request.external_tokens == 0:
                    completion_events[request.key].record(self.compute_stream)

        for wave_index, (planned, wave) in enumerate(
            zip(self.schedule.waves, self.waves, strict=True)
        ):
            slot = wave_index % self.slot_count
            self.compute_stream.wait_event(self._ready[wave_index])
            rows = wave.query_rows
            if run_partial is None:
                wave.wrapper.run(
                    query[:rows],
                    self.staging_key[slot][: wave.token_count],
                    self.staging_value[slot][: wave.token_count],
                    out=partial_output[:rows],
                    lse=partial_lse[:rows],
                    return_lse=True,
                )
            else:
                run_partial(
                    wave,
                    query[:rows],
                    self.staging_key[slot][: wave.token_count],
                    self.staging_value[slot][: wave.token_count],
                    partial_output[:rows],
                    partial_lse[:rows],
                )
            flashinfer.merge_state_in_place(
                output[:rows],
                lse[:rows],
                partial_output[:rows],
                partial_lse[:rows],
            )
            if completion_events is not None:
                for request_key in planned.completed_request_keys:
                    completion_events[request_key].record(self.compute_stream)
            self._slot_free[wave_index].record(self.compute_stream)
            next_wave = wave_index + self.slot_count
            if next_wave < len(self.waves):
                self._enqueue_copy(next_wave)
        if self._serialize_eager_runs:
            self._finished.record(self.compute_stream)


@dataclass(frozen=True)
class FlashInferTierStreamingGraph:
    """Captured finite operator with stable structural and staging addresses."""

    graph: torch.cuda.CUDAGraph
    operator: "FlashInferTierStreamingOperator"
    retained_tensors: tuple[torch.Tensor, ...]
    retained_events: tuple[torch.cuda.Event, ...]

    def replay(
        self,
        external_key: torch.Tensor | None = None,
        external_value: torch.Tensor | None = None,
    ) -> None:
        if (external_key is None) != (external_value is None):
            raise ValueError("graph replay requires both external K and V")
        if external_key is not None and external_value is not None:
            self.operator.bind_external(external_key, external_value)
        self.graph.replay()


class FlashInferTierStreamingOperator:
    """Plan and execute one exact request-aware incremental attention form."""

    def __init__(
        self,
        schedule: TierStreamingSchedule,
        external_key: torch.Tensor,
        external_value: torch.Tensor,
        workspace: torch.Tensor,
        *,
        qo_heads: int,
        backend: str = "fa2",
        slot_count: int = 2,
        device: torch.device | str | int | None = None,
        copy_stream: torch.cuda.Stream | None = None,
        compute_stream: torch.cuda.Stream | None = None,
        compiler_module_tag: str | None = None,
        gpu_initiated_host: bool = False,
    ) -> None:
        if not schedule.requests or not schedule.waves:
            raise ValueError("incremental FlashInfer operator needs external work")
        if external_key.device.type != "cpu" or external_value.device.type != "cpu":
            raise ValueError("incremental FlashInfer source KV must be in host memory")
        if not external_key.is_pinned() or not external_value.is_pinned():
            raise ValueError("incremental FlashInfer source KV must be pinned")
        if (
            external_key.shape != external_value.shape
            or external_key.dtype != external_value.dtype
            or external_key.ndim != 3
        ):
            raise ValueError("incremental FlashInfer source K/V geometry must match")
        if external_key.shape[0] != schedule.external_tokens:
            raise ValueError("incremental FlashInfer source KV disagrees with schedule")
        if not workspace.is_cuda or workspace.dtype != torch.uint8:
            raise ValueError("FlashInfer workspace must be a CUDA byte tensor")
        if qo_heads <= 0 or qo_heads % int(external_key.shape[1]) != 0:
            raise ValueError("query heads must be divisible by KV heads")
        if gpu_initiated_host and compiler_module_tag is None:
            raise ValueError(
                "GPU-initiated host acquisition requires compiler transformation"
            )

        self.schedule = schedule
        self.query_lengths = tuple(
            request.query_tokens for request in schedule.requests
        )
        self.resident_lengths = tuple(
            request.resident_tokens for request in schedule.requests
        )
        self.external_lengths = tuple(
            request.external_tokens for request in schedule.requests
        )
        self.query_offsets = _cumulative(self.query_lengths)
        self.resident_offsets = _cumulative(self.resident_lengths)
        self.external_offsets = _cumulative(self.external_lengths)
        self.qo_heads = qo_heads
        self.kv_heads = int(external_key.shape[1])
        self.head_dim = int(external_key.shape[2])
        self.dtype = external_key.dtype
        self.backend = backend
        self.workspace = workspace
        self._metadata: list[torch.Tensor] = []
        self._compiled_runtime: Runtime | None = None
        self._gpu_initiated_host = gpu_initiated_host
        self._compiled_object_count = 0
        self._compiled_host_indices: list[torch.Tensor] = []
        self._compiled_plans: dict[int, _CompiledWorkPlan] = {}
        self._compiled_completion_plan: DeviceWorkPlan | None = None
        self._compiled_completion_work: list[WorkItem] = []
        self._compiled_completion_dependencies: list[AcquireRequirement] = []
        self._compiled_completion_ranges: list[RequestRange] = []
        self._wrapper_forms: dict[int, OperatorForm] = {}
        self._wrapper_request_counts: dict[int, int] = {}
        self._wrapper_request_slots: dict[int, tuple[int, ...]] = {}
        self._compiler_programs: tuple[JitPhaseProgram, JitPhaseProgram] | None = None
        self._operator_plan: OperatorPlan | None = None
        self._direct_jit_args: list[Any] | None = None
        self._incremental_jit_args: list[Any] | None = None
        self._compiled_work_count = 0
        self._compiled_reduction_group_count = 0
        self._last_compiled_work_count = 0
        if compiler_module_tag is not None:
            self._prepare_compiler_modules(compiler_module_tag)
        self._local_wrapper = self._wrapper(
            self.query_lengths,
            self.query_lengths,
            causal=True,
            request_slots=tuple(range(len(self.schedule.requests))),
            form=OperatorForm.DIRECT,
        )
        self._resident_runs = self._build_resident_runs()
        host_waves = self._build_host_waves(external_key, external_value)
        self._wrapper_wave_indices = {
            id(wave.wrapper): index for index, wave in enumerate(host_waves)
        }
        self.executor = FlashInferTierStreamingExecutor(
            schedule,
            host_waves,
            slot_count=slot_count,
            device=device,
            copy_stream=copy_stream,
            compute_stream=compute_stream,
        )
        if compiler_module_tag is not None:
            self._initialize_compiler_runtime()
            self._load_compiler_pair()
            if self._gpu_initiated_host:
                self._enable_gpu_initiated_host()

    @property
    def staging_tokens(self) -> int:
        return self.executor.staging_tokens

    @property
    def compiler_transformed(self) -> bool:
        return self._compiler_programs is not None

    @property
    def operator_plan(self) -> OperatorPlan | None:
        return self._operator_plan

    @property
    def compiler_runtime_protocol_active(self) -> bool:
        return (
            self._compiler_programs is not None and self._last_compiled_work_count > 0
        )

    @property
    def gpu_initiated_host(self) -> bool:
        return self._gpu_initiated_host

    def verify_compiler_epoch(self) -> None:
        """Fail closed unless the last generated incremental launch retired."""

        if not self.compiler_runtime_protocol_active or self._compiled_runtime is None:
            raise RuntimeError("no generated incremental epoch has executed")
        status = self._compiled_runtime.epoch_status(self._last_compiled_work_count)
        if not status.succeeded:
            raise RuntimeError(
                "generated FlashInfer epoch did not retire exactly "
                f"(new={status.fresh}, pending={status.pending}, "
                f"ready={status.ready}, failed={status.failed}, "
                f"cancelled={status.cancelled})"
            )

    def compiler_epoch_status(self):
        """Return terminal state for the last generated finite operator."""

        if not self.compiler_runtime_protocol_active or self._compiled_runtime is None:
            raise RuntimeError("no generated incremental epoch has executed")
        return self._compiled_runtime.epoch_status(self._last_compiled_work_count)

    def rebind_request(
        self, request_slot: int, request_id: int, generation: int
    ) -> None:
        """Publish a reused request slot before a later eager or graph launch."""

        if self._compiled_runtime is None:
            raise RuntimeError("request rebinding requires a compiled operator")
        if request_slot < 0 or request_slot >= len(self.schedule.requests):
            raise ValueError("request slot is outside the finite operator")
        request = self.schedule.requests[request_slot]
        self._compiled_runtime.set_request(
            request_slot,
            request_id,
            generation,
            tenant_id=request.tenant_id,
            priority=request.priority,
            deadline_clock=request.deadline_ns,
        )

    def cancel_request(self, request_slot: int, generation: int) -> None:
        """Cancel one published generation before a later operator launch."""

        if self._compiled_runtime is None:
            raise RuntimeError("request cancellation requires a compiled operator")
        self._compiled_runtime.cancel_request(request_slot, generation)

    def _prepare_compiler_modules(self, module_tag: str) -> None:
        if os.environ.get("NTA_FLASHINFER_HOOK") != "1":
            raise RuntimeError(
                "compiler tier streaming requires tools/jit/activate.py "
                "--flashinfer-hook"
            )
        workspace = os.environ.get("FLASHINFER_WORKSPACE_BASE")
        if not workspace:
            raise RuntimeError("compiler tier streaming has no FlashInfer workspace")
        if not module_tag.strip():
            raise ValueError("compiler module tag must contain an identifier")
        signature = (
            f"tier_{self.backend}_q{self.qo_heads}_kv{self.kv_heads}_h{self.head_dim}_"
            f"{str(self.dtype).replace('torch.', '')}"
        )
        direct_name = f"nta_sglang_prefill_request_bound_tier_v4_{signature}"
        incremental_name = f"nta_sglang_prefill_demand_acquire_tier_v4_{signature}"
        self._compiler_module_names = (direct_name, incremental_name)
        self._direct_jit_args = request_bound_attention_jit_args(
            direct_name,
            dtype_q=self.dtype,
            dtype_kv=self.dtype,
            dtype_o=self.dtype,
            idtype=torch.int32,
            head_dim_qk=self.head_dim,
            head_dim_vo=self.head_dim,
        )
        self._incremental_jit_args = attention_jit_args(
            incremental_name,
            dtype_q=self.dtype,
            dtype_kv=self.dtype,
            dtype_o=self.dtype,
            idtype=torch.int32,
            head_dim_qk=self.head_dim,
            head_dim_vo=self.head_dim,
        )

    def _initialize_compiler_runtime(self) -> None:
        incremental_wrappers = [
            wave.wrapper
            for wave in self.executor.waves
            if self._wrapper_forms.get(id(wave.wrapper)) == OperatorForm.INCREMENTAL
        ]
        work_count = sum(
            paged_prefill_schedule(wrapper).work_count
            for wrapper in incremental_wrappers
        )
        capacity = work_count
        self._compiled_object_count = (
            2 * len(self.executor.waves) if self._gpu_initiated_host else 1
        )
        if capacity <= 0:
            raise RuntimeError("compiled FlashInfer operator has no incremental work")
        self._compiled_runtime = Runtime(
            RuntimeConfig(
                request_capacity=len(self.schedule.requests),
                object_capacity=self._compiled_object_count,
                intent_capacity=max(capacity, self._compiled_object_count),
                work_ticket_capacity=capacity,
                max_dependencies_per_work_ticket=(2 if self._gpu_initiated_host else 1),
                device_ordinal=self.workspace.device.index
                if self.workspace.device.index is not None
                else -1,
            )
        )
        for slot, request in enumerate(self.schedule.requests):
            self._compiled_runtime.set_request(
                slot,
                request.request_id,
                request.generation,
                tenant_id=request.tenant_id,
                priority=request.priority,
                deadline_clock=request.deadline_ns,
            )
        for wrapper in incremental_wrappers:
            request_slots = self._wrapper_request_slots[id(wrapper)]
            self._compiled_plans[id(wrapper)] = self._build_compiled_plan(
                wrapper, request_slots
            )
        if self._compiled_work_count != work_count:
            raise RuntimeError("compiled FlashInfer ticket allocation is inconsistent")
        self._compiled_completion_plan = DeviceWorkPlan(
            work_count,
            len(self._compiled_completion_dependencies),
            self._compiled_runtime.device_ordinal,
        )
        self._compiled_completion_plan.upload(
            self._compiled_completion_work,
            self._compiled_completion_dependencies,
            self._compiled_completion_ranges,
        )
        self._compiled_completion_plan.synchronize_upload()

    def _wrapper(
        self,
        query_lengths: Sequence[int],
        kv_lengths: Sequence[int],
        *,
        causal: bool,
        request_slots: Sequence[int],
        form: OperatorForm,
    ) -> Any:
        if not query_lengths or len(query_lengths) != len(kv_lengths):
            raise ValueError("FlashInfer operator needs matched query and KV ranges")
        if len(request_slots) != len(query_lengths) or any(
            slot < 0 or slot >= len(self.schedule.requests) for slot in request_slots
        ):
            raise ValueError("FlashInfer operator request slots disagree with ranges")
        if any(query <= 0 or kv <= 0 for query, kv in zip(query_lengths, kv_lengths)):
            raise ValueError("FlashInfer operator ranges must be positive")
        query_indptr = torch.tensor(
            _cumulative(query_lengths), dtype=torch.int32, device=self.workspace.device
        )
        kv_indptr = torch.tensor(
            _cumulative(kv_lengths), dtype=torch.int32, device=self.workspace.device
        )
        self._metadata.extend((query_indptr, kv_indptr))
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace,
            "NHD",
            backend=self.backend,
            jit_args=(
                self._direct_jit_args
                if form == OperatorForm.DIRECT
                else self._incremental_jit_args
            ),
        )
        wrapper.plan(
            query_indptr,
            kv_indptr,
            self.qo_heads,
            self.kv_heads,
            self.head_dim,
            causal=causal,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
            o_data_type=self.dtype,
            non_blocking=False,
            disable_split_kv=False,
        )
        if self._direct_jit_args is not None:
            self._wrapper_forms[id(wrapper)] = form
            self._wrapper_request_counts[id(wrapper)] = len(request_slots)
            self._wrapper_request_slots[id(wrapper)] = tuple(request_slots)
        return wrapper

    def _build_compiled_plan(
        self, wrapper: Any, request_slots: tuple[int, ...]
    ) -> _CompiledWorkPlan:
        runtime = self._compiled_runtime
        if runtime is None:
            raise RuntimeError("compiled FlashInfer plan has no runtime")
        schedule = paged_prefill_schedule(wrapper)
        if not schedule.request_indices or max(schedule.request_indices) >= len(
            request_slots
        ):
            raise RuntimeError("FlashInfer emitted an invalid request coordinate")
        counts = [0] * len(request_slots)
        for request_index in schedule.request_indices:
            counts[request_index] += 1
        if any(count == 0 for count in counts):
            raise RuntimeError("FlashInfer emitted no work for a compiled request")
        contributor_indices = [0] * len(request_slots)
        work_ticket_begin = self._compiled_work_count
        reduction_group_begin = self._compiled_reduction_group_count
        work: list[WorkItem] = []
        dependencies: list[AcquireRequirement] = []
        wave_index = self._wrapper_wave_indices.get(id(wrapper))
        if self._gpu_initiated_host and wave_index is None:
            raise RuntimeError("incremental wrapper has no host wave identity")
        for local_work_ticket, (request_index, logical_work) in enumerate(
            zip(schedule.request_indices, schedule.kv_tile_indices, strict=True)
        ):
            work_ticket = work_ticket_begin + local_work_ticket
            slot = request_slots[request_index]
            request = self.schedule.requests[slot]
            local_dependency_begin = len(dependencies)
            completion_dependency_begin = len(self._compiled_completion_dependencies)
            if self._gpu_initiated_host:
                if wave_index is None:  # pragma: no cover - checked above
                    raise RuntimeError("incremental wrapper has no host wave identity")
                wave = self.executor.waves[wave_index]
                dependency_count = 2
                direct_dependency_count = 0
                requirements = tuple(
                    AcquireRequirement(
                        0,
                        0,
                        0x4E54414800000000 | object_slot,
                        0,
                        object_slot,
                        1,
                        wave.key.nbytes,
                        0,
                    )
                    for object_slot in (2 * wave_index, 2 * wave_index + 1)
                )
            else:
                dependency_count = 1
                direct_dependency_count = 1
                requirements = (direct_requirement(runtime.device_view, 1),)
            dependencies.extend(requirements)
            self._compiled_completion_dependencies.extend(requirements)
            work.append(
                WorkItem(
                    request_index,
                    slot,
                    request.generation,
                    logical_work,
                    local_dependency_begin,
                    dependency_count,
                    direct_dependency_count,
                    work_ticket,
                    reduction_group_begin + request_index,
                    contributor_indices[request_index],
                    counts[request_index],
                    0,
                    0,
                    0,
                    0,
                    0,
                )
            )
            self._compiled_completion_work.append(
                WorkItem(
                    reduction_group_begin + request_index,
                    slot,
                    request.generation,
                    logical_work,
                    completion_dependency_begin,
                    dependency_count,
                    direct_dependency_count,
                    work_ticket,
                    reduction_group_begin + request_index,
                    contributor_indices[request_index],
                    counts[request_index],
                    0,
                    0,
                    0,
                    0,
                    0,
                )
            )
            contributor_indices[request_index] += 1
        request_ranges: list[RequestRange] = []
        cursor = 0
        for request_index, count in enumerate(counts):
            slot = request_slots[request_index]
            request_ranges.append(
                RequestRange(
                    cursor, count, slot, self.schedule.requests[slot].generation
                )
            )
            self._compiled_completion_ranges.append(
                RequestRange(
                    work_ticket_begin + cursor,
                    count,
                    slot,
                    self.schedule.requests[slot].generation,
                )
            )
            cursor += count
        if tuple(schedule.request_indices) != tuple(
            request_index
            for request_index, count in enumerate(counts)
            for _ in range(count)
        ):
            raise RuntimeError("compiled FlashInfer work is not request-contiguous")
        plan = DeviceWorkPlan(
            schedule.work_count, len(dependencies), runtime.device_ordinal
        )
        plan.upload(work, dependencies, request_ranges)
        plan.synchronize_upload()
        self._compiled_work_count += schedule.work_count
        self._compiled_reduction_group_count += len(request_slots)
        return _CompiledWorkPlan(
            plan,
            work_ticket_begin,
            reduction_group_begin,
        )

    def _enable_gpu_initiated_host(self) -> None:
        runtime = self._compiled_runtime
        programs = self._compiler_programs
        if runtime is None or programs is None:
            raise RuntimeError("GPU-initiated host acquisition has no runtime")
        objects: list[IndexedHostObject] = []
        for wave_index, wave in enumerate(self.executor.waves):
            slot = wave_index % self.executor.slot_count
            indices = torch.arange(
                wave.token_count,
                dtype=torch.int32,
                device=self.workspace.device,
            )
            self._compiled_host_indices.append(indices)
            for object_slot, source, staging in (
                (2 * wave_index, wave.key, self.executor.staging_key[slot]),
                (2 * wave_index + 1, wave.value, self.executor.staging_value[slot]),
            ):
                element_bytes = source[0].numel() * source.element_size()
                objects.append(
                    IndexedHostObject(
                        0x4E54414800000000 | object_slot,
                        1,
                        source.data_ptr(),
                        staging.data_ptr(),
                        indices.data_ptr(),
                        indices.data_ptr(),
                        wave.token_count,
                        element_bytes,
                        source.stride(0) * source.element_size(),
                        staging.stride(0) * staging.element_size(),
                        wave.token_count,
                        int(staging.shape[0]),
                    )
                )
        runtime.register_indexed_host_objects(0, objects)
        programs[1].validate_indexed_host_range(
            runtime,
            0,
            self._compiled_object_count,
            self.executor.compute_stream,
        )
        self.executor.compute_stream.synchronize()
        self.executor.transfer_wave = self._progress_gpu_host_wave

    def _progress_gpu_host_wave(
        self, wave_index: int, stream: torch.cuda.Stream
    ) -> None:
        runtime = self._compiled_runtime
        programs = self._compiler_programs
        if runtime is None or programs is None:
            raise RuntimeError("GPU-initiated host wave has no runtime")
        programs[1].progress_indexed_host_range(
            runtime,
            2 * wave_index,
            2,
            stream,
        )

    def _load_compiler_pair(self) -> None:
        workspace = pathlib.Path(os.environ["FLASHINFER_WORKSPACE_BASE"])
        programs: list[JitPhaseProgram] = []
        try:
            for name in self._compiler_module_names:
                modules = sorted(workspace.rglob(f"{name}.so"))
                if len(modules) != 1:
                    raise RuntimeError(
                        f"expected one compiled FlashInfer module {name}.so; "
                        f"found {len(modules)}"
                    )
                programs.append(JitPhaseProgram(modules[0]))
            direct, incremental = programs
            direct.operator_contract.require(
                family=OperatorFamily.FLASHINFER_PAGED_PREFILL,
                form=OperatorForm.DIRECT,
                capabilities=(
                    OperatorCapability.REQUEST_BINDING
                    | OperatorCapability.GRAPH_REPLAY
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
            incremental.operator_contract.require(
                family=OperatorFamily.FLASHINFER_PAGED_PREFILL,
                form=OperatorForm.INCREMENTAL,
                capabilities=(
                    OperatorCapability.REQUEST_BINDING
                    | OperatorCapability.OBJECT_DEPENDENCIES
                    | OperatorCapability.FINITE_DEFERRAL
                    | OperatorCapability.PARTIAL_PUBLICATION
                    | OperatorCapability.COMPLETE_CONTRIBUTOR_MERGE
                    | OperatorCapability.RUNNABLE_COMPACTION
                    | OperatorCapability.GRAPH_REPLAY
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
            plan = require_operator_pair(direct, incremental)
            plan.require(
                family=OperatorFamily.FLASHINFER_PAGED_PREFILL,
                forms=(OperatorForm.DIRECT, OperatorForm.INCREMENTAL),
                coordinate_map=(OperatorCoordinateMap.FLASHINFER_REQUEST_CONTIGUOUS),
                partial_state=OperatorPartialState.ONLINE_SOFTMAX_VALUE_LSE,
                reduction=OperatorReduction.ORDERED_MERGE_STATE,
                flags=_COMPILED_PLAN_FLAGS,
            )
        except Exception:
            for program in programs:
                program.close()
            raise
        self._compiler_programs = (direct, incremental)
        self._operator_plan = plan

    def _build_resident_runs(self) -> tuple[_ResidentRun, ...]:
        runs: list[_ResidentRun] = []
        begin = 0
        while begin < len(self.resident_lengths):
            while (
                begin < len(self.resident_lengths) and self.resident_lengths[begin] == 0
            ):
                begin += 1
            if begin == len(self.resident_lengths):
                break
            end = begin + 1
            while end < len(self.resident_lengths) and self.resident_lengths[end] > 0:
                end += 1
            runs.append(
                _ResidentRun(
                    self.query_offsets[begin],
                    self.query_offsets[end],
                    self.resident_offsets[begin],
                    self.resident_offsets[end],
                    self._wrapper(
                        self.query_lengths[begin:end],
                        self.resident_lengths[begin:end],
                        causal=False,
                        request_slots=tuple(range(begin, end)),
                        form=OperatorForm.DIRECT,
                    ),
                )
            )
            begin = end
        return tuple(runs)

    def _build_host_waves(
        self, external_key: torch.Tensor, external_value: torch.Tensor
    ) -> tuple[FlashInferHostWave, ...]:
        waves: list[FlashInferHostWave] = []
        shape = (self.kv_heads, self.head_dim)
        for wave in self.schedule.waves:
            key = torch.empty(
                (wave.token_count, *shape), dtype=self.dtype, pin_memory=True
            )
            value = torch.empty_like(key, pin_memory=True)
            destination = 0
            for segment in wave.segments:
                source = (
                    self.external_offsets[segment.request_index]
                    + segment.source_token_offset
                )
                count = segment.token_count
                key[destination : destination + count].copy_(
                    external_key[source : source + count]
                )
                value[destination : destination + count].copy_(
                    external_value[source : source + count]
                )
                destination += count
            lengths = tuple(segment.token_count for segment in wave.segments)
            waves.append(
                FlashInferHostWave(
                    key,
                    value,
                    self.query_offsets[wave.active_request_count],
                    self._wrapper(
                        self.query_lengths[: wave.active_request_count],
                        lengths,
                        causal=False,
                        request_slots=tuple(range(wave.active_request_count)),
                        form=OperatorForm.INCREMENTAL,
                    ),
                )
            )
        return tuple(waves)

    def bind_external(
        self, external_key: torch.Tensor, external_value: torch.Tensor
    ) -> None:
        """Refill stable pinned wave buffers for a later graph replay."""

        if (
            external_key.device.type != "cpu"
            or external_value.device.type != "cpu"
            or not external_key.is_pinned()
            or not external_value.is_pinned()
            or external_key.shape != external_value.shape
            or external_key.shape[0] != self.schedule.external_tokens
            or external_key.shape[1:] != (self.kv_heads, self.head_dim)
            or external_key.dtype != self.dtype
            or external_value.dtype != self.dtype
        ):
            raise ValueError("external KV does not match the captured operator plan")
        for planned, wave in zip(self.schedule.waves, self.executor.waves, strict=True):
            destination = 0
            for segment in planned.segments:
                source = (
                    self.external_offsets[segment.request_index]
                    + segment.source_token_offset
                )
                count = segment.token_count
                wave.key[destination : destination + count].copy_(
                    external_key[source : source + count]
                )
                wave.value[destination : destination + count].copy_(
                    external_value[source : source + count]
                )
                destination += count

    def _run_wrapper(
        self,
        wrapper: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        lse: torch.Tensor,
    ) -> None:
        runtime = self._compiled_runtime
        if runtime is None:
            wrapper.run(
                query,
                key,
                value,
                out=output,
                lse=lse,
                return_lse=True,
            )
            return
        form = self._wrapper_forms.get(id(wrapper))
        if form is None:
            raise RuntimeError("compiled FlashInfer wrapper has no typed form")
        runtime_tensor = runtime.device_view_tensor
        if form == OperatorForm.DIRECT:
            request_slots = self._wrapper_request_slots.get(id(wrapper))
            if request_slots is None or request_slots != tuple(
                range(request_slots[0], request_slots[0] + len(request_slots))
            ):
                raise RuntimeError(
                    "direct FlashInfer wrapper requires contiguous request slots"
                )
            wrapper.run(
                query,
                key,
                value,
                runtime_tensor,
                1.0 / self.head_dim**0.5,
                request_slots[0],
                out=output,
                lse=lse,
                return_lse=True,
            )
            return
        else:
            compiled = self._compiled_plans.get(id(wrapper))
            if compiled is None:
                raise RuntimeError("incremental FlashInfer wrapper has no work plan")
            if self._compiler_programs is None:
                raise RuntimeError(
                    "incremental FlashInfer wrapper has no phase program"
                )
            work_items = compiled.plan.work_items_tensor
            dependencies = compiled.plan.dependencies_tensor
            work_count = compiled.plan.work_item_count
            flags = BIND_CURRENT_GENERATION | (compiled.reduction_group_begin << 32)
        request_count = self._wrapper_request_counts.get(id(wrapper))
        if request_count is None:
            raise RuntimeError("compiled FlashInfer wrapper has no request count")
        wrapper.run(
            query,
            key,
            value,
            runtime_tensor,
            work_items,
            dependencies,
            1.0 / self.head_dim**0.5,
            pack_work_metadata(work_count, request_count),
            flags,
            out=output,
            lse=lse,
            return_lse=True,
        )
        if form == OperatorForm.INCREMENTAL:
            compiled.plan.mark_consumed(self.executor.compute_stream)

    def enqueue_base(
        self,
        query: torch.Tensor,
        resident_key: torch.Tensor,
        resident_value: torch.Tensor,
        local_key: torch.Tensor,
        local_value: torch.Tensor,
        output: torch.Tensor,
        lse: torch.Tensor,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
    ) -> None:
        """Enqueue every resident and local contributor in fixed merge order."""

        if (
            resident_key.shape != resident_value.shape
            or local_key.shape != local_value.shape
        ):
            raise ValueError("FlashInfer resident/local K/V geometry must match")
        if resident_key.shape[0] != self.resident_offsets[-1]:
            raise ValueError("resident KV disagrees with the operator plan")
        if local_key.shape[0] != self.query_offsets[-1]:
            raise ValueError("local KV disagrees with the operator plan")
        self._run_wrapper(
            self._local_wrapper,
            query,
            local_key,
            local_value,
            output,
            lse,
        )
        for run in self._resident_runs:
            self._run_wrapper(
                run.wrapper,
                query[run.query_begin : run.query_end],
                resident_key[run.kv_begin : run.kv_end],
                resident_value[run.kv_begin : run.kv_end],
                partial_output[run.query_begin : run.query_end],
                partial_lse[run.query_begin : run.query_end],
            )
            flashinfer.merge_state_in_place(
                output[run.query_begin : run.query_end],
                lse[run.query_begin : run.query_end],
                partial_output[run.query_begin : run.query_end],
                partial_lse[run.query_begin : run.query_end],
            )

    def run(
        self,
        query: torch.Tensor,
        resident_key: torch.Tensor,
        resident_value: torch.Tensor,
        local_key: torch.Tensor,
        local_value: torch.Tensor,
        output: torch.Tensor,
        lse: torch.Tensor,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        *,
        completion_events: dict[RequestKey, torch.cuda.Event] | None = None,
    ) -> None:
        """Enqueue the complete finite incremental operator."""

        if self._compiler_programs is not None:
            if self._compiled_runtime is None or self._compiled_work_count <= 0:
                raise RuntimeError("compiled FlashInfer operator has no finite epoch")
            self._compiler_programs[1].reset(
                self._compiled_runtime,
                self._compiled_object_count,
                self._compiled_work_count,
                self.executor.compute_stream,
            )
            if self._gpu_initiated_host:
                if self._compiled_completion_plan is None:
                    raise RuntimeError(
                        "GPU-initiated host acquisition has no discovery plan"
                    )
                self._compiler_programs[1].invalidate_cached_objects(
                    self._compiled_runtime,
                    0,
                    self._compiled_object_count,
                    self.executor.compute_stream,
                )
                self._compiler_programs[1].discover(
                    self._compiled_runtime,
                    self._compiled_completion_plan,
                    self.executor.compute_stream,
                )
            self._last_compiled_work_count = self._compiled_work_count

        self.executor.run(
            query,
            output,
            lse,
            partial_output,
            partial_lse,
            run_base=lambda: self.enqueue_base(
                query,
                resident_key,
                resident_value,
                local_key,
                local_value,
                output,
                lse,
                partial_output,
                partial_lse,
            ),
            run_partial=lambda wave, wave_query, key, value, wave_output, wave_lse: (
                self._run_wrapper(
                    wave.wrapper,
                    wave_query,
                    key,
                    value,
                    wave_output,
                    wave_lse,
                )
            ),
            completion_events=completion_events,
        )
        if self._compiler_programs is not None:
            if self._compiled_runtime is None or self._compiled_completion_plan is None:
                raise RuntimeError(
                    "compiled FlashInfer operator has no completion plan"
                )
            self._compiler_programs[1].complete_stream_ordered(
                self._compiled_runtime,
                self._compiled_completion_plan,
                self.executor.compute_stream,
            )
            self._compiled_completion_plan.mark_consumed(self.executor.compute_stream)

    def capture(
        self,
        query: torch.Tensor,
        resident_key: torch.Tensor,
        resident_value: torch.Tensor,
        local_key: torch.Tensor,
        local_value: torch.Tensor,
        output: torch.Tensor,
        lse: torch.Tensor,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
    ) -> FlashInferTierStreamingGraph:
        """Capture the fixed finite operator after all structural planning."""

        graph = torch.cuda.CUDAGraph()
        self.executor.compute_stream.synchronize()
        original_stream = self.executor.compute_stream
        original_events = (
            self.executor._start,
            self.executor._ready,
            self.executor._slot_free,
        )
        original_serialization = self.executor._serialize_eager_runs
        capture_start = torch.cuda.Event()
        capture_ready = [torch.cuda.Event() for _ in self.executor.waves]
        capture_slot_free = [torch.cuda.Event() for _ in self.executor.waves]
        self.executor._start = capture_start
        self.executor._ready = capture_ready
        self.executor._slot_free = capture_slot_free
        self.executor._serialize_eager_runs = False
        try:
            with torch.cuda.graph(graph):
                # PyTorch changes the current stream while entering capture.
                # Join the copy stream to that stream so transfer/event nodes,
                # rather than only the compute nodes, enter the graph.
                self.executor.compute_stream = torch.cuda.current_stream()
                self.run(
                    query,
                    resident_key,
                    resident_value,
                    local_key,
                    local_value,
                    output,
                    lse,
                    partial_output,
                    partial_lse,
                )
        finally:
            self.executor.compute_stream = original_stream
            (
                self.executor._start,
                self.executor._ready,
                self.executor._slot_free,
            ) = original_events
            self.executor._serialize_eager_runs = original_serialization
        return FlashInferTierStreamingGraph(
            graph,
            self,
            (
                query,
                resident_key,
                resident_value,
                local_key,
                local_value,
                output,
                lse,
                partial_output,
                partial_lse,
            ),
            (capture_start, *capture_ready, *capture_slot_free),
        )
