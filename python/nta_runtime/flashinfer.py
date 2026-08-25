"""Engine-neutral FlashInfer layer integration for NTA work plans."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from .epoch import BoundedEpoch, EpochResult
from .runtime import AcquireRequirement, DeviceWorkPlan, JitPhaseProgram, Runtime
from .runtime import RequestRange
from .requests import RequestBinding


TENSOR_NAMES = ["nta_runtime", "nta_work_items", "nta_dependencies"]
TENSOR_DTYPES = ["uint8_t", "uint8_t", "uint8_t"]
SCALAR_NAMES = ["sm_scale", "nta_work_count", "nta_skip_merge"]
RUNNABLE_OFFSET_SHIFT = 32
SCALAR_DTYPES = ["double", "int64_t", "int64_t"]
REQUEST_BOUND_TENSOR_NAMES = ["nta_runtime"]
REQUEST_BOUND_TENSOR_DTYPES = ["uint8_t"]
REQUEST_BOUND_SCALAR_NAMES = ["sm_scale", "nta_request_slot_offset"]
REQUEST_BOUND_SCALAR_DTYPES = ["double", "int64_t"]

SKIP_MERGE = 1 << 0
PREACQUIRED = 1 << 1
BIND_CURRENT_GENERATION = 1 << 2
PLANLESS_PREACQUIRED = 1 << 3
RUNNABLE_WORK = 1 << 4
WORK_COUNT_MASK = (1 << 32) - 1

_DEFAULT_ATTENTION_VARIANT = "DefaultAttention<false, false, false, false>"
_DEFAULT_ATTENTION_DECL = "#include <flashinfer/attention/variants.cuh>"


def _current_cuda_stream() -> Any:
    import torch

    return torch.cuda.current_stream()


def _stream_is_capturing() -> bool:
    """Return whether the current stream is inside CUDA graph capture.

    Work-plan consumer fences are host-owned lifetime state. Recording one
    into a graph would make the event a capture-time dependency and leave the
    reusable plan with an event whose lifetime no longer matches graph replay.
    Graph callers publish the fence immediately after replay instead.
    """

    import torch

    return bool(torch.cuda.is_current_stream_capturing())


def direct_requirement(
    direct_base: int,
    bytes: int,
    *,
    direct_tensor_map: int = 0,
) -> AcquireRequirement:
    """Build an exact direct dependency with no object identity payload.

    A direct dependency is already resident in the operator-owned address
    space.  Its object ID, slot, and version are intentionally zero: those
    fields are meaningful only for transport-backed requirements.  Keeping
    construction here prevents framework adapters from inventing magic IDs
    that look like ownership or generation metadata.
    """
    if (
        direct_base <= 0
        or bytes <= 0
        or bytes > (1 << 32) - 1
        or direct_tensor_map < 0
    ):
        raise ValueError("direct dependencies need positive addresses and bytes")
    return AcquireRequirement(
        int(direct_base),
        int(direct_tensor_map),
        0,
        0,
        0,
        0,
        int(bytes),
        0,
    )


def request_ranges_for_schedule(
    bindings: Sequence[RequestBinding], request_indices: Sequence[int]
) -> list[RequestRange]:
    """Build the native contiguous request ranges for a CTA schedule.

    FlashInfer emits contributors grouped by request.  Keeping this check in
    the engine-neutral FlashInfer boundary makes SGLang and vLLM share the
    same O(work-items) validation and prevents either adapter from silently
    assigning a contributor to the wrong request generation.
    """
    ranges: list[RequestRange] = []
    cursor = 0
    for binding in bindings:
        begin = cursor
        while (
            cursor < len(request_indices)
            and int(request_indices[cursor]) == binding.request_index
        ):
            cursor += 1
        if cursor == begin:
            raise RuntimeError(
                f"FlashInfer schedule has no work for request "
                f"{binding.request_index}"
            )
        ranges.append(
            RequestRange(
                begin,
                cursor - begin,
                binding.request_slot,
                binding.generation,
            )
        )
    if cursor != len(request_indices):
        raise RuntimeError(
            "FlashInfer schedule is not grouped contiguously by request"
        )
    return ranges


def pack_work_metadata(work_count: int, request_count: int) -> int:
    if not 0 < work_count <= WORK_COUNT_MASK:
        raise ValueError("FlashInfer work count exceeds packed metadata")
    if not 0 < request_count <= WORK_COUNT_MASK:
        raise ValueError("FlashInfer request count exceeds packed metadata")
    return work_count | (request_count << 32)


def attention_jit_args(
    module_name: str,
    *,
    dtype_q: Any,
    dtype_kv: Any,
    dtype_o: Any,
    idtype: Any,
    head_dim_qk: int,
    head_dim_vo: int,
) -> list[Any]:
    """Build FlashInfer custom-module arguments for an instrumented wrapper."""
    if not module_name or min(head_dim_qk, head_dim_vo) <= 0:
        raise ValueError("FlashInfer module name and head dimensions are required")
    return [
        module_name,
        dtype_q,
        dtype_kv,
        dtype_o,
        idtype,
        head_dim_qk,
        head_dim_vo,
        TENSOR_NAMES,
        TENSOR_DTYPES,
        SCALAR_NAMES,
        SCALAR_DTYPES,
        _DEFAULT_ATTENTION_VARIANT,
        _DEFAULT_ATTENTION_DECL,
    ]


def request_bound_attention_jit_args(
    module_name: str,
    *,
    dtype_q: Any,
    dtype_kv: Any,
    dtype_o: Any,
    idtype: Any,
    head_dim_qk: int,
    head_dim_vo: int,
) -> list[Any]:
    """Build the minimal typed arguments for a request-bound direct module."""
    if not module_name or min(head_dim_qk, head_dim_vo) <= 0:
        raise ValueError("FlashInfer module name and head dimensions are required")
    return [
        module_name,
        dtype_q,
        dtype_kv,
        dtype_o,
        idtype,
        head_dim_qk,
        head_dim_vo,
        REQUEST_BOUND_TENSOR_NAMES,
        REQUEST_BOUND_TENSOR_DTYPES,
        REQUEST_BOUND_SCALAR_NAMES,
        REQUEST_BOUND_SCALAR_DTYPES,
        _DEFAULT_ATTENTION_VARIANT,
        _DEFAULT_ATTENTION_DECL,
    ]


def enqueue_resident_attention(
    runtime: Runtime,
    plan: DeviceWorkPlan,
    wrapper: Any,
    q: Any,
    paged_kv_cache: Any,
    out: Any,
    *,
    sm_scale: float | None = None,
    run_options: dict[str, Any] | None = None,
) -> None:
    """Enqueue an all-direct plan already ordered on the consumer stream."""
    if runtime.device_ordinal != plan.device_ordinal:
        raise ValueError("runtime and work plan must own the same CUDA device")
    if plan.work_item_count <= 0 or plan.has_external:
        raise ValueError("resident attention requires a non-empty all-direct plan")
    scale = 1.0 / math.sqrt(q.shape[-1]) if sm_scale is None else sm_scale
    options = {} if run_options is None else run_options
    wrapper.run(
        q,
        paged_kv_cache,
        runtime.device_view_tensor,
        plan.work_items_tensor,
        plan.dependencies_tensor,
        scale,
        plan.work_item_count,
        0,
        out=out,
        **options,
    )
    if not _stream_is_capturing():
        plan.mark_consumed(_current_cuda_stream())


class FlashInferLayerEpoch:
    """Bind one uploaded work plan to decode or paged-prefill launches."""

    def __init__(
        self,
        runtime: Runtime,
        plan: DeviceWorkPlan,
        phases: JitPhaseProgram,
        *,
        object_count: int,
        max_progress_rounds: int,
        wait_for_plan: bool = True,
    ) -> None:
        if runtime.device_ordinal != plan.device_ordinal:
            raise ValueError("runtime and work plan must own the same CUDA device")
        if plan.work_item_count <= 0:
            raise ValueError("FlashInfer layer epoch needs an uploaded work plan")
        self.runtime = runtime
        self.plan = plan
        self.epoch = BoundedEpoch(
            phases,
            runtime,
            object_count=object_count,
            work_ticket_count=plan.work_item_count,
            max_progress_rounds=max_progress_rounds,
        )
        self._runtime_tensor = runtime.device_view_tensor
        self._work_items_tensor = plan.work_items_tensor
        self._dependencies_tensor = plan.dependencies_tensor
        self._wait_for_plan = wait_for_plan

    def _prepare(self, stream: Any) -> None:
        if self.plan.work_item_count <= 0:
            raise ValueError("FlashInfer layer epoch needs an uploaded work plan")
        self.epoch.work_ticket_count = self.plan.work_item_count
        if self._wait_for_plan and not _stream_is_capturing():
            self.plan.wait_on(stream)

    def _launch(
        self,
        wrapper: Any,
        q: Any,
        paged_kv_cache: Any,
        out: Any,
        sm_scale: float,
        launch_flags: int,
        launch_work_count: int | None = None,
        run_options: dict[str, Any] | None = None,
    ) -> None:
        work_count = (
            self.plan.work_item_count
            if launch_work_count is None
            else int(launch_work_count)
        )
        if work_count <= 0 or work_count > self.plan.work_item_count:
            raise ValueError("FlashInfer launch work count is outside the active plan")
        options = {} if run_options is None else run_options
        wrapper.run(
            q,
            paged_kv_cache,
            self._runtime_tensor,
            self._work_items_tensor,
            self._dependencies_tensor,
            sm_scale,
            work_count,
            launch_flags,
            out=out,
            **options,
        )
        if not _stream_is_capturing():
            self.plan.mark_consumed(_current_cuda_stream())

    def mark_consumed_after_replay(self, stream: Any) -> None:
        """Publish the plan fence after a graph replay has completed enqueue."""

        self.plan.mark_consumed(stream)

    def run_host(
        self,
        wrapper: Any,
        q: Any,
        paged_kv_cache: Any,
        out: Any,
        *,
        progress_blocks: int,
        sm_scale: float | None = None,
        stream: Any = None,
        run_options: dict[str, Any] | None = None,
    ) -> EpochResult:
        passes = self.enqueue_host(
            wrapper,
            q,
            paged_kv_cache,
            out,
            progress_blocks=progress_blocks,
            sm_scale=sm_scale,
            stream=stream,
            run_options=run_options,
        )
        return self.epoch.check(passes, stream)

    def enqueue_resident(
        self,
        wrapper: Any,
        q: Any,
        paged_kv_cache: Any,
        out: Any,
        *,
        sm_scale: float | None = None,
        stream: Any = None,
        run_options: dict[str, Any] | None = None,
    ) -> None:
        """Launch an all-direct plan without transport phase kernels."""
        if self.plan.has_external:
            raise ValueError("resident launch cannot contain external dependencies")
        scale = 1.0 / math.sqrt(q.shape[-1]) if sm_scale is None else sm_scale
        self._prepare(stream)
        self._launch(
            wrapper,
            q,
            paged_kv_cache,
            out,
            scale,
            False,
            None,
            run_options,
        )

    def enqueue_host(
        self,
        wrapper: Any,
        q: Any,
        paged_kv_cache: Any,
        out: Any,
        *,
        progress_blocks: int | tuple[int, ...],
        sm_scale: float | None = None,
        stream: Any = None,
        progress_stream: Any = None,
        ready_event: Any = None,
        ready_work_counts: int | tuple[int, ...] | None = None,
        initial_ready_work_count: int = 0,
        indexed_host_first_object: int | None = None,
        indexed_host_prevalidated: bool = False,
        indexed_host_copy_blocks_per_group: int = 2,
        sync_events: tuple[Any, tuple[Any, ...]] | None = None,
        progress_profile: tuple[Any, Any] | None = None,
        on_discovered: Callable[[Any], None] | None = None,
        run_options: dict[str, Any] | None = None,
    ) -> int:
        """Enqueue a fixed host epoch; call ``check`` after execution."""
        if isinstance(progress_blocks, int):
            if progress_blocks <= 0:
                raise ValueError("host progress block count must be positive")
            block_counts = (progress_blocks,) * self.epoch.max_progress_rounds
        else:
            block_counts = tuple(int(count) for count in progress_blocks)
            if len(block_counts) != self.epoch.max_progress_rounds or any(
                count <= 0 for count in block_counts
            ):
                raise ValueError(
                    "host progress rounds must match the finite epoch bound"
                )
        if ready_work_counts is None:
            launch_counts = (self.plan.work_item_count,) * len(block_counts)
        elif isinstance(ready_work_counts, int):
            launch_counts = (int(ready_work_counts),) * len(block_counts)
        else:
            launch_counts = tuple(int(count) for count in ready_work_counts)
        if (
            len(launch_counts) != len(block_counts)
            or any(
                count <= 0 or count > self.plan.work_item_count
                for count in launch_counts
            )
            or any(
                current < previous
                for previous, current in zip(launch_counts, launch_counts[1:])
            )
        ):
            raise ValueError("runnable launch bounds must be monotonic plan counts")
        initial_ready_work_count = int(initial_ready_work_count)
        if not 0 <= initial_ready_work_count <= self.plan.work_item_count:
            raise ValueError("initial runnable work count is outside the active plan")
        next_indexed_object = (
            None
            if indexed_host_first_object is None
            else int(indexed_host_first_object)
        )
        if next_indexed_object is not None and next_indexed_object < 0:
            raise ValueError("indexed host object offset must be nonnegative")
        indexed_host_copy_blocks_per_group = int(indexed_host_copy_blocks_per_group)
        if not 1 <= indexed_host_copy_blocks_per_group <= 64:
            raise ValueError(
                "indexed host copy blocks per group must be between 1 and 64"
            )

        def progress(blocks: int, target_stream: Any) -> None:
            nonlocal next_indexed_object
            if next_indexed_object is None:
                self.epoch.phases.progress_host(self.runtime, blocks, target_stream)
                return
            if indexed_host_prevalidated:
                self.epoch.phases.progress_validated_indexed_host_range_parallel(
                    self.runtime,
                    next_indexed_object,
                    blocks,
                    indexed_host_copy_blocks_per_group,
                    target_stream,
                )
            else:
                self.epoch.phases.progress_indexed_host_range(
                    self.runtime, next_indexed_object, blocks, target_stream
                )
            next_indexed_object += blocks

        if any(
            count > self.plan.work_item_count - initial_ready_work_count
            for count in launch_counts
        ):
            raise ValueError("resume launch bound exceeds work after initial fragment")
        if stream is None and progress_stream is not None:
            import torch

            stream = torch.cuda.current_stream()
        scale = 1.0 / math.sqrt(q.shape[-1]) if sm_scale is None else sm_scale
        self._prepare(stream)
        has_external = self.plan.has_external
        if ready_event is not None:
            if stream is None:
                raise ValueError("preloaded host work requires an explicit CUDA stream")
            stream.wait_event(ready_event)

        def launch() -> None:
            # Discovery uses FlashInfer's canonical grid. Resume waves below
            # remap a bounded physical prefix through the device runnable set.
            self._launch(
                wrapper,
                q,
                paged_kv_cache,
                out,
                scale,
                BIND_CURRENT_GENERATION if has_external else 0,
                None,
                run_options,
            )

        if not has_external:
            self.epoch.phases.reset(
                self.runtime,
                self.epoch.object_count,
                self.epoch.work_ticket_count,
                stream,
            )
            launch()
            return 0

        def ready(progress_round: int, _final_round: bool) -> None:
            # The merge kernel is request-gated: completed requests publish now
            # while incomplete requests retain their split-K scratch state.
            self._launch(
                wrapper,
                q,
                paged_kv_cache,
                out,
                scale,
                BIND_CURRENT_GENERATION
                | RUNNABLE_WORK
                | (initial_ready_work_count << RUNNABLE_OFFSET_SHIFT),
                launch_counts[progress_round - 1],
                run_options,
            )

        self.epoch.phases.reset(
            self.runtime,
            self.epoch.object_count,
            self.epoch.work_ticket_count,
            stream,
        )
        stream_address = int(getattr(stream, "cuda_stream", stream or 0))
        progress_address = int(
            getattr(progress_stream, "cuda_stream", progress_stream or 0)
        )
        pipelined = progress_stream is not None and stream_address != progress_address
        if not pipelined:
            if progress_profile is not None:
                progress_profile[0].record(stream)
            launch()
            if on_discovered is not None:
                on_discovered(stream)
            for progress_round, blocks in enumerate(block_counts, 1):
                progress(blocks, stream)
                ready(progress_round, progress_round == len(block_counts))
            if progress_profile is not None:
                progress_profile[1].record(stream)
            return len(block_counts)

        import torch

        self.epoch.phases.discover(self.runtime, self.plan, stream)
        if on_discovered is not None:
            on_discovered(stream)
        if sync_events is None:
            discovery_done = torch.cuda.Event()
            arrival_events = tuple(torch.cuda.Event() for _ in block_counts)
        else:
            discovery_done, arrival_events = sync_events
            if len(arrival_events) != len(block_counts):
                raise ValueError(
                    "synchronization events must match the finite progress bound"
                )
        events: list[Any] = [discovery_done]
        discovery_done.record(stream)
        progress_stream.wait_event(discovery_done)
        if progress_profile is not None:
            progress_profile[0].record(progress_stream)
        if initial_ready_work_count:
            self._launch(
                wrapper,
                q,
                paged_kv_cache,
                out,
                scale,
                BIND_CURRENT_GENERATION | RUNNABLE_WORK,
                initial_ready_work_count,
                run_options,
            )
        for progress_round, blocks in enumerate(block_counts, 1):
            progress(blocks, progress_stream)
            arrival = arrival_events[progress_round - 1]
            arrival.record(progress_stream)
            stream.wait_event(arrival)
            ready(progress_round, progress_round == len(block_counts))
            events.append(arrival)
        if progress_profile is not None:
            progress_profile[1].record(progress_stream)
        # Retain event wrappers through at least the next call on this epoch.
        self._inflight_events = tuple(events)
        return self.epoch.max_progress_rounds

    def enqueue_preloaded_host(
        self,
        wrapper: Any,
        q: Any,
        paged_kv_cache: Any,
        out: Any,
        *,
        sm_scale: float | None = None,
        stream: Any = None,
        ready_event: Any = None,
        run_options: dict[str, Any] | None = None,
    ) -> None:
        """Consume host objects staged ahead of the application launch."""
        if not self.plan.has_external:
            raise ValueError("preloaded host launch needs external dependencies")
        scale = 1.0 / math.sqrt(q.shape[-1]) if sm_scale is None else sm_scale
        self._prepare(stream)
        if ready_event is not None:
            stream.wait_event(ready_event)
        self._launch(
            wrapper,
            q,
            paged_kv_cache,
            out,
            scale,
            6,
            None,
            run_options,
        )

    def run_nvme(
        self,
        wrapper: Any,
        q: Any,
        paged_kv_cache: Any,
        out: Any,
        *,
        issue_budget: int,
        completion_budget: int,
        timeout_ns: int = 100_000_000,
        sm_scale: float | None = None,
        stream: Any = None,
        run_options: dict[str, Any] | None = None,
    ) -> EpochResult:
        rounds = self.enqueue_nvme(
            wrapper,
            q,
            paged_kv_cache,
            out,
            issue_budget=issue_budget,
            completion_budget=completion_budget,
            timeout_ns=timeout_ns,
            sm_scale=sm_scale,
            stream=stream,
            run_options=run_options,
        )
        return self.epoch.check(rounds, stream)

    def enqueue_nvme(
        self,
        wrapper: Any,
        q: Any,
        paged_kv_cache: Any,
        out: Any,
        *,
        issue_budget: int,
        completion_budget: int,
        timeout_ns: int = 100_000_000,
        sm_scale: float | None = None,
        stream: Any = None,
        run_options: dict[str, Any] | None = None,
    ) -> int:
        """Enqueue a fixed NVMe epoch; call ``check`` after execution."""
        scale = 1.0 / math.sqrt(q.shape[-1]) if sm_scale is None else sm_scale
        self._prepare(stream)

        def initial() -> None:
            self._launch(
                wrapper,
                q,
                paged_kv_cache,
                out,
                scale,
                BIND_CURRENT_GENERATION,
                None,
                run_options,
            )

        def ready(_progress_round: int, _final_round: bool) -> None:
            self._launch(
                wrapper,
                q,
                paged_kv_cache,
                out,
                scale,
                BIND_CURRENT_GENERATION | RUNNABLE_WORK,
                None,
                run_options,
            )

        self.epoch.enqueue_nvme_fixed(
            initial,
            ready,
            issue_budget=issue_budget,
            completion_budget=completion_budget,
            timeout_ns=timeout_ns,
            stream=stream,
        )
        return self.epoch.max_progress_rounds

    def check(self, progress_rounds: int, stream: Any = None) -> EpochResult:
        return self.epoch.check(progress_rounds, stream)
