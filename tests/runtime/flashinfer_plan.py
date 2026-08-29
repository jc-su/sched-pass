#!/usr/bin/env python3
"""Validate zero-copy reuse of a stock FlashInfer launch plan."""

from __future__ import annotations

import nta_runtime.flashinfer as flashinfer_runtime
from nta_runtime.flashinfer import (
    BIND_CURRENT_GENERATION,
    DYNAMIC_RUNNABLE_WINDOW,
    FlashInferLayerEpoch,
    PREACQUIRED_LAUNCH_FLAGS,
    RUNNABLE_OFFSET_SHIFT,
    RUNNABLE_WORK,
    SKIP_MERGE,
    adopt_planned_flashinfer_state,
    enqueue_event_partitioned_attention,
)


class Module:
    def plan(self) -> None:
        return None

    def paged_run(self) -> None:
        return None


class Wrapper:
    def __init__(self, *, typed: bool, plan: object | None) -> None:
        self._kv_layout = "NHD"
        self._backend = "fa2"
        self.device = "cuda:0"
        self._jit_module = Module() if typed else None
        self._cached_module = self._jit_module or Module()
        self._jit_additional_tensor_names = (
            ["nta_runtime", "nta_work_items"] if typed else []
        )
        self._plan_info = plan
        self._int_workspace_buffer = object()
        self._pin_memory_int_workspace_buffer = object()
        self._kv_lens_buffer = object()
        self._paged_kv_indices_buf = object()
        self._batch_size = 7


class _Event:
    def __init__(self) -> None:
        self.records: list[object] = []

    def record(self, stream: object) -> None:
        self.records.append(stream)


class _Stream:
    def __init__(self, address: int) -> None:
        self.cuda_stream = address
        self.waits: list[_Event] = []

    def wait_event(self, event: _Event) -> None:
        self.waits.append(event)


class _Runtime:
    device_ordinal = 0
    device_view_tensor = object()

    def __init__(self) -> None:
        self.object_waits: list[tuple[int, object]] = []

    def wait_object_range_terminal(
        self, object_slot: int, object_count: int, stream: object
    ) -> None:
        assert object_count == 1
        self.object_waits.append((object_slot, stream))


class _Plan:
    device_ordinal = 0
    work_item_count = 4
    has_external = True
    work_items_tensor = object()
    dependencies_tensor = object()

    def __init__(self) -> None:
        self.consumed: list[object] = []

    def mark_consumed(self, stream: object) -> None:
        self.consumed.append(stream)


class _Phases:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def reset(
        self, runtime: object, object_count: int, work_count: int, stream: object
    ) -> None:
        self.calls.append(("reset", object_count, work_count, stream))

    def discover(self, runtime: object, plan: object, stream: object) -> None:
        self.calls.append(("discover", stream))

    def prepare_ready_window(
        self, runtime: object, maximum_work: int, stream: object
    ) -> None:
        self.calls.append(("prepare", maximum_work, stream))

    def prepare_event_work_partition(
        self,
        runtime: object,
        plan: object,
        direct_work_count: int,
        stream: object,
    ) -> None:
        self.calls.append(("event_partition", plan, direct_work_count, stream))

    def progress_validated_indexed_host_range_parallel(
        self,
        runtime: object,
        first_object: int,
        object_count: int,
        copy_blocks: int,
        stream: object,
    ) -> None:
        self.calls.append(
            ("progress", first_object, object_count, copy_blocks, stream)
        )

    def complete_stream_ordered(
        self, runtime: object, plan: object, stream: object
    ) -> None:
        self.calls.append(("retire", stream))


class _NumericalWrapper:
    def __init__(self) -> None:
        self.launches: list[tuple[int, int]] = []

    def run(
        self,
        _q: object,
        _kv: object,
        _runtime: object,
        _work_items: object,
        _dependencies: object,
        _scale: float,
        work_count: int,
        flags: int,
        *,
        out: object,
        **_options: object,
    ) -> None:
        self.launches.append((work_count, flags))


def _check_stream_ordered_multiwave() -> None:
    runtime = _Runtime()
    plan = _Plan()
    phases = _Phases()
    wrapper = _NumericalWrapper()
    compute = _Stream(1)
    progress = _Stream(2)
    discovery = _Event()
    discovery_profile = (_Event(), _Event())
    arrivals = (_Event(), _Event())
    epoch = FlashInferLayerEpoch(
        runtime,  # type: ignore[arg-type]
        plan,  # type: ignore[arg-type]
        phases,  # type: ignore[arg-type]
        object_count=4,
        max_progress_rounds=2,
        wait_for_plan=False,
        stream_ordered_retirement=True,
    )
    original_capture = flashinfer_runtime._stream_is_capturing
    flashinfer_runtime._stream_is_capturing = lambda: False
    try:
        rounds = epoch.enqueue_host(
            wrapper,
            object(),
            object(),
            object(),
            progress_blocks=(2, 2),
            ready_work_counts=(2, 2),
            ready_work_offsets=(0, 2),
            sm_scale=1.0,
            indexed_host_first_object=0,
            indexed_host_prevalidated=True,
            indexed_host_copy_blocks_per_group=4,
            stream=compute,
            progress_stream=progress,
            sync_events=(discovery, arrivals),
            discovery_profile=discovery_profile,
            complete_stream_ordered=False,
        )
        assert rounds == 2
        assert wrapper.launches == [
            (2, BIND_CURRENT_GENERATION | RUNNABLE_WORK | SKIP_MERGE),
            (
                2,
                BIND_CURRENT_GENERATION
                | RUNNABLE_WORK
                | (2 << RUNNABLE_OFFSET_SHIFT),
            ),
        ]
        assert [call[0] for call in phases.calls] == [
            "reset",
            "discover",
            "progress",
            "progress",
        ]
        assert compute.waits == list(arrivals)
        assert discovery_profile[0].records == [compute]
        assert discovery_profile[1].records == [compute]
        epoch.retire_stream_ordered(compute)
        assert phases.calls[-1] == ("retire", compute)
        assert plan.consumed == [compute]

        dynamic_plan = _Plan()
        dynamic_phases = _Phases()
        dynamic_wrapper = _NumericalWrapper()
        dynamic = FlashInferLayerEpoch(
            runtime,  # type: ignore[arg-type]
            dynamic_plan,  # type: ignore[arg-type]
            dynamic_phases,  # type: ignore[arg-type]
            object_count=4,
            max_progress_rounds=2,
            wait_for_plan=False,
            stream_ordered_retirement=True,
        )
        dynamic.enqueue_host(
            dynamic_wrapper,
            object(),
            object(),
            object(),
            progress_blocks=(2, 2),
            ready_work_counts=(2, 4),
            sm_scale=1.0,
            indexed_host_first_object=0,
            indexed_host_prevalidated=True,
            stream=compute,
            progress_stream=progress,
            sync_events=(discovery, arrivals),
        )
        assert dynamic_wrapper.launches == [
            (
                2,
                BIND_CURRENT_GENERATION
                | RUNNABLE_WORK
                | DYNAMIC_RUNNABLE_WINDOW
                | SKIP_MERGE,
            ),
            (
                4,
                BIND_CURRENT_GENERATION
                | RUNNABLE_WORK
                | DYNAMIC_RUNNABLE_WINDOW,
            ),
        ]
        assert [call[0] for call in dynamic_phases.calls] == [
            "reset",
            "discover",
            "progress",
            "prepare",
            "progress",
            "prepare",
            "retire",
        ]

        invalid = FlashInferLayerEpoch(
            runtime,  # type: ignore[arg-type]
            _Plan(),  # type: ignore[arg-type]
            _Phases(),  # type: ignore[arg-type]
            object_count=4,
            max_progress_rounds=2,
            wait_for_plan=False,
            stream_ordered_retirement=True,
        )
        try:
            invalid.enqueue_host(
                _NumericalWrapper(),
                object(),
                object(),
                object(),
                progress_blocks=(2, 2),
                ready_work_counts=(2, 2),
                ready_work_offsets=(0, 0),
                sm_scale=1.0,
                indexed_host_first_object=0,
                indexed_host_prevalidated=True,
                stream=compute,
                progress_stream=progress,
                sync_events=(discovery, arrivals),
            )
        except ValueError as error:
            assert "runnable launch windows" in str(error)
        else:
            raise AssertionError("stream-ordered execution accepted overlapping waves")

        event_plan = _Plan()
        event_phases = _Phases()
        event_wrapper = _NumericalWrapper()
        ready = (_Event(), _Event())
        enqueue_event_partitioned_attention(
            runtime,  # type: ignore[arg-type]
            event_plan,  # type: ignore[arg-type]
            event_phases,  # type: ignore[arg-type]
            event_wrapper,
            object(),
            object(),
            object(),
            ready_events=ready,
            ready_object_slots=(),
            registration_event=None,
            direct_work_count=2,
            wave_work_counts=(1, 1),
            prepare_partition=True,
            sm_scale=1.0,
            stream=compute,
        )
        assert event_phases.calls == [
            ("event_partition", event_plan, 2, compute)
        ]
        assert event_wrapper.launches == [
            (
                2,
                PREACQUIRED_LAUNCH_FLAGS | RUNNABLE_WORK | SKIP_MERGE,
            ),
            (
                1,
                PREACQUIRED_LAUNCH_FLAGS
                | RUNNABLE_WORK
                | SKIP_MERGE
                | (2 << RUNNABLE_OFFSET_SHIFT),
            ),
            (
                1,
                PREACQUIRED_LAUNCH_FLAGS
                | RUNNABLE_WORK
                | (3 << RUNNABLE_OFFSET_SHIFT),
            ),
        ]
        assert compute.waits[-2:] == list(ready)
        assert event_plan.consumed == [compute]

        object_plan = _Plan()
        object_phases = _Phases()
        object_wrapper = _NumericalWrapper()
        registration = _Event()
        enqueue_event_partitioned_attention(
            runtime,  # type: ignore[arg-type]
            object_plan,  # type: ignore[arg-type]
            object_phases,  # type: ignore[arg-type]
            object_wrapper,
            object(),
            object(),
            object(),
            ready_events=(),
            ready_object_slots=(10, 12),
            registration_event=registration,
            direct_work_count=2,
            wave_work_counts=(0, 2),
            prepare_partition=True,
            sm_scale=1.0,
            stream=compute,
        )
        assert compute.waits[-1] is registration
        assert runtime.object_waits[-2:] == [(10, compute), (12, compute)]
        assert object_wrapper.launches == [
            (
                2,
                PREACQUIRED_LAUNCH_FLAGS | RUNNABLE_WORK | SKIP_MERGE,
            ),
            (
                2,
                PREACQUIRED_LAUNCH_FLAGS
                | RUNNABLE_WORK
                | (2 << RUNNABLE_OFFSET_SHIFT),
            ),
        ]
        assert object_plan.consumed == [compute]
    finally:
        flashinfer_runtime._stream_is_capturing = original_capture


def main() -> None:
    plan = object()
    source = Wrapper(typed=False, plan=plan)
    target = Wrapper(typed=True, plan=None)
    typed_module = target._jit_module
    owned_resources = (
        target._int_workspace_buffer,
        target._pin_memory_int_workspace_buffer,
        target._kv_lens_buffer,
    )
    adopt_planned_flashinfer_state(target, source)
    assert target._plan_info is plan
    assert target._int_workspace_buffer is source._int_workspace_buffer
    assert (
        target._pin_memory_int_workspace_buffer
        is source._pin_memory_int_workspace_buffer
    )
    assert target._kv_lens_buffer is source._kv_lens_buffer
    assert target._paged_kv_indices_buf is source._paged_kv_indices_buf
    assert target._cached_module is typed_module
    assert target._jit_module is typed_module
    assert target._jit_additional_tensor_names == [
        "nta_runtime",
        "nta_work_items",
    ]
    assert target._nta_owned_plan_resources == owned_resources

    # Re-adoption must retain the target's original allocations while moving
    # to the source's newest launch geometry.
    newer_plan = object()
    source._plan_info = newer_plan
    adopt_planned_flashinfer_state(target, source)
    assert target._plan_info is newer_plan
    assert target._nta_owned_plan_resources == owned_resources

    malformed = Wrapper(typed=False, plan=None)
    try:
        adopt_planned_flashinfer_state(target, malformed)
    except RuntimeError as error:
        assert "no completed plan" in str(error)
    else:
        raise AssertionError("FlashInfer accepted an unplanned source wrapper")

    _check_stream_ordered_multiwave()


if __name__ == "__main__":
    main()
