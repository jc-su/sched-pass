"""Owned host-to-HBM transport pipeline for the SGLang integration.

The framework backend decides *what* exact work is required.  This component
owns *how* already-planned host rows are submitted: its CUDA streams, reusable
layer-ready events, SM/copy-engine wave partition, and transfer profiling.
It does not own request identity, admission policy, numerical dispatch, or a
HiCache lease; those remain with the engine and the bridge respectively.
"""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any

import torch

from nta_runtime.engines.sglang_hicache import PendingHostLoad
from nta_runtime.engines.sglang_planning import pipeline_object_range
from nta_runtime.engines.sglang_state import _PrefetchedLayer
from nta_runtime.engines.sglang_transfer import (
    HostMoverController,
    HostTransferLeasePlan,
    MoverProfile,
)
from nta_runtime.runtime import copy_strided_host_runs_async


class SglangHostTransport:
    """Own finite host mover streams and layer-readiness publication."""

    def __init__(
        self,
        *,
        runtime: Any,
        host_movers: HostMoverController,
        object_capacity: int,
        stream_priority: int,
        frontier_layers_per_wave: int,
        sm_acquisition_waves: int,
        copy_engine_max_operations: int,
        profile_barrier: bool,
        profile_cpu: bool,
        profile_transfer: bool,
        stats: dict[str, Any],
        transfer_profiles: list[tuple[Any, Any, int, str]],
        transfer_plan: Callable[[PendingHostLoad], HostTransferLeasePlan],
        transport_program: Callable[[], Any],
        collect_barrier_profiles: Callable[[], None],
    ) -> None:
        if min(
            object_capacity,
            frontier_layers_per_wave,
            sm_acquisition_waves,
            copy_engine_max_operations,
        ) <= 0:
            raise ValueError("SGLang host transport geometry must be positive")
        if stream_priority > 0:
            raise ValueError("SGLang mover stream priority must be non-positive")
        self._runtime = runtime
        self._host_movers = host_movers
        self._object_capacity = object_capacity
        self._frontier_layers_per_wave = frontier_layers_per_wave
        self._sm_acquisition_waves = sm_acquisition_waves
        self._copy_engine_max_operations = copy_engine_max_operations
        self._profile_barrier = profile_barrier
        self._profile_cpu = profile_cpu
        self._profile_transfer = profile_transfer
        self._stats = stats
        self._transfer_profiles = transfer_profiles
        self._transfer_plan = transfer_plan
        self._transport_program = transport_program
        self._collect_barrier_profiles = collect_barrier_profiles
        # CUDA priorities are inverted. Acquisition is deliberately assigned
        # the lowest configured priority so it cannot preempt decode compute.
        self._prefetch_stream = torch.cuda.Stream(priority=stream_priority)
        self._copy_stream = torch.cuda.Stream(priority=stream_priority)
        self._ordered_task_head = torch.zeros(
            1,
            dtype=torch.int32,
            device=f"cuda:{runtime.device_ordinal}",
        )
        self._ready_events: tuple[
            tuple[tuple[torch.cuda.Event, ...], ...], ...
        ] = ()

    @property
    def stream(self) -> torch.cuda.Stream:
        """The stream carrying SM mover work and readiness publication."""

        return self._prefetch_stream

    def synchronize(self) -> None:
        """Quiesce both owned transport streams before teardown."""

        self._prefetch_stream.synchronize()
        self._copy_stream.synchronize()

    def prepare(
        self,
        pending: PendingHostLoad,
        *,
        first_local_layer: int = 0,
        last_local_layer: int | None = None,
    ) -> None:
        """Enqueue one non-overlapping, half-open model-layer frontier."""

        if self._profile_barrier:
            # Shared ready events are re-recorded across leases. Retire all
            # timing readers before publishing the next event generation.
            self._collect_barrier_profiles()
        pipeline_started = time.perf_counter_ns() if self._profile_cpu else 0
        controller = pending.controller
        layer_count = int(controller.layer_num)
        if last_local_layer is None:
            last_local_layer = layer_count
        if not 0 <= first_local_layer < last_local_layer <= layer_count:
            raise RuntimeError("HiCache acquisition frontier is outside the model")
        overlapping_layers = sorted(
            set(range(first_local_layer, last_local_layer))
            & set(pending.prefetched_layers)
        )
        if overlapping_layers:
            raise RuntimeError(
                "HiCache acquisition frontier overlaps published layers "
                f"{overlapping_layers}"
            )

        acquired_layer_count = last_local_layer - first_local_layer
        transfer_plan = self._transfer_plan(pending)
        mover_plan = transfer_plan.mover
        objects_per_layer = transfer_plan.objects_per_layer
        transfer_count = mover_plan.row_count
        if transfer_count <= 0 or transfer_count != int(
            pending.device_indices.numel()
        ):
            raise RuntimeError("HiCache host pipeline has no promoted pages")
        transfer_first_slot = self._transfer_first_slot(
            pending.consumer_index,
            layer_count,
            self._sm_acquisition_waves,
        )
        copy_runs = mover_plan.copy_runs
        sm_transfer_count = mover_plan.sm_row_count
        use_copy_engine = bool(copy_runs)
        use_sm_mover = sm_transfer_count != 0
        layer_geometry = transfer_plan.layer_geometry
        transfer_objects = transfer_plan.indexed_objects
        copy_groups = transfer_plan.copy_groups
        paired_copy = transfer_plan.paired_indexed_copy
        ordered_sm_waves = (
            use_sm_mover
            and transfer_plan.sm_waves_per_layer > 1
            and paired_copy
        )
        ready_events = self._events_for(
            pending,
            layer_count,
            max(1, transfer_plan.sm_waves_per_layer),
        )

        prefetched_layers: dict[int, _PrefetchedLayer] = {}
        profile_start = (
            torch.cuda.Event(enable_timing=True) if self._profile_transfer else None
        )
        profile_finish = (
            torch.cuda.Event(enable_timing=True) if self._profile_transfer else None
        )
        copy_engine_wave_layers = self._frontier_layers_per_wave
        if use_copy_engine:
            operations_per_layer = 2 * len(copy_runs)
            if operations_per_layer > self._copy_engine_max_operations:
                raise RuntimeError(
                    "copy-engine index map exceeds the configured operation bound"
                )
            copy_engine_wave_layers = max(
                1,
                min(
                    self._frontier_layers_per_wave,
                    self._copy_engine_max_operations // operations_per_layer,
                ),
            )

        try:
            producer_stream = torch.cuda.current_stream()
            if use_sm_mover:
                first_object = objects_per_layer * first_local_layer
                last_object = objects_per_layer * last_local_layer
                self._runtime.register_indexed_host_objects(
                    transfer_first_slot + first_object,
                    transfer_objects[first_object:last_object],
                    stream=producer_stream,
                )
            phase_start = (
                pending.producer_event.start_event
                if not pending.prefetched_layers
                else torch.cuda.Event()
            )
            if phase_start is not pending.producer_event.start_event:
                pending.transfer_events += (phase_start,)
            phase_start.record(producer_stream)
            phase_program = self._transport_program() if use_sm_mover else None
            hybrid_parallel = use_copy_engine and use_sm_mover
            copy_stream = self._copy_stream if hybrid_parallel else self._prefetch_stream
            with torch.cuda.stream(self._prefetch_stream):
                phase_start.wait(self._prefetch_stream)
                if profile_start is not None:
                    profile_start.record(self._prefetch_stream)
            if hybrid_parallel:
                with torch.cuda.stream(copy_stream):
                    phase_start.wait(copy_stream)

            local_layer = first_local_layer
            while local_layer < last_local_layer:
                wave_end = min(
                    last_local_layer,
                    local_layer
                    + (
                        copy_engine_wave_layers
                        if use_copy_engine
                        else self._frontier_layers_per_wave
                    ),
                )
                wave_bytes = sum(
                    key_bytes + value_bytes
                    for key_bytes, value_bytes in layer_geometry[
                        local_layer:wave_end
                    ]
                )
                wave_row_bytes = sum(
                    (key_bytes + value_bytes) // transfer_count
                    for key_bytes, value_bytes in layer_geometry[
                        local_layer:wave_end
                    ]
                )
                copy_wave_bytes = mover_plan.copy_row_count * wave_row_bytes
                sm_wave_bytes = sm_transfer_count * wave_row_bytes
                if copy_wave_bytes + sm_wave_bytes != wave_bytes:
                    raise RuntimeError("host mover byte partition is not exact")

                if use_sm_mover:
                    self._enqueue_sm_wave(
                        pending=pending,
                        phase_program=phase_program,
                        transfer_first_slot=transfer_first_slot,
                        local_layer=local_layer,
                        wave_end=wave_end,
                        paired_copy=paired_copy,
                        objects_per_layer=objects_per_layer,
                        ready_events=ready_events,
                        ordered_sm_waves=ordered_sm_waves,
                        sm_wave_bytes=sm_wave_bytes,
                        wave_bytes=wave_bytes,
                        mover_selection_reason=mover_plan.selection_reason,
                    )

                copy_done: torch.cuda.Event | None = None
                if use_copy_engine:
                    copy_done = self._enqueue_copy_wave(
                        pending=pending,
                        copy_stream=copy_stream,
                        hybrid_parallel=hybrid_parallel,
                        copy_groups=copy_groups,
                        copy_runs=copy_runs,
                        local_layer=local_layer,
                        wave_end=wave_end,
                        copy_wave_bytes=copy_wave_bytes,
                        wave_bytes=wave_bytes,
                        mover_selection_reason=mover_plan.selection_reason,
                    )

                with torch.cuda.stream(self._prefetch_stream):
                    if copy_done is not None:
                        pending.transfer_events += (copy_done,)
                        copy_done.wait(self._prefetch_stream)
                        self._stats["hybrid_parallel_waves"] += 1
                    if profile_finish is not None and wave_end == last_local_layer:
                        profile_finish.record(self._prefetch_stream)
                    # A paired one-wave SM launch (and every copy-engine wave)
                    # completes this whole adjacent layer group at one stream
                    # position. Publish one shared fence so the consumer sees
                    # the physical submission boundary rather than several
                    # equivalent host-polling identities.
                    shared_completion = (
                        not use_sm_mover
                        or transfer_plan.sm_waves_per_layer <= 1
                    )
                    shared_ready_event = (
                        ready_events[wave_end - 1][
                            max(1, transfer_plan.sm_waves_per_layer) - 1
                        ]
                        if shared_completion
                        else None
                    )
                    if shared_ready_event is not None:
                        shared_ready_event.record(self._prefetch_stream)
                    for ready_layer in range(local_layer, wave_end):
                        key_bytes, value_bytes = layer_geometry[ready_layer]
                        layer_events = ready_events[ready_layer]
                        layer_transfer = transfer_plan.layers[ready_layer]
                        if shared_ready_event is None and ordered_sm_waves:
                            # Object-owned waves can complete independently
                            # inside the persistent mover. Waiting for one
                            # layer's objects does not order the whole group,
                            # so retain a distinct full-layer fence identity.
                            layer_events[
                                transfer_plan.sm_waves_per_layer - 1
                            ].record(self._prefetch_stream)
                        ready_event = (
                            shared_ready_event
                            if shared_ready_event is not None
                            else layer_events[
                                max(1, transfer_plan.sm_waves_per_layer) - 1
                            ]
                        )
                        layer_first_slot = (
                            None
                            if not use_sm_mover
                            else transfer_first_slot
                            + objects_per_layer * ready_layer
                        )
                        prefetched_layers[ready_layer] = _PrefetchedLayer(
                            key_bytes=key_bytes,
                            value_bytes=value_bytes,
                            ready_event=ready_event,
                            transfer_first_slot=layer_first_slot,
                            transfer_object_id_base=(
                                layer_transfer.indexed_objects[0].object_id
                                if use_sm_mover
                                else None
                            ),
                            transfer_object_version=(
                                layer_transfer.indexed_objects[0].version
                                if use_sm_mover
                                else None
                            ),
                            registration_event=(
                                phase_start if ordered_sm_waves else None
                            ),
                            wave_events=(
                                ()
                                if ordered_sm_waves
                                else (
                                    (ready_event,)
                                    if shared_ready_event is not None
                                    and transfer_plan.sm_waves_per_layer == 1
                                    else layer_events[
                                        : transfer_plan.sm_waves_per_layer
                                    ]
                                    if use_sm_mover
                                    else ()
                                )
                            ),
                            wave_object_slots=(
                                tuple(
                                    layer_first_slot + 2 * wave
                                    for wave in range(
                                        transfer_plan.sm_waves_per_layer
                                    )
                                )
                                if ordered_sm_waves
                                and layer_first_slot is not None
                                else ()
                            ),
                            wave_row_ends=layer_transfer.wave_row_ends,
                        )
                    self._stats["lookahead_copy_waves"] = (
                        self._stats.get("lookahead_copy_waves", 0) + 1
                    )
                local_layer = wave_end
        except Exception:
            self.synchronize()
            self._stats["hicache_fallback_batches"] += 1
            raise

        pending.prefetched_layers.update(prefetched_layers)
        frontier_geometry = layer_geometry[first_local_layer:last_local_layer]
        if profile_start is not None and profile_finish is not None:
            transfer_bytes = sum(
                key_bytes + value_bytes
                for key_bytes, value_bytes in frontier_geometry
            )
            self._transfer_profiles.append(
                (profile_start, profile_finish, transfer_bytes, "pipeline")
            )
        self._stats["prefetched_layers"] += acquired_layer_count
        self._stats["prefetched_host_bytes"] += sum(
            key_bytes + value_bytes for key_bytes, value_bytes in frontier_geometry
        )
        self._stats["lookahead_acquisition_layers"] += acquired_layer_count
        self._stats["lookahead_acquisition_objects"] += (
            objects_per_layer * acquired_layer_count if use_sm_mover else 0
        )
        if paired_copy and use_sm_mover:
            self._stats["paired_lookahead_layers"] = (
                self._stats.get("paired_lookahead_layers", 0)
                + acquired_layer_count
            )
        if self._profile_cpu:
            self._stats["pipeline_cpu_ns"] = self._stats.get(
                "pipeline_cpu_ns", 0
            ) + (time.perf_counter_ns() - pipeline_started)

    def prepare_missing(
        self,
        pending: PendingHostLoad,
        *,
        exclude: frozenset[int] = frozenset(),
    ) -> int:
        """Enqueue every unpublished layer except explicit typed-demand gaps."""

        layer_count = int(pending.controller.layer_num)
        if any(layer < 0 or layer >= layer_count for layer in exclude):
            raise RuntimeError("typed-demand exclusion is outside the model")
        missing = [
            layer
            for layer in range(layer_count)
            if layer not in pending.prefetched_layers and layer not in exclude
        ]
        if not missing:
            return 0
        ranges: list[tuple[int, int]] = []
        range_begin = previous = missing[0]
        for layer in missing[1:]:
            if layer != previous + 1:
                ranges.append((range_begin, previous + 1))
                range_begin = layer
            previous = layer
        ranges.append((range_begin, previous + 1))
        for first_layer, last_layer in ranges:
            self.prepare(
                pending,
                first_local_layer=first_layer,
                last_local_layer=last_layer,
            )
        return len(missing)

    def _events_for(
        self,
        pending: PendingHostLoad,
        layer_count: int,
        event_count: int,
    ) -> tuple[tuple[torch.cuda.Event, ...], ...]:
        controller = pending.controller
        if not self._ready_events:
            self._ready_events = tuple(
                tuple(
                    tuple(
                        torch.cuda.Event(enable_timing=self._profile_barrier)
                        for _ in range(self._sm_acquisition_waves)
                    )
                    for _ in range(layer_count)
                )
                for _ in controller.layer_done_counter.events
            )
        if pending.consumer_index >= len(self._ready_events):
            raise RuntimeError("SGLang published an invalid HiCache producer slot")
        result = self._ready_events[pending.consumer_index]
        if (
            len(result) != layer_count
            or event_count <= 0
            or event_count > self._sm_acquisition_waves
        ):
            raise RuntimeError("SGLang HiCache layer count changed after initialization")
        return tuple(events[:event_count] for events in result)

    def _transfer_first_slot(
        self, consumer_index: int, layer_count: int, waves_per_layer: int
    ) -> int:
        begin, _end = pipeline_object_range(
            self._object_capacity,
            consumer_index,
            layer_count,
            waves_per_layer,
        )
        return begin

    def _enqueue_sm_wave(
        self,
        *,
        pending: PendingHostLoad,
        phase_program: Any,
        transfer_first_slot: int,
        local_layer: int,
        wave_end: int,
        paired_copy: bool,
        objects_per_layer: int,
        ready_events: tuple[tuple[torch.cuda.Event, ...], ...],
        ordered_sm_waves: bool,
        sm_wave_bytes: int,
        wave_bytes: int,
        mover_selection_reason: str,
    ) -> None:
        if phase_program is None:
            raise RuntimeError("SM host mover has no transport program")
        calibration_probe = mover_selection_reason == "calibration_probe_sm"
        profile_sm = self._host_movers.profile_enabled(
            "sm", wave_bytes, complete_calibration=calibration_probe
        )
        profile_start = (
            torch.cuda.Event(enable_timing=True) if profile_sm else None
        )
        profile_finish = (
            torch.cuda.Event(enable_timing=True) if profile_sm else None
        )
        with torch.cuda.stream(self._prefetch_stream):
            if profile_start is not None:
                profile_start.record(self._prefetch_stream)
            if objects_per_layer == 2:
                first_slot = transfer_first_slot + 2 * local_layer
                if paired_copy:
                    phase_program.preload_host_pairs(
                        self._runtime,
                        first_slot,
                        wave_end - local_layer,
                        self._prefetch_stream,
                    )
                else:
                    phase_program.preload_host(
                        self._runtime,
                        first_slot,
                        2 * (wave_end - local_layer),
                        self._prefetch_stream,
                    )
                submissions = 1
            elif ordered_sm_waves:
                if not paired_copy:
                    raise RuntimeError("ordered SM waves require paired KV objects")
                first_slot = transfer_first_slot + objects_per_layer * local_layer
                pair_count = (
                    (wave_end - local_layer) * objects_per_layer // 2
                )
                worker_blocks = min(
                    64,
                    2 * self._frontier_layers_per_wave,
                    2 * pair_count,
                )
                phase_program.preload_host_pairs_ordered(
                    self._runtime,
                    first_slot,
                    pair_count,
                    worker_blocks,
                    self._ordered_task_head,
                    self._prefetch_stream,
                )
                submissions = 1
            else:
                if objects_per_layer <= 2 or objects_per_layer % 2:
                    raise RuntimeError("SM acquisition wave objects are invalid")
                submissions = 0
                for layer in range(local_layer, wave_end):
                    first_slot = transfer_first_slot + objects_per_layer * layer
                    for wave, ready_event in enumerate(ready_events[layer]):
                        if paired_copy:
                            phase_program.preload_host_pairs(
                                self._runtime,
                                first_slot + 2 * wave,
                                1,
                                self._prefetch_stream,
                            )
                        else:
                            phase_program.preload_host(
                                self._runtime,
                                first_slot + 2 * wave,
                                2,
                                self._prefetch_stream,
                            )
                        ready_event.record(self._prefetch_stream)
                        submissions += 1
            if profile_finish is not None:
                profile_finish.record(self._prefetch_stream)
        if profile_start is not None and profile_finish is not None:
            self._host_movers.record_profile(
                MoverProfile(
                    profile_start,
                    profile_finish,
                    "sm",
                    sm_wave_bytes,
                    wave_bytes,
                    submissions,
                    0,
                    calibration_probe,
                )
            )
        self._stats["sm_mover_bytes"] += sm_wave_bytes
        self._stats["sm_acquisition_wave_submissions"] = self._stats.get(
            "sm_acquisition_wave_submissions", 0
        ) + submissions

    def _enqueue_copy_wave(
        self,
        *,
        pending: PendingHostLoad,
        copy_stream: torch.cuda.Stream,
        hybrid_parallel: bool,
        copy_groups: tuple[tuple[Any, ...], ...],
        copy_runs: tuple[Any, ...],
        local_layer: int,
        wave_end: int,
        copy_wave_bytes: int,
        wave_bytes: int,
        mover_selection_reason: str,
    ) -> torch.cuda.Event | None:
        wave_groups = tuple(
            group
            for groups in copy_groups[local_layer:wave_end]
            for group in groups
        )
        copy_operation_count = len(wave_groups) * len(copy_runs)
        if copy_operation_count <= 0:
            raise RuntimeError("copy-engine wave has no physical copy operations")
        calibration_probe = mover_selection_reason == "calibration_probe_copy"
        profile_copy = self._host_movers.profile_enabled(
            "copy_engine", wave_bytes, complete_calibration=calibration_probe
        )
        profile_start = (
            torch.cuda.Event(enable_timing=True) if profile_copy else None
        )
        profile_finish = (
            torch.cuda.Event(enable_timing=True) if profile_copy else None
        )
        copy_done = None
        with torch.cuda.stream(copy_stream):
            if profile_start is not None:
                profile_start.record(copy_stream)
            copy_issue_started = time.perf_counter_ns()
            copy_submissions = copy_strided_host_runs_async(
                wave_groups,
                copy_runs,
                copy_stream,
            )
            copy_issue_ns = time.perf_counter_ns() - copy_issue_started
            if profile_finish is not None:
                profile_finish.record(copy_stream)
            if hybrid_parallel:
                copy_done = torch.cuda.Event()
                copy_done.record(copy_stream)
        self._stats["copy_engine_waves"] += 1
        self._stats["copy_engine_submissions"] += copy_submissions
        self._stats["copy_engine_issue_cpu_ns"] += copy_issue_ns
        self._stats["copy_engine_operations"] += copy_operation_count
        self._stats["copy_engine_bytes"] += copy_wave_bytes
        if profile_start is not None and profile_finish is not None:
            self._host_movers.record_profile(
                MoverProfile(
                    profile_start,
                    profile_finish,
                    "copy_engine",
                    copy_wave_bytes,
                    wave_bytes,
                    copy_operation_count,
                    copy_issue_ns,
                    calibration_probe,
                )
            )
        return copy_done
