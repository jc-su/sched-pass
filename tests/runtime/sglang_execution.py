#!/usr/bin/env python3
"""Exhaustive behavior checks for SGLang's per-layer dispatch state."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

import torch

from nta_runtime.engines.sglang_execution import (
    AttentionDispatchKind,
    SglangAttentionExecutor,
    select_attention_dispatch,
    use_preloaded_stock_alias,
)
from nta_runtime.engines.sglang_verification import SglangAttentionVerifier


class ReadyEvent:
    def __init__(self, ready: bool) -> None:
        self.ready = ready
        self.queries = 0

    def query(self) -> bool:
        self.queries += 1
        return self.ready


def pending(prefetched_layers=None):
    return SimpleNamespace(
        controller=SimpleNamespace(mem_pool_device=SimpleNamespace(start_layer=4)),
        prefetched_layers={} if prefetched_layers is None else prefetched_layers,
    )


def execution(*, dependency: bool, overlap: bool = False, bulk: bool = False):
    return SimpleNamespace(
        uses_dependency_protocol=dependency,
        overlap_initial=overlap,
        uses_device_bulk=bulk,
    )


def main() -> None:
    direct = select_attention_dispatch(
        pending=None,
        host_execution=None,
        tier_is_nvme=False,
        layer_id=4,
    )
    assert direct.kind is AttentionDispatchKind.PREACQUIRED
    assert direct.local_layer == -1

    ready = ReadyEvent(True)
    preloaded = SimpleNamespace(transfer_first_slot=8, ready_event=ready)
    selected = select_attention_dispatch(
        pending=pending({2: preloaded}),
        host_execution=execution(dependency=True, overlap=True),
        tier_is_nvme=False,
        layer_id=6,
    )
    assert selected.kind is AttentionDispatchKind.PRELOADED
    assert ready.queries == 1
    assert use_preloaded_stock_alias(selected, alias_available=True)
    assert not use_preloaded_stock_alias(
        selected,
        alias_available=True,
        typed_observation_required=True,
    )

    arriving_event = ReadyEvent(False)
    arriving = SimpleNamespace(transfer_first_slot=8, ready_event=arriving_event)
    selected = select_attention_dispatch(
        pending=pending({2: arriving}),
        host_execution=execution(dependency=True, overlap=True),
        tier_is_nvme=False,
        layer_id=6,
    )
    assert selected.kind is AttentionDispatchKind.ARRIVING_PREFETCH
    assert arriving_event.queries == 1
    assert not use_preloaded_stock_alias(selected, alias_available=True)
    assert not use_preloaded_stock_alias(selected, alias_available=False)

    # A calibrated EDF result is about the future GPU attention deadline, not
    # host-side readiness at dispatch time. Preserve that decision and let the
    # stock stream wait enforce correctness without a racing event query.
    selected = select_attention_dispatch(
        pending=pending({2: arriving}),
        host_execution=execution(dependency=True, overlap=True),
        tier_is_nvme=False,
        layer_id=6,
        modeled_ready_by_attention=True,
    )
    assert selected.kind is AttentionDispatchKind.PRELOADED
    assert arriving_event.queries == 1

    # A shared producer fence already ordered on the numerical stream is
    # stronger than a racing host-side query. It must not create a second
    # partial consumer for a later layer in the same transport submission.
    selected = select_attention_dispatch(
        pending=pending({2: arriving}),
        host_execution=execution(dependency=True, overlap=True),
        tier_is_nvme=False,
        layer_id=6,
        prefetch_event_ordered=True,
    )
    assert selected.kind is AttentionDispatchKind.PRELOADED
    assert arriving_event.queries == 1

    # The arriving executor consumes the producer's exact wave partition. It
    # must issue only non-empty windows, retain one reusable structural queue,
    # and expose actual rather than canonical launch amplification.
    wave_executor = SglangAttentionExecutor.__new__(SglangAttentionExecutor)
    wave_executor._runtime = object()
    wave_executor._stats = defaultdict(int)
    wave_executor._kernels = SimpleNamespace(
        transport_program=lambda: "phase-program"
    )
    wave_events = (object(), object(), object())
    wave_prefetch = SimpleNamespace(
        transfer_first_slot=8,
        ready_event=wave_events[-1],
        wave_events=wave_events,
        wave_object_slots=(),
        registration_event=None,
        wave_count=3,
    )
    wave_schedule = SimpleNamespace(work_count=7)
    wave_plan = object()
    wave_host_execution = execution(dependency=True, overlap=True)
    wave_executor._materializer = SimpleNamespace(
        require_allocation=lambda _wrapper: SimpleNamespace(
            direct_work_count=2,
            event_partitioned=True,
            event_wave_work_counts=(0, 2, 3),
        )
    )
    wave_executor._record_barrier_arrival = lambda *_args: None
    wave_executor._upload_plan = lambda *_args, **_kwargs: (
        wave_plan,
        wave_schedule,
        0,
        wave_prefetch.ready_event,
        0,
        wave_host_execution,
    )
    wave_batch = SimpleNamespace(
        pending_host_load=object(), arriving_partition_key=None
    )
    wave_calls = []
    with patch(
        "nta_runtime.engines.sglang_execution.enqueue_event_partitioned_attention",
        lambda *args, **kwargs: wave_calls.append((args, kwargs)),
    ):
        for expected_prepare in (True, False):
            outcome = wave_executor._execute_arriving_prefetch(
                dispatch=SimpleNamespace(prefetched=wave_prefetch),
                batch=wave_batch,
                wrapper="wrapper",
                q="query",
                kv_cache="kv",
                output="output",
                layer=SimpleNamespace(layer_id=4, scaling=1.0),
                stream="consumer-stream",
                run_options={},
                final_layer=False,
                verify_execution=False,
                verify_transfer=False,
                tile_compute_ns=11,
            )
            assert outcome.progress_rounds == 2
            assert outcome.progressive_consumer
            assert wave_calls[-1][1]["prepare_partition"] is expected_prepare
    assert wave_calls[0][1]["ready_events"] == wave_events
    assert wave_calls[0][1]["wave_work_counts"] == (0, 2, 3)
    assert wave_executor._stats["arriving_partition_preparations"] == 1
    assert wave_executor._stats["arriving_partition_reuses"] == 1
    assert wave_executor._stats["compact_resume_launches"] == 4
    assert wave_executor._stats["compact_resume_cta_bound"] == 10
    assert wave_executor._stats["canonical_resume_cta_bound"] == 28
    assert wave_executor._stats["event_ordered_wave_launches"] == 4

    # A directory-backed producer is not by itself evidence that a partial
    # consumer exists.  External-only/cache-placement forwards have no direct
    # work to overlap and must wait for the event before using the stock alias.
    no_direct_event = ReadyEvent(False)
    no_direct_prefetch = SimpleNamespace(
        transfer_first_slot=8,
        ready_event=no_direct_event,
    )
    selected = select_attention_dispatch(
        pending=pending({2: no_direct_prefetch}),
        host_execution=execution(dependency=True, overlap=False),
        tier_is_nvme=False,
        layer_id=6,
    )
    assert selected.kind is AttentionDispatchKind.PRELOADED
    assert no_direct_event.queries == 0
    assert use_preloaded_stock_alias(selected, alias_available=True)

    copy_event = ReadyEvent(False)
    copy_prefetch = SimpleNamespace(transfer_first_slot=None, ready_event=copy_event)
    selected = select_attention_dispatch(
        pending=pending({2: copy_prefetch}),
        host_execution=execution(dependency=True, overlap=True),
        tier_is_nvme=False,
        layer_id=6,
    )
    assert selected.kind is AttentionDispatchKind.PRELOADED
    assert copy_event.queries == 0

    nvme = select_attention_dispatch(
        pending=pending(),
        host_execution=None,
        tier_is_nvme=True,
        layer_id=4,
    )
    assert nvme.kind is AttentionDispatchKind.NVME

    incremental = select_attention_dispatch(
        pending=pending(),
        host_execution=execution(dependency=True),
        tier_is_nvme=False,
        layer_id=4,
    )
    assert incremental.kind is AttentionDispatchKind.HOST_INCREMENTAL

    bulk = select_attention_dispatch(
        pending=pending(),
        host_execution=execution(dependency=True, bulk=True),
        tier_is_nvme=False,
        layer_id=4,
    )
    assert bulk.kind is AttentionDispatchKind.HOST_DEVICE_BULK

    malformed = (
        dict(
            pending=None, host_execution=execution(dependency=True), tier_is_nvme=False
        ),
        dict(pending=pending(), host_execution=None, tier_is_nvme=False),
        dict(
            pending=pending(),
            host_execution=execution(dependency=False),
            tier_is_nvme=False,
        ),
        dict(
            pending=pending(),
            host_execution=execution(dependency=True),
            tier_is_nvme=True,
        ),
        dict(
            pending=pending(
                {
                    0: SimpleNamespace(
                        transfer_first_slot=8,
                        ready_event=ReadyEvent(True),
                    )
                }
            ),
            host_execution=None,
            tier_is_nvme=False,
        ),
    )
    for arguments in malformed:
        try:
            select_attention_dispatch(layer_id=4, **arguments)
        except RuntimeError:
            pass
        else:
            raise AssertionError("malformed attention dispatch was accepted")

    graph_executor = SglangAttentionExecutor.__new__(SglangAttentionExecutor)
    graph_executor._stats = {}
    graph_log = []

    class GraphEpoch:
        def mark_consumed_after_replay(self, stream) -> None:
            graph_log.append(("retire", stream))

    graph_stream = object()
    graph_executor._complete_graph_replay(
        GraphEpoch(),
        graph_stream,
        lambda stream: graph_log.append(("publish", stream)),
    )
    assert graph_log == [
        ("retire", graph_stream),
        ("publish", graph_stream),
    ]
    assert graph_executor._stats["progress_feedback_graph_completion_snapshots"] == 1
    graph_executor._complete_graph_replay(GraphEpoch(), graph_stream, None)
    assert graph_log[-1] == ("retire", graph_stream)
    assert graph_executor._stats["progress_feedback_graph_completion_snapshots"] == 1

    orchestration_log = []

    class HostEpoch:
        def check(self, rounds, stream) -> None:
            orchestration_log.append(("check", rounds, stream))

    host_epoch = HostEpoch()
    host_execution = execution(dependency=True)
    prepared = SimpleNamespace(
        device_bulk=False,
        object_count=3,
        host_execution=host_execution,
        epoch=host_epoch,
        progress_rounds=2,
        collect_progress=True,
        coalesce_stream_retirement=False,
        template=SimpleNamespace(progressive_consumer=True),
    )
    host_executor = SglangAttentionExecutor.__new__(SglangAttentionExecutor)
    host_executor._stats = {"ticketed_incremental_launches": 0}
    host_executor._runtime = SimpleNamespace(sticky_failed_count=0)
    host_executor._prepare_host_layer = lambda **kwargs: prepared
    host_executor._progress_publisher = lambda batch, local_layer: (
        orchestration_log.append(("publisher", batch, local_layer)) or None
    )
    host_executor._submit_host_layer = lambda **kwargs: (
        orchestration_log.append(("submit", kwargs["prepared"])) or ("host-output", 17)
    )
    host_executor._account_host_progress = lambda *args: orchestration_log.append(
        ("account", args[0])
    )
    host_executor._record_opportunity = lambda *args: orchestration_log.append(
        ("opportunity", args[0])
    )
    host_batch = SimpleNamespace(pending_host_load=object())
    host_stream = object()
    host_outcome = host_executor.execute_host(
        dispatch=SimpleNamespace(
            kind=AttentionDispatchKind.HOST_INCREMENTAL,
            local_layer=4,
            host_execution=host_execution,
        ),
        batch=host_batch,
        wrapper="wrapper",
        q="query",
        kv_cache="kv",
        output="initial-output",
        layer="layer",
        stream=host_stream,
        run_options={},
        causal=True,
        window_left=-1,
        final_layer=True,
        verify_execution=False,
        verify_transfer=False,
        observe_setup=True,
        enqueue_started_ns=1,
        host_cost_model=object(),
        active_opportunity_batch=9,
    )
    assert host_outcome.output == "host-output"
    assert host_outcome.epoch is host_epoch
    assert host_outcome.indexed_object_count == 3
    assert host_outcome.progress_rounds == 2
    assert host_outcome.progressive_consumer
    assert host_outcome.setup_dispatch_elapsed_ns == 17
    assert host_outcome.deadline_fragment.wrapper == "wrapper"
    assert host_outcome.deadline_fragment.object_count == 3
    assert host_outcome.deadline_fragment.stream is host_stream
    assert host_executor._stats["ticketed_incremental_launches"] == 1
    assert [entry[0] for entry in orchestration_log] == [
        "publisher",
        "submit",
        "check",
        "account",
        "opportunity",
    ]

    host_key = torch.arange(12, dtype=torch.float32).view(3, 4)
    host_value = host_key + 100
    actual_key = torch.zeros((4, 4), dtype=torch.float32)
    actual_value = torch.zeros((4, 4), dtype=torch.float32)
    actual_key[1] = host_key[2]
    actual_key[3] = host_key[0]
    actual_value[1] = host_value[2]
    actual_value[3] = host_value[0]
    transfer_batch = SimpleNamespace(
        pending_host_load=SimpleNamespace(
            controller=SimpleNamespace(
                mem_pool_device=SimpleNamespace(start_layer=5),
                mem_pool_host=SimpleNamespace(
                    k_data_refs=(host_key,),
                    v_data_refs=(host_value,),
                ),
            ),
            materialize_mapping=lambda: {1: 2, 3: 0},
        )
    )
    with patch(
        "torch.cuda.current_stream",
        return_value=SimpleNamespace(synchronize=lambda: None),
    ):
        SglangAttentionVerifier.verify_layer_transfer(
            transfer_batch,
            5,
            (actual_key, actual_value),
        )
        actual_value[3, 0] += 1
        try:
            SglangAttentionVerifier.verify_layer_transfer(
                transfer_batch,
                5,
                (actual_key, actual_value),
            )
        except RuntimeError as error:
            assert "indexed value transfer mismatch" in str(error)
        else:
            raise AssertionError("transfer verifier accepted a corrupted KV row")

    print("sglang_execution=pass")


if __name__ == "__main__":
    main()
