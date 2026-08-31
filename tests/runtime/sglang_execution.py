#!/usr/bin/env python3
"""Exhaustive behavior checks for SGLang's per-layer dispatch state."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

import torch

from nta_runtime.engines.sglang_execution import (
    AttentionDispatch,
    AttentionDispatchKind,
    SglangAttentionExecutor,
    select_attention_dispatch,
    use_preloaded_stock_alias,
)
from nta_runtime.engines.sglang_acquisition_contract import (
    AcquisitionConsumerPlan,
    AcquisitionTier,
    SglangForwardAcquisition,
    SglangLayerAcquisition,
)
from nta_runtime.engines.sglang_verification import SglangAttentionVerifier


class ReadyEvent:
    def __init__(self, ready: bool) -> None:
        self.ready = ready
        self.queries = 0

    def query(self) -> bool:
        self.queries += 1
        return self.ready


class FakeAcquisitionOwner(SglangForwardAcquisition):
    def __init__(self, tier: AcquisitionTier) -> None:
        self._tier = tier
        self.consumed = []
        self.abort_count = 0

    @property
    def tier(self) -> AcquisitionTier:
        return self._tier

    def layer(self, local_layer: int):
        del local_layer
        return None

    def consume_layer(self, layer, stream, *, wait_for_ready: bool) -> None:
        self.consumed.append((layer, stream, wait_for_ready))

    def finish(self, stream) -> None:
        del stream

    def abort_after_quiescence(self) -> None:
        self.abort_count += 1


def acquired_layer(
    event,
    *,
    local_layer: int = 2,
    layer_id: int = 6,
    progressive: bool = False,
    publication=None,
    tier: AcquisitionTier = AcquisitionTier.HOST_STAGED,
):
    owner = FakeAcquisitionOwner(tier)
    if tier is AcquisitionTier.HOST_STAGED:
        if publication is None:
            publication = SimpleNamespace(
                transfer_first_slot=8 if progressive else None
            )
        consumer_plan = AcquisitionConsumerPlan.HOST_MATERIALIZED
        backend_record = None
    else:
        publication = None
        consumer_plan = AcquisitionConsumerPlan.PREACQUIRED
        backend_record = object()
    return SglangLayerAcquisition(
        owner,
        local_layer,
        layer_id,
        event,
        tier,
        consumer_plan,
        publication,
        backend_record,
    )


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
        acquisition=None,
        layer_id=4,
    )
    assert direct.kind is AttentionDispatchKind.PREACQUIRED
    assert direct.local_layer == -1

    ready = ReadyEvent(True)
    preloaded = acquired_layer(ready, progressive=True)
    selected = select_attention_dispatch(
        pending=pending(),
        host_execution=execution(dependency=True, overlap=True),
        acquisition=preloaded,
        layer_id=6,
    )
    assert selected.kind is AttentionDispatchKind.PRELOADED
    assert ready.queries == 0
    assert use_preloaded_stock_alias(selected, alias_available=True)
    assert not use_preloaded_stock_alias(
        selected,
        alias_available=True,
        typed_observation_required=True,
    )

    arriving_event = ReadyEvent(False)
    arriving = acquired_layer(arriving_event, progressive=True)
    selected = select_attention_dispatch(
        pending=pending(),
        host_execution=execution(dependency=True, overlap=True),
        acquisition=arriving,
        layer_id=6,
    )
    assert selected.kind is AttentionDispatchKind.PRELOADED
    assert arriving_event.queries == 0

    selected = select_attention_dispatch(
        pending=pending(),
        host_execution=execution(dependency=True, overlap=True),
        acquisition=arriving,
        layer_id=6,
        progressive_consumer_planned=True,
    )
    assert selected.kind is AttentionDispatchKind.ARRIVING_PREFETCH
    assert arriving_event.queries == 1
    assert not use_preloaded_stock_alias(selected, alias_available=True)
    assert not use_preloaded_stock_alias(selected, alias_available=False)

    # The scheduler can revoke a progressive plan when its calibrated EDF
    # result predicts completion by the future GPU attention deadline. The
    # stock stream wait remains the correctness backstop.
    selected = select_attention_dispatch(
        pending=pending(),
        host_execution=execution(dependency=True, overlap=True),
        acquisition=arriving,
        layer_id=6,
    )
    assert selected.kind is AttentionDispatchKind.PRELOADED
    assert arriving_event.queries == 1

    # A shared producer fence already ordered on the numerical stream is
    # stronger than a racing host-side query. It must not create a second
    # partial consumer for a later layer in the same transport submission.
    selected = select_attention_dispatch(
        pending=pending(),
        host_execution=execution(dependency=True, overlap=True),
        acquisition=arriving,
        layer_id=6,
        prefetch_event_ordered=True,
        progressive_consumer_planned=True,
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
    wave_publication = SimpleNamespace(
        transfer_first_slot=8,
        wave_events=wave_events,
        wave_count=3,
    )
    wave_acquisition = acquired_layer(
        wave_events[-1],
        local_layer=0,
        layer_id=4,
        progressive=True,
        publication=wave_publication,
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
        wave_acquisition.ready_event,
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
                dispatch=SimpleNamespace(acquisition=wave_acquisition),
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
    no_direct_prefetch = acquired_layer(no_direct_event, progressive=True)
    selected = select_attention_dispatch(
        pending=pending(),
        host_execution=execution(dependency=True, overlap=False),
        acquisition=no_direct_prefetch,
        layer_id=6,
    )
    assert selected.kind is AttentionDispatchKind.PRELOADED
    assert no_direct_event.queries == 0
    assert use_preloaded_stock_alias(selected, alias_available=True)

    copy_event = ReadyEvent(False)
    copy_prefetch = acquired_layer(copy_event)
    selected = select_attention_dispatch(
        pending=pending(),
        host_execution=execution(dependency=True, overlap=True),
        acquisition=copy_prefetch,
        layer_id=6,
    )
    assert selected.kind is AttentionDispatchKind.PRELOADED
    assert copy_event.queries == 0

    nvme_acquisition = acquired_layer(
        ReadyEvent(False),
        local_layer=0,
        layer_id=4,
        tier=AcquisitionTier.NVME,
    )
    nvme = select_attention_dispatch(
        pending=pending(),
        host_execution=None,
        acquisition=nvme_acquisition,
        layer_id=4,
    )
    assert nvme.kind is AttentionDispatchKind.PRELOADED

    # Host and NVMe readiness use one executor branch.  The producer owner,
    # not a tier flag, orders the fence; only Host requires per-layer
    # materialization because NVMe prepared the preacquired plan up front.
    common_executor = SglangAttentionExecutor.__new__(SglangAttentionExecutor)
    common_executor._stats = defaultdict(int)
    common_executor._record_barrier_arrival = lambda *_args: None
    common_log = []
    common_executor._upload_plan = lambda *_args, **_kwargs: common_log.append(
        "host-materialize"
    )
    allocation = SimpleNamespace(
        plan=SimpleNamespace(has_external=False, work_item_count=3),
        object_count=0,
    )
    common_executor._materializer = SimpleNamespace(
        require_allocation=lambda _wrapper: allocation
    )
    common_executor.run_preacquired = lambda *_args, **kwargs: common_log.append(
        ("run", kwargs.get("validate_runtime_health", False))
    )
    common_batch = SimpleNamespace(
        pending_host_load=object(),
        semantic_plans={},
    )
    common_executor._execute_preloaded(
        dispatch=AttentionDispatch(
            AttentionDispatchKind.PRELOADED,
            preloaded.local_layer,
            preloaded,
            execution(dependency=True),
        ),
        batch=common_batch,
        wrapper="host-wrapper",
        q="query",
        kv_cache="kv",
        output="output",
        layer=SimpleNamespace(layer_id=6),
        stream="stream",
        run_options={},
    )
    assert preloaded.owner.consumed[-1][2]
    assert common_log == ["host-materialize", ("run", False)]

    nvme_wrapper = object()
    common_batch.semantic_plans[id(nvme_wrapper)] = SimpleNamespace(
        schedule=SimpleNamespace(work_count=3)
    )
    common_log.clear()
    common_executor._execute_preloaded(
        dispatch=AttentionDispatch(
            AttentionDispatchKind.PRELOADED,
            nvme_acquisition.local_layer,
            nvme_acquisition,
            None,
        ),
        batch=common_batch,
        wrapper=nvme_wrapper,
        q="query",
        kv_cache="kv",
        output="output",
        layer=SimpleNamespace(layer_id=4),
        stream="stream",
        run_options={},
    )
    assert nvme_acquisition.owner.consumed[-1][2]
    assert common_log == [("run", True)]
    assert common_executor._stats["nvme_preacquired_launches"] == 1

    incremental = select_attention_dispatch(
        pending=pending(),
        host_execution=execution(dependency=True),
        acquisition=None,
        layer_id=4,
    )
    assert incremental.kind is AttentionDispatchKind.HOST_INCREMENTAL

    bulk = select_attention_dispatch(
        pending=pending(),
        host_execution=execution(dependency=True, bulk=True),
        acquisition=None,
        layer_id=4,
    )
    assert bulk.kind is AttentionDispatchKind.HOST_DEVICE_BULK

    malformed = (
        dict(
            pending=None, host_execution=execution(dependency=True), acquisition=None
        ),
        dict(pending=pending(), host_execution=None, acquisition=None),
        dict(
            pending=pending(),
            host_execution=execution(dependency=False),
            acquisition=None,
        ),
        dict(
            pending=pending(),
            host_execution=execution(dependency=True),
            acquisition=nvme_acquisition,
        ),
        dict(
            pending=pending(),
            host_execution=None,
            acquisition=acquired_layer(
                ReadyEvent(True), local_layer=0, layer_id=4, progressive=True
            ),
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
