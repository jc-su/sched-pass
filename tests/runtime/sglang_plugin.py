#!/usr/bin/env python3
"""Validate SGLang plugin registration in frontend and spawned workers."""

from __future__ import annotations

import inspect
import multiprocessing as mp
import types
from collections import namedtuple
from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import patch

import torch


def load_in_spawn(result) -> None:
    from importlib.metadata import PackageNotFoundError, distribution, entry_points
    import os
    import sys

    from sglang.srt.plugins import load_plugins

    discovered = [
        (entry.name, entry.value) for entry in entry_points(group="sglang.srt.plugins")
    ]
    try:
        dist_path = str(distribution("nta-runtime")._path)
    except PackageNotFoundError:
        dist_path = "missing"
    load_plugins()
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
    from sglang.srt.server_args import ATTENTION_BACKEND_CHOICES

    result.put(
        (
            "nta_flashinfer" in ATTENTION_BACKENDS,
            ATTENTION_BACKEND_CHOICES.count("nta_flashinfer"),
            discovered,
            sys.executable,
            sys.path,
            os.getuid(),
            dist_path,
        )
    )


def main() -> None:
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
    from sglang.srt.server_args import ATTENTION_BACKEND_CHOICES

    from nta_runtime.plugins.sglang import (
        BACKEND_NAME,
        _EXECUTE_DECODE_TARGET,
        _EXECUTE_EXTEND_TARGET,
        _EAGER_LOAD_BATCH_TARGET,
        _PREFILL_FINISH_TARGET,
        _PREBUILT_FINISH_TARGET,
        _attach_request_priorities,
        _require_hooks_installed,
        _retire_prefill_finished_requests,
        _retire_finished_request,
        register,
    )
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    # Profiling is an explicit process-start diagnostic.  The default plugin
    # must not wrap SGLang's hottest forward methods.
    with patch.dict("os.environ", {"NTA_PROFILE_FORWARD": "0"}):
        register()
        register()
    assert BACKEND_NAME in ATTENTION_BACKENDS
    assert ATTENTION_BACKEND_CHOICES.count(BACKEND_NAME) == 1
    assert callable(ATTENTION_BACKENDS[BACKEND_NAME])

    for target in (_EXECUTE_EXTEND_TARGET, _EXECUTE_DECODE_TARGET):
        assert target not in HookRegistry._hooks

    from nta_runtime.plugins.sglang import (
        _ABORT_TARGET,
        _DECODE_GRAPH_REPLAY_VIEW_TARGET,
        _HICACHE_LOAD_TARGET,
        _PREFILL_REQUEST_BIND_TARGET,
        _PREFILL_GRAPH_CAPTURE_PREPARE_TARGET,
        _PREFILL_GRAPH_LOAD_BATCH_TARGET,
        _REQUEST_FINISH_TARGET,
        _RELEASE_TARGET,
        _FORWARD_BATCH_TARGET,
        _PREFILL_ADMISSION_TARGET,
    )

    HookRegistry.apply_hooks()
    _require_hooks_installed(HookRegistry)

    for target in (
        _DECODE_GRAPH_REPLAY_VIEW_TARGET,
        _PREFILL_GRAPH_CAPTURE_PREPARE_TARGET,
        _PREFILL_GRAPH_LOAD_BATCH_TARGET,
    ):
        assert any(
            kind == HookType.AROUND for kind, _, _ in HookRegistry._hooks[target]
        )

    assert any(
        kind == HookType.AROUND
        for kind, _, _ in HookRegistry._hooks[_HICACHE_LOAD_TARGET]
    )
    assert any(
        kind == HookType.AROUND
        for kind, _, _ in HookRegistry._hooks[_PREFILL_REQUEST_BIND_TARGET]
    )
    assert any(
        kind == HookType.BEFORE for kind, _, _ in HookRegistry._hooks[_ABORT_TARGET]
    )
    assert any(
        kind == HookType.BEFORE for kind, _, _ in HookRegistry._hooks[_RELEASE_TARGET]
    )
    assert any(
        kind == HookType.BEFORE
        for kind, _, _ in HookRegistry._hooks[_REQUEST_FINISH_TARGET]
    )
    for target in (_PREFILL_FINISH_TARGET, _PREBUILT_FINISH_TARGET):
        assert any(
            kind == HookType.AFTER for kind, _, _ in HookRegistry._hooks[target]
        )
    assert any(
        kind == HookType.AFTER
        for kind, _, _ in HookRegistry._hooks[_FORWARD_BATCH_TARGET]
    )
    assert any(
        kind == HookType.AROUND
        for kind, _, _ in HookRegistry._hooks[_PREFILL_ADMISSION_TARGET]
    )
    assert any(
        kind == HookType.AROUND
        for kind, _, _ in HookRegistry._hooks[_EAGER_LOAD_BATCH_TARGET]
    )

    # Completion must retire the generation before SGLang releases and reuses
    # its request-pool slot.  The result processor reaches the same backend
    # graph as Scheduler through model_worker.model_runner.
    retired = []
    lifecycle_backend = types.SimpleNamespace(
        retire_request=lambda request_id: retired.append(request_id)
    )
    lifecycle_processor = types.SimpleNamespace(
        model_worker=types.SimpleNamespace(
            model_runner=types.SimpleNamespace(attn_backend=lifecycle_backend)
        )
    )
    _retire_finished_request(
        lifecycle_processor,
        types.SimpleNamespace(rid="completed-request", finished=lambda: True),
    )
    assert retired == ["completed-request"]
    _retire_finished_request(
        lifecycle_processor,
        types.SimpleNamespace(rid="live-request", finished=lambda: False),
    )
    assert retired == ["completed-request"]
    _retire_prefill_finished_requests(
        None,
        lifecycle_processor,
        types.SimpleNamespace(
            reqs=(
                types.SimpleNamespace(rid="prefill-complete", finished=lambda: True),
                types.SimpleNamespace(rid="prefill-live", finished=lambda: False),
            )
        ),
    )
    assert retired == ["completed-request", "prefill-complete"]

    # SGLang 0.5.16 passes capture_hidden_mode through ForwardBatch.init_new.
    # The policy sidecar does not consume that extension argument, but the
    # hook must remain callable when SGLang adds it (or equivalent arguments).
    hook_forward_batch = types.SimpleNamespace(
        rids=("request-0", "request-1"),
        req_pool_indices=torch.tensor((11, 23), dtype=torch.int32),
    )
    hook_batch = types.SimpleNamespace(
        reqs=(types.SimpleNamespace(priority=0), types.SimpleNamespace(priority=0))
    )
    hook_runner = types.SimpleNamespace(
        server_args=types.SimpleNamespace(enable_priority_scheduling=False)
    )
    _attach_request_priorities(
        hook_forward_batch,
        object,
        hook_batch,
        hook_runner,
        capture_hidden_mode=None,
    )
    from nta_runtime.adapters.sglang import SglangAcquisitionSpan

    assert hook_forward_batch._nta_forward_metadata.request_slots == (11, 23)
    assert (
        hook_forward_batch._nta_forward_metadata.acquisitions
        == (SglangAcquisitionSpan.direct(),) * 2
    )

    external_request = types.SimpleNamespace(
        rid="external",
        priority=0,
        host_hit_length=32,
        swa_host_hit_length=0,
        mamba_host_hit_length=0,
        best_match_node=types.SimpleNamespace(id=41),
        last_node=types.SimpleNamespace(id=41),
        prefix_indices=torch.empty((0,), dtype=torch.int64),
        needs_host_load_back=lambda: True,
    )
    resident_request = types.SimpleNamespace(
        rid="resident",
        priority=0,
        host_hit_length=32,
        swa_host_hit_length=0,
        mamba_host_hit_length=0,
        best_match_node=types.SimpleNamespace(id=99),
        last_node=types.SimpleNamespace(id=99),
        prefix_indices=torch.empty((0,), dtype=torch.int64),
        needs_host_load_back=lambda: False,
    )
    mixed_forward_batch = types.SimpleNamespace(
        batch_size=2,
        rids=("external", "resident"),
        req_pool_indices=torch.tensor((7, 9), dtype=torch.int32),
    )
    mixed_schedule_batch = types.SimpleNamespace(
        reqs=(external_request, resident_request),
        decoding_reqs=(resident_request,),
        hicache_consumer_index=3,
    )
    from nta_runtime.plugins.sglang import _capture_prefill_request_binding

    load_queue = []
    adder = types.SimpleNamespace(
        tree_cache=types.SimpleNamespace(
            cache_controller=types.SimpleNamespace(load_queue=load_queue)
        )
    )

    def load_external(_adder, request):
        operation = types.SimpleNamespace(
            id=73,
            node_ids=[41],
            device_indices=torch.arange(32, dtype=torch.int64),
        )
        load_queue.append(operation)
        request.prefix_indices = operation.device_indices
        request.last_node = types.SimpleNamespace(id=41)

    _capture_prefill_request_binding(load_external, adder, external_request)
    _capture_prefill_request_binding(
        lambda _adder, _request: None, adder, resident_request
    )
    _attach_request_priorities(
        mixed_forward_batch,
        object,
        mixed_schedule_batch,
        hook_runner,
    )
    assert mixed_forward_batch._nta_forward_metadata.acquisitions == (
        SglangAcquisitionSpan(73, 41, 0, 32),
        SglangAcquisitionSpan.direct(),
    )
    assert external_request._nta_acquisition_span == SglangAcquisitionSpan.direct()

    from nta_runtime.plugins.sglang import _preserve_eager_load_batch

    eager_view = _preserve_eager_load_batch(
        lambda _runner, _batch: types.SimpleNamespace(),
        object(),
        mixed_forward_batch,
    )
    assert eager_view._nta_forward_metadata is mixed_forward_batch._nta_forward_metadata

    from nta_runtime.engines.sglang_admission import (
        AcquisitionAdmission,
        AdmissionConfig,
        route_prefill_admission,
    )
    from nta_runtime.engines.sglang_hicache import (
        HostLoadProgress,
        LeaseOperationTransfer,
        LeaseWorkDependency,
        PendingHostLoad,
        SglangHiCacheBridge,
    )
    from nta_runtime.tenant import tenant_budget_specs
    from nta_runtime.progress_frontier import FrontierState
    from nta_runtime.requests import RequestBinding, stable_request_id
    from nta_runtime.runtime import RequestProgress
    from nta_runtime.fixed_range_pool import FixedRangePool

    with patch.dict(
        "os.environ",
        {"NTA_TENANT_BUDGETS": "2:1048576,7:2097152"},
        clear=False,
    ):
        assert tenant_budget_specs() == ((2, 1048576), (7, 2097152))

    ranges = FixedRangePool(128, 24, reserved_low=2)
    first = ranges.acquire(11)
    second = ranges.acquire(12)
    assert first.begin >= 2 and first.end <= 128
    assert second.begin >= 2 and second.end <= 128
    assert first.end <= second.begin or second.end <= first.begin
    assert ranges.in_use == 2 and ranges.high_watermark == 2
    ranges.release(first)
    reused = ranges.acquire(13)
    assert reused.slot == first.slot
    assert reused.generation != first.generation
    try:
        ranges.release(first)
    except RuntimeError:
        pass
    else:
        raise AssertionError("range pool accepted a stale lease")
    ranges.release(second)
    ranges.release(reused)

    class Request:
        def __init__(self, rid: str) -> None:
            self.rid = rid
            self.to_finish = None

        def finished(self) -> bool:
            return False

    class Bridge:
        def __init__(self) -> None:
            self.leading_layers = 0
            self.stats = {}
            self.frontier = None

        def progress(self, consumer_index: int) -> HostLoadProgress:
            return HostLoadProgress(
                consumer_index=consumer_index,
                published_layers=12,
                leading_layers=self.leading_layers,
                total_layers=12,
                leading_bytes=self.leading_layers * 64,
                total_bytes=768,
            )

        def transfer_bytes(self, consumer_index: int) -> int:
            assert consumer_index == 3
            return 768

        def record_admission(self, **increments: int) -> None:
            for name, value in increments.items():
                self.stats[name] = self.stats.get(name, 0) + value

        def poll_request_frontier(self, request_ids: set[int]):
            del request_ids
            result = self.frontier
            self.frontier = None
            return result

    clock = [1_000]
    config = AdmissionConfig(True, 4, 100, 1)
    resident = Request("resident")
    scheduler = types.SimpleNamespace(
        running_batch=types.SimpleNamespace(reqs=[resident])
    )
    external = Request("external")
    batch = types.SimpleNamespace(
        reqs=[external], hicache_consumer_index=3, decoding_reqs=None
    )
    bridge = Bridge()
    admission = AcquisitionAdmission(config, clock=lambda: clock[0])
    assert admission.consider(scheduler, batch, bridge) is None
    assert admission.poll(scheduler) is None
    bridge.leading_layers = 4
    clock[0] += 50
    assert admission.poll(scheduler) is batch
    assert bridge.stats["admission_hidden_decode_steps"] == 1
    assert bridge.stats["admission_released_lead"] == 1

    class UnpublishedBridge(Bridge):
        def progress(self, consumer_index: int) -> HostLoadProgress:
            return HostLoadProgress(
                consumer_index=consumer_index,
                published_layers=0,
                leading_layers=0,
                total_layers=12,
                leading_bytes=0,
                total_bytes=768,
            )

    unpublished_bridge = UnpublishedBridge()
    admission = AcquisitionAdmission(config, clock=lambda: clock[0])
    assert admission.consider(scheduler, batch, unpublished_bridge) is batch
    assert unpublished_bridge.stats["admission_released_for_binding"] == 1
    assert not admission.has_staged_batch

    bridge.leading_layers = 0
    admission = AcquisitionAdmission(config, clock=lambda: clock[0])
    assert admission.consider(scheduler, batch, bridge) is None
    clock[0] += 101
    assert admission.poll(scheduler) is batch
    assert bridge.stats["admission_released_deadline"] == 1

    scheduler.running_batch.reqs.clear()
    admission = AcquisitionAdmission(config, clock=lambda: clock[0])
    assert admission.consider(scheduler, batch, bridge) is batch
    assert bridge.stats["admission_released_without_decode"] == 1

    scheduler.running_batch.reqs.append(resident)
    admission = AcquisitionAdmission(config, clock=lambda: clock[0])
    assert admission.consider(scheduler, batch, bridge) is None
    admission.cancel("external", all=False)
    assert external.to_finish is not None
    assert admission.poll(scheduler) is batch
    assert bridge.stats["admission_released_cancelled"] == 1

    mixed = types.SimpleNamespace(
        reqs=[external, resident],
        hicache_consumer_index=3,
        decoding_reqs=[resident],
    )
    admission = AcquisitionAdmission(config, clock=lambda: clock[0])
    assert admission.consider(scheduler, mixed, bridge) is mixed
    assert bridge.stats["admission_released_mixed_batches"] == 1
    assert bridge.stats["admission_external_bytes"] >= 768

    bridge.frontier = types.SimpleNamespace(state=FrontierState.DATA_BLOCKED)
    admission = AcquisitionAdmission(config, clock=lambda: clock[0])
    assert admission.consider(scheduler, batch, bridge) is batch
    assert bridge.stats["admission_feedback_data_blocked"] == 1
    assert bridge.stats["admission_released_feedback_data_blocked"] == 1

    bridge.frontier = types.SimpleNamespace(state=FrontierState.EXECUTABLE)
    admission = AcquisitionAdmission(config, clock=lambda: clock[0])
    assert admission.consider(scheduler, batch, bridge) is None
    assert bridge.stats["admission_feedback_executable"] == 1

    # SGLang 0.5.16's raw prefill seam returns a pair, not a ScheduleBatch.
    # The adapter must preserve the running-batch half while delaying only the
    # newly allocated external batch, and must not call the scheduler again
    # while that batch is staged.
    route_scheduler = types.SimpleNamespace(
        running_batch=types.SimpleNamespace(reqs=[])
    )
    route_running = types.SimpleNamespace(reqs=[resident])
    route_batch = types.SimpleNamespace(
        reqs=[external], hicache_consumer_index=3, decoding_reqs=None
    )
    route_calls = []

    def raw_prefill(_scheduler, **_kwargs):
        route_calls.append(True)
        return route_batch, route_running

    route_clock = [2_000]
    route_admission = AcquisitionAdmission(
        AdmissionConfig(True, 4, 100, 1), clock=lambda: route_clock[0]
    )
    setattr(route_scheduler, "_nta_acquisition_admission", route_admission)
    with patch(
        "nta_runtime.engines.sglang_admission._bridge_for_batch",
        return_value=bridge,
    ):
        first = route_prefill_admission(
            raw_prefill,
            route_scheduler,
            prefill_delayer_single_pass=None,
            running_batch=route_running,
        )
        assert first == (None, route_running)
        bridge.leading_layers = 4
        second = route_prefill_admission(
            raw_prefill,
            route_scheduler,
            prefill_delayer_single_pass=None,
            running_batch=route_running,
        )
    assert second == (route_batch, route_running)
    assert len(route_calls) == 1

    def malformed_prefill(_scheduler, **_kwargs):
        return route_batch

    try:
        route_prefill_admission(
            malformed_prefill,
            types.SimpleNamespace(running_batch=route_running),
            prefill_delayer_single_pass=None,
            running_batch=route_running,
        )
    except RuntimeError as error:
        assert "pinned tuple shape" in str(error)
    else:
        raise AssertionError("prefill admission accepted a malformed return shape")

    class DevicePool:
        pass

    class Snapshot:
        def __init__(self, rows) -> None:
            self.pending = True
            self.rows = rows

        def query(self):
            rows = self.rows
            self.rows = None
            self.pending = False
            return rows

    device_pool = DevicePool()
    progress_bridge = SglangHiCacheBridge(device_pool)

    class QueryEvent:
        def __init__(self, ready: bool = False) -> None:
            self.ready = ready

        def query(self) -> bool:
            return self.ready

    first_ready = QueryEvent()
    second_ready = QueryEvent()
    transfer_progress = PendingHostLoad(
        lease_id=90,
        consumer_index=6,
        host_indices=torch.tensor((1,)),
        device_indices=torch.tensor((2,)),
        producer_event=object(),
        controller=types.SimpleNamespace(layer_num=3),
        node_ids=(),
        operation_transfers=(LeaseOperationTransfer(70, 17, 1),),
        prefetched_layers={
            0: types.SimpleNamespace(
                key_bytes=10, value_bytes=10, ready_event=first_ready
            ),
            1: types.SimpleNamespace(
                key_bytes=20, value_bytes=20, ready_event=second_ready
            ),
        },
        layer_bytes=(20, 40, 60),
    )
    progress_bridge._pending[6] = transfer_progress
    progress = progress_bridge.progress(6)
    assert progress is not None
    assert (
        progress.published_layers,
        progress.leading_layers,
        progress.total_layers,
    ) == (
        2,
        0,
        3,
    )
    assert (progress.leading_bytes, progress.total_bytes) == (0, 120)
    first_ready.ready = True
    progress = progress_bridge.progress(6)
    assert progress is not None
    assert (progress.leading_layers, progress.leading_bytes) == (1, 20)
    second_ready.ready = True
    progress = progress_bridge.progress(6)
    assert progress is not None
    # Layer two was never published: a ready two-layer prefix is not complete.
    assert (progress.leading_layers, progress.total_layers) == (2, 3)
    assert not progress.complete

    from nta_runtime.engines.sglang import (
        _capacity_constrained_transfer_dependencies,
        _project_work_acquisitions,
        _resolve_request_acquisitions,
    )
    from nta_runtime.flashinfer_schedule import Schedule

    acquisitions = (
        SglangAcquisitionSpan(71, 41, 8, 32),
        SglangAcquisitionSpan.direct(),
        SglangAcquisitionSpan(72, 52, 0, 8),
    )
    transfers = {
        71: LeaseOperationTransfer(71, 41, 32),
        72: LeaseOperationTransfer(72, 52, 8),
    }
    assert (
        _resolve_request_acquisitions(acquisitions, transfers, lease_transfer_rows=40)
        == acquisitions
    )
    projected_dependencies = _project_work_acquisitions(
        Schedule(
            (0, 0, 0, 0, 0, 0, 1, 2),
            (0, 0, 1, 1, 2, 2, 0, 0),
            16,
            8,
        ),
        acquisitions,
        (40, 8, 8),
    )
    assert projected_dependencies == (
        LeaseWorkDependency(71, 0, 8),
        LeaseWorkDependency(71, 0, 8),
        LeaseWorkDependency(71, 8, 16),
        LeaseWorkDependency(71, 8, 16),
        LeaseWorkDependency(71, 24, 8),
        LeaseWorkDependency(71, 24, 8),
        None,
        LeaseWorkDependency(72, 0, 8),
    )
    assert _capacity_constrained_transfer_dependencies(
        projected_dependencies, maximum_groups=2
    ) == (
        LeaseWorkDependency(71, 0, 32),
        LeaseWorkDependency(71, 0, 32),
        LeaseWorkDependency(71, 0, 32),
        LeaseWorkDependency(71, 0, 32),
        LeaseWorkDependency(71, 0, 32),
        LeaseWorkDependency(71, 0, 32),
        None,
        LeaseWorkDependency(72, 0, 8),
    )

    class FakeEvent:
        def __init__(self) -> None:
            self.stream = None

        def record(self, stream) -> None:
            self.stream = stream

    class FakeTensor:
        is_cuda = False

    fake_stream = object()
    old_finish = object()
    Ack = namedtuple("Ack", ("start_event", "finish_event", "node_ids"))
    controller = types.SimpleNamespace(ack_load_queue=[])
    pending = types.SimpleNamespace(
        lease_id=91,
        consumer_index=7,
        held_ack=Ack(object(), old_finish, (11,)),
        controller=controller,
        host_indices=FakeTensor(),
        device_indices=FakeTensor(),
    )
    progress_bridge._pending[7] = pending
    progress_bridge._owned[91] = pending
    with patch("torch.cuda.Event", FakeEvent):
        assert progress_bridge.retire(pending, stream=fake_stream)
    assert not progress_bridge.retire(pending, stream=fake_stream)
    assert len(controller.ack_load_queue) == 1
    retired_ack = controller.ack_load_queue[0]
    assert retired_ack.finish_event is not old_finish
    assert retired_ack.finish_event.stream is fake_stream
    resident_id = stable_request_id("resident")
    binding = RequestBinding(0, 0, 4, resident_id, priority=3)
    blocked = RequestProgress(
        resident_id,
        4,
        1,
        1,
        0,
        0,
        0,
        0,
        7,
        4096,
        0,
        0,
        3000,
        3000,
        0,
    )
    progress_bridge.publish_request_progress(
        Snapshot((blocked,)),
        (binding,),
    )
    newer = replace(blocked, generation=5, unavailable_bytes=2048)
    newer_binding = replace(binding, generation=5)
    progress_bridge.publish_request_progress(
        Snapshot((newer,)),
        (newer_binding,),
    )
    frontier = progress_bridge.poll_request_frontier({resident_id})
    assert frontier is not None
    assert frontier.data_blocked == ((resident_id, 5),)
    assert frontier.requests[0].generation == 5
    assert frontier.executable == ()
    assert progress_bridge.poll_request_frontier({resident_id}) is None
    assert progress_bridge.admission_stats()["progress_feedback_consumed"] == 1
    progress_bridge.close()
    try:
        progress_bridge.publish_request_progress(
            Snapshot((blocked,)),
            (binding,),
        )
    except RuntimeError as error:
        assert "closed" in str(error)
    else:
        raise AssertionError("closed HiCache bridge accepted progress feedback")

    from nta_runtime.engines.sglang import (
        NtaFlashInferAttnBackend,
        _ActiveBatch,
        _NvmeSlotLifetime,
        _consumer_contract_for_stats,
        _demand_graph_key,
        _group_external_pages_by_request,
        _page_pairs_for_schedule,
        _pipeline_object_range,
        _plan_cache_signature,
        _require_exact_prefetch_layers,
        _flag_value,
    )
    from nta_runtime.resource_contract import ResourceCapability
    from nta_runtime.flashinfer_schedule import Schedule

    assert NtaFlashInferAttnBackend.__name__ == "NtaFlashInferAttnBackend"
    assert "_create_decode_wrappers" not in NtaFlashInferAttnBackend.__dict__
    assert "_create_prefill_wrappers" not in NtaFlashInferAttnBackend.__dict__

    adopted_schedule = Schedule((0,), (0,), 4, 1)
    adopted_page_pairs = {101: (((7,), (11,)),)}
    adopted_work_dependencies = {101: (None,)}
    adopted_transfer_dependencies = {101: (None,)}
    adopted_batch = _ActiveBatch(
        (),
        {101: adopted_schedule},
        None,
        adopted_page_pairs,
        {},
        {},
        (),
        work_dependencies=adopted_work_dependencies,
        transfer_dependencies=adopted_transfer_dependencies,
    )
    adopted_batch.adopt_wrapper_identity({101: 202})
    assert adopted_batch.schedules == {202: adopted_schedule}
    assert adopted_batch.page_pairs == {202: (((7,), (11,)),)}
    assert adopted_batch.work_dependencies == {202: (None,)}
    assert adopted_batch.transfer_dependencies == {202: (None,)}
    try:
        adopted_batch.adopt_wrapper_identity({303: 404})
    except RuntimeError as error:
        assert "does not cover its schedules" in str(error)
    else:
        raise AssertionError("wrapper adoption accepted stale source identity")

    class EventProbe:
        def __init__(self) -> None:
            self.streams = []

        def record(self, stream) -> None:
            self.streams.append(stream)

    consumer_event = EventProbe()
    nvme_slots = _NvmeSlotLifetime(consumer_event)
    assert nvme_slots.prior_consumer_event(0) is None
    nvme_slots.commit(((0, 4096), (1, 8192)))
    try:
        nvme_slots.prior_consumer_event(0)
    except RuntimeError as error:
        assert "prior-consumer event" in str(error)
    else:
        raise AssertionError("NVMe slot was replaced without a consumer proof")
    nvme_slots.record_consumer("attention-stream")
    assert nvme_slots.prior_consumer_event(0) is consumer_event
    assert nvme_slots.prior_consumer_event(1) is consumer_event
    nvme_slots.commit(((0, 4096), (1, 12288)))
    assert nvme_slots.previous(1) == 12288
    try:
        nvme_slots.prior_consumer_event(1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("NVMe predecessor proof was not single-use")
    nvme_slots.record_consumer("next-attention-stream")
    nvme_slots.commit(((2, 16384),))
    assert nvme_slots.prior_consumer_event(0) is consumer_event
    assert consumer_event.streams == ["attention-stream", "next-attention-stream"]
    assert (
        _require_exact_prefetch_layers(
            {0: object(), 1: object()}, 2, consumer="test consumer"
        )
        == 1
    )
    for malformed_layers in ({1: object()}, {0: object(), 2: object()}):
        try:
            _require_exact_prefetch_layers(
                malformed_layers, 2, consumer="test consumer"
            )
        except RuntimeError as error:
            assert "exact full-model prefetch" in str(error)
        else:
            raise AssertionError("graph consumer accepted incomplete layer readiness")

    class CloseProbe:
        def __init__(self, log, *, fail: bool = False) -> None:
            self.log = log
            self.fail = fail

        def close(self) -> None:
            self.log.append("close")
            if self.fail:
                raise RuntimeError("synthetic close failure")

    partial_log = []
    partial_backend = NtaFlashInferAttnBackend.__new__(NtaFlashInferAttnBackend)
    partial_backend._resources_closed = False
    partial_backend._hicache = CloseProbe(partial_log, fail=True)
    partial_backend._resources = CloseProbe(partial_log)
    partial_backend.__del__()
    assert partial_log == ["close", "close"]
    assert partial_backend._resources_closed

    assert (
        _flag_value(
            ResourceCapability.DIRECT_ADDRESS | ResourceCapability.HOST_REGISTERED
        )
        == 5
    )
    assert (
        _consumer_contract_for_stats({}, engine_version="0.5.16").kind.value
        == "projection_only"
    )
    assert (
        _consumer_contract_for_stats(
            {"stock_prefetched_external_attention_launches": 1},
            engine_version="0.5.16",
        ).kind.value
        == "framework_reference"
    )
    assert (
        _consumer_contract_for_stats(
            {"graph_external_batches": 1}, engine_version="0.5.16"
        ).kind.value
        == "framework_reference"
    )
    assert (
        _consumer_contract_for_stats(
            {
                "stock_prefetched_external_attention_launches": 1,
                "ticketed_incremental_launches": 1,
            },
            engine_version="0.5.16",
        ).kind.value
        == "native_work_unit"
    )
    assert _pipeline_object_range(128, 0, 12) == (104, 128)
    assert _pipeline_object_range(128, 1, 12) == (80, 104)
    for invalid in ((0, 0, 12), (128, -1, 12), (48, 1, 12)):
        try:
            _pipeline_object_range(*invalid)
        except RuntimeError:
            pass
        else:
            raise AssertionError("pipeline object allocator accepted overlap")
    signature = _plan_cache_signature(
        (0, 0), (3, 4), (((10,), (20,)),), (7,), 4096, 4096, None
    )
    remapped = _plan_cache_signature(
        (0, 0), (3, 4), (((11,), (21,)),), (7,), 4096, 4096, None
    )
    rebound = _plan_cache_signature(
        (0, 0), (3, 4), (((10,), (20,)),), (7,), 4096, 4096, None
    )
    reslotted = _plan_cache_signature(
        (0, 0), (3, 4), (((10,), (20,)),), (8,), 4096, 4096, None
    )
    assert signature != remapped, "plan cache aliased different HiCache page rows"
    assert signature == rebound, "request generation invalidated a structural plan"
    assert signature != reslotted, "plan cache aliased a different request slot"

    query = torch.empty((2, 4, 8), dtype=torch.float16)
    key_cache = torch.empty((16, 1), dtype=torch.float16)
    value_cache = torch.empty((16, 1), dtype=torch.float16)
    graph_wrapper = types.SimpleNamespace(
        _plan_info=(1, 2, 3),
        _qo_indptr_buf=torch.tensor((0, 2), dtype=torch.int32),
    )
    graph_plan = types.SimpleNamespace(
        work_items_address=0x1000,
        dependencies_address=0x2000,
    )
    graph_runtime = torch.empty(1, dtype=torch.uint8)
    graph_key_arguments = {
        "operator_family": "decode",
        "wrapper": graph_wrapper,
        "layer_id": 3,
        "plan": graph_plan,
        "runtime_tensor": graph_runtime,
        "work_count": 8,
        "object_count": 4,
        "progress_blocks": (2, 2),
        "ready_work_counts": (4, 8),
        "ready_work_offsets": (),
        "initial_ready_work_count": 0,
        "indexed_copy_blocks_per_group": 2,
        "query": query,
        "kv_cache": (key_cache, value_cache),
        "sm_scale": 0.125,
        "k_scale": 1.0,
        "v_scale": 1.0,
        "causal": False,
        "window_left": -1,
    }
    graph_key = _demand_graph_key(**graph_key_arguments)
    same_graph_key = _demand_graph_key(**graph_key_arguments)
    changed_graph_key = _demand_graph_key(
        **(graph_key_arguments | {"progress_blocks": (4,), "ready_work_counts": (8,)})
    )
    windowed_graph_key = _demand_graph_key(
        **(
            graph_key_arguments
            | {"ready_work_counts": (4, 4), "ready_work_offsets": (0, 4)}
        )
    )
    assert graph_key == same_graph_key
    assert graph_key != changed_graph_key
    assert graph_key != windowed_graph_key

    graph_backend = NtaFlashInferAttnBackend.__new__(NtaFlashInferAttnBackend)
    graph_backend._demand_graphs = {}
    graph_backend._demand_graph_warmups = {}
    graph_backend._demand_graph_capacity = 4
    graph_backend._stats = {
        "demand_graph_warmups": 0,
        "demand_graph_captures": 0,
        "demand_graph_replays": 0,
        "demand_graph_evictions": 0,
    }
    enqueue_calls = []

    def enqueue_graph(query_arg, output_arg, events_arg, callback_arg) -> None:
        enqueue_calls.append((query_arg, output_arg, events_arg, callback_arg))

    eager_events = (object(), (object(), object()))
    callback = object()
    eager_output = torch.empty_like(query)
    assert (
        graph_backend._enqueue_demand_graph(
            graph_key,
            graph_wrapper,
            query,
            eager_output,
            object(),
            enqueue_graph,
            eager_events,
            callback,
            None,
        )
        is eager_output
    )
    assert enqueue_calls == [(query, eager_output, eager_events, callback)]
    assert graph_backend._stats["demand_graph_warmups"] == 1

    class Event:
        pass

    class Graph:
        def __init__(self) -> None:
            self.replays = 0

        def replay(self) -> None:
            self.replays += 1

    with (
        patch("torch.cuda.Event", Event),
        patch("torch.cuda.CUDAGraph", Graph),
        patch("torch.cuda.graph", lambda *args, **kwargs: nullcontext()),
    ):
        captured_output = graph_backend._enqueue_demand_graph(
            graph_key,
            graph_wrapper,
            query,
            torch.empty_like(query),
            object(),
            enqueue_graph,
            eager_events,
            callback,
            None,
        )
    assert graph_backend._stats["demand_graph_captures"] == 1
    assert enqueue_calls[-1][3] is None
    assert len(enqueue_calls[-1][2][1]) == 2
    captured = graph_backend._demand_graphs[graph_key]
    assert captured_output is captured.output
    assert captured.graph.replays == 1
    assert graph_backend._stats["demand_graph_replays"] == 1
    graph_wrapper._qo_indptr_buf = torch.tensor((0, 7), dtype=torch.int32)
    replay_output = graph_backend._enqueue_demand_graph(
        graph_key,
        graph_wrapper,
        query.fill_(3),
        torch.empty_like(query),
        object(),
        enqueue_graph,
        eager_events,
        callback,
        None,
    )
    assert replay_output is captured.output
    assert captured.graph.replays == 2
    assert torch.equal(captured.query, query)
    assert torch.equal(captured.wrapper_metadata[0][1], graph_wrapper._qo_indptr_buf)
    assert graph_backend._stats["demand_graph_replays"] == 2
    graph_backend._discard_demand_graphs(graph_plan)
    assert graph_key not in graph_backend._demand_graphs
    assert graph_key not in graph_backend._demand_graph_warmups
    graph_backend._demand_graph_capacity = 1
    graph_backend._demand_graph_warmups[graph_key] = None
    graph_backend._demand_graphs[graph_key] = captured
    synchronized = []
    graph_backend._reserve_demand_graph_key(
        replace(graph_key, layer_id=4),
        types.SimpleNamespace(synchronize=lambda: synchronized.append(True)),
    )
    assert synchronized == [True]
    assert graph_key not in graph_backend._demand_graphs
    assert graph_backend._stats["demand_graph_evictions"] == 1

    class TensorGeometry:
        def __init__(self, elements: int, element_bytes: int) -> None:
            self._elements = elements
            self._element_bytes = element_bytes

        def __getitem__(self, index: int) -> "TensorGeometry":
            assert index == 0
            return self

        def numel(self) -> int:
            return self._elements

        def element_size(self) -> int:
            return self._element_bytes

    stock_backend = NtaFlashInferAttnBackend.__new__(NtaFlashInferAttnBackend)
    stock_backend._model_layer_count = 2
    stock_backend._stats = {
        "batches": 0,
        "hicache_external_batches": 0,
        "stock_prefetched_external_batches": 0,
    }
    prefetched = {0: object(), 1: object()}
    stock_pending = types.SimpleNamespace(
        prefetched_layers=prefetched,
        prefetch_tensors=(object(),),
        materialize_mapping=lambda: (_ for _ in ()).throw(
            AssertionError("complete prefetch materialized an unused CPU page map")
        ),
    )
    stock_binding = RequestBinding(0, 0, 1, stable_request_id("stock"))
    stock_backend._activate_stock_prefetch((stock_binding,), stock_pending)
    assert stock_backend._active_batch.page_pairs == {}
    assert stock_backend._active_batch.prefetched_layers is prefetched
    assert stock_backend._stats["stock_prefetch_metadata_fastpath_batches"] == 1
    try:
        stock_backend._activate_stock_prefetch(
            (stock_binding,),
            types.SimpleNamespace(prefetched_layers={0: object()}, prefetch_tensors=()),
        )
    except RuntimeError as error:
        assert "exact full-model prefetch" in str(error)
    else:
        raise AssertionError("partial prefetch entered the stock fast path")

    schedule = Schedule((0, 0, 1, 1), (0, 1, 0, 1), 4, 128)
    grouped = _group_external_pages_by_request(
        schedule,
        (
            ((10, 11), (20, 21)),
            ((11, 12), (21, 22)),
            ((), ()),
            ((30,), (40,)),
        ),
    )
    assert grouped == (
        ((10, 11, 12), (20, 21, 22)),
        ((10, 11, 12), (20, 21, 22)),
        ((), ()),
        ((30,), (40,)),
    )
    chunked_pairs = _page_pairs_for_schedule(
        Schedule((0, 0, 1), (0, 1, 0), 4, 3),
        indptr=(0, 8, 12),
        pages=(10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23),
        last_page=(1, 1),
        page_size=1,
        source_by_device={11: 101, 12: 102, 16: 106, 20: 200, 21: 201},
    )
    assert chunked_pairs == (
        ((101, 102), (11, 12)),
        ((106,), (16,)),
        ((200, 201), (20, 21)),
    )

    source = inspect.getsource(NtaFlashInferAttnBackend._upload_plan)
    assert "initial_runnable_tiles" in source
    assert "self._overlap_enabled and prefetched is None" in source
    assert "force_rounds" not in source
    try:
        _group_external_pages_by_request(
            schedule,
            (
                ((10,), (20,)),
                ((11,), (20,)),
                ((), ()),
                ((30,), (40,)),
            ),
        )
    except RuntimeError as error:
        assert "multiple host cache pages" in str(error)
    else:
        raise AssertionError("request grouping accepted an inconsistent page map")

    class Wrapper:
        def __init__(self) -> None:
            self.calls = 0
            self.arguments = ()

        def run(self, *args, **kwargs) -> None:
            self.calls += 1
            self.arguments = args

    backend = NtaFlashInferAttnBackend.__new__(NtaFlashInferAttnBackend)
    binding = types.SimpleNamespace(request_slot=0)
    backend._active_batch = _ActiveBatch((binding,), {}, None, {}, {}, {}, ())
    backend._runtime = types.SimpleNamespace(device_view_tensor=object())
    backend._wrapper_modules = {}
    backend._stats = {"transformed_direct_launches": 0}
    verified_wrappers = []
    backend._phase_program = lambda candidate: verified_wrappers.append(candidate)
    wrapper = Wrapper()
    try:
        backend._run_preacquired_attention(
            wrapper, object(), object(), object(), object(), {}
        )
    except RuntimeError as error:
        assert "compiler-transformed" in str(error)
    else:
        raise AssertionError("NTA attention accepted a stock wrapper")
    assert wrapper.calls == 0
    backend._wrapper_modules[id(wrapper)] = "instrumented_request_bound"
    layer = types.SimpleNamespace(scaling=0.5)
    backend._run_preacquired_attention(wrapper, object(), object(), object(), layer, {})
    assert wrapper.calls == 1
    assert len(wrapper.arguments) == 5
    assert wrapper.arguments[-1] == 0
    assert verified_wrappers == [wrapper]
    assert backend._stats["transformed_direct_launches"] == 1

    backend._active_batch = _ActiveBatch((), {}, object(), {}, {}, {}, ())
    backend._plans = {}
    try:
        backend._run_preacquired_attention(
            wrapper, object(), object(), object(), layer, {}
        )
    except RuntimeError as error:
        assert "validated CTA work plan" in str(error)
    else:
        raise AssertionError("external preacquired attention accepted no work plan")
    assert wrapper.calls == 1

    demand_wrapper = Wrapper()
    demand_schedule = types.SimpleNamespace(work_count=3)
    work_items_tensor = object()
    dependencies_tensor = object()
    demand_plan = types.SimpleNamespace(
        work_item_count=3,
        work_items_tensor=work_items_tensor,
        dependencies_tensor=dependencies_tensor,
        has_external=False,
        mark_consumed=lambda stream: None,
    )
    backend._wrapper_modules[id(demand_wrapper)] = "instrumented_demand_acquire"
    backend._active_batch = _ActiveBatch(
        (binding,), {id(demand_wrapper): demand_schedule}, object(), {}, {}, {}, ()
    )
    backend._plans = {(id(demand_wrapper), -1): types.SimpleNamespace(plan=demand_plan)}
    backend._run_preacquired_attention(
        demand_wrapper, object(), object(), object(), layer, {}
    )
    assert len(demand_wrapper.arguments) == 8
    assert demand_wrapper.arguments[3] is work_items_tensor
    assert demand_wrapper.arguments[4] is dependencies_tensor
    assert demand_wrapper.arguments[-2:] == (3, 6)

    context = mp.get_context("spawn")
    result = context.Queue()
    process = context.Process(target=load_in_spawn, args=(result,))
    process.start()
    process.join(30)
    assert process.exitcode == 0
    observed = result.get(timeout=1)
    assert observed[:2] == (True, 1), observed


if __name__ == "__main__":
    main()
