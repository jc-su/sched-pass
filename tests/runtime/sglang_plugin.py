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
        _attach_request_priorities,
        _require_hooks_installed,
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
        kind == HookType.BEFORE for kind, _, _ in HookRegistry._hooks[_ABORT_TARGET]
    )
    assert any(
        kind == HookType.BEFORE for kind, _, _ in HookRegistry._hooks[_RELEASE_TARGET]
    )
    assert any(
        kind == HookType.BEFORE
        for kind, _, _ in HookRegistry._hooks[_REQUEST_FINISH_TARGET]
    )
    assert any(
        kind == HookType.AFTER
        for kind, _, _ in HookRegistry._hooks[_FORWARD_BATCH_TARGET]
    )
    assert any(
        kind == HookType.AROUND
        for kind, _, _ in HookRegistry._hooks[_PREFILL_ADMISSION_TARGET]
    )

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
    assert hook_forward_batch._nta_forward_metadata.request_slots == (11, 23)

    from nta_runtime.engines.sglang_admission import (
        AcquisitionAdmission,
        AdmissionConfig,
        route_prefill_admission,
    )
    from nta_runtime.engines.sglang_hicache import (
        HostLoadProgress,
        SglangHiCacheBridge,
    )
    from nta_runtime.tenant import tenant_budget_specs
    from nta_runtime.requests import RequestBinding, stable_request_id
    from nta_runtime.runtime import RequestProgress
    from nta_runtime.fixed_range_pool import FixedRangePool

    with patch.dict(
        "os.environ",
        {"NTA_TENANT_BUDGETS": "2:1048576:3,7:2097152"},
        clear=False,
    ):
        assert tenant_budget_specs() == ((2, 1048576, 3), (7, 2097152, 1))

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
            self.critical_plan = None

        def progress(self, consumer_index: int) -> HostLoadProgress:
            return HostLoadProgress(
                consumer_index, self.leading_layers, 12, self.leading_layers * 64, 768
            )

        def transfer_bytes(self, consumer_index: int) -> int:
            assert consumer_index == 3
            return 768

        def record_admission(self, **increments: int) -> None:
            for name, value in increments.items():
                self.stats[name] = self.stats.get(name, 0) + value

        def poll_critical_work(self, request_ids: set[int]):
            del request_ids
            result = self.critical_plan
            self.critical_plan = None
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

    bridge.critical_plan = types.SimpleNamespace(
        compute_order=(),
        data_order=((stable_request_id("resident"), 1),),
    )
    admission = AcquisitionAdmission(config, clock=lambda: clock[0])
    assert admission.consider(scheduler, batch, bridge) is batch
    assert bridge.stats["admission_feedback_data_blocked"] == 1
    assert bridge.stats["admission_released_feedback_data_blocked"] == 1

    bridge.critical_plan = types.SimpleNamespace(
        compute_order=((stable_request_id("resident"), 1),),
        data_order=(),
    )
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
        bandwidth_bytes_per_second=10_000_000_000,
    )
    newer = replace(blocked, generation=5, unavailable_bytes=2048)
    newer_binding = replace(binding, generation=5)
    progress_bridge.publish_request_progress(
        Snapshot((newer,)),
        (newer_binding,),
        bandwidth_bytes_per_second=10_000_000_000,
    )
    critical = progress_bridge.poll_critical_work({resident_id})
    assert critical is not None
    assert critical.data_order == ((resident_id, 5),)
    assert critical.requests[0].request.generation == 5
    assert critical.compute_order == ()
    assert progress_bridge.poll_critical_work({resident_id}) is None
    assert progress_bridge.admission_stats()["progress_feedback_consumed"] == 1
    progress_bridge.close()
    try:
        progress_bridge.publish_request_progress(
            Snapshot((blocked,)),
            (binding,),
            bandwidth_bytes_per_second=10_000_000_000,
        )
    except RuntimeError as error:
        assert "closed" in str(error)
    else:
        raise AssertionError("closed HiCache bridge accepted progress feedback")

    from nta_runtime.engines.sglang import (
        NtaFlashInferAttnBackend,
        _ActiveBatch,
        _consumer_contract_for_stats,
        _demand_graph_key,
        _frontier_transfer_bytes,
        _group_external_pages_by_request,
        _pipeline_object_range,
        _plan_cache_signature,
        _flag_value,
    )
    from nta_runtime.resource_contract import ResourceCapability
    from nta_runtime.flashinfer import (
        request_ranges_for_schedule as _request_ranges,
    )
    from nta_runtime.flashinfer_schedule import Schedule

    assert NtaFlashInferAttnBackend.__name__ == "NtaFlashInferAttnBackend"

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
            {
                "stock_prefetched_external_attention_launches": 1,
                "ticketed_incremental_launches": 1,
            },
            engine_version="0.5.16",
        ).kind.value
        == "native_work_unit"
    )
    ranges = _request_ranges(
        (
            RequestBinding(0, 5, 1, stable_request_id("r0")),
            RequestBinding(1, 9, 1, stable_request_id("r1")),
        ),
        (0, 0, 1),
    )
    assert [
        (item.work_begin, item.work_count, item.request_slot) for item in ranges
    ] == [
        (0, 2, 5),
        (2, 1, 9),
    ]
    for malformed in ((0, 1, 0), (0, 0)):
        try:
            _request_ranges(
                (
                    RequestBinding(0, 5, 1, stable_request_id("r0")),
                    RequestBinding(1, 9, 1, stable_request_id("r1")),
                ),
                malformed,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("malformed request schedule was accepted")

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
    assert graph_key == same_graph_key
    assert graph_key != changed_graph_key

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

    host_pool = types.SimpleNamespace(
        k_data_refs=[TensorGeometry(128, 2), TensorGeometry(128, 2)],
        v_data_refs=[TensorGeometry(128, 2), TensorGeometry(128, 2)],
    )
    pending = types.SimpleNamespace(
        host_indices=TensorGeometry(64, 8),
        controller=types.SimpleNamespace(mem_pool_host=host_pool),
    )
    assert _frontier_transfer_bytes(pending) == 64 * 2 * 2 * 128 * 2
    host_pool.v_data_refs.pop()
    try:
        _frontier_transfer_bytes(pending)
    except RuntimeError as error:
        assert "K/V layer counts disagree" in str(error)
    else:
        raise AssertionError("frontier accepted mismatched K/V layer counts")
    host_pool.v_data_refs.append(TensorGeometry(128, 2))
    frontier_bytes = _frontier_transfer_bytes(pending)
    frontier_backend = NtaFlashInferAttnBackend.__new__(NtaFlashInferAttnBackend)
    frontier_backend._stats = {}
    prepared = []
    frontier_backend._prepare_cross_layer_frontier = prepared.append
    frontier_backend._publish_cross_layer_frontier(pending)
    assert prepared == [pending]
    assert frontier_backend._stats["frontier_proactive_batches"] == 1
    assert frontier_backend._stats["frontier_published_bytes"] == frontier_bytes

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
    source = inspect.getsource(NtaFlashInferAttnBackend._upload_plan)
    assert "initial_runnable_tiles" in source
    assert "direct_work_count if self._overlap_enabled else 0" in source
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
