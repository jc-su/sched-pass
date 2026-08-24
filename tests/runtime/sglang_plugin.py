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
        _retire_finished_request,
        register,
    )

    register()
    register()
    assert BACKEND_NAME in ATTENTION_BACKENDS
    assert ATTENTION_BACKEND_CHOICES.count(BACKEND_NAME) == 1
    assert callable(ATTENTION_BACKENDS[BACKEND_NAME])

    from sglang.srt.model_executor.runner import decode_cuda_graph_runner

    assert getattr(
        decode_cuda_graph_runner.build_replay_fb_view,
        "_nta_preserves_request_metadata",
        False,
    )

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from nta_runtime.plugins.sglang import (
        _ABORT_TARGET,
        _EXTERNAL_ADMISSION_TARGET,
        _HICACHE_LOAD_TARGET,
        _REQUEST_FINISH_TARGET,
        _RELEASE_TARGET,
        _FORWARD_BATCH_TARGET,
        _PREFILL_ADMISSION_TARGET,
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
    assert any(
        kind == HookType.AROUND
        for kind, _, _ in HookRegistry._hooks[_EXTERNAL_ADMISSION_TARGET]
    )
    from nta_runtime.engines.sglang_admission import (
        AcquisitionAdmission,
        AdmissionConfig,
    )
    from nta_runtime.engines.sglang_hicache import (
        HostLoadProgress,
        SglangHiCacheBridge,
    )
    from nta_runtime.requests import RequestBinding, stable_request_id
    from nta_runtime.runtime import RequestProgress
    from nta_runtime.fixed_range_pool import FixedRangePool

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
    critical = progress_bridge.poll_critical_work({resident_id})
    assert critical is not None
    assert critical.data_order == ((resident_id, 4),)
    assert critical.compute_order == ()
    assert progress_bridge.poll_critical_work({resident_id}) is None
    assert progress_bridge.admission_stats()["progress_feedback_consumed"] == 1

    from nta_runtime.engines.sglang import (
        NtaFlashInferAttnBackend,
        _ActiveBatch,
        _demand_graph_key,
        _frontier_transfer_bytes,
        _group_external_pages_by_request,
        _pipeline_object_range,
        _plan_cache_signature,
    )
    from nta_runtime.flashinfer_schedule import Schedule

    assert NtaFlashInferAttnBackend.__name__ == "NtaFlashInferAttnBackend"

    from nta_runtime.engines.sglang_external import (
        VIRTUAL_TOKEN_BASE,
        route_allocator_free,
        route_cache_finished,
        route_external_admission_credit,
        route_init_load_back,
    )

    class ExternalDevicePool:
        pass

    class ExternalAllocator:
        def __init__(self, device_pool) -> None:
            self._kvcache = device_pool
            self.allocations = []
            self.frees = []
            self.attempts = 0

        def alloc(self, count):
            self.attempts += 1
            if self.attempts == 1:
                return None
            rows = torch.arange(100, 100 + count, dtype=torch.int64)
            self.allocations.append(rows.clone())
            return rows

        def available_size(self):
            return 0

        def free(self, rows):
            self.frees.append(rows.clone())

    class ExternalNode:
        def __init__(self, node_id, host_value, parent, evicted=True) -> None:
            self.id = node_id
            self.host_value = host_value
            self.parent = parent
            self.evicted = evicted
            self.protected = 0

        def protect_host(self):
            self.protected += 1

        def release_host(self):
            self.protected -= 1

    external_pool = ExternalDevicePool()
    external_allocator = ExternalAllocator(external_pool)
    external_controller = types.SimpleNamespace(
        mem_pool_device=external_pool,
        mem_pool_device_allocator=external_allocator,
        device="cpu",
    )
    external_bridge = SglangHiCacheBridge(external_pool)
    captured_handles = []
    external_bridge.enable_external_prefixes(4, captured_handles.append)
    resident_node = ExternalNode(1, None, None, evicted=False)
    host_node = ExternalNode(
        2, torch.arange(9, 15, dtype=torch.int64), resident_node
    )
    external_request = types.SimpleNamespace(
        rid="external-request",
        prefix_indices=torch.tensor((1, 2), dtype=torch.int64),
        last_node=resident_node,
        needs_host_load_back=lambda: True,
        swa_host_hit_length=0,
        mamba_host_hit_length=0,
        host_hit_length=6,
    )
    external_evictions = []

    def evict_external(params):
        external_evictions.append(params.num_tokens)
        return types.SimpleNamespace(num_tokens_evicted=params.num_tokens)

    external_cache = types.SimpleNamespace(
        cache_controller=external_controller,
        page_size=1,
        evict=evict_external,
    )
    external_params = types.SimpleNamespace(
        req=external_request,
        best_match_node=host_node,
        host_hit_length=6,
    )
    virtual, last_node = route_init_load_back(
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dense load-back executed")
        ),
        external_cache,
        external_params,
    )
    assert last_node is resident_node
    assert virtual.tolist() == list(range(VIRTUAL_TOKEN_BASE, VIRTUAL_TOKEN_BASE + 6))
    assert external_evictions == [4]
    assert external_allocator.allocations[0].numel() == 4
    assert host_node.protected == 1 and len(captured_handles) == 1
    external_stats = external_bridge.admission_stats()
    assert external_stats["external_live_dense_rows"] == 6
    assert external_stats["external_live_staging_rows"] == 4
    assert external_stats["external_dense_high_water_rows"] == 6
    assert external_stats["external_staging_high_water_rows"] == 4
    finish_reasons = []
    captured_handles[0].retire_callback = (
        lambda reason: finish_reasons.append(reason) is None
    )
    _retire_finished_request(
        types.SimpleNamespace(),
        types.SimpleNamespace(
            rid="external-request",
            finished=lambda: True,
            _nta_external_prefix=captured_handles[0],
        ),
    )
    assert finish_reasons == ["finished"]
    # The cache-release hook is the authoritative retirement edge. It is
    # idempotent when an earlier result hook already completed the lease.
    captured_handles[0]._released = True
    request_finish_calls = []
    external_request.cache_protected_len = 6
    route_cache_finished(
        lambda _cache, _request, is_insert=True: request_finish_calls.append(
            (_request.cache_protected_len, is_insert)
        ),
        external_cache,
        external_request,
    )
    assert request_finish_calls == [(2, False)]
    captured_handles[0]._released = False
    admission_offsets = []
    adder = types.SimpleNamespace(
        tree_cache=external_cache, rem_total_token_offset=50
    )

    def admit(fake_adder, _request):
        admission_offsets.append(fake_adder.rem_total_token_offset)
        fake_adder.rem_total_token_offset += 3
        return "admitted"

    assert (
        route_external_admission_credit(admit, adder, external_request)
        == "admitted"
    )
    assert admission_offsets == [44]
    assert adder.rem_total_token_offset == 53
    physical_frees = []
    route_allocator_free(
        lambda _allocator, rows: physical_frees.append(rows.clone()),
        external_allocator,
        torch.tensor((7, VIRTUAL_TOKEN_BASE, 8), dtype=torch.int64),
    )
    assert physical_frees[0].tolist() == [7, 8]
    captured_handles[0].release_resources()
    assert external_allocator.frees[-1].tolist() == [100, 101, 102, 103]
    assert host_node.protected == 0
    external_stats = external_bridge.admission_stats()
    assert external_stats["external_live_dense_rows"] == 0
    assert external_stats["external_live_staging_rows"] == 0
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
        **(
            graph_key_arguments
            | {"progress_blocks": (4,), "ready_work_counts": (8,)}
        )
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
    )
    backend._wrapper_modules[id(demand_wrapper)] = "instrumented_demand_acquire"
    backend._active_batch = _ActiveBatch(
        (binding,), {id(demand_wrapper): demand_schedule}, object(), {}, {}, {}, ()
    )
    backend._plans = {
        (id(demand_wrapper), -1): types.SimpleNamespace(plan=demand_plan)
    }
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
