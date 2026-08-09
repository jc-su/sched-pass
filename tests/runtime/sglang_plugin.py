#!/usr/bin/env python3
"""Validate SGLang plugin registration in frontend and spawned workers."""

from __future__ import annotations

import inspect
import multiprocessing as mp
import types


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

    from nta_runtime.plugins.sglang import BACKEND_NAME, register

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
        _HICACHE_LOAD_TARGET,
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
        kind == HookType.AFTER
        for kind, _, _ in HookRegistry._hooks[_FORWARD_BATCH_TARGET]
    )
    assert any(
        kind == HookType.AROUND
        for kind, _, _ in HookRegistry._hooks[_PREFILL_ADMISSION_TARGET]
    )

    from nta_runtime.engines.sglang_admission import (
        AcquisitionAdmission,
        AdmissionConfig,
    )
    from nta_runtime.engines.sglang_hicache import HostLoadProgress

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

    from nta_runtime.engines.sglang import (
        NtaFlashInferAttnBackend,
        _ActiveBatch,
        _frontier_transfer_bytes,
        _group_external_pages_by_request,
        _pipeline_object_range,
        _plan_cache_signature,
    )
    from nta_runtime.flashinfer_schedule import Schedule

    assert NtaFlashInferAttnBackend.__name__ == "NtaFlashInferAttnBackend"
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
        (0, 0), (3, 4), (((10,), (20,)),), ((7, 3),), 4096, 4096, None
    )
    remapped = _plan_cache_signature(
        (0, 0), (3, 4), (((11,), (21,)),), ((7, 3),), 4096, 4096, None
    )
    rebound = _plan_cache_signature(
        (0, 0), (3, 4), (((10,), (20,)),), ((7, 3),), 4096, 4096, None
    )
    regenerated = _plan_cache_signature(
        (0, 0), (3, 4), (((10,), (20,)),), ((7, 4),), 4096, 4096, None
    )
    assert signature != remapped, "plan cache aliased different HiCache page rows"
    assert signature == rebound, "request rebinding invalidated a structural plan"
    assert signature != regenerated, "plan cache aliased a reused request generation"

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
    assert "direct_work_count if self._request_overlap else 0" in source
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
