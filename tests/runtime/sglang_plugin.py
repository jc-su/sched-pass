#!/usr/bin/env python3
"""Validate SGLang plugin registration in frontend and spawned workers."""

from __future__ import annotations

import inspect
import multiprocessing as mp
import os
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
    from nta_runtime.engines.sglang_planning import (
        byte_scale_bucket as _byte_scale_bucket,
        calibration_probe_end as _calibration_probe_end,
        demand_overlap_policy as _demand_overlap_policy,
        maximum_mover_wave_bytes as _maximum_mover_wave_bytes,
        mover_layout_required as _mover_layout_required,
        requires_feasible_edf as _requires_feasible_edf,
    )
    from nta_runtime.engines.sglang_materialization import SglangPlanMaterializer
    from nta_runtime.engines.sglang_metadata import SglangMetadataPlanner
    from nta_runtime.engines.sglang_transfer import HostMoverController, MoverProfile
    from nta_runtime.execution_planner import HostExecutionMode
    from nta_runtime.indexed_transfer import IndexedMoverServiceModel
    from nta_runtime.requests import RequestBinding

    binding = RequestBinding(
        request_index=0,
        request_slot=3,
        generation=1,
        request_id=7,
    )
    direct_stats = {
        "multi_request_engine_batches": 0,
        "heterogeneous_engine_batches": 0,
        "availability_heterogeneous_batches": 0,
        "sequence_length_heterogeneous_batches": 0,
        "multi_axis_heterogeneous_batches": 0,
    }
    direct_planner = SglangMetadataPlanner.__new__(SglangMetadataPlanner)
    direct_planner._stats = direct_stats
    direct_planner._record_direct_mixed_heterogeneity(
        types.SimpleNamespace(
            forward_mode=types.SimpleNamespace(is_mixed=lambda: True),
            seq_lens_cpu=(100, 120),
        ),
        (
            binding,
            RequestBinding(
                request_index=1,
                request_slot=4,
                generation=1,
                request_id=8,
            ),
        ),
    )
    assert direct_stats == {
        "multi_request_engine_batches": 1,
        "heterogeneous_engine_batches": 1,
        "availability_heterogeneous_batches": 1,
        "sequence_length_heterogeneous_batches": 1,
        "multi_axis_heterogeneous_batches": 1,
    }
    model_snapshot = object()
    direct_candidate = object()
    fastpath_planner = SglangMetadataPlanner.__new__(SglangMetadataPlanner)
    fastpath_planner._tier_service = types.SimpleNamespace(is_host_staged=True)
    fastpath_planner._tenant_isolation_enabled = False
    fastpath_planner._execution_config = types.SimpleNamespace(
        host_execution_mode=HostExecutionMode.AUTO
    )
    with patch(
        "nta_runtime.engines.sglang_metadata.prove_direct_metadata_execution",
        return_value=direct_candidate,
    ) as direct_proof:
        assert (
            fastpath_planner._bounded_direct_plan(
                {7: object()},
                object(),
                (binding,),
                host_cost_model=model_snapshot,
                calibration_probe=False,
            )
            is direct_candidate
        )
        assert direct_proof.call_args.kwargs["model"] is model_snapshot
        direct_proof.reset_mock()
        assert (
            fastpath_planner._bounded_direct_plan(
                {7: object()},
                object(),
                (binding,),
                host_cost_model=model_snapshot,
                calibration_probe=True,
            )
            is None
        )
        direct_proof.assert_not_called()
    assert not _requires_feasible_edf((binding,), tenant_isolation=False)
    assert _requires_feasible_edf((binding,), tenant_isolation=True)
    assert _requires_feasible_edf(
        (
            binding,
            RequestBinding(
                request_index=1,
                request_slot=4,
                generation=1,
                request_id=8,
            ),
        ),
        tenant_isolation=False,
    )
    try:
        _requires_feasible_edf((), tenant_isolation=False)
    except ValueError:
        pass
    else:
        raise AssertionError("empty acquisition binding set was accepted")

    assert _demand_overlap_policy(
        host_staged=True,
        frontier_enabled=True,
        graph_requested=True,
    ) == (True, False, "finite_demand_graph")
    assert _demand_overlap_policy(
        host_staged=True,
        frontier_enabled=True,
        graph_requested=False,
    ) == (False, True, "first_wave_lookahead")
    assert _demand_overlap_policy(
        host_staged=False,
        frontier_enabled=False,
        graph_requested=True,
    ) == (False, False, "none")

    assert _byte_scale_bucket(1) == 0
    assert _byte_scale_bucket(1 << 20) == 20
    assert _byte_scale_bucket((1 << 20) + 1) == 20
    assert (
        _maximum_mover_wave_bytes(
            ((2, 3), (5, 7), (11, 13)), transfer_count=4, layers_per_wave=2
        )
        == 96
    )
    for invalid in (0, -1):
        try:
            _byte_scale_bucket(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid mover scale was accepted")
    assert _calibration_probe_end(1, 36, 4) == 5
    assert _calibration_probe_end(34, 36, 4) == 36
    for invalid in ((-1, 36, 4), (36, 36, 4), (0, 0, 4), (0, 36, 0)):
        try:
            _calibration_probe_end(*invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid mover calibration frontier was accepted")
    assert not _mover_layout_required("sm", False)
    assert _mover_layout_required("sm", True)
    assert _mover_layout_required("probe_copy", False)
    assert _mover_layout_required("copy_engine", False)
    assert _mover_layout_required("auto", False)
    try:
        _mover_layout_required("invalid", False)
    except ValueError as error:
        assert "unknown" in str(error)
    else:
        raise AssertionError("unknown mover policy was accepted")
    from nta_runtime.plugins.sglang import (
        BACKEND_NAME,
        STATS_SNAPSHOT_RPC_METHOD,
        _CONTROL_RPC_TARGET,
        _EXECUTE_DECODE_TARGET,
        _EXECUTE_EXTEND_TARGET,
        _EAGER_LOAD_BATCH_TARGET,
        _PREFILL_FINISH_TARGET,
        _PREBUILT_FINISH_TARGET,
        _attach_request_priorities,
        _require_hooks_installed,
        _retire_prefill_finished_requests,
        _retire_finished_request,
        _route_stats_snapshot_rpc,
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
        assert any(kind == HookType.AFTER for kind, _, _ in HookRegistry._hooks[target])
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
    assert any(
        kind == HookType.AROUND
        for kind, _, _ in HookRegistry._hooks[_CONTROL_RPC_TARGET]
    )

    # Measurement boundaries are pure control traffic. They publish a
    # quiescent snapshot without manufacturing a resident/external inference
    # request, and unrelated SGLang RPC methods retain the upstream handler.
    published = []
    rpc_backend = types.SimpleNamespace(
        _publish_stats=lambda **kwargs: published.append(kwargs)
    )
    rpc_scheduler = types.SimpleNamespace(
        tp_worker=types.SimpleNamespace(
            model_runner=types.SimpleNamespace(attn_backend=rpc_backend)
        ),
        tp_group=types.SimpleNamespace(cpu_group=object()),
    )
    original_rpcs = []
    with patch("torch.distributed.barrier") as rpc_barrier:
        output = _route_stats_snapshot_rpc(
            lambda _scheduler, request: original_rpcs.append(request.method),
            rpc_scheduler,
            types.SimpleNamespace(method=STATS_SNAPSHOT_RPC_METHOD),
        )
    assert output.success and output.message == ""
    assert published == [{"observation_boundary": True, "wait": True}]
    assert not original_rpcs
    rpc_barrier.assert_called_once_with(group=rpc_scheduler.tp_group.cpu_group)
    _route_stats_snapshot_rpc(
        lambda _scheduler, request: original_rpcs.append(request.method),
        rpc_scheduler,
        types.SimpleNamespace(method="save_remote_model"),
    )
    assert original_rpcs == ["save_remote_model"]

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
        req_pool_indices=types.SimpleNamespace(
            tolist=lambda: (_ for _ in ()).throw(
                AssertionError("request identity synchronized the device slot tensor")
            )
        ),
    )
    hook_batch = types.SimpleNamespace(
        reqs=(
            types.SimpleNamespace(priority=0, req_pool_idx=11),
            types.SimpleNamespace(priority=0, req_pool_idx=23),
        )
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
    assert hook_forward_batch._nta_forward_metadata.tenant_ids == (0, 0)
    assert (
        hook_forward_batch._nta_forward_metadata.acquisitions
        == (SglangAcquisitionSpan.direct(),) * 2
    )

    external_request = types.SimpleNamespace(
        rid="external",
        priority=0,
        req_pool_idx=7,
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
        req_pool_idx=9,
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

    tenant_forward = types.SimpleNamespace(
        batch_size=2,
        rids=("team-a/request-0", "team-b/request-1"),
        req_pool_indices=torch.tensor((13, 17), dtype=torch.int32),
    )
    tenant_batch = types.SimpleNamespace(
        reqs=(
            types.SimpleNamespace(priority=0, req_pool_idx=13),
            types.SimpleNamespace(priority=0, req_pool_idx=17),
        )
    )
    with patch(
        "nta_runtime.plugins.sglang._configured_tenant_mapper",
        return_value=lambda request_id: (7 if request_id.startswith("team-a/") else 11),
    ):
        _attach_request_priorities(
            tenant_forward,
            object,
            tenant_batch,
            hook_runner,
        )
    assert tenant_forward._nta_forward_metadata.tenant_ids == (7, 11)

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
    from nta_runtime.engines.sglang_contracts import (
        LeaseAcquisitionGroup,
        LeaseAcquisitionSlice,
        LeaseOperationRange,
        LeaseOperationTransfer,
    )
    from nta_runtime.engines.sglang_hicache import (
        HostLoadProgress,
        PendingHostLoad,
        SglangHiCacheBridge,
    )
    from nta_runtime.tenant import tenant_budget_specs
    from nta_runtime.progress_frontier import FrontierState
    from nta_runtime.requests import RequestBinding, stable_request_id
    from nta_runtime.runtime import RequestProgress
    from nta_runtime.acquisition_scheduler import (
        AcquisitionServiceCurve,
        LayerAcquisitionModel,
    )
    from nta_runtime.acquisition_scheduler import LayerAcquisition
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

        def deadline_model(self, consumer_index: int, batch: object):
            assert consumer_index == 3
            assert batch is not None
            return LayerAcquisitionModel(
                layer_bytes=(64,) * 12,
                transfer_service_ns=(10,) * 12,
                initial_compute_ns=0,
                inter_layer_compute_ns=20,
            )

        def prepare_admission_acquisition(
            self, consumer_index: int, batch: object
        ) -> bool:
            del consumer_index, batch
            return False

        def start_admission_acquisition(
            self, consumer_index: int, batch: object
        ) -> None:
            raise AssertionError("unprepared admission acquisition was started")

        def poll_request_frontier(self, request_ids: set[int]):
            del request_ids
            result = self.frontier
            self.frontier = None
            return result

    clock = [1_000]
    config = AdmissionConfig(True, 100)
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
    assert bridge.stats["admission_released_feasible"] == 1
    assert bridge.stats["admission_feasibility_infeasible"] >= 1
    assert bridge.stats["admission_feasibility_feasible"] == 1

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

    class PreparedBridge(Bridge):
        def __init__(self) -> None:
            super().__init__()
            self.prepared = False
            self.started = False

        def progress(self, consumer_index: int) -> HostLoadProgress:
            return HostLoadProgress(
                consumer_index=consumer_index,
                published_layers=12 if self.started else 0,
                leading_layers=self.leading_layers if self.started else 0,
                total_layers=12,
                leading_bytes=self.leading_layers * 64 if self.started else 0,
                total_bytes=768,
            )

        def prepare_admission_acquisition(
            self, consumer_index: int, batch: object
        ) -> bool:
            assert consumer_index == 3 and batch is not None
            self.prepared = True
            return True

        def start_admission_acquisition(
            self, consumer_index: int, batch: object
        ) -> None:
            assert self.prepared and consumer_index == 3 and batch is not None
            self.started = True

    prepared_bridge = PreparedBridge()
    admission = AcquisitionAdmission(config, clock=lambda: clock[0])
    assert admission.consider(scheduler, batch, prepared_bridge) is None
    assert prepared_bridge.prepared and prepared_bridge.started
    prepared_bridge.leading_layers = 1
    clock[0] += 1
    assert admission.poll(scheduler) is batch
    assert prepared_bridge.stats["admission_released_feasible"] == 1

    capped_bridge = PreparedBridge()
    admission = AcquisitionAdmission(AdmissionConfig(True, 5), clock=lambda: clock[0])
    assert admission.consider(scheduler, batch, capped_bridge) is batch
    assert capped_bridge.prepared and not capped_bridge.started
    assert capped_bridge.stats["admission_released_partial_slo"] == 1

    bridge.leading_layers = 0
    admission = AcquisitionAdmission(config, clock=lambda: clock[0])
    assert admission.consider(scheduler, batch, bridge) is None
    clock[0] += 101
    assert admission.poll(scheduler) is batch
    assert bridge.stats["admission_released_slo_cap"] == 1

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
        AdmissionConfig(True, 100), clock=lambda: route_clock[0]
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

    # Acquiring a lease has a prepare/commit window outside the bridge lock.
    # Closing in that window must fail the commit and return the held SGLang
    # acknowledgement exactly once; a recycled producer slot must do likewise.
    from sglang.srt.managers.cache_controller import CacheOperation

    class ProducerEvent:
        start_event = object()
        finish_event = object()

    class ProducerCounter:
        def __init__(self) -> None:
            self.events = (ProducerEvent(),)

        @staticmethod
        def update_producer() -> int:
            return 0

    def lease_controller(pool):
        host_pool = types.SimpleNamespace(
            layout="layer_first",
            pin_memory=True,
            k_data_refs=(torch.empty((1, 1)),),
            v_data_refs=(torch.empty((1, 1)),),
        )
        return types.SimpleNamespace(
            mem_pool_device=pool,
            mem_pool_host=host_pool,
            has_draft=False,
            io_backend="kernel",
            page_size=1,
            layer_done_counter=ProducerCounter(),
            load_queue=[CacheOperation(torch.tensor((1,)), torch.tensor((2,)), 7)],
            ack_load_queue=[],
        )

    close_race_pool = DevicePool()
    close_race_pool._get_key_buffer = lambda *_args: None
    close_race_pool._get_value_buffer = lambda *_args: None
    close_race_bridge = SglangHiCacheBridge(close_race_pool)
    close_race_bridge.set_acquire_callback(lambda _pending: None)
    close_race_controller = lease_controller(close_race_pool)

    def close_during_prepare(_pool, _indices):
        close_race_bridge.close()
        return None

    with patch(
        "nta_runtime.connectors.sglang_storage.maybe_resolve_sglang_storage_keys",
        close_during_prepare,
    ):
        try:
            close_race_bridge.acquire_load(close_race_controller)
        except RuntimeError as error:
            assert "closed while acquiring" in str(error)
        else:
            raise AssertionError("HiCache close/acquire race committed a dead lease")
    assert len(close_race_controller.ack_load_queue) == 1

    reused_pool = DevicePool()
    reused_pool._get_key_buffer = lambda *_args: None
    reused_pool._get_value_buffer = lambda *_args: None
    reused_bridge = SglangHiCacheBridge(reused_pool)
    reused_bridge.set_acquire_callback(lambda _pending: None)
    reused_controller = lease_controller(reused_pool)
    reused_bridge._pending[0] = object()
    try:
        reused_bridge.acquire_load(reused_controller)
    except RuntimeError as error:
        assert "reused a live" in str(error)
    else:
        raise AssertionError("HiCache overwrote a live producer slot")
    assert len(reused_controller.ack_load_queue) == 1
    reused_bridge.close()

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
    )
    from nta_runtime.execution_protocol import ProtocolKind
    from nta_runtime.engines.sglang_state import (
        _ActiveBatch,
    )
    from nta_runtime.engines.sglang_graphs import (
        DemandGraphCache,
        demand_graph_key,
    )
    from nta_runtime.engines.sglang_kernels import (
        SglangKernelConfig,
        SglangKernelResources,
        SglangWrapperSet,
    )
    from nta_runtime.engines.sglang_config import (
        SglangObservabilityConfig,
        SglangVerificationConfig,
        incremental_calibration_probe_count,
    )
    from nta_runtime.engines.sglang_calibration import (
        SglangLayerServiceCalibration,
    )
    from nta_runtime.engines.sglang_execution import SglangAttentionExecutor
    from nta_runtime.engines.sglang_semantics import (
        build_semantic_wrapper_plan,
        prove_direct_metadata_execution,
    )
    from nta_runtime.engines.sglang_planning import (
        pipeline_object_id as _pipeline_object_id,
        pipeline_object_range as _pipeline_object_range,
        require_exact_prefetch_layers as _require_exact_prefetch_layers,
        semantic_plan_signature_prefix as _semantic_plan_signature_prefix,
    )
    from nta_runtime.engines.sglang_telemetry import (
        consumer_contract_for_stats as _consumer_contract_for_stats,
        flag_value as _flag_value,
    )
    from nta_runtime.nvme_materialization import NvmeSlotLifetime
    from nta_runtime.resource_contract import ResourceCapability
    from nta_runtime.flashinfer_schedule import Schedule

    assert NtaFlashInferAttnBackend.__name__ == "NtaFlashInferAttnBackend"
    assert SglangLayerServiceCalibration.shape_key(
        types.SimpleNamespace(reqs=(object(), object()), extend_num_tokens=7)
    ) == ("extend", 7, 2)
    assert SglangLayerServiceCalibration.shape_key(
        types.SimpleNamespace(batch_size=2, extend_num_tokens=7)
    ) == ("extend", 7, 2)
    assert SglangLayerServiceCalibration.shape_key(
        types.SimpleNamespace(batch_size=0, rids=(1, 2), extend_num_tokens=7)
    ) == ("extend", 7, 2)
    assert (
        SglangLayerServiceCalibration.shape_key(
            types.SimpleNamespace(batch_size=2, extend_num_tokens=0)
        )
        is None
    )

    # A dense exact lease starts transport when physical ownership is captured,
    # before ForwardBatch metadata or a calibrated EDF proof exists. Tenant
    # isolation remains deferred because request accounting is not yet bound.
    lease_pool = object()
    lease_pending = types.SimpleNamespace(
        controller=types.SimpleNamespace(mem_pool_device=lease_pool, layer_num=4),
        layer_bytes=(),
        prefetched_layers={},
        transfer_plan=None,
        acquisition=None,
    )
    lease_ranges: list[tuple[int, int]] = []
    lease_backend = NtaFlashInferAttnBackend.__new__(NtaFlashInferAttnBackend)
    lease_backend.token_to_kv_pool = lease_pool
    lease_backend._model_layer_count = 4
    lease_backend._tenant_isolation_enabled = False
    lease_backend._execution_config = types.SimpleNamespace(
        protocol=types.SimpleNamespace(kind=ProtocolKind.LATE_BOUND),
        host_execution_mode=HostExecutionMode.AUTO,
    )
    lease_backend._stats = {}

    def account_lease(pending) -> None:
        pending.layer_bytes = (1024,) * 4

    def plan_lease(pending, **_kwargs):
        pending.transfer_plan = object()
        return pending.transfer_plan

    def publish_lease(
        pending,
        *,
        first_local_layer: int,
        last_local_layer: int,
    ) -> None:
        lease_ranges.append((first_local_layer, last_local_layer))
        for local_layer in range(first_local_layer, last_local_layer):
            pending.prefetched_layers[local_layer] = object()

    lease_backend._account_tier_selection = account_lease
    lease_backend._host_transfer_lease_plan = plan_lease
    lease_backend._host_transport = types.SimpleNamespace(prepare=publish_lease)
    lease_backend._hold_host_load(lease_pending)
    assert lease_ranges == [(0, 4)]
    assert lease_pending.acquisition.started
    assert lease_pending.acquisition.model is None
    assert lease_pending.acquisition.fully_published
    assert lease_backend._stats["initial_acquisition_layers"] == 4
    assert lease_backend._stats["initial_typed_gap_layers"] == 0
    assert lease_backend._stats["lease_acquisition_groups_started"] == 1

    isolated_pending = types.SimpleNamespace(
        controller=types.SimpleNamespace(mem_pool_device=lease_pool, layer_num=4),
        layer_bytes=(),
        prefetched_layers={},
        transfer_plan=None,
        acquisition=None,
    )
    lease_backend._tenant_isolation_enabled = True
    lease_backend._stats = {}
    lease_backend._hold_host_load(isolated_pending)
    assert isolated_pending.acquisition is None
    assert not isolated_pending.prefetched_layers
    assert lease_backend._stats["initial_acquisition_layers"] == 0
    assert lease_backend._stats["schedule_bound_acquisition_batches"] == 1

    assert "_create_decode_wrappers" not in NtaFlashInferAttnBackend.__dict__
    assert "_create_prefill_wrappers" not in NtaFlashInferAttnBackend.__dict__
    assert "_phase_program" not in NtaFlashInferAttnBackend.__dict__
    assert "_transport_phase_program" not in NtaFlashInferAttnBackend.__dict__
    with patch.dict(
        os.environ,
        {
            "NTA_VERIFY_ATTENTION": "1",
            "NTA_VERIFY_ATTENTION_MIXED_ONLY": "1",
            "NTA_VERIFY_EXECUTION": "1",
            "NTA_VERIFY_TRANSFER": "1",
            "NTA_VERIFY_INDEX_MAP": "1",
        },
        clear=False,
    ):
        verification = SglangVerificationConfig.from_environment()
    assert verification.attention and verification.attention_mixed_only
    assert verification.execution and verification.transfer
    assert verification.index_map
    with patch.dict(
        os.environ,
        {
            "NTA_VERIFY_ATTENTION": "0",
            "NTA_VERIFY_ATTENTION_MIXED_ONLY": "1",
        },
        clear=False,
    ):
        try:
            SglangVerificationConfig.from_environment()
        except RuntimeError as error:
            assert "requires NTA_VERIFY_ATTENTION" in str(error)
        else:
            raise AssertionError(
                "mixed-only verification lacked attention verification"
            )
    model_runner = types.SimpleNamespace(
        model_config=types.SimpleNamespace(model_path="fixture-model")
    )
    host_tier = types.SimpleNamespace(
        is_host_staged=True,
        tier=types.SimpleNamespace(value="host_staged"),
    )
    with patch.dict(
        os.environ,
        {
            "NTA_PROFILE_CPU": "1",
            "NTA_REVISION": "fixture-revision",
            "NTA_OPPORTUNITY_TRACE_FILE": "/tmp/nta-opportunity.jsonl",
            "NTA_OPPORTUNITY_PARALLEL_SLOTS": "7",
        },
        clear=True,
    ):
        observability = SglangObservabilityConfig.from_environment(
            model_runner=model_runner,
            tier=host_tier,
            opportunity_parallel_slots=80,
        )
        os.environ["NTA_PROFILE_CPU"] = "0"
    assert observability.profile_cpu
    assert observability.revision == "fixture-revision"
    assert observability.opportunity_model == "fixture-model"
    assert observability.opportunity_tier == "host_staged"
    assert observability.opportunity_parallel_slots == 7

    auto_execution = types.SimpleNamespace(
        protocol=types.SimpleNamespace(kind=ProtocolKind.LATE_BOUND),
        host_execution_mode=HostExecutionMode.AUTO,
    )
    with patch.dict(os.environ, {}, clear=True):
        assert (
            incremental_calibration_probe_count(
                execution=auto_execution, host_staged=True
            )
            == 2
        )
        assert (
            incremental_calibration_probe_count(
                execution=auto_execution, host_staged=False
            )
            == 0
        )
    with patch.dict(os.environ, {"NTA_EXECUTION_CALIBRATION_PROBES": "0"}, clear=True):
        assert (
            incremental_calibration_probe_count(
                execution=auto_execution, host_staged=True
            )
            == 0
        )
    with patch.dict(os.environ, {"NTA_EXECUTION_CALIBRATION_PROBES": "1"}, clear=True):
        try:
            incremental_calibration_probe_count(
                execution=auto_execution, host_staged=True
            )
        except RuntimeError as error:
            assert "0 or at least 2" in str(error)
        else:
            raise AssertionError("one AUTO probe cannot close the cost model")
    forced_execution = types.SimpleNamespace(
        protocol=types.SimpleNamespace(kind=ProtocolKind.LATE_BOUND),
        host_execution_mode=HostExecutionMode.DIRECT,
    )
    with patch.dict(os.environ, {"NTA_EXECUTION_CALIBRATION_PROBES": "2"}, clear=True):
        try:
            incremental_calibration_probe_count(
                execution=forced_execution, host_staged=True
            )
        except RuntimeError as error:
            assert "host-staged AUTO" in str(error)
        else:
            raise AssertionError("forced execution accepted AUTO probes")
    with patch.dict(
        os.environ,
        {
            "NTA_REVISION": "fixture-revision",
            "NTA_OPPORTUNITY_TRACE_FILE": "/tmp/nta-opportunity.jsonl",
        },
        clear=True,
    ):
        try:
            SglangObservabilityConfig.from_environment(
                model_runner=model_runner,
                tier=types.SimpleNamespace(
                    is_host_staged=False,
                    tier=types.SimpleNamespace(value="nvme"),
                ),
                opportunity_parallel_slots=80,
            )
        except ValueError as error:
            assert "requires the host_staged tier" in str(error)
        else:
            raise AssertionError("opportunity tracing accepted a non-host tier")

    # Forward lifecycle is intentionally tested without constructing the full
    # SGLang backend.  These methods own only Python-side forward state and the
    # HiCache lease boundary; requiring CUDA allocation here would hide the
    # state-machine behavior behind unrelated integration setup.
    class LifecycleHiCache:
        def __init__(self, pending=None) -> None:
            self.live = {} if pending is None else {pending.consumer_index: pending}
            self.retire_calls = []

        def get(self, consumer_index: int):
            return self.live.get(consumer_index)

        def retire(self, pending, *, stream) -> bool:
            self.retire_calls.append((pending, stream))
            if self.live.get(pending.consumer_index) is not pending:
                return False
            del self.live[pending.consumer_index]
            return True

    def lifecycle_backend(batch, hicache):
        backend = NtaFlashInferAttnBackend.__new__(NtaFlashInferAttnBackend)
        backend._active_batch = batch
        backend._current_engine_batch = object()
        backend._stock_wrapper_for_typed = {17: object()}
        backend._hicache = hicache
        backend._cuda_graph_mode = False
        backend._stats = {
            "forward_lifecycle_completions": 0,
            "forward_lifecycle_aborts": 0,
        }
        return backend

    unfinished_batch = _ActiveBatch(
        bindings=(),
        semantic_plans={},
        pending_host_load=None,
        stream_ordered_epoch=object(),
    )
    unfinished_backend = lifecycle_backend(unfinished_batch, LifecycleHiCache())
    unfinished_engine_batch = unfinished_backend._current_engine_batch
    try:
        unfinished_backend._begin_forward()
    except RuntimeError as error:
        assert "stream-ordered work window" in str(error)
    else:
        raise AssertionError("a new forward replaced an unfinished work epoch")
    assert unfinished_backend._active_batch is unfinished_batch
    assert unfinished_backend._current_engine_batch is unfinished_engine_batch
    assert unfinished_backend._stock_wrapper_for_typed

    live_pending = types.SimpleNamespace(consumer_index=23)
    live_batch = _ActiveBatch(
        bindings=(),
        semantic_plans={},
        pending_host_load=live_pending,
    )
    live_hicache = LifecycleHiCache(live_pending)
    live_backend = lifecycle_backend(live_batch, live_hicache)
    live_engine_batch = live_backend._current_engine_batch
    try:
        live_backend._begin_forward()
    except RuntimeError as error:
        assert "HiCache acquisition lease" in str(error)
    else:
        raise AssertionError("a new forward replaced a live HiCache lease")
    assert live_backend._active_batch is live_batch
    assert live_backend._current_engine_batch is live_engine_batch
    assert live_hicache.get(live_pending.consumer_index) is live_pending

    retired_pending = types.SimpleNamespace(consumer_index=29)
    finished_batch = _ActiveBatch(
        bindings=(),
        semantic_plans={},
        pending_host_load=retired_pending,
    )
    finished_backend = lifecycle_backend(finished_batch, LifecycleHiCache())
    finished_backend._finish_forward(finished_batch)
    assert finished_backend._active_batch is None
    assert finished_backend._current_engine_batch is None
    assert finished_backend._stock_wrapper_for_typed == {}
    assert finished_backend._stats["forward_lifecycle_completions"] == 1
    assert finished_backend._stats["forward_lifecycle_aborts"] == 0

    aborted_pending = types.SimpleNamespace(consumer_index=31)
    aborted_batch = _ActiveBatch(
        bindings=(),
        semantic_plans={},
        pending_host_load=aborted_pending,
    )
    aborted_hicache = LifecycleHiCache(aborted_pending)
    aborted_backend = lifecycle_backend(aborted_batch, aborted_hicache)
    abort_stream = object()
    with (
        patch("torch.cuda.synchronize") as synchronize,
        patch("torch.cuda.current_stream", return_value=abort_stream),
    ):
        assert aborted_backend.abort_active_forward()
        assert not aborted_backend.abort_active_forward()
    synchronize.assert_called_once_with()
    assert aborted_hicache.retire_calls == [(aborted_pending, abort_stream)]
    assert aborted_hicache.get(aborted_pending.consumer_index) is None
    assert aborted_backend._active_batch is None
    assert aborted_backend._current_engine_batch is None
    assert aborted_backend._stock_wrapper_for_typed == {}
    assert aborted_backend._stats["forward_lifecycle_aborts"] == 1
    assert aborted_backend._stats["forward_lifecycle_completions"] == 0

    def mover_controller(
        *, policy: str, frontier_enabled: bool, calibrated: bool = False
    ) -> HostMoverController:
        samples = 3 if calibrated else 0
        return HostMoverController(
            policy=policy,
            default_service_model=IndexedMoverServiceModel(
                sm_bandwidth_bytes_per_second=30_000_000_000,
                copy_bandwidth_bytes_per_second=30_000_000_000,
                copy_operation_ns=100,
                sm_samples=samples,
                copy_samples=samples,
            ),
            calibration_samples=3,
            copy_engine_max_operations=64,
            frontier_layers_per_wave=4,
            profile_transfer=False,
            frontier_enabled=frontier_enabled,
            profile_index_layout=False,
            profile_index_min_bytes=64 * 1024,
            verify_index_map=False,
            stats={},
        )

    assert mover_controller(policy="sm", frontier_enabled=True).profile_enabled(
        "sm", 1 << 20
    )
    assert not mover_controller(policy="sm", frontier_enabled=False).profile_enabled(
        "sm", 1 << 20
    )
    assert mover_controller(policy="auto", frontier_enabled=False).profile_enabled(
        "sm", 1 << 20
    )
    assert mover_controller(
        policy="auto", frontier_enabled=False, calibrated=True
    ).profile_enabled("sm", 1 << 20, complete_calibration=True)

    class _ProfileEvent:
        def __init__(self, milliseconds: float = 0.0) -> None:
            self.milliseconds = milliseconds

        def query(self) -> bool:
            return True

        def elapsed_time(self, _finish) -> float:
            return self.milliseconds

    aggregate_movers = mover_controller(policy="auto", frontier_enabled=False)
    for milliseconds in (2.0, 4.0, 6.0):
        aggregate_movers.record_profile(
            MoverProfile(
                _ProfileEvent(milliseconds),
                _ProfileEvent(),
                "copy_engine",
                1 << 20,
                1 << 20,
                8,
                8_000,
                True,
            )
        )
    aggregate_movers.collect_profiles()
    aggregate_curve = aggregate_movers.service_model(1 << 20)
    assert aggregate_curve.copy_samples == 3
    assert aggregate_curve.copy_bandwidth_bytes_per_second == 262_144_000
    assert aggregate_curve.copy_operation_ns == 1_000
    assert aggregate_movers._stats["host_mover_complete_calibration_frontiers"] == 1
    assert aggregate_movers._stats["host_mover_complete_calibration_wave_samples"] == 3

    context_stats: dict[str, int] = {}
    context_movers = HostMoverController(
        policy="auto",
        default_service_model=IndexedMoverServiceModel(
            sm_bandwidth_bytes_per_second=30_000_000_000,
            copy_bandwidth_bytes_per_second=60_000_000_000,
            copy_operation_ns=10,
            sm_samples=3,
            copy_samples=3,
        ),
        calibration_samples=3,
        copy_engine_max_operations=64,
        frontier_layers_per_wave=4,
        profile_transfer=False,
        frontier_enabled=True,
        profile_index_layout=False,
        profile_index_min_bytes=64 * 1024,
        verify_index_map=False,
        stats=context_stats,
    )

    def pending_mover_plan():
        source = torch.tensor((0, 1), dtype=torch.int32)
        destination = torch.tensor((2, 3), dtype=torch.int32)
        return types.SimpleNamespace(
            mover_plan=None,
            prefetch_tensors=(),
            materialize_device_index_map=lambda: types.SimpleNamespace(
                source_indices=source,
                destination_indices=destination,
            ),
        )

    unbound_plan = context_movers.plan(
        pending_mover_plan(),
        ((1024, 1024),) * 4,
        2,
        layer_service_key=None,
        layer_curve=None,
        collect_layer_profiles=lambda: None,
    )
    assert unbound_plan.kind == "sm"
    assert unbound_plan.selection_reason == "execution_context_unbound"
    assert context_stats["prefetch_mover_plan_execution_context_unbound_leases"] == 1

    bound_curve = AcquisitionServiceCurve(
        samples_ns=(1_000,), minimum_samples=1, maximum_samples=4
    )
    bound_plan = context_movers.plan(
        pending_mover_plan(),
        ((1024, 1024),) * 4,
        2,
        layer_service_key=("extend", 2, 2),
        layer_curve=bound_curve,
        collect_layer_profiles=lambda: None,
    )
    assert bound_plan.kind == "copy_engine"

    pending_calibration = types.SimpleNamespace(
        mover_plan=types.SimpleNamespace(
            row_count=2,
            sm_row_count=2,
            copy_runs=(),
        ),
        row_bytes_by_layer=((1024, 1024),) * 4,
        device_indices=types.SimpleNamespace(numel=lambda: 2),
    )
    assert not mover_controller(policy="sm", frontier_enabled=True).lease_calibrated(
        pending_calibration
    )
    assert mover_controller(
        policy="sm", frontier_enabled=True, calibrated=True
    ).lease_calibrated(pending_calibration)

    # Admission starts one finite, coalesced EDF queue. Layer zero is merely
    # the earliest deadline; it is not the only transport job submitted.
    admission_model = LayerAcquisitionModel(
        layer_bytes=(1,) * 4,
        transfer_service_ns=(50,) * 4,
        initial_compute_ns=0,
        inter_layer_compute_ns=100,
    )
    admission_pending = types.SimpleNamespace(
        controller=types.SimpleNamespace(layer_num=4),
        transfer_plan=object(),
        prefetched_layers={},
        acquisition=LayerAcquisition(admission_model.layer_bytes),
    )
    admission_pending.acquisition.bind_model(admission_model)
    admission_ranges: list[tuple[int, int]] = []

    def publish_admission_range(
        _pending,
        *,
        first_local_layer: int,
        last_local_layer: int,
    ) -> None:
        admission_ranges.append((first_local_layer, last_local_layer))
        for local_layer in range(first_local_layer, last_local_layer):
            _pending.prefetched_layers[local_layer] = object()

    admission_backend = NtaFlashInferAttnBackend.__new__(NtaFlashInferAttnBackend)
    admission_backend._stats = {}
    admission_backend._host_transport = types.SimpleNamespace(
        prepare=publish_admission_range
    )
    admission_backend._admission_deadline_model = (
        lambda pending, _batch: pending.acquisition.model
    )
    admission_backend._start_admission_acquisition(admission_pending, object())
    assert admission_ranges == [(0, 4)]
    assert admission_pending.acquisition.fully_published
    assert admission_backend._stats["host_acquisition_jobs_submitted"] == 4

    # Frontier publication is a behavior contract, not a counter convention.
    # Before mover calibration, completing layer zero may enqueue only one
    # bounded probe wave.  Once the frozen EDF model proves the remaining
    # prefix feasible, the same transition publishes layers 1..35 so they
    # arrive through the ready-stock/preacquired numerical path.
    def exercise_deadline_frontier(
        *, calibrated: bool
    ) -> tuple[list[tuple[int, int]], dict[str, int]]:
        pending = types.SimpleNamespace(
            controller=types.SimpleNamespace(layer_num=36),
            mover_plan=object(),
            prefetched_layers={},
            prefetch_tensors=(),
            acquisition=None,
        )
        model = (
            LayerAcquisitionModel(
                layer_bytes=(1,) * 36,
                transfer_service_ns=(50,) * 36,
                initial_compute_ns=0,
                inter_layer_compute_ns=100,
            )
            if calibrated
            else None
        )
        batch = _ActiveBatch(
            bindings=(),
            semantic_plans={},
            pending_host_load=pending,
            deadline_model=model,
            deadline_model_initialized=calibrated,
        )
        backend = NtaFlashInferAttnBackend.__new__(NtaFlashInferAttnBackend)
        backend._frontier_enabled = True
        backend._frontier_layers_per_wave = 4
        backend._active_batch = batch
        backend._stats = {}
        backend._host_movers = types.SimpleNamespace(
            collect_profiles=lambda: None,
            lease_calibrated=lambda _pending: False,
        )
        backend._layer_calibration = types.SimpleNamespace(
            collect=lambda: None,
            curve=lambda _key: None,
        )
        calls: list[tuple[int, int]] = []

        def prepare(
            _pending,
            *,
            first_local_layer: int,
            last_local_layer: int,
        ) -> None:
            calls.append((first_local_layer, last_local_layer))
            for layer in range(first_local_layer, last_local_layer):
                _pending.prefetched_layers[layer] = object()
            _pending.prefetch_tensors += (object(),)

        backend._host_transport = types.SimpleNamespace(prepare=prepare)
        backend._advance_deadline_frontier(pending, 0)
        return calls, backend._stats

    frontier_calls, frontier_stats = exercise_deadline_frontier(calibrated=False)
    assert frontier_calls == [(1, 5)]
    assert frontier_stats["deadline_frontier_calibration_layers"] == 4
    modeled_calls, modeled_stats = exercise_deadline_frontier(calibrated=True)
    assert modeled_calls == [(1, 36)]
    assert modeled_stats["deadline_frontier_published_layers"] == 35

    quiescence_owner = SglangPlanMaterializer.__new__(SglangPlanMaterializer)
    quiescence_owner._tier_service = types.SimpleNamespace(
        is_nvme=False, is_host_staged=True
    )
    quiescence_owner._indexed_object_quiescence_event = FakeEvent()
    quiescence_owner._indexed_object_quiescence_recorded = False
    quiescence_owner.record_host_consumer(fake_stream, final_layer=False)
    assert not quiescence_owner._indexed_object_quiescence_recorded
    quiescence_owner.record_host_consumer(fake_stream, final_layer=True)
    assert quiescence_owner._indexed_object_quiescence_recorded
    assert quiescence_owner._indexed_object_quiescence_event.stream is fake_stream

    # An arriving consumer shares the proactive acquisition's objects and
    # fence.  Consumer activation must not fabricate a second physical copy,
    # acquisition group, or set of CTA work items in the evidence counters.
    arriving_stats = {
        "demand_host_layers": 0,
        "incremental_host_layers": 0,
        "request_overlap_layers": 0,
        "indexed_host_bytes": 0,
        "native_demand_sm_bytes": 0,
        "indexed_host_objects": 0,
        "request_acquisition_groups": 0,
        "cta_work_items": 11,
    }
    arriving_owner = SglangPlanMaterializer.__new__(SglangPlanMaterializer)
    arriving_owner._stats = arriving_stats
    arriving_owner._record_arriving_consumer_stats(
        types.SimpleNamespace(
            uses_dependency_protocol=True,
            overlap_initial=True,
        )
    )
    assert arriving_stats == {
        "demand_host_layers": 1,
        "incremental_host_layers": 1,
        "request_overlap_layers": 1,
        "indexed_host_bytes": 0,
        "native_demand_sm_bytes": 0,
        "indexed_host_objects": 0,
        "request_acquisition_groups": 0,
        "cta_work_items": 11,
    }

    # Materialization must query the backend's current wrapper registry.  The
    # typed module installer replaces that registry after materializer setup;
    # retaining the original empty dict silently disables the stock fast path.
    wrapper_registry_backend = NtaFlashInferAttnBackend.__new__(
        NtaFlashInferAttnBackend
    )
    wrapper_registry_backend._stock_wrapper_for_typed = {}
    wrapper_registry_owner = SglangPlanMaterializer.__new__(SglangPlanMaterializer)
    wrapper_registry_owner._stock_wrapper_available = (
        wrapper_registry_backend._has_stock_wrapper
    )
    wrapper_registry_backend._stock_wrapper_for_typed = {17: object()}
    assert wrapper_registry_owner._stock_wrapper_available(17)

    try:
        prove_direct_metadata_execution(
            {},
            object(),
            (),
            host_staged=False,
            tenant_isolation=False,
            model=object(),
            mode=object(),
        )
    except RuntimeError as error:
        assert "host-staged only" in str(error)
    else:
        raise AssertionError("physical tier entered the host-only direct proof")

    # Explicit artifact boundaries synchronize once and must retire every
    # event-backed observation before a counter snapshot can be published.
    class PendingMoverProfiles:
        def __init__(self) -> None:
            self.profiles = [object()]

        @property
        def pending_profile_count(self) -> int:
            return len(self.profiles)

    class PendingLayerProfiles:
        def __init__(self, profiles, collect) -> None:
            self.profiles = profiles
            self._collect = collect

        @property
        def pending_count(self) -> int:
            return len(self.profiles)

        def collect(self) -> None:
            self._collect()

    boundary_backend = NtaFlashInferAttnBackend.__new__(NtaFlashInferAttnBackend)
    boundary_movers = PendingMoverProfiles()
    boundary_backend._host_movers = boundary_movers
    boundary_backend._transfer_profiles = [object()]
    boundary_backend._operator_profiles = [object()]
    boundary_backend._barrier_profiles = [object()]
    boundary_calls = []
    boundary_layer_profiles = [object()]

    def retire_transfer_profiles() -> None:
        boundary_calls.append("transfer")
        boundary_movers.profiles.clear()
        boundary_backend._transfer_profiles.clear()
        boundary_backend._operator_profiles.clear()

    def retire_layer_profiles() -> None:
        boundary_calls.append("layer")
        boundary_layer_profiles.clear()

    def retire_barrier_profiles(*, already_synchronized: bool = False) -> None:
        assert already_synchronized
        boundary_calls.append("barrier")
        boundary_backend._barrier_profiles.clear()

    boundary_backend._collect_transfer_profiles = retire_transfer_profiles
    boundary_backend._layer_calibration = PendingLayerProfiles(
        boundary_layer_profiles, retire_layer_profiles
    )
    boundary_backend._collect_barrier_profiles = retire_barrier_profiles
    with patch("torch.cuda.synchronize", lambda: boundary_calls.append("sync")):
        boundary_backend._quiesce_observation_boundary()
    assert boundary_calls == ["sync", "transfer", "layer", "barrier"]

    stale_boundary = NtaFlashInferAttnBackend.__new__(NtaFlashInferAttnBackend)
    stale_boundary._host_movers = PendingMoverProfiles()
    stale_boundary._transfer_profiles = []
    stale_boundary._operator_profiles = []
    stale_boundary._barrier_profiles = []
    stale_boundary._collect_transfer_profiles = lambda: None
    stale_boundary._layer_calibration = PendingLayerProfiles([], lambda: None)
    stale_boundary._collect_barrier_profiles = lambda **_kwargs: None
    with patch("torch.cuda.synchronize", lambda: None):
        try:
            stale_boundary._quiesce_observation_boundary()
        except RuntimeError as error:
            assert "remained pending" in str(error)
        else:
            raise AssertionError("measurement boundary accepted a stale CUDA event")

    adopted_schedule = Schedule((0,), (0,), 4, 1)
    adopted_binding = RequestBinding(0, 17, 3, stable_request_id("semantic-plan"))
    semantic_engine_batch = types.SimpleNamespace(epoch=41, bindings=(adopted_binding,))
    semantic_pending = types.SimpleNamespace(
        lease_id=29,
        row_bytes_by_layer=((8, 8), (8, 8)),
        device_indices=types.SimpleNamespace(numel=lambda: 4),
        operation_ranges=lambda: (LeaseOperationRange(7, 0, 4),),
    )
    adopted_semantic = build_semantic_wrapper_plan(
        engine_batch=semantic_engine_batch,
        tile_compute_ns=3000,
        bindings=(adopted_binding,),
        schedule=adopted_schedule,
        pending=semantic_pending,
        dependency_kind="typed_lease",
        acquisition_slices=(LeaseAcquisitionSlice(7, 0, 4),),
        acquisition_groups=(LeaseAcquisitionGroup(7, 0, 4),),
    )
    assert adopted_semantic.topology.work_count == 1
    assert adopted_semantic.indexed_topology is not None
    adopted_batch = _ActiveBatch(
        bindings=(adopted_binding,),
        semantic_plans={101: adopted_semantic},
        pending_host_load=None,
    )
    adopted_batch.adopt_wrapper_identity({101: 202})
    assert adopted_batch.semantic_plans == {202: adopted_semantic}
    adopted_batch.fragment_lookahead[0] = object()
    try:
        adopted_batch.adopt_wrapper_identity({202: 303})
    except RuntimeError as error:
        assert "after execution began" in str(error)
    else:
        raise AssertionError("wrapper adoption accepted active execution state")
    adopted_batch.fragment_lookahead.clear()
    try:
        adopted_batch.adopt_wrapper_identity({303: 404})
    except RuntimeError as error:
        assert "does not cover its semantic plans" in str(error)
    else:
        raise AssertionError("wrapper adoption accepted stale source identity")

    class EventProbe:
        def __init__(self) -> None:
            self.streams = []

        def record(self, stream) -> None:
            self.streams.append(stream)

    consumer_event = EventProbe()
    nvme_slots = NvmeSlotLifetime(consumer_event)
    assert nvme_slots.prior_consumer_event(0) is None
    nvme_slots.commit(((0, 4096), (1, 8192)))
    try:
        nvme_slots.prior_consumer_event(0)
    except RuntimeError as error:
        assert "prior-consumer event" in str(error)
    else:
        raise AssertionError("NVMe slot was replaced without a consumer proof")
    nvme_slots.record_retirement("attention-stream")
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
    nvme_slots.record_retirement("next-attention-stream")
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

    kernel_log = []
    kernel_resources = SglangKernelResources(
        config=SglangKernelConfig(
            dtype_q=torch.float16,
            dtype_kv=torch.float16,
            head_dim=128,
            num_wrappers=1,
            skip_prefill=False,
            decode_use_tensor_cores=False,
            stream_ordered_retirement=False,
            workspace_buffer=torch.empty(0),
        ),
        stock_wrappers=SglangWrapperSet.capture(
            decode=[object()],
            prefill_paged=[object()],
            prefill_verify=[object()],
        ),
        stats={"verified_operator_modules": 0},
    )
    assert "backend" not in inspect.signature(SglangKernelResources).parameters
    assert len(kernel_resources.stock_wrappers.decode) == 1

    class FakeDecodeWrapper:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    class FakePrefillWrapper(FakeDecodeWrapper):
        pass

    valid_jit_args = [None] * 8
    valid_jit_args[7] = ["nta_runtime", "nta_work_items", "nta_dependencies"]
    with (
        patch(
            "nta_runtime.engines.sglang_kernels.BatchDecodeWithPagedKVCacheWrapper",
            FakeDecodeWrapper,
        ),
        patch(
            "nta_runtime.engines.sglang_kernels.BatchPrefillWithPagedKVCacheWrapper",
            FakePrefillWrapper,
        ),
        patch.object(
            kernel_resources,
            "_jit_arguments",
            return_value=valid_jit_args,
        ),
        patch.object(
            kernel_resources,
            "_materialize_attention_module",
            return_value=object(),
        ),
    ):
        typed_once = kernel_resources.typed_wrappers()
        typed_twice = kernel_resources.typed_wrappers()
    assert typed_once.decode[0] is typed_twice.decode[0]
    assert typed_once.prefill_paged[0] is typed_twice.prefill_paged[0]
    assert typed_once.prefill_verify[0] is typed_twice.prefill_verify[0]
    assert kernel_resources.is_instrumented(typed_once.decode[0])
    assert "decode_demand_acquire" in kernel_resources.module_name(typed_once.decode[0])
    assert typed_once.decode[0].kwargs["jit_args"] is None
    assert typed_once.prefill_paged[0].kwargs["jit_args"] is None

    warmups = []
    setup_order = []

    class PhaseProbe:
        def warmup_indexed_host_validation(self, runtime, stream) -> None:
            warmups.append((runtime, stream))

    setup_runtime = object()
    setup_stream = types.SimpleNamespace(
        synchronize=lambda: setup_order.append("synchronize")
    )
    from nta_runtime.flashinfer_jit import (
        FlashInferMaterializationOrigin,
    )

    kernel_resources._typed_materializations = {
        module_name: types.SimpleNamespace(
            origin=FlashInferMaterializationOrigin.DISK_CACHE_LOAD
        )
        for module_name in set(kernel_resources._wrapper_modules.values())
    }
    with (
        patch.object(
            kernel_resources,
            "operator_module",
            side_effect=lambda _wrapper: object(),
        ) as operator_module,
        patch.object(
            kernel_resources,
            "transport_program",
            side_effect=lambda: (
                setup_order.append("transport"),
                PhaseProbe(),
            )[1],
        ),
    ):
        kernel_resources.prepare_typed_execution_modules(
            runtime=setup_runtime,
            host_staged=True,
            stream=setup_stream,
        )
    assert operator_module.call_count == 2
    assert len(warmups) == 1
    assert all(runtime is setup_runtime for runtime, _stream in warmups)
    assert setup_order == ["transport", "synchronize"]

    kernel_resources._operator_modules["decode"] = CloseProbe(kernel_log, fail=True)
    kernel_resources._transport_program = CloseProbe(kernel_log)
    kernel_errors = kernel_resources.close()
    assert kernel_log == ["close", "close"]
    assert len(kernel_errors) == 1
    assert kernel_resources.close() == ()
    try:
        _ = kernel_resources.stock_wrappers
    except RuntimeError as error:
        assert "closed" in str(error)
    else:
        raise AssertionError("closed kernel owner retained an executable interface")

    SglangKernelResources._require_attention_tensor_abi(valid_jit_args)
    invalid_jit_args = list(valid_jit_args)
    invalid_jit_args[7] = ["nta_runtime", "nta_dependencies"]
    try:
        SglangKernelResources._require_attention_tensor_abi(invalid_jit_args)
    except RuntimeError as error:
        assert "unexpected tensor ABI" in str(error)
    else:
        raise AssertionError("kernel owner accepted an incompatible tensor ABI")

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
    assert _pipeline_object_range(128, 0, 12, 2) == (80, 128)
    assert _pipeline_object_range(128, 1, 12, 2) == (32, 80)
    # The configured wave bound, not one lease's selected wave count, is the
    # namespace stride. Two consumers may independently select one and four
    # active waves without aliasing directory slots or stable object IDs.
    first_range = _pipeline_object_range(1024, 0, 36, 4)
    second_range = _pipeline_object_range(1024, 1, 36, 4)
    assert second_range[1] == first_range[0]
    first_ids = {
        _pipeline_object_id(0, 36, layer, 4) + lane
        for layer in range(36)
        for lane in range(8)
    }
    second_ids = {
        _pipeline_object_id(1, 36, layer, 4) + lane
        for layer in range(36)
        for lane in range(8)
    }
    assert first_ids.isdisjoint(second_ids)
    for invalid in ((0, 0, 12), (128, -1, 12), (48, 1, 12)):
        try:
            _pipeline_object_range(*invalid, 1)
        except RuntimeError:
            pass
        else:
            raise AssertionError("pipeline object allocator accepted overlap")
    signature = _semantic_plan_signature_prefix(
        (0, 0), (3, 4), (((10,), (20,)),), ((7, 3),)
    )
    remapped = _semantic_plan_signature_prefix(
        (0, 0), (3, 4), (((11,), (21,)),), ((7, 3),)
    )
    rebound = _semantic_plan_signature_prefix(
        (0, 0), (3, 4), (((10,), (20,)),), ((7, 4),)
    )
    reslotted = _semantic_plan_signature_prefix(
        (0, 0), (3, 4), (((10,), (20,)),), ((8, 3),)
    )
    assert signature != remapped, "plan cache aliased different HiCache page rows"
    assert signature != rebound, "plan cache aliased a stale request generation"
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
        "stream_address": 0x3000,
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
    graph_key = demand_graph_key(**graph_key_arguments)
    same_graph_key = demand_graph_key(**graph_key_arguments)
    changed_graph_key = demand_graph_key(
        **(graph_key_arguments | {"progress_blocks": (4,), "ready_work_counts": (8,)})
    )
    windowed_graph_key = demand_graph_key(
        **(
            graph_key_arguments
            | {"ready_work_counts": (4, 4), "ready_work_offsets": (0, 4)}
        )
    )
    other_stream_graph_key = demand_graph_key(
        **(graph_key_arguments | {"stream_address": 0x4000})
    )
    assert graph_key == same_graph_key
    assert graph_key != changed_graph_key
    assert graph_key != windowed_graph_key
    assert graph_key != other_stream_graph_key

    graph_stats = {
        "demand_graph_warmups": 0,
        "demand_graph_captures": 0,
        "demand_graph_replays": 0,
        "demand_graph_evictions": 0,
    }
    graph_cache = DemandGraphCache(capacity=1, stats=graph_stats)
    enqueue_calls = []

    def enqueue_graph(query_arg, output_arg, events_arg, callback_arg) -> None:
        enqueue_calls.append((query_arg, output_arg, events_arg, callback_arg))

    eager_events = (object(), (object(), object()))
    callback = object()
    eager_output = torch.empty_like(query)
    assert (
        graph_cache.enqueue(
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
    assert graph_stats["demand_graph_warmups"] == 1

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
        captured_output = graph_cache.enqueue(
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
    assert graph_stats["demand_graph_captures"] == 1
    assert enqueue_calls[-1][3] is None
    assert len(enqueue_calls[-1][2][1]) == 2
    captured = graph_cache.captured(graph_key)
    assert captured is not None
    assert captured_output is captured.output
    assert captured.graph.replays == 1
    assert graph_stats["demand_graph_replays"] == 1
    graph_wrapper._qo_indptr_buf = torch.tensor((0, 7), dtype=torch.int32)
    replay_output = graph_cache.enqueue(
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
    assert graph_stats["demand_graph_replays"] == 2
    graph_cache.discard_plan(graph_plan)
    assert not graph_cache.contains(graph_key)
    assert not graph_cache.tracks(graph_key)
    with (
        patch("torch.cuda.Event", Event),
        patch("torch.cuda.CUDAGraph", Graph),
        patch("torch.cuda.graph", lambda *args, **kwargs: nullcontext()),
    ):
        graph_cache.enqueue(
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
        graph_cache.enqueue(
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
    synchronized = []
    graph_cache.reserve(
        replace(graph_key, layer_id=4),
        types.SimpleNamespace(synchronize=lambda: synchronized.append(True)),
    )
    assert synchronized == [True]
    assert not graph_cache.contains(graph_key)
    assert graph_stats["demand_graph_evictions"] == 1

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

    validation_backend = NtaFlashInferAttnBackend.__new__(NtaFlashInferAttnBackend)
    validation_wrapper = types.SimpleNamespace()
    validation_backend._active_batch = _ActiveBatch(
        bindings=(adopted_binding,),
        semantic_plans={id(validation_wrapper): adopted_semantic},
        pending_host_load=object(),
    )
    validation_backend._stats = {"semantic_wrapper_plan_lookups": 0}
    geometry = (TensorGeometry(1, 8), TensorGeometry(1, 8))
    for _ in range(3):
        assert (
            validation_backend._validate_semantic_wrapper_plan(
                validation_wrapper,
                types.SimpleNamespace(layer_id=0),
                geometry,
                verify=False,
            )
            == 0
        )
    assert validation_backend._stats["semantic_wrapper_plan_lookups"] == 3

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
    assert stock_backend._active_batch.semantic_plans == {}
    assert stock_backend._active_batch.pending_host_load is stock_pending
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

    source = inspect.getsource(SglangPlanMaterializer.upload_plan)
    assert "batch.host_execution" in source
    assert "plan_host_execution" not in source
    assert "force_rounds" not in source

    class Wrapper:
        def __init__(self) -> None:
            self.calls = 0
            self.arguments = ()

        def run(self, *args, **kwargs) -> None:
            self.calls += 1
            self.arguments = args

    binding = types.SimpleNamespace(request_slot=0)
    active_batch = _ActiveBatch(
        bindings=(binding,),
        semantic_plans={},
        pending_host_load=None,
    )
    execution_owner = SglangAttentionExecutor.__new__(SglangAttentionExecutor)
    execution_owner._runtime = types.SimpleNamespace(device_view_tensor=object())
    execution_owner._stats = {"transformed_direct_launches": 0}
    verified_wrappers = []

    class KernelProbe:
        def __init__(self) -> None:
            self.modules = {}

        def is_instrumented(self, candidate) -> bool:
            return id(candidate) in self.modules

        def operator_module(self, candidate) -> None:
            verified_wrappers.append(candidate)

        def module_name(self, candidate) -> str:
            return self.modules[id(candidate)]

        def describe_wrapper_id(self, wrapper_id: int) -> str:
            return self.modules.get(wrapper_id, str(wrapper_id))

    execution_owner._kernels = KernelProbe()
    wrapper = Wrapper()
    try:
        execution_owner.run_preacquired(
            active_batch,
            wrapper,
            object(),
            object(),
            object(),
            object(),
            {},
        )
    except RuntimeError as error:
        assert "compiler-transformed" in str(error)
    else:
        raise AssertionError("NTA attention accepted a stock wrapper")
    assert wrapper.calls == 0

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
    execution_owner._kernels.modules[id(demand_wrapper)] = "instrumented_demand_acquire"
    active_batch = _ActiveBatch(
        bindings=(binding,),
        semantic_plans={
            id(demand_wrapper): types.SimpleNamespace(schedule=demand_schedule)
        },
        pending_host_load=object(),
    )
    demand_allocation = types.SimpleNamespace(plan=demand_plan)
    execution_owner._materializer = types.SimpleNamespace(
        allocation=lambda candidate, layer_id=-1: (
            demand_allocation
            if candidate is demand_wrapper and layer_id == -1
            else None
        ),
        require_allocation=lambda candidate, layer_id=-1: (
            demand_allocation
            if candidate is demand_wrapper and layer_id == -1
            else (_ for _ in ()).throw(RuntimeError("missing test allocation"))
        ),
    )
    layer = types.SimpleNamespace(scaling=0.5)
    with patch("torch.cuda.current_stream", lambda: fake_stream):
        execution_owner.run_preacquired(
            active_batch,
            demand_wrapper,
            object(),
            object(),
            object(),
            layer,
            {},
        )
    assert len(demand_wrapper.arguments) == 8
    assert demand_wrapper.arguments[3] is work_items_tensor
    assert demand_wrapper.arguments[4] is dependencies_tensor
    assert demand_wrapper.arguments[-2:] == (3, 6)
    assert verified_wrappers == [demand_wrapper]
    assert execution_owner._stats["transformed_direct_launches"] == 1

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
