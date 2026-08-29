#!/usr/bin/env python3
"""Test vLLM worker-runtime rollback and idempotent shutdown ownership."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from nta_runtime.engines import vllm as vllm_engine  # noqa: E402
from nta_runtime.engines import vllm_modules  # noqa: E402
from nta_runtime.engines import vllm_worker  # noqa: E402
from nta_runtime.engines.vllm_config import VllmAttentionConfig  # noqa: E402
from nta_runtime.plugins import vllm as vllm_plugin  # noqa: E402
from nta_runtime.adapters.base import EngineBatch, ExactDemandProjection  # noqa: E402
from nta_runtime.adapters.vllm_v1 import (  # noqa: E402
    VllmV1ForwardState,
    VllmV1Hook,
    vllm_v1_forward_state,
)
from nta_runtime.connectors.vllm import (  # noqa: E402
    NtaVllmConnector,
    NtaVllmExternalLease,
    NtaVllmForwardAck,
)
from nta_runtime.connectors.vllm_host import (  # noqa: E402
    VllmHostWorker,
    _resolve_synchronous_load,
)
from nta_runtime.requests import RequestBinding, stable_request_id  # noqa: E402
from nta_runtime.storage_identity import vllm_storage_key  # noqa: E402
from nta_runtime.work_unit import Granularity  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)


class Runner:
    pass


class FakeRuntime:
    def __init__(self) -> None:
        self.config = SimpleNamespace(tenant_capacity=1)
        self.closed = 0

    def set_tenant_budget(self, *_args) -> None:
        raise AssertionError("invalid tenant should be rejected before budget upload")

    def close(self) -> None:
        self.closed += 1


class FakeResources:
    def __init__(self, runtime: FakeRuntime) -> None:
        self.runtime = runtime
        self.tier = SimpleNamespace()
        self.closed = 0

    def close(self) -> None:
        self.closed += 1
        self.runtime.close()


class IdentityRuntime:
    def __init__(self) -> None:
        self.published: list[tuple[int, int, int]] = []
        self.cancelled: list[tuple[int, int]] = []

    def set_request(
        self,
        slot: int,
        request_id: int,
        generation: int,
        **_policy: int,
    ) -> None:
        self.published.append((slot, request_id, generation))

    def cancel_request(self, slot: int, generation: int) -> None:
        self.cancelled.append((slot, generation))


def _connector() -> NtaVllmConnector:
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(),
        speculative_config=None,
    )
    cache = SimpleNamespace(kv_cache_groups=(SimpleNamespace(),))
    return NtaVllmConnector(config, KVConnectorRole.SCHEDULER, cache)


def _physical_connector(catalog: Path) -> NtaVllmConnector:
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"namespace": "vllm-test/tp0"}
        ),
        speculative_config=None,
    )
    cache = SimpleNamespace(
        kv_cache_groups=(
            SimpleNamespace(
                layer_names=("model.layers.0.attn",),
                kv_cache_spec=SimpleNamespace(block_size=4),
            ),
        )
    )
    with patch.dict(
        "os.environ",
        {
            "NTA_SERVING_TIER": "nvme",
            "NTA_TIER_CATALOG": str(catalog),
            "PYTHONHASHSEED": "0",
        },
        clear=False,
    ):
        return NtaVllmConnector(config, KVConnectorRole.SCHEDULER, cache)


def _test_physical_storage_identity() -> None:
    with tempfile.TemporaryDirectory(prefix="nta-vllm-catalog-") as directory:
        path = Path(directory) / "catalog.json"
        path.write_text(
            json.dumps(
                {
                    "schema": 2,
                    "tier": "nvme",
                    "format": "typed-components-v1",
                    "namespace": "vllm-test/tp0",
                    "page_tokens": 4,
                    "layer_count": 1,
                    "components": ["packed_kv"],
                    "alignment_bytes": 4096,
                    "window_bytes": 8192,
                    "entries": [
                        {
                            "storage_key": vllm_storage_key(b"hash-0"),
                            "ordinal": 0,
                            "layer": 0,
                            "components": {"packed_kv": {"offset": 0, "bytes": 4096}},
                        },
                        {
                            "storage_key": vllm_storage_key(b"hash-1"),
                            "ordinal": 1,
                            "layer": 0,
                            "components": {
                                "packed_kv": {"offset": 4096, "bytes": 4096}
                            },
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        connector = _physical_connector(path)
        assert connector.catalog is not None
        key0 = vllm_storage_key(b"hash-0")
        key1 = vllm_storage_key(b"hash-1")
        layout = vllm_engine._physical_transfer_layout(
            connector.catalog,
            catalog_layer=0,
            work_bindings=(
                ((key0, 10), (key1, 11)),
                ((key1, 11),),
                (),
            ),
            row_bytes=4096,
            max_transfer_bytes=8192,
        )
        assert len(layout.runs) == 1
        assert layout.runs[0].destination_first == 10
        assert layout.runs[0].row_count == 2
        assert layout.run_indices_by_work == ((0,), (0,), ())
        bounded = vllm_engine._physical_transfer_layout(
            connector.catalog,
            catalog_layer=0,
            work_bindings=(((key0, 10), (key1, 11)),),
            row_bytes=4096,
            max_transfer_bytes=4096,
        )
        assert bounded.run_indices_by_work == ((0, 1),)
        try:
            vllm_engine._physical_transfer_layout(
                connector.catalog,
                catalog_layer=0,
                work_bindings=(((key0, 10),), ((key1, 10),)),
                row_bytes=4096,
                max_transfer_bytes=8192,
            )
        except RuntimeError as error:
            assert "conflicting" in str(error)
        else:
            raise AssertionError("conflicting vLLM storage binding was accepted")

        partial = SimpleNamespace(
            request_id="partial",
            block_hashes=[b"hash-0", b"hash-1"],
            num_tokens=8,
        )
        partial_tokens, asynchronous = connector.get_num_new_matched_tokens(partial, 0)
        assert partial_tokens == 4 and not asynchronous
        too_short = SimpleNamespace(
            request_id="too-short",
            block_hashes=[b"hash-0"],
            num_tokens=4,
        )
        assert connector.get_num_new_matched_tokens(too_short, 0) == (0, False)
        request = SimpleNamespace(
            request_id="external",
            block_hashes=[b"hash-0", b"hash-1"],
            num_tokens=9,
        )
        tokens, asynchronous = connector.get_num_new_matched_tokens(request, 0)
        assert tokens == 8 and not asynchronous
        connector.update_state_after_alloc(
            request,
            SimpleNamespace(get_block_ids=lambda: ([10, 11, 12],)),
            tokens,
        )
        metadata = connector.build_connector_meta(
            _scheduler_output(
                new=(("external", (10, 11, 12)),),
                scheduled={"external": 1},
            )
        )
        assert metadata.external_lease is not None
        assert metadata.storage_key_tables == (
            (
                vllm_storage_key(b"hash-0"),
                vllm_storage_key(b"hash-1"),
                None,
            ),
        )
        uncommitted_retry = connector.build_connector_meta(
            _scheduler_output(scheduled={"external": 1})
        )
        assert uncommitted_retry.external_lease is not None
        assert uncommitted_retry.storage_key_tables == metadata.storage_key_tables

        # A worker failure must not acknowledge or consume the scheduler's
        # exact storage identities. Only a complete numerical forward returns
        # the lease through KVConnectorWorkerMetadata.
        worker = _physical_connector(path)
        worker.bind_connector_metadata(metadata)
        worker_batch = EngineBatch(
            "vllm",
            0,
            (RequestBinding(0, 0, 1, stable_request_id("external")),),
            Granularity.PAGE_GROUP,
            ExactDemandProjection(((10, 11, 12),), 4096),
        )
        with vllm_v1_forward_state(SimpleNamespace()) as state:
            state.batch = worker_batch
            state.input_batch = SimpleNamespace(req_ids=["external"])
            worker.start_load_kv(SimpleNamespace())
            worker.abort_forward()
        assert worker.build_connector_worker_meta() is None
        assert not worker.has_connector_metadata()

        # A delayed ACK for the aborted lease cannot consume the newer retry.
        connector.update_connector_output(
            SimpleNamespace(
                kv_connector_worker_meta=NtaVllmForwardAck(metadata.external_lease)
            )
        )
        assert connector._storage_keys_by_block["external"][10] == vllm_storage_key(
            b"hash-0"
        )
        retry_lease = uncommitted_retry.external_lease
        assert retry_lease is not None
        conflicting = NtaVllmForwardAck(
            NtaVllmExternalLease(
                retry_lease.lease_id,
                (
                    (
                        "external",
                        (
                            (10, vllm_storage_key(b"hash-0")),
                            (11, "conflicting-key"),
                        ),
                    ),
                ),
            )
        )
        try:
            connector.update_connector_output(
                SimpleNamespace(kv_connector_worker_meta=conflicting)
            )
        except RuntimeError as error:
            assert "conflicts" in str(error)
        else:
            raise AssertionError("conflicting vLLM external ACK was accepted")
        assert set(connector._storage_keys_by_block["external"]) == {10, 11}

        worker.bind_connector_metadata(uncommitted_retry)
        with vllm_v1_forward_state(SimpleNamespace()) as state:
            state.batch = worker_batch
            state.input_batch = SimpleNamespace(req_ids=["external"])
            worker.start_load_kv(SimpleNamespace())
            worker.validate_forward_commit()
            worker.commit_forward()
        acknowledgement = worker.build_connector_worker_meta()
        assert isinstance(acknowledgement, NtaVllmForwardAck)
        worker.clear_connector_metadata()
        connector._expected_worker_acks = 2
        try:
            connector.update_connector_output(
                SimpleNamespace(kv_connector_worker_meta=acknowledgement)
            )
        except RuntimeError as error:
            assert "every worker" in str(error)
        else:
            raise AssertionError("partial vLLM worker ACK was accepted")
        acknowledgement = acknowledgement.aggregate(acknowledgement)
        connector.update_connector_output(
            SimpleNamespace(kv_connector_worker_meta=acknowledgement)
        )
        resident = connector.build_connector_meta(
            _scheduler_output(scheduled={"external": 1})
        )
        assert resident.external_lease is None
        assert resident.storage_key_tables == ((),)


def _scheduler_output(
    *,
    new: tuple[tuple[str, tuple[int, ...]], ...] = (),
    scheduled: dict[str, int] | None = None,
    finished: set[str] | None = None,
    preempted: set[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id=request_id, block_ids=(list(blocks),))
            for request_id, blocks in new
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], resumed_req_ids=set()
        ),
        num_scheduled_tokens={} if scheduled is None else scheduled,
        finished_req_ids=set() if finished is None else finished,
        preempted_req_ids=set() if preempted is None else preempted,
    )


def _test_connector_lifecycle() -> None:
    connector = _connector()
    first = connector.build_connector_meta(
        _scheduler_output(
            new=(("prefill", (10, 11)), ("decode", (20,))),
            scheduled={"prefill": 8, "decode": 1},
        )
    )
    assert first.external_lease is None
    assert all(request.storage_keys == () for request in first.requests)
    assert first.request_ids == ("prefill", "decode")
    aligned = first.aligned_to(("decode", "prefill"))
    assert aligned.request_ids == ("decode", "prefill")
    assert aligned.block_tables == ((20,), (10, 11))

    connector.request_finished(SimpleNamespace(request_id="prefill"), [10, 11])
    connector.request_finished_all_groups(SimpleNamespace(request_id="decode"), ([20],))
    finish_only = connector.build_connector_meta(_scheduler_output())
    assert finish_only.requests == ()
    assert finish_only.finished_request_ids == ("decode", "prefill")

    # A finish-only scheduler step launches no worker forward.  The next real
    # batch must replay both retire notifications before assigning new slots.
    replayed = connector.build_connector_meta(
        _scheduler_output(new=(("next", (30,)),), scheduled={"next": 1})
    )
    assert replayed.request_ids == ("next",)
    assert replayed.finished_request_ids == ("decode", "prefill")
    consumed = connector.build_connector_meta(_scheduler_output(scheduled={"next": 1}))
    assert consumed.finished_request_ids == ()

    # Preemption-only steps also launch no worker forward. They must survive
    # in connector state and terminate the old runtime generation when the
    # next numerical batch arrives.
    preempt_seed = connector.build_connector_meta(
        _scheduler_output(new=(("victim", (40,)),), scheduled={"victim": 1})
    )
    assert preempt_seed.request_ids == ("victim",)
    preempt_only = connector.build_connector_meta(
        _scheduler_output(preempted={"victim"})
    )
    assert preempt_only.requests == ()
    assert preempt_only.finished_request_ids == ("victim",)
    after_preempt = connector.build_connector_meta(
        _scheduler_output(new=(("after", (50,)),), scheduled={"after": 1})
    )
    assert after_preempt.request_ids == ("after",)
    assert after_preempt.finished_request_ids == ("victim",)

    worker = _connector()
    worker.bind_connector_metadata(replayed)
    worker_batch = EngineBatch(
        "vllm",
        0,
        (RequestBinding(0, 1, 1, stable_request_id("next")),),
        Granularity.PAGE_GROUP,
        ExactDemandProjection(((30,),), 4096),
    )
    with vllm_v1_forward_state(SimpleNamespace()) as state:
        state.batch = worker_batch
        state.input_batch = SimpleNamespace(req_ids=["next"])
        worker.start_load_kv(SimpleNamespace())
        assert state.connector_validated
        worker.commit_forward()
        assert worker.build_connector_worker_meta() is None
        assert worker.has_connector_metadata()
        worker.clear_connector_metadata()
        assert not worker.has_connector_metadata()


def _test_typed_destination_setup_dispatch() -> None:
    runner = Runner()
    host_controller = vllm_engine.VllmV1WorkerController(runner)
    host_controller._resources = SimpleNamespace(
        tier=SimpleNamespace(
            is_hbm=False,
            is_host_staged=True,
            is_nvme=False,
            tier=SimpleNamespace(value="host_staged"),
        )
    )
    host_controller.prepare_physical_destinations()

    deferred = vllm_engine.VllmV1WorkerController(runner)
    deferred._resources = SimpleNamespace(
        tier=SimpleNamespace(
            is_hbm=False,
            is_host_staged=False,
            is_nvme=False,
            tier=SimpleNamespace(value="cxl_dax"),
        )
    )
    try:
        deferred.prepare_physical_destinations()
    except RuntimeError as error:
        assert "cxl_dax" in str(error) and "wrong address space" in str(error)
    else:
        raise AssertionError("deferred vLLM address space reached physical setup")


def _test_synchronous_host_load_binding() -> None:
    group = SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))
    cpu_blocks = tuple(
        SimpleNamespace(block_id=100 + index, is_null=False) for index in range(13)
    )
    gpu_ids, selected_cpu = _resolve_synchronous_load(
        local_tokens=0,
        external_tokens=208,
        block_ids_by_group=(list(range(10, 24)),),
        cpu_hit_blocks=(list(cpu_blocks),),
        kv_cache_groups=(group,),
        cp_world_size=1,
    )
    assert gpu_ids == list(range(10, 23))
    assert [block.block_id for block in selected_cpu] == list(range(100, 113))

    # A local two-block prefix places the external suffix at blocks [2, 4),
    # even when all destination blocks have already become hash-visible.
    suffix_gpu, suffix_cpu = _resolve_synchronous_load(
        local_tokens=32,
        external_tokens=32,
        block_ids_by_group=([40, 41, 42, 43, 44],),
        cpu_hit_blocks=(list(cpu_blocks[:2]),),
        kv_cache_groups=(group,),
        cp_world_size=1,
    )
    assert suffix_gpu == [42, 43]
    assert [block.block_id for block in suffix_cpu] == [100, 101]

    try:
        _resolve_synchronous_load(
            local_tokens=32,
            external_tokens=32,
            block_ids_by_group=([40, 41, 42],),
            cpu_hit_blocks=(list(cpu_blocks[:2]),),
            kv_cache_groups=(group,),
            cp_world_size=1,
        )
    except RuntimeError as error:
        assert "omits" in str(error)
    else:
        raise AssertionError("incomplete synchronous host allocation was accepted")


def _test_typed_all_layer_host_preload() -> None:
    worker = object.__new__(VllmHostWorker)
    worker._metadata = SimpleNamespace(
        load_cpu_blocks=[100, 101], load_gpu_blocks=[10, 11]
    )
    worker._resources = {
        "layer.0": SimpleNamespace(row_bytes=4096),
        "layer.1": SimpleNamespace(row_bytes=4096),
    }
    params = object()
    worker._preload_params = params
    worker._preload_event = None
    worker._worker = SimpleNamespace(load_stream="load-stream")

    copied: list[tuple[list[int], list[int], object]] = []
    recorded: list[object] = []
    event = SimpleNamespace(record=lambda stream: recorded.append(stream))
    with (
        patch(
            "nta_runtime.connectors.vllm_host.copy_blocks",
            side_effect=lambda source, destination, value: copied.append(
                (source, destination, value)
            ),
        ),
        patch(
            "nta_runtime.connectors.vllm_host.torch.cuda.Event",
            return_value=event,
        ),
    ):
        assert worker.preload_exact() is event
    assert copied == [([100, 101], [10, 11], params)]
    assert recorded == ["load-stream"]
    try:
        worker.preload_exact()
    except RuntimeError as error:
        assert "twice" in str(error)
    else:
        raise AssertionError("vLLM host preload accepted duplicate submission")

    waits: list[object] = []
    state = VllmV1ForwardState()
    state.host_transfer_pairs = ((100, 10), (101, 11))
    state.host_preload_event = event
    state.host_preload_blocks = 4
    state.host_preload_bytes = 16384
    stream = SimpleNamespace(wait_event=lambda value: waits.append(value))
    for layer_name in ("layer.0", "layer.1"):
        state.begin_host_layer(layer_name)
        first_wait = state.wait_for_host_preload(stream)
        assert first_wait is (layer_name == "layer.0")
        state.finish_host_layer(layer_name)
    state.record_native_launch(
        "prefill",
        2,
        form="request_bound",
        framework_owned=False,
        serving_tier="host_staged",
    )
    evidence = state.commit_evidence()
    assert waits == [event]
    assert evidence["host_preload_waits"] == 1
    assert evidence["host_preload_batches"] == 1
    assert evidence["host_transfer_blocks"] == 4
    assert evidence["host_transfer_bytes"] == 16384
    assert evidence["host_prefill_launches"] == 1

    missing = VllmV1ForwardState()
    missing.host_transfer_pairs = ((100, 10),)
    try:
        missing.wait_for_host_preload(stream)
    except RuntimeError as error:
        assert "no preload event" in str(error)
    else:
        raise AssertionError("direct host attention accepted an unready transfer")

    isolated = VllmV1ForwardState()
    isolated.host_transfer_pairs = ((100, 10),)
    isolated.host_preload_event = event
    isolated.tenant_isolation_enabled = True
    try:
        isolated.wait_for_host_preload(stream)
    except RuntimeError as error:
        assert "tenant byte credits" in str(error)
    else:
        raise AssertionError("finite tenant credits allowed a bulk host preload")


def _test_external_tier_preserves_acquisition_schedule() -> None:
    batch = EngineBatch(
        "vllm",
        0,
        (RequestBinding(0, 0, 1, 1),),
        Granularity.PAGE_GROUP,
        ExactDemandProjection(((7,),), 4096),
    )
    with vllm_v1_forward_state(SimpleNamespace()) as state:
        state.batch = batch
        resident = object.__new__(vllm_engine.NtaVllmFlashInferMetadataBuilder)
        resident._nta_config = VllmAttentionConfig.from_environment(
            default_workspace_bytes=4096,
            environ={
                "NTA_SERVING_TIER": "hbm",
                "NTA_VLLM_NATIVE": "1",
                "FLASHINFER_WORKSPACE_BASE": "/tmp/nta-vllm-config-test",
            },
        )
        assert resident._direct_batch() is batch

        external = object.__new__(vllm_engine.NtaVllmFlashInferMetadataBuilder)
        external._nta_config = VllmAttentionConfig.from_environment(
            default_workspace_bytes=4096,
            environ={
                "NTA_SERVING_TIER": "host_staged",
                "NTA_VLLM_NATIVE": "1",
                "FLASHINFER_WORKSPACE_BASE": "/tmp/nta-vllm-config-test",
            },
        )
        assert external._direct_batch() is None

        # Deployment configuration is immutable after builder construction.
        # A late process-environment mutation cannot switch either numerical
        # path underneath an already planned forward.
        with patch.dict(
            "os.environ",
            {"NTA_SERVING_TIER": "nvme", "NTA_VLLM_COMPARE_STOCK": "1"},
            clear=False,
        ):
            assert resident._direct_batch() is batch
            assert external._direct_batch() is None


def _test_attention_modules_are_prepared_at_setup() -> None:
    config = VllmAttentionConfig.from_environment(
        default_workspace_bytes=4096,
        environ={
            "NTA_SERVING_TIER": "host_staged",
            "NTA_VLLM_NATIVE": "1",
            "FLASHINFER_WORKSPACE_BASE": "/tmp/nta-vllm-config-test",
            "NTA_VLLM_DECODE_MODULE": "decode-override",
            "NTA_VLLM_PREFILL_MODULE": "prefill-override",
        },
    )
    ensured: list[tuple[str, dict[str, object]]] = []
    resolved: list[str] = []

    def ensure(name, _dtype, _head_size, **kwargs):
        ensured.append((name, kwargs))
        return Path(f"/tmp/{name}.so")

    def resolve(name, _workspace):
        resolved.append(name)
        return Path(f"/tmp/{name}.so")

    with (
        patch.object(
            vllm_modules, "_ensure_default_attention_module", side_effect=ensure
        ),
        patch.object(vllm_modules, "_find_module", side_effect=resolve),
    ):
        vllm_modules._prepare_attention_modules(
            config, (torch.float16, torch.float16), 128
        )

    assert ensured == [
        (
            "nta_batch_prefill_vllm_request_bound_v3_binding_fp16",
            {
                "workspace": config.require_workspace(),
                "request_bound": True,
                "mapped_request_slots": True,
            },
        )
    ]
    assert resolved == ["decode-override", "prefill-override"]


def _test_worker_global_external_directory_lifetime() -> None:
    controller = vllm_engine.VllmV1WorkerController(Runner())
    first_state = VllmV1ForwardState()
    controller.begin_forward(first_state)
    first_version, first_event = controller.begin_external_publication("layer-0-stream")
    assert first_version == 1 and first_event is None

    recorded = []
    event = SimpleNamespace(record=lambda stream: recorded.append(stream))
    with patch.object(vllm_engine.torch.cuda, "Event", return_value=event):
        controller.record_external_consumer("layer-0-stream")
    controller.commit_forward(first_state)

    second_state = VllmV1ForwardState()
    controller.begin_forward(second_state)
    second_version, prior_event = controller.begin_external_publication(
        "layer-1-stream"
    )
    assert second_version == 2 and prior_event is event
    assert recorded == ["layer-0-stream"]

    # A publication cannot overlap another one, and abort must fence any
    # setup already enqueued on its stream before releasing the lease.
    try:
        controller.begin_external_publication("layer-2-stream")
    except RuntimeError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("overlapping external publication was accepted")
    abort_recorded = []
    abort_event = SimpleNamespace(record=lambda stream: abort_recorded.append(stream))
    with patch.object(vllm_engine.torch.cuda, "Event", return_value=abort_event):
        controller.abort_forward(second_state)
    assert abort_recorded == ["layer-1-stream"]

    third_state = VllmV1ForwardState()
    controller.begin_forward(third_state)
    third_version, prior_event = controller.begin_external_publication("layer-2-stream")
    assert third_version == 3 and prior_event is abort_event
    final_event = SimpleNamespace(record=lambda _stream: None)
    with patch.object(vllm_engine.torch.cuda, "Event", return_value=final_event):
        controller.record_external_consumer("layer-2-stream")
        controller.commit_forward(third_state)

    # No forward may be committed twice or against another state.
    try:
        controller.commit_forward(third_state)
    except RuntimeError as error:
        assert "wrong forward" in str(error)
    else:
        raise AssertionError("completed external forward was committed twice")


def _test_semantic_layer_identity() -> None:
    runner = Runner()
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=(
            SimpleNamespace(layer_names=("model.layers.0.attn", "model.layers.1.attn")),
        )
    )
    controller = vllm_engine.VllmV1WorkerController(runner)
    assert (
        controller.semantic_layer(SimpleNamespace(layer_name="model.layers.1.attn"))
        == 1
    )
    try:
        controller.semantic_layer(SimpleNamespace(layer_name="model.layers.2.attn"))
    except RuntimeError as error:
        assert "absent" in str(error)
    else:
        raise AssertionError("unknown vLLM semantic layer was accepted")
    controller.close()
    assert not controller._layer_ordinals


def _test_worker_attention_phase_ownership() -> None:
    controller = vllm_engine.VllmV1WorkerController(Runner())
    builds: list[str] = []
    plans: list[str] = []

    def build() -> SimpleNamespace:
        builds.append("incremental")
        return SimpleNamespace(identity=object())

    def first_plan(resource: SimpleNamespace) -> tuple[object, str]:
        plans.append("epoch-7")
        return resource.identity, plans[-1]

    first, first_schedule = controller.attention_phase(
        "incremental",
        ("prefill", "fp16", 128),
        7,
        build,
        first_plan,
        workspace_bytes=4096,
    )
    reused, reused_schedule = controller.attention_phase(
        "incremental",
        ("prefill", "fp16", 128),
        7,
        lambda: (_ for _ in ()).throw(
            AssertionError("same-epoch phase rebuilt its worker resource")
        ),
        lambda _resource: (_ for _ in ()).throw(
            AssertionError("same-epoch phase repeated planner/readback work")
        ),
        workspace_bytes=4096,
    )
    assert reused is first
    assert reused_schedule is first_schedule
    assert builds == ["incremental"] and plans == ["epoch-7"]

    def second_plan(resource: SimpleNamespace) -> tuple[object, str]:
        plans.append("epoch-8")
        return resource.identity, plans[-1]

    replanned, second_schedule = controller.attention_phase(
        "incremental",
        ("prefill", "fp16", 128),
        8,
        build,
        second_plan,
        workspace_bytes=4096,
    )
    assert replanned is first
    assert second_schedule != first_schedule
    assert builds == ["incremental"] and plans == ["epoch-7", "epoch-8"]

    # The form is part of the resource identity: a graph-compatible direct
    # wrapper can never alias an incremental dependency-aware wrapper.
    direct, _ = controller.attention_phase(
        "request_bound",
        ("prefill", "fp16", 128),
        8,
        lambda: SimpleNamespace(identity="direct"),
        lambda _resource: None,
        workspace_bytes=4096,
    )
    assert direct is not first
    assert controller._attention_workspace_bytes == 8192

    try:
        controller.attention_phase(
            "incremental",
            ("prefill", "fp16", 128),
            6,
            build,
            first_plan,
            workspace_bytes=4096,
        )
    except RuntimeError as error:
        assert "backwards" in str(error)
    else:
        raise AssertionError("worker attention phase accepted a stale epoch")

    controller.close()
    assert not controller._attention_phases
    assert controller._attention_workspace_bytes == 0


def _test_connector_request_row_identity() -> None:
    runtime = IdentityRuntime()
    hook = VllmV1Hook(
        runtime,
        2,
        page_bytes=4096,
        version_provider=lambda: "0.26.0",
    )
    metadata_ids = ("decode", "prefill")
    block_tables = ((20,), (10, 11))
    reordered = SimpleNamespace(req_ids=("prefill", "decode"), idx_mapping_np=(7, 3))
    try:
        hook.bind_connector_forward(
            metadata_ids,
            block_tables,
            (),
            input_batch=reordered,
            epoch=0,
        )
    except RuntimeError as error:
        assert "InputBatch.req_ids" in str(error) and "attention row 0" in str(error)
    else:
        raise AssertionError("reordered vLLM request rows were accepted")
    assert runtime.published == []

    aligned = SimpleNamespace(req_ids=metadata_ids, idx_mapping_np=(3, 7))
    batch = hook.bind_connector_forward(
        metadata_ids,
        block_tables,
        (),
        input_batch=aligned,
        epoch=1,
    )
    assert batch.exact_demand == ExactDemandProjection(block_tables, 4096)
    assert len(runtime.published) == 2


def _test_failed_forward_discards_evidence() -> None:
    before = dict(vllm_engine.VLLM_STATS)
    controller = vllm_engine.VllmV1WorkerController(Runner())
    state = VllmV1ForwardState()
    state.execution_owner = controller
    connector_aborts: list[bool] = []
    state.connector_owner = SimpleNamespace(
        abort_forward=lambda: connector_aborts.append(True)
    )
    controller.begin_forward(state)
    state.record_native_launch(
        "decode",
        3,
        form="incremental",
        framework_owned=False,
        serving_tier="nvme",
    )
    state.record_evidence("physical_transfer_bytes", 12288)
    vllm_engine._abort_forward(state)
    assert connector_aborts == [True]
    assert dict(vllm_engine.VLLM_STATS) == before
    try:
        state.commit_evidence()
    except RuntimeError as error:
        assert "finalized" in str(error)
    else:
        raise AssertionError("aborted vLLM evidence was committed")

    # The aborted transaction must release the controller for the next step.
    retry = VllmV1ForwardState()
    controller.begin_forward(retry)
    controller.abort_forward(retry)


def _test_commit_validation_is_nonmutating() -> None:
    """A late validation error must leave every owner abortable."""

    class Connector:
        def __init__(self) -> None:
            self.validates = 0
            self.commits = 0
            self.aborts = 0

        def validate_forward_commit(self) -> None:
            self.validates += 1

        def commit_forward(self) -> None:
            self.commits += 1

        def abort_forward(self) -> None:
            self.aborts += 1

    controller = vllm_engine.VllmV1WorkerController(Runner())
    connector = Connector()
    state = VllmV1ForwardState()
    state.batch = EngineBatch(
        "vllm",
        0,
        (RequestBinding(0, 0, 1, stable_request_id("request")),),
        Granularity.PAGE_GROUP,
        ExactDemandProjection(((10,),), 4096),
    )
    state.execution_owner = controller
    state.connector_owner = connector
    controller.begin_forward(state)
    state.begin_host_layer("layer.0")
    try:
        vllm_engine._commit_forward(state)
    except RuntimeError as error:
        assert "active host layer" in str(error)
    else:
        raise AssertionError("invalid vLLM forward was committed")
    assert controller._active_forward_state is state
    assert connector.validates == connector.commits == 0
    vllm_engine._abort_forward(state)
    assert controller._active_forward_state is None
    assert connector.aborts == 1


def _test_runner_failure_injection() -> None:
    before = dict(vllm_engine.VLLM_STATS)

    class Lease:
        def __init__(self) -> None:
            self.aborts = 0

        def abort_forward(self) -> None:
            self.aborts += 1

    class FailingRunner:
        def execute_model(
            self,
            _scheduler_output,
            _intermediate_tensors=None,
            _dummy_run=False,
            _skip_attn_for_dummy_run=False,
            _is_profile=False,
        ) -> None:
            state = vllm_engine.current_vllm_v1_forward_state()
            assert state is not None
            self.controller = vllm_engine.VllmV1WorkerController(self)
            self.lease = Lease()
            state.execution_owner = self.controller
            state.connector_owner = self.lease
            self.controller.begin_forward(state)
            state.record_native_launch(
                "decode",
                1,
                form="incremental",
                framework_owned=False,
                serving_tier="nvme",
            )
            raise RuntimeError("injected numerical failure")

        def prepare_attn(self, _input_batch):
            raise AssertionError("failure injection should not prepare attention")

        def _dummy_run(self, *_args, **_kwargs):
            raise AssertionError("failure injection should not run warmup")

        def shutdown(self) -> None:
            return

    vllm_plugin._patch_v2_runner(FailingRunner)
    runner = FailingRunner()
    try:
        runner.execute_model(SimpleNamespace(num_scheduled_tokens={"r": 1}))
    except RuntimeError as error:
        assert str(error) == "injected numerical failure"
    else:
        raise AssertionError("injected vLLM worker failure was swallowed")
    assert runner.lease.aborts == 1
    assert runner.controller._active_forward_state is None
    assert dict(vllm_engine.VLLM_STATS) == before


def main() -> None:
    _test_connector_lifecycle()
    _test_typed_destination_setup_dispatch()
    _test_synchronous_host_load_binding()
    _test_typed_all_layer_host_preload()
    _test_external_tier_preserves_acquisition_schedule()
    _test_attention_modules_are_prepared_at_setup()
    _test_worker_global_external_directory_lifetime()
    _test_semantic_layer_identity()
    _test_worker_attention_phase_ownership()
    _test_connector_request_row_identity()
    _test_failed_forward_discards_evidence()
    _test_commit_validation_is_nonmutating()
    _test_runner_failure_injection()
    _test_physical_storage_identity()
    batch = EngineBatch(
        "vllm",
        4,
        (
            RequestBinding(0, 0, 1, 11),
            RequestBinding(1, 1, 1, 12),
        ),
        Granularity.PAGE_GROUP,
        ExactDemandProjection(((10, 11, 12, 13), (20, 21)), 4096),
    )
    state = VllmV1ForwardState()
    state.batch = batch
    state.request_bindings_tensor = torch.tensor([0, 1, 1, 1], dtype=torch.int64)
    assert state.phase_request_bindings(1, 1).tolist() == [1, 1]
    state.storage_key_tables = (
        (None, None, "key-12", "key-13"),
        ("key-20", None),
    )
    assert state.storage_keys_for(batch.phase(1, 1)) == (("key-20", None),)
    split_schedule = SimpleNamespace(kv_chunk_tokens=32)
    assert vllm_engine.NtaVllmFlashInferImpl._physical_pages(
        batch, split_schedule, 0, 1, 16
    ) == (12, 13)
    assert vllm_engine.NtaVllmFlashInferImpl._external_page_bindings(
        batch,
        split_schedule,
        0,
        1,
        16,
        state.storage_key_tables,
    ) == (("key-12", 12), ("key-13", 13))
    assert vllm_engine.NtaVllmFlashInferImpl._physical_pages(
        batch, SimpleNamespace(kv_chunk_tokens=0), 1, 0, 16
    ) == (20, 21)
    for schedule, tile in (
        (SimpleNamespace(kv_chunk_tokens=16), 4),
        (SimpleNamespace(kv_chunk_tokens=15), 0),
    ):
        try:
            vllm_engine.NtaVllmFlashInferImpl._physical_pages(
                batch, schedule, 0, tile, 16
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid vLLM physical page selection was accepted")

    # An invalid immutable deployment contract is rejected before allocation.
    with patch.dict("os.environ", {"NTA_TENANT_BUDGETS": "7:4096"}, clear=False):
        rejected = vllm_engine.VllmV1WorkerController(Runner())
        with patch.object(
            vllm_worker,
            "_build_resources",
            side_effect=AssertionError("invalid config allocated resources"),
        ):
            try:
                rejected._ensure_hook(
                    rejected._runner_ref(),
                    request_capacity=1,
                    page_size=16,
                    page_bytes=4096,
                )
            except RuntimeError as error:
                assert "tenant 7" in str(error)
            else:
                raise AssertionError("invalid vLLM tenant policy was accepted")

    # A mismatch discovered after native ownership opens must close both
    # resource and runtime owners exactly once.
    runtime = FakeRuntime()
    resources = FakeResources(runtime)
    controller = vllm_engine.VllmV1WorkerController(Runner())
    with patch.dict(
        "os.environ",
        {"NTA_TENANT_CAPACITY": "8", "NTA_TENANT_BUDGETS": "7:4096"},
        clear=False,
    ):
        with patch.object(vllm_worker, "_build_resources", return_value=resources):
            try:
                controller._ensure_hook(
                    controller._runner_ref(),
                    request_capacity=1,
                    page_size=16,
                    page_bytes=4096,
                )
            except RuntimeError as error:
                assert "tenant 7" in str(error)
            else:
                raise AssertionError("runtime tenant mismatch was accepted")
    assert resources.closed == 1
    assert runtime.closed == 1
    assert controller._runtime is None
    assert controller._hook is None
    controller.close()
    assert runtime.closed == 1
    print("vllm_lifecycle=pass")


if __name__ == "__main__":
    main()
