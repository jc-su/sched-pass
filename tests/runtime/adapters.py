from nta_runtime.adapters.sglang import SglangAdapter, SglangExecutionConfig
from nta_runtime.adapters.base import ConsumerContract, ConsumerKind, EngineBoundary
from nta_runtime.adapters.vllm import VllmAdapter, VllmSchedulerProjection
from nta_runtime.adapters.vllm_v1 import VllmV1Hook
from nta_runtime.execution_protocol import ProtocolKind
from nta_runtime.work_unit import Granularity


class FakeRuntime:
    def __init__(self) -> None:
        self.published = []
        self.tenant_ids = []
        self.cancelled = []

    def set_request(
        self, slot, request_id, generation, *, tenant_id, priority, deadline_clock
    ):
        self.published.append((slot, request_id, generation, priority, deadline_clock))
        self.tenant_ids.append(tenant_id)

    def cancel_request(self, slot, generation):
        self.cancelled.append((slot, generation))


class FakeForward:
    batch_size = 2
    rids = ("sg-a", "sg-b")
    req_pool_indices = (5, 7)
    _nta_request_priorities = (2, 5)
    _nta_request_tenant_ids = (3, 7)


class FakeVllmSchedulerOutput:
    request_ids = ("vllm-a", "vllm-b")
    request_slots = (4, 6)
    priorities = (1, 3)
    deadline_clocks = (100, 200)
    tenant_ids = (4, 6)
    block_tables = ((10, 11), (20, 21, 22))
    kv_page_bytes = 4096


class FakeVllmV1SchedulerOutput:
    num_scheduled_tokens = {"v1-a": 1, "v1-b": 1}
    finished_req_ids = set()


class FakeVllmV1BlockGroup:
    num_blocks_per_row = (0, 2, 0, 3)

    def get_numpy_array(self):
        return __import__("numpy").array(
            [[0, 0, 0], [10, 11, 0], [0, 0, 0], [20, 21, 22]],
            dtype="int32",
        )


class FakeVllmV1BlockTable:
    block_tables = (FakeVllmV1BlockGroup(),)

    def __getitem__(self, index):
        return self.block_tables[index]


class FakeVllmV1InputBatch:
    req_id_to_index = {"v1-a": 1, "v1-b": 3}
    block_table = FakeVllmV1BlockTable()


class FakeVllmV1Consumer:
    def __init__(self) -> None:
        self.calls = []

    def consumer_contract(self):
        return ConsumerContract.native_work_unit(
            engine="vllm", backend="nta_flashinfer", engine_version="0.13.0"
        )

    def consume(self, batch, **attention_args):
        self.calls.append((batch, attention_args))
        return "attention-output"


def main() -> None:
    config = SglangExecutionConfig.from_environment(
        {
            "NTA_EXECUTION_PROTOCOL": "partial",
            "NTA_EXECUTION_GRANULARITY": "cta_tile",
            "NTA_EXECUTION_MAX_INFLIGHT_UNITS": "32",
        }
    )
    assert config.protocol.kind is ProtocolKind.PARTIAL
    assert config.protocol.max_inflight_units == 32

    runtime = FakeRuntime()
    adapter = VllmAdapter(runtime, 8)
    assert isinstance(adapter, EngineBoundary)
    batch = adapter.bind_batch(
        ("a", "b"),
        (1, 3),
        epoch=4,
        granularity=Granularity.PAGE_GROUP,
    )
    assert batch.engine == "vllm"
    assert batch.epoch == 4
    assert tuple((item.request_slot, item.generation) for item in batch.bindings) == (
        (1, 1),
        (3, 1),
    )
    assert len(runtime.published) == 2
    assert adapter.cancel_matching("a") == 1
    assert runtime.cancelled == [(1, 1)]

    sglang = SglangAdapter(runtime, 8)
    assert isinstance(sglang, EngineBoundary)
    sglang_batch = sglang.bind_forward(
        FakeForward(),
        allow_capture_ids=False,
        stream=None,
        epoch=9,
        granularity=Granularity.CTA_TILE,
    )
    assert sglang_batch.engine == "sglang"
    assert sglang_batch.epoch == 9
    assert tuple(item.priority for item in sglang_batch.bindings) == (2, 5)
    assert tuple(item.tenant_id for item in sglang_batch.bindings) == (3, 7)
    assert sglang_batch.tenant_ids == (3, 7)
    assert tuple(item.request_slot for item in sglang_batch.bindings) == (5, 7)
    vllm_batch = adapter.bind_forward(
        FakeVllmSchedulerOutput(), epoch=5, granularity=Granularity.LAYER
    )
    assert tuple(item.request_slot for item in vllm_batch.bindings) == (4, 6)
    assert tuple(item.deadline_clock for item in vllm_batch.bindings) == (100, 200)
    assert tuple(item.tenant_id for item in vllm_batch.bindings) == (4, 6)
    assert vllm_batch.tenant_ids == (4, 6)
    assert vllm_batch.exact_demand is not None
    assert vllm_batch.exact_demand.request_unit_ids == ((10, 11), (20, 21, 22))
    assert vllm_batch.exact_demand.unit_bytes == 4096
    projection = VllmSchedulerProjection.from_scheduler_output(
        FakeVllmSchedulerOutput()
    )
    assert projection.request_ids == ("vllm-a", "vllm-b")
    try:
        projection_without_demand = VllmSchedulerProjection(
            projection.request_ids, projection.request_slots
        )
        projection_without_demand.exact_demand()
    except RuntimeError as error:
        assert "exact block_tables" in str(error)
    else:
        raise AssertionError("vLLM identity-only projection was accepted")

    v1_hook = VllmV1Hook(
        runtime,
        4,
        page_bytes=4096,
        version_provider=lambda: "0.13.0",
        tenant_for_request=lambda request_id: 9 if request_id == "v1-a" else 11,
    )
    v1_batch = v1_hook.bind_forward(
        FakeVllmV1SchedulerOutput(),
        FakeVllmV1InputBatch(),
        epoch=6,
    )
    assert tuple(item.request_slot for item in v1_batch.bindings) == (0, 1)
    assert v1_batch.exact_demand is not None
    assert v1_batch.exact_demand.request_unit_ids == ((10, 11), (20, 21, 22))
    assert v1_batch.tenant_ids == (9, 11)
    projection_contract = v1_hook.projection_contract()
    assert projection_contract.kind is ConsumerKind.PROJECTION_ONLY
    assert projection_contract.formal_execution is False
    try:
        v1_hook.consume_forward(
            FakeVllmV1SchedulerOutput(),
            FakeVllmV1InputBatch(),
            epoch=6,
        )
    except RuntimeError as error:
        assert "numerical attention consumer" in str(error)
    else:
        raise AssertionError("projection-only vLLM hook executed attention")
    consumer = FakeVllmV1Consumer()
    consuming_hook = VllmV1Hook(
        runtime,
        4,
        page_bytes=4096,
        version_provider=lambda: "0.13.0",
        consumer=consumer,
    )
    assert consuming_hook.consumer_contract().kind is ConsumerKind.NATIVE_WORK_UNIT
    assert (
        consuming_hook.consume_forward(
            FakeVllmV1SchedulerOutput(),
            FakeVllmV1InputBatch(),
            epoch=8,
            attention_metadata="typed-metadata",
        )
        == "attention-output"
    )
    assert len(consumer.calls) == 1
    assert consumer.calls[0][0].exact_demand is not None
    assert consumer.calls[0][1] == {"attention_metadata": "typed-metadata"}
    native_contract = ConsumerContract.native_work_unit(
        engine="sglang", backend="nta_flashinfer", engine_version="0.5.14"
    )
    assert native_contract.as_dict()["numerical_consumer"] is True
    try:
        ConsumerContract(
            engine="vllm",
            backend="test",
            kind="native_work_unit",  # type: ignore[arg-type]
            exact_demand=True,
            typed_work_plan=True,
            native_submission=True,
            numerical_consumer=True,
        )
    except TypeError as error:
        assert "kind" in str(error)
    else:
        raise AssertionError("raw consumer kind was accepted")
    try:
        v1_hook.bind_forward(
            FakeVllmV1SchedulerOutput(),
            FakeVllmV1InputBatch(),
            epoch=7,
        )
    except RuntimeError:
        raise AssertionError("stable vLLM V1 requests were not rebindable")
    print("vllm_v1_hook=pass")
    print("adapters=pass")


if __name__ == "__main__":
    main()
