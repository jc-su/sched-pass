from nta_runtime.adapters.sglang import (
    SglangAdapter,
    SglangExecutionConfig,
    SglangForwardMetadata,
)
from nta_runtime.adapters.base import ConsumerContract, ConsumerKind, EngineBoundary
from nta_runtime.adapters.vllm import VllmAdapter, VllmSchedulerProjection
from nta_runtime.adapters.vllm_v1 import (
    VllmV1Hook,
    VllmV1SchedulerProjection,
    validate_vllm_attention_tier,
)
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
    _nta_forward_metadata = SglangForwardMetadata((5, 7), (2, 5), (3, 7))


class FakeVllmSchedulerOutput:
    request_ids = ("vllm-a", "vllm-b")
    request_slots = (4, 6)
    priorities = (1, 3)
    deadline_clocks = (100, 200)
    tenant_ids = (4, 6)
    block_tables = ((10, 11), (20, 21, 22))
    kv_page_bytes = 4096


class FakeVllmV1SchedulerOutput:
    # Deliberately differ from InputBatch.req_ids: scheduler mapping order is
    # not the attention row order after vLLM compacts its persistent batch.
    num_scheduled_tokens = {"v1-b": 1, "v1-a": 1}
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
    req_ids = ("v1-a", "v1-b")
    req_id_to_index = {"v1-a": 1, "v1-b": 3}
    block_table = FakeVllmV1BlockTable()


class FakeVllmV1Consumer:
    def __init__(self) -> None:
        self.calls = []

    def consumer_contract(self):
        return ConsumerContract.native_work_unit(
            engine="vllm", backend="nta_flashinfer", engine_version="0.26.0"
        )

    def consume(self, batch, **attention_args):
        self.calls.append((batch, attention_args))
        return "attention-output"


class FakeVllmV2Row:
    def __init__(self, values):
        self.values = tuple(values)

    def tolist(self):
        return list(self.values)


class FakeVllmV2Table:
    rows = ((0, 0, 0), (30, 31, 0), (40, 41, 42))

    def __getitem__(self, key):
        row, column = key
        return FakeVllmV2Row(self.rows[row][column])


def main() -> None:
    assert validate_vllm_attention_tier({}) == "host_staged"
    assert validate_vllm_attention_tier({"NTA_SERVING_TIER": "host"}) == "host_staged"
    for value in ("nvme", "cxl", "cxl_dax"):
        try:
            validate_vllm_attention_tier({"NTA_SERVING_TIER": value})
        except RuntimeError as error:
            assert "physical tier" in str(error)
        else:
            raise AssertionError("vLLM physical tier was accepted without native mode")
        assert (
            validate_vllm_attention_tier(
                {
                    "NTA_SERVING_TIER": value,
                    "NTA_VLLM_NATIVE": "1",
                    "NTA_VLLM_PHYSICAL_CATALOG": "1",
                }
            )
            == ("cxl_dax" if value == "cxl" else value)
        )
        try:
            validate_vllm_attention_tier(
                {"NTA_SERVING_TIER": value, "NTA_VLLM_NATIVE": "1"}
            )
        except RuntimeError as error:
            assert "PHYSICAL_CATALOG" in str(error)
        else:
            raise AssertionError("vLLM physical replay profile was implicit")

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
    padded = FakeForward._nta_forward_metadata.pad((5, 7, 8, 9))
    assert padded.request_slots == (5, 7, 8, 9)
    assert padded.priorities == (2, 5, 0, 0)
    assert padded.tenant_ids == (3, 7, 0, 0)
    try:
        FakeForward._nta_forward_metadata.pad((5, 6, 8, 9))
    except ValueError as error:
        assert "live request order" in str(error)
    else:
        raise AssertionError("graph replay accepted reordered live request slots")
    try:
        SglangForwardMetadata((5,), (2, 5), (3, 7))
    except ValueError as error:
        assert "aligned" in str(error)
    else:
        raise AssertionError("unaligned SGLang forward metadata was accepted")
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
    phase = vllm_batch.phase(1, 1)
    assert phase.request_ids == (vllm_batch.request_ids[1],)
    assert phase.bindings[0].request_index == 0
    assert phase.bindings[0].request_slot == vllm_batch.bindings[1].request_slot
    assert phase.exact_demand is not None
    assert phase.exact_demand.request_unit_ids == ((20, 21, 22),)
    try:
        vllm_batch.phase(2, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range engine phase was accepted")
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

    v2_projection = VllmV1SchedulerProjection.from_v2_forward(
        type(
            "SchedulerOutput",
            (),
            {"num_scheduled_tokens": {"v2-b": 1, "v2-a": 1}},
        )(),
        type(
            "InputBatch",
            (),
            {"req_ids": ("v2-a", "v2-b"), "idx_mapping_np": (2, 1)},
        )(),
        block_tables=(FakeVllmV2Table(),),
        num_blocks=((3, 2, 3),),
        page_bytes=4096,
    )
    assert v2_projection.request_ids == ("v2-a", "v2-b")
    assert v2_projection.block_tables == ((40, 41, 42), (30, 31))
    assert v2_projection.request_rows == (2, 1)
    assert v2_projection.exact_demand().unit_bytes == 4096
    v2_hook = VllmV1Hook(
        runtime,
        2,
        page_bytes=4096,
        version_provider=lambda: "0.26.0",
    )
    v2_hook.bind_v2_forward(
        type("SchedulerOutput", (), {"num_scheduled_tokens": {"v2-a": 1, "v2-b": 1}})(),
        type(
            "InputBatch",
            (),
            {"req_ids": ("v2-a", "v2-b"), "idx_mapping_np": (2, 1)},
        )(),
        block_tables=(FakeVllmV2Table(),),
        num_blocks=((3, 2, 3),),
        epoch=10,
    )
    replacement_batch = v2_hook.bind_v2_forward(
        type("SchedulerOutput", (), {"num_scheduled_tokens": {"v2-c": 1, "v2-d": 1}})(),
        type(
            "InputBatch",
            (),
            {"req_ids": ("v2-c", "v2-d"), "idx_mapping_np": (2, 1)},
        )(),
        block_tables=(FakeVllmV2Table(),),
        num_blocks=((3, 2, 3),),
        epoch=11,
    )
    assert len(replacement_batch.bindings) == 2

    # Persistent vLLM rows can swap during compaction.  A two-way swap must
    # preserve both live identities so the following replacement can reclaim
    # both rows without exhausting the bounded adapter slots.
    swap_hook = VllmV1Hook(
        runtime,
        2,
        page_bytes=4096,
        version_provider=lambda: "0.26.0",
    )
    def swap_input(ids, rows):
        return type("InputBatch", (), {"req_ids": ids, "idx_mapping_np": rows})()
    swap_hook.bind_v2_forward(
        type("SchedulerOutput", (), {"num_scheduled_tokens": {"swap-a": 1, "swap-b": 1}})(),
        swap_input(("swap-a", "swap-b"), (1, 2)),
        block_tables=(FakeVllmV2Table(),),
        num_blocks=((3, 2, 3),),
        epoch=12,
    )
    swap_hook.bind_v2_forward(
        type("SchedulerOutput", (), {"num_scheduled_tokens": {"swap-a": 1, "swap-b": 1}})(),
        swap_input(("swap-a", "swap-b"), (2, 1)),
        block_tables=(FakeVllmV2Table(),),
        num_blocks=((3, 2, 3),),
        epoch=13,
    )
    swap_replacement = swap_hook.bind_v2_forward(
        type("SchedulerOutput", (), {"num_scheduled_tokens": {"swap-c": 1, "swap-d": 1}})(),
        swap_input(("swap-c", "swap-d"), (2, 1)),
        block_tables=(FakeVllmV2Table(),),
        num_blocks=((3, 2, 3),),
        epoch=14,
    )
    assert len(swap_replacement.bindings) == 2

    v1_hook = VllmV1Hook(
        runtime,
        4,
        page_bytes=4096,
        version_provider=lambda: "0.26.0",
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
        version_provider=lambda: "0.26.0",
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
        engine="sglang", backend="nta_flashinfer", engine_version="0.5.16"
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
