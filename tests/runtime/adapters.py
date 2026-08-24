from nta_runtime.adapters.sglang import SglangAdapter, SglangExecutionConfig
from nta_runtime.adapters.vllm import VllmAdapter
from nta_runtime.execution_protocol import ProtocolKind
from nta_runtime.requests import RequestBinding
from nta_runtime.work_unit import Granularity


class FakeRuntime:
    def __init__(self) -> None:
        self.published = []
        self.cancelled = []

    def set_request(self, slot, request_id, generation, *, priority, deadline_clock):
        self.published.append((slot, request_id, generation, priority, deadline_clock))

    def cancel_request(self, slot, generation):
        self.cancelled.append((slot, generation))


class FakeForward:
    batch_size = 2
    rids = ("sg-a", "sg-b")
    _nta_request_priorities = (2, 5)


def main() -> None:
    config = SglangExecutionConfig.from_environment(
        {
            "NTA_SGLANG_EXECUTION_PROTOCOL": "partial",
            "NTA_SGLANG_WORK_GRANULARITY": "cta_tile",
            "NTA_SGLANG_MAX_INFLIGHT_UNITS": "32",
        }
    )
    assert config.protocol.kind is ProtocolKind.PARTIAL
    assert config.protocol.max_inflight_units == 32

    runtime = FakeRuntime()
    adapter = VllmAdapter(runtime, 8)
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
    assert tuple(item.request_slot for item in sglang_batch.bindings) == (0, 1)
    print("adapters=pass")


if __name__ == "__main__":
    main()
