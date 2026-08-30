from dataclasses import dataclass

from nta_runtime.nvme_materialization import (
    NvmeSlotLifetime,
    NvmeTensorLane,
    materialize_nvme_run_plan,
    plan_nvme_runs,
    publish_registered_nvme_objects,
    publish_nvme_runs,
    summarize_nvme_runs,
)
from nta_runtime.nvme_granularity import (
    NvmeGranularity,
    NvmeTransferServiceModel,
    choose_nvme_granularity,
    plan_nvme_spans,
)
from nta_runtime.runtime import RegisteredNvmeObjectInstall


@dataclass(frozen=True)
class _Extent:
    offset: int
    bytes: int


class _Event:
    def __init__(self) -> None:
        self.recorded_streams: list[object] = []

    def record(self, stream: object) -> None:
        self.recorded_streams.append(stream)


class _Runtime:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.batch_calls = 0

    def install_registered_nvme_objects_async(
        self, objects: tuple[object, ...], stream: object
    ) -> tuple[int, ...]:
        self.batch_calls += 1
        for object_ in objects:
            self.calls.append(
                (
                    object_.slot,
                    object_.object_id,
                    object_.version,
                    object_.source_byte_offset,
                    object_.bytes,
                    object_.region,
                    object_.destination_device_address,
                    stream,
                    object_.prior_consumer_event,
                )
            )
        return tuple(object_.destination_device_address for object_ in objects)


@dataclass(frozen=True)
class _Region:
    address: int
    bytes: int


def main() -> None:
    service = NvmeTransferServiceModel(
        command_service_ns=13_000,
        read_bandwidth_bytes_per_second=6_700_000_000,
        compaction_bandwidth_bytes_per_second=600_000_000_000,
        compaction_launch_ns=20_000,
    )
    sparse = (tuple(range(0, 32, 2)), tuple(range(16)))
    sparse_direct = plan_nvme_runs(
        (sparse,),
        lane_element_bytes=(4096, 4096),
        lba_size=512,
        max_transfer_bytes=2 * 1024 * 1024,
        object_capacity=64,
    )
    sparse_span = plan_nvme_spans(
        (sparse,),
        lane_element_bytes=(4096, 4096),
        lba_size=512,
        max_transfer_bytes=2 * 1024 * 1024,
        scratch_alignment=4096,
        service_model=service,
    )
    assert sparse_direct.object_count == 32
    assert sparse_span.object_count == 2
    assert sparse_span.physical_bytes == 31 * 2 * 4096
    assert sparse_span.exact_bytes == 16 * 2 * 4096
    assert sparse_span.scratch_bytes == sparse_span.physical_bytes
    assert sparse_span.selected_row_copy_count == 32
    assert len(sparse_span.spans_for(sparse)) == 1
    # The linear sliding-window planner must retain the globally minimal
    # conservative service objective, not substitute a local gap heuristic.
    sources = (0, 2, 3, 9, 11, 12, 18)
    linear_pair = (sources, tuple(range(len(sources))))
    linear_model = NvmeTransferServiceModel(
        command_service_ns=10,
        read_bandwidth_bytes_per_second=1_000_000_000,
        compaction_bandwidth_bytes_per_second=1_000_000_000,
    )
    linear = plan_nvme_spans(
        (linear_pair,),
        lane_element_bytes=(4, 4),
        lba_size=4,
        max_transfer_bytes=32,
        scratch_alignment=4,
        service_model=linear_model,
    )
    row_ns = 8
    fixed_ns = 20
    brute: list[tuple[int, int, int] | None] = [None] * (len(sources) + 1)
    brute[0] = (0, 0, 0)
    for begin in range(len(sources)):
        assert brute[begin] is not None
        for end in range(begin, len(sources)):
            rows = sources[end] - sources[begin] + 1
            if rows > 8:
                break
            current = brute[begin]
            assert current is not None
            candidate = (
                current[0] + fixed_ns + rows * row_ns,
                current[1] + rows * 8,
                current[2] + 2,
            )
            if brute[end + 1] is None or candidate < brute[end + 1]:
                brute[end + 1] = candidate
    linear_objective = (
        len(linear.spans) * fixed_ns
        + sum(span.source_row_count for span in linear.spans) * row_ns,
        linear.physical_bytes,
        linear.command_count,
    )
    assert linear_objective == brute[-1]
    sparse_decision = choose_nvme_granularity(
        direct_command_count=sparse_direct.object_count,
        direct_transfer_bytes=16 * 2 * 4096,
        direct_work_item_count=len(sparse_direct.unique_runs),
        span_command_count=sparse_span.command_count,
        span_transfer_bytes=sparse_span.physical_bytes,
        span_exact_bytes=sparse_span.exact_bytes,
        span_work_item_count=len(sparse_span.spans),
        span_scratch_bytes=sparse_span.scratch_bytes,
        compaction_launch_count=1,
        object_capacity=64,
        work_ticket_capacity=64,
        scratch_capacity_bytes=1 << 20,
        service_model=service,
    )
    assert sparse_decision.kind is NvmeGranularity.SPAN_COMPACT
    assert sparse_decision.reason == "service_cost"
    assert sparse_decision.direct_predicted_ns > sparse_decision.span_predicted_ns

    dense = (tuple(range(16)), tuple(range(16)))
    dense_direct = plan_nvme_runs(
        (dense,),
        lane_element_bytes=(4096, 4096),
        lba_size=512,
        max_transfer_bytes=2 * 1024 * 1024,
        object_capacity=64,
    )
    dense_span = plan_nvme_spans(
        (dense,),
        lane_element_bytes=(4096, 4096),
        lba_size=512,
        max_transfer_bytes=2 * 1024 * 1024,
        scratch_alignment=4096,
        service_model=service,
    )
    dense_decision = choose_nvme_granularity(
        direct_command_count=dense_direct.object_count,
        direct_transfer_bytes=16 * 2 * 4096,
        direct_work_item_count=len(dense_direct.unique_runs),
        span_command_count=dense_span.command_count,
        span_transfer_bytes=dense_span.physical_bytes,
        span_exact_bytes=dense_span.exact_bytes,
        span_work_item_count=len(dense_span.spans),
        span_scratch_bytes=dense_span.scratch_bytes,
        compaction_launch_count=1,
        object_capacity=64,
        work_ticket_capacity=64,
        scratch_capacity_bytes=1 << 20,
        service_model=service,
    )
    assert dense_direct.object_count == dense_span.object_count == 2
    assert dense_decision.kind is NvmeGranularity.DIRECT
    assert dense_decision.reason == "insufficient_gain"

    uncalibrated = choose_nvme_granularity(
        direct_command_count=sparse_direct.object_count,
        direct_transfer_bytes=16 * 2 * 4096,
        direct_work_item_count=len(sparse_direct.unique_runs),
        span_command_count=0,
        span_transfer_bytes=0,
        span_exact_bytes=0,
        span_work_item_count=0,
        span_scratch_bytes=0,
        compaction_launch_count=1,
        object_capacity=64,
        work_ticket_capacity=64,
        scratch_capacity_bytes=1 << 20,
        service_model=NvmeTransferServiceModel(),
    )
    assert uncalibrated.kind is NvmeGranularity.DIRECT
    assert uncalibrated.reason == "uncalibrated"
    scratch_limited = choose_nvme_granularity(
        direct_command_count=sparse_direct.object_count,
        direct_transfer_bytes=16 * 2 * 4096,
        direct_work_item_count=len(sparse_direct.unique_runs),
        span_command_count=sparse_span.command_count,
        span_transfer_bytes=sparse_span.physical_bytes,
        span_exact_bytes=sparse_span.exact_bytes,
        span_work_item_count=len(sparse_span.spans),
        span_scratch_bytes=sparse_span.scratch_bytes,
        compaction_launch_count=1,
        object_capacity=64,
        work_ticket_capacity=64,
        scratch_capacity_bytes=sparse_span.scratch_bytes - 1,
        service_model=service,
    )
    assert scratch_limited.kind is NvmeGranularity.DIRECT
    assert scratch_limited.reason == "scratch_capacity"

    # The second pair is a strict subset of the first. Its transfer run must
    # fan out from the same acquisition owner instead of consuming another
    # directory slot or issuing duplicate NVMe I/O.
    complete = ((0, 1, 2, 3), (10, 11, 12, 13))
    shared_prefix = ((0, 1), (10, 11))
    plan = plan_nvme_runs(
        (complete, shared_prefix, complete),
        lane_element_bytes=(4, 8),
        lba_size=4,
        max_transfer_bytes=16,
        object_capacity=4,
    )
    assert len(plan.pair_runs) == 2
    assert len(plan.unique_runs) == 2
    assert plan.object_count == 4
    assert plan.runs_for(shared_prefix) == (plan.unique_runs[0],)
    summary = summarize_nvme_runs(
        (complete, shared_prefix, complete),
        lane_element_bytes=(4, 8),
        lba_size=4,
        max_transfer_bytes=16,
    )
    assert materialize_nvme_run_plan(summary, object_capacity=4) == plan

    try:
        plan_nvme_runs(
            (complete,),
            lane_element_bytes=(4, 8),
            lba_size=4,
            max_transfer_bytes=16,
            object_capacity=3,
        )
    except RuntimeError as error:
        assert "object slots" in str(error)
    else:
        raise AssertionError("NVMe run planning exceeded the object directory")

    try:
        plan_nvme_runs(
            (((0,), (10,)),),
            lane_element_bytes=(6,),
            lba_size=4,
            max_transfer_bytes=24,
            object_capacity=1,
        )
    except RuntimeError as error:
        assert "LBA materializable" in str(error)
    else:
        raise AssertionError("a partial-LBA NVMe run was accepted")

    key_region = _Region(1_000, 128)
    value_region = _Region(2_000, 256)
    lanes = (
        NvmeTensorLane("key", 1_000, 32, 4, 4, key_region),
        NvmeTensorLane("value", 2_000, 32, 8, 8, value_region),
    )
    resolved: list[tuple[int, tuple[int, ...], str, int]] = []

    def extent(
        layer: int, ordinals: tuple[int, ...], component: str, row_bytes: int
    ) -> _Extent:
        resolved.append((layer, ordinals, component, row_bytes))
        component_base = 0 if component == "key" else 100_000
        return _Extent(
            component_base + ordinals[0] * row_bytes, len(ordinals) * row_bytes
        )

    event = _Event()
    lifetime = NvmeSlotLifetime(event)
    runtime = _Runtime()
    stream = object()
    first = publish_nvme_runs(
        plan,
        lanes,
        runtime=runtime,
        extent_resolver=extent,
        layer_id=7,
        object_version=1,
        object_id_base=0x4E54410000000000,
        stream=stream,
        lifetime=lifetime,
    )
    assert first.object_count == 4
    assert first.transfer_bytes == 48
    assert len(runtime.calls) == 4
    assert runtime.batch_calls == 1
    assert first.counters.fresh_slots == 4
    assert first.counters.quiesced_replacements == 0
    complete_objects = first.objects_for(complete)
    prefix_objects = first.objects_for(shared_prefix)
    assert prefix_objects == complete_objects[:1]
    assert tuple(object_.slot for object_ in prefix_objects[0]) == (0, 1)

    lifetime.record_retirement(stream)
    assert event.recorded_streams == [stream]
    second = publish_nvme_runs(
        plan,
        lanes,
        runtime=runtime,
        extent_resolver=extent,
        layer_id=7,
        object_version=2,
        object_id_base=0x4E54410000000000,
        stream=stream,
        lifetime=lifetime,
    )
    assert second.counters.same_destination_slots == 4
    assert second.counters.quiesced_replacements == 4
    assert runtime.batch_calls == 2
    assert all(call[-1] is event for call in runtime.calls[-4:])

    scoped = publish_nvme_runs(
        plan,
        lanes,
        runtime=runtime,
        extent_resolver=extent,
        layer_id=7,
        object_version=2,
        object_id_base=0x4E54410000000004,
        first_object_slot=4,
        stream=stream,
        lifetime=lifetime,
    )
    assert tuple(slot for slot, _address in scoped.slot_destinations) == (4, 5, 6, 7)
    assert tuple(
        object_.object_id
        for _run, objects in scoped.objects_by_run
        for object_ in objects
    ) == tuple(0x4E54410000000004 + index for index in range(4))
    assert runtime.batch_calls == 3

    calls_before_rejected_reuse = len(runtime.calls)
    try:
        publish_nvme_runs(
            plan,
            lanes,
            runtime=runtime,
            extent_resolver=extent,
            layer_id=7,
            object_version=3,
            object_id_base=0x4E54410000000000,
            stream=stream,
            lifetime=lifetime,
        )
    except RuntimeError as error:
        assert "prior-consumer event" in str(error)
    else:
        raise AssertionError("NVMe slots were replaced before their consumer retired")
    assert len(runtime.calls) == calls_before_rejected_reuse

    # One retirement event is field-scoped over all slots. Independent tenant
    # scopes may consume disjoint subsets in consecutive publication calls.
    lifetime.record_retirement(stream)
    publish_nvme_runs(
        plan,
        lanes,
        runtime=runtime,
        extent_resolver=extent,
        layer_id=8,
        object_version=4,
        object_id_base=0x4E54410000000000,
        first_object_slot=0,
        stream=stream,
        lifetime=lifetime,
    )
    publish_nvme_runs(
        plan,
        lanes,
        runtime=runtime,
        extent_resolver=extent,
        layer_id=8,
        object_version=4,
        object_id_base=0x4E54410000000004,
        first_object_slot=4,
        stream=stream,
        lifetime=lifetime,
    )
    assert all(call[-1] is event for call in runtime.calls[-8:])

    # Every reused slot needs the consumer proof. Passing it only for slot 0
    # would make the native runtime reject slot 1 on the next model layer.
    shared_event = object()
    direct_runtime = _Runtime()
    direct_bindings = (
        RegisteredNvmeObjectInstall(
            0, 100, 9, 0, 16, _Region(4_000, 64), 4_000, shared_event
        ),
        RegisteredNvmeObjectInstall(
            1, 101, 9, 16, 16, _Region(5_000, 64), 5_016, shared_event
        ),
    )
    assert publish_registered_nvme_objects(
        direct_bindings,
        runtime=direct_runtime,
        stream=stream,
    ) == (4_000, 5_016)
    assert [call[-1] for call in direct_runtime.calls] == [shared_event, shared_event]
    assert direct_runtime.batch_calls == 1

    try:
        publish_registered_nvme_objects(
            (
                direct_bindings[0],
                RegisteredNvmeObjectInstall(
                    2, 102, 10, 32, 16, _Region(6_000, 64), 6_000, shared_event
                ),
            ),
            runtime=direct_runtime,
            stream=stream,
        )
    except ValueError as error:
        assert "contiguous" in str(error)
    else:
        raise AssertionError("a non-contiguous NVMe directory batch was accepted")
    assert direct_runtime.batch_calls == 1

    try:
        RegisteredNvmeObjectInstall(0, 100, 1, 0, 32, _Region(4_000, 16), 4_000, None)
    except ValueError as error:
        assert "registered HBM region" in str(error)
    else:
        raise AssertionError("an out-of-region NVMe destination was accepted")

    print("nvme_materialization=pass")


if __name__ == "__main__":
    main()
