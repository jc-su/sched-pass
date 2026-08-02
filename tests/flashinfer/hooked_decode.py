#!/usr/bin/env python3
"""Execute resident and pinned-host deferred paths in hooked FlashInfer decode."""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import statistics

import flashinfer
import torch
from nta_runtime import (
    AcquireRequirement,
    DeviceWorkPlan,
    FlashInferLayerEpoch,
    JitPhaseProgram,
    Placement,
    Replica,
    RequestRange,
    Runtime,
    RuntimeConfig,
    WorkItem,
)
from tools.flashinfer.schedule import decode_schedule, paged_prefill_schedule


OBJECT_ID = 0xC001
GENERATION = 1
TENSOR_NAMES = ["nta_runtime", "nta_work_items", "nta_dependencies"]
TENSOR_DTYPES = ["uint8_t", "uint8_t", "uint8_t"]
SCALAR_NAMES = ["sm_scale", "nta_work_count", "nta_skip_merge"]
SCALAR_DTYPES = ["double", "int64_t", "int64_t"]
VARIANT_NAME = "DefaultAttention<false, false, false, false>"
VARIANT_DECL = "#include <flashinfer/attention/variants.cuh>"


class RuntimeFixture:
    def __init__(
        self,
        kv: torch.Tensor,
        host_source: torch.Tensor | None,
        work_count: int = 1,
        request_indices: list[int] | None = None,
        source_indices: torch.Tensor | None = None,
        destination_indices: torch.Tensor | None = None,
        direct_work_indices: set[int] | None = None,
        partitioned_objects: bool = False,
    ) -> None:
        if work_count <= 0:
            raise ValueError("work_count must be positive")
        self.work_count = work_count
        if request_indices is None:
            request_indices = [0] * work_count
        if len(request_indices) != work_count or min(request_indices) < 0:
            raise ValueError("request_indices must identify every work item")
        request_count = max(request_indices) + 1
        self.kv = kv
        self.host_source = host_source
        byte_count = kv.numel() * kv.element_size()
        indexed = source_indices is not None or destination_indices is not None
        if partitioned_objects and (indexed or host_source is None):
            raise ValueError(
                "partitioned objects need a host source and implicit row maps"
            )
        if indexed:
            if host_source is None or source_indices is None or destination_indices is None:
                raise ValueError("indexed transfers need a host source and both maps")
            if source_indices.numel() != destination_indices.numel():
                raise ValueError("indexed transfer maps must have equal length")
            self.source_indices = source_indices.to(device="cuda", dtype=torch.int32)
            self.destination_indices = destination_indices.to(
                device="cuda", dtype=torch.int32
            )
            element_bytes = kv[0].numel() * kv.element_size()
            transfer_bytes = self.source_indices.numel() * element_bytes
        else:
            self.source_indices = None
            self.destination_indices = None
            element_bytes = 0
            transfer_bytes = byte_count
        object_count = work_count if partitioned_objects else 1
        self.native_runtime = Runtime(
            RuntimeConfig(
                request_capacity=request_count,
                object_capacity=object_count,
                intent_capacity=object_count,
                work_ticket_capacity=work_count,
                max_dependencies_per_work_ticket=1,
            )
        )
        self.native_runtime.set_tenant_budget(0, 2 * byte_count)
        for index in range(request_count):
            self.native_runtime.set_request(
                index,
                17 + index,
                GENERATION,
                priority=4,
                max_outstanding_bytes=2 * byte_count,
            )

        object_ids = [OBJECT_ID] * work_count
        object_slots = [0] * work_count
        transfer_sizes = [transfer_bytes] * work_count
        direct_bases = [0] * work_count
        self.partition_indices: list[torch.Tensor] = []
        if partitioned_objects:
            if kv.shape[0] % work_count != 0:
                raise ValueError("KV rows do not divide across partitioned objects")
            rows_per_object = kv.shape[0] // work_count
            element_bytes = kv[0].numel() * kv.element_size()
            transfer_bytes = rows_per_object * element_bytes
            for index in range(work_count):
                rows = torch.arange(
                    index * rows_per_object,
                    (index + 1) * rows_per_object,
                    dtype=torch.int32,
                    device="cuda",
                )
                self.partition_indices.append(rows)
                object_id = OBJECT_ID + index
                self.native_runtime.register_indexed_host_object(
                    index,
                    object_id,
                    1,
                    host_source.data_ptr(),
                    kv.data_ptr(),
                    rows.data_ptr(),
                    rows.data_ptr(),
                    rows.numel(),
                    element_bytes,
                    host_source.stride(0) * host_source.element_size(),
                    kv.stride(0) * kv.element_size(),
                )
                object_ids[index] = object_id
                object_slots[index] = index
                transfer_sizes[index] = transfer_bytes
        elif indexed:
            self.native_runtime.register_indexed_host_object(
                0,
                OBJECT_ID,
                1,
                host_source.data_ptr(),
                kv.data_ptr(),
                self.source_indices.data_ptr(),
                self.destination_indices.data_ptr(),
                self.source_indices.numel(),
                element_bytes,
                host_source.stride(0) * host_source.element_size(),
                kv.stride(0) * kv.element_size(),
            )
            direct_bases = [0] * work_count
        else:
            placement = (
                Placement.HOST_STAGED
                if host_source is not None
                else Placement.HBM
            )
            source = host_source if host_source is not None else kv
            direct = self.native_runtime.register_object(
                0,
                OBJECT_ID,
                1,
                byte_count,
                [Replica(source.data_ptr(), placement)],
                staging_device_address=kv.data_ptr() if host_source is not None else 0,
            )
            direct_bases = [direct] * work_count
        dependencies = []
        direct_counts = []
        for index in range(work_count):
            direct_base = direct_bases[index]
            if direct_work_indices is not None:
                direct_base = kv.data_ptr() if index in direct_work_indices else 0
            dependencies.append(
                AcquireRequirement(
                    direct_base,
                    0,
                    object_ids[index],
                    0,
                    object_slots[index],
                    1,
                    transfer_sizes[index],
                    0,
                )
            )
            direct_counts.append(int(direct_base != 0))
        contributor_counts = {
            request_index: request_indices.count(request_index)
            for request_index in set(request_indices)
        }
        contributor_indices = {request_index: 0 for request_index in contributor_counts}
        work_items = []
        for index, request_index in enumerate(request_indices):
            work_items.append(
                WorkItem(
                    request_index,
                    request_index,
                    GENERATION,
                    index,
                    index,
                    1,
                    direct_counts[index],
                    index,
                    request_index,
                    contributor_indices[request_index],
                    contributor_counts[request_index],
                    2500,
                )
            )
            contributor_indices[request_index] += 1
        ranges = []
        begin = 0
        while begin < work_count:
            request_index = request_indices[begin]
            end = begin + 1
            while end < work_count and request_indices[end] == request_index:
                end += 1
            if request_index in request_indices[end:]:
                raise RuntimeError("FlashInfer request work must be contiguous")
            ranges.append(
                RequestRange(begin, end - begin, request_index, GENERATION)
            )
            begin = end
        self.plan = DeviceWorkPlan(
            work_count, work_count, self.native_runtime.device_ordinal
        )
        self.plan.upload(work_items, dependencies, ranges, torch.cuda.current_stream())
        self.plan.wait_on(torch.cuda.current_stream())
        self.runtime = self.native_runtime.device_view_tensor
        self.work_items = self.plan.work_items_tensor
        self.requirements = self.plan.dependencies_tensor

    def work_ticket_state(self, index: int = 0) -> int:
        return self.native_runtime.work_ticket_state(index)

    def assert_all_states(self, expected: int) -> None:
        states = [self.work_ticket_state(index) for index in range(self.work_count)]
        if states != [expected] * self.work_count:
            raise RuntimeError(f"work ticket states {states} do not equal {expected}")


class PhaseFunctions:
    def __init__(
        self, module_name: str = "nta_batch_decode_default_v2_hooked"
    ) -> None:
        workspace = pathlib.Path(os.environ["FLASHINFER_WORKSPACE_BASE"])
        modules = list(workspace.rglob(f"{module_name}.so"))
        if len(modules) != 1:
            raise RuntimeError(f"expected one hooked decode module, found {modules}")
        self.program = JitPhaseProgram(modules[0])

    def call(self, name: str, fixture: RuntimeFixture, *arguments: int) -> None:
        stream = torch.cuda.current_stream()
        runtime = fixture.native_runtime
        if name == "nta_jit_reset_epoch":
            self.program.reset(runtime, arguments[0], arguments[1], stream)
        elif name == "nta_jit_progress_host":
            self.program.progress_host(runtime, arguments[0], stream)
        elif name == "nta_jit_publish_ready":
            self.program.publish(runtime, arguments[0], stream)
        elif name == "nta_jit_complete_launched":
            self.program.complete(runtime, arguments[0], stream)
        else:
            raise ValueError(f"unsupported phase function {name}")


def make_wrapper() -> flashinfer.BatchDecodeWithPagedKVCacheWrapper:
    jit_args = [
        "nta_batch_decode_default_v2_hooked",
        torch.float16,
        torch.float16,
        torch.float16,
        torch.int32,
        128,
        128,
        TENSOR_NAMES,
        TENSOR_DTYPES,
        SCALAR_NAMES,
        SCALAR_DTYPES,
        VARIANT_NAME,
        VARIANT_DECL,
    ]
    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    return flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", backend="fa2", jit_args=jit_args
    )


def make_baseline_wrapper() -> flashinfer.BatchDecodeWithPagedKVCacheWrapper:
    jit_args = [
        "nta_batch_decode_default_v2_baseline",
        torch.float16,
        torch.float16,
        torch.float16,
        torch.int32,
        128,
        128,
        [],
        [],
        ["sm_scale"],
        ["double"],
        VARIANT_NAME,
        VARIANT_DECL,
    ]
    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    return flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", backend="fa2", jit_args=jit_args
    )


def make_prefill_wrapper() -> flashinfer.BatchPrefillWithPagedKVCacheWrapper:
    jit_args = [
        "nta_batch_prefill_default_v2_hooked",
        torch.float16,
        torch.float16,
        torch.float16,
        torch.int32,
        128,
        128,
        TENSOR_NAMES,
        TENSOR_DTYPES,
        SCALAR_NAMES,
        SCALAR_DTYPES,
        VARIANT_NAME,
        VARIANT_DECL,
    ]
    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    return flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="fa2", jit_args=jit_args
    )


def plan(
    wrapper: flashinfer.BatchDecodeWithPagedKVCacheWrapper, pages: int = 4
) -> None:
    wrapper.plan(
        torch.tensor([0, pages], dtype=torch.int32, device="cuda"),
        torch.arange(pages, dtype=torch.int32, device="cuda"),
        torch.tensor([16], dtype=torch.int32, device="cuda"),
        4,
        2,
        128,
        16,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
        disable_split_kv=True,
    )


def plan_uniform_batch(
    wrapper: flashinfer.BatchDecodeWithPagedKVCacheWrapper,
    batch_size: int,
    pages_per_request: int,
) -> None:
    wrapper.plan(
        torch.arange(
            0,
            (batch_size + 1) * pages_per_request,
            pages_per_request,
            dtype=torch.int32,
            device="cuda",
        ),
        torch.arange(
            batch_size * pages_per_request, dtype=torch.int32, device="cuda"
        ),
        torch.full((batch_size,), 16, dtype=torch.int32, device="cuda"),
        4,
        2,
        128,
        16,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )


def plan_prefill(
    wrapper: flashinfer.BatchPrefillWithPagedKVCacheWrapper,
    query_tokens: int,
    pages: int,
) -> None:
    wrapper.plan(
        torch.tensor([0, query_tokens], dtype=torch.int32, device="cuda"),
        torch.tensor([0, pages], dtype=torch.int32, device="cuda"),
        torch.arange(pages, dtype=torch.int32, device="cuda"),
        torch.tensor([16], dtype=torch.int32, device="cuda"),
        4,
        2,
        128,
        16,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
        causal=False,
        disable_split_kv=True,
    )


def benchmark(callable_: object, iterations: int = 2_000) -> float:
    for _ in range(20):
        callable_()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        callable_()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1_000.0 / iterations


def run_hooked(
    wrapper: flashinfer.BatchDecodeWithPagedKVCacheWrapper,
    q: torch.Tensor,
    kv: torch.Tensor,
    fixture: RuntimeFixture,
    output: torch.Tensor,
    skip_merge: bool = False,
) -> None:
    wrapper.run(
        q,
        kv,
        fixture.runtime,
        fixture.work_items,
        fixture.requirements,
        1.0 / math.sqrt(128),
        fixture.work_count,
        int(skip_merge),
        out=output,
    )


def run_prefill_hooked(
    wrapper: flashinfer.BatchPrefillWithPagedKVCacheWrapper,
    q: torch.Tensor,
    kv: torch.Tensor,
    fixture: RuntimeFixture,
    output: torch.Tensor,
    skip_merge: bool = False,
) -> None:
    wrapper.run(
        q,
        kv,
        fixture.runtime,
        fixture.work_items,
        fixture.requirements,
        1.0 / math.sqrt(128),
        fixture.work_count,
        int(skip_merge),
        out=output,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanitizer", action="store_true")
    options = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(7)
    shape = (4, 2, 16, 2, 128)
    host_kv = torch.randn(shape, dtype=torch.float16, pin_memory=True)
    reference_kv = host_kv.to("cuda")
    q = torch.randn((1, 4, 128), dtype=torch.float16, device="cuda")

    stock = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda"), "NHD"
    )
    plan(stock)
    expected = stock.run(q, reference_kv)

    hooked = make_wrapper()
    plan(hooked)
    phases = PhaseFunctions()

    resident = RuntimeFixture(reference_kv, None)
    phases.call("nta_jit_reset_epoch", resident, 1, 1)
    resident_output = torch.empty_like(expected)
    run_hooked(hooked, q, reference_kv, resident, resident_output)
    phases.call("nta_jit_complete_launched", resident, 1)
    torch.cuda.synchronize()
    torch.testing.assert_close(resident_output, expected, rtol=2e-3, atol=2e-3)
    resident.assert_all_states(3)
    if resident.native_runtime.work_runnable_ns(1) != (0,):
        raise RuntimeError("resident FlashInfer work reported an arrival delay")

    staging_kv = torch.zeros_like(reference_kv)
    deferred = RuntimeFixture(staging_kv, host_kv)
    phases.call("nta_jit_reset_epoch", deferred, 1, 1)
    deferred_output = torch.full_like(expected, math.nan)
    run_hooked(hooked, q, staging_kv, deferred, deferred_output)
    phases.call("nta_jit_complete_launched", deferred, 1)
    torch.cuda.synchronize()
    deferred.assert_all_states(1)
    if not torch.isnan(deferred_output).all():
        raise RuntimeError("deferred FlashInfer CTA wrote output before readiness")

    phases.call("nta_jit_progress_host", deferred, 1)
    phases.call("nta_jit_publish_ready", deferred, 1)
    torch.cuda.synchronize()
    deferred.assert_all_states(2)
    if deferred.native_runtime.work_runnable_ns(1)[0] <= 0:
        raise RuntimeError("external FlashInfer work omitted its GPU arrival time")
    run_hooked(hooked, q, staging_kv, deferred, deferred_output)
    phases.call("nta_jit_complete_launched", deferred, 1)
    torch.cuda.synchronize()
    deferred.assert_all_states(3)
    torch.testing.assert_close(staging_kv, reference_kv, rtol=0, atol=0)
    torch.testing.assert_close(deferred_output, expected, rtol=2e-3, atol=2e-3)
    maximum = (deferred_output.float() - expected.float()).abs().max().item()

    mixed_batch = 2
    mixed_pages = 4
    mixed_shape = (mixed_batch * mixed_pages, 2, 16, 2, 128)
    mixed_host_kv = torch.randn(mixed_shape, dtype=torch.float16, pin_memory=True)
    mixed_reference_kv = mixed_host_kv.to("cuda")
    mixed_staging_kv = torch.zeros_like(mixed_reference_kv)
    mixed_staging_kv[:mixed_pages].copy_(mixed_reference_kv[:mixed_pages])
    mixed_q = torch.randn(
        (mixed_batch, 4, 128), dtype=torch.float16, device="cuda"
    )
    plan_uniform_batch(stock, mixed_batch, mixed_pages)
    mixed_expected = stock.run(mixed_q, mixed_reference_kv)
    plan_uniform_batch(hooked, mixed_batch, mixed_pages)
    mixed_schedule = decode_schedule(hooked)
    if mixed_schedule.request_indices != (0, 1):
        raise RuntimeError(f"unexpected heterogeneous schedule {mixed_schedule}")
    mixed = RuntimeFixture(
        mixed_staging_kv,
        mixed_host_kv,
        work_count=2,
        request_indices=[0, 1],
        direct_work_indices={0},
    )
    mixed_output = torch.full_like(mixed_expected, math.nan)
    mixed_epoch = FlashInferLayerEpoch(
        mixed.native_runtime,
        mixed.plan,
        phases.program,
        object_count=1,
        max_progress_passes=1,
    )
    mixed_result = mixed_epoch.run_host(
        hooked,
        mixed_q,
        mixed_staging_kv,
        mixed_output,
        progress_blocks=1,
        stream=torch.cuda.current_stream(),
    )
    if mixed_result.progress_passes != 1:
        raise RuntimeError(f"unexpected heterogeneous progress rounds: {mixed_result}")
    mixed.assert_all_states(3)
    torch.testing.assert_close(mixed_output, mixed_expected, rtol=2e-3, atol=2e-3)

    pipelined_staging_kv = torch.zeros_like(mixed_reference_kv)
    pipelined = RuntimeFixture(
        pipelined_staging_kv,
        mixed_host_kv,
        work_count=2,
        request_indices=[0, 1],
        partitioned_objects=True,
    )
    pipelined_output = torch.full_like(mixed_expected, math.nan)
    pipelined_epoch = FlashInferLayerEpoch(
        pipelined.native_runtime,
        pipelined.plan,
        phases.program,
        object_count=2,
        max_progress_passes=2,
    )
    progress_stream = torch.cuda.Stream(priority=0)
    passes = pipelined_epoch.enqueue_host(
        hooked,
        mixed_q,
        pipelined_staging_kv,
        pipelined_output,
        progress_blocks=(1, 1),
        stream=torch.cuda.current_stream(),
        progress_stream=progress_stream,
    )
    pipelined_result = pipelined_epoch.check(passes, torch.cuda.current_stream())
    if pipelined_result.progress_passes != 2:
        raise RuntimeError(
            f"unexpected pipelined host rounds: {pipelined_result}"
        )
    pipelined.assert_all_states(3)
    torch.testing.assert_close(pipelined_staging_kv, mixed_reference_kv, rtol=0, atol=0)
    torch.testing.assert_close(
        pipelined_output, mixed_expected, rtol=2e-3, atol=2e-3
    )

    plan(hooked)
    indexed_staging = torch.zeros_like(reference_kv)
    indexed = RuntimeFixture(
        indexed_staging,
        host_kv,
        source_indices=torch.tensor([3, 1], dtype=torch.int32),
        destination_indices=torch.tensor([0, 2], dtype=torch.int32),
    )
    phases.call("nta_jit_reset_epoch", indexed, 1, 1)
    indexed_output = torch.full_like(expected, math.nan)
    run_hooked(hooked, q, indexed_staging, indexed, indexed_output)
    phases.call("nta_jit_progress_host", indexed, 1)
    phases.call("nta_jit_publish_ready", indexed, 1)
    torch.cuda.synchronize()
    indexed.assert_all_states(2)
    torch.testing.assert_close(indexed_staging[0], reference_kv[3], rtol=0, atol=0)
    torch.testing.assert_close(indexed_staging[2], reference_kv[1], rtol=0, atol=0)
    if torch.count_nonzero(indexed_staging[1]).item() != 0 or torch.count_nonzero(
        indexed_staging[3]
    ).item() != 0:
        raise RuntimeError("indexed host acquisition overwrote an unselected row")
    if options.sanitizer:
        print(
            f"flashinfer_version={flashinfer.__version__} sanitizer_path=pass "
            f"shared_kv_head_ctas=2 indexed_host=pass ready_wave=pass "
            f"max_abs_error={maximum:.6g}"
        )
        return

    split_pages = 256
    split_shape = (split_pages, 2, 16, 2, 128)
    split_host_kv = torch.randn(split_shape, dtype=torch.float16, pin_memory=True)
    split_reference_kv = split_host_kv.to("cuda")
    plan(stock, split_pages)
    split_expected = stock.run(q, split_reference_kv)
    plan(hooked, split_pages)
    split_schedule = decode_schedule(hooked)
    if split_schedule.work_count <= 1 or set(split_schedule.request_indices) != {0}:
        raise RuntimeError(f"expected a split-K decode plan, got {split_schedule}")
    split_work = split_schedule.work_count
    split_staging_kv = torch.zeros_like(split_reference_kv)
    split = RuntimeFixture(split_staging_kv, split_host_kv, split_work)
    phases.call("nta_jit_reset_epoch", split, 1, split_work)
    split_output = torch.full_like(split_expected, 17)
    # Request a merge even though every split-K CTA defers. The device-side
    # epoch gate must leave the output untouched instead of consuming scratch.
    run_hooked(hooked, q, split_staging_kv, split, split_output)
    phases.call("nta_jit_complete_launched", split, split_work)
    torch.cuda.synchronize()
    split.assert_all_states(1)
    if not torch.all(split_output == 17):
        raise RuntimeError("deferred split-K launch consumed incomplete scratch state")
    phases.call("nta_jit_progress_host", split, 1)
    phases.call("nta_jit_publish_ready", split, split_work)
    run_hooked(
        hooked, q, split_staging_kv, split, split_output, skip_merge=True
    )
    phases.call("nta_jit_complete_launched", split, split_work)
    torch.cuda.synchronize()
    split.assert_all_states(3)
    split_epoch = FlashInferLayerEpoch(
        split.native_runtime,
        split.plan,
        phases.program,
        object_count=1,
        max_progress_passes=1,
    )
    split_result = split_epoch.run_host(
        hooked,
        q,
        split_staging_kv,
        split_output,
        progress_blocks=1,
        stream=torch.cuda.current_stream(),
    )
    if split_result.progress_passes != 1:
        raise RuntimeError(f"unexpected host progress rounds: {split_result}")
    split.assert_all_states(3)
    torch.testing.assert_close(split_staging_kv, split_reference_kv, rtol=0, atol=0)
    torch.testing.assert_close(split_output, split_expected, rtol=2e-3, atol=2e-3)
    split_maximum = (split_output.float() - split_expected.float()).abs().max().item()

    mixed_pages = 128
    mixed_shape = (2 * mixed_pages, 2, 16, 2, 128)
    mixed_host_kv = torch.randn(mixed_shape, dtype=torch.float16, pin_memory=True)
    mixed_reference_kv = mixed_host_kv.to("cuda")
    mixed_q = torch.randn((2, 4, 128), dtype=torch.float16, device="cuda")
    plan_uniform_batch(stock, 2, mixed_pages)
    mixed_expected = stock.run(mixed_q, mixed_reference_kv)
    plan_uniform_batch(hooked, 2, mixed_pages)
    mixed_schedule = decode_schedule(hooked)
    mixed_request_indices = list(mixed_schedule.request_indices)
    if set(mixed_request_indices) != {0, 1} or any(
        mixed_request_indices.count(request_index) <= 1 for request_index in (0, 1)
    ):
        raise RuntimeError(
            f"expected two independently split requests, got {mixed_schedule}"
        )
    mixed_staging_kv = torch.zeros_like(mixed_reference_kv)
    mixed_staging_kv[:mixed_pages].copy_(mixed_reference_kv[:mixed_pages])
    resident_work = {
        index
        for index, request_index in enumerate(mixed_request_indices)
        if request_index == 0
    }
    mixed = RuntimeFixture(
        mixed_staging_kv,
        mixed_host_kv,
        mixed_schedule.work_count,
        request_indices=mixed_request_indices,
        direct_work_indices=resident_work,
    )
    phases.call("nta_jit_reset_epoch", mixed, 1, mixed_schedule.work_count)
    mixed_output = torch.full_like(mixed_expected, 17)
    run_hooked(hooked, mixed_q, mixed_staging_kv, mixed, mixed_output)
    torch.cuda.synchronize()
    mixed_states = [
        mixed.work_ticket_state(index) for index in range(mixed_schedule.work_count)
    ]
    for index, request_index in enumerate(mixed_request_indices):
        expected_state = 3 if request_index == 0 else 1
        if mixed_states[index] != expected_state:
            raise RuntimeError(
                "request-local merge setup produced an unexpected ticket state: "
                f"{mixed_states}"
            )
    torch.testing.assert_close(
        mixed_output[0], mixed_expected[0], rtol=2e-3, atol=2e-3
    )
    if not torch.all(mixed_output[1] == 17):
        raise RuntimeError("incomplete request consumed split-K scratch state")

    prefill_query_tokens = 256
    prefill_q = torch.randn(
        (prefill_query_tokens, 4, 128), dtype=torch.float16, device="cuda"
    )
    stock_prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        "NHD",
        backend="fa2",
    )
    plan_prefill(stock_prefill, prefill_query_tokens, 4)
    prefill_expected = stock_prefill.run(prefill_q, reference_kv)
    hooked_prefill = make_prefill_wrapper()
    plan_prefill(hooked_prefill, prefill_query_tokens, 4)
    prefill_schedule = paged_prefill_schedule(hooked_prefill)
    prefill_work = prefill_schedule.work_count
    prefill_phases = PhaseFunctions("nta_batch_prefill_default_v2_hooked")

    prefill_resident = RuntimeFixture(
        reference_kv, None, prefill_work, list(prefill_schedule.request_indices)
    )
    prefill_phases.call("nta_jit_reset_epoch", prefill_resident, 1, prefill_work)
    prefill_resident_output = torch.empty_like(prefill_expected)
    run_prefill_hooked(
        hooked_prefill,
        prefill_q,
        reference_kv,
        prefill_resident,
        prefill_resident_output,
    )
    prefill_phases.call(
        "nta_jit_complete_launched", prefill_resident, prefill_work
    )
    torch.cuda.synchronize()
    prefill_resident.assert_all_states(3)
    torch.testing.assert_close(
        prefill_resident_output, prefill_expected, rtol=2e-3, atol=2e-3
    )

    prefill_staging_kv = torch.zeros_like(reference_kv)
    prefill_deferred = RuntimeFixture(
        prefill_staging_kv,
        host_kv,
        prefill_work,
        list(prefill_schedule.request_indices),
    )
    prefill_phases.call("nta_jit_reset_epoch", prefill_deferred, 1, prefill_work)
    prefill_output = torch.full_like(prefill_expected, math.nan)
    run_prefill_hooked(
        hooked_prefill,
        prefill_q,
        prefill_staging_kv,
        prefill_deferred,
        prefill_output,
        skip_merge=True,
    )
    prefill_phases.call(
        "nta_jit_complete_launched", prefill_deferred, prefill_work
    )
    torch.cuda.synchronize()
    prefill_deferred.assert_all_states(1)
    prefill_phases.call("nta_jit_progress_host", prefill_deferred, 1)
    prefill_phases.call("nta_jit_publish_ready", prefill_deferred, prefill_work)
    torch.cuda.synchronize()
    prefill_deferred.assert_all_states(2)
    run_prefill_hooked(
        hooked_prefill,
        prefill_q,
        prefill_staging_kv,
        prefill_deferred,
        prefill_output,
        skip_merge=True,
    )
    prefill_phases.call(
        "nta_jit_complete_launched", prefill_deferred, prefill_work
    )
    torch.cuda.synchronize()
    prefill_deferred.assert_all_states(3)
    run_prefill_hooked(
        hooked_prefill,
        prefill_q,
        prefill_staging_kv,
        prefill_deferred,
        prefill_output,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(prefill_output, prefill_expected, rtol=2e-3, atol=2e-3)
    prefill_maximum = (
        (prefill_output.float() - prefill_expected.float()).abs().max().item()
    )

    tiny_prefill_q = torch.randn((1, 4, 128), dtype=torch.float16, device="cuda")
    plan_prefill(stock_prefill, 1, 4)
    plan_prefill(hooked_prefill, 1, 4)
    tiny_schedule = paged_prefill_schedule(hooked_prefill)
    tiny_runtime = RuntimeFixture(
        reference_kv,
        None,
        tiny_schedule.work_count,
        list(tiny_schedule.request_indices),
    )
    tiny_baseline_output = torch.empty_like(tiny_prefill_q)
    tiny_hooked_output = torch.empty_like(tiny_prefill_q)

    def tiny_prefill_baseline_call() -> None:
        stock_prefill.run(
            tiny_prefill_q,
            reference_kv,
            1.0 / math.sqrt(128),
            out=tiny_baseline_output,
        )

    def tiny_prefill_hooked_call() -> None:
        runtime_tensor = tiny_runtime.runtime
        hooked_prefill.run(
            tiny_prefill_q,
            reference_kv,
            runtime_tensor,
            runtime_tensor,
            runtime_tensor,
            1.0 / math.sqrt(128),
            1,
            14,
            out=tiny_hooked_output,
        )

    tiny_baseline_samples = []
    tiny_hooked_samples = []
    for sample in range(5):
        if sample % 2 == 0:
            tiny_baseline_samples.append(benchmark(tiny_prefill_baseline_call))
            tiny_hooked_samples.append(benchmark(tiny_prefill_hooked_call))
        else:
            tiny_hooked_samples.append(benchmark(tiny_prefill_hooked_call))
            tiny_baseline_samples.append(benchmark(tiny_prefill_baseline_call))
    tiny_baseline_us = statistics.median(tiny_baseline_samples)
    tiny_hooked_us = statistics.median(tiny_hooked_samples)
    tiny_overhead = (tiny_hooked_us / tiny_baseline_us - 1.0) * 100.0
    torch.testing.assert_close(
        tiny_hooked_output, tiny_baseline_output, rtol=2e-3, atol=2e-3
    )
    tiny_runtime.native_runtime.cancel_request(0, GENERATION)
    cancelled_output = torch.full_like(tiny_prefill_q, 17)
    runtime_tensor = tiny_runtime.runtime
    hooked_prefill.run(
        tiny_prefill_q,
        reference_kv,
        runtime_tensor,
        runtime_tensor,
        runtime_tensor,
        1.0 / math.sqrt(128),
        1,
        14,
        out=cancelled_output,
    )
    torch.cuda.synchronize()
    if not torch.all(cancelled_output == 17):
        raise RuntimeError("cancelled planless request wrote attention output")

    benchmark_batch = 64
    benchmark_pages = 4
    benchmark_kv = torch.randn(
        (benchmark_batch * benchmark_pages, 2, 16, 2, 128),
        dtype=torch.float16,
        device="cuda",
    )
    benchmark_q = torch.randn(
        (benchmark_batch, 4, 128), dtype=torch.float16, device="cuda"
    )
    baseline = make_baseline_wrapper()
    plan_uniform_batch(baseline, benchmark_batch, benchmark_pages)
    plan_uniform_batch(hooked, benchmark_batch, benchmark_pages)
    benchmark_schedule = decode_schedule(hooked)
    if benchmark_schedule.request_indices != tuple(range(benchmark_batch)):
        raise RuntimeError(f"unexpected resident benchmark schedule {benchmark_schedule}")
    benchmark_runtime = RuntimeFixture(
        benchmark_kv,
        None,
        benchmark_batch,
        list(range(benchmark_batch)),
    )
    phases.call("nta_jit_reset_epoch", benchmark_runtime, 1, benchmark_batch)
    baseline_output = torch.empty_like(benchmark_q)
    hooked_output = torch.empty_like(benchmark_q)

    def baseline_call() -> None:
        baseline.run(
            benchmark_q,
            benchmark_kv,
            1.0 / math.sqrt(128),
            out=baseline_output,
        )

    def hooked_call() -> None:
        run_hooked(
            hooked,
            benchmark_q,
            benchmark_kv,
            benchmark_runtime,
            hooked_output,
        )

    baseline_samples = []
    hooked_samples = []
    for sample in range(5):
        if sample % 2 == 0:
            baseline_samples.append(benchmark(baseline_call))
            hooked_samples.append(benchmark(hooked_call))
        else:
            hooked_samples.append(benchmark(hooked_call))
            baseline_samples.append(benchmark(baseline_call))
    baseline_us = statistics.median(baseline_samples)
    hooked_us = statistics.median(hooked_samples)
    torch.testing.assert_close(hooked_output, baseline_output, rtol=2e-3, atol=2e-3)
    overhead = (hooked_us / baseline_us - 1.0) * 100.0
    if overhead > 8.0:
        raise RuntimeError(f"resident hook overhead {overhead:.2f}% exceeds 8%")

    print(
        f"flashinfer_version={flashinfer.__version__} resident=pass "
        f"host_staged=pass indexed_host=pass shared_kv_head_ctas=2 "
        f"ready_wave=pass "
        f"merge_gate=pass "
        f"request_local_merge=pass "
        f"max_abs_error={maximum:.6g} "
        f"split_work={split_work} split_max_abs_error={split_maximum:.6g} "
        f"prefill_work={prefill_work} prefill_max_abs_error={prefill_maximum:.6g} "
        f"tiny_prefill_baseline_us={tiny_baseline_us:.3f} "
        f"tiny_prefill_hooked_us={tiny_hooked_us:.3f} "
        f"tiny_prefill_overhead_pct={tiny_overhead:.2f} "
        f"planless_cancel=pass "
        f"baseline_us={baseline_us:.3f} hooked_us={hooked_us:.3f} "
        f"resident_overhead_pct={overhead:.2f}"
    )


if __name__ == "__main__":
    main()
