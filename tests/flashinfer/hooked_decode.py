#!/usr/bin/env python3
"""Execute resident and pinned-host deferred paths in hooked FlashInfer decode."""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import pathlib
import statistics
import struct

import flashinfer
import torch
from flashinfer.jit.attention.variants import attention_sink_fa2_decl
from tools.flashinfer.schedule import decode_schedule, paged_prefill_schedule


ABI_VERSION = int(os.environ["NTA_ABI_VERSION"])
OBJECT_ID = 0xC001
GENERATION = 1
TENSOR_NAMES = ["sink", "nta_runtime", "nta_work_items", "nta_dependencies"]
TENSOR_DTYPES = ["float", "uint8_t", "uint8_t", "uint8_t"]
SCALAR_NAMES = ["sm_scale", "nta_work_count"]
SCALAR_DTYPES = ["double", "int64_t"]


def device_blob(size: int, packed: bytes = b"") -> torch.Tensor:
    data = bytearray(size)
    data[: len(packed)] = packed
    return torch.tensor(data, dtype=torch.uint8, device="cuda")


class RuntimeFixture:
    def __init__(
        self,
        kv: torch.Tensor,
        host_source: torch.Tensor | None,
        work_count: int = 1,
        request_indices: list[int] | None = None,
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
        source_kind = 2 if host_source is not None else 0
        object_state = 0 if host_source is not None else 3
        source_address = (
            host_source.data_ptr() if host_source is not None else kv.data_ptr()
        )

        requests = b"".join(
            struct.pack(
                "<4Q4I",
                17 + index,
                0,
                2 * byte_count,
                0,
                GENERATION,
                0,
                4,
                0,
            )
            + bytes(16)
            for index in range(request_count)
        )
        self.request = device_blob(64 * request_count, requests)
        self.tenant = device_blob(
            32, struct.pack("<2Q2IQ", 2 * byte_count, 0, 1, 1, 0)
        )
        self.object = device_blob(
            64,
            struct.pack(
                "<4Q6IQ",
                OBJECT_ID,
                kv.data_ptr(),
                byte_count,
                0,
                1,
                object_state,
                0,
                1,
                0,
                0,
                0,
            ),
        )
        self.replica = device_blob(
            64,
            struct.pack(
                "<4Q4I2Q",
                source_address,
                0,
                1_000,
                20_000_000_000,
                source_kind,
                0,
                source_kind,
                2 if host_source is not None else 1,
                0,
                0,
            ),
        )
        backend_bytes = bytearray(5 * 64)
        for index in range(5):
            struct.pack_into(
                "<5Q4IQ",
                backend_bytes,
                index * 64,
                0,
                1_000,
                20_000_000_000,
                0,
                2 * byte_count,
                index,
                1 if index == source_kind else 0,
                index,
                0,
                0,
            )
        self.backends = device_blob(len(backend_bytes), backend_bytes)
        self.intents = device_blob(128)
        self.continuations = device_blob(32 * work_count)
        self.continuation_dependencies = device_blob(16 * work_count)
        self.intent_pool = device_blob(
            64, struct.pack("<2Q4I4Q", 0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
        )
        self.ready = torch.zeros(work_count, dtype=torch.uint32, device="cuda")
        self.ready_count = torch.zeros(1, dtype=torch.uint32, device="cuda")
        self.ready_head = torch.zeros(1, dtype=torch.uint32, device="cuda")
        self.pending = torch.zeros(work_count, dtype=torch.uint32, device="cuda")
        self.pending_count = torch.zeros(1, dtype=torch.uint32, device="cuda")

        runtime = struct.pack(
            "<14Q10I",
            self.request.data_ptr(),
            self.tenant.data_ptr(),
            self.object.data_ptr(),
            self.replica.data_ptr(),
            self.backends.data_ptr(),
            self.intents.data_ptr(),
            self.continuations.data_ptr(),
            self.continuation_dependencies.data_ptr(),
            self.intent_pool.data_ptr(),
            self.ready.data_ptr(),
            self.ready_count.data_ptr(),
            self.ready_head.data_ptr(),
            self.pending.data_ptr(),
            self.pending_count.data_ptr(),
            request_count,
            1,
            1,
            1,
            5,
            1,
            work_count,
            work_count,
            1,
            ABI_VERSION,
        )
        self.runtime = device_blob(192, runtime)
        direct = kv.data_ptr() if host_source is None else 0
        requirement = struct.pack(
            "<4Q4I", direct, 0, OBJECT_ID, 0, 0, 1, byte_count, 0
        )
        self.requirements = device_blob(48 * work_count, requirement * work_count)
        work_items = b"".join(
            struct.pack(
                "<8I",
                request_indices[index],
                request_indices[index],
                GENERATION,
                index,
                index,
                1,
                1 if direct else 0,
                index,
            )
            for index in range(work_count)
        )
        self.work_items = device_blob(32 * work_count, work_items)

    def continuation_state(self, index: int = 0) -> int:
        offset = index * 32 + 16
        state_bytes = self.continuations[offset : offset + 4].cpu().numpy().tobytes()
        return struct.unpack("<I", state_bytes)[0]

    def assert_all_states(self, expected: int) -> None:
        states = [self.continuation_state(index) for index in range(self.work_count)]
        if states != [expected] * self.work_count:
            raise RuntimeError(f"continuation states {states} do not equal {expected}")


class PhaseFunctions:
    def __init__(self, module_name: str = "nta_batch_decode_hooked") -> None:
        workspace = pathlib.Path(os.environ["FLASHINFER_WORKSPACE_BASE"])
        modules = list(workspace.rglob(f"{module_name}.so"))
        if len(modules) != 1:
            raise RuntimeError(f"expected one hooked decode module, found {modules}")
        self.library = ctypes.CDLL(str(modules[0]))
        self.library.nta_jit_abi_version.restype = ctypes.c_uint32
        if self.library.nta_jit_abi_version() != ABI_VERSION:
            raise RuntimeError("hooked module has an incompatible NTA ABI")
        for name, arguments in {
            "nta_jit_reset_epoch": [
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_void_p,
            ],
            "nta_jit_progress_host": [
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_void_p,
            ],
            "nta_jit_publish_ready": [
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_void_p,
            ],
            "nta_jit_complete_launched": [
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_void_p,
            ],
        }.items():
            function = getattr(self.library, name)
            function.argtypes = arguments
            function.restype = ctypes.c_int

    @staticmethod
    def stream() -> ctypes.c_void_p:
        return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    def call(self, name: str, fixture: RuntimeFixture, *arguments: int) -> None:
        status = getattr(self.library, name)(
            ctypes.c_void_p(fixture.runtime.data_ptr()),
            *arguments,
            self.stream(),
        )
        if status != 0:
            raise RuntimeError(f"{name} launch failed with CUDA error {status}")


def make_wrapper() -> flashinfer.BatchDecodeWithPagedKVCacheWrapper:
    jit_args = [
        "nta_batch_decode_hooked",
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
        "AttentionSink",
        attention_sink_fa2_decl,
    ]
    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    return flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", backend="fa2", jit_args=jit_args
    )


def make_baseline_wrapper() -> flashinfer.BatchDecodeWithPagedKVCacheWrapper:
    jit_args = [
        "nta_batch_decode_baseline",
        torch.float16,
        torch.float16,
        torch.float16,
        torch.int32,
        128,
        128,
        ["sink"],
        ["float"],
        ["sm_scale"],
        ["double"],
        "AttentionSink",
        attention_sink_fa2_decl,
    ]
    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    return flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", backend="fa2", jit_args=jit_args
    )


def make_prefill_wrapper() -> flashinfer.BatchPrefillWithPagedKVCacheWrapper:
    jit_args = [
        "nta_batch_prefill_hooked",
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
        "AttentionSink",
        attention_sink_fa2_decl,
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
    sink: torch.Tensor,
    fixture: RuntimeFixture,
    output: torch.Tensor,
) -> None:
    wrapper.run(
        q,
        kv,
        sink,
        fixture.runtime,
        fixture.work_items,
        fixture.requirements,
        1.0 / math.sqrt(128),
        fixture.work_count,
        out=output,
    )


def run_prefill_hooked(
    wrapper: flashinfer.BatchPrefillWithPagedKVCacheWrapper,
    q: torch.Tensor,
    kv: torch.Tensor,
    sink: torch.Tensor,
    fixture: RuntimeFixture,
    output: torch.Tensor,
) -> None:
    wrapper.run(
        q,
        kv,
        sink,
        fixture.runtime,
        fixture.work_items,
        fixture.requirements,
        1.0 / math.sqrt(128),
        fixture.work_count,
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
    sink = torch.full((4,), -math.inf, dtype=torch.float32, device="cuda")

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
    run_hooked(hooked, q, reference_kv, sink, resident, resident_output)
    phases.call("nta_jit_complete_launched", resident, 1)
    torch.cuda.synchronize()
    torch.testing.assert_close(resident_output, expected, rtol=2e-3, atol=2e-3)
    resident.assert_all_states(3)

    staging_kv = torch.zeros_like(reference_kv)
    deferred = RuntimeFixture(staging_kv, host_kv)
    phases.call("nta_jit_reset_epoch", deferred, 1, 1)
    deferred_output = torch.full_like(expected, math.nan)
    run_hooked(hooked, q, staging_kv, sink, deferred, deferred_output)
    phases.call("nta_jit_complete_launched", deferred, 1)
    torch.cuda.synchronize()
    deferred.assert_all_states(1)
    if not torch.isnan(deferred_output).all():
        raise RuntimeError("deferred FlashInfer CTA wrote output before readiness")

    phases.call("nta_jit_progress_host", deferred, 1)
    phases.call("nta_jit_publish_ready", deferred, 1)
    torch.cuda.synchronize()
    deferred.assert_all_states(2)
    run_hooked(hooked, q, staging_kv, sink, deferred, deferred_output)
    phases.call("nta_jit_complete_launched", deferred, 1)
    torch.cuda.synchronize()
    deferred.assert_all_states(3)
    torch.testing.assert_close(staging_kv, reference_kv, rtol=0, atol=0)
    torch.testing.assert_close(deferred_output, expected, rtol=2e-3, atol=2e-3)
    maximum = (deferred_output.float() - expected.float()).abs().max().item()
    if options.sanitizer:
        print(
            f"flashinfer_version={flashinfer.__version__} sanitizer_path=pass "
            f"shared_kv_head_ctas=2 max_abs_error={maximum:.6g}"
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
    split_output = torch.full_like(split_expected, math.nan)
    run_hooked(hooked, q, split_staging_kv, sink, split, split_output)
    phases.call("nta_jit_complete_launched", split, split_work)
    torch.cuda.synchronize()
    split.assert_all_states(1)
    phases.call("nta_jit_progress_host", split, 1)
    phases.call("nta_jit_publish_ready", split, split_work)
    torch.cuda.synchronize()
    split.assert_all_states(2)
    run_hooked(hooked, q, split_staging_kv, sink, split, split_output)
    phases.call("nta_jit_complete_launched", split, split_work)
    torch.cuda.synchronize()
    split.assert_all_states(3)
    torch.testing.assert_close(split_staging_kv, split_reference_kv, rtol=0, atol=0)
    torch.testing.assert_close(split_output, split_expected, rtol=2e-3, atol=2e-3)
    split_maximum = (split_output.float() - split_expected.float()).abs().max().item()

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
    prefill_phases = PhaseFunctions("nta_batch_prefill_hooked")

    prefill_resident = RuntimeFixture(
        reference_kv, None, prefill_work, list(prefill_schedule.request_indices)
    )
    prefill_phases.call("nta_jit_reset_epoch", prefill_resident, 1, prefill_work)
    prefill_resident_output = torch.empty_like(prefill_expected)
    run_prefill_hooked(
        hooked_prefill,
        prefill_q,
        reference_kv,
        sink,
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
        sink,
        prefill_deferred,
        prefill_output,
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
        sink,
        prefill_deferred,
        prefill_output,
    )
    prefill_phases.call(
        "nta_jit_complete_launched", prefill_deferred, prefill_work
    )
    torch.cuda.synchronize()
    prefill_deferred.assert_all_states(3)
    torch.testing.assert_close(prefill_output, prefill_expected, rtol=2e-3, atol=2e-3)
    prefill_maximum = (
        (prefill_output.float() - prefill_expected.float()).abs().max().item()
    )

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
    benchmark_sink = torch.full(
        (4,), -math.inf, dtype=torch.float32, device="cuda"
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
            benchmark_sink,
            1.0 / math.sqrt(128),
            out=baseline_output,
        )

    def hooked_call() -> None:
        run_hooked(
            hooked,
            benchmark_q,
            benchmark_kv,
            benchmark_sink,
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
        f"host_staged=pass shared_kv_head_ctas=2 max_abs_error={maximum:.6g} "
        f"split_work={split_work} split_max_abs_error={split_maximum:.6g} "
        f"prefill_work={prefill_work} prefill_max_abs_error={prefill_maximum:.6g} "
        f"baseline_us={baseline_us:.3f} hooked_us={hooked_us:.3f} "
        f"resident_overhead_pct={overhead:.2f}"
    )


if __name__ == "__main__":
    main()
