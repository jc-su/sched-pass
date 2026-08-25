#!/usr/bin/env python3
"""Measure the ticketed GPU-initiated host mover without attention overlap."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

import flashinfer  # noqa: F401 - loads the TVM FFI symbols used by the JIT module
import torch

from nta_runtime.runtime import (
    AcquireRequirement,
    DeviceWorkPlan,
    IndexedHostObject,
    JitPhaseProgram,
    RequestRange,
    Runtime,
    RuntimeConfig,
    WorkItem,
)


OBJECT_ID = 0x4E5441494F000000
GENERATION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=pathlib.Path, required=True)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--element-bytes", type=int, default=512)
    parser.add_argument("--copy-blocks", default="1,2,4,8,16,32,64")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    args.copy_blocks = tuple(int(value) for value in args.copy_blocks.split(","))
    if (
        min(args.rows, args.element_bytes, args.warmup, args.iterations) <= 0
        or not args.copy_blocks
        or min(args.copy_blocks) <= 0
        or max(args.copy_blocks) > 64
    ):
        parser.error("benchmark geometry must be positive and use at most 64 blocks")
    return args


def main() -> int:
    args = parse_args()
    stream = torch.cuda.current_stream()
    host_key = torch.randint(
        0, 256, (args.rows, args.element_bytes), dtype=torch.uint8, pin_memory=True
    )
    host_value = torch.randint(
        0, 256, (args.rows, args.element_bytes), dtype=torch.uint8, pin_memory=True
    )
    staging_key = torch.zeros_like(host_key, device="cuda")
    staging_value = torch.zeros_like(host_value, device="cuda")
    indices = torch.arange(args.rows, dtype=torch.int32, device="cuda")
    transfer_bytes = 2 * args.rows * args.element_bytes

    runtime = Runtime(
        RuntimeConfig(
            request_capacity=1,
            object_capacity=2,
            intent_capacity=2,
            work_ticket_capacity=1,
            max_dependencies_per_work_ticket=2,
        )
    )
    phases = JitPhaseProgram(args.module)
    plan = DeviceWorkPlan(1, 2, runtime.device_ordinal)
    try:
        runtime.set_tenant_budget(0, 2 * transfer_bytes)
        runtime.set_request(
            0,
            1,
            GENERATION,
            max_outstanding_bytes=2 * transfer_bytes,
        )
        runtime.register_indexed_host_objects(
            0,
            (
                IndexedHostObject(
                    OBJECT_ID,
                    GENERATION,
                    host_key.data_ptr(),
                    staging_key.data_ptr(),
                    indices.data_ptr(),
                    indices.data_ptr(),
                    args.rows,
                    args.element_bytes,
                    args.element_bytes,
                    args.element_bytes,
                    args.rows,
                    args.rows,
                ),
                IndexedHostObject(
                    OBJECT_ID + 1,
                    GENERATION,
                    host_value.data_ptr(),
                    staging_value.data_ptr(),
                    indices.data_ptr(),
                    indices.data_ptr(),
                    args.rows,
                    args.element_bytes,
                    args.element_bytes,
                    args.element_bytes,
                    args.rows,
                    args.rows,
                ),
            ),
            stream,
        )
        plan.upload(
            (
                WorkItem(
                    0,
                    0,
                    GENERATION,
                    0,
                    0,
                    2,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                ),
            ),
            (
                AcquireRequirement(
                    0, 0, OBJECT_ID, 0, 0, GENERATION, transfer_bytes // 2, 0
                ),
                AcquireRequirement(
                    0, 0, OBJECT_ID + 1, 0, 1, GENERATION, transfer_bytes // 2, 0
                ),
            ),
            (RequestRange(0, 1, 0, GENERATION),),
            stream,
        )
        plan.wait_on(stream)
        phases.validate_indexed_host_range(runtime, 0, 2, stream)
        stream.synchronize()

        measurements = []
        for blocks in args.copy_blocks:
            samples = []
            for iteration in range(args.warmup + args.iterations):
                phases.reset(runtime, 2, 1, stream)
                phases.invalidate_cached_objects(runtime, 0, 2, stream)
                phases.discover(runtime, plan, stream)
                start = torch.cuda.Event(enable_timing=True)
                finish = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                phases.progress_validated_indexed_host_range_parallel(
                    runtime, 0, 2, blocks, stream
                )
                finish.record(stream)
                finish.synchronize()
                if iteration >= args.warmup:
                    samples.append(start.elapsed_time(finish) * 1000.0)
            median_us = statistics.median(samples)
            measurements.append(
                {
                    "copy_blocks_per_group": blocks,
                    "median_us": median_us,
                    "minimum_us": min(samples),
                    "maximum_us": max(samples),
                    "gib_per_second": transfer_bytes / median_us / (1024**3) * 1e6,
                }
            )

        torch.testing.assert_close(staging_key.cpu(), host_key, rtol=0, atol=0)
        torch.testing.assert_close(staging_value.cpu(), host_value, rtol=0, atol=0)
        best = min(measurements, key=lambda item: item["median_us"])
        report = {
            "schema": 1,
            "classification": "indexed-host-progress",
            "module": str(args.module.resolve()),
            "rows": args.rows,
            "element_bytes": args.element_bytes,
            "transfer_bytes": transfer_bytes,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "measurements": measurements,
            "best_copy_blocks_per_group": best["copy_blocks_per_group"],
            "best_median_us": best["median_us"],
            "output_parity": True,
        }
        print(json.dumps(report, sort_keys=True))
        return 0
    finally:
        plan.close()
        phases.close()
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
