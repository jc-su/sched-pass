#!/usr/bin/env python3
"""Isolated reproducer for the per-step bounded indexed copy seam.

Registers an engine-shaped K/V indexed object pair over fully controlled
buffers, rewrites the shared index arrays' prefix, bounds the copy with
nta_jit_set_indexed_row_counts, runs the validated indexed progress, and
asserts the staged rows equal their host sources byte for byte while
untouched rows stay untouched. Discriminates a seam defect from an
engine-tensor-semantics defect in the tiered staging divergence.

Requires a compiled phase module; pass it explicitly or via
NTA_PHASE_MODULE.
"""

from __future__ import annotations

import os
import pathlib

import torch

from nta_runtime.runtime import (
    IndexedHostObject,
    JitPhaseProgram,
    Runtime,
    RuntimeConfig,
)

ROWS = 64
HEADS = 2
DIM = 128


def locate_module() -> pathlib.Path:
    configured = os.environ.get("NTA_PHASE_MODULE")
    if configured:
        return pathlib.Path(configured)
    cache = pathlib.Path.home() / ".cache/flashinfer"
    candidates = sorted(
        cache.rglob("nta_sglang_decode_demand*.so"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(
            "no compiled phase module found; set NTA_PHASE_MODULE"
        )
    return candidates[0]


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable; seam reproducer skipped")
        return 0
    # Phase modules link against TVM-FFI, whose symbols the serving process
    # provides by importing flashinfer before loading any module.
    import flashinfer  # noqa: F401

    module = locate_module()
    phases = JitPhaseProgram(module)
    runtime = Runtime(
        RuntimeConfig(
            request_capacity=4,
            object_capacity=8,
            intent_capacity=8,
            work_ticket_capacity=8,
            max_dependencies_per_work_ticket=1,
        )
    )

    def patterned(offset: float) -> torch.Tensor:
        base = torch.arange(
            ROWS * HEADS * DIM, dtype=torch.float32
        ).view(ROWS, HEADS, DIM)
        return (base * 1e-3 + offset).to(torch.float16)

    host_k = patterned(1.0).pin_memory()
    host_v = patterned(2.0).pin_memory()
    device_k = torch.zeros(
        (ROWS, HEADS, DIM), dtype=torch.float16, device="cuda"
    )
    device_v = torch.zeros_like(device_k)

    capacity = 16
    source_index = torch.zeros(capacity, dtype=torch.int32, device="cuda")
    staging_index = torch.zeros(capacity, dtype=torch.int32, device="cuda")

    element = HEADS * DIM * 2
    objects = []
    for source, staging in ((host_k, device_k), (host_v, device_v)):
        objects.append(
            IndexedHostObject(
                0x5345414D_00000000 + len(objects),
                1,
                source.data_ptr(),
                staging.data_ptr(),
                source_index.data_ptr(),
                staging_index.data_ptr(),
                capacity,
                element,
                source.stride(0) * source.element_size(),
                staging.stride(0) * staging.element_size(),
                ROWS,
                ROWS,
            )
        )
    stream = torch.cuda.current_stream()
    runtime.register_indexed_host_objects(0, objects, stream=stream)

    # Step one: a scattered subset, deliberately unordered.
    source_rows = [3, 41, 7, 22, 58]
    staging_rows = [10, 4, 33, 60, 21]
    count = len(source_rows)
    source_index[:count] = torch.tensor(
        source_rows, dtype=torch.int32, device="cuda"
    )
    staging_index[:count] = torch.tensor(
        staging_rows, dtype=torch.int32, device="cuda"
    )
    phases.set_indexed_row_counts(runtime, 0, 2, count, stream=stream)
    phases.progress_validated_indexed_host_range(runtime, 0, 2, stream=stream)
    torch.cuda.synchronize()

    failures = 0
    for name, host, device_buffer in (
        ("K", host_k, device_k),
        ("V", host_v, device_v),
    ):
        staged = device_buffer[
            torch.tensor(staging_rows, dtype=torch.long, device="cuda")
        ].cpu()
        expected = host[torch.tensor(source_rows, dtype=torch.long)]
        if not torch.equal(staged, expected):
            mismatch = (
                staged.view(count, -1) != expected.view(count, -1)
            ).any(dim=1)
            first = int(mismatch.nonzero()[0])
            print(
                f"{name}: DIVERGED at pair {first} "
                f"(src {source_rows[first]} -> dst {staging_rows[first]}): "
                f"staged head {staged.view(count, -1)[first, :4].tolist()} "
                f"expected {expected.view(count, -1)[first, :4].tolist()}"
            )
            failures += 1
        untouched = [r for r in range(ROWS) if r not in staging_rows]
        touched_extra = device_buffer[
            torch.tensor(untouched, dtype=torch.long, device="cuda")
        ]
        if not bool((touched_extra == 0).all()):
            print(f"{name}: untouched destination rows were modified")
            failures += 1

    # Step two: shrink the count and rewrite the prefix; only the new prefix
    # may copy.
    device_k.zero_()
    device_v.zero_()
    source_index[:2] = torch.tensor([11, 12], dtype=torch.int32, device="cuda")
    staging_index[:2] = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    phases.set_indexed_row_counts(runtime, 0, 2, 2, stream=stream)
    phases.progress_validated_indexed_host_range(runtime, 0, 2, stream=stream)
    torch.cuda.synchronize()
    if not torch.equal(device_k[1:3].cpu(), host_k[11:13]):
        print("K: shrunken-count prefix copy diverged")
        failures += 1
    if not bool((device_k[3:] == 0).all()) or not bool(
        (device_k[0] == 0).all()
    ):
        print("K: shrunken count copied beyond its prefix")
        failures += 1

    if failures:
        print(f"seam reproducer FAILED with {failures} defect(s); module {module}")
        return 1
    print(f"indexed row-count seam holds; module {module.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
