#!/usr/bin/env python3
"""Feasibility probe for the CXL devdax tier.

Answers three questions with measurements, not assumptions: (1) what CPU
bandwidth does the CXL window sustain relative to local DDR5; (2) does CUDA
accept a devdax mapping for host registration at all (ZONE_DEVICE pages have
historically been rejected); (3) if the GPU can reach it — registered or via
bounce — what H2D bandwidth does the copy engine sustain from CXL versus
pinned DDR5. The output decides whether CXL enters the design as a direct
replica class now or stays a documented-deferred tier; it makes no serving
claim either way.

Requires root for /dev/dax access:

    sudo env LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
      $(command -v python3) benchmarks/serving/CxlDaxProbe.py \
      --device /dev/dax0.0 --output results/serving/cxl-dax-probe.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import pathlib
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=pathlib.Path, default="/dev/dax0.0")
    parser.add_argument("--window-mib", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if not args.device.exists():
        parser.error(f"dax device does not exist: {args.device}")
    if args.window_mib < 64:
        parser.error("window must be at least 64 MiB")
    return args


def git_value(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def bandwidth_gib(bytes_moved: int, seconds: float) -> float:
    return bytes_moved / (1 << 30) / seconds


def cpu_copy_bandwidth(view: memoryview, scratch: bytearray,
                       iterations: int) -> dict[str, float]:
    import numpy

    source = numpy.frombuffer(view, dtype=numpy.uint8)
    local = numpy.frombuffer(scratch, dtype=numpy.uint8)
    # First touch so page population is not billed to the timed loop.
    local[:] = source
    reads = []
    writes = []
    for _ in range(iterations):
        begin = time.perf_counter()
        local[:] = source
        reads.append(bandwidth_gib(len(source), time.perf_counter() - begin))
        begin = time.perf_counter()
        source_writable = numpy.frombuffer(view, dtype=numpy.uint8)
        source_writable[:] = local
        writes.append(bandwidth_gib(len(local), time.perf_counter() - begin))
    return {
        "read_gib_per_s_best": max(reads),
        "write_gib_per_s_best": max(writes),
    }


def main() -> int:
    args = parse_args()
    window = args.window_mib << 20
    report: dict[str, object] = {
        "schema": 1,
        "classification": "cxl-devdax-feasibility-probe",
        "revision": os.environ.get("NTA_REVISION", git_value("rev-parse", "HEAD")),
        "dirty": bool(git_value("status", "--porcelain")),
        "device": str(args.device),
        "window_bytes": window,
    }

    fd = os.open(args.device, os.O_RDWR)
    try:
        mapping = mmap.mmap(fd, window, mmap.MAP_SHARED,
                            mmap.PROT_READ | mmap.PROT_WRITE)
    finally:
        os.close(fd)
    try:
        view = memoryview(mapping)
        scratch = bytearray(window)
        report["cpu"] = cpu_copy_bandwidth(view, scratch, args.iterations)

        cudart = ctypes.CDLL("libcudart.so")
        cudart.cudaGetErrorString.restype = ctypes.c_char_p

        def cuda_error(status: int) -> str:
            return cudart.cudaGetErrorString(status).decode()

        base = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        registration: dict[str, object] = {}
        registered_flag = None
        for name, flags in (("default", 0), ("mapped", 2), ("iomemory", 4)):
            status = cudart.cudaHostRegister(
                ctypes.c_void_p(base), ctypes.c_size_t(window),
                ctypes.c_uint(flags),
            )
            registration[name] = (
                "ok" if status == 0 else f"error {status}: {cuda_error(status)}"
            )
            if status == 0:
                registered_flag = name
                break
        report["cuda_host_register"] = registration
        report["cuda_registered"] = registered_flag

        device_buffer = ctypes.c_void_p()
        status = cudart.cudaMalloc(ctypes.byref(device_buffer),
                                   ctypes.c_size_t(window))
        if status != 0:
            raise RuntimeError(f"cudaMalloc failed: {cuda_error(status)}")

        def timed_h2d(source_pointer: int, label: str) -> float:
            samples = []
            for _ in range(args.iterations):
                begin = time.perf_counter()
                copy_status = cudart.cudaMemcpy(
                    device_buffer, ctypes.c_void_p(source_pointer),
                    ctypes.c_size_t(window), ctypes.c_int(1),
                )
                cudart.cudaDeviceSynchronize()
                if copy_status != 0:
                    raise RuntimeError(
                        f"{label} cudaMemcpy failed: {cuda_error(copy_status)}"
                    )
                samples.append(
                    bandwidth_gib(window, time.perf_counter() - begin)
                )
            return max(samples)

        report["h2d_from_cxl_gib_per_s"] = timed_h2d(base, "cxl")

        pinned = ctypes.c_void_p()
        status = cudart.cudaHostAlloc(ctypes.byref(pinned),
                                      ctypes.c_size_t(window),
                                      ctypes.c_uint(0))
        if status != 0:
            raise RuntimeError(f"cudaHostAlloc failed: {cuda_error(status)}")
        ctypes.memset(pinned, 0x5A, window)
        report["h2d_from_pinned_ddr5_gib_per_s"] = timed_h2d(
            pinned.value, "pinned"
        )

        if registered_flag is not None:
            cudart.cudaHostUnregister(ctypes.c_void_p(base))
        cudart.cudaFreeHost(pinned)
        cudart.cudaFree(device_buffer)
        del view
    finally:
        mapping.close()

    encoded = json.dumps(report, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        os.chmod(args.output, 0o644)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
