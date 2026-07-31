# Validation

Date: 2026-07-31

This file records reproducible implementation evidence. It is not an
end-to-end serving or SLO result.

## Environment

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability 12.0
- Driver: 595.84
- Device memory: 97,887 MiB
- Compiler: Ubuntu LLVM/Clang 22.1.8
- CUDA device toolkit: 12.9.86
- Target: `sm_120`

CUDA 13 is installed but is not used for device compilation because its
reorganized headers are incompatible with the current Clang CUDA wrapper.

## Compiler evidence

The build emits:

```text
build/kernel/KvAcquire.raw.bc
build/kernel/KvAcquire.lowered.ll
build/kernel/KvAcquire.ptx
build/kernel/KvAcquire.cubin
```

The lowered module contains zero bind/acquire/defer marker symbols and one
compiler-generated `nta_acquire_slow` site. The PTX direct compute entry has no
atomic instruction. Its only calls are on the null/slow and defer edges.

`ptxas` resource usage:

| Kernel | Registers | Shared bytes | Local bytes |
| --- | ---: | ---: | ---: |
| `nta_kv_tile_kernel` | 32 | 1,152 | 0 |
| `nta_progress_host_staging` | 26 | 0 | 0 |
| `nta_reset_epoch` | 10 | 0 | 0 |

## Correctness

`ctest --test-dir build --output-on-failure` passes:

1. valid and rejected LLVM IR transformations;
2. host/device ABI layout;
3. CUDA allocation, publication, and cancellation state; and
4. mixed HBM, mapped-host, and staged-host KV tiles.

The GPU test includes duplicate-object coalescing, cancellation, and stale
request generations. Compute Sanitizer reports:

```text
memcheck:  0 errors
racecheck: 0 hazards, 0 errors, 0 warnings
synccheck: 0 errors
```

## Mechanism throughput

Each row uses 256 requests, one 256 KiB tile per request, 50 captured-graph
iterations, and no cancellation. `logical_GiB/s` counts bytes consumed by KV
dot-product CTAs, so it is not a raw PCIe bandwidth measurement.

| Placement | Graph time (ms) | Logical GiB/s | Verification failures |
| --- | ---: | ---: | ---: |
| HBM resident | 0.094 | 662.20 | 0 |
| CPU DRAM direct | 1.350 | 46.28 | 0 |
| CPU DRAM staged by GPU | 1.426 | 43.82 | 0 |

A coalescing run with 96 requests sharing 24 staged 64 KiB objects produced
exactly 24 GPU transfer issues, despite six cancelled and six stale request
bindings:

```text
graph_ms=0.072 logical_GiB/s=81.65 staged_issues=24
verification_failures=0
```

## Claims not yet validated

- production paged-attention integration and SLO impact;
- direct-path overhead against an untouched production kernel;
- GPU submission to NVMe queues;
- GPU submission and completion over RDMA;
- TMA descriptor rebinding; and
- priority/deadline-aware admission.

These remain gated milestones. No source directory contains a placeholder
backend for them.
