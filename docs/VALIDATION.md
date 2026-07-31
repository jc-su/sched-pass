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

NVMe M4 hardware:

- controller: KIOXIA CD8P E3.S, PCIe 5.0 x4, 1.92 TB;
- queue: one depth-64 I/O SQ/CQ, 4 KiB controller pages, 512-byte LBAs;
- command: NVM READ only in the benchmark;
- destination: CUDA HBM through DMA-BUF or pinned mapped CPU DRAM; and
- launch: one finite CUDA graph, with no persistent kernel and no CPU command
  or completion path.

## Compiler evidence

The build emits:

```text
build/kernel/KvAcquire.raw.bc
build/kernel/KvAcquire.lowered.ll
build/kernel/KvAcquire.ptx
build/kernel/KvAcquire.cubin
```

The lowered module contains zero bind/acquire/defer marker symbols and two
compiler-generated `nta_acquire_slow` sites: KV dot product and NVMe checksum.
The PTX direct KV compute entry has no atomic instruction. Its only calls are on
the null/slow and defer edges. NVMe progress uses `ld.global.cv`, `membar.sys`,
and no CTA barrier.

`ptxas` resource usage:

| Kernel | Registers | Shared bytes | Local bytes |
| --- | ---: | ---: | ---: |
| `nta_kv_tile_kernel` | 32 | 128 | 0 |
| `nta_progress_host_staging` | 26 | 0 | 0 |
| `nta_reset_epoch` | 10 | 0 | 0 |
| `nta_progress_nvme` | 40 | 0 | 0 |
| `nta_nvme_hash_kernel` | 32 | 256 | 0 |

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

## GPU-initiated NVMe

Before controller rebinding, the validation helper saved a 2 MiB reference from
namespace offset zero. Each run used 16 independent 64 KiB ranges, one warmup,
10 measured graph launches, and 128 bounded progress/resume pairs per graph.
All 176 expected commands completed and every GPU checksum matched the block
device reference.

| Destination | Graph time (ms) | Logical MiB/s | Failed CQEs | Verification failures |
| --- | ---: | ---: | ---: | ---: |
| mapped CPU DRAM | 0.743 | 1,345.90 | 0 | 0 |
| DMA-BUF HBM | 0.679 | 1,473.54 | 0 | 0 |

These are finite-graph mechanism measurements, not raw SSD bandwidth. The graph
also runs checksum CTAs and many completion checks after the transfer is ready.
One 2 MiB mapped-DRAM read also completed correctly at 2.488 ms per graph over
three measured iterations.

CUDA reported DMA-BUF support and exported 2 MiB HBM allocations. The NVMe
importer observed one contiguous 2 MiB DMA segment. Explicitly requesting
CUDA's `DMA_BUF_MAPPING_TYPE_PCIE` export flag returned
`CUDA_ERROR_NOT_SUPPORTED` on this driver/platform, while the default DMA-BUF
export was successfully attached and verified by actual NVMe reads into HBM.

The test process required root because unprivileged CUDA I/O-memory registration
of the NVMe doorbell failed with `CUDA_ERROR_NOT_PERMITTED`. After validation,
a background `vmem_sw` consumer was detected on the target controller. The
controller was restored to its original driver, and the setup helper now refuses
to bind while that module is loaded. The project issued no NVMe writes.

## Claims not yet validated

- production paged-attention integration and SLO impact;
- direct-path overhead against an untouched production kernel;
- GPU submission and completion over RDMA;
- TMA descriptor rebinding; and
- priority/deadline-aware admission.

These remain gated milestones. No source directory contains a placeholder
backend for them.
