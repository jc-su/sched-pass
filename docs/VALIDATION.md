# Validation Record

Date: 2026-07-31

This file records reproducible mechanism evidence. It is not an end-to-end
serving result and does not establish production readiness or an OSDI-level
claim. Open gates are listed explicitly at the end.

## Environment

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability 12.0
- Driver: 595.84
- Device memory: 97,887 MiB
- Compiler: Ubuntu LLVM/Clang 22.1.8
- Device toolkit: CUDA 12.9.86
- Target: `sm_120`
- Compute Sanitizer: 2025.3.1
- FlashInfer: 0.6.12
- PyTorch: 2.11.0+cu130

CUDA 13 is installed but is not used for device compilation because its header
layout is incompatible with the current Clang CUDA wrapper.

## Reproduction

Run the complete local mechanism suite:

```bash
NTA_SANITIZE=1 ./scripts/validate-local.sh
```

This builds the pass and runtime, runs CTest, evaluates all direct/staged and
global-load/TMA attention combinations, emits the PTXAS resource report, and
runs memcheck, racecheck, and synccheck. Generated evidence is written under
`results/`, which is excluded from source control.

## Compiler And Runtime Correctness

`ctest --test-dir build --output-on-failure` passes eight tests:

1. FlashInfer CSR-to-NTA adapter validation, including malformed metadata;
2. byte-address and tensor-map LLVM lowering plus rejected unsafe IR;
3. host/device ABI v7 layout;
4. runtime allocation, replicated-object publication, tensor-map binding, and
   cancellation state;
5. mixed HBM/mapped/staged KV acquisition with duplicate coalescing, stale
   generations, cancellation, and repeated intent-slot reuse;
6. staged split-K paged attention with one-page request credit;
7. the same staged attention path using hardware TMA; and
8. a differential GPU run against FlashInfer 0.6.12.

The lowered modules contain no bind/acquire/defer marker calls. Tensor-map sites
carry `!nta.acquire` metadata identifying ABI v7 and the `tensor-map` flavor.
The direct branch does not enter the intent pool. NVMe progress uses one warp,
bounds submission and completion work, and has no CTA barrier.

The mixed TMA attention run reports:

```text
memcheck:  ERROR SUMMARY: 0 errors
racecheck: 0 hazards displayed (0 errors, 0 warnings)
synccheck: ERROR SUMMARY: 0 errors
```

## Paged Attention

The workload uses FP16 query/K/V data, head dimension 128, 16-token KV pages,
one CTA per request-owned page, FP32 partial softmax state, and a deterministic
split-K reduction. Requests have heterogeneous page counts and final-page token
counts. A CPU reference checks every output element.

The workload is formed from FlashInfer's public `kv_indptr`, `kv_indices`, and
`last_page_len` representation. Its page CTAs produce normalized `V` plus base-2
`LSE`, and the reduction compiles FlashInfer's `state_t` from the installed
headers. The differential gate uses 7 heterogeneous requests, 23 physical pages,
reversed physical-page indices, and host-staged NTA acquisition. Against a real
`BatchDecodeWithPagedKVCacheWrapper`, it reports:

```text
flashinfer_version=0.6.12 requests=7 physical_pages=23
max_abs_error=2.71425e-05 mean_abs_error=3.2057e-06 matched=1
```

This validates layout and numerical-state compatibility, not execution of NTA
deferral inside FlashInfer's optimized attention CTA.

For 32 requests, 299 pages, and 50 graph iterations, one local run produced:

| Source | Consumer | Graph ms | Logical GiB/s | Max abs error |
| --- | --- | ---: | ---: | ---: |
| HBM | global loads | 0.022 | 105.31 | 2.61e-8 |
| HBM | TMA | 0.019 | 123.23 | 2.61e-8 |
| mapped CPU DRAM | global loads | 0.067 | 34.25 | 2.61e-8 |
| mapped CPU DRAM | TMA | 0.083 | 27.45 | 2.61e-8 |
| staged CPU DRAM | global loads | 0.326 | 7.00 | 2.61e-8 |
| staged CPU DRAM | TMA | 0.318 | 7.18 | 2.61e-8 |
| mixed | global loads | 0.152 | 14.99 | 2.61e-8 |
| mixed | TMA | 0.208 | 10.97 | 2.61e-8 |

These are single-run mechanism numbers, not confidence intervals. `logical
GiB/s` counts full KV pages consumed by tile CTAs and is not raw PCIe or DRAM
bandwidth. TMA is beneficial for the resident case in this run and is not
universally beneficial, especially for mapped host memory. Publication results
must use isolated sequential trials, confidence intervals, and controlled GPU
clocks.

PTXAS reports no spills:

| Kernel | Registers | Shared bytes | Barriers |
| --- | ---: | ---: | ---: |
| global attention tile | 50 | 576 | 1 |
| global ready tile | 52 | 580 | 1 |
| TMA attention tile | 60 | 8,840 | 1 |
| TMA ready tile | 60 | 8,840 | 1 |
| host staging progress | 26 | 24 | 1 |
| NVMe progress | 50 | 0 | 0 |

The one barrier in the global attention kernel belongs to its reduction. The
TMA barrier is initialized only after acquisition returns a valid descriptor.

## GPU-Initiated NVMe

The tested controller is a KIOXIA CD8P E3.S PCIe 5.0 x4 device. The bootstrap
driver creates one depth-64 I/O SQ/CQ, maps queue and doorbell memory, and
registers either mapped CPU DRAM or CUDA HBM exported through DMA-BUF. One
finite CUDA graph performs discovery, bounded GPU SQE/PRP construction, GPU
doorbells, bounded GPU CQ handling, ready publication, and checksum work. The
CPU does not submit or complete commands on the loaded path.

Earlier hardware validation used 16 independent 64 KiB reads, 10 measured
graph launches, and 128 finite progress/resume pairs. All 176 expected commands
completed and every checksum matched the block-device reference:

| Destination | Graph ms | Logical MiB/s | Failed CQEs | Verification failures |
| --- | ---: | ---: | ---: | ---: |
| mapped CPU DRAM | 0.743 | 1,345.90 | 0 | 0 |
| DMA-BUF HBM | 0.679 | 1,473.54 | 0 | 0 |

The controller is currently owned by the host's `vmem_sw` consumer and has been
restored to its original driver. The safety helper refuses to rebind it while
that consumer is active, so ABI v7 and the latest scheduler changes have not
been revalidated on NVMe hardware. This is a required regression gate.

## Production Gates

The following are not validated:

- NTA deferral inside FlashInfer's optimized prefill/decode CTAs;
- SGLang and vLLM request lifecycle, KV-manager, and attention-backend
  integration;
- TTFT, TPOT, p50/p99 latency, SLO attainment, and serving goodput;
- statistical direct-path overhead against an untouched production kernel;
- automatic compiler recognition of production load/cp.async/TMA address
  cones instead of explicit frontend markers;
- host-staging global priority order, NVMe weighted-fairness hardware
  validation, and starvation aging;
- GPU-initiated RDMA submission/completion on a real RNIC;
- latest-ABI NVMe regression, timeout/reset recovery, multiple queues, and
  multi-tenant security isolation;
- MoE or ANNS generality workloads; and
- literature-complete novelty or a paper-quality baseline/ablation matrix.

The installed vLLM 0.13.0 wheel requires PyTorch 2.9.0, while this environment
has PyTorch 2.11.0+cu130. Its CUDA extension fails to load with an unresolved
`c10_cuda_check_implementation` symbol. A matched container or rebuilt vLLM is
required before serving experiments. This machine also has no Mellanox/RDMA
device, so an RDMA implementation cannot be honestly validated here.
