# Validation Record

Date: 2026-07-31

This record supports a locally validated mechanism prototype. It does not
establish production readiness, an end-to-end serving result, or an OSDI-level
evaluation. The missing evidence is listed explicitly below.

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

```bash
NTA_SANITIZE=1 ./scripts/validate-local.sh
./scripts/measure-direct-overhead.sh
```

The first command builds the pass/runtime, runs CTest, evaluates global-load
and TMA attention across placements, runs the MoE workload, emits PTXAS reports,
and executes memcheck, racecheck, and synccheck. The second runs alternating
process-level baseline/mechanism trials and computes Student-t intervals.
Generated evidence is under `results/` and is excluded from source control.

## Correctness Gates

`ctest --test-dir build --output-on-failure` passes 17 tests:

1. FlashInfer CSR-to-common-plan validation, including grouped pages and bad
   metadata;
2. engine-neutral plan construction and bounded dependency validation;
3. LLVM byte-address, tensor-map, and dependency-set lowering, including
   rejection of live-state, token, missing-binding, non-inlined-helper,
   lane-divergent control, and lane-divergent operand cases;
4. host/device ABI v9 layout;
5. Clang nvcc-shim compilation of a foreign source kernel, including automatic
   optimizer-last lowering, marker removal, metadata, and fast-math forwarding;
6. compilation and linking of FlashInfer's real multi-source custom decode and
   paged-prefill extensions through the same JIT activator and isolated cache;
7. resident, pinned-host deferred, shared-head, split-K decode, and multi-tile
   paged-prefill execution inside those optimized FlashInfer kernels;
8. runtime allocation, object/replica capacities, cancellation, non-owning
   engine allocation registration, reusable pinned/async device-plan upload,
   and runtime binding validation;
9. mixed-tier acquisition with duplicate coalescing, stale generations,
   cancellation, and repeated intent-slot reuse;
10. a three-object-per-CTA mixed-tier dependency set;
11. stale object-version failure without output publication;
12. a 4,096-CTA all-direct scale case with `pending=0`;
13. routed top-2 MoE expert matrices across mixed tiers;
14. the matching direct-address numerical baseline;
15. staged split-K paged attention through the common work plan;
16. the same common-plan attention path using hardware TMA; and
17. differential output validation against FlashInfer 0.6.12.

The pass now proves a canonical finite defer edge and CTA collectivity. Markers
must be inlined into a GPU kernel entry, where the CTA analysis treats kernel
arguments, `blockIdx`, and block/grid dimensions as CTA-uniform. It rejects
non-inlined helpers and control or marker operands derived from `threadIdx`,
lane/warp identity, atomics, volatile loads, local allocation, or unknown
calls. Lowered modules contain no bind/acquire/defer markers and carry ABI-v9
`!nta.acquire` metadata.

Attention global-load and TMA kernels, the generic dependency-set kernel, and
the MoE kernel all consume the same `abi::WorkItem` and
`abi::AcquireRequirement` arrays. `DeviceWorkPlan` supports fixed-capacity
reuse, pinned staging, stream-ordered asynchronous updates, and explicit
cross-stream waits. Attention-only side metadata contains token-count and
request-index fields, not a duplicate acquisition binding.

Readiness publication scans the bounded pending index rather than the entire
continuation directory. The 4,096-CTA resident test confirms that an all-direct
epoch creates no pending entries. Publication uses at most 32 finite CTAs, no
CTA barrier, and a grid-stride loop over actual pending entries.

A separate CUDA-disabled build against the supported LLVM 22 installation
passes all four applicable adapter, plan, IR, and ABI tests.

The JIT activator compiles and links FlashInfer 0.6.12's real multi-source
custom decode and paged-prefill extensions in an isolated NTA cache. The
version-checked overlay places acquisition sites at their global kernel entry
wrappers, and the execution gate exercises resident and deferred continuation
through those optimized kernels.

## Sanitizers And Resources

The latest mixed-tier TMA attention, dependency-set, and MoE runs report:

```text
memcheck:  ERROR SUMMARY: 0 errors
racecheck: 0 hazards displayed (0 errors, 0 warnings)
synccheck: ERROR SUMMARY: 0 errors
```

PTXAS reports no spills. Current key resources are:

| Kernel | Registers | Shared bytes | Barriers |
| --- | ---: | ---: | ---: |
| dependency initial / ready | 60 / 60 | 128 / 132 | 1 / 1 |
| direct numerical baseline | 32 | 128 | 1 |
| MoE initial / ready | 64 / 62 | 0 / 4 | 1 / 1 |
| attention global initial / ready | 62 / 62 | 576 / 580 | 1 / 1 |
| attention TMA initial / ready | 62 / 64 | 8,840 / 8,840 | 1 / 1 |
| pending readiness publication | 24 | 0 | 0 |
| host staging progress | 26 | 24 | 1 |
| NVMe progress | 50 | 0 | 0 |

The barriers in dependency, MoE, and attention kernels belong to numerical
cooperation. Acquisition returns before those barriers are reached. The TMA
barrier is initialized only after the common dependency set is ready and a
valid direct or staged tensor-map descriptor resolves.

## Direct-Path Cost

Ten alternating process-level trials ran 200 captured-graph iterations each.
Both variants used the same graph topology and four-object numerical work; the
baseline bypassed request/acquisition logic only.

| Variant | Mean logical GiB/s | 95% t interval |
| --- | ---: | ---: |
| direct-address baseline | 394.51 | +/- 0.05 |
| ABI-v9 dependency set | 381.29 | +/- 0.04 |

The paired throughput reduction is **3.35% +/- 0.02 percentage points**. This
is a real, nonzero mechanism cost. GPU clocks were not fixed, this is a
microbenchmark rather than an untouched production kernel, and the interval
does not include machine-to-machine variation.

## Paged Attention

The workload uses FP16 Q/K/V, head dimension 128, 16-token pages, one CTA per
request-owned page, FP32 partial softmax state, and deterministic split-K
reduction. It forms the common plan from FlashInfer's public `kv_indptr`,
`kv_indices`, and `last_page_len`, then uses the same dependency records for
global loads or TMA.

The differential gate covers 7 heterogeneous requests, 23 physical pages,
non-identity page indices, and staged acquisition:

```text
flashinfer_version=0.6.12 requests=7 physical_pages=23
max_abs_error=2.71425e-05 mean_abs_error=3.2057e-06 matched=1
```

The optimized-kernel gate additionally executes NTA in FlashInfer 0.6.12:

```text
resident decode: pass
pinned-host deferred decode: Pending -> Ready -> Done, max error 0
shared KV-head CTAs: 2
split-K decode work items: 32, max error 0
FA2 paged-prefill work items: 4, max error 0
```

The matched custom-variant microbenchmark uses 64 requests and 2000 iterations
per alternating sample. The latest local median measured 11.980 us without NTA
fields and 12.739 us with the resident hook, a 6.33% cost. CTest fails above 8%.
Clocks are not fixed, so this is a local regression gate rather than a
paper-quality result.

Compute Sanitizer on the two-head deferred path reports memcheck 0 errors,
racecheck 0 hazards, and synccheck 0 errors.

One latest-code 8-request, 60-page, five-iteration mechanism sample was:

| Source | Consumer | Graph ms | Logical GiB/s | Max abs error |
| --- | --- | ---: | ---: | ---: |
| HBM | global loads | 0.023 | 19.66 | 2.42e-8 |
| HBM | TMA | 0.021 | 21.62 | 2.42e-8 |
| mapped CPU DRAM | global loads | 0.045 | 10.21 | 2.42e-8 |
| mapped CPU DRAM | TMA | 0.035 | 12.93 | 2.42e-8 |
| staged CPU DRAM | global loads | 0.108 | 4.24 | 2.42e-8 |
| staged CPU DRAM | TMA | 0.111 | 4.13 | 2.42e-8 |
| mixed | global loads | 0.084 | 5.42 | 2.42e-8 |
| mixed | TMA | 0.074 | 6.19 | 2.42e-8 |

These are smoke-test mechanism numbers, not controlled performance results.

## MoE Generality

The MoE gate routes each token to two expert matrices, acquires both through
the common plan, performs real matrix-vector products, mixes routed outputs,
and checks every element against a CPU reference. A 64-token, 16-expert,
hidden-size-128 mixed-tier run completed at 0.205 ms per graph, 38.20 logical
GiB/s, with five staged expert transfers and zero numerical failures. This
closes the synthetic-only generality gap, but it is not a production MoE model
or serving baseline.

## GPU-Initiated NVMe

Earlier hardware validation used a dedicated KIOXIA CD8P controller, one
depth-64 SQ/CQ, GPU-built READ SQEs/PRPs, GPU MMIO doorbells, bounded GPU CQ
handling, and either mapped DRAM or DMA-BUF HBM destinations. Sixteen
independent 64-KiB reads completed with matching checksums:

| Destination | Graph ms | Logical MiB/s | Failed CQEs | Failures |
| --- | ---: | ---: | ---: | ---: |
| mapped CPU DRAM | 0.743 | 1,345.90 | 0 | 0 |
| DMA-BUF HBM | 0.679 | 1,473.54 | 0 | 0 |

That run predates ABI v9. The controller is currently owned by the host's
`vmem_sw` consumer, and the safety helper correctly refuses to rebind it.
Latest-code NVMe evidence therefore remains open.

## Open Production And Paper Gates

- vLLM and SGLang lifecycle, KV-manager, cancellation, and CUDA-graph adapters;
- TTFT, TPOT, p50/p99, SLO attainment, serving goodput, CPU use, and SM tax;
- direct-path comparison against untouched production kernels with controlled
  clocks and multiple machines;
- automatic recognition of production load/`cp.async`/TMA address cones rather
  than explicit frontend markers;
- host-staging global priority order, NVMe weighted-fairness hardware results,
  and starvation aging;
- GPU-initiated RDMA submission/completion on a real RNIC;
- ABI-v9 NVMe regression, timeout/reset recovery, multiple queues, and
  multi-tenant security isolation;
- production MoE and optional ANNS baselines; and
- literature-complete novelty analysis plus a paper-quality baseline and
  ablation matrix.

The installed vLLM 0.13.0 wheel requires PyTorch 2.9.0, while this environment
has PyTorch 2.11.0+cu130; its CUDA extension fails to load with an unresolved
symbol. A matched container or rebuild is required. This machine has no
Mellanox/RDMA device. Those are external testbed blockers, not completed gates.
