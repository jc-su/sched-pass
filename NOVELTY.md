# Novelty Audit

Date: 2026-07-31

## Bottom line

The project should **not** be reframed around grouped-LPT, CTA ordering, cache
hints, GPU-initiated NVMe, a unified memory/storage API, or warp-specialized
I/O. Each of those ideas has direct prior art, and several are already used for
LLM inference.

The strongest defensible contribution in the current repository is narrower:

> **An ABI-preserving, late-NVVM bridge that binds live control-plane request
> identity and policy to the CTA/tile identity of an existing production fused
> kernel, and attributes device observations back to requests, without forking
> the kernel source.**

The novelty is the **request-to-tile semantic binding and bidirectional
observe/act path**, not any individual scheduling heuristic or GPU instruction.
The pass supplies the mechanism; the serving integration supplies the live
request binding.

An I/O extension can be novel only if its contribution is an **automatic
compiler transformation of an existing kernel**, not another GPU I/O stack.
The promising extension is compiler-generated access-plan extraction and
finite GPU-side I/O submission for request-bound objects.

This is a literature audit, not a patent search or proof that no unpublished
work exists. Claims should use "to the best of our knowledge" and name the
distinguishing properties explicitly.

## The real problem

A serving scheduler knows request identity, priority, SLO state, cache
residency, and batch churn. A compiled fused kernel sees CTAs, pointer
arithmetic, plan arrays, and loads. The hardware block scheduler sees resource
availability, but not which user request a CTA serves.

Existing engines can form a batch and construct a static kernel plan.
[FlashInfer](https://arxiv.org/abs/2501.01005), for example, has a
load-balanced attention scheduler. That does not provide a general live channel
for an external control plane to:

1. identify the current request served by each tile after continuous-batch
   churn;
2. install per-request decisions into an already optimized kernel;
3. attribute sampled device observations back to the correct request; and
4. do so without changing the kernel launch ABI or maintaining a source fork.

The problem is real. [PackInfer](https://arxiv.org/html/2602.06072) and
[PAT](https://arxiv.org/abs/2511.22333) both show that heterogeneous requests,
prefix locality, and long-request CTAs affect attention performance. Their
solution is a new packing scheduler plus specialized kernel/layout. Sched-pass
should be positioned as a **retrofit mechanism for production kernels**, not as
a better packing heuristic.

## What the repository implements

The current implementation has the following end-to-end path:

1. The SGLang/FlashInfer integration reads the plan's `request_indices` each
   step to obtain the live `tile -> request slot` binding.
2. The control plane expands per-request state into per-tile order/policy
   tables.
3. Fixed-address device tables provide a zero-parameter, graph-replay-compatible
   channel into the kernel.
4. The LLVM pass replaces the logical CTA index with a guarded table
   indirection and can add typed actions or observation.
5. Per-tile timing is folded through the same live binding to update
   per-request estimates.

Important claim boundary: the compiler pass does **not** currently recover full
request identity or storage-object identity by itself. It recognizes task and
streaming-load structure; the serving plugin supplies the semantic
`tile -> request` relation. A paper must describe this as a compiler/runtime
bridge, not as fully automatic semantic recovery from arbitrary NVVM IR.

Also, not every action is bit-exact. CTA permutation, observation, and hints can
be bit-exact; shedding and MoE capacity controls intentionally change work or
quality. The effect taxonomy and neutral fallback are defensible. A blanket
"all actions are bit-exact" claim is not.

## Prior-art collision matrix

| Proposed headline | Closest primary prior art | Verdict |
|---|---|---|
| Grouped-LPT request scheduling | [PackInfer](https://arxiv.org/html/2602.06072) sorts requests by descending length and places each in the least-loaded feasible group | Exact collision; use only as a baseline/action |
| Request grouping plus locality | [PackInfer](https://arxiv.org/abs/2602.06072) jointly groups requests and reorganizes KV; [PAT](https://arxiv.org/abs/2511.22333) packs prefix-sharing queries into CTAs | Not novel |
| Load-balanced attention scheduling | [FlashInfer](https://arxiv.org/abs/2501.01005) already advertises dynamic load-balanced scheduling | Not novel |
| Compiler-inserted GPU observation | [KPerfIR](https://www.usenix.org/conference/osdi25/presentation/guan) implements profiling as compiler passes; [NVBit](https://research.nvidia.com/publication/2019-10_nvbit-dynamic-binary-instrumentation-framework-nvidia-gpus) instruments precompiled GPU code | Observation alone is not novel |
| Automatic warp specialization | [Tawa](https://arxiv.org/abs/2510.14719) automatically partitions tile programs into producer/consumer warps | Not novel |
| Fine-grained communication in a fused kernel | [Syncopate](https://www.usenix.org/conference/osdi26/presentation/qiang) automatically aligns Triton computation with communication-chunk availability | Broad claim is not novel |
| GPU-initiated NVMe | [BaM](https://arxiv.org/abs/2203.04910) lets GPU threads drive NVMe queues | Not novel |
| Asynchronous I/O directly from GPU threads | [AGIO](https://pure.psu.edu/en/publications/asynchrony-and-gpus-bridging-this-dichotomy-for-io-with-agio/) decouples GPU I/O initiation and completion | Not novel |
| GPU submission with CPU completion | [CoPilotIO](https://www.usenix.org/conference/osdi26/presentation/chen-guanyi) uses split SQ/CQ, CPU polling, and barrier-based wakeup | Not novel |
| GPU file or object abstraction | [GeminiFS](https://www.usenix.org/conference/fast25/presentation/qiu) provides GPU file access; [Tutti](https://arxiv.org/html/2605.03375v1) provides a GPU-native KV object abstraction | Not novel |
| GPU-centric SSD-backed KV cache | [Tutti](https://arxiv.org/html/2605.03375v1) integrates GPU io_uring, KV objects, async layerwise I/O, and slack-aware scheduling into vLLM | Direct collision |
| Hierarchical/SLO-aware KV placement | [Strata](https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang) manages HBM/DRAM/SSD; [OrbitFlow](https://arxiv.org/abs/2601.10729) adapts per-request layer placement from runtime feedback | Not novel |
| Fused host-memory KV access | [DirectKV](https://www.usenix.org/conference/osdi26/presentation/luo) uses direct CPU-memory access and warp-level fetch/compute/writeback pipelining | Not novel |
| Fused sparse-KV recall/prefetch | [ECHO](https://www.usenix.org/conference/osdi26/presentation/liu-guangda) uses lossless prefetching and a fused pipelined GPU kernel | Not novel |
| Remote GPU-centered split-phase I/O | [GORIO](https://arxiv.org/abs/2607.04415) keeps miss generation, pending state, and resume decisions on GPU | Broad claim is not novel |
| Compiler-generated I/O prefetch | [Mowry et al.](https://www.usenix.org/conference/osdi-96/automatic-compiler-inserted-io-prefetching-out-core-applications) automatically insert out-of-core I/O prefetch; [PUMP](https://scholars.lib.ntu.edu.tw/entities/publication/7efa6c4a-22b9-48cd-a057-0dd1c2da9aa3) extracts GPU-kernel memory blocks for prefetch | Generic compiler-prefetch claim is not novel |

## Defensible current claim

The paper claim should require the conjunction below. Removing any one item
makes it look like established profiling, scheduling, or instrumentation work.

1. **Live request semantics:** request/tile association is rebound every
   continuous-batching step.
2. **Late production-kernel weaving:** actions are inserted at NVVM IR after the
   optimized kernel has been generated; no maintained FlashInfer kernel fork is
   required.
3. **ABI preservation:** fixed-address or equivalent device channels avoid
   adding launch parameters and remain compatible with graph replay.
4. **Bidirectional binding:** policy enters the kernel and observations leave
   it through the same request/tile relation.
5. **Finite-kernel execution:** the mechanism does not require a persistent
   worker kernel.
6. **Effect-typed safety:** neutral state preserves stock behavior; each action
   has an explicit semantic class and failure mode.

A suitable one-sentence claim is:

> Sched-pass makes an existing fused GPU kernel request-aware by weaving an
> ABI-preserving, per-step request-to-tile control and feedback channel into
> late NVVM IR, enabling finite-kernel request-level actions without rewriting
> the kernel source.

Do not claim that CTA timing is a universal request cost model. A CTA sample is
useful only after aggregation through the live tile/request binding and only as
a relative signal for a specified action. The evidence must show that this
closed loop predicts or corrects a decision better than request length alone.

## A novel-enough I/O extension

The extension should be called **request-semantic I/O weaving**, not a unified
GPU I/O system.

### Compiler transformation

For an explicitly identified external-backed pointer or plan field, the pass:

1. finds the loads whose addresses depend on that external object;
2. backward-slices the pure address/key-generation logic;
3. binds each key to the existing logical tile and request slot;
4. clones the slice into a finite access-plan/issue kernel; and
5. emits descriptors such as
   `{object_id, destination, bytes, request_slot, deadline, generation}`.

The finite issue kernel submits descriptors from the GPU to an existing backend
such as BaM, GeminiFS, AGIO, or CoPilotIO. The project should not build another
NVMe driver. A generation-tagged readiness contract prevents stale completion
from satisfying a later batch.

### Execution model

The first design should **not** inject NVMe submission and polling into the same
attention CTA. Attention cannot consume missing KV before completion, and a CTA
must either poll, block, yield through a supported primitive, or terminate and
resume later. No design can simultaneously remove all four.

Use a finite split phase:

```text
plan/route kernel
    -> compiler-generated GPU issue kernel
    -> asynchronous storage/network backend
    -> readiness generation
    -> original finite compute kernel
```

I/O is GPU-initiated because the GPU constructs and submits the request. A CPU
may reap completions, as in CoPilotIO, without becoming the I/O initiator. Other
ready batch work runs while the object is in flight. The compute CTA never
busy-waits on NVMe.

A common descriptor can target HBM, host DRAM, remote memory, or NVMe, but that
unified surface is engineering, not novelty. The research claim is that the
compiler **derives and installs the request-bound access plan for an existing
kernel** and selects a legal finite execution protocol.

## Evaluation required for the claim

### Current bridge

1. Transform at least two real production kernel families, ideally FlashInfer
   attention and a fused MoE path.
2. Show no kernel-source or launch-ABI changes.
3. Compare stock, bridge-only, open-loop heuristic, and closed-loop policy.
4. Report end-to-end goodput, TTFT/TPOT p95/p99, graph-replay compatibility,
   and instrumentation overhead, not only isolated kernel makespan.
5. Include negative regimes: CLC, L2 hints in DRAM-bound decode, and ordering
   when the grid has many balanced waves.

### I/O extension

1. Automatically derive the same access set as a handwritten implementation.
2. Compare with explicit runtime prefetch and with the selected backend's
   native programming model.
3. Show benefits from the transformation, not merely from using BaM/AGIO.
4. Measure issue overhead, I/O amplification, queue depth, overlap, blocked-SM
   time, and end-to-end SLO attainment.
5. Demonstrate at least two kernels or two different address patterns; a
   kernel-name-specific rewrite is not a compiler contribution.

## Kill criteria

Stop or reframe the I/O extension if any of these holds:

1. The access plan is already completely available in the serving runtime and
   the compiler only copies it into another format.
2. The pass needs kernel-specific names, source edits, or handwritten address
   extractors for each target.
3. The speedup disappears when compared with a native implementation using the
   same I/O backend.
4. The attention CTA must busy-wait for storage completion.
5. The only positive result is a microbenchmark with no end-to-end serving
   benefit.

## Recommended scope

The lower-risk paper is **Semantic Weaving for Request-Aware Production GPU
Kernels**, centered on the implemented request/tile bridge, closed-loop
observation, effect safety, and real SGLang/FlashInfer/MoE integration.

The higher-risk follow-up is **Compiler-Weaved Request-Semantic I/O for Finite
GPU Kernels**. It is worth pursuing only after a small prototype proves
automatic access-plan extraction. Adding generic GPU I/O calls to the current
pass would increase implementation size without creating a defensible new
contribution.
