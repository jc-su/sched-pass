# Related-Work And Novelty Audit

Audit date: 2026-07-31

This is a live design constraint, not a claim of novelty. The project must not
claim "first", "production-ready", or "OSDI-level" until the implementation and
evaluation distinguish it from the systems below.

## Closest Systems

### Syncopate (OSDI 2026)

[Syncopate](https://www.usenix.org/conference/osdi26/presentation/qiang) is the
closest compiler baseline. It introduces a communication-chunk abstraction and
transforms Triton kernels to align fine-grained computation with chunk
availability inside fused kernels. NTA cannot claim novelty from a unified
chunk/acquisition abstraction, compiler-inserted overlap, or preserving a fused
kernel alone.

The remaining candidate distinction is nonresident acquisition whose latency
can outlive a finite CTA: request generation and cancellation are bound to an
external object, the issuing CTA exits without retaining live state, transport
progress is bounded rather than persistent, and only readiness-selected logical
work is relaunched. This distinction must be evaluated against a Syncopate-style
manual or compiler chunk schedule using the same transport.

### Strata (OSDI 2026)

[Strata](https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang)
already provides production hierarchical KV caching, GPU-assisted I/O, transfer
coalescing, and cache-aware batching in SGLang. NTA cannot claim novelty from
KV tiering, coalesced transfers, or scheduler awareness of loading latency.

The relevant comparison is whether in-kernel request/tile binding and
ready-continuation execution improve SLO attainment beyond Strata-style
userspace batch formation for GPU-discovered or late-bound tile demand.

### DirectKV (OSDI 2026)

[DirectKV](https://www.usenix.org/conference/osdi26/presentation/luo) uses
CPU-memory-aware CUDA kernels, warp-level pipelining, and fused KV generation
and attention to consume CPU-resident KV without an HBM staging buffer on
coherent high-bandwidth CPU-GPU systems. NTA cannot claim novelty from direct
host-memory attention, warp pipelining, or TMA-accessible host memory.

DirectKV is the required direct-memory baseline. NTA's staged path is justified
only where the source is command-addressed or where staging wins on the target
PCIe/CXL/NVLink topology.

### CoPilotIO (OSDI 2026)

[CoPilotIO](https://www.usenix.org/conference/osdi26/technical-sessions#copilotio)
has GPUs initiate storage I/O while CPU cores poll completions, using split
submission/completion queues and adaptive co-polling to avoid wasting SMs.
GPU initiation, asynchronous completion, and lower polling occupancy are
therefore not independent contributions.

NTA must compare bounded GPU completion progress with CoPilotIO's CPU completion
service using identical devices, queue depths, object sizes, and CPU accounting.

### GPU-Owned Remote I/O And Communication

[GORIO](https://arxiv.org/abs/2607.04415) keeps ANNS query evolution, page-miss
generation, pending state, and resume decisions on the GPU over NVMe-oF, while
using persistent GPU scheduling and a CPU proxy. It narrows any ANNS generality
claim to the finite-kernel compiler mechanism rather than GPU-owned pending and
resume state.

[GNStor](https://arxiv.org/abs/2606.04908) includes a GPU-centric NVMe-over-RDMA
stack for direct remote all-flash-array access. [GPU-Initiated Networking for
NCCL](https://arxiv.org/abs/2511.15076) exposes direct and proxy device-side
network backends, while [GICC](https://arxiv.org/abs/2604.22126) addresses
bounded finite NIC state and asynchronous resource reclamation. Direct GPU RDMA
submission, a common device API, and bounded transport state are not sufficient
novelty claims.

## Required Novelty Test

The defensible research question is narrower than "unified GPU I/O":

> Can a compiler bind live request/tile semantics to nonresident external
> acquisition in existing finite GPU kernels, preserve the native direct path,
> and convert only missing logical work into generation-safe ready
> continuations, improving end-to-end SLO efficiency over userspace prefetch,
> chunk-compiler overlap, CPU completion, and persistent GPU progress?

The answer is currently unknown. The implementation establishes feasibility for
explicitly marked byte-address and TMA sites over HBM, CPU DRAM, and NVMe. It
does not yet establish automaticity, production benefit, RDMA generality, or
novelty.

## Evidence Needed For A Systems Submission

1. A matched vLLM or SGLang integration with real prefill/decode batches,
   production KV layout, and request cancellation/reuse.
2. Automatic or repeatable compiler recognition on more than one production
   kernel family; explicit markers alone are insufficient.
3. Baselines for Strata-style userspace scheduling, DirectKV-style direct
   access, Syncopate-style chunk overlap, CoPilotIO-style CPU completion, and a
   persistent GPU service.
4. Real NVMe and RNIC experiments with CPU usage, SM tax, TTFT, TPOT, p50/p99,
   goodput, fairness, and failure recovery.
5. Ablations isolating request semantics, continuation selectivity, compiler
   transformation, replica choice, TMA, admission credits, and bounded progress.
6. A workload where exact demand is late-bound inside the GPU; otherwise
   userspace prefetch may close the semantic gap before kernel launch.

Failing any of items 1, 2, 3, or 6 should reframe this project as a GPU I/O
runtime mechanism rather than an OSDI-level compiler/serving contribution.
