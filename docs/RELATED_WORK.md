# Related-Work And Novelty Audit

Audit date: 2026-07-31

This is a live design constraint, not a claim of novelty. The project must not
claim "first", "production-ready", or "OSDI-level" until the implementation and
evaluation distinguish it from the systems below.

## Closest Systems

### FlashInfer

[FlashInfer](https://arxiv.org/abs/2501.01005) already provides optimized paged
prefill/decode, load-balanced KV chunk scheduling, split-K state, and cascade
merge. NTA cannot claim novelty from paged-KV layout, CTA chunking, or associative
`(V, LSE)` reduction. The implementation deliberately adopts those public data
and numerical-state contracts.

The candidate distinction is compiler-generated incremental execution of
FlashInfer chunks whose data becomes available at different times, followed by
a request-bound runnable-work launch that reconstructs the original scheduler
tile and preserves partial reduction state. The current hook and heterogeneous
work remapping execute inside optimized FlashInfer CTAs; the complete compiler
transformation and end-to-end benefit remain unproven.

### Syncopate (OSDI 2026)

[Syncopate](https://www.usenix.org/conference/osdi26/presentation/qiang) is the
closest compiler baseline. It introduces a communication-chunk abstraction and
transforms Triton kernels to align fine-grained computation with chunk
availability inside fused kernels. NTA cannot claim novelty from a unified
chunk/acquisition abstraction, compiler-inserted overlap, or preserving a fused
kernel alone.

The remaining candidate distinction is request-scoped incremental operator
execution when data arrival can outlive a finite CTA: the issuing CTA exits
without retaining live state, transport progress is bounded, partial numerical
state is compiler managed, and actual progress feeds later batch admission.
This distinction must be evaluated against a Syncopate-style compiler chunk
schedule using the same transport.

### Strata (OSDI 2026)

[Strata](https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang)
already provides production hierarchical KV caching, GPU-assisted I/O, transfer
coalescing, and cache-aware batching in SGLang. NTA cannot claim novelty from
KV tiering, coalesced transfers, or scheduler awareness of loading latency.

The relevant comparison is whether compiler-generated partial execution inside
the real operator and feedback of actual partial progress to the batch scheduler
improve beyond a Strata-style coalesced transfer and batch plan. NTA must
preserve bulk behavior for dense demand; the distinction cannot rest on finer
transfer granularity alone.

### ECHO (OSDI 2026)

[ECHO](https://www.usenix.org/conference/osdi26/presentation/liu-guangda)
already performs graph-friendly sparse-KV eviction/recall, lossless intra-query
and inter-query prefetch, and fused overlap between recall and indexer
computation. NTA cannot claim novelty from sparse-KV offload, GPU-graph recall,
or fused index/fetch overlap.

The remaining comparison is a true miss outside ECHO's lossless prefetch
window: whether compiler-generated incremental execution outperforms waiting,
conservative prefetch, or a manually split indexer/attention pipeline.

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

### Tutti

[Tutti](https://arxiv.org/abs/2605.03375) eliminates CPU intervention from the
critical HBM/SSD data and control paths using a GPU-native KV object store,
asynchronous GPU direct object I/O, and slack-aware scheduling. GPU-native KV
objects, CPU-free NVMe submission, and avoiding compute interference are
therefore not independent NTA contributions.

NTA must compare its finite CTA try-issue plus bounded completion model against
Tutti using the same SSD topology and serving workload. The candidate
distinction remains compiler-generated nonresident work ticket at the exact
consumption site, especially for demand discovered after launch.

### SPDK

[SPDK](https://spdk.io/doc/nvme.html) is the required userspace-NVMe bootstrap
and CPU-performance reference, not a novelty baseline. Its public qpair API
retains CPU ownership of trackers, SQ/CQ state, doorbells, and completions. NTA
therefore does not embed SPDK or depend on private qpair layouts to hand a queue
to the GPU. Evaluation should compare controller initialization traces and a
matched SPDK CPU-polling workload against the narrow VFIO/IOMMUFD bootstrap and
finite GPU progress path.

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

The defensible research question is narrower than "unified GPU I/O" and broader
than GPU-selected sparse demand:

> Can a compiler turn an all-or-nothing batched GPU operator into request-scoped
> incremental execution, then let the serving runtime jointly schedule arriving
> data, useful partial computation, and later batch admission without persistent
> GPU workers?

The implementation now establishes feasibility for explicitly marked
byte-address and TMA sites over HBM, CPU DRAM, and NVMe. It also includes two
device-generated-demand fixtures: GPU-routed MoE and query-dependent sparse
attention. In
the latter, the query is materialized on device immediately before an
attention CTA selects top-k external pages; userspace receives neither the
selection nor an inter-kernel synchronization point. A matched one-GPU MoE
mechanism experiment shows a large win over all-expert overfetch and a small
win over a generous CPU-sync lower bound. The sparse path currently establishes
correctness and a controlled crossover: it loses to bulk overfetch for a small
catalog and wins for a large low-selectivity catalog. This is standalone
mechanism evidence, not a production sparse-serving result. The project does
not yet establish automaticity across kernel families, production serving
benefit, RDMA generality, or an OSDI-level result.

The novelty unit is compiler-generated incremental execution of a real atomic
operator plus preservation of request/tile identity across engine planning,
finite-kernel partial work, asynchronous acquisition, completion, numerical
merge, and subsequent batch feedback. Cancellation-safe work tickets,
logical-work remapping, a common source descriptor, TMA after staging,
GPU-initiated submission, priority scheduling, prefetch distance, CTA
permutation, locality placement, and cache hints are supporting techniques and
must not be presented as independent contributions.

## Evidence Needed For A Systems Submission

1. A matched vLLM or SGLang integration with real prefill/decode batches,
   production KV layout, and request cancellation/reuse.
2. Automatic or repeatable compiler recognition on more than one production
   kernel family; explicit markers alone are insufficient.
3. Baselines for Strata-style userspace scheduling, ECHO-style lossless
   prefetch, DirectKV-style direct access, Syncopate-style chunk overlap,
   Tutti-style GPU-native storage, CoPilotIO-style CPU completion, and a
   persistent GPU service.
4. Real NVMe experiments with CPU usage, SM tax, TTFT, TPOT, p50/p99, goodput,
   fairness, and failure recovery. Real RNIC experiments are additionally
   required only for an RDMA or network-I/O claim.
5. Ablations isolating request semantics, incremental execution, compiler
   transformation, elastic grouping, engine progress feedback, replica choice,
   admission credits, and bounded progress.
6. A real FlashInfer/SGLang incremental result on dense mixed-arrival KV, plus a
   real FlashInfer GPU-selected sparse path that keeps selected page IDs on the
   device. The custom fixtures satisfy only mechanism correctness today.

Failing any of items 1, 2, 3, or 6 should reframe this project as a GPU I/O
runtime mechanism rather than an OSDI-level compiler/serving contribution. The
full implementation and evidence order is maintained in `SYSTEM_PLAN.md`.
