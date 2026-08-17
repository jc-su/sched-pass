# Related-Work And Novelty Audit

Audit date: 2026-08-10

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
tile and preserves partial reduction state. The compiler now verifies the
acquire-or-exit boundary and a convergent, exactly-once numerical publication
region with identical request/work-ticket identity. The current hook and
heterogeneous work remapping execute inside optimized FlashInfer CTAs;
automatic same-source form generation and end-to-end benefit remain unproven.

FlashInfer is not the only prior partial-state implementation. Production and
research kernels already partition HBM-resident or non-contiguous KV and merge
softmax state; [ChunkAttention](https://aclanthology.org/2024.acl-long.623/) and
[Pensieve](https://arxiv.org/abs/2312.05516) are representative examples. NTA
therefore spends no novelty claim on `(V, LSE)`, segment attention, or a
temporary partial buffer. The relevant question is who selects executable
contributors after asynchronous external arrival, preserves request lifecycle
through that interval, and feeds exact remaining service into SLO scheduling.

### Lynx

[Lynx](https://arxiv.org/abs/2607.01831) already rejects full-KV-arrival as a
prerequisite for useful decode in disaggregated serving. It transfers a
high-priority most-significant-bit stream, decodes speculatively, then uses the
residual stream to recover high-precision behavior. NTA cannot claim to be the
first system to begin attention before complete KV transfer.

The candidate distinction is exact contributor execution without draft,
verification, or rollback; local HBM/DRAM/NVMe hierarchy rather than only a
network transfer; fixed-capacity staging independent of context length; and
request-generation-safe scheduling in a heterogeneous continuous batch. These
clauses require matched evaluation and are not established by the current
fixed-wave operator result.

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

Strata is also a methodological baseline: NTA should reuse its real
long-context datasets, reuse-distance and request-rate sweeps, tier and page-size
sensitivity, and mechanism ablations. Reusing that evaluation view does not
make NTA another Strata; claiming the same cache-management contribution would.

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

### Serving Schedulers And Semantic-Gap Systems

[Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)
and [Llumnix](https://www.usenix.org/conference/osdi24/presentation/sun-biao)
are scheduling and evaluation references. They require NTA to report maximum
sustainable load under tail-latency SLOs, burstiness and length sensitivity,
and isolated scheduler overhead rather than one favorable kernel latency.
Their iteration- and request-level scheduling mechanisms are controls for
whole-request delay and rebatching, not claims NTA may rename.

[Parrot](https://www.usenix.org/conference/osdi24/presentation/lin-chaofan)
shows that application semantics hidden behind individual API calls can change
serving decisions. NTA addresses a different semantic gap: userspace knows
request lifecycle, tier placement, and SLO state but not the optimized kernel's
request-to-CTA and reduction structure, while the kernel knows the demanded
tile and numerical contributor but not lifecycle or transport state. The
typed frontend and request directory make those facts jointly available at the
consumption boundary. A descriptor alone does not close this gap; identity must
survive acquisition, finite exit, relaunch, reduction, and later admission.

[ServerlessLLM](https://www.usenix.org/conference/osdi24/presentation/fu) is a
methodological reference for separating storage bandwidth, loading pipeline,
component overhead, and end-to-end impact. NTA must similarly prevent a faster
transport microbenchmark from being reported as an operator or serving win.

### Selective KV Systems

[InfiniGen](https://www.usenix.org/conference/osdi24/presentation/lee) requires
selective-KV comparisons to include accuracy or perplexity, sequence/batch
sensitivity, and equal-quality baselines. Sparse attention remains a stress
case for device-discovered demand, not the only workload or a substitute for
the dense fragmented-arrival result.

[SparseServe](https://arxiv.org/abs/2509.24626) combines dynamic sparse
attention with hierarchical HBM/DRAM management, fragmented-transfer paths,
working-set-aware batch sizing, and layer-segmented prefill. NTA cannot claim
novelty from selective KV transfer, bounded layer staging, or converting sparse
working-set reduction into higher admission capacity.

[SPIN](https://arxiv.org/abs/2604.26837) co-designs sparse attention with
hierarchical KV storage through a common partition abstraction, per-request HBM
budgets, GPU-oriented replacement, and active-working-set metadata. A common
page/partition abstraction and request-sized staging policy are therefore not
independent NTA contributions. Both systems are required sparse-serving
controls; NTA must distinguish itself through the compiler-checked contributor
and request-lifecycle contract and must still match their quality constraints.

### GPU-Initiated Storage Primitives

[BaM](https://arxiv.org/abs/2203.04910) (ASPLOS 2023) established GPU threads
submitting NVMe commands through queue pairs mapped into GPU memory, with an
array abstraction and a GPU-resident software cache. GPU-initiated storage
access, fine-grained on-demand fetch, and a device-side cache are therefore
BaM-lineage techniques, not NTA contributions, and the runtime's GPU-initiated
NVMe path belongs to this primitive class. The anticipated reviewer challenge
is exact and must be answered head-on: "is the win just GPU-initiated I/O,
which BaM already provided?"

The recorded answer has two parts. First, the qualifying campaigns to date
stage from the pinned-host tier; the measured paths contain **no GPU-initiated
storage I/O at all**, and both arms use the same host-DMA primitive stock
SGLang uses for load-back. The measured separation is therefore attributable
to what sits above the transfer: which bytes move, how many rows are resident,
and how the engine trusts the result. Second, BaM has no concept of a request.
It exposes a pointer; it has no request lifetimes or generations, no
retraction or cancellation, no multi-tenant batch, no scheduler consuming
progress accounting, no admission, and no SLOs. Every safety mechanism this
project needed — claim identity against pool-row reuse, generation-tagged
retirement, cancellation fencing across streams, verified consumption — exists
because a serving engine sits above the transfer, a layer BaM never addresses.
BaM is cited as the access-primitive lineage; the contribution is the
engine-trust contract that makes device-decided acquisition consumable by a
live serving engine.

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

> Can a compiler expose request-owned data dependencies, executable
> contributors, and exact completion conditions from optimized finite GPU
> operators, enabling an SLO runtime to jointly prioritize external data and
> GPU computation without persistent GPU workers?

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

The novelty unit is a compiler/runtime co-designed **request-scoped incremental
operator**: the typed frontend exposes request, dependency, and associative
reduction structure; LLVM verifies a zero-live-state acquire-or-exit effect and
a convergent exactly-once numerical publication effect; the runtime preserves
that identity across asynchronous acquisition and finite relaunch; and the
engine admits later work using measured request-local data and compute progress.
This applies to every transformed operator invocation, including the direct
form. Only unavailable contributors take the finite split-phase path.

The landscape is usefully divided into four controls: hide an unchanged
operator barrier with caching/scheduling (Strata), remove movement with coherent
or near-data hardware (DirectKV and computational storage), shrink or speculate
on data (ECHO and Lynx), and expose exact external-arrival contributors inside
the finite operator (the NTA candidate). The last category is defensible only if
the real completion-driven system wins against the first three where their
hardware and accuracy assumptions apply.

Cancellation-safe work tickets,
logical-work remapping, a common source descriptor, TMA after staging,
GPU-initiated submission, priority scheduling, prefetch distance, CTA
permutation, locality placement, and cache hints are supporting techniques and
must not be presented as independent contributions.

## Where The Measured Wins Come From (recorded 2026-08-14)

The qualifying campaigns decompose the end-to-end result into ingredients, and
the paper must attribute each honestly rather than let the composition absorb
credit for its parts:

- **~16x fewer KV bytes read per decode step** comes from Quest-lineage
  selection. The selection algorithm is not an NTA contribution; the paper
  claims only its execution and validation on-device inside a live engine.
- **Graph-speed decode** comes from SGLang's CUDA graphs. The NTA contribution
  is capture-compatibility: a decode step whose data identities are chosen
  during execution could not previously replay under a captured graph
  (campaign three and four record graphs enabled in both arms).
- **Transfer primitives** are BaM/HiCache lineage, and the current evidence
  path uses only pinned-host DMA available to both arms identically.
- **The NTA contribution is the composition**: bounded admission without dense
  allocation (external TTFT p95 ratios 0.011x-0.054x across campaigns),
  device-decided selection the engine can trust (claim table, generations,
  fenced retirement, quality gates), and capture-compatible consumption
  (decode TPOT 0.60x stock at the pressure shape). None of the three
  ingredient systems composes with the others today; the contracts that make
  the composition legal are the claimed novelty, and the campaign records in
  `PREREGISTRATION.md` are its evidence.

The empirical closure for this attribution is the host-orchestrated sparse
baseline (RQ3): the same selection quality with host-side fetch and
orchestration — the best system buildable from the ingredient lineages without
the device-side claim chain. If that arm matches the tiered arm, the
composition claim fails and this document's framing must be withdrawn. The
co-resident tail is recorded as an honest cost, not hidden: campaign four
measures resident P99 ITL crossing the absolute 100ms SLO in three of ten
trials at the load-symmetric shape with graphs enabled.

**First measurement (2026-08-15, flagship point):** the host-orchestrated arm
was built with everything held identical — selection algorithm and budget,
bounded admission, transfer primitive, graph replay between refreshes — except
that staging control runs through the host round-trip a BaM-style system
requires. Validated before measurement: quality parity (multikey and needle
1.0 on both arms), byte-exact staging (2,844 host-staged layers verified
against pinned host sources, zero mismatches), and purity witnesses enforced
in both directions. At the capacity shape with refresh interval 1024 (five
paired trials, same registered seeds as the device arm's ten): host-orchestrated
goodput geomean is **0.529x the device chain's** (3.24 versus 6.13 requests
per second), **0.92x dense stock on the registered metric** [0.797, 1.011],
and its resident P99 ITL runs **14.1x stock** — per-layer host staging
serialized on the scheduler thread destroys co-resident tails outright. The
measured sentence for the reviewer: raw fetch plus host orchestration
approximately matches dense serving on goodput while violating co-tenant
isolation; the device-side claim chain roughly doubles goodput on top of it
and keeps the tail bounded. The same-revision paired
measurement (2026-08-17, revision bfc94e1, five paired trials per arm,
seeds verbatim, writeback summaries enabled in both, pinned control
buffers in the host arm, artifacts `results/serving/rq3sr-*`): the device
chain serves **2.11x** stock's registered goodput while the host-orchestrated
arm serves **0.982x** — dense parity — putting the host arm at **0.427x**
the device chain; and the host arm's resident P99 ITL runs **14.4x** stock
against the device chain's 1.15x. The measured sentence for the reviewer,
now methodologically clean: BaM-lineage fetch plus host orchestration
matches dense serving on goodput while destroying co-tenant isolation;
the device-side claim chain is what produces both the goodput win and the
bounded tail. The remaining fairness note stands: this host arm pays one
synchronization per claim-layer refresh because decode queries exist only
mid-forward — a deferred-batched variant serving one-step-stale selections
is the strongest conceivable host system and remains future work, recorded
rather than silently skipped. The refresh-interval ladder also remains
open; the device arm's graph/eager-boundary ordering defect it exposed at
refresh 32 was fixed and validated by the replay battery.

## Evidence Needed For A Systems Submission

1. A matched vLLM or SGLang integration with real prefill/decode batches,
   production KV layout, and request cancellation/reuse.
2. Automatic or repeatable compiler recognition on more than one production
   kernel family; explicit markers alone are insufficient.
3. Baselines for Strata-style userspace scheduling, ECHO-style lossless
   prefetch, SparseServe/SPIN-style hierarchical sparse serving,
   DirectKV-style direct access, Syncopate-style chunk overlap, Tutti-style
   GPU-native storage, CoPilotIO-style CPU completion, and a persistent GPU
   service.
4. Real NVMe experiments with CPU usage, SM tax, TTFT, TPOT, p50/p99, goodput,
   fairness, and failure recovery. Real RNIC experiments are additionally
   required only for an RDMA or network-I/O claim.
5. Ablations isolating request semantics, incremental execution, compiler
   transformation, CTA-count versus critical-work scoring, elastic grouping,
   engine progress feedback, replica choice, admission credits, and bounded
   progress.
6. A real FlashInfer/SGLang incremental result on dense mixed-arrival KV, plus a
   real FlashInfer GPU-selected sparse path that keeps selected page IDs on the
   device. The custom fixtures satisfy only mechanism correctness today.

Failing any of items 1, 2, 3, or 6 should reframe this project as a GPU I/O
runtime mechanism rather than an OSDI-level compiler/serving contribution. The
full implementation and evidence order is maintained in `SYSTEM_PLAN.md`.
