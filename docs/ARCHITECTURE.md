# Nonresident Acquisition Architecture

Status: implementation contract

This document defines the problem, claims, system boundary, execution model, and
implementation gates for the clean `nonresident-acquisition` branch. Decisions
in this document take precedence over the previous prototype's design notes.

Implementation status (2026-07-31): M0-M4 have a working vertical slice tested
on an NVIDIA RTX PRO 6000 Blackwell Server Edition. M5 has a numerically checked
split-K paged-attention mechanism workload plus a public-CSR/attention-state
compatibility layer differentially checked against FlashInfer 0.6.12, not a
serving-framework result. M6
has real TMA descriptor selection and hardware TMA after direct or externally
staged acquisition; automatic production-IR recognition remains open. A KIOXIA
CD8P NVMe controller has DMAed directly into CUDA HBM registered through
DMA-BUF and into registered mapped DRAM. RDMA is not implemented because this
host has no RNIC, and there is no placeholder backend.

## 1. Working thesis

Long-context serving places request-owned KV data across GPU memory, CPU memory,
remote memory, and storage. The serving control plane schedules requests and
knows their SLOs, while optimized GPU kernels consume data as tiles, pages, or
vector ranges. Transport layers see buffers and commands. No layer owns the
complete relation:

```text
request semantics
    -> logical kernel work
    -> external object range
    -> physical source and transport
    -> completion
    -> runnable continuation
```

We will test the following research hypothesis:

> A compiler can bind external data-acquisition sites in existing finite GPU
> kernels to live request semantics, preserve native load/cp.async/TMA fast
> paths for directly addressable data, and convert command-based accesses into
> suspendable acquisition and ready-continuation execution without persistent
> GPU workers or loaded-path CPU control.

The primary workload is external KV for online serving. The mechanism is
designed for request-conditioned external tensors and object ranges rather than
for KV alone.

## 2. Problem

### 2.1 Semantic discontinuity

The serving scheduler knows:

- request and tenant identity;
- deadline, priority, and SLO;
- cancellation and batch generation;
- high-level KV or model placement; and
- global queue and memory pressure.

The GPU kernel knows:

- the current logical tile or token;
- the exact block, expert, or vector selected by GPU execution;
- split-K position and local consumption point; and
- the kernel's actual global-to-shared data-movement pipeline.

The transport knows:

- mapped addresses and registered regions;
- RDMA queue pairs, remote keys, and completion queues;
- NVMe namespaces, LBAs, SQs, and CQs; and
- transfer completion and error status.

Without an explicit binding, transport work is ordered by low-level arrival
rather than request criticality, completions do not directly release request
continuations, and stale work survives cancellation or batch-slot reuse.

### 2.2 Addressability discontinuity

Some sources can be read by the GPU memory system:

- local HBM;
- mapped and pinned CPU memory;
- coherent or CXL-like memory when exposed by the platform; and
- peer GPU memory reachable through the GPU memory fabric.

Other sources require a command interpreted by another device:

- network RDMA over a NIC;
- local NVMe;
- NVMe over Fabrics; and
- remote object or block storage.

A software descriptor can make all of these sources logically addressable. It
cannot make an NVMe LBA or an RDMA remote key physically loadable by TMA.

### 2.3 Lifetime and progress discontinuity

Shared memory and TMA barriers belong to a running CTA. Network and storage
operations can outlive a CTA by microseconds. A compute CTA must not:

- spin on an external completion;
- hold queue credits while waiting;
- hold a lock across asynchronous I/O;
- remain resident solely to provide progress; or
- execute a divergent exit across a required CTA barrier.

The system therefore needs a continuation that survives the issuing CTA.

## 3. Scope

### 3.1 Primary scope

- Read-only or immutable, versioned external objects.
- Continuously batched inference.
- Dense and sparse external KV.
- Finite CUDA kernels compiled through LLVM/NVVM.
- Native global-load, `cp.async`, and TMA acquisition sites.
- HBM, CPU DRAM, RDMA, and NVMe source classes.

### 3.2 Secondary scope

- GPU-routed MoE expert weights.
- ANNS vector blocks.
- Graph adjacency pages when they fit the object-range model.

### 3.3 Explicit non-goals

- A POSIX file system.
- Transparent coherent virtual memory across all tiers.
- A general GPU page-fault handler.
- A replacement for `dma-buf`, GPUDirect, NVSHMEM, or NVMe protocols.
- Persistent GPU kernels.
- A claim that every attention kernel uses TMA.
- Arbitrary suspension at any instruction in an arbitrary kernel.
- Writes, writeback, and cross-tier coherence in the first implementation.

## 4. Terminology

**Request context**
: Live request identity and policy installed by userspace.

**External object**
: An immutable, versioned byte or tensor object with one or more physical
  replicas.

ABI v7 registers each staged directory entry as one acquisition tile. Direct
sources may serve subranges, but a staged miss transfers the complete entry;
this makes duplicate suppression exact without a range-readiness bitmap.

**Acquisition site**
: A compiler-recognized point where logical GPU work requires an external object
  range. The native operation may be a load, `cp.async`, or TMA.

**Direct source**
: A source represented by a GPU-loadable address.

**Transport source**
: A source requiring RDMA, NVMe, or another command protocol.

**Continuation**
: Reconstructible logical work that becomes runnable after all dependencies are
  ready.

**Nonresident acquisition**
: An acquisition whose logical future may outlive the issuing CTA.

## 5. Core data model

The exact binary layout is an implementation detail. The semantic fields are
part of the design contract.

```text
RequestContext {
    request_id
    generation
    tenant_id
    priority
    deadline
    cancelled
    outstanding_bytes
    max_outstanding_bytes
}
```

```text
ExternalObject {
    object_id
    version
    size
    shape
    strides
    element_type
    replicas[]
}
```

```text
Replica {
    kind: HBM | HOST | PEER | RDMA | NVME
    capability: DIRECT | TRANSPORT
    address_or_endpoint
    access_key
    estimated_latency
    estimated_bandwidth
}
```

```text
AcquireIntent {
    request_id
    generation
    continuation_id
    object_id
    object_version
    offset
    bytes
    destination_slot
    deadline
    allowed_sources
}
```

```text
Continuation {
    continuation_id
    request_id
    generation
    logical_tile
    dependency_count
    state
}
```

The implementation uses fixed-size, preallocated tables and an object-keyed
reusable intent pool. It
must not allocate memory on the device hot path.

## 6. Execution model

### 6.1 Acquisition result

A compiler-generated acquisition has three outcomes:

```text
DIRECT(pointer)
PENDING(token)
FAILED(error)
```

The state machine is:

```text
NEW -> QUEUED -> ISSUED -> READY
  |        |        |
  +--------+--------+-> FAILED

Any nonterminal state -> CANCELLED on generation mismatch or cancellation.
```

### 6.2 Direct path

For HBM or another GPU-loadable source:

```text
resolve source
    -> preserve or redirect the native acquisition operation
    -> consume in the current CTA
```

Backend-specific lowering:

- plain or vector load: redirect the source pointer;
- `cp.async`: redirect its global source operand; and
- TMA: replace or select the tensor-map global base address.

An HBM hit must not enter a global intent queue or perform a global atomic.

### 6.3 Deferred path

For RDMA, NVMe, or another command source:

```text
derive request and object
    -> coalesce duplicate intents
    -> reserve request and backend credit
    -> submit or enqueue transport work
    -> publish continuation
    -> end/defer the logical tile
```

The transport writes into a registered HBM staging slot. Completion processing:

```text
validate completion
    -> validate request generation and object version
    -> publish destination visibility
    -> recycle transport and staging credits
    -> decrement continuation dependency count
    -> enqueue continuation when all dependencies are ready
```

A later finite kernel invocation receives the ready continuation and executes
the original native load, `cp.async`, or TMA path from the HBM staging address.

### 6.4 No arbitrary live-state capture

Version 1 only permits deferral at a compiler-proven safe boundary:

- before non-idempotent side effects;
- before shared-memory state must survive;
- before a required CTA barrier;
- with request and logical tile identity sufficient to reconstruct work; and
- with duplicate issue prevented by token state.

If no safe boundary exists, the pass must decline the transformation. It must
not guess.

## 7. Compiler design

The compiler is responsible for semantics and placement, not privileged device
initialization.

### 7.1 Phase A: request binding

Recover or consume metadata for:

```text
logical tile -> request slot -> request generation
```

Initial integrations may provide the live mapping from the serving plan. The
pass must not claim to infer high-level request identity from arbitrary pointer
arithmetic.

### 7.2 Phase B: acquisition-site analysis

Recognize:

- marked external-backed pointer arguments;
- loads whose address cones derive from an external object table;
- inline PTX `cp.async` global-to-shared operations; and
- TMA operations or descriptors when visible through IR or frontend metadata.

The clean implementation currently consumes explicit byte-address or tensor-map
markers and rejects sites without a dominating request binding or canonical
null/defer/return edge. It does not yet infer arbitrary production load,
`cp.async`, or TMA address cones. The previous branch's recognition experiment
is not evidence for this branch.

### 7.3 Phase C: object-key derivation

Derive:

```text
object_id, byte range, logical tile, request, generation
```

Object identity must come from an explicit external-object contract or a
structurally verified address cone. An opaque pointer alone is insufficient.

### 7.4 Phase D: continuation legality

Prove:

- uniform control for the transformed acquisition;
- no divergent exit across a CTA barrier;
- no required shared or register state crosses deferral;
- idempotent re-entry;
- reconstructible logical work; and
- generation-safe batch-slot reuse.

### 7.5 Phase E: lowering

Lower explicit frontend markers to a small internal acquisition IR using
ordinary function calls so this project does not require an LLVM fork:

```llvm
declare ptr @nta_acquire_slow(...)
declare ptr @nta_acquire_tensor_map_slow(...)
declare i1  @nta_request_live(...)
declare void @nta_defer(...)
```

A backend-lowering pass or linked device bitcode specializes these operations.
LLVM optimization may inline the backend fast paths.

## 8. Runtime architecture

### 8.1 Host bootstrap

The host runtime may:

- allocate and pin device tables and staging pools;
- export or import HBM buffers through DMA-BUF;
- register HBM with NIC and NVMe drivers;
- create RDMA QPs and exchange remote keys;
- create NVMe queue pairs and map doorbells;
- install external-object replicas; and
- publish request contexts.

This work is initialization, not per-I/O execution.

### 8.2 Device directory

The device-visible directory contains compact, read-mostly placement entries.
Frequently changing readiness and credit state is stored separately to avoid
invalidating placement metadata.

### 8.3 Backend interface

Each backend implements:

```text
resolve(intent) -> DIRECT | NEED_SUBMIT | FAILED
try_submit(intent) -> token | NO_CREDIT | FAILED
progress_once(budget) -> completion_count
cancel_generation(request, generation)
```

The interface is common. Queue formats, memory ordering, and doorbells remain
backend-specific.

## 9. Bounded progress

No persistent service is permitted.

The host-staging implementation captures discover, progress, and resume as
three finite CUDA graph nodes. Each progress CTA owns at most one published
intent and exits after copying one finite object tile.

The NVMe implementation uses repeated, statically bounded progress/resume nodes
inside one finite graph launch. An NVMe progress invocation checks at most a
completion budget, submits at most an issue budget, batches each doorbell, and
returns immediately when the next CQ phase is absent. It never waits for a new
completion. This turns external latency into graph-level continuation rather
than CTA residency.

A future transport integration may instead let one warp at a
compiler-selected entry or exit point acquire a short-lived progress lease.
The winner would perform bounded work:

```text
drain at most B_rdma completions
drain at most B_nvme completions
advance at most B_host staging chunks
submit at most B_issue intents
```

Required properties:

- no loop waiting for new work;
- no spin waiting for a credit;
- no lock held across asynchronous work;
- no CTA-wide barrier added to a compute path;
- bounded instruction and memory-operation count; and
- a deferred intent remains valid for a future helper.

If no useful kernel is running, hardware cannot generally launch a CUDA kernel
from an NVMe completion. An interrupt or timer may launch one finite drain
kernel. The claim is CPU-free loaded-path progress, not zero CPU under idle
conditions.

## 10. Request-aware admission

The current implementation carries request generation, tenant, priority, and
deadline into every intent. Request, tenant, and backend byte credits are
reserved without waiting and rolled back on admission failure. NVMe scans the
bounded pool and chooses by priority/deadline urgency, then weighted tenant
service, before submission. Host
staging launches independent finite copy CTAs and enforces byte isolation, but
does not yet impose global priority order because those copies can run
concurrently. Remaining policy work is:

- fixed priority or slack buckets rather than a device heap;
- long-horizon aging across urgency classes;
- duplicate-object coalescing; and
- generation-safe cancellation.

The scheduler chooses among replicas using a bounded estimate:

```text
predicted_ready =
    queue_delay
  + setup_cost
  + bytes / expected_bandwidth
  + staging_cost
```

The unified scheduler is not a claim of uniform transport behavior.

## 11. Primary KV path

### 11.1 Why KV is primary

External KV is:

- request-owned;
- continuously growing;
- variable in size;
- SLO-critical for TTFT and TPOT;
- commonly tiered across HBM, CPU memory, remote caches, and SSD; and
- cancelled or replaced with continuous-batch churn.

### 11.2 Dense attention

Userspace often knows the dense KV block list. The compiler contribution is not
discovering that list. It is binding live request semantics to the kernel's
actual tile-consumption path and enabling direct or deferred execution without
rewriting the kernel.

For split-K attention:

```text
ready KV chunks -> compute partial states
missing chunks  -> acquire and defer
all chunks ready -> deterministic final reduction
```

Dense attention cannot finish without all required exact KV. The system may make
partial progress but must not claim otherwise.

### 11.3 Sparse or conditional attention

When GPU-side indexing selects KV blocks, the exact object set is not known to
userspace before execution. This is the strongest semantic-gap case within
attention and should be included if a production kernel is available.

## 12. Secondary applications

### 12.1 MoE experts

The binding is:

```text
request -> token -> routed expert -> expert tensor tile
```

Expert GEMM kernels are naturally tiled and may use TMA on supported
architectures. The system coalesces expert acquisitions while retaining the
request dependencies and deadlines of all dependent tokens.

### 12.2 Graph and ANNS

ANNS vectors can use tensor-like tiled staging. Graph adjacency pages are more
likely to use ordinary loads. These applications validate the generic
object-range and continuation machinery, not the TMA-specific claim.

## 13. TMA position

The top-level claim must not say that all fused attention kernels use TMA.

TMA is one native lowering:

```text
native acquisition = load | cp.async | TMA
```

Known boundaries:

- direct host-memory TMA has prior art;
- peer-reachable TMA has library support;
- IB/RoCE and NVMe are command paths, not TMA-addressable sources; and
- a TMA destination is current-CTA shared memory and cannot survive CTA exit.

The candidate contribution is cross-CTA virtualization of the logical
acquisition, not extension of TMA hardware.

## 14. DMA-BUF position

DMA-BUF is kernel plumbing for sharing an allocated buffer among device drivers.
It may register the HBM staging pool with NIC or NVMe drivers.

M4 uses exactly this boundary: CUDA exports an HBM allocation, the bootstrap
driver attaches and maps it in the NVMe device DMA domain, and the runtime
publishes the resulting per-page DMA addresses. Queue setup and mapping occur
once. No DMA-BUF operation occurs in a GPU kernel or per transfer.

DMA-BUF does not provide:

- request or tensor identity;
- source-tier selection;
- RDMA or NVMe command submission;
- device-side admission;
- continuation scheduling; or
- compiler transformation.

Per-transfer attach/map operations may sleep and must not occur on the GPU hot
path. Registration is performed ahead of time.

## 15. Correctness invariants

1. A completion is accepted only when request generation and object version
   match the intent.
2. A staging slot is not reused until transport completion and all consumers
   release it.
3. Submission publication follows backend-required release ordering before a
   doorbell write.
4. Completion consumption establishes visibility before READY publication.
5. Transport queue exhaustion causes deferral, never spinning while holding
   resources; bootstrap requires one intent slot per independently queued
   object so the intent pool itself cannot exhaust under valid publication.
6. Duplicate intents share a transfer only when object version and byte range
   match.
7. Cancellation cannot make a stale continuation runnable.
8. The pass declines transformations that cannot preserve barrier convergence.
9. The HBM/direct path preserves the original kernel's numerical behavior.
10. Partial attention reduction uses a defined deterministic order when
    bit-exactness is required.

## 16. Efficiency invariants

1. HBM hits perform no global queue operation or global atomic.
2. A warp or CTA emits at most one intent for an identical object range.
3. Device metadata is fixed-size and preallocated.
4. Doorbells are batched when the backend permits it.
5. Progress work has a hard budget.
6. The compiler adds no polling loop.
7. A miss does not retain shared memory or registers across external latency.
8. Transport-specific fast paths remain independently tunable.

## 17. Prior-art boundary

The following are not contributions:

- a unified object or I/O descriptor;
- ABI preservation or fixed virtual addresses;
- GPU-initiated RDMA or NVMe by itself;
- TMA access to mapped host memory;
- TMA-assisted peer communication;
- KV placement and caching;
- request priority ordering;
- compiler-inserted prefetch by itself; or
- DMA-BUF registration.

The candidate contribution is the combination:

> automatic binding of live request semantics to acquisition sites in existing
> finite kernels, direct-path preservation for addressable data, and
> generation-safe suspend/resume of transport-backed logical work across CTA
> lifetimes with bounded nonresident progress.

Required comparison points include:

- Syncopate: compiler-generated chunk-level compute/communication overlap;
- Strata: production hierarchical KV, GPU-assisted I/O, and cache-aware
  scheduling;
- DirectKV: zero-copy CPU-resident KV with fused warp-level pipelines;
- CoPilotIO: GPU storage submission with CPU completion;
- BaM and AGILE: GPU-initiated storage and background GPU progress;
- GORIO and GNStor: GPU-owned remote I/O and NVMe-over-RDMA;
- GIN, GICC, and NVSHMEM: GPU-initiated remote communication;
- DAK: direct host-memory TMA in custom offload kernels; and
- VTC: compiler-managed virtual tensors for movement elimination.

Detailed boundaries are maintained in `docs/RELATED_WORK.md`. The
implementation must not claim "first" until that audit is updated at submission
time.

## 18. Evaluation

### 18.1 Baselines

- untouched production kernel with HBM-resident data;
- CPU/runtime prefetch into HBM;
- direct mapped-host access;
- synchronous GPU-initiated transport;
- dedicated GPU progress service;
- CPU completion service; and
- manually modified application/kernel using the same backend.

### 18.2 Metrics

- TTFT, TPOT, request p50/p99, and SLO attainment;
- useful serving goodput;
- I/O completion and ready-continuation latency;
- CPU cores and CPU time;
- SM occupancy and progress tax;
- register and shared-memory changes;
- HBM staging footprint and bandwidth;
- RDMA/NVMe queue depth and doorbell rate;
- duplicate and wasted I/O;
- cancellation latency;
- fairness and starvation;
- direct-path overhead; and
- idle-fallback frequency.

### 18.3 Required ablations

- request semantics removed;
- continuation scheduling removed;
- direct versus staged host path;
- bounded helper versus persistent service;
- backend batching disabled;
- object coalescing disabled; and
- compiler transformation versus handwritten integration.

## 19. Kill criteria

Stop or reframe the project if any of the following holds:

1. Runtime prefetch matches the transformed system on end-to-end SLO metrics.
2. Direct-path overhead is measurable on HBM-resident production kernels.
3. Safe deferral requires kernel-specific source rewrites rather than a
   repeatable compiler analysis.
4. Relaunch and continuation overhead exceeds saved waiting time.
5. Useful kernel cadence is insufficient and fallback handles most progress.
6. A dedicated service warp is cheaper than bounded distributed progress.
7. External KV misses are too rare in the target serving configuration.
8. The available PCIe topology prevents useful GPU-device peer DMA.
9. Results come from a new backend rather than the request-semantic compiler
   mechanism.

## 20. Implementation sequence

### M0: clean compiler skeleton

- LLVM new-PM plugin.
- Acquisition-site metadata contract.
- IR-only tests for request/object binding.
- No runtime or hardware backend.

Gate: the pass recognizes only explicit fixtures and declines unknown forms.

Status: complete. The pass binds live per-CTA request SSA values, proves the
canonical null/defer/return region, emits direct and slow paths, and leaves
rejected sites untouched with a diagnostic.

### M1: deterministic mock backend

- Device object directory.
- Fixed intent and continuation pools.
- Direct, pending, failed, and cancelled transitions.
- Deterministic completion injection.

Gate: exhaustive generation, cancellation, duplicate, and queue-full tests.

Status: the state machine runs against real CUDA allocations rather than a
disconnected mock. Generation, cancellation, and duplicate transfer behavior
are tested; exhaustive queue-wrap testing remains open.

### M2: finite progress protocol

- Short-lived helper lease.
- Hard issue and completion budgets.
- No-wait credit handling.
- Ready-continuation queue.

Gate: no deadlock at ring wraparound or full queue; bounded instruction count.

Status: complete across repeated epochs. Misses publish to a reusable
object-keyed pool, object CAS suppresses duplicates, and a finite progress grid
precedes ready publication and a finite ready-only resume grid. No kernel polls
or persists.

### M3: CPU DRAM

- Mapped-host direct path.
- Optional HBM staging path.
- Path selection and visibility tests.

Gate: HBM fast path remains statistically indistinguishable from stock.

Status: functionally complete. Mapped direct and GPU-staged paths are
numerically tested. The compiler-generated HBM path has no queue or atomic
instruction; a production-kernel zero-overhead comparison remains evaluation
work rather than an implementation claim.

### M4: NVMe

- Minimal queue setup through a dedicated driver/runtime component.
- GPU command construction and batched submission.
- CQ processing into the common continuation state.

Gate: finite producer/consumer kernels sustain queue progress without a
persistent poller.

Status: complete for a dedicated read-only experiment. The bootstrap driver
resets and identifies the controller, owns one depth-64 queue pair, imports HBM
DMA-BUFs or pins mapped host destinations, and maps coherent queue memory plus
the MMIO doorbell page into CUDA. A bounded device function constructs NVMe
READ SQEs and PRP lists, batches the SQ doorbell, consumes phase-tagged CQEs,
validates object/request generations, and publishes ready continuations. The
benchmark performs no CPU command submission or completion polling.

The tested program emits only READ commands, but the raw queue is exposed to a
trusted privileged process and is not a hardware-enforced read-only interface.

### M5: KV integration

- Existing paged-attention request/tile binding rebuilt cleanly.
- Ready-only tile scheduling.
- Dense split-K or sparse-KV continuation fixture.
- End-to-end SLO experiment.

Gate: improvement over CPU/runtime prefetch using the same data path.

Status: mechanism workload complete, serving gate open. The branch runs a real
FP16, head-dimension-128, page-size-16 split-K attention kernel with
heterogeneous pages per request, stable partial softmax reduction, ready-only
continuations, and a CPU numerical reference. Work formation consumes
FlashInfer's public paged-KV CSR representation, reduction uses FlashInfer's
base-2 `(V, LSE)` state implementation when its headers are available, and a
real FlashInfer decode wrapper is a differential correctness gate. NTA deferral
is not yet inside FlashInfer's optimized CTA, nor is it wired into SGLang/vLLM
request lifecycle and KV management; therefore there is no TTFT/TPOT/SLO claim.

### M6: TMA specialization

- Census actual production IR/PTX forms.
- Recognize or consume metadata for tensor maps.
- Direct descriptor rebind.
- Resume from HBM staging using the original TMA path.

Gate: no claim based solely on a synthetic TMA kernel.

Status: mechanism complete, production census gate open. A distinct compiler
marker preserves direct tensor-map descriptors or selects an HBM staging
descriptor after readiness. The attention CTA initializes its barrier only
after acquisition succeeds, executes `cp.async.bulk.tensor`, and is covered by
memcheck/racecheck/synccheck. Production attention IR recognition and an
untouched-kernel comparison remain required.

### M7: RDMA

- Run on a Mellanox/IBGDA-capable testbed.
- GPU submission and bounded CQ processing.
- Compare with CPU proxy and dedicated GPU progress.

Gate: real network hardware, not loopback or emulation, for performance claims.

### M8: generality

- MoE expert acquisition.
- Optional ANNS or graph object acquisition.

Gate: reuse the same compiler/runtime contracts without kernel-name-specific
logic.

## 21. Target repository layout

The implemented tree is:

```text
CMakeLists.txt
docs/
    ARCHITECTURE.md
    FLASHINFER.md
include/nta/
    AcquireIR.h
    DeviceAPI.cuh
    FlashInferAdapter.h
    HostRuntime.h
    RuntimeABI.h
    Passes.h
lib/
    AcquireAnalysis.cpp
    AcquireLowering.cpp
    ContinuationLowering.cpp
    Plugin.cpp
runtime/
    device/Acquire.cuh
    host/FlashInferAdapter.cpp
    host/Runtime.cpp
tests/
    ir/{batched,reject-*}.ll
    flashinfer/differential_decode.py
    runtime/{AbiTest,FlashInferAdapterTest,RuntimeTest}.cpp
benchmarks/
    kv/{KvAcquire,KvAcquireKernel,KvTypes}
    attention/{PagedAttention,PagedAttentionKernel,PagedAttentionTypes}
    nvme/NvmeRead.cpp
```

No legacy scheduler, CLC, grouped-LPT, cache-hint, or timing implementation is
copied into this branch. A mechanism may be reintroduced only when a milestone
requires it and its tests are rebuilt around this architecture.
