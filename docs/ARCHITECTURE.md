# Nonresident Acquisition Architecture

Status: implementation contract

This document defines the problem, claims, system boundary, execution model, and
implementation gates for the clean `nonresident-acquisition` branch. Decisions
in this document take precedence over the previous prototype's design notes.
`SYSTEM_PLAN.md` governs the target system motivation and implementation order;
this document governs implemented mechanism invariants and status.

Implementation status (2026-08-09): M0-M3 have a working vertical slice tested
on an NVIDIA RTX PRO 6000 Blackwell Server Edition. M5 has a numerically checked
split-K paged-attention mechanism workload and version-checked JIT hooks
executed in FlashInfer 0.6.12 decode and FA2 paged-prefill kernels. This is not
yet a production serving result; SGLang 0.5.14 HiCache and decode graph replay
are locally integrated. M6 has real TMA descriptor selection and
hardware TMA after direct or externally staged acquisition; automatic
production-IR recognition remains open. M4 has a VFIO PCI cdev/IOMMUFD
bootstrap behind the same queue ABI, an exact NVIDIA BAR-VMA GPU-write gate,
and a controller-free tested CTA-side NVMe try-issue path with bounded scheduled
fallback. ABI-v18 hardware qualification completed 32,000 compulsory-miss,
checksum-verified reads through an eight-pass graph on one CD8P controller. The
controller was restored to `nvmex` afterward. RNIC/RDMA is explicitly deferred because this host has no
RNIC, and no active backend is claimed. M8 has a real routed MoE matrix workload
on the common mechanism; production MoE baselines remain open. A 10,000-epoch
runtime/graph lifecycle stress passes; the two-device ownership test is present
but skips on this one-GPU host.

The current end-to-end SGLang/FlashInfer dense result is exact and
fallback-free but negative: the ABI-v23 v10 2K mixed point has 0.977x stock
throughput and 1.012x resident P99 inter-token latency. A five-trial
acquisition-admission ablation was worse and has been removed. Separately, the
canonical FlashInfer bounded-HBM operator now beats atomic CPU-DRAM promotion
by 1.1714x with a paired-bootstrap 95% interval of [1.1660x, 1.1732x] while
using 4x less staging HBM; a heterogeneous-shape run improves by 1.1100x with
4.83x less staging. `TIER_STREAMING.md` defines that result and its claim
boundary. The architecture therefore has a positive operator mechanism and an
open serving integration hypothesis, not a production or OSDI result.
Generated FlashInfer modules now export a versioned family/form/capability
contract plus a typed request-coordinate/partial-state/reduction plan that eager
and graph consumers validate before launch. Paired ragged-prefill direct and
incremental forms are implemented; arbitrary-kernel recognition and the paged
SGLang operator remain open.

ABI v25 defines one engine-neutral device work item, fixed-capacity dependency
segments, completion-driven reverse dependency edges, and a compact runnable
work queue. The completion CTA that satisfies the final dependency performs the
single `Pending -> Ready` transition and appends the ticket; a bounded
publication kernel remains as a compatibility and diagnostic drain. One finite
CTA work ticket can acquire several mixed-tier objects and is published only
after every object identity and version match and every dependency reaches the
literal `Ready` state. Reusable device plans support
pinned, asynchronous updates. Generic compute, paged attention, TMA attention,
and MoE consume this same ABI; FlashInfer is an optional metadata adapter rather
than the core execution model.

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
    -> runnable work ticket
```

We will test the following research hypothesis:

> A compiler can turn an all-or-nothing batched GPU kernel into an incrementally
> executable operator whose runnable request/tile subsets can be co-scheduled
> with data arrival and later request admission across memory tiers, without
> persistent GPU workers or loss of the native complete-data path.

Dense external KV for online serving is the mandatory no-regression and
batch-barrier stress workload. Canonical FlashInfer now demonstrates a dense
CPU-DRAM crossover by preserving `(V, LSE)` partials while bounded staging
overlaps transfer and computation. The remaining flagship gate is to generate
and use that form in real model execution and beat equal-state serving
baselines. The mechanism is designed for request-conditioned
external tensors and object ranges rather than for KV alone. The complete
co-design and evaluation roadmap is in `SYSTEM_PLAN.md`.

## 2. Problem

### 2.1 Two-sided semantic blind spot

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
work tickets, and stale work survives cancellation or batch-slot reuse.

Neither side can reconstruct the missing half from pointers or launch order.
The compiler hook therefore binds userspace lifecycle state to the kernel's
logical tile at the last reconstructible pre-state boundary. The runtime keeps
that identity across an asynchronous transport and maps a physical CTA whose
data became available back to the original logical work. Request policy can
then govern admission while tile availability governs incremental execution,
without making either side infer the other's state.

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

The system therefore needs a work ticket that survives the issuing CTA.

### 2.4 Granularity discontinuity

One I/O and one launch per tile exposes partial computation but wastes bandwidth
for dense, contiguous demand. Waiting for a complete layer preserves transfer
efficiency but creates an all-or-nothing kernel barrier under arrival skew.

The target system continuously groups unavailable object ranges while the saved
transfer and command cost exceeds the request delay and useful compute being
postponed. Direct execution, bulk transfer, fine-grained acquisition, and
whole-request delay are limiting outcomes of this one incremental scheduler,
not independent production policies.

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

ABI v25 registers each staged directory entry as one acquisition tile. Direct
sources may serve subranges, but a staged miss transfers the complete entry;
this makes duplicate suppression exact without a range-availability bitmap.

**Acquisition site**
: A compiler-recognized point where logical GPU work requires an external object
  range. The native operation may be a load, `cp.async`, or TMA.

**Direct source**
: A source represented by a GPU-loadable address.

**Transport source**
: A source requiring RDMA, NVMe, or another command protocol.

**Work ticket**
: Reconstructible logical work that becomes runnable after all dependencies are
  ready.

**Runnable tile**
: One current-generation logical tile whose complete dependency set is
  available. The ABI represents this condition with work-ticket state `Ready`.

**Incremental operator**
: A compiled operator that can execute valid runnable-tile subsets over several
  finite launches and merge only complete partial results.

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
    work ticket_id
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
Work ticket {
    work ticket_id
    request_id
    generation
    logical_tile
    dependency_count
    state
}
```

```text
WorkItem {
    request_slot
    generation
    logical_work
    dependency_begin
    dependency_count
    direct_dependency_count
    work ticket_id
}

AcquireRequirement {
    direct_address_or_zero
    direct_tensor_map_or_zero
    object_id
    object_version
    object_slot
    offset
    bytes
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

### 6.3 Incremental execution cycle

The target runtime operates one bounded cycle:

```text
consume completed transfers
    -> update dependent object and tile counters
    -> append newly runnable logical tiles
    -> group or issue unavailable object ranges by request impact
    -> launch compact runnable work
    -> merge requests with every contributor complete
    -> report partial progress to the engine scheduler
```

The grouping objective uses contiguous bytes, duplicate fan-in, predicted queue
and transfer time, request SLO slack, delayed useful compute, and measured
launch/work-ticket cost. Forced bulk, fine-grained, and whole-request-delay
controls exist for evaluation only. `SYSTEM_PLAN.md` defines the complete target;
the current implementation covers the direct and unavailable-data mechanisms,
not the unified scheduler or engine feedback loop.

### 6.4 Unavailable-data path

For RDMA, NVMe, or another command source:

```text
derive request and object
    -> coalesce duplicate intents
    -> reserve request and backend credit
    -> submit or enqueue transport work
    -> publish work ticket
    -> end/defer the logical tile
```

The transport writes into a registered HBM staging slot. Completion processing:

```text
validate completion
    -> validate request generation and object version
    -> publish destination visibility
    -> recycle transport and staging credits
    -> consult pending work tickets and scan each bounded dependency segment
    -> enqueue the work ticket only when every dependency is ready
```

A later finite kernel invocation receives a work ticket in the literal `Ready`
state and executes the original native load, `cp.async`, or TMA path from the
HBM staging address.

### 6.5 CTA-side try-issue

For an enabled command backend, the application CTA leader may attempt one
non-spinning submission before publishing an intent:

```text
external miss
    -> claim and fully publish a generation-safe work ticket
    -> count backend demand before attempting transport ownership
    -> if this is the oldest demand and a queue lease is immediately available:
         reserve credits, construct command, ring doorbell
       else:
         publish request-aware intent
    -> return the whole CTA
```

The queue lease protects only command construction and publication. It is
released before the CTA returns and is never held across device I/O. The CTA
does not inspect completion state. A separate finite progress invocation drains
the CQ, releases credits, and publishes data availability. ABI v25 implements this for
NVMe; RDMA remains inactive until a real backend and RNIC testbed exist.

The fast path is intentionally opportunistic. Backend demand is atomically
counted before the lease attempt, closing the reservation-to-publication window:
only the oldest pending acquisition may issue directly. Concurrent demand,
queue ownership contention, full SQ/CID state, or failed admission all fall
back immediately. Backend-local counters prevent unrelated host staging work
from suppressing NVMe submission while preserving the global request-aware
scheduler for contested NVMe work. Direct construction is capped at 32
destination PRP pages; larger transfers retain warp-cooperative PRP construction
and batched doorbells in the scheduled progress kernel.

For scheduler-selected indexed CPU-DRAM objects, host progress is also bounded
but uses enough transfer parallelism for serving-sized KV rows. One host call
enqueues three finite kernels on the progress stream:

```text
claim + validate intents and byte credits
    -> copy each object with 16 row-specialized CTAs
    -> publish object transitions and release credits
```

The split is intentional. It prevents any object or work ticket from becoming
available before every copy CTA has retired, while avoiding a grid barrier,
device-side allocation, polling CTA, or persistent kernel. The optimized range
path requires the scheduler's contiguous object interval; arbitrary queued host
work retains the urgency-queue progress kernel.

### 6.6 No arbitrary live-state capture

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

The clean implementation currently consumes explicit byte-address, tensor-map,
or bounded dependency-set markers and rejects sites without a dominating
request binding or canonical ready/defer/return edge. Set lowering is independent
of engine and kernel names. It does not yet infer arbitrary production load,
`cp.async`, or TMA address cones. The previous branch's recognition experiment
is not evidence for this branch.

FlashInfer is the first typed frontend: a version- and source-hash-checked C++
template overlay retains its scheduler request/tile coordinates, inserts the
work hook before shared/variant state initialization, and threads request-local
merge groups into cascade reduction. The LLVM pass remains the backend legality
verifier. Paired direct and preacquired-partial ragged-prefill forms now export
one typed execution plan. A second generated-kernel frontend and automatic
recognition of arbitrary production kernels remain open.

### 7.3 Phase C: object-key derivation

Derive:

```text
object_id, byte range, logical tile, request, generation
```

Object identity must come from an explicit external-object contract or a
structurally verified address cone. An opaque pointer alone is insufficient.

### 7.4 Phase D: work ticket legality

Required legality conditions are:

- uniform control for the transformed acquisition;
- no divergent exit across a CTA barrier;
- no required shared or register state crosses deferral;
- idempotent re-entry;
- reconstructible logical work; and
- generation-safe batch-slot reuse.

The current pass proves domination, marker ABI, a bounded acyclic pending edge,
exactly one matching defer, return from the finite kernel, and no value/state
use on that edge. It also performs target-aware CTA-uniformity validation:
markers must be inlined into a GPU kernel entry; kernel arguments, block
identity, and block/grid dimensions are collective; and thread/lane/warp
identity, atomics, volatile loads, local allocation, and unknown calls taint
control or operands as non-collective. Automatic discovery of unmarked
production address cones remains an open production gate.

Ordinary plan and catalog loads used as collective marker operands are uniform
under the typed frontend contract that those allocations are immutable for the
finite launch. The engine establishes that property through stream/graph
ordering. The verifier rejects volatile and atomic loads and visible
thread-derived operands, but LLVM IR alone cannot prove that an unrelated CTA
never writes a plain global address; unsupported frontends must not assert this
contract without equivalent ownership.

The ABI-v25 IR suite includes a positive device-selected-object case whose slot,
identity, version, range, and direct pointer are loaded from a catalog selected
by CTA identity. The pass lowers that case and tags it `split-phase-cta` while
the existing thread/lane-derived fixtures remain rejected. This establishes
support for explicit GPU-selected object semantics, not automatic discovery of
an unmarked catalog.

For JIT-generated source, the pass also registers at Clang's optimizer-last
extension point. An nvcc-compatible shim translates the generator command,
loads the pass, and isolates artifacts in a cache fingerprinted by ABI and
compiler integration content. This removes a custom offline compilation step;
it does not manufacture missing request/object semantics or a safe deferral
point in an arbitrary kernel.

### 7.5 Phase E: lowering

Lower explicit frontend markers to a small internal acquisition IR using
ordinary function calls so this project does not require an LLVM fork:

```llvm
declare ptr @nta_acquire_slow(...)
declare ptr @nta_acquire_tensor_map_slow(...)
declare i1  @nta_acquire_set_slow(...)
declare i1  @nta_request_live(...)
declare void @nta_defer(...)
```

A backend-lowering pass or linked device bitcode specializes these operations.
LLVM optimization may inline the backend fast paths.

## 8. Runtime architecture

### 8.1 Host bootstrap

The host runtime may:

- allocate and pin device tables and staging pools;
- register HBM with a future validated NIC or peer-memory provider;
- create RDMA QPs and exchange remote keys;
- acquire a dedicated NVMe function through VFIO/IOMMUFD, create queue pairs,
  and map the isolated doorbell page;
- install external-object replicas; and
- publish request contexts.

This work is initialization, not per-I/O execution.

Production engines may non-owningly register existing HBM, mapped-host, and
staging allocations. They retain allocation and graph ownership. A reusable
finite-phase launcher enqueues reset, bounded backend progress, and runnable
work into the engine's stream or existing graph capture. Normal backend
progress publishes newly runnable tickets directly, without a separate launch.

### 8.2 Device directory

The device-visible directory contains compact, read-mostly placement entries.
Frequently changing availability and credit state is stored separately to avoid
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
intent, copies one finite object tile, publishes any newly runnable dependent,
and exits.

The NVMe implementation uses repeated, statically bounded progress/resume nodes
inside one finite graph launch. An NVMe progress invocation checks at most a
completion budget, submits at most an issue budget, batches each doorbell, and
returns immediately when the next CQ phase is absent. It never waits for a new
completion. This turns external latency into graph-level work ticket rather
than CTA residency.

The current graph topology is deliberately fixed and bounded. CUDA conditional
`IF`/`WHILE` nodes are a possible way to skip empty rounds, but CUDA forbids
device graph launch from kernels in a conditional-node body. A future adaptive
graph must choose a compatible control strategy; it cannot combine those two
features as if they were independently composable. See the
[CUDA Graphs programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).

ABI v25 also allows the missing application CTA to attempt one NVMe submission.
A backend-local queue lease serializes it with the finite progress kernel. The
CTA rings at most one doorbell, never waits for the lease or completion, and
publishes an intent on any recoverable failure. This is a latency path, not a
replacement for batched request-aware submission.

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
reserved without waiting and rolled back on admission failure. ABI v25 accounts
pending, executable, completed, and expected compiler-attributed compute plus
unavailable bytes for each request generation. Checked subtraction makes
counter underflow fail closed; rejected same-generation epoch attribution is
observable, while stale generations do not modify replacement request state.
NVMe consumes tagged per-backend
urgency buckets, skipping stale heads without a normal-path capacity scan. On
insertion and requeue, deadline urgency includes live backend outstanding bytes,
estimated latency/bandwidth, and current request compute. Without an explicit
deadline, equal-priority requests receive only a bounded shortest-critical-work
preference; priority classes still dominate. CTA count alone is not a policy
signal. Weighted tenant service is accounted but not yet enforced as
an ordering rule within one bucket. CTA try-issue is opt-in and permitted only
when that backend has no older queued intent; otherwise the same scheduler
orders it. Host
staging launches independent finite copy CTAs and enforces byte isolation, but
does not yet impose global priority order because those copies can run
concurrently. Remaining policy work is:

- fixed priority or slack buckets rather than a device heap;
- long-horizon aging across urgency classes;
- a measured threshold for enabling opportunistic CTA submission; and
- cross-device policy for replicated objects.

The scheduler chooses among replicas using a bounded estimate:

```text
predicted_ready =
    queue_delay
  + setup_cost
  + bytes / expected_bandwidth
  + staging_cost
```

The unified scheduler is not a claim of uniform transport behavior.

The device transport queue consumes this state on every insertion and requeue.
For a request, it estimates acquisition from current queue delay and bytes,
overlaps that with already executable compute, then adds compute still blocked
on data. Deadline slack and priority select the bucket from which transport
service is popped. This is a causal online estimate; future arrivals are not
inputs. A CUDA test proves live compiler-attributed compute changes I/O service
order for equal-priority requests. `CriticalWorkPlan` is the matching host
reference model. SGLang publication of request policy is implemented, while
using snapshots for later batch formation remains an open integration gate.

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

Dense serving is the primary applicability case. The incremental scheduler
should form large groups for homogeneous cold layers, dispatch the direct form
for resident work, expose partial execution under within-batch arrival skew,
and naturally delay requests with no useful runnable work. The contribution is
not discovery of a dense KV list; it is joining request policy, tile
availability, partial numerical progress, and acquisition cost without forcing
one granularity on every batch.

### 11.3 Sparse or conditional attention

When GPU-side indexing selects KV blocks, userspace cannot observe the exact
object set at batch formation without duplicating the selector or synchronizing
after the query is produced. This is the strongest semantic-gap case within
attention.

The implemented mechanism fixture materializes the query in an upstream device
kernel and then performs summary scoring, deterministic top-k page selection,
canonical dependency publication, acquisition, and attention in the same
finite attention CTA. On a miss, the CTA publishes a ticket and exits. The
ready CTA obtains the logical request from that ticket, recomputes the selector,
reloads the selected catalog entries, and consumes the acquired pages. No
selector local/shared state crosses the exit. The test deliberately permutes
compact request indices and runtime request slots and checks the preserved
`(epoch, request slot, generation, logical request, object, version)` identity.
This establishes mechanism feasibility; integration into a production sparse
attention kernel and end-to-end SLO evidence remain open.

The controlled fixture includes a cold-cache, overlapped GPU overfetch policy
that moves every candidate page while running the same device query producer,
selector, and attention math. It is intentionally retained even where it wins:
selective acquisition has a fixed ticket/progress cost and is justified only
after avoided transfer exceeds that cost.

## 12. Secondary applications

### 12.1 MoE experts

The binding is:

```text
request -> token -> routed expert -> expert tensor tile
```

Expert GEMM kernels are naturally tiled and may use TMA on supported
architectures. The system coalesces expert acquisitions while retaining the
request dependencies and deadlines of all dependent tokens.

The implemented mechanism workload computes hidden states and top-k routing on
the GPU. The router writes canonical `WorkItem` and `AcquireRequirement`
records; userspace publishes the expert catalog but does not know the selected
IDs before the consumer launch. This is the concrete device-generated-demand
case. See `docs/DEVICE_ROUTED_MOE.md`.

### 12.2 Graph and ANNS

ANNS vectors can use tensor-like tiled staging. Graph adjacency pages are more
likely to use ordinary loads. These applications validate the generic
object-range and work ticket machinery, not the TMA-specific claim.

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
It could be part of a future direct-HBM transport, but the current system does
not contain a DMA-BUF importer. NVMe destinations are mapped pinned DRAM inside
the private IOMMUFD IOAS because this host has no validated GPU/NVMe P2P route.
This removes dynamic-import invalidation, reservation-fence, and DMA-direction
ambiguity from the active implementation.

DMA-BUF does not provide:

- request or tensor identity;
- source-tier selection;
- RDMA or NVMe command submission;
- device-side admission;
- work ticket scheduling; or
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
   resources. The intent pool is bounded by the maximum active acquisition
   frontier rather than catalog size; unexpected overflow fails affected work
   explicitly instead of spinning or overwriting an intent.
6. Duplicate intents share a transfer only when object version and byte range
   match.
7. Cancellation cannot make a stale work ticket runnable.
8. The pass declines transformations that cannot preserve barrier convergence.
9. The HBM/direct path preserves the original kernel's numerical behavior.
10. Partial attention reduction uses a defined deterministic order when
    bit-exactness is required.
11. Work ticket ownership publishes `Initializing` before writing fields and
    transitions to `Pending` only after the dependency and pending-index records
    are visible; competing CTAs do not wait or issue against partial state.
12. A completion whose object identity/version or work ticket generation is
    stale retires credits and transport context without modifying replacement
    state.

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

> compiler and serving-runtime co-design that transforms an atomic finite GPU
> operator into request-scoped incremental execution, then jointly schedules
> arriving data, useful partial computation, and later request admission while
> preserving the original complete-data path.

Work tickets across CTA lifetimes implement unavailable-data handling. Elastic
coalescing preserves dense transfer efficiency. Neither is claimed to be the
contribution or to dominate independently.

Required comparison points include:

- Syncopate: compiler-generated chunk-level compute/communication overlap;
- Strata: production hierarchical KV, GPU-assisted I/O, and cache-aware
  scheduling;
- ECHO: lossless sparse-KV prefetch and fused recall/indexer overlap;
- DirectKV: zero-copy CPU-resident KV with fused warp-level pipelines;
- CoPilotIO: GPU storage submission with CPU completion;
- Tutti: asynchronous GPU-native KV object I/O and slack-aware scheduling;
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
- instrumented direct path with identical resident data;
- CPU/runtime prefetch into HBM;
- coalesced bulk acquisition;
- engine request skip and rebatch;
- forced fine-grained incremental execution;
- the unified incremental scheduler, a best-fixed whole-trace reference, and
  resettable decision-oracle experiments;
- direct mapped-host access;
- synchronous GPU-initiated transport;
- dedicated GPU progress service;
- CPU completion service; and
- manually modified application/kernel using the same backend.

### 18.2 Metrics

- TTFT, TPOT, request p50/p99, and SLO attainment;
- useful serving goodput;
- I/O completion to runnable-tile publication latency;
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
- work ticket scheduling removed;
- elastic grouping replaced by each forced endpoint;
- partial-progress feedback to the userspace batch scheduler removed;
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
4. Relaunch and work ticket overhead exceeds saved waiting time.
5. Useful kernel cadence is insufficient and fallback handles most progress.
6. A dedicated service warp is cheaper than bounded distributed progress.
7. External KV misses are too rare in the target serving configuration.
8. The available PCIe topology prevents useful GPU-device peer DMA.
9. Results come from a new backend rather than the request-semantic compiler
   mechanism.
10. The incremental scheduler has material regret in identical-snapshot decision
    replays, loses to the best-fixed policy on real traces, or wins only through
    unmatched initial cache state, admission, or input request order.

## 20. Implementation sequence

M0-M8 below record component implementation status. The research priority and
remaining end-to-end work are governed by P0-P7 in `SYSTEM_PLAN.md`; in
particular, incremental dense FlashInfer/SGLang execution precedes new transport
backends.

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
- Fixed intent and work ticket pools.
- Direct, pending, failed, and cancelled transitions.
- Deterministic completion injection.

Gate: exhaustive generation, cancellation, duplicate, and queue-full tests.

Status: the state machine runs against real CUDA allocations rather than a
disconnected mock. Generation, cancellation, duplicate transfer, full/fatal
queue retirement, and stale completion behavior are tested. The current
depth-64 hardware run crossed more than 25 queue wraps without error;
exhaustive randomized model checking remains open.

### M2: finite progress protocol

- Short-lived helper lease.
- Hard issue and completion budgets.
- No-wait credit handling.
- Runnable-tile queue.

Gate: no deadlock at ring wraparound or full queue; bounded instruction count.

Status: complete across repeated epochs. Misses publish to a reusable
object-keyed pool, object CAS suppresses duplicates, and a finite progress grid
precedes data-availability publication and a finite runnable-work grid. No
kernel polls or persists.

### M3: CPU DRAM

- Mapped-host direct path.
- Optional HBM staging path.
- Path selection and visibility tests.

Gate: HBM fast path remains statistically indistinguishable from stock.

Status: functionally complete. Mapped direct and GPU-staged paths are
numerically tested through both runtime-owned allocations and non-owning
registration of engine-managed allocations. Staged acquisition keeps a
16-byte vector path and safely handles arbitrary external alignment. The
runtime enforces a hard byte capacity and records high-water use for staging
allocations it owns; engine-managed staging remains under the engine cache's
eviction policy. The
compiler-generated HBM path has no queue or atomic instruction; a
production-kernel zero-overhead comparison remains evaluation work rather than
an implementation claim.

### M4: NVMe

- Minimal queue setup through a VFIO/IOMMUFD userspace control plane.
- GPU command construction and batched submission.
- CQ processing into the common work ticket state.

Gate: finite producer/consumer kernels sustain queue progress without a
persistent poller.

Status: software mechanism complete; historical ABI-v18 VFIO trusted-mode
hardware qualification passed on the local GPU/CD8P system. A compulsory-miss run issued
and completed all 32,000 measured 64-KiB reads with zero checksum failure at
1,624.42 MiB/s physical throughput. This one-controller point is a correctness and
scaling regression, not production portability or competitive-bandwidth proof.
The control plane resets and
identifies the controller, creates a private translated IOAS, negotiates one
configurable queue, maps host destinations, applies the selected media policy,
and maps separate control and doorbell pages. Construction first validates queue
DMA with a CPU bootstrap READ, then requires a GPU SQ-doorbell store to produce
a successful NVMe completion before publishing the queue. A bounded device function
constructs NVMe READ SQEs and PRP lists, batches the scheduled SQ doorbell,
consumes phase-tagged CQEs, validates object/request generations, and publishes
runnable work tickets. An application CTA can opportunistically construct and
ring one read under a one-shot queue lease, then exits without polling. A GPU
queue-model test exercises direct issue, CQ completion, runnable-work execution,
two-request queue contention, forced-lock fallback, stale completion isolation, NVMe status failure,
malformed-CID queue quiescence with cooperative context/credit reclamation, and
fatal queue retirement.
The benchmark performs no CPU command submission or completion polling.

The device program emits only READ commands. IOMMUFD contains DMA to a private
IOAS. Hardware write protection remains the safe default; explicit trusted-code
mode supports controllers without it and is not a media-containment boundary.
The raw queue is a trusted-process interface rather than a per-command or
multi-tenant security boundary. See `NVME_SECURITY.md`.

### M5: KV integration

- Existing paged-attention request/tile binding rebuilt cleanly.
- Runnable-tile scheduling.
- Dense split-K and query-dependent sparse-KV work ticket fixtures.
- End-to-end SLO experiment.

Gate: improvement over CPU/runtime prefetch using the same data path.

Status: mechanism workload complete, serving gate open. The branch runs a real
FP16, head-dimension-128, page-size-16 split-K attention kernel with
heterogeneous pages per request, stable partial softmax reduction,
runnable-work tickets, and a CPU numerical reference. Work formation consumes
FlashInfer's public paged-KV CSR representation, reduction uses FlashInfer's
base-2 `(V, LSE)` state implementation when its headers are available, and a
real FlashInfer decode wrapper is a differential correctness gate. NTA deferral
also executes in version-checked FlashInfer 0.6.12 decode and FA2 paged-prefill
JIT kernels. SGLang 0.5.14 HiCache is wired through the plugin adapter; vLLM
request lifecycle and KV ownership remain open. Stock FlashInfer decode replays
through SGLang's full CUDA-graph mode after stream-ordered acquisition; the
demand progress loop and paged-prefill graph remain open. There is still no
TTFT/TPOT/SLO benefit claim.

The separate sparse fixture runs one CTA per request over a resident summary
catalog. An upstream device kernel materializes the query; the attention CTA
selects top-k full KV pages, fills the common dependency set, and either
consumes direct pages or exits for finite staged acquisition. Runnable work is
mapped back through the ticket rather than assuming request index equals slot.
This is not yet a FlashInfer or SGLang sparse-attention integration.

The ABI-v25 dependency-set workload separately acquires up to 32 mixed-tier
objects per CTA, supports cancellation, stale generations, stale object
versions, and duplicate coalescing, and resumes only after the complete set is
ready. Global-load and TMA attention now consume the same common work and
dependency records rather than duplicating acquisition metadata in a private
task. This demonstrates kernel-neutral mechanics, not a production serving
result.

### M6: TMA specialization

- Census actual production IR/PTX forms.
- Recognize or consume metadata for tensor maps.
- Direct descriptor rebind.
- Resume from HBM staging using the original TMA path.

Gate: no claim based solely on a synthetic TMA kernel.

Status: mechanism complete, production census gate open. A distinct compiler
marker preserves direct tensor-map descriptors or selects an HBM staging
descriptor after data becomes available. The attention CTA initializes its barrier only
after acquisition succeeds, executes `cp.async.bulk.tensor`, and is covered by
memcheck/racecheck/synccheck. Production attention IR recognition and an
untouched-kernel comparison remain required.

### M7: RDMA

- Run on a Mellanox/IBGDA-capable testbed.
- GPU submission and bounded CQ processing.
- Compare with CPU proxy and dedicated GPU progress.

Gate: real network hardware, not loopback or emulation, for performance claims.

Status: explicitly deferred. `SourceKind::Rdma` remains inactive, so the system
does not advertise an RNIC data path. This does not block the HBM, CPU-DRAM, or
NVMe mechanism; it does limit the current claim to local memory and storage.

### M8: generality

- MoE expert acquisition.
- Optional ANNS or graph object acquisition.

Gate: reuse the same compiler/runtime contracts without kernel-name-specific
logic.

Status: mechanism gate complete for device-routed MoE. A GPU hidden-state
producer and top-k router build the common plan on device without exposing the
selected experts to userspace. Compiler-lowered consumers acquire multiple
versioned matrices, mix expert outputs, and check every result against a CPU
reference across tiers. Matched CPU-sync and all-expert overfetch policies are
implemented. Production MoE model baselines and ANNS remain open.

## 21. Target repository layout

The implemented tree is:

```text
CMakeLists.txt
docs/
    ARCHITECTURE.md
    FLASHINFER.md
    SYSTEM_PLAN.md
include/nta/
    AcquireIR.h
    DeviceAPI.cuh
    DeviceWorkPlan.h
    FlashInferAdapter.h
    HostRuntime.h
    RuntimeABI.h
    WorkPlan.h
    Passes.h
lib/
    AcquireAnalysis.cpp
    AcquireLowering.cpp
    DeferralLowering.cpp
    Plugin.cpp
runtime/
    device/Acquire.cuh
    host/{DeviceWorkPlan,FlashInferAdapter,Runtime,WorkPlan}.cpp
tests/
    ir/{batched,dependency-set,reject-*}.ll
    flashinfer/differential_decode.py
    runtime/{AbiTest,FlashInferAdapterTest,RuntimeTest,WorkPlanTest}.cpp
benchmarks/
    kv/{KvAcquire,KvAcquireKernel,KvTypes}
    attention/{PagedAttention,PagedAttentionKernel,PagedAttentionTypes}
    moe/MoeExperts.cpp
    nvme/NvmeRead.cpp
```

No legacy scheduler, CLC, grouped-LPT, cache-hint, or timing implementation is
copied into this branch. A mechanism may be reintroduced only when a milestone
requires it and its tests are rebuilt around this architecture.
