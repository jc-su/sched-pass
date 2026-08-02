# Request-Aware Incremental Execution Plan

Status: canonical research and implementation roadmap

This document defines the revised problem, co-design, implementation order, and
claim boundary. `ARCHITECTURE.md` remains the mechanism contract. Historical
validation documents retain literal ABI names such as `Ready`; public design
language uses the terms below.

## 1. Terminology

| Design term | Meaning |
| --- | --- |
| Data availability | Whether a tile's complete input dependencies can be consumed now |
| Runnable tile | One logical kernel tile whose dependencies are available and whose request generation remains valid |
| Runnable tile set | The bounded worklist that may execute in the next finite launch |
| Incremental operator | A compiled operator that can execute valid subsets over several finite launches and merge complete partial results |
| Runnable-work launch | A finite launch over a compiler-validated runnable tile set |
| All-or-nothing kernel barrier | The delay caused when an engine waits for every input before launching any part of an otherwise decomposable operator |

`Ready` remains the internal ABI state meaning that all dependencies of one work
ticket are available. It is not the research abstraction or system name.

## 2. Real Problem

### 2.1 Atomic operator boundary

Serving engines invoke optimized attention as an atomic operator:

```text
acquire every required KV page
    -> launch one complete attention operation
    -> expose request output
```

The compiled implementation is not atomic internally. FlashInfer decomposes a
batch into request-owned KV chunks, CTAs, partial softmax states, and a final
reduction. When KV is split across HBM, CPU DRAM, and storage, those chunks can
become available at different times. The operator API does not expose a safe way
to execute the available subset.

This creates an all-or-nothing kernel barrier: a small slow subset can delay
useful computation for the rest of the batch. A serving scheduler may skip a
whole request, but it cannot know whether already available chunks provide
valuable partial progress or how to preserve and merge that progress without
kernel-specific logic.

### 2.2 The three-way semantic gap

No existing component sees the complete relation:

```text
request policy
    -> logical kernel tile
    -> required tensor ranges
    -> physical data arrival
    -> partial numerical progress
    -> request completion
```

The engine knows request identity, generation, cancellation, tenant, deadline,
and admission state. It does not own the kernel's CTA mapping, partial-result
layout, or reduction completeness.

The compiled kernel knows tile coordinates, data consumption, shared-memory
boundaries, and reduction structure. It does not know request SLOs, future batch
choices, transport queue delay, or cancellation after launch.

The transport knows addresses, commands, queue state, and completion. It does
not know which completion advances an urgent request or unlocks useful compute.

Pointers and launch order cannot reconstruct the missing semantics. The system
must preserve request, tile, object, version, and partial-result identity across
all three layers.

### 2.3 Why this is broader than sparse attention

Dense attention eventually consumes all exact KV, so incremental execution does
not reduce required bytes. Its opportunity is to overlap computation on
available chunks with slower data arrival and to complete unaffected requests
earlier. This matters when a batch contains within-request or across-request
arrival skew and insufficient complementary work to hide it at the engine
level.

GPU-selected sparse attention adds a second benefit: it can avoid transferring
unselected pages. It is the strongest device-generated-demand stress case, not
the general motivation.

Device-routed MoE is a secondary generality case. It must be described as a
device-resident routing pipeline unless expert identity is genuinely hidden
inside the consumer kernel.

### 2.4 Granularity conflict

Incremental execution is useful only if it preserves transfer and launch
efficiency. One I/O and one launch per tile would lose badly for dense,
contiguous demand. Waiting for a complete layer loses overlap under skew.

The system therefore controls a continuous grouping decision:

```text
merge acquisition groups while

saved setup + saved commands + saved transfer time
    > added request delay + SLO risk + delayed useful compute
```

This is one request-aware incremental scheduler. Direct execution, large bulk
transfers, fine-grained acquisition, and whole-request delay are limiting
outcomes of the same algorithm, not independent production policies.

## 3. Research Question

> Can a compiler turn an all-or-nothing batched GPU kernel into an incrementally
> executable operator that performs useful request/tile work as data arrives
> from HBM, CPU DRAM, or storage, while a serving runtime jointly schedules data
> grouping, runnable computation, and request admission without sacrificing the
> native all-resident path?

The target is broad applicability and low regret, not strict speedup at every
point. Production has no oracle. Controlled identical-snapshot experiments may
compare each scheduling decision with the best alternative; real traces compare
the complete system against every fixed baseline from the same initial state.

## 4. Co-Design

### 4.1 Compiler: make the operator incremental

The compiler consumes typed request/tile and tensor-dependency semantics before
they are erased by low-level pointer arithmetic. The raw LLVM pass is the final
legality verifier and lowering stage, not a claim that NVVM can infer request
identity from arbitrary addresses.

For each supported operator, the compiler derives:

```text
logical tile -> request binding
logical tile -> object/range dependencies
logical tile -> partial-result slot and reduction group
logical tile -> reconstructible launch coordinates
legal pre-state exit and re-entry boundary
static resource and compute-cost features
```

It emits coordinated forms of the real kernel:

```text
K_direct       original complete-data path
K_incremental  execute a compact runnable tile set
K_discover     optional device-generated dependency discovery
K_merge        merge only complete current-generation contributors
```

`K_direct` must retain the original optimized mainloop and launch geometry. The
incremental form may externalize only reconstructible state; registers, shared
memory, and barriers never survive a finite CTA exit.

### 4.2 Runtime: schedule data and computation together

The runtime maintains four bounded structures:

```text
request table       lifecycle, SLO, cancellation, generation
object directory    replicas, versions, addresses or transport endpoints
missing groups      coalesced unavailable object ranges and their dependents
runnable tile set   compiler-produced logical work whose inputs are available
```

One scheduling round:

```text
consume transport completions
    -> mark object versions available
    -> update dependent tile counters
    -> append newly runnable tiles
    -> score missing groups by request impact and SLO
    -> merge or issue bounded transfers
    -> launch bounded runnable work
    -> merge requests with complete contributors
    -> export progress to the engine
```

No step waits for future work. No CTA polls for an external completion. No
persistent kernel is required.

### 4.3 Engine: schedule requests using actual progress

The engine publishes request lifecycle, SLO slack, page/object mappings, and a
frozen structural plan. The runtime reports:

- completed requests and partial-contributor counts;
- remaining unavailable bytes and predicted arrival by request;
- runnable compute cost by request;
- cancellation, stale completion, and fallback counters; and
- transfer and launch cost observations.

The engine uses this information when admitting the next batch. A completely
blocked request naturally remains outside the runnable set; a request with
valuable available partials can continue. This is more precise than either
always waiting or always skipping the whole request.

### 4.4 Transport: preserve semantics, specialize mechanics

HBM, mapped CPU DRAM, staged CPU DRAM, NVMe, and a future RDMA backend share
request/object/version semantics. They do not share queue formats or memory
ordering.

- HBM uses the original address with no ticket or queue operation.
- Mapped DRAM may be consumed directly when topology and access shape justify
  it.
- Staged DRAM copies into an HBM cache and publishes data availability.
- NVMe uses bounded GPU submission and completion over VFIO-owned queues.
- RDMA remains inactive until a real RNIC implementation and testbed exist.

GPU-initiated NVMe is necessary when device-generated demand should avoid a CPU
round trip; it is not the contribution by itself.

### 4.5 Current implementation boundary

Implemented today are:

- one version- and source-hash-checked typed C++ frontend for FlashInfer decode
  and FA2 paged prefill: it retains scheduler request/tile coordinates, inserts
  the pre-state work hook, and carries request-local merge groups before those
  facts are erased into pointer arithmetic;
- explicit-marker LLVM lowering with post-dominator control-dependence checks,
  fatal rejection of unsafe sites, and request/generation plus object/version
  binding;
- fixed-capacity work tickets, reverse object-to-ticket dependency edges,
  direct exact-once runnable-work publication, tagged per-backend urgency
  queues, and stale-generation isolation;
- direct HBM and mapped-DRAM consumption, staged DRAM, a retained
  `(object_id, version)` HBM staging entry, a hard byte budget for staging
  allocations owned by the runtime, and one-queue VFIO NVMe;
- ABI-v19 work metadata carrying request reduction groups, contributor counts,
  unavailable bytes, and estimated compute cost;
- per-request blocked-byte/runnable-compute/completed-compute summaries and
  request-local complete-contributor counters;
- one bounded per-ticket GPU timestamp array that records when each real
  FlashInfer tile first becomes runnable, allowing measured barrier traces
  without a host poll in each progress round;
- optimized FlashInfer decode and paged-prefill hooks, compact runnable-work
  remapping, and request-local split-K merge gates tested with one complete and
  one blocked request in the same real FlashInfer launch;
- a finite host-DRAM execution model that chooses one bulk round or bounded
  rounds, with two-stream transfer/compute overlap; and
- an SGLang HiCache adapter that publishes request generations and priorities,
  preserves exact page-map identity, exposes the complete demand path and the
  preacquired fast path, and replays instrumented FlashInfer decode CUDA graphs.

Not implemented today are compiler generation of separate complete-data and
incremental forms from one typed operator, a second Triton/MLIR or TileLang
frontend, a measured elastic
range-coalescing objective, runtime-generic HBM eviction/refcounts, graph replay
of the demand-mode progress loop, use of partial progress in engine batch
admission, a real FlashInfer sparse path, multiple NVMe queue pairs, a vLLM
adapter, or RNIC/RDMA. Current-ABI VFIO NVMe has one single-controller local
qualification point, not the multi-platform reliability evidence required for
production. Real dense serving traces have not yet established the P0
opportunity. The remaining sections are target design and claim gates, not
claims that these parts already exist.

## 5. Why The Mechanisms Belong

| Mechanism | System role | Failure without it |
| --- | --- | --- |
| Compiler-generated work mapping | Connect requests, tiles, inputs, and reductions | Operator remains opaque and all-or-nothing |
| Runnable tile worklist | Execute only currently valid work | Full-grid relaunch wastes launch and SM resources |
| Direct kernel form | Preserve the original resident path | Common-case overhead invalidates the design |
| Common object semantics | Retain identity across memory and storage | Completions cannot safely unlock request work |
| Elastic coalescing | Combine nearby demand when delay permits | Fine-grained I/O underuses bandwidth |
| Request-aware scoring | Value data and compute by request impact | Low-value work can delay urgent requests |
| GPU-initiated storage | Submit device-generated misses without host materialization | CPU round trip reintroduces the semantic gap |
| Complete-contributor merge | Preserve numerical correctness across launches | Partial scratch can become visible as final output |
| Engine feedback | Form later batches from actual progress | Userspace remains blind after instrumenting the kernel |

These are supporting mechanisms with explicit responsibilities. The candidate
contribution is their compiler/runtime/engine coordination around incremental
operator execution.

## 6. Scalable Architecture

### 6.1 Complexity targets

Hot-path work must scale with active change, not catalog capacity:

```text
completion processing: O(completions)
newly runnable publication: O(dependency edges touched by completions)
transfer selection: O(active missing groups) initially, then O(1) buckets
runnable launch width: O(runnable tiles), not original full grid
direct execution: identical kernel path with O(1) launch dispatch
```

Full object-table or full work-table scans are qualification fallbacks, not a
production design.

### 6.2 Hierarchical summaries

Maintain batch, request, reduction-group, and tile counters. A completion first
updates its object and direct dependents, then propagates only changed summary
counters. The engine reads per-request summaries rather than downloading page
maps or tile arrays.

### 6.3 Sharding and contention

- Shard missing-group ownership by backend and object hash.
- Use one producer ownership transition per unique object/version.
- Keep request and tenant credits independent of transport queue locks.
- Use multiple NVMe SQ/CQ pairs sized from bandwidth-delay product.
- Assign bounded progress work per queue; never serialize unrelated DRAM and
  NVMe activity on one global lease.
- Partition runnable work by kernel variant and reduction group so compaction
  does not require a global sort.

### 6.4 Memory management

All device metadata is preallocated for a declared capacity. Capacity failure
must reject or defer affected requests before partial execution rather than
silently falling back.

Staged objects use an HBM cache keyed by `(object_id, version)` with reference
counts and bounded eviction. Epoch reset must not discard a valid staged copy.
Structural plans are double buffered and uploaded once per frozen decode step,
not once per layer.

### 6.5 Graph and launch efficiency

- Dispatch untouched FlashInfer for an all-resident batch.
- Capture the incremental phase only after structural plan upload.
- Skip empty transport and runnable-work nodes using graph-compatible device
  predicates.
- Size runnable launches from the compact count where the framework permits;
  otherwise use a fixed graph grid with an early collective bound check.
- Fuse completion publication into backend progress when doing so preserves
  ownership and visibility ordering.

## 7. Implementation Ownership

### Compiler frontend

Introduce typed operations or intrinsics for request binding, object range,
partial contribution, and legal incremental boundary. FlashInfer JIT templates
are the first frontend. A Triton/MLIR or TileLang frontend is required for the
second-kernel-family gate.

### LLVM backend

The LLVM pass verifies control dependence, CTA collectivity, reconstructible
operands, and no live state across the unavailable-data edge. Operator adapters
provide typed reduction groups; the runtime and generated merge form enforce
complete current-generation contributors. The target compiler emits direct and
incremental entry logic and links the backend-neutral device policy.

### Engine-neutral runtime

Keep request tracking, object placement, missing-group formation, availability
counters, runnable-tile compaction, transport credits, and telemetry independent
of SGLang or vLLM types. Existing `WorkPlan` and work-ticket ABI structures may
evolve internally; public naming should describe incremental execution rather
than expose one transport policy.

### Engine adapters

Adapters own request-ID mapping, page-table ownership, cancellation hooks,
stream/graph lifetime, and batch admission. They must not reimplement the
compiler plan or native runtime layouts in Python.

### Transport backends

Each backend owns registration, submission, completion, recovery, and physical
cost estimates. It receives immutable request/object identities and returns
versioned completion; it never decides numerical completion or request reuse.

### Cross-cutting engineering rules

- Keep one native ABI definition and enforce native/C/Python size and offset
  tripwires. Generating the language layouts from that definition remains the
  preferred production endpoint.
- Keep numerical partial state owned by the compiled operator, acquisition
  state owned by the runtime, engine lifecycle owned by the adapter, and queue
  state owned by the backend.
- Make benchmark programs thin clients of production APIs. Shared CUDA error,
  placement, buffer, and telemetry helpers belong in one tested library.
- Reject unsupported kernels, capture states, capacity, and transport failures
  explicitly. Measured runs assert zero fallback rather than converting errors
  into a stock path.
- Remove unused launch templates and counters or give them production callers
  and direct tests. Telemetry must correspond to work that actually ran.
- Keep source overlays version- and hash-checked until upstream typed hooks
  exist. Unknown sources fail closed.
- Require unit tests for state transitions, IR tests for every legality rule,
  differential numerical tests for every generated form, and contention tests
  for every queue path.
- Keep transport optimization out of compiler analysis and engine-specific
  policy out of the native runtime.

## 8. Implementation Plan

### P0: establish the opportunity

Instrument unmodified SGLang and FlashInfer to record per-request page arrival,
operator launch, chunk mapping, available complementary work, and completion.
Compute:

```text
arrival_spread          = max(arrival_i) - median(arrival_i)
available_before_launch = tiles available before atomic launch / total tiles
blocked_compute_area    = sum(max(arrival)-arrival_i) * calibrated_tile_cost_i
```

Replay a resource-constrained incremental schedule offline. Stop or narrow the
project if real dense traces show little exposed opportunity after existing
batch scheduling.

Gate: reproducible traces from at least two real models and CPU-DRAM plus NVMe
arrival distributions demonstrate material all-or-nothing barrier cost.

### P1: compiler soundness and incremental form

- Replace ad-hoc collective analysis with principled control dependence and
  convergence validation.
- Define typed frontend semantics and a versioned compiler plan.
- Emit `K_direct` and `K_incremental` from one real kernel source.
- Stamp work tickets and partial contributors with epoch and generation.
- Refuse final merge until every current contributor is complete.

Gate: differential output, IR rejection tests, memcheck, racecheck, synccheck,
and an untouched resident-path comparison all pass.

### P2: real dense FlashInfer

- Generate request/chunk dependencies from real FlashInfer decode and paged
  prefill scheduling.
- Execute arbitrary runnable chunk subsets in the optimized kernel.
- Preserve FlashInfer `(V, LSE)` partials and deterministic merge order.
- Add the object/version HBM staging cache.
- Remove per-layer plan upload and host synchronization.

Gate: all-resident median overhead at most 5%; dense all-miss performance at
least 90% of the matched bulk path; mixed arrival executes useful partials
before the last page arrives and matches stock output.

### P3: unified incremental scheduler

- Implement elastic grouping with a calibrated request-delay and transfer-cost
  objective.
- Replace capacity scans with changed-object propagation, urgency buckets, and
  compact runnable queues.
- Export per-request progress and blocked-data summaries.
- Keep forced bulk, fine-grained, and whole-request-delay controls for
  evaluation only.

Gate: controlled identical-snapshot decision regret is at most 1.05 median and
1.10 p95; dense grouping reaches the bulk baseline while skewed workloads begin
useful computation earlier.

### P4: SGLang co-scheduling and CUDA graphs

- Replace the current preacquired-only fast result with the real incremental
  FlashInfer path.
- Feed actual partial progress and predicted data arrival into batch admission.
- Extend the implemented decode graph replay to the demand-mode phase and paged
  prefill, with no capture-illegal upload or synchronization.
- Assert zero fallback and identical request/cache traces in measured trials.

Gate: controlled end-to-end TTFT, TPOT, throughput, and SLO goodput improve over
stock layer waiting, coalesced bulk, and request skip/rebatch at equal cache and
admission state.

### P5: NVMe scale and reliability

- Preserve the historical ABI-v18 single-controller qualification as a
  regression, rerun it on ABI v19, and repeat it across platforms.
- Add multiple queue pairs and depth/transfer sizing from Little's law.
- Separate buffer lifetime from transport lifetime and validate backpressure,
  timeout, reset, cancellation, and stale completion behavior.
- Measure CPU use, SM tax, physical bandwidth, and checksums independently of
  attention.

Gate: at least 50% of matched sequential-read bandwidth, no checksum error, and
end-to-end gains attributable to overlap or useful-byte reduction rather than a
changed cache policy.

### P6: GPU-selected sparse FlashInfer and generality

- Keep FlashInfer `top_k_page_table_transform` output on device.
- Feed selected pages into real FlashInfer sparse attention and the same
  incremental runtime.
- Retain custom CUDA sparse attention and MoE only as protocol/fault fixtures.
- Add a second generated-kernel frontend and rerun MoE as a secondary result.

Gate: real-kernel output parity and a measured crossover where elastic grouping
uses large transfers for dense demand and avoids overfetch for low-selectivity
demand.

### P7: production and paper evidence

- Run 24-hour graph, cancellation, generation-reuse, and failure stress.
- Validate multi-GPU ownership and isolation on physical hardware.
- Run long-context datasets with randomized variant order and controlled clocks.
- Compare against direct access, Strata-style coalesced scheduling, ECHO-style
  sparse prefetch, Syncopate-style chunk overlap, CPU completion, and persistent
  GPU progress where artifacts permit.

Gate: every claim maps to a clean revision, real workload, matched baseline,
confidence interval, and published raw artifact.

## 9. Primary Evaluation Matrix

| ID | Variant | Purpose |
| --- | --- | --- |
| B0 | Untouched FlashInfer/SGLang | numerical and all-resident control |
| B1 | Compiler-generated direct form | transformation no-op cost |
| B2 | Layer-complete copy/prefetch | conventional all-or-nothing baseline |
| B3 | Coalesced bulk acquisition | dense transfer baseline |
| B4 | Whole-request skip and rebatch | engine scheduling baseline |
| B5 | Forced fine-grained incremental execution | mechanism endpoint |
| B6 | Unified request-aware incremental scheduler | proposed complete system |
| B7 | Best fixed whole-trace B1-B5 | end-to-end hindsight reference |

B7 is not an online oracle. Per-decision regret is valid only when every
alternative is replayed from an identical batch, cache, and queue snapshot.

All variants use the same model, real FlashInfer compute, request trace, HBM
budget, initial cache state, page size, and admission limit. Report TTFT, TPOT,
throughput, SLO goodput, data bytes by tier, command count, time until first and
all useful partials execute, CPU use, SM tax, HBM footprint, and direct-path
overhead.

## 10. Novelty Boundary

The following are established techniques and remain necessary system parts:

- worklists and kernel variants;
- GPU-initiated NVMe;
- common object descriptors;
- request-aware scheduling;
- adaptive or cost-based grouping; and
- compiler-generated communication overlap.

The candidate contribution is:

> Compiler and serving-runtime co-design that transforms an atomic, finite,
> batched GPU operator into request-scoped incremental execution, then jointly
> schedules arriving data, useful partial computation, and subsequent request
> admission while preserving the original complete-data path.

Syncopate is the closest compiler comparison because it aligns computation with
communication-chunk availability inside transformed Triton kernels. NTA must
demonstrate additional behavior: external latency spanning finite invocations,
request generation and cancellation, compiler-derived partial reductions,
runtime data grouping, and feedback into serving batch formation. Otherwise the
work should be described as Syncopate plus hierarchical I/O.

Strata is the closest serving comparison because it combines GPU-assisted
transfer with cache-aware batching. NTA cannot claim hierarchical caching,
coalescing, or scheduler awareness. The required distinction is safe partial
execution inside the real compiled operator and use of that progress in later
batch decisions.

ECHO is the required sparse comparison. Sparse offload, graph-friendly recall,
and indexer/recall overlap are not independent contributions.

Primary sources:

- [Syncopate](https://www.usenix.org/conference/osdi26/presentation/qiang)
- [Strata](https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang)
- [ECHO](https://www.usenix.org/conference/osdi26/presentation/liu-guangda)
- [FlashInfer top-k](https://docs.flashinfer.ai/api/topk.html)
- [FlashInfer sparse attention](https://docs.flashinfer.ai/api/sparse.html)

## 11. Claims And Kill Criteria

Allowed only after the gates pass:

> Incremental execution reduces the all-or-nothing kernel barrier under
> heterogeneous data arrival, while request-aware grouping preserves bulk I/O
> efficiency and the compiler-generated direct form preserves resident
> performance.

Not allowed:

- strict universal speedup;
- fine-grained acquisition always beats bulk transfer;
- GPU I/O, a descriptor, worklists, or the LLVM pass is novel by itself;
- custom CUDA fixtures establish production performance;
- NTA beats Strata, Syncopate, or ECHO without matched artifacts; or
- local qualification establishes production or OSDI readiness.

Stop or narrow the project if:

1. Real dense serving traces show no material all-or-nothing barrier after
   existing scheduling.
2. Manually splitting FlashInfer cannot expose useful partial computation before
   the final data arrival.
3. The compiler cannot automate that split on more than one real kernel family.
4. Direct-form overhead exceeds 5% or dense grouping remains more than 10%
   behind matched bulk transfer.
5. End-to-end gains disappear against request skip/rebatch at equal cache and
   admission state.
6. Improvements come from a new transport or cache policy rather than
   incremental operator execution.

## 12. Immediate Order

Freeze further mechanism expansion and run P0 on real dense SGLang traces.
In parallel, extend the first typed FlashInfer frontend to generate distinct
direct and incremental forms and finish the P2 direct-path comparison. Continue
P2-P4 on CPU DRAM because that isolates incremental execution from storage
implementation quality. Scale NVMe only after the dense operator result
survives its matched bulk and skip/rebatch baselines. Add the real sparse
FlashInfer stress case and second frontend after that. RDMA remains deferred
until real hardware and a matched network baseline are available.
