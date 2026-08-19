# Request-Aware Incremental GPU Operators

Status: canonical research and implementation roadmap; compiler/runtime plus
bounded external-prefix SGLang slice implemented, exact contributor serving
and OSDI evidence open

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
| Contributor | One request-owned numerical partial with explicit data dependencies and one exact reduction identity |
| Critical work | Current data service plus executable and data-blocked compute that can delay a request; not CTA count alone |

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

There is a strict dense-serving limit: finishing resident CTAs inside one
attention launch does not let that request enter the next transformer layer
while another request's activation is incomplete. CTA progress is valuable
only when it contributes a reusable numerical partial, shortens the current
operator's critical path, or combines with scheduler separation that preserves
end-to-end progress. The v10 SGLang experiment confirms that merely delaying
and later re-forming a dense mixed batch shifts latency and is not a benefit.

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

> In heterogeneous serving, the engine knows request lifecycle, SLO, and
> memory pressure, while GPU execution discovers — after launch — exactly
> which cached data the current query needs. Can a compiler-checked contract
> preserve the request-to-demand-to-consumer association across engine,
> runtime, and GPU, so that selection, validation, and acquisition of
> nonresident data execute device-side under engine admission control,
> without persistent kernels, host identity round trips, or resident-path
> regression?

Selective KV acquisition is the flagship instantiation of that contract:
existing boundaries lose the association between request, dynamic demand,
tier placement, and consumer, forcing dense allocation, overfetch, or a host
control round trip. Exact `(V, LSE)` partial contributors remain an
additional execution form for workloads that require exact attention
(Section 3.2); they are not the explanation for the current headline result.

The target is broad applicability and low regret, not strict speedup at every
point. Production has no oracle. Controlled identical-snapshot experiments may
compare each scheduling decision with the best alternative; real traces compare
the complete system against every fixed baseline from the same initial state.

### 3.1 Decision after the whole-layer experiments

The `0.929x` 4K-load and `0.904x` 8K-saturated results reject a system whose
main action is to move a complete layer and then run attention. More tuning of
that path would reproduce a weaker cache/prefetch scheduler. The project keeps
coalesced movement, cache state, and GPU-initiated I/O as substrates, but no
longer treats them as the performance contribution.

The decisive unit was defined as one request-owned numerical contributor
inside a real operator: a valid experiment must show at least one contributor
executing before the last dependency of its request arrives, preserve its
`(V, LSE)` state, and expose that progress to later admission. The recorded
falsification condition was that if real fragmented traces and manually split
FlashInfer cannot pass that test, the compiler/serving thesis is false and
the project must narrow to a runtime mechanism.

### 3.2 Decision after the selective-KV campaigns (2026-08)

The registered goodput wins (capacity shape `2.1107x` with the CI floor above
the `1.5` bar; three consecutive Poisson-shape passes) execute through
selected claims, bounded device staging, and tiered graph replay. In the
winning trials `ticketed_incremental_launches`, `request_work_completed`, and
`progress_snapshots` are all zero while `selected_compiler_launches` and
`tiered_graph_replay_batches` are in the hundreds to thousands. The
contributor test of Section 3.1 therefore did not produce the win, and the
Section 3.1 falsification condition fired in a form its text did not
anticipate: the project did not narrow to a bare runtime mechanism, because
the compiler still carries the load-bearing association checks — fail-closed
acquisition provenance, request-liveness verification, and generation of the
graph-compatible transformed FlashInfer consumers that the fail-closed
benchmark gates attest (zero stock attention, zero fallback). The research
question above is restated to match this evidence: the contribution is the
preserved request-to-demand-to-consumer association, with selective KV as
the flagship instantiation. Exact contributors, work tickets, CTA suspension,
and progress-guided admission are retained as an additional execution form
for exact-attention workloads and as mechanism evidence, not as the claimed
source of the headline result. The known gap in this framing is recorded in
Section 4 terms: in the winning path the compiler checks liveness and
generates the consumers, but selected staging completes before transformed
direct attention runs; a claim-generation and selected-table consumption
contract consumed inside the compiler-generated kernel is the remaining
compiler-depth work.

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

The LLVM contract now has a delimited numerical effect:

```text
bind(request, generation)
acquire(dependencies, work ticket) or defer-and-exit
begin_partial(work ticket)
    optimized numerical body
commit_partial(reduction group, contributor, cost)
```

Acquisition sites retain the strict CTA-uniform control-dependence proof. The
partial endpoints use a different proof appropriate for optimized SIMT code:
request/reduction operands are CTA-uniform, every path into the region crosses
an identity-matched acquired edge, and publication post-dominates the region
despite thread-divergent loops inside it. Acquisition, region begin, and
publication must carry the same request binding and work ticket. Both endpoints
carry LLVM `convergent` semantics. Lowering emits `!nta.partial` and
function-level `!nta.operator` metadata and calls the generation-checked
ticket/reduction protocol. This is a compiler effect, not sampled monitoring.

The LLVM layer should become more capable, but not by guessing request or
reduction semantics from arbitrary NVVM pointer arithmetic. The implementation
boundary is:

1. A typed FlashInfer/Triton/MLIR frontend supplies request, tile, dependency,
   reduction, and reconstructible-coordinate facts while those facts exist.
2. LLVM proves CTA-uniform acquisition, zero live state on finite exit,
   convergence, identity continuity, and exactly-once publication.
3. LLVM specializes one typed operator into direct and incremental entry forms,
   emits a versioned operator plan, and lowers runtime effects.
4. The engine consumes the plan and measured progress; it does not reimplement
   kernel mapping in Python.

Items 1 and 2 exist for the checked FlashInfer frontend. Item 3 currently
lowers explicitly selected forms but does not yet generate both forms or a full
runtime-consumed plan automatically. That automation plus a second typed
frontend is the next compiler contribution, not broader marker insertion.

#### Current performance boundary

The coalesced SGLang integration now forms a real heterogeneous FlashInfer batch;
the earlier scheduler-segregation blocker is closed. A Qwen2.5-3B smoke point
executed 36 mixed layers with both compiler forms, compacted the combined
initial/resume CTA bounds to 50%, used parallel indexed progress, matched stock
output, and used no fallback or stock attention. It delivered only `0.953x`
stock throughput, however, and a four-resident 4K point delivered `0.921x`.

The last ABI-v24 graph point adds four-layer transfer waves and a
warp-cooperative request guard. It executed 754 transformed direct launches
and two ticketed incremental launches with exact output and no fallback, but
still delivered only `0.956x` stock throughput. Restricting the GPU mover to
one or eight CTAs did not recover the loss.

Three arm-balanced repetitions placed the output-throughput geometric mean at
`0.9447x` with bootstrap interval `[0.9337x, 0.9578x]`, external TTFT at
`1.3386x`, resident P99 inter-token latency at `1.4991x`, and SLO goodput at
`0.7498x`. Three repetitions are diagnostic rather than paper-level
statistical evidence. Dense
early-known acquisition is therefore a measured non-goal for the current
mechanism, not a workload on which to imply a universal win.

This falsifies the claim that CTA-level granularity alone wins. The finite
demand-mode operator loop, structural plan reuse across request generations,
and exact-shape graph replay are now implemented. On clean revision `ae7c56a`,
one coalesced 2K host/2K resident diagnostic executed 5,148 transformed
attention launches, ten ticketed launches, three mixed layers, three demand
graph captures and six graph launches with zero fallback. It reached only
`0.9169x` output throughput; external TTFT was `1.1014x`, resident P99 ITL was
`1.8843x`, and stock-derived SLO goodput was `0.4584x`. One process trial is
diagnostic, not a confidence claim.

The remaining limitation is architectural rather than another launch-control
omission: only mixed incremental layers preserve partial progress. Once the
external request enters the proactive layer frontier, the resident request
cannot independently advance through subsequent transformer layers inside the
same dense forward. The next positive gate is therefore end-to-end
model-generated demand that avoids bytes, or within-request arrival skew where
FlashInfer partials overlap transfer, followed by an engine decision that uses
the exported progress. Transfer tuning on early-known dense demand is not
sufficient evidence for the paper thesis.

The ABI v27 selected-demand path provides the first positive SGLang signal after
that reset. Prefix-summary reuse avoids rescanning host K rows on repeated
claims, selected-row refresh reuse cuts bounded-cache preparation from 684 to 72
launches on the 16K diagnostic, and adaptive no-split execution keeps the mixed
batch in one compiler-generated compact FlashInfer launch after selected rows
are staged. Three dirty-tree Qwen2.5-3B seeds at 16K host + 2K resident, budget
32 pages, refresh interval 1024, reported external P95 TTFT `0.831x` stock and
resident P99 ITL `0.891x`, with zero fallback/stock and 512 staging rows versus
16,382 dense rows. Resident P95 TPOT is still `1.085x` stock, so the current
next gate is quality-controlled goodput and resident-throughput recovery, not a
paper claim. A stricter budget-32 smoke rerun with same-budget selector-quality
metadata and required output parity measured external P95 TTFT `0.848x` and
resident P99 ITL `0.804x`, but budget 64 and 128 regressed despite output
parity. The high-recall budget-128 point meets the recall diagnostic and still
loses badly. The separate task-quality smoke harness passed budgets 32, 64,
and 128 on an easy exact-prefix retrieval task with every selected mechanism
active, but that is only a sanity check. A budget-128 profile attributes the
cliff mainly to the current selected direct-operator path (`881.5 ms` across
2,376 layer invocations) plus CPU enqueue (`98.6 ms`), while selected transfer
timing is still under-instrumented. The next implementation target is therefore
precise: make selected execution page-native or otherwise lower-overhead at
quality budgets, and expose selected staging copy time separately from selected
attention time, before treating selected demand as a paper headline.

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
  fatal rejection of unsafe sites, request/generation plus object/version
  binding, and delimited `begin_partial`/`commit_partial` verification that
  rejects bypassed, duplicate, or acquisition-free publication;
- fixed-capacity work tickets, reverse object-to-ticket dependency edges,
  direct exact-once runnable-work publication, tagged per-backend urgency
  queues, and stale-generation isolation;
- direct HBM and mapped-DRAM consumption, staged DRAM, a retained
  `(object_id, version)` HBM staging entry, a hard byte budget for staging
  allocations owned by the runtime, and one-queue VFIO NVMe;
- ABI-v27 work metadata carrying request reduction groups, contributor counts,
  unavailable bytes, and estimated compute cost;
- per-request pending/runnable/completed/expected compute, blocked-byte, and
  complete-contributor summaries with generation-stamped identities;
- a causal critical-work policy and device transport urgency that combine live
  backend queue delay, transfer service, deadline, priority, and remaining
  compiler-attributed compute instead of ranking by CTA count;
- one bounded per-ticket GPU timestamp array that records when each real
  FlashInfer tile first becomes runnable, allowing measured barrier traces
  without a host poll in each progress round;
- optimized FlashInfer decode and paged-prefill hooks, compact runnable-work
  remapping, physically bounded eager initial/resume grids, and request-local
  split-K merge gates tested with one complete and one blocked request in the
  same real FlashInfer launch;
- a finite host-DRAM execution model that chooses one bulk round or bounded
  rounds, or a coalesced request-level transfer overlapped with resident-request
  compute; it uses two streams, one cached structural plan, GPU directory
  rebinding, and first-wave next-layer fragment acquisition when finer partials
  have predicted value;
- a real FlashInfer GPU-selected page path with a stable device-only index
  table, bounded source/destination validation, cold and retained staging, and
  a no-oracle bulk-versus-indexed cost decision;
- a shared bounded CUDA K/V staging owner with generation-changing leases,
  completion-fenced reuse, allocation-derived HBM accounting, and a production
  caller in the exact-partial FlashInfer operator;
- an SGLang selected-demand path with pre-allocation external-prefix
  interception, bounded physical staging, per-layer device page-to-slot
  retention, live-query selection, device-only miss compaction and validation,
  concurrent generation-tagged claims, pinned source lifetime, and
  compiler-generated request-bound FlashInfer consumption, including
  coalesced-batch peer execution overlapped with external miss transfer; and
- an SGLang HiCache adapter that publishes request generations and priorities,
  preserves exact page-map identity, exposes the complete demand path and the
  transformed direct path, consumes generation-checked compiler progress in
  external-batch admission, replays transformed FlashInfer decode graphs after
  stream-ordered acquisition, and captures exact-shape finite demand decode or
  paged-prefill epochs in a separately keyed operator graph.

Not implemented today are automatic compiler generation of separate
complete-data and incremental launch forms from one typed operator, a second
Triton/MLIR or TileLang frontend, a measured elastic
range-coalescing objective, runtime-generic HBM eviction/refcounts, graph replay
inside SGLang's full model graph, deadline/slack propagation into the admission
consumer, an external-prefix engine allocation state, exact contributors and
graph replay in the selected SGLang path, multiple NVMe queue pairs, a vLLM
adapter, or RNIC/RDMA. Current-ABI VFIO NVMe has one
single-controller local
qualification point, not the multi-platform reliability evidence required for
production. Local dense CPU-DRAM traces have not established the P0 opportunity;
NVMe and additional-model traces remain open. The remaining sections are target
design and claim gates, not claims that these parts already exist.

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
production design. Current eager FlashInfer incremental waves use a
conservative physical bound and map that compact launch through canonical
runnable-work IDs. Demand-mode CUDA graph replay still requires a fixed graph
grid or validated device-updated conditional launch and remains open.

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

- Use the compiler-generated NTA direct form for an all-resident batch; untouched
  FlashInfer is an evaluation control, never an internal fallback in the NTA arm.
- Capture the incremental phase only after structural plan upload.
- Skip empty transport and runnable-work nodes using graph-compatible device
  predicates.
- Size runnable launches from the compact count where the framework permits;
  eager FlashInfer now does this. For graph replay, use a fixed graph grid with
  an early collective bound check until conditional device control is validated.
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

Tier position (2026-08-09, from measurement). NVMe remains a first-class tier:
it is one transport class behind the same replica directory, its current-ABI
qualification stands at 6,743 MiB/s (57% of matched fio), and selection makes
it *more* valuable, not less — dense decode from storage is
bandwidth-impossible, but a 3-6% selected fraction of a cold pool served at
NVMe rates is decode-viable byte arithmetic. This host also carries a real
CXL memory expander (`/dev/dax0.0`, 128 GiB devdax, CXL `mem0`, target node
2), and a root-privileged probe verified the GPU consumes it through the
existing mapped-host mechanism: `cudaHostRegister` on the devdax mapping
succeeds and a full-GiB GPU read measured **22.13 GiB/s** with verified
contents. The measured single-host ladder is therefore HBM ~1.6 TiB/s, DRAM
~41 GiB/s, CXL ~22 GiB/s, NVMe ~6.6 GiB/s. The CXL tier is deferred by
decision, not by capability: it requires no new device mechanism, only the
one ABI generalization it shares with any multi-instance tier — the backend
table is currently keyed by `SourceKind`, conflating transport class with
tier instance, so DRAM and CXL cannot yet carry separate credits, bandwidth
estimates, and admission. Backend *instances* per tier are the recorded
design change for the next ABI revision; replica-level cost fields already
express CXL placement today. Transport choice is likewise a measured policy,
not an identity: copy engines win bulk contiguous movement (0.479x SM-mover
ablation), SM gathers win fragmented selected movement (8.17x selected-page
crossover), and the NVMe queue serves the cold tier; the demand cost model
must select the mover per transfer geometry.

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

Current result: local Llama-160M and Qwen2.5-3B whole-prefix CPU-DRAM traces did
not pass the predeclared opportunity gate after finite SM parallelism was
modeled. They contain layer-complete transfer demand, not fragmented
within-request page arrivals that can preserve split-K partials. They falsify
the mover-only design and forced ticketing for known bulk demand. The new
canonical FlashInfer tier-streaming experiment does preserve real `(V, LSE)`
partials and passes the CPU-DRAM operator gate: `1.1714x` over atomic promotion
with a `[1.1660x, 1.1732x]` 95% interval and `4x` lower HBM staging. A separate
heterogeneous-shape point improves by `1.1100x` with `4.83x` lower staging.
The runtime now owns FlashInfer wrappers, partial execution, merge, bounded
slots, and dynamic-source graph replay; paired compiler artifacts validate one
typed request/reduction plan. The measured implementation uses a fixed host
wave order, so it establishes numerical/operator feasibility and a bounded-HBM
crossover, not completion-driven scheduling or an end-to-end serving gain.
Dense real-model CPU/NVMe arrival traces remain the P0 claim gate.

### P1: compiler soundness and incremental form

- Keep post-dominator control dependence for acquisition. Implemented.
- Require convergent partial endpoints, identity-matched acquired edges, no
  acquisition bypass, and exactly-once post-dominating publication. Implemented.
- Migrate the endpoints from the convergent attribute to explicit LLVM
  convergence-control tokens.
- Define typed frontend semantics and a versioned compiler plan. The JIT now
  exports and validates a schema-versioned family/form/capability/source
  contract. Implemented for canonical ragged prefill.
- Emit `K_direct` and `K_incremental` from one real kernel source. Implemented
  for canonical ragged prefill; paged decode/prefill remain open.
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
- Publish already-available work without constructing suspended-ticket state
  and physically compact both the initial and resumed eager grids. Implemented.

Gate: all-resident median overhead at most 5%; dense all-miss performance at
least 90% of the matched bulk path; mixed arrival executes useful partials
before the last page arrives and matches stock output.

### P3: unified incremental scheduler

- Implement elastic grouping with a calibrated request-delay and transfer-cost
  objective.
- Replace capacity scans with changed-object propagation, urgency buckets, and
  compact runnable queues. Changed-object propagation and transport urgency are
  implemented; exact compact launch sizing remains open.
- Export per-request progress and blocked-data summaries. ABI v27 exports
  pending, runnable, completed, and expected compute plus unavailable bytes,
  checked conservation, and dropped-attribution telemetry.
- Rank data and executable contributors from current request critical work.
  The device I/O queue now consumes live request critical work on insertion and
  requeue; a CUDA test verifies that compiler-attributed compute changes
  service order. SGLang external-batch admission now consumes a nonblocking,
  generation-checked critical-work snapshot; decision-regret and SLO evidence
  remain open.
- Keep forced bulk, fine-grained, and whole-request-delay controls for
  evaluation only.

Gate: controlled identical-snapshot decision regret is at most 1.05 median and
1.10 p95; dense grouping reaches the bulk baseline while skewed workloads begin
useful computation earlier.

### P4: SGLang co-scheduling and CUDA graphs

- Execute the real incremental FlashInfer path with no stock fallback.
  Implemented for eager demand mode.
- Feed actual partial progress and predicted data arrival into batch admission.
  Implemented for the external-batch admission decision; SLO benefit remains
  unvalidated.
- Capture the demand reset/progress/runnable sequence for decode and paged
  prefill without capture-illegal upload or synchronization. Implemented as an
  exact-shape NTA operator graph; integration into SGLang's whole-model graph
  remains open.
- Assert zero fallback and identical request/cache traces in measured trials.
- Treat resident P99 inter-token latency, not causally prior resident TTFT, as
  the primary interference metric. Implemented.
- Do not reinstate the measured admission/re-merge policy without a workload
  where it reduces the critical path; five exact trials regressed throughput,
  resident P99 inter-token latency, and external TTFT.

Gate: controlled end-to-end TTFT, TPOT, throughput, and SLO goodput improve over
stock layer waiting, coalesced bulk, and request skip/rebatch at equal cache and
admission state.

### P5: NVMe scale and reliability

- Preserve the historical ABI-v18 single-controller qualification as a
  regression, rerun it on ABI v27, and repeat it across platforms.
- Add multiple queue pairs and depth/transfer sizing from Little's law.
- Separate buffer lifetime from transport lifetime and validate backpressure,
  timeout, reset, cancellation, and stale completion behavior.
- Measure CPU use, SM tax, physical bandwidth, and checksums independently of
  attention.

Gate: at least 50% of matched sequential-read bandwidth, no checksum error, and
end-to-end gains attributable to overlap or useful-byte reduction rather than a
changed cache policy.

### P6: GPU-selected sparse FlashInfer and generality

- Keep FlashInfer `top_k_page_table_transform` output on device. Implemented.
- Feed selected pages through the incremental runtime into real FlashInfer
  paged decode over compact KV. Implemented at operator level.
- Select bulk candidate transfer or bounded indexed transfer without reading
  selected IDs on the CPU. Implemented and crossed over in a five-point sweep.
- Retain custom CUDA sparse attention and MoE only as protocol/fault fixtures.
- Add end-to-end serving integration, a second generated-kernel frontend, and
  rerun MoE as a secondary result.

Gate: real-kernel output parity and a measured crossover where elastic grouping
uses large transfers for dense demand and avoids overfetch for low-selectivity
demand.

### P7: production and paper evidence

- Run 24-hour graph, cancellation, generation-reuse, and failure stress.
- Validate multi-GPU ownership and isolation on physical hardware.
- Run long-context datasets with randomized variant order and controlled clocks.
- Compare against direct access, Strata-style coalesced scheduling, ECHO-style
  sparse prefetch, SparseServe/SPIN-style hierarchical sparse serving,
  Syncopate-style chunk overlap, CPU completion, and persistent GPU progress
  where artifacts permit.

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

> A compiler-checked arrival-driven contributor form for finite batched GPU
> operators: request-owned CTAs may exit before acquiring external data, later
> execute a zero-frame numerical region, publish exactly one reduction
> contributor, and expose exact remaining data and compute to one SLO policy
> without a persistent kernel or partial-CTA polling.

The runtime and engine co-design makes this effect useful: versioned external
data completion unlocks compact CTA work, complete reduction groups unlock
request output, and actual remaining data/compute feeds later batch admission.
Neither the IR effect nor the scheduler alone is the full contribution.

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

SparseServe and SPIN further remove selective transfer, layer-bounded staging,
per-request sparse working sets, and a common hierarchical page abstraction
from the novelty claim. They are required capacity/goodput and quality controls.
The remaining distinction must come from mechanically checked contributor
execution across finite invocations and its request-lifecycle feedback, not a
new cache policy.

Primary sources:

- [Syncopate](https://www.usenix.org/conference/osdi26/presentation/qiang)
- [Strata](https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang)
- [ECHO](https://www.usenix.org/conference/osdi26/presentation/liu-guangda)
- [SparseServe](https://arxiv.org/abs/2509.24626)
- [SPIN](https://arxiv.org/abs/2604.26837)
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
- NTA beats Strata, Syncopate, ECHO, SparseServe, or SPIN without matched
  artifacts; or
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

The evaluation campaign, its four research questions, gates, model matrix, and
go/no-go rule are fixed in `docs/ONE_GPU_EVALUATION.md` ("Research Questions
And Execution Plan"). The 2A barrier characterization executed on 2026-08-09
and measured zero compute-stream stall at every proactive layer barrier across
2K-24K external prefixes (load/compute 0.05-0.09): dense promotion on this
host is already fully overlapped by the lookahead pipeline, so the planned
streaming-operator integration into dense SGLang promotion is cancelled by
measurement. The campaign centerpiece is now the Quest-retrofit
device-selected demand workload (1D), with the mover-priority interference
series (unblocked; movers now default to the lowest CUDA stream priority) as
the 1C gate. Dense early-known demand is a measured boundary (five end-to-end
diagnostics plus the zero-stall characterization), not a target.

The canonical FlashInfer operator experiment now demonstrates exposed overlap
and a bounded-HBM crossover. The next order is therefore fixed:

1. **Implemented:** move wrapper construction, copy-slot lifetime,
   partial-attention, merge, completion, and graph-safe source rebinding out of
   the benchmark into the engine-neutral FlashInfer runtime operator.
2. **Implemented boundary:** export and fail-closed validate a versioned JIT
   operator contract across native, Python, SGLang eager, and SGLang graph
   paths. This proves module identity but does not generate the operator forms.
3. Generate direct and incremental forms plus request/range/reduction metadata from
   the typed FlashInfer frontend, with LLVM retaining convergence,
   generation-identity, and exactly-once publication proofs.
4. Consume that generated plan in SGLang paged prefill and decode, including
   CUDA graph replay, cancellation, and slot reuse with zero stock fallback.
5. Consume ABI-v27 critical-work snapshots in SGLang admission and acquisition
   grouping, then compare atomic promotion, layer wait, and skip/rebatch from
   identical request/cache states. The policy may use only current progress and
   online service calibration, never a future-arrival trace.
6. Attach VFIO NVMe as a producer for the same bounded slots and collect real
   GPU-timestamped CPU-DRAM plus NVMe traces. Scale queues only after the
   end-to-end critical path is measured.
7. Add a second typed generated-kernel frontend and clean multi-machine
   reproduction. RDMA remains deferred until real hardware and a matched
   network baseline are available.
