# The sched-pass model: request-differentiated execution inside fused kernels

> Formal unification in **THEORY.md**: one game (DAG(μ) × hierarchy × workers),
> three per-request maps (π order, σ residency, φ consumption) + observation as
> the adjoint, cost = pebbling traffic composed in (max,+), safety = an effect
> type system (E0 hints / E1 permutations / E2 ε-budgeted / O monoid writes),
> control = two timescales. Every PTX instrument is one cell of
> (map × level × timescale); the §3a matrix below is its instruction-set
> projection.

2026-07-02. Sources: eKV `MODEL.md`/`GENERALIZATION.md`, CapKV **v2 branch**
(`capkv/mlir_pass/cxx_v2`, SOUNDNESS/V2_REFACTOR docs), and GPU-validated
experiments in `test/` on **two** GPUs: an idle RTX A6000 (sm_86, clang-19 +
CUDA 12.9, cloudsys02) and an **RTX PRO 6000 Blackwell** (sm_120, 188 SMs, 96 GB,
clang-20 + CUDA 12.8, UC-Santa_Cruz-2) — the latter is where the **Cluster
Launch Control** claim path is validated at runtime, not just compiled.

---

## 1. The problem, and why it lives inside the kernel

Continuous batching fuses many requests into one attention kernel launch.
The step time of that launch is a **makespan**:

    T_step = max over CTAs ( CTA residency )          [eKV MODEL.md §8]

Fusion erases request identity: every request receives the *same* service
order, the same cache behavior, the same effort. A single long-KV request
therefore sets T_step for the whole batch — the straggler/hijacking problem —
and every existing remedy (batch composition, chunked prefill, PD
disaggregation, priority queues) operates **outside** the kernel, where the
uniformity it needs to break no longer exists. The kernel is the last black
box: inside it, requests are indistinguishable CTAs.

**The goal**: restore per-request differentiation inside the fused kernel —
who runs when, how each request uses shared resources, and (in extremis) how
much work each request does — without changing results, engines, or launch
sites. eKV proved the *observation* face of this is weavable by a compiler
pass; CapKV proved the *enforcement* face. sched-pass adds the *scheduling*
face, in CUDA via an LLVM pass plugin, with the same discipline.

### 1b. So can we do per-request scheduling *inside* continuous batching?

Yes — in a precise sense, and with precise limits. What the engine's batch
scheduler (vLLM/SGLang) cannot express, and what this pass now does express,
*inside the one fused kernel it launches*, per request identified through μ:

- **order it** (π / `task_order`): serve a request's tiles earlier or later,
  and — via the ticket/CLC work queue — decide which claim a worker takes next.
- **differentiate its resource use** (SHAPE): give an urgent request pinned,
  prefetched L2 lines while a long request discards its lines (measured −54%
  step time under L2 oversubscription, bit-exact).
- **make it do less** (SHED / tau): cap a lagging request's attended KV units,
  ε-budgeted — the elastic-quality lever.
- **measure it individually** (OBSERVE / clock64, %smid): per-request residency
  and which SM it ran on — the input that closes the loop.

All of this is per-request, CTA-uniform (no divergence), CUDA-graph-replay
safe, and bit-exact except the deliberately-approximate SHED. That is genuine
per-request differentiated *execution* within a fused continuous-batching
kernel — the thing fusion had erased.

The honest limits, so the claim is not oversold:

- It is **not CTA→SM placement**. GPU block scheduling is the hardware's; CLC
  redistributes *within* a launched grid, it does not assign CTAs to SMs. We
  schedule *task acquisition order* and *resource share*, not spatial placement.
- It is **within-kernel, not preemption**. A running long request's CTA is not
  context-switched out; the "preempt-like" effect is resource-yielding (polite
  discard, throttle, shed) plus ordering — not a true preempt/restore.
- It **complements** the engine scheduler, it does not replace it. Batch
  composition, admission, and eviction stay upstream; we add the per-request
  layer *inside* the kernel the engine was blind to.
- It is cleanest where **CTA ≈ request** (decode, grid-mapped). Persistent /
  prefill-q-block / grouped-GQA kernels need the μ→request mapping recovered
  per family (eKV's honest boundary); the timer/policy keying assumes a
  recoverable request index.

So: per-request *scheduling of execution and resources inside the fused kernel*
— yes, validated on Blackwell. Per-request *placement onto SMs, or preemption
of running work* — no, that stays the hardware's.

## 2. One object carries everything: the address map μ

eKV's generalization argument (GENERALIZATION.md §1–4) transfers unchanged.
Attention over non-contiguous KV **must** evaluate

    μ(r, p) = KVpool_base + I[g(r, p)]·stride + h(r, p)

to fetch K/V, so μ — materialized per batch — is necessarily a kernel input,
and the point where it is consumed exists in **every** engine and kernel
form. Everything engine-specific is a *parameter* of μ, not a new model:

| engine / form            | index structure `I`      | grading B | at LLVM level, detected by |
|--------------------------|--------------------------|-----------|----------------------------|
| vLLM paged               | `block_table`            | 16        | address depends on a LOADED value (`hasLoadedIndex`) |
| SGLang radix / token pool| `req_to_token` (flat)    | 1         | same — the radix tree is upstream *policy*; the kernel reads a flat array |
| FlashInfer CSR           | `kv_indptr → kv_indices` | page      | same — any indirection depth |
| contiguous / arithmetic  | none: `base + seq·stride`| ∞         | strided stream whose address derives from ctaid(slot) — a **positive** no-loaded-index classification, never a fallback |
| TMA descriptor           | descriptor-carried       | any       | **not visible as plain loads at LLVM IR** — covered by the other observation levels (launch-arg capture / runtime trace), the same L2/L3 hierarchy as eKV |

Both implemented forms are validated (`paged_decode` mode A–E paged,
mode F arithmetic; SGLang-style CSR is the same loaded-index test at depth 2).
The scheduling model below is defined **on μ**, so it applies at any grading:
a "task" is a request tile of μ's domain, and the KV stream it induces is the
resource footprint the scheduler acts on.

## 3. One pointcut, four lever classes

eKV showed one pointcut carries identity + attribution + control. Scheduling
completes the taxonomy. All levers attach at μ; all are data-driven (tables
at fixed addresses / named globals — CUDA-graph-replay legal); all are
CTA-uniform (no warp divergence); all default to stock behavior when unarmed.

| lever | question it answers | mechanism (this repo) | status on sm_86 |
|---|---|---|---|
| **OBSERVE** | who costs what, where is the critical path | clock64 bracket → `atomicrmw add timer[task]` | ✅ both GPUs (ranks all 8 long tasks costliest) |
| **PLACE** (π, claim) | *when/where* does each task run | `task_order[]` remap + persistent-worker ticket queue; **CLC** (`clusterlaunchcontrol.try_cancel`) on sm_100+ — same interface, `emitClaim()` is the one swap point | ✅ ticket (both GPUs) **and CLC (real Blackwell, bit-exact)** |
| **SHAPE** (a_j) | *how* does a task use shared resources | urgent: `prefetch.global.L2::evict_last` (read ahead + pin); **polite: `discard.global.L2` after the demand load** — the long request drops its non-reused KV lines so they stop evicting others' resident data (real per-request cache bypass, additive, bit-exact) | ✅ both tiers on sm_86 *and* sm_120; **−54% step time when the working set exceeds L2** (Blackwell KV>128 MB; A6000 always) |
| **SHED** (φ) | *how much* work does a task do | per-task budget `tau` caps KV-stream trips: inject an iteration counter into the streaming loop, redirect the load to iteration-0's (in-bounds) line once `ctr ≥ tau` — attend to FEWER units (H2O/Quest ε-budgeted). Additive (address `select`, no CFG surgery on the loop); `tau=0` ⇒ bit-exact | ✅ validated (tau=0 bit-exact; tau>0 bounded finite change) |

CapKV's capability check (`owner_of[key] ∈ caps`) is the same pointcut's
*isolation* face. Observation / scheduling / enforcement are three planes of
one architecture, and that is the deep reason the pass methodology transfers:
they all attach where the request's identity becomes physical — at μ.

### 3a. The instrumentation menu (what ptxas actually accepts)

Grounded by probing ptxas 12.8 per arch (`test/probe_instr.sh`), not from
memory — the last round already cost us a wrong guess (`::evict_first` on
`prefetch` is rejected everywhere). `OK` = assembles; `--` = rejected.

| PTX instrumentation | sm_86 | sm_90 | sm_100 | sm_120 | lever / use |
|---|:--:|:--:|:--:|:--:|---|
| `ld.global.cg` (bypass L1, cache L2) | OK | OK | OK | OK | SHAPE — stream with no L1 reuse |
| `ld.global.cs` (streaming/evict-first) | OK | OK | OK | OK | SHAPE — long-KV read that shouldn't pollute |
| `ld.global.lu` (last-use) | OK | OK | OK | OK | SHAPE — free the line after this read |
| `ld.global.L1::evict_first` | OK | OK | OK | OK | SHAPE — L1 eviction priority on the load |
| `createpolicy.fractional.L2` + `ld…L2::cache_hint` | OK | OK | OK | OK | SHAPE — **per-request L2 residency policy** (data-driven, the clean form) |
| `discard.global.L2 [a],128` | OK | OK | OK | OK | SHAPE — **drop a dead line** (shipped: polite bypass) |
| `applypriority.global.L2::evict_normal` | OK | OK | OK | OK | SHAPE — reset residency on resident lines |
| `prefetch.global.L2::evict_last` | OK | OK | OK | OK | SHAPE — read ahead + pin (shipped: urgent) |
| `prefetchu.L1`, `prefetch.global.L1` | OK | OK | OK | OK | SHAPE — latency hiding |
| `st.global.cs` / `st.global.wt` | OK | OK | OK | OK | SHAPE — output write policy |
| `nanosleep.u32` | OK | OK | OK | OK | SHAPE — throttle / yield (resource-yielding preempt) |
| `%smid` | OK | OK | OK | OK | OBSERVE — which SM a request landed on (spatial attribution) |
| `redux.sync` | OK | OK | OK | OK | OBSERVE — cheap in-kernel warp aggregation |
| `clusterlaunchcontrol.try_cancel` | -- | -- | OK | OK | PLACE — hardware work-stealing (shipped: CLC) |
| `griddepcontrol.wait` / `.launch_dependents` | -- | OK | OK | OK | **COOPERATE (new) — programmatic dependent launch**: overlap prefill→decode / reload→attention across kernels |
| `barrier.cluster.*`, `mapa.shared::cluster` | -- | OK | OK | OK | **COOPERATE (new) — CTA-cluster + distributed shared mem**: a request's CTAs share KV tiles across SMs |
| `elect.sync` | -- | OK | OK | OK | leader election (used inside CLC) |

Two corrections to last round: (1) the aggressive cache-bypass forms
(`cg`/`cs`/`discard`/L2-policy) work on the **A6000 too** — they were wrongly
deferred as "future Blackwell tiers"; (2) `discard.global.L2` is the additive
polite bypass that's now shipped (the demand load reads first, then the line is
discarded — bit-exact). And two whole lever classes open up on sm_90+ that we
have not built:

- **COOPERATE / cross-kernel (griddepcontrol, PDL).** A kernel can signal the
  *next* grid to start before it finishes. In continuous batching this is the
  in-hardware way to overlap a KV reload (or prefill) with the attention that
  consumes it — the σ-placement/pipelining lever made physical, one launch
  boundary at a time. This is the highest-value unbuilt lever for the batch
  pipeline.
- **COOPERATE / intra-request (clusters + distributed shared memory).** A
  request whose KV is huge can spread its CTAs across a cluster and share tiles
  through `mapa` distributed shared memory instead of re-reading HBM per CTA —
  a per-request data-placement lever, Hopper+.

Also available but not yet a lever: `cp.async.bulk` (TMA) with cache hints is
the **MOVE** lever — in-kernel KV offload/reload between HBM and cache tiers,
the memory-tier half of the σ placement model (eKV's two-plane split).

## 4. Why clock64 — three answers

1. **Epistemic.** T_step = max over CTAs. You cannot trim a critical path you
   cannot see, and kernel-granularity profilers (CUPTI) cannot see *which
   request* inside a fused launch is critical. The woven timer is per-request
   residency, keyed by the kernel's own request index — the only tool that
   makes the straggler *visible* (eKV: "uniform policies underwhelm by
   design").
2. **Control input.** Every lever consumes it: π needs per-task cost
   estimates (LPT ordering is *built from* measured cycles — demonstrated
   live); λ needs congestion evidence; action constants ΔT/ΔR are calibrated
   from it; and after acting, it verifies the effect (the same instrument
   closes the loop).
3. **Honesty about coupling.** t_j is not a constant of the task: on a
   contended GPU the *optimal order inverted* (segregation beat LPT by 23%
   when foreign jobs saturated DRAM; LPT won by 32% on an idle GPU). An
   offline cost model would have been silently wrong; only a live observation
   plane tracks the coupling. This is the measured justification for
   observation being a *runtime* plane, not a profiling step.

## 5. The math, layered and honest

**Layer 1 — placement (combinatorial outer problem).** Given per-task costs
t_j and W workers claiming from an ordered queue, this is list scheduling.
LPT order (longest first) carries Graham's guarantee
T ≤ (4/3 − 1/(3W))·OPT for makespan; SPT/SRPT minimizes mean completion time
(short-request latency); EDF handles deadlines. The π table expresses all of
them — *which* order is a policy choice over goals (§7). Measured: LPT from
woven-timer costs cut makespan **31–32%** vs identity (405→272 µs, stable
across runs).

**Layer 2 — coupling (the congestion term).** t_j = t_j(a_j; s) where s is
co-runner state: streams share DRAM bandwidth and L2. Measured: the same
schedule set reordered its optimum under external load (§4.3). The tractable
form is mean-field — t̂_j(a; γ) with a broadcast congestion multiplier γ —
never per-pair interference. Consequence for the theory: Layer-1 guarantees
hold for *fixed* t_j; under coupling, ordering and shaping are complementary,
and the controller must re-estimate (the loop, not a solve).

**Layer 3 — actions (marginal exchange, priced externalities).** Per task,
fire an action iff

    q_j·ΔT_j  >  λ·ΔR_j + H
    urgency × own-time saved  >  price × shared-resource cost + overhead

λ is the control plane's *amortized* shadow price on shared resources
(profiled/PID-updated between steps — never an online market clearing);
kernel-side the rule is a data-driven score with a floor (all scores ≤ 0 ⇒
baseline), i.e. **advisory**: the control plane prices, the CTA decides with
ground truth. Actions differ in sign structure: prefetch buys own-time with
bandwidth; polite streaming sells cache residency to reduce others' time;
shedding (tau) sells bounded accuracy for time. Implemented as ~10
CTA-uniform instructions at kernel entry — never an optimizer in the kernel.

**What we do NOT claim** (the lesson of the model debates): no strong-duality
optimality — discrete actions and coupled times break it; the price model is
a *structured surrogate policy* whose value is measured, not proven. The
provable pieces are Layer 1 (fixed-cost bounds) and the safety properties
(§6).

## 6. Physics / architecture grounding

- **Decode attention is bandwidth-bound: the KV read set IS the cost** (eKV
  MODEL.md §3). So per-request time ≈ bytes(μ image)/effective-bandwidth, and
  effective bandwidth is the shared, contended quantity — interference is not
  incidental, it is the cost model.
- **A single CTA is latency-bound** (4 warps of outstanding loads); prefetch
  converts exposed latency into overlapped bandwidth — measured −3…−9% on the
  streaming kernel when the policy fires.
- **L2 (6 MB on GA102) is the pollution channel**: a long-KV stream has no
  temporal reuse, so caching it evicts short requests' reusable lines
  (Belady says streams should bypass). Hence the tier split: urgent pins
  (evict_last), polite declines to pin; true bypass (`ld.global.cs`,
  `discard`) is the sharper future tier.
- **clock64 is per-SM**: only same-SM deltas are meaningful; the timer sums
  deltas (a well-defined additive resource), never compares raw stamps —
  inherited directly from eKV.
- **CTA-uniform ⇒ divergence-free**: every woven decision keys on
  task-uniform values, so differentiation between requests costs no warp
  divergence — the reason per-CTA (not per-thread) is the natural granularity.

## 7. The goal system (not just "don't let slow requests slow the batch")

The same levers serve distinct objectives; π/λ/hints/tau are the policy
surface, chosen per deployment:

| goal | policy | levers |
|---|---|---|
| step throughput / makespan | LPT π | PLACE (+ SHAPE vs coupling) — **demonstrated −32%** |
| short-request tail latency | SPT/SRPT π for shorts, urgent hints | PLACE + SHAPE |
| SLO / deadlines | EDF/slack π, q = urgency, per-class λ | PLACE + SHAPE |
| fairness / weighted sharing | claim weights in π construction | PLACE |
| elastic quality under overload | raise tau on lagging/long requests (ε-budgeted) | SHED |
| isolation / multi-tenancy | CapKV caps at the same pointcut | enforcement plane |
| attribution / billing / diagnosis | observation alone | OBSERVE |

Note the built-in tension LPT-vs-SPT (makespan vs mean latency): the
mechanism is neutral; the π table is exactly where a serving-level scheduler
expresses its objective *inside* the kernel.

## 8. What is testable where (the honest matrix)

| capability | A6000 sm_86 | Blackwell sm_120 |
|---|---|---|
| clock64 timer, task_order π, ticket work-queue | ✅ runtime | ✅ runtime |
| prefetch ± evict_last / shed (tau) | ✅ runtime | ✅ runtime |
| **CLC dynamic claim** | falls back to ticket (sm<100) | ✅ **runtime, bit-exact** (`ELECT`/`SYNCS.ARRIVE.TRANS64`/`MEMBAR.ALL.CTA` in SASS) |
| guard/shed via key-importance mask | shed-by-budget live; key mask designed | same |
| TMA-carried μ | invisible at LLVM IR (launch-arg/trace levels) | same + descriptor ops |

**Measured summary.**
- *A6000, 16 workers / 64 tasks (ticket):* identity 405 → LPT 272 → LPT+hints
  265 µs, all bit-exact; observation ranks stragglers correctly. This is the
  regime where dynamic acquisition + ordering wins big (workers ≪ tasks).
- *Blackwell, CLC:* every schedule **bit-exact on real hardware** — the CLC
  claim path is correct end-to-end. At grid = num_tasks the 188-SM scheduler
  already load-balances, so ordering/stealing add ≈ 0 (64 tasks: ±0.4%; 8192
  tasks: +1.1% CLC-vs-static). The straggler win needs workers ≪ tasks or
  contention; a big idle GPU with a full grid is the easy case the hardware
  already handles. This is the honest boundary of the *value* (not the
  correctness) of the PLACE lever.
- *Instrumentation cost:* indirection + policy ≈ 0; the host-mapped timer
  atomic is the only real overhead (+12 µs on a 48 µs kernel — gate it or move
  the buffer to device memory). A *contended* A6000 inverts schedule
  optimality (segregation beats LPT) — the coupling/γ evidence.

**Reading the two GPUs together** is the point: the *mechanism* (all four
levers, both acquisition modes) is correct and bit-exact everywhere; the
*payoff* of scheduling is a function of the workers-to-tasks ratio, per-task
variance, and contention — exactly the terms of the layered math (§5). CLC is
not a speedup by itself; it is the hardware substrate that makes the full-grid
dynamic-claim regime available, on top of which π/λ/shed express the policy.

## 9. Boundaries

Real attention kernels (softmax, multi-head, warp-specialized) remain to be
woven — the fixture is decode-shaped but synthetic. The project is pure CUDA:
clang-compiled kernels only; nvcc's closed device pipeline cannot load LLVM
pass plugins, so FlashAttention-class targets must build with clang's CUDA
support. The work-queue transform requires launch-side cooperation (W workers
+ primed ticket). Prefetch read-ahead relies on PTX dropping invalid prefetches.
Two-tier cache shaping is the weakest lever so far (+2%) — the sharper
non-additive tiers (load rewriting) are designed but change the admission
discipline, so they wait until the additive levers are exhausted.
