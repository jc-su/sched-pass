# OSDI Evaluation Plan — Complete Ledger and Critical Path

Written 2026-08-20. This is the single place that answers "what must still
be measured before submission," derived from the four registered research
questions in `ONE_GPU_EVALUATION.md`, the novelty requirements in
`RELATED_WORK.md`, and the sealed results in `PREREGISTRATION.md`. It does
not restate protocol; it states **status, gaps, and order**.

Evidence grades used below:
- **SEALED** — pre-registered campaign, ten paired trials, bars evaluated.
- **MEASURED** — real experiment, banked artifact, not campaign-grade.
- **PARTIAL** — some cells of a required matrix exist.
- **MISSING** — not run.

---

## 1. What is banked today

### The headline is real and reproduced

| claim | value | grade |
|---|---|---|
| SLO goodput, capacity shape | **2.1107x**, CI [1.971, 2.222] | SEALED (C4-second, registered seeds) |
| SLO goodput, reproduction | **2.1100x**, CI [2.012, 2.216] | SEALED (chunk campaign, third seed set) |
| SLO goodput, queue shape | 1.6098 / 1.5517 / 1.6575 | SEALED x3 (P4, P4-second, P4-third) |
| external p95 TTFT | **0.057x** = 17.4x faster | SEALED |
| output throughput | 2.09x, CI [2.011, 2.182] | SEALED |
| output exactness | zero divergence, all trials | SEALED |
| mechanism attestation | zero stock attention, zero fallback, positive staged bytes | SEALED |
| device-chain necessity (B1) | device 2.11x vs host-orchestrated **0.982x** (dense parity) | SEALED, same-revision |

The goodput result has now reproduced across **three independent seed
sets** and two shapes. It is the paper's spine and it is not in question.

### The one failing bar, and its measurement problem

| metric | value | status |
|---|---|---|
| resident p95 TTFT | 0.991x | **parity** |
| resident p95 TPOT | 1.186x, CI [1.028, 1.344] | mild |
| resident p99 ITL | 1.303x, CI **[0.979, 1.707]** | fails 1.05 bar; **CI contains 1.0** |

The p99 ITL confidence interval spans 1.0, so the measurement cannot
distinguish "no regression" from "70% regression." Cause is structural:
8 residents x 128 output tokens = ~1024 ITL samples, so p99 rests on the
~10 worst gaps, driven by only 16 extend events per trial. **This metric
is underpowered as configured** (recorded 2026-08-20). Four mechanism
hypotheses were tested against it and all failed to explain it; two
in-flight runs test whether the residual drift is environmental.

---

## 2. Gap analysis by research question

### RQ1 — Value: does exposing device-discovered demand improve serving?

| sub | requirement | status | gap |
|---|---|---|---|
| 1D | headline selection, goodput >=1.5x at quality parity vs strongest baseline | **SEALED** for goodput | quality is synthetic-only; strongest baseline not yet the strongest |
| 1D-quality | LongBench-class task quality vs dense | **MISSING** | needle/multikey synthetic only (`quality-matrix/`) |
| 1A | long-context capacity, arrival sweeps unloaded->overload, throughput-TTFT curves | **MISSING** | no LooGLE/NarrativeQA/ReviewMT sweeps |
| 1B | heterogeneous agent trace, Mooncake-style, thousands of requests, cancellation churn | **MISSING** | campaigns use 16-24 requests |
| 1C | no-regression controls, all-resident ShareGPT + short-context | **PARTIAL** | resident bar unresolved; ShareGPT control not run |

**Model-scale gap (newly identified):** every sealed campaign ran on
**Qwen2.5-3B-Instruct**, but the registered plan names **Qwen2.5-14B as
1A/1B/1C/1D primary** with Llama-3.1-8B as second model. A 3B-only
headline invites "does this hold at production scale?" and the project's
own regime map says the win region depends on attention share, which
grows with model size. **Reproducing the headline on 14B is the single
highest-value missing experiment.**

### RQ2 — Mechanism: when does this beat waiting/pipelining/bulk/rebatch?

| sub | requirement | status | gap |
|---|---|---|---|
| 2A | opportunity characterization | **MEASURED**, fired no-go for dense promotion | complete; decision recorded |
| 2B | crossover: 8K-128K context x resident fraction x skew, arms incl. layer-wise pipelining, bulk, skip/rebatch, hindsight best | **PARTIAL** | selected-pages sweeps exist; full arm set not run at serving level |
| 2C | HBM-budget sweep, prediction within 20% of measurement | **PARTIAL** | cost model exists; prediction-vs-measurement not closed |

### RQ3 — Necessity: is the co-design required?

| ablation | status |
|---|---|
| host-orchestrated (B1) | **SEALED** — the strongest single result |
| no-device-plan, onstream-prep, baseline | **MEASURED** (`mechanism-ablations/`) |
| no request identity (byte/CTA scheduling) | **MISSING** |
| no measured progress (predicted transfer only) | **MISSING** |
| no engine feedback (mechanism without admission) | **MISSING** |
| manual hand-split operator (does LLVM generation add value) | **MISSING** |
| compiler coverage: paged decode, paged prefill | **MEASURED** |
| compiler coverage: device-routed MoE through identical contract | **PARTIAL** — fixture only, not a production operator |
| verifier mutation testing, every mutant rejected + one miscomputing with verification off | **PARTIAL** — reject fixtures exist, systematic mutation missing |
| **in-kernel claim-consumer contract** | **MISSING** — blocker for the "compiler-enforced" claim |

### RQ4 — Robustness: does one policy stay safe across regimes?

| item | status |
|---|---|
| request-slot reuse, graph replay soundness | **MEASURED** (replay battery) |
| transport-geometry matrix | **PARTIAL** — two cells (8.17x fragmented gather, 0.479x bulk mover) |
| resident-tail interference decomposition | **IN PROGRESS** — H-A/H-B pre-declared |
| nonstationary regime switching vs fixed policies + hindsight oracle | **MISSING** |
| cancellation storms | **MISSING** (recycling merged, now unblocked) |
| 24-hour soak | **MISSING** |
| NVMe fault injection | **MISSING** — and NVMe is framed as architecture, not a measured result |

---

## 3. Baseline and competitor strategy

`RELATED_WORK.md` lists ten systems whose mechanisms we must not claim as
novel. We cannot implement ten systems. The honest, reviewer-defensible
strategy is three tiers:

**Tier A — implemented as real arms in our harness (must have).**
1. **Dense promotion / stock SGLang HiCache** — DONE, every campaign.
2. **Host-orchestrated selection (B1)** — DONE. Same selector, same budget,
   host round trip. This is the BaM-lineage answer.
3. **Deferred / one-step-stale host selection with graphs enabled** —
   MISSING. Recorded as "the strongest conceivable host system." Without
   it, a reviewer says our host baseline was handicapped by a
   per-layer synchronization we chose for it.
4. **Layer-wise pipelined promotion** — required present in every dense
   table per RQ2; stock HiCache approximates it, needs to be stated
   explicitly as such or run as its own arm.

**Tier B — faithful mechanism re-implementation as an arm (should have).**
Pick the two closest and reproduce their *mechanism*, not their codebase:
- **ECHO-style** graph-resident eviction/recall with lossless prefetch.
  Closest competitor (OSDI'26, same venue). Our honest delta is
  request-lease governance and standard dense-trained models vs its
  native-sparse indexers.
- **SparseServe/SPIN-style** working-set-aware batching with per-request
  HBM budgets. Both convert sparsity into admission capacity, which is
  adjacent to our capacity claim.

**Tier C — positioned analytically, not measured (acceptable).**
DirectKV, Strata, Syncopate, Tutti, CoPilotIO, persistent GPU service.
Each already has a `RELATED_WORK.md` entry stating what we do not claim.
Reviewers accept this when the entries are specific and the Tier A/B arms
cover the mechanism families.

---

## 4. Prioritized schedule

Estimates assume ~2.5 GPU-hours per ten-trial campaign and a shared box
with co-tenant wait-gates.

### Tier 1 — submission-blocking

| # | experiment | why blocking | est. |
|---|---|---|---|
| 1 | **Resolve resident-tail measurement**: raise resident output tokens for statistical power; re-run capacity campaign | the only failing bar; currently underpowered | 6h |
| 2 | **Matched-load isolation** | we finish 2.4x faster, so tails are compared at unequal load; fairness challenge is certain | 5h |
| 3 | **Headline on Qwen2.5-14B** | 3B-only headline is the biggest scale objection; registered plan names 14B primary | 8h |
| 4 | **LongBench-class quality at the serving budget** | selection is approximate; synthetic needle/multikey is not task quality | 8h |
| 5 | **Deferred/stale host baseline with graphs** | closes the "your host arm was handicapped" objection | 5h |
| 6 | **In-kernel claim-consumer contract** + reject fixtures | makes "compiler-enforced delegation" literally true | build, ~3d |
| 7 | **All-bars campaign + held-out seed confirmation** (Amendment 5) | the registered claim itself | 6h |

### Tier 2 — strongly expected by reviewers

| # | experiment | est. |
|---|---|---|
| 8 | RQ3 remaining ablations (no identity, no measured progress, no engine feedback, hand-split) | 10h |
| 9 | 2B crossover matrix at serving level with all arms incl. hindsight best | 12h |
| 10 | ECHO-style and SparseServe/SPIN-style mechanism arms | build + 10h |
| 11 | 1C no-regression controls (all-resident ShareGPT, short-context) | 5h |
| 12 | Verifier mutation testing, systematic | build + 2h |
| 13 | Cancellation storms (RQ4, now unblocked) | 4h |

### Tier 3 — future-work section, not blocking

24-hour soak; NVMe fault injection and real NVMe leg; 1B thousand-request
agent trace; full transport-geometry matrix; second production operator
family (MoE beyond fixture); Qwen3-30B-A3B MoE serving point.

---

## 5. Risk register

| risk | severity | mitigation |
|---|---|---|
| Resident bar never passes | **high** | Report as characterized tradeoff with parity TTFT + matched-load result. A quantified cost is publishable; the goodput spine is unaffected. |
| Tail drift is environmental and contaminates sealed campaigns | **high** | Drift probe in flight. If positive, add memory hygiene between trials and re-run affected campaigns; disclose in the paper. |
| 3B-only results | **high** | Tier 1 #3. Regime map already argues the win grows with attention share. |
| Reviewer demands direct ECHO comparison | **medium** | Tier B mechanism arm plus explicit delta statement; ECHO targets native-sparse models, we target dense checkpoints. |
| "Compiler is decorative" | **medium** | Tier 1 #6 makes enforcement real; verifier mutation testing evidences it. |
| Quality regression at serving budget on real tasks | **medium** | Tier 1 #4 measures it before we commit to a budget; quality-gated policy option already exists. |
| Co-tenant GPU contention delays everything | **low** | Wait-gates, detached runs, artifact resume already standard. |

---

## 6. Go / no-go decision points

1. **After Tier 1 #1-#2**: if resident tail is at parity under a powered,
   matched-load measurement, the isolation story becomes a positive
   result and the bar is claimed. If it remains >1.05, switch the paper
   to a characterized-tradeoff framing and stop spending on it.
2. **After Tier 1 #3**: if 14B does not reproduce >=1.5x goodput, the
   claim narrows to a stated model-scale regime, with the regime map as
   the supporting argument. This is a scope change, not a failure.
3. **After Tier 1 #4**: if LongBench-class quality drops at budget 64,
   raise the budget to the quality-parity point and re-measure goodput
   there. The paper's number is the goodput **at quality parity**, never
   the best goodput at any budget.
4. **After Tier 1 #6**: only then may the paper say "compiler-enforced."
   Until then the claim is "compiler-verified confinement of staged
   pointers plus runtime-enforced claim confinement," which is what the
   code does today.

---

## 7. What the paper claims, in the order the evidence supports it

1. Tiered sparse-attention serving must decide admission and HBM capacity
   before query-conditioned working-set identities exist.
2. Returning those identities to the host creates a same-step control
   edge; avoiding it by dense promotion, prediction, or stale reuse costs
   capacity, bandwidth, or quality.
3. NTA delegates a bounded, generation-tagged, revocable **lease**; the
   device selects, validates, acquires, and consumes inside it without
   exposing page identities; the compiler verifies the delegation is safe
   to grant.
4. **Result**: 2.11x SLO goodput and 17.4x lower external TTFT at
   byte-exact outputs, reproduced across three seed sets and two shapes.
5. **Necessity**: the same selector with host orchestration reaches only
   dense parity (0.982x), so the win is the delegation, not the policy.
6. **Cost**: co-resident tail behaviour, characterized honestly at
   matched load.
