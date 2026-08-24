# The Causal Chain: Factorization, Decomposition, and the Missing Arm

Written 2026-08-24 after the external deep-review of recent OSDI work
(InfiniGen, ECHO, Strata, DirectKV, Syncopate, MPK, CoPilotIO, Llumnix).
This document does three things: fixes the demand/execution
factorization the evaluation must respect, accounts for what every
banked artifact already proves inside that factorization, and specifies
the one missing execution arm that the causal claim cannot survive
without.

## 1. The factorization

Selection policy decides WHICH data is needed. The execution protocol
decides HOW execution proceeds once demand exists. Every experiment
must vary exactly one.

Demand semantics (rows):
- D1 full exact (dense; every contributor needed)
- D2 fixed selected set (replayed identical selection)
- D3 model-generated top-k (Quest-lineage; approximate vs dense)
- D4 native sparse (model-defined; exact w.r.t. its own semantics) —
  no runnable model on this host; positioned, not measured

Execution protocols (columns):
- E1 all-HBM resident (upper bound)
- E2 atomic full promotion
- E3 layer-wise pipelined promotion (stock HiCache approximates)
- E4 host selection + synchronous round trip (B1)
- E5 deferred / one-step-stale host selection (B2; scaffold built)
- E6 GPU selection + conventional gather (select on device, gather,
  THEN run; no overlap, no cache reuse, no graphs) — **MISSING, the
  decisive arm**
- E7 NTA: claim-governed bounded staging, per-layer overlap,
  graph-compatible consumption, engine admission

## 2. What the winning path mechanically is — stated without romance

The sealed serving wins execute: device-side per-layer selection →
claim (generation-tagged lease, bounded staging rows) → transfer
kernels on claim streams → compute stream waits per-layer copy events →
transformed consumers → epoch-cached graph replay for decode → fenced
retirement. Waiting is stream-event-ordered WITHIN a forward. There is
no completion-driven request resume in the serving path, no ticket
rescheduling, and the exact-contributor machinery is idle there
(counters zero; SYSTEM_PLAN 3.2). Those mechanisms are implemented and
validated at the OPERATOR level (TIER_STREAMING: 1.1714x
[1.1660, 1.1732] over atomic promotion at 4x less staging HBM; ABI
lifecycle stress; replay battery) and are presented as such — never as
the source of the serving numbers.

This is recorded because the alternative — implying the serving win
flows through the resume machinery — is the exact conflation the
external review predicts reviewers will catch. They would be right.

## 3. Decomposition ledger for the headline

Claimed decomposition of the capacity-shape 2.11x (and the confirmed
queue-shape 1.53/1.67), with evidence status:

| component | isolating comparison | status | value |
|---|---|---|---|
| byte reduction (selection alone) | stock (E3/D1) vs B1 (E4/D3): same selected bytes, host execution | **SEALED** | B1 = 0.982x — byte reduction alone contributes ~zero at this shape; the win is execution-side |
| host round-trip elimination | B1 (E4) vs E6, same selected set | **MISSING** (E6 not built) | unknown |
| claim/overlap/graph/cache bundle | E6 vs E7 | **MISSING** (E6 not built) | unknown |
| per-layer sync cost inside E4 | B1 vs B2 (E5) | scaffold only | unknown |
| admission / request-aware batching | E7 with admission ablated (no engine feedback) | MISSING (RQ3 item) | unknown |
| resident-tail cost of E7 | powered campaigns | SEALED | queue: residents 0.78 (better); capacity: 1.0951 |

The first row is the project's strongest under-stated result: most
readers will assume a selective-KV system wins by moving fewer bytes;
we hold a sealed same-revision ablation showing the bytes account for
none of it here. The two MISSING middle rows are the paper's remaining
causal exposure, and both are closed by one build: E6.

## 4. E6: GPU selection + conventional gather (design)

Purpose: identical demand (same selector, envelopes, budget, refresh,
seeds), identical transfer primitive, identical consumer kernels —
execution stripped to "select, gather, then run":

- per-layer device selection exactly as E7 (same scores, same top-k);
- prep + transfer through the SAME indexed kernels;
- then a stream synchronize before attention — no wavefront overlap;
- bounded-cache REUSE disabled: every refresh re-stages the full
  selected set (cache directory cleared per refresh), so cross-refresh
  row reuse contributes nothing;
- tiered decode runs eager (NTA_SGLANG_TIERED_GRAPH=0) — conventional
  gather breaks graph replay by construction, which is part of what E7
  buys and therefore must be part of what E6 lacks;
- claims retained ONLY as memory-safety lifecycle (allocation and
  fenced free are not the mechanism under test; removing them would
  change safety, not execution).

Gate: `NTA_SGLANG_EXECUTION_PROTOCOL=conventional` (default absent =
E7). Campaign: powered capacity + queue shapes, registered seeds,
paired against banked E7 and B1 qualifications. Interpretation rule,
fixed now: E6 is diagnostic; whatever it shows is reported as the
decomposition, including the uncomfortable outcomes (if E6 ~= E7, the
claim/overlap/graph bundle is thin and the paper says the win is
GPU-side selection escaping the round trip; if E6 ~= B1, the round trip
was never the cost and the bundle is everything).

## 5. Story placement decisions (per the external review, adopted)

- **exact / top-k / native-sparse are demand sources, not three
  mechanisms.** The paper states: NTA does not decide which pages
  matter; it governs how execution proceeds once demand exists.
  Quality is reported per demand source (top-k at the measured parity
  budget 384 for dispersed-evidence tasks; synthetic matrix demoted to
  smoke evidence).
- **MoE is contract-generalization evidence** ("the same lease contract
  serves another external object class at operator level"), never a KV
  result and never a headline number, until a production
  expert-offload integration exists.
- **The resume/ticket machinery is the operator-level result plus
  mechanism evidence** (TIER_STREAMING numbers, replay battery, ABI
  stress). Pursuing serving-path completion-driven resume as the
  headline is explicitly declined: 2A measured zero reclaimable
  barrier stall in this family, and P2/P3 recorded the failure modes
  of admission- and waiting-centric framings. The paper's honest
  sentence: demand is BOUND late (discovered on device during
  execution); requests are not SUSPENDED late.
- RQ0 honesty: the 2A zero-stall negative and the P2/P3 negatives are
  presented as the boundary evidence they are.

## 6. What "implement everything first" would mean, and why not

The unimplemented remainder of the original thesis is serving-path
contributor resume. Three sealed negatives (2A no-stall, P2 inversion,
P3 host-bound ITL) say that machinery has no measurable headroom in
this workload family on this hardware. Building it anyway, to justify
a story, inverts the method that produced every credible number here.
The implemented-and-measured mechanism is the paper; E6 is what makes
its causality airtight.
