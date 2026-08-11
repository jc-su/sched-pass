# Pre-Registered Evaluation Protocol (RQ1 headline)

Written before any qualifying run, per the project's discipline: a failed
gate is recorded, not retried until it passes, and operating points are
not shopped after the fact. Changes to this file after the first
qualifying trial must be recorded as amendments with reasons, never
silent edits.

## Claim under test (RQ1)

Under HBM pressure from held long-context working sets, serving
model-selected external KV through bounded staging improves SLO-qualified
goodput over dense HiCache serving at task-quality parity, without
degrading resident-request tails.

## Operating points

- **P1 (primary):** Qwen2.5-3B, 16 external requests x 16,384-token
  host-cached prefixes, 256 output tokens each; 1 resident request x
  2,048 tokens x 256 outputs; device pool capped at 110,000 tokens
  (dense demand ~2.4x pool); arrivals Poisson at 12/s, seed-derived;
  churn per the load harness's eviction guarantee; batch mode separate.
- **P2 (scale variant):** identical except 32 externals (dense demand
  ~4.9x pool). Run only if P1 completes cleanly at n=10.
- Budget and refresh interval for the tiered arm are fixed by the
  quality matrix's output (task #30) before the first qualifying trial;
  the matrix's chosen point is recorded here by amendment, not chosen
  from performance results.

## Arms

- **stock:** unmodified SGLang FlashInfer HiCache.
- **tiered:** NTA selected serving; the run is invalid unless the
  attestation gate passes (zero stock-attention launches, zero HiCache
  fallback, external claims and bounded staging active, device
  compaction active).
- Identical model snapshot, arrival trace, seeds, pool cap, and revision
  across arms within a trial.

## Metrics and SLO definitions

- **SLO-qualified goodput (primary):** completed requests per second
  whose TTFT <= 8.0s and whose P99 ITL <= 100ms, measured over the timed
  phase. External and resident requests both count; a request violating
  either threshold contributes zero.
- Secondary: external TTFT p50/p95, resident P99 ITL, external mean
  TPOT, makespan, physical bytes moved (staging copies + summary scan +
  completion), HBM high-water (dense-equivalent vs staging), admission
  concurrency.

## Bars (all must hold to claim RQ1)

1. Tiered SLO-goodput geomean >= **1.5x** stock at P1, with the 95%
   bootstrap CI excluding 1.0.
2. Resident P99 ITL geomean <= **1.05x** stock (current single-trial
   evidence is 0.47x; the bar is deliberately conservative).
3. Task-quality parity at the chosen budget per the quality matrix's
   pre-declared thresholds.
4. Physical bytes moved (all components) below the stock arm's.
5. Mechanism attestation green on every counted trial.

## Trials, seeds, exclusions

- n = 10 trials per arm per point; seeds 20260901..20260910.
- Clean committed revision; GPU clocks recorded before each trial; no
  co-resident GPU processes (checked before each trial; a contended
  trial is discarded with its contention evidence saved, and rerun).
- A trial that crashes is counted as a failure of its arm unless the
  crash is traced to harness or environment defects, in which case the
  defect is fixed, the fact recorded, and the full n=10 restarted.
- No metric-based exclusion of completed trials, ever.

## Status

- Pre-fix baseline (single trials, committed): tiered TTFT 0.76x,
  resident P99 0.47x, TPOT 2.5x (host-bound, diagnosis confirmed three
  ways), makespan 1.65x. The graph-capturable multi-claim operator
  (design of record, its own exit gate: TPOT <= 1.2x stock, GPU busy
  > 85%) must land before qualifying trials begin.
- **Amendment 1 (2026-08-11):** the quality matrix (needle + multikey
  kinds, count demoted diagnostic-only by stock validation; budgets
  {32, 64, 128} x refresh {1, 1024}) passed every cell at 1.0, equal to
  stock, on the synthetic retrieval-plus-aggregation battery
  (`results/serving/quality-matrix/`). The tiered arm's qualifying
  configuration is therefore fixed at **budget 64, refresh interval
  1024** — chosen by performance among quality-parity points, per
  protocol. The matrix also exposed and fixed a sidecar defect (sub-page
  host hits claimed as external prefixes, suppressing radix insertion
  workload-wide; claims now floor at page size, commit 16a32b1).
