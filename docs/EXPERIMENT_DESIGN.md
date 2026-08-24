# Experiment Design

Status: source of truth for the refactor evaluation

The old result files remain useful for debugging and regression detection, but
they do not answer the redesigned research questions.  Every new artifact
must record the commit, protocol, demand semantics, granularity, workload
trace, and activation counters.

## RQ0 — Where can the mechanism win?

Map the regime before running long serving campaigns.  Sweep:

- candidate units and selected units;
- request batch size and external/resident fraction;
- availability latency and skew;
- compute per work unit;
- work-unit granularity;
- staging capacity and transport bandwidth.

The output is a crossover map, not a single speedup.  A regime is eligible
only when the measured saved waiting/transfer time can exceed selection,
staging, launch, and bookkeeping overhead.

## RQ1 — Does heterogeneous late binding improve end-to-end serving?

Use one exact demand trace across all arms.  The primary shape must contain, in
one continuous batch, resident, ready external, blocked external, and newly
arriving requests.  Include cancellation and request-slot reuse.

The primary numerical contract is exact sparse demand: the sparse mask is
provided by the workload and all arms compute that mask exactly.  Dense exact
attention is a no-regression control.  Top-k/DSA selectors are separate
quality-gated experiments, not the main system claim.

## RQ2 — Which part of the mechanism matters?

Required arms:

```text
B0  dense resident, no nonresident dependency
B1  host promotion + batch barrier
B2  host split/rebatch or layer barrier
B3  device demand + conventional gather (E6)
B4  device demand + late-bound staging, no partial consumer
B5  full work-unit protocol with heterogeneous ready/blocked execution
B6  full protocol with exact partial consumer and continuation
```

All arms use the same request trace, exact demand IDs, selected bytes, output
contract, and transport delay distribution.  B3/B4 isolate demand routing and
staging.  B5 isolates heterogeneous execution.  B6 measures the optional
partial form rather than contaminating the serving headline.

## RQ3 — Is co-design necessary?

Use controlled ablations:

1. host demand materialization instead of device demand;
2. whole-batch readiness instead of work-unit readiness;
3. coarse layer grouping instead of the selected granularity;
4. compiler-generated work mapping replaced by a manually prepared mapping;
5. request-generation checks removed from the performance path but retained in
   a fail-closed shadow validator;
6. engine feedback/admission disabled;
7. bounded staging replaced by unbounded promotion.

The result must report both performance and the activated mechanism counters.
An ablation that does not activate the intended path is invalid, not a
negative result.

## RQ4 — Robustness

Run nonstationary traces with changing skew, batch composition, demand
selectivity, and staging pressure.  Compare the online granularity choice with
fixed coarse/fine choices and a hindsight oracle.  Measure:

- throughput and SLO goodput;
- TTFT, TPOT, and tail ITL;
- useful bytes/s and physical bytes moved;
- runnable/pending work and time-to-runnable;
- staging high-water and claim occupancy;
- selector, protocol, and adapter overhead;
- quality/error only where demand is approximate.

## Fairness and artifact rules

- No arm may use a different demand mask or page order.
- Approximate demand must declare its quality target and cannot be called
  exact because the output happened to match on a smoke workload.
- Every arm must state whether graphs, overlap, cache reuse, and partial state
  are enabled.
- Validators must check the contract appropriate to the arm; an E7 validator
  cannot reject B1 merely because B1 has no device-compaction counter.
- A clean worktree and exact revision are mandatory for qualification.
