# Experiment design

Status: canonical evaluation design for the refactored project.

The central question is whether one exact, late-bound work-unit mechanism
improves execution when a batch contains heterogeneous request state. Every
arm consumes one demand trace and the same selected identities. Selection
quality is outside the serving claim.

## RQ0: map the opportunity before serving

Use `tools/experiments/run_work_unit_matrix.py` to sweep:

- candidate and selected units;
- resident/ready/blocked/new-arrival fractions;
- availability latency and skew;
- compute per unit and transport bandwidth;
- granularity and staging capacity.

This runner validates the real WorkBatch, WorkLedger, generation checks, epoch
checks, partial transitions, and byte accounting. It is a contract/regime
runner, not a GPU performance result.

Use Little's law as a consistency check:

```
L = lambda W
```

For each stratum, record admitted work rate `lambda`, mean number of pending
or runnable units `L`, and mean time in the corresponding state `W`. A
violation indicates dropped observations, an incorrectly defined population,
or a measurement boundary mismatch; it is not a performance conclusion by
itself.

## RQ1: heterogeneous exact serving

The primary workload is one continuous batch containing:

- resident requests;
- external requests whose rows are ready;
- external requests whose rows are blocked;
- newly arriving requests;
- cancellation and request-slot reuse.

The exact demand trace is generated once and replayed by every arm. The primary
serving comparison is dense exact attention plus exact sparse demand with the
same selected IDs, bytes, page order, and output checks.

The SGLang harness is
`benchmarks/serving/CompareSglangHiCacheLoad.py`. It validates placement,
fallback freedom, exact attention accounting, compiler contracts, external
attention coverage, and mechanism counters before reporting SLO/goodput.

## RQ2: causal decomposition

Use matched arms:

```
B0  resident dense conventional baseline
B1  host promotion + batch readiness barrier
B2  host demand materialization + conventional exact gather
B3  device demand + conventional exact gather
B4  device demand + late-bound exact staging
B5  heterogeneous bounded work-unit execution
B6  exact partial consumer continuation
```

B0--B3 isolate demand/transport/control effects. B4--B5 isolate the
execution-side contribution. B6 is optional evidence for the general partial
protocol and is not required for the serving headline.

All arms report:

- selected and candidate units/bytes;
- host round trips and device launches;
- runnable/blocked exposure and state time;
- staging high-water and physical bytes moved;
- granularity, group count, control overhead, and compute;
- request-generation and epoch rejection counts;
- TTFT, TPOT, ITL, tail SLO, throughput, and goodput.

## RQ3: mechanism ablations

Disable exactly one boundary at a time:

1. host-side demand/control instead of device demand;
2. batch readiness instead of work-unit readiness;
3. coarse layer grouping instead of the chosen granularity;
4. manual work mapping instead of compiler-generated coordinates;
5. generation/epoch checks on the hot path, with a shadow-only comparison;
6. engine admission feedback;
7. bounded staging, replaced by dense promotion.

An ablation is invalid if its activation counters show that the intended
boundary never executed.

## RQ4: strata and robustness

Report results by explicit strata rather than one average:

- request state: resident, ready external, blocked external, arrival;
- granularity: request, layer, page group, CTA tile;
- availability skew: low, medium, high;
- staging pressure: under-capacity, near-capacity, over-capacity;
- compute/transfer ratio: control-dominated, balanced, compute-dominated.

A result is interpretable only when the workload stratum, demand trace,
protocol, and mechanism counters are included in the artifact.

## Fairness

No arm may change demand IDs, page order, cache placement, numerical output
contract, or request trace. A different demand trace is a different workload,
not an ablation of execution.
The work-unit matrix and serving harness must be run from a clean revision,
and artifacts must contain that revision and complete activation metadata.
