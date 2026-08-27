# Experiment design

Status: canonical evaluation design for the refactored project.

Experiment drivers and artifact validation live under `experiments/` and
`benchmarks/`; implementation remains under `include/`, `lib/`, `runtime/`,
and `python/nta_runtime/`. Correctness tests under `tests/` do not own the
experiment runner or persistent result files. Use
`docs/ARTIFACT_EVALUATION.md` for the reproducible bundle workflow.

The central question is whether one exact, late-bound work-unit mechanism
improves execution when a batch contains heterogeneous request state. Every
arm consumes one demand trace and the same exact consumed identities. Demand
construction is an input/workload property; selector quality is outside the
serving claim.

## RQ0: map the opportunity before serving

Use `experiments/run_work_unit_matrix.py` to sweep:

- candidate and consumed units;
- resident/ready/blocked/new-arrival fractions;
- availability latency and skew;
- compute per unit and transport bandwidth;
- granularity and staging capacity.

This runner validates the real WorkBatch, WorkLedger, generation checks, epoch
checks, partial transitions, and byte accounting. It is a contract/regime
runner, not a GPU performance result.

The default command emits the full mechanism. `--ablation all` emits every
declared RQ3 ablation as well. `experiments/validate_matrix_artifact.py`
checks that each record has a shared exact-trace hash, a valid activation
counter set, and internally consistent finite-window blocked-cohort
accounting. Its modeled timing fields are explicitly marked
`synthetic_regime_contract`; they cannot be used as serving speedup or
stationary queueing evidence.

The matrix includes the evaluation tiers HBM, host memory, NVMe, and DAX. Host
memory's direct/staged implementation detail is selected inside the native
transport configuration; the tier axis never changes exact demand IDs or
the numerical workload.

For each synthetic stratum, all initially blocked work units enter one finite
cohort and are released at declared uniform interval midpoints over the modeled
availability window. The runner emits the cohort count, window, integrated
pending-unit area, release rate, mean occupancy, mean residence time, and the
release-process name. The validator recomputes these deterministic quantities,
but deliberately emits no Little's-law residual: deriving `L`, `lambda`, and
`W` from the same cohort makes the identity true by construction and is not
queueing evidence. Serving-side Little's-law fields instead use measured client
timestamps and disclose their observable queue scope.

## RQ1: heterogeneous exact serving

The primary workload is one continuous batch containing:

- resident requests;
- external requests whose rows are ready;
- external requests whose rows are blocked;
- newly arriving requests;
- cancellation and request-slot reuse.

The exact demand trace is generated once and replayed by every arm. The primary
serving comparison is dense exact attention plus exact demand with the same
demand IDs, bytes, page order, and output checks.

The SGLang harness is
`benchmarks/serving/CompareSglangHiCacheLoad.py`. It validates placement,
fallback freedom, exact attention accounting, compiler contracts, external
attention coverage, and mechanism counters before reporting SLO/goodput.

## RQ2: causal decomposition

Use matched arms:

```
B0  resident exact conventional baseline
B1  host promotion + batch readiness barrier
B2  host demand materialization + conventional exact gather
B3  device demand + conventional exact gather
B4  device demand + late-bound exact staging
B5  heterogeneous bounded work-unit execution
B6  exact partial consumer continuation
```

B0--B3 isolate residency, host materialization, and device demand while
retaining the same exact demand IDs. B4--B5 isolate the execution-side
contribution. B6 is optional evidence for the general partial protocol and is
not required for the serving headline. B0 is resident exact execution, not a
dense numerical workload, so the matched-demand fairness rule remains valid.

The full paired specification also includes two non-adjacent, pre-registered
comparisons: B3 versus B1 isolates device-side selection from the host-control
round trip, and B5 versus B3 measures the complete device-demand to
heterogeneous-work-unit mechanism jump. These are required because adjacent
arms alone cannot seal the causal decomposition.

All arms report:

- selected and candidate units/bytes;
- host round trips and device launches;
- runnable/blocked exposure and state time;
- staging high-water and physical bytes moved;
- granularity, group count, control overhead, and compute;
- request-generation and epoch rejection counts;
- TTFT, TPOT, ITL, tail SLO, throughput, and goodput.

## RQ3: mechanism ablations

Disable exactly one boundary at a time. The executable names are the same as
the manifest's `ablations` field and can be run with `--ablation all`:

1. host-side demand/control instead of device demand;
2. batch readiness instead of work-unit readiness;
3. coarse layer grouping instead of the chosen granularity;
4. manual work mapping instead of compiler-generated coordinates;
5. generation/epoch checks on the hot path, with a shadow-only comparison;
6. engine admission feedback;
7. bounded staging, replaced by full promotion.

Each record includes `execution_mode` and `activation_counters`. An ablation is
invalid if it is marked applied but the counter for the disabled or replacement
boundary is zero. The validator only applies this rule to arms for which the
manifest declares that ablation applicable; unrelated arms remain useful
matched controls.

## RQ4: strata and robustness

Report results by explicit strata rather than one average:

- request state: resident, ready external, blocked external, arrival;
- granularity: request, layer, page group, CTA tile;
- availability skew: low, medium, high;
- staging pressure: under-capacity, near-capacity, over-capacity;
- compute/transfer ratio: control-dominated, balanced, compute-dominated.

The dependency-free runner emits these strata fields and finite-cohort
accounting residuals. Its `arrival` and `load_ratio` labels are synthetic
strata derived from availability skew and the modeled
compute/transfer/control split; they do not claim a measured arrival process.
Serving runners must preserve the same field names and add measured GPU/engine
counters; synthetic records must never be presented as serving speedups. The
serving harness is the source of timing truth for RQ1/RQ2; the dependency-free
artifact is the source of contract, fairness, and regime truth.

A result is interpretable only when the workload stratum, demand trace,
protocol, and mechanism counters are included in the artifact.

## Fairness

No arm may change demand IDs, page order, cache placement, numerical output
contract, or request trace. A different demand trace is a different workload,
not an ablation of execution.
The work-unit matrix and serving harness must be run from a clean revision,
and artifacts must contain that revision and complete activation metadata.
