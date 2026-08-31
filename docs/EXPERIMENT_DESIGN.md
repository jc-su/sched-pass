# Experiment design

Status: canonical evaluation design for the refactored project.

Experiment drivers and artifact validation live under `experiments/` and
`benchmarks/`; implementation remains under `include/`, `lib/`, `runtime/`,
and `python/nta_runtime/`. Correctness tests under `tests/` do not own the
experiment runner or persistent result files. Use
`docs/ARTIFACT_EVALUATION.md` for the reproducible bundle workflow.

The central question is whether one exact, late-bound acquisition mechanism
improves execution when a batch contains heterogeneous request state. Every
arm consumes one demand trace and the same exact consumed identities. Demand
construction is an input/workload property; selector quality is outside the
serving claim. "Exact selected demand" may contain every candidate block, so
the mechanism is not defined by sparse or approximate attention. Select-then-
compute workloads are an important high-opportunity stratum, not a different
system policy.

## RQ0: map the opportunity before serving

Use `experiments/run_work_unit_matrix.py` to sweep the D0--D6 diagnostic
profiles:

- candidate and consumed units;
- resident/ready/blocked/new-arrival fractions;
- availability latency and skew;
- compute per unit and transport bandwidth;
- granularity and staging capacity.

The D namespace is intentionally not the serving-arm namespace. This runner
validates the real WorkBatch, WorkLedger, generation checks, epoch
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
queueing evidence. Serving reports likewise expose measured client arrival,
departure, residence, and occupancy fields only as descriptive finite-window
accounting, with their observable scope stated explicitly.

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

The formal SGLang worker is
`benchmarks/serving/RunSglangEvaluationArm.py`. It runs one arm per process and
validates placement, fallback freedom, exact attention accounting, compiler
contracts, external attention coverage, and result-derived mechanism counters
before its report enters the randomized paired runner. The older nested
comparison driver is diagnostic only.

## RQ2: causal decomposition

Use four matched serving arms:

```
A0  framework bulk control (stock SGLang HiCache)
A1  exact NTA preacquisition + stock numerical consumer
A2  scheduler-bound exact acquisition + whole-layer stock consumer
A3  the same acquisition path + progressive heterogeneous work-unit consumer
```

A1/A0 measures the exact acquisition boundary. A2/A1 measures moving ownership
from eager lease capture to the scheduler-shape edge without changing the
whole-layer numerical consumer. A3/A2 changes only the consumer to event-wave
partial readiness. Transport engine,
frontier depth, granularity, tier, and request heterogeneity are orthogonal
sweeps; they do not create additional headline mechanisms.

Device-discovered bulk execution remains a diagnostic negative control. It is
not a canonical arm because it uses a different acquisition owner from the
proactive A3 path and would therefore confound A3/A2.

For a 36-layer model, an observation of frontier depth 1 means all 36 layers
used the exact external/preacquired contract, but only the first layer reached
attention before its acquisition completed; the remaining 35 consumed already
ready data. It is neither "1/36 support" nor evidence that deep frontier
behavior matters. The evaluation must therefore sweep controlled frontier
depth (including 0, 1, intermediate values, and all layers) by changing the
transfer/compute ratio, contention, context length, and batch heterogeneity.

All arms report:

- selected and candidate units/bytes;
- host round trips and device launches;
- runnable/blocked exposure and state time;
- staging high-water and physical bytes moved;
- granularity, group count, control overhead, and compute;
- request-generation and epoch rejection counts;
- TTFT, TPOT, ITL, tail SLO, throughput, and goodput.

TTFT is a latency component, not the acceptance criterion.  The primary
loaded-serving metric is joint TTFT/TPOT/p99-ITL SLO-goodput at a stock-only
pre-frozen overload point.  Output-token throughput is reported globally and
by resident/external cohort.  Resident p95 TPOT, resident p99 ITL, and resident
throughput are explicit no-regression controls, so faster external prefill
cannot hide damage to decode service.

## RQ3: diagnostic interventions and robustness

The D0--D6 synthetic profiles can disable exactly one boundary at a time. The
executable names are the same as the diagnostic manifest's `ablations` field
and can be run with `--ablation all`:

1. host-side demand/control instead of device demand;
2. batch readiness instead of work-unit readiness;
3. coarse layer grouping instead of the chosen granularity;
4. manual work mapping instead of compiler-generated coordinates;
5. generation/epoch checks on the hot path, with a shadow-only comparison;
6. engine admission feedback;
7. bounded staging, replaced by full promotion.

Each record includes `execution_mode` and `activation_counters`. An ablation is
invalid if it is marked applied but the counter for the disabled or replacement
boundary is zero. The validator only applies this rule to profiles for which
the manifest declares that ablation applicable. These records prove contract
activation, not serving causality or speedup. Formal serving causality comes
from A0--A3.

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

GPU power and temperature are nuisance variables, not deployment
preconditions or scheduler inputs. A formal paired campaign records one
administrator-selected power limit, begins both arms below the same thermal
bound, rejects foreign GPU processes and actual thermal slowdown, and reports
the observed clock/power range. It does not require a fixed runtime clock or
claim that production holds temperature constant. The production selector
adapts from measured transfer and compute observations; power-limit
sensitivity is a separate robustness experiment and must use identical limits
for stock and NTA.

Likewise, the causal A1--A3 arms fix one common transport engine only to
separate acquisition and consumer effects. Natural-trace full-system runs must
also evaluate the production AUTO selector. Neither selector reads temperature
or power; those values remain artifact telemetry and trial-validity evidence.
