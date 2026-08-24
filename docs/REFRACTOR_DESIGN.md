# Late-Bound Heterogeneous Execution

Status: source of truth for `refactor/late-bound-work-unit`

This document is written from the current code and tests.  Older design notes
and result ledgers are evidence archives; they do not define the refactor.

## 1. One mechanism

The project implements one mechanism:

> Bind each heterogeneous work unit to its request generation, demand, and
> availability as late as correctness permits, then execute bounded runnable
> groups at a granularity chosen from measured transfer, compute, control, and
> availability costs.

The mechanism is not “GPU top-k”, “host-round-trip elimination”, or
“completion resume” individually.  Those are demand sources or execution
forms.  The direct, late-bound, and partial paths share the same work-unit
contract:

```text
unbound -> blocked/ready -> running -> partial/complete -> retired
```

`partial` is a state transition, not a second scheduler.  A direct launch is
the special case in which all work units are ready before execution.

## 2. Contract layers

The native ABI remains the storage contract.  `WorkItem`, `WorkDependency`,
and `RequestProgress` in `include/nta/RuntimeABI.h` are canonical and must not
be duplicated.  The Python semantic contract in
`python/nta_runtime/work_unit.py` adds the concepts that the ABI intentionally
does not encode as policy:

```text
RequestBinding       request slot + generation + request identity
DemandDescriptor     candidate/selected units, bytes, semantics, epoch
WorkUnit              binding + layer/range + dependencies + reduction identity
WorkBatch             one heterogeneous epoch with one execution granularity
Availability          unbound, blocked, ready, running, partial, terminal
ExecutionProtocol     conventional, late-bound, or partial
```

The contract has three non-negotiable rules:

1. A slot/generation mismatch rejects a transition and cannot publish stale
   work.
2. An epoch mismatch rejects demand or completion publication.
3. Approximate demand is explicit.  An approximate selector cannot silently
   satisfy an exact numerical protocol.

## 3. Ownership

```text
SGLang / vLLM adapter
  owns request IDs, scheduler slots, page/block tables, cancellation, graphs

Demand provider
  owns exact sparse masks or experimental selectors

Protocol planner
  owns granularity, runnable grouping, overlap, and partial-form choice

NTA runtime core
  owns generation validation, claims, dependency state, staging capacity,
  completion, retirement, and telemetry

Transport
  owns placement-specific registration, submission, completion, and recovery

Compiler plugin
  owns marker validation, work mapping, and operator contracts

Consumer operator
  owns numerical partial state and exact merge semantics
```

The runtime core must not branch on SGLang or vLLM.  An adapter may translate
engine metadata into the common contract, but it may not implement a second
claim table, generation tracker, or staging policy.

## 4. Granularity

Granularity is a first-class cost decision, not a collection of unrelated
scheduling strategies.  The planner compares candidate group sizes using
measured parameters:

```text
total(group) = transfer + compute + group_control + availability_exposure
```

Fine groups reduce exposure to availability skew but increase control and
launch cost.  Coarse groups amortize control but recreate a barrier.  The
planner in `python/nta_runtime/execution_protocol.py` is intentionally
transparent and calibrated; it is not a quality oracle.

The implementation must record, per epoch:

- selected and candidate units/bytes;
- group size and group count;
- ready, blocked, partial, and complete work;
- selector, staging, transfer, control, and compute time;
- staging high-water and reclaimed capacity;
- request-generation and epoch rejection counts.

## 5. Protocol forms

### Conventional

Select/gather/compute with a batch-level readiness boundary.  This is a
diagnostic baseline, not the contribution.  It must use the same demand and
selected identities as the late-bound form.

### Late-bound

Demand and availability are bound at work-unit granularity.  Ready units may
be grouped and launched while blocked units remain pending.  It does not
require partial numerical state if the consumer can only consume complete
groups.

### Partial

The consumer publishes an exact partial state and later merges only the
current-generation contributors.  It is the general form for true
completion-driven continuation.  It must not be enabled in serving merely to
make a resume counter non-zero; the workload must expose a real critical-path
opportunity.

## 6. Current migration status

Implemented on this branch:

- semantic work-unit and demand contract;
- generation/epoch-checked work ledger;
- explicit granularity cost model;
- common request-identity adapter base;
- SGLang identity adapter;
- vLLM dependency-free adapter seam;
- protocol configuration validation.

Still to migrate:

- route SGLang staging and selection through `DemandDescriptor` and
  `ExecutionProtocolConfig` rather than local flags;
- port the conventional E6 baseline into this branch and validate it against
  the same demand trace;
- move claim/staging state out of the monolithic SGLang backend;
- connect compiler work mapping to the semantic `WorkBatch` contract;
- implement a real vLLM adapter against one pinned vLLM version;
- replace protocol-specific benchmark validators with contract validators.

The E6 implementation in the separate `e6-conventional` worktree is not
merged automatically.  It is an experimental reference until its fairness
and campaign artifacts are reviewed.

## 7. Engineering acceptance gates

The refactor is complete only when:

1. Existing ABI, compiler, runtime, and SGLang tests remain green.
2. Conventional, late-bound, and partial forms share the same work-unit
   identity and demand trace.
3. A stale request generation and stale epoch fail closed in every adapter and
   protocol test.
4. Protocol validators do not require E7-only counters for B1 or E6.
5. The default SGLang path does not enable approximate demand without an
   explicit quality contract.
6. A small-demand regime can automatically choose conventional execution when
   control cost exceeds saved transfer or overlap benefit.
7. The vLLM adapter passes the same identity, cancellation, and contract tests
   without importing SGLang.
