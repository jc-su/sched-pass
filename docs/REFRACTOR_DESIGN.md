# Refactor design: one execution mechanism

Status: canonical design for `refactor/late-bound-work-unit`.

## 1. Mechanism

NTA binds a work unit to three facts as late as correctness permits:

1. request identity: engine slot, request ID, and generation;
2. exact demand: candidate/selected units, byte size, granularity, and epoch;
3. availability: blocked, ready, running, partial, or terminal.

The protocol then launches bounded groups of ready work. Granularity is chosen
from transfer, compute, control, and availability-exposure costs. Conventional,
late-bound, and exact-partial execution are protocol forms of this mechanism;
they are not separate schedulers.

Demand is always exact in the serving contract. A provider supplies the IDs
that the numerical consumer will actually use; selection quality is therefore
not hidden inside the execution mechanism.

## 2. Ownership

```
SGLang / vLLM adapter
  request IDs, engine slots, cancellation, graph metadata, block/page tables

Work-unit core
  identity-bound demand, availability, heterogeneous batch validation

Protocol planner
  granularity, bounded groups, overlap and partial-form configuration

NTA runtime/native bridge
  generation/epoch validation, dependency records, staging/transport ABI,
  completion and telemetry

Compiler plugin
  marker validation, operator mapping, native work coordinates and contracts

Consumer operator
  exact numerical computation and exact partial-state merge
```

No engine adapter creates a second generation table. No compiler pass guesses
request identity. No serving path silently changes the numerical demand
contract.

## 3. Execution flow

For SGLang:

```
scheduler forward
  -> SglangAdapter: rids + real request-pool slots
  -> SglangHiCacheBridge: owns external host load and staging lease
  -> _ActiveBatch: schedules and page mappings
  -> ExecutionSession: one WorkBatch per real attention launch
  -> DeviceWorkPlan.upload_work_units: checked native WorkItems
  -> compiler-instrumented FlashInfer wrapper
  -> session completion + HiCache layer retirement
```

The session is created immediately before each actual attention launch, using
the semantic model layer and current KV-cache geometry. This avoids the prior
error of treating reusable wrapper positions as model layers. Every native
plan unit is looked up by layer, logical tile, and request index and must match
the session's request binding before upload.

SGLang external prefixes are exact: they stage the host rows needed by the
provided page map. Batches may contain resident, ready external, blocked
external, and new requests simultaneously. The admission hook only controls
when an external batch enters the engine; it does not own a duplicate work
ledger.

## 4. Identity and lifetime invariants

- A request-slot reuse increments generation.
- A request generation mismatch rejects native/runtime publication.
- A demand epoch mismatch rejects upload or completion.
- Work IDs are unique within one heterogeneous batch.
- A request index cannot refer to two generations in one batch.
- Exact sparse demand names its selected IDs; exact dense demand names all
  candidates.
- A canceled or finished external prefix releases its staging lease only
  after the owning stream completion edge.

Graph replay preserves request IDs and request-pool slots. A graph path with
missing identity fails closed; a non-contiguous slot layout is rejected by the
graph consumer instead of being mapped to a false contiguous range.

## 5. Granularity

The planner treats granularity as a cost decision:

```
T(group) =
  transfer(group) + compute(group) + control(group)
  + availability_exposure(group)
```

Fine groups reduce exposure to skew but increase launches and bookkeeping.
Coarse groups amortize control but recreate a barrier. The planner's
parameters are hardware/transport measurements, not selector quality.

The implementation records selected/candidate units and bytes, group counts,
availability states, request/epoch rejections, staging capacity, and native
upload counts so the tradeoff can be audited.

## 6. Protocol forms

- **Conventional**: all work must be ready before launch. It is the dense or
  batch-barrier baseline using the same demand trace.
- **Late-bound**: ready work is exposed and bounded while blocked work remains
  pending. This is the serving protocol.
- **Partial**: an exact consumer may publish a partial state and continue it
  later. It is an optional contract-level form and must be used only when the
  workload exposes a real critical-path continuation.

No protocol form is a selector. The current serving claim is about execution
coordination under exact demand.

## 7. Engineering boundaries

The SGLang implementation is pinned to the tested framework version and FA2
FlashInfer path. The vLLM boundary is intentionally structural and
dependency-free: `VllmSchedulerProjection` is the only expected projection
from a pinned vLLM scheduler integration. A future vLLM adapter can change
only that projection and transport binding, not the work-unit core.

Retired selector, tiering, and capture-specialization modules are not
compatibility layers and are not part of this branch. This keeps the active
codebase aligned with the exact system mechanism.
