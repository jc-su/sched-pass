# Refactor design: one execution mechanism

Status: canonical design for `refactor/late-bound-work-unit`.

## 1. Mechanism

NTA binds a work unit to three facts as late as correctness permits:

1. request identity: engine slot, request ID, and generation;
2. exact demand: candidate/consumed units, byte size, granularity, and epoch;
3. availability: blocked, ready, running, partial, or terminal.

The protocol then launches bounded groups of ready work. Granularity is chosen
from transfer, compute, control, and availability-exposure costs. Conventional,
late-bound, and exact-partial execution are protocol forms of this mechanism;
they are not separate schedulers.

Demand is always exact in the serving contract. A provider supplies the IDs
that the numerical consumer will actually use; there is no hidden quality
selector inside the execution mechanism.

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
  typed contract verification, structural access proof, marker lowering, and
  native work coordinates

Consumer operator
  exact numerical computation and exact partial-state merge
```

No engine adapter creates a second generation table. The compiler pass does
not guess request identity from a raw pointer: it proves structural access
shape and lowers typed markers only when the module contract supplies
generation, exact-demand, and tier-ownership guarantees. No serving path
silently changes the numerical demand contract.

Resource ownership is split into three explicit domains. The transport/control
owner drives the backend protocol, the runtime owns the device directory, and
`allocation_owners` records who may allocate or lend payload memory. This
distinction matters for host-staged transfers: the native object API may own
the HBM staging destination, while an engine adapter may register an existing
staging tensor. NVMe and CXL-DAX mappings remain transport-owned; their
steady-state path never includes a host proxy or a per-request control ioctl.

## 3. Execution flow

For SGLang:

```
scheduler forward
  -> SglangAdapter: rids + real request-pool slots
  -> SglangHiCacheBridge: owns exact host load, physical page mapping, and fence
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

SGLang external pages remain ordinary physical page-table entries. The bridge
owns the host source rows until the last layer's CUDA completion edge and
publishes the exact page mapping to the work-unit planner. A batch may contain
resident, externally-ready, externally-blocked, and new requests simultaneously.
There is no virtual-prefix sidecar in the active path: a bounded sidecar would
need a consumer that translates virtual pages and performs an exact partial
attention merge, which is a separate implementation contract.

## 4. Identity and lifetime invariants

- A request-slot reuse increments generation.
- A request generation mismatch rejects native/runtime publication.
- A demand epoch mismatch rejects upload or completion.
- Work IDs are unique within one heterogeneous batch.
- A request index cannot refer to two generations in one batch.
- Exact demand identifies the units consumed by the numerical operator; dense
  demand identifies every candidate.
- A canceled or finished HiCache load releases its physical source ownership
  only after the owning stream completion edge.

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

## 7. Tier and compiler contract

The C++ `TierDescriptor`, device `BackendView` capability mask, C runtime API,
and Python `TierDescriptor` are one ABI-level tier contract. HBM,
HostMapped, HostStaged, NVMe, and CXL DAX are backend classes, not scheduling
policies. NVMe is device-initiated transport; CXL DAX is a bounded
device-visible mapped replica. Hardware-specific qualification is explicit
and fail-closed.

The NVMe implementation has two separate lifecycle owners. The control plane
owns VFIO attachment, controller reset, queue creation, BAR/doorbell state, and
queue quiescence. `VfioNvmeMappingBackend` owns IOMMUFD/peer mapping tokens and
their release. They share only an explicit mapping context; mapping teardown
occurs after queue quiescence and before the IOMMU domain is destroyed. Mapping
and page-list publication are setup operations. The GPU queue consumes the
immutable page list in steady state and does not issue a per-request mapping
operation. Physical replacement may republish only the source-range directory
entry; the runtime reuses a slot-owned HBM/DMA allocation whenever the new
exact extent fits, and allocates/maps only when capacity is insufficient.

Non-owning HostRuntime registrations are setup-time validated CUDA views, not
untyped lifetime claims: HBM, mapped-host, device-visible staged sources, HBM
staging destinations, and indexed arrays are checked against the selected CUDA
device before their addresses enter the device directory. The caller still
owns their storage until the documented acquisition/stream completion edge.

The native request identity uses a compact uint64 value, but collision
provenance is retained only for active slots. Retired spellings are removed on
cancellation or replacement; slot generations reject stale device work. This
keeps collision defense fail-closed without making long-running serving memory
grow with historical request cardinality.

`python/nta_runtime/resource_contract.py` is the shared typed resource
contract for HBM, mapped host, host-staged, NVMe, and CXL-DAX. It records
capabilities, owner, setup requirements, and the steady-state path. The
framework-facing tier service and native `TierDescriptor` use the same
capability vocabulary; no adapter is allowed to infer a host proxy from a
physical-tier failure.

Tier latency and bandwidth are startup calibration inputs rather than hidden
compile-time truths. Native descriptors accept
`NTA_TIER_<HBM|HOST_MAPPED|HOST_STAGED|NVME|CXL|RDMA>_LATENCY_NS` and
`NTA_TIER_<...>_BANDWIDTH_BPS`; absent values use conservative defaults. A
qualification artifact should export the measured values for the same machine
and tier, so the planner and native admission directory are not silently using
different hardware assumptions.

The JIT `OperatorContract` carries four non-inferable semantic obligations:
request-slot/generation identity, exact work-unit demand, typed access proof,
and tier ownership. `TypedInstrumentation.cuh` anchors those values in the
compiled module so `NtaPass` validates the same code that exports the JIT
contract. Raw unmarked kernels remain diagnostic-only.

## 8. Engineering boundaries

The SGLang implementation is pinned to the tested framework version and FA2
FlashInfer path. The vLLM boundary is intentionally structural and
dependency-free: `VllmSchedulerProjection` is the sole framework projection,
and `EngineBoundary` is the common lifecycle interface. Framework-specific
code can change that projection and transport binding, not the work-unit core
or native ABI.

The vLLM projection is executable only when it supplies exact block tables and
page bytes. An identity-only projection is intentionally rejected by
`bind_forward`; it is a contract test seam, not end-to-end serving evidence.

There is no selector-policy taxonomy in the active runtime. Exact dense and
exact sparse demand are input semantics; conventional, late-bound, and
partial are protocol forms of the same mechanism.
