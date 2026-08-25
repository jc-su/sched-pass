# Architecture

This document is the short architectural entry point. The detailed contract
is in [REFRACTOR_DESIGN.md](REFRACTOR_DESIGN.md).

NTA has one execution mechanism: late-bound, generation-bound heterogeneous
work units. Each unit carries exact demand and availability; the protocol
launches bounded ready groups at a measured granularity.

```
engine adapter
  -> EngineBatch / RequestBinding
  -> ServingTierService / exact page catalog
  -> WorkBatch / DemandDescriptor
  -> ExecutionSession / WorkLedger
  -> DeviceWorkPlan semantic-to-native bridge
  -> typed LLVM lowering + contract-checked consumer
```

SGLang and vLLM are adapters, not runtime implementations. The native ABI
stores work and dependencies; the semantic layer validates generation, epoch,
demand, and availability before native submission.

The serving path is exact. Selection is an input trace, not a hidden runtime
policy. Conventional, late-bound, and exact-partial forms
share the same work-unit identity and demand trace.

Resident-only forwards take the framework's reference FlashInfer wrapper. This
is an intentional zero-regression boundary: the compiler/runtime mechanism is
entered only when the forward has an external-tier dependency. A mixed forward
still uses one NTA launch and carries resident and external work units together;
the reference path is not used to hide external work or to change demand.

The native runtime exposes one tier directory for HBM, mapped host memory,
host-staged memory, NVMe, and CXL DAX. The same descriptor is consumed by
host admission, device-visible backend metadata, and experiment telemetry.
The engine-neutral `ServingTierService` is the only serving attachment point:
host-staged uses indexed host objects, NVMe installs catalog-validated HBM
objects and runs the finite device progress loop, and CXL emits catalog-
validated direct dependencies. The Python resource contract makes protocol
ownership, allocation ownership, and runtime directory ownership explicit;
physical-tier configuration never silently
falls back to host data movement.
The NVMe production backend is GPU-controlled READ DMA directly into HBM
through a translated VFIO/IOMMUFD domain; host-mapped DMA is only its matched
baseline.
The LLVM pass uses structural pointer/load proofs only for access shape; the
typed operator contract supplies request-generation identity, exact demand,
and tier ownership. An unmarked or incomplete contract never authorizes a
raw pointer to become a transport request.

See [ENGINE_INTEGRATION.md](ENGINE_INTEGRATION.md) for framework boundaries
and [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) for evaluation rules.
