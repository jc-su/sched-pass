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

`DeviceWorkPlan` has an explicit cross-stream lifetime protocol: `uploadAsync`
publishes a plan, `waitOn` makes that publication visible to a consumer stream,
and `markConsumed` records the point after the consumer's last read. Reuse of
the fixed device allocation waits on those consumer fences; plan destruction is
the only lifetime boundary allowed to quiesce the device. This prevents a
structural plan upload from becoming an implicit data race or a hidden
per-request synchronization point.

Both frameworks can select FlashInfer, but FlashInfer is the shared numerical
operator ABI, not a shared framework lifecycle. SGLang supplies
`ForwardBatch`/HiCache metadata through its attention-backend and plugin
registries; vLLM supplies its pinned `v1` `SchedulerOutput`/`InputBatch`
metadata through the V2 model-runner bridge and official `AttentionBackend`
seam. The common layer begins only after those projections become an
`EngineBatch`; no adapter reaches into the other framework's scheduler or
cache metadata.

The serving path is exact. Selection is an input trace, not a hidden runtime
policy. Conventional, late-bound, and exact-partial forms
share the same work-unit identity and demand trace.

In the SGLang HiCache path, resident-only forwards take the framework's
reference FlashInfer wrapper. This is an intentional zero-regression boundary:
the compiler/runtime mechanism is entered only when that forward has an
external-tier dependency. A mixed SGLang forward still uses one NTA launch and
carries resident and external work units together; the qualified vLLM
consumer has its own resident eager path because vLLM's official
`AttentionBackend` seam replaces the reference implementation for that
profile.
The reference path is not used to hide external work or to change demand.

The performance gate follows the same boundary. The request-bound direct form
is measured against the stock wrapper and must stay within the checked-in
5-percent acceptance budget (with paired confirmation on an apparent failure).
The general incremental protocol has a measurable control cost on a bare
all-resident dependency microbenchmark; that path is not the resident serving
default and is not presented as a zero-overhead kernel. Remote-tier claims must
therefore report both the direct/reference control and the mixed external
mechanism path, including profiler evidence for overlap and transfer cost.

The native runtime exposes one tier directory for HBM, mapped host memory,
host-staged memory, NVMe, and CXL DAX. The same descriptor is consumed by
host admission, device-visible backend metadata, and experiment telemetry.
The engine-neutral `ServingTierService` is the only serving attachment point:
host-staged uses indexed host objects, while NVMe installs catalog-validated
HBM objects and runs the finite device progress loop, reusing a slot's mapped
HBM destination when its capacity is sufficient. The native CXL path emits
catalog-validated direct dependencies; framework adapters reject it until
their numerical page tables can bind the same direct address. The Python
resource contract makes protocol
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
Engine statistics carry a typed consumer contract as well: a scheduler
projection is not a numerical consumer, and artifact gates reject
`projection_only` evidence. The schema is implemented once in
`experiments/consumer_contract.py`; serving and paired-evaluation validators
share it instead of maintaining independent interpretations of the same
claim.

See [ENGINE_INTEGRATION.md](ENGINE_INTEGRATION.md) for framework boundaries
and [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) for evaluation rules.
