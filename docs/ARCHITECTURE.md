# Architecture

This document is the short architectural entry point. The detailed contract
is in [REFRACTOR_DESIGN.md](REFRACTOR_DESIGN.md).

NTA has one execution mechanism: late-bound, generation-bound heterogeneous
work units. Each unit carries exact demand and availability; the protocol
launches bounded ready groups at a measured granularity.

```
engine adapter
  -> EngineBatch / RequestBinding
  -> WorkBatch / DemandDescriptor
  -> ExecutionSession / WorkLedger
  -> DeviceWorkPlan semantic-to-native bridge
  -> compiler-mapped consumer
```

SGLang and vLLM are adapters, not runtime implementations. The native ABI
stores work and dependencies; the semantic layer validates generation, epoch,
demand, and availability before native submission.

The serving path is exact. Selection is an input trace, not a hidden runtime
policy. Conventional, late-bound, and exact-partial forms
share the same work-unit identity and demand trace.

See [ENGINE_INTEGRATION.md](ENGINE_INTEGRATION.md) for framework boundaries
and [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) for evaluation rules.
