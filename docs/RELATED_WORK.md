# Related-work boundary

This document defines the comparison boundary for the current system. It is
not an implementation specification and it makes no “first” or novelty claim
without an experiment that isolates the claimed mechanism.

## What the project does not claim

NTA does not claim novelty for paged KV layout, FlashInfer's attention
mainloop, CUDA graph capture, top-k/DSA selection, host-to-device bandwidth, or
partial softmax reduction. Those are inputs or baselines shared by the
comparison arms. Selection is outside the serving contract.

## Comparison boundary

- FlashInfer supplies optimized paged attention and mergeable attention
  state. NTA supplies the execution contract around heterogeneous availability
  and must preserve the same numerical result.
- SGLang HiCache supplies engine-level cache ownership, block/page metadata,
  and host-cache movement. NTA's question is whether the engine can expose
  exact request-bound work as it becomes runnable without rebuilding the batch
  around one global barrier.
- vLLM and SGLang are framework boundaries, not two NTA mechanisms. Their
  scheduler metadata is projected into the same `RequestBinding` and
  `WorkBatch` contract.
- GPU-routed MoE is an operator-level stress fixture for device-produced
  demand. It is not evidence that MoE has a KV-cache problem and is not the
  primary serving workload.

## System contribution boundary

The contribution under evaluation is one mechanism: generation-bound exact
heterogeneous work units whose demand and availability are bound as late as
correctness permits, then launched in bounded groups at a measured
granularity. The experiment matrix must separately measure:

1. device selection versus host selection;
2. conventional gather versus late-bound execution;
3. homogeneous versus heterogeneous batches; and
4. coarse versus fine work-unit granularity.

This decomposition prevents byte reduction, selector quality, graph effects,
or framework-specific scheduling from being incorrectly credited to the NTA
mechanism.
