# FlashInfer Integration

Status: data/state integration implemented; optimized-kernel continuation hook
and serving-engine adapters remain open.

## Why FlashInfer

FlashInfer is a useful kernel boundary because vLLM and SGLang can both select
its paged prefill/decode implementations. Integrating once at that boundary can
reuse optimized attention and cascade reduction across two serving engines.
It does not replace engine integration: request identity, generation,
cancellation, KV ownership, CUDA-graph lifetime, and SLO policy still belong to
vLLM or SGLang.

The implementation was checked against FlashInfer 0.6.12 and audited against
upstream `main` commit `668a1ba1ca86432c79f6adad37ecfce8d06ec083`.

## Implemented Boundary

`nta::flashinfer::planDecode` consumes FlashInfer's public paged-KV inputs:

```text
kv_indptr, kv_indices, last_page_len
    + request slot/generation bindings
    + physical page/object bindings
    -> request-owned NTA page continuations
```

The adapter validates CSR dimensions, monotonic offsets, final-page lengths,
physical-page bounds, and complete object bindings. It preserves repeated
physical pages and arbitrary page-table order. It does not inspect private
`PlanInfo` offsets or assume one CTA per request.

Attention page CTAs emit the same state consumed by FlashInfer cascade:

```text
V   = normalized partial attention output
LSE = base-2 logsumexp
```

When FlashInfer headers are available, the NTA reduction instantiates
`flashinfer::state_t` directly. CTest also runs the externally acquired NTA
fixture through a real `BatchDecodeWithPagedKVCacheWrapper`; the current local
gate covers heterogeneous request lengths and non-identity physical page
indices.

This is a real compatibility and differential-correctness layer. It is not yet
the final high-performance kernel integration: the current acquisition workload
uses NTA's mechanism attention CTA, while FlashInfer runs as the independent
differential implementation.

## Correct Kernel Hook

FlashInfer prefill and decode already schedule a CTA with a request index and a
KV chunk index. The acquisition hook belongs after those indices and chunk
bounds are known, but before query/KV movement, shared-memory barriers, or live
softmax state are initialized:

```text
FlashInfer plan
  -> CTA request/chunk binding
  -> acquire all physical pages required by this chunk
       ready: continue into the unchanged FlashInfer mainloop
       miss:  record the reconstructible chunk and return the CTA
  -> write (V, LSE) partial
  -> FlashInfer cascade merge after every required chunk is complete
```

Hooking individual `K/V` loads is incorrect because a miss can occur after the
CTA has accumulated softmax state or entered a barrier protocol. Polling inside
the attention CTA is also excluded: storage and network latency can outlive a
finite CTA by orders of magnitude.

A FlashInfer chunk may span several pages. Therefore the production hook needs
one continuation with a bounded dependency set, not one independent relaunch
per load. The next runtime revision must add a page-object list and an atomic
remaining-dependency count. Ready publication occurs only after every required
page is resident in its reserved HBM KV slot. For already resident pages, the
hook is a read-only validation/direct branch and the original FlashInfer pointer
and data-movement path remain unchanged.

This hook should be a small upstreamable template extension, such as an optional
`begin_kv_chunk(...)` policy on the attention variant. NTA's variant emits the
request binding and acquisition markers; the LLVM pass proves the pre-state
deferral boundary and lowers them. A source rewrite of every vector load or a
long-lived FlashInfer fork is not the target design.

## Serving Adapters

The engine adapters should be thin and separate from the kernel integration.

For both vLLM and SGLang they must:

1. Publish request slot, generation, tenant, priority, deadline, and
   cancellation before graph replay.
2. Translate the engine's KV block ownership into physical page/object
   bindings while preserving FlashInfer's `kv_indices` values.
3. Reserve stable HBM slots for command-addressed pages so FlashInfer's native
   page pointers are valid after acquisition completes.
4. Supply the FlashInfer CTA-to-request/chunk mapping through a supported API,
   not version-pinned reads of private workspace offsets.
5. Delay final output consumption until all exact dense-attention chunks have
   produced valid `(V, LSE)` states.
6. Retire generations only after in-flight acquisition and graph work can no
   longer publish stale readiness.

SGLang and vLLM remain separate adapters because they differ in batch formation,
KV allocation, cancellation, and graph orchestration even when both select the
same FlashInfer kernel.

## Production Gates

The FlashInfer path is not production-ready until all of these pass:

- the optional chunk hook is implemented in decode and paged prefill without a
  persistent kernel or CTA-wide completion polling;
- multi-page dependency continuations are race- and cancellation-safe;
- direct-resident overhead is statistically indistinguishable from untouched
  FlashInfer or is small enough to justify with end-to-end benefit;
- real SGLang and vLLM batches pass output, reuse, cancellation, CUDA-graph, and
  long-context tests;
- TTFT, TPOT, p50/p99, SLO goodput, CPU usage, and SM tax are compared with
  userspace prefetch and native tiering baselines; and
- NVMe and RDMA error, timeout, reset, and backpressure behavior is validated.
