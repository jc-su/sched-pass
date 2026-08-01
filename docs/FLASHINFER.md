# FlashInfer Integration

Status: optimized decode and FA2 paged-prefill continuation hooks are
implemented and executed for FlashInfer 0.6.12; serving-engine adapters remain
open.

## Boundary

FlashInfer is a useful kernel boundary because serving engines can select its
paged prefill/decode kernels. Integration there reuses optimized attention, but
does not replace engine ownership of request identity, cancellation, KV
allocation, CUDA graphs, or output lifetime.

`nta::flashinfer::planDecode` translates public paged-KV inputs into the common
work model:

```text
kv_indptr, kv_indices, last_page_len
    + request slot/generation bindings
    + physical page/object bindings
    -> NTA WorkItem and AcquireRequirement arrays
```

It validates CSR dimensions, monotonic offsets, final-page lengths, physical
page bounds, and complete object bindings. Repeated physical pages and arbitrary
page-table order are preserved. `planScheduledDecode` additionally checks NTA
work against the active request/KV-tile order and chunk size produced by
FlashInfer's scheduler.

The implementation is checked against the FlashInfer 0.6.12 wheel headers. The
overlay requires an exact hash of the complete 205-file include tree, exact
decode/prefill hashes, and exact insertion anchors; an unknown revision fails
closed. Overlay creation is process-locked, atomically published, immutable,
and hash-verified on reuse.

## Kernel Hook

Decode and paged prefill know request and KV-tile identity before shared memory,
barriers, TMA/cp.async state, or live softmax values. The overlay adds this
canonical global-kernel entry sequence:

```text
validate active scheduler work
bind request and dependency set
acquire
  ready -> enter unchanged FlashInfer device mainloop
  miss  -> publish finite continuation and return the whole CTA
```

The LLVM pass proves CTA-uniform operands and control, requires the direct
`acquire -> pending branch -> defer -> return` shape, and rejects a hook in a
non-inlined device helper. Direct dependencies take a compiler-generated fast
edge that avoids the noinline acquisition helper while retaining request
liveness and continuation-state checks.

FlashInfer's `AttentionVariant` starts too late and cannot return the whole CTA.
That limitation prevents a variant-only implementation, not this architecture.
The checked source overlay inserts the hook in the global wrappers. A small
upstream `begin_work(...)` template hook is the preferred long-term replacement
for the overlay.

## JIT Delivery

Run a custom FlashInfer generator under:

```bash
tools/jit/activate.py --build-dir build --flashinfer-hook -- \
  python3 generator.py
```

The activator:

- creates an ABI/pass/source-fingerprinted cache;
- verifies and prepares a private 0.6.12 include overlay;
- instruments decode and paged-prefill kernel sources through Clang 22;
- compiles device acquisition code into every hooked kernel source;
- emits reset, host/NVMe progress, publish, completion, and ABI wrappers from
  exactly one source per shared object; and
- leaves FlashInfer planning and TVM-FFI binding sources unchanged.

`JitPhaseProgram` loads those wrappers from the generated shared object and
checks ABI compatibility. `tools/flashinfer/schedule.py` isolates the private
0.6.12 `PlanInfo` layout and extracts active request/KV-tile identity, including
CUDA-graph padding masks. A supported upstream schedule API should replace that
version adapter.

## Finite Execution

One stream-ordered acquisition cycle is:

```text
reset -> FlashInfer initial run -> complete launched work
      -> bounded transport progress -> publish ready
      -> FlashInfer full-grid ready run -> complete launched work
```

`Done` and still-`Pending` CTAs return at the pre-state hook. `Ready` CTAs enter
the unmodified mainloop. Multiple KV-head CTAs may share one x-coordinate work
item; continuation initialization uses a single CAS owner. No CTA polls for
external completion and no persistent kernel is used.

Split-K decode is supported. FlashInfer's stock cascade merge can execute after
the initial miss launch and transiently write incomplete output. The integration
contract forbids output consumption until the ready relaunch and final merge
finish. Suppressing the first merge is a remaining performance optimization.
Ragged prefill is intentionally not hooked because it does not consume external
paged KV through this boundary.

## Validation

The local CTest gate covers:

- real multi-source decode and paged-prefill JIT compilation with NTA Params;
- C and C++ loading of the exported ABI-9 phase functions;
- resident and pinned-host deferred decode;
- two KV-head CTAs sharing one continuation;
- 32-way split-K decode and stock cascade reduction;
- four-work-item FA2 paged prefill; and
- exact output comparison with stock FlashInfer.

Memcheck, racecheck, and synccheck are clean on the shared-continuation deferred
path. A matched 64-request custom-variant microbenchmark has an 8% CTest
regression limit; the latest local median measured 6.33% resident overhead.
This is a local regression gate, not controlled multi-machine performance
evidence.

## Open Gates

- vLLM and SGLang request/KV/cancellation/CUDA-graph lifecycle adapters;
- long-running graph replay, cancellation, generation reuse, and multi-GPU
  stress;
- avoiding unnecessary split-K reduction on miss launches;
- TTFT, TPOT, p50/p99, SLO goodput, CPU use, and SM-tax comparisons;
- current-ABI NVMe error/reset/backpressure reruns and real RNIC RDMA; and
- upstream FlashInfer hook and scheduler-metadata APIs.

Therefore this is a functioning optimized-kernel integration for one validated
FlashInfer revision. It is not yet a production serving integration or an
OSDI-level evaluation.
