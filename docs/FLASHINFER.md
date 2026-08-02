# FlashInfer Integration

Status: optimized decode and FA2 paged-prefill work ticket hooks, native C and
Python engine-runtime bindings, and an owning per-layer FlashInfer executor are
implemented and executed for FlashInfer 0.6.12. The eager SGLang 0.5.14
HiCache lifecycle and full decode CUDA-graph replay are integrated through its
plugin system. FlashInfer top-k selection now feeds bounded device-indexed host
page acquisition and real paged decode without a host identity round trip.
vLLM, serving use of that path, demand-mode graph phases, and paged-prefill
graph validation remain open.

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

This overlay is the first typed compiler frontend, not an NVVM address-cone
heuristic. It operates while FlashInfer still has scheduler request/tile types,
places the work hook before kernel state initialization, and carries
request-local contributor groups into cascade merge. The raw LLVM pass then
checks CTA collectivity, control dependence, and the finite return boundary.

The installed 0.6.12 package also exposes
`top_k_page_table_transform`, `BlockSparseAttentionWrapper`, and
`VariableBlockSparseAttentionWrapper`. The implemented stress path uses the
real top-k transform, a stable GPU index table registered with NTA, and real
FlashInfer paged decode over compact selected KV. The custom query-dependent
sparse benchmark remains only a protocol fixture.

## Kernel Hook

Decode and paged prefill know request and KV-tile identity before shared memory,
barriers, TMA/cp.async state, or live softmax values. The overlay adds this
canonical global-kernel entry sequence:

```text
validate active scheduler work
bind request and dependency set
acquire
  ready -> enter unchanged FlashInfer device mainloop
  miss  -> try one transport submission or publish an intent
           -> record a finite work ticket and return the whole CTA
```

The LLVM pass proves CTA-uniform operands and control, requires the direct
`acquire -> pending branch -> defer -> return` shape, and rejects a hook in a
non-inlined device helper. Direct dependencies take a compiler-generated fast
edge that avoids the noinline acquisition helper while retaining request
liveness and work ticket-state checks.

The hook uses the common backend policy. With CTA NVMe try-issue enabled, the
FlashInfer CTA leader can construct and ring one read before the collective
miss return; contention falls back to the finite scheduler. Completion polling
is never inserted into FlashInfer. This path is covered by the controller-free
device queue integration test, but the optimized FlashInfer tests currently use
resident and pinned-host objects rather than a real NVMe controller.

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
- emits reset, host/NVMe progress, compatibility publication, completion, and
  ABI wrappers from exactly one source per shared object; and
- leaves FlashInfer planning and TVM-FFI binding sources unchanged.

`JitPhaseProgram` loads those wrappers from the generated shared object and
checks ABI compatibility. `tools/flashinfer/schedule.py` isolates the private
0.6.12 `PlanInfo` layout and extracts active request/KV-tile identity, including
CUDA-graph padding masks. A supported upstream schedule API should replace that
version adapter.

The Python integration passes `Runtime` and optional `DeviceWorkPlan`
allocations as zero-copy DLPack byte tensors, satisfying FlashInfer's
custom-tensor ABI without reproducing native layouts in an engine.
Preacquired batches pass only the runtime view and request count; the compiler
emits request-liveness guards without plan, reset, progress, or retirement
work. Demand-driven batches use `FlashInferLayerEpoch`, whose eager `run_*` and
fixed `enqueue_*` methods retain the bounded work-ticket protocol. The fixed
methods can be captured only after structural work-plan upload; the upload path
rejects capture rather than synchronizing illegally. SGLang owns the
request-guarded decode replay path, but does not yet replay these demand-mode
phase nodes.

## Finite Incremental Execution

One unavailable-data cycle in the current implementation is:

```text
reset -> FlashInfer initial run -> complete launched work
      -> bounded transport progress and runnable-work publication
      -> FlashInfer request-bound runnable-work launch
      -> complete launched work
```

`Done` and still-`Pending` CTAs return at the pre-state hook. CTAs whose ticket
is in the literal ABI state `Ready` enter
the unmodified mainloop. Multiple KV-head CTAs may share one x-coordinate work
item; work ticket initialization uses a single CAS owner and publishes
`Initializing -> Pending` only after the complete record is visible. A losing
CTA does not spin or issue against partial state. No CTA polls for external
completion and no persistent kernel is used.

Publication writes original scheduler work-ticket indices into a stable
runnable-tile array. A runnable-work launch maps the physical front-of-grid CTA
prefix through that array before FlashInfer derives request or KV-tile identity.
CTAs outside the published prefix return at the hook. This compacts useful work
inside the fixed framework grid; reducing the physical launch width still
requires graph or device-launch integration.

Work ticket state is scoped to one attention-layer invocation. The
runnable-work launch retires that layer's work as `Done`; the next layer must begin a new
epoch or use a disjoint plan slice. Reusing one completed work ticket array for
every layer would skip later attention kernels and is explicitly outside the
integration contract.

When the engine already enqueued acquisition at its producer boundary, a
post-transfer event orders the consumer stream and the kernel uses the
planless-preacquired mode. Each CTA still validates the current request
generation before touching output, but no work ticket is allocated. This is
the serving fast path; the full cycle above remains available when tile demand
is discovered only at kernel execution.

Split-K decode and paged prefill are phase aware. The custom ABI carries
`nta_skip_merge`; initial and intermediate runnable-work launches write only
scratch partitions, and the final bounded launch performs one stock FlashInfer
reduction. The patched cascade merge maps each decode row directly to its
request reduction group and each prefill row through FlashInfer's query indptr.
It reads device-owned expected/completed/failed contributor arrays after the
grid dependency and skips incomplete or failed request rows before reading
scratch. A complete request may therefore merge while a peer remains blocked.
The overlay modifies one decode, both prefill split-K dispatch sites, and the
cascade merge after verifying exact source hashes and insertion anchors.
Ragged prefill is intentionally not hooked because it does not consume external
paged KV through this boundary.

## Incremental Operator Boundary

FlashInfer supplies the common compute boundary; NTA must not compare a custom
fine-grained kernel against stock FlashInfer and attribute the difference to
incremental execution. The compiler target is one source producing an untouched
complete-data form and an incrementally executable form. Production comparisons
use the same FlashInfer math, KV layout, cache state, request trace, and output
allocation for:

```text
compiler-generated direct form
layer-complete transfer
coalesced bulk acquisition
whole-request skip and rebatch
forced fine-grained incremental execution
unified request-aware incremental scheduler
```

The production scheduler does not select among unrelated implementations. It
continuously groups missing ranges from transfer savings and request delay,
launches the resulting runnable tile set, and feeds actual partial progress into
later batch decisions. Forced endpoints remain only for evaluation. This is the
co-design boundary described in `SYSTEM_PLAN.md`.

For the sparse stress case, selected page IDs must remain on device:

```text
GPU scores
  -> FlashInfer top_k_page_table_transform
  -> NTA request/object binding
  -> real FlashInfer sparse attention
```

A separate selector kernel or graph node is permitted. The implemented path is
stream ordered and updates the exact stable index table consumed by NTA. Its
offline-oracle arm materializes IDs on the CPU before timing; the NTA hot path
does not.

## Validation

Correctness gates below run on ABI v20. Any quoted sanitizer and performance
numbers that predate v20 must be regenerated before use as current evidence.

The local CTest gate covers:

- real multi-source decode and paged-prefill JIT compilation with NTA Params;
- C and C++ loading of the exported ABI-20 phase functions;
- resident and pinned-host deferred decode;
- heterogeneous request remapping where only the nonzero scheduler ticket is
  ready and physical CTA zero must execute it;
- two KV-head CTAs sharing one work ticket;
- 32-way split-K decode, fail-closed incomplete merge, and stock cascade
  reduction after all contributors complete;
- two independently split decode requests where the complete request merges
  and the blocked peer preserves its sentinel output;
- four-work-item FA2 paged prefill; and
- exact output comparison with stock FlashInfer.

`FlashInferSelectedPages.py` additionally checks real selector-to-acquisition
dataflow, bounded GPU-generated source and destination indices, cold and
retained staging, and output parity with stock FlashInfer over the same selected
pages. The five-point sweep and current numbers are recorded in
`VALIDATION.md`.

Memcheck, racecheck, and synccheck are clean on the shared-work ticket deferred
path. A matched 64-request custom-variant microbenchmark has an 8% CTest
regression limit; the latest local median measured 4.63% resident overhead.
This is a local regression gate, not controlled multi-machine performance
evidence.

The matched SGLang 0.5.14 environment completes both a stock smoke workload
and an NTA HiCache workload through the installed `sglang.srt.plugins` entry
point. The plugin registers `nta_flashinfer`, binds request slot generations,
intercepts HiCache loads, and routes external paged-KV batches through the
instrumented wrappers. Resident batches retain the stock wrappers.

```bash
./benchmarks/serving/SglangSmoke.py \
  --model /path/to/local/model --requests 4 --max-new-tokens 8
```

The runner selects a CUDA-compatible host compiler before importing SGLang,
uses an isolated FlashInfer JIT cache, and emits machine-readable JSON. It is a
stock serving baseline and deliberately records `nta_integrated=false`.
`CompareSglangHiCache.py` is the integrated matched gate. Historical local
Llama-160M runs had exact output parity and zero fallback, but the current
schedule-aware five-promotion result was 2.62% slower than stock. See
`SGLANG.md` for the supported profile and the limits of this uncontrolled
single-machine result.

## Open Gates

- compiler-generated direct and incremental forms of the same real FlashInfer
  dense kernel, followed by unified grouping, engine feedback, best-fixed trace
  comparison, and resettable decision-regret evaluation;
- end-to-end SGLang or vLLM use of the GPU-selected page path, including real
  model-generated scores rather than controlled random scores;
- a vLLM request-generation/KV-offload adapter, SGLang demand-mode graph
  phases, and paged-prefill graph validation;
- 24-hour serving graph replay/cancellation soak and execution on multiple
  physical GPUs (the engine-neutral 10,000-epoch lifecycle gate passes locally,
  while the two-GPU test skips on this one-GPU host);
- TTFT, TPOT, p50/p99, SLO goodput, CPU use, and SM-tax comparisons;
- current-ABI NVMe error/reset/backpressure reruns; RNIC/RDMA is deferred from
  the current local-memory-and-storage scope; and
- upstream FlashInfer hook and scheduler-metadata APIs.

Therefore this is a functioning optimized-kernel integration for one validated
FlashInfer revision. It is not yet a production serving integration or an
OSDI-level evaluation.
