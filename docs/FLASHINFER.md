# FlashInfer Integration

Status: optimized decode and FA2 paged-prefill work ticket hooks, native C and
Python engine-runtime bindings, and an owning per-layer FlashInfer executor are
implemented and executed for FlashInfer 0.6.12. The eager SGLang 0.5.14
HiCache lifecycle and full decode CUDA-graph replay are integrated through its
plugin system. FlashInfer top-k selection now feeds bounded device-indexed host
page acquisition and real paged decode without a host identity round trip.
Canonical ragged FlashInfer now also establishes an exact bounded-HBM streaming
crossover over atomic CPU-DRAM promotion. vLLM, compiler generation of that
streaming form, serving use of it, demand-mode graph phases, and paged-prefill
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
Each generated shared object exports a schema-versioned family/form/capability
contract and source fingerprint; runtime and SGLang reject an ABI mismatch,
wrong operator family, missing capability, or unpaired source revision.

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

Demand-mode kernels also delimit their numerical effect:

```text
acquire(work ticket) or collective return
begin_partial(work ticket)
  unchanged FlashInfer numerical mainloop
commit_partial(reduction group, contributor, count, cost)
```

The begin and commit calls carry LLVM `convergent` semantics. The pass proves
that they share the acquisition's request binding and work ticket, that commit
post-dominates begin, and that each region publishes exactly once. It then
erases begin and lowers commit to the generation-checked runtime protocol with
`!nta.partial` and function-level `!nta.operator` metadata. This moves partial
completion out of a hand-written hook and makes malformed incremental control
flow a compile error.

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
Preacquired batches use a separately named request-bound module and pass only
the runtime view and request count; the compiler emits request-liveness guards
without plan, reset, progress, or retirement work. One lane per warp reads the
immutable request directory and broadcasts the CTA-uniform decision, avoiding
both per-lane metadata reads and a CTA barrier. Demand-driven modules require
completion tracking and contain the acquisition plus partial-region contract;
they cannot be invoked as request-bound modules by changing a runtime flag.
Demand-driven batches use `FlashInferLayerEpoch`, whose eager `run_*` and fixed
`enqueue_*` methods retain the bounded work-ticket protocol. The fixed methods
can be captured only after structural work-plan upload; the upload path rejects
capture rather than synchronizing illegally. SGLang owns the request-guarded
whole-model decode replay path. The adapter separately captures exact-shape
demand decode and paged-prefill phase nodes in an NTA operator graph, retaining
and refreshing FlashInfer's dynamic metadata tensors before every replay.

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

Publication updates the original scheduler work ticket. Each FlashInfer wave
retains the canonical grid and request/KV-tile coordinates; the compiler hook
admits only tickets that are available in the current request generation, and
all other CTAs return before state initialization. This avoids mutating one
shared runnable-index list while transfer and compute streams overlap. It does
not compact the physical grid, so reducing launch width still requires graph or
device-launch integration.

Work ticket state is scoped to one attention-layer invocation. The
runnable-work launch retires that layer's work as `Done`; the next layer must begin a new
epoch or use a disjoint plan slice. Reusing one completed work ticket array for
every layer would skip later attention kernels and is explicitly outside the
integration contract.

When the engine already enqueued acquisition at its producer boundary, a
post-transfer event orders the consumer stream and the kernel uses the
transformed direct form without a work ticket. Every CTA still executes the
compiler-inserted current-generation guard. The full ticketed cycle above is
used when tile demand is unresolved at kernel execution.

Split-K decode and paged prefill are phase aware. Every wave writes only the
scratch partitions owned by contributors admitted in that wave, then reaches
FlashInfer's original numerical reduction behind an NTA completion gate. The
patched cascade merge maps each decode row directly to its request reduction
group and each prefill row through FlashInfer's query indptr.
It reads device-owned expected/completed/failed contributor arrays after the
grid dependency and skips incomplete or failed request rows before reading
scratch. A complete request may therefore merge while a peer remains blocked.
The overlay modifies one decode, both prefill split-K dispatch sites, and the
cascade merge after verifying exact source hashes and insertion anchors.
Ragged prefill is compiler transformed for the canonical bounded-HBM operator
experiment. It is not used as SGLang's external paged-KV hook; SGLang demand
uses the version-checked FA2 paged-prefill and decode insertion points.

The overlay also patches the MLA decode kernel anchors
(`BatchDecodeWithPagedKVCacheKernelMLA` entry and exit) so the 0.6.12 source
tree stays hash-consistent under one manifest. That MLA path is
**patched but unvalidated**: no execution gate covers it, no performance or
correctness claim includes it, and the SGLang plugin rejects MLA models at
construction so the unvalidated hook cannot be reached from the supported
profile. Validating or removing the MLA anchors is an open gate; until then any
non-SGLang consumer must treat MLA as unsupported.

For SGLang multi-round CPU-DRAM demand, the structural work/dependency topology
is reused across layers. After layer `L` attention finishes, a GPU kernel
rebinds the indexed directory to layer `L+1`, and a finite copy stages the first
contributor wave while the model executes post-attention work. Epoch reset
preserves those completed directory entries. The next canonical initial wave
consumes them, and bounded progress acquires only the remaining waves. This is
an online layer-order optimization, not an offline demand oracle; page identity
and request generation still come from the current SGLang batch.

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

The first canonical performance result now uses FlashInfer's real ragged
prefill and online-softmax merge, not a custom attention kernel. At the 64K
context/256-query point, bounded double buffering is `1.1714x` faster than one
atomic promotion with a 95% interval of `[1.1660x, 1.1732x]` and uses `4x` less
staging HBM. A separate heterogeneous context/query run is `1.1100x` faster
with `4.83x` less staging. `TIER_STREAMING.md` contains the exact shape,
commands, per-request completion evidence, and claim boundary. The reusable
runtime operator now owns wrapper construction, wave metadata, copy slots,
partials, merge, completion, and dynamic-source graph replay. Paired compiler
artifacts export and validate one typed request-coordinate, `(V, LSE)`, and
ordered-merge plan. This closes the compiler/runtime operator
performance-opportunity gate; it does not close the SGLang integration gate.

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

Correctness gates below run on ABI v25. Any quoted sanitizer and performance
numbers that predate v22 must be regenerated before use as current evidence.

The local CTest gate covers:

- real multi-source decode and paged-prefill JIT compilation with NTA Params;
- C and C++ loading of the exported ABI-25 phase functions;
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
path. A matched 64-request custom-variant microbenchmark has a 5% CTest
regression limit on the compiler-transformed request-bound direct form; the
latest three consecutive cached runs measured 1.76%-2.59% overhead. The
incremental form measured 5.82%-6.41% on the same runs and is
reported separately rather than being mislabeled as resident direct overhead.
This is a local regression gate, not controlled multi-machine performance
evidence.

The matched SGLang 0.5.14 environment completes both a stock smoke workload
and an NTA HiCache workload through the installed `sglang.srt.plugins` entry
point. The plugin registers `nta_flashinfer`, intercepts HiCache loads, and
routes resident, preacquired, graph, and unresolved batches through transformed
wrappers. The benchmark gate rejects any observed stock attention launch.

```bash
./benchmarks/serving/SglangSmoke.py \
  --model /path/to/local/model --requests 4 --max-new-tokens 8
```

The runner selects a CUDA-compatible host compiler before importing SGLang,
uses an isolated FlashInfer JIT cache, and emits machine-readable JSON. It is a
stock serving baseline and deliberately records `nta_integrated=false`.
`CompareSglangHiCache.py` is the integrated matched gate. It rejects fallback,
stock attention launches, missing transformed/ticketed launches, mismatched output, and mismatched
residency sequences. `--verify-transfer` executes synchronous row-by-row KV
verification in a separate arm so it cannot inflate only NTA's timed result.
See `SGLANG.md` for the supported profile and the limits of local
single-machine measurements.

## Open Gates

- best-fixed trace comparison and resettable decision-regret evaluation for the
  SGLang consumer of compiler-generated progress;
- end-to-end SGLang or vLLM use of the GPU-selected page path, including real
  model-generated scores rather than controlled random scores;
- a vLLM request-generation/KV-offload adapter and integration of demand phases
  into SGLang's full model graph (the separately keyed finite operator graph is
  implemented for decode and paged prefill);
- 24-hour serving graph replay/cancellation soak and execution on multiple
  physical GPUs (the engine-neutral 10,000-epoch lifecycle gate passes locally,
  while the two-GPU test skips on this one-GPU host);
- TTFT, TPOT, p50/p99, SLO goodput, CPU use, and SM-tax comparisons;
- current-ABI NVMe error/reset/backpressure reruns; RNIC/RDMA is deferred from
  the current local-memory-and-storage scope; and
- upstream FlashInfer hook and scheduler-metadata APIs.

Therefore this is a functioning optimized-kernel integration plus a positive
canonical-FlashInfer bounded-HBM mechanism for one validated revision. It is
not yet a production serving integration or an OSDI-level evaluation.
