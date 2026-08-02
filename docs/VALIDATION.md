# Validation Record

Date: 2026-08-02

This record supports a locally validated mechanism prototype. It does not
establish production readiness, an end-to-end serving result, or an OSDI-level
evaluation. The missing evidence is listed explicitly below.

Tables and command output retain ABI state `Ready` and legacy CLI policy
`late-bound` exactly as measured. The current design calls these available data,
runnable work, and device-generated demand; results are not renamed after
collection.

The source ABI is v20 and the public C API is v11. The ABI bump adds explicit
source and destination bounds for GPU-generated indexed transfers plus finite
cache invalidation used by cold-cache experiments. Focused compiler, C API,
Python, and real FlashInfer JIT tests have been rerun for v20. The complete
local qualifier must be rerun after the implementation is frozen. Decode-graph
serving and VFIO NVMe results below predate v20 and remain historical local
evidence; clean-revision repeated trials are open.

## Environment

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability 12.0
- Driver: 595.84
- Device memory: 97,887 MiB
- Compiler: Ubuntu LLVM/Clang 22.1.8
- Device toolkit: CUDA 12.9.86
- Target: `sm_120`
- Compute Sanitizer: 2025.3.1
- FlashInfer: 0.6.12
- PyTorch: 2.11.0+cu130

CUDA 13 is installed but is not used for device compilation because its header
layout is incompatible with the current Clang CUDA wrapper.

## Reproduction

```bash
./scripts/qualify-release.py --profile=local
./scripts/measure-direct-overhead.sh
```

The first command builds the pass/runtime, runs CTest, evaluates global-load
and TMA attention across placements, runs the MoE workload, emits PTXAS reports,
executes memcheck, racecheck, and synccheck, runs a CUDA-disabled build, performs
10,000 lifecycle epochs, and runs static/lint/package checks. The second runs
alternating process-level baseline/mechanism trials and computes Student-t intervals.
Generated evidence is under `results/` and is excluded from source control.
The ABI-v20 local qualification must complete all commands successfully before
it may report `READY`. Earlier ABI qualifications covered the full
functional/sanitizer matrix, CPU-only build, 10,000 lifecycle epochs, Clang
static analysis, Python and shell checks, package construction, and patch
hygiene; they do not substitute for the final v20 run.

The schema-2 production and OSDI qualifiers both report `NOT_READY` on this
workspace. The required external evidence manifests do not exist, and the
worktree is not an immutable commit. This is the expected claim boundary, not a
local test failure.

## Correctness Gates

`ctest --test-dir build --output-on-failure` discovers 39 tests. On this
single-GPU host, 38 pass and `nta-multi-gpu` reports the configured skip code
77 because a second physical CUDA device is unavailable. The gates cover:

1. FlashInfer CSR-to-common-plan validation, including grouped pages and bad
   metadata;
2. engine-neutral plan construction and bounded dependency validation;
3. LLVM byte-address, tensor-map, and dependency-set lowering, including
   rejection of live-state, token, missing-binding, non-inlined-helper,
   lane-divergent control, non-dominating divergent control, and divergent PHI
   operand cases;
4. host/device ABI v20 layout, including operation epochs, terminal counters,
   multi-CTA completion state, and the NVMe control-page mirror;
5. Clang nvcc-shim compilation of a foreign source kernel, including automatic
   optimizer-last lowering, marker removal, metadata, and fast-math forwarding;
6. compilation and linking of FlashInfer's real multi-source custom decode and
   paged-prefill extensions through the same JIT activator and isolated cache;
7. resident, pinned-host deferred, shared-head, split-K decode, and multi-tile
   paged-prefill execution inside those optimized FlashInfer kernels, including
   zero-delay resident and positive-delay external per-ticket GPU timestamps;
8. runtime allocation, object/replica capacities, cancellation, non-owning
   engine allocation registration, two-slot pinned/async device-plan upload,
   capture rejection for structural uploads, and runtime binding validation;
9. controller-free GPU validation of direct CTA NVMe SQE construction,
   nonresident completion, ready-only resume, and non-spinning scheduler
   fallback under queue-lock contention, two distinct request CTAs contending
   for one queue, plus stale completion, NVMe status failure, malformed-CID
   queue quiescence, and fatal-queue isolation;
10. mixed-tier acquisition with duplicate coalescing, stale generations,
   cancellation, and repeated intent-slot reuse;
11. a three-object-per-CTA mixed-tier dependency set;
12. stale object-version failure without output publication;
13. a 4,096-CTA all-direct scale case with `pending=0`;
14. GPU-routed top-2 MoE expert matrices across mixed tiers, matched CPU-sync,
   overlapped overfetch, and direct policies, plus inspection of the real
   lowered MoE producer/consumer IR;
15. the matching direct-address numerical baseline;
16. staged split-K paged attention through the common work plan;
17. the same common-plan attention path using hardware TMA;
18. differential output validation against FlashInfer 0.6.12;
19. query-dependent sparse attention whose query is materialized on device,
   whose selector and acquisition execute in one CTA, and whose ready launch
   preserves deliberately permuted request-slot/generation bindings, plus the
   same selector and attention math after an overlapped all-page GPU overfetch;
20. non-owning registration of an engine-managed mapped-DRAM allocation;
21. non-owning staged DRAM with a deliberately unaligned source allocation;
22. 200 repeated mixed lifecycle epochs and 100 repeated unaligned staged
   epochs with cancellation, request-generation reuse, and object-version
   faults;
23. cross-device runtime, object-allocation, plan ownership, request isolation,
   and caller-device restoration when two physical GPUs are available;
24. the versioned C and Python runtime APIs, including stream-ordered indexed
   pinned-host row registration;
25. SGLang plugin discovery in the frontend and a spawned worker, including
   HiCache, abort, and shutdown hooks;
26. stable request identity, prefix cancellation, and generation-safe slot
   reuse;
27. exact SGLang demand-plan cache identity, including host/device page-pair
   remapping;
28. completion-driven dependency arrival under adversarial multi-CTA ordering,
   including exact-once direct runnable-work publication after the last
   dependency;
29. request-level blocked-byte, runnable-compute, and completed-compute
   accounting plus complete and failed reduction groups; and
30. deterministic host execution-policy selection across resident, bulk,
   bounded-round, and invalid cost-model inputs.

The pass now proves a canonical finite defer edge and CTA collectivity. Markers
must be inlined into a GPU kernel entry, where the CTA analysis treats kernel
arguments, `blockIdx`, and block/grid dimensions as CTA-uniform. It rejects
non-inlined helpers and control or marker operands derived from `threadIdx`,
lane/warp identity, atomics, volatile loads, local allocation, or unknown
calls. Control dependence is computed from the post-dominator tree, including
branches that do not dominate the marker; uniform PHIs require every incoming
control dependence to be CTA-uniform. A positive late-bound fixture selects an
object catalog entry from CTA-uniform GPU state. Lowered modules contain no
bind/acquire/defer markers and carry ABI-v20 `!nta.acquire` metadata tagged
`split-phase-cta`.

Attention global-load and TMA kernels, the generic dependency-set kernel, and
the MoE kernel all consume the same `abi::WorkItem` and
`abi::AcquireRequirement` arrays. `DeviceWorkPlan` supports fixed-capacity
reuse, pinned staging, stream-ordered asynchronous updates, and explicit
cross-stream waits. Two staging/event slots permit consecutive uploads without
an unconditional synchronization. Attention-only side metadata contains
token-count and request-index fields, not a duplicate acquisition binding.

Availability changes propagate from each completed object through bounded
reverse dependency edges. The CTA satisfying the final dependency performs an
exact-once state transition and appends that ticket to the compact runnable
queue. Normal progress therefore needs no publication launch. The retained
publication kernel drains the bounded changed/pending indexes for compatibility
and diagnostics; it never scans the full ticket directory. The 4,096-CTA
resident test confirms that an all-direct epoch creates neither pending entries
nor publication traffic. No path polls for a future external completion.

The optimized FlashInfer test additionally verifies a heterogeneous runnable
wave: only scheduler work ticket 1 is available, physical CTA 0 maps through
the compact work array, and request 1 executes. Its two-request split-K test
completes every contributor for request 0 while leaving request 1 blocked. The
device merge emits request 0's stock-equivalent output and preserves request
1's sentinel without reading incomplete scratch.

A separate CUDA-disabled build against the supported LLVM 22 installation
passes all five applicable adapter, plan, IR, ABI, and execution-policy tests.

A 10,000-epoch stress run reuses one runtime, fixed-capacity device work plan,
registered buffers, stream, and captured CUDA graph while rotating request
identity, cancellation, stale generations, and object-version faults. It
completed 10,000 graph launches with no live pending work ticket and no
verification failure. This is a strong lifecycle regression gate, not a
24-hour serving soak.

The JIT activator compiles and links FlashInfer 0.6.12's real multi-source
custom decode and paged-prefill extensions in an isolated NTA cache. The
version-checked overlay places acquisition sites at their global kernel entry
wrappers, and the execution gate exercises resident and deferred work ticket
through those optimized kernels.

## Sanitizers And Resources

The latest mixed-tier TMA attention, query-dependent sparse attention,
dependency-set, MoE, CTA NVMe queue model including malformed-completion
recovery, and unaligned non-owning staged-DRAM runs report:

```text
memcheck:  ERROR SUMMARY: 0 errors
racecheck: 0 hazards displayed (0 errors, 0 warnings)
synccheck: ERROR SUMMARY: 0 errors
```

PTXAS reports no spills. Current key resources are:

| Kernel | Registers | Shared bytes | Barriers |
| --- | ---: | ---: | ---: |
| dependency initial / ready | 64 / 64 | 128 / 132 | 1 / 1 |
| direct numerical baseline | 32 | 128 | 1 |
| MoE initial / ready | 66 / 66 | 0 / 4 | 1 / 1 |
| MoE route / direct baseline | 40 / 40 | 256 / 0 | 0 / 0 |
| MoE all-expert copy / input producer | 22 / 13 | 0 / 0 | 0 / 0 |
| attention global initial / ready | 66 / 66 | 576 / 580 | 1 / 1 |
| attention TMA initial / ready | 68 / 68 | 8,840 / 8,840 | 1 / 1 |
| sparse attention initial / ready | 66 / 70 | 1,104 / 1,108 | 1 / 1 |
| sparse query producer | 14 | 0 | 0 |
| sparse overfetch / cache invalidation | 22 / 10 | 0 / 0 | 0 / 0 |
| NVMe application initial / ready | 60 / 62 | 256 / 264 | 1 / 1 |
| compatibility publication | 24 | 8 | 1 |
| host staging progress | 44 | 32 | 1 |
| NVMe progress | 53 | 0 | 0 |

The barriers in dependency, MoE, and attention kernels belong to numerical
cooperation. Acquisition returns before those barriers are reached. The TMA
barrier is initialized only after the common dependency set is ready and a
valid direct or staged tensor-map descriptor resolves.

Clang static analysis of the host runtime, KV benchmark, and MoE benchmark
reports no findings.

## Real FlashInfer Device-Selected Pages

ABI v20 adds a real FlashInfer pipeline in which
`top_k_page_table_transform` updates a stable CUDA index table, NTA validates
and gathers only those pinned-host KV pages, and a compiler-instrumented
FlashInfer paged-decode kernel consumes the compact KV. The NTA path performs
no host identity round trip. A separate offline oracle materializes the fixed
selected IDs before timing and copies only selected pages; it is a lower bound,
not an implementable online competitor.

```bash
tools/jit/activate.py --build-dir build --flashinfer-hook -- \
  python3 scripts/run-selected-pages-sweep.py \
  --output results/selected-pages-sweep-v20.json \
  --require-peak-speedup
```

The local sweep used 32 requests, 16 selected 16-token pages per request, ten
alternating in-process trials, and 20 iterations per sample. Selector,
acquisition, and attention are stream ordered in each timed pipeline. Every
point passed stock FlashInfer output parity.

| Candidate pages/request | Bytes avoided | Online mode | NTA cold us | Forced overfetch us | Forced-indexed speedup |
| ---: | ---: | --- | ---: | ---: | ---: |
| 16 | 0% | bulk | 267.242 | 171.250 | 0.641x |
| 32 | 50% | bulk | 271.141 | 320.498 | 1.182x |
| 64 | 75% | indexed | 271.656 | 616.583 | 2.270x |
| 128 | 87.5% | indexed | 271.666 | 1,208.602 | 4.449x |
| 256 | 93.75% | indexed | 271.450 | 2,395.940 | 8.826x |

The no-oracle cost model chose bulk at the first two points and indexed
transfer at the final three, so its minimum measured throughput ratio to forced
overfetch was 1.0x. Maximum NTA cold-pipeline regret to the precomputed
selected-copy oracle was 1.673x. Retained-object pipeline medians were 68.6 to
73.0 us.

This is a dirty-worktree, uncontrolled-clock operator crossover with controlled
random scores. It establishes a real performance domain and the dense
counterexample; it does not establish end-to-end SLO gain, model-quality impact,
NVMe benefit, production readiness, or superiority to Strata, ECHO, Syncopate,
or Prism.

## Query-Dependent Sparse Attention

The cold-cache sparse fixture materializes queries on device, scores resident
summaries, and selects two pages per request in the attention CTA. The
late-bound policy moves only those selected pages through the common request
ticket path. The matched overfetch policy copies every candidate page on a
second GPU stream concurrently with query production, then runs the same
selector and attention math. No CPU copy or selector is in either timed path.

Ten randomized process-level pairs ran 100 captured-graph iterations each with
16 requests. Clocks were uncontrolled and the worktree was dirty:

| Candidate pages | Policy | Median graph ms | Staged pages | Overfetch ratio |
| ---: | --- | ---: | ---: | ---: |
| 185 | late-bound | 0.12435 | 32 | 1.00x |
| 185 | overfetch | 0.07262 | 185 | 5.78x |
| 4,096 | late-bound | 0.38711 | 32 | 1.00x |
| 4,096 | overfetch | 0.82028 | 4,096 | 128.00x |

For the small catalog, `overfetch/late-bound` time was `0.5839x +/- 0.0038`, so
selective acquisition lost to the efficient bulk copy. For the large catalog,
the ratio was `2.1191x +/- 0.0017`, so late binding was faster while moving
`128x` fewer pages. Every trial had zero verification failures. This establishes
the expected selectivity crossover in a mechanism workload; it is not a
FlashInfer/SGLang sparse-attention or serving-SLO result.

## CPU DRAM Registration

The runtime supports both owned pinned allocations and non-owning registration
of allocations supplied by an engine. `HostMapped` exposes mapped pinned DRAM
directly to the consumer kernel. `HostStaged` publishes a bounded intent, copies
from mapped pinned DRAM into registered HBM from a finite GPU progress CTA, and
resumes the consumer only after system-visible readiness publication. The CPU
does not perform data movement in either path.

Two 96-object, 64-KiB, 50-iteration functional samples using non-owning
registration were:

| Placement | Source offset | Graph ms | Logical GiB/s | Staged issues | Live pending | Failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mapped CPU DRAM | 0 | 0.143 | 41.02 | 0 | 0 | 0 |
| staged CPU DRAM | 1 | 1.976 | 2.97 | 96 | 0 | 0 |

The one-byte offset forces the staged kernel's alignment-safe byte path. These
are smoke-test mechanism numbers, not controlled bandwidth results. Memcheck,
racecheck, and synccheck report zero errors on that unaligned path.

## Direct-Path Cost

Ten alternating process-level trials ran 200 captured-graph iterations each.
Both variants used the same graph topology and four-object numerical work; the
baseline bypassed request/acquisition logic only.

| Variant | Mean logical GiB/s | 95% t interval |
| --- | ---: | ---: |
| direct-address baseline | 367.86 | +/- 0.41 |
| historical ABI-v14 dependency set | 354.47 | +/- 0.45 |

The paired throughput reduction is **3.64% +/- 0.16 percentage points**. This
is a real, nonzero mechanism cost. GPU clocks were not fixed, this is a
microbenchmark rather than an untouched production kernel, and the interval
does not include machine-to-machine variation.

## Paged Attention

The workload uses FP16 Q/K/V, head dimension 128, 16-token pages, one CTA per
request-owned page, FP32 partial softmax state, and deterministic split-K
reduction. It forms the common plan from FlashInfer's public `kv_indptr`,
`kv_indices`, and `last_page_len`, then uses the same dependency records for
global loads or TMA.

The differential gate covers 7 heterogeneous requests, 23 physical pages,
non-identity page indices, and staged acquisition:

```text
flashinfer_version=0.6.12 requests=7 physical_pages=23
max_abs_error=2.71425e-05 mean_abs_error=3.2057e-06 matched=1
```

The optimized-kernel gate additionally executes NTA in FlashInfer 0.6.12:

```text
resident decode: pass
pinned-host deferred decode: Pending -> Ready -> Done, max error 0
shared KV-head CTAs: 2
split-K decode work items: 32, max error 0
FA2 paged-prefill work items: 4, max error 0
```

The matched custom-variant microbenchmark uses 64 requests and 2000 iterations
per alternating sample. The latest local median measured 12.250 us without NTA
fields and 12.817 us with the resident hook, a 4.63% cost. CTest fails above 8%.
Clocks are not fixed, so this is a local regression gate rather than a
paper-quality result.

Compute Sanitizer on the two-head deferred path reports memcheck 0 errors,
racecheck 0 hazards, and synccheck 0 errors.

One latest-code 8-request, 60-page, five-iteration mechanism sample was:

| Source | Consumer | Graph ms | Logical GiB/s | Max abs error |
| --- | --- | ---: | ---: | ---: |
| HBM | global loads | 0.023 | 19.66 | 2.42e-8 |
| HBM | TMA | 0.021 | 21.62 | 2.42e-8 |
| mapped CPU DRAM | global loads | 0.045 | 10.21 | 2.42e-8 |
| mapped CPU DRAM | TMA | 0.035 | 12.93 | 2.42e-8 |
| staged CPU DRAM | global loads | 0.108 | 4.24 | 2.42e-8 |
| staged CPU DRAM | TMA | 0.111 | 4.13 | 2.42e-8 |
| mixed | global loads | 0.084 | 5.42 | 2.42e-8 |
| mixed | TMA | 0.074 | 6.19 | 2.42e-8 |

These are smoke-test mechanism numbers, not controlled performance results.

## MoE Generality

The MoE gate regenerates hidden states and computes top-k routing on the GPU.
The router builds canonical work and dependency records without making selected
IDs CPU-visible. Compiler-lowered consumers acquire both selected matrices,
perform matrix-vector products, mix outputs, and check every element against a
CPU reference. `late-bound`, `cpu-sync`, and `overfetch` policies have separate
GPU CTests.

A developmental randomized 10-process run with 512 staged experts, eight
tokens, top-2 routing, hidden size 256, and 50 epochs measured a median 0.540 ms
for late-bound acquisition, 0.553 ms for CPU sync, and 2.669 ms for an
overfetch copy overlapped with routing. The paired median speedups were 1.023x
and 4.941x. Late-bound acquisition moved 4 MiB in the final epoch versus
overfetch's 128 MiB. A matched all-resident run measured 0.418 ms late-bound
versus 0.407 ms direct, or 2.7% overhead. All 50 runs had zero numerical
failures. Clocks were uncontrolled and the worktree was dirty, so
these values are mechanism evidence rather than qualification evidence. The
exact design and command are in `docs/DEVICE_ROUTED_MOE.md`.

## GPU-Initiated NVMe

The ABI-v20 controller-free test runs the same compiler-lowered finite CTA and
device transport code against an NVMe queue/control image in CUDA memory. It
checks that the CTA leader constructs a READ SQE and command context, rings the
SQ doorbell, leaves its work ticket `Pending`, and exits with no intent. The
test injects a phase-correct CQE, runs bounded completion plus publication, and
verifies ready-only numerical resume. A second epoch holds the queue lease,
verifies immediate intent fallback, releases the lease, and completes through
scheduled submission. Direct and fallback telemetry and credit release are
checked. A third epoch replaces the object and work ticket before injecting an
old CQE and verifies that the stale completion retires transport credits without
modifying either replacement. The same replacement is performed while an intent
is queued to verify source-routed stale-intent retirement. Additional epochs
inject an NVMe error status and an invalid command ID. A valid error retires
only its command; an invalid ID takes the queue offline and cooperatively
reclaims every active command context, intent, request/tenant/backend credit,
and dependent work ticket. A final epoch forces the queue control page to
`Fatal` and verifies that active queue ownership and queued NVMe intents are
retired without a stranded work ticket. A two-CTA launch then binds two
different requests and objects to one queue; at least one CTA submits directly,
and any lease loser publishes exactly one request-bound scheduler intent. This
is a deterministic protocol test, not NVMe performance evidence.

Historical ABI-v18 validation on 2026-08-01 used a private translated IOMMUFD IOAS and the explicit
`trusted-read-only-code` policy because `nvme id-ctrl` reports `NWPC=0`. The
bootstrap CPU READ qualified queue setup and DMA, and a separate GPU SQ-doorbell
probe produced a successful NVMe CQE before workload publication.

The compulsory-miss benchmark invalidates each retained staging entry inside
the captured graph before every epoch. It also verifies measured submission and
completion deltas equal `requests * iterations`, preventing cache-hit graph
replays from being mislabeled as storage bandwidth. One ABI-v18 run with an
eight-pass finite schedule produced:

| Path | Requests x bytes | Iterations | Mean graph ms | Physical MiB/s | Measured commands | Failed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mixed direct/fallback | 32 x 64 KiB | 1,000 | 1.231 | 1,624.42 | 32,000 / 32,000 | 0 |

The run recorded zero outstanding commands and zero checksum failures. A
separate four-pass, 200-epoch rerun failed closed after 6,344 of 6,400 measured
completions, demonstrating that external-latency variance can exceed a tight
fixed graph bound. The eight-pass run's one queue reached only about 13% of the
controller's nominal sequential bandwidth, so it establishes current-code
correctness and a scaling target while motivating adaptive graph rounds. It
does not establish statistical
superiority, serving-level SLO impact, portability, or production readiness.
The default hardware-write-protect policy still rejects this controller, as
designed.

## SGLang HiCache Integration

SGLang 0.5.14 discovers NTA through its public `sglang.srt.plugins` entry-point
group. The real model run binds engine request IDs and pool slots, registers
the exact HiCache host/device page rows, overlaps layer acquisition, executes
stock FA2 paged prefill after each layer is available, and returns SGLang-owned
output tensors. Separate correctness qualification compares every promoted K/V
row with pinned-host data; timed trials exclude that synchronous verification.

An historical residency-qualified mixed run used Llama-160M, one 96-token
host-cached request, one 64-token fresh request, 15 measured promotions, and
one generated token per request. Stock and NTA observed identical external
attempt indices and generated identical text. NTA reported 15 claimed batches,
180 planless preacquired attention launches, no plan uploads, and zero
fallback. Median batch latency was 14.762 ms stock and 14.079 ms NTA, a 4.62%
reduction and 4.85% promotion-throughput increase. A separate 15-promotion
single-request run measured 13.361 ms stock and 13.016 ms NTA, a 2.58%
reduction. Clocks were uncontrolled and the model is tiny, so these are local
regression results rather than serving-level or statistical superiority. A
subsequent full comparison measured 20.073 ms stock and 13.866 ms NTA, but the
per-request timestamps show that SGLang's fresh peer is sometimes co-batched
and sometimes delayed to a later scheduler step. The harness now reports that
peer delay separately and supports an explicit median-regression limit; the
30.92% aggregate difference is scheduler-sensitive and is not used as a paper
claim.

Those planless-instrumented measurements do not describe the current fast path.
The adapter now rejects instrumented attention after acquisition and the matched
harness requires a positive stock-FlashInfer launch count. Clean-revision
performance reports must use this implementation and the separate transfer
verification arm.

An ABI-v18 decode-graph comparison collected three qualified promotions in
five attempts with the same 96-token hot request and a 64-token peer. Stock
measured 18.614 ms median promotion latency and `nta_flashinfer` measured
11.469 ms, giving a 1.623x promotion-throughput ratio. Generated output and
external-attempt indices matched; NTA reported four captures, ten graph
replays, three claimed HiCache batches, and zero fallback. Host promotion ran
through eager prefill and the subsequent decode ran through the then-current
instrumented resident graph (`graph_external_batches=0`). This historical result
validated graph integration and request metadata preservation, but not
graph-captured demand acquisition or incremental co-scheduling. The current
preacquired graph uses stock FlashInfer. The sample is too small and clocks are
uncontrolled.

An earlier ABI-v20 schedule-aware path was measured with early host acquisition
disabled. It made its decision from the current FlashInfer schedule and exact
HiCache page mapping, selected one tuned bulk round, and overlapped subsequent
layer transfers with attention. Across five qualified promotions, stock
measured 12.993 ms and NTA measured 13.333 ms median promotion latency, a 2.62%
cost and 0.974x throughput ratio. Generated output and residency sequences
matched. NTA recorded five bulk batches, 60 layer prefetches, no CTA plan
uploads, no incremental work, and zero fallback. This is a current-ABI local
regression result for the old policy's no-opportunity branch, not an incremental
speedup or an OSDI result. It is retained as negative evidence and is superseded
as the dense production path by stock attention after stream-ordered
acquisition.

## Open Production And Paper Gates

- GPU-timestamped dense opportunity traces from two real models over CPU DRAM
  and NVMe, followed by the predeclared kill-criterion analysis;
- vLLM request/KV-manager integration, SGLang demand-mode graph phases, and
  paged-prefill graph validation;
- TTFT, TPOT, p50/p99, SLO attainment, serving goodput, CPU use, and SM tax;
- direct-path comparison against untouched production kernels with controlled
  clocks and multiple machines;
- a second typed generated-kernel frontend and automatic direct/incremental
  form generation rather than a single version-pinned FlashInfer overlay;
- host-staging global priority order, NVMe weighted-fairness hardware results,
  and starvation aging;
- GPU-initiated RDMA submission/completion is explicitly deferred; it requires
  a real RNIC before any RDMA claim;
- repeated VFIO hardware regression on multiple machines and SSD/topology
  combinations, direct-versus-scheduled statistical trials, injected
  timeout/reset/AER/hot-unplug recovery, translated-IOMMU fault tests, and
  multiple physical GPUs;
- production MoE and optional ANNS baselines; and
- literature-complete novelty analysis plus a paper-quality baseline and
  ablation matrix.

The installed vLLM 0.13.0 wheel requires PyTorch 2.9.0, while this environment
has PyTorch 2.11.0+cu130; its CUDA extension fails to load with an unresolved
symbol. A matched container or rebuild is required. SGLang 0.5.14 is ABI
compatible with the installed PyTorch 2.11.0+cu130 and FlashInfer 0.6.12. A
real Llama-160M run completed through both the stock FlashInfer backend and the
installed NTA plugin. This machine has no Mellanox/RDMA device. RDMA is outside
the current local-memory-and-storage scope.
