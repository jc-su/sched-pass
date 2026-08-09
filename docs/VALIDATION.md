# Validation Record

Date: 2026-08-09

This record supports a locally validated mechanism prototype. It does not
establish production readiness, an end-to-end serving result, or an OSDI-level
evaluation. The missing evidence is listed explicitly below.

Tables and command output retain ABI state `Ready` and legacy CLI policy
`late-bound` exactly as measured. The current design calls these available data,
runnable work, and device-generated demand; results are not renamed after
collection.

The source ABI is v25 and the public C API is v23 (v23 adds the per-step
indexed row-count bound the selected-demand loop uses to acquire only each
step's misses). ABI v25 makes rejected
same-generation epoch attribution observable, uses checked request-progress
subtraction, and keeps stale generations isolated from reused slots. ABI v24
added exact pending and expected compiler-attributed compute to the contract;
ABI v23 added a monotonic
runtime-lifetime failure counter and GPU-side indexed-directory rebinding used
by asynchronous layer execution. C API v22 adds a reusable, stream-ordered
pinned-host request-progress snapshot; the typed JIT operator-plan
query; v17 added the operator-contract query and capture-safe host-to-device
copy entry point. Focused compiler, C API, Python, and real FlashInfer JIT tests
plus the complete 46-test CTest suite have been rerun for v25. All executable
tests passed; the physical multi-GPU test was skipped on the one-GPU host.
Current-ABI VFIO correctness and bandwidth were rerun below; a clean-revision
repeat and multi-platform reliability campaign remain open.

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
The ABI-v23 local qualification completed successfully on 2026-08-03 and
reported `READY`. It covered the full functional/sanitizer matrix, CPU-only
build and tests, 10,000 lifecycle epochs, Clang static analysis, Python and
shell checks, package construction, and patch hygiene.

The schema-3 production and OSDI qualifiers both report `NOT_READY` on this
workspace. Production lacks its external evidence report and an immutable
clean revision. OSDI additionally lacks the paper evidence report. This is
the expected claim boundary, not a local test failure.

## Canonical FlashInfer Tier Streaming

On 2026-08-03, the request-aware bounded-HBM benchmark passed its separate
mechanism gate using FlashInfer 0.6.12 ragged prefill and
`merge_state_in_place`; it contains no custom attention kernel.

| Workload | Atomic promotion | Bounded stream | Speedup | 95% CI | Staging reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| q=256/request, context=64K/request, resident=0/25/50/100% | 6539.1 us | 5582.2 us | 1.1714x | [1.1660x, 1.1732x] | 4.0x |
| q=128/256/384/512, context=64K/48K/32K/16K | 5008.2 us | 4512.0 us | 1.1100x | [1.1050x, 1.1109x] | 4.83x |

The headline contains 10 arm-rotated trials and 10,000 deterministic paired
bootstrap resamples. Both arms match one complete all-HBM FlashInfer output
within the declared FP16 tolerance. Per-request completion follows data
availability rather than the slowest batch member. The local validator passes
all ten canonical-kernel, compiler-plan, graph, confidence, capacity, and
heterogeneity checks.

These artifacts were collected from a dirty development worktree and therefore
cannot satisfy release provenance. The runtime operator now owns FlashInfer
wrapper construction, wave metadata, copies, partials, merge, completion, and
dynamic-source graph replay. Paired JIT artifacts now validate the typed
request-coordinate and online-softmax execution plan and execute through the
LLVM-transformed path. SGLang does not yet consume this operator. The result
establishes the compiler/runtime operator opportunity and bounded-HBM crossover
only. Full
details are in `TIER_STREAMING.md`.

A separate transport ablation runs the same compiler-transformed operator with
GPU-initiated mapped-host copies. It is numerically exact and passes graph and
request-lifecycle checks, but measures 13.669 ms versus 6.550 ms atomic
copy-engine promotion (`0.479x`). This validates the GPU-originated CPU-DRAM
path but rejects it as the performance policy for known host-resident KV on this
platform.

The checked-in equal-state runner specification also completed 10 randomized
process-level blocks against ABI v24 on this workspace. The long-context paired
ratio has median `1.1672x` (range `[1.1609x, 1.1688x]`), and the heterogeneous
ratio has median `1.1164x` (range `[1.1153x, 1.1218x]`). The run used
uncontrolled clocks and `--allow-dirty`; it is a reproducibility check, not
OSDI qualification evidence.

## Correctness Gates

`ctest --test-dir build --output-on-failure` discovers 47 tests. On this
single-GPU host, 45 pass and `nta-multi-gpu` reports the configured skip code
77 because a second physical CUDA device is unavailable. The gates cover:

1. FlashInfer CSR-to-common-plan validation, including grouped pages and bad
   metadata;
2. engine-neutral plan construction and bounded dependency validation;
3. LLVM byte-address, tensor-map, and dependency-set lowering, including
   rejection of live-state, token, missing-binding, non-inlined-helper,
   lane-divergent control, non-dominating divergent control, and divergent PHI
   operand cases;
4. host/device ABI v25 layout, including operation epochs, checked progress
   conservation, dropped-attribution telemetry, terminal counters,
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
bind/acquire/defer markers and carry ABI-v25 `!nta.acquire` metadata tagged
`split-phase-cta`.

Incremental kernels additionally delimit one numerical region with convergent
begin and publication markers. The pass rejects a region that does not follow
the acquired edge with the same request binding and work ticket, a publication
that can be bypassed, duplicate or missing publication, nonuniform operands,
and non-convergent endpoints. Valid publication lowers to
`nta_commit_partial` with `!nta.partial` metadata and a function-level
`!nta.operator` contract; the runtime then validates request generation,
contributor identity, and reduction-group completion.

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

The eager path now also verifies a compact initial wave followed by a compact
resume at a nonzero runnable-queue offset. Discovery publishes direct and
preloaded work exactly once without constructing suspended dependency state;
unavailable work still enters the full object/version and generation-checked
ticket protocol. The real FlashInfer differential test covers resident plus
offloaded requests in this form and matches stock output.

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

ABI v25 retains the real FlashInfer pipeline in which
`top_k_page_table_transform` updates a stable CUDA index table, NTA validates
and gathers only those pinned-host KV pages, and a compiler-instrumented
FlashInfer paged-decode kernel consumes the compact KV. The NTA path performs
no host identity round trip. A separate offline oracle materializes the fixed
selected IDs before timing and copies only selected pages; it is a lower bound,
not an implementable online competitor.

```bash
tools/jit/activate.py --build-dir build --flashinfer-hook -- \
  python3 scripts/run-selected-pages-sweep.py \
  --output results/selected-pages-sweep-v25-corrected.json \
  --require-peak-speedup
```

The local sweep used 32 requests, 16 selected 16-token pages per request, ten
alternating in-process trials, and 20 iterations per sample. Selector,
acquisition, and attention are stream ordered in each timed pipeline. Every
point passed stock FlashInfer output parity. Both the indexed and all-page
policy arms use compiler-transformed FlashInfer attention, and their versioned
direct/incremental operator contract pair is checked before timing.

| Candidates/request | Avoided | Mode | Online us | Forced indexed us | Overfetch us | Candidates resident us | Speedup vs overfetch (95% CI) |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 0% | bulk | 187.988 | 292.308 | 187.988 | 31.765 | 1.0000x [1.0000, 1.0000] |
| 32 | 50% | indexed | 295.091 | 295.091 | 331.465 | 30.821 | 1.1233x [1.1220, 1.1269] |
| 64 | 75% | indexed | 295.122 | 295.122 | 627.390 | 32.235 | 2.1259x [2.1241, 2.1296] |
| 128 | 87.5% | indexed | 295.462 | 295.462 | 1,217.003 | 31.272 | 4.1190x [4.1157, 4.1215] |
| 256 | 93.75% | indexed | 294.434 | 294.434 | 2,406.441 | 31.254 | 8.1731x [8.1639, 8.1762] |

The no-identity-oracle cost model chose bulk only when all candidate pages were
needed and selected indexed acquisition at 50% selectivity and above. Policy
regret is computed from the chosen and best arms in the same rotated trial; it
was 1.0000x at every measured point. The zero-avoidance online ratio is exactly
1.0000x because the policy chooses bulk, while forcing indexed acquisition is
only 0.6431x. Maximum cold-pipeline regret to the precomputed selected-copy
oracle was 1.721x. The all-candidate-resident arm measured 30.8-32.2 us, making
cold indexed acquisition 9.15x or more slower; this is an optimization gap, not
a competing equal-capacity policy. The executable sweep gate requires at least a five-point crossover,
2x speedup at 75% or greater bytes avoided, at most 1.05x policy regret, at
most 2x oracle regret, exact output, transformed policy forms, and a peak 95%
interval above one.

This is a dirty-worktree, uncontrolled-clock operator crossover with controlled
random scores. It supports the scoped claim that compiler/runtime paired
acquisition forms avoid unnecessary transfer when object identity is produced
on device and that an online bulk/indexed decision avoids the dense indexed
penalty. It does not show that the indexed mechanism itself preserves dense
efficiency, nor does it establish end-to-end
SLO gain, model-quality impact, NVMe benefit, production readiness, or
superiority to Strata, ECHO, Syncopate, or Prism.

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
per alternating sample. The latest three consecutive cached runs measured
1.76%-2.59% overhead for the compiler-transformed request-bound direct form;
CTest fails above 5%. The full incremental work-plan form measured 5.82%-6.41%
and is reported as a separate diagnostic rather than standing in for resident
direct execution.
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

The ABI-v25 controller-free test runs the same compiler-lowered finite CTA and
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

On 2026-08-09, the ABI-v25 one-GPU qualification harness reran a matched
read-only `fio io_uring` baseline and the GPU-controlled VFIO path with 2-MiB
requests at queue depth 32. The controller was attached to a private translated
IOMMUFD IOAS and the BAR-write qualification passed. Across 20 measured GPU
epochs, all 640 commands completed with exact checksums, zero failures, and
zero outstanding commands. The CPU baseline measured 11,812.71 MiB/s and the
GPU-controlled finite scheduler measured 6,870.00 MiB/s, or 58.16% of the
matched baseline. The artifact names clean revision `5c26f8b8aa6c` and exceeds
the predeclared 50% local scaling gate. This remains one controller and one
trial; it is not serving-SLO, portability, or recovery evidence.

## SGLang HiCache Integration

SGLang 0.5.14 discovers NTA through its public `sglang.srt.plugins` entry-point
group. The real model run binds engine request IDs and pool slots, registers
the exact HiCache host/device page rows, overlaps layer acquisition, and returns
SGLang-owned output tensors. NTA always invokes a compiler-transformed FA2
wrapper: transformed direct for resident or one-round ordered data, and
transformed incremental for unavailable request/tile work. The matched harness
rejects stock NTA launches and fallback. Separate correctness qualification
compares every promoted K/V row with pinned-host data; timed trials exclude that
synchronous verification.

Current ABI-v23 smoke measurements are summarized below. The tree was dirty,
clocks were uncontrolled, and the newest fragmented rows have one qualified
external attempt each. They are integration diagnostics, not qualification
data:

| Workload | Stock | NTA | Throughput ratio | Mechanism counters |
| --- | ---: | ---: | ---: | --- |
| Qwen2.5-3B resident, full decode graph, warm JIT cache | 56.791 ms | 55.618 ms | 1.021x | 4,068 direct, 36 captures, 32 replays, 2 contracts, 0 stock/fallback |
| Llama-160M resident | 26.325 ms | 26.392 ms | 0.997x | 228 transformed direct, 0 stock |
| Llama-160M mixed host/resident | 21.013 ms | 21.057 ms | 0.998x | 288 transformed direct, 36 promoted layers |
| Qwen2.5-3B mixed host/resident | 67.027 ms | 62.453 ms | 1.073x | 684 transformed direct, 72 promoted layers |
| Llama-160M forced ticketed | 25.993 ms | 34.584 ms | 0.752x | 12 ticketed layers, 1 structural plan |
| Qwen2.5-3B 4K mixed load | 386.47 ms | 415.96 ms | 0.929x | 2,052 transformed direct, 108 promoted layers, 0 stock/fallback |
| Qwen2.5-3B 8K saturated | 1,331.7 ms | 1,473.7 ms | 0.904x | 5,616 transformed direct, 144 promoted layers, 0 stock/fallback |
| Qwen2.5-3B 8K mixed, two fragment waves | 73.750 ms | 85.473 ms | 0.863x | 2,304 work items, 36 ticketed layers, 35 first-wave lookaheads, 0 stock/fallback |
| Qwen2.5-3B 16K mixed, two fragment waves | 110.205 ms | 118.824 ms | 0.927x | 4,500 work items, 36 ticketed layers, 35 first-wave lookaheads, 0 stock/fallback |
| Qwen2.5-3B 16K compact fragments | 113.150 ms | 121.043 ms | 0.935x | 50.01% combined physical CTA bound, lightweight available-work publication, 0 stock/fallback |
| Qwen2.5-3B 2K coalesced requests | 374.601 ms | 393.013 ms | 0.953x | 36 mixed/ticketed/parallel-progress layers, 50.0% combined CTA bound, 0 stock/fallback |
| Qwen2.5-3B 4K, four resident peers | 446.294 ms | 484.507 ms | 0.921x | 36 mixed/ticketed layers, 132 direct + 32 external work items, 0 stock/fallback |
| Qwen2.5-3B 2K acquisition frontier v10 | 204.643 ms | 209.384 ms | 0.977x | 1 mixed ticketed layer, 35 preacquired suffix layers, 0 stock/fallback |
| Qwen2.5-3B 2K coalesced, finite demand graph replay | 347.890 ms | 379.436 ms | 0.917x | clean `ae7c56a`; 3 mixed, 10 ticketed, 3 captures/6 graph launches, 50% combined CTA bound, 0 stock/fallback |

Generated output matched in every row and fallback was zero. The isolated
`1.073x` Qwen result is only a smoke point; it is too small and noisy for a
performance claim. The forced ticketed and load results are negative evidence:
one known bulk transfer round followed by transformed layer-complete attention
does not repay NTA's planning and launch cost. An earlier Qwen ticketed run selected two
rounds and reached only 0.676x throughput; it exposed a missing host plan-setup
term in the cost model. The model now conservatively includes that term.

The resident graph row is a three-sample integration smoke. Every attention
launch used a contract-validated transformed module, and eager-only reruns show
the transformed kernel itself is not the source of the earlier overhead. The
row performs no ticketed external work, so it satisfies neither the incremental
serving gate nor an end-to-end performance claim. `CompareSglang.py` now primes
each backend's JIT cache in unmeasured processes and requires graph
capture/replay counters when full decode graphs are requested.

The older mixed-load rows executed no ticketed partial work: all attention
launches were transformed direct launches after whole-layer promotion. Those rows
falsify mover-only and admission-only versions of the design. The ABI-v23 8K
and 16K rows do execute real FlashInfer split-K contributors in two waves. A
GPU directory rebind stages the first wave of layer `L+1` during post-attention
compute; the next initial launch consumes it while one finite progress wave
acquires the rest. Exact generations match stock and every NTA attention launch
is transformed. The mechanism is therefore active and correct, but it remains
15.9% slower at 8K and 7.8% slower at the first 16K point. Physical compaction
and lightweight publication reduce the newest 16K cost to 7.0%, but do not
produce a win. The next gate is a repeated measured arrival-skew crossover
against layer wait and skip/rebatch, not another forced dense point.

The coalesced load fixture now enables SGLang mixed-chunk batching and releases
the host-hit request into the running resident batch through the acquisition
admission hook. `NTA_SGLANG_REQUIRE_MIXED_ATTENTION=1` fails closed unless the
real FlashInfer schedule contains both direct and external request work. The 2K
run above proved 17 direct and 16 external work items in one schedule; the 4K
four-resident run proved 132 direct and 32 external work items. Both execute the
compiler-generated direct and incremental forms, so they are valid
cross-request mechanism evidence. They are also negative performance evidence:
the current v10 2K point is 2.26% slower overall, has 1.16% worse resident P99
inter-token latency, and has 2.08% worse external TTFT. The four-resident 4K
point is 7.9% slower.

The last ABI-v24 full-graph path batches four proactive layers per GPU mover
wave and uses the warp-cooperative request guard. Three arm-balanced 4K trials
reported `0.9447x` output-throughput geometric mean with bootstrap interval
`[0.9337x, 0.9578x]`, `1.3386x` external TTFT, and `1.4991x` resident P99
inter-token latency. SLO goodput was `0.7498x`. Outputs were exact, all
attention was transformed, and fallback was zero. Three repetitions are
diagnostic negative evidence for dense, early-known host demand, not a
paper-level confidence claim; they do not invalidate the separately measured
device-selected crossover.

The clean ABI-v25 row excludes two mixed-arrival warmup occurrences so the
timed occurrence uses the finite demand graph rather than charging first-use
setup. External TTFT was `1.1014x`, resident P99 ITL was `1.8843x`, and
stock-derived SLO goodput was `0.4584x`. This rules out missing demand-graph
capture as the complete explanation for the dense loss. It does not establish
a repeated confidence interval or an end-to-end device-selected-demand win.

The clean `bec105f` three-trial diagnostic
(`results/serving/sglang-hicache-load-bec105f-qualification.json`) exercised
both compiler forms with graph replay, physical CTA compaction, exact outputs,
and zero stock or fallback launches. Geometric means were `0.9791x` output
throughput, `0.7771x` SLO goodput (one trial crossed the resident-ITL SLO
threshold, so the two-request goodput metric amplifies a single violation),
`0.9557x` resident P95 TTFT (better), `1.0394x` external P95 TTFT, and
`1.2549x` resident P99 inter-token latency. Mechanism counters explain the
shape: of 5,148 attention launches, 5,138 used the direct form and only 10 were
ticketed incremental, because 350 of 360 external layers were satisfied by
proactive lookahead acquisition before attention reached them and only 3 layers
were mixed. The run therefore measures acquisition, planning, and mover
interference around nearly all-direct attention; it is boundary evidence that
dense, early-known demand exposes almost no incremental opportunity, not a
measurement of incremental execution itself. The consistently regressed metric
is resident P99 inter-token latency, and the acquisition mover streams ran at
elevated CUDA priority (`-1`) above the decode stream in every dense run
recorded above; movers now default to the lowest priority
(`NTA_SGLANG_MOVER_STREAM_PRIORITY=0`).

The corrected-priority interference series ran on 2026-08-09 as the first
ten-trial, arm-balanced, `evidence_grade=qualified` measurement of this
workload (`results/serving/sglang-hicache-load-moverfix-qualification.json`,
clean revision, exact outputs, all attention transformed, zero fallback).
Resident P99 inter-token latency remained **1.2407x** geometric mean, median
1.1763x, bootstrap CI [1.1433, 1.3801]; output throughput was 0.9402x
[0.9174, 0.9642]; external P95 TTFT 1.1219x; goodput 0.8185x, with two trials
(1.754x and 1.586x ITL spikes) crossing the SLO threshold and driving the
goodput tail. The mover-priority hypothesis is therefore **measured and
rejected as the dominant cause**: lowering mover priority did not close the
tail regression. Remaining suspects are copy-kernel SM residency (priority
arbitrates issue, not occupancy), memory-bandwidth contention during decode
steps, and per-layer host planning work; decomposing them is the RQ4
interference ablation. The 1C resident-tail gate (<=1.05x) is failed at this
configuration and remains failed until that decomposition lands.

Both pre-declared decomposition hypotheses then resolved negative at n=10
(`results/serving/sglang-hicache-load-moverpri-neg1-qualification.json`,
`.../sglang-hicache-load-wave1-qualification.json`). H-A: elevated mover
priority measured 1.1357x [0.928, 1.380] — indistinguishable from lowest
priority, so stream scheduling is irrelevant to the regression in either
direction. H-B: one-layer waves measured 1.1667x [0.921, 1.431] — burst size
has no clear effect, and the extremes widened (two trials near 2x, one at
0.484x where the NTA arm's tail beat stock two-fold; one trial's NTA arm
missed every SLO and is recorded as `zero_goodput_trials` rather than
crashing the aggregate). The spread exposes a measurement limitation: with 32
resident output tokens, P99 inter-token latency is effectively the maximum of
about 31 intervals, so single scheduler hiccups in either arm dominate the
statistic. Before any further interference conclusion, the series must be
re-run with a longer decode window (>=256 resident output tokens) so the tail
is a distribution property rather than an extreme; the remaining mechanism
suspects after H-A/H-B are per-transfer PCIe/memory contention and host-side
planning jitter, discriminated next by a copy-engine wave path.

The 256-token long-window series then ran at n=10
(`results/serving/sglang-hicache-load-longwindow-qualification.json`,
qualified, exact outputs, zero fallback). It splits the tail question in two.
Typical-case interference is resolved: median resident P99 inter-token
latency is **1.028x**, and four of ten trials measured the NTA arm's tail
*better* than stock (0.58-0.78x). The geometric mean (1.1614x, CI [0.836,
1.669]) is dominated by three episodic trials (2.19x, 2.35x, 3.25x, one of
which also degraded both arms' TTFT and zeroed goodput). Forensically, the
mechanism counters are identical across all ten trials — 3 graph captures, 4
warmups, 6 replays, 10 ticketed incremental launches, 350 lookahead layers, 3
mixed layers in pathological and healthy trials alike — so the episodes are
not produced by variable acquisition work. The pathology is host- or
scheduler-level variance striking ~3/10 runs. Next diagnostics: per-token
timeline forensics of the pathological artifacts (spike position versus
external arrival and churn phases), then a repeat with host-side controls
(CPU pinning, allocator preallocation). The 1C gate is passed on the median
and failed on the mean until the episodic source is identified; both numbers
are reported.

Per-token forensics of the long-window artifacts sharpen the episodic
signature. Median inter-token latency is identical in both arms of every
trial (10.2-11.5 ms), including the pathological ones. The pathological NTA
arms show two-to-four *consecutive* stretched intervals (43.2/42.6/42.2 ms at
22% of the timeline; 55.9/35.4 at 49%; 37.0/26.0/24.2 at 41%) — one sustained
roughly-100 ms disruption landing mid-decode during the external-arrival and
claim phase — while stock arms in other trials spike up to 37.7 ms with
different placement. Pre-declared hypothesis **H-C**: the claim-time
registration and planning burst serializes the scheduler thread — the pinned
directory-upload ring is compile-time depth 4 (`Runtime.cpp`,
`DirectoryUploadDepth`) and event-synchronizes when exhausted — and when that
burst's stochastic timing collides with resident decode steps, consecutive
intervals stretch. Discriminating tests: deepen or make configurable the
upload ring, move claim-time registration off the scheduler thread, and
re-run the long-window series; the hypothesis is falsified if the clustered
mid-timeline spikes survive both changes.

**H-C is confirmed by the first discriminating test.** With the ring at depth
64 (`NTA_DIRECTORY_UPLOAD_DEPTH=64`, same protocol, n=10, qualified, exact
outputs, zero fallback,
`results/serving/sglang-hicache-load-ring64-qualification.json`), the
episodic spikes vanish entirely: every trial's resident P99 inter-token
ratio lies in [0.938, 1.198] versus the depth-4 series' 2.19-3.25 events,
no trial zeroes its goodput (goodput geometric mean 0.9318), and the ITL
geometric mean improves from 1.1614 to **1.0821** with interval [1.034,
1.128]. The compile-time depth-4 recycling synchronization on the engine
scheduler thread was the episodic tail pathology; the default ring depth is
now 32 with the environment override retained. The residual, now-consistent
~8% tail cost against the 5% 1C gate is a steady-state optimization target
(claim-work off the scheduler thread and the copy-engine wave path are the
next candidates), no longer an episodic failure mode.

Resident TTFT is not used as the interference headline: the workload submits
external requests only after a resident emits its first token, so that TTFT is
causally prior to external acquisition. The benchmark now reports per-request
and aggregate P99 inter-token latency and includes it in the SLO-goodput gate.
This change prevents scheduler startup variance from being mistaken for an
acquisition benefit.

The RQ2/2A barrier characterization
(`benchmarks/serving/OpportunityCharacterize.py`,
`results/serving/opportunity-characterization-qwen3b.json`) measured the
compute-stream stall at every proactive layer-readiness wait on the real
Qwen2.5-3B HiCache workload at external prefixes of 2,048, 8,192, and 24,576
tokens. Every one of 360 waits per point measured **0.000 ms** of stall. The
explanation is the load/compute ratio: wave-pipelined promotion moved
1.0-7.2 GiB in 29-168 ms while the attention operators consumed 316-3,238 ms,
a ratio of 0.05-0.09 that falls as context grows because promoted-prefix
attention grows superlinearly while transfer grows linearly. Consequences:
prefill-side promotion on this host exposes no blocked time for arrival-driven
execution to reclaim, at any context size, because the existing lookahead
pipeline already achieves complete overlap; the 1.1714x streaming-operator
result is real but its baseline (atomic promotion) is not what the engine
deploys; and the streaming-operator integration into dense SGLang promotion is
therefore cancelled by measurement rather than attempted. The remaining
candidate regime for arrival-driven benefit on this stack is per-step
device-selected demand, where per-launch compute is small and per-step
transfer is the object. Single model, CPU-DRAM tier, copy-engine movers, warm
JIT cache; the conclusion is structural (an identical zero across 1,080
measured waits), so clock control is immaterial to it.

A stronger acquisition-admission experiment detached the mixed request until
a complete suffix-layer lead was available and then re-formed the same
compiler-transformed heterogeneous batch. Five arm-balanced exact trials had
zero fallback and all attention transformed, but geometric-mean throughput was
0.972x, resident P99 inter-token latency was 1.048x (95% bootstrap interval
1.010x-1.091x), and external TTFT was 1.295x. That policy was removed from the
source because it shifted delay instead of reducing it. The raw rejected-policy
artifact remains at
`results/serving/sglang-qwen3b-abi23-admission-frontier-v10-2k-qualification.json`.
The current immediate-mixed v10 diagnostic is
`results/serving/sglang-qwen3b-abi23-frontier-v10-immediate-32tok.json`.

Demand host progress no longer copies a megabyte-scale indexed object with one
CTA. It uses three bounded stream-ordered stages: intent/credit claim and index
validation, a 16-CTA-per-object row-specialized copy, and post-copy object/ticket
publication. The real FlashInfer differential covers successful and invalid
indexed ranges through this path. A diagnostic run measured 8.79 GiB/s across
the combined bulk-warmup and demand transfers versus 8.27 GiB/s before the
change; because that old profile did not separate the two transfer classes, the
number is not a standalone demand-bandwidth claim.

Historical 18.614 ms versus 11.469 ms and 13.333 ms versus 12.993 ms results
used stock attention for part of the NTA arm. They are rejected as evidence for
the compiler/runtime contribution and must not be used in a paper claim.

## Open Production And Paper Gates

- GPU-timestamped dense opportunity traces from two real models over CPU DRAM
  and NVMe, followed by the predeclared kill-criterion analysis;
- vLLM request/KV-manager integration and whole-model SGLang demand capture;
  exact-shape finite demand decode/paged-prefill operator graphs are implemented
  but still require clean multi-trial serving evidence;
- TTFT, TPOT, P99 inter-token latency, SLO attainment, serving goodput, CPU
  use, and SM tax;
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
