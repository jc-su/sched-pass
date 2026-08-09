# Nonresident Acquisition

This branch implements request-aware incremental execution for finite GPU
kernels whose tensor data arrives from heterogeneous memory and storage tiers.

The primary application is SLO-critical external KV access in continuously
batched attention. The mechanism also targets GPU-routed MoE experts and, as a
secondary generality case, graph or ANNS objects.

The problem is an all-or-nothing kernel barrier. Serving engines wait for every
input before invoking an optimized attention operator, even though its compiled
implementation contains request-owned chunks that could perform useful partial
work as their data arrives. Userspace knows request lifecycle and SLOs but not
the kernel's safe tile and reduction boundaries. The kernel knows those
boundaries but not request policy or transport delay. NTA preserves
`(epoch, request slot, generation, logical tile, object, version)` across the
compiler, runtime, transport, and engine so they can jointly schedule arriving
data, runnable tiles, and later batch admission. This incremental operator
co-design, not a generic I/O ABI or sparse-attention policy, is the candidate
systems contribution.

The architecture contract and implementation sequence are defined in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
The canonical FlashInfer bounded-HBM crossover, request semantics, reproduction
command, and remaining compiler/SGLang boundary are in
[docs/TIER_STREAMING.md](docs/TIER_STREAMING.md).
The revised motivation, incremental-execution co-design, scalable implementation
order, and claim gates are defined in
[docs/SYSTEM_PLAN.md](docs/SYSTEM_PLAN.md).
The exact tested environment and current measurements are in
[docs/VALIDATION.md](docs/VALIDATION.md).
The current closest-work and novelty boundary is in
[docs/RELATED_WORK.md](docs/RELATED_WORK.md).
The implemented GPU-routed MoE generality mechanism, matched baselines, and
developmental performance result are in
[docs/DEVICE_ROUTED_MOE.md](docs/DEVICE_ROUTED_MOE.md). It is not the primary system
story.
The implemented FlashInfer boundary and remaining serving integration are in
[docs/FLASHINFER.md](docs/FLASHINFER.md).
The installed SGLang plugin, supported profile, fallback contract, and matched
serving benchmark are in [docs/SGLANG.md](docs/SGLANG.md).
The compact host/kernel API, JIT path, and exact transparency contract are in
[docs/INTEGRATION.md](docs/INTEGRATION.md).
The NVMe threat model, containment requirements, and residual trust boundary
are in [docs/NVME_SECURITY.md](docs/NVME_SECURITY.md).
Executable local, production, and OSDI claim gates are defined in
[docs/QUALIFICATION.md](docs/QUALIFICATION.md).
The controlled single-GPU experiment matrix and defensible efficiency claim
are defined in [docs/ONE_GPU_EVALUATION.md](docs/ONE_GPU_EVALUATION.md).

## Implemented vertical slice

The branch contains a working LLVM 22 new-PM pass and real Blackwell CUDA
workloads:

- per-CTA request/generation binding in batched kernels;
- compiler proof of a canonical finite-kernel deferral boundary;
- compiler-verified convergent numerical regions whose publication
  post-dominates acquired execution, with request/reduction operands lowered to
  `!nta.partial` and `!nta.operator` contracts;
- an engine-neutral `WorkPlan` model and ABI-v25 bounded dependency sets, so one
  work ticket can wait for several pages, experts, or object shards;
- one reusable device-plan allocation with two pinned, asynchronous upload
  slots and no unconditional hot-path event synchronization;
- a versioned shared-library C API and owning Python binding for engine
  allocations, work plans, finite JIT phases, work ticket state, and VFIO NVMe;
- a schema-versioned JIT operator contract that records runtime ABI, operator
  family, direct/incremental form, capabilities, and a paired source
  fingerprint; native, Python, SGLang eager, and SGLang graph paths validate it
  before launch;
- a typed operator execution plan that fixes request-coordinate mapping,
  online-softmax partial state, deterministic merge semantics, graph stability,
  generation binding, and source/plan fingerprints across paired forms;
- zero-copy DLPack views of the native runtime and plan plus a bounded
  FlashInfer layer executor whose fixed enqueue path can be captured after its
  structural plan is uploaded;
- full SGLang decode CUDA-graph replay through compiler-transformed FlashInfer
  wrappers after stream-ordered acquisition, with exact live request metadata
  preserved across the engine's padded replay view;
- non-owning registration of existing engine HBM and device-visible host
  allocations;
- public finite-kernel host and device policies, replacing benchmark-specific
  phase launch and marker plumbing;
- direct HBM and mapped-CPU-DRAM paths with no queue or atomic operation;
- staged CPU-DRAM acquisition issued and completed by finite GPU CTAs;
- GPU-initiated NVMe reads into registered mapped DRAM, with per-open queues,
  GPU-built SQEs/PRPs, GPU MMIO doorbells, and GPU CQ handling;
- an optional one-shot NVMe submission directly from a missing application CTA,
  with pre-lease backend demand accounting, a non-spinning queue lease, and
  scheduled fallback under contention, older published work, or exhausted
  credit; two request CTAs contending for one queue are covered by the GPU test;
- a reusable, object-keyed intent pool, duplicate-object coalescing,
  cancellation, fully published generation-safe work tickets, and stale
  completion isolation;
- completion-driven reverse dependency edges, direct exact-once runnable-work
  publication, and tagged urgency queues, so normal progress scales with
  arrivals rather than scanning configured ticket or intent capacity;
- per-request blocked-byte, pending/runnable/completed compute, expected and
  terminal work summaries, checked conservation, dropped-attribution telemetry,
  and request-local reduction counters;
- fixed request, tenant, and backend byte credits plus priority/deadline NVMe
  admission;
- a backend-neutral directory with bounded per-object physical replicas;
- a numerically checked split-K paged-attention workload with heterogeneous
  request lengths that consumes the common work/dependency ABI for both global
  loads and TMA;
- a numerically checked query-dependent sparse-attention workload in which an
  upstream device kernel materializes each query, the attention CTA selects
  top-k pages and acquires them without a host round trip, and a ready launch
  reconstructs deliberately permuted request-slot/generation bindings, plus a
  same-selector overlapped all-page GPU overfetch control;
- a numerically checked GPU-routed top-k MoE workload that builds canonical
  work/dependency records on device and uses the same compiler pass, tier
  directory, finite progress path, and ready scheduling;
- an optional adapter from FlashInfer's public paged-KV CSR tables into the
  common work model, FlashInfer-native `(V, LSE)` cascade state, and a
  differential GPU correctness gate against FlashInfer 0.6.12;
- a real FlashInfer device-demand path whose GPU top-k page-table transform
  updates a stable device index table, whose bounded indexed acquisition moves
  only selected pinned-host KV pages, and whose compiler-instrumented paged
  decode consumes the compact KV without a host identity round trip;
- one no-oracle cost model that dispatches bulk candidate transfer when
  selectivity is low and GPU-indexed transfer when avoided bytes amortize its
  fixed cost;
- real TMA consumption from direct sources or from HBM after external staging,
  with compiler-proven deferral before barrier creation;
- one captured CUDA graph containing discover, bounded progress, and resume
  kernels;
- an inspectable NVVM IR -> pass -> PTX -> cubin build pipeline;
- automatic optimizer-last lowering in Clang JIT builds, with an nvcc-compatible
  compiler shim and an ABI-fingerprinted FlashInfer cache;
- phase-aware FlashInfer split-K execution that refuses to merge each request's
  decode or paged-prefill rows until that request's current-generation
  contributors complete, without blocking complete peers in the same merge;
- request-bound FlashInfer waves that preserve the original request/tile
  schedule and admit contributors from current-generation ticket state while
  transfer and compute streams overlap;
- an installed SGLang 0.5.14 plugin backend that binds real request IDs,
  generations, and priorities for every batch, overlaps HiCache's tuned layer
  copies with model execution, executes resident/preacquired work through the
  transformed direct form, routes unresolved work through generation-keyed
  tickets, mirrors request aborts, reuses one structural plan across layers,
  and fails closed instead of switching to stock attention;
- a hard capacity and high-water telemetry for HBM staging allocations owned by
  the runtime; engine-owned KV staging remains governed by the engine cache.

The core ABI and compiler pass do not depend on FlashInfer, vLLM, SGLang, or a
kernel name. A supported kernel must still expose a reconstructible pre-state
work boundary; arbitrary instruction-level suspension is not implemented. The
version-checked FlashInfer 0.6.12 JIT overlay places deferral before state
initialization in optimized decode and FA2 paged-prefill CTAs without modifying
the installed package. The native/Python engine boundary and eager SGLang
HiCache adapter are implemented. vLLM is not registered: vLLM 0.13 exposes an
experimental KVConnector lifecycle, but its stock offload path completes loads
before attention and its Blackwell FlashInfer backend normally selects TRTLLM
entry points outside the current hook. The compiler currently consumes explicit
acquisition and partial-region markers; automatic recognition of arbitrary
production load/cp.async/TMA address cones is an open gate. The pass proves
CTA-uniform acquisition control and operands, rejects lane-divergent collective
calls, and rejects partial regions with non-convergent endpoints, mismatched
request/work-ticket identity, an acquisition bypass, a bypassed publication,
or multiple publications. FlashInfer demand kernels use compiler-lowered
publication instead of updating reduction counters in adapter code.
The real FlashInfer GPU-selected path is an operator-level result, not a claim
that the current SGLang path uses sparse attention or improves an end-to-end
serving SLO. Dense SGLang remains the production integration target. The older
whole-layer CPU-DRAM path remains a negative baseline. A newer canonical
FlashInfer bounded-HBM experiment establishes reusable numerical overlap.
After moving wrapper construction, copy-slot lifetime, partial execution,
merge, and graph-safe source rebinding into the runtime operator, exact
request-aware streaming is `1.1714x` faster than atomic promotion with a 95%
confidence interval of `[1.1660x, 1.1732x]` and `4x` lower staging capacity.
The heterogeneous-shape run is `1.1100x` faster with `4.83x` lower staging.
Both positive arms use compiler-transformed canonical FlashInfer and validate a
paired typed execution plan, dynamic-source graph replay, generation reuse, and
cancellation isolation; custom CUDA attention and MoE programs remain
correctness fixtures rather than production performance evidence.
The same operator also has a real GPU-initiated mapped-host producer. It passes
the graph, lifecycle, and numerical gates, but measures `0.479x` versus atomic
copy-engine promotion on this host. CPU DRAM therefore remains copy-engine
driven; GPU initiation is retained for device-discovered demand and transports
whose queue semantics justify it, rather than being claimed as universally
faster.
The existing code implements the unavailable-data work-ticket mechanism,
compiler-verified request-local partial reduction, real FlashInfer hooks, and
SGLang decode graph replay. Demand mode reuses one structural plan, rebinds
layer K/V directories on the GPU, and stages one next-layer contributor wave
during post-attention compute before progressing the remaining waves. A
generation-checked post-discovery snapshot now feeds SGLang's external-batch
admission decision. Exact-shape demand decode and paged-prefill epochs warm,
capture, and replay as finite NTA operator graphs with retained FlashInfer
metadata; this is separate from SGLang's full model graph, whose external path
still requires preacquisition. The latest matched fragmented
Qwen2.5-3B CPU-DRAM tests remain negative: `0.863x` stock throughput at 8K and
`0.927x` at 16K. They execute 36 real ticketed FlashInfer layers, 35 first-wave
lookaheads, zero stock launch/fallback, and produce identical output. The trend
does not establish a crossover. The current ABI-v23 v10 2K mixed point is also
negative: exact output, all 1,080 attention launches transformed, zero fallback,
`0.977x` throughput, `1.012x` resident P99 inter-token latency, and `1.021x`
external TTFT. A five-trial admission/re-merge policy regressed the causal tail
metric and was removed. No end-to-end SGLang speedup, production-ready status,
or OSDI-level claim is supported yet. The immediate gate is an end-to-end
model-generated selected-demand workload and a heterogeneous SLO result that
beats equal-state bulk and skip/rebatch baselines with zero stock fallback.
The corrected real FlashInfer device-selected sweep chooses transformed bulk
at zero avoided bytes, where forced indexed acquisition delivers only `0.6431x`
throughput. At 75%-93.75% avoided bytes it reaches `2.1259x`-`8.1731x` over
forced candidate overfetch with same-trial policy regret `1.0000x`. An
all-candidate-resident arm is at least `9.15x` faster than cold indexed
acquisition, so this is useful-byte reduction evidence and an explicit
acquisition-overhead target, not a universal-win claim.
The current warm-cache resident Qwen2.5-3B graph smoke executes 4,068
compiler-transformed attention launches, 36 captures, and 32 replays with zero
stock/fallback launches and measures `1.021x` stock throughput. Three samples
establish graph integration and the resident-overhead gate only; no incremental
external work executes in that row.
There is no placeholder RDMA backend. The only NVMe control plane uses
VFIO PCI cdev ownership and a private IOMMUFD IOAS; the existing device ABI and
finite GPU data path are unchanged. Hardware write protection remains the
default media policy. Dedicated test deployments may explicitly select
`trusted-read-only-code`; this trusts the fixed READ-only device transport but
retains VFIO/IOMMUFD DMA containment. Construction executes an end-to-end GPU
doorbell qualification and requires a valid NVMe completion. The previous
scheduling prototype remains on `main` at commit `4789f16`.

## Build and test

The CUDA 13 header layout is not accepted by current Clang CUDA wrappers. Use a
matched distribution LLVM/Clang 22 and CUDA 12.9:

```bash
cmake -S . -B build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=/usr/lib/llvm-22/lib/cmake/llvm \
  -DNTA_CLANG_CUDA=/usr/bin/clang++-22 \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.9 \
  -DNTA_CUDA_ROOT=/usr/local/cuda-12.9 \
  -DNTA_CUDA_ARCH=sm_120
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Install the native development/runtime artifacts and the dependency-free Python
binding separately:

```bash
cmake --install build --prefix /opt/nta
python3 -m pip install .
export LD_LIBRARY_PATH=/opt/nta/lib:${LD_LIBRARY_PATH:-}
```

For an uninstalled tree, set `NTA_RUNTIME_LIBRARY` to
`build/libnta-runtime.so` and add `python/` to `PYTHONPATH`.

Compile a source-generated CUDA kernel through Clang and the NTA pass without a
separate `opt` step. A source build uses the repository tool; a CMake install
provides the self-contained `nta-jit-activate` command:

```bash
tools/jit/activate.py --build-dir build --flashinfer-hook -- \
  python3 your_flashinfer_or_kernel_generator.py

/opt/nta/bin/nta-jit-activate --flashinfer-hook -- \
  python3 your_flashinfer_or_kernel_generator.py
```

This makes the compiler delivery transparent, not the kernel semantics. The
kernel source must expose one CTA-uniform, pre-state acquisition hook; arbitrary
precompiled cubins cannot be transformed safely. See
[docs/INTEGRATION.md](docs/INTEGRATION.md).

Run the real mixed-placement workload:

```bash
./build/nta-kv-bench \
  --mode=mixed \
  --requests=96 \
  --coalesce=3 \
  --dependencies=4 \
  --tile-bytes=65536 \
  --iterations=50 \
  --cancel-stride=17 \
  --stale-stride=19
```

Supported modes are `resident`, `host-direct`, `host-staged`, and `mixed`.
Generated compiler artifacts are under `build/kernel/`.
Use `--baseline=1` with a direct placement to run the identical multi-object
numerical kernel without request/acquisition logic for resident-path overhead
measurement.

Exercise production-style non-owning CPU-DRAM registration, including an
intentionally unaligned staged source:

```bash
./build/nta-kv-bench \
  --mode=host-staged --requests=96 --tile-bytes=65536 --iterations=50 \
  --external-registration=1 --external-offset=1
```

`HostMapped` lets the GPU consume mapped pinned DRAM directly. `HostStaged`
uses finite GPU progress to copy into caller-provided HBM and publish data
availability; it has vectorized aligned, row-specialized multi-CTA, and safe
unaligned paths. RNIC/RDMA is
deferred and remains inactive rather than being simulated.

Run the engine-neutral MoE generality workload:

```bash
./build/nta-moe-bench \
  --mode=host-staged --policy=late-bound \
  --tokens=8 --experts=512 --top-k=2 --hidden=256 --iterations=50
```

Use `--policy=cpu-sync` and `--policy=overfetch` for sparse-demand baselines.
`--policy=direct` with resident or mapped-host placement is the no-acquisition
control. The randomized specification in `experiments/moe-late-bound.json`
runs all of these comparisons.

Run the split-K attention workload using hardware TMA after external staging:

```bash
./build/nta-paged-attention \
  --mode=mixed \
  --copy=tma \
  --requests=32 \
  --min-pages=4 \
  --max-pages=16 \
  --iterations=50 \
  --progress-passes=1
```

`scripts/validate-local.sh` reproduces the build, tests, global-load/TMA
placement matrix, and PTX resource report. Set `NTA_SANITIZE=1` to include
memcheck, racecheck, and synccheck.
`scripts/measure-direct-overhead.sh` runs alternating process-level trials and
reports paired 95% t intervals.

Run the real FlashInfer GPU-selected page crossover, including forced
overfetch, a precomputed selected-copy oracle, and the online cost decision:

```bash
tools/jit/activate.py --build-dir build --flashinfer-hook -- \
  python3 scripts/run-selected-pages-sweep.py \
  --output results/selected-pages-sweep-v25-corrected.json \
  --require-peak-speedup
```

Run the release qualifier before making a release-readiness claim:

```bash
./scripts/qualify-release.py --profile=local
./scripts/qualify-release.py --profile=production --skip-local
./scripts/qualify-release.py --profile=osdi --skip-local
```

The stronger profiles fail closed until their serving, reliability,
portability, baseline, ablation, and statistical evidence files satisfy the
documented schema. A passing local profile is not a production or paper claim.

Check the matched stock SGLang/FlashInfer serving environment with a local
model:

```bash
./benchmarks/serving/SglangSmoke.py \
  --model /path/to/local/model --requests 4 --max-new-tokens 8
```

The emitted report is marked `nta_integrated=false`; this is a real baseline
smoke test, not NTA serving evidence.

Run a matched, residency-qualified HiCache comparison through the installed
SGLang plugin:

```bash
python benchmarks/serving/CompareSglangHiCache.py \
  --model /path/to/local/model \
  --iterations 10 \
  --hot-tokens 160 \
  --resident-tokens 96 \
  --churn-tokens 320 \
  --max-total-tokens 384 \
  --context-length 512 \
  --cuda-graph-decode full
```

The harness times only attempts whose hot request reports host-cached tokens,
requires the same residency sequence and generated output from stock and NTA,
and fails if it cannot collect the requested number of promotions. The
optimized local Llama-160M runs have exact output and no measured median
regression, but their scheduler-sensitive latency distribution is not
production or OSDI performance evidence; see `docs/SGLANG.md`.

When FlashInfer 0.6.12 is installed, CTest compiles NTA-parameterized decode and
paged-prefill modules, loads the exported phase ABI, and executes resident,
pinned-host, shared-head, split-K decode, and multi-tile prefill gates. The
overlay rejects source hashes or anchors that differ from the validated wheel.
CTest also enables `nta-flashinfer-differential-gpu`. Override header discovery
with `-DNTA_FLASHINFER_INCLUDE_DIR=/path/to/include`.

## GPU-initiated NVMe

The NVMe bootstrap is `vfio:DDDD:BB:SS.F`. A CPU userspace control
plane owns the dedicated PCI function through VFIO/IOMMUFD, initializes the
controller, creates the GPU queue, and maps only its DMA regions and one
doorbell page. The default policy also protects every active namespace; the
explicit trusted-code policy supports controllers without NVMe write-protect
feature `0x84`. Direct HBM import is disabled until the platform supplies a
validated P2P route.
After initialization, the CPU does not construct commands, ring doorbells, or
consume completions. A missing finite application CTA may submit one NVMe read
and then exits; it never polls the CQ. Contended work enters the bounded
request-aware intent scheduler, and the separate finite progress kernel owns
all completion processing.

Build and qualify only an unmounted, otherwise unused test controller:

```bash
cmake --build build --target nta-vfio-nvme-probe nta-nvme-bench
./scripts/nta-vfio-device.sh preflight
NTA_NVME_MEDIA_POLICY=trusted-read-only-code \
  ./scripts/nta-vfio-device.sh bind-and-probe
./scripts/nta-vfio-device.sh restore
./scripts/run-nvme-qualification.py \
  --media-policy=trusted-read-only-code \
  --bytes=$((2 * 1024 * 1024)) \
  --requests=32 \
  --progress-passes=32 \
  --iterations=20 \
  --require-ready
```

The helper refuses mounted/open devices, block holders, shared IOMMU groups,
unsafe no-IOMMU mode, non-4-KiB pages, and active `vmem_sw` references.
`iommu=pt` before binding is not itself a failure: attaching `vfio-pci`
to a private IOAS establishes the translated domain. The raw queue remains a
trusted-process interface, not a multi-tenant security boundary. SPDK is a
matched CPU baseline, not a linked queue owner; its public qpair API does not
transfer raw queue bookkeeping to GPU code.

On the current KIOXIA CD8P, the default policy refuses ownership because the
controller reports `NWPC=0`. The explicit trusted-code policy passes the
end-to-end GPU SQ-doorbell-CQ qualification. A historical ABI-v18 compulsory-miss run
of 32 independent 64-KiB reads over 1,000 graph iterations submitted and
completed all 32,000 measured commands with matching data at 1,624.42 MiB/s
physical throughput and zero failure. The benchmark invalidates its staging entries at
the start of every measured graph; cache-hit replay is therefore not counted as
SSD bandwidth. This is a single-machine mechanism result, not production
serving or paper-level evaluation. The ABI-v25 harness now completes exact
2-MiB reads at 58.16% of a matched `fio` baseline on this host from clean
revision `5c26f8b8aa6c`. This is one-controller local scaling evidence, not
serving or portability evidence.
