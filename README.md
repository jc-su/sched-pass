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
- an engine-neutral `WorkPlan` model and ABI-v20 bounded dependency sets, so one
  work ticket can wait for several pages, experts, or object shards;
- one reusable device-plan allocation with two pinned, asynchronous upload
  slots and no unconditional hot-path event synchronization;
- a versioned shared-library C API and owning Python binding for engine
  allocations, work plans, finite JIT phases, work ticket state, and VFIO NVMe;
- zero-copy DLPack views of the native runtime and plan plus a bounded
  FlashInfer layer executor whose fixed enqueue path can be captured after its
  structural plan is uploaded;
- full SGLang decode CUDA-graph replay through stock FlashInfer wrappers after
  stream-ordered acquisition, with exact live request metadata preserved across
  the engine's padded replay view;
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
- per-request blocked-byte, runnable-compute, completed-compute, and terminal
  work summaries plus request-local reduction counters;
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
- request-bound runnable-work launches that map a compact physical CTA prefix
  back to the original FlashInfer request/tile schedule while the launch is
  stream-ordered;
- an installed SGLang 0.5.14 plugin backend that binds real request IDs,
  generations, and priorities, overlaps HiCache's tuned layer copies with model
  execution, retains stock FlashInfer after acquisition, routes only unresolved
  multi-round work through request-guarded instrumented FlashInfer, mirrors
  request aborts, keys plan reuse by exact host/device page pairs, and fails
  closed on claimed-batch planning errors unless availability-only fallback is
  explicitly enabled.
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
acquisition
markers; automatic recognition of arbitrary production load/cp.async/TMA
address cones is an open gate. The pass does prove CTA-uniform control and
operands for explicit sites; lane-divergent collective calls are rejected.
The real FlashInfer GPU-selected path is an operator-level result, not a claim
that the current SGLang path uses sparse attention or improves an end-to-end
serving SLO. Dense SGLang remains the production integration target. Current
one-GPU dense CPU-DRAM opportunity traces do not justify forced incremental
execution, so the online path dispatches the stock bulk form there. Custom CUDA
attention and MoE programs remain correctness fixtures rather than production
performance evidence.
The existing code implements the unavailable-data work-ticket mechanism,
request-local partial reduction, real FlashInfer hooks, and SGLang decode graph
replay. It does not yet generate direct and incremental forms from one typed
operator, feed progress into SGLang batch admission, or put the demand-mode
progress loop inside SGLang graph replay as described in `SYSTEM_PLAN.md`.
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
uses a finite GPU progress CTA to copy into caller-provided HBM and publish data
availability; it has vectorized aligned and safe unaligned paths. RNIC/RDMA is
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
  --output results/selected-pages-sweep-v20.json \
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
sudo env LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64 \
  ./build/nta-nvme-bench \
  --device=vfio:0000:d8:00.0 \
  --gpu=0 \
  --reference=/tmp/nta-nvme-reference.bin \
  --requests=16 \
  --bytes=65536 \
  --cta-try-issue=1 \
  --progress-passes=8 \
  --iterations=1000 \
  --media-policy=trusted-read-only-code \
  --output=results/hardware/nvme-abi19.json
./scripts/nta-vfio-device.sh restore
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
serving or paper-level evaluation, and it must be rerun on ABI v20 before use as
current evidence.
