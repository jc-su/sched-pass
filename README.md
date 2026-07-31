# Nonresident Acquisition

This branch is a clean-slate implementation of request-semantic external data
acquisition for finite GPU kernels.

The primary application is SLO-critical external KV access in continuously
batched attention. The mechanism also targets GPU-routed MoE experts and, as a
secondary generality case, graph or ANNS objects.

The architecture contract and implementation sequence are defined in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
The exact tested environment and current measurements are in
[docs/VALIDATION.md](docs/VALIDATION.md).
The current closest-work and novelty boundary is in
[docs/RELATED_WORK.md](docs/RELATED_WORK.md).
The implemented FlashInfer boundary and remaining serving integration are in
[docs/FLASHINFER.md](docs/FLASHINFER.md).
The compact host/kernel API, JIT path, and exact transparency contract are in
[docs/INTEGRATION.md](docs/INTEGRATION.md).

## Implemented vertical slice

The branch contains a working LLVM 22 new-PM pass and real Blackwell CUDA
workloads:

- per-CTA request/generation binding in batched kernels;
- compiler proof of a canonical finite-kernel deferral boundary;
- an engine-neutral `WorkPlan` model and ABI-v9 bounded dependency sets, so one
  continuation can wait for several pages, experts, or object shards;
- one reusable device-plan allocation with pinned, asynchronous batch updates;
- non-owning registration of existing engine HBM and device-visible host
  allocations;
- public finite-kernel host and device policies, replacing benchmark-specific
  phase launch and marker plumbing;
- direct HBM and mapped-CPU-DRAM paths with no queue or atomic operation;
- staged CPU-DRAM acquisition issued and completed by finite GPU CTAs;
- direct NVMe reads into either DMA-BUF-registered HBM or registered mapped
  DRAM, with GPU-built SQEs/PRPs, GPU MMIO doorbells, and GPU CQ handling;
- a reusable, object-keyed intent pool, duplicate-object coalescing,
  cancellation, and generation-safe ready-only continuations;
- fixed request, tenant, and backend byte credits plus priority/deadline NVMe
  admission;
- a backend-neutral directory with bounded per-object physical replicas;
- a numerically checked split-K paged-attention workload with heterogeneous
  request lengths that consumes the common work/dependency ABI for both global
  loads and TMA;
- a numerically checked top-k MoE workload using the same plan, compiler pass,
  tier directory, progress path, and ready scheduling;
- an optional adapter from FlashInfer's public paged-KV CSR tables into the
  common work model, FlashInfer-native `(V, LSE)` cascade state, and a
  differential GPU correctness gate against FlashInfer 0.6.12;
- real TMA consumption from direct sources or from HBM after external staging,
  with compiler-proven deferral before barrier creation;
- one captured CUDA graph containing discover, bounded progress, and resume
  kernels;
- an inspectable NVVM IR -> pass -> PTX -> cubin build pipeline;
- automatic optimizer-last lowering in Clang JIT builds, with an nvcc-compatible
  compiler shim and an ABI-fingerprinted FlashInfer cache.

The core ABI and compiler pass do not depend on FlashInfer, vLLM, SGLang, or a
kernel name. A supported kernel must still expose a reconstructible pre-state
work boundary; arbitrary instruction-level suspension is not implemented. The
FlashInfer compatibility layer and JIT transport do not yet place deferral
inside FlashInfer's optimized FMHA CTA, and no vLLM/SGLang request-lifecycle
adapter is present. The compiler currently consumes explicit acquisition
markers; automatic recognition of arbitrary production load/cp.async/TMA
address cones is an open gate. The pass does prove CTA-uniform control and
operands for explicit sites; lane-divergent collective calls are rejected.
There is no placeholder RDMA backend. NVMe is implemented against an isolated
KIOXIA CD8P controller and tested with read-only commands. The previous
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

Compile a source-generated CUDA kernel through Clang and the NTA pass without a
separate `opt` step:

```bash
tools/jit/activate.py --build-dir build -- \
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

Run the engine-neutral MoE generality workload:

```bash
./build/nta-moe-bench \
  --mode=mixed --tokens=64 --experts=16 --top-k=2 --hidden=128 \
  --iterations=50
```

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

When `flashinfer-python` is installed, CMake locates its headers and builds the
attention reduction with `flashinfer::state_t`; CTest also enables the real
multi-source JIT compile and `nta-flashinfer-differential-gpu` gates. Override
header discovery with
`-DNTA_FLASHINFER_INCLUDE_DIR=/path/to/include`.

## GPU-initiated NVMe

The NVMe path has a small, trusted bootstrap driver under
`driver/nta_nvme/`. It exclusively owns one explicitly selected controller,
creates one queue pair, imports CUDA DMA-BUFs, and maps queue/doorbell pages.
After initialization, the CPU does not construct commands, ring doorbells, or
consume completions.

Build and bind only an unmounted, otherwise unused test controller:

```bash
cmake --build build --target nta-nvme-driver
./scripts/nta-nvme-device.sh bind
sudo env LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64 \
  ./build/nta-nvme-bench \
  --destination=hbm \
  --reference=/tmp/nta-nvme-reference.bin \
  --requests=16 \
  --bytes=65536 \
  --progress-passes=128 \
  --iterations=10
./scripts/nta-nvme-device.sh restore
```

The bind helper refuses mounted/open devices, block holders, and this host's
`vmem_sw` backing-store module. NVIDIA also requires privilege for mapping a
third-party PCIe doorbell with `CU_MEMHOSTREGISTER_IOMEMORY`, so the hardware
benchmark runs as root. The driver exposes a raw queue to that trusted process;
it is not a multi-tenant security boundary.
