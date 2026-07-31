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

## Implemented vertical slice

The branch contains a working LLVM 22 new-PM pass and a real Blackwell CUDA
workload:

- per-CTA request/generation binding in batched kernels;
- compiler proof of a canonical finite-kernel deferral boundary;
- direct HBM and mapped-CPU-DRAM paths with no queue or atomic operation;
- staged CPU-DRAM acquisition issued and completed by finite GPU CTAs;
- direct NVMe reads into either DMA-BUF-registered HBM or registered mapped
  DRAM, with GPU-built SQEs/PRPs, GPU MMIO doorbells, and GPU CQ handling;
- a fixed-capacity intent ring, duplicate-object coalescing, cancellation, and
  generation-safe continuations;
- one captured CUDA graph containing discover, bounded progress, and resume
  kernels; and
- an inspectable NVVM IR -> pass -> PTX -> cubin build pipeline.

There is no placeholder RDMA backend. NVMe is implemented against an isolated
KIOXIA CD8P controller and tested with read-only commands. The previous
scheduling prototype remains on `main` at commit `4789f16`.

## Build and test

The CUDA 13 header layout is not accepted by current Clang CUDA wrappers. Use a
matched distribution LLVM/Clang 22 and CUDA 12.9:

```bash
cmake -S . -B build -GNinja \
  -DLLVM_DIR=/usr/lib/llvm-22/lib/cmake/llvm \
  -DNTA_CLANG_CUDA=/usr/bin/clang++-22 \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.9 \
  -DNTA_CUDA_ROOT=/usr/local/cuda-12.9 \
  -DNTA_CUDA_ARCH=sm_120
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run the real mixed-placement workload:

```bash
./build/nta-kv-bench \
  --mode=mixed \
  --requests=96 \
  --coalesce=3 \
  --tile-bytes=65536 \
  --iterations=50 \
  --cancel-stride=17 \
  --stale-stride=19
```

Supported modes are `resident`, `host-direct`, `host-staged`, and `mixed`.
Generated compiler artifacts are under `build/kernel/`.

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
