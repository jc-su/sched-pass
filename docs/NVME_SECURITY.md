# NVMe-to-HBM security and lifecycle contract

Status: qualified on one dedicated controller/GPU platform; trusted-process
research mechanism, not a multi-tenant storage API.

## What the path is

The production data path is:

```text
instrumented GPU consumer
  -> exact work-unit demand and generation claim
  -> GPU-owned NVMe SQ/CQ and doorbell
  -> VFIO-owned NVMe controller
  -> PCIe DMA through a translated IOMMUFD domain
  -> CUDA HBM pinned with NVIDIA persistent peer pages
```

The CPU initializes and tears down the controller. It does not select
application reads, submit them, poll their completions, or copy their payloads.
There is no host-memory data proxy. This is also not GPUDirect Storage: NTA owns
the raw NVMe queue and the compiler/runtime contract that connects device-side
demand to queue submission and consumption.

`NvmeDmaTarget::HostMapped` is an explicit matched baseline. It is never an
implicit fallback for `NvmeDmaTarget::HbmPeer`; direct-HBM construction fails
closed if any required capability is absent.

## Why this is not the DMA-BUF/PAL design

Linux 7.0 `IOMMU_IOAS_MAP_FILE` does not generically import an NVIDIA CUDA
DMA-BUF into an attached VFIO device domain. The proposed DMA-BUF Physical
Address List mapping type is not a usable upstream userspace contract on this
host, and the NVIDIA exporter does not supply that mapping type. The earlier
DMA-BUF/VMM prototype could create an export object but the attached-domain map
failed with `EINVAL`; that is a rejected alternative, not the current path.

The narrow `nta_nvme_p2p` bridge instead uses NVIDIA's persistent peer-memory
API:

1. `nvidia_p2p_get_pages_persistent` pins a 64-KiB-aligned CUDA allocation.
2. `nvidia_p2p_dma_map_pages` returns peer bus addresses for the selected NVMe
   PCI function.
3. The bridge installs identity IOVA-to-peer-address PTEs in that function's
   current translated IOMMUFD paging domain.
4. A typed, per-open-file handle owns the PTEs, NVIDIA DMA mapping, page table,
   and PCI-device reference until explicit unmap or file close.

The explicit PTE step is required. NVIDIA's DMA API does not populate the
separately attached unmanaged IOMMUFD domain; omitting it produced first-level
DMAR faults and no HBM payload even though peer-page pinning itself succeeded.

The bridge accepts only a real NVMe PCI class bound to `vfio-pci`, requires an
IOMMUFD paging-domain cookie, rejects colliding/misaligned/out-of-aperture
IOVAs, rolls back partial maps, never reuses a wrapped handle, and unmaps IOMMU
PTEs before releasing NVIDIA peer pages. Its UAPI is deliberately typed to one
GPU range, one NVMe BDF, and one owned mapping handle; it is not a general
physical-address export service.

## Enforced containment

Transport construction and the transactional qualification scripts require:

- a VFIO PCI cdev attached to a new private IOMMUFD IO address space;
- exactly one PCI function in the target IOMMU group;
- translated IOMMU operation with unsafe no-IOMMU mode disabled;
- NVMe class, NVM command-set support, 4-KiB host/controller pages, supported
  queue depth, and a namespace without metadata or protection information;
- reset plus readable, writable, mmap-capable config/BAR regions; and
- either NVMe Namespace Write Protection feature `0x84`, or an explicit
  `TrustReadOnlyDeviceCode` decision for a dedicated disposable experiment
  controller whose device program emits only opcode `0x02` reads.

The private IOAS contains queue memory, PRP lists, bootstrap scratch pages, and
the explicitly mapped HBM ranges. It prevents the controller from DMA-writing
unrelated system memory. It does not make an arbitrary SQ command safe: after
the GPU publishes an SQE and rings the doorbell, no CPU broker validates that
command. Hardware namespace write protection is therefore the stronger media
boundary. The trusted-code mode assumes the serving process, compiler output,
and loaded cubin are trusted.

The qualification script additionally rejects mounted namespaces, partitions,
block holders, open block-device users, hidden multipath namespaces, malformed
BDFs, and a controller that is not restored to a live kernel namespace.

## Control plane, data plane, and ordering

The CPU control plane exclusively owns reset/enable, Identify, optional write
protection, queue-count negotiation, admin commands, and I/O queue
create/delete. It performs a bounded CPU-issued READ before publication to
validate queue creation and ordinary IOMMU mappings.

The GPU data plane receives only the engine-neutral `NvmeQueueView`. Before the
queue becomes usable, the runtime registers the isolated doorbell page with
CUDA, launches a GPU MMIO probe, and requires a phase-correct successful CQE.
A CPU-visible BAR mapping alone is not accepted as GPU doorbell evidence.

Application execution then performs exact READs into HBM. NVMe progress is
completion-driven inside one graph-capturable GPU launch: a warp drains pending
acquisition intents and controller completions until both counters are idle or
a bounded device timeout expires. The graph-visible `progressRounds` value is
the number of dependency/consumer rounds, not a fixed polling budget. The
one-pass primitive remains only for low-level state-machine tests.

Qualification requires native owner-scope-or-stronger GPUDirect RDMA write
ordering because the consumer reads HBM after GPU-side CQ consumption without
an intervening CPU CUDA flush.

## Teardown

Destination release synchronizes its CUDA device, marks the queue inactive,
deletes the I/O SQ/CQ, and only then releases peer mappings. A peer mapping
release removes its PTEs from the still-attached IOMMUFD domain before NVIDIA
DMA unmap and persistent page release. The mapper file also releases every
remaining handle on close.

If queue deletion or an admin operation fails, the runtime disables bus
mastering and resets the VFIO function before releasing mappings. Whole-queue
quiescence is intentional: there is no trusted host command ledger that could
prove that an HBM range is absent from all already-published GPU SQEs.

`scripts/nta-vfio-device.sh qualify` records the original PCI driver and uses
an exit trap to restore it. Restoration waits for both the driver and a live
namespace, clears `driver_override`, and removes transactional state. A block
device name is not assumed stable across reprobe; ownership follows PCI BDF and
namespace ID.

## Qualification contract

Build/load the kernel-specific bridge, run read-only preflight, and only then
authorize a transactional bind:

```bash
./scripts/nta-nvme-p2p-module.sh load

NTA_NVME_BDF=0000:d8:00.0 \
NTA_NVME_MEDIA_POLICY=trusted-read-only-code \
./scripts/nta-vfio-device.sh preflight

python3 scripts/run-nvme-qualification.py \
  --bdf 0000:d8:00.0 \
  --dma-target hbm-peer \
  --media-policy trusted-read-only-code \
  --bytes 2097152 --requests 32 --progress-rounds 1 --iterations 20 \
  --fio-runtime 10 --minimum-bandwidth-ratio 0.9 \
  --allow-device-rebind --require-ready \
  --output /path/outside/checkout/nvme-qualification.json
```

The runner first measures a read-only `fio` baseline while the kernel owns the
namespace. It captures a reference file by reading the namespace, binds only
the selected controller to VFIO, runs exact GPU-controlled reads, checksum
verifies every destination, counts target-specific DMAR fault messages before
and after the run, and restores the original driver in all normal/error paths.

`ready` requires all of the following: exact selected data verified, zero
verification/transport failures, zero outstanding commands, GPU doorbell
qualification, translated IOMMU, `nvidia-peer-pages` HBM mapping, and no new
target DMAR faults. `performance_qualified` is a separate comparison against
the configured matched-`fio` threshold. A correct but slow transport is never
reported as a performance win.

The rebind flag authorizes temporary driver ownership only. It does not
authorize media formatting, filesystem creation, writes, discard, sanitize,
or deletion. The generated application path issues READ commands only, and
`fio` is invoked with `--readonly --rw=read`.

## Current qualified host (2026-08-25)

The qualified platform runs Linux `7.0.0-30-generic`, NVIDIA driver `595.84`,
CUDA `13.2`, an RTX PRO 6000 Blackwell GPU, and controller `0000:d8:00.0`
(Dell/KIOXIA CD8P, 1.92 TB). The controller lacks Namespace Write Protection
(`NWPC=0`), so this run used the explicit trusted READ-only-code policy.

For 32 concurrent 2-MiB reads, 20 measured graph replays, and one
completion-driven dependency round, the result was:

- 640/640 measured commands completed, with zero failure or outstanding count;
- exact destination checksums and zero verification failures;
- 6,356.3 MiB/s GPU-controlled end-to-end throughput versus 6,155.0 MiB/s
  matched `fio` throughput (`1.0327x`); and
- target DMAR fault-line count unchanged at 9 before and after qualification.

The nine pre-existing fault lines came from rejected development runs before
the explicit IOMMUFD peer PTE fix. Qualification requires the count to remain
unchanged; it does not erase kernel history. The controller was restored to the
kernel `nvme` driver after the run and its namespace returned live.

This is one-platform transport/correctness/performance evidence. It does not
by itself establish serving-level speedup, topology portability, multi-GPU
support, or general NVMe-driver robustness.

## Deliberate limits

- One transport owns one complete NVMe PCI function, one namespace, one CUDA
  device ordinal, and currently one configurable I/O queue.
- Transfers use PRPs with one PRP-list page per CID; NVMe SGLs are not used.
- The out-of-tree bridge is tied to the running kernel and NVIDIA peer-memory
  ABI and must be rebuilt for each kernel/driver combination.
- The bridge accesses the current IOMMUFD domain through kernel interfaces; it
  must unload only after all users and mappings are gone.
- Multi-GPU qualification requires an independent doorbell, peer-page, IOMMU,
  ordering, checksum, and fault-free result for every GPU/controller route.
- AER, surprise removal, suspend, power loss, and broad controller-quirk
  coverage remain future hardware gates. VFIO reset is containment, not a
  replacement for the upstream NVMe driver's lifecycle coverage.
- SPDK remains a useful CPU baseline, but its public qpair API retains CPU
  queue ownership. NTA therefore does not link SPDK into the GPU-owned path.
