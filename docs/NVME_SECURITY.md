# NVMe Security And Lifecycle Contract

Status: dedicated-controller, trusted-process mechanism; not a multi-tenant API

The only backend is VFIO PCI cdev ownership attached to a private IOMMUFD
IO address space. The GPU writes NVMe SQ entries and doorbells without a CPU
proxy, so no software component can validate a command after publication. The
design always requires DMA containment and makes media containment a deployment
policy. `RequireHardwareWriteProtection` is the default and fails closed when
NVMe feature `0x84` is absent. `TrustReadOnlyDeviceCode` explicitly accepts that
the compiler-generated device path is the media-safety boundary. It does not add
a CPU broker or per-command proxy.

## Enforced VFIO Preconditions

`vfio:DDDD:BB:SS.F` construction requires all of the following:

- the PCI function is bound to `vfio-pci` and exposes a VFIO cdev;
- `/dev/iommu` accepts a new IOAS and the device attaches to it;
- the function is alone in its IOMMU group and unsafe no-IOMMU mode is off;
- function reset and readable, writable, mmap-capable BAR0/config regions work;
- host and controller pages are 4 KiB;
- controller registers and the doorbell page are separate one-page VMAs;
- CAP advertises the NVM command set and the requested queue depth;
- the namespace has no metadata or protection information and fits the PRP
  contract; and
- either feature `0x84` read-back confirms basic write protection for every
  active namespace, or the caller explicitly selects trusted READ-only device
  code before an I/O SQ is exposed.

IOMMUFD maps admin queues, I/O queues, PRP arenas, and destination buffers with
read and write permissions because a raw SQ does not reveal DMA direction to
the host runtime. IOVAs exist only in the transport's private IOAS. This blocks
DMA into unrelated host physical memory; it does not make arbitrary commands
inside that IOAS safe.

Hardware protection prevents media writes even if a malformed SQE selects a
different NSID. Under the trusted-code policy, the generated transport hardcodes
opcode `0x02` and the configured NSID; corruption or arbitrary device code can
still issue destructive commands. Both policies assume a trusted serving
process with exclusive controller ownership. Neither protects queue state or
controller availability from that process.

## NVIDIA BAR Compatibility Gate

A CPU BAR mapping is not evidence that an NVIDIA GPU can use the same VMA. The
runtime registers exactly the one-page doorbell VMA using
`CU_MEMHOSTREGISTER_IOMEMORY`, obtains its GPU address, JITs a one-thread probe,
places a valid READ in the I/O SQ, and requires a GPU MMIO SQ-doorbell store to
produce a phase-correct, successful NVMe CQE. Transport construction fails on a
timeout, CID mismatch, or NVMe status error. Kernel completion by itself is not
accepted as evidence that the PCIe transaction reached the controller.

This gate tests the exact VFIO BAR VMA, CUDA device ordinal, NVIDIA driver, and
GPU combination used by the workload. It must be rerun for each qualified
platform and driver release.

The control plane also performs one CPU-issued READ before publishing the queue
to validate queue creation and IOMMU DMA independently. Both reads are bounded
bootstrap qualification operations into private scratch pages. They are not
application I/O and occur before the finite workload data path starts.

## Queue And Mapping Ownership

The CPU bootstrap exclusively owns controller reset, admin SQ/CQ operation,
Identify, optional namespace protection, queue negotiation, queue creation, and
the two qualification reads above. Once the I/O queue is published, application
CTAs may submit one bounded READ and finite progress CTAs submit fallback work
and consume completions. CPU code does not submit application I/O or poll
application completions.

Queue memory is published through the engine-neutral `NvmeQueueView` device
ABI. The compiler and device transport depend on that bounded queue view, not
on VFIO administration details.

Releasing any destination first synchronizes the owning CUDA device, marks the
GPU queue inactive, deletes the I/O SQ/CQ, and only then unmaps its IOVA. If
queue deletion or an admin command fails, bus mastering is cleared and VFIO
function reset is used before mappings are released. This whole-queue quiesce
is intentional: there is no trusted CPU command ledger that can prove a
mapping is absent from already-published GPU SQ entries.

## SPDK Boundary

SPDK is a useful correctness and performance baseline for userspace NVMe
initialization. It is not linked into this backend. SPDK's public NVMe qpair API
owns queue trackers, submission, completion, and doorbell bookkeeping on the
CPU; it does not provide a stable transfer of raw SQ/CQ/doorbell ownership to a
GPU. Integrating below that API would depend on private SPDK internals and leave
two queue owners.

The implementation instead contains the narrow administration path required
for exclusive GPU queue ownership: reset/enable, Identify, write protection,
number-of-queues negotiation, and create/delete SQ/CQ. SPDK should be used as a
matched CPU baseline and its traces should be compared against this bootstrap.

## Deliberate Limits

- Device code emits fixed READ SQEs. The default policy also requires namespace
  hardware write protection; trusted-code mode deliberately relies on the
  generated device path and is not a media-containment boundary.
- VFIO currently exposes one depth-configurable I/O queue per transport and one
  configured namespace for workload reads.
- Transfers use PRPs, with one PRP-list page per CID; NVMe SGLs are not used.
- Direct NVMe DMA into HBM remains disabled until GPU/NVMe P2P topology and
  NVIDIA memory registration are separately validated.
- The VFIO transport owns the complete PCI function. It does not share a
  controller with the kernel NVMe driver or another process.
- One transport is bound to one CUDA device ordinal. Multi-GPU qualification
  requires a tested GPU/BAR route for every GPU. Scale-out uses one independently
  owned controller function per GPU; shared-function brokerage is intentionally
  outside this design. Success on one GPU is not inherited by another.
- AER, surprise removal, suspend, and power-loss testing remain hardware gates.
  VFIO reset contains teardown failures, but does not reproduce the upstream
  NVMe driver's complete quirk and platform lifecycle coverage.

Use `scripts/nta-vfio-device.sh preflight`, then on an otherwise unused test
controller run `bind-and-probe`. The safe policy remains the default. On the
KIOXIA CD8P, which reports `NWPC=0`, use
`NTA_NVME_MEDIA_POLICY=trusted-read-only-code` only when the dedicated-device
threat model is acceptable. On 2026-08-01 that policy passed CPU queue/DMA
qualification, GPU SQ-doorbell-CQ qualification, and verified application READ
workloads; teardown restored `nvmex` after testing.
