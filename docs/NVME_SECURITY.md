# NVMe Security And Lifecycle Contract

Status: trusted single-host mechanism, not a multi-tenant device API

The GPU submits NVMe commands and rings an MMIO doorbell without a CPU proxy.
That removes the point at which a kernel driver could validate each command.
The interface is consequently restricted to `CAP_SYS_RAWIO` callers and is not
a security boundary between mutually untrusted processes, containers, tenants,
or GPUs.

## Enforced Probe Preconditions

The driver refuses to bind unless all of these conditions hold:

- host and controller pages are 4 KiB, so the mapping beginning at BAR offset
  `0x1000` cannot include controller registers before or after the doorbell
  page;
- the controller is in a translated DMA or DMA-FQ IOMMU domain;
- the controller is the only device in its IOMMU group;
- the configured namespace exists, uses no metadata or protection information,
  and supports the PRP transfer contract; and
- the controller supports basic namespace write protection, and a read-back of
  feature `0x84` confirms that every active namespace entered that state.

The IOMMU prevents arbitrary physical-memory DMA. Protecting every active
namespace prevents an I/O submission queue command from selecting another NSID
and modifying media. Neither property
turns a raw queue into an untrusted API: a malicious privileged caller can still
target an IOVA currently mapped in this device's domain, corrupt its own queue,
or ring another queue's doorbell in the same 4 KiB page.

Use VFIO/IOMMUFD plus a dedicated IOMMU address space when untrusted userspace
device ownership is required. Even then, opcode/media safety needs hardware
write protection or a validating command mediator; IOMMU isolation alone does
not validate NVMe commands.

## Queue And Mapping Ownership

Each successful `open()` allocates one SQ/CQ pair, PRP-list arena, command
context array, mapping table, queue ID, and generation. Up to
`max_io_queues` mutually trusted GPU runtimes can coexist. Queue and doorbell
VMAs hold references after file close, so coherent memory and PCI resources are
not freed while a CUDA or inherited mapping remains live.

The first queue page contains state and generation. GPU progress checks it
before consuming or publishing queue work and immediately fails outstanding
continuations if the driver reports fatal, removed, or stale state. The page is
an availability signal, not a security primitive; bus-master disable and IOMMU
translation provide containment after a fatal event.

The runtime records one CUDA device ordinal for every transport, host runtime,
and reusable work plan. CUDA operations switch to that owner and restore the
caller's previous device. Separate GPUs obtain separate NVMe queues by opening
the device independently. This is multi-GPU ownership support, not multi-tenant
isolation, and real multi-GPU doorbell routing still requires platform
qualification.

## DMA-BUF And Host Memory

All mappings use `DMA_BIDIRECTIONAL`; the raw queue cannot determine a command's
direction after the GPU writes it.

DMA-BUF import uses a static attachment. Mapping pins the backing storage and
waits the exporter's reservation fences. The importer adds a write-usage fence
for the full queue-visible lifetime, deletes the SQ before signaling that
fence, then unmaps and detaches. There is no dynamic `move_notify` path and no
stale scatter-gather mapping to rebuild.

`allow_peer2peer` is not enabled. Direct NVMe DMA into CUDA HBM is therefore
disabled by the runtime until a platform-specific, validated P2P path exists.
The supported contained destination is long-term pinned, mapped CPU DRAM. This
avoids treating a successful DMA-BUF export as proof of a safe GPU/NVMe PCIe
route. See the Linux [DMA-BUF](https://docs.kernel.org/driver-api/dma-buf.html)
and [P2PDMA](https://docs.kernel.org/driver-api/pci/p2pdma.html) contracts.

Mapping release quiesces the complete queue. This is intentional: without a
CPU-visible command ledger, the driver cannot prove that one arbitrary mapping
is absent from already-published SQ entries.

## Failure And Device Lifecycle

An admin timeout or CID mismatch poisons every open queue, advances generation,
and disables PCI bus mastering. Recovery is attempted only after all queue and
VMA references close; it performs a PCI function reset, controller
reinitialization, namespace re-identification, write-protect verification, and
queue-count negotiation.

PCI AER/reset callbacks poison active queues and recover only with no live
owners. Suspend is rejected while queues or VMAs exist. Shutdown and removal
mark queue controls before teardown; removal waits for all mapped references.
The namespace is re-identified and write protection is revalidated before the
first queue in each open epoch.

## Deliberate Limits

- GPU code emits only READ SQEs, but the kernel cannot enforce that opcode per
  command without moving submission through a CPU or hardware mediator.
- Queue depth is 64. The controller negotiates one queue per open, up to 32.
- Transfers use PRPs. One page of PRP entries per CID and MDTS bound every
  transfer; NVMe SGLs are not implemented.
- Metadata and protection information are rejected rather than silently
  mishandled.
- One configured namespace is exposed for reads per module instance; it is no
  longer hardcoded to namespace 1. Every active namespace is write-protected
  because the raw SQE still carries a caller-controlled NSID.
- The exact 4 KiB doorbell page necessarily contains all queue doorbells.
- Asynchronous namespace events are not consumed while queues are live. The
  driver assumes exclusive ownership of a dedicated controller and validates
  the namespace at epoch boundaries.
- The implementation does not inherit the upstream NVMe driver's complete
  controller quirk, power-management, and hotplug coverage.

These limits are paper and production gates, not hidden implementation claims.
On the current host, the target SSD's IOMMU group type is `identity`, so the
hardened preflight correctly refuses to bind it. Earlier HBM and mapped-DRAM
measurements predate this contract and are retained only as historical
mechanism evidence.
