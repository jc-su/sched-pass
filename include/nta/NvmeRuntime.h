#pragma once

#include "nta/RuntimeABI.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace nta {

enum class NvmeMediaPolicy {
  RequireHardwareWriteProtection,
  TrustReadOnlyDeviceCode,
};

// The instrumented GPU owns the NVMe queue in either mode. This selects where
// the controller's data DMA lands; host-mapped memory is an explicit baseline,
// while HbmPeer is the production direct-data-plane path.
enum class NvmeDmaTarget {
  HbmPeer,
  HostMapped,
};

// Concrete mapping backend used to make an HBM allocation addressable by the
// VFIO-owned NVMe function. This is a setup-plane property, not a data path.
enum class NvmeHbmMappingBackend : std::uint32_t {
  Unavailable = 0,
  NvidiaPeerPages = 1,
};

struct NvmeTransportOptions {
  // The transport exclusively owns this PCI function through VFIO/IOMMUFD.
  // No implicit controller is selected because binding a device is destructive.
  std::string endpoint;
  int deviceOrdinal = -1;
  std::uint32_t namespaceId = 1;
  std::uint32_t queueDepth = 64;
  std::uint32_t adminTimeoutMs = 10'000;
  NvmeMediaPolicy mediaPolicy = NvmeMediaPolicy::RequireHardwareWriteProtection;
  NvmeDmaTarget dmaTarget = NvmeDmaTarget::HbmPeer;
};

struct NvmeCapabilities {
  std::uint32_t queueDepth;
  std::uint32_t controllerPageSize;
  std::uint32_t lbaSize;
  std::uint32_t maxTransferBytes;
  std::uint64_t namespaceBytes;
  std::uint32_t queueId;
  std::uint32_t queueCount;
  int deviceOrdinal;
  bool supportsHbmPeerDma;
  NvmeHbmMappingBackend hbmMappingBackend;
  bool translatedIommu;
  bool namespaceReadOnly;
  bool gpuDoorbellMappingValidated;
};

struct NvmeQueueStats {
  std::uint64_t submitted;
  std::uint64_t completed;
  std::uint64_t failed;
  std::uint64_t directSubmitted;
  std::uint64_t directFallbacks;
  std::uint32_t outstanding;
  std::uint32_t error;
  std::uint32_t sqTail;
  std::uint32_t cqHead;
  std::uint32_t cqPhase;
  std::uint32_t nextCompletionDword3;
  std::uint64_t hbmRegionRegistrations;
  std::uint64_t hbmRegionBytes;
  std::uint64_t hbmTransferViews;
};

// Native description of the setup-time registration envelope for a
// caller-owned CUDA slice.  Framework allocators may place several logical
// tensors in one CUDA allocation and in the same 64 KiB peer page.  Callers
// use this description to coalesce overlapping envelopes before pinning, so
// every peer PTE has exactly one mapping owner.
struct NvmeHbmRegistrationRange {
  void *allocationAddress;
  std::size_t allocationBytes;
  void *registrationAddress;
  std::size_t registrationBytes;
};

class NvmeBuffer {
public:
  ~NvmeBuffer();

  NvmeBuffer(const NvmeBuffer &) = delete;
  NvmeBuffer &operator=(const NvmeBuffer &) = delete;
  NvmeBuffer(NvmeBuffer &&) noexcept;
  NvmeBuffer &operator=(NvmeBuffer &&) noexcept;

  [[nodiscard]] void *deviceAddress() const noexcept;
  [[nodiscard]] std::uint64_t dmaPageListAddress() const noexcept;
  [[nodiscard]] std::uint32_t dmaPageCount() const noexcept;
  [[nodiscard]] std::uint32_t dmaFirstByteOffset() const noexcept;
  [[nodiscard]] std::size_t bytes() const noexcept;
  [[nodiscard]] NvmeDmaTarget dmaTarget() const noexcept;
  // False for a mapping lease over caller-owned HBM (for example a vLLM KV
  // block).  The lease still owns the NVMe peer mapping and page list, but it
  // never frees the caller's CUDA allocation.
  [[nodiscard]] bool ownsDestinationMemory() const noexcept;

private:
  struct Impl;
  explicit NvmeBuffer(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;

  friend class NvmeTransport;
  friend class NvmeHbmRegion;
};

// Setup-time registration of a stable, caller-owned CUDA allocation.  The
// region owns one peer-page/IOMMU mapping and one immutable device page table.
// Transfer views borrow slices of those resources; creating a view performs no
// ioctl, peer pin, CUDA allocation, or page-table upload.
class NvmeHbmRegion {
public:
  ~NvmeHbmRegion();

  NvmeHbmRegion(const NvmeHbmRegion &) = delete;
  NvmeHbmRegion &operator=(const NvmeHbmRegion &) = delete;
  NvmeHbmRegion(NvmeHbmRegion &&) noexcept;
  NvmeHbmRegion &operator=(NvmeHbmRegion &&) noexcept;

  [[nodiscard]] void *deviceAddress() const noexcept;
  [[nodiscard]] std::size_t bytes() const noexcept;
  [[nodiscard]] std::unique_ptr<NvmeBuffer> view(void *deviceAddress,
                                                 std::size_t bytes) const;

private:
  struct Impl;
  explicit NvmeHbmRegion(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;

  friend class NvmeTransport;
  friend class NvmeBuffer;
};

class NvmeTransport {
public:
  explicit NvmeTransport(std::string vfioEndpoint, int deviceOrdinal = -1);
  explicit NvmeTransport(NvmeTransportOptions options);
  ~NvmeTransport();

  NvmeTransport(const NvmeTransport &) = delete;
  NvmeTransport &operator=(const NvmeTransport &) = delete;
  NvmeTransport(NvmeTransport &&) noexcept;
  NvmeTransport &operator=(NvmeTransport &&) noexcept;

  [[nodiscard]] const NvmeCapabilities &capabilities() const noexcept;
  [[nodiscard]] int deviceOrdinal() const noexcept;
  [[nodiscard]] abi::NvmeQueueView *deviceQueue() const noexcept;
  [[nodiscard]] NvmeQueueStats readStats() const;
  [[nodiscard]] std::unique_ptr<NvmeBuffer> allocate(std::size_t bytes);
  // Validate one caller-owned CUDA slice and describe the minimal peer-page
  // envelope that can be registered.  This is a read-only setup-plane query;
  // it does not pin memory or mutate the IOMMU domain.
  [[nodiscard]] NvmeHbmRegistrationRange
  describeExternalHbm(void *deviceAddress, std::size_t bytes) const;
  // Register a stable caller-owned CUDA range once, before serving. Individual
  // MDTS-bounded transfer views are then derived without setup-plane work.
  [[nodiscard]] std::unique_ptr<NvmeHbmRegion>
  registerExternalHbm(void *deviceAddress, std::size_t bytes);

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;

  friend class NvmeBuffer;
  friend class NvmeHbmRegion;
};

} // namespace nta
