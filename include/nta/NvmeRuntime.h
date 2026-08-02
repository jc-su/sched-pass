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

struct NvmeTransportOptions {
  // The transport exclusively owns this PCI function through VFIO/IOMMUFD.
  // No implicit controller is selected because binding a device is destructive.
  std::string endpoint;
  int deviceOrdinal = -1;
  std::uint32_t namespaceId = 1;
  std::uint32_t queueDepth = 64;
  std::uint32_t adminTimeoutMs = 10'000;
  NvmeMediaPolicy mediaPolicy = NvmeMediaPolicy::RequireHardwareWriteProtection;
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
  bool supportsHbmPeer;
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
  [[nodiscard]] std::size_t bytes() const noexcept;

private:
  struct Impl;
  explicit NvmeBuffer(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;

  friend class NvmeTransport;
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
  [[nodiscard]] std::unique_ptr<NvmeBuffer>
  allocate(std::size_t bytes);

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;

  friend class NvmeBuffer;
};

} // namespace nta
