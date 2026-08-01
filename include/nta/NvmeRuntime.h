#pragma once

#include "nta/RuntimeABI.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace nta {

enum class NvmeDestination {
  Hbm,
  HostMapped,
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
};

struct NvmeQueueStats {
  std::uint64_t submitted;
  std::uint64_t completed;
  std::uint64_t failed;
  std::uint32_t outstanding;
  std::uint32_t error;
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
  [[nodiscard]] NvmeDestination destination() const noexcept;

private:
  struct Impl;
  explicit NvmeBuffer(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;

  friend class NvmeTransport;
};

class NvmeTransport {
public:
  explicit NvmeTransport(std::string devicePath = "/dev/nta_nvme",
                         int deviceOrdinal = -1);
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
  allocate(std::size_t bytes, NvmeDestination destination);

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;

  friend class NvmeBuffer;
};

} // namespace nta
