#pragma once

#include "nta/NvmeRuntime.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace nta::detail {

struct NvmeQueueResources {
  NvmeCapabilities capabilities{};
  void *queueHost = nullptr;
  std::size_t queueBytes = 0;
  std::size_t controlOffset = 0;
  std::size_t sqOffset = 0;
  std::size_t cqOffset = 0;
  std::size_t prpOffset = 0;
  std::uint64_t prpDmaAddress = 0;
  void *doorbellHost = nullptr;
  std::size_t doorbellBytes = 0;
  std::size_t sqDoorbellOffset = 0;
  std::size_t cqDoorbellOffset = 0;
  std::uint32_t generation = 0;
  bool queueIsIoMemory = false;
};

struct NvmeDmaMapping {
  std::uint64_t handle = 0;
  std::vector<std::uint64_t> pages;
};

class NvmeControlPlane {
public:
  virtual ~NvmeControlPlane() = default;

  NvmeControlPlane(const NvmeControlPlane &) = delete;
  NvmeControlPlane &operator=(const NvmeControlPlane &) = delete;

  [[nodiscard]] virtual const NvmeQueueResources &
  resources() const noexcept = 0;
  [[nodiscard]] virtual NvmeDmaMapping mapHost(void *address,
                                               std::size_t bytes) = 0;
  virtual void unmapHost(std::uint64_t handle) noexcept = 0;
  virtual void quiesce() noexcept = 0;

protected:
  NvmeControlPlane() = default;
};

[[nodiscard]] std::unique_ptr<NvmeControlPlane>
createVfioNvmeControlPlane(const NvmeTransportOptions &options);

} // namespace nta::detail
