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
  enum class Kind : std::uint32_t {
    None = 0,
    HostIoas = 1,
    NvidiaPeerPages = 2,
  };

  struct Handle {
    Kind kind = Kind::None;
    std::uint64_t value = 0;

    [[nodiscard]] explicit operator bool() const noexcept {
      return kind != Kind::None && value != 0;
    }
  } handle;
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
  // Pin a CUDA device allocation through NVIDIA's persistent peer-memory API
  // and return DMA addresses valid for this VFIO-owned NVMe function.
  [[nodiscard]] virtual NvmeDmaMapping mapHbm(std::uint64_t gpuAddress,
                                              std::size_t bytes) = 0;
  virtual void unmap(NvmeDmaMapping::Handle handle) noexcept = 0;
  virtual void quiesce() noexcept = 0;

protected:
  NvmeControlPlane() = default;
};

[[nodiscard]] std::unique_ptr<NvmeControlPlane>
createVfioNvmeControlPlane(const NvmeTransportOptions &options);

} // namespace nta::detail
