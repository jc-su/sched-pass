#pragma once

#include "nta/RuntimeABI.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace nta {

class NvmeBuffer;
class NvmeTransport;

enum class Placement {
  Hbm,
  HostMapped,
  HostStaged,
};

struct RuntimeConfig {
  std::uint32_t requestCapacity;
  std::uint32_t objectCapacity;
  std::uint32_t intentCapacity;
  std::uint32_t continuationCapacity;
};

struct ObjectHandle {
  std::uint32_t slot;
  void *directDeviceBase;
};

class HostRuntime {
public:
  explicit HostRuntime(RuntimeConfig config);
  HostRuntime(RuntimeConfig config, std::shared_ptr<NvmeTransport> nvme);
  ~HostRuntime();

  HostRuntime(const HostRuntime &) = delete;
  HostRuntime &operator=(const HostRuntime &) = delete;
  HostRuntime(HostRuntime &&) noexcept;
  HostRuntime &operator=(HostRuntime &&) noexcept;

  void setRequest(std::uint32_t slot, std::uint64_t requestId,
                  std::uint32_t generation, std::uint32_t tenantId = 0,
                  std::uint32_t priority = 0, std::uint64_t deadlineClock = 0);
  void cancelRequest(std::uint32_t slot, std::uint32_t generation);

  ObjectHandle installObject(std::uint32_t slot, std::uint64_t objectId,
                             std::uint32_t version,
                             std::span<const std::byte> contents,
                             Placement placement);
  ObjectHandle installNvmeObject(std::uint32_t slot, std::uint64_t objectId,
                                 std::uint32_t version,
                                 std::uint64_t sourceByteOffset,
                                 std::size_t bytes,
                                 std::unique_ptr<NvmeBuffer> destination);

  [[nodiscard]] abi::RuntimeView *deviceView() const noexcept;
  [[nodiscard]] const RuntimeConfig &config() const noexcept;
  [[nodiscard]] abi::RequestContext readRequest(std::uint32_t slot) const;
  [[nodiscard]] abi::ObjectEntry readObject(std::uint32_t slot) const;
  [[nodiscard]] abi::Continuation readContinuation(std::uint32_t slot) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace nta
