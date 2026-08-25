#pragma once

#include "nta/NvmeRuntime.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
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

struct NvmeMappingToken {
  enum class Kind : std::uint32_t {
    None = 0,
    HostIoas = 1,
    NvidiaPeerPages = 2,
  };

  Kind kind = Kind::None;
  std::uint64_t value = 0;

  [[nodiscard]] explicit operator bool() const noexcept {
    return kind != Kind::None && value != 0;
  }
};

class NvmeMappingBackend;

// A typed, owning mapping lease.  The page vector is the immutable data-plane
// description consumed by the GPU queue; the backend token is released by the
// lease destructor.  Callers cannot accidentally release a peer mapping through
// the host IOAS path, and no steady-state I/O operation needs this object.
class NvmeMapping {
public:
  NvmeMapping() = default;
  ~NvmeMapping();

  NvmeMapping(const NvmeMapping &) = delete;
  NvmeMapping &operator=(const NvmeMapping &) = delete;
  NvmeMapping(NvmeMapping &&) noexcept;
  NvmeMapping &operator=(NvmeMapping &&) noexcept;

  [[nodiscard]] explicit operator bool() const noexcept {
    return token_.operator bool();
  }
  [[nodiscard]] const std::vector<std::uint64_t> &pages() const noexcept {
    return pages_;
  }
  // A peer mapper may have to map an alignment-rounded allocation.  The NVMe
  // command still owns an exact byte range, so retain only the page prefix
  // covered by that command while keeping the backend lease for the full map.
  void retainPagePrefix(std::size_t count);

private:
  NvmeMapping(NvmeMappingBackend *backend, NvmeMappingToken token,
              std::vector<std::uint64_t> pages)
      : backend_(backend), token_(token), pages_(std::move(pages)) {}

  void reset() noexcept;

  NvmeMappingBackend *backend_ = nullptr;
  NvmeMappingToken token_{};
  std::vector<std::uint64_t> pages_;

  friend class NvmeMappingBackend;
};

class NvmeMappingBackend {
public:
  virtual ~NvmeMappingBackend() = default;

  NvmeMappingBackend(const NvmeMappingBackend &) = delete;
  NvmeMappingBackend &operator=(const NvmeMappingBackend &) = delete;

  [[nodiscard]] virtual NvmeMapping mapHost(void *address,
                                            std::size_t bytes) = 0;
  // Pin a CUDA device allocation through the selected peer-memory API and
  // return DMA addresses valid for this VFIO-owned NVMe function.
  [[nodiscard]] virtual NvmeMapping mapHbm(std::uint64_t gpuAddress,
                                           std::size_t bytes) = 0;
  // Stop accepting new mappings and release all mappings owned by the
  // backend.  The control plane calls this only after its queue is quiesced.
  virtual void shutdown() noexcept = 0;

protected:
  NvmeMappingBackend() = default;
  [[nodiscard]] NvmeMapping makeMapping(NvmeMappingToken token,
                                        std::vector<std::uint64_t> pages) {
    return NvmeMapping(this, token, std::move(pages));
  }
  virtual void release(NvmeMappingToken token) noexcept = 0;

  friend class NvmeMapping;
};

class NvmeControlPlane {
public:
  virtual ~NvmeControlPlane() = default;

  NvmeControlPlane(const NvmeControlPlane &) = delete;
  NvmeControlPlane &operator=(const NvmeControlPlane &) = delete;

  [[nodiscard]] virtual const NvmeQueueResources &
  resources() const noexcept = 0;
  [[nodiscard]] virtual NvmeMappingBackend &mappingBackend() noexcept = 0;
  virtual void quiesce() noexcept = 0;

protected:
  NvmeControlPlane() = default;
};

[[nodiscard]] std::unique_ptr<NvmeControlPlane>
createVfioNvmeControlPlane(const NvmeTransportOptions &options);

} // namespace nta::detail
