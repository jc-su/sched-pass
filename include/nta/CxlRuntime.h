#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace nta {

struct CxlDaxOptions {
  // A devdax endpoint is intentionally explicit. The runtime never scans or
  // binds a CXL device on behalf of a serving process.
  std::string endpoint;
  std::size_t windowBytes = 0;
  int deviceOrdinal = -1;
};

struct CxlDaxCapabilities {
  std::size_t windowBytes = 0;
  void *mappedDeviceAddress = nullptr;
  int deviceOrdinal = -1;
  bool hostRegistered = false;
  bool directDeviceVisible = false;
};

class CxlDaxBuffer {
public:
  ~CxlDaxBuffer();

  CxlDaxBuffer(const CxlDaxBuffer &) = delete;
  CxlDaxBuffer &operator=(const CxlDaxBuffer &) = delete;
  CxlDaxBuffer(CxlDaxBuffer &&) noexcept;
  CxlDaxBuffer &operator=(CxlDaxBuffer &&) noexcept;

  [[nodiscard]] void *hostAddress() const noexcept;
  [[nodiscard]] void *deviceAddress() const noexcept;
  [[nodiscard]] std::size_t bytes() const noexcept;

private:
  struct Impl;
  CxlDaxBuffer(std::shared_ptr<Impl> impl, std::size_t bytes);
  std::shared_ptr<Impl> impl_;
  std::size_t bytes_ = 0;

  friend class CxlDaxTransport;
};

// CUDA-visible CXL memory is a mapped host replica, not an NVMe-like queue.
// The transport owns one explicitly supplied devdax mapping and hands out
// bounded slices. Allocation is monotonic for the lifetime of the transport;
// this is deliberate because recycling a DAX slice while a captured graph can
// still reference it is unsafe without an engine completion fence.
class CxlDaxTransport {
public:
  explicit CxlDaxTransport(CxlDaxOptions options);
  ~CxlDaxTransport();

  CxlDaxTransport(const CxlDaxTransport &) = delete;
  CxlDaxTransport &operator=(const CxlDaxTransport &) = delete;
  CxlDaxTransport(CxlDaxTransport &&) noexcept;
  CxlDaxTransport &operator=(CxlDaxTransport &&) noexcept;

  [[nodiscard]] CxlDaxCapabilities capabilities() const noexcept;
  [[nodiscard]] int deviceOrdinal() const noexcept;
  [[nodiscard]] void *deviceAddress() const noexcept;
  [[nodiscard]] bool containsDeviceAddress(const void *address,
                                           std::size_t bytes) const noexcept;
  [[nodiscard]] std::unique_ptr<CxlDaxBuffer>
  allocate(std::size_t bytes, std::size_t alignment = 4096);

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

} // namespace nta
