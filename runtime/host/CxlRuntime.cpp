#include "nta/CxlRuntime.h"

#include "CudaDeviceGuard.h"

#include <cuda_runtime_api.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace nta {
namespace {

void checkCuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

std::size_t roundUp(std::size_t value, std::size_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1U)) != 0) {
    throw std::invalid_argument(
        "CXL allocation alignment must be a power of two");
  }
  if (value > std::numeric_limits<std::size_t>::max() - (alignment - 1U)) {
    throw std::overflow_error("CXL allocation offset overflows size_t");
  }
  return (value + alignment - 1U) & ~(alignment - 1U);
}

} // namespace

struct CxlDaxBuffer::Impl {
  std::shared_ptr<void> owner;
  void *hostAddress = nullptr;
  void *deviceAddress = nullptr;
};

struct CxlDaxTransport::Impl {
  explicit Impl(CxlDaxOptions options) {
    if (options.endpoint.empty()) {
      throw std::invalid_argument("CXL DAX endpoint must be explicit");
    }
    const long pageSize = ::sysconf(_SC_PAGESIZE);
    if (pageSize <= 0 || options.windowBytes == 0 ||
        options.windowBytes % static_cast<std::size_t>(pageSize) != 0) {
      throw std::invalid_argument(
          "CXL DAX windowBytes must be a non-zero page multiple");
    }
    deviceOrdinal = detail::resolveCudaDevice(options.deviceOrdinal);
    detail::CudaDeviceGuard deviceGuard(deviceOrdinal);
    fd = ::open(options.endpoint.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
      throw std::runtime_error("cannot open CXL DAX endpoint " +
                               options.endpoint);
    }
    windowBytes = options.windowBytes;
    struct stat endpointStat{};
    if (::fstat(fd, &endpointStat) != 0) {
      closeFd();
      throw std::runtime_error("cannot stat CXL DAX endpoint");
    }
    if (S_ISREG(endpointStat.st_mode)) {
      closeFd();
      throw std::invalid_argument(
          "CXL DAX endpoint must be a live devdax character device; "
          "regular files are not qualification targets");
    }
    if (!S_ISCHR(endpointStat.st_mode)) {
      closeFd();
      throw std::invalid_argument(
          "CXL DAX endpoint is not a devdax character device");
    }
    if (static_cast<std::uintmax_t>(endpointStat.st_size) < windowBytes &&
        endpointStat.st_size != 0) {
      closeFd();
      throw std::invalid_argument(
          "CXL DAX endpoint is smaller than the requested window");
    }
    mappedHostAddress =
        ::mmap(nullptr, windowBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mappedHostAddress == MAP_FAILED) {
      mappedHostAddress = nullptr;
      closeFd();
      throw std::runtime_error("cannot mmap CXL DAX window");
    }
    try {
      const cudaError_t flagStatus = cudaSetDeviceFlags(cudaDeviceMapHost);
      if (flagStatus != cudaSuccess &&
          flagStatus != cudaErrorSetOnActiveProcess) {
        checkCuda(flagStatus, "cudaSetDeviceFlags(cudaDeviceMapHost)");
      }
      checkCuda(
          cudaHostRegister(mappedHostAddress, windowBytes,
                           cudaHostRegisterMapped | cudaHostRegisterPortable),
          "cudaHostRegister CXL DAX window");
      hostRegistered = true;
      checkCuda(
          cudaHostGetDevicePointer(&mappedDeviceAddress, mappedHostAddress, 0),
          "cudaHostGetDevicePointer CXL DAX window");
      directDeviceVisible = mappedDeviceAddress != nullptr;
      if (!directDeviceVisible) {
        throw std::runtime_error("CXL DAX mapping has no CUDA device address");
      }
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~Impl() { cleanup(); }

  void closeFd() noexcept {
    if (fd >= 0) {
      (void)::close(fd);
      fd = -1;
    }
  }

  void cleanup() noexcept {
    detail::NoexceptCudaDeviceGuard deviceGuard(deviceOrdinal);
    if (hostRegistered && mappedHostAddress != nullptr) {
      (void)cudaHostUnregister(mappedHostAddress);
      hostRegistered = false;
    }
    if (mappedHostAddress != nullptr) {
      (void)::munmap(mappedHostAddress, windowBytes);
      mappedHostAddress = nullptr;
    }
    mappedDeviceAddress = nullptr;
    closeFd();
  }

  int fd = -1;
  int deviceOrdinal = -1;
  std::size_t windowBytes = 0;
  std::size_t nextOffset = 0;
  std::mutex allocationMutex;
  void *mappedHostAddress = nullptr;
  void *mappedDeviceAddress = nullptr;
  bool hostRegistered = false;
  bool directDeviceVisible = false;
};

CxlDaxBuffer::CxlDaxBuffer(std::shared_ptr<Impl> impl, std::size_t bytes)
    : impl_(std::move(impl)), bytes_(bytes) {}

CxlDaxBuffer::~CxlDaxBuffer() = default;
CxlDaxBuffer::CxlDaxBuffer(CxlDaxBuffer &&) noexcept = default;
CxlDaxBuffer &CxlDaxBuffer::operator=(CxlDaxBuffer &&) noexcept = default;

void *CxlDaxBuffer::hostAddress() const noexcept {
  return impl_ == nullptr ? nullptr : impl_->hostAddress;
}

void *CxlDaxBuffer::deviceAddress() const noexcept {
  return impl_ == nullptr ? nullptr : impl_->deviceAddress;
}

std::size_t CxlDaxBuffer::bytes() const noexcept { return bytes_; }

CxlDaxTransport::CxlDaxTransport(CxlDaxOptions options)
    : impl_(std::make_shared<Impl>(std::move(options))) {}
CxlDaxTransport::~CxlDaxTransport() = default;
CxlDaxTransport::CxlDaxTransport(CxlDaxTransport &&) noexcept = default;
CxlDaxTransport &
CxlDaxTransport::operator=(CxlDaxTransport &&) noexcept = default;

CxlDaxCapabilities CxlDaxTransport::capabilities() const noexcept {
  if (impl_ == nullptr) {
    return {};
  }
  return {impl_->windowBytes, impl_->mappedDeviceAddress, impl_->deviceOrdinal,
          impl_->hostRegistered, impl_->directDeviceVisible};
}

int CxlDaxTransport::deviceOrdinal() const noexcept {
  return impl_ == nullptr ? -1 : impl_->deviceOrdinal;
}

void *CxlDaxTransport::deviceAddress() const noexcept {
  return impl_ == nullptr ? nullptr : impl_->mappedDeviceAddress;
}

bool CxlDaxTransport::containsDeviceAddress(const void *address,
                                            std::size_t bytes) const noexcept {
  if (impl_ == nullptr || address == nullptr || bytes == 0 ||
      impl_->mappedDeviceAddress == nullptr) {
    return false;
  }
  const std::uintptr_t base =
      reinterpret_cast<std::uintptr_t>(impl_->mappedDeviceAddress);
  const std::uintptr_t value = reinterpret_cast<std::uintptr_t>(address);
  if (value < base) {
    return false;
  }
  const std::size_t offset = static_cast<std::size_t>(value - base);
  return offset <= impl_->windowBytes && bytes <= impl_->windowBytes - offset;
}

std::unique_ptr<CxlDaxBuffer> CxlDaxTransport::allocate(std::size_t bytes,
                                                        std::size_t alignment) {
  if (impl_ == nullptr || bytes == 0) {
    throw std::invalid_argument(
        "CXL allocation needs a live non-empty transport");
  }
  std::lock_guard lock(impl_->allocationMutex);
  const std::size_t offset = roundUp(impl_->nextOffset, alignment);
  if (offset > impl_->windowBytes || bytes > impl_->windowBytes - offset) {
    throw std::runtime_error("CXL DAX window capacity exhausted");
  }
  impl_->nextOffset = offset + bytes;
  auto bufferImpl = std::make_shared<CxlDaxBuffer::Impl>();
  bufferImpl->owner = impl_;
  bufferImpl->hostAddress =
      static_cast<std::byte *>(impl_->mappedHostAddress) + offset;
  bufferImpl->deviceAddress =
      static_cast<std::byte *>(impl_->mappedDeviceAddress) + offset;
  return std::unique_ptr<CxlDaxBuffer>(
      new CxlDaxBuffer(std::move(bufferImpl), bytes));
}

} // namespace nta
