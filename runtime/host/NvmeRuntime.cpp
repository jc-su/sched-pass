#include "nta/NvmeRuntime.h"

#include "CudaDeviceGuard.h"
#include "nta/NvmeUapi.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <sys/ioctl.h>
#include <sys/mman.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

namespace nta {
namespace {

void checkCuda(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

void checkDriver(CUresult result, const char *operation) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char *name = nullptr;
  const char *description = nullptr;
  (void)cuGetErrorName(result, &name);
  (void)cuGetErrorString(result, &description);
  throw std::runtime_error(
      std::string(operation) + ": " +
      (name == nullptr ? "unknown CUDA driver error" : name) + " (" +
      (description == nullptr ? "no description" : description) + ")");
}

[[noreturn]] void throwSystem(const char *operation) {
  throw std::system_error(errno, std::generic_category(), operation);
}

std::size_t roundUp(std::size_t value, std::size_t alignment) {
  if (value > std::numeric_limits<std::size_t>::max() - (alignment - 1U)) {
    throw std::overflow_error("NVMe allocation size overflows size_t");
  }
  return (value + alignment - 1U) & ~(alignment - 1U);
}

} // namespace

struct NvmeTransport::Impl {
  Impl(const std::string &path, int requestedDevice)
      : deviceOrdinal(detail::resolveCudaDevice(requestedDevice)) {
    detail::CudaDeviceGuard deviceGuard(deviceOrdinal);
    fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
      throwSystem("open NVMe GPU queue device");
    }
    try {
      if (::ioctl(fd, NTA_NVME_IOCTL_GET_INFO, &info) != 0) {
        throwSystem("NTA_NVME_IOCTL_GET_INFO");
      }
      constexpr std::uint32_t requiredCapabilities =
          NTA_NVME_CAP_IOMMU_TRANSLATED | NTA_NVME_CAP_NAMESPACE_READ_ONLY |
          NTA_NVME_CAP_STATIC_DMA_BUF | NTA_NVME_CAP_MULTI_QUEUE |
          NTA_NVME_CAP_TRUSTED_RAW_QUEUE;
      if (info.abi_version != NTA_NVME_ABI_VERSION || info.queue_depth < 2 ||
          info.controller_page_size == 0 ||
          (info.controller_page_size & (info.controller_page_size - 1U)) != 0 ||
          info.queue_bytes == 0 || info.doorbell_mmap_bytes == 0 ||
          info.queue_id == 0 || info.queue_id > info.queue_count ||
          (info.capabilities & requiredCapabilities) != requiredCapabilities) {
        throw std::runtime_error("NVMe driver returned an incompatible ABI");
      }
      capabilities = {
          info.queue_depth,
          info.controller_page_size,
          1U << info.lba_shift,
          info.max_transfer_bytes,
          info.namespace_blocks << info.lba_shift,
          info.queue_id,
          info.queue_count,
          deviceOrdinal,
          false,
          true,
          true,
      };

      const long systemPage = ::sysconf(_SC_PAGESIZE);
      if (systemPage <= 0 ||
          static_cast<std::uint32_t>(systemPage) != info.controller_page_size) {
        throw std::runtime_error(
            "NVMe controller and host page sizes must match");
      }
      queueHost = ::mmap(
          nullptr, info.queue_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
          static_cast<off_t>(NTA_NVME_MMAP_QUEUE_PGOFF * systemPage));
      if (queueHost == MAP_FAILED) {
        queueHost = nullptr;
        throwSystem("mmap NVMe queue memory");
      }
      doorbellHost = ::mmap(
          nullptr, info.doorbell_mmap_bytes, PROT_READ | PROT_WRITE, MAP_SHARED,
          fd, static_cast<off_t>(NTA_NVME_MMAP_DOORBELL_PGOFF * systemPage));
      if (doorbellHost == MAP_FAILED) {
        doorbellHost = nullptr;
        throwSystem("mmap NVMe doorbells");
      }

      checkDriver(cuMemHostRegister(queueHost, info.queue_bytes,
                                    CU_MEMHOSTREGISTER_DEVICEMAP |
                                        CU_MEMHOSTREGISTER_IOMEMORY),
                  "cuMemHostRegister NVMe queue memory");
      queueRegistered = true;
      checkDriver(cuMemHostGetDevicePointer(&queueDevice, queueHost, 0),
                  "cuMemHostGetDevicePointer NVMe queue memory");
      auto *controlHost = reinterpret_cast<nta_nvme_queue_control *>(
          static_cast<std::byte *>(queueHost) + info.control_offset);
      if (controlHost->magic != NTA_NVME_QUEUE_CONTROL_MAGIC ||
          controlHost->abi_version != NTA_NVME_ABI_VERSION ||
          controlHost->state != NTA_NVME_QUEUE_ONLINE ||
          controlHost->generation != info.generation ||
          controlHost->queue_id != info.queue_id) {
        throw std::runtime_error("NVMe queue control page is inconsistent");
      }
      checkDriver(cuMemHostRegister(doorbellHost, info.doorbell_mmap_bytes,
                                    CU_MEMHOSTREGISTER_DEVICEMAP |
                                        CU_MEMHOSTREGISTER_IOMEMORY),
                  "cuMemHostRegister NVMe doorbells");
      doorbellRegistered = true;
      checkDriver(cuMemHostGetDevicePointer(&doorbellDevice, doorbellHost, 0),
                  "cuMemHostGetDevicePointer NVMe doorbells");

      checkCuda(cudaMalloc(reinterpret_cast<void **>(&contexts),
                           sizeof(abi::NvmeCommandContext) * info.queue_depth),
                "cudaMalloc NVMe command contexts");
      checkCuda(cudaMemset(contexts, 0,
                           sizeof(abi::NvmeCommandContext) * info.queue_depth),
                "cudaMemset NVMe command contexts");

      abi::NvmeQueueView hostQueue{
          reinterpret_cast<abi::NvmeSubmission *>(queueDevice + info.sq_offset),
          reinterpret_cast<abi::NvmeCompletion *>(queueDevice + info.cq_offset),
          reinterpret_cast<std::uint64_t *>(queueDevice + info.prp_offset),
          info.prp_dma_address,
          reinterpret_cast<volatile std::uint32_t *>(doorbellDevice +
                                                     info.sq_doorbell_offset),
          reinterpret_cast<volatile std::uint32_t *>(doorbellDevice +
                                                     info.cq_doorbell_offset),
          contexts,
          reinterpret_cast<abi::NvmeQueueControl *>(queueDevice +
                                                    info.control_offset),
          info.queue_depth,
          info.controller_page_size,
          info.lba_shift,
          info.namespace_id,
          0,
          0,
          1,
          0,
          0,
          1,
          0,
          0,
          info.generation,
          info.queue_id,
          0,
          0,
          0,
          0,
      };
      checkCuda(cudaMalloc(reinterpret_cast<void **>(&deviceQueue),
                           sizeof(hostQueue)),
                "cudaMalloc NvmeQueueView");
      checkCuda(cudaMemcpy(deviceQueue, &hostQueue, sizeof(hostQueue),
                           cudaMemcpyHostToDevice),
                "upload NvmeQueueView");
    } catch (...) {
      release();
      throw;
    }
  }

  ~Impl() { release(); }

  void release() noexcept {
    detail::NoexceptCudaDeviceGuard deviceGuard(deviceOrdinal);
    (void)cudaDeviceSynchronize();
    if (deviceQueue != nullptr) {
      (void)cudaFree(deviceQueue);
      deviceQueue = nullptr;
    }
    if (contexts != nullptr) {
      (void)cudaFree(contexts);
      contexts = nullptr;
    }
    if (doorbellRegistered) {
      (void)cuMemHostUnregister(doorbellHost);
      doorbellRegistered = false;
    }
    if (queueRegistered) {
      (void)cuMemHostUnregister(queueHost);
      queueRegistered = false;
    }
    if (doorbellHost != nullptr) {
      (void)::munmap(doorbellHost, info.doorbell_mmap_bytes);
      doorbellHost = nullptr;
    }
    if (queueHost != nullptr) {
      (void)::munmap(queueHost, info.queue_bytes);
      queueHost = nullptr;
    }
    if (fd >= 0) {
      (void)::close(fd);
      fd = -1;
    }
  }

  void releaseMapping(std::uint64_t handle) noexcept {
    if (fd < 0 || handle == 0) {
      return;
    }
    std::scoped_lock lock(ioctlMutex);
    nta_nvme_release request{handle};
    (void)::ioctl(fd, NTA_NVME_IOCTL_RELEASE_DMA_BUF, &request);
  }

  void prepareMappingRelease() noexcept {
    detail::NoexceptCudaDeviceGuard deviceGuard(deviceOrdinal);
    if (fd < 0 || deviceQueue == nullptr || quiesced) {
      return;
    }
    (void)cudaDeviceSynchronize();
    std::scoped_lock lock(ioctlMutex);
    (void)::ioctl(fd, NTA_NVME_IOCTL_QUIESCE);
    quiesced = true;
    const std::uint32_t inactive = 0;
    (void)cudaMemcpy(reinterpret_cast<std::byte *>(deviceQueue) +
                         offsetof(abi::NvmeQueueView, active),
                     &inactive, sizeof(inactive), cudaMemcpyHostToDevice);
  }

  std::vector<std::uint64_t> readDmaPages(std::uint64_t handle,
                                          std::uint32_t count) {
    std::vector<std::uint64_t> pages(count);
    std::uint32_t first = 0;
    while (first < count) {
      nta_nvme_dma_pages request{};
      request.handle = handle;
      request.first_page = first;
      request.page_count =
          std::min<std::uint32_t>(NTA_NVME_MAX_DMA_PAGES, count - first);
      {
        std::scoped_lock lock(ioctlMutex);
        if (::ioctl(fd, NTA_NVME_IOCTL_GET_DMA_PAGES, &request) != 0) {
          throwSystem("NTA_NVME_IOCTL_GET_DMA_PAGES");
        }
      }
      if (request.page_count == 0) {
        throw std::runtime_error("NVMe mapping returned an empty DMA page set");
      }
      std::copy_n(request.addresses, request.page_count, pages.begin() + first);
      first += request.page_count;
    }
    return pages;
  }

  int fd = -1;
  int deviceOrdinal = 0;
  nta_nvme_info info{};
  NvmeCapabilities capabilities{};
  void *queueHost = nullptr;
  void *doorbellHost = nullptr;
  bool queueRegistered = false;
  bool doorbellRegistered = false;
  CUdeviceptr queueDevice = 0;
  CUdeviceptr doorbellDevice = 0;
  abi::NvmeCommandContext *contexts = nullptr;
  abi::NvmeQueueView *deviceQueue = nullptr;
  std::mutex ioctlMutex;
  bool quiesced = false;
};

struct NvmeBuffer::Impl {
  ~Impl() {
    detail::NoexceptCudaDeviceGuard deviceGuard(
        owner == nullptr ? 0 : owner->deviceOrdinal);
    if (owner) {
      owner->prepareMappingRelease();
      owner->releaseMapping(mappingHandle);
    }
    if (devicePageList != nullptr) {
      (void)cudaFree(devicePageList);
    }
    if (destination == NvmeDestination::Hbm && deviceAddress != nullptr) {
      (void)cudaFree(deviceAddress);
    }
    if (hostAddress != nullptr) {
      (void)cudaFreeHost(hostAddress);
    }
  }

  std::shared_ptr<NvmeTransport::Impl> owner;
  NvmeDestination destination = NvmeDestination::Hbm;
  void *hostAddress = nullptr;
  void *deviceAddress = nullptr;
  std::uint64_t *devicePageList = nullptr;
  std::uint64_t mappingHandle = 0;
  std::uint32_t pageCount = 0;
  std::size_t allocationBytes = 0;
};

NvmeBuffer::NvmeBuffer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
NvmeBuffer::~NvmeBuffer() = default;
NvmeBuffer::NvmeBuffer(NvmeBuffer &&) noexcept = default;
NvmeBuffer &NvmeBuffer::operator=(NvmeBuffer &&) noexcept = default;

void *NvmeBuffer::deviceAddress() const noexcept {
  return impl_->deviceAddress;
}

std::uint64_t NvmeBuffer::dmaPageListAddress() const noexcept {
  return reinterpret_cast<std::uint64_t>(impl_->devicePageList);
}

std::uint32_t NvmeBuffer::dmaPageCount() const noexcept {
  return impl_->pageCount;
}

std::size_t NvmeBuffer::bytes() const noexcept {
  return impl_->allocationBytes;
}

NvmeDestination NvmeBuffer::destination() const noexcept {
  return impl_->destination;
}

NvmeTransport::NvmeTransport(std::string devicePath, int deviceOrdinal)
    : impl_(std::make_shared<Impl>(devicePath, deviceOrdinal)) {}

NvmeTransport::~NvmeTransport() = default;
NvmeTransport::NvmeTransport(NvmeTransport &&) noexcept = default;
NvmeTransport &NvmeTransport::operator=(NvmeTransport &&) noexcept = default;

const NvmeCapabilities &NvmeTransport::capabilities() const noexcept {
  return impl_->capabilities;
}

int NvmeTransport::deviceOrdinal() const noexcept {
  return impl_->deviceOrdinal;
}

abi::NvmeQueueView *NvmeTransport::deviceQueue() const noexcept {
  return impl_->deviceQueue;
}

NvmeQueueStats NvmeTransport::readStats() const {
  detail::CudaDeviceGuard deviceGuard(impl_->deviceOrdinal);
  abi::NvmeQueueView queue{};
  checkCuda(cudaMemcpy(&queue, impl_->deviceQueue, sizeof(queue),
                       cudaMemcpyDeviceToHost),
            "download NvmeQueueView");
  return {
      queue.submitted,   queue.completed, queue.failed,
      queue.outstanding, queue.error,
  };
}

std::unique_ptr<NvmeBuffer>
NvmeTransport::allocate(std::size_t bytes, NvmeDestination destination) {
  detail::CudaDeviceGuard deviceGuard(impl_->deviceOrdinal);
  if (impl_->quiesced) {
    throw std::runtime_error("NVMe transport has been quiesced");
  }
  if (bytes == 0 || bytes > impl_->capabilities.maxTransferBytes ||
      bytes % impl_->capabilities.lbaSize != 0) {
    throw std::invalid_argument(
        "NVMe buffer size must be LBA aligned and within MDTS");
  }
  const std::size_t allocationBytes =
      roundUp(bytes, impl_->capabilities.controllerPageSize);
  if (destination == NvmeDestination::Hbm) {
    throw std::invalid_argument(
        "direct HBM NVMe DMA is disabled: the contained driver does not "
        "advertise a validated PCIe P2P route");
  }
  auto buffer = std::make_unique<NvmeBuffer::Impl>();
  buffer->owner = impl_;
  buffer->destination = destination;
  buffer->allocationBytes = allocationBytes;

  try {
    {
      checkCuda(cudaHostAlloc(&buffer->hostAddress, allocationBytes,
                              cudaHostAllocMapped),
                "cudaHostAlloc NVMe mapped destination");
      checkCuda(cudaHostGetDevicePointer(&buffer->deviceAddress,
                                         buffer->hostAddress, 0),
                "cudaHostGetDevicePointer NVMe mapped destination");
      nta_nvme_register_host request{
          reinterpret_cast<std::uint64_t>(buffer->hostAddress),
          allocationBytes,
          0,
          0,
          0,
      };
      {
        std::scoped_lock lock(impl_->ioctlMutex);
        if (::ioctl(impl_->fd, NTA_NVME_IOCTL_REGISTER_HOST, &request) != 0) {
          throwSystem("NTA_NVME_IOCTL_REGISTER_HOST");
        }
      }
      buffer->mappingHandle = request.handle;
      buffer->pageCount = request.dma_pages;
    }

    const std::vector<std::uint64_t> pages =
        impl_->readDmaPages(buffer->mappingHandle, buffer->pageCount);
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&buffer->devicePageList),
                         pages.size() * sizeof(pages.front())),
              "cudaMalloc NVMe DMA page list");
    checkCuda(cudaMemcpy(buffer->devicePageList, pages.data(),
                         pages.size() * sizeof(pages.front()),
                         cudaMemcpyHostToDevice),
              "upload NVMe DMA page list");
    return std::unique_ptr<NvmeBuffer>(new NvmeBuffer(std::move(buffer)));
  } catch (...) {
    throw;
  }
}

} // namespace nta
