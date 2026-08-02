#include "nta/NvmeRuntime.h"

#include "CudaDeviceGuard.h"
#include "NvmeControlPlane.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
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

std::size_t roundUp(std::size_t value, std::size_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1U)) != 0) {
    throw std::invalid_argument(
        "NVMe allocation alignment is not a power of two");
  }
  if (value > std::numeric_limits<std::size_t>::max() - (alignment - 1U)) {
    throw std::overflow_error("NVMe allocation size overflows size_t");
  }
  return (value + alignment - 1U) & ~(alignment - 1U);
}

void writeGpuMmio(CUdeviceptr address, std::uint32_t value) {
  static constexpr char ptx[] = R"ptx(
.version 8.7
.target sm_70
.address_size 64

.visible .entry nta_probe_mmio(
    .param .u64 address,
    .param .u32 value)
{
    .reg .b64 %rd1;
    .reg .b32 %r1;
    ld.param.u64 %rd1, [address];
    ld.param.u32 %r1, [value];
    st.mmio.relaxed.sys.b32 [%rd1], %r1;
    ret;
}
)ptx";
  CUmodule module = nullptr;
  checkDriver(cuModuleLoadData(&module, ptx),
              "cuModuleLoadData NVMe BAR compatibility probe");
  try {
    CUfunction function = nullptr;
    checkDriver(cuModuleGetFunction(&function, module, "nta_probe_mmio"),
                "cuModuleGetFunction NVMe BAR compatibility probe");
    void *arguments[] = {&address, &value};
    checkDriver(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr,
                               arguments, nullptr),
                "cuLaunchKernel NVMe BAR compatibility probe");
    checkDriver(cuCtxSynchronize(),
                "cuCtxSynchronize NVMe BAR compatibility probe");
  } catch (...) {
    (void)cuModuleUnload(module);
    throw;
  }
  checkDriver(cuModuleUnload(module),
              "cuModuleUnload NVMe BAR compatibility probe");
}

template <typename T> T toLittle(T value) {
  static_assert(std::is_unsigned_v<T>);
  if constexpr (std::endian::native == std::endian::little) {
    return value;
  } else if constexpr (sizeof(T) == sizeof(std::uint16_t)) {
    return static_cast<T>(__builtin_bswap16(static_cast<std::uint16_t>(value)));
  } else if constexpr (sizeof(T) == sizeof(std::uint32_t)) {
    return static_cast<T>(__builtin_bswap32(static_cast<std::uint32_t>(value)));
  } else {
    static_assert(sizeof(T) == sizeof(std::uint64_t));
    return static_cast<T>(__builtin_bswap64(static_cast<std::uint64_t>(value)));
  }
}

void qualifyGpuNvmePath(const detail::NvmeQueueResources &resources,
                        CUdeviceptr doorbellDevice,
                        const NvmeTransportOptions &options) {
  constexpr std::uint32_t probeTail = 1;
  const std::uint16_t commandId =
      static_cast<std::uint16_t>(resources.capabilities.queueDepth - 1U);
  auto *submission = reinterpret_cast<abi::NvmeSubmission *>(
      static_cast<std::byte *>(resources.queueHost) + resources.sqOffset);
  auto *completion = reinterpret_cast<abi::NvmeCompletion *>(
      static_cast<std::byte *>(resources.queueHost) + resources.cqOffset);
  std::memset(submission, 0, sizeof(*submission));
  std::memset(completion, 0, sizeof(*completion));
  submission->dword[0] = toLittle<std::uint32_t>(
      0x02U | (static_cast<std::uint32_t>(commandId) << 16U));
  submission->dword[1] = toLittle(options.namespaceId);
  const std::uint64_t prp = toLittle(resources.prpDmaAddress);
  std::memcpy(&submission->dword[6], &prp, sizeof(prp));
  submission->dword[12] = 0;
  std::atomic_thread_fence(std::memory_order_seq_cst);

  writeGpuMmio(doorbellDevice + resources.sqDoorbellOffset, probeTail);
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(options.adminTimeoutMs);
  std::uint32_t commandAndStatus = 0;
  const auto *completionDword =
      reinterpret_cast<volatile std::uint32_t *>(&completion->dword[3]);
  while (true) {
    commandAndStatus = toLittle(*completionDword);
    const std::uint16_t status =
        static_cast<std::uint16_t>(commandAndStatus >> 16U);
    if ((status & 1U) == 1U) {
      break;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error(
          "GPU NVMe path qualification timed out: the BAR VMA registered, but "
          "a GPU MMIO doorbell did not produce a completion; check PCIe peer "
          "routing, root-port topology, and ACS");
    }
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  std::atomic_thread_fence(std::memory_order_acquire);
  const std::uint16_t completionId =
      static_cast<std::uint16_t>(commandAndStatus & 0xffffU);
  const std::uint16_t status =
      static_cast<std::uint16_t>(commandAndStatus >> 16U);
  if (completionId != commandId || (status >> 1U) != 0) {
    throw std::runtime_error(
        "GPU NVMe path qualification returned an invalid completion");
  }
  *reinterpret_cast<volatile std::uint32_t *>(
      static_cast<std::byte *>(resources.doorbellHost) +
      resources.cqDoorbellOffset) = toLittle(probeTail);
  std::atomic_thread_fence(std::memory_order_seq_cst);
}

} // namespace

struct NvmeTransport::Impl {
  struct RetiredMapping {
    void *hostAddress = nullptr;
    std::uint64_t *devicePageList = nullptr;
    std::uint64_t mappingHandle = 0;
    std::uint64_t mappingKey = 0;
  };

  Impl(NvmeTransportOptions options)
      : deviceOrdinal(detail::resolveCudaDevice(options.deviceOrdinal)) {
    if (!options.endpoint.starts_with("vfio:")) {
      throw std::invalid_argument(
          "NVMe endpoint must use explicit vfio:DDDD:BB:SS.F ownership");
    }
    detail::CudaDeviceGuard deviceGuard(deviceOrdinal);
    checkCuda(cudaFree(nullptr), "initialize CUDA context for NVMe transport");
    options.deviceOrdinal = deviceOrdinal;
    try {
      controlPlane = detail::createVfioNvmeControlPlane(options);
      const detail::NvmeQueueResources &resources = controlPlane->resources();
      capabilities = resources.capabilities;
      capabilities.deviceOrdinal = deviceOrdinal;
      if (capabilities.lbaSize > capabilities.controllerPageSize) {
        throw std::runtime_error(
            "NVMe namespace LBA exceeds the one-page qualification buffer");
      }

      unsigned int queueFlags = CU_MEMHOSTREGISTER_DEVICEMAP;
      if (resources.queueIsIoMemory) {
        queueFlags |= CU_MEMHOSTREGISTER_IOMEMORY;
      }
      checkDriver(cuMemHostRegister(resources.queueHost, resources.queueBytes,
                                    queueFlags),
                  "cuMemHostRegister NVMe queue memory");
      queueRegistered = true;
      checkDriver(
          cuMemHostGetDevicePointer(&queueDevice, resources.queueHost, 0),
          "cuMemHostGetDevicePointer NVMe queue memory");

      auto *controlHost = reinterpret_cast<abi::NvmeQueueControl *>(
          static_cast<std::byte *>(resources.queueHost) +
          resources.controlOffset);
      if (controlHost->magic != abi::NvmeQueueControlMagic ||
          controlHost->abiVersion != abi::NvmeQueueAbiVersion ||
          controlHost->state !=
              static_cast<std::uint32_t>(abi::NvmeQueueState::Online) ||
          controlHost->generation != resources.generation ||
          controlHost->queueId != capabilities.queueId) {
        throw std::runtime_error("NVMe queue control page is inconsistent");
      }

      // Registration alone does not prove that GPU PCIe MMIO reaches the
      // controller. The end-to-end qualification below requires an NVMe CQE.
      checkDriver(cuMemHostRegister(resources.doorbellHost,
                                    resources.doorbellBytes,
                                    CU_MEMHOSTREGISTER_DEVICEMAP |
                                        CU_MEMHOSTREGISTER_IOMEMORY),
                  "cuMemHostRegister NVMe doorbell BAR VMA");
      doorbellRegistered = true;
      checkDriver(
          cuMemHostGetDevicePointer(&doorbellDevice, resources.doorbellHost, 0),
          "cuMemHostGetDevicePointer NVMe doorbell BAR VMA");
      const bool trustedReadOnlyCode =
          options.mediaPolicy == NvmeMediaPolicy::TrustReadOnlyDeviceCode;
      if (!capabilities.translatedIommu ||
          (!capabilities.namespaceReadOnly && !trustedReadOnlyCode)) {
        throw std::runtime_error(
            "NVMe media policy rejected the controller: "
            "every active namespace must support hardware write protection or "
            "the transport must explicitly trust read-only device code");
      }
      qualifyGpuNvmePath(resources, doorbellDevice, options);
      capabilities.gpuDoorbellMappingValidated = true;

      checkCuda(
          cudaMalloc(reinterpret_cast<void **>(&contexts),
                     sizeof(abi::NvmeCommandContext) * capabilities.queueDepth),
          "cudaMalloc NVMe command contexts");
      checkCuda(
          cudaMemset(contexts, 0,
                     sizeof(abi::NvmeCommandContext) * capabilities.queueDepth),
          "cudaMemset NVMe command contexts");

      abi::NvmeQueueView hostQueue{};
      hostQueue.submissions = reinterpret_cast<abi::NvmeSubmission *>(
          queueDevice + resources.sqOffset);
      hostQueue.completions = reinterpret_cast<abi::NvmeCompletion *>(
          queueDevice + resources.cqOffset);
      hostQueue.prpLists =
          reinterpret_cast<std::uint64_t *>(queueDevice + resources.prpOffset);
      hostQueue.prpListDmaAddress = resources.prpDmaAddress;
      hostQueue.sqDoorbell = reinterpret_cast<volatile std::uint32_t *>(
          doorbellDevice + resources.sqDoorbellOffset);
      hostQueue.cqDoorbell = reinterpret_cast<volatile std::uint32_t *>(
          doorbellDevice + resources.cqDoorbellOffset);
      hostQueue.contexts = contexts;
      hostQueue.control = reinterpret_cast<abi::NvmeQueueControl *>(
          queueDevice + resources.controlOffset);
      hostQueue.depth = capabilities.queueDepth;
      hostQueue.controllerPageSize = capabilities.controllerPageSize;
      hostQueue.lbaShift =
          static_cast<std::uint32_t>(std::countr_zero(capabilities.lbaSize));
      hostQueue.namespaceId = options.namespaceId;
      hostQueue.sqTail = 1;
      hostQueue.cqHead = 1;
      hostQueue.cqPhase = 1;
      hostQueue.active = 1;
      hostQueue.queueGeneration = resources.generation;
      hostQueue.queueId = capabilities.queueId;
      hostQueue.directMaxPrpPages = std::min<std::uint32_t>(
          32, capabilities.controllerPageSize / sizeof(std::uint64_t));
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
      const std::uint32_t inactive = 0;
      (void)cudaMemcpy(reinterpret_cast<std::byte *>(deviceQueue) +
                           offsetof(abi::NvmeQueueView, active),
                       &inactive, sizeof(inactive), cudaMemcpyHostToDevice);
    }
    if (controlPlane != nullptr) {
      controlPlane->quiesce();
    }
    {
      std::scoped_lock lock(mappingMutex);
      for (RetiredMapping &mapping : retiredMappings) {
        releaseMappingResources(mapping);
      }
      retiredMappings.clear();
    }
    if (deviceQueue != nullptr) {
      (void)cudaFree(deviceQueue);
      deviceQueue = nullptr;
    }
    if (contexts != nullptr) {
      (void)cudaFree(contexts);
      contexts = nullptr;
    }
    const detail::NvmeQueueResources *resources =
        controlPlane == nullptr ? nullptr : &controlPlane->resources();
    if (doorbellRegistered) {
      (void)cuMemHostUnregister(resources->doorbellHost);
      doorbellRegistered = false;
    }
    if (queueRegistered) {
      (void)cuMemHostUnregister(resources->queueHost);
      queueRegistered = false;
    }
    controlPlane.reset();
  }

  bool mappingInFlight(std::uint64_t mappingKey) noexcept {
    if (mappingKey == 0 || contexts == nullptr || capabilities.queueDepth == 0) {
      return false;
    }
    std::vector<abi::NvmeCommandContext> hostContexts(capabilities.queueDepth);
    if (cudaMemcpy(hostContexts.data(), contexts,
                   hostContexts.size() * sizeof(hostContexts.front()),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      return true;
    }
    return std::ranges::any_of(hostContexts, [mappingKey](const auto &context) {
      return context.active != 0 && context.mappingKey == mappingKey;
    });
  }

  void releaseMappingResources(RetiredMapping &mapping) noexcept {
    if (controlPlane != nullptr && mapping.mappingHandle != 0) {
      controlPlane->unmapHost(mapping.mappingHandle);
      mapping.mappingHandle = 0;
    }
    if (mapping.devicePageList != nullptr) {
      (void)cudaFree(mapping.devicePageList);
      mapping.devicePageList = nullptr;
    }
    if (mapping.hostAddress != nullptr) {
      (void)cudaFreeHost(mapping.hostAddress);
      mapping.hostAddress = nullptr;
    }
  }

  void reapMappings() noexcept {
    detail::NoexceptCudaDeviceGuard deviceGuard(deviceOrdinal);
    (void)cudaDeviceSynchronize();
    std::scoped_lock lock(mappingMutex);
    auto mapping = retiredMappings.begin();
    while (mapping != retiredMappings.end()) {
      if (mappingInFlight(mapping->mappingKey)) {
        ++mapping;
        continue;
      }
      releaseMappingResources(*mapping);
      mapping = retiredMappings.erase(mapping);
    }
  }

  void releaseMapping(RetiredMapping mapping) noexcept {
    detail::NoexceptCudaDeviceGuard deviceGuard(deviceOrdinal);
    (void)cudaDeviceSynchronize();
    std::scoped_lock lock(mappingMutex);
    if (mappingInFlight(mapping.mappingKey)) {
      retiredMappings.push_back(mapping);
    } else {
      releaseMappingResources(mapping);
    }
  }

  int deviceOrdinal = 0;
  NvmeCapabilities capabilities{};
  std::unique_ptr<detail::NvmeControlPlane> controlPlane;
  bool queueRegistered = false;
  bool doorbellRegistered = false;
  CUdeviceptr queueDevice = 0;
  CUdeviceptr doorbellDevice = 0;
  abi::NvmeCommandContext *contexts = nullptr;
  abi::NvmeQueueView *deviceQueue = nullptr;
  std::mutex mappingMutex;
  std::vector<RetiredMapping> retiredMappings;
};

struct NvmeBuffer::Impl {
  ~Impl() {
    detail::NoexceptCudaDeviceGuard deviceGuard(
        owner == nullptr ? 0 : owner->deviceOrdinal);
    if (owner != nullptr) {
      owner->releaseMapping(
          {hostAddress, devicePageList, mappingHandle,
           reinterpret_cast<std::uint64_t>(devicePageList)});
      hostAddress = nullptr;
      devicePageList = nullptr;
      mappingHandle = 0;
    }
  }

  std::shared_ptr<NvmeTransport::Impl> owner;
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

NvmeTransport::NvmeTransport(std::string devicePath, int deviceOrdinal)
    : NvmeTransport(
          NvmeTransportOptions{std::move(devicePath), deviceOrdinal}) {}

NvmeTransport::NvmeTransport(NvmeTransportOptions options)
    : impl_(std::make_shared<Impl>(std::move(options))) {}

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
  impl_->reapMappings();
  abi::NvmeQueueView queue{};
  checkCuda(cudaMemcpy(&queue, impl_->deviceQueue, sizeof(queue),
                       cudaMemcpyDeviceToHost),
            "download NvmeQueueView");
  abi::NvmeCompletion nextCompletion{};
  checkCuda(cudaMemcpy(&nextCompletion, queue.completions + queue.cqHead,
                       sizeof(nextCompletion), cudaMemcpyDeviceToHost),
            "download next NVMe completion");
  return {
      queue.submitted,
      queue.completed,
      queue.failed,
      queue.directSubmitted,
      queue.directFallbacks,
      queue.outstanding,
      queue.error,
      queue.sqTail,
      queue.cqHead,
      queue.cqPhase,
      nextCompletion.dword[3],
  };
}

std::unique_ptr<NvmeBuffer>
NvmeTransport::allocate(std::size_t bytes) {
  detail::CudaDeviceGuard deviceGuard(impl_->deviceOrdinal);
  impl_->reapMappings();
  if (bytes == 0 || bytes > impl_->capabilities.maxTransferBytes ||
      bytes % impl_->capabilities.lbaSize != 0) {
    throw std::invalid_argument(
        "NVMe buffer size must be LBA aligned and within MDTS/PRP capacity");
  }
  const std::size_t allocationBytes =
      roundUp(bytes, impl_->capabilities.controllerPageSize);
  auto buffer = std::make_unique<NvmeBuffer::Impl>();
  buffer->owner = impl_;
  buffer->allocationBytes = allocationBytes;
  checkCuda(
      cudaHostAlloc(&buffer->hostAddress, allocationBytes, cudaHostAllocMapped),
      "cudaHostAlloc NVMe mapped destination");
  checkCuda(
      cudaHostGetDevicePointer(&buffer->deviceAddress, buffer->hostAddress, 0),
      "cudaHostGetDevicePointer NVMe mapped destination");

  detail::NvmeDmaMapping mapping =
      impl_->controlPlane->mapHost(buffer->hostAddress, allocationBytes);
  if (mapping.pages.empty() ||
      mapping.pages.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("NVMe backend returned an invalid DMA page list");
  }
  buffer->mappingHandle = mapping.handle;
  buffer->pageCount = static_cast<std::uint32_t>(mapping.pages.size());
  checkCuda(cudaMalloc(reinterpret_cast<void **>(&buffer->devicePageList),
                       mapping.pages.size() * sizeof(mapping.pages.front())),
            "cudaMalloc NVMe DMA page list");
  checkCuda(cudaMemcpy(buffer->devicePageList, mapping.pages.data(),
                       mapping.pages.size() * sizeof(mapping.pages.front()),
                       cudaMemcpyHostToDevice),
            "upload NVMe DMA page list");
  return std::unique_ptr<NvmeBuffer>(new NvmeBuffer(std::move(buffer)));
}

} // namespace nta
