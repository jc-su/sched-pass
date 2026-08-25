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
namespace detail {

NvmeMapping::~NvmeMapping() { reset(); }

NvmeMapping::NvmeMapping(NvmeMapping &&other) noexcept
    : backend_(other.backend_), token_(other.token_),
      pages_(std::move(other.pages_)) {
  other.backend_ = nullptr;
  other.token_ = {};
}

NvmeMapping &NvmeMapping::operator=(NvmeMapping &&other) noexcept {
  if (this != &other) {
    reset();
    backend_ = other.backend_;
    token_ = other.token_;
    pages_ = std::move(other.pages_);
    other.backend_ = nullptr;
    other.token_ = {};
  }
  return *this;
}

void NvmeMapping::reset() noexcept {
  if (backend_ != nullptr && token_) {
    backend_->release(token_);
  }
  backend_ = nullptr;
  token_ = {};
  pages_.clear();
}

void NvmeMapping::retainPagePrefix(std::size_t count) {
  if (count > pages_.size()) {
    throw std::out_of_range("NVMe mapping page prefix exceeds mapped pages");
  }
  pages_.resize(count);
}

} // namespace detail
namespace {

constexpr std::size_t DirectHbmQualificationBytes = 2U * 1024U * 1024U;

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

std::size_t roundUp(std::size_t value, std::size_t alignment);

struct HbmAllocation {
  CUdeviceptr base = 0;
  CUdeviceptr address = 0;
  std::size_t allocationBytes = 0;
  std::size_t mappedBytes = 0;
};

void releaseHbmAllocation(HbmAllocation &allocation) noexcept {
  if (allocation.base != 0) {
    (void)cuMemFree(allocation.base);
  }
  allocation = {};
}

void requireCudaHbmPeerCapability(int deviceOrdinal) {
  CUdevice device = 0;
  checkDriver(cuDeviceGet(&device, deviceOrdinal),
              "cuDeviceGet NVMe HBM peer device");
  int gpuDirectRdmaSupported = 0;
  int gpuDirectRdmaOrdering = 0;
  checkDriver(cuDeviceGetAttribute(
                  &gpuDirectRdmaSupported,
                  CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, device),
              "cuDeviceGetAttribute CUDA GPUDirect RDMA support");
  checkDriver(cuDeviceGetAttribute(
                  &gpuDirectRdmaOrdering,
                  CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING, device),
              "cuDeviceGetAttribute CUDA GPUDirect RDMA ordering");
  // The progress kernel consumes NVMe data after observing its CQE, without a
  // CPU-side CUDA flush between those events. This is correct only when the
  // device natively makes peer writes visible to the owning context.
  if (gpuDirectRdmaSupported == 0 ||
      gpuDirectRdmaOrdering < CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER) {
    throw std::runtime_error(
        "direct NVMe-to-HBM requires CUDA GPUDirect RDMA support and native "
        "owner-scope GPUDirect RDMA write ordering");
  }
}

HbmAllocation allocateHbm(std::size_t bytes, std::size_t pageSize) {
  constexpr std::size_t peerAlignment = 64U * 1024U;
  const std::size_t mappedBytes = roundUp(bytes, peerAlignment);
  if (mappedBytes % pageSize != 0 ||
      mappedBytes >
          std::numeric_limits<std::size_t>::max() - (peerAlignment - 1U)) {
    throw std::overflow_error("NVMe HBM peer allocation size overflows");
  }
  HbmAllocation allocation;
  try {
    allocation.allocationBytes = mappedBytes + peerAlignment - 1U;
    checkDriver(cuMemAlloc(&allocation.base, allocation.allocationBytes),
                "cuMemAlloc NVMe HBM peer destination");
    allocation.address = static_cast<CUdeviceptr>(
        roundUp(static_cast<std::size_t>(allocation.base), peerAlignment));
    allocation.mappedBytes = mappedBytes;
    return allocation;
  } catch (...) {
    releaseHbmAllocation(allocation);
    throw;
  }
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
    void *hostAllocation = nullptr;
    std::uint64_t *devicePageList = nullptr;
    detail::NvmeMapping dmaMapping;
    std::uint64_t mappingKey = 0;
    CUdeviceptr hbmBase = 0;
    CUdeviceptr hbmAddress = 0;
    std::size_t hbmAllocationBytes = 0;
    std::size_t hbmMappedBytes = 0;
    std::size_t allocationBytes = 0;
    std::size_t resourceBytes = 0;
    std::uint32_t pageCount = 0;
    NvmeDmaTarget target = NvmeDmaTarget::HostMapped;
    void *hostAddress = nullptr;
    void *deviceAddress = nullptr;
    bool cacheable = true;
    bool ownsDestinationMemory = true;
  };

  static constexpr std::size_t MappingCacheCapacity = 256;
  static constexpr std::size_t MappingCacheBytes = 256U * 1024U * 1024U;

  Impl(NvmeTransportOptions options)
      : deviceOrdinal(detail::resolveCudaDevice(options.deviceOrdinal)),
        dmaTarget(options.dmaTarget) {
    if (!options.endpoint.starts_with("vfio:")) {
      throw std::invalid_argument(
          "NVMe endpoint must use explicit vfio:DDDD:BB:SS.F ownership");
    }
    detail::CudaDeviceGuard deviceGuard(deviceOrdinal);
    checkCuda(cudaFree(nullptr), "initialize CUDA context for NVMe transport");
    if (dmaTarget == NvmeDmaTarget::HbmPeer) {
      requireCudaHbmPeerCapability(deviceOrdinal);
    }
    options.deviceOrdinal = deviceOrdinal;
    try {
      controlPlane = detail::createVfioNvmeControlPlane(options);
      const detail::NvmeQueueResources &resources = controlPlane->resources();
      capabilities = resources.capabilities;
      capabilities.deviceOrdinal = deviceOrdinal;
      capabilities.supportsHbmPeerDma = false;
      capabilities.hbmMappingBackend = NvmeHbmMappingBackend::Unavailable;
      if (dmaTarget == NvmeDmaTarget::HbmPeer) {
        HbmAllocation preflight;
        detail::NvmeMapping preflightMapping;
        try {
          preflight = allocateHbm(DirectHbmQualificationBytes,
                                  capabilities.controllerPageSize);
          preflightMapping = controlPlane->mappingBackend().mapHbm(
              preflight.address, preflight.mappedBytes);
          if (!preflightMapping || preflightMapping.pages().empty()) {
            throw std::runtime_error(
                "peer mapper returned no NVMe DMA addresses");
          }
          preflightMapping = {};
          releaseHbmAllocation(preflight);
          capabilities.supportsHbmPeerDma = true;
          capabilities.hbmMappingBackend =
              NvmeHbmMappingBackend::NvidiaPeerPages;
        } catch (const std::exception &error) {
          releaseHbmAllocation(preflight);
          throw std::runtime_error(
              "direct NVMe-to-HBM peer-page qualification failed after "
              "VFIO attach: " +
              std::string(error.what()));
        }
      }
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
      for (RetiredMapping &mapping : cachedMappings) {
        releaseMappingResources(mapping);
      }
      cachedMappings.clear();
      cachedBytes = 0;
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
    if (mappingKey == 0 || contexts == nullptr ||
        capabilities.queueDepth == 0) {
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
    mapping.dmaMapping = {};
    if (mapping.devicePageList != nullptr) {
      (void)cudaFree(mapping.devicePageList);
      mapping.devicePageList = nullptr;
    }
    if (mapping.ownsDestinationMemory && mapping.hostAllocation != nullptr) {
      (void)cudaFreeHost(mapping.hostAllocation);
      mapping.hostAllocation = nullptr;
    }
    HbmAllocation hbm{mapping.hbmBase, mapping.hbmAddress,
                      mapping.hbmAllocationBytes, mapping.hbmMappedBytes};
    if (mapping.ownsDestinationMemory) {
      releaseHbmAllocation(hbm);
    }
    mapping.hbmBase = 0;
    mapping.hbmAddress = 0;
    mapping.hbmAllocationBytes = 0;
    mapping.hbmMappedBytes = 0;
    mapping.allocationBytes = 0;
    mapping.resourceBytes = 0;
    mapping.pageCount = 0;
    mapping.hostAddress = nullptr;
    mapping.deviceAddress = nullptr;
  }

  void cacheMappingResourcesLocked(RetiredMapping mapping) noexcept {
    if (!mapping.cacheable || !mapping.dmaMapping ||
        mapping.devicePageList == nullptr ||
        mapping.pageCount == 0 || mapping.allocationBytes == 0 ||
        mapping.resourceBytes == 0) {
      releaseMappingResources(mapping);
      return;
    }
    if (mapping.resourceBytes > MappingCacheBytes) {
      releaseMappingResources(mapping);
      return;
    }
    while (!cachedMappings.empty() &&
           (cachedMappings.size() >= MappingCacheCapacity ||
            cachedBytes > MappingCacheBytes - mapping.resourceBytes)) {
      cachedBytes -= cachedMappings.front().resourceBytes;
      releaseMappingResources(cachedMappings.front());
      cachedMappings.erase(cachedMappings.begin());
    }
    try {
      const std::size_t resourceBytes = mapping.resourceBytes;
      cachedMappings.push_back(std::move(mapping));
      cachedBytes += resourceBytes;
    } catch (...) {
      releaseMappingResources(mapping);
    }
  }

  bool takeCachedMapping(std::size_t bytes, NvmeDmaTarget target,
                         RetiredMapping &mapping) {
    std::scoped_lock lock(mappingMutex);
    const auto cached = std::ranges::find_if(
        cachedMappings, [bytes, target](const RetiredMapping &candidate) {
          return candidate.allocationBytes == bytes &&
                 candidate.target == target;
        });
    if (cached == cachedMappings.end()) {
      return false;
    }
    cachedBytes -= cached->resourceBytes;
    mapping = std::move(*cached);
    cachedMappings.erase(cached);
    return true;
  }

  void reapMappings() noexcept {
    {
      std::scoped_lock lock(mappingMutex);
      if (retiredMappings.empty()) {
        return;
      }
    }
    detail::NoexceptCudaDeviceGuard deviceGuard(deviceOrdinal);
    std::scoped_lock lock(mappingMutex);
    auto mapping = retiredMappings.begin();
    while (mapping != retiredMappings.end()) {
      if (mappingInFlight(mapping->mappingKey)) {
        ++mapping;
        continue;
      }
      cacheMappingResourcesLocked(std::move(*mapping));
      mapping = retiredMappings.erase(mapping);
    }
  }

  void releaseMapping(RetiredMapping mapping) noexcept {
    detail::NoexceptCudaDeviceGuard deviceGuard(deviceOrdinal);
    // Do not turn destruction of a reusable buffer into a device-wide fence.
    // ``mappingInFlight`` performs the narrow queue-context check needed to
    // decide whether the lease can be cached.  If the buffer is still used by
    // an NVMe command, it is retained in ``retiredMappings`` and reclaimed by
    // a later allocation/statistics pass.  Transport shutdown remains the
    // explicit whole-device quiescence boundary in ``release``.
    std::scoped_lock lock(mappingMutex);
    if (mappingInFlight(mapping.mappingKey)) {
      try {
        retiredMappings.push_back(std::move(mapping));
      } catch (...) {
        releaseMappingResources(mapping);
      }
    } else {
      cacheMappingResourcesLocked(std::move(mapping));
    }
  }

  int deviceOrdinal = 0;
  NvmeDmaTarget dmaTarget = NvmeDmaTarget::HbmPeer;
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
  std::vector<RetiredMapping> cachedMappings;
  std::size_t cachedBytes = 0;
};

struct NvmeBuffer::Impl {
  ~Impl() {
    detail::NoexceptCudaDeviceGuard deviceGuard(
        owner == nullptr ? 0 : owner->deviceOrdinal);
    if (owner != nullptr) {
      NvmeTransport::Impl::RetiredMapping mapping;
      mapping.hostAllocation = hostAllocation;
      mapping.devicePageList = devicePageList;
      mapping.dmaMapping = std::move(dmaMapping);
      mapping.mappingKey = reinterpret_cast<std::uint64_t>(devicePageList);
      mapping.hbmBase = hbmBase;
      mapping.hbmAddress = hbmAddress;
      mapping.hbmAllocationBytes = hbmAllocationBytes;
      mapping.hbmMappedBytes = hbmMappedBytes;
      mapping.allocationBytes = allocationBytes;
      mapping.resourceBytes = resourceBytes;
      mapping.pageCount = pageCount;
      mapping.target = target;
      mapping.hostAddress = hostAddress;
      mapping.deviceAddress = deviceAddress;
      mapping.cacheable = cacheable;
      mapping.ownsDestinationMemory = ownsDestinationMemory;
      owner->releaseMapping(std::move(mapping));
      hostAllocation = nullptr;
      hostAddress = nullptr;
      deviceAddress = nullptr;
      devicePageList = nullptr;
      dmaMapping = {};
      hbmBase = 0;
      hbmAddress = 0;
      hbmAllocationBytes = 0;
      hbmMappedBytes = 0;
      resourceBytes = 0;
    }
  }

  // Declaration order is part of the lifetime contract: C++ destroys fields
  // in reverse order, so dmaMapping is released before owner.  The mapping's
  // non-owning backend pointer is consequently valid even when a buffer
  // outlives the public NvmeTransport handle.
  std::shared_ptr<NvmeTransport::Impl> owner;
  void *hostAllocation = nullptr;
  void *hostAddress = nullptr;
  void *deviceAddress = nullptr;
  std::uint64_t *devicePageList = nullptr;
  detail::NvmeMapping dmaMapping;
  std::uint32_t pageCount = 0;
  std::size_t allocationBytes = 0;
  std::size_t resourceBytes = 0;
  CUdeviceptr hbmBase = 0;
  CUdeviceptr hbmAddress = 0;
  std::size_t hbmAllocationBytes = 0;
  std::size_t hbmMappedBytes = 0;
  NvmeDmaTarget target = NvmeDmaTarget::HostMapped;
  bool cacheable = true;
  bool ownsDestinationMemory = true;
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

NvmeDmaTarget NvmeBuffer::dmaTarget() const noexcept { return impl_->target; }

bool NvmeBuffer::ownsDestinationMemory() const noexcept {
  return impl_ != nullptr && impl_->ownsDestinationMemory;
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

std::unique_ptr<NvmeBuffer> NvmeTransport::allocate(std::size_t bytes) {
  detail::CudaDeviceGuard deviceGuard(impl_->deviceOrdinal);
  impl_->reapMappings();
  if (bytes == 0 || bytes > impl_->capabilities.maxTransferBytes ||
      bytes % impl_->capabilities.lbaSize != 0) {
    throw std::invalid_argument(
        "NVMe buffer size must be LBA aligned and within MDTS/PRP capacity");
  }
  const std::size_t allocationBytes =
      roundUp(bytes, impl_->capabilities.controllerPageSize);
  const std::size_t alignment = impl_->capabilities.controllerPageSize;
  auto buffer = std::make_unique<NvmeBuffer::Impl>();
  buffer->owner = impl_;
  buffer->allocationBytes = allocationBytes;
  buffer->target = impl_->dmaTarget;

  NvmeTransport::Impl::RetiredMapping cached;
  if (impl_->takeCachedMapping(allocationBytes, impl_->dmaTarget, cached)) {
    if (cached.dmaMapping && cached.devicePageList != nullptr &&
        cached.pageCount != 0 && cached.deviceAddress != nullptr) {
      buffer->hostAllocation = cached.hostAllocation;
      buffer->hostAddress = cached.hostAddress;
      buffer->deviceAddress = cached.deviceAddress;
      buffer->devicePageList = cached.devicePageList;
      buffer->dmaMapping = std::move(cached.dmaMapping);
      buffer->pageCount = cached.pageCount;
      buffer->hbmBase = cached.hbmBase;
      buffer->hbmAddress = cached.hbmAddress;
      buffer->hbmAllocationBytes = cached.hbmAllocationBytes;
      buffer->hbmMappedBytes = cached.hbmMappedBytes;
      buffer->resourceBytes = cached.resourceBytes;
      return std::unique_ptr<NvmeBuffer>(new NvmeBuffer(std::move(buffer)));
    }
    impl_->releaseMappingResources(cached);
  }

  detail::NvmeMapping mapping;
  if (impl_->dmaTarget == NvmeDmaTarget::HbmPeer) {
    HbmAllocation hbm =
        allocateHbm(allocationBytes, impl_->capabilities.controllerPageSize);
    buffer->hbmBase = hbm.base;
    buffer->hbmAddress = hbm.address;
    buffer->hbmAllocationBytes = hbm.allocationBytes;
    buffer->hbmMappedBytes = hbm.mappedBytes;
    buffer->resourceBytes = hbm.allocationBytes;
    buffer->deviceAddress = reinterpret_cast<void *>(hbm.address);
    hbm = {};
    mapping = impl_->controlPlane->mappingBackend().mapHbm(
        buffer->hbmAddress, buffer->hbmMappedBytes);
    const std::size_t requiredPages =
        allocationBytes / impl_->capabilities.controllerPageSize;
    if (mapping.pages().size() < requiredPages) {
      throw std::runtime_error(
          "NVMe HBM DMA mapping is shorter than the requested transfer");
    }
    // The peer allocation is 64 KiB aligned, while the controller page may be
    // 4 KiB.  Keep the lease for the whole mapping but publish only the exact
    // PRP prefix for this NVMe command.
    mapping.retainPagePrefix(requiredPages);
  } else {
    if (allocationBytes >
        std::numeric_limits<std::size_t>::max() - (alignment - 1U)) {
      throw std::overflow_error("NVMe pinned allocation size overflows size_t");
    }
    const std::size_t pinnedBytes = allocationBytes + alignment - 1U;
    buffer->resourceBytes = pinnedBytes;
    checkCuda(cudaHostAlloc(&buffer->hostAllocation, pinnedBytes,
                            cudaHostAllocMapped),
              "cudaHostAlloc NVMe mapped destination");
    const auto pinnedAddress =
        reinterpret_cast<std::uintptr_t>(buffer->hostAllocation);
    buffer->hostAddress = reinterpret_cast<void *>(
        (pinnedAddress + alignment - 1U) & ~(alignment - 1U));
    checkCuda(cudaHostGetDevicePointer(&buffer->deviceAddress,
                                       buffer->hostAddress, 0),
              "cudaHostGetDevicePointer NVMe mapped destination");
    mapping = impl_->controlPlane->mappingBackend().mapHost(buffer->hostAddress,
                                                            allocationBytes);
  }
  buffer->dmaMapping = std::move(mapping);
  if (!buffer->dmaMapping || buffer->dmaMapping.pages().empty() ||
      buffer->dmaMapping.pages().size() >
          std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("NVMe backend returned an invalid DMA page list");
  }
  buffer->pageCount =
      static_cast<std::uint32_t>(buffer->dmaMapping.pages().size());
  checkCuda(cudaMalloc(reinterpret_cast<void **>(&buffer->devicePageList),
                       buffer->dmaMapping.pages().size() *
                           sizeof(buffer->dmaMapping.pages().front())),
            "cudaMalloc NVMe DMA page list");
  checkCuda(cudaMemcpy(buffer->devicePageList,
                       buffer->dmaMapping.pages().data(),
                       buffer->dmaMapping.pages().size() *
                           sizeof(buffer->dmaMapping.pages().front()),
                       cudaMemcpyHostToDevice),
            "upload NVMe DMA page list");
  return std::unique_ptr<NvmeBuffer>(new NvmeBuffer(std::move(buffer)));
}

std::unique_ptr<NvmeBuffer>
NvmeTransport::mapExternalHbm(void *deviceAddress, std::size_t bytes) {
  detail::CudaDeviceGuard deviceGuard(impl_->deviceOrdinal);
  impl_->reapMappings();
  constexpr std::size_t peerAlignment = 64U * 1024U;
  if (impl_->dmaTarget != NvmeDmaTarget::HbmPeer) {
    throw std::invalid_argument(
        "external HBM NVMe destinations require dmaTarget=hbm-peer");
  }
  if (deviceAddress == nullptr || bytes == 0 ||
      bytes > impl_->capabilities.maxTransferBytes ||
      bytes % impl_->capabilities.lbaSize != 0 ||
      reinterpret_cast<std::uintptr_t>(deviceAddress) % peerAlignment != 0 ||
      bytes % peerAlignment != 0 ||
      bytes % impl_->capabilities.controllerPageSize != 0) {
    throw std::invalid_argument(
        "external HBM NVMe destination must be LBA/page/64 KiB aligned and "
        "within the controller transfer limit");
  }

  cudaPointerAttributes attributes{};
  const cudaError_t attributeStatus =
      cudaPointerGetAttributes(&attributes, deviceAddress);
  if (attributeStatus != cudaSuccess) {
    (void)cudaGetLastError();
    throw std::invalid_argument(
        "external HBM NVMe destination is not a live CUDA allocation");
  }
  if (attributes.type != cudaMemoryTypeDevice ||
      attributes.device != impl_->deviceOrdinal) {
    throw std::invalid_argument(
        "external NVMe destination must be device memory on the transport "
        "CUDA device");
  }

  CUdeviceptr allocationBase = 0;
  std::size_t allocationBytes = 0;
  checkDriver(cuMemGetAddressRange(
                  &allocationBase, &allocationBytes,
                  reinterpret_cast<CUdeviceptr>(deviceAddress)),
              "query external HBM allocation range");
  const CUdeviceptr address = reinterpret_cast<CUdeviceptr>(deviceAddress);
  if (address < allocationBase ||
      static_cast<std::size_t>(address - allocationBase) > allocationBytes ||
      bytes >
          allocationBytes - static_cast<std::size_t>(address - allocationBase)) {
    throw std::invalid_argument(
        "external HBM NVMe destination exceeds its CUDA allocation");
  }

  auto buffer = std::make_unique<NvmeBuffer::Impl>();
  buffer->owner = impl_;
  buffer->target = NvmeDmaTarget::HbmPeer;
  buffer->deviceAddress = deviceAddress;
  buffer->allocationBytes = bytes;
  buffer->resourceBytes = bytes;
  buffer->cacheable = false;
  buffer->ownsDestinationMemory = false;

  detail::NvmeMapping mapping = impl_->controlPlane->mappingBackend().mapHbm(
      address, bytes);
  const std::size_t requiredPages =
      bytes / impl_->capabilities.controllerPageSize;
  if (mapping.pages().size() < requiredPages) {
    throw std::runtime_error(
        "external HBM DMA mapping is shorter than the requested transfer");
  }
  mapping.retainPagePrefix(requiredPages);
  buffer->dmaMapping = std::move(mapping);
  if (!buffer->dmaMapping || buffer->dmaMapping.pages().empty() ||
      buffer->dmaMapping.pages().size() >
          std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error(
        "external HBM backend returned an invalid DMA page list");
  }
  buffer->pageCount =
      static_cast<std::uint32_t>(buffer->dmaMapping.pages().size());
  checkCuda(cudaMalloc(reinterpret_cast<void **>(&buffer->devicePageList),
                       buffer->dmaMapping.pages().size() *
                           sizeof(buffer->dmaMapping.pages().front())),
            "cudaMalloc external NVMe DMA page list");
  checkCuda(cudaMemcpy(buffer->devicePageList,
                       buffer->dmaMapping.pages().data(),
                       buffer->dmaMapping.pages().size() *
                           sizeof(buffer->dmaMapping.pages().front()),
                       cudaMemcpyHostToDevice),
            "upload external NVMe DMA page list");
  return std::unique_ptr<NvmeBuffer>(new NvmeBuffer(std::move(buffer)));
}

} // namespace nta
