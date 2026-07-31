#include "nta/HostRuntime.h"
#include "nta/NvmeRuntime.h"

#include <cuda_runtime_api.h>

#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
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

template <typename T> T *deviceAllocate(std::size_t count) {
  T *pointer = nullptr;
  checkCuda(cudaMalloc(reinterpret_cast<void **>(&pointer), sizeof(T) * count),
            "cudaMalloc");
  checkCuda(cudaMemset(pointer, 0, sizeof(T) * count), "cudaMemset");
  return pointer;
}

template <typename T>
void uploadOne(T *destination, std::uint32_t slot, const T &value) {
  checkCuda(
      cudaMemcpy(destination + slot, &value, sizeof(T), cudaMemcpyHostToDevice),
      "cudaMemcpy host-to-device");
}

template <typename T> T downloadOne(const T *source, std::uint32_t slot) {
  T value{};
  checkCuda(
      cudaMemcpy(&value, source + slot, sizeof(T), cudaMemcpyDeviceToHost),
      "cudaMemcpy device-to-host");
  return value;
}

} // namespace

struct HostRuntime::Impl {
  struct OwnedObject {
    Placement placement;
    void *hostAllocation;
    void *sourceDevice;
    void *stagingDevice;
    std::unique_ptr<NvmeBuffer> nvmeBuffer;
  };

  explicit Impl(RuntimeConfig runtimeConfig,
                std::shared_ptr<NvmeTransport> nvmeTransport = nullptr)
      : config(runtimeConfig), requestsHost(config.requestCapacity),
        requestInstalled(config.requestCapacity, false),
        objects(config.objectCapacity), nvme(std::move(nvmeTransport)) {
    if (config.requestCapacity == 0 || config.objectCapacity == 0 ||
        config.intentCapacity == 0 || config.continuationCapacity == 0) {
      throw std::invalid_argument("all NTA runtime capacities must be nonzero");
    }

    cudaError_t flagsResult = cudaSetDeviceFlags(cudaDeviceMapHost);
    if (flagsResult != cudaSuccess &&
        flagsResult != cudaErrorSetOnActiveProcess) {
      checkCuda(flagsResult, "cudaSetDeviceFlags(cudaDeviceMapHost)");
    }

    int device = 0;
    checkCuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    checkCuda(cudaGetDeviceProperties(&properties, device),
              "cudaGetDeviceProperties");
    if (properties.canMapHostMemory == 0) {
      throw std::runtime_error(
          "selected CUDA device cannot map pinned host memory");
    }

    try {
      requests = deviceAllocate<abi::RequestContext>(config.requestCapacity);
      objectEntries = deviceAllocate<abi::ObjectEntry>(config.objectCapacity);
      intents = deviceAllocate<abi::AcquireIntent>(config.intentCapacity);
      continuations =
          deviceAllocate<abi::Continuation>(config.continuationCapacity);
      intentCount = deviceAllocate<std::uint32_t>(1);

      abi::RuntimeView hostView{
          requests,
          objectEntries,
          intents,
          continuations,
          intentCount,
          nvme == nullptr ? nullptr : nvme->deviceQueue(),
          config.requestCapacity,
          config.objectCapacity,
          config.intentCapacity,
          config.continuationCapacity,
          abi::Version,
          0,
      };
      view = deviceAllocate<abi::RuntimeView>(1);
      checkCuda(
          cudaMemcpy(view, &hostView, sizeof(hostView), cudaMemcpyHostToDevice),
          "upload RuntimeView");
    } catch (...) {
      release();
      throw;
    }
  }

  ~Impl() { release(); }

  void releaseObject(OwnedObject &object) noexcept {
    if (object.nvmeBuffer == nullptr && object.stagingDevice != nullptr) {
      (void)cudaFree(object.stagingDevice);
    }
    if (object.placement == Placement::Hbm && object.sourceDevice != nullptr) {
      (void)cudaFree(object.sourceDevice);
    }
    if (object.hostAllocation != nullptr) {
      (void)cudaFreeHost(object.hostAllocation);
    }
    object = {object.placement, nullptr, nullptr, nullptr, nullptr};
  }

  void release() noexcept {
    for (std::optional<OwnedObject> &object : objects) {
      if (object.has_value()) {
        releaseObject(*object);
        object.reset();
      }
    }
    if (view != nullptr) {
      (void)cudaFree(view);
      view = nullptr;
    }
    if (intentCount != nullptr) {
      (void)cudaFree(intentCount);
      intentCount = nullptr;
    }
    if (continuations != nullptr) {
      (void)cudaFree(continuations);
      continuations = nullptr;
    }
    if (intents != nullptr) {
      (void)cudaFree(intents);
      intents = nullptr;
    }
    if (objectEntries != nullptr) {
      (void)cudaFree(objectEntries);
      objectEntries = nullptr;
    }
    if (requests != nullptr) {
      (void)cudaFree(requests);
      requests = nullptr;
    }
  }

  void checkRequestSlot(std::uint32_t slot) const {
    if (slot >= config.requestCapacity) {
      throw std::out_of_range("request slot exceeds runtime capacity");
    }
  }

  void checkObjectSlot(std::uint32_t slot) const {
    if (slot >= config.objectCapacity) {
      throw std::out_of_range("object slot exceeds runtime capacity");
    }
  }

  RuntimeConfig config;
  abi::RequestContext *requests = nullptr;
  abi::ObjectEntry *objectEntries = nullptr;
  abi::AcquireIntent *intents = nullptr;
  abi::Continuation *continuations = nullptr;
  std::uint32_t *intentCount = nullptr;
  abi::RuntimeView *view = nullptr;
  std::vector<abi::RequestContext> requestsHost;
  std::vector<bool> requestInstalled;
  std::vector<std::optional<OwnedObject>> objects;
  std::shared_ptr<NvmeTransport> nvme;
};

HostRuntime::HostRuntime(RuntimeConfig config)
    : impl_(std::make_unique<Impl>(config)) {}

HostRuntime::HostRuntime(RuntimeConfig config,
                         std::shared_ptr<NvmeTransport> nvme)
    : impl_(std::make_unique<Impl>(config, std::move(nvme))) {}

HostRuntime::~HostRuntime() = default;
HostRuntime::HostRuntime(HostRuntime &&) noexcept = default;
HostRuntime &HostRuntime::operator=(HostRuntime &&) noexcept = default;

void HostRuntime::setRequest(std::uint32_t slot, std::uint64_t requestId,
                             std::uint32_t generation, std::uint32_t tenantId,
                             std::uint32_t priority,
                             std::uint64_t deadlineClock) {
  impl_->checkRequestSlot(slot);
  abi::RequestContext request{
      requestId, deadlineClock, generation, tenantId, priority, 0,
  };
  impl_->requestsHost[slot] = request;
  impl_->requestInstalled[slot] = true;
  uploadOne(impl_->requests, slot, request);
}

void HostRuntime::cancelRequest(std::uint32_t slot, std::uint32_t generation) {
  impl_->checkRequestSlot(slot);
  if (!impl_->requestInstalled[slot]) {
    throw std::invalid_argument("cannot cancel an uninitialized request slot");
  }
  abi::RequestContext &request = impl_->requestsHost[slot];
  if (request.generation != generation) {
    throw std::invalid_argument(
        "cannot cancel a reused request slot with a stale generation");
  }
  request.cancelled = 1;
  uploadOne(impl_->requests, slot, request);
}

ObjectHandle HostRuntime::installObject(std::uint32_t slot,
                                        std::uint64_t objectId,
                                        std::uint32_t version,
                                        std::span<const std::byte> contents,
                                        Placement placement) {
  impl_->checkObjectSlot(slot);
  if (contents.empty()) {
    throw std::invalid_argument(
        "external objects must contain at least one byte");
  }
  if (contents.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("objects are limited to 4 GiB");
  }

  Impl::OwnedObject allocation{placement, nullptr, nullptr, nullptr, nullptr};
  try {
    if (placement == Placement::Hbm) {
      checkCuda(cudaMalloc(&allocation.sourceDevice, contents.size()),
                "cudaMalloc HBM object");
      checkCuda(cudaMemcpy(allocation.sourceDevice, contents.data(),
                           contents.size(), cudaMemcpyHostToDevice),
                "upload HBM object");
    } else {
      checkCuda(cudaHostAlloc(&allocation.hostAllocation, contents.size(),
                              cudaHostAllocMapped),
                "cudaHostAlloc mapped object");
      std::memcpy(allocation.hostAllocation, contents.data(), contents.size());
      checkCuda(cudaHostGetDevicePointer(&allocation.sourceDevice,
                                         allocation.hostAllocation, 0),
                "cudaHostGetDevicePointer");
      if (placement == Placement::HostStaged) {
        checkCuda(cudaMalloc(&allocation.stagingDevice, contents.size()),
                  "cudaMalloc staging object");
      }
    }

    const bool direct = placement != Placement::HostStaged;
    abi::ObjectEntry entry{
        objectId,
        reinterpret_cast<std::uint64_t>(allocation.sourceDevice),
        reinterpret_cast<std::uint64_t>(allocation.stagingDevice),
        contents.size(),
        0,
        0,
        version,
        static_cast<std::uint32_t>(
            placement == Placement::Hbm          ? abi::SourceKind::Hbm
            : placement == Placement::HostMapped ? abi::SourceKind::HostMapped
                                                 : abi::SourceKind::HostStaged),
        static_cast<std::uint32_t>(direct ? abi::ObjectState::Ready
                                          : abi::ObjectState::New),
        0,
    };

    uploadOne(impl_->objectEntries, slot, entry);
    if (impl_->objects[slot].has_value()) {
      impl_->releaseObject(*impl_->objects[slot]);
    }
    void *const directAddress = direct ? allocation.sourceDevice : nullptr;
    impl_->objects[slot] = std::move(allocation);
    return {
        slot,
        directAddress,
    };
  } catch (...) {
    impl_->releaseObject(allocation);
    throw;
  }
}

ObjectHandle HostRuntime::installNvmeObject(
    std::uint32_t slot, std::uint64_t objectId, std::uint32_t version,
    std::uint64_t sourceByteOffset, std::size_t bytes,
    std::unique_ptr<NvmeBuffer> destination) {
  impl_->checkObjectSlot(slot);
  if (impl_->nvme == nullptr || destination == nullptr) {
    throw std::invalid_argument(
        "NVMe transport and destination buffer are required");
  }
  const NvmeCapabilities &capabilities = impl_->nvme->capabilities();
  if (bytes == 0 || bytes > destination->bytes() ||
      bytes % capabilities.lbaSize != 0 ||
      sourceByteOffset % capabilities.lbaSize != 0 ||
      sourceByteOffset > capabilities.namespaceBytes ||
      bytes > capabilities.namespaceBytes - sourceByteOffset) {
    throw std::invalid_argument("NVMe object range is invalid or unaligned");
  }

  Impl::OwnedObject allocation{
      Placement::HostStaged,  nullptr, nullptr, destination->deviceAddress(),
      std::move(destination),
  };
  abi::ObjectEntry entry{
      objectId,
      sourceByteOffset,
      reinterpret_cast<std::uint64_t>(allocation.stagingDevice),
      bytes,
      allocation.nvmeBuffer->dmaPageListAddress(),
      0,
      version,
      static_cast<std::uint32_t>(abi::SourceKind::Nvme),
      static_cast<std::uint32_t>(abi::ObjectState::New),
      allocation.nvmeBuffer->dmaPageCount(),
  };
  uploadOne(impl_->objectEntries, slot, entry);
  if (impl_->objects[slot].has_value()) {
    impl_->releaseObject(*impl_->objects[slot]);
  }
  impl_->objects[slot] = std::move(allocation);
  return {slot, nullptr};
}

abi::RuntimeView *HostRuntime::deviceView() const noexcept {
  return impl_->view;
}

const RuntimeConfig &HostRuntime::config() const noexcept {
  return impl_->config;
}

abi::RequestContext HostRuntime::readRequest(std::uint32_t slot) const {
  impl_->checkRequestSlot(slot);
  return downloadOne(impl_->requests, slot);
}

abi::ObjectEntry HostRuntime::readObject(std::uint32_t slot) const {
  impl_->checkObjectSlot(slot);
  return downloadOne(impl_->objectEntries, slot);
}

abi::Continuation HostRuntime::readContinuation(std::uint32_t slot) const {
  if (slot >= impl_->config.continuationCapacity) {
    throw std::out_of_range("continuation slot exceeds runtime capacity");
  }
  return downloadOne(impl_->continuations, slot);
}

} // namespace nta
