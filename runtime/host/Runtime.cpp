#include "nta/DeviceWorkPlan.h"
#include "nta/HostRuntime.h"
#include "nta/NvmeRuntime.h"
#include "nta/WorkPlan.h"

#include <cuda_runtime_api.h>

#include <array>
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
  struct OwnedReplica {
    Placement placement;
    void *hostAllocation;
    void *sourceDevice;
  };

  struct OwnedObject {
    void *stagingDevice;
    std::unique_ptr<NvmeBuffer> nvmeBuffer;
    std::vector<OwnedReplica> replicas;
  };

  explicit Impl(RuntimeConfig runtimeConfig,
                std::shared_ptr<NvmeTransport> nvmeTransport = nullptr)
      : config(runtimeConfig), requestsHost(config.requestCapacity),
        tenantsHost(config.requestCapacity),
        requestInstalled(config.requestCapacity, false),
        objectInstalled(config.objectCapacity, false),
        objects(config.objectCapacity), nvme(std::move(nvmeTransport)) {
    if (config.requestCapacity == 0 || config.objectCapacity == 0 ||
        config.intentCapacity == 0 || config.continuationCapacity == 0 ||
        config.intentCapacity < config.objectCapacity ||
        config.maxReplicasPerObject == 0 ||
        config.maxDependenciesPerContinuation == 0 ||
        config.objectCapacity > std::numeric_limits<std::uint32_t>::max() /
                                    config.maxReplicasPerObject ||
        config.continuationCapacity >
            std::numeric_limits<std::uint32_t>::max() /
                config.maxDependenciesPerContinuation) {
      throw std::invalid_argument("runtime capacities overflow or intent "
                                  "capacity is below object capacity");
    }
    replicaCapacity = config.objectCapacity * config.maxReplicasPerObject;
    dependencyCapacity =
        config.continuationCapacity * config.maxDependenciesPerContinuation;

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
      tenants = deviceAllocate<abi::TenantContext>(config.requestCapacity);
      objectEntries = deviceAllocate<abi::ObjectEntry>(config.objectCapacity);
      replicaEntries = deviceAllocate<abi::ReplicaEntry>(replicaCapacity);
      backendEntries = deviceAllocate<abi::BackendView>(abi::BackendCount);
      intents = deviceAllocate<abi::IntentSlot>(config.intentCapacity);
      continuations =
          deviceAllocate<abi::Continuation>(config.continuationCapacity);
      dependencies =
          deviceAllocate<abi::ContinuationDependency>(dependencyCapacity);
      intentPool = deviceAllocate<abi::IntentPool>(1);
      readyContinuations =
          deviceAllocate<std::uint32_t>(config.continuationCapacity);
      readyCount = deviceAllocate<std::uint32_t>(1);
      readyHead = deviceAllocate<std::uint32_t>(1);
      pendingContinuations =
          deviceAllocate<std::uint32_t>(config.continuationCapacity);
      pendingCount = deviceAllocate<std::uint32_t>(1);

      const auto backend = [](abi::SourceKind kind, bool active,
                              std::uint64_t state, std::uint64_t latencyNs,
                              std::uint64_t bandwidth) {
        return abi::BackendView{
            state,
            latencyNs,
            bandwidth,
            0,
            UINT64_MAX,
            static_cast<std::uint32_t>(kind),
            active ? 1U : 0U,
            static_cast<std::uint32_t>(kind),
            0,
            0,
        };
      };
      const std::array<abi::BackendView, abi::BackendCount> hostBackends{
          backend(abi::SourceKind::Hbm, true, 0, 0, 1'000'000'000'000ULL),
          backend(abi::SourceKind::HostMapped, true, 0, 300, 50'000'000'000ULL),
          backend(abi::SourceKind::HostStaged, true, 0, 2'000,
                  30'000'000'000ULL),
          backend(abi::SourceKind::Nvme, nvme != nullptr,
                  nvme == nullptr
                      ? 0
                      : reinterpret_cast<std::uint64_t>(nvme->deviceQueue()),
                  80'000, 7'000'000'000ULL),
          backend(abi::SourceKind::Rdma, false, 0, 5'000, 25'000'000'000ULL),
      };
      checkCuda(cudaMemcpy(backendEntries, hostBackends.data(),
                           sizeof(hostBackends), cudaMemcpyHostToDevice),
                "upload backend directory");
      for (abi::TenantContext &tenant : tenantsHost) {
        tenant = {UINT64_MAX, 0, 1, 1, 0};
      }
      checkCuda(cudaMemcpy(tenants, tenantsHost.data(),
                           tenantsHost.size() * sizeof(tenantsHost.front()),
                           cudaMemcpyHostToDevice),
                "upload tenant directory");
      std::vector<abi::IntentSlot> hostIntentSlots(config.intentCapacity);
      for (std::uint32_t slot = 0; slot < config.intentCapacity; ++slot) {
        hostIntentSlots[slot].sequence = slot;
      }
      checkCuda(
          cudaMemcpy(intents, hostIntentSlots.data(),
                     hostIntentSlots.size() * sizeof(hostIntentSlots.front()),
                     cudaMemcpyHostToDevice),
          "initialize intent ring slots");
      const abi::IntentPool hostIntentPool{
          0, 0, config.intentCapacity, 0, 0, 0, {0, 0, 0, 0},
      };
      checkCuda(cudaMemcpy(intentPool, &hostIntentPool, sizeof(hostIntentPool),
                           cudaMemcpyHostToDevice),
                "initialize intent pool");

      abi::RuntimeView hostView{
          requests,
          tenants,
          objectEntries,
          replicaEntries,
          backendEntries,
          intents,
          continuations,
          dependencies,
          intentPool,
          readyContinuations,
          readyCount,
          readyHead,
          pendingContinuations,
          pendingCount,
          config.requestCapacity,
          config.requestCapacity,
          config.objectCapacity,
          replicaCapacity,
          abi::BackendCount,
          config.intentCapacity,
          config.continuationCapacity,
          dependencyCapacity,
          config.maxDependenciesPerContinuation,
          abi::Version,
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
    for (OwnedReplica &replica : object.replicas) {
      if (replica.placement == Placement::Hbm &&
          replica.sourceDevice != nullptr) {
        (void)cudaFree(replica.sourceDevice);
      }
      if (replica.hostAllocation != nullptr) {
        (void)cudaFreeHost(replica.hostAllocation);
      }
    }
    object = {nullptr, nullptr, {}};
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
    if (intentPool != nullptr) {
      (void)cudaFree(intentPool);
      intentPool = nullptr;
    }
    if (readyHead != nullptr) {
      (void)cudaFree(readyHead);
      readyHead = nullptr;
    }
    if (pendingCount != nullptr) {
      (void)cudaFree(pendingCount);
      pendingCount = nullptr;
    }
    if (pendingContinuations != nullptr) {
      (void)cudaFree(pendingContinuations);
      pendingContinuations = nullptr;
    }
    if (readyCount != nullptr) {
      (void)cudaFree(readyCount);
      readyCount = nullptr;
    }
    if (readyContinuations != nullptr) {
      (void)cudaFree(readyContinuations);
      readyContinuations = nullptr;
    }
    if (continuations != nullptr) {
      (void)cudaFree(continuations);
      continuations = nullptr;
    }
    if (dependencies != nullptr) {
      (void)cudaFree(dependencies);
      dependencies = nullptr;
    }
    if (intents != nullptr) {
      (void)cudaFree(intents);
      intents = nullptr;
    }
    if (objectEntries != nullptr) {
      (void)cudaFree(objectEntries);
      objectEntries = nullptr;
    }
    if (backendEntries != nullptr) {
      (void)cudaFree(backendEntries);
      backendEntries = nullptr;
    }
    if (replicaEntries != nullptr) {
      (void)cudaFree(replicaEntries);
      replicaEntries = nullptr;
    }
    if (requests != nullptr) {
      (void)cudaFree(requests);
      requests = nullptr;
    }
    if (tenants != nullptr) {
      (void)cudaFree(tenants);
      tenants = nullptr;
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
  std::uint32_t replicaCapacity = 0;
  std::uint32_t dependencyCapacity = 0;
  abi::RequestContext *requests = nullptr;
  abi::TenantContext *tenants = nullptr;
  abi::ObjectEntry *objectEntries = nullptr;
  abi::ReplicaEntry *replicaEntries = nullptr;
  abi::BackendView *backendEntries = nullptr;
  abi::IntentSlot *intents = nullptr;
  abi::Continuation *continuations = nullptr;
  abi::ContinuationDependency *dependencies = nullptr;
  abi::IntentPool *intentPool = nullptr;
  std::uint32_t *readyContinuations = nullptr;
  std::uint32_t *readyCount = nullptr;
  std::uint32_t *readyHead = nullptr;
  std::uint32_t *pendingContinuations = nullptr;
  std::uint32_t *pendingCount = nullptr;
  abi::RuntimeView *view = nullptr;
  std::vector<abi::RequestContext> requestsHost;
  std::vector<abi::TenantContext> tenantsHost;
  std::vector<bool> requestInstalled;
  std::vector<bool> objectInstalled;
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
                             std::uint64_t deadlineClock,
                             std::uint64_t maxOutstandingBytes) {
  impl_->checkRequestSlot(slot);
  if (tenantId >= impl_->config.requestCapacity) {
    throw std::out_of_range("tenant id exceeds runtime capacity");
  }
  if (impl_->requestInstalled[slot]) {
    const abi::RequestContext current = downloadOne(impl_->requests, slot);
    if (current.outstandingBytes != 0) {
      throw std::logic_error("request slot cannot be reused while acquisition "
                             "bytes are outstanding");
    }
  }
  abi::RequestContext request{
      requestId,
      deadlineClock,
      maxOutstandingBytes,
      0,
      generation,
      tenantId,
      priority,
      0,
  };
  impl_->requestsHost[slot] = request;
  impl_->requestInstalled[slot] = true;
  uploadOne(impl_->requests, slot, request);
}

void HostRuntime::setTenantBudget(std::uint32_t tenantId,
                                  std::uint64_t maxOutstandingBytes,
                                  std::uint32_t weight) {
  if (tenantId >= impl_->config.requestCapacity || weight == 0) {
    throw std::invalid_argument("tenant budget id and weight must be valid");
  }
  abi::TenantContext tenant = downloadOne(impl_->tenants, tenantId);
  if (tenant.outstandingBytes > maxOutstandingBytes) {
    throw std::invalid_argument(
        "tenant budget cannot drop below currently outstanding bytes");
  }
  tenant.maxOutstandingBytes = maxOutstandingBytes;
  tenant.weight = weight;
  tenant.active = 1;
  impl_->tenantsHost[tenantId] = tenant;
  uploadOne(impl_->tenants, tenantId, tenant);
}

void HostRuntime::cancelRequest(std::uint32_t slot, std::uint32_t generation) {
  impl_->checkRequestSlot(slot);
  if (!impl_->requestInstalled[slot]) {
    throw std::invalid_argument("cannot cancel an uninitialized request slot");
  }
  abi::RequestContext request = downloadOne(impl_->requests, slot);
  if (request.generation != generation) {
    throw std::invalid_argument(
        "cannot cancel a reused request slot with a stale generation");
  }
  request.cancelled = 1;
  impl_->requestsHost[slot] = request;
  uploadOne(impl_->requests, slot, request);
}

ObjectHandle HostRuntime::installObject(std::uint32_t slot,
                                        std::uint64_t objectId,
                                        std::uint32_t version,
                                        std::span<const std::byte> contents,
                                        Placement placement) {
  const HostReplicaSpec replica{contents, placement};
  return installReplicatedObject(slot, objectId, version,
                                 std::span<const HostReplicaSpec>(&replica, 1));
}

ObjectHandle HostRuntime::installReplicatedObject(
    std::uint32_t slot, std::uint64_t objectId, std::uint32_t version,
    std::span<const HostReplicaSpec> replicas) {
  impl_->checkObjectSlot(slot);
  if (replicas.empty() ||
      replicas.size() > impl_->config.maxReplicasPerObject) {
    throw std::invalid_argument(
        "replica count must fit the configured per-object capacity");
  }
  const std::size_t bytes = replicas.front().contents.size();
  if (bytes == 0) {
    throw std::invalid_argument(
        "external objects must contain at least one byte");
  }
  if (bytes > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("objects are limited to 4 GiB");
  }
  for (const HostReplicaSpec &replica : replicas) {
    if (replica.contents.size() != bytes) {
      throw std::invalid_argument(
          "all physical replicas must have the same object size");
    }
  }

  Impl::OwnedObject allocation{nullptr, nullptr, {}};
  allocation.replicas.reserve(replicas.size());
  std::vector<abi::ReplicaEntry> replicaEntries;
  replicaEntries.reserve(replicas.size());
  void *directAddress = nullptr;
  std::uint64_t directCost = UINT64_MAX;
  bool hasTransport = false;
  try {
    for (const HostReplicaSpec &spec : replicas) {
      allocation.replicas.push_back({spec.placement, nullptr, nullptr});
      Impl::OwnedReplica &owned = allocation.replicas.back();
      if (spec.placement == Placement::Hbm) {
        checkCuda(cudaMalloc(&owned.sourceDevice, bytes),
                  "cudaMalloc HBM object replica");
        checkCuda(cudaMemcpy(owned.sourceDevice, spec.contents.data(), bytes,
                             cudaMemcpyHostToDevice),
                  "upload HBM object replica");
      } else {
        checkCuda(
            cudaHostAlloc(&owned.hostAllocation, bytes, cudaHostAllocMapped),
            "cudaHostAlloc mapped object replica");
        std::memcpy(owned.hostAllocation, spec.contents.data(), bytes);
        checkCuda(cudaHostGetDevicePointer(&owned.sourceDevice,
                                           owned.hostAllocation, 0),
                  "cudaHostGetDevicePointer object replica");
      }

      const bool direct = spec.placement != Placement::HostStaged;
      hasTransport |= !direct;
      const abi::SourceKind sourceKind =
          spec.placement == Placement::Hbm ? abi::SourceKind::Hbm
          : spec.placement == Placement::HostMapped
              ? abi::SourceKind::HostMapped
              : abi::SourceKind::HostStaged;
      replicaEntries.push_back({
          reinterpret_cast<std::uint64_t>(owned.sourceDevice),
          0,
          sourceKind == abi::SourceKind::Hbm          ? 0ULL
          : sourceKind == abi::SourceKind::HostMapped ? 300ULL
                                                      : 2'000ULL,
          sourceKind == abi::SourceKind::Hbm          ? 1'000'000'000'000ULL
          : sourceKind == abi::SourceKind::HostMapped ? 50'000'000'000ULL
                                                      : 30'000'000'000ULL,
          static_cast<std::uint32_t>(sourceKind),
          0,
          static_cast<std::uint32_t>(sourceKind),
          static_cast<std::uint32_t>(direct ? abi::ReplicaDirect
                                            : abi::ReplicaTransport),
          0,
          0,
      });
      const std::uint64_t candidateCost =
          replicaEntries.back().estimatedLatencyNs +
          bytes * 1'000'000'000ULL /
              replicaEntries.back().estimatedBandwidthBytesPerSecond;
      if (direct && candidateCost < directCost) {
        directAddress = owned.sourceDevice;
        directCost = candidateCost;
      }
    }
    if (hasTransport) {
      checkCuda(cudaMalloc(&allocation.stagingDevice, bytes),
                "cudaMalloc object staging destination");
    }

    const std::uint32_t replicaStart =
        slot * impl_->config.maxReplicasPerObject;
    abi::ObjectEntry entry{
        objectId,
        reinterpret_cast<std::uint64_t>(allocation.stagingDevice),
        bytes,
        0,
        version,
        static_cast<std::uint32_t>(directAddress != nullptr
                                       ? abi::ObjectState::Ready
                                       : abi::ObjectState::New),
        replicaStart,
        static_cast<std::uint32_t>(replicaEntries.size()),
        abi::InvalidIndex,
        0,
        0,
    };

    checkCuda(cudaMemcpy(impl_->replicaEntries + replicaStart,
                         replicaEntries.data(),
                         replicaEntries.size() * sizeof(replicaEntries.front()),
                         cudaMemcpyHostToDevice),
              "upload object replica directory");
    uploadOne(impl_->objectEntries, slot, entry);
    if (impl_->objects[slot].has_value()) {
      impl_->releaseObject(*impl_->objects[slot]);
    }
    impl_->objects[slot] = std::move(allocation);
    impl_->objectInstalled[slot] = true;
    return {slot, directAddress};
  } catch (...) {
    impl_->releaseObject(allocation);
    throw;
  }
}

ObjectHandle
HostRuntime::registerObject(std::uint32_t slot, std::uint64_t objectId,
                            std::uint32_t version, std::size_t bytes,
                            void *stagingDeviceAddress,
                            std::span<const RegisteredReplicaSpec> replicas) {
  impl_->checkObjectSlot(slot);
  if (bytes == 0 || bytes > std::numeric_limits<std::uint32_t>::max() ||
      replicas.empty() ||
      replicas.size() > impl_->config.maxReplicasPerObject) {
    throw std::invalid_argument(
        "registered object size and replicas must fit the runtime ABI");
  }

  std::vector<abi::ReplicaEntry> entries;
  entries.reserve(replicas.size());
  void *directAddress = nullptr;
  std::uint64_t directCost = UINT64_MAX;
  bool hasTransport = false;
  for (const RegisteredReplicaSpec &replica : replicas) {
    if (replica.sourceDeviceAddress == nullptr) {
      throw std::invalid_argument(
          "registered replica needs a device-visible source address");
    }
    const bool direct = replica.placement != Placement::HostStaged;
    hasTransport |= !direct;
    const abi::SourceKind sourceKind =
        replica.placement == Placement::Hbm ? abi::SourceKind::Hbm
        : replica.placement == Placement::HostMapped
            ? abi::SourceKind::HostMapped
            : abi::SourceKind::HostStaged;
    const std::uint64_t defaultLatency =
        sourceKind == abi::SourceKind::Hbm          ? 0
        : sourceKind == abi::SourceKind::HostMapped ? 300
                                                    : 2'000;
    const std::uint64_t defaultBandwidth =
        sourceKind == abi::SourceKind::Hbm          ? 1'000'000'000'000ULL
        : sourceKind == abi::SourceKind::HostMapped ? 50'000'000'000ULL
                                                    : 30'000'000'000ULL;
    const std::uint64_t latency = replica.estimatedLatencyNs == 0
                                      ? defaultLatency
                                      : replica.estimatedLatencyNs;
    const std::uint64_t bandwidth =
        replica.estimatedBandwidthBytesPerSecond == 0
            ? defaultBandwidth
            : replica.estimatedBandwidthBytesPerSecond;
    entries.push_back({
        reinterpret_cast<std::uint64_t>(replica.sourceDeviceAddress),
        0,
        latency,
        bandwidth,
        static_cast<std::uint32_t>(sourceKind),
        0,
        static_cast<std::uint32_t>(sourceKind),
        direct ? abi::ReplicaDirect : abi::ReplicaTransport,
        reinterpret_cast<std::uint64_t>(replica.tensorMap),
        0,
    });
    const std::uint64_t cost = latency + static_cast<std::uint64_t>(bytes) *
                                             1'000'000'000ULL / bandwidth;
    if (direct && cost < directCost) {
      directAddress = const_cast<void *>(replica.sourceDeviceAddress);
      directCost = cost;
    }
  }
  if (hasTransport && stagingDeviceAddress == nullptr) {
    throw std::invalid_argument(
        "registered staged replicas need an HBM staging address");
  }

  const std::uint32_t replicaStart = slot * impl_->config.maxReplicasPerObject;
  const abi::ObjectEntry entry{
      objectId,
      reinterpret_cast<std::uint64_t>(stagingDeviceAddress),
      static_cast<std::uint64_t>(bytes),
      0,
      version,
      static_cast<std::uint32_t>(directAddress == nullptr
                                     ? abi::ObjectState::New
                                     : abi::ObjectState::Ready),
      replicaStart,
      static_cast<std::uint32_t>(entries.size()),
      abi::InvalidIndex,
      0,
      0,
  };
  checkCuda(cudaMemcpy(impl_->replicaEntries + replicaStart, entries.data(),
                       entries.size() * sizeof(entries.front()),
                       cudaMemcpyHostToDevice),
            "upload registered object replicas");
  uploadOne(impl_->objectEntries, slot, entry);
  if (impl_->objects[slot].has_value()) {
    impl_->releaseObject(*impl_->objects[slot]);
    impl_->objects[slot].reset();
  }
  impl_->objectInstalled[slot] = true;
  return {slot, directAddress};
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
      destination->deviceAddress(), std::move(destination), {}};
  const abi::ReplicaEntry replica{
      sourceByteOffset,
      allocation.nvmeBuffer->dmaPageListAddress(),
      80'000,
      7'000'000'000ULL,
      static_cast<std::uint32_t>(abi::SourceKind::Nvme),
      allocation.nvmeBuffer->dmaPageCount(),
      static_cast<std::uint32_t>(abi::SourceKind::Nvme),
      abi::ReplicaTransport,
      0,
      0,
  };
  const std::uint32_t replicaStart = slot * impl_->config.maxReplicasPerObject;
  abi::ObjectEntry entry{
      objectId,
      reinterpret_cast<std::uint64_t>(allocation.stagingDevice),
      bytes,
      0,
      version,
      static_cast<std::uint32_t>(abi::ObjectState::New),
      replicaStart,
      1,
      abi::InvalidIndex,
      0,
      0,
  };
  uploadOne(impl_->replicaEntries, replicaStart, replica);
  uploadOne(impl_->objectEntries, slot, entry);
  if (impl_->objects[slot].has_value()) {
    impl_->releaseObject(*impl_->objects[slot]);
  }
  impl_->objects[slot] = std::move(allocation);
  impl_->objectInstalled[slot] = true;
  return {slot, nullptr};
}

void HostRuntime::bindTensorMaps(std::uint32_t objectSlot,
                                 std::uint32_t relativeReplica,
                                 const void *replicaTensorMap,
                                 const void *stagingTensorMap) {
  impl_->checkObjectSlot(objectSlot);
  if (!impl_->objectInstalled[objectSlot]) {
    throw std::invalid_argument(
        "cannot bind tensor maps before object install");
  }

  abi::ObjectEntry object = downloadOne(impl_->objectEntries, objectSlot);
  if (relativeReplica >= object.replicaCount ||
      object.replicaStart > impl_->replicaCapacity ||
      relativeReplica >= impl_->replicaCapacity - object.replicaStart) {
    throw std::out_of_range("relative replica exceeds object replica range");
  }
  abi::ReplicaEntry replica =
      downloadOne(impl_->replicaEntries, object.replicaStart + relativeReplica);
  replica.tensorMapAddress = reinterpret_cast<std::uint64_t>(replicaTensorMap);
  object.stagingTensorMapAddress =
      reinterpret_cast<std::uint64_t>(stagingTensorMap);
  uploadOne(impl_->replicaEntries, object.replicaStart + relativeReplica,
            replica);
  uploadOne(impl_->objectEntries, objectSlot, object);
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

abi::TenantContext HostRuntime::readTenant(std::uint32_t tenantId) const {
  if (tenantId >= impl_->config.requestCapacity) {
    throw std::out_of_range("tenant id exceeds runtime capacity");
  }
  return downloadOne(impl_->tenants, tenantId);
}

abi::ObjectEntry HostRuntime::readObject(std::uint32_t slot) const {
  impl_->checkObjectSlot(slot);
  return downloadOne(impl_->objectEntries, slot);
}

abi::ReplicaEntry
HostRuntime::readReplica(std::uint32_t objectSlot,
                         std::uint32_t relativeReplica) const {
  impl_->checkObjectSlot(objectSlot);
  const abi::ObjectEntry object = readObject(objectSlot);
  if (relativeReplica >= object.replicaCount ||
      object.replicaStart > impl_->replicaCapacity ||
      relativeReplica >= impl_->replicaCapacity - object.replicaStart) {
    throw std::out_of_range("relative replica exceeds object replica range");
  }
  return downloadOne(impl_->replicaEntries,
                     object.replicaStart + relativeReplica);
}

abi::Continuation HostRuntime::readContinuation(std::uint32_t slot) const {
  if (slot >= impl_->config.continuationCapacity) {
    throw std::out_of_range("continuation slot exceeds runtime capacity");
  }
  return downloadOne(impl_->continuations, slot);
}

abi::ContinuationDependency HostRuntime::readContinuationDependency(
    std::uint32_t continuation, std::uint32_t relativeDependency) const {
  if (continuation >= impl_->config.continuationCapacity ||
      relativeDependency >= impl_->config.maxDependenciesPerContinuation) {
    throw std::out_of_range("continuation dependency exceeds runtime capacity");
  }
  const std::uint32_t index =
      continuation * impl_->config.maxDependenciesPerContinuation +
      relativeDependency;
  return downloadOne(impl_->dependencies, index);
}

abi::IntentPool HostRuntime::readIntentPool() const {
  return downloadOne(impl_->intentPool, 0);
}

std::uint32_t HostRuntime::readPendingCount() const {
  return downloadOne(impl_->pendingCount, 0);
}

DeviceWorkPlan HostRuntime::uploadWorkPlan(const WorkPlan &plan) const {
  if (plan.workItems.size() > impl_->config.continuationCapacity) {
    throw std::invalid_argument(
        "work plan exceeds the runtime continuation capacity");
  }
  for (const abi::WorkItem &work : plan.workItems) {
    if (work.requestSlot >= impl_->config.requestCapacity ||
        !impl_->requestInstalled[work.requestSlot] ||
        work.dependencyCount > impl_->config.maxDependenciesPerContinuation) {
      throw std::invalid_argument(
          "work plan does not fit the runtime request/dependency contract");
    }
  }
  for (const abi::AcquireRequirement &requirement : plan.dependencies) {
    if (requirement.directBase == 0 &&
        (requirement.objectSlot >= impl_->config.objectCapacity ||
         !impl_->objectInstalled[requirement.objectSlot])) {
      throw std::invalid_argument(
          "work plan references an unregistered external object");
    }
  }
  return DeviceWorkPlan(plan);
}

} // namespace nta
