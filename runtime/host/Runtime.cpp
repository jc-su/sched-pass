#include "CudaDeviceGuard.h"
#include "nta/DeviceWorkPlan.h"
#include "nta/CxlRuntime.h"
#include "nta/HostRuntime.h"
#include "nta/NvmeRuntime.h"
#include "nta/WorkPlan.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nta {
namespace {

RuntimeConfig normalizeRuntimeConfig(RuntimeConfig config) {
  if (config.tenantCapacity == 0) {
    config.tenantCapacity = config.requestCapacity;
  }
  if (config.stagingByteCapacity == 0) {
    config.stagingByteCapacity = UINT64_MAX;
  }
  return config;
}

void checkCuda(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

void validateDeviceVisiblePointer(const void *address, int deviceOrdinal,
                                  const char *description) {
  if (address == nullptr) {
    throw std::invalid_argument(std::string(description) + " is null");
  }
  cudaPointerAttributes attributes{};
  const cudaError_t status = cudaPointerGetAttributes(&attributes, address);
  if (status != cudaSuccess) {
    (void)cudaGetLastError();
    throw std::invalid_argument(std::string(description) +
                                " is not a live CUDA allocation");
  }
  const bool deviceAllocation =
      attributes.type == cudaMemoryTypeDevice ||
      attributes.type == cudaMemoryTypeManaged;
  const bool mappedHostAllocation =
      attributes.type == cudaMemoryTypeHost &&
      attributes.devicePointer != nullptr;
  if ((!deviceAllocation && !mappedHostAllocation) ||
      attributes.device != deviceOrdinal) {
    throw std::invalid_argument(std::string(description) +
                                " is not visible on the runtime CUDA device");
  }
}

void validateDeviceAllocation(const void *address, int deviceOrdinal,
                              const char *description) {
  if (address == nullptr) {
    throw std::invalid_argument(std::string(description) + " is null");
  }
  cudaPointerAttributes attributes{};
  const cudaError_t status = cudaPointerGetAttributes(&attributes, address);
  if (status != cudaSuccess) {
    (void)cudaGetLastError();
    throw std::invalid_argument(std::string(description) +
                                " is not a live CUDA allocation");
  }
  if ((attributes.type != cudaMemoryTypeDevice &&
       attributes.type != cudaMemoryTypeManaged) ||
      attributes.device != deviceOrdinal) {
    throw std::invalid_argument(std::string(description) +
                                " must be HBM allocated on the runtime CUDA device");
  }
}

void validateMappedHostPointer(const void *address, int deviceOrdinal,
                               const char *description) {
  if (address == nullptr) {
    throw std::invalid_argument(std::string(description) + " is null");
  }
  cudaPointerAttributes attributes{};
  const cudaError_t status = cudaPointerGetAttributes(&attributes, address);
  if (status != cudaSuccess) {
    (void)cudaGetLastError();
    throw std::invalid_argument(std::string(description) +
                                " is not a live CUDA host mapping");
  }
  if (attributes.type != cudaMemoryTypeHost ||
      attributes.device != deviceOrdinal || attributes.devicePointer == nullptr) {
    throw std::invalid_argument(std::string(description) +
                                " must be a device-visible mapped host allocation");
  }
}

void validateReplicaAddress(const RegisteredReplicaSpec &replica,
                            int deviceOrdinal) {
  switch (replica.placement) {
  case Placement::Hbm:
    validateDeviceAllocation(replica.sourceDeviceAddress, deviceOrdinal,
                             "registered HBM replica source");
    return;
  case Placement::HostMapped:
    validateMappedHostPointer(replica.sourceDeviceAddress, deviceOrdinal,
                              "registered host-mapped replica source");
    return;
  case Placement::HostStaged:
  case Placement::CxlMapped:
    validateDeviceVisiblePointer(replica.sourceDeviceAddress, deviceOrdinal,
                                 "registered replica source");
    return;
  }
  throw std::invalid_argument("unknown registered replica placement");
}

abi::SourceKind sourceKindFor(Placement placement) {
  switch (placement) {
  case Placement::Hbm:
    return abi::SourceKind::Hbm;
  case Placement::HostMapped:
    return abi::SourceKind::HostMapped;
  case Placement::HostStaged:
    return abi::SourceKind::HostStaged;
  case Placement::CxlMapped:
    return abi::SourceKind::Cxl;
  }
  throw std::invalid_argument("unknown runtime placement");
}

std::uint64_t defaultLatency(abi::SourceKind kind) {
  switch (kind) {
  case abi::SourceKind::Hbm:
    return 0;
  case abi::SourceKind::HostMapped:
    return 300;
  case abi::SourceKind::HostStaged:
    return 2'000;
  case abi::SourceKind::Nvme:
    return 80'000;
  case abi::SourceKind::Cxl:
    return 1'500;
  case abi::SourceKind::Rdma:
    return 5'000;
  }
  return UINT64_MAX;
}

std::uint64_t defaultBandwidth(abi::SourceKind kind) {
  switch (kind) {
  case abi::SourceKind::Hbm:
    return 1'000'000'000'000ULL;
  case abi::SourceKind::HostMapped:
    return 50'000'000'000ULL;
  case abi::SourceKind::HostStaged:
    return 30'000'000'000ULL;
  case abi::SourceKind::Nvme:
    return 7'000'000'000ULL;
  case abi::SourceKind::Cxl:
    return 20'000'000'000ULL;
  case abi::SourceKind::Rdma:
    return 25'000'000'000ULL;
  }
  return 1;
}

const char *tierEnvironmentName(abi::SourceKind kind) {
  switch (kind) {
  case abi::SourceKind::Hbm:
    return "HBM";
  case abi::SourceKind::HostMapped:
    return "HOST_MAPPED";
  case abi::SourceKind::HostStaged:
    return "HOST_STAGED";
  case abi::SourceKind::Nvme:
    return "NVME";
  case abi::SourceKind::Cxl:
    return "CXL";
  case abi::SourceKind::Rdma:
    return "RDMA";
  }
  return "UNKNOWN";
}

std::uint64_t configuredTierValue(abi::SourceKind kind, const char *suffix,
                                  std::uint64_t fallback,
                                  bool allowZero) {
  std::string name = "NTA_TIER_";
  name += tierEnvironmentName(kind);
  name += suffix;
  const char *raw = std::getenv(name.c_str());
  if (raw == nullptr || *raw == '\0') {
    return fallback;
  }
  if (*raw == '-') {
    throw std::invalid_argument(name + " must be an unsigned integer");
  }
  errno = 0;
  char *end = nullptr;
  const unsigned long long value = std::strtoull(raw, &end, 10);
  if (errno == ERANGE || end == raw || end == nullptr || *end != '\0' ||
      (!allowZero && value == 0)) {
    throw std::invalid_argument(name + " is invalid");
  }
  return static_cast<std::uint64_t>(value);
}

std::uint64_t configuredTierLatency(abi::SourceKind kind) {
  return configuredTierValue(kind, "_LATENCY_NS", defaultLatency(kind), true);
}

std::uint64_t configuredTierBandwidth(abi::SourceKind kind) {
  return configuredTierValue(kind, "_BANDWIDTH_BPS", defaultBandwidth(kind),
                             false);
}

void validateIndexedHostObject(const IndexedHostObjectSpec &object,
                               int deviceOrdinal) {
  if (object.sourceDeviceAddress == nullptr ||
      object.stagingDeviceAddress == nullptr ||
      object.sourceIndicesDevice == nullptr ||
      object.stagingIndicesDevice == nullptr || object.indexCount == 0 ||
      object.elementBytes == 0 ||
      object.sourceStrideBytes < object.elementBytes ||
      object.stagingStrideBytes < object.elementBytes ||
      object.sourceIndexLimit == 0 || object.stagingIndexLimit == 0 ||
      object.indexCount > std::numeric_limits<std::uint32_t>::max() /
                              object.elementBytes) {
    throw std::invalid_argument("invalid indexed host transfer geometry");
  }
  validateDeviceVisiblePointer(object.sourceDeviceAddress, deviceOrdinal,
                               "indexed source");
  validateDeviceAllocation(object.stagingDeviceAddress, deviceOrdinal,
                           "indexed staging destination");
  validateDeviceAllocation(object.sourceIndicesDevice, deviceOrdinal,
                           "indexed source indices");
  validateDeviceAllocation(object.stagingIndicesDevice, deviceOrdinal,
                           "indexed staging indices");
}

abi::ReplicaEntry makeIndexedHostReplica(const IndexedHostObjectSpec &object,
                                         std::uint64_t latencyNs,
                                         std::uint64_t bandwidthBytes) {
  return {
      reinterpret_cast<std::uint64_t>(object.sourceDeviceAddress),
      reinterpret_cast<std::uint64_t>(object.sourceIndicesDevice),
      latencyNs,
      bandwidthBytes,
      static_cast<std::uint32_t>(abi::SourceKind::HostStaged),
      object.indexCount,
      static_cast<std::uint32_t>(abi::SourceKind::HostStaged),
      abi::ReplicaTransport | abi::ReplicaIndexed,
      abi::packTransferIndexLimits(object.sourceIndexLimit,
                                   object.stagingIndexLimit),
      abi::packTransferStrides(object.sourceStrideBytes,
                               object.stagingStrideBytes),
  };
}

abi::ObjectEntry makeIndexedHostObject(const IndexedHostObjectSpec &object,
                                       std::uint32_t replicaStart) {
  return {
      object.objectId,
      reinterpret_cast<std::uint64_t>(object.stagingDeviceAddress),
      static_cast<std::uint64_t>(object.indexCount) * object.elementBytes,
      0,
      object.version,
      static_cast<std::uint32_t>(object.preacquired
                                     ? abi::ObjectState::Ready
                                     : abi::ObjectState::New),
      replicaStart,
      1,
      object.preacquired ? 0U : abi::InvalidIndex,
      // For indexed objects, flags carries the registered index-array
      // capacity so a device-side row-count update can validate against it.
      object.indexCount,
      reinterpret_cast<std::uint64_t>(object.stagingIndicesDevice),
  };
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
  // Ring recycling waits on the entry's completion event, which stalls the
  // caller (the engine scheduler thread) whenever the ring is shallower than
  // the in-flight claim/plan bursts. The default preserves historical
  // behavior; the override exists for the pre-declared H-C interference
  // discrimination and for engines with deeper burst patterns.
  static std::size_t directoryUploadDepth() {
    const char *configured = std::getenv("NTA_DIRECTORY_UPLOAD_DEPTH");
    if (configured == nullptr || *configured == '\0') {
      // Depth 4 produced the confirmed H-C pathology: recycling syncs on the
      // engine scheduler thread stretched consecutive decode intervals 2-3x
      // whenever claim bursts collided with decode. Depth 32 removes the
      // observed collisions at a bounded pinned-staging cost of
      // (objectCapacity + replicaCapacity) * 64 bytes per slot.
      return 32;
    }
    char *end = nullptr;
    const long value = std::strtol(configured, &end, 10);
    if (end == nullptr || *end != '\0' || value < 1 || value > 4096) {
      throw std::invalid_argument(
          "NTA_DIRECTORY_UPLOAD_DEPTH must be an integer in [1, 4096]");
    }
    return static_cast<std::size_t>(value);
  }
  static constexpr std::size_t RequestUploadDepth = 4;

  struct OwnedReplica {
    Placement placement;
    void *hostAllocation;
    void *sourceDevice;
    std::unique_ptr<CxlDaxBuffer> cxlBuffer;
  };

  struct OwnedObject {
    void *stagingDevice;
    std::unique_ptr<NvmeBuffer> nvmeBuffer;
    std::vector<OwnedReplica> replicas;
    std::uint64_t accountedStagingBytes = 0;
  };

  struct RetiredObject {
    OwnedObject object;
    cudaEvent_t complete = nullptr;
  };

  struct DirectoryUpload {
    abi::ObjectEntry *objects = nullptr;
    abi::ReplicaEntry *replicas = nullptr;
    cudaEvent_t complete = nullptr;
    bool pending = false;
  };

  struct RequestUpload {
    abi::RequestContext *requests = nullptr;
    abi::RequestProgress *progress = nullptr;
    cudaEvent_t complete = nullptr;
    bool pending = false;
  };

  explicit Impl(RuntimeConfig runtimeConfig,
                RuntimeBackends runtimeBackends = {})
      : config(normalizeRuntimeConfig(runtimeConfig)),
        requestsHost(config.requestCapacity), tenantsHost(config.tenantCapacity),
        requestInstalled(config.requestCapacity, false),
        objectInstalled(config.objectCapacity, false),
        objects(config.objectCapacity),
        nvme(std::move(runtimeBackends.nvme)),
        cxl(std::move(runtimeBackends.cxl)) {
    config.deviceOrdinal = detail::resolveCudaDevice(config.deviceOrdinal);
    detail::CudaDeviceGuard deviceGuard(config.deviceOrdinal);
    if (nvme != nullptr && nvme->deviceOrdinal() != config.deviceOrdinal) {
      throw std::invalid_argument(
          "HostRuntime and NvmeTransport must own the same CUDA device");
    }
    if (cxl != nullptr && cxl->deviceOrdinal() != config.deviceOrdinal) {
      throw std::invalid_argument(
          "HostRuntime and CxlDaxTransport must own the same CUDA device");
    }
    if (config.requestCapacity == 0 || config.tenantCapacity == 0 ||
        config.objectCapacity == 0 ||
        config.intentCapacity == 0 || config.workTicketCapacity == 0 ||
        config.maxReplicasPerObject == 0 ||
        config.maxDependenciesPerWorkTicket == 0 ||
        config.objectCapacity > std::numeric_limits<std::uint32_t>::max() /
                                    config.maxReplicasPerObject ||
        config.workTicketCapacity >
            std::numeric_limits<std::uint32_t>::max() /
                config.maxDependenciesPerWorkTicket) {
      throw std::invalid_argument("runtime capacities must be finite, non-zero, "
                                  "and must not overflow");
    }
    replicaCapacity = config.objectCapacity * config.maxReplicasPerObject;
    dependencyCapacity =
        config.workTicketCapacity * config.maxDependenciesPerWorkTicket;
    for (std::uint32_t index = 0; index < abi::BackendCount; ++index) {
      const auto kind = static_cast<abi::SourceKind>(index);
      tierLatencies[index] = configuredTierLatency(kind);
      tierBandwidths[index] = configuredTierBandwidth(kind);
    }

    cudaError_t flagsResult = cudaSetDeviceFlags(cudaDeviceMapHost);
    if (flagsResult != cudaSuccess &&
        flagsResult != cudaErrorSetOnActiveProcess) {
      checkCuda(flagsResult, "cudaSetDeviceFlags(cudaDeviceMapHost)");
    }

    cudaDeviceProp properties{};
    checkCuda(cudaGetDeviceProperties(&properties, config.deviceOrdinal),
              "cudaGetDeviceProperties");
    if (properties.canMapHostMemory == 0) {
      throw std::runtime_error(
          "selected CUDA device cannot map pinned host memory");
    }

    try {
      requests = deviceAllocate<abi::RequestContext>(config.requestCapacity);
      tenants = deviceAllocate<abi::TenantContext>(config.tenantCapacity);
      objectEntries = deviceAllocate<abi::ObjectEntry>(config.objectCapacity);
      replicaEntries = deviceAllocate<abi::ReplicaEntry>(replicaCapacity);
      backendEntries = deviceAllocate<abi::BackendView>(abi::BackendCount);
      intents = deviceAllocate<abi::IntentSlot>(config.intentCapacity);
      workTickets =
          deviceAllocate<abi::WorkTicket>(config.workTicketCapacity);
      workRunnableNs =
          deviceAllocate<std::uint64_t>(config.workTicketCapacity);
      checkCuda(cudaMemset(workRunnableNs, 0,
                           config.workTicketCapacity * sizeof(std::uint64_t)),
                "initialize work runnable timestamps");
      dependencies =
          deviceAllocate<abi::WorkDependency>(dependencyCapacity);
      intentPool = deviceAllocate<abi::IntentPool>(1);
      intentQueueEntries =
          deviceAllocate<abi::IntentQueueEntry>(config.intentCapacity);
      intentQueueHeads = deviceAllocate<std::uint64_t>(
          abi::BackendCount * abi::UrgencyBucketCount);
      checkCuda(cudaMemset(intentQueueHeads, 0xff,
                           abi::BackendCount * abi::UrgencyBucketCount *
                               sizeof(std::uint64_t)),
                "initialize intent queue heads");
      readyWorkTickets =
          deviceAllocate<std::uint32_t>(config.workTicketCapacity);
      readyCount = deviceAllocate<std::uint32_t>(1);
      readyHead = deviceAllocate<std::uint32_t>(1);
      pendingWorkTickets =
          deviceAllocate<std::uint32_t>(config.workTicketCapacity);
      pendingCount = deviceAllocate<std::uint32_t>(1);
      ctaCompletions =
          deviceAllocate<std::uint32_t>(config.workTicketCapacity);
      objectDependentHeads =
          deviceAllocate<std::uint32_t>(config.objectCapacity);
      dependencyNext = deviceAllocate<std::uint32_t>(dependencyCapacity);
      dependencySatisfied =
          deviceAllocate<std::uint32_t>(dependencyCapacity);
      remainingDependencies =
          deviceAllocate<std::uint32_t>(config.workTicketCapacity);
      changedWorkTickets =
          deviceAllocate<std::uint32_t>(config.workTicketCapacity);
      changedQueued =
          deviceAllocate<std::uint32_t>(config.workTicketCapacity);
      changedCount = deviceAllocate<std::uint32_t>(1);
      changedOverflow = deviceAllocate<std::uint32_t>(1);
      requestProgress =
          deviceAllocate<abi::RequestProgress>(config.requestCapacity);
      reductionExpected =
          deviceAllocate<std::uint32_t>(config.workTicketCapacity);
      reductionCompleted =
          deviceAllocate<std::uint32_t>(config.workTicketCapacity);
      reductionFailed =
          deviceAllocate<std::uint32_t>(config.workTicketCapacity);
      for (DirectoryUpload &upload : directoryUploads) {
        checkCuda(cudaHostAlloc(
                      reinterpret_cast<void **>(&upload.objects),
                      config.objectCapacity * sizeof(abi::ObjectEntry),
                      cudaHostAllocPortable),
                  "cudaHostAlloc object-directory staging");
        checkCuda(cudaHostAlloc(
                      reinterpret_cast<void **>(&upload.replicas),
                      replicaCapacity * sizeof(abi::ReplicaEntry),
                      cudaHostAllocPortable),
                  "cudaHostAlloc replica-directory staging");
        checkCuda(cudaEventCreateWithFlags(&upload.complete,
                                           cudaEventDisableTiming),
                  "cudaEventCreate directory upload");
      }
      for (RequestUpload &upload : requestUploads) {
        checkCuda(cudaHostAlloc(
                      reinterpret_cast<void **>(&upload.requests),
                      config.requestCapacity * sizeof(abi::RequestContext),
                      cudaHostAllocPortable),
                  "cudaHostAlloc request-directory staging");
        checkCuda(cudaHostAlloc(
                      reinterpret_cast<void **>(&upload.progress),
                      config.requestCapacity * sizeof(abi::RequestProgress),
                      cudaHostAllocPortable),
                  "cudaHostAlloc request-progress staging");
        checkCuda(cudaEventCreateWithFlags(&upload.complete,
                                           cudaEventDisableTiming),
                  "cudaEventCreate request upload");
      }

      const auto backend = [](abi::SourceKind kind, bool active,
                              std::uint64_t state, std::uint64_t latencyNs,
                              std::uint64_t bandwidth,
                              std::uint32_t flags = 0) {
        const TierOwnership ownership = defaultTierOwnership(kind);
        const TierDescriptor descriptor{
            kind,
            defaultTierCapabilities(kind),
            state,
            latencyNs,
            bandwidth,
            active ? 1U : 0U,
            flags,
            static_cast<std::uint32_t>(ownership.protocol),
            static_cast<std::uint32_t>(ownership.payload),
            static_cast<std::uint32_t>(ownership.transferDestination),
            0,
        };
        const std::uint32_t directFlags =
            (descriptor.capabilities & TierDirectAddress) != 0
                ? abi::BackendDeviceVisible
                : 0U;
        return abi::BackendView{
            descriptor.deviceState,
            descriptor.estimatedLatencyNs,
            descriptor.estimatedBandwidthBytesPerSecond,
            0,
            UINT64_MAX,
            static_cast<std::uint32_t>(descriptor.kind),
            descriptor.active,
            static_cast<std::uint32_t>(descriptor.kind),
            descriptor.flags | directFlags |
                encodeTierCapabilities(descriptor.capabilities),
            0,
        };
      };
      const std::array<abi::BackendView, abi::BackendCount> hostBackends{
          backend(abi::SourceKind::Hbm, true, 0,
                  tierLatencies[static_cast<std::size_t>(abi::SourceKind::Hbm)],
                  tierBandwidths[static_cast<std::size_t>(abi::SourceKind::Hbm)]),
          backend(abi::SourceKind::HostMapped, true, 0,
                  tierLatencies[static_cast<std::size_t>(abi::SourceKind::HostMapped)],
                  tierBandwidths[static_cast<std::size_t>(abi::SourceKind::HostMapped)]),
          backend(abi::SourceKind::HostStaged, true, 0,
                  tierLatencies[static_cast<std::size_t>(abi::SourceKind::HostStaged)],
                  tierBandwidths[static_cast<std::size_t>(abi::SourceKind::HostStaged)]),
          backend(abi::SourceKind::Nvme, nvme != nullptr,
                  nvme == nullptr
                      ? 0
                      : reinterpret_cast<std::uint64_t>(nvme->deviceQueue()),
                  tierLatencies[static_cast<std::size_t>(abi::SourceKind::Nvme)],
                  tierBandwidths[static_cast<std::size_t>(abi::SourceKind::Nvme)],
                  nvme != nullptr && config.enableCtaNvmeTryIssue
                      ? abi::BackendCtaTryIssue
                      : 0U),
          backend(abi::SourceKind::Cxl, cxl != nullptr,
                  cxl == nullptr
                      ? 0
                      : reinterpret_cast<std::uint64_t>(cxl->deviceAddress()),
                  tierLatencies[static_cast<std::size_t>(abi::SourceKind::Cxl)],
                  tierBandwidths[static_cast<std::size_t>(abi::SourceKind::Cxl)],
                  cxl != nullptr ? abi::BackendDeviceVisible : 0U),
          backend(abi::SourceKind::Rdma, false, 0,
                  tierLatencies[static_cast<std::size_t>(abi::SourceKind::Rdma)],
                  tierBandwidths[static_cast<std::size_t>(abi::SourceKind::Rdma)]),
      };
      checkCuda(cudaMemcpy(backendEntries, hostBackends.data(),
                           sizeof(hostBackends), cudaMemcpyHostToDevice),
                "upload backend directory");
      for (std::size_t index = 0; index < hostBackends.size(); ++index) {
        const abi::BackendView &entry = hostBackends[index];
        const TierOwnership ownership = defaultTierOwnership(
            static_cast<abi::SourceKind>(entry.sourceKind));
        tierDescriptors[index] = {
            static_cast<abi::SourceKind>(entry.sourceKind),
            decodeTierCapabilities(entry.flags),
            entry.deviceState,
            entry.estimatedLatencyNs,
            entry.estimatedBandwidthBytesPerSecond,
            entry.active,
            entry.flags,
            static_cast<std::uint32_t>(ownership.protocol),
            static_cast<std::uint32_t>(ownership.payload),
            static_cast<std::uint32_t>(ownership.transferDestination),
            0,
        };
      }
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
          workTickets,
          workRunnableNs,
          dependencies,
          intentPool,
          intentQueueEntries,
          intentQueueHeads,
          readyWorkTickets,
          readyCount,
          readyHead,
          pendingWorkTickets,
          pendingCount,
          ctaCompletions,
          objectDependentHeads,
          dependencyNext,
          dependencySatisfied,
          remainingDependencies,
          changedWorkTickets,
          changedQueued,
          changedCount,
          changedOverflow,
          requestProgress,
          reductionExpected,
          reductionCompleted,
          reductionFailed,
          config.requestCapacity,
          config.tenantCapacity,
          config.objectCapacity,
          replicaCapacity,
          abi::BackendCount,
          config.intentCapacity,
          config.workTicketCapacity,
          dependencyCapacity,
          config.maxDependenciesPerWorkTicket,
          0,
          0,
          0,
          0,
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
    for (OwnedReplica &replica : object.replicas) {
      if (replica.placement == Placement::Hbm &&
          replica.sourceDevice != nullptr) {
        (void)cudaFree(replica.sourceDevice);
      }
      if (replica.hostAllocation != nullptr) {
        (void)cudaFreeHost(replica.hostAllocation);
      }
    }
    if (object.accountedStagingBytes <= ownedStagingBytes) {
      ownedStagingBytes -= object.accountedStagingBytes;
    } else {
      ownedStagingBytes = 0;
    }
    object = {nullptr, nullptr, {}, 0};
  }

  void reapRetiredObjects() noexcept {
    detail::NoexceptCudaDeviceGuard deviceGuard(config.deviceOrdinal);
    auto retired = retiredObjects.begin();
    while (retired != retiredObjects.end()) {
      if (retired->complete == nullptr) {
        releaseObject(retired->object);
        retired = retiredObjects.erase(retired);
        continue;
      }
      const cudaError_t status = cudaEventQuery(retired->complete);
      if (status == cudaErrorNotReady) {
        ++retired;
        continue;
      }
      if (status != cudaSuccess) {
        // Keep the allocation alive after an event-query error.  The owning
        // runtime close path performs the final device quiescence and cleans
        // the event; a transient query failure must never turn into a use-
        // after-free on any runtime-owned tier destination.
        (void)cudaGetLastError();
        ++retired;
        continue;
      }
      releaseObject(retired->object);
      (void)cudaEventDestroy(retired->complete);
      retired = retiredObjects.erase(retired);
    }
  }

  void retireObject(OwnedObject object, cudaStream_t stream,
                    cudaEvent_t priorConsumerEvent) {
    if (object.nvmeBuffer == nullptr && object.stagingDevice == nullptr &&
        object.replicas.empty()) {
      releaseObject(object);
      return;
    }
    if (stream == nullptr || priorConsumerEvent == nullptr) {
      throw std::invalid_argument(
          "runtime object retirement requires a stream and prior consumer event");
    }
    RetiredObject retired{std::move(object), nullptr};
    const cudaError_t eventStatus =
        cudaEventCreateWithFlags(&retired.complete, cudaEventDisableTiming);
    if (eventStatus != cudaSuccess) {
      // No retirement event exists to carry the lifetime edge.  Quiesce the
      // known consumer stream before releasing the old object; if even that
      // cannot be established, use the device-wide lifetime boundary.  This
      // branch is exceptional and never participates in steady-state reuse.
      if (cudaStreamWaitEvent(stream, priorConsumerEvent, 0) == cudaSuccess) {
        (void)cudaStreamSynchronize(stream);
      } else {
        (void)cudaDeviceSynchronize();
      }
      releaseObject(retired.object);
      checkCuda(eventStatus, "create runtime object retirement event");
    }
    bool recorded = false;
    try {
      checkCuda(cudaStreamWaitEvent(stream, priorConsumerEvent, 0),
                "wait for previous runtime object consumers");
      checkCuda(cudaEventRecord(retired.complete, stream),
                "record runtime object retirement event");
      recorded = true;
      retiredObjects.push_back(std::move(retired));
    } catch (...) {
      // This is an exceptional CUDA/API failure path.  Make the lifetime
      // boundary safe before releasing the old allocation; steady-state
      // replacement never enters this synchronization.
      if (recorded) {
        (void)cudaEventSynchronize(retired.complete);
      } else {
        (void)cudaStreamSynchronize(stream);
      }
      if (retired.complete != nullptr) {
        (void)cudaEventDestroy(retired.complete);
        retired.complete = nullptr;
      }
      releaseObject(retired.object);
      throw;
    }
  }

  void reserveStaging(std::uint64_t bytes, OwnedObject &object) {
    if (ownedStagingBytes > config.stagingByteCapacity ||
        bytes > config.stagingByteCapacity - ownedStagingBytes) {
      throw std::runtime_error(
          "runtime-owned HBM staging byte capacity exhausted");
    }
    ownedStagingBytes += bytes;
    stagingHighWaterBytes = std::max(stagingHighWaterBytes, ownedStagingBytes);
    object.accountedStagingBytes = bytes;
  }

  void release() noexcept {
    detail::NoexceptCudaDeviceGuard deviceGuard(config.deviceOrdinal);
    // Destruction is the process/lifetime boundary.  The steady-state
    // replacement path uses reapRetiredObjects() and CUDA events instead of
    // entering this synchronous boundary.
    (void)cudaDeviceSynchronize();
    for (RequestUpload &upload : requestUploads) {
      if (upload.pending && upload.complete != nullptr) {
        (void)cudaEventSynchronize(upload.complete);
      }
      if (upload.complete != nullptr) {
        (void)cudaEventDestroy(upload.complete);
      }
      if (upload.progress != nullptr) {
        (void)cudaFreeHost(upload.progress);
      }
      if (upload.requests != nullptr) {
        (void)cudaFreeHost(upload.requests);
      }
      upload = {};
    }
    for (DirectoryUpload &upload : directoryUploads) {
      if (upload.pending && upload.complete != nullptr) {
        (void)cudaEventSynchronize(upload.complete);
      }
      if (upload.complete != nullptr) {
        (void)cudaEventDestroy(upload.complete);
      }
      if (upload.replicas != nullptr) {
        (void)cudaFreeHost(upload.replicas);
      }
      if (upload.objects != nullptr) {
        (void)cudaFreeHost(upload.objects);
      }
      upload = {};
    }
    for (std::optional<OwnedObject> &object : objects) {
      if (object.has_value()) {
        releaseObject(*object);
        object.reset();
      }
    }
    for (RetiredObject &retired : retiredObjects) {
      releaseObject(retired.object);
      if (retired.complete != nullptr) {
        (void)cudaEventDestroy(retired.complete);
      }
    }
    retiredObjects.clear();
    if (view != nullptr) {
      (void)cudaFree(view);
      view = nullptr;
    }
    if (intentPool != nullptr) {
      (void)cudaFree(intentPool);
      intentPool = nullptr;
    }
    if (intentQueueHeads != nullptr) {
      (void)cudaFree(intentQueueHeads);
      intentQueueHeads = nullptr;
    }
    if (intentQueueEntries != nullptr) {
      (void)cudaFree(intentQueueEntries);
      intentQueueEntries = nullptr;
    }
    if (readyHead != nullptr) {
      (void)cudaFree(readyHead);
      readyHead = nullptr;
    }
    if (pendingCount != nullptr) {
      (void)cudaFree(pendingCount);
      pendingCount = nullptr;
    }
    if (pendingWorkTickets != nullptr) {
      (void)cudaFree(pendingWorkTickets);
      pendingWorkTickets = nullptr;
    }
    if (ctaCompletions != nullptr) {
      (void)cudaFree(ctaCompletions);
      ctaCompletions = nullptr;
    }
    if (changedOverflow != nullptr) {
      (void)cudaFree(changedOverflow);
      changedOverflow = nullptr;
    }
    if (requestProgress != nullptr) {
      (void)cudaFree(requestProgress);
      requestProgress = nullptr;
    }
    if (reductionFailed != nullptr) {
      (void)cudaFree(reductionFailed);
      reductionFailed = nullptr;
    }
    if (reductionCompleted != nullptr) {
      (void)cudaFree(reductionCompleted);
      reductionCompleted = nullptr;
    }
    if (reductionExpected != nullptr) {
      (void)cudaFree(reductionExpected);
      reductionExpected = nullptr;
    }
    if (changedCount != nullptr) {
      (void)cudaFree(changedCount);
      changedCount = nullptr;
    }
    if (changedWorkTickets != nullptr) {
      (void)cudaFree(changedWorkTickets);
      changedWorkTickets = nullptr;
    }
    if (changedQueued != nullptr) {
      (void)cudaFree(changedQueued);
      changedQueued = nullptr;
    }
    if (remainingDependencies != nullptr) {
      (void)cudaFree(remainingDependencies);
      remainingDependencies = nullptr;
    }
    if (dependencyNext != nullptr) {
      (void)cudaFree(dependencyNext);
      dependencyNext = nullptr;
    }
    if (dependencySatisfied != nullptr) {
      (void)cudaFree(dependencySatisfied);
      dependencySatisfied = nullptr;
    }
    if (objectDependentHeads != nullptr) {
      (void)cudaFree(objectDependentHeads);
      objectDependentHeads = nullptr;
    }
    if (readyCount != nullptr) {
      (void)cudaFree(readyCount);
      readyCount = nullptr;
    }
    if (readyWorkTickets != nullptr) {
      (void)cudaFree(readyWorkTickets);
      readyWorkTickets = nullptr;
    }
    if (workTickets != nullptr) {
      (void)cudaFree(workTickets);
      workTickets = nullptr;
    }
    if (workRunnableNs != nullptr) {
      (void)cudaFree(workRunnableNs);
      workRunnableNs = nullptr;
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

  void ensureObjectReplaceable(std::uint32_t slot) const {
    checkObjectSlot(slot);
    if (!objectInstalled[slot]) {
      return;
    }
    const abi::ObjectEntry current = downloadOne(objectEntries, slot);
    const auto state = static_cast<abi::ObjectState>(current.state);
    if (current.issueCount != 0 || state == abi::ObjectState::Queued ||
        state == abi::ObjectState::Issued) {
      throw std::runtime_error(
          "object slot is still referenced by the current acquisition epoch; "
          "reset and quiesce the epoch before replacing it");
    }
  }

  void ensureObjectRangeReplaceable(std::uint32_t firstSlot,
                                    std::uint32_t count) const {
    if (count == 0 || firstSlot > config.objectCapacity ||
        count > config.objectCapacity - firstSlot) {
      throw std::out_of_range("object replacement range exceeds capacity");
    }
    bool hasInstalled = false;
    for (std::uint32_t slot = firstSlot; slot < firstSlot + count; ++slot) {
      hasInstalled |= objectInstalled[slot];
    }
    if (!hasInstalled) {
      return;
    }
    std::vector<abi::ObjectEntry> current(count);
    checkCuda(cudaMemcpy(current.data(), objectEntries + firstSlot,
                         current.size() * sizeof(current.front()),
                         cudaMemcpyDeviceToHost),
              "read object replacement state");
    for (std::uint32_t index = 0; index < count; ++index) {
      if (!objectInstalled[firstSlot + index]) {
        continue;
      }
      const abi::ObjectEntry &entry = current[index];
      const auto state = static_cast<abi::ObjectState>(entry.state);
      if (entry.issueCount != 0 || state == abi::ObjectState::Queued ||
          state == abi::ObjectState::Issued) {
        throw std::runtime_error(
            "object slot is still referenced by the current acquisition "
            "epoch; reset and quiesce the epoch before replacing it");
      }
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
  std::array<TierDescriptor, abi::BackendCount> tierDescriptors{};
  abi::IntentSlot *intents = nullptr;
  abi::WorkTicket *workTickets = nullptr;
  std::uint64_t *workRunnableNs = nullptr;
  abi::WorkDependency *dependencies = nullptr;
  abi::IntentPool *intentPool = nullptr;
  abi::IntentQueueEntry *intentQueueEntries = nullptr;
  std::uint64_t *intentQueueHeads = nullptr;
  std::uint32_t *readyWorkTickets = nullptr;
  std::uint32_t *readyCount = nullptr;
  std::uint32_t *readyHead = nullptr;
  std::uint32_t *pendingWorkTickets = nullptr;
  std::uint32_t *pendingCount = nullptr;
  std::uint32_t *ctaCompletions = nullptr;
  std::uint32_t *objectDependentHeads = nullptr;
  std::uint32_t *dependencyNext = nullptr;
  std::uint32_t *dependencySatisfied = nullptr;
  std::uint32_t *remainingDependencies = nullptr;
  std::uint32_t *changedWorkTickets = nullptr;
  std::uint32_t *changedQueued = nullptr;
  std::uint32_t *changedCount = nullptr;
  std::uint32_t *changedOverflow = nullptr;
  abi::RequestProgress *requestProgress = nullptr;
  std::uint32_t *reductionExpected = nullptr;
  std::uint32_t *reductionCompleted = nullptr;
  std::uint32_t *reductionFailed = nullptr;
  abi::RuntimeView *view = nullptr;
  std::vector<abi::RequestContext> requestsHost;
  std::vector<abi::TenantContext> tenantsHost;
  std::vector<bool> requestInstalled;
  std::vector<bool> objectInstalled;
  std::vector<std::optional<OwnedObject>> objects;
  std::vector<RetiredObject> retiredObjects;
  std::shared_ptr<NvmeTransport> nvme;
  std::shared_ptr<CxlDaxTransport> cxl;
  std::array<std::uint64_t, abi::BackendCount> tierLatencies{};
  std::array<std::uint64_t, abi::BackendCount> tierBandwidths{};
  std::vector<DirectoryUpload> directoryUploads =
      std::vector<DirectoryUpload>(directoryUploadDepth());
  std::size_t nextDirectoryUpload = 0;
  std::array<RequestUpload, RequestUploadDepth> requestUploads{};
  std::size_t nextRequestUpload = 0;
  std::uint64_t ownedStagingBytes = 0;
  std::uint64_t stagingHighWaterBytes = 0;
};

HostRuntime::HostRuntime(RuntimeConfig config)
    : impl_(std::make_unique<Impl>(config, RuntimeBackends{})) {}

HostRuntime::HostRuntime(RuntimeConfig config, RuntimeBackends backends)
    : impl_(std::make_unique<Impl>(config, std::move(backends))) {}

HostRuntime::~HostRuntime() = default;
HostRuntime::HostRuntime(HostRuntime &&) noexcept = default;
HostRuntime &HostRuntime::operator=(HostRuntime &&) noexcept = default;

void HostRuntime::setRequest(std::uint32_t slot, std::uint64_t requestId,
                             std::uint32_t generation, std::uint32_t tenantId,
                             std::uint32_t priority,
                             std::uint64_t deadlineClock,
                             std::uint64_t maxOutstandingBytes) {
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
  impl_->checkRequestSlot(slot);
  if (tenantId >= impl_->config.tenantCapacity) {
    throw std::out_of_range("tenant id exceeds runtime capacity");
  }
  if (generation == 0) {
    throw std::invalid_argument("request generation must be positive");
  }
  if (priority >= abi::UrgencyBucketCount) {
    throw std::invalid_argument("request priority exceeds urgency buckets");
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
  uploadOne(impl_->requestProgress, slot,
            abi::RequestProgress{requestId, generation, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0});
}

void HostRuntime::publishRequestsAsync(std::span<const RequestSpec> requests,
                                       cudaStream_t stream) {
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
  if (requests.empty()) {
    throw std::invalid_argument("request publication batch cannot be empty");
  }
  for (std::size_t index = 0; index < requests.size(); ++index) {
    const RequestSpec &request = requests[index];
    impl_->checkRequestSlot(request.slot);
    if (request.tenantId >= impl_->config.tenantCapacity) {
      throw std::out_of_range("tenant id exceeds runtime capacity");
    }
    if (request.generation == 0) {
      throw std::invalid_argument("request generation must be positive");
    }
    if (request.priority >= abi::UrgencyBucketCount) {
      throw std::invalid_argument("request priority exceeds urgency buckets");
    }
    if (index != 0 && requests[index - 1].slot >= request.slot) {
      throw std::invalid_argument(
          "asynchronous request slots must be unique and increasing");
    }
  }

  Impl::RequestUpload &upload =
      impl_->requestUploads[impl_->nextRequestUpload++ %
                            impl_->requestUploads.size()];
  if (upload.pending) {
    checkCuda(cudaEventSynchronize(upload.complete),
              "recycle request upload staging");
    upload.pending = false;
  }

  for (std::size_t index = 0; index < requests.size(); ++index) {
    const RequestSpec &spec = requests[index];
    const abi::RequestContext request{
        spec.requestId,
        spec.deadlineClock,
        spec.maxOutstandingBytes,
        0,
        spec.generation,
        spec.tenantId,
        spec.priority,
        0,
    };
    upload.requests[index] = request;
    upload.progress[index] = abi::RequestProgress{
        spec.requestId, spec.generation, 0, 0, 0, 0, 0, 0,
        0,              0,               0, 0, 0, 0, 0};
  }

  try {
    std::size_t begin = 0;
    while (begin < requests.size()) {
      std::size_t end = begin + 1;
      while (end < requests.size() &&
             requests[end].slot == requests[end - 1].slot + 1) {
        ++end;
      }
      const std::size_t count = end - begin;
      const std::uint32_t firstSlot = requests[begin].slot;
      checkCuda(cudaMemcpyAsync(impl_->requests + firstSlot,
                                upload.requests + begin,
                                count * sizeof(abi::RequestContext),
                                cudaMemcpyHostToDevice, stream),
                "publish request directory asynchronously");
      checkCuda(cudaMemcpyAsync(impl_->requestProgress + firstSlot,
                                upload.progress + begin,
                                count * sizeof(abi::RequestProgress),
                                cudaMemcpyHostToDevice, stream),
                "publish request progress asynchronously");
      begin = end;
    }
    checkCuda(cudaEventRecord(upload.complete, stream),
              "record request directory upload");
    upload.pending = true;
    for (std::size_t index = 0; index < requests.size(); ++index) {
      const std::uint32_t slot = requests[index].slot;
      impl_->requestsHost[slot] = upload.requests[index];
      impl_->requestInstalled[slot] = true;
    }
  } catch (...) {
    (void)cudaStreamSynchronize(stream);
    throw;
  }
}

void HostRuntime::setTenantBudget(std::uint32_t tenantId,
                                  std::uint64_t maxOutstandingBytes,
                                  std::uint32_t weight) {
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
  if (tenantId >= impl_->config.tenantCapacity || weight == 0) {
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
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
  impl_->checkRequestSlot(slot);
  if (generation == 0) {
    throw std::invalid_argument("request generation must be positive");
  }
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
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
  const HostReplicaSpec replica{contents, placement};
  return installReplicatedObject(slot, objectId, version,
                                 std::span<const HostReplicaSpec>(&replica, 1));
}

ObjectHandle HostRuntime::installReplicatedObject(
    std::uint32_t slot, std::uint64_t objectId, std::uint32_t version,
    std::span<const HostReplicaSpec> replicas) {
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
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
  impl_->ensureObjectReplaceable(slot);
  for (const HostReplicaSpec &replica : replicas) {
    if (replica.contents.size() != bytes) {
      throw std::invalid_argument(
          "all physical replicas must have the same object size");
    }
  }

  Impl::OwnedObject allocation{nullptr, nullptr, {}, 0};
  allocation.replicas.reserve(replicas.size());
  std::vector<abi::ReplicaEntry> replicaEntries;
  replicaEntries.reserve(replicas.size());
  void *directAddress = nullptr;
  std::uint64_t directCost = UINT64_MAX;
  bool hasTransport = false;
  try {
    for (const HostReplicaSpec &spec : replicas) {
      allocation.replicas.push_back({spec.placement, nullptr, nullptr, nullptr});
      Impl::OwnedReplica &owned = allocation.replicas.back();
      if (spec.placement == Placement::Hbm) {
        checkCuda(cudaMalloc(&owned.sourceDevice, bytes),
                  "cudaMalloc HBM object replica");
        checkCuda(cudaMemcpy(owned.sourceDevice, spec.contents.data(), bytes,
                             cudaMemcpyHostToDevice),
                  "upload HBM object replica");
      } else if (spec.placement == Placement::CxlMapped) {
        if (impl_->cxl == nullptr) {
          throw std::invalid_argument(
              "CXL placement requires an active CXL DAX transport");
        }
        owned.cxlBuffer = impl_->cxl->allocate(bytes);
        std::memcpy(owned.cxlBuffer->hostAddress(), spec.contents.data(), bytes);
        owned.sourceDevice = owned.cxlBuffer->deviceAddress();
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
      const abi::SourceKind sourceKind = sourceKindFor(spec.placement);
      replicaEntries.push_back({
          reinterpret_cast<std::uint64_t>(owned.sourceDevice),
          0,
          impl_->tierLatencies[static_cast<std::size_t>(sourceKind)],
          impl_->tierBandwidths[static_cast<std::size_t>(sourceKind)],
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
      impl_->reserveStaging(bytes, allocation);
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
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
  impl_->checkObjectSlot(slot);
  if (bytes == 0 || bytes > std::numeric_limits<std::uint32_t>::max() ||
      replicas.empty() ||
      replicas.size() > impl_->config.maxReplicasPerObject) {
    throw std::invalid_argument(
        "registered object size and replicas must fit the runtime ABI");
  }
  impl_->ensureObjectReplaceable(slot);

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
    validateReplicaAddress(replica, impl_->config.deviceOrdinal);
    const bool direct = replica.placement != Placement::HostStaged;
    hasTransport |= !direct;
    const abi::SourceKind sourceKind = sourceKindFor(replica.placement);
    if (replica.placement == Placement::CxlMapped &&
        (impl_->cxl == nullptr ||
         !impl_->cxl->containsDeviceAddress(replica.sourceDeviceAddress,
                                             bytes))) {
      throw std::invalid_argument(
          "CXL replica must belong to the active mapped CXL window");
    }
    const std::uint64_t defaultLatencyNs =
        impl_->tierLatencies[static_cast<std::size_t>(sourceKind)];
    const std::uint64_t defaultBandwidthBytes =
        impl_->tierBandwidths[static_cast<std::size_t>(sourceKind)];
    const std::uint64_t latency = replica.estimatedLatencyNs == 0
                                      ? defaultLatencyNs
                                      : replica.estimatedLatencyNs;
    const std::uint64_t bandwidth =
        replica.estimatedBandwidthBytesPerSecond == 0
            ? defaultBandwidthBytes
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
  if (hasTransport) {
    validateDeviceAllocation(stagingDeviceAddress, impl_->config.deviceOrdinal,
                             "registered staging destination");
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

ObjectHandle HostRuntime::registerIndexedHostObject(
    std::uint32_t slot, std::uint64_t objectId, std::uint32_t version,
    const void *sourceDeviceAddress, void *stagingDeviceAddress,
    const std::uint32_t *sourceIndicesDevice,
    const std::uint32_t *stagingIndicesDevice, std::uint32_t indexCount,
    std::uint32_t elementBytes, std::uint32_t sourceStrideBytes,
    std::uint32_t stagingStrideBytes, std::uint32_t sourceIndexLimit,
    std::uint32_t stagingIndexLimit) {
  const IndexedHostObjectSpec object{
      objectId,          version,
      sourceDeviceAddress, stagingDeviceAddress,
      sourceIndicesDevice, stagingIndicesDevice,
      indexCount,        elementBytes,
      sourceStrideBytes, stagingStrideBytes,
      sourceIndexLimit,  stagingIndexLimit,
      false,
  };
  registerIndexedHostObjects(slot, std::span<const IndexedHostObjectSpec>(&object, 1));
  return {slot, nullptr};
}

void HostRuntime::registerIndexedHostObjects(
    std::uint32_t firstSlot,
    std::span<const IndexedHostObjectSpec> objects) {
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
  if (objects.empty() || firstSlot > impl_->config.objectCapacity ||
      objects.size() > impl_->config.objectCapacity - firstSlot) {
    throw std::invalid_argument("indexed host object range exceeds capacity");
  }
  impl_->ensureObjectRangeReplaceable(
      firstSlot, static_cast<std::uint32_t>(objects.size()));

  std::vector<abi::ObjectEntry> entries;
  std::vector<abi::ReplicaEntry> replicas(
      objects.size() * impl_->config.maxReplicasPerObject);
  entries.reserve(objects.size());
  for (std::size_t index = 0; index < objects.size(); ++index) {
    const IndexedHostObjectSpec &object = objects[index];
    validateIndexedHostObject(object, impl_->config.deviceOrdinal);
    const std::uint32_t slot = firstSlot + static_cast<std::uint32_t>(index);
    const std::uint32_t replicaStart =
        slot * impl_->config.maxReplicasPerObject;
    replicas[index * impl_->config.maxReplicasPerObject] =
        makeIndexedHostReplica(
            object,
            impl_->tierLatencies[
                static_cast<std::size_t>(abi::SourceKind::HostStaged)],
            impl_->tierBandwidths[
                static_cast<std::size_t>(abi::SourceKind::HostStaged)]);
    entries.push_back(makeIndexedHostObject(object, replicaStart));
  }
  const std::uint32_t firstReplica =
      firstSlot * impl_->config.maxReplicasPerObject;
  checkCuda(cudaMemcpy(impl_->replicaEntries + firstReplica, replicas.data(),
                       replicas.size() * sizeof(replicas.front()),
                       cudaMemcpyHostToDevice),
            "upload indexed host replicas");
  checkCuda(cudaMemcpy(impl_->objectEntries + firstSlot, entries.data(),
                       entries.size() * sizeof(entries.front()),
                       cudaMemcpyHostToDevice),
            "upload indexed host objects");
  for (std::size_t index = 0; index < objects.size(); ++index) {
    const std::uint32_t slot = firstSlot + static_cast<std::uint32_t>(index);
    if (impl_->objects[slot].has_value()) {
      impl_->releaseObject(*impl_->objects[slot]);
      impl_->objects[slot].reset();
    }
    impl_->objectInstalled[slot] = true;
  }
}

void HostRuntime::registerIndexedHostObjectsAsync(
    std::uint32_t firstSlot, std::span<const IndexedHostObjectSpec> objects,
    cudaStream_t stream) {
  registerIndexedHostObjectsAsyncQuiesced(firstSlot, objects, stream, nullptr);
}

void HostRuntime::registerIndexedHostObjectsAsyncQuiesced(
    std::uint32_t firstSlot, std::span<const IndexedHostObjectSpec> objects,
    cudaStream_t stream, cudaEvent_t priorConsumerEvent) {
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
  if (objects.empty() || firstSlot > impl_->config.objectCapacity ||
      objects.size() > impl_->config.objectCapacity - firstSlot) {
    throw std::invalid_argument("indexed host object range exceeds capacity");
  }
  impl_->reapRetiredObjects();
  if (priorConsumerEvent == nullptr) {
    impl_->ensureObjectRangeReplaceable(
        firstSlot, static_cast<std::uint32_t>(objects.size()));
    bool replacingOwnedObject = false;
    for (std::size_t index = 0; index < objects.size(); ++index) {
      replacingOwnedObject |= impl_->objects[firstSlot + index].has_value();
    }
    if (replacingOwnedObject) {
      // There is no stream event proving that the previous consumer stopped
      // reading its runtime-owned destination. This is a deliberately loud,
      // exceptional fallback; normal host-staged replacement supplies the
      // event and remains stream ordered.
      checkCuda(cudaDeviceSynchronize(),
                "quiesce host object replacement without consumer event");
    }
  } else {
    checkCuda(cudaStreamWaitEvent(stream, priorConsumerEvent, 0),
              "wait for indexed object consumers");
  }

  Impl::DirectoryUpload &upload =
      impl_->directoryUploads[impl_->nextDirectoryUpload++ %
                              impl_->directoryUploads.size()];
  if (upload.pending) {
    checkCuda(cudaEventSynchronize(upload.complete),
              "recycle directory upload staging");
    upload.pending = false;
  }
  const std::size_t replicaCount =
      objects.size() * impl_->config.maxReplicasPerObject;
  std::memset(upload.replicas, 0,
              replicaCount * sizeof(abi::ReplicaEntry));

  for (std::size_t index = 0; index < objects.size(); ++index) {
    const IndexedHostObjectSpec &object = objects[index];
    validateIndexedHostObject(object, impl_->config.deviceOrdinal);
    const std::uint32_t slot = firstSlot + static_cast<std::uint32_t>(index);
    const std::uint32_t replicaStart =
        slot * impl_->config.maxReplicasPerObject;
    upload.replicas[index * impl_->config.maxReplicasPerObject] =
        makeIndexedHostReplica(
            object,
            impl_->tierLatencies[
                static_cast<std::size_t>(abi::SourceKind::HostStaged)],
            impl_->tierBandwidths[
                static_cast<std::size_t>(abi::SourceKind::HostStaged)]);
    upload.objects[index] = makeIndexedHostObject(object, replicaStart);
  }

  const std::uint32_t firstReplica =
      firstSlot * impl_->config.maxReplicasPerObject;
  checkCuda(cudaMemcpyAsync(impl_->replicaEntries + firstReplica,
                            upload.replicas,
                            replicaCount * sizeof(abi::ReplicaEntry),
                            cudaMemcpyHostToDevice, stream),
            "upload indexed host replicas asynchronously");
  checkCuda(cudaMemcpyAsync(impl_->objectEntries + firstSlot, upload.objects,
                            objects.size() * sizeof(abi::ObjectEntry),
                            cudaMemcpyHostToDevice, stream),
            "upload indexed host objects asynchronously");
  checkCuda(cudaEventRecord(upload.complete, stream),
            "record indexed host directory upload");
  upload.pending = true;

  for (std::size_t index = 0; index < objects.size(); ++index) {
    const std::uint32_t slot = firstSlot + static_cast<std::uint32_t>(index);
    if (impl_->objects[slot].has_value()) {
      Impl::OwnedObject old = std::move(*impl_->objects[slot]);
      impl_->objects[slot].reset();
      if (priorConsumerEvent != nullptr) {
        impl_->retireObject(std::move(old), stream, priorConsumerEvent);
      } else {
        impl_->releaseObject(old);
      }
    }
    impl_->objectInstalled[slot] = true;
  }
}

ObjectHandle HostRuntime::installNvmeObject(
    std::uint32_t slot, std::uint64_t objectId, std::uint32_t version,
    std::uint64_t sourceByteOffset, std::size_t bytes,
    std::unique_ptr<NvmeBuffer> destination) {
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
  impl_->checkObjectSlot(slot);
  if (impl_->nvme == nullptr) {
    throw std::invalid_argument("NVMe transport is required");
  }
  impl_->ensureObjectReplaceable(slot);
  const NvmeCapabilities &capabilities = impl_->nvme->capabilities();
  if (bytes == 0 || bytes % capabilities.lbaSize != 0 ||
      sourceByteOffset % capabilities.lbaSize != 0 ||
      sourceByteOffset > capabilities.namespaceBytes ||
      bytes > capabilities.namespaceBytes - sourceByteOffset) {
    throw std::invalid_argument("NVMe object range is invalid or unaligned");
  }

  // A null destination selects the runtime-owned reuse path. This is the
  // steady-state serving path: replacing an object changes only its source
  // byte range and generation, while the mapped HBM/DMA resources stay
  // owned by the existing slot. A caller that supplies a destination keeps
  // the explicit setup-time allocation path used by standalone benchmarks.
  NvmeBuffer *buffer = destination.get();
  bool reuseExisting = false;
  if (destination == nullptr) {
    if (!impl_->objects[slot].has_value() ||
        impl_->objects[slot]->nvmeBuffer == nullptr ||
        impl_->objects[slot]->nvmeBuffer->bytes() < bytes) {
      destination = impl_->nvme->allocate(bytes);
    } else {
      buffer = impl_->objects[slot]->nvmeBuffer.get();
      reuseExisting = true;
    }
  }
  if (destination != nullptr) {
    buffer = destination.get();
  }
  if (buffer == nullptr || bytes > buffer->bytes()) {
    throw std::invalid_argument("NVMe destination is smaller than the object");
  }

  const abi::ReplicaEntry replica{
      sourceByteOffset,
      buffer->dmaPageListAddress(),
      impl_->tierLatencies[static_cast<std::size_t>(abi::SourceKind::Nvme)],
      impl_->tierBandwidths[static_cast<std::size_t>(abi::SourceKind::Nvme)],
      static_cast<std::uint32_t>(abi::SourceKind::Nvme),
      buffer->dmaPageCount(),
      static_cast<std::uint32_t>(abi::SourceKind::Nvme),
      abi::ReplicaTransport |
          (buffer->dmaTarget() == NvmeDmaTarget::HbmPeer
               ? abi::ReplicaDmaHbm
               : 0U),
      0,
      0,
  };
  const std::uint32_t replicaStart = slot * impl_->config.maxReplicasPerObject;
  abi::ObjectEntry entry{
      objectId,
      reinterpret_cast<std::uint64_t>(buffer->deviceAddress()),
      bytes,
      0,
      version,
      static_cast<std::uint32_t>(abi::ObjectState::New),
      replicaStart,
      1,
      abi::InvalidIndex,
      buffer->dmaTarget() == NvmeDmaTarget::HbmPeer
          ? abi::ReplicaDmaHbm
          : 0U,
      0,
  };
  uploadOne(impl_->replicaEntries, replicaStart, replica);
  uploadOne(impl_->objectEntries, slot, entry);
  if (reuseExisting) {
    // The old slot continues to own the reused buffer. Only its device
    // directory metadata was replaced above.
    impl_->objectInstalled[slot] = true;
    return {slot, buffer->deviceAddress()};
  }
  Impl::OwnedObject allocation{
      buffer->deviceAddress(), std::move(destination), {}, 0};
  if (impl_->objects[slot].has_value()) {
    impl_->releaseObject(*impl_->objects[slot]);
  }
  impl_->objects[slot] = std::move(allocation);
  impl_->objectInstalled[slot] = true;
  return {slot, buffer->deviceAddress()};
}

ObjectHandle HostRuntime::installNvmeObjectAsync(
    std::uint32_t slot, std::uint64_t objectId, std::uint32_t version,
    std::uint64_t sourceByteOffset, std::size_t bytes, cudaStream_t stream,
    cudaEvent_t priorConsumerEvent) {
  return installNvmeObjectAsync(
      slot, objectId, version, sourceByteOffset, bytes, stream,
      priorConsumerEvent, std::unique_ptr<NvmeBuffer>{});
}

ObjectHandle HostRuntime::installNvmeObjectAsync(
    std::uint32_t slot, std::uint64_t objectId, std::uint32_t version,
    std::uint64_t sourceByteOffset, std::size_t bytes, cudaStream_t stream,
    cudaEvent_t priorConsumerEvent, std::unique_ptr<NvmeBuffer> destination) {
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
  impl_->checkObjectSlot(slot);
  if (stream == nullptr) {
    throw std::invalid_argument(
        "stream-ordered NVMe installation requires a CUDA stream");
  }
  if (impl_->nvme == nullptr) {
    throw std::invalid_argument("NVMe transport is required");
  }
  const NvmeCapabilities &capabilities = impl_->nvme->capabilities();
  if (bytes == 0 || bytes % capabilities.lbaSize != 0 ||
      sourceByteOffset % capabilities.lbaSize != 0 ||
      sourceByteOffset > capabilities.namespaceBytes ||
      bytes > capabilities.namespaceBytes - sourceByteOffset) {
    throw std::invalid_argument("NVMe object range is invalid or unaligned");
  }
  impl_->reapRetiredObjects();

  const bool hasCurrent = impl_->objects[slot].has_value();
  if (hasCurrent && priorConsumerEvent == nullptr) {
    throw std::invalid_argument(
        "replacing an NVMe object asynchronously requires its prior consumer event");
  }
  if (hasCurrent && impl_->objects[slot]->nvmeBuffer == nullptr) {
    throw std::logic_error(
        "an asynchronous NVMe slot contains a non-NVMe object");
  }

  NvmeBuffer *buffer = nullptr;
  bool reuseExisting = false;
  if (destination == nullptr && hasCurrent &&
      impl_->objects[slot]->nvmeBuffer != nullptr &&
      impl_->objects[slot]->nvmeBuffer->bytes() >= bytes) {
    buffer = impl_->objects[slot]->nvmeBuffer.get();
    reuseExisting = true;
  } else if (destination == nullptr) {
    destination = impl_->nvme->allocate(bytes);
    buffer = destination.get();
  } else {
    buffer = destination.get();
  }
  if (buffer == nullptr || bytes > buffer->bytes()) {
    throw std::invalid_argument("NVMe destination is smaller than the object");
  }

  const abi::ReplicaEntry replica{
      sourceByteOffset,
      buffer->dmaPageListAddress(),
      impl_->tierLatencies[static_cast<std::size_t>(abi::SourceKind::Nvme)],
      impl_->tierBandwidths[static_cast<std::size_t>(abi::SourceKind::Nvme)],
      static_cast<std::uint32_t>(abi::SourceKind::Nvme),
      buffer->dmaPageCount(),
      static_cast<std::uint32_t>(abi::SourceKind::Nvme),
      abi::ReplicaTransport |
          (buffer->dmaTarget() == NvmeDmaTarget::HbmPeer ? abi::ReplicaDmaHbm
                                                          : 0U),
      0,
      0,
  };
  const std::uint32_t replicaStart = slot * impl_->config.maxReplicasPerObject;
  const abi::ObjectEntry entry{
      objectId,
      reinterpret_cast<std::uint64_t>(buffer->deviceAddress()),
      bytes,
      0,
      version,
      static_cast<std::uint32_t>(abi::ObjectState::New),
      replicaStart,
      1,
      abi::InvalidIndex,
      buffer->dmaTarget() == NvmeDmaTarget::HbmPeer ? abi::ReplicaDmaHbm : 0U,
      0,
  };

  // Reuse the existing pinned directory ring.  The ring event, not a
  // pageable stack temporary, owns the lifetime of the async source bytes.
  Impl::DirectoryUpload &upload =
      impl_->directoryUploads[impl_->nextDirectoryUpload++ %
                              impl_->directoryUploads.size()];
  if (upload.pending) {
    checkCuda(cudaEventSynchronize(upload.complete),
              "recycle directory upload staging");
    upload.pending = false;
  }
  if (hasCurrent) {
    // The old directory entry is still visible to the previous consumer until
    // this event.  Queue the dependency before overwriting the entry; placing
    // it after the H2D update would permit a directory/data race.
    checkCuda(cudaStreamWaitEvent(stream, priorConsumerEvent, 0),
              "wait before replacing NVMe directory entry");
  }
  upload.objects[slot] = entry;
  upload.replicas[replicaStart] = replica;
  checkCuda(cudaMemcpyAsync(impl_->replicaEntries + replicaStart,
                            upload.replicas + replicaStart,
                            sizeof(abi::ReplicaEntry),
                            cudaMemcpyHostToDevice, stream),
            "publish NVMe replica asynchronously");
  checkCuda(cudaMemcpyAsync(impl_->objectEntries + slot, upload.objects + slot,
                            sizeof(abi::ObjectEntry), cudaMemcpyHostToDevice,
                            stream),
            "publish NVMe object asynchronously");
  checkCuda(cudaEventRecord(upload.complete, stream),
            "record NVMe directory upload");
  upload.pending = true;

  if (!reuseExisting) {
    Impl::OwnedObject allocation{
        buffer->deviceAddress(), std::move(destination), {}, 0};
    std::optional<Impl::OwnedObject> old;
    if (impl_->objects[slot].has_value()) {
      old = std::move(*impl_->objects[slot]);
      impl_->objects[slot].reset();
    }
    impl_->objects[slot] = std::move(allocation);
    if (old.has_value()) {
      impl_->retireObject(std::move(*old), stream, priorConsumerEvent);
    }
  }
  impl_->objectInstalled[slot] = true;
  return {slot, buffer->deviceAddress()};
}

ObjectHandle HostRuntime::installNvmeObject(
    std::uint32_t slot, std::uint64_t objectId, std::uint32_t version,
    std::uint64_t sourceByteOffset, std::size_t bytes) {
  return installNvmeObject(slot, objectId, version, sourceByteOffset, bytes,
                           std::unique_ptr<NvmeBuffer>{});
}

void HostRuntime::bindTensorMaps(std::uint32_t objectSlot,
                                 std::uint32_t relativeReplica,
                                 const void *replicaTensorMap,
                                 const void *stagingTensorMap) {
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
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

int HostRuntime::deviceOrdinal() const noexcept {
  return impl_->config.deviceOrdinal;
}

const RuntimeConfig &HostRuntime::config() const noexcept {
  return impl_->config;
}

TierDescriptor HostRuntime::tierDescriptor(abi::SourceKind kind) const {
  const auto index = static_cast<std::uint32_t>(kind);
  if (index >= abi::BackendCount) {
    throw std::out_of_range("source tier exceeds runtime backend directory");
  }
  return impl_->tierDescriptors[index];
}

StagingUsage HostRuntime::stagingUsage() const noexcept {
  impl_->reapRetiredObjects();
  return {impl_->ownedStagingBytes, impl_->config.stagingByteCapacity,
          impl_->stagingHighWaterBytes};
}

abi::RequestContext HostRuntime::readRequest(std::uint32_t slot) const {
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
  impl_->checkRequestSlot(slot);
  return downloadOne(impl_->requests, slot);
}

abi::TenantContext HostRuntime::readTenant(std::uint32_t tenantId) const {
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
  if (tenantId >= impl_->config.tenantCapacity) {
    throw std::out_of_range("tenant id exceeds runtime capacity");
  }
  return downloadOne(impl_->tenants, tenantId);
}

abi::RequestProgress
HostRuntime::readRequestProgress(std::uint32_t slot) const {
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
  impl_->checkRequestSlot(slot);
  checkCuda(cudaDeviceSynchronize(), "quiesce request-progress writers");
  return downloadOne(impl_->requestProgress, slot);
}

std::vector<abi::RequestProgress>
HostRuntime::readRequestProgress(std::uint32_t firstSlot,
                                 std::uint32_t count) const {
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
  if (count == 0 || firstSlot > impl_->config.requestCapacity ||
      count > impl_->config.requestCapacity - firstSlot) {
    throw std::out_of_range("request-progress range exceeds runtime capacity");
  }
  checkCuda(cudaDeviceSynchronize(), "quiesce request-progress writers");
  std::vector<abi::RequestProgress> progress(count);
  checkCuda(cudaMemcpy(progress.data(), impl_->requestProgress + firstSlot,
                       progress.size() * sizeof(progress.front()),
                       cudaMemcpyDeviceToHost),
            "download request-progress range");
  return progress;
}

void HostRuntime::copyRequestProgressAsync(
    std::uint32_t firstSlot,
    std::span<abi::RequestProgress> destination,
    cudaStream_t stream) const {
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
  const std::uint32_t count = static_cast<std::uint32_t>(destination.size());
  if (destination.empty() || destination.size() > UINT32_MAX ||
      firstSlot > impl_->config.requestCapacity ||
      count > impl_->config.requestCapacity - firstSlot) {
    throw std::out_of_range("request-progress snapshot exceeds runtime capacity");
  }
  cudaPointerAttributes attributes{};
  const cudaError_t attributeStatus =
      cudaPointerGetAttributes(&attributes, destination.data());
  if (attributeStatus != cudaSuccess) {
    (void)cudaGetLastError();
    throw std::invalid_argument(
        "request-progress snapshot destination is not CUDA page-locked host memory");
  }
  if (attributes.type != cudaMemoryTypeHost) {
    throw std::invalid_argument(
        "request-progress snapshot destination is not CUDA page-locked host memory");
  }
  checkCuda(cudaMemcpyAsync(destination.data(),
                            impl_->requestProgress + firstSlot,
                            destination.size_bytes(), cudaMemcpyDeviceToHost,
                            stream),
            "enqueue request-progress snapshot");
}

abi::ObjectEntry HostRuntime::readObject(std::uint32_t slot) const {
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
  impl_->checkObjectSlot(slot);
  return downloadOne(impl_->objectEntries, slot);
}

abi::ReplicaEntry
HostRuntime::readReplica(std::uint32_t objectSlot,
                         std::uint32_t relativeReplica) const {
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
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

abi::WorkTicket HostRuntime::readWorkTicket(std::uint32_t slot) const {
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
  if (slot >= impl_->config.workTicketCapacity) {
    throw std::out_of_range("work-ticket slot exceeds runtime capacity");
  }
  return downloadOne(impl_->workTickets, slot);
}

std::vector<std::uint64_t>
HostRuntime::readWorkRunnableNs(std::uint32_t count) const {
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
  if (count == 0 || count > impl_->config.workTicketCapacity) {
    throw std::out_of_range("work-arrival range exceeds runtime capacity");
  }
  std::vector<std::uint64_t> values(count);
  checkCuda(cudaMemcpy(values.data(), impl_->workRunnableNs,
                       values.size() * sizeof(values.front()),
                       cudaMemcpyDeviceToHost),
            "download work runnable timestamps");
  return values;
}

abi::WorkDependency HostRuntime::readWorkDependency(
    std::uint32_t workTicket, std::uint32_t relativeDependency) const {
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
  if (workTicket >= impl_->config.workTicketCapacity ||
      relativeDependency >= impl_->config.maxDependenciesPerWorkTicket) {
    throw std::out_of_range("work-ticket dependency exceeds runtime capacity");
  }
  const std::uint32_t index =
      workTicket * impl_->config.maxDependenciesPerWorkTicket +
      relativeDependency;
  return downloadOne(impl_->dependencies, index);
}

abi::IntentPool HostRuntime::readIntentPool() const {
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
  return downloadOne(impl_->intentPool, 0);
}

std::uint32_t HostRuntime::readPendingCount() const {
  return readEpochStatus(impl_->config.workTicketCapacity).pending;
}

EpochStatus
HostRuntime::readEpochStatus(std::uint32_t workTicketCount) const {
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
  if (workTicketCount == 0 ||
      workTicketCount > impl_->config.workTicketCapacity) {
    throw std::out_of_range(
        "epoch work-ticket count exceeds runtime capacity");
  }
  std::vector<abi::WorkTicket> workTickets(workTicketCount);
  checkCuda(cudaMemcpy(workTickets.data(), impl_->workTickets,
                       workTickets.size() * sizeof(workTickets.front()),
                       cudaMemcpyDeviceToHost),
            "download epoch workTicket states");

  EpochStatus status{};
  status.total = workTicketCount;
  for (const abi::WorkTicket &workTicket : workTickets) {
    switch (static_cast<abi::WorkTicketState>(workTicket.state)) {
    case abi::WorkTicketState::New:
      ++status.fresh;
      break;
    case abi::WorkTicketState::Pending:
      ++status.pending;
      break;
    case abi::WorkTicketState::Ready:
      ++status.ready;
      break;
    case abi::WorkTicketState::Done:
      ++status.done;
      break;
    case abi::WorkTicketState::Cancelled:
      ++status.cancelled;
      break;
    case abi::WorkTicketState::Failed:
      ++status.failed;
      break;
    case abi::WorkTicketState::Initializing:
      ++status.initializing;
      break;
    default:
      throw std::runtime_error("runtime returned an invalid work-ticket state");
    }
  }
  return status;
}

std::uint32_t HostRuntime::readStickyFailedCount() const {
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
  std::uint32_t value = 0;
  const auto *address = reinterpret_cast<const std::byte *>(impl_->view) +
                        offsetof(abi::RuntimeView, stickyFailedCount);
  checkCuda(cudaMemcpy(&value, address, sizeof(value), cudaMemcpyDeviceToHost),
            "download sticky failure count");
  return value;
}

std::uint32_t HostRuntime::readPendingIndexCount() const {
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
  return downloadOne(impl_->pendingCount, 0);
}

DeviceWorkPlan HostRuntime::uploadWorkPlan(const WorkPlan &plan) const {
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
  if (plan.workItems.size() > impl_->config.workTicketCapacity) {
    throw std::invalid_argument(
        "work plan exceeds the runtime work-ticket capacity");
  }
  for (const abi::WorkItem &work : plan.workItems) {
    if (work.requestSlot >= impl_->config.requestCapacity ||
        !impl_->requestInstalled[work.requestSlot] ||
        work.workTicket >= impl_->config.workTicketCapacity ||
        work.reductionGroup >= impl_->config.workTicketCapacity ||
        work.dependencyCount > impl_->config.maxDependenciesPerWorkTicket) {
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
  return DeviceWorkPlan(plan, impl_->config.deviceOrdinal);
}

} // namespace nta
