#include "CudaDeviceGuard.h"
#include "nta/DeviceWorkPlan.h"
#include "nta/HostRuntime.h"
#include "nta/NvmeRuntime.h"
#include "nta/WorkPlan.h"

#include <cuda_runtime_api.h>

#include <algorithm>
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
  static constexpr std::size_t DirectoryUploadDepth = 4;
  static constexpr std::size_t RequestUploadDepth = 4;

  struct OwnedReplica {
    Placement placement;
    void *hostAllocation;
    void *sourceDevice;
  };

  struct OwnedObject {
    void *stagingDevice;
    std::unique_ptr<NvmeBuffer> nvmeBuffer;
    std::vector<OwnedReplica> replicas;
    std::uint64_t accountedStagingBytes = 0;
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
                std::shared_ptr<NvmeTransport> nvmeTransport = nullptr)
      : config(normalizeRuntimeConfig(runtimeConfig)),
        requestsHost(config.requestCapacity), tenantsHost(config.tenantCapacity),
        requestInstalled(config.requestCapacity, false),
        objectInstalled(config.objectCapacity, false),
        objects(config.objectCapacity), nvme(std::move(nvmeTransport)) {
    config.deviceOrdinal = detail::resolveCudaDevice(config.deviceOrdinal);
    detail::CudaDeviceGuard deviceGuard(config.deviceOrdinal);
    if (nvme != nullptr && nvme->deviceOrdinal() != config.deviceOrdinal) {
      throw std::invalid_argument(
          "HostRuntime and NvmeTransport must own the same CUDA device");
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
        return abi::BackendView{
            state,
            latencyNs,
            bandwidth,
            0,
            UINT64_MAX,
            static_cast<std::uint32_t>(kind),
            active ? 1U : 0U,
            static_cast<std::uint32_t>(kind),
            flags,
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
                  80'000, 7'000'000'000ULL,
                  nvme != nullptr && config.enableCtaNvmeTryIssue
                      ? abi::BackendCtaTryIssue
                      : 0U),
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

  void reserveStaging(std::uint64_t bytes, OwnedObject &object) {
    if (bytes > config.stagingByteCapacity - ownedStagingBytes) {
      throw std::runtime_error(
          "runtime-owned HBM staging byte capacity exhausted");
    }
    ownedStagingBytes += bytes;
    stagingHighWaterBytes = std::max(stagingHighWaterBytes, ownedStagingBytes);
    object.accountedStagingBytes = bytes;
  }

  void release() noexcept {
    detail::NoexceptCudaDeviceGuard deviceGuard(config.deviceOrdinal);
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

  RuntimeConfig config;
  std::uint32_t replicaCapacity = 0;
  std::uint32_t dependencyCapacity = 0;
  abi::RequestContext *requests = nullptr;
  abi::TenantContext *tenants = nullptr;
  abi::ObjectEntry *objectEntries = nullptr;
  abi::ReplicaEntry *replicaEntries = nullptr;
  abi::BackendView *backendEntries = nullptr;
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
  std::shared_ptr<NvmeTransport> nvme;
  std::array<DirectoryUpload, DirectoryUploadDepth> directoryUploads{};
  std::size_t nextDirectoryUpload = 0;
  std::array<RequestUpload, RequestUploadDepth> requestUploads{};
  std::size_t nextRequestUpload = 0;
  std::uint64_t ownedStagingBytes = 0;
  std::uint64_t stagingHighWaterBytes = 0;
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
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
  impl_->checkRequestSlot(slot);
  if (tenantId >= impl_->config.tenantCapacity) {
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

  std::vector<abi::ObjectEntry> entries;
  std::vector<abi::ReplicaEntry> replicas(
      objects.size() * impl_->config.maxReplicasPerObject);
  entries.reserve(objects.size());
  for (std::size_t index = 0; index < objects.size(); ++index) {
    const IndexedHostObjectSpec &object = objects[index];
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
    const std::uint32_t slot = firstSlot + static_cast<std::uint32_t>(index);
    const std::uint32_t replicaStart =
        slot * impl_->config.maxReplicasPerObject;
    replicas[index * impl_->config.maxReplicasPerObject] = {
        reinterpret_cast<std::uint64_t>(object.sourceDeviceAddress),
        reinterpret_cast<std::uint64_t>(object.sourceIndicesDevice),
        2'000,
        30'000'000'000ULL,
        static_cast<std::uint32_t>(abi::SourceKind::HostStaged),
        object.indexCount,
        static_cast<std::uint32_t>(abi::SourceKind::HostStaged),
        abi::ReplicaTransport | abi::ReplicaIndexed,
        abi::packTransferIndexLimits(object.sourceIndexLimit,
                                     object.stagingIndexLimit),
        abi::packTransferStrides(object.sourceStrideBytes,
                                 object.stagingStrideBytes),
    };
    entries.push_back({
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
        0,
        reinterpret_cast<std::uint64_t>(object.stagingIndicesDevice),
    });
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
  detail::CudaDeviceGuard deviceGuard(impl_->config.deviceOrdinal);
  if (objects.empty() || firstSlot > impl_->config.objectCapacity ||
      objects.size() > impl_->config.objectCapacity - firstSlot) {
    throw std::invalid_argument("indexed host object range exceeds capacity");
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
    const std::uint32_t slot = firstSlot + static_cast<std::uint32_t>(index);
    const std::uint32_t replicaStart =
        slot * impl_->config.maxReplicasPerObject;
    upload.replicas[index * impl_->config.maxReplicasPerObject] = {
        reinterpret_cast<std::uint64_t>(object.sourceDeviceAddress),
        reinterpret_cast<std::uint64_t>(object.sourceIndicesDevice),
        2'000,
        30'000'000'000ULL,
        static_cast<std::uint32_t>(abi::SourceKind::HostStaged),
        object.indexCount,
        static_cast<std::uint32_t>(abi::SourceKind::HostStaged),
        abi::ReplicaTransport | abi::ReplicaIndexed,
        abi::packTransferIndexLimits(object.sourceIndexLimit,
                                     object.stagingIndexLimit),
        abi::packTransferStrides(object.sourceStrideBytes,
                                 object.stagingStrideBytes),
    };
    upload.objects[index] = {
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
        0,
        reinterpret_cast<std::uint64_t>(object.stagingIndicesDevice),
    };
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
      impl_->releaseObject(*impl_->objects[slot]);
      impl_->objects[slot].reset();
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
      destination->deviceAddress(), std::move(destination), {}, 0};
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

StagingUsage HostRuntime::stagingUsage() const noexcept {
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
