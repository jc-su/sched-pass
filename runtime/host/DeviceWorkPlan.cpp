#include "nta/DeviceWorkPlan.h"
#include "CudaDeviceGuard.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
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

std::uint32_t checkedCount(std::size_t count, const char *field) {
  if (count == 0 || count >= abi::InvalidIndex) {
    throw std::invalid_argument(std::string(field) +
                                " must fit the non-empty NTA ABI");
  }
  return static_cast<std::uint32_t>(count);
}

void validate(const WorkPlan &plan) {
  const std::uint32_t workCount =
      checkedCount(plan.workItems.size(), "work item count");
  const std::uint32_t dependencyCount =
      checkedCount(plan.dependencies.size(), "dependency count");
  (void)checkedCount(plan.requests.size(), "request count");
  const std::uint32_t workTicketBase = plan.workItems.front().workTicket;
  const std::uint32_t reductionGroupBase =
      plan.workItems.front().reductionGroup;
  // Work-plan storage is compact, but ticket identities index the owning
  // RuntimeView.  The default base is zero for numerical plans; a shared
  // acquisition service may upload several compact plans into disjoint
  // runtime ticket ranges.  Reduction groups use the same base so no request
  // in one concurrently live plan aliases another plan's accounting record.
  if (workTicketBase > abi::InvalidIndex - workCount ||
      reductionGroupBase != workTicketBase ||
      reductionGroupBase > abi::InvalidIndex - plan.requests.size()) {
    throw std::invalid_argument("work plan runtime index range is invalid");
  }
  std::uint32_t workCursor = 0;
  for (std::uint32_t requestIndex = 0; requestIndex < plan.requests.size();
       ++requestIndex) {
    const RequestWorkRange &request = plan.requests[requestIndex];
    if (request.workBegin != workCursor || request.workCount == 0 ||
        request.workCount > workCount - workCursor) {
      throw std::invalid_argument(
          "work plan request ranges are not contiguous");
    }
    for (std::uint32_t relative = 0; relative < request.workCount; ++relative) {
      const abi::WorkItem &work = plan.workItems[workCursor + relative];
      if (work.requestIndex != requestIndex ||
          work.requestSlot != request.requestSlot ||
          work.generation != request.generation ||
          work.reductionGroup != reductionGroupBase + requestIndex ||
          work.contributorIndex != relative ||
          work.contributorCount != request.workCount) {
        throw std::invalid_argument(
            "work plan request and work-item bindings disagree");
      }
    }
    workCursor += request.workCount;
  }
  if (workCursor != workCount) {
    throw std::invalid_argument("work plan contains unowned work items");
  }
  struct ObjectUse {
    std::uint32_t references = 0;
    std::uint32_t exclusiveReferences = 0;
  };
  using ObjectKey =
      std::tuple<std::uint32_t, std::uint64_t, std::uint32_t>;
  std::map<ObjectKey, ObjectUse> externalUses;
  for (std::uint32_t index = 0; index < workCount; ++index) {
    const abi::WorkItem &work = plan.workItems[index];
    const bool eventPartition =
        (work.flags & abi::WorkItemEventPartition) != 0;
    if (work.workTicket != workTicketBase + index || work.dependencyCount == 0 ||
        work.directDependencyCount > work.dependencyCount ||
        work.dependencyBegin > dependencyCount ||
        work.dependencyCount > dependencyCount - work.dependencyBegin ||
        (work.flags & ~abi::WorkItemSupportedFlags) != 0 ||
        (!eventPartition && work.completionClass != 0) ||
        (eventPartition &&
         (work.directDependencyCount != work.dependencyCount ||
          (work.completionClass != abi::InvalidIndex &&
           work.completionClass >= abi::MaximumEventCompletionClasses)))) {
      throw std::invalid_argument("work plan contains an invalid work item");
    }
    std::uint32_t directCount = 0;
    for (std::uint32_t dependency = 0; dependency < work.dependencyCount;
         ++dependency) {
      const abi::AcquireRequirement &requirement =
          plan.dependencies[work.dependencyBegin + dependency];
      if (requirement.bytes == 0 ||
          (requirement.flags & ~abi::AcquireRequirementSupportedFlags) != 0 ||
          (requirement.directBase != 0 && requirement.flags != 0)) {
        throw std::invalid_argument(
            "work plan contains an invalid acquisition requirement");
      }
      directCount += requirement.directBase != 0 ? 1U : 0U;
      if (requirement.directBase == 0) {
        ObjectUse &use = externalUses[ObjectKey{
            requirement.objectSlot, requirement.objectId,
            requirement.objectVersion}];
        ++use.references;
        use.exclusiveReferences +=
            (requirement.flags & abi::AcquireOnlineExclusive) != 0 ? 1U
                                                                  : 0U;
      }
    }
    if (directCount != work.directDependencyCount) {
      throw std::invalid_argument(
          "work plan direct dependency count is inconsistent");
    }
  }
  for (const auto &[object, use] : externalUses) {
    (void)object;
    if (use.exclusiveReferences != 0 &&
        (use.exclusiveReferences != 1 || use.references != 1)) {
      throw std::invalid_argument(
          "online-exclusive acquisition objects need exactly one "
          "work-ticket reference");
    }
  }
}

} // namespace

struct DeviceWorkPlan::Impl {
  struct UploadSlot {
    void *host = nullptr;
    cudaEvent_t complete = nullptr;
    bool pending = false;
  };

  struct ConsumerFence {
    cudaStream_t stream = nullptr;
    cudaEvent_t complete = nullptr;
  };

  static constexpr std::size_t UploadDepth = 2;

  Impl(std::uint32_t requestedWorkCapacity,
       std::uint32_t requestedDependencyCapacity, int requestedDevice)
      : workCapacity(requestedWorkCapacity),
        dependencyCapacity(requestedDependencyCapacity),
        deviceOrdinal(detail::resolveCudaDevice(requestedDevice)) {
    detail::CudaDeviceGuard deviceGuard(deviceOrdinal);
    if (workCapacity == 0 || dependencyCapacity == 0) {
      throw std::invalid_argument(
          "device work-plan capacities must be non-zero");
    }
    const std::size_t workBytes =
        static_cast<std::size_t>(workCapacity) * sizeof(abi::WorkItem);
    const std::size_t dependencyBytes =
        static_cast<std::size_t>(dependencyCapacity) *
        sizeof(abi::AcquireRequirement);
    if (workBytes > std::numeric_limits<std::size_t>::max() - dependencyBytes) {
      throw std::overflow_error("device work-plan allocation overflows");
    }
    allocationBytes = workBytes + dependencyBytes;
    dependencyOffset = workBytes;
    try {
      checkCuda(cudaMalloc(&allocation, allocationBytes),
                "cudaMalloc device work plan");
      workItems = static_cast<abi::WorkItem *>(allocation);
      dependencies = reinterpret_cast<abi::AcquireRequirement *>(
          static_cast<std::byte *>(allocation) + dependencyOffset);
      for (UploadSlot &slot : uploads) {
        checkCuda(cudaHostAlloc(&slot.host, allocationBytes,
                                cudaHostAllocPortable),
                  "cudaHostAlloc device work-plan staging");
        checkCuda(cudaEventCreateWithFlags(&slot.complete,
                                           cudaEventDisableTiming),
                  "cudaEventCreate device work-plan upload");
      }
    } catch (...) {
      for (UploadSlot &slot : uploads) {
        if (slot.complete != nullptr) {
          (void)cudaEventDestroy(slot.complete);
        }
        if (slot.host != nullptr) {
          (void)cudaFreeHost(slot.host);
        }
      }
      if (allocation != nullptr) {
        (void)cudaFree(allocation);
      }
      throw;
    }
  }

  ~Impl() {
    detail::NoexceptCudaDeviceGuard deviceGuard(deviceOrdinal);
    for (UploadSlot &slot : uploads) {
      if (slot.pending) {
        (void)cudaEventSynchronize(slot.complete);
      }
      if (slot.complete != nullptr) {
        (void)cudaEventDestroy(slot.complete);
      }
      if (slot.host != nullptr) {
        (void)cudaFreeHost(slot.host);
      }
    }
    for (ConsumerFence &fence : consumerFences) {
      if (fence.complete != nullptr) {
        (void)cudaEventSynchronize(fence.complete);
        (void)cudaEventDestroy(fence.complete);
      }
    }
    if (allocation != nullptr) {
      (void)cudaFree(allocation);
    }
  }

  void uploadAsync(const WorkPlan &plan, cudaStream_t stream) {
    detail::CudaDeviceGuard deviceGuard(deviceOrdinal);
    validate(plan);
    if (plan.workItems.size() > workCapacity ||
        plan.dependencies.size() > dependencyCapacity) {
      throw std::invalid_argument(
          "work plan exceeds the reusable device allocation");
    }
    cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
    checkCuda(cudaStreamIsCapturing(stream, &captureStatus),
              "query device work-plan capture state");
    if (captureStatus != cudaStreamCaptureStatusNone) {
      throw std::logic_error(
          "device work plans must be uploaded before CUDA graph capture");
    }
    waitForConsumers(stream);
    const std::size_t uploadIndex = acquireUploadSlot();
    UploadSlot &upload = uploads[uploadIndex];
    workCount = static_cast<std::uint32_t>(plan.workItems.size());
    dependencyCount = static_cast<std::uint32_t>(plan.dependencies.size());
    const std::size_t workBytes = plan.workItems.size() * sizeof(abi::WorkItem);
    const std::size_t dependencyBytes =
        plan.dependencies.size() * sizeof(abi::AcquireRequirement);
    std::memcpy(upload.host, plan.workItems.data(), workBytes);
    std::memcpy(static_cast<std::byte *>(upload.host) + dependencyOffset,
                plan.dependencies.data(), dependencyBytes);
    checkCuda(cudaMemcpyAsync(allocation, upload.host, workBytes,
                              cudaMemcpyHostToDevice, stream),
              "upload work items asynchronously");
    checkCuda(cudaMemcpyAsync(
                  static_cast<std::byte *>(allocation) + dependencyOffset,
                  static_cast<std::byte *>(upload.host) + dependencyOffset,
                  dependencyBytes, cudaMemcpyHostToDevice, stream),
              "upload work dependencies asynchronously");
    checkCuda(cudaEventRecord(upload.complete, stream),
              "record device work-plan upload");
    upload.pending = true;
    latestUpload = uploadIndex;
    hasUpload = true;
  }

  void waitForConsumers(cudaStream_t stream) {
    if (consumerFences.empty()) {
      return;
    }
    for (const ConsumerFence &fence : consumerFences) {
      checkCuda(cudaStreamWaitEvent(stream, fence.complete, 0),
                "wait for device work-plan consumer");
    }
    for (ConsumerFence &fence : consumerFences) {
      (void)cudaEventDestroy(fence.complete);
      fence.complete = nullptr;
    }
    consumerFences.clear();
  }

  void recordConsumer(cudaStream_t stream) {
    if (!hasUpload) {
      return;
    }
    for (ConsumerFence &fence : consumerFences) {
      if (fence.stream == stream) {
        checkCuda(cudaEventRecord(fence.complete, stream),
                  "record device work-plan consumer");
        return;
      }
    }
    ConsumerFence fence;
    fence.stream = stream;
    checkCuda(cudaEventCreateWithFlags(&fence.complete, cudaEventDisableTiming),
              "create device work-plan consumer event");
    try {
      checkCuda(cudaEventRecord(fence.complete, stream),
                "record device work-plan consumer");
    } catch (...) {
      (void)cudaEventDestroy(fence.complete);
      throw;
    }
    try {
      consumerFences.push_back(fence);
    } catch (...) {
      (void)cudaEventDestroy(fence.complete);
      throw;
    }
  }

  std::size_t acquireUploadSlot() {
    for (std::size_t offset = 0; offset < uploads.size(); ++offset) {
      const std::size_t index = (nextUpload + offset) % uploads.size();
      UploadSlot &slot = uploads[index];
      if (slot.pending) {
        const cudaError_t status = cudaEventQuery(slot.complete);
        if (status == cudaSuccess) {
          slot.pending = false;
        } else if (status == cudaErrorNotReady) {
          continue;
        } else {
          checkCuda(status, "query device work-plan upload");
        }
      }
      nextUpload = (index + 1) % uploads.size();
      return index;
    }

    const std::size_t index = nextUpload;
    checkCuda(cudaEventSynchronize(uploads[index].complete),
              "recycle device work-plan staging");
    uploads[index].pending = false;
    nextUpload = (index + 1) % uploads.size();
    return index;
  }

  void synchronizeUpload() const {
    detail::CudaDeviceGuard deviceGuard(deviceOrdinal);
    for (UploadSlot &slot : uploads) {
      if (slot.pending) {
        checkCuda(cudaEventSynchronize(slot.complete),
                  "synchronize device work-plan upload");
        slot.pending = false;
      }
    }
  }

  void *allocation = nullptr;
  mutable std::array<UploadSlot, UploadDepth> uploads{};
  abi::WorkItem *workItems = nullptr;
  abi::AcquireRequirement *dependencies = nullptr;
  std::size_t allocationBytes = 0;
  std::size_t dependencyOffset = 0;
  std::uint32_t workCapacity = 0;
  std::uint32_t dependencyCapacity = 0;
  std::uint32_t workCount = 0;
  std::uint32_t dependencyCount = 0;
  int deviceOrdinal = 0;
  std::size_t nextUpload = 0;
  std::size_t latestUpload = 0;
  bool hasUpload = false;
  mutable std::vector<ConsumerFence> consumerFences;
};

DeviceWorkPlan::DeviceWorkPlan(const WorkPlan &plan, int deviceOrdinal)
    : impl_(std::make_unique<Impl>(
          checkedCount(plan.workItems.size(), "work item count"),
          checkedCount(plan.dependencies.size(), "dependency count"),
          deviceOrdinal)) {
  upload(plan);
}
DeviceWorkPlan::DeviceWorkPlan(std::uint32_t workItemCapacity,
                               std::uint32_t dependencyCapacity,
                               int deviceOrdinal)
    : impl_(std::make_unique<Impl>(workItemCapacity, dependencyCapacity,
                                   deviceOrdinal)) {}
DeviceWorkPlan::~DeviceWorkPlan() = default;
DeviceWorkPlan::DeviceWorkPlan(DeviceWorkPlan &&) noexcept = default;
DeviceWorkPlan &DeviceWorkPlan::operator=(DeviceWorkPlan &&) noexcept = default;

void DeviceWorkPlan::upload(const WorkPlan &plan) {
  uploadAsync(plan, nullptr);
  synchronizeUpload();
}

void DeviceWorkPlan::uploadAsync(const WorkPlan &plan, cudaStream_t stream) {
  if (impl_ == nullptr) {
    throw std::logic_error("cannot upload through a moved device work plan");
  }
  impl_->uploadAsync(plan, stream);
}

void DeviceWorkPlan::waitOn(cudaStream_t stream) const {
  if (impl_ == nullptr) {
    throw std::logic_error("cannot wait on a moved device work plan");
  }
  detail::CudaDeviceGuard deviceGuard(impl_->deviceOrdinal);
  if (impl_->hasUpload) {
    checkCuda(cudaStreamWaitEvent(stream,
                                  impl_->uploads[impl_->latestUpload].complete,
                                  0),
              "cudaStreamWaitEvent device work plan");
  }
}

void DeviceWorkPlan::markConsumed(cudaStream_t stream) const {
  if (impl_ == nullptr) {
    throw std::logic_error("cannot mark a moved device work plan consumed");
  }
  detail::CudaDeviceGuard deviceGuard(impl_->deviceOrdinal);
  try {
    impl_->recordConsumer(stream);
  } catch (...) {
    // If the fence cannot be recorded, the stream has already observed the
    // plan through waitOn(). Quiesce only this exceptional recovery path so
    // the caller cannot accidentally reuse or destroy live plan storage.
    (void)cudaStreamSynchronize(stream);
    throw;
  }
}

void DeviceWorkPlan::synchronizeUpload() const {
  if (impl_ != nullptr) {
    impl_->synchronizeUpload();
  }
}

const abi::WorkItem *DeviceWorkPlan::workItems() const noexcept {
  return impl_ == nullptr ? nullptr : impl_->workItems;
}

const abi::AcquireRequirement *DeviceWorkPlan::dependencies() const noexcept {
  return impl_ == nullptr ? nullptr : impl_->dependencies;
}

std::uint32_t DeviceWorkPlan::workItemCount() const noexcept {
  return impl_ == nullptr ? 0 : impl_->workCount;
}

std::uint32_t DeviceWorkPlan::dependencyCount() const noexcept {
  return impl_ == nullptr ? 0 : impl_->dependencyCount;
}

std::uint32_t DeviceWorkPlan::workItemCapacity() const noexcept {
  return impl_ == nullptr ? 0 : impl_->workCapacity;
}

std::uint32_t DeviceWorkPlan::dependencyCapacity() const noexcept {
  return impl_ == nullptr ? 0 : impl_->dependencyCapacity;
}

int DeviceWorkPlan::deviceOrdinal() const noexcept {
  return impl_ == nullptr ? -1 : impl_->deviceOrdinal;
}

} // namespace nta
