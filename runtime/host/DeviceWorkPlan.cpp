#include "nta/DeviceWorkPlan.h"
#include "CudaDeviceGuard.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

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
          work.generation != request.generation) {
        throw std::invalid_argument(
            "work plan request and work-item bindings disagree");
      }
    }
    workCursor += request.workCount;
  }
  if (workCursor != workCount) {
    throw std::invalid_argument("work plan contains unowned work items");
  }
  for (std::uint32_t index = 0; index < workCount; ++index) {
    const abi::WorkItem &work = plan.workItems[index];
    if (work.continuation != index || work.dependencyCount == 0 ||
        work.directDependencyCount > work.dependencyCount ||
        work.dependencyBegin > dependencyCount ||
        work.dependencyCount > dependencyCount - work.dependencyBegin) {
      throw std::invalid_argument("work plan contains an invalid work item");
    }
    std::uint32_t directCount = 0;
    for (std::uint32_t dependency = 0; dependency < work.dependencyCount;
         ++dependency) {
      const abi::AcquireRequirement &requirement =
          plan.dependencies[work.dependencyBegin + dependency];
      if (requirement.bytes == 0 || requirement.flags != 0) {
        throw std::invalid_argument(
            "work plan contains an invalid acquisition requirement");
      }
      directCount += requirement.directBase != 0 ? 1U : 0U;
    }
    if (directCount != work.directDependencyCount) {
      throw std::invalid_argument(
          "work plan direct dependency count is inconsistent");
    }
  }
}

} // namespace

struct DeviceWorkPlan::Impl {
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
      checkCuda(
          cudaHostAlloc(&hostStaging, allocationBytes, cudaHostAllocPortable),
          "cudaHostAlloc device work-plan staging");
      checkCuda(
          cudaEventCreateWithFlags(&uploadComplete, cudaEventDisableTiming),
          "cudaEventCreate device work-plan upload");
    } catch (...) {
      if (uploadComplete != nullptr) {
        (void)cudaEventDestroy(uploadComplete);
      }
      if (hostStaging != nullptr) {
        (void)cudaFreeHost(hostStaging);
      }
      if (allocation != nullptr) {
        (void)cudaFree(allocation);
      }
      throw;
    }
  }

  ~Impl() {
    detail::NoexceptCudaDeviceGuard deviceGuard(deviceOrdinal);
    if (uploadPending) {
      (void)cudaEventSynchronize(uploadComplete);
    }
    if (uploadComplete != nullptr) {
      (void)cudaEventDestroy(uploadComplete);
    }
    if (hostStaging != nullptr) {
      (void)cudaFreeHost(hostStaging);
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
    // The staging image cannot be overwritten until its prior H2D transfer
    // has completed. This wait does not wait for later consumer kernels.
    synchronizeUpload();
    workCount = static_cast<std::uint32_t>(plan.workItems.size());
    dependencyCount = static_cast<std::uint32_t>(plan.dependencies.size());
    const std::size_t workBytes = plan.workItems.size() * sizeof(abi::WorkItem);
    const std::size_t dependencyBytes =
        plan.dependencies.size() * sizeof(abi::AcquireRequirement);
    std::memcpy(hostStaging, plan.workItems.data(), workBytes);
    std::memcpy(static_cast<std::byte *>(hostStaging) + dependencyOffset,
                plan.dependencies.data(), dependencyBytes);
    checkCuda(cudaMemcpyAsync(allocation, hostStaging, workBytes,
                              cudaMemcpyHostToDevice, stream),
              "upload work items asynchronously");
    checkCuda(cudaMemcpyAsync(
                  static_cast<std::byte *>(allocation) + dependencyOffset,
                  static_cast<std::byte *>(hostStaging) + dependencyOffset,
                  dependencyBytes, cudaMemcpyHostToDevice, stream),
              "upload work dependencies asynchronously");
    checkCuda(cudaEventRecord(uploadComplete, stream),
              "record device work-plan upload");
    uploadPending = true;
  }

  void synchronizeUpload() const {
    detail::CudaDeviceGuard deviceGuard(deviceOrdinal);
    if (uploadPending) {
      checkCuda(cudaEventSynchronize(uploadComplete),
                "synchronize device work-plan upload");
      uploadPending = false;
    }
  }

  void *allocation = nullptr;
  void *hostStaging = nullptr;
  cudaEvent_t uploadComplete = nullptr;
  abi::WorkItem *workItems = nullptr;
  abi::AcquireRequirement *dependencies = nullptr;
  std::size_t allocationBytes = 0;
  std::size_t dependencyOffset = 0;
  std::uint32_t workCapacity = 0;
  std::uint32_t dependencyCapacity = 0;
  std::uint32_t workCount = 0;
  std::uint32_t dependencyCount = 0;
  int deviceOrdinal = 0;
  mutable bool uploadPending = false;
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
  checkCuda(cudaStreamWaitEvent(stream, impl_->uploadComplete, 0),
            "cudaStreamWaitEvent device work plan");
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
