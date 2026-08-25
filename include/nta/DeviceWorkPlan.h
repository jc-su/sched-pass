#pragma once

#include "nta/WorkPlan.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>

namespace nta {

// Owns a stable device allocation for an engine-neutral WorkPlan. Work
// descriptors and dependency records stay adjacent, and integrations get one
// explicit lifetime for the complete batch description.
class DeviceWorkPlan {
public:
  explicit DeviceWorkPlan(const WorkPlan &plan, int deviceOrdinal = -1);
  DeviceWorkPlan(std::uint32_t workItemCapacity,
                 std::uint32_t dependencyCapacity, int deviceOrdinal = -1);
  ~DeviceWorkPlan();

  DeviceWorkPlan(const DeviceWorkPlan &) = delete;
  DeviceWorkPlan &operator=(const DeviceWorkPlan &) = delete;
  DeviceWorkPlan(DeviceWorkPlan &&) noexcept;
  DeviceWorkPlan &operator=(DeviceWorkPlan &&) noexcept;

  // Reuses the fixed device allocation. Async updates rotate through two
  // pinned host images. waitOn() establishes visibility; markConsumed() then
  // publishes a consumer fence, and a later upload on any stream waits for
  // every consumer fence before overwriting the allocation. Plans are
  // structural graph inputs and must be uploaded before graph capture.
  void upload(const WorkPlan &plan);
  void uploadAsync(const WorkPlan &plan, cudaStream_t stream);
  // Enqueue the publication fence on a consumer stream before launching
  // kernels that read this plan. Call markConsumed() after the last such
  // kernel; the next upload then cannot overwrite the plan early.
  void waitOn(cudaStream_t stream) const;
  void markConsumed(cudaStream_t stream) const;
  void synchronizeUpload() const;

  [[nodiscard]] const abi::WorkItem *workItems() const noexcept;
  [[nodiscard]] const abi::AcquireRequirement *dependencies() const noexcept;
  [[nodiscard]] std::uint32_t workItemCount() const noexcept;
  [[nodiscard]] std::uint32_t dependencyCount() const noexcept;
  [[nodiscard]] std::uint32_t workItemCapacity() const noexcept;
  [[nodiscard]] std::uint32_t dependencyCapacity() const noexcept;
  [[nodiscard]] int deviceOrdinal() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace nta
