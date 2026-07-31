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
  explicit DeviceWorkPlan(const WorkPlan &plan);
  DeviceWorkPlan(std::uint32_t workItemCapacity,
                 std::uint32_t dependencyCapacity);
  ~DeviceWorkPlan();

  DeviceWorkPlan(const DeviceWorkPlan &) = delete;
  DeviceWorkPlan &operator=(const DeviceWorkPlan &) = delete;
  DeviceWorkPlan(DeviceWorkPlan &&) noexcept;
  DeviceWorkPlan &operator=(DeviceWorkPlan &&) noexcept;

  // Reuses the fixed device allocation. Async updates stage through pinned
  // host memory; consumers on another stream must call waitOn().
  void upload(const WorkPlan &plan);
  void uploadAsync(const WorkPlan &plan, cudaStream_t stream);
  void waitOn(cudaStream_t stream) const;
  void synchronizeUpload() const;

  [[nodiscard]] const abi::WorkItem *workItems() const noexcept;
  [[nodiscard]] const abi::AcquireRequirement *dependencies() const noexcept;
  [[nodiscard]] std::uint32_t workItemCount() const noexcept;
  [[nodiscard]] std::uint32_t dependencyCount() const noexcept;
  [[nodiscard]] std::uint32_t workItemCapacity() const noexcept;
  [[nodiscard]] std::uint32_t dependencyCapacity() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace nta
