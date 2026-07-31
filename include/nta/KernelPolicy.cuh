#pragma once

#include "nta/DeviceAPI.cuh"

#include <cstdint>

namespace nta::kernel {

// Request identity needed by a single-object acquisition site. The compiler
// inlines these helpers and proves the resulting marker branch at the kernel
// entry; this type does not add a device-side ABI object.
struct BoundRequest {
  std::uint32_t requestSlot;
  std::uint32_t generation;
  std::uint32_t continuation;
};

// Per-CTA view over one canonical work item. A kernel keeps its own numerical
// parameters; this context carries only request and acquisition semantics.
struct WorkContext {
  abi::WorkItem item;
  const abi::AcquireRequirement *dependencies;

  [[nodiscard]] __device__ __forceinline__ const abi::AcquireRequirement *
  requirement(std::uint32_t index) const {
    return index < item.dependencyCount ? dependencies + index : nullptr;
  }
};

[[nodiscard]] __device__ __forceinline__ bool
acquireWork(abi::RuntimeView *runtime, const abi::WorkItem *workItems,
            const abi::AcquireRequirement *dependencies,
            std::uint32_t workIndex, WorkContext &context) {
  context.item = workItems[workIndex];
  context.dependencies = dependencies + context.item.dependencyBegin;
  __nta_bind_request(context.item.requestSlot, context.item.generation);
  return __nta_acquire_set_marker(
      runtime, context.dependencies, context.item.dependencyCount,
      context.item.directDependencyCount, context.item.continuation);
}

__device__ __forceinline__ void defer(abi::RuntimeView *runtime,
                                      const WorkContext &context) {
  __nta_defer_marker(runtime, context.item.continuation);
}

[[nodiscard]] __device__ __forceinline__ const void *
address(abi::RuntimeView *runtime, const WorkContext &context,
        std::uint32_t dependencyIndex) {
  return nta_requirement_address(runtime, context.requirement(dependencyIndex));
}

[[nodiscard]] __device__ __forceinline__ const void *
tensorMap(abi::RuntimeView *runtime, const WorkContext &context,
          std::uint32_t dependencyIndex) {
  return nta_requirement_tensor_map(runtime,
                                    context.requirement(dependencyIndex));
}

[[nodiscard]] __device__ __forceinline__ void *
acquireAddress(abi::RuntimeView *runtime, const BoundRequest &request,
               const abi::AcquireRequirement &requirement) {
  __nta_bind_request(request.requestSlot, request.generation);
  return __nta_acquire_marker(
      runtime, reinterpret_cast<const void *>(requirement.directBase),
      requirement.objectSlot, requirement.objectId, requirement.objectVersion,
      requirement.offset, requirement.bytes, request.continuation);
}

[[nodiscard]] __device__ __forceinline__ void *
acquireTensorMap(abi::RuntimeView *runtime, const BoundRequest &request,
                 const abi::AcquireRequirement &requirement) {
  __nta_bind_request(request.requestSlot, request.generation);
  return __nta_acquire_tensor_map_marker(
      runtime, reinterpret_cast<const void *>(requirement.directTensorMap),
      requirement.objectSlot, requirement.objectId, requirement.objectVersion,
      requirement.offset, requirement.bytes, request.continuation);
}

__device__ __forceinline__ void defer(abi::RuntimeView *runtime,
                                      const BoundRequest &request) {
  __nta_defer_marker(runtime, request.continuation);
}

} // namespace nta::kernel
