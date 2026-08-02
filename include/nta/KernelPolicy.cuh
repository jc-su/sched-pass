#pragma once

#include "nta/DeviceAPI.cuh"
#include "nta/TicketProtocol.cuh"

#include <cstdint>

namespace nta::kernel {

// Request identity needed by a single-object acquisition site. The compiler
// inlines these helpers and proves the resulting marker branch at the kernel
// entry; this type does not add a device-side ABI object.
struct BoundRequest {
  std::uint32_t requestSlot;
  std::uint32_t generation;
  std::uint32_t workTicket;
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
prepareWorkTicket(abi::RuntimeView *runtime, const abi::WorkItem &item) {
  if (runtime == nullptr || item.workTicket >= runtime->workTicketCapacity ||
      item.reductionGroup >= runtime->workTicketCapacity ||
      item.contributorCount == 0 ||
      item.contributorIndex >= item.contributorCount) {
    device::failWorkTicket(runtime, item.workTicket,
                           abi::WorkTicketState::Failed);
    return false;
  }
  if (threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) {
    return true;
  }
  abi::WorkTicket &ticket = runtime->workTickets[item.workTicket];
  if (atomicAdd(&ticket.state, 0U) ==
      static_cast<std::uint32_t>(abi::WorkTicketState::New)) {
    ticket.estimatedComputeNs = item.estimatedComputeNs;
    ticket.reductionGroup = item.reductionGroup;
    ticket.contributorCount = item.contributorCount;
  }
  return true;
}

[[nodiscard]] __device__ __forceinline__ bool
acquireWork(abi::RuntimeView *runtime, const abi::WorkItem *workItems,
            const abi::AcquireRequirement *dependencies,
            std::uint32_t workIndex, WorkContext &context) {
  context.item = workItems[workIndex];
  context.dependencies = dependencies + context.item.dependencyBegin;
  __nta_bind_request(context.item.requestSlot, context.item.generation);
  if (!prepareWorkTicket(runtime, context.item)) {
    return false;
  }
  return __nta_acquire_set_marker(
      runtime, context.dependencies, context.item.dependencyCount,
      context.item.directDependencyCount, context.item.workTicket);
}

// Engine adapters may cache a structural plan across request-slot
// generations. The caller must first validate runtime and requestSlot; this
// variant binds the generation currently published in the request directory.
[[nodiscard]] __device__ __forceinline__ bool
acquireCurrentWork(abi::RuntimeView *runtime, const abi::WorkItem *workItems,
                   const abi::AcquireRequirement *dependencies,
                   std::uint32_t workIndex, WorkContext &context) {
  context.item = workItems[workIndex];
  context.item.generation =
      runtime->requests[context.item.requestSlot].generation;
  context.dependencies = dependencies + context.item.dependencyBegin;
  __nta_bind_request(context.item.requestSlot, context.item.generation);
  if (!prepareWorkTicket(runtime, context.item)) {
    return false;
  }
  return __nta_acquire_set_marker(
      runtime, context.dependencies, context.item.dependencyCount,
      context.item.directDependencyCount, context.item.workTicket);
}

// Pre-acquired engine batches use requestIndex as a compact runtime slot. The
// dependency event is ordered before the application kernel, so the compiler
// site lowers to a request-liveness guard without allocating a work ticket.
[[nodiscard]] __device__ __forceinline__ bool
acquireCurrentRequest(abi::RuntimeView *runtime, std::uint32_t requestIndex,
                      WorkContext &context) {
  context.item = {requestIndex,
                  requestIndex,
                  runtime->requests[requestIndex].generation,
                  0,
                  0,
                  0,
                  0,
                  abi::InvalidIndex};
  context.dependencies = nullptr;
  __nta_bind_request(requestIndex, context.item.generation);
  return __nta_acquire_set_marker(runtime, nullptr, 0, 0, abi::InvalidIndex);
}

__device__ __forceinline__ void defer(abi::RuntimeView *runtime,
                                      const WorkContext &context) {
  __nta_defer_marker(runtime, context.item.workTicket);
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
      requirement.offset, requirement.bytes, request.workTicket);
}

[[nodiscard]] __device__ __forceinline__ void *
acquireTensorMap(abi::RuntimeView *runtime, const BoundRequest &request,
                 const abi::AcquireRequirement &requirement) {
  __nta_bind_request(request.requestSlot, request.generation);
  return __nta_acquire_tensor_map_marker(
      runtime, reinterpret_cast<const void *>(requirement.directTensorMap),
      requirement.objectSlot, requirement.objectId, requirement.objectVersion,
      requirement.offset, requirement.bytes, request.workTicket);
}

__device__ __forceinline__ void defer(abi::RuntimeView *runtime,
                                      const BoundRequest &request) {
  __nta_defer_marker(runtime, request.workTicket);
}

} // namespace nta::kernel
