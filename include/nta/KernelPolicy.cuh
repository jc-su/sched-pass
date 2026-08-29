#pragma once

#include "nta/DeviceAPI.cuh"
#include "nta/TicketProtocol.cuh"

#include <cstdint>

#ifndef NTA_FLASHINFER_STREAM_ORDERED_DIRECT
#define NTA_FLASHINFER_STREAM_ORDERED_DIRECT 0
#endif

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
  const bool valid =
      runtime != nullptr && item.workTicket < runtime->workTicketCapacity &&
      item.reductionGroup < runtime->workTicketCapacity &&
      item.contributorCount != 0 &&
      item.contributorIndex < item.contributorCount;
  if (!valid) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
      device::failWorkTicket(runtime, item.workTicket,
                             abi::WorkTicketState::Failed);
    }
    return false;
  }

  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    abi::WorkTicket &ticket = runtime->workTickets[item.workTicket];
    if (atomicAdd(&ticket.state, 0U) ==
        static_cast<std::uint32_t>(abi::WorkTicketState::New)) {
      ticket.estimatedComputeNs = item.estimatedComputeNs;
      ticket.reductionGroup = item.reductionGroup;
      ticket.contributorCount = item.contributorCount;
    }
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
#if NTA_FLASHINFER_STREAM_ORDERED_DIRECT
  context.dependencies = nullptr;
#else
  context.dependencies = dependencies + context.item.dependencyBegin;
#endif
  __nta_bind_request(context.item.requestSlot, context.item.generation);
  if (!prepareWorkTicket(runtime, context.item)) {
    return false;
  }
#if NTA_FLASHINFER_STREAM_ORDERED_DIRECT
  return __nta_acquire_set_marker(runtime, nullptr, 0, 0,
                                  context.item.workTicket);
#else
  return __nta_acquire_set_marker(
      runtime, context.dependencies, context.item.dependencyCount,
      context.item.directDependencyCount, context.item.workTicket);
#endif
}

// Bind a current-generation structural plan through one compiler-visible
// acquisition site. Work selected from RuntimeView::readyWorkTickets has
// already had its exact dependency set admitted by discovery/publication, so a
// zero active count consumes that proof without traversing the cone again. The
// original requirements pointer remains an operand, preserving provenance for
// any requirement-address use. A non-published launch retains the full count.
//
// The single marker is intentional: it gives the verifier one dominating
// request guard for the numerical/partial region instead of two conditional
// acquisition sites. New runnable work is valid because discovery publishes it
// only after proving every dependency ready; commitPartial installs its ticket
// identity at retirement, as before.
[[nodiscard]] __device__ __forceinline__ bool
acquireCurrentPlannedWork(abi::RuntimeView *runtime,
                          const abi::WorkItem *workItems,
                          const abi::AcquireRequirement *dependencies,
                          std::uint32_t workIndex, bool published,
                          WorkContext &context) {
  context.item = workItems[workIndex];
  context.item.generation =
      runtime->requests[context.item.requestSlot].generation;
#if NTA_FLASHINFER_STREAM_ORDERED_DIRECT
  context.dependencies = nullptr;
#else
  context.dependencies = dependencies + context.item.dependencyBegin;
#endif
  __nta_bind_request(context.item.requestSlot, context.item.generation);
#if NTA_FLASHINFER_STREAM_ORDERED_DIRECT
  // This form is admitted only for one finite, fully published ready window.
  // Discovery already validated every exact dependency and the following
  // same-stream retirement kernel owns ticket initialization/completion. Do
  // not re-read or mutate ticket state in every numerical CTA: that would
  // violate the stream-ordered ownership contract.
  // launchWorkIndex has already checked the runtime ABI, queue bounds, and
  // work-ticket index. The uploaded typed plan owns the remaining structural
  // fields; adding a second conditional here would create a compiler-visible
  // path around the numerical acquisition region.
  (void)dependencies;
  (void)published;
  return __nta_acquire_set_marker(runtime, nullptr, 0, 0,
                                  context.item.workTicket);
#else
  if (!prepareWorkTicket(runtime, context.item)) {
    return false;
  }
  const std::uint32_t dependencyCount =
      published ? 0U : context.item.dependencyCount;
  const std::uint32_t directDependencyCount =
      published ? 0U : context.item.directDependencyCount;
  return __nta_acquire_set_marker(runtime, context.dependencies,
                                  dependencyCount, directDependencyCount,
                                  context.item.workTicket);
#endif
}

// A stream-ordered acquisition event can satisfy the data dependency while the
// structural work plan still supplies the exact CTA-to-request mapping. Keep
// the compiler-visible request guard, but do not rediscover dependencies or
// mutate work-ticket state for this launch.
__device__ __forceinline__ void
preparePreacquiredWork(const abi::WorkItem *workItems,
                       const abi::AcquireRequirement *dependencies,
                       std::uint32_t workIndex, WorkContext &context) {
  context.item = workItems[workIndex];
  context.dependencies =
      dependencies == nullptr ? nullptr
                              : dependencies + context.item.dependencyBegin;
}

[[nodiscard]] __device__ __forceinline__ bool
acquirePreacquiredWork(abi::RuntimeView *runtime, WorkContext &context) {
  context.item.generation =
      runtime->requests[context.item.requestSlot].generation;
  context.item.workTicket = abi::InvalidIndex;
  context.item.reductionGroup = abi::InvalidIndex;
  context.item.contributorIndex = 0;
  context.item.contributorCount = 0;
  context.item.estimatedComputeNs = 0;
  __nta_bind_request(context.item.requestSlot, context.item.generation);
  return __nta_acquire_set_marker(runtime, nullptr, 0, 0, abi::InvalidIndex);
}

// Verify event-published resource identity before crossing the single
// compiler-visible request acquisition edge. The marker remains the final
// numerical guard, so its ready edge directly dominates the partial region;
// stale metadata still fails closed before any staged address is consumed.
[[nodiscard]] __device__ __forceinline__ bool
validatePreacquiredWork(abi::RuntimeView *runtime,
                        const WorkContext &context) {
  const bool hasExternal =
      context.item.directDependencyCount < context.item.dependencyCount;
  if (!hasExternal) {
    return true;
  }
  if (context.dependencies == nullptr) {
    return false;
  }
  __shared__ std::uint32_t exactReady;
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    exactReady = 1U;
    // directDependencyCount is accounting metadata, not an ordering promise.
    // Validate every externally acquired requirement so a future plan layout
    // cannot accidentally hide a stale object behind an interleaved direct
    // requirement.
    for (std::uint32_t index = 0; index < context.item.dependencyCount;
         ++index) {
      const abi::AcquireRequirement &requirement = context.dependencies[index];
      if (requirement.directBase != 0) {
        continue;
      }
      if (requirement.objectSlot >= runtime->objectCapacity) {
        exactReady = 0U;
        break;
      }
      const abi::ObjectEntry &object =
          runtime->objects[requirement.objectSlot];
      const auto state = static_cast<abi::ObjectState>(atomicAdd(
          const_cast<std::uint32_t *>(&object.state), 0U));
      if (object.objectId != requirement.objectId ||
          object.version != requirement.objectVersion ||
          state != abi::ObjectState::Ready ||
          requirement.offset > object.bytes ||
          requirement.bytes > object.bytes - requirement.offset) {
        exactReady = 0U;
        break;
      }
    }
    if (exactReady == 0U) {
      device::recordFailure(runtime);
    }
  }
  __syncthreads();
  if (exactReady == 0U) {
    return false;
  }
  return true;
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

__device__ __forceinline__ void beginPartial(abi::RuntimeView *runtime,
                                             const WorkContext &context) {
  __nta_bind_request(context.item.requestSlot, context.item.generation);
  __nta_begin_partial_marker(runtime, context.item.workTicket);
}

// Publish one request-owned numerical partial. The compiler turns this marker
// into the generation-checked ticket/reduction protocol; callers do not update
// completion counters directly.
__device__ __forceinline__ void commitPartial(abi::RuntimeView *runtime,
                                              const WorkContext &context) {
  __nta_bind_request(context.item.requestSlot, context.item.generation);
#if NTA_FLASHINFER_STREAM_ORDERED_DIRECT
  __nta_commit_stream_ordered_partial_marker(
      runtime, context.item.workTicket, context.item.reductionGroup,
      context.item.contributorIndex, context.item.contributorCount,
      context.item.estimatedComputeNs);
#else
  __nta_commit_partial_marker(
      runtime, context.item.workTicket, context.item.reductionGroup,
      context.item.contributorIndex, context.item.contributorCount,
      context.item.estimatedComputeNs);
#endif
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
