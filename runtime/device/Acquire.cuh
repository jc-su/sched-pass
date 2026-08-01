#pragma once

#include "nta/RuntimeABI.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#ifndef NTA_DEVICE_PHASE_KERNELS
#define NTA_DEVICE_PHASE_KERNELS 1
#endif

namespace nta::device {

__device__ __forceinline__ bool requestLive(abi::RuntimeView *runtime,
                                            std::uint32_t requestSlot,
                                            std::uint32_t generation) {
  if (runtime == nullptr || runtime->abiVersion != abi::Version ||
      requestSlot >= runtime->requestCapacity) {
    return false;
  }
  const abi::RequestContext &request = runtime->requests[requestSlot];
  return request.generation == generation && request.cancelled == 0;
}

__device__ __forceinline__ void failContinuation(abi::RuntimeView *runtime,
                                                 std::uint32_t continuation,
                                                 abi::ContinuationState state) {
  if (runtime != nullptr && continuation < runtime->continuationCapacity &&
      threadIdx.x == 0) {
    atomicExch(&runtime->continuations[continuation].state,
               static_cast<std::uint32_t>(state));
  }
}

__device__ __forceinline__ bool
dependencyRange(abi::RuntimeView *runtime, std::uint32_t continuation,
                std::uint32_t dependencyCount, std::uint32_t &dependencyStart) {
  if (runtime == nullptr || runtime->dependencies == nullptr ||
      continuation >= runtime->continuationCapacity || dependencyCount == 0 ||
      dependencyCount > runtime->maxDependenciesPerContinuation ||
      runtime->maxDependenciesPerContinuation == 0 ||
      continuation > runtime->dependencyCapacity /
                         runtime->maxDependenciesPerContinuation) {
    return false;
  }
  dependencyStart = continuation * runtime->maxDependenciesPerContinuation;
  return dependencyStart <= runtime->dependencyCapacity &&
         dependencyCount <= runtime->dependencyCapacity - dependencyStart;
}

__device__ __forceinline__ bool
initializeContinuation(abi::RuntimeView *runtime, std::uint32_t requestSlot,
                       std::uint32_t generation, std::uint32_t continuation,
                       const abi::AcquireRequirement *requirements,
                       std::uint32_t requirementCount) {
  std::uint32_t dependencyStart = 0;
  if (!dependencyRange(runtime, continuation, requirementCount,
                       dependencyStart) ||
      requirements == nullptr || requestSlot >= runtime->requestCapacity ||
      runtime->pendingContinuations == nullptr ||
      runtime->pendingCount == nullptr) {
    failContinuation(runtime, continuation, abi::ContinuationState::Failed);
    return false;
  }

  abi::Continuation &record = runtime->continuations[continuation];
  const auto state = static_cast<abi::ContinuationState>(atomicCAS(
      &record.state, static_cast<std::uint32_t>(abi::ContinuationState::New),
      static_cast<std::uint32_t>(abi::ContinuationState::Pending)));
  if (state == abi::ContinuationState::Cancelled ||
      state == abi::ContinuationState::Failed) {
    return false;
  }
  if (state != abi::ContinuationState::New) {
    return true;
  }

  for (std::uint32_t index = 0; index < requirementCount; ++index) {
    runtime->dependencies[dependencyStart + index] = {
        requirements[index].objectId,
        requirements[index].objectSlot,
        requirements[index].objectVersion,
    };
  }
  const abi::RequestContext &request = runtime->requests[requestSlot];
  record.requestId = request.requestId;
  record.requestSlot = requestSlot;
  record.generation = generation;
  record.dependencyCount = requirementCount;
  record.logicalTile = continuation;
  record.dependencyStart = dependencyStart;
  __threadfence();
  const std::uint32_t ticket = atomicAdd(runtime->pendingCount, 1U);
  if (ticket < runtime->continuationCapacity) {
    runtime->pendingContinuations[ticket] = continuation;
    __threadfence();
  } else {
    atomicExch(&record.state,
               static_cast<std::uint32_t>(abi::ContinuationState::Failed));
    return false;
  }
  return true;
}

__device__ __forceinline__ std::uint32_t
loadIoCoherent(const std::uint32_t *address) {
  std::uint32_t value;
  asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(value) : "l"(address));
  return value;
}

__device__ __forceinline__ void systemIoFence() {
  asm volatile("membar.sys;" ::: "memory");
}

__device__ __forceinline__ std::uint64_t globalTimerNs() {
  std::uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ __forceinline__ bool tryReserveCounter(std::uint64_t *counter,
                                                  std::uint64_t maximum,
                                                  std::uint64_t bytes) {
  if (bytes > maximum) {
    return false;
  }
  auto *outstanding = reinterpret_cast<unsigned long long *>(counter);
  const unsigned long long previous = atomicAdd(outstanding, bytes);
  if (previous <= maximum - bytes) {
    return true;
  }
  atomicAdd(outstanding, 0ULL - static_cast<unsigned long long>(bytes));
  return false;
}

__device__ __forceinline__ void releaseCounter(std::uint64_t *counter,
                                               std::uint64_t bytes) {
  if (bytes != 0) {
    atomicAdd(reinterpret_cast<unsigned long long *>(counter),
              0ULL - static_cast<unsigned long long>(bytes));
  }
}

__device__ __forceinline__ bool
tryReserveRequestBytes(abi::RuntimeView *runtime, std::uint32_t requestSlot,
                       std::uint32_t generation, std::uint64_t bytes) {
  if (requestSlot >= runtime->requestCapacity) {
    return false;
  }
  abi::RequestContext &request = runtime->requests[requestSlot];
  return request.generation == generation &&
         tryReserveCounter(&request.outstandingBytes,
                           request.maxOutstandingBytes, bytes);
}

__device__ __forceinline__ void releaseRequestBytes(abi::RuntimeView *runtime,
                                                    std::uint32_t requestSlot,
                                                    std::uint32_t generation,
                                                    std::uint64_t bytes) {
  if (bytes == 0 || requestSlot >= runtime->requestCapacity) {
    return;
  }
  abi::RequestContext &request = runtime->requests[requestSlot];
  if (request.generation == generation) {
    releaseCounter(&request.outstandingBytes, bytes);
  }
}

__device__ __forceinline__ bool tryReserveTenantBytes(abi::RuntimeView *runtime,
                                                      std::uint32_t tenantId,
                                                      std::uint64_t bytes) {
  if (tenantId >= runtime->tenantCapacity) {
    return false;
  }
  abi::TenantContext &tenant = runtime->tenants[tenantId];
  return tenant.active != 0 &&
         tryReserveCounter(&tenant.outstandingBytes, tenant.maxOutstandingBytes,
                           bytes);
}

__device__ __forceinline__ void releaseTenantBytes(abi::RuntimeView *runtime,
                                                   std::uint32_t tenantId,
                                                   std::uint64_t bytes) {
  if (bytes != 0 && tenantId < runtime->tenantCapacity) {
    releaseCounter(&runtime->tenants[tenantId].outstandingBytes, bytes);
  }
}

__device__ __forceinline__ std::uint32_t urgencyBucket(std::uint32_t priority,
                                                       std::uint64_t deadline,
                                                       std::uint64_t now) {
  std::uint32_t urgency = priority > 7U ? 7U : priority;
  if (deadline == 0) {
    return urgency;
  }
  const std::uint64_t slack = deadline > now ? deadline - now : 0;
  const std::uint32_t deadlineUrgency = slack <= 50'000ULL      ? 7U
                                        : slack <= 200'000ULL   ? 6U
                                        : slack <= 1'000'000ULL ? 5U
                                        : slack <= 5'000'000ULL ? 4U
                                                                : 0U;
  return urgency > deadlineUrgency ? urgency : deadlineUrgency;
}

__device__ __forceinline__ abi::BackendView *backend(abi::RuntimeView *runtime,
                                                     abi::SourceKind kind) {
  const std::uint32_t index = static_cast<std::uint32_t>(kind);
  if (runtime == nullptr || runtime->backends == nullptr ||
      index >= runtime->backendCapacity) {
    return nullptr;
  }
  abi::BackendView &entry = runtime->backends[index];
  return entry.sourceKind == index ? &entry : nullptr;
}

__device__ __forceinline__ bool
tryReserveBackendBytes(abi::RuntimeView *runtime, abi::SourceKind kind,
                       std::uint64_t bytes) {
  abi::BackendView *entry = backend(runtime, kind);
  return entry != nullptr && entry->active != 0 &&
         tryReserveCounter(&entry->outstandingBytes, entry->maxOutstandingBytes,
                           bytes);
}

__device__ __forceinline__ void releaseBackendBytes(abi::RuntimeView *runtime,
                                                    abi::SourceKind kind,
                                                    std::uint64_t bytes) {
  abi::BackendView *entry = backend(runtime, kind);
  if (entry != nullptr) {
    releaseCounter(&entry->outstandingBytes, bytes);
  }
}

__device__ __forceinline__ abi::NvmeQueueView *
nvmeQueue(abi::RuntimeView *runtime) {
  abi::BackendView *entry = backend(runtime, abi::SourceKind::Nvme);
  return entry == nullptr || entry->active == 0 || entry->deviceState == 0
             ? nullptr
             : reinterpret_cast<abi::NvmeQueueView *>(entry->deviceState);
}

__device__ __forceinline__ bool
nvmeQueueOnline(const abi::NvmeQueueView &queue) {
  if (queue.control == nullptr)
    return false;
  const abi::NvmeQueueControl &control = *queue.control;
  return loadIoCoherent(&control.magic) == abi::NvmeQueueControlMagic &&
         loadIoCoherent(&control.abiVersion) == abi::NvmeDriverAbiVersion &&
         loadIoCoherent(&control.queueId) == queue.queueId &&
         loadIoCoherent(&control.generation) == queue.queueGeneration &&
         loadIoCoherent(&control.state) ==
             static_cast<std::uint32_t>(abi::NvmeQueueState::Online);
}

__device__ __forceinline__ void failNvmeQueue(abi::RuntimeView *runtime,
                                              abi::NvmeQueueView &queue,
                                              std::uint32_t lane,
                                              std::uint32_t error) {
  for (std::uint32_t commandId = lane; commandId < queue.depth;
       commandId += warpSize) {
    abi::NvmeCommandContext &stored = queue.contexts[commandId];
    if (atomicExch(&stored.active, 0U) == 0U)
      continue;
    const abi::NvmeCommandContext context = stored;
    if (context.objectSlot < runtime->objectCapacity) {
      atomicExch(&runtime->objects[context.objectSlot].state,
                 static_cast<std::uint32_t>(abi::ObjectState::Failed));
    }
    if (context.continuation < runtime->continuationCapacity) {
      atomicExch(&runtime->continuations[context.continuation].state,
                 static_cast<std::uint32_t>(abi::ContinuationState::Failed));
    }
    atomicAdd(reinterpret_cast<unsigned long long *>(&queue.failed), 1ULL);
    releaseRequestBytes(runtime, context.requestSlot, context.generation,
                        context.bytes);
    releaseTenantBytes(runtime, context.tenantId, context.bytes);
    releaseBackendBytes(runtime, abi::SourceKind::Nvme, context.backendBytes);
  }
  __syncwarp();
  if (lane == 0) {
    queue.outstanding = 0;
    queue.active = 0;
    queue.error = error;
  }
}

__device__ __forceinline__ const abi::ReplicaEntry *
replica(abi::RuntimeView *runtime, const abi::ObjectEntry &object,
        std::uint32_t relativeIndex) {
  if (runtime->replicas == nullptr || relativeIndex >= object.replicaCount ||
      object.replicaStart > runtime->replicaCapacity ||
      relativeIndex >= runtime->replicaCapacity - object.replicaStart) {
    return nullptr;
  }
  return &runtime->replicas[object.replicaStart + relativeIndex];
}

__device__ __forceinline__ std::uint64_t
replicaReadyCost(const abi::ReplicaEntry &replica, std::uint64_t bytes) {
  if (replica.estimatedBandwidthBytesPerSecond == 0) {
    return UINT64_MAX;
  }
  const std::uint64_t transfer =
      bytes > UINT64_MAX / 1'000'000'000ULL
          ? UINT64_MAX
          : bytes * 1'000'000'000ULL / replica.estimatedBandwidthBytesPerSecond;
  return replica.estimatedLatencyNs > UINT64_MAX - transfer
             ? UINT64_MAX
             : replica.estimatedLatencyNs + transfer;
}

__device__ __forceinline__ std::uint64_t
loadCounter(const std::uint64_t *counter) {
  return atomicAdd(reinterpret_cast<unsigned long long *>(
                       const_cast<std::uint64_t *>(counter)),
                   0ULL);
}

__device__ __forceinline__ bool reserveIntent(abi::RuntimeView *runtime,
                                              std::uint32_t key,
                                              std::uint32_t &slotIndex,
                                              abi::IntentSlot *&slot) {
  if (runtime->intentPool == nullptr || runtime->intents == nullptr ||
      runtime->intentPool->capacity == 0 ||
      runtime->intentPool->capacity > runtime->intentCapacity) {
    return false;
  }
  abi::IntentPool &pool = *runtime->intentPool;
  for (std::uint32_t probe = 0; probe < pool.capacity; ++probe) {
    const std::uint32_t candidate = (key + probe) % pool.capacity;
    abi::IntentSlot &candidateSlot = runtime->intents[candidate];
    if (atomicCAS(&candidateSlot.intent.valid, 0U, 2U) == 0U) {
      slotIndex = candidate;
      slot = &candidateSlot;
      return true;
    }
  }
  atomicAdd(&pool.overflow, 1U);
  return false;
}

__device__ __forceinline__ void publishIntent(abi::RuntimeView *runtime,
                                              abi::IntentSlot &slot) {
  __threadfence();
  atomicAdd(
      reinterpret_cast<unsigned long long *>(&runtime->intentPool->enqueued),
      1ULL);
  atomicAdd(&runtime->intentPool->active, 1U);
  __threadfence();
  atomicExch(&slot.intent.valid, 1U);
}

__device__ __forceinline__ bool claimIntent(abi::IntentSlot &slot) {
  return atomicCAS(&slot.intent.valid, 1U, 2U) == 1U;
}

__device__ __forceinline__ void consumeIntent(abi::RuntimeView *runtime,
                                              abi::IntentSlot &slot) {
  atomicAdd(reinterpret_cast<unsigned long long *>(&slot.sequence), 1ULL);
  __threadfence();
  atomicExch(&slot.intent.valid, 0U);
  atomicAdd(
      reinterpret_cast<unsigned long long *>(&runtime->intentPool->consumed),
      1ULL);
  atomicSub(&runtime->intentPool->active, 1U);
}

} // namespace nta::device

extern "C" __device__ __forceinline__ __attribute__((used)) bool
nta_request_live(nta::abi::RuntimeView *runtime, std::uint32_t requestSlot,
                 std::uint32_t generation) {
  return nta::device::requestLive(runtime, requestSlot, generation);
}

extern "C" __device__ __attribute__((used, noinline)) void *
nta_acquire_slow(nta::abi::RuntimeView *runtime, std::uint32_t requestSlot,
                 std::uint32_t generation, std::uint32_t objectSlot,
                 std::uint64_t objectId, std::uint32_t objectVersion,
                 std::uint64_t offset, std::uint32_t bytes,
                 std::uint32_t continuation) {
  using namespace nta;
  if (!device::requestLive(runtime, requestSlot, generation)) {
    device::failContinuation(runtime, continuation,
                             abi::ContinuationState::Cancelled);
    return nullptr;
  }
  if (objectSlot >= runtime->objectCapacity ||
      continuation >= runtime->continuationCapacity) {
    device::failContinuation(runtime, continuation,
                             abi::ContinuationState::Failed);
    return nullptr;
  }

  abi::ObjectEntry &object = runtime->objects[objectSlot];
  if (object.objectId != objectId || object.version != objectVersion ||
      offset > object.bytes || bytes > object.bytes - offset) {
    device::failContinuation(runtime, continuation,
                             abi::ContinuationState::Failed);
    return nullptr;
  }

  const auto state =
      static_cast<abi::ObjectState>(atomicAdd(&object.state, 0U));
  std::uint32_t selectedReplica = abi::InvalidIndex;
  std::uint64_t selectedCost = UINT64_MAX;
  std::uint32_t directReplica = abi::InvalidIndex;
  std::uint64_t directCost = UINT64_MAX;
  for (std::uint32_t replicaIndex = 0; replicaIndex < object.replicaCount;
       ++replicaIndex) {
    const abi::ReplicaEntry *candidate =
        device::replica(runtime, object, replicaIndex);
    if (candidate == nullptr ||
        candidate->backendIndex >= runtime->backendCapacity) {
      continue;
    }
    const auto kind = static_cast<abi::SourceKind>(candidate->sourceKind);
    abi::BackendView *candidateBackend = device::backend(runtime, kind);
    if (candidateBackend == nullptr || candidateBackend->active == 0) {
      continue;
    }
    const bool direct = (candidate->flags & abi::ReplicaDirect) != 0 &&
                        candidate->sourceAddress != 0;
    const bool transport = (candidate->flags & abi::ReplicaTransport) != 0;
    const std::uint64_t cost = device::replicaReadyCost(*candidate, bytes);
    if (direct && (directReplica == abi::InvalidIndex || cost < directCost)) {
      directReplica = replicaIndex;
      directCost = cost;
    } else if (transport &&
               (selectedReplica == abi::InvalidIndex || cost < selectedCost)) {
      selectedReplica = replicaIndex;
      selectedCost = cost;
    }
  }
  if (directReplica != abi::InvalidIndex) {
    const abi::ReplicaEntry *direct =
        device::replica(runtime, object, directReplica);
    return reinterpret_cast<std::byte *>(direct->sourceAddress) + offset;
  }
  const abi::ReplicaEntry *selected =
      selectedReplica == abi::InvalidIndex
          ? nullptr
          : device::replica(runtime, object, selectedReplica);
  if (selected == nullptr || object.stagingAddress == 0 ||
      (selected->sourceKind ==
           static_cast<std::uint32_t>(abi::SourceKind::HostStaged) &&
       selected->sourceAddress == 0) ||
      (selected->sourceKind ==
           static_cast<std::uint32_t>(abi::SourceKind::Nvme) &&
       (device::nvmeQueue(runtime) == nullptr ||
        selected->dmaPageListAddress == 0 || selected->dmaPageCount == 0))) {
    device::failContinuation(runtime, continuation,
                             abi::ContinuationState::Failed);
    return nullptr;
  }
  // Directory entries are acquisition tiles. A staged transfer owns
  // the whole tile, so duplicate suppression cannot alias different ranges.
  if (offset != 0 || bytes != object.bytes) {
    device::failContinuation(runtime, continuation,
                             abi::ContinuationState::Failed);
    return nullptr;
  }
  if (state == abi::ObjectState::Ready) {
    return reinterpret_cast<std::byte *>(object.stagingAddress) + offset;
  }
  if (state == abi::ObjectState::Failed) {
    device::failContinuation(runtime, continuation,
                             abi::ContinuationState::Failed);
    return nullptr;
  }

  if (threadIdx.x == 0) {
    abi::Continuation &record = runtime->continuations[continuation];
    const auto continuationState =
        static_cast<abi::ContinuationState>(atomicAdd(&record.state, 0U));
    if (continuationState == abi::ContinuationState::New) {
      const abi::AcquireRequirement requirement{
          0, 0, objectId, offset, objectSlot, objectVersion, bytes, 0};
      (void)device::initializeContinuation(runtime, requestSlot, generation,
                                           continuation, &requirement, 1);
    }
  }

  if (threadIdx.x == 0 &&
      atomicCAS(&object.state,
                static_cast<std::uint32_t>(abi::ObjectState::New),
                static_cast<std::uint32_t>(abi::ObjectState::Queued)) ==
          static_cast<std::uint32_t>(abi::ObjectState::New)) {
    object.selectedReplica = selectedReplica;
    std::uint32_t intentIndex = abi::InvalidIndex;
    abi::IntentSlot *intentSlot = nullptr;
    if (device::reserveIntent(runtime, objectSlot, intentIndex, intentSlot)) {
      (void)intentIndex;
      abi::AcquireIntent &intent = intentSlot->intent;
      intent.objectId = objectId;
      intent.offset = offset;
      intent.bytes = bytes;
      intent.requestSlot = requestSlot;
      intent.generation = generation;
      intent.objectSlot = objectSlot;
      intent.objectVersion = objectVersion;
      intent.continuation = continuation;
      const abi::RequestContext &request = runtime->requests[requestSlot];
      intent.priority = request.priority;
      intent.tenantId = request.tenantId;
      intent.deadlineClock = request.deadlineClock;
      atomicAdd(reinterpret_cast<unsigned long long *>(&object.issueCount),
                1ULL);
      device::publishIntent(runtime, *intentSlot);
    } else {
      atomicExch(&object.state,
                 static_cast<std::uint32_t>(abi::ObjectState::Failed));
      device::failContinuation(runtime, continuation,
                               abi::ContinuationState::Failed);
    }
  }
  return nullptr;
}

extern "C" __device__ __attribute__((used, noinline)) void *
nta_acquire_tensor_map_slow(nta::abi::RuntimeView *runtime,
                            std::uint32_t requestSlot, std::uint32_t generation,
                            std::uint32_t objectSlot, std::uint64_t objectId,
                            std::uint32_t objectVersion, std::uint64_t offset,
                            std::uint32_t bytes, std::uint32_t continuation) {
  void *address =
      nta_acquire_slow(runtime, requestSlot, generation, objectSlot, objectId,
                       objectVersion, offset, bytes, continuation);
  if (address == nullptr || runtime == nullptr ||
      objectSlot >= runtime->objectCapacity) {
    return nullptr;
  }

  nta::abi::ObjectEntry &object = runtime->objects[objectSlot];
  const auto *byteAddress = static_cast<const std::byte *>(address);
  const auto *stagingAddress =
      reinterpret_cast<const std::byte *>(object.stagingAddress + offset);
  if (byteAddress == stagingAddress) {
    if (object.stagingTensorMapAddress != 0) {
      return reinterpret_cast<void *>(object.stagingTensorMapAddress);
    }
  } else {
    for (std::uint32_t index = 0; index < object.replicaCount; ++index) {
      const nta::abi::ReplicaEntry *candidate =
          nta::device::replica(runtime, object, index);
      if (candidate != nullptr && candidate->tensorMapAddress != 0 &&
          byteAddress == reinterpret_cast<const std::byte *>(
                             candidate->sourceAddress + offset)) {
        return reinterpret_cast<void *>(candidate->tensorMapAddress);
      }
    }
  }

  nta::device::failContinuation(runtime, continuation,
                                nta::abi::ContinuationState::Failed);
  return nullptr;
}

extern "C" __device__ __forceinline__ __attribute__((used)) bool
nta_acquire_set_slow(nta::abi::RuntimeView *runtime, std::uint32_t requestSlot,
                     std::uint32_t generation,
                     const nta::abi::AcquireRequirement *requirements,
                     std::uint32_t requirementCount,
                     std::uint32_t directRequirementCount,
                     std::uint32_t continuation) {
  using namespace nta;
  // The compiler-emitted request-live guard dominates this internal helper.
  // Transport misses revalidate in nta_acquire_slow before publishing work.
  std::uint32_t dependencyStart = 0;
  if (requirements == nullptr || directRequirementCount > requirementCount ||
      !device::dependencyRange(runtime, continuation, requirementCount,
                               dependencyStart)) {
    device::failContinuation(runtime, continuation,
                             abi::ContinuationState::Failed);
    return false;
  }
  (void)dependencyStart;
  if (directRequirementCount == requirementCount) {
    return true;
  }
  bool allReady = true;
  for (std::uint32_t index = 0; index < requirementCount; ++index) {
    const abi::AcquireRequirement &requirement = requirements[index];
    if (requirement.directBase != 0) {
      continue;
    }
    if (requirement.objectSlot >= runtime->objectCapacity) {
      allReady = false;
      continue;
    }
    abi::ObjectEntry &object = runtime->objects[requirement.objectSlot];
    const auto objectState =
        static_cast<abi::ObjectState>(atomicAdd(&object.state, 0U));
    allReady &= object.objectId == requirement.objectId &&
                object.version == requirement.objectVersion &&
                requirement.offset <= object.bytes &&
                requirement.bytes <= object.bytes - requirement.offset &&
                object.stagingAddress != 0 &&
                objectState == abi::ObjectState::Ready;
  }
  if (allReady) {
    return true;
  }

  const auto continuationState = static_cast<abi::ContinuationState>(
      atomicAdd(&runtime->continuations[continuation].state, 0U));
  if (continuationState == abi::ContinuationState::Cancelled ||
      continuationState == abi::ContinuationState::Failed) {
    return false;
  }

  if (threadIdx.x == 0) {
    (void)device::initializeContinuation(runtime, requestSlot, generation,
                                         continuation, requirements,
                                         requirementCount);
  }

  bool ready = true;
  for (std::uint32_t index = 0; index < requirementCount; ++index) {
    const abi::AcquireRequirement &requirement = requirements[index];
    if (requirement.directBase != 0) {
      continue;
    }
    ready &= nta_acquire_slow(runtime, requestSlot, generation,
                              requirement.objectSlot, requirement.objectId,
                              requirement.objectVersion, requirement.offset,
                              requirement.bytes, continuation) != nullptr;
  }
  return ready;
}

extern "C" __device__ __forceinline__ __attribute__((used)) const void *
nta_requirement_address(nta::abi::RuntimeView *runtime,
                        const nta::abi::AcquireRequirement *requirement) {
  using namespace nta;
  if (requirement == nullptr) {
    return nullptr;
  }
  if (requirement->directBase != 0) {
    return reinterpret_cast<const std::byte *>(requirement->directBase) +
           requirement->offset;
  }
  if (runtime == nullptr ||
      requirement->objectSlot >= runtime->objectCapacity) {
    return nullptr;
  }
  abi::ObjectEntry &object = runtime->objects[requirement->objectSlot];
  const auto objectState =
      static_cast<abi::ObjectState>(atomicAdd(&object.state, 0U));
  if (object.objectId != requirement->objectId ||
      object.version != requirement->objectVersion ||
      objectState != abi::ObjectState::Ready ||
      requirement->offset > object.bytes ||
      requirement->bytes > object.bytes - requirement->offset) {
    return nullptr;
  }
  return reinterpret_cast<const std::byte *>(object.stagingAddress) +
         requirement->offset;
}

extern "C" __device__ __forceinline__ __attribute__((used)) const void *
nta_requirement_tensor_map(nta::abi::RuntimeView *runtime,
                           const nta::abi::AcquireRequirement *requirement) {
  using namespace nta;
  if (requirement == nullptr) {
    return nullptr;
  }
  if (requirement->directTensorMap != 0) {
    return reinterpret_cast<const void *>(requirement->directTensorMap);
  }
  if (runtime == nullptr ||
      requirement->objectSlot >= runtime->objectCapacity) {
    return nullptr;
  }
  abi::ObjectEntry &object = runtime->objects[requirement->objectSlot];
  const auto objectState =
      static_cast<abi::ObjectState>(atomicAdd(&object.state, 0U));
  if (object.objectId != requirement->objectId ||
      object.version != requirement->objectVersion ||
      objectState != abi::ObjectState::Ready ||
      object.stagingTensorMapAddress == 0) {
    return nullptr;
  }
  return reinterpret_cast<const void *>(object.stagingTensorMapAddress);
}

#if NTA_DEVICE_PHASE_KERNELS
extern "C" __global__ void nta_progress_nvme(nta::abi::RuntimeView *runtime,
                                             std::uint32_t issueBudget,
                                             std::uint32_t completionBudget) {
  using namespace nta;
  if (runtime == nullptr || blockIdx.x != 0 || threadIdx.x >= warpSize) {
    return;
  }
  abi::NvmeQueueView *queuePointer = device::nvmeQueue(runtime);
  if (queuePointer == nullptr) {
    return;
  }
  const std::uint32_t lane = threadIdx.x;
  abi::NvmeQueueView &queue = *queuePointer;
  if (queue.active == 0 || queue.depth < 2 || queue.controllerPageSize == 0) {
    return;
  }
  bool queueOnline = lane == 0 && device::nvmeQueueOnline(queue);
  queueOnline = __shfl_sync(0xffffffffU, queueOnline, 0);
  if (!queueOnline) {
    device::failNvmeQueue(runtime, queue, lane, 0xfffffffcU);
    return;
  }

  if (lane == 0) {
    std::uint32_t drained = 0;
    while (drained < completionBudget && queue.outstanding != 0) {
      abi::NvmeCompletion &completion = queue.completions[queue.cqHead];
      const std::uint32_t commandAndStatus =
          device::loadIoCoherent(&completion.dword[3]);
      const std::uint32_t statusField = commandAndStatus >> 16U;
      if ((statusField & 1U) != queue.cqPhase) {
        break;
      }
      device::systemIoFence();

      const std::uint32_t commandId = commandAndStatus & 0xffffU;
      bool valid = commandId < queue.depth;
      if (valid) {
        abi::NvmeCommandContext &stored = queue.contexts[commandId];
        valid = atomicAdd(&stored.active, 0U) != 0;
        if (valid) {
          const abi::NvmeCommandContext context = stored;
          valid = context.objectSlot < runtime->objectCapacity;
          if (valid) {
            abi::ObjectEntry &object = runtime->objects[context.objectSlot];
            const abi::ReplicaEntry *replica =
                device::replica(runtime, object, object.selectedReplica);
            valid = object.objectId == context.objectId &&
                    object.version == context.objectVersion &&
                    replica != nullptr &&
                    replica->sourceKind ==
                        static_cast<std::uint32_t>(abi::SourceKind::Nvme);
            if (valid && (statusField >> 1U) == 0) {
              atomicExch(&object.state,
                         static_cast<std::uint32_t>(abi::ObjectState::Ready));
              ++queue.completed;
            } else {
              atomicExch(&object.state,
                         static_cast<std::uint32_t>(abi::ObjectState::Failed));
              device::failContinuation(runtime, context.continuation,
                                       abi::ContinuationState::Failed);
              ++queue.failed;
              queue.error = statusField >> 1U;
            }
          }
          device::releaseRequestBytes(runtime, context.requestSlot,
                                      context.generation, context.bytes);
          device::releaseTenantBytes(runtime, context.tenantId, context.bytes);
          device::releaseBackendBytes(runtime, abi::SourceKind::Nvme,
                                      context.backendBytes);
          atomicExch(&stored.active, 0U);
        }
      }
      if (!valid) {
        ++queue.failed;
        queue.error = 0xffffffffU;
      }

      queue.cqHead++;
      if (queue.cqHead == queue.depth) {
        queue.cqHead = 0;
        queue.cqPhase ^= 1U;
      }
      --queue.outstanding;
      ++drained;
    }
    if (drained != 0) {
      device::systemIoFence();
      *queue.cqDoorbell = queue.cqHead;
    }
  }
  __syncwarp();

  std::uint32_t issued = 0U;
  for (std::uint32_t attempt = 0; attempt < issueBudget; ++attempt) {
    std::uint32_t ticket = abi::InvalidIndex;
    std::uint32_t intentSlotIndex = abi::InvalidIndex;
    std::uint32_t objectSlot = abi::InvalidIndex;
    std::uint32_t commandId = abi::InvalidIndex;
    std::uint32_t submissionSlot = 0;
    std::uint32_t action = 0;
    std::uint64_t chargedBytes = 0;
    std::uint64_t backendBytes = 0;
    if (lane == 0 && queue.outstanding + 1U < queue.depth) {
      abi::AcquireIntent *selected = nullptr;
      abi::IntentSlot *selectedSlot = nullptr;
      abi::ObjectEntry *object = nullptr;
      const std::uint64_t now = device::globalTimerNs();
      std::uint32_t bestUrgency = 0;
      std::uint64_t bestWeightedService = UINT64_MAX;
      std::uint64_t bestSlack = UINT64_MAX;
      for (std::uint32_t candidateTicket = 0;
           candidateTicket < runtime->intentPool->capacity; ++candidateTicket) {
        abi::IntentSlot &candidateSlot = runtime->intents[candidateTicket];
        abi::AcquireIntent &candidate = candidateSlot.intent;
        if (atomicAdd(&candidate.valid, 0U) != 1U ||
            candidate.objectSlot >= runtime->objectCapacity) {
          continue;
        }
        abi::ObjectEntry &candidateObject =
            runtime->objects[candidate.objectSlot];
        const abi::ReplicaEntry *candidateReplica = device::replica(
            runtime, candidateObject, candidateObject.selectedReplica);
        if (candidateReplica != nullptr &&
            candidateReplica->sourceKind ==
                static_cast<std::uint32_t>(abi::SourceKind::Nvme)) {
          const std::uint32_t urgency = device::urgencyBucket(
              candidate.priority, candidate.deadlineClock, now);
          const std::uint64_t slack = candidate.deadlineClock == 0
                                          ? UINT64_MAX
                                          : (candidate.deadlineClock > now
                                                 ? candidate.deadlineClock - now
                                                 : 0);
          const abi::TenantContext *tenant =
              candidate.tenantId < runtime->tenantCapacity
                  ? &runtime->tenants[candidate.tenantId]
                  : nullptr;
          const std::uint64_t weightedService =
              tenant == nullptr || tenant->weight == 0
                  ? UINT64_MAX
                  : device::loadCounter(&tenant->serviceBytes) / tenant->weight;
          if (selected == nullptr || urgency > bestUrgency ||
              (urgency == bestUrgency &&
               weightedService < bestWeightedService) ||
              (urgency == bestUrgency &&
               weightedService == bestWeightedService && slack < bestSlack) ||
              (urgency == bestUrgency &&
               weightedService == bestWeightedService && slack == bestSlack &&
               candidateTicket < ticket)) {
            selected = &candidate;
            selectedSlot = &candidateSlot;
            object = &candidateObject;
            ticket = candidateTicket;
            bestUrgency = urgency;
            bestWeightedService = weightedService;
            bestSlack = slack;
          }
        }
      }
      if (selected != nullptr) {
        const abi::ReplicaEntry *replica =
            device::replica(runtime, *object, object->selectedReplica);
        const std::uint32_t expectedPages = static_cast<std::uint32_t>(
            (object->bytes + queue.controllerPageSize - 1U) /
            queue.controllerPageSize);
        const bool valid =
            replica != nullptr && object->objectId == selected->objectId &&
            object->version == selected->objectVersion &&
            selected->offset == 0 && selected->bytes == object->bytes &&
            replica->dmaPageCount == expectedPages && object->bytes != 0 &&
            object->bytes % (1ULL << queue.lbaShift) == 0 &&
            replica->sourceAddress % (1ULL << queue.lbaShift) == 0 &&
            replica->dmaPageCount <=
                queue.controllerPageSize / sizeof(std::uint64_t);
        if (!valid) {
          atomicExch(&object->state,
                     static_cast<std::uint32_t>(abi::ObjectState::Failed));
          device::failContinuation(runtime, selected->continuation,
                                   abi::ContinuationState::Failed);
          if (device::claimIntent(*selectedSlot)) {
            device::consumeIntent(runtime, *selectedSlot);
          }
          ++queue.failed;
          queue.error = 0xfffffffeU;
          action = 1;
        } else {
          const bool live = device::requestLive(runtime, selected->requestSlot,
                                                selected->generation);
          bool admitted = true;
          if (live) {
            admitted = device::tryReserveRequestBytes(
                runtime, selected->requestSlot, selected->generation,
                object->bytes);
            if (admitted && !device::tryReserveTenantBytes(
                                runtime, selected->tenantId, object->bytes)) {
              device::releaseRequestBytes(runtime, selected->requestSlot,
                                          selected->generation, object->bytes);
              admitted = false;
            }
          }
          if (admitted && !device::tryReserveBackendBytes(
                              runtime, abi::SourceKind::Nvme, object->bytes)) {
            if (live) {
              device::releaseRequestBytes(runtime, selected->requestSlot,
                                          selected->generation, object->bytes);
              device::releaseTenantBytes(runtime, selected->tenantId,
                                         object->bytes);
            }
            admitted = false;
          }
          if (!admitted) {
            action = 1;
          } else {
            chargedBytes = live ? object->bytes : 0;
            backendBytes = object->bytes;
            for (std::uint32_t searched = 0; searched < queue.depth;
                 ++searched) {
              const std::uint32_t candidate =
                  (queue.cidCursor + searched) % queue.depth;
              if (atomicAdd(&queue.contexts[candidate].active, 0U) == 0) {
                commandId = candidate;
                queue.cidCursor = (candidate + 1U) % queue.depth;
                break;
              }
            }
            if (commandId != abi::InvalidIndex) {
              if (device::claimIntent(*selectedSlot)) {
                objectSlot = selected->objectSlot;
                intentSlotIndex = ticket;
                submissionSlot = queue.sqTail;
                action = 2;
                if (selected->tenantId < runtime->tenantCapacity) {
                  atomicAdd(
                      reinterpret_cast<unsigned long long *>(
                          &runtime->tenants[selected->tenantId].serviceBytes),
                      static_cast<unsigned long long>(object->bytes));
                }
              } else {
                device::releaseRequestBytes(runtime, selected->requestSlot,
                                            selected->generation, chargedBytes);
                device::releaseTenantBytes(runtime, selected->tenantId,
                                           chargedBytes);
                device::releaseBackendBytes(runtime, abi::SourceKind::Nvme,
                                            backendBytes);
                chargedBytes = 0;
                backendBytes = 0;
                action = 1;
              }
            } else {
              device::releaseRequestBytes(runtime, selected->requestSlot,
                                          selected->generation, chargedBytes);
              device::releaseTenantBytes(runtime, selected->tenantId,
                                         chargedBytes);
              device::releaseBackendBytes(runtime, abi::SourceKind::Nvme,
                                          backendBytes);
              chargedBytes = 0;
              backendBytes = 0;
            }
          }
        }
      }
    }

    action = __shfl_sync(0xffffffffU, action, 0);
    if (action == 0) {
      break;
    }
    if (action == 1) {
      continue;
    }
    intentSlotIndex = __shfl_sync(0xffffffffU, intentSlotIndex, 0);
    objectSlot = __shfl_sync(0xffffffffU, objectSlot, 0);
    commandId = __shfl_sync(0xffffffffU, commandId, 0);
    submissionSlot = __shfl_sync(0xffffffffU, submissionSlot, 0);

    abi::IntentSlot &selectedSlot = runtime->intents[intentSlotIndex];
    abi::AcquireIntent &selected = selectedSlot.intent;
    abi::ObjectEntry &object = runtime->objects[objectSlot];
    const abi::ReplicaEntry &replica =
        *device::replica(runtime, object, object.selectedReplica);
    abi::NvmeSubmission &submission = queue.submissions[submissionSlot];
    if (lane < 16) {
      submission.dword[lane] = 0;
    }
    const auto *dmaPages =
        reinterpret_cast<const std::uint64_t *>(replica.dmaPageListAddress);
    if (replica.dmaPageCount > 2) {
      auto *prpList = reinterpret_cast<std::uint64_t *>(
          reinterpret_cast<std::byte *>(queue.prpLists) +
          static_cast<std::uint64_t>(commandId) * queue.controllerPageSize);
      for (std::uint32_t page = lane + 1U; page < replica.dmaPageCount;
           page += warpSize) {
        prpList[page - 1U] = dmaPages[page];
      }
    }
    __syncwarp();

    if (lane == 0) {
      const std::uint64_t firstPrp = dmaPages[0];
      const std::uint64_t secondPrp =
          replica.dmaPageCount == 2
              ? dmaPages[1]
              : (replica.dmaPageCount > 2
                     ? queue.prpListDmaAddress +
                           static_cast<std::uint64_t>(commandId) *
                               queue.controllerPageSize
                     : 0);
      const std::uint64_t lba = replica.sourceAddress >> queue.lbaShift;
      const std::uint32_t lbaCount =
          static_cast<std::uint32_t>(object.bytes >> queue.lbaShift);
      submission.dword[0] = 0x02U | (commandId << 16U);
      submission.dword[1] = queue.namespaceId;
      submission.dword[6] = static_cast<std::uint32_t>(firstPrp);
      submission.dword[7] = static_cast<std::uint32_t>(firstPrp >> 32U);
      submission.dword[8] = static_cast<std::uint32_t>(secondPrp);
      submission.dword[9] = static_cast<std::uint32_t>(secondPrp >> 32U);
      submission.dword[10] = static_cast<std::uint32_t>(lba);
      submission.dword[11] = static_cast<std::uint32_t>(lba >> 32U);
      submission.dword[12] = lbaCount - 1U;

      abi::NvmeCommandContext &context = queue.contexts[commandId];
      context.objectId = object.objectId;
      context.bytes = chargedBytes;
      context.backendBytes = backendBytes;
      context.objectSlot = selected.objectSlot;
      context.objectVersion = selected.objectVersion;
      context.requestSlot = selected.requestSlot;
      context.generation = selected.generation;
      context.continuation = selected.continuation;
      context.tenantId = selected.tenantId;
      atomicExch(&context.active, 1U);
      atomicExch(&object.state,
                 static_cast<std::uint32_t>(abi::ObjectState::Issued));
      device::consumeIntent(runtime, selectedSlot);
      queue.sqTail++;
      if (queue.sqTail == queue.depth) {
        queue.sqTail = 0;
      }
      ++queue.outstanding;
      ++queue.submitted;
      ++issued;
    }
    __syncwarp();
  }
  issued = __shfl_sync(0xffffffffU, issued, 0);
  queueOnline = lane == 0 && device::nvmeQueueOnline(queue);
  queueOnline = __shfl_sync(0xffffffffU, queueOnline, 0);
  if (issued != 0 && !queueOnline) {
    device::failNvmeQueue(runtime, queue, lane, 0xfffffffcU);
  } else if (lane == 0 && issued != 0) {
    device::systemIoFence();
    *queue.sqDoorbell = queue.sqTail;
  }
}
#endif

extern "C" __device__ __attribute__((used, noinline)) void
nta_defer(nta::abi::RuntimeView *runtime, std::uint32_t requestSlot,
          std::uint32_t generation, std::uint32_t continuation) {
  using namespace nta;
  if (runtime == nullptr || continuation >= runtime->continuationCapacity ||
      threadIdx.x != 0) {
    return;
  }

  abi::Continuation &record = runtime->continuations[continuation];
  const auto currentState =
      static_cast<abi::ContinuationState>(atomicAdd(&record.state, 0U));
  if (currentState == abi::ContinuationState::Cancelled ||
      currentState == abi::ContinuationState::Failed ||
      currentState == abi::ContinuationState::Ready ||
      currentState == abi::ContinuationState::Done) {
    return;
  }
  if (!device::requestLive(runtime, requestSlot, generation)) {
    record.requestSlot = requestSlot;
    record.generation = generation;
    atomicExch(&record.state,
               static_cast<std::uint32_t>(abi::ContinuationState::Cancelled));
    return;
  }

  const abi::RequestContext &request = runtime->requests[requestSlot];
  record.requestId = request.requestId;
  record.requestSlot = requestSlot;
  record.generation = generation;
  record.logicalTile = continuation;
  if (currentState == abi::ContinuationState::New) {
    // initializeContinuation owns both the Pending transition and pending
    // index publication. A live New record here has no resumable dependency
    // state and must fail closed instead of becoming an invisible waiter.
    atomicExch(&record.state,
               static_cast<std::uint32_t>(abi::ContinuationState::Failed));
  }
}

#if NTA_DEVICE_PHASE_KERNELS
extern "C" __global__ void
nta_progress_host_staging(nta::abi::RuntimeView *runtime) {
  using namespace nta;
  if (runtime == nullptr || runtime->intentPool == nullptr ||
      blockIdx.x >= runtime->intentPool->capacity) {
    return;
  }
  abi::IntentSlot &intentSlot = runtime->intents[blockIdx.x];

  abi::AcquireIntent &intent = intentSlot.intent;
  if (atomicAdd(&intent.valid, 0U) != 1U) {
    return;
  }
  if (intent.objectSlot >= runtime->objectCapacity) {
    if (threadIdx.x == 0 && device::claimIntent(intentSlot)) {
      device::consumeIntent(runtime, intentSlot);
    }
    return;
  }
  abi::ObjectEntry &object = runtime->objects[intent.objectSlot];
  const abi::ReplicaEntry *replica =
      device::replica(runtime, object, object.selectedReplica);
  if (replica == nullptr ||
      replica->sourceKind !=
          static_cast<std::uint32_t>(abi::SourceKind::HostStaged)) {
    return;
  }
  if (object.objectId != intent.objectId ||
      object.version != intent.objectVersion || intent.offset != 0 ||
      intent.bytes != object.bytes || intent.offset > object.bytes ||
      intent.bytes > object.bytes - intent.offset) {
    if (threadIdx.x == 0) {
      atomicExch(&object.state,
                 static_cast<std::uint32_t>(abi::ObjectState::Failed));
      device::failContinuation(runtime, intent.continuation,
                               abi::ContinuationState::Failed);
      if (device::claimIntent(intentSlot)) {
        device::consumeIntent(runtime, intentSlot);
      }
    }
    return;
  }

  __shared__ std::uint32_t admitted;
  __shared__ std::uint64_t chargedBytes;
  __shared__ std::uint64_t backendBytes;
  if (threadIdx.x == 0) {
    const bool live =
        device::requestLive(runtime, intent.requestSlot, intent.generation);
    bool accepted = true;
    if (live) {
      accepted = device::tryReserveRequestBytes(
          runtime, intent.requestSlot, intent.generation, intent.bytes);
      if (accepted && !device::tryReserveTenantBytes(runtime, intent.tenantId,
                                                     intent.bytes)) {
        device::releaseRequestBytes(runtime, intent.requestSlot,
                                    intent.generation, intent.bytes);
        accepted = false;
      }
    }
    if (accepted && !device::tryReserveBackendBytes(
                        runtime, abi::SourceKind::HostStaged, intent.bytes)) {
      if (live) {
        device::releaseRequestBytes(runtime, intent.requestSlot,
                                    intent.generation, intent.bytes);
        device::releaseTenantBytes(runtime, intent.tenantId, intent.bytes);
      }
      accepted = false;
    }
    admitted = accepted ? 1U : 0U;
    if (accepted && !device::claimIntent(intentSlot)) {
      if (live) {
        device::releaseRequestBytes(runtime, intent.requestSlot,
                                    intent.generation, intent.bytes);
        device::releaseTenantBytes(runtime, intent.tenantId, intent.bytes);
      }
      device::releaseBackendBytes(runtime, abi::SourceKind::HostStaged,
                                  intent.bytes);
      accepted = false;
      admitted = 0;
    }
    chargedBytes = live && accepted ? intent.bytes : 0;
    backendBytes = accepted ? intent.bytes : 0;
  }
  __syncthreads();
  if (admitted == 0) {
    return;
  }

  auto *source = reinterpret_cast<const std::byte *>(replica->sourceAddress) +
                 intent.offset;
  auto *destination =
      reinterpret_cast<std::byte *>(object.stagingAddress) + intent.offset;

  const std::uint32_t vectorBytes = intent.bytes & ~15U;
  for (std::uint32_t byte = threadIdx.x * 16U; byte < vectorBytes;
       byte += blockDim.x * 16U) {
    *reinterpret_cast<uint4 *>(destination + byte) =
        *reinterpret_cast<const uint4 *>(source + byte);
  }
  for (std::uint32_t byte = vectorBytes + threadIdx.x; byte < intent.bytes;
       byte += blockDim.x) {
    destination[byte] = source[byte];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    __threadfence_system();
    atomicExch(&object.state,
               static_cast<std::uint32_t>(abi::ObjectState::Ready));
    device::consumeIntent(runtime, intentSlot);
    device::releaseRequestBytes(runtime, intent.requestSlot, intent.generation,
                                chargedBytes);
    device::releaseTenantBytes(runtime, intent.tenantId, chargedBytes);
    device::releaseBackendBytes(runtime, abi::SourceKind::HostStaged,
                                backendBytes);
  }
}
#endif

#if NTA_DEVICE_PHASE_KERNELS
extern "C" __global__ void nta_publish_ready(nta::abi::RuntimeView *runtime,
                                             std::uint32_t pendingBudget) {
  using namespace nta;
  if (runtime == nullptr || runtime->pendingCount == nullptr ||
      runtime->pendingContinuations == nullptr) {
    return;
  }
  const std::uint32_t pendingCount =
      min(min(atomicAdd(runtime->pendingCount, 0U), pendingBudget),
          runtime->continuationCapacity);
  const std::uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t stride = blockDim.x * gridDim.x;
  for (std::uint32_t pendingIndex = thread; pendingIndex < pendingCount;
       pendingIndex += stride) {
    const std::uint32_t continuationIndex =
        runtime->pendingContinuations[pendingIndex];
    if (continuationIndex >= runtime->continuationCapacity) {
      continue;
    }
    abi::Continuation &continuation = runtime->continuations[continuationIndex];
    if (atomicAdd(&continuation.state, 0U) !=
        static_cast<std::uint32_t>(abi::ContinuationState::Pending)) {
      continue;
    }
    if (!device::requestLive(runtime, continuation.requestSlot,
                             continuation.generation)) {
      atomicCAS(&continuation.state,
                static_cast<std::uint32_t>(abi::ContinuationState::Pending),
                static_cast<std::uint32_t>(abi::ContinuationState::Cancelled));
      continue;
    }
    std::uint32_t dependencyStart = 0;
    if (!device::dependencyRange(runtime, continuationIndex,
                                 continuation.dependencyCount,
                                 dependencyStart) ||
        dependencyStart != continuation.dependencyStart) {
      atomicCAS(&continuation.state,
                static_cast<std::uint32_t>(abi::ContinuationState::Pending),
                static_cast<std::uint32_t>(abi::ContinuationState::Failed));
      continue;
    }

    bool ready = true;
    bool failed = false;
    for (std::uint32_t index = 0; index < continuation.dependencyCount;
         ++index) {
      const abi::ContinuationDependency dependency =
          runtime->dependencies[dependencyStart + index];
      if (dependency.objectSlot >= runtime->objectCapacity) {
        failed = true;
        break;
      }
      abi::ObjectEntry &object = runtime->objects[dependency.objectSlot];
      if (object.objectId != dependency.objectId ||
          object.version != dependency.objectVersion) {
        failed = true;
        break;
      }
      const auto objectState =
          static_cast<abi::ObjectState>(atomicAdd(&object.state, 0U));
      if (objectState == abi::ObjectState::Failed) {
        failed = true;
        break;
      }
      ready &= objectState == abi::ObjectState::Ready;
    }
    if (failed) {
      atomicCAS(&continuation.state,
                static_cast<std::uint32_t>(abi::ContinuationState::Pending),
                static_cast<std::uint32_t>(abi::ContinuationState::Failed));
      continue;
    }
    if (!ready) {
      continue;
    }
    if (atomicCAS(&continuation.state,
                  static_cast<std::uint32_t>(abi::ContinuationState::Pending),
                  static_cast<std::uint32_t>(abi::ContinuationState::Ready)) !=
        static_cast<std::uint32_t>(abi::ContinuationState::Pending)) {
      continue;
    }

    const std::uint32_t ticket = atomicAdd(runtime->readyCount, 1U);
    if (ticket < runtime->continuationCapacity) {
      runtime->readyContinuations[ticket] = continuationIndex;
    } else {
      atomicExch(&continuation.state,
                 static_cast<std::uint32_t>(abi::ContinuationState::Failed));
    }
  }
}
#endif

#if NTA_DEVICE_PHASE_KERNELS
extern "C" __global__ void nta_reset_epoch(nta::abi::RuntimeView *runtime,
                                           std::uint32_t objectCount,
                                           std::uint32_t continuationCount) {
  using namespace nta;
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index == 0) {
    *runtime->readyCount = 0;
    *runtime->readyHead = 0;
    *runtime->pendingCount = 0;
  }
  if (index < objectCount && index < runtime->objectCapacity) {
    abi::ObjectEntry &object = runtime->objects[index];
    object.issueCount = 0;
    const abi::ReplicaEntry *replica =
        device::replica(runtime, object, object.selectedReplica);
    if (replica != nullptr &&
        replica->sourceKind ==
            static_cast<std::uint32_t>(abi::SourceKind::HostStaged)) {
      object.state = static_cast<std::uint32_t>(abi::ObjectState::New);
    } else if (replica != nullptr &&
               replica->sourceKind ==
                   static_cast<std::uint32_t>(abi::SourceKind::Nvme) &&
               (device::nvmeQueue(runtime) == nullptr ||
                device::nvmeQueue(runtime)->outstanding == 0)) {
      object.state = static_cast<std::uint32_t>(abi::ObjectState::New);
    }
  }
  if (index < continuationCount && index < runtime->continuationCapacity) {
    abi::Continuation &continuation = runtime->continuations[index];
    continuation.state =
        static_cast<std::uint32_t>(abi::ContinuationState::New);
    continuation.dependencyCount = 0;
    continuation.dependencyStart = abi::InvalidIndex;
  }
  if (index == 0 && device::nvmeQueue(runtime) != nullptr) {
    device::nvmeQueue(runtime)->intentCursor = 0;
  }
}

// Retire only work that executed in the preceding stream-ordered kernel.
// Pending work remains eligible for readiness publication and a later launch.
extern "C" __global__ void
nta_complete_launched(nta::abi::RuntimeView *runtime,
                      std::uint32_t continuationCount) {
  if (runtime == nullptr) {
    return;
  }
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= continuationCount || index >= runtime->continuationCapacity) {
    return;
  }
  nta::abi::Continuation &continuation = runtime->continuations[index];
  const std::uint32_t done =
      static_cast<std::uint32_t>(nta::abi::ContinuationState::Done);
  (void)atomicCAS(&continuation.state,
                  static_cast<std::uint32_t>(nta::abi::ContinuationState::New),
                  done);
  (void)atomicCAS(
      &continuation.state,
      static_cast<std::uint32_t>(nta::abi::ContinuationState::Ready), done);
}
#endif
