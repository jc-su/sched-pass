#pragma once

#include "nta/RuntimeABI.h"
#include "nta/TicketProtocol.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#ifndef NTA_DEVICE_PHASE_KERNELS
#define NTA_DEVICE_PHASE_KERNELS 1
#endif

namespace nta::device {

__device__ __forceinline__ std::uint64_t globalTimerNs();

__device__ __forceinline__ bool publishRunnableWork(abi::RuntimeView *runtime,
                                                    std::uint32_t workTicket) {
  if (runtime == nullptr || runtime->workTickets == nullptr ||
      runtime->remainingDependencies == nullptr ||
      runtime->readyCount == nullptr || runtime->readyWorkTickets == nullptr ||
      workTicket >= runtime->workTicketCapacity) {
    return false;
  }
  abi::WorkTicket &ticket = runtime->workTickets[workTicket];
  const std::uint32_t pending =
      static_cast<std::uint32_t>(abi::WorkTicketState::Pending);
  if (atomicAdd(&ticket.state, 0U) != pending ||
      atomicAdd(&runtime->remainingDependencies[workTicket], 0U) != 0U) {
    return false;
  }
  if (ticket.epoch != currentEpoch(runtime) ||
      ticket.requestSlot >= runtime->requestCapacity) {
    if (atomicCAS(&ticket.state, pending,
                  static_cast<std::uint32_t>(abi::WorkTicketState::Failed)) ==
        pending) {
      recordTerminalWork(runtime, ticket, abi::WorkTicketState::Pending,
                         abi::WorkTicketState::Failed);
      recordFailure(runtime);
    }
    return false;
  }
  const abi::RequestContext &request = runtime->requests[ticket.requestSlot];
  if (request.generation != ticket.generation || request.cancelled != 0) {
    if (atomicCAS(&ticket.state, pending,
                  static_cast<std::uint32_t>(
                      abi::WorkTicketState::Cancelled)) == pending) {
      recordTerminalWork(runtime, ticket, abi::WorkTicketState::Pending,
                         abi::WorkTicketState::Cancelled);
    }
    return false;
  }
  if (atomicCAS(&ticket.state, pending,
                static_cast<std::uint32_t>(abi::WorkTicketState::Ready)) !=
      pending) {
    return false;
  }
  if (runtime->workRunnableNs != nullptr) {
    const std::uint64_t now = globalTimerNs();
    runtime->workRunnableNs[workTicket] =
        now >= runtime->epochStartClock ? now - runtime->epochStartClock : 0;
  }
  recordRunnableWork(runtime, ticket);
  const std::uint32_t slot = atomicAdd(runtime->readyCount, 1U);
  if (slot < runtime->workTicketCapacity) {
    runtime->readyWorkTickets[slot] = workTicket;
    return true;
  }
  atomicExch(&ticket.state,
             static_cast<std::uint32_t>(abi::WorkTicketState::Failed));
  recordTerminalWork(runtime, ticket, abi::WorkTicketState::Ready,
                     abi::WorkTicketState::Failed);
  recordFailure(runtime);
  return false;
}

__device__ __forceinline__ bool enqueueChangedWork(abi::RuntimeView *runtime,
                                                   std::uint32_t workTicket) {
  if (runtime == nullptr || runtime->changedWorkTickets == nullptr ||
      runtime->changedQueued == nullptr || runtime->changedCount == nullptr ||
      runtime->changedOverflow == nullptr ||
      workTicket >= runtime->workTicketCapacity) {
    return false;
  }
  if (atomicCAS(&runtime->changedQueued[workTicket], 0U, 1U) != 0U) {
    return true;
  }
  const std::uint32_t slot = atomicAdd(runtime->changedCount, 1U);
  if (slot < runtime->workTicketCapacity) {
    runtime->changedWorkTickets[slot] = workTicket;
    return true;
  }
  atomicExch(runtime->changedOverflow, 1U);
  // The bounded pending index remains the fail-closed fallback.
  return true;
}

__device__ __forceinline__ bool
dependencyRange(abi::RuntimeView *runtime, std::uint32_t workTicket,
                std::uint32_t dependencyCount, std::uint32_t &dependencyStart) {
  if (runtime == nullptr || runtime->dependencies == nullptr ||
      workTicket >= runtime->workTicketCapacity ||
      dependencyCount > runtime->maxDependenciesPerWorkTicket ||
      runtime->maxDependenciesPerWorkTicket == 0 ||
      workTicket >
          runtime->dependencyCapacity / runtime->maxDependenciesPerWorkTicket) {
    return false;
  }
  dependencyStart = workTicket * runtime->maxDependenciesPerWorkTicket;
  return dependencyStart <= runtime->dependencyCapacity &&
         dependencyCount <= runtime->dependencyCapacity - dependencyStart;
}

__device__ __forceinline__ bool
dependencyBelongsToTicket(const abi::RuntimeView *runtime,
                          std::uint32_t dependency, std::uint32_t workTicket) {
  if (runtime == nullptr || workTicket >= runtime->workTicketCapacity ||
      dependency >= runtime->dependencyCapacity) {
    return false;
  }
  const abi::WorkTicket &ticket = runtime->workTickets[workTicket];
  return ticket.epoch == currentEpoch(runtime) &&
         dependency >= ticket.dependencyStart &&
         dependency - ticket.dependencyStart < ticket.dependencyCount;
}

__device__ __forceinline__ bool satisfyDependency(abi::RuntimeView *runtime,
                                                  std::uint32_t dependency) {
  if (runtime == nullptr || runtime->dependencySatisfied == nullptr ||
      runtime->remainingDependencies == nullptr ||
      runtime->maxDependenciesPerWorkTicket == 0 ||
      dependency >= runtime->dependencyCapacity) {
    return false;
  }
  const std::uint32_t workTicket =
      dependency / runtime->maxDependenciesPerWorkTicket;
  if (!dependencyBelongsToTicket(runtime, dependency, workTicket)) {
    return false;
  }
  abi::WorkTicket &ticket = runtime->workTickets[workTicket];
  const std::uint32_t state = atomicAdd(&ticket.state, 0U);
  if (state != static_cast<std::uint32_t>(abi::WorkTicketState::Initializing) &&
      state != static_cast<std::uint32_t>(abi::WorkTicketState::Pending)) {
    return false;
  }
  if (atomicCAS(&runtime->dependencySatisfied[dependency], 0U, 1U) != 0U) {
    return true;
  }
  const std::uint32_t previous =
      atomicSub(&runtime->remainingDependencies[workTicket], 1U);
  if (previous == 0U) {
    atomicAdd(&runtime->remainingDependencies[workTicket], 1U);
    atomicExch(runtime->changedOverflow, 1U);
    return false;
  }
  if (previous == 1U &&
      atomicAdd(&ticket.state, 0U) ==
          static_cast<std::uint32_t>(abi::WorkTicketState::Pending)) {
    return publishRunnableWork(runtime, workTicket);
  }
  return true;
}

__device__ __forceinline__ bool failDependency(abi::RuntimeView *runtime,
                                               std::uint32_t dependency) {
  if (runtime == nullptr || runtime->dependencySatisfied == nullptr ||
      runtime->maxDependenciesPerWorkTicket == 0 ||
      dependency >= runtime->dependencyCapacity) {
    return false;
  }
  const std::uint32_t workTicket =
      dependency / runtime->maxDependenciesPerWorkTicket;
  if (!dependencyBelongsToTicket(runtime, dependency, workTicket)) {
    return false;
  }
  abi::WorkTicket &ticket = runtime->workTickets[workTicket];
  (void)atomicCAS(&runtime->dependencySatisfied[dependency], 0U, 2U);
  if (atomicCAS(&ticket.state,
                static_cast<std::uint32_t>(abi::WorkTicketState::Pending),
                static_cast<std::uint32_t>(abi::WorkTicketState::Failed)) ==
      static_cast<std::uint32_t>(abi::WorkTicketState::Pending)) {
    recordTerminalWork(runtime, ticket, abi::WorkTicketState::Pending,
                       abi::WorkTicketState::Failed);
    recordFailure(runtime);
  }
  return true;
}

__device__ __forceinline__ bool
initializeWorkTicket(abi::RuntimeView *runtime, std::uint32_t requestSlot,
                     std::uint32_t generation, std::uint32_t workTicket,
                     const abi::AcquireRequirement *requirements,
                     std::uint32_t requirementCount) {
  std::uint32_t dependencyStart = 0;
  if (requirementCount == 0 ||
      !dependencyRange(runtime, workTicket, requirementCount,
                       dependencyStart) ||
      requirements == nullptr || requestSlot >= runtime->requestCapacity ||
      runtime->pendingWorkTickets == nullptr ||
      runtime->pendingCount == nullptr ||
      runtime->objectDependentHeads == nullptr ||
      runtime->dependencyNext == nullptr ||
      runtime->dependencySatisfied == nullptr ||
      runtime->remainingDependencies == nullptr ||
      runtime->changedQueued == nullptr ||
      runtime->maxDependenciesPerWorkTicket == 0) {
    failWorkTicket(runtime, workTicket, abi::WorkTicketState::Failed);
    return false;
  }

  abi::WorkTicket &record = runtime->workTickets[workTicket];
  const auto state = static_cast<abi::WorkTicketState>(atomicCAS(
      &record.state, static_cast<std::uint32_t>(abi::WorkTicketState::New),
      static_cast<std::uint32_t>(abi::WorkTicketState::Initializing)));
  if (state == abi::WorkTicketState::Cancelled ||
      state == abi::WorkTicketState::Failed) {
    return false;
  }
  if (state != abi::WorkTicketState::New) {
    return ticketMatches(runtime, record, requestSlot, generation) &&
           (state == abi::WorkTicketState::Pending ||
            state == abi::WorkTicketState::Ready ||
            state == abi::WorkTicketState::Done);
  }

  std::uint32_t externalCount = 0;
  for (std::uint32_t index = 0; index < requirementCount; ++index) {
    externalCount += requirements[index].directBase == 0 ? 1U : 0U;
  }

  const abi::RequestContext &request = runtime->requests[requestSlot];
  record.requestId = request.requestId;
  record.requestSlot = requestSlot;
  record.generation = generation;
  record.dependencyCount = externalCount;
  record.logicalTile = workTicket;
  record.dependencyStart = dependencyStart;
  record.epoch = currentEpoch(runtime);
  std::uint64_t unavailableBytes = 0;
  for (std::uint32_t index = 0; index < requirementCount; ++index) {
    if (requirements[index].directBase == 0) {
      unavailableBytes += requirements[index].bytes;
    }
  }
  record.unavailableBytes = unavailableBytes;
  runtime->remainingDependencies[workTicket] = externalCount;
  runtime->changedQueued[workTicket] = 0;

  std::uint32_t externalIndex = 0;
  for (std::uint32_t index = 0; index < requirementCount; ++index) {
    if (requirements[index].directBase != 0) {
      continue;
    }
    const std::uint32_t dependency = dependencyStart + externalIndex++;
    const abi::AcquireRequirement &requirement = requirements[index];
    runtime->dependencies[dependency] = {
        requirement.objectId,
        requirement.objectSlot,
        requirement.objectVersion,
    };
    runtime->dependencyNext[dependency] = abi::InvalidIndex;
    runtime->dependencySatisfied[dependency] = 0;
  }
  __threadfence();

  bool setupFailed = request.generation != generation || request.cancelled != 0;
  externalIndex = 0;
  for (std::uint32_t index = 0; index < requirementCount; ++index) {
    const abi::AcquireRequirement &requirement = requirements[index];
    if (requirement.directBase != 0) {
      continue;
    }
    const std::uint32_t dependency = dependencyStart + externalIndex++;
    if (requirement.objectSlot >= runtime->objectCapacity) {
      setupFailed = true;
      runtime->dependencySatisfied[dependency] = 2U;
      continue;
    }
    abi::ObjectEntry &object = runtime->objects[requirement.objectSlot];
    if (object.objectId != requirement.objectId ||
        object.version != requirement.objectVersion) {
      setupFailed = true;
      runtime->dependencySatisfied[dependency] = 2U;
      continue;
    }
    runtime->dependencyNext[dependency] = atomicExch(
        &runtime->objectDependentHeads[requirement.objectSlot], dependency);
  }
  __threadfence();

  // Reconcile after every edge is visible. If completion raced ahead of edge
  // insertion, this pass observes the terminal object state; if completion
  // followed insertion, the per-edge CAS makes the duplicate observation free.
  for (std::uint32_t dependencyOffset = 0; dependencyOffset < externalCount;
       ++dependencyOffset) {
    const std::uint32_t dependency = dependencyStart + dependencyOffset;
    if (runtime->dependencySatisfied[dependency] == 2U) {
      continue;
    }
    const abi::WorkDependency &entry = runtime->dependencies[dependency];
    abi::ObjectEntry &object = runtime->objects[entry.objectSlot];
    const auto objectState =
        static_cast<abi::ObjectState>(atomicAdd(&object.state, 0U));
    if (objectState == abi::ObjectState::Ready) {
      (void)satisfyDependency(runtime, dependency);
    } else if (objectState == abi::ObjectState::Failed) {
      runtime->dependencySatisfied[dependency] = 2U;
      setupFailed = true;
    }
  }

  const std::uint32_t ticket = atomicAdd(runtime->pendingCount, 1U);
  if (ticket < runtime->workTicketCapacity) {
    runtime->pendingWorkTickets[ticket] = workTicket;
    __threadfence();
    atomicExch(&record.state,
               static_cast<std::uint32_t>(abi::WorkTicketState::Pending));
    recordPendingWork(runtime, record);
    for (std::uint32_t dependencyOffset = 0; dependencyOffset < externalCount;
         ++dependencyOffset) {
      setupFailed |=
          runtime->dependencySatisfied[dependencyStart + dependencyOffset] ==
          2U;
    }
    if (setupFailed &&
        atomicCAS(&record.state,
                  static_cast<std::uint32_t>(abi::WorkTicketState::Pending),
                  static_cast<std::uint32_t>(abi::WorkTicketState::Failed)) ==
            static_cast<std::uint32_t>(abi::WorkTicketState::Pending)) {
      recordTerminalWork(runtime, record, abi::WorkTicketState::Pending,
                         abi::WorkTicketState::Failed);
      recordFailure(runtime);
      return false;
    }
    if (atomicAdd(&runtime->remainingDependencies[workTicket], 0U) == 0U) {
      return publishRunnableWork(runtime, workTicket);
    }
  } else {
    atomicExch(&record.state,
               static_cast<std::uint32_t>(abi::WorkTicketState::Failed));
    recordTerminalWork(runtime, record, abi::WorkTicketState::New,
                       abi::WorkTicketState::Failed);
    recordFailure(runtime);
    return false;
  }
  return true;
}

__device__ __forceinline__ void
publishObjectTransition(abi::RuntimeView *runtime, std::uint32_t objectSlot,
                        abi::ObjectState state) {
  if (runtime == nullptr || objectSlot >= runtime->objectCapacity ||
      runtime->objectDependentHeads == nullptr ||
      runtime->dependencyNext == nullptr ||
      runtime->dependencySatisfied == nullptr ||
      runtime->remainingDependencies == nullptr ||
      runtime->maxDependenciesPerWorkTicket == 0) {
    return;
  }
  std::uint32_t dependency = runtime->objectDependentHeads[objectSlot];
  std::uint32_t traversed = 0;
  while (dependency != abi::InvalidIndex &&
         dependency < runtime->dependencyCapacity &&
         traversed++ < runtime->dependencyCapacity) {
    const std::uint32_t next = runtime->dependencyNext[dependency];
    const abi::WorkDependency &entry = runtime->dependencies[dependency];
    if (entry.objectSlot != objectSlot) {
      atomicExch(runtime->changedOverflow, 1U);
    } else if (state == abi::ObjectState::Ready) {
      (void)satisfyDependency(runtime, dependency);
    } else if (state == abi::ObjectState::Failed) {
      (void)failDependency(runtime, dependency);
    }
    dependency = next;
  }
  if (dependency != abi::InvalidIndex && runtime->changedOverflow != nullptr) {
    atomicExch(runtime->changedOverflow, 1U);
  }
}

// Indexed range completion has two materially different dependency
// topologies. A private acquisition object has exactly one reverse edge and
// can satisfy it directly in O(1). A shared object can fan out to hundreds of
// work tickets; serially walking that list is slower than the existing
// parallel pending-ticket scan. Preserve that scan by setting the fail-closed
// overflow bit whenever the reverse list is not exactly one valid edge.
//
// This is a structural choice, not a size threshold: the one-edge case is
// exact and every other shape retains the general publication path.
__device__ __forceinline__ bool
publishPrivateIndexedObject(abi::RuntimeView *runtime,
                            std::uint32_t objectSlot) {
  if (runtime == nullptr || objectSlot >= runtime->objectCapacity ||
      runtime->objects == nullptr || runtime->objectDependentHeads == nullptr ||
      runtime->dependencyNext == nullptr || runtime->dependencies == nullptr ||
      runtime->changedOverflow == nullptr) {
    return false;
  }
  const std::uint32_t dependency = runtime->objectDependentHeads[objectSlot];
  if (dependency == abi::InvalidIndex ||
      dependency >= runtime->dependencyCapacity ||
      runtime->dependencyNext[dependency] != abi::InvalidIndex) {
    atomicExch(runtime->changedOverflow, 1U);
    return false;
  }
  const abi::WorkDependency &entry = runtime->dependencies[dependency];
  const abi::ObjectEntry &object = runtime->objects[objectSlot];
  if (entry.objectSlot != objectSlot || entry.objectId != object.objectId ||
      entry.objectVersion != object.version) {
    atomicExch(runtime->changedOverflow, 1U);
    return false;
  }
  if (!satisfyDependency(runtime, dependency)) {
    atomicExch(runtime->changedOverflow, 1U);
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

__device__ __forceinline__ void storeMmio(volatile std::uint32_t *address,
                                          std::uint32_t value) {
  asm volatile("st.mmio.relaxed.sys.b32 [%0], %1;"
               :
               : "l"(address), "r"(value)
               : "memory");
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
  return tryReserveCounter(&tenant.outstandingBytes, tenant.maxOutstandingBytes,
                           bytes);
}

__device__ __forceinline__ void releaseTenantBytes(abi::RuntimeView *runtime,
                                                   std::uint32_t tenantId,
                                                   std::uint64_t bytes) {
  if (bytes != 0 && tenantId < runtime->tenantCapacity) {
    releaseCounter(&runtime->tenants[tenantId].outstandingBytes, bytes);
  }
}

__device__ __forceinline__ std::uint64_t saturatingAdd(std::uint64_t left,
                                                       std::uint64_t right) {
  return right > UINT64_MAX - left ? UINT64_MAX : left + right;
}

__device__ __forceinline__ std::uint64_t
bytesToNanoseconds(std::uint64_t bytes, std::uint64_t bandwidth) {
  if (bytes == 0 || bandwidth == 0) {
    return 0;
  }
  constexpr std::uint64_t billion = 1'000'000'000ULL;
  const std::uint64_t seconds = bytes / bandwidth;
  const std::uint64_t remainder = bytes % bandwidth;
  if (seconds > UINT64_MAX / billion || remainder > UINT64_MAX / billion) {
    return UINT64_MAX;
  }
  const std::uint64_t product = remainder * billion;
  return saturatingAdd(seconds * billion,
                       product / bandwidth + (product % bandwidth != 0));
}

__device__ __forceinline__ std::uint64_t
criticalServiceNs(abi::RuntimeView *runtime, const abi::AcquireIntent &intent,
                  abi::SourceKind source) {
  if (runtime == nullptr) {
    return 0;
  }
  std::uint64_t acquisitionNs = 0;
  const std::uint32_t sourceIndex = static_cast<std::uint32_t>(source);
  if (runtime->backends != nullptr && sourceIndex < runtime->backendCapacity) {
    const abi::BackendView &backend = runtime->backends[sourceIndex];
    if (backend.sourceKind == sourceIndex &&
        backend.estimatedBandwidthBytesPerSecond != 0) {
      const std::uint64_t queued =
          atomicAdd(reinterpret_cast<unsigned long long *>(
                        const_cast<std::uint64_t *>(&backend.outstandingBytes)),
                    0ULL);
      acquisitionNs = saturatingAdd(
          backend.estimatedLatencyNs,
          bytesToNanoseconds(saturatingAdd(queued, intent.bytes),
                             backend.estimatedBandwidthBytesPerSecond));
    }
  }

  std::uint64_t pendingComputeNs = 0;
  std::uint64_t runnableComputeNs = 0;
  if (runtime->requestProgress != nullptr &&
      intent.requestSlot < runtime->requestCapacity) {
    const abi::RequestProgress &progress =
        runtime->requestProgress[intent.requestSlot];
    if (progress.generation == intent.generation) {
      pendingComputeNs = atomicAdd(
          reinterpret_cast<unsigned long long *>(
              const_cast<std::uint64_t *>(&progress.pendingComputeNs)),
          0ULL);
      runnableComputeNs = atomicAdd(
          reinterpret_cast<unsigned long long *>(
              const_cast<std::uint64_t *>(&progress.runnableComputeNs)),
          0ULL);
    }
  }
  // The first queue insertion precedes Pending publication. Include the ticket
  // estimate exactly once in that initializing window; requeues observe it in
  // RequestProgress instead.
  if (runtime->workTickets != nullptr &&
      intent.workTicket < runtime->workTicketCapacity) {
    const abi::WorkTicket &ticket = runtime->workTickets[intent.workTicket];
    if (atomicAdd(const_cast<std::uint32_t *>(&ticket.state), 0U) ==
        static_cast<std::uint32_t>(abi::WorkTicketState::Initializing)) {
      pendingComputeNs =
          saturatingAdd(pendingComputeNs, ticket.estimatedComputeNs);
    }
  }
  return saturatingAdd(acquisitionNs > runnableComputeNs ? acquisitionNs
                                                         : runnableComputeNs,
                       pendingComputeNs);
}

__device__ __forceinline__ bool
intentQueueAvailable(const abi::RuntimeView *runtime) {
  return runtime != nullptr && runtime->intentQueueEntries != nullptr &&
         runtime->intentQueueControls != nullptr &&
         runtime->intentQueueHeap != nullptr && runtime->intents != nullptr &&
         runtime->intentCapacity != 0 &&
         runtime->backendCapacity <= abi::BackendCount;
}

__device__ __forceinline__ abi::IntentQueueControl *
intentQueueControl(abi::RuntimeView *runtime, abi::SourceKind source) {
  const std::uint32_t sourceIndex = static_cast<std::uint32_t>(source);
  return !intentQueueAvailable(runtime) ||
                 sourceIndex >= runtime->backendCapacity ||
                 sourceIndex >= abi::BackendCount
             ? nullptr
             : &runtime->intentQueueControls[sourceIndex];
}

__device__ __forceinline__ abi::IntentQueueNode *
intentQueueHeap(abi::RuntimeView *runtime, abi::SourceKind source) {
  const std::uint32_t sourceIndex = static_cast<std::uint32_t>(source);
  return !intentQueueAvailable(runtime) ||
                 sourceIndex >= runtime->backendCapacity ||
                 sourceIndex >= abi::BackendCount
             ? nullptr
             : runtime->intentQueueHeap +
                   static_cast<std::uint64_t>(sourceIndex) *
                       runtime->intentCapacity;
}

__device__ __forceinline__ void
lockIntentQueue(abi::IntentQueueControl &control) {
  // Every contender that reaches this loop is resident. The owner executes a
  // bounded heap operation and never waits for transport or another lock, so
  // this device mutex cannot form a lock cycle. Insert/pop are O(log C);
  // tombstone compaction is O(C) but occurs only after C deferred removals.
  while (atomicCAS(&control.lock, 0U, 1U) != 0U) {
  }
  __threadfence();
}

__device__ __forceinline__ bool
tryLockIntentQueue(abi::IntentQueueControl &control) {
  const bool acquired = atomicCAS(&control.lock, 0U, 1U) == 0U;
  if (acquired) {
    __threadfence();
  }
  return acquired;
}

__device__ __forceinline__ void
unlockIntentQueue(abi::IntentQueueControl &control) {
  __threadfence();
  atomicExch(&control.lock, 0U);
}

// Strict weak ordering for the production transport queue. This is exact EDF
// for timed intents: the smallest absolute deadline always wins. EDF permits
// arbitrary ties; we make them deterministic with higher caller priority,
// then larger critical service (least laxity at an equal deadline), then the
// first insertion sequence. deadlineClock == 0 is not a synthetic infinity:
// it is an explicit best-effort class after every timed intent. Within that
// class, priority is followed by known shortest critical service to minimize
// mean completion time; an unavailable (zero) estimate sorts last.
__device__ __forceinline__ bool
intentQueueEntryPrecedes(const abi::IntentQueueEntry &left,
                         const abi::IntentQueueEntry &right) {
  const bool leftTimed = left.deadlineClock != 0;
  const bool rightTimed = right.deadlineClock != 0;
  if (leftTimed != rightTimed) {
    return leftTimed;
  }
  if (leftTimed && left.deadlineClock != right.deadlineClock) {
    return left.deadlineClock < right.deadlineClock;
  }
  if (left.priority != right.priority) {
    return left.priority > right.priority;
  }
  if (left.criticalServiceNs != right.criticalServiceNs) {
    if (leftTimed) {
      return left.criticalServiceNs > right.criticalServiceNs;
    }
    const bool leftKnown = left.criticalServiceNs != 0;
    const bool rightKnown = right.criticalServiceNs != 0;
    if (leftKnown != rightKnown) {
      return leftKnown;
    }
    return left.criticalServiceNs < right.criticalServiceNs;
  }
  if (left.stableSequence != right.stableSequence) {
    return left.stableSequence < right.stableSequence;
  }
  return left.intentSequence < right.intentSequence;
}

__device__ __forceinline__ bool
intentQueueNodeCurrent(abi::RuntimeView *runtime, abi::SourceKind source,
                       const abi::IntentQueueNode &node,
                       abi::IntentQueueEntry *&entry) {
  entry = nullptr;
  if (node.slotIndex >= runtime->intentCapacity) {
    return false;
  }
  abi::IntentQueueEntry &candidate =
      runtime->intentQueueEntries[node.slotIndex];
  abi::IntentSlot &slot = runtime->intents[node.slotIndex];
  const std::uint32_t sourceIndex = static_cast<std::uint32_t>(source);
  if (candidate.intentSequence != node.intentSequence ||
      atomicAdd(&candidate.state, 0U) !=
          static_cast<std::uint32_t>(abi::IntentQueueState::Queued) ||
      candidate.epoch != currentEpoch(runtime) ||
      candidate.sourceKind != sourceIndex ||
      slot.sequence != node.intentSequence || slot.epoch != candidate.epoch ||
      slot.sourceKind != sourceIndex ||
      atomicAdd(&slot.intent.valid, 0U) != 1U) {
    return false;
  }
  entry = &candidate;
  return true;
}

__device__ __forceinline__ bool
intentQueueNodePrecedes(abi::RuntimeView *runtime, abi::SourceKind source,
                        const abi::IntentQueueNode &left,
                        const abi::IntentQueueNode &right) {
  abi::IntentQueueEntry *leftEntry = nullptr;
  abi::IntentQueueEntry *rightEntry = nullptr;
  const bool leftCurrent =
      intentQueueNodeCurrent(runtime, source, left, leftEntry);
  const bool rightCurrent =
      intentQueueNodeCurrent(runtime, source, right, rightEntry);
  if (leftCurrent != rightCurrent) {
    return leftCurrent;
  }
  if (leftCurrent) {
    if (intentQueueEntryPrecedes(*leftEntry, *rightEntry)) {
      return true;
    }
    if (intentQueueEntryPrecedes(*rightEntry, *leftEntry)) {
      return false;
    }
  }
  return left.intentSequence != right.intentSequence
             ? left.intentSequence < right.intentSequence
             : left.slotIndex < right.slotIndex;
}

__device__ __forceinline__ void
setIntentHeapNode(abi::RuntimeView *runtime, abi::SourceKind source,
                  abi::IntentQueueNode *heap, std::uint32_t index,
                  const abi::IntentQueueNode &node) {
  heap[index] = node;
  if (node.slotIndex < runtime->intentCapacity) {
    abi::IntentQueueEntry &entry = runtime->intentQueueEntries[node.slotIndex];
    if (entry.intentSequence == node.intentSequence &&
        entry.sourceKind == static_cast<std::uint32_t>(source)) {
      entry.heapIndex = index;
    }
  }
}

__device__ __forceinline__ void siftIntentUp(abi::RuntimeView *runtime,
                                             abi::SourceKind source,
                                             abi::IntentQueueNode *heap,
                                             std::uint32_t index) {
  while (index != 0) {
    const std::uint32_t parent = (index - 1U) / 2U;
    if (!intentQueueNodePrecedes(runtime, source, heap[index], heap[parent])) {
      break;
    }
    const abi::IntentQueueNode childNode = heap[index];
    const abi::IntentQueueNode parentNode = heap[parent];
    setIntentHeapNode(runtime, source, heap, parent, childNode);
    setIntentHeapNode(runtime, source, heap, index, parentNode);
    index = parent;
  }
}

__device__ __forceinline__ void siftIntentDown(abi::RuntimeView *runtime,
                                               abi::SourceKind source,
                                               abi::IntentQueueNode *heap,
                                               std::uint32_t size,
                                               std::uint32_t index) {
  for (;;) {
    const std::uint32_t left = index * 2U + 1U;
    if (left >= size) {
      return;
    }
    const std::uint32_t right = left + 1U;
    const std::uint32_t first =
        right < size && intentQueueNodePrecedes(runtime, source, heap[right],
                                                heap[left])
            ? right
            : left;
    if (!intentQueueNodePrecedes(runtime, source, heap[first], heap[index])) {
      return;
    }
    const abi::IntentQueueNode childNode = heap[first];
    const abi::IntentQueueNode parentNode = heap[index];
    setIntentHeapNode(runtime, source, heap, index, childNode);
    setIntentHeapNode(runtime, source, heap, first, parentNode);
    index = first;
  }
}

__device__ __forceinline__ void
removeIntentHeapAt(abi::RuntimeView *runtime, abi::SourceKind source,
                   abi::IntentQueueControl &control, abi::IntentQueueNode *heap,
                   std::uint32_t index) {
  if (index >= control.size) {
    return;
  }
  const abi::IntentQueueNode removed = heap[index];
  const std::uint32_t lastIndex = control.size - 1U;
  const abi::IntentQueueNode last = heap[lastIndex];
  control.size = lastIndex;
  heap[lastIndex] = {0, abi::InvalidIndex, 0};
  if (removed.slotIndex < runtime->intentCapacity) {
    abi::IntentQueueEntry &entry =
        runtime->intentQueueEntries[removed.slotIndex];
    if (entry.intentSequence == removed.intentSequence &&
        entry.sourceKind == static_cast<std::uint32_t>(source)) {
      entry.heapIndex = abi::InvalidIndex;
    }
  }
  if (index == lastIndex) {
    return;
  }
  setIntentHeapNode(runtime, source, heap, index, last);
  if (index != 0 && intentQueueNodePrecedes(runtime, source, heap[index],
                                            heap[(index - 1U) / 2U])) {
    siftIntentUp(runtime, source, heap, index);
  } else {
    siftIntentDown(runtime, source, heap, control.size, index);
  }
}

__device__ __forceinline__ void
compactIntentHeap(abi::RuntimeView *runtime, abi::SourceKind source,
                  abi::IntentQueueControl &control,
                  abi::IntentQueueNode *heap) {
  const std::uint32_t oldSize = min(control.size, runtime->intentCapacity);
  std::uint32_t write = 0;
  for (std::uint32_t read = 0; read < oldSize; ++read) {
    abi::IntentQueueEntry *entry = nullptr;
    if (intentQueueNodeCurrent(runtime, source, heap[read], entry)) {
      setIntentHeapNode(runtime, source, heap, write++, heap[read]);
    }
  }
  for (std::uint32_t index = write; index < oldSize; ++index) {
    heap[index] = {0, abi::InvalidIndex, 0};
  }
  control.size = write;
  for (std::uint32_t parent = write / 2U; parent != 0; --parent) {
    siftIntentDown(runtime, source, heap, write, parent - 1U);
  }
}

__device__ __forceinline__ bool pushIntentHeap(abi::RuntimeView *runtime,
                                               abi::SourceKind source,
                                               abi::IntentQueueControl &control,
                                               abi::IntentQueueNode *heap,
                                               std::uint32_t slotIndex) {
  if (control.size >= runtime->intentCapacity) {
    compactIntentHeap(runtime, source, control, heap);
  }
  if (control.size >= runtime->intentCapacity) {
    return false;
  }
  abi::IntentQueueEntry &entry = runtime->intentQueueEntries[slotIndex];
  const std::uint32_t index = control.size++;
  setIntentHeapNode(runtime, source, heap, index,
                    {entry.intentSequence, slotIndex, 0});
  siftIntentUp(runtime, source, heap, index);
  return true;
}

__device__ __forceinline__ bool queueIntent(abi::RuntimeView *runtime,
                                            abi::IntentSlot &slot,
                                            abi::SourceKind source) {
  if (!intentQueueAvailable(runtime)) {
    return false;
  }
  const std::uint32_t slotIndex =
      static_cast<std::uint32_t>(&slot - runtime->intents);
  if (slotIndex >= runtime->intentCapacity) {
    return false;
  }
  abi::IntentQueueControl *control = intentQueueControl(runtime, source);
  abi::IntentQueueNode *heap = intentQueueHeap(runtime, source);
  if (control == nullptr || heap == nullptr) {
    return false;
  }
  abi::IntentQueueEntry &entry = runtime->intentQueueEntries[slotIndex];
  if (atomicCAS(&entry.state,
                static_cast<std::uint32_t>(abi::IntentQueueState::Free),
                static_cast<std::uint32_t>(abi::IntentQueueState::Preparing)) !=
      static_cast<std::uint32_t>(abi::IntentQueueState::Free)) {
    return false;
  }
  const std::uint64_t service = criticalServiceNs(runtime, slot.intent, source);
  lockIntentQueue(*control);
  const std::uint32_t sourceIndex = static_cast<std::uint32_t>(source);
  const bool current =
      atomicAdd(&entry.state, 0U) ==
          static_cast<std::uint32_t>(abi::IntentQueueState::Preparing) &&
      slot.epoch == currentEpoch(runtime) && slot.sourceKind == sourceIndex &&
      atomicAdd(&slot.intent.valid, 0U) == 1U;
  bool queued = false;
  if (current) {
    entry.intentSequence = slot.sequence;
    entry.stableSequence = control->nextSequence++;
    entry.deadlineClock = slot.intent.deadlineClock;
    entry.criticalServiceNs = service;
    entry.heapIndex = abi::InvalidIndex;
    entry.epoch = slot.epoch;
    entry.sourceKind = sourceIndex;
    entry.priority = slot.intent.priority;
    __threadfence();
    atomicExch(&entry.state,
               static_cast<std::uint32_t>(abi::IntentQueueState::Queued));
    queued = pushIntentHeap(runtime, source, *control, heap, slotIndex);
  }
  if (!queued) {
    atomicExch(&entry.state,
               static_cast<std::uint32_t>(abi::IntentQueueState::Free));
  }
  unlockIntentQueue(*control);
  return queued;
}

#if defined(NTA_TEST_INTENT_QUEUE_INTERNALS)
// Raw heap operations are exposed only to the queue ordering/concurrency
// fixtures. Production dispatch must use claimAdmissibleIntent so byte-credit
// constraints cannot turn pop/requeue into same-round starvation.
__device__ __forceinline__ std::uint32_t popIntent(abi::RuntimeView *runtime,
                                                   abi::SourceKind source) {
  if (!intentQueueAvailable(runtime)) {
    return abi::InvalidIndex;
  }
  abi::IntentQueueControl *control = intentQueueControl(runtime, source);
  abi::IntentQueueNode *heap = intentQueueHeap(runtime, source);
  if (control == nullptr || heap == nullptr) {
    return abi::InvalidIndex;
  }
  lockIntentQueue(*control);
  std::uint32_t selected = abi::InvalidIndex;
  while (control->size != 0) {
    const abi::IntentQueueNode node = heap[0];
    abi::IntentQueueEntry *entry = nullptr;
    const bool current = intentQueueNodeCurrent(runtime, source, node, entry);
    removeIntentHeapAt(runtime, source, *control, heap, 0);
    if (!current && node.slotIndex < runtime->intentCapacity) {
      abi::IntentQueueEntry &stale =
          runtime->intentQueueEntries[node.slotIndex];
      if (stale.intentSequence == node.intentSequence) {
        (void)atomicCAS(
            &stale.state,
            static_cast<std::uint32_t>(abi::IntentQueueState::Queued),
            static_cast<std::uint32_t>(abi::IntentQueueState::Free));
      }
    }
    if (current &&
        atomicCAS(&entry->state,
                  static_cast<std::uint32_t>(abi::IntentQueueState::Queued),
                  static_cast<std::uint32_t>(abi::IntentQueueState::Claimed)) ==
            static_cast<std::uint32_t>(abi::IntentQueueState::Queued)) {
      selected = node.slotIndex;
      break;
    }
  }
  unlockIntentQueue(*control);
  return selected;
}

__device__ __forceinline__ bool requeueIntent(abi::RuntimeView *runtime,
                                              std::uint32_t slotIndex,
                                              abi::SourceKind source) {
  if (!intentQueueAvailable(runtime) || slotIndex >= runtime->intentCapacity) {
    return false;
  }
  abi::IntentQueueEntry &entry = runtime->intentQueueEntries[slotIndex];
  if (atomicCAS(&entry.state,
                static_cast<std::uint32_t>(abi::IntentQueueState::Claimed),
                static_cast<std::uint32_t>(abi::IntentQueueState::Preparing)) !=
      static_cast<std::uint32_t>(abi::IntentQueueState::Claimed)) {
    return false;
  }
  abi::IntentSlot &slot = runtime->intents[slotIndex];
  abi::IntentQueueControl *control = intentQueueControl(runtime, source);
  abi::IntentQueueNode *heap = intentQueueHeap(runtime, source);
  if (control == nullptr || heap == nullptr) {
    atomicExch(&entry.state,
               static_cast<std::uint32_t>(abi::IntentQueueState::Claimed));
    return false;
  }
  const std::uint64_t service = criticalServiceNs(runtime, slot.intent, source);
  lockIntentQueue(*control);
  const std::uint32_t sourceIndex = static_cast<std::uint32_t>(source);
  const bool current =
      atomicAdd(&entry.state, 0U) ==
          static_cast<std::uint32_t>(abi::IntentQueueState::Preparing) &&
      entry.intentSequence == slot.sequence &&
      entry.epoch == currentEpoch(runtime) && slot.epoch == entry.epoch &&
      entry.sourceKind == sourceIndex && slot.sourceKind == sourceIndex &&
      atomicAdd(&slot.intent.valid, 0U) == 1U;
  bool queued = false;
  if (current) {
    entry.deadlineClock = slot.intent.deadlineClock;
    entry.criticalServiceNs = service;
    entry.heapIndex = abi::InvalidIndex;
    entry.priority = slot.intent.priority;
    __threadfence();
    atomicExch(&entry.state,
               static_cast<std::uint32_t>(abi::IntentQueueState::Queued));
    queued = pushIntentHeap(runtime, source, *control, heap, slotIndex);
  }
  if (!queued) {
    atomicExch(&entry.state,
               static_cast<std::uint32_t>(abi::IntentQueueState::Free));
  }
  unlockIntentQueue(*control);
  return queued;
}
#endif

__device__ __forceinline__ void
retireIntentQueueEntry(abi::RuntimeView *runtime, std::uint32_t slotIndex) {
  if (!intentQueueAvailable(runtime) || slotIndex >= runtime->intentCapacity) {
    return;
  }
  abi::IntentQueueEntry &entry = runtime->intentQueueEntries[slotIndex];
  const std::uint32_t queued =
      static_cast<std::uint32_t>(abi::IntentQueueState::Queued);
  if (atomicAdd(&entry.state, 0U) != queued) {
    atomicExch(&entry.state,
               static_cast<std::uint32_t>(abi::IntentQueueState::Free));
    return;
  }
  const std::uint64_t intentSequence = entry.intentSequence;
  const std::uint32_t sourceIndex = entry.sourceKind;
  const std::uint32_t heapIndex = entry.heapIndex;
  if (atomicCAS(&entry.state, queued,
                static_cast<std::uint32_t>(abi::IntentQueueState::Free)) !=
      queued) {
    atomicExch(&entry.state,
               static_cast<std::uint32_t>(abi::IntentQueueState::Free));
    return;
  }
  if (sourceIndex >= runtime->backendCapacity ||
      sourceIndex >= abi::BackendCount) {
    return;
  }

  // Directly claimed indexed transfers and cancellation can retire an intent
  // without popIntent. Remove that node when the queue is immediately
  // available, but never make completion wait behind another queue user. A
  // failed try-lock leaves a generation-tagged tombstone that pop or bounded
  // compaction will discard safely.
  const auto source = static_cast<abi::SourceKind>(sourceIndex);
  abi::IntentQueueControl &control = runtime->intentQueueControls[sourceIndex];
  abi::IntentQueueNode *heap = intentQueueHeap(runtime, source);
  if (heap == nullptr || !tryLockIntentQueue(control)) {
    return;
  }
  if (heapIndex < control.size && heap[heapIndex].slotIndex == slotIndex &&
      heap[heapIndex].intentSequence == intentSequence) {
    removeIntentHeapAt(runtime, source, control, heap, heapIndex);
  }
  unlockIntentQueue(control);
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

__device__ __forceinline__ void consumeIntent(abi::RuntimeView *runtime,
                                              abi::IntentSlot &slot);
__device__ __forceinline__ bool claimIntent(abi::IntentSlot &slot);

enum class DispatchCreditState : std::uint32_t {
  Ready,
  Blocked,
  Impossible,
};

struct AdmittedIntent {
  std::uint32_t slotIndex = abi::InvalidIndex;
  std::uint64_t requestBytes = 0;
  std::uint64_t backendBytes = 0;
  bool admitted = false;
  bool terminal = false;
};

// IntentQueueControl::reserved is deliberately runtime-internal state.  The
// first word records that discovery proved one finite intent range is already
// in EDF order; the second binds that proof to the exact [first, count) range.
// Dynamic workloads leave both words zero and continue to use the heap.
inline constexpr std::uint64_t OrderedIntentWindowMagic =
    0x4e54414f52444552ULL; // "NTAORDER"

__device__ __forceinline__ std::uint64_t
orderedIntentWindowGeometry(std::uint32_t firstSlot, std::uint32_t slotCount) {
  return (static_cast<std::uint64_t>(firstSlot) << 32U) | slotCount;
}

__device__ __forceinline__ DispatchCreditState
combineCreditState(DispatchCreditState left, DispatchCreditState right) {
  if (left == DispatchCreditState::Impossible ||
      right == DispatchCreditState::Impossible) {
    return DispatchCreditState::Impossible;
  }
  return left == DispatchCreditState::Blocked ||
                 right == DispatchCreditState::Blocked
             ? DispatchCreditState::Blocked
             : DispatchCreditState::Ready;
}

__device__ __forceinline__ DispatchCreditState creditState(
    const std::uint64_t *counter, std::uint64_t maximum, std::uint64_t bytes) {
  if (bytes > maximum) {
    return DispatchCreditState::Impossible;
  }
  const std::uint64_t outstanding =
      atomicAdd(reinterpret_cast<unsigned long long *>(
                    const_cast<std::uint64_t *>(counter)),
                0ULL);
  return outstanding <= maximum - bytes ? DispatchCreditState::Ready
                                        : DispatchCreditState::Blocked;
}

__device__ __forceinline__ DispatchCreditState
intentCreditState(abi::RuntimeView *runtime, const abi::AcquireIntent &intent,
                  abi::SourceKind source) {
  abi::BackendView *backendEntry = backend(runtime, source);
  if (backendEntry == nullptr || backendEntry->active == 0) {
    return DispatchCreditState::Blocked;
  }
  DispatchCreditState state =
      creditState(&backendEntry->outstandingBytes,
                  backendEntry->maxOutstandingBytes, intent.bytes);
  if (!requestLive(runtime, intent.requestSlot, intent.generation)) {
    return state;
  }
  if (intent.requestSlot >= runtime->requestCapacity) {
    return DispatchCreditState::Impossible;
  }
  const abi::RequestContext &request = runtime->requests[intent.requestSlot];
  state = combineCreditState(state, creditState(&request.outstandingBytes,
                                                request.maxOutstandingBytes,
                                                intent.bytes));
  if (intent.tenantId >= runtime->tenantCapacity) {
    return DispatchCreditState::Impossible;
  }
  const abi::TenantContext &tenant = runtime->tenants[intent.tenantId];
  return combineCreditState(state, creditState(&tenant.outstandingBytes,
                                               tenant.maxOutstandingBytes,
                                               intent.bytes));
}

__device__ __forceinline__ bool
reserveIntentCredits(abi::RuntimeView *runtime,
                     const abi::AcquireIntent &intent, abi::SourceKind source,
                     std::uint64_t &requestBytes, std::uint64_t &backendBytes) {
  const bool live = requestLive(runtime, intent.requestSlot, intent.generation);
  if (live && !tryReserveRequestBytes(runtime, intent.requestSlot,
                                      intent.generation, intent.bytes)) {
    return false;
  }
  if (live && !tryReserveTenantBytes(runtime, intent.tenantId, intent.bytes)) {
    releaseRequestBytes(runtime, intent.requestSlot, intent.generation,
                        intent.bytes);
    return false;
  }
  if (!tryReserveBackendBytes(runtime, source, intent.bytes)) {
    if (live) {
      releaseRequestBytes(runtime, intent.requestSlot, intent.generation,
                          intent.bytes);
      releaseTenantBytes(runtime, intent.tenantId, intent.bytes);
    }
    return false;
  }
  requestBytes = live ? intent.bytes : 0;
  backendBytes = intent.bytes;
  return true;
}

__device__ __forceinline__ void
releaseIntentCredits(abi::RuntimeView *runtime,
                     const abi::AcquireIntent &intent, abi::SourceKind source,
                     std::uint64_t requestBytes, std::uint64_t backendBytes) {
  releaseRequestBytes(runtime, intent.requestSlot, intent.generation,
                      requestBytes);
  releaseTenantBytes(runtime, intent.tenantId, requestBytes);
  releaseBackendBytes(runtime, source, backendBytes);
}

__device__ __forceinline__ void
recordIntentCredits(abi::IntentSlot &slot, std::uint64_t requestBytes,
                    std::uint64_t backendBytes) {
  slot.chargedRequestBytes = requestBytes;
  slot.chargedBackendBytes = backendBytes;
}

__device__ __forceinline__ void
releaseRecordedIntentCredits(abi::RuntimeView *runtime, abi::IntentSlot &slot,
                             const abi::AcquireIntent &intent,
                             abi::SourceKind source) {
  const std::uint64_t requestBytes = slot.chargedRequestBytes;
  const std::uint64_t backendBytes = slot.chargedBackendBytes;
  slot.chargedRequestBytes = 0;
  slot.chargedBackendBytes = 0;
  releaseIntentCredits(runtime, intent, source, requestBytes, backendBytes);
}

// Pop the earliest intent that can actually reserve all of its dispatch
// credits. A plain heap-root pop is not sufficient here: an earlier request
// can be temporarily blocked by its byte window while a later request remains
// feasible. Requeueing the blocked root lets the same progress launch select
// it repeatedly and can starve the feasible work. The uncontended fast path is
// still O(log C); only a constrained dispatch scans the bounded heap and then
// removes the earliest feasible node in O(C + log C).
__device__ __forceinline__ AdmittedIntent
claimAdmissibleIntent(abi::RuntimeView *runtime, abi::SourceKind source) {
  AdmittedIntent result{};
  abi::IntentQueueControl *control = intentQueueControl(runtime, source);
  abi::IntentQueueNode *heap = intentQueueHeap(runtime, source);
  if (control == nullptr || heap == nullptr) {
    return result;
  }

  lockIntentQueue(*control);
  std::uint32_t readyIndex = abi::InvalidIndex;
  std::uint32_t impossibleIndex = abi::InvalidIndex;
  if (control->size != 0) {
    abi::IntentQueueEntry *root = nullptr;
    if (intentQueueNodeCurrent(runtime, source, heap[0], root)) {
      const DispatchCreditState rootState = intentCreditState(
          runtime, runtime->intents[heap[0].slotIndex].intent, source);
      if (rootState == DispatchCreditState::Ready) {
        readyIndex = 0;
      } else if (rootState == DispatchCreditState::Impossible) {
        impossibleIndex = 0;
      }
    }
  }
  for (std::uint32_t index = 0;
       readyIndex == abi::InvalidIndex && index < control->size; ++index) {
    abi::IntentQueueEntry *entry = nullptr;
    if (!intentQueueNodeCurrent(runtime, source, heap[index], entry)) {
      continue;
    }
    const abi::IntentSlot &slot = runtime->intents[heap[index].slotIndex];
    const DispatchCreditState state =
        intentCreditState(runtime, slot.intent, source);
    std::uint32_t *selected = nullptr;
    if (state == DispatchCreditState::Ready) {
      selected = &readyIndex;
    } else if (state == DispatchCreditState::Impossible) {
      selected = &impossibleIndex;
    } else {
      continue;
    }
    if (*selected == abi::InvalidIndex) {
      *selected = index;
      continue;
    }
    abi::IntentQueueEntry *previous = nullptr;
    if (!intentQueueNodeCurrent(runtime, source, heap[*selected], previous) ||
        intentQueueEntryPrecedes(*entry, *previous)) {
      *selected = index;
    }
  }

  const bool terminal =
      readyIndex == abi::InvalidIndex && impossibleIndex != abi::InvalidIndex;
  const std::uint32_t selectedIndex =
      readyIndex != abi::InvalidIndex ? readyIndex : impossibleIndex;
  if (selectedIndex == abi::InvalidIndex) {
    unlockIntentQueue(*control);
    return result;
  }

  const abi::IntentQueueNode selectedNode = heap[selectedIndex];
  abi::IntentQueueEntry &entry =
      runtime->intentQueueEntries[selectedNode.slotIndex];
  abi::IntentSlot &slot = runtime->intents[selectedNode.slotIndex];
  if (!terminal &&
      !reserveIntentCredits(runtime, slot.intent, source, result.requestBytes,
                            result.backendBytes)) {
    unlockIntentQueue(*control);
    return {};
  }
  const std::uint32_t queued =
      static_cast<std::uint32_t>(abi::IntentQueueState::Queued);
  if (atomicCAS(&entry.state, queued,
                static_cast<std::uint32_t>(abi::IntentQueueState::Claimed)) !=
      queued) {
    releaseIntentCredits(runtime, slot.intent, source, result.requestBytes,
                         result.backendBytes);
    unlockIntentQueue(*control);
    return {};
  }
  removeIntentHeapAt(runtime, source, *control, heap, selectedIndex);
  if (!claimIntent(slot)) {
    atomicExch(&entry.state,
               static_cast<std::uint32_t>(abi::IntentQueueState::Free));
    releaseIntentCredits(runtime, slot.intent, source, result.requestBytes,
                         result.backendBytes);
    unlockIntentQueue(*control);
    return {};
  }
  result.slotIndex = selectedNode.slotIndex;
  result.admitted = !terminal;
  result.terminal = terminal;
  unlockIntentQueue(*control);
  return result;
}

// Fast path for a finite, simultaneously released EDF window whose producer
// has proved that intent slots are in nondecreasing deadline order. The cursor
// advances in O(1) when the earliest intent has credits. If that intent is
// temporarily blocked, a bounded forward scan retains the generic queue's
// work-conserving behavior without paying heap maintenance on every 4-KiB I/O.
inline constexpr std::uint32_t OrderedIntentValidationThreads = 256;

struct OrderedIntentValidationScratch {
  abi::IntentQueueEntry entries[OrderedIntentValidationThreads];
  std::uint32_t present[OrderedIntentValidationThreads];
  std::uint32_t ordered;
  std::uint32_t havePrevious;
  abi::IntentQueueEntry previous;
};

// Validate one finite intent image collectively. Global slot reads and EDF-key
// construction are parallel; lane zero performs only the ordered comparison in
// shared memory. Invalid images remain fail-closed and are queued through the
// generic heap by the caller.
__device__ __forceinline__ bool
validateOrderedIntentWindow(abi::RuntimeView *runtime, abi::SourceKind source,
                            std::uint32_t firstSlot, std::uint32_t slotCount,
                            OrderedIntentValidationScratch &scratch) {
  if (runtime == nullptr || runtime->intents == nullptr || slotCount == 0 ||
      blockDim.x != OrderedIntentValidationThreads || blockDim.y != 1 ||
      blockDim.z != 1 || firstSlot > runtime->intentCapacity ||
      slotCount > runtime->intentCapacity - firstSlot) {
    return false;
  }
  abi::IntentQueueControl *control = intentQueueControl(runtime, source);
  if (control == nullptr) {
    return false;
  }
  if (threadIdx.x == 0) {
    control->reserved[0] = 0;
    control->reserved[1] = 0;
    scratch.ordered = control->size == 0 ? 1U : 0U;
    scratch.havePrevious = 0;
  }
  __syncthreads();

  const std::uint32_t end = firstSlot + slotCount;
  const std::uint32_t sourceIndex = static_cast<std::uint32_t>(source);
  for (std::uint32_t base = firstSlot; base < end;
       base += OrderedIntentValidationThreads) {
    const std::uint32_t index = base + threadIdx.x;
    scratch.present[threadIdx.x] = 0;
    if (index < end) {
      abi::IntentSlot &slot = runtime->intents[index];
      if (atomicAdd(&slot.intent.valid, 0U) == 1U &&
          slot.epoch == currentEpoch(runtime)) {
        if (slot.sourceKind != sourceIndex || slot.intent.objectSlot != index ||
            slot.intent.deadlineClock == 0) {
          atomicExch(&scratch.ordered, 0U);
        } else {
          abi::IntentQueueEntry current{};
          current.intentSequence = slot.sequence;
          current.stableSequence = index;
          current.deadlineClock = slot.intent.deadlineClock;
          current.criticalServiceNs =
              criticalServiceNs(runtime, slot.intent, source);
          current.priority = slot.intent.priority;
          scratch.entries[threadIdx.x] = current;
          scratch.present[threadIdx.x] = 1U;
        }
      }
    }
    __syncthreads();
    if (threadIdx.x == 0 && scratch.ordered != 0) {
      const std::uint32_t remaining = end - base;
      const std::uint32_t chunkCount =
          remaining < OrderedIntentValidationThreads
              ? remaining
              : OrderedIntentValidationThreads;
      for (std::uint32_t offset = 0; offset < chunkCount; ++offset) {
        if (scratch.present[offset] == 0) {
          continue;
        }
        const abi::IntentQueueEntry current = scratch.entries[offset];
        if (scratch.havePrevious != 0 &&
            intentQueueEntryPrecedes(current, scratch.previous)) {
          scratch.ordered = 0;
          break;
        }
        scratch.previous = current;
        scratch.havePrevious = 1U;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && scratch.ordered != 0) {
    control->size = firstSlot;
    control->reserved[1] = orderedIntentWindowGeometry(firstSlot, slotCount);
    __threadfence();
    control->reserved[0] = OrderedIntentWindowMagic;
    __threadfence();
  }
  __syncthreads();
  return scratch.ordered != 0;
}

__device__ __forceinline__ bool orderedIntentCurrent(abi::RuntimeView *runtime,
                                                     abi::SourceKind source,
                                                     std::uint32_t index) {
  if (runtime == nullptr || runtime->intents == nullptr ||
      index >= runtime->intentCapacity) {
    return false;
  }
  abi::IntentSlot &slot = runtime->intents[index];
  return atomicAdd(&slot.intent.valid, 0U) == 1U &&
         slot.epoch == currentEpoch(runtime) &&
         slot.sourceKind == static_cast<std::uint32_t>(source);
}

__device__ __forceinline__ AdmittedIntent claimOrderedAdmissibleIntent(
    abi::RuntimeView *runtime, abi::SourceKind source, std::uint32_t firstSlot,
    std::uint32_t slotCount, std::uint32_t &cursor) {
  AdmittedIntent result{};
  if (runtime == nullptr || runtime->intents == nullptr || slotCount == 0 ||
      firstSlot > runtime->intentCapacity ||
      slotCount > runtime->intentCapacity - firstSlot) {
    return result;
  }
  const std::uint32_t end = firstSlot + slotCount;
  if (cursor < firstSlot || cursor > end) {
    cursor = firstSlot;
  }
  while (cursor < end && !orderedIntentCurrent(runtime, source, cursor)) {
    ++cursor;
  }

  std::uint32_t ready = abi::InvalidIndex;
  std::uint32_t impossible = abi::InvalidIndex;
  for (std::uint32_t index = cursor; index < end; ++index) {
    if (!orderedIntentCurrent(runtime, source, index)) {
      continue;
    }
    const DispatchCreditState state =
        intentCreditState(runtime, runtime->intents[index].intent, source);
    if (state == DispatchCreditState::Ready) {
      ready = index;
      break;
    }
    if (state == DispatchCreditState::Impossible &&
        impossible == abi::InvalidIndex) {
      impossible = index;
    }
  }
  const bool terminal =
      ready == abi::InvalidIndex && impossible != abi::InvalidIndex;
  const std::uint32_t selected =
      ready != abi::InvalidIndex ? ready : impossible;
  if (selected == abi::InvalidIndex) {
    return result;
  }

  abi::IntentSlot &slot = runtime->intents[selected];
  if (!terminal &&
      !reserveIntentCredits(runtime, slot.intent, source, result.requestBytes,
                            result.backendBytes)) {
    return result;
  }
  if (!claimIntent(slot)) {
    if (!terminal) {
      releaseIntentCredits(runtime, slot.intent, source, result.requestBytes,
                           result.backendBytes);
    }
    result.requestBytes = 0;
    result.backendBytes = 0;
    return result;
  }
  if (selected == cursor) {
    ++cursor;
  }
  result.slotIndex = selected;
  result.admitted = !terminal;
  result.terminal = terminal;
  return result;
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
         loadIoCoherent(&control.abiVersion) == abi::NvmeQueueAbiVersion &&
         loadIoCoherent(&control.queueId) == queue.queueId &&
         loadIoCoherent(&control.generation) == queue.queueGeneration &&
         loadIoCoherent(&control.state) ==
             static_cast<std::uint32_t>(abi::NvmeQueueState::Online);
}

__device__ __forceinline__ void failNvmeQueue(abi::RuntimeView *runtime,
                                              abi::NvmeQueueView &queue,
                                              std::uint32_t lane,
                                              std::uint32_t error) {
  if (lane == 0) {
    // Publish the terminal state before scanning. A CTA that loses the queue
    // lease can then retire its own newly-published fallback intent if this
    // warp has already passed that slot.
    atomicExch(&queue.active, 0U);
    queue.error = error;
  }
  __syncwarp();
  for (std::uint32_t commandId = lane; commandId < queue.depth;
       commandId += warpSize) {
    abi::NvmeCommandContext &stored = queue.contexts[commandId];
    if (atomicExch(&stored.active, 0U) == 0U)
      continue;
    const abi::NvmeCommandContext context = stored;
    const bool current = context.epoch == currentEpoch(runtime);
    if (current && context.objectSlot < runtime->objectCapacity) {
      abi::ObjectEntry &object = runtime->objects[context.objectSlot];
      if (object.objectId == context.objectId &&
          object.version == context.objectVersion) {
        atomicExch(&object.state,
                   static_cast<std::uint32_t>(abi::ObjectState::Failed));
        publishObjectTransition(runtime, context.objectSlot,
                                abi::ObjectState::Failed);
      }
    }
    if (current) {
      failBoundWorkTicket(runtime, context.workTicket, context.requestSlot,
                          context.generation);
    }
    atomicAdd(reinterpret_cast<unsigned long long *>(&queue.failed), 1ULL);
    releaseRequestBytes(runtime, context.requestSlot, context.generation,
                        context.bytes);
    releaseTenantBytes(runtime, context.tenantId, context.bytes);
    releaseBackendBytes(runtime, abi::SourceKind::Nvme, context.backendBytes);
  }
  if (runtime->intentPool != nullptr && runtime->intents != nullptr) {
    const std::uint32_t intentCapacity =
        min(runtime->intentPool->capacity, runtime->intentCapacity);
    for (std::uint32_t intentIndex = lane; intentIndex < intentCapacity;
         intentIndex += warpSize) {
      abi::IntentSlot &slot = runtime->intents[intentIndex];
      if (atomicAdd(&slot.intent.valid, 0U) != 1U ||
          slot.sourceKind !=
              static_cast<std::uint32_t>(abi::SourceKind::Nvme) ||
          !claimIntent(slot)) {
        continue;
      }
      const abi::AcquireIntent intent = slot.intent;
      const bool current = slot.epoch == currentEpoch(runtime);
      if (current && intent.objectSlot < runtime->objectCapacity) {
        abi::ObjectEntry &object = runtime->objects[intent.objectSlot];
        if (object.objectId == intent.objectId &&
            object.version == intent.objectVersion) {
          atomicExch(&object.state,
                     static_cast<std::uint32_t>(abi::ObjectState::Failed));
          publishObjectTransition(runtime, intent.objectSlot,
                                  abi::ObjectState::Failed);
        }
      }
      if (current) {
        failBoundWorkTicket(runtime, intent.workTicket, intent.requestSlot,
                            intent.generation);
      }
      atomicAdd(reinterpret_cast<unsigned long long *>(&queue.failed), 1ULL);
      consumeIntent(runtime, slot);
    }
  }
  __syncwarp();
  if (lane == 0) {
    queue.outstanding = 0;
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
loadCounter(const std::uint64_t *counter) {
  return atomicAdd(reinterpret_cast<unsigned long long *>(
                       const_cast<std::uint64_t *>(counter)),
                   0ULL);
}

__device__ __forceinline__ std::uint64_t
replicaReadyCost(const abi::ReplicaEntry &replica,
                 const abi::BackendView &backend, std::uint64_t bytes) {
  if (replica.estimatedBandwidthBytesPerSecond == 0) {
    return UINT64_MAX;
  }
  const auto transferTime = [&](std::uint64_t queuedBytes) {
    return queuedBytes > UINT64_MAX / 1'000'000'000ULL
               ? UINT64_MAX
               : queuedBytes * 1'000'000'000ULL /
                     replica.estimatedBandwidthBytesPerSecond;
  };
  const std::uint64_t transfer = transferTime(bytes);
  const std::uint64_t queued =
      transferTime(loadCounter(&backend.outstandingBytes));
  if (replica.estimatedLatencyNs > UINT64_MAX - transfer ||
      replica.estimatedLatencyNs + transfer > UINT64_MAX - queued) {
    return UINT64_MAX;
  }
  return replica.estimatedLatencyNs + transfer + queued;
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
                                              abi::IntentSlot &slot,
                                              abi::SourceKind source,
                                              bool queueImmediately) {
  slot.sourceKind = static_cast<std::uint32_t>(source);
  slot.epoch = currentEpoch(runtime);
  recordIntentCredits(slot, 0, 0);
  __threadfence();
  atomicAdd(
      reinterpret_cast<unsigned long long *>(&runtime->intentPool->enqueued),
      1ULL);
  atomicAdd(&runtime->intentPool->active, 1U);
  __threadfence();
  atomicExch(&slot.intent.valid, 1U);
  __threadfence();
  if (queueImmediately && !queueIntent(runtime, slot, source) &&
      intentQueueAvailable(runtime)) {
    atomicAdd(&runtime->intentPool->overflow, 1U);
  }
}

__device__ __forceinline__ bool claimIntent(abi::IntentSlot &slot) {
  return atomicCAS(&slot.intent.valid, 1U, 2U) == 1U;
}

__device__ __forceinline__ void consumeIntent(abi::RuntimeView *runtime,
                                              abi::IntentSlot &slot) {
  const auto source = static_cast<abi::SourceKind>(slot.sourceKind);
  abi::BackendView *entry = backend(runtime, source);
  if (entry != nullptr) {
    atomicAdd(
        reinterpret_cast<unsigned long long *>(&entry->pendingAcquisitions),
        0ULL - 1ULL);
  }
  if (intentQueueAvailable(runtime)) {
    const std::uint32_t slotIndex =
        static_cast<std::uint32_t>(&slot - runtime->intents);
    if (slotIndex < runtime->intentCapacity) {
      retireIntentQueueEntry(runtime, slotIndex);
    }
  }
  atomicAdd(reinterpret_cast<unsigned long long *>(&slot.sequence), 1ULL);
  __threadfence();
  atomicExch(&slot.intent.valid, 0U);
  atomicAdd(
      reinterpret_cast<unsigned long long *>(&runtime->intentPool->consumed),
      1ULL);
  atomicSub(&runtime->intentPool->active, 1U);
}

enum class TryIssueResult : std::uint32_t {
  Unavailable,
  Issued,
  Failed,
};

__device__ __forceinline__ void
recordDirectFallback(abi::NvmeQueueView &queue) {
  atomicAdd(reinterpret_cast<unsigned long long *>(&queue.directFallbacks),
            1ULL);
}

__device__ __forceinline__ bool
validNvmeTransfer(abi::RuntimeView *runtime, const abi::NvmeQueueView &queue,
                  const abi::AcquireIntent &intent,
                  const abi::ObjectEntry &object,
                  const abi::ReplicaEntry *replica) {
  const std::uint64_t firstByteOffset =
      replica == nullptr ? UINT64_MAX : replica->transferShape;
  if (replica == nullptr || queue.controllerPageSize == 0 ||
      queue.lbaShift >= 32U || object.bytes == 0 ||
      queue.submissions == nullptr || queue.contexts == nullptr ||
      queue.completions == nullptr || queue.sqDoorbell == nullptr ||
      queue.cqDoorbell == nullptr || object.objectId != intent.objectId ||
      object.version != intent.objectVersion || intent.offset != 0 ||
      intent.bytes != object.bytes || replica->dmaPageListAddress == 0 ||
      firstByteOffset >= queue.controllerPageSize ||
      object.bytes > UINT64_MAX - (queue.controllerPageSize - 1U)) {
    return false;
  }
  const auto *dmaPages =
      reinterpret_cast<const std::uint64_t *>(replica->dmaPageListAddress);
  const std::uint64_t expectedPages64 =
      (firstByteOffset + object.bytes + queue.controllerPageSize - 1U) /
      queue.controllerPageSize;
  // PRP1 names the first (possibly offset) data page. A one-page PRP list can
  // therefore name controllerPageSize / 8 additional pages. In particular,
  // an MDTS-sized transfer into a mid-page HBM destination legitimately uses
  // one more total page than the aligned form.
  const std::uint64_t maxPrpPages =
      static_cast<std::uint64_t>(queue.controllerPageSize) /
          sizeof(std::uint64_t) +
      1U;
  const std::uint64_t lbaSize = 1ULL << queue.lbaShift;
  const std::uint64_t lbaCount = object.bytes >> queue.lbaShift;
  return expectedPages64 <= UINT32_MAX &&
         replica->dmaPageCount == static_cast<std::uint32_t>(expectedPages64) &&
         dmaPages[0] % queue.controllerPageSize == 0 &&
         object.bytes % lbaSize == 0 && replica->sourceAddress % lbaSize == 0 &&
         lbaCount != 0 && lbaCount <= 65'536ULL &&
         replica->dmaPageCount <= maxPrpPages &&
         (replica->dmaPageCount <= 2 ||
          (queue.prpLists != nullptr && queue.prpListDmaAddress != 0)) &&
         replica->sourceKind ==
             static_cast<std::uint32_t>(abi::SourceKind::Nvme) &&
         intent.objectSlot < runtime->objectCapacity;
}

struct NvmeAdmission {
  std::uint64_t requestBytes;
  std::uint64_t backendBytes;
  bool admitted;
};

inline constexpr std::uint32_t NvmeIssueBatchCapacity = 64;
inline constexpr std::uint32_t NvmeCompletionBatchCapacity = 64;

struct NvmeCompletionDescriptor {
  std::uint32_t commandId;
  std::uint32_t status;
};

struct NvmeCompletionBatch {
  NvmeCompletionDescriptor entries[NvmeCompletionBatchCapacity];
  std::uint32_t count;
  std::uint32_t nextHead;
  std::uint32_t nextPhase;
  std::uint32_t malformed;
  std::uint32_t firstError;
};

struct NvmeIssueDescriptor {
  std::uint32_t intentSlotIndex;
  std::uint32_t objectSlot;
  std::uint32_t commandId;
  std::uint32_t submissionSlot;
  std::uint32_t dmaPageCount;
  std::uint64_t requestBytes;
  std::uint64_t backendBytes;
};

struct NvmeIssueBatch {
  NvmeIssueDescriptor entries[NvmeIssueBatchCapacity];
  std::uint32_t count;
  std::uint32_t maximumDmaPageCount;
};

// Consume a bounded CQ prefix as one I/O-coherence transaction. NVMe publishes
// each completion only after its DMA is globally visible; lane zero first
// observes and claims a contiguous phase-valid prefix, then one system fence
// orders all of those observations before warp-parallel readiness publication.
// Context state 2 is private to the queue lease and prevents a malformed
// duplicate command ID from being processed twice.
__device__ __forceinline__ bool
drainNvmeCompletionBatch(abi::RuntimeView *runtime, abi::NvmeQueueView &queue,
                         std::uint32_t completionBudget, std::uint32_t lane,
                         NvmeCompletionBatch &batch) {
  if (lane == 0) {
    batch.count = 0;
    batch.nextHead = queue.cqHead;
    batch.nextPhase = queue.cqPhase;
    batch.malformed = 0;
    batch.firstError = 0;
    const std::uint32_t maximum = min(
        min(completionBudget, NvmeCompletionBatchCapacity), queue.outstanding);
    while (batch.count < maximum) {
      abi::NvmeCompletion &completion = queue.completions[batch.nextHead];
      const std::uint32_t commandAndStatus =
          loadIoCoherent(&completion.dword[3]);
      const std::uint32_t statusField = commandAndStatus >> 16U;
      if ((statusField & 1U) != batch.nextPhase) {
        break;
      }
      const std::uint32_t commandId = commandAndStatus & 0xffffU;
      if (commandId >= queue.depth ||
          atomicCAS(&queue.contexts[commandId].active, 1U, 2U) != 1U) {
        batch.malformed = 1;
        break;
      }
      batch.entries[batch.count++] = {commandId, statusField >> 1U};
      ++batch.nextHead;
      if (batch.nextHead == queue.depth) {
        batch.nextHead = 0;
        batch.nextPhase ^= 1U;
      }
    }
    if (batch.count != 0) {
      systemIoFence();
    }
  }
  __syncwarp();

  std::uint32_t completed = 0;
  std::uint32_t failed = 0;
  for (std::uint32_t index = lane; index < batch.count; index += warpSize) {
    const NvmeCompletionDescriptor completion = batch.entries[index];
    abi::NvmeCommandContext &stored = queue.contexts[completion.commandId];
    const abi::NvmeCommandContext context = stored;
    bool objectCurrent = context.epoch == currentEpoch(runtime) &&
                         context.objectSlot < runtime->objectCapacity;
    if (objectCurrent) {
      abi::ObjectEntry &object = runtime->objects[context.objectSlot];
      const abi::ReplicaEntry *selected =
          replica(runtime, object, object.selectedReplica);
      objectCurrent = object.objectId == context.objectId &&
                      object.version == context.objectVersion &&
                      selected != nullptr &&
                      selected->sourceKind ==
                          static_cast<std::uint32_t>(abi::SourceKind::Nvme);
      if (objectCurrent && completion.status == 0) {
        atomicExch(&object.state,
                   static_cast<std::uint32_t>(abi::ObjectState::Ready));
        publishObjectTransition(runtime, context.objectSlot,
                                abi::ObjectState::Ready);
        ++completed;
      } else if (objectCurrent) {
        atomicExch(&object.state,
                   static_cast<std::uint32_t>(abi::ObjectState::Failed));
        publishObjectTransition(runtime, context.objectSlot,
                                abi::ObjectState::Failed);
        failBoundWorkTicket(runtime, context.workTicket, context.requestSlot,
                            context.generation);
        ++failed;
        atomicCAS(&batch.firstError, 0U, completion.status);
      }
    }
    if (!objectCurrent) {
      ++failed;
      atomicCAS(&batch.firstError, 0U, 0xfffffffbU);
    }
    releaseRequestBytes(runtime, context.requestSlot, context.generation,
                        context.bytes);
    releaseTenantBytes(runtime, context.tenantId, context.bytes);
    releaseBackendBytes(runtime, abi::SourceKind::Nvme, context.backendBytes);
    atomicExch(&stored.active, 0U);
  }

  for (std::uint32_t offset = warpSize / 2U; offset != 0; offset /= 2U) {
    completed += __shfl_down_sync(0xffffffffU, completed, offset);
    failed += __shfl_down_sync(0xffffffffU, failed, offset);
  }
  if (lane == 0 && batch.count != 0) {
    queue.completed += completed;
    queue.failed += failed;
    if (batch.firstError != 0) {
      queue.error = batch.firstError;
    }
    queue.cqHead = batch.nextHead;
    queue.cqPhase = batch.nextPhase;
    queue.outstanding -= batch.count;
    systemIoFence();
    storeMmio(queue.cqDoorbell, queue.cqHead);
  }
  __syncwarp();
  return batch.malformed != 0;
}

enum class NvmeIssueSelection : std::uint32_t {
  Empty,
  Retired,
  Ready,
  Fatal,
};

__device__ __forceinline__ NvmeAdmission
tryAdmitNvme(abi::RuntimeView *runtime, const abi::AcquireIntent &intent,
             std::uint64_t bytes) {
  const bool live = requestLive(runtime, intent.requestSlot, intent.generation);
  bool admitted = true;
  if (live) {
    admitted = tryReserveRequestBytes(runtime, intent.requestSlot,
                                      intent.generation, bytes);
    if (admitted && !tryReserveTenantBytes(runtime, intent.tenantId, bytes)) {
      releaseRequestBytes(runtime, intent.requestSlot, intent.generation,
                          bytes);
      admitted = false;
    }
  }
  if (admitted &&
      !tryReserveBackendBytes(runtime, abi::SourceKind::Nvme, bytes)) {
    if (live) {
      releaseRequestBytes(runtime, intent.requestSlot, intent.generation,
                          bytes);
      releaseTenantBytes(runtime, intent.tenantId, bytes);
    }
    admitted = false;
  }
  return {live && admitted ? bytes : 0, admitted ? bytes : 0, admitted};
}

__device__ __forceinline__ void
releaseNvmeAdmission(abi::RuntimeView *runtime,
                     const abi::AcquireIntent &intent,
                     const NvmeAdmission &admission) {
  releaseRequestBytes(runtime, intent.requestSlot, intent.generation,
                      admission.requestBytes);
  releaseTenantBytes(runtime, intent.tenantId, admission.requestBytes);
  releaseBackendBytes(runtime, abi::SourceKind::Nvme, admission.backendBytes);
}

__device__ __forceinline__ void
prepareNvmeRead(abi::NvmeQueueView &queue, const abi::ReplicaEntry &replica,
                std::uint32_t commandId, std::uint32_t submissionSlot,
                std::uint32_t worker, std::uint32_t workers) {
  abi::NvmeSubmission &submission = queue.submissions[submissionSlot];
  for (std::uint32_t dword = worker; dword < 16; dword += workers) {
    submission.dword[dword] = 0;
  }
  if (replica.dmaPageCount <= 2) {
    return;
  }
  const auto *dmaPages =
      reinterpret_cast<const std::uint64_t *>(replica.dmaPageListAddress);
  auto *prpList = reinterpret_cast<std::uint64_t *>(
      reinterpret_cast<std::byte *>(queue.prpLists) +
      static_cast<std::uint64_t>(commandId) * queue.controllerPageSize);
  for (std::uint32_t page = worker + 1U; page < replica.dmaPageCount;
       page += workers) {
    prpList[page - 1U] = dmaPages[page];
  }
}

__device__ __forceinline__ void
publishNvmeReadState(abi::RuntimeView *runtime, abi::NvmeQueueView &queue,
                     abi::ObjectEntry &object, const abi::ReplicaEntry &replica,
                     const abi::AcquireIntent &intent,
                     const NvmeAdmission &admission, std::uint32_t commandId,
                     std::uint32_t submissionSlot,
                     abi::IntentSlot *consumedIntent) {
  abi::NvmeSubmission &submission = queue.submissions[submissionSlot];
  const auto *dmaPages =
      reinterpret_cast<const std::uint64_t *>(replica.dmaPageListAddress);
  const std::uint64_t firstPrp = dmaPages[0] + replica.transferShape;
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
  context.bytes = admission.requestBytes;
  context.backendBytes = admission.backendBytes;
  context.mappingKey = replica.dmaPageListAddress;
  context.objectSlot = intent.objectSlot;
  context.objectVersion = intent.objectVersion;
  context.requestSlot = intent.requestSlot;
  context.generation = intent.generation;
  context.workTicket = intent.workTicket;
  context.tenantId = intent.tenantId;
  context.epoch = currentEpoch(runtime);
  __threadfence();
  atomicExch(&context.active, 1U);
  atomicExch(&object.state,
             static_cast<std::uint32_t>(abi::ObjectState::Issued));
  if (consumedIntent != nullptr) {
    consumeIntent(runtime, *consumedIntent);
  }
}

__device__ __forceinline__ void
publishNvmeRead(abi::RuntimeView *runtime, abi::NvmeQueueView &queue,
                abi::ObjectEntry &object, const abi::ReplicaEntry &replica,
                const abi::AcquireIntent &intent,
                const NvmeAdmission &admission, std::uint32_t commandId,
                std::uint32_t submissionSlot, abi::IntentSlot *consumedIntent,
                bool directSubmission) {
  publishNvmeReadState(runtime, queue, object, replica, intent, admission,
                       commandId, submissionSlot, consumedIntent);
  queue.sqTail = (submissionSlot + 1U) % queue.depth;
  ++queue.outstanding;
  ++queue.submitted;
  if (directSubmission) {
    ++queue.directSubmitted;
  }
}

__device__ __forceinline__ void
rejectNvmeDispatch(abi::RuntimeView *runtime, abi::NvmeQueueView &queue,
                   abi::IntentSlot &selectedSlot,
                   const AdmittedIntent &dispatch, std::uint32_t error) {
  abi::AcquireIntent &selected = selectedSlot.intent;
  if (selected.objectSlot < runtime->objectCapacity) {
    abi::ObjectEntry &object = runtime->objects[selected.objectSlot];
    if (object.objectId == selected.objectId &&
        object.version == selected.objectVersion) {
      atomicExch(&object.state,
                 static_cast<std::uint32_t>(abi::ObjectState::Failed));
      publishObjectTransition(runtime, selected.objectSlot,
                              abi::ObjectState::Failed);
    }
  }
  releaseIntentCredits(runtime, selected, abi::SourceKind::Nvme,
                       dispatch.requestBytes, dispatch.backendBytes);
  failBoundWorkTicket(runtime, selected.workTicket, selected.requestSlot,
                      selected.generation);
  consumeIntent(runtime, selectedSlot);
  ++queue.failed;
  queue.error = error;
}

// Convert one already-admitted intent into a fully validated SQ descriptor.
// The same fail-closed transition is shared by generic and ordered dispatch;
// the caller chooses only the scheduling policy and construction granularity.
__device__ __forceinline__ NvmeIssueSelection
selectNvmeIssue(abi::RuntimeView *runtime, abi::NvmeQueueView &queue,
                const AdmittedIntent &dispatch, std::uint32_t submissionSlot,
                NvmeIssueDescriptor &selectedIssue) {
  const std::uint32_t intentSlotIndex = dispatch.slotIndex;
  if (intentSlotIndex == abi::InvalidIndex) {
    return NvmeIssueSelection::Empty;
  }
  if (intentSlotIndex >= runtime->intentPool->capacity ||
      intentSlotIndex >= runtime->intentCapacity) {
    ++queue.failed;
    queue.error = 0xfffffffbU;
    return NvmeIssueSelection::Fatal;
  }

  abi::IntentSlot &selectedSlot = runtime->intents[intentSlotIndex];
  abi::AcquireIntent &selected = selectedSlot.intent;
  if (dispatch.terminal || selected.objectSlot >= runtime->objectCapacity) {
    rejectNvmeDispatch(runtime, queue, selectedSlot, dispatch, 0xfffffffbU);
    return NvmeIssueSelection::Retired;
  }

  abi::ObjectEntry &object = runtime->objects[selected.objectSlot];
  const abi::ReplicaEntry *replica =
      device::replica(runtime, object, object.selectedReplica);
  if (!validNvmeTransfer(runtime, queue, selected, object, replica)) {
    rejectNvmeDispatch(runtime, queue, selectedSlot, dispatch, 0xfffffffeU);
    return NvmeIssueSelection::Retired;
  }

  std::uint32_t commandId = abi::InvalidIndex;
  for (std::uint32_t searched = 0; searched < queue.depth; ++searched) {
    const std::uint32_t candidate = (queue.cidCursor + searched) % queue.depth;
    // The sole queue owner issues at most free-CIDs - 1 commands per batch and
    // advances from the selected CID. It therefore cannot wrap to a CID chosen
    // earlier in this batch before finding another genuinely free context.
    if (atomicAdd(&queue.contexts[candidate].active, 0U) == 0) {
      commandId = candidate;
      queue.cidCursor = (candidate + 1U) % queue.depth;
      break;
    }
  }
  if (commandId == abi::InvalidIndex) {
    rejectNvmeDispatch(runtime, queue, selectedSlot, dispatch, 0xfffffffaU);
    return NvmeIssueSelection::Retired;
  }

  selectedIssue = {
      intentSlotIndex,       selected.objectSlot,   commandId,
      submissionSlot,        replica->dmaPageCount, dispatch.requestBytes,
      dispatch.backendBytes,
  };
  return NvmeIssueSelection::Ready;
}

// The CTA miss path attempts at most one lock acquisition and never observes a
// completion. Contended or policy-sensitive work falls back to the global
// intent scheduler, preserving bounded CTA residency and request fairness.
__device__ __forceinline__ TryIssueResult
tryIssueNvmeFromCta(abi::RuntimeView *runtime, const abi::AcquireIntent &intent,
                    abi::ObjectEntry &object) {
  abi::BackendView *entry = backend(runtime, abi::SourceKind::Nvme);
  abi::NvmeQueueView *queuePointer = nvmeQueue(runtime);
  if (entry == nullptr || queuePointer == nullptr ||
      (entry->flags & abi::BackendCtaTryIssue) == 0) {
    return TryIssueResult::Unavailable;
  }

  abi::NvmeQueueView &queue = *queuePointer;
  // A published intent has already entered request-aware ordering. New misses
  // must not bypass it, while simultaneous unpublished misses may still race
  // for the one-shot queue lease.
  if (runtime->intentPool == nullptr ||
      atomicAdd(&runtime->intentPool->active, 0U) != 0U ||
      atomicCAS(&queue.ownerLock, 0U, 1U) != 0U) {
    recordDirectFallback(queue);
    return TryIssueResult::Unavailable;
  }

  TryIssueResult result = TryIssueResult::Unavailable;
  const abi::ReplicaEntry *selected =
      replica(runtime, object, object.selectedReplica);
  if (queue.active == 0 || !nvmeQueueOnline(queue)) {
    atomicExch(&object.state,
               static_cast<std::uint32_t>(abi::ObjectState::Failed));
    publishObjectTransition(runtime, intent.objectSlot,
                            abi::ObjectState::Failed);
    failBoundWorkTicket(runtime, intent.workTicket, intent.requestSlot,
                        intent.generation);
    atomicAdd(reinterpret_cast<unsigned long long *>(&queue.failed), 1ULL);
    atomicCAS(&queue.error, 0U, 0xfffffffcU);
    result = TryIssueResult::Failed;
  } else if (queue.depth < 2 || queue.outstanding + 1U >= queue.depth) {
    recordDirectFallback(queue);
  } else if (!validNvmeTransfer(runtime, queue, intent, object, selected)) {
    atomicExch(&object.state,
               static_cast<std::uint32_t>(abi::ObjectState::Failed));
    publishObjectTransition(runtime, intent.objectSlot,
                            abi::ObjectState::Failed);
    failBoundWorkTicket(runtime, intent.workTicket, intent.requestSlot,
                        intent.generation);
    ++queue.failed;
    queue.error = 0xfffffffeU;
    result = TryIssueResult::Failed;
  } else if (queue.directMaxPrpPages == 0 ||
             selected->dmaPageCount > queue.directMaxPrpPages) {
    recordDirectFallback(queue);
  } else {
    const NvmeAdmission admission = tryAdmitNvme(runtime, intent, object.bytes);
    std::uint32_t commandId = abi::InvalidIndex;
    if (admission.admitted) {
      for (std::uint32_t searched = 0; searched < queue.depth; ++searched) {
        const std::uint32_t candidate =
            (queue.cidCursor + searched) % queue.depth;
        if (atomicAdd(&queue.contexts[candidate].active, 0U) == 0U) {
          commandId = candidate;
          queue.cidCursor = (candidate + 1U) % queue.depth;
          break;
        }
      }
    }

    if (!admission.admitted || commandId == abi::InvalidIndex) {
      releaseNvmeAdmission(runtime, intent, admission);
      recordDirectFallback(queue);
    } else {
      const std::uint32_t submissionSlot = queue.sqTail;
      prepareNvmeRead(queue, *selected, commandId, submissionSlot, 0, 1);
      publishNvmeRead(runtime, queue, object, *selected, intent, admission,
                      commandId, submissionSlot, nullptr, true);
      systemIoFence();
      storeMmio(queue.sqDoorbell, queue.sqTail);
      result = TryIssueResult::Issued;
    }
  }

  __threadfence();
  atomicExch(&queue.ownerLock, 0U);
  return result;
}

__device__ __forceinline__ TryIssueResult
tryIssueFromCta(abi::RuntimeView *runtime, abi::SourceKind source,
                const abi::AcquireIntent &intent, abi::ObjectEntry &object) {
  switch (source) {
  case abi::SourceKind::Nvme:
    return tryIssueNvmeFromCta(runtime, intent, object);
  case abi::SourceKind::Hbm:
  case abi::SourceKind::HostMapped:
  case abi::SourceKind::HostStaged:
  case abi::SourceKind::Cxl:
  case abi::SourceKind::Rdma:
    return TryIssueResult::Unavailable;
  }
  return TryIssueResult::Unavailable;
}

__device__ __forceinline__ bool backendAcceptsIntent(abi::RuntimeView *runtime,
                                                     abi::SourceKind source) {
  abi::BackendView *entry = backend(runtime, source);
  if (entry == nullptr || entry->active == 0) {
    return false;
  }
  if (source != abi::SourceKind::Nvme) {
    return true;
  }
  if (entry->deviceState == 0) {
    return false;
  }
  const auto *queue =
      reinterpret_cast<const abi::NvmeQueueView *>(entry->deviceState);
  return atomicAdd(const_cast<std::uint32_t *>(&queue->active), 0U) != 0U &&
         nvmeQueueOnline(*queue);
}

} // namespace nta::device

extern "C" __device__ __forceinline__ __attribute__((used)) bool
nta_request_live(nta::abi::RuntimeView *runtime, std::uint32_t requestSlot,
                 std::uint32_t generation) {
  return nta::device::requestLive(runtime, requestSlot, generation);
}

// Request-directory publication is ordered on the application stream, and the
// synchronous cancellation API waits for prior GPU work. The directory is thus
// immutable for one finite kernel launch. The compiler separately proves that
// every lane supplies the same slot and generation, so one lane per warp can
// load the entry and broadcast the CTA-uniform decision without a CTA barrier.
__device__ __forceinline__ bool
nta_request_live_warp(nta::abi::RuntimeView *runtime, std::uint32_t requestSlot,
                      std::uint32_t generation) {
  const std::uint32_t linearThread =
      threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const unsigned active = __activemask();
  const int leader = __ffs(static_cast<int>(active)) - 1;
  const bool live =
      static_cast<int>(linearThread & 31U) == leader
          ? nta::device::requestLive(runtime, requestSlot, generation)
          : false;
  return __shfl_sync(active, static_cast<unsigned>(live), leader) != 0;
}

extern "C" __device__ __forceinline__ __attribute__((used)) bool
nta_request_live_cta(nta::abi::RuntimeView *runtime, std::uint32_t requestSlot,
                     std::uint32_t generation) {
  return nta_request_live_warp(runtime, requestSlot, generation);
}

extern "C" __device__ __forceinline__ __attribute__((used)) bool
nta_request_live_work_cta(nta::abi::RuntimeView *runtime,
                          std::uint32_t requestSlot, std::uint32_t generation,
                          std::uint32_t workTicket) {
  const bool live = nta_request_live_warp(runtime, requestSlot, generation);
  if (!live && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    nta::device::failWorkTicket(runtime, workTicket,
                                nta::abi::WorkTicketState::Cancelled);
  }
  return live;
}

static __device__ __attribute__((noinline)) void *
nta_acquire_slow_impl(nta::abi::RuntimeView *runtime, std::uint32_t requestSlot,
                      std::uint32_t generation, std::uint32_t objectSlot,
                      std::uint64_t objectId, std::uint32_t objectVersion,
                      std::uint64_t offset, std::uint32_t bytes,
                      std::uint32_t workTicket, bool designatedLeader,
                      std::uint64_t deadlineClock, bool deferIntentQueue) {
  using namespace nta;
  if (!device::requestLive(runtime, requestSlot, generation)) {
    device::failWorkTicket(runtime, workTicket,
                           abi::WorkTicketState::Cancelled);
    return nullptr;
  }
  if (objectSlot >= runtime->objectCapacity ||
      workTicket >= runtime->workTicketCapacity) {
    device::failWorkTicket(runtime, workTicket, abi::WorkTicketState::Failed);
    return nullptr;
  }

  abi::ObjectEntry &object = runtime->objects[objectSlot];
  if (object.objectId != objectId || object.version != objectVersion ||
      offset > object.bytes || bytes > object.bytes - offset) {
    device::failWorkTicket(runtime, workTicket, abi::WorkTicketState::Failed);
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
    const std::uint64_t cost =
        device::replicaReadyCost(*candidate, *candidateBackend, bytes);
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
    device::failWorkTicket(runtime, workTicket, abi::WorkTicketState::Failed);
    return nullptr;
  }
  // Directory entries are acquisition tiles. A staged transfer owns
  // the whole tile, so duplicate suppression cannot alias different ranges.
  if (offset != 0 || bytes != object.bytes) {
    device::failWorkTicket(runtime, workTicket, abi::WorkTicketState::Failed);
    return nullptr;
  }
  if (state == abi::ObjectState::Ready) {
    return reinterpret_cast<std::byte *>(object.stagingAddress) + offset;
  }
  if (state == abi::ObjectState::Failed) {
    device::failWorkTicket(runtime, workTicket, abi::WorkTicketState::Failed);
    return nullptr;
  }

  if (designatedLeader) {
    abi::WorkTicket &record = runtime->workTickets[workTicket];
    const auto workTicketState =
        static_cast<abi::WorkTicketState>(atomicAdd(&record.state, 0U));
    if (workTicketState == abi::WorkTicketState::New) {
      const abi::AcquireRequirement requirement{
          0, 0, objectId, offset, objectSlot, objectVersion, bytes, 0};
      (void)device::initializeWorkTicket(runtime, requestSlot, generation,
                                         workTicket, &requirement, 1);
    }
  }

  const abi::WorkTicket &workTicketRecord = runtime->workTickets[workTicket];
  if (designatedLeader &&
      atomicAdd(const_cast<std::uint32_t *>(&workTicketRecord.state), 0U) ==
          static_cast<std::uint32_t>(abi::WorkTicketState::Pending) &&
      workTicketRecord.requestSlot == requestSlot &&
      workTicketRecord.generation == generation &&
      atomicCAS(&object.state,
                static_cast<std::uint32_t>(abi::ObjectState::New),
                static_cast<std::uint32_t>(abi::ObjectState::Queued)) ==
          static_cast<std::uint32_t>(abi::ObjectState::New)) {
    object.selectedReplica = selectedReplica;
    const abi::RequestContext &request = runtime->requests[requestSlot];
    abi::AcquireIntent pending{};
    pending.objectId = objectId;
    pending.offset = offset;
    pending.bytes = bytes;
    pending.requestSlot = requestSlot;
    pending.generation = generation;
    pending.objectSlot = objectSlot;
    pending.objectVersion = objectVersion;
    pending.workTicket = workTicket;
    pending.priority = request.priority;
    pending.tenantId = request.tenantId;
    pending.deadlineClock = deadlineClock;
    atomicAdd(reinterpret_cast<unsigned long long *>(&object.issueCount), 1ULL);

    const auto source = static_cast<abi::SourceKind>(selected->sourceKind);
    abi::BackendView *sourceBackend = device::backend(runtime, source);
    if (sourceBackend == nullptr) {
      atomicExch(&object.state,
                 static_cast<std::uint32_t>(abi::ObjectState::Failed));
      device::publishObjectTransition(runtime, objectSlot,
                                      abi::ObjectState::Failed);
      device::failBoundWorkTicket(runtime, workTicket, requestSlot, generation);
      return nullptr;
    }
    atomicAdd(reinterpret_cast<unsigned long long *>(
                  &sourceBackend->pendingAcquisitions),
              1ULL);
    const device::TryIssueResult directResult =
        device::tryIssueFromCta(runtime, source, pending, object);
    if (directResult != device::TryIssueResult::Unavailable) {
      atomicAdd(reinterpret_cast<unsigned long long *>(
                    &sourceBackend->pendingAcquisitions),
                0ULL - 1ULL);
    } else {
      std::uint32_t intentIndex = abi::InvalidIndex;
      abi::IntentSlot *intentSlot = nullptr;
      if (device::reserveIntent(runtime, objectSlot, intentIndex, intentSlot)) {
        (void)intentIndex;
        pending.valid = 2U;
        intentSlot->intent = pending;
        const bool backendQueued =
            device::backendAcceptsIntent(runtime, source);
        device::publishIntent(runtime, *intentSlot, source,
                              !deferIntentQueue || !backendQueued);
        // Discovery publishes demand and dependency identity only.  Transport
        // ownership belongs to the selected progress backend: the generic
        // queue claims EDF/credit-governed intents, while the indexed-range
        // progress kernel claims its explicit contiguous range.  In
        // particular, ReplicaIndicesValidated is an access-safety proof, not
        // permission to bypass queue ordering.
        if (!backendQueued && device::claimIntent(*intentSlot)) {
          atomicExch(&object.state,
                     static_cast<std::uint32_t>(abi::ObjectState::Failed));
          device::publishObjectTransition(runtime, objectSlot,
                                          abi::ObjectState::Failed);
          device::failBoundWorkTicket(runtime, workTicket, requestSlot,
                                      generation);
          device::consumeIntent(runtime, *intentSlot);
        }
      } else {
        atomicAdd(reinterpret_cast<unsigned long long *>(
                      &sourceBackend->pendingAcquisitions),
                  0ULL - 1ULL);
        atomicExch(&object.state,
                   static_cast<std::uint32_t>(abi::ObjectState::Failed));
        device::publishObjectTransition(runtime, objectSlot,
                                        abi::ObjectState::Failed);
        device::failBoundWorkTicket(runtime, workTicket, requestSlot,
                                    generation);
      }
    }
  }
  return nullptr;
}

extern "C" __device__ __attribute__((used, noinline)) void *
nta_acquire_slow(nta::abi::RuntimeView *runtime, std::uint32_t requestSlot,
                 std::uint32_t generation, std::uint32_t objectSlot,
                 std::uint64_t objectId, std::uint32_t objectVersion,
                 std::uint64_t offset, std::uint32_t bytes,
                 std::uint32_t workTicket) {
  const bool designatedLeader =
      threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0;
  const std::uint64_t deadlineClock =
      runtime != nullptr && requestSlot < runtime->requestCapacity
          ? runtime->requests[requestSlot].deadlineClock
          : 0;
  return nta_acquire_slow_impl(
      runtime, requestSlot, generation, objectSlot, objectId, objectVersion,
      offset, bytes, workTicket, designatedLeader, deadlineClock, false);
}

extern "C" __device__ __attribute__((used, noinline)) void *
nta_acquire_tensor_map_slow(nta::abi::RuntimeView *runtime,
                            std::uint32_t requestSlot, std::uint32_t generation,
                            std::uint32_t objectSlot, std::uint64_t objectId,
                            std::uint32_t objectVersion, std::uint64_t offset,
                            std::uint32_t bytes, std::uint32_t workTicket) {
  void *address =
      nta_acquire_slow(runtime, requestSlot, generation, objectSlot, objectId,
                       objectVersion, offset, bytes, workTicket);
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

  nta::device::failWorkTicket(runtime, workTicket,
                              nta::abi::WorkTicketState::Failed);
  return nullptr;
}

static __device__ __attribute__((noinline)) std::uint32_t
nta_acquire_set_leader_with_deadline(
    nta::abi::RuntimeView *runtime, std::uint32_t requestSlot,
    std::uint32_t generation, const nta::abi::AcquireRequirement *requirements,
    std::uint32_t requirementCount, std::uint32_t directRequirementCount,
    std::uint32_t workTicket, std::uint64_t deadlineClock,
    bool deferIntentQueue) {
  using namespace nta;
  if (!device::requestLive(runtime, requestSlot, generation)) {
    device::failWorkTicket(runtime, workTicket,
                           abi::WorkTicketState::Cancelled);
    return 0;
  }

  std::uint32_t dependencyStart = 0;
  const bool validSet =
      directRequirementCount <= requirementCount &&
      (requirementCount == 0 ||
       (requirements != nullptr &&
        (directRequirementCount == requirementCount ||
         device::dependencyRange(runtime, workTicket, requirementCount,
                                 dependencyStart))));
  (void)dependencyStart;
  if (!validSet) {
    device::failWorkTicket(runtime, workTicket, abi::WorkTicketState::Failed);
    return 0;
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
    return 1;
  }

  const auto state = static_cast<abi::WorkTicketState>(
      atomicAdd(&runtime->workTickets[workTicket].state, 0U));
  if (state == abi::WorkTicketState::Cancelled ||
      state == abi::WorkTicketState::Failed) {
    return 0;
  }
  (void)device::initializeWorkTicket(runtime, requestSlot, generation,
                                     workTicket, requirements,
                                     requirementCount);
  bool acquired = true;
  for (std::uint32_t index = 0; index < requirementCount; ++index) {
    const abi::AcquireRequirement &requirement = requirements[index];
    if (requirement.directBase != 0) {
      continue;
    }
    // The caller is the designated owner for one work item. It need not be
    // physical lane zero: the generic discovery kernel assigns one independent
    // item to each thread so it can compact many items without one tiny CTA per
    // item. Numerical CTA callers continue to use nta_acquire_slow, whose
    // wrapper preserves the lane-zero ownership contract.
    acquired &= nta_acquire_slow_impl(
                    runtime, requestSlot, generation, requirement.objectSlot,
                    requirement.objectId, requirement.objectVersion,
                    requirement.offset, requirement.bytes, workTicket, true,
                    deadlineClock, deferIntentQueue) != nullptr;
  }
  return acquired ? 1U : 0U;
}

extern "C" __device__ __attribute__((used, noinline)) std::uint32_t
nta_acquire_set_leader(nta::abi::RuntimeView *runtime,
                       std::uint32_t requestSlot, std::uint32_t generation,
                       const nta::abi::AcquireRequirement *requirements,
                       std::uint32_t requirementCount,
                       std::uint32_t directRequirementCount,
                       std::uint32_t workTicket) {
  const std::uint64_t deadlineClock =
      runtime != nullptr && requestSlot < runtime->requestCapacity
          ? runtime->requests[requestSlot].deadlineClock
          : 0;
  return nta_acquire_set_leader_with_deadline(
      runtime, requestSlot, generation, requirements, requirementCount,
      directRequirementCount, workTicket, deadlineClock, false);
}

extern "C" __device__ __forceinline__ __attribute__((used)) bool
nta_acquire_set_slow(nta::abi::RuntimeView *runtime, std::uint32_t requestSlot,
                     std::uint32_t generation,
                     const nta::abi::AcquireRequirement *requirements,
                     std::uint32_t requirementCount,
                     std::uint32_t directRequirementCount,
                     std::uint32_t workTicket) {
  // A fully direct work item has no external dependency graph to initialize.
  // Keep the request/ticket identity check, but avoid the leader/shared-memory
  // rendezvous used by the heterogeneous path.  The work-plan validator and
  // the per-requirement address helper still provide the fail-closed checks;
  // this branch only removes redundant dependency admission work.
  if (requirementCount == directRequirementCount &&
      (requirementCount == 0 || requirements != nullptr)) {
    return nta_request_live_work_cta(runtime, requestSlot, generation,
                                     workTicket);
  }
  __shared__ std::uint32_t ctaReady;
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    ctaReady = nta_acquire_set_leader(runtime, requestSlot, generation,
                                      requirements, requirementCount,
                                      directRequirementCount, workTicket);
  }
  __syncthreads();
  return ctaReady != 0;
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
__device__ __forceinline__ std::uint32_t
issueNvmeGenericSerial(nta::abi::RuntimeView *runtime,
                       nta::abi::NvmeQueueView &queue,
                       std::uint32_t issueBudget, std::uint32_t lane) {
  using namespace nta;
  std::uint32_t issued = 0U;
  for (std::uint32_t attempt = 0; attempt < issueBudget; ++attempt) {
    device::NvmeIssueDescriptor descriptor{};
    device::NvmeIssueSelection selection = device::NvmeIssueSelection::Empty;
    if (lane == 0 && queue.outstanding + 1U < queue.depth) {
      const device::AdmittedIntent dispatch =
          device::claimAdmissibleIntent(runtime, abi::SourceKind::Nvme);
      selection = device::selectNvmeIssue(runtime, queue, dispatch,
                                          queue.sqTail, descriptor);
    }

    const std::uint32_t action =
        __shfl_sync(0xffffffffU, static_cast<std::uint32_t>(selection), 0);
    if (action ==
            static_cast<std::uint32_t>(device::NvmeIssueSelection::Empty) ||
        action ==
            static_cast<std::uint32_t>(device::NvmeIssueSelection::Fatal)) {
      break;
    }
    if (action ==
        static_cast<std::uint32_t>(device::NvmeIssueSelection::Retired)) {
      continue;
    }
    descriptor.intentSlotIndex =
        __shfl_sync(0xffffffffU, descriptor.intentSlotIndex, 0);
    descriptor.objectSlot = __shfl_sync(0xffffffffU, descriptor.objectSlot, 0);
    descriptor.commandId = __shfl_sync(0xffffffffU, descriptor.commandId, 0);
    descriptor.submissionSlot =
        __shfl_sync(0xffffffffU, descriptor.submissionSlot, 0);
    descriptor.requestBytes =
        __shfl_sync(0xffffffffU, descriptor.requestBytes, 0);
    descriptor.backendBytes =
        __shfl_sync(0xffffffffU, descriptor.backendBytes, 0);

    abi::IntentSlot &selectedSlot =
        runtime->intents[descriptor.intentSlotIndex];
    abi::AcquireIntent &selected = selectedSlot.intent;
    abi::ObjectEntry &object = runtime->objects[descriptor.objectSlot];
    const abi::ReplicaEntry &replica =
        *device::replica(runtime, object, object.selectedReplica);
    device::prepareNvmeRead(queue, replica, descriptor.commandId,
                            descriptor.submissionSlot, lane, warpSize);
    __syncwarp();
    if (lane == 0) {
      const device::NvmeAdmission admission{descriptor.requestBytes,
                                            descriptor.backendBytes, true};
      device::publishNvmeRead(runtime, queue, object, replica, selected,
                              admission, descriptor.commandId,
                              descriptor.submissionSlot, &selectedSlot, false);
      ++issued;
    }
    __syncwarp();
  }
  return __shfl_sync(0xffffffffU, issued, 0);
}

__device__ __forceinline__ void
progressNvmeOnce(nta::abi::RuntimeView *runtime, std::uint32_t issueBudget,
                 std::uint32_t completionBudget, std::uint32_t orderedFirstSlot,
                 std::uint32_t orderedSlotCount, std::uint32_t *orderedCursor,
                 bool verifyControlPage) {
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
  bool ownsQueue = lane == 0 && atomicCAS(&queue.ownerLock, 0U, 1U) == 0U;
  ownsQueue = __shfl_sync(0xffffffffU, ownsQueue, 0);
  if (!ownsQueue) {
    return;
  }
  bool queueOnline =
      lane == 0 && (!verifyControlPage || device::nvmeQueueOnline(queue));
  queueOnline = __shfl_sync(0xffffffffU, queueOnline, 0);
  if (!queueOnline) {
    device::failNvmeQueue(runtime, queue, lane, 0xfffffffcU);
    if (lane == 0) {
      __threadfence();
      atomicExch(&queue.ownerLock, 0U);
    }
    return;
  }

  __shared__ device::NvmeCompletionBatch completionBatch;
  const bool malformedCompletion = device::drainNvmeCompletionBatch(
      runtime, queue, completionBudget, lane, completionBatch);
  // A command status error is terminal for this single-queue runtime.  The
  // completion itself has already released its credits; failNvmeQueue retires
  // every still-active context and every not-yet-issued intent.  Returning
  // from a completion-driven worker with queue.error set but outstanding I/O
  // left behind would otherwise strand credits and directory ownership.
  if (malformedCompletion || completionBatch.firstError != 0) {
    const std::uint32_t error = malformedCompletion
                                    ? 0xffffffffU
                                    : completionBatch.firstError;
    device::failNvmeQueue(runtime, queue, lane, error);
    if (lane == 0) {
      __threadfence();
      atomicExch(&queue.ownerLock, 0U);
    }
    return;
  }
  __syncwarp();

  std::uint32_t issued = 0;
  if (orderedSlotCount == 0 || orderedCursor == nullptr) {
    // The generic heap supports arbitrary object sizes and dynamic duplicate
    // suppression. Preserve its proven whole-warp PRP construction instead of
    // extending the finite-window optimization beyond its typed contract.
    issued = issueNvmeGenericSerial(runtime, queue, issueBudget, lane);
  } else {
    // The validated ordered cursor serializes admission, but it does not
    // require serial SQE construction. Select a bounded batch on lane zero,
    // preserving exact EDF and byte-credit order, then let warp lanes build
    // independent small commands concurrently. Large PRP lists retain a
    // cooperative whole-warp writer.
    __shared__ device::NvmeIssueBatch issueBatch;
    if (lane == 0) {
      issueBatch.count = 0;
      issueBatch.maximumDmaPageCount = 0;
      const std::uint32_t maximumBatch =
          min(issueBudget, device::NvmeIssueBatchCapacity);
      while (issueBatch.count < maximumBatch &&
             queue.outstanding + issueBatch.count + 1U < queue.depth) {
        const device::AdmittedIntent dispatch =
            device::claimOrderedAdmissibleIntent(
                runtime, abi::SourceKind::Nvme, orderedFirstSlot,
                orderedSlotCount, *orderedCursor);
        device::NvmeIssueDescriptor descriptor{};
        const device::NvmeIssueSelection selection = device::selectNvmeIssue(
            runtime, queue, dispatch,
            (queue.sqTail + issueBatch.count) % queue.depth, descriptor);
        if (selection == device::NvmeIssueSelection::Empty ||
            selection == device::NvmeIssueSelection::Fatal) {
          break;
        }
        if (selection == device::NvmeIssueSelection::Retired) {
          continue;
        }
        issueBatch.entries[issueBatch.count++] = descriptor;
        issueBatch.maximumDmaPageCount =
            max(issueBatch.maximumDmaPageCount, descriptor.dmaPageCount);
      }
    }
    __syncwarp();

    issued = issueBatch.count;
    if (issueBatch.maximumDmaPageCount <= 2U) {
      for (std::uint32_t index = lane; index < issued; index += warpSize) {
        const device::NvmeIssueDescriptor descriptor =
            issueBatch.entries[index];
        abi::IntentSlot &selectedSlot =
            runtime->intents[descriptor.intentSlotIndex];
        abi::AcquireIntent &selected = selectedSlot.intent;
        abi::ObjectEntry &object = runtime->objects[descriptor.objectSlot];
        const abi::ReplicaEntry &replica =
            *device::replica(runtime, object, object.selectedReplica);
        device::prepareNvmeRead(queue, replica, descriptor.commandId,
                                descriptor.submissionSlot, 0, 1);
        const device::NvmeAdmission admission{descriptor.requestBytes,
                                              descriptor.backendBytes, true};
        device::publishNvmeReadState(runtime, queue, object, replica, selected,
                                     admission, descriptor.commandId,
                                     descriptor.submissionSlot, &selectedSlot);
      }
      __syncwarp();
    } else {
      for (std::uint32_t index = 0; index < issued; ++index) {
        const device::NvmeIssueDescriptor descriptor =
            issueBatch.entries[index];
        abi::IntentSlot &selectedSlot =
            runtime->intents[descriptor.intentSlotIndex];
        abi::AcquireIntent &selected = selectedSlot.intent;
        abi::ObjectEntry &object = runtime->objects[descriptor.objectSlot];
        const abi::ReplicaEntry &replica =
            *device::replica(runtime, object, object.selectedReplica);
        device::prepareNvmeRead(queue, replica, descriptor.commandId,
                                descriptor.submissionSlot, lane, warpSize);
        __syncwarp();
        if (lane == 0) {
          const device::NvmeAdmission admission{descriptor.requestBytes,
                                                descriptor.backendBytes, true};
          device::publishNvmeReadState(
              runtime, queue, object, replica, selected, admission,
              descriptor.commandId, descriptor.submissionSlot, &selectedSlot);
        }
        __syncwarp();
      }
    }
    if (lane == 0 && issued != 0) {
      queue.sqTail = (queue.sqTail + issued) % queue.depth;
      queue.outstanding += issued;
      queue.submitted += issued;
    }
    __syncwarp();
  }
  queueOnline =
      lane == 0 && (!verifyControlPage || device::nvmeQueueOnline(queue));
  queueOnline = __shfl_sync(0xffffffffU, queueOnline, 0);
  if (issued != 0 && !queueOnline) {
    device::failNvmeQueue(runtime, queue, lane, 0xfffffffcU);
  } else if (lane == 0 && issued != 0) {
    device::systemIoFence();
    device::storeMmio(queue.sqDoorbell, queue.sqTail);
  }
  if (lane == 0) {
    __threadfence();
    atomicExch(&queue.ownerLock, 0U);
  }
}

extern "C" __global__ void nta_progress_nvme(nta::abi::RuntimeView *runtime,
                                             std::uint32_t issueBudget,
                                             std::uint32_t completionBudget) {
  progressNvmeOnce(runtime, issueBudget, completionBudget, 0, 0, nullptr, true);
}

__device__ __forceinline__ void
progressNvmeUntil(nta::abi::RuntimeView *runtime, std::uint32_t firstIntent,
                  std::uint32_t intentCount, std::uint32_t issueBudget,
                  std::uint32_t completionBudget, std::uint64_t timeoutNs) {
  using namespace nta;
  if (runtime == nullptr || blockIdx.x != 0 || threadIdx.x >= warpSize ||
      timeoutNs == 0 ||
      (intentCount != 0 &&
       (firstIntent > runtime->intentCapacity ||
        intentCount > runtime->intentCapacity - firstIntent))) {
    return;
  }
  const std::uint32_t lane = threadIdx.x;
  abi::IntentQueueControl *control =
      device::intentQueueControl(runtime, abi::SourceKind::Nvme);
  bool ordered = lane == 0 && intentCount != 0 && control != nullptr &&
                 control->reserved[0] == device::OrderedIntentWindowMagic &&
                 control->reserved[1] == device::orderedIntentWindowGeometry(
                                             firstIntent, intentCount);
  ordered = __shfl_sync(0xffffffffU, ordered, 0);
  const std::uint32_t orderedCount = ordered ? intentCount : 0U;
  std::uint32_t *cursor = ordered ? &control->size : nullptr;
  const std::uint64_t start = device::globalTimerNs();
  bool verifyControlPage = true;
  for (;;) {
    progressNvmeOnce(runtime, issueBudget, completionBudget, firstIntent,
                     orderedCount, cursor, verifyControlPage);
    // The control page is CPU-owned setup/lifetime state. Runtime teardown is
    // ordered after this kernel, so only the CQ is a changing I/O-coherent
    // input during steady-state polling. Re-reading five control words twice
    // per round would turn PCIe coherence into the data path.
    verifyControlPage = false;
    __syncwarp();

    bool finished = false;
    bool timedOut = false;
    if (lane == 0) {
      abi::NvmeQueueView *queue = device::nvmeQueue(runtime);
      abi::BackendView *entry = device::backend(runtime, abi::SourceKind::Nvme);
      const std::uint64_t pending =
          entry == nullptr ? 0
                           : atomicAdd(reinterpret_cast<unsigned long long *>(
                                           &entry->pendingAcquisitions),
                                       0ULL);
      finished = queue == nullptr || queue->error != 0 ||
                 (queue->outstanding == 0 && pending == 0);
      timedOut = !finished && device::globalTimerNs() - start >= timeoutNs;
    }
    finished = __shfl_sync(0xffffffffU, finished, 0);
    timedOut = __shfl_sync(0xffffffffU, timedOut, 0);
    if (finished) {
      return;
    }
    if (timedOut) {
      abi::NvmeQueueView *queue = device::nvmeQueue(runtime);
      if (queue != nullptr) {
        device::failNvmeQueue(runtime, *queue, lane, 0xfffffff9U);
      }
      return;
    }
    __nanosleep(1000U);
  }
}

extern "C" __global__ void nta_progress_nvme_until_idle(
    nta::abi::RuntimeView *runtime, std::uint32_t issueBudget,
    std::uint32_t completionBudget, std::uint64_t timeoutNs) {
  progressNvmeUntil(runtime, 0, 0, issueBudget, completionBudget, timeoutNs);
}

extern "C" __global__ void nta_progress_nvme_ordered_until_idle(
    nta::abi::RuntimeView *runtime, std::uint32_t firstIntent,
    std::uint32_t intentCount, std::uint32_t issueBudget,
    std::uint32_t completionBudget, std::uint64_t timeoutNs) {
  progressNvmeUntil(runtime, firstIntent, intentCount, issueBudget,
                    completionBudget, timeoutNs);
}

#endif

extern "C" __device__ __attribute__((used, noinline)) void
nta_defer(nta::abi::RuntimeView *runtime, std::uint32_t requestSlot,
          std::uint32_t generation, std::uint32_t workTicket) {
  using namespace nta;
  if (runtime == nullptr || workTicket >= runtime->workTicketCapacity ||
      threadIdx.x != 0) {
    return;
  }

  abi::WorkTicket &record = runtime->workTickets[workTicket];
  const auto currentState =
      static_cast<abi::WorkTicketState>(atomicAdd(&record.state, 0U));
  if (currentState == abi::WorkTicketState::Cancelled ||
      currentState == abi::WorkTicketState::Failed ||
      currentState == abi::WorkTicketState::Ready ||
      currentState == abi::WorkTicketState::Done ||
      currentState == abi::WorkTicketState::Initializing) {
    return;
  }
  if (!device::requestLive(runtime, requestSlot, generation)) {
    record.requestSlot = requestSlot;
    record.generation = generation;
    if (atomicExch(&record.state, static_cast<std::uint32_t>(
                                      abi::WorkTicketState::Cancelled)) ==
        static_cast<std::uint32_t>(abi::WorkTicketState::Pending)) {
      device::recordTerminalWork(runtime, record, abi::WorkTicketState::Pending,
                                 abi::WorkTicketState::Cancelled);
    }
    return;
  }

  const abi::RequestContext &request = runtime->requests[requestSlot];
  record.requestId = request.requestId;
  record.requestSlot = requestSlot;
  record.generation = generation;
  record.logicalTile = workTicket;
  record.epoch = device::currentEpoch(runtime);
  if (currentState == abi::WorkTicketState::New) {
    // initializeWorkTicket owns both the Pending transition and pending
    // index publication. A live New record here has no resumable dependency
    // state and must fail closed instead of becoming an invisible waiter.
    atomicExch(&record.state,
               static_cast<std::uint32_t>(abi::WorkTicketState::Failed));
  }
}

namespace nta::device {

template <bool StreamOrdered>
__device__ __forceinline__ void
commitPartial(nta::abi::RuntimeView *runtime, std::uint32_t requestSlot,
              std::uint32_t generation, std::uint32_t workTicket,
              std::uint32_t reductionGroup, std::uint32_t contributorIndex,
              std::uint32_t contributorCount,
              std::uint64_t estimatedComputeNs) {
  using namespace nta;

  // PREACQUIRED launches share the incremental module but deliberately carry
  // InvalidIndex because their stream/event fence, rather than a ticket, owns
  // completion.  The fields are CTA-uniform, so reject this no-op case before
  // the convergent barrier.  Keeping the marker and its static control-flow
  // shape intact lets the compiler pass continue to prove a complete partial
  // publication for the ticketed form.
  if (runtime == nullptr || runtime->abiVersion != abi::Version) {
    return;
  }
  if (workTicket >= runtime->workTicketCapacity) {
    return;
  }
  if (runtime->ctaCompletions == nullptr ||
      reductionGroup >= runtime->workTicketCapacity || contributorCount == 0 ||
      contributorIndex >= contributorCount) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
      device::failWorkTicket(runtime, workTicket, abi::WorkTicketState::Failed);
    }
    return;
  }

  // Every writer makes its numerical partial globally visible before the CTA
  // collectively publishes completion. The final sibling CTA may then make the
  // request-local reduction group mergeable.
  if constexpr (!StreamOrdered) {
    __threadfence();
  }
  __syncthreads();
  if (threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) {
    return;
  }
  const std::uint64_t siblingCount64 =
      static_cast<std::uint64_t>(gridDim.y) * gridDim.z;
  if (siblingCount64 == 0 || siblingCount64 > UINT32_MAX) {
    device::failWorkTicket(runtime, workTicket, abi::WorkTicketState::Failed);
    return;
  }
  const std::uint32_t siblingCount = static_cast<std::uint32_t>(siblingCount64);
  const std::uint32_t completed =
      atomicAdd(&runtime->ctaCompletions[workTicket], 1U) + 1U;
  if (completed < siblingCount) {
    return;
  }
  if (completed != siblingCount) {
    device::failWorkTicket(runtime, workTicket, abi::WorkTicketState::Failed);
    return;
  }

  abi::WorkTicket &ticket = runtime->workTickets[workTicket];
  const auto state =
      static_cast<abi::WorkTicketState>(atomicAdd(&ticket.state, 0U));
  if (state == abi::WorkTicketState::New) {
    if (!device::requestLive(runtime, requestSlot, generation)) {
      device::failWorkTicket(runtime, workTicket,
                             abi::WorkTicketState::Cancelled);
      return;
    }
    ticket.requestId = runtime->requests[requestSlot].requestId;
    ticket.requestSlot = requestSlot;
    ticket.generation = generation;
    ticket.logicalTile = workTicket;
    ticket.epoch = device::currentEpoch(runtime);
    ticket.unavailableBytes = 0;
    ticket.estimatedComputeNs = estimatedComputeNs;
    ticket.reductionGroup = reductionGroup;
    ticket.contributorCount = contributorCount;
    __threadfence();
  } else if (!device::ticketMatches(runtime, ticket, requestSlot, generation) ||
             ticket.reductionGroup != reductionGroup ||
             ticket.contributorCount != contributorCount) {
    device::failWorkTicket(runtime, workTicket, abi::WorkTicketState::Failed);
    return;
  }
  (void)device::completeWorkTicket(runtime, workTicket);
}

} // namespace nta::device

extern "C" __device__ __attribute__((used, noinline, convergent)) void
nta_commit_partial(nta::abi::RuntimeView *runtime, std::uint32_t requestSlot,
                   std::uint32_t generation, std::uint32_t workTicket,
                   std::uint32_t reductionGroup, std::uint32_t contributorIndex,
                   std::uint32_t contributorCount,
                   std::uint64_t estimatedComputeNs) {
  nta::device::commitPartial<false>(
      runtime, requestSlot, generation, workTicket, reductionGroup,
      contributorIndex, contributorCount, estimatedComputeNs);
}

extern "C" __device__ __attribute__((used, noinline, convergent)) void
nta_commit_stream_ordered_partial(
    nta::abi::RuntimeView *runtime, std::uint32_t requestSlot,
    std::uint32_t generation, std::uint32_t workTicket,
    std::uint32_t reductionGroup, std::uint32_t contributorIndex,
    std::uint32_t contributorCount, std::uint64_t estimatedComputeNs) {
  nta::device::commitPartial<true>(runtime, requestSlot, generation, workTicket,
                                   reductionGroup, contributorIndex,
                                   contributorCount, estimatedComputeNs);
}

#if NTA_DEVICE_PHASE_KERNELS
namespace nta::device {

__device__ __forceinline__ uint4 loadNoAllocate(const uint4 *address) {
  uint4 value;
#ifdef NTA_STAGING_STREAMING
  // Read-once pinned-host source: evict-first so one-shot transfer lines
  // do not displace co-residents' L2 working set (see storeNoAllocate).
  asm volatile("ld.global.cs.v4.b32 {%0,%1,%2,%3},[%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(address));
#else
  asm volatile("ld.global.L1::no_allocate.v4.b32 {%0,%1,%2,%3},[%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(address));
#endif
  return value;
}

__device__ __forceinline__ void storeNoAllocate(uint4 *address,
                                                const uint4 &value) {
#ifdef NTA_STAGING_STREAMING
  // Cache-streaming staging (prototype "polite" tier, L2 edition): staged
  // KV is written once and read once per selection epoch, so allocate it
  // evict-first at BOTH levels instead of letting fresh staging lines
  // evict co-residents' decode working set from L2. Compile-time policy;
  // the JIT shim refuses this define without a streaming-tagged cache
  // workspace so a toggled env can never silently reuse stale kernels.
  asm volatile("st.global.cs.v4.b32 [%0],{%1,%2,%3,%4};"
               :
               : "l"(address), "r"(value.x), "r"(value.y), "r"(value.z),
                 "r"(value.w)
               : "memory");
#else
  asm volatile("st.global.L1::no_allocate.v4.b32 [%0],{%1,%2,%3,%4};"
               :
               : "l"(address), "r"(value.x), "r"(value.y), "r"(value.z),
                 "r"(value.w)
               : "memory");
#endif
}

__device__ __forceinline__ void copyIndexedHostObject(
    const abi::ObjectEntry &object, const abi::ReplicaEntry &replica,
    std::uint32_t objectBlock = 0, std::uint32_t blocksPerObject = 1) {
  const auto *source =
      reinterpret_cast<const std::byte *>(replica.sourceAddress);
  auto *destination = reinterpret_cast<std::byte *>(object.stagingAddress);
  const auto *sourceIndices =
      reinterpret_cast<const std::uint32_t *>(replica.dmaPageListAddress);
  const auto *destinationIndices =
      reinterpret_cast<const std::uint32_t *>(object.stagingTensorMapAddress);
  const std::uint32_t sourceStride =
      abi::sourceTransferStride(replica.transferShape);
  const std::uint32_t destinationStride =
      abi::destinationTransferStride(replica.transferShape);
  const std::uint32_t elementBytes =
      static_cast<std::uint32_t>(object.bytes / replica.dmaPageCount);
  const bool vectorAligned =
      ((reinterpret_cast<std::uintptr_t>(source) |
        reinterpret_cast<std::uintptr_t>(destination) | sourceStride |
        destinationStride | elementBytes) &
       (alignof(uint4) - 1U)) == 0;
  constexpr std::uint32_t ThreadsPerWorker = 32;
  const std::uint32_t lane = threadIdx.x % ThreadsPerWorker;
  const std::uint32_t worker = threadIdx.x / ThreadsPerWorker;
  const std::uint32_t workersPerBlock = blockDim.x / ThreadsPerWorker;
  const std::uint32_t firstElement = objectBlock * workersPerBlock + worker;
  const std::uint32_t elementStride = workersPerBlock * blocksPerObject;
  if (vectorAligned) {
    const std::uint32_t vectorsPerElement = elementBytes / sizeof(uint4);
    for (std::uint32_t element = firstElement; element < replica.dmaPageCount;
         element += elementStride) {
      const std::uint32_t sourceIndex =
          __shfl_sync(0xffffffffU, lane == 0 ? sourceIndices[element] : 0U, 0);
      const std::uint32_t destinationIndex = __shfl_sync(
          0xffffffffU, lane == 0 ? destinationIndices[element] : 0U, 0);
      for (std::uint32_t within = lane; within < vectorsPerElement;
           within += ThreadsPerWorker) {
        auto *target =
            destination +
            static_cast<std::uint64_t>(destinationIndex) * destinationStride +
            static_cast<std::uint64_t>(within) * sizeof(uint4);
        const auto *origin =
            source + static_cast<std::uint64_t>(sourceIndex) * sourceStride +
            static_cast<std::uint64_t>(within) * sizeof(uint4);
        storeNoAllocate(
            reinterpret_cast<uint4 *>(target),
            loadNoAllocate(reinterpret_cast<const uint4 *>(origin)));
      }
    }
    return;
  }
  for (std::uint32_t element = firstElement; element < replica.dmaPageCount;
       element += elementStride) {
    const std::uint32_t sourceIndex =
        __shfl_sync(0xffffffffU, lane == 0 ? sourceIndices[element] : 0U, 0);
    const std::uint32_t destinationIndex = __shfl_sync(
        0xffffffffU, lane == 0 ? destinationIndices[element] : 0U, 0);
    for (std::uint32_t within = lane; within < elementBytes;
         within += ThreadsPerWorker) {
      destination[static_cast<std::uint64_t>(destinationIndex) *
                      destinationStride +
                  within] =
          source[static_cast<std::uint64_t>(sourceIndex) * sourceStride +
                 within];
    }
  }
}

__device__ __forceinline__ void
validateIndexedTransferIndices(const abi::ObjectEntry &object,
                               const abi::ReplicaEntry &replica,
                               std::uint32_t *invalid) {
  const auto *sourceIndices =
      reinterpret_cast<const std::uint32_t *>(replica.dmaPageListAddress);
  const auto *destinationIndices =
      reinterpret_cast<const std::uint32_t *>(object.stagingTensorMapAddress);
  const std::uint32_t sourceLimit =
      abi::sourceTransferIndexLimit(replica.tensorMapAddress);
  const std::uint32_t destinationLimit =
      abi::destinationTransferIndexLimit(replica.tensorMapAddress);
  for (std::uint32_t element = threadIdx.x; element < replica.dmaPageCount;
       element += blockDim.x) {
    if (sourceIndices[element] >= sourceLimit ||
        destinationIndices[element] >= destinationLimit) {
      atomicExch(invalid, 1U);
    }
  }
}

} // namespace nta::device

// Scheduler-selected finite prefetch. It moves registered indexed host objects
// ahead of their consumer kernels; a consumer still validates object identity,
// version, request liveness, and data availability at its CTA entry.
extern "C" __global__ __launch_bounds__(1024, 1) void nta_preload_indexed_host(
    nta::abi::RuntimeView *runtime, std::uint32_t firstObject,
    std::uint32_t objectCount) {
  using namespace nta;
  constexpr std::uint32_t BlocksPerObject = 2;
  const std::uint32_t relativeObject = blockIdx.x / BlocksPerObject;
  const std::uint32_t objectBlock = blockIdx.x % BlocksPerObject;
  const std::uint64_t slot64 =
      static_cast<std::uint64_t>(firstObject) + relativeObject;
  if (runtime == nullptr || relativeObject >= objectCount ||
      slot64 >= runtime->objectCapacity) {
    return;
  }
  const std::uint32_t slot = static_cast<std::uint32_t>(slot64);
  abi::ObjectEntry &object = runtime->objects[slot];
  const abi::ReplicaEntry *replica = device::replica(runtime, object, 0);
  const std::uint32_t sourceStride =
      replica == nullptr ? 0
                         : abi::sourceTransferStride(replica->transferShape);
  const std::uint32_t destinationStride =
      replica == nullptr
          ? 0
          : abi::destinationTransferStride(replica->transferShape);
  const std::uint32_t sourceLimit =
      replica == nullptr
          ? 0
          : abi::sourceTransferIndexLimit(replica->tensorMapAddress);
  const std::uint32_t destinationLimit =
      replica == nullptr
          ? 0
          : abi::destinationTransferIndexLimit(replica->tensorMapAddress);
  const bool valid =
      replica != nullptr &&
      replica->sourceKind ==
          static_cast<std::uint32_t>(abi::SourceKind::HostStaged) &&
      (replica->flags & abi::ReplicaIndexed) != 0 &&
      replica->sourceAddress != 0 && replica->dmaPageListAddress != 0 &&
      replica->dmaPageCount != 0 && object.stagingAddress != 0 &&
      object.stagingTensorMapAddress != 0 && object.bytes != 0 &&
      sourceLimit != 0 && destinationLimit != 0 &&
      object.bytes % replica->dmaPageCount == 0 &&
      sourceStride >= object.bytes / replica->dmaPageCount &&
      destinationStride >= object.bytes / replica->dmaPageCount;

  __shared__ std::uint32_t invalidIndex;
  if (threadIdx.x == 0) {
    invalidIndex = 0;
  }
  __syncthreads();
  if (valid) {
    device::validateIndexedTransferIndices(object, *replica, &invalidIndex);
  }
  __syncthreads();
  const bool bounded = valid && invalidIndex == 0;

  if (threadIdx.x == 0 && objectBlock == 0) {
    if (!bounded) {
      atomicExch(&object.state,
                 static_cast<std::uint32_t>(abi::ObjectState::Failed));
    } else {
      object.selectedReplica = 0;
      atomicExch(&object.state,
                 static_cast<std::uint32_t>(abi::ObjectState::Issued));
    }
  }
  if (!bounded) {
    return;
  }
  device::copyIndexedHostObject(object, *replica, objectBlock, BlocksPerObject);
  __syncthreads();
  // The stream event recorded after this finite kernel is the publication
  // boundary. Block zero updates the directory for the consumer's post-event
  // identity/availability guard; later object blocks never mutate this state.
  if (threadIdx.x == 0 && objectBlock == 0) {
    __threadfence_system();
    atomicExch(&object.state,
               static_cast<std::uint32_t>(abi::ObjectState::Ready));
  }
}

namespace nta::device {

template <std::uint32_t ElementBytes, std::uint32_t ThreadsPerRow>
__device__ __forceinline__ void copyIndexedHostObjectFixed(
    const abi::ObjectEntry &object, const abi::ReplicaEntry &replica,
    std::uint32_t objectBlock = 0, std::uint32_t blocksPerObject = 1) {
  static_assert(ElementBytes % (ThreadsPerRow * sizeof(uint4)) == 0);
  constexpr std::uint32_t Segments =
      ElementBytes / (ThreadsPerRow * sizeof(uint4));
  const std::uint32_t lane = threadIdx.x % ThreadsPerRow;
  const std::uint32_t worker = threadIdx.x / ThreadsPerRow;
  const std::uint32_t workersPerBlock = blockDim.x / ThreadsPerRow;
  const auto *sourceIndices =
      reinterpret_cast<const std::uint32_t *>(replica.dmaPageListAddress);
  const auto *destinationIndices =
      reinterpret_cast<const std::uint32_t *>(object.stagingTensorMapAddress);
  const std::uint32_t sourceStride =
      abi::sourceTransferStride(replica.transferShape);
  const std::uint32_t destinationStride =
      abi::destinationTransferStride(replica.transferShape);
  const auto *source =
      reinterpret_cast<const std::byte *>(replica.sourceAddress);
  auto *destination = reinterpret_cast<std::byte *>(object.stagingAddress);

  const std::uint32_t firstElement = objectBlock * workersPerBlock + worker;
  const std::uint32_t elementStride = workersPerBlock * blocksPerObject;
  for (std::uint32_t element = firstElement; element < replica.dmaPageCount;
       element += elementStride) {
    const std::uint32_t sourceIndex = sourceIndices[element];
    const std::uint32_t destinationIndex = destinationIndices[element];
    uint4 values[Segments];
#pragma unroll
    for (std::uint32_t segment = 0; segment < Segments; ++segment) {
      const std::uint32_t vector = segment * ThreadsPerRow + lane;
      values[segment] = loadNoAllocate(reinterpret_cast<const uint4 *>(
          source + static_cast<std::uint64_t>(sourceIndex) * sourceStride +
          static_cast<std::uint64_t>(vector) * sizeof(uint4)));
    }
#pragma unroll
    for (std::uint32_t segment = 0; segment < Segments; ++segment) {
      const std::uint32_t vector = segment * ThreadsPerRow + lane;
      storeNoAllocate(
          reinterpret_cast<uint4 *>(
              destination +
              static_cast<std::uint64_t>(destinationIndex) * destinationStride +
              static_cast<std::uint64_t>(vector) * sizeof(uint4)),
          values[segment]);
    }
  }
}

template <std::uint32_t ElementBytes, std::uint32_t ThreadsPerRow>
__device__ __forceinline__ void copyIndexedHostPair(
    const abi::ObjectEntry &keyObject, const abi::ReplicaEntry &keyReplica,
    const abi::ObjectEntry &valueObject, const abi::ReplicaEntry &valueReplica,
    std::uint32_t pairBlock, std::uint32_t blocksPerPair) {
  static_assert(ElementBytes % (ThreadsPerRow * sizeof(uint4)) == 0);
  constexpr std::uint32_t Segments =
      ElementBytes / (ThreadsPerRow * sizeof(uint4));
  const std::uint32_t lane = threadIdx.x % ThreadsPerRow;
  const std::uint32_t worker = threadIdx.x / ThreadsPerRow;
  const std::uint32_t workersPerBlock = blockDim.x / ThreadsPerRow;
  const std::uint32_t firstElement = pairBlock * workersPerBlock + worker;
  const std::uint32_t elementStride = workersPerBlock * blocksPerPair;
  const auto *sourceIndices =
      reinterpret_cast<const std::uint32_t *>(keyReplica.dmaPageListAddress);
  const auto *destinationIndices = reinterpret_cast<const std::uint32_t *>(
      keyObject.stagingTensorMapAddress);
  const std::uint32_t keySourceStride =
      abi::sourceTransferStride(keyReplica.transferShape);
  const std::uint32_t keyDestinationStride =
      abi::destinationTransferStride(keyReplica.transferShape);
  const std::uint32_t valueSourceStride =
      abi::sourceTransferStride(valueReplica.transferShape);
  const std::uint32_t valueDestinationStride =
      abi::destinationTransferStride(valueReplica.transferShape);
  const auto *keySource =
      reinterpret_cast<const std::byte *>(keyReplica.sourceAddress);
  auto *keyDestination =
      reinterpret_cast<std::byte *>(keyObject.stagingAddress);
  const auto *valueSource =
      reinterpret_cast<const std::byte *>(valueReplica.sourceAddress);
  auto *valueDestination =
      reinterpret_cast<std::byte *>(valueObject.stagingAddress);

  for (std::uint32_t element = firstElement; element < keyReplica.dmaPageCount;
       element += elementStride) {
    // Redundant index loads within a row subgroup hit in L1 and avoid a warp
    // shuffle on the PCIe copy's critical instruction stream.
    const std::uint32_t sourceIndex = sourceIndices[element];
    const std::uint32_t destinationIndex = destinationIndices[element];
    uint4 values[Segments];
#pragma unroll
    for (std::uint32_t segment = 0; segment < Segments; ++segment) {
      const std::uint32_t vector = segment * ThreadsPerRow + lane;
      values[segment] = loadNoAllocate(reinterpret_cast<const uint4 *>(
          keySource +
          static_cast<std::uint64_t>(sourceIndex) * keySourceStride +
          static_cast<std::uint64_t>(vector) * sizeof(uint4)));
    }
#pragma unroll
    for (std::uint32_t segment = 0; segment < Segments; ++segment) {
      const std::uint32_t vector = segment * ThreadsPerRow + lane;
      storeNoAllocate(reinterpret_cast<uint4 *>(
                          keyDestination +
                          static_cast<std::uint64_t>(destinationIndex) *
                              keyDestinationStride +
                          static_cast<std::uint64_t>(vector) * sizeof(uint4)),
                      values[segment]);
    }
#pragma unroll
    for (std::uint32_t segment = 0; segment < Segments; ++segment) {
      const std::uint32_t vector = segment * ThreadsPerRow + lane;
      values[segment] = loadNoAllocate(reinterpret_cast<const uint4 *>(
          valueSource +
          static_cast<std::uint64_t>(sourceIndex) * valueSourceStride +
          static_cast<std::uint64_t>(vector) * sizeof(uint4)));
    }
#pragma unroll
    for (std::uint32_t segment = 0; segment < Segments; ++segment) {
      const std::uint32_t vector = segment * ThreadsPerRow + lane;
      storeNoAllocate(reinterpret_cast<uint4 *>(
                          valueDestination +
                          static_cast<std::uint64_t>(destinationIndex) *
                              valueDestinationStride +
                          static_cast<std::uint64_t>(vector) * sizeof(uint4)),
                      values[segment]);
    }
  }
}

template <std::uint32_t ElementBytes, std::uint32_t ThreadsPerRow>
__device__ __forceinline__ void copyIndexedHostObjectPairFixed(
    const abi::ObjectEntry &firstObject, const abi::ReplicaEntry &firstReplica,
    const abi::ObjectEntry &secondObject,
    const abi::ReplicaEntry &secondReplica, std::uint32_t objectBlock,
    std::uint32_t blocksPerObject) {
  static_assert(ElementBytes % (ThreadsPerRow * sizeof(uint4)) == 0);
  constexpr std::uint32_t Segments =
      ElementBytes / (ThreadsPerRow * sizeof(uint4));
  const std::uint32_t lane = threadIdx.x % ThreadsPerRow;
  const std::uint32_t worker = threadIdx.x / ThreadsPerRow;
  const std::uint32_t workersPerBlock = blockDim.x / ThreadsPerRow;
  const auto *sourceIndices =
      reinterpret_cast<const std::uint32_t *>(firstReplica.dmaPageListAddress);
  const auto *destinationIndices = reinterpret_cast<const std::uint32_t *>(
      firstObject.stagingTensorMapAddress);
  const std::uint32_t sourceStride =
      abi::sourceTransferStride(firstReplica.transferShape);
  const std::uint32_t destinationStride =
      abi::destinationTransferStride(firstReplica.transferShape);
  const auto *firstSource =
      reinterpret_cast<const std::byte *>(firstReplica.sourceAddress);
  const auto *secondSource =
      reinterpret_cast<const std::byte *>(secondReplica.sourceAddress);
  auto *firstDestination =
      reinterpret_cast<std::byte *>(firstObject.stagingAddress);
  auto *secondDestination =
      reinterpret_cast<std::byte *>(secondObject.stagingAddress);

  const std::uint32_t firstElement = objectBlock * workersPerBlock + worker;
  const std::uint32_t elementStride = workersPerBlock * blocksPerObject;
  for (std::uint32_t element = firstElement;
       element < firstReplica.dmaPageCount; element += elementStride) {
    const std::uint32_t sourceIndex = sourceIndices[element];
    const std::uint32_t destinationIndex = destinationIndices[element];
    uint4 values[Segments];
#pragma unroll
    for (std::uint32_t segment = 0; segment < Segments; ++segment) {
      const std::uint32_t vector = segment * ThreadsPerRow + lane;
      values[segment] = loadNoAllocate(reinterpret_cast<const uint4 *>(
          firstSource + static_cast<std::uint64_t>(sourceIndex) * sourceStride +
          static_cast<std::uint64_t>(vector) * sizeof(uint4)));
    }
#pragma unroll
    for (std::uint32_t segment = 0; segment < Segments; ++segment) {
      const std::uint32_t vector = segment * ThreadsPerRow + lane;
      storeNoAllocate(
          reinterpret_cast<uint4 *>(
              firstDestination +
              static_cast<std::uint64_t>(destinationIndex) * destinationStride +
              static_cast<std::uint64_t>(vector) * sizeof(uint4)),
          values[segment]);
    }
#pragma unroll
    for (std::uint32_t segment = 0; segment < Segments; ++segment) {
      const std::uint32_t vector = segment * ThreadsPerRow + lane;
      values[segment] = loadNoAllocate(reinterpret_cast<const uint4 *>(
          secondSource +
          static_cast<std::uint64_t>(sourceIndex) * sourceStride +
          static_cast<std::uint64_t>(vector) * sizeof(uint4)));
    }
#pragma unroll
    for (std::uint32_t segment = 0; segment < Segments; ++segment) {
      const std::uint32_t vector = segment * ThreadsPerRow + lane;
      storeNoAllocate(
          reinterpret_cast<uint4 *>(
              secondDestination +
              static_cast<std::uint64_t>(destinationIndex) * destinationStride +
              static_cast<std::uint64_t>(vector) * sizeof(uint4)),
          values[segment]);
    }
  }
}

template <std::uint32_t BlocksPerPair>
__device__ __forceinline__ void
preloadIndexedHostPair(abi::RuntimeView *runtime, std::uint32_t firstObject,
                       std::uint32_t pairCount, std::uint32_t relativePair,
                       std::uint32_t pairBlock, std::uint32_t *invalidIndex) {
  const std::uint64_t keySlot64 =
      static_cast<std::uint64_t>(firstObject) + 2ULL * relativePair;
  if (runtime == nullptr || relativePair >= pairCount ||
      keySlot64 >= runtime->objectCapacity ||
      keySlot64 + 1 >= runtime->objectCapacity) {
    return;
  }
  const std::uint32_t keySlot = static_cast<std::uint32_t>(keySlot64);
  const std::uint32_t valueSlot = keySlot + 1;
  abi::ObjectEntry &keyObject = runtime->objects[keySlot];
  abi::ObjectEntry &valueObject = runtime->objects[valueSlot];
  const abi::ReplicaEntry *keyReplica = replica(runtime, keyObject, 0);
  const abi::ReplicaEntry *valueReplica = replica(runtime, valueObject, 0);
  const std::uint32_t elementBytes =
      keyReplica == nullptr || keyReplica->dmaPageCount == 0
          ? 0
          : static_cast<std::uint32_t>(keyObject.bytes /
                                       keyReplica->dmaPageCount);
  const bool supportedElement = elementBytes == 128 || elementBytes == 256 ||
                                elementBytes == 512 || elementBytes == 1024 ||
                                elementBytes == 2048;
  const bool valid =
      keyReplica != nullptr && valueReplica != nullptr && supportedElement &&
      (keyReplica->flags & abi::ReplicaIndexed) != 0 &&
      (valueReplica->flags & abi::ReplicaIndexed) != 0 &&
      keyReplica->sourceKind ==
          static_cast<std::uint32_t>(abi::SourceKind::HostStaged) &&
      valueReplica->sourceKind ==
          static_cast<std::uint32_t>(abi::SourceKind::HostStaged) &&
      keyReplica->dmaPageCount == valueReplica->dmaPageCount &&
      keyObject.bytes ==
          static_cast<std::uint64_t>(keyReplica->dmaPageCount) * elementBytes &&
      valueObject.bytes ==
          static_cast<std::uint64_t>(valueReplica->dmaPageCount) *
              elementBytes &&
      keyReplica->dmaPageListAddress != 0 &&
      keyReplica->dmaPageListAddress == valueReplica->dmaPageListAddress &&
      keyObject.stagingTensorMapAddress != 0 &&
      keyObject.stagingTensorMapAddress ==
          valueObject.stagingTensorMapAddress &&
      keyReplica->sourceAddress != 0 && valueReplica->sourceAddress != 0 &&
      keyObject.stagingAddress != 0 && valueObject.stagingAddress != 0 &&
      abi::sourceTransferStride(keyReplica->transferShape) >= elementBytes &&
      abi::destinationTransferStride(keyReplica->transferShape) >=
          elementBytes &&
      abi::sourceTransferStride(valueReplica->transferShape) >= elementBytes &&
      abi::destinationTransferStride(valueReplica->transferShape) >=
          elementBytes;

  if (threadIdx.x == 0) {
    *invalidIndex = 0;
  }
  __syncthreads();
  if (valid) {
    validateIndexedTransferIndices(keyObject, *keyReplica, invalidIndex);
    validateIndexedTransferIndices(valueObject, *valueReplica, invalidIndex);
  }
  __syncthreads();
  const bool bounded = valid && *invalidIndex == 0;
  if (threadIdx.x == 0 && pairBlock == 0) {
    keyObject.selectedReplica = 0;
    valueObject.selectedReplica = 0;
    atomicExch(&keyObject.state,
               static_cast<std::uint32_t>(bounded ? abi::ObjectState::Issued
                                                  : abi::ObjectState::Failed));
    atomicExch(&valueObject.state,
               static_cast<std::uint32_t>(bounded ? abi::ObjectState::Issued
                                                  : abi::ObjectState::Failed));
    if (!bounded) {
      recordFailure(runtime);
    }
  }
  if (!bounded) {
    return;
  }
  switch (elementBytes) {
  case 128:
    copyIndexedHostPair<128, 8>(keyObject, *keyReplica, valueObject,
                                *valueReplica, pairBlock, BlocksPerPair);
    break;
  case 256:
    copyIndexedHostPair<256, 8>(keyObject, *keyReplica, valueObject,
                                *valueReplica, pairBlock, BlocksPerPair);
    break;
  case 512:
    copyIndexedHostPair<512, 32>(keyObject, *keyReplica, valueObject,
                                 *valueReplica, pairBlock, BlocksPerPair);
    break;
  case 1024:
    copyIndexedHostPair<1024, 16>(keyObject, *keyReplica, valueObject,
                                  *valueReplica, pairBlock, BlocksPerPair);
    break;
  case 2048:
    copyIndexedHostPair<2048, 32>(keyObject, *keyReplica, valueObject,
                                  *valueReplica, pairBlock, BlocksPerPair);
    break;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const std::uint64_t completed =
        atomicAdd(reinterpret_cast<unsigned long long *>(&keyObject.issueCount),
                  1ULL) +
        1ULL;
    if (completed == BlocksPerPair) {
      __threadfence_system();
      (void)atomicAdd(
          reinterpret_cast<unsigned long long *>(&keyObject.issueCount),
          0ULL - static_cast<unsigned long long>(BlocksPerPair));
      atomicExch(&keyObject.state,
                 static_cast<std::uint32_t>(abi::ObjectState::Ready));
      atomicExch(&valueObject.state,
                 static_cast<std::uint32_t>(abi::ObjectState::Ready));
    } else if (completed > BlocksPerPair) {
      atomicExch(&keyObject.state,
                 static_cast<std::uint32_t>(abi::ObjectState::Failed));
      atomicExch(&valueObject.state,
                 static_cast<std::uint32_t>(abi::ObjectState::Failed));
      recordFailure(runtime);
    }
  }
}

} // namespace nta::device

// KV-aware lookahead mover. Two CTAs cooperatively copy one adjacent K/V
// object pair while preserving the generic object-directory representation.
// Pair completion is published only by the second finishing CTA; unlike the
// earlier event-only implementation, ObjectState::Ready is itself now a sound
// device-side acquisition fence.
extern "C" __global__ __launch_bounds__(
    1024, 1) void nta_preload_indexed_host_pairs(nta::abi::RuntimeView *runtime,
                                                 std::uint32_t firstObject,
                                                 std::uint32_t pairCount) {
  constexpr std::uint32_t BlocksPerPair = 2;
  const std::uint32_t relativePair = blockIdx.x / BlocksPerPair;
  const std::uint32_t pairBlock = blockIdx.x % BlocksPerPair;
  __shared__ std::uint32_t invalidIndex;
  nta::device::preloadIndexedHostPair<BlocksPerPair>(
      runtime, firstObject, pairCount, relativePair, pairBlock, &invalidIndex);
}

// One finite worker grid consumes pair-block tasks in directory order. The
// directory is laid out (EDF layer, exact wave, K/V), so bounding the worker
// count preserves work-conserving earliest-deadline issue while retaining the
// same eight-CTA transfer parallelism as the high-throughput whole-layer path.
extern "C" __global__
__launch_bounds__(1024, 1) void nta_preload_indexed_host_pairs_ordered(
    nta::abi::RuntimeView *runtime, std::uint32_t firstObject,
    std::uint32_t pairCount, std::uint32_t *taskHead) {
  constexpr std::uint32_t BlocksPerPair = 2;
  __shared__ std::uint32_t task;
  __shared__ std::uint32_t invalidIndex;
  const std::uint64_t taskCount64 =
      static_cast<std::uint64_t>(pairCount) * BlocksPerPair;
  if (runtime == nullptr || taskHead == nullptr || taskCount64 > UINT32_MAX) {
    return;
  }
  const std::uint32_t taskCount = static_cast<std::uint32_t>(taskCount64);
  while (true) {
    if (threadIdx.x == 0) {
      task = atomicAdd(taskHead, 1U);
    }
    __syncthreads();
    if (task >= taskCount) {
      return;
    }
    nta::device::preloadIndexedHostPair<BlocksPerPair>(
        runtime, firstObject, pairCount, task / BlocksPerPair,
        task % BlocksPerPair, &invalidIndex);
    __syncthreads();
  }
}

extern "C" __global__ void
nta_progress_host_staging(nta::abi::RuntimeView *runtime) {
  using namespace nta;
  if (runtime == nullptr || runtime->intentPool == nullptr) {
    return;
  }
  __shared__ std::uint32_t selectedIntent;
  __shared__ std::uint32_t selectedFromQueue;
  __shared__ std::uint32_t terminalDispatch;
  __shared__ std::uint64_t chargedBytes;
  __shared__ std::uint64_t backendBytes;
  if (threadIdx.x == 0) {
    selectedFromQueue = device::intentQueueAvailable(runtime) ? 1U : 0U;
    terminalDispatch = 0;
    chargedBytes = 0;
    backendBytes = 0;
    if (selectedFromQueue != 0U) {
      const device::AdmittedIntent dispatch =
          device::claimAdmissibleIntent(runtime, abi::SourceKind::HostStaged);
      selectedIntent = dispatch.slotIndex;
      terminalDispatch = dispatch.terminal ? 1U : 0U;
      chargedBytes = dispatch.requestBytes;
      backendBytes = dispatch.backendBytes;
    } else {
      selectedIntent = static_cast<std::uint32_t>(blockIdx.x);
    }
  }
  __syncthreads();
  if (selectedIntent == abi::InvalidIndex ||
      selectedIntent >= runtime->intentPool->capacity ||
      selectedIntent >= runtime->intentCapacity) {
    return;
  }
  abi::IntentSlot &intentSlot = runtime->intents[selectedIntent];

  abi::AcquireIntent &intent = intentSlot.intent;
  const std::uint32_t ownedIntent = selectedFromQueue != 0U ? 2U : 1U;
  if (atomicAdd(&intent.valid, 0U) != ownedIntent ||
      intentSlot.sourceKind !=
          static_cast<std::uint32_t>(abi::SourceKind::HostStaged)) {
    return;
  }
  if (intentSlot.epoch != device::currentEpoch(runtime)) {
    if (threadIdx.x == 0) {
      const bool owned =
          selectedFromQueue != 0U || device::claimIntent(intentSlot);
      if (owned) {
        device::releaseIntentCredits(runtime, intent,
                                     abi::SourceKind::HostStaged, chargedBytes,
                                     backendBytes);
        device::consumeIntent(runtime, intentSlot);
      }
    }
    return;
  }
  if (intent.objectSlot >= runtime->objectCapacity) {
    if (threadIdx.x == 0) {
      const bool owned =
          selectedFromQueue != 0U || device::claimIntent(intentSlot);
      if (owned) {
        device::releaseIntentCredits(runtime, intent,
                                     abi::SourceKind::HostStaged, chargedBytes,
                                     backendBytes);
        device::failBoundWorkTicket(runtime, intent.workTicket,
                                    intent.requestSlot, intent.generation);
        device::consumeIntent(runtime, intentSlot);
      }
    }
    return;
  }
  abi::ObjectEntry &object = runtime->objects[intent.objectSlot];
  const abi::ReplicaEntry *replica =
      device::replica(runtime, object, object.selectedReplica);
  const bool objectCurrent = object.objectId == intent.objectId &&
                             object.version == intent.objectVersion;
  const bool indexed =
      replica != nullptr && (replica->flags & abi::ReplicaIndexed) != 0;
  const std::uint32_t sourceStride =
      replica == nullptr ? 0
                         : abi::sourceTransferStride(replica->transferShape);
  const std::uint32_t destinationStride =
      replica == nullptr
          ? 0
          : abi::destinationTransferStride(replica->transferShape);
  const std::uint32_t sourceLimit =
      replica == nullptr
          ? 0
          : abi::sourceTransferIndexLimit(replica->tensorMapAddress);
  const std::uint32_t destinationLimit =
      replica == nullptr
          ? 0
          : abi::destinationTransferIndexLimit(replica->tensorMapAddress);
  const bool indexedShapeValid =
      !indexed ||
      (replica->dmaPageListAddress != 0 &&
       object.stagingTensorMapAddress != 0 && replica->dmaPageCount != 0 &&
       sourceLimit != 0 && destinationLimit != 0 &&
       intent.bytes % replica->dmaPageCount == 0 &&
       sourceStride >= intent.bytes / replica->dmaPageCount &&
       destinationStride >= intent.bytes / replica->dmaPageCount);
  const bool transferValid =
      objectCurrent && replica != nullptr && replica->sourceAddress != 0 &&
      object.stagingAddress != 0 &&
      replica->sourceKind ==
          static_cast<std::uint32_t>(abi::SourceKind::HostStaged) &&
      intent.offset == 0 && intent.bytes == object.bytes &&
      intent.offset <= object.bytes &&
      intent.bytes <= object.bytes - intent.offset && indexedShapeValid;
  if (terminalDispatch != 0U) {
    if (threadIdx.x == 0) {
      if (objectCurrent) {
        atomicExch(&object.state,
                   static_cast<std::uint32_t>(abi::ObjectState::Failed));
        device::publishObjectTransition(runtime, intent.objectSlot,
                                        abi::ObjectState::Failed);
      }
      device::failBoundWorkTicket(runtime, intent.workTicket,
                                  intent.requestSlot, intent.generation);
      device::consumeIntent(runtime, intentSlot);
    }
    return;
  }
  __shared__ std::uint32_t invalidIndex;
  if (threadIdx.x == 0) {
    invalidIndex = 0;
  }
  __syncthreads();
  if (transferValid && indexed &&
      (replica->flags & abi::ReplicaIndicesValidated) == 0) {
    device::validateIndexedTransferIndices(object, *replica, &invalidIndex);
  }
  __syncthreads();
  if (!transferValid || invalidIndex != 0) {
    if (threadIdx.x == 0) {
      const bool owned =
          selectedFromQueue != 0U || device::claimIntent(intentSlot);
      if (owned) {
        device::releaseIntentCredits(runtime, intent,
                                     abi::SourceKind::HostStaged, chargedBytes,
                                     backendBytes);
        if (objectCurrent) {
          atomicExch(&object.state,
                     static_cast<std::uint32_t>(abi::ObjectState::Failed));
          device::publishObjectTransition(runtime, intent.objectSlot,
                                          abi::ObjectState::Failed);
        }
        device::failBoundWorkTicket(runtime, intent.workTicket,
                                    intent.requestSlot, intent.generation);
        device::consumeIntent(runtime, intentSlot);
      }
    }
    return;
  }

  __shared__ std::uint32_t admitted;
  if (threadIdx.x == 0) {
    if (selectedFromQueue != 0U) {
      admitted = 1U;
    } else {
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
  }
  __syncthreads();
  if (admitted == 0) {
    return;
  }

  auto *source = reinterpret_cast<const std::byte *>(replica->sourceAddress);
  auto *destination = reinterpret_cast<std::byte *>(object.stagingAddress);

  if (indexed) {
    const std::uint32_t elementBytes = intent.bytes / replica->dmaPageCount;
    switch (elementBytes) {
    case 128:
      device::copyIndexedHostObjectFixed<128, 8>(object, *replica);
      break;
    case 256:
      device::copyIndexedHostObjectFixed<256, 8>(object, *replica);
      break;
    case 512:
      device::copyIndexedHostObjectFixed<512, 8>(object, *replica);
      break;
    case 1024:
      device::copyIndexedHostObjectFixed<1024, 16>(object, *replica);
      break;
    case 2048:
      device::copyIndexedHostObjectFixed<2048, 32>(object, *replica);
      break;
    default:
      device::copyIndexedHostObject(object, *replica);
      break;
    }
  } else {
    const bool vectorAligned =
        ((reinterpret_cast<std::uintptr_t>(source) |
          reinterpret_cast<std::uintptr_t>(destination)) &
         (alignof(uint4) - 1U)) == 0;
    if (vectorAligned) {
      const std::uint32_t vectorBytes = intent.bytes & ~15U;
      for (std::uint32_t byte = threadIdx.x * sizeof(uint4); byte < vectorBytes;
           byte += blockDim.x * sizeof(uint4)) {
        device::storeNoAllocate(
            reinterpret_cast<uint4 *>(destination + byte),
            device::loadNoAllocate(
                reinterpret_cast<const uint4 *>(source + byte)));
      }
      for (std::uint32_t byte = vectorBytes + threadIdx.x; byte < intent.bytes;
           byte += blockDim.x) {
        destination[byte] = source[byte];
      }
    } else {
      for (std::uint32_t byte = threadIdx.x; byte < intent.bytes;
           byte += blockDim.x) {
        destination[byte] = source[byte];
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    __threadfence_system();
    atomicExch(&object.state,
               static_cast<std::uint32_t>(abi::ObjectState::Ready));
    device::publishObjectTransition(runtime, intent.objectSlot,
                                    abi::ObjectState::Ready);
    device::consumeIntent(runtime, intentSlot);
    device::releaseIntentCredits(runtime, intent, abi::SourceKind::HostStaged,
                                 chargedBytes, backendBytes);
  }
}

// Scheduler-selected indexed host progress uses three finite kernels. The
// claim phase preserves the generic intent/credit protocol. The copy phase
// gives each adjacent object pair a small block quota, matching the K/V access
// shape while bounding PCIe read pressure. Stream ordering keeps publication
// after every copy CTA has retired.
extern "C" __global__ void
nta_validate_indexed_host_range(nta::abi::RuntimeView *runtime,
                                std::uint32_t firstObject,
                                std::uint32_t objectCount) {
  using namespace nta;
  const std::uint32_t relative = blockIdx.x;
  const std::uint64_t objectSlot64 =
      static_cast<std::uint64_t>(firstObject) + relative;
  if (runtime == nullptr || relative >= objectCount ||
      objectSlot64 >= runtime->objectCapacity) {
    return;
  }
  abi::ObjectEntry &object = runtime->objects[objectSlot64];
  abi::ReplicaEntry *replica =
      object.replicaCount == 1 && object.replicaStart < runtime->replicaCapacity
          ? &runtime->replicas[object.replicaStart]
          : nullptr;
  const std::uint32_t sourceStride =
      replica == nullptr ? 0
                         : abi::sourceTransferStride(replica->transferShape);
  const std::uint32_t destinationStride =
      replica == nullptr
          ? 0
          : abi::destinationTransferStride(replica->transferShape);
  const bool valid =
      replica != nullptr &&
      replica->sourceKind ==
          static_cast<std::uint32_t>(abi::SourceKind::HostStaged) &&
      (replica->flags & abi::ReplicaIndexed) != 0 &&
      replica->sourceAddress != 0 && object.stagingAddress != 0 &&
      replica->dmaPageListAddress != 0 && object.stagingTensorMapAddress != 0 &&
      replica->dmaPageCount != 0 &&
      abi::sourceTransferIndexLimit(replica->tensorMapAddress) != 0 &&
      abi::destinationTransferIndexLimit(replica->tensorMapAddress) != 0 &&
      object.bytes != 0 && object.bytes % replica->dmaPageCount == 0 &&
      sourceStride >= object.bytes / replica->dmaPageCount &&
      destinationStride >= object.bytes / replica->dmaPageCount;
  __shared__ std::uint32_t invalidIndex;
  if (threadIdx.x == 0) {
    invalidIndex = 0;
  }
  __syncthreads();
  if (valid) {
    device::validateIndexedTransferIndices(object, *replica, &invalidIndex);
  }
  __syncthreads();
  if (threadIdx.x != 0) {
    return;
  }
  if (!valid || invalidIndex != 0) {
    atomicExch(&object.state,
               static_cast<std::uint32_t>(abi::ObjectState::Failed));
    device::recordFailure(runtime);
    return;
  }
  atomicOr(&replica->flags, abi::ReplicaIndicesValidated);
}

extern "C" __global__ void
nta_claim_indexed_host_range(nta::abi::RuntimeView *runtime,
                             std::uint32_t firstObject,
                             std::uint32_t objectCount) {
  using namespace nta;
  const std::uint32_t relative = blockIdx.x;
  const std::uint64_t objectSlot64 =
      static_cast<std::uint64_t>(firstObject) + relative;
  if (runtime == nullptr || runtime->intentPool == nullptr ||
      relative >= objectCount || objectSlot64 >= runtime->objectCapacity ||
      objectSlot64 >= runtime->intentCapacity ||
      objectSlot64 >= runtime->intentPool->capacity) {
    return;
  }
  const std::uint32_t objectSlot = static_cast<std::uint32_t>(objectSlot64);
  abi::IntentSlot &intentSlot = runtime->intents[objectSlot];
  abi::AcquireIntent &intent = intentSlot.intent;
  abi::ObjectEntry &object = runtime->objects[objectSlot];
  const abi::ReplicaEntry *replica =
      device::replica(runtime, object, object.selectedReplica);
  const std::uint32_t sourceStride =
      replica == nullptr ? 0
                         : abi::sourceTransferStride(replica->transferShape);
  const std::uint32_t destinationStride =
      replica == nullptr
          ? 0
          : abi::destinationTransferStride(replica->transferShape);
  const std::uint32_t sourceLimit =
      replica == nullptr
          ? 0
          : abi::sourceTransferIndexLimit(replica->tensorMapAddress);
  const std::uint32_t destinationLimit =
      replica == nullptr
          ? 0
          : abi::destinationTransferIndexLimit(replica->tensorMapAddress);
  const bool valid =
      atomicAdd(&intent.valid, 0U) == 1U &&
      intentSlot.epoch == device::currentEpoch(runtime) &&
      intentSlot.sourceKind ==
          static_cast<std::uint32_t>(abi::SourceKind::HostStaged) &&
      atomicAdd(&object.state, 0U) ==
          static_cast<std::uint32_t>(abi::ObjectState::Queued) &&
      intent.objectSlot == objectSlot && intent.objectId == object.objectId &&
      intent.objectVersion == object.version && intent.offset == 0 &&
      intent.bytes == object.bytes && replica != nullptr &&
      replica->sourceKind ==
          static_cast<std::uint32_t>(abi::SourceKind::HostStaged) &&
      (replica->flags & abi::ReplicaIndexed) != 0 &&
      replica->sourceAddress != 0 && object.stagingAddress != 0 &&
      replica->dmaPageListAddress != 0 && object.stagingTensorMapAddress != 0 &&
      replica->dmaPageCount != 0 && sourceLimit != 0 && destinationLimit != 0 &&
      object.bytes != 0 && object.bytes % replica->dmaPageCount == 0 &&
      sourceStride >= object.bytes / replica->dmaPageCount &&
      destinationStride >= object.bytes / replica->dmaPageCount;

  __shared__ std::uint32_t invalidIndex;
  if (threadIdx.x == 0) {
    invalidIndex = 0;
  }
  __syncthreads();
  if (valid && (replica->flags & abi::ReplicaIndicesValidated) == 0) {
    device::validateIndexedTransferIndices(object, *replica, &invalidIndex);
  }
  __syncthreads();
  if (threadIdx.x != 0) {
    return;
  }
  if (!valid || invalidIndex != 0) {
    if (device::claimIntent(intentSlot)) {
      atomicExch(&object.state,
                 static_cast<std::uint32_t>(abi::ObjectState::Failed));
      device::publishObjectTransition(runtime, objectSlot,
                                      abi::ObjectState::Failed);
      device::failBoundWorkTicket(runtime, intent.workTicket,
                                  intent.requestSlot, intent.generation);
      device::consumeIntent(runtime, intentSlot);
    }
    return;
  }

  std::uint64_t requestBytes = 0;
  std::uint64_t backendBytes = 0;
  bool accepted = device::reserveIntentCredits(
      runtime, intent, abi::SourceKind::HostStaged, requestBytes, backendBytes);
  if (accepted && !device::claimIntent(intentSlot)) {
    device::releaseIntentCredits(runtime, intent, abi::SourceKind::HostStaged,
                                 requestBytes, backendBytes);
    accepted = false;
  }
  if (accepted) {
    device::recordIntentCredits(intentSlot, requestBytes, backendBytes);
    __threadfence();
    atomicExch(&object.state,
               static_cast<std::uint32_t>(abi::ObjectState::Issued));
  }
}

extern "C" __global__ void nta_copy_indexed_host_range(
    nta::abi::RuntimeView *runtime, std::uint32_t firstObject,
    std::uint32_t objectCount, std::uint32_t blocksPerObject) {
  using namespace nta;
  if (runtime == nullptr || blocksPerObject == 0) {
    return;
  }
  constexpr std::uint32_t ObjectsPerGroup = 2;
  const std::uint32_t relativeGroup = blockIdx.x / blocksPerObject;
  const std::uint32_t objectBlock = blockIdx.x % blocksPerObject;
  const std::uint32_t firstRelative = relativeGroup * ObjectsPerGroup;
  const std::uint64_t firstSlot64 =
      static_cast<std::uint64_t>(firstObject) + firstRelative;
  const std::uint64_t secondSlot64 = firstSlot64 + 1U;
  if (firstRelative + 1U < objectCount &&
      secondSlot64 < runtime->objectCapacity) {
    abi::ObjectEntry &first = runtime->objects[firstSlot64];
    abi::ObjectEntry &second = runtime->objects[secondSlot64];
    const abi::ReplicaEntry *firstReplica =
        device::replica(runtime, first, first.selectedReplica);
    const abi::ReplicaEntry *secondReplica =
        device::replica(runtime, second, second.selectedReplica);
    const bool paired =
        atomicAdd(&first.state, 0U) ==
            static_cast<std::uint32_t>(abi::ObjectState::Issued) &&
        atomicAdd(&second.state, 0U) ==
            static_cast<std::uint32_t>(abi::ObjectState::Issued) &&
        firstReplica != nullptr && secondReplica != nullptr &&
        firstReplica->dmaPageCount != 0 &&
        firstReplica->dmaPageCount == secondReplica->dmaPageCount &&
        firstReplica->dmaPageListAddress == secondReplica->dmaPageListAddress &&
        first.stagingTensorMapAddress == second.stagingTensorMapAddress &&
        firstReplica->transferShape == secondReplica->transferShape &&
        first.bytes == second.bytes;
    if (paired) {
      const std::uint32_t elementBytes =
          static_cast<std::uint32_t>(first.bytes / firstReplica->dmaPageCount);
      switch (elementBytes) {
      case 128:
        device::copyIndexedHostObjectPairFixed<128, 8>(
            first, *firstReplica, second, *secondReplica, objectBlock,
            blocksPerObject);
        return;
      case 256:
        device::copyIndexedHostObjectPairFixed<256, 8>(
            first, *firstReplica, second, *secondReplica, objectBlock,
            blocksPerObject);
        return;
      case 512:
        device::copyIndexedHostObjectPairFixed<512, 32>(
            first, *firstReplica, second, *secondReplica, objectBlock,
            blocksPerObject);
        return;
      case 1024:
        device::copyIndexedHostObjectPairFixed<1024, 16>(
            first, *firstReplica, second, *secondReplica, objectBlock,
            blocksPerObject);
        return;
      case 2048:
        device::copyIndexedHostObjectPairFixed<2048, 32>(
            first, *firstReplica, second, *secondReplica, objectBlock,
            blocksPerObject);
        return;
      default:
        break;
      }
    }
  }
  for (std::uint32_t groupObject = 0; groupObject < ObjectsPerGroup;
       ++groupObject) {
    const std::uint32_t relative = firstRelative + groupObject;
    const std::uint64_t objectSlot64 =
        static_cast<std::uint64_t>(firstObject) + relative;
    if (relative >= objectCount || objectSlot64 >= runtime->objectCapacity) {
      continue;
    }
    abi::ObjectEntry &object = runtime->objects[objectSlot64];
    if (atomicAdd(&object.state, 0U) !=
        static_cast<std::uint32_t>(abi::ObjectState::Issued)) {
      continue;
    }
    const abi::ReplicaEntry *replica =
        device::replica(runtime, object, object.selectedReplica);
    if (replica == nullptr || replica->dmaPageCount == 0) {
      continue;
    }
    const std::uint32_t elementBytes =
        static_cast<std::uint32_t>(object.bytes / replica->dmaPageCount);
    switch (elementBytes) {
    case 128:
      device::copyIndexedHostObjectFixed<128, 8>(object, *replica, objectBlock,
                                                 blocksPerObject);
      break;
    case 256:
      device::copyIndexedHostObjectFixed<256, 8>(object, *replica, objectBlock,
                                                 blocksPerObject);
      break;
    case 512:
      device::copyIndexedHostObjectFixed<512, 8>(object, *replica, objectBlock,
                                                 blocksPerObject);
      break;
    case 1024:
      device::copyIndexedHostObjectFixed<1024, 16>(
          object, *replica, objectBlock, blocksPerObject);
      break;
    case 2048:
      device::copyIndexedHostObjectFixed<2048, 32>(
          object, *replica, objectBlock, blocksPerObject);
      break;
    default:
      device::copyIndexedHostObject(object, *replica, objectBlock,
                                    blocksPerObject);
      break;
    }
  }
}

extern "C" __global__ void
nta_finalize_indexed_host_range(nta::abi::RuntimeView *runtime,
                                std::uint32_t firstObject,
                                std::uint32_t objectCount) {
  using namespace nta;
  const std::uint32_t relative = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint64_t objectSlot64 =
      static_cast<std::uint64_t>(firstObject) + relative;
  if (runtime == nullptr || relative >= objectCount ||
      objectSlot64 >= runtime->objectCapacity ||
      objectSlot64 >= runtime->intentCapacity) {
    return;
  }
  const std::uint32_t objectSlot = static_cast<std::uint32_t>(objectSlot64);
  abi::ObjectEntry &object = runtime->objects[objectSlot];
  abi::IntentSlot &intentSlot = runtime->intents[objectSlot];
  abi::AcquireIntent &intent = intentSlot.intent;
  if (atomicAdd(&object.state, 0U) !=
          static_cast<std::uint32_t>(abi::ObjectState::Issued) ||
      atomicAdd(&intent.valid, 0U) != 2U || intent.objectSlot != objectSlot ||
      intentSlot.epoch != device::currentEpoch(runtime)) {
    return;
  }
  __threadfence_system();
  atomicExch(&object.state,
             static_cast<std::uint32_t>(abi::ObjectState::Ready));
  // Private per-work groups publish directly. Shared-prefix/high-fanout groups
  // set changedOverflow and are handled by the following parallel full scan.
  (void)device::publishPrivateIndexedObject(runtime, objectSlot);
  device::releaseRecordedIntentCredits(runtime, intentSlot, intent,
                                       abi::SourceKind::HostStaged);
  device::consumeIntent(runtime, intentSlot);
}
#endif

#if NTA_DEVICE_PHASE_KERNELS
extern "C" __global__ void nta_publish_ready(nta::abi::RuntimeView *runtime,
                                             std::uint32_t pendingBudget) {
  using namespace nta;
  if (runtime == nullptr || runtime->pendingCount == nullptr ||
      runtime->pendingWorkTickets == nullptr || blockIdx.x != 0) {
    return;
  }
  __shared__ std::uint32_t workCount;
  __shared__ std::uint32_t changedMode;
  if (threadIdx.x == 0) {
    const bool supportsChanged = runtime->changedWorkTickets != nullptr &&
                                 runtime->changedQueued != nullptr &&
                                 runtime->changedCount != nullptr &&
                                 runtime->changedOverflow != nullptr;
    const bool overflow =
        supportsChanged && atomicExch(runtime->changedOverflow, 0U) != 0U;
    const std::uint32_t changed =
        supportsChanged ? atomicExch(runtime->changedCount, 0U) : 0U;
    changedMode = supportsChanged && !overflow ? 1U : 0U;
    workCount =
        changedMode != 0
            ? min(changed, runtime->workTicketCapacity)
            : min(min(atomicAdd(runtime->pendingCount, 0U), pendingBudget),
                  runtime->workTicketCapacity);
  }
  __syncthreads();

  for (std::uint32_t workIndex = threadIdx.x; workIndex < workCount;
       workIndex += blockDim.x) {
    const std::uint32_t workTicketIndex =
        changedMode != 0 ? runtime->changedWorkTickets[workIndex]
                         : runtime->pendingWorkTickets[workIndex];
    if (workTicketIndex >= runtime->workTicketCapacity) {
      continue;
    }
    if (runtime->changedQueued != nullptr) {
      atomicExch(&runtime->changedQueued[workTicketIndex], 0U);
    }
    abi::WorkTicket &workTicket = runtime->workTickets[workTicketIndex];
    if (atomicAdd(&workTicket.state, 0U) !=
        static_cast<std::uint32_t>(abi::WorkTicketState::Pending)) {
      continue;
    }
    if (workTicket.epoch != device::currentEpoch(runtime)) {
      if (atomicCAS(&workTicket.state,
                    static_cast<std::uint32_t>(abi::WorkTicketState::Pending),
                    static_cast<std::uint32_t>(abi::WorkTicketState::Failed)) ==
          static_cast<std::uint32_t>(abi::WorkTicketState::Pending)) {
        device::recordTerminalWork(runtime, workTicket,
                                   abi::WorkTicketState::Pending,
                                   abi::WorkTicketState::Failed);
        device::recordFailure(runtime);
      }
      continue;
    }
    if (!device::requestLive(runtime, workTicket.requestSlot,
                             workTicket.generation)) {
      if (atomicCAS(
              &workTicket.state,
              static_cast<std::uint32_t>(abi::WorkTicketState::Pending),
              static_cast<std::uint32_t>(abi::WorkTicketState::Cancelled)) ==
          static_cast<std::uint32_t>(abi::WorkTicketState::Pending)) {
        device::recordTerminalWork(runtime, workTicket,
                                   abi::WorkTicketState::Pending,
                                   abi::WorkTicketState::Cancelled);
      }
      continue;
    }
    std::uint32_t dependencyStart = 0;
    if (!device::dependencyRange(runtime, workTicketIndex,
                                 workTicket.dependencyCount, dependencyStart) ||
        dependencyStart != workTicket.dependencyStart) {
      if (atomicCAS(&workTicket.state,
                    static_cast<std::uint32_t>(abi::WorkTicketState::Pending),
                    static_cast<std::uint32_t>(abi::WorkTicketState::Failed)) ==
          static_cast<std::uint32_t>(abi::WorkTicketState::Pending)) {
        device::recordTerminalWork(runtime, workTicket,
                                   abi::WorkTicketState::Pending,
                                   abi::WorkTicketState::Failed);
        device::recordFailure(runtime);
      }
      continue;
    }

    bool ready = true;
    bool failed = false;
    for (std::uint32_t index = 0; index < workTicket.dependencyCount; ++index) {
      const abi::WorkDependency dependency =
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
      if (atomicCAS(&workTicket.state,
                    static_cast<std::uint32_t>(abi::WorkTicketState::Pending),
                    static_cast<std::uint32_t>(abi::WorkTicketState::Failed)) ==
          static_cast<std::uint32_t>(abi::WorkTicketState::Pending)) {
        device::recordTerminalWork(runtime, workTicket,
                                   abi::WorkTicketState::Pending,
                                   abi::WorkTicketState::Failed);
        device::recordFailure(runtime);
      }
      continue;
    }
    if (!ready) {
      continue;
    }
    // A full-scan publication is also the dependency-accounting fallback used
    // by high-fanout indexed range completion. The ticket is still Pending, so
    // this thread owns its dependency records until publishRunnableWork wins
    // the state transition below. Preserve the same exact bookkeeping as the
    // object-centric linked-list path before exposing the ticket as runnable.
    for (std::uint32_t index = 0; index < workTicket.dependencyCount; ++index) {
      atomicCAS(&runtime->dependencySatisfied[dependencyStart + index], 0U, 1U);
    }
    __threadfence();
    atomicExch(&runtime->remainingDependencies[workTicketIndex], 0U);
    (void)device::publishRunnableWork(runtime, workTicketIndex);
  }
}
#endif

#if NTA_DEVICE_PHASE_KERNELS
extern "C" __global__ void nta_rebind_indexed_host_pairs(
    nta::abi::RuntimeView *runtime, std::uint32_t firstObject,
    std::uint32_t pairCount, std::uint64_t keySource, std::uint64_t keyStaging,
    std::uint64_t valueSource, std::uint64_t valueStaging) {
  using namespace nta;
  const std::uint32_t relative = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint64_t objectCount = static_cast<std::uint64_t>(pairCount) * 2U;
  const std::uint64_t slot64 =
      static_cast<std::uint64_t>(firstObject) + relative;
  if (runtime == nullptr || relative >= objectCount ||
      slot64 >= runtime->objectCapacity) {
    return;
  }
  const std::uint32_t slot = static_cast<std::uint32_t>(slot64);
  abi::ObjectEntry &object = runtime->objects[slot];
  if (object.replicaCount != 1 ||
      object.replicaStart >= runtime->replicaCapacity) {
    device::recordFailure(runtime);
    return;
  }
  const bool key = (relative & 1U) == 0;
  abi::ReplicaEntry &replica = runtime->replicas[object.replicaStart];
  replica.sourceAddress = key ? keySource : valueSource;
  object.stagingAddress = key ? keyStaging : valueStaging;
  object.selectedReplica = abi::InvalidIndex;
  object.issueCount = 0;
  __threadfence();
  object.state = static_cast<std::uint32_t>(abi::ObjectState::New);
}

extern "C" __global__ void
nta_invalidate_cached_objects(nta::abi::RuntimeView *runtime,
                              std::uint32_t firstObject,
                              std::uint32_t objectCount) {
  using namespace nta;
  const std::uint32_t relative = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint64_t slot64 =
      static_cast<std::uint64_t>(firstObject) + relative;
  if (runtime == nullptr || relative >= objectCount ||
      slot64 >= runtime->objectCapacity) {
    return;
  }
  abi::ObjectEntry &object =
      runtime->objects[static_cast<std::uint32_t>(slot64)];
  (void)atomicCAS(&object.state,
                  static_cast<std::uint32_t>(abi::ObjectState::Ready),
                  static_cast<std::uint32_t>(abi::ObjectState::New));
}

extern "C" __global__ void nta_reset_epoch(nta::abi::RuntimeView *runtime,
                                           std::uint32_t objectCount,
                                           std::uint32_t workTicketCount) {
  using namespace nta;
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t stride = blockDim.x * gridDim.x;
  if (index == 0) {
    runtime->epochStartClock = device::globalTimerNs();
    std::uint32_t epoch = atomicAdd(&runtime->epoch, 1U) + 1U;
    if (epoch == 0) {
      atomicExch(&runtime->epoch, 1U);
    }
    runtime->completedCount = 0;
    runtime->failedCount = 0;
    *runtime->readyCount = 0;
    *runtime->readyHead = 0;
    runtime->readyWindowEnd = 0;
    *runtime->pendingCount = 0;
    if (runtime->changedCount != nullptr) {
      *runtime->changedCount = 0;
    }
    if (runtime->changedOverflow != nullptr) {
      *runtime->changedOverflow = 0;
    }
  }
  for (std::uint32_t requestSlot = index;
       runtime->requestProgress != nullptr &&
       requestSlot < runtime->requestCapacity;
       requestSlot += stride) {
    const abi::RequestContext &request = runtime->requests[requestSlot];
    runtime->requestProgress[requestSlot] = {
        request.requestId, request.generation, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
  }
  for (std::uint32_t source = index;
       runtime->intentQueueControls != nullptr && source < abi::BackendCount;
       source += stride) {
    runtime->intentQueueControls[source] = {};
  }
  for (std::uint32_t intentSlot = index;
       runtime->intentQueueEntries != nullptr &&
       intentSlot < runtime->intentCapacity;
       intentSlot += stride) {
    runtime->intentQueueEntries[intentSlot].state = 0;
  }
  for (std::uint32_t objectSlot = index; objectSlot < runtime->objectCapacity;
       objectSlot += stride) {
    if (objectSlot < objectCount) {
      runtime->objects[objectSlot].issueCount = 0;
    }
    if (runtime->objectDependentHeads != nullptr) {
      runtime->objectDependentHeads[objectSlot] = abi::InvalidIndex;
    }
    // Ready staging remains an HBM cache entry keyed by (objectId, version).
    // Registering a changed directory key invalidates it explicitly.
  }
  if (index < workTicketCount && index < runtime->workTicketCapacity) {
    if (runtime->workRunnableNs != nullptr) {
      runtime->workRunnableNs[index] = 0;
    }
    abi::WorkTicket &workTicket = runtime->workTickets[index];
    workTicket.state = static_cast<std::uint32_t>(abi::WorkTicketState::New);
    workTicket.dependencyCount = 0;
    workTicket.dependencyStart = abi::InvalidIndex;
    workTicket.epoch = 0;
    workTicket.unavailableBytes = 0;
    workTicket.estimatedComputeNs = 0;
    workTicket.reductionGroup = abi::InvalidIndex;
    workTicket.contributorCount = 0;
    if (runtime->reductionExpected != nullptr) {
      runtime->reductionExpected[index] = 0;
    }
    if (runtime->reductionCompleted != nullptr) {
      runtime->reductionCompleted[index] = 0;
    }
    if (runtime->reductionFailed != nullptr) {
      runtime->reductionFailed[index] = 0;
    }
    if (runtime->ctaCompletions != nullptr) {
      runtime->ctaCompletions[index] = 0;
    }
    if (runtime->remainingDependencies != nullptr) {
      runtime->remainingDependencies[index] = 0;
    }
    if (runtime->changedQueued != nullptr) {
      runtime->changedQueued[index] = 0;
    }
  }
}

// Retire only work that executed in the preceding stream-ordered kernel.
// Pending work remains eligible for runnable-work publication and a later
// launch.
extern "C" __global__ void
nta_complete_launched(nta::abi::RuntimeView *runtime,
                      std::uint32_t workTicketCount) {
  if (runtime == nullptr) {
    return;
  }
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= workTicketCount || index >= runtime->workTicketCapacity) {
    return;
  }
  nta::abi::WorkTicket &workTicket = runtime->workTickets[index];
  if (atomicAdd(&workTicket.state, 0U) ==
      static_cast<std::uint32_t>(nta::abi::WorkTicketState::New)) {
    workTicket.epoch = nta::device::currentEpoch(runtime);
  }
  (void)nta::device::completeWorkTicket(runtime, index);
}

// Retire one exact work plan after its application kernel. Kernel-launch stream
// order supplies the data-publication boundary, so each ticket is initialized
// and completed once instead of every CTA contending on the same record.
extern "C" __global__ void
nta_complete_stream_ordered(nta::abi::RuntimeView *runtime,
                            const nta::abi::WorkItem *workItems,
                            std::uint32_t workItemCount) {
  using namespace nta;
  if (runtime == nullptr || workItems == nullptr) {
    return;
  }
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= workItemCount) {
    return;
  }
  const abi::WorkItem item = workItems[index];
  if (item.workTicket >= runtime->workTicketCapacity ||
      item.requestSlot >= runtime->requestCapacity ||
      item.reductionGroup >= runtime->workTicketCapacity ||
      item.contributorCount == 0 ||
      item.contributorIndex >= item.contributorCount) {
    device::recordFailure(runtime);
    return;
  }

  abi::WorkTicket &ticket = runtime->workTickets[item.workTicket];
  const auto state =
      static_cast<abi::WorkTicketState>(atomicAdd(&ticket.state, 0U));
  const abi::RequestContext &request = runtime->requests[item.requestSlot];
  if (state == abi::WorkTicketState::New) {
    ticket.requestId = request.requestId;
    ticket.requestSlot = item.requestSlot;
    ticket.generation = request.generation;
    ticket.dependencyCount = 0;
    ticket.logicalTile = item.logicalWork;
    ticket.dependencyStart = abi::InvalidIndex;
    ticket.epoch = device::currentEpoch(runtime);
    ticket.unavailableBytes = 0;
    ticket.estimatedComputeNs = item.estimatedComputeNs;
    ticket.reductionGroup = item.reductionGroup;
    ticket.contributorCount = item.contributorCount;
  } else if (state != abi::WorkTicketState::Ready ||
             !device::ticketMatches(runtime, ticket, item.requestSlot,
                                    request.generation) ||
             ticket.logicalTile != item.logicalWork ||
             ticket.reductionGroup != item.reductionGroup ||
             ticket.contributorCount != item.contributorCount) {
    if (state != abi::WorkTicketState::Done &&
        state != abi::WorkTicketState::Cancelled) {
      device::recordFailure(runtime);
    }
    return;
  }

  if (request.cancelled != 0) {
    const std::uint32_t expected = static_cast<std::uint32_t>(state);
    if (atomicCAS(&ticket.state, expected,
                  static_cast<std::uint32_t>(
                      abi::WorkTicketState::Cancelled)) == expected) {
      device::recordTerminalWork(runtime, ticket, state,
                                 abi::WorkTicketState::Cancelled);
    }
    return;
  }
  (void)device::completeWorkTicket(runtime, item.workTicket);
}
#endif
