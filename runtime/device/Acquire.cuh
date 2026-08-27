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

__device__ __forceinline__ std::uint32_t
urgencyBucket(std::uint32_t priority, std::uint64_t deadline, std::uint64_t now,
              std::uint64_t criticalService) {
  std::uint32_t urgency = priority > 7U ? 7U : priority;
  if (deadline == 0) {
    // With no explicit SLO, combine the caller's priority with a bounded
    // shortest-critical-work preference to reduce mean completion time.
    const std::uint32_t serviceUrgency = criticalService == 0              ? 0U
                                         : criticalService <= 100'000ULL   ? 3U
                                         : criticalService <= 500'000ULL   ? 2U
                                         : criticalService <= 2'000'000ULL ? 1U
                                                                           : 0U;
    return urgency > serviceUrgency ? urgency : serviceUrgency;
  }
  const std::uint64_t slack = deadline > now ? deadline - now : 0;
  const std::uint32_t deadlineUrgency =
      slack == 0 || (criticalService != 0 && slack <= criticalService) ? 7U
      : criticalService != 0 &&
              slack <= saturatingAdd(criticalService, criticalService)
          ? 6U
      : criticalService != 0 &&
              slack <=
                  saturatingAdd(saturatingAdd(criticalService, criticalService),
                                saturatingAdd(criticalService, criticalService))
          ? 5U
      : slack <= 5'000'000ULL ? 4U
                              : 0U;
  return urgency > deadlineUrgency ? urgency : deadlineUrgency;
}

__device__ __forceinline__ std::uint64_t taggedIntentHead(std::uint32_t tag,
                                                          std::uint32_t index) {
  return (static_cast<std::uint64_t>(tag) << 32U) | index;
}

__device__ __forceinline__ bool
intentQueueAvailable(const abi::RuntimeView *runtime) {
  return runtime != nullptr && runtime->intentQueueEntries != nullptr &&
         runtime->intentQueueHeads != nullptr && runtime->intents != nullptr &&
         runtime->intentCapacity != 0;
}

__device__ __forceinline__ bool pushIntentQueueEntry(abi::RuntimeView *runtime,
                                                     std::uint32_t slotIndex,
                                                     abi::SourceKind source,
                                                     std::uint32_t urgency) {
  if (!intentQueueAvailable(runtime) || slotIndex >= runtime->intentCapacity ||
      urgency >= abi::UrgencyBucketCount) {
    return false;
  }
  const std::uint32_t sourceIndex = static_cast<std::uint32_t>(source);
  if (sourceIndex >= runtime->backendCapacity ||
      sourceIndex >= abi::BackendCount) {
    return false;
  }
  abi::IntentQueueEntry &entry = runtime->intentQueueEntries[slotIndex];
  entry.epoch = currentEpoch(runtime);
  entry.sourceKind = sourceIndex;
  entry.urgency = urgency;
  auto *head = reinterpret_cast<unsigned long long *>(
      &runtime
           ->intentQueueHeads[sourceIndex * abi::UrgencyBucketCount + urgency]);
  unsigned long long observed = atomicAdd(head, 0ULL);
  for (;;) {
    entry.next = static_cast<std::uint32_t>(observed);
    __threadfence();
    const std::uint32_t tag = static_cast<std::uint32_t>(observed >> 32U);
    const unsigned long long desired = taggedIntentHead(tag + 1U, slotIndex);
    const unsigned long long prior = atomicCAS(head, observed, desired);
    if (prior == observed) {
      return true;
    }
    observed = prior;
  }
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
  abi::IntentQueueEntry &entry = runtime->intentQueueEntries[slotIndex];
  if (atomicCAS(&entry.state, 0U, 1U) != 0U) {
    return false;
  }
  entry.sequence = slot.sequence;
  const std::uint32_t urgency = urgencyBucket(
      slot.intent.priority, slot.intent.deadlineClock, globalTimerNs(),
      criticalServiceNs(runtime, slot.intent, source));
  if (pushIntentQueueEntry(runtime, slotIndex, source, urgency)) {
    return true;
  }
  atomicExch(&entry.state, 0U);
  return false;
}

__device__ __forceinline__ std::uint32_t popIntent(abi::RuntimeView *runtime,
                                                   abi::SourceKind source) {
  if (!intentQueueAvailable(runtime)) {
    return abi::InvalidIndex;
  }
  const std::uint32_t sourceIndex = static_cast<std::uint32_t>(source);
  if (sourceIndex >= runtime->backendCapacity ||
      sourceIndex >= abi::BackendCount) {
    return abi::InvalidIndex;
  }
  for (std::uint32_t reverse = 0; reverse < abi::UrgencyBucketCount;
       ++reverse) {
    const std::uint32_t urgency = abi::UrgencyBucketCount - 1U - reverse;
    auto *head = reinterpret_cast<unsigned long long *>(
        &runtime->intentQueueHeads[sourceIndex * abi::UrgencyBucketCount +
                                   urgency]);
    for (std::uint32_t attempt = 0; attempt < runtime->intentCapacity;
         ++attempt) {
      const unsigned long long observed = atomicAdd(head, 0ULL);
      const std::uint32_t slotIndex = static_cast<std::uint32_t>(observed);
      if (slotIndex == abi::InvalidIndex) {
        break;
      }
      if (slotIndex >= runtime->intentCapacity) {
        return abi::InvalidIndex;
      }
      abi::IntentQueueEntry &entry = runtime->intentQueueEntries[slotIndex];
      const std::uint32_t tag = static_cast<std::uint32_t>(observed >> 32U);
      const unsigned long long desired = taggedIntentHead(tag + 1U, entry.next);
      if (atomicCAS(head, observed, desired) != observed) {
        continue;
      }
      if (atomicCAS(&entry.state, 1U, 2U) != 1U) {
        continue;
      }
      abi::IntentSlot &slot = runtime->intents[slotIndex];
      const bool valid = entry.epoch == currentEpoch(runtime) &&
                         entry.sourceKind == sourceIndex &&
                         entry.urgency == urgency &&
                         entry.sequence == slot.sequence &&
                         atomicAdd(&slot.intent.valid, 0U) == 1U &&
                         slot.sourceKind == sourceIndex;
      if (valid) {
        return slotIndex;
      }
      atomicExch(&entry.state, 0U);
    }
  }
  return abi::InvalidIndex;
}

__device__ __forceinline__ bool requeueIntent(abi::RuntimeView *runtime,
                                              std::uint32_t slotIndex,
                                              abi::SourceKind source) {
  if (!intentQueueAvailable(runtime) || slotIndex >= runtime->intentCapacity) {
    return false;
  }
  abi::IntentQueueEntry &entry = runtime->intentQueueEntries[slotIndex];
  if (atomicCAS(&entry.state, 2U, 1U) != 2U) {
    return false;
  }
  abi::IntentSlot &slot = runtime->intents[slotIndex];
  entry.sequence = slot.sequence;
  const std::uint32_t urgency = urgencyBucket(
      slot.intent.priority, slot.intent.deadlineClock, globalTimerNs(),
      criticalServiceNs(runtime, slot.intent, source));
  if (pushIntentQueueEntry(runtime, slotIndex, source, urgency)) {
    return true;
  }
  atomicExch(&entry.state, 0U);
  return false;
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
         loadIoCoherent(&control.abiVersion) == abi::NvmeQueueAbiVersion &&
         loadIoCoherent(&control.queueId) == queue.queueId &&
         loadIoCoherent(&control.generation) == queue.queueGeneration &&
         loadIoCoherent(&control.state) ==
             static_cast<std::uint32_t>(abi::NvmeQueueState::Online);
}

__device__ __forceinline__ void consumeIntent(abi::RuntimeView *runtime,
                                              abi::IntentSlot &slot);
__device__ __forceinline__ bool claimIntent(abi::IntentSlot &slot);

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
                                              abi::SourceKind source) {
  slot.sourceKind = static_cast<std::uint32_t>(source);
  slot.epoch = currentEpoch(runtime);
  __threadfence();
  atomicAdd(
      reinterpret_cast<unsigned long long *>(&runtime->intentPool->enqueued),
      1ULL);
  atomicAdd(&runtime->intentPool->active, 1U);
  __threadfence();
  atomicExch(&slot.intent.valid, 1U);
  __threadfence();
  if (!queueIntent(runtime, slot, source) && intentQueueAvailable(runtime)) {
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
      atomicExch(&runtime->intentQueueEntries[slotIndex].state, 0U);
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

__device__ __forceinline__ bool tryClaimValidatedIndexedHostIntent(
    abi::RuntimeView *runtime, abi::IntentSlot &intentSlot,
    abi::ObjectEntry &object, const abi::ReplicaEntry &replica) {
  if ((replica.flags & (abi::ReplicaIndexed | abi::ReplicaIndicesValidated)) !=
          (abi::ReplicaIndexed | abi::ReplicaIndicesValidated) ||
      replica.sourceKind !=
          static_cast<std::uint32_t>(abi::SourceKind::HostStaged) ||
      replica.sourceAddress == 0 || object.stagingAddress == 0) {
    return false;
  }
  const abi::AcquireIntent intent = intentSlot.intent;
  const bool live = requestLive(runtime, intent.requestSlot, intent.generation);
  bool accepted = true;
  if (live) {
    accepted = tryReserveRequestBytes(runtime, intent.requestSlot,
                                      intent.generation, intent.bytes);
    if (accepted &&
        !tryReserveTenantBytes(runtime, intent.tenantId, intent.bytes)) {
      releaseRequestBytes(runtime, intent.requestSlot, intent.generation,
                          intent.bytes);
      accepted = false;
    }
  }
  if (accepted && !tryReserveBackendBytes(runtime, abi::SourceKind::HostStaged,
                                          intent.bytes)) {
    if (live) {
      releaseRequestBytes(runtime, intent.requestSlot, intent.generation,
                          intent.bytes);
      releaseTenantBytes(runtime, intent.tenantId, intent.bytes);
    }
    accepted = false;
  }
  if (accepted && !claimIntent(intentSlot)) {
    if (live) {
      releaseRequestBytes(runtime, intent.requestSlot, intent.generation,
                          intent.bytes);
      releaseTenantBytes(runtime, intent.tenantId, intent.bytes);
    }
    releaseBackendBytes(runtime, abi::SourceKind::HostStaged, intent.bytes);
    accepted = false;
  }
  if (accepted) {
    __threadfence();
    atomicExch(&object.state,
               static_cast<std::uint32_t>(abi::ObjectState::Issued));
  }
  return accepted;
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
  const std::uint64_t lbaSize = 1ULL << queue.lbaShift;
  const std::uint64_t lbaCount = object.bytes >> queue.lbaShift;
  return expectedPages64 <= UINT32_MAX &&
         replica->dmaPageCount == static_cast<std::uint32_t>(expectedPages64) &&
         dmaPages[0] % queue.controllerPageSize == 0 &&
         object.bytes % lbaSize == 0 && replica->sourceAddress % lbaSize == 0 &&
         lbaCount != 0 && lbaCount <= 65'536ULL &&
         replica->dmaPageCount <=
             queue.controllerPageSize / sizeof(std::uint64_t) &&
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
publishNvmeRead(abi::RuntimeView *runtime, abi::NvmeQueueView &queue,
                abi::ObjectEntry &object, const abi::ReplicaEntry &replica,
                const abi::AcquireIntent &intent,
                const NvmeAdmission &admission, std::uint32_t commandId,
                std::uint32_t submissionSlot, abi::IntentSlot *consumedIntent,
                bool directSubmission) {
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
  queue.sqTail = (submissionSlot + 1U) % queue.depth;
  ++queue.outstanding;
  ++queue.submitted;
  if (directSubmission) {
    ++queue.directSubmitted;
  }
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

extern "C" __device__ __attribute__((used, noinline)) void *
nta_acquire_slow(nta::abi::RuntimeView *runtime, std::uint32_t requestSlot,
                 std::uint32_t generation, std::uint32_t objectSlot,
                 std::uint64_t objectId, std::uint32_t objectVersion,
                 std::uint64_t offset, std::uint32_t bytes,
                 std::uint32_t workTicket) {
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

  if (threadIdx.x == 0) {
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
  if (threadIdx.x == 0 &&
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
    pending.deadlineClock = request.deadlineClock;
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
        device::publishIntent(runtime, *intentSlot, source);
        if (source == abi::SourceKind::HostStaged) {
          (void)device::tryClaimValidatedIndexedHostIntent(runtime, *intentSlot,
                                                           object, *selected);
        }
        if (!device::backendAcceptsIntent(runtime, source) &&
            device::claimIntent(*intentSlot)) {
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

extern "C" __device__ __attribute__((used, noinline)) std::uint32_t
nta_acquire_set_leader(nta::abi::RuntimeView *runtime,
                       std::uint32_t requestSlot, std::uint32_t generation,
                       const nta::abi::AcquireRequirement *requirements,
                       std::uint32_t requirementCount,
                       std::uint32_t directRequirementCount,
                       std::uint32_t workTicket) {
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
    acquired &= nta_acquire_slow(runtime, requestSlot, generation,
                                 requirement.objectSlot, requirement.objectId,
                                 requirement.objectVersion, requirement.offset,
                                 requirement.bytes, workTicket) != nullptr;
  }
  return acquired ? 1U : 0U;
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
__device__ __forceinline__ void
progressNvmeOnce(nta::abi::RuntimeView *runtime, std::uint32_t issueBudget,
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
  bool ownsQueue = lane == 0 && atomicCAS(&queue.ownerLock, 0U, 1U) == 0U;
  ownsQueue = __shfl_sync(0xffffffffU, ownsQueue, 0);
  if (!ownsQueue) {
    return;
  }
  bool queueOnline = lane == 0 && device::nvmeQueueOnline(queue);
  queueOnline = __shfl_sync(0xffffffffU, queueOnline, 0);
  if (!queueOnline) {
    device::failNvmeQueue(runtime, queue, lane, 0xfffffffcU);
    if (lane == 0) {
      __threadfence();
      atomicExch(&queue.ownerLock, 0U);
    }
    return;
  }

  bool malformedCompletion = false;
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
      bool contextValid = commandId < queue.depth;
      if (contextValid) {
        abi::NvmeCommandContext &stored = queue.contexts[commandId];
        contextValid = atomicAdd(&stored.active, 0U) != 0;
        if (contextValid) {
          const abi::NvmeCommandContext context = stored;
          bool objectCurrent = context.epoch == device::currentEpoch(runtime) &&
                               context.objectSlot < runtime->objectCapacity;
          if (objectCurrent) {
            abi::ObjectEntry &object = runtime->objects[context.objectSlot];
            const abi::ReplicaEntry *replica =
                device::replica(runtime, object, object.selectedReplica);
            objectCurrent =
                object.objectId == context.objectId &&
                object.version == context.objectVersion && replica != nullptr &&
                replica->sourceKind ==
                    static_cast<std::uint32_t>(abi::SourceKind::Nvme);
            if (objectCurrent && (statusField >> 1U) == 0) {
              atomicExch(&object.state,
                         static_cast<std::uint32_t>(abi::ObjectState::Ready));
              device::publishObjectTransition(runtime, context.objectSlot,
                                              abi::ObjectState::Ready);
              ++queue.completed;
            } else if (objectCurrent) {
              atomicExch(&object.state,
                         static_cast<std::uint32_t>(abi::ObjectState::Failed));
              device::publishObjectTransition(runtime, context.objectSlot,
                                              abi::ObjectState::Failed);
              device::failBoundWorkTicket(runtime, context.workTicket,
                                          context.requestSlot,
                                          context.generation);
              ++queue.failed;
              queue.error = statusField >> 1U;
            }
          }
          if (!objectCurrent) {
            ++queue.failed;
            queue.error = 0xfffffffbU;
          }
          device::releaseRequestBytes(runtime, context.requestSlot,
                                      context.generation, context.bytes);
          device::releaseTenantBytes(runtime, context.tenantId, context.bytes);
          device::releaseBackendBytes(runtime, abi::SourceKind::Nvme,
                                      context.backendBytes);
          atomicExch(&stored.active, 0U);
        }
      }
      if (!contextValid) {
        malformedCompletion = true;
        break;
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
      device::storeMmio(queue.cqDoorbell, queue.cqHead);
    }
  }
  malformedCompletion = __shfl_sync(0xffffffffU, malformedCompletion, 0);
  if (malformedCompletion) {
    device::failNvmeQueue(runtime, queue, lane, 0xffffffffU);
    if (lane == 0) {
      __threadfence();
      atomicExch(&queue.ownerLock, 0U);
    }
    return;
  }
  __syncwarp();

  std::uint32_t issued = 0U;
  for (std::uint32_t attempt = 0; attempt < issueBudget; ++attempt) {
    std::uint32_t intentSlotIndex = abi::InvalidIndex;
    std::uint32_t objectSlot = abi::InvalidIndex;
    std::uint32_t commandId = abi::InvalidIndex;
    std::uint32_t submissionSlot = 0;
    std::uint32_t action = 0;
    std::uint64_t chargedBytes = 0;
    std::uint64_t backendBytes = 0;
    if (lane == 0 && queue.outstanding + 1U < queue.depth) {
      intentSlotIndex = device::popIntent(runtime, abi::SourceKind::Nvme);
      if (intentSlotIndex != abi::InvalidIndex &&
          intentSlotIndex < runtime->intentPool->capacity &&
          intentSlotIndex < runtime->intentCapacity) {
        abi::IntentSlot &selectedSlot = runtime->intents[intentSlotIndex];
        abi::AcquireIntent &selected = selectedSlot.intent;
        if (selected.objectSlot >= runtime->objectCapacity) {
          if (device::claimIntent(selectedSlot)) {
            device::failBoundWorkTicket(runtime, selected.workTicket,
                                        selected.requestSlot,
                                        selected.generation);
            device::consumeIntent(runtime, selectedSlot);
          }
          ++queue.failed;
          queue.error = 0xfffffffbU;
          action = 1;
        } else {
          abi::ObjectEntry &object = runtime->objects[selected.objectSlot];
          const abi::ReplicaEntry *replica =
              device::replica(runtime, object, object.selectedReplica);
          const bool objectCurrent = object.objectId == selected.objectId &&
                                     object.version == selected.objectVersion;
          if (!device::validNvmeTransfer(runtime, queue, selected, object,
                                         replica)) {
            if (device::claimIntent(selectedSlot)) {
              if (objectCurrent) {
                atomicExch(&object.state, static_cast<std::uint32_t>(
                                              abi::ObjectState::Failed));
                device::publishObjectTransition(runtime, selected.objectSlot,
                                                abi::ObjectState::Failed);
              }
              device::failBoundWorkTicket(runtime, selected.workTicket,
                                          selected.requestSlot,
                                          selected.generation);
              device::consumeIntent(runtime, selectedSlot);
            }
            ++queue.failed;
            queue.error = 0xfffffffeU;
            action = 1;
          } else {
            const device::NvmeAdmission admission =
                device::tryAdmitNvme(runtime, selected, object.bytes);
            if (admission.admitted) {
              chargedBytes = admission.requestBytes;
              backendBytes = admission.backendBytes;
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
            }
            if (admission.admitted && commandId != abi::InvalidIndex &&
                device::claimIntent(selectedSlot)) {
              objectSlot = selected.objectSlot;
              submissionSlot = queue.sqTail;
              action = 2;
            } else {
              device::releaseNvmeAdmission(runtime, selected, admission);
              chargedBytes = 0;
              backendBytes = 0;
              if (atomicAdd(&selected.valid, 0U) == 1U &&
                  !device::requeueIntent(runtime, intentSlotIndex,
                                         abi::SourceKind::Nvme)) {
                atomicAdd(&runtime->intentPool->overflow, 1U);
              }
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
    device::prepareNvmeRead(queue, replica, commandId, submissionSlot, lane,
                            warpSize);
    __syncwarp();

    if (lane == 0) {
      const device::NvmeAdmission admission{chargedBytes, backendBytes, true};
      device::publishNvmeRead(runtime, queue, object, replica, selected,
                              admission, commandId, submissionSlot,
                              &selectedSlot, false);
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
  progressNvmeOnce(runtime, issueBudget, completionBudget);
}

extern "C" __global__ void nta_progress_nvme_until_idle(
    nta::abi::RuntimeView *runtime, std::uint32_t issueBudget,
    std::uint32_t completionBudget, std::uint64_t timeoutNs) {
  using namespace nta;
  if (runtime == nullptr || blockIdx.x != 0 || threadIdx.x >= warpSize ||
      timeoutNs == 0) {
    return;
  }
  const std::uint32_t lane = threadIdx.x;
  const std::uint64_t start = device::globalTimerNs();
  for (;;) {
    progressNvmeOnce(runtime, issueBudget, completionBudget);
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

} // namespace nta::device

// KV-aware lookahead mover. Two CTAs cooperatively copy one adjacent K/V
// object pair while preserving the generic object-directory representation.
extern "C" __global__ __launch_bounds__(
    1024, 1) void nta_preload_indexed_host_pairs(nta::abi::RuntimeView *runtime,
                                                 std::uint32_t firstObject,
                                                 std::uint32_t pairCount) {
  using namespace nta;
  constexpr std::uint32_t BlocksPerPair = 2;
  const std::uint32_t relativePair = blockIdx.x / BlocksPerPair;
  const std::uint32_t pairBlock = blockIdx.x % BlocksPerPair;
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
  const abi::ReplicaEntry *keyReplica = device::replica(runtime, keyObject, 0);
  const abi::ReplicaEntry *valueReplica =
      device::replica(runtime, valueObject, 0);
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

  __shared__ std::uint32_t invalidIndex;
  if (threadIdx.x == 0) {
    invalidIndex = 0;
  }
  __syncthreads();
  if (valid) {
    device::validateIndexedTransferIndices(keyObject, *keyReplica,
                                           &invalidIndex);
    device::validateIndexedTransferIndices(valueObject, *valueReplica,
                                           &invalidIndex);
  }
  __syncthreads();
  const bool bounded = valid && invalidIndex == 0;
  if (threadIdx.x == 0 && pairBlock == 0) {
    keyObject.selectedReplica = 0;
    valueObject.selectedReplica = 0;
    atomicExch(&keyObject.state,
               static_cast<std::uint32_t>(bounded ? abi::ObjectState::Issued
                                                  : abi::ObjectState::Failed));
    atomicExch(&valueObject.state,
               static_cast<std::uint32_t>(bounded ? abi::ObjectState::Issued
                                                  : abi::ObjectState::Failed));
  }
  if (!bounded) {
    return;
  }
  switch (elementBytes) {
  case 128:
    device::copyIndexedHostPair<128, 8>(keyObject, *keyReplica, valueObject,
                                        *valueReplica, pairBlock,
                                        BlocksPerPair);
    break;
  case 256:
    device::copyIndexedHostPair<256, 8>(keyObject, *keyReplica, valueObject,
                                        *valueReplica, pairBlock,
                                        BlocksPerPair);
    break;
  case 512:
    device::copyIndexedHostPair<512, 32>(keyObject, *keyReplica, valueObject,
                                         *valueReplica, pairBlock,
                                         BlocksPerPair);
    break;
  case 1024:
    device::copyIndexedHostPair<1024, 16>(keyObject, *keyReplica, valueObject,
                                          *valueReplica, pairBlock,
                                          BlocksPerPair);
    break;
  case 2048:
    device::copyIndexedHostPair<2048, 32>(keyObject, *keyReplica, valueObject,
                                          *valueReplica, pairBlock,
                                          BlocksPerPair);
    break;
  }
  __syncthreads();
  if (threadIdx.x == 0 && pairBlock == 0) {
    __threadfence_system();
    atomicExch(&keyObject.state,
               static_cast<std::uint32_t>(abi::ObjectState::Ready));
    atomicExch(&valueObject.state,
               static_cast<std::uint32_t>(abi::ObjectState::Ready));
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
  if (threadIdx.x == 0) {
    selectedFromQueue = device::intentQueueAvailable(runtime) ? 1U : 0U;
    selectedIntent =
        selectedFromQueue != 0U
            ? device::popIntent(runtime, abi::SourceKind::HostStaged)
            : static_cast<std::uint32_t>(blockIdx.x);
  }
  __syncthreads();
  if (selectedIntent == abi::InvalidIndex ||
      selectedIntent >= runtime->intentPool->capacity ||
      selectedIntent >= runtime->intentCapacity) {
    return;
  }
  abi::IntentSlot &intentSlot = runtime->intents[selectedIntent];

  abi::AcquireIntent &intent = intentSlot.intent;
  if (atomicAdd(&intent.valid, 0U) != 1U ||
      intentSlot.sourceKind !=
          static_cast<std::uint32_t>(abi::SourceKind::HostStaged)) {
    return;
  }
  if (intentSlot.epoch != device::currentEpoch(runtime)) {
    if (threadIdx.x == 0 && device::claimIntent(intentSlot)) {
      device::consumeIntent(runtime, intentSlot);
    }
    return;
  }
  if (intent.objectSlot >= runtime->objectCapacity) {
    if (threadIdx.x == 0 && device::claimIntent(intentSlot)) {
      device::failBoundWorkTicket(runtime, intent.workTicket,
                                  intent.requestSlot, intent.generation);
      device::consumeIntent(runtime, intentSlot);
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
  __shared__ std::uint32_t invalidIndex;
  if (threadIdx.x == 0) {
    invalidIndex = 0;
  }
  __syncthreads();
  if (transferValid && indexed) {
    device::validateIndexedTransferIndices(object, *replica, &invalidIndex);
  }
  __syncthreads();
  if (!transferValid || invalidIndex != 0) {
    if (threadIdx.x == 0) {
      if (device::claimIntent(intentSlot)) {
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
    if (threadIdx.x == 0 && selectedFromQueue != 0U &&
        atomicAdd(&intentSlot.intent.valid, 0U) == 1U &&
        !device::requeueIntent(runtime, selectedIntent,
                               abi::SourceKind::HostStaged)) {
      atomicAdd(&runtime->intentPool->overflow, 1U);
    }
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
    device::releaseRequestBytes(runtime, intent.requestSlot, intent.generation,
                                chargedBytes);
    device::releaseTenantBytes(runtime, intent.tenantId, chargedBytes);
    device::releaseBackendBytes(runtime, abi::SourceKind::HostStaged,
                                backendBytes);
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

  const bool live =
      device::requestLive(runtime, intent.requestSlot, intent.generation);
  bool accepted = true;
  if (live) {
    accepted = device::tryReserveRequestBytes(runtime, intent.requestSlot,
                                              intent.generation, intent.bytes);
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
  if (accepted && !device::claimIntent(intentSlot)) {
    if (live) {
      device::releaseRequestBytes(runtime, intent.requestSlot,
                                  intent.generation, intent.bytes);
      device::releaseTenantBytes(runtime, intent.tenantId, intent.bytes);
    }
    device::releaseBackendBytes(runtime, abi::SourceKind::HostStaged,
                                intent.bytes);
    accepted = false;
  }
  if (accepted) {
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
  // Indexed range progress is a high-fanout path: one K/V wave commonly feeds
  // hundreds of attention work tickets. Walking each object's singly-linked
  // dependency list here would serialize that fanout on one thread per object.
  // Force the stream-ordered publication kernel to scan pending work in
  // parallel after every object in this range has reached its terminal state.
  if (runtime->changedOverflow != nullptr) {
    atomicExch(runtime->changedOverflow, 1U);
  }
  const bool live =
      device::requestLive(runtime, intent.requestSlot, intent.generation);
  device::releaseRequestBytes(runtime, intent.requestSlot, intent.generation,
                              live ? intent.bytes : 0);
  device::releaseTenantBytes(runtime, intent.tenantId, live ? intent.bytes : 0);
  device::releaseBackendBytes(runtime, abi::SourceKind::HostStaged,
                              intent.bytes);
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
  for (std::uint32_t queueHead = index;
       runtime->intentQueueHeads != nullptr &&
       queueHead < abi::BackendCount * abi::UrgencyBucketCount;
       queueHead += stride) {
    runtime->intentQueueHeads[queueHead] = device::taggedIntentHead(
        device::currentEpoch(runtime), abi::InvalidIndex);
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
      device::recordFailure(runtime);
    }
    return;
  }
  (void)device::completeWorkTicket(runtime, item.workTicket);
}
#endif
