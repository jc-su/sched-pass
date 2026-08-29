#pragma once

#include "nta/RuntimeABI.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace nta::device {

__device__ __forceinline__ std::uint32_t
currentEpoch(const abi::RuntimeView *runtime);

__device__ __forceinline__ void recordFailure(abi::RuntimeView *runtime);

__device__ __forceinline__ void
recordDroppedAttribution(abi::RequestProgress &progress) {
  atomicAdd(reinterpret_cast<unsigned long long *>(
                &progress.droppedAttributions),
            1ULL);
}

__device__ __forceinline__ abi::RequestProgress *
requestProgress(abi::RuntimeView *runtime, const abi::WorkTicket &ticket) {
  if (runtime == nullptr || runtime->requestProgress == nullptr ||
      ticket.requestSlot >= runtime->requestCapacity) {
    return nullptr;
  }
  abi::RequestProgress &progress = runtime->requestProgress[ticket.requestSlot];
  if (progress.requestId != ticket.requestId ||
      progress.generation != ticket.generation) {
    return nullptr;
  }
  const std::uint32_t epoch = currentEpoch(runtime);
  const std::uint32_t observed = atomicCAS(&progress.epoch, 0U, epoch);
  if (observed != 0U && observed != epoch) {
    recordDroppedAttribution(progress);
    return nullptr;
  }
  return &progress;
}

__device__ __forceinline__ void addProgressValue(std::uint64_t *value,
                                                 std::uint64_t increment) {
  if (increment != 0) {
    atomicAdd(reinterpret_cast<unsigned long long *>(value),
              static_cast<unsigned long long>(increment));
  }
}

__device__ __forceinline__ bool subtractProgressValue(std::uint64_t *value,
                                                      std::uint64_t decrement) {
  if (decrement == 0) {
    return true;
  }
  auto *atomicValue = reinterpret_cast<unsigned long long *>(value);
  unsigned long long observed = atomicAdd(atomicValue, 0ULL);
  while (observed >= decrement) {
    const unsigned long long previous = atomicCAS(
        atomicValue, observed, observed - static_cast<unsigned long long>(decrement));
    if (previous == observed) {
      return true;
    }
    observed = previous;
  }
  return false;
}

__device__ __forceinline__ bool subtractProgressCounter(std::uint32_t *value,
                                                        std::uint32_t decrement) {
  std::uint32_t observed = atomicAdd(value, 0U);
  while (observed >= decrement) {
    const std::uint32_t previous = atomicCAS(value, observed, observed - decrement);
    if (previous == observed) {
      return true;
    }
    observed = previous;
  }
  return false;
}

__device__ __forceinline__ bool
recordReductionExpected(abi::RuntimeView *runtime,
                        const abi::WorkTicket &ticket) {
  if (runtime->reductionExpected == nullptr ||
      runtime->reductionCompleted == nullptr ||
      runtime->reductionFailed == nullptr) {
    return true;
  }
  if (ticket.reductionGroup >= runtime->workTicketCapacity ||
      ticket.contributorCount == 0) {
    return false;
  }
  std::uint32_t &expected =
      runtime->reductionExpected[ticket.reductionGroup];
  const std::uint32_t observed =
      atomicCAS(&expected, 0U, ticket.contributorCount);
  return observed == 0U || observed == ticket.contributorCount;
}

__device__ __forceinline__ void
recordReductionTerminal(abi::RuntimeView *runtime,
                        const abi::WorkTicket &ticket,
                        abi::WorkTicketState terminal) {
  if (runtime->reductionExpected == nullptr ||
      runtime->reductionCompleted == nullptr ||
      runtime->reductionFailed == nullptr ||
      ticket.reductionGroup >= runtime->workTicketCapacity) {
    return;
  }
  if (!recordReductionExpected(runtime, ticket) ||
      terminal != abi::WorkTicketState::Done) {
    atomicAdd(&runtime->reductionFailed[ticket.reductionGroup], 1U);
    return;
  }
  atomicAdd(&runtime->reductionCompleted[ticket.reductionGroup], 1U);
}

__device__ __forceinline__ void
recordPendingWork(abi::RuntimeView *runtime, const abi::WorkTicket &ticket) {
  abi::RequestProgress *progress = requestProgress(runtime, ticket);
  if (progress != nullptr) {
    atomicAdd(&progress->expectedWork, 1U);
    addProgressValue(&progress->expectedComputeNs, ticket.estimatedComputeNs);
    atomicAdd(&progress->pendingWork, 1U);
    addProgressValue(&progress->pendingComputeNs, ticket.estimatedComputeNs);
    addProgressValue(&progress->unavailableBytes, ticket.unavailableBytes);
  }
  if (!recordReductionExpected(runtime, ticket) &&
      runtime->reductionFailed != nullptr &&
      ticket.reductionGroup < runtime->workTicketCapacity) {
    atomicAdd(&runtime->reductionFailed[ticket.reductionGroup], 1U);
  }
}

__device__ __forceinline__ void
recordRunnableWork(abi::RuntimeView *runtime, const abi::WorkTicket &ticket) {
  abi::RequestProgress *progress = requestProgress(runtime, ticket);
  if (progress != nullptr) {
    const bool bytesRetired = subtractProgressValue(
        &progress->unavailableBytes, ticket.unavailableBytes);
    const bool computeRetired = subtractProgressValue(
        &progress->pendingComputeNs, ticket.estimatedComputeNs);
    const bool workRetired =
        subtractProgressCounter(&progress->pendingWork, 1U);
    if (!bytesRetired || !computeRetired || !workRetired) {
      recordDroppedAttribution(*progress);
      recordFailure(runtime);
      return;
    }
    addProgressValue(&progress->runnableComputeNs, ticket.estimatedComputeNs);
    atomicAdd(&progress->runnableWork, 1U);
  }
}

__device__ __forceinline__ void
recordTerminalWork(abi::RuntimeView *runtime, const abi::WorkTicket &ticket,
                   abi::WorkTicketState previous,
                   abi::WorkTicketState terminal) {
  abi::RequestProgress *progress = requestProgress(runtime, ticket);
  if (progress != nullptr) {
    if (previous == abi::WorkTicketState::New) {
      atomicAdd(&progress->expectedWork, 1U);
      addProgressValue(&progress->expectedComputeNs, ticket.estimatedComputeNs);
    } else if (previous == abi::WorkTicketState::Pending) {
      const bool bytesRetired = subtractProgressValue(
          &progress->unavailableBytes, ticket.unavailableBytes);
      const bool computeRetired = subtractProgressValue(
          &progress->pendingComputeNs, ticket.estimatedComputeNs);
      const bool workRetired =
          subtractProgressCounter(&progress->pendingWork, 1U);
      if (!bytesRetired || !computeRetired || !workRetired) {
        recordDroppedAttribution(*progress);
        recordFailure(runtime);
        recordReductionTerminal(runtime, ticket,
                                abi::WorkTicketState::Failed);
        return;
      }
    } else if (previous == abi::WorkTicketState::Ready) {
      const bool computeRetired = subtractProgressValue(
          &progress->runnableComputeNs, ticket.estimatedComputeNs);
      const bool workRetired =
          subtractProgressCounter(&progress->runnableWork, 1U);
      if (!computeRetired || !workRetired) {
        recordDroppedAttribution(*progress);
        recordFailure(runtime);
        recordReductionTerminal(runtime, ticket,
                                abi::WorkTicketState::Failed);
        return;
      }
    }
    if (terminal == abi::WorkTicketState::Done) {
      addProgressValue(&progress->completedComputeNs,
                       ticket.estimatedComputeNs);
      atomicAdd(&progress->completedWork, 1U);
    } else if (terminal == abi::WorkTicketState::Cancelled) {
      atomicAdd(&progress->cancelledWork, 1U);
    } else {
      atomicAdd(&progress->failedWork, 1U);
    }
  }
  recordReductionTerminal(runtime, ticket, terminal);
}

__device__ __forceinline__ std::uint32_t
currentEpoch(const abi::RuntimeView *runtime) {
  return runtime == nullptr
             ? 0U
             : atomicAdd(const_cast<std::uint32_t *>(&runtime->epoch), 0U);
}

__device__ __forceinline__ bool
ticketMatches(const abi::RuntimeView *runtime, const abi::WorkTicket &ticket,
              std::uint32_t requestSlot, std::uint32_t generation) {
  return ticket.epoch == currentEpoch(runtime) &&
         ticket.requestSlot == requestSlot && ticket.generation == generation;
}

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

__device__ __forceinline__ void recordFailure(abi::RuntimeView *runtime) {
  atomicAdd(&runtime->failedCount, 1U);
  atomicAdd(&runtime->stickyFailedCount, 1U);
}

__device__ __forceinline__ void failWorkTicket(abi::RuntimeView *runtime,
                                               std::uint32_t workTicket,
                                               abi::WorkTicketState state) {
  if (runtime == nullptr || workTicket >= runtime->workTicketCapacity ||
      threadIdx.x != 0) {
    return;
  }
  abi::WorkTicket &ticket = runtime->workTickets[workTicket];
  ticket.epoch = currentEpoch(runtime);
  const std::uint32_t terminal = static_cast<std::uint32_t>(state);
  const std::uint32_t fresh =
      static_cast<std::uint32_t>(abi::WorkTicketState::New);
  const std::uint32_t pending =
      static_cast<std::uint32_t>(abi::WorkTicketState::Pending);
  const std::uint32_t ready =
      static_cast<std::uint32_t>(abi::WorkTicketState::Ready);
  abi::WorkTicketState previous = abi::WorkTicketState::Initializing;
  if (atomicCAS(&ticket.state, fresh, terminal) == fresh) {
    previous = abi::WorkTicketState::New;
  } else if (atomicCAS(&ticket.state, pending, terminal) == pending) {
    previous = abi::WorkTicketState::Pending;
  } else if (atomicCAS(&ticket.state, ready, terminal) == ready) {
    previous = abi::WorkTicketState::Ready;
  }
  const bool changed = previous != abi::WorkTicketState::Initializing;
  if (changed) {
    recordTerminalWork(runtime, ticket, previous, state);
    // Cancellation is a normal request-lifecycle terminal state and already
    // has independent per-request accounting.  Poisoning the runtime-lifetime
    // failure latch here would make one cancelled fan-out consumer invalidate
    // unrelated requests that share the same physical acquisition.  Only an
    // integrity/protocol failure belongs in the sticky failure sequence.
    if (state != abi::WorkTicketState::Cancelled) {
      recordFailure(runtime);
    }
  }
}

__device__ __forceinline__ void
failBoundWorkTicket(abi::RuntimeView *runtime, std::uint32_t workTicket,
                    std::uint32_t requestSlot, std::uint32_t generation) {
  if (runtime == nullptr || workTicket >= runtime->workTicketCapacity) {
    return;
  }
  abi::WorkTicket &record = runtime->workTickets[workTicket];
  if (ticketMatches(runtime, record, requestSlot, generation) &&
      atomicCAS(&record.state,
                static_cast<std::uint32_t>(abi::WorkTicketState::Pending),
                static_cast<std::uint32_t>(abi::WorkTicketState::Failed)) ==
          static_cast<std::uint32_t>(abi::WorkTicketState::Pending)) {
    recordTerminalWork(runtime, record, abi::WorkTicketState::Pending,
                       abi::WorkTicketState::Failed);
    recordFailure(runtime);
  }
}

__device__ __forceinline__ bool
completeWorkTicket(abi::RuntimeView *runtime, std::uint32_t workTicket) {
  if (runtime == nullptr || workTicket >= runtime->workTicketCapacity) {
    return false;
  }
  abi::WorkTicket &ticket = runtime->workTickets[workTicket];
  if (ticket.epoch != currentEpoch(runtime)) {
    return false;
  }
  const std::uint32_t done =
      static_cast<std::uint32_t>(abi::WorkTicketState::Done);
  const std::uint32_t fresh =
      static_cast<std::uint32_t>(abi::WorkTicketState::New);
  const std::uint32_t ready =
      static_cast<std::uint32_t>(abi::WorkTicketState::Ready);
  abi::WorkTicketState previous = abi::WorkTicketState::Initializing;
  if (atomicCAS(&ticket.state, fresh, done) == fresh) {
    previous = abi::WorkTicketState::New;
  } else if (atomicCAS(&ticket.state, ready, done) == ready) {
    previous = abi::WorkTicketState::Ready;
  }
  const bool changed = previous != abi::WorkTicketState::Initializing;
  if (changed) {
    recordTerminalWork(runtime, ticket, previous, abi::WorkTicketState::Done);
    atomicAdd(&runtime->completedCount, 1U);
  }
  return changed || atomicAdd(&ticket.state, 0U) == done;
}

__device__ __forceinline__ bool
completeBoundWorkTicket(abi::RuntimeView *runtime, std::uint32_t requestSlot,
                        std::uint32_t generation,
                        std::uint32_t workTicket) {
  if (runtime == nullptr || workTicket >= runtime->workTicketCapacity ||
      requestSlot >= runtime->requestCapacity) {
    return false;
  }
  abi::WorkTicket &ticket = runtime->workTickets[workTicket];
  const auto state =
      static_cast<abi::WorkTicketState>(atomicAdd(&ticket.state, 0U));
  if (state == abi::WorkTicketState::New) {
    if (!requestLive(runtime, requestSlot, generation)) {
      failWorkTicket(runtime, workTicket, abi::WorkTicketState::Cancelled);
      return false;
    }
    ticket.requestId = runtime->requests[requestSlot].requestId;
    ticket.requestSlot = requestSlot;
    ticket.generation = generation;
    ticket.logicalTile = workTicket;
    ticket.epoch = currentEpoch(runtime);
    __threadfence();
  } else if (!ticketMatches(runtime, ticket, requestSlot, generation)) {
    failWorkTicket(runtime, workTicket, abi::WorkTicketState::Failed);
    return false;
  }
  return completeWorkTicket(runtime, workTicket);
}

} // namespace nta::device
