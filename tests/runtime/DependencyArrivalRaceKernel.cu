#include "nta/RuntimeABI.h"

#define NTA_DEVICE_PHASE_KERNELS 0
#include "runtime/device/Acquire.cuh"

#include <cstdint>

extern "C" __global__ void nta_test_dependency_arrival_race(
    nta::abi::RuntimeView *runtime, std::uint32_t *observation) {
  using namespace nta;
  if (blockIdx.x != 0 || threadIdx.x >= 32) {
    return;
  }

  if (threadIdx.x == 0) {
    runtime->epoch = 1;
    runtime->completedCount = 0;
    runtime->failedCount = 0;
    *runtime->changedCount = 0;
    *runtime->changedOverflow = 0;
    *runtime->pendingCount = 0;

    runtime->objects[0] = {
        91,
        0,
        4096,
        0,
        7,
        static_cast<std::uint32_t>(abi::ObjectState::Issued),
        0,
        0,
        0,
        0,
        0,
    };
    runtime->workTickets[0] = {
        42,
        0,
        3,
        static_cast<std::uint32_t>(abi::WorkTicketState::Initializing),
        1,
        0,
        0,
        1,
        4096,
        2500,
        0,
        1,
    };
    runtime->reductionExpected[0] = 0;
    runtime->reductionCompleted[0] = 0;
    runtime->reductionFailed[0] = 0;
    runtime->dependencies[0] = {91, 0, 7};
    runtime->dependencyNext[0] = abi::InvalidIndex;
    runtime->dependencySatisfied[0] = 0;
    runtime->remainingDependencies[0] = 1;
    runtime->changedQueued[0] = 0;
    runtime->objectDependentHeads[0] = 0;
    __threadfence();
  }
  __syncthreads();

  if (threadIdx.x == 1) {
    atomicExch(&runtime->objects[0].state,
               static_cast<std::uint32_t>(abi::ObjectState::Ready));
    device::publishObjectTransition(runtime, 0, abi::ObjectState::Ready);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    runtime->pendingWorkTickets[0] = 0;
    *runtime->pendingCount = 1;
    atomicExch(&runtime->workTickets[0].state,
               static_cast<std::uint32_t>(abi::WorkTicketState::Pending));
    device::recordPendingWork(runtime, runtime->workTickets[0]);
    if (atomicAdd(&runtime->remainingDependencies[0], 0U) == 0U) {
      (void)device::enqueueChangedWork(runtime, 0);
    }
  }
  __syncthreads();

  // A duplicate observation after Pending must not append another queue item.
  if (threadIdx.x == 1) {
    device::publishObjectTransition(runtime, 0, abi::ObjectState::Ready);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    const abi::RequestProgress &progress = runtime->requestProgress[0];
    observation[0] = runtime->remainingDependencies[0];
    observation[1] = *runtime->changedCount;
    observation[2] = runtime->changedQueued[0];
    observation[3] = runtime->dependencySatisfied[0];
    observation[4] = runtime->workTickets[0].state;
    observation[5] = progress.expectedWork;
    observation[6] = progress.pendingWork;
    observation[7] = *runtime->changedOverflow;
  }
}

extern "C" __global__ void nta_test_request_reduction_groups(
    nta::abi::RuntimeView *runtime, std::uint32_t *observation) {
  using namespace nta;
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  runtime->epoch = 3;
  runtime->completedCount = 0;
  runtime->failedCount = 0;
  for (std::uint32_t group = 0; group < 2; ++group) {
    runtime->reductionExpected[group] = 0;
    runtime->reductionCompleted[group] = 0;
    runtime->reductionFailed[group] = 0;
  }
  runtime->workTickets[0] = {
      42, 0, 3, static_cast<std::uint32_t>(abi::WorkTicketState::New),
      0,  0, 0, 3, 0, 2500, 0, 2,
  };
  runtime->workTickets[1] = {
      42, 0, 3, static_cast<std::uint32_t>(abi::WorkTicketState::New),
      0,  1, 0, 3, 0, 2500, 0, 2,
  };
  runtime->workTickets[2] = {
      43, 1, 4, static_cast<std::uint32_t>(abi::WorkTicketState::New),
      0,  2, 0, 3, 0, 4000, 1, 1,
  };
  (void)device::completeWorkTicket(runtime, 0);
  (void)device::completeWorkTicket(runtime, 2);
  device::failWorkTicket(runtime, 1, abi::WorkTicketState::Failed);

  observation[0] = runtime->reductionExpected[0];
  observation[1] = runtime->reductionCompleted[0];
  observation[2] = runtime->reductionFailed[0];
  observation[3] = runtime->reductionExpected[1];
  observation[4] = runtime->reductionCompleted[1];
  observation[5] = runtime->reductionFailed[1];
  observation[6] = runtime->completedCount;
  observation[7] = runtime->failedCount;
}

extern "C" __global__ void nta_test_intent_priority_queue(
    nta::abi::RuntimeView *runtime, std::uint32_t *observation) {
  using namespace nta;
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  runtime->epoch = 2;
  constexpr std::uint32_t priorities[4] = {1, 7, 4, 7};
  for (std::uint32_t head = 0;
       head < abi::BackendCount * abi::UrgencyBucketCount; ++head) {
    runtime->intentQueueHeads[head] =
        device::taggedIntentHead(2, abi::InvalidIndex);
  }
  for (std::uint32_t slotIndex = 0; slotIndex < 4; ++slotIndex) {
    abi::IntentSlot &slot = runtime->intents[slotIndex];
    slot.sequence = slotIndex;
    slot.sourceKind = static_cast<std::uint32_t>(abi::SourceKind::HostStaged);
    slot.epoch = 2;
    slot.intent = {};
    slot.intent.priority = priorities[slotIndex];
    slot.intent.valid = 1;
    runtime->intentQueueEntries[slotIndex] = {};
    if (!device::queueIntent(runtime, slot, abi::SourceKind::HostStaged)) {
      observation[7] = 1;
      return;
    }
  }

  const std::uint32_t first =
      device::popIntent(runtime, abi::SourceKind::HostStaged);
  observation[0] = first;
  observation[1] =
      device::requeueIntent(runtime, first, abi::SourceKind::HostStaged)
          ? device::popIntent(runtime, abi::SourceKind::HostStaged)
          : abi::InvalidIndex;
  observation[2] = device::popIntent(runtime, abi::SourceKind::HostStaged);
  observation[3] = device::popIntent(runtime, abi::SourceKind::HostStaged);
  observation[4] = device::popIntent(runtime, abi::SourceKind::HostStaged);
  observation[5] = device::popIntent(runtime, abi::SourceKind::HostStaged);
}
