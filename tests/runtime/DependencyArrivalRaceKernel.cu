#include "nta/RuntimeABI.h"

#define NTA_DEVICE_PHASE_KERNELS 0
#define NTA_TEST_INTENT_QUEUE_INTERNALS 1
#include "runtime/device/Acquire.cuh"

#include <cstdint>

namespace {

__device__ void resetIntentQueue(nta::abi::RuntimeView *runtime,
                                 nta::abi::SourceKind source,
                                 std::uint32_t epoch) {
  const std::uint32_t sourceIndex = static_cast<std::uint32_t>(source);
  runtime->epoch = epoch;
  runtime->intentQueueControls[sourceIndex] = {};
  nta::abi::IntentQueueNode *heap =
      runtime->intentQueueHeap +
      static_cast<std::uint64_t>(sourceIndex) * runtime->intentCapacity;
  for (std::uint32_t index = 0; index < runtime->intentCapacity; ++index) {
    runtime->intentQueueEntries[index] = {};
    runtime->intentQueueEntries[index].heapIndex = nta::abi::InvalidIndex;
    heap[index] = {0, nta::abi::InvalidIndex, 0};
    runtime->intents[index].intent.valid = 0;
  }
}

__device__ void
prepareIntent(nta::abi::RuntimeView *runtime, std::uint32_t slotIndex,
              std::uint64_t intentSequence, std::uint64_t deadlineClock,
              std::uint32_t priority, std::uint32_t requestSlot = 0,
              std::uint32_t generation = 1,
              nta::abi::SourceKind source = nta::abi::SourceKind::HostStaged) {
  nta::abi::IntentSlot &slot = runtime->intents[slotIndex];
  slot.sequence = intentSequence;
  slot.sourceKind = static_cast<std::uint32_t>(source);
  slot.epoch = runtime->epoch;
  slot.intent = {};
  slot.intent.bytes = 4096;
  slot.intent.requestSlot = requestSlot;
  slot.intent.generation = generation;
  slot.intent.objectSlot = slotIndex;
  slot.intent.priority = priority;
  slot.intent.deadlineClock = deadlineClock;
  slot.intent.valid = 1;
}

} // namespace

extern "C" __global__ void
nta_test_dependency_arrival_race(nta::abi::RuntimeView *runtime,
                                 std::uint32_t *observation) {
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
        91, 0, 4096, 0, 7, static_cast<std::uint32_t>(abi::ObjectState::Issued),
        0,  0, 0,    0, 0,
    };
    runtime->workTickets[0] = {
        42,   0,
        3,    static_cast<std::uint32_t>(abi::WorkTicketState::Initializing),
        1,    0,
        0,    1,
        4096, 2500,
        0,    1,
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

extern "C" __global__ void
nta_test_request_reduction_groups(nta::abi::RuntimeView *runtime,
                                  std::uint32_t *observation) {
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
      42, 0,    3, static_cast<std::uint32_t>(abi::WorkTicketState::New),
      0,  0,    0, 3,
      0,  2500, 0, 2,
  };
  runtime->workTickets[1] = {
      42, 0,    3, static_cast<std::uint32_t>(abi::WorkTicketState::New),
      0,  1,    0, 3,
      0,  2500, 0, 2,
  };
  runtime->workTickets[2] = {
      43, 1,    4, static_cast<std::uint32_t>(abi::WorkTicketState::New),
      0,  2,    0, 3,
      0,  4000, 1, 1,
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

extern "C" __global__ void
nta_test_cancelled_intent_credit_release(nta::abi::RuntimeView *runtime,
                                         std::uint32_t *observation) {
  using namespace nta;
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  constexpr std::uint64_t bytes = 4096;
  constexpr abi::SourceKind source = abi::SourceKind::HostStaged;
  abi::RequestContext &request = runtime->requests[0];
  request.generation = 3;
  request.tenantId = 0;
  request.cancelled = 0;
  request.maxOutstandingBytes = bytes;
  request.outstandingBytes = 0;
  runtime->tenants[0].maxOutstandingBytes = bytes;
  runtime->tenants[0].outstandingBytes = 0;
  abi::BackendView &backend =
      runtime->backends[static_cast<std::uint32_t>(source)];
  backend = {0,     0,
             1,     0,
             bytes, static_cast<std::uint32_t>(source),
             1,     static_cast<std::uint32_t>(source),
             0,     0};
  abi::IntentSlot &slot = runtime->intents[0];
  slot = {};
  slot.intent.bytes = bytes;
  slot.intent.requestSlot = 0;
  slot.intent.generation = 3;
  slot.intent.tenantId = 0;
  std::uint64_t requestBytes = 0;
  std::uint64_t backendBytes = 0;
  const bool admitted = device::reserveIntentCredits(
      runtime, slot.intent, source, requestBytes, backendBytes);
  device::recordIntentCredits(slot, requestBytes, backendBytes);
  request.cancelled = 1;
  device::releaseRecordedIntentCredits(runtime, slot, slot.intent, source);
  observation[0] = admitted ? 1U : 0U;
  observation[1] = static_cast<std::uint32_t>(request.outstandingBytes);
  observation[2] =
      static_cast<std::uint32_t>(runtime->tenants[0].outstandingBytes);
  observation[3] = static_cast<std::uint32_t>(backend.outstandingBytes);
  observation[4] = static_cast<std::uint32_t>(slot.chargedRequestBytes);
  observation[5] = static_cast<std::uint32_t>(slot.chargedBackendBytes);
}

extern "C" __global__ void
nta_test_explicit_indexed_claim_credit_lifetime(nta::abi::RuntimeView *runtime,
                                                std::uint32_t *observation) {
  using namespace nta;
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  constexpr std::uint64_t bytes = 4096;
  constexpr abi::SourceKind source = abi::SourceKind::HostStaged;
  abi::RequestContext &request = runtime->requests[0];
  request.generation = 3;
  request.tenantId = 0;
  request.cancelled = 0;
  request.maxOutstandingBytes = bytes;
  request.outstandingBytes = 0;
  runtime->tenants[0].maxOutstandingBytes = bytes;
  runtime->tenants[0].outstandingBytes = 0;
  abi::BackendView &backend =
      runtime->backends[static_cast<std::uint32_t>(source)];
  backend = {0,     0,
             1,     0,
             bytes, static_cast<std::uint32_t>(source),
             1,     static_cast<std::uint32_t>(source),
             0,     0};
  abi::IntentSlot &slot = runtime->intents[0];
  slot = {};
  slot.intent.bytes = bytes;
  slot.intent.requestSlot = 0;
  slot.intent.generation = 3;
  slot.intent.tenantId = 0;
  slot.intent.valid = 1;
  abi::ObjectEntry &object = runtime->objects[0];
  object = {};
  object.stagingAddress = 1;
  object.state = static_cast<std::uint32_t>(abi::ObjectState::Queued);
  // The explicit indexed-range claim phase owns reservation and Issued
  // publication.  Index validation above is deliberately irrelevant to this
  // ownership transition: EDF progress may consume the same validated intent
  // through its queue instead.
  std::uint64_t requestBytes = 0;
  std::uint64_t backendBytes = 0;
  bool admitted = device::reserveIntentCredits(runtime, slot.intent, source,
                                               requestBytes, backendBytes);
  if (admitted && !device::claimIntent(slot)) {
    device::releaseIntentCredits(runtime, slot.intent, source, requestBytes,
                                 backendBytes);
    admitted = false;
  }
  if (admitted) {
    device::recordIntentCredits(slot, requestBytes, backendBytes);
    __threadfence();
    atomicExch(&object.state,
               static_cast<std::uint32_t>(abi::ObjectState::Issued));
  }
  observation[0] = admitted ? 1U : 0U;
  observation[1] = object.state;
  observation[2] = slot.intent.valid;
  observation[3] = static_cast<std::uint32_t>(slot.chargedRequestBytes);
  observation[4] = static_cast<std::uint32_t>(slot.chargedBackendBytes);
  observation[5] = static_cast<std::uint32_t>(request.outstandingBytes);
  observation[6] =
      static_cast<std::uint32_t>(runtime->tenants[0].outstandingBytes);
  observation[7] = static_cast<std::uint32_t>(backend.outstandingBytes);

  device::releaseRecordedIntentCredits(runtime, slot, slot.intent, source);
  observation[8] = static_cast<std::uint32_t>(slot.chargedRequestBytes);
  observation[9] = static_cast<std::uint32_t>(slot.chargedBackendBytes);
  observation[10] = static_cast<std::uint32_t>(request.outstandingBytes);
  observation[11] =
      static_cast<std::uint32_t>(runtime->tenants[0].outstandingBytes);
  observation[12] = static_cast<std::uint32_t>(backend.outstandingBytes);
}

extern "C" __global__ void
nta_test_indexed_publication_topology(nta::abi::RuntimeView *runtime,
                                      std::uint32_t *observation) {
  using namespace nta;
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  runtime->epoch = 9;
  runtime->completedCount = 0;
  runtime->failedCount = 0;
  *runtime->readyCount = 0;
  *runtime->pendingCount = 0;
  *runtime->changedCount = 0;
  *runtime->changedOverflow = 0;
  runtime->requests[0].generation = 3;
  runtime->requests[0].cancelled = 0;
  runtime->requests[1].generation = 4;
  runtime->requests[1].cancelled = 0;
  runtime->requestProgress[0] = {};
  runtime->reductionExpected[0] = 0;
  runtime->reductionCompleted[0] = 0;
  runtime->reductionFailed[0] = 0;
  runtime->objects[0] = {
      101, 0, 4096, 0, 5, static_cast<std::uint32_t>(abi::ObjectState::Ready),
      0,   0, 0,    0, 0,
  };
  runtime->workTickets[0] = {
      42,   0,    3, static_cast<std::uint32_t>(abi::WorkTicketState::Pending),
      1,    0,    0, 9,
      4096, 2500, 0, 1,
  };
  runtime->dependencies[0] = {101, 0, 5};
  runtime->dependencyNext[0] = abi::InvalidIndex;
  runtime->dependencySatisfied[0] = 0;
  runtime->remainingDependencies[0] = 1;
  runtime->objectDependentHeads[0] = 0;
  device::recordPendingWork(runtime, runtime->workTickets[0]);
  const bool privatePublished = device::publishPrivateIndexedObject(runtime, 0);
  observation[0] = privatePublished ? 1U : 0U;
  observation[1] = runtime->dependencySatisfied[0];
  observation[2] = runtime->remainingDependencies[0];
  observation[3] = *runtime->readyCount;
  observation[4] = runtime->workTickets[0].state;
  observation[5] = *runtime->changedOverflow;

  // Two valid reverse edges are a shared object. The topology-aware helper
  // must not serialize that fanout or satisfy only its list head; it delegates
  // the complete set to the existing parallel full-scan publication kernel.
  *runtime->readyCount = 0;
  *runtime->changedOverflow = 0;
  runtime->objects[1] = {
      102, 0, 4096, 0, 6, static_cast<std::uint32_t>(abi::ObjectState::Ready),
      0,   0, 0,    0, 0,
  };
  for (std::uint32_t ticket = 0; ticket < 2; ++ticket) {
    runtime->workTickets[ticket] = {
        42 + ticket, ticket,
        3 + ticket,  static_cast<std::uint32_t>(abi::WorkTicketState::Pending),
        1,           ticket,
        0,           9,
        4096,        2500,
        ticket,      1,
    };
    runtime->dependencies[ticket] = {102, 1, 6};
    runtime->dependencySatisfied[ticket] = 0;
    runtime->remainingDependencies[ticket] = 1;
  }
  runtime->dependencyNext[0] = 1;
  runtime->dependencyNext[1] = abi::InvalidIndex;
  runtime->objectDependentHeads[1] = 0;
  const bool sharedPublished = device::publishPrivateIndexedObject(runtime, 1);
  observation[6] = sharedPublished ? 1U : 0U;
  observation[7] = runtime->dependencySatisfied[0];
  observation[8] = runtime->dependencySatisfied[1];
  observation[9] = *runtime->readyCount;
  observation[10] = *runtime->changedOverflow;
}

extern "C" __global__ void
nta_test_intent_deadline_queue(nta::abi::RuntimeView *runtime,
                               std::uint32_t *observation) {
  using namespace nta;
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  constexpr abi::SourceKind source = abi::SourceKind::HostStaged;
  std::uint32_t setupFailure = 0;

  // Absolute deadlines dominate both caller priority and best-effort work.
  // Slots 1 and 3 share the same deadline, so caller priority breaks that tie.
  resetIntentQueue(runtime, source, 2);
  prepareIntent(runtime, 0, 10, 0, UINT32_MAX);
  prepareIntent(runtime, 1, 11, 1'000, 1);
  prepareIntent(runtime, 2, 12, 900, 0);
  prepareIntent(runtime, 3, 13, 1'000, 9);
  for (std::uint32_t slotIndex = 0; slotIndex < 4; ++slotIndex) {
    if (!device::queueIntent(runtime, runtime->intents[slotIndex], source)) {
      setupFailure |= 1U;
    }
  }
  for (std::uint32_t index = 0; index < 4; ++index) {
    observation[index] = device::popIntent(runtime, source);
  }
  observation[4] = device::popIntent(runtime, source);

  // Equal key insertion is FIFO, and a credit requeue preserves the original
  // stable sequence rather than turning the queue back into a stack.
  resetIntentQueue(runtime, source, 3);
  prepareIntent(runtime, 0, 20, 2'000, 7);
  prepareIntent(runtime, 1, 21, 2'000, 7);
  setupFailure |=
      device::queueIntent(runtime, runtime->intents[0], source) ? 0U : 2U;
  setupFailure |=
      device::queueIntent(runtime, runtime->intents[1], source) ? 0U : 2U;
  const std::uint32_t first = device::popIntent(runtime, source);
  observation[5] = first;
  observation[6] = first != abi::InvalidIndex &&
                           device::requeueIntent(runtime, first, source)
                       ? device::popIntent(runtime, source)
                       : abi::InvalidIndex;
  observation[7] = device::popIntent(runtime, source);
  observation[8] = device::popIntent(runtime, source);

  // Critical service is only a tie breaker. At an equal timed deadline the
  // longer critical path has less laxity; in best-effort the shorter known
  // path minimizes mean completion time.
  runtime->backends[static_cast<std::uint32_t>(abi::SourceKind::HostStaged)] = {
      0,          0,
      0,          0,
      UINT64_MAX, static_cast<std::uint32_t>(abi::SourceKind::HostStaged),
      1,          static_cast<std::uint32_t>(abi::SourceKind::HostStaged),
      0,          0,
  };
  runtime->requestProgress[0] = {
      42, 5, 1, 1, 0, 0, 0, 0, 4, 4096, 0, 0, 600'000, 600'000, 0,
  };
  runtime->requestProgress[1] = {
      43, 6, 1, 1, 0, 0, 0, 0, 4, 4096, 0, 0, 10'000, 10'000, 0,
  };
  resetIntentQueue(runtime, source, 4);
  for (std::uint32_t slotIndex = 0; slotIndex < 2; ++slotIndex) {
    prepareIntent(runtime, slotIndex, 30 + slotIndex, 3'000, 4, slotIndex,
                  5 + slotIndex);
    setupFailure |=
        device::queueIntent(runtime, runtime->intents[slotIndex], source) ? 0U
                                                                          : 4U;
  }
  observation[9] = device::popIntent(runtime, source);

  resetIntentQueue(runtime, source, 4);
  for (std::uint32_t slotIndex = 0; slotIndex < 2; ++slotIndex) {
    prepareIntent(runtime, slotIndex, 40 + slotIndex, 0, 4, slotIndex,
                  5 + slotIndex);
    setupFailure |=
        device::queueIntent(runtime, runtime->intents[slotIndex], source) ? 0U
                                                                          : 8U;
  }
  observation[10] = device::popIntent(runtime, source);

  // A node whose slot generation changed is discarded, even if its obsolete
  // deadline would otherwise win.
  resetIntentQueue(runtime, source, 5);
  prepareIntent(runtime, 0, 50, 100, 1);
  setupFailure |=
      device::queueIntent(runtime, runtime->intents[0], source) ? 0U : 16U;
  runtime->intents[0].sequence = 51;
  prepareIntent(runtime, 1, 60, 200, 1);
  setupFailure |=
      device::queueIntent(runtime, runtime->intents[1], source) ? 0U : 16U;
  observation[11] = device::popIntent(runtime, source);
  observation[12] = device::popIntent(runtime, source);
  observation[13] = runtime->intentQueueEntries[0].state;

  // Reusing one physical slot while its cancelled node is still in the heap
  // must select the new generation once and never resurrect the old one.
  resetIntentQueue(runtime, source, 6);
  prepareIntent(runtime, 0, 70, 100, 1);
  setupFailure |=
      device::queueIntent(runtime, runtime->intents[0], source) ? 0U : 32U;
  atomicExch(&runtime->intentQueueEntries[0].state,
             static_cast<std::uint32_t>(abi::IntentQueueState::Free));
  runtime->intents[0].intent.valid = 0;
  prepareIntent(runtime, 0, 71, 200, 1);
  setupFailure |=
      device::queueIntent(runtime, runtime->intents[0], source) ? 0U : 32U;
  observation[14] = device::popIntent(runtime, source);
  observation[15] = device::popIntent(runtime, source);
  observation[16] =
      runtime->intentQueueControls[static_cast<std::uint32_t>(source)].size;
  observation[17] = setupFailure;
}

extern "C" __global__ void
nta_test_intent_queue_concurrency(nta::abi::RuntimeView *runtime,
                                  std::uint32_t *observation) {
  using namespace nta;
  if (blockIdx.x != 0 || threadIdx.x >= 32) {
    return;
  }
  constexpr abi::SourceKind source = abi::SourceKind::HostStaged;
  constexpr std::uint64_t deadlines[4] = {400, 100, 300, 200};
  if (threadIdx.x == 0) {
    resetIntentQueue(runtime, source, 7);
    for (std::uint32_t slot = 0; slot < 4; ++slot) {
      prepareIntent(runtime, slot, 80 + slot, deadlines[slot], 1);
    }
  }
  __syncthreads();

  if (threadIdx.x < 4) {
    observation[threadIdx.x] =
        device::queueIntent(runtime, runtime->intents[threadIdx.x], source)
            ? 1U
            : 0U;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    observation[4] =
        runtime->intentQueueControls[static_cast<std::uint32_t>(source)].size;
  }

  std::uint32_t selected = abi::InvalidIndex;
  if (threadIdx.x < 4) {
    selected = device::popIntent(runtime, source);
    observation[5 + threadIdx.x] = selected;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const abi::IntentQueueControl &control =
        runtime->intentQueueControls[static_cast<std::uint32_t>(source)];
    observation[9] = control.size;
    observation[10] = control.lock;
    observation[11] = 0;
  }
  __syncthreads();

  if (threadIdx.x < 4 && selected != abi::InvalidIndex &&
      device::requeueIntent(runtime, selected, source)) {
    atomicAdd(&observation[11], 1U);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (std::uint32_t index = 0; index < 4; ++index) {
      observation[12 + index] = device::popIntent(runtime, source);
    }
    const abi::IntentQueueControl &control =
        runtime->intentQueueControls[static_cast<std::uint32_t>(source)];
    observation[16] = control.size;
    observation[17] = control.lock;
  }
}

extern "C" __global__ void
nta_test_constrained_edf_dispatch(nta::abi::RuntimeView *runtime,
                                  std::uint32_t *observation) {
  using namespace nta;
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  constexpr abi::SourceKind source = abi::SourceKind::HostStaged;
  constexpr std::uint32_t sourceIndex = static_cast<std::uint32_t>(source);
  constexpr std::uint64_t bytes = 4096;

  resetIntentQueue(runtime, source, 8);
  runtime->intentPool->active = 2;
  runtime->intentPool->enqueued = 2;
  runtime->intentPool->consumed = 0;
  runtime->backends[sourceIndex] = {
      0, 0, 1'000'000'000, 0, 4 * bytes, sourceIndex, 1, sourceIndex, 0, 2,
  };
  runtime->tenants[0] = {4 * bytes, 0};
  runtime->requests[0] = {100, 100, bytes, bytes, 1, 0, 0, 0};
  runtime->requests[1] = {101, 200, bytes, 0, 1, 0, 0, 0};
  prepareIntent(runtime, 0, 90, 100, 0, 0, 1);
  prepareIntent(runtime, 1, 91, 200, 0, 1, 1);
  observation[0] =
      device::queueIntent(runtime, runtime->intents[0], source) ? 1U : 0U;
  observation[1] =
      device::queueIntent(runtime, runtime->intents[1], source) ? 1U : 0U;

  // Slot 0 has the earlier deadline but its request window is occupied. EDF
  // over the feasible set must dispatch slot 1 without removing/requeueing 0.
  const device::AdmittedIntent first =
      device::claimAdmissibleIntent(runtime, source);
  observation[2] = first.slotIndex;
  observation[3] = first.admitted ? 1U : 0U;
  observation[4] = runtime->requests[1].outstandingBytes;
  observation[5] = runtime->tenants[0].outstandingBytes;
  observation[6] = runtime->backends[sourceIndex].outstandingBytes;
  if (first.slotIndex != abi::InvalidIndex) {
    abi::IntentSlot &slot = runtime->intents[first.slotIndex];
    device::releaseIntentCredits(runtime, slot.intent, source,
                                 first.requestBytes, first.backendBytes);
    device::consumeIntent(runtime, slot);
  }

  // Simulate completion of the operation occupying request 0's one-page
  // window. The previously blocked root must now be selected exactly once.
  runtime->requests[0].outstandingBytes = 0;
  const device::AdmittedIntent second =
      device::claimAdmissibleIntent(runtime, source);
  observation[7] = second.slotIndex;
  observation[8] = second.admitted ? 1U : 0U;
  observation[9] = runtime->requests[0].outstandingBytes;
  if (second.slotIndex != abi::InvalidIndex) {
    abi::IntentSlot &slot = runtime->intents[second.slotIndex];
    device::releaseIntentCredits(runtime, slot.intent, source,
                                 second.requestBytes, second.backendBytes);
    device::consumeIntent(runtime, slot);
  }
  observation[10] = runtime->intentQueueControls[sourceIndex].size;
  observation[11] = runtime->intentPool->active;
  observation[12] = static_cast<std::uint32_t>(
      runtime->backends[sourceIndex].pendingAcquisitions);
  observation[13] = static_cast<std::uint32_t>(
      runtime->backends[sourceIndex].outstandingBytes);
  observation[14] =
      static_cast<std::uint32_t>(runtime->tenants[0].outstandingBytes);
}

extern "C" __global__ void
nta_test_ordered_intent_window_validation(nta::abi::RuntimeView *runtime,
                                          std::uint32_t *observation) {
  using namespace nta;
  if (blockIdx.x != 0 || blockDim.x != device::OrderedIntentValidationThreads) {
    return;
  }
  constexpr abi::SourceKind source = abi::SourceKind::Nvme;
  constexpr std::uint32_t sourceIndex = static_cast<std::uint32_t>(source);
  constexpr std::uint64_t bytes = 4096;
  __shared__ device::OrderedIntentValidationScratch scratch;

  if (threadIdx.x == 0) {
    resetIntentQueue(runtime, source, 9);
    runtime->intentPool->active = 3;
    runtime->intentPool->enqueued = 3;
    runtime->intentPool->consumed = 0;
    runtime->backends[sourceIndex] = {
        0, 0, 1'000'000'000, 0, 8 * bytes, sourceIndex, 1, sourceIndex, 0, 3,
    };
    runtime->tenants[0] = {8 * bytes, 0};
    runtime->requests[0] = {100, 0, 8 * bytes, 0, 1, 0, 0, 0};
    prepareIntent(runtime, 0, 100, 100, 1, 0, 1, source);
    prepareIntent(runtime, 1, 101, 200, 1, 0, 1, source);
    prepareIntent(runtime, 2, 102, 300, 1, 0, 1, source);
  }
  __syncthreads();
  const bool ordered =
      device::validateOrderedIntentWindow(runtime, source, 0, 3, scratch);
  abi::IntentQueueControl &control = runtime->intentQueueControls[sourceIndex];
  if (threadIdx.x == 0) {
    observation[0] = ordered ? 1U : 0U;
    observation[1] = control.size;
    observation[2] = control.reserved[0] == device::OrderedIntentWindowMagic;
    observation[3] =
        control.reserved[1] == device::orderedIntentWindowGeometry(0, 3);
    const device::AdmittedIntent first = device::claimOrderedAdmissibleIntent(
        runtime, source, 0, 3, control.size);
    observation[4] = first.slotIndex;
    observation[5] = control.size;
    if (first.slotIndex != abi::InvalidIndex) {
      abi::IntentSlot &slot = runtime->intents[first.slotIndex];
      device::releaseIntentCredits(runtime, slot.intent, source,
                                   first.requestBytes, first.backendBytes);
      device::consumeIntent(runtime, slot);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    resetIntentQueue(runtime, source, 10);
    prepareIntent(runtime, 0, 110, 300, 1, 0, 1, source);
    prepareIntent(runtime, 1, 111, 100, 1, 0, 1, source);
    prepareIntent(runtime, 2, 112, 200, 1, 0, 1, source);
  }
  __syncthreads();
  const bool deadlineOrdered =
      device::validateOrderedIntentWindow(runtime, source, 0, 3, scratch);
  if (threadIdx.x == 0) {
    observation[6] = deadlineOrdered ? 1U : 0U;
    observation[7] = control.reserved[0] == 0;
    observation[8] = control.reserved[1] == 0;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    resetIntentQueue(runtime, source, 11);
    prepareIntent(runtime, 0, 120, 100, 1, 0, 1, source);
    prepareIntent(runtime, 1, 121, 100, 2, 0, 1, source);
  }
  __syncthreads();
  const bool priorityOrdered =
      device::validateOrderedIntentWindow(runtime, source, 0, 2, scratch);
  if (threadIdx.x == 0) {
    observation[9] = priorityOrdered ? 1U : 0U;
    observation[10] = control.reserved[0] == 0;
  }
}

extern "C" __global__ void nta_test_ordered_intent_window_validation_chunks(
    nta::abi::RuntimeView *runtime, std::uint32_t *observation) {
  using namespace nta;
  constexpr std::uint32_t count =
      device::OrderedIntentValidationThreads * 2U + 1U;
  constexpr abi::SourceKind source = abi::SourceKind::Nvme;
  constexpr std::uint32_t sourceIndex = static_cast<std::uint32_t>(source);
  if (blockIdx.x != 0 || blockDim.x != device::OrderedIntentValidationThreads ||
      runtime == nullptr || runtime->intentCapacity < count) {
    return;
  }
  __shared__ device::OrderedIntentValidationScratch scratch;

  if (threadIdx.x == 0) {
    resetIntentQueue(runtime, source, 12);
    runtime->intentPool->active = count;
    runtime->intentPool->enqueued = count;
    runtime->intentPool->consumed = 0;
    runtime->backends[sourceIndex] = {
        0, 0, 1'000'000'000, 0, 8ULL * 4096 * count, sourceIndex,
        1, sourceIndex, 0, count,
    };
    runtime->tenants[0] = {8ULL * 4096 * count, 0};
    runtime->requests[0] = {100, 0, 8ULL * 4096 * count, 0, 1, 0, 0, 0};
  }
  __syncthreads();
  for (std::uint32_t slot = threadIdx.x; slot < count; slot += blockDim.x) {
    prepareIntent(runtime, slot, 1'000 + slot, 10'000 + slot, 1, 0, 1,
                  source);
  }
  __syncthreads();
  const bool ordered =
      device::validateOrderedIntentWindow(runtime, source, 0, count, scratch);
  abi::IntentQueueControl &control = runtime->intentQueueControls[sourceIndex];
  if (threadIdx.x == 0) {
    observation[0] = ordered ? 1U : 0U;
    observation[1] = control.reserved[0] == device::OrderedIntentWindowMagic;
    observation[2] =
        control.reserved[1] == device::orderedIntentWindowGeometry(0, count);
    observation[3] = control.size;
    resetIntentQueue(runtime, source, 13);
  }
  __syncthreads();
  for (std::uint32_t slot = threadIdx.x; slot < count; slot += blockDim.x) {
    const std::uint64_t deadline =
        slot == device::OrderedIntentValidationThreads ? 1U : 20'000 + slot;
    prepareIntent(runtime, slot, 2'000 + slot, deadline, 1, 0, 1, source);
  }
  __syncthreads();
  const bool crossChunkOrdered =
      device::validateOrderedIntentWindow(runtime, source, 0, count, scratch);
  if (threadIdx.x == 0) {
    observation[4] = crossChunkOrdered ? 1U : 0U;
    observation[5] = control.reserved[0] == 0;
    observation[6] = control.reserved[1] == 0;
  }
}

extern "C" __global__ void
nta_test_request_progress_fail_closed(nta::abi::RuntimeView *runtime,
                                      std::uint32_t *observation) {
  using namespace nta;
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  runtime->epoch = 5;
  runtime->failedCount = 0;
  const std::uint32_t stickyBefore = runtime->stickyFailedCount;
  runtime->reductionExpected[0] = 0;
  runtime->reductionCompleted[0] = 0;
  runtime->reductionFailed[0] = 0;
  runtime->requestProgress[0] = {
      42, 3, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0,
  };
  runtime->workTickets[0] = {
      42,   0,    3, static_cast<std::uint32_t>(abi::WorkTicketState::Pending),
      1,    0,    0, 5,
      4096, 2500, 0, 1,
  };

  device::recordPendingWork(runtime, runtime->workTickets[0]);
  observation[0] = static_cast<std::uint32_t>(
      runtime->requestProgress[0].droppedAttributions);
  observation[1] = runtime->requestProgress[0].expectedWork;

  runtime->workTickets[0].generation = 4;
  device::recordPendingWork(runtime, runtime->workTickets[0]);
  observation[2] = static_cast<std::uint32_t>(
      runtime->requestProgress[0].droppedAttributions);

  runtime->workTickets[0].generation = 3;
  runtime->requestProgress[0].epoch = 5;
  device::recordRunnableWork(runtime, runtime->workTickets[0]);
  observation[3] = static_cast<std::uint32_t>(
      runtime->requestProgress[0].droppedAttributions);
  observation[4] = runtime->failedCount;
  observation[5] = runtime->stickyFailedCount - stickyBefore;
  observation[6] = runtime->requestProgress[0].pendingWork;
  observation[7] =
      static_cast<std::uint32_t>(runtime->requestProgress[0].unavailableBytes);
  observation[8] =
      static_cast<std::uint32_t>(runtime->requestProgress[0].pendingComputeNs);
}
