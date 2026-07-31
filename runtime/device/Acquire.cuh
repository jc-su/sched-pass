#pragma once

#include "nta/RuntimeABI.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

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
  if (continuation < runtime->continuationCapacity && threadIdx.x == 0) {
    atomicExch(&runtime->continuations[continuation].state,
               static_cast<std::uint32_t>(state));
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

  const auto sourceKind = static_cast<abi::SourceKind>(object.sourceKind);
  const auto state =
      static_cast<abi::ObjectState>(atomicAdd(&object.state, 0U));
  if (sourceKind == abi::SourceKind::Hbm ||
      sourceKind == abi::SourceKind::HostMapped) {
    return reinterpret_cast<std::byte *>(object.sourceAddress) + offset;
  }
  if ((sourceKind != abi::SourceKind::HostStaged &&
       sourceKind != abi::SourceKind::Nvme) ||
      object.stagingAddress == 0 ||
      (sourceKind == abi::SourceKind::HostStaged &&
       object.sourceAddress == 0) ||
      (sourceKind == abi::SourceKind::Nvme &&
       (runtime->nvme == nullptr || object.dmaPageListAddress == 0 ||
        object.dmaPageCount == 0))) {
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

  if (threadIdx.x == 0 &&
      atomicCAS(&object.state,
                static_cast<std::uint32_t>(abi::ObjectState::New),
                static_cast<std::uint32_t>(abi::ObjectState::Queued)) ==
          static_cast<std::uint32_t>(abi::ObjectState::New)) {
    const std::uint32_t ticket = atomicAdd(runtime->intentCount, 1U);
    if (ticket < runtime->intentCapacity) {
      abi::AcquireIntent &intent = runtime->intents[ticket];
      intent.objectId = objectId;
      intent.offset = offset;
      intent.bytes = bytes;
      intent.requestSlot = requestSlot;
      intent.generation = generation;
      intent.objectSlot = objectSlot;
      intent.objectVersion = objectVersion;
      intent.continuation = continuation;
      atomicAdd(reinterpret_cast<unsigned long long *>(&object.issueCount),
                1ULL);
      __threadfence();
      atomicExch(&intent.valid, 1U);
    } else {
      atomicSub(runtime->intentCount, 1U);
      atomicExch(&object.state,
                 static_cast<std::uint32_t>(abi::ObjectState::New));
    }
  }
  return nullptr;
}

extern "C" __global__ void nta_progress_nvme(nta::abi::RuntimeView *runtime,
                                             std::uint32_t issueBudget,
                                             std::uint32_t completionBudget) {
  using namespace nta;
  if (runtime == nullptr || runtime->nvme == nullptr || blockIdx.x != 0 ||
      threadIdx.x != 0) {
    return;
  }
  abi::NvmeQueueView &queue = *runtime->nvme;
  if (queue.active == 0 || queue.depth < 2 || queue.controllerPageSize == 0) {
    return;
  }

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
      const abi::NvmeCommandContext context = queue.contexts[commandId];
      valid = context.objectSlot < runtime->objectCapacity;
      if (valid) {
        abi::ObjectEntry &object = runtime->objects[context.objectSlot];
        valid = object.objectId == context.objectId &&
                object.version == context.objectVersion &&
                object.sourceKind ==
                    static_cast<std::uint32_t>(abi::SourceKind::Nvme);
        if (valid && (statusField >> 1U) == 0) {
          atomicExch(&object.state,
                     static_cast<std::uint32_t>(abi::ObjectState::Ready));
          if (context.continuation < runtime->continuationCapacity) {
            abi::Continuation &continuation =
                runtime->continuations[context.continuation];
            continuation.dependencyCount = 0;
            if (device::requestLive(runtime, context.requestSlot,
                                    context.generation)) {
              atomicExch(
                  &continuation.state,
                  static_cast<std::uint32_t>(abi::ContinuationState::Ready));
            } else {
              atomicExch(&continuation.state,
                         static_cast<std::uint32_t>(
                             abi::ContinuationState::Cancelled));
            }
          }
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

  const std::uint32_t intentCount = atomicAdd(runtime->intentCount, 0U);
  std::uint32_t issued = 0;
  while (issued < issueBudget && queue.outstanding + 1U < queue.depth) {
    std::uint32_t ticket = queue.intentCursor;
    abi::AcquireIntent *selected = nullptr;
    abi::ObjectEntry *object = nullptr;
    for (; ticket < intentCount && ticket < runtime->intentCapacity; ++ticket) {
      abi::AcquireIntent &candidate = runtime->intents[ticket];
      if (atomicAdd(&candidate.valid, 0U) == 0 ||
          candidate.objectSlot >= runtime->objectCapacity) {
        continue;
      }
      abi::ObjectEntry &candidateObject =
          runtime->objects[candidate.objectSlot];
      if (candidateObject.sourceKind ==
          static_cast<std::uint32_t>(abi::SourceKind::Nvme)) {
        selected = &candidate;
        object = &candidateObject;
        break;
      }
    }
    queue.intentCursor = ticket + (selected == nullptr ? 0U : 1U);
    if (selected == nullptr) {
      break;
    }

    const std::uint32_t expectedPages = static_cast<std::uint32_t>(
        (object->bytes + queue.controllerPageSize - 1U) /
        queue.controllerPageSize);
    const bool valid =
        object->objectId == selected->objectId &&
        object->version == selected->objectVersion && selected->offset == 0 &&
        selected->bytes == object->bytes &&
        object->dmaPageCount == expectedPages && object->bytes != 0 &&
        object->bytes % (1ULL << queue.lbaShift) == 0 &&
        object->sourceAddress % (1ULL << queue.lbaShift) == 0 &&
        object->dmaPageCount <=
            queue.controllerPageSize / sizeof(std::uint64_t);
    if (!valid) {
      atomicExch(&object->state,
                 static_cast<std::uint32_t>(abi::ObjectState::Failed));
      device::failContinuation(runtime, selected->continuation,
                               abi::ContinuationState::Failed);
      atomicExch(&selected->valid, 0U);
      ++queue.failed;
      queue.error = 0xfffffffeU;
      continue;
    }

    const std::uint32_t commandId = queue.sqTail;
    abi::NvmeSubmission &submission = queue.submissions[commandId];
    for (std::uint32_t dword = 0; dword < 16; ++dword) {
      submission.dword[dword] = 0;
    }

    const auto *dmaPages =
        reinterpret_cast<const std::uint64_t *>(object->dmaPageListAddress);
    const std::uint64_t firstPrp = dmaPages[0];
    std::uint64_t secondPrp = 0;
    if (object->dmaPageCount == 2) {
      secondPrp = dmaPages[1];
    } else if (object->dmaPageCount > 2) {
      auto *prpList = reinterpret_cast<std::uint64_t *>(
          reinterpret_cast<std::byte *>(queue.prpLists) +
          static_cast<std::uint64_t>(commandId) * queue.controllerPageSize);
      for (std::uint32_t page = 1; page < object->dmaPageCount; ++page) {
        prpList[page - 1U] = dmaPages[page];
      }
      secondPrp =
          queue.prpListDmaAddress +
          static_cast<std::uint64_t>(commandId) * queue.controllerPageSize;
    }

    const std::uint64_t lba = object->sourceAddress >> queue.lbaShift;
    const std::uint32_t lbaCount =
        static_cast<std::uint32_t>(object->bytes >> queue.lbaShift);
    submission.dword[0] = 0x02U | (commandId << 16U);
    submission.dword[1] = queue.namespaceId;
    submission.dword[6] = static_cast<std::uint32_t>(firstPrp);
    submission.dword[7] = static_cast<std::uint32_t>(firstPrp >> 32U);
    submission.dword[8] = static_cast<std::uint32_t>(secondPrp);
    submission.dword[9] = static_cast<std::uint32_t>(secondPrp >> 32U);
    submission.dword[10] = static_cast<std::uint32_t>(lba);
    submission.dword[11] = static_cast<std::uint32_t>(lba >> 32U);
    submission.dword[12] = lbaCount - 1U;

    queue.contexts[commandId] = {
        object->objectId,
        selected->objectSlot,
        selected->objectVersion,
        selected->requestSlot,
        selected->generation,
        selected->continuation,
        ticket,
    };
    atomicExch(&object->state,
               static_cast<std::uint32_t>(abi::ObjectState::Issued));
    atomicExch(&selected->valid, 0U);
    queue.sqTail++;
    if (queue.sqTail == queue.depth) {
      queue.sqTail = 0;
    }
    ++queue.outstanding;
    ++queue.submitted;
    ++issued;
  }
  if (issued != 0) {
    device::systemIoFence();
    *queue.sqDoorbell = queue.sqTail;
  }
}

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
      currentState == abi::ContinuationState::Failed) {
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
  record.dependencyCount = 1;
  record.logicalTile = continuation;
  atomicExch(&record.state,
             static_cast<std::uint32_t>(abi::ContinuationState::Pending));
}

extern "C" __global__ void
nta_progress_host_staging(nta::abi::RuntimeView *runtime) {
  using namespace nta;
  const std::uint32_t ticket = blockIdx.x;
  const std::uint32_t count = atomicAdd(runtime->intentCount, 0U);
  if (ticket >= count || ticket >= runtime->intentCapacity) {
    return;
  }

  abi::AcquireIntent &intent = runtime->intents[ticket];
  if (atomicAdd(&intent.valid, 0U) == 0 ||
      intent.objectSlot >= runtime->objectCapacity) {
    return;
  }
  abi::ObjectEntry &object = runtime->objects[intent.objectSlot];
  if (object.objectId != intent.objectId ||
      object.version != intent.objectVersion || intent.offset != 0 ||
      intent.bytes != object.bytes || intent.offset > object.bytes ||
      intent.bytes > object.bytes - intent.offset ||
      object.sourceKind !=
          static_cast<std::uint32_t>(abi::SourceKind::HostStaged)) {
    if (threadIdx.x == 0) {
      atomicExch(&object.state,
                 static_cast<std::uint32_t>(abi::ObjectState::Failed));
      device::failContinuation(runtime, intent.continuation,
                               abi::ContinuationState::Failed);
    }
    return;
  }

  auto *source =
      reinterpret_cast<const std::byte *>(object.sourceAddress) + intent.offset;
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
    if (intent.continuation < runtime->continuationCapacity) {
      abi::Continuation &continuation =
          runtime->continuations[intent.continuation];
      if (device::requestLive(runtime, intent.requestSlot, intent.generation)) {
        continuation.dependencyCount = 0;
        atomicExch(&continuation.state,
                   static_cast<std::uint32_t>(abi::ContinuationState::Ready));
      } else {
        atomicExch(&continuation.state, static_cast<std::uint32_t>(
                                            abi::ContinuationState::Cancelled));
      }
    }
    atomicExch(&intent.valid, 0U);
  }
}

extern "C" __global__ void nta_reset_epoch(nta::abi::RuntimeView *runtime,
                                           std::uint32_t objectCount,
                                           std::uint32_t continuationCount) {
  using namespace nta;
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index == 0) {
    *runtime->intentCount = 0;
  }
  if (index < objectCount && index < runtime->objectCapacity) {
    abi::ObjectEntry &object = runtime->objects[index];
    object.issueCount = 0;
    if (object.sourceKind ==
        static_cast<std::uint32_t>(abi::SourceKind::HostStaged)) {
      object.state = static_cast<std::uint32_t>(abi::ObjectState::New);
    } else if (object.sourceKind ==
                   static_cast<std::uint32_t>(abi::SourceKind::Nvme) &&
               (runtime->nvme == nullptr || runtime->nvme->outstanding == 0)) {
      object.state = static_cast<std::uint32_t>(abi::ObjectState::New);
    }
  }
  if (index < continuationCount && index < runtime->continuationCapacity) {
    abi::Continuation &continuation = runtime->continuations[index];
    continuation.state =
        static_cast<std::uint32_t>(abi::ContinuationState::New);
    continuation.dependencyCount = 0;
  }
  if (index == 0 && runtime->nvme != nullptr) {
    runtime->nvme->intentCursor = 0;
  }
}
