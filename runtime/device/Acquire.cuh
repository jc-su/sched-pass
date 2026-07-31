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
  if (sourceKind != abi::SourceKind::HostStaged || object.stagingAddress == 0 ||
      object.sourceAddress == 0) {
    device::failContinuation(runtime, continuation,
                             abi::ContinuationState::Failed);
    return nullptr;
  }
  // ABI v1 directory entries are acquisition tiles. A staged transfer owns
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
    }
  }
  if (index < continuationCount && index < runtime->continuationCapacity) {
    abi::Continuation &continuation = runtime->continuations[index];
    continuation.state =
        static_cast<std::uint32_t>(abi::ContinuationState::New);
    continuation.dependencyCount = 0;
  }
}
