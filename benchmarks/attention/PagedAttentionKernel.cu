#include "benchmarks/attention/PagedAttentionTypes.h"
#include "nta/DeviceAPI.cuh"
#include "runtime/device/Acquire.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda/barrier>

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace {

using nta::benchmark::AttentionHeadDimension;
using nta::benchmark::AttentionPageTokens;
using nta::benchmark::AttentionRequest;
using nta::benchmark::AttentionTilePartial;
using nta::benchmark::AttentionTileTask;

__device__ __forceinline__ float blockSum128(float value, float *scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (std::uint32_t width = AttentionHeadDimension / 2; width != 0;
       width /= 2) {
    if (threadIdx.x < width) {
      scratch[threadIdx.x] += scratch[threadIdx.x + width];
    }
    __syncthreads();
  }
  return scratch[0];
}

__device__ __forceinline__ std::uint32_t
popReadyContinuation(nta::abi::RuntimeView *runtime) {
  __shared__ std::uint32_t selected;
  if (threadIdx.x == 0) {
    selected = nta::abi::InvalidIndex;
    for (;;) {
      const std::uint32_t head = atomicAdd(runtime->readyHead, 0U);
      const std::uint32_t count = atomicAdd(runtime->readyCount, 0U);
      if (head >= count) {
        break;
      }
      if (atomicCAS(runtime->readyHead, head, head + 1U) == head) {
        selected = runtime->readyContinuations[head];
        break;
      }
    }
  }
  __syncthreads();
  return selected;
}

__device__ __forceinline__ void
computeAttentionTile(nta::abi::RuntimeView *runtime,
                     const AttentionTileTask &task,
                     std::uint32_t taskIndex, const __half *page,
                     const __half *queries, AttentionTilePartial *partials) {
  nta::abi::Continuation &continuation =
      runtime->continuations[task.continuation];
  const __half *keys = page;
  const __half *values =
      page + AttentionPageTokens * AttentionHeadDimension;
  const __half *query =
      queries + task.requestIndex * AttentionHeadDimension;
  AttentionTilePartial &partial = partials[taskIndex];

  __shared__ float reduction[AttentionHeadDimension];
  __shared__ float logits[AttentionPageTokens];
  for (std::uint32_t token = 0; token < task.tokenCount; ++token) {
    const float product = __half2float(query[threadIdx.x]) *
                          __half2float(keys[token * AttentionHeadDimension +
                                            threadIdx.x]);
    const float dot = blockSum128(product, reduction);
    if (threadIdx.x == 0) {
      logits[token] = dot * 0.08838834764831845F;
    }
    __syncthreads();
  }

  float tileMaximum = -FLT_MAX;
  if (threadIdx.x == 0) {
    for (std::uint32_t token = 0; token < task.tokenCount; ++token) {
      tileMaximum = fmaxf(tileMaximum, logits[token]);
    }
    reduction[0] = tileMaximum;
  }
  __syncthreads();
  tileMaximum = reduction[0];

  float numerator = 0.0F;
  float denominator = 0.0F;
  for (std::uint32_t token = 0; token < task.tokenCount; ++token) {
    const float weight = expf(logits[token] - tileMaximum);
    denominator += weight;
    numerator += weight *
                 __half2float(values[token * AttentionHeadDimension +
                                     threadIdx.x]);
  }
  partial.numerator[threadIdx.x] = numerator;
  if (threadIdx.x == 0) {
    partial.maxLogit = tileMaximum;
    partial.sumExp = denominator;
    __threadfence();
    partial.valid = 1;
    atomicExch(&continuation.state,
               static_cast<std::uint32_t>(nta::abi::ContinuationState::Done));
  }
}

__device__ __forceinline__ void
runAttentionTile(nta::abi::RuntimeView *runtime, const AttentionTileTask *tasks,
                 std::uint32_t taskIndex, const __half *queries,
                 AttentionTilePartial *partials) {
  const AttentionTileTask task = tasks[taskIndex];
  __nta_bind_request(task.requestSlot, task.generation);
  void *address = __nta_acquire_marker(
      runtime, reinterpret_cast<const void *>(task.directBase), task.objectSlot,
      task.objectId, task.objectVersion, 0, task.bytes, task.continuation);
  if (address == nullptr) {
    __nta_defer_marker(runtime, task.continuation);
    return;
  }
  computeAttentionTile(runtime, task, taskIndex,
                       static_cast<const __half *>(address), queries, partials);
}

__device__ __forceinline__ void runAttentionTileTma(
    nta::abi::RuntimeView *runtime, const AttentionTileTask *tasks,
    std::uint32_t taskIndex, const __half *queries,
    AttentionTilePartial *partials) {
  const AttentionTileTask task = tasks[taskIndex];
  __nta_bind_request(task.requestSlot, task.generation);
  void *descriptor = __nta_acquire_tensor_map_marker(
      runtime, reinterpret_cast<const void *>(task.directTensorMap),
      task.objectSlot, task.objectId, task.objectVersion, 0, task.bytes,
      task.continuation);
  if (descriptor == nullptr) {
    __nta_defer_marker(runtime, task.continuation);
    return;
  }

  constexpr std::uint32_t PageElements =
      2U * AttentionPageTokens * AttentionHeadDimension;
  constexpr std::uint32_t PageBytes = PageElements * sizeof(__half);
  using BlockBarrier = cuda::barrier<cuda::thread_scope_block>;
  __shared__ __align__(128) __half page[PageElements];
  __shared__ __align__(8) unsigned char barrierStorage[sizeof(BlockBarrier)];
  BlockBarrier &barrier = *reinterpret_cast<BlockBarrier *>(barrierStorage);
  if (threadIdx.x == 0) {
    init(&barrier, blockDim.x);
  }
  __syncthreads();

  BlockBarrier::arrival_token token;
  if (threadIdx.x == 0) {
    cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(
        page, static_cast<const CUtensorMap *>(descriptor), 0, 0, barrier);
    token = cuda::device::barrier_arrive_tx(barrier, 1, PageBytes);
  } else {
    token = barrier.arrive();
  }
  barrier.wait(static_cast<decltype(token) &&>(token));
  computeAttentionTile(runtime, task, taskIndex, page, queries, partials);
}

} // namespace

extern "C" __global__ void nta_attention_tile_kernel(
    nta::abi::RuntimeView *runtime, const AttentionTileTask *tasks,
    std::uint32_t taskCount, const __half *queries,
    AttentionTilePartial *partials) {
  const std::uint32_t taskIndex = blockIdx.x;
  if (blockDim.x != AttentionHeadDimension || taskIndex >= taskCount) {
    return;
  }
  runAttentionTile(runtime, tasks, taskIndex, queries, partials);
}

extern "C" __global__ void nta_attention_ready_kernel(
    nta::abi::RuntimeView *runtime, const AttentionTileTask *tasks,
    std::uint32_t taskCount, const __half *queries,
    AttentionTilePartial *partials) {
  if (blockDim.x != AttentionHeadDimension) {
    return;
  }
  const std::uint32_t continuation = popReadyContinuation(runtime);
  if (continuation >= runtime->continuationCapacity) {
    return;
  }
  const std::uint32_t taskIndex =
      runtime->continuations[continuation].logicalTile;
  if (taskIndex < taskCount) {
    runAttentionTile(runtime, tasks, taskIndex, queries, partials);
  }
}

extern "C" __global__ void nta_attention_tma_tile_kernel(
    nta::abi::RuntimeView *runtime, const AttentionTileTask *tasks,
    std::uint32_t taskCount, const __half *queries,
    AttentionTilePartial *partials) {
  const std::uint32_t taskIndex = blockIdx.x;
  if (blockDim.x != AttentionHeadDimension || taskIndex >= taskCount) {
    return;
  }
  runAttentionTileTma(runtime, tasks, taskIndex, queries, partials);
}

extern "C" __global__ void nta_attention_tma_ready_kernel(
    nta::abi::RuntimeView *runtime, const AttentionTileTask *tasks,
    std::uint32_t taskCount, const __half *queries,
    AttentionTilePartial *partials) {
  if (blockDim.x != AttentionHeadDimension) {
    return;
  }
  const std::uint32_t continuation = popReadyContinuation(runtime);
  if (continuation >= runtime->continuationCapacity) {
    return;
  }
  const std::uint32_t taskIndex =
      runtime->continuations[continuation].logicalTile;
  if (taskIndex < taskCount) {
    runAttentionTileTma(runtime, tasks, taskIndex, queries, partials);
  }
}

extern "C" __global__ void nta_attention_reduce_kernel(
    nta::abi::RuntimeView *runtime, const AttentionRequest *requests,
    std::uint32_t requestCount, const AttentionTilePartial *partials,
    float *output) {
  const std::uint32_t requestIndex = blockIdx.x;
  if (blockDim.x != AttentionHeadDimension || requestIndex >= requestCount) {
    return;
  }
  const AttentionRequest request = requests[requestIndex];
  if (!nta::device::requestLive(runtime, request.requestSlot,
                                request.generation)) {
    return;
  }

  float maximum = -FLT_MAX;
  bool ready = true;
  for (std::uint32_t tile = 0; tile < request.tileCount; ++tile) {
    const AttentionTilePartial &partial = partials[request.tileBegin + tile];
    if (partial.valid == 0) {
      ready = false;
      break;
    }
    maximum = fmaxf(maximum, partial.maxLogit);
  }
  if (!ready) {
    return;
  }

  float denominator = 0.0F;
  float numerator = 0.0F;
  for (std::uint32_t tile = 0; tile < request.tileCount; ++tile) {
    const AttentionTilePartial &partial = partials[request.tileBegin + tile];
    const float scale = expf(partial.maxLogit - maximum);
    denominator += scale * partial.sumExp;
    numerator += scale * partial.numerator[threadIdx.x];
  }
  output[requestIndex * AttentionHeadDimension + threadIdx.x] =
      numerator / denominator;
}
