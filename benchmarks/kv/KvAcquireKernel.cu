#include "benchmarks/kv/KvTypes.h"
#include "nta/DeviceAPI.cuh"
#include "runtime/device/Acquire.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace {

__device__ __forceinline__ float warpSum(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  return value;
}

} // namespace

extern "C" __global__ void
nta_kv_tile_kernel(nta::abi::RuntimeView *runtime,
                   const nta::benchmark::TileTask *tasks,
                   std::uint32_t taskCount, const float *query, float *output,
                   std::uint32_t phase) {
  const std::uint32_t taskIndex = blockIdx.x;
  if (taskIndex >= taskCount) {
    return;
  }

  const nta::benchmark::TileTask task = tasks[taskIndex];
  const bool direct = task.directBase != 0;
  if (phase != 0 && direct) {
    return;
  }

  nta::abi::Continuation &continuation =
      runtime->continuations[task.continuation];

  __nta_bind_request(task.requestSlot, task.generation);
  void *address = __nta_acquire_marker(
      runtime, reinterpret_cast<const void *>(task.directBase), task.objectSlot,
      task.objectId, task.objectVersion, task.offset, task.bytes,
      task.continuation);
  if (address == nullptr) {
    __nta_defer_marker(runtime, task.continuation);
    return;
  }

  const auto *values = static_cast<const float *>(address);
  const std::uint32_t count = task.bytes / sizeof(float);
  float partial = 0.0F;
  for (std::uint32_t element = threadIdx.x; element < count;
       element += blockDim.x) {
    partial = fmaf(values[element], query[element], partial);
  }

  partial = warpSum(partial);
  __shared__ float warpTotals[32];
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t warp = threadIdx.x >> 5U;
  if (lane == 0) {
    warpTotals[warp] = partial;
  }
  __syncthreads();

  if (warp == 0) {
    const std::uint32_t warpCount = (blockDim.x + 31U) / 32U;
    float total = lane < warpCount ? warpTotals[lane] : 0.0F;
    total = warpSum(total);
    if (lane == 0) {
      output[taskIndex] = total;
      __threadfence();
      continuation.state =
          static_cast<std::uint32_t>(nta::abi::ContinuationState::Done);
    }
  }
}
