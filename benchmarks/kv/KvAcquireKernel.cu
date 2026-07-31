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

__device__ __forceinline__ std::uint64_t warpSum64(std::uint64_t value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  return value;
}

} // namespace

namespace {

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
runKvTile(nta::abi::RuntimeView *runtime,
          const nta::benchmark::TileTask *tasks, std::uint32_t taskIndex,
          const float *query, float *output) {

  const nta::benchmark::TileTask task = tasks[taskIndex];
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

__device__ __forceinline__ void
runNvmeHash(nta::abi::RuntimeView *runtime,
            const nta::benchmark::TileTask *tasks, std::uint32_t taskIndex,
            std::uint64_t *output) {
  const nta::benchmark::TileTask task = tasks[taskIndex];
  nta::abi::Continuation &continuation =
      runtime->continuations[task.continuation];

  __nta_bind_request(task.requestSlot, task.generation);
  void *address = __nta_acquire_marker(
      runtime, nullptr, task.objectSlot, task.objectId, task.objectVersion,
      task.offset, task.bytes, task.continuation);
  if (address == nullptr) {
    __nta_defer_marker(runtime, task.continuation);
    return;
  }

  const auto *values = static_cast<const std::uint32_t *>(address);
  const std::uint32_t count = task.bytes / sizeof(std::uint32_t);
  std::uint64_t partial = 0;
  for (std::uint32_t element = threadIdx.x; element < count;
       element += blockDim.x) {
    const std::uint32_t value = nta::device::loadIoCoherent(values + element);
    partial += static_cast<std::uint64_t>(value) * (element + 1ULL);
  }

  partial = warpSum64(partial);
  __shared__ std::uint64_t warpTotals64[32];
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t warp = threadIdx.x >> 5U;
  if (lane == 0) {
    warpTotals64[warp] = partial;
  }
  __syncthreads();

  if (warp == 0) {
    const std::uint32_t warpCount = (blockDim.x + 31U) / 32U;
    std::uint64_t total = lane < warpCount ? warpTotals64[lane] : 0;
    total = warpSum64(total);
    if (lane == 0) {
      output[taskIndex] = total;
      __threadfence_system();
      atomicExch(&continuation.state,
                 static_cast<std::uint32_t>(nta::abi::ContinuationState::Done));
    }
  }
}

__device__ __forceinline__ void
runDependencyTile(nta::abi::RuntimeView *runtime,
                  const nta::benchmark::WorkTask *tasks,
                  const nta::abi::AcquireRequirement *requirements,
                  std::uint32_t taskIndex, const float *query, float *output) {
  const nta::benchmark::WorkTask task = tasks[taskIndex];
  nta::abi::Continuation &continuation =
      runtime->continuations[task.continuation];
  const nta::abi::AcquireRequirement *dependencies =
      requirements + task.dependencyBegin;

  __nta_bind_request(task.requestSlot, task.generation);
  const bool ready = __nta_acquire_set_marker(
      runtime, dependencies, task.dependencyCount, task.directDependencyCount,
      task.continuation);
  if (!ready) {
    __nta_defer_marker(runtime, task.continuation);
    return;
  }

  float partial = 0.0F;
  for (std::uint32_t dependency = 0; dependency < task.dependencyCount;
       ++dependency) {
    const auto *values = static_cast<const float *>(
        nta_requirement_address(runtime, dependencies + dependency));
    if (values == nullptr) {
      nta::device::failContinuation(runtime, task.continuation,
                                    nta::abi::ContinuationState::Failed);
      return;
    }
    const std::uint32_t count = dependencies[dependency].bytes / sizeof(float);
    const float weight = static_cast<float>(dependency + 1U);
    for (std::uint32_t element = threadIdx.x; element < count;
         element += blockDim.x) {
      partial = fmaf(values[element] * weight, query[element], partial);
    }
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

__device__ __forceinline__ void
runDependencyBaseline(const nta::benchmark::WorkTask *tasks,
                      const nta::abi::AcquireRequirement *requirements,
                      std::uint32_t taskIndex, const float *query,
                      float *output) {
  const nta::benchmark::WorkTask task = tasks[taskIndex];
  const nta::abi::AcquireRequirement *dependencies =
      requirements + task.dependencyBegin;

  float partial = 0.0F;
  for (std::uint32_t dependency = 0; dependency < task.dependencyCount;
       ++dependency) {
    const auto *values = reinterpret_cast<const float *>(
        dependencies[dependency].directBase + dependencies[dependency].offset);
    const std::uint32_t count = dependencies[dependency].bytes / sizeof(float);
    const float weight = static_cast<float>(dependency + 1U);
    for (std::uint32_t element = threadIdx.x; element < count;
         element += blockDim.x) {
      partial = fmaf(values[element] * weight, query[element], partial);
    }
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
    }
  }
}

} // namespace

extern "C" __global__ void
nta_kv_tile_kernel(nta::abi::RuntimeView *runtime,
                   const nta::benchmark::TileTask *tasks,
                   std::uint32_t taskCount, const float *query, float *output,
                   std::uint32_t phase) {
  const std::uint32_t taskIndex = blockIdx.x;
  if (phase != 0 || taskIndex >= taskCount) {
    return;
  }
  runKvTile(runtime, tasks, taskIndex, query, output);
}

extern "C" __global__ void
nta_kv_ready_kernel(nta::abi::RuntimeView *runtime,
                    const nta::benchmark::TileTask *tasks,
                    std::uint32_t taskCount, const float *query, float *output) {
  const std::uint32_t continuation = popReadyContinuation(runtime);
  if (continuation >= runtime->continuationCapacity) {
    return;
  }
  const std::uint32_t taskIndex =
      runtime->continuations[continuation].logicalTile;
  if (taskIndex < taskCount) {
    runKvTile(runtime, tasks, taskIndex, query, output);
  }
}

extern "C" __global__ void nta_nvme_hash_kernel(
    nta::abi::RuntimeView *runtime, const nta::benchmark::TileTask *tasks,
    std::uint32_t taskCount, std::uint64_t *output, std::uint32_t phase) {
  const std::uint32_t taskIndex = blockIdx.x;
  if (phase != 0 || taskIndex >= taskCount) {
    return;
  }
  runNvmeHash(runtime, tasks, taskIndex, output);
}

extern "C" __global__ void nta_nvme_ready_hash_kernel(
    nta::abi::RuntimeView *runtime, const nta::benchmark::TileTask *tasks,
    std::uint32_t taskCount, std::uint64_t *output) {
  const std::uint32_t continuation = popReadyContinuation(runtime);
  if (continuation >= runtime->continuationCapacity) {
    return;
  }
  const std::uint32_t taskIndex =
      runtime->continuations[continuation].logicalTile;
  if (taskIndex < taskCount) {
    runNvmeHash(runtime, tasks, taskIndex, output);
  }
}

extern "C" __global__ void nta_dependency_tile_kernel(
    nta::abi::RuntimeView *runtime, const nta::benchmark::WorkTask *tasks,
    std::uint32_t taskCount, const nta::abi::AcquireRequirement *requirements,
    const float *query, float *output, std::uint32_t phase) {
  const std::uint32_t taskIndex = blockIdx.x;
  if (phase != 0 || taskIndex >= taskCount) {
    return;
  }
  runDependencyTile(runtime, tasks, requirements, taskIndex, query, output);
}

extern "C" __global__ void nta_dependency_ready_kernel(
    nta::abi::RuntimeView *runtime, const nta::benchmark::WorkTask *tasks,
    std::uint32_t taskCount, const nta::abi::AcquireRequirement *requirements,
    const float *query, float *output) {
  const std::uint32_t continuation = popReadyContinuation(runtime);
  if (continuation >= runtime->continuationCapacity) {
    return;
  }
  const std::uint32_t taskIndex =
      runtime->continuations[continuation].logicalTile;
  if (taskIndex < taskCount) {
    runDependencyTile(runtime, tasks, requirements, taskIndex, query, output);
  }
}

extern "C" __global__ void
nta_dependency_baseline_kernel(const nta::benchmark::WorkTask *tasks,
                               std::uint32_t taskCount,
                               const nta::abi::AcquireRequirement *requirements,
                               const float *query, float *output) {
  const std::uint32_t taskIndex = blockIdx.x;
  if (taskIndex < taskCount) {
    runDependencyBaseline(tasks, requirements, taskIndex, query, output);
  }
}
