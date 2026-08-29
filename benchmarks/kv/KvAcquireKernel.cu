#include "benchmarks/kv/KvTypes.h"
#include "nta/KernelPolicy.cuh"
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
popReadyWorkTicket(nta::abi::RuntimeView *runtime) {
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
        selected = runtime->readyWorkTickets[head];
        break;
      }
    }
  }
  __syncthreads();
  return selected;
}

__device__ __forceinline__ void runKvTile(nta::abi::RuntimeView *runtime,
                                          const nta::benchmark::TileTask *tasks,
                                          std::uint32_t taskIndex,
                                          const float *query, float *output) {

  const nta::benchmark::TileTask task = tasks[taskIndex];
  const nta::kernel::BoundRequest request{task.requestSlot, task.generation,
                                          task.workTicket};
  const nta::abi::AcquireRequirement requirement{
      task.directBase, 0,
      task.objectId,   task.offset,
      task.objectSlot, task.objectVersion,
      task.bytes,      0};
  void *address = nta::kernel::acquireAddress(runtime, request, requirement);
  if (address == nullptr) {
    nta::kernel::defer(runtime, request);
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
      (void)nta::device::completeBoundWorkTicket(
          runtime, task.requestSlot, task.generation, task.workTicket);
    }
  }
}

__device__ __forceinline__ void
runNvmeHash(nta::abi::RuntimeView *runtime,
            const nta::benchmark::TileTask *tasks, std::uint32_t taskIndex,
            std::uint64_t *output) {
  const nta::benchmark::TileTask task = tasks[taskIndex];
  const nta::kernel::BoundRequest request{task.requestSlot, task.generation,
                                          task.workTicket};
  const nta::abi::AcquireRequirement requirement{0,
                                                 0,
                                                 task.objectId,
                                                 task.offset,
                                                 task.objectSlot,
                                                 task.objectVersion,
                                                 task.bytes,
                                                 0};
  void *address = nta::kernel::acquireAddress(runtime, request, requirement);
  if (address == nullptr) {
    nta::kernel::defer(runtime, request);
    return;
  }

  const auto *values = static_cast<const std::uint32_t *>(address);
  const std::uint32_t count = task.bytes / sizeof(std::uint32_t);
  const bool directHbm =
      (runtime->objects[task.objectSlot].flags & nta::abi::ReplicaDmaHbm) != 0;
  std::uint64_t partial = 0;
  for (std::uint32_t element = threadIdx.x; element < count;
       element += blockDim.x) {
    // Host-mapped DMA requires the uncached/coherent load. Direct NVMe-to-HBM
    // data is ordinary CUDA global memory and should use the normal cache path.
    const std::uint32_t value =
        directHbm ? values[element]
                  : nta::device::loadIoCoherent(values + element);
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
      (void)nta::device::completeBoundWorkTicket(
          runtime, task.requestSlot, task.generation, task.workTicket);
    }
  }
}

__device__ __forceinline__ void
runDependencyTile(nta::abi::RuntimeView *runtime,
                  const nta::abi::WorkItem *tasks,
                  const nta::abi::AcquireRequirement *requirements,
                  std::uint32_t taskIndex, const float *query, float *output) {
  nta::kernel::WorkContext work{};
  const bool ready =
      nta::kernel::acquireWork(runtime, tasks, requirements, taskIndex, work);
  const nta::abi::WorkItem task = work.item;
  if (!ready) {
    nta::kernel::defer(runtime, work);
    return;
  }

  float partial = 0.0F;
  for (std::uint32_t dependency = 0; dependency < task.dependencyCount;
       ++dependency) {
    const nta::abi::AcquireRequirement *requirement =
        work.requirement(dependency);
    const auto *values = static_cast<const float *>(
        nta::kernel::address(runtime, work, dependency));
    if (requirement == nullptr || values == nullptr) {
      nta::device::failWorkTicket(runtime, task.workTicket,
                                  nta::abi::WorkTicketState::Failed);
      return;
    }
    const std::uint32_t count = requirement->bytes / sizeof(float);
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
      (void)nta::device::completeBoundWorkTicket(
          runtime, task.requestSlot, task.generation, task.workTicket);
    }
  }
}

__device__ __forceinline__ void
runDependencyBaseline(const nta::abi::WorkItem *tasks,
                      const nta::abi::AcquireRequirement *requirements,
                      std::uint32_t taskIndex, const float *query,
                      float *output) {
  const nta::abi::WorkItem task = tasks[taskIndex];
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

__device__ __forceinline__ void
runMoeTile(nta::abi::RuntimeView *runtime, const nta::abi::WorkItem *tasks,
           const nta::abi::AcquireRequirement *requirements,
           std::uint32_t taskIndex, const float *input, float *output,
           std::uint32_t hiddenSize) {
  nta::kernel::WorkContext work{};
  const bool ready =
      nta::kernel::acquireWork(runtime, tasks, requirements, taskIndex, work);
  const nta::abi::WorkItem task = work.item;
  if (!ready) {
    nta::kernel::defer(runtime, work);
    return;
  }

  float mixed = 0.0F;
  const float *token = input + task.logicalWork * hiddenSize;
  for (std::uint32_t dependency = 0; dependency < task.dependencyCount;
       ++dependency) {
    const auto *weights = static_cast<const float *>(
        nta::kernel::address(runtime, work, dependency));
    if (weights == nullptr) {
      nta::device::failWorkTicket(runtime, task.workTicket,
                                  nta::abi::WorkTicketState::Failed);
      return;
    }
    const float gate = 1.0F / static_cast<float>(dependency + 1U);
    float expertOutput = 0.0F;
    const float *row = weights + threadIdx.x * hiddenSize;
    for (std::uint32_t inputIndex = 0; inputIndex < hiddenSize; ++inputIndex) {
      expertOutput = fmaf(row[inputIndex], token[inputIndex], expertOutput);
    }
    mixed = fmaf(gate, expertOutput, mixed);
  }
  output[task.logicalWork * hiddenSize + threadIdx.x] = mixed;
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    (void)nta::device::completeBoundWorkTicket(
        runtime, task.requestSlot, task.generation, task.workTicket);
  }
}

__device__ __forceinline__ void
runMoeBaseline(const nta::abi::WorkItem *tasks,
               const nta::abi::AcquireRequirement *requirements,
               std::uint32_t taskIndex, const float *input, float *output,
               std::uint32_t hiddenSize) {
  const nta::abi::WorkItem task = tasks[taskIndex];
  const nta::abi::AcquireRequirement *dependencies =
      requirements + task.dependencyBegin;
  const float *token = input + task.logicalWork * hiddenSize;
  float mixed = 0.0F;
  for (std::uint32_t dependency = 0; dependency < task.dependencyCount;
       ++dependency) {
    const auto *weights = reinterpret_cast<const float *>(
        dependencies[dependency].directBase + dependencies[dependency].offset);
    const float gate = 1.0F / static_cast<float>(dependency + 1U);
    float expertOutput = 0.0F;
    const float *row = weights + threadIdx.x * hiddenSize;
    for (std::uint32_t inputIndex = 0; inputIndex < hiddenSize; ++inputIndex) {
      expertOutput = fmaf(row[inputIndex], token[inputIndex], expertOutput);
    }
    mixed = fmaf(gate, expertOutput, mixed);
  }
  output[task.logicalWork * hiddenSize + threadIdx.x] = mixed;
}

__device__ __forceinline__ std::uint32_t
routeHash(std::uint32_t token, std::uint32_t expert, std::uint32_t epoch) {
  std::uint32_t value =
      token * 0x9e3779b9U ^ expert * 0x85ebca6bU ^ epoch * 0xc2b2ae35U;
  value ^= value >> 16U;
  value *= 0x7feb352dU;
  value ^= value >> 15U;
  value *= 0x846ca68bU;
  return value ^ (value >> 16U);
}

__device__ __forceinline__ float routeScore(const float *gateWeights,
                                            const float *token,
                                            std::uint32_t expert,
                                            std::uint32_t hiddenSize) {
  const std::uint32_t lane = threadIdx.x & 31U;
  const float *expertWeights = gateWeights + expert * hiddenSize;
  float partial = 0.0F;
  for (std::uint32_t index = lane; index < hiddenSize; index += 32U) {
    partial = fmaf(expertWeights[index], token[index], partial);
  }
  partial = warpSum(partial);
  return partial;
}

__device__ __forceinline__ void buildMoePlan(
    nta::abi::RuntimeView *runtime,
    const nta::benchmark::MoeExpertDescriptor *experts,
    const float *gateWeights, const float *input, nta::abi::WorkItem *workItems,
    nta::abi::AcquireRequirement *requirements, std::uint32_t *selectedExperts,
    std::uint32_t tokenCount, std::uint32_t expertCount, std::uint32_t topK,
    std::uint32_t hiddenSize, bool preacquired) {
  const std::uint32_t tokenIndex = blockIdx.x;
  if (tokenIndex >= tokenCount || blockDim.x != 32 || topK == 0 || topK > 32) {
    return;
  }

  __shared__ float topScores[32];
  __shared__ std::uint32_t topExperts[32];
  if (threadIdx.x == 0) {
    for (std::uint32_t rank = 0; rank < topK; ++rank) {
      topScores[rank] = -CUDART_INF_F;
      topExperts[rank] = nta::abi::InvalidIndex;
    }
  }

  const float *token = input + tokenIndex * hiddenSize;
  for (std::uint32_t expert = 0; expert < expertCount; ++expert) {
    const float score = routeScore(gateWeights, token, expert, hiddenSize);
    if (threadIdx.x == 0) {
      for (std::uint32_t rank = 0; rank < topK; ++rank) {
        if (score > topScores[rank] ||
            (score == topScores[rank] && expert < topExperts[rank])) {
          for (std::uint32_t move = topK - 1; move > rank; --move) {
            topScores[move] = topScores[move - 1];
            topExperts[move] = topExperts[move - 1];
          }
          topScores[rank] = score;
          topExperts[rank] = expert;
          break;
        }
      }
    }
  }

  if (threadIdx.x != 0) {
    return;
  }
  const nta::abi::RequestContext request = runtime->requests[tokenIndex];
  const std::uint32_t dependencyBegin = tokenIndex * topK;
  std::uint32_t directDependencyCount = topK;
  for (std::uint32_t rank = 0; rank < topK; ++rank) {
    const std::uint32_t expert = topExperts[rank];
    const nta::benchmark::MoeExpertDescriptor descriptor = experts[expert];
    const std::uint64_t directBase =
        preacquired ? descriptor.consumeBase : descriptor.directBase;
    if (directBase == 0) {
      directDependencyCount = 0;
    }
    selectedExperts[dependencyBegin + rank] = expert;
    requirements[dependencyBegin + rank] = {directBase,
                                            0,
                                            descriptor.objectId,
                                            0,
                                            descriptor.objectSlot,
                                            descriptor.objectVersion,
                                            descriptor.bytes,
                                            0};
  }
  workItems[tokenIndex] = {
      tokenIndex,
      tokenIndex,
      request.generation,
      tokenIndex,
      dependencyBegin,
      topK,
      directDependencyCount,
      tokenIndex,
      tokenIndex,
      0,
      1,
      0,
      0,
      0,
      0,
  };
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

extern "C" __global__ void nta_kv_ready_kernel(
    nta::abi::RuntimeView *runtime, const nta::benchmark::TileTask *tasks,
    std::uint32_t taskCount, const float *query, float *output) {
  const std::uint32_t workTicket = popReadyWorkTicket(runtime);
  if (workTicket >= runtime->workTicketCapacity) {
    return;
  }
  const std::uint32_t taskIndex = runtime->workTickets[workTicket].logicalTile;
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

extern "C" __global__ void
nta_nvme_benchmark_invalidate(nta::abi::RuntimeView *runtime,
                              std::uint32_t objectCount) {
  const std::uint32_t objectSlot = blockIdx.x * blockDim.x + threadIdx.x;
  if (runtime == nullptr || objectSlot >= objectCount ||
      objectSlot >= runtime->objectCapacity) {
    return;
  }
  // Benchmark-only compulsory-miss control. Production epoch reset deliberately
  // retains matching (objectId, version) staging entries as an HBM cache.
  runtime->objects[objectSlot].state =
      static_cast<std::uint32_t>(nta::abi::ObjectState::New);
}

extern "C" __global__ void
nta_nvme_ready_hash_kernel(nta::abi::RuntimeView *runtime,
                           const nta::benchmark::TileTask *tasks,
                           std::uint32_t taskCount, std::uint64_t *output) {
  const std::uint32_t workTicket = popReadyWorkTicket(runtime);
  if (workTicket >= runtime->workTicketCapacity) {
    return;
  }
  const std::uint32_t taskIndex = runtime->workTickets[workTicket].logicalTile;
  if (taskIndex < taskCount) {
    runNvmeHash(runtime, tasks, taskIndex, output);
  }
}

extern "C" __global__ void nta_dependency_tile_kernel(
    nta::abi::RuntimeView *runtime, const nta::abi::WorkItem *tasks,
    std::uint32_t taskCount, const nta::abi::AcquireRequirement *requirements,
    const float *query, float *output, std::uint32_t phase) {
  const std::uint32_t taskIndex = blockIdx.x;
  if (phase != 0 || taskIndex >= taskCount) {
    return;
  }
  runDependencyTile(runtime, tasks, requirements, taskIndex, query, output);
}

extern "C" __global__ void nta_dependency_ready_kernel(
    nta::abi::RuntimeView *runtime, const nta::abi::WorkItem *tasks,
    std::uint32_t taskCount, const nta::abi::AcquireRequirement *requirements,
    const float *query, float *output) {
  const std::uint32_t workTicket = popReadyWorkTicket(runtime);
  if (workTicket >= runtime->workTicketCapacity) {
    return;
  }
  const std::uint32_t taskIndex = runtime->workTickets[workTicket].logicalTile;
  if (taskIndex < taskCount) {
    runDependencyTile(runtime, tasks, requirements, taskIndex, query, output);
  }
}

extern "C" __global__ void
nta_dependency_baseline_kernel(const nta::abi::WorkItem *tasks,
                               std::uint32_t taskCount,
                               const nta::abi::AcquireRequirement *requirements,
                               const float *query, float *output) {
  const std::uint32_t taskIndex = blockIdx.x;
  if (taskIndex < taskCount) {
    runDependencyBaseline(tasks, requirements, taskIndex, query, output);
  }
}

extern "C" __global__ void nta_moe_tile_kernel(
    nta::abi::RuntimeView *runtime, const nta::abi::WorkItem *tasks,
    std::uint32_t taskCount, const nta::abi::AcquireRequirement *requirements,
    const float *input, float *output, std::uint32_t hiddenSize) {
  const std::uint32_t taskIndex = blockIdx.x;
  if (taskIndex >= taskCount || hiddenSize == 0 || blockDim.x != hiddenSize) {
    return;
  }
  runMoeTile(runtime, tasks, requirements, taskIndex, input, output,
             hiddenSize);
}

extern "C" __global__ void nta_moe_ready_kernel(
    nta::abi::RuntimeView *runtime, const nta::abi::WorkItem *tasks,
    std::uint32_t taskCount, const nta::abi::AcquireRequirement *requirements,
    const float *input, float *output, std::uint32_t hiddenSize) {
  if (hiddenSize == 0 || blockDim.x != hiddenSize) {
    return;
  }
  const std::uint32_t workTicket = popReadyWorkTicket(runtime);
  if (workTicket >= runtime->workTicketCapacity) {
    return;
  }
  const std::uint32_t taskIndex = runtime->workTickets[workTicket].logicalTile;
  if (taskIndex < taskCount) {
    runMoeTile(runtime, tasks, requirements, taskIndex, input, output,
               hiddenSize);
  }
}

extern "C" __global__ void nta_moe_baseline_kernel(
    const nta::abi::WorkItem *tasks, std::uint32_t taskCount,
    const nta::abi::AcquireRequirement *requirements, const float *input,
    float *output, std::uint32_t hiddenSize) {
  const std::uint32_t taskIndex = blockIdx.x;
  if (taskIndex >= taskCount || hiddenSize == 0 || blockDim.x != hiddenSize) {
    return;
  }
  runMoeBaseline(tasks, requirements, taskIndex, input, output, hiddenSize);
}

extern "C" __global__ void nta_moe_advance_epoch_kernel(std::uint32_t *epoch) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *epoch += 1U;
  }
}

extern "C" __global__ void
nta_moe_prepare_input_kernel(const float *baseInput, float *input,
                             const std::uint32_t *epoch,
                             std::uint32_t elementCount) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elementCount) {
    return;
  }
  const std::uint32_t noise = routeHash(index, index >> 5U, *epoch);
  const float delta =
      (static_cast<float>(noise & 0xffffU) / 65535.0F - 0.5F) * 0.5F;
  input[index] = baseInput[index] + delta;
}

extern "C" __global__ void nta_moe_route_kernel(
    nta::abi::RuntimeView *runtime,
    const nta::benchmark::MoeExpertDescriptor *experts,
    const float *gateWeights, const float *input, nta::abi::WorkItem *workItems,
    nta::abi::AcquireRequirement *requirements, std::uint32_t *selectedExperts,
    std::uint32_t tokenCount, std::uint32_t expertCount, std::uint32_t topK,
    std::uint32_t hiddenSize, std::uint32_t preacquired) {
  buildMoePlan(runtime, experts, gateWeights, input, workItems, requirements,
               selectedExperts, tokenCount, expertCount, topK, hiddenSize,
               preacquired != 0);
}

extern "C" __global__ void
nta_moe_copy_all_kernel(const nta::benchmark::MoeExpertDescriptor *experts,
                        std::uint32_t expertCount) {
  const std::uint32_t expert = blockIdx.x;
  if (expert >= expertCount) {
    return;
  }
  const nta::benchmark::MoeExpertDescriptor descriptor = experts[expert];
  if ((descriptor.flags & nta::benchmark::MoeExpertStaged) == 0 ||
      descriptor.sourceBase == 0 || descriptor.consumeBase == 0) {
    return;
  }
  const auto *source = reinterpret_cast<const uint4 *>(descriptor.sourceBase);
  auto *destination = reinterpret_cast<uint4 *>(descriptor.consumeBase);
  const std::uint32_t vectorCount = descriptor.bytes / sizeof(uint4);
  for (std::uint32_t index = threadIdx.x; index < vectorCount;
       index += blockDim.x) {
    nta::device::storeNoAllocate(destination + index,
                                 nta::device::loadNoAllocate(source + index));
  }
}
