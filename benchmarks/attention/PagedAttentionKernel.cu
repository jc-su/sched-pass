#include "benchmarks/attention/PagedAttentionTypes.h"
#include "nta/KernelPolicy.cuh"
#include "runtime/device/Acquire.cuh"

#if __has_include(<cuda/barrier>)
#include <cuda/barrier>
#elif __has_include(<cccl/cuda/barrier>)
#include <cccl/cuda/barrier>
#else
#error "the selected CUDA toolkit has no supported barrier header"
#endif
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#if defined(NTA_USE_FLASHINFER_STATE)
#include <flashinfer/attention/state.cuh>
#endif

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

using nta::benchmark::AttentionHeadDimension;
using nta::benchmark::AttentionPageDescriptor;
using nta::benchmark::AttentionPageNeedsStaging;
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

__device__ __forceinline__ void
computeAttentionTile(nta::abi::RuntimeView *runtime,
                     const AttentionTileTask &task, std::uint32_t taskIndex,
                     const nta::abi::WorkItem &work, const __half *page,
                     const __half *queries, AttentionTilePartial *partials) {
  const __half *keys = page;
  const __half *values = page + AttentionPageTokens * AttentionHeadDimension;
  const __half *query = queries + task.requestIndex * AttentionHeadDimension;
  AttentionTilePartial &partial = partials[taskIndex];

  __shared__ float reduction[AttentionHeadDimension];
  __shared__ float logits[AttentionPageTokens];
  for (std::uint32_t token = 0; token < task.tokenCount; ++token) {
    const float product =
        __half2float(query[threadIdx.x]) *
        __half2float(keys[token * AttentionHeadDimension + threadIdx.x]);
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
    numerator +=
        weight *
        __half2float(values[token * AttentionHeadDimension + threadIdx.x]);
  }
  partial.output[threadIdx.x] = numerator / denominator;
  if (threadIdx.x == 0) {
    partial.lse = (tileMaximum + logf(denominator)) * 1.4426950408889634F;
    __threadfence();
    partial.valid = 1;
    (void)nta::device::completeBoundWorkTicket(
        runtime, work.requestSlot, work.generation, work.workTicket);
  }
}

constexpr std::uint32_t SparseTopKLimit = 8;

__device__ __forceinline__ void selectSparsePages(
    const AttentionPageDescriptor *catalog,
    const std::uint32_t *candidateOffsets, const __half *summaries,
    const __half *queries, std::uint32_t requestIndex, std::uint32_t topK,
    nta::abi::WorkItem *workItems, nta::abi::AcquireRequirement *requirements,
    std::uint32_t *selectedObjectSlots, bool preacquired, bool splitWork) {
  __shared__ float reduction[AttentionHeadDimension];
  __shared__ float topScores[SparseTopKLimit];
  __shared__ std::uint32_t topSlots[SparseTopKLimit];
  if (threadIdx.x == 0) {
    for (std::uint32_t rank = 0; rank < topK; ++rank) {
      topScores[rank] = -FLT_MAX;
      topSlots[rank] = nta::abi::InvalidIndex;
    }
  }
  __syncthreads();

  const __half *query = queries + requestIndex * AttentionHeadDimension;
  const std::uint32_t begin = candidateOffsets[requestIndex];
  const std::uint32_t end = candidateOffsets[requestIndex + 1U];
  for (std::uint32_t candidate = begin; candidate < end; ++candidate) {
    const __half *summary = summaries + candidate * AttentionHeadDimension;
    const float score = blockSum128(__half2float(query[threadIdx.x]) *
                                        __half2float(summary[threadIdx.x]),
                                    reduction);
    if (threadIdx.x == 0) {
      for (std::uint32_t rank = 0; rank < topK; ++rank) {
        if (score > topScores[rank] ||
            (score == topScores[rank] && candidate < topSlots[rank])) {
          for (std::uint32_t move = topK - 1U; move > rank; --move) {
            topScores[move] = topScores[move - 1U];
            topSlots[move] = topSlots[move - 1U];
          }
          topScores[rank] = score;
          topSlots[rank] = candidate;
          break;
        }
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    std::uint32_t directCount = 0;
    const std::uint32_t dependencyBegin = requestIndex * topK;
    for (std::uint32_t rank = 0; rank < topK; ++rank) {
      const std::uint32_t candidate = topSlots[rank];
      const AttentionPageDescriptor descriptor = catalog[candidate];
      const std::uint64_t directBase =
          preacquired ? descriptor.consumeBase : descriptor.directBase;
      selectedObjectSlots[dependencyBegin + rank] = candidate;
      requirements[dependencyBegin + rank] = {
          directBase,
          0,
          descriptor.objectId,
          0,
          descriptor.objectSlot,
          descriptor.objectVersion,
          descriptor.bytes,
          0,
      };
      directCount += directBase != 0 ? 1U : 0U;
    }
    // Discovery changes the selected dependency set only. Request ownership,
    // reduction identity, and the host-calibrated cost remain structural plan
    // metadata and must survive device-side page selection.
    if (splitWork) {
      for (std::uint32_t rank = 0; rank < topK; ++rank) {
        const std::uint32_t workIndex = dependencyBegin + rank;
        nta::abi::WorkItem &work = workItems[workIndex];
        work.dependencyBegin = workIndex;
        work.dependencyCount = 1;
        work.directDependencyCount =
            requirements[workIndex].directBase != 0 ? 1U : 0U;
      }
    } else {
      nta::abi::WorkItem &work = workItems[requestIndex];
      work.dependencyBegin = dependencyBegin;
      work.dependencyCount = topK;
      work.directDependencyCount = directCount;
    }
    __threadfence();
  }
  __syncthreads();
}

__device__ __forceinline__ void runSelectedSparseAttention(
    nta::abi::RuntimeView *runtime, const AttentionPageDescriptor *catalog,
    const __half *queries, std::uint32_t requestIndex,
    std::uint32_t requestCount, std::uint32_t topK,
    nta::abi::WorkItem *workItems, nta::abi::AcquireRequirement *requirements,
    const std::uint32_t *selectedObjectSlots, float *output) {
  if (requestIndex >= requestCount || topK == 0 || topK > SparseTopKLimit) {
    return;
  }
  nta::kernel::WorkContext work{};
  const bool ready = nta::kernel::acquireWork(runtime, workItems, requirements,
                                              requestIndex, work);
  if (!ready) {
    nta::kernel::defer(runtime, work);
    return;
  }

  __shared__ float reduction[AttentionHeadDimension];
  __shared__ float runningMaximum;
  __shared__ float runningDenominator;
  __shared__ float previousScale;
  __shared__ float tokenScale;
  if (threadIdx.x == 0) {
    runningMaximum = -FLT_MAX;
    runningDenominator = 0.0F;
  }
  __syncthreads();

  const __half *query = queries + requestIndex * AttentionHeadDimension;
  float numerator = 0.0F;
  const std::uint32_t selectionBegin = requestIndex * topK;
  for (std::uint32_t rank = 0; rank < topK; ++rank) {
    // Reload selection after the deferral boundary. No shared/local selector
    // state is carried across a CTA exit.
    const std::uint32_t catalogIndex =
        selectedObjectSlots[selectionBegin + rank];
    const AttentionPageDescriptor descriptor = catalog[catalogIndex];
    const auto *page =
        static_cast<const __half *>(nta::kernel::address(runtime, work, rank));
    if (page == nullptr) {
      nta::device::failWorkTicket(runtime, work.item.workTicket,
                                  nta::abi::WorkTicketState::Failed);
      return;
    }
    const __half *keys = page;
    const __half *values = page + AttentionPageTokens * AttentionHeadDimension;
    for (std::uint32_t token = 0; token < descriptor.tokenCount; ++token) {
      const float score =
          blockSum128(
              __half2float(query[threadIdx.x]) *
                  __half2float(
                      keys[token * AttentionHeadDimension + threadIdx.x]),
              reduction) *
          0.08838834764831845F;
      if (threadIdx.x == 0) {
        const float updatedMaximum = fmaxf(runningMaximum, score);
        previousScale = runningDenominator == 0.0F
                            ? 0.0F
                            : expf(runningMaximum - updatedMaximum);
        tokenScale = expf(score - updatedMaximum);
        runningDenominator = runningDenominator * previousScale + tokenScale;
        runningMaximum = updatedMaximum;
      }
      __syncthreads();
      numerator = numerator * previousScale +
                  tokenScale *
                      __half2float(
                          values[token * AttentionHeadDimension + threadIdx.x]);
    }
  }
  output[requestIndex * AttentionHeadDimension + threadIdx.x] =
      numerator / runningDenominator;
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    (void)nta::device::completeBoundWorkTicket(runtime, work.item.requestSlot,
                                               work.item.generation,
                                               work.item.workTicket);
  }
}

__device__ __forceinline__ void runSparseAttention(
    nta::abi::RuntimeView *runtime, const AttentionPageDescriptor *catalog,
    const std::uint32_t *candidateOffsets, const __half *summaries,
    const __half *queries, std::uint32_t requestIndex,
    std::uint32_t requestCount, std::uint32_t topK,
    nta::abi::WorkItem *workItems, nta::abi::AcquireRequirement *requirements,
    std::uint32_t *selectedObjectSlots, float *output, bool preacquired) {
  if (requestIndex >= requestCount || topK == 0 || topK > SparseTopKLimit ||
      candidateOffsets[requestIndex + 1U] - candidateOffsets[requestIndex] <
          topK) {
    return;
  }
  selectSparsePages(catalog, candidateOffsets, summaries, queries, requestIndex,
                    topK, workItems, requirements, selectedObjectSlots,
                    preacquired, false);
  runSelectedSparseAttention(runtime, catalog, queries, requestIndex,
                             requestCount, topK, workItems, requirements,
                             selectedObjectSlots, output);
}

__device__ __forceinline__ void runSelectedSparseTile(
    nta::abi::RuntimeView *runtime, const AttentionPageDescriptor *catalog,
    const __half *queries, std::uint32_t taskIndex, std::uint32_t selectedCount,
    std::uint32_t topK, const nta::abi::WorkItem *workItems,
    const nta::abi::AcquireRequirement *requirements,
    const std::uint32_t *selectedObjectSlots, AttentionTilePartial *partials) {
  if (taskIndex >= selectedCount || topK == 0) {
    return;
  }
  const std::uint32_t catalogIndex = selectedObjectSlots[taskIndex];
  const AttentionPageDescriptor descriptor = catalog[catalogIndex];
  const AttentionTileTask task{
      descriptor.objectSlot,
      taskIndex / topK,
      descriptor.tokenCount,
      0,
  };
  nta::kernel::WorkContext work{};
  const bool ready = nta::kernel::acquireWork(runtime, workItems, requirements,
                                              taskIndex, work);
  if (!ready) {
    nta::kernel::defer(runtime, work);
    return;
  }
  const auto *page =
      static_cast<const __half *>(nta::kernel::address(runtime, work, 0));
  if (page == nullptr) {
    nta::device::failWorkTicket(runtime, work.item.workTicket,
                                nta::abi::WorkTicketState::Failed);
    return;
  }
  computeAttentionTile(runtime, task, taskIndex, work.item, page, queries,
                       partials);
}

__device__ __forceinline__ void
runAttentionTile(nta::abi::RuntimeView *runtime, const AttentionTileTask *tasks,
                 const nta::abi::WorkItem *workItems,
                 const nta::abi::AcquireRequirement *requirements,
                 std::uint32_t taskIndex, const __half *queries,
                 AttentionTilePartial *partials) {
  const AttentionTileTask task = tasks[taskIndex];
  nta::kernel::WorkContext work{};
  const bool ready = nta::kernel::acquireWork(runtime, workItems, requirements,
                                              taskIndex, work);
  if (!ready) {
    nta::kernel::defer(runtime, work);
    return;
  }
  const auto *page =
      static_cast<const __half *>(nta::kernel::address(runtime, work, 0));
  if (page == nullptr) {
    nta::device::failWorkTicket(runtime, work.item.workTicket,
                                nta::abi::WorkTicketState::Failed);
    return;
  }
  computeAttentionTile(runtime, task, taskIndex, work.item, page, queries,
                       partials);
}

__device__ __forceinline__ void runAttentionTileTma(
    nta::abi::RuntimeView *runtime, const AttentionTileTask *tasks,
    const nta::abi::WorkItem *workItems,
    const nta::abi::AcquireRequirement *requirements, std::uint32_t taskIndex,
    const __half *queries, AttentionTilePartial *partials) {
  const AttentionTileTask task = tasks[taskIndex];
  nta::kernel::WorkContext work{};
  const bool ready = nta::kernel::acquireWork(runtime, workItems, requirements,
                                              taskIndex, work);
  if (!ready) {
    nta::kernel::defer(runtime, work);
    return;
  }
  const void *descriptor = nta::kernel::tensorMap(runtime, work, 0);
  if (descriptor == nullptr) {
    nta::device::failWorkTicket(runtime, work.item.workTicket,
                                nta::abi::WorkTicketState::Failed);
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
  computeAttentionTile(runtime, task, taskIndex, work.item, page, queries,
                       partials);
}

__device__ __forceinline__ void
copySparsePage(const AttentionPageDescriptor &descriptor) {
  if ((descriptor.flags & AttentionPageNeedsStaging) == 0 ||
      descriptor.sourceBase == 0 || descriptor.consumeBase == 0) {
    return;
  }
  const auto *source = reinterpret_cast<const uint4 *>(descriptor.sourceBase);
  auto *destination = reinterpret_cast<uint4 *>(descriptor.consumeBase);
  const std::uint32_t vectors = descriptor.bytes / sizeof(uint4);
  for (std::uint32_t vector = threadIdx.x; vector < vectors;
       vector += blockDim.x) {
    nta::device::storeNoAllocate(destination + vector,
                                 nta::device::loadNoAllocate(source + vector));
  }
  auto *destinationBytes =
      reinterpret_cast<std::byte *>(descriptor.consumeBase);
  const auto *sourceBytes =
      reinterpret_cast<const std::byte *>(descriptor.sourceBase);
  for (std::uint32_t byte = vectors * sizeof(uint4) + threadIdx.x;
       byte < descriptor.bytes; byte += blockDim.x) {
    destinationBytes[byte] = sourceBytes[byte];
  }
}

} // namespace

extern "C" __global__ void nta_attention_tile_kernel(
    nta::abi::RuntimeView *runtime, const AttentionTileTask *tasks,
    const nta::abi::WorkItem *workItems, std::uint32_t taskCount,
    const nta::abi::AcquireRequirement *requirements, const __half *queries,
    AttentionTilePartial *partials) {
  const std::uint32_t taskIndex = blockIdx.x;
  if (blockDim.x != AttentionHeadDimension || taskIndex >= taskCount) {
    return;
  }
  runAttentionTile(runtime, tasks, workItems, requirements, taskIndex, queries,
                   partials);
}

extern "C" __global__ void nta_attention_ready_kernel(
    nta::abi::RuntimeView *runtime, const AttentionTileTask *tasks,
    const nta::abi::WorkItem *workItems, std::uint32_t taskCount,
    const nta::abi::AcquireRequirement *requirements, const __half *queries,
    AttentionTilePartial *partials) {
  if (blockDim.x != AttentionHeadDimension) {
    return;
  }
  const std::uint32_t workTicket = popReadyWorkTicket(runtime);
  if (workTicket >= runtime->workTicketCapacity) {
    return;
  }
  // The runnable queue stores canonical WorkItem indices. logicalTile is a
  // framework semantic coordinate and may be request-local (FlashInfer KV
  // tile indices restart at zero for each request), so it cannot index this
  // batch-global task array.
  const std::uint32_t taskIndex = workTicket;
  if (taskIndex < taskCount) {
    runAttentionTile(runtime, tasks, workItems, requirements, taskIndex,
                     queries, partials);
  }
}

extern "C" __global__ void nta_attention_tma_tile_kernel(
    nta::abi::RuntimeView *runtime, const AttentionTileTask *tasks,
    const nta::abi::WorkItem *workItems, std::uint32_t taskCount,
    const nta::abi::AcquireRequirement *requirements, const __half *queries,
    AttentionTilePartial *partials) {
  const std::uint32_t taskIndex = blockIdx.x;
  if (blockDim.x != AttentionHeadDimension || taskIndex >= taskCount) {
    return;
  }
  runAttentionTileTma(runtime, tasks, workItems, requirements, taskIndex,
                      queries, partials);
}

extern "C" __global__ void nta_attention_tma_ready_kernel(
    nta::abi::RuntimeView *runtime, const AttentionTileTask *tasks,
    const nta::abi::WorkItem *workItems, std::uint32_t taskCount,
    const nta::abi::AcquireRequirement *requirements, const __half *queries,
    AttentionTilePartial *partials) {
  if (blockDim.x != AttentionHeadDimension) {
    return;
  }
  const std::uint32_t workTicket = popReadyWorkTicket(runtime);
  if (workTicket >= runtime->workTicketCapacity) {
    return;
  }
  const std::uint32_t taskIndex = workTicket;
  if (taskIndex < taskCount) {
    runAttentionTileTma(runtime, tasks, workItems, requirements, taskIndex,
                        queries, partials);
  }
}

extern "C" __global__ void nta_sparse_query_kernel(const __half *hidden,
                                                   __half *queries,
                                                   std::uint32_t requestCount) {
  const std::uint32_t requestIndex = blockIdx.x;
  const std::uint32_t dimension = threadIdx.x;
  if (requestIndex >= requestCount || dimension >= AttentionHeadDimension) {
    return;
  }
  const std::size_t base =
      static_cast<std::size_t>(requestIndex) * AttentionHeadDimension;
  const std::uint32_t neighbor = (dimension + 17U) % AttentionHeadDimension;
  const float projected = 1.375F * __half2float(hidden[base + dimension]) +
                          0.625F * __half2float(hidden[base + neighbor]) +
                          0.001F * static_cast<float>(requestIndex + 1U);
  queries[base + dimension] = __float2half(tanhf(projected));
}

extern "C" __global__ void nta_sparse_select_kernel(
    const AttentionPageDescriptor *catalog,
    const std::uint32_t *candidateOffsets, const __half *summaries,
    const __half *queries, std::uint32_t requestCount, std::uint32_t topK,
    nta::abi::WorkItem *workItems, nta::abi::AcquireRequirement *requirements,
    std::uint32_t *selectedObjectSlots, std::uint32_t preacquired,
    std::uint32_t splitWork) {
  const std::uint32_t requestIndex = blockIdx.x;
  if (blockDim.x != AttentionHeadDimension || requestIndex >= requestCount ||
      topK == 0 || topK > SparseTopKLimit ||
      candidateOffsets[requestIndex + 1U] - candidateOffsets[requestIndex] <
          topK) {
    return;
  }
  selectSparsePages(catalog, candidateOffsets, summaries, queries, requestIndex,
                    topK, workItems, requirements, selectedObjectSlots,
                    preacquired != 0, splitWork != 0);
}

extern "C" __global__ void nta_sparse_selected_consume_kernel(
    nta::abi::RuntimeView *runtime, const AttentionPageDescriptor *catalog,
    const __half *queries, std::uint32_t requestCount, std::uint32_t topK,
    nta::abi::WorkItem *workItems, nta::abi::AcquireRequirement *requirements,
    const std::uint32_t *selectedObjectSlots, float *output) {
  if (blockDim.x == AttentionHeadDimension) {
    runSelectedSparseAttention(runtime, catalog, queries, blockIdx.x,
                               requestCount, topK, workItems, requirements,
                               selectedObjectSlots, output);
  }
}

extern "C" __global__ void nta_sparse_partial_kernel(
    nta::abi::RuntimeView *runtime, const AttentionPageDescriptor *catalog,
    const __half *queries, std::uint32_t selectedCount, std::uint32_t topK,
    const nta::abi::WorkItem *workItems,
    const nta::abi::AcquireRequirement *requirements,
    const std::uint32_t *selectedObjectSlots, AttentionTilePartial *partials) {
  if (blockDim.x == AttentionHeadDimension) {
    runSelectedSparseTile(runtime, catalog, queries, blockIdx.x, selectedCount,
                          topK, workItems, requirements, selectedObjectSlots,
                          partials);
  }
}

extern "C" __global__ void nta_sparse_partial_ready_kernel(
    nta::abi::RuntimeView *runtime, const AttentionPageDescriptor *catalog,
    const __half *queries, std::uint32_t selectedCount, std::uint32_t topK,
    const nta::abi::WorkItem *workItems,
    const nta::abi::AcquireRequirement *requirements,
    const std::uint32_t *selectedObjectSlots, AttentionTilePartial *partials) {
  if (blockDim.x != AttentionHeadDimension) {
    return;
  }
  const std::uint32_t workTicket = popReadyWorkTicket(runtime);
  if (workTicket >= runtime->workTicketCapacity) {
    return;
  }
  const std::uint32_t taskIndex = workTicket;
  runSelectedSparseTile(runtime, catalog, queries, taskIndex, selectedCount,
                        topK, workItems, requirements, selectedObjectSlots,
                        partials);
}

extern "C" __global__ void
nta_sparse_copy_selected_kernel(const AttentionPageDescriptor *catalog,
                                const std::uint32_t *selectedCatalogIndices,
                                std::uint32_t selectedCount) {
  const std::uint32_t selectedIndex = blockIdx.x;
  if (selectedIndex >= selectedCount) {
    return;
  }
  const std::uint32_t catalogIndex = selectedCatalogIndices[selectedIndex];
  copySparsePage(catalog[catalogIndex]);
}

extern "C" __global__ void nta_sparse_attention_kernel(
    nta::abi::RuntimeView *runtime, const AttentionPageDescriptor *catalog,
    const std::uint32_t *candidateOffsets, const __half *summaries,
    const __half *queries, std::uint32_t requestCount, std::uint32_t topK,
    nta::abi::WorkItem *workItems, nta::abi::AcquireRequirement *requirements,
    std::uint32_t *selectedObjectSlots, float *output,
    std::uint32_t preacquired) {
  if (blockDim.x == AttentionHeadDimension) {
    runSparseAttention(runtime, catalog, candidateOffsets, summaries, queries,
                       blockIdx.x, requestCount, topK, workItems, requirements,
                       selectedObjectSlots, output, preacquired != 0);
  }
}

extern "C" __global__ void nta_sparse_attention_ready_kernel(
    nta::abi::RuntimeView *runtime, const AttentionPageDescriptor *catalog,
    const std::uint32_t *candidateOffsets, const __half *summaries,
    const __half *queries, std::uint32_t requestCount, std::uint32_t topK,
    nta::abi::WorkItem *workItems, nta::abi::AcquireRequirement *requirements,
    std::uint32_t *selectedObjectSlots, float *output,
    std::uint32_t preacquired) {
  (void)candidateOffsets;
  (void)summaries;
  (void)preacquired;
  if (blockDim.x != AttentionHeadDimension) {
    return;
  }
  const std::uint32_t workTicket = popReadyWorkTicket(runtime);
  if (workTicket >= runtime->workTicketCapacity) {
    return;
  }
  const std::uint32_t requestIndex = workTicket;
  if (requestIndex < requestCount) {
    runSelectedSparseAttention(runtime, catalog, queries, requestIndex,
                               requestCount, topK, workItems, requirements,
                               selectedObjectSlots, output);
  }
}

extern "C" __global__ void
nta_sparse_copy_all_kernel(const AttentionPageDescriptor *catalog,
                           std::uint32_t candidateCount) {
  const std::uint32_t candidate = blockIdx.x;
  if (candidate >= candidateCount) {
    return;
  }
  copySparsePage(catalog[candidate]);
}

extern "C" __global__ void
nta_sparse_invalidate_staging_kernel(nta::abi::RuntimeView *runtime,
                                     const AttentionPageDescriptor *catalog,
                                     std::uint32_t candidateCount) {
  const std::uint32_t candidate = blockIdx.x * blockDim.x + threadIdx.x;
  if (candidate >= candidateCount) {
    return;
  }
  const AttentionPageDescriptor descriptor = catalog[candidate];
  if ((descriptor.flags & AttentionPageNeedsStaging) != 0 &&
      descriptor.objectSlot < runtime->objectCapacity) {
    atomicExch(&runtime->objects[descriptor.objectSlot].state,
               static_cast<std::uint32_t>(nta::abi::ObjectState::New));
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
    maximum = fmaxf(maximum, partial.lse);
  }
  if (!ready) {
    return;
  }

#if defined(NTA_USE_FLASHINFER_STATE)
  flashinfer::state_t<1> state;
  for (std::uint32_t tile = 0; tile < request.tileCount; ++tile) {
    const AttentionTilePartial &partial = partials[request.tileBegin + tile];
    flashinfer::vec_t<float, 1> value;
    value[0] = partial.output[threadIdx.x];
    state.merge(value, partial.lse, 1.0F);
  }
  state.normalize();
  output[requestIndex * AttentionHeadDimension + threadIdx.x] = state.o[0];
#else
  float denominator = 0.0F;
  float numerator = 0.0F;
  for (std::uint32_t tile = 0; tile < request.tileCount; ++tile) {
    const AttentionTilePartial &partial = partials[request.tileBegin + tile];
    const float scale = exp2f(partial.lse - maximum);
    denominator += scale;
    numerator += scale * partial.output[threadIdx.x];
  }
  output[requestIndex * AttentionHeadDimension + threadIdx.x] =
      numerator / denominator;
#endif
}
