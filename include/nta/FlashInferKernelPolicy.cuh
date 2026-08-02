#pragma once

#include "nta/KernelPolicy.cuh"
#include "nta/TicketProtocol.cuh"

#include <cstdint>
#include <type_traits>
#include <utility>

namespace nta::flashinfer {

inline constexpr std::uint64_t SkipMerge = 1ULL << 0;
inline constexpr std::uint64_t Preacquired = 1ULL << 1;
inline constexpr std::uint64_t BindCurrentGeneration = 1ULL << 2;
inline constexpr std::uint64_t PlanlessPreacquired = 1ULL << 3;
inline constexpr std::uint64_t RunnableWork = 1ULL << 4;

template <typename Params, typename = void>
struct HasWorkPlan : std::false_type {};

template <typename Params>
struct HasWorkPlan<
    Params,
    std::void_t<decltype(std::declval<const Params &>().nta_runtime),
                decltype(std::declval<const Params &>().nta_work_items),
                decltype(std::declval<const Params &>().nta_dependencies),
                decltype(std::declval<const Params &>().nta_work_count),
                decltype(std::declval<const Params &>().nta_skip_merge)>>
    : std::true_type {};

template <typename Params>
inline constexpr bool HasWorkPlanV = HasWorkPlan<Params>::value;

template <typename Params, typename = void>
struct HasPagedBatchSize : std::false_type {};

template <typename Params>
struct HasPagedBatchSize<
    Params,
    std::void_t<decltype(std::declval<const Params &>().paged_kv.batch_size)>>
    : std::true_type {};

template <typename Params>
[[nodiscard]] inline std::uint32_t requestGroupCount(const Params &params) {
  if constexpr (HasPagedBatchSize<Params>::value) {
    return static_cast<std::uint32_t>(params.paged_kv.batch_size);
  } else {
    return static_cast<std::uint32_t>(params.padded_batch_size);
  }
}

// FlashInfer's x scheduler coordinate is canonical work. All y-dimension KV
// head CTAs share that work item; initializeWorkTicket arbitrates duplicate
// discovery and a stream-ordered completion phase retires the item only after
// the complete attention launch finishes.
template <typename Params>
[[nodiscard]] __device__ __forceinline__ bool
validWork(const Params &params, std::uint32_t schedulerIndex,
          std::uint32_t requestIndex) {
  if constexpr (!HasWorkPlanV<Params>) {
    (void)params;
    (void)schedulerIndex;
    (void)requestIndex;
    return false;
  } else {
    if (params.nta_runtime == nullptr || params.nta_work_items == nullptr ||
        params.nta_dependencies == nullptr || params.nta_work_count <= 0 ||
        schedulerIndex >= static_cast<std::uint64_t>(params.nta_work_count)) {
      return false;
    }
    auto *runtime = reinterpret_cast<abi::RuntimeView *>(params.nta_runtime);
    const auto *items =
        reinterpret_cast<const abi::WorkItem *>(params.nta_work_items);
    const abi::WorkItem &item = items[schedulerIndex];
    return runtime != nullptr && runtime->abiVersion == abi::Version &&
           item.requestIndex == requestIndex &&
           item.requestSlot < runtime->requestCapacity &&
           item.workTicket < runtime->workTicketCapacity &&
           item.reductionGroup < runtime->workTicketCapacity &&
           item.contributorCount != 0 &&
           item.contributorIndex < item.contributorCount;
  }
}

template <typename Params>
[[nodiscard]] __device__ __forceinline__ abi::RuntimeView *
runtime(const Params &params) {
  if constexpr (HasWorkPlanV<Params>) {
    return reinterpret_cast<abi::RuntimeView *>(params.nta_runtime);
  } else {
    (void)params;
    return nullptr;
  }
}

template <typename Params>
[[nodiscard]] __device__ __forceinline__ const abi::WorkItem *
workItems(const Params &params) {
  return reinterpret_cast<const abi::WorkItem *>(params.nta_work_items);
}

template <typename Params>
[[nodiscard]] __device__ __forceinline__ const abi::AcquireRequirement *
dependencies(const Params &params) {
  return reinterpret_cast<const abi::AcquireRequirement *>(
      params.nta_dependencies);
}

[[nodiscard]] __device__ __forceinline__ bool
shouldRun(abi::RuntimeView *runtime, const kernel::WorkContext &work) {
  abi::WorkTicket &ticket = runtime->workTickets[work.item.workTicket];
  const auto state =
      static_cast<abi::WorkTicketState>(atomicAdd(&ticket.state, 0U));
  return state == abi::WorkTicketState::New ||
         (state == abi::WorkTicketState::Ready &&
          device::ticketMatches(runtime, ticket, work.item.requestSlot,
                                work.item.generation));
}

template <typename Params>
[[nodiscard]] __device__ __forceinline__ bool
tracksCompletion(const Params &params) {
  if constexpr (HasWorkPlanV<Params>) {
    // Bit 1 denotes a stream-ordered, pre-acquired dependency set. Its
    // availability event replaces per-launch ticket reset and retirement.
    return (static_cast<std::uint64_t>(params.nta_skip_merge) & Preacquired) ==
           0;
  } else {
    (void)params;
    return false;
  }
}

template <typename Params>
[[nodiscard]] __device__ __forceinline__ bool
bindsCurrentGeneration(const Params &params) {
  if constexpr (HasWorkPlanV<Params>) {
    // Bit 2 allows an engine to reuse a structural plan while request slots
    // are rebound. validWork has already checked the slot bounds.
    return (static_cast<std::uint64_t>(params.nta_skip_merge) &
            BindCurrentGeneration) != 0;
  } else {
    (void)params;
    return false;
  }
}

template <typename Params>
[[nodiscard]] __device__ __forceinline__ bool
usesPlanlessPreacquired(const Params &params) {
  if constexpr (HasWorkPlanV<Params>) {
    return (static_cast<std::uint64_t>(params.nta_skip_merge) &
            PlanlessPreacquired) != 0;
  } else {
    (void)params;
    return false;
  }
}

template <typename Params>
[[nodiscard]] __device__ __forceinline__ bool
usesRunnableWork(const Params &params) {
  if constexpr (HasWorkPlanV<Params>) {
    return (static_cast<std::uint64_t>(params.nta_skip_merge) & RunnableWork) !=
           0;
  } else {
    (void)params;
    return false;
  }
}

// Runnable work is physically compacted at the front of the fixed framework
// grid. Entries are canonical work-ticket indices, which DeviceWorkPlan keeps
// identical to the upstream scheduler x-coordinate.
template <typename Params>
[[nodiscard]] __device__ __forceinline__ std::uint32_t
launchWorkIndex(const Params &params, abi::RuntimeView *runtime,
                std::uint32_t launchIndex) {
  if (!usesRunnableWork(params)) {
    return launchIndex;
  }
  if (runtime == nullptr || runtime->abiVersion != abi::Version ||
      runtime->readyCount == nullptr || runtime->readyWorkTickets == nullptr ||
      params.nta_work_count <= 0) {
    return abi::InvalidIndex;
  }
  // Publication and this launch are stream ordered. No producer mutates the
  // runnable-work set while an application kernel consumes it.
  const std::uint32_t ready = *runtime->readyCount;
  if (launchIndex >= ready ||
      launchIndex >= static_cast<std::uint64_t>(params.nta_work_count)) {
    return abi::InvalidIndex;
  }
  const std::uint32_t workIndex = runtime->readyWorkTickets[launchIndex];
  return workIndex < static_cast<std::uint64_t>(params.nta_work_count)
             ? workIndex
             : abi::InvalidIndex;
}

template <typename Params>
[[nodiscard]] __device__ __forceinline__ bool
shouldRun(const Params &params, abi::RuntimeView *runtime,
          const kernel::WorkContext &work) {
  return !tracksCompletion(params) || shouldRun(runtime, work);
}

// FlashInfer maps one canonical x-coordinate to one or more y/z head CTAs.
// Every participating CTA calls this after its output stores. The final sibling
// retires the ticket, replacing a stream-ordered full-grid completion launch.
template <typename Params>
__device__ __forceinline__ void finish(const Params &params,
                                       abi::RuntimeView *runtime,
                                       const kernel::WorkContext &work) {
  if (!tracksCompletion(params)) {
    return;
  }
  __syncthreads();
  if (runtime == nullptr || runtime->ctaCompletions == nullptr ||
      work.item.workTicket >= runtime->workTicketCapacity || threadIdx.x != 0 ||
      threadIdx.y != 0 || threadIdx.z != 0) {
    return;
  }
  const std::uint64_t siblingCount64 =
      static_cast<std::uint64_t>(gridDim.y) * gridDim.z;
  if (siblingCount64 == 0 || siblingCount64 > UINT32_MAX) {
    atomicExch(&runtime->workTickets[work.item.workTicket].state,
               static_cast<std::uint32_t>(abi::WorkTicketState::Failed));
    return;
  }
  const std::uint32_t completed =
      atomicAdd(&runtime->ctaCompletions[work.item.workTicket], 1U) + 1U;
  if (completed != static_cast<std::uint32_t>(siblingCount64)) {
    return;
  }
  abi::WorkTicket &ticket = runtime->workTickets[work.item.workTicket];
  const auto state =
      static_cast<abi::WorkTicketState>(atomicAdd(&ticket.state, 0U));
  if (state == abi::WorkTicketState::New) {
    if (!device::requestLive(runtime, work.item.requestSlot,
                             work.item.generation)) {
      device::failWorkTicket(runtime, work.item.workTicket,
                             abi::WorkTicketState::Cancelled);
      return;
    }
    ticket.requestId = runtime->requests[work.item.requestSlot].requestId;
    ticket.requestSlot = work.item.requestSlot;
    ticket.generation = work.item.generation;
    ticket.logicalTile = work.item.workTicket;
    ticket.epoch = device::currentEpoch(runtime);
    ticket.unavailableBytes = 0;
    ticket.estimatedComputeNs = work.item.estimatedComputeNs;
    ticket.reductionGroup = work.item.reductionGroup;
    ticket.contributorCount = work.item.contributorCount;
    __threadfence();
  } else if (!device::ticketMatches(runtime, ticket, work.item.requestSlot,
                                    work.item.generation)) {
    device::failWorkTicket(runtime, work.item.workTicket,
                           abi::WorkTicketState::Failed);
    return;
  }
  (void)device::completeWorkTicket(runtime, work.item.workTicket);
}

[[nodiscard]] __device__ __forceinline__ bool
epochComplete(const abi::RuntimeView *runtime, std::uint32_t expectedWork) {
  return runtime != nullptr && runtime->abiVersion == abi::Version &&
         atomicAdd(const_cast<std::uint32_t *>(&runtime->failedCount), 0U) ==
             0U &&
         atomicAdd(const_cast<std::uint32_t *>(&runtime->completedCount), 0U) ==
             expectedWork;
}

struct MergeGate {
  const abi::RuntimeView *runtime;
};

template <typename Params>
[[nodiscard]] inline MergeGate mergeGate(const Params &params) {
  if constexpr (HasWorkPlanV<Params>) {
    if (params.nta_runtime != nullptr && params.nta_work_count > 0 &&
        (static_cast<std::uint64_t>(params.nta_skip_merge) & Preacquired) ==
            0) {
      const auto *runtime =
          reinterpret_cast<const abi::RuntimeView *>(params.nta_runtime);
      return {runtime};
    }
  }
  return {nullptr};
}

template <typename Params>
[[nodiscard]] inline bool shouldMerge(const Params &params) {
  if constexpr (HasWorkPlanV<Params>) {
    return (static_cast<std::uint64_t>(params.nta_skip_merge) & SkipMerge) == 0;
  } else {
    (void)params;
    return true;
  }
}

} // namespace nta::flashinfer
