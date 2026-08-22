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
inline constexpr std::uint64_t RunnableOffsetShift = 32;
inline constexpr std::uint64_t WorkCountMask = 0xffffffffULL;

[[nodiscard]] constexpr std::uint64_t
packWorkMetadata(std::uint32_t workCount, std::uint32_t requestCount) {
  return (static_cast<std::uint64_t>(requestCount) << 32U) | workCount;
}

template <typename Params>
[[nodiscard]] __host__ __device__ __forceinline__ std::uint32_t
workCount(const Params &params) {
  return static_cast<std::uint32_t>(
      static_cast<std::uint64_t>(params.nta_work_count) & WorkCountMask);
}

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
struct HasRequestBinding : std::false_type {};

template <typename Params>
struct HasRequestBinding<
    Params,
    std::void_t<decltype(std::declval<const Params &>().nta_runtime),
                decltype(std::declval<const Params &>().nta_request_slot_offset)>>
    : std::true_type {};

template <typename Params>
inline constexpr bool HasRequestBindingV = HasRequestBinding<Params>::value;

template <typename Params, typename = void>
struct HasPagedBatchSize : std::false_type {};

template <typename Params>
struct HasPagedBatchSize<
    Params,
    std::void_t<decltype(std::declval<const Params &>().paged_kv.batch_size)>>
    : std::true_type {};

template <typename Params>
[[nodiscard]] inline std::uint32_t requestGroupCount(const Params &params) {
  if constexpr (HasWorkPlanV<Params>) {
    const std::uint32_t encoded = static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(params.nta_work_count) >> 32U);
    if (encoded != 0) {
      return encoded;
    }
  }
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
        workCount(params) == 0) {
      return false;
    }
#if !NTA_FLASHINFER_STREAM_ORDERED_DIRECT
    if (params.nta_dependencies == nullptr) {
      return false;
    }
#endif
    auto *runtime = reinterpret_cast<abi::RuntimeView *>(params.nta_runtime);
    const bool runnable =
        (static_cast<std::uint64_t>(params.nta_skip_merge) & RunnableWork) != 0;
    const std::uint64_t schedulerLimit =
        runnable ? runtime->workTicketCapacity
                 : static_cast<std::uint64_t>(workCount(params));
    if (schedulerIndex >= schedulerLimit) {
      return false;
    }
    const auto *items =
        reinterpret_cast<const abi::WorkItem *>(params.nta_work_items);
    const abi::WorkItem &item = items[schedulerIndex];
    const bool baseValid =
           runtime != nullptr && runtime->abiVersion == abi::Version &&
           item.requestIndex == requestIndex &&
           item.requestSlot < runtime->requestCapacity &&
           item.workTicket < runtime->workTicketCapacity &&
           item.reductionGroup < runtime->workTicketCapacity;
#if NTA_FLASHINFER_STREAM_ORDERED_DIRECT
    return baseValid;
#else
    return baseValid &&
           item.contributorCount != 0 &&
           item.contributorIndex < item.contributorCount;
#endif
  }
}

template <typename Params>
[[nodiscard]] __device__ __forceinline__ abi::RuntimeView *
runtime(const Params &params) {
  if constexpr (HasWorkPlanV<Params> || HasRequestBindingV<Params>) {
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

// Compile-time request-bound kernels retain NTA's per-request generation guard
// but deliberately cannot discover or issue external dependencies. They are
// valid only after stream-ordered acquisition has completed.
template <typename Params>
[[nodiscard]] __device__ __forceinline__ bool
validRequestBoundLaunch(const Params &params, abi::RuntimeView *runtime) {
  if constexpr (!HasRequestBindingV<Params>) {
    (void)params;
    (void)runtime;
    return false;
  } else {
    return runtime != nullptr && runtime->abiVersion == abi::Version &&
           runtime->requests != nullptr;
  }
}

template <typename Params>
[[nodiscard]] __device__ __forceinline__ bool
bindValidatedRequestOnly(const Params &params, abi::RuntimeView *runtime,
                         std::uint32_t requestIndex) {
  if constexpr (!HasRequestBindingV<Params>) {
    (void)params;
    (void)runtime;
    (void)requestIndex;
    return false;
  } else {
    const std::uint64_t requestSlot64 =
        static_cast<std::uint64_t>(params.nta_request_slot_offset) + requestIndex;
    if (requestSlot64 >= runtime->requestCapacity) {
      return false;
    }
    const std::uint32_t requestSlot = static_cast<std::uint32_t>(requestSlot64);
    const std::uint32_t generation = runtime->requests[requestSlot].generation;
    __nta_bind_request(requestSlot, generation);
    return __nta_acquire_set_marker(runtime, nullptr, 0, 0, abi::InvalidIndex);
  }
}

// Field-presence trait for claim-consumer params: kernels whose Params
// carry the lease identity triple participate in the in-kernel claim
// contract; kernels without it keep the request-only guard.
template <typename Params, typename = void>
struct HasClaimBinding : std::false_type {};
template <typename Params>
struct HasClaimBinding<
    Params, std::void_t<decltype(Params::nta_claim_slot),
                        decltype(Params::nta_claim_generation),
                        decltype(Params::nta_claim_row_bound),
                        decltype(Params::nta_table_stamp)>> : std::true_type {};
template <typename Params>
inline constexpr bool HasClaimBindingV = HasClaimBinding<Params>::value;

// In-kernel claim-consumer contract (ABI v28). CTA-uniform, evaluated
// before the mainloop touches lease storage: the consumer proves it is
// reading its own live claim — slot in range, generation matched, row
// valid (retirement republishes valid=0 under the same generation, and
// cancellation flows through retirement), and the selected table stamped
// by the prep launch this consumer was planned against. Any mismatch
// refuses the launch; fail-closed, never a fallback read. Extent
// checking is enforced by the prep kernel writing only inside
// [leaseBase, leaseBase+leaseExtent) and stagedRows bounding row
// indices; the consumer re-checks stagedRows against its plan bound.
template <typename Params>
[[nodiscard]] __device__ __forceinline__ bool
bindValidatedClaimConsumer(const Params &params, abi::RuntimeView *runtime,
                           std::uint32_t plannedRowBound) {
  if constexpr (!HasClaimBindingV<Params>) {
    (void)params;
    (void)runtime;
    (void)plannedRowBound;
    return false;
  } else {
    if (runtime == nullptr || runtime->abiVersion != abi::Version ||
        runtime->claims == nullptr) {
      return false;
    }
    const std::uint32_t slot =
        static_cast<std::uint32_t>(params.nta_claim_slot);
    if (slot >= runtime->claimCapacity) {
      return false;
    }
    const abi::ClaimContext &claim = runtime->claims[slot];
    if (claim.generation !=
            static_cast<std::uint32_t>(params.nta_claim_generation) ||
        claim.valid == 0u) {
      return false;
    }
    if (plannedRowBound > claim.stagedRows) {
      return false;
    }
    if (claim.tableStamp !=
        static_cast<std::uint64_t>(params.nta_table_stamp)) {
      return false;
    }
    return true;
  }
}

template <typename Params>
[[nodiscard]] __host__ __device__ __forceinline__ bool
usesRunnableWork(const Params &params) {
  if constexpr (HasWorkPlanV<Params>) {
    return (static_cast<std::uint64_t>(params.nta_skip_merge) & RunnableWork) !=
           0;
  } else {
    (void)params;
    return false;
  }
}

template <typename Params>
[[nodiscard]] __host__ __device__ __forceinline__ std::uint32_t
runnableWorkOffset(const Params &params) {
  if constexpr (HasWorkPlanV<Params>) {
    return static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(params.nta_skip_merge) >>
        RunnableOffsetShift);
  } else {
    (void)params;
    return 0;
  }
}

template <typename Params>
[[nodiscard]] __host__ __device__ __forceinline__ std::uint32_t
reductionGroupOffset(const Params &params) {
  return usesRunnableWork(params) ? 0U : runnableWorkOffset(params);
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
      workCount(params) == 0) {
    return abi::InvalidIndex;
  }
  // Publication and this launch are stream ordered. No producer mutates the
  // runnable-work set while an application kernel consumes it.
  const std::uint32_t ready = *runtime->readyCount;
  const std::uint32_t offset = runnableWorkOffset(params);
  const std::uint64_t queueIndex =
      static_cast<std::uint64_t>(offset) + launchIndex;
  if (offset > ready || queueIndex >= ready ||
      launchIndex >= static_cast<std::uint64_t>(workCount(params))) {
    return abi::InvalidIndex;
  }
  const std::uint32_t workIndex = runtime->readyWorkTickets[queueIndex];
  return workIndex < runtime->workTicketCapacity
             ? workIndex
             : abi::InvalidIndex;
}

// In runnable mode nta_work_count is a conservative physical launch bound;
// canonical identity and active count remain device-owned in readyWorkTickets.
// Initial/discovery launches retain the framework's complete scheduler grid.
template <typename Params>
[[nodiscard]] inline std::uint32_t
launchWorkCount(const Params &params, std::uint32_t frameworkWorkCount) {
  if constexpr (HasWorkPlanV<Params>) {
    const std::uint64_t flags =
        static_cast<std::uint64_t>(params.nta_skip_merge);
    const std::uint64_t requested = workCount(params);
    if ((flags & RunnableWork) != 0 && requested != 0) {
      return static_cast<std::uint32_t>(
          requested < frameworkWorkCount ? requested : frameworkWorkCount);
    }
  }
  return frameworkWorkCount;
}

template <typename Params>
[[nodiscard]] __device__ __forceinline__ bool
shouldRun(const Params &params, abi::RuntimeView *runtime,
          const kernel::WorkContext &work) {
  if (!tracksCompletion(params)) {
    return true;
  }
  // Ticket state is mutated by sibling CTAs. Independent global loads can
  // therefore split one CTA around the early return and leave only part of it
  // entering FlashInfer's barrier-bearing numerical body. Snapshot once and
  // broadcast the decision collectively.
  __shared__ std::uint32_t runDecision;
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    runDecision = shouldRun(runtime, work) ? 1U : 0U;
  }
  __syncthreads();
  return runDecision != 0;
}

// FlashInfer maps one canonical x-coordinate to one or more y/z head CTAs.
// Every participating CTA publishes after its output stores. Publication is an
// explicit compiler effect: the pass proves its acquired-path placement and
// lowers it to the generation-checked request-local reduction protocol.
template <typename Params>
__device__ __forceinline__ void finish(const Params &params,
                                       abi::RuntimeView *runtime,
                                       const kernel::WorkContext &work) {
  (void)params;
  kernel::commitPartial(runtime, work);
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
  std::uint32_t reductionGroupOffset;
};

template <typename Params>
[[nodiscard]] inline MergeGate mergeGate(const Params &params) {
#if NTA_FLASHINFER_STREAM_ORDERED_DIRECT
  // The request directory is frozen for this finite graph launch. FlashInfer's
  // own stream order makes every partial visible to its merge kernel; the
  // exact work plan is retired immediately after wrapper completion.
  (void)params;
  return {nullptr, 0};
#else
  if constexpr (HasWorkPlanV<Params>) {
    if (params.nta_runtime != nullptr && workCount(params) != 0 &&
        (static_cast<std::uint64_t>(params.nta_skip_merge) & Preacquired) ==
            0) {
      const auto *runtime =
          reinterpret_cast<const abi::RuntimeView *>(params.nta_runtime);
      return {runtime, reductionGroupOffset(params)};
    }
  }
  return {nullptr, 0};
#endif
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
