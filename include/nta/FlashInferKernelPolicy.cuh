#pragma once

#include "nta/KernelPolicy.cuh"

#include <cstdint>
#include <type_traits>
#include <utility>

namespace nta::flashinfer {

template <typename Params, typename = void>
struct HasWorkPlan : std::false_type {};

template <typename Params>
struct HasWorkPlan<
    Params,
    std::void_t<decltype(std::declval<const Params &>().nta_runtime),
                decltype(std::declval<const Params &>().nta_work_items),
                decltype(std::declval<const Params &>().nta_dependencies),
                decltype(std::declval<const Params &>().nta_work_count)>>
    : std::true_type {};

template <typename Params>
inline constexpr bool HasWorkPlanV = HasWorkPlan<Params>::value;

// FlashInfer's x scheduler coordinate is canonical work. All y-dimension KV
// head CTAs share that work item; initializeContinuation arbitrates duplicate
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
    return item.requestIndex == requestIndex &&
           item.continuation < runtime->continuationCapacity;
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
  const auto state = static_cast<abi::ContinuationState>(
      atomicAdd(&runtime->continuations[work.item.continuation].state, 0U));
  return state == abi::ContinuationState::New ||
         state == abi::ContinuationState::Ready;
}

} // namespace nta::flashinfer
