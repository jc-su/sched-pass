#pragma once

#include "nta/RuntimeABI.h"

#if !defined(__CUDACC__)
#error "nta/DeviceAPI.cuh requires CUDA compilation"
#endif

extern "C" __device__ void __nta_bind_request(std::uint32_t requestSlot,
                                              std::uint32_t generation);

// Generic compiler-visible sites are readiness guards. A miss must have been
// published by typed discovery; it cannot create a shared group from inside a
// numerical grid because no complete-fanout seal exists there.
extern "C" __device__ void *
__nta_acquire_marker(nta::abi::RuntimeView *runtime, const void *directBase,
                     std::uint32_t objectSlot, std::uint64_t objectId,
                     std::uint32_t objectVersion, std::uint64_t offset,
                     std::uint32_t bytes, std::uint32_t workTicket);

extern "C" __device__ void *__nta_acquire_tensor_map_marker(
    nta::abi::RuntimeView *runtime, const void *directTensorMap,
    std::uint32_t objectSlot, std::uint64_t objectId,
    std::uint32_t objectVersion, std::uint64_t offset, std::uint32_t bytes,
    std::uint32_t workTicket);

// Explicit online-miss entry point. The caller must provide
// AcquireOnlineExclusive and guarantee that the object generation has exactly
// one external consumer work ticket in the current acquisition epoch.
extern "C" __device__ void *nta_acquire_exclusive_slow(
    nta::abi::RuntimeView *runtime, std::uint32_t requestSlot,
    std::uint32_t generation, std::uint32_t objectSlot,
    std::uint64_t objectId, std::uint32_t objectVersion, std::uint64_t offset,
    std::uint32_t bytes, std::uint32_t workTicket,
    std::uint32_t requirementFlags);

// Collective CTA operation. True means every requirement can be consumed;
// false must lead to exactly one __nta_defer_marker call and a kernel return.
extern "C" __device__ bool
__nta_acquire_set_marker(nta::abi::RuntimeView *runtime,
                         const nta::abi::AcquireRequirement *requirements,
                         std::uint32_t requirementCount,
                         std::uint32_t directRequirementCount,
                         std::uint32_t workTicket);

extern "C" __device__ const void *
nta_requirement_address(nta::abi::RuntimeView *runtime,
                        const nta::abi::AcquireRequirement *requirement);

extern "C" __device__ const void *
nta_requirement_tensor_map(nta::abi::RuntimeView *runtime,
                           const nta::abi::AcquireRequirement *requirement);

extern "C" __device__ void __nta_defer_marker(nta::abi::RuntimeView *runtime,
                                              std::uint32_t workTicket);

// Delimits the numerical region that produces one durable partial. The marker
// is placed after data and ticket-state filters, so skipped CTAs do not acquire
// a publication obligation.
extern "C" __device__ __attribute__((convergent)) void
__nta_begin_partial_marker(nta::abi::RuntimeView *runtime,
                           std::uint32_t workTicket);

// Collective tail effect for an incremental CTA. Every thread reaches the
// marker after the partial output is visible. The compiler proves that this
// publication post-dominates the acquired path and binds request identity.
extern "C" __device__ __attribute__((convergent)) void
__nta_commit_partial_marker(nta::abi::RuntimeView *runtime,
                            std::uint32_t workTicket,
                            std::uint32_t reductionGroup,
                            std::uint32_t contributorIndex,
                            std::uint32_t contributorCount,
                            std::uint64_t estimatedComputeNs);

// Exact publication for a finite kernel whose consumer is ordered after the
// complete launch on the same CUDA stream or graph path.
extern "C" __device__ __attribute__((convergent)) void
__nta_commit_stream_ordered_partial_marker(
    nta::abi::RuntimeView *runtime, std::uint32_t workTicket,
    std::uint32_t reductionGroup, std::uint32_t contributorIndex,
    std::uint32_t contributorCount, std::uint64_t estimatedComputeNs);
