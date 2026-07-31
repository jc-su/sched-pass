#pragma once

#include "nta/RuntimeABI.h"

#if !defined(__CUDACC__)
#error "nta/DeviceAPI.cuh requires CUDA compilation"
#endif

extern "C" __device__ void __nta_bind_request(std::uint32_t requestSlot,
                                              std::uint32_t generation);

extern "C" __device__ void *
__nta_acquire_marker(nta::abi::RuntimeView *runtime, const void *directBase,
                     std::uint32_t objectSlot, std::uint64_t objectId,
                     std::uint32_t objectVersion, std::uint64_t offset,
                     std::uint32_t bytes, std::uint32_t continuation);

extern "C" __device__ void __nta_defer_marker(nta::abi::RuntimeView *runtime,
                                              std::uint32_t continuation);
