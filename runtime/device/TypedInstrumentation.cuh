#pragma once

#include <cstdint>

#if !defined(__CUDACC__)
#error "TypedInstrumentation.cuh requires CUDA compilation"
#endif

// The frontend supplies these values when it opts into the typed operator
// contract. They are deliberately emitted as device constants: the LLVM pass
// can validate the contract while compiling the same module that exports the
// JIT ABI, so a runtime-side claim cannot be detached from the instrumented
// code that consumes it.
#ifndef NTA_TYPED_OPERATOR_CONTRACT
#define NTA_TYPED_OPERATOR_CONTRACT 0
#endif
#ifndef NTA_OPERATOR_INSTRUMENTATION_FLAGS
#define NTA_OPERATOR_INSTRUMENTATION_FLAGS 0ULL
#endif
#ifndef NTA_OPERATOR_IDENTITY_BINDING
#define NTA_OPERATOR_IDENTITY_BINDING 0U
#endif
#ifndef NTA_OPERATOR_DEMAND_BINDING
#define NTA_OPERATOR_DEMAND_BINDING 0U
#endif
#ifndef NTA_OPERATOR_ACCESS_PROOF
#define NTA_OPERATOR_ACCESS_PROOF 0U
#endif
#ifndef NTA_OPERATOR_GRANULARITY_BYTES
#define NTA_OPERATOR_GRANULARITY_BYTES 0U
#endif
#ifndef NTA_OPERATOR_TIER_MASK
#define NTA_OPERATOR_TIER_MASK 0ULL
#endif

#if NTA_TYPED_OPERATOR_CONTRACT
extern "C" __device__ __constant__ __attribute__((used)) const
    std::uint64_t nta_jit_instrumentation_flags =
        NTA_OPERATOR_INSTRUMENTATION_FLAGS;
extern "C" __device__ __constant__ __attribute__((used)) const
    std::uint32_t nta_jit_identity_binding = NTA_OPERATOR_IDENTITY_BINDING;
extern "C" __device__ __constant__ __attribute__((used)) const
    std::uint32_t nta_jit_demand_binding = NTA_OPERATOR_DEMAND_BINDING;
extern "C" __device__ __constant__ __attribute__((used)) const
    std::uint32_t nta_jit_access_proof = NTA_OPERATOR_ACCESS_PROOF;
extern "C" __device__ __constant__ __attribute__((used)) const
    std::uint32_t nta_jit_granularity_bytes = NTA_OPERATOR_GRANULARITY_BYTES;
extern "C" __device__ __constant__ __attribute__((used)) const
    std::uint64_t nta_jit_tier_mask = NTA_OPERATOR_TIER_MASK;
#endif
