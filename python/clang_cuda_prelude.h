// clang_cuda_prelude.h -- force-included (clang -include) before FlashInfer's
// headers to close the nvcc-vs-clang dialect gap that blocks compiling
// FlashInfer's CUDA under clang (our LLVM pass plugin only runs under clang,
// and this node's nvcc predates compute_120a anyway).
//
// The gap: nvcc exposes UNQUALIFIED global min()/max() as host+device builtins
// for all arithmetic types (same and mixed). FlashInfer calls e.g.
//   min(num_blocks_per_sm, ceil_div(int(nnz), num_sms))   // both int, HOST launch
//   min((kv_tile_idx+1)*max_chunk_size, kv_len)           // mixed, __device__
// clang has std::min but no global ::min, and CUDA's ::min are __device__-only,
// so the host launch code and mixed-type calls fail to resolve.
//
// Fix: provide global __host__ __device__ templated min/max over arithmetic
// types. Overload resolution keeps this SAFE with no ambiguity:
//   * DEVICE call, exact type (min(int,int)): CUDA's own non-template __device__
//     min beats this template -> CUDA's is used, device codegen unchanged.
//   * DEVICE call, mixed type: no exact CUDA overload -> this template fills it.
//   * HOST call (any arithmetic types): CUDA's __device__ min is not viable
//     from host -> this template is the only candidate.
// clang parses host functions in BOTH passes, so the template must exist
// unconditionally (guarding by __CUDA_ARCH__ would hide it from host functions
// during the device pass -- the bug this file's history records).
#pragma once

#if defined(__CUDACC__)
#include <type_traits>

template <class A, class B,
          class = std::enable_if_t<std::is_arithmetic<A>::value &&
                                   std::is_arithmetic<B>::value>>
__host__ __device__ constexpr std::common_type_t<A, B> min(A a, B b) {
  using C = std::common_type_t<A, B>;
  return (static_cast<C>(b) < static_cast<C>(a)) ? static_cast<C>(b)
                                                 : static_cast<C>(a);
}
template <class A, class B,
          class = std::enable_if_t<std::is_arithmetic<A>::value &&
                                   std::is_arithmetic<B>::value>>
__host__ __device__ constexpr std::common_type_t<A, B> max(A a, B b) {
  using C = std::common_type_t<A, B>;
  return (static_cast<C>(a) < static_cast<C>(b)) ? static_cast<C>(b)
                                                 : static_cast<C>(a);
}
#endif

// nvcc float2 SIMD intrinsics absent from clang's CUDA headers (FlashInfer
// norm/fused_dit_layernorm.cuh). Scalar-composed with matching rounding.
#if defined(__clang__) && defined(__CUDA__)
__device__ __forceinline__ float2 __fadd2_rn(float2 a, float2 b) {
  return make_float2(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y));
}
__device__ __forceinline__ float2 __fmul2_rn(float2 a, float2 b) {
  return make_float2(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y));
}
__device__ __forceinline__ float2 __fsub2_rn(float2 a, float2 b) {
  return make_float2(__fsub_rn(a.x, b.x), __fsub_rn(a.y, b.y));
}
// fused multiply-add, per-lane round-to-nearest (nvcc maps this to the
// packed fma.rn.f32x2 on sm_100+; per-lane __fmaf_rn is bit-identical).
__device__ __forceinline__ float2 __ffma2_rn(float2 a, float2 b, float2 c) {
  return make_float2(__fmaf_rn(a.x, b.x, c.x), __fmaf_rn(a.y, b.y, c.y));
}
#endif
