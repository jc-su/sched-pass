#pragma once

// Clang CUDA does not provide nvcc's unqualified host/device min/max overloads
// used by several JIT kernel libraries. Exact CUDA device overloads still win
// overload resolution; these templates cover host and mixed-type calls.
#if defined(__CUDACC__)
#include <type_traits>

template <class A, class B,
          class = std::enable_if_t<std::is_arithmetic<A>::value &&
                                   std::is_arithmetic<B>::value>>
__host__ __device__ constexpr std::common_type_t<A, B> min(A a, B b) {
  using C = std::common_type_t<A, B>;
  return static_cast<C>(b) < static_cast<C>(a) ? static_cast<C>(b)
                                               : static_cast<C>(a);
}

template <class A, class B,
          class = std::enable_if_t<std::is_arithmetic<A>::value &&
                                   std::is_arithmetic<B>::value>>
__host__ __device__ constexpr std::common_type_t<A, B> max(A a, B b) {
  using C = std::common_type_t<A, B>;
  return static_cast<C>(a) < static_cast<C>(b) ? static_cast<C>(b)
                                               : static_cast<C>(a);
}
#endif

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
__device__ __forceinline__ float2 __ffma2_rn(float2 a, float2 b, float2 c) {
  return make_float2(__fmaf_rn(a.x, b.x, c.x),
                     __fmaf_rn(a.y, b.y, c.y));
}
#endif
