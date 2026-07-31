// cache_hint_bw.cu -- EVIDENCE for the SHAPE-is-dead-on-DRAM-bound claim.
// Reads a >L2 array (single-pass, no reuse -- the decode KV access shape) with
// each cache policy and measures achieved DRAM bandwidth. If cache hints cannot
// help a DRAM-bound streaming read, every policy lands at the same bandwidth.
// If any differs, the claim is wrong.
//
//   0 default  ld.global
//   1 cs       ld.global.cs               (streaming / evict-first-ish)
//   2 evict_last  createpolicy.L2::evict_last + ld.L2::cache_hint
//   3 evict_first createpolicy.L2::evict_first + ld.L2::cache_hint
//   4 discard  ld.global then discard.global.L2 (per 128B line)
//
// Build: clang++-22 -x cuda --cuda-gpu-arch=sm_120a --cuda-path=/usr/local/cuda-12.9 \
//        -O3 -std=c++17 test/cache_hint_bw.cu -o build/chbw -lcudart && ./build/chbw
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

template <int POLICY>
__global__ void stream(const float *a, size_t n, float *out) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  unsigned long long pol = 0;
  if (POLICY == 2)
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  if (POLICY == 3)
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  float s = 0.f;
  for (size_t j = i; j < n; j += stride) {
    const float *p = a + j;
    float v;
    if (POLICY == 0)
      asm volatile("ld.global.f32 %0, [%1];" : "=f"(v) : "l"(p));
    else if (POLICY == 1)
      asm volatile("ld.global.cs.f32 %0, [%1];" : "=f"(v) : "l"(p));
    else if (POLICY == 2 || POLICY == 3)
      asm volatile("ld.global.L2::cache_hint.f32 %0, [%1], %2;"
                   : "=f"(v) : "l"(p), "l"(pol));
    else if (POLICY == 4) {
      asm volatile("ld.global.f32 %0, [%1];" : "=f"(v) : "l"(p));
      if ((j & 31) == 0) // one discard per 128B line
        asm volatile("discard.global.L2 [%0], 128;" ::"l"(p));
    }
    s += v;
  }
  if (s == 1.2345e30f) out[i] = s; // defeat DCE; never true
}

template <int POLICY>
float run(const float *a, size_t n, float *out, int iters) {
  int grid = 0, block = 0;
  cudaOccupancyMaxPotentialBlockSize(&grid, &block, stream<POLICY>, 0, 0);
  grid *= 4;
  cudaEvent_t s, e;
  cudaEventCreate(&s); cudaEventCreate(&e);
  stream<POLICY><<<grid, block>>>(a, n, out); // warm
  cudaDeviceSynchronize();
  cudaEventRecord(s);
  for (int k = 0; k < iters; k++) stream<POLICY><<<grid, block>>>(a, n, out);
  cudaEventRecord(e); cudaEventSynchronize(e);
  float ms = 0; cudaEventElapsedTime(&ms, s, e);
  return (n * sizeof(float) * (double)iters) / (ms * 1e-3) / 1e9; // GB/s
}

int main() {
  size_t bytes = (size_t)2 * 1024 * 1024 * 1024; // 2 GiB >> L2
  size_t n = bytes / sizeof(float);
  float *a, *out;
  cudaMalloc(&a, bytes); cudaMalloc(&out, 1 << 20);
  cudaMemset(a, 0, bytes);
  int props_dev; cudaGetDevice(&props_dev);
  cudaDeviceProp pr; cudaGetDeviceProperties(&pr, props_dev);
  printf("== cache-hint streaming bandwidth, %zu MiB single-pass (>L2=%d MiB), "
         "sm_%d%d ==\n", bytes >> 20, pr.l2CacheSize >> 20, pr.major, pr.minor);
  const int IT = 20;
  printf("  %-14s %8.1f GB/s\n", "default",     run<0>(a, n, out, IT));
  printf("  %-14s %8.1f GB/s\n", "cs(stream)",  run<1>(a, n, out, IT));
  printf("  %-14s %8.1f GB/s\n", "evict_last",  run<2>(a, n, out, IT));
  printf("  %-14s %8.1f GB/s\n", "evict_first", run<3>(a, n, out, IT));
  printf("  %-14s %8.1f GB/s\n", "discard.L2",  run<4>(a, n, out, IT));
  printf("== if all ~equal: cache hints cannot help a DRAM-bound single-pass "
         "read (SHAPE dead, evidenced) ==\n");
  cudaFree(a); cudaFree(out);
  return 0;
}
