// cache_reuse.cu -- the grounded SHAPE test: does cache MANAGEMENT pay in an
// L2-bound REUSE+STREAM mix (prefill's regime), where it could NOT on decode's
// no-reuse DRAM-bound stream?
//
// Setup: a REUSED array R (fits in L2) read many times, interleaved with a
// STREAMING array S (>> L2) read once. Under default LRU the streaming flood
// evicts R -> its re-reads miss to DRAM. SHAPE pins R (L2::evict_last) and
// streams S (L2::evict_first) so R stays hot. If SHAPE > default, cache
// management is a real lever in the reuse regime; if equal, SHAPE is dead
// everywhere (even with reuse + contention).
//
// Build: clang++-22 -x cuda --cuda-gpu-arch=sm_120a --cuda-path=/usr/local/cuda-12.9 \
//        -O3 -std=c++17 test/cache_reuse.cu -o build/cru -lcudart && ./build/cru
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

// POLICY 0 = default LRU; 1 = SHAPE (pin R evict_last, stream S evict_first)
template <int POLICY>
__global__ void mix(const float *R, size_t Rn, const float *S, size_t Sn,
                    float *out) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  unsigned long long pin = 0, str = 0;
  if (POLICY == 1) {
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pin));
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(str));
  }
  float acc = 0.f;
  for (size_t j = i; j < Sn; j += stride) {
    const float *rp = R + (j & (Rn - 1)); // reused (Rn power-of-2), contended
    const float *sp = S + j;              // streamed once
    float r, x;
    if (POLICY == 0) {
      asm volatile("ld.global.f32 %0, [%1];" : "=f"(r) : "l"(rp));
      asm volatile("ld.global.f32 %0, [%1];" : "=f"(x) : "l"(sp));
    } else {
      asm volatile("ld.global.L2::cache_hint.f32 %0, [%1], %2;"
                   : "=f"(r) : "l"(rp), "l"(pin));
      asm volatile("ld.global.L2::cache_hint.f32 %0, [%1], %2;"
                   : "=f"(x) : "l"(sp), "l"(str));
    }
    acc += r + x;
  }
  if (acc == 1.2345e30f) out[i] = acc;
}

template <int POLICY>
float run(const float *R, size_t Rn, const float *S, size_t Sn, float *out,
          int iters) {
  int grid = 0, block = 0;
  cudaOccupancyMaxPotentialBlockSize(&grid, &block, mix<POLICY>, 0, 0);
  grid *= 4;
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  mix<POLICY><<<grid, block>>>(R, Rn, S, Sn, out);
  cudaDeviceSynchronize();
  cudaEventRecord(s);
  for (int k = 0; k < iters; k++) mix<POLICY><<<grid, block>>>(R, Rn, S, Sn, out);
  cudaEventRecord(e); cudaEventSynchronize(e);
  float ms = 0; cudaEventElapsedTime(&ms, s, e);
  return ms / iters;
}

int main() {
  int dev; cudaGetDevice(&dev); cudaDeviceProp pr; cudaGetDeviceProperties(&pr, dev);
  size_t Rbytes = (size_t)64 * 1024 * 1024;   // reused, fits in L2 (128 MiB)
  size_t Sbytes = (size_t)2 * 1024 * 1024 * 1024; // streamed, >> L2
  size_t Rn = Rbytes / sizeof(float), Sn = Sbytes / sizeof(float);
  float *R, *S, *out;
  cudaMalloc(&R, Rbytes); cudaMalloc(&S, Sbytes); cudaMalloc(&out, 1 << 20);
  cudaMemset(R, 0, Rbytes); cudaMemset(S, 0, Sbytes);
  printf("== SHAPE in the L2-REUSE regime (R=%zu MiB reused, S=%zu MiB streamed, "
         "L2=%d MiB), sm_%d%d ==\n", Rbytes >> 20, Sbytes >> 20,
         pr.l2CacheSize >> 20, pr.major, pr.minor);
  const int IT = 20;
  float a = run<0>(R, Rn, S, Sn, out, IT);
  float b = run<1>(R, Rn, S, Sn, out, IT);
  printf("  default LRU        %8.3f ms/iter\n", a);
  printf("  SHAPE (pin+stream) %8.3f ms/iter   %+.1f%%\n", b, 100 * (b - a) / a);
  printf("== SHAPE < default => cache MANAGEMENT pays in the reuse regime (a 3rd "
         "lever, prefill); ~equal => SHAPE dead even with reuse+contention ==\n");
  cudaFree(R); cudaFree(S); cudaFree(out);
  return 0;
}
