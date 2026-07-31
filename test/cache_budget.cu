// cache_budget.cu -- is L2 residency a CONTINUOUS per-request BUDGET knob?
// createpolicy.fractional pins a FRACTION f of a request's reused lines. If time
// varies smoothly with f, we can ALLOCATE L2 per request by measured reuse/
// compute (OBSERVE -> SHAPE): give the high-reuse requests a bigger fraction.
// This is the compute-aware cache-allocation mechanism (beyond binary pin/bypass).
//
// Build: clang++-22 -x cuda --cuda-gpu-arch=sm_120a --cuda-path=/usr/local/cuda-12.9 \
//        -O3 -std=c++17 test/cache_budget.cu -o build/cb -lcudart && ./build/cb
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__global__ void mix(const float *R, size_t Rn, const float *S, size_t Sn,
                    float frac, float *out) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  unsigned long long pin, str;
  // fraction f of R's lines get evict_last (pinned); rest fall to evict_normal
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, %1;"
               : "=l"(pin) : "f"(frac));
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(str));
  float acc = 0.f;
  for (size_t j = i; j < Sn; j += stride) {
    const float *rp = R + (j & (Rn - 1));
    const float *sp = S + j;
    float r, x;
    asm volatile("ld.global.L2::cache_hint.f32 %0, [%1], %2;" : "=f"(r) : "l"(rp), "l"(pin));
    asm volatile("ld.global.L2::cache_hint.f32 %0, [%1], %2;" : "=f"(x) : "l"(sp), "l"(str));
    acc += r + x;
  }
  if (acc == 1.2345e30f) out[i] = acc;
}

float run(const float *R, size_t Rn, const float *S, size_t Sn, float frac,
          float *out, int iters) {
  int grid = 0, block = 0;
  cudaOccupancyMaxPotentialBlockSize(&grid, &block, mix, 0, 0);
  grid *= 4;
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  mix<<<grid, block>>>(R, Rn, S, Sn, frac, out);
  cudaDeviceSynchronize();
  cudaEventRecord(s);
  for (int k = 0; k < iters; k++) mix<<<grid, block>>>(R, Rn, S, Sn, frac, out);
  cudaEventRecord(e); cudaEventSynchronize(e);
  float ms = 0; cudaEventElapsedTime(&ms, s, e);
  return ms / iters;
}

int main() {
  int dev; cudaGetDevice(&dev); cudaDeviceProp pr; cudaGetDeviceProperties(&pr, dev);
  size_t Rb = (size_t)384 * 1024 * 1024, Sb = (size_t)2 * 1024 * 1024 * 1024;
  size_t Rn = Rb / 4, Sn = Sb / 4;
  float *R, *S, *out;
  cudaMalloc(&R, Rb); cudaMalloc(&S, Sb); cudaMalloc(&out, 1 << 20);
  cudaMemset(R, 0, Rb); cudaMemset(S, 0, Sb);
  printf("== L2 residency BUDGET sweep (R=%zu MiB reused, L2=%d MiB), sm_%d%d ==\n",
         Rb >> 20, pr.l2CacheSize >> 20, pr.major, pr.minor);
  const int IT = 20;
  for (float f : {0.0f, 0.25f, 0.5f, 0.75f, 1.0f})
    printf("  pin fraction %.2f   %8.3f ms/iter\n", f, run(R, Rn, S, Sn, f, out, IT));
  printf("== monotone in f => L2 residency is a CONTINUOUS per-request budget "
         "knob (OBSERVE-measured reuse -> SHAPE fraction) ==\n");
  cudaFree(R); cudaFree(S); cudaFree(out);
  return 0;
}
