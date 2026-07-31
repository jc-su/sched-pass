// paged_softmax.cu -- a real online-softmax paged-decode attention kernel with
// a C launcher, for the sched-pass dynamic-loop demo. One warp per request;
// KV stored per page as [token][2*SD] (K in lanes 0..SD-1, V in SD..2SD-1).
// The score = warp-shuffle dot(q,k) feeds fmaxf (running max) and __expf
// (weight): the online-softmax signature the pass's shed score-mask keys on.
//
// This is the kernel the Python control plane (sched_rt.py) weaves via the
// clang JIT + baked-address ABI -- no runtime/sched_rt.h, no __sched_* globals;
// the pass bakes the plane's table addresses from SCHED_BAKE_* at compile time.
#include <cuda_runtime.h>

#define SD 32 // head dim == warp

extern "C" __global__ void paged_softmax(const float *__restrict__ kv,
                                         const int *__restrict__ bt,
                                         const int *__restrict__ nbl,
                                         const float *__restrict__ q,
                                         float *__restrict__ out, int bt_stride,
                                         int page_tokens) {
  int seq = blockIdx.x;
  int lane = threadIdx.x; // blockDim.x == 32
  float qv = q[seq * SD + lane];
  float m = -1e30f, l = 0.f, acc = 0.f;
  int nb = nbl[seq];
  for (int b = 0; b < nb; ++b) {
    int page = bt[seq * bt_stride + b];
    const float *base = kv + (long long)page * page_tokens * (2 * SD);
#pragma clang loop unroll(disable)
    for (int t = 0; t < page_tokens; ++t) {
      float s = base[t * 2 * SD + lane] * qv; // K: the woven stream site
#pragma unroll
      for (int o = 16; o; o >>= 1)
        s += __shfl_xor_sync(0xffffffffu, s, o);
      s *= 0.125f;
      float mn = fmaxf(m, s);
      float c = __expf(m - mn);
      float w = __expf(s - mn);
      float vx = base[t * 2 * SD + SD + lane]; // V
      l = l * c + w;
      acc = acc * c + w * vx;
      m = mn;
    }
  }
  out[seq * SD + lane] = acc / l;
}

extern "C" void launch_paged_softmax(const void *kv, const void *bt,
                                     const void *nbl, const void *q, void *out,
                                     int nseq, int bt_stride, int page_tokens,
                                     void *stream) {
  paged_softmax<<<nseq, SD, 0, (cudaStream_t)stream>>>(
      (const float *)kv, (const int *)bt, (const int *)nbl, (const float *)q,
      (float *)out, bt_stride, page_tokens);
}

// Occupancy of THIS kernel as compiled (regs/smem included), for the control
// plane's R = blocks_per_sm * SM_count (the CLC resident-prefix model; see
// experiments/clc/FINDINGS.md). The kernel owner computes blocks/SM because
// only it holds the function handle; Python multiplies by SM count
// (SchedPlane.r_from_occupancy). Returns 0 on error -- treat as unknown.
extern "C" int paged_softmax_occupancy() {
  int blocks = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, paged_softmax, SD,
                                                    0) != cudaSuccess)
    return 0;
  return blocks;
}
