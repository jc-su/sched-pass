// clc_membound_probe.cu -- is there a CLC benefit on MEMORY-bound decode that
// is INDEPENDENT of task ordering?
//
// The compute-bound pipeline probe showed CLC's makespan win is entirely
// "rescue a bad task order", and static+LPT dominates. But the CLC study saw
// +65% on UNIFORM short decode -- uniform work has NO order to fix, so that
// win, if real, is a MEMORY-SYSTEM effect (R persistent streaming workers vs
// static's N one-shot blocks), i.e. a residency lever, not an ordering lever.
//
// This probe isolates it: UNIFORM per-task work (no ordering advantage
// possible), a real global-memory KV stream, three schedules --
//   static     : grid=tasks, one task/block (hardware push)
//   clc_exposed: R persistent workers, claim AFTER body (late binding)
//   clc_pipe   : R persistent workers, claim-ahead (overlap; safe iff uniform)
// swept across MEMORY FOOTPRINT (distinct pages -> L2-resident vs DRAM-bound)
// and N/R. If CLC beats static on uniform memory work, CLC is a genuine
// residency lever complementary to task_order. If it ties/loses, it is not.
//
// Correctness: every mode must be bit-exact vs the static reference.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define D 128 // threads/block == head dim; coalesced KV stream

__device__ __forceinline__ void clc_issue(uint64_t *bar, uint8_t *res) {
#if __CUDA_ARCH__ >= 1000
  uint32_t bs = (uint32_t)__cvta_generic_to_shared(bar);
  uint32_t rs = (uint32_t)__cvta_generic_to_shared(res);
  asm volatile(
      "{\n"
      "  .reg .b64 %%tmp;\n"
      "  mbarrier.init.shared::cta.b64 [%0], 1;\n"
      "  fence.proxy.async.shared::cta;\n"
      "  clusterlaunchcontrol.try_cancel.async.shared::cta."
      "mbarrier::complete_tx::bytes.b128 [%1], [%0];\n"
      "  mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 %%tmp, [%0], 16;\n"
      "}\n" ::"r"(bs),
      "r"(rs)
      : "memory");
#endif
}
__device__ __forceinline__ uint32_t clc_collect(uint64_t *bar, uint8_t *res,
                                               uint32_t fallback) {
  uint32_t out = fallback;
#if __CUDA_ARCH__ >= 1000
  uint32_t bs = (uint32_t)__cvta_generic_to_shared(bar);
  uint32_t rs = (uint32_t)__cvta_generic_to_shared(res);
  asm volatile(
      "{\n"
      "  .reg .pred %%pc;\n"
      "  .reg .b128 %%rq;\n"
      "  .reg .b32 %%cx;\n"
      "L_mb_wait_%=:\n"
      "  mbarrier.try_wait.parity.shared::cta.b64 %%pc, [%1], 0;\n"
      "  @!%%pc bra L_mb_wait_%=;\n"
      "  ld.shared.b128 %%rq, [%2];\n"
      "  clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 %%pc, %%rq;\n"
      "  mov.u32 %%cx, %3;\n"
      "  @%%pc clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 "
      "{%%cx, _, _, _}, %%rq;\n"
      "  mov.u32 %0, %%cx;\n"
      "}\n"
      : "=r"(out)
      : "r"(bs), "r"(rs), "r"(fallback)
      : "memory");
#endif
  return out;
}

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t e = (call);                                                    \
    if (e != cudaSuccess) {                                                    \
      std::fprintf(stderr, "FATAL %s:%d %s: %s\n", __FILE__, __LINE__, #call,  \
                   cudaGetErrorString(e));                                     \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

// One task: stream nb pages of KV, accumulate (one FMA/load -> memory-bound).
__device__ __forceinline__ float stream_task(const float *__restrict__ kv,
                                            const int *__restrict__ bt, int nb,
                                            int bt_stride, int page_tokens,
                                            uint32_t raw) {
  int d = threadIdx.x;
  float acc = 0.f;
  for (int b = 0; b < nb; ++b) {
    int page = bt[raw * bt_stride + b];
    const float *base = kv + (long long)page * page_tokens * D;
    for (int t = 0; t < page_tokens; ++t)
      acc += base[t * D + d];
  }
  return acc;
}

__global__ void static_decode(const float *kv, const int *bt, const int *nbl,
                             float *out, uint32_t tasks, int bt_stride,
                             int page_tokens) {
  uint32_t raw = blockIdx.x;
  if (raw >= tasks) return;
  out[raw * D + threadIdx.x] =
      stream_task(kv, bt, nbl[raw], bt_stride, page_tokens, raw);
}

__global__ void clc_exposed_decode(const float *kv, const int *bt,
                                  const int *nbl, float *out, uint32_t tasks,
                                  int bt_stride, int page_tokens) {
  __shared__ __align__(16) uint8_t res[16];
  __shared__ __align__(8) uint64_t bar;
  __shared__ uint32_t next_raw;
  uint32_t raw = blockIdx.x;
  while (raw < tasks) {
    out[raw * D + threadIdx.x] =
        stream_task(kv, bt, nbl[raw], bt_stride, page_tokens, raw);
    __syncthreads();
    if (threadIdx.x == 0) {
      clc_issue(&bar, res);
      next_raw = clc_collect(&bar, res, tasks); // late binding: claim when free
    }
    __syncthreads();
    raw = next_raw;
  }
}

__global__ void clc_pipe_decode(const float *kv, const int *bt, const int *nbl,
                              float *out, uint32_t tasks, int bt_stride,
                              int page_tokens) {
  __shared__ __align__(16) uint8_t res[16];
  __shared__ __align__(8) uint64_t bar;
  __shared__ uint32_t next_raw;
  uint32_t raw = blockIdx.x;
  if (threadIdx.x == 0) clc_issue(&bar, res); // claim-ahead (uniform: safe)
  __syncthreads();
  while (raw < tasks) {
    out[raw * D + threadIdx.x] =
        stream_task(kv, bt, nbl[raw], bt_stride, page_tokens, raw);
    __syncthreads();
    if (threadIdx.x == 0) {
      uint32_t claimed = clc_collect(&bar, res, tasks);
      if (claimed < tasks) clc_issue(&bar, res);
      next_raw = claimed;
    }
    __syncthreads();
    raw = next_raw;
  }
}

template <class F> static float timeit(F f, int iters) {
  f(); CHECK(cudaDeviceSynchronize());
  cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
  CHECK(cudaEventRecord(a));
  for (int i = 0; i < iters; ++i) f();
  CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
  float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, a, b));
  CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
  return 1000.f * ms / iters;
}

int main(int argc, char **argv) {
  uint32_t tasks = argc > 1 ? std::atoi(argv[1]) : 8192;
  int nb = argc > 2 ? std::atoi(argv[2]) : 4;         // pages per task (uniform)
  int page_tokens = argc > 3 ? std::atoi(argv[3]) : 64;
  int iters = argc > 4 ? std::atoi(argv[4]) : 40;

  int dev = 0, sm = 0, occ = 0, l2 = 0;
  cudaDeviceProp prop{};
  CHECK(cudaGetDevice(&dev));
  CHECK(cudaGetDeviceProperties(&prop, dev));
  CHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev));
  CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, clc_exposed_decode,
                                                      D, 0));
  uint32_t R = (uint32_t)occ * sm;
  size_t page_floats = (size_t)page_tokens * D;
  size_t page_bytes = page_floats * sizeof(float);
  std::printf("== CLC memory-bound decode probe on %s sm_%d%d ==\n", prop.name,
              prop.major, prop.minor);
  std::printf("   tasks=%u nb=%d page_tokens=%d D=%d R=%u (occ=%d x %d SMs) "
              "L2=%d MB page=%zu KB iters=%d\n",
              tasks, nb, page_tokens, D, R, occ, sm, l2 >> 20, page_bytes >> 10,
              iters);

  // biggest footprint we sweep: all accesses distinct
  size_t max_pages = (size_t)tasks * nb;
  float *d_kv;
  CHECK(cudaMalloc(&d_kv, max_pages * page_bytes));
  { // touch it so it's paged in; content value irrelevant to timing
    CHECK(cudaMemset(d_kv, 1, max_pages * page_bytes));
  }
  int *d_bt, *d_nbl;
  float *d_out, *d_ref;
  CHECK(cudaMalloc(&d_bt, (size_t)tasks * nb * sizeof(int)));
  CHECK(cudaMalloc(&d_nbl, tasks * sizeof(int)));
  CHECK(cudaMalloc(&d_out, (size_t)tasks * D * sizeof(float)));
  CHECK(cudaMalloc(&d_ref, (size_t)tasks * D * sizeof(float)));
  std::vector<int> nbl(tasks, nb);
  CHECK(cudaMemcpy(d_nbl, nbl.data(), tasks * sizeof(int),
                   cudaMemcpyHostToDevice));

  std::printf("\n  footprint model: bt[raw*nb+b] = (raw*nb+b) %% npages\n");
  std::printf("  %-10s %8s %11s %11s %11s %9s %9s %9s\n", "npages", "footMB",
              "static_us", "exposed_us", "pipe_us", "st_GBps", "ex_vs_st",
              "pp_vs_st");

  size_t npages_list[] = {256, 1024, 4096, 16384, 65536, 262144};
  std::vector<int> bt((size_t)tasks * nb);
  std::vector<float> h_ref(tasks * D), h_out(tasks * D);
  for (size_t npages : npages_list) {
    if (npages > max_pages) npages = max_pages;
    for (size_t i = 0; i < (size_t)tasks * nb; ++i)
      bt[i] = (int)(i % npages);
    CHECK(cudaMemcpy(d_bt, bt.data(), (size_t)tasks * nb * sizeof(int),
                     cudaMemcpyHostToDevice));
    double footMB = (double)(npages * page_bytes) / (1 << 20);
    double bytes = (double)tasks * nb * page_bytes; // bytes moved per launch

    auto st = [&] {
      static_decode<<<tasks, D>>>(d_kv, d_bt, d_nbl, d_out, tasks, nb,
                                  page_tokens);
    };
    auto ex = [&] {
      clc_exposed_decode<<<tasks, D>>>(d_kv, d_bt, d_nbl, d_out, tasks, nb,
                                       page_tokens);
    };
    auto pp = [&] {
      clc_pipe_decode<<<tasks, D>>>(d_kv, d_bt, d_nbl, d_out, tasks, nb,
                                    page_tokens);
    };

    // reference + correctness
    st(); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_ref.data(), d_out, (size_t)tasks * D * sizeof(float),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemset(d_out, 0, (size_t)tasks * D * sizeof(float)));
    ex(); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out.data(), d_out, (size_t)tasks * D * sizeof(float),
                     cudaMemcpyDeviceToHost));
    bool ex_ok = std::memcmp(h_ref.data(), h_out.data(),
                             (size_t)tasks * D * sizeof(float)) == 0;
    CHECK(cudaMemset(d_out, 0, (size_t)tasks * D * sizeof(float)));
    pp(); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out.data(), d_out, (size_t)tasks * D * sizeof(float),
                     cudaMemcpyDeviceToHost));
    bool pp_ok = std::memcmp(h_ref.data(), h_out.data(),
                             (size_t)tasks * D * sizeof(float)) == 0;

    float st_us = timeit(st, iters);
    float ex_us = timeit(ex, iters);
    float pp_us = timeit(pp, iters);
    double st_gbps = bytes / (st_us * 1e3);
    std::printf("  %-10zu %8.0f %11.2f %11.2f %11.2f %9.0f %+8.1f%% %+8.1f%%%s%s\n",
                npages, footMB, st_us, ex_us, pp_us, st_gbps,
                100.0 * (ex_us - st_us) / st_us,
                100.0 * (pp_us - st_us) / st_us, ex_ok ? "" : " EX!BITEXACT",
                pp_ok ? "" : " PP!BITEXACT");
  }
  std::printf("\n  (neg %% = CLC faster than static; st_GBps = static achieved "
              "read BW. L2=%d MB: footprint>L2 => DRAM-bound)\n",
              l2 >> 20);
  CHECK(cudaFree(d_kv)); CHECK(cudaFree(d_bt)); CHECK(cudaFree(d_nbl));
  CHECK(cudaFree(d_out)); CHECK(cudaFree(d_ref));
  return 0;
}
