// clc_pipeline_probe.cu -- does software-pipelining the CLC claim hide its
// latency, and does that flip CLC from tie/loss to win?
//
// Every existing sched-pass CLC probe (and the SchedWorkQueue driver) issues
// try_cancel AFTER the task body -- claim latency fully EXPOSED, serialized
// between tasks. NVIDIA's CUTLASS pattern issues try_cancel ASYNC and EARLY so
// the cancel overlaps compute. This probe measures three schedules on the same
// synthetic work so the only variable is claim placement:
//
//   static     : grid=tasks, no CLC (hardware push-dispatch baseline)
//   exposed    : per task -> burn(); issue+collect   (the CURRENT model)
//   pipelined  : issue once; loop { burn(); collect; issue-next }  (overlapped)
//
// If pipelined ~= static (claim hidden) while exposed loses, the win is real
// and the fix is a driver change (software-pipeline emitClaim), NOT a CLC
// limitation. Work here is a clock-burn (compute) to ISOLATE claim-latency
// hiding; a memory-bound follow-up is the natural next step.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct Stats {
  uint32_t processed;
  uint32_t attempts;
  uint32_t successes;
  uint64_t claim_cycles; // sum of exposed collect-wait cycles (tid0)
};

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t e = (call);                                                    \
    if (e != cudaSuccess) {                                                    \
      std::fprintf(stderr, "FATAL %s:%d %s: %s\n", __FILE__, __LINE__, #call,  \
                   cudaGetErrorString(e));                                     \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

__device__ __forceinline__ uint64_t clock64_dev() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t));
  return t;
}
__device__ __forceinline__ void burn(uint32_t cycles) {
  uint64_t start = clock64_dev();
  uint32_t x = threadIdx.x;
  while ((uint64_t)(clock64_dev() - start) < cycles)
    asm volatile("add.u32 %0, %0, 1;" : "+r"(x)::"memory");
}
__device__ __forceinline__ bool is_long(uint32_t raw, uint32_t tasks,
                                        uint32_t long_every, uint32_t layout) {
  if (!long_every) return false;
  uint32_t nlong = max(1u, tasks / long_every);
  if (layout == 1) return raw < nlong;              // long-prefix
  if (layout == 2) return raw >= tasks - nlong;     // long-suffix
  return raw % long_every == 0;                     // interleaved
}

// --- CLC, split into issue (async) and collect (wait+decode) ---------------
// Shared result/barrier are CUDA __shared__ so their state persists between the
// two halves while the task body runs in between (that is the whole point).
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
      "L_pipe_wait_%=:\n"
      "  mbarrier.try_wait.parity.shared::cta.b64 %%pc, [%1], 0;\n"
      "  @!%%pc bra L_pipe_wait_%=;\n"
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

// ---- static baseline: grid=tasks, no CLC ----------------------------------
__global__ void static_probe(uint32_t tasks, uint32_t long_every,
                            uint32_t short_c, uint32_t long_c, uint32_t layout,
                            uint32_t *visits) {
  uint32_t raw = blockIdx.x;
  burn(is_long(raw, tasks, long_every, layout) ? long_c : short_c);
  if (threadIdx.x == 0) atomicAdd(&visits[raw], 1);
}

// ---- exposed CLC: burn THEN issue+collect (the current model) --------------
__global__ void clc_exposed(uint32_t tasks, uint32_t long_every,
                           uint32_t short_c, uint32_t long_c, uint32_t layout,
                           uint32_t *visits, Stats *st) {
  __shared__ __align__(16) uint8_t res[16];
  __shared__ __align__(8) uint64_t bar;
  __shared__ uint32_t next_raw;
  uint32_t worker = blockIdx.x, raw = worker;
  if (threadIdx.x == 0) { st[worker] = {0, 0, 0, 0}; }
  __syncthreads();
  while (raw < tasks) {
    burn(is_long(raw, tasks, long_every, layout) ? long_c : short_c);
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(&visits[raw], 1);
      st[worker].processed++;
      uint64_t t0 = clock64_dev();
      clc_issue(&bar, res);
      uint32_t claimed = clc_collect(&bar, res, tasks); // latency EXPOSED here
      st[worker].claim_cycles += clock64_dev() - t0;
      st[worker].attempts++;
      if (claimed < tasks) st[worker].successes++;
      next_raw = claimed;
    }
    __syncthreads();
    raw = next_raw;
  }
}

// ---- pipelined CLC: issue-ahead, body overlaps the in-flight cancel ---------
__global__ void clc_pipelined(uint32_t tasks, uint32_t long_every,
                             uint32_t short_c, uint32_t long_c, uint32_t layout,
                             uint32_t *visits, Stats *st) {
  __shared__ __align__(16) uint8_t res[16];
  __shared__ __align__(8) uint64_t bar;
  __shared__ uint32_t next_raw;
  uint32_t worker = blockIdx.x, raw = worker;
  if (threadIdx.x == 0) {
    st[worker] = {0, 0, 0, 0};
    clc_issue(&bar, res);   // cancel #1 in flight BEFORE the first body
    st[worker].attempts++;
  }
  __syncthreads();
  while (raw < tasks) {
    burn(is_long(raw, tasks, long_every, layout) ? long_c : short_c); // overlaps
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(&visits[raw], 1);
      st[worker].processed++;
      uint64_t t0 = clock64_dev();
      uint32_t claimed = clc_collect(&bar, res, tasks); // wait already hidden
      st[worker].claim_cycles += clock64_dev() - t0;
      if (claimed < tasks) {
        st[worker].successes++;
        clc_issue(&bar, res);       // cancel for the NEXT body, overlaps it
        st[worker].attempts++;
      }
      next_raw = claimed;
    }
    __syncthreads();
    raw = next_raw;
  }
}

template <class F> static float timeit(F f, int iters) {
  f(); CHECK(cudaDeviceSynchronize());   // warm
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
  uint32_t threads = argc > 2 ? std::atoi(argv[2]) : 128;
  uint32_t long_every = argc > 3 ? std::atoi(argv[3]) : 0;   // 0 = uniform
  uint32_t layout = argc > 4 ? std::atoi(argv[4]) : 0;
  int iters = argc > 5 ? std::atoi(argv[5]) : 50;
  uint32_t long_mult = argc > 6 ? std::atoi(argv[6]) : 8; // long task work multiplier

  int dev = 0, sm = 0, occ = 0;
  cudaDeviceProp prop{};
  CHECK(cudaGetDevice(&dev));
  CHECK(cudaGetDeviceProperties(&prop, dev));
  CHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev));
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, clc_pipelined,
                                                      threads, 0));
  uint32_t R = (uint32_t)occ * sm;
  std::printf("== CLC pipeline probe on %s sm_%d%d ==\n", prop.name, prop.major,
              prop.minor);
  std::printf("   tasks=%u threads=%u R=%u (occ=%d x %d SMs) long_every=%u "
              "layout=%u iters=%d\n",
              tasks, threads, R, occ, sm, long_every, layout, iters);

  uint32_t *d_visits;
  Stats *d_st;
  CHECK(cudaMalloc(&d_visits, tasks * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_st, tasks * sizeof(Stats)));

  auto claimed_cyc = [&](uint32_t ntasks) {
    std::vector<Stats> h(ntasks);
    CHECK(cudaMemcpy(h.data(), d_st, ntasks * sizeof(Stats),
                     cudaMemcpyDeviceToHost));
    uint64_t c = 0, a = 0;
    for (auto &s : h) { c += s.claim_cycles; a += s.attempts; }
    return a ? (double)c / a : 0.0;
  };

  std::printf("\n  %-8s %10s %10s %10s %8s %8s %10s %10s\n", "work", "static_us",
              "exposed_us", "pipe_us", "exp_vs_st", "pip_vs_st", "exp_claim",
              "pip_claim");
  // work-size sweep: from claim-latency-scale up to well-amortized
  uint32_t works[] = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
  for (uint32_t w : works) {
    uint32_t sc = w, lc = long_every ? w * long_mult : w;
    auto st_launch = [&] {
      CHECK(cudaMemset(d_visits, 0, tasks * sizeof(uint32_t)));
      static_probe<<<tasks, threads>>>(tasks, long_every, sc, lc, layout,
                                       d_visits);
    };
    auto ex_launch = [&] {
      CHECK(cudaMemset(d_visits, 0, tasks * sizeof(uint32_t)));
      CHECK(cudaMemset(d_st, 0, tasks * sizeof(Stats)));
      clc_exposed<<<tasks, threads>>>(tasks, long_every, sc, lc, layout,
                                      d_visits, d_st);
    };
    auto pp_launch = [&] {
      CHECK(cudaMemset(d_visits, 0, tasks * sizeof(uint32_t)));
      CHECK(cudaMemset(d_st, 0, tasks * sizeof(Stats)));
      clc_pipelined<<<tasks, threads>>>(tasks, long_every, sc, lc, layout,
                                        d_visits, d_st);
    };
    float st_us = timeit(st_launch, iters);
    float ex_us = timeit(ex_launch, iters);
    double ex_claim = claimed_cyc(tasks);
    float pp_us = timeit(pp_launch, iters);
    double pp_claim = claimed_cyc(tasks);
    std::printf("  %-8u %10.2f %10.2f %10.2f %+7.1f%% %+7.1f%% %10.0f %10.0f\n",
                w, st_us, ex_us, pp_us, 100.0 * (ex_us - st_us) / st_us,
                100.0 * (pp_us - st_us) / st_us, ex_claim, pp_claim);
  }
  std::printf("\n  (neg %% = faster than static; exp_claim/pip_claim = mean "
              "collect-wait cycles/attempt on tid0)\n");
  CHECK(cudaFree(d_visits));
  CHECK(cudaFree(d_st));
  return 0;
}
