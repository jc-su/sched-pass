// clc_noise_probe.cu -- CLC's niche, quantified: scheduling under COST
// UNCERTAINTY. task_order (pi) is open-loop: it needs a cost prediction
// BEFORE launch. CLC is closed-loop: late binding needs no prediction. The
// pipeline probe showed static+LPT beats CLC when the prediction is ORACLE
// (correct); this probe measures both as the prediction degrades:
//
//   estimate  c_hat_i = c_i * max(0.05, 1 + eps*u_i),  u_i ~ U(-1,1) fixed LCG
//   pi(eps)   = argsort(c_hat) descending  (LPT on the ESTIMATE)
//   schedules: static+pi(eps)   (grid=tasks, task = order[blockIdx.x])
//              clc+pi(eps)      (same pi; workers late-bind raw via try_cancel)
//
// Expected: static degrades as eps grows (mispredicted longs land late and
// straggle); CLC stays ~flat (whoever frees first takes the next task
// regardless of where pi put it). The eps where clc+pi(eps) crosses below
// static+pi(eps) IS the control plane's CLC arming threshold: arm CLC when
// prediction confidence is worse than the crossing point.
//
// Work model matches the regime where ordering matters most (pipeline probe
// sweet spot): 25% long x LONG_MULT, base WORK cycles, tasks >> R.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

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

// The single-blob late-binding claim (issue+collect AFTER the body -- the
// measured contract; see clc_pipeline_probe.cu for why claim-ahead loses).
__device__ __forceinline__ uint32_t clc_try_cancel(uint32_t fallback) {
#if __CUDA_ARCH__ >= 1000
  uint32_t out;
  asm volatile(
      "{\n"
      "  .reg .pred %%pc;\n"
      "  .shared .align 16 .b8 _noise_clc_res[16];\n"
      "  .shared .align 8 .b64 _noise_clc_bar;\n"
      "  .reg .b128 %%rq;\n"
      "  .reg .b64 %%tmp;\n"
      "  .reg .b32 %%resa, %%bara, %%cx;\n"
      "  mov.u32 %%resa, _noise_clc_res;\n"
      "  mov.u32 %%bara, _noise_clc_bar;\n"
      "  mbarrier.init.shared::cta.b64 [%%bara], 1;\n"
      "  fence.proxy.async.shared::cta;\n"
      "  clusterlaunchcontrol.try_cancel.async.shared::cta."
      "mbarrier::complete_tx::bytes.b128 [%%resa], [%%bara];\n"
      "  mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 "
      "%%tmp, [%%bara], 16;\n"
      "L_noise_clc_wait:\n"
      "  mbarrier.try_wait.parity.shared::cta.b64 %%pc, [%%bara], 0;\n"
      "  @!%%pc bra L_noise_clc_wait;\n"
      "  ld.shared.b128 %%rq, [%%resa];\n"
      "  clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 %%pc, %%rq;\n"
      "  mov.u32 %%cx, %1;\n"
      "  @%%pc clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 "
      "{%%cx, _, _, _}, %%rq;\n"
      "  mov.u32 %0, %%cx;\n"
      "}\n"
      : "=r"(out)
      : "r"(fallback)
      : "memory");
  return out;
#else
  return fallback;
#endif
}

// pi is data for BOTH schedules: task = order[raw]. Cost lives per TASK.
__global__ void static_probe(uint32_t tasks, const int *__restrict__ order,
                             const uint32_t *__restrict__ cost,
                             uint32_t *visits) {
  uint32_t raw = blockIdx.x;
  uint32_t task = order[raw];
  burn(cost[task]);
  if (threadIdx.x == 0)
    atomicAdd(&visits[task], 1);
}

__global__ void clc_probe(uint32_t tasks, const int *__restrict__ order,
                          const uint32_t *__restrict__ cost, uint32_t *visits) {
  __shared__ uint32_t next_raw;
  uint32_t raw = blockIdx.x;
  while (raw < tasks) {
    uint32_t task = order[raw];
    burn(cost[task]);
    if (threadIdx.x == 0) {
      atomicAdd(&visits[task], 1);
      next_raw = clc_try_cancel(tasks); // LATE BIND: claim after the body
    }
    __syncthreads();
    raw = next_raw;
  }
}

template <class F> static float timeit(F f, int iters) {
  f();
  CHECK(cudaDeviceSynchronize()); // warm
  cudaEvent_t a, b;
  CHECK(cudaEventCreate(&a));
  CHECK(cudaEventCreate(&b));
  CHECK(cudaEventRecord(a));
  for (int i = 0; i < iters; ++i)
    f();
  CHECK(cudaEventRecord(b));
  CHECK(cudaEventSynchronize(b));
  float ms = 0.f;
  CHECK(cudaEventElapsedTime(&ms, a, b));
  CHECK(cudaEventDestroy(a));
  CHECK(cudaEventDestroy(b));
  return 1000.f * ms / iters;
}

// Deterministic per-(index,seed) uniform in [-1, 1].
static double unoise(uint32_t i, uint32_t seed) {
  uint32_t s = i * 2654435761u ^ seed * 2246822519u;
  s ^= s >> 15;
  s *= 2246822519u;
  s ^= s >> 13;
  return (double)(s & 0xFFFFFF) / (double)0x7FFFFF - 1.0; // [-1, 1]
}

int main(int argc, char **argv) {
  uint32_t tasks = argc > 1 ? std::atoi(argv[1]) : 8192;
  uint32_t threads = argc > 2 ? std::atoi(argv[2]) : 128;
  uint32_t work = argc > 3 ? std::atoi(argv[3]) : 1024;      // short cycles
  uint32_t long_mult = argc > 4 ? std::atoi(argv[4]) : 32;   // long multiplier
  uint32_t long_every = argc > 5 ? std::atoi(argv[5]) : 4;   // 1/4 long
  int iters = argc > 6 ? std::atoi(argv[6]) : 40;
  uint32_t seed = argc > 7 ? std::atoi(argv[7]) : 1;

  int dev = 0, sm = 0, occ = 0;
  cudaDeviceProp prop{};
  CHECK(cudaGetDevice(&dev));
  CHECK(cudaGetDeviceProperties(&prop, dev));
  CHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev));
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, clc_probe, threads,
                                                      0));
  uint32_t R = (uint32_t)occ * sm;
  std::printf("== CLC noise probe on %s sm_%d%d ==\n", prop.name, prop.major,
              prop.minor);
  std::printf("   tasks=%u threads=%u work=%u x%u every %u, R=%u, iters=%d, "
              "seed=%u\n",
              tasks, threads, work, long_mult, long_every, R, iters, seed);

  // True per-task cost: interleaved longs (raw layout is irrelevant -- pi
  // fully determines placement in both schedules).
  std::vector<uint32_t> cost_h(tasks);
  uint32_t n_long = 0;
  for (uint32_t i = 0; i < tasks; ++i) {
    bool lg = long_every && (i % long_every == 0);
    cost_h[i] = lg ? work * long_mult : work;
    n_long += lg;
  }

  uint32_t *d_cost, *d_visits;
  int *d_order;
  CHECK(cudaMalloc(&d_cost, tasks * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_visits, tasks * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_order, tasks * sizeof(int)));
  CHECK(cudaMemcpy(d_cost, cost_h.data(), tasks * sizeof(uint32_t),
                   cudaMemcpyHostToDevice));

  auto run_pair = [&](const std::vector<int> &order, float &st_us,
                      float &clc_us, bool &exact) {
    CHECK(cudaMemcpy(d_order, order.data(), tasks * sizeof(int),
                     cudaMemcpyHostToDevice));
    auto st = [&] {
      static_probe<<<tasks, threads>>>(tasks, d_order, d_cost, d_visits);
    };
    auto cl = [&] {
      clc_probe<<<tasks, threads>>>(tasks, d_order, d_cost, d_visits);
    };
    // exactly-once check on the CLC schedule (one un-timed run)
    CHECK(cudaMemset(d_visits, 0, tasks * sizeof(uint32_t)));
    cl();
    CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> v(tasks);
    CHECK(cudaMemcpy(v.data(), d_visits, tasks * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost));
    exact = true;
    for (uint32_t i = 0; i < tasks; ++i)
      exact &= (v[i] == 1);
    st_us = timeit(st, iters);
    clc_us = timeit(cl, iters);
  };

  // pi from a noisy estimate: LPT on c_hat.
  auto noisy_order = [&](double eps) {
    std::vector<double> chat(tasks);
    for (uint32_t i = 0; i < tasks; ++i) {
      double f = 1.0 + eps * unoise(i, seed);
      chat[i] = (double)cost_h[i] * std::max(0.05, f);
    }
    std::vector<int> ord(tasks);
    for (uint32_t i = 0; i < tasks; ++i)
      ord[i] = (int)i;
    std::stable_sort(ord.begin(), ord.end(),
                     [&](int a, int b) { return chat[a] > chat[b]; });
    return ord;
  };
  // prediction quality: fraction of true longs ranked in the first n_long.
  auto long_recall = [&](const std::vector<int> &ord) {
    uint32_t hit = 0;
    for (uint32_t k = 0; k < n_long; ++k)
      hit += cost_h[ord[k]] > work;
    return 100.0 * hit / std::max(1u, n_long);
  };

  std::printf("\n  %-10s %9s %11s %11s %11s %10s\n", "pi", "recall%",
              "static_us", "clc_us", "clc_vs_st", "st_vs_orac");
  float oracle_us = 0.f;
  double epss[] = {0.0, 0.25, 0.5, 1.0, 2.0, 4.0};
  for (double eps : epss) {
    auto ord = noisy_order(eps);
    float st, cl;
    bool exact;
    run_pair(ord, st, cl, exact);
    if (eps == 0.0)
      oracle_us = st;
    char tag[32];
    std::snprintf(tag, sizeof(tag), "eps=%.2f", eps);
    std::printf("  %-10s %9.1f %11.2f %11.2f %+10.1f%% %+9.1f%%%s\n", tag,
                long_recall(ord), st, cl, 100.0 * (cl - st) / st,
                100.0 * (st - oracle_us) / oracle_us,
                exact ? "" : "  !EXACTLY-ONCE");
  }
  { // no-information limit: identity pi (scattered longs, no sorting at all)
    std::vector<int> ord(tasks);
    for (uint32_t i = 0; i < tasks; ++i)
      ord[i] = (int)i;
    float st, cl;
    bool exact;
    run_pair(ord, st, cl, exact);
    std::printf("  %-10s %9.1f %11.2f %11.2f %+10.1f%% %+9.1f%%%s\n",
                "identity", long_recall(ord), st, cl, 100.0 * (cl - st) / st,
                100.0 * (st - oracle_us) / oracle_us,
                exact ? "" : "  !EXACTLY-ONCE");
  }
  std::printf("\n  (recall%% = true longs ranked in the top-%u; clc_vs_st < 0 "
              "=> CLC wins at that prediction quality;\n   st_vs_orac = how "
              "much static's makespan degrades vs oracle LPT. The crossing "
              "eps is the CLC arming threshold.)\n",
              n_long);
  CHECK(cudaFree(d_cost));
  CHECK(cudaFree(d_visits));
  CHECK(cudaFree(d_order));
  return 0;
}
