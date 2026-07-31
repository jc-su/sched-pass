// Standalone Blackwell CLC probe.
//
// This intentionally does not use the sched-pass plugin. It manually emits the
// CLC claim sequence and records observable behavior: claim success rate,
// claimed raw ctaids, per-task visit counts, and claim latency.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct WorkerStats {
  uint32_t processed;
  uint32_t attempts;
  uint32_t successes;
  uint32_t first_claim;
  uint32_t last_claim;
  uint64_t claim_cycles;
  uint64_t max_claim_cycles;
};

enum LayoutMode : uint32_t {
  LayoutInterleaved = 0,
  LayoutLongPrefix = 1,
  LayoutLongSuffix = 2,
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

__device__ __forceinline__ void burn(uint32_t iters) {
  uint64_t start = clock64_dev();
  uint32_t x = threadIdx.x;
  while ((uint64_t)(clock64_dev() - start) < iters)
    asm volatile("add.u32 %0, %0, 1;" : "+r"(x) :: "memory");
}

__device__ __forceinline__ uint32_t clc_try_cancel(uint32_t fallback) {
#if __CUDA_ARCH__ >= 1000
  uint32_t out;
  asm volatile(
      "{\n"
      "  .reg .pred %%pc;\n"
      "  .shared .align 16 .b8 _probe_clc_res[16];\n"
      "  .shared .align 8 .b64 _probe_clc_bar;\n"
      "  .reg .b128 %%rq;\n"
      "  .reg .b64 %%tmp;\n"
      "  .reg .b32 %%resa, %%bara, %%cx;\n"
      "  mov.u32 %%resa, _probe_clc_res;\n"
      "  mov.u32 %%bara, _probe_clc_bar;\n"
      "  mbarrier.init.shared::cta.b64 [%%bara], 1;\n"
      "  fence.proxy.async.shared::cta;\n"
      "  clusterlaunchcontrol.try_cancel.async.shared::cta."
      "mbarrier::complete_tx::bytes.b128 [%%resa], [%%bara];\n"
      "  mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 "
      "%%tmp, [%%bara], 16;\n"
      "L_probe_clc_wait:\n"
      "  mbarrier.try_wait.parity.shared::cta.b64 %%pc, [%%bara], 0;\n"
      "  @!%%pc bra L_probe_clc_wait;\n"
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

__device__ __forceinline__ bool is_long_task(uint32_t raw, uint32_t tasks,
                                             uint32_t long_every,
                                             uint32_t layout) {
  if (!long_every)
    return false;
  uint32_t nlong = max(1u, tasks / long_every);
  if (layout == LayoutLongPrefix)
    return raw < nlong;
  if (layout == LayoutLongSuffix)
    return raw >= tasks - nlong;
  return raw % long_every == 0;
}

__global__ void static_probe(uint32_t tasks, uint32_t long_every,
                             uint32_t short_iters, uint32_t long_iters,
                             uint32_t layout,
                             uint32_t smem_bytes,
                             uint32_t *visits) {
  extern __shared__ unsigned char dyn_smem[];
  if (threadIdx.x == 0 && smem_bytes)
    dyn_smem[0] = (unsigned char)blockIdx.x;
  uint32_t raw = blockIdx.x;
  uint32_t iters = is_long_task(raw, tasks, long_every, layout) ? long_iters
                                                                : short_iters;
  burn(iters);
  if (threadIdx.x == 0)
    atomicAdd(&visits[raw], 1);
}

__global__ void clc_probe(uint32_t tasks, uint32_t long_every,
                          uint32_t short_iters, uint32_t long_iters,
                          uint32_t layout,
                          uint32_t smem_bytes,
                          uint32_t *visits, uint32_t *claimed_hist,
                          WorkerStats *stats) {
  extern __shared__ unsigned char dyn_smem[];
  __shared__ uint32_t next_raw;
  __shared__ uint64_t claim_cycles;

  const uint32_t worker = blockIdx.x;
  uint32_t raw = worker;

  if (threadIdx.x == 0) {
    if (smem_bytes)
      dyn_smem[0] = (unsigned char)worker;
    stats[worker].processed = 0;
    stats[worker].attempts = 0;
    stats[worker].successes = 0;
    stats[worker].first_claim = 0xffffffffu;
    stats[worker].last_claim = 0xffffffffu;
    stats[worker].claim_cycles = 0;
    stats[worker].max_claim_cycles = 0;
  }
  __syncthreads();

  while (raw < tasks) {
    uint32_t iters = is_long_task(raw, tasks, long_every, layout) ? long_iters
                                                                  : short_iters;
    burn(iters);

    if (threadIdx.x == 0) {
      atomicAdd(&visits[raw], 1);
      stats[worker].processed++;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      uint64_t t0 = clock64_dev();
      uint32_t claimed = clc_try_cancel(tasks);
      uint64_t dt = clock64_dev() - t0;
      next_raw = claimed;
      claim_cycles = dt;

      stats[worker].attempts++;
      stats[worker].claim_cycles += dt;
      stats[worker].max_claim_cycles =
          max(stats[worker].max_claim_cycles, dt);
      if (claimed < tasks) {
        stats[worker].successes++;
        if (stats[worker].first_claim == 0xffffffffu)
          stats[worker].first_claim = claimed;
        stats[worker].last_claim = claimed;
        atomicAdd(&claimed_hist[claimed], 1);
      }
    }
    __syncthreads();
    raw = next_raw;
    (void)claim_cycles;
  }
}

static float time_static(uint32_t tasks, uint32_t threads, uint32_t long_every,
                         uint32_t short_iters, uint32_t long_iters,
                         uint32_t layout,
                         uint32_t smem_bytes,
                         uint32_t *d_visits) {
  CHECK(cudaMemset(d_visits, 0, tasks * sizeof(uint32_t)));
  static_probe<<<tasks, threads, smem_bytes>>>(tasks, long_every, short_iters,
                                               long_iters, layout, smem_bytes,
                                               d_visits);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemset(d_visits, 0, tasks * sizeof(uint32_t)));
  cudaEvent_t t0, t1;
  CHECK(cudaEventCreate(&t0));
  CHECK(cudaEventCreate(&t1));
  CHECK(cudaEventRecord(t0));
  static_probe<<<tasks, threads, smem_bytes>>>(tasks, long_every, short_iters,
                                               long_iters, layout, smem_bytes,
                                               d_visits);
  CHECK(cudaGetLastError());
  CHECK(cudaEventRecord(t1));
  CHECK(cudaEventSynchronize(t1));
  float ms = 0.f;
  CHECK(cudaEventElapsedTime(&ms, t0, t1));
  CHECK(cudaEventDestroy(t0));
  CHECK(cudaEventDestroy(t1));
  return 1000.f * ms;
}

static float time_clc(uint32_t tasks, uint32_t threads, uint32_t long_every,
                      uint32_t short_iters, uint32_t long_iters,
                      uint32_t layout,
                      uint32_t smem_bytes,
                      uint32_t *d_visits, uint32_t *d_claimed,
                      WorkerStats *d_stats) {
  CHECK(cudaMemset(d_visits, 0, tasks * sizeof(uint32_t)));
  CHECK(cudaMemset(d_claimed, 0, tasks * sizeof(uint32_t)));
  CHECK(cudaMemset(d_stats, 0, tasks * sizeof(WorkerStats)));
  clc_probe<<<tasks, threads, smem_bytes>>>(tasks, long_every, short_iters,
                                            long_iters, layout, smem_bytes,
                                            d_visits, d_claimed, d_stats);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemset(d_visits, 0, tasks * sizeof(uint32_t)));
  CHECK(cudaMemset(d_claimed, 0, tasks * sizeof(uint32_t)));
  CHECK(cudaMemset(d_stats, 0, tasks * sizeof(WorkerStats)));
  cudaEvent_t t0, t1;
  CHECK(cudaEventCreate(&t0));
  CHECK(cudaEventCreate(&t1));
  CHECK(cudaEventRecord(t0));
  clc_probe<<<tasks, threads, smem_bytes>>>(tasks, long_every, short_iters,
                                            long_iters, layout, smem_bytes,
                                            d_visits, d_claimed, d_stats);
  CHECK(cudaGetLastError());
  CHECK(cudaEventRecord(t1));
  CHECK(cudaEventSynchronize(t1));
  float ms = 0.f;
  CHECK(cudaEventElapsedTime(&ms, t0, t1));
  CHECK(cudaEventDestroy(t0));
  CHECK(cudaEventDestroy(t1));
  return 1000.f * ms;
}

int main(int argc, char **argv) {
  uint32_t tasks = argc > 1 ? std::atoi(argv[1]) : 4096;
  uint32_t threads = argc > 2 ? std::atoi(argv[2]) : 128;
  uint32_t long_every = argc > 3 ? std::atoi(argv[3]) : 8;
  uint32_t short_iters = argc > 4 ? std::atoi(argv[4]) : 256;
  uint32_t long_iters = argc > 5 ? std::atoi(argv[5]) : 8192;
  uint32_t layout = argc > 6 ? std::atoi(argv[6]) : LayoutInterleaved;
  uint32_t smem_bytes = argc > 7 ? std::atoi(argv[7]) : 0;
  if (!tasks || !threads) {
    std::fprintf(stderr, "usage: %s [tasks] [threads] [long_every] "
                         "[short_iters] [long_iters] [layout] [smem_bytes]\n",
                 argv[0]);
    return 2;
  }

  int dev = 0;
  cudaDeviceProp prop{};
  CHECK(cudaGetDevice(&dev));
  CHECK(cudaGetDeviceProperties(&prop, dev));
  int sm_count = 0;
  CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));
  std::printf("== CLC probe on %s sm_%d%d ==\n", prop.name, prop.major,
              prop.minor);
  std::printf("   tasks=%u threads=%u long_every=%u short_iters=%u "
              "long_iters=%u layout=%u smem_bytes=%u\n",
              tasks, threads, long_every, short_iters, long_iters, layout,
              smem_bytes);

  uint32_t *d_visits = nullptr, *d_claimed = nullptr;
  WorkerStats *d_stats = nullptr;
  CHECK(cudaMalloc(&d_visits, tasks * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_claimed, tasks * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_stats, tasks * sizeof(WorkerStats)));

  float static_us =
      time_static(tasks, threads, long_every, short_iters, long_iters, layout,
                  smem_bytes,
                  d_visits);
  float clc_us = time_clc(tasks, threads, long_every, short_iters, long_iters,
                          layout, smem_bytes, d_visits, d_claimed, d_stats);

  std::vector<uint32_t> visits(tasks), claimed(tasks);
  std::vector<WorkerStats> stats(tasks);
  CHECK(cudaMemcpy(visits.data(), d_visits, tasks * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(claimed.data(), d_claimed, tasks * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(stats.data(), d_stats, tasks * sizeof(WorkerStats),
                   cudaMemcpyDeviceToHost));

  uint64_t processed = 0, attempts = 0, successes = 0, cyc = 0, max_cyc = 0;
  uint32_t active_workers = 0, multi_claim_workers = 0, max_processed = 0;
  uint32_t max_successes_worker = 0;
  uint32_t first_claim_min = 0xffffffffu, first_claim_max = 0;
  uint32_t last_claim_min = 0xffffffffu, last_claim_max = 0;
  for (const WorkerStats &s : stats) {
    processed += s.processed;
    attempts += s.attempts;
    successes += s.successes;
    cyc += s.claim_cycles;
    max_cyc = std::max(max_cyc, s.max_claim_cycles);
    max_processed = std::max(max_processed, s.processed);
    max_successes_worker = std::max(max_successes_worker, s.successes);
    if (s.successes > 1)
      multi_claim_workers++;
    if (s.successes > 0) {
      first_claim_min = std::min(first_claim_min, s.first_claim);
      first_claim_max = std::max(first_claim_max, s.first_claim);
      last_claim_min = std::min(last_claim_min, s.last_claim);
      last_claim_max = std::max(last_claim_max, s.last_claim);
    }
    if (s.processed)
      active_workers++;
  }

  uint32_t missed = 0, duplicates = 0, claimed_unique = 0;
  uint32_t duplicate_claims = 0;
  uint32_t claimed_min = 0xffffffffu, claimed_max = 0;
  for (uint32_t i = 0; i < tasks; ++i) {
    missed += visits[i] == 0;
    duplicates += visits[i] > 1;
    if (claimed[i] > 0) {
      claimed_unique++;
      duplicate_claims += claimed[i] > 1;
      claimed_min = std::min(claimed_min, i);
      claimed_max = std::max(claimed_max, i);
    }
  }

  bool csv = std::getenv("CLC_PROBE_CSV") != nullptr;
  double success_rate =
      attempts ? 100.0 * (double)successes / (double)attempts : 0.0;
  double avg_claim = attempts ? (double)cyc / (double)attempts : 0.0;
  double delta = 100.f * (static_us - clc_us) / static_us;
  int occ_blocks = 0;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occ_blocks, clc_probe, threads, smem_bytes));
  uint32_t predicted_r = (uint32_t)occ_blocks * (uint32_t)sm_count;
  uint32_t expected_active = std::min(tasks, predicted_r);
  uint32_t expected_claimed = tasks > predicted_r ? tasks - predicted_r : 0;
  uint32_t claim_range_holes = 0;
  if (claimed_unique) {
    uint32_t claim_span = claimed_max - claimed_min + 1;
    claim_range_holes = claim_span - claimed_unique;
  }
  bool structural_ok = missed == 0 && duplicates == 0 &&
                       duplicate_claims == 0 && claim_range_holes == 0 &&
                       active_workers == expected_active;
  if (tasks <= predicted_r) {
    structural_ok = structural_ok && successes == 0 && claimed_unique == 0;
  } else {
    structural_ok = structural_ok && claimed_unique == expected_claimed &&
                    claimed_min == predicted_r && claimed_max == tasks - 1;
  }
  if (csv) {
    std::printf("tasks,threads,long_every,short_cycles,long_cycles,layout,static_us,"
                "clc_us,delta_pct,processed,active_workers,missed,duplicates,"
                "attempts,successes,success_rate,unique_claimed,claimed_min,"
                "claimed_max,claim_cycles_avg,claim_cycles_max,smem_bytes,"
                "occ_blocks_per_sm,predicted_r,max_processed,"
                "multi_claim_workers,max_successes_worker,first_claim_min,"
                "first_claim_max,last_claim_min,last_claim_max,sm_count,"
                "expected_active_workers,expected_claimed,duplicate_claims,"
                "claim_range_holes,structural_ok\n");
    std::printf("%u,%u,%u,%u,%u,%u,%.3f,%.3f,%.3f,%llu,%u,%u,%u,%llu,%llu,"
                "%.3f,%u,%u,%u,%.3f,%llu,%u,%d,%u,%u,%u,%u,%u,%u,%u,%u,%d,"
                "%u,%u,%u,%u,%u\n",
                tasks, threads, long_every, short_iters, long_iters, layout, static_us,
                clc_us, delta, (unsigned long long)processed, active_workers,
                missed, duplicates, (unsigned long long)attempts,
                (unsigned long long)successes, success_rate, claimed_unique,
                claimed_unique ? claimed_min : 0, claimed_unique ? claimed_max : 0,
                avg_claim, (unsigned long long)max_cyc, smem_bytes, occ_blocks,
                predicted_r, max_processed, multi_claim_workers,
                max_successes_worker,
                first_claim_min == 0xffffffffu ? 0 : first_claim_min,
                first_claim_max,
                last_claim_min == 0xffffffffu ? 0 : last_claim_min,
                last_claim_max, sm_count, expected_active, expected_claimed,
                duplicate_claims, claim_range_holes, structural_ok ? 1 : 0);
    CHECK(cudaFree(d_visits));
    CHECK(cudaFree(d_claimed));
    CHECK(cudaFree(d_stats));
    return missed || duplicates ? 1 : 0;
  }

  std::printf("   static_us=%.2f clc_us=%.2f delta=%+.1f%%\n", static_us,
              clc_us, delta);
  std::printf("   processed=%llu active_workers=%u missed=%u duplicates=%u\n",
              (unsigned long long)processed, active_workers, missed,
              duplicates);
  std::printf("   attempts=%llu successes=%llu success_rate=%.2f%% "
              "unique_claimed=%u\n",
              (unsigned long long)attempts, (unsigned long long)successes,
              success_rate, claimed_unique);
  std::printf("   occupancy blocks/SM=%d predicted_R=%u observed_active=%u\n",
              occ_blocks, predicted_r, active_workers);
  std::printf("   expected_active=%u expected_claimed=%u duplicate_claims=%u "
              "claim_range_holes=%u structural_ok=%s\n",
              expected_active, expected_claimed, duplicate_claims,
              claim_range_holes, structural_ok ? "yes" : "no");
  if (successes)
    std::printf("   first_claim=%u..%u last_claim=%u..%u "
                "multi_claim_workers=%u max_processed=%u\n",
                first_claim_min, first_claim_max, last_claim_min,
                last_claim_max, multi_claim_workers, max_processed);
  if (claimed_unique)
    std::printf("   claimed_raw_range=%u..%u\n", claimed_min, claimed_max);
  std::printf("   claim_cycles_avg=%.1f claim_cycles_max=%llu\n",
              avg_claim, (unsigned long long)max_cyc);

  std::printf("   first claimed raw ids:");
  uint32_t printed = 0;
  for (uint32_t i = 0; i < tasks && printed < 24; ++i) {
    if (claimed[i]) {
      std::printf(" %u", i);
      printed++;
    }
  }
  std::printf("%s\n", printed ? "" : " none");

  CHECK(cudaFree(d_visits));
  CHECK(cudaFree(d_claimed));
  CHECK(cudaFree(d_stats));
  return missed || duplicates ? 1 : 0;
}
