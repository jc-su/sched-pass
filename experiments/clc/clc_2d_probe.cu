// Standalone Blackwell CLC 2D-grid probe.
//
// This probe records the full CTA id returned by
// clusterlaunchcontrol.query_cancel.get_first_ctaid.v4. It tests whether CLC's
// observed suffix behavior is tied to CUDA's linear block order for 2D grids.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct CtaId {
  uint32_t x;
  uint32_t y;
  uint32_t z;
  uint32_t canceled;
};

struct WorkerStats2D {
  uint32_t processed;
  uint32_t attempts;
  uint32_t successes;
  uint32_t first_linear;
  uint32_t last_linear;
  uint32_t first_x;
  uint32_t first_y;
  uint32_t last_x;
  uint32_t last_y;
  uint64_t claim_cycles;
  uint64_t max_claim_cycles;
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

__device__ __forceinline__ uint32_t linear_id(uint32_t x, uint32_t y,
                                              uint32_t z, uint32_t grid_x,
                                              uint32_t grid_y) {
  return x + y * grid_x + z * grid_x * grid_y;
}

__device__ __forceinline__ CtaId clc_try_cancel_3d() {
  CtaId out;
#if __CUDA_ARCH__ >= 1000
  asm volatile(
      "{\n"
      "  .reg .pred %%pc;\n"
      "  .shared .align 16 .b8 _probe2d_clc_res[16];\n"
      "  .shared .align 8 .b64 _probe2d_clc_bar;\n"
      "  .reg .b128 %%rq;\n"
      "  .reg .b64 %%tmp;\n"
      "  .reg .b32 %%resa, %%bara, %%cx, %%cy, %%cz, %%cw, %%ok;\n"
      "  mov.u32 %%resa, _probe2d_clc_res;\n"
      "  mov.u32 %%bara, _probe2d_clc_bar;\n"
      "  mov.u32 %%cx, 0xffffffff;\n"
      "  mov.u32 %%cy, 0xffffffff;\n"
      "  mov.u32 %%cz, 0xffffffff;\n"
      "  mov.u32 %%ok, 0;\n"
      "  mbarrier.init.shared::cta.b64 [%%bara], 1;\n"
      "  fence.proxy.async.shared::cta;\n"
      "  clusterlaunchcontrol.try_cancel.async.shared::cta."
      "mbarrier::complete_tx::bytes.b128 [%%resa], [%%bara];\n"
      "  mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 "
      "%%tmp, [%%bara], 16;\n"
      "L_probe2d_clc_wait:\n"
      "  mbarrier.try_wait.parity.shared::cta.b64 %%pc, [%%bara], 0;\n"
      "  @!%%pc bra L_probe2d_clc_wait;\n"
      "  ld.shared.b128 %%rq, [%%resa];\n"
      "  clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 %%pc, %%rq;\n"
      "  selp.u32 %%ok, 1, 0, %%pc;\n"
      "  @%%pc clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 "
      "{%%cx, %%cy, %%cz, %%cw}, %%rq;\n"
      "  mov.u32 %0, %%cx;\n"
      "  mov.u32 %1, %%cy;\n"
      "  mov.u32 %2, %%cz;\n"
      "  mov.u32 %3, %%ok;\n"
      "}\n"
      : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.canceled)
      :
      : "memory");
#else
  out.x = out.y = out.z = 0xffffffffu;
  out.canceled = 0;
#endif
  return out;
}

__global__ void static_2d_probe(uint32_t grid_x, uint32_t grid_y,
                                uint32_t work_cycles, uint32_t smem_bytes,
                                uint32_t *visits) {
  extern __shared__ unsigned char dyn_smem[];
  if (threadIdx.x == 0 && smem_bytes)
    dyn_smem[0] = (unsigned char)blockIdx.x;
  uint32_t raw = linear_id(blockIdx.x, blockIdx.y, blockIdx.z, grid_x, grid_y);
  burn(work_cycles);
  if (threadIdx.x == 0)
    atomicAdd(&visits[raw], 1);
}

__global__ void clc_2d_probe(uint32_t grid_x, uint32_t grid_y,
                             uint32_t tasks, uint32_t work_cycles,
                             uint32_t smem_bytes, uint32_t *visits,
                             uint32_t *claimed_hist, WorkerStats2D *stats) {
  extern __shared__ unsigned char dyn_smem[];
  __shared__ CtaId next;

  const uint32_t worker =
      linear_id(blockIdx.x, blockIdx.y, blockIdx.z, grid_x, grid_y);
  CtaId cur{blockIdx.x, blockIdx.y, blockIdx.z, 1};

  if (threadIdx.x == 0) {
    if (smem_bytes)
      dyn_smem[0] = (unsigned char)worker;
    stats[worker].processed = 0;
    stats[worker].attempts = 0;
    stats[worker].successes = 0;
    stats[worker].first_linear = 0xffffffffu;
    stats[worker].last_linear = 0xffffffffu;
    stats[worker].first_x = 0xffffffffu;
    stats[worker].first_y = 0xffffffffu;
    stats[worker].last_x = 0xffffffffu;
    stats[worker].last_y = 0xffffffffu;
    stats[worker].claim_cycles = 0;
    stats[worker].max_claim_cycles = 0;
  }
  __syncthreads();

  while (cur.canceled) {
    uint32_t raw = linear_id(cur.x, cur.y, cur.z, grid_x, grid_y);
    if (raw >= tasks)
      break;

    burn(work_cycles);
    if (threadIdx.x == 0) {
      atomicAdd(&visits[raw], 1);
      stats[worker].processed++;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      uint64_t t0 = clock64_dev();
      CtaId claimed = clc_try_cancel_3d();
      uint64_t dt = clock64_dev() - t0;
      next = claimed;

      stats[worker].attempts++;
      stats[worker].claim_cycles += dt;
      stats[worker].max_claim_cycles =
          max(stats[worker].max_claim_cycles, dt);
      if (claimed.canceled) {
        uint32_t claimed_linear =
            linear_id(claimed.x, claimed.y, claimed.z, grid_x, grid_y);
        stats[worker].successes++;
        if (stats[worker].first_linear == 0xffffffffu) {
          stats[worker].first_linear = claimed_linear;
          stats[worker].first_x = claimed.x;
          stats[worker].first_y = claimed.y;
        }
        stats[worker].last_linear = claimed_linear;
        stats[worker].last_x = claimed.x;
        stats[worker].last_y = claimed.y;
        if (claimed_linear < tasks)
          atomicAdd(&claimed_hist[claimed_linear], 1);
      }
    }
    __syncthreads();
    cur = next;
  }
}

static float time_static(uint32_t grid_x, uint32_t grid_y, uint32_t threads,
                         uint32_t work_cycles, uint32_t smem_bytes,
                         uint32_t *d_visits) {
  uint32_t tasks = grid_x * grid_y;
  dim3 grid(grid_x, grid_y, 1);
  CHECK(cudaMemset(d_visits, 0, tasks * sizeof(uint32_t)));
  static_2d_probe<<<grid, threads, smem_bytes>>>(grid_x, grid_y, work_cycles,
                                                smem_bytes, d_visits);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemset(d_visits, 0, tasks * sizeof(uint32_t)));

  cudaEvent_t t0, t1;
  CHECK(cudaEventCreate(&t0));
  CHECK(cudaEventCreate(&t1));
  CHECK(cudaEventRecord(t0));
  static_2d_probe<<<grid, threads, smem_bytes>>>(grid_x, grid_y, work_cycles,
                                                smem_bytes, d_visits);
  CHECK(cudaGetLastError());
  CHECK(cudaEventRecord(t1));
  CHECK(cudaEventSynchronize(t1));
  float ms = 0.f;
  CHECK(cudaEventElapsedTime(&ms, t0, t1));
  CHECK(cudaEventDestroy(t0));
  CHECK(cudaEventDestroy(t1));
  return 1000.f * ms;
}

static float time_clc(uint32_t grid_x, uint32_t grid_y, uint32_t threads,
                      uint32_t work_cycles, uint32_t smem_bytes,
                      uint32_t *d_visits, uint32_t *d_claimed,
                      WorkerStats2D *d_stats) {
  uint32_t tasks = grid_x * grid_y;
  dim3 grid(grid_x, grid_y, 1);
  CHECK(cudaMemset(d_visits, 0, tasks * sizeof(uint32_t)));
  CHECK(cudaMemset(d_claimed, 0, tasks * sizeof(uint32_t)));
  CHECK(cudaMemset(d_stats, 0, tasks * sizeof(WorkerStats2D)));
  clc_2d_probe<<<grid, threads, smem_bytes>>>(grid_x, grid_y, tasks,
                                             work_cycles, smem_bytes, d_visits,
                                             d_claimed, d_stats);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemset(d_visits, 0, tasks * sizeof(uint32_t)));
  CHECK(cudaMemset(d_claimed, 0, tasks * sizeof(uint32_t)));
  CHECK(cudaMemset(d_stats, 0, tasks * sizeof(WorkerStats2D)));

  cudaEvent_t t0, t1;
  CHECK(cudaEventCreate(&t0));
  CHECK(cudaEventCreate(&t1));
  CHECK(cudaEventRecord(t0));
  clc_2d_probe<<<grid, threads, smem_bytes>>>(grid_x, grid_y, tasks,
                                             work_cycles, smem_bytes, d_visits,
                                             d_claimed, d_stats);
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
  uint32_t grid_x = argc > 1 ? std::atoi(argv[1]) : 64;
  uint32_t grid_y = argc > 2 ? std::atoi(argv[2]) : 128;
  uint32_t threads = argc > 3 ? std::atoi(argv[3]) : 128;
  uint32_t work_cycles = argc > 4 ? std::atoi(argv[4]) : 4096;
  uint32_t smem_bytes = argc > 5 ? std::atoi(argv[5]) : 0;
  if (!grid_x || !grid_y || !threads) {
    std::fprintf(stderr,
                 "usage: %s [grid_x] [grid_y] [threads] [work_cycles] "
                 "[smem_bytes]\n",
                 argv[0]);
    return 2;
  }

  uint32_t tasks = grid_x * grid_y;
  int dev = 0;
  cudaDeviceProp prop{};
  CHECK(cudaGetDevice(&dev));
  CHECK(cudaGetDeviceProperties(&prop, dev));
  int sm_count = 0;
  CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));

  std::printf("== CLC 2D probe on %s sm_%d%d ==\n", prop.name, prop.major,
              prop.minor);
  std::printf("   grid=%ux%u tasks=%u threads=%u work_cycles=%u "
              "smem_bytes=%u\n",
              grid_x, grid_y, tasks, threads, work_cycles, smem_bytes);

  uint32_t *d_visits = nullptr, *d_claimed = nullptr;
  WorkerStats2D *d_stats = nullptr;
  CHECK(cudaMalloc(&d_visits, tasks * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_claimed, tasks * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_stats, tasks * sizeof(WorkerStats2D)));

  float static_us =
      time_static(grid_x, grid_y, threads, work_cycles, smem_bytes, d_visits);
  float clc_us = time_clc(grid_x, grid_y, threads, work_cycles, smem_bytes,
                          d_visits, d_claimed, d_stats);

  std::vector<uint32_t> visits(tasks), claimed(tasks);
  std::vector<WorkerStats2D> stats(tasks);
  CHECK(cudaMemcpy(visits.data(), d_visits, tasks * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(claimed.data(), d_claimed, tasks * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(stats.data(), d_stats, tasks * sizeof(WorkerStats2D),
                   cudaMemcpyDeviceToHost));

  uint64_t processed = 0, attempts = 0, successes = 0, cyc = 0, max_cyc = 0;
  uint32_t active_workers = 0, multi_claim_workers = 0, max_processed = 0;
  uint32_t first_claim_min = 0xffffffffu, first_claim_max = 0;
  uint32_t last_claim_min = 0xffffffffu, last_claim_max = 0;
  for (const WorkerStats2D &s : stats) {
    processed += s.processed;
    attempts += s.attempts;
    successes += s.successes;
    cyc += s.claim_cycles;
    max_cyc = std::max(max_cyc, s.max_claim_cycles);
    max_processed = std::max(max_processed, s.processed);
    if (s.successes > 1)
      multi_claim_workers++;
    if (s.successes > 0) {
      first_claim_min = std::min(first_claim_min, s.first_linear);
      first_claim_max = std::max(first_claim_max, s.first_linear);
      last_claim_min = std::min(last_claim_min, s.last_linear);
      last_claim_max = std::max(last_claim_max, s.last_linear);
    }
    if (s.processed)
      active_workers++;
  }

  uint32_t missed = 0, duplicates = 0, claimed_unique = 0;
  uint32_t duplicate_claims = 0, claim_range_holes = 0;
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
  if (claimed_unique)
    claim_range_holes = (claimed_max - claimed_min + 1) - claimed_unique;

  int occ_blocks = 0;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occ_blocks, clc_2d_probe, threads, smem_bytes));
  uint32_t predicted_r = (uint32_t)occ_blocks * (uint32_t)sm_count;
  uint32_t expected_active = std::min(tasks, predicted_r);
  uint32_t expected_claimed = tasks > predicted_r ? tasks - predicted_r : 0;
  bool structural_ok = missed == 0 && duplicates == 0 &&
                       duplicate_claims == 0 && claim_range_holes == 0 &&
                       active_workers == expected_active;
  if (tasks <= predicted_r) {
    structural_ok = structural_ok && successes == 0 && claimed_unique == 0;
  } else {
    structural_ok = structural_ok && claimed_unique == expected_claimed &&
                    claimed_min == predicted_r && claimed_max == tasks - 1;
  }

  uint32_t claimed_min_x = claimed_unique ? claimed_min % grid_x : 0;
  uint32_t claimed_min_y = claimed_unique ? claimed_min / grid_x : 0;
  uint32_t claimed_max_x = claimed_unique ? claimed_max % grid_x : 0;
  uint32_t claimed_max_y = claimed_unique ? claimed_max / grid_x : 0;
  uint32_t r_x = predicted_r < tasks ? predicted_r % grid_x : 0;
  uint32_t r_y = predicted_r < tasks ? predicted_r / grid_x : 0;

  double success_rate =
      attempts ? 100.0 * (double)successes / (double)attempts : 0.0;
  double avg_claim = attempts ? (double)cyc / (double)attempts : 0.0;
  double delta = 100.0 * (double)(static_us - clc_us) / (double)static_us;

  if (std::getenv("CLC_PROBE_CSV")) {
    std::printf("grid_x,grid_y,tasks,threads,work_cycles,smem_bytes,static_us,"
                "clc_us,delta_pct,processed,active_workers,missed,duplicates,"
                "attempts,successes,success_rate,unique_claimed,claimed_min,"
                "claimed_max,claimed_min_x,claimed_min_y,claimed_max_x,"
                "claimed_max_y,claim_cycles_avg,claim_cycles_max,"
                "occ_blocks_per_sm,predicted_r,sm_count,expected_active_workers,"
                "expected_claimed,r_x,r_y,first_claim_min,first_claim_max,"
                "last_claim_min,last_claim_max,max_processed,"
                "multi_claim_workers,duplicate_claims,claim_range_holes,"
                "structural_ok\n");
    std::printf("%u,%u,%u,%u,%u,%u,%.3f,%.3f,%.3f,%llu,%u,%u,%u,%llu,%llu,"
                "%.3f,%u,%u,%u,%u,%u,%u,%u,%.3f,%llu,%d,%u,%d,%u,%u,%u,%u,"
                "%u,%u,%u,%u,%u,%u,%u,%u,%u\n",
                grid_x, grid_y, tasks, threads, work_cycles, smem_bytes,
                static_us, clc_us, delta, (unsigned long long)processed,
                active_workers, missed, duplicates,
                (unsigned long long)attempts, (unsigned long long)successes,
                success_rate, claimed_unique, claimed_unique ? claimed_min : 0,
                claimed_unique ? claimed_max : 0, claimed_min_x, claimed_min_y,
                claimed_max_x, claimed_max_y, avg_claim,
                (unsigned long long)max_cyc, occ_blocks, predicted_r, sm_count,
                expected_active, expected_claimed, r_x, r_y,
                first_claim_min == 0xffffffffu ? 0 : first_claim_min,
                first_claim_max,
                last_claim_min == 0xffffffffu ? 0 : last_claim_min,
                last_claim_max, max_processed, multi_claim_workers,
                duplicate_claims, claim_range_holes, structural_ok ? 1 : 0);
  } else {
    std::printf("   static_us=%.2f clc_us=%.2f delta=%+.1f%%\n", static_us,
                clc_us, delta);
    std::printf("   occupancy blocks/SM=%d predicted_R=%u R_coord=(%u,%u) "
                "observed_active=%u\n",
                occ_blocks, predicted_r, r_x, r_y, active_workers);
    std::printf("   processed=%llu missed=%u duplicates=%u attempts=%llu "
                "successes=%llu success_rate=%.2f%%\n",
                (unsigned long long)processed, missed, duplicates,
                (unsigned long long)attempts, (unsigned long long)successes,
                success_rate);
    if (claimed_unique)
      std::printf("   claimed_linear=%u..%u claimed_coord=(%u,%u)..(%u,%u)\n",
                  claimed_min, claimed_max, claimed_min_x, claimed_min_y,
                  claimed_max_x, claimed_max_y);
    std::printf("   first_claim=%u..%u last_claim=%u..%u "
                "max_processed=%u multi_claim_workers=%u\n",
                first_claim_min == 0xffffffffu ? 0 : first_claim_min,
                first_claim_max,
                last_claim_min == 0xffffffffu ? 0 : last_claim_min,
                last_claim_max, max_processed, multi_claim_workers);
    std::printf("   duplicate_claims=%u claim_range_holes=%u "
                "structural_ok=%s claim_cycles_avg=%.1f max=%llu\n",
                duplicate_claims, claim_range_holes,
                structural_ok ? "yes" : "no", avg_claim,
                (unsigned long long)max_cyc);
  }

  CHECK(cudaFree(d_visits));
  CHECK(cudaFree(d_claimed));
  CHECK(cudaFree(d_stats));
  return missed || duplicates || !structural_ok ? 1 : 0;
}
