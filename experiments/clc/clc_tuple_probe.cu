// Standalone Blackwell CLC tuple probe.
//
// Captures the full get_first_ctaid.v4 tuple for 1D/2D/3D grids and checks
// whether CLC linearizes CTA ids as x + y*grid_x + z*grid_x*grid_y.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct CtaTuple {
  uint32_t x;
  uint32_t y;
  uint32_t z;
  uint32_t w;
  uint32_t canceled;
};

struct TupleStats {
  uint32_t processed;
  uint32_t attempts;
  uint32_t successes;
  uint32_t first_linear;
  uint32_t last_linear;
  uint32_t w_min;
  uint32_t w_max;
  uint32_t w_nonzero;
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
    asm volatile("add.u32 %0, %0, 1;" : "+r"(x) :: "memory");
}

__device__ __forceinline__ uint32_t linear_id(uint32_t x, uint32_t y,
                                              uint32_t z, uint32_t gx,
                                              uint32_t gy) {
  return x + y * gx + z * gx * gy;
}

__device__ __forceinline__ CtaTuple clc_try_cancel_tuple() {
  CtaTuple out;
#if __CUDA_ARCH__ >= 1000
  asm volatile(
      "{\n"
      "  .reg .pred %%pc;\n"
      "  .shared .align 16 .b8 _tuple_clc_res[16];\n"
      "  .shared .align 8 .b64 _tuple_clc_bar;\n"
      "  .reg .b128 %%rq;\n"
      "  .reg .b64 %%tmp;\n"
      "  .reg .b32 %%resa, %%bara, %%cx, %%cy, %%cz, %%cw, %%ok;\n"
      "  mov.u32 %%resa, _tuple_clc_res;\n"
      "  mov.u32 %%bara, _tuple_clc_bar;\n"
      "  mov.u32 %%cx, 0xffffffff;\n"
      "  mov.u32 %%cy, 0xffffffff;\n"
      "  mov.u32 %%cz, 0xffffffff;\n"
      "  mov.u32 %%cw, 0xffffffff;\n"
      "  mov.u32 %%ok, 0;\n"
      "  mbarrier.init.shared::cta.b64 [%%bara], 1;\n"
      "  fence.proxy.async.shared::cta;\n"
      "  clusterlaunchcontrol.try_cancel.async.shared::cta."
      "mbarrier::complete_tx::bytes.b128 [%%resa], [%%bara];\n"
      "  mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 "
      "%%tmp, [%%bara], 16;\n"
      "L_tuple_clc_wait:\n"
      "  mbarrier.try_wait.parity.shared::cta.b64 %%pc, [%%bara], 0;\n"
      "  @!%%pc bra L_tuple_clc_wait;\n"
      "  ld.shared.b128 %%rq, [%%resa];\n"
      "  clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 %%pc, %%rq;\n"
      "  selp.u32 %%ok, 1, 0, %%pc;\n"
      "  @%%pc clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 "
      "{%%cx, %%cy, %%cz, %%cw}, %%rq;\n"
      "  mov.u32 %0, %%cx;\n"
      "  mov.u32 %1, %%cy;\n"
      "  mov.u32 %2, %%cz;\n"
      "  mov.u32 %3, %%cw;\n"
      "  mov.u32 %4, %%ok;\n"
      "}\n"
      : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w),
        "=r"(out.canceled)
      :
      : "memory");
#else
  out.x = out.y = out.z = out.w = 0xffffffffu;
  out.canceled = 0;
#endif
  return out;
}

__global__ void clc_tuple_probe(uint32_t gx, uint32_t gy, uint32_t gz,
                                uint32_t tasks, uint32_t work_cycles,
                                uint32_t *visits, uint32_t *claimed_hist,
                                TupleStats *stats, uint32_t *w_hist) {
  __shared__ CtaTuple next;
  uint32_t worker = linear_id(blockIdx.x, blockIdx.y, blockIdx.z, gx, gy);
  CtaTuple cur{blockIdx.x, blockIdx.y, blockIdx.z, 0, 1};

  if (threadIdx.x == 0) {
    stats[worker].processed = 0;
    stats[worker].attempts = 0;
    stats[worker].successes = 0;
    stats[worker].first_linear = 0xffffffffu;
    stats[worker].last_linear = 0xffffffffu;
    stats[worker].w_min = 0xffffffffu;
    stats[worker].w_max = 0;
    stats[worker].w_nonzero = 0;
  }
  __syncthreads();

  while (cur.canceled) {
    uint32_t raw = linear_id(cur.x, cur.y, cur.z, gx, gy);
    if (raw >= tasks)
      break;

    burn(work_cycles);
    if (threadIdx.x == 0) {
      atomicAdd(&visits[raw], 1);
      stats[worker].processed++;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      CtaTuple claimed = clc_try_cancel_tuple();
      next = claimed;
      stats[worker].attempts++;
      if (claimed.canceled) {
        uint32_t lin = linear_id(claimed.x, claimed.y, claimed.z, gx, gy);
        stats[worker].successes++;
        if (stats[worker].first_linear == 0xffffffffu)
          stats[worker].first_linear = lin;
        stats[worker].last_linear = lin;
        stats[worker].w_min = min(stats[worker].w_min, claimed.w);
        stats[worker].w_max = max(stats[worker].w_max, claimed.w);
        stats[worker].w_nonzero += claimed.w != 0;
        if (lin < tasks)
          atomicAdd(&claimed_hist[lin], 1);
        atomicAdd(&w_hist[claimed.w & 255u], 1);
      }
    }
    __syncthreads();
    cur = next;
  }
}

int main(int argc, char **argv) {
  uint32_t gx = argc > 1 ? std::atoi(argv[1]) : 64;
  uint32_t gy = argc > 2 ? std::atoi(argv[2]) : 16;
  uint32_t gz = argc > 3 ? std::atoi(argv[3]) : 8;
  uint32_t threads = argc > 4 ? std::atoi(argv[4]) : 128;
  uint32_t work_cycles = argc > 5 ? std::atoi(argv[5]) : 4096;
  if (!gx || !gy || !gz || !threads) {
    std::fprintf(stderr,
                 "usage: %s [grid_x] [grid_y] [grid_z] [threads] "
                 "[work_cycles]\n",
                 argv[0]);
    return 2;
  }

  uint32_t tasks = gx * gy * gz;
  dim3 grid(gx, gy, gz);
  int dev = 0;
  cudaDeviceProp prop{};
  CHECK(cudaGetDevice(&dev));
  CHECK(cudaGetDeviceProperties(&prop, dev));
  int sm_count = 0;
  CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));

  uint32_t *d_visits = nullptr, *d_claimed = nullptr, *d_w_hist = nullptr;
  TupleStats *d_stats = nullptr;
  CHECK(cudaMalloc(&d_visits, tasks * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_claimed, tasks * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_stats, tasks * sizeof(TupleStats)));
  CHECK(cudaMalloc(&d_w_hist, 256 * sizeof(uint32_t)));
  CHECK(cudaMemset(d_visits, 0, tasks * sizeof(uint32_t)));
  CHECK(cudaMemset(d_claimed, 0, tasks * sizeof(uint32_t)));
  CHECK(cudaMemset(d_stats, 0, tasks * sizeof(TupleStats)));
  CHECK(cudaMemset(d_w_hist, 0, 256 * sizeof(uint32_t)));

  clc_tuple_probe<<<grid, threads>>>(gx, gy, gz, tasks, work_cycles, d_visits,
                                     d_claimed, d_stats, d_w_hist);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  std::vector<uint32_t> visits(tasks), claimed(tasks), w_hist(256);
  std::vector<TupleStats> stats(tasks);
  CHECK(cudaMemcpy(visits.data(), d_visits, tasks * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(claimed.data(), d_claimed, tasks * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(stats.data(), d_stats, tasks * sizeof(TupleStats),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(w_hist.data(), d_w_hist, 256 * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost));

  uint64_t processed = 0, attempts = 0, successes = 0, w_nonzero = 0;
  uint32_t active_workers = 0, max_processed = 0;
  uint32_t first_claim_min = 0xffffffffu, first_claim_max = 0;
  uint32_t last_claim_min = 0xffffffffu, last_claim_max = 0;
  uint32_t w_min = 0xffffffffu, w_max = 0;
  for (const TupleStats &s : stats) {
    processed += s.processed;
    attempts += s.attempts;
    successes += s.successes;
    w_nonzero += s.w_nonzero;
    max_processed = std::max(max_processed, s.processed);
    if (s.processed)
      active_workers++;
    if (s.successes) {
      first_claim_min = std::min(first_claim_min, s.first_linear);
      first_claim_max = std::max(first_claim_max, s.first_linear);
      last_claim_min = std::min(last_claim_min, s.last_linear);
      last_claim_max = std::max(last_claim_max, s.last_linear);
      w_min = std::min(w_min, s.w_min);
      w_max = std::max(w_max, s.w_max);
    }
  }

  uint32_t missed = 0, duplicates = 0, claimed_unique = 0;
  uint32_t duplicate_claims = 0, claim_range_holes = 0;
  uint32_t claimed_min = 0xffffffffu, claimed_max = 0;
  for (uint32_t i = 0; i < tasks; ++i) {
    missed += visits[i] == 0;
    duplicates += visits[i] > 1;
    if (claimed[i]) {
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
      &occ_blocks, clc_tuple_probe, threads, 0));
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

  uint32_t w_bins = 0;
  for (uint32_t v : w_hist)
    w_bins += v != 0;

  if (std::getenv("CLC_PROBE_CSV")) {
    std::printf("grid_x,grid_y,grid_z,tasks,threads,work_cycles,processed,"
                "active_workers,missed,duplicates,attempts,successes,"
                "unique_claimed,claimed_min,claimed_max,first_claim_min,"
                "first_claim_max,last_claim_min,last_claim_max,w_min,w_max,"
                "w_nonzero,w_bins,occ_blocks_per_sm,predicted_r,sm_count,"
                "expected_active_workers,expected_claimed,max_processed,"
                "duplicate_claims,claim_range_holes,structural_ok\n");
    std::printf("%u,%u,%u,%u,%u,%u,%llu,%u,%u,%u,%llu,%llu,%u,%u,%u,%u,%u,%u,"
                "%u,%u,%u,%llu,%u,%d,%u,%d,%u,%u,%u,%u,%u,%u\n",
                gx, gy, gz, tasks, threads, work_cycles,
                (unsigned long long)processed, active_workers, missed,
                duplicates, (unsigned long long)attempts,
                (unsigned long long)successes, claimed_unique,
                claimed_unique ? claimed_min : 0, claimed_unique ? claimed_max : 0,
                first_claim_min == 0xffffffffu ? 0 : first_claim_min,
                first_claim_max,
                last_claim_min == 0xffffffffu ? 0 : last_claim_min,
                last_claim_max, w_min == 0xffffffffu ? 0 : w_min, w_max,
                (unsigned long long)w_nonzero, w_bins, occ_blocks, predicted_r,
                sm_count, expected_active, expected_claimed, max_processed,
                duplicate_claims, claim_range_holes, structural_ok ? 1 : 0);
  } else {
    std::printf("== CLC tuple probe on %s sm_%d%d ==\n", prop.name, prop.major,
                prop.minor);
    std::printf("   grid=%ux%ux%u tasks=%u R=%u active=%u\n", gx, gy, gz,
                tasks, predicted_r, active_workers);
    std::printf("   claimed=%u..%u missed=%u duplicates=%u structural_ok=%s\n",
                claimed_unique ? claimed_min : 0, claimed_unique ? claimed_max : 0,
                missed, duplicates, structural_ok ? "yes" : "no");
    std::printf("   first=%u..%u last=%u..%u w=%u..%u w_nonzero=%llu bins=%u\n",
                first_claim_min == 0xffffffffu ? 0 : first_claim_min,
                first_claim_max,
                last_claim_min == 0xffffffffu ? 0 : last_claim_min,
                last_claim_max, w_min == 0xffffffffu ? 0 : w_min, w_max,
                (unsigned long long)w_nonzero, w_bins);
  }

  CHECK(cudaFree(d_visits));
  CHECK(cudaFree(d_claimed));
  CHECK(cudaFree(d_stats));
  CHECK(cudaFree(d_w_hist));
  return structural_ok ? 0 : 1;
}
