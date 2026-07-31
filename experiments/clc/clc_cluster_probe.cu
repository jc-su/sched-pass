// Standalone Blackwell CLC cluster-size probe.
//
// CLC cancels clusters, not abstract software tasks. This probe launches with
// runtime cluster dimensions and has exactly one CTA in each running cluster
// issue try_cancel. The whole running cluster then covers the canceled cluster's
// CTA range.

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

struct WorkerStatsCluster {
  uint32_t processed;
  uint32_t attempts;
  uint32_t successes;
  uint32_t first_claim_base;
  uint32_t last_claim_base;
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

__device__ __forceinline__ CtaId clc_try_cancel_cluster() {
  CtaId out;
#if __CUDA_ARCH__ >= 1000
  asm volatile(
      "{\n"
      "  .reg .pred %%pc;\n"
      "  .shared .align 16 .b8 _cluster_clc_res[16];\n"
      "  .shared .align 8 .b64 _cluster_clc_bar;\n"
      "  .reg .b128 %%rq;\n"
      "  .reg .b64 %%tmp;\n"
      "  .reg .b32 %%resa, %%bara, %%cx, %%cy, %%cz, %%cw, %%ok;\n"
      "  mov.u32 %%resa, _cluster_clc_res;\n"
      "  mov.u32 %%bara, _cluster_clc_bar;\n"
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
      "L_cluster_clc_wait:\n"
      "  mbarrier.try_wait.parity.shared::cta.b64 %%pc, [%%bara], 0;\n"
      "  @!%%pc bra L_cluster_clc_wait;\n"
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

__global__ void clc_cluster_probe(uint32_t tasks, uint32_t work_cycles,
                                  uint32_t cluster_x, uint32_t *visits,
                                  uint32_t *claimed_hist,
                                  uint32_t *claimed_base_hist,
                                  uint32_t *arrival_counts,
                                  uint32_t *claim_base_slots,
                                  uint32_t *claim_valid_slots,
                                  uint32_t *claim_epochs,
                                  WorkerStatsCluster *stats) {
  uint32_t rank = blockIdx.x % cluster_x;
  uint32_t slot = blockIdx.x / cluster_x;
  uint32_t raw = blockIdx.x;
  bool have_work = true;
  bool stolen = false;
  uint32_t iter = 0;

  if (threadIdx.x == 0) {
    stats[raw].processed = 0;
    stats[raw].attempts = 0;
    stats[raw].successes = 0;
    stats[raw].first_claim_base = 0xffffffffu;
    stats[raw].last_claim_base = 0xffffffffu;
  }
  __syncthreads();

  while (have_work) {
    if (raw < tasks) {
      burn(work_cycles);
      if (threadIdx.x == 0) {
        atomicAdd(&visits[raw], 1);
        if (stolen)
          atomicAdd(&claimed_hist[raw], 1);
        stats[blockIdx.x].processed++;
      }
    }

    __syncthreads();
    if (threadIdx.x == 0)
      atomicAdd(&arrival_counts[slot], 1);

    uint32_t target_arrivals = (iter + 1) * cluster_x;
    while (((volatile uint32_t *)arrival_counts)[slot] < target_arrivals)
      ;

    if (rank == 0 && threadIdx.x == 0) {
      CtaId claimed = clc_try_cancel_cluster();
      stats[blockIdx.x].attempts++;
      claim_base_slots[slot] = claimed.x;
      claim_valid_slots[slot] = claimed.canceled;
      if (claimed.canceled) {
        uint32_t base = claimed.x;
        stats[blockIdx.x].successes++;
        if (stats[blockIdx.x].first_claim_base == 0xffffffffu)
          stats[blockIdx.x].first_claim_base = base;
        stats[blockIdx.x].last_claim_base = base;
        if (base < tasks)
          atomicAdd(&claimed_base_hist[base], 1);
      }
      __threadfence();
      claim_epochs[slot] = iter + 1;
    }

    while (((volatile uint32_t *)claim_epochs)[slot] < iter + 1)
      ;

    uint32_t valid = ((volatile uint32_t *)claim_valid_slots)[slot];
    uint32_t base = ((volatile uint32_t *)claim_base_slots)[slot];
    have_work = valid != 0;
    if (have_work) {
      raw = base + rank;
      stolen = true;
    }
    iter++;
    __syncthreads();
  }
}

static void launch_cluster(uint32_t tasks, uint32_t threads, uint32_t cluster_x,
                           uint32_t work_cycles, uint32_t *d_visits,
                           uint32_t *d_claimed, uint32_t *d_claimed_base,
                           uint32_t *d_arrival, uint32_t *d_claim_base_slots,
                           uint32_t *d_claim_valid_slots,
                           uint32_t *d_claim_epochs,
                           WorkerStatsCluster *d_stats) {
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = cluster_x;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;

  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(tasks, 1, 1);
  cfg.blockDim = dim3(threads, 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = 0;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;

  CHECK(cudaLaunchKernelEx(&cfg, clc_cluster_probe, tasks, work_cycles,
                           cluster_x, d_visits, d_claimed, d_claimed_base,
                           d_arrival, d_claim_base_slots, d_claim_valid_slots,
                           d_claim_epochs, d_stats));
}

int main(int argc, char **argv) {
  uint32_t tasks = argc > 1 ? std::atoi(argv[1]) : 8192;
  uint32_t threads = argc > 2 ? std::atoi(argv[2]) : 128;
  uint32_t cluster_x = argc > 3 ? std::atoi(argv[3]) : 2;
  uint32_t work_cycles = argc > 4 ? std::atoi(argv[4]) : 4096;
  if (!tasks || !threads || !cluster_x || tasks % cluster_x) {
    std::fprintf(stderr,
                 "usage: %s [tasks divisible by cluster_x] [threads] "
                 "[cluster_x] [work_cycles]\n",
                 argv[0]);
    return 2;
  }

  int dev = 0;
  cudaDeviceProp prop{};
  CHECK(cudaGetDevice(&dev));
  CHECK(cudaGetDeviceProperties(&prop, dev));
  int sm_count = 0;
  CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));

  CHECK(cudaFuncSetAttribute(clc_cluster_probe,
                             cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

  uint32_t *d_visits = nullptr, *d_claimed = nullptr, *d_claimed_base = nullptr;
  uint32_t *d_arrival = nullptr, *d_claim_base_slots = nullptr;
  uint32_t *d_claim_valid_slots = nullptr, *d_claim_epochs = nullptr;
  WorkerStatsCluster *d_stats = nullptr;
  uint32_t cluster_slots = tasks / cluster_x;
  CHECK(cudaMalloc(&d_visits, tasks * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_claimed, tasks * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_claimed_base, tasks * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_arrival, cluster_slots * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_claim_base_slots, cluster_slots * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_claim_valid_slots, cluster_slots * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_claim_epochs, cluster_slots * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_stats, tasks * sizeof(WorkerStatsCluster)));
  CHECK(cudaMemset(d_visits, 0, tasks * sizeof(uint32_t)));
  CHECK(cudaMemset(d_claimed, 0, tasks * sizeof(uint32_t)));
  CHECK(cudaMemset(d_claimed_base, 0, tasks * sizeof(uint32_t)));
  CHECK(cudaMemset(d_arrival, 0, cluster_slots * sizeof(uint32_t)));
  CHECK(cudaMemset(d_claim_base_slots, 0, cluster_slots * sizeof(uint32_t)));
  CHECK(cudaMemset(d_claim_valid_slots, 0, cluster_slots * sizeof(uint32_t)));
  CHECK(cudaMemset(d_claim_epochs, 0, cluster_slots * sizeof(uint32_t)));
  CHECK(cudaMemset(d_stats, 0, tasks * sizeof(WorkerStatsCluster)));

  launch_cluster(tasks, threads, cluster_x, work_cycles, d_visits, d_claimed,
                 d_claimed_base, d_arrival, d_claim_base_slots,
                 d_claim_valid_slots, d_claim_epochs, d_stats);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  std::vector<uint32_t> visits(tasks), claimed(tasks), claimed_base(tasks);
  std::vector<WorkerStatsCluster> stats(tasks);
  CHECK(cudaMemcpy(visits.data(), d_visits, tasks * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(claimed.data(), d_claimed, tasks * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(claimed_base.data(), d_claimed_base,
                   tasks * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(stats.data(), d_stats, tasks * sizeof(WorkerStatsCluster),
                   cudaMemcpyDeviceToHost));

  cudaLaunchAttribute occ_attr{};
  occ_attr.id = cudaLaunchAttributeClusterDimension;
  occ_attr.val.clusterDim.x = cluster_x;
  occ_attr.val.clusterDim.y = 1;
  occ_attr.val.clusterDim.z = 1;
  cudaLaunchConfig_t occ_cfg{};
  occ_cfg.gridDim = dim3(tasks, 1, 1);
  occ_cfg.blockDim = dim3(threads, 1, 1);
  occ_cfg.dynamicSmemBytes = 0;
  occ_cfg.attrs = &occ_attr;
  occ_cfg.numAttrs = 1;
  int active_clusters = 0;
  CHECK(cudaOccupancyMaxActiveClusters(&active_clusters, clc_cluster_probe,
                                       &occ_cfg));
  uint32_t predicted_r = (uint32_t)active_clusters * cluster_x;

  uint64_t processed = 0, attempts = 0, successes = 0;
  uint32_t active_ctas = 0, active_leaders = 0, max_processed = 0;
  uint32_t first_claim_base_min = 0xffffffffu, first_claim_base_max = 0;
  uint32_t last_claim_base_min = 0xffffffffu, last_claim_base_max = 0;
  for (uint32_t i = 0; i < tasks; ++i) {
    const WorkerStatsCluster &s = stats[i];
    processed += s.processed;
    attempts += s.attempts;
    successes += s.successes;
    max_processed = std::max(max_processed, s.processed);
    if (s.processed)
      active_ctas++;
    if (s.attempts)
      active_leaders++;
    if (s.successes) {
      first_claim_base_min = std::min(first_claim_base_min, s.first_claim_base);
      first_claim_base_max = std::max(first_claim_base_max, s.first_claim_base);
      last_claim_base_min = std::min(last_claim_base_min, s.last_claim_base);
      last_claim_base_max = std::max(last_claim_base_max, s.last_claim_base);
    }
  }

  uint32_t missed = 0, duplicates = 0, claimed_unique = 0;
  uint32_t duplicate_claims = 0, claim_range_holes = 0;
  uint32_t claimed_min = 0xffffffffu, claimed_max = 0;
  uint32_t base_unique = 0, base_duplicates = 0;
  uint32_t base_min = 0xffffffffu, base_max = 0, base_alignment_errors = 0;
  for (uint32_t i = 0; i < tasks; ++i) {
    missed += visits[i] == 0;
    duplicates += visits[i] > 1;
    if (claimed[i]) {
      claimed_unique++;
      duplicate_claims += claimed[i] > 1;
      claimed_min = std::min(claimed_min, i);
      claimed_max = std::max(claimed_max, i);
    }
    if (claimed_base[i]) {
      base_unique++;
      base_duplicates += claimed_base[i] > 1;
      base_min = std::min(base_min, i);
      base_max = std::max(base_max, i);
      if (i % cluster_x)
        base_alignment_errors++;
    }
  }
  if (claimed_unique)
    claim_range_holes = (claimed_max - claimed_min + 1) - claimed_unique;

  uint32_t expected_active = std::min(tasks, predicted_r);
  uint32_t expected_claimed = tasks > predicted_r ? tasks - predicted_r : 0;
  uint32_t expected_claimed_clusters = expected_claimed / cluster_x;
  bool structural_ok = missed == 0 && duplicates == 0 &&
                       duplicate_claims == 0 && claim_range_holes == 0 &&
                       base_duplicates == 0 && base_alignment_errors == 0 &&
                       active_ctas == expected_active;
  if (tasks <= predicted_r) {
    structural_ok = structural_ok && successes == 0 && claimed_unique == 0 &&
                    base_unique == 0;
  } else {
    structural_ok = structural_ok && claimed_unique == expected_claimed &&
                    claimed_min == predicted_r && claimed_max == tasks - 1 &&
                    base_unique == expected_claimed_clusters &&
                    base_min == predicted_r &&
                    base_max == tasks - cluster_x;
  }

  if (std::getenv("CLC_PROBE_CSV")) {
    std::printf("tasks,threads,cluster_x,work_cycles,active_clusters,"
                "predicted_r,sm_count,processed,active_ctas,active_leaders,"
                "missed,duplicates,attempts,successes,unique_claimed,"
                "claimed_min,claimed_max,base_unique,base_min,base_max,"
                "base_duplicates,base_alignment_errors,claim_range_holes,"
                "first_claim_base_min,first_claim_base_max,last_claim_base_min,"
                "last_claim_base_max,max_processed,expected_active_ctas,"
                "expected_claimed,expected_claimed_clusters,structural_ok\n");
    std::printf("%u,%u,%u,%u,%d,%u,%d,%llu,%u,%u,%u,%u,%llu,%llu,%u,%u,%u,"
                "%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u\n",
                tasks, threads, cluster_x, work_cycles, active_clusters,
                predicted_r, sm_count, (unsigned long long)processed,
                active_ctas, active_leaders, missed, duplicates,
                (unsigned long long)attempts, (unsigned long long)successes,
                claimed_unique, claimed_unique ? claimed_min : 0,
                claimed_unique ? claimed_max : 0, base_unique,
                base_unique ? base_min : 0, base_unique ? base_max : 0,
                base_duplicates, base_alignment_errors, claim_range_holes,
                first_claim_base_min == 0xffffffffu ? 0 : first_claim_base_min,
                first_claim_base_max,
                last_claim_base_min == 0xffffffffu ? 0 : last_claim_base_min,
                last_claim_base_max, max_processed, expected_active,
                expected_claimed, expected_claimed_clusters,
                structural_ok ? 1 : 0);
  } else {
    std::printf("== CLC cluster probe on %s sm_%d%d ==\n", prop.name,
                prop.major, prop.minor);
    std::printf("   tasks=%u threads=%u cluster_x=%u work_cycles=%u\n", tasks,
                threads, cluster_x, work_cycles);
    std::printf("   active_clusters=%d predicted_R_cta=%u active_ctas=%u "
                "active_leaders=%u\n",
                active_clusters, predicted_r, active_ctas, active_leaders);
    std::printf("   processed=%llu missed=%u duplicates=%u attempts=%llu "
                "successes=%llu\n",
                (unsigned long long)processed, missed, duplicates,
                (unsigned long long)attempts, (unsigned long long)successes);
    if (claimed_unique)
      std::printf("   claimed_cta=%u..%u claimed_bases=%u..%u bases=%u\n",
                  claimed_min, claimed_max, base_min, base_max, base_unique);
    std::printf("   first_base=%u..%u last_base=%u..%u max_processed=%u "
                "structural_ok=%s\n",
                first_claim_base_min == 0xffffffffu ? 0 : first_claim_base_min,
                first_claim_base_max,
                last_claim_base_min == 0xffffffffu ? 0 : last_claim_base_min,
                last_claim_base_max, max_processed,
                structural_ok ? "yes" : "no");
  }

  CHECK(cudaFree(d_visits));
  CHECK(cudaFree(d_claimed));
  CHECK(cudaFree(d_claimed_base));
  CHECK(cudaFree(d_arrival));
  CHECK(cudaFree(d_claim_base_slots));
  CHECK(cudaFree(d_claim_valid_slots));
  CHECK(cudaFree(d_claim_epochs));
  CHECK(cudaFree(d_stats));
  return structural_ok ? 0 : 1;
}
