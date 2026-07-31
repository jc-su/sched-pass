// Standalone Blackwell CLC runtime-behavior probe.
//
// Measures the observable runtime contract around CLC attempts:
// - success/failure counts and latency
// - terminal failed-attempt behavior
// - active worker distribution across SMs
// - whether a CTA observes a stable SM id while it processes claimed work

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <numeric>
#include <vector>

struct RuntimeStats {
  uint32_t processed;
  uint32_t attempts;
  uint32_t successes;
  uint32_t failures;
  uint32_t initial_smid;
  uint32_t last_smid;
  uint32_t smid_changes;
  uint32_t first_claim;
  uint32_t last_claim;
  uint64_t success_cycles_sum;
  uint64_t failure_cycles_sum;
  uint32_t success_cycles_min;
  uint32_t success_cycles_max;
  uint32_t failure_cycles_min;
  uint32_t failure_cycles_max;
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

__device__ __forceinline__ uint32_t smid_dev() {
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return smid;
}

__device__ __forceinline__ void burn(uint32_t cycles) {
  uint64_t start = clock64_dev();
  uint32_t x = threadIdx.x;
  while ((uint64_t)(clock64_dev() - start) < cycles)
    asm volatile("add.u32 %0, %0, 1;" : "+r"(x) :: "memory");
}

__device__ __forceinline__ uint32_t clc_try_cancel_1d(uint32_t fallback) {
  uint32_t out;
#if __CUDA_ARCH__ >= 1000
  asm volatile(
      "{\n"
      "  .reg .pred %%pc;\n"
      "  .shared .align 16 .b8 _runtime_clc_res[16];\n"
      "  .shared .align 8 .b64 _runtime_clc_bar;\n"
      "  .reg .b128 %%rq;\n"
      "  .reg .b64 %%tmp;\n"
      "  .reg .b32 %%resa, %%bara, %%cx;\n"
      "  mov.u32 %%resa, _runtime_clc_res;\n"
      "  mov.u32 %%bara, _runtime_clc_bar;\n"
      "  mbarrier.init.shared::cta.b64 [%%bara], 1;\n"
      "  fence.proxy.async.shared::cta;\n"
      "  clusterlaunchcontrol.try_cancel.async.shared::cta."
      "mbarrier::complete_tx::bytes.b128 [%%resa], [%%bara];\n"
      "  mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 "
      "%%tmp, [%%bara], 16;\n"
      "L_runtime_clc_wait:\n"
      "  mbarrier.try_wait.parity.shared::cta.b64 %%pc, [%%bara], 0;\n"
      "  @!%%pc bra L_runtime_clc_wait;\n"
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

__global__ void clc_runtime_probe(uint32_t tasks, uint32_t work_cycles,
                                  uint32_t *visits, uint32_t *claimed_hist,
                                  RuntimeStats *stats) {
  __shared__ uint32_t next_raw;
  uint32_t worker = blockIdx.x;
  uint32_t raw = worker;

  if (threadIdx.x == 0) {
    uint32_t smid = smid_dev();
    stats[worker].processed = 0;
    stats[worker].attempts = 0;
    stats[worker].successes = 0;
    stats[worker].failures = 0;
    stats[worker].initial_smid = smid;
    stats[worker].last_smid = smid;
    stats[worker].smid_changes = 0;
    stats[worker].first_claim = 0xffffffffu;
    stats[worker].last_claim = 0xffffffffu;
    stats[worker].success_cycles_sum = 0;
    stats[worker].failure_cycles_sum = 0;
    stats[worker].success_cycles_min = 0xffffffffu;
    stats[worker].success_cycles_max = 0;
    stats[worker].failure_cycles_min = 0xffffffffu;
    stats[worker].failure_cycles_max = 0;
  }
  __syncthreads();

  while (raw < tasks) {
    burn(work_cycles);
    if (threadIdx.x == 0) {
      atomicAdd(&visits[raw], 1);
      stats[worker].processed++;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      uint64_t t0 = clock64_dev();
      uint32_t claimed = clc_try_cancel_1d(tasks);
      uint32_t dt = (uint32_t)(clock64_dev() - t0);
      uint32_t now_smid = smid_dev();

      if (now_smid != stats[worker].last_smid)
        stats[worker].smid_changes++;
      stats[worker].last_smid = now_smid;

      stats[worker].attempts++;
      next_raw = claimed;
      if (claimed < tasks) {
        stats[worker].successes++;
        stats[worker].success_cycles_sum += dt;
        stats[worker].success_cycles_min =
            min(stats[worker].success_cycles_min, dt);
        stats[worker].success_cycles_max =
            max(stats[worker].success_cycles_max, dt);
        if (stats[worker].first_claim == 0xffffffffu)
          stats[worker].first_claim = claimed;
        stats[worker].last_claim = claimed;
        atomicAdd(&claimed_hist[claimed], 1);
      } else {
        stats[worker].failures++;
        stats[worker].failure_cycles_sum += dt;
        stats[worker].failure_cycles_min =
            min(stats[worker].failure_cycles_min, dt);
        stats[worker].failure_cycles_max =
            max(stats[worker].failure_cycles_max, dt);
      }
    }
    __syncthreads();
    raw = next_raw;
  }
}

static double mean_u32(const std::vector<uint32_t> &v) {
  if (v.empty())
    return 0.0;
  uint64_t sum = std::accumulate(v.begin(), v.end(), uint64_t{0});
  return (double)sum / (double)v.size();
}

static uint32_t pct(std::vector<uint32_t> v, double p) {
  if (v.empty())
    return 0;
  std::sort(v.begin(), v.end());
  size_t idx = (size_t)((double)(v.size() - 1) * p + 0.5);
  return v[idx];
}

int main(int argc, char **argv) {
  uint32_t tasks = argc > 1 ? std::atoi(argv[1]) : 8192;
  uint32_t threads = argc > 2 ? std::atoi(argv[2]) : 128;
  uint32_t work_cycles = argc > 3 ? std::atoi(argv[3]) : 4096;
  uint32_t smem_bytes = argc > 4 ? std::atoi(argv[4]) : 0;
  if (!tasks || !threads) {
    std::fprintf(stderr,
                 "usage: %s [tasks] [threads] [work_cycles] [smem_bytes]\n",
                 argv[0]);
    return 2;
  }

  int dev = 0;
  cudaDeviceProp prop{};
  CHECK(cudaGetDevice(&dev));
  CHECK(cudaGetDeviceProperties(&prop, dev));
  int sm_count = 0;
  CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));

  uint32_t *d_visits = nullptr, *d_claimed = nullptr;
  RuntimeStats *d_stats = nullptr;
  CHECK(cudaMalloc(&d_visits, tasks * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_claimed, tasks * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_stats, tasks * sizeof(RuntimeStats)));
  CHECK(cudaMemset(d_visits, 0, tasks * sizeof(uint32_t)));
  CHECK(cudaMemset(d_claimed, 0, tasks * sizeof(uint32_t)));
  CHECK(cudaMemset(d_stats, 0, tasks * sizeof(RuntimeStats)));

  clc_runtime_probe<<<tasks, threads, smem_bytes>>>(
      tasks, work_cycles, d_visits, d_claimed, d_stats);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  std::vector<uint32_t> visits(tasks), claimed(tasks);
  std::vector<RuntimeStats> stats(tasks);
  CHECK(cudaMemcpy(visits.data(), d_visits, tasks * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(claimed.data(), d_claimed, tasks * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(stats.data(), d_stats, tasks * sizeof(RuntimeStats),
                   cudaMemcpyDeviceToHost));

  uint64_t processed = 0, attempts = 0, successes = 0, failures = 0;
  uint64_t success_cycles_sum = 0, failure_cycles_sum = 0;
  uint32_t active_workers = 0, smid_changes = 0;
  uint32_t success_cycles_min = 0xffffffffu, success_cycles_max = 0;
  uint32_t failure_cycles_min = 0xffffffffu, failure_cycles_max = 0;
  uint32_t first_claim_min = 0xffffffffu, first_claim_max = 0;
  uint32_t last_claim_min = 0xffffffffu, last_claim_max = 0;
  uint32_t smid_min = 0xffffffffu, smid_max = 0;
  std::map<uint32_t, uint32_t> sm_hist;
  std::vector<uint32_t> success_cycle_worker_avg;
  std::vector<uint32_t> failure_cycle_worker_avg;

  for (const RuntimeStats &s : stats) {
    processed += s.processed;
    attempts += s.attempts;
    successes += s.successes;
    failures += s.failures;
    smid_changes += s.smid_changes;
    if (!s.processed)
      continue;

    active_workers++;
    sm_hist[s.initial_smid]++;
    smid_min = std::min(smid_min, s.initial_smid);
    smid_max = std::max(smid_max, s.initial_smid);

    if (s.successes) {
      first_claim_min = std::min(first_claim_min, s.first_claim);
      first_claim_max = std::max(first_claim_max, s.first_claim);
      last_claim_min = std::min(last_claim_min, s.last_claim);
      last_claim_max = std::max(last_claim_max, s.last_claim);
      success_cycles_sum += s.success_cycles_sum;
      success_cycles_min = std::min(success_cycles_min, s.success_cycles_min);
      success_cycles_max = std::max(success_cycles_max, s.success_cycles_max);
      success_cycle_worker_avg.push_back(
          (uint32_t)(s.success_cycles_sum / s.successes));
    }
    if (s.failures) {
      failure_cycles_sum += s.failure_cycles_sum;
      failure_cycles_min = std::min(failure_cycles_min, s.failure_cycles_min);
      failure_cycles_max = std::max(failure_cycles_max, s.failure_cycles_max);
      failure_cycle_worker_avg.push_back(
          (uint32_t)(s.failure_cycles_sum / s.failures));
    }
  }

  std::vector<uint32_t> active_per_sm;
  for (const auto &kv : sm_hist)
    active_per_sm.push_back(kv.second);
  uint32_t active_sms = (uint32_t)active_per_sm.size();
  uint32_t active_per_sm_min =
      active_per_sm.empty() ? 0
                            : *std::min_element(active_per_sm.begin(),
                                                active_per_sm.end());
  uint32_t active_per_sm_max =
      active_per_sm.empty() ? 0
                            : *std::max_element(active_per_sm.begin(),
                                                active_per_sm.end());

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
      &occ_blocks, clc_runtime_probe, threads, smem_bytes));
  uint32_t predicted_r = (uint32_t)occ_blocks * (uint32_t)sm_count;
  uint32_t expected_active = std::min(tasks, predicted_r);
  uint32_t expected_claimed = tasks > predicted_r ? tasks - predicted_r : 0;

  bool structural_ok = missed == 0 && duplicates == 0 &&
                       duplicate_claims == 0 && claim_range_holes == 0 &&
                       active_workers == expected_active &&
                       attempts == processed && failures == active_workers &&
                       successes == expected_claimed;
  if (tasks <= predicted_r) {
    structural_ok = structural_ok && claimed_unique == 0;
  } else {
    structural_ok = structural_ok && claimed_unique == expected_claimed &&
                    claimed_min == predicted_r && claimed_max == tasks - 1;
  }

  double success_cycle_mean =
      successes ? (double)success_cycles_sum / (double)successes : 0.0;
  double failure_cycle_mean =
      failures ? (double)failure_cycles_sum / (double)failures : 0.0;
  bool terminal_failures_equal_active = failures == active_workers;

  if (std::getenv("CLC_PROBE_CSV")) {
    std::printf("tasks,threads,work_cycles,smem_bytes,processed,"
                "active_workers,missed,duplicates,attempts,successes,"
                "failures,unique_claimed,claimed_min,claimed_max,"
                "first_claim_min,first_claim_max,last_claim_min,"
                "last_claim_max,occ_blocks_per_sm,predicted_r,sm_count,"
                "expected_active_workers,expected_claimed,smid_min,smid_max,"
                "active_sms,active_workers_per_sm_min,"
                "active_workers_per_sm_max,active_workers_per_sm_mean,"
                "smid_changes,success_cycle_min,success_cycle_p50_worker_avg,"
                "success_cycle_p90_worker_avg,success_cycle_max,"
                "success_cycle_mean,failure_cycle_min,"
                "failure_cycle_p50_worker_avg,failure_cycle_p90_worker_avg,"
                "failure_cycle_max,failure_cycle_mean,"
                "terminal_failures_equal_active_workers,duplicate_claims,"
                "claim_range_holes,structural_ok\n");
    std::printf("%u,%u,%u,%u,%llu,%u,%u,%u,%llu,%llu,%llu,%u,%u,%u,%u,%u,%u,"
                "%u,%d,%u,%d,%u,%u,%u,%u,%u,%u,%u,%.3f,%u,%u,%u,%u,%u,%.3f,"
                "%u,%u,%u,%u,%.3f,%u,%u,%u,%u\n",
                tasks, threads, work_cycles, smem_bytes,
                (unsigned long long)processed, active_workers, missed,
                duplicates, (unsigned long long)attempts,
                (unsigned long long)successes, (unsigned long long)failures,
                claimed_unique, claimed_unique ? claimed_min : 0,
                claimed_unique ? claimed_max : 0,
                first_claim_min == 0xffffffffu ? 0 : first_claim_min,
                first_claim_max,
                last_claim_min == 0xffffffffu ? 0 : last_claim_min,
                last_claim_max, occ_blocks, predicted_r, sm_count,
                expected_active, expected_claimed,
                smid_min == 0xffffffffu ? 0 : smid_min, smid_max, active_sms,
                active_per_sm_min, active_per_sm_max, mean_u32(active_per_sm),
                smid_changes,
                success_cycles_min == 0xffffffffu ? 0 : success_cycles_min,
                pct(success_cycle_worker_avg, 0.50),
                pct(success_cycle_worker_avg, 0.90), success_cycles_max,
                success_cycle_mean,
                failure_cycles_min == 0xffffffffu ? 0 : failure_cycles_min,
                pct(failure_cycle_worker_avg, 0.50),
                pct(failure_cycle_worker_avg, 0.90), failure_cycles_max,
                failure_cycle_mean, terminal_failures_equal_active ? 1 : 0,
                duplicate_claims, claim_range_holes, structural_ok ? 1 : 0);
  } else {
    std::printf("== CLC runtime probe on %s sm_%d%d ==\n", prop.name,
                prop.major, prop.minor);
    std::printf("   tasks=%u threads=%u work_cycles=%u smem=%u R=%u\n", tasks,
                threads, work_cycles, smem_bytes, predicted_r);
    std::printf("   processed=%llu active=%u attempts=%llu successes=%llu "
                "failures=%llu terminal_failures=%s\n",
                (unsigned long long)processed, active_workers,
                (unsigned long long)attempts, (unsigned long long)successes,
                (unsigned long long)failures,
                terminal_failures_equal_active ? "yes" : "no");
    std::printf("   claimed=%u..%u missed=%u duplicates=%u holes=%u "
                "structural_ok=%s\n",
                claimed_unique ? claimed_min : 0,
                claimed_unique ? claimed_max : 0, missed, duplicates,
                claim_range_holes, structural_ok ? "yes" : "no");
    std::printf("   active_sms=%u smid=%u..%u workers/SM=%u..%u mean=%.3f "
                "smid_changes=%u\n",
                active_sms, smid_min == 0xffffffffu ? 0 : smid_min, smid_max,
                active_per_sm_min, active_per_sm_max, mean_u32(active_per_sm),
                smid_changes);
    std::printf("   success cycles mean=%.1f min=%u max=%u; failure cycles "
                "mean=%.1f min=%u max=%u\n",
                success_cycle_mean,
                success_cycles_min == 0xffffffffu ? 0 : success_cycles_min,
                success_cycles_max, failure_cycle_mean,
                failure_cycles_min == 0xffffffffu ? 0 : failure_cycles_min,
                failure_cycles_max);
  }

  CHECK(cudaFree(d_visits));
  CHECK(cudaFree(d_claimed));
  CHECK(cudaFree(d_stats));
  return structural_ok ? 0 : 1;
}
