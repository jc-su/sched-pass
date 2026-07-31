// Standalone Blackwell CLC pressure probe.
//
// Runs a CLC kernel while an independent pressure kernel is active in another
// stream. This tests whether observable CLC success/failure behavior changes
// under concurrent scheduler pressure.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <numeric>
#include <vector>

struct RuntimeStatsPressure {
  uint32_t processed;
  uint32_t attempts;
  uint32_t successes;
  uint32_t failures;
  uint32_t initial_smid;
  uint32_t smid_changes;
  uint32_t first_claim;
  uint32_t last_claim;
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

__device__ __forceinline__ uint64_t globaltimer_dev() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
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
      "  .shared .align 16 .b8 _pressure_clc_res[16];\n"
      "  .shared .align 8 .b64 _pressure_clc_bar;\n"
      "  .reg .b128 %%rq;\n"
      "  .reg .b64 %%tmp;\n"
      "  .reg .b32 %%resa, %%bara, %%cx;\n"
      "  mov.u32 %%resa, _pressure_clc_res;\n"
      "  mov.u32 %%bara, _pressure_clc_bar;\n"
      "  mbarrier.init.shared::cta.b64 [%%bara], 1;\n"
      "  fence.proxy.async.shared::cta;\n"
      "  clusterlaunchcontrol.try_cancel.async.shared::cta."
      "mbarrier::complete_tx::bytes.b128 [%%resa], [%%bara];\n"
      "  mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 "
      "%%tmp, [%%bara], 16;\n"
      "L_pressure_clc_wait:\n"
      "  mbarrier.try_wait.parity.shared::cta.b64 %%pc, [%%bara], 0;\n"
      "  @!%%pc bra L_pressure_clc_wait;\n"
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

__global__ void pressure_kernel(uint32_t cycles, uint32_t *smids,
                                uint32_t *started, uint64_t *times) {
  if (threadIdx.x == 0) {
    atomicMin((unsigned long long *)&times[0],
              (unsigned long long)globaltimer_dev());
    smids[blockIdx.x] = smid_dev();
    atomicAdd(started, 1);
  }
  burn(cycles);
  if (threadIdx.x == 0) {
    atomicMax((unsigned long long *)&times[1],
              (unsigned long long)globaltimer_dev());
  }
}

__global__ void clc_pressure_runtime(uint32_t tasks, uint32_t work_cycles,
                                     uint32_t *visits,
                                     uint32_t *claimed_hist,
                                     RuntimeStatsPressure *stats,
                                     uint64_t *times) {
  __shared__ uint32_t next_raw;
  uint32_t worker = blockIdx.x;
  uint32_t raw = worker;

  if (threadIdx.x == 0) {
    atomicMin((unsigned long long *)&times[0],
              (unsigned long long)globaltimer_dev());
    uint32_t smid = smid_dev();
    stats[worker].processed = 0;
    stats[worker].attempts = 0;
    stats[worker].successes = 0;
    stats[worker].failures = 0;
    stats[worker].initial_smid = smid;
    stats[worker].smid_changes = 0;
    stats[worker].first_claim = 0xffffffffu;
    stats[worker].last_claim = 0xffffffffu;
  }
  __syncthreads();

  while (raw < tasks) {
    burn(work_cycles);
    if (threadIdx.x == 0) {
      atomicAdd(&visits[raw], 1);
      if (smid_dev() != stats[worker].initial_smid)
        stats[worker].smid_changes++;
      stats[worker].processed++;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      uint32_t claimed = clc_try_cancel_1d(tasks);
      next_raw = claimed;
      stats[worker].attempts++;
      if (claimed < tasks) {
        stats[worker].successes++;
        if (stats[worker].first_claim == 0xffffffffu)
          stats[worker].first_claim = claimed;
        stats[worker].last_claim = claimed;
        atomicAdd(&claimed_hist[claimed], 1);
      } else {
        stats[worker].failures++;
      }
    }
    __syncthreads();
    raw = next_raw;
  }
  if (threadIdx.x == 0) {
    atomicMax((unsigned long long *)&times[1],
              (unsigned long long)globaltimer_dev());
  }
}

static double mean_u32(const std::vector<uint32_t> &v) {
  if (v.empty())
    return 0.0;
  uint64_t sum = std::accumulate(v.begin(), v.end(), uint64_t{0});
  return (double)sum / (double)v.size();
}

int main(int argc, char **argv) {
  uint32_t tasks = argc > 1 ? std::atoi(argv[1]) : 8192;
  uint32_t threads = argc > 2 ? std::atoi(argv[2]) : 128;
  uint32_t work_cycles = argc > 3 ? std::atoi(argv[3]) : 4096;
  uint32_t pressure_blocks_per_sm = argc > 4 ? std::atoi(argv[4]) : 1;
  uint32_t pressure_threads = argc > 5 ? std::atoi(argv[5]) : 128;
  uint32_t pressure_cycles = argc > 6 ? std::atoi(argv[6]) : 20000000;
  int priority_mode = argc > 7 ? std::atoi(argv[7]) : 0;
  uint32_t pressure_dynamic_smem = argc > 8 ? std::atoi(argv[8]) : 0;
  uint32_t pressure_blocks_override = argc > 9 ? std::atoi(argv[9]) : 0;
  int launch_order = argc > 10 ? std::atoi(argv[10]) : 0;
  // priority_mode: -1 pressure higher, 0 same/default, 1 CLC higher.
  // launch_order: 0 pressure first, 1 CLC first.
  if (!tasks || !threads || !pressure_threads) {
    std::fprintf(stderr,
                 "usage: %s [tasks] [threads] [work_cycles] "
                 "[pressure_blocks_per_sm] [pressure_threads] "
                 "[pressure_cycles] [priority_mode] [pressure_dynamic_smem] "
                 "[pressure_blocks_override] [launch_order]\n",
                 argv[0]);
    return 2;
  }

  int dev = 0;
  cudaDeviceProp prop{};
  CHECK(cudaGetDevice(&dev));
  CHECK(cudaGetDeviceProperties(&prop, dev));
  int sm_count = 0;
  CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));

  int low_pri = 0, high_pri = 0;
  CHECK(cudaDeviceGetStreamPriorityRange(&low_pri, &high_pri));
  int clc_pri = 0;
  int pressure_pri = 0;
  if (priority_mode < 0) {
    pressure_pri = high_pri;
    clc_pri = low_pri;
  } else if (priority_mode > 0) {
    pressure_pri = low_pri;
    clc_pri = high_pri;
  }

  cudaStream_t pressure_stream = nullptr, clc_stream = nullptr;
  CHECK(cudaStreamCreateWithPriority(&pressure_stream, cudaStreamNonBlocking,
                                     pressure_pri));
  CHECK(cudaStreamCreateWithPriority(&clc_stream, cudaStreamNonBlocking,
                                     clc_pri));
  if (pressure_dynamic_smem) {
    CHECK(cudaFuncSetAttribute(pressure_kernel,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               pressure_dynamic_smem));
  }

  cudaEvent_t pressure_start = nullptr, pressure_end = nullptr;
  cudaEvent_t clc_start = nullptr, clc_end = nullptr;
  CHECK(cudaEventCreate(&pressure_start));
  CHECK(cudaEventCreate(&pressure_end));
  CHECK(cudaEventCreate(&clc_start));
  CHECK(cudaEventCreate(&clc_end));

  uint32_t pressure_blocks =
      pressure_blocks_override ? pressure_blocks_override
                               : pressure_blocks_per_sm * (uint32_t)sm_count;
  uint32_t *d_pressure_smids = nullptr, *d_pressure_started = nullptr;
  uint32_t *d_visits = nullptr, *d_claimed = nullptr;
  RuntimeStatsPressure *d_stats = nullptr;
  uint64_t *d_pressure_times = nullptr, *d_clc_times = nullptr;
  CHECK(cudaMalloc(&d_pressure_smids, pressure_blocks * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_pressure_started, sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_visits, tasks * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_claimed, tasks * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_stats, tasks * sizeof(RuntimeStatsPressure)));
  CHECK(cudaMalloc(&d_pressure_times, 2 * sizeof(uint64_t)));
  CHECK(cudaMalloc(&d_clc_times, 2 * sizeof(uint64_t)));
  uint64_t time_init[2] = {UINT64_MAX, 0};
  CHECK(cudaMemcpy(d_pressure_times, time_init, sizeof(time_init),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_clc_times, time_init, sizeof(time_init),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemset(d_pressure_smids, 0xff, pressure_blocks * sizeof(uint32_t)));
  CHECK(cudaMemset(d_pressure_started, 0, sizeof(uint32_t)));
  CHECK(cudaMemset(d_visits, 0, tasks * sizeof(uint32_t)));
  CHECK(cudaMemset(d_claimed, 0, tasks * sizeof(uint32_t)));
  CHECK(cudaMemset(d_stats, 0, tasks * sizeof(RuntimeStatsPressure)));

  auto launch_pressure = [&]() {
    if (pressure_blocks) {
      CHECK(cudaEventRecord(pressure_start, pressure_stream));
      pressure_kernel<<<pressure_blocks, pressure_threads, pressure_dynamic_smem,
                        pressure_stream>>>(pressure_cycles, d_pressure_smids,
                                           d_pressure_started,
                                           d_pressure_times);
      CHECK(cudaGetLastError());
      CHECK(cudaEventRecord(pressure_end, pressure_stream));
    }
  };
  auto launch_clc = [&]() {
    CHECK(cudaEventRecord(clc_start, clc_stream));
    clc_pressure_runtime<<<tasks, threads, 0, clc_stream>>>(
        tasks, work_cycles, d_visits, d_claimed, d_stats, d_clc_times);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(clc_end, clc_stream));
  };

  if (launch_order == 1) {
    launch_clc();
    launch_pressure();
  } else {
    launch_pressure();
    launch_clc();
  }
  CHECK(cudaDeviceSynchronize());

  float pressure_ms = 0.0f, clc_ms = 0.0f;
  float pstart_to_cstart_ms = 0.0f, pstart_to_cend_ms = 0.0f;
  float overlap_ms = 0.0f;
  uint32_t clc_started_after_pressure_end = 0;
  if (pressure_blocks) {
    CHECK(cudaEventElapsedTime(&pressure_ms, pressure_start, pressure_end));
    if (launch_order == 0) {
      CHECK(cudaEventElapsedTime(&pstart_to_cstart_ms, pressure_start,
                                 clc_start));
      CHECK(cudaEventElapsedTime(&pstart_to_cend_ms, pressure_start, clc_end));
      float overlap_begin = std::max(0.0f, pstart_to_cstart_ms);
      float overlap_end = std::min(pressure_ms, pstart_to_cend_ms);
      overlap_ms = std::max(0.0f, overlap_end - overlap_begin);
      clc_started_after_pressure_end = pstart_to_cstart_ms >= pressure_ms;
    }
  }
  CHECK(cudaEventElapsedTime(&clc_ms, clc_start, clc_end));

  std::vector<uint32_t> pressure_smids(pressure_blocks);
  std::vector<uint32_t> visits(tasks), claimed(tasks);
  std::vector<RuntimeStatsPressure> stats(tasks);
  uint32_t pressure_started = 0;
  uint64_t pressure_times[2] = {UINT64_MAX, 0};
  uint64_t clc_times[2] = {UINT64_MAX, 0};
  if (pressure_blocks)
    CHECK(cudaMemcpy(pressure_smids.data(), d_pressure_smids,
                     pressure_blocks * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(&pressure_started, d_pressure_started, sizeof(uint32_t),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(visits.data(), d_visits, tasks * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(claimed.data(), d_claimed, tasks * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(stats.data(), d_stats,
                   tasks * sizeof(RuntimeStatsPressure),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(pressure_times, d_pressure_times, sizeof(pressure_times),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(clc_times, d_clc_times, sizeof(clc_times),
                   cudaMemcpyDeviceToHost));

  double pressure_global_us = 0.0, clc_global_us = 0.0;
  double global_start_delta_us = 0.0, global_end_delta_us = 0.0;
  double global_overlap_us = 0.0;
  uint32_t clc_started_after_pressure_end_global = 0;
  if (pressure_blocks && pressure_times[0] != UINT64_MAX &&
      pressure_times[1] >= pressure_times[0] && clc_times[0] != UINT64_MAX &&
      clc_times[1] >= clc_times[0]) {
    pressure_global_us =
        (double)(pressure_times[1] - pressure_times[0]) / 1000.0;
    clc_global_us = (double)(clc_times[1] - clc_times[0]) / 1000.0;
    auto diff_us = [](uint64_t a, uint64_t b) {
      if (a >= b)
        return (double)(a - b) / 1000.0;
      return -(double)(b - a) / 1000.0;
    };
    global_start_delta_us = diff_us(clc_times[0], pressure_times[0]);
    global_end_delta_us = diff_us(clc_times[1], pressure_times[0]);
    uint64_t overlap_start = std::max(pressure_times[0], clc_times[0]);
    uint64_t overlap_end = std::min(pressure_times[1], clc_times[1]);
    if (overlap_end > overlap_start)
      global_overlap_us = (double)(overlap_end - overlap_start) / 1000.0;
    clc_started_after_pressure_end_global = clc_times[0] >= pressure_times[1];
  } else if (clc_times[0] != UINT64_MAX && clc_times[1] >= clc_times[0]) {
    clc_global_us = (double)(clc_times[1] - clc_times[0]) / 1000.0;
  }

  std::vector<uint32_t> pressure_per_sm(sm_count, 0);
  for (uint32_t smid : pressure_smids) {
    if (smid < pressure_per_sm.size())
      pressure_per_sm[smid]++;
  }
  uint32_t pressure_active_sms = 0;
  for (uint32_t v : pressure_per_sm)
    pressure_active_sms += v != 0;

  uint64_t processed = 0, attempts = 0, successes = 0, failures = 0;
  uint32_t active_workers = 0, smid_changes = 0;
  std::map<uint32_t, uint32_t> clc_sm_hist;
  uint32_t first_claim_min = 0xffffffffu, first_claim_max = 0;
  uint32_t last_claim_min = 0xffffffffu, last_claim_max = 0;
  for (const RuntimeStatsPressure &s : stats) {
    processed += s.processed;
    attempts += s.attempts;
    successes += s.successes;
    failures += s.failures;
    smid_changes += s.smid_changes;
    if (s.processed) {
      active_workers++;
      clc_sm_hist[s.initial_smid]++;
    }
    if (s.successes) {
      first_claim_min = std::min(first_claim_min, s.first_claim);
      first_claim_max = std::max(first_claim_max, s.first_claim);
      last_claim_min = std::min(last_claim_min, s.last_claim);
      last_claim_max = std::max(last_claim_max, s.last_claim);
    }
  }

  std::vector<uint32_t> clc_per_sm;
  for (const auto &kv : clc_sm_hist)
    clc_per_sm.push_back(kv.second);
  uint32_t clc_active_sms = (uint32_t)clc_per_sm.size();
  uint32_t clc_per_sm_min =
      clc_per_sm.empty() ? 0
                         : *std::min_element(clc_per_sm.begin(),
                                             clc_per_sm.end());
  uint32_t clc_per_sm_max =
      clc_per_sm.empty() ? 0
                         : *std::max_element(clc_per_sm.begin(),
                                             clc_per_sm.end());

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
    claim_range_holes = claimed_max - claimed_min + 1 - claimed_unique;

  int occ_blocks = 0;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occ_blocks, clc_pressure_runtime, threads, 0));
  uint32_t predicted_r = (uint32_t)occ_blocks * (uint32_t)sm_count;
  bool exactly_once = missed == 0 && duplicates == 0 &&
                      duplicate_claims == 0 && claim_range_holes == 0 &&
                      processed == tasks && attempts == tasks;
  bool suffix_matches_active = true;
  if (tasks <= active_workers) {
    suffix_matches_active = successes == 0 && claimed_unique == 0;
  } else {
    suffix_matches_active =
        successes == tasks - active_workers &&
        claimed_unique == tasks - active_workers &&
        claimed_min == active_workers && claimed_max == tasks - 1;
  }
  bool terminal_failures_equal_active = failures == active_workers;
  bool structural_ok = exactly_once && suffix_matches_active &&
                       terminal_failures_equal_active && smid_changes == 0;

  if (std::getenv("CLC_PROBE_CSV")) {
    std::printf("tasks,threads,work_cycles,pressure_blocks_per_sm,"
                "pressure_blocks_override,pressure_blocks,pressure_threads,"
                "pressure_cycles,"
                "pressure_dynamic_smem,priority_mode,launch_order,low_priority,"
                "high_priority,clc_priority,"
                "pressure_priority,pressure_started,pressure_active_sms,"
                "pressure_per_sm_min,pressure_per_sm_max,pressure_per_sm_mean,"
                "pressure_ms,clc_ms,pstart_to_cstart_ms,pstart_to_cend_ms,"
                "overlap_ms,clc_started_after_pressure_end,"
                "pressure_global_us,clc_global_us,global_start_delta_us,"
                "global_end_delta_us,global_overlap_us,"
                "clc_started_after_pressure_end_global,"
                "processed,active_workers,clc_active_sms,clc_per_sm_min,"
                "clc_per_sm_max,clc_per_sm_mean,missed,duplicates,attempts,"
                "successes,failures,unique_claimed,claimed_min,claimed_max,"
                "first_claim_min,first_claim_max,last_claim_min,last_claim_max,"
                "occ_blocks_per_sm,predicted_r,sm_count,smid_changes,"
                "duplicate_claims,claim_range_holes,exactly_once,"
                "suffix_matches_active,terminal_failures_equal_active_workers,"
                "structural_ok\n");
    std::printf("%u,%u,%u,%u,%u,%u,%u,%u,%u,%d,%d,%d,%d,%d,%d,%u,%u,%u,%u,%.3f,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,%u,%.3f,%.3f,%.3f,%.3f,%.3f,%u,%llu,"
                "%u,%u,%u,%u,%.3f,%u,%u,%llu,%llu,%llu,%u,%u,%u,%u,%u,%u,%u,"
                "%d,%u,%d,%u,%u,%u,%u,%u,%u,%u\n",
                tasks, threads, work_cycles, pressure_blocks_per_sm,
                pressure_blocks_override, pressure_blocks, pressure_threads,
                pressure_cycles, pressure_dynamic_smem, priority_mode,
                launch_order, low_pri, high_pri, clc_pri, pressure_pri,
                pressure_started, pressure_active_sms,
                pressure_per_sm.empty()
                    ? 0
                    : *std::min_element(pressure_per_sm.begin(),
                                        pressure_per_sm.end()),
                pressure_per_sm.empty()
                    ? 0
                    : *std::max_element(pressure_per_sm.begin(),
                                        pressure_per_sm.end()),
                mean_u32(pressure_per_sm), pressure_ms, clc_ms,
                pstart_to_cstart_ms, pstart_to_cend_ms, overlap_ms,
                clc_started_after_pressure_end, pressure_global_us,
                clc_global_us, global_start_delta_us, global_end_delta_us,
                global_overlap_us, clc_started_after_pressure_end_global,
                (unsigned long long)processed,
                active_workers, clc_active_sms, clc_per_sm_min, clc_per_sm_max,
                mean_u32(clc_per_sm), missed, duplicates,
                (unsigned long long)attempts, (unsigned long long)successes,
                (unsigned long long)failures, claimed_unique,
                claimed_unique ? claimed_min : 0,
                claimed_unique ? claimed_max : 0,
                first_claim_min == 0xffffffffu ? 0 : first_claim_min,
                first_claim_max,
                last_claim_min == 0xffffffffu ? 0 : last_claim_min,
                last_claim_max, occ_blocks, predicted_r, sm_count,
                smid_changes, duplicate_claims, claim_range_holes,
                exactly_once ? 1 : 0, suffix_matches_active ? 1 : 0,
                terminal_failures_equal_active ? 1 : 0,
                structural_ok ? 1 : 0);
  } else {
    std::printf("== CLC pressure probe on %s sm_%d%d ==\n", prop.name,
                prop.major, prop.minor);
    std::printf("   pressure blocks/SM=%u override=%u launched=%u "
                "active_sms=%u perSM=%u..%u threads=%u smem=%u "
                "priority_mode=%d launch_order=%d\n",
                pressure_blocks_per_sm, pressure_blocks_override,
                pressure_blocks, pressure_active_sms,
                pressure_per_sm.empty()
                    ? 0
                    : *std::min_element(pressure_per_sm.begin(),
                                        pressure_per_sm.end()),
                pressure_per_sm.empty()
                    ? 0
                    : *std::max_element(pressure_per_sm.begin(),
                                        pressure_per_sm.end()),
                pressure_threads, pressure_dynamic_smem, priority_mode,
                launch_order);
    std::printf("   timing pressure=%.3fms clc=%.3fms "
                "clc_start_after_pstart=%.3fms overlap=%.3fms "
                "clc_after_pressure=%s\n",
                pressure_ms, clc_ms, pstart_to_cstart_ms, overlap_ms,
                clc_started_after_pressure_end ? "yes" : "no");
    std::printf("   device timing pressure=%.3fus clc=%.3fus "
                "clc_start_after_pstart=%.3fus actual_overlap=%.3fus "
                "clc_after_pressure=%s\n",
                pressure_global_us, clc_global_us, global_start_delta_us,
                global_overlap_us,
                clc_started_after_pressure_end_global ? "yes" : "no");
    std::printf("   standalone_R=%u active_workers=%u clc_sms=%u "
                "clc_perSM=%u..%u\n",
                predicted_r, active_workers, clc_active_sms, clc_per_sm_min,
                clc_per_sm_max);
    std::printf("   attempts=%llu successes=%llu failures=%llu claimed=%u..%u "
                "exactly_once=%s suffix_matches_active=%s structural_ok=%s\n",
                (unsigned long long)attempts, (unsigned long long)successes,
                (unsigned long long)failures,
                claimed_unique ? claimed_min : 0,
                claimed_unique ? claimed_max : 0,
                exactly_once ? "yes" : "no",
                suffix_matches_active ? "yes" : "no",
                structural_ok ? "yes" : "no");
  }

  CHECK(cudaStreamDestroy(pressure_stream));
  CHECK(cudaStreamDestroy(clc_stream));
  CHECK(cudaEventDestroy(pressure_start));
  CHECK(cudaEventDestroy(pressure_end));
  CHECK(cudaEventDestroy(clc_start));
  CHECK(cudaEventDestroy(clc_end));
  CHECK(cudaFree(d_pressure_smids));
  CHECK(cudaFree(d_pressure_started));
  CHECK(cudaFree(d_visits));
  CHECK(cudaFree(d_claimed));
  CHECK(cudaFree(d_stats));
  CHECK(cudaFree(d_pressure_times));
  CHECK(cudaFree(d_clc_times));
  return structural_ok ? 0 : 1;
}
