// Standalone Blackwell CLC mapping probe.
//
// Records the runtime mapping:
// - initial raw CTA id -> worker -> SM id
// - claimed suffix raw id -> claiming worker -> worker SM id
//
// This answers the scheduling-policy question that the aggregate probes cannot:
// how task_order[raw] maps onto SM residency and claimed suffix execution.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

struct TaskRecord {
  uint32_t raw;
  uint32_t worker;
  uint32_t initial_smid;
  uint32_t exec_smid;
  uint32_t ordinal;
  uint32_t claimed;
};

struct MappingStats {
  uint32_t processed;
  uint32_t attempts;
  uint32_t successes;
  uint32_t failures;
  uint32_t initial_smid;
  uint32_t smid_changes;
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
      "  .shared .align 16 .b8 _mapping_clc_res[16];\n"
      "  .shared .align 8 .b64 _mapping_clc_bar;\n"
      "  .reg .b128 %%rq;\n"
      "  .reg .b64 %%tmp;\n"
      "  .reg .b32 %%resa, %%bara, %%cx;\n"
      "  mov.u32 %%resa, _mapping_clc_res;\n"
      "  mov.u32 %%bara, _mapping_clc_bar;\n"
      "  mbarrier.init.shared::cta.b64 [%%bara], 1;\n"
      "  fence.proxy.async.shared::cta;\n"
      "  clusterlaunchcontrol.try_cancel.async.shared::cta."
      "mbarrier::complete_tx::bytes.b128 [%%resa], [%%bara];\n"
      "  mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 "
      "%%tmp, [%%bara], 16;\n"
      "L_mapping_clc_wait:\n"
      "  mbarrier.try_wait.parity.shared::cta.b64 %%pc, [%%bara], 0;\n"
      "  @!%%pc bra L_mapping_clc_wait;\n"
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

__global__ void clc_mapping_probe(uint32_t tasks, uint32_t work_cycles,
                                  TaskRecord *records,
                                  MappingStats *stats) {
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
    stats[worker].smid_changes = 0;
  }
  __syncthreads();

  while (raw < tasks) {
    burn(work_cycles);
    if (threadIdx.x == 0) {
      uint32_t exec_smid = smid_dev();
      if (exec_smid != stats[worker].initial_smid)
        stats[worker].smid_changes++;
      TaskRecord rec;
      rec.raw = raw;
      rec.worker = worker;
      rec.initial_smid = stats[worker].initial_smid;
      rec.exec_smid = exec_smid;
      rec.ordinal = stats[worker].processed;
      rec.claimed = raw == worker ? 0 : 1;
      records[raw] = rec;
      stats[worker].processed++;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      uint32_t claimed = clc_try_cancel_1d(tasks);
      next_raw = claimed;
      stats[worker].attempts++;
      if (claimed < tasks)
        stats[worker].successes++;
      else
        stats[worker].failures++;
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

  TaskRecord *d_records = nullptr;
  MappingStats *d_stats = nullptr;
  CHECK(cudaMalloc(&d_records, tasks * sizeof(TaskRecord)));
  CHECK(cudaMalloc(&d_stats, tasks * sizeof(MappingStats)));
  CHECK(cudaMemset(d_records, 0xff, tasks * sizeof(TaskRecord)));
  CHECK(cudaMemset(d_stats, 0, tasks * sizeof(MappingStats)));

  clc_mapping_probe<<<tasks, threads, smem_bytes>>>(tasks, work_cycles,
                                                    d_records, d_stats);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  std::vector<TaskRecord> records(tasks);
  std::vector<MappingStats> stats(tasks);
  CHECK(cudaMemcpy(records.data(), d_records, tasks * sizeof(TaskRecord),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(stats.data(), d_stats, tasks * sizeof(MappingStats),
                   cudaMemcpyDeviceToHost));

  int occ_blocks = 0;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occ_blocks, clc_mapping_probe, threads, smem_bytes));
  uint32_t predicted_r = (uint32_t)occ_blocks * (uint32_t)sm_count;
  uint32_t expected_active = std::min(tasks, predicted_r);
  uint32_t expected_claimed = tasks > predicted_r ? tasks - predicted_r : 0;

  uint64_t processed = 0, attempts = 0, successes = 0, failures = 0;
  uint32_t active_workers = 0, smid_changes = 0;
  std::vector<uint32_t> per_sm(sm_count, 0);
  for (uint32_t i = 0; i < tasks; ++i) {
    const MappingStats &s = stats[i];
    processed += s.processed;
    attempts += s.attempts;
    successes += s.successes;
    failures += s.failures;
    smid_changes += s.smid_changes;
    if (s.processed) {
      active_workers++;
      if (s.initial_smid < per_sm.size())
        per_sm[s.initial_smid]++;
    }
  }

  uint32_t missed = 0, duplicates = 0, claimed_records = 0, bad_records = 0;
  uint32_t claimed_min = 0xffffffffu, claimed_max = 0;
  std::vector<uint32_t> visits(tasks, 0);
  for (const TaskRecord &r : records) {
    if (r.raw < tasks)
      visits[r.raw]++;
    else
      bad_records++;
    if (r.claimed) {
      claimed_records++;
      claimed_min = std::min(claimed_min, r.raw);
      claimed_max = std::max(claimed_max, r.raw);
    }
    if (r.exec_smid != r.initial_smid)
      bad_records++;
  }
  for (uint32_t v : visits) {
    missed += v == 0;
    duplicates += v > 1;
  }
  uint32_t claim_range_holes = 0;
  if (claimed_records)
    claim_range_holes = claimed_max - claimed_min + 1 - claimed_records;

  uint32_t per_sm_min = per_sm.empty()
                            ? 0
                            : *std::min_element(per_sm.begin(), per_sm.end());
  uint32_t per_sm_max = per_sm.empty()
                            ? 0
                            : *std::max_element(per_sm.begin(), per_sm.end());

  bool structural_ok = missed == 0 && duplicates == 0 && bad_records == 0 &&
                       active_workers == expected_active &&
                       attempts == processed && successes == expected_claimed &&
                       failures == expected_active && smid_changes == 0;
  if (tasks <= predicted_r) {
    structural_ok = structural_ok && claimed_records == 0;
  } else {
    structural_ok = structural_ok && claimed_records == expected_claimed &&
                    claimed_min == predicted_r && claimed_max == tasks - 1 &&
                    claim_range_holes == 0;
  }

  if (std::getenv("CLC_MAP_EVENTS_CSV")) {
    std::printf("raw,worker,initial_smid,exec_smid,ordinal,claimed\n");
    for (const TaskRecord &r : records) {
      std::printf("%u,%u,%u,%u,%u,%u\n", r.raw, r.worker, r.initial_smid,
                  r.exec_smid, r.ordinal, r.claimed);
    }
  } else if (std::getenv("CLC_PROBE_CSV")) {
    std::printf("tasks,threads,work_cycles,smem_bytes,processed,"
                "active_workers,attempts,successes,failures,claimed_records,"
                "claimed_min,claimed_max,missed,duplicates,bad_records,"
                "claim_range_holes,occ_blocks_per_sm,predicted_r,sm_count,"
                "expected_active_workers,expected_claimed,workers_per_sm_min,"
                "workers_per_sm_max,workers_per_sm_mean,smid_changes,"
                "structural_ok\n");
    std::printf("%u,%u,%u,%u,%llu,%u,%llu,%llu,%llu,%u,%u,%u,%u,%u,%u,%u,%d,"
                "%u,%d,%u,%u,%u,%u,%.3f,%u,%u\n",
                tasks, threads, work_cycles, smem_bytes,
                (unsigned long long)processed, active_workers,
                (unsigned long long)attempts, (unsigned long long)successes,
                (unsigned long long)failures, claimed_records,
                claimed_records ? claimed_min : 0,
                claimed_records ? claimed_max : 0, missed, duplicates,
                bad_records, claim_range_holes, occ_blocks, predicted_r,
                sm_count, expected_active, expected_claimed, per_sm_min,
                per_sm_max, mean_u32(per_sm), smid_changes,
                structural_ok ? 1 : 0);
  } else {
    std::printf("== CLC mapping probe on %s sm_%d%d ==\n", prop.name,
                prop.major, prop.minor);
    std::printf("   tasks=%u threads=%u work_cycles=%u smem=%u R=%u\n", tasks,
                threads, work_cycles, smem_bytes, predicted_r);
    std::printf("   active=%u attempts=%llu successes=%llu failures=%llu "
                "claimed=%u..%u structural_ok=%s\n",
                active_workers, (unsigned long long)attempts,
                (unsigned long long)successes, (unsigned long long)failures,
                claimed_records ? claimed_min : 0,
                claimed_records ? claimed_max : 0,
                structural_ok ? "yes" : "no");
    std::printf("   workers/SM=%u..%u mean=%.3f smid_changes=%u\n",
                per_sm_min, per_sm_max, mean_u32(per_sm), smid_changes);
  }

  CHECK(cudaFree(d_records));
  CHECK(cudaFree(d_stats));
  return structural_ok ? 0 : 1;
}
