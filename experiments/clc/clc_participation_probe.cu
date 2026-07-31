// Standalone Blackwell CLC participation probe.
//
// Tests what happens when only a subset of launched CTAs enter the CLC loop
// while the rest finish normally. This probes whether the suffix model requires
// full participation by every resident worker.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct WorkerStatsPart {
  uint32_t processed;
  uint32_t attempts;
  uint32_t successes;
  uint32_t participates;
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
      "  .shared .align 16 .b8 _part_clc_res[16];\n"
      "  .shared .align 8 .b64 _part_clc_bar;\n"
      "  .reg .b128 %%rq;\n"
      "  .reg .b64 %%tmp;\n"
      "  .reg .b32 %%resa, %%bara, %%cx;\n"
      "  mov.u32 %%resa, _part_clc_res;\n"
      "  mov.u32 %%bara, _part_clc_bar;\n"
      "  mbarrier.init.shared::cta.b64 [%%bara], 1;\n"
      "  fence.proxy.async.shared::cta;\n"
      "  clusterlaunchcontrol.try_cancel.async.shared::cta."
      "mbarrier::complete_tx::bytes.b128 [%%resa], [%%bara];\n"
      "  mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 "
      "%%tmp, [%%bara], 16;\n"
      "L_part_clc_wait:\n"
      "  mbarrier.try_wait.parity.shared::cta.b64 %%pc, [%%bara], 0;\n"
      "  @!%%pc bra L_part_clc_wait;\n"
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

__global__ void clc_participation_probe(uint32_t tasks, uint32_t work_cycles,
                                        uint32_t claim_stride,
                                        uint32_t *visits,
                                        uint32_t *claimed_hist,
                                        WorkerStatsPart *stats) {
  __shared__ uint32_t next_raw;
  uint32_t worker = blockIdx.x;
  uint32_t raw = worker;
  bool participate = (worker % claim_stride) == 0;

  if (threadIdx.x == 0) {
    stats[worker].processed = 0;
    stats[worker].attempts = 0;
    stats[worker].successes = 0;
    stats[worker].participates = participate ? 1 : 0;
    stats[worker].first_claim = 0xffffffffu;
    stats[worker].last_claim = 0xffffffffu;
  }
  __syncthreads();

  while (raw < tasks) {
    burn(work_cycles);
    if (threadIdx.x == 0) {
      atomicAdd(&visits[raw], 1);
      stats[worker].processed++;
    }
    __syncthreads();

    if (!participate)
      break;

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
      }
    }
    __syncthreads();
    raw = next_raw;
  }
}

int main(int argc, char **argv) {
  uint32_t tasks = argc > 1 ? std::atoi(argv[1]) : 8192;
  uint32_t threads = argc > 2 ? std::atoi(argv[2]) : 128;
  uint32_t claim_stride = argc > 3 ? std::atoi(argv[3]) : 1;
  uint32_t work_cycles = argc > 4 ? std::atoi(argv[4]) : 4096;
  if (!tasks || !threads || !claim_stride) {
    std::fprintf(stderr,
                 "usage: %s [tasks] [threads] [claim_stride] [work_cycles]\n",
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
  WorkerStatsPart *d_stats = nullptr;
  CHECK(cudaMalloc(&d_visits, tasks * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_claimed, tasks * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_stats, tasks * sizeof(WorkerStatsPart)));
  CHECK(cudaMemset(d_visits, 0, tasks * sizeof(uint32_t)));
  CHECK(cudaMemset(d_claimed, 0, tasks * sizeof(uint32_t)));
  CHECK(cudaMemset(d_stats, 0, tasks * sizeof(WorkerStatsPart)));

  clc_participation_probe<<<tasks, threads>>>(tasks, work_cycles, claim_stride,
                                              d_visits, d_claimed, d_stats);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  std::vector<uint32_t> visits(tasks), claimed(tasks);
  std::vector<WorkerStatsPart> stats(tasks);
  CHECK(cudaMemcpy(visits.data(), d_visits, tasks * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(claimed.data(), d_claimed, tasks * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(stats.data(), d_stats, tasks * sizeof(WorkerStatsPart),
                   cudaMemcpyDeviceToHost));

  uint64_t processed = 0, attempts = 0, successes = 0;
  uint32_t active_workers = 0, participants_launched = 0;
  uint32_t max_processed = 0, multi_claim_workers = 0;
  uint32_t first_claim_min = 0xffffffffu, first_claim_max = 0;
  uint32_t last_claim_min = 0xffffffffu, last_claim_max = 0;
  for (const WorkerStatsPart &s : stats) {
    processed += s.processed;
    attempts += s.attempts;
    successes += s.successes;
    max_processed = std::max(max_processed, s.processed);
    if (s.processed)
      active_workers++;
    if (s.participates && s.processed)
      participants_launched++;
    if (s.successes > 1)
      multi_claim_workers++;
    if (s.successes) {
      first_claim_min = std::min(first_claim_min, s.first_claim);
      first_claim_max = std::max(first_claim_max, s.first_claim);
      last_claim_min = std::min(last_claim_min, s.last_claim);
      last_claim_max = std::max(last_claim_max, s.last_claim);
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
      &occ_blocks, clc_participation_probe, threads, 0));
  uint32_t predicted_r = (uint32_t)occ_blocks * (uint32_t)sm_count;
  bool exactly_once = missed == 0 && duplicates == 0 && duplicate_claims == 0;

  if (std::getenv("CLC_PROBE_CSV")) {
    std::printf("tasks,threads,claim_stride,work_cycles,processed,"
                "active_workers,participants_launched,missed,duplicates,"
                "attempts,successes,unique_claimed,claimed_min,claimed_max,"
                "first_claim_min,first_claim_max,last_claim_min,last_claim_max,"
                "occ_blocks_per_sm,predicted_r,sm_count,max_processed,"
                "multi_claim_workers,duplicate_claims,claim_range_holes,"
                "exactly_once\n");
    std::printf("%u,%u,%u,%u,%llu,%u,%u,%u,%u,%llu,%llu,%u,%u,%u,%u,%u,%u,%u,"
                "%d,%u,%d,%u,%u,%u,%u,%u\n",
                tasks, threads, claim_stride, work_cycles,
                (unsigned long long)processed, active_workers,
                participants_launched, missed, duplicates,
                (unsigned long long)attempts, (unsigned long long)successes,
                claimed_unique, claimed_unique ? claimed_min : 0,
                claimed_unique ? claimed_max : 0,
                first_claim_min == 0xffffffffu ? 0 : first_claim_min,
                first_claim_max,
                last_claim_min == 0xffffffffu ? 0 : last_claim_min,
                last_claim_max, occ_blocks, predicted_r, sm_count,
                max_processed, multi_claim_workers, duplicate_claims,
                claim_range_holes, exactly_once ? 1 : 0);
  } else {
    std::printf("== CLC participation probe on %s sm_%d%d ==\n", prop.name,
                prop.major, prop.minor);
    std::printf("   tasks=%u threads=%u stride=%u R=%u\n", tasks, threads,
                claim_stride, predicted_r);
    std::printf("   active_workers=%u participants_launched=%u processed=%llu "
                "exactly_once=%s\n",
                active_workers, participants_launched,
                (unsigned long long)processed, exactly_once ? "yes" : "no");
    std::printf("   claims=%u range=%u..%u holes=%u first=%u..%u last=%u..%u "
                "max_processed=%u\n",
                claimed_unique, claimed_unique ? claimed_min : 0,
                claimed_unique ? claimed_max : 0, claim_range_holes,
                first_claim_min == 0xffffffffu ? 0 : first_claim_min,
                first_claim_max,
                last_claim_min == 0xffffffffu ? 0 : last_claim_min,
                last_claim_max, max_processed);
  }

  CHECK(cudaFree(d_visits));
  CHECK(cudaFree(d_claimed));
  CHECK(cudaFree(d_stats));
  return exactly_once ? 0 : 1;
}
