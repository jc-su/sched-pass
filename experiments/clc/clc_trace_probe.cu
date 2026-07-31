// Standalone Blackwell CLC trace probe.
//
// Records the first successful CLC claim completions in observed atomic order.
// This does not expose NVIDIA's internal queue; it shows what a kernel can
// observe while using clusterlaunchcontrol.try_cancel.

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

struct TraceEvent {
  uint32_t seq;
  uint32_t worker_linear;
  uint32_t worker_x;
  uint32_t worker_y;
  uint32_t processed_before;
  uint32_t claimed_linear;
  uint32_t claimed_x;
  uint32_t claimed_y;
  uint32_t claimed_z;
  uint64_t claim_cycles;
};

struct WorkerStatsTrace {
  uint32_t processed;
  uint32_t attempts;
  uint32_t successes;
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

__device__ __forceinline__ CtaId clc_try_cancel_3d_trace() {
  CtaId out;
#if __CUDA_ARCH__ >= 1000
  asm volatile(
      "{\n"
      "  .reg .pred %%pc;\n"
      "  .shared .align 16 .b8 _trace_clc_res[16];\n"
      "  .shared .align 8 .b64 _trace_clc_bar;\n"
      "  .reg .b128 %%rq;\n"
      "  .reg .b64 %%tmp;\n"
      "  .reg .b32 %%resa, %%bara, %%cx, %%cy, %%cz, %%cw, %%ok;\n"
      "  mov.u32 %%resa, _trace_clc_res;\n"
      "  mov.u32 %%bara, _trace_clc_bar;\n"
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
      "L_trace_clc_wait:\n"
      "  mbarrier.try_wait.parity.shared::cta.b64 %%pc, [%%bara], 0;\n"
      "  @!%%pc bra L_trace_clc_wait;\n"
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

__global__ void clc_trace_probe(uint32_t grid_x, uint32_t grid_y,
                                uint32_t tasks, uint32_t work_cycles,
                                uint32_t *visits, uint32_t *claimed_hist,
                                WorkerStatsTrace *stats, TraceEvent *trace,
                                uint32_t *trace_count, uint32_t trace_cap) {
  __shared__ CtaId next;

  const uint32_t worker =
      linear_id(blockIdx.x, blockIdx.y, blockIdx.z, grid_x, grid_y);
  CtaId cur{blockIdx.x, blockIdx.y, blockIdx.z, 1};

  if (threadIdx.x == 0) {
    stats[worker].processed = 0;
    stats[worker].attempts = 0;
    stats[worker].successes = 0;
    stats[worker].first_claim = 0xffffffffu;
    stats[worker].last_claim = 0xffffffffu;
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
      CtaId claimed = clc_try_cancel_3d_trace();
      uint64_t dt = clock64_dev() - t0;
      next = claimed;

      stats[worker].attempts++;
      if (claimed.canceled) {
        uint32_t claimed_linear =
            linear_id(claimed.x, claimed.y, claimed.z, grid_x, grid_y);
        stats[worker].successes++;
        if (stats[worker].first_claim == 0xffffffffu)
          stats[worker].first_claim = claimed_linear;
        stats[worker].last_claim = claimed_linear;
        if (claimed_linear < tasks)
          atomicAdd(&claimed_hist[claimed_linear], 1);

        uint32_t seq = atomicAdd(trace_count, 1);
        if (seq < trace_cap) {
          trace[seq].seq = seq;
          trace[seq].worker_linear = worker;
          trace[seq].worker_x = blockIdx.x;
          trace[seq].worker_y = blockIdx.y;
          trace[seq].processed_before = stats[worker].processed;
          trace[seq].claimed_linear = claimed_linear;
          trace[seq].claimed_x = claimed.x;
          trace[seq].claimed_y = claimed.y;
          trace[seq].claimed_z = claimed.z;
          trace[seq].claim_cycles = dt;
        }
      }
    }
    __syncthreads();
    cur = next;
  }
}

int main(int argc, char **argv) {
  uint32_t grid_x = argc > 1 ? std::atoi(argv[1]) : 8192;
  uint32_t grid_y = argc > 2 ? std::atoi(argv[2]) : 1;
  uint32_t threads = argc > 3 ? std::atoi(argv[3]) : 128;
  uint32_t work_cycles = argc > 4 ? std::atoi(argv[4]) : 4096;
  uint32_t trace_cap = argc > 5 ? std::atoi(argv[5]) : 256;
  if (!grid_x || !grid_y || !threads || !trace_cap) {
    std::fprintf(stderr,
                 "usage: %s [grid_x] [grid_y] [threads] [work_cycles] "
                 "[trace_cap]\n",
                 argv[0]);
    return 2;
  }

  uint32_t tasks = grid_x * grid_y;
  dim3 grid(grid_x, grid_y, 1);
  int dev = 0;
  cudaDeviceProp prop{};
  CHECK(cudaGetDevice(&dev));
  CHECK(cudaGetDeviceProperties(&prop, dev));
  int sm_count = 0;
  CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));

  uint32_t *d_visits = nullptr, *d_claimed = nullptr, *d_trace_count = nullptr;
  WorkerStatsTrace *d_stats = nullptr;
  TraceEvent *d_trace = nullptr;
  CHECK(cudaMalloc(&d_visits, tasks * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_claimed, tasks * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_stats, tasks * sizeof(WorkerStatsTrace)));
  CHECK(cudaMalloc(&d_trace, trace_cap * sizeof(TraceEvent)));
  CHECK(cudaMalloc(&d_trace_count, sizeof(uint32_t)));
  CHECK(cudaMemset(d_visits, 0, tasks * sizeof(uint32_t)));
  CHECK(cudaMemset(d_claimed, 0, tasks * sizeof(uint32_t)));
  CHECK(cudaMemset(d_stats, 0, tasks * sizeof(WorkerStatsTrace)));
  CHECK(cudaMemset(d_trace, 0, trace_cap * sizeof(TraceEvent)));
  CHECK(cudaMemset(d_trace_count, 0, sizeof(uint32_t)));

  clc_trace_probe<<<grid, threads>>>(grid_x, grid_y, tasks, work_cycles,
                                     d_visits, d_claimed, d_stats, d_trace,
                                     d_trace_count, trace_cap);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  std::vector<uint32_t> visits(tasks), claimed(tasks);
  std::vector<WorkerStatsTrace> stats(tasks);
  std::vector<TraceEvent> trace(trace_cap);
  uint32_t trace_count = 0;
  CHECK(cudaMemcpy(visits.data(), d_visits, tasks * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(claimed.data(), d_claimed, tasks * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(stats.data(), d_stats, tasks * sizeof(WorkerStatsTrace),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(trace.data(), d_trace, trace_cap * sizeof(TraceEvent),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(&trace_count, d_trace_count, sizeof(uint32_t),
                   cudaMemcpyDeviceToHost));

  uint64_t processed = 0, attempts = 0, successes = 0;
  uint32_t active_workers = 0, max_processed = 0, multi_claim_workers = 0;
  for (const WorkerStatsTrace &s : stats) {
    processed += s.processed;
    attempts += s.attempts;
    successes += s.successes;
    max_processed = std::max(max_processed, s.processed);
    if (s.successes > 1)
      multi_claim_workers++;
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
      &occ_blocks, clc_trace_probe, threads, 0));
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

  bool monotonic_trace = true;
  uint32_t ntrace = std::min(trace_count, trace_cap);
  for (uint32_t i = 1; i < ntrace; ++i) {
    if (trace[i].claimed_linear <= trace[i - 1].claimed_linear)
      monotonic_trace = false;
  }

  if (std::getenv("CLC_TRACE_EVENTS_CSV")) {
    std::printf("seq,worker_linear,worker_x,worker_y,processed_before,"
                "claimed_linear,claimed_x,claimed_y,claimed_z,claim_cycles\n");
    for (uint32_t i = 0; i < ntrace; ++i) {
      const TraceEvent &e = trace[i];
      std::printf("%u,%u,%u,%u,%u,%u,%u,%u,%u,%llu\n", e.seq,
                  e.worker_linear, e.worker_x, e.worker_y, e.processed_before,
                  e.claimed_linear, e.claimed_x, e.claimed_y, e.claimed_z,
                  (unsigned long long)e.claim_cycles);
    }
  } else if (std::getenv("CLC_PROBE_CSV")) {
    std::printf("grid_x,grid_y,tasks,threads,work_cycles,trace_cap,"
                "trace_count,trace_recorded,monotonic_trace,processed,"
                "active_workers,missed,duplicates,attempts,successes,"
                "unique_claimed,claimed_min,claimed_max,occ_blocks_per_sm,"
                "predicted_r,sm_count,expected_active_workers,"
                "expected_claimed,max_processed,multi_claim_workers,"
                "duplicate_claims,claim_range_holes,structural_ok\n");
    std::printf("%u,%u,%u,%u,%u,%u,%u,%u,%u,%llu,%u,%u,%u,%llu,%llu,%u,%u,%u,"
                "%d,%u,%d,%u,%u,%u,%u,%u,%u,%u\n",
                grid_x, grid_y, tasks, threads, work_cycles, trace_cap,
                trace_count, ntrace, monotonic_trace ? 1 : 0,
                (unsigned long long)processed, active_workers, missed,
                duplicates, (unsigned long long)attempts,
                (unsigned long long)successes, claimed_unique,
                claimed_unique ? claimed_min : 0, claimed_unique ? claimed_max : 0,
                occ_blocks, predicted_r, sm_count, expected_active,
                expected_claimed, max_processed, multi_claim_workers,
                duplicate_claims, claim_range_holes, structural_ok ? 1 : 0);
  } else {
    std::printf("== CLC trace probe on %s sm_%d%d ==\n", prop.name, prop.major,
                prop.minor);
    std::printf("   grid=%ux%u tasks=%u threads=%u work_cycles=%u "
                "trace=%u/%u\n",
                grid_x, grid_y, tasks, threads, work_cycles, ntrace,
                trace_count);
    std::printf("   R=%u active=%u claimed=%u..%u missed=%u duplicates=%u "
                "structural_ok=%s monotonic_trace=%s\n",
                predicted_r, active_workers, claimed_unique ? claimed_min : 0,
                claimed_unique ? claimed_max : 0, missed, duplicates,
                structural_ok ? "yes" : "no",
                monotonic_trace ? "yes" : "no");
    for (uint32_t i = 0; i < std::min(ntrace, 32u); ++i) {
      const TraceEvent &e = trace[i];
      std::printf("   trace[%u] worker=%u(%u,%u) processed=%u claim=%u(%u,%u,%u)"
                  " cycles=%llu\n",
                  i, e.worker_linear, e.worker_x, e.worker_y,
                  e.processed_before, e.claimed_linear, e.claimed_x,
                  e.claimed_y, e.claimed_z, (unsigned long long)e.claim_cycles);
    }
  }

  CHECK(cudaFree(d_visits));
  CHECK(cudaFree(d_claimed));
  CHECK(cudaFree(d_stats));
  CHECK(cudaFree(d_trace));
  CHECK(cudaFree(d_trace_count));
  return structural_ok ? 0 : 1;
}
