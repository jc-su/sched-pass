// Standalone Blackwell CLC CUDA Graph replay probe.
//
// Tests whether CLC's observable claim contract resets cleanly across repeated
// stream launches and CUDA Graph replays.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct GraphStats {
  uint32_t processed;
  uint32_t attempts;
  uint32_t successes;
  uint32_t failures;
  uint32_t first_claim;
  uint32_t last_claim;
};

struct ReplaySummary {
  uint32_t processed;
  uint32_t active_workers;
  uint32_t attempts;
  uint32_t successes;
  uint32_t failures;
  uint32_t missed;
  uint32_t duplicates;
  uint32_t unique_claimed;
  uint32_t duplicate_claims;
  uint32_t claim_range_holes;
  uint32_t claimed_min;
  uint32_t claimed_max;
  uint32_t first_claim_min;
  uint32_t first_claim_max;
  uint32_t last_claim_min;
  uint32_t last_claim_max;
  uint32_t structural_ok;
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
      "  .shared .align 16 .b8 _graph_clc_res[16];\n"
      "  .shared .align 8 .b64 _graph_clc_bar;\n"
      "  .reg .b128 %%rq;\n"
      "  .reg .b64 %%tmp;\n"
      "  .reg .b32 %%resa, %%bara, %%cx;\n"
      "  mov.u32 %%resa, _graph_clc_res;\n"
      "  mov.u32 %%bara, _graph_clc_bar;\n"
      "  mbarrier.init.shared::cta.b64 [%%bara], 1;\n"
      "  fence.proxy.async.shared::cta;\n"
      "  clusterlaunchcontrol.try_cancel.async.shared::cta."
      "mbarrier::complete_tx::bytes.b128 [%%resa], [%%bara];\n"
      "  mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 "
      "%%tmp, [%%bara], 16;\n"
      "L_graph_clc_wait:\n"
      "  mbarrier.try_wait.parity.shared::cta.b64 %%pc, [%%bara], 0;\n"
      "  @!%%pc bra L_graph_clc_wait;\n"
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

__global__ void clc_graph_kernel(uint32_t tasks, uint32_t work_cycles,
                                 uint32_t *visits, uint32_t *claimed_hist,
                                 GraphStats *stats) {
  __shared__ uint32_t next_raw;
  uint32_t worker = blockIdx.x;
  uint32_t raw = worker;

  if (threadIdx.x == 0) {
    stats[worker].processed = 0;
    stats[worker].attempts = 0;
    stats[worker].successes = 0;
    stats[worker].failures = 0;
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
}

static ReplaySummary summarize(uint32_t tasks, uint32_t predicted_r,
                               const std::vector<uint32_t> &visits,
                               const std::vector<uint32_t> &claimed,
                               const std::vector<GraphStats> &stats) {
  ReplaySummary s{};
  s.claimed_min = 0xffffffffu;
  s.first_claim_min = 0xffffffffu;
  s.last_claim_min = 0xffffffffu;

  for (const GraphStats &st : stats) {
    s.processed += st.processed;
    s.attempts += st.attempts;
    s.successes += st.successes;
    s.failures += st.failures;
    if (st.processed)
      s.active_workers++;
    if (st.successes) {
      s.first_claim_min = std::min(s.first_claim_min, st.first_claim);
      s.first_claim_max = std::max(s.first_claim_max, st.first_claim);
      s.last_claim_min = std::min(s.last_claim_min, st.last_claim);
      s.last_claim_max = std::max(s.last_claim_max, st.last_claim);
    }
  }

  for (uint32_t i = 0; i < tasks; ++i) {
    s.missed += visits[i] == 0;
    s.duplicates += visits[i] > 1;
    if (claimed[i]) {
      s.unique_claimed++;
      s.duplicate_claims += claimed[i] > 1;
      s.claimed_min = std::min(s.claimed_min, i);
      s.claimed_max = std::max(s.claimed_max, i);
    }
  }
  if (s.unique_claimed)
    s.claim_range_holes = s.claimed_max - s.claimed_min + 1 - s.unique_claimed;

  uint32_t expected_active = std::min(tasks, predicted_r);
  uint32_t expected_claimed = tasks > predicted_r ? tasks - predicted_r : 0;
  bool ok = s.processed == tasks && s.attempts == tasks &&
            s.active_workers == expected_active &&
            s.successes == expected_claimed &&
            s.failures == expected_active && s.missed == 0 &&
            s.duplicates == 0 && s.duplicate_claims == 0 &&
            s.claim_range_holes == 0;
  if (tasks <= predicted_r) {
    ok = ok && s.unique_claimed == 0;
  } else {
    ok = ok && s.unique_claimed == expected_claimed &&
         s.claimed_min == predicted_r && s.claimed_max == tasks - 1;
  }
  s.structural_ok = ok ? 1 : 0;
  if (!s.unique_claimed)
    s.claimed_min = 0;
  if (s.first_claim_min == 0xffffffffu)
    s.first_claim_min = 0;
  if (s.last_claim_min == 0xffffffffu)
    s.last_claim_min = 0;
  return s;
}

int main(int argc, char **argv) {
  uint32_t tasks = argc > 1 ? std::atoi(argv[1]) : 8192;
  uint32_t threads = argc > 2 ? std::atoi(argv[2]) : 128;
  uint32_t work_cycles = argc > 3 ? std::atoi(argv[3]) : 4096;
  uint32_t smem_bytes = argc > 4 ? std::atoi(argv[4]) : 0;
  uint32_t replays = argc > 5 ? std::atoi(argv[5]) : 5;
  uint32_t use_graph = argc > 6 ? std::atoi(argv[6]) : 1;
  if (!tasks || !threads || !replays) {
    std::fprintf(stderr,
                 "usage: %s [tasks] [threads] [work_cycles] [smem_bytes] "
                 "[replays] [use_graph]\n",
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
  GraphStats *d_stats = nullptr;
  CHECK(cudaMalloc(&d_visits, tasks * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_claimed, tasks * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_stats, tasks * sizeof(GraphStats)));

  cudaStream_t stream = nullptr;
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  if (use_graph) {
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    clc_graph_kernel<<<tasks, threads, smem_bytes, stream>>>(
        tasks, work_cycles, d_visits, d_claimed, d_stats);
    CHECK(cudaGetLastError());
    CHECK(cudaStreamEndCapture(stream, &graph));
    CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  }

  int occ_blocks = 0;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occ_blocks, clc_graph_kernel, threads, smem_bytes));
  uint32_t predicted_r = (uint32_t)occ_blocks * (uint32_t)sm_count;

  std::vector<uint32_t> visits(tasks), claimed(tasks);
  std::vector<GraphStats> stats(tasks);
  std::vector<ReplaySummary> summaries;
  summaries.reserve(replays);

  for (uint32_t replay = 0; replay < replays; ++replay) {
    CHECK(cudaMemsetAsync(d_visits, 0, tasks * sizeof(uint32_t), stream));
    CHECK(cudaMemsetAsync(d_claimed, 0, tasks * sizeof(uint32_t), stream));
    CHECK(cudaMemsetAsync(d_stats, 0, tasks * sizeof(GraphStats), stream));
    if (use_graph) {
      CHECK(cudaGraphLaunch(graph_exec, stream));
    } else {
      clc_graph_kernel<<<tasks, threads, smem_bytes, stream>>>(
          tasks, work_cycles, d_visits, d_claimed, d_stats);
      CHECK(cudaGetLastError());
    }
    CHECK(cudaMemcpyAsync(visits.data(), d_visits, tasks * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(claimed.data(), d_claimed, tasks * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(stats.data(), d_stats, tasks * sizeof(GraphStats),
                          cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));
    summaries.push_back(summarize(tasks, predicted_r, visits, claimed, stats));
  }

  uint32_t invalid = 0;
  for (const ReplaySummary &s : summaries)
    invalid += s.structural_ok == 0;

  auto min_field = [&](auto field) {
    uint32_t v = field(summaries[0]);
    for (const ReplaySummary &s : summaries)
      v = std::min(v, field(s));
    return v;
  };
  auto max_field = [&](auto field) {
    uint32_t v = field(summaries[0]);
    for (const ReplaySummary &s : summaries)
      v = std::max(v, field(s));
    return v;
  };

  if (std::getenv("CLC_PROBE_CSV")) {
    std::printf("tasks,threads,work_cycles,smem_bytes,replays,use_graph,"
                "occ_blocks_per_sm,predicted_r,sm_count,invalid_replays,"
                "active_workers_min,active_workers_max,successes_min,"
                "successes_max,failures_min,failures_max,claimed_min_min,"
                "claimed_min_max,claimed_max_min,claimed_max_max,missed_max,"
                "duplicates_max,duplicate_claims_max,claim_range_holes_max,"
                "all_structural_ok\n");
    std::printf("%u,%u,%u,%u,%u,%u,%d,%u,%d,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,"
                "%u,%u,%u,%u,%u\n",
                tasks, threads, work_cycles, smem_bytes, replays, use_graph,
                occ_blocks, predicted_r, sm_count, invalid,
                min_field([](const ReplaySummary &s) {
                  return s.active_workers;
                }),
                max_field([](const ReplaySummary &s) {
                  return s.active_workers;
                }),
                min_field([](const ReplaySummary &s) { return s.successes; }),
                max_field([](const ReplaySummary &s) { return s.successes; }),
                min_field([](const ReplaySummary &s) { return s.failures; }),
                max_field([](const ReplaySummary &s) { return s.failures; }),
                min_field([](const ReplaySummary &s) { return s.claimed_min; }),
                max_field([](const ReplaySummary &s) { return s.claimed_min; }),
                min_field([](const ReplaySummary &s) { return s.claimed_max; }),
                max_field([](const ReplaySummary &s) { return s.claimed_max; }),
                max_field([](const ReplaySummary &s) { return s.missed; }),
                max_field([](const ReplaySummary &s) { return s.duplicates; }),
                max_field([](const ReplaySummary &s) {
                  return s.duplicate_claims;
                }),
                max_field([](const ReplaySummary &s) {
                  return s.claim_range_holes;
                }),
                invalid == 0 ? 1 : 0);
  } else {
    std::printf("== CLC graph probe on %s sm_%d%d ==\n", prop.name,
                prop.major, prop.minor);
    std::printf("   mode=%s replays=%u predicted_R=%u invalid=%u\n",
                use_graph ? "graph" : "stream", replays, predicted_r,
                invalid);
    std::printf("   active_workers=%u..%u successes=%u..%u failures=%u..%u "
                "claimed_min=%u..%u claimed_max=%u..%u structural_ok=%s\n",
                min_field([](const ReplaySummary &s) {
                  return s.active_workers;
                }),
                max_field([](const ReplaySummary &s) {
                  return s.active_workers;
                }),
                min_field([](const ReplaySummary &s) { return s.successes; }),
                max_field([](const ReplaySummary &s) { return s.successes; }),
                min_field([](const ReplaySummary &s) { return s.failures; }),
                max_field([](const ReplaySummary &s) { return s.failures; }),
                min_field([](const ReplaySummary &s) { return s.claimed_min; }),
                max_field([](const ReplaySummary &s) { return s.claimed_min; }),
                min_field([](const ReplaySummary &s) { return s.claimed_max; }),
                max_field([](const ReplaySummary &s) { return s.claimed_max; }),
                invalid == 0 ? "yes" : "no");
  }

  if (graph_exec)
    CHECK(cudaGraphExecDestroy(graph_exec));
  if (graph)
    CHECK(cudaGraphDestroy(graph));
  CHECK(cudaStreamDestroy(stream));
  CHECK(cudaFree(d_visits));
  CHECK(cudaFree(d_claimed));
  CHECK(cudaFree(d_stats));
  return invalid == 0 ? 0 : 1;
}
