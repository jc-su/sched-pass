// hetero_batch.cu - the motivating experiment: a heterogeneous continuous
// batch where slow (long-KV) requests straggle the whole step, and the woven
// scheduler fixes it WITHOUT touching results.
//
// 64 logical tasks: 8 long (32 KV blocks) + 56 short (4 KV blocks), served by
// W=16 persistent worker CTAs (the SCHED_WORKQUEUE transform; ticket claim =
// the software CLC). Step time = makespan over workers, so a bad service
// ORDER (longs claimed last) leaves 8 workers grinding long requests after
// all shorts are done -- the classic straggler tail. The closed loop:
//
//   1. PROFILE   one step with the woven clock64 timer -> per-task cycles
//                (this is WHY the timer exists: makespan = max over workers,
//                so the scheduler must know each task's cost to order them)
//   2. DECIDE    control plane sorts pi by measured cycles descending (LPT --
//                Graham's list scheduling; guarantees <= 4/3 OPT makespan)
//   3. ENFORCE   writes pi into __sched_task_order; workers claim through it
//   4. VERIFY    outputs stay BIT-EXACT (order changes when, never what)
//
// Also exercises the cache-policy tier: long tasks hinted POLITE stream their
// KV with L2::evict_first (no reuse -> don't pollute), shorts hinted URGENT
// prefetch with L2::evict_last. Compile: build_and_run.sh (work-queue env).
//
#include "../runtime/sched_rt.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

#ifndef NSEQ
#define NSEQ 64
#endif
#define D 128
#define PAGE_TOKENS 64
#ifndef LONGB
#define LONGB 32 // blocks per long task (KV footprint knob)
#endif
#define SHORTB 4 // blocks per short task
#define NPAGES (NSEQ * LONGB)
#define KV_PAD 8192
// Launch model differs by task-acquisition mode:
//   ticket (default): W persistent workers pull NSEQ tasks off the counter.
//   CLC (SCHED_FIXTURE_CLC): launch NSEQ blocks; the hardware try_cancel
//     steals UNLAUNCHED blocks of THIS grid for load balancing, so the grid
//     must span all tasks (a smaller grid has no unlaunched blocks to steal).
#ifdef SCHED_FIXTURE_CLC
#define W NSEQ
#define CLC_MODE 1
#else
#define W 16 // persistent workers
#define CLC_MODE 0
#endif

static bool is_long(int task) { return task % 8 == 0; } // 8 long, 56 short

__global__ void hetero_decode(const float *__restrict__ kv,
                              const int *__restrict__ bt,
                              const int *__restrict__ nbl,
                              const float *__restrict__ w,
                              float *__restrict__ out, int bt_stride,
                              int page_tokens) {
  int seq = blockIdx.x;
  int d = threadIdx.x;
  int nb = nbl[seq]; // per-request KV length: the heterogeneity
  float acc = 0.f;
  for (int b = 0; b < nb; ++b) {
    int page = bt[seq * bt_stride + b];
    const float *base = kv + (long long)page * page_tokens * D;
    for (int t = 0; t < page_tokens; ++t)
      acc += base[t * D + d] * w[t];
  }
  out[seq * D + d] = acc;
}

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t e = (call);                                                    \
    if (e != cudaSuccess) {                                                    \
      std::fprintf(stderr, "FATAL %s:%d %s: %s\n", __FILE__, __LINE__, #call,  \
                   cudaGetErrorString(e));                                     \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

static int g_fail = 0;
static void expect(bool ok, const char *what) {
  std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok)
    g_fail = 1;
}

int main() {
  std::vector<float> kv((size_t)NPAGES * PAGE_TOKENS * D + KV_PAD);
  std::vector<float> w(PAGE_TOKENS);
  std::vector<int> bt(NSEQ * LONGB);
  std::vector<int> nbl(NSEQ);
  unsigned s = 777;
  auto rnd = [&s]() {
    s = s * 1664525u + 1013904223u;
    return ((s >> 8) & 0xffff) / 65536.0f - 0.5f;
  };
  for (auto &v : kv)
    v = rnd();
  for (auto &v : w)
    v = rnd();
  for (int t = 0; t < NSEQ; ++t) {
    nbl[t] = is_long(t) ? LONGB : SHORTB;
    for (int b = 0; b < LONGB; ++b)
      bt[t * LONGB + b] = (t * LONGB + b * 17 + 5) % NPAGES;
  }

  float *d_kv, *d_w, *d_out;
  int *d_bt, *d_nbl;
  CHECK(cudaMalloc(&d_kv, kv.size() * sizeof(float)));
  CHECK(cudaMalloc(&d_w, w.size() * sizeof(float)));
  CHECK(cudaMalloc(&d_bt, bt.size() * sizeof(int)));
  CHECK(cudaMalloc(&d_nbl, nbl.size() * sizeof(int)));
  CHECK(cudaMalloc(&d_out, (size_t)NSEQ * D * sizeof(float)));
  CHECK(cudaMemcpy(d_kv, kv.data(), kv.size() * sizeof(float),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_w, w.data(), w.size() * sizeof(float),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_bt, bt.data(), bt.size() * sizeof(int),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_nbl, nbl.data(), nbl.size() * sizeof(int),
                   cudaMemcpyHostToDevice));

  auto launch = [&](int grid) {
    hetero_decode<<<grid, D>>>(d_kv, d_bt, d_nbl, d_w, d_out, LONGB,
                               PAGE_TOKENS);
    CHECK(cudaGetLastError());
  };
  auto fetch = [&]() {
    std::vector<float> h(NSEQ * D);
    CHECK(cudaMemcpy(h.data(), d_out, h.size() * sizeof(float),
                     cudaMemcpyDeviceToHost));
    return h;
  };

  std::printf("== hetero-batch scheduler demo: %d tasks (%d long x%d blocks, "
              "%d short x%d), %d workers ==\n",
              NSEQ, 8, LONGB, NSEQ - 8, SHORTB, W);

  // Golden: stock run (not armed -> WQ driver takes the stock path = STATIC:
  // each block serves only its own ctaid, no work-stealing).
  CHECK(cudaMemset(d_out, 0, (size_t)NSEQ * D * sizeof(float)));
  launch(NSEQ);
  CHECK(cudaDeviceSynchronize());
  std::vector<float> golden = fetch();

  // Static baseline makespan (the stock path: grid=NSEQ, no claim loop). This
  // is what the dynamic acquisition layer (ticket / CLC) must beat when the
  // load is imbalanced and exceeds resident capacity -- stragglers serialize.
  auto bench_static = [&]() -> float {
    const int iters = 100;
    launch(NSEQ);
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i)
      launch(NSEQ);
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CHECK(cudaEventDestroy(t0));
    CHECK(cudaEventDestroy(t1));
    return 1000.f * ms / iters;
  };
  float t_static = bench_static(); // measured UNARMED = the static schedule
  std::printf("       makespan %-24s %8.2f us/step\n",
              "static (no stealing)", t_static);

  if (!sched_rt_init())
    return 2;
  sched_rt_set_num_tasks(NSEQ);
  sched_rt_push_ctrl();

  auto run_once = [&](int grid) {
    sched_rt_queue_reset(grid);
    CHECK(cudaMemset(d_out, 0, (size_t)NSEQ * D * sizeof(float)));
    launch(grid);
    CHECK(cudaDeviceSynchronize());
  };

  auto bench = [&](const char *name) -> float {
    const int iters = 100;
    run_once(W); // warm
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) {
      sched_rt_queue_reset(W);
      launch(W);
    }
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CHECK(cudaEventElapsedTime(&ms, t0, t1));
    float us = 1000.f * ms / iters;
    std::printf("       makespan %-24s %8.2f us/step\n", name, us);
    CHECK(cudaEventDestroy(t0));
    CHECK(cudaEventDestroy(t1));
    return us;
  };
  auto set_order = [&](const std::vector<int32_t> &tasks) {
    std::vector<int32_t> order(SCHED_MAX_TASKS);
    for (int i = 0; i < SCHED_MAX_TASKS; ++i)
      order[i] = i;
    for (size_t i = 0; i < tasks.size(); ++i)
      order[i] = tasks[i];
    sched_rt_set_order(order.data(), SCHED_MAX_TASKS);
  };

  // ---- 0. the per-step DISARM switch: armed but num_tasks == 0 -------------
  // The driver must take the STOCK path (static grid + pi remap): this is the
  // baked-ABI fail-safe (an unprogrammed plane cannot eat the launch) and the
  // control plane's per-step claim-loop toggle -- a data write, no relaunch
  // change. Ordering must stay LIVE on the stock path (reversed pi is an E1
  // permutation -> bit-exact), so "claim off" never means "pi off".
  sched_rt_set_num_tasks(0);
  sched_rt_push_ctrl();
  run_once(NSEQ); // full grid: every block serves one task via the stock path
  expect(std::memcmp(fetch().data(), golden.data(),
                     golden.size() * sizeof(float)) == 0,
         "num_tasks=0 (disarm switch): stock static path bit-exact");
  {
    std::vector<int32_t> tasks;
    for (int t = NSEQ - 1; t >= 0; --t)
      tasks.push_back(t);
    set_order(tasks);
  }
  run_once(NSEQ);
  expect(std::memcmp(fetch().data(), golden.data(),
                     golden.size() * sizeof(float)) == 0,
         "num_tasks=0 + reversed pi: ordering live on the stock path");
  set_order({}); // identity again
  sched_rt_set_num_tasks(NSEQ);
  sched_rt_push_ctrl();

  // ---- 1. identity order: correctness + baseline ---------------------------
  run_once(W);
  expect(std::memcmp(fetch().data(), golden.data(),
                     golden.size() * sizeof(float)) == 0,
         "identity order, 16 workers: bit-exact vs stock");
  float t_ident = bench("identity");

  // ---- 2. segregated order: shorts first, longs last ----------------------
  // Classic list-scheduling calls this the adversarial straggler order (the
  // longs land at the end with no work to overlap them). But it is ALSO the
  // temporally SEGREGATED schedule: the longs run in a quiet phase with no
  // short-churn contending for DRAM/L2. Which effect dominates is exactly
  // what per-task times coupling through shared resources decides.
  {
    std::vector<int32_t> tasks;
    for (int t = 0; t < NSEQ; ++t)
      if (!is_long(t))
        tasks.push_back(t);
    for (int t = 0; t < NSEQ; ++t)
      if (is_long(t))
        tasks.push_back(t);
    set_order(tasks);
  }
  run_once(W);
  expect(std::memcmp(fetch().data(), golden.data(),
                     golden.size() * sizeof(float)) == 0,
         "segregated order (longs last): bit-exact");
  float t_seg = bench("segregated (longs last)");

  // ---- 3. CLOSED LOOP: profile with the woven timer, order by LPT ----------
  sched_rt_timer_clear();
  set_order({}); // identity for the profiling step
  run_once(W);
  std::vector<std::pair<unsigned long long, int32_t>> cost(NSEQ);
  for (int t = 0; t < NSEQ; ++t)
    cost[t] = {sched_rt_timer(t), t};
  std::sort(cost.rbegin(), cost.rend()); // descending measured cycles
  bool ordering_sane = true;
  for (int i = 0; i < 8; ++i) // the 8 costliest measured tasks are the longs
    if (!is_long(cost[i].second))
      ordering_sane = false;
  expect(ordering_sane,
         "profile: timer ranks the 8 long tasks costliest (observation works)");
  {
    std::vector<int32_t> tasks;
    for (auto &c : cost)
      tasks.push_back(c.second);
    set_order(tasks);
  }
  run_once(W);
  expect(std::memcmp(fetch().data(), golden.data(),
                     golden.size() * sizeof(float)) == 0,
         "LPT order (from measured cycles): bit-exact");
  float t_lpt = bench("LPT (longs first)");

  // ---- 4. cache-policy tier on top of LPT ---------------------------------
  for (int t = 0; t < NSEQ; ++t)
    sched_rt_set_task(t, 0.f, 0, is_long(t) ? 2 /*polite*/ : 1 /*urgent*/);
  sched_rt_push_ctrl();
  run_once(W);
  expect(std::memcmp(fetch().data(), golden.data(),
                     golden.size() * sizeof(float)) == 0,
         "LPT + cache hints (longs polite, shorts evict_last): bit-exact");
  float t_hint = bench("LPT + cache hints");

  // ---- 5. both levers: segregated order + cache hints ----------------------
  {
    std::vector<int32_t> tasks;
    for (int t = 0; t < NSEQ; ++t)
      if (!is_long(t))
        tasks.push_back(t);
    for (int t = 0; t < NSEQ; ++t)
      if (is_long(t))
        tasks.push_back(t);
    set_order(tasks);
  }
  run_once(W);
  expect(std::memcmp(fetch().data(), golden.data(),
                     golden.size() * sizeof(float)) == 0,
         "segregated + cache hints: bit-exact");
  float t_seg_hint = bench("segregated + cache hints");

  // ---- the honest reading ---------------------------------------------------
  // On this GPU/kernel the SEGREGATED schedule beats LPT interleaving: the
  // longs' per-iteration latency is lower in a quiet phase than mixed with
  // 8 workers churning shorts -- i.e. t_j is NOT a constant of the task, it
  // couples to co-runners through DRAM/L2 (the congestion term gamma of the
  // model is first-order, not a refinement). Ordering (pi) and resource
  // shaping (hints) are therefore complementary levers: the hints recover
  // part of the interleaving loss without changing the order.
  std::printf("       -- findings --\n");
  std::printf("       segregation vs LPT interleave : %+.1f%%  (order lever)\n",
              100.f * (t_lpt - t_seg) / t_lpt);
  std::printf("       cache hints on top of LPT     : %+.1f%%  (shaping lever)\n",
              100.f * (t_lpt - t_hint) / t_lpt);
  std::printf("       both levers (seg + hints)     : %+.1f%% vs identity\n",
              100.f * (t_ident - t_seg_hint) / t_ident);
  // The acquisition layer's own win: dynamic (ticket/CLC work-stealing)
  // vs the static schedule, on this imbalanced load.
  float t_dyn_best = std::min(std::min(t_ident, t_lpt), t_seg);
  std::printf("       dynamic (%s) vs static        : %+.1f%%  (%d tasks, %d workers)\n",
              CLC_MODE ? "CLC   " : "ticket", 100.f * (t_static - t_dyn_best) / t_static,
              NSEQ, W);
  expect(t_ident > 0 && t_seg > 0 && t_lpt > 0 && t_hint > 0 &&
             t_seg_hint > 0 && t_static > 0,
         "all schedules measured");
  // Whether reordering / stealing WINS is workload- and hardware-dependent
  // (big win when workers << tasks or under contention; ~0 when the grid
  // spans all tasks on an underutilized GPU that already load-balances). So
  // this is reported, not asserted -- the load-bearing guarantee is that
  // every schedule is bit-exact (checked per mode above), i.e. the scheduler
  // only ever changes WHEN/HOW, never WHAT.

  std::printf(g_fail ? "== FIXTURE FAILED ==\n" : "== ALL PASS ==\n");
  return g_fail;
}
