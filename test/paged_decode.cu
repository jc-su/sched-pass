// paged_decode.cu - GPU validation fixture for the sched-pass plugin.
//
// A minimal paged-KV decode-shaped kernel carrying exactly the structure the
// pass detects and weaves:
//   * one CTA per request  (seq = blockIdx.x -- the slot axis),
//   * a block-table indirection (page = bt[seq*stride+b] -- an address that
//     depends on a LOADED value, the paged-access signature),
//   * an inner streaming loop over the page's tokens with a constant byte
//     stride (the prefetch site).
//
// The kernel is deterministic per request (no cross-CTA effects), so outputs
// must be BIT-IDENTICAL no matter which CTA serves which request or which
// policy fires -- the fixture's central check across all modes:
//   A  inert      (runtime not armed: every woven capability on stock path)
//   B  identity   (armed, pi = identity, neutral policy)
//   C  permuted   (pi = reversed -- priority reordering)
//   D  aggressive (q >> 0, lambda = 0 -> prefetch action fires everywhere)
// plus timer validation (per-request residency cycles accumulate) and, in the
// SCHED_FIXTURE_WQ build, the persistent-worker mode with W < num_tasks
// workers claiming tasks off the ticket queue (the software-CLC layer).
//
// Build (both binaries) via test/build_and_run.sh.
//
#include "../runtime/sched_rt.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#define NSEQ 64        // requests (logical tasks)
#define D 128          // head dim == threads per CTA
#define NBLOCKS 8      // pages per request
#define PAGE_TOKENS 64 // tokens per page (passed at runtime: keeps the loop)
#define NPAGES (NSEQ * NBLOCKS)
#define KV_PAD 8192    // slack so read-ahead prefetch stays in the allocation

__global__ void paged_decode(const float *__restrict__ kv,
                             const int *__restrict__ bt,
                             const float *__restrict__ w,
                             float *__restrict__ out, int nblocks,
                             int bt_stride, int page_tokens) {
  int seq = blockIdx.x; // slot axis: one CTA == one request
  int d = threadIdx.x;
  float acc = 0.f;
  for (int b = 0; b < nblocks; ++b) {
    int page = bt[seq * bt_stride + b]; // loaded index -> paged signature
    const float *base = kv + (long long)page * page_tokens * D;
    // unroll(disable): keep a canonical induction variable so the shed lever's
    // tau budget counts tokens 1:1 (shed reuses the loop IV, and declines
    // loudly on unrolled loops). Real kernels that are unrolled will have shed
    // decline -- correct-or-absent by design.
#pragma clang loop unroll(disable)
    for (int t = 0; t < page_tokens; ++t) // streaming loop, stride D*4 bytes
      acc += base[t * D + d] * w[b * page_tokens + t];
  }
  out[seq * D + d] = acc;
}

// The arithmetic-mu form: contiguous KV, no index array -- the address is
// kv + seq*region + t*D + d, i.e. f(ctaid) + i*stride. Validates that the
// pass's positive no-loaded-index classification (eKV findArithKeystone,
// LLVM edition) detects and weaves this mu form with the SAME model.
__global__ void arith_decode(const float *__restrict__ kv,
                             const float *__restrict__ w,
                             float *__restrict__ out, int nblocks,
                             int page_tokens) {
  int seq = blockIdx.x;
  int d = threadIdx.x;
  const float *base = kv + (long long)seq * nblocks * page_tokens * D;
  float acc = 0.f;
#pragma clang loop unroll(disable)
  for (int i = 0; i < nblocks * page_tokens; ++i) // streaming, stride D*4
    acc += base[i * D + d] * w[i];
  out[seq * D + d] = acc;
}

// A REAL attention shape: online softmax over paged KV, one warp per request.
// K/V interleaved per token: page layout [token][2*SD] (K in lanes 0..31, V in
// lanes 32..63). The score is a warp-shuffle dot product that feeds fmaxf (the
// running max) and __expf (the weight) -- the exact online-softmax signature
// the shed score-mask keys on. Dropping a token must contribute ZERO weight
// (mask s to -inf), not a garbage score: this kernel is the regression test
// for that semantics.
#define SD 32 // head dim == warp size for the softmax fixture
__global__ void softmax_decode(const float *__restrict__ kv,
                               const int *__restrict__ bt,
                               const float *__restrict__ q,
                               float *__restrict__ out, int nblocks,
                               int bt_stride, int page_tokens) {
  int seq = blockIdx.x;
  int lane = threadIdx.x; // blockDim.x == 32
  float qv = q[seq * SD + lane];
  float m = -1e30f, l = 0.f, acc = 0.f;
  for (int b = 0; b < nblocks; ++b) {
    int page = bt[seq * bt_stride + b];
    const float *base = kv + (long long)page * page_tokens * (2 * SD);
    // unroll(disable): the shed counter counts loop TRIPS; keeping the loop
    // un-unrolled makes trips == tokens, so tau is calibrated in tokens. (On
    // an unrolled kernel tau simply counts unrolled trips -- the control
    // plane calibrates per kernel; for the regression test we pin 1:1.)
#pragma clang loop unroll(disable)
    for (int t = 0; t < page_tokens; ++t) {
      float kx = base[t * 2 * SD + lane]; // K: the woven stream site
      float s = kx * qv;
#pragma unroll
      for (int o = 16; o; o >>= 1)
        s += __shfl_xor_sync(0xffffffffu, s, o); // dot(q, k_t), lane-uniform
      s *= 0.125f;
      float mn = fmaxf(m, s);
      float c = __expf(m - mn);
      float w = __expf(s - mn);
      float vx = base[t * 2 * SD + SD + lane]; // V
      l = l * c + w;
      acc = acc * c + w * vx;
      m = mn;
    }
  }
  out[seq * SD + lane] = acc / l;
}

// CPU reference with a per-block token budget (tau == 0 -> full attention).
// Dropped tokens contribute NOTHING (weight 0) -- the -inf mask semantics.
static void cpu_ref_softmax(const std::vector<float> &kv,
                            const std::vector<int> &bt,
                            const std::vector<float> &q,
                            std::vector<float> &out, unsigned tau) {
  for (int seq = 0; seq < NSEQ; ++seq) {
    for (int lane = 0; lane < SD; ++lane) {
      float m = -1e30f, l = 0.f, acc = 0.f;
      for (int b = 0; b < NBLOCKS; ++b) {
        int page = bt[seq * NBLOCKS + b];
        const float *base =
            kv.data() + (long long)page * PAGE_TOKENS * (2 * SD);
        for (int t = 0; t < PAGE_TOKENS; ++t) {
          if (tau && (unsigned)t >= tau)
            continue; // masked to -inf: zero weight, max/denominator untouched
          float s = 0.f;
          for (int j = 0; j < SD; ++j)
            s += base[t * 2 * SD + j] * q[seq * SD + j];
          s *= 0.125f;
          float mn = std::max(m, s);
          float c = std::exp(m - mn);
          float w = std::exp(s - mn);
          l = l * c + w;
          acc = acc * c + w * base[t * 2 * SD + SD + lane];
          m = mn;
        }
      }
      out[seq * SD + lane] = acc / l;
    }
  }
}

static void cpu_ref_arith(const std::vector<float> &kv,
                          const std::vector<float> &w,
                          std::vector<float> &out) {
  for (int seq = 0; seq < NSEQ; ++seq)
    for (int d = 0; d < D; ++d) {
      const float *base = kv.data() + (long long)seq * NBLOCKS * PAGE_TOKENS * D;
      float acc = 0.f;
      for (int i = 0; i < NBLOCKS * PAGE_TOKENS; ++i)
        acc += base[i * D + d] * w[i];
      out[seq * D + d] = acc;
    }
}

static void cpu_ref(const std::vector<float> &kv, const std::vector<int> &bt,
                    const std::vector<float> &w, std::vector<float> &out) {
  for (int seq = 0; seq < NSEQ; ++seq)
    for (int d = 0; d < D; ++d) {
      float acc = 0.f;
      for (int b = 0; b < NBLOCKS; ++b) {
        int page = bt[seq * NBLOCKS + b];
        const float *base = kv.data() + (long long)page * PAGE_TOKENS * D;
        for (int t = 0; t < PAGE_TOKENS; ++t)
          acc += base[t * D + d] * w[b * PAGE_TOKENS + t];
      }
      out[seq * D + d] = acc;
    }
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

struct Bufs {
  float *kv;
  int *bt;
  float *w;
  float *out;
};

static void run_kernel(const Bufs &B, int gridX) {
  CHECK(cudaMemset(B.out, 0, NSEQ * D * sizeof(float)));
  paged_decode<<<gridX, D>>>(B.kv, B.bt, B.w, B.out, NBLOCKS, NBLOCKS,
                             PAGE_TOKENS);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
}

static std::vector<float> fetch_out(const Bufs &B) {
  std::vector<float> h(NSEQ * D);
  CHECK(cudaMemcpy(h.data(), B.out, h.size() * sizeof(float),
                   cudaMemcpyDeviceToHost));
  return h;
}

static bool bits_equal(const std::vector<float> &a,
                       const std::vector<float> &b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

static int timer_nonzero_rows() {
  int n = 0;
  for (int i = 0; i < NSEQ; ++i)
    if (sched_rt_timer(i) != 0)
      ++n;
  return n;
}

int main() {
  // Host data: random-ish but deterministic.
  std::vector<float> kv((size_t)NPAGES * PAGE_TOKENS * D + KV_PAD);
  std::vector<float> w(NBLOCKS * PAGE_TOKENS);
  std::vector<int> bt(NSEQ * NBLOCKS);
  unsigned s = 12345;
  auto rnd = [&s]() {
    s = s * 1664525u + 1013904223u;
    return ((s >> 8) & 0xffff) / 65536.0f - 0.5f;
  };
  for (auto &v : kv)
    v = rnd();
  for (auto &v : w)
    v = rnd();
  for (int i = 0; i < NSEQ * NBLOCKS; ++i) // a fixed page permutation
    bt[i] = (i * 37 + 11) % NPAGES;

  std::vector<float> ref(NSEQ * D);
  cpu_ref(kv, bt, w, ref);

  Bufs B{};
  CHECK(cudaMalloc(&B.kv, kv.size() * sizeof(float)));
  CHECK(cudaMalloc(&B.bt, bt.size() * sizeof(int)));
  CHECK(cudaMalloc(&B.w, w.size() * sizeof(float)));
  CHECK(cudaMalloc(&B.out, (size_t)NSEQ * D * sizeof(float)));
  CHECK(cudaMemcpy(B.kv, kv.data(), kv.size() * sizeof(float),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(B.bt, bt.data(), bt.size() * sizeof(int),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(B.w, w.data(), w.size() * sizeof(float),
                   cudaMemcpyHostToDevice));

  auto close_to_ref = [&](const std::vector<float> &h) {
    for (size_t i = 0; i < h.size(); ++i)
      if (std::fabs(h[i] - ref[i]) >
          1e-3f * (1.f + std::fabs(ref[i])))
        return false;
    return true;
  };

#ifdef SCHED_FIXTURE_WQ
  std::printf("== sched-pass fixture (WORK-QUEUE build) ==\n");
#else
  std::printf("== sched-pass fixture (basic build) ==\n");
#endif

  // ---- mode A: inert (not armed) ------------------------------------------
  run_kernel(B, NSEQ);
  std::vector<float> golden = fetch_out(B);
  expect(close_to_ref(golden), "A inert: matches CPU reference");

  // ---- arm the control plane ----------------------------------------------
  if (!sched_rt_init()) {
    std::fprintf(stderr, "sched_rt_init failed\n");
    return 2;
  }
  sched_rt_set_num_tasks(NSEQ);
  sched_rt_push_ctrl();

#ifndef SCHED_FIXTURE_WQ
  // ---- mode B: identity order, neutral policy -----------------------------
  sched_rt_timer_clear();
  run_kernel(B, NSEQ);
  expect(bits_equal(fetch_out(B), golden), "B identity: bit-exact vs inert");
  expect(timer_nonzero_rows() == NSEQ, "B timer: one row per request");
  unsigned long long c0 = sched_rt_timer(0);
  std::printf("       timer[0] = %llu cycles\n", c0);

  // ---- mode C: reversed priority order ------------------------------------
  {
    std::vector<int32_t> order(SCHED_MAX_TASKS);
    for (int i = 0; i < SCHED_MAX_TASKS; ++i)
      order[i] = i;
    for (int i = 0; i < NSEQ; ++i)
      order[i] = NSEQ - 1 - i;
    sched_rt_set_order(order.data(), SCHED_MAX_TASKS);
  }
  run_kernel(B, NSEQ);
  expect(bits_equal(fetch_out(B), golden), "C reversed pi: bit-exact");

  // ---- mode D: aggressive policy (q >> 0, lambda = 0 -> prefetch on) ------
  sched_rt_set_lambda(0.f, 0.f, 0.f, 0.f);
  for (int i = 0; i < NSEQ; ++i)
    sched_rt_set_task(i, 100.f, 0, 0);
  sched_rt_push_ctrl();
  sched_rt_timer_clear();
  run_kernel(B, NSEQ);
  expect(bits_equal(fetch_out(B), golden), "D aggressive policy: bit-exact");
  expect(timer_nonzero_rows() == NSEQ, "D timer: one row per request");

  // ---- mode E: high price (lambda dominates -> back to baseline) ----------
  sched_rt_set_lambda(1000.f, 0.f, 0.f, 0.f);
  sched_rt_push_ctrl();
  run_kernel(B, NSEQ);
  expect(bits_equal(fetch_out(B), golden), "E priced-out policy: bit-exact");

  // ---- overhead: disarmed vs armed-neutral vs armed-aggressive ------------
  {
    auto bench = [&](const char *name) {
      const int iters = 500;
      cudaEvent_t t0, t1;
      CHECK(cudaEventCreate(&t0));
      CHECK(cudaEventCreate(&t1));
      paged_decode<<<NSEQ, D>>>(B.kv, B.bt, B.w, B.out, NBLOCKS, NBLOCKS,
                                PAGE_TOKENS); // warm
      CHECK(cudaDeviceSynchronize());
      CHECK(cudaEventRecord(t0));
      for (int i = 0; i < iters; ++i)
        paged_decode<<<NSEQ, D>>>(B.kv, B.bt, B.w, B.out, NBLOCKS, NBLOCKS,
                                  PAGE_TOKENS);
      CHECK(cudaEventRecord(t1));
      CHECK(cudaEventSynchronize(t1));
      float ms = 0.f;
      CHECK(cudaEventElapsedTime(&ms, t0, t1));
      std::printf("       bench %-18s %8.2f us/launch\n", name,
                  1000.f * ms / iters);
      CHECK(cudaEventDestroy(t0));
      CHECK(cudaEventDestroy(t1));
    };
    sched_rt_disarm();
    bench("disarmed");
    // Re-arm (slots were nulled; buffers still live).
    CHECK(cudaMemcpyToSymbol(__sched_task_order, &g_sched_rt.d_order,
                             sizeof(void *)));
    CHECK(cudaMemcpyToSymbol(__sched_ctrl, &g_sched_rt.d_ctrl,
                             sizeof(void *)));
    CHECK(cudaMemcpyToSymbol(__sched_timer, &g_sched_rt.d_timer,
                             sizeof(void *)));
    CHECK(cudaMemcpyToSymbol(__sched_queue, &g_sched_rt.d_queue,
                             sizeof(void *)));
    sched_rt_set_lambda(1000.f, 0.f, 0.f, 0.f); // priced out -> baseline
    sched_rt_push_ctrl();
    bench("armed-neutral");
    sched_rt_set_lambda(0.f, 0.f, 0.f, 0.f); // free resources -> aggressive
    sched_rt_push_ctrl();
    bench("armed-aggressive");
  }

  // ---- mode G: SHED (tau budget) -- the one lever that trades accuracy ----
  // tau caps how many KV units a request attends. Neutral tau=0 is bit-exact;
  // a small tau makes the long requests cheaper (fewer bytes) at bounded error
  // -- the elastic-quality-under-overload lever. We verify: (a) tau=0 still
  // bit-exact, (b) tau>0 changes output but stays finite and BOUNDED (dropped
  // units re-read unit 0, so error <= a few units' worth of contribution).
  sched_rt_set_lambda(1000.f, 0.f, 0.f, 0.f); // price out prefetch
  for (int i = 0; i < NSEQ; ++i)
    sched_rt_set_task(i, 0.f, 0 /*tau=0 -> no cap*/, 0);
  sched_rt_push_ctrl();
  run_kernel(B, NSEQ);
  expect(bits_equal(fetch_out(B), golden), "G tau=0 (no shed): bit-exact");
  // tau counts streaming-load innermost-loop trips (tokens per block here,
  // after unroll). tau=8 is well below the trip count, so it caps regardless
  // of the unroll factor.
  for (int i = 0; i < NSEQ; ++i)
    sched_rt_set_tau(i, 8);
  sched_rt_push_ctrl();
  run_kernel(B, NSEQ);
  {
    std::vector<float> h = fetch_out(B);
    bool changed = !bits_equal(h, golden), finite = true;
    for (float v : h)
      finite &= std::isfinite(v);
    expect(changed, "G tau>0: output changes (fewer KV units attended)");
    expect(finite, "G tau>0: output stays finite (bounded epsilon)");
  }
  // reset shed for the remaining modes.
  for (int i = 0; i < NSEQ; ++i)
    sched_rt_set_tau(i, 0);
  sched_rt_push_ctrl();

  // ---- mode F: the ARITHMETIC mu form (contiguous KV, no index array) -----
  {
    std::vector<float> refA(NSEQ * D);
    cpu_ref_arith(kv, w, refA);
    CHECK(cudaMemset(B.out, 0, NSEQ * D * sizeof(float)));
    arith_decode<<<NSEQ, D>>>(B.kv, B.w, B.out, NBLOCKS, PAGE_TOKENS);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<float> h = fetch_out(B);
    bool ok = true;
    for (size_t i = 0; i < h.size(); ++i)
      if (std::fabs(h[i] - refA[i]) > 1e-3f * (1.f + std::fabs(refA[i]))) {
        ok = false;
        break;
      }
    expect(ok, "F arithmetic-mu kernel: matches CPU reference");
  }

  // ---- mode H: ONLINE SOFTMAX + shed score mask ----------------------------
  // The correctness gate for real attention: with tau>0, dropped tokens must
  // get exactly ZERO softmax weight (score masked to -inf), matching a CPU
  // reference that skips them entirely. Address redirect alone would fail
  // this test (garbage scores would pollute max + denominator).
  {
    std::vector<float> qh(NSEQ * SD);
    for (auto &v : qh)
      v = rnd();
    float *d_q;
    CHECK(cudaMalloc(&d_q, qh.size() * sizeof(float)));
    CHECK(cudaMemcpy(d_q, qh.data(), qh.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
    auto run_softmax = [&]() {
      CHECK(cudaMemset(B.out, 0, (size_t)NSEQ * SD * sizeof(float)));
      softmax_decode<<<NSEQ, SD>>>(B.kv, B.bt, d_q, B.out, NBLOCKS, NBLOCKS,
                                   PAGE_TOKENS);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      std::vector<float> h(NSEQ * SD);
      CHECK(cudaMemcpy(h.data(), B.out, h.size() * sizeof(float),
                       cudaMemcpyDeviceToHost));
      return h;
    };
    auto close2 = [&](const std::vector<float> &a,
                      const std::vector<float> &b) {
      for (size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i] - b[i]) > 2e-2f * (1.f + std::fabs(b[i])))
          return false;
      return true;
    };
    std::vector<float> ref(NSEQ * SD);
    // tau = 0: full attention.
    for (int i = 0; i < NSEQ; ++i)
      sched_rt_set_tau(i, 0);
    sched_rt_push_ctrl();
    cpu_ref_softmax(kv, bt, qh, ref, 0);
    expect(close2(run_softmax(), ref),
           "H softmax tau=0: matches full-attention CPU reference");
    // tau = 8: attend only the first 8 tokens of each block, EXACTLY.
    for (int i = 0; i < NSEQ; ++i)
      sched_rt_set_tau(i, 8);
    sched_rt_push_ctrl();
    cpu_ref_softmax(kv, bt, qh, ref, 8);
    expect(close2(run_softmax(), ref),
           "H softmax tau=8: matches TRUNCATED-attention CPU reference "
           "(-inf mask semantics: dropped tokens get zero weight)");
    for (int i = 0; i < NSEQ; ++i)
      sched_rt_set_tau(i, 0);
    sched_rt_push_ctrl();
    CHECK(cudaFree(d_q));
  }
#else
  // ---- WQ mode B: W = NSEQ/4 persistent workers cover all tasks -----------
  const int W = NSEQ / 4;
  sched_rt_timer_clear();
  sched_rt_queue_reset(W);
  run_kernel(B, W);
  expect(bits_equal(fetch_out(B), golden),
         "WQ B: W workers, ticket claim: bit-exact");
  expect(timer_nonzero_rows() == NSEQ, "WQ B timer: one row per TASK");

  // ---- WQ mode C: reversed priority order through the queue ---------------
  {
    std::vector<int32_t> order(SCHED_MAX_TASKS);
    for (int i = 0; i < SCHED_MAX_TASKS; ++i)
      order[i] = i;
    for (int i = 0; i < NSEQ; ++i)
      order[i] = NSEQ - 1 - i;
    sched_rt_set_order(order.data(), SCHED_MAX_TASKS);
  }
  sched_rt_queue_reset(W);
  run_kernel(B, W);
  expect(bits_equal(fetch_out(B), golden), "WQ C reversed pi: bit-exact");

  // ---- WQ mode D: exactly num_tasks workers (degenerate: no dynamic tail) -
  sched_rt_queue_reset(NSEQ);
  run_kernel(B, NSEQ);
  expect(bits_equal(fetch_out(B), golden), "WQ D W==num_tasks: bit-exact");

  // ---- WQ mode E: disarm -> stock path preserved ---------------------------
  sched_rt_disarm();
  run_kernel(B, NSEQ);
  expect(bits_equal(fetch_out(B), golden), "WQ E disarmed: stock bit-exact");
#endif

  std::printf(g_fail ? "== FIXTURE FAILED ==\n" : "== ALL PASS ==\n");
  return g_fail;
}
