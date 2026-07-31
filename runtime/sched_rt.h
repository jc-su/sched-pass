// sched_rt.h - the sched-pass runtime contract (host + device side).
//
// The AOT-CUDA analogue of eKV's compile-time self-allocation: an LLVM pass
// running inside clang cannot allocate GPU buffers (compile process != run
// process), so instead the pass weaves against these NAMED device globals and
// the host arms them at runtime via cudaMemcpyToSymbol. Until armed they are
// null and every woven capability takes its stock path -- fail-safe by
// construction, the eKV rule.
//
// Include this header in exactly ONE .cu translation unit of the target app
// (it defines the device globals). The ABI (struct layout, table sizes) is
// pinned by static_asserts and must match lib/SchedUtil.h.
//
// Buffers:
//   __sched_task_order : i32[SCHED_MAX_TASKS], device mem. pi -- the priority
//                        order; task = order[raw]. Host default: identity.
//   __sched_ctrl       : SchedCtrl*, device mem. lambda + per-task policy rows.
//   __sched_timer      : u64[SCHED_MAX_TASKS], HOST-MAPPED pinned memory. The
//                        GPU atomically adds per-task residency cycles; the
//                        host (or an external observer) reads after sync.
//   __sched_queue      : u32*, device mem. The work-queue ticket counter
//                        (SCHED_WORKQUEUE builds). Prime to gridDim before
//                        each launch wave: tickets gridDim..num_tasks-1 are
//                        the dynamically claimed tail.
//
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

// ABI CONTRACT (host-app named-global path): this macro sizes the allocations
// AND the compile-time `SchedCtrl.task[SCHED_MAX_TASKS]` array below, while the
// PASS bounds every woven task index by its own `SCHED_MAX_TASKS` env
// (lib/SchedUtil.cpp). The two MUST match, or a woven kernel can index
// ctrl->task[] past this struct. They share ONE knob by design: default 4096
// on both sides; to change it, set the pass env AND build this TU with
// -DSCHED_MAX_TASKS=<same> (this `#ifndef` makes that override win). The
// host-side setters (sched_rt_set_order/_size) clamp writes to this size, so
// only the pass-env-vs-macro agreement is the operator's responsibility;
// sched_rt_init() publishes the runtime value in /tmp/sched_rt_<pid>.json
// ("max_tasks") for cross-checking against the build env. (The baked/FlashInfer
// path is unaffected: Python sizes the arena to its SCHED_MAX_TASKS -- one
// source, compute_bake_env.)
#ifndef SCHED_MAX_TASKS
#define SCHED_MAX_TASKS 4096
#endif

struct SchedPolicyRow {
  float q;              // urgency (deadline-slack derived, clipped, >= 0)
  uint16_t tau;         // shed budget: max KV units to attend (0 = unbounded)
  uint8_t hint;         // control-plane advisory action hint
  uint8_t _pad;
};

struct SchedCtrl {
  uint32_t generation;  // bumped by the control plane on each rewrite
  uint32_t num_tasks;   // logical tasks this step (work-queue bound; 0 = the
                        // WQ driver takes the stock path -- per-step disarm)
  float lambda[4];      // shadow prices: {bw, l2, smem, comp}
  uint32_t flags;       // bit0: TIMER OFF (per-step observation gate; the
                        // baked ABI cannot null its slots, so gating is data.
                        // 0 = timer on, the historical default)
  uint32_t order_size;  // tile count the installed pi was built for; the
                        // weave honors the order table only when this
                        // equals the launch's nctaid (bijectivity under
                        // scheduler overlap). 0 = unchecked (legacy).
  SchedPolicyRow task[SCHED_MAX_TASKS];
};

#define SCHED_CTRL_FLAG_TIMER_OFF 1u

// ABI pins -- must match lib/SchedUtil.h (kCtrl*Off).
static_assert(offsetof(SchedCtrl, num_tasks) == 4, "sched ABI: num_tasks");
static_assert(offsetof(SchedCtrl, lambda) == 8, "sched ABI: lambda");
static_assert(offsetof(SchedCtrl, flags) == 24, "sched ABI: flags");
static_assert(offsetof(SchedCtrl, order_size) == 28, "sched ABI: order_size");
static_assert(offsetof(SchedCtrl, task) == 32, "sched ABI: rows offset");
static_assert(sizeof(SchedPolicyRow) == 8, "sched ABI: row size");
static_assert(offsetof(SchedPolicyRow, tau) == 4, "sched ABI: row tau");
static_assert(offsetof(SchedPolicyRow, hint) == 6, "sched ABI: row hint");

// --- device-side slots (the pass finds these by name) ----------------------
extern "C" {
__device__ int32_t *__sched_task_order = nullptr;
__device__ SchedCtrl *__sched_ctrl = nullptr;
__device__ unsigned long long *__sched_timer = nullptr;
__device__ uint32_t *__sched_queue = nullptr;
}

// --- host-side control plane ------------------------------------------------
struct SchedRtState {
  int32_t *d_order = nullptr;
  SchedCtrl *d_ctrl = nullptr;
  unsigned long long *h_timer = nullptr; // host-mapped view
  unsigned long long *d_timer = nullptr; // device view of the same memory
  uint32_t *d_queue = nullptr;
  SchedCtrl h_ctrl; // host mirror, pushed wholesale by sched_rt_push_ctrl
  bool armed = false;
};

static SchedRtState g_sched_rt;

#define SCHED_RT_CHECK(call)                                                   \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::fprintf(stderr, "[sched_rt] %s failed: %s\n", #call,                \
                   cudaGetErrorString(_e));                                    \
      return false;                                                            \
    }                                                                          \
  } while (0)

// Allocate all tables, set identity/neutral defaults, arm the device slots,
// and publish the buffer addresses for external observers (the eKV JSON).
static inline bool sched_rt_init() {
  SchedRtState &S = g_sched_rt;
  if (S.armed)
    return true;
  // RE-ARM after disarm: buffers persist across disarm (only the device
  // slots were nulled), so re-point the slots at them without re-allocating
  // (re-allocation would leak). Jump straight to the arm step.
  if (S.d_order != nullptr)
    goto arm;

  // pi: identity by default == stock CTA->task mapping.
  SCHED_RT_CHECK(cudaMalloc(&S.d_order, SCHED_MAX_TASKS * sizeof(int32_t)));
  {
    int32_t *ident = (int32_t *)std::malloc(SCHED_MAX_TASKS * sizeof(int32_t));
    for (int i = 0; i < SCHED_MAX_TASKS; ++i)
      ident[i] = i;
    SCHED_RT_CHECK(cudaMemcpy(S.d_order, ident,
                              SCHED_MAX_TASKS * sizeof(int32_t),
                              cudaMemcpyHostToDevice));
    std::free(ident);
  }

  // Policy control block: neutral (lambda=0, q=0 -> every score = -H < 0 ->
  // baseline action everywhere).
  std::memset(&S.h_ctrl, 0, sizeof(SchedCtrl));
  SCHED_RT_CHECK(cudaMalloc(&S.d_ctrl, sizeof(SchedCtrl)));
  SCHED_RT_CHECK(
      cudaMemcpy(S.d_ctrl, &S.h_ctrl, sizeof(SchedCtrl), cudaMemcpyHostToDevice));

  // Timer: host-mapped pinned, so an external reader can consume it with no
  // GPU API calls (the eKV zero-touch readout path).
  SCHED_RT_CHECK(cudaHostAlloc(&S.h_timer,
                               SCHED_MAX_TASKS * sizeof(unsigned long long),
                               cudaHostAllocMapped));
  std::memset(S.h_timer, 0, SCHED_MAX_TASKS * sizeof(unsigned long long));
  SCHED_RT_CHECK(cudaHostGetDevicePointer(&S.d_timer, S.h_timer, 0));

  // Work-queue ticket counter.
  SCHED_RT_CHECK(cudaMalloc(&S.d_queue, sizeof(uint32_t)));
  SCHED_RT_CHECK(cudaMemset(S.d_queue, 0, sizeof(uint32_t)));

  // Arm the device slots (this is the moment instrumentation goes live).
arm:
  SCHED_RT_CHECK(cudaMemcpyToSymbol(__sched_task_order, &S.d_order,
                                    sizeof(void *)));
  SCHED_RT_CHECK(cudaMemcpyToSymbol(__sched_ctrl, &S.d_ctrl, sizeof(void *)));
  SCHED_RT_CHECK(cudaMemcpyToSymbol(__sched_timer, &S.d_timer, sizeof(void *)));
  SCHED_RT_CHECK(cudaMemcpyToSymbol(__sched_queue, &S.d_queue, sizeof(void *)));

  char path[256];
  std::snprintf(path, sizeof(path), "/tmp/sched_rt_%d.json", (int)getpid());
  if (FILE *f = std::fopen(path, "w")) {
    std::fprintf(f,
                 "{\"pid\":%d,\"max_tasks\":%d,"
                 "\"order_dev\":%llu,\"ctrl_dev\":%llu,"
                 "\"timer_host_va\":%llu,\"timer_bytes\":%zu,"
                 "\"queue_dev\":%llu}\n",
                 (int)getpid(), SCHED_MAX_TASKS,
                 (unsigned long long)(uintptr_t)S.d_order,
                 (unsigned long long)(uintptr_t)S.d_ctrl,
                 (unsigned long long)(uintptr_t)S.h_timer,
                 SCHED_MAX_TASKS * sizeof(unsigned long long),
                 (unsigned long long)(uintptr_t)S.d_queue);
    std::fclose(f);
    std::fprintf(stderr, "[sched_rt] armed; published %s\n", path);
  }
  S.armed = true;
  return true;
}

// Disarm: null every slot -> woven kernels take the stock path again.
static inline bool sched_rt_disarm() {
  void *nul = nullptr;
  SCHED_RT_CHECK(cudaMemcpyToSymbol(__sched_task_order, &nul, sizeof(void *)));
  SCHED_RT_CHECK(cudaMemcpyToSymbol(__sched_ctrl, &nul, sizeof(void *)));
  SCHED_RT_CHECK(cudaMemcpyToSymbol(__sched_timer, &nul, sizeof(void *)));
  SCHED_RT_CHECK(cudaMemcpyToSymbol(__sched_queue, &nul, sizeof(void *)));
  // Clear armed so a later sched_rt_init() RE-ARMS (re-points the device
  // slots at the still-live buffers) instead of returning early. Buffers are
  // NOT freed here -- disarm is reversible by construction.
  g_sched_rt.armed = false;
  return true;
}

// pi: install a priority order (order[i] = the task the i-th claim serves).
static inline bool sched_rt_set_order(const int32_t *order, unsigned n) {
  if (n > SCHED_MAX_TASKS)
    n = SCHED_MAX_TASKS;
  SCHED_RT_CHECK(cudaMemcpy(g_sched_rt.d_order, order, n * sizeof(int32_t),
                            cudaMemcpyHostToDevice));
  return true;
}

static inline void sched_rt_set_lambda(float bw, float l2, float smem,
                                       float comp) {
  g_sched_rt.h_ctrl.lambda[0] = bw;
  g_sched_rt.h_ctrl.lambda[1] = l2;
  g_sched_rt.h_ctrl.lambda[2] = smem;
  g_sched_rt.h_ctrl.lambda[3] = comp;
}
static inline void sched_rt_set_task(unsigned i, float q, uint16_t tau,
                                     uint8_t hint) {
  if (i >= SCHED_MAX_TASKS)
    return;
  g_sched_rt.h_ctrl.task[i] = {q, tau, hint, 0};
}
// Shed budget only (keep q/hint): cap request i to `tau` KV units.
static inline void sched_rt_set_tau(unsigned i, uint16_t tau) {
  if (i < SCHED_MAX_TASKS)
    g_sched_rt.h_ctrl.task[i].tau = tau;
}
static inline void sched_rt_set_num_tasks(uint32_t n) {
  g_sched_rt.h_ctrl.num_tasks = n;
}
// Stamp the tile count the installed pi permutation is FOR (takes effect on
// the next push; write the order table first). The weave honors the table
// only when this equals the launch's grid size -- set it with every
// set_order under concurrent launch/rewrite schedulers. 0 = unchecked.
static inline void sched_rt_set_order_size(uint32_t n) {
  g_sched_rt.h_ctrl.order_size = n;
}
// Per-step observation gate (takes effect on the next push): suppress the
// timer's PCIe atomic on non-probe steps. Default (flags=0) is timer ON.
static inline void sched_rt_set_timer_enabled(bool on) {
  if (on)
    g_sched_rt.h_ctrl.flags &= ~SCHED_CTRL_FLAG_TIMER_OFF;
  else
    g_sched_rt.h_ctrl.flags |= SCHED_CTRL_FLAG_TIMER_OFF;
}

// Push the host mirror to the device (one step's policy rewrite).
static inline bool sched_rt_push_ctrl() {
  g_sched_rt.h_ctrl.generation++;
  SCHED_RT_CHECK(cudaMemcpy(g_sched_rt.d_ctrl, &g_sched_rt.h_ctrl,
                            sizeof(SchedCtrl), cudaMemcpyHostToDevice));
  return true;
}

// Prime the ticket counter for a work-queue launch of `workers` CTAs: workers
// pre-claim tickets 0..workers-1 as their ctaid, dynamic claims start there.
static inline bool sched_rt_queue_reset(uint32_t workers) {
  SCHED_RT_CHECK(cudaMemcpy(g_sched_rt.d_queue, &workers, sizeof(uint32_t),
                            cudaMemcpyHostToDevice));
  return true;
}

static inline unsigned long long sched_rt_timer(unsigned i) {
  return (i < SCHED_MAX_TASKS && g_sched_rt.h_timer) ? g_sched_rt.h_timer[i]
                                                     : 0ull;
}
static inline void sched_rt_timer_clear() {
  if (g_sched_rt.h_timer)
    std::memset(g_sched_rt.h_timer, 0,
                SCHED_MAX_TASKS * sizeof(unsigned long long));
}
