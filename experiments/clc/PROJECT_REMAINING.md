# Project-Specific Remaining Investigation

Date: 2026-07-03.
Machine: NVIDIA RTX PRO 6000 Blackwell Server Edition, `sm_120`.

This note covers what remains after the standalone CLC black-box study. The
CLC primitive is understood well enough; the remaining questions are about the
actual sched-pass / FlashInfer / SGLang integration.

## Status

Broad CLC research is no longer the blocker. The remaining blocker is
implementation-specific:

```text
Host-app CLC fixture: works and runs on Blackwell.
FlashInfer baked-ABI path: task_order/timer work, but CLC work-queue transform
                           is currently skipped.
```

The reason is mechanical. `SchedWorkQueuePass` currently requires
`runtime/sched_rt.h` globals:

```text
__sched_queue
__sched_ctrl
```

FlashInfer JIT kernels use the baked-address ABI instead and do not declare
those globals. A compile check with:

```bash
SCHED_WORKQUEUE=1 SCHED_CLC=1 SCHED_DEBUG=1 \
SCHED_BAKE_TASK_ORDER=1 SCHED_BAKE_CTRL=2 \
SCHED_BAKE_TIMER=3 SCHED_BAKE_QUEUE=4 \
clang++-22 ... python/kernels/paged_softmax.cu
```

reported:

```text
paged_softmax: no __sched_queue/__sched_ctrl in module
(TU does not include sched_rt.h) -> work-queue transform SKIPPED loudly
```

So, today, real FlashInfer tests prove live `task_order[]`, policy/timer, and
fixed-VA cache correctness, but not FlashInfer CLC claiming.

## Validation Run

Project validation gate:

```text
./test/run_all.sh
```

with local Blackwell defaults:

```text
LLVM_DIR=/usr/lib/llvm-22/lib/cmake/llvm
CLANG=clang++-22
CUDA=/usr/local/cuda-12.9
ARCH=sm_120a
```

Result:

```text
10 passed, 0 failed
```

Passed gates:

```text
cmake+ninja plugin
paged_decode fixture
work-queue ticket claim
CLC try_cancel host-app fixture
dynamic loop baked ABI
armed pi on FlashInfer decode
cross-process JIT-cache contract
```

## Real FlashInfer / Baked ABI

`python/test_flashinfer_arm.py` passed:

```text
pi permutation on real FlashInfer decode is BIT-EXACT
woven timer populated on real FlashInfer
```

`python/test_fixed_va.py` passed:

```text
both processes obtained canonical VA
table addresses identical across processes
armed pi permutation bit-exact in both processes
phase 2 reused cached kernels unmodified
```

This proves that, for real FlashInfer decode:

```text
raw CTA -> task_order[raw] -> FlashInfer request/tile
```

is live and correct.

## Larger CLC Host-App Fixture

The default CLC fixture uses `NSEQ=64`, which is below the local CLC threshold
`R=2256`, so it cannot prove suffix claiming.

A larger host-app CLC fixture was compiled without changing source:

```bash
SCHED_WORKQUEUE=1 SCHED_CLC=1 clang++-22 ... \
  -DSCHED_FIXTURE_CLC=1 -DNSEQ=4096 -DLONGB=8 \
  test/hetero_batch.cu -o build/hclc_n4096_l8
```

Result:

```text
all schedules bit-exact
dynamic (CLC) vs static: -10.3%
```

Interpretation:

- Host-app CLC correctness works with `N > R`.
- In this memory-heavy fixture, CLC overhead loses to static by about 10%.
- This is not a CLC no-go; it means CLC should be enabled only in regimes where
  tail imbalance is large enough to amortize claim overhead.

## Decode-Shaped Task Ordering

`python/eval_trace.py` was run with real baked-ABI task ordering:

```text
NSEQ=4096, STEPS=10:
  identity      0.500 ms
  lpt-oracle    0.443 ms  (-11.5%)
  lpt-timer     0.439 ms  (-12.2%)
  exact outputs yes

NSEQ=8192, STEPS=10:
  identity      0.708 ms
  lpt-oracle    0.558 ms  (-21.2%)
  lpt-timer     0.533 ms  (-24.7%)
  exact outputs yes
```

The timer-derived order ranked true long requests first:

```text
top long-count of measured order = 100% true long requests
```

This is strong evidence that `task_order[]` is useful before CLC is wired into
the FlashInfer baked path.

## Metadata Cost

Isolated baked-ABI overhead on a small uniform `NSEQ=8192` paged-softmax kernel:

```text
stock                         0.155760 ms
indirect_only                 0.169585 ms   +8.88%
indirect_policy_no_timer      0.212329 ms  +36.32%
indirect_shed_no_timer        0.171097 ms   +9.85%
full_no_timer                 0.221028 ms  +41.90%
full_with_timer               0.437706 ms +181.01%
```

Interpretation:

- The compiler/pass itself is not the cost: a no-indirect empty weave measured
  about `+0.29%` in a separate run.
- `task_order[]` alone is a small per-CTA metadata read, but still visible on a
  very short kernel.
- The host-mapped timer is too expensive for every serving step. Use it for
  probe/adaptation steps, sample it, or disable it in steady state.
- Policy loads/prefetch logic are also visible on tiny kernels. Enable policy
  only when the expected cache/bandwidth benefit exceeds the control overhead.

## Remaining Work

### DONE (2026-07-03): WorkQueue baked ABI + arming/gating mechanics

`SchedWorkQueuePass` now supports both ABIs (`rtAvailable`/`rtBuffer` like
SchedWeave), and the semantics were tightened for production:

```text
CLC mode (SCHED_CLC=1, sm_100+): requires ctrl ONLY -- the hardware queue of
  unlaunched blocks IS the queue; SCHED_BAKE_QUEUE is unnecessary.
Ticket mode: requires queue + ctrl (unchanged).
Verified: baked compile with only ORDER/CTRL/TIMER baked reports
  "persistent-worker transform applied (CLC claim, ctrl-only)" and the PTX
  contains the clusterlaunchcontrol try_cancel/is_canceled/get_first_ctaid
  sequence. Ticket-without-queue still SKIPS loudly.

num_tasks == 0 -> the driver takes the STOCK path (static grid + pi remap).
  This is (a) the baked-ABI fail-safe: baked slots are constants, the null
  check cannot fire, and an unprogrammed zeroed arena must not eat the
  launch; and (b) the PER-STEP ARMING SWITCH: the control plane toggles the
  claim loop by writing num_tasks (N = on, 0 = off), no recompile, and in
  CLC mode no launch change (grid == tasks either way).
  NOTE the stock path now applies the pi remap when the order table is armed:
  "claim loop off" must not also turn ordering off (static+LPT is the best
  schedule under a good estimate; the claim loop is the uncertainty hedge).

ctrl->flags bit0 (was reserved sentinel_key; offset 24 pinned) = TIMER OFF:
  per-step observation gating for the baked ABI (which cannot null its
  slots). Zero default = timer on = historical behavior. SchedPlane
  .set_timer_enabled() / sched_rt_set_timer_enabled().

Late-binding contract codified in emitClaim(): the claim is issued strictly
  AFTER the body. Claim-ahead (prologue issue, the tempting async-overlap
  pattern) reserves a task while the worker is busy -> head-of-line blocking;
  measured -19% -> +21% flip (experiments/clc/clc_pipeline_probe.cu).
```

The arming POLICY (when to write num_tasks > 0) is uncertainty-gated:
pi is open-loop and needs the cost estimate to be right; late binding does
not. `clc_noise_probe.cu` measures static+pi(eps) vs CLC+pi(eps) as the
estimate degrades -- the crossing eps calibrates SCHED_CLC_RESID in
`sched_sglang_plugin.py` (which arms only when estimator uncertainty exceeds
it AND ntiles > R).

### DONE (2026-07-03, second pass): FlashInfer CLC runtime + R + soundness

`python/test_flashinfer_clc.py` (in `run_all.sh`, 16/16 green) proves on the
REAL BatchDecode kernel: 1D armed identity / reversed-pi / `num_tasks=0`
disarm all bit-exact vs the pi-only golden; the cached .so SASS contains
`UGETNEXTWORKID`; and the JIT cache key now carries the WQ/CLC mode tag
(`va...-nN-wqclc`), so a pi-only kernel can never be silently reused for a
CLC compile.

Two soundness items found and fixed in this pass:

```text
GRID-SHAPE GUARD: tickets and the CLC decode enumerate only the slot axis,
  but FlashInfer decode launches grid = (padded_batch, num_kv_heads) -- 2D.
  A claimed block's task would run under the WRONG y. The driver now gates
  the dynamic path on nctaid.{other axes} == 1 at runtime; any other shape
  takes the stock path (static + pi). Verified bit-exact on the real
  num_kv_heads=8 launch. Multi-axis claiming (linearize + v4 decode) is
  future work.

R FOR THE REAL KERNEL: SchedPlane.r_for_cached_so extracts the cubin
  (cuobjdump -xelf) and queries cuOccupancyMaxActiveBlocksPerMultiprocessor
  via the DRIVER API (the runtime-API stub path is unreliable in a torch
  process: two cudart instances, fatbin registered with only one ->
  cudaErrorInvalidResourceHandle). Measured: the woven BatchDecode kernel
  runs 3..5 blocks/SM -> R = 564..940 on the 188-SM sm_120 -- far below the
  light-probe 2256. Typical decode batches are N <= R for THIS kernel: the
  arming gate's N > R condition correctly keeps CLC off in the common case;
  it can arm on large split-kv tile counts or smaller-R variants.
```

### Still remaining

1. Serving-trace eval (the ship/no-ship gate): TPOT p50/p99 + throughput
   under a production-like length mix on a live SGLang server
   (`SCHED_SGLANG_ENFORCE=1`, `SCHED_TIMER_EVERY` sampling, CLC auto-arm).
   sglang 0.5.14 is installed on this node; needs model weights + a load
   generator.
2. sm_100 (GB200/B200) validation pass: R recomputes from the occupancy API
   by construction, but the tie/win boundaries and the noise-probe crossing
   should be re-measured once on that silicon.

Recommended production policy (unchanged in spirit, now mechanized):

```text
Lead with task_order[] (static+pi) -- it wins when the estimate is good.
Sample the timer (SCHED_TIMER_EVERY), do not pay it every step.
Arm CLC per step only under high estimator uncertainty AND ntiles > R.
```
