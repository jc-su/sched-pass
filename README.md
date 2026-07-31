# sched-pass

An **LLVM new-PM pass plugin** that weaves per-request scheduling capabilities
into CUDA kernels for continuous-batching workloads — the CUDA/AOT successor to
the eKV/CapKV Triton-MLIR pass methodology. Validated end-to-end on **two GPUs**:
RTX A6000 (sm_86, clang-19 + CUDA 12.9) and **RTX PRO 6000 Blackwell**
(`sm_120a`, clang-22/LLVM-22 + CUDA 12.9 — clang-22's max supported CUDA; note
CUDA 13 relocated the `crt/` internal headers clang needs, so 12.9 is the
settled pair) — the latter runs the real **Cluster Launch Control** claim path,
bit-exact on hardware. All fixture modes bit-exact on both.

## What it weaves

| Capability | Mechanism | Cost (fixture, sm_86) |
|---|---|---|
| **Task indirection** (π) | `task = __sched_task_order[ctaid]`; all prior `ctaid` uses remapped. Control plane orders the array by urgency — priority reordering with zero launch changes. | ~0 |
| **Price-guided policy** | `act = q·dT − λ_bw·dR − H > 0` from `__sched_ctrl` (per-task urgency q, shadow price λ); fires a `prefetch.L2` D-iterations ahead in the detected KV-streaming loop, via a branchless `select` on the prefetch address. | ~0 disabled; **−3% runtime** when firing |
| **Feedback timer** | `clock64` bracket, `atomicrmw add` of per-task residency cycles into host-mapped `__sched_timer[task]` (tid==0-gated). | ~16% on this tiny kernel (PCIe atomic per CTA) — optional, `SCHED_NO_TIMER` |
| **Shed** (φ) | `tau` per-task budget caps KV-stream trips, counted by the loop's **canonical induction variable** (never an injected counter). Two drop semantics by kernel structure: linear contraction → mask the loaded **value** to 0; online softmax → mask the **score** to −inf (`exp(−inf−m)=0`, exact truncated attention). Dominance-checked replacement; loops with no canonical IV **decline loudly** (correct-or-absent). `tau=0` ⇒ bit-exact. | validated (tau=0 bit-exact; softmax tau>0 matches a truncated-attention CPU reference exactly) |
| **Work queue** (acquisition layer) | `SCHED_WORKQUEUE=1`: kernel → persistent worker; body outlined to `k.sched_body(args…, task)`. Two claim modes behind one interface (`emitClaim`): **ticket** (`atomicAdd` counter — pre-Blackwell software CLC) and **CLC** (`SCHED_CLC=1`, sm_100+: real `clusterlaunchcontrol.try_cancel` — lowers to `ELECT`/`SYNCS.ARRIVE.TRANS64` SASS). | ticket both GPUs; **CLC bit-exact on real Blackwell** |

Safety discipline (inherited from eKV): every woven effect is a pure read, an
idempotent write, a commutative atomic add, or an architectural hint — CUDA-graph
replay-safe, and **null runtime slots ⇒ bit-exact stock behavior**. (Shed is the
sole exception: it changes results by ε, gated on `tau>0`.)

Ticket vs CLC launch model: the **ticket** queue launches a few persistent
workers (`W ≪ tasks`) that pull tasks off a counter — the regime where dynamic
acquisition wins big (A6000: −32% makespan). **CLC** launches a full grid
(`grid = tasks`) and steals *unlaunched* blocks of that grid for load balancing;
on a big GPU whose scheduler already balances, CLC ties static (correctness is
the guarantee, not a speedup) — see `MODEL.md §8`.

## Layout

```
lib/SchedPlugin.cpp      llvmGetPassPluginInfo (pipeline name: sched-weave)
lib/SchedWeave.cpp       indirection + policy + timer (one module pass)
lib/SchedWorkQueue.cpp   persistent-worker transform (ticket + real CLC claim)
lib/SchedUtil.{h,cpp}    kernel selection, sregs, clock64, runtime slots
include/sched/           public pass API
runtime/sched_rt.h       the runtime contract: __sched_* device globals,
                         host arm/disarm/setters, JSON publication
test/paged_decode.cu     GPU fixture: 5 modes + overhead bench (basic),
                         4 modes (work-queue build)
test/build_and_run.sh    one-shot build + run (cloudsys01 defaults)
```

## Build & run (cloudsys01)

```bash
./test/build_and_run.sh
# or by hand:
cmake -S . -B build -DLLVM_DIR=/usr/lib/llvm-19/lib/cmake/llvm -GNinja && ninja -C build
clang++-19 -x cuda --cuda-gpu-arch=sm_86 --cuda-path=/usr/local/cuda-12.9 -O2 \
  -fpass-plugin=$PWD/build/libSchedPass.so app.cu -lcudart -o app
```

The target app includes `runtime/sched_rt.h` in one TU and calls
`sched_rt_init()`; until then (and after `sched_rt_disarm()`) every woven
kernel runs stock. Compile-time knobs (env at clang time): `SCHED_SLOT_AXIS`,
`SCHED_MAX_TASKS`, `SCHED_NO_{INDIRECT,TIMER,POLICY,SHED}`, `SCHED_PF_DIST`,
`SCHED_DT/DR/H`, `SCHED_WORKQUEUE`, `SCHED_CLC` (Blackwell), `SCHED_WQ_KERNELS`,
`SCHED_DEBUG`.

On Blackwell (LLVM 20 + CUDA 12.8): build the plugin with
`-DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm`, then
`test/blackwell_clc_gate.sh` compiles the CLC work-queue fixture for sm_100 and
sm_120 and shows the woven `clusterlaunchcontrol` PTX + Blackwell CLC SASS.

Testing on plain IR: `opt -load-pass-plugin=build/libSchedPass.so
-passes=sched-weave in.ll -S`.

## One-command validation

```sh
PY=python ./test/run_all.sh     # builds the plugin, gates the whole suite
```
Covers: plugin build, host-ABI fixtures (paged A–H incl. shed, work-queue,
real CLC incl. the `num_tasks=0` per-step disarm switch — stock static path
with π still live), baked-ABI dynamic loop (all levers on the JIT kernel),
the baked timer gate (`ctrl.flags` bit0 suppresses the PCIe atomic per step —
sampled observation), armed-π on real FlashInfer, and the cross-process
fixed-VA JIT-cache contract (`test_fixed_va.py`: two fresh processes, same
canonical VA, cached kernel reused with no recompile, bit-exact).

Work-queue ABIs: ticket mode needs `queue`+`ctrl`; **CLC mode needs `ctrl`
only** (the hardware queue of unlaunched blocks is the queue), so FlashInfer
baked-ABI kernels can weave the CLC claim path. The claim is late-binding by
measured contract (claim-ahead breaks load balancing: −19%→+21%,
`experiments/clc/clc_pipeline_probe.cu`); the serving plugin arms it per step
(`num_tasks>0`) only under high estimator uncertainty
(`experiments/clc/clc_noise_probe.cu` calibrates the threshold).

## Serving-scale evaluation (`test/py/eval_trace.py`)

The makespan claim, measured end-to-end on RTX PRO 6000 Blackwell at serving
scale (8192 decode tiles, 10% long-tail x8 KV, 30 steps/policy, all outputs
bit-exact vs identity):

| within-step order (pi) | step time | vs stock |
|---|---|---|
| stock (unwoven kernel) | 0.603 ms | — |
| woven, identity pi | 0.637 ms | +5.6% (the observation overhead: PCIe timer atomics + table reads) |
| sorted ascending (longs last) | 0.603 ms | ±0% |
| LPT oracle (true lengths) | 0.547 ms | −9.3% |
| **LPT from the woven timer** (closed loop, no oracle) | **0.529 ms** | **−12.3%** |

Net accounting: the closed loop pays its own observation cost and still beats
the untouched stock kernel by 12%.

**Regime map** (same eval, closed-loop `lpt-timer` vs woven identity, 20
steps/policy; timer ranked 100% of true longs at every scale):

| batch (tiles) | 256 | 2048 | 8192 |
|---|---|---|---|
| lpt-timer step-time gain | −13.2% | −10.6% | −16.0% |

The win holds at realistic decode batch sizes, not just at queue-saturating
scale — ordering pays through KV locality and the issue tail even below one
SM wave.

The closed loop recovers most of the oracle gain: the woven `clock64` timer's
per-tile cycles from one probe step rank 100% of the true long requests first.
Sorted orders beat scattered identity even when longs go last — pi acts through
BOTH the queue tail and KV locality (the coupling term in THEORY.md #3).

## Fixture results

```
A6000 (idle, cloudsys02):
  basic:  A inert / B identity / C reversed-π / D aggressive / E priced-out /
          F arithmetic-μ / G shed (tau) — all bit-exact; timer one row/request
  wq:     16 workers cover 64 tasks via ticket claims — bit-exact
  hetero: 64 tasks (8 long, 56 short), 16 workers — the closed loop:
          profile (woven clock64) → LPT π → enforce → verify
            identity 405 → LPT 272 (-32%) → LPT+hints 265 µs/step, all bit-exact
          contended GPU inverts the optimum (segregation wins) — coupling γ

Blackwell RTX PRO 6000 (sm_120, UC-Santa_Cruz-2):
  basic:  all 11 modes bit-exact (incl. shed)
  wq:     ticket — bit-exact
  CLC:    hetero with real clusterlaunchcontrol.try_cancel — BIT-EXACT on
          hardware; SASS shows ELECT / SYNCS.ARRIVE.TRANS64 / MEMBAR.ALL.CTA.
          grid=tasks on 188 SMs: hardware already balances, so CLC≈static
          (64 tasks ±0.4%, 8192 tasks +1.1%) — correctness is the result;
          the makespan win lives in the workers≪tasks regime (see A6000).
```

Detection covers both μ forms at LLVM level: index-based (paged **and** CSR at
any depth — vLLM block_table, SGLang req_to_token/kv_indices) and arithmetic
(contiguous, `base + seq·stride`, a positive no-loaded-index classification).
TMA-carried μ is the documented boundary (launch-arg/trace levels).

**This project is pure CUDA.** The pipeline is: CUDA C++ (FlashAttention-class
kernels) → `clang -fpass-plugin` (the passes run on device LLVM IR) → PTX →
SASS. No Triton anywhere. (Practical note: nvcc cannot load LLVM pass plugins —
its device pipeline is closed — so target kernels must be built with clang's
CUDA support.)

Docs: `THEORY.md` — the unified formal model (one game, three per-request maps
π/σ/φ + observation adjoint, pebbling-in-(max,+) cost, effect-type safety,
regime predictions with hardware confirmations). `MODEL.md` — the systems
model (problem, μ abstraction, lever taxonomy, instrumentation matrix, goal
system, measured results). `DESIGN.md` — heritage notes: how each idiom of the
reference **Triton** passes (eKV/CapKV, which operate on TTIR) was ported to
its LLVM-IR equivalent. That table is a porting recipe from the reference
implementations, not a pipeline — nothing in sched-pass consumes TTIR.
