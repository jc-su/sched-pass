# Wiring sched-pass into SGLang: userspace, GPU metadata, and the math

2026-07-02. The integration design for running the woven scheduler under a
real serving engine. Answers three questions: what lives in **userspace**,
what **metadata lives on the GPU** (and why that is the point), and what
**math** the loop needs — in dependency order, i.e. what must exist *before*
Lagrangian pricing is even meaningful.

---

## 0. Yes, that is the goal

> "the GPU not static anymore, is this our goal?"

Exactly. Today the GPU side of a serving engine is *static*: the kernel's
behavior is fixed at compile time, identical for every request in every batch;
all adaptivity lives in Python, at batch granularity, between launches. The
woven kernel inverts this: **the kernel becomes an interpreter of GPU-resident
policy tables, and the control plane reprograms the GPU between steps by
writing data** — no recompile, no relaunch of different code, CUDA-graph-safe
(the tables are read at fixed addresses; policy is data, mechanism is code).
Per-request behavior — order, cache treatment, attention budget — changes at
step granularity (ms) while the code stays frozen. That is "the GPU is not
static anymore," and the eKV/CapKV lineage is exactly this move: eKV made GPU
*observation* dynamic, CapKV made GPU *authorization* dynamic, sched-pass
makes GPU *scheduling* dynamic.

## 1. What we weave (the kernel side; already built)

SGLang with the **FlashInfer backend** runs CUDA attention kernels
(`BatchDecodeWithPagedKVCache` family) over a CSR-style μ
(`kv_indptr`/`kv_indices` — grading B = page, exactly the loaded-index form
our detection already covers; the radix tree stays upstream, the kernel reads
the flat arrays — eKV GENERALIZATION.md §3). The build constraint from
README applies: **FlashInfer's kernels must be compiled with clang** (its JIT
invokes nvcc by default; either point its JIT at clang++ + `-fpass-plugin`,
or AOT-build the needed kernels with clang once and let SGLang load them).
That build integration is the first concrete engineering task, and the main
schedule risk (cutlass-heavy templates under clang).

Grid shape note: decode grids are (batch, heads…) with the request on one
axis — `SCHED_SLOT_AXIS` selects it; CLC mode needs no launch change at all
(the native grid already spans all tasks; `try_cancel` steals unlaunched
blocks of it). The ticket/persistent mode is NOT for serving — CLC is.

## 2. Userspace: the control plane inside SGLang

Four small pieces, all attachable without forking SGLang (the eKV lesson:
a `sitecustomize.py` / plugin-entry-point bootstrap reaches the spawned
scheduler process with zero engine edits).

**(a) Identity binding — slot ↔ request, every step.** In SGLang's scheduler,
each forward batch assigns request `rid` to batch index `i` (its position in
`req_pool_indices` / the decode batch). That index IS our task id (what
`ctaid(slotAxis)` reads, what the timer rows key on). The binding changes as
requests join/leave the batch, so userspace maintains, per step:
`bind[i] = (rid, generation)`. Everything downstream (attribution, π, τ) is
keyed through this table; without it every measurement is attributed to the
wrong request the moment the batch mutates. This is eKV's anchor problem and
it is solved the same way: rebuild the binding at batch-formation time, bump
`generation` in SchedCtrl so stale GPU rows are ignored.

**(b) The pre-launch write (arm the step).** At `prepare_for_decode` /
before `model_runner.forward`:
  1. build π: `task_order[i]` from the policy (§4) using per-request
     estimates t̂ — one 4·B-byte H2D copy;
  2. fill SchedCtrl rows: `q[i]` (urgency from SLO slack), `tau[i]` (quality
     budget class), `hint[i]` (urgent/polite from KV length), `lambda`
     (congestion prices), `num_tasks = batch`, `generation++` — one struct
     copy;
  3. order it **before** the forward's kernels on the same stream (or record
     an event) — the CapKV `wait_for_metadata` rule: the kernel must never
     read half-written tables. Cost: two small async copies per step (µs).

**(c) The post-step read (close the loop).** After the step's stream sync
(SGLang already syncs to sample tokens): read the host-mapped timer rows for
live slots, fold into the estimator (§4.1), clear. Zero GPU API calls — the
timer buffer is host-mapped pinned memory (the eKV zero-touch readout).

**(d) The policy tick.** Every step (or every K steps): update t̂ and the
congestion index, recompute π/q/τ/hint for the next step. Pure Python/numpy
on ≤batch-size arrays — microseconds. This component is currently what the
fixtures hand-drive; making it a class (`SchedControlPlane.step()`) is the
"controller artifact" gap named in THEORY.md §7.

## 3. GPU-resident metadata: what and why

What must live ON the GPU is exactly the data the kernel must read at
runtime — anything the kernel needs per-request that the launch cannot carry
(CUDA-graph replay forbids per-launch argument changes; fused kernels have no
per-request arguments at all):

| table | size | written | read by kernel | role |
|---|---|---|---|---|
| `task_order[]` | 4·B | per step | once per CTA | π — who is served when |
| `SchedCtrl.lambda[4]` | 16 B | per tick | once per CTA | prices (σ actions) |
| `SchedCtrl.task[i] = {q, tau, hint}` | 8·B | per step | once per CTA | urgency / budget / cache class |
| `SchedCtrl.{num_tasks, generation}` | 8 B | per step | once per CTA | bounds + staleness guard |
| `__sched_timer[]` (host-mapped) | 8·B | zeroed | atomic add per CTA | the adjoint: per-request cycles OUT |
| `__sched_imp[]` (optional) | 4·K | per tick | per unit (L2-hot) | key-level importance for guard-style φ |

B = max batch, K = KV units. Total: tens of KB — negligible. Everything is
double-buffer-free because of the generation stamp + the fail-safe defaults
(row unset ⇒ neutral ⇒ stock behavior); a late write costs one step of
staleness, never correctness (E0/E1 types; E2 is budget-gated).

That table IS the "GPU not static" mechanism: the kernel reads policy state
that userspace rewrites at ms cadence, while the code (and any captured CUDA
graph) stays frozen.

## 4. The math, in dependency order (what comes before 拉格朗日)

The earlier Lagrangian debate started at the wrong end. Pricing is the LAST
layer, useful only once several resources genuinely couple. What the SGLang
loop needs first, bottom-up — each layer is prerequisite for the one above:

**4.1 Identity & measurement (measure theory of the loop, humble but
load-bearing).** A consistent map (rid ↔ slot ↔ timer row) across steps with
a generation stamp; per-request cycles as an *additive measure* over CTAs/
steps (why the timer is a commutative-monoid sum of same-SM deltas — eKV's
contract). Without this algebra every later number is attributed garbage.
*Status: mechanism built; the binding table is §2(a) work.*

**4.2 Estimation (regression + smoothing, not optimization).** The cost
model the scheduler actually consumes:
    t̂_r = α_f · kv_len_r + β_f   (per kernel family f, fitted online)
    residual_r ← EWMA of (t_r − t̂_r)     (per-request idiosyncrasy)
    γ ← EWMA of step-level congestion (measured vs predicted makespan)
Plus hysteresis/dead-bands so policy flips don't oscillate (the contended-GPU
inversion says the environment is nonstationary; damping is a stability
requirement, not a nicety). This is ordinary statistics — least squares and
exponential smoothing — and it is MOST of the math the loop runs.

**4.3 Ordering (elementary scheduling theory).** Given t̂: π by sorting.
LPT for makespan (Graham's 4/3 bound), SPT/SRPT for mean latency, EDF by
`deadline_r − elapsed_r − t̂_r` for SLOs. Exchange arguments, not convex
optimization — a sort per step. The *objective choice* is the deployment
policy; the table expresses any of them.

**4.4 Budgeting (quality math for φ).** τ from an ε-mass contract: attended
fraction vs accuracy loss (the H2O/Quest empirical bound; must be measured as
NLL-vs-τ on the served model — eKV's Gate-3 methodology). τ is a *quality*
decision, not a performance one; it needs its own guardrails (floor, newest-
pin) before any performance math touches it.

**4.5 Only now, pricing (拉格朗日, in its honest role).** When actions
compete for multiple coupled resources (prefetch buys time with bandwidth;
discard sells L2 residency; τ sells accuracy), the clean way to make the
kernel-side decision local is a price vector λ — the Lagrangian view. But per
the earlier debates and THEORY.md: non-convex, discrete, coupled ⇒ no strong
duality, no online market-clearing. λ's legitimate role here is
**offline/slow-timescale calibration**: fit the score constants
(q·ΔT − λ·ΔR − H) from profiled regressions per kernel family and hardware,
update λ at the §2(d) tick from measured congestion (a PID-with-dead-band,
i.e. dual-ascent-*shaped* feedback, not a solver). Pricing without 4.1–4.2 is
numerology; with them it is a principled scorer whose value is measured.

So the answer to "what math before Lagrangian": **an attribution algebra, a
regression estimator with damping, sorting-based scheduling with classical
bounds, and an ε quality contract — in that order.** The Lagrangian is the
roof, not the foundation.

## 4b. What is BUILT vs BLOCKED (2026-07-02, Blackwell node)

Validated on the RTX PRO 6000 (sm_120), SGLang 0.5.14 + FlashInfer 0.5.3 +
clang-20 / CUDA 12.8:

- ✅ **Baked-address ABI** (`SCHED_BAKE_*` → `inttoptr`): the pass weaves a
  kernel with NO `__sched_*` globals; the Python plane passes its table device
  addresses to the compile. Validated at PTX level and end-to-end.
- ✅ **Python control plane** (`python/sched_rt.py`): owns `order`/`ctrl`/
  `timer` as CUDA tensors, bakes addresses, reprograms by writing data.
- ✅ **End-to-end dynamic loop on REAL online-softmax paged attention**
  (`test/py/test_dynamic_loop.py`, `kernels/paged_softmax.cu`, clang JIT + baked
  ABI): woven+neutral == stock (bit-exact); the woven timer sees the straggler
  (long request ~8× short cycles); π reprogrammed between launches → bit-exact;
  LPT-from-measured-cycles → bit-exact. **This is "the GPU is not static" on
  real attention — the code is frozen, behavior changes by writing data.**
- ✅ **SGLang plugin** (`python/sched_sglang_plugin.py`): registers BEFORE/AFTER
  hooks on `Scheduler.run_batch` via SGLang's real plugin `HookRegistry` (zero
  engine edits). Binds rid↔slot from `req_pool_indices`, builds LPT/urgency π
  from `seq_lens`/last-step cycles, writes the tables, reads the timer.
  Smoke-tested against the live registry: correct LPT order, timer→estimator.
- ✅ **`FLASHINFER_NVCC` clang shim** (`python/nvcc_clang_shim.py`): translates
  FlashInfer's nvcc-style JIT flags to clang CUDA (+`-fpass-plugin`), incl.
  `sm_120a`→`sm_120`. Necessary anyway: this node's `/usr/bin/nvcc` predates
  `compute_120a`, so even stock FlashInfer JIT fails — clang is the only path.
- ✅ **RESOLVED: weaving SGLang's REAL attention kernel (FlashInfer).** The
  FlashInfer batch-decode kernel now JIT-compiles with clang + the LLVM pass
  plugin and runs **bit-identical** to the unwoven baseline (sum 12.9997 both
  ways, `test/py/test_flashinfer_weave.py`); the pass weaves 14 real FlashInfer
  kernels and applies the `task_order` (π) indirection to them, and it detects
  FlashInfer's paged (CSR) KV gather. Five fixes, all reproducible:
  1. `clang_cuda_prelude.h` (force-included) supplies global `__host__
     __device__` `min`/`max` templates — nvcc exposes these as builtins, clang
     doesn't (has only `std::min` and device-only `::min`).
  2. the shim adds `-isystem <cuda>/include` — `/usr/include/cooperative_groups.h`
     symlinked to an OLD CUDA toolkit and shadowed 12.8's, so clang parsed the
     wrong header (the "grid::barrier_arrive not found" wall was this, not a
     real incompatibility).
  3. the shim drops/translates nvcc-only flags (`-Xfatbin=-compress-all`,
     `--threads`, `-use_fast_math`, `-static-global-template-stub`, ...).
  4. the shim maps `sm_120f`→`sm_120a` — CUDA 13 + FlashInfer emit the
     family-specific `sm_120f`, which clang-22 doesn't know, but it accepts the
     arch-conditional `sm_120a` (same Blackwell SM). *(On CUDA 12.8 this needed
     the extra `FLASHINFER_CUDA_ARCH_LIST=12.0f` env to bypass FlashInfer's
     CUDA≥12.9 gate; on the current CUDA 13 toolchain that bypass is gone.)*
  5. `python/patch_flashinfer.py` adds one missing `template` disambiguator in
     `vec_dtypes.cuh` (`::cast<>` → `::template cast<>`) — a standards bug clang
     enforces and nvcc tolerates. Idempotent, one line.
- ✅ **ARMED pi on real FlashInfer (bit-exact). **Cross-process contract now proven** (`test/py/test_fixed_va.py`):
  SchedArena maps the tables at a canonical fixed VA (OS `mmap` +
  `cuMemHostRegister(DEVICEMAP)`; UVA makes devptr == hostptr -- CUDA's own
  fixed-VA reservation is a hint the driver ignores, measured), and
  `bake_env()` keys `FLASHINFER_WORKSPACE_BASE` by the actual base. A second
  process lands at the same VA, reuses the cached .so with NO recompile, and
  the armed permutation stays bit-exact. This was the JIT-cache blocker for a
  real serving run; it is closed.** `test/py/test_flashinfer_arm.py`
  bakes the SchedPlane addresses into FlashInfer's JIT compile and drives a
  task_order PERMUTATION through the real decode kernel: identity vs permuted
  CTA->tile order are BIT-IDENTICAL (sum 12.8487), and the woven clock64 timer
  measured 8 tiles. Slot axis = x: FlashInfer's decode reads
  `batch_idx = request_indices[blockIdx.x]` (decode.cuh:417), so our
  `task_order[blockIdx.x]` composes with FlashInfer's own indirection --
  permuting which CTA serves which tile (E1: every tile done once -> output
  unchanged). This is scheduling the real kernel, not just weaving it.

  Honest remaining boundary: FlashInfer's KV stream uses cp.async/vectorized
  loads whose SCEV stride the pass's constant-stride *site* detection doesn't
  match, so the prefetch/shed levers **decline loudly** on it (reported, never
  mis-woven) while π-indirection + the timer apply. Refining site detection for
  cp.async streams is the next lever-coverage task.
- ⚠️ **Known issue:** the shed lever's score-mask has an open codegen
  interaction with the other levers on the multi-lever JIT softmax kernel
  (pure-arithmetic masking, no OOB by design, but a fault remains under
  `SCHED_RUN_SHED`); the score-mask semantics are gated on the host-app fixture
  (`paged_decode` mode H). The exact levers (π/policy/timer) are unaffected.

## 4c. The armed serving runbook (production wiring, all pieces validated)

`scripts/serve_sglang_armed.sh` launches a real SGLang server with the woven
decode path. The chain, each link gated by `test/run_all.sh`:

1. **Compiler**: `FLASHINFER_NVCC` -> `nvcc_clang_shim.py` -> clang-22 +
   `-fpass-plugin=libSchedPass.so`. Decode kernels weave pi/timer/policy/shed;
   prefill stays on the triton backend (confines the clang-CUDA surface to
   kernels the suite validates).
2. **Bootstrap**: `PYTHONPATH=python/` + `SCHED_SGLANG=1` -> `sitecustomize.py`
   registers `sched_sglang_plugin` inside the spawned scheduler process (zero
   engine edits -- the eKV pattern).
3. **Arming**: the plugin's first batch creates the one-per-process SchedPlane
   (fixed-VA SchedArena, capacity `SCHED_MAX_TASKS`) and calls
   `arm_process_env()`, so every later JIT compile bakes the canonical
   addresses and lands in the va-keyed cache -- reused across restarts
   (`test_fixed_va.py` proves the contract).
4. **Control loop**: BEFORE run_batch -> bind `req_pool_indices`, order by
   measured cycles (LPT when `SCHED_SGLANG_ENFORCE=1`, observe-only default),
   push; AFTER -> read the timer, fold into the estimator
   (`sched_controller.SchedControlPlane` is the reference implementation).

rid<->TILE binding (CLOSED): when FlashInfer's decode plan splits a request's
KV across tiles, the kernel runs `padded_batch_size` tiles with
`batch_idx = request_indices[blockIdx.x]`; the plugin's before-hook now reads
`request_indices` out of the wrapper's int workspace buffer at the byte
offset recorded in `_plan_info` (DecodePlanInfo.ToVector(), 10 int64s --
version-pinned to flashinfer 0.6.x, guarded by the length/offset checks) and
builds pi over TILES; per-request cycles aggregate tile timer rows in the
after-hook. Any layout mismatch falls back soft to one-tile-per-request.

Per-step knobs the plugin now drives (all data writes, no recompile):
  * SCHED_TIMER_EVERY=N -- observation cadence: the timer's PCIe atomic is
    armed 1 step in N via ctrl.flags bit0 (probe steps), off otherwise.
  * SCHED_SGLANG_CLC=off|on|auto + SCHED_CLC_RESID + SCHED_CLC_R -- CLC
    arming: num_tasks>0 arms the woven claim loop, num_tasks=0 takes the
    driver's stock path (static grid + pi -- ordering stays on). auto arms
    only when estimator uncertainty > threshold (default 0.75, calibrated by
    experiments/clc/clc_noise_probe.cu: LPT survives +-50% cost noise, CLC
    pays only under severe order breakdown) AND ntiles > R.
  * SCHED_KV_BYTES_PER_TOKEN -- enables sigma hints only when the batch KV
    working set exceeds L2 (the capacity regime where residency control pays).

## 5. Milestones (each independently measurable)

1. **FlashInfer-under-clang**: build the decode kernel with clang+plugin;
   detection finds the CSR μ; A/B bit-exact unarmed. (Biggest risk first.)
2. **Observe-only in SGLang**: bind identities, read timers; produce the
   per-request residency distribution under a real trace — the straggler
   evidence on a real engine (also: does the timer's PCIe atomic matter at
   serving batch sizes? measure, maybe move to device memory).
3. **π live**: EDF/LPT ordering each step; measure P50/P99 vs baseline on a
   mixed-length trace; CLC on (grid unchanged).
4. **σ live**: hint long requests polite (discard) / urgent pinned; measure
   under KV footprints exceeding L2 (the −54% regime on the serving kernel).
5. **φ live behind SLO pressure**: τ on lagging requests, NLL-vs-τ curve.
6. **The paper's closed-loop figure**: all levers under one trace, ablated.
