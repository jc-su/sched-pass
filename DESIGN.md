# sched-pass: per-task resource-policy weaving as an LLVM pass plugin

Design doc, 2026-07-02. **Status: implemented and GPU-validated** (see README.md
and `test/paged_decode.cu`).

> **Scope clarification (read first).** sched-pass is a **pure-CUDA** project:
> clang compiles CUDA C++ (FlashAttention-class kernels), our passes run on the
> device **LLVM IR**, output is PTX/SASS. Triton appears in this document ONLY
> as the home of the two reference implementations (eKV, CapKV — Triton MLIR
> passes on TTIR) whose methodology we ported. Wherever this doc says
> "TTIR idiom → LLVM idiom" it is a **porting recipe from the reference
> passes**, not a compile pipeline; the Triton-integration options in §6.2 are
> historical alternatives that were NOT pursued. Nothing in sched-pass parses,
> consumes, or emits TTIR.

Deviations from the original plan: the clang-CUDA path (§6.3) is the sole
target; the runtime contract uses named `__sched_*` device globals armed via
`cudaMemcpyToSymbol` (runtime/sched_rt.h) instead of dlopen self-allocation
(AOT compile ≠ run process); detection needs no TTIR handshake for any
capability (§5.4's self-contained LLVM detection held up in practice, for both
the index-based and arithmetic μ forms). Source studies: `cloudsys01:~/Dev/eKV/mlir_pass` (eKVTimer.cpp,
eKVPlugin.cpp) and `cloudsys01:~/Dev/CapKV/capkv/mlir_pass/cxx_v2` (MaterializeArgs /
TaintAnalysis / CheckEmission / Plugin). Target: pre-Blackwell GPUs (sm_86/sm_90 — **no
CLC**), so only the non-CLC capabilities from the scheduling model are in scope:

1. **Task indirection** — `task_id = task_order[program_id]`, so the control plane
   chooses which logical request-tile each CTA serves (priority ordering without
   touching the launch site).
2. **Per-task policy** — each CTA reads `policy[task_id]` (urgency `q`, price vector
   `λ`, action hints) at entry and selects one of 2–4 *parameterized* actions
   (prefetch distance, L2 residency priority, optional throttle), i.e. the local
   marginal-exchange rule `act if q·ΔT > λ·ΔR + H`.
3. **Feedback timer** — eKV-style clock64 bracket + per-slot atomic-add so the control
   plane can close the loop (measure the gap the policy is supposed to shrink).

---

## 1. What eKV and CapKV actually do (the reusable pattern)

Both are **out-of-tree Triton MLIR pass plugins** (`tritonGetPluginInfo()` ABI,
Triton PR #8401), loaded via `TRITON_PLUGIN_PATHS` and run inside the JIT pipeline via
`knobs.runtime.add_stages_inspection_hook` — no engine edits, no text round-trips.
Between them they establish a five-part pattern:

| Part | eKV | CapKV |
|---|---|---|
| **Selection** | *Structural* keystone detection: the index-map → KV-gather → contraction triad, found by bounded backward SSA walks (`derivesFrom*`, stop-at-load to pick the leaf indirection level), region-aware via `LoopLikeOpInterface` (iter-arg → init), view-stripping (casts/splat/broadcast), `gatherView` unifying plain loads + TMA descriptors. No names, no config; unrecognized ⇒ **skip loudly**, never guess. | `capkv.auto`: a pointer arg dereferenced through a *loaded* index (the paged-access signature) is protected; explicit per-arg attrs otherwise. |
| **Dataflow analysis** | Backward: recover row form (paged `seq_idx*stride` XOR CSR `kv_indptr[seq]`), stride/seq from the *same* multiply; structural `num_seqs` / `cu_seqlens_q`. | Forward taint from protected args (worklist over def-use, `LoopLikeOpInterface` + `RegionBranchOpInterface` for loops/ifs); backward `traceKey` (both operands of addptr/mul/add, descriptor base, loop edges) to recover the page-id/row key. |
| **State handshake** | Pass **self-allocates** buffers at compile time by `dlopen("libcudart")` inside the running process (host-mapped pinned for capture/timer, device memory for hot-path tables), publishes addresses as JSON lines to `/tmp/ekv_pass_<pid>.json`; external readers (eBPF / proc-mem) consume. Per-compile buffer sets to avoid cross-kernel pollution. | Control plane pins tables at **fixed device addresses**; the pass bakes them into the kernel as `int_to_ptr(const)` — zero-param ABI, launch site untouched. |
| **Weaving** | Additive, data-plane-only edits: clock64 (inline asm) bracket at entry + every `tt.return`, `atomic_rmw ADD` into `ctrl[row]`; masked vector capture of the block-table row; guard = load-mask AND + `-inf` select on softmax scores; tap = post-softmax scatter-add. Head/first-CTA gates against PCIe atomic storms. | One check engine: `emitGrant` evaluates a DNF over gather/key atoms (shared gathers, bounds-masked); apply = AND into the load/store mask, **redirect** `select(ok, page, sentinel)` into the address cone for maskless vectorized reads (perf: +33% → ~0%), `scf.if` for TMA. Unresolved key ⇒ `cap_ok = false` (fail closed). |
| **Safety discipline** | Every woven effect is a commutative-monoid write (atomic add), an idempotent write, or a pure read ⇒ **CUDA-graph-replay safe**. Defaults are stock behavior (importance=max, tau=0 ⇒ keep-all). | Fail-closed denial; PTX verifier as an independent soundness gate; stores never redirected (covert channel). |

The two complementary admission stances matter for sched-pass: eKV is *fail-open*
(observability must never change results; defaults = stock), CapKV is *fail-closed*
(security must never leak). A **scheduling policy is an eKV-style capability**: the
woven code must be bit-exact w.r.t. results and default to stock behavior
(`task_order[i] = i`, `policy = neutral`) until the control plane writes.

---

## 2. Where an LLVM pass fits (and why bother vs. staying in TTIR)

Pipeline positions:

```
Triton:  Python AST → TTIR → TTGIR → LLVM IR (NVVM) → PTX → cubin
                       ^eKV/CapKV live here          ^sched-pass (LLVM) lives here
CUDA C++ (clang):  AST → LLVM IR (NVVM) → PTX → cubin        (-fpass-plugin)
```

**What you gain at the LLVM level**
- *Frontend generality*: the same .so instruments Triton kernels, clang-compiled CUDA
  (FlashInfer/CUTLASS-style kernels), and anything else that produces NVVM IR. TTIR
  passes only ever see Triton.
- *Post-layout truth*: you weave after Triton's layout/pipelining passes, so nothing
  downstream can move, fuse, or re-layout your instrumentation; PTX-level actions
  (prefetch, `applypriority`) map 1:1.
- The pass-manager mechanics you already know transfer directly: the new PM is
  concept-based (`PassInfoMixin<T>` + `PreservedAnalyses run(Module&, ...)`), the
  plugin ABI (`llvmGetPassPluginInfo`) is the exact analogue of
  `tritonGetPluginInfo`.

**What you lose**
- *Tensor semantics*. TTIR's `tt.load` is a whole-CTA op on a tensor of pointers — a
  gather is syntactically visible, masks are first-class, and CTA-uniformity is free.
  LLVM NVVM IR is **per-thread scalar/short-vector** code after layout assignment:
  gathers are unrolled lane loads, loops are CFG+PHI, contractions are
  `llvm.nvvm.mma.*` intrinsics or FMA chains. eKV's keystone triad is re-derivable but
  substantially harder and more brittle here.
- *Block-level ops*. Anything "per CTA once" must be explicitly gated
  (`tid==0` predicate) — at TTIR that came for free.

**Decision: hybrid, with the LLVM pass carrying the mechanical weaving.**
Structural *detection* (which kernel is attention; which axis is the request slot;
where the KV streaming loop is) stays cheap at TTIR — eKV already does it. The
*capabilities* (indirection, policy, timer, actions) are woven at LLVM IR, where they
are frontend-agnostic. Handshake between the two: function-level markers.
Concretely, three workable coordination channels, in order of preference:

1. **Self-contained LLVM detection** (phase 1 targets it): the slot axis and the
   streaming loop are recoverable at LLVM level with modest analysis (see §5); no TTIR
   help needed for the minimal capability set.
2. **Named-symbol handshake**: a TTIR pre-pass (10 lines in the existing eKV plugin)
   renames/annotates the kernel (e.g. appends `.__sched<axis>`) — names survive
   lowering to the LLVM function name.
3. **JSON side-channel** keyed by kernel name, exactly like `/tmp/ekv_pass_<pid>.json`.

---

## 3. Plugin skeleton (new pass manager)

The LLVM analogue of `eKVPlugin.cpp`. One .so, three passes, registered both for
`opt -passes=...` (testing) and pipeline extension points (production).

```cpp
//===- SchedPlugin.cpp - sched-pass LLVM plugin entry ---------------------===//
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

// NOTE: LLVM ≥ ~22 renames the mix-ins to OptionalPassInfoMixin /
// RequiredPassInfoMixin (see "Writing an LLVM Pass"); on ≤ 21 use PassInfoMixin.
// The run() signature is identical either way.
class SchedTaskIndirectionPass : public PassInfoMixin<SchedTaskIndirectionPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};
class SchedPolicyPass : public PassInfoMixin<SchedPolicyPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};
class SchedTimerPass : public PassInfoMixin<SchedTimerPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "sched-pass", "0.1.0",
          [](PassBuilder &PB) {
            // opt -load-pass-plugin=libSchedPass.so -passes='sched-weave'
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "sched-weave") {
                    MPM.addPass(SchedTaskIndirectionPass());
                    MPM.addPass(SchedPolicyPass());
                    MPM.addPass(SchedTimerPass());
                    return true;
                  }
                  return false;
                });
            // clang -fpass-plugin=libSchedPass.so : run automatically at the
            // end of the optimization pipeline (post-vectorizer, pre-codegen).
            PB.registerOptimizerLastEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel, ThinOrFullLTOPhase) {
                  MPM.addPass(SchedTaskIndirectionPass());
                  MPM.addPass(SchedPolicyPass());
                  MPM.addPass(SchedTimerPass());
                });
          }};
}
```

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(sched-pass CXX)
find_package(LLVM REQUIRED CONFIG)   # match the LLVM that will load the plugin
add_library(SchedPass MODULE lib/SchedPlugin.cpp lib/SchedTaskIndirection.cpp
            lib/SchedPolicy.cpp lib/SchedTimer.cpp lib/SchedRuntime.cpp)
target_include_directories(SchedPass PRIVATE ${LLVM_INCLUDE_DIRS} include)
target_compile_features(SchedPass PRIVATE cxx_std_17)
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
target_compile_definitions(SchedPass PRIVATE ${LLVM_DEFINITIONS_LIST})
# no llvm libs linked: MODULE plugins resolve symbols from the host (opt/clang)
```

Config mirrors eKV: env vars (`SCHED_SLOT_AXIS`, `SCHED_MAX_TASKS`, `SCHED_ACTIONS`,
`SCHED_TIMER`, `SCHED_DEBUG`, `SCHED_DETECT_ONLY`) — no per-kernel Python installer.

---

## 4. Mechanism mapping: TTIR idiom → LLVM idiom

Everything eKV/CapKV do has a direct LLVM equivalent; the table is the porting recipe.

| TTIR (eKV/CapKV) | LLVM NVVM equivalent | Notes |
|---|---|---|
| plugin entry `tritonGetPluginInfo` | `llvmGetPassPluginInfo` | same shape: name/version/register-callbacks |
| pass = `PassWrapper<T, OperationPass<ModuleOp>>` | `PassInfoMixin<T>` + `run(Module&, ModuleAnalysisManager&)` | concept-based, no inheritance interface |
| select kernels: walk `tt.FuncOp` | `for (Function &F : M)` where F has calling conv `ptx_kernel` / is listed in `!nvvm.annotations` with `"kernel"` | helper `isKernel(F)` |
| `tt.get_program_id(axis)` | `@llvm.nvvm.read.ptx.sreg.ctaid.{x,y,z}` | CTA-uniform value |
| — (free at TTIR: ops are per-CTA) | gate per-CTA effects on `tid.{x,y,z}==0` (`@llvm.nvvm.read.ptx.sreg.tid.*`) | **the** key semantic shift: LLVM IR is per-thread |
| `ElementwiseInlineAsmOp "mov.u64 $0, %clock64;"` | `call i64 asm sideeffect "mov.u64 $0, %clock64;", "=l"()` | identical string; (`llvm.readcyclecounter` also lowers to `%clock64` on NVPTX but the asm form is what eKV validated) |
| baked address: `arith.constant` + `tt.int_to_ptr` | `inttoptr (i64 <addr> to ptr addrspace(1))` constant expr | global address space = 1 |
| `tt.atomic_rmw ADD, RELAXED, gpu-scope` | `atomicrmw add ptr addrspace(1) %p, i64 %v monotonic, syncscope("device")` | same commutative-monoid admission rule |
| `tt.atomic_rmw XCHG` (idempotent publish) | `atomicrmw xchg ... monotonic` | gate to one thread of one CTA |
| `tt.load` mask AND (guard) | no first-class mask on scalar loads ⇒ use **CapKV's redirect**: `select i1 %ok, %addr, %sentinel_addr` feeding the load, or branch-predicate | redirect is *the* natural LLVM form; it was CapKV's fast path anyway |
| `scf.for` iter-args, `LoopLikeOpInterface` | `LoopInfo` + PHI nodes; `ScalarEvolution` for `base + i*stride` recognition | region-aware walk → PHI-aware walk |
| forward taint over def-use | identical worklist over `Value::users()`, crossing PHIs | simpler: no region interfaces needed |
| backward `traceKey` through addptr/mul/add/casts | walk through `getelementptr`/`add`/`mul`/`sext`/`zext`/`inttoptr` | `stripView` ≈ peeling casts |
| self-alloc via `dlopen("libcudart")` in-pass | **same code verbatim** when the pass runs inside a JIT process (Triton); for clang AOT the pass instead emits a call to a tiny runtime shim (`libsched_rt.so`, constructor-allocated) or reads addresses from env | `SchedRuntime.cpp` shared with eKV's mapAlloc/devAlloc |
| publish `/tmp/ekv_pass_<pid>.json` | identical (`/tmp/sched_pass_<pid>.json`) | one line per instrumented kernel |
| idempotency attr `ekv.instrumented` | named metadata `!sched.instrumented` on the function, or a module flag | prevents double weaving on recompiles |

---

## 5. The three passes in detail

### 5.1 SchedTaskIndirectionPass — `task_id = task_order[ctaid]`

*What it does.* In each selected kernel, find the calls to
`llvm.nvvm.read.ptx.sreg.ctaid.<slotAxis>`; replace their **uses** with a value loaded
from the control plane's mapping table:

```llvm
; before
%pid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
; ... %pid used as the request/tile index everywhere ...

; after
%pid   = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
%inb   = icmp ult i32 %pid, <MAX_TASKS>              ; row guard (eKV lesson)
%idx   = select i1 %inb, i32 %pid, i32 0
%gep   = getelementptr i32, ptr addrspace(1) inttoptr (i64 <TASK_ORDER> to ptr addrspace(1)), i32 %idx
%tid_r = load i32, ptr addrspace(1) %gep, align 4, !invariant.load !0
%task  = select i1 %inb, i32 %tid_r, i32 %pid        ; OOB ⇒ identity (fail-safe)
; replaceAllUsesWith(%pid → %task)  [except inside the weave itself]
```

Costs: one CTA-uniform 4-byte load per CTA, served from L2/L1 after the first CTA
touches the line. `!invariant.load` is legal *within one kernel execution* (the table
is written only between launches) and lets LLVM hoist/CSE it.

Fail-safe defaults (eKV's "stock behavior by construction"): the control plane
initializes `task_order[i] = i`; the OOB arm falls back to the raw pid; if the table
was never allocated the pass simply doesn't weave (loud skip).

Graph-replay admission: pure read of a fixed-address buffer — data-driven, not
control-flow-at-launch, so CUDA-graph legal (same argument as eKV's guard tables).

*Which ctaid axis*: `SCHED_SLOT_AXIS` (default x), same contract as `EKV_SLOT_AXIS`.
Only that axis is remapped; head/other axes untouched.

*Honest boundary* (inherited from eKV): this is meaningful where CTA ≈ request tile —
decode-shaped, grid-mapped kernels. Persistent kernels and prefill q-block grids need
the axis identified per kernel family (channel 2/3 in §2), or are skipped loudly.

### 5.2 SchedPolicyPass — read λ + task meta, score 2–4 actions, apply parameters

Layout of the control block (device memory, *not* host-mapped — hot-path reads must
not cross PCIe; eKV's guard-table lesson):

```c
struct SchedCtrl {            // one per instrumented kernel, baked base address
  u32   generation;           // bumped by control plane each rewrite
  f32   lambda[4];            // price vector: {bw, l2, smem, comp}
  // per-task rows, indexed by task_id:
  struct { f32 q;             // urgency (deadline-slack derived, clipped)
           u16 len_bucket;    // work-size class
           u8  hint;          // control-plane action hint (advisory)
           u8  _pad; } task[MAX_TASKS];
};
```

Woven at kernel entry (all CTA-uniform, so every branch below is warp-uniform —
no divergence):

```llvm
%q      = load f32, ... task[%task].q
%lam_bw = load f32, ... lambda[0]        ; etc.
; score_k = q * dT_k - dot(lambda, dR_k) - H_k     (dT_k, dR_k, H_k: compile-time
;                                                   constants per action, from
;                                                   offline profiling / env)
; action = argmax_k score_k, floor at 0 ⇒ baseline
```

With 2–4 actions this is a handful of FMAs and selects — the "small scoring function,
never an optimizer in the kernel" rule from the model discussion.

**Applying the action — additive, parameterized, never restructuring.** The eKV
admission discipline (additive dataflow weaving only) restricts which "actions" are
implementable safely, and that's a feature. Three that fit:

1. **Prefetch distance** (the flagship). Find the KV streaming loop (§5.4); inside it,
   insert a predicated PTX prefetch on the address the loop will touch `D` iterations
   ahead:
   ```llvm
   br i1 %act_aggr, label %pf, label %cont     ; uniform branch
   pf:
     %pfaddr = getelementptr ... (%i + D)
     call void asm sideeffect "prefetch.global.L2 [$0];", "l"(ptr addrspace(1) %pfaddr)
     br label %cont
   ```
   Pure additive (no existing load touched), bit-exact, replay-safe (prefetch is a
   hint, architecturally a no-op on results).
2. **L2 residency priority**: `applypriority.global.L2::evict_{first,normal,last}`
   inline asm on the task's KV region base — the "resource yielding" action: a relaxed
   task marks its KV lines evict-first, ceding L2 to urgent tasks. Also additive.
3. **Throttle** (optional, most aggressive): a `nanosleep.u32` backoff inserted at the
   top of the streaming loop for `act_yield` CTAs, directly ceding memory-issue slots.
   Bit-exact but *does* change timing of the CTA itself — this is the point; keep it
   behind `SCHED_ACTIONS=throttle`.

What we deliberately do **not** do at this level: rewriting existing loads' cache
modifiers (`ld.global.cg` etc.). It requires replacing loads with inline asm — a
restructuring edit that breaks the additive admission rule and fights instruction
selection. If cache-modifier control proves necessary, do it as a TTIR pass on
`tt.load`'s first-class cache-modifier attribute instead (right level for it).

### 5.3 SchedTimerPass — the feedback loop (eKV timer, ported)

Verbatim port of `emitTimer`:

- entry: `%t0 = clock64` (thread 0 of the CTA suffices; the value is only used by
  thread 0's atomic);
- at **every** `ret` instruction: `%t1 = clock64`, `%dur = sub`,
  `atomicrmw add` into `u64 timer[task_id]` — predicated on
  `tid==0 && ctaid.<headAxis>==0` (the PCIe-atomic-storm gate, now explicit because
  LLVM IR is per-thread);
- row guard `task_id < MAX_TASKS` folded into the predicate;
- timer buffer host-mapped (read by the external observer post-sync), allocated by the
  same `mapAlloc` path.

This closes the loop the model needs: `consumption[task]` vs `wall_kernel` is the
"hijacking gap" figure, and it is also the calibration source for the per-action
`dT_k/dR_k` constants.

### 5.4 Detection at LLVM level (what replaces the keystone triad)

Phase 1 needs only two structural facts, both recoverable without tensor semantics:

- **Selection** (which kernels to weave): kernel linkage + a *paged-access signature*
  in LLVM terms — a load whose pointer operand's SCEV/def-chain passes through the
  result of another load (CapKV's `hasLoadedIndex`, ported to `getelementptr`
  chains). That is exactly the "address depends on loaded data" test and survives
  lowering intact.
- **The streaming loop** (for prefetch placement): the deepest `LoopInfo` loop whose
  body contains those dependent loads, with the address an affine
  `{base, +, stride}` SCEV — insert the prefetch on `base + (iv + D)*stride`.

The full eKV triad (contraction test etc.) is *not* required for correctness because
every woven effect is fail-safe; it's an over-selection filter. If foreign kernels
(layernorms, GEMVs) match the paged signature, they get a policy block that never
fires (neutral defaults) — noisy but harmless; tighten with the name/JSON handshake
(§2, channel 2/3) rather than re-deriving the triad in scalar IR.

`SCHED_DETECT_ONLY=1` mirrors `EKV_DETECT_ONLY`: print selection + loop + slot-axis
findings, weave nothing. Build this first; it is the debugging backbone.

---

## 6. How the plugin gets into each pipeline

1. **Offline / testing (first target).**
   ```bash
   TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=./dump python bench_attn.py   # dumps .llir
   opt -load-pass-plugin=./libSchedPass.so -passes=sched-weave \
       dump/<hash>/kernel.llir -S -o kernel.sched.ll
   llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_86 kernel.sched.ll -o kernel.ptx
   ```
   Iterate on IR correctness with FileCheck tests on small .ll fixtures (the
   `tests/ekv_timer.mlir` analogue).

2. **Triton JIT (production path).** Triton links its own LLVM statically and has no
   LLVM-pass plugin hook; two working options:
   - **Override mechanism**: `TRITON_KERNEL_OVERRIDE=1` + `TRITON_OVERRIDE_DIR` — dump
     `llir`, transform with `opt` (matching LLVM major), drop into the override dir,
     recompile. Zero Triton changes; good for experiments on real serving kernels.
   - **Stage hook**: the same `add_stages_inspection_hook` used by eKV/CapKV also
     exposes the `llir` stage; a small Python shim writes the module out, shells to
     `opt`, and swaps the artifact. Text round-trip, but at the *last* stage, where
     eKV's "no text round-trip" concern (MLIR verifier, attrs) doesn't bite.
   - If neither is acceptable long-term, the fallback is to keep this weaving in a
     TTIR plugin (eKV infra) and accept Triton-only scope — the LLVM plugin still
     pays for itself via path 3.

3. **clang CUDA AOT**: `clang++ -x cuda --cuda-gpu-arch=sm_86
   -fpass-plugin=./libSchedPass.so ...` — device IR flows through the registered
   `OptimizerLastEP` callback. This is what makes the capability frontend-agnostic
   (vLLM's C++/CUDA kernels, FlashInfer). Here the pass cannot dlopen-allocate
   (compile ≠ run process), so it emits references to `__sched_ctrl_<kernel>` global
   symbols that `libsched_rt.so` (LD_PRELOAD or linked) resolves and allocates at
   load, publishing the same JSON.

---

## 7. Control-plane contract (unchanged from the model discussion)

- Control plane (in-process hook or external daemon) writes, once per step:
  `task_order[]` (priority-bucketed: urgent tiles first), `lambda[4]`, per-task
  `{q, len_bucket, hint}`. All writes are between-launch data writes to fixed
  addresses ⇒ CUDA-graph-replay legal (the mapped-memory reverse direction).
- Advisory semantics: kernel-side scoring treats `hint` as a permission, not a
  command; the guardrail is the score floor (all scores ≤ 0 ⇒ baseline).
- Neutral state = stock kernel: identity `task_order`, `q = 0`, `λ = 0` ⇒ every
  score is `-H_k < 0` ⇒ baseline action, only overheads are the entry loads.
- The timer buffer flows the other way (GPU → host-mapped), read after stream sync.

## 8. Overhead budget & ablation plan (what to measure before believing anything)

Per-CTA added work, target envelope:
- 1 × u32 load (task_order) + ~24B policy row load + ~8 FMA/select — entry only;
- prefetch asm: 1 predicated instruction per loop iteration (aggressive arm only);
- timer: 2 × clock64 + 1 × PCIe atomic per CTA, gated to one thread/head-plane.

Ablation ladder (each step vs. the previous, decode-shaped mixed batch):
1. stock → +indirection (identity table): must be ≈ 0;
2. +policy loads, neutral λ: must be ≈ 0;
3. +timer: quantifies observation cost (eKV numbers are the prior);
4. +policy active, uniform batch: should be ≈ 0 (scores floor to baseline);
5. +policy active, mixed short/long batch: the payoff row — short-request p99 and
   the consumption-vs-wait gap must move.

Go/no-go mirrors the research plan: if (5) shows no gap reduction on a strong
baseline (chunked prefill assumptions, decode-only), the enforcement story needs
rework before more machinery is added.

## 9. Phasing

1. **P0 — skeleton + detect-only** (plugin loads in `opt`, selects kernels, finds
   slot axis + streaming loop, prints; FileCheck tests on dumped Triton llir).
2. **P1 — SchedTimerPass** (port of eKV emitTimer; validates the whole
   alloc/publish/weave/read loop at LLVM level with a capability that's already
   proven at TTIR level — differential-test against eKV's numbers on the same kernel).
3. **P2 — TaskIndirection** (+ identity-table ablation).
4. **P3 — Policy** (prefetch action first, then applypriority; scoring constants from
   offline profiling).
5. **P4 — clang CUDA path** (runtime shim, one hand-written paged-attention .cu as
   the fixture).
6. **P5 (optional, later)** — software work-stealing (persistent-kernel transform:
   wrap body in a ticket loop over `atomicrmw add` on a queue head). This is the
   pre-Blackwell CLC substitute but it changes grid semantics and needs launch-site
   cooperation — a deliberate break from the zero-touch discipline; only attempt once
   P2/P3 prove value. On Blackwell hardware, this slot is where CLC would plug in.
