//===- SchedUtil.h - shared helpers for the sched-pass plugin --*- C++ -*-===//
//
// The LLVM-level analogues of eKV/CapKV's TTIR idioms:
//   * kernel selection        (ptx_kernel CC or nvvm.annotations "kernel")
//   * special-register reads  (ctaid/tid via llvm.nvvm.read.ptx.sreg.*)
//   * clock64                 (inline asm, the exact string eKV validated)
//   * the runtime contract    (named __sched_* device globals holding pointers
//                              to control-plane buffers; null == stock behavior)
//
// The runtime contract replaces eKV's baked-address zero-param ABI: an AOT
// clang compile cannot dlopen-allocate (compile process != run process), so
// the pass weaves against named device globals that runtime/sched_rt.h
// defines and the host fills via cudaMemcpyToSymbol. Null pointer == the
// woven code takes the stock path (fail-safe by construction, the eKV rule).
//
//===----------------------------------------------------------------------===//
#ifndef SCHED_UTIL_H
#define SCHED_UTIL_H

#include "llvm/Config/llvm-config.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"

#include <cstdint>

namespace sched {

// LLVM 19/20 compat: Intrinsic::getDeclaration was renamed.
inline llvm::Function *intrinsicDecl(llvm::Module *M, llvm::Intrinsic::ID ID) {
#if LLVM_VERSION_MAJOR >= 20
  return llvm::Intrinsic::getOrInsertDeclaration(M, ID);
#else
  return llvm::Intrinsic::getDeclaration(M, ID);
#endif
}

// Compile-time configuration, read from the environment (the pass runs inside
// the clang process, exactly like eKV's EKV_* env config inside the JIT).
struct Config {
  int slotAxis = 0;            // SCHED_SLOT_AXIS   0=x 1=y 2=z (request axis)
  unsigned maxTasks = 4096;    // SCHED_MAX_TASKS   rows in every table (ABI)
  bool indirect = true;        // SCHED_NO_INDIRECT disables task_order remap
  bool timer = true;           // SCHED_NO_TIMER    disables the clock64 bracket
  bool policy = true;          // SCHED_NO_POLICY   disables policy + prefetch
  // Shed (tau budget) reuses the loop's CANONICAL induction variable as its
  // trip counter (not an injected PHI -- that was the old fragility) and
  // masks via the loaded value/softmax score with a dominance-checked
  // replacement, so the woven IR is always valid. On a loop with no canonical
  // IV (fully unrolled/rotated) it DECLINES LOUDLY -- correct-or-absent, never
  // wrong (the eKV rule). Default on; SCHED_NO_SHED disables.
  bool shed = true;            // SCHED_NO_SHED     disables the tau shed lever
  bool clc = false;            // SCHED_CLC         claim via CLC (sm_100+)
  // Timer INDIRECTION (baked/JIT ABI): the timer slot's first 8 bytes hold a
  // POINTER to the actual row table instead of being the table. The host
  // retargets the channel per process by writing that word: a cudaMalloc'd
  // DEVICE buffer (atomics ~free, measured -0.5% vs +26.5% host-mapped --
  // eval_timer_channel.py) or a host-mapped table for zero-touch observers.
  // Word == 0 (the zeroed arena default) -> timer off, fail-safe. Keeps the
  // cross-process JIT-cache contract: the baked address (the arena word) is
  // fixed-VA; only its CONTENTS change per process. Baked-ABI oriented; host
  // apps using runtime/sched_rt.h keep the direct-table layout.
  bool timerIndirect = false;  // SCHED_TIMER_INDIRECT
  // Same indirection for the task_order table: the arena ORDER word holds a
  // retargetable pointer to a DEVICE order tensor (kernel-resident cost/order
  // table), so the control loop installs orders device->device with NO host
  // sync -- removes the per-step .tolist()/sort drain that plan_every masks.
  // 0 pointer -> identity fallback (fail-safe, same as the direct table).
  bool orderIndirect = false;  // SCHED_ORDER_INDIRECT
  // PDL (programmatic dependent launch, sm_90+): overlap consecutive kernels.
  // The weave emits griddepcontrol.wait AFTER the entry table reads (so the
  // control-plane PCIe reads hide inside the PREVIOUS kernel's tail) and
  // griddepcontrol.launch_dependents at each return. Both are E0 scheduling
  // hints: no-ops unless the launch site opts in with
  // cudaLaunchAttributeProgrammaticStreamSerialization.
  bool pdl = false;            // SCHED_PDL         weave PDL overlap points
  unsigned prefetchDist = 8;   // SCHED_PF_DIST     iterations ahead
  // Marginal-exchange constants for the aggressive action (offline-profiled):
  // act iff q*dT - lambda_bw*dR - H > 0.
  double dT = 1.0;             // SCHED_DT
  double dR = 1.0;             // SCHED_DR
  double H = 0.25;             // SCHED_H
  bool debug = false;          // SCHED_DEBUG       loud detection/skip notes
  // Baked-ABI addresses (SCHED_BAKE_<NAME>); 0 = not baked. Keyed by the bare
  // capability name ("task_order","ctrl","timer","queue").
  uint64_t bakeOrder = 0, bakeCtrl = 0, bakeTimer = 0, bakeQueue = 0;
  uint64_t bakedAddr(llvm::StringRef Name) const;
  static Config fromEnv();
};

// Names of the runtime-contract device globals (see runtime/sched_rt.h).
constexpr const char *kTaskOrderSym = "__sched_task_order"; // i32[maxTasks]
constexpr const char *kCtrlSym = "__sched_ctrl";            // SchedCtrl*
constexpr const char *kTimerSym = "__sched_timer";          // u64[maxTasks]
constexpr const char *kQueueSym = "__sched_queue";          // u32 ticket ctr
// SchedCtrl byte offsets (static_assert-pinned in runtime/sched_rt.h).
constexpr unsigned kCtrlNumTasksOff = 4;   // u32 num_tasks
constexpr unsigned kCtrlLambdaOff = 8;     // f32 lambda[4]
// u32 flags word (was reserved sentinel_key; ABI offset unchanged).
// bit0 = TIMER OFF: per-step observation gating. The baked ABI's armed flag
// is a compile-time constant, so the pointer null-check cannot disarm the
// timer at runtime; this data bit can (write flags + push, no recompile).
// Zero (the memset default) = timer ON = the historical behavior.
constexpr unsigned kCtrlFlagsOff = 24;
constexpr unsigned kCtrlFlagTimerOff = 1u; // bit0
// u32 order_size (was reserved num_keys; ABI offset pinned): the tile count
// the installed pi permutation was built for. The weave honors the order
// table ONLY when order_size == nctaid(slot) -- a whole-launch, uniform
// validity check that keeps pi BIJECTIVE under scheduler overlap (a stale
// different-size permutation otherwise duplicates some tiles and drops
// others after per-entry clamping: exactly-once broken, one stale token).
// 0 = UNCHECKED (legacy host-app mode; the per-entry clamp still guards
// faults). The control plane stamps it with each order install.
constexpr unsigned kCtrlOrderSizeOff = 28;
constexpr unsigned kCtrlRowsOff = 32;      // SchedPolicyRow rows, 8 B each
constexpr unsigned kCtrlRowSize = 8;
constexpr unsigned kCtrlRowQOff = 0;       // f32 q within a row
constexpr unsigned kCtrlRowTauOff = 4;     // u16 tau within a row (shed)
constexpr unsigned kCtrlRowHintOff = 6;    // u8 hint within a row
// Hint values (control-plane advisory action; 0 = decide by score).
constexpr unsigned kHintAuto = 0;   // urgent iff q*dT - lambda*dR - H > 0
constexpr unsigned kHintUrgent = 1; // prefetch ahead, L2::evict_last
constexpr unsigned kHintPolite = 2; // prefetch ahead, L2::evict_first (stream)

// Function-attribute markers (the ekv.instrumented analogues).
constexpr const char *kInstrumentedAttr = "sched-instrumented";
constexpr const char *kTaskBodyAttr = "sched-task-body";   // task id = last arg
constexpr const char *kWqDriverAttr = "sched-wq-driver";   // do not weave

bool isNVPTX(const llvm::Module &M);
bool isKernel(const llvm::Function &F);

// sm_NN parsed from the function's "target-cpu" attribute; 0 if unknown.
unsigned smVersion(const llvm::Function &F);

// ctaid/tid/nctaid intrinsic id for an axis (0=x 1=y 2=z).
llvm::Intrinsic::ID ctaidIntrinsic(int axis);
llvm::Intrinsic::ID tidIntrinsic(int axis);
llvm::Intrinsic::ID nctaidIntrinsic(int axis);

llvm::Value *readSReg(llvm::IRBuilderBase &B, llvm::Intrinsic::ID ID);

// 64-bit per-SM cycle counter, the eKV timer source:
//   mov.u64 $0, %clock64;
llvm::Value *readClock64(llvm::IRBuilderBase &B);

// i1: tid.x == 0 && tid.y == 0 && tid.z == 0. LLVM NVVM IR is per-THREAD
// (unlike TTIR's per-CTA ops), so every once-per-CTA effect must be gated.
llvm::Value *threadIsZero(llvm::IRBuilderBase &B);

// CTA-wide barrier (__syncthreads). LLVM <= 21: llvm.nvvm.barrier0;
// LLVM >= 22: llvm.nvvm.barrier.cta.sync.aligned.all(0).
void emitBarrier0(llvm::IRBuilderBase &B);

// The __sched_* global if this module declares it (i.e. the TU includes
// runtime/sched_rt.h), else null -> the capability is skipped loudly.
llvm::GlobalVariable *rtSlot(llvm::Module &M, llvm::StringRef Name);

// Load the buffer pointer out of a runtime slot; returns the generic-AS
// pointer value plus (via NonNull) the i1 "control plane armed this" flag.
llvm::Value *loadRtPointer(llvm::IRBuilderBase &B, llvm::GlobalVariable *GV,
                           llvm::Value **NonNull);

// addrspacecast a loaded (generic) buffer pointer to global AS(1) -- the
// buffers are always device/host-mapped allocations, and AS(1) accesses lower
// to ld.global/atom.global instead of generic forms.
llvm::Value *toGlobalAS(llvm::IRBuilderBase &B, llvm::Value *P);

// --- runtime-buffer resolution: two ABIs, one interface --------------------
// A capability's buffer is reached one of two ways:
//   * HOST-APP ABI: the TU includes runtime/sched_rt.h, so the module declares
//     a `__sched_<name>` global holding the device pointer; we LOAD it.
//   * BAKED ABI (JIT / foreign kernels, the eKV zero-param model): the module
//     does NOT declare the global; the buffer's fixed device ADDRESS is passed
//     to the compiler via env `SCHED_BAKE_<NAME>=<decimal addr>` and baked as
//     inttoptr(addr). Enables weaving a kernel compiled by FlashInfer's JIT
//     subprocess, whose control tables the Python plane owns.
// rtAvailable: is this capability reachable at all (global present OR baked)?
bool rtAvailable(llvm::Module &M, const Config &C, llvm::StringRef Name);

// rtBuffer: the buffer pointer in GLOBAL AS(1), plus (via NonNull) the i1
// "armed" flag. Handles both ABIs. Null-constant + false when unavailable.
llvm::Value *rtBuffer(llvm::IRBuilderBase &B, const Config &C,
                      llvm::StringRef Name, llvm::Value **NonNull);

void debugNote(const Config &C, const llvm::Twine &Msg);

// Print the capability manifest (SchedManifest.h) -- the operator/census/docs
// view of every woven instrument. csv=false: human table; csv=true: one
// machine row per capability (name,effect,minSm,knob,order,tag) for the
// Python mirror's consistency test. Triggered by SCHED_MANIFEST_DUMP[=csv]
// at plugin-load (see SchedPlugin.cpp).
void dumpManifest(bool csv);

} // namespace sched

#endif // SCHED_UTIL_H
