//===- sched/SchedPasses.h - sched-pass public pass API --------*- C++ -*-===//
//
// sched-pass ships two new-PM module passes, run in order as `sched-weave`:
//
//   SchedWorkQueuePass  - (opt-in, SCHED_WORKQUEUE=1) the task-acquisition
//                         layer: persistent-worker transform with an atomic
//                         ticket claim -- the pre-Blackwell software analogue
//                         of Cluster Launch Control. On sm_100+ the claim
//                         block is the CLC slot (see SchedWorkQueue.cpp).
//
//   SchedWeavePass      - task_order indirection, price-guided policy
//                         (prefetch action), and the clock64 feedback timer.
//
// Loaded as an LLVM pass plugin (llvmGetPassPluginInfo, the LLVM analogue of
// Triton's tritonGetPluginInfo):
//   * testing:     opt -load-pass-plugin=libSchedPass.so -passes=sched-weave
//   * production:  clang++ -x cuda ... -fpass-plugin=libSchedPass.so
//                  (runs at the OptimizerLast extension point, device IR only)
//
//===----------------------------------------------------------------------===//
#ifndef SCHED_PASSES_H
#define SCHED_PASSES_H

#include "llvm/IR/PassManager.h"

namespace sched {

// NOTE: on LLVM <= 21 the CRTP mix-in is PassInfoMixin; newer LLVM renames it
// to OptionalPassInfoMixin (an optional pass may be skipped by the manager).
// The run() contract is identical.
class SchedWorkQueuePass : public llvm::PassInfoMixin<SchedWorkQueuePass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &AM);
};

class SchedWeavePass : public llvm::PassInfoMixin<SchedWeavePass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &AM);
};

} // namespace sched

#endif // SCHED_PASSES_H
