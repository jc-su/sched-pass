//===- SchedPlugin.cpp - sched-pass LLVM plugin entry point --------------===//
//
// llvmGetPassPluginInfo() -- the LLVM new-PM analogue of Triton's
// tritonGetPluginInfo() (which eKVPlugin.cpp / CapKVPlugin.cpp export).
// Registers the sched passes two ways, mirroring the Triton plugin's
// registerPass + addPass callbacks:
//
//   * pipeline parsing:  opt -load-pass-plugin=libSchedPass.so
//                            -passes=sched-weave  in.ll -S -o out.ll
//   * OptimizerLast EP:  clang++ -x cuda --cuda-gpu-arch=sm_86 -O2
//                            -fpass-plugin=libSchedPass.so kernel.cu
//     (fires for BOTH host and device compilations; the passes themselves
//      no-op on non-NVPTX modules, so host IR passes through untouched)
//
//===----------------------------------------------------------------------===//
#include "sched/SchedPasses.h"
#include "SchedUtil.h"

#include <cstdlib>

#include "llvm/Config/llvm-config.h"
#include "llvm/Passes/PassBuilder.h"
// LLVM 22 moved PassPlugin.h from llvm/Passes/ to llvm/Plugins/ (and bumped
// LLVM_PLUGIN_API_VERSION 1 -> 2). PassBuilder.h stayed in llvm/Passes/.
#if LLVM_VERSION_MAJOR >= 22
#include "llvm/Plugins/PassPlugin.h"
#else
#include "llvm/Passes/PassPlugin.h"
#endif

using namespace llvm;

static void addSchedPasses(ModulePassManager &MPM) {
  MPM.addPass(sched::SchedWorkQueuePass()); // opt-in via SCHED_WORKQUEUE=1
  MPM.addPass(sched::SchedWeavePass());
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  // Operator/census/docs hook: dump the capability manifest at plugin load.
  // `SCHED_MANIFEST_DUMP=csv` emits the machine table the Python mirror's
  // consistency test parses; any other value prints the human table.
  if (const char *M = std::getenv("SCHED_MANIFEST_DUMP"))
    sched::dumpManifest(StringRef(M) == "csv");
  return {LLVM_PLUGIN_API_VERSION, "sched-pass", "0.1.0",
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "sched-weave") {
                    addSchedPasses(MPM);
                    return true;
                  }
                  if (Name == "sched-workqueue") {
                    MPM.addPass(sched::SchedWorkQueuePass());
                    return true;
                  }
                  return false;
                });
            PB.registerOptimizerLastEPCallback(
#if LLVM_VERSION_MAJOR >= 20
                [](ModulePassManager &MPM, OptimizationLevel,
                   ThinOrFullLTOPhase) { addSchedPasses(MPM); });
#else
                [](ModulePassManager &MPM, OptimizationLevel) {
                  addSchedPasses(MPM);
                });
#endif
          }};
}
