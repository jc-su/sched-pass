#include "nta/Passes.h"

#include "llvm/Passes/PassBuilder.h"
#if __has_include("llvm/Plugins/PassPlugin.h")
#include "llvm/Plugins/PassPlugin.h"
#else
#include "llvm/Passes/PassPlugin.h"
#endif

using namespace llvm;

namespace {

PassPluginLibraryInfo pluginInfo() {
  return {
      LLVM_PLUGIN_API_VERSION,
      "NtaPass",
      LLVM_VERSION_STRING,
      [](PassBuilder &passBuilder) {
        passBuilder.registerPipelineParsingCallback(
            [](StringRef name, ModulePassManager &manager,
               ArrayRef<PassBuilder::PipelineElement>) {
              if (name != "nta-acquire") {
                return false;
              }
              manager.addPass(nta::AcquireLoweringPass());
              return true;
            });
      },
  };
}

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return pluginInfo();
}
