#pragma once

#include "llvm/IR/PassManager.h"

namespace nta {

class AcquireLoweringPass : public llvm::PassInfoMixin<AcquireLoweringPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &module,
                              llvm::ModuleAnalysisManager &analysisManager);
};

} // namespace nta
