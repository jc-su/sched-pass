#pragma once

#include "AcquireAnalysisInternal.h"

namespace llvm {
class Module;
} // namespace llvm

namespace nta {

bool lowerPartialCommits(llvm::Module &module, const FunctionPlan &plan);

} // namespace nta
