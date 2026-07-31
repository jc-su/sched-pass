#pragma once

#include "AcquireAnalysisInternal.h"

namespace llvm {
class Module;
} // namespace llvm

namespace nta {

bool lowerDeferrals(llvm::Module &module, const FunctionPlan &plan);

} // namespace nta
