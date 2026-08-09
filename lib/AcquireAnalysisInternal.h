#pragma once

#include <string>
#include <vector>

namespace llvm {
class CallInst;
class Function;
} // namespace llvm

namespace nta {

struct BoundSite {
  llvm::CallInst *marker;
  llvm::CallInst *binding;
};

struct RejectedSite {
  llvm::CallInst *marker;
  std::string reason;
};

struct FunctionPlan {
  std::vector<llvm::CallInst *> bindings;
  std::vector<BoundSite> acquisitions;
  std::vector<BoundSite> deferrals;
  std::vector<BoundSite> partialBegins;
  std::vector<BoundSite> partialCommits;
  std::vector<RejectedSite> rejected;
};

FunctionPlan analyzeAcquisitions(llvm::Function &function);

} // namespace nta
