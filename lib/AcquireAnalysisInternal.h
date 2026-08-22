#pragma once

#include <string>
#include <vector>

namespace llvm {
class CallInst;
class Function;
class Instruction;
class LoopInfo;
class ScalarEvolution;
} // namespace llvm

namespace nta {

struct BoundSite {
  llvm::CallInst *marker;
  llvm::CallInst *binding;
};

struct RejectedSite {
  llvm::Instruction *marker;
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

// Diagnostic-only structural recognition of paged-KV candidate sites in
// unmarked kernels (NTA_DISCOVERY_NOTES=1); proposes, never authorizes.
void discoverPagedCandidates(llvm::Function &function, llvm::LoopInfo &loops,
                             llvm::ScalarEvolution &scalarEvolution);

} // namespace nta
