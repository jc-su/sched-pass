#include "AcquireAnalysisInternal.h"

#include "nta/AcquireIR.h"

#include "llvm/IR/CFG.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/ADT/DenseMap.h"

#include <optional>
#include <string>
#include <unordered_set>

using namespace llvm;

namespace nta {
namespace {

Function *calledFunction(CallInst &call) {
  Value *callee = call.getCalledOperand()->stripPointerCasts();
  return dyn_cast<Function>(callee);
}

bool hasName(CallInst &call, StringRef name) {
  Function *callee = calledFunction(call);
  return callee != nullptr && callee->getName() == name;
}

bool isInteger(Value *value, unsigned bits) {
  return value->getType()->isIntegerTy(bits);
}

std::optional<std::string> validateBinding(CallInst &call) {
  if (call.arg_size() != ir::BindArgumentCount || !call.getType()->isVoidTy()) {
    return "request binding must be void (i32 request_slot, i32 generation)";
  }
  if (!isInteger(call.getArgOperand(ir::RequestSlot), 32) ||
      !isInteger(call.getArgOperand(ir::RequestGeneration), 32)) {
    return "request binding arguments must both be i32";
  }
  return std::nullopt;
}

std::optional<std::string> validateAcquire(CallInst &call) {
  if (call.arg_size() != ir::AcquireArgumentCount ||
      !call.getType()->isPointerTy()) {
    return "acquisition marker has an incompatible ABI";
  }

  if (!call.getArgOperand(ir::Runtime)->getType()->isPointerTy() ||
      !call.getArgOperand(ir::DirectBase)->getType()->isPointerTy() ||
      !isInteger(call.getArgOperand(ir::ObjectSlot), 32) ||
      !isInteger(call.getArgOperand(ir::ObjectId), 64) ||
      !isInteger(call.getArgOperand(ir::ObjectVersion), 32) ||
      !isInteger(call.getArgOperand(ir::Offset), 64) ||
      !isInteger(call.getArgOperand(ir::Bytes), 32) ||
      !isInteger(call.getArgOperand(ir::Continuation), 32)) {
    return "acquisition marker argument types do not match the marker contract";
  }
  return std::nullopt;
}

std::optional<std::string> validateAcquireSet(CallInst &call) {
  if (call.arg_size() != ir::AcquireSetArgumentCount ||
      !call.getType()->isIntegerTy(1)) {
    return "dependency-set marker has an incompatible ABI";
  }
  if (!call.getArgOperand(ir::SetRuntime)->getType()->isPointerTy() ||
      !call.getArgOperand(ir::Requirements)->getType()->isPointerTy() ||
      !isInteger(call.getArgOperand(ir::RequirementCount), 32) ||
      !isInteger(call.getArgOperand(ir::DirectRequirementCount), 32) ||
      !isInteger(call.getArgOperand(ir::SetContinuation), 32)) {
    return "dependency-set arguments do not match the marker contract";
  }
  return std::nullopt;
}

std::optional<std::string> validateDefer(CallInst &call) {
  if (call.arg_size() != ir::DeferArgumentCount ||
      !call.getType()->isVoidTy() ||
      !call.getArgOperand(ir::DeferRuntime)->getType()->isPointerTy() ||
      !isInteger(call.getArgOperand(ir::DeferContinuation), 32)) {
    return "defer marker has an incompatible ABI";
  }
  return std::nullopt;
}

CallInst *nearestDominatingBinding(CallInst &site,
                                   ArrayRef<CallInst *> bindings,
                                   DominatorTree &dominatorTree) {
  CallInst *nearest = nullptr;
  for (CallInst *candidate : bindings) {
    if (!dominatorTree.dominates(candidate, &site)) {
      continue;
    }
    if (nearest == nullptr || dominatorTree.dominates(nearest, candidate)) {
      nearest = candidate;
    }
  }
  return nearest;
}

bool isNull(Value *value) {
  auto *constant = dyn_cast<Constant>(value);
  return constant != nullptr && constant->isNullValue();
}

// NTA markers are CTA collectives. LLVM's generic GPU uniformity analysis is
// intentionally thread-level and classifies blockIdx as divergent, so use the
// narrower contract needed here: kernel arguments and block/grid dimensions
// are CTA-uniform; thread, lane, and warp identities are not.
class CtaUniformity {
public:
  bool isUniform(Value *value) {
    if (isa<Constant>(value) || isa<Argument>(value) || isa<GlobalValue>(value)) {
      return true;
    }
    auto *instruction = dyn_cast<Instruction>(value);
    if (instruction == nullptr) {
      return false;
    }
    const auto found = states_.find(instruction);
    if (found != states_.end()) {
      // A recursive SSA edge is provisionally uniform. Any divergent seed in
      // the cycle still propagates when the outer query completes.
      return found->second != Divergent;
    }
    states_[instruction] = Visiting;
    const bool uniform = classify(*instruction);
    states_[instruction] = uniform ? Uniform : Divergent;
    return uniform;
  }

private:
  enum State : unsigned { Visiting, Uniform, Divergent };

  bool uniformOperands(Instruction &instruction) {
    for (Use &operand : instruction.operands()) {
      if (!isUniform(operand.get())) {
        return false;
      }
    }
    return true;
  }

  bool classifyCall(CallBase &call) {
    Function *callee = call.getCalledFunction();
    if (callee == nullptr) {
      return false;
    }
    const StringRef name = callee->getName();
    if (name.starts_with("llvm.nvvm.read.ptx.sreg.ctaid.") ||
        name.starts_with("llvm.nvvm.read.ptx.sreg.ntid.") ||
        name.starts_with("llvm.nvvm.read.ptx.sreg.nctaid.") ||
        name == "llvm.nvvm.read.ptx.sreg.warpsize") {
      return true;
    }
    if (name.starts_with("llvm.nvvm.") ||
        name.starts_with("llvm.amdgcn.workitem.") ||
        name.starts_with("llvm.amdgcn.mbcnt.")) {
      return false;
    }
    if (callee->isIntrinsic()) {
      return uniformOperands(call);
    }
    if (name == ir::AcquireMarker || name == ir::AcquireTensorMapMarker ||
        name == ir::AcquireSetMarker) {
      return uniformOperands(call);
    }
    return false;
  }

  bool classify(Instruction &instruction) {
    if (isa<AllocaInst>(instruction) || isa<AtomicRMWInst>(instruction) ||
        isa<AtomicCmpXchgInst>(instruction)) {
      return false;
    }
    if (auto *load = dyn_cast<LoadInst>(&instruction)) {
      return !load->isVolatile() && !load->isAtomic() &&
             isUniform(load->getPointerOperand());
    }
    if (auto *phi = dyn_cast<PHINode>(&instruction)) {
      if (!uniformOperands(*phi)) {
        return false;
      }
      Value *first = phi->getIncomingValue(0);
      for (unsigned index = 1; index < phi->getNumIncomingValues(); ++index) {
        if (phi->getIncomingValue(index) != first) {
          return false;
        }
      }
      return true;
    }
    if (auto *call = dyn_cast<CallBase>(&instruction)) {
      return classifyCall(*call);
    }
    return uniformOperands(instruction);
  }

  DenseMap<Instruction *, State> states_;
};

std::optional<std::string>
validateCtaCollective(CallInst &marker, CallInst &binding,
                      DominatorTree &dominatorTree) {
  const CallingConv::ID callingConvention =
      marker.getFunction()->getCallingConv();
  if (callingConvention != CallingConv::PTX_Kernel &&
      callingConvention != CallingConv::AMDGPU_KERNEL &&
      callingConvention != CallingConv::SPIR_KERNEL) {
    return "acquisition markers must be inlined into a GPU kernel entry";
  }
  CtaUniformity uniformity;
  for (Use &argument : marker.args()) {
    if (!uniformity.isUniform(argument.get())) {
      return "acquisition marker has a non-CTA-uniform operand";
    }
  }
  for (Use &argument : binding.args()) {
    if (!uniformity.isUniform(argument.get())) {
      return "request binding has a non-CTA-uniform operand";
    }
  }

  BasicBlock *markerBlock = marker.getParent();
  DomTreeNode *node = dominatorTree.getNode(markerBlock);
  for (node = node == nullptr ? nullptr : node->getIDom(); node != nullptr;
       node = node->getIDom()) {
    BasicBlock *controlBlock = node->getBlock();
    Instruction *terminator = controlBlock->getTerminator();
    Value *condition = nullptr;
    if (auto *branch = dyn_cast<BranchInst>(terminator);
        branch != nullptr && branch->isConditional()) {
      condition = branch->getCondition();
    } else if (auto *select = dyn_cast<SwitchInst>(terminator)) {
      condition = select->getCondition();
    }
    unsigned reachableSuccessors = 0;
    for (unsigned successor = 0; successor < terminator->getNumSuccessors();
         ++successor) {
      reachableSuccessors +=
          isPotentiallyReachable(terminator->getSuccessor(successor),
                                 markerBlock, nullptr, &dominatorTree)
              ? 1U
              : 0U;
    }
    const bool controlsMarker = reachableSuccessors != 0 &&
                                reachableSuccessors !=
                                    terminator->getNumSuccessors();
    if (controlsMarker && condition != nullptr &&
        !uniformity.isUniform(condition)) {
      return "acquisition marker is control-dependent on a non-CTA-uniform "
             "branch";
    }
  }
  return std::nullopt;
}

struct AcquisitionBranch {
  Instruction *condition;
  BasicBlock *pending;
  BasicBlock *ready;
};

std::optional<AcquisitionBranch> acquisitionBranch(CallInst &marker) {
  if (marker.getType()->isIntegerTy(1)) {
    if (!marker.hasOneUse()) {
      return std::nullopt;
    }
    auto *branch = dyn_cast<BranchInst>(*marker.user_begin());
    if (branch == nullptr || !branch->isConditional() ||
        branch->getCondition() != &marker) {
      return std::nullopt;
    }
    return AcquisitionBranch{branch, branch->getSuccessor(1),
                             branch->getSuccessor(0)};
  }

  ICmpInst *nullComparison = nullptr;
  for (User *user : marker.users()) {
    auto *comparison = dyn_cast<ICmpInst>(user);
    if (comparison == nullptr || !(isNull(comparison->getOperand(0)) ||
                                   isNull(comparison->getOperand(1)))) {
      continue;
    }
    if (nullComparison != nullptr) {
      return std::nullopt;
    }
    nullComparison = comparison;
  }
  if (nullComparison == nullptr ||
      (nullComparison->getPredicate() != ICmpInst::ICMP_EQ &&
       nullComparison->getPredicate() != ICmpInst::ICMP_NE) ||
      !nullComparison->hasOneUse()) {
    return std::nullopt;
  }

  auto *branch = dyn_cast<BranchInst>(*nullComparison->user_begin());
  if (branch == nullptr || !branch->isConditional()) {
    return std::nullopt;
  }

  const bool trueMeansPending =
      nullComparison->getPredicate() == ICmpInst::ICMP_EQ;
  return AcquisitionBranch{
      nullComparison,
      branch->getSuccessor(trueMeansPending ? 0 : 1),
      branch->getSuccessor(trueMeansPending ? 1 : 0),
  };
}

bool isHarmlessPendingInstruction(Instruction &instruction) {
  if (isa<ReturnInst>(instruction) || isa<BranchInst>(instruction) ||
      isa<DbgInfoIntrinsic>(instruction)) {
    return true;
  }
  if (auto *call = dyn_cast<CallInst>(&instruction)) {
    return hasName(*call, ir::DeferMarker) ||
           (calledFunction(*call) != nullptr &&
            calledFunction(*call)->isIntrinsic());
  }
  return isa<PHINode>(instruction);
}

std::optional<std::string>
validateDeferralBoundary(CallInst &marker, DominatorTree &dominatorTree) {
  std::optional<AcquisitionBranch> branch = acquisitionBranch(marker);
  if (!branch.has_value()) {
    return marker.getType()->isIntegerTy(1)
               ? "dependency-set result must directly branch to ready/pending"
               : "acquired pointer must have one canonical null branch";
  }
  const bool dependencySet = marker.getType()->isIntegerTy(1);
  const unsigned runtimeArgument = dependencySet
                                       ? static_cast<unsigned>(ir::SetRuntime)
                                       : static_cast<unsigned>(ir::Runtime);
  const unsigned continuationArgument =
      dependencySet ? static_cast<unsigned>(ir::SetContinuation)
                    : static_cast<unsigned>(ir::Continuation);

  for (User *user : marker.users()) {
    if (user == branch->condition) {
      continue;
    }
    auto *instruction = dyn_cast<Instruction>(user);
    if (instruction == nullptr ||
        !dominatorTree.dominates(branch->ready, instruction->getParent())) {
      return "acquired value is used outside the ready edge";
    }
  }

  unsigned deferCount = 0;
  bool foundReturn = false;
  SmallVector<BasicBlock *, 4> worklist{branch->pending};
  std::unordered_set<BasicBlock *> visited;
  while (!worklist.empty()) {
    BasicBlock *block = worklist.pop_back_val();
    if (!visited.insert(block).second) {
      return "pending edge contains a cycle";
    }
    if (visited.size() > 8) {
      return "pending edge is not a bounded canonical deferral";
    }

    for (Instruction &instruction : *block) {
      if (auto *call = dyn_cast<CallInst>(&instruction);
          call != nullptr && hasName(*call, ir::DeferMarker)) {
        if (call->getArgOperand(ir::DeferRuntime) !=
                marker.getArgOperand(runtimeArgument) ||
            call->getArgOperand(ir::DeferContinuation) !=
                marker.getArgOperand(continuationArgument)) {
          return "pending edge defers a different acquisition token";
        }
        ++deferCount;
      } else if (!isHarmlessPendingInstruction(instruction)) {
        return "pending edge contains state that cannot cross CTA deferral";
      }
    }

    Instruction *terminator = block->getTerminator();
    if (isa<ReturnInst>(terminator)) {
      foundReturn = true;
      continue;
    }
    auto *branch = dyn_cast<BranchInst>(terminator);
    if (branch == nullptr || branch->isConditional()) {
      return "pending edge must end through an unconditional return path";
    }
    worklist.push_back(branch->getSuccessor(0));
  }

  if (deferCount != 1 || !foundReturn) {
    return "pending edge must defer exactly once and return from the finite kernel";
  }
  return std::nullopt;
}

} // namespace

FunctionPlan analyzeAcquisitions(Function &function) {
  FunctionPlan plan;
  SmallVector<CallInst *, 8> acquireMarkers;
  SmallVector<CallInst *, 8> deferMarkers;

  for (BasicBlock &block : function) {
    for (Instruction &instruction : block) {
      auto *call = dyn_cast<CallInst>(&instruction);
      if (call == nullptr) {
        continue;
      }
      if (hasName(*call, ir::BindMarker)) {
        if (std::optional<std::string> error = validateBinding(*call)) {
          plan.rejected.push_back({call, std::move(*error)});
        } else {
          plan.bindings.push_back(call);
        }
      } else if (hasName(*call, ir::AcquireMarker) ||
                 hasName(*call, ir::AcquireTensorMapMarker) ||
                 hasName(*call, ir::AcquireSetMarker)) {
        acquireMarkers.push_back(call);
      } else if (hasName(*call, ir::DeferMarker)) {
        deferMarkers.push_back(call);
      }
    }
  }

  if (acquireMarkers.empty() && deferMarkers.empty()) {
    return plan;
  }

  DominatorTree dominatorTree(function);
  for (CallInst *marker : acquireMarkers) {
    const bool set = hasName(*marker, ir::AcquireSetMarker);
    if (std::optional<std::string> error =
            set ? validateAcquireSet(*marker) : validateAcquire(*marker)) {
      plan.rejected.push_back({marker, std::move(*error)});
      continue;
    }
    CallInst *binding =
        nearestDominatingBinding(*marker, plan.bindings, dominatorTree);
    if (binding == nullptr) {
      plan.rejected.push_back(
          {marker, "no valid request binding dominates acquisition"});
      continue;
    }
    if (std::optional<std::string> error =
            validateCtaCollective(*marker, *binding, dominatorTree)) {
      plan.rejected.push_back({marker, std::move(*error)});
      continue;
    }
    if (std::optional<std::string> error =
            validateDeferralBoundary(*marker, dominatorTree)) {
      plan.rejected.push_back({marker, std::move(*error)});
      continue;
    }
    plan.acquisitions.push_back({marker, binding});
  }

  for (CallInst *marker : deferMarkers) {
    if (std::optional<std::string> error = validateDefer(*marker)) {
      plan.rejected.push_back({marker, std::move(*error)});
      continue;
    }
    CallInst *binding =
        nearestDominatingBinding(*marker, plan.bindings, dominatorTree);
    if (binding == nullptr) {
      plan.rejected.push_back(
          {marker, "no valid request binding dominates deferral"});
      continue;
    }
    plan.deferrals.push_back({marker, binding});
  }

  return plan;
}

} // namespace nta
