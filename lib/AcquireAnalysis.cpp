#include "AcquireAnalysisInternal.h"

#include "nta/AcquireIR.h"

#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"

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
    return "acquisition marker argument types do not match ABI v1";
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

std::optional<BasicBlock *> pendingSuccessor(CallInst &marker) {
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
  return branch->getSuccessor(trueMeansPending ? 0 : 1);
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

std::optional<std::string> validateDeferralBoundary(CallInst &marker) {
  std::optional<BasicBlock *> pending = pendingSuccessor(marker);
  if (!pending.has_value()) {
    return "acquired pointer must have one canonical null branch";
  }

  bool foundDefer = false;
  bool foundReturn = false;
  SmallVector<BasicBlock *, 4> worklist{*pending};
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
        foundDefer = true;
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

  if (!foundDefer || !foundReturn) {
    return "pending edge must defer and return from the finite kernel";
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
      } else if (hasName(*call, ir::AcquireMarker)) {
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
    if (std::optional<std::string> error = validateAcquire(*marker)) {
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
    if (std::optional<std::string> error = validateDeferralBoundary(*marker)) {
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
