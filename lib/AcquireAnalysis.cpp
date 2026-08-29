#include "AcquireAnalysisInternal.h"

#include "nta/AcquireIR.h"
#include "nta/RuntimeABI.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Operator.h"
#include <cstdint>
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
      !isInteger(call.getArgOperand(ir::WorkTicket), 32)) {
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
      !isInteger(call.getArgOperand(ir::SetWorkTicket), 32)) {
    return "dependency-set arguments do not match the marker contract";
  }
  return std::nullopt;
}

std::optional<std::string> validateDefer(CallInst &call) {
  if (call.arg_size() != ir::DeferArgumentCount ||
      !call.getType()->isVoidTy() ||
      !call.getArgOperand(ir::DeferRuntime)->getType()->isPointerTy() ||
      !isInteger(call.getArgOperand(ir::DeferWorkTicket), 32)) {
    return "defer marker has an incompatible ABI";
  }
  return std::nullopt;
}

std::optional<std::string> validatePartialCommit(CallInst &call) {
  if (call.arg_size() != ir::CommitPartialArgumentCount ||
      !call.getType()->isVoidTy() ||
      !call.getArgOperand(ir::CommitRuntime)->getType()->isPointerTy() ||
      !isInteger(call.getArgOperand(ir::CommitWorkTicket), 32) ||
      !isInteger(call.getArgOperand(ir::CommitReductionGroup), 32) ||
      !isInteger(call.getArgOperand(ir::CommitContributorIndex), 32) ||
      !isInteger(call.getArgOperand(ir::CommitContributorCount), 32) ||
      !isInteger(call.getArgOperand(ir::CommitEstimatedComputeNs), 64)) {
    return "partial-publication marker has an incompatible ABI";
  }
  if (!call.isConvergent()) {
    return "partial-publication marker must carry LLVM convergent semantics";
  }
  return std::nullopt;
}

std::optional<std::string> validatePartialBegin(CallInst &call) {
  if (call.arg_size() != ir::BeginPartialArgumentCount ||
      !call.getType()->isVoidTy() ||
      !call.getArgOperand(ir::BeginRuntime)->getType()->isPointerTy() ||
      !isInteger(call.getArgOperand(ir::BeginWorkTicket), 32)) {
    return "partial-region marker has an incompatible ABI";
  }
  if (!call.isConvergent()) {
    return "partial-region marker must carry LLVM convergent semantics";
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

// NTA markers are CTA collectives. The typed frontend asserts that ordinary
// plan/catalog loads used by a marker are immutable for the launch; the engine
// establishes that contract through stream ordering. This verifier rejects
// visible divergence but cannot prove absence of a concurrent writer from a
// plain LLVM load. LLVM's generic GPU uniformity analysis is thread-level and
// classifies blockIdx as divergent, so use the narrower CTA contract here:
// kernel arguments and block/grid dimensions are CTA-uniform; thread, lane,
// and warp identities are not.
class CtaUniformity {
public:
  CtaUniformity(PostDominatorTree &postDominatorTree, LoopInfo &loopInfo)
      : postDominatorTree_(postDominatorTree), loopInfo_(loopInfo) {}

  bool isUniform(Value *value) {
    if (isa<Constant>(value) || isa<Argument>(value) ||
        isa<GlobalValue>(value)) {
      return true;
    }
    auto *instruction = dyn_cast<Instruction>(value);
    if (instruction == nullptr) {
      return false;
    }
    const bool root = evaluationDepth_ == 0 && !settling_;
    ++evaluationDepth_;
    const auto found = states_.find(instruction);
    if (found != states_.end()) {
      if (found->second == Visiting || found->second == Provisional) {
        // Uniform loop-carried values require an optimistic fixed-point seed.
        // Record the recursive region and settle it as one monotone fixed point
        // when the outer query returns. Erase-and-recompute makes a length-N
        // cycle take 2^N classifications and can hang JIT activation.
        markProvisional(instruction);
        ++provisionalEpoch_;
        --evaluationDepth_;
        return true;
      }
      const bool uniform = found->second == Uniform;
      --evaluationDepth_;
      if (root) {
        settleProvisional();
        return states_.lookup(instruction) == Uniform;
      }
      return uniform;
    }
    const std::uint64_t provisionalEpochBefore = provisionalEpoch_;
    states_[instruction] = Visiting;
    const bool uniform = classify(*instruction);
    if (!uniform) {
      // Divergence is monotone and remains sound even when its proof traversed
      // an optimistic recursive edge.
      states_[instruction] = Divergent;
    } else if (provisionalEpoch_ == provisionalEpochBefore) {
      states_[instruction] = Uniform;
    } else {
      // Every value derived from an optimistic recursive edge belongs to the
      // same pending fixed-point region. Do not publish its optimistic answer
      // before that region has stabilized.
      markProvisional(instruction);
      states_[instruction] = Provisional;
    }
    --evaluationDepth_;
    if (root) {
      settleProvisional();
      return states_.lookup(instruction) == Uniform;
    }
    return uniform;
  }

  Value *divergentControl(BasicBlock *target) {
    // Post-dominance alone misses loop self-control: a block non-strictly
    // post-dominates itself even when a lane-varying latch decides how many
    // times each lane executes it. Every exit condition of an enclosing loop
    // therefore participates in CTA-uniform control.
    for (Loop *loop = loopInfo_.getLoopFor(target); loop != nullptr;
         loop = loop->getParentLoop()) {
      SmallVector<BasicBlock *, 8> exitingBlocks;
      loop->getExitingBlocks(exitingBlocks);
      for (BasicBlock *exiting : exitingBlocks) {
        if (Value *condition = branchCondition(exiting->getTerminator());
            condition != nullptr && !isUniform(condition)) {
          return condition;
        }
      }
    }

    for (BasicBlock &controlBlockValue : *target->getParent()) {
      BasicBlock *controlBlock = &controlBlockValue;
      Instruction *terminator = controlBlock->getTerminator();
      Value *condition = branchCondition(terminator);
      if (condition == nullptr) {
        continue;
      }
      bool postDominatesSuccessor = false;
      for (unsigned successor = 0; successor < terminator->getNumSuccessors();
           ++successor) {
        postDominatesSuccessor |= postDominatorTree_.dominates(
            target, terminator->getSuccessor(successor));
      }
      if (postDominatesSuccessor &&
          !postDominatorTree_.dominates(target, controlBlock) &&
          !isUniform(condition)) {
        return condition;
      }
    }
    return nullptr;
  }

private:
  enum State : unsigned { Visiting, Provisional, Uniform, Divergent };

  static Value *branchCondition(Instruction *terminator) {
    if (auto *branch = dyn_cast<BranchInst>(terminator);
        branch != nullptr && branch->isConditional()) {
      return branch->getCondition();
    }
    if (auto *selection = dyn_cast<SwitchInst>(terminator)) {
      return selection->getCondition();
    }
    return nullptr;
  }

  void markProvisional(Instruction *instruction) {
    if (provisionalInstructionSet_.insert(instruction).second) {
      provisionalInstructions_.push_back(instruction);
    }
  }

  void markProvisional(AllocaInst *object) {
    if (provisionalLocalSet_.insert(object).second) {
      provisionalLocals_.push_back(object);
    }
  }

  // At -O0, clang keeps the typed work context in local memory and lowers
  // aggregate initialization to memcpy/memset.  The address of an alloca is
  // lane-private, but a field loaded from it can still be CTA-uniform when
  // every write uses the same uniform source and offset.  Keep this proof
  // separate from ordinary SSA uniformity: treating every alloca as a
  // uniform value would incorrectly bless thread-local scratch variables.
  AllocaInst *localStorageBase(Value *value) const {
    value = value->stripPointerCasts();
    if (auto *alloca = dyn_cast<AllocaInst>(value)) {
      return alloca;
    }
    if (auto *gep = dyn_cast<GetElementPtrInst>(value)) {
      return localStorageBase(gep->getPointerOperand());
    }
    return nullptr;
  }

  bool uniformLocalAddress(Value *value) {
    value = value->stripPointerCasts();
    if (isa<AllocaInst>(value)) {
      return true;
    }
    if (auto *gep = dyn_cast<GetElementPtrInst>(value)) {
      if (!uniformLocalAddress(gep->getPointerOperand())) {
        return false;
      }
      for (Value *index : gep->indices()) {
        if (!isUniform(index)) {
          return false;
        }
      }
      return true;
    }
    if (auto *load = dyn_cast<LoadInst>(value)) {
      return uniformLocalLoad(*load);
    }
    return false;
  }

  bool classifyLocalObject(AllocaInst &object) {
    bool sawWrite = false;
    for (BasicBlock &block : *object.getFunction()) {
      for (Instruction &instruction : block) {
        if (auto *store = dyn_cast<StoreInst>(&instruction)) {
          if (localStorageBase(store->getPointerOperand()) != &object) {
            continue;
          }
          const bool uniformValue =
              isUniform(store->getValueOperand()) ||
              uniformLocalAddress(store->getValueOperand());
          if (store->isAtomic() ||
              !uniformLocalAddress(store->getPointerOperand()) ||
              !uniformValue || !hasUniformControl(store->getParent())) {
            return false;
          }
          sawWrite = true;
          continue;
        }

        if (auto *transfer = dyn_cast<MemTransferInst>(&instruction)) {
          if (localStorageBase(transfer->getRawDest()) != &object) {
            continue;
          }
          if (transfer->isVolatile() ||
              !uniformLocalAddress(transfer->getRawDest()) ||
              !isUniform(transfer->getRawSource()) ||
              !hasUniformControl(transfer->getParent())) {
            return false;
          }
          sawWrite = true;
          continue;
        }

        if (auto *set = dyn_cast<MemSetInst>(&instruction)) {
          if (localStorageBase(set->getRawDest()) != &object) {
            continue;
          }
          if (set->isVolatile() || !uniformLocalAddress(set->getRawDest()) ||
              !isUniform(set->getValue()) ||
              !hasUniformControl(set->getParent())) {
            return false;
          }
          sawWrite = true;
          continue;
        }

        auto *call = dyn_cast<CallBase>(&instruction);
        if (call == nullptr || call->getCalledFunction() == nullptr) {
          continue;
        }
        const StringRef name = call->getCalledFunction()->getName();
        if (!call->getCalledFunction()->isIntrinsic() ||
            (!name.starts_with("llvm.lifetime.") &&
             !name.starts_with("llvm.dbg."))) {
          for (Value *argument : call->args()) {
            if (localStorageBase(argument) == &object) {
              return false;
            }
          }
        }
      }
    }
    return sawWrite;
  }

  bool localObjectUniform(AllocaInst &object) {
    const auto found = localStates_.find(&object);
    if (found != localStates_.end()) {
      if (found->second == Visiting || found->second == Provisional) {
        markProvisional(&object);
        ++provisionalEpoch_;
        return true;
      }
      return found->second == Uniform;
    }
    const std::uint64_t provisionalEpochBefore = provisionalEpoch_;
    localStates_[&object] = Visiting;
    const bool uniform = classifyLocalObject(object);
    if (!uniform) {
      localStates_[&object] = Divergent;
    } else if (provisionalEpoch_ == provisionalEpochBefore) {
      localStates_[&object] = Uniform;
    } else {
      markProvisional(&object);
      localStates_[&object] = Provisional;
    }
    return uniform;
  }

  bool uniformLocalLoad(LoadInst &load) {
    AllocaInst *object = localStorageBase(load.getPointerOperand());
    return object != nullptr && !load.isVolatile() && !load.isAtomic() &&
           uniformLocalAddress(load.getPointerOperand()) &&
           localObjectUniform(*object);
  }

  bool uniformOperands(Instruction &instruction) {
    for (Use &operand : instruction.operands()) {
      if (!isUniform(operand.get())) {
        return false;
      }
    }
    return true;
  }

  bool hasUniformControl(BasicBlock *target) {
    return divergentControl(target) == nullptr;
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
             (isUniform(load->getPointerOperand()) || uniformLocalLoad(*load));
    }
    if (auto *phi = dyn_cast<PHINode>(&instruction)) {
      if (!uniformOperands(*phi)) {
        return false;
      }
      Value *first = phi->getIncomingValue(0);
      bool identical = true;
      for (unsigned index = 1; index < phi->getNumIncomingValues(); ++index) {
        identical &= phi->getIncomingValue(index) == first;
      }
      if (identical) {
        return true;
      }
      for (unsigned index = 0; index < phi->getNumIncomingValues(); ++index) {
        if (!hasUniformControl(phi->getIncomingBlock(index))) {
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

  void settleProvisional() {
    if (provisionalInstructions_.empty() && provisionalLocals_.empty()) {
      return;
    }
    settling_ = true;
    std::size_t initializedInstructions = 0;
    std::size_t initializedLocals = 0;
    bool changed = false;
    do {
      // Classification while settling can expose a previously short-circuited
      // recursive edge. Admit any such node into the same optimistic region
      // before the next pass.
      while (initializedInstructions < provisionalInstructions_.size()) {
        states_[provisionalInstructions_[initializedInstructions++]] = Uniform;
      }
      while (initializedLocals < provisionalLocals_.size()) {
        localStates_[provisionalLocals_[initializedLocals++]] = Uniform;
      }

      changed = false;
      for (Instruction *instruction : provisionalInstructions_) {
        if (states_.lookup(instruction) == Uniform && !classify(*instruction)) {
          states_[instruction] = Divergent;
          changed = true;
        }
      }
      for (AllocaInst *object : provisionalLocals_) {
        if (localStates_.lookup(object) == Uniform &&
            !classifyLocalObject(*object)) {
          localStates_[object] = Divergent;
          changed = true;
        }
      }
      changed |= initializedInstructions != provisionalInstructions_.size() ||
                 initializedLocals != provisionalLocals_.size();
    } while (changed);
    settling_ = false;
    provisionalInstructions_.clear();
    provisionalInstructionSet_.clear();
    provisionalLocals_.clear();
    provisionalLocalSet_.clear();
  }

  DenseMap<Instruction *, State> states_;
  DenseMap<AllocaInst *, State> localStates_;
  SmallVector<Instruction *, 32> provisionalInstructions_;
  SmallPtrSet<Instruction *, 32> provisionalInstructionSet_;
  SmallVector<AllocaInst *, 16> provisionalLocals_;
  SmallPtrSet<AllocaInst *, 16> provisionalLocalSet_;
  std::uint64_t provisionalEpoch_ = 0;
  std::uint32_t evaluationDepth_ = 0;
  bool settling_ = false;
  PostDominatorTree &postDominatorTree_;
  LoopInfo &loopInfo_;
};

std::optional<std::string>
validateCtaCollective(CallInst &marker, CallInst &binding,
                      PostDominatorTree &postDominatorTree, LoopInfo &loopInfo,
                      StringRef effectName = "acquisition marker",
                      bool validateControlDependence = true) {
  const CallingConv::ID callingConvention =
      marker.getFunction()->getCallingConv();
  if (callingConvention != CallingConv::PTX_Kernel &&
      callingConvention != CallingConv::AMDGPU_KERNEL &&
      callingConvention != CallingConv::SPIR_KERNEL) {
    if (effectName == "acquisition marker") {
      return "acquisition markers must be inlined into a GPU kernel entry";
    }
    return effectName.str() + " must be inlined into a GPU kernel entry";
  }
  CtaUniformity uniformity(postDominatorTree, loopInfo);
  for (Use &argument : marker.args()) {
    if (!uniformity.isUniform(argument.get())) {
      return effectName.str() + " has a non-CTA-uniform operand";
    }
  }
  for (Use &argument : binding.args()) {
    if (!uniformity.isUniform(argument.get())) {
      return "request binding has a non-CTA-uniform operand";
    }
  }

  if (!validateControlDependence) {
    return std::nullopt;
  }

  if (Value *condition = uniformity.divergentControl(marker.getParent())) {
    std::string detail =
        effectName.str() + " is control-dependent on a non-CTA-uniform branch";
    detail += " (" + marker.getFunction()->getName().str() + ":";
    detail += condition->hasName() ? condition->getName().str() : "unnamed";
    detail += ")";
    return detail;
  }
  return std::nullopt;
}

struct AcquisitionBranch {
  Instruction *condition;
  BasicBlock *pending;
  BasicBlock *ready;
};

User *onlyLiveUse(Value *value) {
  User *live = nullptr;
  for (User *user : value->users()) {
    // Frontend lowering can leave a dead integer cast next to the branch that
    // consumes a convergent result.  It carries no observable behavior and is
    // safe to ignore; side-effecting or live forwarding uses remain strict.
    if (user->use_empty() && isa<CastInst>(user)) {
      continue;
    }
    if (live != nullptr) {
      return nullptr;
    }
    live = user;
  }
  return live;
}

bool isCanonicalReadyForward(BasicBlock *ready, BasicBlock *target) {
  auto *forward = dyn_cast<BranchInst>(ready->getTerminator());
  return forward != nullptr && !forward->isConditional() &&
         forward->getSuccessor(0) == target;
}

Value *valueOnAcquiredEdge(Value *value, const AcquisitionBranch &branch) {
  auto *phi = dyn_cast<PHINode>(value);
  if (phi == nullptr) {
    return value;
  }
  BasicBlock *incomingBlock = branch.condition->getParent();
  if (phi->getParent() != branch.ready) {
    // Optimized generated kernels may perform one ready-edge reload before
    // joining a common numerical block. Accept only that canonical shape: the
    // ready block must end in one unconditional edge directly to the PHI
    // block. Arbitrary diamonds or multi-hop value forwarding remain rejected.
    if (!isCanonicalReadyForward(branch.ready, phi->getParent())) {
      return value;
    }
    incomingBlock = branch.ready;
  }
  const int incoming = phi->getBasicBlockIndex(incomingBlock);
  return incoming >= 0 ? phi->getIncomingValue(incoming) : value;
}

bool sameValueOnAcquiredEdge(Value *merged, Value *source,
                             const AcquisitionBranch &branch) {
  return valueOnAcquiredEdge(merged, branch) == source;
}

bool sameBindingOnAcquiredEdge(CallInst &merged, CallInst &source,
                               const AcquisitionBranch &branch) {
  return sameValueOnAcquiredEdge(merged.getArgOperand(ir::RequestSlot),
                                 source.getArgOperand(ir::RequestSlot),
                                 branch) &&
         sameValueOnAcquiredEdge(merged.getArgOperand(ir::RequestGeneration),
                                 source.getArgOperand(ir::RequestGeneration),
                                 branch);
}

bool sameBinding(CallInst &left, CallInst &right) {
  return left.getArgOperand(ir::RequestSlot) ==
             right.getArgOperand(ir::RequestSlot) &&
         left.getArgOperand(ir::RequestGeneration) ==
             right.getArgOperand(ir::RequestGeneration);
}

// Generated attention kernels share one large numerical region between the
// dense and partial paths.  Clang represents the optional begin/commit pair as
// two triangles guarded by the same immutable SSA predicate:
//
//          guard(C)                         guard(C)
//       marker edge                     marker edge
//       begin -> join     numerical      commit -> join
//       bypass -> join      region       bypass -> join
//
// Ordinary dominance cannot see that a commit reached with C == P must also
// have taken the begin edge with C == P, and ordinary post-dominance includes
// the infeasible opposite edge at the second guard.  Recognize only this
// canonical, symmetric shape.  Equality of the SSA condition makes the path
// correlation exact; loop guards are deliberately excluded so one static pair
// cannot represent multiple dynamic regions.
struct ConditionalMarkerGuard {
  BranchInst *branch;
  BasicBlock *markerBlock;
  BasicBlock *join;
  bool markerOnTrue;
};

std::optional<ConditionalMarkerGuard>
conditionalMarkerGuard(CallInst &marker) {
  BasicBlock *markerBlock = marker.getParent();
  BasicBlock *predecessor = markerBlock->getSinglePredecessor();
  if (predecessor == nullptr) {
    return std::nullopt;
  }
  auto *guard = dyn_cast<BranchInst>(predecessor->getTerminator());
  auto *joinBranch = dyn_cast<BranchInst>(markerBlock->getTerminator());
  if (guard == nullptr || !guard->isConditional() || joinBranch == nullptr ||
      joinBranch->isConditional()) {
    return std::nullopt;
  }

  const bool markerOnTrue = guard->getSuccessor(0) == markerBlock;
  if (!markerOnTrue && guard->getSuccessor(1) != markerBlock) {
    return std::nullopt;
  }
  BasicBlock *join = joinBranch->getSuccessor(0);
  const unsigned bypassIndex = markerOnTrue ? 1U : 0U;
  if (guard->getSuccessor(bypassIndex) != join) {
    return std::nullopt;
  }
  return ConditionalMarkerGuard{guard, markerBlock, join, markerOnTrue};
}

bool isCorrelatedGuardedRegion(CallInst &begin, CallInst &commit,
                               DominatorTree &dominatorTree,
                               PostDominatorTree &postDominatorTree,
                               LoopInfo &loopInfo) {
  const std::optional<ConditionalMarkerGuard> beginGuard =
      conditionalMarkerGuard(begin);
  const std::optional<ConditionalMarkerGuard> commitGuard =
      conditionalMarkerGuard(commit);
  if (!beginGuard.has_value() || !commitGuard.has_value() ||
      beginGuard->branch->getCondition() !=
          commitGuard->branch->getCondition() ||
      beginGuard->markerOnTrue != commitGuard->markerOnTrue) {
    return false;
  }
  if (loopInfo.getLoopFor(beginGuard->branch->getParent()) != nullptr ||
      loopInfo.getLoopFor(commitGuard->branch->getParent()) != nullptr) {
    return false;
  }

  // Every path after begin reaches the second guard.  On that guard the same
  // SSA value necessarily selects commit.  Conversely, the first guard
  // dominates the second, so every execution of commit selected begin first.
  return dominatorTree.dominates(beginGuard->branch,
                                 commitGuard->branch) &&
         postDominatorTree.dominates(commitGuard->branch->getParent(),
                                     beginGuard->markerBlock);
}

bool reachableWithoutEdges(
    Function &function, BasicBlock *target,
    ArrayRef<std::pair<BasicBlock *, BasicBlock *>> removedEdges) {
  SmallVector<BasicBlock *, 16> worklist{&function.getEntryBlock()};
  SmallPtrSet<BasicBlock *, 16> visited;
  while (!worklist.empty()) {
    BasicBlock *block = worklist.pop_back_val();
    if (!visited.insert(block).second) {
      continue;
    }
    if (block == target) {
      return true;
    }
    for (BasicBlock *successor : successors(block)) {
      bool removed = false;
      for (const auto &edge : removedEdges) {
        removed |= edge.first == block && edge.second == successor;
      }
      if (!removed) {
        worklist.push_back(successor);
      }
    }
  }
  return false;
}

std::optional<AcquisitionBranch> acquisitionBranch(CallInst &marker) {
  if (marker.getType()->isIntegerTy(1)) {
    User *markerUse = onlyLiveUse(&marker);
    if (markerUse == nullptr) {
      return std::nullopt;
    }
    auto *user = dyn_cast<Instruction>(markerUse);
    if (user == nullptr) {
      return std::nullopt;
    }
    if (auto *branch = dyn_cast<BranchInst>(user);
        branch != nullptr && branch->isConditional() &&
        branch->getCondition() == &marker) {
      return AcquisitionBranch{branch, branch->getSuccessor(1),
                               branch->getSuccessor(0)};
    }

    if (auto *phi = dyn_cast<PHINode>(user)) {
      auto *branch = dyn_cast<BranchInst>(onlyLiveUse(phi));
      if (phi->getBasicBlockIndex(marker.getParent()) >= 0 &&
          phi->getIncomingValueForBlock(marker.getParent()) == &marker &&
          branch != nullptr && branch->isConditional() &&
          branch->getCondition() == phi) {
        return AcquisitionBranch{branch, branch->getSuccessor(1),
                                 branch->getSuccessor(0)};
      }
      return std::nullopt;
    }

    // At frontend -O0, a convergent result can cross one unconditional edge
    // and a trivial PHI before the ready/pending branch.  This is still the
    // same canonical deferral boundary; accepting only this exact shape keeps
    // arbitrary result laundering rejected.
    auto *forward = dyn_cast<BranchInst>(user);
    if (forward == nullptr || forward->isConditional()) {
      return std::nullopt;
    }
    BasicBlock *successor = forward->getSuccessor(0);
    for (Instruction &instruction : *successor) {
      auto *phi = dyn_cast<PHINode>(&instruction);
      if (phi == nullptr) {
        break;
      }
      if (phi->getBasicBlockIndex(marker.getParent()) < 0 ||
          phi->getIncomingValueForBlock(marker.getParent()) != &marker ||
          onlyLiveUse(phi) == nullptr) {
        continue;
      }
      auto *branch = dyn_cast<BranchInst>(onlyLiveUse(phi));
      if (branch == nullptr || !branch->isConditional() ||
          branch->getCondition() != phi) {
        continue;
      }
      return AcquisitionBranch{branch, branch->getSuccessor(1),
                               branch->getSuccessor(0)};
    }
    return std::nullopt;
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

// Pointer provenance for the staged-consumption legality clause. The walk is
// deliberately strict: it follows SSA pointer derivation (casts, GEPs, phis,
// selects) and stops at everything else. A pointer that reaches staged memory
// through a load or an opaque call does not certify, and the clause rejects
// it — fail-closed matches the rest of the verifier.
void collectProvenanceRoots(Value *pointer, SmallPtrSetImpl<Value *> &roots) {
  SmallVector<Value *, 8> worklist{pointer};
  SmallPtrSet<Value *, 16> visited;
  while (!worklist.empty()) {
    Value *value = worklist.pop_back_val()->stripPointerCasts();
    if (!visited.insert(value).second) {
      continue;
    }
    if (auto *gep = dyn_cast<GEPOperator>(value)) {
      worklist.push_back(gep->getPointerOperand());
      continue;
    }
    if (auto *phi = dyn_cast<PHINode>(value)) {
      for (Value *incoming : phi->incoming_values()) {
        worklist.push_back(incoming);
      }
      continue;
    }
    if (auto *select = dyn_cast<SelectInst>(value)) {
      worklist.push_back(select->getTrueValue());
      worklist.push_back(select->getFalseValue());
      continue;
    }
    if (auto *frozen = dyn_cast<FreezeInst>(value)) {
      worklist.push_back(frozen->getOperand(0));
      continue;
    }
    roots.insert(value);
  }
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
  const unsigned workTicketArgument =
      dependencySet ? static_cast<unsigned>(ir::SetWorkTicket)
                    : static_cast<unsigned>(ir::WorkTicket);
  const auto *requirementCount =
      dependencySet
          ? dyn_cast<ConstantInt>(marker.getArgOperand(ir::RequirementCount))
          : nullptr;
  const auto *directRequirementCount =
      dependencySet ? dyn_cast<ConstantInt>(
                          marker.getArgOperand(ir::DirectRequirementCount))
                    : nullptr;
  const bool requestGuardOnly =
      requirementCount != nullptr && requirementCount->isZero() &&
      directRequirementCount != nullptr && directRequirementCount->isZero();

  for (User *user : marker.users()) {
    if (user == branch->condition) {
      continue;
    }
    if (auto *forward = dyn_cast<BranchInst>(user);
        forward != nullptr && !forward->isConditional() &&
        forward->getSuccessor(0) == branch->condition->getParent()) {
      continue;
    }
    if (auto *phi = dyn_cast<PHINode>(user);
        phi != nullptr &&
        phi == cast<BranchInst>(branch->condition)->getCondition()) {
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
            call->getArgOperand(ir::DeferWorkTicket) !=
                marker.getArgOperand(workTicketArgument)) {
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

  const unsigned expectedDefers = requestGuardOnly ? 0U : 1U;
  if (deferCount != expectedDefers || !foundReturn) {
    return requestGuardOnly
               ? "request-guard false edge must return without deferral"
               : "pending edge must defer exactly once and return from the "
                 "finite kernel";
  }
  return std::nullopt;
}

} // namespace

FunctionPlan analyzeAcquisitions(Function &function) {
  FunctionPlan plan;
  SmallVector<CallInst *, 8> acquireMarkers;
  SmallVector<CallInst *, 8> deferMarkers;
  SmallVector<CallInst *, 8> partialBeginMarkers;
  SmallVector<CallInst *, 8> partialCommitMarkers;
  SmallVector<CallInst *, 8> requirementAddressCalls;

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
      } else if (hasName(*call, ir::RequirementAddress) ||
                 hasName(*call, ir::RequirementTensorMap)) {
        requirementAddressCalls.push_back(call);
      } else if (hasName(*call, ir::BeginPartialMarker)) {
        partialBeginMarkers.push_back(call);
      } else if (hasName(*call, ir::CommitPartialMarker) ||
                 hasName(*call, ir::StreamCommitPartialMarker)) {
        partialCommitMarkers.push_back(call);
      }
    }
  }

  if (acquireMarkers.empty() && deferMarkers.empty() &&
      partialBeginMarkers.empty() && partialCommitMarkers.empty() &&
      requirementAddressCalls.empty()) {
    return plan;
  }

  DominatorTree dominatorTree(function);
  PostDominatorTree postDominatorTree(function);
  LoopInfo loopInfo(dominatorTree);
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
    if (std::optional<std::string> error = validateCtaCollective(
            *marker, *binding, postDominatorTree, loopInfo)) {
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

  // Staged-consumption legality clause. Generated operators may dereference
  // staged rows only through the pointer their acquisition marker returned, or
  // through nta_requirement_address on the ready edge of the dependency-set
  // acquisition that staged them. Any other pointer into staged memory skips
  // the request-liveness and generation checks the marker certifies.
  for (CallInst *call : requirementAddressCalls) {
    if (call->arg_size() != 2 || !call->getType()->isPointerTy() ||
        !call->getArgOperand(0)->getType()->isPointerTy() ||
        !call->getArgOperand(1)->getType()->isPointerTy()) {
      plan.rejected.push_back(
          {call, "requirement address helper has an incompatible ABI"});
      continue;
    }
    SmallPtrSet<Value *, 8> roots;
    collectProvenanceRoots(call->getArgOperand(1), roots);
    bool derived = false;
    bool onReadyEdge = false;
    for (BoundSite &acquisition : plan.acquisitions) {
      if (!hasName(*acquisition.marker, ir::AcquireSetMarker) ||
          acquisition.marker->getArgOperand(ir::SetRuntime) !=
              call->getArgOperand(0) ||
          !roots.count(acquisition.marker->getArgOperand(ir::Requirements))) {
        continue;
      }
      derived = true;
      std::optional<AcquisitionBranch> branch =
          acquisitionBranch(*acquisition.marker);
      if (branch.has_value() &&
          dominatorTree.dominates(branch->ready, call->getParent())) {
        onReadyEdge = true;
        break;
      }
    }
    if (!derived) {
      plan.rejected.push_back(
          {call, "requirement address does not derive from a bound "
                 "dependency-set acquisition"});
    } else if (!onReadyEdge) {
      plan.rejected.push_back(
          {call, "requirement address is reachable without its dependency-set "
                 "acquisition"});
    }
  }

  SmallPtrSet<Value *, 8> stagedBases;
  for (BoundSite &acquisition : plan.acquisitions) {
    if (hasName(*acquisition.marker, ir::AcquireSetMarker)) {
      continue;
    }
    Value *base =
        acquisition.marker->getArgOperand(ir::DirectBase)->stripPointerCasts();
    if (!isNull(base)) {
      stagedBases.insert(base);
    }
  }
  if (!stagedBases.empty()) {
    // Forward closure over the staged bases: derivation-only instructions
    // (GEPs, casts, freeze, phis, selects) propagate the taint, and every
    // other use of a tainted value is judged by kind. The default for an
    // unlisted instruction kind is rejection, so a use form this list has
    // never seen — today's aggregates and vectors included — cannot
    // launder the pointer past the clause.
    static const char *const derefReason =
        "staged base is dereferenced outside its acquisition marker";
    static const char *const callReason = "staged base escapes through a call";
    static const char *const valueReason =
        "staged base escapes as a stored or converted value";
    SmallVector<Value *, 8> worklist(stagedBases.begin(), stagedBases.end());
    SmallPtrSet<Value *, 16> tainted(stagedBases.begin(), stagedBases.end());
    SmallPtrSet<Instruction *, 8> rejectedInstructions;
    auto reject = [&](Instruction *instruction, const char *reason) {
      if (rejectedInstructions.insert(instruction).second) {
        plan.rejected.push_back({instruction, reason});
      }
    };
    while (!worklist.empty()) {
      Value *value = worklist.pop_back_val();
      for (Use &use : value->uses()) {
        auto *instruction = dyn_cast<Instruction>(use.getUser());
        if (instruction == nullptr) {
          // A constant expression folding a staged base has no program
          // point to anchor a diagnostic; taint its result instead so
          // every eventual instruction use is judged.
          auto *expression = dyn_cast<ConstantExpr>(use.getUser());
          if (expression != nullptr && tainted.insert(expression).second) {
            worklist.push_back(expression);
          }
          continue;
        }
        const bool propagates =
            isa<GetElementPtrInst>(instruction) || isa<CastInst>(instruction) ||
            isa<FreezeInst>(instruction) || isa<PHINode>(instruction) ||
            isa<SelectInst>(instruction);
        if (propagates && !isa<PtrToIntInst>(instruction)) {
          if (tainted.insert(instruction).second) {
            worklist.push_back(instruction);
          }
          continue;
        }
        if (isa<ICmpInst>(instruction)) {
          // Address comparisons dereference nothing; the canonical null
          // test on acquisition results depends on them.
          continue;
        }
        if (isa<LoadInst>(instruction)) {
          reject(instruction, derefReason);
          continue;
        }
        if (isa<StoreInst>(instruction)) {
          reject(instruction,
                 use.getOperandNo() == StoreInst::getPointerOperandIndex()
                     ? derefReason
                     : valueReason);
          continue;
        }
        if (isa<AtomicRMWInst>(instruction) ||
            isa<AtomicCmpXchgInst>(instruction)) {
          reject(instruction,
                 use.getOperandNo() == 0 ? derefReason : valueReason);
          continue;
        }
        if (auto *call = dyn_cast<CallBase>(instruction)) {
          // NTA markers receive the base by design; non-accessing
          // intrinsics are not dereferences. Memory intrinsics and every
          // other callee, known or unknown, take the pointer somewhere
          // this clause cannot follow.
          Function *callee =
              dyn_cast<Function>(call->getCalledOperand()->stripPointerCasts());
          const StringRef name =
              callee != nullptr ? callee->getName() : StringRef();
          const bool ntaMarker =
              name == ir::AcquireMarker || name == ir::AcquireTensorMapMarker ||
              name == ir::AcquireSetMarker || name == ir::DeferMarker ||
              name == ir::BindMarker || name == ir::BeginPartialMarker ||
              name == ir::CommitPartialMarker ||
              name == ir::StreamCommitPartialMarker;
          const bool nonAccessing =
              isa<DbgInfoIntrinsic>(call) ||
              (callee != nullptr &&
               (callee->getIntrinsicID() == Intrinsic::assume ||
                callee->getIntrinsicID() == Intrinsic::lifetime_start ||
                callee->getIntrinsicID() == Intrinsic::lifetime_end));
          if (ntaMarker || nonAccessing) {
            continue;
          }
          reject(instruction,
                 isa<AnyMemIntrinsic>(call) ? derefReason : callReason);
          continue;
        }
        // Returns, pointer-to-integer conversions, aggregate and vector
        // packing, and anything not enumerated above.
        reject(instruction, valueReason);
      }
    }
  }

  for (CallInst *marker : partialBeginMarkers) {
    if (std::optional<std::string> error = validatePartialBegin(*marker)) {
      plan.rejected.push_back({marker, std::move(*error)});
      continue;
    }
    CallInst *binding =
        nearestDominatingBinding(*marker, plan.bindings, dominatorTree);
    if (binding == nullptr) {
      plan.rejected.push_back(
          {marker, "no valid request binding dominates partial region"});
      continue;
    }
    if (std::optional<std::string> error =
            validateCtaCollective(*marker, *binding, postDominatorTree,
                                  loopInfo, "partial-region marker", false)) {
      plan.rejected.push_back({marker, std::move(*error)});
      continue;
    }

    SmallVector<std::pair<BasicBlock *, BasicBlock *>, 4> acquiredEdges;
    for (BoundSite &acquisition : plan.acquisitions) {
      const bool dependencySet =
          hasName(*acquisition.marker, ir::AcquireSetMarker);
      const unsigned runtimeArgument =
          dependencySet ? static_cast<unsigned>(ir::SetRuntime)
                        : static_cast<unsigned>(ir::Runtime);
      std::optional<AcquisitionBranch> branch =
          acquisitionBranch(*acquisition.marker);
      const unsigned ticketArgument =
          dependencySet ? static_cast<unsigned>(ir::SetWorkTicket)
                        : static_cast<unsigned>(ir::WorkTicket);
      const bool matches =
          acquisition.marker->getArgOperand(runtimeArgument) ==
              marker->getArgOperand(ir::BeginRuntime) &&
          branch.has_value() &&
          (dominatorTree.dominates(branch->ready, marker->getParent()) ||
           isCanonicalReadyForward(branch->ready, marker->getParent())) &&
          sameBindingOnAcquiredEdge(*binding, *acquisition.binding, *branch) &&
          sameValueOnAcquiredEdge(
              marker->getArgOperand(ir::BeginWorkTicket),
              acquisition.marker->getArgOperand(ticketArgument), *branch);
      if (!matches) {
        continue;
      }
      acquiredEdges.emplace_back(branch->condition->getParent(), branch->ready);
    }
    if (acquiredEdges.empty()) {
      plan.rejected.push_back(
          {marker,
           "partial region is not on an acquired path with the same request "
           "binding and work ticket"});
      continue;
    }
    if (reachableWithoutEdges(function, marker->getParent(), acquiredEdges)) {
      plan.rejected.push_back(
          {marker, "partial region has a path that bypasses acquisition"});
      continue;
    }
    plan.partialBegins.push_back({marker, binding});
  }

  DenseMap<CallInst *, unsigned> commitsPerBegin;
  for (CallInst *marker : partialCommitMarkers) {
    if (std::optional<std::string> error = validatePartialCommit(*marker)) {
      plan.rejected.push_back({marker, std::move(*error)});
      continue;
    }
    CallInst *binding =
        nearestDominatingBinding(*marker, plan.bindings, dominatorTree);
    if (binding == nullptr) {
      plan.rejected.push_back(
          {marker, "no valid request binding dominates partial publication"});
      continue;
    }
    if (std::optional<std::string> error = validateCtaCollective(
            *marker, *binding, postDominatorTree, loopInfo,
            "partial-publication marker", false)) {
      plan.rejected.push_back({marker, std::move(*error)});
      continue;
    }

    BoundSite *matched = nullptr;
    bool regionWithoutPublication = false;
    for (BoundSite &begin : plan.partialBegins) {
      if (begin.marker->getArgOperand(ir::BeginRuntime) !=
          marker->getArgOperand(ir::CommitRuntime)) {
        continue;
      }
      if (!sameBinding(*begin.binding, *binding) ||
          begin.marker->getArgOperand(ir::BeginWorkTicket) !=
              marker->getArgOperand(ir::CommitWorkTicket)) {
        continue;
      }
      const bool correlatedGuardedRegion = isCorrelatedGuardedRegion(
          *begin.marker, *marker, dominatorTree, postDominatorTree, loopInfo);
      if (!dominatorTree.dominates(begin.marker, marker) &&
          !correlatedGuardedRegion) {
        continue;
      }
      if (!postDominatorTree.dominates(marker->getParent(),
                                       begin.marker->getParent()) &&
          !correlatedGuardedRegion) {
        regionWithoutPublication = true;
        continue;
      }
      if (matched != nullptr) {
        plan.rejected.push_back(
            {marker, "partial publication is ambiguous across regions"});
        matched = nullptr;
        break;
      }
      matched = &begin;
    }

    if (matched == nullptr) {
      if (plan.rejected.empty() || plan.rejected.back().marker != marker) {
        plan.rejected.push_back(
            {marker,
             regionWithoutPublication
                 ? "partial publication must post-dominate its numerical region"
                 : "partial publication is not in a matching numerical "
                   "region"});
      }
      continue;
    }
    if (++commitsPerBegin[matched->marker] != 1) {
      plan.rejected.push_back(
          {marker, "partial numerical region publishes more than once"});
      continue;
    }
    plan.partialCommits.push_back({marker, binding});
  }

  for (const BoundSite &begin : plan.partialBegins) {
    if (commitsPerBegin.lookup(begin.marker) == 0) {
      plan.rejected.push_back(
          {begin.marker, "partial numerical region has no publication"});
    }
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
