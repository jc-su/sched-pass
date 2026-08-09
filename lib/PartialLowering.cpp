#include "PartialLoweringInternal.h"

#include "nta/AcquireIR.h"
#include "nta/RuntimeABI.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace nta {

bool lowerPartialCommits(Module &module, const FunctionPlan &plan) {
  if (plan.partialBegins.empty() && plan.partialCommits.empty()) {
    return false;
  }

  LLVMContext &context = module.getContext();
  Type *voidType = Type::getVoidTy(context);
  Type *i32 = Type::getInt32Ty(context);
  Type *i64 = Type::getInt64Ty(context);
  Function *operatorFunction =
      !plan.partialCommits.empty()
          ? plan.partialCommits.front().marker->getFunction()
          : plan.partialBegins.front().marker->getFunction();

  for (const BoundSite &site : plan.partialBegins) {
    site.marker->eraseFromParent();
  }

  for (const BoundSite &site : plan.partialCommits) {
    CallInst *marker = site.marker;
    CallInst *binding = site.binding;
    Value *runtime = marker->getArgOperand(ir::CommitRuntime);
    Value *requestSlot = binding->getArgOperand(ir::RequestSlot);
    Value *generation = binding->getArgOperand(ir::RequestGeneration);

    const bool streamOrdered =
        marker->getCalledFunction()->getName() == ir::StreamCommitPartialMarker;
    if (streamOrdered) {
      // The verifier has proven tail placement and collective control. A
      // finite same-stream launch provides the publication boundary, and the
      // runtime retires its exact work plan in the following graph node.
      marker->eraseFromParent();
      continue;
    }
    FunctionCallee commit = module.getOrInsertFunction(
        ir::CommitPartial,
        FunctionType::get(
            voidType, {runtime->getType(), i32, i32, i32, i32, i32, i32, i64},
            false));
    if (auto *commitFunction = dyn_cast<Function>(commit.getCallee())) {
      commitFunction->setConvergent();
    }
    IRBuilder<> builder(marker);
    builder.SetCurrentDebugLocation(marker->getDebugLoc());
    CallInst *lowered = builder.CreateCall(
        commit, {runtime, requestSlot, generation,
                 marker->getArgOperand(ir::CommitWorkTicket),
                 marker->getArgOperand(ir::CommitReductionGroup),
                 marker->getArgOperand(ir::CommitContributorIndex),
                 marker->getArgOperand(ir::CommitContributorCount),
                 marker->getArgOperand(ir::CommitEstimatedComputeNs)});
    lowered->setConvergent();
    Metadata *fields[] = {
        MDString::get(context, "request-bound-partial"),
        ConstantAsMetadata::get(ConstantInt::get(i32, abi::Version)),
        MDString::get(context, "split-phase-cta"),
        MDString::get(context, "request-local-reduction"),
    };
    lowered->setMetadata(ir::PartialMetadata, MDNode::get(context, fields));
    marker->eraseFromParent();
  }

  Metadata *operatorFields[] = {
      MDString::get(context, "request-bound-incremental-cta"),
      ConstantAsMetadata::get(ConstantInt::get(i32, abi::Version)),
      ConstantAsMetadata::get(ConstantInt::get(
          i32, static_cast<std::uint32_t>(plan.partialCommits.size()))),
  };
  operatorFunction->setMetadata(ir::OperatorMetadata,
                                MDNode::get(context, operatorFields));
  return true;
}

} // namespace nta
