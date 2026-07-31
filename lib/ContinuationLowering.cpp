#include "ContinuationLoweringInternal.h"

#include "nta/AcquireIR.h"
#include "nta/RuntimeABI.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace nta {

bool lowerDeferrals(Module &module, const FunctionPlan &plan) {
  if (plan.deferrals.empty()) {
    return false;
  }

  LLVMContext &context = module.getContext();
  Type *voidType = Type::getVoidTy(context);
  Type *i32 = Type::getInt32Ty(context);

  for (const BoundSite &site : plan.deferrals) {
    CallInst *marker = site.marker;
    CallInst *binding = site.binding;
    Value *runtime = marker->getArgOperand(ir::DeferRuntime);
    Value *continuation = marker->getArgOperand(ir::DeferContinuation);
    Value *requestSlot = binding->getArgOperand(ir::RequestSlot);
    Value *generation = binding->getArgOperand(ir::RequestGeneration);

    FunctionCallee defer = module.getOrInsertFunction(
        ir::Defer, FunctionType::get(
                       voidType, {runtime->getType(), i32, i32, i32}, false));
    IRBuilder<> builder(marker);
    CallInst *lowered = builder.CreateCall(
        defer, {runtime, requestSlot, generation, continuation});
    Metadata *fields[] = {
        MDString::get(context, "request-bound"),
        ConstantAsMetadata::get(ConstantInt::get(i32, abi::Version)),
    };
    lowered->setMetadata(ir::AcquisitionMetadata, MDNode::get(context, fields));
    marker->eraseFromParent();
  }
  return true;
}

} // namespace nta
