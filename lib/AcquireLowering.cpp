#include "AcquireAnalysisInternal.h"
#include "ContinuationLoweringInternal.h"

#include "nta/AcquireIR.h"
#include "nta/Passes.h"
#include "nta/RuntimeABI.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/WithColor.h"

#include <utility>
#include <vector>

#define DEBUG_TYPE "nta-acquire"

using namespace llvm;

STATISTIC(AcquisitionsLowered, "Number of NTA acquisitions lowered");
STATISTIC(SitesRejected, "Number of unsafe NTA sites rejected");

namespace nta {
namespace {

MDNode *acquisitionMetadata(LLVMContext &context) {
  Metadata *fields[] = {
      MDString::get(context, "request-bound"),
      ConstantAsMetadata::get(
          ConstantInt::get(Type::getInt32Ty(context), abi::Version)),
  };
  return MDNode::get(context, fields);
}

bool lowerAcquisition(Module &module, const BoundSite &site) {
  CallInst *marker = site.marker;
  CallInst *binding = site.binding;
  Function *function = marker->getFunction();
  LLVMContext &context = module.getContext();

  Value *runtime = marker->getArgOperand(ir::Runtime);
  Value *directBase = marker->getArgOperand(ir::DirectBase);
  Value *requestSlot = binding->getArgOperand(ir::RequestSlot);
  Value *generation = binding->getArgOperand(ir::RequestGeneration);
  Value *objectSlot = marker->getArgOperand(ir::ObjectSlot);
  Value *objectId = marker->getArgOperand(ir::ObjectId);
  Value *objectVersion = marker->getArgOperand(ir::ObjectVersion);
  Value *offset = marker->getArgOperand(ir::Offset);
  Value *bytes = marker->getArgOperand(ir::Bytes);
  Value *continuation = marker->getArgOperand(ir::Continuation);

  Type *pointerType = marker->getType();
  Type *i1 = Type::getInt1Ty(context);
  Type *i8 = Type::getInt8Ty(context);
  Type *i32 = Type::getInt32Ty(context);
  Type *i64 = Type::getInt64Ty(context);

  FunctionCallee requestLive = module.getOrInsertFunction(
      ir::RequestLive,
      FunctionType::get(i1, {runtime->getType(), i32, i32}, false));
  FunctionCallee acquireSlow = module.getOrInsertFunction(
      ir::AcquireSlow,
      FunctionType::get(
          pointerType,
          {runtime->getType(), i32, i32, i32, i64, i32, i64, i32, i32}, false));

  BasicBlock *entry = marker->getParent();
  Instruction *afterMarker = marker->getNextNode();
  BasicBlock *continuationBlock =
      entry->splitBasicBlock(afterMarker, "nta.acquire.cont");
  entry->getTerminator()->eraseFromParent();

  BasicBlock *resolveBlock = BasicBlock::Create(context, "nta.acquire.resolve",
                                                function, continuationBlock);
  BasicBlock *directBlock = BasicBlock::Create(context, "nta.acquire.direct",
                                               function, continuationBlock);
  BasicBlock *slowBlock = BasicBlock::Create(context, "nta.acquire.slow",
                                             function, continuationBlock);

  IRBuilder<> entryBuilder(entry);
  CallInst *live = entryBuilder.CreateCall(
      requestLive, {runtime, requestSlot, generation}, "nta.request.live");
  entryBuilder.CreateCondBr(live, resolveBlock, continuationBlock);

  IRBuilder<> resolveBuilder(resolveBlock);
  Value *hasDirect = resolveBuilder.CreateIsNotNull(directBase, "nta.direct");
  resolveBuilder.CreateCondBr(hasDirect, directBlock, slowBlock);

  IRBuilder<> directBuilder(directBlock);
  Value *directAddress = directBuilder.CreateInBoundsGEP(i8, directBase, offset,
                                                         "nta.direct.address");
  directBuilder.CreateBr(continuationBlock);

  IRBuilder<> slowBuilder(slowBlock);
  CallInst *slow = slowBuilder.CreateCall(acquireSlow,
                                          {runtime, requestSlot, generation,
                                           objectSlot, objectId, objectVersion,
                                           offset, bytes, continuation},
                                          "nta.pending.address");
  slow->setMetadata(ir::AcquisitionMetadata, acquisitionMetadata(context));
  slowBuilder.CreateBr(continuationBlock);

  IRBuilder<> continuationBuilder(&continuationBlock->front());
  PHINode *result =
      continuationBuilder.CreatePHI(pointerType, 3, "nta.address");
  result->addIncoming(ConstantPointerNull::get(cast<PointerType>(pointerType)),
                      entry);
  result->addIncoming(directAddress, directBlock);
  result->addIncoming(slow, slowBlock);

  marker->replaceAllUsesWith(result);
  marker->eraseFromParent();
  ++AcquisitionsLowered;
  return true;
}

void removeUnusedMarker(Module &module, StringRef name) {
  Function *function = module.getFunction(name);
  if (function != nullptr && function->isDeclaration() &&
      function->use_empty()) {
    function->eraseFromParent();
  }
}

} // namespace

PreservedAnalyses
AcquireLoweringPass::run(Module &module,
                         ModuleAnalysisManager &analysisManager) {
  (void)analysisManager;
  std::vector<FunctionPlan> plans;
  plans.reserve(module.size());

  for (Function &function : module) {
    if (!function.isDeclaration()) {
      plans.push_back(analyzeAcquisitions(function));
    }
  }

  bool changed = false;
  bool rejectedAny = false;
  for (FunctionPlan &plan : plans) {
    for (const RejectedSite &rejected : plan.rejected) {
      ++SitesRejected;
      rejectedAny = true;
      WithColor::warning(errs(), "nta")
          << rejected.marker->getFunction()->getName() << ": "
          << rejected.reason << '\n';
    }

    for (const BoundSite &site : plan.acquisitions) {
      changed |= lowerAcquisition(module, site);
    }
    changed |= lowerDeferrals(module, plan);

    if (plan.rejected.empty()) {
      for (CallInst *binding : plan.bindings) {
        binding->eraseFromParent();
        changed = true;
      }
    }
  }

  if (changed) {
    module.addModuleFlag(Module::ModFlagBehavior::Warning,
                         ir::LoweredModuleFlag, abi::Version);
    removeUnusedMarker(module, ir::BindMarker);
    removeUnusedMarker(module, ir::AcquireMarker);
    removeUnusedMarker(module, ir::DeferMarker);
  }

  return changed || rejectedAny ? PreservedAnalyses::none()
                                : PreservedAnalyses::all();
}

} // namespace nta
