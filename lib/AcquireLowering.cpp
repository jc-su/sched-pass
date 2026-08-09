#include "AcquireAnalysisInternal.h"
#include "DeferralLoweringInternal.h"
#include "PartialLoweringInternal.h"

#include "nta/AcquireIR.h"
#include "nta/Passes.h"
#include "nta/RuntimeABI.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <utility>
#include <vector>

#define DEBUG_TYPE "nta-acquire"

using namespace llvm;

STATISTIC(AcquisitionsLowered, "Number of NTA acquisitions lowered");
STATISTIC(RequestGuardsInlined, "Number of request-only guards inlined");
STATISTIC(SitesRejected, "Number of unsafe NTA sites rejected");

namespace nta {
namespace {

MDNode *acquisitionMetadata(LLVMContext &context, bool tensorMap) {
  Metadata *fields[] = {
      MDString::get(context, "request-bound"),
      ConstantAsMetadata::get(
          ConstantInt::get(Type::getInt32Ty(context), abi::Version)),
      MDString::get(context, tensorMap ? "tensor-map" : "byte-address"),
      MDString::get(context, "split-phase-cta"),
  };
  return MDNode::get(context, fields);
}

MDNode *dependencySetMetadata(LLVMContext &context) {
  Metadata *fields[] = {
      MDString::get(context, "request-bound"),
      ConstantAsMetadata::get(
          ConstantInt::get(Type::getInt32Ty(context), abi::Version)),
      MDString::get(context, "dependency-set"),
      MDString::get(context, "split-phase-cta"),
  };
  return MDNode::get(context, fields);
}

bool lowerDependencySet(Module &module, const BoundSite &site,
                        bool streamOrderedCompletion,
                        SmallPtrSetImpl<Function *> &cleanupFunctions) {
  CallInst *marker = site.marker;
  CallInst *binding = site.binding;
  LLVMContext &context = module.getContext();

  Value *runtime = marker->getArgOperand(ir::SetRuntime);
  Value *requirements = marker->getArgOperand(ir::Requirements);
  Value *requirementCount = marker->getArgOperand(ir::RequirementCount);
  Value *directRequirementCount =
      marker->getArgOperand(ir::DirectRequirementCount);
  Value *workTicket = marker->getArgOperand(ir::SetWorkTicket);
  Value *requestSlot = binding->getArgOperand(ir::RequestSlot);
  Value *generation = binding->getArgOperand(ir::RequestGeneration);
  Type *i1 = Type::getInt1Ty(context);
  Type *i32 = Type::getInt32Ty(context);

  const auto *constantRequirementCount =
      dyn_cast<ConstantInt>(requirementCount);
  const auto *constantDirectRequirementCount =
      dyn_cast<ConstantInt>(directRequirementCount);
  const auto *constantWorkTicket = dyn_cast<ConstantInt>(workTicket);
  const bool requestGuardOnly =
      constantRequirementCount != nullptr &&
      constantRequirementCount->isZero() &&
      constantDirectRequirementCount != nullptr &&
      constantDirectRequirementCount->isZero();

  if (requestGuardOnly) {
    // A finite stream-ordered operator retires its exact work plan after the
    // application launch. Its in-kernel acquire remains a request-liveness
    // guard, but must not race that batched retirement by mutating tickets.
    const bool tracksWork =
        !streamOrderedCompletion &&
        (constantWorkTicket == nullptr ||
         constantWorkTicket->getZExtValue() != abi::InvalidIndex);
    FunctionCallee requestLiveCta = module.getOrInsertFunction(
        tracksWork ? ir::RequestLiveWorkCta : ir::RequestLiveCta,
        FunctionType::get(
            i1,
            tracksWork
                ? ArrayRef<Type *>{runtime->getType(), i32, i32, i32}
                : ArrayRef<Type *>{runtime->getType(), i32, i32},
            false));
    IRBuilder<> builder(marker);
    builder.SetCurrentDebugLocation(marker->getDebugLoc());
    SmallVector<Value *, 4> arguments{runtime, requestSlot, generation};
    if (tracksWork) {
      arguments.push_back(workTicket);
    }
    CallInst *ready = builder.CreateCall(
        requestLiveCta, arguments, "nta.request.collective.live");
    ready->setMetadata(ir::AcquisitionMetadata, dependencySetMetadata(context));
    marker->replaceAllUsesWith(ready);
    marker->eraseFromParent();
    Function *guard = ready->getCalledFunction();
    if (guard != nullptr && !guard->isDeclaration()) {
      Function *caller = ready->getFunction();
      InlineFunctionInfo inlineInfo;
      if (InlineFunction(*ready, inlineInfo).isSuccess()) {
        cleanupFunctions.insert(caller);
        ++RequestGuardsInlined;
      }
    }
    ++AcquisitionsLowered;
    return true;
  }

  FunctionCallee acquireSet = module.getOrInsertFunction(
      ir::AcquireSetSlow,
      FunctionType::get(i1,
                        {runtime->getType(), i32, i32, requirements->getType(),
                         i32, i32, i32},
                        false));

  IRBuilder<> builder(marker);
  builder.SetCurrentDebugLocation(marker->getDebugLoc());
  CallInst *ready =
      builder.CreateCall(acquireSet,
                         {runtime, requestSlot, generation, requirements,
                          requirementCount, directRequirementCount, workTicket},
                         "nta.dependencies.ready");
  ready->setMetadata(ir::AcquisitionMetadata, dependencySetMetadata(context));
  marker->replaceAllUsesWith(ready);
  marker->eraseFromParent();
  ++AcquisitionsLowered;
  return true;
}

bool lowerAcquisition(Module &module, const BoundSite &site,
                      bool streamOrderedCompletion,
                      SmallPtrSetImpl<Function *> &cleanupFunctions) {
  CallInst *marker = site.marker;
  const auto *called =
      dyn_cast<Function>(marker->getCalledOperand()->stripPointerCasts());
  if (called != nullptr && called->getName() == ir::AcquireSetMarker) {
    return lowerDependencySet(module, site, streamOrderedCompletion,
                              cleanupFunctions);
  }
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
  Value *workTicket = marker->getArgOperand(ir::WorkTicket);
  const auto *markerFunction =
      dyn_cast<Function>(marker->getCalledOperand()->stripPointerCasts());
  const bool tensorMap =
      markerFunction != nullptr &&
      markerFunction->getName() == ir::AcquireTensorMapMarker;

  Type *pointerType = marker->getType();
  Type *i1 = Type::getInt1Ty(context);
  Type *i8 = Type::getInt8Ty(context);
  Type *i32 = Type::getInt32Ty(context);
  Type *i64 = Type::getInt64Ty(context);

  FunctionCallee requestLive = module.getOrInsertFunction(
      ir::RequestLive,
      FunctionType::get(i1, {runtime->getType(), i32, i32}, false));
  FunctionCallee acquireSlow = module.getOrInsertFunction(
      tensorMap ? ir::AcquireTensorMapSlow : ir::AcquireSlow,
      FunctionType::get(
          pointerType,
          {runtime->getType(), i32, i32, i32, i64, i32, i64, i32, i32}, false));

  BasicBlock *entry = marker->getParent();
  Instruction *afterMarker = marker->getNextNode();
  BasicBlock *workTicketBlock =
      entry->splitBasicBlock(afterMarker, "nta.acquire.cont");
  entry->getTerminator()->eraseFromParent();

  BasicBlock *resolveBlock = BasicBlock::Create(context, "nta.acquire.resolve",
                                                function, workTicketBlock);
  BasicBlock *directBlock = BasicBlock::Create(context, "nta.acquire.direct",
                                               function, workTicketBlock);
  BasicBlock *slowBlock = BasicBlock::Create(context, "nta.acquire.slow",
                                             function, workTicketBlock);

  IRBuilder<> entryBuilder(entry);
  entryBuilder.SetCurrentDebugLocation(marker->getDebugLoc());
  CallInst *live = entryBuilder.CreateCall(
      requestLive, {runtime, requestSlot, generation}, "nta.request.live");
  entryBuilder.CreateCondBr(live, resolveBlock, workTicketBlock);

  IRBuilder<> resolveBuilder(resolveBlock);
  resolveBuilder.SetCurrentDebugLocation(marker->getDebugLoc());
  Value *hasDirect = resolveBuilder.CreateIsNotNull(directBase, "nta.direct");
  resolveBuilder.CreateCondBr(hasDirect, directBlock, slowBlock);

  IRBuilder<> directBuilder(directBlock);
  directBuilder.SetCurrentDebugLocation(marker->getDebugLoc());
  Value *directAddress = directBase;
  if (!tensorMap) {
    directAddress = directBuilder.CreateInBoundsGEP(i8, directBase, offset,
                                                    "nta.direct.address");
  }
  directBuilder.CreateBr(workTicketBlock);

  IRBuilder<> slowBuilder(slowBlock);
  slowBuilder.SetCurrentDebugLocation(marker->getDebugLoc());
  CallInst *slow = slowBuilder.CreateCall(acquireSlow,
                                          {runtime, requestSlot, generation,
                                           objectSlot, objectId, objectVersion,
                                           offset, bytes, workTicket},
                                          "nta.pending.address");
  slow->setMetadata(ir::AcquisitionMetadata,
                    acquisitionMetadata(context, tensorMap));
  slowBuilder.CreateBr(workTicketBlock);

  IRBuilder<> workTicketBuilder(&workTicketBlock->front());
  workTicketBuilder.SetCurrentDebugLocation(marker->getDebugLoc());
  PHINode *result = workTicketBuilder.CreatePHI(pointerType, 3, "nta.address");
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
  SmallPtrSet<Function *, 8> cleanupFunctions;
  for (FunctionPlan &plan : plans) {
    for (const RejectedSite &rejected : plan.rejected) {
      ++SitesRejected;
      rejectedAny = true;
      WithColor::error(errs(), "nta")
          << rejected.marker->getFunction()->getName() << ": "
          << rejected.reason << '\n';
    }
  }

  // Leaving an unsafe marker unresolved turns a verifier failure into a later
  // linker error and allows JIT callers to miss the actual cause. Reject the
  // module before applying any partial lowering.
  if (rejectedAny) {
    report_fatal_error("NTA acquisition verification failed", false);
  }

  for (FunctionPlan &plan : plans) {
    const bool streamOrderedCompletion = llvm::any_of(
        plan.partialCommits, [](const BoundSite &site) {
          const auto *called = dyn_cast<Function>(
              site.marker->getCalledOperand()->stripPointerCasts());
          return called != nullptr &&
                 called->getName() == ir::StreamCommitPartialMarker;
        });
    for (const BoundSite &site : plan.acquisitions) {
      changed |= lowerAcquisition(module, site, streamOrderedCompletion,
                                  cleanupFunctions);
    }
    changed |= lowerDeferrals(module, plan);
    changed |= lowerPartialCommits(module, plan);

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
    removeUnusedMarker(module, ir::AcquireTensorMapMarker);
    removeUnusedMarker(module, ir::AcquireSetMarker);
    removeUnusedMarker(module, ir::DeferMarker);
    removeUnusedMarker(module, ir::BeginPartialMarker);
    removeUnusedMarker(module, ir::CommitPartialMarker);
    removeUnusedMarker(module, ir::StreamCommitPartialMarker);

    // The plugin runs at optimizer-last so lowering can verify the final CUDA
    // control flow. Explicitly inlined request guards would otherwise miss the
    // normal scalar cleanup pipeline and inflate attention-kernel live ranges.
    FunctionPassManager cleanup;
    cleanup.addPass(InstCombinePass());
    cleanup.addPass(SimplifyCFGPass());
    cleanup.addPass(EarlyCSEPass());
    cleanup.addPass(InstCombinePass());
    cleanup.addPass(SimplifyCFGPass());
    auto &functionAnalyses =
        analysisManager
            .getResult<FunctionAnalysisManagerModuleProxy>(module)
            .getManager();
    for (Function *function : cleanupFunctions) {
      functionAnalyses.invalidate(*function, PreservedAnalyses::none());
      cleanup.run(*function, functionAnalyses);
    }
  }

  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

} // namespace nta
