//===- AcquireDiscovery.cpp - structural paged-candidate discovery -------===//
//
// Diagnostic-only port of the prototype pass's structural recognition
// (docs/RECOGNITION_LINEAGE.md): for kernels carrying NO NTA markers,
// recognize the paged-access signature — an innermost-loop, constant-
// stride global load whose address cone passes through the result of
// ANOTHER load (the block-table / page-table row: the identity-carrying
// access) — and report each site as a CANDIDATE acquisition boundary.
//
// Discovery proposes; it never authorizes. Candidates are emitted as
// remarks under NTA_DISCOVERY_NOTES=1 so the typed-frontend gap for a
// kernel family is measurable, and the eKV loud-skip rule is kept: a
// function with loaded-index gathers where no strided site matches says
// so explicitly rather than being silently classified.
//
//===----------------------------------------------------------------------===//
#include "AcquireAnalysisInternal.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

using namespace llvm;

namespace nta {
namespace {

// The paged-access signature (prototype hasLoadedIndex, CapKV lineage):
// does the address cone pass through the result of another load?
bool hasLoadedIndex(Value *value) {
  SmallVector<Value *, 16> work{value};
  SmallPtrSet<Value *, 32> seen;
  unsigned steps = 0;
  while (!work.empty() && steps++ < 256) {
    Value *current = work.pop_back_val();
    if (!seen.insert(current).second) {
      continue;
    }
    if (isa<LoadInst>(current)) {
      return true;
    }
    auto *instruction = dyn_cast<Instruction>(current);
    if (instruction == nullptr) {
      continue;
    }
    if (auto *gep = dyn_cast<GetElementPtrInst>(instruction)) {
      for (Value *operand : gep->operands()) {
        work.push_back(operand);
      }
      continue;
    }
    if (auto *phi = dyn_cast<PHINode>(instruction)) {
      for (Value *incoming : phi->incoming_values()) {
        work.push_back(incoming);
      }
      continue;
    }
    if (isa<BinaryOperator>(instruction) || isa<CastInst>(instruction) ||
        isa<SelectInst>(instruction) || isa<UnaryOperator>(instruction)) {
      for (Value *operand : instruction->operands()) {
        work.push_back(operand);
      }
    }
  }
  return false;
}

} // namespace

void discoverPagedCandidates(Function &function, LoopInfo &loops,
                             ScalarEvolution &scalarEvolution) {
  if (std::getenv("NTA_DISCOVERY_NOTES") == nullptr) {
    return;
  }
  SmallVector<Loop *, 8> innermost;
  for (Loop *top : loops) {
    SmallVector<Loop *, 8> work{top};
    while (!work.empty()) {
      Loop *loop = work.pop_back_val();
      for (Loop *sub : *loop) {
        work.push_back(sub);
      }
      if (loop->getSubLoops().empty()) {
        innermost.push_back(loop);
      }
    }
  }
  bool sawLoadedIndex = false;
  unsigned candidates = 0;
  for (Loop *loop : innermost) {
    for (BasicBlock *block : loop->blocks()) {
      for (Instruction &instruction : *block) {
        auto *load = dyn_cast<LoadInst>(&instruction);
        if (load == nullptr) {
          continue;
        }
        Value *pointer = load->getPointerOperand();
        if (pointer->getType()->getPointerAddressSpace() == 3) {
          continue; // shared memory is never the KV stream
        }
        if (!hasLoadedIndex(pointer)) {
          continue;
        }
        sawLoadedIndex = true;
        if (!scalarEvolution.isSCEVable(pointer->getType())) {
          continue;
        }
        const auto *recurrence =
            dyn_cast<SCEVAddRecExpr>(scalarEvolution.getSCEV(pointer));
        if (recurrence == nullptr || recurrence->getLoop() != loop) {
          continue;
        }
        const auto *step =
            dyn_cast<SCEVConstant>(recurrence->getStepRecurrence(scalarEvolution));
        if (step == nullptr || step->getAPInt() == 0) {
          continue;
        }
        ++candidates;
        WithColor::remark(errs(), "nta")
            << function.getName()
            << ": structural paged candidate (loaded-index cone, stride "
            << step->getAPInt().getSExtValue()
            << " B) — unmarked, skipped loudly; a typed frontend or marker "
               "is required before delegation\n";
      }
    }
  }
  // FlashInfer's real KV stream is cp.async, invisible to load-stride
  // analysis (census 2026-08-22: every production paged-decode kernel
  // carries the loaded-index signature yet zero plain-load sites
  // qualify). Recognize the inline-asm copies and classify their global
  // source cones the same way (prototype AsyncSite lineage): the first
  // "l"-constrained input operand is the gmem source.
  for (Instruction &instruction : instructions(function)) {
    auto *call = dyn_cast<CallInst>(&instruction);
    if (call == nullptr) {
      continue;
    }
    auto *assembly = dyn_cast<InlineAsm>(call->getCalledOperand());
    if (assembly == nullptr) {
      continue;
    }
    StringRef text = assembly->getAsmString();
    if (!text.contains("cp.async") || !text.contains("global")) {
      continue;
    }
    SmallVector<StringRef, 8> constraints;
    StringRef(assembly->getConstraintString())
        .split(constraints, ',', -1, /*KeepEmpty=*/false);
    int argumentIndex = 0;
    int globalSource = -1;
    for (StringRef constraint : constraints) {
      if (constraint.starts_with("=") || constraint.starts_with("~")) {
        continue;
      }
      if (constraint == "l" && globalSource < 0) {
        globalSource = argumentIndex;
      }
      ++argumentIndex;
    }
    if (globalSource < 0 ||
        static_cast<unsigned>(globalSource) >= call->arg_size()) {
      continue;
    }
    Value *source = call->getArgOperand(globalSource);
    if (!source->getType()->isIntegerTy(64) &&
        !source->getType()->isPointerTy()) {
      continue;
    }
    const bool paged = hasLoadedIndex(source);
    sawLoadedIndex |= paged;
    ++candidates;
    WithColor::remark(errs(), "nta")
        << function.getName() << ": structural cp.async candidate ("
        << (paged ? "loaded-index" : "direct") << " source cone) — "
           "unmarked, skipped loudly; a typed frontend or marker is "
           "required before delegation\n";
    if (candidates >= 8) {
      break;
    }
  }
  if (sawLoadedIndex && candidates == 0) {
    WithColor::remark(errs(), "nta")
        << function.getName()
        << ": index-based gather present but no strided site matched — "
           "skipped loudly, never re-bound\n";
  }
}

} // namespace nta
