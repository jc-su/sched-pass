//===- SchedWeave.cpp - task indirection + policy + timer weaving --------===//
//
// The LLVM-IR port of the eKV weaving discipline for the continuous-batching
// scheduling model. One module pass, three capabilities per selected function,
// all additive data-plane edits (pure reads / commutative atomic adds /
// architectural hints), all fail-safe when the control plane is absent:
//
//   1. TASK INDIRECTION   task = task_order[ctaid(slotAxis)]; every prior use
//                         of ctaid(slotAxis) is remapped to `task`. The control
//                         plane orders the array by request urgency (urgent
//                         tiles first), so priority is expressed with zero
//                         launch-site changes. Null table / OOB pid -> identity.
//
//   2. POLICY             q = ctrl->task[task].q; lambda = ctrl->lambda[bw];
//                         act = (q*dT - lambda*dR - H > 0)      [uniform i1]
//                         For each detected KV-streaming load (a load whose
//                         address depends on ANOTHER load -- CapKV's paged
//                         signature -- with a constant-stride AddRec address),
//                         weave: prefetch.L2 [select(act, addr+D*stride, addr)]
//                         The select-address trick keeps the loop branchless:
//                         disabled -> prefetch of the current line (an L2 hit,
//                         ~free). All operands are CTA-uniform: no divergence.
//
//   3. TIMER              t0 = clock64 at entry; at every ret: one gated
//                         (tid==0) atomicrmw add of (t1-t0) into timer[task].
//                         The per-request residency signal that closes the
//                         control loop (eKV emitTimer, per-thread edition).
//
// Functions marked "sched-task-body" (produced by SchedWorkQueuePass) take the
// task id from their last parameter instead of ctaid -- same weave otherwise.
//
//===----------------------------------------------------------------------===//
#include "SchedUtil.h"
#include "sched/SchedPasses.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

namespace sched {

namespace {

// A KV-streaming load worth acting on: constant stride (bytes) per iteration
// of its innermost loop, address derived from a loaded value (paged/CSR mu)
// or from the slot-axis ctaid (arithmetic mu).
struct PrefetchSite {
  LoadInst *Load;
  int64_t StepBytes;
};

// A cp.async global->shared stream site (FlashInfer's real KV load path): an
// inline-asm `cp.async.cg.shared.global [smem], [gmem], size` whose GLOBAL
// SOURCE we can warm ahead of time. The compute later reads from smem (an
// AS=3 load the PrefetchSite detector deliberately skips), so this is the ONLY
// handle on the global KV stream in a cp.async kernel.
struct AsyncSite {
  CallInst *Copy; // the cp.async inline-asm call
  Value *Gmem;    // its global source pointer operand (the "l"-constrained arg)
};

// The two policy tiers a site can fire (both CTA-uniform i1):
//   urgent -> prefetch ahead with L2::evict_last  (keep my lines resident)
//   polite -> prefetch ahead with L2::evict_first (stream: lines die young,
//             the long request stops polluting L2 for everyone else)
struct PolicyFlags {
  Value *Urgent = nullptr;
  Value *Polite = nullptr;
};

// CapKV's hasLoadedIndex, ported to scalar IR: does `V`'s def-cone pass
// through the result of another load? (The paged-access signature: the
// address is data-dependent on memory -- a block-table / page-id load.)
bool hasLoadedIndex(Value *V) {
  SmallVector<Value *, 16> Work{V};
  SmallPtrSet<Value *, 32> Seen;
  unsigned Steps = 0;
  while (!Work.empty() && Steps++ < 256) {
    Value *Cur = Work.pop_back_val();
    if (!Seen.insert(Cur).second)
      continue;
    if (isa<LoadInst>(Cur))
      return true;
    auto *I = dyn_cast<Instruction>(Cur);
    if (!I)
      continue;
    if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
      for (Value *Op : GEP->operands())
        Work.push_back(Op);
      continue;
    }
    if (auto *PN = dyn_cast<PHINode>(I)) {
      for (Value *In : PN->incoming_values())
        Work.push_back(In);
      continue;
    }
    if (isa<BinaryOperator>(I) || isa<CastInst>(I) || isa<SelectInst>(I) ||
        isa<UnaryOperator>(I))
      for (Value *Op : I->operands())
        Work.push_back(Op);
  }
  return false;
}

// Does `V`'s def-cone reach a read of ctaid(slotAxis)? The arithmetic-mu
// signature: the request's KV region is addressed as base + seq*stride with
// no index array (eKV's findArithKeystone, scalar-IR edition). Never crosses
// a load (crossing one would make it index-based, a different mu form).
bool reachesSlotCtaid(Value *V, Intrinsic::ID SlotID) {
  SmallVector<Value *, 16> Work{V};
  SmallPtrSet<Value *, 32> Seen;
  unsigned Steps = 0;
  while (!Work.empty() && Steps++ < 256) {
    Value *Cur = Work.pop_back_val();
    if (!Seen.insert(Cur).second)
      continue;
    if (auto *CI = dyn_cast<CallInst>(Cur)) {
      if (CI->getIntrinsicID() == SlotID)
        return true;
      continue;
    }
    auto *I = dyn_cast<Instruction>(Cur);
    if (!I || isa<LoadInst>(I))
      continue; // stop at loads: that would be the index-based form
    if (auto *PN = dyn_cast<PHINode>(I)) {
      for (Value *In : PN->incoming_values())
        Work.push_back(In);
      continue;
    }
    if (isa<GetElementPtrInst>(I) || isa<BinaryOperator>(I) ||
        isa<CastInst>(I) || isa<SelectInst>(I))
      for (Value *Op : I->operands())
        Work.push_back(Op);
  }
  return false;
}

// Innermost-loop KV-stream loads with a constant-stride AddRec address. Two
// mu forms, one model (eKV GENERALIZATION.md: the grading/style is a
// parameter, not the model):
//   * index-based (paged B=16, CSR B=1, any indirection depth): the address
//     depends on a LOADED value -- preferred, tried first;
//   * arithmetic (B=infinity, contiguous): no loaded index anywhere in the
//     loops AND the address derives from ctaid(slotAxis). A POSITIVE
//     classification, not a fallback: a kernel that HAS loaded-index loads
//     but matched none is skipped loudly, never silently re-bound (the eKV
//     rule). TMA-descriptor streams are not visible as plain loads at this
//     level -- the launch-arg / runtime-trace observation levels cover them.
// Run BEFORE any mutation (analyses must be valid); recorded Instruction*s
// survive later CFG splits. One site per loop, small global cap.
void findPrefetchSites(Function &F, LoopInfo &LI, ScalarEvolution &SE,
                       const Config &C,
                       SmallVectorImpl<PrefetchSite> &Sites) {
  Intrinsic::ID SlotID = ctaidIntrinsic(C.slotAxis);
  bool SawLoadedIndex = false;

  SmallVector<Loop *, 8> Inner;
  for (Loop *Top : LI) {
    SmallVector<Loop *, 8> Work{Top};
    while (!Work.empty()) {
      Loop *L = Work.pop_back_val();
      for (Loop *Sub : *L)
        Work.push_back(Sub);
      if (L->getSubLoops().empty())
        Inner.push_back(L);
    }
  }

  auto strideOf = [&](LoadInst *Ld, Loop *L) -> int64_t {
    Value *P = Ld->getPointerOperand();
    if (P->getType()->getPointerAddressSpace() == 3)
      return 0; // shared memory: never a KV stream
    if (!SE.isSCEVable(P->getType()))
      return 0;
    const auto *AR = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(P));
    if (!AR || AR->getLoop() != L)
      return 0;
    const auto *Step = dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE));
    return Step ? Step->getAPInt().getSExtValue() : 0;
  };

  // Pass 1: index-based mu (paged / CSR, any depth). ALL matching loads in a
  // loop are sites (K and V streams of an attention body are distinct loads
  // in the same loop and must both be shaped/shed), small global cap.
  for (Loop *L : Inner) {
    if (Sites.size() >= 4)
      break;
    for (BasicBlock *BB : L->blocks()) {
      for (Instruction &I : *BB) {
        auto *Ld = dyn_cast<LoadInst>(&I);
        if (!Ld)
          continue;
        if (!hasLoadedIndex(Ld->getPointerOperand()))
          continue;
        SawLoadedIndex = true;
        int64_t Bytes = strideOf(Ld, L);
        if (Bytes == 0)
          continue;
        Sites.push_back({Ld, Bytes});
        debugNote(C, "prefetch site (index-based mu) in " + F.getName() +
                         ": stride " + Twine(Bytes) + " B");
        if (Sites.size() >= 4)
          break;
      }
      if (Sites.size() >= 4)
        break;
    }
  }
  if (!Sites.empty())
    return;
  if (SawLoadedIndex) {
    debugNote(C, F.getName() + ": index-based KV gather present but no "
                               "strided site matched -> skip LOUDLY (a "
                               "detection gap; NOT re-bound as arithmetic)");
    return;
  }

  // Pass 2: arithmetic mu (contiguous KV, address = f(ctaid) + i*stride).
  for (Loop *L : Inner) {
    if (Sites.size() >= 2)
      break;
    for (BasicBlock *BB : L->blocks()) {
      bool Found = false;
      for (Instruction &I : *BB) {
        auto *Ld = dyn_cast<LoadInst>(&I);
        if (!Ld)
          continue;
        int64_t Bytes = strideOf(Ld, L);
        if (Bytes == 0)
          continue;
        if (!reachesSlotCtaid(Ld->getPointerOperand(), SlotID))
          continue; // not the per-request stream (e.g. shared weights)
        Sites.push_back({Ld, Bytes});
        debugNote(C, "prefetch site (arithmetic mu) in " + F.getName() +
                         ": stride " + Twine(Bytes) + " B");
        Found = true;
        break;
      }
      if (Found || Sites.size() >= 2)
        break;
    }
  }
}

// cp.async global->shared copies in F (FlashInfer's KV stream): inline-asm
// calls whose template mentions "cp.async" and "global". The GLOBAL source is
// the input operand constrained "l" (a 64-bit address register); the shared
// destination is "r" (a 32-bit smem offset) and the sizes are "n" immediates,
// so the single "l" input is unambiguously the gmem pointer. We only need the
// address to warm it -- a pure E0 hint -- so parsing the constraint string is
// enough (no PTX text parsing). Small cap: the innermost KV copy dominates.
void findAsyncStreamSites(Function &F, const Config &C,
                          SmallVectorImpl<AsyncSite> &Sites) {
  for (Instruction &I : instructions(F)) {
    auto *CI = dyn_cast<CallInst>(&I);
    if (!CI)
      continue;
    auto *IA = dyn_cast<InlineAsm>(CI->getCalledOperand());
    if (!IA)
      continue;
    StringRef Asm = IA->getAsmString();
    if (!Asm.contains("cp.async") || !Asm.contains("global"))
      continue;
    // Walk input constraints in order; the k-th input constraint binds call
    // arg k. Find the first "l" input -> the gmem source.
    StringRef Cons = IA->getConstraintString();
    SmallVector<StringRef, 8> Parts;
    Cons.split(Parts, ',', -1, /*KeepEmpty=*/false);
    int ArgIdx = 0, GmemArg = -1;
    for (StringRef P : Parts) {
      if (P.starts_with("=") || P.starts_with("~"))
        continue; // output / clobber: consumes no call arg
      if (P == "l" && GmemArg < 0)
        GmemArg = ArgIdx;
      ++ArgIdx;
    }
    if (GmemArg < 0 || (unsigned)GmemArg >= CI->arg_size())
      continue;
    Value *Gmem = CI->getArgOperand(GmemArg);
    if (!Gmem->getType()->isIntegerTy(64) && !Gmem->getType()->isPointerTy())
      continue;
    Sites.push_back({CI, Gmem});
    debugNote(C, "cp.async stream site in " + F.getName() +
                     " (global source -> warmable)");
    if (Sites.size() >= 4)
      break;
  }
}

class Weaver {
public:
  Weaver(Function &F, const Config &C) : F(F), M(*F.getParent()), C(C) {}

  bool run(LoopInfo *LI, ScalarEvolution *SE) {
    LIref = LI;
    bool IsBody = F.hasFnAttribute(kTaskBodyAttr);

    // -------- phase 0: detection while analyses are valid ----------------
    // Stream-load sites feed BOTH policy (prefetch ahead) AND shed (mask the
    // loaded value/score), so detect them whenever EITHER lever is on --
    // otherwise SCHED_NO_POLICY silently disables shed too (the levers must
    // be independent, per their SCHED_NO_* contract). Emission stays gated
    // per lever below: prefetch only under C.policy, shed only under C.shed.
    SmallVector<PrefetchSite, 4> Sites;
    SmallVector<AsyncSite, 4> AsyncSites;
    bool CtrlAvail = rtAvailable(M, C, kCtrlSym);
    if ((C.policy || C.shed) && CtrlAvail && LI && SE)
      findPrefetchSites(F, *LI, *SE, C, Sites);
    if (C.policy && CtrlAvail && LI && SE)
      // cp.async KV streams (FlashInfer): a separate handle -- the compute
      // reads from shared, so the only global-stream site is the async copy.
      // Prefetch-only, so gate on policy alone.
      findAsyncStreamSites(F, C, AsyncSites);

    // Pre-existing ctaid(slotAxis) calls -- collected BEFORE we emit our own.
    SmallVector<CallInst *, 4> OldPid;
    if (!IsBody) {
      Intrinsic::ID SlotID = ctaidIntrinsic(C.slotAxis);
      for (Instruction &I : instructions())
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (CI->getIntrinsicID() == SlotID)
            OldPid.push_back(CI);
    }

    // -------- phase 1: entry weaving --------------------------------------
    BasicBlock &EB = F.getEntryBlock();
    BasicBlock::iterator IP = EB.getFirstInsertionPt();
    while (IP != EB.end() && isa<AllocaInst>(&*IP))
      ++IP;
    if (IP == EB.end())
      return false; // degenerate entry; nothing to weave safely
    IRBuilder<> B(&EB, IP);

    bool TimerAvail = rtAvailable(M, C, kTimerSym);
    Value *T0 = (C.timer && TimerAvail) ? readClock64(B) : nullptr;
    // Latch the timer-off gate at ENTRY (not at each return): the flag value
    // AT LAUNCH decides whether THIS launch writes its timer row. Reading it
    // at the return (the old behavior) let the host race the flag -- flipping
    // it off mid-flight suppressed the probe's own epilogue write, and leaving
    // it on let off-cadence launches accumulate into the same rows, distorting
    // per-request cost by presence-duration under batch churn. With the entry
    // latch, a probe samples EXACTLY one step (host pushes flags-off after the
    // probe; the in-flight probe already latched ON). (#1 timer-race fix.)
    Value *TimerOff = T0 ? readTimerOff(B) : nullptr;

    Value *Task = nullptr;
    if (IsBody) {
      Task = F.getArg(F.arg_size() - 1);
    } else {
      Value *Pid = readSReg(B, ctaidIntrinsic(C.slotAxis));
      bool OrderAvail = rtAvailable(M, C, kTaskOrderSym);
      Task = (C.indirect && OrderAvail) ? emitRemap(B, Pid) : Pid;
      if (C.indirect && OrderAvail) {
        // A pre-existing ctaid call can BE the split-point instruction (it is
        // often the kernel's first op), so erase them BEFORE repositioning
        // the builder -- never leave B pointing at a freed instruction.
        for (CallInst *CI : OldPid) {
          CI->replaceAllUsesWith(Task);
          CI->eraseFromParent();
        }
        if (!OldPid.empty())
          debugNote(C, F.getName() + ": remapped " + Twine(OldPid.size()) +
                           " ctaid use(s) through task_order");
        BasicBlock *Tail = cast<Instruction>(Task)->getParent();
        B.SetInsertPoint(Tail, Tail->getFirstInsertionPt());
      }
    }

    PolicyFlags Flags;
    Value *Budget = nullptr;
    if (!Sites.empty() || !AsyncSites.empty()) {
      if (C.policy) // prefetch flags only when the policy lever is on
        Flags = emitPolicyFlags(B, Task);
      if (C.shed && !Sites.empty())
        Budget = emitBudget(B, Task); // tau -> per-task iteration cap
    }

    // -------- phase 2: body weaving ---------------------------------------
    // shed FIRST: it queries LoopInfo (LIref), and emitPrefetch below splits
    // the load's block (invalidating LoopInfo). shed itself adds only a
    // counter PHI + a select -- no block splits -- so LIref stays valid for it.
    //
    // shed = redirect the KV stream load to iteration 0's line once the
    // request's per-loop trip count exceeds its budget -- attend to FEWER
    // units (H2O/Quest epsilon-budgeted sparsity). Additive (the load stays,
    // its ADDRESS is select'd); dropped units re-read unit 0, a bounded
    // epsilon. NOT bit-exact by design -- the one lever that trades accuracy
    // for time; budget==0 (tau default) -> no redirect -> stock.
    if (Budget)
      emitShedAll(Sites, Budget);

    if (Flags.Urgent)
      for (PrefetchSite &S : Sites)
        emitPrefetch(S, Flags);

    // cp.async streams: warm the urgent request's global source ahead of the
    // async copy. Pure E0 hint (prefetch can only populate L2, never change
    // results); the polite/discard tier is DECLINED on an async source -- you
    // cannot safely bypass/evict a line an in-flight cp.async is still reading.
    if (Flags.Urgent)
      for (AsyncSite &A : AsyncSites)
        emitAsyncPrefetch(A, Flags.Urgent);

    if (T0)
      emitTimer(T0, TimerOff, Task);

    // PDL (sm_90+, opt-in): `wait` here -- AFTER the table reads above, BEFORE
    // the original body -- so when the launch site enables programmatic
    // stream serialization, the control-plane PCIe reads overlap the previous
    // kernel's tail and only the data-dependent body waits.
    // `launch_dependents` at each return lets the NEXT kernel's prologue
    // start under this one's epilogue. Both are E0: pure scheduling hints,
    // no-ops without the launch attribute.
    if (C.pdl && smVersion(F) >= 90 && !IsBody)
      emitPDL(B);

    F.addFnAttr(kInstrumentedAttr);
    return true;
  }

private:
  Function &F;
  Module &M;
  const Config &C;
  LoopInfo *LIref = nullptr;

  iterator_range<inst_iterator> instructions() { return llvm::instructions(F); }

  // task = (order != null && pid < maxTasks) ? order[pid] : pid
  // Splits the entry block; leaves the builder positioned in the tail block so
  // subsequent entry weaving stays ordered after the phi.
  Value *emitRemap(IRBuilder<> &B, Value *Pid) {
    Value *Armed = nullptr;
    Value *G = rtBuffer(B, C, kTaskOrderSym, &Armed); // already global-AS
    Value *InB = B.CreateICmpULT(Pid, B.getInt32(C.maxTasks));
    Value *Use = B.CreateAnd(Armed, InB);

    Instruction *SplitPt = &*B.GetInsertPoint();
    BasicBlock *Head = SplitPt->getParent();
    Instruction *ThenTerm =
        SplitBlockAndInsertIfThen(Use, SplitPt, /*Unreachable=*/false);
    IRBuilder<> TB(ThenTerm);
    // ORDER INDIRECTION (SCHED_ORDER_INDIRECT): the arena ORDER word holds a
    // retargetable pointer to the DEVICE order table (kernel-resident, so the
    // control loop installs device->device with no host sync). Load it; a 0
    // pointer (unarmed) forces an OOB index -> the identity fallback below.
    // Two selects, NO extra branch: reading the arena slot as i32 when unarmed
    // is a safe MAPPED read whose value the ChanOk select discards.
    Value *Base = G;
    Value *ChanOk = nullptr;
    if (C.orderIndirect) {
      Value *Ptr = TB.CreateLoad(TB.getInt64Ty(), G, "sched.order.chan");
      ChanOk = TB.CreateICmpNE(Ptr, TB.getInt64(0), "sched.order.chan.armed");
      Value *GInt = TB.CreatePtrToInt(G, TB.getInt64Ty());
      Value *SafePtr = TB.CreateSelect(ChanOk, Ptr, GInt, "sched.order.ptr");
      Base = TB.CreateIntToPtr(SafePtr,
                               PointerType::get(TB.getContext(), /*AS=*/1),
                               "sched.order.rows");
    }
    Value *GEP = TB.CreateGEP(TB.getInt32Ty(), Base, Pid);
    Value *Mapped = TB.CreateLoad(TB.getInt32Ty(), GEP, "sched.task.mapped");
    if (ChanOk) // unarmed pointer -> OOB sentinel -> identity (Safe picks Pid)
      Mapped = TB.CreateSelect(ChanOk, Mapped, TB.getInt32(C.maxTasks),
                               "sched.task.mapped.armed");
    // WRITE-RACE SAFETY, two layers (the table is read at EXECUTION time and
    // the host reprograms it between launches; an overlap scheduler races):
    //  1. PER-LAUNCH BOUND: clamp the mapped task to nctaid(slot); an
    //     out-of-range entry falls back to pid. Faults become impossible
    //     (measured before: cudaErrorIllegalAddress).
    //  2. WHOLE-LAUNCH VALIDITY (bijectivity): clamping a stale
    //     DIFFERENT-SIZE permutation would duplicate some tiles and drop
    //     others (exactly-once broken -> one stale token). ctrl->order_size
    //     stamps the tile count the permutation was built for; unless it
    //     equals nctaid, the ENTIRE launch takes identity -- uniform
    //     decision, bijective either way. order_size==0 = unchecked
    //     (legacy host-app mode; layer 1 still guards).
    Value *NTiles = readSReg(TB, nctaidIntrinsic(C.slotAxis));
    Value *SizeOK = ConstantInt::getTrue(TB.getContext());
    if (rtAvailable(M, C, kCtrlSym)) {
      Value *CArmed = nullptr;
      Value *Ctrl = rtBuffer(TB, C, kCtrlSym, &CArmed);
      Instruction *SzTerm = SplitBlockAndInsertIfThen(
          CArmed, &*TB.GetInsertPoint(), /*Unreachable=*/false);
      BasicBlock *SzHead = SzTerm->getParent()->getSinglePredecessor();
      IRBuilder<> SB(SzTerm);
      Value *OszP = SB.CreateGEP(SB.getInt8Ty(), Ctrl,
                                 SB.getInt64(kCtrlOrderSizeOff));
      Value *Osz = SB.CreateLoad(SB.getInt32Ty(), OszP, "sched.order.size");
      Value *Unchecked = SB.CreateICmpEQ(Osz, SB.getInt32(0));
      Value *Match = SB.CreateICmpEQ(Osz, NTiles);
      Value *OkThen = SB.CreateOr(Unchecked, Match, "sched.order.valid");
      BasicBlock *SzTail = TB.GetInsertPoint()->getParent();
      IRBuilder<> SP(SzTail, SzTail->begin());
      PHINode *SzPhi = SP.CreatePHI(SP.getInt1Ty(), 2, "sched.order.szok");
      SzPhi->addIncoming(OkThen, SzTerm->getParent());
      SzPhi->addIncoming(SP.getTrue(), SzHead); // ctrl unarmed -> unchecked
      SizeOK = SzPhi;
      TB.SetInsertPoint(SzTail, SzTail->getFirstInsertionPt());
    }
    Value *InGrid = TB.CreateICmpULT(Mapped, NTiles, "sched.task.ingrid");
    Value *Ok = TB.CreateAnd(InGrid, SizeOK, "sched.task.ok");
    Value *Safe = TB.CreateSelect(Ok, Mapped, Pid, "sched.task.safe");

    BasicBlock *Tail = SplitPt->getParent();
    IRBuilder<> PB(Tail, Tail->begin());
    PHINode *Phi = PB.CreatePHI(PB.getInt32Ty(), 2, "sched.task");
    Phi->addIncoming(Safe, ThenTerm->getParent());
    Phi->addIncoming(Pid, Head);

    // Do NOT reposition B on SplitPt here: SplitPt may be an old ctaid call
    // the caller is about to erase. The caller repositions after the phi.
    B.SetInsertPoint(Tail, Tail->getFirstInsertionPt());
    return Phi;
  }

  // Read the per-task policy row and derive the two uniform action flags:
  //   urgent = (hint==URGENT) | (hint==AUTO & q*dT - lambda_bw*dR - H > 0)
  //   polite = (hint==POLITE)
  // ctrl == null / OOB task -> both false (stock behavior).
  PolicyFlags emitPolicyFlags(IRBuilder<> &B, Value *Task) {
    Value *Armed = nullptr;
    Value *G = rtBuffer(B, C, kCtrlSym, &Armed); // global-AS ctrl base
    Value *InB = B.CreateICmpULT(Task, B.getInt32(C.maxTasks));
    Value *Use = B.CreateAnd(Armed, InB);

    Instruction *SplitPt = &*B.GetInsertPoint();
    BasicBlock *Head = SplitPt->getParent();
    Instruction *ThenTerm =
        SplitBlockAndInsertIfThen(Use, SplitPt, /*Unreachable=*/false);
    IRBuilder<> TB(ThenTerm);
    Type *F32 = TB.getFloatTy();
    Value *LamP = TB.CreateGEP(TB.getInt8Ty(), G,
                               TB.getInt64(kCtrlLambdaOff)); // lambda[0] = bw
    Value *Lam = TB.CreateLoad(F32, LamP, "sched.lambda.bw");
    Value *RowOff = TB.CreateAdd(
        TB.CreateMul(TB.CreateZExt(Task, TB.getInt64Ty()),
                     TB.getInt64(kCtrlRowSize)),
        TB.getInt64(kCtrlRowsOff));
    Value *QP = TB.CreateGEP(TB.getInt8Ty(), G, RowOff);
    Value *Q = TB.CreateLoad(F32, QP, "sched.q");
    Value *HintP = TB.CreateGEP(
        TB.getInt8Ty(), G,
        TB.CreateAdd(RowOff, TB.getInt64(kCtrlRowHintOff)));
    Value *Hint = TB.CreateLoad(TB.getInt8Ty(), HintP, "sched.hint");
    // score = q*dT - lambda*dR - H  (the marginal exchange)
    Value *Score = TB.CreateFSub(
        TB.CreateFSub(TB.CreateFMul(Q, ConstantFP::get(F32, C.dT)),
                      TB.CreateFMul(Lam, ConstantFP::get(F32, C.dR))),
        ConstantFP::get(F32, C.H));
    Value *ScoreOk =
        TB.CreateFCmpOGT(Score, ConstantFP::get(F32, 0.0), "sched.score.ok");
    Value *HintAuto = TB.CreateICmpEQ(Hint, TB.getInt8(kHintAuto));
    Value *HintUrgent = TB.CreateICmpEQ(Hint, TB.getInt8(kHintUrgent));
    Value *HintPolite = TB.CreateICmpEQ(Hint, TB.getInt8(kHintPolite));
    Value *UrgentThen = TB.CreateOr(
        HintUrgent, TB.CreateAnd(HintAuto, ScoreOk), "sched.urgent.raw");

    BasicBlock *Tail = SplitPt->getParent();
    IRBuilder<> PB(Tail, Tail->begin());
    PHINode *UrgentPhi = PB.CreatePHI(PB.getInt1Ty(), 2, "sched.urgent");
    UrgentPhi->addIncoming(UrgentThen, ThenTerm->getParent());
    UrgentPhi->addIncoming(PB.getFalse(), Head);
    PHINode *PolitePhi = PB.CreatePHI(PB.getInt1Ty(), 2, "sched.polite");
    PolitePhi->addIncoming(HintPolite, ThenTerm->getParent());
    PolitePhi->addIncoming(PB.getFalse(), Head);

    B.SetInsertPoint(SplitPt);
    return {UrgentPhi, PolitePhi};
  }

  // budget = ctrl != null && task < max ? (tau ? tau : UINT32_MAX) : UINT32_MAX
  // tau is the per-task row field (u16); 0 (the fail-safe default) -> no cap.
  Value *emitBudget(IRBuilder<> &B, Value *Task) {
    Value *Armed = nullptr;
    Value *G = rtBuffer(B, C, kCtrlSym, &Armed); // global-AS ctrl base
    Value *InB = B.CreateICmpULT(Task, B.getInt32(C.maxTasks));
    Value *Use = B.CreateAnd(Armed, InB);

    Instruction *SplitPt = &*B.GetInsertPoint();
    BasicBlock *Head = SplitPt->getParent();
    Instruction *ThenTerm =
        SplitBlockAndInsertIfThen(Use, SplitPt, /*Unreachable=*/false);
    IRBuilder<> TB(ThenTerm);
    Value *RowOff = TB.CreateAdd(
        TB.CreateMul(TB.CreateZExt(Task, TB.getInt64Ty()),
                     TB.getInt64(kCtrlRowSize)),
        TB.getInt64(kCtrlRowsOff));
    Value *TauP = TB.CreateGEP(
        TB.getInt8Ty(), G, TB.CreateAdd(RowOff, TB.getInt64(kCtrlRowTauOff)));
    Value *Tau = TB.CreateZExt(TB.CreateLoad(TB.getInt16Ty(), TauP), // u16
                               TB.getInt32Ty(), "sched.tau");
    Value *NoCap = TB.CreateICmpEQ(Tau, TB.getInt32(0));
    Value *BThen = TB.CreateSelect(NoCap, TB.getInt32(0xFFFFFFFFu), Tau);

    BasicBlock *Tail = SplitPt->getParent();
    IRBuilder<> PB(Tail, Tail->begin());
    PHINode *Phi = PB.CreatePHI(PB.getInt32Ty(), 2, "sched.budget");
    Phi->addIncoming(BThen, ThenTerm->getParent());
    Phi->addIncoming(PB.getInt32(0xFFFFFFFFu), Head); // unarmed -> no cap
    B.SetInsertPoint(SplitPt);
    return Phi;
  }

  // The loop's per-iteration trip counter as an i32 that is 0 on the first
  // iteration and increments by 1. Rather than INJECT a counter (fragile: at
  // OptimizerLast the loop is unrolled/rotated, so a hand-built PHI over the
  // header's predecessors is easily unsound -- the source of every past shed
  // miscompile), we REUSE the loop's canonical induction variable when it has
  // one. That IV is already valid SSA that dominates the whole body, so the
  // mask select built from it is always well-formed. If the loop has no
  // canonical IV (e.g. fully unrolled, or a non-{0,+,1} form), we DECLINE --
  // shed does not fire on that site (correct-or-absent, never wrong; the eKV
  // rule). Returns the IV coerced to i32, or null to decline.
  Value *shedCounter(Loop *L, IRBuilder<> &AtHeaderTerm) {
    PHINode *IV = L->getCanonicalInductionVariable();
    if (!IV)
      return nullptr;
    if (IV->getType()->isIntegerTy(32))
      return IV;
    if (IV->getType()->getIntegerBitWidth() > 32)
      return AtHeaderTerm.CreateTrunc(IV, AtHeaderTerm.getInt32Ty(),
                                      "sched.shed.ctr");
    return AtHeaderTerm.CreateZExt(IV, AtHeaderTerm.getInt32Ty(),
                                   "sched.shed.ctr");
  }

  // Coordinate shed across ALL stream sites. Two drop semantics, both PURE
  // ARITHMETIC on the loaded value / score -- NEVER on the address (address
  // arithmetic risks OOB and depends on a fragile per-loop trip count):
  //   * SOFTMAX kernel (some site's value feeds an online-softmax score): mask
  //     the SCORE to -inf (exp(-inf-m)=0). The dropped token contributes EXACTLY
  //     zero WEIGHT, zeroing every stream's contribution -- so non-score loads
  //     (V) are left entirely alone.
  //   * LINEAR kernel (no softmax anywhere): mask the loaded VALUE to 0
  //     (v' = keep ? v : 0). For a contraction sum += v*w this zeroes the
  //     dropped term exactly. The demand read still happens at its valid
  //     address (bandwidth is not saved, but that is the honest cost of a
  //     safe, additive shed; a bandwidth-saving variant needs a proper
  //     predicated load, a later tier).
  void emitShedAll(ArrayRef<PrefetchSite> Sites, Value *Budget) {
    bool Softmax = false;
    for (const PrefetchSite &S : Sites)
      if (findSoftmaxScore(S.Load)) {
        Softmax = true;
        break;
      }
    for (const PrefetchSite &S : Sites) {
      LoadInst *Ld = S.Load;
      Loop *L = LIref->getLoopFor(Ld->getParent());
      if (!L) {
        debugNote(C, F.getName() + ": shed skipped -- load not in a loop");
        continue;
      }
      Value *Score = findSoftmaxScore(Ld);
      if (Softmax && !Score)
        continue; // V-like load in a softmax kernel: weight already masked
      // The counter is the loop's canonical IV (a header PHI dominating the
      // whole body); a trunc/zext of it is placed at the header terminator.
      // shedCounter returns null on a loop with no canonical IV -> decline.
      IRBuilder<> HB(L->getHeader()->getTerminator());
      Value *Ctr = shedCounter(L, HB);
      if (!Ctr) {
        debugNote(C, F.getName() + ": shed DECLINED -- loop has no canonical "
                                   "induction variable (unrolled/rotated); "
                                   "correct-or-absent, do not shed this site");
        continue;
      }
      if (Score) {
        if (emitShedScoreMask(Ld, Ctr, Budget))
          debugNote(C, F.getName() + ": shed softmax score mask woven (-inf)");
        continue;
      }
      // linear: v' = keep ? v : 0. Keep is built AT the select's insertion
      // point (right after the load) from Ctr -- Ctr dominates it, and the
      // select then dominates exactly the uses it replaces (DT-checked).
      if (!Ld->getType()->isFloatingPointTy())
        continue; // only float streams contribute to a contraction
      Instruction *SelIP = Ld->getNextNode();
      if (!SelIP)
        continue;
      IRBuilder<> AB(SelIP);
      Value *Keep = AB.CreateICmpULT(Ctr, Budget, "sched.keep");
      Value *Zero = ConstantFP::get(Ld->getType(), 0.0);
      auto *Sel =
          cast<Instruction>(AB.CreateSelect(Keep, Ld, Zero, "sched.shed.val"));
      DominatorTree DT(F);
      Ld->replaceUsesWithIf(Sel, [&](Use &U) {
        return U.getUser() != Sel && DT.dominates(Sel, U);
      });
      debugNote(C, F.getName() + ": shed linear value-mask woven (->0)");
    }
  }

  // Is `I` an exp-family call? Covers libdevice (__nv_expf/__nv_fast_expf/
  // expf...), llvm.exp/exp2, and the PTX fast path llvm.nvvm.ex2.approx.*.
  // Excludes the false friends frexp/ldexp (the eKV lesson).
  static bool isExpCall(Instruction *I) {
    auto *CI = dyn_cast<CallInst>(I);
    if (!CI)
      return false;
    switch (CI->getIntrinsicID()) {
    case Intrinsic::exp:
    case Intrinsic::exp2:
#if LLVM_VERSION_MAJOR >= 22
    case Intrinsic::nvvm_ex2_approx:      // type-overloaded since LLVM 22
    case Intrinsic::nvvm_ex2_approx_ftz:
#else
    case Intrinsic::nvvm_ex2_approx_f:
    case Intrinsic::nvvm_ex2_approx_d:
#endif
      return true;
    default:
      break;
    }
    Function *Callee = CI->getCalledFunction();
    if (!Callee)
      return false;
    StringRef N = Callee->getName();
    return N.contains("exp") && !N.contains("frexp") && !N.contains("ldexp");
  }

  // The running-max update of online softmax: llvm.maxnum, OR a libdevice
  // fmax call (clang lowers fmaxf to __nv_fmaxf, not always llvm.maxnum), OR
  // the fcmp+select form (fmax open-coded). Detected structurally.
  static bool isMaxCall(CallInst *CI) {
    if (CI->getIntrinsicID() == Intrinsic::maxnum ||
        CI->getIntrinsicID() == Intrinsic::nvvm_fmax_f)
      return true;
    Function *F = CI->getCalledFunction();
    if (!F)
      return false;
    StringRef N = F->getName();
    return (N.contains("fmax") || N.contains("__nv_fmax")) &&
           !N.contains("fmaxnan");
  }

  // Does `V` reach an exp call's operand within a few arithmetic steps?
  static bool reachesExp(Value *V, unsigned Depth = 12) {
    SmallVector<Value *, 8> Work{V};
    SmallPtrSet<Value *, 16> Seen;
    unsigned Steps = 0;
    while (!Work.empty() && Steps++ < 64) {
      Value *Cur = Work.pop_back_val();
      if (!Seen.insert(Cur).second)
        continue;
      for (User *U : Cur->users()) {
        auto *I = dyn_cast<Instruction>(U);
        if (!I)
          continue;
        if (isExpCall(I))
          return true;
        if (isa<BinaryOperator>(I) || isa<CastInst>(I) || isa<SelectInst>(I))
          Work.push_back(I);
      }
    }
    return false;
  }

  // The online-softmax SCORE downstream of the stream load: a value derived
  // from the load (crossing warp shuffles: the dot-product reduction) that is
  // consumed BOTH by llvm.maxnum (the running-max update) and, transitively,
  // by an exp call (the weight). That pair is the online-softmax signature --
  // the same "contraction that flows into exp" role eKV keys on at TTIR,
  // recovered here from scalar dataflow.
  Value *findSoftmaxScore(LoadInst *Ld) {
    SmallVector<Value *, 16> Work{Ld};
    SmallPtrSet<Value *, 32> Seen;
    unsigned Steps = 0;
    while (!Work.empty() && Steps++ < 256) {
      Value *Cur = Work.pop_back_val();
      if (!Seen.insert(Cur).second)
        continue;
      for (User *U : Cur->users()) {
        auto *I = dyn_cast<Instruction>(U);
        if (!I)
          continue;
        // The running max consumes the score in one of three forms:
        //   llvm.maxnum / __nv_fmax call, OR an ordered fcmp (open-coded fmax:
        //   select(fcmp ogt, ...)). In every form, if Cur also reaches an exp
        //   it is the softmax score.
        bool feedsMax = false;
        if (auto *CI = dyn_cast<CallInst>(I))
          feedsMax = isMaxCall(CI);
        else if (auto *FC = dyn_cast<FCmpInst>(I))
          feedsMax = FC->isRelational();
        if (feedsMax && reachesExp(Cur))
          return Cur;
        if (auto *CI = dyn_cast<CallInst>(I)) {
          // Cross warp shuffles (the reduction) but nothing else.
          if (Function *Callee = CI->getCalledFunction())
            if (Callee->getName().starts_with("llvm.nvvm.shfl"))
              Work.push_back(CI);
          continue;
        }
        if (isa<BinaryOperator>(I) || isa<CastInst>(I) || isa<SelectInst>(I))
          Work.push_back(I);
      }
    }
    return nullptr;
  }

  // s' = select(keep, s, -inf), replacing every prior use of s (running max,
  // exp chains). Requires s in the load's block so Keep (defined at the load)
  // dominates -- true for straight-line online-softmax bodies; anything else
  // declines loudly and the kernel keeps redirect-only shed semantics OFF the
  // menu (the control plane must not set tau on it).
  bool emitShedScoreMask(LoadInst *Ld, Value *Ctr, Value *Budget) {
    Value *Score = findSoftmaxScore(Ld);
    if (!Score)
      return false; // linear kernel: redirect alone is exact
    auto *SI = dyn_cast<Instruction>(Score);
    if (!SI)
      return false;
    Instruction *SelIP = SI->getNextNode();
    if (!SelIP)
      return false;
    // Build Keep and the select AT the score's own position, from Ctr (which
    // dominates it): s' = keep ? s : -inf. Then replace only the score uses
    // the select provably dominates (DT-checked) -- always valid SSA.
    IRBuilder<> B(SelIP);
    Value *Keep = B.CreateICmpULT(Ctr, Budget, "sched.keep");
    Value *NegInf = ConstantFP::getInfinity(SI->getType(), /*Negative=*/true);
    auto *Sel =
        cast<Instruction>(B.CreateSelect(Keep, SI, NegInf, "sched.shed.score"));
    DominatorTree DT(F);
    SI->replaceUsesWithIf(Sel, [&](Use &U) {
      return U.getUser() != Sel && DT.dominates(Sel, U);
    });
    return true;
  }

  // Per-request cache differentiation, both additive (the demand load is never
  // touched) and CTA-uniform (the flags are task-uniform -> no divergence):
  //
  //   URGENT (latency-critical short request): read ahead AND pin --
  //     prefetch.global.L2::evict_last [addr + D*stride]. The demand load then
  //     hits a resident, low-eviction-priority line.
  //
  //   POLITE (long streaming request): stop polluting shared L2 --
  //     discard.global.L2 [line(addr)], 128 emitted AFTER the demand load, so
  //     the value is already in registers; the discard tells L2 the just-read
  //     KV line is dead (no reuse), freeing the set for latency-critical
  //     requests. Bit-exact (read completed; discard only drops a cache copy,
  //     re-fetched from HBM if ever needed again). Verified by the probe to
  //     assemble on sm_86 AND sm_120. This is the real "cache bypass" lever;
  //     it replaces the previous weak plain-prefetch polite tier.
  //
  // On sm < 80 (no L2 priority / discard) urgent collapses to prefetch.L2 and
  // polite is a no-op. PTX drops prefetch/discard on invalid addresses, so
  // read-ahead / discard past the last block is benign.
  void emitPrefetch(PrefetchSite &S, PolicyFlags Flags) {
    bool HasL2Ctl = smVersion(F) >= 80;

    // Predicated prefetch-ahead BEFORE the load (urgent pin, or sm<80 either).
    auto emitPrefetchAhead = [&](Value *Flag, const char *Asm) {
      if (!Flag)
        return;
      IRBuilder<> B(S.Load);
      Value *P = S.Load->getPointerOperand();
      Value *Ahead = B.CreateGEP(
          B.getInt8Ty(), P, B.getInt64(S.StepBytes * (int64_t)C.prefetchDist),
          "sched.pf.ahead");
      Instruction *ThenTerm =
          SplitBlockAndInsertIfThen(Flag, S.Load, /*Unreachable=*/false);
      IRBuilder<> TB(ThenTerm);
      FunctionType *FT =
          FunctionType::get(TB.getVoidTy(), {Ahead->getType()}, false);
      InlineAsm *IA = InlineAsm::get(FT, Asm, "l", /*hasSideEffects=*/true);
      TB.CreateCall(FT, IA, {Ahead});
    };
    // Predicated discard AFTER the load (polite bypass): the line is dead post
    // read. Insert at the load's next insertion point so the value dominates.
    auto emitDiscardAfter = [&](Value *Flag) {
      if (!Flag)
        return;
      Instruction *After = S.Load->getNextNode();
      if (!After)
        return;
      Value *P = S.Load->getPointerOperand();
      Instruction *ThenTerm =
          SplitBlockAndInsertIfThen(Flag, After, /*Unreachable=*/false);
      IRBuilder<> TB(ThenTerm);
      // PTX ISA: `discard.global.L2 [a], 128` REQUIRES `a` 128-byte aligned
      // (unaligned -> UB). Align the demand-load pointer DOWN to the L2 line.
      // Safety: KV is READ-ONLY during decode (written once at prefill, DRAM
      // authoritative), so dropping the L2 copy is bit-exact -- a re-read
      // re-fetches identical bytes; discard has no writeback and cannot change
      // a value. CAVEAT (why this stays behind the conservative `Polite`
      // flag): under GQA multiple q-heads share one request's KV, so an early
      // discard can force a sibling head to re-fetch from DRAM -- a PERF risk,
      // never a correctness one. (Reviewer flagged the missing alignment.)
      Value *Pi = TB.CreatePtrToInt(P, TB.getInt64Ty());
      Value *Aligned = TB.CreateAnd(Pi, TB.getInt64(~INT64_C(127)));
      Value *AP = TB.CreateIntToPtr(Aligned, P->getType());
      FunctionType *FT =
          FunctionType::get(TB.getVoidTy(), {AP->getType()}, false);
      InlineAsm *IA = InlineAsm::get(FT, "discard.global.L2 [$0], 128;", "l",
                                     /*hasSideEffects=*/true);
      TB.CreateCall(FT, IA, {AP});
    };

    if (HasL2Ctl) {
      emitPrefetchAhead(Flags.Urgent, "prefetch.global.L2::evict_last [$0];");
      emitDiscardAfter(Flags.Polite);
    } else {
      IRBuilder<> B(S.Load);
      Value *Any = B.CreateOr(Flags.Urgent, Flags.Polite);
      emitPrefetchAhead(Any, "prefetch.L2 [$0];");
    }
  }

  // Warm a cp.async global source when the request is urgent: a predicated
  // `prefetch.global.L2::evict_last [gmem]` right before the async copy. This
  // pins the urgent request's KV in L2 so its next tiles hit. Pure E0 hint --
  // no correctness effect, no address arithmetic (the SAME line, warmed early).
  void emitAsyncPrefetch(AsyncSite &A, Value *Urgent) {
    if (smVersion(F) < 80)
      return; // no L2 prefetch control
    Value *G = A.Gmem;
    Instruction *ThenTerm =
        SplitBlockAndInsertIfThen(Urgent, A.Copy, /*Unreachable=*/false);
    IRBuilder<> TB(ThenTerm);
    if (!G->getType()->isIntegerTy(64))
      G = TB.CreatePtrToInt(G, TB.getInt64Ty(), "sched.async.gmem");
    FunctionType *FT = FunctionType::get(TB.getVoidTy(), {G->getType()}, false);
    InlineAsm *IA = InlineAsm::get(
        FT, "prefetch.global.L2::evict_last [$0];", "l", /*hasSideEffects=*/true);
    TB.CreateCall(FT, IA, {G});
    debugNote(C, F.getName() + ": cp.async urgent L2 warm woven");
  }

  // griddepcontrol weave: wait at the current insert point (post-table-read),
  // launch_dependents before every return. Executed by every thread; the
  // hardware treats the CTA as signalled once all its threads arrive.
  void emitPDL(IRBuilder<> &B) {
    auto emitAsm = [](IRBuilderBase &IB, const char *Str) {
      FunctionType *FT = FunctionType::get(IB.getVoidTy(), false);
      IB.CreateCall(FT, InlineAsm::get(FT, Str, "", /*hasSideEffects=*/true));
    };
    emitAsm(B, "griddepcontrol.wait;");
    for (BasicBlock &BB : F)
      if (auto *RI = dyn_cast<ReturnInst>(BB.getTerminator())) {
        IRBuilder<> RB(RI);
        emitAsm(RB, "griddepcontrol.launch_dependents;");
      }
    debugNote(C, F.getName() + ": PDL overlap points woven (wait after table "
                               "reads; launch_dependents at returns)");
  }

  // Per return: dur = clock64 - t0; if (tid==0 && timer && task < max
  //                                      && !(ctrl->flags & TIMER_OFF))
  //   atomicrmw add timer[task], dur   (commutative monoid: replay-safe)
  //
  // The flags gate is the PER-STEP observation switch: in the baked ABI the
  // armed flag is a compile-time constant, so nulling the slot cannot disarm
  // the timer at runtime -- but the control plane can set ctrl->flags bit0
  // (+push) to suppress the PCIe atomic on non-probe steps (the timer costs
  // ~+5.6% at serving scale, +181% on tiny kernels -- sample it, don't pay it
  // every step). flags==0 (the zeroed default) keeps the timer on: fail-open
  // to the historical behavior. The flags load itself derefs ctrl, so it is
  // nested UNDER the ctrl-armed check (host ABI: unarmed ctrl -> no deref,
  // timer fires as before).
  // Read the timer-off gate (ctrl.flags bit0), guarded for a possibly-null
  // ctrl (the named-global path). Returns an i1 that is TRUE when the timer is
  // gated OFF. Called at ENTRY so the value latches the flag at LAUNCH time
  // (see the #1 timer-race fix note at the entry weave).
  Value *readTimerOff(IRBuilder<> &B) {
    if (!rtAvailable(M, C, kCtrlSym))
      return B.getFalse();
    Value *CArmed = nullptr;
    Value *Ctrl = rtBuffer(B, C, kCtrlSym, &CArmed);
    Instruction *SplitPt = &*B.GetInsertPoint();
    Instruction *FlagTerm =
        SplitBlockAndInsertIfThen(CArmed, SplitPt, /*Unreachable=*/false);
    BasicBlock *FlagHead = FlagTerm->getParent()->getSinglePredecessor();
    IRBuilder<> FB(FlagTerm);
    Value *FlagsP =
        FB.CreateGEP(FB.getInt8Ty(), Ctrl, FB.getInt64(kCtrlFlagsOff));
    Value *Flags = FB.CreateLoad(FB.getInt32Ty(), FlagsP, "sched.flags");
    Value *OffThen = FB.CreateICmpNE(
        FB.CreateAnd(Flags, FB.getInt32(kCtrlFlagTimerOff)), FB.getInt32(0),
        "sched.timer.off");
    BasicBlock *Tail = SplitPt->getParent();
    IRBuilder<> PB(Tail, Tail->begin());
    PHINode *Off = PB.CreatePHI(PB.getInt1Ty(), 2, "sched.timer.gate");
    Off->addIncoming(OffThen, FlagTerm->getParent());
    Off->addIncoming(PB.getFalse(), FlagHead);
    B.SetInsertPoint(SplitPt);
    return Off;
  }

  void emitTimer(Value *T0, Value *TimerOff, Value *Task) {
    SmallVector<ReturnInst *, 4> Rets;
    for (BasicBlock &BB : F)
      if (auto *RI = dyn_cast<ReturnInst>(BB.getTerminator()))
        Rets.push_back(RI);
    for (ReturnInst *RI : Rets) {
      IRBuilder<> B(RI);
      Value *T1 = readClock64(B);
      Value *Dur = B.CreateSub(T1, T0, "sched.cycles");
      Value *Tz = threadIsZero(B);
      Value *Armed = nullptr;
      Value *G = rtBuffer(B, C, kTimerSym, &Armed); // global-AS timer base
      Value *InB = B.CreateICmpULT(Task, B.getInt32(C.maxTasks));
      // Gate on the ENTRY-latched timer-off, not a fresh flag read: flipping
      // ctrl.flags mid-flight cannot change THIS launch's decision (#1 fix).
      Value *On = TimerOff ? B.CreateNot(TimerOff, "sched.timer.on")
                           : B.getTrue();
      Value *Ok =
          B.CreateAnd(B.CreateAnd(B.CreateAnd(Tz, Armed), InB), On);
      Instruction *ThenTerm =
          SplitBlockAndInsertIfThen(Ok, RI, /*Unreachable=*/false);
      IRBuilder<> TB(ThenTerm);
      Value *Base = G;
      if (C.timerIndirect) {
        // Channel select: the slot's word 0 holds the row-table address the
        // host retargets per process (device buffer or host-mapped table).
        // 0 (zeroed arena) -> off. The address is a LOADED value -- an
        // opaque register, immune to the ptxas immediate-factoring hazard
        // the baked-slot scheme exists to avoid.
        Value *Chan =
            TB.CreateLoad(TB.getInt64Ty(), G, "sched.timer.chan");
        Value *ChanOk = TB.CreateICmpNE(Chan, TB.getInt64(0),
                                        "sched.timer.chan.armed");
        ThenTerm = SplitBlockAndInsertIfThen(ChanOk, ThenTerm,
                                             /*Unreachable=*/false);
        TB.SetInsertPoint(ThenTerm);
        Base = TB.CreateIntToPtr(
            Chan, PointerType::get(TB.getContext(), /*AS=*/1),
            "sched.timer.rows");
      }
      Value *GEP = TB.CreateGEP(TB.getInt64Ty(), Base,
                                TB.CreateZExt(Task, TB.getInt64Ty()));
      TB.CreateAtomicRMW(AtomicRMWInst::Add, GEP, Dur, MaybeAlign(8),
                         AtomicOrdering::Monotonic);
    }
  }
};

} // namespace

PreservedAnalyses SchedWeavePass::run(Module &M, ModuleAnalysisManager &AM) {
  if (!isNVPTX(M))
    return PreservedAnalyses::all();
  Config C = Config::fromEnv();

  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    bool IsBody = F.hasFnAttribute(kTaskBodyAttr);
    if (!IsBody && !isKernel(F))
      continue;
    if (F.hasFnAttribute(kWqDriverAttr) || F.hasFnAttribute(kInstrumentedAttr))
      continue;
    LoopInfo *LI = nullptr;
    ScalarEvolution *SE = nullptr;
    // BOTH policy (prefetch-site detection) and shed (stream-load detection +
    // canonical-IV counter) need LoopInfo/SCEV. Compute them when EITHER
    // lever is on, or shed is silently disabled by SCHED_NO_POLICY (the
    // levers must honor their independent SCHED_NO_* contracts).
    if (C.policy || C.shed) {
      LI = &FAM.getResult<LoopAnalysis>(F);
      SE = &FAM.getResult<ScalarEvolutionAnalysis>(F);
    }
    debugNote(C, "weaving " + F.getName() +
                     (IsBody ? " (task body)" : " (kernel)"));
    Changed |= Weaver(F, C).run(LI, SE);
    if (Changed) {
      FAM.invalidate(F, PreservedAnalyses::none());
    }
  }
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

} // namespace sched
