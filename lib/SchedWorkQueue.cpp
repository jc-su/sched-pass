//===- SchedWorkQueue.cpp - persistent-worker transform (the CLC layer) --===//
//
// The task-acquisition layer of the scheduling model:
//
//     j = C_m(worker, t; pi)        m in { static, ticket, CLC }
//
// On Blackwell (sm_100+), m = CLC: a worker CTA claims a not-yet-launched
// block with clusterlaunchcontrol.try_cancel and takes over its ctaid. This
// box is sm_86, so this pass implements m = ticket, the software substitute
// with the same shape: transform each selected kernel into a persistent
// worker that (1) serves its own ctaid first, then (2) claims further logical
// tasks from a global atomic ticket counter until tasks run out. The
// task_order indirection (pi) is applied to every claimed raw id, so the
// control plane's priority ordering governs BOTH the static prefix and the
// dynamically claimed tail -- exactly the role pi plays for CLC on Blackwell.
//
// Transform (compile-time opt-in: SCHED_WORKQUEUE=1 [+ SCHED_WQ_KERNELS
// name filter]; the launch site must cooperate by launching W <= num_tasks
// worker CTAs and priming the ticket counter to W -- see runtime/sched_rt.h):
//
//   __global__ k(args...)  ==>   private void k.sched_body(args..., i32 task)
//                                __global__ k(args...):
//                                  if (!queue || !ctrl)          // stock path
//                                    { k.sched_body(args..., ctaid); return; }
//                                  ntasks = ctrl->num_tasks
//                                  if (ntasks == 0)              // fail-safe +
//                                    { k.sched_body(args..., ctaid); return; }
//                                    // ^ per-step disarm: see below
//                                  raw = ctaid; if (raw >= ntasks) return;
//                                  loop:
//                                    task = task_order ? task_order[raw] : raw
//                                    k.sched_body(args..., task)
//                                    barrier0
//                                    if (tid == 0)
//                                      shared.next = atomicAdd(queue, 1)
//                                    barrier0
//                                    raw = shared.next
//                                    if (raw < ntasks) goto loop
//
// The `ntasks == 0 -> stock` branch is load-bearing twice over:
//   * BAKED-ABI FAIL-SAFE: baked slots are inttoptr constants, so the pointer
//     null-check can never fire -- an unprogrammed (zeroed) arena would
//     otherwise make every block exit at `raw >= 0` and silently drop the
//     whole launch. num_tasks==0 (the zeroed default) must mean STOCK.
//   * PER-STEP ARMING SWITCH: the control plane toggles dynamic acquisition
//     per step by WRITING num_tasks (N = claim loop on, 0 = stock static),
//     no recompile and -- in CLC mode, where grid == tasks either way -- no
//     launch-site change. This is how the uncertainty-gated CLC policy arms.
//
// The body clone carries "sched-task-body", so SchedWeavePass then weaves the
// timer/policy against the task PARAMETER (per-task attribution even though
// one CTA now serves many tasks). The driver is marked "sched-wq-driver" and
// is never woven itself.
//
// --- CLC on Blackwell (m = CLC, see emitClaimCLC) -----------------------
// With SCHED_CLC=1 and sm_100+ the claim block is the CLC sequence (PTX ISA
// 8.7, CUDA 12.9 "Work Stealing with Cluster Launch Control"): try_cancel
// steals a not-yet-launched block of THIS grid, so the launch site launches
// grid == tasks and needs NO ticket counter -- the hardware is the queue.
// CLC mode therefore requires only `ctrl` (num_tasks); `queue` is unused.
//
// CLAIM PLACEMENT IS A MEASURED CONTRACT -- LATE BINDING ONLY. The claim is
// issued AFTER the body completes (issue+collect together), never ahead:
// try_cancel removes a block from the launch pool at ISSUE time, so issuing
// the next claim before the body (the tempting "overlap the async cancel"
// pattern) reserves a task while this worker is still busy and idle workers
// cannot steal it -- head-of-line blocking behind stragglers. Measured on
// RTX PRO 6000 (sm_120), experiments/clc/clc_pipeline_probe.cu: claim-ahead
// flips the heterogeneous-batch result from -19% (win) to +21% (loss), even
// though it does hide the ~900-cycle collect wait (~110 when overlapped).
// Work-stealing's value IS the late bind; do not "optimize" this into a
// prologue issue.
//
// The claim source changes (hardware queue of unlaunched blocks vs. a ticket
// counter); pi, the policy weave, and the timer are IDENTICAL. That interface
// stability is the point: emitClaim() is the single function to swap.
//
//===----------------------------------------------------------------------===//
#include "SchedUtil.h"
#include "sched/SchedPasses.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <cstdlib>
#include <string>

using namespace llvm;

namespace sched {

namespace {

bool nameSelected(StringRef Name) {
  const char *Filter = std::getenv("SCHED_WQ_KERNELS");
  if (!Filter || !*Filter)
    return true; // no filter: all kernels
  StringRef List(Filter);
  SmallVector<StringRef, 4> Parts;
  List.split(Parts, ',', -1, /*KeepEmpty=*/false);
  for (StringRef P : Parts)
    if (Name.contains(P.trim()))
      return true;
  return false;
}

class WorkQueueTransform {
public:
  WorkQueueTransform(Function &F, const Config &C)
      : F(F), M(*F.getParent()), Ctx(F.getContext()), C(C) {}

  bool run() {
    // Both ABIs: named globals (sched_rt.h TU) OR baked env addresses
    // (SCHED_BAKE_* -- the FlashInfer JIT path). CLC (sm_100+) has no ticket
    // counter -- the hardware queue of unlaunched blocks IS the queue -- so
    // it requires only ctrl (num_tasks); ticket mode requires queue + ctrl.
    UseCLC = C.clc && smVersion(F) >= 100;
    if (!rtAvailable(M, C, kCtrlSym) ||
        (!UseCLC && !rtAvailable(M, C, kQueueSym))) {
      debugNote(C, F.getName() +
                       (UseCLC ? ": no ctrl (named global or baked address)"
                               : ": no queue/ctrl (named global or baked "
                                 "address)") +
                       " -> work-queue transform SKIPPED loudly");
      return false;
    }

    Function *Body = cloneBody();
    if (!Body)
      return false;
    buildDriver(Body);
    debugNote(C, F.getName() + ": persistent-worker transform applied (" +
                     (UseCLC ? "CLC claim, ctrl-only" : "ticket claim") + ")");
    return true;
  }

private:
  Function &F;
  Module &M;
  LLVMContext &Ctx;
  const Config &C;
  bool UseCLC = false; // SCHED_CLC && sm_100+: claim via CLC, queue unused

  // Clone the kernel into a private body function with one extra i32 `task`
  // parameter; inside the clone, every ctaid(slotAxis) read becomes `task`.
  Function *cloneBody() {
    FunctionType *FT = F.getFunctionType();
    SmallVector<Type *, 8> Params(FT->params().begin(), FT->params().end());
    Params.push_back(Type::getInt32Ty(Ctx));
    FunctionType *BT =
        FunctionType::get(Type::getVoidTy(Ctx), Params, /*isVarArg=*/false);
    Function *Body = Function::Create(BT, GlobalValue::InternalLinkage,
                                      F.getName() + ".sched_body", M);

    ValueToValueMapTy VMap;
    auto NewArg = Body->arg_begin();
    for (Argument &A : F.args()) {
      NewArg->setName(A.getName());
      VMap[&A] = &*NewArg++;
    }
    Body->getArg(Body->arg_size() - 1)->setName("sched.task");

    SmallVector<ReturnInst *, 4> Rets;
    CloneFunctionInto(Body, &F, VMap, CloneFunctionChangeType::LocalChangesOnly,
                      Rets);
    // The clone copied F's attributes; the body is a plain device function.
    Body->setCallingConv(CallingConv::C);
    Body->addFnAttr(Attribute::NoInline); // keep the per-task call boundary
    Body->addFnAttr(kTaskBodyAttr);

    // ctaid(slotAxis) -> task param.
    Value *TaskArg = Body->getArg(Body->arg_size() - 1);
    Intrinsic::ID SlotID = ctaidIntrinsic(C.slotAxis);
    SmallVector<CallInst *, 4> Olds;
    for (Instruction &I : instructions(*Body))
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (CI->getIntrinsicID() == SlotID)
          Olds.push_back(CI);
    for (CallInst *CI : Olds) {
      CI->replaceAllUsesWith(TaskArg);
      CI->eraseFromParent();
    }
    return Body;
  }

  // Claim the next raw task id, broadcast CTA-wide via a shared slot. Two
  // realizations of the SAME interface (this is the model's task-acquisition
  // layer C_m); callers -- pi remap, body call, done test -- are identical.
  // LATE BINDING: called strictly AFTER the body (see the header contract;
  // claim-ahead measurably breaks load balancing).
  // Ntasks = the loop-entry num_tasks SNAPSHOT (one launch, one value). The CLC
  // fallback must use it, NOT a fresh reload: the loop terminates on
  // `Next >= Ntasks` (entry snapshot), so a failed-claim fallback loaded fresh
  // could return a DIFFERENT num_tasks (if the host rewrote it mid-launch) and
  // desync termination -- the invariant the reviewer flagged.
  Value *emitClaim(IRBuilder<> &B, Value *QueueP, GlobalVariable *SharedSlot,
                   Value *Ntasks) {
    if (UseCLC)
      return emitClaimCLC(B, SharedSlot, Ntasks);
    return emitClaimTicket(B, QueueP, SharedSlot);
  }

  // m = ticket (pre-Blackwell software CLC): barrier; tid0 atomically takes a
  // ticket and shares it; barrier; everyone reads it.
  Value *emitClaimTicket(IRBuilder<> &B, Value *QueueP,
                         GlobalVariable *SharedSlot) {
    emitBarrier0(B);
    Value *Tz = threadIsZero(B);
    Instruction *SplitPt = &*B.GetInsertPoint();
    Instruction *ThenTerm =
        SplitBlockAndInsertIfThen(Tz, SplitPt, /*Unreachable=*/false);
    {
      IRBuilder<> TB(ThenTerm);
      Value *G = toGlobalAS(TB, QueueP);
      Value *Ticket =
          TB.CreateAtomicRMW(AtomicRMWInst::Add, G, TB.getInt32(1),
                             MaybeAlign(4), AtomicOrdering::Monotonic);
      TB.CreateStore(Ticket, SharedSlot);
    }
    B.SetInsertPoint(SplitPt);
    emitBarrier0(B);
    return B.CreateLoad(B.getInt32Ty(), SharedSlot, "sched.raw.next");
  }

  // m = CLC (Blackwell sm_100+): the HARDWARE claim. try_cancel asks the grid
  // scheduler to hand back a not-yet-launched block's ctaid.x; is_canceled
  // says whether we got one; get_first_ctaid.x decodes it. If nothing is left
  // to steal we return num_tasks (a sentinel >= ntasks -> the driver's done
  // test fires, ending the loop) -- no global ticket counter at all.
  //
  // The whole sequence is emitted as ONE multi-output inline-asm blob so the
  // register/shared/barrier choreography (which LLVM must not reorder or
  // register-allocate across) stays intact. Only tid0 runs it; the result is
  // broadcast through the same shared slot as the ticket path, so callers are
  // byte-for-byte identical. Validated to assemble by Blackwell ptxas
  // (sm_100/sm_120) in tests/clc_probe.ptx.
  Value *emitClaimCLC(IRBuilder<> &B, GlobalVariable *SharedSlot,
                      Value *Ntasks) {
    emitBarrier0(B);
    Value *Tz = threadIsZero(B);
    Instruction *SplitPt = &*B.GetInsertPoint();
    Instruction *ThenTerm =
        SplitBlockAndInsertIfThen(Tz, SplitPt, /*Unreachable=*/false);
    {
      IRBuilder<> TB(ThenTerm);
      // out: i32 claimed ctaid.x (== the raw task); in: i32 fallback (ntasks)
      // The asm allocates its own .shared res/bar (function-scoped names are
      // fine: one worker CTA runs one claim at a time, gated by barriers).
      const char *Asm =
          "{\n"
          "  .reg .pred %pc;\n"
          "  .shared .align 16 .b8 _sched_clc_res[16];\n"
          "  .shared .align 8 .b64 _sched_clc_bar;\n"
          "  .reg .b128 %rq;\n"
          "  .reg .b64 %tmp;\n"
          "  .reg .b32 %resa, %bara, %cx;\n"
          "  mov.u32 %resa, _sched_clc_res;\n"
          "  mov.u32 %bara, _sched_clc_bar;\n"
          "  mbarrier.init.shared::cta.b64 [%bara], 1;\n"
          "  fence.proxy.async.shared::cta;\n"
          "  clusterlaunchcontrol.try_cancel.async.shared::cta."
          "mbarrier::complete_tx::bytes.b128 [%resa], [%bara];\n"
          "  mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 "
          "%tmp, [%bara], 16;\n"
          "L_sched_clc_wait:\n"
          "  mbarrier.try_wait.parity.shared::cta.b64 %pc, [%bara], 0;\n"
          "  @!%pc bra L_sched_clc_wait;\n"
          "  ld.shared.b128 %rq, [%resa];\n"
          "  clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 %pc, %rq;\n"
          "  mov.u32 %cx, $1;\n" // default = fallback (ntasks)
          "  @%pc clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 "
          "{%cx, _, _, _}, %rq;\n"
          "  mov.u32 $0, %cx;\n"
          "}\n";
      FunctionType *FT = FunctionType::get(
          TB.getInt32Ty(), {TB.getInt32Ty()}, /*isVarArg=*/false);
      Value *Fallback = Ntasks; // entry SNAPSHOT (not a reload): matches the
                                // loop's `Next >= Ntasks` termination exactly
      InlineAsm *IA = InlineAsm::get(FT, Asm, "=r,r", /*hasSideEffects=*/true);
      Value *Claimed = TB.CreateCall(FT, IA, {Fallback}, "sched.clc.raw");
      TB.CreateStore(Claimed, SharedSlot);
    }
    B.SetInsertPoint(SplitPt);
    emitBarrier0(B);
    return B.CreateLoad(B.getInt32Ty(), SharedSlot, "sched.raw.next");
  }

  // ctrl->num_tasks (loaded fresh at the claim site so it dominates).
  Value *getNumTasks(IRBuilder<> &B) {
    Value *Ctrl = rtBuffer(B, C, kCtrlSym, nullptr);
    Value *G = toGlobalAS(B, Ctrl);
    Value *P = B.CreateGEP(B.getInt8Ty(), G, B.getInt64(kCtrlNumTasksOff));
    return B.CreateLoad(B.getInt32Ty(), P, "sched.ntasks.clc");
  }

  // task = (order && raw < maxTasks) ? order[raw] : raw   (pi, hand-rolled
  // select-free-of-splits variant for the driver's manually built CFG).
  Value *emitOrderRemap(IRBuilder<> &B, Value *Raw, Function *Driver) {

    if (!rtAvailable(*B.GetInsertBlock()->getModule(), C, kTaskOrderSym) || !C.indirect)
      return Raw;
    Value *Armed = nullptr;
    Value *Order = rtBuffer(B, C, kTaskOrderSym, &Armed);
    Value *InB = B.CreateICmpULT(Raw, B.getInt32(C.maxTasks));
    Value *Use = B.CreateAnd(Armed, InB);
    Instruction *SplitPt = &*B.GetInsertPoint();
    BasicBlock *Head = SplitPt->getParent();
    Instruction *ThenTerm =
        SplitBlockAndInsertIfThen(Use, SplitPt, /*Unreachable=*/false);
    IRBuilder<> TB(ThenTerm);
    Value *G = toGlobalAS(TB, Order);
    Value *GEP = TB.CreateGEP(TB.getInt32Ty(), G, Raw);
    Value *Mapped = TB.CreateLoad(TB.getInt32Ty(), GEP);
    // Write-race safety, both layers (see SchedWeave::emitRemap): clamp to
    // num_tasks (this claim wave's logical range -- fault-proof) AND honor
    // the order table only when ctrl->order_size matches (bijectivity;
    // 0 = unchecked). ctrl is armed on this path (driver entry check), so
    // the direct load needs no nested guard.
    Value *NT = getNumTasks(TB);
    Value *CtrlG = rtBuffer(TB, C, kCtrlSym, nullptr);
    Value *OszP = TB.CreateGEP(TB.getInt8Ty(), CtrlG,
                               TB.getInt64(kCtrlOrderSizeOff));
    Value *Osz = TB.CreateLoad(TB.getInt32Ty(), OszP, "sched.order.size");
    Value *SzOk = TB.CreateOr(TB.CreateICmpEQ(Osz, TB.getInt32(0)),
                              TB.CreateICmpEQ(Osz, NT), "sched.order.valid");
    Value *InGrid = TB.CreateICmpULT(Mapped, NT, "sched.task.ingrid");
    Value *Ok = TB.CreateAnd(InGrid, SzOk, "sched.task.ok");
    Value *Safe = TB.CreateSelect(Ok, Mapped, Raw, "sched.task.safe");
    BasicBlock *Tail = SplitPt->getParent();
    IRBuilder<> PB(Tail, Tail->begin());
    PHINode *Phi = PB.CreatePHI(PB.getInt32Ty(), 2, "sched.task");
    Phi->addIncoming(Safe, ThenTerm->getParent());
    Phi->addIncoming(Raw, Head);
    B.SetInsertPoint(SplitPt);
    return Phi;
  }

  void buildDriver(Function *Body) {
    // Per-CTA shared slot for broadcasting the claimed ticket.
    auto *SharedSlot = new GlobalVariable(
        M, Type::getInt32Ty(Ctx), /*isConstant=*/false,
        GlobalValue::InternalLinkage, UndefValue::get(Type::getInt32Ty(Ctx)),
        F.getName() + ".sched_wq_next", nullptr,
        GlobalValue::NotThreadLocal, /*AddressSpace=*/3);

    F.deleteBody();
    F.addFnAttr(kWqDriverAttr);
    F.addFnAttr(kInstrumentedAttr);

    BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", &F);
    BasicBlock *Stock = BasicBlock::Create(Ctx, "sched.stock", &F);
    BasicBlock *Dyn = BasicBlock::Create(Ctx, "sched.dyn", &F);
    BasicBlock *Bound = BasicBlock::Create(Ctx, "sched.bound", &F);
    BasicBlock *LoopBB = BasicBlock::Create(Ctx, "sched.loop", &F);
    BasicBlock *Exit = BasicBlock::Create(Ctx, "sched.exit", &F);

    SmallVector<Value *, 8> Args;
    for (Argument &A : F.args())
      Args.push_back(&A);

    // entry: armed ? dyn : stock. CLC needs only ctrl (no ticket counter);
    // ticket needs queue && ctrl. In the baked ABI these flags are constant
    // true -- the REAL runtime gate is then the num_tasks==0 check below.
    IRBuilder<> B(Entry);
    Value *QArmed = nullptr, *CArmed = nullptr;
    Value *QueueP = UseCLC ? nullptr : rtBuffer(B, C, kQueueSym, &QArmed);
    Value *CtrlP = rtBuffer(B, C, kCtrlSym, &CArmed);
    Value *First = readSReg(B, ctaidIntrinsic(C.slotAxis));
    Value *Armed = UseCLC ? CArmed : B.CreateAnd(QArmed, CArmed);
    // GRID-SHAPE SOUNDNESS GUARD: the claim enumerates ONLY the slot axis
    // (a ticket is a scalar; CLC's decode here takes get_first_ctaid.x), so
    // on a multi-dim grid a claimed block's task would execute under the
    // WRONG other-axis coordinates (e.g. FlashInfer decode launches
    // grid = (padded_batch, num_kv_heads)). Dynamic claim therefore requires
    // the two non-slot axes to be 1 AT RUNTIME; any other shape takes the
    // stock path (static grid + pi remap) -- correct-or-absent, never wrong.
    // Full multi-axis claiming (linearize + decode the v4 tuple) is future
    // work; see experiments/clc/FINDINGS.md "2D Grid Behavior".
    Value *N1 = readSReg(B, nctaidIntrinsic((C.slotAxis + 1) % 3));
    Value *N2 = readSReg(B, nctaidIntrinsic((C.slotAxis + 2) % 3));
    Value *Grid1D = B.CreateAnd(B.CreateICmpEQ(N1, B.getInt32(1)),
                                B.CreateICmpEQ(N2, B.getInt32(1)),
                                "sched.grid1d");
    Armed = B.CreateAnd(Armed, Grid1D);
    B.CreateCondBr(Armed, Dyn, Stock);

    // stock: the original kernel shape (grid = tasks, one task per block),
    // but WITH the pi remap when the order table is armed -- "dynamic claim
    // off" must not also turn off ordering (static+LPT is the best schedule
    // when costs are predictable; the claim loop is the uncertainty hedge).
    // Unarmed order (or disarm) -> identity -> truly stock, fail-safe.
    {
      IRBuilder<> SB(Stock);
      Instruction *SRet = SB.CreateRetVoid();
      IRBuilder<> WB(SRet);
      Value *STask = emitOrderRemap(WB, First, &F);
      SmallVector<Value *, 8> SArgs(Args);
      SArgs.push_back(STask);
      CallInst *CI = WB.CreateCall(Body->getFunctionType(), Body, SArgs);
      CI->setCallingConv(CallingConv::C);
    }

    // dyn: num_tasks == 0 -> STOCK. Baked-ABI fail-safe (an unprogrammed,
    // zeroed plane must not eat the launch: the armed flags above are baked
    // constants and cannot protect) AND the per-step arming switch (the
    // control plane writes num_tasks: N = claim loop, 0 = stock static).
    IRBuilder<> DB(Dyn);
    Value *G = toGlobalAS(DB, CtrlP);
    Value *NtP = DB.CreateGEP(DB.getInt8Ty(), G, DB.getInt64(kCtrlNumTasksOff));
    Value *Ntasks = DB.CreateLoad(DB.getInt32Ty(), NtP, "sched.ntasks");
    DB.CreateCondBr(DB.CreateICmpEQ(Ntasks, DB.getInt32(0)), Stock, Bound);

    // bound: surplus blocks of a padded launch (First >= ntasks > 0) exit;
    // they are exactly the CLC-claimable/cancelable tail.
    IRBuilder<> BB(Bound);
    BB.CreateCondBr(BB.CreateICmpUGE(First, Ntasks), Exit, LoopBB);

    // loop: serve, then claim the next task (late binding: claim AFTER body).
    IRBuilder<> LB(LoopBB);
    PHINode *Raw = LB.CreatePHI(LB.getInt32Ty(), 2, "sched.raw");
    Raw->addIncoming(First, Bound);
    // Manual CFG from here on: keep a movable split point via a dummy br.
    Instruction *Term = LB.CreateBr(Exit); // placeholder terminator
    IRBuilder<> WB(Term);
    Value *Task = emitOrderRemap(WB, Raw, &F);
    SmallVector<Value *, 8> LArgs(Args);
    LArgs.push_back(Task);
    CallInst *CI = WB.CreateCall(Body->getFunctionType(), Body, LArgs);
    CI->setCallingConv(CallingConv::C);
    Value *Next = emitClaim(WB, QueueP, SharedSlot, Ntasks);
    Value *Done = WB.CreateICmpUGE(Next, Ntasks, "sched.done");
    // Replace the placeholder br with the real backedge.
    BasicBlock *LatchBB = Term->getParent();
    Term->eraseFromParent();
    IRBuilder<> XB(LatchBB);
    XB.CreateCondBr(Done, Exit, LoopBB);
    Raw->addIncoming(Next, LatchBB);

    IRBuilder<> EB(Exit);
    EB.CreateRetVoid();
  }
};

} // namespace

PreservedAnalyses SchedWorkQueuePass::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  if (!isNVPTX(M))
    return PreservedAnalyses::all();
  if (!std::getenv("SCHED_WORKQUEUE"))
    return PreservedAnalyses::all();
  Config C = Config::fromEnv();

  SmallVector<Function *, 4> Kernels;
  for (Function &F : M)
    if (!F.isDeclaration() && isKernel(F) && nameSelected(F.getName()) &&
        !F.hasFnAttribute(kWqDriverAttr))
      Kernels.push_back(&F);

  bool Changed = false;
  for (Function *F : Kernels)
    Changed |= WorkQueueTransform(*F, C).run();
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

} // namespace sched
