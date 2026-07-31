//===- SchedUtil.cpp - shared helpers for the sched-pass plugin ----------===//
#include "SchedUtil.h"
#include "sched/SchedManifest.h"

#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include <cstdlib>

using namespace llvm;

namespace sched {

static int envInt(const char *K, int Dflt) {
  const char *V = std::getenv(K);
  return V ? std::atoi(V) : Dflt;
}
static double envF(const char *K, double Dflt) {
  const char *V = std::getenv(K);
  return V ? std::atof(V) : Dflt;
}

Config Config::fromEnv() {
  Config C;
  C.slotAxis = envInt("SCHED_SLOT_AXIS", 0);
  // Clamp maxTasks to a sane range: a bad/negative env would otherwise cast
  // to a huge unsigned and size every table absurdly (or, baked, mismatch
  // the arena). 1..1<<20 spans every realistic batch; out-of-range -> default.
  // ABI: this bounds every woven task index. Host-app path -- it MUST equal
  // runtime/sched_rt.h's SCHED_MAX_TASKS macro (that struct's task[] array is
  // sized by it); default 4096 on both sides, override both together. Baked
  // path -- Python's compute_bake_env sets this to match the arena it sizes.
  {
    int mt = envInt("SCHED_MAX_TASKS", 4096);
    C.maxTasks = (mt > 0 && mt <= (1 << 20)) ? (unsigned)mt : 4096u;
  }
  C.indirect = !std::getenv("SCHED_NO_INDIRECT");
  C.timer = !std::getenv("SCHED_NO_TIMER");
  C.policy = !std::getenv("SCHED_NO_POLICY");
  C.shed = !std::getenv("SCHED_NO_SHED"); // default on (see Config::shed)
  C.clc = std::getenv("SCHED_CLC") != nullptr;
  C.timerIndirect = std::getenv("SCHED_TIMER_INDIRECT") != nullptr;
  C.orderIndirect = std::getenv("SCHED_ORDER_INDIRECT") != nullptr;
  C.pdl = std::getenv("SCHED_PDL") != nullptr;
  C.prefetchDist = (unsigned)envInt("SCHED_PF_DIST", 8);
  C.dT = envF("SCHED_DT", 1.0);
  C.dR = envF("SCHED_DR", 1.0);
  C.H = envF("SCHED_H", 0.25);
  C.debug = std::getenv("SCHED_DEBUG") != nullptr;
  auto envAddr = [](const char *K) -> uint64_t {
    const char *V = std::getenv(K);
    return V ? std::strtoull(V, nullptr, 10) : 0ull;
  };
  C.bakeOrder = envAddr("SCHED_BAKE_TASK_ORDER");
  C.bakeCtrl = envAddr("SCHED_BAKE_CTRL");
  C.bakeTimer = envAddr("SCHED_BAKE_TIMER");
  C.bakeQueue = envAddr("SCHED_BAKE_QUEUE");
  return C;
}

uint64_t Config::bakedAddr(StringRef Name) const {
  if (Name == "task_order") return bakeOrder;
  if (Name == "ctrl") return bakeCtrl;
  if (Name == "timer") return bakeTimer;
  if (Name == "queue") return bakeQueue;
  return 0;
}

bool isNVPTX(const Module &M) {
  return Triple(M.getTargetTriple()).isNVPTX();
}

// Kernel = ptx_kernel calling convention (modern clang) OR an
// nvvm.annotations {fn, "kernel", 1} entry (the older encoding).
bool isKernel(const Function &F) {
  if (F.getCallingConv() == CallingConv::PTX_Kernel)
    return true;
  const auto *NMD = F.getParent()->getNamedMetadata("nvvm.annotations");
  if (!NMD)
    return false;
  for (const MDNode *Op : NMD->operands()) {
    if (Op->getNumOperands() < 2)
      continue;
    auto *FnMD = mdconst::dyn_extract_or_null<Function>(Op->getOperand(0));
    auto *Kind = dyn_cast<MDString>(Op->getOperand(1));
    if (FnMD == &F && Kind && Kind->getString() == "kernel")
      return true;
  }
  return false;
}

Intrinsic::ID ctaidIntrinsic(int Axis) {
  switch (Axis) {
  case 1: return Intrinsic::nvvm_read_ptx_sreg_ctaid_y;
  case 2: return Intrinsic::nvvm_read_ptx_sreg_ctaid_z;
  default: return Intrinsic::nvvm_read_ptx_sreg_ctaid_x;
  }
}
Intrinsic::ID tidIntrinsic(int Axis) {
  switch (Axis) {
  case 1: return Intrinsic::nvvm_read_ptx_sreg_tid_y;
  case 2: return Intrinsic::nvvm_read_ptx_sreg_tid_z;
  default: return Intrinsic::nvvm_read_ptx_sreg_tid_x;
  }
}
Intrinsic::ID nctaidIntrinsic(int Axis) {
  switch (Axis) {
  case 1: return Intrinsic::nvvm_read_ptx_sreg_nctaid_y;
  case 2: return Intrinsic::nvvm_read_ptx_sreg_nctaid_z;
  default: return Intrinsic::nvvm_read_ptx_sreg_nctaid_x;
  }
}

unsigned smVersion(const Function &F) {
  if (!F.hasFnAttribute("target-cpu"))
    return 0;
  StringRef CPU = F.getFnAttribute("target-cpu").getValueAsString();
  if (!CPU.consume_front("sm_"))
    return 0;
  unsigned N = 0;
  // "sm_100a"/"sm_120f" style suffixes: parse the leading digits.
  CPU = CPU.take_while([](char c) { return c >= '0' && c <= '9'; });
  if (CPU.getAsInteger(10, N))
    return 0;
  return N;
}

Value *readSReg(IRBuilderBase &B, Intrinsic::ID ID) {
  Module *M = B.GetInsertBlock()->getModule();
  return B.CreateCall(intrinsicDecl(M, ID));
}

Value *readClock64(IRBuilderBase &B) {
  FunctionType *FT = FunctionType::get(B.getInt64Ty(), /*isVarArg=*/false);
  InlineAsm *IA = InlineAsm::get(FT, "mov.u64 $0, %clock64;", "=l",
                                 /*hasSideEffects=*/true);
  return B.CreateCall(FT, IA);
}

Value *threadIsZero(IRBuilderBase &B) {
  Value *X = readSReg(B, tidIntrinsic(0));
  Value *Y = readSReg(B, tidIntrinsic(1));
  Value *Z = readSReg(B, tidIntrinsic(2));
  Value *Or = B.CreateOr(B.CreateOr(X, Y), Z);
  return B.CreateICmpEQ(Or, B.getInt32(0));
}

void emitBarrier0(IRBuilderBase &B) {
  Module *M = B.GetInsertBlock()->getModule();
#if LLVM_VERSION_MAJOR >= 22
  // llvm.nvvm.barrier0 was retired; the replacement takes the barrier id.
  Function *Bar =
      intrinsicDecl(M, Intrinsic::nvvm_barrier_cta_sync_aligned_all);
  B.CreateCall(Bar, {B.getInt32(0)});
#else
  B.CreateCall(intrinsicDecl(M, Intrinsic::nvvm_barrier0));
#endif
}

GlobalVariable *rtSlot(Module &M, StringRef Name) {
  GlobalVariable *GV = M.getGlobalVariable(Name, /*AllowInternal=*/true);
  if (!GV || !GV->getValueType()->isPointerTy())
    return nullptr;
  return GV;
}

Value *loadRtPointer(IRBuilderBase &B, GlobalVariable *GV, Value **NonNull) {
  LLVMContext &Ctx = B.getContext();
  PointerType *GenPtr = PointerType::get(Ctx, /*AS=*/0);
  Value *P = B.CreateLoad(GenPtr, GV, GV->getName() + ".p");
  if (NonNull)
    *NonNull = B.CreateICmpNE(P, ConstantPointerNull::get(GenPtr),
                              GV->getName() + ".armed");
  return P;
}

Value *toGlobalAS(IRBuilderBase &B, Value *P) {
  PointerType *G1 = PointerType::get(B.getContext(), /*AS=*/1);
  if (P->getType() == G1)
    return P;
  return B.CreateAddrSpaceCast(P, G1);
}

// Strip the "__sched_" prefix to the bare capability name the baked-ABI keys on.
static StringRef bareName(StringRef Sym) {
  Sym.consume_front("__sched_");
  return Sym;
}

bool rtAvailable(Module &M, const Config &C, StringRef Name) {
  if (rtSlot(M, Name))
    return true; // host-app ABI: the global is present
  return C.bakedAddr(bareName(Name)) != 0; // baked ABI: env address
}

Value *rtBuffer(IRBuilderBase &B, const Config &C, StringRef Name,
                Value **NonNull) {
  Module &M = *B.GetInsertBlock()->getModule();
  if (GlobalVariable *GV = rtSlot(M, Name)) {
    Value *P = loadRtPointer(B, GV, NonNull); // load the device ptr from global
    return toGlobalAS(B, P);
  }
  uint64_t Addr = C.bakedAddr(bareName(Name));
  PointerType *G1 = PointerType::get(B.getContext(), /*AS=*/1);
  if (!Addr) {
    if (NonNull)
      *NonNull = ConstantInt::getFalse(B.getContext());
    return ConstantPointerNull::get(G1);
  }
  // Baked: stash the address in a module-internal DEVICE GLOBAL and LOAD it,
  // exactly like the host-app ABI loads its named slots. Never emit the
  // address as an i64 immediate: ptxas factors large related immediates
  // across predicated regions (order/ctrl bases from consecutive tensor
  // allocations share high bits), and its re-association emitted an address
  // chain whose high 32 bits were never written (IADD3 lo, -0x80000000 ...
  // IADD.64 with a junk high register -- the shed-enable fault on sm_120).
  // A loaded address is an opaque register; there is nothing to factor.
  // NOTE the slot must NOT be isConstant: LLVM folds loads of constant
  // initializers back into ... the immediate this exists to avoid.
  std::string SlotName = (Name + ".bakedslot").str();
  GlobalVariable *Slot = M.getGlobalVariable(SlotName, /*AllowInternal=*/true);
  if (!Slot) {
    Type *I64 = Type::getInt64Ty(B.getContext());
    Slot = new GlobalVariable(M, I64, /*isConstant=*/false,
                              GlobalValue::InternalLinkage,
                              ConstantInt::get(I64, Addr), SlotName,
                              /*InsertBefore=*/nullptr,
                              GlobalValue::NotThreadLocal, /*AS=*/1);
    Slot->setAlignment(Align(8));
  }
  Value *AddrV =
      B.CreateLoad(Type::getInt64Ty(B.getContext()), Slot, Name + ".bakedaddr");
  Value *P = B.CreateIntToPtr(AddrV, G1, Name + ".baked");
  if (NonNull)
    *NonNull = ConstantInt::getTrue(B.getContext());
  return P;
}

void debugNote(const Config &C, const Twine &Msg) {
  if (C.debug)
    errs() << "[sched] " << Msg << "\n";
}

void dumpManifest(bool csv) {
  // Assert the manifest's declared order matches its row order (the invariant
  // SchedPlugin relies on when it derives pass order from the table).
  for (unsigned i = 1; i < kManifestSize; ++i)
    if (kManifest[i].order < kManifest[i - 1].order)
      errs() << "[sched] MANIFEST ORDER BROKEN at " << kManifest[i].name
             << "\n";
  if (csv) {
    errs() << "name,effect,minSm,knob,disableKnob,order,tag\n";
    for (const Capability &c : kManifest)
      errs() << c.name << ',' << effectName(c.effect) << ',' << c.minSm << ','
             << c.knob << ',' << (c.disableKnob ? 1 : 0) << ',' << c.order
             << ',' << c.tag << '\n';
    return;
  }
  errs() << "== sched capability manifest (" << kManifestSize
         << " instruments) ==\n";
  for (const Capability &c : kManifest) {
    errs() << "  [" << c.order << "] " << c.name << "  " << effectName(c.effect)
           << "  gate=sm_" << c.minSm << "  knob=" << c.knob
           << (c.disableKnob ? "(NO_)" : "(on)") << "  slots=" << c.slots
           << (c.tag[0] ? (Twine("  tag=") + c.tag).str() : "") << "\n";
    errs() << "        " << c.contract << "\n";
  }
}

} // namespace sched
