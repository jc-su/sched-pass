; weave_basic.ll -- FileCheck golden-IR gate for the core weave shapes (D3).
; CPU-only: runs under opt, no GPU. Exercises, on a minimal NVPTX kernel with
; a paged-signature loop (address depends on a loaded value):
;   * pi remap: ctaid read feeds task_order[pid] behind an armed+bounds guard,
;     merged by a PHI; the ORIGINAL ctaid use is rewritten to the PHI.
;   * timer: clock64 bracket, tid==0 gate, and the ctrl.flags TIMER-OFF gate
;     nested under the ctrl-armed check; atomicrmw add monotonic.
;   * fail-safe CFG: every capability guarded by its slot's null-check.
;
; RUN: opt -load-pass-plugin=%PLUGIN -passes=sched-weave %s -S | FileCheck %s

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@__sched_task_order = global ptr null
@__sched_ctrl = global ptr null
@__sched_timer = global ptr null

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()

define ptx_kernel void @paged_kernel(ptr %kv, ptr %bt, ptr %out, i32 %n) {
entry:
  %pid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %bt.gep = getelementptr i32, ptr %bt, i32 %i
  %page = load i32, ptr %bt.gep
  %off = mul i32 %page, 64
  %off.pid = add i32 %off, %pid
  %kv.gep = getelementptr float, ptr %kv, i32 %off.pid
  %v = load float, ptr %kv.gep
  %out.gep = getelementptr float, ptr %out, i32 %i
  store float %v, ptr %out.gep
  %i.next = add i32 %i, 1
  %done = icmp eq i32 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; --- timer-off gate LATCHED AT ENTRY (before remap; #1 timer-race fix) ------
; The flag is read ONCE at launch, not re-read at each return, so flipping
; ctrl.flags mid-flight cannot suppress this launch's own write.
; CHECK-LABEL: define ptx_kernel void @paged_kernel
; CHECK: %sched.flags = load i32, ptr addrspace(1)
; CHECK: %sched.timer.off = icmp ne i32
; CHECK: %sched.timer.gate = phi i1

; --- pi remap: armed && in-bounds guard, table load, merge PHI --------------
; CHECK: [[ORD:%.*]] = load ptr, ptr @__sched_task_order
; CHECK: [[ARMED:%.*]] = icmp ne ptr [[ORD]], null
; CHECK: [[INB:%.*]] = icmp ult i32 [[PID:%.*]], 4096
; CHECK: and i1 [[ARMED]], [[INB]]
; CHECK: [[MAPPED:%.*]] = load i32, ptr addrspace(1) {{%.*}}, align 4
; write-race safety: whole-launch order_size validity + per-entry clamp
; CHECK: [[NT:%.*]] = {{(tail )?}}call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
; CHECK: %sched.order.size = load i32, ptr addrspace(1)
; CHECK: %sched.order.valid = or i1
; CHECK: %sched.order.szok = phi i1
; CHECK: %sched.task.ingrid = icmp ult i32 [[MAPPED]], [[NT]]
; CHECK: %sched.task.ok = and i1 %sched.task.ingrid, %sched.order.szok
; CHECK: [[SAFE:%.*]] = select i1 %sched.task.ok, i32 [[MAPPED]], i32 [[PID]]
; CHECK: [[TASK:%.*]] = phi i32 [ [[SAFE]], {{%.*}} ], [ [[PID]], {{%.*}} ]

; --- the remapped task feeds the ORIGINAL address arithmetic ----------------
; CHECK: add i32 {{%.*}}, [[TASK]]

; --- timer atomic at each return, gated on the ENTRY-latched off value ------
; CHECK: %sched.timer.on = xor i1 %sched.timer.gate, true
; CHECK: atomicrmw add ptr addrspace(1) {{%.*}}, i64 %sched.cycles monotonic
