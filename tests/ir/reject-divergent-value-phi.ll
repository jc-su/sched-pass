; RUN: not opt -load-pass-plugin %plugin -passes=nta-acquire-lowering -S %s -o /dev/null 2>&1 | FileCheck %s
; CHECK: nta: error: reject_divergent_value_phi: request binding has a non-CTA-uniform operand
; CHECK: LLVM ERROR: NTA acquisition verification failed

target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare void @__nta_bind_request(i32, i32)
declare ptr @__nta_acquire_marker(ptr, ptr, i32, i64, i32, i64, i32, i32)
declare void @__nta_defer_marker(ptr, i32)

define ptx_kernel void @reject_divergent_value_phi(ptr %runtime, ptr %direct) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %lane = icmp eq i32 %tid, 0
  br i1 %lane, label %left, label %right

left:
  br label %join

right:
  br label %join

join:
  %request_slot = phi i32 [ 0, %left ], [ 1, %right ]
  call void @__nta_bind_request(i32 %request_slot, i32 1)
  %acquired = call ptr @__nta_acquire_marker(ptr %runtime, ptr %direct, i32 0, i64 1, i32 1, i64 0, i32 64, i32 0)
  %pending = icmp eq ptr %acquired, null
  br i1 %pending, label %defer, label %ready

defer:
  call void @__nta_defer_marker(ptr %runtime, i32 0)
  ret void

ready:
  ret void
}
