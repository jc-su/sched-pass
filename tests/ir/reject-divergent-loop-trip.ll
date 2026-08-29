declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare void @__nta_bind_request(i32, i32)
declare ptr @__nta_acquire_marker(ptr, ptr, i32, i64, i32, i64, i32, i32)
declare void @__nta_defer_marker(ptr, i32)

; A rotated loop can execute a different number of iterations in each lane.
; Its latch is also the loop body, so non-strict post-dominance of the block by
; itself must not hide the divergent trip count from the loop-carried value.
define ptx_kernel void @reject_divergent_loop_trip(
    ptr %runtime, ptr %direct, i32 %uniform.limit) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %lane = and i32 %tid, 7
  %limit = add i32 %uniform.limit, %lane
  br label %loop

loop:
  %iteration = phi i32 [ 0, %entry ], [ %next, %loop ]
  %next = add i32 %iteration, 1
  %again = icmp ult i32 %next, %limit
  br i1 %again, label %loop, label %acquire

acquire:
  call void @__nta_bind_request(i32 %next, i32 1)
  %address = call ptr @__nta_acquire_marker(
      ptr %runtime, ptr %direct, i32 0, i64 1, i32 1, i64 0, i32 4, i32 0)
  %pending = icmp eq ptr %address, null
  br i1 %pending, label %defer, label %ready

defer:
  call void @__nta_defer_marker(ptr %runtime, i32 0)
  ret void

ready:
  ret void
}
