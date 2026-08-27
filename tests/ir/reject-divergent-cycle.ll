declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare void @__nta_bind_request(i32, i32)
declare ptr @__nta_acquire_marker(ptr, ptr, i32, i64, i32, i64, i32, i32)
declare void @__nta_defer_marker(ptr, i32)

; The recursive edge is provisionally uniform, but the cycle also contains a
; lane-varying seed. No member of this SCC may survive as cached Uniform.
define ptx_kernel void @reject_divergent_cycle(ptr %runtime, ptr %direct,
                                               i1 %iterate) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %loop

loop:
  %cycle = phi i32 [ 0, %entry ], [ %next, %loop ]
  %request.slot = add i32 %cycle, 1
  %next = add i32 %request.slot, %tid
  br i1 %iterate, label %loop, label %acquire

acquire:
  call void @__nta_bind_request(i32 %request.slot, i32 1)
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
