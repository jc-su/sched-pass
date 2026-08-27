declare void @__nta_bind_request(i32, i32)
declare ptr @__nta_acquire_marker(ptr, ptr, i32, i64, i32, i64, i32, i32)
declare void @__nta_defer_marker(ptr, i32)

; A uniform loop-carried induction cycle must remain admissible. Treating every
; recursive SSA edge as divergent would reject this ordinary kernel shape.
define ptx_kernel void @uniform_cycle(ptr %runtime, ptr %direct, ptr %output,
                                      i1 %iterate) {
entry:
  br label %loop

loop:
  %induction = phi i32 [ 0, %entry ], [ %next, %loop ]
  %request.slot = add i32 %induction, 1
  %next = add i32 %request.slot, 1
  br i1 %iterate, label %loop, label %acquire

acquire:
  call void @__nta_bind_request(i32 %request.slot, i32 1)
  %address = call ptr @__nta_acquire_marker(
      ptr %runtime, ptr %direct, i32 0, i64 1, i32 1, i64 0, i32 4, i32 0)
  %pending = icmp eq ptr %address, null
  br i1 %pending, label %defer, label %consume

defer:
  call void @__nta_defer_marker(ptr %runtime, i32 0)
  ret void

consume:
  %value = load i32, ptr %address, align 4
  store i32 %value, ptr %output, align 4
  ret void
}
