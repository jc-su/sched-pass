declare void @__nta_bind_request(i32, i32)
declare ptr @__nta_acquire_marker(ptr, ptr, i32, i64, i32, i64, i32, i32)
declare void @__nta_defer_marker(ptr, i32)

; Repeated operands around a recursive SSA edge made erase-and-recompute visit
; this cone exponentially. The verifier must settle the complete optimistic
; region once and retain the ordinary CTA-uniform loop-carried value.
define ptx_kernel void @uniform_cycle_deep(ptr %runtime, ptr %direct,
                                           i1 %iterate) {
entry:
  br label %loop

loop:
  %cycle = phi i32 [ 1, %entry ], [ %v28, %loop ]
  %v01 = add i32 %cycle, %cycle
  %v02 = add i32 %v01, %v01
  %v03 = add i32 %v02, %v02
  %v04 = add i32 %v03, %v03
  %v05 = add i32 %v04, %v04
  %v06 = add i32 %v05, %v05
  %v07 = add i32 %v06, %v06
  %v08 = add i32 %v07, %v07
  %v09 = add i32 %v08, %v08
  %v10 = add i32 %v09, %v09
  %v11 = add i32 %v10, %v10
  %v12 = add i32 %v11, %v11
  %v13 = add i32 %v12, %v12
  %v14 = add i32 %v13, %v13
  %v15 = add i32 %v14, %v14
  %v16 = add i32 %v15, %v15
  %v17 = add i32 %v16, %v16
  %v18 = add i32 %v17, %v17
  %v19 = add i32 %v18, %v18
  %v20 = add i32 %v19, %v19
  %v21 = add i32 %v20, %v20
  %v22 = add i32 %v21, %v21
  %v23 = add i32 %v22, %v22
  %v24 = add i32 %v23, %v23
  %v25 = add i32 %v24, %v24
  %v26 = add i32 %v25, %v25
  %v27 = add i32 %v26, %v26
  %v28 = add i32 %v27, %v27
  br i1 %iterate, label %loop, label %acquire

acquire:
  call void @__nta_bind_request(i32 %v28, i32 1)
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
