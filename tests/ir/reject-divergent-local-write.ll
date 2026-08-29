declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare void @__nta_bind_request(i32, i32)
declare ptr @__nta_acquire_marker(ptr, ptr, i32, i64, i32, i64, i32, i32)
declare void @__nta_defer_marker(ptr, i32)

; The alloca base is lane-private but structurally identical. A lane-varying
; GEP used by one of its writes means a later fixed-field load is not proven
; CTA-uniform: different lanes may have initialized different fields.
define ptx_kernel void @reject_divergent_local_write(ptr %runtime,
                                                     ptr %direct) {
entry:
  %local = alloca [2 x i32], align 8
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %lane = and i32 %tid, 1
  %lane.field = getelementptr inbounds [2 x i32], ptr %local, i32 0, i32 %lane
  store i32 7, ptr %lane.field, align 4
  %request.field = getelementptr inbounds [2 x i32], ptr %local, i32 0, i32 0
  %request.slot = load i32, ptr %request.field, align 4
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
