declare void @__nta_bind_request(i32, i32)
declare i1 @__nta_acquire_set_marker(ptr, ptr, i32, i32, i32)
declare void @__nta_defer_marker(ptr, i32)
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

define ptx_kernel void @divergent_token(ptr %runtime, ptr %requirements,
                             i32 %request.slot, i32 %generation) {
entry:
  call void @__nta_bind_request(i32 %request.slot, i32 %generation)
  %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ready = call i1 @__nta_acquire_set_marker(
      ptr %runtime, ptr %requirements, i32 1, i32 0, i32 %thread)
  br i1 %ready, label %exit, label %defer

defer:
  call void @__nta_defer_marker(ptr %runtime, i32 %thread)
  ret void

exit:
  ret void
}
