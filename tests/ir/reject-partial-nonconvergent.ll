declare void @__nta_bind_request(i32, i32)
declare i1 @__nta_acquire_set_marker(ptr, ptr, i32, i32, i32)
declare void @__nta_defer_marker(ptr, i32)
declare void @__nta_begin_partial_marker(ptr, i32)
declare void @__nta_commit_partial_marker(ptr, i32, i32, i32, i32, i64)

define ptx_kernel void @partial_nonconvergent(
    ptr %runtime, ptr %requirements, i32 %request.slot, i32 %generation,
    i32 %work.ticket) {
entry:
  call void @__nta_bind_request(i32 %request.slot, i32 %generation)
  %ready = call i1 @__nta_acquire_set_marker(
      ptr %runtime, ptr %requirements, i32 1, i32 0, i32 %work.ticket)
  br i1 %ready, label %compute, label %defer

defer:
  call void @__nta_defer_marker(ptr %runtime, i32 %work.ticket)
  ret void

compute:
  call void @__nta_begin_partial_marker(ptr %runtime, i32 %work.ticket)
  call void @__nta_commit_partial_marker(
      ptr %runtime, i32 %work.ticket, i32 0, i32 0, i32 1, i64 1)
  ret void
}
