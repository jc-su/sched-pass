declare void @__nta_bind_request(i32, i32)
declare void @__nta_begin_partial_marker(ptr, i32) convergent
declare void @__nta_commit_partial_marker(ptr, i32, i32, i32, i32, i64) convergent

define ptx_kernel void @partial_without_acquire(
    ptr %runtime, i32 %request.slot, i32 %generation, i32 %work.ticket) {
entry:
  call void @__nta_bind_request(i32 %request.slot, i32 %generation)
  call void @__nta_begin_partial_marker(ptr %runtime, i32 %work.ticket)
  call void @__nta_commit_partial_marker(
      ptr %runtime, i32 %work.ticket, i32 0, i32 0, i32 1, i64 1)
  ret void
}
