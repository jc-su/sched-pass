declare void @__nta_bind_request(i32, i32)
declare i1 @__nta_acquire_set_marker(ptr, ptr, i32, i32, i32)
declare void @__nta_defer_marker(ptr, i32)
declare void @__nta_begin_partial_marker(ptr, i32) convergent
declare void @__nta_commit_partial_marker(ptr, i32, i32, i32, i32, i64) convergent
declare void @write_partial(ptr)

define ptx_kernel void @incremental_attention_tile(
    ptr %runtime, ptr %requirements, ptr %partial, i32 %request.slot,
    i32 %generation, i32 %work.ticket, i32 %reduction.group,
    i32 %contributor.index, i32 %contributor.count) {
entry:
  call void @__nta_bind_request(i32 %request.slot, i32 %generation)
  %ready = call i1 @__nta_acquire_set_marker(
      ptr %runtime, ptr %requirements, i32 2, i32 0, i32 %work.ticket)
  br i1 %ready, label %compute, label %defer

defer:
  call void @__nta_defer_marker(ptr %runtime, i32 %work.ticket)
  ret void

compute:
  call void @__nta_begin_partial_marker(ptr %runtime, i32 %work.ticket)
  call void @write_partial(ptr %partial)
  call void @__nta_commit_partial_marker(
      ptr %runtime, i32 %work.ticket, i32 %reduction.group,
      i32 %contributor.index, i32 %contributor.count, i64 1200)
  ret void
}
