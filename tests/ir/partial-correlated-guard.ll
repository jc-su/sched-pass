declare void @__nta_bind_request(i32, i32)
declare i1 @__nta_acquire_set_marker(ptr, ptr, i32, i32, i32)
declare void @__nta_begin_partial_marker(ptr, i32) convergent
declare void @__nta_commit_stream_ordered_partial_marker(
    ptr, i32, i32, i32, i32, i64) convergent
declare void @write_partial(ptr)

; Clang emits this pair of correlated triangles when dense and partial work
; share one numerical body.  The repeated condition is one immutable SSA value,
; so begin and commit execute together even though neither marker dominates the
; other in the path-insensitive CFG.
define ptx_kernel void @correlated_partial_attention_tile(
    ptr %runtime, ptr %partial, i32 %request.slot, i32 %generation,
    i32 %work.ticket, i32 %reduction.group, i32 %contributor.index,
    i32 %contributor.count) {
entry:
  call void @__nta_bind_request(i32 %request.slot, i32 %generation)
  %ready = call i1 @__nta_acquire_set_marker(
      ptr %runtime, ptr null, i32 0, i32 0, i32 %work.ticket)
  br i1 %ready, label %choose, label %cancelled

cancelled:
  ret void

choose:
  %dense = icmp eq i32 %work.ticket, -1
  br i1 %dense, label %compute, label %partial.begin

partial.begin:
  call void @__nta_begin_partial_marker(ptr %runtime, i32 %work.ticket)
  br label %compute

compute:
  call void @write_partial(ptr %partial)
  br i1 %dense, label %exit, label %partial.commit

partial.commit:
  call void @__nta_commit_stream_ordered_partial_marker(
      ptr %runtime, i32 %work.ticket, i32 %reduction.group,
      i32 %contributor.index, i32 %contributor.count, i64 1200)
  br label %exit

exit:
  ret void
}
