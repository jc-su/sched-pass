declare void @__nta_bind_request(i32, i32)
declare i1 @__nta_acquire_set_marker(ptr, ptr, i32, i32, i32)
declare void @__nta_begin_partial_marker(ptr, i32) convergent
declare void @__nta_commit_stream_ordered_partial_marker(
    ptr, i32, i32, i32, i32, i64) convergent
declare void @write_partial(ptr)

@request_indices = external global ptr

; A generated operator may reload framework metadata on the acquired edge
; before joining its common numerical block.  The request/ticket PHIs still
; select exactly the values belonging to that acquired edge.
define ptx_kernel void @forwarded_ready_attention_tile(
    ptr %runtime, ptr %partial, i1 %choose.forwarded,
    i32 %request.slot, i32 %generation, i32 %work.ticket,
    i32 %other.request.slot, i32 %other.generation, i32 %other.work.ticket,
    i32 %reduction.group, i32 %contributor.index, i32 %contributor.count) {
entry:
  br i1 %choose.forwarded, label %forwarded.acquire, label %direct.acquire

forwarded.acquire:
  call void @__nta_bind_request(i32 %request.slot, i32 %generation)
  %forwarded.ready = call i1 @__nta_acquire_set_marker(
      ptr %runtime, ptr null, i32 0, i32 0, i32 %work.ticket)
  br i1 %forwarded.ready, label %acquired.reload, label %forwarded.cancelled

forwarded.cancelled:
  ret void

acquired.reload:
  %indices = load ptr, ptr @request_indices
  br label %compute

direct.acquire:
  call void @__nta_bind_request(i32 %other.request.slot, i32 %other.generation)
  %direct.ready = call i1 @__nta_acquire_set_marker(
      ptr %runtime, ptr null, i32 0, i32 0, i32 %other.work.ticket)
  br i1 %direct.ready, label %compute, label %direct.cancelled

direct.cancelled:
  ret void

compute:
  %bound.slot = phi i32 [ %request.slot, %acquired.reload ],
                         [ %other.request.slot, %direct.acquire ]
  %bound.generation = phi i32 [ %generation, %acquired.reload ],
                               [ %other.generation, %direct.acquire ]
  %bound.ticket = phi i32 [ %work.ticket, %acquired.reload ],
                           [ %other.work.ticket, %direct.acquire ]
  call void @__nta_bind_request(i32 %bound.slot, i32 %bound.generation)
  call void @__nta_begin_partial_marker(ptr %runtime, i32 %bound.ticket)
  call void @write_partial(ptr %partial)
  call void @__nta_commit_stream_ordered_partial_marker(
      ptr %runtime, i32 %bound.ticket, i32 %reduction.group,
      i32 %contributor.index, i32 %contributor.count, i64 1200)
  ret void
}
