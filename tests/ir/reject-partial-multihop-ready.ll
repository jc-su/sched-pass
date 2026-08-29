declare void @__nta_bind_request(i32, i32)
declare i1 @__nta_acquire_set_marker(ptr, ptr, i32, i32, i32)
declare void @__nta_begin_partial_marker(ptr, i32) convergent
declare void @__nta_commit_stream_ordered_partial_marker(
    ptr, i32, i32, i32, i32, i64) convergent

; Only one canonical unconditional ready-edge forwarding block is accepted.
; A second hop is deliberately outside the verifier's structural contract.
define ptx_kernel void @multihop_ready_attention_tile(
    ptr %runtime, i32 %request.slot, i32 %generation, i32 %work.ticket,
    i32 %reduction.group, i32 %contributor.index, i32 %contributor.count) {
entry:
  call void @__nta_bind_request(i32 %request.slot, i32 %generation)
  %ready = call i1 @__nta_acquire_set_marker(
      ptr %runtime, ptr null, i32 0, i32 0, i32 %work.ticket)
  br i1 %ready, label %acquired.reload, label %cancelled

cancelled:
  ret void

acquired.reload:
  br label %acquired.forward

acquired.forward:
  br label %compute

compute:
  %bound.slot = phi i32 [ %request.slot, %acquired.forward ]
  %bound.generation = phi i32 [ %generation, %acquired.forward ]
  %bound.ticket = phi i32 [ %work.ticket, %acquired.forward ]
  call void @__nta_bind_request(i32 %bound.slot, i32 %bound.generation)
  call void @__nta_begin_partial_marker(ptr %runtime, i32 %bound.ticket)
  call void @__nta_commit_stream_ordered_partial_marker(
      ptr %runtime, i32 %bound.ticket, i32 %reduction.group,
      i32 %contributor.index, i32 %contributor.count, i64 1200)
  ret void
}
