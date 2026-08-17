declare void @__nta_bind_request(i32, i32)
declare ptr @__nta_acquire_marker(ptr, ptr, i32, i64, i32, i64, i32, i32)
declare void @__nta_defer_marker(ptr, i32)
declare void @consume_pair({ ptr, i64 })

; Packing the raw direct base into an aggregate hides it from any checker
; that enumerates pointer-typed operands; the taint closure rejects the
; packing itself.
define ptx_kernel void @staged_aggregate_escape(ptr %runtime, ptr %direct,
                          ptr %output, i32 %request.slot, i32 %generation,
                          i32 %object.slot, i64 %object.id,
                          i32 %object.version, i64 %offset,
                          i32 %bytes, i32 %workTicket) {
entry:
  call void @__nta_bind_request(i32 %request.slot, i32 %generation)
  %address = call ptr @__nta_acquire_marker(
      ptr %runtime, ptr %direct, i32 %object.slot, i64 %object.id,
      i32 %object.version, i64 %offset, i32 %bytes, i32 %workTicket)
  %pending = icmp eq ptr %address, null
  br i1 %pending, label %defer, label %consume

defer:
  call void @__nta_defer_marker(ptr %runtime, i32 %workTicket)
  ret void

consume:
  %value = load float, ptr %address, align 4
  store float %value, ptr %output, align 4
  %packed = insertvalue { ptr, i64 } poison, ptr %direct, 0
  %sealed = insertvalue { ptr, i64 } %packed, i64 %offset, 1
  call void @consume_pair({ ptr, i64 } %sealed)
  ret void
}
