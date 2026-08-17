declare void @__nta_bind_request(i32, i32)
declare ptr @__nta_acquire_marker(ptr, ptr, i32, i64, i32, i64, i32, i32)
declare void @__nta_defer_marker(ptr, i32)
declare void @helper_touches_memory(ptr)

; The consume edge reads through the acquired address, but also hands the raw
; direct base to an ordinary function, escaping the legality clause.
define ptx_kernel void @staged_call_escape(ptr %runtime, ptr %direct,
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
  %sidedoor = getelementptr inbounds i8, ptr %direct, i64 64
  call void @helper_touches_memory(ptr %sidedoor)
  ret void
}
