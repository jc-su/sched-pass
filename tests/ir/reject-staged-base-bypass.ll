declare void @__nta_bind_request(i32, i32)
declare ptr @__nta_acquire_marker(ptr, ptr, i32, i64, i32, i64, i32, i32)
declare void @__nta_defer_marker(ptr, i32)

; The consume edge reads through the acquired address, but also reads the raw
; direct base the marker was given, skipping the liveness/generation check.
define ptx_kernel void @staged_bypass(ptr %runtime, ptr %direct, ptr %output,
                          i32 %request.slot, i32 %generation,
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
  %sidedoor = getelementptr inbounds i8, ptr %direct, i64 64
  %stolen = load float, ptr %sidedoor, align 4
  %sum = fadd float %value, %stolen
  store float %sum, ptr %output, align 4
  ret void
}
