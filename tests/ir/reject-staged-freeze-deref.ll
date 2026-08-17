declare void @__nta_bind_request(i32, i32)
declare ptr @__nta_acquire_marker(ptr, ptr, i32, i64, i32, i64, i32, i32)
declare void @__nta_defer_marker(ptr, i32)

; Freezing the raw direct base launders it past backward provenance walks;
; the forward taint closure must follow the freeze to the dereference.
define ptx_kernel void @staged_freeze_deref(ptr %runtime, ptr %direct,
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
  %frozen = freeze ptr %direct
  %sidedoor = getelementptr inbounds i8, ptr %frozen, i64 64
  %stolen = load float, ptr %sidedoor, align 4
  store float %stolen, ptr %output, align 4
  ret void
}
