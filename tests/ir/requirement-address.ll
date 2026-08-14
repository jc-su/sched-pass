declare void @__nta_bind_request(i32, i32)
declare i1 @__nta_acquire_set_marker(ptr, ptr, i32, i32, i32)
declare void @__nta_defer_marker(ptr, i32)
declare ptr @nta_requirement_address(ptr, ptr)

define ptx_kernel void @requirement_consumer(ptr %runtime, ptr %requirements,
                               ptr %output, i32 %count,
                               i32 %request.slot, i32 %generation,
                               i32 %workTicket) {
entry:
  call void @__nta_bind_request(i32 %request.slot, i32 %generation)
  %ready = call i1 @__nta_acquire_set_marker(
      ptr %runtime, ptr %requirements, i32 %count, i32 0,
      i32 %workTicket)
  br i1 %ready, label %consume, label %defer

defer:
  call void @__nta_defer_marker(ptr %runtime, i32 %workTicket)
  ret void

consume:
  %second = getelementptr inbounds i8, ptr %requirements, i64 32
  %staged = call ptr @nta_requirement_address(ptr %runtime, ptr %second)
  %value = load float, ptr %staged, align 4
  store float %value, ptr %output, align 4
  ret void
}
