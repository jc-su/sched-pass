declare void @__nta_bind_request(i32, i32)
declare i1 @__nta_acquire_set_marker(ptr, ptr, i32, i32, i32)
declare void @__nta_defer_marker(ptr, i32)
declare ptr @nta_requirement_address(ptr, ptr)
declare void @consume_address(ptr)

; The requirement address is computed before the acquisition branch resolves,
; so it is reachable on the pending edge as well as the ready edge.
define ptx_kernel void @early_requirement(ptr %runtime, ptr %requirements,
                               i32 %count, i32 %request.slot, i32 %generation,
                               i32 %workTicket) {
entry:
  call void @__nta_bind_request(i32 %request.slot, i32 %generation)
  %ready = call i1 @__nta_acquire_set_marker(
      ptr %runtime, ptr %requirements, i32 %count, i32 0,
      i32 %workTicket)
  %staged = call ptr @nta_requirement_address(ptr %runtime, ptr %requirements)
  br i1 %ready, label %consume, label %defer

defer:
  call void @__nta_defer_marker(ptr %runtime, i32 %workTicket)
  ret void

consume:
  call void @consume_address(ptr %staged)
  ret void
}
