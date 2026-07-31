declare void @__nta_bind_request(i32, i32)
declare i1 @__nta_acquire_set_marker(ptr, ptr, i32, i32, i32)
declare void @__nta_defer_marker(ptr, i32)

define void @device_helper(ptr %runtime, ptr %requirements, i32 %request.slot,
                           i32 %generation, i32 %continuation) {
entry:
  call void @__nta_bind_request(i32 %request.slot, i32 %generation)
  %ready = call i1 @__nta_acquire_set_marker(
      ptr %runtime, ptr %requirements, i32 1, i32 0, i32 %continuation)
  br i1 %ready, label %consume, label %defer

defer:
  call void @__nta_defer_marker(ptr %runtime, i32 %continuation)
  ret void

consume:
  ret void
}
