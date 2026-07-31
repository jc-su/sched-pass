declare void @__nta_bind_request(i32, i32)
declare i1 @__nta_acquire_set_marker(ptr, ptr, i32, i32, i32)
declare void @__nta_defer_marker(ptr, i32)
declare void @side_effect()

define void @unsafe_dependency_set(ptr %runtime, ptr %requirements,
                                   i32 %request.slot, i32 %generation,
                                   i32 %continuation) {
entry:
  call void @__nta_bind_request(i32 %request.slot, i32 %generation)
  %ready = call i1 @__nta_acquire_set_marker(
      ptr %runtime, ptr %requirements, i32 2, i32 0, i32 %continuation)
  br i1 %ready, label %consume, label %defer

defer:
  call void @side_effect()
  call void @__nta_defer_marker(ptr %runtime, i32 %continuation)
  ret void

consume:
  ret void
}
