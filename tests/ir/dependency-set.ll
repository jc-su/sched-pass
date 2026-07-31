declare void @__nta_bind_request(i32, i32)
declare i1 @__nta_acquire_set_marker(ptr, ptr, i32, i32, i32)
declare void @__nta_defer_marker(ptr, i32)
declare void @consume_requirements(ptr, i32)

define void @multi_object_tile(ptr %runtime, ptr %requirements, i32 %count,
                               i32 %request.slot, i32 %generation,
                               i32 %continuation) {
entry:
  call void @__nta_bind_request(i32 %request.slot, i32 %generation)
  %ready = call i1 @__nta_acquire_set_marker(
      ptr %runtime, ptr %requirements, i32 %count, i32 0,
      i32 %continuation)
  br i1 %ready, label %consume, label %defer

defer:
  call void @__nta_defer_marker(ptr %runtime, i32 %continuation)
  ret void

consume:
  call void @consume_requirements(ptr %requirements, i32 %count)
  ret void
}
