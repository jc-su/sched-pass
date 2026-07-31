declare void @__nta_bind_request(i32, i32)
declare ptr @__nta_acquire_marker(ptr, ptr, i32, i64, i32, i64, i32, i32)
declare void @__nta_defer_marker(ptr, i32)
declare void @side_effect()

define void @unsafe_pending_edge(ptr %runtime, i32 %slot, i32 %generation) {
entry:
  call void @__nta_bind_request(i32 %slot, i32 %generation)
  %address = call ptr @__nta_acquire_marker(
      ptr %runtime, ptr null, i32 0, i64 42, i32 1, i64 0, i32 4, i32 0)
  %pending = icmp eq ptr %address, null
  br i1 %pending, label %defer, label %consume

defer:
  call void @side_effect()
  call void @__nta_defer_marker(ptr %runtime, i32 0)
  ret void

consume:
  ret void
}
