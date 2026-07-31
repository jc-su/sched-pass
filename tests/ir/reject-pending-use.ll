declare void @__nta_bind_request(i32, i32)
declare ptr @__nta_acquire_marker(ptr, ptr, i32, i64, i32, i64, i32, i32)
declare void @__nta_defer_marker(ptr, i32)

define void @pending_use(ptr %runtime, i32 %request.slot, i32 %generation,
                         i32 %continuation) {
entry:
  call void @__nta_bind_request(i32 %request.slot, i32 %generation)
  %address = call ptr @__nta_acquire_marker(
      ptr %runtime, ptr null, i32 0, i64 1, i32 1, i64 0, i32 16,
      i32 %continuation)
  %pending = icmp eq ptr %address, null
  br i1 %pending, label %defer, label %consume

defer:
  %invalid = ptrtoint ptr %address to i64
  call void @__nta_defer_marker(ptr %runtime, i32 %continuation)
  ret void

consume:
  ret void
}
