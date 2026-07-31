declare ptr @__nta_acquire_marker(ptr, ptr, i32, i64, i32, i64, i32, i32)
declare void @__nta_defer_marker(ptr, i32)

define void @missing_request_binding(ptr %runtime, ptr %output) {
entry:
  %address = call ptr @__nta_acquire_marker(
      ptr %runtime, ptr null, i32 0, i64 42, i32 1, i64 0, i32 4, i32 0)
  %pending = icmp eq ptr %address, null
  br i1 %pending, label %defer, label %consume

defer:
  call void @__nta_defer_marker(ptr %runtime, i32 0)
  ret void

consume:
  %value = load float, ptr %address, align 4
  store float %value, ptr %output, align 4
  ret void
}
