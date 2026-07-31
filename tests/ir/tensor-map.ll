declare void @__nta_bind_request(i32, i32)
declare ptr @__nta_acquire_tensor_map_marker(ptr, ptr, i32, i64, i32, i64,
                                             i32, i32)
declare void @__nta_defer_marker(ptr, i32)

define ptx_kernel void @tensor_map_tile(ptr %runtime, ptr %direct.map,
                             i32 %request.slot, i32 %generation,
                             i32 %object.slot, i64 %object.id,
                             i32 %object.version, i32 %bytes,
                             i32 %continuation) {
entry:
  call void @__nta_bind_request(i32 %request.slot, i32 %generation)
  %map = call ptr @__nta_acquire_tensor_map_marker(
      ptr %runtime, ptr %direct.map, i32 %object.slot, i64 %object.id,
      i32 %object.version, i64 0, i32 %bytes, i32 %continuation)
  %pending = icmp eq ptr %map, null
  br i1 %pending, label %defer, label %consume

defer:
  call void @__nta_defer_marker(ptr %runtime, i32 %continuation)
  ret void

consume:
  ; A frontend TMA operation consumes the returned descriptor after this edge.
  ret void
}
