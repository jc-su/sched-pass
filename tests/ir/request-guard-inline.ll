target triple = "nvptx64-nvidia-cuda"

%runtime = type { i32, ptr, i32 }

define internal i1 @nta_request_live_cta(ptr %runtime, i32 %slot, i32 %generation) {
entry:
  %valid = icmp ne ptr %runtime, null
  ret i1 %valid
}

declare void @__nta_bind_request(i32, i32)
declare i1 @__nta_acquire_set_marker(ptr, ptr, i32, i32, i32)

define ptx_kernel void @request_guard(ptr %runtime, i32 %slot, i32 %generation) {
entry:
  call void @__nta_bind_request(i32 %slot, i32 %generation)
  %ready = call i1 @__nta_acquire_set_marker(
      ptr %runtime, ptr null, i32 0, i32 0, i32 -1)
  br i1 %ready, label %run, label %exit

run:
  br label %exit

exit:
  ret void
}
