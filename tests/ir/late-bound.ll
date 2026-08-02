%requirement = type { ptr, i32, i64, i32, i64, i32 }

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare void @__nta_bind_request(i32, i32)
declare ptr @__nta_acquire_marker(ptr, ptr, i32, i64, i32, i64, i32, i32)
declare void @__nta_defer_marker(ptr, i32)

; The catalog entry is selected on the GPU from a CTA-uniform value. The host
; supplies the catalog, but does not bind this launch's CTA to one object.
define ptx_kernel void @late_bound_tile(ptr %runtime, ptr %catalog, ptr %output,
                                        i32 %catalog.mask, i32 %request.slot,
                                        i32 %generation, i32 %workTicket) {
entry:
  %cta = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %selected = and i32 %cta, %catalog.mask
  %entry.ptr = getelementptr %requirement, ptr %catalog, i32 %selected
  %direct.ptr = getelementptr %requirement, ptr %entry.ptr, i32 0, i32 0
  %slot.ptr = getelementptr %requirement, ptr %entry.ptr, i32 0, i32 1
  %id.ptr = getelementptr %requirement, ptr %entry.ptr, i32 0, i32 2
  %version.ptr = getelementptr %requirement, ptr %entry.ptr, i32 0, i32 3
  %offset.ptr = getelementptr %requirement, ptr %entry.ptr, i32 0, i32 4
  %bytes.ptr = getelementptr %requirement, ptr %entry.ptr, i32 0, i32 5
  %direct = load ptr, ptr %direct.ptr, align 8
  %object.slot = load i32, ptr %slot.ptr, align 4
  %object.id = load i64, ptr %id.ptr, align 8
  %object.version = load i32, ptr %version.ptr, align 4
  %offset = load i64, ptr %offset.ptr, align 8
  %bytes = load i32, ptr %bytes.ptr, align 4
  call void @__nta_bind_request(i32 %request.slot, i32 %generation)
  %address = call ptr @__nta_acquire_marker(
      ptr %runtime, ptr %direct, i32 %object.slot, i64 %object.id,
      i32 %object.version, i64 %offset, i32 %bytes, i32 %workTicket)
  %pending = icmp eq ptr %address, null
  br i1 %pending, label %defer, label %consume

defer:
  call void @__nta_defer_marker(ptr %runtime, i32 %workTicket)
  ret void

consume:
  %value = load i32, ptr %address, align 4
  store i32 %value, ptr %output, align 4
  ret void
}
