; A typed module with an incomplete semantic contract must fail closed before
; the pass considers any acquisition marker.
@nta_jit_instrumentation_flags = constant i64 7
@nta_jit_identity_binding = constant i32 1
@nta_jit_demand_binding = constant i32 1
@nta_jit_access_proof = constant i32 3
@nta_jit_granularity_bytes = constant i32 0
@nta_jit_tier_mask = constant i64 63

declare void @__nta_bind_request(i32, i32)
declare ptr @__nta_acquire_marker(ptr, ptr, i32, i64, i32, i64, i32, i32)
declare void @__nta_defer_marker(ptr, i32)

define ptx_kernel void @typed_contract_invalid(
    ptr %runtime, ptr %direct, ptr %output, i32 %request.slot,
    i32 %generation, i32 %object.slot, i64 %object.id,
    i32 %object.version, i64 %offset, i32 %bytes, i32 %workTicket) {
entry:
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
