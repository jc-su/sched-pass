; A compiler-generated typed contract without typed markers is not an
; authorized instrumented operator.  The pass must fail closed.
@nta_jit_instrumentation_flags = constant i64 15
@nta_jit_identity_binding = constant i32 1
@nta_jit_demand_binding = constant i32 1
@nta_jit_access_proof = constant i32 3
@nta_jit_granularity_bytes = constant i32 0
@nta_jit_tier_mask = constant i64 63

define ptx_kernel void @typed_contract_without_markers(ptr %output) {
entry:
  store i32 1, ptr %output, align 4
  ret void
}
