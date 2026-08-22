// Device-side reject fixtures for the in-kernel claim-consumer contract.
// Each block evaluates one crafted bindings row against the published
// claim table; the host asserts exactly which cases accept and which
// refuse. The kernel never touches lease storage — it tests the guard.
#include "nta/FlashInferKernelPolicy.cuh"

struct FixtureParams {
  std::uint32_t nta_request_slot_offset;
  const long long *nta_claim_bindings;
};

extern "C" __global__ void nta_claim_consumer_fixture(
    nta::abi::RuntimeView *runtime, const long long *bindings,
    int *verdicts) {
  FixtureParams params{0u, bindings};
  static_assert(nta::flashinfer::HasClaimBindingV<FixtureParams>);
  verdicts[blockIdx.x] = nta::flashinfer::bindValidatedClaimConsumer(
                             params, runtime, blockIdx.x)
                             ? 1
                             : 0;
}
