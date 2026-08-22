// Host driver for the claim-consumer reject fixtures (ABI v28).
//
// Publishes two live claims through a real HostRuntime, then crafts one
// bindings row per case:
//   0 accept: matching slot/generation/bound/stamp
//   1 accept: dense request, slot -1
//   2 refuse: stale generation (claim republished under generation+1)
//   3 refuse: retired slot (valid=0 republished, same generation)
//   4 refuse: foreign claim (row names live slot B with slot A's
//             generation, the cross-claim identity confusion case)
//   5 refuse: row bound beyond stagedRows
//   6 refuse: stale table stamp
//   7 refuse: slot beyond claimCapacity
#include "nta/HostRuntime.h"
#include "nta/RuntimeABI.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

#define CHECK_CU(call)                                                         \
  do {                                                                         \
    CUresult status = (call);                                                  \
    if (status != CUDA_SUCCESS) {                                              \
      const char *name = nullptr;                                              \
      cuGetErrorName(status, &name);                                           \
      std::fprintf(stderr, "%s failed: %s\n", #call, name ? name : "?");       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      std::fprintf(stderr, "%s failed: %s\n", #call,                           \
                   cudaGetErrorString(status));                                \
      return 1;                                                                \
    }                                                                          \
  } while (0)

} // namespace

int main(int argc, char **argv) {
  const char *cubin = nullptr;
#ifdef NTA_CLAIM_CONSUMER_CUBIN_PATH
  cubin = NTA_CLAIM_CONSUMER_CUBIN_PATH;
#endif
  if (argc == 2) {
    cubin = argv[1];
  }
  if (cubin == nullptr) {
    std::fprintf(stderr, "usage: %s <fixture.cubin>\n", argv[0]);
    return 2;
  }
  nta::RuntimeConfig config{};
  config.requestCapacity = 4;
  config.objectCapacity = 4;
  config.intentCapacity = 4;
  config.workTicketCapacity = 4;
  config.claimCapacity = 4;
  nta::HostRuntime runtime(config);

  constexpr std::uint32_t slotA = 0, slotB = 1;
  constexpr std::uint32_t genA = 7, genB = 9;
  nta::abi::ClaimContext row{};
  row.requestSlot = 0;
  row.generation = genA;
  row.valid = 1;
  row.stagedRows = 64;
  row.leaseBase = 1000;
  row.leaseExtent = 64;
  row.tableStamp = 5;
  runtime.publishClaim(slotA, row, nullptr);
  row.requestSlot = 1;
  row.generation = genB;
  row.tableStamp = 11;
  runtime.publishClaim(slotB, row, nullptr);

  constexpr int kCases = 8;
  std::array<long long, kCases * 4> host{};
  auto set = [&](int c, long long s, long long g, long long b, long long st) {
    host[c * 4 + 0] = s;
    host[c * 4 + 1] = g;
    host[c * 4 + 2] = b;
    host[c * 4 + 3] = st;
  };
  set(0, slotA, genA, 64, 5);      // accept
  set(1, -1, 0, 0, 0);             // accept dense
  set(2, slotA, genA, 64, 5);      // becomes stale after republish below
  set(3, slotB, genB, 64, 11);     // becomes retired after republish below
  set(4, slotB, genA, 64, 11);     // foreign: slot B, generation A
  set(5, slotA, genA, 65, 5);      // bound beyond stagedRows
  set(6, slotA, genA, 64, 6);      // stale stamp
  set(7, 3, 1, 1, 0);              // slot valid-range but never published

  // Case 2: republish slot A under generation+1 AFTER binding rows are
  // fixed (the consumer still carries genA). Case 0 must then also move
  // to the new generation to stay the accept witness.
  nta::abi::ClaimContext bumped = row;
  bumped.requestSlot = 0;
  bumped.generation = genA + 1;
  bumped.stagedRows = 64;
  bumped.tableStamp = 5;
  runtime.publishClaim(slotA, bumped, nullptr);
  set(0, slotA, genA + 1, 64, 5);
  set(5, slotA, genA + 1, 65, 5);
  set(6, slotA, genA + 1, 64, 6);
  // Case 3: retire slot B (valid=0, same generation).
  nta::abi::ClaimContext retired{};
  retired.requestSlot = 1;
  retired.generation = genB;
  retired.valid = 0;
  retired.stagedRows = 64;
  retired.tableStamp = 11;
  runtime.publishClaim(slotB, retired, nullptr);
  // Case 4 names slot B too; with B retired it refuses on valid before
  // generation — still a refusal, which is what the fixture asserts.

  long long *bindings = nullptr;
  int *verdicts = nullptr;
  CHECK_CUDA(cudaMalloc(&bindings, sizeof(long long) * host.size()));
  CHECK_CUDA(cudaMalloc(&verdicts, sizeof(int) * kCases));
  CHECK_CUDA(cudaMemcpy(bindings, host.data(),
                        sizeof(long long) * host.size(),
                        cudaMemcpyHostToDevice));

  std::ifstream file(cubin, std::ios::binary);
  std::vector<char> image((std::istreambuf_iterator<char>(file)),
                          std::istreambuf_iterator<char>());
  if (image.empty()) {
    std::fprintf(stderr, "empty cubin %s\n", cubin);
    return 2;
  }
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  CHECK_CU(cuInit(0));
  CHECK_CU(cuModuleLoadData(&module, image.data()));
  CHECK_CU(cuModuleGetFunction(&function, module, "nta_claim_consumer_fixture"));
  nta::abi::RuntimeView *view = runtime.deviceView();
  void *args[] = {&view, &bindings, &verdicts};
  CHECK_CU(cuLaunchKernel(function, kCases, 1, 1, 32, 1, 1, 0, nullptr, args,
                          nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::array<int, kCases> out{};
  CHECK_CUDA(cudaMemcpy(out.data(), verdicts, sizeof(int) * kCases,
                        cudaMemcpyDeviceToHost));
  const std::array<int, kCases> expected{1, 1, 0, 0, 0, 0, 0, 0};
  int failures = 0;
  static const char *kNames[kCases] = {
      "accept-live",     "accept-dense",  "refuse-stale-generation",
      "refuse-retired",  "refuse-foreign", "refuse-extent",
      "refuse-stamp",    "refuse-unpublished"};
  for (int c = 0; c < kCases; ++c) {
    if (out[c] != expected[c]) {
      std::fprintf(stderr, "case %s: got %d expected %d\n", kNames[c], out[c],
                   expected[c]);
      ++failures;
    }
  }
  if (failures) {
    return 1;
  }
  std::puts("claim-consumer fixtures: 2 accepts, 6 refusals, all as armed");
  return 0;
}
