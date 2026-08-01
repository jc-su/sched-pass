#pragma once

#if !NTA_DEVICE_PHASE_KERNELS
#error "nta JIT runtime wrappers require NTA_DEVICE_PHASE_KERNELS=1"
#endif

#include "runtime/device/Acquire.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace nta::jit {

inline cudaError_t launchStatus() { return cudaPeekAtLastError(); }

} // namespace nta::jit

extern "C" __attribute__((visibility("default"))) std::uint32_t
nta_jit_abi_version() {
  return nta::abi::Version;
}

extern "C" __attribute__((visibility("default"))) cudaError_t
nta_jit_reset_epoch(void *runtime, std::uint32_t objectCount,
                    std::uint32_t continuationCount, cudaStream_t stream) {
  constexpr std::uint32_t threads = 256;
  const std::uint32_t count = std::max(objectCount, continuationCount);
  if (runtime == nullptr || count == 0) {
    return cudaErrorInvalidValue;
  }
  nta_reset_epoch<<<(count + threads - 1U) / threads, threads, 0, stream>>>(
      static_cast<nta::abi::RuntimeView *>(runtime), objectCount,
      continuationCount);
  return nta::jit::launchStatus();
}

extern "C" __attribute__((visibility("default"))) cudaError_t
nta_jit_progress_host(void *runtime, std::uint32_t blocks,
                      cudaStream_t stream) {
  if (runtime == nullptr || blocks == 0) {
    return cudaErrorInvalidValue;
  }
  nta_progress_host_staging<<<blocks, 256, 0, stream>>>(
      static_cast<nta::abi::RuntimeView *>(runtime));
  return nta::jit::launchStatus();
}

extern "C" __attribute__((visibility("default"))) cudaError_t
nta_jit_progress_nvme(void *runtime, std::uint32_t issueBudget,
                      std::uint32_t completionBudget, cudaStream_t stream) {
  if (runtime == nullptr || issueBudget == 0 || completionBudget == 0) {
    return cudaErrorInvalidValue;
  }
  nta_progress_nvme<<<1, 32, 0, stream>>>(
      static_cast<nta::abi::RuntimeView *>(runtime), issueBudget,
      completionBudget);
  return nta::jit::launchStatus();
}

extern "C" __attribute__((visibility("default"))) cudaError_t
nta_jit_publish_ready(void *runtime, std::uint32_t pendingBudget,
                      cudaStream_t stream) {
  constexpr std::uint32_t threads = 256;
  if (runtime == nullptr || pendingBudget == 0) {
    return cudaErrorInvalidValue;
  }
  const std::uint32_t blocks =
      std::min(32U, (pendingBudget + threads - 1U) / threads);
  nta_publish_ready<<<blocks, threads, 0, stream>>>(
      static_cast<nta::abi::RuntimeView *>(runtime), pendingBudget);
  return nta::jit::launchStatus();
}

extern "C" __attribute__((visibility("default"))) cudaError_t
nta_jit_complete_launched(void *runtime, std::uint32_t continuationCount,
                          cudaStream_t stream) {
  constexpr std::uint32_t threads = 256;
  if (runtime == nullptr || continuationCount == 0) {
    return cudaErrorInvalidValue;
  }
  nta_complete_launched<<<(continuationCount + threads - 1U) / threads, threads,
                          0, stream>>>(
      static_cast<nta::abi::RuntimeView *>(runtime), continuationCount);
  return nta::jit::launchStatus();
}
