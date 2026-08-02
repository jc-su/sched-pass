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
nta_jit_invalidate_cached_objects(void *runtime, std::uint32_t firstObject,
                                  std::uint32_t objectCount,
                                  cudaStream_t stream) {
  constexpr std::uint32_t threads = 256;
  if (runtime == nullptr || objectCount == 0) {
    return cudaErrorInvalidValue;
  }
  nta_invalidate_cached_objects<<<(objectCount + threads - 1U) / threads,
                                  threads, 0, stream>>>(
      static_cast<nta::abi::RuntimeView *>(runtime), firstObject, objectCount);
  return nta::jit::launchStatus();
}

extern "C" __attribute__((visibility("default"))) cudaError_t
nta_jit_reset_epoch(void *runtime, std::uint32_t objectCount,
                    std::uint32_t workTicketCount, cudaStream_t stream) {
  constexpr std::uint32_t threads = 256;
  const std::uint32_t count = std::max(objectCount, workTicketCount);
  if (runtime == nullptr || count == 0) {
    return cudaErrorInvalidValue;
  }
  nta_reset_epoch<<<(count + threads - 1U) / threads, threads, 0, stream>>>(
      static_cast<nta::abi::RuntimeView *>(runtime), objectCount,
      workTicketCount);
  return nta::jit::launchStatus();
}

extern "C" __attribute__((visibility("default"))) cudaError_t
nta_jit_preload_host(void *runtime, std::uint32_t firstObject,
                     std::uint32_t objectCount, cudaStream_t stream) {
  if (runtime == nullptr || objectCount == 0) {
    return cudaErrorInvalidValue;
  }
  constexpr std::uint32_t blocksPerObject = 2;
  nta_preload_indexed_host<<<objectCount * blocksPerObject, 1024, 0, stream>>>(
      static_cast<nta::abi::RuntimeView *>(runtime), firstObject, objectCount);
  return nta::jit::launchStatus();
}

extern "C" __attribute__((visibility("default"))) cudaError_t
nta_jit_progress_host(void *runtime, std::uint32_t blocks,
                      cudaStream_t stream) {
  if (runtime == nullptr || blocks == 0) {
    return cudaErrorInvalidValue;
  }
  nta_progress_host_staging<<<blocks, 1024, 0, stream>>>(
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
  nta_publish_ready<<<1, threads, 0, stream>>>(
      static_cast<nta::abi::RuntimeView *>(runtime), pendingBudget);
  return nta::jit::launchStatus();
}

extern "C" __attribute__((visibility("default"))) cudaError_t
nta_jit_complete_launched(void *runtime, std::uint32_t workTicketCount,
                          cudaStream_t stream) {
  constexpr std::uint32_t threads = 256;
  if (runtime == nullptr || workTicketCount == 0) {
    return cudaErrorInvalidValue;
  }
  nta_complete_launched<<<(workTicketCount + threads - 1U) / threads, threads,
                          0, stream>>>(
      static_cast<nta::abi::RuntimeView *>(runtime), workTicketCount);
  return nta::jit::launchStatus();
}
