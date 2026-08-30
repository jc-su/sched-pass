#include "nta/HostRuntime.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef NTA_DEPENDENCY_RACE_CUBIN_PATH
#error "NTA_DEPENDENCY_RACE_CUBIN_PATH must identify the test device image"
#endif

namespace {

void checkCuda(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

void checkDriver(CUresult result, const char *operation) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char *description = nullptr;
  (void)cuGetErrorString(result, &description);
  throw std::runtime_error(
      std::string(operation) + ": " +
      (description == nullptr ? "unknown error" : description));
}

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

} // namespace

int main() {
  CUmodule module = nullptr;
  std::uint32_t *deviceObservation = nullptr;
  try {
    nta::RuntimeConfig config{4, 4, 4, 4, 1, 1, -1, false, 1};
    nta::HostRuntime runtime(config);
    runtime.setRequest(0, 42, 3);
    runtime.setRequest(1, 43, 4);

    checkDriver(cuModuleLoad(&module, NTA_DEPENDENCY_RACE_CUBIN_PATH),
                "cuModuleLoad");
    CUfunction kernel = nullptr;
    checkDriver(cuModuleGetFunction(&kernel, module,
                                    "nta_test_dependency_arrival_race"),
                "cuModuleGetFunction");
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&deviceObservation),
                         24 * sizeof(std::uint32_t)),
              "cudaMalloc observation");
    CUdeviceptr runtimeAddress =
        reinterpret_cast<CUdeviceptr>(runtime.deviceView());
    CUdeviceptr observationAddress =
        reinterpret_cast<CUdeviceptr>(deviceObservation);
    void *arguments[] = {&runtimeAddress, &observationAddress};
    checkDriver(cuLaunchKernel(kernel, 1, 1, 1, 32, 1, 1, 0, nullptr, arguments,
                               nullptr),
                "cuLaunchKernel");

    std::array<std::uint32_t, 24> observation{};
    checkCuda(cudaMemcpy(observation.data(), deviceObservation,
                         sizeof(observation), cudaMemcpyDeviceToHost),
              "download dependency-race observation");

    require(observation[0] == 0, "arrival did not satisfy the dependency");
    require(observation[1] == 1 && observation[2] == 1,
            "arrival did not publish exactly one changed ticket");
    require(observation[3] == 1,
            "dependency satisfaction was not recorded exactly once");
    require(observation[4] ==
                static_cast<std::uint32_t>(nta::abi::WorkTicketState::Pending),
            "race test did not leave a publishable pending ticket");
    require(observation[5] == 1 && observation[6] == 1,
            "request progress did not account for the pending ticket");
    require(observation[7] == 0, "race test overflowed the changed queue");

    checkDriver(
        cuModuleGetFunction(&kernel, module, "nta_test_intent_deadline_queue"),
        "cuModuleGetFunction deadline queue");
    checkCuda(cudaMemset(deviceObservation, 0, sizeof(observation)),
              "clear deadline-queue observation");
    checkDriver(cuLaunchKernel(kernel, 1, 1, 1, 32, 1, 1, 0, nullptr, arguments,
                               nullptr),
                "cuLaunchKernel deadline queue");
    checkCuda(cudaMemcpy(observation.data(), deviceObservation,
                         sizeof(observation), cudaMemcpyDeviceToHost),
              "download deadline-queue observation");
    require(observation[0] == 2 && observation[1] == 3 && observation[2] == 1 &&
                observation[3] == 0 && observation[4] == nta::abi::InvalidIndex,
            "intent queue did not order absolute deadlines before priority and "
            "best effort");
    require(observation[5] == 0 && observation[6] == 0 && observation[7] == 1 &&
                observation[8] == nta::abi::InvalidIndex,
            "equal-deadline FIFO or stable requeue ordering failed");
    require(observation[9] == 0 && observation[10] == 1,
            "critical-service deadline and best-effort tie breakers were not "
            "explicit");
    require(observation[11] == 1 && observation[12] == nta::abi::InvalidIndex &&
                observation[13] == static_cast<std::uint32_t>(
                                       nta::abi::IntentQueueState::Free),
            "stale intent generation was not discarded safely");
    require(observation[14] == 0 && observation[15] == nta::abi::InvalidIndex &&
                observation[16] == 0,
            "stale heap node aliased a reused intent slot");
    require(observation[17] == 0, "intent deadline queue setup failed");

    checkDriver(cuModuleGetFunction(&kernel, module,
                                    "nta_test_intent_queue_concurrency"),
                "cuModuleGetFunction concurrent intent queue");
    checkCuda(cudaMemset(deviceObservation, 0, sizeof(observation)),
              "clear concurrent queue observation");
    checkDriver(cuLaunchKernel(kernel, 1, 1, 1, 32, 1, 1, 0, nullptr, arguments,
                               nullptr),
                "cuLaunchKernel concurrent intent queue");
    checkCuda(cudaMemcpy(observation.data(), deviceObservation,
                         sizeof(observation), cudaMemcpyDeviceToHost),
              "download concurrent queue observation");
    require(std::all_of(observation.begin(), observation.begin() + 4,
                        [](std::uint32_t queued) { return queued == 1; }) &&
                observation[4] == 4,
            "concurrent producers did not preserve bounded queue capacity");
    std::array<std::uint32_t, 4> concurrentlyPopped{
        observation[5], observation[6], observation[7], observation[8]};
    std::sort(concurrentlyPopped.begin(), concurrentlyPopped.end());
    require(concurrentlyPopped == std::array<std::uint32_t, 4>{0, 1, 2, 3} &&
                observation[9] == 0 && observation[10] == 0,
            "concurrent consumers duplicated, lost, or stranded an intent");
    require(observation[11] == 4 && observation[12] == 1 &&
                observation[13] == 3 && observation[14] == 2 &&
                observation[15] == 0 && observation[16] == 0 &&
                observation[17] == 0,
            "concurrent requeue did not restore exact deadline order");

    checkDriver(cuModuleGetFunction(&kernel, module,
                                    "nta_test_constrained_edf_dispatch"),
                "cuModuleGetFunction constrained EDF dispatch");
    checkCuda(cudaMemset(deviceObservation, 0, sizeof(observation)),
              "clear constrained-EDF observation");
    checkDriver(cuLaunchKernel(kernel, 1, 1, 1, 32, 1, 1, 0, nullptr, arguments,
                               nullptr),
                "cuLaunchKernel constrained EDF dispatch");
    checkCuda(cudaMemcpy(observation.data(), deviceObservation,
                         sizeof(observation), cudaMemcpyDeviceToHost),
              "download constrained-EDF observation");
    require(observation[0] == 1 && observation[1] == 1 && observation[2] == 1 &&
                observation[3] == 1 && observation[4] == 4096 &&
                observation[5] == 4096 && observation[6] == 4096,
            "constrained EDF did not skip a blocked root and reserve the "
            "feasible request atomically");
    require(
        observation[7] == 0 && observation[8] == 1 && observation[9] == 4096 &&
            observation[10] == 0 && observation[11] == 0 &&
            observation[12] == 0 && observation[13] == 0 &&
            observation[14] == 0,
        "constrained EDF did not retire the blocked root after credit release");

    checkDriver(
        cuModuleGetFunction(&kernel, module,
                            "nta_test_ordered_intent_window_validation"),
        "cuModuleGetFunction ordered intent validation");
    checkCuda(cudaMemset(deviceObservation, 0, sizeof(observation)),
              "clear ordered-intent observation");
    checkDriver(cuLaunchKernel(kernel, 1, 1, 1, 256, 1, 1, 0, nullptr,
                               arguments, nullptr),
                "cuLaunchKernel ordered intent validation");
    checkCuda(cudaMemcpy(observation.data(), deviceObservation,
                         sizeof(observation), cudaMemcpyDeviceToHost),
              "download ordered-intent observation");
    require(std::equal(
                observation.begin(), observation.begin() + 11,
                std::array<std::uint32_t, 11>{1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1}
                    .begin()),
            "ordered intent proof accepted an unsorted deadline/priority image "
            "or lost its persistent cursor");

    nta::RuntimeConfig chunkConfig{1, 513, 513, 1, 1, 1, -1, false, 1};
    nta::HostRuntime chunkRuntime(chunkConfig);
    chunkRuntime.setRequest(0, 99, 1);
    CUdeviceptr chunkRuntimeAddress =
        reinterpret_cast<CUdeviceptr>(chunkRuntime.deviceView());
    void *chunkArguments[] = {&chunkRuntimeAddress, &observationAddress};
    checkDriver(cuModuleGetFunction(
                    &kernel, module,
                    "nta_test_ordered_intent_window_validation_chunks"),
                "cuModuleGetFunction chunked ordered intent validation");
    checkCuda(cudaMemset(deviceObservation, 0, sizeof(observation)),
              "clear chunked ordered-intent observation");
    checkDriver(cuLaunchKernel(kernel, 1, 1, 1, 256, 1, 1, 0, nullptr,
                               chunkArguments, nullptr),
                "cuLaunchKernel chunked ordered intent validation");
    checkCuda(cudaMemcpy(observation.data(), deviceObservation,
                         sizeof(observation), cudaMemcpyDeviceToHost),
              "download chunked ordered-intent observation");
    require(std::equal(
                observation.begin(), observation.begin() + 7,
                std::array<std::uint32_t, 7>{1, 1, 1, 0, 0, 1, 1}.begin()),
            "ordered intent proof lost ordering across a 256-slot chunk edge");

    checkDriver(cuModuleGetFunction(&kernel, module,
                                    "nta_test_request_progress_fail_closed"),
                "cuModuleGetFunction request progress fail closed");
    checkCuda(cudaMemset(deviceObservation, 0, sizeof(observation)),
              "clear request-progress observation");
    checkDriver(cuLaunchKernel(kernel, 1, 1, 1, 32, 1, 1, 0, nullptr, arguments,
                               nullptr),
                "cuLaunchKernel request progress fail closed");
    checkCuda(cudaMemcpy(observation.data(), deviceObservation,
                         sizeof(observation), cudaMemcpyDeviceToHost),
              "download request-progress observation");
    require(observation[0] == 1 && observation[1] == 0 && observation[2] == 1 &&
                observation[3] == 2 && observation[4] == 1 &&
                observation[5] == 1 && observation[6] == 0 &&
                observation[7] == 0 && observation[8] == 0,
            "request progress did not isolate stale generations and fail on "
            "underflow");

    checkDriver(cuModuleGetFunction(&kernel, module,
                                    "nta_test_request_reduction_groups"),
                "cuModuleGetFunction request reduction groups");
    checkCuda(cudaMemset(deviceObservation, 0, sizeof(observation)),
              "clear request-reduction observation");
    checkDriver(cuLaunchKernel(kernel, 1, 1, 1, 32, 1, 1, 0, nullptr, arguments,
                               nullptr),
                "cuLaunchKernel request reduction groups");
    checkCuda(cudaMemcpy(observation.data(), deviceObservation,
                         sizeof(observation), cudaMemcpyDeviceToHost),
              "download request-reduction observation");
    require(std::equal(
                observation.begin(), observation.begin() + 8,
                std::array<std::uint32_t, 8>{2, 1, 1, 1, 1, 0, 2, 1}.begin()),
            "request reduction counters did not conserve terminal work");

    checkDriver(cuModuleGetFunction(&kernel, module,
                                    "nta_test_cancelled_intent_credit_release"),
                "cuModuleGetFunction cancelled intent credit release");
    checkCuda(cudaMemset(deviceObservation, 0, sizeof(observation)),
              "clear cancelled-credit observation");
    checkDriver(cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, arguments,
                               nullptr),
                "cuLaunchKernel cancelled intent credit release");
    checkCuda(cudaMemcpy(observation.data(), deviceObservation,
                         sizeof(observation), cudaMemcpyDeviceToHost),
              "download cancelled-credit observation");
    require(std::equal(observation.begin(), observation.begin() + 6,
                       std::array<std::uint32_t, 6>{1, 0, 0, 0, 0, 0}.begin()),
            "cancellation leaked indexed request, tenant, or backend credits");

    checkDriver(
        cuModuleGetFunction(&kernel, module,
                            "nta_test_explicit_indexed_claim_credit_lifetime"),
        "cuModuleGetFunction explicit indexed claim credit lifetime");
    checkCuda(cudaMemset(deviceObservation, 0, sizeof(observation)),
              "clear explicit indexed claim observation");
    checkDriver(cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, arguments,
                               nullptr),
                "cuLaunchKernel explicit indexed claim credit lifetime");
    checkCuda(cudaMemcpy(observation.data(), deviceObservation,
                         sizeof(observation), cudaMemcpyDeviceToHost),
              "download fast indexed claim observation");
    require(std::equal(
                observation.begin(), observation.begin() + 13,
                std::array<std::uint32_t, 13>{
                    1,
                    static_cast<std::uint32_t>(nta::abi::ObjectState::Issued),
                    2,
                    4096,
                    4096,
                    4096,
                    4096,
                    4096,
                    0,
                    0,
                    0,
                    0,
                    0,
                }
                    .begin()),
            "explicit indexed claim did not persist and release exact byte "
            "credits");

    checkDriver(cuModuleGetFunction(&kernel, module,
                                    "nta_test_indexed_publication_topology"),
                "cuModuleGetFunction indexed publication topology");
    checkCuda(cudaMemset(deviceObservation, 0, sizeof(observation)),
              "clear indexed-publication observation");
    checkDriver(cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, arguments,
                               nullptr),
                "cuLaunchKernel indexed publication topology");
    checkCuda(cudaMemcpy(observation.data(), deviceObservation,
                         sizeof(observation), cudaMemcpyDeviceToHost),
              "download indexed-publication observation");
    require(
        observation[0] == 1 && observation[1] == 1 && observation[2] == 0 &&
            observation[3] == 1 &&
            observation[4] ==
                static_cast<std::uint32_t>(nta::abi::WorkTicketState::Ready) &&
            observation[5] == 0,
        "private indexed object did not publish its unique consumer directly");
    require(observation[6] == 0 && observation[7] == 0 && observation[8] == 0 &&
                observation[9] == 0 && observation[10] == 1,
            "shared indexed object bypassed the high-fanout full-scan path");
  } catch (const std::exception &error) {
    if (deviceObservation != nullptr) {
      (void)cudaFree(deviceObservation);
    }
    if (module != nullptr) {
      (void)cuModuleUnload(module);
    }
    std::cerr << "dependency-arrival race test failed: " << error.what()
              << '\n';
    return 1;
  }
  checkCuda(cudaFree(deviceObservation), "cudaFree observation");
  checkDriver(cuModuleUnload(module), "cuModuleUnload");
  return 0;
}
