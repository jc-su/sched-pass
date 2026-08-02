#include "nta/HostRuntime.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

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
  throw std::runtime_error(std::string(operation) + ": " +
                           (description == nullptr ? "unknown error"
                                                   : description));
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
                         8 * sizeof(std::uint32_t)),
              "cudaMalloc observation");
    CUdeviceptr runtimeAddress =
        reinterpret_cast<CUdeviceptr>(runtime.deviceView());
    CUdeviceptr observationAddress =
        reinterpret_cast<CUdeviceptr>(deviceObservation);
    void *arguments[] = {&runtimeAddress, &observationAddress};
    checkDriver(cuLaunchKernel(kernel, 1, 1, 1, 32, 1, 1, 0, nullptr,
                               arguments, nullptr),
                "cuLaunchKernel");

    std::array<std::uint32_t, 8> observation{};
    checkCuda(cudaMemcpy(observation.data(), deviceObservation,
                         sizeof(observation), cudaMemcpyDeviceToHost),
              "download dependency-race observation");

    require(observation[0] == 0, "arrival did not satisfy the dependency");
    require(observation[1] == 1 && observation[2] == 1,
            "arrival did not publish exactly one changed ticket");
    require(observation[3] == 1,
            "dependency satisfaction was not recorded exactly once");
    require(observation[4] == static_cast<std::uint32_t>(
                                  nta::abi::WorkTicketState::Pending),
            "race test did not leave a publishable pending ticket");
    require(observation[5] == 1 && observation[6] == 1,
            "request progress did not account for the pending ticket");
    require(observation[7] == 0, "race test overflowed the changed queue");

    checkDriver(cuModuleGetFunction(&kernel, module,
                                    "nta_test_intent_priority_queue"),
                "cuModuleGetFunction priority queue");
    checkCuda(cudaMemset(deviceObservation, 0, sizeof(observation)),
              "clear priority-queue observation");
    checkDriver(cuLaunchKernel(kernel, 1, 1, 1, 32, 1, 1, 0, nullptr,
                               arguments, nullptr),
                "cuLaunchKernel priority queue");
    checkCuda(cudaMemcpy(observation.data(), deviceObservation,
                         sizeof(observation), cudaMemcpyDeviceToHost),
              "download priority-queue observation");
    require(observation[0] == 3 && observation[1] == 3 &&
                observation[2] == 1 && observation[3] == 2 &&
                observation[4] == 0 &&
                observation[5] == nta::abi::InvalidIndex,
            "intent queue did not preserve urgency and tagged requeue order");
    require(observation[7] == 0, "intent priority queue setup failed");

    checkDriver(cuModuleGetFunction(&kernel, module,
                                    "nta_test_request_reduction_groups"),
                "cuModuleGetFunction request reduction groups");
    checkCuda(cudaMemset(deviceObservation, 0, sizeof(observation)),
              "clear request-reduction observation");
    checkDriver(cuLaunchKernel(kernel, 1, 1, 1, 32, 1, 1, 0, nullptr,
                               arguments, nullptr),
                "cuLaunchKernel request reduction groups");
    checkCuda(cudaMemcpy(observation.data(), deviceObservation,
                         sizeof(observation), cudaMemcpyDeviceToHost),
              "download request-reduction observation");
    require(observation == std::array<std::uint32_t, 8>{2, 1, 1, 1, 1, 0, 2, 1},
            "request reduction counters did not conserve terminal work");
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
