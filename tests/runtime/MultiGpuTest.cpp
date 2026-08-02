#include "nta/DeviceWorkPlan.h"
#include "nta/HostRuntime.h"
#include "nta/WorkPlan.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

namespace {

void checkCuda(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

struct DeviceFixture {
  explicit DeviceFixture(int device, std::span<const std::byte> contents)
      : runtime(makeConfig(device)) {
    runtime.setRequest(0, 1000U + static_cast<std::uint64_t>(device), 7);
    object =
        runtime.installObject(0, 2000U + static_cast<std::uint64_t>(device), 3,
                              contents, nta::Placement::Hbm);
    require(object.directDeviceBase != nullptr,
            "multi-GPU fixture did not install HBM data");

    nta::WorkPlanBuilder builder(1);
    const std::uint32_t request = builder.addRequest({0, 7});
    const nta::abi::AcquireRequirement dependency = nta::makeRequirement(
        {reinterpret_cast<std::uint64_t>(object.directDeviceBase), 0,
         2000U + static_cast<std::uint64_t>(device), 0, 3,
         static_cast<std::uint32_t>(contents.size())});
    (void)builder.addWork(
        request, 0,
        std::span<const nta::abi::AcquireRequirement>(&dependency, 1));
    plan = std::make_unique<nta::DeviceWorkPlan>(builder.finish(), device);
  }

  static nta::RuntimeConfig makeConfig(int device) {
    nta::RuntimeConfig config{1, 1, 1, 1};
    config.deviceOrdinal = device;
    return config;
  }

  nta::HostRuntime runtime;
  nta::ObjectHandle object{};
  std::unique_ptr<nta::DeviceWorkPlan> plan;
};

} // namespace

int main() {
  try {
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount < 2) {
      std::cout << "SKIP: multi-GPU qualification requires two CUDA devices\n";
      return 77;
    }

    int originalDevice = 0;
    checkCuda(cudaGetDevice(&originalDevice), "cudaGetDevice");
    std::array<std::byte, 4096> contents{};
    for (std::size_t index = 0; index < contents.size(); ++index) {
      contents[index] = std::byte(index & 0xffU);
    }

    DeviceFixture first(0, contents);
    DeviceFixture second(1, contents);
    require(first.runtime.deviceOrdinal() == 0 &&
                second.runtime.deviceOrdinal() == 1 &&
                first.plan->deviceOrdinal() == 0 &&
                second.plan->deviceOrdinal() == 1,
            "runtime or device plan lost its CUDA device owner");

    cudaPointerAttributes firstAttributes{};
    cudaPointerAttributes secondAttributes{};
    checkCuda(cudaPointerGetAttributes(&firstAttributes,
                                       first.object.directDeviceBase),
              "cudaPointerGetAttributes GPU 0");
    checkCuda(cudaPointerGetAttributes(&secondAttributes,
                                       second.object.directDeviceBase),
              "cudaPointerGetAttributes GPU 1");
    require(firstAttributes.type == cudaMemoryTypeDevice &&
                firstAttributes.device == 0 &&
                secondAttributes.type == cudaMemoryTypeDevice &&
                secondAttributes.device == 1,
            "object allocations were installed on the wrong CUDA device");

    require(first.runtime.readRequest(0).requestId == 1000 &&
                second.runtime.readRequest(0).requestId == 1001,
            "cross-device runtime lookup returned the wrong request");
    int restoredDevice = -1;
    checkCuda(cudaGetDevice(&restoredDevice), "cudaGetDevice restored");
    require(restoredDevice == originalDevice,
            "runtime operations did not restore the caller CUDA device");

    std::cout << "multi-GPU runtime ownership validation passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "multi-GPU runtime validation failed: " << error.what()
              << '\n';
    return 1;
  }
}
