#include "nta/HostRuntime.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

} // namespace

int main() {
  try {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
      std::cout << "SKIP: no CUDA device\n";
      return 0;
    }

    nta::HostRuntime runtime({4, 3, 3, 4});
    bool uninitializedCancelRejected = false;
    try {
      runtime.cancelRequest(3, 0);
    } catch (const std::invalid_argument &) {
      uninitializedCancelRejected = true;
    }
    require(uninitializedCancelRejected,
            "uninitialized request cancellation must be rejected");

    runtime.setRequest(0, 1001, 7, 2, 3, 9000);
    const nta::abi::RequestContext request = runtime.readRequest(0);
    require(request.requestId == 1001 && request.generation == 7,
            "request publication failed");

    std::array<std::byte, 4096> contents{};
    for (std::size_t i = 0; i < contents.size(); ++i) {
      contents[i] = std::byte(i & 0xffU);
    }

    const nta::ObjectHandle hbm =
        runtime.installObject(0, 2001, 1, contents, nta::Placement::Hbm);
    const nta::ObjectHandle mapped =
        runtime.installObject(1, 2002, 2, contents, nta::Placement::HostMapped);
    const nta::ObjectHandle staged =
        runtime.installObject(2, 2003, 3, contents, nta::Placement::HostStaged);

    require(hbm.directDeviceBase != nullptr,
            "HBM object must expose a direct pointer");
    require(mapped.directDeviceBase != nullptr,
            "mapped host object must expose a direct pointer");
    require(staged.directDeviceBase == nullptr,
            "staged host object must enter the acquisition path");

    const nta::abi::ObjectEntry stagedEntry = runtime.readObject(2);
    require(stagedEntry.sourceAddress != 0 && stagedEntry.stagingAddress != 0,
            "staged object addresses were not installed");
    require(stagedEntry.state ==
                static_cast<std::uint32_t>(nta::abi::ObjectState::New),
            "staged object must begin nonresident");

    runtime.cancelRequest(0, 7);
    require(runtime.readRequest(0).cancelled == 1,
            "request cancellation was not published");

    bool staleCancelRejected = false;
    try {
      runtime.cancelRequest(0, 6);
    } catch (const std::invalid_argument &) {
      staleCancelRejected = true;
    }
    require(staleCancelRejected,
            "stale generation cancellation must be rejected");

    std::cout << "NTA host runtime allocation/state tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "runtime test failed: " << error.what() << '\n';
    return 1;
  }
}
