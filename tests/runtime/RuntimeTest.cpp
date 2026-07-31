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

    bool undersizedIntentPoolRejected = false;
    try {
      nta::HostRuntime invalid({1, 2, 1, 1});
    } catch (const std::invalid_argument &) {
      undersizedIntentPoolRejected = true;
    }
    require(undersizedIntentPoolRejected,
            "intent pool must cover every independently queued object");
    bool dependencyCapacityRejected = false;
    try {
      nta::HostRuntime invalid({1, 1, 1, 1, 1, 0});
    } catch (const std::invalid_argument &) {
      dependencyCapacityRejected = true;
    }
    require(dependencyCapacityRejected,
            "dependency capacity must be finite and non-zero");

    nta::HostRuntime runtime({4, 3, 3, 4, 2});
    bool uninitializedCancelRejected = false;
    try {
      runtime.cancelRequest(3, 0);
    } catch (const std::invalid_argument &) {
      uninitializedCancelRejected = true;
    }
    require(uninitializedCancelRejected,
            "uninitialized request cancellation must be rejected");

    runtime.setRequest(0, 1001, 7, 2, 3, 9000);
    runtime.setTenantBudget(2, 1ULL << 20U, 3);
    const nta::abi::RequestContext request = runtime.readRequest(0);
    require(request.requestId == 1001 && request.generation == 7,
            "request publication failed");
    require(runtime.readTenant(2).maxOutstandingBytes == (1ULL << 20U),
            "tenant budget publication failed");

    nta::abi::RuntimeView hostView{};
    require(cudaMemcpy(&hostView, runtime.deviceView(), sizeof(hostView),
                       cudaMemcpyDeviceToHost) == cudaSuccess,
            "runtime view download failed");
    require(hostView.objectCapacity == 3 && hostView.replicaCapacity == 6,
            "object and replica capacities were transposed");
    require(hostView.maxDependenciesPerContinuation == 8 &&
                hostView.dependencyCapacity == 32 &&
                hostView.dependencies != nullptr,
            "continuation dependency storage was not installed");
    const std::uint64_t syntheticOutstanding = 4096;
    require(cudaMemcpy(&hostView.requests[0].outstandingBytes,
                       &syntheticOutstanding, sizeof(syntheticOutstanding),
                       cudaMemcpyHostToDevice) == cudaSuccess,
            "request counter injection failed");
    runtime.cancelRequest(0, 7);
    require(runtime.readRequest(0).outstandingBytes == syntheticOutstanding,
            "cancellation reset a live request credit counter");
    bool liveSlotReuseRejected = false;
    try {
      runtime.setRequest(0, 1002, 8);
    } catch (const std::logic_error &) {
      liveSlotReuseRejected = true;
    }
    require(liveSlotReuseRejected,
            "request slot reuse must wait for outstanding acquisition bytes");
    const std::uint64_t zeroOutstanding = 0;
    require(cudaMemcpy(&hostView.requests[0].outstandingBytes, &zeroOutstanding,
                       sizeof(zeroOutstanding), cudaMemcpyHostToDevice) ==
                cudaSuccess,
            "request counter reset failed");

    require(cudaMemcpy(&hostView.tenants[2].outstandingBytes,
                       &syntheticOutstanding, sizeof(syntheticOutstanding),
                       cudaMemcpyHostToDevice) == cudaSuccess,
            "tenant counter injection failed");
    runtime.setTenantBudget(2, 1ULL << 21U, 4);
    require(runtime.readTenant(2).outstandingBytes == syntheticOutstanding,
            "tenant policy update reset a live credit counter");
    require(cudaMemcpy(&hostView.tenants[2].outstandingBytes, &zeroOutstanding,
                       sizeof(zeroOutstanding), cudaMemcpyHostToDevice) ==
                cudaSuccess,
            "tenant counter reset failed");

    std::array<std::byte, 4096> contents{};
    for (std::size_t i = 0; i < contents.size(); ++i) {
      contents[i] = std::byte(i & 0xffU);
    }

    const std::array<nta::HostReplicaSpec, 2> replicaSpecs{{
        {contents, nta::Placement::HostMapped},
        {contents, nta::Placement::Hbm},
    }};
    const nta::ObjectHandle hbm = runtime.installReplicatedObject(
        0, 2001, 1, replicaSpecs);
    const nta::ObjectHandle mapped =
        runtime.installObject(1, 2002, 2, contents, nta::Placement::HostMapped);
    const nta::ObjectHandle staged =
        runtime.installObject(2, 2003, 3, contents, nta::Placement::HostStaged);

    require(hbm.directDeviceBase != nullptr,
            "HBM object must expose a direct pointer");
    require(runtime.readObject(0).replicaCount == 2 &&
                runtime.readReplica(0, 1).sourceKind ==
                    static_cast<std::uint32_t>(
                        nta::abi::SourceKind::Hbm) &&
                reinterpret_cast<std::uint64_t>(hbm.directDeviceBase) ==
                    runtime.readReplica(0, 1).sourceAddress,
            "replicated object directory was not installed");
    require(mapped.directDeviceBase != nullptr,
            "mapped host object must expose a direct pointer");
    require(staged.directDeviceBase == nullptr,
            "staged host object must enter the acquisition path");

    const nta::abi::ObjectEntry stagedEntry = runtime.readObject(2);
    const nta::abi::ReplicaEntry stagedReplica = runtime.readReplica(2);
    require(stagedReplica.sourceAddress != 0 && stagedEntry.stagingAddress != 0,
            "staged object addresses were not installed");
    require(stagedEntry.state ==
                static_cast<std::uint32_t>(nta::abi::ObjectState::New),
            "staged object must begin nonresident");
    const auto *const replicaMap = reinterpret_cast<const void *>(0x1000ULL);
    const auto *const stagingMap = reinterpret_cast<const void *>(0x2000ULL);
    runtime.bindTensorMaps(2, 0, replicaMap, stagingMap);
    require(runtime.readReplica(2).tensorMapAddress == 0x1000ULL &&
                runtime.readObject(2).stagingTensorMapAddress == 0x2000ULL,
            "tensor-map binding was not published");
    const nta::abi::IntentPool intentPool = runtime.readIntentPool();
    require(intentPool.capacity == runtime.config().intentCapacity,
            "intent pool capacity was not installed");

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
