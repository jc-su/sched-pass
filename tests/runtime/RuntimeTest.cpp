#include "nta/DeviceWorkPlan.h"
#include "nta/HostRuntime.h"
#include "nta/NvmeRuntime.h"
#include "nta/WorkPlan.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>

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
    int originalDevice = 0;
    require(cudaGetDevice(&originalDevice) == cudaSuccess,
            "current CUDA device query failed");
    bool invalidDeviceRejected = false;
    try {
      nta::RuntimeConfig invalidDevice{1, 1, 1, 1};
      invalidDevice.deviceOrdinal = deviceCount;
      nta::HostRuntime invalid(invalidDevice);
    } catch (const std::out_of_range &) {
      invalidDeviceRejected = true;
    }
    require(invalidDeviceRejected, "invalid CUDA device ordinal was accepted");

    bool nonVfioNvmeRejected = false;
    try {
      nta::NvmeTransportOptions invalidNvme;
      invalidNvme.endpoint = "/dev/nta_nvme";
      invalidNvme.deviceOrdinal = originalDevice;
      nta::NvmeTransport invalid(std::move(invalidNvme));
    } catch (const std::invalid_argument &) {
      nonVfioNvmeRejected = true;
    }
    require(nonVfioNvmeRejected,
            "NVMe transport accepted a non-VFIO ownership endpoint");

    nta::HostRuntime boundedIntentPool({1, 2, 1, 1});
    require(boundedIntentPool.config().intentCapacity == 1,
            "intent pool must be sized by the active acquisition frontier");
    bool dependencyCapacityRejected = false;
    try {
      nta::HostRuntime invalid({1, 1, 1, 1, 1, 0});
    } catch (const std::invalid_argument &) {
      dependencyCapacityRejected = true;
    }
    require(dependencyCapacityRejected,
            "dependency capacity must be finite and non-zero");

    nta::HostRuntime runtime({4, 4, 4, 4, 2});
    require(runtime.deviceOrdinal() == originalDevice,
            "runtime did not retain its CUDA device owner");
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
    require(hostView.objectCapacity == 4 && hostView.replicaCapacity == 8,
            "object and replica capacities were transposed");
    require(hostView.maxDependenciesPerWorkTicket == 8 &&
                hostView.dependencyCapacity == 32 &&
                hostView.dependencies != nullptr &&
                hostView.workRunnableNs != nullptr &&
                hostView.pendingWorkTickets != nullptr &&
                hostView.pendingCount != nullptr &&
                hostView.objectDependentHeads != nullptr &&
                hostView.dependencyNext != nullptr &&
                hostView.remainingDependencies != nullptr &&
                hostView.changedWorkTickets != nullptr &&
                hostView.changedCount != nullptr &&
                hostView.changedOverflow != nullptr &&
                hostView.requestProgress != nullptr,
            "workTicket dependency storage was not installed");
    const nta::abi::RequestProgress requestProgress =
        runtime.readRequestProgress(0);
    require(requestProgress.requestId == 1001 &&
                requestProgress.generation == 7 &&
                requestProgress.expectedWork == 0,
            "request progress was not initialized with request identity");
    std::uint32_t pendingCount = 1;
    require(cudaMemcpy(&pendingCount, hostView.pendingCount,
                       sizeof(pendingCount),
                       cudaMemcpyDeviceToHost) == cudaSuccess &&
                pendingCount == 0 && runtime.readPendingCount() == 0 &&
                runtime.readPendingIndexCount() == 0,
            "pending workTicket index was not initialized");
    const nta::EpochStatus initialEpoch = runtime.readEpochStatus(4);
    require(initialEpoch.total == 4 && initialEpoch.fresh == 4 &&
                initialEpoch.pending == 0 && !initialEpoch.succeeded() &&
                !initialEpoch.hasFailure(),
            "epoch status did not summarize the workTicket prefix");
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
                       sizeof(zeroOutstanding),
                       cudaMemcpyHostToDevice) == cudaSuccess,
            "request counter reset failed");

    require(cudaMemcpy(&hostView.tenants[2].outstandingBytes,
                       &syntheticOutstanding, sizeof(syntheticOutstanding),
                       cudaMemcpyHostToDevice) == cudaSuccess,
            "tenant counter injection failed");
    runtime.setTenantBudget(2, 1ULL << 21U, 4);
    require(runtime.readTenant(2).outstandingBytes == syntheticOutstanding,
            "tenant policy update reset a live credit counter");
    require(cudaMemcpy(&hostView.tenants[2].outstandingBytes, &zeroOutstanding,
                       sizeof(zeroOutstanding),
                       cudaMemcpyHostToDevice) == cudaSuccess,
            "tenant counter reset failed");

    std::array<std::byte, 4096> contents{};
    for (std::size_t i = 0; i < contents.size(); ++i) {
      contents[i] = std::byte(i & 0xffU);
    }

    const std::array<nta::HostReplicaSpec, 2> replicaSpecs{{
        {contents, nta::Placement::HostMapped},
        {contents, nta::Placement::Hbm},
    }};
    const nta::ObjectHandle hbm =
        runtime.installReplicatedObject(0, 2001, 1, replicaSpecs);
    const nta::ObjectHandle mapped =
        runtime.installObject(1, 2002, 2, contents, nta::Placement::HostMapped);
    const nta::ObjectHandle staged =
        runtime.installObject(2, 2003, 3, contents, nta::Placement::HostStaged);

    require(hbm.directDeviceBase != nullptr,
            "HBM object must expose a direct pointer");
    require(runtime.readObject(0).replicaCount == 2 &&
                runtime.readReplica(0, 1).sourceKind ==
                    static_cast<std::uint32_t>(nta::abi::SourceKind::Hbm) &&
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

    nta::RuntimeConfig stagingConfig{1, 2, 1, 1};
    stagingConfig.stagingByteCapacity = contents.size();
    nta::HostRuntime boundedStaging(stagingConfig);
    boundedStaging.installObject(0, 3001, 1, contents,
                                 nta::Placement::HostStaged);
    const nta::StagingUsage fullUsage = boundedStaging.stagingUsage();
    require(fullUsage.bytes == contents.size() &&
                fullUsage.capacity == contents.size() &&
                fullUsage.highWaterBytes == contents.size(),
            "runtime-owned staging usage is incorrect");
    bool stagingCapacityRejected = false;
    try {
      boundedStaging.installObject(1, 3002, 1, contents,
                                   nta::Placement::HostStaged);
    } catch (const std::runtime_error &) {
      stagingCapacityRejected = true;
    }
    require(stagingCapacityRejected &&
                boundedStaging.stagingUsage().bytes == contents.size(),
            "runtime-owned staging capacity did not fail atomically");
    boundedStaging.installObject(0, 3001, 2, contents,
                                 nta::Placement::HostMapped);
    require(boundedStaging.stagingUsage().bytes == 0,
            "replaced staging allocation remained charged");

    void *borrowedDevice = nullptr;
    require(cudaMalloc(&borrowedDevice, contents.size()) == cudaSuccess,
            "borrowed HBM allocation failed");
    require(cudaMemcpy(borrowedDevice, contents.data(), contents.size(),
                       cudaMemcpyHostToDevice) == cudaSuccess,
            "borrowed HBM upload failed");
    const nta::RegisteredReplicaSpec invalidStagedReplica{
        borrowedDevice, nta::Placement::HostStaged};
    bool missingStagingRejected = false;
    try {
      runtime.registerObject(3, 2004, 4, contents.size(), nullptr,
                             std::span<const nta::RegisteredReplicaSpec>(
                                 &invalidStagedReplica, 1));
    } catch (const std::invalid_argument &) {
      missingStagingRejected = true;
    }
    require(missingStagingRejected,
            "non-owning staged registration accepted no HBM destination");
    const nta::RegisteredReplicaSpec borrowedReplica{borrowedDevice,
                                                     nta::Placement::Hbm};
    const nta::ObjectHandle borrowed = runtime.registerObject(
        3, 2004, 4, contents.size(), nullptr,
        std::span<const nta::RegisteredReplicaSpec>(&borrowedReplica, 1));
    require(borrowed.directDeviceBase == borrowedDevice &&
                runtime.readObject(3).state ==
                    static_cast<std::uint32_t>(nta::abi::ObjectState::Ready),
            "non-owning HBM registration failed");
    require(cudaFree(borrowedDevice) == cudaSuccess,
            "runtime incorrectly took ownership of registered HBM");

    void *indexedHost = nullptr;
    void *indexedHostDevice = nullptr;
    void *indexedStaging = nullptr;
    std::uint32_t *indexedSource = nullptr;
    std::uint32_t *indexedDestination = nullptr;
    require(cudaHostAlloc(&indexedHost, 64, cudaHostAllocMapped) ==
                    cudaSuccess &&
                cudaHostGetDevicePointer(&indexedHostDevice, indexedHost, 0) ==
                    cudaSuccess &&
                cudaMalloc(&indexedStaging, 64) == cudaSuccess &&
                cudaMalloc(reinterpret_cast<void **>(&indexedSource),
                           sizeof(std::uint32_t)) == cudaSuccess &&
                cudaMalloc(reinterpret_cast<void **>(&indexedDestination),
                           sizeof(std::uint32_t)) == cudaSuccess,
            "preacquired indexed allocation failed");
    const nta::IndexedHostObjectSpec preacquired{2005,
                                                 5,
                                                 indexedHostDevice,
                                                 indexedStaging,
                                                 indexedSource,
                                                 indexedDestination,
                                                 1,
                                                 64,
                                                 64,
                                                 64,
                                                 true};
    runtime.registerIndexedHostObjects(
        3, std::span<const nta::IndexedHostObjectSpec>(&preacquired, 1));
    const nta::abi::ObjectEntry preacquiredEntry = runtime.readObject(3);
    require(preacquiredEntry.state ==
                    static_cast<std::uint32_t>(nta::abi::ObjectState::Ready) &&
                preacquiredEntry.selectedReplica == 0,
            "preacquired indexed object was not published ready");
    require(cudaFree(indexedDestination) == cudaSuccess &&
                cudaFree(indexedSource) == cudaSuccess &&
                cudaFree(indexedStaging) == cudaSuccess &&
                cudaFreeHost(indexedHost) == cudaSuccess,
            "preacquired indexed allocation release failed");

    nta::WorkPlanBuilder planBuilder(2);
    const std::uint32_t planRequest = planBuilder.addRequest({0, 7});
    const std::array<nta::abi::AcquireRequirement, 2> planDependencies{{
        nta::makeRequirement(
            {reinterpret_cast<std::uint64_t>(hbm.directDeviceBase), 0, 2001, 0,
             1, static_cast<std::uint32_t>(contents.size())}),
        nta::makeRequirement(
            {0, 0, 2003, 2, 3, static_cast<std::uint32_t>(contents.size())}),
    }};
    (void)planBuilder.addWork(planRequest, 42, planDependencies);
    const nta::WorkPlan hostPlan = planBuilder.finish();
    nta::DeviceWorkPlan devicePlan = runtime.uploadWorkPlan(hostPlan);
    nta::abi::WorkItem uploadedWork{};
    nta::abi::AcquireRequirement uploadedDependency{};
    require(devicePlan.workItemCount() == 1 &&
                devicePlan.dependencyCount() == 2 &&
                cudaMemcpy(&uploadedWork, devicePlan.workItems(),
                           sizeof(uploadedWork),
                           cudaMemcpyDeviceToHost) == cudaSuccess &&
                cudaMemcpy(&uploadedDependency, devicePlan.dependencies() + 1,
                           sizeof(uploadedDependency),
                           cudaMemcpyDeviceToHost) == cudaSuccess &&
                uploadedWork.logicalWork == 42 &&
                uploadedWork.directDependencyCount == 1 &&
                uploadedDependency.objectId == 2003,
            "canonical device work-plan upload failed");
    require(devicePlan.workItemCapacity() == 1 &&
                devicePlan.dependencyCapacity() == 2 &&
                devicePlan.deviceOrdinal() == originalDevice,
            "device work-plan capacities were not retained");

    nta::DeviceWorkPlan reusablePlan(2, 4);
    cudaStream_t uploadStream = nullptr;
    cudaStream_t consumerStream = nullptr;
    require(cudaStreamCreateWithFlags(&uploadStream, cudaStreamNonBlocking) ==
                cudaSuccess,
            "reusable plan stream creation failed");
    require(cudaStreamCreateWithFlags(&consumerStream, cudaStreamNonBlocking) ==
                cudaSuccess,
            "reusable plan consumer stream creation failed");
    const nta::abi::WorkItem *const reusableWorkAddress =
        reusablePlan.workItems();
    const nta::abi::AcquireRequirement *const reusableDependencyAddress =
        reusablePlan.dependencies();
    reusablePlan.uploadAsync(hostPlan, uploadStream);
    reusablePlan.waitOn(consumerStream);
    nta::abi::WorkItem asynchronouslyUploaded{};
    require(cudaMemcpyAsync(&asynchronouslyUploaded, reusablePlan.workItems(),
                            sizeof(asynchronouslyUploaded),
                            cudaMemcpyDeviceToHost,
                            consumerStream) == cudaSuccess &&
                cudaStreamSynchronize(consumerStream) == cudaSuccess &&
                asynchronouslyUploaded.logicalWork == 42,
            "cross-stream work-plan publication failed");
    nta::WorkPlan updatedPlan = hostPlan;
    updatedPlan.workItems[0].logicalWork = 77;
    reusablePlan.uploadAsync(updatedPlan, uploadStream);
    reusablePlan.waitOn(consumerStream);
    require(cudaMemcpyAsync(&asynchronouslyUploaded, reusablePlan.workItems(),
                            sizeof(asynchronouslyUploaded),
                            cudaMemcpyDeviceToHost,
                            consumerStream) == cudaSuccess &&
                cudaStreamSynchronize(consumerStream) == cudaSuccess &&
                asynchronouslyUploaded.logicalWork == 77 &&
                reusablePlan.workItems() == reusableWorkAddress &&
                reusablePlan.dependencies() == reusableDependencyAddress,
            "reusable work-plan allocation changed across updates");
    nta::WorkPlan queuedPlan = hostPlan;
    queuedPlan.workItems[0].logicalWork = 88;
    reusablePlan.uploadAsync(queuedPlan, uploadStream);
    queuedPlan.workItems[0].logicalWork = 99;
    reusablePlan.uploadAsync(queuedPlan, uploadStream);
    reusablePlan.waitOn(consumerStream);
    require(cudaMemcpyAsync(&asynchronouslyUploaded, reusablePlan.workItems(),
                            sizeof(asynchronouslyUploaded),
                            cudaMemcpyDeviceToHost,
                            consumerStream) == cudaSuccess &&
                cudaStreamSynchronize(consumerStream) == cudaSuccess &&
                asynchronouslyUploaded.logicalWork == 99,
            "double-buffered work-plan uploads were not stream ordered");
    require(cudaStreamDestroy(consumerStream) == cudaSuccess,
            "reusable plan consumer stream destruction failed");
    require(cudaStreamDestroy(uploadStream) == cudaSuccess,
            "reusable plan stream destruction failed");

    nta::WorkPlan invalidRuntimePlan = hostPlan;
    invalidRuntimePlan.workItems[0].requestSlot = 3;
    bool invalidRuntimeBindingRejected = false;
    try {
      (void)runtime.uploadWorkPlan(invalidRuntimePlan);
    } catch (const std::invalid_argument &) {
      invalidRuntimeBindingRejected = true;
    }
    require(invalidRuntimeBindingRejected,
            "runtime must reject an uninstalled request binding");
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

    if (deviceCount > 1) {
      const int secondDevice = originalDevice == 0 ? 1 : 0;
      nta::RuntimeConfig secondConfig{1, 1, 1, 1};
      secondConfig.deviceOrdinal = secondDevice;
      nta::HostRuntime secondRuntime(secondConfig);
      secondRuntime.setRequest(0, 9001, 1);
      require(secondRuntime.deviceOrdinal() == secondDevice &&
                  secondRuntime.readRequest(0).requestId == 9001,
              "second GPU runtime did not retain independent state");

      require(cudaSetDevice(secondDevice) == cudaSuccess,
              "cross-device guard test could not switch devices");
      require(runtime.readRequest(0).requestId == 1001,
              "runtime operation used the caller's current CUDA device");
      int restoredDevice = -1;
      require(cudaGetDevice(&restoredDevice) == cudaSuccess &&
                  restoredDevice == secondDevice,
              "runtime device guard did not restore caller state");
      require(cudaSetDevice(originalDevice) == cudaSuccess,
              "cross-device guard test could not restore its device");
    }

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
