#pragma once

#if !NTA_DEVICE_PHASE_KERNELS
#error "nta JIT runtime wrappers require NTA_DEVICE_PHASE_KERNELS=1"
#endif

#include "nta/KernelPolicy.cuh"
#include "nta/OperatorContract.h"
#include "runtime/device/Acquire.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace nta::jit {

inline cudaError_t launchStatus() { return cudaPeekAtLastError(); }

} // namespace nta::jit

#ifndef NTA_OPERATOR_FAMILY
#define NTA_OPERATOR_FAMILY 0
#endif
#ifndef NTA_OPERATOR_FORM
#define NTA_OPERATOR_FORM 0
#endif
#ifndef NTA_OPERATOR_CAPABILITIES
#define NTA_OPERATOR_CAPABILITIES 0ULL
#endif
#ifndef NTA_OPERATOR_SOURCE_HASH_LOW
#define NTA_OPERATOR_SOURCE_HASH_LOW 0ULL
#endif
#ifndef NTA_OPERATOR_SOURCE_HASH_HIGH
#define NTA_OPERATOR_SOURCE_HASH_HIGH 0ULL
#endif
#ifndef NTA_OPERATOR_SUPPORTED_FORMS
#define NTA_OPERATOR_SUPPORTED_FORMS 6U
#endif
#ifndef NTA_OPERATOR_COORDINATE_MAP
#define NTA_OPERATOR_COORDINATE_MAP 0U
#endif
#ifndef NTA_OPERATOR_PARTIAL_STATE
#define NTA_OPERATOR_PARTIAL_STATE 0U
#endif
#ifndef NTA_OPERATOR_REDUCTION
#define NTA_OPERATOR_REDUCTION 0U
#endif
#ifndef NTA_OPERATOR_PLAN_FLAGS
#define NTA_OPERATOR_PLAN_FLAGS 0U
#endif
#ifndef NTA_OPERATOR_PLAN_HASH_LOW
#define NTA_OPERATOR_PLAN_HASH_LOW NTA_OPERATOR_SOURCE_HASH_LOW
#endif
#ifndef NTA_OPERATOR_PLAN_HASH_HIGH
#define NTA_OPERATOR_PLAN_HASH_HIGH NTA_OPERATOR_SOURCE_HASH_HIGH
#endif

extern "C" __attribute__((visibility("default")))
const nta::operator_contract::Contract *nta_jit_operator_contract() {
  static constexpr nta::operator_contract::Contract contract{
      nta::operator_contract::Magic,
      nta::operator_contract::SchemaVersion,
      sizeof(nta::operator_contract::Contract),
      nta::abi::Version,
      NTA_OPERATOR_FAMILY,
      NTA_OPERATOR_FORM,
      0,
      NTA_OPERATOR_CAPABILITIES,
      NTA_OPERATOR_SOURCE_HASH_LOW,
      NTA_OPERATOR_SOURCE_HASH_HIGH,
  };
  return &contract;
}

extern "C" __attribute__((visibility("default")))
const nta::operator_contract::Plan *
nta_jit_operator_plan() {
  static constexpr nta::operator_contract::Plan plan{
      nta::operator_contract::PlanMagic,
      nta::operator_contract::PlanSchemaVersion,
      sizeof(nta::operator_contract::Plan),
      nta::abi::Version,
      NTA_OPERATOR_FAMILY,
      NTA_OPERATOR_SUPPORTED_FORMS,
      NTA_OPERATOR_COORDINATE_MAP,
      NTA_OPERATOR_PARTIAL_STATE,
      NTA_OPERATOR_REDUCTION,
      NTA_OPERATOR_PLAN_FLAGS,
      0,
      NTA_OPERATOR_SOURCE_HASH_LOW,
      NTA_OPERATOR_SOURCE_HASH_HIGH,
      NTA_OPERATOR_PLAN_HASH_LOW,
      NTA_OPERATOR_PLAN_HASH_HIGH,
  };
  return &plan;
}

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
nta_jit_validate_indexed_host_range(void *runtime, std::uint32_t firstObject,
                                    std::uint32_t objectCount,
                                    cudaStream_t stream) {
  constexpr std::uint32_t threads = 256;
  if (runtime == nullptr || objectCount == 0) {
    return cudaErrorInvalidValue;
  }
  nta_validate_indexed_host_range<<<objectCount, threads, 0, stream>>>(
      static_cast<nta::abi::RuntimeView *>(runtime), firstObject, objectCount);
  return nta::jit::launchStatus();
}

extern "C" __attribute__((visibility("default"))) cudaError_t
nta_jit_rebind_indexed_host_pairs(void *runtime, std::uint32_t firstObject,
                                  std::uint32_t pairCount,
                                  std::uint64_t keySource,
                                  std::uint64_t keyStaging,
                                  std::uint64_t valueSource,
                                  std::uint64_t valueStaging,
                                  cudaStream_t stream) {
  constexpr std::uint32_t threads = 256;
  if (runtime == nullptr || pairCount == 0 || pairCount > UINT32_MAX / 2U ||
      keySource == 0 || keyStaging == 0 || valueSource == 0 ||
      valueStaging == 0) {
    return cudaErrorInvalidValue;
  }
  const std::uint32_t objectCount = pairCount * 2U;
  nta_rebind_indexed_host_pairs<<<(objectCount + threads - 1U) / threads,
                                  threads, 0, stream>>>(
      static_cast<nta::abi::RuntimeView *>(runtime), firstObject, pairCount,
      keySource, keyStaging, valueSource, valueStaging);
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

extern "C" __global__ void
nta_discover_work(nta::abi::RuntimeView *runtime,
                  const nta::abi::WorkItem *workItems,
                  const nta::abi::AcquireRequirement *dependencies,
                  std::uint32_t workItemCount) {
  if (runtime == nullptr || workItems == nullptr || dependencies == nullptr ||
      threadIdx.x != 0 || blockIdx.x >= workItemCount) {
    return;
  }
  const nta::abi::WorkItem item = workItems[blockIdx.x];
  if (item.requestSlot >= runtime->requestCapacity ||
      item.workTicket >= runtime->workTicketCapacity ||
      item.reductionGroup >= runtime->workTicketCapacity ||
      item.contributorCount == 0 ||
      item.contributorIndex >= item.contributorCount ||
      item.directDependencyCount > item.dependencyCount) {
    nta::device::failWorkTicket(runtime, item.workTicket,
                                nta::abi::WorkTicketState::Failed);
    return;
  }
  nta::abi::WorkTicket &ticket = runtime->workTickets[item.workTicket];
  if (atomicAdd(&ticket.state, 0U) ==
      static_cast<std::uint32_t>(nta::abi::WorkTicketState::New)) {
    ticket.estimatedComputeNs = item.estimatedComputeNs;
    ticket.reductionGroup = item.reductionGroup;
    ticket.contributorCount = item.contributorCount;
  }
  const std::uint32_t generation =
      runtime->requests[item.requestSlot].generation;
  const bool available = nta_acquire_set_slow(
      runtime, item.requestSlot, generation,
      dependencies + item.dependencyBegin, item.dependencyCount,
      item.directDependencyCount, item.workTicket);
  if (nta::device::ticketMatches(runtime, ticket, item.requestSlot,
                                 generation)) {
    ticket.logicalTile = item.logicalWork;
  }
  if (available &&
      atomicAdd(&ticket.state, 0U) ==
          static_cast<std::uint32_t>(nta::abi::WorkTicketState::New)) {
    // Available work needs only canonical identity publication. Keep its
    // ticket New so the application CTA performs the normal generation-bound
    // completion transition without paying the dependency protocol used by a
    // suspended ticket. changedQueued is reset per epoch and serves as the
    // exact-once discovery publication guard.
    if (runtime->changedQueued == nullptr || runtime->readyCount == nullptr ||
        runtime->readyWorkTickets == nullptr ||
        atomicCAS(&runtime->changedQueued[item.workTicket], 0U, 2U) != 0U) {
      return;
    }
    const std::uint32_t slot = atomicAdd(runtime->readyCount, 1U);
    if (slot >= runtime->workTicketCapacity) {
      atomicExch(&runtime->changedQueued[item.workTicket], 0U);
      nta::device::failWorkTicket(runtime, item.workTicket,
                                  nta::abi::WorkTicketState::Failed);
      return;
    }
    runtime->readyWorkTickets[slot] = item.workTicket;
    __threadfence();
  }
}

extern "C" __attribute__((visibility("default"))) cudaError_t
nta_jit_discover_work(void *runtime, const void *workItems,
                      const void *dependencies, std::uint32_t workItemCount,
                      cudaStream_t stream) {
  if (runtime == nullptr || workItems == nullptr || dependencies == nullptr ||
      workItemCount == 0) {
    return cudaErrorInvalidValue;
  }
  nta_discover_work<<<workItemCount, 1, 0, stream>>>(
      static_cast<nta::abi::RuntimeView *>(runtime),
      static_cast<const nta::abi::WorkItem *>(workItems),
      static_cast<const nta::abi::AcquireRequirement *>(dependencies),
      workItemCount);
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
nta_jit_preload_host_pairs(void *runtime, std::uint32_t firstObject,
                           std::uint32_t pairCount, cudaStream_t stream) {
  constexpr std::uint32_t blocksPerPair = 2;
  if (runtime == nullptr || pairCount == 0) {
    return cudaErrorInvalidValue;
  }
  nta_preload_indexed_host_pairs<<<pairCount * blocksPerPair, 1024, 0,
                                   stream>>>(
      static_cast<nta::abi::RuntimeView *>(runtime), firstObject, pairCount);
  return nta::jit::launchStatus();
}

extern "C" __global__ void nta_alias_preloaded_objects(
    nta::abi::RuntimeView *runtime, std::uint32_t sourceFirst,
    std::uint32_t destinationFirst, std::uint32_t objectCount,
    std::uint64_t objectIdBase, std::uint32_t version) {
  const std::uint32_t relative = blockIdx.x * blockDim.x + threadIdx.x;
  if (runtime == nullptr || relative >= objectCount) {
    return;
  }
  const std::uint64_t source64 =
      static_cast<std::uint64_t>(sourceFirst) + relative;
  const std::uint64_t destination64 =
      static_cast<std::uint64_t>(destinationFirst) + relative;
  if (source64 >= runtime->objectCapacity ||
      destination64 >= runtime->objectCapacity) {
    return;
  }
  const nta::abi::ObjectEntry source = runtime->objects[source64];
  nta::abi::ObjectEntry alias = source;
  alias.objectId = objectIdBase + relative;
  alias.version = version;
  alias.issueCount = 0;
  const bool valid =
      source.state ==
          static_cast<std::uint32_t>(nta::abi::ObjectState::Ready) &&
      source.stagingAddress != 0 && source.bytes != 0 &&
      source.replicaCount != 0 &&
      source.replicaStart < runtime->replicaCapacity &&
      source.replicaCount <= runtime->replicaCapacity - source.replicaStart;
  alias.state = static_cast<std::uint32_t>(
      valid ? nta::abi::ObjectState::Ready : nta::abi::ObjectState::Failed);
  runtime->objects[destination64] = alias;
}

extern "C" __attribute__((visibility("default"))) cudaError_t
nta_jit_alias_preloaded_objects(void *runtime, std::uint32_t sourceFirst,
                                std::uint32_t destinationFirst,
                                std::uint32_t objectCount,
                                std::uint64_t objectIdBase,
                                std::uint32_t version, cudaStream_t stream) {
  constexpr std::uint32_t threads = 256;
  if (runtime == nullptr || objectCount == 0 || version == 0) {
    return cudaErrorInvalidValue;
  }
  nta_alias_preloaded_objects<<<(objectCount + threads - 1U) / threads, threads,
                                0, stream>>>(
      static_cast<nta::abi::RuntimeView *>(runtime), sourceFirst,
      destinationFirst, objectCount, objectIdBase, version);
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
nta_jit_progress_indexed_host_range(void *runtime, std::uint32_t firstObject,
                                    std::uint32_t objectCount,
                                    cudaStream_t stream) {
  constexpr std::uint32_t claimThreads = 256;
  constexpr std::uint32_t copyThreads = 1024;
  constexpr std::uint32_t objectsPerGroup = 2;
  constexpr std::uint32_t blocksPerObject = 2;
  constexpr std::uint32_t finalizeThreads = 256;
  if (runtime == nullptr || objectCount == 0) {
    return cudaErrorInvalidValue;
  }
  auto *view = static_cast<nta::abi::RuntimeView *>(runtime);
  nta_claim_indexed_host_range<<<objectCount, claimThreads, 0, stream>>>(
      view, firstObject, objectCount);
  cudaError_t status = nta::jit::launchStatus();
  if (status != cudaSuccess) {
    return status;
  }
  const std::uint32_t objectGroups =
      (objectCount + objectsPerGroup - 1U) / objectsPerGroup;
  nta_copy_indexed_host_range<<<objectGroups * blocksPerObject, copyThreads, 0,
                                stream>>>(view, firstObject, objectCount,
                                          blocksPerObject);
  status = nta::jit::launchStatus();
  if (status != cudaSuccess) {
    return status;
  }
  nta_finalize_indexed_host_range<<<(objectCount + finalizeThreads - 1U) /
                                        finalizeThreads,
                                    finalizeThreads, 0, stream>>>(
      view, firstObject, objectCount);
  return nta::jit::launchStatus();
}

// Per-step selection needs a variable acquisition count: the selector
// rewrites the registered index arrays in place with this step's misses and
// then bounds the copy to that count. ObjectEntry::flags carries the
// registered capacity; element geometry is preserved by scaling bytes with
// the count. State returns to New so a shrunken set can never appear Ready
// from a previous step, and any violation fails the object closed.
extern "C" __global__ void
nta_set_indexed_row_counts(nta::abi::RuntimeView *runtime,
                           std::uint32_t firstObject,
                           std::uint32_t objectCount, std::uint32_t rowCount) {
  using namespace nta;
  const std::uint32_t relative = blockIdx.x * blockDim.x + threadIdx.x;
  if (runtime == nullptr || relative >= objectCount) {
    return;
  }
  const std::uint64_t slot64 =
      static_cast<std::uint64_t>(firstObject) + relative;
  if (slot64 >= runtime->objectCapacity) {
    return;
  }
  abi::ObjectEntry &object = runtime->objects[slot64];
  const abi::ReplicaEntry *replica = device::replica(runtime, object, 0);
  if (replica == nullptr || (replica->flags & abi::ReplicaIndexed) == 0 ||
      rowCount == 0 || object.flags == 0 || rowCount > object.flags ||
      replica->dmaPageCount == 0) {
    atomicExch(&object.state,
               static_cast<std::uint32_t>(abi::ObjectState::Failed));
    return;
  }
  const std::uint64_t elementBytes = object.bytes / replica->dmaPageCount;
  const_cast<abi::ReplicaEntry *>(replica)->dmaPageCount = rowCount;
  object.bytes = elementBytes * rowCount;
  atomicExch(&object.state,
             static_cast<std::uint32_t>(abi::ObjectState::New));
}

extern "C" __attribute__((visibility("default"))) cudaError_t
nta_jit_set_indexed_row_counts(void *runtime, std::uint32_t firstObject,
                               std::uint32_t objectCount,
                               std::uint32_t rowCount, cudaStream_t stream) {
  constexpr std::uint32_t threads = 256;
  if (runtime == nullptr || objectCount == 0) {
    return cudaErrorInvalidValue;
  }
  nta_set_indexed_row_counts<<<(objectCount + threads - 1U) / threads, threads,
                               0, stream>>>(
      static_cast<nta::abi::RuntimeView *>(runtime), firstObject, objectCount,
      rowCount);
  return nta::jit::launchStatus();
}

extern "C" __attribute__((visibility("default"))) cudaError_t
nta_jit_progress_validated_indexed_host_range(void *runtime,
                                              std::uint32_t firstObject,
                                              std::uint32_t objectCount,
                                              cudaStream_t stream) {
  constexpr std::uint32_t copyThreads = 1024;
  constexpr std::uint32_t objectsPerGroup = 2;
  constexpr std::uint32_t blocksPerObject = 2;
  constexpr std::uint32_t finalizeThreads = 256;
  if (runtime == nullptr || objectCount == 0) {
    return cudaErrorInvalidValue;
  }
  auto *view = static_cast<nta::abi::RuntimeView *>(runtime);
  const std::uint32_t objectGroups =
      (objectCount + objectsPerGroup - 1U) / objectsPerGroup;
  nta_copy_indexed_host_range<<<objectGroups * blocksPerObject, copyThreads, 0,
                                stream>>>(view, firstObject, objectCount,
                                          blocksPerObject);
  cudaError_t status = nta::jit::launchStatus();
  if (status != cudaSuccess) {
    return status;
  }
  nta_finalize_indexed_host_range<<<(objectCount + finalizeThreads - 1U) /
                                        finalizeThreads,
                                    finalizeThreads, 0, stream>>>(
      view, firstObject, objectCount);
  return nta::jit::launchStatus();
}

extern "C" __attribute__((visibility("default"))) cudaError_t
nta_jit_progress_validated_indexed_host_range_parallel(
    void *runtime, std::uint32_t firstObject, std::uint32_t objectCount,
    std::uint32_t copyBlocksPerGroup, cudaStream_t stream) {
  constexpr std::uint32_t copyThreads = 1024;
  constexpr std::uint32_t objectsPerGroup = 2;
  constexpr std::uint32_t finalizeThreads = 256;
  constexpr std::uint32_t maxCopyBlocksPerGroup = 64;
  if (runtime == nullptr || objectCount == 0 || copyBlocksPerGroup == 0 ||
      copyBlocksPerGroup > maxCopyBlocksPerGroup) {
    return cudaErrorInvalidValue;
  }
  const std::uint32_t objectGroups =
      (objectCount + objectsPerGroup - 1U) / objectsPerGroup;
  if (objectGroups > 0xffffffffU / copyBlocksPerGroup) {
    return cudaErrorInvalidValue;
  }
  auto *view = static_cast<nta::abi::RuntimeView *>(runtime);
  nta_copy_indexed_host_range<<<objectGroups * copyBlocksPerGroup, copyThreads,
                                0, stream>>>(view, firstObject, objectCount,
                                             copyBlocksPerGroup);
  cudaError_t status = nta::jit::launchStatus();
  if (status != cudaSuccess) {
    return status;
  }
  nta_finalize_indexed_host_range<<<(objectCount + finalizeThreads - 1U) /
                                        finalizeThreads,
                                    finalizeThreads, 0, stream>>>(
      view, firstObject, objectCount);
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

extern "C" __attribute__((visibility("default"))) cudaError_t
nta_jit_complete_stream_ordered(void *runtime, const void *workItems,
                                std::uint32_t workItemCount,
                                cudaStream_t stream) {
  constexpr std::uint32_t threads = 256;
  if (runtime == nullptr || workItems == nullptr || workItemCount == 0) {
    return cudaErrorInvalidValue;
  }
  nta_complete_stream_ordered<<<(workItemCount + threads - 1U) / threads,
                                threads, 0, stream>>>(
      static_cast<nta::abi::RuntimeView *>(runtime),
      static_cast<const nta::abi::WorkItem *>(workItems), workItemCount);
  return nta::jit::launchStatus();
}
