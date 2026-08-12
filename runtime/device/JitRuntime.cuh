#pragma once

#if !NTA_DEVICE_PHASE_KERNELS
#error "nta JIT runtime wrappers require NTA_DEVICE_PHASE_KERNELS=1"
#endif

#include "nta/KernelPolicy.cuh"
#include "nta/OperatorContract.h"
#include "runtime/device/Acquire.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

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
                           std::uint32_t firstObject, std::uint32_t objectCount,
                           std::uint32_t rowCount) {
  using namespace nta;
  const std::uint32_t relative = blockIdx.x;
  if (runtime == nullptr || relative >= objectCount) {
    return;
  }
  const std::uint64_t slot64 =
      static_cast<std::uint64_t>(firstObject) + relative;
  if (slot64 >= runtime->objectCapacity) {
    return;
  }
  abi::ObjectEntry &object = runtime->objects[slot64];
  abi::ReplicaEntry *replica =
      const_cast<abi::ReplicaEntry *>(device::replica(runtime, object, 0));
  __shared__ std::uint32_t validGeometry;
  __shared__ std::uint32_t invalidIndex;
  if (threadIdx.x == 0) {
    validGeometry =
        replica != nullptr && (replica->flags & abi::ReplicaIndexed) != 0 &&
                rowCount != 0 && object.flags != 0 &&
                rowCount <= object.flags && replica->dmaPageCount != 0
            ? 1U
            : 0U;
    invalidIndex = 0;
    if (validGeometry == 0) {
      atomicExch(&object.state,
                 static_cast<std::uint32_t>(abi::ObjectState::Failed));
      device::recordFailure(runtime);
    } else {
      const std::uint64_t elementBytes = object.bytes / replica->dmaPageCount;
      replica->dmaPageCount = rowCount;
      object.bytes = elementBytes * rowCount;
      object.selectedReplica = 0;
      atomicAnd(&replica->flags, ~abi::ReplicaIndicesValidated);
    }
  }
  __syncthreads();
  if (validGeometry == 0) {
    return;
  }
  device::validateIndexedTransferIndices(object, *replica, &invalidIndex);
  __syncthreads();
  if (threadIdx.x != 0) {
    return;
  }
  if (invalidIndex != 0) {
    atomicExch(&object.state,
               static_cast<std::uint32_t>(abi::ObjectState::Failed));
    device::recordFailure(runtime);
    return;
  }
  atomicOr(&replica->flags, abi::ReplicaIndicesValidated);
  // Issued is published only after the selector's current prefix has passed
  // the same source/destination bounds used by the generic indexed path.
  atomicExch(&object.state,
             static_cast<std::uint32_t>(abi::ObjectState::Issued));
}

extern "C" __attribute__((visibility("default"))) cudaError_t
nta_jit_set_indexed_row_counts(void *runtime, std::uint32_t firstObject,
                               std::uint32_t objectCount,
                               std::uint32_t rowCount, cudaStream_t stream) {
  constexpr std::uint32_t threads = 256;
  if (runtime == nullptr || objectCount == 0) {
    return cudaErrorInvalidValue;
  }
  nta_set_indexed_row_counts<<<objectCount, threads, 0, stream>>>(
      static_cast<nta::abi::RuntimeView *>(runtime), firstObject, objectCount,
      rowCount);
  return nta::jit::launchStatus();
}

// Convert device-selected logical pages into the bounded indexed transfer
// prefix without returning either the selection or the miss count to the host.
// One CTA is intentional: selection budgets are small, deterministic ordering
// makes the resulting transfer list reproducible, and K/V objects share it.
extern "C" __global__ void nta_prepare_selected_indexed_rows(
    nta::abi::RuntimeView *runtime, std::uint32_t firstObject,
    std::uint32_t objectCount, const std::int64_t *selectedPages,
    std::uint32_t selectedPageCount, std::uint32_t pageTokens,
    std::uint32_t tokenCount, const std::uint32_t *hostRows,
    const std::uint32_t *deviceRows, std::uint32_t *stagedPages,
    std::uint32_t *sourceIndices, std::uint32_t *stagingIndices,
    std::uint32_t capacity, std::uint64_t *copiedRows) {
  using namespace nta;
  __shared__ std::uint32_t rowCount;
  __shared__ std::uint32_t invalid;
  __shared__ std::uint32_t invalidIndex;
  if (threadIdx.x == 0) {
    rowCount = 0;
    invalid = 0;
    const std::uint32_t pageCount =
        pageTokens == 0 ? 0 : (tokenCount + pageTokens - 1U) / pageTokens;
    if (runtime == nullptr || selectedPages == nullptr || hostRows == nullptr ||
        deviceRows == nullptr || stagedPages == nullptr ||
        sourceIndices == nullptr || stagingIndices == nullptr ||
        copiedRows == nullptr || objectCount == 0 || selectedPageCount == 0 ||
        pageCount == 0 || capacity == 0 || selectedPageCount > pageCount ||
        firstObject > runtime->objectCapacity ||
        objectCount > runtime->objectCapacity - firstObject) {
      invalid = 1;
    }
    for (std::uint32_t index = 0; invalid == 0 && index < selectedPageCount;
         ++index) {
      const std::int64_t page = selectedPages[index];
      if (page < 0 || static_cast<std::uint64_t>(page) >= pageCount) {
        invalid = 1;
        break;
      }
      for (std::uint32_t prior = 0; prior < index; ++prior) {
        if (selectedPages[prior] == page) {
          invalid = 1;
          break;
        }
      }
    }
    for (std::uint32_t index = 0; invalid == 0 && index < selectedPageCount;
         ++index) {
      const std::uint32_t page =
          static_cast<std::uint32_t>(selectedPages[index]);
      if (stagedPages[page] != 0) {
        continue;
      }
      const std::uint32_t begin = page * pageTokens;
      const std::uint32_t end = min(tokenCount, begin + pageTokens);
      if (end < begin || end - begin > capacity - rowCount) {
        invalid = 1;
        break;
      }
      for (std::uint32_t position = begin; position < end; ++position) {
        sourceIndices[rowCount] = hostRows[position];
        stagingIndices[rowCount] = deviceRows[position];
        ++rowCount;
      }
      stagedPages[page] = 1;
    }
  }
  __syncthreads();

  for (std::uint32_t relative = 0; relative < objectCount; ++relative) {
    abi::ObjectEntry &object = runtime->objects[firstObject + relative];
    abi::ReplicaEntry *replica =
        const_cast<abi::ReplicaEntry *>(device::replica(runtime, object, 0));
    if (threadIdx.x == 0) {
      invalidIndex = 0;
      const bool geometry =
          invalid == 0 && replica != nullptr &&
          (replica->flags & abi::ReplicaIndexed) != 0 &&
          replica->dmaPageCount != 0 && object.bytes != 0 &&
          object.bytes % replica->dmaPageCount == 0 &&
          object.flags >= capacity &&
          replica->dmaPageListAddress ==
              reinterpret_cast<std::uint64_t>(sourceIndices) &&
          object.stagingTensorMapAddress ==
              reinterpret_cast<std::uint64_t>(stagingIndices);
      if (!geometry) {
        invalidIndex = 1;
      } else if (rowCount != 0) {
        const std::uint64_t elementBytes = object.bytes / replica->dmaPageCount;
        replica->dmaPageCount = rowCount;
        object.bytes = elementBytes * rowCount;
        object.selectedReplica = 0;
        atomicAnd(&replica->flags, ~abi::ReplicaIndicesValidated);
      }
    }
    __syncthreads();
    if (invalidIndex == 0 && rowCount != 0) {
      device::validateIndexedTransferIndices(object, *replica, &invalidIndex);
    }
    __syncthreads();
    if (threadIdx.x == 0 && invalidIndex != 0) {
      invalid = 1;
    }
    __syncthreads();
  }

  if (threadIdx.x != 0) {
    return;
  }
  if (invalid != 0) {
    for (std::uint32_t relative = 0; relative < objectCount; ++relative) {
      atomicExch(&runtime->objects[firstObject + relative].state,
                 static_cast<std::uint32_t>(abi::ObjectState::Failed));
    }
    device::recordFailure(runtime);
    return;
  }
  for (std::uint32_t relative = 0; relative < objectCount; ++relative) {
    abi::ObjectEntry &object = runtime->objects[firstObject + relative];
    abi::ReplicaEntry *replica =
        const_cast<abi::ReplicaEntry *>(device::replica(runtime, object, 0));
    if (rowCount != 0) {
      atomicOr(&replica->flags, abi::ReplicaIndicesValidated);
    }
    __threadfence();
    atomicExch(&object.state, static_cast<std::uint32_t>(
                                  rowCount == 0 ? abi::ObjectState::Ready
                                                : abi::ObjectState::Issued));
  }
  if (rowCount != 0) {
    atomicAdd(reinterpret_cast<unsigned long long *>(copiedRows),
              static_cast<unsigned long long>(rowCount));
  }
}

extern "C" __attribute__((visibility("default"))) cudaError_t
nta_jit_prepare_selected_indexed_rows(
    void *runtime, std::uint32_t firstObject, std::uint32_t objectCount,
    const std::int64_t *selectedPages, std::uint32_t selectedPageCount,
    std::uint32_t pageTokens, std::uint32_t tokenCount,
    const std::uint32_t *hostRows, const std::uint32_t *deviceRows,
    std::uint32_t *stagedPages, std::uint32_t *sourceIndices,
    std::uint32_t *stagingIndices, std::uint32_t capacity,
    std::uint64_t *copiedRows, cudaStream_t stream) {
  if (runtime == nullptr || selectedPages == nullptr || hostRows == nullptr ||
      deviceRows == nullptr || stagedPages == nullptr ||
      sourceIndices == nullptr || stagingIndices == nullptr ||
      copiedRows == nullptr || objectCount == 0 || selectedPageCount == 0 ||
      pageTokens == 0 || tokenCount == 0 || capacity == 0) {
    return cudaErrorInvalidValue;
  }
  nta_prepare_selected_indexed_rows<<<1, 256, 0, stream>>>(
      static_cast<nta::abi::RuntimeView *>(runtime), firstObject, objectCount,
      selectedPages, selectedPageCount, pageTokens, tokenCount, hostRows,
      deviceRows, stagedPages, sourceIndices, stagingIndices, capacity,
      copiedRows);
  return nta::jit::launchStatus();
}

// Map a device-selected logical page set into a bounded physical staging
// cache. The selected table is emitted in caller order, while only cache
// misses enter the validated indexed-copy prefix. One CTA serializes slot
// replacement so a page selected by this invocation cannot be evicted by a
// later miss from the same set.
static __device__ void ntaPrepareSelectedRowsClaim(
    nta::abi::RuntimeView *runtime, std::uint32_t firstObject,
    std::uint32_t objectCount, const std::int64_t *selectedPages,
    std::uint32_t selectedPageCount, std::uint32_t pageTokens,
    std::uint32_t tokenCount, const std::uint32_t *hostRows,
    const std::uint32_t *deviceRows, std::int64_t *cachedPages,
    std::uint32_t cacheSlotCount, std::uint32_t *selectedRows,
    std::uint32_t *sourceIndices, std::uint32_t *stagingIndices,
    std::uint32_t capacity, std::uint64_t *copiedRows) {
  using namespace nta;
  // Parallel restructure of a formerly thread-0-serial kernel whose
  // O(budget^2 x slots) global-memory scans measured 6.1ms per launch and
  // 52.6 percent of all GPU time at budget 128. Semantics are preserved
  // exactly: identical validation conditions, first-fit eviction in
  // selected order (miss i takes the i-th non-hit slot, which equals the
  // serial first-fit because slots are never released mid-pass), and
  // identical output ordering. Small tables live in shared memory; the
  // static capacity below fails closed, never silently truncates.
  constexpr std::uint32_t MaxTrackedSlots = 512;
  __shared__ std::uint32_t missRowCount;
  __shared__ std::uint32_t selectedRowCount;
  __shared__ std::uint32_t invalid;
  __shared__ std::uint32_t invalidIndex;
  __shared__ std::int64_t sharedCached[MaxTrackedSlots];
  __shared__ std::int64_t sharedSelected[MaxTrackedSlots];
  __shared__ std::uint32_t sharedSlot[MaxTrackedSlots];
  __shared__ std::uint32_t sharedIsMiss[MaxTrackedSlots];
  __shared__ std::uint32_t sharedSlotTaken[MaxTrackedSlots];
  __shared__ std::uint32_t sharedSelOffset[MaxTrackedSlots];
  __shared__ std::uint32_t sharedMissOffset[MaxTrackedSlots];
  if (threadIdx.x == 0) {
    missRowCount = 0;
    selectedRowCount = 0;
    invalid = 0;
    const std::uint32_t pageCount =
        pageTokens == 0 ? 0 : (tokenCount + pageTokens - 1U) / pageTokens;
    if (runtime == nullptr || selectedPages == nullptr || hostRows == nullptr ||
        deviceRows == nullptr || cachedPages == nullptr ||
        selectedRows == nullptr || sourceIndices == nullptr ||
        stagingIndices == nullptr || copiedRows == nullptr ||
        objectCount == 0 || selectedPageCount == 0 || pageCount == 0 ||
        capacity == 0 || capacity % pageTokens != 0 ||
        cacheSlotCount != capacity / pageTokens ||
        selectedPageCount > cacheSlotCount || selectedPageCount > pageCount ||
        cacheSlotCount > MaxTrackedSlots ||
        firstObject > runtime->objectCapacity ||
        objectCount > runtime->objectCapacity - firstObject) {
      invalid = 1;
    }
  }
  __syncthreads();
  const std::uint32_t pageCount =
      pageTokens == 0 ? 0 : (tokenCount + pageTokens - 1U) / pageTokens;
  if (invalid == 0) {
    for (std::uint32_t index = threadIdx.x; index < cacheSlotCount;
         index += blockDim.x) {
      sharedCached[index] = cachedPages[index];
      sharedSlotTaken[index] = 0;
    }
    for (std::uint32_t index = threadIdx.x; index < selectedPageCount;
         index += blockDim.x) {
      sharedSelected[index] = selectedPages[index];
      sharedSlot[index] = MaxTrackedSlots;
      sharedIsMiss[index] = 0;
    }
  }
  __syncthreads();
  if (invalid == 0) {
    // Bounds and duplicate validation over shared copies.
    for (std::uint32_t index = threadIdx.x; index < selectedPageCount;
         index += blockDim.x) {
      const std::int64_t page = sharedSelected[index];
      if (page < 0 || static_cast<std::uint64_t>(page) >= pageCount) {
        atomicOr(&invalid, 1U);
        continue;
      }
      for (std::uint32_t prior = 0; prior < index; ++prior) {
        if (sharedSelected[prior] == page) {
          atomicOr(&invalid, 1U);
          break;
        }
      }
    }
  }
  __syncthreads();
  if (invalid == 0) {
    // Hit detection: distinct selected pages cannot hit the same slot, so
    // the slot-taken writes are race-free by construction.
    for (std::uint32_t index = threadIdx.x; index < selectedPageCount;
         index += blockDim.x) {
      const std::int64_t page = sharedSelected[index];
      for (std::uint32_t candidate = 0; candidate < cacheSlotCount;
           ++candidate) {
        if (sharedCached[candidate] == page) {
          sharedSlot[index] = candidate;
          sharedSlotTaken[candidate] = 1;
          break;
        }
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && invalid == 0) {
    // Sequential first-fit assignment and offset prefix over shared
    // tables: bounded by the slot cap, a few hundred shared-memory
    // iterations, microseconds not milliseconds.
    std::uint32_t freeCursor = 0;
    std::uint32_t runningSelected = 0;
    std::uint32_t runningMiss = 0;
    for (std::uint32_t index = 0; invalid == 0 && index < selectedPageCount;
         ++index) {
      const bool hit = sharedSlot[index] != MaxTrackedSlots;
      if (!hit) {
        while (freeCursor < cacheSlotCount && sharedSlotTaken[freeCursor]) {
          ++freeCursor;
        }
        if (freeCursor == cacheSlotCount) {
          invalid = 1;
          break;
        }
        sharedSlot[index] = freeCursor;
        sharedSlotTaken[freeCursor] = 1;
        sharedIsMiss[index] = 1;
      }
      const std::uint32_t page =
          static_cast<std::uint32_t>(sharedSelected[index]);
      const std::uint32_t begin = page * pageTokens;
      const std::uint32_t end = min(tokenCount, begin + pageTokens);
      const std::uint32_t rows = end - begin;
      const std::uint32_t physicalBegin = sharedSlot[index] * pageTokens;
      if (end < begin || rows > capacity - runningSelected ||
          rows > capacity - physicalBegin ||
          (!hit && rows > capacity - runningMiss)) {
        invalid = 1;
        break;
      }
      sharedSelOffset[index] = runningSelected;
      sharedMissOffset[index] = runningMiss;
      runningSelected += rows;
      if (!hit) {
        runningMiss += rows;
      }
    }
    if (invalid == 0) {
      selectedRowCount = runningSelected;
      missRowCount = runningMiss;
    }
  }
  __syncthreads();
  if (invalid == 0) {
    // Parallel emission in the exact ordering the serial code produced.
    for (std::uint32_t index = threadIdx.x; index < selectedPageCount;
         index += blockDim.x) {
      const std::uint32_t page =
          static_cast<std::uint32_t>(sharedSelected[index]);
      const std::uint32_t begin = page * pageTokens;
      const std::uint32_t rows = min(tokenCount, begin + pageTokens) - begin;
      const std::uint32_t physicalBegin = sharedSlot[index] * pageTokens;
      const bool miss = sharedIsMiss[index] != 0;
      for (std::uint32_t row = 0; row < rows; ++row) {
        const std::uint32_t physical = deviceRows[physicalBegin + row];
        selectedRows[sharedSelOffset[index] + row] = physical;
        if (miss) {
          sourceIndices[sharedMissOffset[index] + row] = hostRows[begin + row];
          stagingIndices[sharedMissOffset[index] + row] = physical;
        }
      }
      if (miss) {
        cachedPages[sharedSlot[index]] = sharedSelected[index];
      }
    }
  }
  __syncthreads();

  for (std::uint32_t relative = 0; relative < objectCount; ++relative) {
    abi::ObjectEntry &object = runtime->objects[firstObject + relative];
    abi::ReplicaEntry *replica =
        const_cast<abi::ReplicaEntry *>(device::replica(runtime, object, 0));
    if (threadIdx.x == 0) {
      invalidIndex = 0;
      const bool geometry =
          invalid == 0 && replica != nullptr &&
          (replica->flags & abi::ReplicaIndexed) != 0 &&
          replica->dmaPageCount != 0 && object.bytes != 0 &&
          object.bytes % replica->dmaPageCount == 0 &&
          object.flags >= capacity &&
          replica->dmaPageListAddress ==
              reinterpret_cast<std::uint64_t>(sourceIndices) &&
          object.stagingTensorMapAddress ==
              reinterpret_cast<std::uint64_t>(stagingIndices);
      if (!geometry) {
        invalidIndex = 1;
      } else if (missRowCount != 0) {
        const std::uint64_t elementBytes = object.bytes / replica->dmaPageCount;
        replica->dmaPageCount = missRowCount;
        object.bytes = elementBytes * missRowCount;
        object.selectedReplica = 0;
        atomicAnd(&replica->flags, ~abi::ReplicaIndicesValidated);
      }
    }
    __syncthreads();
    if (invalidIndex == 0 && missRowCount != 0) {
      device::validateIndexedTransferIndices(object, *replica, &invalidIndex);
    }
    __syncthreads();
    if (threadIdx.x == 0 && invalidIndex != 0) {
      invalid = 1;
    }
    __syncthreads();
  }

  if (threadIdx.x != 0) {
    return;
  }
  if (invalid != 0) {
    for (std::uint32_t relative = 0; relative < objectCount; ++relative) {
      atomicExch(&runtime->objects[firstObject + relative].state,
                 static_cast<std::uint32_t>(abi::ObjectState::Failed));
    }
    device::recordFailure(runtime);
    return;
  }
  for (std::uint32_t relative = 0; relative < objectCount; ++relative) {
    abi::ObjectEntry &object = runtime->objects[firstObject + relative];
    abi::ReplicaEntry *replica =
        const_cast<abi::ReplicaEntry *>(device::replica(runtime, object, 0));
    if (missRowCount != 0) {
      atomicOr(&replica->flags, abi::ReplicaIndicesValidated);
    }
    __threadfence();
    atomicExch(&object.state,
               static_cast<std::uint32_t>(missRowCount == 0
                                              ? abi::ObjectState::Ready
                                              : abi::ObjectState::Issued));
  }
  if (missRowCount != 0) {
    atomicAdd(reinterpret_cast<unsigned long long *>(copiedRows),
              static_cast<unsigned long long>(missRowCount));
  }
}

extern "C" __global__ void nta_prepare_bounded_selected_indexed_rows(
    nta::abi::RuntimeView *runtime, std::uint32_t firstObject,
    std::uint32_t objectCount, const std::int64_t *selectedPages,
    std::uint32_t selectedPageCount, std::uint32_t pageTokens,
    std::uint32_t tokenCount, const std::uint32_t *hostRows,
    const std::uint32_t *deviceRows, std::int64_t *cachedPages,
    std::uint32_t cacheSlotCount, std::uint32_t *selectedRows,
    std::uint32_t *sourceIndices, std::uint32_t *stagingIndices,
    std::uint32_t capacity, std::uint64_t *copiedRows) {
  ntaPrepareSelectedRowsClaim(
      runtime, firstObject, objectCount, selectedPages, selectedPageCount,
      pageTokens, tokenCount, hostRows, deviceRows, cachedPages,
      cacheSlotCount, selectedRows, sourceIndices, stagingIndices, capacity,
      copiedRows);
}

extern "C" __global__ void nta_prepare_claim_table_selected_rows(
    nta::abi::RuntimeView *runtime, const std::int32_t *valid,
    const std::int32_t *objectSlots, const std::int32_t *capacityWords,
    const std::int32_t *selectedCounts, const std::int32_t *tokenCounts,
    const std::int64_t *selectedPagesBase, std::int64_t *cachedPagesBase,
    const std::int32_t *hostRowsBase, const std::int32_t *stagingRowsBase,
    std::uint32_t *selectedRowsBase, std::uint32_t *sourceIndicesBase,
    std::uint32_t *stagingIndicesBase, std::uint64_t *copiedRowsBase,
    std::uint32_t maxClaims, std::uint32_t maxBudgetPages,
    std::uint32_t layerCount, std::uint32_t localLayer,
    std::uint32_t maxClaimTokens, std::uint32_t pageTokens) {
  // One block per claim-table row; the launch shape is fixed at the
  // table's capacity so a captured graph replays over any live set, and
  // invalid rows retire in a few instructions via the validity word.
  const std::uint32_t claim = blockIdx.x;
  if (claim >= maxClaims || valid == nullptr || valid[claim] == 0 ||
      localLayer >= layerCount) {
    return;
  }
  const std::uint32_t count =
      static_cast<std::uint32_t>(selectedCounts[claim]);
  if (count == 0) {
    return;
  }
  const std::uint32_t capacity =
      static_cast<std::uint32_t>(capacityWords[claim]);
  const std::uint64_t rowStride =
      static_cast<std::uint64_t>(maxBudgetPages) * pageTokens;
  ntaPrepareSelectedRowsClaim(
      runtime, static_cast<std::uint32_t>(objectSlots[claim]), 2U,
      selectedPagesBase + static_cast<std::uint64_t>(claim) * maxBudgetPages,
      count, pageTokens, static_cast<std::uint32_t>(tokenCounts[claim]),
      reinterpret_cast<const std::uint32_t *>(hostRowsBase) +
          static_cast<std::uint64_t>(claim) * maxClaimTokens,
      reinterpret_cast<const std::uint32_t *>(stagingRowsBase) +
          static_cast<std::uint64_t>(claim) * rowStride,
      cachedPagesBase +
          (static_cast<std::uint64_t>(claim) * layerCount + localLayer) *
              maxBudgetPages,
      capacity / pageTokens,
      selectedRowsBase + static_cast<std::uint64_t>(claim) * rowStride,
      sourceIndicesBase + static_cast<std::uint64_t>(claim) * rowStride,
      stagingIndicesBase + static_cast<std::uint64_t>(claim) * rowStride,
      capacity, copiedRowsBase + claim);
}

extern "C" __attribute__((visibility("default"))) cudaError_t
nta_jit_prepare_claim_table_selected_rows(
    void *runtime, const std::int32_t *valid, const std::int32_t *objectSlots,
    const std::int32_t *capacityWords, const std::int32_t *selectedCounts,
    const std::int32_t *tokenCounts, const std::int64_t *selectedPagesBase,
    std::int64_t *cachedPagesBase, const std::int32_t *hostRowsBase,
    const std::int32_t *stagingRowsBase, std::uint32_t *selectedRowsBase,
    std::uint32_t *sourceIndicesBase, std::uint32_t *stagingIndicesBase,
    std::uint64_t *copiedRowsBase, std::uint32_t maxClaims,
    std::uint32_t maxBudgetPages, std::uint32_t layerCount,
    std::uint32_t localLayer, std::uint32_t maxClaimTokens,
    std::uint32_t pageTokens, cudaStream_t stream) {
  if (runtime == nullptr || valid == nullptr || objectSlots == nullptr ||
      capacityWords == nullptr || selectedCounts == nullptr ||
      tokenCounts == nullptr || selectedPagesBase == nullptr ||
      cachedPagesBase == nullptr || hostRowsBase == nullptr ||
      stagingRowsBase == nullptr || selectedRowsBase == nullptr ||
      sourceIndicesBase == nullptr || stagingIndicesBase == nullptr ||
      copiedRowsBase == nullptr || maxClaims == 0 || maxBudgetPages == 0 ||
      layerCount == 0 || localLayer >= layerCount || maxClaimTokens == 0 ||
      pageTokens == 0) {
    return cudaErrorInvalidValue;
  }
  nta_prepare_claim_table_selected_rows<<<maxClaims, 256, 0, stream>>>(
      static_cast<nta::abi::RuntimeView *>(runtime), valid, objectSlots,
      capacityWords, selectedCounts, tokenCounts, selectedPagesBase,
      cachedPagesBase, hostRowsBase, stagingRowsBase, selectedRowsBase,
      sourceIndicesBase, stagingIndicesBase, copiedRowsBase, maxClaims,
      maxBudgetPages, layerCount, localLayer, maxClaimTokens, pageTokens);
  return nta::jit::launchStatus();
}

extern "C" __attribute__((visibility("default"))) cudaError_t
nta_jit_prepare_bounded_selected_indexed_rows(
    void *runtime, std::uint32_t firstObject, std::uint32_t objectCount,
    const std::int64_t *selectedPages, std::uint32_t selectedPageCount,
    std::uint32_t pageTokens, std::uint32_t tokenCount,
    const std::uint32_t *hostRows, const std::uint32_t *deviceRows,
    std::int64_t *cachedPages, std::uint32_t cacheSlotCount,
    std::uint32_t *selectedRows, std::uint32_t *sourceIndices,
    std::uint32_t *stagingIndices, std::uint32_t capacity,
    std::uint64_t *copiedRows, cudaStream_t stream) {
  if (runtime == nullptr || selectedPages == nullptr || hostRows == nullptr ||
      deviceRows == nullptr || cachedPages == nullptr ||
      selectedRows == nullptr || sourceIndices == nullptr ||
      stagingIndices == nullptr || copiedRows == nullptr || objectCount == 0 ||
      selectedPageCount == 0 || pageTokens == 0 || tokenCount == 0 ||
      cacheSlotCount == 0 || capacity == 0) {
    return cudaErrorInvalidValue;
  }
  nta_prepare_bounded_selected_indexed_rows<<<1, 256, 0, stream>>>(
      static_cast<nta::abi::RuntimeView *>(runtime), firstObject, objectCount,
      selectedPages, selectedPageCount, pageTokens, tokenCount, hostRows,
      deviceRows, cachedPages, cacheSlotCount, selectedRows, sourceIndices,
      stagingIndices, capacity, copiedRows);
  return nta::jit::launchStatus();
}

extern "C" __global__ void nta_reduce_mapped_key_pages(
    const std::byte *source, std::uint32_t sourceRows,
    std::uint64_t sourceStrideBytes, std::uint32_t firstRow,
    std::uint32_t tokenCount, std::uint32_t pageTokens,
    std::uint32_t elementCount, std::uint32_t elementType, float *outputMin,
    float *outputMax) {
  const std::uint32_t pageCount = (tokenCount + pageTokens - 1U) / pageTokens;
  const std::uint64_t outputCount =
      static_cast<std::uint64_t>(pageCount) * elementCount;
  for (std::uint64_t output =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       output < outputCount;
       output += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const std::uint32_t page = output / elementCount;
    const std::uint32_t element = output % elementCount;
    const std::uint32_t begin = page * pageTokens;
    const std::uint32_t end = min(tokenCount, begin + pageTokens);
    float minimum = CUDART_INF_F;
    float maximum = -CUDART_INF_F;
    for (std::uint32_t token = begin; token < end; ++token) {
      const std::uint64_t row = static_cast<std::uint64_t>(firstRow) + token;
      if (row >= sourceRows) {
        continue;
      }
      const std::byte *address = source + row * sourceStrideBytes +
                                 static_cast<std::uint64_t>(element) * 2U;
      const float value =
          elementType == 0
              ? __half2float(*reinterpret_cast<const __half *>(address))
              : __bfloat162float(
                    *reinterpret_cast<const __nv_bfloat16 *>(address));
      minimum = fminf(minimum, value);
      maximum = fmaxf(maximum, value);
    }
    outputMin[output] = minimum;
    outputMax[output] = maximum;
  }
}

extern "C" __attribute__((visibility("default"))) cudaError_t
nta_jit_reduce_mapped_key_pages(const void *source, std::uint32_t sourceRows,
                                std::uint64_t sourceStrideBytes,
                                std::uint32_t firstRow,
                                std::uint32_t tokenCount,
                                std::uint32_t pageTokens, std::uint32_t kvHeads,
                                std::uint32_t headDim,
                                std::uint32_t elementType, float *outputMin,
                                float *outputMax, cudaStream_t stream) {
  constexpr std::uint32_t threads = 256;
  if (source == nullptr || outputMin == nullptr || outputMax == nullptr ||
      sourceRows == 0 || tokenCount == 0 || pageTokens == 0 || kvHeads == 0 ||
      headDim == 0 || elementType > 1 || firstRow >= sourceRows ||
      tokenCount > sourceRows - firstRow ||
      headDim > std::numeric_limits<std::uint32_t>::max() / kvHeads ||
      sourceStrideBytes < static_cast<std::uint64_t>(kvHeads) * headDim * 2U) {
    return cudaErrorInvalidValue;
  }
  const std::uint64_t pageCount =
      (static_cast<std::uint64_t>(tokenCount) + pageTokens - 1U) / pageTokens;
  const std::uint64_t outputCount = pageCount * kvHeads * headDim;
  const std::uint64_t blocks64 = (outputCount + threads - 1U) / threads;
  const std::uint32_t blocks =
      static_cast<std::uint32_t>(std::min<std::uint64_t>(blocks64, 65535U));
  nta_reduce_mapped_key_pages<<<blocks, threads, 0, stream>>>(
      static_cast<const std::byte *>(source), sourceRows, sourceStrideBytes,
      firstRow, tokenCount, pageTokens, kvHeads * headDim, elementType,
      outputMin, outputMax);
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
