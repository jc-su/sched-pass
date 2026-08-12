#pragma once

#include "nta/FinitePhase.h"
#include "nta/OperatorContract.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <string_view>

namespace nta {

// Owning host adapter for phase launchers exported by an instrumented JIT
// shared object. Loading the same object as the kernel framework is safe:
// dlopen retains the existing mapping and this class holds one reference.
class JitPhaseProgram {
public:
  explicit JitPhaseProgram(std::string_view sharedObject);
  ~JitPhaseProgram();

  JitPhaseProgram(const JitPhaseProgram &) = delete;
  JitPhaseProgram &operator=(const JitPhaseProgram &) = delete;
  JitPhaseProgram(JitPhaseProgram &&) noexcept;
  JitPhaseProgram &operator=(JitPhaseProgram &&) noexcept;

  [[nodiscard]] const operator_contract::Contract &
  operatorContract() const noexcept;
  [[nodiscard]] const operator_contract::Plan &operatorPlan() const noexcept;

  void reset(cudaStream_t stream, abi::RuntimeView *runtime,
             std::uint32_t objectCount, std::uint32_t workTicketCount) const;
  void discover(cudaStream_t stream, abi::RuntimeView *runtime,
                const abi::WorkItem *workItems,
                const abi::AcquireRequirement *dependencies,
                std::uint32_t workItemCount) const;
  void invalidateCachedObjects(cudaStream_t stream, abi::RuntimeView *runtime,
                               std::uint32_t firstObject,
                               std::uint32_t objectCount) const;
  void validateIndexedHostRange(cudaStream_t stream, abi::RuntimeView *runtime,
                                std::uint32_t firstObject,
                                std::uint32_t objectCount) const;
  void rebindIndexedHostPairs(cudaStream_t stream, abi::RuntimeView *runtime,
                              std::uint32_t firstObject,
                              std::uint32_t pairCount, std::uint64_t keySource,
                              std::uint64_t keyStaging,
                              std::uint64_t valueSource,
                              std::uint64_t valueStaging) const;
  void preloadHost(cudaStream_t stream, abi::RuntimeView *runtime,
                   std::uint32_t firstObject, std::uint32_t objectCount) const;
  void preloadHostPairs(cudaStream_t stream, abi::RuntimeView *runtime,
                        std::uint32_t firstObject,
                        std::uint32_t pairCount) const;
  void aliasPreloadedObjects(cudaStream_t stream, abi::RuntimeView *runtime,
                             std::uint32_t sourceFirst,
                             std::uint32_t destinationFirst,
                             std::uint32_t objectCount,
                             std::uint64_t objectIdBase,
                             std::uint32_t version) const;
  void progressHost(cudaStream_t stream, abi::RuntimeView *runtime,
                    std::uint32_t blocks) const;
  void progressIndexedHostRange(cudaStream_t stream, abi::RuntimeView *runtime,
                                std::uint32_t firstObject,
                                std::uint32_t objectCount) const;
  void progressValidatedIndexedHostRange(cudaStream_t stream,
                                         abi::RuntimeView *runtime,
                                         std::uint32_t firstObject,
                                         std::uint32_t objectCount) const;
  void progressValidatedIndexedHostRangeParallel(
      cudaStream_t stream, abi::RuntimeView *runtime, std::uint32_t firstObject,
      std::uint32_t objectCount, std::uint32_t copyBlocksPerGroup) const;
  // Bound the next validated indexed copy of each listed object to the
  // in-place-rewritten prefix of its registered index arrays. The per-step
  // selection loop uses this to acquire only the current step's misses.
  void setIndexedRowCounts(cudaStream_t stream, abi::RuntimeView *runtime,
                           std::uint32_t firstObject, std::uint32_t objectCount,
                           std::uint32_t rowCount) const;
  void prepareSelectedIndexedRows(
      cudaStream_t stream, abi::RuntimeView *runtime, std::uint32_t firstObject,
      std::uint32_t objectCount, const std::int64_t *selectedPages,
      std::uint32_t selectedPageCount, std::uint32_t pageTokens,
      std::uint32_t tokenCount, const std::uint32_t *hostRows,
      const std::uint32_t *deviceRows, std::uint32_t *stagedPages,
      std::uint32_t *sourceIndices, std::uint32_t *stagingIndices,
      std::uint32_t capacity, std::uint64_t *copiedRows) const;
  void prepareBoundedSelectedIndexedRows(
      cudaStream_t stream, abi::RuntimeView *runtime, std::uint32_t firstObject,
      std::uint32_t objectCount, const std::int64_t *selectedPages,
      std::uint32_t selectedPageCount, std::uint32_t pageTokens,
      std::uint32_t tokenCount, const std::uint32_t *hostRows,
      const std::uint32_t *deviceRows, std::int64_t *cachedPages,
      std::uint32_t cacheSlotCount, std::uint32_t *selectedRows,
      std::uint32_t *sourceIndices, std::uint32_t *stagingIndices,
      std::uint32_t capacity, std::uint64_t *copiedRows) const;
  void prepareClaimTableSelectedRows(
      cudaStream_t stream, abi::RuntimeView *runtime,
      const std::int32_t *valid, const std::int32_t *objectSlots,
      const std::int32_t *capacityWords, const std::int32_t *selectedCounts,
      const std::int32_t *tokenCounts, const std::int64_t *selectedPagesBase,
      std::int64_t *cachedPagesBase, const std::int32_t *hostRowsBase,
      const std::int32_t *stagingRowsBase, std::uint32_t *selectedRowsBase,
      std::uint32_t *sourceIndicesBase, std::uint32_t *stagingIndicesBase,
      std::uint64_t *copiedRowsBase, std::uint32_t maxClaims,
      std::uint32_t maxBudgetPages, std::uint32_t layerCount,
      std::uint32_t localLayer, std::uint32_t maxClaimTokens,
      std::uint32_t pageTokens) const;
  void buildCompactPlan(cudaStream_t stream,
                        const std::int32_t *denseIndices,
                        const std::int32_t *denseOffsets,
                        const std::int32_t *boundLengths,
                        const std::int32_t *nonprefixOffsets,
                        const std::int32_t *nonprefixIndices,
                        const std::int32_t *claimRowCounts,
                        const std::int32_t *compactOffsets,
                        std::int32_t *compactIndices,
                        std::uint32_t batchSize) const;
  void reduceMappedIndexedKeyPages(
      cudaStream_t stream, const void *source, std::uint32_t sourceRows,
      std::uint64_t sourceStrideBytes, const std::int32_t *rowIndices,
      std::uint32_t tokenCount, std::uint32_t pageTokens,
      std::uint32_t kvHeads, std::uint32_t headDim, std::uint32_t elementType,
      float *outputMin, float *outputMax) const;
  void reduceMappedKeyPages(cudaStream_t stream, const void *source,
                            std::uint32_t sourceRows,
                            std::uint64_t sourceStrideBytes,
                            std::uint32_t firstRow, std::uint32_t tokenCount,
                            std::uint32_t pageTokens, std::uint32_t kvHeads,
                            std::uint32_t headDim, std::uint32_t elementType,
                            float *outputMin, float *outputMax) const;
  void progressNvme(cudaStream_t stream, abi::RuntimeView *runtime,
                    std::uint32_t issueBudget,
                    std::uint32_t completionBudget) const;
  void publish(cudaStream_t stream, abi::RuntimeView *runtime,
               std::uint32_t pendingBudget) const;
  void complete(cudaStream_t stream, abi::RuntimeView *runtime,
                std::uint32_t workTicketCount) const;
  void completeStreamOrdered(cudaStream_t stream, abi::RuntimeView *runtime,
                             const abi::WorkItem *workItems,
                             std::uint32_t workItemCount) const;

  template <typename Initial, typename Ready>
  void enqueueHost(cudaStream_t stream, abi::RuntimeView *runtime,
                   const HostPhaseConfig &config, Initial &&initial,
                   Ready &&ready) const {
    reset(stream, runtime, config.objectCount, config.workTicketCount);
    initial();
    complete(stream, runtime, config.workTicketCount);
    for (std::uint32_t pass = 0; pass < config.progressPasses; ++pass) {
      progressHost(stream, runtime, config.progressBlocks);
      ready();
      complete(stream, runtime, config.workTicketCount);
    }
  }

  template <typename Initial, typename Ready>
  void enqueueNvme(cudaStream_t stream, abi::RuntimeView *runtime,
                   const NvmePhaseConfig &config, Initial &&initial,
                   Ready &&ready) const {
    reset(stream, runtime, config.objectCount, config.workTicketCount);
    initial();
    complete(stream, runtime, config.workTicketCount);
    for (std::uint32_t pass = 0; pass < config.progressPasses; ++pass) {
      progressNvme(stream, runtime, config.issueBudget,
                   config.completionBudget);
      ready();
      complete(stream, runtime, config.workTicketCount);
    }
  }

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace nta
