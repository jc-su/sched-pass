#include "nta/JitPhase.h"

#include <dlfcn.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace nta {
namespace {

using AbiVersion = std::uint32_t (*)();
using OperatorContract = const operator_contract::Contract *(*)();
using OperatorPlan = const operator_contract::Plan *(*)();
using Reset = cudaError_t (*)(void *, std::uint32_t, std::uint32_t,
                              cudaStream_t);
using Discover = cudaError_t (*)(void *, const void *, const void *,
                                 std::uint32_t, cudaStream_t);
using InvalidateCachedObjects = cudaError_t (*)(void *, std::uint32_t,
                                                std::uint32_t, cudaStream_t);
using ValidateIndexedHostRange = cudaError_t (*)(void *, std::uint32_t,
                                                 std::uint32_t, cudaStream_t);
using RebindIndexedHostPairs = cudaError_t (*)(void *, std::uint32_t,
                                               std::uint32_t, std::uint64_t,
                                               std::uint64_t, std::uint64_t,
                                               std::uint64_t, cudaStream_t);
using PreloadHost = cudaError_t (*)(void *, std::uint32_t, std::uint32_t,
                                    cudaStream_t);
using AliasPreloadedObjects = cudaError_t (*)(void *, std::uint32_t,
                                              std::uint32_t, std::uint32_t,
                                              std::uint64_t, std::uint32_t,
                                              cudaStream_t);
using ProgressHost = cudaError_t (*)(void *, std::uint32_t, cudaStream_t);
using ProgressIndexedHostRange = cudaError_t (*)(void *, std::uint32_t,
                                                 std::uint32_t, cudaStream_t);
using ProgressIndexedHostRangeParallel = cudaError_t (*)(void *, std::uint32_t,
                                                         std::uint32_t,
                                                         std::uint32_t,
                                                         cudaStream_t);
using PrepareSelectedIndexedRows = cudaError_t (*)(
    void *, std::uint32_t, std::uint32_t, const std::int64_t *, std::uint32_t,
    std::uint32_t, std::uint32_t, const std::uint32_t *, const std::uint32_t *,
    std::uint32_t *, std::uint32_t *, std::uint32_t *, std::uint32_t,
    std::uint64_t *, cudaStream_t);
using PrepareBoundedSelectedIndexedRows = cudaError_t (*)(
    void *, std::uint32_t, std::uint32_t, const std::int64_t *, std::uint32_t,
    std::uint32_t, std::uint32_t, const std::uint32_t *, const std::uint32_t *,
    std::int64_t *, std::uint32_t, std::uint32_t *, std::uint32_t *,
    std::uint32_t *, std::uint32_t, std::uint64_t *, cudaStream_t);
using ReduceMappedIndexedKeyPages = cudaError_t (*)(
    const void *, std::uint32_t, std::uint64_t, const std::int32_t *,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    float *, float *, cudaStream_t);
using ReduceMappedKeyPages = cudaError_t (*)(const void *, std::uint32_t,
                                             std::uint64_t, std::uint32_t,
                                             std::uint32_t, std::uint32_t,
                                             std::uint32_t, std::uint32_t,
                                             std::uint32_t, float *, float *,
                                             cudaStream_t);
using ProgressNvme = cudaError_t (*)(void *, std::uint32_t, std::uint32_t,
                                     cudaStream_t);
using ProgressNvmeUntilIdle = cudaError_t (*)(void *, std::uint32_t,
                                              std::uint32_t, std::uint64_t,
                                              cudaStream_t);
using Publish = cudaError_t (*)(void *, std::uint32_t, cudaStream_t);
using Complete = cudaError_t (*)(void *, std::uint32_t, cudaStream_t);
using CompleteStreamOrdered = cudaError_t (*)(void *, const void *,
                                              std::uint32_t, cudaStream_t);

template <typename Function> Function load(void *library, const char *name) {
  dlerror();
  void *symbol = dlsym(library, name);
  const char *error = dlerror();
  if (error != nullptr || symbol == nullptr) {
    throw std::runtime_error(std::string("cannot load ") + name + ": " +
                             (error == nullptr ? "symbol is null" : error));
  }
  return reinterpret_cast<Function>(symbol);
}

void check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

} // namespace

struct JitPhaseProgram::Impl {
  explicit Impl(std::string_view path) {
    const std::string terminated(path);
    library = dlopen(terminated.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
      const char *error = dlerror();
      throw std::runtime_error(
          std::string("cannot load instrumented JIT module: ") +
          (error == nullptr ? "unknown dynamic-loader error" : error));
    }
    try {
      const AbiVersion version =
          load<AbiVersion>(library, "nta_jit_abi_version");
      if (version() != abi::Version) {
        throw std::runtime_error(
            "instrumented JIT module uses an incompatible NTA ABI");
      }
      const OperatorContract readContract =
          load<OperatorContract>(library, "nta_jit_operator_contract");
      const operator_contract::Contract *loadedContract = readContract();
      if (loadedContract == nullptr) {
        throw std::runtime_error(
            "instrumented JIT module returned a null operator contract");
      }
      contract = *loadedContract;
      operator_contract::validate(contract);
      const OperatorPlan readPlan =
          load<OperatorPlan>(library, "nta_jit_operator_plan");
      const operator_contract::Plan *loadedPlan = readPlan();
      if (loadedPlan == nullptr) {
        throw std::runtime_error(
            "instrumented JIT module returned a null operator plan");
      }
      plan = *loadedPlan;
      operator_contract::validate(plan, contract);
      reset = load<Reset>(library, "nta_jit_reset_epoch");
      discover = load<Discover>(library, "nta_jit_discover_work");
      invalidateCachedObjects = load<InvalidateCachedObjects>(
          library, "nta_jit_invalidate_cached_objects");
      validateIndexedHostRange = load<ValidateIndexedHostRange>(
          library, "nta_jit_validate_indexed_host_range");
      rebindIndexedHostPairs = load<RebindIndexedHostPairs>(
          library, "nta_jit_rebind_indexed_host_pairs");
      preloadHost = load<PreloadHost>(library, "nta_jit_preload_host");
      preloadHostPairs =
          load<PreloadHost>(library, "nta_jit_preload_host_pairs");
      aliasPreloadedObjects = load<AliasPreloadedObjects>(
          library, "nta_jit_alias_preloaded_objects");
      progressHost = load<ProgressHost>(library, "nta_jit_progress_host");
      progressIndexedHostRange = load<ProgressIndexedHostRange>(
          library, "nta_jit_progress_indexed_host_range");
      progressValidatedIndexedHostRange = load<ProgressIndexedHostRange>(
          library, "nta_jit_progress_validated_indexed_host_range");
      progressValidatedIndexedHostRangeParallel =
          load<ProgressIndexedHostRangeParallel>(
              library,
              "nta_jit_progress_validated_indexed_host_range_parallel");
      setIndexedRowCounts = load<ProgressIndexedHostRangeParallel>(
          library, "nta_jit_set_indexed_row_counts");
      prepareSelectedIndexedRows = load<PrepareSelectedIndexedRows>(
          library, "nta_jit_prepare_selected_indexed_rows");
      prepareBoundedSelectedIndexedRows =
          load<PrepareBoundedSelectedIndexedRows>(
              library, "nta_jit_prepare_bounded_selected_indexed_rows");
      reduceMappedKeyPages = load<ReduceMappedKeyPages>(
          library, "nta_jit_reduce_mapped_key_pages");
      reduceMappedIndexedKeyPages = load<ReduceMappedIndexedKeyPages>(
          library, "nta_jit_reduce_mapped_indexed_key_pages");
      progressNvme = load<ProgressNvme>(library, "nta_jit_progress_nvme");
      progressNvmeUntilIdle = load<ProgressNvmeUntilIdle>(
          library, "nta_jit_progress_nvme_until_idle");
      publish = load<Publish>(library, "nta_jit_publish_ready");
      complete = load<Complete>(library, "nta_jit_complete_launched");
      completeStreamOrdered = load<CompleteStreamOrdered>(
          library, "nta_jit_complete_stream_ordered");
    } catch (...) {
      dlclose(library);
      library = nullptr;
      throw;
    }
  }

  ~Impl() {
    if (library != nullptr) {
      dlclose(library);
    }
  }

  void *library = nullptr;
  Reset reset = nullptr;
  Discover discover = nullptr;
  InvalidateCachedObjects invalidateCachedObjects = nullptr;
  ValidateIndexedHostRange validateIndexedHostRange = nullptr;
  RebindIndexedHostPairs rebindIndexedHostPairs = nullptr;
  PreloadHost preloadHost = nullptr;
  PreloadHost preloadHostPairs = nullptr;
  AliasPreloadedObjects aliasPreloadedObjects = nullptr;
  ProgressHost progressHost = nullptr;
  ProgressIndexedHostRange progressIndexedHostRange = nullptr;
  ProgressIndexedHostRange progressValidatedIndexedHostRange = nullptr;
  ProgressIndexedHostRangeParallel progressValidatedIndexedHostRangeParallel =
      nullptr;
  ProgressIndexedHostRangeParallel setIndexedRowCounts = nullptr;
  PrepareSelectedIndexedRows prepareSelectedIndexedRows = nullptr;
  PrepareBoundedSelectedIndexedRows prepareBoundedSelectedIndexedRows = nullptr;
  ReduceMappedKeyPages reduceMappedKeyPages = nullptr;
  ReduceMappedIndexedKeyPages reduceMappedIndexedKeyPages = nullptr;
  ProgressNvme progressNvme = nullptr;
  ProgressNvmeUntilIdle progressNvmeUntilIdle = nullptr;
  Publish publish = nullptr;
  Complete complete = nullptr;
  CompleteStreamOrdered completeStreamOrdered = nullptr;
  operator_contract::Contract contract{};
  operator_contract::Plan plan{};
};

JitPhaseProgram::JitPhaseProgram(std::string_view sharedObject)
    : impl_(std::make_unique<Impl>(sharedObject)) {}

JitPhaseProgram::~JitPhaseProgram() = default;
JitPhaseProgram::JitPhaseProgram(JitPhaseProgram &&) noexcept = default;
JitPhaseProgram &
JitPhaseProgram::operator=(JitPhaseProgram &&) noexcept = default;

const operator_contract::Contract &
JitPhaseProgram::operatorContract() const noexcept {
  return impl_->contract;
}

const operator_contract::Plan &JitPhaseProgram::operatorPlan() const noexcept {
  return impl_->plan;
}

void JitPhaseProgram::reset(cudaStream_t stream, abi::RuntimeView *runtime,
                            std::uint32_t objectCount,
                            std::uint32_t workTicketCount) const {
  if (runtime == nullptr || workTicketCount == 0) {
    throw std::invalid_argument(
        "JIT phase reset needs runtime objects and work tickets");
  }
  check(impl_->reset(runtime, objectCount, workTicketCount, stream),
        "nta_jit_reset_epoch");
}

void JitPhaseProgram::discover(cudaStream_t stream, abi::RuntimeView *runtime,
                               const abi::WorkItem *workItems,
                               const abi::AcquireRequirement *dependencies,
                               std::uint32_t workItemCount) const {
  if (runtime == nullptr || workItems == nullptr || dependencies == nullptr ||
      workItemCount == 0) {
    throw std::invalid_argument(
        "JIT discovery needs runtime work items and dependencies");
  }
  check(
      impl_->discover(runtime, workItems, dependencies, workItemCount, stream),
      "nta_jit_discover_work");
}

void JitPhaseProgram::invalidateCachedObjects(cudaStream_t stream,
                                              abi::RuntimeView *runtime,
                                              std::uint32_t firstObject,
                                              std::uint32_t objectCount) const {
  if (runtime == nullptr || objectCount == 0) {
    throw std::invalid_argument(
        "JIT cache invalidation needs a runtime and non-zero object count");
  }
  check(
      impl_->invalidateCachedObjects(runtime, firstObject, objectCount, stream),
      "nta_jit_invalidate_cached_objects");
}

void JitPhaseProgram::validateIndexedHostRange(
    cudaStream_t stream, abi::RuntimeView *runtime, std::uint32_t firstObject,
    std::uint32_t objectCount) const {
  if (runtime == nullptr || objectCount == 0) {
    throw std::invalid_argument("JIT indexed host validation needs a runtime "
                                "and non-zero object count");
  }
  check(impl_->validateIndexedHostRange(runtime, firstObject, objectCount,
                                        stream),
        "nta_jit_validate_indexed_host_range");
}

void JitPhaseProgram::rebindIndexedHostPairs(
    cudaStream_t stream, abi::RuntimeView *runtime, std::uint32_t firstObject,
    std::uint32_t pairCount, std::uint64_t keySource, std::uint64_t keyStaging,
    std::uint64_t valueSource, std::uint64_t valueStaging) const {
  if (runtime == nullptr || pairCount == 0 || keySource == 0 ||
      keyStaging == 0 || valueSource == 0 || valueStaging == 0) {
    throw std::invalid_argument(
        "JIT indexed host rebinding needs object pairs and K/V addresses");
  }
  check(impl_->rebindIndexedHostPairs(runtime, firstObject, pairCount,
                                      keySource, keyStaging, valueSource,
                                      valueStaging, stream),
        "nta_jit_rebind_indexed_host_pairs");
}

void JitPhaseProgram::preloadHost(cudaStream_t stream,
                                  abi::RuntimeView *runtime,
                                  std::uint32_t firstObject,
                                  std::uint32_t objectCount) const {
  if (runtime == nullptr || objectCount == 0) {
    throw std::invalid_argument(
        "JIT host preload needs a runtime and non-zero object count");
  }
  check(impl_->preloadHost(runtime, firstObject, objectCount, stream),
        "nta_jit_preload_host");
}

void JitPhaseProgram::preloadHostPairs(cudaStream_t stream,
                                       abi::RuntimeView *runtime,
                                       std::uint32_t firstObject,
                                       std::uint32_t pairCount) const {
  if (runtime == nullptr || pairCount == 0) {
    throw std::invalid_argument(
        "JIT paired host preload needs a runtime and non-zero pair count");
  }
  check(impl_->preloadHostPairs(runtime, firstObject, pairCount, stream),
        "nta_jit_preload_host_pairs");
}

void JitPhaseProgram::aliasPreloadedObjects(
    cudaStream_t stream, abi::RuntimeView *runtime, std::uint32_t sourceFirst,
    std::uint32_t destinationFirst, std::uint32_t objectCount,
    std::uint64_t objectIdBase, std::uint32_t version) const {
  if (runtime == nullptr || objectCount == 0 || version == 0) {
    throw std::invalid_argument(
        "JIT object aliasing needs a runtime, objects, and version");
  }
  check(impl_->aliasPreloadedObjects(runtime, sourceFirst, destinationFirst,
                                     objectCount, objectIdBase, version,
                                     stream),
        "nta_jit_alias_preloaded_objects");
}

void JitPhaseProgram::progressHost(cudaStream_t stream,
                                   abi::RuntimeView *runtime,
                                   std::uint32_t blocks) const {
  if (runtime == nullptr || blocks == 0) {
    throw std::invalid_argument(
        "JIT host progress needs a runtime and non-zero blocks");
  }
  check(impl_->progressHost(runtime, blocks, stream), "nta_jit_progress_host");
}

void JitPhaseProgram::progressIndexedHostRange(
    cudaStream_t stream, abi::RuntimeView *runtime, std::uint32_t firstObject,
    std::uint32_t objectCount) const {
  if (runtime == nullptr || objectCount == 0) {
    throw std::invalid_argument(
        "JIT indexed host progress needs a runtime and non-zero object count");
  }
  check(impl_->progressIndexedHostRange(runtime, firstObject, objectCount,
                                        stream),
        "nta_jit_progress_indexed_host_range");
}

void JitPhaseProgram::progressValidatedIndexedHostRange(
    cudaStream_t stream, abi::RuntimeView *runtime, std::uint32_t firstObject,
    std::uint32_t objectCount) const {
  if (runtime == nullptr || objectCount == 0) {
    throw std::invalid_argument("JIT validated indexed progress needs a "
                                "runtime and non-zero object count");
  }
  check(impl_->progressValidatedIndexedHostRange(runtime, firstObject,
                                                 objectCount, stream),
        "nta_jit_progress_validated_indexed_host_range");
}

void JitPhaseProgram::setIndexedRowCounts(cudaStream_t stream,
                                          abi::RuntimeView *runtime,
                                          std::uint32_t firstObject,
                                          std::uint32_t objectCount,
                                          std::uint32_t rowCount) const {
  if (runtime == nullptr || objectCount == 0 || rowCount == 0) {
    throw std::invalid_argument(
        "JIT indexed row-count update needs a runtime, objects, and rows");
  }
  check(impl_->setIndexedRowCounts(runtime, firstObject, objectCount, rowCount,
                                   stream),
        "nta_jit_set_indexed_row_counts");
}

void JitPhaseProgram::prepareSelectedIndexedRows(
    cudaStream_t stream, abi::RuntimeView *runtime, std::uint32_t firstObject,
    std::uint32_t objectCount, const std::int64_t *selectedPages,
    std::uint32_t selectedPageCount, std::uint32_t pageTokens,
    std::uint32_t tokenCount, const std::uint32_t *hostRows,
    const std::uint32_t *deviceRows, std::uint32_t *stagedPages,
    std::uint32_t *sourceIndices, std::uint32_t *stagingIndices,
    std::uint32_t capacity, std::uint64_t *copiedRows) const {
  if (runtime == nullptr || selectedPages == nullptr || hostRows == nullptr ||
      deviceRows == nullptr || stagedPages == nullptr ||
      sourceIndices == nullptr || stagingIndices == nullptr ||
      copiedRows == nullptr || objectCount == 0 || selectedPageCount == 0 ||
      pageTokens == 0 || tokenCount == 0 || capacity == 0) {
    throw std::invalid_argument(
        "JIT selected-row preparation needs bounded device arrays");
  }
  check(impl_->prepareSelectedIndexedRows(
            runtime, firstObject, objectCount, selectedPages, selectedPageCount,
            pageTokens, tokenCount, hostRows, deviceRows, stagedPages,
            sourceIndices, stagingIndices, capacity, copiedRows, stream),
        "nta_jit_prepare_selected_indexed_rows");
}

void JitPhaseProgram::prepareBoundedSelectedIndexedRows(
    cudaStream_t stream, abi::RuntimeView *runtime, std::uint32_t firstObject,
    std::uint32_t objectCount, const std::int64_t *selectedPages,
    std::uint32_t selectedPageCount, std::uint32_t pageTokens,
    std::uint32_t tokenCount, const std::uint32_t *hostRows,
    const std::uint32_t *deviceRows, std::int64_t *cachedPages,
    std::uint32_t cacheSlotCount, std::uint32_t *selectedRows,
    std::uint32_t *sourceIndices, std::uint32_t *stagingIndices,
    std::uint32_t capacity, std::uint64_t *copiedRows) const {
  if (runtime == nullptr || selectedPages == nullptr || hostRows == nullptr ||
      deviceRows == nullptr || cachedPages == nullptr ||
      selectedRows == nullptr || sourceIndices == nullptr ||
      stagingIndices == nullptr || copiedRows == nullptr || objectCount == 0 ||
      selectedPageCount == 0 || pageTokens == 0 || tokenCount == 0 ||
      cacheSlotCount == 0 || capacity == 0) {
    throw std::invalid_argument(
        "JIT bounded selected-row preparation needs bounded device arrays");
  }
  check(impl_->prepareBoundedSelectedIndexedRows(
            runtime, firstObject, objectCount, selectedPages, selectedPageCount,
            pageTokens, tokenCount, hostRows, deviceRows, cachedPages,
            cacheSlotCount, selectedRows, sourceIndices, stagingIndices,
            capacity, copiedRows, stream),
        "nta_jit_prepare_bounded_selected_indexed_rows");
}

void JitPhaseProgram::reduceMappedIndexedKeyPages(
    cudaStream_t stream, const void *source, std::uint32_t sourceRows,
    std::uint64_t sourceStrideBytes, const std::int32_t *rowIndices,
    std::uint32_t tokenCount, std::uint32_t pageTokens, std::uint32_t kvHeads,
    std::uint32_t headDim, std::uint32_t elementType, float *outputMin,
    float *outputMax) const {
  if (source == nullptr || rowIndices == nullptr || outputMin == nullptr ||
      outputMax == nullptr || sourceRows == 0 || sourceStrideBytes == 0 ||
      tokenCount == 0 || pageTokens == 0 || kvHeads == 0 || headDim == 0 ||
      elementType > 1) {
    throw std::invalid_argument(
        "JIT indexed mapped key reduction needs bounded geometry");
  }
  if (impl_->reduceMappedIndexedKeyPages == nullptr) {
    throw std::runtime_error(
        "phase module lacks nta_jit_reduce_mapped_indexed_key_pages");
  }
  check(impl_->reduceMappedIndexedKeyPages(
            source, sourceRows, sourceStrideBytes, rowIndices, tokenCount,
            pageTokens, kvHeads, headDim, elementType, outputMin, outputMax,
            stream),
        "nta_jit_reduce_mapped_indexed_key_pages");
}

void JitPhaseProgram::reduceMappedKeyPages(
    cudaStream_t stream, const void *source, std::uint32_t sourceRows,
    std::uint64_t sourceStrideBytes, std::uint32_t firstRow,
    std::uint32_t tokenCount, std::uint32_t pageTokens, std::uint32_t kvHeads,
    std::uint32_t headDim, std::uint32_t elementType, float *outputMin,
    float *outputMax) const {
  if (source == nullptr || outputMin == nullptr || outputMax == nullptr ||
      sourceRows == 0 || sourceStrideBytes == 0 || tokenCount == 0 ||
      pageTokens == 0 || kvHeads == 0 || headDim == 0 || elementType > 1) {
    throw std::invalid_argument(
        "JIT mapped key reduction needs bounded source and output geometry");
  }
  check(impl_->reduceMappedKeyPages(source, sourceRows, sourceStrideBytes,
                                    firstRow, tokenCount, pageTokens, kvHeads,
                                    headDim, elementType, outputMin, outputMax,
                                    stream),
        "nta_jit_reduce_mapped_key_pages");
}

void JitPhaseProgram::progressValidatedIndexedHostRangeParallel(
    cudaStream_t stream, abi::RuntimeView *runtime, std::uint32_t firstObject,
    std::uint32_t objectCount, std::uint32_t copyBlocksPerGroup) const {
  if (runtime == nullptr || objectCount == 0 || copyBlocksPerGroup == 0) {
    throw std::invalid_argument("JIT parallel validated indexed progress needs "
                                "a runtime and non-zero geometry");
  }
  check(impl_->progressValidatedIndexedHostRangeParallel(
            runtime, firstObject, objectCount, copyBlocksPerGroup, stream),
        "nta_jit_progress_validated_indexed_host_range_parallel");
}

void JitPhaseProgram::progressNvme(cudaStream_t stream,
                                   abi::RuntimeView *runtime,
                                   std::uint32_t issueBudget,
                                   std::uint32_t completionBudget) const {
  if (runtime == nullptr || issueBudget == 0 || completionBudget == 0) {
    throw std::invalid_argument(
        "JIT NVMe progress needs a runtime and non-zero budgets");
  }
  check(impl_->progressNvme(runtime, issueBudget, completionBudget, stream),
        "nta_jit_progress_nvme");
}

void JitPhaseProgram::progressNvmeUntilIdle(
    cudaStream_t stream, abi::RuntimeView *runtime, std::uint32_t issueBudget,
    std::uint32_t completionBudget, std::uint64_t timeoutNs) const {
  if (runtime == nullptr || issueBudget == 0 || completionBudget == 0 ||
      timeoutNs == 0) {
    throw std::invalid_argument(
        "JIT NVMe progress-until-idle needs a runtime, budgets, and timeout");
  }
  check(impl_->progressNvmeUntilIdle(runtime, issueBudget, completionBudget,
                                     timeoutNs, stream),
        "nta_jit_progress_nvme_until_idle");
}

void JitPhaseProgram::publish(cudaStream_t stream, abi::RuntimeView *runtime,
                              std::uint32_t pendingBudget) const {
  if (runtime == nullptr || pendingBudget == 0) {
    throw std::invalid_argument(
        "JIT publication needs a runtime and pending budget");
  }
  check(impl_->publish(runtime, pendingBudget, stream),
        "nta_jit_publish_ready");
}

void JitPhaseProgram::complete(cudaStream_t stream, abi::RuntimeView *runtime,
                               std::uint32_t workTicketCount) const {
  if (runtime == nullptr || workTicketCount == 0) {
    throw std::invalid_argument(
        "JIT completion needs a runtime and work-ticket count");
  }
  check(impl_->complete(runtime, workTicketCount, stream),
        "nta_jit_complete_launched");
}

void JitPhaseProgram::completeStreamOrdered(cudaStream_t stream,
                                            abi::RuntimeView *runtime,
                                            const abi::WorkItem *workItems,
                                            std::uint32_t workItemCount) const {
  if (runtime == nullptr || workItems == nullptr || workItemCount == 0) {
    throw std::invalid_argument(
        "stream-ordered completion needs a runtime and exact work plan");
  }
  check(impl_->completeStreamOrdered(runtime, workItems, workItemCount, stream),
        "nta_jit_complete_stream_ordered");
}

} // namespace nta
