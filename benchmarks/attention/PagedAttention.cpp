#include "benchmarks/CommonCuda.h"
#include "benchmarks/attention/PagedAttentionTypes.h"
#include "nta/DeviceWorkPlan.h"
#include "nta/FinitePhase.h"
#include "nta/FlashInferAdapter.h"
#include "nta/HostRuntime.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef NTA_ATTENTION_CUBIN_PATH
#error "NTA_ATTENTION_CUBIN_PATH is required"
#endif

namespace {

using nta::benchmark::AttentionHeadDimension;
using nta::benchmark::AttentionPageDescriptor;
using nta::benchmark::AttentionPageNeedsStaging;
using nta::benchmark::AttentionPageTokens;
using nta::benchmark::AttentionRequest;
using nta::benchmark::AttentionTilePartial;
using nta::benchmark::AttentionTileTask;
using nta::benchmark::checkCuda;
using nta::benchmark::checkDriver;
using nta::benchmark::DeviceBuffer;

enum class Mode { Resident, HostDirect, HostStaged, Mixed };
enum class CopyMode { Global, Tma };
enum class SparsePolicy { LateBound, Overfetch };

struct Options {
  Mode mode = Mode::Mixed;
  std::uint32_t requests = 32;
  std::uint32_t minPages = 4;
  std::uint32_t maxPages = 16;
  std::uint32_t iterations = 20;
  std::uint32_t progressPasses = 1;
  std::uint32_t requestCreditPages = 0;
  std::uint32_t sparseTopK = 0;
  SparsePolicy sparsePolicy = SparsePolicy::LateBound;
  CopyMode copyMode = CopyMode::Global;
  bool json = false;
  std::string dumpOutput;
};

class KernelModule {
public:
  KernelModule() {
    checkDriver(cuModuleLoad(&module_, NTA_ATTENTION_CUBIN_PATH),
                "cuModuleLoad attention cubin");
    load(tile_, "nta_attention_tile_kernel");
    load(ready_, "nta_attention_ready_kernel");
    load(tmaTile_, "nta_attention_tma_tile_kernel");
    load(tmaReady_, "nta_attention_tma_ready_kernel");
    load(sparseQuery_, "nta_sparse_query_kernel");
    load(sparse_, "nta_sparse_attention_kernel");
    load(sparseReady_, "nta_sparse_attention_ready_kernel");
    load(sparseCopyAll_, "nta_sparse_copy_all_kernel");
    load(sparseInvalidate_, "nta_sparse_invalidate_staging_kernel");
    load(reduce_, "nta_attention_reduce_kernel");
  }
  ~KernelModule() {
    if (module_ != nullptr) {
      (void)cuModuleUnload(module_);
    }
  }

  void tile(CUstream stream, CUfunction function,
            nta::abi::RuntimeView *runtime, const AttentionTileTask *tasks,
            const nta::DeviceWorkPlan &plan, const __half *queries,
            AttentionTilePartial *partials) const {
    CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
    CUdeviceptr taskAddress = reinterpret_cast<CUdeviceptr>(tasks);
    CUdeviceptr workAddress = reinterpret_cast<CUdeviceptr>(plan.workItems());
    CUdeviceptr dependencyAddress =
        reinterpret_cast<CUdeviceptr>(plan.dependencies());
    CUdeviceptr queryAddress = reinterpret_cast<CUdeviceptr>(queries);
    CUdeviceptr partialAddress = reinterpret_cast<CUdeviceptr>(partials);
    std::uint32_t taskCount = plan.workItemCount();
    void *arguments[] = {
        &runtimeAddress,    &taskAddress,  &workAddress,    &taskCount,
        &dependencyAddress, &queryAddress, &partialAddress,
    };
    launch(function, taskCount, AttentionHeadDimension, stream, arguments,
           "attention tile");
  }

  void discover(CUstream stream, nta::abi::RuntimeView *runtime,
                const AttentionTileTask *tasks, const nta::DeviceWorkPlan &plan,
                const __half *queries, AttentionTilePartial *partials) const {
    tile(stream, tile_, runtime, tasks, plan, queries, partials);
  }

  void discoverTma(CUstream stream, nta::abi::RuntimeView *runtime,
                   const AttentionTileTask *tasks,
                   const nta::DeviceWorkPlan &plan, const __half *queries,
                   AttentionTilePartial *partials) const {
    tile(stream, tmaTile_, runtime, tasks, plan, queries, partials);
  }

  void ready(CUstream stream, nta::abi::RuntimeView *runtime,
             const AttentionTileTask *tasks, const nta::DeviceWorkPlan &plan,
             const __half *queries, AttentionTilePartial *partials) const {
    tile(stream, ready_, runtime, tasks, plan, queries, partials);
  }

  void readyTma(CUstream stream, nta::abi::RuntimeView *runtime,
                const AttentionTileTask *tasks, const nta::DeviceWorkPlan &plan,
                const __half *queries, AttentionTilePartial *partials) const {
    tile(stream, tmaReady_, runtime, tasks, plan, queries, partials);
  }

  void reduce(CUstream stream, nta::abi::RuntimeView *runtime,
              const AttentionRequest *requests, std::uint32_t requestCount,
              const AttentionTilePartial *partials, float *output) const {
    CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
    CUdeviceptr requestAddress = reinterpret_cast<CUdeviceptr>(requests);
    CUdeviceptr partialAddress = reinterpret_cast<CUdeviceptr>(partials);
    CUdeviceptr outputAddress = reinterpret_cast<CUdeviceptr>(output);
    void *arguments[] = {
        &runtimeAddress, &requestAddress, &requestCount,
        &partialAddress, &outputAddress,
    };
    launch(reduce_, requestCount, AttentionHeadDimension, stream, arguments,
           "attention reduce");
  }

  void sparse(CUstream stream, CUfunction function,
              nta::abi::RuntimeView *runtime,
              const AttentionPageDescriptor *catalog,
              const std::uint32_t *candidateOffsets, const __half *summaries,
              const __half *queries, std::uint32_t requestCount,
              std::uint32_t topK, const nta::DeviceWorkPlan &plan,
              std::uint32_t *selectedCatalogIndices, float *output,
              bool preacquired) const {
    CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
    CUdeviceptr catalogAddress = reinterpret_cast<CUdeviceptr>(catalog);
    CUdeviceptr offsetAddress = reinterpret_cast<CUdeviceptr>(candidateOffsets);
    CUdeviceptr summaryAddress = reinterpret_cast<CUdeviceptr>(summaries);
    CUdeviceptr queryAddress = reinterpret_cast<CUdeviceptr>(queries);
    CUdeviceptr workAddress = reinterpret_cast<CUdeviceptr>(
        const_cast<nta::abi::WorkItem *>(plan.workItems()));
    CUdeviceptr dependencyAddress = reinterpret_cast<CUdeviceptr>(
        const_cast<nta::abi::AcquireRequirement *>(plan.dependencies()));
    CUdeviceptr selectionAddress =
        reinterpret_cast<CUdeviceptr>(selectedCatalogIndices);
    CUdeviceptr outputAddress = reinterpret_cast<CUdeviceptr>(output);
    std::uint32_t acquired = preacquired ? 1U : 0U;
    void *arguments[] = {
        &runtimeAddress,    &catalogAddress,   &offsetAddress, &summaryAddress,
        &queryAddress,      &requestCount,     &topK,          &workAddress,
        &dependencyAddress, &selectionAddress, &outputAddress, &acquired,
    };
    launch(function, requestCount, AttentionHeadDimension, stream, arguments,
           "sparse attention");
  }

  void produceSparseQueries(CUstream stream, const __half *hidden,
                            __half *queries, std::uint32_t requestCount) const {
    CUdeviceptr hiddenAddress = reinterpret_cast<CUdeviceptr>(hidden);
    CUdeviceptr queryAddress = reinterpret_cast<CUdeviceptr>(queries);
    void *arguments[] = {&hiddenAddress, &queryAddress, &requestCount};
    launch(sparseQuery_, requestCount, AttentionHeadDimension, stream,
           arguments, "sparse query producer");
  }

  void discoverSparse(CUstream stream, nta::abi::RuntimeView *runtime,
                      const AttentionPageDescriptor *catalog,
                      const std::uint32_t *candidateOffsets,
                      const __half *summaries, const __half *queries,
                      std::uint32_t requestCount, std::uint32_t topK,
                      const nta::DeviceWorkPlan &plan,
                      std::uint32_t *selectedCatalogIndices, float *output,
                      bool preacquired = false) const {
    sparse(stream, sparse_, runtime, catalog, candidateOffsets, summaries,
           queries, requestCount, topK, plan, selectedCatalogIndices, output,
           preacquired);
  }

  void readySparse(CUstream stream, nta::abi::RuntimeView *runtime,
                   const AttentionPageDescriptor *catalog,
                   const std::uint32_t *candidateOffsets,
                   const __half *summaries, const __half *queries,
                   std::uint32_t requestCount, std::uint32_t topK,
                   const nta::DeviceWorkPlan &plan,
                   std::uint32_t *selectedCatalogIndices, float *output,
                   bool preacquired = false) const {
    sparse(stream, sparseReady_, runtime, catalog, candidateOffsets, summaries,
           queries, requestCount, topK, plan, selectedCatalogIndices, output,
           preacquired);
  }

  void copySparseCatalog(CUstream stream,
                         const AttentionPageDescriptor *catalog,
                         std::uint32_t candidateCount) const {
    CUdeviceptr catalogAddress = reinterpret_cast<CUdeviceptr>(catalog);
    void *arguments[] = {&catalogAddress, &candidateCount};
    launch(sparseCopyAll_, candidateCount, 256, stream, arguments,
           "sparse overfetch copy");
  }

  void invalidateSparseStaging(CUstream stream, nta::abi::RuntimeView *runtime,
                               const AttentionPageDescriptor *catalog,
                               std::uint32_t candidateCount) const {
    CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
    CUdeviceptr catalogAddress = reinterpret_cast<CUdeviceptr>(catalog);
    void *arguments[] = {&runtimeAddress, &catalogAddress, &candidateCount};
    launch(sparseInvalidate_, (candidateCount + 255U) / 256U, 256, stream,
           arguments, "invalidate sparse staging cache");
  }

  [[nodiscard]] CUmodule module() const noexcept { return module_; }

private:
  void load(CUfunction &function, const char *name) {
    checkDriver(cuModuleGetFunction(&function, module_, name), name);
  }

  static void launch(CUfunction function, std::uint32_t blocks,
                     std::uint32_t threads, CUstream stream, void **arguments,
                     const char *name) {
    checkDriver(cuLaunchKernel(function, blocks, 1, 1, threads, 1, 1, 0, stream,
                               arguments, nullptr),
                name);
  }

  CUmodule module_ = nullptr;
  CUfunction tile_ = nullptr;
  CUfunction ready_ = nullptr;
  CUfunction tmaTile_ = nullptr;
  CUfunction tmaReady_ = nullptr;
  CUfunction sparseQuery_ = nullptr;
  CUfunction sparse_ = nullptr;
  CUfunction sparseReady_ = nullptr;
  CUfunction sparseCopyAll_ = nullptr;
  CUfunction sparseInvalidate_ = nullptr;
  CUfunction reduce_ = nullptr;
};

std::uint32_t parsePositive(std::string_view value, std::string_view option) {
  const std::string storage(value);
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(storage.c_str(), &end, 10);
  if (end == storage.c_str() || *end != '\0' || parsed == 0 ||
      parsed > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("invalid value for " + std::string(option));
  }
  return static_cast<std::uint32_t>(parsed);
}

Options parseOptions(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    const std::size_t equals = argument.find('=');
    if (equals == std::string_view::npos) {
      throw std::invalid_argument("options must use --name=value syntax");
    }
    const std::string_view name = argument.substr(0, equals);
    const std::string_view value = argument.substr(equals + 1);
    if (name == "--mode") {
      if (value == "resident") {
        options.mode = Mode::Resident;
      } else if (value == "host-direct") {
        options.mode = Mode::HostDirect;
      } else if (value == "host-staged") {
        options.mode = Mode::HostStaged;
      } else if (value == "mixed") {
        options.mode = Mode::Mixed;
      } else {
        throw std::invalid_argument("unknown attention mode");
      }
    } else if (name == "--requests") {
      options.requests = parsePositive(value, name);
    } else if (name == "--min-pages") {
      options.minPages = parsePositive(value, name);
    } else if (name == "--max-pages") {
      options.maxPages = parsePositive(value, name);
    } else if (name == "--iterations") {
      options.iterations = parsePositive(value, name);
    } else if (name == "--progress-passes") {
      options.progressPasses = parsePositive(value, name);
    } else if (name == "--request-credit-pages") {
      options.requestCreditPages = parsePositive(value, name);
    } else if (name == "--sparse-top-k") {
      options.sparseTopK = parsePositive(value, name);
    } else if (name == "--sparse-policy") {
      if (value == "late-bound") {
        options.sparsePolicy = SparsePolicy::LateBound;
      } else if (value == "overfetch") {
        options.sparsePolicy = SparsePolicy::Overfetch;
      } else {
        throw std::invalid_argument("unknown sparse attention policy");
      }
    } else if (name == "--copy") {
      if (value == "global") {
        options.copyMode = CopyMode::Global;
      } else if (value == "tma") {
        options.copyMode = CopyMode::Tma;
      } else {
        throw std::invalid_argument("unknown copy mode");
      }
    } else if (name == "--json") {
      if (value != "0" && value != "1") {
        throw std::invalid_argument("--json must be 0 or 1");
      }
      options.json = value == "1";
    } else if (name == "--dump-output") {
      if (value.empty()) {
        throw std::invalid_argument("--dump-output requires a path");
      }
      options.dumpOutput = value;
    } else {
      throw std::invalid_argument("unknown option " + std::string(name));
    }
  }
  if (options.minPages > options.maxPages) {
    throw std::invalid_argument("--min-pages must not exceed --max-pages");
  }
  if (options.sparseTopK > 8 || options.sparseTopK > options.minPages) {
    throw std::invalid_argument(
        "--sparse-top-k must be at most 8 and no larger than --min-pages");
  }
  if (options.sparseTopK != 0 && options.copyMode != CopyMode::Global) {
    throw std::invalid_argument(
        "query-dependent sparse attention currently uses global loads");
  }
  return options;
}

nta::Placement placementFor(Mode mode, std::uint32_t tile) {
  if (mode == Mode::Resident) {
    return nta::Placement::Hbm;
  }
  if (mode == Mode::HostDirect) {
    return nta::Placement::HostMapped;
  }
  if (mode == Mode::HostStaged) {
    return nta::Placement::HostStaged;
  }
  switch (tile % 3U) {
  case 0:
    return nta::Placement::Hbm;
  case 1:
    return nta::Placement::HostMapped;
  default:
    return nta::Placement::HostStaged;
  }
}

const char *modeName(Mode mode) {
  switch (mode) {
  case Mode::Resident:
    return "resident";
  case Mode::HostDirect:
    return "host-direct";
  case Mode::HostStaged:
    return "host-staged";
  case Mode::Mixed:
    return "mixed";
  }
  return "unknown";
}

const char *copyModeName(CopyMode mode) {
  return mode == CopyMode::Tma ? "tma" : "global";
}

const char *sparsePolicyName(SparsePolicy policy) {
  return policy == SparsePolicy::Overfetch ? "overfetch" : "late-bound";
}

CUtensorMap encodePageTensorMap(void *address) {
  CUtensorMap tensorMap{};
  constexpr cuuint64_t globalDimensions[2] = {AttentionHeadDimension,
                                              2U * AttentionPageTokens};
  constexpr cuuint64_t globalStrides[1] = {AttentionHeadDimension *
                                           sizeof(__half)};
  constexpr cuuint32_t boxDimensions[2] = {AttentionHeadDimension,
                                           2U * AttentionPageTokens};
  constexpr cuuint32_t elementStrides[2] = {1, 1};
  checkDriver(cuTensorMapEncodeTiled(
                  &tensorMap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, address,
                  globalDimensions, globalStrides, boxDimensions,
                  elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                  CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
              "cuTensorMapEncodeTiled attention page");
  return tensorMap;
}

std::vector<float>
referenceAttention(const std::vector<AttentionRequest> &requests,
                   const std::vector<AttentionTileTask> &tasks,
                   const std::vector<__half> &queries,
                   const std::vector<__half> &pages) {
  const std::size_t valuesPerPage =
      2ULL * AttentionPageTokens * AttentionHeadDimension;
  std::vector<float> output(requests.size() * AttentionHeadDimension);
  for (std::uint32_t requestIndex = 0; requestIndex < requests.size();
       ++requestIndex) {
    const AttentionRequest request = requests[requestIndex];
    std::vector<float> logits;
    std::vector<std::uint32_t> tokenTasks;
    std::vector<std::uint32_t> tokenOffsets;
    for (std::uint32_t tile = 0; tile < request.tileCount; ++tile) {
      const std::uint32_t taskIndex = request.tileBegin + tile;
      const AttentionTileTask task = tasks[taskIndex];
      const __half *keys =
          pages.data() +
          static_cast<std::size_t>(task.objectSlot) * valuesPerPage;
      for (std::uint32_t token = 0; token < task.tokenCount; ++token) {
        float dot = 0.0F;
        for (std::uint32_t dimension = 0; dimension < AttentionHeadDimension;
             ++dimension) {
          dot +=
              __half2float(
                  queries[requestIndex * AttentionHeadDimension + dimension]) *
              __half2float(keys[token * AttentionHeadDimension + dimension]);
        }
        logits.push_back(dot * 0.08838834764831845F);
        tokenTasks.push_back(task.objectSlot);
        tokenOffsets.push_back(token);
      }
    }
    const float maximum = *std::max_element(logits.begin(), logits.end());
    float denominator = 0.0F;
    for (float logit : logits) {
      denominator += std::exp(logit - maximum);
    }
    for (std::uint32_t dimension = 0; dimension < AttentionHeadDimension;
         ++dimension) {
      float numerator = 0.0F;
      for (std::size_t token = 0; token < logits.size(); ++token) {
        const __half *values = pages.data() +
                               tokenTasks[token] * valuesPerPage +
                               AttentionPageTokens * AttentionHeadDimension;
        numerator +=
            std::exp(logits[token] - maximum) *
            __half2float(values[tokenOffsets[token] * AttentionHeadDimension +
                                dimension]);
      }
      output[requestIndex * AttentionHeadDimension + dimension] =
          numerator / denominator;
    }
  }
  return output;
}

struct FixtureHeader {
  std::uint32_t magic;
  std::uint32_t version;
  std::uint32_t requestCount;
  std::uint32_t physicalPageCount;
  std::uint32_t headDimension;
  std::uint32_t pageTokens;
  std::uint32_t referencedPageCount;
  std::uint32_t reserved;
};
static_assert(sizeof(FixtureHeader) == 32);

template <typename T>
void writeArray(std::ofstream &output, std::span<const T> values) {
  output.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size_bytes()));
  if (!output) {
    throw std::runtime_error("failed to write FlashInfer fixture");
  }
}

void writeFixture(const std::string &path,
                  std::span<const std::int32_t> kvIndptr,
                  std::span<const std::int32_t> kvIndices,
                  std::span<const std::int32_t> lastPageLen,
                  std::span<const __half> queries,
                  std::span<const __half> pages,
                  std::span<const float> outputValues) {
  static_assert(sizeof(__half) == sizeof(std::uint16_t));
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot open FlashInfer fixture output");
  }
  const FixtureHeader header{
      0x4e544146U,
      1,
      static_cast<std::uint32_t>(lastPageLen.size()),
      static_cast<std::uint32_t>(
          pages.size() / (2ULL * AttentionPageTokens * AttentionHeadDimension)),
      AttentionHeadDimension,
      AttentionPageTokens,
      static_cast<std::uint32_t>(kvIndices.size()),
      0,
  };
  writeArray(output, std::span<const FixtureHeader>(&header, 1));
  writeArray(output, kvIndptr);
  writeArray(output, kvIndices);
  writeArray(output, lastPageLen);
  writeArray(output, queries);
  writeArray(output, pages);
  writeArray(output, outputValues);
}

int runSparseAttention(
    const Options &options, std::span<const std::int32_t> kvIndptr,
    std::span<const std::int32_t> kvIndices,
    std::span<const std::int32_t> lastPageLen, std::span<const __half> queries,
    std::span<const __half> pages,
    std::span<const nta::flashinfer::PageBinding> pageBindings,
    nta::HostRuntime &runtime) {
  const std::uint32_t requestCount = options.requests;
  const std::uint32_t topK = options.sparseTopK;
  const std::uint32_t candidateCount =
      static_cast<std::uint32_t>(kvIndices.size());
  const std::size_t valuesPerPage =
      2ULL * AttentionPageTokens * AttentionHeadDimension;
  const std::size_t pageBytes = valuesPerPage * sizeof(__half);

  std::vector<std::uint32_t> candidateOffsets(kvIndptr.size());
  std::transform(
      kvIndptr.begin(), kvIndptr.end(), candidateOffsets.begin(),
      [](std::int32_t value) { return static_cast<std::uint32_t>(value); });
  std::vector<AttentionPageDescriptor> catalog(candidateCount);
  std::vector<__half> summaries(static_cast<std::size_t>(candidateCount) *
                                AttentionHeadDimension);
  for (std::uint32_t request = 0; request < requestCount; ++request) {
    const std::uint32_t begin = candidateOffsets[request];
    const std::uint32_t end = candidateOffsets[request + 1U];
    for (std::uint32_t logical = begin; logical < end; ++logical) {
      const std::uint32_t physical =
          static_cast<std::uint32_t>(kvIndices[logical]);
      const nta::flashinfer::PageBinding &binding = pageBindings[physical];
      const nta::abi::ObjectEntry object =
          runtime.readObject(binding.objectSlot);
      const nta::abi::ReplicaEntry replica =
          runtime.readReplica(object.replicaStart);
      const bool needsStaging =
          binding.directBase == 0 && object.stagingAddress != 0;
      catalog[logical] = {
          binding.directBase,
          replica.sourceAddress,
          needsStaging ? object.stagingAddress : binding.directBase,
          binding.objectId,
          binding.objectSlot,
          binding.objectVersion,
          binding.bytes,
          logical + 1U == end ? static_cast<std::uint32_t>(lastPageLen[request])
                              : AttentionPageTokens,
          needsStaging ? AttentionPageNeedsStaging : 0U,
          0,
      };
      const __half *source =
          pages.data() + static_cast<std::size_t>(physical) * valuesPerPage;
      std::copy_n(source, AttentionHeadDimension,
                  summaries.data() + static_cast<std::size_t>(logical) *
                                         AttentionHeadDimension);
    }
  }

  nta::WorkPlanBuilder builder(topK);
  std::vector<std::uint32_t> requestSlots(requestCount);
  for (std::uint32_t request = 0; request < requestCount; ++request) {
    const std::uint32_t requestSlot = requestCount - request - 1U;
    requestSlots[request] = requestSlot;
    const std::uint32_t requestIndex =
        builder.addRequest({requestSlot, requestSlot + 1U});
    std::vector<nta::abi::AcquireRequirement> requirements;
    requirements.reserve(topK);
    for (std::uint32_t rank = 0; rank < topK; ++rank) {
      const AttentionPageDescriptor &descriptor =
          catalog[candidateOffsets[request] + rank];
      requirements.push_back({
          descriptor.directBase,
          0,
          descriptor.objectId,
          0,
          descriptor.objectSlot,
          descriptor.objectVersion,
          descriptor.bytes,
          0,
      });
    }
    (void)builder.addWork(requestIndex, request, requirements);
  }
  nta::DeviceWorkPlan devicePlan = runtime.uploadWorkPlan(builder.finish());

  DeviceBuffer<AttentionPageDescriptor> deviceCatalog(catalog.size());
  DeviceBuffer<std::uint32_t> deviceOffsets(candidateOffsets.size());
  DeviceBuffer<__half> deviceSummaries(summaries.size());
  DeviceBuffer<__half> deviceHidden(queries.size());
  DeviceBuffer<__half> deviceQueries(queries.size());
  DeviceBuffer<std::uint32_t> deviceSelections(
      static_cast<std::size_t>(requestCount) * topK);
  DeviceBuffer<float> deviceOutput(static_cast<std::size_t>(requestCount) *
                                   AttentionHeadDimension);
  checkCuda(cudaMemcpy(deviceCatalog.get(), catalog.data(),
                       catalog.size() * sizeof(catalog.front()),
                       cudaMemcpyHostToDevice),
            "upload sparse page catalog");
  checkCuda(
      cudaMemcpy(deviceOffsets.get(), candidateOffsets.data(),
                 candidateOffsets.size() * sizeof(candidateOffsets.front()),
                 cudaMemcpyHostToDevice),
      "upload sparse candidate offsets");
  checkCuda(cudaMemcpy(deviceSummaries.get(), summaries.data(),
                       summaries.size() * sizeof(summaries.front()),
                       cudaMemcpyHostToDevice),
            "upload sparse summaries");
  checkCuda(cudaMemcpy(deviceHidden.get(), queries.data(),
                       queries.size() * sizeof(queries.front()),
                       cudaMemcpyHostToDevice),
            "upload sparse hidden state");

  KernelModule kernels;
  nta::FinitePhaseProgram phases(kernels.module());
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graphExec = nullptr;
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  cudaStream_t overfetchStream = nullptr;
  cudaEvent_t overfetchFork = nullptr;
  cudaEvent_t overfetchReady = nullptr;
  checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags sparse attention");
  checkCuda(cudaEventCreate(&begin), "cudaEventCreate sparse begin");
  checkCuda(cudaEventCreate(&end), "cudaEventCreate sparse end");
  if (options.sparsePolicy == SparsePolicy::Overfetch) {
    checkCuda(
        cudaStreamCreateWithFlags(&overfetchStream, cudaStreamNonBlocking),
        "create sparse overfetch stream");
    checkCuda(cudaEventCreateWithFlags(&overfetchFork, cudaEventDisableTiming),
              "create sparse overfetch fork event");
    checkCuda(cudaEventCreateWithFlags(&overfetchReady, cudaEventDisableTiming),
              "create sparse overfetch ready event");
  }
  const CUstream driverStream = reinterpret_cast<CUstream>(stream);
  const CUstream overfetchDriverStream =
      reinterpret_cast<CUstream>(overfetchStream);
  const bool preacquired = options.sparsePolicy == SparsePolicy::Overfetch;
  const std::uint32_t progressPasses =
      !preacquired &&
              (options.mode == Mode::HostStaged || options.mode == Mode::Mixed)
          ? options.progressPasses
          : 0;

  checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
            "cudaStreamBeginCapture sparse attention");
  phases.enqueueHost(
      driverStream, runtime.deviceView(),
      {candidateCount, requestCount, candidateCount, progressPasses},
      [&] {
        checkCuda(cudaMemsetAsync(deviceSelections.get(), 0xff,
                                  static_cast<std::size_t>(requestCount) *
                                      topK * sizeof(std::uint32_t),
                                  stream),
                  "clear sparse selections");
        checkCuda(cudaMemsetAsync(deviceOutput.get(), 0,
                                  static_cast<std::size_t>(requestCount) *
                                      AttentionHeadDimension * sizeof(float),
                                  stream),
                  "clear sparse output");
        if (!preacquired) {
          kernels.invalidateSparseStaging(driverStream, runtime.deviceView(),
                                          deviceCatalog.get(), candidateCount);
        }
        if (preacquired) {
          checkCuda(cudaEventRecord(overfetchFork, stream),
                    "fork sparse overfetch graph");
          checkCuda(cudaStreamWaitEvent(overfetchStream, overfetchFork),
                    "start sparse overfetch copy");
          kernels.copySparseCatalog(overfetchDriverStream, deviceCatalog.get(),
                                    candidateCount);
          checkCuda(cudaEventRecord(overfetchReady, overfetchStream),
                    "publish sparse overfetch copy");
        }
        kernels.produceSparseQueries(driverStream, deviceHidden.get(),
                                     deviceQueries.get(), requestCount);
        if (preacquired) {
          checkCuda(cudaStreamWaitEvent(stream, overfetchReady),
                    "join sparse overfetch graph");
        }
        kernels.discoverSparse(
            driverStream, runtime.deviceView(), deviceCatalog.get(),
            deviceOffsets.get(), deviceSummaries.get(), deviceQueries.get(),
            requestCount, topK, devicePlan, deviceSelections.get(),
            deviceOutput.get(), preacquired);
      },
      [&] {
        kernels.readySparse(driverStream, runtime.deviceView(),
                            deviceCatalog.get(), deviceOffsets.get(),
                            deviceSummaries.get(), deviceQueries.get(),
                            requestCount, topK, devicePlan,
                            deviceSelections.get(), deviceOutput.get(), false);
      });
  checkCuda(cudaStreamEndCapture(stream, &graph),
            "cudaStreamEndCapture sparse attention");
  checkCuda(cudaGraphInstantiate(&graphExec, graph, 0),
            "cudaGraphInstantiate sparse attention");

  for (int warmup = 0; warmup < 3; ++warmup) {
    checkCuda(cudaGraphLaunch(graphExec, stream), "sparse attention warmup");
  }
  checkCuda(cudaStreamSynchronize(stream),
            "sparse attention warmup synchronize");
  checkCuda(cudaEventRecord(begin, stream), "cudaEventRecord sparse begin");
  for (std::uint32_t iteration = 0; iteration < options.iterations;
       ++iteration) {
    checkCuda(cudaGraphLaunch(graphExec, stream),
              "sparse attention measured launch");
  }
  checkCuda(cudaEventRecord(end, stream), "cudaEventRecord sparse end");
  checkCuda(cudaEventSynchronize(end), "cudaEventSynchronize sparse end");

  float elapsedMilliseconds = 0;
  checkCuda(cudaEventElapsedTime(&elapsedMilliseconds, begin, end),
            "cudaEventElapsedTime sparse attention");
  std::vector<std::uint32_t> selected(static_cast<std::size_t>(requestCount) *
                                      topK);
  std::vector<float> actual(static_cast<std::size_t>(requestCount) *
                            AttentionHeadDimension);
  std::vector<__half> materializedQueries(queries.size());
  checkCuda(cudaMemcpy(selected.data(), deviceSelections.get(),
                       selected.size() * sizeof(selected.front()),
                       cudaMemcpyDeviceToHost),
            "download sparse selections");
  checkCuda(cudaMemcpy(actual.data(), deviceOutput.get(),
                       actual.size() * sizeof(actual.front()),
                       cudaMemcpyDeviceToHost),
            "download sparse output");
  checkCuda(cudaMemcpy(materializedQueries.data(), deviceQueries.get(),
                       materializedQueries.size() *
                           sizeof(materializedQueries.front()),
                       cudaMemcpyDeviceToHost),
            "download materialized sparse queries");

  std::vector<AttentionRequest> selectedRequests;
  std::vector<AttentionTileTask> selectedTasks;
  selectedRequests.reserve(requestCount);
  selectedTasks.reserve(static_cast<std::size_t>(requestCount) * topK);
  for (std::uint32_t request = 0; request < requestCount; ++request) {
    selectedRequests.push_back({request * topK, topK, request, request + 1U});
    for (std::uint32_t rank = 0; rank < topK; ++rank) {
      const std::uint32_t catalogIndex = selected[request * topK + rank];
      if (catalogIndex >= catalog.size()) {
        throw std::runtime_error("sparse selector published an invalid page");
      }
      const AttentionPageDescriptor &descriptor = catalog[catalogIndex];
      selectedTasks.push_back(
          {descriptor.objectSlot, request, descriptor.tokenCount, 0});
    }
  }
  const std::vector<float> expected =
      referenceAttention(selectedRequests, selectedTasks, materializedQueries,
                         std::vector<__half>(pages.begin(), pages.end()));

  std::uint32_t failures = 0;
  float maximumError = 0.0F;
  for (std::size_t element = 0; element < expected.size(); ++element) {
    const float error = std::abs(actual[element] - expected[element]);
    maximumError = std::max(maximumError, error);
    failures += error > 2.0e-4F ? 1U : 0U;
  }
  for (std::uint32_t request = 0; request < requestCount; ++request) {
    const nta::abi::WorkTicket ticket = runtime.readWorkTicket(request);
    failures += ticket.state != static_cast<std::uint32_t>(
                                    nta::abi::WorkTicketState::Done)
                    ? 1U
                    : 0U;
    if (!preacquired) {
      failures += ticket.requestSlot != requestSlots[request] ||
                          ticket.generation != requestSlots[request] + 1U ||
                          ticket.logicalTile != request
                      ? 1U
                      : 0U;
    }
    failures += runtime.readRequest(requestSlots[request]).outstandingBytes != 0
                    ? 1U
                    : 0U;
  }
  if (!preacquired) {
    for (const AttentionTileTask &task : selectedTasks) {
      failures +=
          runtime.readObject(task.objectSlot).state !=
                  static_cast<std::uint32_t>(nta::abi::ObjectState::Ready)
              ? 1U
              : 0U;
    }
  }
  const nta::abi::IntentPool pool = runtime.readIntentPool();
  failures += pool.active != 0 || pool.overflow != 0 ? 1U : 0U;

  const double milliseconds = elapsedMilliseconds / options.iterations;
  const double usefulGib = static_cast<double>(requestCount) * topK *
                           pageBytes / (1024.0 * 1024.0 * 1024.0);
  const std::uint32_t stagedCandidates = static_cast<std::uint32_t>(
      std::count_if(catalog.begin(), catalog.end(),
                    [](const AttentionPageDescriptor &page) {
                      return (page.flags & AttentionPageNeedsStaging) != 0;
                    }));
  std::uint32_t selectedStaged = 0;
  for (std::uint32_t catalogIndex : selected) {
    selectedStaged +=
        (catalog[catalogIndex].flags & AttentionPageNeedsStaging) != 0 ? 1U
                                                                       : 0U;
  }
  const std::uint32_t movedPages =
      preacquired ? stagedCandidates : selectedStaged;
  const double overfetchRatio =
      selectedStaged == 0 ? 1.0
                          : static_cast<double>(movedPages) / selectedStaged;
  std::cout << "mode=" << modeName(options.mode) << " demand_visibility=gpu-cta"
            << " policy=" << sparsePolicyName(options.sparsePolicy)
            << " query_materialization=device-before-selector"
            << " request_binding=permuted"
            << " requests=" << requestCount
            << " candidate_pages=" << candidateCount << " top_k=" << topK
            << " selected_pages=" << requestCount * topK
            << " staged_pages_moved=" << movedPages
            << " overfetch_ratio=" << std::fixed << std::setprecision(2)
            << overfetchRatio << " graph_ms=" << std::fixed
            << std::setprecision(3) << milliseconds
            << " useful_GiB/s=" << std::setprecision(2)
            << usefulGib / (milliseconds / 1000.0)
            << " max_abs_error=" << std::scientific << maximumError
            << " intents_enqueued=" << pool.enqueued
            << " intents_consumed=" << pool.consumed
            << " progress_passes=" << options.progressPasses
            << " verification_failures=" << failures << '\n';
  if (options.json) {
    std::cout << "{\"schema\":1,\"policy\":\""
              << sparsePolicyName(options.sparsePolicy)
              << "\",\"graph_ms\":" << std::fixed << std::setprecision(6)
              << milliseconds << ",\"useful_gib_per_second\":"
              << usefulGib / (milliseconds / 1000.0)
              << ",\"candidate_pages\":" << candidateCount
              << ",\"selected_pages\":" << requestCount * topK
              << ",\"staged_pages_moved\":" << movedPages
              << ",\"overfetch_ratio\":" << overfetchRatio
              << ",\"verification_failures\":" << failures << "}\n";
  }

  (void)cudaEventDestroy(end);
  (void)cudaEventDestroy(begin);
  if (overfetchReady != nullptr) {
    (void)cudaEventDestroy(overfetchReady);
  }
  if (overfetchFork != nullptr) {
    (void)cudaEventDestroy(overfetchFork);
  }
  if (overfetchStream != nullptr) {
    (void)cudaStreamDestroy(overfetchStream);
  }
  (void)cudaGraphExecDestroy(graphExec);
  (void)cudaGraphDestroy(graph);
  (void)cudaStreamDestroy(stream);
  return failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parseOptions(argc, argv);
    checkDriver(cuInit(0), "cuInit");
    checkCuda(cudaSetDevice(0), "cudaSetDevice");

    std::vector<std::int32_t> kvIndptr(options.requests + 1U, 0);
    std::vector<std::int32_t> lastPageLen(options.requests, 0);
    std::uint32_t taskCount = 0;
    for (std::uint32_t request = 0; request < options.requests; ++request) {
      const std::uint32_t requestPages =
          options.minPages +
          request % (options.maxPages - options.minPages + 1U);
      taskCount += requestPages;
      kvIndptr[request + 1U] = static_cast<std::int32_t>(taskCount);
      lastPageLen[request] =
          static_cast<std::int32_t>(1U + (request * 7U) % AttentionPageTokens);
    }
    std::vector<std::int32_t> kvIndices(taskCount);
    for (std::uint32_t logicalPage = 0; logicalPage < taskCount;
         ++logicalPage) {
      kvIndices[logicalPage] =
          static_cast<std::int32_t>(taskCount - logicalPage - 1U);
    }

    const std::size_t valuesPerPage =
        2ULL * AttentionPageTokens * AttentionHeadDimension;
    const std::size_t pageBytes = valuesPerPage * sizeof(__half);
    std::vector<__half> queries(options.requests * AttentionHeadDimension);
    std::vector<__half> pages(static_cast<std::size_t>(taskCount) *
                              valuesPerPage);
    std::mt19937 generator(7);
    std::uniform_real_distribution<float> distribution(-0.25F, 0.25F);
    for (__half &value : queries) {
      value = __float2half(distribution(generator));
    }
    for (__half &value : pages) {
      value = __float2half(distribution(generator));
    }

    nta::HostRuntime runtime(
        {options.requests, taskCount, taskCount, taskCount});
    for (std::uint32_t request = 0; request < options.requests; ++request) {
      const std::uint64_t requestCredit =
          options.requestCreditPages == 0
              ? UINT64_MAX
              : static_cast<std::uint64_t>(options.requestCreditPages) *
                    pageBytes;
      runtime.setRequest(request, 0x4154544e00000000ULL + request, request + 1U,
                         request % 4U, request % 8U, 0, requestCredit);
    }

    std::vector<nta::flashinfer::PageBinding> pageBindings(taskCount);
    std::unique_ptr<DeviceBuffer<CUtensorMap>> deviceTensorMaps;
    std::vector<CUtensorMap> tensorMaps;
    if (options.copyMode == CopyMode::Tma) {
      deviceTensorMaps =
          std::make_unique<DeviceBuffer<CUtensorMap>>(2ULL * taskCount);
      tensorMaps.resize(2ULL * taskCount);
    }
    for (std::uint32_t physicalPage = 0; physicalPage < taskCount;
         ++physicalPage) {
      const auto *page = reinterpret_cast<const std::byte *>(
          pages.data() +
          static_cast<std::size_t>(physicalPage) * valuesPerPage);
      const std::uint64_t objectId = 0x4b56504700000000ULL + physicalPage;
      const nta::Placement placement = placementFor(options.mode, physicalPage);
      const nta::ObjectHandle object = runtime.installObject(
          physicalPage, objectId, 1,
          std::span<const std::byte>(page, pageBytes), placement);
      std::uint64_t directTensorMap = 0;
      if (options.copyMode == CopyMode::Tma) {
        const nta::abi::ObjectEntry objectEntry =
            runtime.readObject(physicalPage);
        const nta::abi::ReplicaEntry replica =
            runtime.readReplica(physicalPage);
        tensorMaps[2ULL * physicalPage] = encodePageTensorMap(
            reinterpret_cast<void *>(replica.sourceAddress));
        const void *replicaMap = deviceTensorMaps->get() + 2ULL * physicalPage;
        const void *stagingMap = nullptr;
        if (placement == nta::Placement::HostStaged) {
          tensorMaps[2ULL * physicalPage + 1ULL] = encodePageTensorMap(
              reinterpret_cast<void *>(objectEntry.stagingAddress));
          stagingMap = deviceTensorMaps->get() + 2ULL * physicalPage + 1ULL;
        } else {
          directTensorMap = reinterpret_cast<std::uint64_t>(replicaMap);
        }
        runtime.bindTensorMaps(physicalPage, 0, replicaMap, stagingMap);
      }
      pageBindings[physicalPage] = {
          reinterpret_cast<std::uint64_t>(object.directDeviceBase),
          directTensorMap,
          objectId,
          physicalPage,
          1,
          static_cast<std::uint32_t>(pageBytes),
      };
    }
    if (deviceTensorMaps != nullptr) {
      checkCuda(cudaMemcpy(deviceTensorMaps->get(), tensorMaps.data(),
                           tensorMaps.size() * sizeof(tensorMaps.front()),
                           cudaMemcpyHostToDevice),
                "upload attention tensor maps");
    }
    if (options.sparseTopK != 0) {
      if (!options.dumpOutput.empty()) {
        throw std::invalid_argument(
            "--dump-output is unavailable for sparse selected-page output");
      }
      return runSparseAttention(options, kvIndptr, kvIndices, lastPageLen,
                                queries, pages, pageBindings, runtime);
    }
    std::vector<nta::flashinfer::RequestBinding> requestBindings;
    requestBindings.reserve(options.requests);
    for (std::uint32_t request = 0; request < options.requests; ++request) {
      requestBindings.push_back({request, request + 1U});
    }
    const nta::flashinfer::DecodePlan flashInferPlan =
        nta::flashinfer::planDecode({
            AttentionPageTokens,
            kvIndptr,
            kvIndices,
            lastPageLen,
            requestBindings,
            pageBindings,
        });
    std::vector<AttentionRequest> requests;
    requests.reserve(flashInferPlan.requests.size());
    for (const nta::flashinfer::RequestChunks &request :
         flashInferPlan.requests) {
      requests.push_back({request.chunkBegin, request.chunkCount,
                          request.requestSlot, request.generation});
    }
    std::vector<AttentionTileTask> tasks;
    tasks.reserve(flashInferPlan.chunks.size());
    for (const nta::flashinfer::DecodeChunk &chunk : flashInferPlan.chunks) {
      tasks.push_back({
          chunk.physicalPage,
          chunk.requestIndex,
          chunk.tokenCount,
          0,
      });
    }
    if (flashInferPlan.work.workItems.size() != tasks.size()) {
      throw std::logic_error(
          "attention requires one common work item per physical-page tile");
    }
    for (const nta::abi::WorkItem &workItem : flashInferPlan.work.workItems) {
      if (workItem.dependencyCount != 1U) {
        throw std::logic_error(
            "attention requires exactly one dependency per physical-page tile");
      }
    }
    nta::DeviceWorkPlan devicePlan =
        runtime.uploadWorkPlan(flashInferPlan.work);
    const std::vector<float> expected =
        referenceAttention(requests, tasks, queries, pages);

    DeviceBuffer<AttentionTileTask> deviceTasks(tasks.size());
    DeviceBuffer<AttentionRequest> deviceRequests(requests.size());
    DeviceBuffer<__half> deviceQueries(queries.size());
    DeviceBuffer<AttentionTilePartial> devicePartials(tasks.size());
    DeviceBuffer<float> deviceOutput(expected.size());
    checkCuda(cudaMemcpy(deviceTasks.get(), tasks.data(),
                         tasks.size() * sizeof(tasks.front()),
                         cudaMemcpyHostToDevice),
              "upload attention tasks");
    checkCuda(cudaMemcpy(deviceRequests.get(), requests.data(),
                         requests.size() * sizeof(requests.front()),
                         cudaMemcpyHostToDevice),
              "upload attention requests");
    checkCuda(cudaMemcpy(deviceQueries.get(), queries.data(),
                         queries.size() * sizeof(queries.front()),
                         cudaMemcpyHostToDevice),
              "upload attention queries");

    KernelModule kernels;
    nta::FinitePhaseProgram phases(kernels.module());
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "cudaStreamCreateWithFlags");
    checkCuda(cudaEventCreate(&begin), "cudaEventCreate begin");
    checkCuda(cudaEventCreate(&end), "cudaEventCreate end");
    const CUstream driverStream = reinterpret_cast<CUstream>(stream);

    checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
              "cudaStreamBeginCapture");
    const std::uint32_t progressPasses =
        options.mode == Mode::HostStaged || options.mode == Mode::Mixed
            ? options.progressPasses
            : 0;
    phases.enqueueHost(
        driverStream, runtime.deviceView(),
        {taskCount, taskCount, taskCount, progressPasses},
        [&] {
          checkCuda(cudaMemsetAsync(devicePartials.get(), 0,
                                    tasks.size() * sizeof(AttentionTilePartial),
                                    stream),
                    "clear attention partials");
          checkCuda(cudaMemsetAsync(deviceOutput.get(), 0,
                                    expected.size() * sizeof(float), stream),
                    "clear attention output");
          if (options.copyMode == CopyMode::Tma) {
            kernels.discoverTma(driverStream, runtime.deviceView(),
                                deviceTasks.get(), devicePlan,
                                deviceQueries.get(), devicePartials.get());
          } else {
            kernels.discover(driverStream, runtime.deviceView(),
                             deviceTasks.get(), devicePlan, deviceQueries.get(),
                             devicePartials.get());
          }
        },
        [&] {
          if (options.copyMode == CopyMode::Tma) {
            kernels.readyTma(driverStream, runtime.deviceView(),
                             deviceTasks.get(), devicePlan, deviceQueries.get(),
                             devicePartials.get());
          } else {
            kernels.ready(driverStream, runtime.deviceView(), deviceTasks.get(),
                          devicePlan, deviceQueries.get(),
                          devicePartials.get());
          }
        });
    kernels.reduce(driverStream, runtime.deviceView(), deviceRequests.get(),
                   options.requests, devicePartials.get(), deviceOutput.get());
    checkCuda(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
    checkCuda(cudaGraphInstantiate(&graphExec, graph, 0),
              "cudaGraphInstantiate");

    for (int warmup = 0; warmup < 3; ++warmup) {
      checkCuda(cudaGraphLaunch(graphExec, stream), "attention warmup");
    }
    checkCuda(cudaStreamSynchronize(stream), "attention warmup synchronize");
    checkCuda(cudaEventRecord(begin, stream), "cudaEventRecord begin");
    for (std::uint32_t iteration = 0; iteration < options.iterations;
         ++iteration) {
      checkCuda(cudaGraphLaunch(graphExec, stream),
                "attention measured launch");
    }
    checkCuda(cudaEventRecord(end, stream), "cudaEventRecord end");
    checkCuda(cudaEventSynchronize(end), "cudaEventSynchronize end");

    float elapsedMilliseconds = 0;
    checkCuda(cudaEventElapsedTime(&elapsedMilliseconds, begin, end),
              "cudaEventElapsedTime");
    std::vector<float> actual(expected.size());
    checkCuda(cudaMemcpy(actual.data(), deviceOutput.get(),
                         actual.size() * sizeof(actual.front()),
                         cudaMemcpyDeviceToHost),
              "download attention output");
    if (!options.dumpOutput.empty()) {
      writeFixture(options.dumpOutput, kvIndptr, kvIndices, lastPageLen,
                   queries, pages, actual);
    }

    std::uint32_t outputFailures = 0;
    std::uint32_t workTicketFailures = 0;
    std::uint32_t partialFailures = 0;
    std::uint32_t objectFailures = 0;
    float maximumError = 0;
    for (std::size_t element = 0; element < expected.size(); ++element) {
      const float error = std::abs(actual[element] - expected[element]);
      maximumError = std::max(maximumError, error);
      if (error > 2.0e-4F) {
        ++outputFailures;
      }
    }
    std::vector<AttentionTilePartial> hostPartials(tasks.size());
    checkCuda(cudaMemcpy(hostPartials.data(), devicePartials.get(),
                         hostPartials.size() * sizeof(hostPartials.front()),
                         cudaMemcpyDeviceToHost),
              "download attention partials");
    for (std::uint32_t task = 0; task < taskCount; ++task) {
      if (hostPartials[task].valid == 0) {
        ++partialFailures;
      }
      if (runtime.readWorkTicket(task).state !=
          static_cast<std::uint32_t>(nta::abi::WorkTicketState::Done)) {
        ++workTicketFailures;
      }
      const nta::abi::ObjectEntry object = runtime.readObject(task);
      if (object.state !=
          static_cast<std::uint32_t>(nta::abi::ObjectState::Ready)) {
        ++objectFailures;
        if (objectFailures <= 4) {
          std::cerr << "unready_task=" << task
                    << " object_state=" << object.state
                    << " issue_count=" << object.issueCount
                    << " work_ticket_state="
                    << runtime.readWorkTicket(task).state << '\n';
        }
      }
    }
    for (std::uint32_t request = 0; request < options.requests; ++request) {
      if (runtime.readRequest(request).outstandingBytes != 0) {
        ++workTicketFailures;
      }
    }
    const nta::abi::IntentPool intentPool = runtime.readIntentPool();
    if (intentPool.active != 0 || intentPool.overflow != 0) {
      ++objectFailures;
    }
    const std::uint32_t failures =
        outputFailures + workTicketFailures + partialFailures + objectFailures;

    const double milliseconds = elapsedMilliseconds / options.iterations;
    const double logicalGib =
        static_cast<double>(taskCount) * pageBytes / (1024.0 * 1024.0 * 1024.0);
    std::cout << "mode=" << modeName(options.mode)
              << " copy=" << copyModeName(options.copyMode)
              << " requests=" << options.requests << " kv_pages=" << taskCount
              << " graph_ms=" << std::fixed << std::setprecision(3)
              << milliseconds << " logical_GiB/s=" << std::setprecision(2)
              << logicalGib / (milliseconds / 1000.0)
              << " max_abs_error=" << std::scientific << maximumError
              << " output_failures=" << outputFailures
              << " partial_failures=" << partialFailures
              << " work_ticket_failures=" << workTicketFailures
              << " object_failures=" << objectFailures
              << " intents_enqueued=" << intentPool.enqueued
              << " intents_consumed=" << intentPool.consumed
              << " intent_overflow=" << intentPool.overflow
              << " progress_passes=" << options.progressPasses
              << " request_credit_pages=" << options.requestCreditPages
              << " verification_failures=" << failures << '\n';

    (void)cudaEventDestroy(end);
    (void)cudaEventDestroy(begin);
    (void)cudaGraphExecDestroy(graphExec);
    (void)cudaGraphDestroy(graph);
    (void)cudaStreamDestroy(stream);
    return failures == 0 ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "nta-paged-attention failed: " << error.what() << '\n';
    return 1;
  }
}
