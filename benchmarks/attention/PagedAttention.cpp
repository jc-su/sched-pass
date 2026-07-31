#include "benchmarks/attention/PagedAttentionTypes.h"
#include "nta/HostRuntime.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
using nta::benchmark::AttentionPageTokens;
using nta::benchmark::AttentionRequest;
using nta::benchmark::AttentionTilePartial;
using nta::benchmark::AttentionTileTask;

enum class Mode { Resident, HostDirect, HostStaged, Mixed };
enum class CopyMode { Global, Tma };

struct Options {
  Mode mode = Mode::Mixed;
  std::uint32_t requests = 32;
  std::uint32_t minPages = 4;
  std::uint32_t maxPages = 16;
  std::uint32_t iterations = 20;
  std::uint32_t progressPasses = 1;
  std::uint32_t requestCreditPages = 0;
  CopyMode copyMode = CopyMode::Global;
};

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
  const char *name = nullptr;
  const char *description = nullptr;
  (void)cuGetErrorName(result, &name);
  (void)cuGetErrorString(result, &description);
  throw std::runtime_error(
      std::string(operation) + ": " +
      (name == nullptr ? "unknown CUDA driver error" : name) + " (" +
      (description == nullptr ? "no description" : description) + ")");
}

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t count) {
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&pointer_), count * sizeof(T)),
              "cudaMalloc DeviceBuffer");
  }
  ~DeviceBuffer() {
    if (pointer_ != nullptr) {
      (void)cudaFree(pointer_);
    }
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  [[nodiscard]] T *get() const noexcept { return pointer_; }

private:
  T *pointer_ = nullptr;
};

class KernelModule {
public:
  KernelModule() {
    checkDriver(cuModuleLoad(&module_, NTA_ATTENTION_CUBIN_PATH),
                "cuModuleLoad attention cubin");
    load(reset_, "nta_reset_epoch");
    load(progress_, "nta_progress_host_staging");
    load(publish_, "nta_publish_ready");
    load(tile_, "nta_attention_tile_kernel");
    load(ready_, "nta_attention_ready_kernel");
    load(tmaTile_, "nta_attention_tma_tile_kernel");
    load(tmaReady_, "nta_attention_tma_ready_kernel");
    load(reduce_, "nta_attention_reduce_kernel");
  }
  ~KernelModule() {
    if (module_ != nullptr) {
      (void)cuModuleUnload(module_);
    }
  }

  void reset(CUstream stream, nta::abi::RuntimeView *runtime,
             std::uint32_t objectCount,
             std::uint32_t continuationCount) const {
    const std::uint32_t threads = 256;
    const std::uint32_t blocks =
        (std::max(objectCount, continuationCount) + threads - 1U) / threads;
    CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
    void *arguments[] = {
        &runtimeAddress,
        &objectCount,
        &continuationCount,
    };
    launch(reset_, blocks, threads, stream, arguments, "reset");
  }

  void tile(CUstream stream, CUfunction function,
            nta::abi::RuntimeView *runtime, const AttentionTileTask *tasks,
            std::uint32_t taskCount, const __half *queries,
            AttentionTilePartial *partials) const {
    CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
    CUdeviceptr taskAddress = reinterpret_cast<CUdeviceptr>(tasks);
    CUdeviceptr queryAddress = reinterpret_cast<CUdeviceptr>(queries);
    CUdeviceptr partialAddress = reinterpret_cast<CUdeviceptr>(partials);
    void *arguments[] = {
        &runtimeAddress,
        &taskAddress,
        &taskCount,
        &queryAddress,
        &partialAddress,
    };
    launch(function, taskCount, AttentionHeadDimension, stream, arguments,
           "attention tile");
  }

  void discover(CUstream stream, nta::abi::RuntimeView *runtime,
                const AttentionTileTask *tasks, std::uint32_t taskCount,
                const __half *queries, AttentionTilePartial *partials) const {
    tile(stream, tile_, runtime, tasks, taskCount, queries, partials);
  }

  void discoverTma(CUstream stream, nta::abi::RuntimeView *runtime,
                   const AttentionTileTask *tasks, std::uint32_t taskCount,
                   const __half *queries,
                   AttentionTilePartial *partials) const {
    tile(stream, tmaTile_, runtime, tasks, taskCount, queries, partials);
  }

  void ready(CUstream stream, nta::abi::RuntimeView *runtime,
             const AttentionTileTask *tasks, std::uint32_t taskCount,
             const __half *queries, AttentionTilePartial *partials) const {
    tile(stream, ready_, runtime, tasks, taskCount, queries, partials);
  }

  void readyTma(CUstream stream, nta::abi::RuntimeView *runtime,
                const AttentionTileTask *tasks, std::uint32_t taskCount,
                const __half *queries,
                AttentionTilePartial *partials) const {
    tile(stream, tmaReady_, runtime, tasks, taskCount, queries, partials);
  }

  void progress(CUstream stream, nta::abi::RuntimeView *runtime,
                std::uint32_t capacity) const {
    CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
    void *arguments[] = {&runtimeAddress};
    launch(progress_, capacity, 256, stream, arguments, "host progress");
  }

  void publish(CUstream stream, nta::abi::RuntimeView *runtime,
               std::uint32_t continuationCount) const {
    const std::uint32_t threads = 256;
    const std::uint32_t blocks =
        (continuationCount + threads - 1U) / threads;
    CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
    void *arguments[] = {&runtimeAddress, &continuationCount};
    launch(publish_, blocks, threads, stream, arguments, "publish ready");
  }

  void reduce(CUstream stream, nta::abi::RuntimeView *runtime,
              const AttentionRequest *requests, std::uint32_t requestCount,
              const AttentionTilePartial *partials, float *output) const {
    CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
    CUdeviceptr requestAddress = reinterpret_cast<CUdeviceptr>(requests);
    CUdeviceptr partialAddress = reinterpret_cast<CUdeviceptr>(partials);
    CUdeviceptr outputAddress = reinterpret_cast<CUdeviceptr>(output);
    void *arguments[] = {
        &runtimeAddress,
        &requestAddress,
        &requestCount,
        &partialAddress,
        &outputAddress,
    };
    launch(reduce_, requestCount, AttentionHeadDimension, stream, arguments,
           "attention reduce");
  }

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
  CUfunction reset_ = nullptr;
  CUfunction progress_ = nullptr;
  CUfunction publish_ = nullptr;
  CUfunction tile_ = nullptr;
  CUfunction ready_ = nullptr;
  CUfunction tmaTile_ = nullptr;
  CUfunction tmaReady_ = nullptr;
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
    } else if (name == "--copy") {
      if (value == "global") {
        options.copyMode = CopyMode::Global;
      } else if (value == "tma") {
        options.copyMode = CopyMode::Tma;
      } else {
        throw std::invalid_argument("unknown copy mode");
      }
    } else {
      throw std::invalid_argument("unknown option " + std::string(name));
    }
  }
  if (options.minPages > options.maxPages) {
    throw std::invalid_argument("--min-pages must not exceed --max-pages");
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

CUtensorMap encodePageTensorMap(void *address) {
  CUtensorMap tensorMap{};
  constexpr cuuint64_t globalDimensions[2] = {
      AttentionHeadDimension, 2U * AttentionPageTokens};
  constexpr cuuint64_t globalStrides[1] = {
      AttentionHeadDimension * sizeof(__half)};
  constexpr cuuint32_t boxDimensions[2] = {
      AttentionHeadDimension, 2U * AttentionPageTokens};
  constexpr cuuint32_t elementStrides[2] = {1, 1};
  checkDriver(
      cuTensorMapEncodeTiled(
          &tensorMap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, address,
          globalDimensions, globalStrides, boxDimensions, elementStrides,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
          CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled attention page");
  return tensorMap;
}

std::vector<float> referenceAttention(
    const std::vector<AttentionRequest> &requests,
    const std::vector<AttentionTileTask> &tasks,
    const std::vector<__half> &queries, const std::vector<__half> &pages) {
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
      const __half *keys = pages.data() + taskIndex * valuesPerPage;
      for (std::uint32_t token = 0; token < task.tokenCount; ++token) {
        float dot = 0.0F;
        for (std::uint32_t dimension = 0;
             dimension < AttentionHeadDimension; ++dimension) {
          dot += __half2float(
                     queries[requestIndex * AttentionHeadDimension + dimension]) *
                 __half2float(keys[token * AttentionHeadDimension + dimension]);
        }
        logits.push_back(dot * 0.08838834764831845F);
        tokenTasks.push_back(taskIndex);
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
        const __half *values =
            pages.data() + tokenTasks[token] * valuesPerPage +
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

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parseOptions(argc, argv);
    checkDriver(cuInit(0), "cuInit");
    checkCuda(cudaSetDevice(0), "cudaSetDevice");

    std::vector<AttentionRequest> requests(options.requests);
    std::uint32_t taskCount = 0;
    for (std::uint32_t request = 0; request < options.requests; ++request) {
      const std::uint32_t pages =
          options.minPages + request % (options.maxPages - options.minPages + 1U);
      requests[request] = {taskCount, pages, request, request + 1U};
      taskCount += pages;
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
      runtime.setRequest(request, 0x4154544e00000000ULL + request,
                         request + 1U, request % 4U, request % 8U, 0,
                         requestCredit);
    }

    std::vector<AttentionTileTask> tasks(taskCount);
    std::unique_ptr<DeviceBuffer<CUtensorMap>> deviceTensorMaps;
    std::vector<CUtensorMap> tensorMaps;
    if (options.copyMode == CopyMode::Tma) {
      deviceTensorMaps =
          std::make_unique<DeviceBuffer<CUtensorMap>>(2ULL * taskCount);
      tensorMaps.resize(2ULL * taskCount);
    }
    std::uint32_t taskIndex = 0;
    for (std::uint32_t request = 0; request < options.requests; ++request) {
      for (std::uint32_t tile = 0; tile < requests[request].tileCount;
           ++tile, ++taskIndex) {
        const std::uint32_t tokenCount =
            tile + 1U == requests[request].tileCount
                ? 1U + (request * 7U) % AttentionPageTokens
                : AttentionPageTokens;
        const auto *page = reinterpret_cast<const std::byte *>(
            pages.data() + static_cast<std::size_t>(taskIndex) * valuesPerPage);
        const std::uint64_t objectId = 0x4b56504700000000ULL + taskIndex;
        const nta::Placement placement = placementFor(options.mode, taskIndex);
        const nta::ObjectHandle object = runtime.installObject(
            taskIndex, objectId, 1, std::span<const std::byte>(page, pageBytes),
            placement);
        std::uint64_t directTensorMap = 0;
        if (options.copyMode == CopyMode::Tma) {
          const nta::abi::ObjectEntry objectEntry = runtime.readObject(taskIndex);
          const nta::abi::ReplicaEntry replica = runtime.readReplica(taskIndex);
          tensorMaps[2ULL * taskIndex] = encodePageTensorMap(
              reinterpret_cast<void *>(replica.sourceAddress));
          const void *replicaMap = deviceTensorMaps->get() + 2ULL * taskIndex;
          const void *stagingMap = nullptr;
          if (placement == nta::Placement::HostStaged) {
            tensorMaps[2ULL * taskIndex + 1ULL] = encodePageTensorMap(
                reinterpret_cast<void *>(objectEntry.stagingAddress));
            stagingMap =
                deviceTensorMaps->get() + 2ULL * taskIndex + 1ULL;
          } else {
            directTensorMap = reinterpret_cast<std::uint64_t>(replicaMap);
          }
          runtime.bindTensorMaps(taskIndex, 0, replicaMap, stagingMap);
        }
        tasks[taskIndex] = {
            reinterpret_cast<std::uint64_t>(object.directDeviceBase),
            directTensorMap,
            objectId,
            request,
            request + 1U,
            taskIndex,
            1,
            static_cast<std::uint32_t>(pageBytes),
            taskIndex,
            request,
            tokenCount,
        };
      }
    }
    if (deviceTensorMaps != nullptr) {
      checkCuda(cudaMemcpy(deviceTensorMaps->get(), tensorMaps.data(),
                           tensorMaps.size() * sizeof(tensorMaps.front()),
                           cudaMemcpyHostToDevice),
                "upload attention tensor maps");
    }
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
    kernels.reset(driverStream, runtime.deviceView(), taskCount, taskCount);
    checkCuda(cudaMemsetAsync(devicePartials.get(), 0,
                              tasks.size() * sizeof(AttentionTilePartial), stream),
              "clear attention partials");
    checkCuda(cudaMemsetAsync(deviceOutput.get(), 0,
                              expected.size() * sizeof(float), stream),
              "clear attention output");
    if (options.copyMode == CopyMode::Tma) {
      kernels.discoverTma(driverStream, runtime.deviceView(), deviceTasks.get(),
                          taskCount, deviceQueries.get(), devicePartials.get());
    } else {
      kernels.discover(driverStream, runtime.deviceView(), deviceTasks.get(),
                       taskCount, deviceQueries.get(), devicePartials.get());
    }
    if (options.mode == Mode::HostStaged || options.mode == Mode::Mixed) {
      for (std::uint32_t pass = 0; pass < options.progressPasses; ++pass) {
        kernels.progress(driverStream, runtime.deviceView(), taskCount);
        kernels.publish(driverStream, runtime.deviceView(), taskCount);
        if (options.copyMode == CopyMode::Tma) {
          kernels.readyTma(driverStream, runtime.deviceView(), deviceTasks.get(),
                           taskCount, deviceQueries.get(),
                           devicePartials.get());
        } else {
          kernels.ready(driverStream, runtime.deviceView(), deviceTasks.get(),
                        taskCount, deviceQueries.get(), devicePartials.get());
        }
      }
    }
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
      checkCuda(cudaGraphLaunch(graphExec, stream), "attention measured launch");
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

    std::uint32_t outputFailures = 0;
    std::uint32_t continuationFailures = 0;
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
      if (runtime.readContinuation(task).state !=
          static_cast<std::uint32_t>(nta::abi::ContinuationState::Done)) {
        ++continuationFailures;
      }
      const nta::abi::ObjectEntry object = runtime.readObject(task);
      if (object.state !=
          static_cast<std::uint32_t>(nta::abi::ObjectState::Ready)) {
        ++objectFailures;
        if (objectFailures <= 4) {
          std::cerr << "unready_task=" << task << " object_state="
                    << object.state << " issue_count=" << object.issueCount
                    << " continuation_state="
                    << runtime.readContinuation(task).state << '\n';
        }
      }
    }
    for (std::uint32_t request = 0; request < options.requests; ++request) {
      if (runtime.readRequest(request).outstandingBytes != 0) {
        ++continuationFailures;
      }
    }
    const nta::abi::IntentPool intentPool = runtime.readIntentPool();
    if (intentPool.active != 0 || intentPool.overflow != 0) {
      ++objectFailures;
    }
    const std::uint32_t failures =
        outputFailures + continuationFailures + partialFailures + objectFailures;

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
              << " continuation_failures=" << continuationFailures
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
