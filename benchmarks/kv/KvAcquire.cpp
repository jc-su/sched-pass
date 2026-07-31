#include "benchmarks/kv/KvTypes.h"
#include "nta/HostRuntime.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef NTA_KV_CUBIN_PATH
#error "NTA_KV_CUBIN_PATH must identify the instrumented device image"
#endif

namespace {

enum class Mode {
  Resident,
  HostDirect,
  HostStaged,
  Mixed,
};

struct Options {
  Mode mode = Mode::Mixed;
  std::uint32_t requests = 96;
  std::uint32_t tileBytes = 64 * 1024;
  std::uint32_t iterations = 20;
  std::uint32_t cancelStride = 0;
  std::uint32_t staleStride = 0;
  std::uint32_t coalesce = 1;
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
    checkCuda(
        cudaMalloc(reinterpret_cast<void **>(&pointer_), sizeof(T) * count),
        "cudaMalloc DeviceBuffer");
  }
  ~DeviceBuffer() {
    if (pointer_ != nullptr) {
      (void)cudaFree(pointer_);
    }
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  T *get() const noexcept { return pointer_; }

private:
  T *pointer_ = nullptr;
};

class KernelModule {
public:
  KernelModule() {
    checkDriver(cuModuleLoad(&module_, NTA_KV_CUBIN_PATH),
                "cuModuleLoad instrumented cubin");
    checkDriver(cuModuleGetFunction(&reset_, module_, "nta_reset_epoch"),
                "cuModuleGetFunction nta_reset_epoch");
    checkDriver(cuModuleGetFunction(&compute_, module_, "nta_kv_tile_kernel"),
                "cuModuleGetFunction nta_kv_tile_kernel");
    checkDriver(
        cuModuleGetFunction(&progress_, module_, "nta_progress_host_staging"),
        "cuModuleGetFunction nta_progress_host_staging");
  }

  ~KernelModule() {
    if (module_ != nullptr) {
      (void)cuModuleUnload(module_);
    }
  }
  KernelModule(const KernelModule &) = delete;
  KernelModule &operator=(const KernelModule &) = delete;

  void launchReset(CUstream stream, nta::abi::RuntimeView *runtime,
                   std::uint32_t objectCount,
                   std::uint32_t continuationCount) const {
    const std::uint32_t threads = 256;
    const std::uint32_t blocks =
        (std::max(objectCount, continuationCount) + threads - 1) / threads;
    CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
    void *arguments[] = {
        &runtimeAddress,
        &objectCount,
        &continuationCount,
    };
    checkDriver(cuLaunchKernel(reset_, blocks, 1, 1, threads, 1, 1, 0, stream,
                               arguments, nullptr),
                "cuLaunchKernel reset");
  }

  void launchCompute(CUstream stream, nta::abi::RuntimeView *runtime,
                     const nta::benchmark::TileTask *tasks,
                     std::uint32_t taskCount, const float *query, float *output,
                     std::uint32_t phase) const {
    CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
    CUdeviceptr taskAddress = reinterpret_cast<CUdeviceptr>(tasks);
    CUdeviceptr queryAddress = reinterpret_cast<CUdeviceptr>(query);
    CUdeviceptr outputAddress = reinterpret_cast<CUdeviceptr>(output);
    void *arguments[] = {
        &runtimeAddress, &taskAddress,   &taskCount,
        &queryAddress,   &outputAddress, &phase,
    };
    checkDriver(cuLaunchKernel(compute_, taskCount, 1, 1, 256, 1, 1, 0, stream,
                               arguments, nullptr),
                "cuLaunchKernel compute");
  }

  void launchProgress(CUstream stream, nta::abi::RuntimeView *runtime,
                      std::uint32_t capacity) const {
    CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
    void *arguments[] = {&runtimeAddress};
    checkDriver(cuLaunchKernel(progress_, capacity, 1, 1, 256, 1, 1, 0, stream,
                               arguments, nullptr),
                "cuLaunchKernel progress");
  }

private:
  CUmodule module_ = nullptr;
  CUfunction reset_ = nullptr;
  CUfunction compute_ = nullptr;
  CUfunction progress_ = nullptr;
};

std::uint32_t parsePositive(std::string_view text, std::string_view option) {
  char *end = nullptr;
  const std::string storage(text);
  const unsigned long parsed = std::strtoul(storage.c_str(), &end, 10);
  if (end == storage.c_str() || *end != '\0' || parsed == 0 ||
      parsed > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("invalid value for " + std::string(option));
  }
  return static_cast<std::uint32_t>(parsed);
}

Mode parseMode(std::string_view value) {
  if (value == "resident") {
    return Mode::Resident;
  }
  if (value == "host-direct") {
    return Mode::HostDirect;
  }
  if (value == "host-staged") {
    return Mode::HostStaged;
  }
  if (value == "mixed") {
    return Mode::Mixed;
  }
  throw std::invalid_argument("unknown --mode value");
}

std::string_view modeName(Mode mode) {
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
      options.mode = parseMode(value);
    } else if (name == "--requests") {
      options.requests = parsePositive(value, name);
    } else if (name == "--tile-bytes") {
      options.tileBytes = parsePositive(value, name);
    } else if (name == "--iterations") {
      options.iterations = parsePositive(value, name);
    } else if (name == "--cancel-stride") {
      options.cancelStride = value == "0" ? 0 : parsePositive(value, name);
    } else if (name == "--stale-stride") {
      options.staleStride = value == "0" ? 0 : parsePositive(value, name);
    } else if (name == "--coalesce") {
      options.coalesce = parsePositive(value, name);
    } else {
      throw std::invalid_argument("unknown option " + std::string(name));
    }
  }
  if (options.tileBytes % 16 != 0) {
    throw std::invalid_argument(
        "--tile-bytes must be a positive multiple of 16");
  }
  return options;
}

nta::Placement placementFor(Mode mode, std::uint32_t index) {
  switch (mode) {
  case Mode::Resident:
    return nta::Placement::Hbm;
  case Mode::HostDirect:
    return nta::Placement::HostMapped;
  case Mode::HostStaged:
    return nta::Placement::HostStaged;
  case Mode::Mixed:
    switch (index % 3) {
    case 0:
      return nta::Placement::Hbm;
    case 1:
      return nta::Placement::HostMapped;
    default:
      return nta::Placement::HostStaged;
    }
  }
  return nta::Placement::Hbm;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parseOptions(argc, argv);
    checkDriver(cuInit(0), "cuInit");
    checkCuda(cudaSetDevice(0), "cudaSetDevice");

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
      throw std::runtime_error("no CUDA device is available");
    }
    cudaDeviceProp properties{};
    checkCuda(cudaGetDeviceProperties(&properties, 0),
              "cudaGetDeviceProperties");

    const std::uint32_t elements = options.tileBytes / sizeof(float);
    const std::uint32_t objectCount =
        options.requests / options.coalesce +
        (options.requests % options.coalesce != 0 ? 1U : 0U);
    std::vector<float> query(elements);
    for (std::uint32_t i = 0; i < elements; ++i) {
      query[i] = 0.25F + static_cast<float>((i * 7U) % 29U) / 31.0F;
    }

    nta::HostRuntime runtime(
        {options.requests, objectCount, objectCount, options.requests});
    std::vector<nta::benchmark::TileTask> tasks(options.requests);
    std::vector<std::vector<float>> objectData(objectCount);
    std::vector<nta::ObjectHandle> objects(objectCount);
    std::vector<float> expected(options.requests, 0.0F);
    std::vector<bool> cancelled(options.requests, false);
    std::vector<bool> stale(options.requests, false);

    for (std::uint32_t object = 0; object < objectCount; ++object) {
      objectData[object].resize(elements);
      for (std::uint32_t element = 0; element < elements; ++element) {
        objectData[object][element] =
            0.5F +
            static_cast<float>((object * 13U + element * 5U) % 97U) / 101.0F;
      }
      const std::span<const float> floats(objectData[object]);
      objects[object] = runtime.installObject(
          object, 200000U + object, 1, std::as_bytes(floats),
          placementFor(options.mode, object));
    }

    for (std::uint32_t task = 0; task < options.requests; ++task) {
      const std::uint32_t generation = 100U + task;
      runtime.setRequest(task, 100000U + task, generation, task % 4, task % 3);
      if (options.cancelStride != 0 && task % options.cancelStride == 0) {
        runtime.cancelRequest(task, generation);
        cancelled[task] = true;
      }
      const std::uint32_t taskGeneration =
          options.staleStride != 0 && task % options.staleStride == 0
              ? generation - 1
              : generation;
      stale[task] = taskGeneration != generation;

      const std::uint32_t object = task / options.coalesce;
      double reference = 0.0;
      for (std::uint32_t element = 0; element < elements; ++element) {
        const float value = objectData[object][element];
        reference += static_cast<double>(value) * query[element];
      }
      expected[task] = static_cast<float>(reference);

      tasks[task] = {
          reinterpret_cast<std::uint64_t>(objects[object].directDeviceBase),
          200000U + object,
          0,
          task,
          taskGeneration,
          object,
          1,
          options.tileBytes,
          task,
          0,
          0,
      };
    }

    DeviceBuffer<nta::benchmark::TileTask> deviceTasks(tasks.size());
    DeviceBuffer<float> deviceQuery(query.size());
    DeviceBuffer<float> deviceOutput(options.requests);
    checkCuda(cudaMemcpy(deviceTasks.get(), tasks.data(),
                         sizeof(tasks.front()) * tasks.size(),
                         cudaMemcpyHostToDevice),
              "upload tasks");
    checkCuda(cudaMemcpy(deviceQuery.get(), query.data(),
                         sizeof(float) * query.size(), cudaMemcpyHostToDevice),
              "upload query");

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
    kernels.launchReset(driverStream, runtime.deviceView(), objectCount,
                        options.requests);
    checkCuda(cudaMemsetAsync(deviceOutput.get(), 0,
                              sizeof(float) * options.requests, stream),
              "cudaMemsetAsync output");
    kernels.launchCompute(driverStream, runtime.deviceView(), deviceTasks.get(),
                          options.requests, deviceQuery.get(),
                          deviceOutput.get(), 0);
    kernels.launchProgress(driverStream, runtime.deviceView(), objectCount);
    kernels.launchCompute(driverStream, runtime.deviceView(), deviceTasks.get(),
                          options.requests, deviceQuery.get(),
                          deviceOutput.get(), 1);
    checkCuda(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
    checkCuda(cudaGraphInstantiate(&graphExec, graph, 0),
              "cudaGraphInstantiate");

    for (int warmup = 0; warmup < 3; ++warmup) {
      checkCuda(cudaGraphLaunch(graphExec, stream), "cudaGraphLaunch warmup");
    }
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize warmup");

    checkCuda(cudaEventRecord(begin, stream), "cudaEventRecord begin");
    for (std::uint32_t iteration = 0; iteration < options.iterations;
         ++iteration) {
      checkCuda(cudaGraphLaunch(graphExec, stream), "cudaGraphLaunch measured");
    }
    checkCuda(cudaEventRecord(end, stream), "cudaEventRecord end");
    checkCuda(cudaEventSynchronize(end), "cudaEventSynchronize");

    float elapsedMilliseconds = 0.0F;
    checkCuda(cudaEventElapsedTime(&elapsedMilliseconds, begin, end),
              "cudaEventElapsedTime");

    std::vector<float> output(options.requests);
    checkCuda(cudaMemcpy(output.data(), deviceOutput.get(),
                         sizeof(float) * output.size(), cudaMemcpyDeviceToHost),
              "download output");

    std::uint32_t failures = 0;
    std::uint64_t stagedIssues = 0;
    for (std::uint32_t task = 0; task < options.requests; ++task) {
      const nta::abi::Continuation continuation =
          runtime.readContinuation(task);
      if (cancelled[task] || stale[task]) {
        if (continuation.state != static_cast<std::uint32_t>(
                                      nta::abi::ContinuationState::Cancelled) ||
            output[task] != 0.0F) {
          ++failures;
        }
      } else {
        const float tolerance =
            std::max(0.02F, std::abs(expected[task]) * 2.0e-5F);
        if (continuation.state !=
                static_cast<std::uint32_t>(nta::abi::ContinuationState::Done) ||
            std::abs(output[task] - expected[task]) > tolerance) {
          ++failures;
        }
      }
    }
    for (std::uint32_t object = 0; object < objectCount; ++object) {
      stagedIssues += runtime.readObject(object).issueCount;
    }

    const double millisecondsPerBatch =
        elapsedMilliseconds / options.iterations;
    const double gibPerSecond =
        (static_cast<double>(options.requests) * options.tileBytes /
         (1024.0 * 1024.0 * 1024.0)) /
        (millisecondsPerBatch / 1000.0);

    std::cout << "device=" << properties.name
              << " mode=" << modeName(options.mode)
              << " requests=" << options.requests << " objects=" << objectCount
              << " coalesce=" << options.coalesce
              << " tile_bytes=" << options.tileBytes
              << " graph_ms=" << std::fixed << std::setprecision(3)
              << millisecondsPerBatch
              << " logical_GiB/s=" << std::setprecision(2) << gibPerSecond
              << " staged_issues=" << stagedIssues << " cancelled="
              << std::count(cancelled.begin(), cancelled.end(), true)
              << " stale=" << std::count(stale.begin(), stale.end(), true)
              << " verification_failures=" << failures << '\n';

    (void)cudaEventDestroy(end);
    (void)cudaEventDestroy(begin);
    (void)cudaGraphExecDestroy(graphExec);
    (void)cudaGraphDestroy(graph);
    (void)cudaStreamDestroy(stream);
    return failures == 0 ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "nta-kv-bench failed: " << error.what() << '\n';
    return 1;
  }
}
