#include "benchmarks/kv/KvTypes.h"
#include "nta/HostRuntime.h"
#include "nta/NvmeRuntime.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef NTA_KV_CUBIN_PATH
#error "NTA_KV_CUBIN_PATH must identify the instrumented device image"
#endif

namespace {

struct Options {
  std::string device = "/dev/nta_nvme";
  std::string reference = "/tmp/nta-nvme-reference.bin";
  nta::NvmeDestination destination = nta::NvmeDestination::Hbm;
  std::uint64_t sourceOffset = 0;
  std::uint32_t bytes = 64U * 1024U;
  std::uint32_t requests = 16;
  std::uint32_t progressPasses = 64;
  std::uint32_t iterations = 20;
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

std::uint64_t parseInteger(std::string_view text, std::string_view option,
                           bool allowZero) {
  char *end = nullptr;
  const std::string storage(text);
  const unsigned long long parsed = std::strtoull(storage.c_str(), &end, 10);
  if (end == storage.c_str() || *end != '\0' || (!allowZero && parsed == 0)) {
    throw std::invalid_argument("invalid value for " + std::string(option));
  }
  return parsed;
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
    if (name == "--device") {
      options.device = value;
    } else if (name == "--reference") {
      options.reference = value;
    } else if (name == "--destination") {
      if (value == "hbm") {
        options.destination = nta::NvmeDestination::Hbm;
      } else if (value == "host-mapped") {
        options.destination = nta::NvmeDestination::HostMapped;
      } else {
        throw std::invalid_argument("unknown --destination value");
      }
    } else if (name == "--source-offset") {
      options.sourceOffset = parseInteger(value, name, true);
    } else if (name == "--bytes") {
      const std::uint64_t parsed = parseInteger(value, name, false);
      if (parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("--bytes exceeds uint32_t");
      }
      options.bytes = static_cast<std::uint32_t>(parsed);
    } else if (name == "--requests") {
      const std::uint64_t parsed = parseInteger(value, name, false);
      if (parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("--requests exceeds uint32_t");
      }
      options.requests = static_cast<std::uint32_t>(parsed);
    } else if (name == "--progress-passes") {
      const std::uint64_t parsed = parseInteger(value, name, false);
      if (parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("--progress-passes exceeds uint32_t");
      }
      options.progressPasses = static_cast<std::uint32_t>(parsed);
    } else if (name == "--iterations") {
      const std::uint64_t parsed = parseInteger(value, name, false);
      if (parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("--iterations exceeds uint32_t");
      }
      options.iterations = static_cast<std::uint32_t>(parsed);
    } else {
      throw std::invalid_argument("unknown option " + std::string(name));
    }
  }
  if (options.bytes % sizeof(std::uint32_t) != 0) {
    throw std::invalid_argument("--bytes must be a multiple of four");
  }
  return options;
}

std::vector<std::byte> readReference(const Options &options) {
  std::ifstream input(options.reference, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open reference file " + options.reference);
  }
  const std::uint64_t total =
      static_cast<std::uint64_t>(options.bytes) * options.requests;
  if (total > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("reference size exceeds size_t");
  }
  std::vector<std::byte> bytes(static_cast<std::size_t>(total));
  input.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
  if (input.gcount() != static_cast<std::streamsize>(bytes.size())) {
    throw std::runtime_error("reference file is shorter than --bytes");
  }
  return bytes;
}

std::uint64_t checksum(const std::byte *bytes, std::size_t byteCount) {
  std::uint64_t result = 0;
  for (std::size_t index = 0; index < byteCount / sizeof(std::uint32_t);
       ++index) {
    std::uint32_t value;
    std::memcpy(&value, bytes + index * sizeof(value), sizeof(value));
    result += static_cast<std::uint64_t>(value) * (index + 1ULL);
  }
  return result;
}

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t count) {
    checkCuda(
        cudaMalloc(reinterpret_cast<void **>(&pointer_), count * sizeof(T)),
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
    checkDriver(cuModuleLoad(&module_, NTA_KV_CUBIN_PATH),
                "cuModuleLoad instrumented cubin");
    checkDriver(cuModuleGetFunction(&reset_, module_, "nta_reset_epoch"),
                "cuModuleGetFunction nta_reset_epoch");
    checkDriver(cuModuleGetFunction(&hash_, module_, "nta_nvme_hash_kernel"),
                "cuModuleGetFunction nta_nvme_hash_kernel");
    checkDriver(cuModuleGetFunction(&ready_, module_,
                                    "nta_nvme_ready_hash_kernel"),
                "cuModuleGetFunction nta_nvme_ready_hash_kernel");
    checkDriver(cuModuleGetFunction(&publish_, module_, "nta_publish_ready"),
                "cuModuleGetFunction nta_publish_ready");
    checkDriver(cuModuleGetFunction(&progress_, module_, "nta_progress_nvme"),
                "cuModuleGetFunction nta_progress_nvme");
  }
  ~KernelModule() {
    if (module_ != nullptr) {
      (void)cuModuleUnload(module_);
    }
  }

  void reset(CUstream stream, nta::abi::RuntimeView *runtime,
             std::uint32_t count) const {
    CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
    void *arguments[] = {&runtimeAddress, &count, &count};
    checkDriver(cuLaunchKernel(reset_, 1, 1, 1, 256, 1, 1, 0, stream, arguments,
                               nullptr),
                "cuLaunchKernel reset");
  }

  void hash(CUstream stream, nta::abi::RuntimeView *runtime,
            const nta::benchmark::TileTask *tasks, std::uint32_t count,
            std::uint64_t *output, std::uint32_t phase) const {
    CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
    CUdeviceptr taskAddress = reinterpret_cast<CUdeviceptr>(tasks);
    CUdeviceptr outputAddress = reinterpret_cast<CUdeviceptr>(output);
    void *arguments[] = {
        &runtimeAddress, &taskAddress, &count, &outputAddress, &phase,
    };
    checkDriver(cuLaunchKernel(hash_, count, 1, 1, 256, 1, 1, 0, stream,
                               arguments, nullptr),
                "cuLaunchKernel NVMe hash");
  }

  void progress(CUstream stream, nta::abi::RuntimeView *runtime) const {
    std::uint32_t issueBudget = 32;
    std::uint32_t completionBudget = 32;
    CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
    void *arguments[] = {
        &runtimeAddress,
        &issueBudget,
        &completionBudget,
    };
    checkDriver(cuLaunchKernel(progress_, 1, 1, 1, 32, 1, 1, 0, stream,
                               arguments, nullptr),
                "cuLaunchKernel NVMe progress");
  }

  void publish(CUstream stream, nta::abi::RuntimeView *runtime,
               std::uint32_t continuationCount) const {
    const std::uint32_t threads = 256;
    const std::uint32_t blocks =
        (continuationCount + threads - 1U) / threads;
    CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
    void *arguments[] = {&runtimeAddress, &continuationCount};
    checkDriver(cuLaunchKernel(publish_, blocks, 1, 1, threads, 1, 1, 0,
                               stream, arguments, nullptr),
                "cuLaunchKernel publish NVMe ready");
  }

  void ready(CUstream stream, nta::abi::RuntimeView *runtime,
             const nta::benchmark::TileTask *tasks, std::uint32_t count,
             std::uint64_t *output) const {
    CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
    CUdeviceptr taskAddress = reinterpret_cast<CUdeviceptr>(tasks);
    CUdeviceptr outputAddress = reinterpret_cast<CUdeviceptr>(output);
    void *arguments[] = {
        &runtimeAddress,
        &taskAddress,
        &count,
        &outputAddress,
    };
    checkDriver(cuLaunchKernel(ready_, count, 1, 1, 256, 1, 1, 0, stream,
                               arguments, nullptr),
                "cuLaunchKernel NVMe ready hash");
  }

private:
  CUmodule module_ = nullptr;
  CUfunction reset_ = nullptr;
  CUfunction hash_ = nullptr;
  CUfunction ready_ = nullptr;
  CUfunction publish_ = nullptr;
  CUfunction progress_ = nullptr;
};

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parseOptions(argc, argv);
    const std::vector<std::byte> reference = readReference(options);
    std::vector<std::uint64_t> expected(options.requests);
    for (std::uint32_t request = 0; request < options.requests; ++request) {
      expected[request] = checksum(
          reference.data() + static_cast<std::size_t>(request) * options.bytes,
          options.bytes);
    }

    checkDriver(cuInit(0), "cuInit");
    checkCuda(cudaSetDevice(0), "cudaSetDevice");
    auto transport = std::make_shared<nta::NvmeTransport>(options.device);
    const nta::NvmeCapabilities capabilities = transport->capabilities();
    if (options.bytes % capabilities.lbaSize != 0 ||
        options.sourceOffset % capabilities.lbaSize != 0) {
      throw std::invalid_argument("NVMe source range must be LBA aligned");
    }

    const std::uint64_t totalSourceBytes =
        static_cast<std::uint64_t>(options.bytes) * options.requests;
    if (options.sourceOffset > capabilities.namespaceBytes ||
        totalSourceBytes > capabilities.namespaceBytes - options.sourceOffset) {
      throw std::invalid_argument("requested NVMe range exceeds namespace");
    }
    nta::HostRuntime runtime({options.requests, options.requests,
                              options.requests, options.requests},
                             transport);
    std::vector<nta::benchmark::TileTask> hostTasks(options.requests);
    for (std::uint32_t request = 0; request < options.requests; ++request) {
      std::unique_ptr<nta::NvmeBuffer> destination =
          transport->allocate(options.bytes, options.destination);
      const std::uint64_t objectId = 0x4e54414e00000000ULL + request;
      runtime.installNvmeObject(request, objectId, 1,
                                options.sourceOffset +
                                    static_cast<std::uint64_t>(request) *
                                        options.bytes,
                                options.bytes, std::move(destination));
      runtime.setRequest(request, 0x4e54410000000000ULL + request,
                         request + 1U);
      hostTasks[request] = {
          0,
          objectId,
          0,
          request,
          request + 1U,
          request,
          1,
          options.bytes,
          request,
          0,
          0,
      };
    }
    DeviceBuffer<nta::benchmark::TileTask> tasks(options.requests);
    DeviceBuffer<std::uint64_t> output(options.requests);
    checkCuda(cudaMemcpy(tasks.get(), hostTasks.data(),
                         sizeof(hostTasks.front()) * hostTasks.size(),
                         cudaMemcpyHostToDevice),
              "upload NVMe task");

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
    kernels.reset(driverStream, runtime.deviceView(), options.requests);
    checkCuda(cudaMemsetAsync(output.get(), 0,
                              sizeof(std::uint64_t) * options.requests, stream),
              "cudaMemsetAsync output");
    kernels.hash(driverStream, runtime.deviceView(), tasks.get(),
                 options.requests, output.get(), 0);
    for (std::uint32_t pass = 0; pass < options.progressPasses; ++pass) {
      kernels.progress(driverStream, runtime.deviceView());
      kernels.publish(driverStream, runtime.deviceView(), options.requests);
      kernels.ready(driverStream, runtime.deviceView(), tasks.get(),
                    options.requests, output.get());
    }
    checkCuda(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
    checkCuda(cudaGraphInstantiate(&graphExec, graph, 0),
              "cudaGraphInstantiate");

    checkCuda(cudaGraphLaunch(graphExec, stream), "cudaGraphLaunch warmup");
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize warmup");
    const nta::NvmeQueueStats warmupStats = transport->readStats();
    if (warmupStats.error != 0 || warmupStats.outstanding != 0) {
      throw std::runtime_error(
          "bounded warmup did not finish; increase --progress-passes");
    }
    for (std::uint32_t request = 0; request < options.requests; ++request) {
      if (runtime.readContinuation(request).state !=
          static_cast<std::uint32_t>(nta::abi::ContinuationState::Done)) {
        throw std::runtime_error(
            "bounded warmup did not resume every continuation");
      }
    }

    checkCuda(cudaEventRecord(begin, stream), "cudaEventRecord begin");
    for (std::uint32_t iteration = 0; iteration < options.iterations;
         ++iteration) {
      checkCuda(cudaGraphLaunch(graphExec, stream), "cudaGraphLaunch measured");
    }
    checkCuda(cudaEventRecord(end, stream), "cudaEventRecord end");
    checkCuda(cudaEventSynchronize(end), "cudaEventSynchronize end");

    float elapsedMilliseconds = 0.0F;
    checkCuda(cudaEventElapsedTime(&elapsedMilliseconds, begin, end),
              "cudaEventElapsedTime");
    std::vector<std::uint64_t> actual(options.requests);
    checkCuda(cudaMemcpy(actual.data(), output.get(),
                         sizeof(actual.front()) * actual.size(),
                         cudaMemcpyDeviceToHost),
              "download NVMe checksum");
    const nta::NvmeQueueStats stats = transport->readStats();
    std::uint32_t verificationFailures = 0;
    for (std::uint32_t request = 0; request < options.requests; ++request) {
      const nta::abi::Continuation continuation =
          runtime.readContinuation(request);
      if (actual[request] != expected[request] ||
          continuation.state !=
              static_cast<std::uint32_t>(nta::abi::ContinuationState::Done)) {
        ++verificationFailures;
      }
    }
    if (stats.error != 0 || stats.outstanding != 0 || stats.failed != 0) {
      ++verificationFailures;
    }
    const bool verified = verificationFailures == 0;
    const double milliseconds = elapsedMilliseconds / options.iterations;
    const double mibPerSecond = (static_cast<double>(options.bytes) *
                                 options.requests / (1024.0 * 1024.0)) /
                                (milliseconds / 1000.0);

    std::cout << "destination="
              << (options.destination == nta::NvmeDestination::Hbm
                      ? "hbm-dmabuf"
                      : "host-mapped")
              << " requests=" << options.requests << " bytes=" << options.bytes
              << " progress_passes=" << options.progressPasses
              << " graph_ms=" << std::fixed << std::setprecision(3)
              << milliseconds << " MiB/s=" << std::setprecision(2)
              << mibPerSecond << " submitted=" << stats.submitted
              << " completed=" << stats.completed << " failed=" << stats.failed
              << " verification_failures=" << verificationFailures << '\n';

    (void)cudaEventDestroy(end);
    (void)cudaEventDestroy(begin);
    (void)cudaGraphExecDestroy(graphExec);
    (void)cudaGraphDestroy(graph);
    (void)cudaStreamDestroy(stream);
    return verified ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "nta-nvme-bench failed: " << error.what() << '\n';
    return 1;
  }
}
