#include "nta/DeviceWorkPlan.h"
#include "nta/FinitePhase.h"
#include "nta/HostRuntime.h"
#include "nta/WorkPlan.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

enum class Mode { Resident, HostDirect, HostStaged, Mixed };

struct Options {
  Mode mode = Mode::Mixed;
  std::uint32_t tokens = 64;
  std::uint32_t experts = 16;
  std::uint32_t topK = 2;
  std::uint32_t hidden = 128;
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

std::uint32_t parsePositive(std::string_view text, std::string_view option) {
  const std::string storage(text);
  char *end = nullptr;
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

nta::Placement placement(Mode mode, std::uint32_t expert) {
  if (mode == Mode::Resident) {
    return nta::Placement::Hbm;
  }
  if (mode == Mode::HostDirect) {
    return nta::Placement::HostMapped;
  }
  if (mode == Mode::HostStaged) {
    return nta::Placement::HostStaged;
  }
  switch (expert % 3U) {
  case 0:
    return nta::Placement::Hbm;
  case 1:
    return nta::Placement::HostMapped;
  default:
    return nta::Placement::HostStaged;
  }
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
    } else if (name == "--tokens") {
      options.tokens = parsePositive(value, name);
    } else if (name == "--experts") {
      options.experts = parsePositive(value, name);
    } else if (name == "--top-k") {
      options.topK = parsePositive(value, name);
    } else if (name == "--hidden") {
      options.hidden = parsePositive(value, name);
    } else if (name == "--iterations") {
      options.iterations = parsePositive(value, name);
    } else {
      throw std::invalid_argument("unknown option " + std::string(name));
    }
  }
  if (options.topK > options.experts || options.topK > 32) {
    throw std::invalid_argument("--top-k must fit the expert set and NTA ABI");
  }
  if (options.hidden > 256 || options.hidden % 32 != 0) {
    throw std::invalid_argument("--hidden must be a multiple of 32 up to 256");
  }
  return options;
}

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t count) {
    checkCuda(
        cudaMalloc(reinterpret_cast<void **>(&pointer_), count * sizeof(T)),
        "cudaMalloc MoE buffer");
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
                "cuModuleLoad MoE cubin");
    load(tile_, "nta_moe_tile_kernel");
    load(ready_, "nta_moe_ready_kernel");
  }
  ~KernelModule() {
    if (module_ != nullptr) {
      (void)cuModuleUnload(module_);
    }
  }
  KernelModule(const KernelModule &) = delete;
  KernelModule &operator=(const KernelModule &) = delete;

  void compute(CUfunction kernel, CUstream stream,
               nta::abi::RuntimeView *runtime, const nta::DeviceWorkPlan &plan,
               const float *input, float *output, std::uint32_t hidden) const {
    CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
    CUdeviceptr workAddress = reinterpret_cast<CUdeviceptr>(plan.workItems());
    CUdeviceptr dependencyAddress =
        reinterpret_cast<CUdeviceptr>(plan.dependencies());
    CUdeviceptr inputAddress = reinterpret_cast<CUdeviceptr>(input);
    CUdeviceptr outputAddress = reinterpret_cast<CUdeviceptr>(output);
    std::uint32_t workCount = plan.workItemCount();
    void *arguments[] = {
        &runtimeAddress, &workAddress,   &workCount, &dependencyAddress,
        &inputAddress,   &outputAddress, &hidden};
    launch(kernel, workCount, hidden, stream, arguments,
           kernel == tile_ ? "initial MoE compute" : "resumed MoE compute");
  }

  [[nodiscard]] CUfunction tile() const noexcept { return tile_; }
  [[nodiscard]] CUfunction ready() const noexcept { return ready_; }
  [[nodiscard]] CUmodule module() const noexcept { return module_; }

private:
  void load(CUfunction &function, const char *name) {
    checkDriver(cuModuleGetFunction(&function, module_, name), name);
  }

  static void launch(CUfunction function, std::uint32_t blocks,
                     std::uint32_t threads, CUstream stream, void **arguments,
                     const char *operation) {
    checkDriver(cuLaunchKernel(function, blocks, 1, 1, threads, 1, 1, 0, stream,
                               arguments, nullptr),
                operation);
  }

  CUmodule module_ = nullptr;
  CUfunction tile_ = nullptr;
  CUfunction ready_ = nullptr;
};

std::vector<std::uint32_t> route(std::uint32_t token, std::uint32_t expertCount,
                                 std::uint32_t topK) {
  std::vector<std::uint32_t> selected;
  selected.reserve(topK);
  std::uint32_t candidate = (token * 5U + 3U) % expertCount;
  while (selected.size() < topK) {
    if (std::find(selected.begin(), selected.end(), candidate) ==
        selected.end()) {
      selected.push_back(candidate);
    }
    candidate = (candidate + 7U) % expertCount;
    if (std::find(selected.begin(), selected.end(), candidate) !=
        selected.end()) {
      candidate = (candidate + 1U) % expertCount;
    }
  }
  return selected;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parseOptions(argc, argv);
    checkDriver(cuInit(0), "cuInit");
    checkCuda(cudaSetDevice(0), "cudaSetDevice");

    const std::size_t matrixElements =
        static_cast<std::size_t>(options.hidden) * options.hidden;
    const std::uint32_t matrixBytes =
        static_cast<std::uint32_t>(matrixElements * sizeof(float));
    nta::HostRuntime runtime({options.tokens, options.experts, options.experts,
                              options.tokens, 1, options.topK});

    std::vector<std::vector<float>> weights(options.experts);
    std::vector<nta::ObjectHandle> objects(options.experts);
    for (std::uint32_t expert = 0; expert < options.experts; ++expert) {
      weights[expert].resize(matrixElements);
      for (std::size_t element = 0; element < matrixElements; ++element) {
        weights[expert][element] =
            static_cast<float>((expert * 17U + element * 3U) % 101U) / 127.0F -
            0.35F;
      }
      objects[expert] = runtime.installObject(
          expert, 700000U + expert, 1,
          std::as_bytes(std::span<const float>(weights[expert])),
          placement(options.mode, expert));
    }

    std::vector<float> input(static_cast<std::size_t>(options.tokens) *
                             options.hidden);
    std::vector<float> expected(input.size(), 0.0F);
    for (std::size_t element = 0; element < input.size(); ++element) {
      input[element] = 0.2F + static_cast<float>((element * 11U) % 67U) / 89.0F;
    }

    nta::WorkPlanBuilder builder(options.topK);
    for (std::uint32_t token = 0; token < options.tokens; ++token) {
      const std::uint32_t generation = 900U + token;
      runtime.setRequest(token, 800000U + token, generation, token % 4,
                         token % 3);
      const std::vector<std::uint32_t> selected =
          route(token, options.experts, options.topK);
      std::vector<nta::abi::AcquireRequirement> dependencies;
      dependencies.reserve(options.topK);
      for (std::uint32_t expert : selected) {
        dependencies.push_back({
            reinterpret_cast<std::uint64_t>(objects[expert].directDeviceBase),
            0,
            700000U + expert,
            0,
            expert,
            1,
            matrixBytes,
            0,
        });
      }
      const std::uint32_t request = builder.addRequest({token, generation});
      (void)builder.addWork(request, token, dependencies);

      for (std::uint32_t outputIndex = 0; outputIndex < options.hidden;
           ++outputIndex) {
        float mixed = 0.0F;
        for (std::uint32_t dependency = 0; dependency < selected.size();
             ++dependency) {
          const float *row = weights[selected[dependency]].data() +
                             outputIndex * options.hidden;
          float expertOutput = 0.0F;
          for (std::uint32_t inputIndex = 0; inputIndex < options.hidden;
               ++inputIndex) {
            expertOutput = std::fma(row[inputIndex],
                                    input[token * options.hidden + inputIndex],
                                    expertOutput);
          }
          mixed = std::fma(1.0F / static_cast<float>(dependency + 1U),
                           expertOutput, mixed);
        }
        expected[token * options.hidden + outputIndex] = mixed;
      }
    }

    const nta::WorkPlan hostPlan = builder.finish();
    nta::DeviceWorkPlan devicePlan = runtime.uploadWorkPlan(hostPlan);
    DeviceBuffer<float> deviceInput(input.size());
    DeviceBuffer<float> deviceOutput(input.size());
    checkCuda(cudaMemcpy(deviceInput.get(), input.data(),
                         input.size() * sizeof(float), cudaMemcpyHostToDevice),
              "upload MoE input");

    KernelModule kernels;
    nta::FinitePhaseProgram phases(kernels.module());
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "create MoE stream");
    checkCuda(cudaEventCreate(&begin), "create MoE begin event");
    checkCuda(cudaEventCreate(&end), "create MoE end event");
    const CUstream driverStream = reinterpret_cast<CUstream>(stream);

    checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
              "begin MoE graph capture");
    phases.enqueueHost(
        driverStream, runtime.deviceView(),
        {options.experts, options.tokens, options.experts, 1},
        [&] {
          checkCuda(cudaMemsetAsync(deviceOutput.get(), 0,
                                    input.size() * sizeof(float), stream),
                    "clear MoE output");
          kernels.compute(kernels.tile(), driverStream, runtime.deviceView(),
                          devicePlan, deviceInput.get(), deviceOutput.get(),
                          options.hidden);
        },
        [&] {
          kernels.compute(kernels.ready(), driverStream, runtime.deviceView(),
                          devicePlan, deviceInput.get(), deviceOutput.get(),
                          options.hidden);
        });
    checkCuda(cudaStreamEndCapture(stream, &graph), "end MoE graph capture");
    checkCuda(cudaGraphInstantiate(&graphExec, graph, 0),
              "instantiate MoE graph");

    for (int warmup = 0; warmup < 2; ++warmup) {
      checkCuda(cudaGraphLaunch(graphExec, stream), "launch MoE warmup");
    }
    checkCuda(cudaStreamSynchronize(stream), "synchronize MoE warmup");
    checkCuda(cudaEventRecord(begin, stream), "record MoE begin");
    for (std::uint32_t iteration = 0; iteration < options.iterations;
         ++iteration) {
      checkCuda(cudaGraphLaunch(graphExec, stream), "launch MoE graph");
    }
    checkCuda(cudaEventRecord(end, stream), "record MoE end");
    checkCuda(cudaEventSynchronize(end), "synchronize MoE end");

    float elapsed = 0.0F;
    checkCuda(cudaEventElapsedTime(&elapsed, begin, end), "measure MoE graph");
    std::vector<float> output(input.size());
    checkCuda(cudaMemcpy(output.data(), deviceOutput.get(),
                         output.size() * sizeof(float), cudaMemcpyDeviceToHost),
              "download MoE output");

    std::uint32_t failures = 0;
    float maxError = 0.0F;
    for (std::uint32_t token = 0; token < options.tokens; ++token) {
      if (runtime.readContinuation(token).state !=
          static_cast<std::uint32_t>(nta::abi::ContinuationState::Done)) {
        ++failures;
        continue;
      }
      for (std::uint32_t index = 0; index < options.hidden; ++index) {
        const std::size_t offset =
            static_cast<std::size_t>(token) * options.hidden + index;
        const float error = std::abs(output[offset] - expected[offset]);
        maxError = std::max(maxError, error);
        const float tolerance =
            std::max(2.0e-4F, std::abs(expected[offset]) * 2.0e-5F);
        failures += error > tolerance ? 1U : 0U;
      }
    }
    std::uint64_t stagedIssues = 0;
    for (std::uint32_t expert = 0; expert < options.experts; ++expert) {
      stagedIssues += runtime.readObject(expert).issueCount;
    }

    const double graphMilliseconds = elapsed / options.iterations;
    const double logicalGiB = static_cast<double>(options.tokens) *
                              options.topK * matrixBytes /
                              (1024.0 * 1024.0 * 1024.0);
    std::cout << "workload=moe mode=" << modeName(options.mode)
              << " tokens=" << options.tokens << " experts=" << options.experts
              << " top_k=" << options.topK << " hidden=" << options.hidden
              << " graph_ms=" << std::fixed << std::setprecision(3)
              << graphMilliseconds << " logical_GiB/s=" << std::setprecision(2)
              << logicalGiB / (graphMilliseconds / 1000.0)
              << " staged_issues=" << stagedIssues
              << " pending=" << runtime.readPendingCount()
              << " max_abs_error=" << std::scientific << maxError
              << " verification_failures=" << failures << '\n';

    (void)cudaEventDestroy(end);
    (void)cudaEventDestroy(begin);
    (void)cudaGraphExecDestroy(graphExec);
    (void)cudaGraphDestroy(graph);
    (void)cudaStreamDestroy(stream);
    return failures == 0 ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "nta-moe-bench failed: " << error.what() << '\n';
    return 1;
  }
}
