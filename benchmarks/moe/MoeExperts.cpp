#include "benchmarks/CommonCuda.h"
#include "benchmarks/kv/KvTypes.h"
#include "nta/FinitePhase.h"
#include "nta/HostRuntime.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
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

using nta::benchmark::checkCuda;
using nta::benchmark::checkDriver;
using nta::benchmark::DeviceBuffer;
using nta::benchmark::PinnedBuffer;

enum class Mode { Resident, HostDirect, HostStaged, Mixed };
enum class Policy { LateBound, CpuSync, Overfetch, Direct };

struct Options {
  Mode mode = Mode::Mixed;
  Policy policy = Policy::LateBound;
  std::uint32_t tokens = 64;
  std::uint32_t experts = 16;
  std::uint32_t topK = 2;
  std::uint32_t hidden = 128;
  std::uint32_t iterations = 20;
  bool json = false;
};

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

bool parseBoolean(std::string_view text, std::string_view option) {
  if (text == "0") {
    return false;
  }
  if (text == "1") {
    return true;
  }
  throw std::invalid_argument("invalid value for " + std::string(option));
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

Policy parsePolicy(std::string_view value) {
  if (value == "late-bound") {
    return Policy::LateBound;
  }
  if (value == "cpu-sync") {
    return Policy::CpuSync;
  }
  if (value == "overfetch") {
    return Policy::Overfetch;
  }
  if (value == "direct") {
    return Policy::Direct;
  }
  throw std::invalid_argument("unknown --policy value");
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

std::string_view policyName(Policy policy) {
  switch (policy) {
  case Policy::LateBound:
    return "late-bound";
  case Policy::CpuSync:
    return "cpu-sync";
  case Policy::Overfetch:
    return "overfetch";
  case Policy::Direct:
    return "direct";
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
    } else if (name == "--policy") {
      options.policy = parsePolicy(value);
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
    } else if (name == "--json") {
      options.json = parseBoolean(value, name);
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
  if (options.tokens >
      std::numeric_limits<std::uint32_t>::max() / options.hidden) {
    throw std::invalid_argument("--tokens times --hidden exceeds GPU indexing");
  }
  if (options.tokens >
      std::numeric_limits<std::uint32_t>::max() / options.topK) {
    throw std::invalid_argument("--tokens times --top-k exceeds GPU indexing");
  }
  if (options.policy == Policy::Direct && options.mode != Mode::Resident &&
      options.mode != Mode::HostDirect) {
    throw std::invalid_argument(
        "--policy=direct requires resident or host-direct placement");
  }
  return options;
}

class KernelModule {
public:
  KernelModule() {
    checkDriver(cuModuleLoad(&module_, NTA_KV_CUBIN_PATH),
                "cuModuleLoad MoE cubin");
    load(tile_, "nta_moe_tile_kernel");
    load(ready_, "nta_moe_ready_kernel");
    load(baseline_, "nta_moe_baseline_kernel");
    load(advanceEpoch_, "nta_moe_advance_epoch_kernel");
    load(prepareInput_, "nta_moe_prepare_input_kernel");
    load(route_, "nta_moe_route_kernel");
    load(copyAll_, "nta_moe_copy_all_kernel");
  }
  ~KernelModule() {
    if (module_ != nullptr) {
      (void)cuModuleUnload(module_);
    }
  }
  KernelModule(const KernelModule &) = delete;
  KernelModule &operator=(const KernelModule &) = delete;

  void prepareEpoch(CUstream stream, std::uint32_t *epoch,
                    const float *baseInput, float *input,
                    std::uint32_t elementCount) const {
    CUdeviceptr epochAddress = reinterpret_cast<CUdeviceptr>(epoch);
    void *arguments[] = {&epochAddress};
    launch(advanceEpoch_, 1, 1, stream, arguments, "advance MoE epoch");

    CUdeviceptr baseAddress = reinterpret_cast<CUdeviceptr>(baseInput);
    CUdeviceptr inputAddress = reinterpret_cast<CUdeviceptr>(input);
    void *prepareArguments[] = {&baseAddress, &inputAddress, &epochAddress,
                                &elementCount};
    launch(prepareInput_, (elementCount + 255U) / 256U, 256, stream,
           prepareArguments, "prepare MoE input");
  }

  void route(CUstream stream, nta::abi::RuntimeView *runtime,
             const nta::benchmark::MoeExpertDescriptor *experts,
             const float *gateWeights, const float *input,
             nta::abi::WorkItem *workItems,
             nta::abi::AcquireRequirement *requirements,
             std::uint32_t *selectedExperts, std::uint32_t tokens,
             std::uint32_t expertCount, std::uint32_t topK,
             std::uint32_t hidden, bool preacquired) const {
    CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
    CUdeviceptr expertAddress = reinterpret_cast<CUdeviceptr>(experts);
    CUdeviceptr gateAddress = reinterpret_cast<CUdeviceptr>(gateWeights);
    CUdeviceptr inputAddress = reinterpret_cast<CUdeviceptr>(input);
    CUdeviceptr workAddress = reinterpret_cast<CUdeviceptr>(workItems);
    CUdeviceptr requirementAddress =
        reinterpret_cast<CUdeviceptr>(requirements);
    CUdeviceptr selectedAddress =
        reinterpret_cast<CUdeviceptr>(selectedExperts);
    std::uint32_t acquired = preacquired ? 1U : 0U;
    void *arguments[] = {&runtimeAddress,  &expertAddress,
                         &gateAddress,     &inputAddress,
                         &workAddress,     &requirementAddress,
                         &selectedAddress, &tokens,
                         &expertCount,     &topK,
                         &hidden,          &acquired};
    launch(route_, tokens, 32, stream, arguments, "route MoE experts");
  }

  void compute(CUfunction kernel, CUstream stream,
               nta::abi::RuntimeView *runtime,
               const nta::abi::WorkItem *workItems,
               const nta::abi::AcquireRequirement *requirements,
               std::uint32_t workCount, const float *input, float *output,
               std::uint32_t hidden) const {
    CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
    CUdeviceptr workAddress = reinterpret_cast<CUdeviceptr>(workItems);
    CUdeviceptr requirementAddress =
        reinterpret_cast<CUdeviceptr>(requirements);
    CUdeviceptr inputAddress = reinterpret_cast<CUdeviceptr>(input);
    CUdeviceptr outputAddress = reinterpret_cast<CUdeviceptr>(output);
    void *arguments[] = {
        &runtimeAddress, &workAddress,   &workCount, &requirementAddress,
        &inputAddress,   &outputAddress, &hidden};
    launch(kernel, workCount, hidden, stream, arguments,
           kernel == tile_ ? "initial MoE compute" : "resumed MoE compute");
  }

  void baseline(CUstream stream, const nta::abi::WorkItem *workItems,
                const nta::abi::AcquireRequirement *requirements,
                std::uint32_t workCount, const float *input, float *output,
                std::uint32_t hidden) const {
    CUdeviceptr workAddress = reinterpret_cast<CUdeviceptr>(workItems);
    CUdeviceptr requirementAddress =
        reinterpret_cast<CUdeviceptr>(requirements);
    CUdeviceptr inputAddress = reinterpret_cast<CUdeviceptr>(input);
    CUdeviceptr outputAddress = reinterpret_cast<CUdeviceptr>(output);
    void *arguments[] = {&workAddress,  &workCount,     &requirementAddress,
                         &inputAddress, &outputAddress, &hidden};
    launch(baseline_, workCount, hidden, stream, arguments,
           "overfetch MoE compute");
  }

  void copyAll(CUstream stream,
               const nta::benchmark::MoeExpertDescriptor *experts,
               std::uint32_t expertCount) const {
    CUdeviceptr expertAddress = reinterpret_cast<CUdeviceptr>(experts);
    void *arguments[] = {&expertAddress, &expertCount};
    launch(copyAll_, expertCount, 256, stream, arguments,
           "copy all MoE experts");
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
  CUfunction baseline_ = nullptr;
  CUfunction advanceEpoch_ = nullptr;
  CUfunction prepareInput_ = nullptr;
  CUfunction route_ = nullptr;
  CUfunction copyAll_ = nullptr;
};

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
    const std::size_t selectionCount =
        static_cast<std::size_t>(options.tokens) * options.topK;
    const std::uint32_t intentCapacity = static_cast<std::uint32_t>(
        std::min<std::size_t>(options.experts, selectionCount));
    nta::HostRuntime runtime({options.tokens, options.experts, intentCapacity,
                              options.tokens, 1, options.topK});

    std::vector<std::vector<float>> weights(options.experts);
    std::vector<nta::ObjectHandle> objects(options.experts);
    std::vector<nta::benchmark::MoeExpertDescriptor> catalog(options.experts);
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
      const nta::abi::ObjectEntry object = runtime.readObject(expert);
      const nta::abi::ReplicaEntry replica = runtime.readReplica(expert);
      const bool staged =
          replica.sourceKind ==
          static_cast<std::uint32_t>(nta::abi::SourceKind::HostStaged);
      const std::uint64_t directBase =
          reinterpret_cast<std::uint64_t>(objects[expert].directDeviceBase);
      catalog[expert] = {directBase,
                         directBase == 0 ? object.stagingAddress : directBase,
                         staged ? replica.sourceAddress : 0,
                         object.objectId,
                         expert,
                         object.version,
                         matrixBytes,
                         staged ? nta::benchmark::MoeExpertStaged : 0U,
                         0,
                         0};
    }

    std::vector<float> input(static_cast<std::size_t>(options.tokens) *
                             options.hidden);
    for (std::size_t element = 0; element < input.size(); ++element) {
      input[element] = 0.2F + static_cast<float>((element * 11U) % 67U) / 89.0F;
    }
    std::vector<float> gateWeights(static_cast<std::size_t>(options.experts) *
                                   options.hidden);
    for (std::size_t element = 0; element < gateWeights.size(); ++element) {
      gateWeights[element] =
          (static_cast<float>((element * 13U + 5U) % 97U) / 96.0F - 0.5F) *
          0.02F;
    }
    for (std::uint32_t token = 0; token < options.tokens; ++token) {
      runtime.setRequest(token, 800000U + token, 900U + token, token % 4,
                         token % 3);
    }

    DeviceBuffer<nta::benchmark::MoeExpertDescriptor> deviceCatalog(
        catalog.size());
    DeviceBuffer<float> deviceGateWeights(gateWeights.size());
    DeviceBuffer<float> deviceBaseInput(input.size());
    DeviceBuffer<float> deviceInput(input.size());
    DeviceBuffer<float> deviceOutput(input.size());
    DeviceBuffer<nta::abi::WorkItem> deviceWork(options.tokens);
    DeviceBuffer<nta::abi::AcquireRequirement> deviceRequirements(
        selectionCount);
    DeviceBuffer<std::uint32_t> deviceSelected(selectionCount);
    DeviceBuffer<std::uint32_t> deviceEpoch(1);
    PinnedBuffer<std::uint32_t> cpuVisibleSelection(selectionCount);
    checkCuda(cudaMemcpy(deviceCatalog.get(), catalog.data(),
                         catalog.size() * sizeof(catalog.front()),
                         cudaMemcpyHostToDevice),
              "upload MoE expert catalog");
    checkCuda(cudaMemcpy(deviceGateWeights.get(), gateWeights.data(),
                         gateWeights.size() * sizeof(float),
                         cudaMemcpyHostToDevice),
              "upload MoE gate weights");
    checkCuda(cudaMemcpy(deviceBaseInput.get(), input.data(),
                         input.size() * sizeof(float), cudaMemcpyHostToDevice),
              "upload MoE base input");
    checkCuda(cudaMemset(deviceEpoch.get(), 0, sizeof(std::uint32_t)),
              "initialize MoE epoch");

    KernelModule kernels;
    nta::FinitePhaseProgram phases(kernels.module());
    cudaStream_t stream = nullptr;
    cudaStream_t overfetchStream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    cudaEvent_t overfetchFork = nullptr;
    cudaEvent_t overfetchReady = nullptr;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "create MoE stream");
    checkCuda(cudaEventCreate(&begin), "create MoE begin event");
    checkCuda(cudaEventCreate(&end), "create MoE end event");
    if (options.policy == Policy::Overfetch) {
      checkCuda(
          cudaStreamCreateWithFlags(&overfetchStream, cudaStreamNonBlocking),
          "create MoE overfetch stream");
      checkCuda(
          cudaEventCreateWithFlags(&overfetchFork, cudaEventDisableTiming),
          "create MoE overfetch fork event");
      checkCuda(
          cudaEventCreateWithFlags(&overfetchReady, cudaEventDisableTiming),
          "create MoE overfetch ready event");
    }
    const CUstream driverStream = reinterpret_cast<CUstream>(stream);
    const CUstream overfetchDriverStream =
        reinterpret_cast<CUstream>(overfetchStream);

    const auto clearOutput = [&] {
      checkCuda(cudaMemsetAsync(deviceOutput.get(), 0,
                                input.size() * sizeof(float), stream),
                "clear MoE output");
    };
    const auto route = [&](bool preacquired) {
      kernels.prepareEpoch(driverStream, deviceEpoch.get(),
                           deviceBaseInput.get(), deviceInput.get(),
                           static_cast<std::uint32_t>(input.size()));
      kernels.route(driverStream, runtime.deviceView(), deviceCatalog.get(),
                    deviceGateWeights.get(), deviceInput.get(),
                    deviceWork.get(), deviceRequirements.get(),
                    deviceSelected.get(), options.tokens, options.experts,
                    options.topK, options.hidden, preacquired);
    };
    const auto initialCompute = [&] {
      kernels.compute(kernels.tile(), driverStream, runtime.deviceView(),
                      deviceWork.get(), deviceRequirements.get(),
                      options.tokens, deviceInput.get(), deviceOutput.get(),
                      options.hidden);
    };
    const auto readyCompute = [&] {
      kernels.compute(kernels.ready(), driverStream, runtime.deviceView(),
                      deviceWork.get(), deviceRequirements.get(),
                      options.tokens, deviceInput.get(), deviceOutput.get(),
                      options.hidden);
    };

    const auto enqueueLateBound = [&] {
      phases.enqueueHost(
          driverStream, runtime.deviceView(),
          {options.experts, options.tokens, intentCapacity, 1},
          [&] {
            clearOutput();
            route(false);
            initialCompute();
          },
          readyCompute);
    };
    const auto enqueueOverfetch = [&] {
      phases.reset(driverStream, runtime.deviceView(), options.experts,
                   options.tokens);
      clearOutput();
      checkCuda(cudaEventRecord(overfetchFork, stream),
                "fork MoE overfetch graph");
      checkCuda(cudaStreamWaitEvent(overfetchStream, overfetchFork),
                "start MoE overfetch copy");
      kernels.copyAll(overfetchDriverStream, deviceCatalog.get(),
                      options.experts);
      checkCuda(cudaEventRecord(overfetchReady, overfetchStream),
                "publish MoE overfetch copy");
      route(true);
      checkCuda(cudaStreamWaitEvent(stream, overfetchReady),
                "join MoE overfetch graph");
      kernels.baseline(driverStream, deviceWork.get(), deviceRequirements.get(),
                       options.tokens, deviceInput.get(), deviceOutput.get(),
                       options.hidden);
    };
    const auto enqueueDirect = [&] {
      phases.reset(driverStream, runtime.deviceView(), options.experts,
                   options.tokens);
      clearOutput();
      route(true);
      kernels.baseline(driverStream, deviceWork.get(), deviceRequirements.get(),
                       options.tokens, deviceInput.get(), deviceOutput.get(),
                       options.hidden);
    };
    const auto runCpuSyncEpoch = [&] {
      phases.reset(driverStream, runtime.deviceView(), options.experts,
                   options.tokens);
      clearOutput();
      route(false);
      checkCuda(cudaMemcpyAsync(cpuVisibleSelection.get(), deviceSelected.get(),
                                selectionCount * sizeof(std::uint32_t),
                                cudaMemcpyDeviceToHost, stream),
                "publish MoE route to CPU");
      checkCuda(cudaStreamSynchronize(stream), "synchronize CPU-visible route");
      initialCompute();
      phases.complete(driverStream, runtime.deviceView(), options.tokens);
      phases.progressHost(driverStream, runtime.deviceView(), intentCapacity);
      phases.publish(driverStream, runtime.deviceView(), options.tokens);
      readyCompute();
      phases.complete(driverStream, runtime.deviceView(), options.tokens);
    };

    if (options.policy != Policy::CpuSync) {
      checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                "begin MoE graph capture");
      if (options.policy == Policy::LateBound) {
        enqueueLateBound();
      } else if (options.policy == Policy::Overfetch) {
        enqueueOverfetch();
      } else {
        enqueueDirect();
      }
      checkCuda(cudaStreamEndCapture(stream, &graph), "end MoE graph capture");
      checkCuda(cudaGraphInstantiate(&graphExec, graph, 0),
                "instantiate MoE graph");
    }

    for (int warmup = 0; warmup < 2; ++warmup) {
      if (options.policy == Policy::CpuSync) {
        runCpuSyncEpoch();
      } else {
        checkCuda(cudaGraphLaunch(graphExec, stream), "launch MoE warmup");
      }
    }
    checkCuda(cudaStreamSynchronize(stream), "synchronize MoE warmup");

    checkCuda(cudaEventRecord(begin, stream), "record MoE begin");
    const auto wallBegin = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0; iteration < options.iterations;
         ++iteration) {
      if (options.policy == Policy::CpuSync) {
        runCpuSyncEpoch();
      } else {
        checkCuda(cudaGraphLaunch(graphExec, stream), "launch MoE graph");
      }
    }
    checkCuda(cudaEventRecord(end, stream), "record MoE end");
    checkCuda(cudaEventSynchronize(end), "synchronize MoE end");
    const auto wallEnd = std::chrono::steady_clock::now();

    float elapsed = 0.0F;
    checkCuda(cudaEventElapsedTime(&elapsed, begin, end), "measure MoE epoch");
    std::vector<float> output(input.size());
    std::vector<std::uint32_t> selected(selectionCount);
    checkCuda(cudaMemcpy(output.data(), deviceOutput.get(),
                         output.size() * sizeof(float), cudaMemcpyDeviceToHost),
              "download MoE output");
    checkCuda(cudaMemcpy(selected.data(), deviceSelected.get(),
                         selected.size() * sizeof(std::uint32_t),
                         cudaMemcpyDeviceToHost),
              "download MoE route");
    checkCuda(cudaMemcpy(input.data(), deviceInput.get(),
                         input.size() * sizeof(float), cudaMemcpyDeviceToHost),
              "download MoE input");

    std::vector<float> expected(input.size(), 0.0F);
    std::uint32_t failures = 0;
    for (std::uint32_t token = 0; token < options.tokens; ++token) {
      if ((options.policy == Policy::LateBound ||
           options.policy == Policy::CpuSync) &&
          runtime.readWorkTicket(token).state !=
              static_cast<std::uint32_t>(nta::abi::WorkTicketState::Done)) {
        ++failures;
      }
      for (std::uint32_t outputIndex = 0; outputIndex < options.hidden;
           ++outputIndex) {
        float mixed = 0.0F;
        for (std::uint32_t dependency = 0; dependency < options.topK;
             ++dependency) {
          const std::uint32_t expert =
              selected[static_cast<std::size_t>(token) * options.topK +
                       dependency];
          if (expert >= options.experts) {
            ++failures;
            continue;
          }
          const float *row =
              weights[expert].data() + outputIndex * options.hidden;
          float expertOutput = 0.0F;
          for (std::uint32_t inputIndex = 0; inputIndex < options.hidden;
               ++inputIndex) {
            expertOutput = std::fma(
                row[inputIndex],
                input[static_cast<std::size_t>(token) * options.hidden +
                      inputIndex],
                expertOutput);
          }
          mixed = std::fma(1.0F / static_cast<float>(dependency + 1U),
                           expertOutput, mixed);
        }
        expected[static_cast<std::size_t>(token) * options.hidden +
                 outputIndex] = mixed;
      }
    }

    float maxError = 0.0F;
    for (std::size_t index = 0; index < output.size(); ++index) {
      const float error = std::abs(output[index] - expected[index]);
      maxError = std::max(maxError, error);
      const float tolerance =
          std::max(2.0e-4F, std::abs(expected[index]) * 2.0e-5F);
      failures += error > tolerance ? 1U : 0U;
    }

    std::vector<bool> unique(options.experts, false);
    for (std::uint32_t expert : selected) {
      if (expert < options.experts) {
        unique[expert] = true;
      }
    }
    std::uint32_t uniqueExperts = 0;
    std::uint32_t selectedStagedExperts = 0;
    std::uint32_t allStagedExperts = 0;
    std::uint64_t stagedIssues = 0;
    for (std::uint32_t expert = 0; expert < options.experts; ++expert) {
      uniqueExperts += unique[expert] ? 1U : 0U;
      const bool staged =
          (catalog[expert].flags & nta::benchmark::MoeExpertStaged) != 0;
      selectedStagedExperts += staged && unique[expert] ? 1U : 0U;
      allStagedExperts += staged ? 1U : 0U;
      stagedIssues += runtime.readObject(expert).issueCount;
    }

    const std::uint64_t selectiveBytes =
        static_cast<std::uint64_t>(selectedStagedExperts) * matrixBytes;
    const std::uint64_t transferredBytes =
        static_cast<std::uint64_t>(options.policy == Policy::Overfetch
                                       ? allStagedExperts
                                       : selectedStagedExperts) *
        matrixBytes;
    const double transferAmplification =
        selectiveBytes == 0
            ? 1.0
            : static_cast<double>(transferredBytes) / selectiveBytes;
    const double epochMilliseconds = elapsed / options.iterations;
    const double wallMilliseconds =
        std::chrono::duration<double, std::milli>(wallEnd - wallBegin).count() /
        options.iterations;
    const double tokensPerSecond =
        options.tokens / (epochMilliseconds / 1000.0);

    const double finalEpochPhysicalTransferMiB =
        static_cast<double>(transferredBytes) / (1024.0 * 1024.0);
    const std::uint32_t cpuRouteSyncs =
        options.policy == Policy::CpuSync ? options.iterations : 0U;
    const std::uint32_t pending = runtime.readPendingCount();
    const std::uint32_t pendingIndex = runtime.readPendingIndexCount();
    if (options.json) {
      std::cout << std::setprecision(9) << "{\"schema\":1,\"workload\":\"moe\","
                << "\"route_visibility\":\"gpu\",\"policy\":\""
                << policyName(options.policy) << "\",\"mode\":\""
                << modeName(options.mode) << "\",\"tokens\":" << options.tokens
                << ",\"experts\":" << options.experts
                << ",\"top_k\":" << options.topK
                << ",\"hidden\":" << options.hidden
                << ",\"epoch_ms\":" << epochMilliseconds
                << ",\"wall_ms\":" << wallMilliseconds
                << ",\"tokens_per_second\":" << tokensPerSecond
                << ",\"selected_unique\":" << uniqueExperts
                << ",\"staged_selected_unique\":" << selectedStagedExperts
                << ",\"final_epoch_physical_transfer_mib\":"
                << finalEpochPhysicalTransferMiB
                << ",\"transfer_amplification\":" << transferAmplification
                << ",\"cpu_route_syncs\":" << cpuRouteSyncs
                << ",\"staged_issues\":" << stagedIssues
                << ",\"pending\":" << pending
                << ",\"pending_index\":" << pendingIndex
                << ",\"max_abs_error\":" << maxError
                << ",\"verification_failures\":" << failures << "}\n";
    } else {
      std::cout << "workload=moe route_visibility=gpu policy="
                << policyName(options.policy)
                << " mode=" << modeName(options.mode)
                << " tokens=" << options.tokens
                << " experts=" << options.experts << " top_k=" << options.topK
                << " hidden=" << options.hidden << " epoch_ms=" << std::fixed
                << std::setprecision(3) << epochMilliseconds
                << " wall_ms=" << wallMilliseconds
                << " tokens/s=" << std::setprecision(2) << tokensPerSecond
                << " selected_unique=" << uniqueExperts
                << " staged_selected_unique=" << selectedStagedExperts
                << " final_epoch_physical_transfer_MiB="
                << finalEpochPhysicalTransferMiB
                << " transfer_amplification=" << transferAmplification
                << " cpu_route_syncs=" << cpuRouteSyncs
                << " staged_issues=" << stagedIssues << " pending=" << pending
                << " pending_index=" << pendingIndex
                << " max_abs_error=" << std::scientific << maxError
                << " verification_failures=" << failures << '\n';
    }

    (void)cudaEventDestroy(end);
    (void)cudaEventDestroy(begin);
    if (graphExec != nullptr) {
      (void)cudaGraphExecDestroy(graphExec);
    }
    if (graph != nullptr) {
      (void)cudaGraphDestroy(graph);
    }
    if (overfetchReady != nullptr) {
      (void)cudaEventDestroy(overfetchReady);
    }
    if (overfetchFork != nullptr) {
      (void)cudaEventDestroy(overfetchFork);
    }
    if (overfetchStream != nullptr) {
      (void)cudaStreamDestroy(overfetchStream);
    }
    (void)cudaStreamDestroy(stream);
    return failures == 0 ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "nta-moe-bench failed: " << error.what() << '\n';
    return 1;
  }
}
