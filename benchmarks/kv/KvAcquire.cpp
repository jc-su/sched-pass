#include "benchmarks/CommonCuda.h"
#include "benchmarks/kv/KvTypes.h"
#include "nta/DeviceWorkPlan.h"
#include "nta/FinitePhase.h"
#include "nta/HostRuntime.h"
#include "nta/JitPhase.h"
#include "nta/WorkPlan.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef NTA_KV_CUBIN_PATH
#error "NTA_KV_CUBIN_PATH must identify the instrumented device image"
#endif

namespace {

using nta::benchmark::checkCuda;
using nta::benchmark::checkDriver;
using nta::benchmark::DeviceBuffer;

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
  std::uint32_t objectStaleStride = 0;
  std::uint32_t coalesce = 1;
  std::uint32_t dependencies = 1;
  std::uint32_t lifecycleEpochs = 1;
  bool baseline = false;
  bool externalRegistration = false;
  std::uint32_t externalOffset = 0;
};

class RegisteredObjectBuffer {
public:
  RegisteredObjectBuffer() = default;
  ~RegisteredObjectBuffer() { release(); }
  RegisteredObjectBuffer(const RegisteredObjectBuffer &) = delete;
  RegisteredObjectBuffer &operator=(const RegisteredObjectBuffer &) = delete;
  RegisteredObjectBuffer(RegisteredObjectBuffer &&other) noexcept
      : sourceAllocation_(std::exchange(other.sourceAllocation_, nullptr)),
        sourceDevice_(std::exchange(other.sourceDevice_, nullptr)),
        stagingDevice_(std::exchange(other.stagingDevice_, nullptr)),
        hostSource_(std::exchange(other.hostSource_, false)) {}
  RegisteredObjectBuffer &operator=(RegisteredObjectBuffer &&other) noexcept {
    if (this != &other) {
      release();
      sourceAllocation_ = std::exchange(other.sourceAllocation_, nullptr);
      sourceDevice_ = std::exchange(other.sourceDevice_, nullptr);
      stagingDevice_ = std::exchange(other.stagingDevice_, nullptr);
      hostSource_ = std::exchange(other.hostSource_, false);
    }
    return *this;
  }

  void initialize(std::span<const std::byte> contents, nta::Placement placement,
                  std::uint32_t sourceOffset) {
    if (sourceAllocation_ != nullptr || contents.empty()) {
      throw std::logic_error(
          "registered object buffer initialization is invalid");
    }
    const std::size_t allocationBytes = contents.size() + sourceOffset;
    try {
      if (placement == nta::Placement::Hbm) {
        checkCuda(cudaMalloc(&sourceAllocation_, allocationBytes),
                  "cudaMalloc registered HBM source");
        sourceDevice_ =
            static_cast<std::byte *>(sourceAllocation_) + sourceOffset;
        checkCuda(cudaMemcpy(sourceDevice_, contents.data(), contents.size(),
                             cudaMemcpyHostToDevice),
                  "upload registered HBM source");
      } else {
        hostSource_ = true;
        checkCuda(cudaHostAlloc(&sourceAllocation_, allocationBytes,
                                cudaHostAllocMapped),
                  "cudaHostAlloc registered DRAM source");
        std::memcpy(static_cast<std::byte *>(sourceAllocation_) + sourceOffset,
                    contents.data(), contents.size());
        void *mappedBase = nullptr;
        checkCuda(cudaHostGetDevicePointer(&mappedBase, sourceAllocation_, 0),
                  "cudaHostGetDevicePointer registered DRAM source");
        sourceDevice_ = static_cast<std::byte *>(mappedBase) + sourceOffset;
      }
      if (placement == nta::Placement::HostStaged) {
        checkCuda(cudaMalloc(&stagingDevice_, contents.size()),
                  "cudaMalloc registered DRAM staging");
      }
    } catch (...) {
      release();
      throw;
    }
  }

  [[nodiscard]] const void *sourceDevice() const noexcept {
    return sourceDevice_;
  }
  [[nodiscard]] void *stagingDevice() const noexcept { return stagingDevice_; }

private:
  void release() noexcept {
    if (stagingDevice_ != nullptr) {
      (void)cudaFree(stagingDevice_);
      stagingDevice_ = nullptr;
    }
    if (sourceAllocation_ != nullptr) {
      if (hostSource_) {
        (void)cudaFreeHost(sourceAllocation_);
      } else {
        (void)cudaFree(sourceAllocation_);
      }
      sourceAllocation_ = nullptr;
    }
    sourceDevice_ = nullptr;
    hostSource_ = false;
  }

  void *sourceAllocation_ = nullptr;
  void *sourceDevice_ = nullptr;
  void *stagingDevice_ = nullptr;
  bool hostSource_ = false;
};

class KernelModule {
public:
  KernelModule() {
    checkDriver(cuModuleLoad(&module_, NTA_KV_CUBIN_PATH),
                "cuModuleLoad instrumented cubin");
    checkDriver(cuModuleGetFunction(&dependencyReady_, module_,
                                    "nta_dependency_ready_kernel"),
                "cuModuleGetFunction nta_dependency_ready_kernel");
    checkDriver(cuModuleGetFunction(&dependencyBaseline_, module_,
                                    "nta_dependency_baseline_kernel"),
                "cuModuleGetFunction nta_dependency_baseline_kernel");
  }

  ~KernelModule() {
    if (module_ != nullptr) {
      (void)cuModuleUnload(module_);
    }
  }
  KernelModule(const KernelModule &) = delete;
  KernelModule &operator=(const KernelModule &) = delete;

  void launchDependencyReady(CUstream stream, nta::abi::RuntimeView *runtime,
                             const nta::abi::WorkItem *tasks,
                             std::uint32_t taskCount,
                             const nta::abi::AcquireRequirement *requirements,
                             const float *query, float *output) const {
    CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
    CUdeviceptr taskAddress = reinterpret_cast<CUdeviceptr>(tasks);
    CUdeviceptr requirementAddress =
        reinterpret_cast<CUdeviceptr>(requirements);
    CUdeviceptr queryAddress = reinterpret_cast<CUdeviceptr>(query);
    CUdeviceptr outputAddress = reinterpret_cast<CUdeviceptr>(output);
    void *arguments[] = {&runtimeAddress,     &taskAddress,  &taskCount,
                         &requirementAddress, &queryAddress, &outputAddress};
    checkDriver(cuLaunchKernel(dependencyReady_, taskCount, 1, 1, 256, 1, 1, 0,
                               stream, arguments, nullptr),
                "cuLaunchKernel dependency ready");
  }

  void
  launchDependencyBaseline(CUstream stream, const nta::abi::WorkItem *tasks,
                           std::uint32_t taskCount,
                           const nta::abi::AcquireRequirement *requirements,
                           const float *query, float *output) const {
    CUdeviceptr taskAddress = reinterpret_cast<CUdeviceptr>(tasks);
    CUdeviceptr requirementAddress =
        reinterpret_cast<CUdeviceptr>(requirements);
    CUdeviceptr queryAddress = reinterpret_cast<CUdeviceptr>(query);
    CUdeviceptr outputAddress = reinterpret_cast<CUdeviceptr>(output);
    void *arguments[] = {&taskAddress, &taskCount, &requirementAddress,
                         &queryAddress, &outputAddress};
    checkDriver(cuLaunchKernel(dependencyBaseline_, taskCount, 1, 1, 256, 1, 1,
                               0, stream, arguments, nullptr),
                "cuLaunchKernel dependency baseline");
  }

  [[nodiscard]] CUmodule module() const noexcept { return module_; }

private:
  CUmodule module_ = nullptr;
  CUfunction dependencyReady_ = nullptr;
  CUfunction dependencyBaseline_ = nullptr;
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
    } else if (name == "--object-stale-stride") {
      options.objectStaleStride = value == "0" ? 0 : parsePositive(value, name);
    } else if (name == "--coalesce") {
      options.coalesce = parsePositive(value, name);
    } else if (name == "--dependencies") {
      options.dependencies = parsePositive(value, name);
    } else if (name == "--lifecycle-epochs") {
      options.lifecycleEpochs = parsePositive(value, name);
    } else if (name == "--baseline") {
      if (value != "0" && value != "1") {
        throw std::invalid_argument("--baseline must be 0 or 1");
      }
      options.baseline = value == "1";
    } else if (name == "--external-registration") {
      if (value != "0" && value != "1") {
        throw std::invalid_argument("--external-registration must be 0 or 1");
      }
      options.externalRegistration = value == "1";
    } else if (name == "--external-offset") {
      options.externalOffset = value == "0" ? 0 : parsePositive(value, name);
    } else {
      throw std::invalid_argument("unknown option " + std::string(name));
    }
  }
  if (options.tileBytes % 16 != 0) {
    throw std::invalid_argument(
        "--tile-bytes must be a positive multiple of 16");
  }
  if (options.dependencies > 32) {
    throw std::invalid_argument("--dependencies must not exceed 32");
  }
  if (options.baseline &&
      (options.mode == Mode::HostStaged || options.mode == Mode::Mixed ||
       options.cancelStride != 0 || options.staleStride != 0 ||
       options.objectStaleStride != 0)) {
    throw std::invalid_argument(
        "--baseline requires a direct placement and live request bindings");
  }
  if (options.objectStaleStride != 0 && options.mode != Mode::HostStaged) {
    throw std::invalid_argument(
        "--object-stale-stride requires --mode=host-staged");
  }
  if (options.externalOffset >= 16) {
    throw std::invalid_argument("--external-offset must be below 16 bytes");
  }
  if (options.externalOffset != 0 && !options.externalRegistration) {
    throw std::invalid_argument(
        "--external-offset requires --external-registration=1");
  }
  if (options.externalOffset != 0 && options.mode != Mode::HostStaged) {
    throw std::invalid_argument(
        "--external-offset requires --mode=host-staged");
  }
  const std::uint64_t largestGeneration =
      100ULL +
      static_cast<std::uint64_t>(options.lifecycleEpochs - 1U) *
          (static_cast<std::uint64_t>(options.requests) + 1ULL) +
      options.requests;
  if (largestGeneration > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("lifecycle generations exceed the NTA ABI");
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
    const std::uint32_t objectGroups =
        options.requests / options.coalesce +
        (options.requests % options.coalesce != 0 ? 1U : 0U);
    if (objectGroups >
        std::numeric_limits<std::uint32_t>::max() / options.dependencies) {
      throw std::overflow_error("object count exceeds the NTA ABI");
    }
    const std::uint32_t objectCount = objectGroups * options.dependencies;
    std::vector<float> query(elements);
    for (std::uint32_t i = 0; i < elements; ++i) {
      query[i] = 0.25F + static_cast<float>((i * 7U) % 29U) / 31.0F;
    }

    std::vector<RegisteredObjectBuffer> registeredObjects;
    registeredObjects.reserve(options.externalRegistration ? objectCount : 0);
    nta::HostRuntime runtime({options.requests, objectCount, objectCount,
                              options.requests, 1, options.dependencies});
    std::vector<nta::abi::AcquireRequirement> requirements(
        static_cast<std::size_t>(options.requests) * options.dependencies);
    nta::WorkPlanBuilder planBuilder(options.dependencies);
    std::vector<std::vector<float>> objectData(objectCount);
    std::vector<nta::ObjectHandle> objects(objectCount);
    std::vector<float> expected(options.requests, 0.0F);
    std::vector<bool> cancelled(options.requests, false);
    std::vector<bool> stale(options.requests, false);
    std::vector<bool> objectStale(options.requests, false);

    for (std::uint32_t object = 0; object < objectCount; ++object) {
      objectData[object].resize(elements);
      for (std::uint32_t element = 0; element < elements; ++element) {
        objectData[object][element] =
            0.5F +
            static_cast<float>((object * 13U + element * 5U) % 97U) / 101.0F;
      }
      const std::span<const float> floats(objectData[object]);
      const std::span<const std::byte> contents = std::as_bytes(floats);
      const nta::Placement placement = placementFor(options.mode, object);
      if (options.externalRegistration) {
        RegisteredObjectBuffer allocation;
        allocation.initialize(contents, placement, options.externalOffset);
        const nta::RegisteredReplicaSpec replica{allocation.sourceDevice(),
                                                 placement};
        objects[object] = runtime.registerObject(
            object, 200000U + object, 1, contents.size(),
            allocation.stagingDevice(),
            std::span<const nta::RegisteredReplicaSpec>(&replica, 1));
        registeredObjects.push_back(std::move(allocation));
      } else {
        objects[object] = runtime.installObject(object, 200000U + object, 1,
                                                contents, placement);
      }
    }

    for (std::uint32_t task = 0; task < options.requests; ++task) {
      const std::uint32_t generation = 100U + task;
      runtime.setRequest(task, 100000U + task, generation,
                         (task / options.coalesce) % 4, task % 3);
      if (options.cancelStride != 0 && task % options.cancelStride == 0) {
        runtime.cancelRequest(task, generation);
        cancelled[task] = true;
      }
      const std::uint32_t taskGeneration =
          options.staleStride != 0 && task % options.staleStride == 0
              ? generation - 1
              : generation;
      stale[task] = taskGeneration != generation;

      const std::uint32_t objectBegin =
          (task / options.coalesce) * options.dependencies;
      objectStale[task] = options.objectStaleStride != 0 &&
                          task % options.objectStaleStride == 0;
      double reference = 0.0;
      for (std::uint32_t dependency = 0; dependency < options.dependencies;
           ++dependency) {
        const std::uint32_t object = objectBegin + dependency;
        for (std::uint32_t element = 0; element < elements; ++element) {
          const float value = objectData[object][element];
          reference += static_cast<double>(value) * query[element] *
                       static_cast<double>(dependency + 1U);
        }
        requirements[static_cast<std::size_t>(task) * options.dependencies +
                     dependency] = {
            reinterpret_cast<std::uint64_t>(objects[object].directDeviceBase),
            0,
            200000U + object,
            0,
            object,
            objectStale[task] && dependency == 0 ? 2U : 1U,
            options.tileBytes,
            0,
        };
      }
      expected[task] = static_cast<float>(reference);

      const std::uint32_t request =
          planBuilder.addRequest({task, taskGeneration});
      const std::span<const nta::abi::AcquireRequirement> taskRequirements(
          requirements.data() +
              static_cast<std::size_t>(task) * options.dependencies,
          options.dependencies);
      (void)planBuilder.addWork(request, task, taskRequirements);
    }

    nta::WorkPlan workPlan = planBuilder.finish();
    nta::DeviceWorkPlan deviceWorkPlan = runtime.uploadWorkPlan(workPlan);

    DeviceBuffer<float> deviceQuery(query.size());
    DeviceBuffer<float> deviceOutput(options.requests);
    checkCuda(cudaMemcpy(deviceQuery.get(), query.data(),
                         sizeof(float) * query.size(), cudaMemcpyHostToDevice),
              "upload query");

    KernelModule kernels;
    nta::FinitePhaseProgram phases(kernels.module());
    nta::JitPhaseProgram typedPhases(NTA_TRANSPORT_PROGRAM_PATH);
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
    phases.enqueueHost(
        driverStream, runtime.deviceView(),
        {objectCount, options.requests, objectCount, 1},
        [&] {
          checkCuda(cudaMemsetAsync(deviceOutput.get(), 0,
                                    sizeof(float) * options.requests, stream),
                    "cudaMemsetAsync output");
          if (options.baseline) {
            kernels.launchDependencyBaseline(
                driverStream, deviceWorkPlan.workItems(), options.requests,
                deviceWorkPlan.dependencies(), deviceQuery.get(),
                deviceOutput.get());
          } else {
            typedPhases.discover(
                stream, runtime.deviceView(), deviceWorkPlan.workItems(),
                deviceWorkPlan.dependencies(), options.requests);
          }
        },
        [&] {
          if (!options.baseline) {
            kernels.launchDependencyReady(
                driverStream, runtime.deviceView(), deviceWorkPlan.workItems(),
                options.requests, deviceWorkPlan.dependencies(),
                deviceQuery.get(), deviceOutput.get());
          }
        });
    checkCuda(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
    checkCuda(cudaGraphInstantiate(&graphExec, graph, 0),
              "cudaGraphInstantiate");

    for (int warmup = 0; warmup < 3; ++warmup) {
      checkCuda(cudaGraphLaunch(graphExec, stream), "cudaGraphLaunch warmup");
    }
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize warmup");

    std::vector<float> output(options.requests);
    std::uint32_t failures = 0;
    const auto prepareLifecycleEpoch = [&](std::uint32_t epoch) {
      for (std::uint32_t task = 0; task < options.requests; ++task) {
        const std::uint32_t generation =
            100U + epoch * (options.requests + 1U) + task;
        runtime.setRequest(
            task,
            100000ULL + static_cast<std::uint64_t>(epoch) * options.requests +
                task,
            generation, (task / options.coalesce) % 4, task % 3);
        cancelled[task] = options.cancelStride != 0 &&
                          (task + epoch) % options.cancelStride == 0;
        if (cancelled[task]) {
          runtime.cancelRequest(task, generation);
        }
        const std::uint32_t taskGeneration =
            options.staleStride != 0 &&
                    (task + epoch) % options.staleStride == 0
                ? generation - 1U
                : generation;
        stale[task] = taskGeneration != generation;
        objectStale[task] = options.objectStaleStride != 0 &&
                            (task + epoch) % options.objectStaleStride == 0;
        workPlan.requests[task].generation = taskGeneration;
        workPlan.workItems[task].generation = taskGeneration;
        for (std::uint32_t dependency = 0; dependency < options.dependencies;
             ++dependency) {
          const std::size_t index =
              static_cast<std::size_t>(task) * options.dependencies +
              dependency;
          requirements[index].objectVersion =
              objectStale[task] && dependency == 0 ? 2U : 1U;
          workPlan.dependencies[index].objectVersion =
              requirements[index].objectVersion;
        }
      }
      deviceWorkPlan.uploadAsync(workPlan, stream);
    };

    const auto verifyEpoch = [&] {
      checkCuda(cudaMemcpy(output.data(), deviceOutput.get(),
                           sizeof(float) * output.size(),
                           cudaMemcpyDeviceToHost),
                "download lifecycle output");
      for (std::uint32_t task = 0; task < options.requests; ++task) {
        const nta::abi::WorkTicket workTicket =
            runtime.readWorkTicket(task);
        if (options.baseline) {
          const float tolerance =
              std::max(0.02F, std::abs(expected[task]) * 2.0e-5F);
          if (std::abs(output[task] - expected[task]) > tolerance) {
            std::cerr << "baseline mismatch task=" << task
                      << " output=" << output[task]
                      << " expected=" << expected[task] << '\n';
            ++failures;
          }
        } else if (cancelled[task] || stale[task]) {
          if (workTicket.state !=
                  static_cast<std::uint32_t>(
                      nta::abi::WorkTicketState::Cancelled) ||
              output[task] != 0.0F) {
            std::cerr << "cancel/stale mismatch task=" << task
                      << " state=" << workTicket.state
                      << " output=" << output[task] << '\n';
            ++failures;
          }
        } else if (objectStale[task]) {
          if (workTicket.state != static_cast<std::uint32_t>(
                                        nta::abi::WorkTicketState::Failed) ||
              output[task] != 0.0F) {
            std::cerr << "object-stale mismatch task=" << task
                      << " state=" << workTicket.state
                      << " output=" << output[task] << '\n';
            ++failures;
          }
        } else {
          const float tolerance =
              std::max(0.02F, std::abs(expected[task]) * 2.0e-5F);
          if (workTicket.state != static_cast<std::uint32_t>(
                                        nta::abi::WorkTicketState::Done) ||
              std::abs(output[task] - expected[task]) > tolerance) {
            std::cerr << "ready mismatch task=" << task
                      << " state=" << workTicket.state
                      << " logical=" << workTicket.logicalTile
                      << " output=" << output[task]
                      << " expected=" << expected[task] << '\n';
            ++failures;
          }
        }
      }
      if (runtime.readPendingCount() != 0) {
        ++failures;
      }
    };

    checkCuda(cudaEventRecord(begin, stream), "cudaEventRecord begin");
    for (std::uint32_t epoch = 0; epoch < options.lifecycleEpochs; ++epoch) {
      if (epoch != 0) {
        prepareLifecycleEpoch(epoch);
      }
      for (std::uint32_t iteration = 0; iteration < options.iterations;
           ++iteration) {
        checkCuda(cudaGraphLaunch(graphExec, stream),
                  "cudaGraphLaunch measured");
      }
      if (epoch + 1U == options.lifecycleEpochs) {
        checkCuda(cudaEventRecord(end, stream), "cudaEventRecord end");
      }
      checkCuda(cudaStreamSynchronize(stream),
                "cudaStreamSynchronize lifecycle epoch");
      verifyEpoch();
    }

    float elapsedMilliseconds = 0.0F;
    checkCuda(cudaEventElapsedTime(&elapsedMilliseconds, begin, end),
              "cudaEventElapsedTime");

    std::uint64_t stagedIssues = 0;
    for (std::uint32_t object = 0; object < objectCount; ++object) {
      stagedIssues += runtime.readObject(object).issueCount;
    }

    const std::uint64_t graphLaunches =
        static_cast<std::uint64_t>(options.lifecycleEpochs) *
        options.iterations;
    const double millisecondsPerBatch =
        elapsedMilliseconds / static_cast<double>(graphLaunches);
    const double gibPerSecond =
        (static_cast<double>(options.requests) * options.tileBytes *
         options.dependencies / (1024.0 * 1024.0 * 1024.0)) /
        (millisecondsPerBatch / 1000.0);

    std::cout << "device=" << properties.name
              << " mode=" << modeName(options.mode)
              << " requests=" << options.requests << " objects=" << objectCount
              << " coalesce=" << options.coalesce
              << " dependencies=" << options.dependencies
              << " lifecycle_epochs=" << options.lifecycleEpochs
              << " graph_launches=" << graphLaunches
              << " baseline=" << (options.baseline ? 1 : 0)
              << " external_registration="
              << (options.externalRegistration ? 1 : 0)
              << " external_offset=" << options.externalOffset
              << " tile_bytes=" << options.tileBytes
              << " graph_ms=" << std::fixed << std::setprecision(3)
              << millisecondsPerBatch
              << " logical_GiB/s=" << std::setprecision(2) << gibPerSecond
              << " staged_issues=" << stagedIssues << " cancelled="
              << std::count(cancelled.begin(), cancelled.end(), true)
              << " stale=" << std::count(stale.begin(), stale.end(), true)
              << " object_stale="
              << std::count(objectStale.begin(), objectStale.end(), true)
              << " pending=" << runtime.readPendingCount()
              << " pending_index=" << runtime.readPendingIndexCount()
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
