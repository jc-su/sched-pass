#include "benchmarks/CommonCuda.h"
#include "benchmarks/kv/KvTypes.h"
#include "nta/FinitePhase.h"
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
#include <sstream>
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

struct Options {
  std::string device;
  std::string reference = "/tmp/nta-nvme-reference.bin";
  std::string output;
  std::string revision;
  int gpu = 0;
  std::uint64_t sourceOffset = 0;
  std::uint32_t bytes = 64U * 1024U;
  std::uint32_t requests = 16;
  std::uint32_t progressPasses = 64;
  std::uint32_t iterations = 20;
  std::uint32_t namespaceId = 1;
  std::uint32_t queueDepth = 64;
  std::uint32_t adminTimeoutMs = 10'000;
  nta::NvmeMediaPolicy mediaPolicy =
      nta::NvmeMediaPolicy::RequireHardwareWriteProtection;
  bool ctaTryIssue = true;
};

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
  if (const char *revision = std::getenv("NTA_REVISION")) {
    options.revision = revision;
  }
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
    } else if (name == "--output") {
      options.output = value;
    } else if (name == "--revision") {
      options.revision = value;
    } else if (name == "--gpu") {
      const std::uint64_t parsed = parseInteger(value, name, true);
      if (parsed >
          static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("--gpu exceeds int");
      }
      options.gpu = static_cast<int>(parsed);
    } else if (name == "--reference") {
      options.reference = value;
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
    } else if (name == "--namespace") {
      const std::uint64_t parsed = parseInteger(value, name, false);
      if (parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("--namespace exceeds uint32_t");
      }
      options.namespaceId = static_cast<std::uint32_t>(parsed);
    } else if (name == "--queue-depth") {
      const std::uint64_t parsed = parseInteger(value, name, false);
      if (parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("--queue-depth exceeds uint32_t");
      }
      options.queueDepth = static_cast<std::uint32_t>(parsed);
    } else if (name == "--admin-timeout-ms") {
      const std::uint64_t parsed = parseInteger(value, name, false);
      if (parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("--admin-timeout-ms exceeds uint32_t");
      }
      options.adminTimeoutMs = static_cast<std::uint32_t>(parsed);
    } else if (name == "--media-policy") {
      if (value == "hardware-write-protect") {
        options.mediaPolicy =
            nta::NvmeMediaPolicy::RequireHardwareWriteProtection;
      } else if (value == "trusted-read-only-code") {
        options.mediaPolicy = nta::NvmeMediaPolicy::TrustReadOnlyDeviceCode;
      } else {
        throw std::invalid_argument("unknown --media-policy value");
      }
    } else if (name == "--cta-try-issue") {
      const std::uint64_t parsed = parseInteger(value, name, true);
      if (parsed > 1) {
        throw std::invalid_argument("--cta-try-issue must be zero or one");
      }
      options.ctaTryIssue = parsed != 0;
    } else {
      throw std::invalid_argument("unknown option " + std::string(name));
    }
  }
  if (options.bytes % sizeof(std::uint32_t) != 0) {
    throw std::invalid_argument("--bytes must be a multiple of four");
  }
  if (!options.device.starts_with("vfio:")) {
    throw std::invalid_argument(
        "--device=vfio:DDDD:BB:SS.F is required for exclusive NVMe ownership");
  }
  if (!options.output.empty() && options.revision.empty()) {
    throw std::invalid_argument(
        "--revision or NTA_REVISION is required for result provenance");
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

std::string jsonString(std::string_view value) {
  std::ostringstream output;
  output << '"';
  for (const char character : value) {
    switch (character) {
    case '"':
      output << "\\\"";
      break;
    case '\\':
      output << "\\\\";
      break;
    case '\n':
      output << "\\n";
      break;
    case '\r':
      output << "\\r";
      break;
    case '\t':
      output << "\\t";
      break;
    default:
      if (static_cast<unsigned char>(character) < 0x20U) {
        output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
               << static_cast<unsigned int>(
                      static_cast<unsigned char>(character))
               << std::dec << std::setfill(' ');
      } else {
        output << character;
      }
    }
  }
  output << '"';
  return output.str();
}

class KernelModule {
public:
  KernelModule() {
    checkDriver(cuModuleLoad(&module_, NTA_KV_CUBIN_PATH),
                "cuModuleLoad instrumented cubin");
    checkDriver(cuModuleGetFunction(&hash_, module_, "nta_nvme_hash_kernel"),
                "cuModuleGetFunction nta_nvme_hash_kernel");
    checkDriver(
        cuModuleGetFunction(&ready_, module_, "nta_nvme_ready_hash_kernel"),
        "cuModuleGetFunction nta_nvme_ready_hash_kernel");
    checkDriver(cuModuleGetFunction(&invalidate_, module_,
                                    "nta_nvme_benchmark_invalidate"),
                "cuModuleGetFunction nta_nvme_benchmark_invalidate");
  }
  ~KernelModule() {
    if (module_ != nullptr) {
      (void)cuModuleUnload(module_);
    }
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

  void invalidate(CUstream stream, nta::abi::RuntimeView *runtime,
                  std::uint32_t objectCount) const {
    CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
    void *arguments[] = {&runtimeAddress, &objectCount};
    const std::uint32_t blocks = (objectCount + 255U) / 256U;
    checkDriver(cuLaunchKernel(invalidate_, blocks, 1, 1, 256, 1, 1, 0,
                               stream, arguments, nullptr),
                "cuLaunchKernel NVMe benchmark invalidation");
  }

  [[nodiscard]] CUmodule module() const noexcept { return module_; }

private:
  CUmodule module_ = nullptr;
  CUfunction hash_ = nullptr;
  CUfunction ready_ = nullptr;
  CUfunction invalidate_ = nullptr;
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
    nta::NvmeTransportOptions transportOptions;
    transportOptions.endpoint = options.device;
    transportOptions.deviceOrdinal = options.gpu;
    transportOptions.namespaceId = options.namespaceId;
    transportOptions.queueDepth = options.queueDepth;
    transportOptions.adminTimeoutMs = options.adminTimeoutMs;
    transportOptions.mediaPolicy = options.mediaPolicy;
    auto transport =
        std::make_shared<nta::NvmeTransport>(std::move(transportOptions));
    // Keep BAR qualification and queue setup ahead of benchmark allocations so
    // a failure cannot leave partially initialized workload state.
    checkCuda(cudaSetDevice(options.gpu), "cudaSetDevice");
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
    nta::RuntimeConfig runtimeConfig{options.requests, options.requests,
                                     options.requests, options.requests};
    runtimeConfig.deviceOrdinal = options.gpu;
    runtimeConfig.enableCtaNvmeTryIssue = options.ctaTryIssue;
    nta::HostRuntime runtime(runtimeConfig,
                             nta::RuntimeBackends{transport, nullptr});
    std::vector<nta::benchmark::TileTask> hostTasks(options.requests);
    for (std::uint32_t request = 0; request < options.requests; ++request) {
      std::unique_ptr<nta::NvmeBuffer> destination =
          transport->allocate(options.bytes);
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
    kernels.invalidate(driverStream, runtime.deviceView(), options.requests);
    phases.enqueueNvme(
        driverStream, runtime.deviceView(),
        {options.requests, options.requests, options.progressPasses, 32, 32},
        [&] {
          checkCuda(cudaMemsetAsync(output.get(), 0,
                                    sizeof(std::uint64_t) * options.requests,
                                    stream),
                    "cudaMemsetAsync output");
          kernels.hash(driverStream, runtime.deviceView(), tasks.get(),
                       options.requests, output.get(), 0);
        },
        [&] {
          kernels.ready(driverStream, runtime.deviceView(), tasks.get(),
                        options.requests, output.get());
        });
    checkCuda(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
    checkCuda(cudaGraphInstantiate(&graphExec, graph, 0),
              "cudaGraphInstantiate");

    checkCuda(cudaGraphLaunch(graphExec, stream), "cudaGraphLaunch warmup");
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize warmup");
    nta::NvmeQueueStats warmupStats = transport->readStats();
    if (warmupStats.error != 0 || warmupStats.outstanding != 0) {
      std::ostringstream message;
      message
          << "bounded warmup did not finish: submitted="
          << warmupStats.submitted << " completed=" << warmupStats.completed
          << " failed=" << warmupStats.failed
          << " outstanding=" << warmupStats.outstanding
          << " error=" << warmupStats.error << " sq_tail=" << warmupStats.sqTail
          << " cq_head=" << warmupStats.cqHead
          << " cq_phase=" << warmupStats.cqPhase << " next_cqe_dw3=0x"
          << std::hex << warmupStats.nextCompletionDword3 << std::dec
          << " direct_submitted=" << warmupStats.directSubmitted
          << " direct_fallbacks=" << warmupStats.directFallbacks
          << "; increase --progress-passes only when outstanding is nonzero";
      throw std::runtime_error(message.str());
    }
    for (std::uint32_t request = 0; request < options.requests; ++request) {
      if (runtime.readWorkTicket(request).state !=
          static_cast<std::uint32_t>(nta::abi::WorkTicketState::Done)) {
        throw std::runtime_error(
            "bounded warmup did not resume every workTicket");
      }
    }

    const nta::NvmeQueueStats measurementStart = transport->readStats();
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
    const std::uint64_t measuredSubmitted =
        stats.submitted - measurementStart.submitted;
    const std::uint64_t measuredCompleted =
        stats.completed - measurementStart.completed;
    const std::uint64_t measuredDirectSubmitted =
        stats.directSubmitted - measurementStart.directSubmitted;
    const std::uint64_t measuredDirectFallbacks =
        stats.directFallbacks - measurementStart.directFallbacks;
    const std::uint64_t expectedCommands =
        static_cast<std::uint64_t>(options.requests) * options.iterations;
    std::uint32_t verificationFailures = 0;
    for (std::uint32_t request = 0; request < options.requests; ++request) {
      const nta::abi::WorkTicket workTicket =
          runtime.readWorkTicket(request);
      if (actual[request] != expected[request] ||
          workTicket.state !=
              static_cast<std::uint32_t>(nta::abi::WorkTicketState::Done)) {
        ++verificationFailures;
      }
    }
    if (stats.error != 0 || stats.outstanding != 0 || stats.failed != 0) {
      ++verificationFailures;
    }
    if (measuredSubmitted != expectedCommands ||
        measuredCompleted != expectedCommands) {
      ++verificationFailures;
    }
    const bool verified = verificationFailures == 0;
    const double milliseconds = elapsedMilliseconds / options.iterations;
    const double mibPerSecond = (static_cast<double>(options.bytes) *
                                 options.requests / (1024.0 * 1024.0)) /
                                (milliseconds / 1000.0);

    std::cout << "destination=host-mapped backend=vfio-iommufd"
              << " media_policy="
              << (options.mediaPolicy ==
                          nta::NvmeMediaPolicy::TrustReadOnlyDeviceCode
                      ? "trusted-read-only-code"
                      : "hardware-write-protect")
              << " requests=" << options.requests << " bytes=" << options.bytes
              << " cta_try_issue=" << (options.ctaTryIssue ? 1 : 0)
              << " progress_passes=" << options.progressPasses
              << " graph_ms=" << std::fixed << std::setprecision(3)
              << milliseconds << " MiB/s=" << std::setprecision(2)
              << mibPerSecond << " measured_submitted=" << measuredSubmitted
              << " measured_completed=" << measuredCompleted
              << " measured_direct_submitted=" << measuredDirectSubmitted
              << " measured_direct_fallbacks=" << measuredDirectFallbacks
              << " submitted=" << stats.submitted
              << " direct_submitted=" << stats.directSubmitted
              << " direct_fallbacks=" << stats.directFallbacks
              << " completed=" << stats.completed << " failed=" << stats.failed
              << " verification_failures=" << verificationFailures << '\n';

    if (!options.output.empty()) {
      std::ofstream artifact(options.output, std::ios::trunc);
      if (!artifact) {
        throw std::runtime_error("cannot open result file " + options.output);
      }
      artifact << "{\n"
               << "  \"schema\": 1,\n"
               << "  \"classification\": \"nta-vfio-nvme-read\",\n"
               << "  \"revision\": " << jsonString(options.revision) << ",\n"
               << "  \"device\": " << jsonString(options.device) << ",\n"
               << "  \"gpu\": " << options.gpu << ",\n"
               << "  \"runtime_abi\": " << nta::abi::Version << ",\n"
               << "  \"namespace_id\": " << options.namespaceId << ",\n"
               << "  \"namespace_bytes\": " << capabilities.namespaceBytes
               << ",\n"
               << "  \"lba_size\": " << capabilities.lbaSize << ",\n"
               << "  \"max_transfer_bytes\": "
               << capabilities.maxTransferBytes << ",\n"
               << "  \"queue_depth\": " << capabilities.queueDepth << ",\n"
               << "  \"queue_count\": " << capabilities.queueCount << ",\n"
               << "  \"translated_iommu\": "
               << (capabilities.translatedIommu ? "true" : "false") << ",\n"
               << "  \"gpu_doorbell_mapping_validated\": "
               << (capabilities.gpuDoorbellMappingValidated ? "true"
                                                            : "false")
               << ",\n"
               << "  \"media_policy\": "
               << jsonString(options.mediaPolicy ==
                                     nta::NvmeMediaPolicy::TrustReadOnlyDeviceCode
                                 ? "trusted-read-only-code"
                                 : "hardware-write-protect")
               << ",\n"
               << "  \"destination\": \"host-mapped\",\n"
               << "  \"requests\": " << options.requests << ",\n"
               << "  \"bytes_per_request\": " << options.bytes << ",\n"
               << "  \"iterations\": " << options.iterations << ",\n"
               << "  \"progress_passes\": " << options.progressPasses
               << ",\n"
               << "  \"cta_try_issue\": "
               << (options.ctaTryIssue ? "true" : "false") << ",\n"
               << "  \"graph_ms\": " << std::setprecision(9) << milliseconds
               << ",\n"
               << "  \"logical_mib_per_second\": " << mibPerSecond << ",\n"
               << "  \"physical_mib_per_second\": " << mibPerSecond << ",\n"
               << "  \"expected_commands\": " << expectedCommands << ",\n"
               << "  \"measured_submitted\": " << measuredSubmitted
               << ",\n"
               << "  \"measured_completed\": " << measuredCompleted
               << ",\n"
               << "  \"measured_direct_submitted\": "
               << measuredDirectSubmitted << ",\n"
               << "  \"measured_direct_fallbacks\": "
               << measuredDirectFallbacks << ",\n"
               << "  \"submitted\": " << stats.submitted << ",\n"
               << "  \"direct_submitted\": " << stats.directSubmitted
               << ",\n"
               << "  \"direct_fallbacks\": " << stats.directFallbacks
               << ",\n"
               << "  \"completed\": " << stats.completed << ",\n"
               << "  \"failed\": " << stats.failed << ",\n"
               << "  \"outstanding\": " << stats.outstanding << ",\n"
               << "  \"verification_failures\": " << verificationFailures
               << ",\n"
               << "  \"verified\": " << (verified ? "true" : "false")
               << "\n}\n";
      if (!artifact) {
        throw std::runtime_error("cannot write result file " + options.output);
      }
    }

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
