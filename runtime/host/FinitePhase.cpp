#include "nta/FinitePhase.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace nta {
namespace {

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

CUfunction load(CUmodule module, const char *name) {
  if (module == nullptr) {
    throw std::invalid_argument("finite phase program needs a CUDA module");
  }
  CUfunction function = nullptr;
  checkDriver(cuModuleGetFunction(&function, module, name), name);
  return function;
}

void launch(CUfunction function, std::uint32_t blocks, std::uint32_t threads,
            CUstream stream, void **arguments, const char *operation) {
  if (blocks == 0 || threads == 0) {
    throw std::invalid_argument(
        "finite phase launch dimensions must be non-zero");
  }
  checkDriver(cuLaunchKernel(function, blocks, 1, 1, threads, 1, 1, 0, stream,
                             arguments, nullptr),
              operation);
}

} // namespace

FinitePhaseProgram::FinitePhaseProgram(CUmodule module)
    : reset_(load(module, "nta_reset_epoch")),
      progressHost_(load(module, "nta_progress_host_staging")),
      progressNvme_(load(module, "nta_progress_nvme")),
      publish_(load(module, "nta_publish_ready")) {}

void FinitePhaseProgram::reset(CUstream stream, abi::RuntimeView *runtime,
                               std::uint32_t objectCount,
                               std::uint32_t continuationCount) const {
  if (runtime == nullptr || objectCount == 0 || continuationCount == 0) {
    throw std::invalid_argument(
        "finite phase reset needs runtime objects and continuations");
  }
  constexpr std::uint32_t Threads = 256;
  const std::uint32_t blocks =
      (std::max(objectCount, continuationCount) + Threads - 1U) / Threads;
  CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
  void *arguments[] = {&runtimeAddress, &objectCount, &continuationCount};
  launch(reset_, blocks, Threads, stream, arguments, "nta_reset_epoch");
}

void FinitePhaseProgram::progressHost(CUstream stream,
                                      abi::RuntimeView *runtime,
                                      std::uint32_t blocks) const {
  if (runtime == nullptr) {
    throw std::invalid_argument("host progress needs a runtime");
  }
  CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
  void *arguments[] = {&runtimeAddress};
  launch(progressHost_, blocks, 256, stream, arguments,
         "nta_progress_host_staging");
}

void FinitePhaseProgram::progressNvme(CUstream stream,
                                      abi::RuntimeView *runtime,
                                      std::uint32_t issueBudget,
                                      std::uint32_t completionBudget) const {
  if (runtime == nullptr || issueBudget == 0 || completionBudget == 0) {
    throw std::invalid_argument(
        "NVMe progress needs a runtime and non-zero budgets");
  }
  CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
  void *arguments[] = {&runtimeAddress, &issueBudget, &completionBudget};
  launch(progressNvme_, 1, 32, stream, arguments, "nta_progress_nvme");
}

void FinitePhaseProgram::publish(CUstream stream, abi::RuntimeView *runtime,
                                 std::uint32_t pendingBudget) const {
  if (runtime == nullptr || pendingBudget == 0) {
    throw std::invalid_argument(
        "readiness publication needs a runtime and pending budget");
  }
  constexpr std::uint32_t Threads = 256;
  const std::uint32_t blocks =
      std::min(32U, (pendingBudget + Threads - 1U) / Threads);
  CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
  void *arguments[] = {&runtimeAddress, &pendingBudget};
  launch(publish_, blocks, Threads, stream, arguments, "nta_publish_ready");
}

} // namespace nta
