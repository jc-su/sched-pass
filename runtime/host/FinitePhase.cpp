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
      progressNvmeUntilIdle_(load(module, "nta_progress_nvme_until_idle")),
      progressNvmeOrderedUntilIdle_(
          load(module, "nta_progress_nvme_ordered_until_idle")),
      publish_(load(module, "nta_publish_ready")),
      complete_(load(module, "nta_complete_launched")) {}

void FinitePhaseProgram::reset(CUstream stream, abi::RuntimeView *runtime,
                               std::uint32_t objectCount,
                               std::uint32_t workTicketCount) const {
  if (runtime == nullptr || objectCount == 0 || workTicketCount == 0) {
    throw std::invalid_argument(
        "finite phase reset needs runtime objects and work tickets");
  }
  constexpr std::uint32_t Threads = 256;
  const std::uint32_t blocks =
      (std::max(objectCount, workTicketCount) + Threads - 1U) / Threads;
  CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
  void *arguments[] = {&runtimeAddress, &objectCount, &workTicketCount};
  launch(reset_, blocks, Threads, stream, arguments, "nta_reset_epoch");
}

void FinitePhaseProgram::progressNvmeUntilIdle(
    CUstream stream, abi::RuntimeView *runtime, std::uint32_t issueBudget,
    std::uint32_t completionBudget, std::uint64_t timeoutNs) const {
  if (runtime == nullptr || issueBudget == 0 || completionBudget == 0 ||
      timeoutNs == 0) {
    throw std::invalid_argument(
        "NVMe progress-until-idle needs a runtime, budgets, and timeout");
  }
  CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
  void *arguments[] = {&runtimeAddress, &issueBudget, &completionBudget,
                       &timeoutNs};
  launch(progressNvmeUntilIdle_, 1, 32, stream, arguments,
         "nta_progress_nvme_until_idle");
}

void FinitePhaseProgram::progressNvmeOrderedUntilIdle(
    CUstream stream, abi::RuntimeView *runtime, std::uint32_t firstIntent,
    std::uint32_t intentCount, std::uint32_t issueBudget,
    std::uint32_t completionBudget, std::uint64_t timeoutNs) const {
  if (runtime == nullptr || intentCount == 0 || issueBudget == 0 ||
      completionBudget == 0 || timeoutNs == 0 ||
      firstIntent > UINT32_MAX - intentCount) {
    throw std::invalid_argument(
        "ordered NVMe progress needs a bounded intent range, budgets, and "
        "timeout");
  }
  CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
  void *arguments[] = {&runtimeAddress, &firstIntent, &intentCount, &issueBudget,
                       &completionBudget, &timeoutNs};
  launch(progressNvmeOrderedUntilIdle_, 1, 32, stream, arguments,
         "nta_progress_nvme_ordered_until_idle");
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
        "availability publication needs a runtime and pending budget");
  }
  constexpr std::uint32_t Threads = 256;
  CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
  void *arguments[] = {&runtimeAddress, &pendingBudget};
  launch(publish_, 1, Threads, stream, arguments, "nta_publish_ready");
}

void FinitePhaseProgram::complete(CUstream stream, abi::RuntimeView *runtime,
                                  std::uint32_t workTicketCount) const {
  if (runtime == nullptr || workTicketCount == 0) {
    throw std::invalid_argument(
        "completion needs a runtime and work-ticket count");
  }
  constexpr std::uint32_t Threads = 256;
  const std::uint32_t blocks = (workTicketCount + Threads - 1U) / Threads;
  CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
  void *arguments[] = {&runtimeAddress, &workTicketCount};
  launch(complete_, blocks, Threads, stream, arguments,
         "nta_complete_launched");
}

} // namespace nta
