#include "nta/JitPhase.h"

#include <dlfcn.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace nta {
namespace {

using AbiVersion = std::uint32_t (*)();
using Reset = cudaError_t (*)(void *, std::uint32_t, std::uint32_t,
                              cudaStream_t);
using PreloadHost = cudaError_t (*)(void *, std::uint32_t, std::uint32_t,
                                    cudaStream_t);
using ProgressHost = cudaError_t (*)(void *, std::uint32_t, cudaStream_t);
using ProgressNvme = cudaError_t (*)(void *, std::uint32_t, std::uint32_t,
                                     cudaStream_t);
using Publish = cudaError_t (*)(void *, std::uint32_t, cudaStream_t);
using Complete = cudaError_t (*)(void *, std::uint32_t, cudaStream_t);

template <typename Function> Function load(void *library, const char *name) {
  dlerror();
  void *symbol = dlsym(library, name);
  const char *error = dlerror();
  if (error != nullptr || symbol == nullptr) {
    throw std::runtime_error(std::string("cannot load ") + name + ": " +
                             (error == nullptr ? "symbol is null" : error));
  }
  return reinterpret_cast<Function>(symbol);
}

void check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

} // namespace

struct JitPhaseProgram::Impl {
  explicit Impl(std::string_view path) {
    const std::string terminated(path);
    library = dlopen(terminated.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
      const char *error = dlerror();
      throw std::runtime_error(
          std::string("cannot load instrumented JIT module: ") +
          (error == nullptr ? "unknown dynamic-loader error" : error));
    }
    try {
      const AbiVersion version =
          load<AbiVersion>(library, "nta_jit_abi_version");
      if (version() != abi::Version) {
        throw std::runtime_error(
            "instrumented JIT module uses an incompatible NTA ABI");
      }
      reset = load<Reset>(library, "nta_jit_reset_epoch");
      preloadHost = load<PreloadHost>(library, "nta_jit_preload_host");
      progressHost = load<ProgressHost>(library, "nta_jit_progress_host");
      progressNvme = load<ProgressNvme>(library, "nta_jit_progress_nvme");
      publish = load<Publish>(library, "nta_jit_publish_ready");
      complete = load<Complete>(library, "nta_jit_complete_launched");
    } catch (...) {
      dlclose(library);
      library = nullptr;
      throw;
    }
  }

  ~Impl() {
    if (library != nullptr) {
      dlclose(library);
    }
  }

  void *library = nullptr;
  Reset reset = nullptr;
  PreloadHost preloadHost = nullptr;
  ProgressHost progressHost = nullptr;
  ProgressNvme progressNvme = nullptr;
  Publish publish = nullptr;
  Complete complete = nullptr;
};

JitPhaseProgram::JitPhaseProgram(std::string_view sharedObject)
    : impl_(std::make_unique<Impl>(sharedObject)) {}

JitPhaseProgram::~JitPhaseProgram() = default;
JitPhaseProgram::JitPhaseProgram(JitPhaseProgram &&) noexcept = default;
JitPhaseProgram &
JitPhaseProgram::operator=(JitPhaseProgram &&) noexcept = default;

void JitPhaseProgram::reset(cudaStream_t stream, abi::RuntimeView *runtime,
                            std::uint32_t objectCount,
                            std::uint32_t workTicketCount) const {
  if (runtime == nullptr || workTicketCount == 0) {
    throw std::invalid_argument(
        "JIT phase reset needs runtime objects and work tickets");
  }
  check(impl_->reset(runtime, objectCount, workTicketCount, stream),
        "nta_jit_reset_epoch");
}

void JitPhaseProgram::preloadHost(cudaStream_t stream,
                                  abi::RuntimeView *runtime,
                                  std::uint32_t firstObject,
                                  std::uint32_t objectCount) const {
  if (runtime == nullptr || objectCount == 0) {
    throw std::invalid_argument(
        "JIT host preload needs a runtime and non-zero object count");
  }
  check(impl_->preloadHost(runtime, firstObject, objectCount, stream),
        "nta_jit_preload_host");
}

void JitPhaseProgram::progressHost(cudaStream_t stream,
                                   abi::RuntimeView *runtime,
                                   std::uint32_t blocks) const {
  if (runtime == nullptr || blocks == 0) {
    throw std::invalid_argument(
        "JIT host progress needs a runtime and non-zero blocks");
  }
  check(impl_->progressHost(runtime, blocks, stream), "nta_jit_progress_host");
}

void JitPhaseProgram::progressNvme(cudaStream_t stream,
                                   abi::RuntimeView *runtime,
                                   std::uint32_t issueBudget,
                                   std::uint32_t completionBudget) const {
  if (runtime == nullptr || issueBudget == 0 || completionBudget == 0) {
    throw std::invalid_argument(
        "JIT NVMe progress needs a runtime and non-zero budgets");
  }
  check(impl_->progressNvme(runtime, issueBudget, completionBudget, stream),
        "nta_jit_progress_nvme");
}

void JitPhaseProgram::publish(cudaStream_t stream, abi::RuntimeView *runtime,
                              std::uint32_t pendingBudget) const {
  if (runtime == nullptr || pendingBudget == 0) {
    throw std::invalid_argument(
        "JIT publication needs a runtime and pending budget");
  }
  check(impl_->publish(runtime, pendingBudget, stream),
        "nta_jit_publish_ready");
}

void JitPhaseProgram::complete(cudaStream_t stream, abi::RuntimeView *runtime,
                               std::uint32_t workTicketCount) const {
  if (runtime == nullptr || workTicketCount == 0) {
    throw std::invalid_argument(
        "JIT completion needs a runtime and work-ticket count");
  }
  check(impl_->complete(runtime, workTicketCount, stream),
        "nta_jit_complete_launched");
}

} // namespace nta
