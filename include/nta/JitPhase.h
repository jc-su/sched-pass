#pragma once

#include "nta/FinitePhase.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <string_view>

namespace nta {

// Owning host adapter for phase launchers exported by an instrumented JIT
// shared object. Loading the same object as the kernel framework is safe:
// dlopen retains the existing mapping and this class holds one reference.
class JitPhaseProgram {
public:
  explicit JitPhaseProgram(std::string_view sharedObject);
  ~JitPhaseProgram();

  JitPhaseProgram(const JitPhaseProgram &) = delete;
  JitPhaseProgram &operator=(const JitPhaseProgram &) = delete;
  JitPhaseProgram(JitPhaseProgram &&) noexcept;
  JitPhaseProgram &operator=(JitPhaseProgram &&) noexcept;

  void reset(cudaStream_t stream, abi::RuntimeView *runtime,
             std::uint32_t objectCount, std::uint32_t continuationCount) const;
  void progressHost(cudaStream_t stream, abi::RuntimeView *runtime,
                    std::uint32_t blocks) const;
  void progressNvme(cudaStream_t stream, abi::RuntimeView *runtime,
                    std::uint32_t issueBudget,
                    std::uint32_t completionBudget) const;
  void publish(cudaStream_t stream, abi::RuntimeView *runtime,
               std::uint32_t pendingBudget) const;
  void complete(cudaStream_t stream, abi::RuntimeView *runtime,
                std::uint32_t continuationCount) const;

  template <typename Initial, typename Ready>
  void enqueueHost(cudaStream_t stream, abi::RuntimeView *runtime,
                   const HostPhaseConfig &config, Initial &&initial,
                   Ready &&ready) const {
    reset(stream, runtime, config.objectCount, config.continuationCount);
    initial();
    complete(stream, runtime, config.continuationCount);
    for (std::uint32_t pass = 0; pass < config.progressPasses; ++pass) {
      progressHost(stream, runtime, config.progressBlocks);
      publish(stream, runtime, config.continuationCount);
      ready();
      complete(stream, runtime, config.continuationCount);
    }
  }

  template <typename Initial, typename Ready>
  void enqueueNvme(cudaStream_t stream, abi::RuntimeView *runtime,
                   const NvmePhaseConfig &config, Initial &&initial,
                   Ready &&ready) const {
    reset(stream, runtime, config.objectCount, config.continuationCount);
    initial();
    complete(stream, runtime, config.continuationCount);
    for (std::uint32_t pass = 0; pass < config.progressPasses; ++pass) {
      progressNvme(stream, runtime, config.issueBudget,
                   config.completionBudget);
      publish(stream, runtime, config.continuationCount);
      ready();
      complete(stream, runtime, config.continuationCount);
    }
  }

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace nta
