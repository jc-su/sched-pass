#pragma once

#include "nta/RuntimeABI.h"

#include <cuda.h>

#include <cstdint>

namespace nta {

struct HostPhaseConfig {
  std::uint32_t objectCount;
  std::uint32_t workTicketCount;
  std::uint32_t progressBlocks;
  std::uint32_t progressRounds = 1;
};

struct NvmePhaseConfig {
  std::uint32_t objectCount;
  std::uint32_t workTicketCount;
  // Number of dependency/consumer rounds, not a polling-loop trip count.
  std::uint32_t progressRounds = 1;
  std::uint32_t issueBudget = 32;
  std::uint32_t completionBudget = 32;
  std::uint64_t progressTimeoutNs = 100'000'000ULL;
};

// Non-owning launcher for the finite acquisition phases linked into an
// instrumented cubin. It can enqueue into an ordinary stream or an existing
// CUDA graph capture; the caller retains ownership of the module and stream.
class FinitePhaseProgram {
public:
  explicit FinitePhaseProgram(CUmodule module);

  void reset(CUstream stream, abi::RuntimeView *runtime,
             std::uint32_t objectCount, std::uint32_t workTicketCount) const;
  void progressHost(CUstream stream, abi::RuntimeView *runtime,
                    std::uint32_t blocks) const;
  void progressNvme(CUstream stream, abi::RuntimeView *runtime,
                    std::uint32_t issueBudget,
                    std::uint32_t completionBudget) const;
  void progressNvmeUntilIdle(CUstream stream, abi::RuntimeView *runtime,
                             std::uint32_t issueBudget,
                             std::uint32_t completionBudget,
                             std::uint64_t timeoutNs) const;
  void publish(CUstream stream, abi::RuntimeView *runtime,
               std::uint32_t pendingBudget) const;
  void complete(CUstream stream, abi::RuntimeView *runtime,
                std::uint32_t workTicketCount) const;

  template <typename Initial, typename Ready>
  void enqueueHost(CUstream stream, abi::RuntimeView *runtime,
                   const HostPhaseConfig &config, Initial &&initial,
                   Ready &&ready) const {
    reset(stream, runtime, config.objectCount, config.workTicketCount);
    initial();
    complete(stream, runtime, config.workTicketCount);
    for (std::uint32_t round = 0; round < config.progressRounds; ++round) {
      progressHost(stream, runtime, config.progressBlocks);
      ready();
      complete(stream, runtime, config.workTicketCount);
    }
  }

  template <typename Initial, typename Ready>
  void enqueueNvme(CUstream stream, abi::RuntimeView *runtime,
                   const NvmePhaseConfig &config, Initial &&initial,
                   Ready &&ready) const {
    reset(stream, runtime, config.objectCount, config.workTicketCount);
    initial();
    complete(stream, runtime, config.workTicketCount);
    for (std::uint32_t round = 0; round < config.progressRounds; ++round) {
      progressNvmeUntilIdle(stream, runtime, config.issueBudget,
                            config.completionBudget,
                            config.progressTimeoutNs);
      ready();
      complete(stream, runtime, config.workTicketCount);
    }
  }

private:
  CUfunction reset_ = nullptr;
  CUfunction progressHost_ = nullptr;
  CUfunction progressNvme_ = nullptr;
  CUfunction progressNvmeUntilIdle_ = nullptr;
  CUfunction publish_ = nullptr;
  CUfunction complete_ = nullptr;
};

} // namespace nta
