#include "nta/DeviceWorkPlan.h"
#include "nta/HostRuntime.h"
#include "nta/JitPhase.h"
#include "nta/WorkPlan.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef NTA_TRANSPORT_PROGRAM_PATH
#error "NTA_TRANSPORT_PROGRAM_PATH must identify the transport program"
#endif

namespace {

void checkCuda(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

class IndexedBuffers {
public:
  static constexpr std::uint32_t Rows = 4;
  static constexpr std::uint32_t Bytes = Rows * sizeof(std::uint32_t);

  IndexedBuffers() {
    checkCuda(cudaHostAlloc(reinterpret_cast<void **>(&host_), Bytes,
                            cudaHostAllocMapped),
              "allocate mapped Host source");
    checkCuda(cudaHostGetDevicePointer(&sourceDevice_, host_, 0),
              "resolve mapped Host source");
    checkCuda(cudaMalloc(&staging_, Bytes), "allocate HBM staging");
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&sourceIndices_),
                         Rows * sizeof(std::uint32_t)),
              "allocate source indices");
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&stagingIndices_),
                         Rows * sizeof(std::uint32_t)),
              "allocate staging indices");
    const std::array<std::uint32_t, Rows> indices{0, 1, 2, 3};
    for (std::uint32_t row = 0; row < Rows; ++row) {
      host_[row] = 100U + row;
    }
    checkCuda(cudaMemset(staging_, 0, Bytes), "clear HBM staging");
    checkCuda(cudaMemcpy(sourceIndices_, indices.data(), sizeof(indices),
                         cudaMemcpyHostToDevice),
              "upload source indices");
    checkCuda(cudaMemcpy(stagingIndices_, indices.data(), sizeof(indices),
                         cudaMemcpyHostToDevice),
              "upload staging indices");
  }

  ~IndexedBuffers() {
    (void)cudaFree(stagingIndices_);
    (void)cudaFree(sourceIndices_);
    (void)cudaFree(staging_);
    (void)cudaFreeHost(host_);
  }

  IndexedBuffers(const IndexedBuffers &) = delete;
  IndexedBuffers &operator=(const IndexedBuffers &) = delete;

  [[nodiscard]] nta::IndexedHostObjectSpec object(std::uint64_t objectId,
                                                   std::uint32_t version) const {
    return {objectId,
            version,
            sourceDevice_,
            staging_,
            sourceIndices_,
            stagingIndices_,
            Rows,
            sizeof(std::uint32_t),
            sizeof(std::uint32_t),
            sizeof(std::uint32_t),
            Rows,
            Rows};
  }

  [[nodiscard]] bool stagingIsZero() const {
    std::array<std::uint32_t, Rows> values{};
    checkCuda(cudaMemcpy(values.data(), staging_, Bytes, cudaMemcpyDeviceToHost),
              "download HBM staging");
    for (const std::uint32_t value : values) {
      if (value != 0) {
        return false;
      }
    }
    return true;
  }

private:
  std::uint32_t *host_ = nullptr;
  void *sourceDevice_ = nullptr;
  void *staging_ = nullptr;
  std::uint32_t *sourceIndices_ = nullptr;
  std::uint32_t *stagingIndices_ = nullptr;
};

void runScenario(bool cancelBeforeIssue) {
  constexpr std::uint64_t ObjectId = 0x494e4445584544ULL;
  constexpr std::uint32_t Version = 7;
  nta::RuntimeConfig config{1, 1, 1, 1, 1, 1};
  nta::HostRuntime runtime(config);
  runtime.setRequest(0, 91, 3, 0, 0, 700,
                     cancelBeforeIssue ? UINT64_MAX : 0);

  IndexedBuffers buffers;
  const nta::IndexedHostObjectSpec object = buffers.object(ObjectId, Version);
  runtime.registerIndexedHostObjects(
      0, std::span<const nta::IndexedHostObjectSpec>(&object, 1));

  nta::WorkPlanBuilder builder(1);
  const std::uint32_t request = builder.addRequest({0, 3});
  const nta::abi::AcquireRequirement requirement = nta::makeRequirement(
      {0, 0, ObjectId, 0, Version, IndexedBuffers::Bytes});
  (void)builder.addWork(
      request, 0,
      std::span<const nta::abi::AcquireRequirement>(&requirement, 1), 1000, 700);
  nta::WorkPlan plan = builder.finish();
  nta::DeviceWorkPlan devicePlan(plan);
  nta::JitPhaseProgram phases(NTA_TRANSPORT_PROGRAM_PATH);

  cudaStream_t stream = nullptr;
  checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            "create lifecycle stream");
  try {
    phases.reset(stream, runtime.deviceView(), 1, 1);
    phases.discoverUnqueuedHost(stream, runtime.deviceView(),
                                devicePlan.workItems(),
                                devicePlan.dependencies(), 1);
    checkCuda(cudaStreamSynchronize(stream), "seal indexed discovery");
    if (cancelBeforeIssue) {
      runtime.cancelRequest(0, 3);
    }
    const std::uint32_t stickyBefore = runtime.readStickyFailedCount();
    phases.progressValidatedIndexedHostRangeParallel(
        stream, runtime.deviceView(), 0, 1, 1);
    checkCuda(cudaStreamSynchronize(stream), "retire indexed acquisition");

    const nta::abi::ObjectEntry observedObject = runtime.readObject(0);
    const nta::abi::IntentPool pool = runtime.readIntentPool();
    const nta::abi::WorkTicket ticket = runtime.readWorkTicket(0);
    require(buffers.stagingIsZero(),
            "terminal indexed acquisition performed a data transfer");
    require(pool.active == 0 && pool.enqueued == pool.consumed,
            "terminal indexed acquisition leaked its intent");
    if (cancelBeforeIssue) {
      require(observedObject.state == static_cast<std::uint32_t>(
                                          nta::abi::ObjectState::New) &&
                  ticket.state == static_cast<std::uint32_t>(
                                      nta::abi::WorkTicketState::Cancelled) &&
                  runtime.readStickyFailedCount() == stickyBefore,
              "normal indexed cancellation poisoned or stranded the group");
    } else {
      require(observedObject.state == static_cast<std::uint32_t>(
                                          nta::abi::ObjectState::Failed) &&
                  ticket.state == static_cast<std::uint32_t>(
                                      nta::abi::WorkTicketState::Failed) &&
                  runtime.readStickyFailedCount() > stickyBefore,
              "violated unqueued-credit contract did not fail closed");
    }
  } catch (...) {
    (void)cudaStreamDestroy(stream);
    throw;
  }
  checkCuda(cudaStreamDestroy(stream), "destroy lifecycle stream");
}

} // namespace

int main() {
  try {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
      std::cout << "SKIP: no CUDA device\n";
      return 0;
    }
    runScenario(true);
    runScenario(false);
    std::cout << "indexed intent lifecycle validation passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "indexed intent lifecycle validation failed: " << error.what()
              << '\n';
    return 1;
  }
}
