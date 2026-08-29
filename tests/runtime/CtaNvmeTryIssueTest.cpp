#include "benchmarks/kv/KvTypes.h"
#include "nta/FinitePhase.h"
#include "nta/RuntimeABI.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NTA_KV_CUBIN_PATH
#error "NTA_KV_CUBIN_PATH must identify the instrumented device image"
#endif

namespace {

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

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename T> class DeviceArray {
public:
  explicit DeviceArray(std::size_t count) : count_(count) {
    checkCuda(
        cudaMalloc(reinterpret_cast<void **>(&pointer_), count * sizeof(T)),
        "cudaMalloc test array");
    checkCuda(cudaMemset(pointer_, 0, count * sizeof(T)),
              "cudaMemset test array");
  }

  ~DeviceArray() {
    if (pointer_ != nullptr) {
      (void)cudaFree(pointer_);
    }
  }

  DeviceArray(const DeviceArray &) = delete;
  DeviceArray &operator=(const DeviceArray &) = delete;

  [[nodiscard]] T *get() const noexcept { return pointer_; }

  void upload(const T &value, std::size_t index = 0) {
    require(index < count_, "test upload index is out of range");
    checkCuda(cudaMemcpy(pointer_ + index, &value, sizeof(value),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy test upload");
  }

  [[nodiscard]] T download(std::size_t index = 0) const {
    require(index < count_, "test download index is out of range");
    T value{};
    checkCuda(cudaMemcpy(&value, pointer_ + index, sizeof(value),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy test download");
    return value;
  }

private:
  T *pointer_ = nullptr;
  std::size_t count_ = 0;
};

class KernelModule {
public:
  KernelModule() {
    checkDriver(cuModuleLoad(&module_, NTA_KV_CUBIN_PATH),
                "cuModuleLoad test cubin");
    checkDriver(cuModuleGetFunction(&initial_, module_, "nta_nvme_hash_kernel"),
                "cuModuleGetFunction initial");
    checkDriver(
        cuModuleGetFunction(&ready_, module_, "nta_nvme_ready_hash_kernel"),
        "cuModuleGetFunction ready");
  }

  ~KernelModule() {
    if (module_ != nullptr) {
      (void)cuModuleUnload(module_);
    }
  }

  void initial(CUstream stream, nta::abi::RuntimeView *runtime,
               const nta::benchmark::TileTask *task, std::uint64_t *output,
               std::uint32_t count = 1) const {
    launch(initial_, stream, runtime, task, output, count, true);
  }

  void ready(CUstream stream, nta::abi::RuntimeView *runtime,
             const nta::benchmark::TileTask *task, std::uint64_t *output,
             std::uint32_t count = 1) const {
    launch(ready_, stream, runtime, task, output, count, false);
  }

  [[nodiscard]] CUmodule module() const noexcept { return module_; }

private:
  static void launch(CUfunction function, CUstream stream,
                     nta::abi::RuntimeView *runtime,
                     const nta::benchmark::TileTask *task,
                     std::uint64_t *output, std::uint32_t count, bool initial) {
    constexpr std::uint32_t Phase = 0;
    CUdeviceptr runtimeAddress = reinterpret_cast<CUdeviceptr>(runtime);
    CUdeviceptr taskAddress = reinterpret_cast<CUdeviceptr>(task);
    CUdeviceptr outputAddress = reinterpret_cast<CUdeviceptr>(output);
    void *initialArguments[] = {&runtimeAddress, &taskAddress, &count,
                                &outputAddress,
                                const_cast<std::uint32_t *>(&Phase)};
    void *readyArguments[] = {&runtimeAddress, &taskAddress, &count,
                              &outputAddress};
    checkDriver(cuLaunchKernel(function, count, 1, 1, 256, 1, 1, 0, stream,
                               initial ? initialArguments : readyArguments,
                               nullptr),
                initial ? "cuLaunchKernel initial" : "cuLaunchKernel ready");
  }

  CUmodule module_ = nullptr;
  CUfunction initial_ = nullptr;
  CUfunction ready_ = nullptr;
};

class QueueFixture {
public:
  static constexpr std::uint32_t Capacity = 2;
  static constexpr std::uint32_t Depth = 4;
  static constexpr std::uint32_t PageBytes = 4096;
  static constexpr std::uint32_t PageCount = 3;
  static constexpr std::uint32_t Bytes = PageBytes * PageCount;
  static constexpr std::uint64_t ObjectId = 0x4e544154455354ULL;
  // PRPs contain transport/IOMMU addresses, not CUDA virtual addresses. Use a
  // deterministic aligned IOVA fixture while staging remains real HBM.
  static constexpr std::uint64_t DmaBase = 0x4'0000'0000ULL;

  QueueFixture()
      : requests(Capacity), tenants(1), requestProgress(Capacity),
        objects(Capacity), replicas(Capacity),
        backends(nta::abi::BackendCount), intents(Capacity),
        workTickets(Capacity), dependencies(Capacity), intentPool(1),
        intentQueueEntries(Capacity),
        intentQueueControls(nta::abi::BackendCount),
        intentQueueHeap(nta::abi::BackendCount * Capacity),
        readyWorkTickets(Capacity), readyCount(1), readyHead(1),
        pendingWorkTickets(Capacity), pendingCount(1), ctaCompletions(Capacity),
        objectDependentHeads(Capacity), dependencyNext(Capacity),
        dependencySatisfied(Capacity), remainingDependencies(Capacity),
        changedWorkTickets(Capacity), changedQueued(Capacity), changedCount(1),
        changedOverflow(1),
        submissions(Depth), completions(Depth),
        prpLists(Depth * PageBytes / sizeof(std::uint64_t)), contexts(Depth),
        control(1), sqDoorbell(1), cqDoorbell(1), queue(1),
        dmaPages(Capacity * PageCount),
        staging(Capacity * Bytes / sizeof(std::uint32_t)), runtime(1),
        tasks(Capacity), output(Capacity) {
    initialize();
  }

  void initialize() {
    std::vector<std::uint32_t> contents(Bytes / sizeof(std::uint32_t));
    expected = 0;
    for (std::uint32_t index = 0; index < contents.size(); ++index) {
      contents[index] = index * 17U + 3U;
      expected += static_cast<std::uint64_t>(contents[index]) * (index + 1ULL);
    }
    checkCuda(cudaMemcpy(staging.get(), contents.data(), Bytes,
                         cudaMemcpyHostToDevice),
              "upload staged test contents");

    requests.upload({1, 0, Bytes, 0, 7, 0, 4, 0});
    requestProgress.upload({1, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    tenants.upload({Capacity * Bytes, 0});
    objects.upload({ObjectId, reinterpret_cast<std::uint64_t>(staging.get()),
                    Bytes, 0, 3,
                    static_cast<std::uint32_t>(nta::abi::ObjectState::New), 0,
                    1, 0, 0, 0});
    for (std::uint32_t page = 0; page < PageCount; ++page) {
      dmaPages.upload(DmaBase + static_cast<std::uint64_t>(page) * PageBytes,
                      page);
    }
    replicas.upload({0, reinterpret_cast<std::uint64_t>(dmaPages.get()), 80'000,
                     7'000'000'000ULL,
                     static_cast<std::uint32_t>(nta::abi::SourceKind::Nvme),
                     PageCount,
                     static_cast<std::uint32_t>(nta::abi::SourceKind::Nvme),
                     nta::abi::ReplicaTransport, 0, 0});

    for (std::uint32_t index = 0; index < nta::abi::BackendCount; ++index) {
      nta::abi::BackendView backend{};
      backend.sourceKind = index;
      backend.backendIndex = index;
      backends.upload(backend, index);
    }

    nta::abi::NvmeQueueControl controlValue{};
    controlValue.magic = nta::abi::NvmeQueueControlMagic;
    controlValue.abiVersion = nta::abi::NvmeQueueAbiVersion;
    controlValue.state =
        static_cast<std::uint32_t>(nta::abi::NvmeQueueState::Online);
    controlValue.generation = 9;
    controlValue.queueId = 1;
    control.upload(controlValue);

    nta::abi::NvmeQueueView queueValue{};
    queueValue.submissions = submissions.get();
    queueValue.completions = completions.get();
    queueValue.prpLists = prpLists.get();
    queueValue.prpListDmaAddress = 0x100000;
    queueValue.sqDoorbell = sqDoorbell.get();
    queueValue.cqDoorbell = cqDoorbell.get();
    queueValue.contexts = contexts.get();
    queueValue.control = control.get();
    queueValue.depth = Depth;
    queueValue.controllerPageSize = PageBytes;
    queueValue.lbaShift = 9;
    queueValue.namespaceId = 1;
    queueValue.cqPhase = 1;
    queueValue.active = 1;
    queueValue.queueGeneration = 9;
    queueValue.queueId = 1;
    queueValue.directMaxPrpPages = 32;
    queue.upload(queueValue);

    nta::abi::BackendView nvmeBackend{};
    nvmeBackend.deviceState = reinterpret_cast<std::uint64_t>(queue.get());
    nvmeBackend.outstandingBytes = 0;
    nvmeBackend.maxOutstandingBytes = Capacity * Bytes;
    nvmeBackend.sourceKind =
        static_cast<std::uint32_t>(nta::abi::SourceKind::Nvme);
    nvmeBackend.active = 1;
    nvmeBackend.backendIndex =
        static_cast<std::uint32_t>(nta::abi::SourceKind::Nvme);
    nvmeBackend.flags = nta::abi::BackendCtaTryIssue;
    backends.upload(nvmeBackend,
                    static_cast<std::uint32_t>(nta::abi::SourceKind::Nvme));

    nta::abi::IntentSlot intent{};
    intent.sequence = 0;
    intents.upload(intent);
    intentPool.upload({0, 0, Capacity, 0, 0, 0, {0, 0, 0, 0}});
    for (std::uint32_t index = 0; index < Capacity; ++index) {
      intentQueueEntries.upload({}, index);
    }
    for (std::uint32_t index = 0; index < nta::abi::BackendCount; ++index) {
      intentQueueControls.upload({}, index);
    }
    for (std::uint32_t index = 0;
         index < nta::abi::BackendCount * Capacity; ++index) {
      intentQueueHeap.upload({0, nta::abi::InvalidIndex, 0}, index);
    }
    workTickets.upload(
        {0, 0, 0, static_cast<std::uint32_t>(nta::abi::WorkTicketState::New), 0,
         0, nta::abi::InvalidIndex, 0, 0, 0, 0, 1});
    for (std::uint32_t slot = 0; slot < Capacity; ++slot) {
      objectDependentHeads.upload(nta::abi::InvalidIndex, slot);
    }

    nta::abi::RuntimeView runtimeValue{};
    runtimeValue.requests = requests.get();
    runtimeValue.tenants = tenants.get();
    runtimeValue.objects = objects.get();
    runtimeValue.replicas = replicas.get();
    runtimeValue.backends = backends.get();
    runtimeValue.intents = intents.get();
    runtimeValue.workTickets = workTickets.get();
    runtimeValue.dependencies = dependencies.get();
    runtimeValue.intentPool = intentPool.get();
    runtimeValue.intentQueueEntries = intentQueueEntries.get();
    runtimeValue.intentQueueControls = intentQueueControls.get();
    runtimeValue.intentQueueHeap = intentQueueHeap.get();
    runtimeValue.readyWorkTickets = readyWorkTickets.get();
    runtimeValue.readyCount = readyCount.get();
    runtimeValue.readyHead = readyHead.get();
    runtimeValue.pendingWorkTickets = pendingWorkTickets.get();
    runtimeValue.pendingCount = pendingCount.get();
    runtimeValue.ctaCompletions = ctaCompletions.get();
    runtimeValue.objectDependentHeads = objectDependentHeads.get();
    runtimeValue.dependencyNext = dependencyNext.get();
    runtimeValue.dependencySatisfied = dependencySatisfied.get();
    runtimeValue.remainingDependencies = remainingDependencies.get();
    runtimeValue.changedWorkTickets = changedWorkTickets.get();
    runtimeValue.changedQueued = changedQueued.get();
    runtimeValue.changedCount = changedCount.get();
    runtimeValue.changedOverflow = changedOverflow.get();
    runtimeValue.requestProgress = requestProgress.get();
    runtimeValue.requestCapacity = Capacity;
    runtimeValue.tenantCapacity = 1;
    runtimeValue.objectCapacity = Capacity;
    runtimeValue.replicaCapacity = Capacity;
    runtimeValue.backendCapacity = nta::abi::BackendCount;
    runtimeValue.intentCapacity = Capacity;
    runtimeValue.workTicketCapacity = Capacity;
    runtimeValue.dependencyCapacity = Capacity;
    runtimeValue.maxDependenciesPerWorkTicket = 1;
    runtimeValue.abiVersion = nta::abi::Version;
    runtime.upload(runtimeValue);

    tasks.upload({0, ObjectId, 0, 0, 7, 0, 3, Bytes, 0, 0, 0});
  }

  void initializeSecondRequest() {
    constexpr std::uint32_t slot = 1;
    constexpr std::uint64_t objectId = ObjectId + 1;
    std::vector<std::uint32_t> contents(Bytes / sizeof(std::uint32_t));
    for (std::uint32_t index = 0; index < contents.size(); ++index) {
      contents[index] = index * 19U + 5U;
    }
    checkCuda(cudaMemcpy(staging.get() + contents.size(), contents.data(),
                         Bytes, cudaMemcpyHostToDevice),
              "upload second staged test contents");
    requests.upload({2, 0, Bytes, 0, 8, 0, 3, 0}, slot);
    requestProgress.upload({2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                           slot);
    objects.upload(
        {objectId,
         reinterpret_cast<std::uint64_t>(staging.get() + contents.size()),
         Bytes, 0, 4, static_cast<std::uint32_t>(nta::abi::ObjectState::New),
         slot, 1, 0, 0, 0},
        slot);
    for (std::uint32_t page = 0; page < PageCount; ++page) {
      dmaPages.upload(
          DmaBase + static_cast<std::uint64_t>(slot * PageCount + page) *
                        PageBytes,
          slot * PageCount + page);
    }
    replicas.upload(
        {163'840, reinterpret_cast<std::uint64_t>(dmaPages.get() + PageCount),
         80'000, 7'000'000'000ULL,
         static_cast<std::uint32_t>(nta::abi::SourceKind::Nvme), PageCount,
         static_cast<std::uint32_t>(nta::abi::SourceKind::Nvme),
         nta::abi::ReplicaTransport, 0, 0},
        slot);
    workTickets.upload(
        {0, 0, 0, static_cast<std::uint32_t>(nta::abi::WorkTicketState::New), 0,
         0, nta::abi::InvalidIndex, 0, 0, 0, slot, 1},
        slot);
    tasks.upload({0, objectId, 0, slot, 8, slot, 4, Bytes, slot, 0, 0}, slot);
  }

  void completeCurrentSubmission(std::uint32_t statusCode = 0) {
    const nta::abi::NvmeQueueView queueValue = queue.download();
    require(queueValue.outstanding == 1,
            "test queue must have exactly one outstanding command");
    const std::uint32_t submissionIndex =
        (queueValue.sqTail + Depth - 1U) % Depth;
    const nta::abi::NvmeSubmission submission =
        submissions.download(submissionIndex);
    const std::uint32_t commandId = submission.dword[0] >> 16U;
    require((submission.dword[0] & 0xffU) == 0x02U,
            "CTA did not construct an NVMe read command");
    require(commandId < Depth && contexts.download(commandId).active == 1,
            "CTA did not publish an active command context");
    nta::abi::NvmeCompletion completion{};
    completion.dword[3] =
        commandId | ((queueValue.cqPhase | (statusCode << 1U)) << 16U);
    completions.upload(completion, queueValue.cqHead);
  }

  void injectMalformedCompletion() {
    const nta::abi::NvmeQueueView queueValue = queue.download();
    require(queueValue.outstanding == 1,
            "malformed completion test needs one outstanding command");
    nta::abi::NvmeCompletion completion{};
    completion.dword[3] = Depth | (queueValue.cqPhase << 16U);
    completions.upload(completion, queueValue.cqHead);
  }

  [[nodiscard]] std::uint32_t activeContextCount() const {
    std::uint32_t active = 0;
    for (std::uint32_t commandId = 0; commandId < Depth; ++commandId) {
      active += contexts.download(commandId).active != 0 ? 1U : 0U;
    }
    return active;
  }

  void resetForFallback() {
    nta::abi::ObjectEntry object = objects.download();
    object.state = static_cast<std::uint32_t>(nta::abi::ObjectState::New);
    object.issueCount = 0;
    objects.upload(object);
    workTickets.upload(
        {0, 0, 0, static_cast<std::uint32_t>(nta::abi::WorkTicketState::New), 0,
         0, nta::abi::InvalidIndex, 0, 0, 0, 0, 1});
    requestProgress.upload({1, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    pendingCount.upload(0);
    ctaCompletions.upload(0);
    readyCount.upload(0);
    readyHead.upload(0);
    objectDependentHeads.upload(nta::abi::InvalidIndex);
    remainingDependencies.upload(0);
    changedCount.upload(0);
    changedOverflow.upload(0);
    nta::abi::IntentSlot emptyIntent{};
    intents.upload(emptyIntent);
    intentPool.upload({0, 0, Capacity, 0, 0, 0, {0, 0, 0, 0}});
    output.upload(0);
    nta::abi::NvmeQueueView queueValue = queue.download();
    require(queueValue.outstanding == 0,
            "cannot reset fixture with outstanding commands");
    queueValue.ownerLock = 1;
    queue.upload(queueValue);
  }

  void resetForOfflineQueue() { resetForFallback(); }

  void makeQueueFatalAndReleaseLease() {
    nta::abi::NvmeQueueControl controlValue = control.download();
    controlValue.state =
        static_cast<std::uint32_t>(nta::abi::NvmeQueueState::Fatal);
    control.upload(controlValue);
    releaseQueueLease();
  }

  void releaseQueueLease() {
    nta::abi::NvmeQueueView queueValue = queue.download();
    queueValue.ownerLock = 0;
    queue.upload(queueValue);
  }

  void replaceIssuedObject() {
    nta::abi::ObjectEntry object = objects.download();
    object.objectId = ObjectId + 1;
    object.version += 1;
    object.state = static_cast<std::uint32_t>(nta::abi::ObjectState::New);
    objects.upload(object);
    nta::abi::WorkTicket workTicket = workTickets.download();
    workTicket.generation += 1;
    workTicket.state =
        static_cast<std::uint32_t>(nta::abi::WorkTicketState::Pending);
    workTickets.upload(workTicket);
  }

  DeviceArray<nta::abi::RequestContext> requests;
  DeviceArray<nta::abi::TenantContext> tenants;
  DeviceArray<nta::abi::RequestProgress> requestProgress;
  DeviceArray<nta::abi::ObjectEntry> objects;
  DeviceArray<nta::abi::ReplicaEntry> replicas;
  DeviceArray<nta::abi::BackendView> backends;
  DeviceArray<nta::abi::IntentSlot> intents;
  DeviceArray<nta::abi::WorkTicket> workTickets;
  DeviceArray<nta::abi::WorkDependency> dependencies;
  DeviceArray<nta::abi::IntentPool> intentPool;
  DeviceArray<nta::abi::IntentQueueEntry> intentQueueEntries;
  DeviceArray<nta::abi::IntentQueueControl> intentQueueControls;
  DeviceArray<nta::abi::IntentQueueNode> intentQueueHeap;
  DeviceArray<std::uint32_t> readyWorkTickets;
  DeviceArray<std::uint32_t> readyCount;
  DeviceArray<std::uint32_t> readyHead;
  DeviceArray<std::uint32_t> pendingWorkTickets;
  DeviceArray<std::uint32_t> pendingCount;
  DeviceArray<std::uint32_t> ctaCompletions;
  DeviceArray<std::uint32_t> objectDependentHeads;
  DeviceArray<std::uint32_t> dependencyNext;
  DeviceArray<std::uint32_t> dependencySatisfied;
  DeviceArray<std::uint32_t> remainingDependencies;
  DeviceArray<std::uint32_t> changedWorkTickets;
  DeviceArray<std::uint32_t> changedQueued;
  DeviceArray<std::uint32_t> changedCount;
  DeviceArray<std::uint32_t> changedOverflow;
  DeviceArray<nta::abi::NvmeSubmission> submissions;
  DeviceArray<nta::abi::NvmeCompletion> completions;
  DeviceArray<std::uint64_t> prpLists;
  DeviceArray<nta::abi::NvmeCommandContext> contexts;
  DeviceArray<nta::abi::NvmeQueueControl> control;
  DeviceArray<std::uint32_t> sqDoorbell;
  DeviceArray<std::uint32_t> cqDoorbell;
  DeviceArray<nta::abi::NvmeQueueView> queue;
  DeviceArray<std::uint64_t> dmaPages;
  DeviceArray<std::uint32_t> staging;
  DeviceArray<nta::abi::RuntimeView> runtime;
  DeviceArray<nta::benchmark::TileTask> tasks;
  DeviceArray<std::uint64_t> output;
  std::uint64_t expected = 0;
};

void verifyDirectPath(QueueFixture &fixture, const KernelModule &kernels,
                      const nta::FinitePhaseProgram &phases, CUstream stream) {
  kernels.initial(stream, fixture.runtime.get(), fixture.tasks.get(),
                  fixture.output.get());
  checkDriver(cuStreamSynchronize(stream), "synchronize direct issue");

  const nta::abi::NvmeQueueView issued = fixture.queue.download();
  const nta::abi::BackendView backend = fixture.backends.download(
      static_cast<std::uint32_t>(nta::abi::SourceKind::Nvme));
  if (!(issued.submitted == 1 && issued.directSubmitted == 1 &&
        issued.directFallbacks == 0 && issued.outstanding == 1 &&
        issued.ownerLock == 0 && backend.pendingAcquisitions == 0)) {
    throw std::runtime_error(
        "CTA direct issue counters are inconsistent: submitted=" +
        std::to_string(issued.submitted) +
        " direct=" + std::to_string(issued.directSubmitted) +
        " fallback=" + std::to_string(issued.directFallbacks) +
        " outstanding=" + std::to_string(issued.outstanding) +
        " owner=" + std::to_string(issued.ownerLock) +
        " pending=" + std::to_string(backend.pendingAcquisitions));
  }
  require(fixture.intentPool.download().active == 0,
          "direct issue unexpectedly published a scheduled intent");
  require(fixture.objects.download().state ==
              static_cast<std::uint32_t>(nta::abi::ObjectState::Issued),
          "direct issue did not move the object to Issued");
  require(fixture.workTickets.download().state ==
              static_cast<std::uint32_t>(nta::abi::WorkTicketState::Pending),
          "direct issue did not leave a pending workTicket");
  require(fixture.objectDependentHeads.download() != nta::abi::InvalidIndex &&
              fixture.remainingDependencies.download() == 1 &&
              fixture.changedCount.download() == 0,
          "direct issue did not publish its reverse dependency edge");
  const nta::abi::RequestProgress pendingProgress =
      fixture.requestProgress.download();
  require(
      pendingProgress.expectedWork == 1 && pendingProgress.pendingWork == 1 &&
          pendingProgress.runnableWork == 0 &&
          pendingProgress.completedWork == 0 &&
          pendingProgress.pendingComputeNs == pendingProgress.expectedComputeNs,
      "request feedback did not account for pending work");
  const std::uint32_t submissionIndex =
      (issued.sqTail + QueueFixture::Depth - 1U) % QueueFixture::Depth;
  const nta::abi::NvmeSubmission submission =
      fixture.submissions.download(submissionIndex);
  const std::uint32_t commandId = submission.dword[0] >> 16U;
  const std::uint64_t firstPrp =
      static_cast<std::uint64_t>(submission.dword[6]) |
      (static_cast<std::uint64_t>(submission.dword[7]) << 32U);
  const std::uint64_t secondPrp =
      static_cast<std::uint64_t>(submission.dword[8]) |
      (static_cast<std::uint64_t>(submission.dword[9]) << 32U);
  const std::uint64_t expectedPrpList =
      issued.prpListDmaAddress +
      static_cast<std::uint64_t>(commandId) * QueueFixture::PageBytes;
  const std::size_t listBase = static_cast<std::size_t>(commandId) *
                               QueueFixture::PageBytes / sizeof(std::uint64_t);
  require(firstPrp == fixture.dmaPages.download(0) &&
              secondPrp == expectedPrpList &&
              fixture.prpLists.download(listBase) ==
                  fixture.dmaPages.download(1) &&
              fixture.prpLists.download(listBase + 1) ==
                  fixture.dmaPages.download(2),
          "multi-page command did not construct the NVMe PRP list");

  fixture.completeCurrentSubmission();
  phases.progressNvme(stream, fixture.runtime.get(), 1, 1);
  checkDriver(cuStreamSynchronize(stream),
              "synchronize completion-driven publication");
  require(fixture.remainingDependencies.download() == 0 &&
              fixture.changedCount.download() == 0 &&
              fixture.changedOverflow.download() == 0,
          "object completion did not directly publish exactly one ticket");
  phases.publish(stream, fixture.runtime.get(), 1);
  kernels.ready(stream, fixture.runtime.get(), fixture.tasks.get(),
                fixture.output.get());
  checkDriver(cuStreamSynchronize(stream), "synchronize direct completion");
  require(fixture.queue.download().completed == 1 &&
              fixture.queue.download().outstanding == 0,
          "bounded progress did not retire the direct command");
  require(fixture.workTickets.download().state ==
                  static_cast<std::uint32_t>(nta::abi::WorkTicketState::Done) &&
              fixture.output.download() == fixture.expected,
          "runnable-work launch did not consume the completed object");
  const nta::abi::RequestProgress completedProgress =
      fixture.requestProgress.download();
  require(completedProgress.expectedWork == 1 &&
              completedProgress.pendingWork == 0 &&
              completedProgress.runnableWork == 0 &&
              completedProgress.completedWork == 1 &&
              completedProgress.failedWork == 0 &&
              completedProgress.pendingComputeNs == 0 &&
              completedProgress.completedComputeNs ==
                  completedProgress.expectedComputeNs,
          "request feedback did not account for completed work");
}

void verifyConcurrentDirectPath(QueueFixture &fixture,
                                const KernelModule &kernels, CUstream stream) {
  fixture.initializeSecondRequest();
  kernels.initial(stream, fixture.runtime.get(), fixture.tasks.get(),
                  fixture.output.get(), QueueFixture::Capacity);
  checkDriver(cuStreamSynchronize(stream),
              "synchronize concurrent direct issue");

  const nta::abi::NvmeQueueView queue = fixture.queue.download();
  const nta::abi::BackendView backend = fixture.backends.download(
      static_cast<std::uint32_t>(nta::abi::SourceKind::Nvme));
  const nta::abi::IntentPool pool = fixture.intentPool.download();
  if (!(queue.directSubmitted >= 1 &&
        queue.directSubmitted + pool.active == QueueFixture::Capacity &&
        queue.outstanding == queue.directSubmitted &&
        queue.directFallbacks == pool.active &&
        backend.pendingAcquisitions == pool.active)) {
    throw std::runtime_error(
        "concurrent CTAs lost or duplicated NVMe acquisition state: direct=" +
        std::to_string(queue.directSubmitted) +
        " fallback=" + std::to_string(queue.directFallbacks) +
        " outstanding=" + std::to_string(queue.outstanding) +
        " active=" + std::to_string(pool.active) +
        " pending=" + std::to_string(backend.pendingAcquisitions));
  }
  for (std::uint32_t slot = 0; slot < QueueFixture::Capacity; ++slot) {
    const auto state = static_cast<nta::abi::ObjectState>(
        fixture.objects.download(slot).state);
    require(
        (state == nta::abi::ObjectState::Issued ||
         state == nta::abi::ObjectState::Queued) &&
            fixture.objects.download(slot).issueCount == 1 &&
            fixture.workTickets.download(slot).state ==
                static_cast<std::uint32_t>(nta::abi::WorkTicketState::Pending),
        "concurrent request did not retain one live request-bound ticket");
  }
}

void verifyFallbackPath(QueueFixture &fixture, const KernelModule &kernels,
                        const nta::FinitePhaseProgram &phases,
                        CUstream stream) {
  fixture.resetForFallback();
  kernels.initial(stream, fixture.runtime.get(), fixture.tasks.get(),
                  fixture.output.get());
  checkDriver(cuStreamSynchronize(stream), "synchronize fallback publication");

  nta::abi::NvmeQueueView queue = fixture.queue.download();
  const nta::abi::BackendView queuedBackend = fixture.backends.download(
      static_cast<std::uint32_t>(nta::abi::SourceKind::Nvme));
  require(queue.directSubmitted == 0 && queue.directFallbacks == 1 &&
              queue.outstanding == 0,
          "contended CTA did not take the non-spinning fallback");
  require(fixture.intentPool.download().active == 1 &&
              fixture.intents.download().intent.valid == 1 &&
              queuedBackend.pendingAcquisitions == 1,
          "fallback did not publish one scheduler intent");

  queue.ownerLock = 0;
  fixture.queue.upload(queue);
  phases.progressNvme(stream, fixture.runtime.get(), 1, 1);
  checkDriver(cuStreamSynchronize(stream), "synchronize scheduled issue");
  require(fixture.intentPool.download().active == 0 &&
              fixture.queue.download().outstanding == 1 &&
              fixture.backends
                      .download(static_cast<std::uint32_t>(
                          nta::abi::SourceKind::Nvme))
                      .pendingAcquisitions == 0,
          "bounded scheduler did not issue the fallback intent");

  fixture.completeCurrentSubmission();
  phases.progressNvme(stream, fixture.runtime.get(), 1, 1);
  phases.publish(stream, fixture.runtime.get(), 1);
  kernels.ready(stream, fixture.runtime.get(), fixture.tasks.get(),
                fixture.output.get());
  checkDriver(cuStreamSynchronize(stream), "synchronize fallback completion");
  require(fixture.workTickets.download().state ==
                  static_cast<std::uint32_t>(nta::abi::WorkTicketState::Done) &&
              fixture.output.download() == fixture.expected,
          "scheduled fallback did not resume the workTicket");
}

void verifyStaleCompletionIsolation(QueueFixture &fixture,
                                    const KernelModule &kernels,
                                    const nta::FinitePhaseProgram &phases,
                                    CUstream stream) {
  kernels.initial(stream, fixture.runtime.get(), fixture.tasks.get(),
                  fixture.output.get());
  checkDriver(cuStreamSynchronize(stream), "synchronize stale direct issue");
  fixture.completeCurrentSubmission();
  fixture.replaceIssuedObject();

  phases.progressNvme(stream, fixture.runtime.get(), 1, 1);
  checkDriver(cuStreamSynchronize(stream), "synchronize stale completion");

  const nta::abi::NvmeQueueView queue = fixture.queue.download();
  const nta::abi::ObjectEntry object = fixture.objects.download();
  const nta::abi::WorkTicket workTicket = fixture.workTickets.download();
  const nta::abi::RequestContext request = fixture.requests.download();
  const nta::abi::TenantContext tenant = fixture.tenants.download();
  const nta::abi::BackendView backend = fixture.backends.download(
      static_cast<std::uint32_t>(nta::abi::SourceKind::Nvme));
  require(queue.outstanding == 0 && queue.completed == 0 && queue.failed == 1 &&
              queue.error != 0,
          "stale completion was not retired as a queue error");
  require(object.objectId == QueueFixture::ObjectId + 1 &&
              object.version == 4 &&
              object.state ==
                  static_cast<std::uint32_t>(nta::abi::ObjectState::New),
          "stale completion modified the replacement object");
  require(workTicket.generation == 8 &&
              workTicket.state == static_cast<std::uint32_t>(
                                      nta::abi::WorkTicketState::Pending),
          "stale completion modified the replacement workTicket");
  require(request.outstandingBytes == 0 && tenant.outstandingBytes == 0 &&
              backend.outstandingBytes == 0,
          "stale completion leaked admission credits");
}

void verifyNvmeStatusFailure(QueueFixture &fixture, const KernelModule &kernels,
                             const nta::FinitePhaseProgram &phases,
                             CUstream stream) {
  kernels.initial(stream, fixture.runtime.get(), fixture.tasks.get(),
                  fixture.output.get());
  checkDriver(cuStreamSynchronize(stream), "synchronize status issue");
  fixture.completeCurrentSubmission(1);
  phases.progressNvme(stream, fixture.runtime.get(), 1, 1);
  checkDriver(cuStreamSynchronize(stream), "synchronize status failure");

  const nta::abi::NvmeQueueView queue = fixture.queue.download();
  const nta::abi::RequestContext request = fixture.requests.download();
  const nta::abi::TenantContext tenant = fixture.tenants.download();
  const nta::abi::BackendView backend = fixture.backends.download(
      static_cast<std::uint32_t>(nta::abi::SourceKind::Nvme));
  require(queue.active == 1 && queue.outstanding == 0 && queue.completed == 0 &&
              queue.failed == 1 && queue.error == 1 &&
              fixture.activeContextCount() == 0,
          "NVMe status failure did not retire exactly one command");
  require(fixture.objects.download().state ==
                  static_cast<std::uint32_t>(nta::abi::ObjectState::Failed) &&
              fixture.workTickets.download().state ==
                  static_cast<std::uint32_t>(nta::abi::WorkTicketState::Failed),
          "NVMe status failure did not fail dependent state");
  require(request.outstandingBytes == 0 && tenant.outstandingBytes == 0 &&
              backend.outstandingBytes == 0,
          "NVMe status failure leaked admission credits");
}

void verifyMalformedCompletionFailure(QueueFixture &fixture,
                                      const KernelModule &kernels,
                                      const nta::FinitePhaseProgram &phases,
                                      CUstream stream) {
  kernels.initial(stream, fixture.runtime.get(), fixture.tasks.get(),
                  fixture.output.get());
  checkDriver(cuStreamSynchronize(stream), "synchronize malformed issue");
  fixture.injectMalformedCompletion();
  phases.progressNvme(stream, fixture.runtime.get(), 1, 1);
  checkDriver(cuStreamSynchronize(stream),
              "synchronize malformed completion failure");

  const nta::abi::NvmeQueueView queue = fixture.queue.download();
  const nta::abi::RequestContext request = fixture.requests.download();
  const nta::abi::TenantContext tenant = fixture.tenants.download();
  const nta::abi::BackendView backend = fixture.backends.download(
      static_cast<std::uint32_t>(nta::abi::SourceKind::Nvme));
  require(queue.active == 0 && queue.ownerLock == 0 && queue.outstanding == 0 &&
              queue.error == 0xffffffffU && queue.failed == 1 &&
              fixture.activeContextCount() == 0,
          "malformed completion did not quiesce and reclaim the queue");
  require(fixture.objects.download().state ==
                  static_cast<std::uint32_t>(nta::abi::ObjectState::Failed) &&
              fixture.workTickets.download().state ==
                  static_cast<std::uint32_t>(nta::abi::WorkTicketState::Failed),
          "malformed completion did not fail dependent state");
  require(request.outstandingBytes == 0 && tenant.outstandingBytes == 0 &&
              backend.outstandingBytes == 0,
          "malformed completion leaked admission credits");
}

void verifyStaleIntentIsolation(QueueFixture &fixture,
                                const KernelModule &kernels,
                                const nta::FinitePhaseProgram &phases,
                                CUstream stream) {
  fixture.resetForFallback();
  kernels.initial(stream, fixture.runtime.get(), fixture.tasks.get(),
                  fixture.output.get());
  checkDriver(cuStreamSynchronize(stream), "synchronize stale intent publish");
  require(fixture.intentPool.download().active == 1,
          "stale-intent test did not publish a fallback intent");
  fixture.replaceIssuedObject();
  fixture.releaseQueueLease();

  phases.progressNvme(stream, fixture.runtime.get(), 1, 1);
  checkDriver(cuStreamSynchronize(stream), "synchronize stale intent retire");

  const nta::abi::NvmeQueueView queue = fixture.queue.download();
  const nta::abi::ObjectEntry object = fixture.objects.download();
  const nta::abi::WorkTicket workTicket = fixture.workTickets.download();
  const nta::abi::BackendView backend = fixture.backends.download(
      static_cast<std::uint32_t>(nta::abi::SourceKind::Nvme));
  require(fixture.intentPool.download().active == 0 &&
              backend.pendingAcquisitions == 0 && queue.outstanding == 0 &&
              queue.failed == 1 && queue.error != 0,
          "stale NVMe intent was not retired");
  require(object.objectId == QueueFixture::ObjectId + 1 &&
              object.version == 4 &&
              object.state ==
                  static_cast<std::uint32_t>(nta::abi::ObjectState::New),
          "stale NVMe intent modified the replacement object");
  require(workTicket.generation == 8 &&
              workTicket.state == static_cast<std::uint32_t>(
                                      nta::abi::WorkTicketState::Pending),
          "stale NVMe intent modified the replacement workTicket");
}

void verifyOfflineFailure(QueueFixture &fixture, const KernelModule &kernels,
                          const nta::FinitePhaseProgram &phases,
                          CUstream stream) {
  fixture.resetForOfflineQueue();
  kernels.initial(stream, fixture.runtime.get(), fixture.tasks.get(),
                  fixture.output.get());
  checkDriver(cuStreamSynchronize(stream),
              "synchronize fatal fallback publication");
  require(fixture.intentPool.download().active == 1 &&
              fixture.backends
                      .download(static_cast<std::uint32_t>(
                          nta::abi::SourceKind::Nvme))
                      .pendingAcquisitions == 1,
          "fatal test did not begin with one queued fallback intent");
  fixture.makeQueueFatalAndReleaseLease();
  phases.progressNvme(stream, fixture.runtime.get(), 1, 1);
  checkDriver(cuStreamSynchronize(stream), "synchronize offline queue failure");

  const nta::abi::NvmeQueueView queue = fixture.queue.download();
  const nta::abi::BackendView backend = fixture.backends.download(
      static_cast<std::uint32_t>(nta::abi::SourceKind::Nvme));
  require(queue.active == 0 && queue.ownerLock == 0 && queue.error != 0,
          "fatal queue was not taken offline and unlocked");
  require(fixture.intentPool.download().active == 0 &&
              backend.pendingAcquisitions == 0,
          "fatal queue left a scheduled NVMe intent stranded");
  require(fixture.objects.download().state ==
                  static_cast<std::uint32_t>(nta::abi::ObjectState::Failed) &&
              fixture.workTickets.download().state ==
                  static_cast<std::uint32_t>(nta::abi::WorkTicketState::Failed),
          "fatal queue did not fail dependent object and workTicket state");
}

} // namespace

int main() {
  try {
    checkDriver(cuInit(0), "cuInit");
    checkCuda(cudaSetDevice(0), "cudaSetDevice");
    KernelModule kernels;
    nta::FinitePhaseProgram phases(kernels.module());
    CUstream stream = nullptr;
    checkDriver(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING),
                "cuStreamCreate");
    QueueFixture directFixture;
    verifyDirectPath(directFixture, kernels, phases, stream);
    QueueFixture fallbackFixture;
    verifyFallbackPath(fallbackFixture, kernels, phases, stream);
    QueueFixture staleFixture;
    verifyStaleCompletionIsolation(staleFixture, kernels, phases, stream);
    QueueFixture statusFixture;
    verifyNvmeStatusFailure(statusFixture, kernels, phases, stream);
    QueueFixture malformedFixture;
    verifyMalformedCompletionFailure(malformedFixture, kernels, phases, stream);
    QueueFixture staleIntentFixture;
    verifyStaleIntentIsolation(staleIntentFixture, kernels, phases, stream);
    QueueFixture offlineFixture;
    verifyOfflineFailure(offlineFixture, kernels, phases, stream);
    QueueFixture concurrentFixture;
    verifyConcurrentDirectPath(concurrentFixture, kernels, stream);
    checkDriver(cuStreamDestroy(stream), "cuStreamDestroy");
    std::cout << "CTA NVMe try-issue and bounded fallback validation passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "CTA NVMe try-issue validation failed: " << error.what()
              << '\n';
    return 1;
  }
}
