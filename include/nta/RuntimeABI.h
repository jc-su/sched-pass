#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace nta::abi {

inline constexpr std::uint32_t Version = 10;
inline constexpr std::uint32_t InvalidIndex = 0xffffffffU;
inline constexpr std::uint32_t BackendCount = 5;

enum class SourceKind : std::uint32_t {
  Hbm = 0,
  HostMapped = 1,
  HostStaged = 2,
  Nvme = 3,
  Rdma = 4,
};

enum ReplicaFlags : std::uint32_t {
  ReplicaDirect = 1U << 0,
  ReplicaTransport = 1U << 1,
};

enum class ObjectState : std::uint32_t {
  New = 0,
  Queued = 1,
  Issued = 2,
  Ready = 3,
  Failed = 4,
};

enum class ContinuationState : std::uint32_t {
  New = 0,
  Pending = 1,
  Ready = 2,
  Done = 3,
  Cancelled = 4,
  Failed = 5,
};

struct alignas(32) RequestContext {
  std::uint64_t requestId;
  std::uint64_t deadlineClock;
  std::uint64_t maxOutstandingBytes;
  std::uint64_t outstandingBytes;
  std::uint32_t generation;
  std::uint32_t tenantId;
  std::uint32_t priority;
  std::uint32_t cancelled;
};
static_assert(sizeof(RequestContext) == 64);

struct alignas(32) TenantContext {
  std::uint64_t maxOutstandingBytes;
  std::uint64_t outstandingBytes;
  std::uint32_t weight;
  std::uint32_t active;
  std::uint64_t serviceBytes;
};
static_assert(sizeof(TenantContext) == 32);

struct alignas(64) ObjectEntry {
  std::uint64_t objectId;
  std::uint64_t stagingAddress;
  std::uint64_t bytes;
  std::uint64_t issueCount;
  std::uint32_t version;
  std::uint32_t state;
  std::uint32_t replicaStart;
  std::uint32_t replicaCount;
  std::uint32_t selectedReplica;
  std::uint32_t flags;
  std::uint64_t stagingTensorMapAddress;
};
static_assert(sizeof(ObjectEntry) == 64);

struct alignas(64) ReplicaEntry {
  std::uint64_t sourceAddress;
  std::uint64_t dmaPageListAddress;
  std::uint64_t estimatedLatencyNs;
  std::uint64_t estimatedBandwidthBytesPerSecond;
  std::uint32_t sourceKind;
  std::uint32_t dmaPageCount;
  std::uint32_t backendIndex;
  std::uint32_t flags;
  std::uint64_t tensorMapAddress;
  std::uint64_t reserved1;
};
static_assert(sizeof(ReplicaEntry) == 64);

struct alignas(64) BackendView {
  std::uint64_t deviceState;
  std::uint64_t estimatedLatencyNs;
  std::uint64_t estimatedBandwidthBytesPerSecond;
  std::uint64_t outstandingBytes;
  std::uint64_t maxOutstandingBytes;
  std::uint32_t sourceKind;
  std::uint32_t active;
  std::uint32_t backendIndex;
  std::uint32_t reserved0;
  std::uint64_t reserved1;
};
static_assert(sizeof(BackendView) == 64);

struct alignas(64) AcquireIntent {
  std::uint64_t objectId;
  std::uint64_t offset;
  std::uint64_t bytes;
  std::uint32_t requestSlot;
  std::uint32_t generation;
  std::uint32_t objectSlot;
  std::uint32_t objectVersion;
  std::uint32_t continuation;
  std::uint32_t valid;
  std::uint32_t priority;
  std::uint32_t tenantId;
  std::uint64_t deadlineClock;
};
static_assert(sizeof(AcquireIntent) == 64);

struct alignas(128) IntentSlot {
  AcquireIntent intent;
  std::uint64_t sequence;
  std::uint64_t reserved[7];
};
static_assert(sizeof(IntentSlot) == 128);

struct alignas(64) IntentPool {
  std::uint64_t enqueued;
  std::uint64_t consumed;
  std::uint32_t capacity;
  std::uint32_t active;
  std::uint32_t overflow;
  std::uint32_t reserved0;
  std::uint64_t reserved[4];
};
static_assert(sizeof(IntentPool) == 64);

// A kernel work item may require several independently resident objects. The
// compiler treats an array of these records as one finite deferral boundary.
struct alignas(16) AcquireRequirement {
  std::uint64_t directBase;
  std::uint64_t directTensorMap;
  std::uint64_t objectId;
  std::uint64_t offset;
  std::uint32_t objectSlot;
  std::uint32_t objectVersion;
  std::uint32_t bytes;
  std::uint32_t flags;
};
static_assert(sizeof(AcquireRequirement) == 48);

struct alignas(16) ContinuationDependency {
  std::uint64_t objectId;
  std::uint32_t objectSlot;
  std::uint32_t objectVersion;
};
static_assert(sizeof(ContinuationDependency) == 16);

// Canonical per-CTA work descriptor shared by every frontend and device
// kernel. Dependency records live in one batch-level contiguous array.
struct alignas(32) WorkItem {
  std::uint32_t requestIndex;
  std::uint32_t requestSlot;
  std::uint32_t generation;
  std::uint32_t logicalWork;
  std::uint32_t dependencyBegin;
  std::uint32_t dependencyCount;
  std::uint32_t directDependencyCount;
  std::uint32_t continuation;
};
static_assert(sizeof(WorkItem) == 32);

struct alignas(32) Continuation {
  std::uint64_t requestId;
  std::uint32_t requestSlot;
  std::uint32_t generation;
  std::uint32_t state;
  std::uint32_t dependencyCount;
  std::uint32_t logicalTile;
  std::uint32_t dependencyStart;
};
static_assert(sizeof(Continuation) == 32);

struct alignas(64) NvmeSubmission {
  std::uint32_t dword[16];
};
static_assert(sizeof(NvmeSubmission) == 64);

struct alignas(16) NvmeCompletion {
  std::uint32_t dword[4];
};
static_assert(sizeof(NvmeCompletion) == 16);

struct alignas(32) NvmeCommandContext {
  std::uint64_t objectId;
  std::uint64_t bytes;
  std::uint64_t backendBytes;
  std::uint32_t objectSlot;
  std::uint32_t objectVersion;
  std::uint32_t requestSlot;
  std::uint32_t generation;
  std::uint32_t continuation;
  std::uint32_t tenantId;
  std::uint32_t active;
};
static_assert(sizeof(NvmeCommandContext) == 64);

inline constexpr std::uint32_t NvmeQueueControlMagic = 0x4e544151U;
inline constexpr std::uint32_t NvmeDriverAbiVersion = 2U;

enum class NvmeQueueState : std::uint32_t {
  Offline = 0,
  Online = 1,
  Quiesced = 2,
  Fatal = 3,
  Removed = 4,
};

struct alignas(64) NvmeQueueControl {
  std::uint32_t magic;
  std::uint32_t abiVersion;
  std::uint32_t state;
  std::uint32_t generation;
  std::uint32_t queueId;
  std::uint32_t reserved0;
  std::uint64_t reserved1[5];
};
static_assert(sizeof(NvmeQueueControl) == 64);

struct alignas(64) NvmeQueueView {
  NvmeSubmission *submissions;
  NvmeCompletion *completions;
  std::uint64_t *prpLists;
  std::uint64_t prpListDmaAddress;
  volatile std::uint32_t *sqDoorbell;
  volatile std::uint32_t *cqDoorbell;
  NvmeCommandContext *contexts;
  NvmeQueueControl *control;
  std::uint32_t depth;
  std::uint32_t controllerPageSize;
  std::uint32_t lbaShift;
  std::uint32_t namespaceId;
  std::uint32_t sqTail;
  std::uint32_t cqHead;
  std::uint32_t cqPhase;
  std::uint32_t outstanding;
  std::uint32_t intentCursor;
  std::uint32_t active;
  std::uint32_t error;
  std::uint32_t cidCursor;
  std::uint32_t queueGeneration;
  std::uint32_t queueId;
  std::uint64_t submitted;
  std::uint64_t completed;
  std::uint64_t failed;
  std::uint64_t reserved1;
};
static_assert(sizeof(NvmeQueueView) == 192);

struct alignas(64) RuntimeView {
  RequestContext *requests;
  TenantContext *tenants;
  ObjectEntry *objects;
  ReplicaEntry *replicas;
  BackendView *backends;
  IntentSlot *intents;
  Continuation *continuations;
  ContinuationDependency *dependencies;
  IntentPool *intentPool;
  std::uint32_t *readyContinuations;
  std::uint32_t *readyCount;
  std::uint32_t *readyHead;
  std::uint32_t *pendingContinuations;
  std::uint32_t *pendingCount;
  std::uint32_t requestCapacity;
  std::uint32_t tenantCapacity;
  std::uint32_t objectCapacity;
  std::uint32_t replicaCapacity;
  std::uint32_t backendCapacity;
  std::uint32_t intentCapacity;
  std::uint32_t continuationCapacity;
  std::uint32_t dependencyCapacity;
  std::uint32_t maxDependenciesPerContinuation;
  std::uint32_t abiVersion;
};
static_assert(sizeof(RuntimeView) == 192);

static_assert(std::is_standard_layout_v<RequestContext>);
static_assert(std::is_standard_layout_v<TenantContext>);
static_assert(std::is_standard_layout_v<ObjectEntry>);
static_assert(std::is_standard_layout_v<ReplicaEntry>);
static_assert(std::is_standard_layout_v<BackendView>);
static_assert(std::is_standard_layout_v<AcquireIntent>);
static_assert(std::is_standard_layout_v<IntentSlot>);
static_assert(std::is_standard_layout_v<IntentPool>);
static_assert(std::is_standard_layout_v<AcquireRequirement>);
static_assert(std::is_standard_layout_v<ContinuationDependency>);
static_assert(std::is_standard_layout_v<WorkItem>);
static_assert(std::is_standard_layout_v<Continuation>);
static_assert(std::is_standard_layout_v<NvmeQueueControl>);
static_assert(std::is_standard_layout_v<NvmeQueueView>);
static_assert(std::is_standard_layout_v<RuntimeView>);
static_assert(std::is_trivially_copyable_v<AcquireRequirement>);
static_assert(std::is_trivially_copyable_v<ContinuationDependency>);
static_assert(std::is_trivially_copyable_v<WorkItem>);
static_assert(std::is_trivially_copyable_v<RuntimeView>);

} // namespace nta::abi
