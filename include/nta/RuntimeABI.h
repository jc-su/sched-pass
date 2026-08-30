#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace nta::abi {

inline constexpr std::uint32_t Version = 37;
inline constexpr std::uint32_t InvalidIndex = 0xffffffffU;
inline constexpr std::uint32_t BackendCount = 6;

enum class SourceKind : std::uint32_t {
  Hbm = 0,
  HostMapped = 1,
  HostStaged = 2,
  Nvme = 3,
  Cxl = 4,
  Rdma = 5,
};

enum ReplicaFlags : std::uint32_t {
  ReplicaDirect = 1U << 0,
  ReplicaTransport = 1U << 1,
  // Host-staged replica whose source and destination are indexed rows. The
  // two index arrays contain uint32_t entries; transferShape packs the source
  // and destination row strides in bytes.
  ReplicaIndexed = 1U << 2,
  // The current index arrays were checked against both directory bounds by a
  // stream-ordered validation kernel. Address rebinding does not invalidate
  // this bit because the index arrays and transfer geometry are unchanged.
  ReplicaIndicesValidated = 1U << 3,
  // The transport target is CUDA HBM pinned through NVIDIA's persistent
  // peer-memory API and mapped for the VFIO-owned NVMe function. Consumers can
  // use normal HBM loads in this mode.
  ReplicaDmaHbm = 1U << 4,
};

enum BackendFlags : std::uint32_t {
  // A finite application CTA may attempt one transport submission before it
  // publishes an intent. Completion processing always remains out of line.
  BackendCtaTryIssue = 1U << 0,
  // The backend exposes device-visible state that can be consumed directly
  // by a typed acquisition site. This is distinct from CTA submission: a
  // mapped CXL replica is direct, while an NVMe queue is device-initiated.
  BackendDeviceVisible = 1U << 1,
  // Bits 8..12 carry the shared TierCapability mask. Keeping the mask in
  // the existing fixed-size directory entry makes tier qualification visible
  // to device admission without changing the CUDA ABI layout.
  BackendTierCapabilityShift = 8,
  BackendTierCapabilityMask = 0x1fU << BackendTierCapabilityShift,
};

enum class ObjectState : std::uint32_t {
  New = 0,
  Queued = 1,
  Issued = 2,
  Ready = 3,
  Failed = 4,
};

enum class WorkTicketState : std::uint32_t {
  New = 0,
  Pending = 1,
  Ready = 2,
  Done = 3,
  Cancelled = 4,
  Failed = 5,
  Initializing = 6,
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

// Per-tenant staging isolation.  The host owns maxOutstandingBytes; device
// acquisition paths atomically own outstandingBytes.  A zero maximum disables
// acquisition for the tenant without a second, potentially inconsistent
// active flag.
struct alignas(16) TenantContext {
  std::uint64_t maxOutstandingBytes;
  std::uint64_t outstandingBytes;
};
static_assert(sizeof(TenantContext) == 16);

struct alignas(32) RequestProgress {
  std::uint64_t requestId;
  std::uint32_t generation;
  std::uint32_t expectedWork;
  std::uint32_t pendingWork;
  std::uint32_t runnableWork;
  std::uint32_t completedWork;
  std::uint32_t failedWork;
  std::uint32_t cancelledWork;
  std::uint32_t epoch;
  std::uint64_t unavailableBytes;
  std::uint64_t runnableComputeNs;
  std::uint64_t completedComputeNs;
  // Compiler-attributed service still blocked on external data and total
  // service represented by this request's contributors in the current epoch.
  // These make progress useful to an SLO policy without treating CTA count as
  // a proxy for heterogeneous work.
  std::uint64_t pendingComputeNs;
  std::uint64_t expectedComputeNs;
  // Attributions rejected because a slot, generation, or epoch changed before
  // publication. A nonzero value makes conservation failure observable.
  std::uint64_t droppedAttributions;
};
static_assert(sizeof(RequestProgress) == 96);

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
  // Indexed host replicas pack row strides here. NVMe replicas store the
  // controller-page offset of PRP1; dmaPageListAddress always names aligned
  // page bases, which lets registered HBM regions share one immutable table.
  std::uint64_t transferShape;
};
static_assert(sizeof(ReplicaEntry) == 64);

[[nodiscard]] constexpr std::uint64_t
packTransferStrides(std::uint32_t sourceStride,
                    std::uint32_t destinationStride) noexcept {
  return static_cast<std::uint64_t>(sourceStride) |
         (static_cast<std::uint64_t>(destinationStride) << 32U);
}

[[nodiscard]] constexpr std::uint32_t
sourceTransferStride(std::uint64_t shape) noexcept {
  return static_cast<std::uint32_t>(shape);
}

[[nodiscard]] constexpr std::uint32_t
destinationTransferStride(std::uint64_t shape) noexcept {
  return static_cast<std::uint32_t>(shape >> 32U);
}

[[nodiscard]] constexpr std::uint64_t
packTransferIndexLimits(std::uint32_t sourceLimit,
                        std::uint32_t destinationLimit) noexcept {
  return static_cast<std::uint64_t>(sourceLimit) |
         (static_cast<std::uint64_t>(destinationLimit) << 32U);
}

[[nodiscard]] constexpr std::uint32_t
sourceTransferIndexLimit(std::uint64_t limits) noexcept {
  return static_cast<std::uint32_t>(limits);
}

[[nodiscard]] constexpr std::uint32_t
destinationTransferIndexLimit(std::uint64_t limits) noexcept {
  return static_cast<std::uint32_t>(limits >> 32U);
}

struct alignas(64) BackendView {
  std::uint64_t deviceState;
  std::uint64_t estimatedLatencyNs;
  std::uint64_t estimatedBandwidthBytesPerSecond;
  std::uint64_t outstandingBytes;
  std::uint64_t maxOutstandingBytes;
  std::uint32_t sourceKind;
  std::uint32_t active;
  std::uint32_t backendIndex;
  std::uint32_t flags;
  std::uint64_t pendingAcquisitions;
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
  std::uint32_t workTicket;
  std::uint32_t valid;
  std::uint32_t priority;
  std::uint32_t tenantId;
  std::uint64_t deadlineClock;
};
static_assert(sizeof(AcquireIntent) == 64);

struct alignas(128) IntentSlot {
  AcquireIntent intent;
  std::uint64_t sequence;
  std::uint32_t sourceKind;
  std::uint32_t epoch;
  // Credits are reserved at dispatch but released by a later kernel for
  // prevalidated indexed ranges. Persist the actual reservation rather than
  // inferring it from request liveness at completion: cancellation may change
  // liveness while the transfer is in flight.
  std::uint64_t chargedRequestBytes;
  std::uint64_t chargedBackendBytes;
  std::uint64_t reserved[4];
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

enum class IntentQueueState : std::uint32_t {
  Free = 0,
  Queued = 1,
  Claimed = 2,
  Preparing = 3,
};

// Immutable ordering key for one publication of an intent slot. Timed work is
// ordered by absolute deadline first. Equal deadlines use caller priority,
// critical service, and stableSequence. deadlineClock == 0 is explicitly
// best-effort and follows all timed work. intentSequence binds this record to
// the current IntentSlot lifetime and prevents stale heap nodes from acquiring
// a reused slot.
struct alignas(64) IntentQueueEntry {
  std::uint64_t intentSequence;
  std::uint64_t stableSequence;
  std::uint64_t deadlineClock;
  std::uint64_t criticalServiceNs;
  std::uint32_t heapIndex;
  std::uint32_t state;
  std::uint32_t epoch;
  std::uint32_t sourceKind;
  std::uint32_t priority;
  std::uint32_t reserved[3];
};
static_assert(sizeof(IntentQueueEntry) == 64);

// Heap nodes retain the complete slot lifetime rather than only an index. A
// node left behind by cancellation or epoch retirement is therefore a bounded
// tombstone, never an ABA alias for a later intent in the same slot.
struct alignas(16) IntentQueueNode {
  std::uint64_t intentSequence;
  std::uint32_t slotIndex;
  std::uint32_t reserved;
};
static_assert(sizeof(IntentQueueNode) == 16);

// One control record per backend. Queue operations are device-linearized by
// lock; storage is a fixed intentCapacity-node heap per backend. Constrained
// dispatch may reserve byte credits while holding this lock so that EDF is
// evaluated over the actually feasible set. The lock is never held across a
// data movement or I/O operation.
struct alignas(32) IntentQueueControl {
  std::uint64_t nextSequence;
  std::uint32_t size;
  std::uint32_t lock;
  std::uint64_t reserved[2];
};
static_assert(sizeof(IntentQueueControl) == 32);

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

struct alignas(16) WorkDependency {
  std::uint64_t objectId;
  std::uint32_t objectSlot;
  std::uint32_t objectVersion;
};
static_assert(sizeof(WorkDependency) == 16);

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
  std::uint32_t workTicket;
  std::uint32_t reductionGroup;
  std::uint32_t contributorIndex;
  std::uint32_t contributorCount;
  std::uint32_t estimatedComputeNs;
  // Readiness deadline relative to RuntimeView::epochStartClock. Zero keeps
  // the request-level absolute deadline. A positive value lets a producer
  // describe transformer-layer arrival order without translating the GPU
  // global timer into a host clock domain.
  std::uint64_t readyDeadlineOffsetNs;
  // Explicit tail padding keeps the C, Python, and device-array stride equal.
  // A non-zero value denotes an ABI extension this runtime cannot interpret.
  std::uint32_t reserved2;
  std::uint32_t reserved3;
};
static_assert(sizeof(WorkItem) == 64);

struct alignas(32) WorkTicket {
  std::uint64_t requestId;
  std::uint32_t requestSlot;
  std::uint32_t generation;
  std::uint32_t state;
  std::uint32_t dependencyCount;
  std::uint32_t logicalTile;
  std::uint32_t dependencyStart;
  // Operation identity is distinct from request-slot generation. The same
  // request and ticket index may be reused by consecutive transformer layers.
  std::uint32_t epoch;
  std::uint64_t unavailableBytes;
  std::uint64_t estimatedComputeNs;
  std::uint32_t reductionGroup;
  std::uint32_t contributorCount;
};
static_assert(sizeof(WorkTicket) == 64);

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
  std::uint64_t mappingKey;
  std::uint32_t objectSlot;
  std::uint32_t objectVersion;
  std::uint32_t requestSlot;
  std::uint32_t generation;
  std::uint32_t workTicket;
  std::uint32_t tenantId;
  std::uint32_t epoch;
  std::uint32_t active;
};
static_assert(sizeof(NvmeCommandContext) == 64);

inline constexpr std::uint32_t NvmeQueueControlMagic = 0x4e544151U;
inline constexpr std::uint32_t NvmeQueueAbiVersion = 2U;

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
  std::uint32_t ownerLock;
  std::uint32_t active;
  std::uint32_t error;
  std::uint32_t cidCursor;
  std::uint32_t queueGeneration;
  std::uint32_t queueId;
  std::uint64_t submitted;
  std::uint64_t completed;
  std::uint64_t failed;
  std::uint64_t directSubmitted;
  std::uint64_t directFallbacks;
  std::uint32_t directMaxPrpPages;
  std::uint32_t reserved1;
};
static_assert(sizeof(NvmeQueueView) == 192);

struct alignas(64) RuntimeView {
  RequestContext *requests;
  TenantContext *tenants;
  ObjectEntry *objects;
  ReplicaEntry *replicas;
  BackendView *backends;
  IntentSlot *intents;
  WorkTicket *workTickets;
  // Relative device-global nanoseconds at which each ticket first became
  // runnable in the current epoch. Zero denotes work available at epoch start.
  std::uint64_t *workRunnableNs;
  WorkDependency *dependencies;
  IntentPool *intentPool;
  IntentQueueEntry *intentQueueEntries;
  IntentQueueControl *intentQueueControls;
  IntentQueueNode *intentQueueHeap;
  std::uint32_t *readyWorkTickets;
  std::uint32_t *readyCount;
  std::uint32_t *readyHead;
  std::uint32_t *pendingWorkTickets;
  std::uint32_t *pendingCount;
  // Number of application CTAs that retired each canonical work ticket in the
  // current epoch. Framework kernels with a head dimension use this to let the
  // last sibling CTA publish Done without an extra completion kernel.
  std::uint32_t *ctaCompletions;
  // Reverse dependency index for completion-driven runnable-work discovery.
  // Each object head points into the fixed per-ticket dependency array; the
  // ticket owning an edge is dependency_index / maxDependenciesPerWorkTicket.
  std::uint32_t *objectDependentHeads;
  std::uint32_t *dependencyNext;
  // Set exactly once when an object transition satisfies this dependency.
  // This closes the race between reverse-edge installation and completion.
  std::uint32_t *dependencySatisfied;
  std::uint32_t *remainingDependencies;
  // Deferred publication queue for integrations that separate dependency
  // completion from runnable publication. Built-in backends publish directly;
  // the explicit publish phase consumes this queue or the bounded pending
  // index.
  std::uint32_t *changedWorkTickets;
  // Per-ticket deferred-publication membership bit.
  std::uint32_t *changedQueued;
  std::uint32_t *changedCount;
  std::uint32_t *changedOverflow;
  RequestProgress *requestProgress;
  // Per-request reduction state. Work items in one request share a reduction
  // group, allowing split-work merges to proceed independently across requests.
  std::uint32_t *reductionExpected;
  std::uint32_t *reductionCompleted;
  std::uint32_t *reductionFailed;
  std::uint32_t requestCapacity;
  std::uint32_t tenantCapacity;
  std::uint32_t objectCapacity;
  std::uint32_t replicaCapacity;
  std::uint32_t backendCapacity;
  std::uint32_t intentCapacity;
  std::uint32_t workTicketCapacity;
  std::uint32_t dependencyCapacity;
  std::uint32_t maxDependenciesPerWorkTicket;
  // Frozen exclusive end of the runnable queue window consumed by the current
  // numerical launch.  The compute stream snapshots readyCount here before a
  // launch, while the progress stream remains free to append later arrivals.
  // This field intentionally occupies the former alignment padding before the
  // 64-bit epoch clock, so ABI v33 adds semantics without growing RuntimeView.
  std::uint32_t readyWindowEnd;
  std::uint64_t epochStartClock;
  // Device-owned epoch and terminal counters. reset_epoch advances epoch after
  // clearing all ticket records; application and progress kernels reject work
  // whose ticket/intent/transport context belongs to a different operation.
  std::uint32_t epoch;
  std::uint32_t completedCount;
  std::uint32_t failedCount;
  std::uint32_t abiVersion;
  // Runtime-lifetime failure sequence; epoch reset deliberately preserves it.
  std::uint32_t stickyFailedCount;
};
static_assert(sizeof(RuntimeView) == 320);

static_assert(std::is_standard_layout_v<RequestContext>);
static_assert(std::is_standard_layout_v<TenantContext>);
static_assert(std::is_standard_layout_v<RequestProgress>);
static_assert(std::is_standard_layout_v<ObjectEntry>);
static_assert(std::is_standard_layout_v<ReplicaEntry>);
static_assert(std::is_standard_layout_v<BackendView>);
static_assert(std::is_standard_layout_v<AcquireIntent>);
static_assert(std::is_standard_layout_v<IntentSlot>);
static_assert(std::is_standard_layout_v<IntentPool>);
static_assert(std::is_standard_layout_v<IntentQueueEntry>);
static_assert(std::is_standard_layout_v<IntentQueueNode>);
static_assert(std::is_standard_layout_v<IntentQueueControl>);
static_assert(std::is_standard_layout_v<AcquireRequirement>);
static_assert(std::is_standard_layout_v<WorkDependency>);
static_assert(std::is_standard_layout_v<WorkItem>);
static_assert(std::is_standard_layout_v<WorkTicket>);
static_assert(std::is_standard_layout_v<NvmeQueueControl>);
static_assert(std::is_standard_layout_v<NvmeQueueView>);
static_assert(std::is_standard_layout_v<RuntimeView>);
static_assert(std::is_trivially_copyable_v<AcquireRequirement>);
static_assert(std::is_trivially_copyable_v<WorkDependency>);
static_assert(std::is_trivially_copyable_v<WorkItem>);
static_assert(std::is_trivially_copyable_v<RuntimeView>);

} // namespace nta::abi
