#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace nta::abi {

inline constexpr std::uint32_t Version = 2;
inline constexpr std::uint32_t InvalidIndex = 0xffffffffU;

enum class SourceKind : std::uint32_t {
  Hbm = 0,
  HostMapped = 1,
  HostStaged = 2,
  Nvme = 3,
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
  std::uint32_t generation;
  std::uint32_t tenantId;
  std::uint32_t priority;
  std::uint32_t cancelled;
};
static_assert(sizeof(RequestContext) == 32);

struct alignas(64) ObjectEntry {
  std::uint64_t objectId;
  std::uint64_t sourceAddress;
  std::uint64_t stagingAddress;
  std::uint64_t bytes;
  std::uint64_t dmaPageListAddress;
  std::uint64_t issueCount;
  std::uint32_t version;
  std::uint32_t sourceKind;
  std::uint32_t state;
  std::uint32_t dmaPageCount;
};
static_assert(sizeof(ObjectEntry) == 64);

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
  std::uint64_t reserved0;
  std::uint64_t reserved1;
};
static_assert(sizeof(AcquireIntent) == 64);

struct alignas(32) Continuation {
  std::uint64_t requestId;
  std::uint32_t requestSlot;
  std::uint32_t generation;
  std::uint32_t state;
  std::uint32_t dependencyCount;
  std::uint32_t logicalTile;
  std::uint32_t reserved;
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
  std::uint32_t objectSlot;
  std::uint32_t objectVersion;
  std::uint32_t requestSlot;
  std::uint32_t generation;
  std::uint32_t continuation;
  std::uint32_t intentTicket;
};
static_assert(sizeof(NvmeCommandContext) == 32);

struct alignas(64) NvmeQueueView {
  NvmeSubmission *submissions;
  NvmeCompletion *completions;
  std::uint64_t *prpLists;
  std::uint64_t prpListDmaAddress;
  volatile std::uint32_t *sqDoorbell;
  volatile std::uint32_t *cqDoorbell;
  NvmeCommandContext *contexts;
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
  std::uint32_t reserved0;
  std::uint64_t submitted;
  std::uint64_t completed;
  std::uint64_t failed;
  std::uint64_t reserved1;
};
static_assert(sizeof(NvmeQueueView) == 192);

struct alignas(64) RuntimeView {
  RequestContext *requests;
  ObjectEntry *objects;
  AcquireIntent *intents;
  Continuation *continuations;
  std::uint32_t *intentCount;
  NvmeQueueView *nvme;
  std::uint32_t requestCapacity;
  std::uint32_t objectCapacity;
  std::uint32_t intentCapacity;
  std::uint32_t continuationCapacity;
  std::uint32_t abiVersion;
  std::uint32_t reserved0;
};
static_assert(sizeof(RuntimeView) == 128);

static_assert(std::is_standard_layout_v<RequestContext>);
static_assert(std::is_standard_layout_v<ObjectEntry>);
static_assert(std::is_standard_layout_v<AcquireIntent>);
static_assert(std::is_standard_layout_v<Continuation>);
static_assert(std::is_standard_layout_v<NvmeQueueView>);
static_assert(std::is_standard_layout_v<RuntimeView>);
static_assert(std::is_trivially_copyable_v<RuntimeView>);

} // namespace nta::abi
