#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace nta::abi {

inline constexpr std::uint32_t Version = 1;
inline constexpr std::uint32_t InvalidIndex = 0xffffffffU;

enum class SourceKind : std::uint32_t {
  Hbm = 0,
  HostMapped = 1,
  HostStaged = 2,
};

enum class ObjectState : std::uint32_t {
  New = 0,
  Queued = 1,
  Ready = 2,
  Failed = 3,
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
  std::uint32_t version;
  std::uint32_t sourceKind;
  std::uint32_t state;
  std::uint32_t reserved;
  std::uint64_t issueCount;
  std::uint64_t reserved2;
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

struct alignas(64) RuntimeView {
  RequestContext *requests;
  ObjectEntry *objects;
  AcquireIntent *intents;
  Continuation *continuations;
  std::uint32_t *intentCount;
  std::uint32_t requestCapacity;
  std::uint32_t objectCapacity;
  std::uint32_t intentCapacity;
  std::uint32_t continuationCapacity;
  std::uint32_t abiVersion;
  std::uint32_t reserved0;
};
static_assert(sizeof(RuntimeView) == 64);

static_assert(std::is_standard_layout_v<RequestContext>);
static_assert(std::is_standard_layout_v<ObjectEntry>);
static_assert(std::is_standard_layout_v<AcquireIntent>);
static_assert(std::is_standard_layout_v<Continuation>);
static_assert(std::is_standard_layout_v<RuntimeView>);
static_assert(std::is_trivially_copyable_v<RuntimeView>);

} // namespace nta::abi
