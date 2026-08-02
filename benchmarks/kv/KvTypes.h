#pragma once

#include <cstdint>

namespace nta::benchmark {

struct alignas(64) TileTask {
  std::uint64_t directBase;
  std::uint64_t objectId;
  std::uint64_t offset;
  std::uint32_t requestSlot;
  std::uint32_t generation;
  std::uint32_t objectSlot;
  std::uint32_t objectVersion;
  std::uint32_t bytes;
  std::uint32_t workTicket;
  std::uint64_t reserved0;
  std::uint64_t reserved1;
};
static_assert(sizeof(TileTask) == 64);

enum MoeExpertFlags : std::uint32_t {
  MoeExpertStaged = 1U << 0,
};

// Device-visible capabilities for one expert. The GPU router turns selected
// descriptors into canonical WorkItem and AcquireRequirement records.
struct alignas(64) MoeExpertDescriptor {
  std::uint64_t directBase;
  std::uint64_t consumeBase;
  std::uint64_t sourceBase;
  std::uint64_t objectId;
  std::uint32_t objectSlot;
  std::uint32_t objectVersion;
  std::uint32_t bytes;
  std::uint32_t flags;
  std::uint64_t reserved0;
  std::uint64_t reserved1;
};
static_assert(sizeof(MoeExpertDescriptor) == 64);

} // namespace nta::benchmark
