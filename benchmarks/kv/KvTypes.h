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
  std::uint32_t continuation;
  std::uint64_t reserved0;
  std::uint64_t reserved1;
};
static_assert(sizeof(TileTask) == 64);

} // namespace nta::benchmark
