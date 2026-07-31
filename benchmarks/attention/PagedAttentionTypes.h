#pragma once

#include <cstdint>

namespace nta::benchmark {

inline constexpr std::uint32_t AttentionHeadDimension = 128;
inline constexpr std::uint32_t AttentionPageTokens = 16;

struct alignas(16) AttentionTileTask {
  std::uint32_t objectSlot;
  std::uint32_t requestIndex;
  std::uint32_t tokenCount;
  std::uint32_t reserved;
};

struct alignas(16) AttentionRequest {
  std::uint32_t tileBegin;
  std::uint32_t tileCount;
  std::uint32_t requestSlot;
  std::uint32_t generation;
};

struct alignas(16) AttentionTilePartial {
  // FlashInfer cascade state: normalized attention output plus base-2 LSE.
  float lse;
  float reserved0;
  std::uint32_t valid;
  std::uint32_t reserved1;
  float output[AttentionHeadDimension];
};

static_assert(sizeof(AttentionTileTask) == 16);
static_assert(sizeof(AttentionRequest) == 16);
static_assert(sizeof(AttentionTilePartial) == 528);

} // namespace nta::benchmark
