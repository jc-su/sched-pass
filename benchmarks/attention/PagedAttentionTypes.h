#pragma once

#include <cstdint>

namespace nta::benchmark {

inline constexpr std::uint32_t AttentionHeadDimension = 128;
inline constexpr std::uint32_t AttentionPageTokens = 16;

struct alignas(16) AttentionTileTask {
  std::uint64_t directBase;
  std::uint64_t directTensorMap;
  std::uint64_t objectId;
  std::uint32_t requestSlot;
  std::uint32_t generation;
  std::uint32_t objectSlot;
  std::uint32_t objectVersion;
  std::uint32_t bytes;
  std::uint32_t continuation;
  std::uint32_t requestIndex;
  std::uint32_t tokenCount;
};

struct alignas(16) AttentionRequest {
  std::uint32_t tileBegin;
  std::uint32_t tileCount;
  std::uint32_t requestSlot;
  std::uint32_t generation;
};

struct alignas(16) AttentionTilePartial {
  float maxLogit;
  float sumExp;
  std::uint32_t valid;
  std::uint32_t reserved;
  float numerator[AttentionHeadDimension];
};

static_assert(sizeof(AttentionTileTask) == 64);
static_assert(sizeof(AttentionRequest) == 16);
static_assert(sizeof(AttentionTilePartial) == 528);

} // namespace nta::benchmark
