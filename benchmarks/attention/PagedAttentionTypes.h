#pragma once

#include <cstdint>

namespace nta::benchmark {

inline constexpr std::uint32_t AttentionHeadDimension = 128;
inline constexpr std::uint32_t AttentionPageTokens = 16;

enum AttentionPageFlags : std::uint32_t {
  AttentionPageNeedsStaging = 1U << 0,
};

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

// Immutable device catalog entry used by query-dependent sparse attention.
// Compact summaries stay resident; the selected full page may be direct or
// command-addressed through the common acquisition directory.
struct alignas(64) AttentionPageDescriptor {
  std::uint64_t directBase;
  std::uint64_t sourceBase;
  std::uint64_t consumeBase;
  std::uint64_t objectId;
  std::uint32_t objectSlot;
  std::uint32_t objectVersion;
  std::uint32_t bytes;
  std::uint32_t tokenCount;
  std::uint32_t flags;
  std::uint32_t reserved;
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
static_assert(sizeof(AttentionPageDescriptor) == 64);
static_assert(sizeof(AttentionTilePartial) == 528);

} // namespace nta::benchmark
