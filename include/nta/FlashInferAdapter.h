#pragma once

#include "nta/WorkPlan.h"

#include <cstdint>
#include <span>
#include <vector>

namespace nta::flashinfer {

using RequestBinding = nta::RequestBinding;
using PageBinding = nta::ObjectBinding;

struct DecodeBatchView {
  std::uint32_t pageSize;
  std::span<const std::int32_t> kvIndptr;
  std::span<const std::int32_t> kvIndices;
  std::span<const std::int32_t> lastPageLen;
  std::span<const RequestBinding> requests;
  std::span<const PageBinding> physicalPages;
  std::uint32_t maxPagesPerWorkItem = 1;
};

struct DecodeChunk {
  std::uint32_t requestIndex;
  std::uint32_t requestSlot;
  std::uint32_t generation;
  std::uint32_t pageTableOffset;
  std::uint32_t physicalPage;
  std::uint32_t tokenCount;
  std::uint32_t workTicket;
  PageBinding page;
};

struct RequestChunks {
  std::uint32_t chunkBegin;
  std::uint32_t chunkCount;
  std::uint32_t requestSlot;
  std::uint32_t generation;
};

struct DecodePlan {
  nta::WorkPlan work;
  std::vector<DecodeChunk> chunks;
  std::vector<RequestChunks> requests;
};

struct DecodeScheduleView {
  std::span<const std::int32_t> requestIndices;
  std::span<const std::int32_t> kvTileIndices;
  std::uint32_t kvChunkTokens;
};

// Converts FlashInfer's public paged-KV CSR representation into one finite NTA
// work ticket per logical KV chunk. Physical page reuse and arbitrary
// page-table order are preserved.
DecodePlan planDecode(const DecodeBatchView &batch);

// Uses schedule identity extracted by a version-specific frontend and rejects
// any mismatch with the engine-neutral plan. Active scheduler entries must be
// supplied without CUDA-graph padding.
DecodePlan planScheduledDecode(const DecodeBatchView &batch,
                               const DecodeScheduleView &schedule);

} // namespace nta::flashinfer
