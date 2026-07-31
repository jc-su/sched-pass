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
  std::uint32_t continuation;
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

// Converts FlashInfer's public paged-KV CSR representation into one finite NTA
// continuation per logical KV page. Physical page reuse and arbitrary
// page-table order are preserved; no FlashInfer private PlanInfo layout is
// inspected.
DecodePlan planDecode(const DecodeBatchView &batch);

} // namespace nta::flashinfer
