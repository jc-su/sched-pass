#include "nta/FlashInferAdapter.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace nta::flashinfer {
namespace {

std::uint32_t checkedIndex(std::int32_t value, const char *field) {
  if (value < 0) {
    throw std::invalid_argument(std::string(field) + " must be non-negative");
  }
  return static_cast<std::uint32_t>(value);
}

} // namespace

DecodePlan planDecode(const DecodeBatchView &batch) {
  if (batch.pageSize == 0) {
    throw std::invalid_argument("FlashInfer page size must be non-zero");
  }
  if (batch.kvIndptr.size() != batch.requests.size() + 1U ||
      batch.lastPageLen.size() != batch.requests.size()) {
    throw std::invalid_argument(
        "FlashInfer CSR dimensions do not match the request batch");
  }
  if (batch.kvIndptr.empty() || batch.kvIndptr.front() != 0) {
    throw std::invalid_argument("FlashInfer kv_indptr must start at zero");
  }

  const std::uint32_t referencedPages =
      checkedIndex(batch.kvIndptr.back(), "FlashInfer kv_indptr");
  if (referencedPages > batch.kvIndices.size()) {
    throw std::invalid_argument(
        "FlashInfer kv_indptr exceeds the kv_indices buffer");
  }

  DecodePlan plan;
  plan.chunks.reserve(referencedPages);
  plan.requests.reserve(batch.requests.size());
  for (std::uint32_t requestIndex = 0; requestIndex < batch.requests.size();
       ++requestIndex) {
    const std::uint32_t begin =
        checkedIndex(batch.kvIndptr[requestIndex], "FlashInfer kv_indptr");
    const std::uint32_t end =
        checkedIndex(batch.kvIndptr[requestIndex + 1U], "FlashInfer kv_indptr");
    if (end < begin || end > referencedPages) {
      throw std::invalid_argument(
          "FlashInfer kv_indptr must be monotonic and in bounds");
    }

    const std::uint32_t pageCount = end - begin;
    if (pageCount == 0) {
      throw std::invalid_argument(
          "NTA FlashInfer decode requests need at least one KV page");
    }
    const std::uint32_t finalPageTokens = checkedIndex(
        batch.lastPageLen[requestIndex], "FlashInfer last_page_len");
    if (finalPageTokens == 0 || finalPageTokens > batch.pageSize) {
      throw std::invalid_argument(
          "FlashInfer last_page_len is invalid for the request page count");
    }

    const RequestBinding binding = batch.requests[requestIndex];
    const std::uint32_t chunkBegin =
        static_cast<std::uint32_t>(plan.chunks.size());
    for (std::uint32_t pageOffset = begin; pageOffset < end; ++pageOffset) {
      const std::uint32_t physicalPage =
          checkedIndex(batch.kvIndices[pageOffset], "FlashInfer kv_indices");
      if (physicalPage >= batch.physicalPages.size()) {
        throw std::invalid_argument(
            "FlashInfer kv_indices references an unbound physical page");
      }
      const PageBinding page = batch.physicalPages[physicalPage];
      if (page.bytes == 0) {
        throw std::invalid_argument(
            "every referenced FlashInfer page needs a complete NTA binding");
      }
      if (plan.chunks.size() >=
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::overflow_error("FlashInfer chunk count exceeds the NTA ABI");
      }
      const std::uint32_t continuation =
          static_cast<std::uint32_t>(plan.chunks.size());
      plan.chunks.push_back({
          requestIndex,
          binding.requestSlot,
          binding.generation,
          pageOffset,
          physicalPage,
          pageOffset + 1U == end ? finalPageTokens : batch.pageSize,
          continuation,
          page,
      });
    }
    plan.requests.push_back(
        {chunkBegin, pageCount, binding.requestSlot, binding.generation});
  }
  return plan;
}

} // namespace nta::flashinfer
