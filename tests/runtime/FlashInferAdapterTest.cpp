#include "nta/FlashInferAdapter.h"

#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using nta::flashinfer::DecodeBatchView;
using nta::flashinfer::DecodeScheduleView;
using nta::flashinfer::PageBinding;
using nta::flashinfer::RequestBinding;

bool rejects(const std::function<void()> &operation) {
  try {
    operation();
  } catch (const std::invalid_argument &) {
    return true;
  }
  return false;
}

} // namespace

int main() {
  const std::vector<std::int32_t> indptr{0, 2, 3};
  const std::vector<std::int32_t> indices{2, 0, 2};
  const std::vector<std::int32_t> lastPageLen{5, 16};
  const std::vector<RequestBinding> requests{{7, 11}, {9, 13}};
  const std::vector<PageBinding> pages{
      {0x1000, 0x2000, 101, 3, 2, 8192},
      {0x3000, 0x4000, 102, 4, 2, 8192},
      {0x5000, 0x6000, 103, 5, 4, 8192},
  };
  const DecodeBatchView view{16, indptr, indices, lastPageLen, requests, pages};
  const nta::flashinfer::DecodePlan plan = nta::flashinfer::planDecode(view);

  bool ok = plan.requests.size() == 2 && plan.chunks.size() == 3;
  ok &= plan.work.requests.size() == 2 && plan.work.workItems.size() == 3 &&
        plan.work.dependencies.size() == 3;
  ok &= plan.requests[0].chunkBegin == 0 && plan.requests[0].chunkCount == 2;
  ok &= plan.requests[1].chunkBegin == 2 && plan.requests[1].chunkCount == 1;
  ok &= plan.chunks[0].physicalPage == 2 && plan.chunks[0].tokenCount == 16;
  ok &= plan.chunks[1].physicalPage == 0 && plan.chunks[1].tokenCount == 5;
  ok &= plan.chunks[2].physicalPage == 2 && plan.chunks[2].tokenCount == 16;
  ok &= plan.chunks[0].requestSlot == 7 && plan.chunks[2].requestSlot == 9;
  ok &= plan.chunks[0].continuation == 0 && plan.chunks[2].continuation == 2;
  ok &= plan.chunks[0].page.objectSlot == 5 &&
        plan.chunks[2].page.objectSlot == 5;
  ok &= plan.work.dependencies[0].objectSlot == 5 &&
        plan.work.workItems[2].requestSlot == 9;

  const DecodeBatchView groupedView{16,       indptr, indices, lastPageLen,
                                    requests, pages,  2};
  const nta::flashinfer::DecodePlan grouped =
      nta::flashinfer::planDecode(groupedView);
  ok &= grouped.work.workItems.size() == 2 &&
        grouped.work.workItems[0].dependencyCount == 2 &&
        grouped.work.workItems[0].directDependencyCount == 2 &&
        grouped.work.workItems[1].dependencyCount == 1;
  ok &= grouped.chunks[0].continuation == 0 &&
        grouped.chunks[1].continuation == 0 &&
        grouped.chunks[2].continuation == 1;

  const std::vector<std::int32_t> scheduledRequests{0, 1};
  const std::vector<std::int32_t> scheduledTiles{0, 0};
  const auto scheduled = nta::flashinfer::planScheduledDecode(
      view, DecodeScheduleView{scheduledRequests, scheduledTiles, 32});
  ok &= scheduled.work.workItems.size() == 2 &&
        scheduled.work.workItems[0].dependencyCount == 2;
  const std::vector<std::int32_t> wrongScheduledRequests{1, 0};
  ok &= rejects([&] {
    (void)nta::flashinfer::planScheduledDecode(
        view, DecodeScheduleView{wrongScheduledRequests, scheduledTiles, 32});
  });
  ok &= rejects([&] {
    (void)nta::flashinfer::planScheduledDecode(
        view, DecodeScheduleView{scheduledRequests, scheduledTiles, 17});
  });

  std::vector<std::int32_t> badIndptr = indptr;
  badIndptr[1] = 4;
  ok &= rejects([&] {
    (void)nta::flashinfer::planDecode(
        {16, badIndptr, indices, lastPageLen, requests, pages});
  });
  ok &= rejects([&] {
    (void)nta::flashinfer::planDecode(
        {16, indptr, indices, lastPageLen, requests, pages, 0});
  });
  std::vector<std::int32_t> badIndices = indices;
  badIndices[0] = -1;
  ok &= rejects([&] {
    (void)nta::flashinfer::planDecode(
        {16, indptr, badIndices, lastPageLen, requests, pages});
  });
  std::vector<std::int32_t> badLastPageLen = lastPageLen;
  badLastPageLen[0] = 17;
  ok &= rejects([&] {
    (void)nta::flashinfer::planDecode(
        {16, indptr, indices, badLastPageLen, requests, pages});
  });
  const std::vector<std::int32_t> emptyIndptr{0, 0};
  const std::vector<std::int32_t> emptyLastPageLen{0};
  ok &= rejects([&] {
    (void)nta::flashinfer::planDecode(
        {16,
         emptyIndptr,
         {},
         emptyLastPageLen,
         std::span<const RequestBinding>(requests.data(), 1),
         pages});
  });

  if (!ok) {
    std::cerr << "FlashInfer adapter validation failed\n";
    return 1;
  }
  std::cout << "FlashInfer adapter validation passed\n";
  return 0;
}
