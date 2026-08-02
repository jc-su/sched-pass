#include "nta/WorkPlan.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace nta {
namespace {

std::uint32_t checkedSize(std::size_t size, const char *field) {
  if (size >= abi::InvalidIndex) {
    throw std::overflow_error(std::string(field) + " exceeds the NTA ABI");
  }
  return static_cast<std::uint32_t>(size);
}

} // namespace

WorkPlanBuilder::WorkPlanBuilder(std::uint32_t maxDependenciesPerWorkItem)
    : maxDependenciesPerWorkItem_(maxDependenciesPerWorkItem) {
  if (maxDependenciesPerWorkItem == 0) {
    throw std::invalid_argument(
        "max dependencies per work item must be non-zero");
  }
}

std::uint32_t WorkPlanBuilder::addRequest(RequestBinding request) {
  if (finished_) {
    throw std::logic_error("cannot append to a finished work plan");
  }
  const std::uint32_t requestIndex =
      checkedSize(plan_.requests.size(), "request count");
  plan_.requests.push_back({checkedSize(plan_.workItems.size(), "work count"),
                            0, request.requestSlot, request.generation});
  return requestIndex;
}

std::uint32_t WorkPlanBuilder::addWork(
    std::uint32_t requestIndex, std::uint32_t logicalWork,
    std::span<const abi::AcquireRequirement> requirements,
    std::uint32_t estimatedComputeNs) {
  if (finished_) {
    throw std::logic_error("cannot append to a finished work plan");
  }
  if (requestIndex >= plan_.requests.size() ||
      requestIndex + 1U != plan_.requests.size()) {
    throw std::invalid_argument(
        "work items must be appended contiguously to the current request");
  }
  if (requirements.empty() ||
      requirements.size() > maxDependenciesPerWorkItem_) {
    throw std::invalid_argument(
        "work dependency count exceeds its configured bound");
  }
  for (const abi::AcquireRequirement &requirement : requirements) {
    if (requirement.bytes == 0 || requirement.flags != 0) {
      throw std::invalid_argument(
          "work dependencies need non-zero bytes and supported flags");
    }
  }

  RequestWorkRange &range = plan_.requests[requestIndex];
  if (range.workCount == std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("request work count exceeds the NTA ABI");
  }
  const std::uint32_t workTicket =
      checkedSize(plan_.workItems.size(), "work count");
  const std::uint32_t dependencyBegin =
      checkedSize(plan_.dependencies.size(), "dependency count");
  if (requirements.size() >
      std::numeric_limits<std::uint32_t>::max() - dependencyBegin) {
    throw std::overflow_error("dependency count exceeds the NTA ABI");
  }
  plan_.dependencies.insert(plan_.dependencies.end(), requirements.begin(),
                            requirements.end());
  std::uint32_t directDependencyCount = 0;
  for (const abi::AcquireRequirement &requirement : requirements) {
    directDependencyCount += requirement.directBase != 0 ? 1U : 0U;
  }
  plan_.workItems.push_back({
      requestIndex,
      range.requestSlot,
      range.generation,
      logicalWork,
      dependencyBegin,
      static_cast<std::uint32_t>(requirements.size()),
      directDependencyCount,
      workTicket,
      requestIndex,
      range.workCount,
      0,
      estimatedComputeNs,
      0,
      0,
      0,
      0,
  });
  ++range.workCount;
  return workTicket;
}

WorkPlan WorkPlanBuilder::finish() {
  if (finished_) {
    throw std::logic_error("work plan was already finished");
  }
  finished_ = true;
  for (const RequestWorkRange &request : plan_.requests) {
    if (request.workCount == 0) {
      throw std::invalid_argument("every request needs at least one work item");
    }
    for (std::uint32_t offset = 0; offset < request.workCount; ++offset) {
      plan_.workItems[request.workBegin + offset].contributorCount =
          request.workCount;
    }
  }
  return std::move(plan_);
}

} // namespace nta
