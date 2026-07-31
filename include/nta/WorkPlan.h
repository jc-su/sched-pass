#pragma once

#include "nta/RuntimeABI.h"

#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace nta {

// Engine and kernel-neutral host representation. Frontends translate their
// page tables, expert maps, graph shards, or application objects into this
// compact plan; kernels consume only AcquireRequirement arrays.
struct RequestBinding {
  std::uint32_t requestSlot;
  std::uint32_t generation;
};

struct ObjectBinding {
  std::uint64_t directBase;
  std::uint64_t directTensorMap;
  std::uint64_t objectId;
  std::uint32_t objectSlot;
  std::uint32_t objectVersion;
  std::uint32_t bytes;
};

using WorkItem = abi::WorkItem;

struct RequestWorkRange {
  std::uint32_t workBegin;
  std::uint32_t workCount;
  std::uint32_t requestSlot;
  std::uint32_t generation;
};

struct WorkPlan {
  std::vector<abi::AcquireRequirement> dependencies;
  std::vector<WorkItem> workItems;
  std::vector<RequestWorkRange> requests;
};

class WorkPlanBuilder {
public:
  explicit WorkPlanBuilder(std::uint32_t maxDependenciesPerWorkItem);

  std::uint32_t addRequest(RequestBinding request);
  std::uint32_t addWork(std::uint32_t requestIndex, std::uint32_t logicalWork,
                        std::span<const abi::AcquireRequirement> requirements);
  WorkPlan finish();

private:
  std::uint32_t maxDependenciesPerWorkItem_;
  WorkPlan plan_;
  bool finished_ = false;
};

inline abi::AcquireRequirement makeRequirement(const ObjectBinding &object,
                                               std::uint64_t offset = 0,
                                               std::uint32_t bytes = 0) {
  if (offset > object.bytes) {
    throw std::invalid_argument("object requirement offset is out of bounds");
  }
  const std::uint64_t remaining = object.bytes - offset;
  const std::uint64_t selectedBytes = bytes == 0 ? remaining : bytes;
  if (selectedBytes == 0 || selectedBytes > remaining) {
    throw std::invalid_argument("object requirement range is out of bounds");
  }
  return {
      object.directBase,
      object.directTensorMap,
      object.objectId,
      offset,
      object.objectSlot,
      object.objectVersion,
      static_cast<std::uint32_t>(selectedBytes),
      0,
  };
}

} // namespace nta
