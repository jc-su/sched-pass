#include "nta/WorkPlan.h"

#include <array>
#include <functional>
#include <iostream>
#include <stdexcept>

namespace {

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
  nta::WorkPlanBuilder builder(3);
  const std::uint32_t first = builder.addRequest({4, 9});
  const std::array<nta::abi::AcquireRequirement, 2> graphShards{{
      {0, 0, 1001, 0, 7, 2, 4096, 0},
      {0x10000, 0, 1002, 128, 8, 3, 2048, 0},
  }};
  const std::uint32_t continuation = builder.addWork(first, 31, graphShards);
  nta::WorkPlan plan = builder.finish();

  bool ok = continuation == 0 && plan.requests.size() == 1 &&
        plan.workItems.size() == 1 && plan.dependencies.size() == 2;
  ok &= plan.workItems[0].requestSlot == 4 &&
        plan.workItems[0].generation == 9 &&
        plan.workItems[0].logicalWork == 31 &&
        plan.workItems[0].dependencyCount == 2 &&
        plan.workItems[0].directDependencyCount == 1;
  ok &= plan.dependencies[1].directBase == 0x10000 &&
        plan.dependencies[1].offset == 128;

  ok &= rejects([] { nta::WorkPlanBuilder invalid(0); });
  ok &= rejects([] {
    nta::WorkPlanBuilder invalid(1);
    const std::uint32_t request = invalid.addRequest({0, 1});
    const std::array<nta::abi::AcquireRequirement, 2> dependencies{{
        {0, 0, 1, 0, 0, 1, 4, 0},
        {0, 0, 2, 0, 1, 1, 4, 0},
    }};
    (void)invalid.addWork(request, 0, dependencies);
  });
  ok &= rejects([] {
    nta::WorkPlanBuilder invalid(1);
    (void)invalid.addRequest({0, 1});
    (void)invalid.finish();
  });
  ok &= rejects([] {
    const nta::ObjectBinding object{0, 0, 1, 0, 1, 4096};
    (void)nta::makeRequirement(object, 4090, 16);
  });

  if (!ok) {
    std::cerr << "engine-neutral work-plan validation failed\n";
    return 1;
  }
  std::cout << "engine-neutral work-plan validation passed\n";
  return 0;
}
