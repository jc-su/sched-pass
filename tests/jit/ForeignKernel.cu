#include "nta/KernelPolicy.cuh"
#include "runtime/device/Acquire.cuh"

extern "C" __global__ void nta_foreign_kernel(
    nta::abi::RuntimeView *runtime, const nta::abi::WorkItem *workItems,
    const nta::abi::AcquireRequirement *requirements,
    std::uint32_t workCount, float *output) {
  const std::uint32_t workIndex = blockIdx.x;
  if (workIndex >= workCount) {
    return;
  }
  nta::kernel::WorkContext work{};
  if (!nta::kernel::acquireWork(runtime, workItems, requirements, workIndex,
                                work)) {
    nta::kernel::defer(runtime, work);
    return;
  }
  const auto *input =
      static_cast<const float *>(nta::kernel::address(runtime, work, 0));
  if (input == nullptr) {
    nta::device::failWorkTicket(runtime, work.item.workTicket,
                                  nta::abi::WorkTicketState::Failed);
    return;
  }
  if (threadIdx.x == 0) {
    output[work.item.logicalWork] = input[0];
  }
}
