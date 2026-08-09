#include "nta/RuntimeABI.h"

#include <cstddef>
#include <iostream>
#include <type_traits>

int main() {
  using namespace nta::abi;

  static_assert(alignof(RequestContext) == 32);
  static_assert(alignof(TenantContext) == 32);
  static_assert(alignof(RequestProgress) == 32);
  static_assert(sizeof(RequestProgress) == 96);
  static_assert(offsetof(RequestProgress, pendingComputeNs) == 64);
  static_assert(offsetof(RequestProgress, expectedComputeNs) == 72);
  static_assert(offsetof(RequestProgress, droppedAttributions) == 80);
  static_assert(alignof(ObjectEntry) == 64);
  static_assert(alignof(ReplicaEntry) == 64);
  static_assert(alignof(BackendView) == 64);
  static_assert(alignof(AcquireIntent) == 64);
  static_assert(alignof(IntentSlot) == 128);
  static_assert(alignof(IntentPool) == 64);
  static_assert(alignof(AcquireRequirement) == 16);
  static_assert(alignof(WorkDependency) == 16);
  static_assert(alignof(WorkItem) == 32);
  static_assert(alignof(WorkTicket) == 32);
  static_assert(alignof(NvmeQueueControl) == 64);
  static_assert(alignof(NvmeQueueView) == 64);
  static_assert(alignof(RuntimeView) == 64);

  static_assert(offsetof(ObjectEntry, state) == 36);
  static_assert(offsetof(AcquireIntent, valid) == 44);
  static_assert(offsetof(IntentSlot, sourceKind) == 72);
  static_assert(offsetof(WorkTicket, state) == 16);
  static_assert(offsetof(WorkTicket, epoch) == 32);
  static_assert(sizeof(WorkItem) == 64);
  static_assert(offsetof(WorkItem, reductionGroup) == 32);
  static_assert(offsetof(WorkItem, estimatedComputeNs) == 44);
  static_assert(offsetof(WorkItem, reserved0) == 48);
  static_assert(sizeof(WorkTicket) == 64);
  static_assert(offsetof(WorkTicket, unavailableBytes) == 40);
  static_assert(offsetof(WorkTicket, estimatedComputeNs) == 48);
  static_assert(offsetof(WorkTicket, reductionGroup) == 56);
  static_assert(offsetof(WorkTicket, contributorCount) == 60);
  static_assert(offsetof(BackendView, flags) == 52);
  static_assert(offsetof(BackendView, pendingAcquisitions) == 56);
  static_assert(offsetof(NvmeQueueView, ownerLock) == 96);
  static_assert(offsetof(NvmeQueueView, directMaxPrpPages) == 160);
  static_assert(offsetof(RuntimeView, workRunnableNs) == 56);
  static_assert(offsetof(RuntimeView, dependencies) == 64);
  static_assert(offsetof(RuntimeView, intentPool) == 72);
  static_assert(offsetof(RuntimeView, intentQueueEntries) == 80);
  static_assert(offsetof(RuntimeView, readyWorkTickets) == 96);
  static_assert(offsetof(RuntimeView, pendingWorkTickets) == 120);
  static_assert(offsetof(RuntimeView, ctaCompletions) == 136);
  static_assert(offsetof(RuntimeView, objectDependentHeads) == 144);
  static_assert(offsetof(RuntimeView, dependencySatisfied) == 160);
  static_assert(offsetof(RuntimeView, changedQueued) == 184);
  static_assert(offsetof(RuntimeView, changedOverflow) == 200);
  static_assert(offsetof(RuntimeView, requestProgress) == 208);
  static_assert(offsetof(RuntimeView, reductionExpected) == 216);
  static_assert(offsetof(RuntimeView, reductionCompleted) == 224);
  static_assert(offsetof(RuntimeView, reductionFailed) == 232);
  static_assert(offsetof(RuntimeView, requestCapacity) == 240);
  static_assert(offsetof(RuntimeView, epochStartClock) == 280);
  static_assert(offsetof(RuntimeView, epoch) == 288);
  static_assert(offsetof(RuntimeView, abiVersion) == 300);
  static_assert(offsetof(RuntimeView, stickyFailedCount) == 304);
  static_assert(sizeof(RuntimeView) == 320);
  static_assert(sourceTransferStride(packTransferStrides(17, 31)) == 17);
  static_assert(destinationTransferStride(packTransferStrides(17, 31)) == 31);
  static_assert(sourceTransferIndexLimit(packTransferIndexLimits(23, 47)) ==
                23);
  static_assert(
      destinationTransferIndexLimit(packTransferIndexLimits(23, 47)) == 47);
  if (Version != 25 || InvalidIndex != 0xffffffffU || BackendCount != 5 ||
      !std::is_trivially_copyable_v<ObjectEntry>) {
    return 1;
  }

  std::cout << "NTA ABI v" << Version << ": " << sizeof(RuntimeView)
            << "-byte runtime view\n";
  return 0;
}
