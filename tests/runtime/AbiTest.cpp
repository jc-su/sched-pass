#include "nta/OperatorContract.h"
#include "nta/RuntimeABI.h"
#include "nta/Tier.h"

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <type_traits>

int main() {
  using namespace nta::abi;

  static_assert(alignof(RequestContext) == 32);
  static_assert(alignof(TenantContext) == 16);
  static_assert(sizeof(TenantContext) == 16);
  static_assert(offsetof(TenantContext, outstandingBytes) == 8);
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
  static_assert(alignof(IntentQueueEntry) == 64);
  static_assert(alignof(IntentQueueNode) == 16);
  static_assert(alignof(IntentQueueControl) == 32);
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
  static_assert(offsetof(IntentSlot, chargedRequestBytes) == 80);
  static_assert(offsetof(IntentSlot, chargedBackendBytes) == 88);
  static_assert(sizeof(IntentQueueEntry) == 64);
  static_assert(offsetof(IntentQueueEntry, deadlineClock) == 16);
  static_assert(offsetof(IntentQueueEntry, state) == 36);
  static_assert(offsetof(IntentQueueEntry, priority) == 48);
  static_assert(sizeof(IntentQueueNode) == 16);
  static_assert(offsetof(IntentQueueNode, slotIndex) == 8);
  static_assert(sizeof(IntentQueueControl) == 32);
  static_assert(offsetof(IntentQueueControl, size) == 8);
  static_assert(offsetof(IntentQueueControl, lock) == 12);
  static_assert(offsetof(WorkTicket, state) == 16);
  static_assert(offsetof(WorkTicket, epoch) == 32);
  static_assert(sizeof(WorkItem) == 64);
  static_assert(offsetof(WorkItem, reductionGroup) == 32);
  static_assert(offsetof(WorkItem, estimatedComputeNs) == 44);
  static_assert(offsetof(WorkItem, readyDeadlineOffsetNs) == 48);
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
  static_assert(offsetof(RuntimeView, intentQueueControls) == 88);
  static_assert(offsetof(RuntimeView, intentQueueHeap) == 96);
  static_assert(offsetof(RuntimeView, readyWorkTickets) == 104);
  static_assert(offsetof(RuntimeView, pendingWorkTickets) == 128);
  static_assert(offsetof(RuntimeView, ctaCompletions) == 144);
  static_assert(offsetof(RuntimeView, objectDependentHeads) == 152);
  static_assert(offsetof(RuntimeView, dependencySatisfied) == 168);
  static_assert(offsetof(RuntimeView, changedQueued) == 192);
  static_assert(offsetof(RuntimeView, changedOverflow) == 208);
  static_assert(offsetof(RuntimeView, requestProgress) == 216);
  static_assert(offsetof(RuntimeView, reductionExpected) == 224);
  static_assert(offsetof(RuntimeView, reductionCompleted) == 232);
  static_assert(offsetof(RuntimeView, reductionFailed) == 240);
  static_assert(offsetof(RuntimeView, requestCapacity) == 248);
  static_assert(offsetof(RuntimeView, readyWindowEnd) == 284);
  static_assert(offsetof(RuntimeView, epochStartClock) == 288);
  static_assert(offsetof(RuntimeView, epoch) == 296);
  static_assert(offsetof(RuntimeView, abiVersion) == 308);
  static_assert(offsetof(RuntimeView, stickyFailedCount) == 312);
  static_assert(sizeof(RuntimeView) == 320);
  static_assert(sourceTransferStride(packTransferStrides(17, 31)) == 17);
  static_assert(destinationTransferStride(packTransferStrides(17, 31)) == 31);
  static_assert(sourceTransferIndexLimit(packTransferIndexLimits(23, 47)) ==
                23);
  static_assert(
      destinationTransferIndexLimit(packTransferIndexLimits(23, 47)) == 47);
  static_assert(nta::decodeTierCapabilities(nta::encodeTierCapabilities(
                    nta::TierDirectAddress | nta::TierHostRegistered)) ==
                (nta::TierDirectAddress | nta::TierHostRegistered));
  if (Version != 36 || InvalidIndex != 0xffffffffU || BackendCount != 6 ||
      !std::is_trivially_copyable_v<ObjectEntry>) {
    return 1;
  }

  nta::operator_contract::Contract contract{
      nta::operator_contract::Magic,
      nta::operator_contract::SchemaVersion,
      sizeof(nta::operator_contract::Contract),
      Version,
      static_cast<std::uint32_t>(nta::operator_contract::Family::Generic),
      static_cast<std::uint32_t>(nta::operator_contract::Form::Direct),
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
  };
  nta::operator_contract::validate(contract);
  contract.instrumentationFlags = 1ULL << 63U;
  try {
    nta::operator_contract::validate(contract);
    return 1;
  } catch (const std::runtime_error &) {
    // Unknown semantic flags must not cross the JIT/runtime boundary.
  }

  std::cout << "NTA ABI v" << Version << ": " << sizeof(RuntimeView)
            << "-byte runtime view\n";
  return 0;
}
