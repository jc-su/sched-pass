#include "nta/NvmeUapi.h"
#include "nta/RuntimeABI.h"

#include <cstddef>
#include <iostream>
#include <type_traits>

int main() {
  using namespace nta::abi;

  static_assert(alignof(RequestContext) == 32);
  static_assert(alignof(TenantContext) == 32);
  static_assert(alignof(ObjectEntry) == 64);
  static_assert(alignof(ReplicaEntry) == 64);
  static_assert(alignof(BackendView) == 64);
  static_assert(alignof(AcquireIntent) == 64);
  static_assert(alignof(IntentSlot) == 128);
  static_assert(alignof(IntentPool) == 64);
  static_assert(alignof(AcquireRequirement) == 16);
  static_assert(alignof(ContinuationDependency) == 16);
  static_assert(alignof(WorkItem) == 32);
  static_assert(alignof(Continuation) == 32);
  static_assert(alignof(NvmeQueueControl) == 64);
  static_assert(alignof(NvmeQueueView) == 64);
  static_assert(alignof(RuntimeView) == 64);

  static_assert(offsetof(ObjectEntry, state) == 36);
  static_assert(offsetof(AcquireIntent, valid) == 44);
  static_assert(offsetof(Continuation, state) == 16);
  static_assert(offsetof(RuntimeView, dependencies) == 56);
  static_assert(offsetof(RuntimeView, intentPool) == 64);
  static_assert(offsetof(RuntimeView, readyContinuations) == 72);
  static_assert(offsetof(RuntimeView, pendingContinuations) == 96);
  static_assert(sizeof(nta_nvme_queue_control) == 64);
  static_assert(sizeof(nta_nvme_info) == 128);
  static_assert(sizeof(nta_nvme_import) == 48);
  static_assert(sizeof(nta_nvme_register_host) == 32);
  static_assert(sizeof(nta_nvme_dma_pages) == 2064);

  static_assert(sizeof(NvmeQueueControl) == sizeof(nta_nvme_queue_control));
  static_assert(offsetof(NvmeQueueControl, state) ==
                offsetof(nta_nvme_queue_control, state));
  static_assert(offsetof(NvmeQueueControl, generation) ==
                offsetof(nta_nvme_queue_control, generation));
  static_assert(offsetof(NvmeQueueControl, queueId) ==
                offsetof(nta_nvme_queue_control, queue_id));

  if (Version != 10 || NvmeDriverAbiVersion != NTA_NVME_ABI_VERSION ||
      NvmeQueueControlMagic != NTA_NVME_QUEUE_CONTROL_MAGIC ||
      InvalidIndex != 0xffffffffU || BackendCount != 5 ||
      !std::is_trivially_copyable_v<ObjectEntry>) {
    return 1;
  }

  std::cout << "NTA ABI v" << Version << ": " << sizeof(RuntimeView)
            << "-byte runtime view\n";
  return 0;
}
