#include "nta/NvmeUapi.h"
#include "nta/RuntimeABI.h"

#include <cstddef>
#include <iostream>
#include <type_traits>

int main() {
  using namespace nta::abi;

  static_assert(alignof(RequestContext) == 32);
  static_assert(alignof(ObjectEntry) == 64);
  static_assert(alignof(AcquireIntent) == 64);
  static_assert(alignof(Continuation) == 32);
  static_assert(alignof(NvmeQueueView) == 64);
  static_assert(alignof(RuntimeView) == 64);

  static_assert(offsetof(ObjectEntry, state) == 56);
  static_assert(offsetof(AcquireIntent, valid) == 44);
  static_assert(offsetof(Continuation, state) == 16);
  static_assert(offsetof(RuntimeView, intentCount) == 32);
  static_assert(offsetof(RuntimeView, nvme) == 40);
  static_assert(sizeof(nta_nvme_info) == 104);
  static_assert(sizeof(nta_nvme_import) == 48);
  static_assert(sizeof(nta_nvme_register_host) == 32);
  static_assert(sizeof(nta_nvme_dma_pages) == 2064);

  if (Version != 2 || InvalidIndex != 0xffffffffU ||
      !std::is_trivially_copyable_v<ObjectEntry>) {
    return 1;
  }

  std::cout << "NTA ABI v" << Version << ": " << sizeof(RuntimeView)
            << "-byte runtime view\n";
  return 0;
}
