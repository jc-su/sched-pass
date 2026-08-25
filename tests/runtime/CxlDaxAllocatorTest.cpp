#include "nta/CxlDaxAllocator.h"

#include <cstddef>
#include <iostream>
#include <stdexcept>

int main() {
  nta::detail::CxlDaxRangeAllocator allocator(16 * 4096);
  const auto first = allocator.allocate(4096, 4096);
  const auto aligned = allocator.allocate(4096, 8192);
  if (first.offset != 0 || aligned.offset != 8192 ||
      allocator.allocatedBytes() != 8192 ||
      allocator.availableBytes() != 14 * 4096) {
    std::cerr << "CXL allocator initial placement failed\n";
    return 1;
  }

  allocator.release(first);
  const auto reused = allocator.allocate(4096, 4096);
  if (reused.offset != first.offset) {
    std::cerr << "CXL allocator did not reuse a released slice\n";
    return 1;
  }

  allocator.release(reused);
  allocator.release(aligned);
  if (allocator.allocatedBytes() != 0 ||
      allocator.availableBytes() != allocator.capacity()) {
    std::cerr << "CXL allocator did not coalesce released slices\n";
    return 1;
  }

  bool rejected = false;
  try {
    (void)allocator.allocate(0, 4096);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  if (!rejected) {
    std::cerr << "CXL allocator accepted an empty allocation\n";
    return 1;
  }

  rejected = false;
  try {
    (void)allocator.allocate(4096, 3000);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  if (!rejected) {
    std::cerr << "CXL allocator accepted a non-power-of-two alignment\n";
    return 1;
  }

  std::cout << "cxl_dax_allocator=pass\n";
  return 0;
}
