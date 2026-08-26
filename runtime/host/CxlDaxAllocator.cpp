#include "nta/CxlDaxAllocator.h"

#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>

namespace nta::detail {

CxlDaxRangeAllocator::CxlDaxRangeAllocator(std::size_t capacity)
    : capacity_(capacity) {}

std::size_t CxlDaxRangeAllocator::roundUp(std::size_t value,
                                          std::size_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1U)) != 0) {
    throw std::invalid_argument(
        "CXL allocation alignment must be a power of two");
  }
  if (value > std::numeric_limits<std::size_t>::max() - (alignment - 1U)) {
    throw std::overflow_error("CXL allocation offset overflows size_t");
  }
  return (value + alignment - 1U) & ~(alignment - 1U);
}

CxlDaxAllocation CxlDaxRangeAllocator::allocate(std::size_t bytes,
                                                std::size_t alignment) {
  if (bytes == 0) {
    throw std::invalid_argument("CXL allocation needs non-zero bytes");
  }
  if (alignment == 0 || (alignment & (alignment - 1U)) != 0) {
    throw std::invalid_argument(
        "CXL allocation alignment must be a power-of-two value");
  }

  std::lock_guard lock(mutex_);
  if (bytes > capacity_ - allocatedBytes_) {
    throw std::runtime_error("CXL DAX window capacity exhausted");
  }
  for (auto range = freeRanges_.begin(); range != freeRanges_.end(); ++range) {
    const std::size_t candidate = roundUp(range->first, alignment);
    if (range->second >
        std::numeric_limits<std::size_t>::max() - range->first) {
      throw std::overflow_error("CXL free range overflows size_t");
    }
    const std::size_t rangeEnd = range->first + range->second;
    if (candidate > rangeEnd || bytes > rangeEnd - candidate) {
      continue;
    }

    const std::size_t prefixBytes = candidate - range->first;
    const std::size_t suffixBytes = rangeEnd - (candidate + bytes);
    const std::size_t rangeBegin = range->first;
    if (prefixBytes > std::numeric_limits<std::size_t>::max() - bytes) {
      throw std::overflow_error("CXL reservation size overflows size_t");
    }
    const std::size_t reservationBytes = bytes + prefixBytes;
    const auto [active, inserted] = activeAllocations_.emplace(
        rangeBegin, ActiveAllocation{reservationBytes, bytes});
    if (!inserted) {
      throw std::logic_error("CXL allocator live reservation overlaps free range");
    }
    auto original = freeRanges_.extract(range);
    bool prefixInserted = false;
    bool suffixInserted = false;
    try {
      if (prefixBytes != 0) {
        prefixInserted =
            freeRanges_.emplace(rangeBegin, prefixBytes).second;
        if (!prefixInserted) {
          throw std::logic_error("CXL allocator free-range insertion collided");
        }
      }
      if (suffixBytes != 0) {
        suffixInserted =
            freeRanges_.emplace(candidate + bytes, suffixBytes).second;
        if (!suffixInserted) {
          throw std::logic_error("CXL allocator free-range insertion collided");
        }
      }
    } catch (...) {
      if (prefixInserted) {
        freeRanges_.erase(rangeBegin);
      }
      if (suffixInserted) {
        freeRanges_.erase(candidate + bytes);
      }
      freeRanges_.insert(std::move(original));
      activeAllocations_.erase(active);
      throw;
    }
    allocatedBytes_ += bytes;
    return {candidate, rangeBegin, reservationBytes, bytes};
  }

  const std::size_t reservationOffset = nextOffset_;
  const std::size_t offset = roundUp(reservationOffset, alignment);
  if (offset > capacity_ || bytes > capacity_ - offset) {
    throw std::runtime_error("CXL DAX window capacity exhausted");
  }
  if (offset > std::numeric_limits<std::size_t>::max() - bytes) {
    throw std::overflow_error("CXL allocation end overflows size_t");
  }
  const std::size_t end = offset + bytes;
  const std::size_t reservationBytes = end - reservationOffset;
  const bool inserted = activeAllocations_
                            .emplace(reservationOffset,
                                     ActiveAllocation{reservationBytes, bytes})
                            .second;
  if (!inserted) {
    throw std::logic_error("CXL allocator live reservation overlaps tail");
  }
  nextOffset_ = end;
  allocatedBytes_ += bytes;
  return {offset, reservationOffset, reservationBytes, bytes};
}

void CxlDaxRangeAllocator::release(
    const CxlDaxAllocation &allocation) noexcept {
  if (allocation.reservationBytes == 0 || allocation.payloadBytes == 0) {
    return;
  }

  std::lock_guard lock(mutex_);
  const auto active = activeAllocations_.find(allocation.reservationOffset);
  if (active == activeAllocations_.end() ||
      active->second.reservationBytes != allocation.reservationBytes ||
      active->second.payloadBytes != allocation.payloadBytes) {
    return;
  }
  if (allocation.payloadBytes > allocatedBytes_ ||
      allocation.reservationOffset > capacity_ ||
      allocation.reservationBytes > capacity_ - allocation.reservationOffset) {
    return;
  }

  const std::size_t begin = allocation.reservationOffset;
  const std::size_t end = begin + allocation.reservationBytes;
  auto next = freeRanges_.lower_bound(begin);
  auto previous = next;
  if (previous != freeRanges_.begin()) {
    --previous;
    if (previous->first >
            std::numeric_limits<std::size_t>::max() - previous->second ||
        previous->first + previous->second != begin) {
      previous = freeRanges_.end();
    }
  } else {
    previous = freeRanges_.end();
  }

  std::map<std::size_t, std::size_t>::node_type previousNode;
  if (previous != freeRanges_.end()) {
    previousNode = freeRanges_.extract(previous);
  }
  next = freeRanges_.lower_bound(begin);
  std::map<std::size_t, std::size_t>::node_type nextNode;
  if (next != freeRanges_.end() && end == next->first) {
    nextNode = freeRanges_.extract(next);
  }

  std::size_t mergedBegin = begin;
  std::size_t mergedEnd = end;
  if (previousNode) {
    mergedBegin = previousNode.key();
    mergedEnd = end;
    if (previousNode.mapped() >
            std::numeric_limits<std::size_t>::max() - previousNode.key() ||
        previousNode.key() + previousNode.mapped() != begin) {
      // This is unreachable for a node selected above.  Restore the extracted
      // node and leave the allocator unchanged if the invariant is ever
      // violated by a future change.
      freeRanges_.insert(std::move(previousNode));
      if (nextNode) {
        freeRanges_.insert(std::move(nextNode));
      }
      return;
    }
    mergedBegin = previousNode.key();
  }
  if (nextNode) {
    if (nextNode.key() >
            std::numeric_limits<std::size_t>::max() - nextNode.mapped() ||
        nextNode.key() + nextNode.mapped() != end) {
      if (previousNode) {
        freeRanges_.insert(std::move(previousNode));
      }
      freeRanges_.insert(std::move(nextNode));
      return;
    }
    mergedEnd = nextNode.key() + nextNode.mapped();
  }

  try {
    const auto [unused, inserted] =
        freeRanges_.emplace(mergedBegin, mergedEnd - mergedBegin);
    (void)unused;
    if (!inserted) {
      if (previousNode) {
        freeRanges_.insert(std::move(previousNode));
      }
      if (nextNode) {
        freeRanges_.insert(std::move(nextNode));
      }
      return;
    }
  } catch (...) {
    if (previousNode) {
      freeRanges_.insert(std::move(previousNode));
    }
    if (nextNode) {
      freeRanges_.insert(std::move(nextNode));
    }
    return;
  }

  activeAllocations_.erase(active);
  allocatedBytes_ -= allocation.payloadBytes;
  while (!freeRanges_.empty()) {
    auto tail = std::prev(freeRanges_.end());
    if (tail->first > std::numeric_limits<std::size_t>::max() - tail->second ||
        tail->first + tail->second != nextOffset_) {
      break;
    }
    nextOffset_ = tail->first;
    freeRanges_.erase(tail);
  }
}

std::size_t CxlDaxRangeAllocator::capacity() const noexcept {
  std::lock_guard lock(mutex_);
  return capacity_;
}

std::size_t CxlDaxRangeAllocator::allocatedBytes() const noexcept {
  std::lock_guard lock(mutex_);
  return allocatedBytes_;
}

std::size_t CxlDaxRangeAllocator::availableBytes() const noexcept {
  std::lock_guard lock(mutex_);
  return capacity_ - allocatedBytes_;
}

} // namespace nta::detail
