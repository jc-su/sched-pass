#pragma once

#include <cstddef>
#include <map>
#include <mutex>

namespace nta::detail {

// A bounded, device-independent allocator for slices of one mapped CXL-DAX
// window.  The transport owns the mapping; this class owns only reservations.
// Keeping the reservation ledger separate makes its lifetime and failure
// behavior testable without pretending that ordinary host memory is CXL.
struct CxlDaxAllocation {
  std::size_t offset = 0;
  std::size_t reservationOffset = 0;
  std::size_t reservationBytes = 0;
  std::size_t payloadBytes = 0;
};

class CxlDaxRangeAllocator final {
public:
  explicit CxlDaxRangeAllocator(std::size_t capacity);

  CxlDaxRangeAllocator(const CxlDaxRangeAllocator &) = delete;
  CxlDaxRangeAllocator &operator=(const CxlDaxRangeAllocator &) = delete;

  [[nodiscard]] CxlDaxAllocation allocate(std::size_t bytes,
                                          std::size_t alignment);
  void release(const CxlDaxAllocation &allocation) noexcept;

  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] std::size_t allocatedBytes() const noexcept;
  [[nodiscard]] std::size_t availableBytes() const noexcept;

private:
  struct ActiveAllocation {
    std::size_t reservationBytes;
    std::size_t payloadBytes;
  };

  [[nodiscard]] static std::size_t roundUp(std::size_t value,
                                           std::size_t alignment);

  mutable std::mutex mutex_;
  std::size_t capacity_ = 0;
  std::size_t nextOffset_ = 0;
  std::size_t allocatedBytes_ = 0;
  std::map<std::size_t, std::size_t> freeRanges_;
  // A release is accepted only for the exact live reservation returned by
  // allocate(). This makes the noexcept RAII release path idempotent against
  // accidental duplicate or forged release records.
  std::map<std::size_t, ActiveAllocation> activeAllocations_;
};

} // namespace nta::detail
