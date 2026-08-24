#pragma once

#include "nta/RuntimeABI.h"

#include <cstdint>

namespace nta {

// One backend description is shared by host registration, device admission,
// and experiment telemetry.  The compiler never selects a concrete backend;
// it emits an exact requirement and the runtime resolves that requirement
// against these descriptors.
enum TierCapability : std::uint32_t {
  TierDirectAddress = 1U << 0,
  TierDeviceInitiated = 1U << 1,
  TierHostRegistered = 1U << 2,
  TierPersistentStorage = 1U << 3,
  TierIndexedTransfer = 1U << 4,
};

[[nodiscard]] constexpr std::uint32_t
encodeTierCapabilities(std::uint32_t capabilities) noexcept {
  return (capabilities << abi::BackendTierCapabilityShift) &
         abi::BackendTierCapabilityMask;
}

[[nodiscard]] constexpr std::uint32_t
decodeTierCapabilities(std::uint32_t backendFlags) noexcept {
  return (backendFlags & abi::BackendTierCapabilityMask) >>
         abi::BackendTierCapabilityShift;
}

struct TierDescriptor {
  abi::SourceKind kind;
  std::uint32_t capabilities;
  std::uint64_t deviceState;
  std::uint64_t estimatedLatencyNs;
  std::uint64_t estimatedBandwidthBytesPerSecond;
  std::uint32_t active;
  std::uint32_t flags;
};
static_assert(sizeof(TierDescriptor) == 40);

[[nodiscard]] constexpr std::uint32_t
defaultTierCapabilities(abi::SourceKind kind) noexcept {
  switch (kind) {
  case abi::SourceKind::Hbm:
    return TierDirectAddress;
  case abi::SourceKind::HostMapped:
    return TierDirectAddress | TierHostRegistered;
  case abi::SourceKind::HostStaged:
    return TierIndexedTransfer;
  case abi::SourceKind::Nvme:
    return TierDeviceInitiated | TierPersistentStorage;
  case abi::SourceKind::Cxl:
    return TierDirectAddress | TierHostRegistered | TierPersistentStorage;
  case abi::SourceKind::Rdma:
    return TierPersistentStorage;
  }
  return 0;
}

} // namespace nta
