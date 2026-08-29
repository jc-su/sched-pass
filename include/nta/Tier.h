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

enum class TierOwner : std::uint32_t {
  None = 0,
  Engine = 1,
  Runtime = 2,
  Transport = 3,
};

struct TierOwnership {
  TierOwner protocol = TierOwner::None;
  TierOwner payload = TierOwner::None;
  TierOwner transferDestination = TierOwner::None;
  TierOwner mapping = TierOwner::None;
  TierOwner directory = TierOwner::Runtime;
};

[[nodiscard]] constexpr TierOwnership
defaultTierOwnership(abi::SourceKind kind) noexcept {
  switch (kind) {
  case abi::SourceKind::Hbm:
    return {TierOwner::Engine, TierOwner::Engine, TierOwner::None,
            TierOwner::None, TierOwner::Runtime};
  case abi::SourceKind::HostMapped:
    return {TierOwner::Engine, TierOwner::Engine, TierOwner::None,
            TierOwner::Engine, TierOwner::Runtime};
  case abi::SourceKind::HostStaged:
    return {TierOwner::Runtime, TierOwner::Engine, TierOwner::Engine,
            TierOwner::None, TierOwner::Runtime};
  case abi::SourceKind::Nvme:
    return {TierOwner::Transport, TierOwner::Transport, TierOwner::Engine,
            TierOwner::Transport, TierOwner::Runtime};
  case abi::SourceKind::Cxl:
    return {TierOwner::Transport, TierOwner::Transport, TierOwner::None,
            TierOwner::Transport, TierOwner::Runtime};
  case abi::SourceKind::Rdma:
    return {TierOwner::Transport, TierOwner::Transport, TierOwner::Engine,
            TierOwner::Transport, TierOwner::Runtime};
  }
  return {};
}

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
  // Ownership fields mirror ResourceContract.  They are setup/lifetime
  // metadata; the device BackendView still carries only data-path capability
  // bits so this descriptor does not enlarge the steady-state GPU directory.
  std::uint32_t protocolOwner;
  std::uint32_t payloadOwner;
  std::uint32_t transferDestinationOwner;
  std::uint32_t mappingOwner;
  std::uint32_t directoryOwner;
  std::uint32_t reserved;
};
static_assert(sizeof(TierDescriptor) == 64);

[[nodiscard]] constexpr std::uint32_t
defaultTierCapabilities(abi::SourceKind kind) noexcept {
  switch (kind) {
  case abi::SourceKind::Hbm:
    return TierDirectAddress;
  case abi::SourceKind::HostMapped:
    return TierDirectAddress | TierHostRegistered;
  case abi::SourceKind::HostStaged:
    return TierHostRegistered | TierIndexedTransfer;
  case abi::SourceKind::Nvme:
    return TierDeviceInitiated | TierPersistentStorage;
  case abi::SourceKind::Cxl:
    return TierDirectAddress | TierHostRegistered;
  case abi::SourceKind::Rdma:
    return TierPersistentStorage;
  }
  return 0;
}

} // namespace nta
