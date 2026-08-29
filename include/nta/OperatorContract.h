#pragma once

#include "nta/RuntimeABI.h"

#include <cstdint>
#include <stdexcept>

namespace nta::operator_contract {

inline constexpr std::uint32_t Magic = 0x4f41544eU; // "NTAO"
inline constexpr std::uint16_t SchemaVersion = 1;
inline constexpr std::uint32_t PlanMagic = 0x5041544eU; // "NTAP"
inline constexpr std::uint16_t PlanSchemaVersion = 1;

enum class Family : std::uint32_t {
  Generic = 0,
  FlashInferDecode = 1,
  FlashInferPagedPrefill = 2,
};

enum class Form : std::uint32_t {
  Unspecified = 0,
  Direct = 1,
  Incremental = 2,
};

enum Capability : std::uint64_t {
  RequestBinding = 1ULL << 0,
  ObjectDependencies = 1ULL << 1,
  FiniteDeferral = 1ULL << 2,
  PartialPublication = 1ULL << 3,
  CompleteContributorMerge = 1ULL << 4,
  RunnableCompaction = 1ULL << 5,
  GraphReplay = 1ULL << 6,
  TypedFlashInferFrontend = 1ULL << 7,
  PreacquiredPartialEntry = 1ULL << 8,
  StreamOrderedRetirement = 1ULL << 9,
};

enum class CoordinateMap : std::uint32_t {
  Unspecified = 0,
  FlashInferRequestContiguous = 1,
};

enum class PartialState : std::uint32_t {
  None = 0,
  OnlineSoftmaxValueLse = 1,
};

enum class Reduction : std::uint32_t {
  None = 0,
  OrderedMergeState = 1,
};

enum PlanFlag : std::uint32_t {
  FixedCapacity = 1U << 0,
  GraphStable = 1U << 1,
  ExternalWaveSources = 1U << 2,
  GenerationBound = 1U << 3,
  ExactCompleteMerge = 1U << 4,
};

enum InstrumentationFlag : std::uint64_t {
  TypedAccessLowering = 1ULL << 0,
  ExactDemand = 1ULL << 1,
  GenerationSafeIdentity = 1ULL << 2,
  TierOwnership = 1ULL << 3,
};

enum class IdentityBinding : std::uint32_t {
  None = 0,
  RequestSlotGeneration = 1,
};

enum class DemandBinding : std::uint32_t {
  None = 0,
  ExactWorkUnit = 1,
};

enum class AccessProof : std::uint32_t {
  None = 0,
  LoadedIndexStride = 1,
  CpAsyncGlobal = 2,
  TypedFrontend = 3,
};

struct Contract {
  std::uint32_t magic;
  std::uint16_t schemaVersion;
  std::uint16_t structBytes;
  std::uint32_t runtimeAbiVersion;
  std::uint32_t family;
  std::uint32_t form;
  std::uint32_t reserved;
  std::uint64_t capabilities;
  std::uint64_t sourceFingerprintLow;
  std::uint64_t sourceFingerprintHigh;
  std::uint64_t instrumentationFlags;
  std::uint32_t identityBinding;
  std::uint32_t demandBinding;
  std::uint32_t accessProof;
  std::uint32_t granularityBytes;
  std::uint64_t tierMask;
};

static_assert(sizeof(Contract) == 80);

// Shared semantic plan for all generated forms of one typed operator. Unlike
// Contract::form, supportedForms and planFingerprint must be identical in the
// direct and incremental modules. This prevents the runtime from pairing two
// independently generated kernels that happen to use the same source label.
struct Plan {
  std::uint32_t magic;
  std::uint16_t schemaVersion;
  std::uint16_t structBytes;
  std::uint32_t runtimeAbiVersion;
  std::uint32_t family;
  std::uint32_t supportedForms;
  std::uint32_t coordinateMap;
  std::uint32_t partialState;
  std::uint32_t reduction;
  std::uint32_t flags;
  std::uint32_t reserved;
  std::uint64_t sourceFingerprintLow;
  std::uint64_t sourceFingerprintHigh;
  std::uint64_t planFingerprintLow;
  std::uint64_t planFingerprintHigh;
};

static_assert(sizeof(Plan) == 72);

[[nodiscard]] inline constexpr std::uint32_t formBit(Form form) {
  return form == Form::Unspecified ? 0U
                                   : 1U << static_cast<std::uint32_t>(form);
}

inline void validate(const Contract &contract) {
  constexpr std::uint64_t validCapabilities =
      RequestBinding | ObjectDependencies | FiniteDeferral |
      PartialPublication | CompleteContributorMerge | RunnableCompaction |
      GraphReplay | TypedFlashInferFrontend | PreacquiredPartialEntry |
      StreamOrderedRetirement;
  constexpr std::uint64_t validInstrumentationFlags =
      TypedAccessLowering | ExactDemand | GenerationSafeIdentity |
      TierOwnership;
  if (contract.magic != Magic || contract.schemaVersion != SchemaVersion ||
      contract.structBytes != sizeof(Contract)) {
    throw std::runtime_error(
        "JIT module has an incompatible operator contract");
  }
  if (contract.runtimeAbiVersion != abi::Version) {
    throw std::runtime_error(
        "JIT operator contract uses an incompatible runtime ABI");
  }
  if (contract.family >
          static_cast<std::uint32_t>(Family::FlashInferPagedPrefill) ||
      contract.form > static_cast<std::uint32_t>(Form::Incremental)) {
    throw std::runtime_error(
        "JIT operator contract contains an unknown family or form");
  }
  if (contract.identityBinding >
          static_cast<std::uint32_t>(IdentityBinding::RequestSlotGeneration) ||
      contract.demandBinding >
          static_cast<std::uint32_t>(DemandBinding::ExactWorkUnit) ||
      contract.accessProof >
          static_cast<std::uint32_t>(AccessProof::TypedFrontend)) {
    throw std::runtime_error(
        "JIT operator contract contains an unknown instrumentation proof");
  }
  if (contract.reserved != 0 ||
      (contract.capabilities & ~validCapabilities) != 0 ||
      (contract.instrumentationFlags & ~validInstrumentationFlags) != 0) {
    throw std::runtime_error(
        "JIT operator contract contains unknown flags or reserved bits");
  }
  const std::uint64_t validTierMask =
      (1ULL << (static_cast<std::uint32_t>(abi::SourceKind::Rdma) + 1U)) - 1ULL;
  if ((contract.tierMask & ~validTierMask) != 0) {
    throw std::runtime_error(
        "JIT operator contract names an unknown source tier");
  }
}

inline void validate(const Plan &plan, const Contract &contract) {
  constexpr std::uint32_t validForms =
      formBit(Form::Direct) | formBit(Form::Incremental);
  constexpr std::uint32_t validPlanFlags =
      FixedCapacity | GraphStable | ExternalWaveSources | GenerationBound |
      ExactCompleteMerge;
  if (plan.magic != PlanMagic || plan.schemaVersion != PlanSchemaVersion ||
      plan.structBytes != sizeof(Plan)) {
    throw std::runtime_error("JIT module has an incompatible operator plan");
  }
  if (plan.runtimeAbiVersion != abi::Version ||
      plan.runtimeAbiVersion != contract.runtimeAbiVersion) {
    throw std::runtime_error(
        "JIT operator plan uses an incompatible runtime ABI");
  }
  if (plan.reserved != 0 || (plan.supportedForms & ~validForms) != 0 ||
      (plan.flags & ~validPlanFlags) != 0) {
    throw std::runtime_error(
        "JIT operator plan contains unknown flags or reserved bits");
  }
  if (plan.family != contract.family ||
      (plan.supportedForms & formBit(static_cast<Form>(contract.form))) == 0U) {
    throw std::runtime_error(
        "JIT operator plan does not describe the module contract");
  }
  if (plan.sourceFingerprintLow != contract.sourceFingerprintLow ||
      plan.sourceFingerprintHigh != contract.sourceFingerprintHigh) {
    throw std::runtime_error(
        "JIT operator plan and module contract have different sources");
  }
  if (plan.coordinateMap > static_cast<std::uint32_t>(
                               CoordinateMap::FlashInferRequestContiguous) ||
      plan.partialState >
          static_cast<std::uint32_t>(PartialState::OnlineSoftmaxValueLse) ||
      plan.reduction >
          static_cast<std::uint32_t>(Reduction::OrderedMergeState)) {
    throw std::runtime_error("JIT operator plan contains an unknown semantic");
  }
  if (plan.planFingerprintLow == 0 && plan.planFingerprintHigh == 0) {
    throw std::runtime_error("JIT operator plan has an empty fingerprint");
  }
}

inline void validatePair(const Contract &directContract, const Plan &directPlan,
                         const Contract &incrementalContract,
                         const Plan &incrementalPlan) {
  validate(directContract);
  validate(incrementalContract);
  validate(directPlan, directContract);
  validate(incrementalPlan, incrementalContract);
  if (directContract.form != static_cast<std::uint32_t>(Form::Direct) ||
      incrementalContract.form !=
          static_cast<std::uint32_t>(Form::Incremental) ||
      directContract.family != incrementalContract.family ||
      directContract.instrumentationFlags !=
          incrementalContract.instrumentationFlags ||
      directContract.identityBinding != incrementalContract.identityBinding ||
      directContract.demandBinding != incrementalContract.demandBinding ||
      directContract.accessProof != incrementalContract.accessProof ||
      directContract.granularityBytes != incrementalContract.granularityBytes ||
      directContract.tierMask != incrementalContract.tierMask ||
      directPlan.supportedForms != incrementalPlan.supportedForms ||
      directPlan.coordinateMap != incrementalPlan.coordinateMap ||
      directPlan.partialState != incrementalPlan.partialState ||
      directPlan.reduction != incrementalPlan.reduction ||
      directPlan.flags != incrementalPlan.flags ||
      directPlan.sourceFingerprintLow != incrementalPlan.sourceFingerprintLow ||
      directPlan.sourceFingerprintHigh !=
          incrementalPlan.sourceFingerprintHigh ||
      directPlan.planFingerprintLow != incrementalPlan.planFingerprintLow ||
      directPlan.planFingerprintHigh != incrementalPlan.planFingerprintHigh) {
    throw std::runtime_error(
        "JIT direct and incremental modules have incompatible operator plans");
  }
}

} // namespace nta::operator_contract
