#pragma once

#include "nta/OperatorContract.h"

#include <cstdint>

// Host-visible metadata emitted exactly once by a typed numerical module.
//
// This header deliberately contains no acquisition or transport kernel.  The
// compiler-produced operator module proves how numerical work binds request
// identity and exact demand; the runtime-owned transport program implements
// how those dependencies make progress.  Keeping the two code domains apart
// prevents a transport-only change from invalidating every numerical JIT
// artifact.

#ifndef NTA_OPERATOR_FAMILY
#define NTA_OPERATOR_FAMILY 0
#endif
#ifndef NTA_OPERATOR_FORM
#define NTA_OPERATOR_FORM 0
#endif
#ifndef NTA_OPERATOR_CAPABILITIES
#define NTA_OPERATOR_CAPABILITIES 0ULL
#endif
#ifndef NTA_OPERATOR_SOURCE_HASH_LOW
#define NTA_OPERATOR_SOURCE_HASH_LOW 0ULL
#endif
#ifndef NTA_OPERATOR_SOURCE_HASH_HIGH
#define NTA_OPERATOR_SOURCE_HASH_HIGH 0ULL
#endif
#ifndef NTA_OPERATOR_SUPPORTED_FORMS
#define NTA_OPERATOR_SUPPORTED_FORMS 6U
#endif
#ifndef NTA_OPERATOR_COORDINATE_MAP
#define NTA_OPERATOR_COORDINATE_MAP 0U
#endif
#ifndef NTA_OPERATOR_PARTIAL_STATE
#define NTA_OPERATOR_PARTIAL_STATE 0U
#endif
#ifndef NTA_OPERATOR_REDUCTION
#define NTA_OPERATOR_REDUCTION 0U
#endif
#ifndef NTA_OPERATOR_PLAN_FLAGS
#define NTA_OPERATOR_PLAN_FLAGS 0U
#endif
#ifndef NTA_OPERATOR_PLAN_HASH_LOW
#define NTA_OPERATOR_PLAN_HASH_LOW NTA_OPERATOR_SOURCE_HASH_LOW
#endif
#ifndef NTA_OPERATOR_PLAN_HASH_HIGH
#define NTA_OPERATOR_PLAN_HASH_HIGH NTA_OPERATOR_SOURCE_HASH_HIGH
#endif
#ifndef NTA_OPERATOR_INSTRUMENTATION_FLAGS
#define NTA_OPERATOR_INSTRUMENTATION_FLAGS 0ULL
#endif
#ifndef NTA_OPERATOR_IDENTITY_BINDING
#define NTA_OPERATOR_IDENTITY_BINDING 0U
#endif
#ifndef NTA_OPERATOR_DEMAND_BINDING
#define NTA_OPERATOR_DEMAND_BINDING 0U
#endif
#ifndef NTA_OPERATOR_ACCESS_PROOF
#define NTA_OPERATOR_ACCESS_PROOF 0U
#endif
#ifndef NTA_OPERATOR_GRANULARITY_BYTES
#define NTA_OPERATOR_GRANULARITY_BYTES 0U
#endif
#ifndef NTA_OPERATOR_TIER_MASK
#define NTA_OPERATOR_TIER_MASK 0ULL
#endif

extern "C" __attribute__((visibility("default")))
const nta::operator_contract::Contract *nta_jit_operator_contract() {
  static constexpr nta::operator_contract::Contract contract{
      nta::operator_contract::Magic,
      nta::operator_contract::SchemaVersion,
      sizeof(nta::operator_contract::Contract),
      nta::abi::Version,
      NTA_OPERATOR_FAMILY,
      NTA_OPERATOR_FORM,
      0,
      NTA_OPERATOR_CAPABILITIES,
      NTA_OPERATOR_SOURCE_HASH_LOW,
      NTA_OPERATOR_SOURCE_HASH_HIGH,
      NTA_OPERATOR_INSTRUMENTATION_FLAGS,
      NTA_OPERATOR_IDENTITY_BINDING,
      NTA_OPERATOR_DEMAND_BINDING,
      NTA_OPERATOR_ACCESS_PROOF,
      NTA_OPERATOR_GRANULARITY_BYTES,
      NTA_OPERATOR_TIER_MASK,
  };
  return &contract;
}

extern "C" __attribute__((visibility("default")))
const nta::operator_contract::Plan *nta_jit_operator_plan() {
  static constexpr nta::operator_contract::Plan plan{
      nta::operator_contract::PlanMagic,
      nta::operator_contract::PlanSchemaVersion,
      sizeof(nta::operator_contract::Plan),
      nta::abi::Version,
      NTA_OPERATOR_FAMILY,
      NTA_OPERATOR_SUPPORTED_FORMS,
      NTA_OPERATOR_COORDINATE_MAP,
      NTA_OPERATOR_PARTIAL_STATE,
      NTA_OPERATOR_REDUCTION,
      NTA_OPERATOR_PLAN_FLAGS,
      0,
      NTA_OPERATOR_SOURCE_HASH_LOW,
      NTA_OPERATOR_SOURCE_HASH_HIGH,
      NTA_OPERATOR_PLAN_HASH_LOW,
      NTA_OPERATOR_PLAN_HASH_HIGH,
  };
  return &plan;
}

extern "C" __attribute__((visibility("default"))) std::uint32_t
nta_jit_abi_version() {
  return nta::abi::Version;
}
