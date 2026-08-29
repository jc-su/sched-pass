// Runtime-owned transport phases shared by every typed operator.
//
// This module intentionally contains no numerical kernel and is not passed
// through NtaPass. It exports the finite transport ABI from JitRuntime.cuh so
// host staging, NVMe progress, publication, and completion do not depend on a
// framework-specific FlashInfer shared object.

#define NTA_DEVICE_PHASE_KERNELS 1
#define NTA_OPERATOR_FAMILY 0U
#define NTA_OPERATOR_FORM 2U
#define NTA_OPERATOR_CAPABILITIES 78ULL
#define NTA_OPERATOR_SOURCE_HASH_LOW 0x4e54415452414e53ULL
#define NTA_OPERATOR_SOURCE_HASH_HIGH 0x504f525450484153ULL
#define NTA_OPERATOR_SUPPORTED_FORMS 4U
#define NTA_OPERATOR_COORDINATE_MAP 0U
#define NTA_OPERATOR_PARTIAL_STATE 0U
#define NTA_OPERATOR_REDUCTION 0U
#define NTA_OPERATOR_PLAN_FLAGS 15U
#define NTA_OPERATOR_PLAN_HASH_LOW 0x5452414e53504f52ULL
#define NTA_OPERATOR_PLAN_HASH_HIGH 0x5450524f4752414dULL
#define NTA_OPERATOR_INSTRUMENTATION_FLAGS 8ULL
#define NTA_OPERATOR_IDENTITY_BINDING 0U
#define NTA_OPERATOR_DEMAND_BINDING 0U
#define NTA_OPERATOR_ACCESS_PROOF 0U
#define NTA_OPERATOR_GRANULARITY_BYTES 0U
#define NTA_OPERATOR_TIER_MASK 63ULL

#include "runtime/device/JitRuntime.cuh"
