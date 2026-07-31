#pragma once

#include "llvm/ADT/StringRef.h"

namespace nta::ir {

inline constexpr llvm::StringLiteral BindMarker = "__nta_bind_request";
inline constexpr llvm::StringLiteral AcquireMarker = "__nta_acquire_marker";
inline constexpr llvm::StringLiteral AcquireTensorMapMarker =
    "__nta_acquire_tensor_map_marker";
inline constexpr llvm::StringLiteral AcquireSetMarker =
    "__nta_acquire_set_marker";
inline constexpr llvm::StringLiteral DeferMarker = "__nta_defer_marker";

inline constexpr llvm::StringLiteral RequestLive = "nta_request_live";
inline constexpr llvm::StringLiteral AcquireSlow = "nta_acquire_slow";
inline constexpr llvm::StringLiteral AcquireTensorMapSlow =
    "nta_acquire_tensor_map_slow";
inline constexpr llvm::StringLiteral AcquireSetSlow = "nta_acquire_set_slow";
inline constexpr llvm::StringLiteral Defer = "nta_defer";

inline constexpr llvm::StringLiteral AcquisitionMetadata = "nta.acquire";
inline constexpr llvm::StringLiteral LoweredModuleFlag = "nta.lowered";

enum AcquireArgument : unsigned {
  Runtime = 0,
  DirectBase = 1,
  ObjectSlot = 2,
  ObjectId = 3,
  ObjectVersion = 4,
  Offset = 5,
  Bytes = 6,
  Continuation = 7,
  AcquireArgumentCount = 8,
};

enum AcquireSetArgument : unsigned {
  SetRuntime = 0,
  Requirements = 1,
  RequirementCount = 2,
  DirectRequirementCount = 3,
  SetContinuation = 4,
  AcquireSetArgumentCount = 5,
};

enum BindArgument : unsigned {
  RequestSlot = 0,
  RequestGeneration = 1,
  BindArgumentCount = 2,
};

enum DeferArgument : unsigned {
  DeferRuntime = 0,
  DeferContinuation = 1,
  DeferArgumentCount = 2,
};

} // namespace nta::ir
