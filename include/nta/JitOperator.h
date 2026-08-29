#pragma once

#include "nta/OperatorContract.h"

#include <memory>
#include <string_view>

namespace nta {

// Owning metadata view of one compiler-produced numerical operator module.
// Numerical modules export only their verified contract and plan; transport
// launchers are owned by the independent JitPhaseProgram resource.
class JitOperatorModule {
public:
  explicit JitOperatorModule(std::string_view sharedObject);
  ~JitOperatorModule();

  JitOperatorModule(const JitOperatorModule &) = delete;
  JitOperatorModule &operator=(const JitOperatorModule &) = delete;
  JitOperatorModule(JitOperatorModule &&) noexcept;
  JitOperatorModule &operator=(JitOperatorModule &&) noexcept;

  [[nodiscard]] const operator_contract::Contract &
  operatorContract() const noexcept;
  [[nodiscard]] const operator_contract::Plan &operatorPlan() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace nta
