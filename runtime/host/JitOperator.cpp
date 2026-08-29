#include "nta/JitOperator.h"

#include <dlfcn.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace nta {
namespace {

using AbiVersion = std::uint32_t (*)();
using OperatorContract = const operator_contract::Contract *(*)();
using OperatorPlan = const operator_contract::Plan *(*)();

template <typename Function> Function load(void *library, const char *name) {
  dlerror();
  void *symbol = dlsym(library, name);
  const char *error = dlerror();
  if (error != nullptr || symbol == nullptr) {
    throw std::runtime_error(std::string("cannot load ") + name + ": " +
                             (error == nullptr ? "symbol is null" : error));
  }
  return reinterpret_cast<Function>(symbol);
}

} // namespace

struct JitOperatorModule::Impl {
  explicit Impl(std::string_view path) {
    const std::string terminated(path);
    library = dlopen(terminated.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
      const char *error = dlerror();
      throw std::runtime_error(
          std::string("cannot load typed operator module: ") +
          (error == nullptr ? "unknown dynamic-loader error" : error));
    }
    try {
      const AbiVersion version =
          load<AbiVersion>(library, "nta_jit_abi_version");
      if (version() != abi::Version) {
        throw std::runtime_error(
            "typed operator module uses an incompatible NTA ABI");
      }
      const OperatorContract readContract =
          load<OperatorContract>(library, "nta_jit_operator_contract");
      const operator_contract::Contract *loadedContract = readContract();
      if (loadedContract == nullptr) {
        throw std::runtime_error(
            "typed operator module returned a null operator contract");
      }
      contract = *loadedContract;
      operator_contract::validate(contract);

      const OperatorPlan readPlan =
          load<OperatorPlan>(library, "nta_jit_operator_plan");
      const operator_contract::Plan *loadedPlan = readPlan();
      if (loadedPlan == nullptr) {
        throw std::runtime_error(
            "typed operator module returned a null operator plan");
      }
      plan = *loadedPlan;
      operator_contract::validate(plan, contract);
    } catch (...) {
      dlclose(library);
      library = nullptr;
      throw;
    }
  }

  ~Impl() {
    if (library != nullptr) {
      dlclose(library);
    }
  }

  void *library = nullptr;
  operator_contract::Contract contract{};
  operator_contract::Plan plan{};
};

JitOperatorModule::JitOperatorModule(std::string_view sharedObject)
    : impl_(std::make_unique<Impl>(sharedObject)) {}

JitOperatorModule::~JitOperatorModule() = default;
JitOperatorModule::JitOperatorModule(JitOperatorModule &&) noexcept = default;
JitOperatorModule &
JitOperatorModule::operator=(JitOperatorModule &&) noexcept = default;

const operator_contract::Contract &
JitOperatorModule::operatorContract() const noexcept {
  return impl_->contract;
}

const operator_contract::Plan &
JitOperatorModule::operatorPlan() const noexcept {
  return impl_->plan;
}

} // namespace nta
