#include "nta/JitOperator.h"

#include <exception>
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: nta-jit-operator-load <typed-module.so>\n";
    return 2;
  }
  try {
    nta::JitOperatorModule module(argv[1]);
    const nta::operator_contract::Contract &contract =
        module.operatorContract();
    const nta::operator_contract::Plan &plan = module.operatorPlan();
    std::cout << "jit_operator_abi=" << nta::abi::Version
              << " operator_family=" << contract.family
              << " operator_form=" << contract.form
              << " operator_capabilities=" << contract.capabilities
              << " plan_forms=" << plan.supportedForms
              << " plan_partial_state=" << plan.partialState
              << " plan_reduction=" << plan.reduction << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
