#include "nta/JitPhase.h"

#include <exception>
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: nta-jit-phase-load <instrumented-module.so>\n";
    return 2;
  }
  try {
    nta::JitPhaseProgram phases(argv[1]);
    const nta::operator_contract::Contract &contract =
        phases.operatorContract();
    const nta::operator_contract::Plan &plan = phases.operatorPlan();
    std::cout << "jit_phase_abi=" << nta::abi::Version
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
