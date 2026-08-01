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
    std::cout << "jit_phase_abi=" << nta::abi::Version << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
