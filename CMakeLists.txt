# sched-pass: LLVM new-PM pass plugin for per-request scheduling weaving.
# Build against the LLVM that will load it (opt / clang of the SAME major):
#   cmake -S . -B build -DLLVM_DIR=/usr/lib/llvm-19/lib/cmake/llvm -GNinja
#   ninja -C build
cmake_minimum_required(VERSION 3.20)
project(sched-pass C CXX) # C too: LLVM's FindLibEdit probes a .c file

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Building sched-pass against LLVM ${LLVM_PACKAGE_VERSION} in ${LLVM_DIR}")

add_library(SchedPass MODULE
  lib/SchedPlugin.cpp
  lib/SchedUtil.cpp
  lib/SchedWeave.cpp
  lib/SchedWorkQueue.cpp
)
target_include_directories(SchedPass PRIVATE ${LLVM_INCLUDE_DIRS} include)
target_compile_features(SchedPass PRIVATE cxx_std_17)

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")
target_compile_definitions(SchedPass PRIVATE ${LLVM_DEFINITIONS_LIST})

if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(SchedPass PRIVATE -fno-rtti)
endif()

# MODULE plugin: symbols resolve from the loading process (opt/clang);
# do NOT link LLVM libraries (double-registration crashes).
