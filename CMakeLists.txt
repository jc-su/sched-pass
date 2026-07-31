cmake_minimum_required(VERSION 3.24)
project(nonresident-acquisition LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NTA_ENABLE_CUDA "Build the real GPU host-memory acquisition workload" ON)
option(NTA_WARNINGS_AS_ERRORS "Treat project warnings as errors" ON)

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Building NTA against LLVM ${LLVM_PACKAGE_VERSION} in ${LLVM_DIR}")

add_library(NtaPass MODULE
  lib/AcquireAnalysis.cpp
  lib/AcquireLowering.cpp
  lib/ContinuationLowering.cpp
  lib/Plugin.cpp
)
target_include_directories(NtaPass SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
if(EXISTS "/usr/include/llvm-${LLVM_VERSION_MAJOR}/llvm")
  target_include_directories(
    NtaPass SYSTEM BEFORE PRIVATE "/usr/include/llvm-${LLVM_VERSION_MAJOR}"
  )
endif()
target_include_directories(NtaPass PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
)
target_compile_features(NtaPass PRIVATE cxx_std_20)

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")
target_compile_definitions(NtaPass PRIVATE ${LLVM_DEFINITIONS_LIST})
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(NtaPass PRIVATE -fno-rtti)
endif()
if(NTA_WARNINGS_AS_ERRORS)
  target_compile_options(NtaPass PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

# A pass plugin resolves LLVM symbols from opt/clang. Linking another LLVM copy
# into the module can double-register command-line options.
if(APPLE)
  target_link_options(NtaPass PRIVATE "-Wl,-undefined,dynamic_lookup")
elseif(UNIX)
  target_link_options(NtaPass PRIVATE "-Wl,--unresolved-symbols=ignore-all")
endif()

enable_testing()

add_test(
  NAME nta-ir
  COMMAND
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/ir/run.sh
    $<TARGET_FILE:NtaPass>
    ${LLVM_TOOLS_BINARY_DIR}/opt
    ${CMAKE_CURRENT_BINARY_DIR}/ir-test
)

add_executable(nta-abi-test tests/runtime/AbiTest.cpp)
target_include_directories(nta-abi-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(NTA_WARNINGS_AS_ERRORS)
  target_compile_options(nta-abi-test PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
add_test(NAME nta-abi COMMAND nta-abi-test)

if(NTA_ENABLE_CUDA)
  find_package(CUDAToolkit 12.8 REQUIRED)
  if(NOT NTA_CLANG_CUDA)
    if(EXISTS "${LLVM_TOOLS_BINARY_DIR}/clang++")
      set(
        NTA_CLANG_CUDA
        "${LLVM_TOOLS_BINARY_DIR}/clang++"
        CACHE FILEPATH "Clang CUDA compiler paired with LLVM"
      )
    else()
      find_program(NTA_CLANG_CUDA NAMES clang++ REQUIRED)
    endif()
  endif()

  set(NTA_CUDA_ARCH "sm_120" CACHE STRING "CUDA architecture for the workload")
  set(
    NTA_CUDA_ROOT
    "${CUDAToolkit_LIBRARY_ROOT}"
    CACHE PATH "CUDA toolkit root used by clang CUDA"
  )
  if(NOT EXISTS "${NTA_CUDA_ROOT}/include/cuda_runtime.h")
    get_filename_component(NTA_CUDA_ROOT "${CUDAToolkit_BIN_DIR}/.." ABSOLUTE)
  endif()

  add_library(nta-runtime-host STATIC
    runtime/host/NvmeRuntime.cpp
    runtime/host/Runtime.cpp
  )
  target_include_directories(nta-runtime-host PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(nta-runtime-host PUBLIC CUDA::cuda_driver CUDA::cudart)
  if(NTA_WARNINGS_AS_ERRORS)
    target_compile_options(
      nta-runtime-host PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
  endif()

  find_program(
    NTA_PTXAS
    NAMES ptxas
    HINTS "${NTA_CUDA_ROOT}/bin"
    REQUIRED
    NO_DEFAULT_PATH
  )

  set(NTA_KV_RAW_BC "${CMAKE_CURRENT_BINARY_DIR}/kernel/KvAcquire.raw.bc")
  set(
    NTA_KV_LOWERED_IR
    "${CMAKE_CURRENT_BINARY_DIR}/kernel/KvAcquire.lowered.ll"
  )
  set(NTA_KV_PTX "${CMAKE_CURRENT_BINARY_DIR}/kernel/KvAcquire.ptx")
  set(NTA_KV_CUBIN "${CMAKE_CURRENT_BINARY_DIR}/kernel/KvAcquire.cubin")
  add_custom_command(
    OUTPUT "${NTA_KV_CUBIN}"
    BYPRODUCTS
      "${NTA_KV_RAW_BC}"
      "${NTA_KV_LOWERED_IR}"
      "${NTA_KV_PTX}"
    COMMAND
      ${CMAKE_COMMAND} -E make_directory
      "${CMAKE_CURRENT_BINARY_DIR}/kernel"
    COMMAND
      "${NTA_CLANG_CUDA}"
      -O3
      -std=c++20
      --cuda-device-only
      --cuda-path=${NTA_CUDA_ROOT}
      --cuda-gpu-arch=${NTA_CUDA_ARCH}
      -Wno-unknown-cuda-version
      -I${CMAKE_CURRENT_SOURCE_DIR}/include
      -I${CMAKE_CURRENT_SOURCE_DIR}
      -emit-llvm
      -c
      ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/kv/KvAcquireKernel.cu
      -o "${NTA_KV_RAW_BC}"
    COMMAND
      ${LLVM_TOOLS_BINARY_DIR}/opt
      -load-pass-plugin=$<TARGET_FILE:NtaPass>
      -passes=nta-acquire,always-inline
      -S
      "${NTA_KV_RAW_BC}"
      -o "${NTA_KV_LOWERED_IR}"
    COMMAND
      ${LLVM_TOOLS_BINARY_DIR}/llc
      -march=nvptx64
      -mcpu=${NTA_CUDA_ARCH}
      -O3
      "${NTA_KV_LOWERED_IR}"
      -o "${NTA_KV_PTX}"
    COMMAND
      "${NTA_PTXAS}"
      -arch=${NTA_CUDA_ARCH}
      -O3
      "${NTA_KV_PTX}"
      -o "${NTA_KV_CUBIN}"
    DEPENDS
      NtaPass
      benchmarks/kv/KvAcquireKernel.cu
      benchmarks/kv/KvTypes.h
      include/nta/DeviceAPI.cuh
      include/nta/RuntimeABI.h
      runtime/device/Acquire.cuh
    VERBATIM
    COMMAND_EXPAND_LISTS
  )
  add_custom_target(nta-kv-kernel ALL DEPENDS "${NTA_KV_CUBIN}")

  add_executable(nta-kv-bench benchmarks/kv/KvAcquire.cpp)
  add_dependencies(nta-kv-bench nta-kv-kernel)
  target_include_directories(nta-kv-bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_definitions(
    nta-kv-bench PRIVATE NTA_KV_CUBIN_PATH="${NTA_KV_CUBIN}"
  )
  target_link_libraries(
    nta-kv-bench PRIVATE nta-runtime-host CUDA::cuda_driver CUDA::cudart
  )
  if(NTA_WARNINGS_AS_ERRORS)
    target_compile_options(
      nta-kv-bench PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
  endif()

  add_executable(nta-nvme-bench benchmarks/nvme/NvmeRead.cpp)
  add_dependencies(nta-nvme-bench nta-kv-kernel)
  target_include_directories(nta-nvme-bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_definitions(
    nta-nvme-bench PRIVATE NTA_KV_CUBIN_PATH="${NTA_KV_CUBIN}"
  )
  target_link_libraries(
    nta-nvme-bench PRIVATE nta-runtime-host CUDA::cuda_driver CUDA::cudart
  )
  if(NTA_WARNINGS_AS_ERRORS)
    target_compile_options(
      nta-nvme-bench PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
  endif()

  add_executable(nta-runtime-test tests/runtime/RuntimeTest.cpp)
  target_include_directories(nta-runtime-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(nta-runtime-test PRIVATE nta-runtime-host CUDA::cudart)
  if(NTA_WARNINGS_AS_ERRORS)
    target_compile_options(
      nta-runtime-test PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
  endif()
  add_test(NAME nta-runtime COMMAND nta-runtime-test)
  add_test(
    NAME nta-kv-gpu
    COMMAND
      $<TARGET_FILE:nta-kv-bench>
      --mode=mixed
      --requests=24
      --tile-bytes=16384
      --iterations=3
      --cancel-stride=11
      --stale-stride=13
      --coalesce=3
  )
  set_tests_properties(nta-kv-gpu PROPERTIES TIMEOUT 60)
endif()

add_custom_target(
  nta-nvme-driver
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/nta-nvme-device.sh build
  USES_TERMINAL
)
