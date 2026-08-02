cmake_minimum_required(VERSION 3.24)
project(nonresident-acquisition LANGUAGES C CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NTA_ENABLE_CUDA "Build the real GPU host-memory acquisition workload" ON)
option(NTA_WARNINGS_AS_ERRORS "Treat project warnings as errors" ON)
option(NTA_ENABLE_FLASHINFER_VALIDATION
       "Use FlashInfer headers and enable differential validation when available"
       ON)
set(NTA_FLASHINFER_INCLUDE_DIR "" CACHE PATH
    "Path containing flashinfer/page.cuh and flashinfer/attention/state.cuh")

find_package(LLVM REQUIRED CONFIG)
find_package(Python3 COMPONENTS Interpreter QUIET)
message(STATUS "Building NTA against LLVM ${LLVM_PACKAGE_VERSION} in ${LLVM_DIR}")

add_library(NtaPass MODULE
  lib/AcquireAnalysis.cpp
  lib/AcquireLowering.cpp
  lib/DeferralLowering.cpp
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

add_library(nta-work-plan STATIC
  runtime/host/WorkPlan.cpp
)
set_target_properties(nta-work-plan PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(nta-work-plan PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
if(NTA_WARNINGS_AS_ERRORS)
  target_compile_options(
    nta-work-plan PRIVATE -Wall -Wextra -Wpedantic -Werror
  )
endif()

add_library(nta-flashinfer-adapter STATIC
  runtime/host/FlashInferAdapter.cpp
)
target_include_directories(nta-flashinfer-adapter PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(nta-flashinfer-adapter PUBLIC nta-work-plan)
if(NTA_WARNINGS_AS_ERRORS)
  target_compile_options(
    nta-flashinfer-adapter PRIVATE -Wall -Wextra -Wpedantic -Werror
  )
endif()

add_executable(
  nta-flashinfer-adapter-test tests/runtime/FlashInferAdapterTest.cpp
)
target_link_libraries(
  nta-flashinfer-adapter-test PRIVATE nta-flashinfer-adapter
)
if(NTA_WARNINGS_AS_ERRORS)
  target_compile_options(
    nta-flashinfer-adapter-test PRIVATE -Wall -Wextra -Wpedantic -Werror
  )
endif()
add_test(NAME nta-flashinfer-adapter COMMAND nta-flashinfer-adapter-test)

add_executable(nta-work-plan-test tests/runtime/WorkPlanTest.cpp)
target_link_libraries(nta-work-plan-test PRIVATE nta-work-plan)
if(NTA_WARNINGS_AS_ERRORS)
  target_compile_options(
    nta-work-plan-test PRIVATE -Wall -Wextra -Wpedantic -Werror
  )
endif()
add_test(NAME nta-work-plan COMMAND nta-work-plan-test)

install(DIRECTORY include/nta DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
  TARGETS nta-work-plan nta-flashinfer-adapter
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

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

if(Python3_Interpreter_FOUND)
  add_test(
    NAME nta-execution-policy
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/execution_policy.py
  )
  set_tests_properties(
    nta-execution-policy PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
  )
endif()

install(
  TARGETS NtaPass
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/nta
)

if(NTA_ENABLE_CUDA)
  find_package(CUDAToolkit 12.8 REQUIRED)
  if(NTA_ENABLE_FLASHINFER_VALIDATION AND
     NOT NTA_FLASHINFER_INCLUDE_DIR AND Python3_Interpreter_FOUND)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c
        "import importlib.util,pathlib; s=importlib.util.find_spec('flashinfer'); print(pathlib.Path(s.origin).parent/'data'/'include' if s else '')"
      OUTPUT_VARIABLE NTA_FLASHINFER_INCLUDE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  endif()
  set(NTA_FLASHINFER_DEVICE_FLAGS)
  if(NTA_ENABLE_FLASHINFER_VALIDATION AND
     EXISTS "${NTA_FLASHINFER_INCLUDE_DIR}/flashinfer/attention/state.cuh")
    set(NTA_HAVE_FLASHINFER TRUE)
    list(APPEND NTA_FLASHINFER_DEVICE_FLAGS
      -I${NTA_FLASHINFER_INCLUDE_DIR}
      -DNTA_USE_FLASHINFER_STATE=1
    )
    message(STATUS
      "FlashInfer device-state integration: ${NTA_FLASHINFER_INCLUDE_DIR}")
  elseif(NTA_ENABLE_FLASHINFER_VALIDATION)
    message(WARNING
      "FlashInfer headers were not found; building the compatible fallback state")
  endif()
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
    runtime/host/DeviceWorkPlan.cpp
    runtime/host/FinitePhase.cpp
    runtime/host/JitPhase.cpp
    runtime/host/NvmeRuntime.cpp
    runtime/host/NvmeVfioControlPlane.cpp
    runtime/host/Runtime.cpp
  )
  set_target_properties(nta-runtime-host PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(nta-runtime-host PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(
    nta-runtime-host PUBLIC nta-work-plan CUDA::cuda_driver CUDA::cudart
                             ${CMAKE_DL_LIBS}
  )
  if(NTA_WARNINGS_AS_ERRORS)
    target_compile_options(
      nta-runtime-host PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
  endif()

  add_library(nta-runtime SHARED runtime/host/RuntimeC.cpp)
  target_include_directories(nta-runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(nta-runtime PRIVATE nta-runtime-host CUDA::cudart)
  if(NTA_WARNINGS_AS_ERRORS)
    target_compile_options(
      nta-runtime PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
  endif()
  install(
    TARGETS nta-runtime nta-runtime-host
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
  install(
    DIRECTORY include/nta
    DESTINATION ${CMAKE_INSTALL_DATADIR}/nta/include
  )
  install(
    FILES runtime/device/Acquire.cuh runtime/device/JitRuntime.cuh
    DESTINATION ${CMAKE_INSTALL_DATADIR}/nta/runtime/device
  )
  install(
    PROGRAMS tools/jit/nvcc_clang.py
    DESTINATION ${CMAKE_INSTALL_DATADIR}/nta/tools/jit
  )
  install(
    FILES tools/jit/clang_cuda_prelude.h
    DESTINATION ${CMAKE_INSTALL_DATADIR}/nta/tools/jit
  )
  install(
    PROGRAMS tools/flashinfer/prepare_overlay.py
    DESTINATION ${CMAKE_INSTALL_DATADIR}/nta/tools/flashinfer
  )
  install(
    PROGRAMS tools/jit/activate.py
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    RENAME nta-jit-activate
  )

  add_executable(nta-runtime-c-api-test tests/runtime/RuntimeCApiTest.cpp)
  target_include_directories(nta-runtime-c-api-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(nta-runtime-c-api-test PRIVATE nta-runtime CUDA::cudart)
  if(NTA_WARNINGS_AS_ERRORS)
    target_compile_options(
      nta-runtime-c-api-test PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
  endif()
  add_test(NAME nta-runtime-c-api COMMAND nta-runtime-c-api-test)
  if(Python3_Interpreter_FOUND)
    add_test(
      NAME nta-request-slots
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/request_slots.py
    )
    add_test(
      NAME nta-sglang-plugin
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/sglang_plugin.py
    )
    add_test(
      NAME nta-python-runtime
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/python_runtime.py
    )
    add_test(
      NAME nta-qualification-runner
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/qualification_runner.py
    )
    add_test(
      NAME nta-qualification-evidence
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/qualification_evidence.py
    )
    add_test(
      NAME nta-opportunity-analysis
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/opportunity.py
    )
    set_tests_properties(
      nta-request-slots nta-sglang-plugin nta-qualification-runner
      nta-qualification-evidence
      nta-opportunity-analysis
      PROPERTIES TIMEOUT 60
    )
    set_tests_properties(nta-python-runtime nta-sglang-plugin PROPERTIES
      ENVIRONMENT
        "NTA_RUNTIME_LIBRARY=$<TARGET_FILE:nta-runtime>;PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
      TIMEOUT 60
    )
    set_tests_properties(nta-request-slots PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
    )
    set_tests_properties(nta-opportunity-analysis PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
    )
  endif()

  add_executable(nta-jit-phase-load tests/runtime/JitPhaseLoad.cpp)
  target_link_libraries(nta-jit-phase-load PRIVATE nta-runtime-host)
  if(NTA_WARNINGS_AS_ERRORS)
    target_compile_options(
      nta-jit-phase-load PRIVATE -Wall -Wextra -Wpedantic -Werror
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
      include/nta/KernelPolicy.cuh
      include/nta/RuntimeABI.h
      runtime/device/Acquire.cuh
    VERBATIM
    COMMAND_EXPAND_LISTS
  )
  add_custom_target(nta-kv-kernel ALL DEPENDS "${NTA_KV_CUBIN}")

  if(Python3_Interpreter_FOUND)
    add_test(
      NAME nta-moe-lowering
      COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/ir/moe_lowering.py
        ${NTA_KV_LOWERED_IR}
    )
    set_tests_properties(nta-moe-lowering PROPERTIES TIMEOUT 30)
  endif()

  add_test(
    NAME nta-clang-jit
    COMMAND
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/jit/run.sh
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/jit/nvcc_clang.py
      $<TARGET_FILE:NtaPass>
      ${NTA_CLANG_CUDA}
      ${NTA_CUDA_ROOT}
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${NTA_CUDA_ARCH}
      ${LLVM_TOOLS_BINARY_DIR}/llvm-dis
      ${CMAKE_CURRENT_BINARY_DIR}/jit-test
  )
  if(NTA_HAVE_FLASHINFER AND Python3_Interpreter_FOUND)
    add_test(
      NAME nta-flashinfer-jit-compile
      COMMAND
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/jit/activate.py
        --build-dir ${CMAKE_CURRENT_BINARY_DIR}
        --cache-root ${CMAKE_CURRENT_BINARY_DIR}/flashinfer-jit-cache
        --clang ${NTA_CLANG_CUDA}
        --cuda-path ${NTA_CUDA_ROOT}
        --flashinfer-hook
        --
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/flashinfer/jit_compile.py
    )
    set_tests_properties(nta-flashinfer-jit-compile PROPERTIES TIMEOUT 300)
    add_test(
      NAME nta-flashinfer-hooked-decode
      COMMAND
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/jit/activate.py
        --build-dir ${CMAKE_CURRENT_BINARY_DIR}
        --cache-root ${CMAKE_CURRENT_BINARY_DIR}/flashinfer-jit-cache
        --clang ${NTA_CLANG_CUDA}
        --cuda-path ${NTA_CUDA_ROOT}
        --flashinfer-hook
        --
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/flashinfer/hooked_decode.py
    )
    set_tests_properties(
      nta-flashinfer-jit-compile nta-flashinfer-hooked-decode
      PROPERTIES RUN_SERIAL TRUE TIMEOUT 300
    )
  endif()

  set(NTA_ATTENTION_RAW_BC
      "${CMAKE_CURRENT_BINARY_DIR}/kernel/PagedAttention.raw.bc")
  set(NTA_ATTENTION_LOWERED_IR
      "${CMAKE_CURRENT_BINARY_DIR}/kernel/PagedAttention.lowered.ll")
  set(NTA_ATTENTION_PTX
      "${CMAKE_CURRENT_BINARY_DIR}/kernel/PagedAttention.ptx")
  set(NTA_ATTENTION_CUBIN
      "${CMAKE_CURRENT_BINARY_DIR}/kernel/PagedAttention.cubin")
  add_custom_command(
    OUTPUT "${NTA_ATTENTION_CUBIN}"
    BYPRODUCTS
      "${NTA_ATTENTION_RAW_BC}"
      "${NTA_ATTENTION_LOWERED_IR}"
      "${NTA_ATTENTION_PTX}"
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
      ${NTA_FLASHINFER_DEVICE_FLAGS}
      -emit-llvm
      -c
      ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/attention/PagedAttentionKernel.cu
      -o "${NTA_ATTENTION_RAW_BC}"
    COMMAND
      ${LLVM_TOOLS_BINARY_DIR}/opt
      -load-pass-plugin=$<TARGET_FILE:NtaPass>
      -passes=nta-acquire,always-inline
      -S
      "${NTA_ATTENTION_RAW_BC}"
      -o "${NTA_ATTENTION_LOWERED_IR}"
    COMMAND
      ${LLVM_TOOLS_BINARY_DIR}/llc
      -march=nvptx64
      -mcpu=${NTA_CUDA_ARCH}
      -O3
      "${NTA_ATTENTION_LOWERED_IR}"
      -o "${NTA_ATTENTION_PTX}"
    COMMAND
      "${NTA_PTXAS}"
      -arch=${NTA_CUDA_ARCH}
      -O3
      "${NTA_ATTENTION_PTX}"
      -o "${NTA_ATTENTION_CUBIN}"
    DEPENDS
      NtaPass
      benchmarks/attention/PagedAttentionKernel.cu
      benchmarks/attention/PagedAttentionTypes.h
      include/nta/DeviceAPI.cuh
      include/nta/KernelPolicy.cuh
      include/nta/RuntimeABI.h
      runtime/device/Acquire.cuh
    VERBATIM
    COMMAND_EXPAND_LISTS
  )
  add_custom_target(
    nta-attention-kernel ALL DEPENDS "${NTA_ATTENTION_CUBIN}"
  )

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

  add_executable(nta-moe-bench benchmarks/moe/MoeExperts.cpp)
  add_dependencies(nta-moe-bench nta-kv-kernel)
  target_include_directories(nta-moe-bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_definitions(
    nta-moe-bench PRIVATE NTA_KV_CUBIN_PATH="${NTA_KV_CUBIN}"
  )
  target_link_libraries(
    nta-moe-bench PRIVATE nta-runtime-host CUDA::cuda_driver CUDA::cudart
  )
  if(NTA_WARNINGS_AS_ERRORS)
    target_compile_options(
      nta-moe-bench PRIVATE -Wall -Wextra -Wpedantic -Werror
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

  add_executable(nta-vfio-nvme-probe tools/VfioNvmeProbe.cpp)
  target_include_directories(nta-vfio-nvme-probe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(
    nta-vfio-nvme-probe PRIVATE nta-runtime-host CUDA::cuda_driver CUDA::cudart
  )
  if(NTA_WARNINGS_AS_ERRORS)
    target_compile_options(
      nta-vfio-nvme-probe PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
  endif()

  add_executable(
    nta-paged-attention benchmarks/attention/PagedAttention.cpp
  )
  add_dependencies(nta-paged-attention nta-attention-kernel)
  target_include_directories(nta-paged-attention PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_definitions(
    nta-paged-attention
    PRIVATE NTA_ATTENTION_CUBIN_PATH="${NTA_ATTENTION_CUBIN}"
  )
  target_link_libraries(
    nta-paged-attention PRIVATE nta-runtime-host nta-flashinfer-adapter
                                CUDA::cuda_driver CUDA::cudart
  )
  if(NTA_WARNINGS_AS_ERRORS)
    target_compile_options(
      nta-paged-attention PRIVATE -Wall -Wextra -Wpedantic -Werror
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

  set(NTA_DEPENDENCY_RACE_PTX
      "${CMAKE_CURRENT_BINARY_DIR}/kernel/DependencyArrivalRace.ptx")
  set(NTA_DEPENDENCY_RACE_CUBIN
      "${CMAKE_CURRENT_BINARY_DIR}/kernel/DependencyArrivalRace.cubin")
  add_custom_command(
    OUTPUT "${NTA_DEPENDENCY_RACE_CUBIN}"
    BYPRODUCTS "${NTA_DEPENDENCY_RACE_PTX}"
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
      -S
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/DependencyArrivalRaceKernel.cu
      -o "${NTA_DEPENDENCY_RACE_PTX}"
    COMMAND
      "${NTA_PTXAS}"
      -arch=${NTA_CUDA_ARCH}
      -O3
      "${NTA_DEPENDENCY_RACE_PTX}"
      -o "${NTA_DEPENDENCY_RACE_CUBIN}"
    DEPENDS
      tests/runtime/DependencyArrivalRaceKernel.cu
      include/nta/RuntimeABI.h
      include/nta/TicketProtocol.cuh
      runtime/device/Acquire.cuh
    VERBATIM
  )
  add_custom_target(
    nta-dependency-arrival-race-kernel
    DEPENDS "${NTA_DEPENDENCY_RACE_CUBIN}"
  )
  add_executable(
    nta-dependency-arrival-race-test
    tests/runtime/DependencyArrivalRaceTest.cpp
  )
  add_dependencies(
    nta-dependency-arrival-race-test nta-dependency-arrival-race-kernel
  )
  target_include_directories(nta-dependency-arrival-race-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries(
    nta-dependency-arrival-race-test
    PRIVATE nta-runtime-host CUDA::cuda_driver CUDA::cudart
  )
  target_compile_definitions(
    nta-dependency-arrival-race-test PRIVATE
    NTA_DEPENDENCY_RACE_CUBIN_PATH="${NTA_DEPENDENCY_RACE_CUBIN}"
  )
  add_test(
    NAME nta-dependency-arrival-race
    COMMAND nta-dependency-arrival-race-test
  )
  set_tests_properties(nta-dependency-arrival-race PROPERTIES TIMEOUT 60)

  add_executable(nta-multi-gpu-test tests/runtime/MultiGpuTest.cpp)
  target_include_directories(nta-multi-gpu-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(
    nta-multi-gpu-test PRIVATE nta-runtime-host CUDA::cudart
  )
  if(NTA_WARNINGS_AS_ERRORS)
    target_compile_options(
      nta-multi-gpu-test PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
  endif()
  add_test(NAME nta-multi-gpu COMMAND nta-multi-gpu-test)
  set_tests_properties(
    nta-multi-gpu PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77
  )

  add_executable(
    nta-cta-nvme-try-issue-test tests/runtime/CtaNvmeTryIssueTest.cpp
  )
  add_dependencies(nta-cta-nvme-try-issue-test nta-kv-kernel)
  target_include_directories(nta-cta-nvme-try-issue-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_definitions(
    nta-cta-nvme-try-issue-test
    PRIVATE NTA_KV_CUBIN_PATH="${NTA_KV_CUBIN}"
  )
  target_link_libraries(
    nta-cta-nvme-try-issue-test
    PRIVATE nta-runtime-host CUDA::cuda_driver CUDA::cudart
  )
  if(NTA_WARNINGS_AS_ERRORS)
    target_compile_options(
      nta-cta-nvme-try-issue-test PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
  endif()
  add_test(NAME nta-cta-nvme-try-issue COMMAND nta-cta-nvme-try-issue-test)
  set_tests_properties(nta-cta-nvme-try-issue PROPERTIES TIMEOUT 60)

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
  add_test(
    NAME nta-kv-registered-host-direct-gpu
    COMMAND
      $<TARGET_FILE:nta-kv-bench>
      --mode=host-direct
      --requests=18
      --tile-bytes=8192
      --iterations=3
      --coalesce=2
      --dependencies=3
      --external-registration=1
  )
  set_tests_properties(
    nta-kv-registered-host-direct-gpu PROPERTIES TIMEOUT 60
  )
  add_test(
    NAME nta-kv-registered-host-staged-gpu
    COMMAND
      $<TARGET_FILE:nta-kv-bench>
      --mode=host-staged
      --requests=18
      --tile-bytes=8192
      --iterations=3
      --coalesce=2
      --dependencies=3
      --external-registration=1
      --external-offset=1
  )
  set_tests_properties(
    nta-kv-registered-host-staged-gpu PROPERTIES TIMEOUT 60
  )
  add_test(
    NAME nta-kv-dependency-set-gpu
    COMMAND
      $<TARGET_FILE:nta-kv-bench>
      --mode=mixed
      --requests=18
      --tile-bytes=8192
      --iterations=3
      --cancel-stride=11
      --stale-stride=13
      --coalesce=2
      --dependencies=3
  )
  set_tests_properties(nta-kv-dependency-set-gpu PROPERTIES TIMEOUT 60)
  add_test(
    NAME nta-kv-object-version-fault-gpu
    COMMAND
      $<TARGET_FILE:nta-kv-bench>
      --mode=host-staged
      --requests=24
      --tile-bytes=4096
      --iterations=2
      --coalesce=2
      --dependencies=3
      --object-stale-stride=5
  )
  set_tests_properties(nta-kv-object-version-fault-gpu PROPERTIES TIMEOUT 60)
  add_test(
    NAME nta-kv-pending-index-scale-gpu
    COMMAND
      $<TARGET_FILE:nta-kv-bench>
      --mode=resident
      --requests=4096
      --tile-bytes=1024
      --iterations=2
      --coalesce=64
      --dependencies=4
  )
  set_tests_properties(
    nta-kv-pending-index-scale-gpu PROPERTIES
    TIMEOUT 60
    PASS_REGULAR_EXPRESSION
      "pending=0 pending_index=0 verification_failures=0"
  )
  add_test(
    NAME nta-kv-lifecycle-stress-gpu
    COMMAND
      $<TARGET_FILE:nta-kv-bench>
      --mode=mixed
      --requests=48
      --tile-bytes=8192
      --iterations=1
      --lifecycle-epochs=200
      --cancel-stride=11
      --stale-stride=13
      --coalesce=3
      --dependencies=4
  )
  set_tests_properties(
    nta-kv-lifecycle-stress-gpu PROPERTIES
    TIMEOUT 120
    PASS_REGULAR_EXPRESSION
      "lifecycle_epochs=200 graph_launches=200.*pending=0.*verification_failures=0"
  )
  add_test(
    NAME nta-kv-lifecycle-object-fault-gpu
    COMMAND
      $<TARGET_FILE:nta-kv-bench>
      --mode=host-staged
      --requests=24
      --tile-bytes=4096
      --iterations=1
      --lifecycle-epochs=100
      --object-stale-stride=7
      --coalesce=2
      --dependencies=3
      --external-registration=1
      --external-offset=1
  )
  set_tests_properties(
    nta-kv-lifecycle-object-fault-gpu PROPERTIES
    TIMEOUT 120
    PASS_REGULAR_EXPRESSION
      "lifecycle_epochs=100 graph_launches=100.*pending=0.*verification_failures=0"
  )
  add_test(
    NAME nta-moe-mixed-tier-gpu
    COMMAND
      $<TARGET_FILE:nta-moe-bench>
      --mode=mixed
      --policy=late-bound
      --tokens=24
      --experts=9
      --top-k=2
      --hidden=64
      --iterations=2
  )
  set_tests_properties(
    nta-moe-mixed-tier-gpu PROPERTIES
    TIMEOUT 60
    PASS_REGULAR_EXPRESSION
      "route_visibility=gpu policy=late-bound.*verification_failures=0"
  )
  add_test(
    NAME nta-moe-cpu-sync-gpu
    COMMAND
      $<TARGET_FILE:nta-moe-bench>
      --mode=host-staged
      --policy=cpu-sync
      --tokens=8
      --experts=16
      --top-k=2
      --hidden=64
      --iterations=2
  )
  set_tests_properties(
    nta-moe-cpu-sync-gpu PROPERTIES
    TIMEOUT 60
    PASS_REGULAR_EXPRESSION
      "route_visibility=gpu policy=cpu-sync.*verification_failures=0"
  )
  add_test(
    NAME nta-moe-overfetch-gpu
    COMMAND
      $<TARGET_FILE:nta-moe-bench>
      --mode=host-staged
      --policy=overfetch
      --tokens=8
      --experts=16
      --top-k=2
      --hidden=64
      --iterations=2
  )
  set_tests_properties(
    nta-moe-overfetch-gpu PROPERTIES
    TIMEOUT 60
    PASS_REGULAR_EXPRESSION
      "route_visibility=gpu policy=overfetch.*verification_failures=0"
  )
  add_test(
    NAME nta-moe-direct-gpu
    COMMAND
      $<TARGET_FILE:nta-moe-bench>
      --mode=resident
      --policy=direct
      --tokens=8
      --experts=16
      --top-k=2
      --hidden=64
      --iterations=2
  )
  set_tests_properties(
    nta-moe-direct-gpu PROPERTIES
    TIMEOUT 60
    PASS_REGULAR_EXPRESSION
      "route_visibility=gpu policy=direct.*verification_failures=0"
  )
  add_test(
    NAME nta-kv-direct-baseline-gpu
    COMMAND
      $<TARGET_FILE:nta-kv-bench>
      --mode=resident
      --requests=18
      --tile-bytes=8192
      --iterations=3
      --coalesce=2
      --dependencies=3
      --baseline=1
  )
  set_tests_properties(nta-kv-direct-baseline-gpu PROPERTIES TIMEOUT 60)
  add_test(
    NAME nta-paged-attention-gpu
    COMMAND
      $<TARGET_FILE:nta-paged-attention>
      --mode=host-staged
      --requests=4
      --min-pages=2
      --max-pages=4
      --iterations=2
      --progress-passes=4
      --request-credit-pages=1
  )
  set_tests_properties(nta-paged-attention-gpu PROPERTIES TIMEOUT 60)
  add_test(
    NAME nta-paged-attention-sparse-late-bound-gpu
    COMMAND
      $<TARGET_FILE:nta-paged-attention>
      --mode=host-staged
      --requests=4
      --min-pages=4
      --max-pages=6
      --sparse-top-k=2
      --iterations=2
      --progress-passes=1
      --request-credit-pages=2
  )
  set_tests_properties(
    nta-paged-attention-sparse-late-bound-gpu PROPERTIES
    TIMEOUT 60
    PASS_REGULAR_EXPRESSION
      "demand_visibility=gpu-cta.*verification_failures=0"
  )
  add_test(
    NAME nta-paged-attention-sparse-overfetch-gpu
    COMMAND
      $<TARGET_FILE:nta-paged-attention>
      --mode=host-staged
      --requests=4
      --min-pages=4
      --max-pages=6
      --sparse-top-k=2
      --sparse-policy=overfetch
      --iterations=2
      --progress-passes=1
      --request-credit-pages=2
  )
  set_tests_properties(
    nta-paged-attention-sparse-overfetch-gpu PROPERTIES
    TIMEOUT 60
    PASS_REGULAR_EXPRESSION
      "policy=overfetch.*verification_failures=0"
  )
  add_test(
    NAME nta-paged-attention-tma-gpu
    COMMAND
      $<TARGET_FILE:nta-paged-attention>
      --mode=host-staged
      --copy=tma
      --requests=4
      --min-pages=2
      --max-pages=4
      --iterations=2
      --progress-passes=4
      --request-credit-pages=1
  )
  set_tests_properties(nta-paged-attention-tma-gpu PROPERTIES TIMEOUT 60)
  if(NTA_HAVE_FLASHINFER AND Python3_Interpreter_FOUND)
    add_test(
      NAME nta-flashinfer-differential-gpu
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/flashinfer/differential_decode.py"
              $<TARGET_FILE:nta-paged-attention>
    )
    set_tests_properties(
      nta-flashinfer-differential-gpu PROPERTIES
      TIMEOUT 180
      ENVIRONMENT
        "FLASHINFER_NVCC=${CMAKE_CURRENT_SOURCE_DIR}/tools/flashinfer/nvcc_compat.py;NTA_REAL_NVCC=${CUDAToolkit_NVCC_EXECUTABLE}"
    )
  endif()
endif()

add_custom_target(
  nta-vfio-preflight
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/nta-vfio-device.sh preflight
  USES_TERMINAL
)
