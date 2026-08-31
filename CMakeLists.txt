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
set(NTA_TEST_JIT_CACHE_ROOT
    "${CMAKE_CURRENT_BINARY_DIR}/flashinfer-jit-cache"
    CACHE PATH
    "Writable external root for FlashInfer/JIT CTest artifacts")
set(NTA_TEST_VLLM_MODEL "$ENV{NTA_TEST_VLLM_MODEL}" CACHE PATH
    "Local model directory for the optional paired vLLM serving E2E gate")
if(NTA_TEST_VLLM_MODEL AND NOT IS_DIRECTORY "${NTA_TEST_VLLM_MODEL}")
  message(FATAL_ERROR
    "NTA_TEST_VLLM_MODEL is configured but is not a directory: "
    "${NTA_TEST_VLLM_MODEL}")
endif()

# Physical tests never guess a PCI function or bind one as a side effect of a
# normal build. Once an operator has explicitly qualified and (optionally)
# kept a controller on VFIO, these cache values let CTest exercise that same
# endpoint instead of reporting an avoidable environment skip. They are empty
# by default so a build cannot claim a physical tier merely because a machine
# happens to have an NVMe device.
set(NTA_TEST_NVME_ENDPOINT "$ENV{NTA_NVME_ENDPOINT}" CACHE STRING
    "Explicit VFIO NVMe endpoint for physical CTest (e.g. vfio:0000:d8:00.0)")
set(NTA_TEST_NVME_REFERENCE "$ENV{NTA_NVME_REFERENCE}" CACHE FILEPATH
    "Read-only NVMe reference file for the physical attention CTest")
set(NTA_TEST_NVME_MEDIA_POLICY "$ENV{NTA_NVME_MEDIA_POLICY}" CACHE STRING
    "NVMe media policy for physical CTest")
set(NTA_TEST_NVME_DMA_TARGET "$ENV{NTA_NVME_DMA_TARGET}" CACHE STRING
    "NVMe DMA destination for physical CTest")
set(NTA_TEST_NVME_HBM_BACKEND "$ENV{NTA_NVME_HBM_BACKEND}" CACHE STRING
    "Required HBM mapper for physical CTest")
set(NTA_TEST_NVME_COMMAND_SERVICE_NS
    "$ENV{NTA_NVME_COMMAND_SERVICE_NS}" CACHE STRING
    "Measured effective NVMe command service for exact-granularity CTest")
set(NTA_TEST_NVME_READ_BANDWIDTH_BPS
    "$ENV{NTA_NVME_READ_BANDWIDTH_BPS}" CACHE STRING
    "Measured NVMe read bandwidth for exact-granularity CTest")
set(NTA_TEST_NVME_COMPACTION_BANDWIDTH_BPS
    "$ENV{NTA_NVME_COMPACTION_BANDWIDTH_BPS}" CACHE STRING
    "Measured HBM compaction bandwidth for exact-granularity CTest")
set(NTA_TEST_NVME_COMPACTION_LAUNCH_NS
    "$ENV{NTA_NVME_COMPACTION_LAUNCH_NS}" CACHE STRING
    "Measured HBM compaction launch service for exact-granularity CTest")
set(NTA_TEST_NVME_MINIMUM_GAIN "$ENV{NTA_NVME_MINIMUM_GAIN}" CACHE STRING
    "Minimum modeled gain for exact-granularity CTest")
set(_NTA_TEST_NVME_SERVICE_VALUE_COUNT 0)
foreach(_value IN ITEMS
        NTA_TEST_NVME_COMMAND_SERVICE_NS
        NTA_TEST_NVME_READ_BANDWIDTH_BPS
        NTA_TEST_NVME_COMPACTION_BANDWIDTH_BPS)
  if(NOT "${${_value}}" STREQUAL "")
    math(EXPR _NTA_TEST_NVME_SERVICE_VALUE_COUNT
         "${_NTA_TEST_NVME_SERVICE_VALUE_COUNT} + 1")
  endif()
endforeach()
if(NOT _NTA_TEST_NVME_SERVICE_VALUE_COUNT EQUAL 0 AND
   NOT _NTA_TEST_NVME_SERVICE_VALUE_COUNT EQUAL 3)
  message(FATAL_ERROR
    "physical NVMe granularity CTest requires command, read-bandwidth, and "
    "compaction-bandwidth measurements together")
endif()
if(_NTA_TEST_NVME_SERVICE_VALUE_COUNT EQUAL 3)
  set(NTA_TEST_NVME_SERVICE_CALIBRATED TRUE)
  if(NOT NTA_TEST_NVME_COMPACTION_LAUNCH_NS)
    set(NTA_TEST_NVME_COMPACTION_LAUNCH_NS "0")
  endif()
  if(NOT NTA_TEST_NVME_MINIMUM_GAIN)
    set(NTA_TEST_NVME_MINIMUM_GAIN "1.03")
  endif()
else()
  set(NTA_TEST_NVME_SERVICE_CALIBRATED FALSE)
endif()
if(NOT NTA_TEST_NVME_HBM_BACKEND)
  set(NTA_TEST_NVME_HBM_BACKEND "auto" CACHE STRING
      "Required HBM mapper for physical CTest" FORCE)
endif()
set_property(CACHE NTA_TEST_NVME_HBM_BACKEND PROPERTY STRINGS
             auto cuda-dmabuf-ioas nvidia-peer-pages)
if(NOT NTA_TEST_NVME_HBM_BACKEND STREQUAL "auto" AND
   NOT NTA_TEST_NVME_HBM_BACKEND STREQUAL "cuda-dmabuf-ioas" AND
   NOT NTA_TEST_NVME_HBM_BACKEND STREQUAL "nvidia-peer-pages")
  message(FATAL_ERROR
    "NTA_TEST_NVME_HBM_BACKEND must be auto, cuda-dmabuf-ioas, or "
    "nvidia-peer-pages")
endif()
if(NTA_TEST_NVME_DMA_TARGET AND
   NOT NTA_TEST_NVME_DMA_TARGET STREQUAL "hbm-peer" AND
   NOT NTA_TEST_NVME_HBM_BACKEND STREQUAL "auto")
  message(FATAL_ERROR
    "an explicit NTA_TEST_NVME_HBM_BACKEND requires hbm-peer DMA")
endif()
set(NTA_TEST_NVME_USE_SUDO "OFF" CACHE STRING
    "Run explicit physical NVMe CTest commands through sudo -n (OFF or ON)")
set_property(CACHE NTA_TEST_NVME_USE_SUDO PROPERTY STRINGS OFF ON)
if(NOT NTA_TEST_NVME_USE_SUDO STREQUAL "OFF" AND
   NOT NTA_TEST_NVME_USE_SUDO STREQUAL "ON")
  message(FATAL_ERROR "NTA_TEST_NVME_USE_SUDO must be OFF or ON")
endif()
set(NTA_TEST_NVME_MANAGE_SESSION "OFF" CACHE STRING
    "Keep the explicit NVMe endpoint on VFIO across physical CTests (OFF or ON)")
set_property(CACHE NTA_TEST_NVME_MANAGE_SESSION PROPERTY STRINGS OFF ON)
if(NOT NTA_TEST_NVME_MANAGE_SESSION STREQUAL "OFF" AND
   NOT NTA_TEST_NVME_MANAGE_SESSION STREQUAL "ON")
  message(FATAL_ERROR "NTA_TEST_NVME_MANAGE_SESSION must be OFF or ON")
endif()
if(NTA_TEST_NVME_MANAGE_SESSION STREQUAL "ON" AND
   NOT NTA_TEST_NVME_USE_SUDO STREQUAL "ON")
  message(FATAL_ERROR
    "NTA_TEST_NVME_MANAGE_SESSION=ON requires NTA_TEST_NVME_USE_SUDO=ON")
endif()
if(NTA_TEST_NVME_REFERENCE AND
   NOT EXISTS "${NTA_TEST_NVME_REFERENCE}")
  message(FATAL_ERROR
    "NTA_TEST_NVME_REFERENCE is configured but does not exist: "
    "${NTA_TEST_NVME_REFERENCE}. Capture a read-only reference or clear "
    "the cache before configuring a non-physical build.")
endif()
set(NTA_TEST_NVME_ENVIRONMENT)
set(NTA_TEST_NVME_PROBE_ARGUMENTS)
if(NTA_TEST_NVME_ENDPOINT)
  list(APPEND NTA_TEST_NVME_ENVIRONMENT
       "NTA_NVME_ENDPOINT=${NTA_TEST_NVME_ENDPOINT}")
  list(APPEND NTA_TEST_NVME_PROBE_ARGUMENTS
       "${NTA_TEST_NVME_ENDPOINT}" "0" "1" "64")
endif()
if(NTA_TEST_NVME_REFERENCE)
  list(APPEND NTA_TEST_NVME_ENVIRONMENT
       "NTA_NVME_REFERENCE=${NTA_TEST_NVME_REFERENCE}")
endif()
if(NTA_TEST_NVME_MEDIA_POLICY)
  list(APPEND NTA_TEST_NVME_ENVIRONMENT
       "NTA_NVME_MEDIA_POLICY=${NTA_TEST_NVME_MEDIA_POLICY}")
endif()
if(NTA_TEST_NVME_DMA_TARGET)
  list(APPEND NTA_TEST_NVME_ENVIRONMENT
       "NTA_NVME_DMA_TARGET=${NTA_TEST_NVME_DMA_TARGET}")
endif()
list(APPEND NTA_TEST_NVME_ENVIRONMENT
     "NTA_NVME_HBM_BACKEND=${NTA_TEST_NVME_HBM_BACKEND}")
if(NTA_TEST_NVME_ENDPOINT AND NTA_TEST_NVME_MEDIA_POLICY)
  list(APPEND NTA_TEST_NVME_PROBE_ARGUMENTS "${NTA_TEST_NVME_MEDIA_POLICY}")
elseif(NTA_TEST_NVME_ENDPOINT)
  list(APPEND NTA_TEST_NVME_PROBE_ARGUMENTS "hardware-write-protect")
endif()
if(NTA_TEST_NVME_ENDPOINT AND NTA_TEST_NVME_DMA_TARGET)
  list(APPEND NTA_TEST_NVME_PROBE_ARGUMENTS "${NTA_TEST_NVME_DMA_TARGET}")
elseif(NTA_TEST_NVME_ENDPOINT)
  list(APPEND NTA_TEST_NVME_PROBE_ARGUMENTS "hbm-peer")
endif()
if(NTA_TEST_NVME_ENDPOINT)
  list(APPEND NTA_TEST_NVME_PROBE_ARGUMENTS
       "${NTA_TEST_NVME_HBM_BACKEND}")
endif()
set(NTA_TEST_NVME_PRIVILEGED_PREFIX)
set(NTA_TEST_NVME_COMMAND_PREFIX)
if(NTA_TEST_NVME_USE_SUDO STREQUAL "ON")
  if(NOT NTA_TEST_NVME_ENDPOINT)
    message(FATAL_ERROR
      "NTA_TEST_NVME_USE_SUDO=ON requires an explicit NTA_TEST_NVME_ENDPOINT")
  endif()
  find_program(NTA_SUDO_EXECUTABLE sudo REQUIRED)
  # CTest's ENVIRONMENT property is applied to sudo itself, and sudo's
  # env_reset intentionally removes those variables. Pass the explicit test
  # contract through ``env`` so the privileged child receives exactly the
  # configured endpoint/policy, without relying on a broad sudo -E.
  list(APPEND NTA_TEST_NVME_PRIVILEGED_PREFIX
       "${NTA_SUDO_EXECUTABLE}" "-n" "env")
  list(APPEND NTA_TEST_NVME_PRIVILEGED_PREFIX
       ${NTA_TEST_NVME_ENVIRONMENT})
  set(NTA_TEST_NVME_COMMAND_PREFIX ${NTA_TEST_NVME_PRIVILEGED_PREFIX})
  if(NTA_TEST_NVME_MANAGE_SESSION STREQUAL "ON")
    if(NOT NTA_TEST_NVME_ENDPOINT MATCHES
       "^vfio:[0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f]:[0-9A-Fa-f][0-9A-Fa-f]:[0-9A-Fa-f][0-9A-Fa-f]\\.[0-7]$")
      message(FATAL_ERROR
        "managed physical NVMe CTests require vfio:DDDD:BB:SS.F")
    endif()
    string(REGEX REPLACE "^vfio:" "" NTA_TEST_NVME_BDF
           "${NTA_TEST_NVME_ENDPOINT}")
    list(APPEND NTA_TEST_NVME_COMMAND_PREFIX
      "NTA_ALLOW_DEVICE_REBIND=1"
      "NTA_NVME_BDF=${NTA_TEST_NVME_BDF}"
      "${CMAKE_CURRENT_SOURCE_DIR}/scripts/nta-vfio-device.sh"
      "run" "--")
  endif()
endif()

find_package(LLVM REQUIRED CONFIG)
find_package(Python3 COMPONENTS Interpreter QUIET)
message(STATUS "Building NTA against LLVM ${LLVM_PACKAGE_VERSION} in ${LLVM_DIR}")

add_library(NtaPass MODULE
  lib/AcquireAnalysis.cpp
  lib/AcquireDiscovery.cpp
  lib/AcquireLowering.cpp
  lib/DeferralLowering.cpp
  lib/PartialLowering.cpp
  lib/Plugin.cpp
)
target_include_directories(NtaPass SYSTEM BEFORE PRIVATE ${LLVM_INCLUDE_DIRS})
if(LLVM_MAIN_INCLUDE_DIR)
  # CMake omits implicit system paths such as /usr/local/include.  Re-add the
  # selected package path explicitly so header-layout probing cannot fall
  # through to a different system LLVM installation.
  target_compile_options(NtaPass PRIVATE "-isystem${LLVM_MAIN_INCLUDE_DIR}")
endif()
target_include_directories(NtaPass PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
)
target_compile_features(NtaPass PRIVATE cxx_std_20)

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")
if(EXISTS "${LLVM_MAIN_INCLUDE_DIR}/llvm/Plugins/PassPlugin.h")
  set(NTA_LLVM_PLUGIN_HEADER_PLUGINS 1)
else()
  set(NTA_LLVM_PLUGIN_HEADER_PLUGINS 0)
endif()
# The pass is intentionally resolved by the matching opt/clang host instead of
# linking a second LLVM copy.  Some static LLVM tool builds do not export the
# ABI-check sentinel from the host executable; suppress only that header-level
# link sentinel while retaining the ABI selected by the package's headers.
target_compile_definitions(
  NtaPass PRIVATE
  LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1
  NTA_LLVM_PLUGIN_HEADER_PLUGINS=${NTA_LLVM_PLUGIN_HEADER_PLUGINS}
  ${LLVM_DEFINITIONS_LIST}
)
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

if(NTA_TEST_NVME_MANAGE_SESSION STREQUAL "ON")
  add_test(
    NAME nta-nvme-session-start
    COMMAND ${NTA_TEST_NVME_PRIVILEGED_PREFIX}
      "NTA_ALLOW_DEVICE_REBIND=1"
      "NTA_NVME_BDF=${NTA_TEST_NVME_BDF}"
      "${CMAKE_CURRENT_SOURCE_DIR}/scripts/nta-vfio-device.sh"
      session-start
  )
  add_test(
    NAME nta-nvme-session-stop
    COMMAND ${NTA_TEST_NVME_PRIVILEGED_PREFIX}
      "NTA_NVME_BDF=${NTA_TEST_NVME_BDF}"
      "${CMAKE_CURRENT_SOURCE_DIR}/scripts/nta-vfio-device.sh"
      session-stop
  )
  set_tests_properties(
    nta-nvme-session-start PROPERTIES
      FIXTURES_SETUP nta_nvme_session
      LABELS "physical;nvme;safety"
      RESOURCE_LOCK nta_nvme_physical
      TIMEOUT 30
  )
  set_tests_properties(
    nta-nvme-session-stop PROPERTIES
      FIXTURES_CLEANUP nta_nvme_session
      LABELS "physical;nvme;safety"
      RESOURCE_LOCK nta_nvme_physical
      TIMEOUT 30
      PASS_REGULAR_EXPRESSION "zero_write=pass"
  )
endif()

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

add_library(nta-cxl-dax-allocator STATIC
  runtime/host/CxlDaxAllocator.cpp
)
set_target_properties(nta-cxl-dax-allocator PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(nta-cxl-dax-allocator PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
if(NTA_WARNINGS_AS_ERRORS)
  target_compile_options(
    nta-cxl-dax-allocator PRIVATE -Wall -Wextra -Wpedantic -Werror
  )
endif()

add_executable(
  nta-cxl-dax-allocator-test tests/runtime/CxlDaxAllocatorTest.cpp
)
target_link_libraries(nta-cxl-dax-allocator-test PRIVATE nta-cxl-dax-allocator)
if(NTA_WARNINGS_AS_ERRORS)
  target_compile_options(
    nta-cxl-dax-allocator-test PRIVATE -Wall -Wextra -Wpedantic -Werror
  )
endif()
add_test(NAME nta-cxl-dax-allocator COMMAND nta-cxl-dax-allocator-test)
set_tests_properties(nta-cxl-dax-allocator PROPERTIES LABELS "contract;tier")

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

add_executable(nta-nvme-discovery-test tests/runtime/NvmeDiscoveryTest.cpp)
target_include_directories(nta-nvme-discovery-test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
if(NTA_WARNINGS_AS_ERRORS)
  target_compile_options(
    nta-nvme-discovery-test PRIVATE -Wall -Wextra -Wpedantic -Werror
  )
endif()
add_test(NAME nta-nvme-discovery COMMAND nta-nvme-discovery-test)
set_tests_properties(nta-nvme-discovery PROPERTIES LABELS "contract;tier")

add_executable(nta-cxl-dax-discovery-test tests/runtime/CxlDaxDiscoveryTest.cpp)
target_include_directories(nta-cxl-dax-discovery-test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
if(NTA_WARNINGS_AS_ERRORS)
  target_compile_options(
    nta-cxl-dax-discovery-test PRIVATE -Wall -Wextra -Wpedantic -Werror
  )
endif()
add_test(NAME nta-cxl-dax-discovery COMMAND nta-cxl-dax-discovery-test)
set_tests_properties(nta-cxl-dax-discovery PROPERTIES LABELS "contract;tier")

install(DIRECTORY include/nta DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
  TARGETS nta-work-plan nta-cxl-dax-allocator nta-flashinfer-adapter
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

add_test(
  NAME nta-ir-mutation
  COMMAND
    ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/ir/mutate.py
    $<TARGET_FILE:NtaPass>
    ${LLVM_TOOLS_BINARY_DIR}/opt
    ${CMAKE_CURRENT_BINARY_DIR}/ir-mutation
)

add_executable(nta-abi-test tests/runtime/AbiTest.cpp)
target_include_directories(nta-abi-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(NTA_WARNINGS_AS_ERRORS)
  target_compile_options(nta-abi-test PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
add_test(NAME nta-abi COMMAND nta-abi-test)

if(Python3_Interpreter_FOUND)
  add_test(
    NAME nta-acquisition-scheduler
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/acquisition_scheduler.py
  )
  set_tests_properties(
    nta-acquisition-scheduler PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
  )
  add_test(
    NAME nta-flashinfer-jit-materialization
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/flashinfer_jit.py
  )
  set_tests_properties(
    nta-flashinfer-jit-materialization PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
    LABELS "contract;jit"
  )
  add_test(
    NAME nta-execution-planner
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/execution_planner.py
  )
  set_tests_properties(
    nta-execution-planner PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
  )
  add_test(
    NAME nta-work-unit
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/work_unit.py
  )
  set_tests_properties(
    nta-work-unit PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
  )
  add_test(
    NAME nta-adapters
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/adapters.py
  )
  set_tests_properties(
    nta-adapters PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
  )
  add_test(
    NAME nta-tier-service
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/tier_service.py
  )
  set_tests_properties(
    nta-tier-service PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
  )
  add_test(
    NAME nta-execution-core
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/execution_core.py
  )
  set_tests_properties(
    nta-execution-core PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
  )
  add_test(
    NAME nta-tenant-policy
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/tenant_policy.py
  )
  set_tests_properties(
    nta-tenant-policy PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
  )
  add_test(
    NAME nta-project-audit
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/experiments/project_audit.py
      --root ${CMAKE_CURRENT_SOURCE_DIR}
  )
  set_tests_properties(
    nta-project-audit PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
  )
  add_test(
    NAME nta-experiment-matrix
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/experiments/smoke_matrix.py
  )
  set_tests_properties(
    nta-experiment-matrix PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
  )
  add_test(
    NAME nta-evaluation-contract
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/experiments/validate_evaluation.py
  )
  add_test(
    NAME nta-bailian-workload
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/bailian_workload.py
  )
  set_tests_properties(
    nta-bailian-workload PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}"
  )
  add_test(
    NAME nta-queueing
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/queueing.py
  )
  set_tests_properties(
    nta-queueing PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}"
  )
  add_test(
    NAME nta-serving-report
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/serving_report.py
  )
  add_test(
    NAME nta-vllm-report
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/vllm_report.py
  )
  add_test(
    NAME nta-jit-environment
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/jit_environment.py
  )
  add_test(
    NAME nta-exact-native-bridge
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/exact_native_bridge.py
  )
  set_tests_properties(
    nta-exact-native-bridge PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
  )
  add_test(
    NAME nta-execution-topology
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/execution_topology.py
  )
  set_tests_properties(
    nta-execution-topology PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
  )
  add_test(
    NAME nta-indexed-host-plan
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/indexed_host_plan.py
  )
  set_tests_properties(
    nta-indexed-host-plan PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
  )
  add_test(
    NAME nta-sglang-lease-topology
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/sglang_lease_topology.py
  )
  add_test(
    NAME nta-sglang-execution
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/sglang_execution.py
  )
  set_tests_properties(
    nta-sglang-execution PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
  )
  set_tests_properties(
    nta-sglang-lease-topology PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
  )
  add_test(
    NAME nta-nvme-materialization
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/nvme_materialization.py
  )
  set_tests_properties(
    nta-nvme-materialization PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
  )
  add_test(
    NAME nta-sglang-nvme
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/sglang_nvme.py
  )
  set_tests_properties(
    nta-sglang-nvme PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
  )
  add_test(
    NAME nta-progress-frontier
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/progress_frontier.py
  )
  set_tests_properties(
    nta-progress-frontier PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
  )
  add_test(
    NAME nta-indexed-transfer
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/indexed_transfer.py
  )
  set_tests_properties(
    nta-indexed-transfer PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
  )
  add_test(
    NAME nta-hbm-registration
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/hbm_registration.py
  )
  set_tests_properties(
    nta-hbm-registration PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
  )
  add_test(
    NAME nta-indexed-row-count-seam
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/indexed_row_count_seam.py
  )
  set_tests_properties(
    nta-indexed-row-count-seam PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
  )
  add_test(
    NAME nta-tier-streaming
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/tier_streaming.py
  )
  set_tests_properties(
    nta-tier-streaming PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
  )
  add_test(
    NAME nta-tier-streaming-evidence
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/tier_streaming_evidence.py
  )
  add_test(
    NAME nta-flashinfer-overlay
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/flashinfer_overlay.py
  )
  set_tests_properties(
    nta-flashinfer-overlay PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}"
  )
  add_test(
    NAME nta-flashinfer-plan
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/flashinfer_plan.py
  )
  set_tests_properties(
    nta-flashinfer-plan PROPERTIES
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
  execute_process(
    COMMAND "${NTA_CLANG_CUDA}" --version
    OUTPUT_VARIABLE NTA_CUDA_CLANG_VERSION
    ERROR_VARIABLE NTA_CUDA_CLANG_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
  )
  # The production LLVM package uses the normal optimized CUDA frontend and
  # llc path.  The optional clangir development toolchain is intentionally
  # recognized here because its current SROA/GVN assertions require the
  # conservative pipeline below.  This keeps the safe fallback local to that
  # toolchain instead of making every reproducible build pay its cost.
  set(NTA_CUDA_OPT_DEFAULT 3)
  if(NTA_CUDA_CLANG_VERSION MATCHES "clangir" OR
     LLVM_PACKAGE_VERSION MATCHES "git")
    set(NTA_CUDA_OPT_DEFAULT 0)
  endif()
  set(
    NTA_CUDA_FRONTEND_OPT_LEVEL
    "${NTA_CUDA_OPT_DEFAULT}"
    CACHE STRING
    "Optimization level used while clang emits raw CUDA LLVM bitcode"
  )
  set_property(CACHE NTA_CUDA_FRONTEND_OPT_LEVEL PROPERTY STRINGS 0 1 2 3)
  # Raw bitcode is optimized by the NTA pass pipeline and nvptx llc below.
  # The production default is -O3; the clangir detection above selects -O0
  # only for the known assertion-prone development toolchain.
  set(NTA_CUDA_FRONTEND_OPT_FLAG "-O${NTA_CUDA_FRONTEND_OPT_LEVEL}")
  set(NTA_CUDA_FRONTEND_FLAGS "${NTA_CUDA_FRONTEND_OPT_FLAG}")
  if(NTA_CUDA_FRONTEND_OPT_LEVEL STREQUAL "0")
    list(APPEND NTA_CUDA_FRONTEND_FLAGS -Xclang -disable-O0-optnone)
  endif()
  # These are deliberately the only generic cleanup passes before the
  # contract verifier.  The paired opt build has known assertion failures in
  # broader simplifycfg/instcombine pipelines, while SROA plus early-CSE is
  # sufficient to expose the typed marker and preserve fail-closed checks.
  set(NTA_ACQUIRE_PREPASSES "function(mem2reg,sroa,early-cse)")
  set(
    NTA_CUDA_LLC_OPT_LEVEL
    "${NTA_CUDA_OPT_DEFAULT}"
    CACHE STRING
    "Optimization level used while llc emits PTX from lowered NTA IR"
  )
  set_property(CACHE NTA_CUDA_LLC_OPT_LEVEL PROPERTY STRINGS 0 1 2 3)
  # ptxas remains at -O3 below.  The clangir fallback keeps the LLVM-to-PTX
  # translation conservative because its GVN pipeline currently asserts on
  # this lowered module; stable LLVM uses the optimized llc path.
  set(
    NTA_CUDA_ROOT
    "${CUDAToolkit_LIBRARY_ROOT}"
    CACHE PATH "CUDA toolkit root used by clang CUDA"
  )
  if(NOT EXISTS "${NTA_CUDA_ROOT}/include/cuda_runtime.h")
    get_filename_component(NTA_CUDA_ROOT "${CUDAToolkit_BIN_DIR}/.." ABSOLUTE)
  endif()
  # CUDA 12.x keeps CUDA C++ headers below <root>/include, while CUDA 13.x
  # places them in the target sysroot.  All raw-clang device commands must see
  # the same complete header view as the JIT shim; otherwise a CUDA 13 build
  # fails on headers such as <cuda/barrier> even though nvcc itself is valid.
  set(NTA_CUDA_INCLUDE_DIRS "${NTA_CUDA_ROOT}/include")
  file(GLOB NTA_CUDA_TARGET_INCLUDE_DIRS LIST_DIRECTORIES true
    "${NTA_CUDA_ROOT}/targets/*-linux/include")
  foreach(NTA_CUDA_TARGET_INCLUDE_DIR IN LISTS NTA_CUDA_TARGET_INCLUDE_DIRS)
    if(IS_DIRECTORY "${NTA_CUDA_TARGET_INCLUDE_DIR}")
      list(APPEND NTA_CUDA_INCLUDE_DIRS
        "${NTA_CUDA_TARGET_INCLUDE_DIR}")
      if(IS_DIRECTORY "${NTA_CUDA_TARGET_INCLUDE_DIR}/cccl")
        list(APPEND NTA_CUDA_INCLUDE_DIRS
          "${NTA_CUDA_TARGET_INCLUDE_DIR}/cccl")
      endif()
    endif()
  endforeach()
  list(REMOVE_DUPLICATES NTA_CUDA_INCLUDE_DIRS)
  set(NTA_CUDA_INCLUDE_FLAGS)
  foreach(NTA_CUDA_INCLUDE_DIR IN LISTS NTA_CUDA_INCLUDE_DIRS)
    list(APPEND NTA_CUDA_INCLUDE_FLAGS
      -isystem "${NTA_CUDA_INCLUDE_DIR}")
  endforeach()

  add_library(nta-runtime-host STATIC
    runtime/host/CxlRuntime.cpp
    runtime/host/DeviceWorkPlan.cpp
    runtime/host/FinitePhase.cpp
    runtime/host/JitOperator.cpp
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
    nta-runtime-host PUBLIC nta-work-plan nta-cxl-dax-allocator
                             CUDA::cuda_driver CUDA::cudart
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

  # Transport phases are runtime infrastructure, not part of a numerical
  # operator.  Build them once as a compact generic module so serving paths do
  # not JIT an otherwise-unused FlashInfer attention family merely to launch
  # indexed copies or NVMe progress kernels.
  set(NTA_TRANSPORT_PROGRAM
      "${CMAKE_CURRENT_BINARY_DIR}/libnta-transport-program.so")
  add_custom_command(
    OUTPUT "${NTA_TRANSPORT_PROGRAM}"
    COMMAND
      ${CMAKE_COMMAND} -E env
      NTA_CLANG=${NTA_CLANG_CUDA}
      NTA_CUDA_PATH=${NTA_CUDA_ROOT}
      NTA_PROJECT_ROOT=${CMAKE_CURRENT_SOURCE_DIR}
      NTA_JIT_ONLY=__nta_transport_never_instrument__
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/jit/nvcc_clang.py
      -std=c++20
      --expt-relaxed-constexpr
      -O3
      -lineinfo
      -shared
      -Xcompiler -fPIC
      -arch ${NTA_CUDA_ARCH}
      -I${CMAKE_CURRENT_SOURCE_DIR}/include
      -I${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/device/TransportProgram.cu
      -L${NTA_CUDA_ROOT}/targets/x86_64-linux/lib
      -lcudart
      -o "${NTA_TRANSPORT_PROGRAM}"
    DEPENDS
      runtime/device/TransportProgram.cu
      runtime/device/JitRuntime.cuh
      runtime/device/OperatorMetadata.cuh
      runtime/device/Acquire.cuh
      include/nta/KernelPolicy.cuh
      include/nta/OperatorContract.h
      include/nta/RuntimeABI.h
      include/nta/TicketProtocol.cuh
      tools/jit/nvcc_clang.py
      tools/jit/clang_cuda_prelude.h
      tools/jit/cuda_toolkit.py
    VERBATIM
    COMMAND_EXPAND_LISTS
  )
  add_custom_target(nta-transport-program ALL
                    DEPENDS "${NTA_TRANSPORT_PROGRAM}")
  install(
    TARGETS nta-runtime nta-runtime-host
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
  install(FILES "${NTA_TRANSPORT_PROGRAM}"
          DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(
    DIRECTORY include/nta
    DESTINATION ${CMAKE_INSTALL_DATADIR}/nta/include
  )
  install(
    FILES runtime/device/Acquire.cuh runtime/device/JitRuntime.cuh
          runtime/device/TypedInstrumentation.cuh
    DESTINATION ${CMAKE_INSTALL_DATADIR}/nta/runtime/device
  )
  install(
    PROGRAMS tools/jit/nvcc_clang.py tools/jit/cuda_toolkit.py
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
      NAME nta-sglang-calibration
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/sglang_calibration.py
    )
    add_test(
      NAME nta-sglang-calibration-profile
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/sglang_calibration_profile.py
    )
    add_test(
      NAME nta-sglang-acquisition
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/sglang_acquisition.py
    )
    add_test(
      NAME nta-sglang-lifecycle
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/sglang_lifecycle.py
    )
    add_test(
      NAME nta-sglang-storage-connector
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/sglang_storage_connector.py
    )
    add_test(
      NAME nta-bounded-staging
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/bounded_staging.py
    )
    add_test(
      NAME nta-python-runtime
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/python_runtime.py
    )
    add_test(
      NAME nta-import-boundaries
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/import_boundaries.py
    )
    add_test(
      NAME nta-jit-operator-family
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/jit_operator_family.py
    )
    add_test(
      NAME nta-qualification-runner
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/qualification_runner.py
    )
    add_test(
      NAME nta-serving-trial-gates
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/serving_trial_gates.py
    )
    add_test(
      NAME nta-serving-overload-rate
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/serving_overload_rate.py
    )
    add_test(
      NAME nta-gpu-trial
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/gpu_trial.py
    )
    add_test(
      NAME nta-serving-evidence
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/serving_evidence.py
    )
    add_test(
      NAME nta-serving-path-evidence
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/serving_path_evidence.py
    )
    add_test(
      NAME nta-stats-publisher
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/stats_publisher.py
    )
    add_test(
      NAME nta-vllm-lifecycle
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/vllm_lifecycle.py
    )
    add_test(
      NAME nta-opportunity-analysis
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/opportunity.py
    )
    add_test(
      NAME nta-opportunity-study
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/opportunity_study.py
    )
    add_test(
      NAME nta-tier-qualification
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/tier_qualification.py
    )
    add_test(
      NAME nta-evaluation-spec
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/evaluation_spec.py
    )
    add_test(
      NAME nta-evaluation-artifact
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/evaluation_artifact.py
    )
    add_test(
      NAME nta-performance-gate
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/performance_gate.py
    )
    add_test(
      NAME nta-nvme-safety
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/nvme_safety.py
    )
    add_test(
      NAME nta-physical-tier-safety
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/physical_tier_safety.py
    )
    add_test(
      NAME nta-physical-tier-capability
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/physical_tier_capability.py
    )
    set_tests_properties(
      nta-request-slots nta-sglang-plugin nta-sglang-calibration
      nta-sglang-calibration-profile
      nta-sglang-lifecycle
      nta-sglang-storage-connector
      nta-bounded-staging
      nta-qualification-runner
      nta-serving-trial-gates nta-serving-overload-rate
      nta-gpu-trial
      nta-serving-evidence nta-serving-path-evidence nta-jit-operator-family
      nta-stats-publisher nta-vllm-lifecycle
      nta-opportunity-analysis
      nta-opportunity-study nta-tier-qualification
      nta-evaluation-spec nta-evaluation-artifact nta-performance-gate
      nta-nvme-safety nta-physical-tier-safety nta-physical-tier-capability
      PROPERTIES TIMEOUT 60
    )
    set_tests_properties(nta-python-runtime nta-sglang-plugin PROPERTIES
      ENVIRONMENT
        "NTA_RUNTIME_LIBRARY=$<TARGET_FILE:nta-runtime>;PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
    )
    # This integration test imports SGLang in both the parent and a spawned
    # worker.  On a cold artifact machine those imports legitimately exceed
    # the 60-second unit-test budget even though no runtime work is hung.
    set_tests_properties(nta-sglang-plugin PROPERTIES TIMEOUT 300)
    set_tests_properties(nta-request-slots PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
    )
    set_tests_properties(nta-sglang-calibration PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
    )
    set_tests_properties(nta-sglang-calibration-profile PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
    )
    set_tests_properties(nta-sglang-lifecycle PROPERTIES
      ENVIRONMENT
        "NTA_RUNTIME_LIBRARY=$<TARGET_FILE:nta-runtime>;PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
    )
    set_tests_properties(nta-sglang-storage-connector PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
    )
    set_tests_properties(nta-performance-gate PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}"
    )
    set_tests_properties(nta-bounded-staging PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
    )
    set_tests_properties(nta-opportunity-analysis PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
    )
  endif()

  add_executable(nta-jit-phase-load tests/runtime/JitPhaseLoad.cpp)
  add_dependencies(nta-jit-phase-load nta-transport-program)
  target_link_libraries(nta-jit-phase-load PRIVATE nta-runtime-host)
  if(NTA_WARNINGS_AS_ERRORS)
    target_compile_options(
      nta-jit-phase-load PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
  endif()
  add_test(
    NAME nta-transport-program-load
    COMMAND nta-jit-phase-load "${NTA_TRANSPORT_PROGRAM}"
  )
  set_tests_properties(nta-transport-program-load PROPERTIES TIMEOUT 30)
  if(Python3_Interpreter_FOUND)
    add_test(
      NAME nta-transport-program-standalone-load
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/transport_program_standalone.py
              "${NTA_TRANSPORT_PROGRAM}"
    )
    set_tests_properties(nta-transport-program-standalone-load
                         PROPERTIES TIMEOUT 30)
    add_test(
      NAME nta-hbm-compaction-gpu
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/hbm_compaction.py
              "${NTA_TRANSPORT_PROGRAM}"
    )
    set_tests_properties(
      nta-hbm-compaction-gpu PROPERTIES
      ENVIRONMENT
        "NTA_RUNTIME_LIBRARY=$<TARGET_FILE:nta-runtime>;PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
      LABELS "correctness;nvme;gpu"
      TIMEOUT 30
    )
    add_test(
      NAME nta-hbm-compaction-calibration-gpu
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/nvme/HbmCompactionCalibration.py
              "${NTA_TRANSPORT_PROGRAM}"
              --pool-mib=256
              --trials=11
              --output=${CMAKE_CURRENT_BINARY_DIR}/physical-results/profiling/hbm-compaction-calibration-smoke.json
    )
    set_tests_properties(
      nta-hbm-compaction-calibration-gpu PROPERTIES
      ENVIRONMENT
        "NTA_RUNTIME_LIBRARY=$<TARGET_FILE:nta-runtime>;PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}"
      LABELS "performance;calibration;nvme;gpu"
      PASS_REGULAR_EXPRESSION "nta-hbm-compaction-calibration"
      RUN_SERIAL TRUE
      TIMEOUT 30
    )
    add_test(
      NAME nta-ready-object-gate-gpu
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/ready_object_gate.py
              "${NTA_TRANSPORT_PROGRAM}"
    )
    set_tests_properties(
      nta-ready-object-gate-gpu PROPERTIES
      ENVIRONMENT
        "NTA_RUNTIME_LIBRARY=$<TARGET_FILE:nta-runtime>;PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
      LABELS "correctness;nvme;gpu"
      RUN_SERIAL TRUE
      TIMEOUT 30
    )
  endif()
  add_executable(nta-jit-operator-load tests/runtime/JitOperatorLoad.cpp)
  target_link_libraries(nta-jit-operator-load PRIVATE nta-runtime-host)
  if(NTA_WARNINGS_AS_ERRORS)
    target_compile_options(
      nta-jit-operator-load PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
  endif()
  if(Python3_Interpreter_FOUND)
    add_test(
      NAME nta-ordered-host-preload-gpu
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/ordered_host_preload.py
              "${NTA_TRANSPORT_PROGRAM}"
    )
    set_tests_properties(
      nta-ordered-host-preload-gpu
      PROPERTIES
        ENVIRONMENT
          "NTA_RUNTIME_LIBRARY=$<TARGET_FILE:nta-runtime>;PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
        RUN_SERIAL TRUE
        TIMEOUT 60
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
      ${NTA_CUDA_FRONTEND_FLAGS}
      -std=c++20
      --cuda-device-only
      --cuda-path=${NTA_CUDA_ROOT}
      --cuda-gpu-arch=${NTA_CUDA_ARCH}
      -Wno-unknown-cuda-version
      ${NTA_CUDA_INCLUDE_FLAGS}
      -I${CMAKE_CURRENT_SOURCE_DIR}/include
      -I${CMAKE_CURRENT_SOURCE_DIR}
      -emit-llvm
      -c
      ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/kv/KvAcquireKernel.cu
      -o "${NTA_KV_RAW_BC}"
    COMMAND
      ${LLVM_TOOLS_BINARY_DIR}/opt
      -load-pass-plugin=$<TARGET_FILE:NtaPass>
      -passes=${NTA_ACQUIRE_PREPASSES},nta-acquire,always-inline
      -S
      "${NTA_KV_RAW_BC}"
      -o "${NTA_KV_LOWERED_IR}"
    COMMAND
      ${LLVM_TOOLS_BINARY_DIR}/llc
      -march=nvptx64
      -mcpu=${NTA_CUDA_ARCH}
      -O${NTA_CUDA_LLC_OPT_LEVEL}
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
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
      "import importlib.util; raise SystemExit(0 if all(importlib.util.find_spec(name) for name in ('sglang', 'vllm')) else 1)"
    RESULT_VARIABLE NTA_HAVE_COMMON_ENGINES
    OUTPUT_QUIET
    ERROR_QUIET
  )
  if(NTA_HAVE_COMMON_ENGINES EQUAL 0)
    add_test(
      NAME nta-engine-environment
      COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/engine_environment.py
    )
    set_tests_properties(nta-engine-environment PROPERTIES TIMEOUT 60)
  endif()
  add_test(
    NAME nta-moe-lowering
      COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/ir/moe_lowering.py
        ${NTA_KV_LOWERED_IR}
    )
    set_tests_properties(nta-moe-lowering PROPERTIES TIMEOUT 30)
  endif()

  # CTest may be run once as root for physical tiers and later as the artifact
  # user. Keep this small compiler-output fixture per Unix UID just like the
  # runtime JIT cache, so a privileged run cannot poison a normal run.
  execute_process(
    COMMAND id -u
    OUTPUT_VARIABLE NTA_JIT_UID
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(NOT NTA_JIT_UID)
    set(NTA_JIT_UID unknown)
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
      ${CMAKE_CURRENT_BINARY_DIR}/jit-test/uid-${NTA_JIT_UID}
  )
  if(NTA_HAVE_FLASHINFER AND Python3_Interpreter_FOUND)
    add_test(
      NAME nta-flashinfer-jit-compile
      COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/jit/activate.py
        --build-dir ${CMAKE_CURRENT_BINARY_DIR}
        --cache-root ${NTA_TEST_JIT_CACHE_ROOT}
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
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/jit/activate.py
        --build-dir ${CMAKE_CURRENT_BINARY_DIR}
        --cache-root ${NTA_TEST_JIT_CACHE_ROOT}
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
    if(NTA_TEST_NVME_ENDPOINT AND NTA_TEST_NVME_REFERENCE AND
       NTA_TEST_NVME_MEDIA_POLICY AND
       NTA_TEST_NVME_DMA_TARGET STREQUAL "hbm-peer")
      add_test(
        NAME nta-sglang-nvme-physical
        COMMAND ${NTA_TEST_NVME_COMMAND_PREFIX}
          ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tools/jit/activate.py
          --build-dir ${CMAKE_CURRENT_BINARY_DIR}
          --cache-root ${NTA_TEST_JIT_CACHE_ROOT}
          --clang ${NTA_CLANG_CUDA}
          --cuda-path ${NTA_CUDA_ROOT}
          --
          ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/sglang_nvme_physical.py
      )
      set_tests_properties(
        nta-sglang-nvme-physical PROPERTIES
        RUN_SERIAL TRUE TIMEOUT 180 RESOURCE_LOCK nta_nvme_physical
        LABELS "physical;correctness;nvme;sglang;pipeline"
        PASS_REGULAR_EXPRESSION "sglang_nvme_physical=pass"
        ENVIRONMENT "${NTA_TEST_NVME_ENVIRONMENT}"
      )
      add_test(
        NAME nta-sglang-nvme-publication-performance
        COMMAND ${NTA_TEST_NVME_COMMAND_PREFIX}
          ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tools/jit/activate.py
          --build-dir ${CMAKE_CURRENT_BINARY_DIR}
          --cache-root ${NTA_TEST_JIT_CACHE_ROOT}
          --clang ${NTA_CLANG_CUDA}
          --cuda-path ${NTA_CUDA_ROOT}
          --
          ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/nvme/SglangPipelinePublication.py
          --warmup=1
          --trials=3
          --minimum-publication-prepare-speedup=1.15
          --minimum-publication-wall-speedup=1.02
          --minimum-ordered-wall-speedup=1.25
          --minimum-window-wall-speedup=1.03
          --output=${CMAKE_CURRENT_BINARY_DIR}/physical-results/profiling/sglang-nvme-publication.json
      )
      set_tests_properties(
        nta-sglang-nvme-publication-performance PROPERTIES
        RUN_SERIAL TRUE TIMEOUT 300 RESOURCE_LOCK nta_nvme_physical
        LABELS "physical;performance;nvme;sglang;pipeline"
        ENVIRONMENT "${NTA_TEST_NVME_ENVIRONMENT}"
      )
      if(NTA_TEST_NVME_SERVICE_CALIBRATED)
        add_test(
          NAME nta-sglang-nvme-granularity-sparse-performance
          COMMAND ${NTA_TEST_NVME_COMMAND_PREFIX}
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/jit/activate.py
            --build-dir ${CMAKE_CURRENT_BINARY_DIR}
            --cache-root ${NTA_TEST_JIT_CACHE_ROOT}
            --clang ${NTA_CLANG_CUDA}
            --cuda-path ${NTA_CUDA_ROOT}
            --
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/nvme/SglangPipelinePublication.py
            --warmup=1
            --trials=3
            --source-stride=2
            --command-service-ns=${NTA_TEST_NVME_COMMAND_SERVICE_NS}
            --read-bandwidth-bps=${NTA_TEST_NVME_READ_BANDWIDTH_BPS}
            --compaction-bandwidth-bps=${NTA_TEST_NVME_COMPACTION_BANDWIDTH_BPS}
            --compaction-launch-ns=${NTA_TEST_NVME_COMPACTION_LAUNCH_NS}
            --minimum-granularity-gain=${NTA_TEST_NVME_MINIMUM_GAIN}
            --minimum-granularity-wall-speedup=2.0
            --output=${CMAKE_CURRENT_BINARY_DIR}/physical-results/profiling/sglang-nvme-granularity-sparse.json
        )
        add_test(
          NAME nta-sglang-nvme-granularity-dense-no-regression
          COMMAND ${NTA_TEST_NVME_COMMAND_PREFIX}
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/jit/activate.py
            --build-dir ${CMAKE_CURRENT_BINARY_DIR}
            --cache-root ${NTA_TEST_JIT_CACHE_ROOT}
            --clang ${NTA_CLANG_CUDA}
            --cuda-path ${NTA_CUDA_ROOT}
            --
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/nvme/SglangPipelinePublication.py
            --warmup=2
            --trials=7
            --source-stride=1
            --command-service-ns=${NTA_TEST_NVME_COMMAND_SERVICE_NS}
            --read-bandwidth-bps=${NTA_TEST_NVME_READ_BANDWIDTH_BPS}
            --compaction-bandwidth-bps=${NTA_TEST_NVME_COMPACTION_BANDWIDTH_BPS}
            --compaction-launch-ns=${NTA_TEST_NVME_COMPACTION_LAUNCH_NS}
            --minimum-granularity-gain=${NTA_TEST_NVME_MINIMUM_GAIN}
            --minimum-granularity-wall-speedup=0.97
            --output=${CMAKE_CURRENT_BINARY_DIR}/physical-results/profiling/sglang-nvme-granularity-dense.json
        )
        set_tests_properties(
          nta-sglang-nvme-granularity-sparse-performance
          nta-sglang-nvme-granularity-dense-no-regression
          PROPERTIES
            RUN_SERIAL TRUE TIMEOUT 300 RESOURCE_LOCK nta_nvme_physical
            LABELS "physical;performance;nvme;sglang;pipeline;granularity"
            ENVIRONMENT "${NTA_TEST_NVME_ENVIRONMENT}"
        )
      endif()
      add_test(
        NAME nta-flashinfer-nvme-numerical
        COMMAND ${NTA_TEST_NVME_COMMAND_PREFIX}
          ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tools/jit/activate.py
          --build-dir ${CMAKE_CURRENT_BINARY_DIR}
          --cache-root ${NTA_TEST_JIT_CACHE_ROOT}
          --clang ${NTA_CLANG_CUDA}
          --cuda-path ${NTA_CUDA_ROOT}
          --flashinfer-hook
          --
          ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/flashinfer/hooked_decode.py
          --nvme-numerical-only
      )
      set_tests_properties(
        nta-flashinfer-nvme-numerical PROPERTIES
        RUN_SERIAL TRUE TIMEOUT 300 RESOURCE_LOCK nta_nvme_physical
        LABELS "physical;correctness;nvme;flashinfer;numerical"
        PASS_REGULAR_EXPRESSION
          "nvme_flashinfer_numerical=pass payload_alias=1"
        ENVIRONMENT "${NTA_TEST_NVME_ENVIRONMENT}"
      )
    endif()
    set_tests_properties(
      nta-indexed-row-count-seam
      PROPERTIES DEPENDS nta-flashinfer-jit-compile
    )
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c
        "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('vllm') else 1)"
      RESULT_VARIABLE NTA_HAVE_VLLM
      OUTPUT_QUIET
      ERROR_QUIET
    )
    if(NTA_HAVE_VLLM EQUAL 0)
      add_test(
        NAME nta-vllm-native
        COMMAND
          ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tools/jit/activate.py
          --build-dir ${CMAKE_CURRENT_BINARY_DIR}
          --cache-root ${NTA_TEST_JIT_CACHE_ROOT}
          --clang ${NTA_CLANG_CUDA}
          --cuda-path ${NTA_CUDA_ROOT}
          --flashinfer-hook
          --
          ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/flashinfer/vllm_native.py
      )
      set_tests_properties(
        nta-vllm-native
        PROPERTIES DEPENDS nta-flashinfer-jit-compile
                   RUN_SERIAL TRUE
                   TIMEOUT 300
      )
      if(NTA_TEST_VLLM_MODEL)
        add_test(
          NAME nta-vllm-serving-e2e
          COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/serving/CompareVllmSmoke.py
            --model ${NTA_TEST_VLLM_MODEL}
            --cache-root ${NTA_TEST_JIT_CACHE_ROOT}/vllm-serving-e2e
            --output
              ${CMAKE_CURRENT_BINARY_DIR}/artifacts/vllm-serving-e2e.json
        )
        set_tests_properties(
          nta-vllm-serving-e2e
          PROPERTIES DEPENDS nta-vllm-native
                     RUN_SERIAL TRUE
                     TIMEOUT 1200
                     LABELS "gpu;integration;vllm;serving;e2e"
                     PASS_REGULAR_EXPRESSION
                       "\\\"numerically_equivalent\\\": true"
        )
      endif()
    endif()
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
      ${NTA_CUDA_FRONTEND_FLAGS}
      -std=c++20
      --cuda-device-only
      --cuda-path=${NTA_CUDA_ROOT}
      --cuda-gpu-arch=${NTA_CUDA_ARCH}
      -Wno-unknown-cuda-version
      ${NTA_CUDA_INCLUDE_FLAGS}
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
      -passes=${NTA_ACQUIRE_PREPASSES},nta-acquire,always-inline
      -S
      "${NTA_ATTENTION_RAW_BC}"
      -o "${NTA_ATTENTION_LOWERED_IR}"
    COMMAND
      ${LLVM_TOOLS_BINARY_DIR}/llc
      -march=nvptx64
      -mcpu=${NTA_CUDA_ARCH}
      -O${NTA_CUDA_LLC_OPT_LEVEL}
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
  add_dependencies(nta-kv-bench nta-kv-kernel nta-transport-program)
  target_include_directories(nta-kv-bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_definitions(
    nta-kv-bench PRIVATE
    NTA_KV_CUBIN_PATH="${NTA_KV_CUBIN}"
    NTA_TRANSPORT_PROGRAM_PATH="${NTA_TRANSPORT_PROGRAM}"
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
  add_dependencies(nta-moe-bench nta-kv-kernel nta-transport-program)
  target_include_directories(nta-moe-bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_definitions(
    nta-moe-bench PRIVATE
    NTA_KV_CUBIN_PATH="${NTA_KV_CUBIN}"
    NTA_TRANSPORT_PROGRAM_PATH="${NTA_TRANSPORT_PROGRAM}"
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
  if(NTA_TEST_NVME_ENDPOINT AND NTA_TEST_NVME_REFERENCE AND
     NTA_TEST_NVME_MEDIA_POLICY AND
     NTA_TEST_NVME_DMA_TARGET STREQUAL "hbm-peer")
    add_test(
      NAME nta-nvme-external-hbm-offset-gpu
      COMMAND ${NTA_TEST_NVME_COMMAND_PREFIX}
        $<TARGET_FILE:nta-nvme-bench>
        --device=${NTA_TEST_NVME_ENDPOINT}
        --gpu=0
        --namespace=1
        --queue-depth=64
        --media-policy=${NTA_TEST_NVME_MEDIA_POLICY}
        --dma-target=hbm-peer
        --hbm-backend=${NTA_TEST_NVME_HBM_BACKEND}
        --reference=${NTA_TEST_NVME_REFERENCE}
        --source-offset=512
        --bytes=65536
        --requests=4
        --iterations=2
        --progress-rounds=1
        --external-destination-offset=512
    )
    set_tests_properties(
      nta-nvme-external-hbm-offset-gpu PROPERTIES
      TIMEOUT 120 RESOURCE_LOCK nta_nvme_physical
      LABELS "physical;correctness;nvme;external-hbm"
      ENVIRONMENT "${NTA_TEST_NVME_ENVIRONMENT}"
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
  add_test(
    NAME nta-vfio-nvme-probe
    COMMAND ${NTA_TEST_NVME_COMMAND_PREFIX}
            $<TARGET_FILE:nta-vfio-nvme-probe> ${NTA_TEST_NVME_PROBE_ARGUMENTS}
  )
  set_tests_properties(nta-vfio-nvme-probe PROPERTIES
    TIMEOUT 60 SKIP_RETURN_CODE 77 RESOURCE_LOCK nta_nvme_physical
    LABELS "physical;qualification;nvme"
    ENVIRONMENT "${NTA_TEST_NVME_ENVIRONMENT}"
  )

  add_executable(nta-cxl-dax-probe tools/CxlDaxProbe.cpp)
  target_include_directories(nta-cxl-dax-probe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(
    nta-cxl-dax-probe PRIVATE nta-runtime-host CUDA::cuda_driver CUDA::cudart
  )
  if(NTA_WARNINGS_AS_ERRORS)
    target_compile_options(
      nta-cxl-dax-probe PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
  endif()
  add_test(NAME nta-cxl-dax-probe COMMAND nta-cxl-dax-probe)
  set_tests_properties(nta-cxl-dax-probe PROPERTIES
    TIMEOUT 60 SKIP_RETURN_CODE 77 LABELS "physical;qualification;cxl"
  )
  add_test(
    NAME nta-cxl-dax-rejects-pseudo-endpoint
    COMMAND nta-cxl-dax-probe /dev/zero 1 0 --json=1
  )
  set_tests_properties(
    nta-cxl-dax-rejects-pseudo-endpoint PROPERTIES
    TIMEOUT 60 WILL_FAIL TRUE LABELS "contract;qualification;cxl"
  )

  add_executable(
    nta-paged-attention benchmarks/attention/PagedAttention.cpp
  )
  add_dependencies(
    nta-paged-attention nta-attention-kernel nta-transport-program
  )
  target_include_directories(nta-paged-attention PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_definitions(
    nta-paged-attention
    PRIVATE
    NTA_ATTENTION_CUBIN_PATH="${NTA_ATTENTION_CUBIN}"
    NTA_TRANSPORT_PROGRAM_PATH="${NTA_TRANSPORT_PROGRAM}"
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

  add_executable(
    nta-indexed-intent-lifecycle-test
    tests/runtime/IndexedIntentLifecycleTest.cpp
  )
  add_dependencies(nta-indexed-intent-lifecycle-test nta-transport-program)
  target_include_directories(nta-indexed-intent-lifecycle-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_compile_definitions(
    nta-indexed-intent-lifecycle-test PRIVATE
    NTA_TRANSPORT_PROGRAM_PATH="${NTA_TRANSPORT_PROGRAM}"
  )
  target_link_libraries(
    nta-indexed-intent-lifecycle-test PRIVATE nta-runtime-host CUDA::cudart
  )
  if(NTA_WARNINGS_AS_ERRORS)
    target_compile_options(
      nta-indexed-intent-lifecycle-test
      PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
  endif()
  add_test(
    NAME nta-indexed-intent-lifecycle
    COMMAND nta-indexed-intent-lifecycle-test
  )
  set_tests_properties(nta-indexed-intent-lifecycle PROPERTIES TIMEOUT 60)

  set(NTA_DEPENDENCY_RACE_PTX
      "${CMAKE_CURRENT_BINARY_DIR}/kernel/DependencyArrivalRace.ptx")
  set(NTA_DEPENDENCY_RACE_CUBIN
      "${CMAKE_CURRENT_BINARY_DIR}/kernel/DependencyArrivalRace.cubin")
  add_custom_command(
    OUTPUT "${NTA_DEPENDENCY_RACE_CUBIN}"
    BYPRODUCTS "${NTA_DEPENDENCY_RACE_PTX}"
    COMMAND
      "${NTA_CLANG_CUDA}"
      ${NTA_CUDA_FRONTEND_FLAGS}
      -std=c++20
      --cuda-device-only
      --cuda-path=${NTA_CUDA_ROOT}
      --cuda-gpu-arch=${NTA_CUDA_ARCH}
      -Wno-unknown-cuda-version
      ${NTA_CUDA_INCLUDE_FLAGS}
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
      --progress-rounds=4
      --request-credit-pages=1
  )
  set_tests_properties(nta-paged-attention-gpu PROPERTIES TIMEOUT 60)
  add_test(
    NAME nta-paged-attention-sparse-partial-acquire-gpu
    COMMAND
      $<TARGET_FILE:nta-paged-attention>
      --mode=host-staged
      --requests=4
      --min-pages=4
      --max-pages=6
      --sparse-top-k=2
      --sparse-policy=partial-acquire
      --iterations=2
      --progress-rounds=2
      --request-credit-pages=1
  )
  set_tests_properties(
    nta-paged-attention-sparse-partial-acquire-gpu PROPERTIES
    TIMEOUT 60
    PASS_REGULAR_EXPRESSION
      "policy=partial-acquire.*work_granularity=selected-page.*verification_failures=0"
  )
  add_test(
    NAME nta-paged-attention-sparse-whole-acquire-gpu
    COMMAND
      $<TARGET_FILE:nta-paged-attention>
      --mode=host-staged
      --requests=4
      --min-pages=4
      --max-pages=6
      --sparse-top-k=2
      --sparse-policy=whole-acquire
      --iterations=2
      --progress-rounds=2
      --request-credit-pages=1
  )
  set_tests_properties(
    nta-paged-attention-sparse-whole-acquire-gpu PROPERTIES
    TIMEOUT 60
    PASS_REGULAR_EXPRESSION
      "policy=whole-acquire.*work_granularity=request-selection.*verification_failures=0"
  )
  add_test(
    NAME nta-paged-attention-sparse-selected-gather-gpu
    COMMAND
      $<TARGET_FILE:nta-paged-attention>
      --mode=host-staged
      --requests=4
      --min-pages=4
      --max-pages=6
      --sparse-top-k=2
      --sparse-policy=selected-gather
      --iterations=2
      --progress-rounds=1
      --request-credit-pages=2
  )
  set_tests_properties(
    nta-paged-attention-sparse-selected-gather-gpu PROPERTIES
    TIMEOUT 60
    PASS_REGULAR_EXPRESSION
      "policy=selected-gather.*overfetch_ratio=1.00.*verification_failures=0"
  )
  add_test(
    NAME nta-paged-attention-sparse-selected-partial-gpu
    COMMAND
      $<TARGET_FILE:nta-paged-attention>
      --mode=host-staged
      --requests=4
      --min-pages=4
      --max-pages=6
      --sparse-top-k=2
      --sparse-policy=selected-partial
      --iterations=2
      --progress-rounds=1
      --request-credit-pages=2
  )
  set_tests_properties(
    nta-paged-attention-sparse-selected-partial-gpu PROPERTIES
    TIMEOUT 60
    PASS_REGULAR_EXPRESSION
      "policy=selected-partial.*work_granularity=selected-page.*overfetch_ratio=1.00.*verification_failures=0"
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
      --progress-rounds=1
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
      --progress-rounds=4
      --request-credit-pages=1
  )
  set_tests_properties(nta-paged-attention-tma-gpu PROPERTIES TIMEOUT 60)
  add_test(
    NAME nta-paged-attention-rejects-underprovisioned-progress
    COMMAND
      $<TARGET_FILE:nta-paged-attention>
      --mode=host-staged
      --requests=4
      --min-pages=2
      --max-pages=4
      --iterations=1
      --progress-rounds=1
      --request-credit-pages=1
  )
  set_tests_properties(
    nta-paged-attention-rejects-underprovisioned-progress PROPERTIES
    TIMEOUT 10
    WILL_FAIL TRUE
  )
  add_test(
    NAME nta-paged-attention-dax-gpu
    COMMAND
      $<TARGET_FILE:nta-paged-attention>
      --mode=dax
      --requests=4
      --min-pages=2
      --max-pages=4
      --iterations=2
      --progress-rounds=1
  )
  set_tests_properties(
    nta-paged-attention-dax-gpu PROPERTIES
    TIMEOUT 60
    SKIP_RETURN_CODE 77
  )
  add_test(
    NAME nta-paged-attention-nvme-gpu
    COMMAND ${NTA_TEST_NVME_COMMAND_PREFIX}
      $<TARGET_FILE:nta-paged-attention>
      --mode=nvme
      --requests=4
      --min-pages=2
      --max-pages=4
      --iterations=2
      --progress-rounds=1
  )
  set_tests_properties(
    nta-paged-attention-nvme-gpu PROPERTIES
    TIMEOUT 120 RESOURCE_LOCK nta_nvme_physical
    SKIP_RETURN_CODE 77
    ENVIRONMENT "${NTA_TEST_NVME_ENVIRONMENT}"
  )
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
    add_test(
      NAME nta-flashinfer-tier-streaming-gpu
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/serving/FlashInferTierStreaming.py"
        --batch-size 3
        --query-tokens 16,24,32
        --context-tokens 110,90,90
        --resident-fractions 0.090909,0,0.111111
        --group-tokens 64
        --workspace-mib 64
        --warmup 1
        --iterations 1
        --trials 1
        --verify-graph
    )
    set_tests_properties(
      nta-flashinfer-tier-streaming-gpu PROPERTIES
      TIMEOUT 180
      ENVIRONMENT
        "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python;FLASHINFER_NVCC=${CMAKE_CURRENT_SOURCE_DIR}/tools/flashinfer/nvcc_compat.py;NTA_REAL_NVCC=${CUDAToolkit_NVCC_EXECUTABLE}"
    )
    add_test(
      NAME nta-flashinfer-tier-streaming-compiler-gpu
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/jit/activate.py"
        --build-dir "${CMAKE_CURRENT_BINARY_DIR}"
        --cache-root "${NTA_TEST_JIT_CACHE_ROOT}"
        --cuda-path "${NTA_CUDA_ROOT}"
        --flashinfer-hook
        --
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/serving/FlashInferTierStreaming.py"
        --batch-size 2
        --query-tokens 16
        --context-tokens 128
        --resident-fractions 0,0.5
        --group-tokens 64
        --workspace-mib 64
        --warmup 1
        --iterations 10
        --trials 1
        --verify-graph
        --verify-lifecycle
        --compiler-transform
    )
    set_tests_properties(
      nta-flashinfer-tier-streaming-compiler-gpu PROPERTIES
      TIMEOUT 360
      PASS_REGULAR_EXPRESSION
        "compiler_transformed_attention.*true.*output_parity.*true"
    )
    add_test(
      NAME nta-flashinfer-tier-streaming-gpu-initiated-host-gpu
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/jit/activate.py"
        --build-dir "${CMAKE_CURRENT_BINARY_DIR}"
        --cache-root "${NTA_TEST_JIT_CACHE_ROOT}"
        --cuda-path "${NTA_CUDA_ROOT}"
        --flashinfer-hook
        --
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/serving/FlashInferTierStreaming.py"
        --batch-size 2
        --query-tokens 16
        --context-tokens 128
        --resident-fractions 0,0.5
        --group-tokens 64
        --workspace-mib 64
        --warmup 2
        --iterations 2
        --trials 1
        --verify-graph
        --verify-lifecycle
        --compiler-transform
        --gpu-initiated-host
    )
    set_tests_properties(
      nta-flashinfer-tier-streaming-gpu-initiated-host-gpu PROPERTIES
      TIMEOUT 360
      PASS_REGULAR_EXPRESSION
        "host_transfer_initiator.*gpu.*output_parity.*true"
    )
  endif()
endif()

if(NTA_TEST_NVME_MANAGE_SESSION STREQUAL "ON")
  foreach(_test IN ITEMS
          nta-sglang-nvme-physical
          nta-sglang-nvme-publication-performance
          nta-sglang-nvme-granularity-sparse-performance
          nta-sglang-nvme-granularity-dense-no-regression
          nta-flashinfer-nvme-numerical
          nta-nvme-external-hbm-offset-gpu
          nta-vfio-nvme-probe
          nta-paged-attention-nvme-gpu)
    if(TEST ${_test})
      set_property(
        TEST ${_test} APPEND PROPERTY FIXTURES_REQUIRED nta_nvme_session
      )
    endif()
  endforeach()
endif()

add_custom_target(
  nta-vfio-preflight
  COMMAND
    ${CMAKE_COMMAND} -E env
    ${NTA_TEST_NVME_ENVIRONMENT}
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/nta-vfio-device.sh preflight
  USES_TERMINAL
  COMMAND_EXPAND_LISTS
)
