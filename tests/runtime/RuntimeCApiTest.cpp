#include "nta/RuntimeC.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void requireCuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

void requireOk(nta_status status, const char *operation) {
  if (status != NTA_STATUS_OK) {
    throw std::runtime_error(std::string(operation) + ": " + nta_last_error());
  }
}

} // namespace

int main() {
  try {
    require(nta_runtime_c_api_version() == NTA_RUNTIME_C_API_VERSION,
            "C API version mismatch");
    require(nta_runtime_device_abi_version() != 0,
            "device ABI version is zero");

    nta_runtime_config invalid{};
    invalid.api_version = NTA_RUNTIME_C_API_VERSION;
    nta_runtime *runtime = nullptr;
    require(nta_runtime_create(&invalid, nullptr, &runtime) ==
                NTA_STATUS_INVALID_ARGUMENT,
            "runtime accepted an invalid config size");
    require(runtime == nullptr && nta_last_error()[0] != '\0',
            "runtime error did not fail closed");

    nta_runtime_config config{};
    config.struct_size = sizeof(config);
    config.api_version = NTA_RUNTIME_C_API_VERSION;
    config.request_capacity = 2;
    config.object_capacity = 2;
    config.intent_capacity = 4;
    config.work_ticket_capacity = 2;
    config.max_replicas_per_object = 1;
    config.max_dependencies_per_work_ticket = 1;
    config.device_ordinal = NTA_RUNTIME_USE_CURRENT_DEVICE;
    config.enable_cta_nvme_try_issue = 1;
    requireOk(nta_runtime_create(&config, nullptr, &runtime),
              "create C runtime");
    require(runtime != nullptr && nta_runtime_device_view(runtime) != 0,
            "C runtime has no device view");

    requireOk(nta_runtime_set_tenant_budget(runtime, 0, 4096, 1),
              "set tenant budget");
    requireOk(nta_runtime_set_request(
                  runtime, 0, 17, 3, 0, 4, 0,
                  std::numeric_limits<std::uint64_t>::max()),
              "set request");

    void *deviceObject = nullptr;
    requireCuda(cudaMalloc(&deviceObject, 4096), "cudaMalloc object");
    nta_registered_replica replica{};
    replica.source_device_address =
        reinterpret_cast<std::uintptr_t>(deviceObject);
    replica.placement = NTA_PLACEMENT_HBM;
    std::uint64_t directBase = 0;
    requireOk(nta_runtime_register_object(runtime, 0, 101, 7, 4096, 0,
                                          &replica, 1, &directBase),
              "register C object");
    require(directBase == reinterpret_cast<std::uintptr_t>(deviceObject),
            "C object direct base mismatch");

    nta_device_work_plan *plan = nullptr;
    requireOk(nta_device_work_plan_create(2, 2,
                                          nta_runtime_device_ordinal(runtime),
                                          &plan),
              "create C work plan");
    nta_acquire_requirement dependency{
        directBase, 0, 101, 0, 0, 7, 4096, 0,
    };
    nta_work_item work{0, 0, 3, 11, 0, 1, 1, 0,
                       0, 0, 1, 2500, 0, 0, 0, 0};
    nta_request_work_range request{0, 1, 0, 3};
    requireOk(nta_device_work_plan_upload(plan, &work, 1, &dependency, 1,
                                          &request, 1, 0),
              "upload C work plan");
    requireOk(nta_device_work_plan_synchronize_upload(plan),
              "synchronize C work plan");
    require(nta_device_work_plan_work_items(plan) != 0 &&
                nta_device_work_plan_dependencies(plan) != 0 &&
                nta_device_work_plan_work_item_count(plan) == 1 &&
                nta_device_work_plan_dependency_count(plan) == 1,
            "C work plan did not expose its uploaded view");
    require(nta_device_work_plan_device_ordinal(plan) ==
                nta_runtime_device_ordinal(runtime),
            "C work plan device owner mismatch");

    void *managedTensor = nullptr;
    requireOk(nta_device_pointer_dlpack(nta_runtime_device_view(runtime), 1,
                                        nta_runtime_device_ordinal(runtime),
                                        &managedTensor),
              "create DLPack runtime view");
    require(managedTensor != nullptr, "DLPack runtime view is null");
    nta_dlpack_managed_tensor_destroy(managedTensor);

    std::uint32_t pending = 1;
    requireOk(nta_runtime_read_pending_count(runtime, &pending),
              "read C pending count");
    require(pending == 0, "new C runtime unexpectedly has pending work");
    nta_epoch_status epoch{};
    requireOk(nta_runtime_read_epoch_status(runtime, 2, &epoch),
              "read C epoch status");
    require(epoch.total == 2 && epoch.fresh == 2 && epoch.pending == 0,
            "C epoch status did not summarize work-ticket state");
    nta_request_progress progress{};
    requireOk(nta_runtime_read_request_progress(runtime, 0, &progress),
              "read C request progress");
    require(progress.expected_work == 0 && progress.completed_work == 0 &&
                progress.unavailable_bytes == 0 &&
                progress.runnable_compute_ns == 0 &&
                progress.completed_compute_ns == 0,
            "new C runtime unexpectedly reported request work");
    nta_request_progress progressRange[2]{};
    requireOk(nta_runtime_read_request_progress_range(runtime, 0, 2,
                                                       progressRange),
              "read C request-progress range");
    require(progressRange[0].request_id == 17 &&
                progressRange[0].generation == 3,
            "C request-progress range lost request identity");
    std::uint64_t runnableNs[2]{1, 1};
    requireOk(nta_runtime_read_work_runnable_ns(runtime, 2, runnableNs),
              "read C work runnable timestamps");
    require(runnableNs[0] == 0 && runnableNs[1] == 0,
            "new C runtime unexpectedly reported runnable delay");
    requireOk(nta_stream_synchronize(0), "synchronize C default stream");

    nta_nvme_transport_options nvme{};
    nvme.struct_size = sizeof(nvme);
    nvme.api_version = NTA_RUNTIME_C_API_VERSION;
    nvme.endpoint = "/dev/nta_nvme";
    nvme.device_ordinal = NTA_RUNTIME_USE_CURRENT_DEVICE;
    nvme.namespace_id = 1;
    nvme.queue_depth = 64;
    nvme.admin_timeout_ms = 10'000;
    nvme.media_policy = NTA_NVME_REQUIRE_HARDWARE_WRITE_PROTECTION;
    nta_nvme_transport *transport = nullptr;
    require(nta_nvme_transport_create(&nvme, &transport) ==
                NTA_STATUS_INVALID_ARGUMENT,
            "C API accepted a non-VFIO NVMe endpoint");
    require(transport == nullptr,
            "failed C NVMe creation returned a transport handle");

    nta_device_work_plan_destroy(plan);
    nta_runtime_destroy(runtime);
    requireCuda(cudaFree(deviceObject), "cudaFree object");
    std::cout << "runtime_c_api=pass\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
