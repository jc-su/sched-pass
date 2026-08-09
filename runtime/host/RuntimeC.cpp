#include "nta/RuntimeC.h"

#include "nta/DeviceWorkPlan.h"
#include "nta/HostRuntime.h"
#include "nta/JitPhase.h"
#include "nta/NvmeRuntime.h"
#include "nta/WorkPlan.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct nta_nvme_transport {
  std::shared_ptr<nta::NvmeTransport> value;
};

struct nta_runtime {
  std::shared_ptr<nta::NvmeTransport> nvme;
  std::unique_ptr<nta::HostRuntime> value;
};

struct nta_device_work_plan {
  std::unique_ptr<nta::DeviceWorkPlan> value;
};

struct nta_jit_phase_program {
  std::unique_ptr<nta::JitPhaseProgram> value;
};

namespace {

static_assert(sizeof(nta_work_item) == sizeof(nta::abi::WorkItem));
static_assert(offsetof(nta_work_item, estimated_compute_ns) ==
              offsetof(nta::abi::WorkItem, estimatedComputeNs));
static_assert(sizeof(nta_request_progress) ==
              sizeof(nta::abi::RequestProgress));
static_assert(offsetof(nta_request_progress, dropped_attributions) ==
              offsetof(nta::abi::RequestProgress, droppedAttributions));
static_assert(sizeof(nta_operator_contract) ==
              sizeof(nta::operator_contract::Contract));
static_assert(offsetof(nta_operator_contract, capabilities) ==
              offsetof(nta::operator_contract::Contract, capabilities));
static_assert(sizeof(nta_operator_plan) ==
              sizeof(nta::operator_contract::Plan));
static_assert(offsetof(nta_operator_plan, plan_fingerprint_low) ==
              offsetof(nta::operator_contract::Plan, planFingerprintLow));
static_assert(sizeof(nta_request_spec) == 40);
static_assert(offsetof(nta_request_spec, request_id) == 0);
static_assert(offsetof(nta_request_spec, deadline_clock) == 8);
static_assert(offsetof(nta_request_spec, max_outstanding_bytes) == 16);
static_assert(offsetof(nta_request_spec, slot) == 24);
static_assert(offsetof(nta_request_spec, generation) == 28);
static_assert(offsetof(nta_request_spec, tenant_id) == 32);
static_assert(offsetof(nta_request_spec, priority) == 36);

struct DLDevice {
  std::int32_t deviceType;
  std::int32_t deviceId;
};

struct DLDataType {
  std::uint8_t code;
  std::uint8_t bits;
  std::uint16_t lanes;
};

struct DLTensor {
  void *data;
  DLDevice device;
  std::int32_t ndim;
  DLDataType dtype;
  std::int64_t *shape;
  std::int64_t *strides;
  std::uint64_t byteOffset;
};

struct DLManagedTensor {
  DLTensor tensor;
  void *managerContext;
  void (*deleter)(DLManagedTensor *);
};

struct DeviceByteView {
  DLManagedTensor managed{};
  std::int64_t shape = 0;
};

void deleteDeviceByteView(DLManagedTensor *managed) {
  delete reinterpret_cast<DeviceByteView *>(managed);
}

thread_local std::string LastError;

template <typename Function> nta_status protect(Function &&function) noexcept {
  LastError.clear();
  try {
    function();
    return NTA_STATUS_OK;
  } catch (const std::invalid_argument &error) {
    LastError = error.what();
    return NTA_STATUS_INVALID_ARGUMENT;
  } catch (const std::overflow_error &error) {
    LastError = error.what();
    return NTA_STATUS_INVALID_ARGUMENT;
  } catch (const std::logic_error &error) {
    LastError = error.what();
    return NTA_STATUS_RUNTIME_ERROR;
  } catch (const std::exception &error) {
    LastError = error.what();
    return NTA_STATUS_RUNTIME_ERROR;
  } catch (...) {
    LastError = "unknown NTA runtime failure";
    return NTA_STATUS_INTERNAL_ERROR;
  }
}

template <typename Type>
void requireHandle(const Type *handle, const char *name) {
  if (handle == nullptr || handle->value == nullptr) {
    throw std::invalid_argument(std::string(name) + " is null");
  }
}

void requireVersion(std::uint32_t structSize, std::size_t expectedSize,
                    std::uint32_t version, const char *name) {
  if (structSize != expectedSize) {
    throw std::invalid_argument(std::string(name) + " has an invalid size");
  }
  if (version != NTA_RUNTIME_C_API_VERSION) {
    throw std::invalid_argument(std::string(name) +
                                " uses an incompatible API version");
  }
}

std::size_t checkedSize(std::uint64_t bytes, const char *name) {
  if (bytes == 0 || bytes > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument(std::string(name) +
                                " must fit a non-zero size_t");
  }
  return static_cast<std::size_t>(bytes);
}

nta::Placement placement(std::uint32_t value) {
  switch (value) {
  case NTA_PLACEMENT_HBM:
    return nta::Placement::Hbm;
  case NTA_PLACEMENT_HOST_MAPPED:
    return nta::Placement::HostMapped;
  case NTA_PLACEMENT_HOST_STAGED:
    return nta::Placement::HostStaged;
  default:
    throw std::invalid_argument("registered replica has invalid placement");
  }
}

cudaStream_t stream(std::uint64_t address) noexcept {
  return reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(address));
}

void checkCuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

nta::WorkPlan makeWorkPlan(const nta_work_item *workItems,
                           std::uint32_t workItemCount,
                           const nta_acquire_requirement *dependencies,
                           std::uint32_t dependencyCount,
                           const nta_request_work_range *requests,
                           std::uint32_t requestCount) {
  if (workItems == nullptr || workItemCount == 0 || dependencies == nullptr ||
      dependencyCount == 0 || requests == nullptr || requestCount == 0) {
    throw std::invalid_argument("work-plan upload needs non-empty item, "
                                "dependency, and request arrays");
  }

  nta::WorkPlan result;
  result.workItems.reserve(workItemCount);
  result.dependencies.reserve(dependencyCount);
  result.requests.reserve(requestCount);
  for (std::uint32_t index = 0; index < workItemCount; ++index) {
    const nta_work_item &source = workItems[index];
    if (source.reserved0 != 0 || source.reserved1 != 0 ||
        source.reserved2 != 0 || source.reserved3 != 0) {
      throw std::invalid_argument("work-item reserved fields must be zero");
    }
    result.workItems.push_back(
        {source.request_index, source.request_slot, source.generation,
         source.logical_work, source.dependency_begin, source.dependency_count,
         source.direct_dependency_count, source.work_ticket,
         source.reduction_group, source.contributor_index,
         source.contributor_count, source.estimated_compute_ns, 0, 0, 0, 0});
  }
  for (std::uint32_t index = 0; index < dependencyCount; ++index) {
    const nta_acquire_requirement &source = dependencies[index];
    result.dependencies.push_back({source.direct_base, source.direct_tensor_map,
                                   source.object_id, source.offset,
                                   source.object_slot, source.object_version,
                                   source.bytes, source.flags});
  }
  for (std::uint32_t index = 0; index < requestCount; ++index) {
    const nta_request_work_range &source = requests[index];
    result.requests.push_back({source.work_begin, source.work_count,
                               source.request_slot, source.generation});
  }
  return result;
}

} // namespace

extern "C" {

std::uint32_t nta_runtime_c_api_version(void) {
  return NTA_RUNTIME_C_API_VERSION;
}

std::uint32_t nta_runtime_device_abi_version(void) { return nta::abi::Version; }

const char *nta_last_error(void) { return LastError.c_str(); }

nta_status nta_nvme_transport_create(const nta_nvme_transport_options *options,
                                     nta_nvme_transport **transportOut) {
  if (transportOut != nullptr) {
    *transportOut = nullptr;
  }
  return protect([&] {
    if (options == nullptr || transportOut == nullptr) {
      throw std::invalid_argument(
          "NVMe transport creation needs options and an output handle");
    }
    requireVersion(options->struct_size, sizeof(*options), options->api_version,
                   "NVMe transport options");
    if (options->endpoint == nullptr || options->endpoint[0] == '\0') {
      throw std::invalid_argument("NVMe VFIO endpoint must be explicit");
    }
    nta::NvmeTransportOptions native;
    native.endpoint = options->endpoint;
    native.deviceOrdinal = options->device_ordinal;
    native.namespaceId = options->namespace_id;
    native.queueDepth = options->queue_depth;
    native.adminTimeoutMs = options->admin_timeout_ms;
    switch (options->media_policy) {
    case NTA_NVME_REQUIRE_HARDWARE_WRITE_PROTECTION:
      native.mediaPolicy = nta::NvmeMediaPolicy::RequireHardwareWriteProtection;
      break;
    case NTA_NVME_TRUST_READ_ONLY_DEVICE_CODE:
      native.mediaPolicy = nta::NvmeMediaPolicy::TrustReadOnlyDeviceCode;
      break;
    default:
      throw std::invalid_argument("NVMe media policy is invalid");
    }
    auto handle = std::make_unique<nta_nvme_transport>();
    handle->value = std::make_shared<nta::NvmeTransport>(std::move(native));
    *transportOut = handle.release();
  });
}

void nta_nvme_transport_destroy(nta_nvme_transport *transport) {
  delete transport;
}

nta_status
nta_nvme_transport_get_capabilities(const nta_nvme_transport *transport,
                                    nta_nvme_capabilities *capabilities) {
  return protect([&] {
    requireHandle(transport, "NVMe transport");
    if (capabilities == nullptr) {
      throw std::invalid_argument("NVMe capabilities output is null");
    }
    const nta::NvmeCapabilities &source = transport->value->capabilities();
    *capabilities = {
        source.queueDepth,
        source.controllerPageSize,
        source.lbaSize,
        source.maxTransferBytes,
        source.namespaceBytes,
        source.queueId,
        source.queueCount,
        source.deviceOrdinal,
        source.supportsHbmPeer ? 1U : 0U,
        source.translatedIommu ? 1U : 0U,
        source.namespaceReadOnly ? 1U : 0U,
        source.gpuDoorbellMappingValidated ? 1U : 0U,
    };
  });
}

nta_status nta_nvme_transport_read_stats(const nta_nvme_transport *transport,
                                         nta_nvme_queue_stats *stats) {
  return protect([&] {
    requireHandle(transport, "NVMe transport");
    if (stats == nullptr) {
      throw std::invalid_argument("NVMe stats output is null");
    }
    const nta::NvmeQueueStats source = transport->value->readStats();
    *stats = {
        source.submitted,
        source.completed,
        source.failed,
        source.directSubmitted,
        source.directFallbacks,
        source.outstanding,
        source.error,
        source.sqTail,
        source.cqHead,
        source.cqPhase,
        source.nextCompletionDword3,
    };
  });
}

nta_status nta_runtime_create(const nta_runtime_config *config,
                              nta_nvme_transport *nvme,
                              nta_runtime **runtimeOut) {
  if (runtimeOut != nullptr) {
    *runtimeOut = nullptr;
  }
  return protect([&] {
    if (config == nullptr || runtimeOut == nullptr) {
      throw std::invalid_argument(
          "runtime creation needs a config and an output handle");
    }
    requireVersion(config->struct_size, sizeof(*config), config->api_version,
                   "runtime config");
    nta::RuntimeConfig native{
        config->request_capacity,
        config->object_capacity,
        config->intent_capacity,
        config->work_ticket_capacity,
        config->max_replicas_per_object,
        config->max_dependencies_per_work_ticket,
        config->device_ordinal,
        config->enable_cta_nvme_try_issue != 0,
        config->tenant_capacity,
        config->staging_byte_capacity,
    };
    auto handle = std::make_unique<nta_runtime>();
    if (nvme != nullptr) {
      requireHandle(nvme, "NVMe transport");
      handle->nvme = nvme->value;
      handle->value = std::make_unique<nta::HostRuntime>(native, handle->nvme);
    } else {
      handle->value = std::make_unique<nta::HostRuntime>(native);
    }
    *runtimeOut = handle.release();
  });
}

void nta_runtime_destroy(nta_runtime *runtime) { delete runtime; }

nta_status nta_runtime_set_request(
    nta_runtime *runtime, std::uint32_t slot, std::uint64_t requestId,
    std::uint32_t generation, std::uint32_t tenantId, std::uint32_t priority,
    std::uint64_t deadlineClock, std::uint64_t maxOutstandingBytes) {
  return protect([&] {
    requireHandle(runtime, "runtime");
    runtime->value->setRequest(slot, requestId, generation, tenantId, priority,
                               deadlineClock, maxOutstandingBytes);
  });
}

nta_status nta_runtime_publish_requests_async(nta_runtime *runtime,
                                              const nta_request_spec *requests,
                                              std::uint32_t requestCount,
                                              std::uint64_t cudaStream) {
  return protect([&] {
    requireHandle(runtime, "runtime");
    if (requests == nullptr || requestCount == 0) {
      throw std::invalid_argument("request publication batch is empty");
    }
    std::vector<nta::RequestSpec> native;
    native.reserve(requestCount);
    for (std::uint32_t index = 0; index < requestCount; ++index) {
      native.push_back({
          requests[index].slot,
          requests[index].request_id,
          requests[index].generation,
          requests[index].tenant_id,
          requests[index].priority,
          requests[index].deadline_clock,
          requests[index].max_outstanding_bytes,
      });
    }
    runtime->value->publishRequestsAsync(
        native, reinterpret_cast<cudaStream_t>(cudaStream));
  });
}

nta_status nta_runtime_cancel_request(nta_runtime *runtime, std::uint32_t slot,
                                      std::uint32_t generation) {
  return protect([&] {
    requireHandle(runtime, "runtime");
    runtime->value->cancelRequest(slot, generation);
  });
}

nta_status nta_runtime_set_tenant_budget(nta_runtime *runtime,
                                         std::uint32_t tenantId,
                                         std::uint64_t maxOutstandingBytes,
                                         std::uint32_t weight) {
  return protect([&] {
    requireHandle(runtime, "runtime");
    runtime->value->setTenantBudget(tenantId, maxOutstandingBytes, weight);
  });
}

nta_status nta_runtime_register_object(
    nta_runtime *runtime, std::uint32_t slot, std::uint64_t objectId,
    std::uint32_t version, std::uint64_t bytes,
    std::uint64_t stagingDeviceAddress, const nta_registered_replica *replicas,
    std::uint32_t replicaCount, std::uint64_t *directDeviceBaseOut) {
  if (directDeviceBaseOut != nullptr) {
    *directDeviceBaseOut = 0;
  }
  return protect([&] {
    requireHandle(runtime, "runtime");
    if (replicas == nullptr || replicaCount == 0) {
      throw std::invalid_argument(
          "object registration needs at least one replica");
    }
    std::vector<nta::RegisteredReplicaSpec> native;
    native.reserve(replicaCount);
    for (std::uint32_t index = 0; index < replicaCount; ++index) {
      const nta_registered_replica &source = replicas[index];
      if (source.reserved != 0 || source.source_device_address == 0) {
        throw std::invalid_argument(
            "registered replica has invalid reserved or address fields");
      }
      native.push_back(
          {reinterpret_cast<const void *>(
               static_cast<std::uintptr_t>(source.source_device_address)),
           placement(source.placement),
           reinterpret_cast<const void *>(
               static_cast<std::uintptr_t>(source.tensor_map_address)),
           source.estimated_latency_ns,
           source.estimated_bandwidth_bytes_per_second});
    }
    const nta::ObjectHandle object = runtime->value->registerObject(
        slot, objectId, version, checkedSize(bytes, "object bytes"),
        reinterpret_cast<void *>(
            static_cast<std::uintptr_t>(stagingDeviceAddress)),
        native);
    if (directDeviceBaseOut != nullptr) {
      *directDeviceBaseOut =
          reinterpret_cast<std::uintptr_t>(object.directDeviceBase);
    }
  });
}

nta_status nta_runtime_register_indexed_host_object(
    nta_runtime *runtime, std::uint32_t slot, std::uint64_t objectId,
    std::uint32_t version, std::uint64_t sourceDeviceAddress,
    std::uint64_t stagingDeviceAddress,
    std::uint64_t sourceIndicesDeviceAddress,
    std::uint64_t stagingIndicesDeviceAddress, std::uint32_t indexCount,
    std::uint32_t elementBytes, std::uint32_t sourceStrideBytes,
    std::uint32_t stagingStrideBytes, std::uint32_t sourceIndexLimit,
    std::uint32_t stagingIndexLimit) {
  return protect([&] {
    requireHandle(runtime, "runtime");
    runtime->value->registerIndexedHostObject(
        slot, objectId, version,
        reinterpret_cast<const void *>(
            static_cast<std::uintptr_t>(sourceDeviceAddress)),
        reinterpret_cast<void *>(
            static_cast<std::uintptr_t>(stagingDeviceAddress)),
        reinterpret_cast<const std::uint32_t *>(
            static_cast<std::uintptr_t>(sourceIndicesDeviceAddress)),
        reinterpret_cast<const std::uint32_t *>(
            static_cast<std::uintptr_t>(stagingIndicesDeviceAddress)),
        indexCount, elementBytes, sourceStrideBytes, stagingStrideBytes,
        sourceIndexLimit, stagingIndexLimit);
  });
}

nta_status nta_runtime_register_indexed_host_objects(
    nta_runtime *runtime, std::uint32_t firstSlot,
    const nta_indexed_host_object *objects, std::uint32_t objectCount) {
  return protect([&] {
    requireHandle(runtime, "runtime");
    if (objects == nullptr || objectCount == 0) {
      throw std::invalid_argument("indexed host object batch is empty");
    }
    std::vector<nta::IndexedHostObjectSpec> native;
    native.reserve(objectCount);
    for (std::uint32_t index = 0; index < objectCount; ++index) {
      const nta_indexed_host_object &object = objects[index];
      if ((object.flags & ~NTA_INDEXED_HOST_OBJECT_PREACQUIRED) != 0) {
        throw std::invalid_argument(
            "indexed host object flags are unsupported");
      }
      native.push_back({
          object.object_id,
          object.version,
          reinterpret_cast<const void *>(
              static_cast<std::uintptr_t>(object.source_device_address)),
          reinterpret_cast<void *>(
              static_cast<std::uintptr_t>(object.staging_device_address)),
          reinterpret_cast<const std::uint32_t *>(static_cast<std::uintptr_t>(
              object.source_indices_device_address)),
          reinterpret_cast<const std::uint32_t *>(static_cast<std::uintptr_t>(
              object.staging_indices_device_address)),
          object.index_count,
          object.element_bytes,
          object.source_stride_bytes,
          object.staging_stride_bytes,
          object.source_index_limit,
          object.staging_index_limit,
          (object.flags & NTA_INDEXED_HOST_OBJECT_PREACQUIRED) != 0,
      });
    }
    runtime->value->registerIndexedHostObjects(firstSlot, native);
  });
}

nta_status nta_runtime_register_indexed_host_objects_async(
    nta_runtime *runtime, std::uint32_t firstSlot,
    const nta_indexed_host_object *objects, std::uint32_t objectCount,
    std::uint64_t cudaStream) {
  return protect([&] {
    requireHandle(runtime, "runtime");
    if (objects == nullptr || objectCount == 0) {
      throw std::invalid_argument("indexed host object batch is empty");
    }
    std::vector<nta::IndexedHostObjectSpec> native;
    native.reserve(objectCount);
    for (std::uint32_t index = 0; index < objectCount; ++index) {
      const nta_indexed_host_object &object = objects[index];
      if ((object.flags & ~NTA_INDEXED_HOST_OBJECT_PREACQUIRED) != 0) {
        throw std::invalid_argument(
            "indexed host object flags are unsupported");
      }
      native.push_back({
          object.object_id,
          object.version,
          reinterpret_cast<const void *>(
              static_cast<std::uintptr_t>(object.source_device_address)),
          reinterpret_cast<void *>(
              static_cast<std::uintptr_t>(object.staging_device_address)),
          reinterpret_cast<const std::uint32_t *>(static_cast<std::uintptr_t>(
              object.source_indices_device_address)),
          reinterpret_cast<const std::uint32_t *>(static_cast<std::uintptr_t>(
              object.staging_indices_device_address)),
          object.index_count,
          object.element_bytes,
          object.source_stride_bytes,
          object.staging_stride_bytes,
          object.source_index_limit,
          object.staging_index_limit,
          (object.flags & NTA_INDEXED_HOST_OBJECT_PREACQUIRED) != 0,
      });
    }
    runtime->value->registerIndexedHostObjectsAsync(firstSlot, native,
                                                    stream(cudaStream));
  });
}

nta_status nta_runtime_bind_tensor_maps(nta_runtime *runtime,
                                        std::uint32_t objectSlot,
                                        std::uint32_t relativeReplica,
                                        std::uint64_t replicaTensorMap,
                                        std::uint64_t stagingTensorMap) {
  return protect([&] {
    requireHandle(runtime, "runtime");
    if (replicaTensorMap == 0) {
      throw std::invalid_argument("replica tensor-map address is null");
    }
    runtime->value->bindTensorMaps(
        objectSlot, relativeReplica,
        reinterpret_cast<const void *>(
            static_cast<std::uintptr_t>(replicaTensorMap)),
        reinterpret_cast<const void *>(
            static_cast<std::uintptr_t>(stagingTensorMap)));
  });
}

nta_status nta_runtime_install_nvme_object(
    nta_runtime *runtime, std::uint32_t slot, std::uint64_t objectId,
    std::uint32_t version, std::uint64_t sourceByteOffset, std::uint64_t bytes,
    std::uint64_t *destinationDeviceAddressOut) {
  if (destinationDeviceAddressOut != nullptr) {
    *destinationDeviceAddressOut = 0;
  }
  return protect([&] {
    requireHandle(runtime, "runtime");
    if (runtime->nvme == nullptr) {
      throw std::invalid_argument(
          "runtime was not created with an NVMe transport");
    }
    const std::size_t nativeBytes = checkedSize(bytes, "NVMe object bytes");
    auto destination = runtime->nvme->allocate(nativeBytes);
    const std::uint64_t address =
        reinterpret_cast<std::uintptr_t>(destination->deviceAddress());
    runtime->value->installNvmeObject(slot, objectId, version, sourceByteOffset,
                                      nativeBytes, std::move(destination));
    if (destinationDeviceAddressOut != nullptr) {
      *destinationDeviceAddressOut = address;
    }
  });
}

nta_status nta_runtime_read_pending_count(const nta_runtime *runtime,
                                          std::uint32_t *pendingCount) {
  return protect([&] {
    requireHandle(runtime, "runtime");
    if (pendingCount == nullptr) {
      throw std::invalid_argument("pending-count output is null");
    }
    *pendingCount = runtime->value->readPendingCount();
  });
}

nta_status nta_runtime_read_epoch_status(const nta_runtime *runtime,
                                         std::uint32_t workTicketCount,
                                         nta_epoch_status *status) {
  return protect([&] {
    requireHandle(runtime, "runtime");
    if (status == nullptr) {
      throw std::invalid_argument("epoch-status output is null");
    }
    const nta::EpochStatus source =
        runtime->value->readEpochStatus(workTicketCount);
    *status = {source.total,  source.fresh,       source.pending,
               source.ready,  source.done,        source.cancelled,
               source.failed, source.initializing};
  });
}

nta_status nta_runtime_read_sticky_failed_count(const nta_runtime *runtime,
                                                std::uint32_t *failedCount) {
  return protect([&] {
    requireHandle(runtime, "runtime");
    if (failedCount == nullptr) {
      throw std::invalid_argument("sticky failure-count output is null");
    }
    *failedCount = runtime->value->readStickyFailedCount();
  });
}

nta_status nta_runtime_read_request_progress(const nta_runtime *runtime,
                                             std::uint32_t requestSlot,
                                             nta_request_progress *progress) {
  return protect([&] {
    requireHandle(runtime, "runtime");
    if (progress == nullptr) {
      throw std::invalid_argument("request-progress output is null");
    }
    const nta::abi::RequestProgress source =
        runtime->value->readRequestProgress(requestSlot);
    *progress = {
        source.requestId,
        source.generation,
        source.expectedWork,
        source.pendingWork,
        source.runnableWork,
        source.completedWork,
        source.failedWork,
        source.cancelledWork,
        source.epoch,
        source.unavailableBytes,
        source.runnableComputeNs,
        source.completedComputeNs,
        source.pendingComputeNs,
        source.expectedComputeNs,
        source.droppedAttributions,
        0,
    };
  });
}

nta_status nta_runtime_read_request_progress_range(
    const nta_runtime *runtime, std::uint32_t firstRequestSlot,
    std::uint32_t requestCount, nta_request_progress *progress) {
  return protect([&] {
    requireHandle(runtime, "runtime");
    if (progress == nullptr || requestCount == 0) {
      throw std::invalid_argument("request-progress range output is empty");
    }
    const std::vector<nta::abi::RequestProgress> source =
        runtime->value->readRequestProgress(firstRequestSlot, requestCount);
    for (std::uint32_t index = 0; index < requestCount; ++index) {
      progress[index] = {
          source[index].requestId,
          source[index].generation,
          source[index].expectedWork,
          source[index].pendingWork,
          source[index].runnableWork,
          source[index].completedWork,
          source[index].failedWork,
          source[index].cancelledWork,
          source[index].epoch,
          source[index].unavailableBytes,
          source[index].runnableComputeNs,
          source[index].completedComputeNs,
          source[index].pendingComputeNs,
          source[index].expectedComputeNs,
          source[index].droppedAttributions,
          0,
      };
    }
  });
}

nta_status nta_runtime_copy_request_progress_async(
    const nta_runtime *runtime, std::uint32_t firstRequestSlot,
    std::uint32_t requestCount, std::uint64_t hostDestination,
    std::uint64_t cudaStream) {
  return protect([&] {
    requireHandle(runtime, "runtime");
    if (requestCount == 0 || hostDestination == 0 ||
        hostDestination % alignof(nta::abi::RequestProgress) != 0) {
      throw std::invalid_argument(
          "request-progress snapshot destination is empty or misaligned");
    }
    auto *destination = reinterpret_cast<nta::abi::RequestProgress *>(
        static_cast<std::uintptr_t>(hostDestination));
    runtime->value->copyRequestProgressAsync(
        firstRequestSlot,
        std::span<nta::abi::RequestProgress>(destination, requestCount),
        reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(cudaStream)));
  });
}

nta_status nta_runtime_read_work_ticket_state(const nta_runtime *runtime,
                                              std::uint32_t workTicket,
                                              std::uint32_t *state) {
  return protect([&] {
    requireHandle(runtime, "runtime");
    if (state == nullptr) {
      throw std::invalid_argument("work-ticket state output is null");
    }
    *state = runtime->value->readWorkTicket(workTicket).state;
  });
}

nta_status nta_runtime_read_work_runnable_ns(const nta_runtime *runtime,
                                             std::uint32_t workTicketCount,
                                             std::uint64_t *runnableNs) {
  return protect([&] {
    requireHandle(runtime, "runtime");
    if (runnableNs == nullptr || workTicketCount == 0) {
      throw std::invalid_argument("work-arrival output is empty");
    }
    const std::vector<std::uint64_t> source =
        runtime->value->readWorkRunnableNs(workTicketCount);
    std::copy(source.begin(), source.end(), runnableNs);
  });
}

std::uint64_t nta_runtime_device_view(const nta_runtime *runtime) {
  return runtime == nullptr || runtime->value == nullptr
             ? 0
             : reinterpret_cast<std::uintptr_t>(runtime->value->deviceView());
}

std::int32_t nta_runtime_device_ordinal(const nta_runtime *runtime) {
  return runtime == nullptr || runtime->value == nullptr
             ? NTA_RUNTIME_USE_CURRENT_DEVICE
             : runtime->value->deviceOrdinal();
}

nta_status nta_device_pointer_dlpack(std::uint64_t deviceAddress,
                                     std::uint64_t bytes,
                                     std::int32_t deviceOrdinal,
                                     void **managedTensorOut) {
  if (managedTensorOut != nullptr) {
    *managedTensorOut = nullptr;
  }
  return protect([&] {
    if (managedTensorOut == nullptr || deviceAddress == 0 || bytes == 0 ||
        bytes > static_cast<std::uint64_t>(
                    std::numeric_limits<std::int64_t>::max())) {
      throw std::invalid_argument(
          "DLPack byte view needs an address, size, and output");
    }
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "query CUDA devices");
    int owner = deviceOrdinal;
    if (owner == NTA_RUNTIME_USE_CURRENT_DEVICE) {
      checkCuda(cudaGetDevice(&owner), "query current CUDA device");
    }
    if (owner < 0 || owner >= deviceCount) {
      throw std::out_of_range("DLPack byte view has an invalid CUDA device");
    }

    cudaPointerAttributes attributes{};
    checkCuda(cudaPointerGetAttributes(
                  &attributes, reinterpret_cast<const void *>(
                                   static_cast<std::uintptr_t>(deviceAddress))),
              "inspect DLPack CUDA pointer");
    if ((attributes.type != cudaMemoryTypeDevice &&
         attributes.type != cudaMemoryTypeManaged) ||
        attributes.device != owner) {
      throw std::invalid_argument(
          "DLPack byte view pointer is not owned by the requested CUDA device");
    }

    auto view = std::make_unique<DeviceByteView>();
    view->shape = static_cast<std::int64_t>(bytes);
    view->managed.tensor.data =
        reinterpret_cast<void *>(static_cast<std::uintptr_t>(deviceAddress));
    view->managed.tensor.device = {2, owner}; // kDLCUDA
    view->managed.tensor.ndim = 1;
    view->managed.tensor.dtype = {1, 8, 1}; // kDLUInt, uint8
    view->managed.tensor.shape = &view->shape;
    view->managed.tensor.strides = nullptr;
    view->managed.tensor.byteOffset = 0;
    view->managed.managerContext = nullptr;
    view->managed.deleter = deleteDeviceByteView;
    *managedTensorOut = &view.release()->managed;
  });
}

void nta_dlpack_managed_tensor_destroy(void *managedTensor) {
  if (managedTensor != nullptr) {
    auto *managed = static_cast<DLManagedTensor *>(managedTensor);
    if (managed->deleter != nullptr) {
      managed->deleter(managed);
    }
  }
}

nta_status nta_stream_synchronize(std::uint64_t cudaStream) {
  return protect([&] {
    checkCuda(cudaStreamSynchronize(stream(cudaStream)),
              "synchronize CUDA stream");
  });
}

nta_status nta_copy_host_to_device_async(std::uint64_t destination,
                                         std::uint64_t source,
                                         std::uint64_t bytes,
                                         std::uint64_t cudaStream) {
  return protect([&] {
    if (destination == 0 || source == 0) {
      throw std::invalid_argument("host-to-device copy addresses are null");
    }
    const std::size_t copyBytes =
        checkedSize(bytes, "host-to-device copy bytes");
    checkCuda(cudaMemcpyAsync(reinterpret_cast<void *>(destination),
                              reinterpret_cast<const void *>(source), copyBytes,
                              cudaMemcpyHostToDevice, stream(cudaStream)),
              "enqueue host-to-device copy");
  });
}

nta_status nta_device_work_plan_create(std::uint32_t workItemCapacity,
                                       std::uint32_t dependencyCapacity,
                                       std::int32_t deviceOrdinal,
                                       nta_device_work_plan **planOut) {
  if (planOut != nullptr) {
    *planOut = nullptr;
  }
  return protect([&] {
    if (planOut == nullptr) {
      throw std::invalid_argument("work-plan output handle is null");
    }
    auto handle = std::make_unique<nta_device_work_plan>();
    handle->value = std::make_unique<nta::DeviceWorkPlan>(
        workItemCapacity, dependencyCapacity, deviceOrdinal);
    *planOut = handle.release();
  });
}

void nta_device_work_plan_destroy(nta_device_work_plan *plan) { delete plan; }

nta_status nta_device_work_plan_upload(
    nta_device_work_plan *plan, const nta_work_item *workItems,
    std::uint32_t workItemCount, const nta_acquire_requirement *dependencies,
    std::uint32_t dependencyCount, const nta_request_work_range *requests,
    std::uint32_t requestCount, std::uint64_t cudaStream) {
  return protect([&] {
    requireHandle(plan, "device work plan");
    plan->value->uploadAsync(makeWorkPlan(workItems, workItemCount,
                                          dependencies, dependencyCount,
                                          requests, requestCount),
                             stream(cudaStream));
  });
}

nta_status nta_device_work_plan_wait_on(const nta_device_work_plan *plan,
                                        std::uint64_t cudaStream) {
  return protect([&] {
    requireHandle(plan, "device work plan");
    plan->value->waitOn(stream(cudaStream));
  });
}

nta_status
nta_device_work_plan_synchronize_upload(const nta_device_work_plan *plan) {
  return protect([&] {
    requireHandle(plan, "device work plan");
    plan->value->synchronizeUpload();
  });
}

std::uint64_t
nta_device_work_plan_work_items(const nta_device_work_plan *plan) {
  return plan == nullptr || plan->value == nullptr
             ? 0
             : reinterpret_cast<std::uintptr_t>(plan->value->workItems());
}

std::uint64_t
nta_device_work_plan_dependencies(const nta_device_work_plan *plan) {
  return plan == nullptr || plan->value == nullptr
             ? 0
             : reinterpret_cast<std::uintptr_t>(plan->value->dependencies());
}

std::uint32_t
nta_device_work_plan_work_item_count(const nta_device_work_plan *plan) {
  return plan == nullptr || plan->value == nullptr
             ? 0
             : plan->value->workItemCount();
}

std::uint32_t
nta_device_work_plan_dependency_count(const nta_device_work_plan *plan) {
  return plan == nullptr || plan->value == nullptr
             ? 0
             : plan->value->dependencyCount();
}

std::int32_t
nta_device_work_plan_device_ordinal(const nta_device_work_plan *plan) {
  return plan == nullptr || plan->value == nullptr
             ? NTA_RUNTIME_USE_CURRENT_DEVICE
             : plan->value->deviceOrdinal();
}

nta_status nta_jit_phase_program_create(const char *sharedObject,
                                        nta_jit_phase_program **programOut) {
  if (programOut != nullptr) {
    *programOut = nullptr;
  }
  return protect([&] {
    if (sharedObject == nullptr || sharedObject[0] == '\0' ||
        programOut == nullptr) {
      throw std::invalid_argument(
          "JIT phase creation needs a shared object and output handle");
    }
    auto handle = std::make_unique<nta_jit_phase_program>();
    handle->value = std::make_unique<nta::JitPhaseProgram>(sharedObject);
    *programOut = handle.release();
  });
}

void nta_jit_phase_program_destroy(nta_jit_phase_program *program) {
  delete program;
}

nta_status nta_jit_phase_operator_contract(const nta_jit_phase_program *program,
                                           nta_operator_contract *contractOut) {
  return protect([&] {
    requireHandle(program, "JIT phase program");
    if (contractOut == nullptr) {
      throw std::invalid_argument("JIT operator contract output is null");
    }
    const nta::operator_contract::Contract &contract =
        program->value->operatorContract();
    *contractOut = {
        contract.magic,
        contract.schemaVersion,
        contract.structBytes,
        contract.runtimeAbiVersion,
        contract.family,
        contract.form,
        contract.reserved,
        contract.capabilities,
        contract.sourceFingerprintLow,
        contract.sourceFingerprintHigh,
    };
  });
}

nta_status nta_jit_phase_operator_plan(const nta_jit_phase_program *program,
                                       nta_operator_plan *planOut) {
  return protect([&] {
    requireHandle(program, "JIT phase program");
    if (planOut == nullptr) {
      throw std::invalid_argument("JIT operator plan output is null");
    }
    const nta::operator_contract::Plan &plan = program->value->operatorPlan();
    *planOut = {
        plan.magic,
        plan.schemaVersion,
        plan.structBytes,
        plan.runtimeAbiVersion,
        plan.family,
        plan.supportedForms,
        plan.coordinateMap,
        plan.partialState,
        plan.reduction,
        plan.flags,
        plan.reserved,
        plan.sourceFingerprintLow,
        plan.sourceFingerprintHigh,
        plan.planFingerprintLow,
        plan.planFingerprintHigh,
    };
  });
}

nta_status nta_jit_phase_reset(const nta_jit_phase_program *program,
                               nta_runtime *runtime, std::uint32_t objectCount,
                               std::uint32_t workTicketCount,
                               std::uint64_t cudaStream) {
  return protect([&] {
    requireHandle(program, "JIT phase program");
    requireHandle(runtime, "runtime");
    program->value->reset(stream(cudaStream), runtime->value->deviceView(),
                          objectCount, workTicketCount);
  });
}

nta_status nta_jit_phase_discover(const nta_jit_phase_program *program,
                                  nta_runtime *runtime, std::uint64_t workItems,
                                  std::uint64_t dependencies,
                                  std::uint32_t workItemCount,
                                  std::uint64_t cudaStream) {
  return protect([&] {
    requireHandle(program, "JIT phase program");
    requireHandle(runtime, "runtime");
    program->value->discover(
        stream(cudaStream), runtime->value->deviceView(),
        reinterpret_cast<const nta::abi::WorkItem *>(workItems),
        reinterpret_cast<const nta::abi::AcquireRequirement *>(dependencies),
        workItemCount);
  });
}

nta_status nta_jit_phase_invalidate_cached_objects(
    const nta_jit_phase_program *program, nta_runtime *runtime,
    std::uint32_t firstObject, std::uint32_t objectCount,
    std::uint64_t cudaStream) {
  return protect([&] {
    requireHandle(program, "JIT phase program");
    requireHandle(runtime, "runtime");
    program->value->invalidateCachedObjects(stream(cudaStream),
                                            runtime->value->deviceView(),
                                            firstObject, objectCount);
  });
}

nta_status nta_jit_phase_validate_indexed_host_range(
    const nta_jit_phase_program *program, nta_runtime *runtime,
    std::uint32_t firstObject, std::uint32_t objectCount,
    std::uint64_t cudaStream) {
  return protect([&] {
    requireHandle(program, "JIT phase program");
    requireHandle(runtime, "runtime");
    program->value->validateIndexedHostRange(stream(cudaStream),
                                             runtime->value->deviceView(),
                                             firstObject, objectCount);
  });
}

nta_status nta_jit_phase_rebind_indexed_host_pairs(
    const nta_jit_phase_program *program, nta_runtime *runtime,
    std::uint32_t firstObject, std::uint32_t pairCount, std::uint64_t keySource,
    std::uint64_t keyStaging, std::uint64_t valueSource,
    std::uint64_t valueStaging, std::uint64_t cudaStream) {
  return protect([&] {
    requireHandle(program, "JIT phase program");
    requireHandle(runtime, "runtime");
    program->value->rebindIndexedHostPairs(
        stream(cudaStream), runtime->value->deviceView(), firstObject,
        pairCount, keySource, keyStaging, valueSource, valueStaging);
  });
}

nta_status nta_jit_phase_preload_host(const nta_jit_phase_program *program,
                                      nta_runtime *runtime,
                                      std::uint32_t firstObject,
                                      std::uint32_t objectCount,
                                      std::uint64_t cudaStream) {
  return protect([&] {
    requireHandle(program, "JIT phase program");
    requireHandle(runtime, "runtime");
    program->value->preloadHost(stream(cudaStream),
                                runtime->value->deviceView(), firstObject,
                                objectCount);
  });
}

nta_status nta_jit_phase_preload_host_pairs(
    const nta_jit_phase_program *program, nta_runtime *runtime,
    std::uint32_t firstObject, std::uint32_t pairCount,
    std::uint64_t cudaStream) {
  return protect([&] {
    requireHandle(program, "JIT phase program");
    requireHandle(runtime, "runtime");
    program->value->preloadHostPairs(stream(cudaStream),
                                     runtime->value->deviceView(), firstObject,
                                     pairCount);
  });
}

nta_status nta_jit_phase_alias_preloaded_objects(
    const nta_jit_phase_program *program, nta_runtime *runtime,
    std::uint32_t sourceFirst, std::uint32_t destinationFirst,
    std::uint32_t objectCount, std::uint64_t objectIdBase,
    std::uint32_t version, std::uint64_t cudaStream) {
  return protect([&] {
    requireHandle(program, "JIT phase program");
    requireHandle(runtime, "runtime");
    program->value->aliasPreloadedObjects(
        stream(cudaStream), runtime->value->deviceView(), sourceFirst,
        destinationFirst, objectCount, objectIdBase, version);
  });
}

nta_status nta_jit_phase_progress_host(const nta_jit_phase_program *program,
                                       nta_runtime *runtime,
                                       std::uint32_t blocks,
                                       std::uint64_t cudaStream) {
  return protect([&] {
    requireHandle(program, "JIT phase program");
    requireHandle(runtime, "runtime");
    program->value->progressHost(stream(cudaStream),
                                 runtime->value->deviceView(), blocks);
  });
}

nta_status nta_jit_phase_progress_indexed_host_range(
    const nta_jit_phase_program *program, nta_runtime *runtime,
    std::uint32_t firstObject, std::uint32_t objectCount,
    std::uint64_t cudaStream) {
  return protect([&] {
    requireHandle(program, "JIT phase program");
    requireHandle(runtime, "runtime");
    program->value->progressIndexedHostRange(stream(cudaStream),
                                             runtime->value->deviceView(),
                                             firstObject, objectCount);
  });
}

nta_status nta_jit_phase_progress_validated_indexed_host_range(
    const nta_jit_phase_program *program, nta_runtime *runtime,
    std::uint32_t firstObject, std::uint32_t objectCount,
    std::uint64_t cudaStream) {
  return protect([&] {
    requireHandle(program, "JIT phase program");
    requireHandle(runtime, "runtime");
    program->value->progressValidatedIndexedHostRange(
        stream(cudaStream), runtime->value->deviceView(), firstObject,
        objectCount);
  });
}

nta_status nta_jit_phase_progress_validated_indexed_host_range_parallel(
    const nta_jit_phase_program *program, nta_runtime *runtime,
    std::uint32_t firstObject, std::uint32_t objectCount,
    std::uint32_t copyBlocksPerGroup, std::uint64_t cudaStream) {
  return protect([&] {
    requireHandle(program, "JIT phase program");
    requireHandle(runtime, "runtime");
    program->value->progressValidatedIndexedHostRangeParallel(
        stream(cudaStream), runtime->value->deviceView(), firstObject,
        objectCount, copyBlocksPerGroup);
  });
}

nta_status nta_jit_phase_set_indexed_row_counts(
    const nta_jit_phase_program *program, nta_runtime *runtime,
    std::uint32_t firstObject, std::uint32_t objectCount,
    std::uint32_t rowCount, std::uint64_t cudaStream) {
  return protect([&] {
    requireHandle(program, "JIT phase program");
    requireHandle(runtime, "runtime");
    program->value->setIndexedRowCounts(stream(cudaStream),
                                        runtime->value->deviceView(),
                                        firstObject, objectCount, rowCount);
  });
}

nta_status nta_jit_phase_progress_nvme(const nta_jit_phase_program *program,
                                       nta_runtime *runtime,
                                       std::uint32_t issueBudget,
                                       std::uint32_t completionBudget,
                                       std::uint64_t cudaStream) {
  return protect([&] {
    requireHandle(program, "JIT phase program");
    requireHandle(runtime, "runtime");
    program->value->progressNvme(stream(cudaStream),
                                 runtime->value->deviceView(), issueBudget,
                                 completionBudget);
  });
}

nta_status nta_jit_phase_publish(const nta_jit_phase_program *program,
                                 nta_runtime *runtime,
                                 std::uint32_t pendingBudget,
                                 std::uint64_t cudaStream) {
  return protect([&] {
    requireHandle(program, "JIT phase program");
    requireHandle(runtime, "runtime");
    program->value->publish(stream(cudaStream), runtime->value->deviceView(),
                            pendingBudget);
  });
}

nta_status nta_jit_phase_complete(const nta_jit_phase_program *program,
                                  nta_runtime *runtime,
                                  std::uint32_t workTicketCount,
                                  std::uint64_t cudaStream) {
  return protect([&] {
    requireHandle(program, "JIT phase program");
    requireHandle(runtime, "runtime");
    program->value->complete(stream(cudaStream), runtime->value->deviceView(),
                             workTicketCount);
  });
}

nta_status nta_jit_phase_complete_stream_ordered(
    const nta_jit_phase_program *program, nta_runtime *runtime,
    std::uint64_t workItems, std::uint32_t workItemCount,
    std::uint64_t cudaStream) {
  return protect([&] {
    requireHandle(program, "JIT phase program");
    requireHandle(runtime, "runtime");
    program->value->completeStreamOrdered(
        stream(cudaStream), runtime->value->deviceView(),
        reinterpret_cast<const nta::abi::WorkItem *>(workItems), workItemCount);
  });
}

} // extern "C"
