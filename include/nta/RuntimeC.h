#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NTA_RUNTIME_C_API_VERSION 10U
#define NTA_RUNTIME_USE_CURRENT_DEVICE (-1)

typedef struct nta_runtime nta_runtime;
typedef struct nta_device_work_plan nta_device_work_plan;
typedef struct nta_jit_phase_program nta_jit_phase_program;
typedef struct nta_nvme_transport nta_nvme_transport;

typedef enum nta_status {
  NTA_STATUS_OK = 0,
  NTA_STATUS_INVALID_ARGUMENT = 1,
  NTA_STATUS_RUNTIME_ERROR = 2,
  NTA_STATUS_INTERNAL_ERROR = 3,
} nta_status;

typedef enum nta_placement {
  NTA_PLACEMENT_HBM = 0,
  NTA_PLACEMENT_HOST_MAPPED = 1,
  NTA_PLACEMENT_HOST_STAGED = 2,
} nta_placement;

typedef enum nta_indexed_host_object_flags {
  NTA_INDEXED_HOST_OBJECT_PREACQUIRED = 1U << 0,
} nta_indexed_host_object_flags;

typedef enum nta_nvme_media_policy {
  NTA_NVME_REQUIRE_HARDWARE_WRITE_PROTECTION = 0,
  NTA_NVME_TRUST_READ_ONLY_DEVICE_CODE = 1,
} nta_nvme_media_policy;

typedef struct nta_runtime_config {
  uint32_t struct_size;
  uint32_t api_version;
  uint32_t request_capacity;
  uint32_t object_capacity;
  uint32_t intent_capacity;
  uint32_t work_ticket_capacity;
  uint32_t max_replicas_per_object;
  uint32_t max_dependencies_per_work_ticket;
  int32_t device_ordinal;
  uint32_t enable_cta_nvme_try_issue;
  uint32_t tenant_capacity;
  uint64_t staging_byte_capacity;
} nta_runtime_config;

typedef struct nta_registered_replica {
  uint64_t source_device_address;
  uint64_t tensor_map_address;
  uint64_t estimated_latency_ns;
  uint64_t estimated_bandwidth_bytes_per_second;
  uint32_t placement;
  uint32_t reserved;
} nta_registered_replica;

typedef struct nta_acquire_requirement {
  uint64_t direct_base;
  uint64_t direct_tensor_map;
  uint64_t object_id;
  uint64_t offset;
  uint32_t object_slot;
  uint32_t object_version;
  uint32_t bytes;
  uint32_t flags;
} nta_acquire_requirement;

typedef struct nta_indexed_host_object {
  uint64_t object_id;
  uint64_t source_device_address;
  uint64_t staging_device_address;
  uint64_t source_indices_device_address;
  uint64_t staging_indices_device_address;
  uint32_t version;
  uint32_t index_count;
  uint32_t element_bytes;
  uint32_t source_stride_bytes;
  uint32_t staging_stride_bytes;
  uint32_t flags;
} nta_indexed_host_object;

typedef struct nta_work_item {
  uint32_t request_index;
  uint32_t request_slot;
  uint32_t generation;
  uint32_t logical_work;
  uint32_t dependency_begin;
  uint32_t dependency_count;
  uint32_t direct_dependency_count;
  uint32_t work_ticket;
  uint32_t reduction_group;
  uint32_t contributor_index;
  uint32_t contributor_count;
  uint32_t estimated_compute_ns;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
  uint32_t reserved3;
} nta_work_item;

typedef struct nta_request_work_range {
  uint32_t work_begin;
  uint32_t work_count;
  uint32_t request_slot;
  uint32_t generation;
} nta_request_work_range;

typedef struct nta_nvme_transport_options {
  uint32_t struct_size;
  uint32_t api_version;
  const char *endpoint;
  int32_t device_ordinal;
  uint32_t namespace_id;
  uint32_t queue_depth;
  uint32_t admin_timeout_ms;
  uint32_t media_policy;
} nta_nvme_transport_options;

typedef struct nta_nvme_capabilities {
  uint32_t queue_depth;
  uint32_t controller_page_size;
  uint32_t lba_size;
  uint32_t max_transfer_bytes;
  uint64_t namespace_bytes;
  uint32_t queue_id;
  uint32_t queue_count;
  int32_t device_ordinal;
  uint32_t supports_hbm_peer;
  uint32_t translated_iommu;
  uint32_t namespace_read_only;
  uint32_t gpu_doorbell_mapping_validated;
} nta_nvme_capabilities;

typedef struct nta_nvme_queue_stats {
  uint64_t submitted;
  uint64_t completed;
  uint64_t failed;
  uint64_t direct_submitted;
  uint64_t direct_fallbacks;
  uint32_t outstanding;
  uint32_t error;
  uint32_t sq_tail;
  uint32_t cq_head;
  uint32_t cq_phase;
  uint32_t next_completion_dword3;
} nta_nvme_queue_stats;

typedef struct nta_epoch_status {
  uint32_t total;
  uint32_t fresh;
  uint32_t pending;
  uint32_t ready;
  uint32_t done;
  uint32_t cancelled;
  uint32_t failed;
  uint32_t initializing;
} nta_epoch_status;

typedef struct nta_request_progress {
  uint64_t request_id;
  uint32_t generation;
  uint32_t expected_work;
  uint32_t pending_work;
  uint32_t runnable_work;
  uint32_t completed_work;
  uint32_t failed_work;
  uint32_t cancelled_work;
  uint32_t epoch;
  uint64_t unavailable_bytes;
  uint64_t runnable_compute_ns;
  uint64_t completed_compute_ns;
} nta_request_progress;

uint32_t nta_runtime_c_api_version(void);
uint32_t nta_runtime_device_abi_version(void);
const char *nta_last_error(void);

nta_status nta_nvme_transport_create(
    const nta_nvme_transport_options *options,
    nta_nvme_transport **transport_out);
void nta_nvme_transport_destroy(nta_nvme_transport *transport);
nta_status nta_nvme_transport_get_capabilities(
    const nta_nvme_transport *transport, nta_nvme_capabilities *capabilities);
nta_status nta_nvme_transport_read_stats(const nta_nvme_transport *transport,
                                         nta_nvme_queue_stats *stats);

nta_status nta_runtime_create(const nta_runtime_config *config,
                              nta_nvme_transport *nvme,
                              nta_runtime **runtime_out);
void nta_runtime_destroy(nta_runtime *runtime);
nta_status nta_runtime_set_request(
    nta_runtime *runtime, uint32_t slot, uint64_t request_id,
    uint32_t generation, uint32_t tenant_id, uint32_t priority,
    uint64_t deadline_clock, uint64_t max_outstanding_bytes);
nta_status nta_runtime_cancel_request(nta_runtime *runtime, uint32_t slot,
                                      uint32_t generation);
nta_status nta_runtime_set_tenant_budget(nta_runtime *runtime,
                                         uint32_t tenant_id,
                                         uint64_t max_outstanding_bytes,
                                         uint32_t weight);
nta_status nta_runtime_register_object(
    nta_runtime *runtime, uint32_t slot, uint64_t object_id,
    uint32_t version, uint64_t bytes, uint64_t staging_device_address,
    const nta_registered_replica *replicas, uint32_t replica_count,
    uint64_t *direct_device_base_out);
nta_status nta_runtime_register_indexed_host_object(
    nta_runtime *runtime, uint32_t slot, uint64_t object_id,
    uint32_t version, uint64_t source_device_address,
    uint64_t staging_device_address, uint64_t source_indices_device_address,
    uint64_t staging_indices_device_address, uint32_t index_count,
    uint32_t element_bytes, uint32_t source_stride_bytes,
    uint32_t staging_stride_bytes);
nta_status nta_runtime_register_indexed_host_objects(
    nta_runtime *runtime, uint32_t first_slot,
    const nta_indexed_host_object *objects, uint32_t object_count);
nta_status nta_runtime_register_indexed_host_objects_async(
    nta_runtime *runtime, uint32_t first_slot,
    const nta_indexed_host_object *objects, uint32_t object_count,
    uint64_t cuda_stream);
nta_status nta_runtime_bind_tensor_maps(nta_runtime *runtime,
                                        uint32_t object_slot,
                                        uint32_t relative_replica,
                                        uint64_t replica_tensor_map,
                                        uint64_t staging_tensor_map);
nta_status nta_runtime_install_nvme_object(
    nta_runtime *runtime, uint32_t slot, uint64_t object_id,
    uint32_t version, uint64_t source_byte_offset, uint64_t bytes,
    uint64_t *destination_device_address_out);
nta_status nta_runtime_read_pending_count(const nta_runtime *runtime,
                                          uint32_t *pending_count);
nta_status nta_runtime_read_epoch_status(const nta_runtime *runtime,
                                         uint32_t work_ticket_count,
                                         nta_epoch_status *status);
nta_status nta_runtime_read_request_progress(const nta_runtime *runtime,
                                             uint32_t request_slot,
                                             nta_request_progress *progress);
nta_status nta_runtime_read_request_progress_range(
    const nta_runtime *runtime, uint32_t first_request_slot,
    uint32_t request_count, nta_request_progress *progress);
nta_status nta_runtime_read_work_ticket_state(const nta_runtime *runtime,
                                               uint32_t work_ticket,
                                               uint32_t *state);
nta_status nta_runtime_read_work_runnable_ns(const nta_runtime *runtime,
                                             uint32_t work_ticket_count,
                                             uint64_t *runnable_ns);
uint64_t nta_runtime_device_view(const nta_runtime *runtime);
int32_t nta_runtime_device_ordinal(const nta_runtime *runtime);

// Return a DLPack DLManagedTensor describing a non-owning byte view over an
// existing CUDA allocation. A successful DLPack consumer owns the descriptor,
// never the CUDA allocation. Call nta_dlpack_managed_tensor_destroy only when
// the descriptor was not consumed.
nta_status nta_device_pointer_dlpack(uint64_t device_address, uint64_t bytes,
                                    int32_t device_ordinal,
                                    void **managed_tensor_out);
void nta_dlpack_managed_tensor_destroy(void *managed_tensor);
nta_status nta_stream_synchronize(uint64_t cuda_stream);

nta_status nta_device_work_plan_create(uint32_t work_item_capacity,
                                       uint32_t dependency_capacity,
                                       int32_t device_ordinal,
                                       nta_device_work_plan **plan_out);
void nta_device_work_plan_destroy(nta_device_work_plan *plan);
nta_status nta_device_work_plan_upload(
    nta_device_work_plan *plan, const nta_work_item *work_items,
    uint32_t work_item_count,
    const nta_acquire_requirement *dependencies, uint32_t dependency_count,
    const nta_request_work_range *requests, uint32_t request_count,
    uint64_t cuda_stream);
nta_status nta_device_work_plan_wait_on(const nta_device_work_plan *plan,
                                        uint64_t cuda_stream);
nta_status nta_device_work_plan_synchronize_upload(
    const nta_device_work_plan *plan);
uint64_t nta_device_work_plan_work_items(const nta_device_work_plan *plan);
uint64_t nta_device_work_plan_dependencies(const nta_device_work_plan *plan);
uint32_t nta_device_work_plan_work_item_count(
    const nta_device_work_plan *plan);
uint32_t nta_device_work_plan_dependency_count(
    const nta_device_work_plan *plan);
int32_t nta_device_work_plan_device_ordinal(
    const nta_device_work_plan *plan);

nta_status nta_jit_phase_program_create(const char *shared_object,
                                        nta_jit_phase_program **program_out);
void nta_jit_phase_program_destroy(nta_jit_phase_program *program);
nta_status nta_jit_phase_reset(const nta_jit_phase_program *program,
                               nta_runtime *runtime, uint32_t object_count,
                               uint32_t work_ticket_count,
                               uint64_t cuda_stream);
nta_status nta_jit_phase_preload_host(const nta_jit_phase_program *program,
                                      nta_runtime *runtime,
                                      uint32_t first_object,
                                      uint32_t object_count,
                                      uint64_t cuda_stream);
nta_status nta_jit_phase_progress_host(const nta_jit_phase_program *program,
                                       nta_runtime *runtime, uint32_t blocks,
                                       uint64_t cuda_stream);
nta_status nta_jit_phase_progress_nvme(const nta_jit_phase_program *program,
                                       nta_runtime *runtime,
                                       uint32_t issue_budget,
                                       uint32_t completion_budget,
                                       uint64_t cuda_stream);
nta_status nta_jit_phase_publish(const nta_jit_phase_program *program,
                                 nta_runtime *runtime,
                                 uint32_t pending_budget,
                                 uint64_t cuda_stream);
nta_status nta_jit_phase_complete(const nta_jit_phase_program *program,
                                  nta_runtime *runtime,
                                  uint32_t work_ticket_count,
                                  uint64_t cuda_stream);

#ifdef __cplusplus
}
#endif
