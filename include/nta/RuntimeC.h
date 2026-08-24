#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NTA_RUNTIME_C_API_VERSION 32U
#define NTA_RUNTIME_USE_CURRENT_DEVICE (-1)

typedef struct nta_runtime nta_runtime;
typedef struct nta_device_work_plan nta_device_work_plan;
typedef struct nta_jit_phase_program nta_jit_phase_program;
typedef struct nta_nvme_transport nta_nvme_transport;
typedef struct nta_cxl_dax_transport nta_cxl_dax_transport;

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
  NTA_PLACEMENT_CXL_MAPPED = 3,
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
  uint32_t source_index_limit;
  uint32_t staging_index_limit;
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

typedef struct nta_cxl_dax_options {
  uint32_t struct_size;
  uint32_t api_version;
  const char *endpoint;
  uint64_t window_bytes;
  int32_t device_ordinal;
} nta_cxl_dax_options;

typedef struct nta_cxl_dax_capabilities {
  uint64_t window_bytes;
  uint64_t mapped_device_address;
  int32_t device_ordinal;
  uint32_t host_registered;
  uint32_t direct_device_visible;
} nta_cxl_dax_capabilities;

typedef struct nta_tier_descriptor {
  uint32_t source_kind;
  uint32_t capabilities;
  uint64_t device_state;
  uint64_t estimated_latency_ns;
  uint64_t estimated_bandwidth_bytes_per_second;
  uint32_t active;
  uint32_t flags;
} nta_tier_descriptor;

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
  uint64_t pending_compute_ns;
  uint64_t expected_compute_ns;
  uint64_t dropped_attributions;
  uint64_t reserved;
} nta_request_progress;

typedef struct nta_request_spec {
  uint64_t request_id;
  uint64_t deadline_clock;
  uint64_t max_outstanding_bytes;
  uint32_t slot;
  uint32_t generation;
  uint32_t tenant_id;
  uint32_t priority;
} nta_request_spec;

typedef struct nta_operator_contract {
  uint32_t magic;
  uint16_t schema_version;
  uint16_t struct_bytes;
  uint32_t runtime_abi_version;
  uint32_t family;
  uint32_t form;
  uint32_t reserved;
  uint64_t capabilities;
  uint64_t source_fingerprint_low;
  uint64_t source_fingerprint_high;
  uint64_t instrumentation_flags;
  uint32_t identity_binding;
  uint32_t demand_binding;
  uint32_t access_proof;
  uint32_t granularity_bytes;
  uint64_t tier_mask;
} nta_operator_contract;

typedef struct nta_operator_plan {
  uint32_t magic;
  uint16_t schema_version;
  uint16_t struct_bytes;
  uint32_t runtime_abi_version;
  uint32_t family;
  uint32_t supported_forms;
  uint32_t coordinate_map;
  uint32_t partial_state;
  uint32_t reduction;
  uint32_t flags;
  uint32_t reserved;
  uint64_t source_fingerprint_low;
  uint64_t source_fingerprint_high;
  uint64_t plan_fingerprint_low;
  uint64_t plan_fingerprint_high;
} nta_operator_plan;

uint32_t nta_runtime_c_api_version(void);
uint32_t nta_runtime_device_abi_version(void);
const char *nta_last_error(void);

nta_status nta_nvme_transport_create(const nta_nvme_transport_options *options,
                                     nta_nvme_transport **transport_out);
void nta_nvme_transport_destroy(nta_nvme_transport *transport);
nta_status
nta_nvme_transport_get_capabilities(const nta_nvme_transport *transport,
                                    nta_nvme_capabilities *capabilities);
nta_status nta_nvme_transport_read_stats(const nta_nvme_transport *transport,
                                         nta_nvme_queue_stats *stats);
nta_status nta_cxl_dax_transport_create(const nta_cxl_dax_options *options,
                                        nta_cxl_dax_transport **transport_out);
void nta_cxl_dax_transport_destroy(nta_cxl_dax_transport *transport);
nta_status nta_cxl_dax_transport_get_capabilities(
    const nta_cxl_dax_transport *transport, nta_cxl_dax_capabilities *capabilities);
nta_status nta_runtime_get_tier_descriptor(const nta_runtime *runtime,
                                           uint32_t source_kind,
                                           nta_tier_descriptor *descriptor);

nta_status nta_runtime_create(const nta_runtime_config *config,
                              nta_nvme_transport *nvme,
                              nta_cxl_dax_transport *cxl,
                              nta_runtime **runtime_out);
void nta_runtime_destroy(nta_runtime *runtime);
nta_status nta_runtime_set_request(nta_runtime *runtime, uint32_t slot,
                                   uint64_t request_id, uint32_t generation,
                                   uint32_t tenant_id, uint32_t priority,
                                   uint64_t deadline_clock,
                                   uint64_t max_outstanding_bytes);
nta_status nta_runtime_publish_requests_async(nta_runtime *runtime,
                                              const nta_request_spec *requests,
                                              uint32_t request_count,
                                              uint64_t cuda_stream);
nta_status nta_runtime_cancel_request(nta_runtime *runtime, uint32_t slot,
                                      uint32_t generation);
nta_status nta_runtime_set_tenant_budget(nta_runtime *runtime,
                                         uint32_t tenant_id,
                                         uint64_t max_outstanding_bytes,
                                         uint32_t weight);
nta_status nta_runtime_register_object(nta_runtime *runtime, uint32_t slot,
                                       uint64_t object_id, uint32_t version,
                                       uint64_t bytes,
                                       uint64_t staging_device_address,
                                       const nta_registered_replica *replicas,
                                       uint32_t replica_count,
                                       uint64_t *direct_device_base_out);
nta_status nta_runtime_register_indexed_host_object(
    nta_runtime *runtime, uint32_t slot, uint64_t object_id, uint32_t version,
    uint64_t source_device_address, uint64_t staging_device_address,
    uint64_t source_indices_device_address,
    uint64_t staging_indices_device_address, uint32_t index_count,
    uint32_t element_bytes, uint32_t source_stride_bytes,
    uint32_t staging_stride_bytes, uint32_t source_index_limit,
    uint32_t staging_index_limit);
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
nta_status
nta_runtime_install_nvme_object(nta_runtime *runtime, uint32_t slot,
                                uint64_t object_id, uint32_t version,
                                uint64_t source_byte_offset, uint64_t bytes,
                                uint64_t *destination_device_address_out);
nta_status nta_runtime_read_pending_count(const nta_runtime *runtime,
                                          uint32_t *pending_count);
nta_status nta_runtime_read_epoch_status(const nta_runtime *runtime,
                                         uint32_t work_ticket_count,
                                         nta_epoch_status *status);
nta_status nta_runtime_read_sticky_failed_count(const nta_runtime *runtime,
                                                uint32_t *failed_count);
nta_status nta_runtime_read_request_progress(const nta_runtime *runtime,
                                             uint32_t request_slot,
                                             nta_request_progress *progress);
nta_status nta_runtime_read_request_progress_range(
    const nta_runtime *runtime, uint32_t first_request_slot,
    uint32_t request_count, nta_request_progress *progress);
// host_destination must point to request_count entries in CUDA page-locked
// host memory and remain live until cuda_stream reaches the copy.
nta_status nta_runtime_copy_request_progress_async(const nta_runtime *runtime,
                                                   uint32_t first_request_slot,
                                                   uint32_t request_count,
                                                   uint64_t host_destination,
                                                   uint64_t cuda_stream);
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
nta_status nta_copy_host_to_device_async(uint64_t destination, uint64_t source,
                                         uint64_t bytes, uint64_t cuda_stream);

nta_status nta_device_work_plan_create(uint32_t work_item_capacity,
                                       uint32_t dependency_capacity,
                                       int32_t device_ordinal,
                                       nta_device_work_plan **plan_out);
void nta_device_work_plan_destroy(nta_device_work_plan *plan);
nta_status nta_device_work_plan_upload(
    nta_device_work_plan *plan, const nta_work_item *work_items,
    uint32_t work_item_count, const nta_acquire_requirement *dependencies,
    uint32_t dependency_count, const nta_request_work_range *requests,
    uint32_t request_count, uint64_t cuda_stream);
nta_status nta_device_work_plan_wait_on(const nta_device_work_plan *plan,
                                        uint64_t cuda_stream);
nta_status
nta_device_work_plan_synchronize_upload(const nta_device_work_plan *plan);
uint64_t nta_device_work_plan_work_items(const nta_device_work_plan *plan);
uint64_t nta_device_work_plan_dependencies(const nta_device_work_plan *plan);
uint32_t nta_device_work_plan_work_item_count(const nta_device_work_plan *plan);
uint32_t
nta_device_work_plan_dependency_count(const nta_device_work_plan *plan);
int32_t nta_device_work_plan_device_ordinal(const nta_device_work_plan *plan);

nta_status nta_jit_phase_program_create(const char *shared_object,
                                        nta_jit_phase_program **program_out);
void nta_jit_phase_program_destroy(nta_jit_phase_program *program);
nta_status nta_jit_phase_operator_contract(const nta_jit_phase_program *program,
                                           nta_operator_contract *contract_out);
nta_status nta_jit_phase_operator_plan(const nta_jit_phase_program *program,
                                       nta_operator_plan *plan_out);
nta_status nta_jit_phase_reset(const nta_jit_phase_program *program,
                               nta_runtime *runtime, uint32_t object_count,
                               uint32_t work_ticket_count,
                               uint64_t cuda_stream);
nta_status nta_jit_phase_discover(const nta_jit_phase_program *program,
                                  nta_runtime *runtime, uint64_t work_items,
                                  uint64_t dependencies,
                                  uint32_t work_item_count,
                                  uint64_t cuda_stream);
nta_status nta_jit_phase_invalidate_cached_objects(
    const nta_jit_phase_program *program, nta_runtime *runtime,
    uint32_t first_object, uint32_t object_count, uint64_t cuda_stream);
nta_status nta_jit_phase_validate_indexed_host_range(
    const nta_jit_phase_program *program, nta_runtime *runtime,
    uint32_t first_object, uint32_t object_count, uint64_t cuda_stream);
nta_status nta_jit_phase_rebind_indexed_host_pairs(
    const nta_jit_phase_program *program, nta_runtime *runtime,
    uint32_t first_object, uint32_t pair_count, uint64_t key_source,
    uint64_t key_staging, uint64_t value_source, uint64_t value_staging,
    uint64_t cuda_stream);
nta_status nta_jit_phase_preload_host(const nta_jit_phase_program *program,
                                      nta_runtime *runtime,
                                      uint32_t first_object,
                                      uint32_t object_count,
                                      uint64_t cuda_stream);
nta_status
nta_jit_phase_preload_host_pairs(const nta_jit_phase_program *program,
                                 nta_runtime *runtime, uint32_t first_object,
                                 uint32_t pair_count, uint64_t cuda_stream);
nta_status nta_jit_phase_alias_preloaded_objects(
    const nta_jit_phase_program *program, nta_runtime *runtime,
    uint32_t source_first, uint32_t destination_first, uint32_t object_count,
    uint64_t object_id_base, uint32_t version, uint64_t cuda_stream);
nta_status nta_jit_phase_progress_host(const nta_jit_phase_program *program,
                                       nta_runtime *runtime, uint32_t blocks,
                                       uint64_t cuda_stream);
nta_status nta_jit_phase_progress_indexed_host_range(
    const nta_jit_phase_program *program, nta_runtime *runtime,
    uint32_t first_object, uint32_t object_count, uint64_t cuda_stream);
nta_status nta_jit_phase_progress_validated_indexed_host_range(
    const nta_jit_phase_program *program, nta_runtime *runtime,
    uint32_t first_object, uint32_t object_count, uint64_t cuda_stream);
nta_status nta_jit_phase_progress_validated_indexed_host_range_parallel(
    const nta_jit_phase_program *program, nta_runtime *runtime,
    uint32_t first_object, uint32_t object_count,
    uint32_t copy_blocks_per_group, uint64_t cuda_stream);
/* Bound the next validated indexed copies to the in-place
 * rewritten prefix of each object's registered index arrays. */
nta_status nta_jit_phase_set_indexed_row_counts(
    const nta_jit_phase_program *program, nta_runtime *runtime,
    uint32_t first_object, uint32_t object_count, uint32_t row_count,
    uint64_t cuda_stream);
/* C API v24: turn device-selected logical pages into a validated miss-only
 * indexed transfer without a device-to-host control round trip. */
nta_status nta_jit_phase_prepare_selected_indexed_rows(
    const nta_jit_phase_program *program, nta_runtime *runtime,
    uint32_t first_object, uint32_t object_count, uint64_t selected_pages,
    uint32_t selected_page_count, uint32_t page_tokens, uint32_t token_count,
    uint64_t host_rows, uint64_t device_rows, uint64_t staged_pages,
    uint64_t source_indices, uint64_t staging_indices, uint32_t capacity,
    uint64_t copied_rows, uint64_t cuda_stream);
/* C API v26: map selected pages into a bounded physical cache, emit the
 * consumer table, and validate/compact only cache misses. */
nta_status nta_jit_phase_prepare_bounded_selected_indexed_rows(
    const nta_jit_phase_program *program, nta_runtime *runtime,
    uint32_t first_object, uint32_t object_count, uint64_t selected_pages,
    uint32_t selected_page_count, uint32_t page_tokens, uint32_t token_count,
    uint64_t host_rows, uint64_t device_rows, uint64_t cached_pages,
    uint32_t cache_slot_count, uint64_t selected_rows, uint64_t source_indices,
    uint64_t staging_indices, uint32_t capacity, uint64_t copied_rows,
    uint64_t cuda_stream);
/* C API v25: reduce pinned mapped host key rows directly into device page
 * envelopes without allocating a temporary HBM copy. element_type is 0 for
 * fp16 and 1 for bf16. */
nta_status nta_jit_phase_reduce_mapped_key_pages(
    const nta_jit_phase_program *program, uint64_t source, uint32_t source_rows,
    uint64_t source_stride_bytes, uint32_t first_row, uint32_t token_count,
    uint32_t page_tokens, uint32_t kv_heads, uint32_t head_dim,
    uint32_t element_type, uint64_t output_min, uint64_t output_max,
    uint64_t cuda_stream);
/* C API v29: the fragmented-mapping variant — token rows resolve through
 * a device int32 index array instead of a contiguous base offset. */
nta_status nta_jit_phase_reduce_mapped_indexed_key_pages(
    const nta_jit_phase_program *program, uint64_t source, uint32_t source_rows,
    uint64_t source_stride_bytes, uint64_t row_indices, uint32_t token_count,
    uint32_t page_tokens, uint32_t kv_heads, uint32_t head_dim,
    uint32_t element_type, uint64_t output_min, uint64_t output_max,
    uint64_t cuda_stream);
nta_status nta_jit_phase_progress_nvme(const nta_jit_phase_program *program,
                                       nta_runtime *runtime,
                                       uint32_t issue_budget,
                                       uint32_t completion_budget,
                                       uint64_t cuda_stream);
nta_status nta_jit_phase_publish(const nta_jit_phase_program *program,
                                 nta_runtime *runtime, uint32_t pending_budget,
                                 uint64_t cuda_stream);
nta_status nta_jit_phase_complete(const nta_jit_phase_program *program,
                                  nta_runtime *runtime,
                                  uint32_t work_ticket_count,
                                  uint64_t cuda_stream);
nta_status nta_jit_phase_complete_stream_ordered(
    const nta_jit_phase_program *program, nta_runtime *runtime,
    uint64_t work_items, uint32_t work_item_count, uint64_t cuda_stream);

#ifdef __cplusplus
}
#endif
