#pragma once

#include "nta/RuntimeABI.h"
#include "nta/Tier.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace nta {

class DeviceWorkPlan;
struct WorkPlan;
class NvmeBuffer;
class NvmeTransport;
class CxlDaxTransport;

enum class Placement {
  Hbm,
  HostMapped,
  HostStaged,
  CxlMapped,
};

struct RuntimeBackends {
  std::shared_ptr<NvmeTransport> nvme;
  std::shared_ptr<CxlDaxTransport> cxl;
};

struct RuntimeConfig {
  std::uint32_t requestCapacity;
  std::uint32_t objectCapacity;
  // Maximum simultaneously published acquisitions. This active-frontier
  // bound is independent of the number of objects in the catalog.
  std::uint32_t intentCapacity;
  std::uint32_t workTicketCapacity;
  std::uint32_t maxReplicasPerObject = 1;
  std::uint32_t maxDependenciesPerWorkTicket = 8;
  int deviceOrdinal = -1;
  // Direct CTA submission is useful only for a small, isolated miss. Batched
  // traffic defaults to the request-aware scheduler warp; deployments may
  // enable the one-shot CTA path after measuring their transfer-size regime.
  bool enableCtaNvmeTryIssue = false;
  // Zero selects requestCapacity for one-tenant-per-slot deployments;
  // otherwise tenant storage is independently bounded.
  std::uint32_t tenantCapacity = 0;
  // Bounds only HBM staging destinations allocated and owned by HostRuntime;
  // zero selects an unbounded limit. Engine-registered staging remains
  // governed by the engine's allocator.
  std::uint64_t stagingByteCapacity = UINT64_MAX;
};

struct StagingUsage {
  std::uint64_t bytes = 0;
  std::uint64_t capacity = 0;
  std::uint64_t highWaterBytes = 0;
};

struct HostReplicaSpec {
  std::span<const std::byte> contents;
  Placement placement;
};

// One request-directory update. publishRequestsAsync requires unique slots in
// increasing order and a stream ordered after all work for the old generations.
struct RequestSpec {
  std::uint32_t slot;
  std::uint64_t requestId;
  std::uint32_t generation;
  std::uint32_t tenantId = 0;
  std::uint32_t priority = 0;
  std::uint64_t deadlineClock = 0;
  std::uint64_t maxOutstandingBytes = UINT64_MAX;
};

// Non-owning registration for allocations managed by an inference engine or
// another memory runtime. Every source address must already be device-visible;
// direct sources must satisfy the consumer kernel's alignment requirements.
// HostStaged sources may have arbitrary byte alignment. HostRuntime never frees
// registered source or staging allocations.
struct RegisteredReplicaSpec {
  const void *sourceDeviceAddress;
  Placement placement;
  const void *tensorMap = nullptr;
  std::uint64_t estimatedLatencyNs = 0;
  std::uint64_t estimatedBandwidthBytesPerSecond = 0;
};

struct ObjectHandle {
  std::uint32_t slot;
  void *directDeviceBase;
};

struct IndexedHostObjectSpec {
  std::uint64_t objectId;
  std::uint32_t version;
  const void *sourceDeviceAddress;
  void *stagingDeviceAddress;
  const std::uint32_t *sourceIndicesDevice;
  const std::uint32_t *stagingIndicesDevice;
  std::uint32_t indexCount;
  std::uint32_t elementBytes;
  std::uint32_t sourceStrideBytes;
  std::uint32_t stagingStrideBytes;
  std::uint32_t sourceIndexLimit;
  std::uint32_t stagingIndexLimit;
  // The caller has enqueued the transfer on the same stream after directory
  // publication and will gate every consumer with a post-transfer event.
  bool preacquired = false;
};

struct EpochStatus {
  std::uint32_t total = 0;
  std::uint32_t fresh = 0;
  std::uint32_t pending = 0;
  std::uint32_t ready = 0;
  std::uint32_t done = 0;
  std::uint32_t cancelled = 0;
  std::uint32_t failed = 0;
  std::uint32_t initializing = 0;

  [[nodiscard]] bool succeeded() const noexcept { return done == total; }
  [[nodiscard]] bool hasFailure() const noexcept {
    return cancelled != 0 || failed != 0;
  }
};

class HostRuntime {
public:
  explicit HostRuntime(RuntimeConfig config);
  HostRuntime(RuntimeConfig config, RuntimeBackends backends);
  ~HostRuntime();

  HostRuntime(const HostRuntime &) = delete;
  HostRuntime &operator=(const HostRuntime &) = delete;
  HostRuntime(HostRuntime &&) noexcept;
  HostRuntime &operator=(HostRuntime &&) noexcept;

  void setRequest(std::uint32_t slot, std::uint64_t requestId,
                  std::uint32_t generation, std::uint32_t tenantId = 0,
                  std::uint32_t priority = 0, std::uint64_t deadlineClock = 0,
                  std::uint64_t maxOutstandingBytes = UINT64_MAX);
  // Publish changed request identities without synchronizing the host. The
  // supplied stream is the reuse boundary: prior users of every old slot
  // generation must complete before these copies execute, and all consumers
  // of the new generations must execute after them on this stream.
  void publishRequestsAsync(std::span<const RequestSpec> requests,
                            cudaStream_t stream);
  void cancelRequest(std::uint32_t slot, std::uint32_t generation);
  void setTenantBudget(std::uint32_t tenantId,
                       std::uint64_t maxOutstandingBytes,
                       std::uint32_t weight = 1);

  ObjectHandle installObject(std::uint32_t slot, std::uint64_t objectId,
                             std::uint32_t version,
                             std::span<const std::byte> contents,
                             Placement placement);
  ObjectHandle
  installReplicatedObject(std::uint32_t slot, std::uint64_t objectId,
                          std::uint32_t version,
                          std::span<const HostReplicaSpec> replicas);
  // Registration replaces the device directory entry. A slot may be reused
  // only after the acquisition epoch that referenced its previous entry has
  // been reset and quiesced; the runtime rejects an in-flight replacement.
  // Replica/source buffers remain caller-owned unless an API says otherwise,
  // and must remain live through that acquisition epoch.
  ObjectHandle registerObject(std::uint32_t slot, std::uint64_t objectId,
                              std::uint32_t version, std::size_t bytes,
                              void *stagingDeviceAddress,
                              std::span<const RegisteredReplicaSpec> replicas);
  // Register a non-owning pinned-host to HBM row gather. Index arrays must be
  // uint32_t CUDA allocations and remain live through the acquisition epoch.
  ObjectHandle registerIndexedHostObject(
      std::uint32_t slot, std::uint64_t objectId, std::uint32_t version,
      const void *sourceDeviceAddress, void *stagingDeviceAddress,
      const std::uint32_t *sourceIndicesDevice,
      const std::uint32_t *stagingIndicesDevice, std::uint32_t indexCount,
      std::uint32_t elementBytes, std::uint32_t sourceStrideBytes,
      std::uint32_t stagingStrideBytes, std::uint32_t sourceIndexLimit,
      std::uint32_t stagingIndexLimit);
  void registerIndexedHostObjects(
      std::uint32_t firstSlot, std::span<const IndexedHostObjectSpec> objects);
  void registerIndexedHostObjectsAsync(
      std::uint32_t firstSlot, std::span<const IndexedHostObjectSpec> objects,
      cudaStream_t stream);
  // Replace an indexed directory after the caller has recorded an event on
  // the stream that consumed the previous entries. Borrowed registrations
  // use a stream wait and runtime-owned destinations are retired by the same
  // event-driven reclaimer as NVMe objects; steady-state replacement does not
  // perform a device-wide or host event synchronization. A missing event is
  // valid only for first installation; replacing an owned object then takes a
  // conservative device-quiescence fallback. An invalid event is rejected by
  // CUDA.
  void registerIndexedHostObjectsAsyncQuiesced(
      std::uint32_t firstSlot, std::span<const IndexedHostObjectSpec> objects,
      cudaStream_t stream, cudaEvent_t priorConsumerEvent);
  // Install an NVMe object using a runtime-owned destination. When the slot
  // already owns a large-enough NVMe buffer, only the source-range directory
  // entry is republished; the HBM allocation and DMA mapping remain alive.
  ObjectHandle installNvmeObject(std::uint32_t slot, std::uint64_t objectId,
                                 std::uint32_t version,
                                 std::uint64_t sourceByteOffset,
                                 std::size_t bytes);
  ObjectHandle installNvmeObject(std::uint32_t slot, std::uint64_t objectId,
                                 std::uint32_t version,
                                 std::uint64_t sourceByteOffset,
                                 std::size_t bytes,
                                 std::unique_ptr<NvmeBuffer> destination);
  // Stream-ordered NVMe directory replacement.  ``priorConsumerEvent`` must
  // cover every consumer of the old slot generation.  The new directory
  // entry is published on ``stream`` and the old runtime-owned destination is
  // retired by an event-driven reclaimer; the caller is never forced through
  // a device-wide synchronization in the steady-state replacement path.
  // Passing a null event is valid only when the slot has never been installed.
  ObjectHandle installNvmeObjectAsync(
      std::uint32_t slot, std::uint64_t objectId, std::uint32_t version,
      std::uint64_t sourceByteOffset, std::size_t bytes, cudaStream_t stream,
      cudaEvent_t priorConsumerEvent);
  void bindTensorMaps(std::uint32_t objectSlot, std::uint32_t relativeReplica,
                      const void *replicaTensorMap,
                      const void *stagingTensorMap = nullptr);

  [[nodiscard]] abi::RuntimeView *deviceView() const noexcept;
  [[nodiscard]] int deviceOrdinal() const noexcept;
  [[nodiscard]] const RuntimeConfig &config() const noexcept;
  [[nodiscard]] TierDescriptor tierDescriptor(abi::SourceKind kind) const;
  [[nodiscard]] StagingUsage stagingUsage() const noexcept;
  [[nodiscard]] abi::RequestContext readRequest(std::uint32_t slot) const;
  [[nodiscard]] abi::TenantContext readTenant(std::uint32_t tenantId) const;
  [[nodiscard]] abi::RequestProgress
  readRequestProgress(std::uint32_t slot) const;
  [[nodiscard]] std::vector<abi::RequestProgress>
  readRequestProgress(std::uint32_t firstSlot, std::uint32_t count) const;
  // Enqueue a generation-stamped progress snapshot without synchronizing the
  // device. The destination must be CUDA page-locked host memory and remain
  // live until all prior work on stream has completed.
  void copyRequestProgressAsync(
      std::uint32_t firstSlot, std::span<abi::RequestProgress> destination,
      cudaStream_t stream) const;
  [[nodiscard]] abi::ObjectEntry readObject(std::uint32_t slot) const;
  [[nodiscard]] abi::ReplicaEntry
  readReplica(std::uint32_t objectSlot,
              std::uint32_t relativeReplica = 0) const;
  [[nodiscard]] abi::WorkTicket readWorkTicket(std::uint32_t slot) const;
  // Relative device-global nanoseconds from epoch start until each work ticket
  // first became runnable. A zero value means it was runnable at launch.
  [[nodiscard]] std::vector<std::uint64_t>
  readWorkRunnableNs(std::uint32_t count) const;
  [[nodiscard]] abi::WorkDependency
  readWorkDependency(std::uint32_t workTicket,
                             std::uint32_t relativeDependency) const;
  [[nodiscard]] abi::IntentPool readIntentPool() const;
  // One bulk transfer of the active work-ticket prefix. This is the
  // completion contract used between finite progress rounds.
  [[nodiscard]] EpochStatus
  readEpochStatus(std::uint32_t workTicketCount) const;
  // Monotonic across epoch resets; any non-zero value poisons the runtime.
  [[nodiscard]] std::uint32_t readStickyFailedCount() const;
  // Current work tickets whose state is Pending.
  [[nodiscard]] std::uint32_t readPendingCount() const;
  // Entries appended to the bounded pending index in the current epoch.
  [[nodiscard]] std::uint32_t readPendingIndexCount() const;
  [[nodiscard]] DeviceWorkPlan uploadWorkPlan(const WorkPlan &plan) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace nta
