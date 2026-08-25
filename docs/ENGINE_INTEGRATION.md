# Engine integration boundaries

The runtime core is engine-neutral. Framework code projects scheduler metadata
into the same request-bound work-unit contract.

## SGLang

The installed package uses SGLang's `sglang.srt.plugins` general-plugin
entry point. Registration is performed during SGLang's plugin-loading phase;
attention selection goes through SGLang's `ATTENTION_BACKENDS` registry, and
metadata/lifecycle interception goes through `HookRegistry`. The plugin is
responsible for:

- registering the pinned `nta_flashinfer` backend;
- preserving request IDs and request-pool slots through graph views;
- preserving an optional `_nta_request_tenant_ids` vector through graph views;
- routing HiCache host-load ownership, admission, cancellation, and release;
- attaching priority metadata.

Graph replay metadata is installed as `AROUND` hooks for the prefill static
batch, capture-preparation, and decode replay-view boundaries. It does not
replace those methods directly. This keeps duplicate detection and hook
ordering under SGLang's lifecycle, while the pinned version check makes a
target-path change a hard error when the NTA backend is constructed rather
than a silent partial install.

`SglangAdapter.bind_forward` converts those fields into an
`EngineBatch`. The backend then creates an `ExecutionSession` for each
real FlashInfer attention launch. It does not use batch position as a
persistent request slot when SGLang provides the actual pool slot.

Tenant identity is an aligned, deployment-owned annotation: a plugin or
serving gateway may attach `_nta_request_tenant_ids` to the forward batch, and
the graph hooks copy it with `rids` and pool slots. Missing annotation means
the explicit default tenant 0; it is never inferred from request text or
batch position. Quotas are configured once at worker startup with
`NTA_TENANT_BUDGETS=id:bytes[:weight],...` and are enforced by native device
admission counters.

The SGLang implementation currently requires the tested 0.5.14 API, FA2
FlashInfer kernels, full-attention page geometry, and valid request identity.
Unsupported metadata or graph layouts fail closed.

The stock FlashInfer wrapper is selected for a resident-only forward. This
avoids charging requests that never touch HiCache for NTA instrumentation. An
external forward has two exact cases. If the NTA pipeline has completely
materialized the physical HiCache pages before the first consumer layer, the
framework wrapper is reused after the NTA readiness and lifetime fence; this
keeps the numerical consumer at stock cost. If pages are still arriving, the
typed NTA consumer executes the ready work and accounts blocked work through
the native contract. There is no virtual page-table or approximate attention
fallback. The serving activation gate requires exact demand and zero silent
fallbacks.

### Tier attachment

The serving process selects one `ServingTierService` from `NTA_SERVING_TIER`.
`host_staged` is the default and retains the indexed host path. `nvme`
requires `NTA_NVME_ENDPOINT` and an exact `NTA_TIER_CATALOG`; each requested
device-page group is validated as one contiguous K/V extent, installed as an
HBM object no larger than the controller's advertised MDTS/PRP transfer limit.
The runtime reuses a slot's HBM/DMA buffer when the next exact extent fits, so
mapping and allocation stay in the setup/lifetime path rather than recurring
for every steady-state I/O. Transfers are progressed by the GPU-owned NVMe
queue. The FlashInfer KV chunk therefore
must be small enough for both K/V extents to satisfy that limit; an oversized
exact group fails closed before any object is installed. The host HiCache load
is used only as a lifetime and request-metadata signal in this mode; its bytes
never become a data proxy. `cxl_dax` requires an explicit devdax endpoint,
matching window, and catalog; its K/V extents become direct device
dependencies in the work plan.

There is no physical-tier fallback. Endpoint, catalog, mapping, IOMMU, or
device-visibility failure aborts the selected serving profile. Engine stats
publish the tier, catalog digest, capability evidence, and
`tier_fallback: false`, so an artifact validator can distinguish a real
physical-tier result from a host run.

The selected service exposes the same typed resource contract used by the
native tier descriptor: resource kind, capability set, protocol owner,
allocation owners, runtime-owned directory, setup requirements, and
steady-state path. `NTA_STAGING_BYTE_CAPACITY` limits runtime-owned HBM
staging allocations; engine-owned registered staging is non-owning and must
be governed by the framework allocator's own quota.

## vLLM

The vLLM integration boundary is `VllmV1Hook` for the pinned vLLM 0.13.0 V1
worker. The dependency-free `VllmSchedulerProjection` remains useful for
contract fixtures, while the real hook extracts:

- request IDs;
- the scheduler's request IDs;
- exact block tables from `InputBatch.block_table[0]`'s CPU mirror;
- optional tenant, priority, and deadline annotations supplied by deployment.

`VllmV1Hook.bind_forward` owns a bounded stable request-ID slot table, so a
mutable vLLM input-batch row is never treated as long-lived identity. It binds
the same `RequestIdentityRegistry` and produces the same `EngineBatch` as
SGLang. The hook checks the installed vLLM version, rejects missing rows,
multiple KV groups, empty exact demand, and finish/reschedule ID ambiguity. It
must be called after vLLM's `_update_states` and before the attention launch;
that call is control-plane metadata publication, not a per-request I/O path.
The hook must not add a second generation tracker, policy taxonomy, or native
work ABI. The same `ServingTierService` is passed to a pinned consumer hook;
the tier catalog and native transport are selected once per worker.

vLLM's `vllm.general_plugins` entry point is a process/bootstrap extension
point; it is not a substitute for the numerical `AttentionBackend`/
`AttentionImpl` path. vLLM's V1 `KVConnector` lifecycle is useful for
transport preparation and readiness, but a connector alone is not an NTA
attention consumer. A complete vLLM integration therefore needs a pinned V1
attention consumer that calls the same engine-neutral execution core after
the projection and before the stock FlashInfer numerical result is accepted.
The vLLM and SGLang adapters may share the FlashInfer operator ABI, but they
must not share framework metadata or lifecycle code.

`VllmV1Hook` is now a real pinned worker projection, but it is not by itself a
complete vLLM serving backend: a vLLM model-runner/attention consumer still
has to pass the returned `EngineBatch` into the NTA execution core. An artifact
may not label vLLM results as end-to-end evidence until that consumer is wired
and passes the same exact-demand, correctness, tier-placement, and performance
gates used by SGLang. Version drift is an intentional hard failure, not a
best-effort field guess.

The distinction is machine-checked in engine statistics. A
`consumer_contract.kind` of `projection_only` is valid for adapter tests but
invalid for serving evidence. `native_work_unit` requires exact demand, a
typed work plan, native submission, and numerical consumption; a
`framework_reference` contract is the explicit stock-attention reference
after an exact acquisition fence. This prevents a scheduler projection or a
KV-prefetch hook from being mistaken for the NTA attention consumer. The
artifact validators share `experiments/consumer_contract.py`, so malformed
types and schema drift fail before an artifact is assembled.

## Common contract

Both adapters must satisfy:

- unique slots in a batch;
- stable ID for an unchanged slot;
- generation increment when a slot receives a new request;
- exact epoch assignment by the engine boundary;
- aligned tenant IDs when logical multi-tenant quotas are enabled;
- fail-closed behavior when identity is absent;
- identical work-unit and demand semantics downstream.

The shared `EngineBoundary` protocol is the architectural seam: both engines
produce an `EngineBatch`, and the engine-neutral `ExecutionSession` is the
only owner of work-unit availability after the handoff.

Framework-specific tests belong under `tests/runtime/adapters.py`; core tests
must not import SGLang or vLLM.
