# Engine integration boundaries

The runtime core is engine-neutral. Framework code projects scheduler metadata
into the same request-bound work-unit contract.

## SGLang

The plugin is responsible for:

- registering the pinned `nta_flashinfer` backend;
- preserving request IDs and request-pool slots through graph views;
- routing HiCache host-load ownership, admission, cancellation, and release;
- attaching priority metadata.

`SglangAdapter.bind_forward` converts those fields into an
`EngineBatch`. The backend then creates an `ExecutionSession` for each
real FlashInfer attention launch. It does not use batch position as a
persistent request slot when SGLang provides the actual pool slot.

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
HBM object no larger than the controller's advertised MDTS/PRP transfer limit,
and progressed by the GPU-owned NVMe queue. The FlashInfer KV chunk therefore
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

## vLLM

The vLLM integration boundary is `VllmSchedulerProjection`. A pinned vLLM
adapter supplies:

- request IDs;
- scheduler/request slots;
- optional priorities;
- optional deadline clocks.

`VllmAdapter.bind_forward` binds the projection to the same
`RequestIdentityRegistry` and produces the same `EngineBatch` as SGLang. The
framework-specific hook is responsible for creating the projection from its
current scheduler object; NTA deliberately does not import vLLM internals.
That keeps version churn at one typed boundary while making missing identity a
hard error. A vLLM hook may add block-table and cancellation extraction only at
that boundary; it must not add a second generation tracker, policy taxonomy,
or native work ABI. The same `ServingTierService` is passed to a pinned vLLM
consumer hook; the tier catalog and native transport are selected once per
worker, not once per framework implementation.

The current vLLM adapter is a tested structural seam, not a serving backend:
it proves the common identity/epoch contract without importing vLLM internals.
An artifact may not label vLLM results as end-to-end evidence until a pinned
vLLM hook supplies the projection and passes the same exact-demand,
correctness, tier-placement, and performance gates used by SGLang.

## Common contract

Both adapters must satisfy:

- unique slots in a batch;
- stable ID for an unchanged slot;
- generation increment when a slot receives a new request;
- exact epoch assignment by the engine boundary;
- fail-closed behavior when identity is absent;
- identical work-unit and demand semantics downstream.

The shared `EngineBoundary` protocol is the architectural seam: both engines
produce an `EngineBatch`, and the engine-neutral `ExecutionSession` is the
only owner of work-unit availability after the handoff.

Framework-specific tests belong under `tests/runtime/adapters.py`; core tests
must not import SGLang or vLLM.
