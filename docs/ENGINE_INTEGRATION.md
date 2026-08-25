# Engine integration boundaries

The runtime core is engine-neutral. Framework code projects scheduler metadata
into the same request-bound work-unit contract.

## SGLang

The installed package uses SGLang's `sglang.srt.plugins` general-plugin
entry point. Registration is performed during SGLang's plugin-loading phase;
attention selection goes through SGLang's `ATTENTION_BACKENDS` registry, and
metadata/lifecycle interception goes through `HookRegistry`.

The supported single-engine environment is installable as
`pip install -e '.[sglang]'`. The extra pins the tested SGLang and FlashInfer
versions; the base package does not import either framework. This is important
because SGLang's general
plugin entry point is discovered process-wide and its hook targets are not a
stable cross-version ABI.

The plugin is responsible for:

- registering the pinned `nta_flashinfer` backend;
- preserving request IDs and request-pool slots through graph views;
- preserving one typed `_nta_forward_metadata` sidecar (slots, priorities, and
  optional tenant IDs) through graph views;
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
serving gateway may populate the typed `_nta_forward_metadata` sidecar on the
forward batch, and the graph hooks copy it with `rids` and pool slots. Missing
annotation means the explicit default tenant 0; it is never inferred from
request text or batch position. Quotas are configured once at worker startup with
`NTA_TENANT_BUDGETS=id:bytes[:weight],...` and are enforced by native device
admission counters.

The vLLM worker has no stable upstream tenant field, so its deployment adapter
may use `NTA_TENANT_REQUEST_PREFIXES=tenant_id:request-id-prefix,...` to create
the same explicit mapping. Prefixes must be disjoint and are rejected at
startup when malformed or overlapping; an unmatched request remains tenant 0.
This is an identity adapter, not a classifier or a per-request scheduler
policy.

The SGLang implementation currently requires the tested 0.5.16 API, FA2
FlashInfer kernels, full-attention page geometry, and valid request identity.
Unsupported metadata or graph layouts fail closed.

The serving harnesses configure this boundary before importing SGLang.  A
`nta_flashinfer` invocation of `SglangSmoke.py` or `SglangHiCache.py` therefore
automatically enters the same ABI-tagged clang/pass/overlay environment as
`tools/jit/activate.py`; a custom application still uses the activation
launcher explicitly.  This prevents a module with an NTA filename from being
compiled by stock nvcc without the exported phase ABI.

`SglangSmoke.py` reports backend selection separately from execution evidence:
`nta_backend_selected` means the requested backend was accepted, while
`nta_execution_verified` and `nta_integrated` require a published
`native_work_unit` consumer contract.  A resident-only smoke can therefore
prove framework/JIT startup and numerical generation without being mistaken
for an external-tier mechanism run.

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
`host_staged` is the default host-memory serving tier and retains the indexed
host path. `host_mapped` is not a second hidden host serving mode: it is the
explicit matched baseline for an NVMe DMA destination. `nvme`
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
payload owner, transfer-destination owner, runtime-owned directory, setup
requirements, and steady-state path. `NTA_STAGING_BYTE_CAPACITY` limits the
runtime-owned HBM destination of the host-staged path; the engine-owned pinned
HiCache payload remains under the framework allocator's quota and is borrowed
only until the explicit CUDA completion fence.

## vLLM

vLLM 0.26.0 and SGLang 0.5.16 share the tested runtime core: Torch 2.11.0,
FlashInfer 0.6.14, and the same CUDA/ABI build. Their published wheels do not
currently form a resolver-clean combined extra: vLLM pins
`apache-tvm-ffi==0.1.10`, while SGLang pins `==0.1.11` (and they also differ
on several auxiliary helper pins). This is an upstream packaging conflict,
not an NTA runtime fallback. The project therefore treats the dual-engine
profile as an explicit, no-dependency framework install on top of a validated
core environment:

```bash
python -m pip install torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 \
  flashinfer-python==0.6.14 sglang-kernel==0.4.5
python -m pip install --no-deps vllm==0.26.0 sglang==0.5.16
python -m pip install -e .
```

FlashInfer 0.6.14 does not use the separately versioned 0.6.12 cubin
package. If an older `flashinfer-cubin` distribution is already installed,
remove that stale package before importing FlashInfer; bypassing its version
check would allow incompatible kernels into the experiment. The
`nta-engine-environment` gate is the authoritative check for this complete
matrix.

The framework distributions must be installed only after the deployment has
resolved their non-conflicting serving dependencies. `tests/runtime/engine_environment.py`
and the native CTest gate then verify the actual interpreter, imports, and
framework/plugin versions; artifact metadata records those versions. The
supported project entry point is `pip install -e '.[sglang,vllm]'`. On the
current pinned SGLang/vLLM profile, pip may need the framework dependencies
pre-provisioned because their upstream `numba` and `apache-tvm-ffi` pins
conflict; in that already-provisioned environment the equivalent editable
install is `pip install -e '.[sglang,vllm]' --no-deps`. The environment gate
remains mandatory and prevents an unqualified combination from becoming an
artifact claim.

Serving harnesses share `benchmarks/serving/cuda_environment.py`. They select
the toolkit matching `torch.version.cuda` (or an explicit `--cuda-home`), pass
the same host compiler to tvm-ffi and FlashInfer, and apply the glibc feature
macro workaround only to CUDA toolkits that need it. This avoids silently
using a `/usr/local/cuda` symlink from a different toolkit generation.

The vLLM plugin has two distinct responsibilities:

1. `nta_runtime.plugins.vllm:register` registers
   `AttentionBackendEnum.CUSTOM`. The backend class then installs the worker
   bridge at the execution edge; registration does not import private GPU
   worker modules in the frontend process.
2. `NtaVllmFlashInferImpl.forward` is the numerical consumer. vLLM's enclosing
   `Attention` layer owns the framework KV-cache update; this implementation
   reads the resulting exact demand from the pinned worker projection, resets
   one finite NTA epoch, builds an `ExecutionSession`/`DeviceWorkPlan`, and
   calls the instrumented FlashInfer wrapper with the NTA runtime/work-plan
   ABI. It never performs a duplicate cache write.

The pinned vLLM 0.26 profile uses the `vllm.v1` API with the V2 GPU model
runner. Its bridge intercepts block-table allocation writes into an
adapter-owned CPU mirror, then binds `req_ids`, scheduler output, and that
mirror in `prepare_attn`. It never copies a GPU block table back to the host
per forward. The bridge exposes one context-local typed `EngineBatch` to every
opaque attention op in that forward, owns the same bounded stable
request-generation registry as SGLang, and closes the runtime on runner
shutdown. Select it with vLLM's `--attention-backend CUSTOM` and provide
`FLASHINFER_WORKSPACE_BASE` containing the vLLM-compatible instrumented
FlashInfer module (the defaults are the tensor-core
`nta_batch_prefill_default_v2_hooked` for FP16 and
`nta_batch_prefill_default_v2_hooked_bf16` for BF16; the separate
`nta_batch_decode_default_v2_hooked*` artifacts are retained for the
non-tensor-core decode profile).

The native vLLM consumer is intentionally qualified first for resident CUDA
KV, one KV group, pure single-token decode, FA2 (non-TRTLLM), and eager mode.
It is opt-in with `NTA_VLLM_NATIVE=1`; the default is vLLM's reference
attention because resident-only work does not exercise a remote-tier
dependency and must not pay the NTA protocol overhead.
The builder reports no CUDA-graph support until plan upload/replay has its own
graph-stability gate. Prefill, mixed batches, TRTLLM, and external NVMe/CXL
loads remain explicit fail-closed boundaries; `NTA_VLLM_ALLOW_STOCK_FALLBACK=1`
is a debugging reference only and is invalid for native artifacts.

Consequently, a vLLM artifact can currently claim native NTA execution only for
that resident decode profile. The shared vLLM/SGLang runtime and tenant
contract are integrated, but that does not turn vLLM's resident projection into
an external-tier implementation.

`benchmarks/serving/VllmSmoke.py` is the reproducible resident integration gate:
run it once with `--backend stock` and once with `--backend nta` using the same
model, request count, seed, and limits. The NTA run requires the worker's
`native_work_unit` evidence and reports both text and token-ID correctness
digests. It is a correctness/integration gate, not a remote-tier performance
claim.

vLLM's V1 `KVConnector` remains the correct next seam for external tier
ownership/readiness: scheduler metadata and worker load/fence lifecycle belong
there, while `AttentionImpl` remains the numerical consumer. A connector alone
must never be reported as NTA execution. The project is no longer blocked on
vLLM's attention path, but multi-tier vLLM evidence is gated on a concrete
KVConnector implementation and its exact correctness/ownership tests.

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
