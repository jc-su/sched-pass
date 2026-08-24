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
or native work ABI.

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
