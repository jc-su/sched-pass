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

## vLLM

The vLLM integration boundary is `VllmSchedulerProjection`. A pinned vLLM
adapter supplies:

- request IDs;
- scheduler/request slots;
- optional priorities;
- optional deadline clocks.

`VllmAdapter` binds the projection to the same `RequestIdentityRegistry`
and produces the same `EngineBatch` as SGLang. It intentionally does not
import vLLM internals, so the core contract tests remain independent of a
framework installation.

A future pinned vLLM transport adapter may add block-table extraction and
cancellation hooks. It must not add a second generation tracker, policy
taxonomy, or native work ABI.

## Common contract

Both adapters must satisfy:

- unique slots in a batch;
- stable ID for an unchanged slot;
- generation increment when a slot receives a new request;
- exact epoch assignment by the engine boundary;
- fail-closed behavior when identity is absent;
- identical work-unit and demand semantics downstream.

Framework-specific tests belong under `tests/runtime/adapters.py`; core tests
must not import SGLang or vLLM.
