# NTA: late-bound heterogeneous execution

This branch is the engineering refactor of NTA around one mechanism:

> Bind each heterogeneous work unit to its request generation, demand, and
> availability as late as correctness permits, then execute bounded runnable
> groups at a granularity chosen from measured costs.

The mechanism is shared by conventional, late-bound, and exact partial
execution.  Device selection, request identity, bounded staging, compiler
mapping, and engine feedback are cooperating parts of that mechanism; none is
the headline in isolation.

## Start here

The current source of truth is deliberately small:

- [REFRACTOR_DESIGN.md](docs/REFRACTOR_DESIGN.md) — architecture, ownership,
  invariants, and migration status.
- [EXPERIMENT_DESIGN.md](docs/EXPERIMENT_DESIGN.md) — redesigned evaluation
  questions and fair comparison arms.
- [DOCUMENT_STATUS.md](docs/DOCUMENT_STATUS.md) — which older documents are
  historical evidence rather than implementation authority.

Older result ledgers are retained for provenance.  They may contain useful
measurements, but they do not define this branch's API, mechanism, or claims.

## Architecture

```text
SGLang / vLLM adapter
        │ request slot + generation + engine metadata
        ▼
WorkBatch ── DemandDescriptor ── exact selected work IDs
        │
        ▼
protocol planner ── granularity cost model ── runnable groups
        │
        ▼
NTA runtime ── claims/dependencies ── staging/transport ── telemetry
        │
        ▼
compiler-mapped consumer ── exact partial state / final output
```

The native storage ABI remains canonical in
[include/nta/RuntimeABI.h](include/nta/RuntimeABI.h).  The Python semantic
layer is in:

- [work_unit.py](python/nta_runtime/work_unit.py): demand, identity,
  availability, and heterogeneous batches;
- [execution_protocol.py](python/nta_runtime/execution_protocol.py): protocol
  state transitions and transparent granularity costs;
- [adapters](python/nta_runtime/adapters): engine-specific request translation.

The existing compiler pass and native runtime remain responsible for lowering,
dependency storage, transport, claims, and device execution.  The refactor
does not create a second request tracker or a second native ABI.

## Status of this branch

Implemented and tested:

- exact/approximate demand is explicit in the type contract;
- request generation and demand epoch are checked on every ledger transition;
- conventional, late-bound, and partial protocol forms share one work-unit
  state machine;
- request identity is factored behind a common adapter boundary;
- SGLang identity/configuration integration and a dependency-free vLLM seam;
- a transparent granularity model that can select conventional execution when
  fine-grained control is not worth its cost;
- focused runtime tests and the existing native/compiler test suite.

Still being migrated, and therefore not claimed as complete:

- route the full SGLang staging/selection path through `WorkBatch`;
- merge and fairly validate the conventional E6 execution arm;
- connect compiler work mapping and consumer partial state to the semantic
  contract;
- move remaining claim/staging policy out of the monolithic SGLang backend;
- bind the vLLM seam to one pinned vLLM release;
- replace old protocol-specific qualification scripts with contract-based
  validators.

This distinction is intentional: a configuration seam is not presented as a
fully migrated serving implementation, and historical speedups are not used
as results for the refactor branch.

## Build and tests

Configure and build in the normal project build directory:

```bash
cmake -S . -B build
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

The focused Python contract tests can also run without a GPU:

```bash
PYTHONPATH=python python tests/runtime/work_unit.py
PYTHONPATH=python python tests/runtime/adapters.py
```

GPU, CUDA, NVMe, and serving tests remain capability-gated by CMake and their
existing environment checks.

## Scope

The primary correctness case is exact demand: all arms consume the same
selected work IDs and must produce the same numerical result.  Top-k/DSA and
other approximate selectors are optional quality-gated studies, not a silent
substitute for the exact system mechanism.  SGLang is the first engine
integration; vLLM is designed through the same adapter contract rather than by
copying SGLang-specific policy into the runtime.
