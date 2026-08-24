# NTA: late-bound heterogeneous execution

This branch is organized around one system mechanism:

> Bind every heterogeneous work unit to its request generation, exact demand,
> and availability as late as correctness permits, then execute bounded
> runnable groups at a granularity chosen from measured costs.

The mechanism is an execution protocol, not a top-k/DSA selector. The serving
path is exact: every compared arm consumes the same demand identities and
must preserve the same numerical contract.

## Source of truth

- [docs/REFRACTOR_DESIGN.md](/home/jcsu/Dev/sched-pass/docs/REFRACTOR_DESIGN.md):
  architecture and ownership.
- [docs/ENGINE_INTEGRATION.md](/home/jcsu/Dev/sched-pass/docs/ENGINE_INTEGRATION.md):
  SGLang/vLLM boundaries.
- [docs/EXPERIMENT_DESIGN.md](/home/jcsu/Dev/sched-pass/docs/EXPERIMENT_DESIGN.md):
  redesigned evaluation.
- [docs/DOCUMENT_STATUS.md](/home/jcsu/Dev/sched-pass/docs/DOCUMENT_STATUS.md):
  historical-document policy.

## Runtime structure

```
engine adapter
  -> request slot + generation
WorkBatch / DemandDescriptor
  -> exact heterogeneous work units
protocol planner
  -> granularity and bounded runnable groups
semantic/native bridge
  -> validated WorkItem + dependency ABI
compiler-mapped consumer
  -> exact attention execution and telemetry
```

The core modules are:

- `python/nta_runtime/work_unit.py`: exact demand, identity, availability,
  and heterogeneous batches;
- `python/nta_runtime/execution_protocol.py`: protocol state machine and
  transparent granularity cost model;
- `python/nta_runtime/execution_core.py`: one execution session for one
  attention launch;
- `python/nta_runtime/execution_planner.py`: measured host/device planning;
- `python/nta_runtime/adapters/`: engine-specific metadata projections;
- `python/nta_runtime/runtime.py`: semantic-to-native upload validation.

SGLang is the first serving implementation. It keeps host HiCache ownership,
request identity, demand, and availability at explicit boundaries while
building an execution session for every real FlashInfer attention launch.
The vLLM adapter exposes the same request projection without importing vLLM.

Retired selector-specific serving and graph-specialization paths were removed
from the current implementation. Their old source is recoverable through Git
history; it is not an alternative runtime interface.

## Build and tests

```bash
cmake -S . -B build
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

GPU- and framework-dependent tests remain capability-gated. The dependency-free
contract tests can run with:

```bash
PYTHONPATH=python python tests/runtime/work_unit.py
PYTHONPATH=python python tests/runtime/execution_core.py
PYTHONPATH=python python tests/runtime/adapters.py
PYTHONPATH=python python tools/experiments/run_work_unit_matrix.py
```

The native ABI in `include/nta/RuntimeABI.h` remains the storage contract;
the semantic Python layer adds identity, demand, and availability invariants
without creating a second native ABI.
