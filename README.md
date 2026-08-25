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
- [docs/ARTIFACT_EVALUATION.md](/home/jcsu/Dev/sched-pass/docs/ARTIFACT_EVALUATION.md):
  source boundaries and reproducible artifact profiles.

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
typed LLVM-lowered consumer
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

The native tier contract covers HBM, mapped host memory, host-staged memory,
NVMe, and CXL DAX. `nta_runtime_get_tier_descriptor` and
`Runtime.tier_descriptor()` expose the same capability and ownership metadata
used by device admission and experiments. CXL qualification is explicit via
`nta-cxl-dax-probe`; without a qualified endpoint the CXL backend stays
inactive.

The NVMe production target is a direct READ DMA into HBM, not host staging.
`runtime/host/` owns VFIO/IOMMUFD administration, `kernel/nta_nvme_p2p/` is the
narrow NVIDIA peer-page mapper for the attached translated domain, and the
instrumented GPU owns application SQ publication and CQ progress. The
host-mapped destination remains an explicit baseline and cannot satisfy the
direct-HBM artifact gate. See `docs/NVME_SECURITY.md` for the threat model and
transactional qualification procedure.

JIT modules carry a typed operator contract. The LLVM pass validates the
compiled contract and lowers typed acquisition markers; structural discovery
of raw pointer cones remains diagnostic and never authorizes an unmarked
kernel.

SGLang is the first serving implementation. It keeps host HiCache ownership,
request identity, demand, and availability at explicit boundaries while
building an execution session for every real FlashInfer attention launch.
The vLLM adapter exposes the same request projection without importing vLLM.
Logical multi-tenant admission is part of that shared contract: an aligned
tenant annotation is carried with each request binding, and optional startup
quotas (`NTA_TENANT_BUDGETS=id:bytes[:weight],...`) bound device staging rather
than adding a framework-specific scheduler.

Retired selector-specific serving and graph-specialization paths were removed
from the current implementation. Their old source is recoverable through Git
history; it is not an alternative runtime interface.

## Build and tests

```bash
cmake -S . -B build -GNinja \
  -DLLVM_DIR="${LLVM_DIR:-/usr/lib/llvm-22/lib/cmake/llvm}"
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

GPU- and framework-dependent tests remain capability-gated. The dependency-free
contract tests can run with:

```bash
PYTHONPATH=python python tests/runtime/work_unit.py
PYTHONPATH=python python tests/runtime/execution_core.py
PYTHONPATH=python python tests/runtime/adapters.py
PYTHONPATH=python python experiments/run_work_unit_matrix.py
PYTHONPATH=python python experiments/run_work_unit_matrix.py \
  --ablation all --output /tmp/nta-matrix.json
python experiments/validate_matrix_artifact.py \
  /tmp/nta-matrix.json --require-all-ablations
```

The dependency-free matrix labels its timing as modeled regime data. Only the
SGLang serving harness reports GPU/engine timing; the matrix is the executable
fairness, activation, tier, granularity, and Little's-law contract.

For artifact evaluation, use `experiments/reproduce.py`; it records the exact
commands, environment, revision, machine metadata, and raw logs in a separate
artifact directory.

Serving artifacts are independently checked by
[`experiments/validate_serving_report.py`](/home/jcsu/Dev/sched-pass/experiments/validate_serving_report.py);
comparison results with divergent stock/NTA outputs cannot pass the artifact
gate.

The OSDI evaluation contract, Bailian workload preparation, arrival provenance,
three-tier qualification rules, profiling, and regression checks are in
[`docs/OSDI_EVALUATION.md`](/home/jcsu/Dev/sched-pass/docs/OSDI_EVALUATION.md).
The machine-readable contract is
[`experiments/evaluation-manifest.json`](/home/jcsu/Dev/sched-pass/experiments/evaluation-manifest.json).
NVMe and DAX trials additionally require the machine-checked physical-tier
artifact produced by
[`experiments/qualify_tiers.py`](/home/jcsu/Dev/sched-pass/experiments/qualify_tiers.py);
missing hardware is a skip, never a passing result.

The native ABI in `include/nta/RuntimeABI.h` remains the storage contract;
the semantic Python layer adds identity, demand, and availability invariants
without creating a second native ABI.

The sparse selector in `benchmarks/attention/PagedAttention.cpp` is an
isolated exact/overfetch diagnostic baseline for transport experiments; it is
not part of the serving headline or an approximate-attention claim.

## Getting started: artifact smoke test

This is the short kick-the-tires path for an artifact evaluator. It needs only
the checked-out source, Python, a matching LLVM package, and (for the native
test profile) the CUDA toolchain available on the host:

```bash
python experiments/reproduce.py \
  --profile core \
  --output /tmp/nta-artifact-core
python experiments/validate_bundle.py /tmp/nta-artifact-core
```

The output is outside the checkout and contains the exact commands, logs,
revision, source digest, machine metadata, and the validated B0--B6 contract
matrix. The matrix timing is modeled contract data, not serving evidence.

## Detailed artifact evaluation

The complete reproduction map is in
[`experiments/README.md`](/home/jcsu/Dev/sched-pass/experiments/README.md) and
[`docs/ARTIFACT_EVALUATION.md`](/home/jcsu/Dev/sched-pass/docs/ARTIFACT_EVALUATION.md).
The sequence is: normalize and validate the Bailian structure trace, qualify
the physical tiers, run paired exact-demand trials, validate every report,
then profile and compare against a machine-specific baseline. Missing NVMe or
DAX hardware is recorded as `SKIP`; it is never converted into a performance
claim.
