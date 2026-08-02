# Release Qualification

The repository has three evidence levels. They are deliberately separate:

- `local`: implementation correctness and bounded-mechanism qualification on
  the current host;
- `production`: a clean revision plus real serving, soak, fault-recovery, and
  portability evidence; and
- `osdi`: production evidence plus matched baselines, ablations, controlled
  trials, a device-generated-demand workload, and clean-host artifact
  reproduction.

Run the complete local gate and write its JSON report:

```bash
./scripts/qualify-release.py --profile=local
```

After collecting external evidence, evaluate the stronger claims without
rerunning an already successful local gate:

```bash
./scripts/qualify-release.py --profile=production --skip-local
./scripts/qualify-release.py --profile=osdi --skip-local
```

Reports are written under `results/qualification/`. A non-ready verdict exits
with status 2. CI and release automation must use the exit status; the existence
of a report does not imply that the profile passed.

The local profile configures both CUDA and CUDA-disabled builds from source,
runs all CTests and sanitizers, executes 10,000 lifecycle epochs, runs Clang
static analysis, Ruff, a no-isolation Python wheel build, and ShellCheck, and
fingerprints the exact tracked and untracked workspace. `--skip-local` accepts
a prior report only when that fingerprint still matches.

The stock serving environment can be checked independently with:

```bash
./benchmarks/serving/SglangSmoke.py \
  --model /path/to/local/model --requests 4 --max-new-tokens 8
```

This emits `nta_integrated=false`, so it cannot satisfy the production
serving-integration check. The installed plugin path is exercised with:

```bash
python benchmarks/serving/CompareSglangHiCache.py \
  --model /path/to/local/model --iterations 10 \
  --hot-tokens 160 --resident-tokens 96 --churn-tokens 320 \
  --max-total-tokens 384 --context-length 512 \
  --cuda-graph-decode full --verify-transfer
```

That comparison is a local integration gate. Production qualification still
requires the multi-model, multi-trace, 24-hour evidence below.

Matched benchmark, serving, baseline, and ablation commands should be run from
a schema-1 trial specification with:

```bash
./scripts/run-qualified-trials.py \
  --spec experiments/sparse-attention-late-bound.json \
  --output-dir results/osdi/sparse-attention

./scripts/run-qualified-trials.py \
  --spec experiments/moe-late-bound.json \
  --output-dir results/osdi/device-routed-moe
```

These checked-in mechanism specifications do not constitute the full OSDI
matrix. Site-specific serving, storage, portability, and fault-injection
specifications must name the qualified models and hardware explicitly; the raw
outputs from all specifications are referenced by the evidence manifests.

The runner refuses a dirty worktree by default, verifies an optional exact
revision, randomizes repetition order and variant order within complete trial
blocks, preserves every raw log with its digest, and
emits per-variant 95% confidence summaries. `--allow-dirty` exists only for
runner development and cannot satisfy the clean-revision release gate.

## Production Evidence

`results/qualification/production-evidence.json` uses schema 2. Schema 1 is
rejected because it could qualify the old preacquired path without exercising
incremental demand execution.

The `artifacts` manifest must include `serving`, `correctness`, `reliability`,
and `portability` classes. Each entry contains a repository-relative `path` and
its `sha256`. Every JSON or JSONL record must carry the exact qualified
`revision`.

The `serving` object must establish all of the following:

- a real SGLang or vLLM integration with `mechanism_mode` equal to
  `incremental_demand`, zero fallback, matched cache/admission state, stock
  output correctness, separately verified transfers, complete attention-layer
  execution, and zero post-acquisition instrumented launches;
- demand-mode decode CUDA graph replay and paged-prefill integration;
- no more than 5% p50 resident overhead and at least 90% of matched dense bulk
  throughput;
- real `host_staged` and `nvme` serving paths;
- at least two models, three traces, and a 24-hour soak; and
- TTFT, TPOT, SLO attainment, goodput, CPU utilization, and SM utilization.

The `reliability` object still requires NVMe status, malformed-CQE, timeout,
controller-reset, IOMMU-fault, and process-crash injection with zero leaks and
bounded recovery. `portability` requires two machines, two GPU models, two NVMe
models, and a physical multi-GPU run.

## OSDI Evidence

`results/qualification/osdi-evidence.json` also uses schema 2. Its artifact
manifest must include `opportunity`, `dense_flashinfer`, `sparse_flashinfer`,
`baselines`, `ablations`, `statistics`, and `reproduction`.

The executable gate now matches `SYSTEM_PLAN.md` and requires:

- GPU-timestamped dense opportunity traces from at least two models over both
  CPU DRAM and NVMe, with material measured barrier cost;
- same-source direct and incremental compiler forms for at least two generated
  kernel families, convergence validation, and differential correctness;
- real FlashInfer decode and paged-prefill execution where useful partials run
  before the last arrival, output matches stock, and demand mode replays in a
  CUDA graph;
- one elastic online policy with engine admission feedback and identical-state
  decision regret no worse than 1.05 median and 1.10 p95;
- GPU-selected pages consumed by a real FlashInfer selector and attention
  kernel with no NTA hot-path host identity round trip, at least five
  selectivity points, a measured crossover, at least 2x peak speedup over
  forced overfetch, and at most 2x regret to a precomputed selected-copy oracle;
- no more than 5% p50 resident overhead, at least 90% of dense bulk throughput,
  and an end-to-end gain over equal-state request skip/rebatch;
- at least ten controlled independent trials with confidence intervals; and
- clean-host reproduction plus published raw results.

Required baselines are untouched FlashInfer, the compiler direct form,
layer-complete prefetch, coalesced bulk, request skip/rebatch, forced fine
incremental execution, the best fixed hindsight policy, CPU completion, and a
persistent progress comparison. Required ablations cover request semantics,
the runnable tile set, incremental kernel form, complete-contributor merge,
elastic grouping, replica selection, engine progress feedback, and CTA
try-issue.

Both evidence files summarize raw artifacts; booleans do not replace logs,
per-request timestamps, device telemetry, fault timelines, or checksums. A
clean worktree is checked separately.

RNIC/RDMA is outside the scoped claim and therefore is not a release
requirement. The paper must describe the system as local HBM/CPU-DRAM/NVMe
acquisition and must not claim a functioning network backend.

## Claim Rule

Use "production-ready" only when the production qualifier returns `READY` for
the exact clean revision being released. Use "OSDI-validated" only when the
OSDI qualifier returns `READY` and the linked raw evidence has been reviewed.
Local qualification supports only the phrase "locally validated mechanism."
