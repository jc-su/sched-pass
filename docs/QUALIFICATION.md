# Release Qualification

The repository has three evidence levels. They are deliberately separate:

- `local`: implementation correctness and bounded-mechanism qualification on
  the current host;
- `production`: a clean revision plus real serving, soak, fault-recovery, and
  portability evidence; and
- `osdi`: production evidence plus matched baselines, ablations, controlled
  trials, a device-generated-demand workload, and clean-host artifact
  reproduction.

Current verdict: the local mechanism suite passes on the one-GPU host, while
`production` and `osdi` remain `NOT_READY`. The fixed-arrival-order canonical
FlashInfer bounded-HBM gate passes with a stable 1.1714x speedup and 4x staging
reduction. The corrected paired GPU-selected FlashInfer policy chooses the
transformed bulk arm at zero byte avoidance, where forced indexed acquisition
would deliver only `0.6431x` throughput. It reaches `8.1731x` over forced
candidate overfetch at 93.75% avoided bytes, with a peak 95% interval of
`[8.1639x, 8.1762x]` and same-trial maximum decision regret of `1.0000x`.
The all-candidate-resident control is 9.15x or more faster than cold indexed
acquisition, exposing the remaining acquisition cost. A warm-cache resident SGLang graph smoke is within the
direct-path bound, while the latest ticketed external SGLang measurements
remain regressions. The positive operator results are compiler-transformed
with validated typed plans, but they do not establish an end-to-end serving
gain. Completing evidence paperwork alone cannot change the release verdict.

In the latest three arm-balanced Qwen2.5-3B graph trials, the fully transformed
mixed host/resident path achieved `0.9447x` output-throughput geometric mean
with bootstrap interval `[0.9337x, 0.9578x]`. External TTFT was `1.3386x` and
resident P99 inter-token latency was `1.4991x` stock; SLO goodput was only
`0.7498x`. Three trials are diagnostic, not paper-level statistical evidence.
This rejects the dense
early-known-demand performance hypothesis; it is not folded into the positive
device-selected acquisition claim.

The ABI-v25 finite demand graph closes the earlier eager-control gap but does
not reverse that dense boundary. On clean revision `ae7c56a`, one
performance-excluded-warmup 2K host/2K resident run replayed the demand graph,
used both compiler forms, compacted the combined physical CTA bound to 50%, and
reported zero stock/fallback launches with exact output. It delivered `0.9169x`
output throughput, `1.1014x` external TTFT, `1.8843x` resident P99 ITL, and
`0.4584x` stock-derived SLO goodput. This is one diagnostic process trial, not
paper evidence; it proves graph activation and preserves the negative result.

The latest `local` qualifier returned `READY` after 47 CTests (one multi-GPU
case skipped on the one-GPU host), CUDA sanitizer coverage, a 10,000-epoch
lifecycle stress, CPU-only tests, Clang static analysis, Python package checks,
ShellCheck, and patch hygiene. This is an implementation-quality result, not a
production or paper-evidence verdict.

Validate the local performance-mechanism artifact independently with:

```bash
python scripts/validate-tier-streaming-results.py \
  --headline results/serving/tier-streaming-compiler-headline.json \
  --heterogeneous results/serving/tier-streaming-compiler-heterogeneous.json \
  --require-compiler-transform \
  --output results/serving/tier-streaming-compiler-qualification.json
```

This checks canonical FlashInfer execution, numerical parity, at least ten
headline trials, a paired-bootstrap interval above 1.0, at least 1.15x effect
size, at least 4x staging reduction, per-request completion, heterogeneous
request shapes, explicit compiler-path classification, dynamic-source graph
replay, generation reuse, and cancellation isolation. Passing it supports only
the mechanism phrase defined in
`TIER_STREAMING.md`; it cannot return `production` or `osdi` `READY`.

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
static analysis, Ruff, a no-isolation Python wheel build, and ShellCheck.
`--skip-local` accepts a prior report only for the same clean Git revision.

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

./scripts/run-qualified-trials.py \
  --spec experiments/tier-streaming-compiler.json \
  --output-dir results/osdi/tier-streaming-compiler
```

These checked-in mechanism specifications do not constitute the full OSDI
matrix. Site-specific serving, storage, portability, and fault-injection
specifications must name the qualified models and hardware explicitly. Their
reducers populate the production and OSDI evidence reports.

The runner refuses a dirty worktree by default, verifies an optional exact
revision, randomizes repetition order and variant order within complete trial
blocks, preserves every raw log, and emits per-variant 95% confidence
summaries. `--allow-dirty` exists only for runner development and cannot
satisfy the clean-revision release gate.

## Production Evidence

`results/qualification/production-evidence.json` uses schema 3 and records the
exact qualified `revision` with `dirty=false`. Earlier schemas are rejected
because they could qualify a path that dispatched stock attention after
transfer instead of exercising the compiler/runtime mechanism.

The `serving` object must establish all of the following:

- a real SGLang or vLLM integration with `mechanism_mode` equal to
  `request_aware_dual_form`, zero fallback, matched cache/admission state,
  stock-output correctness, separately verified transfers, complete
  attention-layer execution, generation-safe bounded-HBM request completion,
  warm JIT caches, validated versioned compiler contracts with at least one
  direct/incremental pair, positive transformed-direct and ticketed launch
  counts, and zero stock attention launches in the NTA arm;
- whole-model decode replay, paged-prefill integration, and positive finite
  demand-operator graph capture/replay counters;
- no more than 5% p50 resident overhead and at least 90% of matched dense bulk
  throughput;
- simultaneous real `host_staged` and `nvme` serving paths with cancellation
  and request-slot reuse;
- at least two models, three traces, and a 24-hour soak; and
- TTFT, TPOT, P99 inter-token latency, SLO attainment, goodput, CPU
  utilization, and SM utilization.

The `reliability` object still requires NVMe status, malformed-CQE, timeout,
controller-reset, IOMMU-fault, and process-crash injection with zero leaks and
bounded recovery. `portability` requires two machines, two GPU models, two NVMe
models, and a physical multi-GPU run.

## OSDI Evidence

`results/qualification/osdi-evidence.json` also uses schema 3 and must identify
the same clean revision.

The executable gate now matches `SYSTEM_PLAN.md` and requires:

- GPU-timestamped dense opportunity traces from at least two models over both
  CPU DRAM and NVMe, with material measured barrier cost;
- same-source direct and incremental compiler forms for at least two generated
  kernel families, convergence validation, acquired-edge/request/work-ticket
  identity proofs, exactly-once publication, versioned operator contracts,
  runtime-ABI validation, matching source fingerprints, and differential
  correctness;
- real FlashInfer decode and paged-prefill execution where useful partials run
  before the last arrival, output matches stock, and demand mode replays in
  finite decode and paged-prefill CUDA operator graphs, with every NTA attention
  launch accounted to a transformed form,
  zero stock launches, generation-safe request completion, at least 4x staging
  reduction, at least 1.15x speedup over atomic promotion, and a 95% speedup
  interval whose lower bound exceeds 1.0;
- one long-context/agent workload that jointly exercises mixed resident and
  external requests, heterogeneous context/prefix state, fragmented KV,
  simultaneous CPU DRAM and NVMe, admission churn, cancellation, and slot
  reuse;
- one elastic online policy with engine admission feedback and identical-state
  decision regret no worse than 1.05 median and 1.10 p95;
- ABI-v25 request accounting in every mechanism-active run, with pending,
  executable, completed, and expected compiler-attributed compute conserved;
  dropped attribution must remain zero, and the device critical-work policy
  must beat or match CTA-count-only and byte-only ablations without using
  future arrivals;
- GPU-selected pages consumed by a real FlashInfer selector and paired
  compiler-transformed attention forms with no NTA hot-path host identity
  round trip, at least five selectivity points, a measured crossover, a peak
  confidence interval above one, at least 2x peak speedup over forced
  overfetch, an explicit forced-indexed zero-avoidance result, an
  all-candidate-resident baseline, at most 1.05x same-trial online-policy
  regret, and at most 2x regret to a precomputed selected-copy oracle;
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

Both evidence files are reducer outputs. They do not replace raw logs,
per-request timestamps, device telemetry, or fault timelines, which must be
published with a paper artifact. Local SHA-256 fields were deliberately removed:
a report and hash created beside one another provide no independent trust. Git
revision, clean-tree state, exact commands, raw samples, and independent
artifact review provide the useful provenance.

This cleanup does not remove hashes used by the running system. Operator source
and plan fingerprints reject mismatched direct/incremental kernels; the JIT
cache key prevents reuse across ABI or source changes; the FlashInfer overlay
validator refuses to patch an unknown upstream tree; and output digests compare
complete generated responses compactly. Those uses protect correctness rather
than decorate evidence files.

RNIC/RDMA is outside the scoped claim and therefore is not a release
requirement. The paper must describe the system as local HBM/CPU-DRAM/NVMe
acquisition and must not claim a functioning network backend.

## Claim Rule

Use "production-ready" only when the production qualifier returns `READY` for
the exact clean revision being released. Use "OSDI-validated" only when the
OSDI qualifier returns `READY` and the linked raw evidence has been reviewed.
Local qualification supports only the phrase "locally validated mechanism."
