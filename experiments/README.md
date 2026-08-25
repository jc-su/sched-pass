# Experiment and artifact map

`experiments/` is the only canonical orchestration layer. It owns workload
normalization, trial specifications, artifact provenance, result validation,
profiling, and statistical analysis. It imports the implementation under test;
it does not define a second runtime state machine or native ABI.

## Canonical pipeline

| Question | Entrypoints | Output gate |
| --- | --- | --- |
| RQ0 workload opportunity | `prepare_bailian.py`, `validate_workload.py`, `analyze_workload.py` | structure/demand/arrival digests and explicit arrival provenance |
| Physical tier capability | `qualify_tiers.py`, `validate_tier_qualification.py` | exact HBM/host/NVMe/DAX qualification; missing hardware is skip |
| Hardware preflight | `inspect_hardware.py` | read-only GPU/NVMe/DAX capability inventory; never binds devices |
| RQ1--RQ3 paired execution | `run_evaluation.py`, `analyze_evaluation.py` | exact demand, paired metadata, six strata, causal comparisons, bootstrap CI, Little's Law |
| RQ4 cost and regression | `profile.py`, `check_regression.py`, `validate_performance_artifact.py` | complete profiler + baseline + measured report + passing regression gate |
| Reproduction packaging | `reproduce.py`, `validate_bundle.py` | self-contained external bundle with command and digest provenance |

Fetch the public Bailian objects without cloning Git LFS and verify their
immutable object digests before normalization:

```bash
python experiments/fetch_bailian.py \
  --output-dir /path/outside/checkout/qwen-bailian-usagetraces-anon
```

The source data stays outside Git.  The resulting normalized manifest records
the source filename and SHA-256, while the paired experiment bundle copies
only the normalized records that it actually replays.

For a bounded single-host serving trial, select a deterministic source prefix;
the manifest records both the selected count and the full source count:

```bash
python experiments/prepare_bailian.py \
  --input /path/to/qwen_traceA_blksz_16.jsonl \
  --arrival-mode trace --state-policy root_resident \
  --max-requests 1024 \
  --manifest /tmp/traceA-1024/manifest.json \
  --records /tmp/traceA-1024/records.jsonl
```

The machine-readable contracts are `evaluation-manifest.json`,
`tier-qualification.schema.json`, and `artifact-manifest.json`. A trial that
does not satisfy these contracts is rejected before its timing can be used.

Generate a complete paired B0--B6 specification from concrete commands with:

```bash
python experiments/make_evaluation_spec.py \
  --workload-manifest /path/to/workload/manifest.json \
  --tier host_mem --output /tmp/paired-evaluation.json \
  --arm-command 'B0=...' --arm-command 'B1=...' --arm-command 'B2=...' \
  --arm-command 'B3=...' --arm-command 'B4=...' --arm-command 'B5=...' \
  --arm-command 'B6=...'
```

The generator expands the declared strata and the adjacent causal pairs plus
the two decisive cross-boundary pairs (`B3` vs `B1` for host-control
round-trip isolation and `B5` vs `B3` for the complete mechanism jump). It
requires a concrete command for every arm and validates the resulting spec
before writing it; no missing arm is silently treated as a baseline.
Generated specifications carry `evaluation_profile=osdi-complete`. The runner
requires all B0--B6 arms, every canonical causal boundary in every declared
stratum, and at least six strata for that profile. The checked-in example spec
is marked `evaluation_profile=contract` because it is only a minimal API
fixture; it must not be used as an OSDI result.

For machine capability provenance, run the read-only artifact profile:

```bash
python experiments/reproduce.py --profile hardware \
  --output /tmp/nta-artifacts/hardware
python experiments/validate_bundle.py /tmp/nta-artifacts/hardware
```

## Framework boundary

`benchmarks/serving/CompareSglangHiCacheLoad.py` is the canonical SGLang
serving comparison driver. It invokes the stock and NTA arms on the same
normalized workload and emits the structured report checked by
`validate_serving_report.py`. `SglangHiCacheLoad.py` is its single-backend
worker.

The remaining files under `benchmarks/serving/` are specialized diagnostics
or operator-level studies used by native/FlashInfer tests. They are not
alternate runtime interfaces and are not part of the OSDI headline unless a
trial specification explicitly promotes them with the same report contract.
The implementation remains in `include/`, `lib/`, `runtime/`, and
`python/nta_runtime/`; benchmark code is never imported by the serving
runtime.

## Artifact rules

- Use a fresh directory outside the checkout for every run.
- Keep anonymized source data outside Git; bundle only the normalized manifest
  and records needed to replay a trial.
- Use `trace` only when timestamps are present. Offline order is not labeled
  production arrival; use `batch_release` or calibrated open-loop mode.
- Bailian rows describe conversation structure, not this machine's HBM state.
  Use `--state-policy root_resident` to construct the mixed resident/follow-up
  serving setup explicitly; the manifest records that the state labels are
  synthetic.
- Require exact demand and correctness digests for every compared arm.
- Require a real tier qualification artifact before any NVMe/DAX trial.
- Treat modeled matrix timing, missing profilers, and unavailable hardware as
  contract/status evidence, never as serving speedup evidence.
- An `osdi-complete` artifact must include a `performance/` directory with
  `profile.json`, `baseline.json`, `measured.json`, and `regression.json`.
  Build it outside the checkout and pass it with
  `--performance-evidence`; the bundle validator rejects a missing or
  unavailable profiler instead of silently downgrading the claim.
