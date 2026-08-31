# OSDI evaluation protocol

This project has one mechanism claim: exact, late-bound heterogeneous work
units let a GPU execute the runnable subset of a mixed batch while preserving
request identity, tier ownership, and exact demand.  The compiler pass and the
runtime are one co-designed implementation of that mechanism.  They are not
separate scheduling policies, and the evaluation does not use approximate
top-k/DSA attention as evidence for the system claim.

The executable contract is
[`experiments/evaluation-manifest.json`](../experiments/evaluation-manifest.json).
Validate it before collecting results:

```bash
python experiments/validate_evaluation.py
```

## Workload construction

The public Bailian/ServeGen exports are anonymized.  Online traces with
timestamps are replayable as relative arrival traces; offline traces preserve
lengths, block hashes, and order but do not contain an arrival process.  The
normalizer therefore has three explicit modes:

The four public Qwen-Bailian objects are Git-LFS files.  Fetch and verify them
without placing the raw data in the checkout:

```bash
python experiments/fetch_bailian.py \
  --output-dir /path/outside/checkout/qwen-bailian-usagetraces-anon
```

The fetch helper checks the published object size and SHA-256.  Use the
verified file path as `--input`; the normalized manifest then records the
source digest and every paired serving arm receives the same demand digest.

```bash
# Online trace: preserve relative timestamp burstiness.
python experiments/prepare_bailian.py \
  --input online.jsonl --arrival-mode trace \
  --manifest online.manifest.json --records online.records.jsonl

# Offline trace: characterize cache topology and release as one batch.
python experiments/prepare_bailian.py \
  --input offline.jsonl --arrival-mode batch_release \
  --synthesize-prompts \
  --manifest offline.manifest.json --records offline.records.jsonl

# Offline trace: use a separately supplied timestamped reference to calibrate
# an open-loop rate; this is a controlled synthetic arrival, not production data.
python experiments/prepare_bailian.py \
  --input offline.jsonl --arrival-mode calibrated_open_loop \
  --arrival-reference online.jsonl --target-rate 12 \
  --manifest calibrated.manifest.json --records calibrated.records.jsonl
```

`hash_ids` are interpreted as 16-token block identities.  Prompt synthesis
creates deterministic token IDs and text placeholders that preserve shared
prefix topology and input length.  It explicitly records
`semantic_representativeness_claim: false`; generated prompt text is an input
shape vehicle, not evidence about language quality or production semantics.

The public Bailian rows do not label the device-cache state of each request.
For a mixed serving trial, derive that setup explicitly from conversation
structure instead of treating row order as a cache observation:

```bash
python experiments/prepare_bailian.py \
  --input qwen_traceA_blksz_16.jsonl --arrival-mode trace \
  --time-scale 0.25 --state-policy root_resident --synthesize-prompts \
  --manifest traceA.manifest.json --records traceA.records.jsonl
```

Use `--max-requests N` for a bounded single-host replay.  This is a
deterministic source-prefix selection, not a change to the demand semantics;
the manifest records the selected count and the full source count, while the
source file digest remains that of the complete public object.

`root_resident` warms root/turn-zero requests into the device tier and routes
follow-up turns through the external tier.  The manifest marks this as a
synthetic serving setup, not a production-cache-state claim.  A paper result
must report both the source arrival provenance and this state-construction
policy.  If a trace has no timestamps, use `batch_release` or an explicitly
calibrated open-loop rate; never use offline row order as production arrival.

Generate the RQ0 opportunity artifact separately from serving trials:

```bash
python experiments/analyze_workload.py online.manifest.json \
  --output /tmp/nta-artifacts/online-rq0.json
```

This records prefix reuse, length, state heterogeneity, and arrival-burst
statistics together with the same workload and demand digests used by paired
serving arms.

## RQs and evidence

RQ0 characterizes the opportunity: prefix reuse, request-state heterogeneity,
length distributions, burstiness, and exact candidate-block shape.  The
anonymized trace explicitly reports compute/transfer regime as
`trace_only_not_identifiable`; that regime is measured in native tier reports
and serving profiles rather than inferred from prompt lengths.  RQ1 is the
paired serving result against stock SGLang HiCache under identical exact
demand. RQ2 is the causal decomposition across framework bulk control, exact
eager NTA preacquisition, scheduler-bound whole-layer acquisition, and
progressive heterogeneous work-unit consumption.
RQ3 reports request-state, tier, load, arrival, and granularity strata.  RQ4
reports control and resource overhead with profiler artifacts.

For serving, a resident-only forward is an explicit framework-reference
control and is not counted as an NTA-transformed launch. A mixed forward is the
mechanism case: resident and external work coexist in one NTA launch, with
external attention required to be exact and fallback-free. This split prevents
instrumentation overhead on requests that do not exercise a remote tier while
keeping the mixed-batch mechanism measurable.

The paired plan has three adjacent boundaries: A1/A0 for exact acquisition,
A2/A1 for scheduler-bound ownership, and A3/A2 for progressive work-unit
consumption. All three pairs consume the same exact demand and numerical
contract; A2 and A3 additionally share the same acquisition owner and differ
only in consumer readiness.

The generated trial specification is explicitly marked
`evaluation_profile=osdi-complete`. This profile is a machine-checked gate,
not a documentation label: it requires all A0--A3 arms, all three canonical
causal boundaries in every declared stratum, and at least six strata. Minimal
fixtures use `evaluation_profile=contract` and cannot be presented as the
complete evaluation.

Every serving report must include p50/p95/p99 TTFT/TPOT/ITL percentiles, SLO goodput,
throughput, queue/admission delay, selected and physical KV accounting (or an
explicit `not_exposed` status), exact-demand identity, activation counters,
output correctness, and machine metadata. Native tier reports must provide
byte-accurate useful/physical bytes; the SGLang engine report currently exposes
exact cached-token counts and explicitly records why byte geometry is not
available through its public result metadata. The serving harness also passes
normalized request IDs through to the engine request `rid` and records them,
so the demand digest is not merely an offline annotation.
Synthetic matrix timing is contract evidence only.  A speedup claim requires a
structured serving artifact with the same demand trace and correctness digest.

The headline serving verdict is not TTFT alone.  Formal load reports use one
fixed joint request SLO: TTFT <= 8 s, TPOT <= 50 ms, and request p99 ITL <=
100 ms, with exact token timestamps required.  Goodput is the number of
requests satisfying all three conditions divided by measured elapsed time.
The older TTFT+p99-ITL metric remains in artifacts under its explicit legacy
name; it is never relabeled as the joint metric.  Repeated controlled trials
also gate resident p95 TPOT and p99 ITL at no more than 1.05x stock, resident
and total output-token throughput at no less than 0.95x stock, exact output,
fallback freedom, and at least 100 external-request observations per arm.

An overloaded rate must be selected without looking at NTA.  First collect
stock-only reports at at least three offered rates, with at least 100 requests
and exact token timing in every report.  Freeze the first stable multi-signal
knee before running either paired arm:

For a normalized workload, keep one rate-bearing manifest and pass
`--scale-workload-arrivals-to-request-rate` at each pilot rate.  The harness
uniformly time-dilates the recorded offsets, preserves request order and exact
demand, and records the source rate, target rate, and scale.  It rejects a
CLI/manifest rate mismatch without that explicit transform.

```bash
python experiments/freeze_serving_overload_rate.py \
  --input 8=/path/stock-rate-8.json \
  --input 12=/path/stock-rate-12.json \
  --input 18=/path/stock-rate-18.json \
  --output /path/frozen-overload-rate.json
```

The frozen artifact records all input hashes, the stock revision, workload,
machine, SLO contract, and deterministic selection rule.  A rate chosen after
observing NTA is diagnostic only and cannot support the SLO-goodput claim.

The target serving tiers are HBM reference, host-staged memory, VFIO NVMe, and
CXL DAX. The current serving path names host memory explicitly as
`host_staged`; it does not silently switch between mapped and staged host
access. `host_mapped` is retained as a matched NVMe DMA-destination baseline,
not as another serving tier. NVMe requires the explicit read-only VFIO/IOMMU qualification;
its admission gate proves exact GPU-controlled reads directly into HBM through
the NVIDIA peer-page backend, translated-IOMMU containment, no new target DMAR
fault, and zero outstanding work. A host-mapped destination is a baseline and
cannot satisfy direct-HBM admission. The matched fio ratio remains a separate
performance result, and its Boolean classification must agree with the recorded
ratio and threshold. DAX requires an explicit CUDA-visible `devdax` mapping
and a numerical consumer that can use the mapped resource. That SGLang
numerical boundary is currently deferred, so DAX is not yet a completed
serving tier. Missing or deferred capability is a `SKIP`, never a fabricated
result.

The native tier commands use the same paged-attention workload and correctness
checks:

```bash
# Host-memory reference/variant.
build/nta-paged-attention --mode=host-staged --json=1 ...

# A pre-qualified VFIO controller and a read-only reference range.
build/nta-paged-attention --mode=nvme \
  --nvme-endpoint=vfio:DDDD:BB:SS.F \
  --nvme-reference=/path/to/preloaded/reference.bin --json=1 ...

# A live devdax endpoint; regular files are rejected by the runtime.
build/nta-paged-attention --mode=dax \
  --cxl-endpoint=/dev/daxX.Y --cxl-window-mib=1024 --json=1 ...
```

Run `scripts/run-nvme-qualification.py --allow-device-rebind` only after
reviewing its read-only preflight and confirming that the controller is an
isolated experiment device; this is the only path that is allowed to rebind a
PCI controller. Then run
`build/nta-cxl-dax-probe ... --json=1` first.  Assemble those reports with the
HBM and host-memory native JSON reports into one admission artifact:

```bash
python experiments/qualify_tiers.py \
  --hbm-report /path/to/hbm.stdout \
  --host-mem-report /path/to/host.stdout \
  --nvme-report /path/to/nvme-qualification.json \
  --dax-report /path/to/dax.stdout \
  --output /tmp/nta-artifacts/tier-qualification.json
python experiments/validate_tier_qualification.py \
  /tmp/nta-artifacts/tier-qualification.json
```

An evaluation specification containing an NVMe or DAX arm must set
`"tier_qualification"` to this artifact.  The runner refuses a missing,
skipped, failed, or non-exact qualification; it never turns a capability skip
into a data point. An NVMe arm must also use the queue-depth recommendation for
the same transfer granularity recorded by qualification; changing either is a
new service point and requires requalification. The evaluation reproduction
profile copies the normalized
workload and the qualification artifact into its bundle, rewrites the spec to
those copies, and validates both digests.

## Statistical and profiling rules

Warmups are excluded, arm order is randomized in paired blocks, and the
preferred protocol uses ten repetitions (five is the minimum).  Results are
reported per stratum before any aggregate.  Use the existing qualified-trial
runner for paired commands, and store its raw logs outside the checkout.  For
hardware profiling:

For a complete OSDI trial matrix, generate a specification with
`experiments/make_evaluation_spec.py` from six or more normalized scenarios,
then run it from a clean revision:

```bash
python experiments/run_evaluation.py \
  --spec /path/to/paired-evaluation.json \
  --output-dir /tmp/nta-artifacts/paired-serving
```

The wrapper reuses the randomized qualified-trial engine and adds exact
tier/stratum/demand metadata to every record. It then emits three checked
reports: `evaluation-report.json` (canonical), `strata-report.json`, and
`causal-report.json`. Validate the completed bundle before copying it into an
artifact archive:

```bash
python experiments/validate_evaluation_artifact.py /tmp/nta-artifacts/paired-serving
```

The serving report uses descriptive finite-window arrival/departure accounting
from measured client timestamps. Its queue scope is explicitly client
admission delay; it is not an internal scheduler queue claim unless the engine
exports that queue timestamp.

```bash
python experiments/profile.py --output /tmp/nta-profile --tool auto -- \
  /path/to/one/qualified/trial
```

`nsys`, `ncu`, and `perf` are selected only when installed and the exact
profiler command is recorded.  A missing profiler is reported as unavailable,
not silently replaced with a wall-clock claim.  Capture a machine-specific
baseline only after correctness passes; compare it with
`experiments/check_regression.py`.  The baseline schema intentionally contains
no repository-wide “golden” timing number because shared GPU clocks and
co-tenants make that number non-portable.

The methodology follows the evidence discipline used by tiered-memory and
hierarchical-cache systems: evaluate the actual tier, expose overlap and
transfer costs, report tail behavior and interference, and separate mechanism
ablations from end-to-end serving results.  See [Strata](https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang),
[the OSDI 2024 CXL tiering study](https://www.usenix.org/conference/osdi24/presentation/zhong-yuhong),
and [the OSDI 2025 beyond-hotness study](https://www.usenix.org/conference/osdi25/presentation/liu).
