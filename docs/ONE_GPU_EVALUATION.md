# One-GPU Evaluation Plan

This document defines the single-GPU evaluation for NTA. It distinguishes
mechanism evidence from end-to-end serving evidence and tests request-aware
incremental execution of an otherwise all-or-nothing GPU operator.

## Research Question

Can a compiler expose request-owned data dependencies, executable contributors,
and exact completion conditions in real FlashInfer kernels, then let one SLO
runtime jointly prioritize external data and GPU computation without a
persistent kernel or an all-resident regression?

The intended result is not "CTA granularity is always faster than layer
granularity." The testable claim is:

> Incremental execution reduces the all-or-nothing kernel barrier under
> heterogeneous data arrival, while request-aware grouping preserves dense
> transfer efficiency and the compiler-generated direct form preserves the
> resident path.

The bounded operator-feasibility gate passes with canonical FlashInfer math:
bounded CPU-DRAM streaming is `1.1714x` faster than atomic promotion at the
64K-context/256-query point with `4x` lower staging HBM, and a separate
heterogeneous context/query point improves by `1.1100x` with `4.83x` lower
staging. Both confidence intervals exclude 1.0. Both runs also verify
dynamic-source graph replay, request-slot generation reuse, and cancellation
isolation. The measured producer order is fixed by the host, so this result
proves useful exact partials and bounded HBM, not the completion-driven
scheduler or an end-to-end serving gain. SGLang's paged decode operator does
not yet consume the generated arrival-driven plan.

The mapped-host GPU-initiated producer is also implemented and exact, but is a
negative ablation (`0.479x` versus atomic copy-engine promotion at the headline
shape). CPU-DRAM performance claims therefore use the copy engine. NVMe must be
measured separately because its queue, latency, and P2P path are physically
different.

## Target Serving Scenario

The primary scenario is heterogeneous batch-barrier amplification in
long-context and agent serving. One admitted attention batch contains requests
with different context lengths, prefix-cache histories, and cancellation
lifetimes. Their KV pages are fragmented across HBM and external tiers; a
single request can span resident HBM, CPU DRAM, and NVMe while another request
is fully resident. Conventional launch boundaries either wait for the slowest
request's complete layer or remove and reform work at request granularity.

NTA instead preserves request identity at the kernel boundary and compiles one
operator into direct and incremental forms. The direct form handles resident
and already-ordered data with a generation guard and no tickets. The
incremental form externalizes only unavailable request/tile work, so finite
kernel launches can consume arrived data without retaining a polling CTA. The
online policy selects between these NTA forms; it never declares a win by
dispatching the NTA arm to an untouched kernel.

The full evaluation must cover all of these dimensions together:

- mixed resident and offloaded requests;
- different context lengths and prefix-cache states;
- fragmented KV placement within requests;
- simultaneous CPU-DRAM and NVMe sources; and
- admission churn, request cancellation, and slot reuse.

The current real SGLang integration validates resident plus CPU-DRAM mixtures.
The VFIO NVMe transport is validated separately at mechanism level. A single
canonical FlashInfer run with simultaneous CPU-DRAM and NVMe pages remains an
open claim gate and must not be inferred from those separate results.

Production execution has no oracle. For end-to-end traces, replay every forced
policy from the same initial state and report B6 against each one and the
best-fixed whole-trace result. Report decision regret only for controlled points
where the identical batch, cache, and queue snapshot is restored before every
alternative:

```text
regret = incremental_scheduler_time / min(valid forced-endpoint times)
```

## What The Related Evaluations Establish

### Strata

Strata evaluates hierarchical KV caching with Llama-3.1-8B,
Qwen2.5-14B-Instruct-1M, and Llama-3.1-70B. Its principal datasets are LooGLE,
NarrativeQA, ReviewMT, and ShareGPT. It preserves conversation dependencies,
uses Poisson arrivals where source timestamps are absent, sweeps request rate,
and reports average TTFT and output-token throughput. Its important ablations
separate GPU-assisted I/O, cache-aware scheduling, locality-preserving
matching, delay-hit handling, and batch balancing. It also evaluates minimum,
shuffled, and maximum reuse distance, page sizes from 32 to 1,024 for the
layer-wise baseline, delayed cache hits from the Mooncake Tool-Agent trace,
and SSD performance.

Strata must not be described simply as "layer-granular I/O." It synchronizes
cache data availability per layer, but retains small logical KV pages and implements
the transfer path as a separate GPU kernel whose threads move 128-byte chunks.
It uses a few large CTAs to confine I/O to a small number of SMs and bypasses
cache where possible. NTA targets partial execution inside the compiled
operator, not a smaller memory instruction size.

### Prism

Prism addresses cross-model GPU allocation with elastic memory ballooning. It
uses 32 H100 GPUs and production Hyperbolic and Arena-Chat traces spanning 58
models. Its evaluation calibrates per-model TTFT and TPOT SLOs from dedicated
GPU baselines, sweeps offered load, SLO scale, and GPU count, and separately
measures memory-sharing, placement, local scheduling, activation latency, and
elastic-memory overhead.

Prism is not a direct baseline for NTA. Its useful methodology here is
dedicated-baseline SLO calibration, bursty contention traces, load scaling,
and an all-resident/no-op overhead test.

### Sarathi-Serve And Llumnix

Sarathi-Serve evaluates four model/deployment configurations, two workload
families, maximum sustainable load under strict and relaxed tail-latency SLOs,
and the isolated overhead and contribution of each scheduling mechanism.
Llumnix uses 16 GPUs, ShareGPT and BurstGPT length distributions, generated
long-tail mixes, Poisson and Gamma arrivals, and explicit burstiness sweeps; it
reports mean and tail behavior, migration interference, priority, and cost.

Their methodological requirement for NTA is a capacity curve under an SLO, not
one latency point: sweep arrival rate through saturation, include burstiness and
length heterogeneity, report p50/p95/p99 TTFT and TPOT plus goodput, and isolate
incremental execution from grouping and admission in ablations.

### InfiniGen And ECHO

InfiniGen evaluates multiple model architectures and sizes, UVM and explicit
offload environments, batch/sequence sensitivity, wall-clock performance, and
accuracy/perplexity against selective-KV and quantization baselines. ECHO uses
real sparse-attention serving on an eight-GPU system, compares vLLM and SGLang,
and separates graph-friendly cache management, numerical prefetch, and fused
kernel overlap.

Their methodological requirement is that avoided-byte results include model
quality, candidate fraction, context length, and an equal-accuracy or lossless
comparison. Sparse selectivity cannot stand in for the dense fragmented-KV
claim, and a custom kernel cannot stand in for canonical FlashInfer.

### ServerlessLLM, Parrot, And Prism Production Replay

ServerlessLLM combines storage/load microbenchmarks, component breakdowns, and
end-to-end model-start scenarios across model sizes. Parrot starts from a
measured semantic gap between application workflows and request-level serving,
then evaluates complete multi-request applications rather than only its
metadata abstraction. Prism validates production deployment with shadow replay
of the same online workload.

Their methodological requirement is a chain of evidence: first quantify the
semantic gap, then show the mechanism in isolation, then replay the same
requests through the complete serving system. NTA must archive identical
request order, cache state, placement, and output hashes for every arm.

### Combined Evaluation Rule

No single related paper defines the matrix. The OSDI-level method used here is:

1. Diagnose the phenomenon from real or faithfully replayed traces before
   selecting a favorable mechanism point.
2. Compare against structurally different alternatives: layer wait, coalesced
   bulk, request skip/rebatch, compiler direct form, and forced incremental.
3. Sweep offered load, burstiness, context length, HBM pressure, page
   fragmentation, tier latency, and request mix.
4. Report end-to-end SLO goodput and tail latency separately from bandwidth,
   launch tax, SM occupancy, useful partial work, and physical bytes.
5. Include no-op, dense all-miss, sensitivity, ablation, failure, and quality
   cases, with randomized repeated trials and confidence intervals.
6. Require mechanism-active accounting: every proposed-system attention launch
   names its compiler form, every contributor has a request generation and
   completion state, and zero stock fallback is asserted rather than inferred.

Primary sources:

- Strata: <https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang>
- Prism: <https://www.usenix.org/conference/osdi26/presentation/yu-shan>
- Sarathi-Serve: <https://www.usenix.org/conference/osdi24/presentation/agrawal>
- Llumnix: <https://www.usenix.org/conference/osdi24/presentation/sun-biao>
- InfiniGen: <https://www.usenix.org/conference/osdi24/presentation/lee>
- ECHO: <https://www.usenix.org/conference/osdi26/presentation/liu-guangda>
- ServerlessLLM: <https://www.usenix.org/conference/osdi24/presentation/fu>
- Parrot: <https://www.usenix.org/conference/osdi24/presentation/lin-chaofan>

## Testbed

The current single-GPU host is:

| Resource | Configuration |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, SM 12.0 |
| Driver | 595.84 |
| CPU | Intel Xeon 6787P, 86 physical cores |
| DRAM | 251 GiB |
| NVMe | Lexar ARES 4 TB, Crucial P3 4 TB, Dell PM1733a 3.84 TB, Dell CD8P 1.92 TB |

Use the enterprise PM1733a for the primary NVMe result after checking its PCIe
topology and moving only that function to VFIO. Keep the system and dataset
volumes on other devices. Record the exact PCIe link width/speed, IOMMU mode,
GPU/SSD topology, firmware, CPU governor, GPU clocks, and thermal state.

## Models And Datasets

### Models

Use one production-sized dense model as the primary result and one smaller
model for rapid sweeps:

| Role | Model | Reason |
| --- | --- | --- |
| Primary | Qwen2.5-14B-Instruct | Matches a Strata model, is ungated, and fits in 98 GiB with a controlled KV budget |
| Reproduction | Llama-3.1-8B-Instruct | Matches Strata if access is available |
| Fast sweep | Locally cached Qwen3-8B | Shortens the full parameter sweep |
| MoE generality | Locally cached Qwen3-30B-A3B | Exercises device-resident expert routing after the attention study |

Use FP16/BF16 consistently across variants. Fix the HBM allocation available
to model weights and KV cache instead of allowing the large local GPU to hide
the nonresident path.

### Dataset Priority

1. **LooGLE Wikipedia** is the primary long-context/context-sharing workload.
   Reuse Strata's organization by document and preserve all questions for a
   selected document. The official dataset is available at
   <https://github.com/bigai-nlco/LooGLE>.
2. **NarrativeQA** supplies longer documents and many queries per document.
   Use the official release at <https://github.com/google-deepmind/narrativeqa>.
3. **ReviewMT** supplies multi-turn, long-context agent-like reuse. Use the
   official release at <https://github.com/chengtan9907/ReviewMT>.
4. **ShareGPT** is the short-context/no-regression workload. It is already in
   the local Hugging Face cache as
   `anon8231489123/ShareGPT_Vicuna_unfiltered`.
5. **Mooncake Tool-Agent** supplies delayed-hit and real agent timing patterns.
   Use the trace from the Mooncake repository rather than synthesizing delayed
   resolution from ordinary QA data.

Run two scales:

| Scale | LooGLE | NarrativeQA | ReviewMT | ShareGPT |
| --- | ---: | ---: | ---: | ---: |
| Pilot | 20 documents / at least 200 queries | 20 documents | 20 contexts | 2,000 requests |
| Paper | 105 documents / 2,410 queries | 50 documents / about 1,461 queries | 100 contexts / about 1,092 queries | at least 20,000 requests |

The paper scale follows Strata's published context counts where possible. The
preprocessor must emit token counts, document/conversation identity, turn
dependencies, arrival time, and a content hash. Bucket long contexts at 8K,
16K, 32K, 64K, and 128K tokens; never silently truncate a sample into a
different bucket.

## Baselines

All baselines must use the same model, attention backend, HBM budget, KV page
size, request order, cache contents, and admission limit.

| ID | Variant | Purpose |
| --- | --- | --- |
| B0 | Untouched SGLang + FlashInfer, all resident | Numerical and resident-path control |
| B1 | Compiler-generated direct FlashInfer form, all resident | Transformation no-op cost |
| B2 | Layer-wise `cudaMemcpyAsync` before attention | Conventional all-or-nothing baseline |
| B3 | Coalesced separate GPU I/O kernel, then complete attention | Strata-style mechanism baseline, clearly labeled as a reimplementation |
| B4 | Whole-request skip and rebatch using the same arrival estimates | Engine scheduling baseline |
| B5 | Forced fine-grained incremental execution | Mechanism endpoint |
| B6 | Unified request-aware incremental scheduler | Proposed complete system |
| B7 | Best fixed whole-trace B1-B5 | End-to-end hindsight reference |

Do not label B3 as Strata or claim that NTA beats Strata unless Strata's actual
artifact is run under the same engine and hardware. B2-B6 must share the same
cache policy, initial cache contents, request trace, and admission limit so that
execution policy, not cache hit rate or workload drift, explains the result.
B7 is computed from complete forced-mode trace replays; it is neither an
executable policy nor a per-decision oracle. A separate resettable decision
oracle is permitted only in controlled E1/E2 points with identical snapshots.

## Experiment Matrix

### E0: Correctness And Resident Cost

Run B0 and B1 for decode, paged prefill, split-K decode, and chunked prefill.
Sweep batch size 1, 8, 32, and 64 and context length 1K, 8K, 32K, and 64K.
Check output against untouched FlashInfer and report kernel latency, TTFT, TPOT,
registers, shared memory, achieved occupancy, and graph compatibility.

Claim gate: median all-resident end-to-end overhead at most 5%, with the 95%
confidence interval reported. A larger overhead is a result to fix, not hide.

### E1: Transfer Efficiency

Run B2-B7 for mapped CPU DRAM, staged CPU DRAM, and NVMe. Sweep:

- transfer object: 4, 16, 64, 256, and 1,024 KiB;
- total outstanding demand: 1 MiB through 8 GiB;
- resident fraction: 0%, 25%, 50%, 75%, and 100%;
- required fraction of cataloged objects: 12.5%, 25%, 50%, and 100%;
- duplicate fan-in: 1, 2, 4, and 8 CTAs per object; and
- homogeneous, per-request clustered, and randomly mixed placement.

Report physical bytes, useful bytes, I/O commands, coalesced requests, queue
depth, p50/p95/p99 acquisition latency, achieved bandwidth, GPU SM tax, and L2
traffic. Define useful-byte efficiency as:

```text
useful_byte_efficiency = bytes consumed by completed work / bytes moved into HBM
overfetch_ratio        = bytes moved into HBM / bytes consumed by completed work
```

### E2: Incremental-Execution Crossover

This is the central mechanism and policy experiment. Use identical real
FlashInfer attention work and compare B2-B7 in four regimes:

1. dense homogeneous all-miss, where every tile is required;
2. mixed residency within one request and within one batch;
3. mixed I/O latency, including HBM + DRAM and HBM + DRAM + NVMe; and
4. GPU-selected sparse demand, first with a controlled tile mask and then with
   FlashInfer GPU top-k page-table transformation feeding real sparse attention.

Sweep 1, 4, 16, and 64 requests; 4 through 256 pages per request; and KV page
sizes of 1, 16, and 32 tokens where supported. Report time until the first
useful partial, first request, 50% of requests, and all requests complete. Also
report blocked CTAs, runnable CTAs per finite round, empty runnable-work
launches, and time spent in progress kernels.

For each geometry, replay at least four arrival orders: homogeneous, one delayed
request, interleaved tiers within every request, and an adversarial order that
delays the lowest-index contributor. Deterministic output must be independent of
arrival order. Compare three online scores from the same current state: CTA
count only, bytes plus transport time, and ABI-v25 critical work (data service
plus pending/runnable compute). This isolates whether compiler-attributed cost
and the request bridge improve decisions beyond trivial ordering.

Fine-grained incremental execution wins only when avoided transfer and earlier
useful compute exceed its command, bookkeeping, and resumption cost. For a
baseline transfer of `B_layer` bytes and an incremental transfer of `B_inc`
bytes, the expected
boundary is:

```text
(B_layer / BW_layer) - hidden_layer
  > (B_inc / BW_inc) + incremental_overhead - hidden_incremental
```

Plot this crossover rather than reporting only the favorable region.
For B6, additionally report acquisition-group size, issue time, runnable-tile
count, predicted versus observed request delay, and errors grouped by regime.
Report median/p95 decision regret only for resettable points. The incremental
result is not valid if it wins by changing
the input request order or initial cache state relative to the forced modes;
policy-caused state changes during a trace are part of the measured system and
must be reported rather than treated as counterfactual oracle evidence.

### E3: Long-Context Serving

Run LooGLE, NarrativeQA, and ReviewMT with B0-B7. For datasets without
timestamps, generate Poisson arrivals with three fixed seeds. Sweep offered
load from 20% of saturation through overload. Preserve turn dependencies and
evaluate three cache-distance orders:

- minimum: queries sharing a context are adjacent;
- shuffled: context groups are randomly interleaved; and
- maximum: queries for a context are spaced as evenly as possible.

Report request throughput, output-token throughput, p50/p95/p99 TTFT, TPOT,
end-to-end latency, recomputed tokens, cache hit/delay-hit rate, bytes by tier,
CPU utilization, SM tax, and HBM footprint. Plot throughput versus TTFT rather
than selecting one request rate.

### E4: Bursty SLO Contention

Adopt Prism's methodology without claiming its multi-model result. Calibrate
TTFT and TPOT SLOs from B0's dedicated p95 values, then test 1x, 2x, and 5x SLO
scales. Mix two request classes on the same model:

- latency-sensitive ShareGPT decode; and
- long-context LooGLE or ReviewMT requests with nonresident KV.

Alternate 30-60 second burst phases in which each class becomes dominant, then
replay the Mooncake timing trace. Report SLO attainment and goodput, not only
mean latency. A separate optional two-model experiment can stress isolation,
but NTA must not claim Prism-style model-memory ballooning.

### E5: Delayed Hits And Coalescing

Replay the Mooncake Tool-Agent trace while varying cache-resolution latency and
timestamp scale. Compare no delay-hit handling, bulk waiting, and NTA
object-generation validation plus duplicate coalescing. Report stale work,
cancelled bytes, duplicate bytes avoided, queue occupancy, TTFT, and SLO
goodput.

## Measurement Protocol

- Use at least 10 independent process-level trials per point and randomize
  variant order with `scripts/run-qualified-trials.py`.
- Use three arrival seeds for trace experiments and report both per-seed and
  aggregate intervals.
- Separate compile/load/warm-up from measurement. Recreate the requested cache
  state before every measured trial.
- Lock GPU clocks when permitted; otherwise record clocks, power, and
  temperature and classify the result as uncontrolled.
- Pin the serving process and host memory allocation to the CPU NUMA node
  nearest the GPU/NVMe path.
- Precondition NVMe, use direct I/O, reserve a disjoint LBA range, verify every
  payload, and report both logical and device bytes.
- Capture Nsight Systems timelines for representative points and Nsight Compute
  counters for resident, bulk-I/O, and incremental kernels.
- Archive raw logs, trial order, configuration, source revision, dirty status,
  model/tokenizer hashes, dataset hashes, and machine metadata.

## Required Figures

1. Throughput versus TTFT for LooGLE and NarrativeQA.
2. SLO goodput versus offered load for the bursty two-class trace.
3. Useful-byte efficiency and overfetch versus required-tile fraction.
4. Incremental-execution crossover heat map and resettable decision regret over
   resident fraction, required fraction, and I/O/compute ratio.
5. Time-to-first-partial/50%/complete CDF under heterogeneous data arrival.
6. Dense all-miss bandwidth and all-resident overhead.
7. Ablation of request semantics, CTA-count versus critical-work scoring, engine
   progress feedback, elastic grouping, CTA direct issue, and incremental
   execution.
8. Nsight timeline showing direct CTAs computing while blocked CTAs have exited
   and finite progress handles external data.

## Claim Gates

The stronger efficiency claim is supported only if all of the following hold:

1. B6 stays within 5% median and 10% p95 decision regret on the declared
   training-independent resettable crossover sweep, and matches or beats the
   best-fixed B7 confidence interval on real traces.
2. B5 beats B2-B4 at equal cache hit rate and equal HBM budget in at least a
   dense mixed-residency regime and the real FlashInfer GPU-selected regime.
3. The B6 gain remains in end-to-end SGLang TTFT/SLO goodput, not only in the
   standalone numerical kernel.
4. Physical bytes, exposed stall, or useful partial work explain the gain; a
   changed admission policy, request order, or cache hit rate does not.
5. Dense all-miss B6 performance is within 10% of B3, and all-resident B1/B6
   overhead stays within 5% of B0.
6. The result includes confidence intervals and at least one real NVMe path.

Passing these gates supports "request-aware incremental execution reduces the
all-or-nothing kernel barrier while preserving dense transfer efficiency across
the measured regimes." It does not support "always faster" or "faster than
Strata" without its matched artifact.

## Current Implementation Status

Runnable now:

- numerical paged attention with one CTA per physical KV page;
- query-dependent sparse attention with device-produced queries, same-CTA
  selection/acquisition, permuted request-slot bindings, CPU-reference
  validation, and a matched overlapped all-page GPU overfetch control;
- resident, mapped-DRAM, staged-DRAM, and mixed placement;
- global-load and TMA consumers;
- real FlashInfer resident/deferred correctness and resident-hook latency;
- real FlashInfer split-K decode with canonical request/tile coordinates,
  request-local merge gates, one preloaded contributor wave, and one remaining
  demand wave;
- real FlashInfer GPU top-k page selection feeding a stable device index table,
  bounded pinned-host gather, compact paged decode, matched candidate
  overfetch, and a precomputed selected-copy oracle;
- GPU-initiated VFIO NVMe mechanism and preliminary hardware trials; and
- GPU-hidden-state top-k MoE routing with device-built dependencies plus
  matched `late-bound` (legacy CLI name), `cpu-sync`, and `overfetch` controls;
  and
- randomized process-level trial collection with provenance.

These establish mechanism feasibility. The custom sparse-attention and MoE
programs are not primary performance evidence.

Matched real SGLang/Qwen2.5-3B whole-prefix CPU-DRAM runs remain negative
evidence. Earlier 4K and saturated 8K transformed-direct tests delivered
`0.929x` and `0.904x` stock throughput and executed no ticketed partial work.
The new fragmented tests do: every external layer uses real FlashInfer split-K,
the first next-layer wave overlaps post-attention compute, and the remaining
wave is acquired through tickets. With zero stock launch/fallback and identical
output, the 8K test delivered `0.863x` and the 16K test delivered `0.927x`.
The narrowing gap is consistent with transfer becoming more exposed, but one
sample per point establishes neither a crossover nor a serving gain.

The physically compact 16K rerun reduced resumed application CTA positions to
`50.98%` of the canonical grids but initially delivered only `0.806x`; the
cause was full suspended-ticket initialization for work that was already
available. Lightweight exact-once publication removed that accidental work.
The next matched run compacted the combined initial and resume grids to
`50.01%`, matched stock output, used zero fallback/stock launches, and delivered
`0.935x` throughput (`121.043 ms` versus `113.150 ms`). This is still a `6.98%`
latency cost and therefore negative evidence. It rejects empty-CTA elimination
as a sufficient optimization and motivates the coalesced request-level overlap
form for dense batches.

The coalesced load fixture now establishes cross-request execution but not a
performance win. SGLang mixed-chunk batching plus NTA's acquisition admission
hook formed one real FlashInfer schedule with 17 direct and 16 external work
items at the 2K point. All 36 layers used ticketed incremental attention and the
new bounded parallel indexed-progress path; the combined initial/resume CTA
bound was 50.0% of two full grids. Output matched and stock/fallback counters
were zero, but throughput was only `0.953x`, resident TPOT was `1.055x`, and
external TTFT was `1.230x` stock. A four-resident 4K point reached only `0.921x`.
These rows validate the batch-barrier experiment itself and reject the current
eager per-layer control implementation as the paper result.

The ABI-v23 v10 acquisition-frontier rerun corrected the causal metric. Since
external arrivals are gated on the resident's first token, resident TTFT cannot
measure external interference. The harness now records P99 inter-token latency.
The current immediate-mixed point is 0.977x throughput, 1.012x resident P99
inter-token latency, and 1.021x external TTFT. A five-trial policy that delayed
the external request and re-formed the transformed mixed batch measured 0.972x
throughput, 1.048x resident P99 inter-token latency, and 1.295x external TTFT;
it was removed. Dense admission shifting is therefore a rejected ablation, not
the primary result.

ABI v25 exports blocked-byte, pending-compute, executable-compute,
completed-compute, and expected-compute summaries plus observable dropped
attribution. The device
transport queue recomputes deadline urgency from live queue delay, transfer
service, and compiler-attributed request compute on insertion and requeue. The
device transport path consumes this state directly: insertion and requeue map
live critical service to urgency buckets, and the GPU test verifies that it
changes service order for equal-priority requests. The Python
`CriticalWorkPlan` is a reference/control-plane model, not the hot-path
scheduler. SGLang batch admission still does not consume progress snapshots,
so this remains mechanism status rather than an E3/E4 result.

The implemented five-point operator sweep captures the required crossover: a
forced indexed path loses at zero byte avoidance, while the online cost model
keeps bulk there and selects indexed transfer at 75% or greater avoidance. The
corrected peak is 8.1731x over forced candidate overfetch. At zero avoidance,
forced indexed acquisition is 0.6431x while the online policy selects bulk and
is exactly 1.0000x. See `VALIDATION.md` for the exact claim boundary and
resident-candidate upper bound.

Still required before E3/E4 are paper evidence:

- demand-mode graph/device control that removes per-layer Python launch and
  event construction while preserving compiler-generated direct/incremental
  forms;

- repeated workloads with measured per-tile arrival skew where transformed
  incremental FlashInfer beats transformed direct, whole-layer wait, and skip/rebatch under
  the same kernel math, cache state, and request order; forced ticketing is
  currently slower on the all-known CPU-DRAM fixture;
- one canonical FlashInfer batch containing resident, CPU-DRAM, and NVMe pages,
  including cancellation and request-slot reuse while I/O is outstanding;
- end-to-end serving use of the implemented GPU-selected FlashInfer path with
  model-generated scores and quality evaluation;
- trace preprocessing and an arrival-driven serving client with per-request
  TTFT/TPOT JSON output;
- physical-byte, queue, SM-tax, per-ticket availability, and partial-progress
  telemetry; and
- a clean revision plus controlled repeated trials.

A preliminary local mechanism sweep on this machine used 32 requests, 577 KV
pages, and 100 graph iterations. It measured approximately 134 GiB/s resident,
36.7 GiB/s mapped DRAM, 7.8 GiB/s staged DRAM, and 18.3 GiB/s mixed with global
loads. TMA was equal or slower in this point. These values establish neither a
layer-wise comparison nor a serving claim.
