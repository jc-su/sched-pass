# One-GPU Evaluation Plan

This document defines the single-GPU evaluation for NTA. It distinguishes
mechanism evidence from end-to-end serving evidence and tests request-aware
incremental execution of an otherwise all-or-nothing GPU operator.

## Research Question

Can a compiler turn a real batched attention kernel into an incrementally
executable operator, then let the runtime jointly group arriving data, launch
runnable request/tile work, and inform later batch admission without a
persistent kernel or an all-resident regression?

The intended result is not "CTA granularity is always faster than layer
granularity." The testable claim is:

> Incremental execution reduces the all-or-nothing kernel barrier under
> heterogeneous data arrival, while request-aware grouping preserves dense
> transfer efficiency and the compiler-generated direct form preserves the
> resident path.

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

Primary sources:

- Strata: <https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang>
- Prism: <https://www.usenix.org/conference/osdi26/presentation/yu-shan>

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
7. Ablation of request semantics, engine progress feedback, elastic grouping,
   CTA direct issue, and incremental execution.
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

The implemented five-point operator sweep captures the required crossover: a
forced indexed path loses at zero byte avoidance, while the online cost model
keeps bulk there and selects indexed transfer at 75% or greater avoidance. The
current peak is 8.826x over forced candidate overfetch; see `VALIDATION.md` for
the exact claim boundary.

Still required before E3/E4 are paper evidence:

- demand-mode SGLang execution of the request/tile ticket path under a
  production dense mixed-residency workload; the current integrated fast path
  is preacquired and does not exercise unavailable-data work tickets;
- compiler-generated direct and incremental FlashInfer forms plus the unified
  grouping scheduler under the same kernel math and cache policy;
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
