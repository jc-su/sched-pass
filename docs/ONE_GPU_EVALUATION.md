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

## Research Questions And Execution Plan (2026-08-10)

This section is the authoritative campaign plan. The E0-E5 matrix below remains
the detailed sweep inventory; each sweep now serves exactly one of four
research questions, and no experiment runs unless its RQ, gate, and baseline
set are declared here first. The paper carries three insights, each answered by
one or two RQs:

1. Model-generated demand is a first-class runtime data dependency that
   existing systems serve with workload-specific point mechanisms
   (motivation; RQ1, RQ2).
2. Execution should advance only at exact, independently committable
   contributor boundaries; byte arrival and CTA completion are not progress.
   The dense negative series is this insight's boundary evidence (RQ2, RQ3).
3. The contributor commit and suspension contract is compiler-checkable, and
   must be: the post-dominance control-dependence defect was invisible to
   review and caught only by mechanical checking (RQ3).

### RQ1 — Value: does exposing device-discovered demand improve end-to-end serving?

- **1D (headline): model-generated selection.** A training-free Quest-style
  page selector (per-page key summaries scored against the live query,
  device-resident top-k) retrofitted onto a dense GQA checkpoint, tiered KV
  under HBM pressure, served through the installed SGLang plugin. Metrics:
  capacity curves (sustainable request rate at 99% SLO attainment), TTFT/TPOT
  p50/p99, goodput, bytes moved, and task quality deltas against dense
  attention on LongBench-class suites. Baselines: full promotion, overfetch
  candidates, host-side selection with the identity round trip, and
  prediction-based prefetch. Gate: `>=1.5x` goodput at quality parity against
  the strongest baseline.
- **1A: hierarchical long-context capacity.** LooGLE/NarrativeQA/ReviewMT
  arrival sweeps from unloaded to overload; throughput-TTFT curves, not single
  points. Gate: no worse than the strongest layer-wise baseline.
- **1B: heterogeneous agent trace.** Mooncake-style repeated-prefix arrivals,
  mixed resident/DRAM/NVMe placement, cancellation and admission churn,
  thousands of requests. Gate: goodput and wasted-byte improvement under churn.
- **1C: no-regression controls.** All-resident ShareGPT, short-context decode,
  all-resident long-context decode. Gate: direct form within `3%` of stock
  **and resident P99 inter-token latency within `1.05x`**, ten trials.

### RQ2 — Mechanism: when does contributor-committed execution beat waiting, layer pipelining, bulk, or rebatch?

- **2A (runs before any further integration): opportunity characterization.**
  Instrument the default proactive path on real traces with CUDA events:
  per-layer promotion time, attention compute time, blocked-at-barrier time,
  and reusable-partial fraction. Output decides where 2B/1A run and whether
  streaming integration proceeds at all. Implemented:
  `benchmarks/serving/OpportunityCharacterize.py` drives the real
  `SglangHiCacheLoad.py` workload with `NTA_SGLANG_PROFILE_BARRIER`,
  `NTA_SGLANG_PROFILE_GPU`, and `NTA_SGLANG_PROFILE_TRANSFER` enabled, merges
  the per-process device-event counters, fails closed if profiling did not
  engage, and reports per-point blocked fraction, load/compute ratio, and
  per-layer stall against a declared opportunity threshold.
- **2B: operator crossover.** Context 8K-128K, resident fraction 0-100%,
  arrival skew, group size, staging slots. Arms: wait-all, **layer-wise
  pipelined promotion** (the strongest dense baseline; stock HiCache
  approximates it and it must be present in every dense table), coalesced
  bulk, forced fine incremental, skip/rebatch, NTA adaptive, hindsight best.
- **2C: HBM-budget sweep with predicted crossover.** The bytes/bandwidth cost
  model predicts the selectivity and budget crossovers first; the sweep then
  measures them. Gate: prediction within `20%` of measurement, and bounded
  staging sustains contexts atomic promotion cannot fit.

Dense expectations are pre-declared: against layer-wise pipelining the
realistic dense outcome is TTFT parity plus bounded staging and tail
isolation, not a headline speedup. The dense series exists to bound regret and
to evidence insight 2, and five prior dense negatives already delimit it.

### RQ3 — Necessity: is the compiler/runtime/engine co-design required?

One matched trace, one component removed at a time: no incremental form
(bulk-only), no request identity (byte/CTA scheduling), no measured progress
(predicted transfer time only), no reusable partials (discard available work),
no engine feedback (mechanism without admission), manual hand-split operator
(does LLVM generation add value), direct-only. Compiler coverage: paged
decode, paged prefill, and device-routed MoE through the identical contract,
each with stock-output parity, cancellation and generation reuse, graph
replay, and convergence rejection. **Verifier mutation testing:** mutate
kernels across each legality condition, require every mutant rejected, and
demonstrate one representative mutant miscomputing with verification disabled.

### RQ4 — Robustness: does one online policy stay efficient and safe across regimes?

Nonstationary regime-switching trace against every fixed policy and the
hindsight oracle (median regret `<=1.05`, p95 `<=1.10`, reported as *measured
oracle regret*, never as a bound); the transport-geometry matrix (copy engine
versus SM gather versus NVMe across object size and fragmentation — the
`8.17x` fragmented-gather win and the `0.479x` bulk-mover loss are the two
already-measured cells); cancellation storms; NVMe fault injection; 24-hour
soak; request-slot reuse; and the resident-tail interference decomposition.

Interference decomposition, pre-declared hypotheses (2026-08-09): the
qualified ten-trial series measured resident P99 ITL `1.2407x` under
lowest-priority movers, so priority is not dominant. H-A: elevating mover
priority (`NTA_SGLANG_MOVER_STREAM_PRIORITY=-1`, n=10) worsens the tail
measurably; if it does not, stream scheduling is irrelevant to the regression
entirely. H-B: the tail is driven by wave bursts colliding with decode steps —
`NTA_SGLANG_FRONTIER_LAYERS_PER_WAVE=1` (n=10) shrinks the tail at some
external-TTFT cost; if it does not, the cause is per-transfer contention
(PCIe/L2) or host-side planning, pointing to a copy-engine wave path or
CPU-side batching as the fix. Each run records its hypothesis outcome
regardless of direction.

### Models

| Model | Role | Why |
| --- | --- | --- |
| Qwen2.5-14B-Instruct | 1A/1B/1C primary; 1D with Quest retrofit | Strata-comparable dense checkpoint that fits the GPU |
| Llama-3.1-8B-Instruct | reproduction + fast sweeps; 1D second model | community-standard reproduction target |
| Qwen3-30B-A3B | RQ3 MoE family; optional 1B MoE serving point | 128-expert top-8 MoE that fits 96 GiB; experts tier to host DRAM |

Natively sparse production models (DeepSeek V3.2-class) exceed one-GPU memory;
training-free selection retrofitted onto dense checkpoints is both the
deployable form of this workload and the honest local instantiation.
Model-generated scores are mandatory for 1D: the controlled-random-score sweep
remains mechanism evidence only.

Selection quality now has real-model measurements
(`benchmarks/serving/QuestRecall.py`; artifacts
`results/serving/quest-recall-{llama160m,qwen25-3b}.json`). The harness
reconstructs each layer's decode query from the layer's own projections and
rotary embeddings and refuses to score unless the reconstruction reproduces
the model's actual attention row (max observed error 4.1e-6 across 108
layer evaluations). Findings at 2,048 tokens: envelope selection tracks the
true-attention oracle within 2 points on Llama-160M and within ~5 points on
Qwen2.5-3B with sink+recent retention (recall@25% pages: quest 0.807 vs
oracle 0.854) — the selector mechanism is near-oracle, and the binding
constraint is the diffuseness of short-context attention itself (the oracle
also captures only 85% at 25%). The 1D workload must therefore run at long
context (16K+ tokens), where the candidate pool makes equal-mass selectivity
a small fraction; the harness's verified logit-reconstruction path supports
that without materializing attention, and per-KV-head selection remains a
known refinement. This is measured selector evidence, not a quality-parity
claim: 1D's gate still requires end-to-end task-quality evaluation.

The 16K budget table now exists through the certified long-context path
(verification on a materialized 512-token prefix, then reconstruction-only
scoring; fp32 throughout because bfloat16 attention rows legitimately
diverge ~1e-1 from exact and would force a meaningless tolerance). Two
artifacts: `quest-recall-qwen25-3b-16k.json` (documents replicated x64) and
`quest-recall-qwen25-3b-16k-distinct.json` (every repository document once,
rotated per prompt). Replication depresses recall by 2-4.5 points at every
budget — repetitive text spreads attention over near-duplicate keys — so
replicated-prompt recall is a pessimistic bound, and long-context recall
work must use distinct corpora. Distinct-corpus results at 16,384 tokens
(1,024 pages), sink 1 and recent 2 retained: quest 0.717 at a 3.1% budget,
0.762 at 6.2%, 0.813 at 12.5%, 0.875 at 25%, with the oracle at 0.764 /
0.812 / 0.861 / 0.914. Two readings. First, the envelope selector stays
within 4-6 points of perfect selection at every budget; the binding
constraint is the oracle ceiling itself. Second, that ceiling is partly
structural to this checkpoint: two KV heads with group size eight force
sixteen query heads' preferences into one aggregated page ranking — the
harshest head-mixing configuration — and per-KV-head selection (a finer
acquisition unit the indexed machinery's strided rows can express) is the
recorded lever if task quality demands it. Operating decision for 1D: enter
the engine stage at budgets 128 and 256 (87.5% and 75% byte avoidance,
whose acquisition speedups are already measured at 4.12x and 2.13x), and
let the task-quality gate — not recall — select the shipping point.
`SglangSelectedLoad.py` now accepts a same-budget `QuestRecall.py` report and
turns those recall numbers into a hard gate, so a selected-serving speedup is
not publishable unless the quality artifact matches the serving model, context
length, page size, and selected-page budget.

Qwen3-30B-A3B was smoke-verified on this host on 2026-08-09 through stock
SGLang 0.5.14 with full decode CUDA graphs (4 requests, 255.9 output
tokens/s, `mem_fraction_static=0.85`, `nta_integrated=false`). Enabling it
required fixing a harness defect: the smoke benchmark exported the CUDA host
C++ compiler as `CC`, which breaks Triton's C11 launcher builds the moment a
model JITs Triton kernels, while FlashInfer's ninja simultaneously requires a
CUDA-compatible `-ccbin` from the same variable; both are satisfied by the
CUDA-matched C driver (`gcc-14`).

### Regime map: where the mechanism can win at all (2026-08-11)

`benchmarks/serving/RegimeMap.py` (artifact
`results/analysis/regime-map.json`) computes the two quantities that bound
any KV-selection mechanism's payoff — attention-byte share of the decode
step and KV pressure against the pool — from measured constants (HBM
~1.6TB/s ladder, anchored stock TPOT) and the actual model geometries.
The verdict on our own history: the operating point every selected-serving
experiment ran at (Qwen2.5-3B, 2 KV heads = 36KB/token, 16K context,
batch <= 2, 96GB pool) has attention share **3.6%** and available win
**-5.1%** — the mechanism floor exceeds the entire prize, so no
implementation quality could have won there. The recorded losses were the
benchmark's geometry, not the mechanism's verdict; conversely nothing won
there is worth claiming.

The win region is reachable *with the already-integrated model*: at
(64K, batch 16), (32K, batch 32), or (131K, batch 8) the same Qwen2.5-3B
reaches ~70% attention share and ~65% available step win; at (32K,
batch 64) and beyond, dense becomes infeasible outright while selected
serving admits 7-48x more requests — the capacity axis finally exists.
Qwen3-4B/8B (144KB/token) reach the frontier at batch 4-8 (T_base
estimated, pending anchor). `StockDecodeSweep.py` validates the map's
physics empirically before any derived point is used.

Consequences, in order: (1) concurrent per-request claims are the
critical-path engineering — every winning point has batch >> 1; (2) all
future selected-serving evaluation runs only at map-positive points, with
the map's negative region published as the applicability boundary
(small-KV models at small batch are *architecturally* outside the
mechanism's scope, and our measured negatives there document it); (3) the
quality gate re-couples at the map-chosen budget and context, since
keep-rate at 32K/budget-128 is 6.25%.

### Execution order and go/no-go

1. Mover-priority interference rerun (1C metric only) — already unblocked.
2. 2A opportunity characterization on real traces. **Executed 2026-08-09:**
   zero barrier stall at every measured point (2,048-24,576 external tokens,
   360 waits each, load/compute 0.05-0.09); see `docs/VALIDATION.md`. The
   dense-promotion 2A gate therefore fired **no-go**.
3. ~~Streaming-operator integration into the SGLang paged path~~ — cancelled
   by the 2A measurement for dense promotion: there is no blocked time to
   reclaim, and the deployed lookahead pipeline already fully overlaps
   transfer. The streaming operator remains operator-level evidence against
   atomic promotion and a candidate for genuinely bandwidth-bound tiers
   (multi-promotion contention, NVMe) if 2A over those configurations shows
   nonzero stall. 2B proceeds as the controlled crossover study only.
4. Quest-retrofit selector and the 1D workload — **now the campaign
   centerpiece**; then 1A/1B.
5. RQ3 ablations and RQ4 robustness last.

**1D stage 4 first measurement (2026-08-09,
`results/serving/selected-load-16k-b128-first.json`):** the three-arm
harness (`benchmarks/serving/SglangSelectedLoad.py`) measured the v1
tiered path at 16K/budget-128: the dense NTA arm reproduced stock output
exactly at 1.05x external TTFT p95, and tiered served 84.4% avoided
tokens but ran 7.5x slower than dense promotion (0.62s vs 0.08s external
TTFT p95). The regression is orchestration, not transfer, with causes
ranked by inspection of the v1 path: the dual verification wrapper pair
was still planned and run on every serving layer; every layer paid a host
`plan()`; per-layer host synchronization; and no hit-skipping (~33%
measured excess copies). The optimization round in response: single
planned wrapper when verification is off, plan-once-per-forward with
in-place per-layer indices rewrite (sound because FlashInfer's decode
planner consumes only indptr lengths, and retention keeps the tail page
so kept counts are layer-invariant), fixed-shape device-side selection
(no per-layer host sync in the steady path), and hit-skipped staging
(steady selection costs one boolean readback, zero copies). Verification
modes cover both paths: `NTA_SGLANG_SELECTED_TIERED_VERIFY=1` runs the
reference path (dual wrappers, byte verification, device-vs-reference
selection cross-check); `=fast` runs the timed fast path plus per-layer
independent recomputation. The n=10 campaign starts only after the rerun
shows tiered at parity or better against dense promotion.

**Optimization ladder, measured (single trials, tiered vs dense; artifacts
`results/serving/selected-load-16k-b128-{first,decodefast,extendfast,scan,
misspath}.json`):** external TTFT p95 7.50x → 5.52x (decode fast path) →
5.83x (extend fast path — which proved the residual was not extend
orchestration) → 2.18x (streamed envelope scan) → 1.96x (device-side miss
staging, fp16 reductions); resident P99 ITL 8.04x → ~1.0x (parity from the
decode round on); resident TPOT p95 7.74x → 1.01x. The extend round's flat
TTFT localized the true cost: the claim-time envelope build was a 36-layer
CPU gather over 16K rows (~0.4s). The streamed scan moves key bytes once
through an 8MB GPU scratch (~295MB, no HBM capacity held) and is verified
bit-exact against the CPU reference — min/max are order-invariant, so
unlike the attention cross-check this equality is exact by construction.

**Serving-integrity finding (claim supersession):** arrivals outpace
decode (83ms spacing vs ~400ms of decode), so the original single-claim
implementation superseded live requests. Those requests then fell through to
dense attention over a prefix that was ~84% unstaged, silently reading stale
pool rows. A deferred-completion repair was also unsound because SGLang could
recycle the released host rows before the deferred copy. The implementation
now owns concurrent claims by monotonically increasing claim ID, assigns each
claim a generation-tagged fixed object-directory range, and scans the whole
batch so every live external request is compacted in the same FlashInfer plan.
Partial slot-number collisions cannot bind an unowned claim; after first use,
request identity is authoritative. A claim retires only at request finish,
cancellation, or backend shutdown.

The same gate then exposed the deeper invariant: SGLang's host-node
protection is keyed to load completion, and a claim that fakes producer
completion forfeits it at birth — `loading_check` unlocks the host radix
nodes as soon as the finish event fires, and churn write-back recycles
the claim's host rows while it is still staging from them (observed as
staged-vs-host divergence whose staged side held the scan-era bytes).
Stage-time byte verification cannot catch this class — it compares the
copy against the same recycled source — which is why the mechanism, not
a check, had to change: a claimed load now holds its `HiCacheAck` until
`retire()`, keeping the host source lock-pinned for the claim's entire
lifetime. Tiered serving's correctness contract is therefore explicit:
**the claim pins its host source; release and completion are one
atomic scheduler-thread event.** Retirement replaces SGLang's already-fired
producer completion with an event recorded after the final copy or attention
consumer on the actual CUDA stream, so host-row recycling cannot race DMA.

The SGLang adapter now establishes the mechanism needed for an HBM-capacity
result: it intercepts load-back before dense allocation, represents the host
prefix with virtual request-table IDs, and leases only the selected-row budget
from the physical allocator. Request finish retires the claim and returns the
lease after its CUDA completion fence. One 1,022-token Qwen2.5-3B smoke used
256 physical rows and avoided 766 dense slots with allocator conservation.
That single point is a correctness/capacity checkpoint, not an admission,
goodput, latency, or OSDI-level result.

ABI v27 removes two additional bypasses from this prototype. Selected page
identities and the miss count now remain on the GPU through validation,
hit filtering, indexed-list construction, and acquisition; the compact table
is consumed by request-bound compiler-generated FlashInfer wrappers. ABI v27
adds bounded physical page placement: repeated selected pages remain in their
per-layer staging slots and only misses are copied. The 1D harness rejects zero
compiler launches, zero device compaction, zero cache launches or hits, zero
copied rows, a nonselective table, or any fallback. A fused mapped-host summary phase
also eliminates the temporary K staging buffer. A paired 16K-row local
diagnostic measured that phase at 0.185 ms versus 0.174 ms for optimized
copy-and-reduce, so copy-and-reduce remains the default until an uncontended
shape sweep finds a crossover. These are implementation facts, not a new
serving result; the three-arm workload must be rerun on ABI v27.

Go/no-go: if the integrated streaming operator cannot beat the layer-wise arm
at real opportunity points by `>=1.10x` with the direct path within `3%`, and
1D cannot reach its gate, the OSDI serving thesis is not claimable; the work
reframes as a compiler/runtime mechanism paper. A failed gate is recorded, not
retried until it passes.

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
count only, bytes plus transport time, and ABI-v27 critical work (data service
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
- an SGLang external-prefix sidecar that avoids dense KV allocation, retains
  selected pages in bounded per-layer physical slots, copies only misses, and
  consumes physical tables through compiler-generated FlashInfer wrappers;
- a coalesced decode form that executes resident request subgroups while
  external miss-only transfers run on separate CUDA streams, with evidence
  gates requiring positive overlap, cache reuse, zero fallback, and zero stock
  attention;
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

ABI v27 exports blocked-byte, pending-compute, executable-compute,
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
