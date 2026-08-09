# Request-Aware Tier Streaming

Status: locally qualified compiler/runtime mechanism

## Problem

A tiered serving batch is heterogeneous at request granularity. Requests have
different generations, tenants, priorities, deadlines, query/context lengths,
resident KV fractions, external sources, and cancellation times. Conventional
attention integration either promotes every external KV byte before attention
or removes whole requests from the batch. Both choices erase useful structure:
exact attention is a mergeable fold over request-owned KV ranges.

NTA keeps the relation below intact:

```text
(request ID, generation, policy)
    -> exact KV range and physical tier
    -> transfer group and bounded HBM slot
    -> FlashInfer (V, LSE) partial
    -> request-local merge and completion
```

The complete system target is a compiler-generated dual-form operator:

- `direct`: consume already resident data through the unchanged optimized
  mainloop;
- `stream`: double-buffer external ranges through bounded HBM, execute exact
  partial attention as each range arrives, and merge only current-generation
  contributors; and
- `bulk`: the largest legal `stream` group, not a dispatch to stock attention.

The engine may choose a group size, but every form retains NTA request identity
and every measured NTA arm must account for all attention launches. There is no
performance claim based on bypassing the compiler/runtime path.

## Implemented Mechanism

`python/nta_runtime/tier_streaming.py` implements the engine-neutral finite
planner, and `python/nta_runtime/flashinfer_tier_streaming.py` implements the
reusable double-buffer executor. Together they:

- keys work by `(request_id, generation)`;
- removes cancelled generations before forming work;
- carries tenant, priority, and deadline metadata into request-owned segments;
- coalesces one segment from each active request into efficient transfer waves;
- bounds staging by `slot_count * maximum_wave_tokens`;
- selects direct, bulk, or streaming execution from calibrated transfer,
  attention, launch, and HBM-capacity inputs without a future oracle;
- own copy-slot lifetime, transfer/compute event ordering, canonical
  FlashInfer partial launches, online-softmax merges, and generation-keyed
  completion publication outside benchmark code; and
- capture transfer and compute streams together, then rebind stable pinned-host
  wave buffers between graph replays without capture-illegal synchronization.

The FlashInfer JIT frontend emits paired direct and incremental modules from the
canonical ragged-prefill template. Each module exports a versioned operator
contract, and both export the same typed execution plan: request-contiguous
coordinates, online-softmax `(V, LSE)` partial state, ordered merge, fixed
capacity, graph-stable addresses, external wave sources, generation binding,
and complete-contributor merge. The runtime refuses a mismatched pair before
launch. The LLVM pass remains the convergence and marker-legality backend.

The current canonical benchmark uses FlashInfer 0.6.12
`BatchPrefillWithRaggedKVCacheWrapper(return_lse=True)` and
`merge_state_in_place`. It contains no handwritten attention kernel, and the
positive streaming arm reports `compiler_transformed_attention=true`. The three
matched arms are:

1. all-HBM canonical FlashInfer, used as the numerical and physical lower
   bound;
2. atomic promotion, which overlaps one full CPU-DRAM copy with resident/local
   attention and then computes one external partial; and
3. bounded double-buffer streaming, which overlaps request-owned transfer
   waves with real FlashInfer partials and records per-request completion.

The output is exact attention over all resident, external, and local causal KV;
streaming does not drop tokens or approximate softmax.

## Current One-GPU Result

Hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition. Software: FlashInfer
0.6.12, PyTorch 2.11.0+cu130, and CUDA 13.0 runtime.

| Workload | Atomic promotion | Bounded stream | Speedup | 95% CI | HBM staging reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 requests, 256 query tokens, 64K context, resident fractions 0/25/50/100% | 6539.1 us | 5582.2 us | 1.1714x | [1.1660x, 1.1732x] | 4.0x |
| query tokens 128/256/384/512, contexts 64K/48K/32K/16K, resident fractions 0/25/50/100% | 5008.2 us | 4512.0 us | 1.1100x | [1.1050x, 1.1109x] | 4.83x |

Both runs pass differential output checks against one complete canonical
FlashInfer call. Graph validation additionally changes every external pinned
source after capture and compares replay with eager execution, preventing a
compute-only capture from passing. Lifecycle validation reuses a request slot
with a new generation, cancels that generation, verifies unaffected request
outputs, and reuses the slot again through the same captured graph. The
headline uses 10 arm-rotated trials and a deterministic 10,000-resample paired
bootstrap. The resident-only request completes at 1.06 ms while external
requests complete at 4.32, 5.22, and 5.72 ms. This is
observable per-request progress rather than a batch-wide prefetch result.

The compiler/runtime operator also supports a GPU-initiated CPU-DRAM producer:
finite progress kernels copy registered mapped-host rows into the same bounded
slots, with no CPU-issued per-wave copies. At the headline shape it is exact and
passes dynamic-source graph replay, generation reuse, and cancellation, but it
measures 13.669 ms versus 6.550 ms for atomic copy-engine promotion (`0.479x`).
Mapped-host GPU loads plus per-wave progress kernels are the wrong physical path
for known CPU-DRAM demand on this machine. This is a transport ablation, not a
failure of the direct/stream operator: CPU DRAM uses CUDA's copy engine while
device-discovered or queue-backed sources may use GPU initiation.

The full crossover sweep is equally important. Small query chunks and small
groups can lose to atomic promotion because partial launches and merges are not
free. The policy therefore treats grouping as an online resource/performance
decision. When full promotion exceeds the HBM budget, bounded streaming remains
the feasible exact form even if its latency is slightly higher; when both fit,
the calibrated prediction must clear a minimum gain before selecting streaming.

The checked-in `experiments/tier-streaming-compiler.json` specification now
contains both rows, three matched arms per row, ten randomized complete blocks,
and the graph/lifecycle checks. Its older generated output predates the current
ABI and aggregate completion protocol and is not evidence for these numbers.
The ABI-v24 development run also completed all 60 fresh processes: the median
paired speedups are `1.1672x` for the long-context row and `1.1164x` for the
heterogeneous row. Because the worktree was dirty and clocks were uncontrolled,
those process-level results establish stability only.

Reproduce and validate the local result with:

```bash
./tools/jit/activate.py --build-dir build \
  --cache-root results/serving/compiler-tier-cache --flashinfer-hook -- \
  python benchmarks/serving/FlashInferTierStreaming.py \
  --query-tokens 256 --context-tokens 65536 --group-tokens 6144 \
  --warmup 3 --iterations 5 --trials 10 --verify-graph --verify-lifecycle \
  --compiler-transform \
  --output results/serving/tier-streaming-compiler-headline.json

python scripts/validate-tier-streaming-results.py \
  --headline results/serving/tier-streaming-compiler-headline.json \
  --heterogeneous results/serving/tier-streaming-compiler-heterogeneous.json \
  --require-compiler-transform \
  --output results/serving/tier-streaming-compiler-qualification.json

./scripts/run-qualified-trials.py \
  --spec experiments/tier-streaming-compiler.json \
  --output-dir results/osdi/tier-streaming-compiler
```

## Claim Boundary

The local result establishes a real performance opportunity and passes the
repository's bounded-HBM mechanism gate. It does **not** yet establish an
end-to-end serving, production, NVMe-attention, or OSDI claim.

Three missing connections are decisive:

1. SGLang currently integrates the older paged work-ticket path, not this
   bounded-staging numerical executor. Decode and paged prefill must use the
   generated forms in eager and CUDA-graph execution with zero stock fallback.
2. CPU DRAM is the only producer with a positive result through this canonical
   operator. Both copy-engine and GPU-initiated mapped-host producers populate
   the same bounded slots, but the latter is a measured negative. The existing
   VFIO/NVMe transport must populate these slots and be evaluated on qualified
   P2P and staged routes.
3. The evaluation must repeat the same mechanism under real model traces and
   CPU-DRAM plus NVMe arrivals, compare equal-state bulk, layer wait, and
   skip/rebatch baselines, and measure TTFT, TPOT, P99, throughput, goodput,
   HBM footprint, bytes, CPU use, and SM use on clean revisions.

Therefore the accurate current phrase is:

> locally qualified compiler-transformed canonical-FlashInfer bounded-HBM
> mechanism with a stable request-heterogeneous crossover.

Use "OSDI-validated" only after `scripts/qualify-release.py --profile=osdi`
returns `READY` for the exact clean revision and its raw artifact manifest.

## Implementation Order

1. **Implemented:** promote copy-slot, partial-attention, merge, and completion
   execution into an engine-neutral FlashInfer runtime operator.
2. **Implemented for ragged prefill:** emit paired direct and stream forms plus
   a typed request/range/reduction plan; retain LLVM convergence validation as
   the backend proof.
3. Consume the generated operator plan in SGLang HiCache for paged prefill and
   decode, including request cancellation, slot reuse, and graph replay.
4. Feed per-request completed and remaining data/compute into SGLang admission;
   bulk and stream remain two group sizes of the same mechanism.
5. Attach CPU-DRAM and VFIO NVMe producers to the same wave slots, then run the
   matched long-context/agent matrix and all required baselines.
6. Add a second generated kernel family and multi-machine reproduction before
   making a venue-level novelty or performance claim.
