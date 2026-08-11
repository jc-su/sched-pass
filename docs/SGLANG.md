# SGLang Integration

Status: SGLang 0.5.14 and FlashInfer 0.6.12 are integrated through SGLang's
installed plugin mechanism. The adapter handles eager CPU-DRAM HiCache
promotion and full decode CUDA-graph replay. Preacquired and resident work uses
the compiler-transformed direct form; unresolved multi-round demand uses the
same transformed wrapper with work tickets. Multi-round demand can stage the
first contributor wave for layer `L+1` after layer `L` attention and overlap it
with post-attention model compute; later waves remain ticketed. Stock fallback
is not part of the NTA backend. This is a locally validated integration, not a
production-readiness claim.

## Plugin Boundary

Installing `nta-runtime` publishes this entry point:

```text
sglang.srt.plugins: nta = nta_runtime.plugins.sglang:register
```

SGLang loads it in frontend and spawned scheduler processes. Registration adds
the `nta_flashinfer` attention backend and five lifecycle hooks:

1. `HiCacheController.start_loading`: claim a supported host load before
   SGLang copies it, preserving its page map, completion events, and ACKs.
2. `Scheduler.abort_request`: mirror exact, prefix, or abort-all selection into
   NTA's current request generations before SGLang mutates its queues.
3. `Scheduler.release_host_resources`: atomically flush engine statistics on
   graceful scheduler shutdown.
4. `ForwardBatch.init_new`: attach request priorities to the exact engine batch.
5. `Scheduler._get_new_batch_prefill_raw`: observe bounded acquisition progress
   for external-only admission; an already formed mixed batch is released
   immediately because delaying and re-forming it regressed tail latency.

The plugin also preserves live request IDs and priorities when SGLang 0.5.14
constructs its padded decode-graph replay view. The patch is version-pinned and
fails closed if upstream metadata no longer matches the validated structure;
synthetic padding slots receive distinct non-user request IDs.

Every transformed shared object exports a schema-1 operator contract containing
the runtime ABI, FlashInfer family, direct or incremental form, capability bits,
and a 128-bit source fingerprint. Eager and graph paths validate the contract
before first launch. When both forms execute, SGLang also requires their source
fingerprints to match; module names, counters, or successful symbol loading are
not accepted as compiler evidence by themselves.

Before attention planning, the adapter binds `ForwardBatch.rids` to compact
engine-neutral request slots with monotonically changing generations. Changed
bindings are bulk-published from pinned staging on the current CUDA stream; no
per-request synchronization is performed. At SGLang's HiCache producer point,
the preacquired path enqueues each layer's existing tuned indexed-copy kernel
on a dedicated finite CUDA stream and records a preallocated data-available
event. The consumer stream waits only for the layer it is about to use, then
transformed FlashInfer checks the current request generation before consuming
that layer.

```text
SGLang request and HiCache page map
  -> early layer-wise indexed copy on a finite CUDA stream
  -> preallocated per-layer data-available event
  -> transformed direct FlashInfer request guard
  -> SGLang-owned output tensor and HiCache completion

Unresolved demand
  -> compact request slot/generation binding
  -> ABI-27 guarded incremental CTAs
```

The resident-only preacquired direct path allocates no per-batch `WorkItem`
array, publishes no object directory, and launches no
reset/progress/completion kernels. Preacquired external layers retain a
validated structural work plan so CTA-to-request identity cannot be inferred
from launch shape alone. `NTA_SGLANG_PIPELINE_HOST=0` selects the
schedule-aware policy path: it extracts FlashInfer's request/KV-tile schedule
and evaluates the calibrated bulk-versus-incremental cost model before building
a device plan. A one-round decision uses SGLang's tuned layer transfer and the
transformed direct wrapper without allocating or uploading CTA plan state. A
multi-round decision registers exact indexed K/V objects and runs bounded CTA
acquisition epochs. `NTA_SGLANG_FORCE_INCREMENTAL=1` is an evaluation ablation
that overrides the cost gate. The structural work plan is uploaded once. Later
layers use a GPU directory-rebind kernel instead of rebuilding object
descriptors on the model thread. When multiple rounds are selected and one
FlashInfer wrapper serves the model, the first K/V contributor wave for the
next layer is copied after the current attention kernel and overlaps the MLP.
The next epoch preserves those completed directory entries, executes their
contributors in its initial transformed launch, and progresses only the
remaining waves. `NTA_SGLANG_FRAGMENT_LOOKAHEAD=0` disables this optimization
for ablation. The final model boundary checks the epoch and a monotonic sticky
failure counter; transfer-verification modes retain per-layer checks. All
instrumented paths preserve request-generation checks and the finite-kernel
contract.

No CPU thread copies KV or polls a device queue. CPU userspace still performs
normal batch planning and launches finite CUDA kernels. This distinction is
important: CPU-free data movement does not mean CPU-free kernel submission.

## Supported Profile

The adapter fails at construction for incompatible global settings. Planning
errors fail closed before attention; there is no stock fallback mode.

| Property | Supported |
| --- | --- |
| SGLang | exactly 0.5.14 |
| FlashInfer | exactly 0.6.12, FA2 decode and paged prefill |
| Attention | standard MHA/GQA, no MLA, no logit cap |
| KV dtype | FP16 or BF16 |
| HiCache | pinned host memory, kernel backend, page size 1 |
| Host layout | `page_first` or `layer_first` |
| Speculative decoding | disabled |
| CUDA graphs | full decode replay; prefill disabled in the validated profile |
| Resident-only batch | transformed direct FlashInfer wrapper, no tickets |

Unsupported HiCache ownership, malformed page maps, schedule mismatch, or
capacity exhaustion is recorded before attention and raises. A claimed batch
never silently switches after partially executing attention. `hicache_claimed_batches`,
`hicache_fallback_batches`, and the last fallback reason are written to the
engine statistics file.

Incremental-plan reuse is keyed by exact host/device page-pair tuples,
FlashInfer request/tile schedule, and request slot/generation pairs. One
structural plan is reused across transformer layers; each layer republishes its
K/V addresses through the stream-ordered object directory. Shape equality
alone is not a cache hit: a different page map or generation advances the
object version and uploads a new plan.

## Selected-Demand Diagnostic

`NTA_SGLANG_SELECTED_SERVE=1` and
`NTA_SGLANG_SELECTED_TIERED=1` enable the device-selected integration. Live
queries select logical external-prefix pages; ABI v27 validates and compacts
misses on the GPU, the indexed path acquires their K/V rows, and
compiler-generated request-bound FlashInfer wrappers consume one compact table
for every claimed and resident request. Claims pin host sources until the last
stream-ordered use and bind prefix identity to immutable request-table
positions, not recyclable allocator slot numbers.

The adapter intercepts host load-back before dense device allocation, publishes
virtual request-table identities, and leases only the configured selected-row
budget from SGLang's physical allocator. Cache finish retires the compiler
claim, waits for its completion fence, releases staging, and then lets SGLang
free the physical suffix; allocator conservation is checked by SGLang itself.
Resident batches retain CUDA graph replay. External-prefix batches run the NTA
selected eager form and fail if they reach a dense captured graph. The
selected eager form and fail if they reach a dense captured graph. A per-layer
device page-to-slot table retains selected K/V inside the lease; the phase
emits the physical FlashInfer table and copies misses only. The selected-load
harness uses coalesced batching. Resident peer requests run through a
compiler-generated request subgroup while external misses progress on a
transfer stream; only the external subgroup waits at its consumer edge. The
harness requires positive overlap, cache execution and reuse, activation and
high-water capacity evidence, zero fallback, and zero stock attention.

## Running

Build the native runtime, install the Python package, and run SGLang under the
JIT activator so FlashInfer receives the ABI-fingerprinted source overlay:

```bash
cmake --build build -j
python -m pip install -e .

tools/jit/activate.py --build-dir build --flashinfer-hook -- \
  python your_sglang_program.py
```

A CMake install places the pass, private device headers, shim, overlay builder,
and runtime in one prefix, so the source checkout is not needed at deployment:

```bash
cmake --install build --prefix /opt/nta
/opt/nta/bin/nta-jit-activate --flashinfer-hook -- \
  python your_sglang_program.py
```

The engine configuration must select:

```python
engine = sglang.Engine(
    model_path=model_path,
    attention_backend="nta_flashinfer",
    enable_hierarchical_cache=True,
    hicache_io_backend="kernel",
    hicache_write_policy="write_through",
    hicache_mem_layout="page_first",
    cuda_graph_backend_decode="full",
    cuda_graph_backend_prefill="disabled",
)
```

Structural graph planning delegates to SGLang and FlashInfer before capture.
Replay invokes the real transformed FA2 wrapper with graph-stable runtime
arguments and current request bindings. A pending HiCache transfer is ordered
before graph replay; CUDA
stream-capture isolation forbids capturing a wait on uncaptured producer-stream
work. SGLang's full model graph therefore uses preacquired external data.

Demand mode has a separate finite operator graph. For an exact structural key,
the first epoch runs eagerly, the second captures and launches the
reset/discover/progress/runnable sequence, and later epochs replay it. The key
covers the operator family, plan/runtime addresses, launch bounds, query and KV
layout, scales, FlashInfer plan record, and every metadata-tensor layout.
Captured page-table tensors remain owned by the graph and receive stream-ordered
copies from the current FlashInfer plan before replay. Work-plan contents,
request generations, object versions, and source addresses remain dynamic
stream inputs. `NTA_SGLANG_DEMAND_GRAPH=0` disables this path for an eager
ablation; it never selects stock attention.

`NTA_ENGINE_STATS_FILE=/path/report.json` enables per-process statistics. Normal
demand mode is stream ordered across layers and checks the final epoch plus the
runtime-lifetime failure sequence at the model boundary. The following
additional checks are available only for qualification:

- `NTA_SGLANG_VERIFY_TRANSFER=1`: compare every promoted K/V row with host. This
  synchronizes and copies data, so it must run in a separate correctness arm and
  never in a timed performance arm. `CompareSglangHiCache.py --verify-transfer`
  enforces that separation.
- `NTA_SGLANG_VERIFY_ATTENTION=1`: compare every layer with stock FlashInfer.
- `NTA_SGLANG_VERIFY_EXECUTION=1`: fill and scan output for unwritten or
  non-finite values.

## Measured Opportunity Trace

Demand mode can append GPU-timestamped per-tile availability records:

```bash
NTA_SGLANG_PIPELINE_HOST=0 \
NTA_SGLANG_FORCE_INCREMENTAL=1 \
NTA_OPPORTUNITY_MEASURE_COMPUTE=1 \
NTA_OPPORTUNITY_TRACE_FILE=results/opportunity/sglang-host.jsonl \
NTA_OPPORTUNITY_TIER=host_staged \
NTA_OPPORTUNITY_MODEL=/path/to/model \
NTA_REVISION="$(git rev-parse HEAD)" \
python your_sglang_workload.py

PARALLEL_SLOTS="$(python3 -c \
  'import torch; print(torch.cuda.get_device_properties(0).multi_processor_count)')"
./scripts/analyze-opportunity.py \
  results/opportunity/sglang-host.jsonl \
  --output results/opportunity/model-host-analysis.json \
  --parallel-slots "${PARALLEL_SLOTS}" \
  --material-delay-ns 50000 \
  --require-proceed

./scripts/summarize-opportunity-study.py \
  results/opportunity/model-a-host-analysis.json \
  results/opportunity/model-b-nvme-analysis.json \
  --output results/opportunity/study.json \
  --require-proceed
```

ABI v27 stores one relative `%globaltimer` timestamp per bounded work ticket.
Zero means the tile was resident or already staged at epoch start; a positive
value is written exactly once when the ticket enters the runnable tile set.
With `NTA_OPPORTUNITY_MEASURE_COMPUTE=1`, `compute_ns` is calibrated by a
GPU-event-timed all-resident FlashInfer launch; otherwise it is explicitly
modeled. `available_ns` is a device observation. The analyzer and multi-trace
study reducer are offline evaluation tools, not oracles used by the online
scheduler.

## Cancellation And Stale Work

Cancellation and stale requests are general serving behavior. Agent workloads
can increase their frequency through client timeouts, branch abandonment,
tool-loop termination, retries, and request-slot reuse, but the mechanism is
not specific to agents.

An abort marks the currently bound generation cancelled. A finite CUDA kernel
already executing is not forcibly preempted. Subsequent device liveness checks
cancel its work tickets, and late object or I/O completion is accepted only
when request slot, request generation, object ID, and object version still
match. Rebinding the same engine slot increments its generation, so old work
cannot make data available for the new occupant.

## Incremental Co-Scheduling Target

The adapter never selects an untouched attention kernel for an NTA batch. It
selects between two compiler-generated forms of the same FlashInfer operator:

- **transformed direct** checks request generation and consumes data already
  ordered on the CUDA stream, without allocating tickets; and
- **transformed incremental** externalizes unavailable request/tile work into
  bounded tickets and relaunches only runnable work.

The current serving-adapter default is early layer-preacquired execution. The
schedule-aware eager path chooses transformed direct execution when its online
model predicts that one coalesced transfer round is cheaper, and transformed
incremental execution when multiple useful rounds should repay ticket and
launch setup. This is mechanism specialization, not dispatch to vanilla
FlashInfer. After compiler discovery, a bounded pinned snapshot publishes
generation-checked pending/runnable/completed work and blocked bytes. SGLang's
external-only admission hook consumes the resulting critical-work plan: it
continues resident overlap when executable contributors remain and releases a
staged external batch when the active resident set is data-blocked or terminal.
Mixed batches are not delayed because the measured re-merge policy shifted
instead of reducing latency.

SGLang publishes request generation, priority, cancellation, page mapping, and
batch admission state. NTA returns per-request completed contributors,
remaining unavailable bytes, predicted data service, and runnable compute cost
without downloading per-tile arrays. Deadline/slack propagation and a measured
closed-loop SLO win remain open. Forced layer-wait, bulk, fine-grained, and
skip/rebatch paths remain matched baselines rather than production policy modes.

## Matched Benchmark

Use the comparison harness instead of timing arbitrary HiCache hits:

```bash
python benchmarks/serving/CompareSglangHiCache.py \
  --model /path/to/model \
  --iterations 5 \
  --hot-tokens 96 \
  --resident-tokens 64 \
  --churn-tokens 184 \
  --max-total-tokens 192 \
  --context-length 256 \
  --cuda-graph-decode full \
  --max-latency-regression-percent 5
```

The mechanism-active operator-graph gate uses the same harness with demand
mode forced and the model graph disabled:

```bash
NTA_SGLANG_PIPELINE_HOST=0 \
NTA_SGLANG_FORCE_INCREMENTAL=1 \
NTA_SGLANG_FRAGMENT_LOOKAHEAD=0 \
NTA_SGLANG_CROSS_LAYER_FRONTIER=0 \
NTA_SGLANG_REQUIRE_MIXED_ATTENTION=1 \
python benchmarks/serving/CompareSglangHiCache.py \
  --model /path/to/model --iterations 10 \
  --hot-tokens 96 --resident-tokens 64 --churn-tokens 184 \
  --max-total-tokens 192 --context-length 256 \
  --cuda-graph-decode disabled --require-demand-graph \
  --require-physical-compaction
```

`--require-demand-graph` rejects reports unless the NTA arm has positive eager
warmup, capture, and graph-launch counters. SGLang model-graph counters cannot
satisfy this gate. `--require-physical-compaction` is deliberately separate:
it rejects a run unless a mixed layer's incremental resume grid launches fewer
CTAs than the canonical full grid. A single-request, one-CTA layer can exercise
acquisition and graph replay but cannot provide compaction evidence.

An iteration qualifies only when SGLang reports host-cached tokens for the hot
request. The runner collects the requested number of qualified promotions,
requires identical stock/NTA residency sequences and generated output, and
fails on insufficient host promotions. This avoids averaging resident hits
into an external-I/O result.

For the heterogeneous arrival test, run arm-balanced trials and retain every
raw pair:

```bash
python benchmarks/serving/CompareSglangHiCacheLoadTrials.py \
  --trials 5 \
  --artifact-dir results/serving/hicache-trials \
  --output results/serving/hicache-qualification.json \
  -- \
  --model /path/to/model \
  --external-requests 1 --external-tokens 2048 \
  --resident-requests 1 --resident-tokens 2048 \
  --resident-output-tokens 32 --external-output-tokens 1 \
  --batch-mode coalesced --cuda-graph-decode disabled \
  --require-demand-graph
```

The reducer alternates the first arm and reports geometric means with bootstrap
95% intervals. Exact generated output, all transformed attention, and zero
fallback are mandatory. Each arm runs two performance-excluded mixed-arrival
occurrences: the first warms and the second captures the finite operator graph,
so the timed NTA occurrence replays it. `--require-demand-graph` rejects an
eager-only report. Resident P99
inter-token latency is the causal
interference metric; resident TTFT occurs before the external arrival in this
fixture and is reported only as a general latency diagnostic.

The last ABI-v24 serving diagnostics require every NTA attention launch to be
compiler-transformed and reject stock launch or fallback counters. They are
dirty-tree, uncontrolled-clock smoke runs, not qualification results:

| Workload | Stock | NTA | Throughput ratio | Active NTA form |
| --- | ---: | ---: | ---: | --- |
| Qwen2.5-3B resident, full decode graph, warm JIT cache | 56.791 ms | 55.618 ms | 1.021x | 4,068 direct launches, 36 captures, 32 replays, 2 verified modules |
| Llama-160M resident | 26.325 ms | 26.392 ms | 0.997x | 228 direct launches |
| Llama-160M, 1,024-token host prefix + 512-token resident peer | 21.013 ms | 21.057 ms | 0.998x | 288 direct launches, 36 promoted layers |
| Qwen2.5-3B, 2,048-token host prefix + 1,024-token resident peer | 67.027 ms | 62.453 ms | 1.073x | 684 direct launches, 72 promoted layers |
| Llama-160M forced incremental | 25.993 ms | 34.584 ms | 0.752x | 12 ticketed layers, one reused plan |
| Qwen2.5-3B, 8K host + 8K resident, two fragment waves | 73.750 ms | 85.473 ms | 0.863x | 36 ticketed layers, 35 first-wave lookaheads, 0 stock/fallback |
| Qwen2.5-3B, 16K host + 16K resident, two fragment waves | 110.205 ms | 118.824 ms | 0.927x | 36 ticketed layers, 35 first-wave lookaheads, 0 stock/fallback |
| Qwen2.5-3B, coalesced 2K host + 2K resident | 374.601 ms | 393.013 ms | 0.953x | 36 mixed/ticketed/parallel-progress layers, 0 stock/fallback |
| Qwen2.5-3B, coalesced 4K host + four 4K resident | 446.294 ms | 484.507 ms | 0.921x | 36 mixed/ticketed layers, 132 direct + 32 external work items |
| Qwen2.5-3B, coalesced 4K host + 4K resident, full graph | 211.949 ms | 221.794 ms | 0.956x | 754 direct + 2 ticketed launches, 70 bounded-lookahead layers, 0 stock/fallback |

All rows produced identical output and reported zero fallback. The Qwen row
contains only two qualified promotions and may include process-level scheduler
variance. It is evidence that the active transformed path can compose with a
real engine without the prior regression, not a speedup claim. The forced row
shows that tickets are still too expensive when all demand is known before the
operator and one bulk round is available. The fragmented rows use real
FlashInfer split-K contributors and exact output, but each has one sample. The
16K point narrows the gap from 0.863x at 8K to 0.927x as transfer grows. Neither
is a speedup or paper-quality evidence. The coalesced rows close the earlier
scheduler-segregation gap: SGLang forms a real mixed FlashInfer schedule and NTA
executes both compiler forms. They remain negative because eager per-layer
control and resume cost exceed the overlap benefit. Demand-mode graph/device
control is the next performance gate.

The latest graph run also uses a warp-cooperative compiler-lowered request
guard: one lane per warp reads the immutable request directory and broadcasts
the generation/cancellation decision without a CTA barrier. The isolated tiny
request-bound prefill measured 12.311 us versus 12.998 us stock, so the direct
guard is not the dominant external-serving regression. One-, eight-, and
default-width GPU mover diagnostics remained negative; the dense loss is
instead concentrated in ticketed first-layer setup and transfer interference
with resident decode.

ABI v27 selected-demand mode moves the positive path away from dense early-known
promotion. The SGLang selected harness now runs stock plus tiered selected NTA by
default and keeps the dense NTA arm as an opt-in diagnostic. With Qwen2.5-3B,
16K host prefix, 2K resident peer, budget 32 pages, and selected refresh interval
1024, three dirty-tree seeds produced external P95 TTFT geomean `0.831x` stock
and resident P99 ITL geomean `0.891x`. Resident P95 TPOT remained `1.085x`
stock. Every run used bounded external staging, prefix-summary reuse,
selected-row-table reuse, compiler-generated attention, zero stock attention,
and zero HiCache fallback. This is the current best SGLang signal, not a
production or OSDI-level claim.

The stricter output-match smoke rerun keeps that point alive: budget 32 with
same-budget `QuestRecall.py` metadata and `--require-tiered-output-match`
measured external P95 TTFT `0.848x` stock and resident P99 ITL `0.804x`, with
resident P95 TPOT still `1.113x`. The generated text matched stock and every
mechanism counter remained active. Budget 64 and 128 also preserved generated
text but regressed (`1.434x` and `6.323x` external P95 TTFT respectively), so
the current implementation has a low-budget win and a high-quality/high-budget
overhead cliff.

`benchmarks/serving/SglangSelectedQuality.py` now provides a separate scored
quality smoke. On one exact-prefix retrieval task, stock and selected budgets
32, 64, and 128 all passed while the selected arms exercised external claims,
bounded staging, device compaction, compiler-generated attention, zero stock
attention, and zero HiCache fallback. That result is a sanity check, not
paper-level quality evidence. The profiled budget-128 load run instead points
at the next integration target: 2,376 selected direct-operator layer
invocations consumed `881.5 ms` of GPU time and CPU enqueue consumed `98.6 ms`,
while selected transfer timing is not yet separated. The selected path needs a
page-native or lower-overhead operator form before the high-recall operating
point can be a headline.

Three arm-balanced repetitions of that graph workload produced an
output-throughput geometric mean of `0.9447x` with bootstrap interval
`[0.9337x, 0.9578x]`. External TTFT was `1.3386x` and resident P99 inter-token
latency was `1.4991x`; SLO goodput was `0.7498x`. Every output matched, all
attention was transformed, and fallback remained zero. Three repetitions are
a diagnostic counter-result, not paper-level statistical evidence. The positive
path requires device-selected demand
that avoids transfer the early-known bulk baseline cannot avoid.

The final clean `bec105f` three-trial diagnostic sharpened that boundary:
`0.9791x` throughput, `0.7771x` goodput, `0.9557x` resident P95 TTFT (better),
`1.0394x` external P95 TTFT, and `1.2549x` resident P99 inter-token latency,
with 5,138 of 5,148 launches in the direct form and only 10 ticketed
incremental launches because lookahead acquisition satisfied 350 of 360
external layers before attention. Dense early-known demand exposes almost no
incremental opportunity; the persistent causal regression is resident-tail
interference from acquisition movers, which ran at elevated stream priority in
every recorded dense run. Movers now default to the lowest CUDA priority
(`NTA_SGLANG_MOVER_STREAM_PRIORITY`), and the dense interference series must be
re-measured under that default before any further dense conclusion. Detailed
counters are in `docs/VALIDATION.md` and
`results/serving/sglang-hicache-load-bec105f-qualification.json`. The
qualification runner now defaults to ten paired trials and stamps
`evidence_grade` (`diagnostic` below ten) into every report so a three-trial
run can no longer be mistaken for qualified evidence.

The resident graph row has only three measured samples and performs no external
acquisition. It proves that contract-validated transformed graph replay can meet
the resident overhead gate; it is not evidence for incremental execution. The
matched harness now enables full decode graphs and runs an unmeasured priming
process for each backend by default so first-use JIT compilation cannot be
reported as serving speedup.

Older measurements that dispatched preacquired or graph work to stock
FlashInfer are rejected as contribution evidence, including the 18.614 ms
versus 11.469 ms and 13.333 ms versus 12.993 ms comparisons. They may describe
transfer scheduling, but they do not measure the compiler/runtime mechanism
required by this project.

The ABI-v23 v10 immediate-mixed point produced exact output with 1,080
compiler-transformed attention launches, one mixed ticketed layer, 35
preacquired suffix layers, and zero stock/fallback. It measured 0.977x stock
throughput, 1.012x resident P99 inter-token latency, and 1.021x external TTFT.
A five-trial admission/re-merge policy worsened all causal metrics and was
removed. These results validate integration and falsify the current dense
performance hypothesis; they do not support a production or OSDI claim.

## Other Engines

The runtime, ABI, compiler pass, request tracker, indexed-host registration,
and FlashInfer hook are engine-neutral. The SGLang module is only a lifecycle
adapter.

No vLLM backend is registered today. vLLM 0.13.0 has an experimental
KVConnector interface carrying request IDs and block-transfer metadata. Its
stock offloading connector completes loading before attention, while its
Blackwell FlashInfer backend normally selects TRTLLM attention entry points
outside NTA's current wrapper hook. A correct integration therefore needs one
connector/backend adapter that owns generation policy, block mapping, transfer
events, cancellation, kernel selection, and stream/graph lifetime together.
The installed vLLM CUDA extension is also ABI-incompatible with the local
PyTorch, so that path cannot be executed on this host. Guessing missing state
would reintroduce the semantic gap this project is meant to close.

TensorRT-LLM or another engine can integrate through the same host contract
when it exposes equivalent lifecycle metadata. Kernel integration additionally
requires source or JIT access to a CTA-uniform pre-state hook; precompiled
cubins cannot be transformed safely.
