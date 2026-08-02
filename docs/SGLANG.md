# SGLang Integration

Status: SGLang 0.5.14 and FlashInfer 0.6.12 are integrated through SGLang's
installed plugin mechanism. The adapter handles eager CPU-DRAM HiCache
promotion and full decode CUDA-graph replay. Preacquired and resident work uses
stock FlashInfer; only unresolved multi-round demand work uses instrumented
wrappers. This is a locally validated integration, not a production-readiness
claim.

## Plugin Boundary

Installing `nta-runtime` publishes this entry point:

```text
sglang.srt.plugins: nta = nta_runtime.plugins.sglang:register
```

SGLang loads it in frontend and spawned scheduler processes. Registration adds
the `nta_flashinfer` attention backend and three lifecycle hooks:

1. `HiCacheController.start_loading`: claim a supported host load before
   SGLang copies it, preserving its page map, completion events, and ACKs.
2. `Scheduler.abort_request`: mirror exact, prefix, or abort-all selection into
   NTA's current request generations before SGLang mutates its queues.
3. `Scheduler.release_host_resources`: atomically flush engine statistics on
   graceful scheduler shutdown.

The plugin also preserves live request IDs and priorities when SGLang 0.5.14
constructs its padded decode-graph replay view. The patch is version-pinned and
fails closed if upstream metadata no longer matches the validated structure;
synthetic padding slots receive distinct non-user request IDs.

The adapter binds `ForwardBatch.rids` to compact engine-neutral request slots
with monotonically changing generations. At SGLang's HiCache producer point,
the default path enqueues each layer's existing tuned indexed-copy kernel on a
dedicated finite CUDA stream and records a preallocated data-available event. The
consumer stream waits only for the layer it is about to use. The
unchanged stock FlashInfer kernel then consumes that layer. Request binding is
retained for lifecycle accounting and for incremental batches, where the
compiler-instrumented CTA checks the bound generation before attention.

```text
SGLang request and HiCache page map
  -> early layer-wise indexed copy on a finite CUDA stream
  -> preallocated per-layer data-available event
  -> compact request slot/generation binding
  -> stock FlashInfer after acquisition, or ABI-20 guarded incremental CTAs
  -> SGLang-owned output tensor and HiCache completion
```

This preacquired path allocates no per-batch `WorkItem` array, publishes no
object directory, launches no reset/progress/completion kernels, and performs
no model-thread page-map download. `NTA_SGLANG_PIPELINE_HOST=0` selects the
schedule-aware policy path: it extracts FlashInfer's request/KV-tile schedule
and evaluates the calibrated bulk-versus-incremental cost model before building
a device plan. A one-round decision uses SGLang's tuned layer transfer and the
stock FlashInfer wrapper without allocating or uploading CTA plan state. A
multi-round decision registers exact indexed K/V objects and runs bounded CTA
acquisition epochs. `NTA_SGLANG_FORCE_INCREMENTAL=1` is an evaluation ablation
that overrides the cost gate. Every incremental epoch is checked before its
output is returned; transport failure, cancellation, or an exhausted finite
bound therefore fails closed rather than exposing an incomplete merge. All
instrumented paths preserve request-generation checks and the finite-kernel
contract.

No CPU thread copies KV or polls a device queue. CPU userspace still performs
normal batch planning and launches finite CUDA kernels. This distinction is
important: CPU-free data movement does not mean CPU-free kernel submission.

## Supported Profile

The adapter fails at construction for incompatible global settings. Once it
claims a HiCache batch, planning errors fail closed before attention. Setting
`NTA_SGLANG_ALLOW_FALLBACK=1` explicitly enables stock fallback for
availability testing; measured runs require that variable unset and the
fallback counter equal zero.

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
| Resident-only batch | untouched stock FlashInfer wrapper |

Unsupported HiCache ownership, malformed page maps, schedule mismatch, or
capacity exhaustion is recorded before attention and raises by default. An
explicit availability fallback calls SGLang's original layer-wise transfer
before the stock attention path. A claimed batch never silently switches after
partially executing attention. `hicache_claimed_batches`,
`hicache_fallback_batches`, and the last fallback reason are written to the
engine statistics file.

Incremental-plan reuse is keyed by the exact immutable host/device page-pair
tuples, FlashInfer request/tile schedule, request slots, KV allocation addresses,
byte extents, and prefetch state. Transient generations are rebound from the
runtime request directory at the compiler hook. Shape equality alone is not a
cache hit: a different page map forces indexed-object registration, advances
the object version, and uploads a new plan.

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
Replay binds the current padded request IDs/generations and invokes the real
stock FA2 wrapper. A pending HiCache transfer is ordered before graph replay;
CUDA stream-capture isolation forbids capturing a wait on uncaptured
producer-stream work. Eager prefill retains per-layer copy/compute overlap. The
demand-mode reset/progress/runnable loop is not yet embedded in SGLang's replay
graph.

`NTA_ENGINE_STATS_FILE=/path/report.json` enables per-process statistics. Demand
mode already synchronizes once per layer to enforce its terminal epoch result.
The following additional checks are available only for qualification:

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

ABI v20 stores one relative `%globaltimer` timestamp per bounded work ticket.
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

The current serving-adapter default is early layer-preacquired execution. The
schedule-aware eager path additionally chooses tuned bulk transfer when its
online model predicts no benefit and compiler-generated incremental FlashInfer
execution when it predicts multiple useful rounds. Progress is not yet fed back
into SGLang's next batch admission decision, and the incremental loop is not yet
inside decode graph replay.

SGLang will publish request generation, SLO slack, cancellation, page mapping,
and batch admission state. NTA will return per-request completed contributors,
remaining unavailable bytes, predicted data arrival, and runnable compute cost.
The scheduler will use those summaries for the next batch without downloading
per-tile arrays. Forced layer-wait, bulk, fine-grained, and skip/rebatch paths
remain matched baselines rather than production policy modes.

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

An iteration qualifies only when SGLang reports host-cached tokens for the hot
request. The runner collects the requested number of qualified promotions,
requires identical stock/NTA residency sequences and generated output, and
fails on insufficient host promotions. This avoids averaging resident hits
into an external-I/O result.

An ABI-v18 local graph run collected three qualified host promotions in five
attempts with a 96-token hot request and 64-token fresh peer. Stock measured
18.614 ms median promotion latency; `nta_flashinfer` measured 11.469 ms, a
38.39% reduction and 1.623x promotion-throughput ratio. Generated output and
qualified-attempt indices matched, and NTA recorded four graph captures, ten
decode graph replays, three claimed HiCache batches, and zero fallback. The
promotions executed in eager prefill and the following decode used resident
graph replay (`graph_external_batches=0`), so this result validates composition
of promotion with graph decode, not graph-captured demand acquisition or
incremental execution. Five attempts on a tiny model with uncontrolled clocks
are insufficient for a paper performance claim.

The one-GPU Llama-160M runs predate ABI v18 and are retained as historical
integration measurements, not current-ABI evidence. They measured 15 qualified promotions. The
single-request case used a 96-token host-cached prefix. A conservative mixed
run added a 64-token fresh request:

| Workload | Stock | `nta_flashinfer` | Latency change |
| --- | ---: | ---: | ---: |
| host promotion | 13.361 ms | 13.016 ms | -2.58% |
| host promotion + fresh peer | 14.762 ms | 14.079 ms | -4.62% |

The runs used identical qualified-attempt indices, generated identical output,
and reported zero fallback. The corresponding promotion-throughput changes
were +2.65% and +4.85%. These are local regression measurements on one tiny
model with uncontrolled clocks, not a general performance or production
claim. A subsequent complete comparison measured 20.073 ms stock and 13.866
ms NTA because SGLang co-batched the peer more often in the NTA process. The
report therefore also records per-request latency and peer delay; controlled
arrival traces and repeated confidence intervals are required before
attributing that scheduler-sensitive 30.92% difference to the mechanism.

An ABI-v20 schedule-aware run disabled early acquisition so the adapter had to
choose after observing the real FlashInfer schedule. For a 96-token host-cached
request, the online model selected one bulk round, issued all 12 layer
promotions on the transfer stream, and overlapped later layers with attention.
Five qualified promotions measured 12.993 ms stock and 13.333 ms NTA median
latency, a 2.62% cost and 0.974x throughput ratio. Output and residency traces
matched, with zero fallback, zero plan uploads, and zero CTA work because this
trace had no predicted incremental opportunity. This clears the local 5%
regression gate and demonstrates low-regret selection; it is not evidence that
incremental execution improves a workload with arrival skew.

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
