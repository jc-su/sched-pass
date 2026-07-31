# ROADMAP — from validated mechanism to OSDI-level system

MoE ARM WIRED + REAL-TRACE LOADGEN (2026-07-10, python/sched_moe_hook.py,
python/sched_trace_loadgen.py, test/py/test_moe_hook.py). The moat generalizes off
attention: the SAME binding, but the Rosetta stone is the ROUTING. SGLang's
FusedMoE.forward(hidden_states, topk_output) is intercepted; topk_output.topk_ids
= per-token expert assignment = the m_indptr the kernel binds to. The control
plane reads that GLOBAL, per-forward, DATA-DEPENDENT routing (unknowable at plan
time -- the whole point vs attention's static plan), detects the hot expert, and
CAPS it (GShard/Switch capacity, E2 drop-tokens by zeroing the routing WEIGHT, not
the id -> safe on any fused backend). Two wireup facts that MATTER:
  * CAPTURE-SAFE by construction: SGLang captures the decode forward (MoE incl.)
    in a CUDA graph. The cap is all fixed-shape ops (scatter_add not bincount;
    masked_fill not boolean-assign; NO .item() in the hot path; UNCONDITIONAL
    replace -- no host branch on the drop count). Bit-exact when balanced (mask is
    a no-op) AND capturable when it fires.
  * VERIFIED against the REAL sglang types (test_moe_hook.py, 13/13, CPU, no MoE
    model): moe_expert_cap transforms a real StandardTopKOutput (hot expert capped
    to C, topk_ids + router_logits untouched, namedtuple identity preserved);
    register_moe_cap() actually patches FusedMoE.forward (identity changes,
    idempotent); the wrapped forward feeds orig a CAPPED routing; fail-soft
    passthrough on the forms carrying no topk_ids (BypassedTopKOutput,
    TritonKernelTopKOutput). Armed on the FIRST run_batch (FusedMoE imported by
    model-load-time, before the first MoE forward), gated SCHED_MOE_CAP.
REAL-TRACE LOADGEN (the "real trace" the user asked for): Qwen-Bailian
usage traces (alibaba-edu/qwen-bailian-usagetraces-anon). Raw prompts are absent
by design, but each record carries input_length/output_length, arrival timestamp,
and hash_ids -- the 16-token KV-block hashes (blksz_16 == our PAGE=16) that ENCODE
the shared-prefix structure. Fetched WITHOUT git-lfs via the media CDN
(scripts/fetch_trace.sh; the .jsonl are 133-byte LFS pointers, media.githubusercontent.com
serves the real 56MB). To-C sample (972 recs, ~5min): input_length heavy-tailed
(p50=829, p90=5116, max=16744 -- 20x spread => dispersion/straggler regime),
output p50=292, 3.2 req/s, 71% text, 23% multi-turn, and 39.4% KV-block REUSE (the
radix/shared-prefix regime). RE-CRAFT is faithful: each hash_id -> a deterministic
16-token block, prompt = concat of blocks => two requests sharing a hash_id PREFIX
share an input_ids PREFIX, so SGLang's radix tree sees the SAME reuse the
production trace had (measured: crafted block reuse 39.4% == trace hash reuse
39.4%; 972/972 lengths exact). Sent as input_ids (/generate) -> exact, no
tokenizer round-trip. selftest offline (shared-prefix reconstruction + real
fixture), both in the gate.
LIVE TRACE-DRIVEN DECODE A/B (2026-07-10, run_trace_ab.sh path, Qwen3-8B, real
qwen_traceA sample, 200 reqs). Booted the WOVEN server (decode woven, pi+timer,
plane@0x5c00) and a TRUE STOCK server (0 [sched-sglang] lines, identical backends
decode=flashinfer/prefill=triton, same KV/model) and replayed the SAME determin-
istic re-crafted input_ids at both. WIREUP PROVEN: 200/200 served through the
armed woven path, real trace, reconstructed shared prefixes. BUT the throughput
A/B was a TRAP I nearly fell into: woven-1st-boot=67s looked like a 2.4x win over
stock=164s. Re-running to CONTROL for boot order exposed it as a pure GPU-THERMAL
artifact -- the numbers degrade MONOTONICALLY by run position regardless of arm:
woven-fresh 67s -> stock 164s -> stock-warm 187s -> woven-hot 212s (GPU 77->84C,
first run on a cool GPU gets boost clocks, every later run is throttled).
CONTROLLING for position, woven == stock on decode: NO effect, NO regression.
This is a LIVE confirmation of the standing finding -- decode is DRAM-bound +
predictable, the WORST regime for these levers (~12% ceiling, here buried under
thermal noise). The moat does not pay on decode throughput and we do not claim it
does; its value is the harder regimes (prefill split-skew -40%, MoE cap, straggler
shed -49..-96%). LESSON (same discipline the user enforced on me): a single-run
A/B on a thermally-unpinned GPU is not evidence; the honest decode verdict is
parity. A real decode A/B would need locked clocks (nvidia-smi -lgc) + interleaved
cool-down, and is not worth it since parity is the expected + acceptable result.

MoE CAP VERIFIED LIVE ON GPU, NO MODEL DOWNLOAD (2026-07-10,
test/py/test_moe_forward_gpu.py). Turns out a MoE model download is NOT needed to
test the cap in a REAL forward: instantiate one sglang FusedMoE layer standalone
(single-rank init_distributed_environment + initialize_model_parallel +
set_global_server_args_for_scheduler(ServerArgs(model_path=<cached dense>)),
random weights) and run its actual triton fused-MoE kernel on the Blackwell GPU.
Measured (E=8, hidden=512, top_k=2, 256 tokens; kernel is DETERMINISTIC here,
noise floor 0.0):
  * generous capacity -> forward BIT-EXACT vs uncapped (E2 degrades to identity).
  * tight capacity + hot expert -> exactly the 105 dropped tokens' outputs change;
    UNTOUCHED tokens BIT-EXACT (targeted, no collateral -- same "lights unaffected"
    property as the attention straggler shed).
  * register_moe_cap() HOOK path -> output BIT-EXACT-identical to the manual cap
    (wireup exact) and != uncapped (the cap fired INSIDE FusedMoE.forward).
So the cap is now validated end-to-end in a live GPU MoE forward through the real
hook. In the gate (self-skips without a GPU).

FULL MoE SERVING RUN, LIVE ON Qwen3-30B-A3B (2026-07-10). Pulled the real
Qwen3-30B-A3B (128 experts, top_k=8, 48 layers, ~57GB) and served it woven with
SCHED_MOE_CAP=1 SCHED_MOE_OBSERVE=1 --disable-cuda-graph, trace-driven. PROVEN
end to end: "MoE expert cap armed on FusedMoE.forward"; the cap fires across all 48
layers every forward (5000+ in the first ~30s), server serves 30/30 fail-soft.
Verified Qwen3MoE routes through the patched FusedMoE (get_moe_impl_class ->
FusedMoE for single-GPU/no-EP; self.experts(hidden,topk_output) -> forward).
TWO honest findings the run forced out:
  * REGIME GUARD: at capacity_factor=1.0 the observe log showed ~68% of routed
    slots dropped -- because per-batch capacity is a LARGE-batch construct and
    tiny autoregressive DECODE batches (nt~16, C=int(16*8/128)=1) are all
    small-sample lumpiness, not imbalance. Added min_capacity (skip when C<8):
    now only PREFILL batches cap (288 of 3200 forwards), decode is skipped. Same
    "measure the regime" discipline as the decode/thermal findings.
  * RE-CRAFT IS STRUCTURAL, NOT SEMANTIC: even guarded, prefill drops stayed ~62%
    -- because the loadgen re-crafts prompts from hash_ids as RANDOM token-ids
    (faithful for LENGTH + shared-prefix STRUCTURE, which is what the radix tree /
    attention levers react to, but NOT semantic). Random tokens -> garbage router
    logits -> DEGENERATE routing -> high drops. CONFIRMED by a real-text probe on
    the SAME armed server: "The capital of France is" -> " Paris. ... London ...",
    "List three colors:" -> " red, green, blue ...". So on REAL text the cap at
    1.25x is near-transparent (Qwen3-30B is load-balance-trained -> few overflow),
    i.e. E2 correctly degrades to ~identity. The high trace-replay drop rate is a
    random-token artifact, NOT real imbalance.
NET: the MoE cap WIREUP + mechanism are proven live on a real 30B MoE (armed,
fires, fail-soft, regime-guarded, near-transparent on real text). Measuring a
MEANINGFUL real-imbalance NUMBER would need real prompt TEXT, which this
anonymized trace lacks by design -- the trace drives the STRUCTURAL (attention/
radix) levers correctly, not the SEMANTIC (MoE routing) distribution. The cap
ACTION is crowded (GShard/Switch/Megablocks); the contribution is the BINDING
GENERALITY (topk_ids as the same Rosetta stone as request_indices).

STRAGGLER SHED MEASURED + THE SKEW QUESTION (2026-07-09,
test/py/eval_straggler_shed.py). Bimodal BS=1024, 69 stragglers @512 pages + rest
@1: shedding ONLY the stragglers' KV (light requests bit-exact) collapses the
makespan -- cap256 -48.9%, cap128 -73.5%, cap64 -87%. 6.7% of requests hold the
entire makespan hostage; the control plane detects them (global view) and the
kernel sheds exactly them (local action) -- the moat. eKV/H2O shed UNIFORMLY; we
shed the STRAGGLER, targeted.
THE SKEW QUESTION (can CLC/SHAPE give the straggler MORE, driven by userspace
info?) -- checked by physics, not dismissed:
  * SHAPE-skew (pin the straggler's KV in L2): FUTILE. The straggler is DRAM-
    BANDWIDTH-bound (long single-pass KV); cache residency does not reduce DRAM
    reads (cache_hint_bw: hints are BW-invariant). Targeting the straggler does
    not change its physics. Out.
  * RESOURCE-skew (give the straggler more SMs = OVER-SPLIT it): GENUINE and in
    the moat's domain. straggler_time = work / parallelism; more split -> more
    SMs -> faster, and it is BIT-EXACT (as exact as stock split-kv, just more
    chunks). NOT HW-subsumed: FlashInfer splits by a UNIFORM heuristic, not
    straggler-aware; the control plane's GLOBAL batch view can compute the
    makespan-OPTIMAL split (skew parallelism to the stragglers, take it from the
    1-page lights that finish instantly anyway). CLC per se does NOT skew (it
    claims tiles; it does not create parallelism -- the acceleration is the
    SPLIT). So the user's "skew to the straggler" = control-plane-driven OPTIMAL
    SPLIT ALLOCATION -- a THIRD straggler tool, BIT-EXACT (vs shed's approximate),
    untested but grounded. TWO straggler tools now: SHED (E2, approximate,
    -49..-96% measured) and SPLIT-SKEW (E1-ish, bit-exact, control-plane-optimal).

SPLIT-SKEW REVIVES FOR PREFILL -- REGIME MATTERS (2026-07-09,
test/py/eval_split_skew_prefill.py). Same total prefill work, varied parallelism:
1x1024 (one long prompt) 72.2us vs 16x256 (parallel) 43.4us -> a single long
prompt is 40% SLOWER because FlashInfer UNDER-splits it (128 tiles vs 256).
Prefill is L2/compute-bound, NOT DRAM-walled, so MORE parallelism helps. So
SPLIT-SKEW REVIVES for PREFILL stragglers (bit-exact -40%, control-plane over-
splits the straggler) -- exactly the chunked-prefill case. THE LESSON (PI's
point): decode is the EASY, predictable, DRAM-walled regime -- the WORST case for
these levers -- and testing only decode UNDERSOLD them. The SAME lever
(split-skew) is FUTILE on decode and a -40% WIN on prefill. Verdicts are
PER-REGIME, and real serving (chunked prefill, MoE, continuous batching) lives in
the harder regimes where the levers revive. Multi-regime lever map:
  decode  (DRAM, predictable): ORDER -12, OBSERVE 95%, SHED straggler -49..-96;
                               split-skew/SHAPE/CLC futile (DRAM/HW-local).
  prefill (L2/compute):        ORDER -15, OBSERVE, SPLIT-SKEW -40 (straggler);
                               SHAPE futile (reuse in shared).
  MoE     (MISPREDICTED cost): OBSERVE-driven closed loop is the KILLER case --
                               routing makes cost data-dependent -> static
                               scheduling FAILS -> the 95% misprediction-recovery
                               (measured on a synthetic mispredicted order) is
                               REALIZED NATURALLY here; + CLC for expert imbalance.
                               [untested -- needs a MoE kernel, the top future
                               experiment because the thesis is STRONGEST here.]
  cont.batch (dynamic):        the moat's core -- per-request info, per-step.

SPLIT-SKEW FUTILE FOR DECODE STRAGGLERS (2026-07-09, test/py/eval_split_skew.py).
Fixed total KV=8192 pages, varied parallelism (1 long req -> 512 short): makespan
INVARIANT (~70us for nreq 1..512). Two independent kills: (1) DRAM-WALLED -- same
total KV = same makespan regardless of split; more SMs wait on DRAM (SHAPE
physics). (2) FlashInfer ALREADY over-splits the straggler: one 8192-page request
-> 274 TILES; no under-parallelization to exploit. So "give the straggler more
resources/SMs" (the skew idea) is FUTILE for DRAM-bound decode stragglers -- they
are DRAM-starved, not compute-starved, and already parallelized. RESULT: for a
DRAM-bound decode straggler the ONLY lever that helps is SHED (reduce KV traffic;
measured -49..-96%). Both "give more" tools -- SPLIT-SKEW and SHAPE -- are futile
(DRAM wall). The straggler contribution is clean: OBSERVE detects the straggler
(global view) -> SHED exactly it (E2, targeted, the light requests bit-exact).
(Prefill stragglers are compute/L2-bound, NOT DRAM-bound -- split-skew is
plausible THERE, untested; but the batch-straggler case the PI asked about is
decode, and it is DRAM-walled.)

MoE IS THE STRONG REGIME -- FAST KERNEL, IMBALANCE COSTS +58..72% (2026-07-09,
test/py/eval_moe_fast.py, grouped_mm_bf16 cudnn tensor-core path; needed
pip install nvidia-cudnn-frontend). Balanced 1534us @358 TFLOP/s (near-peak) vs
1-hot-expert 2421/2646/2585us = +58/+72/+68% (4x/8x/12x skew). The earlier
SegmentGEMM +7% was a SLOW-PATH ARTIFACT (1.7% peak, overhead-bound) that MASKED
the effect -- corrected. The fast grouped GEMM does NOT absorb expert imbalance:
the hot expert is a REAL straggler and the tiny experts under-fill the GPU. This
is the OPPOSITE of DRAM-balanced decode (+0% imbalance). => MoE is THE moat regime
the PI predicted: the control plane sees the routing counts (m_indptr -- global,
per-step, DATA-DEPENDENT, unknowable statically) and can CAP the hot expert (E2
drop-tokens / capacity -- the expert-level straggler shed) or skew resources to
it. Decode UNDERSOLD the thesis (+0%); MoE is where it SHINES (+58..72%
opportunity). The straggler-shed thesis generalizes cleanly to MoE experts, moat-
driven. RECOVERY MEASURED (test/py/eval_moe_capacity.py): +62% penalty; the
control plane CAPS the hot expert (E2 drop-tokens, routing-count-driven) ->
recovers 34% @keep-50%, 45% @keep-25%, 60% @keep-12%. So the MoE moat ACTION
WORKS -- the expert-level straggler shed. Recovery is PARTIAL (60% not 100%: the
tiny under-filling experts remain), and drop-tokens is a KNOWN MoE technique --
the moat's angle is the TARGETING (cap EXACTLY the hot expert, detected globally
per-step from the routing counts), not the drop mechanism. MoE arc COMPLETE:
+62% opportunity, 60% recovered by a targeted routing-driven action. The
multi-regime evidence now spans FOUR regimes (decode floor +0% imbalance; prefill
split-skew -40%; MoE +62% -> 60% recovered; closed loop 95%; straggler shed
-49..-96%) -- the moat pays MORE as the regime gets less predictable, exactly the
PI's thesis.
THE REQUEST<->TILE BINDING (the moat's mechanism, PI asked "how to correspond
request-id in the kernel"). Three pieces, all built:
  1. FIXED-VA BAKED tables: the control plane puts its per-request tables (order,
     timer, budget, ctrl) at a CANONICAL VA (SchedArena, MAP_FIXED 0x5C00...) and
     BAKES that address into the kernel at JIT compile (SCHED_BAKE_*). The kernel
     reads memory at a COMPILE-TIME-CONSTANT address -- no pointer-passing.
  2. request_indices = the ROSETTA STONE: FlashInfer's plan maps tile->request
     (DecodePlanInfo[3] / PrefillPlanInfo[4], reverse-engineered). Both sides use
     it: control plane writes table[tile] = f(info[request_indices[tile]]); kernel
     at ctaid reads table[order[ctaid]] -> its request's info -> acts.
  3. BIDIRECTIONAL + LATE-BOUND: kernel writes timer[tile]; control plane folds
     cost[req] = sum over tiles with request_indices[tile]==req (exact,
     split-safe). And request_indices is RE-READ EVERY STEP (batch churns), so the
     rid<->tile correspondence is LIVE, not frozen -- "now we associate them"
     dynamically. BEFORE: the kernel had the static plan but NO per-request
     control info and no way to report back. NOW: the bridge carries per-request
     info IN (baked tile-indexed table) and observations OUT (folded timer), using
     request_indices as the shared index. THAT association IS the moat.

THE MOAT (thesis, 2026-07-09 -- stated by the PI, and it reframes everything
below). The contribution is NOT "a compiler pass" and NOT "a bag of levers." It
is: a woven, BIDIRECTIONAL, LATE-BOUND bridge that carries CONTROL-PLANE
PER-REQUEST INFORMATION INTO the fused GPU kernel, so the kernel ACTS on it at
runtime -- information that was previously BOUND (frozen into a static plan at
plan time, invisible to the kernel). The kernel READS live per-request control
state (order, budget, hints) and WRITES per-request observations (cost); the
GLOBAL control plane (which alone knows all requests' costs, priorities, SLAs,
load) thereby STEERS LOCAL kernel execution, per-request, per-step, bit-exact by
effect-typing. Stock fused kernels are per-request-BLIND; we make them
per-request-AWARE.
WHY THIS EXPLAINS THE WHOLE TAXONOMY (the deep point): the levers that PAY all
need GLOBAL, CROSS-REQUEST information that ONLY the control plane has --
  * ORDER: the cross-request cost RANKING (-12/-15%)
  * OBSERVE: per-request cost MEASUREMENT feeding the loop (95% recovery)
  * E2/straggler-shed: the per-request SLA BUDGET
...and the levers that FAIL are LOCAL / kernel-internal, which the hardware or
the kernel author already handles -- CLC (load-balance: HW block scheduler),
SHAPE (cache: shared-memory blocking), PDL (overlap: device-resident tables).
So the moat's DOMAIN is exactly the global-information actions, and those are
exactly the winners. The negatives are not failures -- they are EVIDENCE that
the value lives in bridging global per-request info, not in re-doing what the HW
already does locally. The closed loop is the flagship: it exercises the bridge in
BOTH directions (kernel observes cost -> control plane relearns -> kernel reads
new order -> acts). No one bridges GLOBAL per-request control info into UNMODIFIED
fused kernels, LIVE and BIT-EXACT. THAT is the moat; the pass is just how it is
built, and the levers are what the newly-informed kernel chooses to do.



EXACT PER-REQUEST + PREFILL WIRE-UP (2026-07-09). Done + verified:
* PREFILL exact per-request attribution. PrefillPlanInfo.ToVector() is 15 int64s
  (empirically re-derived: test/py/probe_prefill_plan.py) --
  [0]padded_batch_size [4]request_indices_offset(bytes) [14]split_kv.
  request_indices[tile] is non-decreasing, in [0,B), covers every request: the
  exact one-request->several-tiles map. Prefill ALWAYS multi-tiles a request
  (qo x kv chunks), so no split-gate (unlike decode, which gates on split_kv@9).
  PROOF (test/py/test_flashinfer_prefill.py, 8/8 PASS): longest request
  qo_len=566 -> 36 grid tiles, timer folds to EXACTLY that one request's cost,
  zero leakage from any other request. Not "≈" -- exact, by the binding being a
  valid partition of tiles into requests.
* Plugin refactor (no duplication): shared _plan_request_indices(w,n,want_len,
  req_field,require_flag) reads request_indices for BOTH decode (10,3,gate@9) and
  prefill (15,4,no-gate); shared _model_runner/_backend_wrapper resolve the
  decode/prefill FlashInfer wrappers. clone()-before-view() makes the byte-slice
  int32-aligned for any offset.
* EXACT fold, VECTORIZED: _consume_probe now scatter_add's per-tile cycles into
  per-request (was a python per-tile loop) -- exact regardless of split, and off
  the hot path's python-loop cost.
* Prefill SERVING wire-up (opt-in): _attn_mode() dispatches decode vs prefill;
  the run_batch hooks fire for both when SCHED_WEAVE_PREFILL=1;
  serve_sglang_armed.sh sets SCHED_WEAVE_ONLY=batch_decode,batch_prefill under
  that flag. Gated OFF by default -- decode-only stays the proven path; the
  prefill MECHANISM is proven by the microbench, live-serving validation is the
  remaining step.
* Device-L2 timer x prefill: FIXED. The earlier "device timer faults prefill"
  was a SYMPTOM of the broken scalar kernel (pre-macro-fix), not a channel bug --
  re-validated 8/8 PASS on the correct fast-path kernel with
  SCHED_TEST_DEVICE_TIMER=1 (the serving-default channel).
Regression: decode still bit-exact (test_flashinfer_arm.py); control-plane suite
(manifest/dynamic_loop/timer_gate/timer_indirect/failsafe/controller) all PASS.
* LIVE SERVING VALIDATED (2026-07-09). Real SGLang server, llama-160m,
  SCHED_WEAVE_PREFILL=1: boots + JIT-compiles the woven prefill kernel under
  clang + serves. A 411-token prompt (multi-tile prefill, cta_tile_q=128) ->
  woven-armed (observe) output is BIT-EXACT vs a stock (SCHED_PLUGIN="" clang-
  unwoven) boot: 24/24 output_ids identical. Wired: serve_sglang_armed.sh swaps
  the prefill backend triton->flashinfer under the flag (Triton prefill is off
  the woven path -- weaving batch_prefill alone was a silent no-op without this);
  plugin _attn_mode() gates prefill batches; SCHED_WEAVE_PREFILL exported to the
  scheduler proc. Boot method note: harness run_in_background works; setsid-nohup
  (start_server_detached.sh) and "pkill;boot" compounds die silently here --
  boot/poll/generate/kill as SEPARATE calls.
Remaining: GQA-3B serving-gate at SCHED_WEAVE_PREFILL=1.

SHED cp.async WEAVE IS THE WRONG MECHANISM FOR DRAM (2026-07-09, verified in
SchedWeave.cpp). emitShedScoreMask does `s' = keep ? s : -inf` -- masks the Q.K
SCORE AFTER the K is loaded -> changes accuracy (truncated attention) but saves
ZERO DRAM. AsyncSite (cp.async) is collected for PREFETCH only; no shed path, and
adding one cannot save DRAM: (a) masking the async score = same, no save; (b)
predicating the cp.async out breaks commit_group/wait_group pipeline accounting;
(c) early loop-exit saves DRAM but only as a SUFFIX (oldest-page) truncation, not
observation-chosen. The DRAM save is PLAN-LEVEL: shorten the per-request
kv_indptr to the budget -> FlashInfer never issues the cp.async for dropped pages.
That is a control-plane change, NOT a pass weave -- and it IS eKV/Quest (page
selection). So the in-kernel cp.async shed weave does not yield a non-eKV DRAM
win. The ONLY non-eKV angle is the per-request SLA-driven BUDGET (a scheduling
decision -- which requests may be approximated, composed with ORDER+OBSERVE),
layered on top of plan-level (eKV-style) truncation. Honest verdict: the E2 DRAM
lever is fundamentally eKV; do NOT reimplement it in-kernel. Genuine
contributions remain the FRAMEWORK + CLOSED LOOP + the measured lever TAXONOMY.

SHED (E2) WORKS ON sm_120 -- codegen fault already fixed (2026-07-09). The
paged_decode fixture (test/paged_decode.cu) passes ALL shed cases on sm_120a: the
old IADD.64 junk-high-register fault is resolved by the bakedslot address load
(rtBuffer, SchedUtil.cpp:213). Crucially for the bit-exact question:
  * G tau=0 (shed OFF): BIT-EXACT vs inert. => opt-in E2 keeps a bit-exact DEFAULT.
  * H softmax tau=8 (shed ON): matches the TRUNCATED-attention CPU reference
    EXACTLY (dropped tokens get zero weight, -inf mask semantics). => the E2
    approximation is PRINCIPLED and WELL-DEFINED -- it is exactly attention over
    the top-budget KV (H2O/Quest sparsity), NOT an ad-hoc drop. Error = the
    DROPPED ATTENTION MASS (analyzable), which the OBSERVE lever MINIMIZES by
    shedding the LOW-attention KV.
BIT-EXACT VERDICT (honest): E2 shed OFF = bit-exact (default, proven); ON =
approximate but EXACTLY truncated-attention (principled). The "safe-by-
construction" claim, corrected: the effect type makes it EXPLICIT + ISOLATED +
BUDGETED + PRINCIPLED (well-defined truncated semantics), which beats an ad-hoc
approximation -- but the epsilon bound (dropped mass <= eps) is EARNED by the
drop policy (observation-driven), not automatic. REMAINING to deploy on serving:
weave the shed redirect onto FlashInfer's CP.ASYNC KV loads (it currently targets
ld.global -- same cp.async gap as SHAPE) + the observation-driven drop policy
(drop lowest measured attention mass). Default OFF -> bit-exact ships untouched.

SHAPE DOES NOT SURVIVE EVEN VIA SHARED PREFIX -- HW CAPTURES IT (2026-07-09,
test/py/eval_shared_prefix.py). Hypothesized SHAPE survives via cross-request
shared-prefix reuse (radix-tree-known, moat-enabled). MEASURED: shared-prefix
decode (240 pages re-read 256x + unique tails) is -26% vs a unique batch reading
the same page count -> the HW's L2 LRU ALREADY captures the shared-prefix reuse
(the prefix is re-read 256x = very hot -> LRU keeps it near-optimally). So SHAPE-
pinning has little room. RECONCILE with cache_reuse (-22%): that was INFREQUENT
reuse (32x) drowned by a HUGE streaming flood (2GB) so LRU evicted it between
uses -- CONTRIVED. Shared prefixes are FREQUENT reuse -> LRU holds them. SHAPE
helps only when reuse is infrequent AND the flood is large enough to evict
between uses, which is NOT how FlashInfer's access patterns behave. So SHAPE is
genuinely SUBSUMED on FlashInfer, shared prefixes included. 4th over-optimistic
hypothesis the regime-check discipline corrected. CLC-on-MoE (the other survival
lead) is BLOCKED to test: the grouped GEMM is a precompiled cudnn/cutlass library,
NOT weavable JIT source -- our pass can only weave FlashInfer's JIT kernels, so
CLC's MoE survival is untestable in our framework (honest limitation).

SHAPE HAS NO FLASHINFER APPLICATION -- CORRECTION (2026-07-09). ncu the dominant
prefill kernel: L2 throughput 72% but L2 HIT RATE = 1.07%. So FlashInfer prefill's
L2 is SINGLE-PASS traffic (cp.async global->shared passing through L2), NOT reuse
-- the reuse is captured in SHARED MEMORY by FlashInfer's blocking. SHAPE-bypass
protects L2 REUSE; there is 1% of it here. So SHAPE is FUTILE on FlashInfer
prefill, same as decode, same reason (no re-read to catch). CORRECTION: the
earlier "SHAPE is a 3rd lever for prefill" was PREMATURE -- I proved the lever in
a SYNTHETIC reuse benchmark (-22%) but did not check whether FLASHINFER exposes
L2 reuse. It does not (well-optimized kernels keep reuse in shared). So SHAPE is a
real lever with NO application to FlashInfer's kernels. Applicable in-kernel
levers on FlashInfer remain ORDER + OBSERVE -- the ones ORTHOGONAL to the kernel's
own optimization (cross-tile scheduling + measurement); cache management is
subsumed by FlashInfer's shared-memory blocking.

PDL STAYS OUT -- BENEFIT MOOTED BY OUR OWN DESIGN (2026-07-09). WORK sweep on the
PDL fixture (producer tail 512/4096/16384): PDL vs plain = -5.9/-2.0/-3.4%, but
the SAME WORK=512 read +5.6% in an earlier run -> an ~11% sign-flip = pure NOISE
(idle-GPU timing, not gated), and no trend with tail length. So PDL is NOISE-
LEVEL, no measurable win. And the reason is structural: PDL's benefit is hiding
the consumer's PROLOGUE (control-plane reads) in the producer's tail -- but we
already made the order/timer tables DEVICE-RESIDENT (use_device_order/timer), so
those reads are cheap DEVICE reads, not expensive PCIe. There is almost nothing
left for PDL to hide. So PDL is dead NOT for the engineering reason (the launch
attribute -- the fixture has it) but because our own device-resident design
already captured the latency PDL would overlap. The "engineering unblock" would
reveal no win. Stays OUT (could be re-examined in a real busy chunked-prefill
pipeline, but the cheap-reads reasoning says the ceiling is small).

CLC WIN IS NARROW -- A WAVE-BOUNDARY SWEET SPOT, NOT A REGIME (2026-07-09,
boundary sweep). Interleaved median-of-3 static-vs-CLC across BS at bimodal:
BS=256 +2.1%, 512 -16.4%, 1024 +11.8%, 2048 +2.8%, 4096 +5.9%. CLC wins ONLY at
BS~512 (~= SM-resident capacity, ~one wave) and LOSES at every other BS.
CORRECTION: the clean -17% at BS=512 was real but I over-generalized it to "CLC
revives on outliers." It does NOT -- it is a NARROW wave-boundary sweet spot
(BS ~= resident capacity), not a broad regime. So CLC is NOT a robust third
lever; it is a special-case win at one BS. Honest verdict restored: the robust
in-kernel levers on FlashInfer are ORDER + OBSERVE; CLC is a narrow special case,
SHAPE has no FlashInfer application, PDL is mooted. The STRAGGLER framing (below)
is the better path: detect the straggler (OBSERVE) and remove its makespan impact
by ORDER (schedule first, bit-exact) or E2-budget (shed it, approximate) -- not
by CLC's narrow packing.

CLC REVIVED -- A THIRD LEVER, CLEAN (2026-07-09). INTERLEAVED, drift-controlled
static-vs-CLC at BS=512 bimodal (512x tile-cost spread), 4 rounds alternating
same-BS: CLC beats static grouped-LPT by -18.9/-15.8/-3.6/-21.1% (median ~-17%),
EVERY round. So CLC is NOT dead -- it is REGIME-GATED: it LOSES on mild
dispersion (HW balances, +9% overhead, measured earlier) but WINS ~17% on
EXTREME-OUTLIER batches, where a few heavy tiles dominate the makespan, static
ORDER goes flat (cannot shrink outliers), and CLC's dynamic claim packs the light
work around them. This VALIDATES the existing imbalance-gated CLC arming (arm when
max/mean is extreme -- exactly bimodal; disarm on uniform). THIRD LEVER with a
measured regime. Levers now: ORDER (mild dispersion, regime-robust), OBSERVE
(closed loop), CLC (extreme-outlier regime) -- and OBSERVE measures the dispersion
that SELECTS between ORDER and CLC. (Open: the BS-range of the CLC win -- clean at
512; larger BS was mixed in the cross-process run, needs the interleaved test too.)

CLC HINT IN THE EXTREME-OUTLIER REGIME (2026-07-09, test/py/eval_clc_vs_static.py
SCHED_BIMODAL). Bimodal 512x tile-cost spread (8% heavy @512 pages + 92% light):
static ORDER-gain goes FLAT (+0.1..-4%, vs -12/-15% on lognormal) because a few
heavy OUTLIERS dominate the makespan -- reordering cannot shrink them, so LPT
loses its grip. CLC's order-gain is consistently LARGER (-16% @BS512, where it is
also absolute-faster). So the honest CLC verdict is not "dead" but REGIME-GATED:
mild dispersion -> ORDER handles it, CLC loses; EXTREME outlier heterogeneity ->
ORDER weakens, CLC starts to matter (a HINT, mixed across BS + cross-run noise,
needs an interleaved same-process test to confirm). The one lead where CLC is not
strictly dominated.

SHAPE MECHANISM = BYPASS, NOT A GRADED BUDGET (2026-07-09, test/cache_budget.cu).
Tested whether createpolicy.fractional.L2 gives a CONTINUOUS per-request cache
BUDGET (OBSERVE-measured reuse -> SHAPE fraction, the compute-aware allocation
idea). It does NOT: pin fraction 0.0->1.0 is FLAT (1.42ms R<L2; 1.94ms R>L2,
within noise). Reason: either the reused set FITS L2 (fraction irrelevant once
the streaming flood is bypassed) or it does NOT (cannot fit regardless) -- no
middle regime where a graded fraction pays. So the grounded SHAPE win (-22%,
cache_reuse) comes from the BINARY BYPASS of the one-pass/streaming data
(evict_first), which frees L2 for whatever reused set fits -- NOT from pinning
and NOT from a per-request fractional budget. Compute-aware CACHE ALLOCATION
(more L2 to heavier requests) is therefore NOT grounded; compute-aware ORDER
(schedule by measured cost) IS. Honest: SHAPE = bypass the pollution, one knob.

SHAPE IS ALIVE IN THE L2-REUSE REGIME -- A THIRD LEVER (2026-07-09,
test/cache_reuse.cu). The regime-check discipline paid off: SHAPE is DEAD on
decode (DRAM-bound, no reuse -- cache_hint_bw all-equal) but ALIVE where there is
REUSE + L2 CONTENTION (prefill's regime, ncu-confirmed L2-bound). Measured
(sm_120, R=64MiB reused + S=2GiB streamed, L2=128MiB): default LRU 1.817 ms vs
SHAPE (pin R evict_last + stream S evict_first) 1.419 ms = -21.9%. Mechanism:
the streaming flood evicts the reused data under LRU -> re-reads miss to DRAM;
pinning the reused plane keeps it hot. So the SHAPE lever is REGIME-DEPENDENT,
not globally dead -- it pays exactly when there is a re-read to protect. THIRD
lever, grounded. CAVEAT (do not over-claim): this proves the lever IN the regime
(synthetic reuse+stream); DEPLOYING it on FlashInfer prefill needs the pass to
weave the hints onto the right loads AND prefill to have exploitable L2 reuse not
already captured by shared-memory blocking -- an unverified build/test. But the
LEVER is real. LEVER COUNT (grounded): ORDER (regime-robust), OBSERVE (closed
loop), SHAPE (L2-reuse regime) -> THREE, each with a measured regime.

PREFILL IS A DIFFERENT REGIME -- ORDER IS REGIME-ROBUST (2026-07-09). ncu on
BatchPrefillWithPagedKVCacheKernel (dispersed qo_len): DRAM 0.03%, L2 72%, SM 8%
-> prefill is L2-BOUND (data served from L2 with reuse), the OPPOSITE of decode's
78%-DRAM. And ORDER STILL PAYS (test/py/eval_pi_prefill.py): grouped-LPT -15.2%
@B=48/366tiles, -8.5% @B=256/1873tiles, LPT<id<SPT throughout (SPT +3..+8%). So
grouped-LPT balances per-tile WORK across waves REGARDLESS of the bottleneck
resource (DRAM for decode, L2 for prefill) -- the win is regime-robust, not a
decode artifact. The OBSERVE loop drove the prefill order too (timer-measured
cost). CONTRIBUTION UPGRADE: it is NOT "two levers on one regime" -- it is TWO
regime-robust levers (ORDER, OBSERVE) validated across BOTH the DRAM-bound
(decode) AND L2-bound (prefill) regimes, plus the measured regime->bottleneck
map. The genuinely-new-LEVER bet that is grounded: CLC-CANCELLATION for
speculative decode (CLC's one HW-unique capability -- retract REJECTED-branch
work -- in its native regime; decode had nothing to cancel, spec-decode does).

ncu SETTLES SHAPE + JUSTIFIES MOVE (2026-07-09). Profiled BatchDecodeWithPaged
KVCacheKernel at BS=8192 (GQA-3B): DRAM 78% of peak, L2 36%, SM 38% -> decode is
DRAM-BANDWIDTH-BOUND. CONSEQUENCES, data-driven:
  * SHAPE (L2 residency/prefetch/bypass) is DEAD for decode: cache hints move
    data WITHIN the hierarchy but do NOT reduce DRAM reads, and DRAM is the wall.
    A characterized NEGATIVE (do not build the cp.async SHAPE weave for decode).
  * MOVE (KV tiering / TMA) is THE lever: the ONLY way to speed a DRAM-bound
    kernel is to READ LESS from DRAM. This is eKV's territory (two-plane KV
    split) -- and OUR system does it BETTER: OBSERVATION-DRIVEN. The woven timer
    (and attention mass) measures which KV is HOT; TMA (cp.async.bulk) keeps the
    hot plane in a fast tier, cold in HBM -> fewer DRAM reads. eKV's split is
    STATIC; ours is closed-loop dynamic. THIS is the high-value unbuilt lever and
    the ncu data is its justification.
  * ORDER still pays even DRAM-bound: tiles carry DIFFERENT DRAM traffic (KV
    lengths), so LPT balances the DRAM work across waves (the measured -12%).
REFRAMED CONTRIBUTION (data-justified, not thin): a woven, observation-driven
scheduler that manages BOTH compute ordering (ORDER, grouped-LPT) AND memory
traffic (MOVE, observation-driven KV tiering), sensed by one channel (OBSERVE,
the timer), all effect-safe/bit-exact. SHAPE/CLC/PDL are characterized negatives
that BOUND the design space (each with a measured reason). That is a full-lever
systems result: three levers that pay, three bounded, unified by observation.

LEVER SWEEP -- THE WINS ARE THE HW-LACKING LEVERS (2026-07-09). Systematic verdict
across the instrument menu (SHAPE/OBSERVE/PLACE/COOPERATE/MOVE), measured not
guessed:
  * pi (grouped-LPT, ORDER)      -> WIN (-12% predictable). HW does NOT order tiles
    by cost.
  * timer (OBSERVE) + closed loop -> WIN (95% penalty recovery). HW does NOT
    measure per-request cost.
  * CLC (PLACE/steal)            -> NEGATIVE (HW block scheduler already balances;
    claim only adds ~9%).
  * PDL (COOPERATE/overlap)      -> fixture +5.6% (idle GPU, small kernels, nothing
    to hide) AND blocked in serving: needs cudaLaunchAttributeProgrammaticStream-
    Serialization on the LAUNCH, which FlashInfer/SGLang do not set. Not a flag.
  * policy/cache (SHAPE, L2 residency/prefetch/bypass) -> DEAD for decode,
    FUNDAMENTALLY (not fixably). NOW EVIDENCED (test/cache_hint_bw.cu, sm_120,
    2 GiB single-pass read >> L2 128 MiB): default 1514.8, cs 1513.8, evict_last
    1514.8, evict_first 1532.4 GB/s -- all within 1% -> passive cache hints CANNOT
    move a DRAM-bound single-pass read (the decode KV shape). And discard.L2 =
    53.9 GB/s (28x SLOWER) -> the "polite bypass" is an expensive per-line
    transaction, actively HARMFUL on a hot read (assumption corrected). CORRECTION to an earlier wrong claim: the
    ld.global-vs-cp.async attachment detail is a RED HERRING. ncu shows decode is
    DRAM-BOUND (78%) and the GQA KV reuse is ALREADY captured in shared memory ->
    each KV byte is read from HBM exactly once. Cache hints only move data WITHIN
    the L1/L2/shared hierarchy; NONE reduce DRAM reads. So attaching hints to
    cp.async (the "fix" I proposed) is FUTILE -- there is no re-read to save.
    SHAPE has no headroom on a DRAM-bound kernel regardless of which instruction
    it targets. Struck: "working on the cp.async stream is the fix."
PATTERN (the honest, publishable finding): on modern GPUs the woven scheduler's
value concentrates in the levers the HARDWARE LACKS -- per-request ORDERING and
per-request OBSERVATION -- while load-balancing (CLC), kernel-overlap (PDL), and
cache-management (SHAPE) are largely HW-subsumed or blocked. A lever TAXONOMY
with measured verdicts is itself a contribution (bounds the design space). To
make SHAPE pay on FlashInfer would require weaving cache policy onto CP.ASYNC
(not ld.global) loads -- concrete but uncertain-payoff (narrow regime).

CLOSED-LOOP RECOVERY -- THE VISIBLE WIN + THE CONTRIBUTION, MEASURED (2026-07-09,
test/py/eval_closed_loop.py). What CLC could NOT do, observation does. Real
FlashInfer decode, GQA-3B shape, BS=8192, wave regime:
    oracle (true KV order)          2299 us   baseline
    mispredicted (wrong cost model) 2609 us   +13.5%   wrong-order penalty
    observed (woven timer->reorder) 2315 us    +0.7%   RECOVERED
The woven timer measures true per-tile cost (order-invariant, 8192/8192 tiles),
the estimator reorders by THAT, and ONE observation recovers 95% of the
misprediction penalty -- with NO oracle knowledge. This is the "pays for its own
observation" thesis made visible on the real kernel, and it is the mechanism
that actually rescues a bad schedule (CLC does not -- see below). HEADLINE
CONTRIBUTION, data-backed: a woven, effect-safe, OBSERVATION-DRIVEN scheduler --
(1) the timer measures true cost INSIDE the fused kernel (O-effect, bit-exact),
(2) the estimator corrects the order, (3) grouped-LPT places it (-12% makespan
when predictable). Two visible wins: -12% predictable (grouped-LPT) and 95%
penalty-recovery under misprediction (closed loop). Never slower (levers gated,
bit-exact identity elsewhere).

CLC MISPREDICTION EXPERIMENT -- DECISIVE NEGATIVE, REFRAMES THE CONTRIBUTION
(2026-07-09, test/py/eval_clc_mispredict.py). Tested THE hypothesis that would
make CLC a positive contribution: does the dynamic queue RESCUE a mispredicted
order (the recall<75% regime CLC supposedly serves)? Real decode kernel (MQA 1D
so the claim engages), fill order degraded to recall R, makespan vs each phase's
own oracle:
    recall   static-pi   CLC-queue
    1.00       +0.1%       -0.1%
    0.75      +23.6%      +25.1%
    0.50      +18.8%      +20.0%
    0.25      +25.2%      +27.4%
    0.00      +19.0%      +20.3%
CLC degrades IDENTICALLY to static (and its oracle is already 9% slower: 1328 vs
1217us). CLC does NOT rescue misprediction. WHY (fundamental): the CLC claim
reassigns WHICH SM runs the next tile, but the tile SEQUENCE still follows the
fill order (tile = order[claimed_ctaid]); a scrambled fill -> scrambled claim
sequence -> same wave imbalance. CLC balances OCCUPANCY, it does NOT reorder by
cost. => CLC is not a scheduling contribution on this hardware -- occupancy is
already HW-balanced, and the claim only adds overhead (confirms P2).
REFRAME (the real contribution, now data-backed): the FILL ORDER is what sets
makespan (~20% swing predictable->mispredicted). What CORRECTS a bad order is
not a claim mechanism -- it is OBSERVATION: the woven TIMER measures true per-
tile cost, the estimator relearns the order. So the star is the CLOSED LOOP
(woven timer -> estimator -> grouped-LPT order), the "pays for its own
observation" thesis -- NOT CLC. CLC drops to a documented negative result that
bounds the design space. Levers that carry weight: grouped-LPT (order) +
closed-loop timer/estimator (misprediction rescue) + effect-safe weaving.

KERNEL-RESIDENT ORDER TABLE -- ABI WIRED + BIT-EXACT (2026-07-09). The device-
resident order channel (mirror of the device-timer indirection):
  * Pass (SchedWeave.cpp emitRemap): SCHED_ORDER_INDIRECT -- the arena ORDER word
    is a retargetable POINTER to a DEVICE order tensor; load it, deref; 0 ->
    OOB sentinel -> identity (fail-safe). Implemented with TWO SELECTS, no CFG
    surgery (reading the arena slot as i32 when unarmed is a safe mapped read the
    select discards). Config: Config::orderIndirect; SchedUtil parses it.
  * Runtime (sched_rt.py): use_device_order() allocates an identity device
    tensor + writes its ptr to the ORDER word; set_order installs DEVICE->DEVICE
    (a device-tensor order NEVER leaves the GPU); bake key SCHED_ORDER_INDIRECT,
    cache tag -oi.
  * VERIFIED (test/py/test_device_order.py, 6/6): device-resident; device->device
    install; arena ORDER array untouched; swap-pairs + reversed permutations
    BIT-EXACT (E1); reset->identity fail-safe. Regression: the non-indirect
    DEFAULT path is still bit-exact (test_flashinfer_arm.py) after the pass edit.
  * Plugin: the STATIC production path (was plain per-tile argsort!) now uses
    grouped-LPT via region_order(cost,0) -- the measured Pareto win, finally on
    the shipping path.
HONEST SCOPE (Increment 2, the last mile): full ZERO-sync requires DECOUPLING
the per-step device cost/order from the cadence-gated model FIT. The estimator
is host-side (needs host KV to fit alpha/beta), so today cost is host-computed ->
order is host -> install is host->device. The zero-sync design: compute
cost_dev = alpha*kv_dev + beta + resid_dev + order_dev = grouped_lpt(cost_dev)
ON DEVICE every step (alpha/beta current scalars), and re-FIT the model on the
host only on the probe cadence. The ABI above makes this a pure plugin change;
its E2E payoff is below the server-state noise floor (per P2/enforce A/B), so the
ABI foundation is the shippable deliverable and the decoupling is a clean
follow-up, not a blocker. Levers now: static grouped-LPT (default, on-device-
capable) | CLC (uncertainty-gated specialist) -- one E1 proof covers both.

P2 DECIDED -- DYNAMIC CLC DOES NOT BEAT STATIC GROUPED-LPT (2026-07-09,
test/py/eval_clc_vs_static.py). The fork that gated the megakernel direction,
measured cleanly on a 1D MQA grid (NKV=1, where the CLC claim ENGAGES; 2D GQA
takes the shape-guard stock path). Same batches, SAME grouped-LPT fill, only the
DRAIN differs (static-pi vs CLC dynamic try_cancel). ABSOLUTE ordered step us
(lower wins):
    BS     static-pi   CLC-queue   CLC penalty
    512      135.5u      177.2u      +31%
    2048     528.0u      588.4u      +11%
    4096     863.4u      961.2u      +11%
    8192    1200.5u     1358.1u      +13%
Static wins at EVERY size. RIGOR: absolute times are cross-PROCESS (each weave is
its own compile/cache), so the +11..31% gap carries cross-run clock-drift risk;
the DRIFT-INVARIANT metric is the within-run order-gain, and there CLC == static
(-14.5% vs -15.2% @8192): the dynamic drain adds NO order-benefit over the static
order. The CLC kernel is also architecturally heavier (persistent loop +
try_cancel claim per tile -> baseline overhead visible at identity, 1588 vs
1416u), which the claim does NOT recover because the HW's own greedy CTA
assignment already balances a well-predicted order. Net: no upside, structural
downside. Direct confirmation of the documented CLC law (pays only under SEVERE
order breakdown, recall <75%). CONSEQUENCES for the plan:
  * The megakernel/CLC-as-default (was "P3 high-risk") is DE-SCOPED -- NOT worth
    building; static grouped-LPT is the production lever.
  * CLC stays as-is: the uncertainty-GATED fallback for the mispredicted-cost
    regime (the controller already arms it only when recall is low) -- correct
    and already wired; it is a SPECIALIST, not the default.
  * NOTE (real limitation, honest): CLC currently engages only on 1D grids; GQA
    serving (2D) runs static grouped-LPT regardless. A 2D CLC extension is NOT
    justified by this data (CLC loses even where it engages).
This is the plan SIMPLIFYING: "make it perfect" here means DON'T over-build --
the cheap static lever wins, the dynamic mechanism is a gated specialist.

GROUPED-LPT LANDED (2026-07-09, test/py/eval_pi_grouped.py) -- locality-preserving
refinement, Pareto-beats per-tile LPT, ZERO kernel change (control-plane order,
still E1/bit-exact). Chunk tiles into blocks of B adjacent indices, LPT-order the
BLOCKS, keep original order within a block -> adjacent CTAs keep adjacent KV (L2
reuse) while blocks load-balance. Measured (stable across re-runs, sm_120):
    BS      per-tile LPT (B=1)   grouped B=16   grouped B=32
    2048       ~0%                 -2.5%          -2.8%
    8192      -10.4%              -11.0%         -10.2%
B=16 ERASES the mid-batch L2-scramble penalty AND keeps the large-batch win ->
strict improvement. Wired: SchedPlane._grouped_lpt + region_order static path
(SCHED_LPT_BLOCK, default 16); the CLC region-aware path keeps true per-tile LPT.
Suite green. This is the "smaller, lower-risk, attacks the mid-range penalty"
lever; static per-tile LPT stays as B=1 (the discipline's simplest E1 transform).

NORTH STAR + PHASED PLAN (the lever FAMILY, not one number):
  P0 DONE: bit-exact E2E, exact per-request, prefill wired+validated, isolated
     lever measured (pi -12% @8192, LPT<id<SPT signature).
  P1 DONE: grouped-LPT (locality-preserving, Pareto win).
  P2 NEXT (low-risk, decisive): isolated static-grouped-LPT vs CLC-queue sweep --
     does the DYNAMIC queue's marginal gain over grouped-LPT justify its atomic-
     claim overhead? (grouped-LPT already ate the cheap locality win, so this
     bounds the dynamic ceiling honestly.)
  P3 if P2+: SCHED_DISPATCH={static|persistent|clc} switch, arch-default clc on
     sm_100+ (Blackwell) / persistent on sm_70-90 / static for large-BS; move the
     cost table KERNEL-RESIDENT (zero host per-step -> removes the plan_every
     band-aid, scheduling every step for free).
  P4 PLT: linearity correctness theorem (each tile claimed exactly once => E1,
     covers ALL dispatch modes uniformly); DECLARED capability contract (compile-
     checked, kills the __CUDACC_VER_MAJOR__-style pattern-match fragility).
  P5 honest E2E: restart-INTERLEAVED serving harness (kills server-state drift)
     -> report the realization gap NEXT TO the isolated ceiling.

PI MAKESPAN -- CLEAN ISOLATED VERDICT (2026-07-09, test/py/eval_pi_makespan.py).
The decisive measurement the serving A/B could not grant: ONE fixed dispersed
decode batch, woven kernel, ONLY task_order differs (identity vs LPT vs SPT),
device timer OFF (timing makespan). Reproduces within +-0.5% across runs:
    BS     LPT-vs-identity   SPT-vs-identity
    256      -0.1% (1 wave)     -0.1%
    512      -2.8%              +0.1%
    1024     +2.2%              +4.2%
    2048     +3.6%              +7.3%
    4096     -4.8%              +1.7%
    8192    -12.2%              -2.6%
FINDINGS: (1) pi (LPT) reduces decode makespan in the WAVE-SERIALIZED regime,
-5%..-12%, and the benefit GROWS with batch size (more CTAs than SM slots -> more
waves -> more list-scheduling room). (2) The signature is textbook LPT:
LPT <= identity <= SPT everywhere; SPT (shortest-first) is often WORSE than no
reorder (+7.3% @2048) -- proving it is the LPT ORDER that pays, not reorder
noise. (3) Mid-range (1024-2048) is a stable TRANSITION: below the wave
threshold the makespan win is ~0 but the order indirection + packing disruption
costs a few %, so pi should be GATED to BS above the crossover (~4096 here). This
is the honest, decisive lever measurement: pi HAS teeth (double-digit at serving-
scale batches), it is just invisible through the serving loop's server-state
noise. The crossover BS and the LPT<id<SPT spread are the paper's core plot.

ENFORCE-MODE LIVE A/B (2026-07-09) -- mechanism works; E2E signal below the
server-state noise floor (CONFIRMS the "E2E ~80% noise" thesis). Real Qwen2.5-3B
server, decode-woven, radix ON, unique random input_ids (radix-neutral: cache
stays on -- production-realistic -- but never HITS, so the scheduler is what's
measured), lognormal-dispersed load, fixed seed (observe & enforce see the
IDENTICAL request set).
* enforce mode BOOTS + runs live (mode=enforce, pi reorder active, 0 errors,
  correct) -- the closed loop is production-posture end to end.
* conc=256, disp 3.0x: enforce ~= observe (224 vs 213 req/s; within run noise).
* conc=512, disp 4.2x: enforce 150-153 req/s (stable) vs observe 74-108
  (high variance) LOOKED like a win -- but observe THROUGHPUT DEGRADES
  MONOTONICALLY across identical repeats (79->60->50->46->...->18 req/s), i.e.
  server-state accumulation (radix-tree growth + KV/memory pressure over
  thousands of unique prefixes), NOT pi. Total swing 18..224 req/s (12x) is
  driven by boot-freshness/cache state, dwarfing any scheduling effect.
CONCLUSION: on this stack the E2E throughput of pi cannot be isolated above the
server-state noise (drift >> signal) -- exactly the documented reason the
TRUSTWORTHY levers are the ISOLATED microbenches: pi is memory-cost-free (ncu,
both shapes), timer overhead +0.59% at 1-in-8, and pi is bit-exact in live
serving. A clean E2E verdict would need per-run server RESTARTS (kill drift) +
INTERLEAVED observe/enforce + many repeats + a statistical test -- ~20+ boots;
deferred as its own rigorous harness, not a casual A/B. The contribution rests
on the mechanism + honest characterization, not an E2E throughput headline the
environment cannot grant.
Note: radix stays ON throughout (production requires it); the A/B is radix-
NEUTRAL by construction (unique prompts), not radix-OFF.



Date: 2026-07-04. Target: OSDI '27 (abstract ~Nov 2026, paper ~Dec 2026),
with the system production-usable independently of the paper.

Status baseline: 23/23 suite green (run_all.sh, 2026-07-09 -- 21 core + woven
device-order (SCHED_ORDER_INDIRECT bit-exact) + woven prefill (bit-exact pi +
exact per-request fold), now all in the CI gate; serving-gate woven==stock
8-token identity passes). The whole session's build -- prefill weave, kernel-
resident device order, grouped-LPT, the request<->tile binding, the shed lever --
is integrated and green. MoE cap is demonstrated PLAN-LEVEL (+62%->60% recovery);
its LIVE wireup is a NEW MoE serving hook (below), the next integration. Mechanism complete and bit-exact on real
FlashInfer (π/timer/CLC work-queue, dual ABI, fixed-VA JIT cache, per-step
arming via `num_tasks`, sampled observation via `ctrl.flags`, uncertainty-
calibrated CLC arming, 2D-grid soundness guard). What follows is ordered by
**decision value**: the earliest items change what the later items should be.

The thesis to defend: *per-request scheduling woven INSIDE fused kernels,
bit-exact by construction, driven by a closed loop that pays for its own
observation.* Contribution stack: (1) the effect-type discipline + capability
abstraction, (2) the weaving mechanism (dual ABI, fixed-VA cache contract),
(3) the regime-gated closed-loop controller, (4) the honest hardware
characterization (CLC contract, late-binding law, noise-calibrated arming).

---

## Workstream A — the motivating measurement (DECIDES THE PAPER; do first)

**A1. Residual tile-time dispersion under FlashInfer split_kv. — DONE
(2026-07-04; script since CONSOLIDATED into `test/py/eval_motivation.py`,
which archives the sweep to `data/mot_dispersion.csv` + plots it). VERDICT:
π survives; the threat INVERTS.** Measured on the real woven BatchDecode
(sm_120, nkv=1,
tile-level woven timer, bs ∈ {256..2048}, four mixes):

```text
bs=256  (split ACTIVE, chunk 128..1024 tok, tiles 320..376):
  split flattens request dispersion 7.16 -> 1.36 p99/p50 (sharegpt-ish),
  6.49 -> 3.11 (25%x32: chunking BOUNDS the max chunk, it does not equalize
  tiles -- 64-tok short tiles vs 1024-tok long chunks coexist).
  BUT tiles < R: one wave, nothing queues -- pi has no role here anyway.
bs>=512 (split SELF-DISABLES: tiles == bs, chunk == 0):
  the planner splits only to CREATE parallelism for small grids, not to
  balance big ones. Full request dispersion lands on the tiles:
  p99/p50 = 5.44..16.0 across realistic mixes.
bs=2048 (> R: the QUEUED regime, pi's regime):
  dispersion 5.8 (sharegpt-ish) / 6.8 (25%x32), max/mean 3.2..4.9.
```

So split_kv and π are complementary BY THE PLANNER'S OWN LOGIC: split_kv
owns tiles≪R (parallelism creation), π owns tiles>R (queue ordering); they
never compete in-regime, and at serving scale the straggler dispersion π
needs is fully present (5.8–6.8x). Bonus finding: a GAP REGIME (~512 ≤ tiles
≤ R) where split has turned off but nothing queues — the straggler is fully
exposed and only more splitting would help; FlashInfer's split threshold
looks miscalibrated there (should split until tiles ≥ R). A concrete,
tool-discovered improvement to report upstream. Also documented:
`disable_split_kv` is silently ignored on the non-TC decode path in 0.6.x —
the unsplit baseline must be computed by request-level tile aggregation.
*Decision taken:* π-over-tiles is the headline; A2 reframes (below).

**A2 (reframed by A1). Closed-loop π on the real kernel in its own regime.
— HEADLINE MEASURED (2026-07-04; script since CONSOLIDATED into
`test/py/eval_motivation.py`, part C -> `data/mot_policy.csv`; re-measured
2026-07-08 at -35.4% closed-loop, 88% straggler recall, bit-exact).** Real
BatchDecode, bs=2048 tiles (sharegpt-ish mix, the QUEUED regime), production
observation mode (timer gated OFF on serving steps, ON for one probe step),
30 steps/policy, ALL bit-exact:

```text
identity   425.6 us/step      --
reversed   417.2 us/step    -2.0%
lpt-kvlen  284.2 us/step   -33.2%   (deployable length-oracle)
lpt-timer  282.3 us/step   -33.7%   (CLOSED LOOP, no oracle; probe recall 95%)
probe step (timer ON): +38.2% on 1-in-N steps
  -> amortized net: ~-29% at N=8, ~-32.5% at N=32
```

The closed loop slightly BEATS the length oracle (cycles capture true cost,
not just length). ~3x the synthetic eval_trace result (-12.3%) because real
dispersion at serving scale (5.8x p99/p50) has more tail to reclaim. The
+38% probe cost concretely motivates E1 (device-buffer timer).
*Remaining in A2:* the regime-map figure (split / gap / queue) and a
multi-step closed-loop run (order drift under batch churn) — then this folds
into B3's serving table.

## Workstream B — end-to-end serving evaluation (THE SHIP GATE)

**B1. Live woven SGLang server. — LIVE (2026-07-07).** sglang 0.5.14 +
JackFram/llama-160m (MHA -> the validated non-TC decode path), all four
signals green on one boot: temp-0 output token-identical to the stock boot;
the `va...-n4096-ti` woven cache (baked canonical addresses + device-timer
layout); 3 batch_decode compiles WITH -fpass-plugin; and
`R resolved from woven kernel: 752` -- the auto-R wiring live in the loop.
First serving numbers (observe-only, full observation stack: 1-in-8 probes,
device timer, event-based reads): 88.7 req/s, 8531 tok/s, TPOT p50 7.09 ms.

The bring-up flushed out and fixed, in order: __ffma2_rn prelude gap; CCCL
deduction-guide + IntFastDiv patches (norm compiles under clang); the
SCHED_WEAVE_ONLY = weave-decode-only semantics (also prevents rope/norm CTAs
polluting the decode timer rows); fake CUDA_HOME + nvcc version-banner
impersonation (SGLang's tvm-ffi JIT bypasses FLASHINFER_NVCC and parses
`nvcc --version`); sitecustomize exclusion via SCHED_SITE_OFF (importing
torch inside the shim deadlocked the version probe; positive argv detection
is unreliable at site time and silently disarmed everything -- exclusion
with an env marker at the compile entrypoints is the robust semantics);
IMPORT-TIME env pre-arming from the PREDICTED canonical VA (FlashInfer
freezes its workspace dir as a module constant at import -- batch-time
arming silently reused stock kernels; the plane verifies the prediction on
creation and re-arms with the actual base on a canonical miss:
correct-or-recompile, never wrong); and scripts/start_server_detached.sh
(SSH-drop-proof launches, stale-log-proof health probing).
**B2/B3 first config — MEASURED (2026-07-07, `scripts/bench_ab.sh`,
results in `bench_ab_results3.txt`).** llama-160m, random 128/128, n=300,
conc=64, same boot, `--disable-cuda-graph`:

```text
config             req/s   tok/s   TPOT p50   TTFT p50
stock (no plugin)  81.62   7846    7.72 ms    20.22 ms
woven observe      79.09   7603    7.79 ms    20.39 ms
woven ENFORCE      81.35   7821    7.68 ms    19.82 ms
```

ENFORCE vs stock is a statistical tie (run-to-run spread +-3-4% exceeds all
config deltas): the full stack -- woven kernels, device-timer probes,
event-based reads, damped LPT, arming decisions -- costs ~ZERO under real
continuous batching. No pi GAIN is expected or claimed at 128-token KV on a
160M model (no straggler tail); this config banks the overhead + stability
claims. The bench flushed one more fail-safe hole, now closed structurally:
the pre-import-fix boots had written a BAKED kernel into the DEFAULT
FlashInfer cache (env-ordering window), which stock mode then dereferenced
-> cudaErrorIllegalAddress. The shim now enforces the invariant "bake vars
without a va-keyed workspace -> strip and compile unwoven, loudly"; the
poisoned artifact was purged.

**B3 queued-regime round (2026-07-07, `bench_queued_results.txt`):**
conc=2048 tiles > R(752) -- the first E2E config where pi has a queue.
llama-160m, in 64..256, out 128, n=4000:

```text
config     req/s    tok/s     TPOT p50   TPOT p99   TTFT p50
stock      252.4    20316     44.6 ms    86.6 ms    3197 ms
observe    256.9    20677     54.0 ms    73.3 ms    2428 ms
enforce    256.9    20681     47.1 ms    83.7 ms    3067 ms
```

Read HONESTLY: throughput woven rows +1.8% over stock (likely noise);
latency medians/p99 swing 10-25% BETWEEN IDENTICAL-CONFIG runs at this
depth (admission churn, TTFT ~3 s dominates) -- no pi latency claim can be
made from single runs here. The structural reason pi cannot show through on
this model: at 27 ms ITL the decode-attention slice of a 160M step is ~1-2
ms even at 2048 tiles; pi's -33.7% of that slice is invisible in E2E noise.
CONFIRMED expectation, not a failure: the E2E GAIN row requires an
attention-dominant config.

**RESOLVED — NOT A BUG (2026-07-07, two bisect campaigns + determinism
micro-test): the "GQA corruption" was SAMPLING NOISE.** Five identical
`temperature:0` requests to the SAME booted 3B server produced five
different outputs (even with top_k=1): SGLang's `sampling_defaults='model'`
merges the Qwen generation_config (temp/top_p) OVER the request, so every
"coherent vs garbage" text verdict in both bisect ladders was a per-launch
coin flip. The weave, the plugin, and the bootstrap are all exonerated --
consistent with every in-process gate (3 repros, 8-variant census, 48-step
sequence) being bit-exact throughout. Real-greedy serving needs
`--sampling-defaults openai` (+ fixed seed for cross-boot compares).

THE RULE THIS BUYS (now policy): serving correctness gates are PINNED-GREEDY
TOKEN-ID COMPARISONS against a stock reference -- never text-coherence
heuristics. (The 160m validations used token identity and were never wrong;
the kernel gates use bit-exactness and were never wrong. The one ad-hoc
coherence check in the 3B smoke cost two bisect campaigns.) Positive
residue: the lazy sitecustomize (no interpreter-start imports -- correct
engineering regardless), the disciplined one-script/env-delta ladder
(`scripts/bisect_ladder.sh`), the lever-mask cache keys, and the env guards
(SCHED_HOOKS_NOOP / SCHED_NO_INIT_HOOK / SCHED_SITE_REGNOOP) as permanent
debug instrumentation.

**Superseded record of the hunt (kept for methodology honesty):**
Qwen2.5-3B (16 qo / 2 kv = group 8, hd 128, classic non-TC decode forced via
SGLANG_FLASHINFER_USE_TENSOR_CORE=false):

```text
clang UNWOVEN (stock mode):      coherent ("...speed of light is constant")
woven, ANY lever subset:         fluent English, WRONG continuation --
  full / no-shed / no-shed-no-policy / timer-only / pi-only ALL corrupt.
```

Since timer-only and pi-only are output-neutral BY TYPE and each corrupts,
the fault is not a capability's semantics: something the weave does to this
kernel variant structurally (entry-block split/PHI on ITS ctaid usage,
baked-slot global interaction, or a clang codegen sensitivity the pass
tickles) breaks it. The 160m (group 1) serving path is token-identical and
fully gated -- variant-specific. The fluent-but-wrong-prompt signature
suggests the kernel reads wrong KV/positions, not garbage math.

QUARANTINE: 3B/GQA serving stays UNWOVEN (SCHED_SITE_OFF=1) until
root-caused.

DEBUG ITERATION 1 (2026-07-07, `test/py/test_gqa_repro.py`): the kernel weave
ALONE is EXONERATED -- one-shot bs=8, full weave, both group1-control and
the exact group8/hd128 3B shape are BIT-EXACT vs stock outside SGLang. So
the corruption REQUIRES serving context. Remaining deltas, ranked: multi-
step decode with per-step re-plan at churning batch sizes (the small-batch
split regime), the plugin's per-step control writes, the device-timer (-ti)
build (iteration 1 compiled non-ti), radix-cache page patterns, and the
overlap scheduler. ITERATION 2 (`test/py/test_gqa_repro_steps.py`) adds all
of the first three in-process (48 steps, bs churn 2/4/8, plugin-equivalent
writes, -ti build, sequence bit-compare); `test/py/census_variants.py` maps
one-shot exactness + detector notes across group x head-dim (D2). If
iteration 2 is exact too, the fault lives in SGLang-only state (radix
reuse / overlap) -> instrument the serving path directly next. The cache
key now includes the lever mask (`-no{itps}`), added for this bisect.

**SERVING-CORRECTNESS SAGA — FINAL STATE (2026-07-07, `env_bisect.txt`,
`bisect_ladder.txt`, `clean_3b_results.txt`):**
ROOT CAUSE of the wrong-topic generations: `CUDA_HOME` repointed at the
fakecuda wrapper dir corrupts TRITON's runtime-compiled GQA prefill
(confirmed by single-delta: pure upstream + CUDA_HOME alone reproduces;
even a COMPLETE symlink overlay corrupts; ~/.triton caches the poison
across boots). FIXED: the serve script now routes tvm-ffi's nvcc via
PATH-front wrapper and UNSETS CUDA_HOME/CUDA_PATH; pure-upstream Triton
discovery preserved (verified deterministic-and-correct at the pure level).
Secondary confounds found on the way: SGLang `sampling_defaults='model'`
makes temperature:0 non-greedy (gates must pin `--sampling-defaults
openai`); the Veri-R1 3B checkpoint has degenerate near-tie logits (swapped
for Qwen2.5-3B-Instruct).
REMAINING UPSTREAM LIMIT: this SGLang stack is not batch-invariant -- even
true-greedy single requests flip a near-tie token within ~12-30 tokens run
to run, and `--enable-deterministic-inference` does not boot on this
configuration. So serving-level BITWISE gates are capped at short-prefix
token-ID comparison (~8 tokens) or logprob/top-1 comparison per step; the
authoritative correctness gates remain the in-process kernel-level ones
(bit-exact throughout, never wavered). Next session: 8-token-ID woven-vs-
stock gate, then the headline bench.
ENFORCE REGIME GATE (implemented): un-gated pi ordering cost ~14% at 3B
conc-256 (one wave, tiles <= R -- ordering provably useless there; observe
== stock confirmed the delta was the estimator/sort python). The plugin now
skips estimation+ordering entirely when R > 0 and ntiles <= R (THEORY S9's
gate applied to pi itself). Consequence for experiment design: on 96GB,
tiles > R AND attention-dominant long-KV cannot coexist at 3B scale -- the
E2E queue-regime gain row must be sized (~1B model or moderate KV); the
kernel-level -33.7% (A2) remains the demonstrated headroom.

**RESOLVED (2026-07-07/08): the "GQA-3B woven serving discrepancy" was
CROSS-BOOT ENVIRONMENTAL SENSITIVITY, not a weave defect.** The decisive
observations: (a) the woven divergence appears at TOKEN 1 -- produced by
PREFILL's logits before any woven decode kernel runs (the decode weave is
structurally alibied); (b) STOCK ITSELF gives different pinned-seed answers
across boots ('a city in the state of New York' vs 'a key component of the
overall system.'), each boot internally deterministic. On this
non-batch-invariant stack, the 3B's near-tie raw-completion logits flip on
ANY boot-to-boot difference; a woven boot is simply a different boot. The
serving gate now runs a SECOND STOCK BOOT as a control and SKIPS LOUDLY
when stock disagrees with itself (cross-boot token identity ungrantable;
weave correctness rests on the in-process bit-exact gates, which never
wavered). The 160m is boot-stable and its WOVEN==STOCK identity is a real,
passing verdict. Also hardened on the way: serving compiles now weave the
MINIMAL surface (pi+timer; shed/policy off by default -- shed carries the
known host-quarantined codegen issue and the plugin never uses either).

**Superseded initial report:** (stock stable at 4 tokens; woven
NON-deterministic and different). First clean-env test of the woven GQA
serving path post-fakecuda-fix; in-process GQA gates remain bit-exact, so
the delta is serving-context-specific (plugin-boot vs stock-boot under the
one-script rule: sitecustomize registration + woven decode kernel + device
timer channel + radix-cache interaction are the remaining ingredients).
Next session: run `scripts/bisect_ladder.sh` semantics against THIS gate
(4-token identity as the verdict, not coherence). Also fixed this round:
the queued-regime enforce stall -- the controller's python predict+sort
over 2048 tiles per step stalled the scheduler thread (bench died twice);
ordering hot path now VECTORIZED (torch argsort over tensorized
t_hat = alpha*kv + beta + resid), estimation stays in the controller,
churn damping retired (ordering now ~50us).

**FIXED (2026-07-07): enforce-mode table-write RACE under the overlap
scheduler (cudaErrorIllegalAddress).** The woven kernel reads the order
table at EXECUTION time; SGLang's overlap loop calls run_batch while the
previous step is still on the GPU, so a begin-hook set_order sized for the
NEW batch can send the IN-FLIGHT launch out of its arrays when the batch
grew/shrank. (Observe mode never writes -> unaffected; one-wave enforce was
accidentally shielded by the regime gate -- no writes.) Fix: an `inflight`
CUDA event recorded after EVERY step; order installs only when (a) the
event has fired and (b) tile count matches the installed permutation's
size; on size change the table retires to identity while idle. pi therefore
engages in stable same-size stretches -- exactly where ordering pays.
STRUCTURAL FOLLOW-UP for D1: kernel-side per-launch bound on the mapped
task (mapped < nctaid.slot ? mapped : pid) so a raced write can never fault
-- with the write-gating keeping permutation bijectivity (the guard alone
would trade a crash for duplicate/missing tiles).

**DEAD-CODE / UNWIRED AUDIT (2026-07-08).** Removed as genuinely dead
(zero callers, confirmed): `emitShed` single-site wrapper + its stale
address-redirect comment (superseded by `emitShedAll` value/score masking);
the never-implemented importance table (`kImpSym`/`__sched_imp`/`bakeImp`/
`SCHED_BAKE_IMP`/bakedAddr "imp"); `SchedPlane.addrs()` (superseded by
`compute_bake_env`); `SchedPlane.r_for_cufunction` (redundant with
`r_for_cached_so`). Kept + DOCUMENTED as intentionally-unwired seams (real
control surfaces, not bugs): `SchedControlPlane.observe_step`/`gamma` (the
lambda-pricing congestion half, THEORY 'the roof'); `set_lambda` (sigma
shadow prices -- host-fixture-only until cp.async detection lands);
`reset_queue` (Python peer of the C ticket API, for sm<100/worker-pool
serving). Verified: IR gate green, all fixture modes bit-exact, imports
clean, no dangling references.

**EXTERNAL REVIEW RESPONSE (2026-07-08).** Fixed the concrete bugs an
outside code review surfaced (all verified real against source first):
- disarm/re-arm: `sched_rt_disarm` now clears `armed`; `sched_rt_init`
  re-arms without re-allocating (buffers persist -> no leak).
- lever independence: `SCHED_NO_POLICY` no longer disables shed -- LoopInfo/
  SCEV and stream-site detection now run when policy OR shed is on, emission
  still gated per lever (verified: shed weaves under NO_POLICY; paged fixture
  14/14 bit-exact).
- `register()` idempotent (`_STATE['registered']` guard) -- no double-install
  via sitecustomize + entry point.
- hook errors: log the FIRST per site ALWAYS + a `hook_errors` counter in the
  stats line (was silent unless SCHED_DEBUG) -- degraded control plane is now
  visible; still never crashes serving.
- one-wave gate retires a stale order (reset_order + order_size=0) instead of
  leaving a prior queued step's permutation honored.
- `SCHED_MAX_TASKS` clamped (bad/negative no longer casts to a huge unsigned).
- effective-config dump at registration (resolved mode/levers/channel) --
  operators can read the active mode at a glance.
- doc drift fixed (plugin docstring: both halves live; README: CLC is real,
  not a stub).
- repo hygiene: real git repo; transient run logs moved to `results/`
  (gitignored) so stale artifacts can't be misread as current state -- which
  is exactly what happened (the review's "GQA serving broken" headline cited
  `suite_final.txt`/`p0_finish.txt`, both superseded by the cross-boot-control
  resolution: the current serving gate is ALL PASS with 3B skipped as
  boot-non-deterministic); `requirements.txt` pins flashinfer 0.6.12 /
  sglang 0.5.14 (the private layouts we read).

Still-open review points (agreed): typed-config-object over the broad env
surface (env stays the compile wire format; a Python dataclass becomes the
single source of truth -- overlaps D1); more IR fixtures (WQ/PDL/shed-only/
grid-guard); CI packaging (pyproject, GPU+CPU lanes); and the E2E pi latency
GAIN in the queued attention-dominant regime (the one real perf claim still
to land).

**DONE (2026-07-08): repo reorg + D1 capability manifest.**
- Test/eval code separated from library: `python/` is now runtime-library
  only (sched_rt, sched_controller, sched_sglang_plugin, sitecustomize,
  nvcc_clang_shim, patch_flashinfer, kernels/); all test_*/eval_*/plot/census
  moved to `test/py/` (path-bootstrapped to find the library; run_all.sh +
  scripts updated). Full suite re-run on the new layout: **20/20 PASS**
  (confirms both the reorg AND the review bug-fixes end to end).
- D1: `include/sched/SchedManifest.h` -- one declarative row per instrument
  (effect type, arch gate, slots, compile knob, pass/emit order, cache tag,
  contract), the single source of truth. `dumpManifest()` prints it at
  plugin-load (`SCHED_MANIFEST_DUMP[=csv]`) for operators/census/docs;
  `python/sched_rt.py::MANIFEST` mirrors it and `test/py/test_manifest.py`
  asserts C++<->Python consistency + order-monotonicity (wired into the
  suite, now 21 gates); `MANIFEST.md` is generated by
  `scripts/gen_manifest_doc.sh` (can't drift). Order-invariants learned this
  project (WQ-before-weave, shed-before-prefetch, claim-after-body, pi
  bijectivity, 1D-grid) are now DATA in the manifest, not comments. Scope
  note: Config stays the env snapshot and the cache-tag logic is unchanged
  (both tested); the manifest is the STRUCTURE over them -- fully deriving
  Config/cache-key from the manifest is a safe future refinement.

**EXTERNAL REVIEW ROUND 2 (2026-07-08) -- all 4 findings fixed (verified):**
- F1 (real, gate broke): `run_all.sh` derived `opt` from `command -v` -> a
  foreign LLVM's opt loaded nothing -> empty manifest dump -> gate FAIL
  (reviewer saw 16/1). Fixed: derive `LLVM_BIN` from `LLVM_DIR` (same LLVM the
  plugin builds against). `SKIP_FLASHINFER=1` suite now **17/0**.
- F2 (real, our regression): the one-wave stale-order RETIRE was dead code --
  `_STATE['order_n']` was checked/cleared but never SET after either
  `install_order()`. Now set at both install sites, so same-size stale
  permutations are actually retired to identity in one-wave steps.
- F3 (real, latent host-app): pass env `SCHED_MAX_TASKS` vs runtime macro can
  diverge (the macro sizes the compile-time `SchedCtrl.task[]` array, so it
  must stay compile-time). Documented the ABI contract loudly at BOTH sites
  (default 4096; override both together via `-DSCHED_MAX_TASKS`); host setters
  already clamp writes; runtime publishes its value in the init JSON for
  cross-check. Serving/baked path is unaffected (Python arena is single-source).
- F4 (fair overclaim): the manifest wasn't fully source-of-truth. Now the
  Python cache-key disable-MASK is DERIVED by iterating `MANIFEST` (no second
  knob list -- a new lever keys the cache automatically), the mirror carries
  `tag`, and `test_manifest.py` checks tag + `order == row index` too. Comments
  corrected to state precisely what is DERIVED (docs, cache-mask) vs ASSERTED
  (emit order, checked by the test) vs MIRRORED (C++<->Python row-for-row).

**E2E HEADLINE BLOCKED BY A HOST-TOOLCHAIN CLASH (2026-07-08), not by us.**
Attempted the queued (conc 1024 > R~940) + attention-dominant (in=1536)
Qwen-3B A/B/C. The STOCK server (SCHED_SITE_OFF=1, zero weaving) fails to
boot: long/paged inputs make FlashInfer JIT-compile `batch_prefill_paged/
ragged` kernels, and on THIS node neither compiler can build them:
  * real nvcc 12.9: cudafe rejects glibc 2.41's C23 `sinpi/cospi/sinpif/
    cospif` (declared `noexcept(true)` under `_GNU_SOURCE`, recomputed in
    every libc header so un-suppressable via macro) against CUDA's own
    unspec'd decls -- "exception specification is incompatible". `-ccbin
    clang` doesn't help (the clash is in cudafe, not the host pass); CUDA's
    header is root-owned (a 4-line `noexcept(true)` patch needs sudo).
  * clang-22 (our decode-weaving compiler): independent `prefill.cuh` gaps on
    that same paged bf16 kernel (`__ldca` overload, unresolved overloads at
    2474/1201/1519, non-static-member at 399) -- deeper than the decode path
    patch_flashinfer.py already closes.
SCOPE: entirely orthogonal to sched-pass (fails with the plugin INERT); blocks
LONG-INPUT serving only -- every short-input gate boots and passes, which is
why it stayed hidden. The kernel-level -35% pi result (MOTIVATION.md, woven
decode, bit-exact) is unaffected; the regime it needs at serving scale is a
compiler-availability wall on this box, not a design limit. UNBLOCK: (a) sudo
patch CUDA's crt/math_functions.h to add `noexcept(true)` to the 4 decls -> nvcc
compiles prefill, route prefill via SCHED_REAL_NVCC (already wired in the shim,
just needs a working nvcc); (b) newer CUDA / older glibc; (c) extend clang's
prefill.cuh support. Only then does the queued attention-dominant E2E row run.

**E2E HEADLINE RAN (2026-07-08, after the toolchain unblock) -- NEGATIVE in
this regime, and the reason is instructive.** Qwen-3B, queued (conc 1024 >
R~940) + attention-dominant (in=1536, out=128, ratio 0.25), --disable-cuda-graph,
`results/bench_queued_3b.txt`:

| config | req/s | tok/s | med TPOT | vs stock |
|---|---|---|---|---|
| stock | 31.40 | 2530 | 355 ms | -- |
| observe | 11.46 | 924 | 1138 ms | **-63.5%** |
| enforce | 10.37 | 835 | 1193 ms | -67% |

The kernel-level -35% does NOT survive to E2E here. CRUCIAL: observe ~= enforce
(-63.5% vs -67%), and observe does NO scheduling (bind + timer only) -- so ~86%
of the loss is WOVEN-PATH / per-step CONTROL-PLANE overhead, NOT the pi decision
(pi adds only -9.5% on top). At short inputs / lower conc (B3) this same overhead
was ~0%, so it is regime-dependent and explodes at conc 1024 + long inputs +
no-cuda-graph. Diagnostic in flight (`results/diag_noop.txt`): woven kernel +
NO-OP hook, to split Python-hook cost (fixable: cadence/cuda-graph/lighter bind)
from woven-kernel cost (occupancy/regpressure -- harder). Honest headline
UNCHANGED: kernel-level -35% bit-exact stands; E2E serving win is NOT
demonstrated and, as currently implemented, regresses at scale.

**CADENCE FIX for the -63% overhead (2026-07-08) -- PRIMARY overhead solved,
SECONDARY found.** Root-caused the -63% observe regression to the per-step
control-plane hook doing 3 GPU->CPU reads/step (slot indices, seq_lens,
plan_info) that DRAIN the stream and collapse SGLang's overlap pipeline. Fix:
`SCHED_PLAN_EVERY` cadence (default 8) -- re-plan 1 step in N, off-cadence
steps return immediately (no reads, no push); staleness is safe (order_size
guard retires a stale-size order to identity). Re-measured, this session,
2000 prompts, identical queued config:
| config | req/s | med TPOT | note |
|---|---|---|---|
| observe BEFORE fix | 11.46 | 1138 | the -63% |
| **observe AFTER fix** | **43.83** | **237** | **3.8x recovery -- overhead GONE** |
| enforce BEFORE fix | 10.37 | 1193 | |
| enforce AFTER fix | 13.06 | 942 | improved but STILL slow |
| stock (now / earlier) | 20.46 / 31.40 | 574 | baseline swings -- HIGH variance |

ROBUST conclusion (both observe numbers same-session): the cadence fix
eliminates the per-step-hook overhead. NOT-yet-solved: (1) enforce is still
far below observe (43.83->13.06) -- a SECOND overhead specific to the enforce
path, most likely CLC AUTO-ARMING (num_tasks>0 switches the woven kernel to
the persistent-worker/claim path and PERSISTS across off-cadence steps; test
next with SCHED_SGLANG_CLC=off = pure pi ordering). (2) Absolute win/loss vs
stock is UNRELIABLE from single runs (stock 20-31 req/s across boots --
non-deterministic random workload + cache/clock warmth); a clean A/B needs
fixed-seed identical workloads + repeats + ideally cuda-graph. So: the E2E pi
WIN is still NOT demonstrated; but the catastrophic overhead is understood and
its primary component fixed.

CLC-OFF TEST (the SECOND overhead, confirmed): enforce + `SCHED_SGLANG_CLC=off`
(pure pi ordering, no claim loop) = **28.71 req/s** vs 13.06 with CLC-auto --
a 2.2x recovery. So CLC AUTO-ARMING was indeed the second overhead (at conc
1024 the cold estimator arms the claim loop -> slow persistent-worker path,
persisting across cadence steps). With BOTH overheads out (cadence + CLC-off),
enforce went 10.37 -> 28.71, now COMPETITIVE with stock's 20-31 band.
Remaining gap: observe (43.83) > enforce-CLC-off (28.71) -- pi ORDERING itself
costs ~35% here (install_order H2D + permuted-tile KV access), and this
uniform-dispersion random workload (p99/p50 ~1.5) does NOT repay it. The
kernel-level -35% needs the HIGH (lognormal, ~6x) dispersion the microbench
had. So the honest final state: the two overheads that made it "so bad" are
understood + fixed (enforce -67% -> competitive); the E2E pi WIN is still not
demonstrated, now for a PRINCIPLED reason -- the serving workload lacks the
straggler dispersion pi exploits, and ordering has a real cost. To show a win:
a high-dispersion trace (not uniform random) + fixed-seed repeats + cheaper
ordering (cut the permuted-access cost). Defaults changed: consider
`clc=off` as the serving default (auto-arm hurts at scale).

NCU COUNTER PASS (2026-07-08, `test/py/prof_pi_cost.py`) -- REFUTES the
"pi ordering costs ~35%" hypothesis. Profiled the real woven decode (bs=2048
dispersed) under identity vs reversed task_order:
| metric | identity | reversed | delta |
|---|---|---|---|
| sectors/request (coalescing) | 12.91 | 12.91 | +0.0% |
| L2 hit rate | 1.22% | 1.21% | -0.8% |
| occupancy | 15.1% | 15.0% | -1.0% |
The permutation is MEMORY-COST-FREE at the kernel level: each tile reads its
OWN request's contiguous KV regardless of issue order, so reordering WHICH tile
runs WHEN changes neither coalescing nor L2 nor occupancy. pi is a clean E1
permutation -- reorders WHEN, not WHAT, and (now confirmed) not HOW-memory. The
decode is DRAM-streaming-bound (L2 hit ~1.2%), which pi does not touch (but
residency/sigma hints might). CONSEQUENCE: the earlier E2E enforce(28.71) <
observe(43.83) gap was NOT an ordering cost -- there is none -- it was
measurement NOISE (baseline swung 20-31) + the tiny install_order H2D. So the
path to an E2E pi WIN is purely a HIGH-DISPERSION workload (where the -35%
makespan benefit materializes); there is no ordering-cost to engineer away.

HIGH-DISPERSION E2E (2026-07-09, `test/py/loadgen_dispersed.py` -- controlled
lognormal input lengths, p99/p50=4.7x, FIXED SEED so all configs get the
IDENTICAL request set: the noise-killer the uniform-random A/B lacked).
Qwen-3B, conc 1024 (>R=752), out 128:
| config | req/s | vs stock | server decode tok/s |
|---|---|---|---|
| stock | 67.72 | -- | ~25100 |
| observe (cadence) | 55.92 | -17% | -- |
| enforce +CLC-auto | 26.45 | -61% | ~7500 |
| enforce +CLC-off | 22.81 | -66% | ~7500 |

CLEAN conclusions (identical workload, not noise):
1. The cadence fix cut OBSERVE overhead from -63% to -17% -- real progress,
   but observation still costs 17% (timer/binding on the 1/8 cadence).
2. ENFORCE still regresses ~3x (server decode 25100 -> 7500 tok/s -- a real
   GPU-side drop, not a client artifact), and it is NOT CLC (off is equally
   bad). MECHANISM STILL OPEN -- two suspects RULED OUT, one caveat found:
   - NOT a control-plane sync: install_order/push are host memcpy into the
     UVA arena, no CUDA calls (sched_rt.py:22,263). So the enforce Python path
     does not drain the pipeline via a sync.
   - The ncu "permutation is memory-free" pass was on a 1D nkv=1 MICROBENCH,
     NOT the 2D GQA serving kernel (num_kv_heads=2). It may not transfer --
     the serving-shape kernel's response to a permuted order is UNVERIFIED.
   So the 3x is EITHER (a) the 2D GQA woven kernel actually costs under a
   permuted task_order (needs an ncu pass on the REAL serving shape), OR (b) a
   host-side scheduler-thread stall the cadence doesn't fully remove. An nsys
   timeline of the enforce serving loop (host vs GPU gap) + an ncu pass on the
   nkv=2 kernel would decide. This is THE open blocker; I over-claimed "-35%
   is cost-free" earlier -- it is cost-free on the 1D microbench, unproven on
   the serving kernel.
3. The arm-gate fix (imbalance>3.0 veto) WORKS as designed -- it armed on this
   4.7x-dispersed batch (correctly, per the win-regime) and stayed off on
   uniform. But it is MOOT here: CLC is gated off on the 2D GQA decode grid, so
   arming only adds driver overhead. => on the real 2D decode, CLC cannot help
   until multi-axis claiming exists; `CLC=off` remains right FOR NOW (user
   wants it on -- honest answer: it is dormant regardless, so off avoids its
   arming cost until multi-axis lands).

NET: pi's kernel benefit is real + cost-free (ncu -35%, clean); CLC and the
kernel are exonerated; the E2E win is blocked SOLELY by the enforce-path
per-step cost (install_order sync suspected). That is now the ONE thing between
the -35% and an E2E win. Deliverable banked: a clean fixed-seed high-dispersion
load harness that finally makes serving A/Bs reproducible.

NCU ON THE REAL GQA SERVING SHAPE (2026-07-09, corrected) -- the earlier
"cost-free" pass was on a 1D nkv=1 microbench; re-ran on the 2D GQA kernel
(BatchDecodeWithPagedKVCacheKernel, nqo=16/nkv=2, group 8). Result UNCHANGED:
sectors/req 13.09 identical, L2 0.85% identical, occupancy 22.3% identical
under identity vs reversed. So permutation IS memory-free on the real serving
kernel too => the E2E enforce 3x is HOST-SIDE (GPU idle waiting on the
scheduler-thread hook), definitively not the kernel. nsys would only visualize
that idle gap.

EXTERNAL REVIEW ROUND 3 (2026-07-09) -- 6 findings, all verified real; fixed 5,
1 documented:
- #5 FIXED: tile-binding used a BYTE offset (v[3]) to slice/bound
  `_int_workspace_buffer` by ELEMENT -- wrong unless the buffer is uint8. Now
  bounds in bytes (numel*element_size) and slices via a uint8 view (correct for
  any dtype) + a one-time dtype log.
- #4 FIXED: controller `_span` was order-INDEPENDENT (max(t_hat)+k*1e-9), so the
  hysteresis dead-band never fired -- the class was silently frozen (serving
  sidesteps it via inflight-event gating). Now total completion time (sum of
  prefix sums), genuinely order-dependent.
- #6 FIXED: patch_flashinfer.py `[skip]` (neither pre/post pattern found =
  clang-vs-nvcc gap UNCLOSED) now `[DRIFT]` and exits nonzero -- a CI failure,
  not a silent success.
- #3 FIXED: `discard.global.L2 [a],128` requires 128B alignment (PTX ISA);
  emitted on the raw load ptr (UB). Now aligns down to the L2 line. Bit-exact
  for READ-ONLY KV (DRAM authoritative, no writeback); documented the GQA
  KV-sharing perf caveat (kept behind the conservative Polite flag).
- #2 FIXED (the specific race): CLC claim FALLBACK reloaded ctrl->num_tasks
  fresh while the loop terminates on the entry SNAPSHOT -> desync if the host
  rewrites num_tasks mid-launch. Threaded the entry Ntasks SSA into
  emitClaim->emitClaimCLC. (The order-size-guarded remap keeps its reload as
  defense-in-depth; bijectivity is guaranteed by the order_size header.)
  Verified: CLC fixture bit-exact, -11.1%.
- #1 FIXED (reconsidered -- it was worse than "scale only"): the timer stayed
  enabled across off-cadence steps and the buffer is cleared only on probe
  steps, so a probe accumulated ~plan_every steps AND each request accumulated
  only for the steps it was PRESENT -> under batch churn the per-request cost is
  distorted by presence-duration, which can CORRUPT the LPT ranking pi depends
  on (not a harmless scale factor). Fixed WITHOUT the heavy ring-buffer: LATCH
  the timer-off flag at kernel ENTRY (readTimerOff, alongside T0) and gate the
  return-site write on the latch instead of re-reading ctrl.flags at return.
  Now flipping the flag mid-flight cannot suppress the in-flight probe's own
  write (it latched ON at entry), and the host pushes flags-OFF in on_batch_end
  right after the probe -> every off-cadence launch latches OFF -> a probe
  samples EXACTLY one step. Verified: IR gate updated + green (flag read now at
  entry, before remap), test_timer_gate/indirect PASS, dynamic-loop all-levers
  bit-exact. So all 6 review findings are now FIXED.
Verified: all C++ fixtures bit-exact after rebuild (paged, CLC), IR gate green,
Python controller PASS, all 3 Python modules parse.

ISOLATED TIMER OVERHEAD (2026-07-09, test/py/eval_timer_overhead.py) -- the
CLEAN number, and it CORRECTS an over-attribution. Real GQA decode (nqo16/nkv2,
2048 tiles, device-L2 channel):
| timer | overhead |
|---|---|
| OFF (766 us/step baseline) | -- |
| ON every step | +4.7% |
| 1-in-8 (current) | +0.59% |
| 1-in-16 / 32 / 64 | +0.29% / +0.15% / +0.07% |
=> the device-L2 timer is CHEAP; at 1-in-8 it is +0.59%, at 1-in-64 +0.07%
(eKV-free). The sampling is TEMPORAL (which steps); WITHIN a probe it is
per-CTA complete. Safe to coarsen the cadence (decode cost is step-stable).

CORRECTION to the previous entry's claim ("the #1 timer bug was the E2E -66%
blocker"): REFUTED by this bench. The timer even stuck-ON every step costs only
+4.7% on the kernel, not 64%. So the E2E enforce "22.81 -> 66.22 (parity)"
swing was MEASUREMENT NOISE (the documented 31-66 req/s variance), NOT the #1
fix. This is the SECOND E2E causal claim a clean microbench has overturned (the
first: ncu vs the pi-ordering "cost"). META-LESSON now explicit: the E2E
serving numbers are ~80% noise (radix-cache prefix reuse + overlap-scheduler
timing + preemption); single-run causal attributions from them are unreliable.
TRUSTWORTHY facts are ISOLATED: kernel pi -35% (bit-exact), timer +0.59%@1-in-8.
The #1 fix stands as a CORRECTNESS fix (ranking distortion under churn), NOT a
measured perf win. Cadence sweep (plan_every 8/16/32) was likewise
noise-dominated (8->66, 16->53, 32->{31.76,57.72} back-to-back same server) --
uninterpretable. A clean E2E win/loss REQUIRES: --disable-radix-cache + warmup
+ N repeats + CI. Until then, NO E2E gain OR regression is claimable.

PREFILL UNBLOCKED (2026-07-09) -- the FlashInfer-prefill-under-clang compile
gap is CLOSED, and the woven prefill kernel COMPILES + WEAVES. The gap was
tiny: FlashInfer already `.template`s most dependent member-template calls (22
sites) but MISSED 9 (3 load_128b_async, 4 vec_cast::cast, 2 get_permuted_offset)
+ 1 `__ldca` (an L1-cache HINT, dropped to a plain load -> bit-exact). Formalized
in patch_flashinfer.py (4 new entries; the check now tests `old` BEFORE `new`
so mixed patched/unpatched files patch correctly). Result:
- `batch_prefill_paged_kernel_mask_2.cu` compiles under clang: 0 errors.
- WITH -fpass-plugin: `[sched] weaving BatchPrefillWithPagedKVCacheKernel ...
  remapped 1 ctaid use through task_order` -- pi applies to prefill; the
  split-kv `PersistentVariableLengthMergeStatesKernel` also woven. 836 KB obj.
- Reproducible: restore fresh prefill.cuh -> patch_flashinfer.py -> compiles;
  --check idempotent (exit 0).
RESOLVED -- THE CLANG "WALL" WAS A ONE-MACRO GUARD (2026-07-09). CORRECTION:
the "fundamental clang codegen gap" below was WRONG -- I over-read the census
(clang emits 0 cp.async/ldmatrix) as "clang can't", when the cause was a MACRO
GUARD. FlashInfer + CUTLASS gate their inline-PTX fast path (cp.async in
cp_async.cuh:37, ldmatrix/stmatrix/mma in mma.cuh:30,36) behind
`#if __CUDACC_VER_MAJOR__ >= 11` -- an nvcc-ONLY macro clang does not define
(clang defines __CUDACC__ but not __CUDACC_VER_*). So clang silently compiled
the SCALAR fallback (mismatched barriers -> race -> crash). The fast path is
compiler-agnostic INLINE ASM, so defining the macro makes clang emit it:
  clang cp.async 0->909, ldmatrix 0->704 (== nvcc 704), stmatrix 0->112 (==nvcc)
FIX (nvcc_clang_shim.py): add `-D__CUDACC_VER_MAJOR__=<maj> -D__CUDACC_VER_MINOR__
=<min>` (derived from CUDA_PATH). RESULT -- test/py/test_flashinfer_prefill.py:
  [PASS] woven prefill produces finite output (NO crash)
  [PASS] reversed task_order -> BIT-EXACT (pi is E1 on prefill)
  [PASS] woven timer on all 366 tiles
  + decode still bit-exact (no regression).
=> OPTION 1 (LLVM-for-EVERYTHING) IS VIABLE. Prefill weaves via the LLVM pass,
bit-exact, no PTX weaver needed. The whole clang toolchain is unblocked by one
-D. (Gemini's push -- "LLVM CAN lower these" -- was right; the issue was never
clang's backend, it was FlashInfer's nvcc-version guard excluding clang.)
REMAINING for prefill: exact per-request (binding-table design), wire into
serving (SCHED_WEAVE_ONLY=batch_decode,batch_prefill), re-test use_device_timer
x prefill (earlier crash was on the BROKEN scalar kernel; may work now).
The PTX-on-nvcc analysis below is SUPERSEDED (kept for the record).

BISECTED THE CLANG WALL (2026-07-09) -- DECISIVE, and it forks the design.
clang-vs-nvcc PTX instruction census on the prefill kernel:
  cp.async   0 vs 979   (async global->shared copy pipeline)
  ldmatrix   0 vs 704   (tensor-core matrix load)
  stmatrix   0 vs 112   (tensor-core matrix store)
  red.global 0 vs 819   (cooperative reduction)
  st.shared  2775 vs 95 (clang's SCALAR fallback)
  bar.sync   28 vs 82   (fewer barriers -> broken sync)
clang's NVPTX backend does NOT generate FlashInfer prefill's tensor-core +
async-pipeline fast path; it falls back to scalar copies with mismatched
synchronization -> race -> illegal access -> crash. This is a FUNDAMENTAL
clang-CUDA gap (a whole class of intrinsics: cp.async / ldmatrix / stmatrix),
NOT a targeted-patchable bug like the 9 .template sites. Fixing it = upstream
clang NVPTX work (huge) or rewriting FlashInfer's fast path (defeats it).
=> Option 1 (fix clang, keep LLVM-for-all) is NOT viable for prefill.
=> Option 2 (PTX-on-nvcc weaver) is the path -- and this VALIDATES it: nvcc
   emits the correct tensor-core kernel, we weave its PTX (which HAS the
   cp.async/ldmatrix; our pi/timer edits never touch them). Decode works under
   clang because it is memory-bound streaming (no tensor-core path); prefill is
   compute-bound matmul (needs it). Design decision: build the PTX-on-nvcc
   weaver for prefill (pi/timer/per-request first); keep the LLVM pass for
   decode OR eventually unify everything on PTX-on-nvcc.

RUNTIME WALL (2026-07-09, deeper than the compile gap): the clang-compiled
prefill kernel COMPILES but CRASHES at runtime ("unspecified launch failure" =
illegal access), while the IDENTICAL patched source compiled by REAL NVCC RUNS
CORRECTLY (test/py/test_flashinfer_prefill.py + isolation). Clean isolation:
  * real nvcc, unwoven, +plane   -> OK (sum finite)
  * clang, SITE_OFF (no weave), no device timer -> CRASH
  * clang + weave -> CRASH
So it is CLANG CUDA CODEGEN mishandling the ADVANCED prefill kernel (cp.async /
mbarrier / cutlass MMA / sm_90+ async-pipeline features) -- NOT our weave, NOT
the patches (nvcc compiles the same patched source fine), NOT the plane. This
is a clang-CUDA-maturity wall: the DECODE kernel is simpler and clang codegens
it correctly; the PREFILL kernel uses the cutting-edge path clang miscompiles.
CONSEQUENCE: prefill weaving via clang is BLOCKED at the codegen level -- a much
harder problem than the compile gap (a clang bug / unsupported feature, not a
patch). Options: (a) bisect which prefill.cuh feature clang miscompiles + work
around / report upstream (deep, uncertain); (b) use a SIMPLER FlashInfer prefill
backend clang handles (if one exists); (c) accept decode-only weaving (nvcc
compiles prefill for correctness, unwoven -- the current serving default). Also
found: use_device_timer() faults the prefill run even under real nvcc (separate
device-timer x prefill bug, moot while prefill can't run woven).
The compile-gap work STANDS (patch_flashinfer entries valid + reproducible) and
would be needed the moment clang codegen is fixed; it just is not sufficient.

So prefill weaving is UNBLOCKED at the compile/weave level (matches eKV's
prefill coverage, on the clang/LLVM path, no Triton). REMAINING for full
prefill: (1) EXACT per-request attribution -- prefill tiles are (query-block x
request); a long prompt spans many tiles, so per-request = sum over the
request's tiles. Two routes: the prefill plan's request_indices binding (Python,
like decode's _tile_binding) OR the clean on-GPU form -- key the timer by the
loaded request-index mu (eKV's mu-based attribution, exact + split-safe + no
private-field read). (2) bit-exact RUNTIME verify of woven prefill (pi is E1 ->
must be bit-identical). (3) wire prefill into serving (SCHED_WEAVE_ONLY +=
batch_prefill; the plugin's prefill-phase hook). (4) prefill gains are TTFT /
goodput (not TPOT), and smaller/less-novel than decode (prefill cost is known a
priori) -- broadens the framework, decode stays the headline.

Harness note: backgrounded `bench_ab` launches failed silently this session
(detach-context fragility + I repeatedly `pkill`ed my own job in status
checks); the reliable path was DECOMPOSED synchronous calls (boot / poll /
bench / kill as separate Bash invocations).

**B3 remaining:** the attention-dominant HEADLINE config -- a 1-3B model
(Qwen2.5-3B is cached locally; NOTE group size 8 => SGLang may pick the
tensor-core decode path, where tile binding falls back 1-tile/req by design
and pi/timer weave generically but are unvalidated -- validate first) with
2-8k KV; plus TPOT p99 with repeats (>=5 runs/config for CIs), cadence
sweep, CLC auto-arm row, and a cuda-graph-enabled round (the init-time
plane hook, added 2026-07-07, unblocks capture).
**B4. Contention scenario.** Co-running job; validate the γ-inversion finding
live and that hysteresis prevents policy oscillation.
*Acceptance:* p99 TPOT improvement ≥ the kernel-level −12..−25% discounted by
attention's share of step time, at ≤1% overhead in observe-only mode; no
correctness drift over ≥1M requests (bit-exactness spot-checks + finite
outputs).

## Workstream C — the formal spine (math; parallel anytime)

**C1. LPT-robustness theorem.** List scheduling with ranking noise: bound
makespan(π̂) ≤ (4/3 + f(inversion mass))·OPT; tie f to the measured
recall→penalty curve (clc_noise_probe). Corollary: the 0.75 arming threshold
becomes derived, not tuned.
**C2. Effect-type algebra.** Composition laws (E0 commutes; E1∘E1=E1; O =
commutative monoid ⇒ replay/reorder-safe; E2(ε1)∘E2(ε2) ⊑ E2(ε1+ε2)) + a
soundness statement: any composition of woven capabilities with τ=0 is
extensionally identity. Paper appendix rigor; mechanization optional.
**C3. Sampled-observation stability.** EWMA + deadband under nonstationary
costs: bound estimator staleness vs SCHED_TIMER_EVERY; give the rule for
choosing N (probe rate ≥ workload mixing rate).

## Workstream D — architecture redesign (the capability manifest)

**D1. Capability manifest.** One declarative table per instrument —
{name, effect type, arch gate, detection predicate, arming slot, decline
conditions, cost model, composition/order constraints} — as the single source
of truth driving pass registration/order, runtime-slot resolution, decline
reporting, and generated docs. Kills the comment-encoded invariants
(shed-before-prefetch, WQ-before-weave, claim-after-body). This converts "a
bag of instruments" (π, prefetch, discard, shed, timer, ticket, CLC, PDL,
future TMA/cluster) into a typed, extensible algebra — the abstraction OSDI
remembers.
**D2. Weavability census.** Run the passes over the FlashInfer (+vLLM) kernel
corpus; report per capability: woven / declined(reason). Doubles as the
honest scope statement and a CI regression artifact.
**D3. FileCheck IR unit tests.** Golden .ll fixtures per weave (remap PHI
shape, shed select+dominance, timer gate CFG, WQ driver skeleton, grid
guard). CPU-only CI.
**D4. Multi-axis claim.** Today the dynamic layer requires 1D grids (guard).
num_kv_heads>1 is the COMMON case, so if serving CLC matters at all this is
required: linearize raw = x + y·gx + z·gx·gy, decode the full v4 tuple,
remap through π over linear ids. Decide AFTER A/B: if CLC never arms in
serving (R analysis suggests it rarely will), keep the guard + document.
**D5. TMA-era detection or scope statement.** cp.async.bulk.tensor descriptor
sites are invisible as loads; either detect at descriptor level or state the
boundary and route those kernels to launch-arg/trace-level observation.

## Workstream E — observation channel redesign (GPU arch)

**E1. Timer channel ablation. — DONE (2026-07-04,
`test/py/eval_timer_channel.py`).** paged softmax, 8192 tiles, long-tail mix:

```text
timer OFF (flags gate)        568.1 us/step     --
host-mapped PCIe atomics      718.3 us/step   +26.5%   (the current design)
device-buffer atomics         565.2 us/step    -0.5%   (FREE)
device readback+clear (D2H)   127   us/probe
LPT rankings across channels: 100% agreement
```

VERDICT: collection should be device-side (free); only READOUT needs
sampling (127 us D2H per probe) -- strictly better than today, where the
flags gate must suppress the expensive collection itself.

**IMPLEMENTED (2026-07-04): SCHED_TIMER_INDIRECT** -- the baked timer slot
(fixed VA, cache-stable, tag `-ti`) holds a retargetable POINTER word: 0 =
off (fail-safe), else the row table (device buffer via
`SchedPlane.use_device_timer()`, or host-mapped for zero-touch observers).
One extra cached load per kernel exit; the loaded address is an opaque
register (immune to the ptxas immediate hazard). Gated by
`test_timer_indirect.py` in run_all: word-0 fail-safe, device rows, flags
composition, bit-exactness, cost attribution -- all green. The SGLang plugin
uses the device channel BY DEFAULT (`SCHED_TIMER_DEVICE=0` reverts).
Remaining in E: E2 (globaltimer+smid vs clock64 estimator quality).

**Also implemented (2026-07-04), the serving-loop production fixes:**
- OVERLAP-SAFE OBSERVATION: on_batch_end records a CUDA event (no
  synchronize -- a sync would serialize SGLang's overlapped event loop);
  the read happens at a later on_batch_begin once the event has fired, and
  a new probe never starts while one is unconsumed.
- AUTO-R: the plugin resolves R from the cached woven decode .so
  (driver-API occupancy) lazily after the first JIT; SCHED_CLC_R overrides.
- OPS VISIBILITY: one stats line per SCHED_STATS_EVERY steps (tiles, split,
  R, probes folded, uncertainty, mode).
**E2. clock64 vs %globaltimer+%smid.** Residency conflates queue/execute/
contention and mixes DVFS clock domains across SMs; measure regression fit
quality (R² of t̂ = α·kv_len+β) under both sources.
**E3. σ vs host-side equivalents.** Woven discard/evict_last vs
cudaAccessPolicyWindow/L2 persistence/carveout: where does per-REQUEST
granularity (which host APIs cannot express) actually win? This ablation
justifies weaving per se against the obvious reviewer question.

## Workstream F — generality & portability

**F1. sm_100 (GB200/B200) pass.** R recomputes by construction; re-measure
the tie/win boundaries + noise crossing once.
**F2. Second μ instance.** One more kernel family live (vLLM decode, or
SGLang triton prefill via the eKV lineage) to defend generality.
**F3. sm_86 ticket regime regression** stays in CI (the workers≪tasks win).

## Workstream P — production hardening (code)

**P1. Failure injection tests:** unprogrammed plane (→stock, covered), VA
miss (→recompile-once, covered by design; add test), plan-layout mismatch
(→1-tile fallback; add test), batch > max_tasks (→loud error, covered),
SGLang attr drift (add smoke).
**P2. CI split:** GPU suite (run_all.sh) on the Blackwell runner; CPU-only
FileCheck + python unit tests everywhere.
**P3. Config surface:** one generated table of every env knob (compile-time
vs runtime, default, safety class).
**P4. Plugin observability:** export step time, arming decisions,
uncertainty, probe cadence as metrics (ops need to see the controller think).
**P5. Version pins:** FlashInfer plan-layout guard (exists) + explicit
version check with loud downgrade to request-level binding.

---

## Ordering & dependencies

```
now ──► A1 (dispersion) ──► A2 (vs split_kv) ──► paper framing decision
   ├──► B1..B2 (server+load) ─────► B3/B4 (metric table)  [needs A1's mixes]
   ├──► C1..C3 (parallel, no GPU)
   ├──► D2 (census), D3 (FileCheck), P1 (failure tests)   [cheap, start now]
   └──► E1/E2 (timer ablation) ──► feeds B3 cadence choice
A/B results ──► D1 (manifest refactor: keep what matters), D4 (only if CLC
arms in serving), D5, E3 ──► F1/F2 ──► camera-ready evaluation
```

Pacing to OSDI '27 (≈5 months): A+B+E1 within 4–6 weeks (they are
measurements, the infra exists); C in parallel; D1 refactor mid-window (after
the census says which capabilities carry weight); F + full ablation grid in
the final 6 weeks. The paper stands on A2 + B3 + C1/C2 + the CLC
characterization; everything else is defense in depth.
