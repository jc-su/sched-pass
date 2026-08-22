# What the Prototype Pass Already Proved About Request Identity

Recorded 2026-08-22 after re-reading the original `main`-branch prototype
(worktree at `~/Dev/sched-pass-main`, initial commit 4789f16; DESIGN.md,
`lib/SchedWeave.cpp`, GPU-validated on sm_86 and Blackwell per its README).
This corrects an overstatement in the current branch's framing and pins
what is portable.

## The correction

The current branch's docs say the pass "does not infer request identity
from arbitrary pointer arithmetic" and treats explicit markers as the
only recognition path. The first half is right; the implied
impossibility is not. The prototype demonstrated, in LLVM IR on real
FlashAttention-class CUDA kernels, that **the paged-KV family carries
request identity structurally**, in three mutually exclusive forms:

- **Index-based mu (paged/CSR):** `hasLoadedIndex` — a bounded backward
  walk (GEP/phi/binop/cast/select, 256-step cap) finding an address
  data-dependent on ANOTHER load. That loaded value IS the page-table /
  block-table row: the identity-carrying access, found with no names
  and no annotations (CapKV's `capkv.auto` signature, ported to scalar
  IR).
- **Arithmetic mu (contiguous):** `reachesSlotCtaid` — the address
  derives from ctaid(slotAxis) WITHOUT crossing a load (crossing one
  would make it the other form). A positive classification, not a
  fallback.
- **cp.async streams:** FlashInfer's actual KV path recognized at the
  inline-asm level, extracting the global source operand from the
  "l"-constrained argument.

Selection is disciplined by SCEV: only constant-stride AddRec addresses
in innermost loops, never address-space-3, K and V streams both
captured. And the load-bearing rule, stated in eKV's terms: a kernel
with index-based gathers where no site matches is **skipped loudly,
never silently re-bound** — recognition is fail-closed, exactly like
the current verifier.

Request identity operationally meant: `task = task_order[ctaid]` with
null-table/OOB defaulting to identity (fail-safe), per-task policy
reads, per-task timers — and eKV's row-form recovery (seq and stride
from the SAME multiply; CSR `kv_indptr[seq]`) recovers the request
coordinate from the kernel's own index math.

The honest limits the prototype itself recorded: TMA-descriptor streams
are invisible as plain IR loads (covered there by launch-arg/runtime
observation levels, not the pass), and nothing here infers identity for
kernels OUTSIDE these structural families. "Arbitrary production
kernels" stays open; the paged family does not.

## What the current branch should adopt

1. **Structural candidate discovery feeding the existing verifier.**
   Port `hasLoadedIndex` + the SCEV stride filter + cp.async source
   extraction as a discovery phase that proposes candidate acquisition
   sites in unmarked kernels. Candidates never self-authorize: each must
   pass the same legality gates as marked sites (dominating binding —
   now derivable for the paged family via row-form recovery —
   uniformity, escape closure, defer discipline) or the kernel is
   skipped loudly. This turns "markers only" into "typed frontend OR
   structural recognition, both verified," and directly answers the
   ARCHITECTURE 7.2 open item for load/cp.async cones. Validation on
   this branch: accept fixtures cut from real FlashInfer IR plus a
   mutation-harness extension (the discovery must never fire on a
   taint-mutated cone).
2. **Row-granular refusal by address-cone redirection.** CapKV's
   `select(ok, page, sentinel)` woven INTO the address cone cost ~0%
   (vs +33% for mask-and-branch) and stayed graph-replay-safe. The
   claim-consumer contract currently refuses at launch granularity; a
   sentinel-redirect per out-of-lease row is the finer fail-closed
   action and composes with the existing check.
3. **Graph-safety as a named invariant class.** The prototype states
   what this branch learned empirically: every woven effect must be a
   commutative-monoid write, an idempotent write, or a pure read —
   that class is CUDA-graph-replay-safe by construction. Adopt as a
   stated contract for all instrumentation (the forward profiler and
   bindings fills already conform).
4. **The two admission stances as vocabulary.** eKV = fail-open
   observability (defaults are stock; results bit-exact); CapKV =
   fail-closed protection. This branch's counters/profilers are the
   first stance; the claim contract is the second. The paper should
   name both.
5. **Fail-safe identity defaults.** `task_order == null -> identity`
   is the same design as the bindings tensor's slot = -1 row; keep the
   continuity explicit.

## Evidence hygiene

ARCHITECTURE 7.2's sentence "the previous branch's recognition
experiment is not evidence for this branch" stands: adoption means
porting the detection INTO this branch's pass behind its verifier and
re-validating with this branch's fixtures — not citing the prototype's
results. The prototype is the existence proof and the design source;
the ledger item is the port.

## Integration lineage (code-verified 2026-08-22, prototype python/ vs ours)

Read from code, not docs: `nvcc_clang_shim.py`, `sitecustomize.py`,
`patch_flashinfer.py`, `sched_sglang_plugin.py` against our
`tools/jit/nvcc_clang.py`, `tools/jit/activate.py`,
`tools/flashinfer/prepare_overlay.py`.

**Already inherited (and in one place improved).** Our shim carries the
prototype's full translation discipline: the bisected
`__CUDACC_VER_MAJOR__` define (without it clang silently compiles
FlashInfer/CUTLASS's SCALAR fallback instead of the cp.async/ldmatrix
inline-asm fast path — races, and a silent both-arms performance lie),
toolkit `-isystem` ordering, the dialect prelude, arch mapping,
`-Xptxas`/deps/`-ccbin`/fatbin translation, selective instrumentation
with `NTA_REAL_NVCC` routing, and shim logging. Header patching lives in
`prepare_overlay.py` (`patch_header`/`patch_prefill_header`) with
source-hash checks. One improvement over the prototype: the shim REFUSES
an instrumented compile whose workspace lacks the NTA cache tag —
fail-closed where the prototype could only pre-arm the env at import and
verify later. The prototype's hazard note stands as the reason the guard
exists: FlashInfer freezes its workspace path at module import, so a
late-armed process silently reuses stock-cached kernels.

**Landmine worth keeping visible (root-caused in the prototype,
2026-07-07):** importing torch at interpreter start — even doing nothing
else — corrupted Qwen generation content under SGLang; the fix was a
lazy meta-path finder that registers only after
`sglang.srt.managers.scheduler` finishes importing (torch initialized in
sglang's own order) yet before FlashInfer import. Our `activate.py`
avoids the trap by exec-ing the serving process with env pre-set and
importing nothing early; any future bootstrap change (a sitecustomize, a
pre-import hook) must preserve that ordering.

**Gap to adopt — observability hooks must not crash serving.** The
prototype's `_hook_error` discipline: an observation hook never takes
down the serving process; the first error per site prints loudly
("control plane degraded; serving continues"), later ones count, and the
count rides the periodic stats line. Our mechanism guards rightly raise
(CapKV stance), but our observability class — the per-forward profiler,
co-tenant sampler, composition counters — currently raises through the
serving path (eKV-stance code with CapKV-stance failure behavior). The
profiler's event synchronize is also illegal inside any future
graph-captured path. QUEUED CHANGE (branch `forward-profile`, after the
running campaign releases that worktree): wrap the observability class
fail-open with first-error-loud counters exported through stats, keyed
by the stance taxonomy above.

**Performance candidates for the resident tail (from the prototype's
policy tier, aimed at the current 1.095-vs-1.05 bar).**
1. `prefetch.L2::evict_first` on claim-staging streams — the prototype's
   "polite" tier: streamed lines die young and stop polluting L2 for
   co-residents. Our staging_mixed forwards sit exactly at the resident
   p99 boundary; cache-polite staging is a cheap, graph-safe (pure-hint)
   candidate for part of the ~5 ms delta, mechanism-owned rather than
   config.
2. The branchless select-address prefetch trick (disabled -> prefetch of
   the current line, an L2 hit, ~free) as the wiring pattern for any
   woven hint: no divergence, no extra branches, replay-safe.
3. cp.async source warming (the prototype's AsyncSite): warm the global
   source of the next staging wave during compute — relevant to the
   staging wavefront if transfer, not launch, ever dominates a shape.


## Census on production IR (2026-08-22, code-run, not claimed from docs)

The stock (non-NTA) FlashInfer 0.6.12 paged-decode kernel from a banked
JIT cache, compiled to device IR through the same clang path serving
uses, then run under `NTA_DISCOVERY_NOTES=1`:

- Plain-load pass alone: every `BatchDecodeWithPagedKVCacheDevice`
  instantiation reports "index-based gather present but no strided site
  matched — skipped loudly." The identity signature fires on all of
  them; zero sites qualify, because **FlashInfer's real KV stream is
  cp.async**, invisible to load-stride analysis — exactly the gap the
  prototype's AsyncSite existed for.
- With the AsyncSite port (first "l"-constrained inline-asm operand,
  source cone classified by the same loaded-index walk): **79 cp.async
  candidates, every one with a loaded-index source cone**, and no
  unmatched-gather skips remain. The structural signature recovers the
  identity-carrying access for the actual production KV path.

What this buys: measured coverage evidence for the recognition claim
(the paged family's identity is structurally recoverable in production
kernels, not just fixtures); the concrete path to deriving bind
operands for that family; and a demonstration that the loud-skip
discipline surfaces its own blind spots — the census itself found the
cp.async gap within the hour.
