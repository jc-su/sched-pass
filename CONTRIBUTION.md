# Contribution — the request↔tile binding bridge

## The core (the moat), one sentence

**A woven, bidirectional, late-bound, bit-exact bridge that carries the control
plane's *global per-request information* into an *unmodified* fused GPU kernel and
carries per-request *observation* back out — turning a per-request-*blind* fused
kernel into a per-request-*aware* one, live, every step.**

Before: the control plane (SGLang) knew requests but saw the kernel as a black
box; the kernel had the static plan but no per-request cost/priority/SLA and no
way to report back. The bridge *associates* them. Nobody else has this — eKV
forks the kernel, H2O is a static policy, and the hardware has no per-request
semantics.

## The mechanism — six wired pieces (all real code, all bit-exact)

| # | Piece | What it does | Code |
|---|-------|--------------|------|
| 1 | **Baked fixed-VA tables** | control-plane tables (order/timer/budget/ctrl) live at a canonical VA (`MAP_FIXED 0x5C00…`); the address is compiled into the kernel (`SCHED_BAKE_*`) — no pointer plumbing | `python/sched_rt.py` `compute_bake_env` |
| 2 | **Rosetta stone (generalizes)** | the plan's index that maps a unit of kernel work → the request/route it serves. **Attention:** `request_indices` (`tile→slot`; `DecodePlanInfo[3]`, `PrefillPlanInfo[4]`). **MoE:** `topk_output.topk_ids` (`token→expert` = `m_indptr`) — same idea, different kernel | `_plan_request_indices` (attn); `sched_moe_hook.py` (MoE) |
| 3 | **Info IN** | `table[tile] = f(per_request_info[request_indices[tile]])`, installed device→device | plugin cost/`install_order`; `sched_rt.py` `use_device_order` |
| 4 | **Kernel acts** | `task = order[ctaid]`; the woven kernel reads its request's row and reorders/sheds/hints. E1 = bit-exact permutation | `lib/SchedWeave.cpp` `emitRemap` |
| 5 | **Obs OUT** | kernel writes `timer[tile]`; control folds `cost[rid] = Σ_{request_indices[tile]==slot} timer[tile]` (exact, split-safe) | plugin `_consume_probe` (`scatter_add_`) |
| 6 | **Late-bound** | `request_indices` re-read *every step* — the rid↔tile association is live, not frozen (batch churns) | plugin `_tile_binding` / `_prefill_tile_binding` |

The **closed loop** is the bridge in both directions: kernel observes cost (5) →
control relearns → kernel reads the new order (3,4) → acts. That round trip is
the moat working end to end.

## Evidence — what a per-request-aware kernel then *does* (measured)

The bridge is the contribution; the levers are what the newly-informed kernel
chooses to do. Verdicts are **per regime** (measured, not assumed):

| Regime | Bottleneck | Actions that pay | Number |
|--------|-----------|------------------|--------|
| **Decode** | DRAM, *predictable* | ORDER; OBSERVE→reorder; SHED straggler | −12%; **95% misprediction recovery**; straggler −49…−96% |
| **Prefill / chunked** | L2/compute | ORDER; **SPLIT-SKEW** straggler | −15%; **−40%** (over-split the straggler, bit-exact) |
| **MoE** | compute, *mispredicted* | OBSERVE routing counts; cap hot expert (E2) | **live on Qwen3-30B-A3B**: cap armed in `FusedMoE.forward`, fires across all 48 layers, fail-soft, regime-guarded, **near-transparent on real text** (E2→≈identity when balanced). Capture-safe; 13/13 real-type + live-GPU-forward tests |
| **Continuous batch** | dynamic | the moat's core (per-request, per-step) | — |

Flagship figures: the **95% misprediction recovery** (`eval_closed_loop.py`) and
the **straggler shed** (`eval_straggler_shed.py`, −49% at 50% straggler accuracy,
light requests bit-exact).

## The boundary (why the taxonomy is what it is)

The moat is an **information** bridge: it wins on actions that only need to know
*which request* (ORDER/OBSERVE/SHED/split-skew), and it does **not** win where the
wall is a physical resource the hardware already saturates:
- **CLC** — HW block scheduler already balances (narrow BS≈wave sweet spot only).
- **SHAPE** — subsumed by FlashInfer's shared-memory blocking (1% L2 reuse).
- **PDL** — mooted by our own device-resident tables (nothing left to hide).
- **split-skew on decode** — DRAM-walled; FlashInfer already 274-way splits.

These negatives are not failures — each has a measured reason, and together they
*define* where the moat pays: information actions, not resource actions.

## One-line positioning

*Per-request scheduling woven inside unmodified fused attention kernels, driven by
a global control plane through a bidirectional live binding, bit-exact by
effect-typing — the thing that makes a fused kernel per-request-aware.*
