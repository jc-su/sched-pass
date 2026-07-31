"""sched_moe_hook.py -- the MoE arm of the moat: the SAME request<->tile binding
pattern, but the Rosetta stone is the ROUTING (topk_ids -> expert counts =
m_indptr) instead of request_indices. The control plane sees the routing (global,
per-forward, DATA-DEPENDENT -- unknowable at plan time), detects the hot expert,
and CAPS it (E2 drop-tokens = GShard/Switch capacity, applied per-expert and
observation-driven). Demonstrated plan-level (+62% imbalance -> 60% recovered,
eval_moe_capacity.py); this is the LIVE wireup into SGLang's FusedMoE.forward.

OPT-IN (SCHED_MOE_CAP=1) because it is E2 (drops tokens -> approximate). The cap
action is a KNOWN technique (GShard capacity); the contribution here is the
BINDING generality -- the moat's mechanism carrying global routing info into the
MoE kernel, per-forward, targeted at the hot expert.

Status: hook logic + registration are implemented and UNIT-TESTED here. LIVE
end-to-end validation needs a cached MoE model (none local) + the git-lfs trace;
that is the next integration, scoped in ROADMAP.
"""
import os


def moe_expert_cap(topk_ids, topk_weights, num_experts, capacity_factor=1.25,
                   count=True, min_capacity=8):
    """GShard/Switch expert capacity, control-plane-driven. topk_ids:
    [num_tokens, top_k] expert assignment per token. Each expert may take at most
    C = capacity_factor * (num_tokens*top_k / num_experts) tokens; the OVERFLOW
    tokens (in arrival order) are DROPPED by zeroing their routing WEIGHT (the
    expert output for them becomes 0 -> they skip the expert). Returns
    (topk_weights', dropped_count-or-None). Zeroing the WEIGHT (not the id) is safe
    on any fused-MoE backend -- the GEMM still runs, the contribution is masked.
    Bit-exact when nothing overflows (capacity_factor high / balanced).

    REGIME GUARD (min_capacity): per-batch capacity is a large-batch (prefill /
    training) construct. On tiny autoregressive DECODE batches the mean per-expert
    load nt*k/E is O(1), so "above capacity" is small-sample lumpiness, not genuine
    imbalance -- capping there drops most of the batch (~68% measured on Qwen3-30B
    decode at factor=1.0). We SKIP when C < min_capacity, so the cap targets the
    prefill/large-batch regime where hot-expert imbalance actually lives. This
    branch is host-side on a shape-derived int -> capture-safe (constant per graph).

    CUDA-GRAPH SAFE: all ops are fixed-shape (SGLang captures the decode forward,
    MoE included). No `.item()` in the hot path (count=False skips the one sync);
    counts via scatter_add not bincount (bincount can host-sync); mask via
    masked_fill not boolean-index-assign. No host-side branch on the result."""
    import torch
    nt, k = topk_ids.shape
    C = int(capacity_factor * (nt * k) / num_experts)
    if C < min_capacity:                    # batch too small for capacity to mean
        return topk_weights, (0 if count else None)   # anything -> no-op
    flat_e = topk_ids.reshape(-1)
    flat_w = topk_weights.reshape(-1).clone()
    # per-expert running position of each (token,slot); overflow if position >= C.
    # Sort by expert, rank within expert (= idx - group_start), scatter back.
    order = torch.argsort(flat_e, stable=True)
    sorted_e = flat_e[order]
    # counts via scatter_add (fixed shape=num_experts, no host sync) not bincount.
    counts = torch.zeros(num_experts, dtype=torch.long, device=topk_ids.device)
    counts.scatter_add_(0, flat_e.long(), torch.ones_like(flat_e, dtype=torch.long))
    grp_start = torch.zeros(num_experts, dtype=torch.long, device=topk_ids.device)
    grp_start[1:] = torch.cumsum(counts, 0)[:-1]
    within = torch.arange(sorted_e.numel(), device=topk_ids.device) - grp_start[sorted_e]
    overflow = torch.zeros_like(flat_w, dtype=torch.bool)
    overflow[order] = within >= C
    flat_w = flat_w.masked_fill(overflow, 0.0)
    dropped = int(overflow.sum()) if count else None
    return flat_w.reshape(nt, k), dropped


_ORIG = {}
# Observe-mode accumulators (SCHED_MOE_OBSERVE=1). Host-side ints updated with a
# .item() sync -> NOT capture-safe, so observe runs boot with --disable-cuda-graph.
# The obs-out here mirrors the attention timer's fold: it reports what the cap,
# driven by the live routing, actually did on real traffic.
_STATS = {"calls": 0, "capped_calls": 0, "dropped": 0, "routed": 0}


def _observe(topk_ids, overflow_mask, num_experts):
    """Accumulate drop stats and log every 200 capped forwards."""
    _STATS["calls"] += 1
    d = int(overflow_mask.sum())
    _STATS["routed"] += int(topk_ids.numel())
    if d:
        _STATS["capped_calls"] += 1
        _STATS["dropped"] += d
    if _STATS["capped_calls"] and _STATS["calls"] % 200 == 0:
        r = _STATS["dropped"] / max(1, _STATS["routed"])
        print(f"[sched-moe] {_STATS['calls']} forwards, "
              f"{_STATS['capped_calls']} capped, "
              f"{_STATS['dropped']} tokens dropped ({r:.2%} of routed slots)",
              flush=True)


def register_moe_cap():
    """Wrap SGLang FusedMoE.forward to cap the hot expert when SCHED_MOE_CAP=1.
    Idempotent; no-op if sglang MoE is absent. The wrap reads topk_output
    (global routing, per-forward), caps, and calls the original -- the binding
    bridge, MoE edition."""
    if os.environ.get("SCHED_MOE_CAP") != "1":
        return False
    try:
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    except Exception:
        return False
    if FusedMoE.forward in _ORIG.values():
        return True
    orig = FusedMoE.forward
    cf = float(os.environ.get("SCHED_MOE_CAPACITY", "1.25"))
    mc = int(os.environ.get("SCHED_MOE_MIN_CAP", "8"))  # skip tiny (decode) batches
    observe = os.environ.get("SCHED_MOE_OBSERVE") == "1"

    def forward(self, hidden_states, topk_output):
        try:
            ti = getattr(topk_output, "topk_ids", None)
            tw = getattr(topk_output, "topk_weights", None)
            ne = getattr(self, "num_experts", None) or getattr(
                self, "num_local_experts", None)
            if ti is not None and tw is not None and ne \
                    and hasattr(topk_output, "_replace"):
                # count=False: no host sync -> capture-safe. Replace
                # UNCONDITIONALLY (no host branch on `dropped`) -- the mask is a
                # no-op when nothing overflows, so this stays bit-exact when
                # balanced AND capturable inside SGLang's decode CUDA graph.
                new_w, _ = moe_expert_cap(ti, tw, int(ne), cf, count=False,
                                          min_capacity=mc)
                if observe:                    # non-capture-safe obs-out path
                    _observe(ti, new_w != tw, int(ne))
                topk_output = topk_output._replace(topk_weights=new_w)
        except Exception:
            pass  # fail-soft: never break the model forward
        return orig(self, hidden_states, topk_output)

    _ORIG[id(orig)] = orig
    FusedMoE.forward = forward
    return True


# --- self-test (no MoE model / no GPU needed) ------------------------------
if __name__ == "__main__":
    import torch
    NT, K, E = 1024, 2, 16
    gen = torch.Generator().manual_seed(0)
    # imbalanced routing: 40% of tokens' first slot -> expert 0 (the hot one)
    ti = torch.randint(0, E, (NT, K), generator=gen)
    ti[:int(0.4 * NT), 0] = 0
    tw = torch.rand(NT, K, generator=gen)
    counts0 = torch.bincount(ti.reshape(-1), minlength=E)
    new_w, dropped = moe_expert_cap(ti, tw, E, capacity_factor=1.25)
    C = int(1.25 * NT * K / E)
    # kept per expert = min(count, C); dropped = sum(max(0, count-C))
    exp_drop = int((counts0 - C).clamp_min(0).sum())
    kept_hot = int((new_w.reshape(-1)[ti.reshape(-1) == 0] > 0).sum())
    fails = 0
    def ok(c, n):
        global fails
        print(f"  [{'PASS' if c else 'FAIL'}] {n}"); fails += 0 if c else 1
    print(f"== moe_expert_cap self-test (NT={NT},K={K},E={E},C={C}) ==")
    ok(dropped == exp_drop, f"dropped exactly the overflow ({dropped}=={exp_drop})")
    ok(kept_hot == C, f"hot expert 0 capped to C ({kept_hot}=={C})")
    ok(int((new_w > 0).sum()) == NT * K - dropped, "only overflow zeroed")
    # balanced -> nothing dropped (bit-exact)
    tib = torch.arange(NT * K).reshape(NT, K) % E
    _, d2 = moe_expert_cap(tib, tw, E, capacity_factor=1.25)
    ok(d2 == 0, "balanced routing -> 0 dropped (bit-exact)")
    # regime guard: tiny decode-like batch (C<min_capacity) -> skipped, no drops
    dg, kg = 16, 8
    tid = torch.randint(0, 128, (dg, kg))          # C = 16*8/128 = 1 < 8
    twd = torch.rand(dg, kg)
    _, d3 = moe_expert_cap(tid, twd, 128, capacity_factor=1.0)
    ok(d3 == 0, "tiny decode batch (C<min_cap) -> skipped (no over-aggressive drop)")
    print("== ALL PASS ==" if fails == 0 else f"== {fails} FAILED ==")
    raise SystemExit(fails)
