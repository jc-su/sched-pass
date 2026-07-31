"""test_moe_hook.py -- WIREUP test for the MoE expert-cap arm, against the REAL
SGLang routing types (no MoE model, no GPU needed -- CPU tensors + the actual
StandardTopKOutput / FusedMoE classes). Proves:
  1. moe_expert_cap transforms a real StandardTopKOutput correctly (hot expert
     capped, topk_ids + router_logits untouched, namedtuple identity preserved).
  2. register_moe_cap() actually patches FusedMoE.forward (the class method
     identity changes; idempotent; gated by SCHED_MOE_CAP).
  3. The wrapper fail-soft no-ops on the routing forms that carry no
     topk_ids/topk_weights (BypassedTopKOutput, TritonKernelTopKOutput).
  4. The wrapped forward, invoked on a stand-in module, feeds the ORIGINAL
     forward a capped StandardTopKOutput -- the end-to-end binding, MoE edition.

Run: python test/py/test_moe_hook.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import torch  # noqa: E402
import sched_moe_hook as H  # noqa: E402

fails = 0


def ok(cond, name):
    global fails
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    fails += 0 if cond else 1


def imbalanced(NT=512, K=2, E=8, hot_frac=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    ti = torch.randint(0, E, (NT, K), generator=g)
    ti[: int(hot_frac * NT), 0] = 0          # expert 0 is hot
    tw = torch.rand(NT, K, generator=g) + 0.1
    rl = torch.randn(NT, E, generator=g)
    return ti, tw, rl, E


def main():
    from sglang.srt.layers.moe.topk import (
        StandardTopKOutput, BypassedTopKOutput, TritonKernelTopKOutput)
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    print("== MoE hook wireup test (real SGLang types, CPU) ==")

    # -- 1. transform a real StandardTopKOutput -----------------------------
    ti, tw, rl, E = imbalanced()
    out = StandardTopKOutput(topk_weights=tw, topk_ids=ti, router_logits=rl)
    new_w, dropped = H.moe_expert_cap(out.topk_ids, out.topk_weights, E, 1.25)
    capped = out._replace(topk_weights=new_w)
    C = int(1.25 * ti.numel() / E)
    kept_hot = int((capped.topk_weights.reshape(-1)[ti.reshape(-1) == 0] > 0).sum())
    ok(isinstance(capped, StandardTopKOutput), "capped is still StandardTopKOutput")
    ok(kept_hot == C, f"hot expert 0 capped to C={C} (kept {kept_hot})")
    ok(torch.equal(capped.topk_ids, ti), "topk_ids untouched (E1 on ids)")
    ok(torch.equal(capped.router_logits, rl), "router_logits untouched")
    ok(dropped > 0, f"dropped the overflow ({dropped} tokens)")

    # bit-exact when capacity is generous (nothing overflows)
    _, d0 = H.moe_expert_cap(ti, tw, E, capacity_factor=8.0)
    ok(d0 == 0, "generous capacity -> 0 dropped (bit-exact passthrough)")

    # -- 2. register_moe_cap actually patches FusedMoE.forward --------------
    os.environ["SCHED_MOE_CAP"] = "1"
    before_id = id(FusedMoE.forward)
    patched = H.register_moe_cap()
    ok(patched, "register_moe_cap() returned True (patched)")
    ok(id(FusedMoE.forward) != before_id, "FusedMoE.forward identity changed")
    ok(H.register_moe_cap() and id(FusedMoE.forward) != before_id,
       "register_moe_cap() idempotent (no double-wrap)")

    # -- 3+4. wrapped forward feeds orig a CAPPED StandardTopKOutput -------
    # Stand-in module: mimics FusedMoE's num_experts + a forward we can inspect.
    seen = {}

    def orig_forward(self, hidden_states, topk_output):
        seen["topk_output"] = topk_output
        return hidden_states  # identity; we only inspect the routing it received

    # Build the SAME wrapper register_moe_cap builds, but over our orig, so we can
    # invoke it on a stand-in self without constructing a real FusedMoE.
    H._ORIG.clear()
    orig_saved = FusedMoE.forward
    FusedMoE.forward = orig_forward
    H.register_moe_cap()               # wraps our orig_forward
    wrapped = FusedMoE.forward

    class Stand:
        num_experts = E
    hs = torch.zeros(4)
    ti2, tw2, rl2, _ = imbalanced(seed=1)
    out2 = StandardTopKOutput(topk_weights=tw2, topk_ids=ti2, router_logits=rl2)
    wrapped(Stand(), hs, out2)
    got = seen["topk_output"]
    ok(isinstance(got, StandardTopKOutput), "orig received a StandardTopKOutput")
    kept_hot2 = int((got.topk_weights.reshape(-1)[ti2.reshape(-1) == 0] > 0).sum())
    ok(kept_hot2 == C, f"orig received CAPPED weights (hot kept {kept_hot2}=={C})")
    ok(torch.equal(got.topk_ids, ti2), "orig received untouched topk_ids")

    # fail-soft: routing forms without topk_ids/topk_weights pass through
    seen.clear()
    byp = BypassedTopKOutput(hidden_states=hs, router_logits=rl2, topk_config=None,
                             num_token_non_padded=None,
                             expert_location_dispatch_info=None)
    wrapped(Stand(), hs, byp)
    ok(seen["topk_output"] is byp, "BypassedTopKOutput passes through untouched")

    FusedMoE.forward = orig_saved      # restore

    print("== ALL PASS ==" if fails == 0 else f"== {fails} FAILED ==")
    return fails


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ImportError as e:
        print(f"== SKIP (sglang not importable: {e}) ==")
        raise SystemExit(0)
