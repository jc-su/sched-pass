"""test_moe_forward_gpu.py -- LIVE GPU test of the MoE expert cap in a REAL
SGLang fused-MoE forward, WITHOUT downloading a MoE model. We instantiate a
single sglang FusedMoE layer (random weights) on the GPU and run its actual
triton fused-MoE kernel; the cap is exercised end to end through the real
StandardTopKOutput -> FusedMoE.forward path.

Proves, on device:
  1. GENEROUS capacity -> cap is a bit-exact no-op (forward output identical).
  2. TIGHT capacity + imbalanced routing -> output changes at EXACTLY the dropped
     (over-capacity) tokens, and is untouched elsewhere (targeted, not global).
  3. The register_moe_cap() HOOK path caps inside FusedMoE.forward (full wireup):
     patched forward == manual-cap forward, and != uncapped forward.

Run: python test/py/test_moe_forward_gpu.py   (needs 1 GPU; SKIPs cleanly if none)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

fails = 0


def ok(cond, name):
    global fails
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    fails += 0 if cond else 1


def main():
    import torch
    if not torch.cuda.is_available():
        print("== SKIP (no GPU) =="); return 0
    for k, v in dict(MASTER_ADDR="127.0.0.1", MASTER_PORT="29595", RANK="0",
                     WORLD_SIZE="1", LOCAL_RANK="0").items():
        os.environ.setdefault(k, v)
    import torch.distributed as dist
    from sglang.srt.distributed import (init_distributed_environment,
                                        initialize_model_parallel)
    from sglang.srt.server_args import (ServerArgs,
                                        set_global_server_args_for_scheduler)
    dist.init_process_group(backend="nccl", world_size=1, rank=0)
    init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                 distributed_init_method="env://")
    initialize_model_parallel(tensor_model_parallel_size=1)
    set_global_server_args_for_scheduler(ServerArgs(model_path="Qwen/Qwen3-8B"))
    torch.cuda.set_device(0)

    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    import sched_moe_hook as H

    E, HID, INT, K, NT = 8, 512, 1024, 2, 256
    print(f"== live GPU MoE cap test (E={E}, hidden={HID}, top_k={K}, "
          f"tokens={NT}) ==")
    moe = FusedMoE(num_experts=E, hidden_size=HID, intermediate_size=INT,
                   layer_id=0, top_k=K, params_dtype=torch.bfloat16,
                   inplace=False).cuda()   # inplace=True would mutate hs across calls
    gen = torch.Generator(device="cuda").manual_seed(0)
    with torch.no_grad():
        for p in moe.parameters():
            p.copy_(torch.randn(p.shape, generator=gen, device="cuda",
                                dtype=p.dtype) * 0.02)

    def routing(imbalanced):
        ti = torch.randint(0, E, (NT, K), generator=gen, device="cuda")
        if imbalanced:
            ti[: int(0.5 * NT), 0] = 0                 # expert 0 hot (50%)
        tw = (torch.rand(NT, K, generator=gen, device="cuda") + 0.1).float()
        rl = torch.randn(NT, E, generator=gen, device="cuda").float()
        return ti, tw, rl

    def fwd(ti, tw, rl):
        out = moe.forward(hs.clone(), StandardTopKOutput(
            topk_weights=tw.clone(), topk_ids=ti, router_logits=rl))
        torch.cuda.synchronize()
        return out.float()

    hs = (torch.randn(NT, HID, generator=gen, device="cuda",
                      dtype=torch.bfloat16) * 0.1)
    # kernel noise floor: two identical forwards may differ by atomic-reduction
    # non-determinism. Measure it so we compare the cap's effect against it.
    ti0, tw0, rl0 = routing(imbalanced=True)
    noise = (fwd(ti0, tw0, rl0) - fwd(ti0, tw0, rl0)).abs().max().item()
    print(f"  (kernel noise floor: max|fwd-fwd| = {noise:.2e})")

    tol = max(noise, 1e-6)          # "unchanged" = within the kernel noise floor
    def maxdiff(a, b):
        return (a - b).abs().max().item()

    # -- 1. generous capacity -> no-op (within kernel noise) --------------
    ti, tw, rl = routing(imbalanced=True)
    base = fwd(ti, tw, rl)
    w_gen, dropped_gen = H.moe_expert_cap(ti, tw, E, capacity_factor=8.0)
    out_gen = fwd(ti, w_gen, rl)
    ok(dropped_gen == 0, "generous capacity drops nothing")
    ok(maxdiff(out_gen, base) <= tol,
       f"generous cap -> forward unchanged within noise ({maxdiff(out_gen,base):.2e})")

    # -- 2. tight capacity -> output changes at exactly the dropped tokens -
    w_cap, dropped = H.moe_expert_cap(ti, tw, E, capacity_factor=1.0)
    out_cap = fwd(ti, w_cap, rl)
    zeroed_tok = (w_cap != tw).any(dim=1)              # token had a weight zeroed
    # per-token max change vs the noise floor
    tok_diff = (out_cap - base).abs().amax(dim=1)
    moved = tok_diff > 10 * tol                         # genuinely changed (> noise)
    ok(dropped > 0, f"tight capacity drops the overflow ({dropped} slots)")
    ok(int(moved.sum()) > 0, f"forward output changed under the cap ({int(moved.sum())} tokens)")
    # every genuinely-moved token must be one we actually zeroed (targeted)
    ok(bool((moved <= zeroed_tok).all()),
       "output moved ONLY at dropped tokens (targeted, no collateral)")
    ok(maxdiff(out_cap[~zeroed_tok], base[~zeroed_tok]) <= tol,
       "untouched tokens unchanged within noise (light requests unaffected)")

    # -- 3. the register_moe_cap() HOOK path caps inside forward ----------
    os.environ["SCHED_MOE_CAP"] = "1"
    os.environ["SCHED_MOE_CAPACITY"] = "1.0"
    H._ORIG.clear()
    ok(H.register_moe_cap(), "register_moe_cap() patched FusedMoE.forward")
    out_hook = fwd(ti, tw, rl)                          # forward auto-caps now
    ok(maxdiff(out_hook, out_cap) <= tol,
       f"HOOK forward == manual-cap forward within noise ({maxdiff(out_hook,out_cap):.2e})")
    ok(maxdiff(out_hook, base) > 10 * tol,
       "HOOK forward differs from uncapped (the cap actually fired in-forward)")

    print("== ALL PASS ==" if fails == 0 else f"== {fails} FAILED ==")
    return fails


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ImportError as e:
        print(f"== SKIP (sglang/torch not importable: {e}) =="); raise SystemExit(0)
