"""eval_moe_imbalance.py -- the MoE regime: does EXPERT IMBALANCE (data-dependent
routing) hurt the grouped GEMM, opening room for the moat? SegmentGEMM's seg_lens
ARE the routing counts (tokens per expert). Same TOTAL tokens (same total FLOPs),
compare BALANCED routing (uniform experts) vs IMBALANCED (a hot expert -- real
top-k routing). If imbalanced makespan >> balanced, the hot expert is a straggler
the kernel does not absorb -> the control plane (which sees the routing counts)
can skew/schedule -> the moat's regime. Unlike decode, MoE cost is only known
AFTER routing, per step -- exactly what the woven observation is for.

Run: FLASHINFER_NVCC=/usr/local/cuda-12.9/bin/nvcc python test/py/eval_moe_imbalance.py
"""
import os
import sys

import torch

E, D, DO = 16, 2048, 2048       # experts, in-dim, out-dim
T = 16384                        # total tokens (fixed -> fixed total FLOPs)
ITERS = 40


def step_us(fn, iters=ITERS):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters


def main():
    dev = "cuda"
    import flashinfer
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    seg = flashinfer.SegmentGEMMWrapper(ws)
    weights = torch.randn(E, D, DO, dtype=torch.float16, device=dev) * 0.02
    x = torch.randn(T, D, dtype=torch.float16, device=dev)

    def makespan(seg_lens):
        sl = seg_lens.to(torch.int64).cuda()
        run = lambda: seg.run(x, weights, E, False, seg_lens=sl)
        for _ in range(4):
            run()
        torch.cuda.synchronize()
        return min(step_us(run) for _ in range(3))

    print(f"== MoE grouped-GEMM: expert imbalance (E={E}, D={D}, T={T} tokens, "
          f"fixed total FLOPs) ==")
    bal = torch.full((E,), T // E, dtype=torch.int64)
    print(f"  balanced (uniform {T//E}/expert)        {makespan(bal):8.1f} us")
    for hot_frac in (0.25, 0.5, 0.75):
        hot = int(T * hot_frac)
        rest = torch.full((E - 1,), (T - hot) // (E - 1), dtype=torch.int64)
        im = torch.cat([torch.tensor([hot + (T - hot - int(rest.sum()))]), rest])
        p99 = float(im.max()) / (T / E)
        print(f"  imbalanced (1 hot={hot}, {p99:.1f}x mean)  "
              f"{makespan(im):8.1f} us   {100*(makespan(im)-makespan(bal))/makespan(bal):+.0f}% vs balanced")
    print("== imbalanced >> balanced => the hot expert is a straggler the kernel "
          "does NOT absorb -> control-plane skew/schedule room (the MoE moat) ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
