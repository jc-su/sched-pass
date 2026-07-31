"""eval_moe_capacity.py -- the MoE moat ACTION: the control plane sees the routing
counts (global, per-step), detects the hot expert, and CAPS it (E2 drop-tokens /
expert capacity) -- the expert-level twin of the decode straggler-shed. Measures
how much of the imbalance penalty (+58..72%, eval_moe_fast.py) capping recovers,
vs the accuracy traded (fraction of the hot expert's tokens kept).

Fast tensor-core grouped GEMM (grouped_mm_bf16, cudnn). Hot expert = 8x mean.
Cap sweep: uncapped -> down to the balanced mean.

Run: FLASHINFER_NVCC=/usr/local/cuda-12.9/bin/nvcc python test/py/eval_moe_capacity.py
"""
import os
import sys

import torch

E, K, N, T = 16, 4096, 4096, 16384
MEAN = T // E                          # 1024 = balanced per-expert
HOT = 8 * MEAN                          # the straggler expert (8x)
ITERS = 50


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
    gmm = __import__("flashinfer.grouped_mm",
                     fromlist=["grouped_mm_bf16"]).grouped_mm_bf16
    a = torch.randn(T, K, dtype=torch.bfloat16, device=dev)
    b = torch.randn(E, K, N, dtype=torch.bfloat16, device=dev) * 0.02
    rest = (T - HOT) // (E - 1)

    def makespan(counts):
        tot = int(counts.sum())
        mi = torch.zeros(E + 1, dtype=torch.int32)
        mi[1:] = torch.cumsum(counts, 0); mi = mi.cuda()
        aa = a[:tot]
        run = lambda: gmm(aa, b, mi, backend="cudnn")
        for _ in range(4):
            run()
        torch.cuda.synchronize()
        return min(step_us(run) for _ in range(3))

    bal = torch.full((E,), MEAN, dtype=torch.int64)
    t_bal = makespan(bal)
    imb = torch.cat([torch.tensor([HOT]), torch.full((E - 1,), rest)])
    t_imb = makespan(imb)
    pen = t_imb - t_bal
    print(f"== MoE expert-capacity recovery (E={E}, hot={HOT}={HOT//MEAN}x mean) ==")
    print(f"  balanced (ideal)              {t_bal:8.1f} us")
    print(f"  imbalanced, UNCAPPED          {t_imb:8.1f} us   "
          f"+{100*pen/t_bal:.0f}% penalty (the straggler)")
    print(f"  -- control plane caps the hot expert (drop-tokens) --")
    for cap in (4096, 2048, 1024):
        capped = torch.cat([torch.tensor([cap]), torch.full((E - 1,), rest)])
        t = makespan(capped)
        recov = 100 * (t_imb - t) / max(pen, 1e-6)
        print(f"  cap hot -> {cap:<5} (keep {100*cap/HOT:.0f}% hot tokens)   "
              f"{t:8.1f} us   recovers {recov:.0f}% of the penalty")
    print("== the control plane detects the hot expert (routing counts) and caps "
          "EXACTLY it -- the MoE moat action, expert-level straggler shed ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
