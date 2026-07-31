"""eval_moe_fast.py -- MoE with the FAST tensor-core grouped GEMM
(grouped_mm_bf16, cudnn). Does EXPERT IMBALANCE (data-dependent routing) hurt,
opening room for the moat? m_indptr = routing counts. Same TOTAL tokens (fixed
FLOPs), balanced vs imbalanced (a hot expert). If imbalanced >> balanced, the hot
expert is a straggler the kernel does not absorb -> control-plane schedule/skew
room (the MoE moat); if ~equal, cudlass/cudnn already balances it.

Run: FLASHINFER_NVCC=/usr/local/cuda-12.9/bin/nvcc python test/py/eval_moe_fast.py
"""
import os
import sys

import torch

E, K, N, T = 16, 4096, 4096, 16384      # experts, in, out, total tokens
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
    gmm = getattr(flashinfer, "grouped_mm_bf16", None) or \
        __import__("flashinfer.grouped_mm", fromlist=["grouped_mm_bf16"]).grouped_mm_bf16
    a = torch.randn(T, K, dtype=torch.bfloat16, device=dev)
    b = (torch.randn(E, K, N, dtype=torch.bfloat16, device=dev) * 0.02)

    def makespan(counts):
        mi = torch.zeros(E + 1, dtype=torch.int32)
        mi[1:] = torch.cumsum(counts, 0)
        mi = mi.cuda()
        try:
            run = lambda: gmm(a, b, mi, backend="cudnn")
            run()
        except Exception:
            run = lambda: gmm(a, b, mi)   # default backend
        for _ in range(4):
            run()
        torch.cuda.synchronize()
        return min(step_us(run) for _ in range(3))

    flop = 2 * T * K * N
    print(f"== MoE FAST grouped-GEMM (grouped_mm_bf16, E={E}, K=N={K}, "
          f"T={T}, {flop/1e9:.0f} GFLOP) ==")
    bal = torch.full((E,), T // E, dtype=torch.int64)
    tb = makespan(bal)
    print(f"  balanced ({T//E}/expert)         {tb:8.1f} us   "
          f"{flop/(tb*1e-6)/1e12:.0f} TFLOP/s")
    for frac in (0.25, 0.5, 0.75):
        hot = int(T * frac)
        rest = torch.full((E - 1,), (T - hot) // (E - 1), dtype=torch.int64)
        im = torch.cat([torch.tensor([T - int(rest.sum())]), rest])
        skew = float(im.max()) / (T / E)
        t = makespan(im)
        print(f"  imbalanced (hot {skew:.0f}x mean)     {t:8.1f} us   "
              f"{100*(t-tb)/tb:+.0f}% vs balanced")
    print("== imbalanced >> balanced => hot expert is a straggler the kernel does "
          "NOT absorb -> MoE moat room; ~equal => cudnn already balances ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
