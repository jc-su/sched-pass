"""eval_split_skew_prefill.py -- does split-skew revive for PREFILL stragglers?
Decode was DRAM-walled (parallelism-invariant). Prefill is L2/compute-bound -- a
DIFFERENT wall. Hold total prefill work ~constant (N requests x L tokens, causal
work ~ N*L^2) and vary parallelism: 1 long prompt (serial) -> many short
(parallel). If makespan DROPS with parallelism, a prefill straggler is
under-parallelized and split-skew (control-plane-optimal split) has room -- the
chunked-prefill case the PI raised. If INVARIANT, prefill is L2-walled too.

Run: FLASHINFER_NVCC=/usr/local/cuda-12.9/bin/nvcc python test/py/eval_split_skew_prefill.py
"""
import os
import sys

import torch

PAGE, HD, NQO, NKV = 16, 128, 16, 2
ITERS = 40


def step_us(w, q, kv, iters=ITERS):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(iters):
        w.run(q, kv)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters


def makespan(nreq, L, dev):
    import flashinfer
    qo = torch.full((nreq,), L, dtype=torch.int32)
    pages = ((qo + PAGE - 1) // PAGE).to(torch.int32)
    npages = int(pages.sum()); tot = int(qo.sum())
    qi = torch.zeros(nreq + 1, dtype=torch.int32); qi[1:] = torch.cumsum(qo, 0)
    ki = torch.zeros(nreq + 1, dtype=torch.int32); ki[1:] = torch.cumsum(pages, 0)
    last = ((qo - 1) % PAGE + 1).to(torch.int32)
    qi, ki, last = qi.cuda(), ki.cuda(), last.cuda()
    kvi = torch.arange(npages, dtype=torch.int32, device=dev)
    q = torch.randn(tot, NQO, HD, dtype=torch.float16, device=dev)
    kv = torch.randn(npages, 2, PAGE, NKV, HD, dtype=torch.float16, device=dev)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev), "NHD")
    w.plan(qi, ki, kvi, last, NQO, NKV, HD, PAGE, causal=True,
           q_data_type=torch.float16, kv_data_type=torch.float16)
    v = getattr(w, "_plan_info", None)
    vv = v.tolist() if hasattr(v, "tolist") else list(v)
    ntiles = int(vv[0]) if vv and int(vv[0]) > 0 else nreq
    for _ in range(4):
        w.run(q, kv)
    torch.cuda.synchronize()
    return min(step_us(w, q, kv) for _ in range(3)), ntiles


def main():
    dev = "cuda"
    # ~constant total causal work N*L^2 = 1024^2: (1,1024),(4,512),(16,256),(64,128)
    print(f"== split-skew PREFILL: ~fixed total work N*L^2, vary parallelism ==")
    print(f"{'nreq':>6} {'L':>6} {'N*L^2':>10} {'ntiles':>8} {'makespan':>10}")
    for nreq, L in ((1, 1024), (4, 512), (16, 256), (64, 128), (256, 64)):
        t, ntiles = makespan(nreq, L, dev)
        print(f"{nreq:>6} {L:>6} {nreq*L*L:>10} {ntiles:>8} {t:>9.1f}u")
    print("== DROPS with parallelism => prefill straggler under-parallelized, "
          "split-skew REVIVES (chunked-prefill); INVARIANT => L2-walled too ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
