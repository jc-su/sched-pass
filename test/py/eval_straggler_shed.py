"""eval_straggler_shed.py -- the moat, made concrete: the CONTROL PLANE detects
the straggler (global view: which request dominates the makespan) and steers the
kernel to do LESS for EXACTLY it (E2 shed = cap its attended KV). Stock kernels
are per-request-blind and can't do this; the straggler blocks the whole batch.

Bimodal batch (the straggler shape): 8% heavy (512 pages) + 92% light (1 page).
Baseline = full KV (the straggler dominates). Then cap the STRAGGLERS' KV to a
budget (leave the light requests bit-exact) and measure the makespan collapse vs
the accuracy traded (fraction of the straggler's KV attended).

Run:
  FLASHINFER_NVCC=/usr/local/cuda-12.9/bin/nvcc python test/py/eval_straggler_shed.py
"""
import os
import sys

import torch

PAGE, HD, NQO, NKV = 16, 128, 16, 2
BS = 1024
ITERS = 60


def step_us(w, q, kv, iters=ITERS):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(iters):
        w.run(q, kv)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters


def makespan(pages_capped, kv, dev):
    import flashinfer
    n = pages_capped.numel()
    indptr = torch.zeros(n + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(pages_capped, 0)
    total = int(indptr[-1])
    indptr = indptr.cuda()
    indices = torch.arange(total, dtype=torch.int32, device=dev) % kv.shape[0]
    last = torch.full((n,), PAGE, dtype=torch.int32, device=dev)
    q = torch.randn(n, NQO, HD, dtype=torch.float16, device=dev)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev), "NHD")
    w.plan(indptr, indices, last, NQO, NKV, HD, PAGE,
           data_type=torch.float16, q_data_type=torch.float16)
    for _ in range(4):
        w.run(q, kv)
    torch.cuda.synchronize()
    return min(step_us(w, q, kv) for _ in range(3))


def main():
    dev = "cuda"
    gen = torch.Generator().manual_seed(1)
    heavy = (torch.rand(BS, generator=gen) < 0.08)
    full = torch.where(heavy, 512, 1).to(torch.int32)     # the straggler batch
    maxp = int(full.max())
    kv = torch.randn(maxp * BS // 4, 2, PAGE, NKV, HD, dtype=torch.float16, device=dev)

    print(f"== straggler shed: control plane caps the STRAGGLERS' KV "
          f"(BS={BS}, {int(heavy.sum())} stragglers @512 pages + rest @1) ==")
    base = makespan(full, kv, dev)
    print(f"  cap=512 (baseline, straggler unshed)   {base:8.1f} us   "
          f"attend 100%")
    for cap in (256, 128, 64, 32, 16):
        capped = torch.minimum(full, torch.tensor(cap, dtype=torch.int32))
        t = makespan(capped, kv, dev)
        # accuracy proxy: fraction of the straggler's KV still attended
        attend = 100.0 * cap / 512
        print(f"  cap={cap:<4} (stragglers shed)             {t:8.1f} us   "
              f"{100*(t-base)/base:+.1f}% makespan   attend {attend:.0f}% of "
              f"straggler KV")
    print("== the control plane knows WHICH request is the straggler (global "
          "view); the kernel sheds EXACTLY it (local action) -- the moat ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
