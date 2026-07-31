"""eval_split_skew.py -- does giving a decode straggler MORE parallelism help,
or is it DRAM-walled? Hold TOTAL KV work constant (same DRAM traffic) and vary
how it is split across requests: 1 long request (max serial) -> many short
(max parallel). FlashInfer split-kv parallelizes the long one internally.
  * makespan INVARIANT to parallelism -> the straggler is DRAM-bandwidth-walled;
    more SMs just wait on DRAM -> SPLIT-SKEW FUTILE (same physics as SHAPE).
  * makespan DROPS with parallelism -> the long request is under-parallelized ->
    split-skew has room (control-plane-optimal split could beat the heuristic).

Run: FLASHINFER_NVCC=/usr/local/cuda-12.9/bin/nvcc python test/py/eval_split_skew.py
"""
import os
import sys

import torch

PAGE, HD, NQO, NKV = 16, 128, 16, 2
TOTAL_PAGES = 8192            # fixed total KV work (fixed DRAM traffic)
ITERS = 60


def step_us(w, q, kv, iters=ITERS):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(iters):
        w.run(q, kv)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters


def makespan(nreq, pages_each, kv, dev):
    import flashinfer
    pages = torch.full((nreq,), pages_each, dtype=torch.int32)
    indptr = torch.zeros(nreq + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(pages, 0)
    total = int(indptr[-1]); indptr = indptr.cuda()
    indices = torch.arange(total, dtype=torch.int32, device=dev) % kv.shape[0]
    last = torch.full((nreq,), PAGE, dtype=torch.int32, device=dev)
    q = torch.randn(nreq, NQO, HD, dtype=torch.float16, device=dev)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev), "NHD")
    w.plan(indptr, indices, last, NQO, NKV, HD, PAGE,
           data_type=torch.float16, q_data_type=torch.float16)
    v = getattr(w, "_plan_info", None)
    vv = v.tolist() if hasattr(v, "tolist") else list(v)
    ntiles = int(vv[0]) if vv and int(vv[0]) > 0 else nreq
    for _ in range(4):
        w.run(q, kv)
    torch.cuda.synchronize()
    return min(step_us(w, q, kv) for _ in range(3)), ntiles


def main():
    dev = "cuda"
    kv = torch.randn(TOTAL_PAGES, 2, PAGE, NKV, HD, dtype=torch.float16, device=dev)
    print(f"== split-skew: fixed total KV={TOTAL_PAGES} pages, vary parallelism "
          f"(nreq x pages) ==")
    print(f"{'nreq':>6} {'pages/req':>10} {'ntiles(FI split)':>17} {'makespan':>10}")
    for nreq, pe in ((1, 8192), (8, 1024), (64, 128), (512, 16), (2048, 4)):
        t, ntiles = makespan(nreq, pe, kv, dev)
        print(f"{nreq:>6} {pe:>10} {ntiles:>17} {t:>9.1f}u")
    print("== INVARIANT makespan => DRAM-walled, split-skew FUTILE for decode; "
          "DROPS with parallelism => under-parallelized, split-skew has room ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
