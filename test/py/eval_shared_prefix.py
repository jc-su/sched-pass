"""eval_shared_prefix.py -- does SHAPE survive on FlashInfer via cross-request
SHARED-PREFIX reuse? Two decode batches that READ the same number of KV pages:
  * SHARED : N requests share a large prefix (same physical pages) + unique tail.
             Few DISTINCT pages -> the shared prefix is re-read N times (reuse).
  * UNIQUE : N requests, all-distinct pages -> no reuse (the earlier 1%-L2 case).
If SHARED << UNIQUE makespan, the HW's L2 ALREADY captures the shared-prefix
reuse (SHAPE subsumed even here). If SHARED ~= UNIQUE, the HW evicts the shared
prefix under pressure -> SHAPE-pinning (control plane knows it is shared via the
radix tree) has room. This decides SHAPE's survival in the common shared-prompt
regime.

Run: FLASHINFER_NVCC=/usr/local/cuda-12.9/bin/nvcc python test/py/eval_shared_prefix.py
"""
import os
import sys

import torch

PAGE, HD, NQO, NKV = 16, 128, 16, 2
N = 256                 # requests
P = 240                 # shared-prefix pages (3840 tokens)
Sfx = 16                # unique tail pages per request (256 tokens)
PER = P + Sfx           # pages/request (both batches read this many)
ITERS = 60


def step_us(w, q, kv, iters=ITERS):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(iters):
        w.run(q, kv)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters


def makespan(kv_indices_rows, npages_distinct, dev):
    import flashinfer
    indptr = torch.arange(0, N * PER + 1, PER, dtype=torch.int32, device=dev)
    indices = torch.cat(kv_indices_rows).to(torch.int32).cuda()
    last = torch.full((N,), PAGE, dtype=torch.int32, device=dev)
    q = torch.randn(N, NQO, HD, dtype=torch.float16, device=dev)
    kv = torch.randn(npages_distinct, 2, PAGE, NKV, HD, dtype=torch.float16, device=dev)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev), "NHD")
    w.plan(indptr, indices, last, NQO, NKV, HD, PAGE,
           data_type=torch.float16, q_data_type=torch.float16)
    for _ in range(5):
        w.run(q, kv)
    torch.cuda.synchronize()
    return min(step_us(w, q, kv) for _ in range(3))


def main():
    dev = "cuda"
    # SHARED: pages [0,P) shared; unique tail [P + i*Sfx, ...)
    shared_rows = [torch.cat([torch.arange(P),
                              torch.arange(P + i * Sfx, P + i * Sfx + Sfx)])
                   for i in range(N)]
    shared_distinct = P + N * Sfx
    # UNIQUE: every request all-distinct
    uniq_rows = [torch.arange(i * PER, i * PER + PER) for i in range(N)]
    uniq_distinct = N * PER

    pages_read = N * PER
    print(f"== shared-prefix SHAPE test (N={N}, {PER} pages/req, {pages_read} "
          f"pages READ both; shared prefix={P} pages re-read {N}x) ==")
    ms = makespan(shared_rows, shared_distinct, dev)
    mu = makespan(uniq_rows, uniq_distinct, dev)
    print(f"  SHARED  ({shared_distinct} distinct pages)   {ms:8.1f} us")
    print(f"  UNIQUE  ({uniq_distinct} distinct pages)   {mu:8.1f} us")
    print(f"  shared vs unique: {100*(ms-mu)/mu:+.1f}%")
    print("== SHARED << UNIQUE => HW L2 already captures shared reuse (SHAPE "
          "subsumed); ~equal => HW evicts it, SHAPE-pinning has room ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
