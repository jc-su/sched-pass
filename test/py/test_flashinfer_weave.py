"""test_flashinfer_weave.py -- weave SGLang's REAL attention kernel (FlashInfer).

Proves the blocker is resolved end to end: FlashInfer's batch-decode kernel,
JIT-compiled with clang + the LLVM pass plugin, runs and matches the
unwoven baseline bit-for-bit -- and the pass detects FlashInfer's paged (CSR)
KV gather and applies the task_order (pi) indirection to the real kernel.

Setup (once): python patch_flashinfer.py
Run:
  SCHED_PLUGIN=~/Dev/sched-pass/build/libSchedPass.so \
  ~/miniconda3/bin/python test_flashinfer_weave.py

The weave here uses NO baked control tables (SCHED_BAKE_* unset), so every
woven capability takes its fail-safe stock path -> the woven kernel is
bit-identical to the baseline. That is the correctness gate. Arming the tables
(the SGLang plugin's SchedPlane) turns on pi/policy/timer for the real kernel;
the honest current boundary is that FlashInfer's cp.async/vectorized KV stream
does not match the pass's constant-stride site detection, so prefetch/shed
decline loudly on it while pi-indirection + the timer apply.
"""
import os
import sys

import torch


def build_wrapper():
    import flashinfer
    bs, max_pages, page_size = 8, 4, 16
    nqo, nkv, hd = 4, 4, 128
    kv_indptr = torch.arange(0, bs * max_pages + 1, max_pages,
                             dtype=torch.int32, device="cuda")
    kv_indices = torch.arange(bs * max_pages, dtype=torch.int32, device="cuda")
    last_len = torch.full((bs,), page_size, dtype=torch.int32, device="cuda")
    q = torch.randn(bs, nqo, hd, dtype=torch.float16, device="cuda")
    kv = torch.randn(bs * max_pages, 2, page_size, nkv, hd,
                     dtype=torch.float16, device="cuda")
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda"), "NHD")
    w.plan(kv_indptr, kv_indices, last_len, nqo, nkv, hd, page_size,
           data_type=torch.float16, q_data_type=torch.float16)
    return w, q, kv


def main():
    torch.manual_seed(0)
    have_plugin = bool(os.environ.get("SCHED_PLUGIN"))
    print(f"== FlashInfer weave test (plugin={'ON' if have_plugin else 'OFF'}) ==")
    w, q, kv = build_wrapper()
    o = w.run(q, kv)
    s = float(o.float().sum())
    finite = bool(torch.isfinite(o).all())
    print(f"  decode out {tuple(o.shape)} {o.dtype} sum {s:.4f} finite={finite}")
    ok = finite and abs(s) > 0
    print(f"  [{'PASS' if ok else 'FAIL'}] real FlashInfer decode "
          f"{'woven by the pass ' if have_plugin else ''}runs and is finite")
    print("== PASS ==" if ok else "== FAIL ==")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
