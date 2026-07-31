"""test_flashinfer_prefill.py -- prove the woven FlashInfer PREFILL kernel is
BIT-EXACT under a task_order permutation (E1), the runtime validation of the
prefill-under-clang unblock. Mirrors test_flashinfer_arm.py but on
BatchPrefillWithPagedKVCacheWrapper (BatchPrefillWithPagedKVCacheKernel).

Weaves prefill (SCHED_WEAVE_ONLY=batch_prefill). Dispersed prompt lengths so
the plan splits long requests across multiple qo-tiles -- the exact
"one request -> several tiles" case.

Run (armed):
  SCHED_PLUGIN=.../libSchedPass.so FLASHINFER_NVCC="python3 .../nvcc_clang_shim.py" \
  SCHED_CLANG=clang++-22 SCHED_CUDA_PATH=/usr/local/cuda-12.9 SCHED_ARCH=sm_120a \
  SCHED_WEAVE_ONLY=batch_prefill python test/py/test_flashinfer_prefill.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))  # library
from sched_rt import SchedPlane

PAGE, HD, NQO, NKV = 16, 128, 16, 2      # Qwen-3B GQA shape


def main():
    os.environ.setdefault("SCHED_WEAVE_ONLY", "batch_prefill")
    dev = "cuda"
    plane = SchedPlane(max_tasks=8192, device=dev)
    # Timer channel: host-mapped by default; SCHED_TEST_DEVICE_TIMER=1 selects
    # the device-L2 channel (the SERVING default). The device channel used to
    # FAULT prefill -- but that was on the BROKEN scalar kernel (pre the
    # __CUDACC_VER_MAJOR__ fast-path unblock); re-validated here on the correct
    # fast-path kernel.
    if os.environ.get("SCHED_TEST_DEVICE_TIMER") == "1":
        plane.use_device_timer()
        print("  [cfg] device-L2 timer channel (serving default)")
    plane.arm_process_env(os.environ["SCHED_PLUGIN"])
    import flashinfer

    # dispersed prompt lengths (tokens) -> some long requests split across tiles
    gen = torch.Generator().manual_seed(7)
    B = 48
    qo_len = torch.exp(torch.normal(4.5, 0.8, (B,), generator=gen)) \
        .clamp(16, 1024).to(torch.int32)                 # ~90 median, up to 1024
    pages = ((qo_len + PAGE - 1) // PAGE).to(torch.int32)  # KV = own prompt
    npages = int(pages.sum())
    total_qo = int(qo_len.sum())

    qo_indptr = torch.zeros(B + 1, dtype=torch.int32)
    qo_indptr[1:] = torch.cumsum(qo_len, 0)
    kv_indptr = torch.zeros(B + 1, dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(pages, 0)
    last = ((qo_len - 1) % PAGE + 1).to(torch.int32)
    qo_indptr, kv_indptr, last = qo_indptr.cuda(), kv_indptr.cuda(), last.cuda()
    kv_indices = torch.arange(npages, dtype=torch.int32, device=dev)
    q = torch.randn(total_qo, NQO, HD, dtype=torch.float16, device=dev)
    kv = torch.randn(npages, 2, PAGE, NKV, HD, dtype=torch.float16, device=dev)

    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev), "NHD")
    w.plan(qo_indptr, kv_indptr, kv_indices, last, NQO, NKV, HD, PAGE,
           causal=True, q_data_type=torch.float16, kv_data_type=torch.float16)

    # tiles the plan produced (grid over qo-tiles); pi permutes them
    v = getattr(w, "_plan_info", None)
    v = (v.tolist() if hasattr(v, "tolist") else list(v)) if v is not None else []
    ntiles = int(v[0]) if (len(v) >= 1 and int(v[0]) > 0) else B

    fails = [0]

    def ok(cond, name):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        fails[0] += 0 if cond else 1

    print(f"== FlashInfer PREFILL armed-pi test (B={B}, {total_qo} qo tokens, "
          f"{ntiles} tiles, dispersed) ==")
    plane.set_num_tasks(0)
    plane.reset_order()
    plane.set_timer_enabled(False)
    plane.push()
    for _ in range(3):
        w.run(q, kv)                                     # warm / JIT
    torch.cuda.synchronize()

    golden = w.run(q, kv).clone()
    ok(torch.isfinite(golden).all(), "identity (woven) produces finite output")

    # reversed order over the real tile count -> E1 permutation, must be exact
    rev = torch.arange(plane.N, dtype=torch.int32)
    rev[:ntiles] = torch.arange(ntiles - 1, -1, -1, dtype=torch.int32)
    plane.install_order(rev, ntiles)
    plane.push()
    permuted = w.run(q, kv).clone()
    ok(torch.equal(permuted, golden),
       "reversed task_order -> BIT-EXACT vs identity (pi is E1 on prefill)")

    # timer fires on the woven prefill kernel
    plane.reset_order()
    plane.set_timer_enabled(True)
    plane.clear_timer()
    plane.push()
    w.run(q, kv)
    torch.cuda.synchronize()
    cyc = plane.read_timer()[:ntiles]
    ok(int((cyc > 0).sum()) > 0,
       f"woven timer populated on prefill ({int((cyc > 0).sum())}/{ntiles} "
       f"tiles > 0)")

    # --- EXACT per-request attribution under split (one req -> several tiles) --
    # Read request_indices via the SAME production code path the serving plugin
    # uses (_plan_request_indices), then fold the woven per-tile timer into
    # per-request cost. A long request whose QO/KV was split across several grid
    # tiles must be attributed to EXACTLY that one request -- proven here by the
    # binding being a valid PARTITION of tiles into requests (non-decreasing,
    # covers all), so scatter_add attributes each tile to exactly one owner.
    from sched_sglang_plugin import _plan_request_indices
    tile_to_req, pt = _plan_request_indices(w, B, 15, 4, require_flag=None)
    ok(tile_to_req is not None, "prefill request_indices readable (15-int plan)")
    if tile_to_req is not None:
        tr = torch.tensor(tile_to_req[:pt], dtype=torch.long)
        ok(bool((tr[1:] >= tr[:-1]).all()),
           "request_indices non-decreasing (tiles grouped by request)")
        covers = int(tr.unique().numel())
        ok(covers == B, f"binding covers every request ({covers}/{B})")
        counts = torch.bincount(tr.clamp(0, B - 1), minlength=B)
        ok(int(counts.max()) > 1,
           f"a request spans multiple tiles (max {int(counts.max())}/req -> "
           f"split case exercised)")
        cyc_all = plane.read_timer()[:pt].to(torch.int64)
        per_req = torch.zeros(B, dtype=torch.int64) \
            .scatter_add_(0, tr.clamp(0, B - 1), cyc_all)
        got = int((per_req > 0).sum())
        ok(got == B, f"every request attributed a folded cost ({got}/{B} > 0)")
        lo = int(qo_len.argmax())
        print(f"  [info] longest req {lo}: qo_len={int(qo_len[lo])} -> "
              f"{int(counts[lo])} tiles, exact folded cost="
              f"{int(per_req[lo])} cyc (no other request's tiles leak in)")

    print("== ALL PASS ==" if fails[0] == 0 else f"== {fails[0]} FAILED ==")
    return fails[0]


if __name__ == "__main__":
    raise SystemExit(main())
