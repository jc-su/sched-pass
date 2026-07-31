"""eval_pi_prefill.py -- does ORDER pay in the PREFILL regime (L2-bound, unlike
decode's DRAM-bound)? Measures woven prefill makespan under identity vs
grouped-LPT vs SPT, cost from the WOVEN TIMER (observation-driven). A different
bottleneck may make ORDER pay differently -- the point is to STOP testing only
the HW-subsumed decode regime and measure where the physics differ.

Run (armed):
  SCHED_PLUGIN=.../libSchedPass.so FLASHINFER_NVCC="python3 .../nvcc_clang_shim.py" \
  SCHED_CLANG=clang++-22 SCHED_CUDA_PATH=/usr/local/cuda-12.9 SCHED_ARCH=sm_120a \
  SCHED_WEAVE_ONLY=batch_prefill python test/py/eval_pi_prefill.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
from sched_rt import SchedPlane

PAGE, HD, NQO, NKV = 16, 128, 16, 2
ITERS = 40


def step_us(w, q, kv, iters=ITERS):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(iters):
        w.run(q, kv)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters


def main():
    dev = "cuda"
    plane = SchedPlane(max_tasks=16384, device=dev)
    plane.use_device_timer()
    plane.arm_process_env(os.environ["SCHED_PLUGIN"])

    for B in (48, 128, 256):
        gen = torch.Generator().manual_seed(7)
        qo = torch.exp(torch.normal(4.5, 0.8, (B,), generator=gen)) \
            .clamp(16, 1024).to(torch.int32)
        pages = ((qo + PAGE - 1) // PAGE).to(torch.int32)
        npages = int(pages.sum()); tot = int(qo.sum())
        qi = torch.zeros(B + 1, dtype=torch.int32); qi[1:] = torch.cumsum(qo, 0)
        ki = torch.zeros(B + 1, dtype=torch.int32); ki[1:] = torch.cumsum(pages, 0)
        last = ((qo - 1) % PAGE + 1).to(torch.int32)
        qi, ki, last = qi.cuda(), ki.cuda(), last.cuda()
        kvi = torch.arange(npages, dtype=torch.int32, device=dev)
        q = torch.randn(tot, NQO, HD, dtype=torch.float16, device=dev)
        kv = torch.randn(npages, 2, PAGE, NKV, HD, dtype=torch.float16, device=dev)
        import flashinfer
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev), "NHD")
        w.plan(qi, ki, kvi, last, NQO, NKV, HD, PAGE, causal=True,
               q_data_type=torch.float16, kv_data_type=torch.float16)
        v = getattr(w, "_plan_info", None)
        vv = v.tolist() if hasattr(v, "tolist") else list(v)
        ntiles = int(vv[0]) if vv and int(vv[0]) > 0 else B

        for _ in range(4):
            w.run(q, kv)
        torch.cuda.synchronize()

        def makespan(order):
            plane.set_timer_enabled(False)
            if order is None:
                plane.reset_order()
            else:
                plane.install_order(order, ntiles)
            plane.push()
            return min(step_us(w, q, kv) for _ in range(4))

        t_id = makespan(None)
        # observe true per-tile cost via the woven timer
        plane.set_timer_enabled(True); plane.clear_timer(); plane.reset_order()
        plane.push(); w.run(q, kv); torch.cuda.synchronize()
        cyc = plane.read_timer()[:ntiles].float()
        lpt = SchedPlane.region_order(cyc, 0)
        spt = torch.argsort(cyc, descending=False).to(torch.int32)
        t_lpt = makespan(lpt)
        t_spt = makespan(spt)
        plane.set_timer_enabled(False); plane.reset_order(); plane.push()
        print(f"  B={B:>4} tiles={ntiles:>5}  identity={t_id:8.1f}u  "
              f"LPT={t_lpt:8.1f}u ({100*(t_lpt-t_id)/t_id:+.1f}%)  "
              f"SPT={t_spt:8.1f}u ({100*(t_spt-t_id)/t_id:+.1f}%)")
        del w, q, kv
        torch.cuda.empty_cache()
    print("== ORDER on PREFILL (L2-bound regime): LPT<id<SPT => it pays here too;"
          " ~0 => the L2 bottleneck is order-invariant (a different verdict) ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
