"""eval_clc_vs_static.py -- P2: does the DYNAMIC CLC work-queue beat STATIC
grouped-LPT? Isolated, 1D grid (num_kv_heads=1) so the CLC claim path actually
ENGAGES (on a 2D GQA grid the driver's shape guard takes the stock path, so CLC
is a no-op there -- that is exactly why this must be measured on 1D, and why a
2D extension is the open question this bench informs).

Mode auto-selected from the JIT env:
  no SCHED_WORKQUEUE  -> STATIC pi kernel; arm grouped-LPT order (block 16).
  SCHED_WORKQUEUE=1 SCHED_CLC=1 -> CLC persistent-worker; arm num_tasks=ntiles
                                   + cost-ordered fill (region_order).
Prints, per BS: identity-order step us (baseline) and best-ordered step us (the
comparable number). Run BOTH phases (driver below) and compare the ORDERED us --
lower wins. Same seed/shape across phases so the batches are identical.

Driver (runs both, compares):
  SCHED_PLUGIN=... FLASHINFER_NVCC="python3 .../nvcc_clang_shim.py" \
  SCHED_CLANG=clang++-22 SCHED_CUDA_PATH=/usr/local/cuda-12.9 SCHED_ARCH=sm_120a \
  python test/py/eval_clc_vs_static.py --drive
"""
import os
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
from sched_rt import SchedPlane

PAGE, HD, NQO, NKV = 16, 128, 8, 1         # MQA: 1 KV head -> 1D grid -> CLC
#   engages; group_size = NQO/NKV = 8 (supported), same per-tile work as GQA-3B
ITERS = 80
MU, SIGMA = 2.6, 1.1
BSS = ([int(os.environ["SCHED_BENCH_BS"])] if os.environ.get("SCHED_BENCH_BS")
       else [512, 1024, 2048, 4096, 8192])


def step_us(w, q, kv, iters=ITERS):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(iters):
        w.run(q, kv)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters


def build(bs, dev, gen):
    if os.environ.get("SCHED_BIMODAL"):
        # extreme heterogeneity: 8% very-heavy (512 pages) + 92% light (1) --
        # the mixed prefill/decode shape (512x tile-cost spread). Does the
        # dynamic CLC claim rescue THIS where mild lognormal dispersion did not?
        heavy = (torch.rand(bs, generator=gen) < 0.08)
        pages = torch.where(heavy, 512, 1).to(torch.int32)
    else:
        pages = torch.exp(torch.normal(MU, SIGMA, (bs,), generator=gen)) \
            .clamp(1, 512).to(torch.int32)
    npages = int(pages.sum())
    indptr = torch.zeros(bs + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(pages, 0)
    indptr = indptr.cuda()
    indices = torch.arange(npages, dtype=torch.int32, device=dev)
    last = torch.full((bs,), PAGE, dtype=torch.int32, device=dev)
    q = torch.randn(bs, NQO, HD, dtype=torch.float16, device=dev)
    kv = torch.randn(npages, 2, PAGE, NKV, HD, dtype=torch.float16, device=dev)
    import flashinfer
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev), "NHD")
    w.plan(indptr, indices, last, NQO, NKV, HD, PAGE,
           data_type=torch.float16, q_data_type=torch.float16)
    return w, q, kv, pages


def phase():
    clc = bool(os.environ.get("SCHED_WORKQUEUE"))
    mode = "CLC-queue" if clc else "static-pi"
    dev = "cuda"
    plane = SchedPlane(max_tasks=16384, device=dev)
    plane.use_device_timer(); plane.set_timer_enabled(False)
    plane.arm_process_env(os.environ["SCHED_PLUGIN"])
    print(f"== P2 phase: {mode} (nkv={NKV} -> 1D grid) ==")
    print(f"{'BS':>6} {'identity':>10} {'ordered':>10} {'ord-vs-id':>10}")
    gen = torch.Generator().manual_seed(1)
    for bs in BSS:
        w, q, kv, pages = build(bs, dev, gen)
        cost = pages.to(torch.float32)
        # SAME fill order (grouped-LPT) in both phases -> isolates the DRAIN
        # mechanism: static-pi vs CLC dynamic claim. Fill held constant.
        order = SchedPlane._grouped_lpt(cost, 16)
        for _ in range(4):
            w.run(q, kv)
        torch.cuda.synchronize()
        # identity baseline: disarm (num_tasks 0 kills the CLC claim; static
        # kernel just uses ctaid)
        plane.set_num_tasks(0); plane.reset_order(); plane.push()
        t_id = min(step_us(w, q, kv) for _ in range(5))
        # armed: CLC needs num_tasks=ntiles to run the claim loop; static only
        # needs the order installed
        if clc:
            plane.set_num_tasks(bs)
        plane.install_order(order, bs); plane.push()
        t_ord = min(step_us(w, q, kv) for _ in range(5))
        plane.set_num_tasks(0); plane.reset_order(); plane.push()
        print(f"{bs:>6} {t_id:>9.1f}u {t_ord:>9.1f}u "
              f"{100*(t_ord-t_id)/t_id:>+9.2f}%")
        del w, q, kv
        torch.cuda.empty_cache()
    return 0


def drive():
    env = dict(os.environ)
    base = [sys.executable, os.path.abspath(__file__)]
    print("### PHASE 1: STATIC (grouped-LPT) ###")
    subprocess.run(base, env={k: v for k, v in env.items()
                              if k not in ("SCHED_WORKQUEUE", "SCHED_CLC")})
    print("\n### PHASE 2: CLC work-queue ###")
    subprocess.run(base, env=dict(env, SCHED_WORKQUEUE="1", SCHED_CLC="1"))
    print("\n== compare the 'ordered' us column across phases: lower wins. "
          "static works on 2D GQA today; CLC needs 1D (or a 2D extension) ==")


if __name__ == "__main__":
    if "--drive" in sys.argv:
        drive()
    else:
        raise SystemExit(phase())
