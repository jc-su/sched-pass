"""eval_motivation.py -- collect and ARCHIVE the motivation data (real woven
FlashInfer decode, this GPU), saved as CSVs + raw arrays under data/ for
reproducible plotting (plot_motivation.py). Three datasets, one storyline:

  A. tile-time dispersion at serving scale  -> the straggler EXISTS
     (per-tile woven-timer cycles; raw dump at one bs for the distribution).
  B. dispersion vs batch size + R           -> split_kv does NOT remove it
     (the regime map: split-active / gap / queued; R from the woven kernel).
  C. closed-loop pi step time               -> the tail is RECLAIMABLE
     (identity / reversed / lpt-oracle / lpt-from-woven-timer; bit-exact).

Run (armed, like run_all.sh):
  SCHED_PLUGIN=.../libSchedPass.so FLASHINFER_NVCC="python3 .../nvcc_clang_shim.py" \
  SCHED_CLANG=clang++-22 SCHED_CUDA_PATH=/usr/local/cuda-12.9 SCHED_ARCH=sm_120a \
  python eval_motivation.py
"""
import csv
import glob
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))  # library
from sched_rt import SchedPlane

PAGE, HD, NKV = 16, 128, 1          # nkv=1 -> 1D grid, pure KV-length dispersion
MIX_MU, MIX_SIGMA = 2.5, 1.0        # sharegpt-ish lognormal page counts
BS_SWEEP = [128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]
BS_DIST = 2048                      # the bs whose raw tile-cycles we dump
MAX_TASKS = 16384


def mix_pages(bs, gen):
    return torch.exp(torch.normal(MIX_MU, MIX_SIGMA, (bs,), generator=gen)) \
        .clamp(1, 128).to(torch.int32)


def build(bs, pages, dev):
    npages = int(pages.sum())
    indptr = torch.zeros(bs + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(pages, 0)
    indptr = indptr.to(dev)
    indices = torch.arange(npages, dtype=torch.int32, device=dev)
    last = torch.full((bs,), PAGE, dtype=torch.int32, device=dev)
    q = torch.randn(bs, 1, HD, dtype=torch.float16, device=dev)
    kv = torch.randn(npages, 2, PAGE, NKV, HD, dtype=torch.float16, device=dev)
    return indptr, indices, last, q, kv


def plan_tiles(w, bs):
    v = getattr(w, "_plan_info", None)
    try:
        v = v.tolist() if hasattr(v, "tolist") else list(v)
        if len(v) == 10:
            return (int(v[0]) if int(v[0]) >= bs else bs, bool(v[9]))
    except Exception:
        pass
    return bs, False


def step_us(w, q, kv, iters=20):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(iters):
        w.run(q, kv)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters


def main():
    torch.cuda.init()
    dev = "cuda"
    plane = SchedPlane(max_tasks=MAX_TASKS, device=dev)
    plane.use_device_timer()
    plane.arm_process_env(os.environ["SCHED_PLUGIN"])
    import flashinfer

    def wrapper():
        return flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev),
            "NHD")

    gen = torch.Generator().manual_seed(42)
    R = 0

    # ---- A + B: dispersion sweep -------------------------------------------
    disp_rows = []
    for bs in BS_SWEEP:
        pages = mix_pages(bs, gen)
        indptr, indices, last, q, kv = build(bs, pages, dev)
        w = wrapper()
        w.plan(indptr, indices, last, 1, NKV, HD, PAGE,
               data_type=torch.float16, q_data_type=torch.float16)
        ntiles, split = plan_tiles(w, bs)
        if ntiles > plane.N:
            continue
        plane.set_num_tasks(0); plane.reset_order(); plane.push()
        w.run(q, kv); torch.cuda.synchronize()  # warm/JIT
        if R == 0:
            base = os.environ.get("FLASHINFER_WORKSPACE_BASE", "")
            sos = sorted(glob.glob(os.path.join(
                base, ".cache", "flashinfer", "**", "*batch_decode*.so"),
                recursive=True))
            if sos:
                # R must use the kernel's real block size; the decode kernel
                # launches ~128-thread blocks (occupancy on 1 thread is
                # meaningless -> a hugely inflated, wrong R).
                R = plane.r_for_cached_so(sos[0], "BatchDecode", 128)
        # median over repeats of the per-tile percentiles
        p50s, p99s, mxs, means = [], [], [], []
        raw = None
        for _ in range(3):
            plane.clear_timer(); w.run(q, kv); torch.cuda.synchronize()
            cyc = plane.read_timer()[:ntiles].float()
            cyc = cyc[cyc > 0]
            if cyc.numel() < 2:
                continue
            p50s.append(float(cyc.quantile(0.5)))
            p99s.append(float(cyc.quantile(0.99)))
            mxs.append(float(cyc.max())); means.append(float(cyc.mean()))
            if bs == BS_DIST and raw is None:
                raw = cyc.cpu()
        if not p50s:
            continue
        md = lambda v: sorted(v)[len(v) // 2]
        p50, p99, mx, mean = md(p50s), md(p99s), md(mxs), md(means)
        row = dict(bs=bs, tiles=ntiles, split_active=int(split), R=R,
                   p50=p50, p99=p99, max=mx, mean=mean,
                   ratio_p99_p50=p99 / max(p50, 1.0),
                   max_over_mean=mx / max(mean, 1.0),
                   queued=int(R > 0 and ntiles > R),
                   step_us=step_us(w, q, kv))
        disp_rows.append(row)
        print(f"  bs={bs:5d} tiles={ntiles:5d} split={int(split)} "
              f"p99/p50={row['ratio_p99_p50']:.2f} "
              f"max/mean={row['max_over_mean']:.2f} "
              f"{'QUEUED' if row['queued'] else 'one-wave'}")
        if bs == BS_DIST and raw is not None:
            with open(os.path.join(DATA, "mot_tilecycles.csv"), "w",
                      newline="") as f:
                wr = csv.writer(f); wr.writerow(["tile", "cycles"])
                for i, c in enumerate(raw.tolist()):
                    wr.writerow([i, int(c)])
            print(f"  [saved] raw per-tile cycles at bs={bs} "
                  f"({raw.numel()} tiles)")

    with open(os.path.join(DATA, "mot_dispersion.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(disp_rows[0].keys()))
        wr.writeheader(); wr.writerows(disp_rows)
    print(f"  [saved] mot_dispersion.csv ({len(disp_rows)} rows, R={R})")

    # ---- C: closed-loop pi reclaim at BS_DIST ------------------------------
    bs = BS_DIST
    pages = mix_pages(bs, torch.Generator().manual_seed(7))
    indptr, indices, last, q, kv = build(bs, pages, dev)
    w = wrapper()
    w.plan(indptr, indices, last, 1, NKV, HD, PAGE,
           data_type=torch.float16, q_data_type=torch.float16)
    ntiles, _ = plan_tiles(w, bs)
    plane.set_num_tasks(0)
    # probe under identity to fill the timer, then LPT from measured cycles.
    plane.set_timer_enabled(True); plane.reset_order(); plane.push()
    plane.clear_timer(); w.run(q, kv); torch.cuda.synchronize()
    cyc = plane.read_timer()[:ntiles].clone()
    nlong = max(1, ntiles // 10)
    true_long = set(torch.argsort(pages, descending=True)[:nlong].tolist())
    meas_long = set(torch.argsort(cyc, descending=True)[:nlong].tolist())
    recall = len(true_long & meas_long) / nlong
    plane.set_timer_enabled(False); plane.push()  # measure without probe cost
    orders = {
        "identity": torch.arange(ntiles, dtype=torch.int32),
        "reversed": torch.arange(ntiles - 1, -1, -1, dtype=torch.int32),
        "lpt-oracle": torch.argsort(pages, descending=True).to(torch.int32),
        "lpt-timer": torch.argsort(cyc, descending=True).to(torch.int32),
    }
    golden, pol_rows = None, []
    for name, order in orders.items():
        full = torch.arange(plane.N, dtype=torch.int32); full[:ntiles] = order
        plane.install_order(full, ntiles); plane.push()
        out = w.run(q, kv).clone()
        exact = golden is None or torch.equal(out, golden)
        if golden is None:
            golden = out
        t = step_us(w, q, kv)
        pol_rows.append(dict(policy=name, step_us=t, bit_exact=int(exact),
                             recall=(recall if name == "lpt-timer" else "")))
        print(f"  {name:11s} {t:8.1f} us  bit_exact={exact}")
    base = next(r["step_us"] for r in pol_rows if r["policy"] == "identity")
    for r in pol_rows:
        r["norm_vs_identity"] = r["step_us"] / base
    with open(os.path.join(DATA, "mot_policy.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(pol_rows[0].keys()))
        wr.writeheader(); wr.writerows(pol_rows)
    print(f"  [saved] mot_policy.csv  (timer->pi recall {100*recall:.0f}%)")
    print("== MOTIVATION DATA COLLECTED ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
