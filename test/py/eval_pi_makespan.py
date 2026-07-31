"""eval_pi_makespan.py -- THE clean, isolated test of pi's makespan benefit.

The serving A/B is 80% server-state noise (radix-tree growth, admission, overlap
scheduler); this bypasses ALL of it. One FIXED dispersed decode batch, woven
kernel, measure the step wall-clock under:
  * IDENTITY order  (order[ctaid] = ctaid)           -- baseline
  * LPT order       (order[ctaid] = ctaid-th longest tile) -- longest KV first
  * SPT order       (shortest first)                 -- the anti-LPT control
Same batch, same kernel, ONLY the task_order permutation differs (E1: output is
bit-exact regardless -- we are timing, not checking correctness here). Sweeps
batch size: pi can only help when tiles WAVE-SERIALIZE (more resident CTAs than
SM slots); below that they run concurrently and order is irrelevant. If LPT ==
identity across the sweep, pi has no makespan lever on THIS hardware -- an honest,
decisive null. If LPT < identity at large BS, that is exactly the queued regime
where it pays, measured cleanly.

Run (armed):
  SCHED_PLUGIN=.../libSchedPass.so FLASHINFER_NVCC="python3 .../nvcc_clang_shim.py" \
  SCHED_CLANG=clang++-22 SCHED_CUDA_PATH=/usr/local/cuda-12.9 SCHED_ARCH=sm_120a \
  python test/py/eval_pi_makespan.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
from sched_rt import SchedPlane

PAGE, HD, NQO, NKV = 16, 128, 16, 2       # Qwen-3B GQA decode shape
ITERS = 80
MU, SIGMA = 2.6, 1.1                        # lognormal pages/request -> KV disp


def step_us(w, q, kv, iters=ITERS):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(iters):
        w.run(q, kv)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters


def build(bs, dev, gen):
    pages = torch.exp(torch.normal(MU, SIGMA, (bs,), generator=gen)) \
        .clamp(1, 512).to(torch.int32)                  # dispersed KV depth
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


def main():
    dev = "cuda"
    plane = SchedPlane(max_tasks=16384, device=dev)
    plane.use_device_timer()
    plane.arm_process_env(os.environ["SCHED_PLUGIN"])
    plane.set_timer_enabled(False)              # time makespan, not the timer

    print(f"== pi makespan: identity vs LPT vs SPT, isolated dispersed decode "
          f"(nqo={NQO}/nkv={NKV}) ==")
    print(f"{'BS':>6} {'costdisp':>9} {'identity':>10} {'LPT':>10} {'SPT':>10} "
          f"{'LPT-vs-id':>10} {'SPT-vs-id':>10}")
    gen = torch.Generator().manual_seed(1)
    for bs in (256, 512, 1024, 2048, 4096, 8192):
        w, q, kv, pages = build(bs, dev, gen)
        # cost proxy = KV pages (decode attention reads all KV). LPT: longest
        # first; SPT: shortest first. order[ctaid] = tile index in that order.
        cost = pages.to(torch.float32)
        p99, p50 = float(cost.quantile(0.99)), float(max(cost.median(), 1))
        lpt = torch.argsort(cost, descending=True).to(torch.int32)
        spt = torch.argsort(cost, descending=False).to(torch.int32)
        idn = torch.arange(bs, dtype=torch.int32)

        for _ in range(4):                       # warm/JIT
            w.run(q, kv)
        torch.cuda.synchronize()

        def measure(order):
            plane.set_num_tasks(0)               # disarm -> stock path
            plane.reset_order(); plane.push()
            t_id = min(step_us(w, q, kv) for _ in range(4))
            plane.install_order(order, bs)       # arm pi with this permutation
            plane.push()
            t = min(step_us(w, q, kv) for _ in range(4))
            plane.set_num_tasks(0); plane.reset_order(); plane.push()
            return t_id, t

        # identity baseline is measured inside each (num_tasks=0); take once
        plane.set_num_tasks(0); plane.reset_order(); plane.push()
        t_idn = min(step_us(w, q, kv) for _ in range(5))
        plane.install_order(lpt, bs); plane.push()
        t_lpt = min(step_us(w, q, kv) for _ in range(5))
        plane.install_order(spt, bs); plane.push()
        t_spt = min(step_us(w, q, kv) for _ in range(5))
        plane.set_num_tasks(0); plane.reset_order(); plane.push()

        d_lpt = 100 * (t_lpt - t_idn) / t_idn
        d_spt = 100 * (t_spt - t_idn) / t_idn
        print(f"{bs:>6} {p99/p50:>8.1f}x {t_idn:>9.1f}u {t_lpt:>9.1f}u "
              f"{t_spt:>9.1f}u {d_lpt:>+9.2f}% {d_spt:>+9.2f}%")
        del w, q, kv
        torch.cuda.empty_cache()
    print("== interpretation: LPT-vs-id < 0 => pi reduces makespan (wave regime);"
          " ~0 => tiles fit concurrently, order is irrelevant on this HW ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
