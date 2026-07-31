"""eval_closed_loop.py -- THE contribution, made visible: the woven TIMER
observes true per-tile cost, the estimator relearns the order, and ONE
observation RECOVERS a mispredicted schedule. This is the "pays for its own
observation" thesis on the real decode kernel -- and, per eval_clc_mispredict.py,
it is what actually rescues a bad order (NOT the CLC claim).

Sequence (real FlashInfer decode, GQA-3B serving shape, wave regime):
  oracle      = grouped-LPT over TRUE cost (KV)        -> best makespan
  mispredicted= grouped-LPT over a SCRAMBLED cost      -> +X% (wrong order)
  observed    = run mispredicted with the woven TIMER armed -> read per-tile
                cycles (true cost, order-invariant) -> grouped-LPT over the
                MEASURED cost -> should recover to ~oracle
The recovery is driven ENTIRELY by the woven observation -- no oracle knowledge.

Run (armed):
  SCHED_PLUGIN=.../libSchedPass.so FLASHINFER_NVCC="python3 .../nvcc_clang_shim.py" \
  SCHED_CLANG=clang++-22 SCHED_CUDA_PATH=/usr/local/cuda-12.9 SCHED_ARCH=sm_120a \
  SCHED_WEAVE_ONLY=batch_decode python test/py/eval_closed_loop.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
from sched_rt import SchedPlane

PAGE, HD, NQO, NKV = 16, 128, 16, 2        # GQA-3B serving shape
ITERS = 80
MU, SIGMA = 2.6, 1.1
BS = 8192


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

    gen = torch.Generator().manual_seed(1)
    pages = torch.exp(torch.normal(MU, SIGMA, (BS,), generator=gen)) \
        .clamp(1, 512).to(torch.int32)
    npages = int(pages.sum())
    indptr = torch.zeros(BS + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(pages, 0)
    indptr = indptr.cuda()
    indices = torch.arange(npages, dtype=torch.int32, device=dev)
    last = torch.full((BS,), PAGE, dtype=torch.int32, device=dev)
    q = torch.randn(BS, NQO, HD, dtype=torch.float16, device=dev)
    kv = torch.randn(npages, 2, PAGE, NKV, HD, dtype=torch.float16, device=dev)
    import flashinfer
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev), "NHD")
    w.plan(indptr, indices, last, NQO, NKV, HD, PAGE,
           data_type=torch.float16, q_data_type=torch.float16)

    for _ in range(4):
        w.run(q, kv)
    torch.cuda.synchronize()

    def makespan(order):
        plane.set_timer_enabled(False)
        plane.install_order(order, BS); plane.push()
        return min(step_us(w, q, kv) for _ in range(5))

    print(f"== closed loop: observe -> recover a mispredicted order "
          f"(GQA {NQO}/{NKV}, BS={BS}) ==")

    # 1) ORACLE: order by true cost (KV)
    oracle = SchedPlane.region_order(pages.float(), 0)
    t_oracle = makespan(oracle)

    # 2) MISPREDICTED: order by a scrambled cost (a wrong model)
    sgen = torch.Generator().manual_seed(7)
    scrambled = pages.float().clone()
    perm = torch.randperm(BS, generator=sgen)
    scrambled = scrambled[perm]                 # cost detached from tile
    bad = SchedPlane.region_order(scrambled, 0)
    t_bad = makespan(bad)

    # 3) OBSERVE: run the mispredicted order with the woven TIMER armed, read
    #    per-tile cycles (true cost, order-invariant), reorder by THAT.
    plane.set_timer_enabled(True); plane.clear_timer()
    plane.install_order(bad, BS); plane.push()
    w.run(q, kv); torch.cuda.synchronize()
    cyc = plane.read_timer()[:BS].float()
    measured_ok = int((cyc > 0).sum())
    learned = SchedPlane.region_order(cyc, 0)   # order by MEASURED cost
    t_learned = makespan(learned)

    print(f"  oracle (true KV order)      {t_oracle:8.1f} us   baseline")
    print(f"  mispredicted (scrambled)    {t_bad:8.1f} us   "
          f"{100*(t_bad-t_oracle)/t_oracle:+.1f}%  <- wrong order penalty")
    print(f"  observed (woven timer)      {t_learned:8.1f} us   "
          f"{100*(t_learned-t_oracle)/t_oracle:+.1f}%  <- recovered by "
          f"observation ({measured_ok}/{BS} tiles measured)")
    recov = (t_bad - t_learned) / max(t_bad - t_oracle, 1e-6)
    print(f"  => observation recovered {100*recov:.0f}% of the misprediction "
          f"penalty, driven only by the woven timer (no oracle knowledge)")
    plane.set_timer_enabled(False); plane.reset_order(); plane.push()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
