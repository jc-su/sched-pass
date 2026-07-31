"""eval_clc_mispredict.py -- THE CLC contribution experiment: does the dynamic
work-queue RESCUE a MISPREDICTED order, where static pi cannot?

P2 showed static grouped-LPT beats CLC when cost is PREDICTABLE (KV = true cost).
CLC's documented law: it pays only under SEVERE order breakdown (recall <75%).
This measures exactly that -- WITHOUT needing a MoE/spec model. The real decode
kernel's true cost IS ~KV; we degrade the INSTALLED order to recall R (heavy
tiles scattered), simulating a mispredicted cost model (MoE routing / spec accept
/ mixed prefill -- regimes where KV does not predict compute):
  * STATIC kernel + recall-R order: the order IS the schedule -> a wrong order is
    a wrong makespan. Degrades as R falls.
  * CLC kernel + recall-R fill: the hardware claims dynamically as SMs free, so a
    wrong FILL order is rebalanced at execution -> makespan should stay near the
    oracle even at low recall. That recovery IS the contribution.
Drift-invariant: each phase measures every recall vs its OWN oracle (recall 1.0),
same run. 1D MQA grid (NKV=1) so the CLC claim engages.

Driver: python test/py/eval_clc_mispredict.py --drive  (runs static then CLC)
"""
import os
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
from sched_rt import SchedPlane

PAGE, HD, NQO, NKV = 16, 128, 8, 1         # MQA 1D grid -> CLC engages
ITERS = 80
MU, SIGMA = 2.6, 1.1
BS = 8192                                   # wave regime (ordering matters)
RECALLS = [1.0, 0.75, 0.5, 0.25, 0.0]


def step_us(w, q, kv, iters=ITERS):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(iters):
        w.run(q, kv)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters


def degrade(true_order, recall, gen):
    """Keep `recall` of positions at their oracle slot; shuffle the rest.
    recall=1 -> oracle (LPT); recall=0 -> fully scrambled."""
    n = true_order.numel()
    order = true_order.clone()
    nshuf = int(round((1.0 - recall) * n))
    if nshuf > 1:
        pos = torch.randperm(n, generator=gen)[:nshuf]
        order[pos] = order[pos][torch.randperm(nshuf, generator=gen)]
    return order.to(torch.int32)


def main():
    clc = bool(os.environ.get("SCHED_WORKQUEUE"))
    mode = "CLC-queue" if clc else "static-pi"
    dev = "cuda"
    plane = SchedPlane(max_tasks=16384, device=dev)
    plane.use_device_timer(); plane.set_timer_enabled(False)
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

    true_order = torch.argsort(pages.float(), descending=True).to(torch.int32)
    rgen = torch.Generator().manual_seed(7)
    for _ in range(4):
        w.run(q, kv)
    torch.cuda.synchronize()

    def measure(order):
        if clc:
            plane.set_num_tasks(BS)     # arm the dynamic claim
        plane.install_order(order, BS); plane.push()
        t = min(step_us(w, q, kv) for _ in range(5))
        plane.set_num_tasks(0); plane.reset_order(); plane.push()
        return t

    t_oracle = measure(true_order)      # recall 1.0 = LPT oracle
    print(f"== {mode}: makespan vs RECALL (BS={BS}, oracle={t_oracle:.0f}us) ==")
    print(f"{'recall':>8} {'step_us':>9} {'vs-oracle':>10}")
    for r in RECALLS:
        order = degrade(true_order, r, rgen)
        t = measure(order)
        print(f"{r:>8.2f} {t:>8.1f}u {100*(t-t_oracle)/t_oracle:>+9.2f}%")
    print(f"== interpretation: static DEGRADES as recall falls (wrong order = "
          f"wrong makespan); CLC should stay ~flat (dynamic claim rebalances) ==")
    return 0


def drive():
    env = dict(os.environ)
    base = [sys.executable, os.path.abspath(__file__)]
    print("### STATIC pi ###")
    subprocess.run(base, env={k: v for k, v in env.items()
                              if k not in ("SCHED_WORKQUEUE", "SCHED_CLC")})
    print("\n### CLC work-queue ###")
    subprocess.run(base, env=dict(env, SCHED_WORKQUEUE="1", SCHED_CLC="1"))
    print("\n== compare vs-oracle degradation: if CLC's is FLATTER than static's "
          "at recall<0.75, the dynamic queue RESCUES misprediction (its regime) ==")


if __name__ == "__main__":
    if "--drive" in sys.argv:
        drive()
    else:
        raise SystemExit(main())
