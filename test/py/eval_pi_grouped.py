"""eval_pi_grouped.py -- does LOCALITY-PRESERVING grouped-LPT remove the
mid-range penalty while keeping the large-BS makespan win?

eval_pi_makespan.py found: full per-tile LPT wins big at large BS (-12% @8192)
but LOSES a few % at mid BS (+3.6% @2048) because sorting by length SCRAMBLES
the identity order's L2 locality (adjacent CTAs -> adjacent requests -> adjacent
KV). Grouped-LPT sorts at BLOCK granularity B: B adjacent tiles stay together
(locality preserved) but blocks are LPT-ordered (coarse load balance).
  B=1     == full per-tile LPT   (max balance, min locality)
  B=bs    == identity            (max locality, no balance)
The sweep locates a B that is <=0% at the mid BS AND still strongly negative at
the large BS -- the locality/makespan sweet spot. Same E1 permutation (bit-exact),
zero kernel change: pure control-plane order.
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
from sched_rt import SchedPlane

PAGE, HD, NQO, NKV = 16, 128, 16, 2
ITERS = 80
MU, SIGMA = 2.6, 1.1


def step_us(w, q, kv, iters=ITERS):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(iters):
        w.run(q, kv)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters


def grouped_lpt(cost, bs, B):
    """Block-LPT permutation: chunk into ceil(bs/B) blocks of B ADJACENT tiles,
    order blocks by descending block cost, keep original (ascending) order within
    a block. Returns an int32 permutation of [0,bs)."""
    if B <= 1:
        return torch.argsort(cost, descending=True).to(torch.int32)
    nb = (bs + B - 1) // B
    pad = nb * B - bs
    c = torch.cat([cost, torch.zeros(pad, dtype=cost.dtype)])
    bcost = c.reshape(nb, B).sum(1)
    border = torch.argsort(bcost, descending=True).to(torch.int64)
    base = border[:, None] * B + torch.arange(B, dtype=torch.int64)[None, :]
    order = base.flatten()
    order = order[order < bs]                       # drop padding slots
    return order.to(torch.int32)


def build(bs, dev, gen):
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


def main():
    dev = "cuda"
    plane = SchedPlane(max_tasks=16384, device=dev)
    plane.use_device_timer()
    plane.arm_process_env(os.environ["SCHED_PLUGIN"])
    plane.set_timer_enabled(False)

    BLOCKS = [1, 4, 8, 16, 32, 64, 128, 512]
    print(f"== grouped-LPT locality sweep (nqo={NQO}/nkv={NKV}); % vs identity, "
          f"more negative = faster ==")
    header = "   BS " + "".join(f"  B={b:<4}" for b in BLOCKS)
    print(header)
    gen = torch.Generator().manual_seed(1)
    for bs in (2048, 8192):                          # mid-penalty + large-win
        w, q, kv, pages = build(bs, dev, gen)
        cost = pages.to(torch.float32)
        for _ in range(4):
            w.run(q, kv)
        torch.cuda.synchronize()
        plane.set_num_tasks(0); plane.reset_order(); plane.push()
        t_id = min(step_us(w, q, kv) for _ in range(5))
        row = f"{bs:>6}"
        for B in BLOCKS:
            plane.install_order(grouped_lpt(cost, bs, B), bs); plane.push()
            t = min(step_us(w, q, kv) for _ in range(5))
            row += f" {100*(t-t_id)/t_id:>+6.2f}"
        plane.set_num_tasks(0); plane.reset_order(); plane.push()
        print(row + f"   (identity={t_id:.0f}us)")
        del w, q, kv
        torch.cuda.empty_cache()
    print("== B=1 is full per-tile LPT; look for a B where the MID BS (2048) is "
          "<=0 while the LARGE BS (8192) stays strongly negative ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
