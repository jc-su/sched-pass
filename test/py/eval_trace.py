"""eval_trace.py -- the serving-trace evaluation: does closed-loop pi pay on a
heteroskedastic decode batch, on the REAL woven kernel, on this GPU?

Continuous-batching decode is a sequence of steps; each step launches one grid
over the active batch's tiles and the step ends when the LAST tile finishes
(the tropical/makespan law, THEORY.md #3). pi (task_order) controls the order
the hardware issues tiles within the launch, so it pays exactly when tiles
queue (grid >> one SM wave) and lengths are heteroskedastic: LPT-style orders
stop a straggler from being issued last and stretching the tail.

This harness measures that end to end, closed loop:
  * a length-mixed batch (long-tail: FRAC_LONG of requests have LONG_X more KV
    pages) of NSEQ requests, one warp-CTA tile each (paged_softmax.cu -- online
    softmax over paged KV, the decode shape);
  * policies:  identity  |  reversed (adversarial: longs issued last)
               lpt-oracle (longs first, from true lengths)
               lpt-timer  (longs first, from the WOVEN TIMER's measured cycles
                           of the previous step -- the closed loop; no oracle)
  * STEPS decode steps per policy; per-step wall time via CUDA events;
  * every policy's output is checked BIT-EXACT vs identity (E1: pi reorders
    WHEN, never WHAT).

Honest regime note: on an idle GPU with grid <= one wave there is no queue and
every pi ties (measured and reported, not hidden) -- scale NSEQ up (default
8192) to enter the queueing regime. On a contended GPU the optimum can invert
(the coupling gamma finding); this harness reports, it does not assume.

Run:  SCHED_PLUGIN=.../libSchedPass.so python eval_trace.py
Env:  NSEQ (8192) STEPS (30) FRAC_LONG (0.1) LONG_X (8) SCHED_ARCH (sm_120a)
"""
import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))  # library
from sched_rt import SchedPlane
from test_dynamic_loop import compile_kernel, run  # same kernel, same launcher

NSEQ = int(os.environ.get("NSEQ", 8192))
STEPS = int(os.environ.get("STEPS", 30))
FRAC_LONG = float(os.environ.get("FRAC_LONG", 0.1))
LONG_X = int(os.environ.get("LONG_X", 8))
SD, PT = 32, 32
NBLK_SHORT = 3
NBLK_LONG = NBLK_SHORT * LONG_X
PAD = 4096

# test_dynamic_loop's module-level NSEQ is baked into run(); override its view.
import test_dynamic_loop as tdl
tdl.NSEQ = NSEQ


def step_time_ms(lib, kv, bt, nbl, q, out, iters=1):
    """Wall time of one decode step (one launch over all NSEQ tiles)."""
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        lib.launch_paged_softmax(kv.data_ptr(), bt.data_ptr(), nbl.data_ptr(),
                                 q.data_ptr(), out.data_ptr(), NSEQ, NBLK_LONG,
                                 PT, __import__("ctypes").c_void_p(
                                     torch.cuda.current_stream().cuda_stream))
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    torch.manual_seed(0)
    dev = "cuda"
    is_long = torch.rand(NSEQ) < FRAC_LONG
    nbl = torch.where(is_long, NBLK_LONG, NBLK_SHORT).to(torch.int32).to(dev)
    n_long = int(is_long.sum())
    print(f"== eval: NSEQ={NSEQ} tiles ({n_long} long x{LONG_X}), "
          f"{STEPS} steps/policy ==")

    npages = NSEQ * NBLK_SHORT + n_long * (NBLK_LONG - NBLK_SHORT)
    kv = torch.randn(npages * PT * 2 * SD + PAD, device=dev)
    q = torch.randn(NSEQ, SD, device=dev)
    out = torch.zeros(NSEQ, SD, device=dev)
    # block table: each request's pages, packed
    bt = torch.zeros(NSEQ, NBLK_LONG, dtype=torch.int32)
    page = 0
    for r in range(NSEQ):
        k = NBLK_LONG if is_long[r] else NBLK_SHORT
        bt[r, :k] = torch.arange(page, page + k, dtype=torch.int32)
        page += k
    bt = bt.to(dev)

    # stock baseline: same kernel, NO plugin -- the woven overhead accounting.
    stock = compile_kernel(os.path.join(tempfile.gettempdir(), "eval_stock.so"),
                           dict(os.environ))
    plane = SchedPlane(max_tasks=NSEQ, device=dev)
    env = plane.bake_env(os.environ["SCHED_PLUGIN"])
    env["SCHED_MAX_TASKS"] = str(NSEQ)
    lib = compile_kernel(os.path.join(tempfile.gettempdir(), "eval_woven.so"), env)
    plane.set_num_tasks(NSEQ)
    plane.push()
    t_stock = torch.tensor([step_time_ms(stock, kv, bt, nbl, q, out)
                            for _ in range(STEPS)])
    print(f"  {'stock (unwoven)':22s} step {t_stock.mean():7.3f} ms "
          f"(min {t_stock.min():7.3f})  <- overhead baseline")

    lengths = nbl.cpu()
    orders = {
        "identity": torch.arange(NSEQ, dtype=torch.int32),
        "reversed-adversarial": None,  # longs LAST: sort ascending by length
        "lpt-oracle": torch.argsort(lengths, descending=True).to(torch.int32),
        "lpt-timer": None,  # closed loop, built from the previous step's timer
    }
    orders["reversed-adversarial"] = torch.argsort(
        lengths, descending=False).to(torch.int32)

    golden = None
    results = {}
    for name, order in orders.items():
        if name == "lpt-timer":
            # closed loop: one probe step under identity fills the timer, then
            # order = argsort(measured cycles) descending. No oracle knowledge.
            plane.reset_order()
            plane.clear_timer()
            step_time_ms(lib, kv, bt, nbl, q, out)
            cyc = plane.read_timer()[:NSEQ]
            order = torch.argsort(cyc, descending=True).to(torch.int32)
            agree = (nbl.cpu()[order[:n_long].long()] == NBLK_LONG).float().mean()
            print(f"  [timer->pi] top-{n_long} of measured order are "
                  f"{100*agree:.0f}% true longs")
        plane.set_order(order)
        times = [step_time_ms(lib, kv, bt, nbl, q, out) for _ in range(STEPS)]
        o = out.clone()
        if golden is None:
            golden = o
            exact = True
        else:
            exact = torch.equal(o, golden)
        t = torch.tensor(times)
        results[name] = (t.mean().item(), t.min().item(), exact)
        print(f"  {name:22s} step {t.mean():7.3f} ms (min {t.min():7.3f})  "
              f"bit-exact={exact}")

    base = results["identity"][0]
    print("-- summary (vs identity) --")
    fails = 0
    for name, (mean, _, exact) in results.items():
        print(f"  {name:22s} {100*(mean-base)/base:+6.1f}%  exact={exact}")
        if not exact:
            fails += 1
    print("== ALL EXACT ==" if fails == 0 else f"== {fails} POLICIES DIVERGED ==")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
