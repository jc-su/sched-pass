"""test_timer_gate.py -- the ctrl.flags timer gate, on device, baked ABI.

The baked ABI cannot null its slots (baked addresses are compile-time
constants), so per-step observation gating is DATA: ctrl.flags bit0 = timer
OFF, written by SchedPlane.set_timer_enabled() and read by the woven timer
under the ctrl-armed check. This test proves the whole loop on the real JIT
kernel:

  1. default (flags=0): timer rows populate (the historical behavior);
  2. set_timer_enabled(False) + push: the SAME kernel writes NO timer rows
     (the PCIe atomic is suppressed -- the sampled-observation mode);
  3. set_timer_enabled(True) + push: rows populate again;
  4. outputs are BIT-EXACT across all three (observation must never change
     results -- the O effect type).

Run:  SCHED_PLUGIN=.../libSchedPass.so python test_timer_gate.py
"""
import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))  # library
from sched_rt import SchedPlane
from test_dynamic_loop import compile_kernel, run, NSEQ, SD, PT, NBLK_LONG, \
    NBLK_SHORT, NPAGES, PAD, is_long


def main():
    torch.cuda.init()
    dev = "cuda"
    g = torch.Generator().manual_seed(3)
    kv = (torch.rand(NPAGES * PT * 2 * SD + PAD, generator=g) - 0.5).to(dev)
    q = (torch.rand(NSEQ * SD, generator=g) - 0.5).to(dev)
    bt = torch.empty(NSEQ * NBLK_LONG, dtype=torch.int32)
    nbl = torch.empty(NSEQ, dtype=torch.int32)
    for t in range(NSEQ):
        nbl[t] = NBLK_LONG if is_long(t) else NBLK_SHORT
        for b in range(NBLK_LONG):
            bt[t * NBLK_LONG + b] = (t * NBLK_LONG + b * 7 + 3) % NPAGES
    bt, nbl = bt.to(dev), nbl.to(dev)
    out = torch.zeros(NSEQ * SD, device=dev)

    fails = [0]
    def ok(cond, name):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            fails[0] += 1

    print("== timer gate (ctrl.flags bit0) on the baked-ABI JIT kernel ==")
    plane = SchedPlane(max_tasks=NSEQ, device=dev)
    env = plane.bake_env(os.environ["SCHED_PLUGIN"])
    env["PATH"] = os.environ["PATH"]
    lib = compile_kernel(os.path.join(tempfile.gettempdir(),
                                      "paged_timer_gate.so"), env)
    plane.set_num_tasks(NSEQ)

    # 1. default: timer on.
    plane.set_timer_enabled(True)
    plane.push()
    plane.clear_timer()
    golden = run(lib, kv, bt, nbl, q, out, NBLK_LONG)
    rows_on = int((plane.read_timer() > 0).sum())
    ok(rows_on == NSEQ, f"timer ON: one row per request ({rows_on}/{NSEQ})")

    # 2. gated off: same kernel, zero rows.
    plane.set_timer_enabled(False)
    plane.push()
    plane.clear_timer()
    o_off = run(lib, kv, bt, nbl, q, out, NBLK_LONG)
    rows_off = int((plane.read_timer() > 0).sum())
    ok(rows_off == 0, f"timer OFF (flags bit0): zero rows ({rows_off})")
    ok(torch.equal(o_off, golden), "timer OFF: output bit-exact")

    # 3. re-enabled: rows again.
    plane.set_timer_enabled(True)
    plane.push()
    plane.clear_timer()
    o_on = run(lib, kv, bt, nbl, q, out, NBLK_LONG)
    rows_again = int((plane.read_timer() > 0).sum())
    ok(rows_again == NSEQ, f"timer re-enabled: rows return ({rows_again}/{NSEQ})")
    ok(torch.equal(o_on, golden), "timer re-enabled: output bit-exact")

    print("== ALL PASS ==" if fails[0] == 0 else f"== {fails[0]} FAILED ==")
    return fails[0]


if __name__ == "__main__":
    raise SystemExit(main())
