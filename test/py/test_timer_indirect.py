"""test_timer_indirect.py -- the DEVICE observation channel (E1's verdict),
end to end on the baked-ABI JIT kernel.

SCHED_TIMER_INDIRECT weaves the timer slot as a POINTER word the host
retargets per process: 0 = off (fail-safe, the zeroed arena default), else
the row-table address (device buffer: atomics ~free; host-mapped: zero-touch
observers). This test proves:

  1. compiled indirect + channel word 0: NO rows written, outputs bit-exact
     (fail-safe by construction);
  2. use_device_timer(): rows land in DEVICE memory, one per request, outputs
     bit-exact (O never changes results);
  3. the ctrl.flags cadence gate still suppresses collection on top;
  4. cycles rank the long requests first (the estimator input is sane).

Run:  SCHED_PLUGIN=.../libSchedPass.so python test_timer_indirect.py
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
    g = torch.Generator().manual_seed(9)
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

    print("== device timer channel (SCHED_TIMER_INDIRECT) on the JIT kernel ==")
    plane = SchedPlane(max_tasks=NSEQ, device=dev)
    # Compile with the indirect layout but the channel STILL UNARMED (word 0):
    # bake_env only tags -ti after use_device_timer, so set the env by hand
    # for the compile, then arm the channel afterward.
    env = plane.bake_env(os.environ["SCHED_PLUGIN"])
    env["SCHED_TIMER_INDIRECT"] = "1"
    env["FLASHINFER_WORKSPACE_BASE"] += "-ti"
    env["PATH"] = os.environ["PATH"]
    lib = compile_kernel(os.path.join(tempfile.gettempdir(),
                                      "paged_timer_indirect.so"), env)
    plane.set_num_tasks(NSEQ)
    plane.set_timer_enabled(True)
    plane.push()

    # 1. channel word 0 -> no rows, bit-exact.
    golden = run(lib, kv, bt, nbl, q, out, NBLK_LONG)
    host_rows = int((torch.frombuffer(
        bytearray(plane.arena.read(0x80000, 8 * NSEQ)),
        dtype=torch.int64) != 0).sum())
    ok(host_rows == 0, f"channel word 0: no rows anywhere ({host_rows})")

    # 2. device channel armed: rows in device memory, output unchanged.
    plane.use_device_timer()
    plane.clear_timer()
    o2 = run(lib, kv, bt, nbl, q, out, NBLK_LONG)
    cyc = plane.read_timer()
    ok(torch.equal(o2, golden), "device channel: output bit-exact")
    rows = int((cyc > 0).sum())
    ok(rows == NSEQ, f"device channel: one row per request ({rows}/{NSEQ})")

    # 3. flags cadence gate still gates collection.
    plane.set_timer_enabled(False)
    plane.push()
    plane.clear_timer()
    o3 = run(lib, kv, bt, nbl, q, out, NBLK_LONG)
    ok(int((plane.read_timer() > 0).sum()) == 0,
       "flags gate suppresses device collection")
    ok(torch.equal(o3, golden), "gated step: output bit-exact")
    plane.set_timer_enabled(True)
    plane.push()

    # 4. estimator sanity: longs rank first.
    long_c = cyc[[t for t in range(NSEQ) if is_long(t)]].float().mean()
    short_c = cyc[[t for t in range(NSEQ) if not is_long(t)]].float().mean()
    ok(float(long_c) > 2 * float(short_c),
       f"device cycles attribute cost (long {long_c:.0f} >> short "
       f"{short_c:.0f})")

    print("== ALL PASS ==" if fails[0] == 0 else f"== {fails[0]} FAILED ==")
    return fails[0]


if __name__ == "__main__":
    raise SystemExit(main())
