"""eval_timer_channel.py -- E1: the observation-channel ablation.

The woven timer currently atomicAdds into HOST-MAPPED pinned memory: zero-touch
readout (plain host read), but every CTA exit pays a system-scope PCIe atomic
-- measured +38% on a probe step at serving scale, +181% on tiny kernels;
hence the sampled-observation design (ctrl.flags gate, SCHED_TIMER_EVERY).

The alternative: point the timer at DEVICE memory (atom.global.add.u64 into
L2, ~free) and pay one explicit D2H copy + clear per PROBE step. If that is
~baseline, sampling becomes unnecessary complexity for most deployments and
the estimator gets every-step freshness.

Method (single process, so fixed-VA is irrelevant): compile the SAME kernel
twice with different SCHED_BAKE_TIMER -- (A) the arena's host-mapped window
(current design), (B) a cudaMalloc'd device buffer -- plus (C) timer gated
off via ctrl.flags as the baseline. Decode-shaped paged softmax, NSEQ tiles,
long-tail mix. Reports step time per variant, the D2H readback+clear cost,
and cross-checks that A and B report consistent cycle magnitudes.

Run:  NSEQ=8192 SCHED_PLUGIN=... python eval_timer_channel.py
"""
import ctypes
import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))  # library
from sched_rt import SchedPlane

import test_dynamic_loop as tdl
from test_dynamic_loop import compile_kernel

NSEQ = int(os.environ.get("NSEQ", "8192"))
tdl.NSEQ = NSEQ
SD, PT = tdl.SD, tdl.PT
NBLK_LONG, NBLK_SHORT = 24, 3
PAD = 4096
STEPS = int(os.environ.get("STEPS", "30"))


def is_long(t):
    return t % 8 == 0


def main():
    torch.cuda.init()
    dev = "cuda"
    plane = SchedPlane(max_tasks=min(NSEQ, 16384), device=dev)

    # workload: long-tail decode shape (10-ish% long x8 KV)
    npages = NSEQ * NBLK_LONG
    g = torch.Generator().manual_seed(5)
    kv = (torch.rand(npages * PT * 2 * SD + PAD, generator=g) - 0.5).to(dev)
    q = (torch.rand(NSEQ * SD, generator=g) - 0.5).to(dev)
    bt = torch.empty(NSEQ * NBLK_LONG, dtype=torch.int32)
    nbl = torch.empty(NSEQ, dtype=torch.int32)
    for t in range(NSEQ):
        nbl[t] = NBLK_LONG if is_long(t) else NBLK_SHORT
        for b in range(NBLK_LONG):
            bt[t * NBLK_LONG + b] = (t * NBLK_LONG + b * 7 + 3) % npages
    bt, nbl = bt.to(dev), nbl.to(dev)
    out = torch.zeros(NSEQ * SD, device=dev)

    def run(lib):
        lib.launch_paged_softmax(kv.data_ptr(), bt.data_ptr(), nbl.data_ptr(),
                                 q.data_ptr(), out.data_ptr(), NSEQ,
                                 NBLK_LONG, PT,
                                 ctypes.c_void_p(
                                     torch.cuda.current_stream().cuda_stream))

    def step_us(lib, iters=STEPS):
        run(lib); torch.cuda.synchronize()  # warm
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(iters):
            run(lib)
        e1.record(); torch.cuda.synchronize()
        return e0.elapsed_time(e1) * 1000 / iters

    fails = [0]
    def ok(cond, name):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            fails[0] += 1

    print(f"== E1: timer channel ablation (paged softmax, {NSEQ} tiles, "
          f"long-tail mix, {STEPS} steps) ==")

    # (A) current design: host-mapped arena window.
    env_a = plane.bake_env(os.environ["SCHED_PLUGIN"])
    env_a["PATH"] = os.environ["PATH"]
    lib_a = compile_kernel(os.path.join(tempfile.gettempdir(),
                                        "tc_hostmapped.so"), env_a)

    # (B) device-buffer timer: same weave, timer baked at cudaMalloc'd memory.
    devbuf = torch.zeros(plane.N, dtype=torch.int64, device=dev)
    env_b = dict(env_a)
    env_b["SCHED_BAKE_TIMER"] = str(devbuf.data_ptr())
    lib_b = compile_kernel(os.path.join(tempfile.gettempdir(),
                                        "tc_devicebuf.so"), env_b)

    plane.set_num_tasks(NSEQ)

    # (C) baseline: timer gated off (both variants share ctrl).
    plane.set_timer_enabled(False)
    plane.push()
    t_off = step_us(lib_a)
    print(f"       timer OFF (flags gate)        {t_off:8.1f} us/step   --")

    # (A) host-mapped, timer on.
    plane.set_timer_enabled(True)
    plane.push()
    plane.clear_timer()
    t_host = step_us(lib_a)
    cyc_host = plane.read_timer()[:NSEQ].clone()
    print(f"       host-mapped PCIe atomics      {t_host:8.1f} us/step   "
          f"{100*(t_host-t_off)/t_off:+6.1f}%")

    # (B) device-buffer, timer on; readback = D2H copy + clear per probe.
    devbuf.zero_()
    t_dev = step_us(lib_b)
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    cyc_dev = devbuf.cpu()          # the probe-step readback
    devbuf.zero_()                  # and clear
    e1.record(); torch.cuda.synchronize()
    t_read = e0.elapsed_time(e1) * 1000
    print(f"       device-buffer atomics         {t_dev:8.1f} us/step   "
          f"{100*(t_dev-t_off)/t_off:+6.1f}%")
    print(f"       device readback+clear (D2H)   {t_read:8.1f} us/probe")

    # correctness: both channels observed every tile with consistent scale.
    ok(int((cyc_host > 0).sum()) == NSEQ, "host-mapped: one row per tile")
    ok(int((cyc_dev > 0).sum()) == NSEQ, "device-buffer: one row per tile")
    ratio = float(cyc_dev.float().mean() / max(float(cyc_host.float().mean()),
                                               1.0))
    ok(0.5 < ratio < 2.0,
       f"cycle magnitudes consistent across channels (ratio {ratio:.2f})")
    # both orders' LPT ranking should agree on the true longs.
    nlong = NSEQ // 8
    top_h = set(torch.argsort(cyc_host, descending=True)[:nlong].tolist())
    top_d = set(torch.argsort(cyc_dev, descending=True)[:nlong].tolist())
    agree = len(top_h & top_d) / nlong
    ok(agree > 0.9, f"LPT top-{nlong} agreement across channels "
                    f"({100*agree:.0f}%)")

    print("-- verdict --")
    print(f"  host-mapped observation tax : {100*(t_host-t_off)/t_off:+6.1f}% "
          f"per observed step (why sampling exists)")
    print(f"  device-buffer observation   : {100*(t_dev-t_off)/t_off:+6.1f}% "
          f"per step + {t_read:.0f} us/probe readback")
    print("== ALL PASS ==" if fails[0] == 0 else f"== {fails[0]} FAILED ==")
    return fails[0]


if __name__ == "__main__":
    raise SystemExit(main())
