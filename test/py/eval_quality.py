"""eval_quality.py -- the E2(epsilon) quality contract, measured.

Shed (tau) is the one lever that trades accuracy for time; THEORY.md #4 says
it is admissible ONLY behind an explicit budget with a quality contract. This
harness measures that contract on the real online-softmax paged-attention JIT
kernel: sweep tau, and for each setting report
  * exactness vs the truncated-attention CPU reference (the SEMANTICS gate:
    dropping a token must contribute exactly zero weight), and
  * quality vs FULL attention (cosine similarity + max relative error --
    the epsilon the control plane spends when it sets tau).

Gates: tau=0 matches full attention; every tau matches its truncated reference;
quality degrades monotonically as tau shrinks (sanity of the epsilon knob).

Run:  SCHED_PLUGIN=.../libSchedPass.so python eval_quality.py
"""
import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))  # library
from sched_rt import SchedPlane
from test_dynamic_loop import (NSEQ, NBLK_LONG, compile_kernel, cpu_ref,
                               make_inputs, run)


def main():
    torch.manual_seed(0)
    kv, bt, nbl, q, out = make_inputs()

    plane = SchedPlane(max_tasks=NSEQ)
    env = plane.bake_env(os.environ["SCHED_PLUGIN"])
    lib = compile_kernel(os.path.join(tempfile.gettempdir(), "qual_woven.so"),
                         env)
    plane.set_num_tasks(NSEQ)
    plane.push()

    full = cpu_ref(kv, bt, nbl, q, tau=0).to("cuda")
    fails = 0
    def ok(cond, name):
        nonlocal fails
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        fails += 0 if cond else 1

    print(f"== shed quality curve (E2 contract), {NSEQ} requests ==")
    print(f"  {'tau':>5s} {'exact-vs-trunc-ref':>19s} {'cos-vs-full':>12s} "
          f"{'max-rel-err':>12s}")
    last_cos = None
    curve_ok = True
    for tau in (0, 24, 16, 8, 4):
        plane.set_rows(tau=[tau] * NSEQ, n=NSEQ)
        plane.push()
        o = run(lib, kv, bt, nbl, q, out, NBLK_LONG)
        ref = cpu_ref(kv, bt, nbl, q, tau=tau).to("cuda")
        # tau=0 vs the CPU full reference is allclose, not bit-exact (GPU
        # accumulation order differs; bit-exactness of tau=0 vs the GPU stock
        # kernel is gated separately in test_dynamic_loop).
        exact = torch.allclose(o, ref, atol=2e-2, rtol=2e-2)
        cos = torch.nn.functional.cosine_similarity(
            o.flatten(), full.flatten(), dim=0).item()
        rel = ((o - full).abs() / full.abs().clamp_min(1e-6)).max().item()
        print(f"  {tau:5d} {str(exact):>19s} {cos:12.6f} {rel:12.4f}")
        if not exact:
            curve_ok = False
        if last_cos is not None and cos > last_cos + 1e-6:
            curve_ok = False  # smaller budget must not IMPROVE quality
        last_cos = cos

    ok(curve_ok, "tau=0 matches full; every tau matches its truncated reference; "
                 "quality monotone in the budget")
    print("== ALL PASS ==" if fails == 0 else "== FAILED ==")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
