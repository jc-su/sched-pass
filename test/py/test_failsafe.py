"""test_failsafe.py -- P1 failure injection: every discovered failure mode
must downgrade LOUDLY AND SAFELY, never silently corrupt.

  1. canonical-VA miss: occupy the canonical VA first, then create the plane
     -> lands elsewhere, canonical=False, and bake_env keys the cache by the
     ACTUAL base (correct-or-recompile);
  2. batch > max_tasks: the plugin raises a loud error, never truncates;
  3. plan-layout mismatch: tile binding falls back to 1-tile-per-request
     (fail-soft) on a wrapper whose _plan_info is not the known layout;
  4. shim bake-invariant: SCHED_BAKE_* with an unkeyed workspace -> the shim
     strips the bake vars (compiles unwoven) and says so on stderr.

Run:  SCHED_PLUGIN=.../libSchedPass.so python test_failsafe.py
"""
import ctypes
import os
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))  # library
import sched_rt
from sched_rt import SchedPlane, DEFAULT_VA_BASE, ARENA_BYTES


def main():
    fails = [0]
    def ok(cond, name):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            fails[0] += 1

    print("== failure injection: loud, safe downgrades ==")

    # 1. canonical-VA miss -> non-canonical arena + actual-base cache key.
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_long]
    blocker = libc.mmap(ctypes.c_void_p(DEFAULT_VA_BASE), ARENA_BYTES, 0x3,
                        0x22 | 0x100000, -1, 0)  # occupy the canonical VA
    ok(blocker == DEFAULT_VA_BASE, "test setup: canonical VA occupied")
    plane = SchedPlane(max_tasks=64)
    ok(not plane.arena.canonical, "VA miss detected (canonical=False, loud)")
    env = plane.bake_env("/nonexistent/plugin.so")
    ok(f"va{plane.arena.base:x}" in env["FLASHINFER_WORKSPACE_BASE"],
       "cache keyed by ACTUAL base -> stale canonical kernels cannot load")
    ok(plane.arena.base != DEFAULT_VA_BASE,
       "arena relocated, correctness preserved")

    # 2. batch > max_tasks: loud error, never truncation.
    import sched_sglang_plugin as plug
    plug._PLANE = plane
    try:
        plug._plane(4096)
        ok(False, "oversized batch raises")
    except RuntimeError as e:
        ok("SCHED_MAX_TASKS" in str(e), "oversized batch raises loudly")

    # 3. plan-layout mismatch -> 1-tile-per-request fallback.
    class FakeWrapper:
        _plan_info = [1, 2, 3]          # not the 10-int64 decode layout
        _int_workspace_buffer = torch.zeros(64, dtype=torch.uint8)
    class FakeBackend:
        decode_wrapper = FakeWrapper()
    class FakeMR:
        attn_backend = FakeBackend()
    class FakeTP:
        model_runner = FakeMR()
    class FakeSched:
        tp_worker = FakeTP()
    t2r, ntiles = plug._tile_binding(FakeSched(), 8)
    ok(t2r is None and ntiles == 8,
       "unknown plan layout -> 1-tile-per-request fallback (fail-soft)")

    # 4. shim invariant: bake vars + unkeyed workspace -> stripped, loud.
    env = dict(os.environ)
    env["SCHED_BAKE_TASK_ORDER"] = str(DEFAULT_VA_BASE)
    env["FLASHINFER_WORKSPACE_BASE"] = "/tmp/not-keyed"
    env.pop("SCHED_WEAVE_ONLY", None)
    r = subprocess.run([sys.executable,
                        os.path.join(HERE, "..", "..", "python", "nvcc_clang_shim.py"), "--version"],
                       env=env, capture_output=True, text=True, timeout=30)
    # --version short-circuits before the strip; use a compile-shaped argv.
    r = subprocess.run([sys.executable,
                        os.path.join(HERE, "..", "..", "python", "nvcc_clang_shim.py"),
                        "-c", "/dev/null", "-o", "/dev/null"],
                       env=env, capture_output=True, text=True, timeout=60)
    ok("stripped" in r.stderr,
       "shim strips bake vars for unkeyed workspace (loud on stderr)")

    print("== ALL PASS ==" if fails[0] == 0 else f"== {fails[0]} FAILED ==")
    return fails[0]


if __name__ == "__main__":
    raise SystemExit(main())
