"""test_fixed_va.py -- the JIT-cache contract, end to end across processes.

Production requirement: a woven FlashInfer kernel compiled (with baked table
addresses) in one process must be VALID and ARMED in a later process without
recompiling. SchedArena provides this by mapping the control tables at a
canonical fixed VA; bake_env() keys the FlashInfer cache dir by the actual
base. This test proves the whole contract:

  phase 1 (fresh cache root): plane at fixed VA -> JIT-compile woven decode ->
          armed pi permutation bit-exact -> record base, .so mtimes, checksum.
  phase 2 (NEW process, same cache root): plane must land at the SAME VA, the
          cached .so must be reused UNMODIFIED (no recompile), the armed
          permutation must be bit-exact, and outputs must equal phase 1's.

Run (orchestrator; spawns both phases):
    SCHED_PLUGIN=.../libSchedPass.so python test_fixed_va.py
"""
import glob
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))  # library

MARK = "SCHEDJSON "


def build_decode():
    """The same tiny paged-decode workload as test_flashinfer_arm."""
    import torch
    import flashinfer
    torch.manual_seed(0)
    dev = "cuda"
    bs, max_pages, page_size = 8, 8, 16
    nqo, nkv, hd = 8, 8, 128
    kv_indptr = torch.arange(0, bs * max_pages + 1, max_pages,
                             dtype=torch.int32, device=dev)
    kv_indices = torch.arange(bs * max_pages, dtype=torch.int32, device=dev)
    last_len = torch.full((bs,), page_size, dtype=torch.int32, device=dev)
    q = torch.randn(bs, nqo, hd, dtype=torch.float16, device=dev)
    kv = torch.randn(bs * max_pages, 2, page_size, nkv, hd,
                     dtype=torch.float16, device=dev)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev), "NHD")
    w.plan(kv_indptr, kv_indices, last_len, nqo, nkv, hd, page_size,
           data_type=torch.float16, q_data_type=torch.float16)
    return w, q, kv, bs


def phase() -> int:
    import torch
    from sched_rt import SchedPlane
    plane = SchedPlane(max_tasks=1024)
    plane.arm_process_env(os.environ["SCHED_PLUGIN"])
    plane.set_num_tasks(plane.N)
    plane.push()

    w, q, kv, bs = build_decode()  # JIT happens here (or cache hit)
    plane.reset_order()
    torch.cuda.synchronize()
    golden = w.run(q, kv).clone()

    order = torch.arange(plane.N, dtype=torch.int32)
    order[:bs] = order[:bs].view(-1, 2).flip(1).reshape(-1)  # swap pairs
    plane.set_order(order)
    permuted = w.run(q, kv).clone()

    # note: glob's ** skips hidden dirs, and FlashInfer nests under .cache/
    cache_dir = os.path.join(os.environ["FLASHINFER_WORKSPACE_BASE"],
                             ".cache", "flashinfer")
    sos = sorted(glob.glob(os.path.join(cache_dir, "**", "*.so"),
                           recursive=True))
    print(MARK + json.dumps({
        "base": plane.arena.base,
        "canonical": plane.arena.canonical,
        "golden_sum": float(golden.float().sum()),
        "bit_exact": bool(torch.equal(golden, permuted)),
        "so_mtimes": {os.path.relpath(p, cache_dir): os.path.getmtime(p)
                      for p in sos},
    }))
    return 0


def orchestrate() -> int:
    root = tempfile.mkdtemp(prefix="sched-cache-")
    env = dict(os.environ, SCHED_CACHE_ROOT=root)

    def run_phase(n):
        r = subprocess.run([sys.executable, os.path.abspath(__file__),
                            "--phase", str(n)],
                           env=env, capture_output=True, text=True,
                           timeout=1800)
        for line in r.stdout.splitlines():
            if line.startswith(MARK):
                return json.loads(line[len(MARK):])
        print(r.stdout[-1500:], r.stderr[-1500:], sep="\n")
        raise SystemExit(f"phase {n} produced no result")

    p1 = run_phase(1)
    p2 = run_phase(2)

    fails = 0
    def ok(cond, name):
        nonlocal fails
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        fails += 0 if cond else 1

    print(f"== fixed-VA cross-process JIT cache (root {root}) ==")
    print(f"  phase1 base 0x{p1['base']:x} canonical={p1['canonical']} | "
          f"phase2 base 0x{p2['base']:x} canonical={p2['canonical']}")
    ok(p1["canonical"] and p2["canonical"],
       "both processes obtained the canonical VA")
    ok(p1["base"] == p2["base"], "table addresses identical across processes")
    ok(p1["bit_exact"] and p2["bit_exact"],
       "armed pi permutation bit-exact in both processes")
    ok(p1["golden_sum"] == p2["golden_sum"],
       f"outputs identical across processes (sum {p1['golden_sum']:.4f})")
    ok(len(p2["so_mtimes"]) > 0 and p2["so_mtimes"] == p1["so_mtimes"],
       f"phase 2 REUSED the cached kernels unmodified "
       f"({len(p2['so_mtimes'])} .so, no recompile)")
    print("== PASS ==" if fails == 0 else f"== {fails} FAILED ==")
    return fails


if __name__ == "__main__":
    if "--phase" in sys.argv:
        sys.exit(phase())
    sys.exit(orchestrate())
