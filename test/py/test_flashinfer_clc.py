"""test_flashinfer_clc.py -- the real-FlashInfer CLC work-queue path, A/B.

Two subprocess phases over identical seeded inputs (the test_fixed_va
pattern; the arena is one-per-process and the JIT env must differ):

  phase golden: pi-only weave (no SCHED_WORKQUEUE) -> reference outputs.
  phase clc:    SCHED_WORKQUEUE=1 SCHED_CLC=1 -> the persistent-worker + CLC
                claim driver on the REAL BatchDecode kernel (baked ABI,
                ctrl-only; cache dir gets the -wqclc mode tag). Asserts:
      * 1D grid (num_kv_heads=1): armed identity, reversed pi, and the
        num_tasks=0 disarm switch are ALL bit-exact vs golden;
      * 2D grid (num_kv_heads=8): grid = (bs, 8) -> the driver's grid-shape
        guard must take the stock path -- bit-exact (the soundness guard:
        tickets/CLC enumerate only the slot axis);
      * the cached .so's SASS contains the CLC claim (UGETNEXTWORKID);
      * reports R for the woven kernel (dlopen + occupancy on the stub) and
        whether the batch can steal at all (N > R).

Run:  SCHED_PLUGIN=.../libSchedPass.so python test_flashinfer_clc.py
"""
import glob
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))  # library

MARK = "SCHEDJSON "
BS_1D = int(os.environ.get("SCHED_CLC_TEST_BS", "1024"))


def build_decode(bs, nkv, nqo):
    import torch
    import flashinfer
    torch.manual_seed(0)
    dev = "cuda"
    max_pages, page_size, hd = 8, 16, 128
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
    return w, q, kv


def _ntiles(w, bs):
    """grid.x of the planned decode launch: padded_batch_size when the plan
    exposes the 10-int64 DecodePlanInfo, else bs."""
    v = getattr(w, "_plan_info", None)
    try:
        v = v.tolist() if hasattr(v, "tolist") else list(v)
        if len(v) == 10 and int(v[0]) >= bs:
            return int(v[0])
    except Exception:
        pass
    return bs


def phase(mode) -> int:
    import torch
    from sched_rt import SchedPlane
    plane = SchedPlane(max_tasks=4096)
    plane.arm_process_env(os.environ["SCHED_PLUGIN"])

    res = {"mode": mode}
    out_dir = os.environ["SCHED_CLC_TEST_DIR"]

    # --- 1D case: num_kv_heads = 1 -> grid (ntiles, 1, 1): claim path live --
    w, q, kv = build_decode(BS_1D, 1, 1)
    nt = _ntiles(w, BS_1D)
    plane.set_num_tasks(nt if mode == "clc" else 0)
    plane.push()
    plane.reset_order()
    torch.cuda.synchronize()
    o1 = w.run(q, kv).clone()
    res["ntiles_1d"] = nt

    if mode == "golden":
        torch.save({"o1": o1.cpu()}, os.path.join(out_dir, "golden_1d.pt"))
    else:
        g = torch.load(os.path.join(out_dir, "golden_1d.pt"))["o1"].cuda()
        res["bit_exact_identity"] = bool(torch.equal(o1, g))
        # reversed pi over the tile range: E1 permutation, must stay exact.
        order = torch.arange(plane.N, dtype=torch.int32)
        order[:nt] = torch.arange(nt - 1, -1, -1, dtype=torch.int32)
        plane.set_order(order)
        res["bit_exact_reversed"] = bool(torch.equal(w.run(q, kv), g))
        plane.reset_order()
        # the per-step disarm switch: num_tasks=0 -> stock static + pi.
        plane.set_num_tasks(0)
        plane.push()
        res["bit_exact_disarmed"] = bool(torch.equal(w.run(q, kv), g))
        plane.set_num_tasks(nt)
        plane.push()

    # --- 2D case: num_kv_heads = 8 -> grid (bs, 8): guard must go stock ----
    del w, q, kv
    torch.cuda.empty_cache()
    w2, q2, kv2 = build_decode(8, 8, 8)
    plane.set_num_tasks(8 if mode == "clc" else 0)
    plane.push()
    o2 = w2.run(q2, kv2).clone()
    if mode == "golden":
        torch.save({"o2": o2.cpu()}, os.path.join(out_dir, "golden_2d.pt"))
    else:
        g2 = torch.load(os.path.join(out_dir, "golden_2d.pt"))["o2"].cuda()
        res["bit_exact_2d_guard"] = bool(torch.equal(o2, g2))

        # SASS + R evidence from the cached woven .so.
        cache = os.path.join(os.environ["FLASHINFER_WORKSPACE_BASE"],
                             ".cache", "flashinfer")
        sos = sorted(glob.glob(os.path.join(cache, "**", "*batch_decode*.so"),
                               recursive=True))
        res["clc_sass"] = False
        if sos:
            cuobjdump = os.path.join(
                os.environ.get("SCHED_CUDA_PATH", "/usr/local/cuda-12.9"),
                "bin", "cuobjdump")
            if not os.path.exists(cuobjdump):
                cuobjdump = "cuobjdump"  # hope for PATH
            d = subprocess.run([cuobjdump, "--dump-sass", sos[0]],
                               capture_output=True, text=True, timeout=300)
            res["clc_sass"] = "UGETNEXTWORKID" in d.stdout
            res["R"] = plane.r_for_cached_so(sos[0], "BatchDecode", 128)
            res["can_steal_1d"] = res.get("R", 0) > 0 and nt > res["R"]
    print(MARK + json.dumps(res))
    return 0


def orchestrate() -> int:
    # SCHED_CLC_TEST_CACHE reuses a cache root (fast reruns; the JIT cache is
    # keyed by va+N+mode, so stale-mode reuse is impossible by construction).
    root = os.environ.get("SCHED_CLC_TEST_CACHE") or \
        tempfile.mkdtemp(prefix="sched-clc-cache-")
    ddir = tempfile.mkdtemp(prefix="sched-clc-golden-")

    def run_phase(mode, extra_env):
        env = dict(os.environ, SCHED_CACHE_ROOT=root, SCHED_CLC_TEST_DIR=ddir)
        env.update(extra_env)
        r = subprocess.run([sys.executable, os.path.abspath(__file__),
                            "--phase", mode], env=env, capture_output=True,
                           text=True, timeout=3600)
        for line in r.stdout.splitlines():
            if line.startswith(MARK):
                return json.loads(line[len(MARK):])
        print(r.stdout[-2000:], r.stderr[-2000:], sep="\n")
        raise SystemExit(f"phase {mode} produced no result")

    g = run_phase("golden", {})
    c = run_phase("clc", {"SCHED_WORKQUEUE": "1", "SCHED_CLC": "1"})

    fails = 0
    def ok(cond, name):
        nonlocal fails
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        fails += 0 if cond else 1

    print(f"== FlashInfer CLC work-queue A/B (bs={BS_1D}, cache {root}) ==")
    print(f"  woven kernel R = {c.get('R', '?')}  ntiles = {c['ntiles_1d']}  "
          f"can_steal = {c.get('can_steal_1d', '?')}")
    ok(c.get("bit_exact_identity"), "1D armed identity: bit-exact vs pi-only")
    ok(c.get("bit_exact_reversed"), "1D armed reversed pi: bit-exact")
    ok(c.get("bit_exact_disarmed"),
       "num_tasks=0 disarm switch: stock static path bit-exact")
    ok(c.get("bit_exact_2d_guard"),
       "2D grid (num_kv_heads=8): shape guard -> stock, bit-exact")
    ok(c.get("clc_sass"), "cached .so SASS contains the CLC claim "
                          "(UGETNEXTWORKID)")
    print("== ALL PASS ==" if fails == 0 else f"== {fails} FAILED ==")
    return fails


if __name__ == "__main__":
    if "--phase" in sys.argv:
        sys.exit(phase(sys.argv[sys.argv.index("--phase") + 1]))
    sys.exit(orchestrate())
