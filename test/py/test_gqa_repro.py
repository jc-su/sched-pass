"""test_gqa_repro.py -- P0 minimal repro: does weaving corrupt the GQA
(group>1) FlashInfer decode variant OUTSIDE SGLang?

Serving bisect showed: clang-unwoven 3B coherent; ANY weave (even timer-only
or pi-only, output-neutral by type) -> fluent wrong-prompt text. This strips
SGLang away: same wrapper, same seeded inputs, stock phase vs woven phase
(subprocess isolation, the test_fixed_va pattern), BIT-compare.

  GQA config under test: nqo=16, nkv=2 (group 8), hd=128  -- the 3B shape.
  Control config:        nqo=8,  nkv=8 (group 1), hd=128  -- the validated
                         shape (test_flashinfer_arm) -- must stay exact.

Also captures, per woven compile, the pass's SCHED_DEBUG notes (which
detectors fired) and the cached .so path (for the PTX/SASS diff step).

Run:  SCHED_PLUGIN=... FLASHINFER_NVCC="python3 .../nvcc_clang_shim.py" \
      python test_gqa_repro.py
"""
import glob
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))  # library

MARK = "SCHEDJSON "


def build_and_run(nqo, nkv, hd):
    import torch
    import flashinfer
    torch.manual_seed(7)
    dev = "cuda"
    bs, max_pages, page_size = 8, 8, 16
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
    torch.cuda.synchronize()
    return w.run(q, kv).clone()


def phase(mode, nqo, nkv, hd) -> int:
    import torch
    out_dir = os.environ["GQA_TEST_DIR"]
    tag = f"{nqo}_{nkv}_{hd}"
    if mode == "woven":
        from sched_rt import SchedPlane
        plane = SchedPlane(max_tasks=1024)
        plane.arm_process_env(os.environ["SCHED_PLUGIN"])
        plane.set_num_tasks(1024)
        plane.push()
    o = build_and_run(nqo, nkv, hd)
    torch.save(o.cpu(), os.path.join(out_dir, f"{mode}_{tag}.pt"))
    sos = []
    base = os.environ.get("FLASHINFER_WORKSPACE_BASE",
                          os.path.expanduser("~"))
    sos = sorted(glob.glob(os.path.join(
        base, ".cache", "flashinfer", "**", "*batch_decode*.so"),
        recursive=True))
    print(MARK + json.dumps({"so": sos[0] if sos else ""}))
    return 0


def orchestrate() -> int:
    root = tempfile.mkdtemp(prefix="sched-gqa-cache-")
    ddir = tempfile.mkdtemp(prefix="sched-gqa-out-")

    def run_phase(mode, nqo, nkv, hd):
        env = dict(os.environ, SCHED_CACHE_ROOT=root, GQA_TEST_DIR=ddir,
                   SCHED_DEBUG="1")
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--phase", mode,
             str(nqo), str(nkv), str(hd)],
            env=env, capture_output=True, text=True, timeout=3600)
        so = ""
        for line in r.stdout.splitlines():
            if line.startswith(MARK):
                so = json.loads(line[len(MARK):])["so"]
        dbg = [l for l in (r.stdout + r.stderr).splitlines()
               if l.startswith("[sched]")]
        if not so and mode == "woven":
            print(r.stdout[-1200:], r.stderr[-1200:], sep="\n")
        return so, dbg

    import torch
    fails = 0
    def ok(cond, name):
        nonlocal fails
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        fails += 0 if cond else 1

    print(f"== GQA weave repro (cache {root}) ==")
    for (nqo, nkv, hd, label) in ((8, 8, 128, "group1-control"),
                                  (16, 2, 128, "group8-3B-shape")):
        tag = f"{nqo}_{nkv}_{hd}"
        run_phase("stock", nqo, nkv, hd)
        so, dbg = run_phase("woven", nqo, nkv, hd)
        a = torch.load(os.path.join(ddir, f"stock_{tag}.pt"))
        b = torch.load(os.path.join(ddir, f"woven_{tag}.pt"))
        exact = torch.equal(a, b)
        close = torch.allclose(a.float(), b.float(), atol=2e-2, rtol=2e-2)
        ok(exact, f"{label}: woven-identity bit-exact vs stock")
        if not exact:
            print(f"     | close={close} maxdiff="
                  f"{(a.float()-b.float()).abs().max().item():.4f}")
            print(f"     | woven .so: {so}")
        for l in dbg[:8]:
            print(f"     | {l}")
    print("== ALL PASS ==" if fails == 0 else f"== {fails} FAILED ==")
    return fails


if __name__ == "__main__":
    if "--phase" in sys.argv:
        i = sys.argv.index("--phase")
        sys.exit(phase(sys.argv[i + 1], int(sys.argv[i + 2]),
                       int(sys.argv[i + 3]), int(sys.argv[i + 4])))
    sys.exit(orchestrate())
