"""test_gqa_repro_steps.py -- GQA corruption repro, iteration 2: MULTI-STEP.

Iteration 1 (test_gqa_repro): one-shot run, group-8/hd-128, woven-identity
vs stock -> BIT-EXACT. So the kernel weave alone is fine; serving corruption
needs more of the serving context. This adds, still without SGLang:
  * many sequential decode steps,
  * re-plan() between steps with CHANGING batch sizes (2/4/8 cycling -- the
    small-batch split_kv regime serving hits during ramp),
  * plugin-equivalent control writes between steps (num_tasks push, timer
    enable/disable via flags, clear, device-channel reads),
  * the DEVICE timer channel (-ti build), matching serving's default.
Compares the FULL SEQUENCE of outputs bit-exactly vs the stock phase.

Run:  SCHED_PLUGIN=... FLASHINFER_NVCC=... python test_gqa_repro_steps.py
"""
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))  # library

MARK = "SCHEDJSON "
NQO, NKV, HD = 16, 2, 128
STEPS = 48


def phase(mode) -> int:
    import torch
    import numpy as np
    out_dir = os.environ["GQA_TEST_DIR"]
    plane = None
    if mode == "woven":
        from sched_rt import SchedPlane
        plane = SchedPlane(max_tasks=1024)
        if os.environ.get("REPRO_TI", "1") != "0":
            plane.use_device_timer()
        plane.arm_process_env(os.environ["SCHED_PLUGIN"])
        plane.push()
    import flashinfer
    torch.manual_seed(11)
    dev = "cuda"
    max_pages, page_size = 16, 16
    BSMAX = 8
    kv_indptr_full = torch.arange(0, BSMAX * max_pages + 1, max_pages,
                                  dtype=torch.int32)
    kv_indices = torch.arange(BSMAX * max_pages, dtype=torch.int32,
                              device=dev)
    kv = torch.randn(BSMAX * max_pages, 2, page_size, NKV, HD,
                     dtype=torch.float16, device=dev)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev), "NHD")

    outs = []
    for s in range(STEPS):
        bs = (2, 4, 8)[s % 3]  # churn: forces re-plan + split-regime changes
        q = torch.randn(bs, NQO, HD, dtype=torch.float16, device=dev,
                        generator=torch.Generator(dev).manual_seed(100 + s))
        indptr = kv_indptr_full[:bs + 1].to(dev)
        last = torch.full((bs,), page_size, dtype=torch.int32, device=dev)
        w.plan(indptr, kv_indices, last, NQO, NKV, HD, page_size,
               data_type=torch.float16, q_data_type=torch.float16)
        if plane is not None:
            # plugin-equivalent per-step control writes
            plane.set_num_tasks(0)
            plane.set_timer_enabled(s % 8 == 0)
            plane.push()
            if s % 8 == 0:
                plane.clear_timer()
        o = w.run(q, kv)
        outs.append(o.detach().cpu().clone())
        if plane is not None and s % 8 == 7:
            _ = plane.read_timer()  # device-channel side-stream read
    torch.save(outs, os.path.join(out_dir, f"{mode}_steps.pt"))
    print(MARK + json.dumps({"steps": len(outs)}))
    return 0


def orchestrate() -> int:
    root = tempfile.mkdtemp(prefix="sched-gqa2-cache-")
    ddir = tempfile.mkdtemp(prefix="sched-gqa2-out-")

    def run_phase(mode):
        env = dict(os.environ, SCHED_CACHE_ROOT=root, GQA_TEST_DIR=ddir,
                   SCHED_DEBUG="1")
        r = subprocess.run([sys.executable, os.path.abspath(__file__),
                            "--phase", mode], env=env, capture_output=True,
                           text=True, timeout=3600)
        if MARK not in r.stdout:
            print(r.stdout[-1200:], r.stderr[-1200:], sep="\n")
            raise SystemExit(f"phase {mode} failed")

    import torch
    print(f"== GQA multi-step repro (g8 hd128, {STEPS} steps, churn 2/4/8, "
          f"device-timer build; cache {root}) ==")
    run_phase("stock")
    run_phase("woven")
    A = torch.load(os.path.join(ddir, "stock_steps.pt"))
    B = torch.load(os.path.join(ddir, "woven_steps.pt"))
    bad = [i for i, (a, b) in enumerate(zip(A, B)) if not torch.equal(a, b)]
    if bad:
        i = bad[0]
        d = (A[i].float() - B[i].float()).abs().max().item()
        print(f"  [FAIL] steps diverge: first at step {i} "
              f"(of {len(bad)} bad; maxdiff {d:.4f})")
        print("== 1 FAILED ==")
        return 1
    print(f"  [PASS] all {len(A)} steps bit-exact (plan churn + control "
          f"writes + device channel)")
    print("== ALL PASS ==")
    return 0


if __name__ == "__main__":
    if "--phase" in sys.argv:
        sys.exit(phase(sys.argv[sys.argv.index("--phase") + 1]))
    sys.exit(orchestrate())
