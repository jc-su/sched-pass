"""census_variants.py -- D2 weavability census over FlashInfer decode
variants, doubling as the GQA-corruption boundary map.

For each (group_size, head_dim) variant of the classic BatchDecode kernel:
compile stock and woven-identity (subprocess phases, isolated cache), then
report BIT-EXACTNESS plus the pass's own SCHED_DEBUG weave/decline notes.
Output: one table -- which variants weave what, and which corrupt (the P0's
boundary). Run after test_gqa_repro (same machinery, fuller matrix).

Run:  SCHED_PLUGIN=... FLASHINFER_NVCC=... python census_variants.py
Env:  CENSUS_GROUPS (default "1,2,4,8"), CENSUS_HDS ("64,128")
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))  # library

GROUPS = [int(g) for g in os.environ.get("CENSUS_GROUPS", "1,2,4,8").split(",")]
HDS = [int(h) for h in os.environ.get("CENSUS_HDS", "64,128").split(",")]
NKV = 2  # fixed kv heads; nqo = group * NKV


def main():
    import torch
    root = tempfile.mkdtemp(prefix="sched-census-cache-")
    ddir = tempfile.mkdtemp(prefix="sched-census-out-")
    repro = os.path.join(HERE, "test_gqa_repro.py")

    def run_phase(mode, nqo, nkv, hd):
        env = dict(os.environ, SCHED_CACHE_ROOT=root, GQA_TEST_DIR=ddir,
                   SCHED_DEBUG="1")
        r = subprocess.run([sys.executable, repro, "--phase", mode,
                            str(nqo), str(nkv), str(hd)],
                           env=env, capture_output=True, text=True,
                           timeout=3600)
        notes = [l for l in (r.stdout + r.stderr).splitlines()
                 if l.startswith("[sched]")]
        return r.returncode, notes

    def summarize(notes):
        s = []
        joined = "\n".join(notes)
        if "remapped" in joined:
            s.append("pi")
        if "prefetch site" in joined:
            s.append("prefetch")
        if "cp.async stream site" in joined:
            s.append("cpasync")
        if "shed" in joined and "DECLINED" not in joined and \
                "skipped" not in joined:
            s.append("shed")
        if "DECLINED" in joined or "skip" in joined.lower():
            s.append("declines")
        return ",".join(s) or "none-noted"

    print(f"== decode-variant census (nkv={NKV}; cache {root}) ==")
    print(f"  {'variant':>16s} {'bit-exact':>9s}  weave-notes")
    bad = []
    for hd in HDS:
        for g in GROUPS:
            nqo = g * NKV
            tag = f"{nqo}_{NKV}_{hd}"
            rc1, _ = run_phase("stock", nqo, NKV, hd)
            rc2, notes = run_phase("woven", nqo, NKV, hd)
            if rc1 != 0 or rc2 != 0:
                print(f"  g{g:<2d} hd{hd:<8d} {'PHASE-FAIL':>9s}")
                bad.append((g, hd, "phase-fail"))
                continue
            a = torch.load(os.path.join(ddir, f"stock_{tag}.pt"))
            b = torch.load(os.path.join(ddir, f"woven_{tag}.pt"))
            exact = torch.equal(a, b)
            print(f"  g{g:<2d} hd{hd:<8d} {'YES' if exact else 'NO':>9s}  "
                  f"{summarize(notes)}")
            if not exact:
                bad.append((g, hd,
                            f"maxdiff={(a.float()-b.float()).abs().max():.3f}"))
    print("-- corruption boundary --")
    for g, hd, d in bad:
        print(f"  CORRUPT: group={g} hd={hd} ({d})")
    if not bad:
        print("  none: all variants bit-exact")
    print("== CENSUS DONE ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
