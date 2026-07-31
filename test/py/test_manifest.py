"""test_manifest.py -- D1 cross-language consistency gate (CPU-only).

The capability manifest lives in TWO places by necessity (a C++ pass header
and a Python control-plane mirror). This gate proves they agree: it dumps the
authoritative C++ manifest (SchedManifest.h) by loading the plugin under
`SCHED_MANIFEST_DUMP=csv` via `opt`, parses the CSV, and asserts the Python
mirror (sched_rt.MANIFEST) matches it row-for-row -- so a capability added or
renamed on one side cannot silently drift from the other. Also checks the
manifest's own invariant: order ranks are monotonic (the pass/emit order the
plugin derives from the table).

Run:  PLUGIN=build/libSchedPass.so LLVM_BIN=/usr/lib/llvm-22/bin \
      python test/py/test_manifest.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))  # library
from sched_rt import MANIFEST

ROOT = os.path.dirname(os.path.dirname(HERE))
PLUGIN = os.environ.get("PLUGIN", os.path.join(ROOT, "build",
                                               "libSchedPass.so"))
LLVM_BIN = os.environ.get("LLVM_BIN", "/usr/lib/llvm-22/bin")


def dump_cpp_manifest():
    """Load the plugin under opt with SCHED_MANIFEST_DUMP=csv; the manifest is
    printed at plugin-load (llvmGetPassPluginInfo), before any pass runs, so a
    trivial `-passes=verify` on empty IR suffices."""
    env = dict(os.environ, SCHED_MANIFEST_DUMP="csv")
    r = subprocess.run(
        [os.path.join(LLVM_BIN, "opt"), f"-load-pass-plugin={PLUGIN}",
         "-passes=verify", "-disable-output", "/dev/null"],
        env=env, capture_output=True, text=True, timeout=60)
    rows = []
    for line in (r.stderr + r.stdout).splitlines():
        parts = line.split(",")
        if len(parts) == 7 and parts[0] not in ("name",) and parts[5].lstrip(
                "-").isdigit():
            rows.append(parts)  # name,effect,minSm,knob,disableKnob,order,tag
    return rows


def main():
    fails = 0

    def ok(cond, name):
        nonlocal fails
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        fails += 0 if cond else 1

    print("== capability manifest: C++ <-> Python consistency ==")
    cpp = dump_cpp_manifest()
    ok(len(cpp) == len(MANIFEST),
       f"row count matches ({len(cpp)} C++ vs {len(MANIFEST)} python)")
    if len(cpp) != len(MANIFEST):
        for c in cpp:
            print("     | cpp:", ",".join(c))
        print("== 1 FAILED =="); return 1

    orders = [int(c[5]) for c in cpp]
    ok(orders == sorted(orders), f"pass/emit order monotonic {orders}")
    # list position IS the emit order: the C++ `order` field must equal the row
    # index (so the manifest genuinely drives order, not just documents it).
    ok(orders == list(range(len(orders))),
       f"order field == row index {orders}")

    for i, (c, (name, effect, min_sm, knob, disable, tag)) in enumerate(
            zip(cpp, MANIFEST)):
        # compare every field the manifest declares, INCLUDING tag + order
        row = (f"{name}/{effect}/sm{min_sm}/{knob}/"
               f"{'NO_' if disable else 'on'}/tag={tag or '-'}/ord={i}")
        c_row = (f"{c[0]}/{c[1]}/sm{c[2]}/{c[3]}/"
                 f"{'NO_' if c[4] == '1' else 'on'}/tag={c[6] or '-'}/ord={c[5]}")
        ok(row == c_row, f"{name}: python == C++ (all fields)")
        if row != c_row:
            print(f"     | py : {row}")
            print(f"     | cpp: {c_row}")

    print("== ALL PASS ==" if fails == 0 else f"== {fails} FAILED ==")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
