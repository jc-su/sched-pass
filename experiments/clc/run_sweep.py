#!/usr/bin/env python3
"""Run synthetic Blackwell CLC characterization sweeps.

The probe binary is standalone CUDA; this script coordinates repeatable runs,
validates the observable CLC contract, and writes raw/summary CSV files under
experiments/clc/results.
"""

import argparse
import csv
import os
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "build" / "clc_probe"
OUTDIR = Path(__file__).resolve().parent / "results"

LAYOUT_NAMES = {
    0: "interleaved",
    1: "long-prefix",
    2: "long-suffix",
}

NUMERIC = {
    "tasks": int,
    "threads": int,
    "long_every": int,
    "short_cycles": int,
    "long_cycles": int,
    "layout": int,
    "static_us": float,
    "clc_us": float,
    "delta_pct": float,
    "processed": int,
    "active_workers": int,
    "missed": int,
    "duplicates": int,
    "attempts": int,
    "successes": int,
    "success_rate": float,
    "unique_claimed": int,
    "claimed_min": int,
    "claimed_max": int,
    "claim_cycles_avg": float,
    "claim_cycles_max": int,
    "smem_bytes": int,
    "occ_blocks_per_sm": int,
    "predicted_r": int,
    "max_processed": int,
    "multi_claim_workers": int,
    "max_successes_worker": int,
    "first_claim_min": int,
    "first_claim_max": int,
    "last_claim_min": int,
    "last_claim_max": int,
    "sm_count": int,
    "expected_active_workers": int,
    "expected_claimed": int,
    "duplicate_claims": int,
    "claim_range_holes": int,
    "structural_ok": int,
}

CFG_FIELDS = (
    "tasks",
    "threads",
    "long_every",
    "short_cycles",
    "long_cycles",
    "layout",
    "smem_bytes",
)


def cfg(suite, tasks, threads, long_every, short_cycles, long_cycles,
        layout=0, smem_bytes=0):
    return {
        "suite": suite,
        "tasks": tasks,
        "threads": threads,
        "long_every": long_every,
        "short_cycles": short_cycles,
        "long_cycles": long_cycles,
        "layout": layout,
        "smem_bytes": smem_bytes,
    }


def parse_probe_csv(stdout):
    lines = [ln for ln in stdout.splitlines() if ln and not ln.startswith("==")]
    header_idx = next(i for i, ln in enumerate(lines) if ln.startswith("tasks,"))
    row = next(csv.DictReader(lines[header_idx:header_idx + 2]))
    return {k: NUMERIC[k](v) for k, v in row.items()}


def run_one(config):
    env = dict(os.environ)
    env["CLC_PROBE_CSV"] = "1"
    cmd = [str(BIN)] + [str(config[k]) for k in CFG_FIELDS]
    cp = subprocess.run(cmd, env=env, text=True, capture_output=True,
                        check=True)
    row = parse_probe_csv(cp.stdout)
    row["suite"] = config["suite"]
    row["layout_name"] = LAYOUT_NAMES.get(row["layout"], str(row["layout"]))
    return row


def validate_row(row):
    errors = []
    tasks = row["tasks"]
    predicted_r = row["predicted_r"]
    expected_active = min(tasks, predicted_r)
    expected_claimed = max(0, tasks - predicted_r)

    if row["missed"] != 0:
        errors.append(f"missed={row['missed']}")
    if row["duplicates"] != 0:
        errors.append(f"duplicates={row['duplicates']}")
    if row["duplicate_claims"] != 0:
        errors.append(f"duplicate_claims={row['duplicate_claims']}")
    if row["claim_range_holes"] != 0:
        errors.append(f"claim_range_holes={row['claim_range_holes']}")
    if row["active_workers"] != expected_active:
        errors.append(
            f"active_workers={row['active_workers']} expected={expected_active}")
    if row["expected_active_workers"] != expected_active:
        errors.append("probe expected_active mismatch")
    if row["expected_claimed"] != expected_claimed:
        errors.append("probe expected_claimed mismatch")

    if tasks <= predicted_r:
        if row["successes"] != 0 or row["unique_claimed"] != 0:
            errors.append("unexpected claim below/equal R")
    else:
        if row["unique_claimed"] != expected_claimed:
            errors.append(
                f"unique_claimed={row['unique_claimed']} expected={expected_claimed}")
        if row["claimed_min"] != predicted_r:
            errors.append(
                f"claimed_min={row['claimed_min']} expected={predicted_r}")
        if row["claimed_max"] != tasks - 1:
            errors.append(
                f"claimed_max={row['claimed_max']} expected={tasks - 1}")

    if row["structural_ok"] != 1:
        errors.append("probe structural_ok=0")
    return errors


def threshold_configs():
    task_points = (1024, 1128, 1536, 2048, 2256, 2304, 3072, 4096, 4512,
                   4608, 8192, 16384)
    for threads in (64, 128, 256):
        for tasks in task_points:
            yield cfg("threshold", tasks, threads, 0, 8192, 8192)


def predict_r(threads, smem_bytes):
    probe = cfg("predict", 8192, threads, 0, 1024, 1024, 0, smem_bytes)
    row = run_one(probe)
    return row["predicted_r"]


def occupancy_configs():
    smem_values = (0, 4096, 8192, 12288, 16384, 24576, 32768)
    for threads in (64, 128, 256):
        for smem_bytes in smem_values:
            predicted_r = predict_r(threads, smem_bytes)
            task_points = sorted({
                max(1, predicted_r - 1),
                predicted_r,
                predicted_r + max(64, min(512, predicted_r // 4)),
            })
            print(f"[predict] threads={threads} smem={smem_bytes} "
                  f"predicted_R={predicted_r} tasks={task_points}")
            for tasks in task_points:
                yield cfg("occupancy", tasks, threads, 0, 4096, 4096,
                          0, smem_bytes)


def claim_order_configs():
    for tasks in (8192, 16384):
        for smem_bytes in (0, 16384):
            yield cfg("claim-order", tasks, 128, 0, 4096, 4096,
                      0, smem_bytes)
            for layout in (0, 1, 2):
                yield cfg("claim-order", tasks, 128, 8, 1024, 32768,
                          layout, smem_bytes)


def workload_configs():
    for tasks in (4096, 8192, 16384):
        for long_every in (8, 4):
            for layout in (0, 1, 2):
                yield cfg("workload", tasks, 128, long_every, 1024, 32768,
                          layout, 0)

    for threads in (64, 128, 256):
        for short_cycles, long_cycles in ((256, 8192), (1024, 32768),
                                          (4096, 131072)):
            yield cfg("workload", 8192, threads, 8, short_cycles,
                      long_cycles, 0, 0)


def configs_for_suite(suite):
    if suite == "threshold":
        return list(threshold_configs())
    if suite == "occupancy":
        return list(occupancy_configs())
    if suite == "claim-order":
        return list(claim_order_configs())
    if suite == "workload":
        return list(workload_configs())
    if suite == "all":
        out = []
        for maker in (threshold_configs, occupancy_configs,
                      claim_order_configs, workload_configs):
            out.extend(maker())
        return out
    raise ValueError(suite)


def group_key(row):
    return (row["suite"], row["tasks"], row["threads"], row["long_every"],
            row["short_cycles"], row["long_cycles"], row["layout"],
            row["smem_bytes"])


def mean(rows, field):
    return statistics.fmean(r[field] for r in rows)


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)

    out = []
    for key, rs in sorted(grouped.items()):
        base = {
            "suite": key[0],
            "tasks": key[1],
            "threads": key[2],
            "long_every": key[3],
            "short_cycles": key[4],
            "long_cycles": key[5],
            "layout": key[6],
            "layout_name": LAYOUT_NAMES.get(key[6], str(key[6])),
            "smem_bytes": key[7],
            "repeats": len(rs),
            "static_us_mean": mean(rs, "static_us"),
            "clc_us_mean": mean(rs, "clc_us"),
            "delta_pct_mean": mean(rs, "delta_pct"),
            "delta_pct_stdev": statistics.stdev(r["delta_pct"] for r in rs)
            if len(rs) > 1 else 0.0,
            "success_rate_mean": mean(rs, "success_rate"),
            "active_workers_mean": mean(rs, "active_workers"),
            "occ_blocks_per_sm_mean": mean(rs, "occ_blocks_per_sm"),
            "predicted_r_mean": mean(rs, "predicted_r"),
            "expected_claimed_mean": mean(rs, "expected_claimed"),
            "unique_claimed_mean": mean(rs, "unique_claimed"),
            "claimed_min_min": min(r["claimed_min"] for r in rs),
            "claimed_max_max": max(r["claimed_max"] for r in rs),
            "first_claim_min_min": min(r["first_claim_min"] for r in rs),
            "first_claim_max_max": max(r["first_claim_max"] for r in rs),
            "last_claim_min_min": min(r["last_claim_min"] for r in rs),
            "last_claim_max_max": max(r["last_claim_max"] for r in rs),
            "claim_cycles_avg_mean": mean(rs, "claim_cycles_avg"),
            "claim_cycles_max_max": max(r["claim_cycles_max"] for r in rs),
            "max_processed_max": max(r["max_processed"] for r in rs),
            "multi_claim_workers_mean": mean(rs, "multi_claim_workers"),
            "max_successes_worker_max": max(r["max_successes_worker"]
                                            for r in rs),
            "missed_max": max(r["missed"] for r in rs),
            "duplicates_max": max(r["duplicates"] for r in rs),
            "duplicate_claims_max": max(r["duplicate_claims"] for r in rs),
            "claim_range_holes_max": max(r["claim_range_holes"] for r in rs),
            "structural_ok_min": min(r["structural_ok"] for r in rs),
            "valid_runs": sum(1 for r in rs if r["validation"] == "ok"),
        }
        out.append(base)
    return out


def fieldnames(rows):
    out = []
    for row in rows:
        for key in row.keys():
            if key not in out:
                out.append(key)
    return out


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=("threshold", "occupancy",
                                        "claim-order", "workload", "all"),
                    default="all")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    if not BIN.exists():
        raise SystemExit(f"probe binary not found: {BIN}; run build_run.sh first")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    rows = []
    failures = []
    cfgs = configs_for_suite(args.suite)
    total = len(cfgs) * args.repeats
    n = 0
    for config in cfgs:
        for rep in range(args.repeats):
            n += 1
            row = run_one(config)
            row["repeat"] = rep
            errors = validate_row(row)
            row["validation"] = "ok" if not errors else "; ".join(errors)
            rows.append(row)
            if errors:
                failures.append((row, errors))
            print(f"[{n:03d}/{total}] suite={config['suite']} "
                  f"tasks={config['tasks']} threads={config['threads']} "
                  f"smem={config['smem_bytes']} layout={row['layout_name']} "
                  f"R={row['predicted_r']} claim={row['claimed_min']}.."
                  f"{row['claimed_max']} delta={row['delta_pct']:+.1f}% "
                  f"valid={row['validation'] == 'ok'}")

    raw_path = OUTDIR / f"clc_{args.suite}_raw_{stamp}.csv"
    summary_path = OUTDIR / f"clc_{args.suite}_summary_{stamp}.csv"
    write_csv(raw_path, rows)
    write_csv(summary_path, aggregate(rows))
    print(f"raw={raw_path}")
    print(f"summary={summary_path}")

    if failures:
        print("validation failures:")
        for row, errors in failures[:20]:
            print(f"  suite={row['suite']} tasks={row['tasks']} "
                  f"threads={row['threads']} smem={row['smem_bytes']}: "
                  f"{'; '.join(errors)}")
        if len(failures) > 20:
            print(f"  ... {len(failures) - 20} more")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
