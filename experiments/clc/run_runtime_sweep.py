#!/usr/bin/env python3
"""Run CLC runtime-behavior sweeps.

This runner focuses on observable GPU runtime facts instead of speed:

* attempts/process/failure counts
* terminal failed-cancel behavior
* success/failure claim latency
* active CTA distribution over SM ids
* SM id stability while a CTA processes claimed work
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
BIN = ROOT / "build" / "clc_runtime_probe"
OUTDIR = Path(__file__).resolve().parent / "results"

NUMERIC = {
    "tasks": int,
    "threads": int,
    "work_cycles": int,
    "smem_bytes": int,
    "processed": int,
    "active_workers": int,
    "missed": int,
    "duplicates": int,
    "attempts": int,
    "successes": int,
    "failures": int,
    "unique_claimed": int,
    "claimed_min": int,
    "claimed_max": int,
    "first_claim_min": int,
    "first_claim_max": int,
    "last_claim_min": int,
    "last_claim_max": int,
    "occ_blocks_per_sm": int,
    "predicted_r": int,
    "sm_count": int,
    "expected_active_workers": int,
    "expected_claimed": int,
    "smid_min": int,
    "smid_max": int,
    "active_sms": int,
    "active_workers_per_sm_min": int,
    "active_workers_per_sm_max": int,
    "active_workers_per_sm_mean": float,
    "smid_changes": int,
    "success_cycle_min": int,
    "success_cycle_p50_worker_avg": int,
    "success_cycle_p90_worker_avg": int,
    "success_cycle_max": int,
    "success_cycle_mean": float,
    "failure_cycle_min": int,
    "failure_cycle_p50_worker_avg": int,
    "failure_cycle_p90_worker_avg": int,
    "failure_cycle_max": int,
    "failure_cycle_mean": float,
    "terminal_failures_equal_active_workers": int,
    "duplicate_claims": int,
    "claim_range_holes": int,
    "structural_ok": int,
}


def cfg(suite, tasks, threads=128, work_cycles=4096, smem_bytes=0):
    return {
        "suite": suite,
        "tasks": tasks,
        "threads": threads,
        "work_cycles": work_cycles,
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
    cp = subprocess.run(
        [str(BIN), str(config["tasks"]), str(config["threads"]),
         str(config["work_cycles"]), str(config["smem_bytes"])],
        env=env, text=True, capture_output=True, check=True)
    row = parse_probe_csv(cp.stdout)
    row["suite"] = config["suite"]
    return row


def configs_for_suite(suite):
    if suite in ("threshold", "all"):
        for tasks in (1024, 2256, 2304, 8192):
            yield cfg("runtime-threshold", tasks, 128, 4096, 0)

    if suite in ("threads", "all"):
        for threads in (64, 128, 256):
            yield cfg("runtime-threads", 8192, threads, 4096, 0)

    if suite in ("smem", "all"):
        for smem_bytes in (0, 8192, 16384, 24576):
            yield cfg("runtime-smem", 8192, 128, 4096, smem_bytes)

    if suite in ("latency", "all"):
        for work_cycles in (0, 128, 1024, 4096):
            yield cfg("runtime-latency", 8192, 128, work_cycles, 0)


def validate(row):
    errors = []
    tasks = row["tasks"]
    predicted_r = row["predicted_r"]
    expected_active = min(tasks, predicted_r)
    expected_claimed = max(0, tasks - predicted_r)

    for field in ("missed", "duplicates", "duplicate_claims",
                  "claim_range_holes", "smid_changes"):
        if row[field] != 0:
            errors.append(f"{field}={row[field]}")

    if row["processed"] != tasks:
        errors.append(f"processed={row['processed']} expected={tasks}")
    if row["attempts"] != tasks:
        errors.append(f"attempts={row['attempts']} expected={tasks}")
    if row["active_workers"] != expected_active:
        errors.append("active_workers mismatch")
    if row["successes"] != expected_claimed:
        errors.append("successes mismatch")
    if row["failures"] != expected_active:
        errors.append("failures mismatch")
    if row["terminal_failures_equal_active_workers"] != 1:
        errors.append("terminal failures mismatch")
    if row["expected_active_workers"] != expected_active:
        errors.append("probe expected active mismatch")
    if row["expected_claimed"] != expected_claimed:
        errors.append("probe expected claimed mismatch")

    if tasks <= predicted_r:
        if row["unique_claimed"] != 0 or row["claimed_min"] != 0:
            errors.append("unexpected claim below/equal R")
    else:
        if row["unique_claimed"] != expected_claimed:
            errors.append("unique_claimed mismatch")
        if row["claimed_min"] != predicted_r:
            errors.append("claimed_min mismatch")
        if row["claimed_max"] != tasks - 1:
            errors.append("claimed_max mismatch")

    if row["structural_ok"] != 1:
        errors.append("probe structural_ok=0")

    # The local Blackwell reports dense SM ids 0..SM_count-1. Runtime launch
    # distribution should be balanced to within one CTA for nonclustered grids.
    if row["active_sms"] > 0:
        diff = row["active_workers_per_sm_max"] - row["active_workers_per_sm_min"]
        if diff > 1:
            errors.append(f"workers_per_sm imbalance={diff}")

    return errors


def group_key(row):
    return (row["suite"], row["tasks"], row["threads"], row["work_cycles"],
            row["smem_bytes"])


def mean(rows, field):
    return statistics.fmean(r[field] for r in rows)


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)

    out = []
    for key, rs in sorted(grouped.items()):
        out.append({
            "suite": key[0],
            "tasks": key[1],
            "threads": key[2],
            "work_cycles": key[3],
            "smem_bytes": key[4],
            "repeats": len(rs),
            "predicted_r_mean": mean(rs, "predicted_r"),
            "active_workers_mean": mean(rs, "active_workers"),
            "attempts_mean": mean(rs, "attempts"),
            "successes_mean": mean(rs, "successes"),
            "failures_mean": mean(rs, "failures"),
            "active_sms_mean": mean(rs, "active_sms"),
            "workers_per_sm_min_min": min(r["active_workers_per_sm_min"]
                                          for r in rs),
            "workers_per_sm_max_max": max(r["active_workers_per_sm_max"]
                                          for r in rs),
            "smid_changes_max": max(r["smid_changes"] for r in rs),
            "success_cycle_mean_mean": mean(rs, "success_cycle_mean"),
            "success_cycle_p50_worker_avg_mean": mean(
                rs, "success_cycle_p50_worker_avg"),
            "success_cycle_p90_worker_avg_mean": mean(
                rs, "success_cycle_p90_worker_avg"),
            "failure_cycle_mean_mean": mean(rs, "failure_cycle_mean"),
            "failure_cycle_p50_worker_avg_mean": mean(
                rs, "failure_cycle_p50_worker_avg"),
            "failure_cycle_p90_worker_avg_mean": mean(
                rs, "failure_cycle_p90_worker_avg"),
            "missed_max": max(r["missed"] for r in rs),
            "duplicates_max": max(r["duplicates"] for r in rs),
            "duplicate_claims_max": max(r["duplicate_claims"] for r in rs),
            "claim_range_holes_max": max(r["claim_range_holes"] for r in rs),
            "terminal_failures_equal_active_min": min(
                r["terminal_failures_equal_active_workers"] for r in rs),
            "structural_ok_min": min(r["structural_ok"] for r in rs),
            "valid_runs": sum(1 for r in rs if r["validation"] == "ok"),
        })
    return out


def fieldnames(rows):
    out = []
    for row in rows:
        for key in row.keys():
            if key not in out:
                out.append(key)
    return out


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=("threshold", "threads", "smem",
                                        "latency", "all"),
                    default="all")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    if not BIN.exists():
        raise SystemExit(f"missing {BIN}; run TARGET=clc_runtime_probe build")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rows = []
    failures = []
    configs = list(configs_for_suite(args.suite))
    for config in configs:
        for repeat in range(args.repeats):
            row = run_one(config)
            row["repeat"] = repeat
            errors = validate(row)
            row["validation"] = "ok" if not errors else ";".join(errors)
            rows.append(row)
            status = "ok" if not errors else row["validation"]
            print(f"[runtime] config={config} repeat={repeat} {status}")
            if errors:
                failures.append(row)

    raw_path = OUTDIR / f"clc_runtime_{args.suite}_raw_{stamp}.csv"
    summary_path = OUTDIR / f"clc_runtime_{args.suite}_summary_{stamp}.csv"
    write_csv(raw_path, rows)
    write_csv(summary_path, aggregate(rows))
    print(f"raw={raw_path}")
    print(f"summary={summary_path}")

    if failures:
        raise SystemExit(f"{len(failures)} invalid rows")


if __name__ == "__main__":
    main()
