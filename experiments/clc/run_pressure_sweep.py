#!/usr/bin/env python3
"""Run CLC inter-kernel pressure and launch-order sweeps."""

import argparse
import csv
import os
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "build" / "clc_pressure_probe"
OUTDIR = Path(__file__).resolve().parent / "results"

INT_FIELDS = {
    "tasks",
    "threads",
    "work_cycles",
    "pressure_blocks_per_sm",
    "pressure_blocks_override",
    "pressure_blocks",
    "pressure_threads",
    "pressure_cycles",
    "pressure_dynamic_smem",
    "priority_mode",
    "launch_order",
    "low_priority",
    "high_priority",
    "clc_priority",
    "pressure_priority",
    "pressure_started",
    "pressure_active_sms",
    "pressure_per_sm_min",
    "pressure_per_sm_max",
    "clc_started_after_pressure_end",
    "clc_started_after_pressure_end_global",
    "processed",
    "active_workers",
    "clc_active_sms",
    "clc_per_sm_min",
    "clc_per_sm_max",
    "missed",
    "duplicates",
    "attempts",
    "successes",
    "failures",
    "unique_claimed",
    "claimed_min",
    "claimed_max",
    "first_claim_min",
    "first_claim_max",
    "last_claim_min",
    "last_claim_max",
    "occ_blocks_per_sm",
    "predicted_r",
    "sm_count",
    "smid_changes",
    "duplicate_claims",
    "claim_range_holes",
    "exactly_once",
    "suffix_matches_active",
    "terminal_failures_equal_active_workers",
    "structural_ok",
}

FLOAT_FIELDS = {
    "pressure_per_sm_mean",
    "pressure_ms",
    "clc_ms",
    "pstart_to_cstart_ms",
    "pstart_to_cend_ms",
    "overlap_ms",
    "pressure_global_us",
    "clc_global_us",
    "global_start_delta_us",
    "global_end_delta_us",
    "global_overlap_us",
    "clc_per_sm_mean",
}


def cfg(name, tasks=8192, threads=128, work_cycles=4096,
        pressure_blocks_per_sm=0, pressure_threads=1024,
        pressure_cycles=200_000_000, priority_mode=0,
        pressure_dynamic_smem=0, pressure_blocks_override=0,
        launch_order=0):
    return {
        "suite": name,
        "tasks": tasks,
        "threads": threads,
        "work_cycles": work_cycles,
        "pressure_blocks_per_sm": pressure_blocks_per_sm,
        "pressure_threads": pressure_threads,
        "pressure_cycles": pressure_cycles,
        "priority_mode": priority_mode,
        "pressure_dynamic_smem": pressure_dynamic_smem,
        "pressure_blocks_override": pressure_blocks_override,
        "launch_order": launch_order,
    }


def configs_for_suite(suite):
    if suite in ("basic", "all"):
        yield cfg("pressure-baseline", pressure_cycles=0)
        yield cfg("pressure-one-light", pressure_threads=128,
                  pressure_blocks_override=1)
        yield cfg("pressure-one-heavy", pressure_threads=1024,
                  pressure_blocks_override=1)
        yield cfg("pressure-half-heavy", pressure_threads=1024,
                  pressure_blocks_override=94)
        yield cfg("pressure-full-light", pressure_blocks_per_sm=1,
                  pressure_threads=128)
        yield cfg("pressure-full-heavy", pressure_blocks_per_sm=1,
                  pressure_threads=1024)

    if suite in ("priority", "all"):
        for priority_mode in (-1, 0, 1):
            yield cfg("pressure-priority", priority_mode=priority_mode,
                      pressure_threads=1024, pressure_blocks_override=1)

    if suite in ("order", "all"):
        yield cfg("pressure-clc-first-short", work_cycles=2_000_000,
                  pressure_threads=1024, pressure_blocks_override=1,
                  launch_order=1)
        yield cfg("pressure-clc-first-long", work_cycles=20_000_000,
                  pressure_threads=1024, pressure_blocks_override=1,
                  launch_order=1)

    if suite in ("smem", "all"):
        yield cfg("pressure-dynamic-smem-one", pressure_threads=128,
                  pressure_dynamic_smem=32768,
                  pressure_blocks_override=1)
        yield cfg("pressure-dynamic-smem-full", pressure_threads=128,
                  pressure_dynamic_smem=32768,
                  pressure_blocks_per_sm=1)


def parse_probe_csv(stdout):
    lines = [ln for ln in stdout.splitlines() if ln and not ln.startswith("==")]
    header_idx = next(i for i, ln in enumerate(lines) if ln.startswith("tasks,"))
    row = next(csv.DictReader(lines[header_idx:header_idx + 2]))
    out = {}
    for key, value in row.items():
        if key in INT_FIELDS:
            out[key] = int(value)
        elif key in FLOAT_FIELDS:
            out[key] = float(value)
        else:
            out[key] = value
    return out


def run_one(config):
    env = dict(os.environ)
    env["CLC_PROBE_CSV"] = "1"
    cmd = [
        str(BIN),
        str(config["tasks"]),
        str(config["threads"]),
        str(config["work_cycles"]),
        str(config["pressure_blocks_per_sm"]),
        str(config["pressure_threads"]),
        str(config["pressure_cycles"]),
        str(config["priority_mode"]),
        str(config["pressure_dynamic_smem"]),
        str(config["pressure_blocks_override"]),
        str(config["launch_order"]),
    ]
    cp = subprocess.run(cmd, env=env, text=True, capture_output=True,
                        check=True)
    row = parse_probe_csv(cp.stdout)
    row["suite"] = config["suite"]
    return row


def validate(row):
    errors = []
    tasks = row["tasks"]
    active_workers = row["active_workers"]
    expected_claimed = max(0, tasks - active_workers)

    for field in ("missed", "duplicates", "duplicate_claims",
                  "claim_range_holes", "smid_changes"):
        if row[field] != 0:
            errors.append(f"{field}={row[field]}")

    if row["exactly_once"] != 1:
        errors.append("exactly_once=0")
    if row["suffix_matches_active"] != 1:
        errors.append("suffix_matches_active=0")
    if row["terminal_failures_equal_active_workers"] != 1:
        errors.append("terminal failures mismatch")
    if row["structural_ok"] != 1:
        errors.append("structural_ok=0")
    if row["processed"] != tasks or row["attempts"] != tasks:
        errors.append("processed/attempts mismatch")
    if row["successes"] != expected_claimed:
        errors.append("successes mismatch")
    if row["failures"] != active_workers:
        errors.append("failures mismatch")
    if row["pressure_blocks"] > 0:
        if row["pressure_started"] != row["pressure_blocks"]:
            errors.append("pressure_started mismatch")
        if row["global_overlap_us"] > 1.0:
            errors.append(f"unexpected actual overlap={row['global_overlap_us']}")
    if row["active_workers"] != min(tasks, row["predicted_r"]):
        errors.append("active_workers != standalone predicted R")

    return errors


def group_key(row):
    return (
        row["suite"],
        row["tasks"],
        row["threads"],
        row["work_cycles"],
        row["pressure_blocks_per_sm"],
        row["pressure_blocks_override"],
        row["pressure_threads"],
        row["pressure_cycles"],
        row["pressure_dynamic_smem"],
        row["priority_mode"],
        row["launch_order"],
    )


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
            "pressure_blocks_per_sm": key[4],
            "pressure_blocks_override": key[5],
            "pressure_threads": key[6],
            "pressure_cycles": key[7],
            "pressure_dynamic_smem": key[8],
            "priority_mode": key[9],
            "launch_order": key[10],
            "repeats": len(rs),
            "predicted_r_min": min(r["predicted_r"] for r in rs),
            "predicted_r_max": max(r["predicted_r"] for r in rs),
            "active_workers_min": min(r["active_workers"] for r in rs),
            "active_workers_max": max(r["active_workers"] for r in rs),
            "pressure_active_sms_min": min(r["pressure_active_sms"] for r in rs),
            "pressure_active_sms_max": max(r["pressure_active_sms"] for r in rs),
            "pressure_global_us_mean": mean(rs, "pressure_global_us"),
            "clc_global_us_mean": mean(rs, "clc_global_us"),
            "global_start_delta_us_mean": mean(rs, "global_start_delta_us"),
            "global_overlap_us_max": max(r["global_overlap_us"] for r in rs),
            "clc_after_pressure_global_min": min(
                r["clc_started_after_pressure_end_global"] for r in rs),
            "clc_after_pressure_global_max": max(
                r["clc_started_after_pressure_end_global"] for r in rs),
            "missed_max": max(r["missed"] for r in rs),
            "duplicates_max": max(r["duplicates"] for r in rs),
            "duplicate_claims_max": max(r["duplicate_claims"] for r in rs),
            "claim_range_holes_max": max(r["claim_range_holes"] for r in rs),
            "structural_ok_min": min(r["structural_ok"] for r in rs),
            "validation": "ok" if all(r["validation"] == "ok" for r in rs)
            else "failed",
        })
    return out


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=("basic", "priority", "order", "smem",
                                        "all"), default="all")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    if not BIN.exists():
        raise SystemExit(f"missing {BIN}; build clc_pressure_probe first")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rows = []
    failures = []
    for config in configs_for_suite(args.suite):
        for repeat in range(args.repeats):
            row = run_one(config)
            row["repeat"] = repeat
            errors = validate(row)
            row["validation"] = "ok" if not errors else ";".join(errors)
            rows.append(row)
            status = row["validation"]
            print(f"[pressure] {config['suite']} repeat={repeat} {status}")
            if errors:
                failures.append(row)

    raw_path = OUTDIR / f"clc_pressure_{args.suite}_raw_{stamp}.csv"
    summary_path = OUTDIR / f"clc_pressure_{args.suite}_summary_{stamp}.csv"
    write_csv(raw_path, rows)
    write_csv(summary_path, aggregate(rows))
    print(f"raw={raw_path}")
    print(f"summary={summary_path}")

    if failures:
        raise SystemExit(f"{len(failures)} invalid pressure rows")


if __name__ == "__main__":
    main()
